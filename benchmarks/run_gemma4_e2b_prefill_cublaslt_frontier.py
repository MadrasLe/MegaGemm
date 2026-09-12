#!/usr/bin/env python3
"""One-load full-model cuBLASLt frontier for Gemma 4 E2B/L4 B8 prefill.

The frontier first screens explicit cuBLASLt heuristic indices on the four
exact gate-up GEMMs used by B8/P512 and B8/P2048.  It then rotates the best
algorithm combinations against production inside the same loaded model.  Only
the full-model phase can recommend promotion; the per-GEMM screen is a search
stage, not performance evidence by itself.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import benchmark_inference_matrix as matrix
from benchmarks.run_gemma4_e2b_phase_split import (
    DEFAULT_MODEL,
    _configure_megagemm_profile,
    _runner_args,
)


PROMPT_TO_SEQUENCE = {512: 521, 2048: 2057}
OUTPUT_FEATURES = (12288, 24576)
EXPECTED_DENSE_LAYERS = 35
MAX_ABS_ERROR = 0.25


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _measure(
    call: Callable[[], Any],
    *,
    warmups: int,
    repeats: int,
) -> list[float]:
    import torch

    for _ in range(warmups):
        call()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _mlp_modules(model) -> list[Any]:
    modules = [
        module
        for module in model.modules()
        if hasattr(module, "_gemma4_e2b_prefill_cublaslt_gateup_enabled")
    ]
    counts = {
        output_features: sum(
            int(module.gate_up_proj.weight.shape[0]) == output_features
            for module in modules
        )
        for output_features in OUTPUT_FEATURES
    }
    if len(modules) != EXPECTED_DENSE_LAYERS or counts != {12288: 15, 24576: 20}:
        raise RuntimeError(
            "unexpected E2B MLP topology: "
            f"total={len(modules)} output_feature_counts={counts}"
        )
    return modules


def _representative_weights(modules: list[Any]) -> dict[int, Any]:
    weights: dict[int, Any] = {}
    for module in modules:
        weight = module.gate_up_proj.weight
        weights.setdefault(int(weight.shape[0]), weight)
    return weights


def _valid_algorithm(row: dict[str, Any], maximum_spread: float) -> bool:
    samples = [float(value) for value in row.get("samples_ms") or ()]
    max_abs_error = row.get("max_abs_error")
    return bool(
        row.get("error") is None
        and row.get("finite") is True
        and max_abs_error is not None
        and float(max_abs_error) <= MAX_ABS_ERROR
        and samples
        and min(samples) > 0.0
        and max(samples) / min(samples) <= maximum_spread
    )


def _screen_shape(
    *,
    sequence_len: int,
    output_features: int,
    weight,
    maximum_algorithms: int,
    warmups: int,
    repeats: int,
    maximum_spread: float,
) -> dict[str, Any]:
    import torch
    from megagemm.kernels.mlp_prefill_native import (
        cublaslt_bf16_algorithm_count_cuda,
        cublaslt_bf16_linear_cuda,
    )

    rows = 8 * sequence_len
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260912 + sequence_len + output_features)
    x = torch.empty(
        (rows, 1536),
        device="cuda",
        dtype=torch.bfloat16,
    ).normal_(mean=0.0, std=0.2, generator=generator)
    weight_t = weight.t()
    reference = torch.empty(
        (rows, output_features),
        device="cuda",
        dtype=torch.bfloat16,
    )
    candidate = torch.empty_like(reference)

    def baseline_call():
        return torch.mm(x, weight_t, out=reference)

    baseline_first = _measure(
        baseline_call,
        warmups=warmups,
        repeats=repeats,
    )
    algorithm_count = cublaslt_bf16_algorithm_count_cuda(
        x,
        weight,
        maximum_algorithms,
    )
    algorithms: list[dict[str, Any]] = []
    for algorithm_index in range(algorithm_count):
        def candidate_call(index: int = algorithm_index):
            return cublaslt_bf16_linear_cuda(
                x,
                weight,
                out=candidate,
                algorithm_index=index,
            )

        try:
            samples = _measure(
                candidate_call,
                warmups=warmups,
                repeats=repeats,
            )
            candidate_call()
            torch.cuda.synchronize()
            delta = (candidate - reference).abs()
            row = {
                "algorithm_index": algorithm_index,
                "median_ms": _median(samples),
                "samples_ms": samples,
                "finite": bool(torch.isfinite(candidate).all().item()),
                "max_abs_error": float(delta.max().item()),
                "mean_abs_error": float(delta.mean(dtype=torch.float32).item()),
                "error": None,
            }
            del delta
        except Exception as exc:
            row = {
                "algorithm_index": algorithm_index,
                "median_ms": None,
                "samples_ms": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        row["valid"] = _valid_algorithm(row, maximum_spread)
        algorithms.append(row)

    baseline_recheck = _measure(
        baseline_call,
        warmups=warmups,
        repeats=repeats,
    )
    baseline_median = min(_median(baseline_first), _median(baseline_recheck))
    valid_algorithms = [row for row in algorithms if row["valid"]]
    valid_algorithms.sort(key=lambda row: float(row["median_ms"]))
    top = valid_algorithms[:2]
    del candidate, reference, x
    torch.cuda.empty_cache()
    return {
        "shape": {
            "batch_size": 8,
            "sequence_len": sequence_len,
            "m": rows,
            "k": 1536,
            "n": output_features,
            "dtype": "bf16",
        },
        "algorithm_count": algorithm_count,
        "baseline_first_ms": baseline_first,
        "baseline_recheck_ms": baseline_recheck,
        "baseline_median_ms": baseline_median,
        "algorithms": algorithms,
        "top_algorithms": [
            {
                "algorithm_index": int(row["algorithm_index"]),
                "median_ms": float(row["median_ms"]),
                "micro_speedup": baseline_median / float(row["median_ms"]),
            }
            for row in top
        ],
        "valid": bool(top),
    }


def _candidate_maps(screen: dict[str, Any], sequence_len: int) -> list[dict[Any, int]]:
    choices: dict[int, list[int]] = {}
    for output_features in OUTPUT_FEATURES:
        row = screen[f"s{sequence_len}_n{output_features}"]
        choices[output_features] = [
            int(item["algorithm_index"])
            for item in row["top_algorithms"]
        ]
        if not choices[output_features]:
            raise RuntimeError(
                f"no valid cuBLASLt algorithms for S{sequence_len}/N{output_features}"
            )
        if len(choices[output_features]) == 1:
            choices[output_features].append(choices[output_features][0])

    maps = [
        {
            (sequence_len, 12288): small,
            (sequence_len, 24576): large,
        }
        for small, large in (
            (choices[12288][0], choices[24576][0]),
            (choices[12288][1], choices[24576][0]),
            (choices[12288][0], choices[24576][1]),
            (choices[12288][1], choices[24576][1]),
        )
    ]
    unique: list[dict[Any, int]] = []
    seen: set[tuple[tuple[Any, int], ...]] = set()
    for algorithm_map in maps:
        signature = tuple(sorted(algorithm_map.items()))
        if signature not in seen:
            unique.append(algorithm_map)
            seen.add(signature)
    return unique


def _apply_case(modules: list[Any], algorithms: dict[Any, int] | None) -> None:
    for module in modules:
        module._gemma4_e2b_prefill_cublaslt_gateup_enabled = algorithms is not None
        module._gemma4_e2b_prefill_cublaslt_gateup_algorithms = dict(algorithms or {})
        module._gemma4_e2b_prefill_cublaslt_gateup_hits = 0
        module._gemma4_e2b_prefill_cublaslt_gateup_runtime_disabled = False
        module._gemma4_e2b_prefill_cublaslt_gateup_failure = ""


def _run_once(
    runner,
    modules: list[Any],
    prompts: list[str],
    *,
    prompt_tokens: int,
    case: str,
    algorithms: dict[Any, int] | None,
    repeat: int,
) -> dict[str, Any]:
    _apply_case(modules, algorithms)
    matrix.sync_cuda()
    result = runner(prompts, 1)
    matrix.sync_cuda()
    if int(result["generated_tokens"]) != len(prompts):
        raise RuntimeError(
            f"{case}/p{prompt_tokens} generated {result['generated_tokens']} tokens; "
            f"expected {len(prompts)}"
        )
    extra = dict(result.get("extra") or {})
    scheduler = dict(extra.get("scheduler_stats") or {})
    runtime = dict(extra.get("decode_runtime_stats") or {})
    hits = sum(
        int(module._gemma4_e2b_prefill_cublaslt_gateup_hits)
        for module in modules
    )
    disabled = sum(
        bool(module._gemma4_e2b_prefill_cublaslt_gateup_runtime_disabled)
        for module in modules
    )
    failures = sorted(
        {
            str(module._gemma4_e2b_prefill_cublaslt_gateup_failure)
            for module in modules
            if module._gemma4_e2b_prefill_cublaslt_gateup_failure
        }
    )
    return {
        "case": case,
        "prompt_tokens": prompt_tokens,
        "repeat": repeat,
        "algorithms": (
            {
                f"s{key[0]}_n{key[1]}": int(value)
                for key, value in sorted(algorithms.items())
            }
            if algorithms is not None
            else None
        ),
        "wall_ms": float(result["elapsed_s"]) * 1000.0,
        "prefill_ms": float(scheduler.get("prefill_time_ms") or 0.0),
        "generated_token_digest": extra.get("generated_token_digest"),
        "candidate_hits": hits,
        "candidate_disabled_layers": disabled,
        "candidate_failures": failures,
        "production_routes": {
            "sliding_layers": int(
                runtime.get("gemma4_e2b_l4_sliding_prefill_enabled_layers") or 0
            ),
            "full_layers": int(
                runtime.get("gemma4_e2b_l4_full_prefill_expand_enabled_layers") or 0
            ),
            "activation_layers": int(
                runtime.get(
                    "gemma4_e2b_b8_prefill_gated_activation_enabled_layers"
                )
                or 0
            ),
            "activation_failure": str(
                runtime.get("gemma4_e2b_b8_prefill_gated_activation_failure") or ""
            ),
        },
    }


def _summarize(
    samples: list[dict[str, Any]],
    *,
    cases_by_prompt: dict[int, list[tuple[str, dict[Any, int] | None]]],
    maximum_spread: float,
    minimum_prefill_speedup: float,
    minimum_wall_speedup: float,
) -> dict[str, Any]:
    shape_policy: dict[str, Any] = {}
    rows: dict[str, Any] = {}
    for prompt_tokens, cases in cases_by_prompt.items():
        prompt_rows = [
            row for row in samples if int(row["prompt_tokens"]) == prompt_tokens
        ]
        production = [row for row in prompt_rows if row["case"] == "production"]
        production_prefill = _median([float(row["prefill_ms"]) for row in production])
        production_wall = _median([float(row["wall_ms"]) for row in production])
        production_digests = {
            str(row["generated_token_digest"])
            for row in production
            if row.get("generated_token_digest")
        }
        candidates: list[dict[str, Any]] = []
        for case, algorithms in cases:
            current = [row for row in prompt_rows if row["case"] == case]
            prefill = [float(row["prefill_ms"]) for row in current]
            wall = [float(row["wall_ms"]) for row in current]
            digests = {
                str(row["generated_token_digest"])
                for row in current
                if row.get("generated_token_digest")
            }
            expected_hits = 0 if algorithms is None else EXPECTED_DENSE_LAYERS
            valid = bool(
                len(current) >= 3
                and prefill
                and min(prefill) > 0.0
                and max(prefill) / min(prefill) <= maximum_spread
                and len(digests) == 1
                and digests == production_digests
                and all(int(row["candidate_hits"]) == expected_hits for row in current)
                and all(not row["candidate_disabled_layers"] for row in current)
                and all(not row["candidate_failures"] for row in current)
                and all(
                    row["production_routes"] == {
                        "sliding_layers": 28,
                        "full_layers": 7,
                        "activation_layers": 35,
                        "activation_failure": "",
                    }
                    for row in current
                )
            )
            median_prefill = _median(prefill)
            median_wall = _median(wall)
            summary = {
                "case": case,
                "algorithms": current[0]["algorithms"] if current else None,
                "valid": valid,
                "median_prefill_ms": median_prefill,
                "median_wall_ms": median_wall,
                "prefill_speedup": (
                    production_prefill / median_prefill if median_prefill else 0.0
                ),
                "wall_speedup": production_wall / median_wall if median_wall else 0.0,
                "spread_ratio": max(prefill) / min(prefill) if prefill else math.inf,
            }
            rows[f"p{prompt_tokens}/{case}"] = summary
            if algorithms is not None and valid:
                candidates.append(summary)
        winner = (
            max(candidates, key=lambda row: float(row["prefill_speedup"]))
            if candidates
            else None
        )
        promote = bool(
            winner
            and float(winner["prefill_speedup"]) >= minimum_prefill_speedup
            and float(winner["wall_speedup"]) >= minimum_wall_speedup
        )
        shape_policy[f"b8/p{prompt_tokens}"] = {
            "decision": "PROMOTE_CUBLASLT" if promote else "KEEP_TORCH_MM",
            "winner": winner["case"] if promote else "production",
            "algorithms": winner["algorithms"] if promote else None,
            "prefill_speedup": float(winner["prefill_speedup"]) if winner else 0.0,
            "wall_speedup": float(winner["wall_speedup"]) if winner else 0.0,
        }
    promoted = [
        key for key, value in shape_policy.items()
        if value["decision"] == "PROMOTE_CUBLASLT"
    ]
    return {
        "decision": "PROMOTE_SHAPE_DISPATCH" if promoted else "KEEP_TORCH_MM",
        "apply_change": bool(promoted),
        "promoted_shapes": promoted,
        "shape_policy": shape_policy,
        "rows": rows,
        "promotion_rule": (
            "natural-token equality, 35/35 candidate hits, no runtime fallback, "
            "stable full-model samples, and both prefill and wall-time gates"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    gpu = torch.cuda.get_device_name(0)
    if "L4" not in gpu.upper():
        raise RuntimeError(f"this gate requires NVIDIA L4, got {gpu}")

    prompts_requested = matrix.parse_csv_ints(args.prompt_tokens)
    if prompts_requested != [512, 2048] or args.batch_size != 8:
        raise ValueError("this gate is exact to B8 and prompts 512,2048")
    args.backend = "megagemm"
    _configure_megagemm_profile(args.model)
    os.environ["MEGAGEMM_BENCHMARK_TOKEN_DIGEST"] = "1"
    os.environ.pop("MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID", None)
    # Import MegaGemm only after the controlled profile is installed because
    # several kernel switches are deliberately resolved at module import.
    from megagemm.kernels.mlp_prefill_native import HAS_CUBLASLT_BF16_LINEAR

    if not HAS_CUBLASLT_BF16_LINEAR:
        raise RuntimeError("the focused MegaGemm cuBLASLt extension is unavailable")

    tokenizer = matrix.load_tokenizer(
        args.tokenizer or args.model,
        local_files_only=args.local_files_only,
    )
    prompts = {
        prompt: matrix.build_prompts(tokenizer, args.batch_size, prompt)[0]
        for prompt in prompts_requested
    }
    runner = matrix.make_runner(_runner_args(args), tokenizer)
    engine = getattr(runner, "_megagemm_engine", None)
    if engine is None:
        raise RuntimeError("MegaGemm runner did not expose its loaded engine")
    modules = _mlp_modules(engine.model)
    weights = _representative_weights(modules)

    print("Gemma 4 E2B/L4 B8 prefill cuBLASLt frontier", flush=True)
    print("  four exact GEMM screens + full-model final gate", flush=True)
    print("  workloads: P512/P2048, O1, natural greedy tokens", flush=True)
    print("  model loads: 1; profiler: disabled", flush=True)

    screen: dict[str, Any] = {}
    for prompt_tokens in prompts_requested:
        sequence_len = PROMPT_TO_SEQUENCE[prompt_tokens]
        for output_features in OUTPUT_FEATURES:
            key = f"s{sequence_len}_n{output_features}"
            print(f"Screen {key}", flush=True)
            with torch.inference_mode():
                screen[key] = _screen_shape(
                    sequence_len=sequence_len,
                    output_features=output_features,
                    weight=weights[output_features],
                    maximum_algorithms=args.maximum_algorithms,
                    warmups=args.screen_warmups,
                    repeats=args.screen_repeats,
                    maximum_spread=args.screen_maximum_spread,
                )
            print(
                "  top=" + json.dumps(screen[key]["top_algorithms"]),
                flush=True,
            )

    cases_by_prompt: dict[int, list[tuple[str, dict[Any, int] | None]]] = {}
    for prompt_tokens in prompts_requested:
        sequence_len = PROMPT_TO_SEQUENCE[prompt_tokens]
        candidates = _candidate_maps(screen, sequence_len)
        cases_by_prompt[prompt_tokens] = [
            ("production", None),
            *[
                (f"cublaslt_rank{index}", algorithms)
                for index, algorithms in enumerate(candidates)
            ],
        ]

    for prompt_tokens, cases in cases_by_prompt.items():
        for case, algorithms in cases:
            print(f"Warmup {case}/p{prompt_tokens}", flush=True)
            _run_once(
                runner,
                modules,
                prompts[prompt_tokens],
                prompt_tokens=prompt_tokens,
                case=case,
                algorithms=algorithms,
                repeat=0,
            )

    schedule = [
        (prompt_tokens, case, algorithms)
        for prompt_tokens, cases in cases_by_prompt.items()
        for case, algorithms in cases
    ]
    samples: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        current = list(schedule)
        if repeat % 2 == 0:
            current.reverse()
        rotation = (repeat - 1) % len(current)
        current = current[rotation:] + current[:rotation]
        for prompt_tokens, case, algorithms in current:
            sample = _run_once(
                runner,
                modules,
                prompts[prompt_tokens],
                prompt_tokens=prompt_tokens,
                case=case,
                algorithms=algorithms,
                repeat=repeat,
            )
            samples.append(sample)
            print(
                f"Run {repeat}/{args.repeats} {case}/p{prompt_tokens}: "
                f"prefill={sample['prefill_ms']:.2f}ms "
                f"wall={sample['wall_ms']:.2f}ms hits={sample['candidate_hits']}",
                flush=True,
            )

    summary = _summarize(
        samples,
        cases_by_prompt=cases_by_prompt,
        maximum_spread=args.maximum_spread,
        minimum_prefill_speedup=args.minimum_prefill_speedup,
        minimum_wall_speedup=args.minimum_wall_speedup,
    )
    payload = {
        "benchmark": "gemma4_e2b_l4_b8_prefill_cublaslt_frontier",
        "schema_version": 1,
        "model": args.model,
        "hardware_label": "1xl4",
        "dtype": "bf16",
        "method": {
            "model_loads": 1,
            "screen_role": "candidate search only",
            "promotion_evidence": "rotated full-model production comparison",
            "correctness": "natural-token digest and exact 35-layer hit audit",
        },
        "screen": screen,
        "samples": samples,
        "summary": summary,
        "system": {
            "gpu": matrix.gpu_snapshot(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "git": matrix.git_snapshot(),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("DECISION " + json.dumps(summary, sort_keys=True), flush=True)
    print(f"Wrote: {args.output}", flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-tokens", default="512,2048")
    parser.add_argument("--maximum-algorithms", type=int, default=16)
    parser.add_argument("--screen-warmups", type=int, default=2)
    parser.add_argument("--screen-repeats", type=int, default=5)
    parser.add_argument("--screen-maximum-spread", type=float, default=1.15)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--maximum-spread", type=float, default=1.06)
    parser.add_argument("--minimum-prefill-speedup", type=float, default=1.01)
    parser.add_argument("--minimum-wall-speedup", type=float, default=1.005)
    parser.add_argument("--max-seq-len", type=int, default=2304)
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.repeats < 3 or args.screen_repeats < 3:
        raise SystemExit("full-model and screen repeats must both be at least 3")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
