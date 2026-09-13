#!/usr/bin/env python3
"""One-load full-model frontier for the Gemma 4 E2B prefill MLP body.

Candidates fuse the gate/up GEMM and GELU-tanh multiply so the wide ``[M,2I]``
activation is never written to global memory.  A small numeric preflight uses
the real model weights, then every surviving launch geometry is measured in
the uninstrumented B8/P512 and B8/P2048 first-token paths.
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
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import benchmark_inference_matrix as matrix
from benchmarks.run_gemma4_e2b_phase_split import (
    DEFAULT_MODEL,
    _configure_megagemm_profile,
    _runner_args,
)


CONFIGS: tuple[tuple[str, dict[str, int]], ...] = (
    ("bm32_bn32_bk32_w4_s3", {"block_m": 32, "block_n": 32, "block_k": 32, "num_warps": 4, "num_stages": 3}),
    ("bm64_bn32_bk32_w4_s3", {"block_m": 64, "block_n": 32, "block_k": 32, "num_warps": 4, "num_stages": 3}),
    ("bm64_bn64_bk32_w4_s3", {"block_m": 64, "block_n": 64, "block_k": 32, "num_warps": 4, "num_stages": 3}),
    ("bm128_bn32_bk32_w8_s3", {"block_m": 128, "block_n": 32, "block_k": 32, "num_warps": 8, "num_stages": 3}),
    ("bm64_bn32_bk64_w4_s2", {"block_m": 64, "block_n": 32, "block_k": 64, "num_warps": 4, "num_stages": 2}),
    ("bm128_bn32_bk64_w8_s2", {"block_m": 128, "block_n": 32, "block_k": 64, "num_warps": 8, "num_stages": 2}),
)
EXPECTED_DENSE_LAYERS = 35


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        return 0.0
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def _mlp_modules(model) -> list[Any]:
    modules = [
        module
        for module in model.modules()
        if hasattr(module, "_gemma4_e2b_prefill_fused_gateup_enabled")
    ]
    large = sum(int(module.intermediate_size) == 12288 for module in modules)
    small = sum(int(module.intermediate_size) == 6144 for module in modules)
    if len(modules) != EXPECTED_DENSE_LAYERS or (large, small) != (20, 15):
        raise RuntimeError(
            "unexpected E2B MLP topology: "
            f"total={len(modules)} large={large} small={small}; expected 35/20/15"
        )
    return modules


def _numeric_preflight(modules: list[Any]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from megagemm.kernels.gemma4_e2b_prefill_mlp import (
        gemma4_e2b_prefill_fused_gateup_geglu,
    )

    representatives = {
        int(module.intermediate_size): module.gate_up_proj.weight
        for module in modules
    }
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0xE2B4)
    rows: list[dict[str, Any]] = []
    for intermediate in (6144, 12288):
        weight = representatives[intermediate]
        x = torch.randn(
            (16, int(weight.shape[1])),
            device=weight.device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        gate_up = torch.mm(x, weight.transpose(0, 1))
        gate, up = gate_up.split(intermediate, dim=-1)
        reference = F.gelu(gate, approximate="tanh").mul_(up)
        for name, config in CONFIGS:
            error = None
            try:
                candidate = gemma4_e2b_prefill_fused_gateup_geglu(
                    x,
                    weight,
                    intermediate_size=intermediate,
                    **config,
                )
                repeat = gemma4_e2b_prefill_fused_gateup_geglu(
                    x,
                    weight,
                    intermediate_size=intermediate,
                    **config,
                )
                torch.cuda.synchronize()
                delta = (candidate.float() - reference.float()).abs()
                ref_norm = float(torch.linalg.vector_norm(reference.float()).item())
                rel_l2 = float(torch.linalg.vector_norm(delta).item()) / max(ref_norm, 1e-12)
                cosine = float(
                    torch.nn.functional.cosine_similarity(
                        candidate.float().flatten(),
                        reference.float().flatten(),
                        dim=0,
                    ).item()
                )
                repeat_max = float((candidate - repeat).abs().max().item())
                maximum = float(delta.max().item())
                finite = bool(torch.isfinite(candidate).all().item())
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                rel_l2 = float("inf")
                cosine = 0.0
                repeat_max = float("inf")
                maximum = float("inf")
                finite = False
            valid = bool(
                error is None
                and finite
                and repeat_max == 0.0
                and rel_l2 <= 0.004
                and cosine >= 0.99999
                and maximum <= 0.125
            )
            row = {
                "case": name,
                "intermediate_size": intermediate,
                "valid": valid,
                "finite": finite,
                "max_abs_error": maximum,
                "relative_l2_error": rel_l2,
                "cosine": cosine,
                "repeat_max_abs_error": repeat_max,
                "error": error,
            }
            rows.append(row)
            print("PREFLIGHT " + json.dumps(row, sort_keys=True), flush=True)
        del x, gate_up, gate, up, reference
    valid_cases = [
        name
        for name, _ in CONFIGS
        if all(row["valid"] for row in rows if row["case"] == name)
    ]
    return {"rows": rows, "valid_cases": valid_cases}


def _apply_case(modules: list[Any], config: dict[str, int] | None) -> None:
    for module in modules:
        module._gemma4_e2b_prefill_fused_gateup_enabled = config is not None
        module._gemma4_e2b_prefill_fused_gateup_config = dict(config or {})
        module._gemma4_e2b_prefill_fused_gateup_hits = 0
        module._gemma4_e2b_prefill_fused_gateup_runtime_disabled = False
        module._gemma4_e2b_prefill_fused_gateup_failure = ""


def _run_once(
    runner,
    modules: list[Any],
    prompts: list[str],
    *,
    case: str,
    config: dict[str, int] | None,
    prompt_tokens: int,
    repeat: int,
    warmup: bool,
) -> dict[str, Any]:
    _apply_case(modules, config)
    matrix.sync_cuda()
    result = runner(prompts, 1)
    matrix.sync_cuda()
    expected = len(prompts)
    if int(result["generated_tokens"]) != expected:
        raise RuntimeError(f"{case}/p{prompt_tokens} generated the wrong token count")
    extra = dict(result.get("extra") or {})
    scheduler = dict(extra.get("scheduler_stats") or {})
    hits = sum(int(module._gemma4_e2b_prefill_fused_gateup_hits) for module in modules)
    disabled = sum(
        bool(module._gemma4_e2b_prefill_fused_gateup_runtime_disabled)
        for module in modules
    )
    failures = sorted(
        {
            str(module._gemma4_e2b_prefill_fused_gateup_failure)
            for module in modules
            if module._gemma4_e2b_prefill_fused_gateup_failure
        }
    )
    return {
        "case": case,
        "config": config,
        "prompt_tokens": prompt_tokens,
        "repeat": repeat,
        "warmup": warmup,
        "prefill_ms": float(scheduler.get("prefill_time_ms") or 0.0),
        "wall_ms": float(result["elapsed_s"]) * 1000.0,
        "generated_token_digest": extra.get("generated_token_digest"),
        "candidate_hits": hits,
        "candidate_disabled_layers": disabled,
        "candidate_failures": failures,
    }


def summarize(
    samples: list[dict[str, Any]],
    *,
    active_cases: list[tuple[str, dict[str, int] | None]],
    prompts: list[int],
    maximum_spread: float,
    minimum_prefill_speedup: float,
    minimum_wall_speedup: float,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    shape_policy: dict[str, dict[str, Any]] = {}
    gate_valid = True
    for prompt in prompts:
        prompt_rows = [row for row in samples if row["prompt_tokens"] == prompt]
        baseline_rows = [row for row in prompt_rows if row["case"] == "production"]
        baseline_prefill = _median([row["prefill_ms"] for row in baseline_rows])
        baseline_wall = _median([row["wall_ms"] for row in baseline_rows])
        baseline_digests = {row["generated_token_digest"] for row in baseline_rows}
        candidates = []
        for case, config in active_cases:
            current = [row for row in prompt_rows if row["case"] == case]
            prefill = [float(row["prefill_ms"]) for row in current]
            wall = [float(row["wall_ms"]) for row in current]
            digests = {row["generated_token_digest"] for row in current}
            expected_hits = 0 if config is None else EXPECTED_DENSE_LAYERS
            valid = bool(
                len(current) >= 3
                and prefill
                and min(prefill) > 0.0
                and max(prefill) / min(prefill) <= maximum_spread
                and len(digests) == 1
                and digests == baseline_digests
                and all(row["candidate_hits"] == expected_hits for row in current)
                and all(not row["candidate_disabled_layers"] for row in current)
                and all(not row["candidate_failures"] for row in current)
            )
            median_prefill = _median(prefill)
            median_wall = _median(wall)
            item = {
                "case": case,
                "config": config,
                "valid": valid,
                "median_prefill_ms": median_prefill,
                "median_wall_ms": median_wall,
                "prefill_speedup": baseline_prefill / median_prefill if median_prefill else 0.0,
                "wall_speedup": baseline_wall / median_wall if median_wall else 0.0,
                "spread_ratio": max(prefill) / min(prefill) if prefill and min(prefill) else float("inf"),
                "hit_counts": [row["candidate_hits"] for row in current],
            }
            rows[f"p{prompt}/{case}"] = item
            if config is not None and valid:
                candidates.append(item)
            if config is None:
                gate_valid = gate_valid and valid
        winner = max(candidates, key=lambda row: row["prefill_speedup"]) if candidates else None
        qualified = bool(
            winner
            and winner["prefill_speedup"] >= minimum_prefill_speedup
            and winner["wall_speedup"] >= minimum_wall_speedup
        )
        gate_valid = gate_valid and bool(candidates)
        shape_policy[f"b8/p{prompt}"] = {
            "decision": "PROMOTE_FUSED_GATEUP" if qualified else "KEEP_PRODUCTION",
            "winner": winner["case"] if qualified and winner else "production",
            "config": winner["config"] if qualified and winner else None,
            "prefill_speedup": winner["prefill_speedup"] if winner else 0.0,
            "wall_speedup": winner["wall_speedup"] if winner else 0.0,
        }
    promoted = [key for key, value in shape_policy.items() if value["decision"] == "PROMOTE_FUSED_GATEUP"]
    selected = [shape_policy[key]["prefill_speedup"] for key in promoted]
    return {
        "decision": "INVALID_GATE" if not gate_valid else ("PROMOTE_SHAPE_DISPATCH" if promoted else "KEEP_PRODUCTION"),
        "valid": gate_valid,
        "apply_change": bool(gate_valid and promoted),
        "promoted_shapes": promoted,
        "shape_policy": shape_policy,
        "selected_prefill_geomean_speedup": _geomean(selected),
        "thresholds": {
            "maximum_spread": maximum_spread,
            "minimum_prefill_speedup": minimum_prefill_speedup,
            "minimum_wall_speedup": minimum_wall_speedup,
        },
        "cases": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("this exact-shape gate requires NVIDIA L4")
    prompts_requested = matrix.parse_csv_ints(args.prompt_tokens)
    if args.batch_size != 8 or prompts_requested != [512, 2048]:
        raise ValueError("this gate requires B8 and prompt tokens 512,2048")

    args.backend = "megagemm"
    _configure_megagemm_profile(args.model)
    os.environ["MEGAGEMM_BENCHMARK_TOKEN_DIGEST"] = "1"
    os.environ.pop("MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID", None)
    tokenizer = matrix.load_tokenizer(args.tokenizer or args.model, local_files_only=args.local_files_only)
    prompt_cache = {
        prompt: matrix.build_prompts(tokenizer, args.batch_size, prompt)[0]
        for prompt in prompts_requested
    }
    runner = matrix.make_runner(_runner_args(args), tokenizer)
    engine = getattr(runner, "_megagemm_engine", None)
    if engine is None:
        raise RuntimeError("MegaGemm runner did not expose the loaded engine")
    modules = _mlp_modules(engine.model)

    print("Gemma 4 E2B/L4 fused prefill MLP-body frontier", flush=True)
    print("  shape: B8, P512/P2048, BF16, 20xI12288 + 15xI6144", flush=True)
    print("  operation: fused gate/up GEMM -> BF16 GELU-tanh x up", flush=True)
    print("  model loads: 1; competing engine install: disabled", flush=True)
    preflight = _numeric_preflight(modules)
    config_by_name = dict(CONFIGS)
    active_cases: list[tuple[str, dict[str, int] | None]] = [("production", None)]
    active_cases.extend(
        (name, config_by_name[name]) for name in preflight["valid_cases"]
    )
    if len(active_cases) == 1:
        raise RuntimeError("no fused launch geometry passed numeric preflight")
    print("ACTIVE_CASES " + json.dumps([name for name, _ in active_cases]), flush=True)

    for case, config in active_cases:
        for prompt in prompts_requested:
            print(f"Warmup {case}/p{prompt}", flush=True)
            _run_once(
                runner, modules, prompt_cache[prompt], case=case, config=config,
                prompt_tokens=prompt, repeat=0, warmup=True,
            )

    base_schedule = [
        (case, config, prompt)
        for prompt in prompts_requested
        for case, config in active_cases
    ]
    samples: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        schedule = list(base_schedule)
        if repeat % 2 == 0:
            schedule.reverse()
        rotation = (repeat - 1) % len(schedule)
        schedule = schedule[rotation:] + schedule[:rotation]
        for case, config, prompt in schedule:
            sample = _run_once(
                runner, modules, prompt_cache[prompt], case=case, config=config,
                prompt_tokens=prompt, repeat=repeat, warmup=False,
            )
            samples.append(sample)
            print(
                f"Run {repeat}/{args.repeats} {case}/p{prompt}: "
                f"prefill={sample['prefill_ms']:.2f}ms wall={sample['wall_ms']:.2f}ms "
                f"hits={sample['candidate_hits']} errors={sample['candidate_failures']}",
                flush=True,
            )

    summary = summarize(
        samples,
        active_cases=active_cases,
        prompts=prompts_requested,
        maximum_spread=args.maximum_spread,
        minimum_prefill_speedup=args.minimum_prefill_speedup,
        minimum_wall_speedup=args.minimum_wall_speedup,
    )
    payload = {
        "benchmark": "gemma4_e2b_l4_b8_prefill_fused_gateup_frontier",
        "schema_version": 1,
        "model": args.model,
        "workloads": {"batch_size": 8, "prompt_tokens": prompts_requested, "output_tokens": 1},
        "method": {
            "model_loads": 1,
            "numeric_preflight": "real E2B weights, both intermediate widths",
            "timing": "uninstrumented full-model first-token and engine prefill wall",
            "correctness": "natural-token digest, 35/35 hits, deterministic numeric gate",
        },
        "preflight": preflight,
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
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DECISION " + json.dumps(summary, sort_keys=True), flush=True)
    print(f"Wrote: {args.output}", flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-tokens", default="512,2048")
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
    if args.repeats < 3:
        raise SystemExit("--repeats must be at least 3")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

