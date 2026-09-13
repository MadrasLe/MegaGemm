#!/usr/bin/env python3
"""One-load full-model PLE-tail frontier for Gemma 4 E2B/L4 B8 prefill.

The candidate reuses the production-proven residual/scale RMSNorm kernel to
fuse post-PLE RMSNorm, the PLE residual add, and the layer scalar.  GEMM paths
are unchanged.  Only rotated, natural-token full-model runs can promote it.
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


EXPECTED_DENSE_LAYERS = 35
CANDIDATE = "ple_tail_fused"


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        return 0.0
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def _layers(model) -> list[Any]:
    layers = [
        layer
        for layer in model.layers
        if hasattr(layer, "_gemma4_e2b_prefill_ple_tail_enabled")
        and bool(layer.is_gemma4)
        and not bool(layer.is_moe_layer)
        and layer.per_layer_input_gate is not None
        and layer.per_layer_projection is not None
        and layer.post_per_layer_input_norm is not None
    ]
    if len(layers) != EXPECTED_DENSE_LAYERS:
        raise RuntimeError(
            f"unexpected E2B dense PLE topology: {len(layers)}; expected 35"
        )
    return layers


def _numeric_preflight(layer) -> dict[str, Any]:
    import torch
    from megagemm.kernels.rmsnorm_triton import (
        rmsnorm_triton_residual_scale_next,
    )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(0xE2B4)
    branch = torch.randn(
        (32, 1536), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    residual = torch.randn(
        (32, 1536), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    with torch.inference_mode():
        branch_norm = layer.post_per_layer_input_norm(branch)
        reference = (residual + branch_norm).to(torch.bfloat16)
        reference = (
            reference * layer.layer_scalar.to(dtype=reference.dtype)
        ).to(reference.dtype)
        candidate, next_norm = rmsnorm_triton_residual_scale_next(
            branch,
            residual,
            layer.post_per_layer_input_norm.weight,
            layer.layer_scalar,
            None,
            layer.post_per_layer_input_norm.eps,
            norm_offset=bool(layer.post_per_layer_input_norm.offset),
        )
        repeat, _ = rmsnorm_triton_residual_scale_next(
            branch,
            residual,
            layer.post_per_layer_input_norm.weight,
            layer.layer_scalar,
            None,
            layer.post_per_layer_input_norm.eps,
            norm_offset=bool(layer.post_per_layer_input_norm.offset),
        )
        torch.cuda.synchronize()
        delta = candidate.float() - reference.float()
        max_abs = float(delta.abs().max().item())
        mean_abs = float(delta.abs().mean().item())
        rel_l2 = float(torch.linalg.vector_norm(delta).item()) / max(
            float(torch.linalg.vector_norm(reference.float()).item()), 1e-12
        )
        repeat_max = float((candidate - repeat).abs().max().item())
        valid = bool(
            next_norm is None
            and torch.isfinite(candidate).all().item()
            and repeat_max == 0.0
            and max_abs <= 0.03125
            and rel_l2 <= 0.001
        )
    row = {
        "valid": valid,
        "finite": bool(torch.isfinite(candidate).all().item()),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "relative_l2_error": rel_l2,
        "repeat_max_abs_error": repeat_max,
    }
    print("PREFLIGHT " + json.dumps(row, sort_keys=True), flush=True)
    return row


def _apply_case(layers: list[Any], enabled: bool) -> None:
    for layer in layers:
        layer._gemma4_e2b_prefill_ple_tail_enabled = enabled
        # The frontier deliberately measures both S521 and S2057. Production
        # policy is sequence-scoped, so clear its filter inside this A/B gate.
        layer._gemma4_e2b_prefill_ple_tail_sequences = set()
        layer._gemma4_e2b_prefill_ple_tail_hits = 0
        layer._gemma4_e2b_prefill_ple_tail_runtime_disabled = False
        layer._gemma4_e2b_prefill_ple_tail_failure = ""


def _run_once(
    runner,
    layers: list[Any],
    prompts: list[str],
    *,
    prompt_tokens: int,
    case: str,
    repeat: int,
) -> dict[str, Any]:
    _apply_case(layers, case == CANDIDATE)
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
    hits = sum(int(layer._gemma4_e2b_prefill_ple_tail_hits) for layer in layers)
    disabled = sum(
        bool(layer._gemma4_e2b_prefill_ple_tail_runtime_disabled)
        for layer in layers
    )
    failures = sorted(
        {
            str(layer._gemma4_e2b_prefill_ple_tail_failure)
            for layer in layers
            if layer._gemma4_e2b_prefill_ple_tail_failure
        }
    )
    return {
        "case": case,
        "prompt_tokens": prompt_tokens,
        "repeat": repeat,
        "prefill_ms": float(scheduler.get("prefill_time_ms") or 0.0),
        "wall_ms": float(result["elapsed_s"]) * 1000.0,
        "generated_token_digest": extra.get("generated_token_digest"),
        "candidate_hits": hits,
        "candidate_disabled_layers": disabled,
        "candidate_failures": failures,
    }


def _paired_speedups(
    production: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    metric: str,
) -> list[float]:
    baseline = {int(row["repeat"]): float(row[metric]) for row in production}
    current = {int(row["repeat"]): float(row[metric]) for row in candidate}
    return [
        baseline[repeat] / current[repeat]
        for repeat in sorted(baseline.keys() & current.keys())
        if baseline[repeat] > 0.0 and current[repeat] > 0.0
    ]


def summarize(
    samples: list[dict[str, Any]],
    *,
    prompts: list[int],
    maximum_spread: float,
    minimum_prefill_speedup: float,
    minimum_wall_speedup: float,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    shape_policy: dict[str, Any] = {}
    gate_valid = True
    for prompt_tokens in prompts:
        prompt_rows = [
            row for row in samples if int(row["prompt_tokens"]) == prompt_tokens
        ]
        production = [row for row in prompt_rows if row["case"] == "production"]
        candidate = [row for row in prompt_rows if row["case"] == CANDIDATE]
        production_prefill = [float(row["prefill_ms"]) for row in production]
        candidate_prefill = [float(row["prefill_ms"]) for row in candidate]
        production_wall = [float(row["wall_ms"]) for row in production]
        candidate_wall = [float(row["wall_ms"]) for row in candidate]
        production_digests = {row["generated_token_digest"] for row in production}
        candidate_digests = {row["generated_token_digest"] for row in candidate}
        paired_prefill = _paired_speedups(production, candidate, "prefill_ms")
        paired_wall = _paired_speedups(production, candidate, "wall_ms")
        valid = bool(
            len(production) >= 3
            and len(candidate) >= 3
            and min(production_prefill + candidate_prefill, default=0.0) > 0.0
            and max(production_prefill) / min(production_prefill) <= maximum_spread
            and max(candidate_prefill) / min(candidate_prefill) <= maximum_spread
            and len(production_digests) == 1
            and None not in production_digests
            and candidate_digests == production_digests
            and all(int(row["candidate_hits"]) == 0 for row in production)
            and all(
                int(row["candidate_hits"]) == EXPECTED_DENSE_LAYERS
                for row in candidate
            )
            and all(not row["candidate_disabled_layers"] for row in candidate)
            and all(not row["candidate_failures"] for row in candidate)
            and len(paired_prefill) >= 3
            and len(paired_wall) >= 3
        )
        production_prefill_median = _median(production_prefill)
        candidate_prefill_median = _median(candidate_prefill)
        production_wall_median = _median(production_wall)
        candidate_wall_median = _median(candidate_wall)
        ratio_prefill = (
            production_prefill_median / candidate_prefill_median
            if candidate_prefill_median else 0.0
        )
        ratio_wall = (
            production_wall_median / candidate_wall_median
            if candidate_wall_median else 0.0
        )
        paired_prefill_median = _median(paired_prefill)
        paired_wall_median = _median(paired_wall)
        promote = bool(
            valid
            and ratio_prefill >= minimum_prefill_speedup
            and paired_prefill_median >= minimum_prefill_speedup
            and ratio_wall >= minimum_wall_speedup
            and paired_wall_median >= minimum_wall_speedup
        )
        key = f"b8/p{prompt_tokens}"
        rows[key] = {
            "valid": valid,
            "production_prefill_ms": production_prefill_median,
            "candidate_prefill_ms": candidate_prefill_median,
            "production_wall_ms": production_wall_median,
            "candidate_wall_ms": candidate_wall_median,
            "median_ratio_prefill_speedup": ratio_prefill,
            "paired_median_prefill_speedup": paired_prefill_median,
            "paired_prefill_speedups": paired_prefill,
            "median_ratio_wall_speedup": ratio_wall,
            "paired_median_wall_speedup": paired_wall_median,
            "paired_wall_speedups": paired_wall,
        }
        shape_policy[key] = {
            "decision": "PROMOTE_PLE_TAIL" if promote else "KEEP_PRODUCTION",
            "winner": CANDIDATE if promote else "production",
            "prefill_speedup": min(ratio_prefill, paired_prefill_median),
            "wall_speedup": min(ratio_wall, paired_wall_median),
        }
        gate_valid = gate_valid and valid
    promoted = [
        key for key, row in shape_policy.items()
        if row["decision"] == "PROMOTE_PLE_TAIL"
    ]
    selected = [shape_policy[key]["prefill_speedup"] for key in promoted]
    return {
        "decision": (
            "INVALID_GATE" if not gate_valid
            else ("PROMOTE_SHAPE_DISPATCH" if promoted else "KEEP_PRODUCTION")
        ),
        "valid": gate_valid,
        "apply_change": bool(gate_valid and promoted),
        "promoted_shapes": promoted,
        "shape_policy": shape_policy,
        "rows": rows,
        "selected_prefill_geomean_speedup": _geomean(selected),
        "thresholds": {
            "maximum_spread": maximum_spread,
            "minimum_prefill_speedup": minimum_prefill_speedup,
            "minimum_wall_speedup": minimum_wall_speedup,
        },
        "promotion_rule": (
            "natural-token equality, 35/35 fused hits, no fallback, stable "
            "samples, and both ratio-of-medians and paired-median gates"
        ),
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
        raise RuntimeError("MegaGemm runner did not expose the loaded engine")
    layers = _layers(engine.model)

    print("Gemma 4 E2B/L4 B8 prefill PLE-tail frontier", flush=True)
    print("  fused body: post-PLE RMSNorm + residual add + layer scale", flush=True)
    print("  workloads: P512/P2048, O1, BF16, natural greedy tokens", flush=True)
    print("  model loads: 1; GEMMs unchanged; profiler disabled", flush=True)
    preflight = _numeric_preflight(layers[0])
    if not preflight["valid"]:
        raise RuntimeError("fused PLE tail failed numeric preflight")

    cases = ("production", CANDIDATE)
    for prompt_tokens in prompts_requested:
        for case in cases:
            print(f"Warmup {case}/p{prompt_tokens}", flush=True)
            _run_once(
                runner, layers, prompts[prompt_tokens],
                prompt_tokens=prompt_tokens, case=case, repeat=0,
            )

    schedule = [
        (prompt_tokens, case)
        for prompt_tokens in prompts_requested
        for case in cases
    ]
    samples: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        current = list(reversed(schedule)) if repeat % 2 == 0 else list(schedule)
        rotation = (repeat - 1) % len(current)
        current = current[rotation:] + current[:rotation]
        for prompt_tokens, case in current:
            sample = _run_once(
                runner, layers, prompts[prompt_tokens],
                prompt_tokens=prompt_tokens, case=case, repeat=repeat,
            )
            samples.append(sample)
            print(
                f"Run {repeat}/{args.repeats} {case}/p{prompt_tokens}: "
                f"prefill={sample['prefill_ms']:.2f}ms wall={sample['wall_ms']:.2f}ms "
                f"hits={sample['candidate_hits']}",
                flush=True,
            )

    summary = summarize(
        samples,
        prompts=prompts_requested,
        maximum_spread=args.maximum_spread,
        minimum_prefill_speedup=args.minimum_prefill_speedup,
        minimum_wall_speedup=args.minimum_wall_speedup,
    )
    payload = {
        "benchmark": "gemma4_e2b_l4_b8_prefill_ple_tail_frontier",
        "schema_version": 1,
        "model": args.model,
        "hardware_label": "1xl4",
        "dtype": "bf16",
        "workloads": {"batch_size": 8, "prompt_tokens": prompts_requested, "output_tokens": 1},
        "method": {
            "model_loads": 1,
            "gemm_paths": "unchanged",
            "timing": "rotated uninstrumented full-model comparisons",
            "correctness": "numeric preflight plus natural-token digest and 35/35 hits",
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
        raise SystemExit("full-model repeats must be at least 3")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
