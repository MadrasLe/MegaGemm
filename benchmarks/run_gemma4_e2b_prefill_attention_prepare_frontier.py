#!/usr/bin/env python3
"""One-load full-model gate for the Gemma 4 E2B/L4 attention frontend.

The experiment compares production with four shape-aware Triton launch
policies at B8/P512 and B8/P2048.  A candidate is valid only when every one of
the 15 KV-source layers uses the fused Q/K/V RMSNorm + RoPE + layout path,
natural greedy tokens match production, no layer falls back, and both the
engine prefill time and uninstrumented wall time are stable.
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


# Each candidate specifies (num_warps, num_stages) independently for the
# sliding H256 and full-attention H512 frontends.  The winning pair is selected
# independently for S521 and S2057 by the full-model gate.
CASES: tuple[tuple[str, dict[int, tuple[int, int, bool]] | None], ...] = (
    ("production", None),
    ("onepass_w4", {256: (4, 2, False), 512: (4, 2, False)}),
    ("split_h256w2_h512w4", {256: (2, 2, True), 512: (4, 2, True)}),
    ("split_h256w4_h512w4", {256: (4, 2, True), 512: (4, 2, True)}),
    ("split_h256w4_h512w8", {256: (4, 2, True), 512: (8, 2, True)}),
)
EXPECTED_ATTENTION_LAYERS = 35
EXPECTED_KV_SOURCE_LAYERS = 15


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        return 0.0
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def _attention_modules(model) -> tuple[list[Any], list[Any]]:
    modules = [
        module
        for module in model.modules()
        if hasattr(module, "_gemma4_e2b_l4_fused_attn_prepare_enabled")
    ]
    sources = [module for module in modules if not bool(module.is_kv_shared)]
    if len(modules) != EXPECTED_ATTENTION_LAYERS or len(sources) != EXPECTED_KV_SOURCE_LAYERS:
        raise RuntimeError(
            "unexpected E2B attention topology: "
            f"total={len(modules)} kv_sources={len(sources)}; expected 35/15"
        )
    head_dims = sorted({int(module.head_dim) for module in sources})
    if head_dims != [256, 512]:
        raise RuntimeError(f"unexpected E2B source head dimensions: {head_dims}")
    return modules, sources


def _apply_case(
    modules: list[Any],
    launch_by_head_dim: dict[int, tuple[int, int, bool]] | None,
) -> None:
    enabled = launch_by_head_dim is not None
    for module in modules:
        module._gemma4_e2b_l4_fused_attn_prepare_enabled = enabled
        module._gemma4_e2b_l4_fused_attn_prepare_launch_by_shape = (
            {
                (521, head_dim): launch
                for head_dim, launch in (launch_by_head_dim or {}).items()
            }
            | {
                (2057, head_dim): launch
                for head_dim, launch in (launch_by_head_dim or {}).items()
            }
        )
        module._gemma4_e2b_l4_fused_attn_prepare_hits = 0
        module._gemma4_e2b_l4_fused_attn_prepare_failure = ""
        module._gemma4_fused_attn_prepare_hits = 0
        module._gemma4_fused_attn_prepare_skip_reason = ""
        module._gemma4_fused_attn_prepare_disabled = False


def _run_once(
    runner,
    modules: list[Any],
    sources: list[Any],
    prompts: list[str],
    *,
    case_name: str,
    launch_by_head_dim: dict[int, tuple[int, int, bool]] | None,
    prompt_tokens: int,
    repeat: int,
    warmup: bool,
) -> dict[str, Any]:
    _apply_case(modules, launch_by_head_dim)
    matrix.sync_cuda()
    result = runner(prompts, 1)
    matrix.sync_cuda()
    if int(result["generated_tokens"]) != len(prompts):
        raise RuntimeError(
            f"{case_name}/p{prompt_tokens} generated {result['generated_tokens']} "
            f"tokens; expected {len(prompts)}"
        )
    extra = dict(result.get("extra") or {})
    scheduler = dict(extra.get("scheduler_stats") or {})
    runtime = dict(extra.get("decode_runtime_stats") or {})
    candidate_hits = sum(
        int(module._gemma4_e2b_l4_fused_attn_prepare_hits) for module in sources
    )
    disabled_layers = sum(
        bool(module._gemma4_fused_attn_prepare_disabled) for module in sources
    )
    failures = sorted(
        {
            str(module._gemma4_e2b_l4_fused_attn_prepare_failure)
            for module in sources
            if module._gemma4_e2b_l4_fused_attn_prepare_failure
        }
    )
    return {
        "case": case_name,
        "launch_by_head_dim": launch_by_head_dim,
        "prompt_tokens": prompt_tokens,
        "repeat": repeat,
        "warmup": warmup,
        "wall_ms": float(result["elapsed_s"]) * 1000.0,
        "prefill_ms": float(scheduler.get("prefill_time_ms") or 0.0),
        "generated_token_digest": extra.get("generated_token_digest"),
        "candidate_hits": candidate_hits,
        "candidate_disabled_layers": disabled_layers,
        "candidate_failures": failures,
        "source_layers_by_head_dim": {
            str(head_dim): sum(int(module.head_dim) == head_dim for module in sources)
            for head_dim in (256, 512)
        },
        "production_routes": {
            "sliding_layers": int(
                runtime.get("gemma4_e2b_l4_sliding_prefill_enabled_layers") or 0
            ),
            "full_layers": int(
                runtime.get("gemma4_e2b_l4_full_prefill_expand_enabled_layers") or 0
            ),
            "full_error": str(
                runtime.get("gemma4_e2b_l4_full_prefill_expand_error") or ""
            ),
        },
    }


def summarize(
    samples: list[dict[str, Any]],
    *,
    prompts: list[int],
    maximum_spread: float,
    minimum_prefill_speedup: float,
    minimum_wall_speedup: float,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    shape_policy: dict[str, dict[str, Any]] = {}
    experiment_valid = True
    for prompt in prompts:
        prompt_rows = [row for row in samples if int(row["prompt_tokens"]) == prompt]
        baseline_rows = [row for row in prompt_rows if row["case"] == "production"]
        baseline_prefill = _median([float(row["prefill_ms"]) for row in baseline_rows])
        baseline_wall = _median([float(row["wall_ms"]) for row in baseline_rows])
        baseline_digests = {
            str(row["generated_token_digest"])
            for row in baseline_rows
            if row.get("generated_token_digest")
        }
        candidates: list[dict[str, Any]] = []
        for case_name, launch in CASES:
            current = [row for row in prompt_rows if row["case"] == case_name]
            prefill_values = [float(row["prefill_ms"]) for row in current]
            wall_values = [float(row["wall_ms"]) for row in current]
            digests = {
                str(row["generated_token_digest"])
                for row in current
                if row.get("generated_token_digest")
            }
            expected_hits = 0 if launch is None else EXPECTED_KV_SOURCE_LAYERS
            route_valid = all(
                row["production_routes"]["sliding_layers"] == 28
                and row["production_routes"]["full_layers"] == 7
                and not row["production_routes"]["full_error"]
                for row in current
            )
            valid = bool(
                len(current) >= 3
                and min(prefill_values, default=0.0) > 0.0
                and max(prefill_values) / min(prefill_values) <= maximum_spread
                and digests == baseline_digests
                and len(digests) == 1
                and all(int(row["candidate_hits"]) == expected_hits for row in current)
                and all(not row["candidate_disabled_layers"] for row in current)
                and all(not row["candidate_failures"] for row in current)
                and route_valid
            )
            median_prefill = _median(prefill_values)
            median_wall = _median(wall_values)
            summary_row = {
                "case": case_name,
                "launch_by_head_dim": launch,
                "valid": valid,
                "median_prefill_ms": median_prefill,
                "median_wall_ms": median_wall,
                "prefill_speedup": (
                    baseline_prefill / median_prefill if median_prefill > 0.0 else 0.0
                ),
                "wall_speedup": baseline_wall / median_wall if median_wall > 0.0 else 0.0,
                "spread_ratio": (
                    max(prefill_values) / min(prefill_values)
                    if prefill_values and min(prefill_values) > 0.0
                    else float("inf")
                ),
                "hit_counts": [int(row["candidate_hits"]) for row in current],
            }
            rows[f"p{prompt}/{case_name}"] = summary_row
            if launch is not None and valid:
                candidates.append(summary_row)
            if launch is None:
                experiment_valid = experiment_valid and valid

        winner = max(candidates, key=lambda row: float(row["prefill_speedup"])) if candidates else None
        qualified = bool(
            winner
            and float(winner["prefill_speedup"]) >= minimum_prefill_speedup
            and float(winner["wall_speedup"]) >= minimum_wall_speedup
        )
        experiment_valid = experiment_valid and bool(candidates)
        shape_policy[f"b8/p{prompt}"] = {
            "decision": "PROMOTE_FUSED" if qualified else "KEEP_PRODUCTION",
            "winner": winner["case"] if qualified and winner else "production",
            "launch_by_head_dim": winner["launch_by_head_dim"] if qualified and winner else None,
            "prefill_speedup": winner["prefill_speedup"] if winner else 0.0,
            "wall_speedup": winner["wall_speedup"] if winner else 0.0,
        }

    promoted = [key for key, row in shape_policy.items() if row["decision"] == "PROMOTE_FUSED"]
    selected = [float(shape_policy[key]["prefill_speedup"]) for key in promoted]
    decision = (
        "INVALID_GATE"
        if not experiment_valid
        else ("PROMOTE_SHAPE_DISPATCH" if promoted else "KEEP_PRODUCTION")
    )
    return {
        "decision": decision,
        "apply_change": bool(experiment_valid and promoted),
        "valid": experiment_valid,
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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    gpu = torch.cuda.get_device_name(0)
    if "L4" not in gpu.upper():
        raise RuntimeError(f"this gate requires NVIDIA L4, got {gpu}")
    prompts_requested = matrix.parse_csv_ints(args.prompt_tokens)
    if args.batch_size != 8 or prompts_requested != [512, 2048]:
        raise ValueError("this exact-shape gate requires B8 and P512,P2048")

    args.backend = "megagemm"
    _configure_megagemm_profile(args.model)
    os.environ["MEGAGEMM_BENCHMARK_TOKEN_DIGEST"] = "1"
    os.environ.pop("MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID", None)
    tokenizer = matrix.load_tokenizer(
        args.tokenizer or args.model,
        local_files_only=args.local_files_only,
    )
    prompt_cache = {
        prompt: matrix.build_prompts(tokenizer, args.batch_size, prompt)[0]
        for prompt in prompts_requested
    }
    runner = matrix.make_runner(_runner_args(args), tokenizer)
    engine = getattr(runner, "_megagemm_engine", None)
    if engine is None:
        raise RuntimeError("MegaGemm runner did not expose its loaded engine")
    modules, sources = _attention_modules(engine.model)

    print("Gemma 4 E2B/L4 B8 prefill attention-prepare frontier", flush=True)
    print(f"  gpu: {gpu}", flush=True)
    print("  workloads: P512/P2048, O1, natural greedy tokens", flush=True)
    print("  cases: production + four H256/H512 Triton launch policies", flush=True)
    print(f"  topology: {len(modules)} attention layers, {len(sources)} KV sources", flush=True)
    print("  model loads: 1; profiler: disabled", flush=True)

    for case_name, launch in CASES:
        for prompt in prompts_requested:
            print(f"Warmup {case_name}/p{prompt}", flush=True)
            _run_once(
                runner,
                modules,
                sources,
                prompt_cache[prompt],
                case_name=case_name,
                launch_by_head_dim=launch,
                prompt_tokens=prompt,
                repeat=0,
                warmup=True,
            )

    schedule = [
        (case_name, launch, prompt)
        for prompt in prompts_requested
        for case_name, launch in CASES
    ]
    samples: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        current = list(reversed(schedule)) if repeat % 2 == 0 else list(schedule)
        rotation = (repeat - 1) % len(current)
        current = current[rotation:] + current[:rotation]
        for case_name, launch, prompt in current:
            sample = _run_once(
                runner,
                modules,
                sources,
                prompt_cache[prompt],
                case_name=case_name,
                launch_by_head_dim=launch,
                prompt_tokens=prompt,
                repeat=repeat,
                warmup=False,
            )
            samples.append(sample)
            print(
                f"Run {repeat}/{args.repeats} {case_name}/p{prompt}: "
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
        "benchmark": "gemma4_e2b_l4_prefill_attention_prepare_frontier",
        "schema_version": 1,
        "model": args.model,
        "hardware_label": "1xl4",
        "dtype": "bf16",
        "workloads": {"batch_size": 8, "prompt_tokens": prompts_requested, "output_tokens": 1},
        "method": {
            "model_loads": 1,
            "warmups_per_case_shape": 1,
            "repeats": args.repeats,
            "timing": "uninstrumented full-model wall and engine prefill wall",
            "correctness": "natural-token digest plus exact 15/15 KV-source hit audit",
        },
        "system": {
            "gpu": matrix.gpu_snapshot(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "git": matrix.git_snapshot(),
        },
        "samples": samples,
        "summary": summary,
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
