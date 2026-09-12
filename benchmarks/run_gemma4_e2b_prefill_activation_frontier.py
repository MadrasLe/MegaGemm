#!/usr/bin/env python3
"""One-load full-model gate for Gemma 4 E2B/L4 prefill GELU×value fusion.

The gate rotates production and four Triton launch shapes through the exact
B8/P512 and B8/P2048 workloads.  It measures uninstrumented first-token wall
time and the engine's internal prefill wall time, checks natural-token digests,
and proves all 35 dense MLP layers actually entered the candidate path.
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


CASES: tuple[tuple[str, int | None], ...] = (
    ("production", None),
    ("fused_b128", 128),
    ("fused_b256", 256),
    ("fused_b512", 512),
    ("fused_b1024", 1024),
)
EXPECTED_DENSE_LAYERS = 35


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _geomean(values: list[float]) -> float:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    if len(positive) != len(values) or not positive:
        return 0.0
    return float(math.exp(sum(math.log(value) for value in positive) / len(positive)))


def _activation_modules(model) -> list[Any]:
    modules = [
        module
        for module in model.modules()
        if hasattr(module, "_gemma4_e2b_prefill_gated_activation_enabled")
    ]
    large = sum(int(module.intermediate_size) == 12288 for module in modules)
    small = sum(int(module.intermediate_size) == 6144 for module in modules)
    if len(modules) != EXPECTED_DENSE_LAYERS or (large, small) != (20, 15):
        raise RuntimeError(
            "unexpected E2B dense MLP topology: "
            f"total={len(modules)} large={large} small={small}; "
            "expected 35/20/15"
        )
    return modules


def _apply_case(modules: list[Any], block_size: int | None) -> None:
    enabled = block_size is not None
    for module in modules:
        module._gemma4_e2b_prefill_gated_activation_enabled = enabled
        if block_size is not None:
            module._gemma4_e2b_prefill_gated_activation_block_size = int(block_size)
        module._gemma4_e2b_prefill_gated_activation_hits = 0
        module._gemma4_e2b_prefill_gated_activation_runtime_disabled = False
        module._gemma4_e2b_prefill_gated_activation_failure = ""


def _run_once(
    runner,
    modules: list[Any],
    prompts: list[str],
    *,
    case_name: str,
    block_size: int | None,
    prompt_tokens: int,
    repeat: int,
    warmup: bool,
) -> dict[str, Any]:
    _apply_case(modules, block_size)
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
    hits = sum(
        int(module._gemma4_e2b_prefill_gated_activation_hits)
        for module in modules
    )
    failures = sorted(
        {
            str(module._gemma4_e2b_prefill_gated_activation_failure)
            for module in modules
            if module._gemma4_e2b_prefill_gated_activation_failure
        }
    )
    disabled_layers = sum(
        bool(module._gemma4_e2b_prefill_gated_activation_runtime_disabled)
        for module in modules
    )
    return {
        "case": case_name,
        "block_size": block_size,
        "prompt_tokens": prompt_tokens,
        "repeat": repeat,
        "warmup": warmup,
        "wall_ms": float(result["elapsed_s"]) * 1000.0,
        "prefill_ms": float(scheduler.get("prefill_time_ms") or 0.0),
        "generated_token_digest": extra.get("generated_token_digest"),
        "candidate_hits": hits,
        "candidate_disabled_layers": disabled_layers,
        "candidate_failures": failures,
        "production_routes": {
            "sliding_layers": int(
                runtime.get("gemma4_e2b_l4_sliding_prefill_enabled_layers") or 0
            ),
            "full_layers": int(
                runtime.get("gemma4_e2b_l4_full_prefill_expand_enabled_layers")
                or 0
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
    case_blocks = dict(CASES)
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
        for case_name, block_size in CASES:
            current = [row for row in prompt_rows if row["case"] == case_name]
            prefill_values = [float(row["prefill_ms"]) for row in current]
            wall_values = [float(row["wall_ms"]) for row in current]
            digests = {
                str(row["generated_token_digest"])
                for row in current
                if row.get("generated_token_digest")
            }
            expected_hits = 0 if block_size is None else EXPECTED_DENSE_LAYERS
            route_valid = all(
                row["production_routes"]["sliding_layers"] == 28
                and row["production_routes"]["full_layers"] == 7
                and not row["production_routes"]["full_error"]
                for row in current
            )
            valid = bool(
                len(current) >= 3
                and prefill_values
                and min(prefill_values) > 0.0
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
                "block_size": block_size,
                "prompt_tokens": prompt,
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
                "digests": sorted(digests),
                "hit_counts": [int(row["candidate_hits"]) for row in current],
            }
            rows[f"p{prompt}/{case_name}"] = summary_row
            if block_size is not None and valid:
                candidates.append(summary_row)
            if block_size is None:
                experiment_valid = experiment_valid and valid

        winner = (
            max(candidates, key=lambda row: float(row["prefill_speedup"]))
            if candidates
            else None
        )
        qualified = bool(
            winner is not None
            and float(winner["prefill_speedup"]) >= minimum_prefill_speedup
            and float(winner["wall_speedup"]) >= minimum_wall_speedup
        )
        experiment_valid = experiment_valid and bool(candidates)
        shape_policy[f"b8/p{prompt}"] = {
            "decision": "PROMOTE_FUSED" if qualified else "KEEP_PRODUCTION",
            "winner": winner["case"] if qualified and winner else "production",
            "block_size": winner["block_size"] if qualified and winner else None,
            "prefill_speedup": winner["prefill_speedup"] if winner else 0.0,
            "wall_speedup": winner["wall_speedup"] if winner else 0.0,
        }

    promoted = [key for key, row in shape_policy.items() if row["decision"] == "PROMOTE_FUSED"]
    selected_prefill = [
        float(shape_policy[f"b8/p{prompt}"]["prefill_speedup"])
        for prompt in prompts
        if shape_policy[f"b8/p{prompt}"]["decision"] == "PROMOTE_FUSED"
    ]
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
        "selected_prefill_geomean_speedup": _geomean(selected_prefill),
        "thresholds": {
            "maximum_spread": maximum_spread,
            "minimum_prefill_speedup": minimum_prefill_speedup,
            "minimum_wall_speedup": minimum_wall_speedup,
        },
        "cases": rows,
        "case_blocks": case_blocks,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    gpu = torch.cuda.get_device_name(0)
    if "L4" not in gpu.upper():
        raise RuntimeError(f"this gate requires NVIDIA L4, got {gpu}")

    prompts_requested = matrix.parse_csv_ints(args.prompt_tokens)
    if prompts_requested != [512, 2048]:
        raise ValueError("this exact-shape gate requires --prompt-tokens 512,2048")
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
        raise RuntimeError("MegaGemm benchmark runner did not expose its loaded engine")
    modules = _activation_modules(engine.model)

    print("Gemma 4 E2B/L4 prefill activation frontier", flush=True)
    print(f"  gpu: {gpu}", flush=True)
    print("  workloads: B8/P512/O1 and B8/P2048/O1", flush=True)
    print("  cases: production + Triton blocks 128/256/512/1024", flush=True)
    print("  model loads: 1", flush=True)
    print("  tokens: natural greedy; profiler: disabled", flush=True)

    for case_name, block_size in CASES:
        for prompt in prompts_requested:
            print(f"Warmup {case_name}/p{prompt}", flush=True)
            _run_once(
                runner,
                modules,
                prompt_cache[prompt],
                case_name=case_name,
                block_size=block_size,
                prompt_tokens=prompt,
                repeat=0,
                warmup=True,
            )

    base_schedule = [
        (case_name, block_size, prompt)
        for prompt in prompts_requested
        for case_name, block_size in CASES
    ]
    samples: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        schedule = list(base_schedule)
        if repeat % 2 == 0:
            schedule.reverse()
        rotation = (repeat - 1) % len(schedule)
        schedule = schedule[rotation:] + schedule[:rotation]
        for case_name, block_size, prompt in schedule:
            sample = _run_once(
                runner,
                modules,
                prompt_cache[prompt],
                case_name=case_name,
                block_size=block_size,
                prompt_tokens=prompt,
                repeat=repeat,
                warmup=False,
            )
            samples.append(sample)
            print(
                f"Run {repeat}/{args.repeats} {case_name}/p{prompt}: "
                f"prefill={sample['prefill_ms']:.2f}ms "
                f"wall={sample['wall_ms']:.2f}ms hits={sample['candidate_hits']}",
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
        "benchmark": "gemma4_e2b_l4_prefill_activation_frontier",
        "schema_version": 1,
        "model": args.model,
        "hardware_label": "1xl4",
        "dtype": "bf16",
        "workloads": {
            "batch_size": args.batch_size,
            "prompt_tokens": prompts_requested,
            "output_tokens": 1,
        },
        "method": {
            "model_loads": 1,
            "warmups_per_case_shape": 1,
            "repeats": args.repeats,
            "schedule": "alternating-direction rotated case/shape matrix",
            "timing": "uninstrumented full-model wall and engine prefill wall",
            "correctness": "natural-token digest plus exact candidate hit audit",
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
    if args.batch_size != 8:
        raise SystemExit("this gate is exact to batch size 8")
    if args.repeats < 3:
        raise SystemExit("--repeats must be at least 3")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
