#!/usr/bin/env python3
"""One-load, full-model Gemma 4 E2B/L4/B8 compute frontier.

The screen keeps the promoted one-step CUDA Graph with a 16-token scheduler
burst fixed and changes only one compute family at a time.  Every candidate is
measured on P512/P2048 and O16/O128 with natural greedy tokens.  A fresh paired
production-versus-combination phase validates the family winners after the
screen, without loading the model again.

This is deliberately not a kernel microbenchmark.  Setup, compilation and graph
capture are excluded from measured samples, and no profiler runs in the timing
region.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import benchmark_inference_matrix as matrix
from benchmarks.run_gemma4_e2b_b1_decode_frontier_gate import (
    _configure_environment,
    _run,
)
from benchmarks.run_gemma4_e2b_b1_execution_gate import (
    compare_generated_tokens,
    effective_max_prompt_tokens,
)


@dataclass(frozen=True)
class Workload:
    prompt_tokens: int
    output_tokens: int

    @property
    def key(self) -> str:
        return f"p{self.prompt_tokens}_o{self.output_tokens}"


@dataclass(frozen=True)
class ComputeCase:
    name: str
    family: str
    h256_warps: int = 2
    h512_short_segments: int = 32
    h512_long_segments: int = 32
    h512_tile: int = 16
    h512_warps: int = 4
    h512_stages: int = 3
    h512_reduce_warps: int = 4
    lm_block_n: int = 256
    lm_block_k: int = 128
    lm_warps: int = 4
    lm_stages: int = 2
    mlp_mode: str = "production"

    def h512_segments(self, prompt_tokens: int) -> int:
        return (
            self.h512_short_segments
            if int(prompt_tokens) <= 768
            else self.h512_long_segments
        )


PRODUCTION = ComputeCase("production", "production")
SCREEN_CASES = (
    PRODUCTION,
    ComputeCase("sliding_h256_w1", "attention", h256_warps=1),
    ComputeCase(
        "full_h512_short_seg8",
        "attention",
        h512_short_segments=8,
    ),
    ComputeCase(
        "full_h512_short_seg16",
        "attention",
        h512_short_segments=16,
    ),
    ComputeCase(
        "full_h512_short_seg8_tile32",
        "attention",
        h512_short_segments=8,
        h512_tile=32,
    ),
    # This is a full-model control for the earlier LM-head result.  It remains
    # in the screen because CUDA Graph promotion changed host execution, but it
    # must still win the complete request rather than its isolated kernel.
    ComputeCase("lm_head_bn64", "lm_head", lm_block_n=64),
    ComputeCase("mlp_fused_gateup", "mlp", mlp_mode="fused_gateup"),
    ComputeCase("mlp_deepfusion_down", "mlp", mlp_mode="deepfusion"),
    ComputeCase("mlp_fused_both", "mlp", mlp_mode="fused_both"),
)

CONTROLLED_ATTENTION_ENV = (
    "MEGAGEMM_PAGED_DECODE_WARPS_H256",
    "MEGAGEMM_GEMMA4_E2B_L4_H512_ATTN_SEGMENTS",
    "MEGAGEMM_GEMMA4_E2B_L4_H512_ATTN_TILE",
    "MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_WARPS",
    "MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_STAGES",
    "MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_REDUCE_WARPS",
)

COUNTER_KEYS = (
    "gemma4_flat_fused_gateup_hits",
    "gemma4_flat_deepfusion_hits",
    "gemma4_cublaslt_gateup_decode_hits",
    "fused_rmsnorm_lm_head_argmax_hits",
)


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        return 0.0
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def _counter_snapshot(model: Any) -> dict[str, int]:
    stats = model.decode_runtime_stats()
    paged = stats.get("paged_decode_runtime") or {}
    result = {key: int(stats.get(key) or 0) for key in COUNTER_KEYS}
    result.update(
        {
            "paged_gqa2_direct_hits": int(paged.get("gqa2_direct_hits") or 0),
            "paged_grouped_segmented_hits": int(
                paged.get("grouped_segmented_hits") or 0
            ),
        }
    )
    return result


def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: int(after.get(key, 0) - before.get(key, 0)) for key in before}


def _production_mlp_state(model: Any) -> dict[str, Any]:
    names = (
        "_gemma4_flat_policy_fused_gateup_rows",
        "_gemma4_flat_policy_deepfusion_rows",
        "_gemma4_flat_policy_cublas_gateup_rows",
        "_gemma4_flat_policy_cublas_down_rows",
        "_gemma4_flat_cublaslt_gateup_enabled",
        "_gemma4_flat_cublaslt_gateup_algorithms",
    )
    return {name: getattr(model, name) for name in names}


def _restore_mlp_state(model: Any, state: dict[str, Any]) -> None:
    for name, value in state.items():
        setattr(model, name, value)
    model._gemma4_flat_fused_gateup_use_cache.clear()
    model._gemma4_flat_deepfusion_use_cache.clear()
    model._gemma4_flat_fused_gateup_runtime_disabled = False
    model._gemma4_flat_cublaslt_gateup_runtime_disabled = False
    model._gemma4_flat_cublaslt_gateup_failure = ""


def _apply_case(
    model: Any,
    lm_kernel: Any,
    case: ComputeCase,
    workload: Workload,
    production_mlp_state: dict[str, Any],
) -> None:
    for name in CONTROLLED_ATTENTION_ENV:
        os.environ.pop(name, None)
    os.environ.update(
        {
            "MEGAGEMM_PAGED_DECODE_WARPS_H256": str(case.h256_warps),
            "MEGAGEMM_GEMMA4_E2B_L4_H512_ATTN_SEGMENTS": str(
                case.h512_segments(workload.prompt_tokens)
            ),
            "MEGAGEMM_GEMMA4_E2B_L4_H512_ATTN_TILE": str(case.h512_tile),
            "MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_WARPS": str(
                case.h512_warps
            ),
            "MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_STAGES": str(
                case.h512_stages
            ),
            "MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_REDUCE_WARPS": str(
                case.h512_reduce_warps
            ),
        }
    )
    from megagemm.kernels import paged_attention

    paged_attention._GROUPED_SEGMENTED_DECODE_DISABLED = False
    paged_attention._GROUPED_SEGMENTED_DECODE_FAILURE = ""
    paged_attention._GROUPED_SEGMENTED_DECODE_SELECTED_SEGMENTS.clear()
    paged_attention._GROUPED_SEGMENTED_DECODE_SELECTED_TILE_SIZES.clear()

    lm_kernel._CFG_FORCED_BN = int(case.lm_block_n)
    lm_kernel._CFG_FORCED_BK = int(case.lm_block_k)
    lm_kernel._CFG_FORCED_WARPS = int(case.lm_warps)
    lm_kernel._CFG_FORCED_STAGES = int(case.lm_stages)
    lm_kernel._CFG_TRITON_REDUCE = True

    _restore_mlp_state(model, production_mlp_state)
    if case.mlp_mode in ("fused_gateup", "fused_both"):
        model._gemma4_flat_policy_cublas_gateup_rows = ()
        model._gemma4_flat_policy_fused_gateup_rows = (8,)
    if case.mlp_mode in ("deepfusion", "fused_both"):
        model._gemma4_flat_policy_cublas_down_rows = ()
        model._gemma4_flat_policy_deepfusion_rows = (8,)


def _graph_errors(row: dict[str, Any]) -> list[str]:
    graph = (row.get("scheduler_stats") or {}).get("decode_cuda_graphs") or {}
    errors: list[str] = []
    if int(graph.get("failures") or 0):
        errors.append(f"CUDA Graph failure: {graph.get('last_failure') or 'unknown'}")
    if not graph.get("enabled"):
        errors.append("promoted B8 CUDA Graph is disabled")
    if int(graph.get("token_burst_size") or 0) != 16:
        errors.append(
            f"graph token burst is {graph.get('token_burst_size')}, expected 16"
        )
    if int(graph.get("replays") or 0) <= 0:
        errors.append("CUDA Graph did not replay")
    if not graph.get("request_scheduler_reused"):
        errors.append("request scheduler/graph owner was not reused")
    return errors


def _validate_measurement(
    row: dict[str, Any],
    reference: dict[str, Any],
    workload: Workload,
) -> list[str]:
    errors: list[str] = []
    expected = 8 * workload.output_tokens
    if int(row.get("generated_tokens") or 0) != expected:
        errors.append(
            f"generated {row.get('generated_tokens')} tokens, expected {expected}"
        )
    if row.get("lengths") != [workload.output_tokens] * 8:
        errors.append("per-request output lengths differ from the workload")
    if not compare_generated_tokens(row, reference)["exact_match"]:
        errors.append("natural greedy tokens differ from production")
    if row.get("engine_prompt_lengths") != reference.get("engine_prompt_lengths"):
        errors.append("effective prompt lengths differ from production")
    if int(
        (row.get("scheduler_stats") or {}).get("benchmark_forced_token_id", -1)
    ) != -1:
        errors.append("forced-token benchmarking must remain disabled")
    if not math.isfinite(float(row.get("elapsed_s") or 0.0)):
        errors.append("wall time is not finite")
    if workload.output_tokens > 1:
        errors.extend(_graph_errors(row))
    return errors


def _route_errors(case: ComputeCase, delta: dict[str, int]) -> list[str]:
    errors: list[str] = []
    if case.mlp_mode in ("fused_gateup", "fused_both") and not delta[
        "gemma4_flat_fused_gateup_hits"
    ]:
        errors.append("requested fused gate-up route produced no hits")
    if case.mlp_mode in ("deepfusion", "fused_both") and not delta[
        "gemma4_flat_deepfusion_hits"
    ]:
        errors.append("requested deepfusion down route produced no hits")
    return errors


def _attention_route_errors(
    case: ComputeCase,
    workload: Workload,
    runtime_stats: dict[str, Any],
) -> list[str]:
    paged = runtime_stats.get("paged_decode_runtime") or {}
    topology = "e2b_l4_full_h512_gqa8"
    selected_segments = paged.get("grouped_segmented_selected_segments") or {}
    selected_tiles = paged.get("grouped_segmented_selected_tile_sizes") or {}
    expected_segments = case.h512_segments(workload.prompt_tokens)
    errors: list[str] = []
    if int(selected_segments.get(topology) or 0) != expected_segments:
        errors.append(
            f"H512 route selected {selected_segments.get(topology)} segments, "
            f"expected {expected_segments}"
        )
    if int(selected_tiles.get(topology) or 0) != case.h512_tile:
        errors.append(
            f"H512 route selected tile {selected_tiles.get(topology)}, "
            f"expected {case.h512_tile}"
        )
    return errors


def summarize_screen(
    samples: list[dict[str, Any]],
    cases: tuple[ComputeCase, ...],
    workloads: tuple[Workload, ...],
    *,
    repeats: int,
    maximum_spread: float,
) -> dict[str, Any]:
    by_case: dict[str, dict[str, Any]] = {}
    for case in cases:
        scenario_rows: dict[str, Any] = {}
        speedups: list[float] = []
        valid = True
        for workload in workloads:
            found = [
                row
                for row in samples
                if row["case"] == case.name and row["workload"] == workload.key
            ]
            rates = [float(row["decode_tps"]) for row in found]
            total_rates = [float(row["output_tps"]) for row in found]
            errors = [error for row in found for error in row.get("errors", ())]
            spread = max(rates) / min(rates) if rates and min(rates) > 0.0 else None
            scenario_valid = bool(
                len(found) == repeats
                and not errors
                and spread is not None
                and spread <= maximum_spread
            )
            scenario_rows[workload.key] = {
                "valid": scenario_valid,
                "samples": len(found),
                "median_decode_tps": _median(rates),
                "median_output_tps": _median(total_rates),
                "spread": spread,
                "errors": errors,
            }
            valid = valid and scenario_valid
        by_case[case.name] = {
            "family": case.family,
            "config": asdict(case),
            "valid": valid,
            "scenarios": scenario_rows,
        }

    baseline = by_case[PRODUCTION.name]
    for case in cases:
        row = by_case[case.name]
        speedups = []
        total_speedups = []
        for workload in workloads:
            candidate = row["scenarios"][workload.key]
            production = baseline["scenarios"][workload.key]
            decode_base = float(production["median_decode_tps"])
            total_base = float(production["median_output_tps"])
            decode_speedup = (
                float(candidate["median_decode_tps"]) / decode_base
                if decode_base > 0.0
                else 0.0
            )
            total_speedup = (
                float(candidate["median_output_tps"]) / total_base
                if total_base > 0.0
                else 0.0
            )
            candidate["decode_speedup_vs_production"] = decode_speedup
            candidate["output_speedup_vs_production"] = total_speedup
            speedups.append(decode_speedup)
            total_speedups.append(total_speedup)
        row["geomean_decode_speedup"] = _geomean(speedups)
        row["worst_decode_speedup"] = min(speedups, default=0.0)
        row["geomean_output_speedup"] = _geomean(total_speedups)
        row["worst_output_speedup"] = min(total_speedups, default=0.0)

    family_winners: dict[str, str] = {}
    for family in ("attention", "lm_head", "mlp"):
        eligible = [
            case
            for case in cases
            if case.family == family and by_case[case.name]["valid"]
        ]
        winner = max(
            eligible,
            key=lambda case: float(by_case[case.name]["geomean_decode_speedup"]),
            default=PRODUCTION,
        )
        winner_row = by_case[winner.name]
        if (
            float(winner_row["geomean_decode_speedup"]) <= 1.0
            or float(winner_row["worst_decode_speedup"]) < 0.99
        ):
            winner = PRODUCTION
        family_winners[family] = winner.name
    return {
        "valid": bool(baseline["valid"]),
        "cases": by_case,
        "family_winners": family_winners,
    }


def combine_family_winners(
    cases: tuple[ComputeCase, ...],
    winners: dict[str, str],
) -> ComputeCase:
    by_name = {case.name: case for case in cases}
    combined = PRODUCTION
    attention = by_name.get(winners.get("attention", "production"), PRODUCTION)
    lm_head = by_name.get(winners.get("lm_head", "production"), PRODUCTION)
    mlp = by_name.get(winners.get("mlp", "production"), PRODUCTION)
    combined = replace(
        combined,
        name="best_combination",
        family="combined",
        h256_warps=attention.h256_warps,
        h512_short_segments=attention.h512_short_segments,
        h512_long_segments=attention.h512_long_segments,
        h512_tile=attention.h512_tile,
        h512_warps=attention.h512_warps,
        h512_stages=attention.h512_stages,
        h512_reduce_warps=attention.h512_reduce_warps,
        lm_block_n=lm_head.lm_block_n,
        lm_block_k=lm_head.lm_block_k,
        lm_warps=lm_head.lm_warps,
        lm_stages=lm_head.lm_stages,
        mlp_mode=mlp.mlp_mode,
    )
    return combined


def final_decision(
    summary: dict[str, Any],
    *,
    minimum_speedup: float,
    maximum_spread: float,
    policy_changed: bool = True,
) -> dict[str, Any]:
    production = summary["cases"].get("production") or {}
    candidate = summary["cases"].get("best_combination") or {}
    valid = bool(production.get("valid") and candidate.get("valid"))
    decode_speedup = float(candidate.get("geomean_decode_speedup") or 0.0)
    output_speedup = float(candidate.get("geomean_output_speedup") or 0.0)
    worst_decode = float(candidate.get("worst_decode_speedup") or 0.0)
    worst_output = float(candidate.get("worst_output_speedup") or 0.0)
    promote = bool(
        policy_changed
        and valid
        and decode_speedup >= minimum_speedup
        and output_speedup >= minimum_speedup
        and worst_decode >= 0.995
        and worst_output >= 0.995
        and all(
            float(item.get("spread") or float("inf")) <= maximum_spread
            for item in candidate.get("scenarios", {}).values()
        )
    )
    return {
        "decision": "PROMOTE_COMBINATION" if promote else "KEEP_PRODUCTION",
        "apply_change": promote,
        "valid": valid,
        "geomean_decode_speedup": decode_speedup,
        "geomean_output_speedup": output_speedup,
        "worst_decode_speedup": worst_decode,
        "worst_output_speedup": worst_output,
        "minimum_speedup": minimum_speedup,
        "policy_changed": bool(policy_changed),
    }


def changes_compute_policy(case: ComputeCase) -> bool:
    return replace(case, name=PRODUCTION.name, family=PRODUCTION.family) != PRODUCTION


def _rotated(cases: tuple[ComputeCase, ...], repeat: int) -> list[ComputeCase]:
    ordered = list(cases)
    if repeat % 2:
        ordered.reverse()
    offset = repeat % len(ordered)
    return ordered[offset:] + ordered[:offset]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--screen-repeats", type=int, default=2)
    parser.add_argument("--final-repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--maximum-spread", type=float, default=1.08)
    parser.add_argument("--minimum-speedup", type=float, default=1.015)
    parser.add_argument("--max-seq-len", type=int, default=2304)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir")
    args = parser.parse_args(argv)
    if args.screen_repeats < 2 or args.final_repeats < 2 or args.warmups < 1:
        parser.error("screen/final repeats must be >=2 and warmups must be >=1")

    _configure_environment(args.model, -1)
    import torch
    if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name().upper():
        raise SystemExit("This frontier requires an NVIDIA L4; no model was loaded.")
    from megagemm.engine import InferenceEngine

    lm_kernel = importlib.import_module("megagemm.kernels.lm_head_argmax")
    engine = InferenceEngine(
        args.model,
        dtype=torch.bfloat16,
        device="cuda",
        max_batch_size=8,
        max_seq_len=args.max_seq_len,
        num_blocks=0,
        block_size=16,
        kv_alloc="auto",
        cache_dir=args.cache_dir,
    )
    model = engine.model
    if getattr(model.runtime_policy, "name", "") != "gemma4-e2b-l4":
        raise RuntimeError("loaded model did not resolve the E2B/L4 RuntimePolicy")
    production_mlp_state = _production_mlp_state(model)
    workloads = tuple(
        Workload(prompt, output)
        for prompt in (512, 2048)
        for output in (16, 128)
    )
    prompts = {
        prompt: matrix.build_prompts(engine.tokenizer, 8, prompt)[0]
        for prompt in (512, 2048)
    }
    print("Gemma 4 E2B/L4/B8 full-model compute frontier", flush=True)
    print("  fixed execution: one-step CUDA Graph / scheduler burst 16", flush=True)
    print("  workloads: P512/P2048 x O16/O128", flush=True)
    print(f"  screen cases: {len(SCREEN_CASES)}", flush=True)
    print("  model loads: 1", flush=True)
    print("  tokens: natural greedy; profiler: disabled", flush=True)

    schedulers: dict[tuple[str, str], Any] = {}
    references: dict[str, dict[str, Any]] = {}
    short_references: dict[int, dict[str, Any]] = {}
    setup_audits: dict[str, Any] = {}
    phase_samples: dict[str, list[dict[str, Any]]] = {
        "screen": [],
        "final": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(stage: str, **extra: Any) -> None:
        # Preserve completed work if a later experimental graph is rejected or
        # the transient Colab runtime ends.  The final write replaces this
        # progress document with the complete decision artifact.
        progress = {
            "status": "running",
            "stage": stage,
            "model_loads": 1,
            "setup_audits": setup_audits,
            "screen_samples": phase_samples["screen"],
            "final_samples": phase_samples["final"],
            **extra,
        }
        args.output.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    def run_case(case: ComputeCase, workload: Workload) -> dict[str, Any]:
        _apply_case(model, lm_kernel, case, workload, production_mlp_state)
        key = (case.name, workload.key)
        engine._last_scheduler = schedulers.get(key)
        row = _run(engine, prompts[workload.prompt_tokens], workload.output_tokens)
        schedulers[key] = engine._last_scheduler
        return row

    # Production references are captured once per exact workload and reused for
    # all natural-greedy correctness checks.
    for prompt_tokens in (512, 2048):
        short_references[prompt_tokens] = run_case(
            PRODUCTION, Workload(prompt_tokens, 1)
        )
    for workload in workloads:
        references[workload.key] = run_case(PRODUCTION, workload)
        effective = effective_max_prompt_tokens(references[workload.key], 8)
        if effective + workload.output_tokens > args.max_seq_len:
            raise RuntimeError(
                f"{workload.key}: effective prompt plus output exceeds max sequence "
                f"length ({effective}+{workload.output_tokens}>{args.max_seq_len})"
            )
    checkpoint("references_complete")

    # Compile/capture outside the measurement region and prove that each case
    # actually selected its requested route.
    for case in SCREEN_CASES:
        for workload in workloads:
            short_errors: list[str] = []
            if workload.output_tokens == 16:
                short_workload = Workload(workload.prompt_tokens, 1)
                run_case(case, short_workload)
                short_row = run_case(case, short_workload)
                short_errors = _validate_measurement(
                    short_row,
                    short_references[workload.prompt_tokens],
                    short_workload,
                )
            before = _counter_snapshot(model)
            # First call owns compilation/capture; the following calls prove
            # steady reuse.  Neither is admitted into measured samples.
            run_case(case, workload)
            cold_runtime = model.decode_runtime_stats()
            row = None
            for _ in range(args.warmups):
                row = run_case(case, workload)
            after = _counter_snapshot(model)
            assert row is not None
            errors = short_errors + _validate_measurement(
                row, references[workload.key], workload
            )
            delta = _counter_delta(before, after)
            errors.extend(_route_errors(case, delta))
            errors.extend(_attention_route_errors(case, workload, cold_runtime))
            setup_audits[f"{case.name}/{workload.key}"] = {
                "errors": errors,
                "counter_delta": delta,
                "scheduler_stats": row["scheduler_stats"],
                "capture_runtime_stats": cold_runtime,
                "steady_runtime_stats": model.decode_runtime_stats(),
            }
            checkpoint("screen_setup", current=f"{case.name}/{workload.key}")
            if errors:
                print(
                    f"SETUP REJECT {case.name}/{workload.key}: {errors}",
                    flush=True,
                )

    def measure(
        cases: tuple[ComputeCase, ...],
        repeats: int,
        phase: str,
    ) -> list[dict[str, Any]]:
        rows = phase_samples[phase]
        for repeat in range(repeats):
            for workload in workloads:
                for case in _rotated(cases, repeat):
                    audit_key = (
                        f"final/{case.name}/{workload.key}"
                        if phase == "final"
                        else f"{case.name}/{workload.key}"
                    )
                    audit = setup_audits.get(audit_key) or {}
                    if audit.get("errors"):
                        continue
                    short_workload = Workload(workload.prompt_tokens, 1)
                    short = run_case(case, short_workload)
                    long = run_case(case, workload)
                    errors = _validate_measurement(
                        short,
                        short_references[workload.prompt_tokens],
                        short_workload,
                    )
                    errors += _validate_measurement(
                        long, references[workload.key], workload
                    )
                    decode_s = float(long["elapsed_s"]) - float(short["elapsed_s"])
                    if decode_s <= 0.0 or not math.isfinite(decode_s):
                        errors.append("paired incremental decode time is not positive")
                    decode_tokens = 8 * (workload.output_tokens - 1)
                    result = {
                        "phase": phase,
                        "repeat": repeat + 1,
                        "case": case.name,
                        "family": case.family,
                        "workload": workload.key,
                        "decode_tps": decode_tokens / decode_s if decode_s > 0.0 else 0.0,
                        "output_tps": float(long["output_tps"]),
                        "short": short,
                        "long": long,
                        "errors": errors,
                    }
                    rows.append(result)
                    checkpoint(
                        f"{phase}_measurement",
                        current=f"{case.name}/{workload.key}/{repeat + 1}",
                    )
                    print(
                        f"{phase} {case.name} {workload.key} {repeat + 1}/{repeats}: "
                        f"decode={result['decode_tps']:.2f} total={result['output_tps']:.2f} "
                        f"errors={errors}",
                        flush=True,
                    )
        return rows

    screen_samples = measure(SCREEN_CASES, args.screen_repeats, "screen")
    screen = summarize_screen(
        screen_samples,
        SCREEN_CASES,
        workloads,
        repeats=args.screen_repeats,
        maximum_spread=args.maximum_spread,
    )
    combined = combine_family_winners(SCREEN_CASES, screen["family_winners"])
    print(
        "SCREEN WINNERS " + json.dumps(screen["family_winners"], sort_keys=True),
        flush=True,
    )
    print("COMBINATION " + json.dumps(asdict(combined), sort_keys=True), flush=True)

    # Capture the combination and a fresh production graph for a final paired
    # A/B.  They remain in the same process and on the same loaded model.
    final_cases = (PRODUCTION, combined)
    for case in final_cases:
        for workload in workloads:
            schedulers.pop((case.name, workload.key), None)
            schedulers.pop((case.name, Workload(workload.prompt_tokens, 1).key), None)
    for case in final_cases:
        for workload in workloads:
            short_errors: list[str] = []
            if workload.output_tokens == 16:
                short_workload = Workload(workload.prompt_tokens, 1)
                run_case(case, short_workload)
                short_row = run_case(case, short_workload)
                short_errors = _validate_measurement(
                    short_row,
                    short_references[workload.prompt_tokens],
                    short_workload,
                )
            before = _counter_snapshot(model)
            run_case(case, workload)
            cold_runtime = model.decode_runtime_stats()
            row = run_case(case, workload)
            after = _counter_snapshot(model)
            errors = short_errors + _validate_measurement(
                row, references[workload.key], workload
            )
            delta = _counter_delta(before, after)
            errors.extend(_route_errors(case, delta))
            errors.extend(_attention_route_errors(case, workload, cold_runtime))
            setup_audits[f"final/{case.name}/{workload.key}"] = {
                "errors": errors,
                "counter_delta": delta,
                "scheduler_stats": row["scheduler_stats"],
                "capture_runtime_stats": cold_runtime,
            }
            checkpoint(
                "final_setup",
                current=f"{case.name}/{workload.key}",
                screen=screen,
                combined_case=asdict(combined),
            )
    final_samples = measure(final_cases, args.final_repeats, "final")
    final_summary = summarize_screen(
        final_samples,
        final_cases,
        workloads,
        repeats=args.final_repeats,
        maximum_spread=args.maximum_spread,
    )
    decision = final_decision(
        final_summary,
        minimum_speedup=args.minimum_speedup,
        maximum_spread=args.maximum_spread,
        policy_changed=changes_compute_policy(combined),
    )
    excluded = {
        "cublaslt_gateup": (
            "Prior exact L4/B8 BF16 sweep found torch.mm at 30.976 us versus "
            "cuBLASLt algo 0 at 31.099 us for N=12288, and 305.843 versus "
            "305.787 us for N=24576; there is no two-shape win to retest."
        ),
        "lm_head_bn64_prior": (
            "Retained as a graph-era control despite the prior seven-pair "
            "full-model result of only 1.00039x median paired speedup."
        ),
    }
    payload = {
        "status": "passed" if decision["valid"] else "failed",
        "method": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "model": args.model,
            "model_loads": 1,
            "batch_size": 8,
            "dtype": "bf16",
            "execution": "default E2B/L4 B8 RuntimePolicy graph burst16",
            "natural_greedy_tokens": True,
            "profiler_in_timing": False,
            "workloads": [asdict(workload) for workload in workloads],
        },
        "screen": screen,
        "screen_samples": screen_samples,
        "combined_case": asdict(combined),
        "final": final_summary,
        "final_samples": final_samples,
        "decision": decision,
        "setup_audits": setup_audits,
        "excluded_by_existing_evidence": excluded,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("DECISION " + json.dumps(decision, sort_keys=True), flush=True)
    print(f"Wrote: {args.output}", flush=True)
    return 0 if decision["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
