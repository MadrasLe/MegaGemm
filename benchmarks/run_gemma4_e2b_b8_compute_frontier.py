#!/usr/bin/env python3
"""One-load, full-model Gemma 4 E2B/L4/B8 compute frontier.

The screen keeps the promoted one-step CUDA Graph with a 16-token scheduler
burst fixed and changes only one compute family at a time.  Screen workloads
are selectable; the final paired production-versus-combination phase always
validates P512/P2048 and O16/O128 with natural greedy tokens, without loading
the model again.

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
    lm_block_n: int = 64
    lm_block_k: int = 128
    lm_warps: int = 4
    lm_stages: int = 2
    lm_triton_reduce: bool = True
    mlp_mode: str = "production"
    mlp_core_mode: str = "production"
    mlp_activation_block_size: int = 512
    mlp_tc_block_n: int = 64
    mlp_tc_block_k: int = 32
    mlp_tc_warps: int = 4
    mlp_tc_stages: int = 2
    ple_conditioned: bool = False
    ple_block_size: int = 256

    def h512_segments(self, prompt_tokens: int) -> int:
        return (
            self.h512_short_segments
            if int(prompt_tokens) <= 768
            else self.h512_long_segments
        )


PRODUCTION = ComputeCase("production", "production", lm_block_n=64)
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
    # LM-head frontier. BN64/BK128/W4/S2/TRITON_REDUCE=1 is production.
    ComputeCase("lm_head_bn32", "lm_head", lm_block_n=32),
    ComputeCase("lm_head_bn128", "lm_head", lm_block_n=128),
    ComputeCase("lm_head_bk64", "lm_head", lm_block_k=64),
    ComputeCase("lm_head_bk256", "lm_head", lm_block_k=256),
    ComputeCase("lm_head_w2", "lm_head", lm_warps=2),
    ComputeCase("lm_head_w8", "lm_head", lm_warps=8),
    ComputeCase("lm_head_s1", "lm_head", lm_stages=1),
    ComputeCase("lm_head_s3", "lm_head", lm_stages=3),
    ComputeCase(
        "lm_head_torch_reduce",
        "lm_head",
        lm_triton_reduce=False,
    ),
    # Former default retained as a regression control.
    ComputeCase("lm_head_bn256", "lm_head", lm_block_n=256),
    ComputeCase("mlp_fused_gateup", "mlp", mlp_mode="fused_gateup"),
    ComputeCase("mlp_deepfusion_down", "mlp", mlp_mode="deepfusion"),
    ComputeCase("mlp_fused_both", "mlp", mlp_mode="fused_both"),
    # Large dense MLP chain: only the 20 exact I=12288 E2B layers change.
    ComputeCase(
        "mlp_gated_act_bs128",
        "mlp_core",
        mlp_core_mode="gated_activation",
        mlp_activation_block_size=128,
    ),
    ComputeCase(
        "mlp_gated_act_bs256",
        "mlp_core",
        mlp_core_mode="gated_activation",
        mlp_activation_block_size=256,
    ),
    ComputeCase(
        "mlp_gated_act_bs512",
        "mlp_core",
        mlp_core_mode="gated_activation",
        mlp_activation_block_size=512,
    ),
    ComputeCase(
        "mlp_gated_act_bs1024",
        "mlp_core",
        mlp_core_mode="gated_activation",
        mlp_activation_block_size=1024,
    ),
    ComputeCase(
        "mlp_tc_bn32_bk32_w4_s2",
        "mlp_core",
        mlp_core_mode="tensorcore_down",
        mlp_tc_block_n=32,
    ),
    ComputeCase(
        "mlp_tc_bn64_bk32_w4_s2",
        "mlp_core",
        mlp_core_mode="tensorcore_down",
    ),
    ComputeCase(
        "mlp_tc_bn128_bk32_w4_s2",
        "mlp_core",
        mlp_core_mode="tensorcore_down",
        mlp_tc_block_n=128,
    ),
    ComputeCase(
        "mlp_tc_bn64_bk64_w4_s2",
        "mlp_core",
        mlp_core_mode="tensorcore_down",
        mlp_tc_block_k=64,
    ),
    ComputeCase(
        "mlp_tc_bn64_bk32_w8_s2",
        "mlp_core",
        mlp_core_mode="tensorcore_down",
        mlp_tc_warps=8,
    ),
    ComputeCase(
        "mlp_tc_bn64_bk32_w4_s3",
        "mlp_core",
        mlp_core_mode="tensorcore_down",
        mlp_tc_stages=3,
    ),
    # PLE is a separate residual tail, but it belongs in the same full-model
    # optimization gate because it contributes to every E2B decode layer.
    ComputeCase(
        "ple_conditioned_bs128",
        "ple",
        ple_conditioned=True,
        ple_block_size=128,
    ),
    ComputeCase(
        "ple_conditioned_bs256",
        "ple",
        ple_conditioned=True,
        ple_block_size=256,
    ),
    ComputeCase(
        "ple_conditioned_bs512",
        "ple",
        ple_conditioned=True,
        ple_block_size=512,
    ),
    ComputeCase(
        "ple_conditioned_bs1024",
        "ple",
        ple_conditioned=True,
        ple_block_size=1024,
    ),
)

FINAL_WORKLOADS = tuple(
    Workload(prompt, output)
    for prompt in (512, 2048)
    for output in (16, 128)
)


def select_screen_cases(names: str | None) -> tuple[ComputeCase, ...]:
    if not names:
        return SCREEN_CASES
    by_name = {case.name: case for case in SCREEN_CASES}
    requested = [item.strip() for item in names.split(",") if item.strip()]
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError("unknown screen case(s): " + ", ".join(unknown))
    ordered_names = [PRODUCTION.name, *requested]
    selected: list[ComputeCase] = []
    seen: set[str] = set()
    for name in ordered_names:
        if name not in seen:
            selected.append(by_name[name])
            seen.add(name)
    return tuple(selected)


def select_screen_workloads(
    prompt_tokens: str,
    output_tokens: str,
) -> tuple[Workload, ...]:
    def parse(raw: str, label: str) -> set[int]:
        try:
            values = {int(item.strip()) for item in raw.split(",") if item.strip()}
        except ValueError as exc:
            raise ValueError(f"invalid {label}: {raw}") from exc
        if not values:
            raise ValueError(f"{label} cannot be empty")
        return values

    prompts = parse(prompt_tokens, "screen prompt tokens")
    outputs = parse(output_tokens, "screen output tokens")
    supported_prompts = {workload.prompt_tokens for workload in FINAL_WORKLOADS}
    supported_outputs = {workload.output_tokens for workload in FINAL_WORKLOADS}
    bad_prompts = sorted(prompts - supported_prompts)
    bad_outputs = sorted(outputs - supported_outputs)
    if bad_prompts:
        raise ValueError(f"unsupported screen prompt token(s): {bad_prompts}")
    if bad_outputs:
        raise ValueError(f"unsupported screen output token(s): {bad_outputs}")
    return tuple(
        workload
        for workload in FINAL_WORKLOADS
        if workload.prompt_tokens in prompts and workload.output_tokens in outputs
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
    "gemma4_e2b_b8_gated_activation_hits",
    "gemma4_e2b_b8_tensorcore_down_hits",
    "gemma4_ple_conditioned_gelu_decode_hits",
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


def _prepare_production_mlp_state(model: Any) -> dict[str, Any]:
    """Materialize lazy flat-decode policy before recording the baseline."""
    model._prepare_flat_decode()
    if not bool(getattr(model, "_flat_decode_ready", False)):
        raise RuntimeError(
            "flat decode is not ready: "
            + str(getattr(model, "_flat_decode_failed_reason", "unknown"))
        )
    return _production_mlp_state(model)


def _restore_mlp_state(model: Any, state: dict[str, Any]) -> None:
    for name, value in state.items():
        setattr(model, name, value)
    for name in (
        "_gemma4_flat_fused_gateup_use_cache",
        "_gemma4_flat_deepfusion_use_cache",
    ):
        cache = getattr(model, name, None)
        if cache is not None:
            cache.clear()
    model._gemma4_flat_fused_gateup_runtime_disabled = False
    model._gemma4_flat_cublaslt_gateup_runtime_disabled = False
    model._gemma4_flat_cublaslt_gateup_failure = ""
    model._gemma4_flat_b8_gated_activation_enabled = False
    model._gemma4_flat_b8_gated_activation_runtime_disabled = False
    model._gemma4_flat_b8_gated_activation_failure = ""
    model._gemma4_flat_b8_tensorcore_down_enabled = False
    model._gemma4_flat_b8_tensorcore_down_runtime_disabled = False
    model._gemma4_flat_b8_tensorcore_down_failure = ""
    model._gemma4_flat_ple_conditioned_gelu_enabled = False
    model._gemma4_flat_ple_conditioned_gelu_runtime_disabled = False
    model._gemma4_flat_ple_conditioned_gelu_first_failure = ""


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
    lm_kernel._CFG_TRITON_REDUCE = bool(case.lm_triton_reduce)

    _restore_mlp_state(model, production_mlp_state)
    if case.mlp_mode in ("fused_gateup", "fused_both"):
        model._gemma4_flat_policy_cublas_gateup_rows = ()
        model._gemma4_flat_policy_fused_gateup_rows = (8,)
    if case.mlp_mode in ("deepfusion", "fused_both"):
        model._gemma4_flat_policy_cublas_down_rows = ()
        model._gemma4_flat_policy_deepfusion_rows = (8,)
    if case.mlp_core_mode == "gated_activation":
        model._gemma4_flat_b8_gated_activation_enabled = True
        model._gemma4_flat_b8_gated_activation_block_size = int(
            case.mlp_activation_block_size
        )
    elif case.mlp_core_mode == "tensorcore_down":
        model._gemma4_flat_b8_tensorcore_down_enabled = True
        model._gemma4_flat_b8_tensorcore_down_config = (
            int(case.mlp_tc_block_n),
            int(case.mlp_tc_block_k),
            int(case.mlp_tc_warps),
            int(case.mlp_tc_stages),
        )
    model._gemma4_flat_ple_conditioned_gelu_enabled = bool(case.ple_conditioned)
    model._gemma4_flat_ple_conditioned_gelu_block_size = int(case.ple_block_size)


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
    token_comparison = compare_generated_tokens(row, reference)
    if not token_comparison["exact_match"]:
        errors.append(
            "natural greedy tokens differ from production: "
            f"agreement={token_comparison['token_agreement']:.6f} "
            f"first={token_comparison['first_divergence']}"
        )
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


def _route_errors(
    case: ComputeCase,
    delta: dict[str, int],
    runtime_stats: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if case.mlp_mode in ("fused_gateup", "fused_both") and not delta[
        "gemma4_flat_fused_gateup_hits"
    ]:
        errors.append("requested fused gate-up route produced no hits")
    if case.mlp_mode in ("deepfusion", "fused_both") and not delta[
        "gemma4_flat_deepfusion_hits"
    ]:
        errors.append("requested deepfusion down route produced no hits")
    if case.mlp_core_mode == "gated_activation":
        if not delta["gemma4_e2b_b8_gated_activation_hits"]:
            errors.append("requested fused GeGLU activation route produced no hits")
        if runtime_stats.get("gemma4_e2b_b8_gated_activation_runtime_disabled"):
            errors.append(
                "fused GeGLU activation disabled at runtime: "
                + str(runtime_stats.get("gemma4_e2b_b8_gated_activation_failure"))
            )
        if int(
            runtime_stats.get("gemma4_e2b_b8_gated_activation_block_size") or 0
        ) != int(case.mlp_activation_block_size):
            errors.append("fused GeGLU activation selected the wrong block size")
    if case.mlp_core_mode == "tensorcore_down":
        if not delta["gemma4_e2b_b8_tensorcore_down_hits"]:
            errors.append("requested Tensor Core GeGLU+down route produced no hits")
        if runtime_stats.get("gemma4_e2b_b8_tensorcore_down_runtime_disabled"):
            errors.append(
                "Tensor Core GeGLU+down disabled at runtime: "
                + str(runtime_stats.get("gemma4_e2b_b8_tensorcore_down_failure"))
            )
        expected_config = [
            int(case.mlp_tc_block_n),
            int(case.mlp_tc_block_k),
            int(case.mlp_tc_warps),
            int(case.mlp_tc_stages),
        ]
        if runtime_stats.get("gemma4_e2b_b8_tensorcore_down_config") != expected_config:
            errors.append("Tensor Core GeGLU+down selected the wrong launch config")
    if case.ple_conditioned:
        if not delta["gemma4_ple_conditioned_gelu_decode_hits"]:
            errors.append("requested fused PLE conditioned GELU route produced no hits")
        if runtime_stats.get("gemma4_ple_conditioned_gelu_runtime_disabled"):
            errors.append(
                "fused PLE conditioned GELU disabled at runtime: "
                + str(runtime_stats.get("gemma4_ple_conditioned_gelu_first_failure"))
            )
        if int(runtime_stats.get("gemma4_ple_conditioned_gelu_block_size") or 0) != int(
            case.ple_block_size
        ):
            errors.append("fused PLE conditioned GELU selected the wrong block size")
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


def _lm_route_errors(case: ComputeCase, lm_kernel: Any) -> list[str]:
    runtime = lm_kernel.lm_head_argmax_runtime_config()
    expected = {
        "forced_block_n": int(case.lm_block_n),
        "forced_block_k": int(case.lm_block_k),
        "forced_num_warps": int(case.lm_warps),
        "forced_num_stages": int(case.lm_stages),
        "triton_reduce": bool(case.lm_triton_reduce),
    }
    errors: list[str] = []
    for key, value in expected.items():
        if runtime.get(key) != value:
            errors.append(
                f"LM-head route {key}={runtime.get(key)!r}, expected {value!r}"
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
    for family in ("attention", "lm_head", "mlp", "mlp_core", "ple"):
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
    mlp_core = by_name.get(winners.get("mlp_core", "production"), PRODUCTION)
    ple = by_name.get(winners.get("ple", "production"), PRODUCTION)
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
        lm_triton_reduce=lm_head.lm_triton_reduce,
        mlp_mode=mlp.mlp_mode,
        mlp_core_mode=mlp_core.mlp_core_mode,
        mlp_activation_block_size=mlp_core.mlp_activation_block_size,
        mlp_tc_block_n=mlp_core.mlp_tc_block_n,
        mlp_tc_block_k=mlp_core.mlp_tc_block_k,
        mlp_tc_warps=mlp_core.mlp_tc_warps,
        mlp_tc_stages=mlp_core.mlp_tc_stages,
        ple_conditioned=ple.ple_conditioned,
        ple_block_size=ple.ple_block_size,
    )
    return combined


def final_decision(
    summary: dict[str, Any],
    *,
    minimum_speedup: float,
    minimum_output_speedup: float = 1.0,
    maximum_spread: float,
    policy_changed: bool = True,
) -> dict[str, Any]:
    production = summary["cases"].get("production") or {}
    if not policy_changed:
        valid = bool(production.get("valid"))
        return {
            "decision": "KEEP_PRODUCTION",
            "apply_change": False,
            "valid": valid,
            "geomean_decode_speedup": 1.0,
            "geomean_output_speedup": 1.0,
            "worst_decode_speedup": 1.0,
            "worst_output_speedup": 1.0,
            "minimum_speedup": minimum_speedup,
            "minimum_decode_speedup": minimum_speedup,
            "minimum_output_speedup": minimum_output_speedup,
            "policy_changed": False,
        }
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
        and output_speedup >= minimum_output_speedup
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
        "minimum_decode_speedup": minimum_speedup,
        "minimum_output_speedup": minimum_output_speedup,
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


def _discard_case_schedulers(
    schedulers: dict[tuple[str, str], Any],
    case: ComputeCase,
    workloads: tuple[Workload, ...],
) -> None:
    for prompt_tokens in {workload.prompt_tokens for workload in workloads}:
        schedulers.pop((case.name, Workload(prompt_tokens, 1).key), None)
    for workload in workloads:
        schedulers.pop((case.name, workload.key), None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--screen-repeats", type=int, default=2)
    parser.add_argument("--final-repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--maximum-spread", type=float, default=1.08)
    parser.add_argument("--minimum-speedup", type=float, default=1.015)
    parser.add_argument("--minimum-output-speedup", type=float, default=1.0)
    parser.add_argument(
        "--screen-case-names",
        help=(
            "comma-separated subset of screen cases; production is always "
            "included"
        ),
    )
    parser.add_argument(
        "--screen-prompt-tokens",
        default="512,2048",
        help="screen-only prompt lengths; final validation always uses 512,2048",
    )
    parser.add_argument(
        "--screen-output-tokens",
        default="16,128",
        help="screen-only output lengths; final validation always uses 16,128",
    )
    parser.add_argument("--max-seq-len", type=int, default=2304)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir")
    args = parser.parse_args(argv)
    if args.screen_repeats < 2 or args.final_repeats < 2 or args.warmups < 1:
        parser.error("screen/final repeats must be >=2 and warmups must be >=1")
    try:
        screen_cases = select_screen_cases(args.screen_case_names)
        screen_workloads = select_screen_workloads(
            args.screen_prompt_tokens,
            args.screen_output_tokens,
        )
    except ValueError as exc:
        parser.error(str(exc))

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
    # Gemma4 creates the decision caches and resolves RuntimePolicy row sets
    # lazily in _prepare_flat_decode().  Capture only after that initialization;
    # the constructor's empty tuples are not the production policy.
    production_mlp_state = _prepare_production_mlp_state(model)
    final_workloads = FINAL_WORKLOADS
    prompts = {
        prompt: matrix.build_prompts(engine.tokenizer, 8, prompt)[0]
        for prompt in (512, 2048)
    }
    print("Gemma 4 E2B/L4/B8 full-model compute frontier", flush=True)
    print("  fixed execution: one-step CUDA Graph / scheduler burst 16", flush=True)
    print(
        "  screen workloads: "
        + ", ".join(workload.key for workload in screen_workloads),
        flush=True,
    )
    print("  final workloads: P512/P2048 x O16/O128", flush=True)
    print(f"  screen cases: {len(screen_cases)}", flush=True)
    print(
        "  selected: " + ", ".join(case.name for case in screen_cases),
        flush=True,
    )
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
    for workload in final_workloads:
        references[workload.key] = run_case(PRODUCTION, workload)
        effective = effective_max_prompt_tokens(references[workload.key], 8)
        if effective + workload.output_tokens > args.max_seq_len:
            raise RuntimeError(
                f"{workload.key}: effective prompt plus output exceeds max sequence "
                f"length ({effective}+{workload.output_tokens}>{args.max_seq_len})"
            )
    checkpoint("references_complete")

    def setup_cases(
        cases: tuple[ComputeCase, ...],
        phase_workloads: tuple[Workload, ...],
        *,
        audit_prefix: str,
        stage: str,
        checkpoint_extra: dict[str, Any] | None = None,
    ) -> None:
        """Compile, capture, warm and audit without admitting setup into timing."""
        extra = checkpoint_extra or {}
        for case in cases:
            # Reference and previous-phase schedulers may own graphs for the
            # same logical case. A phase gets fresh, explicitly audited owners.
            _discard_case_schedulers(schedulers, case, phase_workloads)
            short_errors_by_prompt: dict[int, list[str]] = {}
            for prompt_tokens in sorted(
                {workload.prompt_tokens for workload in phase_workloads}
            ):
                short_workload = Workload(prompt_tokens, 1)
                run_case(case, short_workload)
                short_row = None
                for _ in range(args.warmups):
                    short_row = run_case(case, short_workload)
                assert short_row is not None
                short_errors_by_prompt[prompt_tokens] = _validate_measurement(
                    short_row,
                    short_references[prompt_tokens],
                    short_workload,
                )

            for workload in phase_workloads:
                before = _counter_snapshot(model)
                # First call owns compilation/capture; following calls prove
                # steady reuse. Neither is admitted into measured samples.
                run_case(case, workload)
                cold_runtime = model.decode_runtime_stats()
                row = None
                for _ in range(args.warmups):
                    row = run_case(case, workload)
                after = _counter_snapshot(model)
                steady_runtime = model.decode_runtime_stats()
                assert row is not None
                errors = list(short_errors_by_prompt[workload.prompt_tokens])
                errors.extend(
                    _validate_measurement(row, references[workload.key], workload)
                )
                delta = _counter_delta(before, after)
                errors.extend(_route_errors(case, delta, steady_runtime))
                errors.extend(
                    _attention_route_errors(case, workload, cold_runtime)
                )
                errors.extend(_lm_route_errors(case, lm_kernel))
                lm_runtime = lm_kernel.lm_head_argmax_runtime_config()
                audit_key = f"{audit_prefix}{case.name}/{workload.key}"
                setup_audits[audit_key] = {
                    "errors": errors,
                    "counter_delta": delta,
                    "lm_head_runtime_config": lm_runtime,
                    "scheduler_stats": row["scheduler_stats"],
                    "capture_runtime_stats": cold_runtime,
                    "steady_runtime_stats": steady_runtime,
                }
                checkpoint(
                    stage,
                    current=f"{case.name}/{workload.key}",
                    **extra,
                )
                if errors:
                    print(
                        f"SETUP REJECT {case.name}/{workload.key}: {errors}",
                        flush=True,
                    )

    # Compile/capture outside the measurement region and prove that every case
    # selected the requested attention, LM-head and MLP routes.
    setup_cases(
        screen_cases,
        screen_workloads,
        audit_prefix="",
        stage="screen_setup",
    )

    def measure(
        cases: tuple[ComputeCase, ...],
        repeats: int,
        phase: str,
        phase_workloads: tuple[Workload, ...],
    ) -> list[dict[str, Any]]:
        rows = phase_samples[phase]
        for repeat in range(repeats):
            for workload in phase_workloads:
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

    screen_samples = measure(
        screen_cases,
        args.screen_repeats,
        "screen",
        screen_workloads,
    )
    screen = summarize_screen(
        screen_samples,
        screen_cases,
        screen_workloads,
        repeats=args.screen_repeats,
        maximum_spread=args.maximum_spread,
    )
    combined = combine_family_winners(screen_cases, screen["family_winners"])
    print(
        "SCREEN WINNERS " + json.dumps(screen["family_winners"], sort_keys=True),
        flush=True,
    )
    print("COMBINATION " + json.dumps(asdict(combined), sort_keys=True), flush=True)

    # Capture the combination and a fresh production graph for a final paired
    # A/B.  They remain in the same process and on the same loaded model.
    policy_changed = changes_compute_policy(combined)
    # If every family lost the screen, a second copy of production cannot add
    # evidence.  Measure production once in the final phase and exit cleanly.
    final_cases = (PRODUCTION, combined) if policy_changed else (PRODUCTION,)
    setup_cases(
        final_cases,
        final_workloads,
        audit_prefix="final/",
        stage="final_setup",
        checkpoint_extra={
            "screen": screen,
            "combined_case": asdict(combined),
        },
    )
    final_samples = measure(
        final_cases,
        args.final_repeats,
        "final",
        final_workloads,
    )
    final_summary = summarize_screen(
        final_samples,
        final_cases,
        final_workloads,
        repeats=args.final_repeats,
        maximum_spread=args.maximum_spread,
    )
    decision = final_decision(
        final_summary,
        minimum_speedup=args.minimum_speedup,
        minimum_output_speedup=args.minimum_output_speedup,
        maximum_spread=args.maximum_spread,
        policy_changed=policy_changed,
    )
    excluded = {
        "cublaslt_gateup": (
            "Prior exact L4/B8 BF16 sweep found torch.mm at 30.976 us versus "
            "cuBLASLt algo 0 at 31.099 us for N=12288, and 305.843 versus "
            "305.787 us for N=24576; there is no two-shape win to retest."
        ),
        "lm_head_bn256_previous_default": (
            "Retained as a regression control after the exact B8 graph-era "
            "gate promoted BN64 for the M8/K1536/N262144 BF16 L4 shape."
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
            "screen_workloads": [
                asdict(workload) for workload in screen_workloads
            ],
            "final_workloads": [
                asdict(workload) for workload in final_workloads
            ],
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
