#!/usr/bin/env python3
"""One-load full-model decode frontier gate for Gemma 4 E2B on one L4.

The gate measures three independent B1/P2048/O1-O128 experiment families:

* production, dense attention-to-MLP bridge, forced fused RMSNorm LM head,
  and their combination;
* the exact B1/Q8/KV1/H512 full-attention grouped/segmented frontier;
* the exact 20-layer 1536->24576 gate-up and 12288->1536 down frontier.

Every case runs on the same loaded model.  The forced-token route keeps the
autoregressive input identical while the real LM head and argmax still run.
The primary metric is paired incremental decode throughput: O128 wall time
minus the matching O1 wall time in the same repeat.

--suite h512-followup keeps the promoted B1 bridge enabled in every case and
tests only TILE64 launch configurations against current production.
--suite mlp-gemv keeps the bridge, LM head and attention fixed, comparing
projection-specific GEMV tiles and split-K reductions in eleven full-model cases.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
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


@dataclass(frozen=True)
class DecodeCase:
    family: str
    name: str
    bridge: bool = False
    force_fused_lm_head: bool = False
    attention_segments: int = 0
    attention_tile: int = 0
    attention_warps: int = 4
    attention_stages: int = 3
    attention_reduce_warps: int = 4
    large_gateup: bool = False
    large_down: bool = False
    gemv_gateup: str = ""
    gemv_down: str = ""
    prepared_gemv: bool = False


CORE_CASES = (
    DecodeCase("core_factorial", "production"),
    DecodeCase("core_factorial", "bridge_b1", bridge=True),
    DecodeCase(
        "core_factorial",
        "forced_fused_rms_lm_head",
        force_fused_lm_head=True,
    ),
    DecodeCase(
        "core_factorial",
        "bridge_plus_fused_rms_lm_head",
        bridge=True,
        force_fused_lm_head=True,
    ),
)

ATTENTION_CASES = (
    DecodeCase("full_attention_h512", "production"),
    *(
        DecodeCase(
            "full_attention_h512",
            (
                f"grouped_seg{segments}_tile{tile}_w{warps}_"
                f"s{stages}_r{reduce_warps}"
            ),
            attention_segments=segments,
            attention_tile=tile,
            attention_warps=warps,
            attention_stages=stages,
            attention_reduce_warps=reduce_warps,
        )
        for segments, tile, warps, stages, reduce_warps in (
            (4, 32, 4, 2, 4),
            (8, 16, 2, 2, 2),
            (8, 16, 4, 3, 4),
            (8, 32, 2, 2, 2),
            (8, 32, 4, 3, 4),
            (8, 64, 4, 3, 4),
            (16, 16, 4, 3, 4),
            (16, 32, 4, 3, 4),
            (32, 16, 4, 3, 4),
        )
    ),
)

MLP_CASES = (
    DecodeCase("large_mlp", "production"),
    DecodeCase("large_mlp", "fused_large_gateup", large_gateup=True),
    DecodeCase("large_mlp", "deepfusion_large_down", large_down=True),
    DecodeCase(
        "large_mlp",
        "fused_large_gateup_plus_down",
        large_gateup=True,
        large_down=True,
    ),
)

FAMILIES = (
    ("core_factorial", CORE_CASES),
    ("full_attention_h512", ATTENTION_CASES),
    ("large_mlp", MLP_CASES),
)

H512_FOLLOWUP_CASES = (
    DecodeCase("h512_followup", "production", bridge=True),
    *(
        DecodeCase(
            "h512_followup",
            f"grouped_seg8_tile64_w{warps}_s{stages}_r4",
            bridge=True,
            attention_segments=8,
            attention_tile=64,
            attention_warps=warps,
            attention_stages=stages,
        )
        for warps, stages in ((4, 1), (4, 2), (4, 3), (8, 2))
    ),
)

COUNTER_KEYS = (
    "gemma4_dense_attn_mlp_bridge_decode_hits",
    "fused_rmsnorm_lm_head_argmax_hits",
    "gemma4_flat_fused_gateup_hits",
    "gemma4_flat_deepfusion_hits",
    "gemma4_e2b_b1_large_gateup_hits",
    "gemma4_e2b_b1_large_down_hits",
    "gemma4_b1_mlp_gemv_gateup_hits",
    "gemma4_b1_mlp_gemv_down_hits",
    "gemma4_b1_mlp_prepared_gateup_hits",
    "gemma4_b1_mlp_prepared_down_hits",
)
PAGED_COUNTER_KEYS = (
    "gqa2_direct_hits",
    "generic_direct_hits",
    "grouped_segmented_hits",
)


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _geomean(values: list[float]) -> float:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not positive:
        return 0.0
    return float(math.exp(sum(math.log(value) for value in positive) / len(positive)))


def _configure_environment(model: str, forced_token_id: int) -> dict[str, str]:
    from benchmarks.run_publication_gpu_suite import profile_environment

    for key in tuple(os.environ):
        if key.startswith("MEGAGEMM_"):
            os.environ.pop(key, None)
    profile = profile_environment("gemma4-e2b-fast", model)
    profile.update(
        {
            "MEGAGEMM_BENCHMARK_TOKEN_DIGEST": "1",
            "MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID": str(forced_token_id),
            # Allocates the B1 bridge buffer but does not make it production.
            "MEGAGEMM_GEMMA4_E2B_L4_B1_DECODE_FRONTIER_EXPERIMENT": "1",
            "MEGAGEMM_GEMMA4_E2B_L4_B1_H512_GROUPED_ATTN_DECODE": "0",
            "MEGAGEMM_GEMMA4_E2B_L4_B1_LARGE_GATEUP_EXPERIMENT": "0",
            "MEGAGEMM_GEMMA4_E2B_L4_B1_LARGE_DOWN_EXPERIMENT": "0",
        }
    )
    os.environ.update(profile)
    return profile


def _generated_token_rows(scheduler: Any) -> list[list[int]]:
    completed = sorted(
        getattr(scheduler, "_completed", ()) or (),
        key=lambda request: int(request.request_id),
    )
    return [
        [int(token_id) for token_id in request.generated_ids]
        for request in completed
    ]


def _token_digest(scheduler: Any) -> tuple[str, list[int]]:
    rows = _generated_token_rows(scheduler)
    encoded = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return hashlib.sha256(encoded).hexdigest(), [len(row) for row in rows]


def _run(engine: Any, prompts: list[str], max_new_tokens: int) -> dict[str, Any]:
    from benchmarks import benchmark_inference_matrix as matrix

    matrix.sync_cuda()
    start = time.perf_counter()
    engine.generate_batch(
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        ignore_eos=True,
        decode_outputs=False,
    )
    matrix.sync_cuda()
    elapsed_s = time.perf_counter() - start
    scheduler = getattr(engine, "_last_scheduler", None)
    if scheduler is None:
        raise RuntimeError("engine did not retain its scheduler")
    scheduler_stats = scheduler.get_stats()
    digest, lengths = _token_digest(scheduler)
    generated_ids = _generated_token_rows(scheduler)
    generated = int(scheduler_stats.get("total_tokens") or 0)
    prompt_lengths = [
        len(request.prompt_ids)
        for request in getattr(scheduler, "_completed", ())
    ]
    return {
        "elapsed_s": elapsed_s,
        "generated_tokens": generated,
        "output_tps": generated / elapsed_s,
        "scheduler_stats": scheduler_stats,
        "engine_prompt_tokens": sum(prompt_lengths),
        "engine_prompt_lengths": prompt_lengths,
        "engine_max_prompt_tokens": max(prompt_lengths, default=0),
        "digest": digest,
        "lengths": lengths,
        "generated_ids": generated_ids,
    }


def _runtime_stats(model: Any) -> dict[str, Any]:
    stats = model.decode_runtime_stats()
    if not isinstance(stats, dict):
        raise RuntimeError("model did not return decode runtime stats")
    return stats


def _counter_snapshot(model: Any) -> dict[str, int]:
    stats = _runtime_stats(model)
    paged = stats.get("paged_decode_runtime") or {}
    result = {key: int(stats.get(key) or 0) for key in COUNTER_KEYS}
    result.update(
        {f"paged_{key}": int(paged.get(key) or 0) for key in PAGED_COUNTER_KEYS}
    )
    return result


def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: int(after.get(key, 0) - before.get(key, 0)) for key in after}


def _capture_lm_state(model: Any) -> dict[str, Any]:
    names = (
        "_fused_rmsnorm_lm_head_argmax_key",
        "_fused_rmsnorm_lm_head_argmax_use",
        "_fused_rmsnorm_lm_head_argmax_checked",
        "_fused_rmsnorm_lm_head_argmax_disable",
        "_fused_rmsnorm_lm_head_argmax_error",
        "_fused_rmsnorm_lm_head_argmax_skip_reason",
        "_fused_lm_head_argmax_key",
        "_fused_lm_head_argmax_use",
        "_fused_lm_head_argmax_checked",
        "_fused_lm_head_argmax_disable",
        "_fused_lm_head_argmax_error",
        "_fused_lm_head_argmax_skip_reason",
    )
    return {name: getattr(model, name) for name in names}


def _restore_lm_state(model: Any, state: dict[str, Any]) -> None:
    for name, value in state.items():
        setattr(model, name, value)


def _force_fused_rms_lm_head(model: Any, production_state: dict[str, Any]) -> None:
    _restore_lm_state(model, production_state)
    model._fused_rmsnorm_lm_head_argmax_checked = True
    model._fused_rmsnorm_lm_head_argmax_use = True
    model._fused_rmsnorm_lm_head_argmax_disable = False
    model._fused_rmsnorm_lm_head_argmax_error = ""
    model._fused_lm_head_argmax_checked = True
    model._fused_lm_head_argmax_use = False
    model._fused_lm_head_argmax_disable = False
    model._fused_lm_head_argmax_error = ""


def _reset_attention_failure() -> None:
    from megagemm.kernels import paged_attention

    paged_attention._GROUPED_SEGMENTED_DECODE_DISABLED = False
    paged_attention._GROUPED_SEGMENTED_DECODE_FAILURE = ""
    paged_attention._GROUPED_SEGMENTED_DECODE_LOGGED = False
    paged_attention._GROUPED_SEGMENTED_DECODE_SELECTED_SEGMENTS.clear()
    paged_attention._GROUPED_SEGMENTED_DECODE_SELECTED_TILE_SIZES.clear()


def _apply_case(
    model: Any,
    case: DecodeCase,
    production_lm_state: dict[str, Any],
) -> None:
    model._gemma4_flat_dense_attn_mlp_bridge_enabled = bool(case.bridge)
    model._gemma4_flat_dense_attn_mlp_bridge_runtime_disabled = False
    model._gemma4_flat_dense_attn_mlp_bridge_failure = ""

    if case.force_fused_lm_head:
        _force_fused_rms_lm_head(model, production_lm_state)
    else:
        _restore_lm_state(model, production_lm_state)

    attention_enabled = case.attention_segments > 0
    os.environ["MEGAGEMM_GEMMA4_E2B_L4_B1_H512_GROUPED_ATTN_DECODE"] = (
        "1" if attention_enabled else "0"
    )
    if attention_enabled:
        os.environ["MEGAGEMM_GEMMA4_E2B_L4_B1_H512_ATTN_SEGMENTS"] = str(
            case.attention_segments
        )
        os.environ["MEGAGEMM_GEMMA4_E2B_L4_B1_H512_ATTN_TILE"] = str(
            case.attention_tile
        )
        os.environ["MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_WARPS"] = str(
            case.attention_warps
        )
        os.environ["MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_STAGES"] = str(
            case.attention_stages
        )
        os.environ[
            "MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_REDUCE_WARPS"
        ] = str(case.attention_reduce_warps)
    _reset_attention_failure()

    model._gemma4_flat_b1_large_gateup_experiment = bool(case.large_gateup)
    model._gemma4_flat_b1_large_down_experiment = bool(case.large_down)
    model._gemma4_flat_fused_gateup_runtime_disabled = False
    model._gemma4_b1_mlp_gemv_configs = {
        operation: config for operation, config in (
            ("gateup", case.gemv_gateup), ("down", case.gemv_down)
        ) if config
    }
    model._gemma4_b1_mlp_gemv_failures = {}
    model._gemma4_b1_mlp_prepared_routes = None
    model._gemma4_b1_mlp_gemv_dispatch = "prepared" if case.prepared_gemv else "legacy"
    if case.prepared_gemv:
        model._prepare_gemma4_b1_mlp_gemv_routes()


def _validate_lm_head(
    model: Any,
    production_lm_state: dict[str, Any],
) -> dict[str, Any]:
    import torch

    generator = torch.Generator(device=model.lm_head.weight.device)
    generator.manual_seed(9032026)
    hidden = torch.randn(
        (1, 1, 1536),
        generator=generator,
        device=model.lm_head.weight.device,
        dtype=torch.bfloat16,
    )
    with torch.inference_mode():
        _restore_lm_state(model, production_lm_state)
        production = model._decode_next_token_greedy(hidden).clone()
        _force_fused_rms_lm_head(model, production_lm_state)
        candidate = model._decode_next_token_greedy(hidden).clone()
        if hidden.is_cuda:
            torch.cuda.synchronize()
    result = {
        "production_token": int(production.item()),
        "forced_fused_token": int(candidate.item()),
        "exact_token_match": bool(torch.equal(production, candidate)),
    }
    _restore_lm_state(model, production_lm_state)
    return result


def _tensor_error(candidate: Any, reference: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    candidate_fp32 = candidate.float()
    reference_fp32 = reference.float()
    delta = candidate_fp32 - reference_fp32
    reference_norm = float(torch.linalg.vector_norm(reference_fp32).item())
    delta_norm = float(torch.linalg.vector_norm(delta).item())
    return {
        "finite": bool(torch.isfinite(candidate).all().item()),
        "exact": bool(torch.equal(candidate, reference)),
        "max_abs_error": float(delta.abs().max().item()),
        "mean_abs_error": float(delta.abs().mean().item()),
        "relative_l2_error": delta_norm / max(reference_norm, 1.0e-12),
        "cosine": float(
            functional.cosine_similarity(
                candidate_fp32.flatten(),
                reference_fp32.flatten(),
                dim=0,
            ).item()
        ),
    }


def _validate_attention_cases(
    cases: tuple[DecodeCase, ...],
    *,
    context: int,
    table_blocks: int,
) -> dict[str, Any]:
    import torch
    from benchmarks import run_gemma4_grouped_segmented_attention_microbench as shared
    from megagemm.kernels.paged_attention import (
        _triton_paged_decode_grouped_segmented_fused,
    )

    shared.ROWS = 1
    topology = shared.Topology(
        name="e2b_l4_b1_full_h512_gqa8",
        q_heads=8,
        kv_heads=1,
        head_dim=512,
        layers=7,
        sliding_window=None,
        prior_vllm_us=0.0,
    )
    tensors = shared.make_inputs(
        topology,
        context=context,
        table_blocks=table_blocks,
        norm_eps=1.0e-6,
    )
    scale = 1.0 / math.sqrt(topology.head_dim)
    reference = shared.torch_reference(
        query=tensors["query_fp32"],
        kv_cache=tensors["kv_cache"],
        block_table=tensors["block_table"],
        context=context,
        scale=scale,
        sliding_window=None,
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        if case.attention_segments <= 0:
            continue
        os.environ["MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_WARPS"] = str(
            case.attention_warps
        )
        os.environ["MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_STAGES"] = str(
            case.attention_stages
        )
        os.environ[
            "MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_REDUCE_WARPS"
        ] = str(case.attention_reduce_warps)
        out = torch.empty_like(tensors["raw_query"])
        try:
            with torch.inference_mode():
                _triton_paged_decode_grouped_segmented_fused(
                    tensors["raw_query"],
                    tensors["kv_cache"],
                    tensors["block_table"],
                    tensors["seq_lens"],
                    scale,
                    tensors["cos"],
                    tensors["sin"],
                    tensors["positions"],
                    half_rotate=True,
                    rotary_dim=topology.head_dim,
                    q_norm_weight=tensors["norm_weight"],
                    norm_eps=1.0e-6,
                    out=out,
                    force=True,
                    num_segments_override=case.attention_segments,
                    tile_size_override=case.attention_tile,
                )
                torch.cuda.synchronize()
                first = out.clone()
                _triton_paged_decode_grouped_segmented_fused(
                    tensors["raw_query"],
                    tensors["kv_cache"],
                    tensors["block_table"],
                    tensors["seq_lens"],
                    scale,
                    tensors["cos"],
                    tensors["sin"],
                    tensors["positions"],
                    half_rotate=True,
                    rotary_dim=topology.head_dim,
                    q_norm_weight=tensors["norm_weight"],
                    norm_eps=1.0e-6,
                    out=out,
                    force=True,
                    num_segments_override=case.attention_segments,
                    tile_size_override=case.attention_tile,
                )
                torch.cuda.synchronize()
                second = out.clone()
            metrics = _tensor_error(first, reference)
            metrics["repeat_exact"] = bool(torch.equal(first, second))
            metrics["error"] = None
            metrics["correct"] = bool(
                metrics["finite"]
                and metrics["repeat_exact"]
                and metrics["max_abs_error"] <= shared.MAX_ABS_ERROR
                and metrics["relative_l2_error"] <= shared.MAX_RELATIVE_L2_ERROR
                and metrics["cosine"] >= shared.MIN_COSINE
            )
        except Exception as exc:
            metrics = {
                "correct": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        row = {"case": case.name, "context": context, "config": asdict(case), **metrics}
        rows.append(row)
        print("ATTENTION PREFLIGHT " + json.dumps(row), flush=True)
    os.environ["MEGAGEMM_GEMMA4_E2B_L4_B1_H512_GROUPED_ATTN_DECODE"] = "0"
    return {
        "shape": {
            "batch_size": 1,
            "q_heads": 8,
            "kv_heads": 1,
            "head_dim": 512,
            "context": context,
            "block_size": 16,
            "dtype": "bf16",
        },
        "all_correct": bool(rows and all(row["correct"] for row in rows)),
        "cases": rows,
    }


def _validate_large_mlp(model: Any) -> dict[str, Any]:
    import torch
    from megagemm.models.llama import (
        deepfusion_swiglu_down,
        fused_rmsnorm_linear,
    )

    large = [
        (index, weights)
        for index, weights in enumerate(model._flat_layer_weights)
        if (
            not weights.is_moe
            and int(weights.intermediate_size) == 12288
            and weights.gate_up_weight is not None
            and weights.gate_up_wt is not None
            and weights.down_weight is not None
        )
    ]
    if len(large) != 20:
        return {
            "all_correct": False,
            "error": f"expected 20 large dense layers, found {len(large)}",
        }
    layer_index, weights = large[0]
    generator = torch.Generator(device=weights.gate_up_weight.device)
    generator.manual_seed(9042026)
    hidden = torch.randn(
        (1, 1536),
        generator=generator,
        device=weights.gate_up_weight.device,
        dtype=torch.bfloat16,
    )
    try:
        with torch.inference_mode():
            normalized = model._gemma4_flat_rmsnorm(
                hidden,
                weights.pre_ff_norm_weight,
                model._flat_norm_eps,
                False,
            )
            gate_reference = model._flat_fp_linear(
                normalized,
                weights.gate_up_wt,
                weights.gate_up_bias,
                model._gemma4_flat_gate_up_bufs[layer_index],
            ).clone()
            gate_candidate = fused_rmsnorm_linear(
                hidden,
                weights.pre_ff_norm_weight,
                model._flat_norm_eps,
                weights.gate_up_weight,
                weights.gate_up_bias,
                norm_offset=False,
                out=model._gemma4_flat_gate_up_bufs[layer_index],
                max_rows_override=1,
            ).clone()
            down_reference = model._gemma4_flat_baseline_down(
                gate_reference,
                weights,
                layer_index,
            ).clone()
            down_candidate = deepfusion_swiglu_down(
                gate_reference,
                weights.down_weight,
                weights.down_bias,
                out=model._gemma4_flat_down_bufs[layer_index],
                activation="gelu_tanh",
            ).clone()
            torch.cuda.synchronize()
        gate_metrics = _tensor_error(gate_candidate, gate_reference)
        down_metrics = _tensor_error(down_candidate, down_reference)
        gate_metrics["correct"] = bool(
            gate_metrics["finite"]
            and gate_metrics["max_abs_error"] <= 0.25
            and gate_metrics["cosine"] >= 0.9999
        )
        down_metrics["correct"] = bool(
            down_metrics["finite"]
            and down_metrics["relative_l2_error"] <= 0.02
            and down_metrics["cosine"] >= 0.999
        )
        return {
            "all_correct": bool(
                gate_metrics["correct"] and down_metrics["correct"]
            ),
            "large_layer_count": len(large),
            "representative_layer": layer_index,
            "gateup_1536x24576": gate_metrics,
            "down_12288x1536": down_metrics,
            "error": None,
        }
    except Exception as exc:
        return {
            "all_correct": False,
            "large_layer_count": len(large),
            "representative_layer": layer_index,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _audit_pair(
    case: DecodeCase,
    delta: dict[str, int],
    stats: dict[str, Any],
    *,
    decode_steps: int,
) -> list[str]:
    errors: list[str] = []
    for operation, config in (("gateup", case.gemv_gateup), ("down", case.gemv_down)):
        hits = delta.get(f"gemma4_b1_mlp_gemv_{operation}_hits", 0)
        expected = 20 * decode_steps if config else 0
        if hits != expected:
            errors.append(f"B1 MLP GEMV {operation} hits={hits}, expected {expected}")
        prepared_hits = delta.get(f"gemma4_b1_mlp_prepared_{operation}_hits", 0)
        expected_prepared = expected if case.prepared_gemv else 0
        if prepared_hits != expected_prepared:
            errors.append(f"prepared {operation} hits={prepared_hits}, expected {expected_prepared}")
        selected = (stats.get("gemma4_b1_mlp_gemv_configs") or {}).get(operation, "")
        if selected != config:
            errors.append(f"B1 MLP GEMV {operation} config={selected!r}, expected {config!r}")
    if stats.get("gemma4_b1_mlp_gemv_failures"):
        errors.append("B1 MLP GEMV fallback: " + str(stats["gemma4_b1_mlp_gemv_failures"]))
    expected_dispatch = "prepared" if case.prepared_gemv else "legacy"
    if stats.get("gemma4_b1_mlp_gemv_dispatch", "legacy") != expected_dispatch:
        errors.append("MLP GEMV dispatch does not match the requested case")
    bridge_hits = delta["gemma4_dense_attn_mlp_bridge_decode_hits"]
    lm_hits = delta["fused_rmsnorm_lm_head_argmax_hits"]
    grouped_hits = delta["paged_grouped_segmented_hits"]
    large_gateup_hits = delta["gemma4_e2b_b1_large_gateup_hits"]
    large_down_hits = delta["gemma4_e2b_b1_large_down_hits"]
    if case.bridge and bridge_hits != 35 * decode_steps:
        errors.append(
            f"B1 dense bridge hits={bridge_hits}, expected {35 * decode_steps}"
        )
    if not case.bridge and bridge_hits != 0:
        errors.append(f"production unexpectedly produced {bridge_hits} bridge hits")
    if case.force_fused_lm_head and lm_hits != decode_steps:
        errors.append(
            f"forced fused RMSNorm LM head hits={lm_hits}, expected {decode_steps}"
        )
    if case.attention_segments and grouped_hits != 7 * decode_steps:
        errors.append(
            f"B1 H512 grouped attention hits={grouped_hits}, "
            f"expected {7 * decode_steps}"
        )
    if not case.attention_segments and grouped_hits != 0:
        errors.append(
            f"production unexpectedly produced {grouped_hits} grouped attention hits"
        )
    if case.large_gateup and large_gateup_hits != 20 * decode_steps:
        errors.append(
            f"1536->24576 fused gate-up hits={large_gateup_hits}, "
            f"expected {20 * decode_steps}"
        )
    if not case.large_gateup and large_gateup_hits != 0:
        errors.append(f"production unexpectedly produced {large_gateup_hits} large gate-up hits")
    if case.large_down and large_down_hits != 20 * decode_steps:
        errors.append(
            f"12288->1536 deepfusion down hits={large_down_hits}, "
            f"expected {20 * decode_steps}"
        )
    if not case.large_down and large_down_hits != 0:
        errors.append(f"production unexpectedly produced {large_down_hits} large down hits")

    paged = stats.get("paged_decode_runtime") or {}
    gqa2_hits = delta["paged_gqa2_direct_hits"]
    generic_hits = delta["paged_generic_direct_hits"]
    if gqa2_hits != 28 * decode_steps:
        errors.append(
            f"sliding GQA2 hits={gqa2_hits}, expected {28 * decode_steps}"
        )
    expected_generic = 0 if case.attention_segments else 7 * decode_steps
    if generic_hits != expected_generic:
        errors.append(
            f"full generic hits={generic_hits}, expected {expected_generic}"
        )
    if case.attention_segments:
        topology = "e2b_l4_b1_full_h512_gqa8"
        selected_segments = paged.get("grouped_segmented_selected_segments") or {}
        selected_tiles = paged.get("grouped_segmented_selected_tile_sizes") or {}
        if int(selected_segments.get(topology) or 0) != case.attention_segments:
            errors.append("B1 H512 grouped attention selected the wrong segment count")
        if int(selected_tiles.get(topology) or 0) != case.attention_tile:
            errors.append("B1 H512 grouped attention selected the wrong tile size")
    if bool(paged.get("grouped_segmented_disabled")):
        errors.append(
            "grouped attention disabled itself: "
            + str(paged.get("grouped_segmented_failure") or "unknown failure")
        )
    if bool(stats.get("gemma4_dense_attn_mlp_bridge_runtime_disabled")):
        errors.append(
            "dense bridge disabled itself: "
            + str(stats.get("gemma4_dense_attn_mlp_bridge_failure") or "unknown failure")
        )
    if bool(stats.get("fused_rmsnorm_lm_head_argmax_disabled")):
        errors.append(
            "fused RMSNorm LM head disabled itself: "
            + str(stats.get("fused_rmsnorm_lm_head_argmax_error") or "unknown failure")
        )
    if bool(stats.get("gemma4_flat_fused_gateup_runtime_disabled")):
        errors.append("fused gate-up disabled itself")
    return errors


def _measure_pair(
    engine: Any,
    prompts: list[str],
    case: DecodeCase,
    production_lm_state: dict[str, Any],
    *,
    repeat: int,
    long_tokens: int,
) -> dict[str, Any]:
    _apply_case(engine.model, case, production_lm_state)
    before = _counter_snapshot(engine.model)
    short = _run(engine, prompts, 1)
    long = _run(engine, prompts, long_tokens)
    after = _counter_snapshot(engine.model)
    delta = _counter_delta(before, after)
    decode_s = float(long["elapsed_s"]) - float(short["elapsed_s"])
    stats = _runtime_stats(engine.model)
    errors = _audit_pair(
        case,
        delta,
        stats,
        decode_steps=long_tokens - 1,
    )
    if short["generated_tokens"] != 1:
        errors.append(f"O1 generated {short['generated_tokens']} tokens instead of 1")
    if long["generated_tokens"] != long_tokens:
        errors.append(
            f"O{long_tokens} generated {long['generated_tokens']} tokens instead of {long_tokens}"
        )
    if decode_s <= 0.0:
        errors.append("paired incremental decode wall time is not positive")
    expected_forced_token = int(os.environ.get("MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID", "-1"))
    for label, measurement, count in (("O1", short, 1), ("long", long, long_tokens)):
        if measurement["lengths"] != [count]:
            errors.append(f"{label} generated unexpected per-request lengths")
        observed_forced = measurement["scheduler_stats"].get("benchmark_forced_token_id", -1)
        if expected_forced_token < 0 or observed_forced != expected_forced_token:
            errors.append(f"{label} forced-token route is not active as configured")
    return {
        "family": case.family,
        "case": case.name,
        "case_config": asdict(case),
        "repeat": repeat,
        "short": short,
        "long": long,
        "paired_incremental_decode_ms": decode_s * 1000.0,
        "paired_incremental_decode_tps": (
            (long_tokens - 1) / decode_s if decode_s > 0.0 else 0.0
        ),
        "counter_delta": delta,
        "runtime_state": {
            "bridge_enabled": bool(
                stats.get("gemma4_dense_attn_mlp_bridge_decode_enabled")
            ),
            "fused_rms_lm_head_use": bool(
                stats.get("fused_rmsnorm_lm_head_argmax_use")
            ),
            "large_gateup_experiment": bool(
                stats.get("gemma4_e2b_b1_large_gateup_experiment")
            ),
            "large_down_experiment": bool(
                stats.get("gemma4_e2b_b1_large_down_experiment")
            ),
            "paged_decode_runtime": stats.get("paged_decode_runtime") or {},
            "mlp_gemv_configs": stats.get("gemma4_b1_mlp_gemv_configs") or {},
            "mlp_gemv_failures": stats.get("gemma4_b1_mlp_gemv_failures") or {},
            "mlp_gemv_dispatch": stats.get("gemma4_b1_mlp_gemv_dispatch", "legacy"),
        },
        "errors": errors,
    }


def _summarize_family(
    family: str,
    cases: tuple[DecodeCase, ...],
    samples: list[dict[str, Any]],
    *,
    minimum_speedup: float,
    maximum_spread: float,
) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {case.name: [] for case in cases}
    for row in samples:
        by_case[str(row["case"])].append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for case in cases:
        rows = by_case[case.name]
        rates = [float(row["paired_incremental_decode_tps"]) for row in rows]
        summaries[case.name] = {
            "config": asdict(case),
            "samples": len(rows),
            "median_incremental_decode_tps": _median(rates),
            "geomean_incremental_decode_tps": _geomean(rates),
            "spread_ratio": (
                max(rates) / min(rates) if rates and min(rates) > 0.0 else float("inf")
            ),
            "all_counter_deltas": [row["counter_delta"] for row in rows],
            "errors": [error for row in rows for error in row["errors"]],
        }

    baseline_tps = float(summaries["production"]["median_incremental_decode_tps"])
    for name, row in summaries.items():
        value = float(row["median_incremental_decode_tps"])
        row["speedup_vs_production"] = value / baseline_tps if baseline_tps > 0.0 else 0.0
        full_model_rates = [float(sample["long"]["output_tps"])
                            for sample in by_case[name] if "output_tps" in sample["long"]]
        if full_model_rates:
            row["median_end_to_end_output_tps"] = _median(full_model_rates)

    digest_sets: dict[int, set[str]] = {}
    for row in samples:
        digest_sets.setdefault(1, set()).add(str(row["short"]["digest"]))
        digest_sets.setdefault(
            int(sum(row["long"]["lengths"])), set()
        ).add(str(row["long"]["digest"]))
    route_digest_match = all(len(values) == 1 for values in digest_sets.values())
    stable = all(
        float(row["spread_ratio"]) <= maximum_spread for row in summaries.values()
    )
    errors = [
        f"{name}: {error}"
        for name, row in summaries.items()
        for error in row["errors"]
    ]
    if not route_digest_match:
        errors.append("forced-route token digests differ between policies")
    if not stable:
        errors.append("one or more policy rates exceeded maximum spread")
    valid = bool(not errors)
    winner_name, winner = max(
        summaries.items(),
        key=lambda item: float(item[1]["median_incremental_decode_tps"]),
    )
    winner_speedup = float(winner["speedup_vs_production"])
    if not valid:
        decision = "INVALID_GATE"
    elif winner_name != "production" and winner_speedup >= minimum_speedup:
        decision = "CANDIDATE_WINS_FULL_MODEL"
    else:
        decision = "KEEP_PRODUCTION"
    return {
        "family": family,
        "decision": decision,
        "valid": valid,
        "stable": stable,
        "route_digest_match": route_digest_match,
        "minimum_speedup": minimum_speedup,
        "maximum_spread": maximum_spread,
        "production_decode_tps": baseline_tps,
        "winner": winner_name,
        "winner_speedup": winner_speedup,
        "cases": summaries,
        "errors": errors,
    }


def _rotated(cases: tuple[DecodeCase, ...], repeat: int) -> list[DecodeCase]:
    ordered = list(cases)
    if repeat % 2 == 0:
        ordered.reverse()
    offset = (repeat - 1) % len(ordered)
    return ordered[offset:] + ordered[:offset]


def _preflight_errors(case: DecodeCase, correctness: dict[str, Any]) -> list[str]:
    if case.name == "production":
        return []
    if case.gemv_gateup or case.gemv_down:
        errors = []
        for operation, config in (("gateup", case.gemv_gateup), ("down", case.gemv_down)):
            if not config:
                continue
            probe = (correctness.get("projections") or {}).get(config) or {}
            if not probe.get("correct"):
                errors.append(f"{operation}/{config}: " + str(
                    probe.get("error") or "missing or failed MLP numerical/repeatability check"
                ))
        return errors
    if case.attention_segments:
        probes = [row for row in correctness.get("cases", []) if row["case"] == case.name]
        if not probes:
            return ["missing attention correctness preflight"]
        return [
            f"attention preflight context={row.get('context')}: "
            + str(row.get("error") or "numerical tolerance or repeatability failed")
            for row in probes if not row.get("correct")
        ]
    if not correctness.get("all_correct"):
        return [str(correctness.get("error") or "family correctness preflight failed")]
    return []


def _warmup_pair(engine, prompts, case, production_lm_state, *, long_tokens, warmups):
    """Traverse every measured decode position before accepting a candidate."""
    for warmup in range(max(1, warmups)):
        print(f"Warmup {case.name} {warmup + 1}/{max(1, warmups)} O1/O{long_tokens}", flush=True)
        try:
            row = _measure_pair(
                engine, prompts, case, production_lm_state,
                repeat=0, long_tokens=long_tokens,
            )
        except Exception as exc:
            return [f"warmup failed: {type(exc).__name__}: {exc}"]
        if row["errors"]:
            return row["errors"]
    return []


def _validate_greedy_route(engine, prompts, case, production_lm_state, *, tokens):
    """Independent output check: this probe must not use the forced-token route."""
    previous = os.environ.get("MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID")
    try:
        os.environ["MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID"] = "-1"
        _apply_case(engine.model, case, production_lm_state)
        result = _run(engine, prompts, tokens)
        result["valid"] = bool(
            result["scheduler_stats"].get("benchmark_forced_token_id") == -1
            and result["lengths"] == [tokens]
        )
        return result
    except Exception as exc:
        return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if previous is None:
            os.environ.pop("MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID", None)
        else:
            os.environ["MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID"] = previous


def _summarize_followup(cases, samples, rejected, *, minimum_speedup, maximum_spread):
    """Reject failed candidates individually without disguising their failures."""
    rejected = {name: list(errors) for name, errors in rejected.items()}
    for case in cases:
        if case.name in rejected:
            continue
        rows = [row for row in samples if row["case"] == case.name]
        rates = [float(row["paired_incremental_decode_tps"]) for row in rows]
        errors = [error for row in rows for error in row["errors"]]
        if len(rows) < 3:
            errors.append("fewer than three measured samples")
        if not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
            errors.append("invalid incremental decode rates")
        elif max(rates) / min(rates) > maximum_spread:
            errors.append("rate spread exceeds maximum")
        if errors:
            rejected[case.name] = errors
    eligible = tuple(case for case in cases if case.name not in rejected)
    if "production" in rejected:
        return {
            "family": cases[0].family, "valid": False, "decision": "INVALID_BASELINE",
            "winner": None, "winner_speedup": 0.0, "cases": {},
            "rejected_cases": rejected, "errors": rejected["production"],
        }
    eligible_names = {case.name for case in eligible}
    report = _summarize_family(
        cases[0].family, eligible,
        [row for row in samples if row["case"] in eligible_names],
        minimum_speedup=minimum_speedup, maximum_spread=maximum_spread,
    )
    report["rejected_cases"] = rejected
    report["all_candidates_valid"] = not rejected
    if len(eligible) == 1:
        report["decision"] = "NO_VALID_CANDIDATE"
        report["baseline_valid"] = report["valid"]
        report["valid"] = False
        report["winner"] = None
        report["winner_speedup"] = 0.0
        report["errors"].append("no candidate completed a valid comparison against production")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "h512-followup", "mlp-gemv", "mlp-dispatch"), default="all")
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--forced-token-id", type=int, default=42)
    parser.add_argument("--max-seq-len", type=int, default=2304)
    parser.add_argument("--minimum-speedup", type=float, default=1.01)
    parser.add_argument("--maximum-spread", type=float, default=1.06)
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-exit", action="store_true")
    args = parser.parse_args()
    mlp_gemv = args.suite in ("mlp-gemv", "mlp-dispatch")
    followup = args.suite != "all"
    families = (("h512_followup", H512_FOLLOWUP_CASES),) if followup else FAMILIES
    if mlp_gemv:
        from benchmarks.gemma4_b1_mlp_gemv_frontier import build_cases, validate_projections
        families = (("mlp_gemv", build_cases()),)
        if args.suite == "mlp-dispatch":
            from benchmarks.gemma4_b1_mlp_gemv_frontier import build_dispatch_cases
            families = (("mlp_gemv", build_dispatch_cases()),)
    if args.repeats < 3:
        raise SystemExit("--repeats must be at least 3")
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    if args.max_new_tokens <= 1:
        raise SystemExit("--max-new-tokens must exceed 1 for paired decode")
    if args.prompt_tokens + args.max_new_tokens > args.max_seq_len:
        raise SystemExit("prompt plus output exceeds --max-seq-len")

    profile = _configure_environment(args.model, args.forced_token_id)

    import torch
    from benchmarks import benchmark_inference_matrix as matrix
    from megagemm.engine import InferenceEngine

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    gpu = torch.cuda.get_device_name(0)
    if "l4" not in gpu.lower():
        raise SystemExit(f"NVIDIA L4 required, found {gpu}")

    compile_report = {"skipped": True}
    if mlp_gemv:
        from benchmarks.compile_gemma4_b1_mlp_gemv import compile_kernels
        import traceback

        print("MLP compiler preflight: sm_89; model loads: 0", flush=True)
        try:
            compile_report = compile_kernels()
        except Exception:
            compile_report = {"any_biasless_compiled": False, "error": traceback.format_exc()}
        if not compile_report.get("any_biasless_compiled"):
            payload = {
                "benchmark": "gemma4_e2b_l4_b1_decode_frontier", "suite": args.suite,
                "valid": False, "decision": "NO_COMPILED_CANDIDATE",
                "method": {"model_loads": 0}, "compiler_preflight": compile_report,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print("NO_COMPILED_CANDIDATE: stopped before tokenizer/model download.", flush=True)
            print(f"Wrote: {args.output}", flush=True)
            return 2

    tokenizer = matrix.load_tokenizer(
        args.model,
        local_files_only=args.local_files_only,
    )
    prompts, prompt_actual = matrix.build_prompts(
        tokenizer,
        1,
        args.prompt_tokens,
    )
    print("Gemma 4 E2B/L4 B1 decode frontier", flush=True)
    print(f"  gpu: {gpu}", flush=True)
    print(f"  shape: B1/P{args.prompt_tokens}/O1-O{args.max_new_tokens}/BF16", flush=True)
    print(f"  suite: {args.suite}", flush=True)
    print(f"  experiment families: {sum(len(cases) for _, cases in families)} cases", flush=True)
    print(f"  full-length warmup: O1/O{args.max_new_tokens}", flush=True)
    if followup:
        print("  production baseline: B1 bridge enabled; production LM head", flush=True)
    print("  model loads: 1", flush=True)

    engine = InferenceEngine(
        args.model,
        dtype=torch.bfloat16,
        device="cuda",
        max_batch_size=1,
        max_seq_len=args.max_seq_len,
        num_blocks=0,
        block_size=16,
        kv_alloc="auto",
        cache_dir=args.cache_dir,
    )
    model = engine.model
    model._prepare_flat_decode()
    if not model._flat_decode_ready:
        raise RuntimeError(
            "flat decode is not ready: "
            + str(getattr(model, "_flat_decode_failed_reason", "unknown"))
        )

    # Bootstrap allocates B1 buffers and calibrates the actual production LM
    # head once.  It is intentionally outside all measured samples.
    bootstrap = _run(engine, prompts, 2)
    if model._gemma4_flat_dense_attn_mlp_input_bufs is None:
        raise RuntimeError("B1 dense bridge experiment buffers were not allocated")
    model._gemma4_flat_dense_attn_mlp_bridge_enabled = False
    model._gemma4_flat_b1_large_gateup_experiment = False
    model._gemma4_flat_b1_large_down_experiment = False
    production_lm_state = _capture_lm_state(model)
    lm_correctness = (
        {"skipped": True, "reason": "LM head unchanged in followup"}
        if followup else _validate_lm_head(model, production_lm_state)
    )
    attention_cases = H512_FOLLOWUP_CASES if followup else ATTENTION_CASES
    engine_prompt = int(
        bootstrap.get("engine_prompt_tokens")
        or (bootstrap["scheduler_stats"].get("prefill_chunk_plan") or {}).get("total_prompt_tokens")
        or prompt_actual
    )
    if engine_prompt + args.max_new_tokens > args.max_seq_len:
        raise SystemExit(
            f"tokenized engine prompt ({engine_prompt}) plus output "
            f"({args.max_new_tokens}) exceeds --max-seq-len={args.max_seq_len}"
        )
    attention_correctness = {"skipped": True, "cases": [], "all_correct": True} if mlp_gemv else _validate_attention_cases(
        attention_cases,
        context=engine_prompt + 1,
        table_blocks=math.ceil(args.max_seq_len / 16),
    )
    if followup and not mlp_gemv:
        end_correctness = _validate_attention_cases(
            attention_cases, context=engine_prompt + args.max_new_tokens - 1,
            table_blocks=math.ceil(args.max_seq_len / 16),
        )
        attention_correctness["cases"].extend(end_correctness["cases"])
        attention_correctness["all_correct"] &= end_correctness["all_correct"]
        attention_correctness["end_shape"] = end_correctness["shape"]
    mlp_correctness = {"skipped": True} if followup else _validate_large_mlp(model)
    if mlp_gemv:
        mlp_correctness = validate_projections(model)
    correctness_by_family = {
        "core_factorial": {
            "all_correct": bool(lm_correctness.get("exact_token_match")),
            "lm_head": lm_correctness,
        },
        "full_attention_h512": attention_correctness,
        "large_mlp": mlp_correctness,
        "h512_followup": attention_correctness,
        "mlp_gemv": mlp_correctness,
    }

    all_samples: list[dict[str, Any]] = []
    family_reports: dict[str, dict[str, Any]] = {}
    for family, cases in families:
        print(f"\n=== {family} ===", flush=True)
        rejected: dict[str, list[str]] = {}
        active_cases = []
        greedy_checks = {}
        for case in cases:
            reasons = _preflight_errors(case, correctness_by_family[family])
            if not reasons:
                reasons = _warmup_pair(
                    engine, prompts, case, production_lm_state,
                    long_tokens=args.max_new_tokens, warmups=args.warmups,
                )
            if followup and not reasons:
                probe = _validate_greedy_route(
                    engine, prompts, case, production_lm_state,
                    tokens=min(16, args.max_new_tokens),
                )
                greedy_checks[case.name] = probe
                baseline_probe = greedy_checks.get("production") or {}
                if not probe["valid"] or not baseline_probe.get("valid"):
                    reasons.append("unforced greedy probe was not valid: " + str(
                        probe.get("error") or baseline_probe.get("error") or "route or length mismatch"
                    ))
                elif probe["digest"] != baseline_probe["digest"]:
                    reasons.append("unforced greedy token digest differs from production")
                print(f"GREEDY {case.name}: valid={not reasons}", flush=True)
            if reasons:
                rejected[case.name] = reasons
                print(f"REJECT {case.name}: " + json.dumps(reasons), flush=True)
                if case.name == "production":
                    break
            else:
                active_cases.append(case)
        family_samples: list[dict[str, Any]] = []
        if "production" in rejected:
            active_cases = []
        for repeat in range(1, args.repeats + 1):
            for case in _rotated(tuple(active_cases), repeat) if active_cases else []:
                print(
                    f"Run family={family} case={case.name} "
                    f"repeat={repeat}/{args.repeats}",
                    flush=True,
                )
                row = _measure_pair(
                    engine,
                    prompts,
                    case,
                    production_lm_state,
                    repeat=repeat,
                    long_tokens=args.max_new_tokens,
                )
                family_samples.append(row)
                all_samples.append(row)
                print(
                    f"  decode={row['paired_incremental_decode_tps']:.2f} tok/s "
                    f"wall={row['paired_incremental_decode_ms']:.2f}ms "
                    f"errors={len(row['errors'])}",
                    flush=True,
                )
                for error in row["errors"]:
                    print(f"    ERROR: {error}", flush=True)
        report = _summarize_followup(
            cases, family_samples, rejected,
            minimum_speedup=args.minimum_speedup, maximum_spread=args.maximum_spread,
        ) if followup else _summarize_family(
            family,
            cases,
            family_samples,
            minimum_speedup=args.minimum_speedup,
            maximum_spread=args.maximum_spread,
        )
        report["correctness"] = correctness_by_family[family]
        report["rejected_cases"] = report.get("rejected_cases", rejected)
        report["greedy_checks"] = greedy_checks
        if not followup and (rejected or not correctness_by_family[family].get("all_correct")):
            report["valid"] = False
            report["decision"] = "INVALID_GATE"
            report["errors"].append(
                f"{family} candidate correctness preflight failed"
            )
        family_reports[family] = report
        print(
            f"DECISION family={family} decision={report['decision']} "
            f"winner={report['winner']} speedup={report['winner_speedup']:.4f}x",
            flush=True,
        )

    _apply_case(model, H512_FOLLOWUP_CASES[0] if followup else CORE_CASES[0], production_lm_state)
    valid = bool(all(report["valid"] for report in family_reports.values()))
    payload = {
        "benchmark": "gemma4_e2b_l4_b1_decode_frontier",
        "schema_version": 1,
        "suite": args.suite,
        "valid": valid,
        "model": args.model,
        "shape": {
            "batch_size": 1,
            "prompt_tokens_requested": args.prompt_tokens,
            "prompt_tokens_actual": prompt_actual,
            "engine_prompt_tokens_including_template": engine_prompt,
            "max_new_tokens": args.max_new_tokens,
            "dtype": "bf16",
            "hardware": "NVIDIA L4",
        },
        "method": {
            "model_loads": 1,
            "paired_decode": "O(max)-O1 wall time within repeat",
            "schedule": "alternating-direction rotated cases",
            "warmups_per_case": max(1, args.warmups),
            "warmup_output_tokens": args.max_new_tokens,
            "baseline_bridge_enabled": followup,
            "repeats": args.repeats,
            "forced_token_id": args.forced_token_id,
            "promotion_scope": "exact measured B1/P2048 decode shape only",
        },
        "lm_head_correctness_probe": lm_correctness,
        "attention_correctness_preflight": attention_correctness,
        "large_mlp_correctness_preflight": mlp_correctness,
        "compiler_preflight": compile_report,
        "families": family_reports,
        "profile_environment": profile,
        "system": {
            "git": matrix.git_snapshot(),
            "gpu": matrix.gpu_snapshot(),
            "nvidia_smi": matrix.nvidia_smi_snapshot(),
            "packages": matrix.installed_package_versions(),
        },
        "samples": all_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\n" + "=" * 104)
    print("GEMMA 4 E2B / L4 / B1 — ONE-LOAD DECODE FRONTIER")
    print("=" * 104)
    for family, report in family_reports.items():
        print(f"\n{family}: {report['decision']}")
        for name, row in report["cases"].items():
            e2e = row.get("median_end_to_end_output_tps")
            print(
                f"  {name:<38} "
                f"decode={row['median_incremental_decode_tps']:8.2f} tok/s "
                f"speedup={row['speedup_vs_production']:.4f}x "
                f"spread={row['spread_ratio']:.4f}"
                + (f" end-to-end={e2e:.2f} tok/s" if e2e is not None else "")
            )
        for name, reasons in report.get("rejected_cases", {}).items():
            print(f"  REJECTED {name}: " + json.dumps(reasons))
    if not followup:
        print(f"\nLM-head exact token probe: {lm_correctness['exact_token_match']}")
    print(f"VALID: {valid}")
    print(f"Wrote: {args.output}")
    if (args.strict_exit or mlp_gemv) and not valid:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
