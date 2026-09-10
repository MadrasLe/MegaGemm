"""Run a compact, same-environment MegaGemm/vLLM publication matrix.

This wrapper launches each backend in a fresh child process, keeps the workload
identical, disables vLLM prefix caching, writes the standard JSONL/JSON/CSV
artifacts, emits comparison CSV files, and packages the run directory as ZIP.
It does not install or upgrade dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "benchmarks" / "benchmark_inference_matrix.py"
COMPARE = ROOT / "benchmarks" / "compare_inference_summaries.py"


@dataclass(frozen=True)
class Variant:
    name: str
    backend: str
    dtype: str = "fp16"
    quantize: str | None = None


VARIANTS = {
    "megagemm-fp16": Variant("megagemm-fp16", "megagemm"),
    "vllm-fp16": Variant("vllm-fp16", "vllm"),
    "megagemm-bf16": Variant("megagemm-bf16", "megagemm", dtype="bf16"),
    "vllm-bf16": Variant("vllm-bf16", "vllm", dtype="bf16"),
    "megagemm-int8": Variant(
        "megagemm-int8", "megagemm", dtype="fp16", quantize="int8"
    ),
}


# Settings shared only where E2B and E4B have the same measured behavior.
# The checkpoint-specific profiles below deliberately keep scheduler and
# RMSNorm decisions separate: E2B is 35L/1536H/GQA8 with 15 KV layers, while
# E4B is 42L/2560H/GQA4 with 24 KV layers.  Treating them as one runtime profile
# caused a severe E2B regression that a structural-only audit failed to catch.
GEMMA4_DENSE_COMMON_PROFILE = {
    "MEGAGEMM_FLAT_DECODE": "1",
    "MEGAGEMM_FUSED_ROPE_ATTN": "1",
    "MEGAGEMM_FAST_GEMV": "1",
    "MEGAGEMM_DEEPFUSION_MLP": "1",
    "MEGAGEMM_GEMMA4_FUSED_QKV_DECODE": "1",
    "MEGAGEMM_GEMMA4_FUSED_GATEUP_DECODE": "1",
    "MEGAGEMM_GEMMA4_DEEPFUSION_MLP_DECODE": "1",
    "MEGAGEMM_FUSED_LM_HEAD_ARGMAX_DECODE": "1",
    "MEGAGEMM_FUSED_RMSNORM_LM_HEAD_ARGMAX_DECODE": "1",
    "MEGAGEMM_GEMMA4_IMPLICIT_CAUSAL_PREFILL": "1",
    "MEGAGEMM_GEMMA4_LONG_SLIDING_PREFILL": "1",
    "MEGAGEMM_GEMMA4_LONG_FULL_PREFILL": "1",
    "MEGAGEMM_GEMMA4_VECTORIZED_PREFILL_KV": "1",
    "MEGAGEMM_GEMMA4_FUSED_DUAL_FFN_NORM_PREFILL": "1",
    "MEGAGEMM_GEMMA4_FUSED_ADD_FFN_NORM_PREFILL": "1",
    "MEGAGEMM_GEMMA4_FUSED_POST_FFN_NORMS_PREFILL": "1",
}


# Preserve E2B's measured L4 paths. Scheduler/graph selection intentionally
# comes from RuntimePolicy because it is batch-scoped: B4 uses the promoted
# one-step CUDA Graph, while B1/B2/B8 keep their independently measured paths.
GEMMA4_E2B_FAST_PROFILE = {
    **GEMMA4_DENSE_COMMON_PROFILE,
    "MEGAGEMM_DISABLE_CUDA_RMSNORM": "1",
}


# E4B benefits from the native CUDA RMSNorm and one-token scheduler route.
GEMMA4_E4B_FAST_PROFILE = {
    **GEMMA4_DENSE_COMMON_PROFILE,
    "MEGAGEMM_DECODE_CUDA_GRAPHS": "0",
    "MEGAGEMM_DECODE_CUDA_GRAPHS_PREFER_STEP": "0",
    "MEGAGEMM_DISABLE_CUDA_RMSNORM": "0",
    "MEGAGEMM_DECODE_PREFER_STEP": "1",
    "MEGAGEMM_REUSE_REQUEST_SCHEDULER": "1",
}


MEGAGEMM_PROFILES = {
    "none": {},
    "gemma4-e2b-fast": GEMMA4_E2B_FAST_PROFILE,
    "gemma4-e4b-fast": GEMMA4_E4B_FAST_PROFILE,
}


GEMMA4_PROFILE_REQUIREMENTS = {
    "gemma4-e2b-fast": {
        "model_marker": "e2b",
        "prefer_step": False,
        "reuse_scheduler": False,
        "decode_cuda_graph_batches": (4,),
        "reuse_scheduler_batches": (4,),
        "require_dense_post_norm_chain": True,
        "require_e2b_h512_dense_bridge_pair": True,
        "require_bf16_batch8_cublas_mlp": True,
        "require_e2b_l4_sliding_prefill": True,
        "require_e2b_l4_full_prefill_expand": True,
        "topology": {
            "num_hidden_layers": 35,
            "hidden_size": 1536,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 20,
            "kv_cache_layers": 15,
            "sliding_attention_layers": 28,
            "full_attention_layers": 7,
        },
        # Conservative regression floors derived from the validated L4/BF16
        # paths. The long B8 floor includes the promoted sliding-prefill Triton
        # result (130.48 tok/s) with enough margin for run-to-run variance.
        "l4_bf16_floors": {
            ("single", 1, 128): 15.0,
            ("batch", 8, 128): 120.0,
            ("long_context", 1, 2048): 14.0,
            ("long_context", 8, 2048): 115.0,
        },
    },
    "gemma4-e4b-fast": {
        "model_marker": "e4b",
        "prefer_step": True,
        "reuse_scheduler": True,
        "decode_cuda_graph_batches": (),
        "reuse_scheduler_batches": (),
        "require_dense_post_norm_chain": False,
        "require_bf16_batch8_cublas_mlp": False,
        "require_e2b_l4_sliding_prefill": False,
        "require_e2b_l4_full_prefill_expand": False,
        "topology": {
            "num_hidden_layers": 42,
            "hidden_size": 2560,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "num_kv_shared_layers": 18,
            "kv_cache_layers": 24,
            "sliding_attention_layers": 35,
            "full_attention_layers": 7,
        },
        "l4_bf16_floors": {
            ("single", 1, 128): 12.0,
            ("batch", 8, 128): 90.0,
            ("long_context", 1, 2048): 11.0,
            ("long_context", 8, 2048): 45.0,
        },
    },
}


def resolve_megagemm_profile(requested: str, model: str) -> str:
    if requested != "auto":
        return requested
    normalized = model.lower()
    if "gemma-4-" in normalized and "e2b" in normalized:
        return "gemma4-e2b-fast"
    if "gemma-4-" in normalized and "e4b" in normalized:
        return "gemma4-e4b-fast"
    return "none"


def profile_environment(profile: str, model: str) -> dict[str, str]:
    requirement = GEMMA4_PROFILE_REQUIREMENTS.get(profile)
    if requirement and requirement["model_marker"] not in model.lower():
        raise ValueError(
            f"profile {profile!r} is not valid for model {model!r}; "
            "use --megagemm-profile auto"
        )
    return dict(MEGAGEMM_PROFILES[profile])


def child_environment(
    variant: Variant,
    profile: str,
    model: str,
) -> dict[str, str]:
    env = os.environ.copy()
    if variant.backend == "megagemm":
        if profile in GEMMA4_PROFILE_REQUIREMENTS:
            # A publication profile is a clean environment, not a partial
            # overlay on arbitrary notebook experiments.
            for key in tuple(env):
                if key.startswith("MEGAGEMM_"):
                    env.pop(key, None)
            # Deterministic cuBLAS workspaces were copied from a separate A100
            # validation harness and are not part of the dense L4 speed path.
            env.pop("CUBLAS_WORKSPACE_CONFIG", None)
        env.update(profile_environment(profile, model))
    return env


def parse_variants(raw: str) -> list[Variant]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise SystemExit(
            f"unknown variants: {', '.join(unknown)}; valid: {', '.join(VARIANTS)}"
        )
    if not names:
        raise SystemExit("at least one variant is required")
    return [VARIANTS[name] for name in names]


def shell_join(command: list[str]) -> str:
    return shlex.join(command)


def auto_hardware_label() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "no-gpu"
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0).lower()
        aliases = (
            ("tesla t4", "t4"),
            ("nvidia t4", "t4"),
            ("l4", "l4"),
            ("a100", "a100"),
            ("h100", "h100"),
            ("a10g", "a10g"),
            ("v100", "v100"),
            ("p100", "p100"),
        )
        suffix = next((label for needle, label in aliases if needle in name), None)
        if suffix is None:
            suffix = "".join(ch for ch in name if ch.isalnum())[:24] or "gpu"
        return f"{count}x{suffix}"
    except Exception:
        return "unknown-gpu"


def command_for(
    args: argparse.Namespace,
    variant: Variant,
    *,
    hardware_label: str,
    run_dir: Path,
    run_id: str,
    warmup: int,
    max_seq_len: int,
    megagemm_profile: str,
) -> list[str]:
    command = [
        sys.executable,
        str(MATRIX),
        "--backend",
        variant.backend,
        "--model",
        args.model,
        "--hardware-label",
        hardware_label,
        "--batch-sizes",
        args.batch_sizes,
        "--prompt-tokens",
        args.prompt_tokens,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(warmup),
        "--out-dir",
        str(run_dir),
        "--run-id",
        f"{run_id}_{variant.name}",
        "--device",
        "cuda",
        "--dtype",
        variant.dtype,
        "--max-seq-len",
        str(max_seq_len),
        "--max-batch-size",
        str(args.max_batch_size),
        "--ignore-eos",
    ]
    if variant.quantize:
        command.extend(["--quantize", variant.quantize])
    if variant.backend == "vllm":
        command.extend(
            [
                "--vllm-tensor-parallel-size",
                "1",
                "--vllm-gpu-memory-utilization",
                str(args.vllm_gpu_memory_utilization),
                "--vllm-max-model-len",
                str(max_seq_len),
                "--vllm-disable-prefix-caching",
            ]
        )
        if args.vllm_max_num_seqs > 0:
            command.extend(["--vllm-max-num-seqs", str(args.vllm_max_num_seqs)])
        if args.vllm_max_num_batched_tokens > 0:
            command.extend(
                [
                    "--vllm-max-num-batched-tokens",
                    str(args.vllm_max_num_batched_tokens),
                ]
            )
        if args.vllm_language_model_only:
            command.append("--vllm-language-model-only")
        if args.vllm_enforce_eager:
            command.append("--vllm-enforce-eager")
    return command


def summary_path(
    run_dir: Path,
    run_id: str,
    hardware_label: str,
    variant: Variant,
) -> Path:
    return run_dir / (
        f"{run_id}_{variant.name}_{hardware_label}_{variant.backend}_summary.json"
    )


def raw_path(
    run_dir: Path,
    run_id: str,
    hardware_label: str,
    variant: Variant,
) -> Path:
    return run_dir / f"{run_id}_{variant.name}_{hardware_label}_{variant.backend}.jsonl"


def validate_existing_variant_artifacts(
    path: Path,
    *,
    args: argparse.Namespace,
    variant: Variant,
    hardware_label: str,
    warmup: int,
    max_seq_len: int,
) -> None:
    """Reject stale artifacts before a resumed publication run reuses them."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("args") or {}
    expected = {
        "backend": variant.backend,
        "model": args.model,
        "hardware_label": hardware_label,
        "batch_sizes": args.batch_sizes,
        "prompt_tokens": args.prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "warmup": warmup,
        "dtype": variant.dtype,
        "quantize": variant.quantize,
        "max_seq_len": max_seq_len,
        "max_batch_size": args.max_batch_size,
    }
    mismatches = {
        key: {"expected": value, "recorded": recorded.get(key)}
        for key, value in expected.items()
        if recorded.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"cannot resume {variant.name}: existing summary does not match "
            f"this workload: {mismatches}"
        )


def raw_artifact_errors(path: Path) -> list[str]:
    """Return publication-blocking errors recorded in a backend JSONL."""
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception as exc:
        return [f"could not read raw artifact: {type(exc).__name__}: {exc}"]
    if not rows:
        return ["raw artifact contains no rows"]
    failed = [row for row in rows if not row.get("ok")]
    if not failed:
        return []
    reasons = sorted(
        {str(row.get("error") or "unknown backend failure") for row in failed}
    )
    return [
        f"{len(failed)}/{len(rows)} benchmark row(s) failed: "
        + "; ".join(reasons)
    ]


def _max_counter(stats: list[dict], key: str) -> int:
    return max((int(item.get(key, 0) or 0) for item in stats), default=0)


def audit_gemma4_dense_fast_path(path: Path, profile: str) -> dict:
    """Validate checkpoint topology, runtime route, and known L4 performance."""
    requirement = GEMMA4_PROFILE_REQUIREMENTS.get(profile)
    if requirement is None:
        raise ValueError(f"profile {profile!r} has no Gemma 4 fast-path audit")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    successful = [row for row in rows if row.get("ok")]
    failed = [row for row in rows if not row.get("ok")]
    if not successful:
        return {
            "status": "failed",
            "raw_jsonl": str(path),
            "successful_rows": 0,
            "errors": ["no successful benchmark rows"],
        }

    decode_stats = [
        row.get("decode_runtime_stats")
        for row in successful
        if isinstance(row.get("decode_runtime_stats"), dict)
    ]
    scheduler_stats = [
        row.get("scheduler_stats")
        for row in successful
        if isinstance(row.get("scheduler_stats"), dict)
    ]
    graph_stats = [
        item.get("decode_cuda_graphs")
        for item in scheduler_stats
        if isinstance(item.get("decode_cuda_graphs"), dict)
    ]
    execution_stats = [
        item.get("decode_execution")
        for item in scheduler_stats
        if isinstance(item.get("decode_execution"), dict)
    ]
    paged_stats = [
        item.get("paged_decode_runtime")
        for item in decode_stats
        if isinstance(item.get("paged_decode_runtime"), dict)
    ]
    topology_stats = [
        row.get("model_topology")
        for row in successful
        if isinstance(row.get("model_topology"), dict)
    ]
    prefer_step = bool(requirement["prefer_step"])
    expected_reuse = bool(requirement["reuse_scheduler"])
    expected_graph_batches = tuple(
        int(value) for value in requirement.get("decode_cuda_graph_batches", ())
    )
    expected_reuse_batches = tuple(
        int(value) for value in requirement.get("reuse_scheduler_batches", ())
    )
    successful_batches = {
        int(row.get("batch_size", 0) or 0) for row in successful
    }
    graph_batches_present = successful_batches.intersection(expected_graph_batches)
    reuse_batches_present = successful_batches.intersection(expected_reuse_batches)

    errors: list[str] = []
    if failed:
        failure_kinds = sorted(
            {str(row.get("error") or "unknown benchmark failure") for row in failed}
        )
        errors.append(
            f"{len(failed)} benchmark row(s) failed: " + "; ".join(failure_kinds)
        )
    if len(decode_stats) != len(successful):
        errors.append("decode_runtime_stats missing from one or more rows")
    if len(topology_stats) != len(successful):
        errors.append("model_topology missing from one or more rows")
    expected_topology = dict(requirement["topology"])
    topology_mismatches: dict[str, list[object]] = {}
    for key, expected in expected_topology.items():
        actual_values = sorted(
            {item.get(key) for item in topology_stats},
            key=lambda value: str(value),
        )
        if actual_values != [expected]:
            topology_mismatches[key] = actual_values
    if topology_mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected_topology[key]!r})"
            for key, actual in topology_mismatches.items()
        )
        errors.append(f"checkpoint topology does not match {profile}: {details}")
    model_marker = str(requirement["model_marker"])
    row_models = {
        str(row.get("model")).lower()
        for row in successful
        if row.get("model")
    }
    if row_models and any(model_marker not in model for model in row_models):
        errors.append(
            f"benchmark rows do not match the {profile} checkpoint family"
        )
    if decode_stats and not all(item.get("flat_decode_ready") for item in decode_stats):
        reasons = sorted(
            {
                str(item.get("flat_decode_failed_reason") or "unknown")
                for item in decode_stats
                if not item.get("flat_decode_ready")
            }
        )
        errors.append(f"flat decode not ready: {', '.join(reasons)}")
    if decode_stats and any(item.get("flat_decode_failed") for item in decode_stats):
        errors.append("flat decode reported a runtime failure")
    # Graph/reuse can be global (E4B reuse) or batch-scoped (E2B B4 graph).
    # Only rows for a promoted batch may capture or replay a graph.
    if (expected_reuse or graph_batches_present or reuse_batches_present) and len(
        graph_stats
    ) != len(successful):
        errors.append("decode_cuda_graphs stats missing from one or more rows")
    graph_replays = _max_counter(graph_stats, "replays")
    graph_captures = _max_counter(graph_stats, "captures")
    graph_failures = _max_counter(graph_stats, "failures")
    for row in successful:
        scheduler = row.get("scheduler_stats")
        if not isinstance(scheduler, dict):
            continue
        batch_size = int(row.get("batch_size", 0) or 0)
        graph = scheduler.get("decode_cuda_graphs")
        if not isinstance(graph, dict):
            continue
        captures = int(graph.get("captures", 0) or 0)
        replays = int(graph.get("replays", 0) or 0)
        if not expected_graph_batches and graph.get("enabled"):
            errors.append("decode CUDA Graphs were unexpectedly enabled")
        if batch_size in expected_graph_batches:
            if not graph.get("enabled"):
                errors.append(
                    f"decode CUDA Graphs disabled for promoted batch B{batch_size}"
                )
            if captures <= 0 or replays <= 0:
                errors.append(
                    f"promoted batch B{batch_size} did not capture and replay its graph"
                )
        elif captures > 0 or replays > 0:
            errors.append(
                f"decode CUDA Graphs ran outside promoted batches: B{batch_size}"
            )
        graph_policy_batches = graph.get("decode_cuda_graph_policy_batches")
        if graph_policy_batches is not None and tuple(graph_policy_batches) != expected_graph_batches:
            errors.append(
                "decode CUDA Graph batch policy does not match the publication profile"
            )
        reuse_policy_batches = graph.get("request_scheduler_reuse_policy_batches")
        if reuse_policy_batches is not None and tuple(reuse_policy_batches) != expected_reuse_batches:
            errors.append(
                "request scheduler reuse batch policy does not match the publication profile"
            )
        reuse_count = int(graph.get("request_scheduler_reuse_count", 0) or 0)
        if batch_size in expected_reuse_batches:
            if reuse_count <= 0:
                errors.append(
                    f"promoted batch B{batch_size} did not reuse its request scheduler"
                )
        elif not expected_reuse and reuse_count > 0:
            errors.append(
                f"request scheduler reuse ran outside promoted batches: B{batch_size}"
            )
    if graph_failures > 0:
        failure_messages = sorted(
            {
                str(item.get("last_failure") or "unspecified")
                for item in graph_stats
                if int(item.get("failures", 0) or 0) > 0
            }
        )
        errors.append(
            f"decode CUDA Graphs reported {graph_failures} failure(s): "
            + "; ".join(failure_messages)
        )
    if len(execution_stats) != len(successful):
        errors.append("decode_execution stats missing from one or more rows")
    if execution_stats and not all(
        bool(item.get("prefer_step")) == prefer_step for item in execution_stats
    ):
        errors.append("scheduler decode route does not match the model profile")
    decode_step_batches = _max_counter(execution_stats, "decode_step_batches")
    multi_step_batches = _max_counter(execution_stats, "multi_step_batches")
    if prefer_step:
        if decode_step_batches <= 0:
            errors.append("E4B flat decode_step route was never exercised")
        if multi_step_batches > 0:
            errors.append("E4B unexpectedly used the slow decode_multi_step route")
    elif graph_batches_present and decode_step_batches <= 0:
        errors.append("E2B promoted graph decode_step route was never exercised")
    elif successful_batches.difference(expected_graph_batches) and multi_step_batches <= 0:
        errors.append("E2B flat decode_multi_step route was never exercised")

    request_scheduler_reuse_count = _max_counter(
        graph_stats, "request_scheduler_reuse_count"
    )
    if expected_reuse and request_scheduler_reuse_count <= 0:
        errors.append("E4B request scheduler reuse was never exercised")
    if not expected_reuse and not expected_reuse_batches and request_scheduler_reuse_count > 0:
        errors.append("request scheduler was unexpectedly reused")

    dense_tail_env_requested = os.environ.get(
        "MEGAGEMM_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    dense_tail_required = bool(
        requirement.get("require_dense_post_norm_chain", False)
    )
    dense_tail_requested = dense_tail_required or dense_tail_env_requested
    dense_tail_enabled = bool(
        decode_stats
        and all(
            item.get("gemma4_dense_post_norm_chain_decode_enabled")
            for item in decode_stats
        )
    )
    dense_tail_hits = _max_counter(
        decode_stats, "gemma4_dense_post_norm_chain_decode_hits"
    )
    if dense_tail_requested and not dense_tail_enabled:
        errors.append("requested Gemma 4 dense post-norm chain was not enabled")
    if dense_tail_requested and dense_tail_hits <= 0:
        errors.append("requested Gemma 4 dense post-norm chain was never exercised")

    require_batch8_cublas_mlp = bool(
        requirement.get("require_bf16_batch8_cublas_mlp", False)
    )
    batch8_rows = [
        row
        for row in successful
        if int(row.get("batch_size", 0) or 0) == 8
        and str(row.get("dtype") or "").lower() == "bf16"
        and "l4" in str(row.get("hardware_label") or "").lower()
        and isinstance(row.get("decode_runtime_stats"), dict)
    ]
    batch8_decode_stats = [row["decode_runtime_stats"] for row in batch8_rows]
    batch8_paged_stats = [
        stats["paged_decode_runtime"]
        for stats in batch8_decode_stats
        if isinstance(stats.get("paged_decode_runtime"), dict)
    ]
    require_e2b_h512_dense_bridge_pair = bool(
        requirement.get("require_e2b_h512_dense_bridge_pair", False)
    )
    e2b_h512_dense_bridge_pair_applicable = bool(
        require_e2b_h512_dense_bridge_pair and batch8_rows
    )
    pair_policy_enabled = bool(
        batch8_decode_stats
        and all(
            bool(
                (stats.get("runtime_policy") or {}).get(
                    "gemma4_e2b_h512_dense_bridge_pair"
                )
            )
            for stats in batch8_decode_stats
        )
    )
    dense_bridge_enabled = bool(
        batch8_decode_stats
        and all(
            stats.get("gemma4_dense_attn_mlp_bridge_decode_enabled")
            for stats in batch8_decode_stats
        )
    )
    dense_bridge_hits = _max_counter(
        batch8_decode_stats,
        "gemma4_dense_attn_mlp_bridge_decode_hits",
    )
    dense_bridge_disabled = any(
        stats.get("gemma4_dense_attn_mlp_bridge_runtime_disabled")
        for stats in batch8_decode_stats
    )
    dense_bridge_failures = sorted(
        {
            str(stats.get("gemma4_dense_attn_mlp_bridge_failure"))
            for stats in batch8_decode_stats
            if stats.get("gemma4_dense_attn_mlp_bridge_failure")
        }
    )
    h512_grouped_hits = _max_counter(
        batch8_paged_stats,
        "grouped_segmented_hits",
    )
    h512_segments = sorted(
        {
            int(
                (stats.get("grouped_segmented_selected_segments") or {}).get(
                    "e2b_l4_full_h512_gqa8", 0
                )
                or 0
            )
            for stats in batch8_paged_stats
        }
    )
    h512_tiles = sorted(
        {
            int(
                (stats.get("grouped_segmented_selected_tile_sizes") or {}).get(
                    "e2b_l4_full_h512_gqa8", 0
                )
                or 0
            )
            for stats in batch8_paged_stats
        }
    )
    h512_grouped_disabled = any(
        stats.get("grouped_segmented_disabled") for stats in batch8_paged_stats
    )
    h512_grouped_failures = sorted(
        {
            str(stats.get("grouped_segmented_failure"))
            for stats in batch8_paged_stats
            if stats.get("grouped_segmented_failure")
        }
    )
    if e2b_h512_dense_bridge_pair_applicable:
        if not pair_policy_enabled:
            errors.append("E2B L4 H512+dense-bridge pair policy was not enabled")
        if not dense_bridge_enabled or dense_bridge_hits <= 0:
            errors.append("E2B L4 dense attention-to-MLP bridge was not exercised")
        if dense_bridge_disabled or dense_bridge_failures:
            errors.append(
                "E2B L4 dense bridge disabled itself: "
                + "; ".join(dense_bridge_failures or ["unspecified"])
            )
        if h512_grouped_hits <= 0:
            errors.append("E2B L4 grouped full-H512 attention was not exercised")
        if h512_segments != [32] or h512_tiles != [16]:
            errors.append(
                "E2B L4 H512 launch does not match promoted segments/tile: "
                f"segments={h512_segments!r}, tiles={h512_tiles!r}"
            )
        if h512_grouped_disabled or h512_grouped_failures:
            errors.append(
                "E2B L4 grouped H512 attention disabled itself: "
                + "; ".join(h512_grouped_failures or ["unspecified"])
            )
    fused_gateup_policy_rows = sorted(
        {
            tuple(int(value) for value in stats.get(
                "gemma4_policy_bf16_fused_gateup_rows", []
            ))
            for stats in batch8_decode_stats
            if stats.get("gemma4_policy_bf16_fused_gateup_rows", [])
        }
    )
    deepfusion_policy_rows = sorted(
        {
            tuple(int(value) for value in stats.get(
                "gemma4_policy_bf16_deepfusion_rows", []
            ))
            for stats in batch8_decode_stats
            if stats.get("gemma4_policy_bf16_deepfusion_rows", [])
        }
    )
    cublas_gateup_policy_rows = sorted(
        {
            tuple(int(value) for value in stats.get(
                "gemma4_policy_bf16_cublas_gateup_rows", []
            ))
            for stats in batch8_decode_stats
        }
    )
    cublas_down_policy_rows = sorted(
        {
            tuple(int(value) for value in stats.get(
                "gemma4_policy_bf16_cublas_down_rows", []
            ))
            for stats in batch8_decode_stats
        }
    )
    batch8_fused_gateup_hits = _max_counter(
        batch8_decode_stats, "gemma4_flat_fused_gateup_hits"
    )
    batch8_deepfusion_hits = _max_counter(
        batch8_decode_stats, "gemma4_flat_deepfusion_hits"
    )
    batch8_cublas_mlp_applicable = bool(
        require_batch8_cublas_mlp and batch8_rows
    )
    if batch8_cublas_mlp_applicable:
        if cublas_gateup_policy_rows != [(8,)]:
            errors.append(
                "E2B L4 BF16 cuBLAS gate-up policy rows do not equal [8]: "
                f"{cublas_gateup_policy_rows!r}"
            )
        if cublas_down_policy_rows != [(8,)]:
            errors.append(
                "E2B L4 BF16 cuBLAS down policy rows do not equal [8]: "
                f"{cublas_down_policy_rows!r}"
            )
        if fused_gateup_policy_rows:
            errors.append(
                "E2B L4 BF16 batch-8 still promotes fused gate-up rows: "
                f"{fused_gateup_policy_rows!r}"
            )
        if deepfusion_policy_rows:
            errors.append(
                "E2B L4 BF16 batch-8 still promotes deepfusion rows: "
                f"{deepfusion_policy_rows!r}"
            )
        if batch8_fused_gateup_hits > 0:
            errors.append(
                "E2B L4 BF16 batch-8 fused gate-up path unexpectedly ran"
            )
        if batch8_deepfusion_hits > 0:
            errors.append(
                "E2B L4 BF16 batch-8 deepfusion path unexpectedly ran"
            )

    require_e2b_l4_sliding_prefill = bool(
        requirement.get("require_e2b_l4_sliding_prefill", False)
    )
    e2b_l4_long_rows = [
        row
        for row in successful
        if str(row.get("scenario") or "") == "long_context"
        and int(row.get("batch_size", 0) or 0) == 8
        and int(row.get("prompt_tokens_requested_per_request", 0) or 0) >= 2048
        and str(row.get("dtype") or "").lower() == "bf16"
        and "l4" in str(row.get("hardware_label") or "").lower()
        and isinstance(row.get("decode_runtime_stats"), dict)
    ]
    e2b_l4_long_stats = [row["decode_runtime_stats"] for row in e2b_l4_long_rows]
    e2b_l4_sliding_prefill_applicable = bool(
        require_e2b_l4_sliding_prefill and e2b_l4_long_rows
    )
    e2b_l4_sliding_prefill_enabled = bool(
        e2b_l4_long_stats
        and all(
            stats.get("gemma4_e2b_l4_sliding_prefill_enabled")
            for stats in e2b_l4_long_stats
        )
    )
    e2b_l4_sliding_prefill_hits = _max_counter(
        e2b_l4_long_stats,
        "gemma4_e2b_l4_sliding_prefill_hits",
    )
    if e2b_l4_sliding_prefill_applicable:
        if not e2b_l4_sliding_prefill_enabled:
            errors.append(
                "E2B L4 BF16 long sliding-prefill policy was not enabled"
            )
        if e2b_l4_sliding_prefill_hits <= 0:
            errors.append(
                "E2B L4 BF16 long sliding-prefill Triton kernel was never exercised"
            )

    require_e2b_l4_full_prefill_expand = bool(
        requirement.get("require_e2b_l4_full_prefill_expand", False)
    )
    e2b_l4_full_prefill_expand_applicable = bool(
        require_e2b_l4_full_prefill_expand and e2b_l4_long_rows
    )
    e2b_l4_full_prefill_expand_enabled = bool(
        e2b_l4_long_stats
        and all(
            stats.get("gemma4_e2b_l4_full_prefill_expand_enabled")
            for stats in e2b_l4_long_stats
        )
    )
    e2b_l4_full_prefill_expand_hits = _max_counter(
        e2b_l4_long_stats,
        "gemma4_e2b_l4_full_prefill_expand_hits",
    )
    e2b_l4_full_prefill_expand_enabled_layers = sorted(
        {
            int(
                stats.get(
                    "gemma4_e2b_l4_full_prefill_expand_enabled_layers",
                    0,
                )
                or 0
            )
            for stats in e2b_l4_long_stats
        }
    )
    e2b_l4_full_prefill_expand_errors = sorted(
        {
            str(stats.get("gemma4_e2b_l4_full_prefill_expand_error") or "")
            for stats in e2b_l4_long_stats
            if stats.get("gemma4_e2b_l4_full_prefill_expand_error")
        }
    )
    if e2b_l4_full_prefill_expand_applicable:
        if not e2b_l4_full_prefill_expand_enabled:
            errors.append(
                "E2B L4 BF16 long full-H512 expanded implicit-causal prefill "
                "policy was not enabled"
            )
        if e2b_l4_full_prefill_expand_hits <= 0:
            errors.append(
                "E2B L4 BF16 long full-H512 expanded implicit-causal prefill "
                "path was never exercised"
            )
        expected_full_layers = int(
            requirement.get("topology", {}).get("full_attention_layers", 0)
            or 0
        )
        if e2b_l4_full_prefill_expand_enabled_layers != [expected_full_layers]:
            errors.append(
                "E2B L4 BF16 long full-H512 expanded implicit-causal prefill "
                "path is not scoped to the expected full-attention layers: "
                f"{e2b_l4_full_prefill_expand_enabled_layers!r} != "
                f"[{expected_full_layers}]"
            )
        if e2b_l4_full_prefill_expand_errors:
            errors.append(
                "E2B L4 BF16 long full-H512 expanded implicit-causal prefill "
                "reported runtime errors: "
                + "; ".join(e2b_l4_full_prefill_expand_errors)
            )

    performance_gate = {
        "applicable": False,
        "hardware": None,
        "dtype": None,
        "measurements": {},
        "floors": {},
    }
    hardware_values = {
        str(row.get("hardware_label") or "").lower() for row in successful
    }
    dtype_values = {str(row.get("dtype") or "").lower() for row in successful}
    if (
        hardware_values
        and all("l4" in value for value in hardware_values)
        and dtype_values == {"bf16"}
    ):
        performance_gate["applicable"] = True
        performance_gate["hardware"] = sorted(hardware_values)
        performance_gate["dtype"] = "bf16"
        floors = dict(requirement["l4_bf16_floors"])
        for key, floor in floors.items():
            scenario, batch_size, prompt_tokens = key
            values = [
                float(row.get("output_tps", 0.0) or 0.0)
                for row in successful
                if str(row.get("scenario")) == scenario
                and int(row.get("batch_size", 0) or 0) == batch_size
                and int(row.get("prompt_tokens_requested_per_request", 0) or 0)
                == prompt_tokens
            ]
            label = f"{scenario}/b{batch_size}/p{prompt_tokens}"
            performance_gate["floors"][label] = floor
            if not values:
                continue
            measured = float(statistics.median(values))
            performance_gate["measurements"][label] = measured
            if measured < floor:
                errors.append(
                    f"L4 BF16 regression gate failed for {label}: "
                    f"{measured:.2f} < {floor:.2f} tok/s"
                )

    report = {
        "status": "failed" if errors else "passed",
        "profile": profile,
        "raw_jsonl": str(path),
        "successful_rows": len(successful),
        "failed_rows": len(failed),
        "required": {
            "flat_decode_ready": bool(
                decode_stats and all(item.get("flat_decode_ready") for item in decode_stats)
            ),
            "decode_mode": (
                "flat_single_step_eager"
                if prefer_step
                else (
                    "batch_scoped_cuda_graph"
                    if expected_graph_batches
                    else "flat_multi_step_eager"
                )
            ),
            "prefer_step": prefer_step,
            "decode_step_batches": decode_step_batches,
            "multi_step_batches": multi_step_batches,
            "decode_cuda_graphs_disabled": bool(
                not graph_stats
                or not any(item.get("enabled") for item in graph_stats)
            ),
            "decode_cuda_graph_captures": graph_captures,
            "decode_cuda_graph_replays": graph_replays,
            "decode_cuda_graph_failures": graph_failures,
            "decode_cuda_graph_policy_batches": list(expected_graph_batches),
            "request_scheduler_reuse_expected": expected_reuse,
            "request_scheduler_reuse_policy_batches": list(
                expected_reuse_batches
            ),
            "request_scheduler_reuse_count": request_scheduler_reuse_count,
            "dense_post_norm_chain_required": dense_tail_required,
            "dense_post_norm_chain_enabled": dense_tail_enabled,
            "dense_post_norm_chain_hits": dense_tail_hits,
            "batch8_cublas_mlp_required": require_batch8_cublas_mlp,
            "batch8_cublas_mlp_applicable": batch8_cublas_mlp_applicable,
            "batch8_fused_gateup_policy_rows": [
                list(rows) for rows in fused_gateup_policy_rows
            ],
            "batch8_deepfusion_policy_rows": [
                list(rows) for rows in deepfusion_policy_rows
            ],
            "batch8_cublas_gateup_policy_rows": [
                list(rows) for rows in cublas_gateup_policy_rows
            ],
            "batch8_cublas_down_policy_rows": [
                list(rows) for rows in cublas_down_policy_rows
            ],
            "batch8_fused_gateup_hits": batch8_fused_gateup_hits,
            "batch8_deepfusion_hits": batch8_deepfusion_hits,
            "e2b_h512_dense_bridge_pair_required": (
                require_e2b_h512_dense_bridge_pair
            ),
            "e2b_h512_dense_bridge_pair_applicable": (
                e2b_h512_dense_bridge_pair_applicable
            ),
            "e2b_h512_dense_bridge_pair_policy_enabled": pair_policy_enabled,
            "e2b_h512_grouped_attention_hits": h512_grouped_hits,
            "e2b_h512_grouped_attention_segments": h512_segments,
            "e2b_h512_grouped_attention_tiles": h512_tiles,
            "e2b_dense_attn_mlp_bridge_enabled": dense_bridge_enabled,
            "e2b_dense_attn_mlp_bridge_hits": dense_bridge_hits,
            "e2b_l4_sliding_prefill_required": require_e2b_l4_sliding_prefill,
            "e2b_l4_sliding_prefill_applicable": (
                e2b_l4_sliding_prefill_applicable
            ),
            "e2b_l4_sliding_prefill_enabled": e2b_l4_sliding_prefill_enabled,
            "e2b_l4_sliding_prefill_hits": e2b_l4_sliding_prefill_hits,
            "e2b_l4_full_prefill_expand_required": (
                require_e2b_l4_full_prefill_expand
            ),
            "e2b_l4_full_prefill_expand_applicable": (
                e2b_l4_full_prefill_expand_applicable
            ),
            "e2b_l4_full_prefill_expand_enabled": (
                e2b_l4_full_prefill_expand_enabled
            ),
            "e2b_l4_full_prefill_expand_hits": e2b_l4_full_prefill_expand_hits,
            "e2b_l4_full_prefill_expand_enabled_layers": (
                e2b_l4_full_prefill_expand_enabled_layers
            ),
            "e2b_l4_full_prefill_expand_errors": (
                e2b_l4_full_prefill_expand_errors
            ),
        },
        "expected_topology": expected_topology,
        "observed_topology": topology_stats[0] if topology_stats else None,
        "performance_gate": performance_gate,
        # These are performance-gated.  Zero means the profile requested the
        # path but its kernel gate rejected this exact model/batch/GPU shape.
        "selected_kernel_counters": {
            "gemma4_flat_fused_qkv_layers": _max_counter(
                decode_stats, "gemma4_flat_fused_qkv_layers"
            ),
            "fused_rope_attention": _max_counter(
                decode_stats, "fused_rope_attn_total_hits"
            ),
            "gemma4_fused_qkv_prefill": _max_counter(
                decode_stats, "gemma4_fused_qkv_prefill_hits"
            ),
            "gemma4_fused_attention_prepare": _max_counter(
                decode_stats, "gemma4_fused_attn_prepare_hits"
            ),
            "gemma4_vectorized_prefill_kv": _max_counter(
                decode_stats, "gemma4_batch_prefill_vectorized_kv_hits"
            ),
            "gemma4_e2b_l4_sliding_prefill": _max_counter(
                decode_stats, "gemma4_e2b_l4_sliding_prefill_hits"
            ),
            "gemma4_e2b_l4_full_prefill_expand": _max_counter(
                decode_stats, "gemma4_e2b_l4_full_prefill_expand_hits"
            ),
            "gemma4_flat_fused_gateup": _max_counter(
                decode_stats, "gemma4_flat_fused_gateup_hits"
            ),
            "gemma4_flat_deepfusion": _max_counter(
                decode_stats, "gemma4_flat_deepfusion_hits"
            ),
            "gemma4_dense_post_norm_chain": _max_counter(
                decode_stats, "gemma4_dense_post_norm_chain_decode_hits"
            ),
            "gemma4_dense_next_attn_norm": _max_counter(
                decode_stats, "gemma4_dense_next_attn_norm_decode_hits"
            ),
            "gemma4_dense_attn_mlp_bridge": dense_bridge_hits,
            "gemma4_e2b_h512_grouped_attention": h512_grouped_hits,
            "gemma4_fused_dual_ffn_norm_prefill": _max_counter(
                decode_stats, "gemma4_fused_dual_ffn_norm_prefill_hits"
            ),
            "gemma4_fused_add_ffn_norm_prefill": _max_counter(
                decode_stats, "gemma4_fused_add_ffn_norm_prefill_hits"
            ),
            "gemma4_fused_post_ffn_norm_prefill": _max_counter(
                decode_stats, "gemma4_fused_post_ffn_norm_prefill_hits"
            ),
            "paged_attention_generic_direct": _max_counter(
                paged_stats, "generic_direct_hits"
            ),
            "paged_attention_gqa2_direct": _max_counter(
                paged_stats, "gqa2_direct_hits"
            ),
        },
        "selected_lm_head": {
            "fused_rmsnorm_lm_head_argmax": any(
                bool(item.get("fused_rmsnorm_lm_head_argmax_use"))
                for item in decode_stats
            ),
            "fused_lm_head_argmax": any(
                bool(item.get("fused_lm_head_argmax_use")) for item in decode_stats
            ),
            "rmsnorm_skip_reasons": sorted(
                {
                    str(item.get("fused_rmsnorm_lm_head_argmax_skip_reason"))
                    for item in decode_stats
                    if item.get("fused_rmsnorm_lm_head_argmax_skip_reason")
                }
            ),
            "lm_head_skip_reasons": sorted(
                {
                    str(item.get("fused_lm_head_argmax_skip_reason"))
                    for item in decode_stats
                    if item.get("fused_lm_head_argmax_skip_reason")
                }
            ),
        },
        "errors": errors,
    }
    return report


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compact same-environment MegaGemm/vLLM GPU matrix"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--hardware-label", default="")
    parser.add_argument(
        "--variants",
        default="megagemm-fp16,vllm-fp16",
        help=(
            "Comma-separated: megagemm-fp16,vllm-fp16,megagemm-bf16,"
            "vllm-bf16,megagemm-int8"
        ),
    )
    parser.add_argument("--batch-sizes", default="1,8")
    parser.add_argument("--prompt-tokens", default="128,512,2048")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--megagemm-profile",
        choices=["auto", *MEGAGEMM_PROFILES],
        default="auto",
        help=(
            "MegaGemm child-process profile. auto selects the distinct "
            "gemma4-e2b-fast or gemma4-e4b-fast path and otherwise selects none."
        ),
    )
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=0)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--vllm-language-model-only", action="store_true")
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--out-dir", default="bench_results/publication_gpu")
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Reuse matching raw/summary artifacts in the run directory, "
            "audit them again, and execute only missing variants."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    megagemm_profile = resolve_megagemm_profile(args.megagemm_profile, args.model)
    effective_warmup = max(args.warmup, 3) if megagemm_profile != "none" else args.warmup
    effective_max_seq_len = args.max_seq_len
    if megagemm_profile in GEMMA4_PROFILE_REQUIREMENTS:
        max_prompt_tokens = max(
            int(part.strip())
            for part in args.prompt_tokens.split(",")
            if part.strip()
        )
        # Chat templates add a few tokens beyond the requested prompt length.
        # Keep a bounded safety margin, then align the KV capacity to 256 tokens.
        workload_capacity = max_prompt_tokens + args.max_new_tokens + 64
        workload_capacity = ((workload_capacity + 255) // 256) * 256
        if args.max_seq_len < workload_capacity:
            raise SystemExit(
                f"--max-seq-len={args.max_seq_len} is too small for this workload; "
                f"need at least {workload_capacity}"
            )
        effective_max_seq_len = min(args.max_seq_len, workload_capacity)
    hardware_label = args.hardware_label or auto_hardware_label()
    run_id = args.run_id or f"publication_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.out_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir = run_dir / run_id

    commands = [
        command_for(
            args,
            variant,
            hardware_label=hardware_label,
            run_dir=run_dir,
            run_id=run_id,
            warmup=effective_warmup,
            max_seq_len=effective_max_seq_len,
            megagemm_profile=megagemm_profile,
        )
        for variant in variants
    ]

    print("Publication GPU suite")
    print(f"  model:    {args.model}")
    print(f"  hardware: {hardware_label}")
    print(f"  variants: {', '.join(variant.name for variant in variants)}")
    print(f"  MegaGemm profile: {megagemm_profile}")
    print(f"  warmups: {effective_warmup}")
    print(f"  effective max sequence: {effective_max_seq_len}")
    print(f"  output:   {run_dir}")
    for command in commands:
        print(f"\n{shell_join(command)}")
    if args.dry_run:
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "run_id": run_id,
        "model": args.model,
        "hardware_label": hardware_label,
        "variants": [variant.name for variant in variants],
        "megagemm_profile": megagemm_profile,
        "megagemm_profile_env": profile_environment(megagemm_profile, args.model),
        "effective_warmup": effective_warmup,
        "effective_max_seq_len": effective_max_seq_len,
        "fastpath_audits": {},
        "args": vars(args),
        "commands": commands,
    }
    write_manifest(manifest_path, manifest)

    for variant, command in zip(variants, commands):
        print(f"\n=== {variant.name} ===", flush=True)
        variant_raw_path = raw_path(
            run_dir, run_id, hardware_label, variant
        )
        variant_summary_path = summary_path(
            run_dir, run_id, hardware_label, variant
        )
        existing_artifacts = bool(
            args.resume_existing
            and variant_raw_path.exists()
            and variant_summary_path.exists()
        )
        existing_errors: list[str] = []
        if existing_artifacts:
            validate_existing_variant_artifacts(
                variant_summary_path,
                args=args,
                variant=variant,
                hardware_label=hardware_label,
                warmup=effective_warmup,
                max_seq_len=effective_max_seq_len,
            )
            existing_errors = raw_artifact_errors(variant_raw_path)
        reuse_existing = bool(existing_artifacts and not existing_errors)
        if reuse_existing:
            print(
                f"  resume: reusing matching artifacts for {variant.name}",
                flush=True,
            )
        else:
            if existing_errors:
                print(
                    f"  resume: rerunning {variant.name}; existing artifact "
                    f"is incomplete ({' | '.join(existing_errors)})",
                    flush=True,
                )
            subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                env=child_environment(variant, megagemm_profile, args.model),
            )
            produced_errors = raw_artifact_errors(variant_raw_path)
            if produced_errors:
                raise RuntimeError(
                    f"{variant.name} did not produce a valid publication "
                    f"artifact: {' | '.join(produced_errors)}"
                )
        if (
            variant.backend == "megagemm"
            and megagemm_profile in GEMMA4_PROFILE_REQUIREMENTS
        ):
            audit = audit_gemma4_dense_fast_path(
                variant_raw_path, megagemm_profile
            )
            audit_path = run_dir / f"fastpath_audit_{variant.name}.json"
            audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
            manifest["fastpath_audits"][variant.name] = audit
            write_manifest(manifest_path, manifest)
            print(
                "  fast-path audit: "
                f"{audit['status']} ({audit_path})",
                flush=True,
            )
            if audit["status"] != "passed":
                raise RuntimeError(
                    "Gemma 4 dense fast-path audit failed: "
                    + " | ".join(audit.get("errors") or ["unknown error"])
                )

    for vllm in (item for item in variants if item.backend == "vllm"):
        right = summary_path(run_dir, run_id, hardware_label, vllm)
        for left in (
            item
            for item in variants
            if item.backend == "megagemm" and item.dtype == vllm.dtype
        ):
            left_path = summary_path(run_dir, run_id, hardware_label, left)
            if not left_path.exists() or not right.exists():
                continue
            csv_path = run_dir / f"compare_{left.name}_vs_{vllm.name}.csv"
            compare_command = [
                sys.executable,
                str(COMPARE),
                "--left",
                str(left_path),
                "--right",
                str(right),
                "--left-name",
                left.name,
                "--right-name",
                vllm.name,
                "--csv",
                str(csv_path),
            ]
            subprocess.run(compare_command, cwd=ROOT, check=True)

    archive_path = shutil.make_archive(str(run_dir), "zip", root_dir=run_dir)
    print(f"\nAttach this artifact to the benchmark report: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
