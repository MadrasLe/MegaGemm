"""CUDA-event stage profile for the Gemma 4 E2B L4 B8/P2048 prefill path.

This is a diagnostic benchmark, not an end-to-end throughput comparison.  It
warms one loaded MegaGemm engine without instrumentation, enables the existing
CUDA-event timing only for measured requests, and reports the stages that own
the prefill wall time.  Sliding H256 and full H512 attention are kept separate.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
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


ATTENTION_AGGREGATES = ("qkv", "attn_prepare", "attn_core", "o_proj")
GROUPS = {
    "setup": {
        "prefill_setup_ms",
        "embedding_ms",
        "per_layer_input_ms",
    },
    "attention": {
        "qkv_sliding_ms",
        "qkv_full_ms",
        "attn_prepare_sliding_ms",
        "attn_prepare_full_ms",
        "attn_core_sliding_ms",
        "attn_core_full_ms",
        "o_proj_sliding_ms",
        "o_proj_full_ms",
        "kv_write_ms",
    },
    "mlp": {
        "mlp_native_ms",
        "gate_up_ms",
        "activation_ms",
        "down_proj_ms",
    },
    "norm_residual_ple": {
        "gemma4_norms_ms",
        "gemma4_residual_scale_ms",
        "ple_ms",
    },
    "output": {
        "final_norm_ms",
        "lm_head_ms",
        "logit_cap_ms",
    },
}


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def canonical_stage_keys(stage_samples: list[dict[str, float]]) -> list[str]:
    keys = {
        key
        for sample in stage_samples
        for key in sample
        if key.endswith("_ms") and key != "total_ms"
    }
    for aggregate in ATTENTION_AGGREGATES:
        if f"{aggregate}_sliding_ms" in keys or f"{aggregate}_full_ms" in keys:
            keys.discard(f"{aggregate}_ms")
    return sorted(keys)


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise RuntimeError("no measured prefill samples")
    stage_samples = [dict(sample["prefill_stage_timing"]) for sample in samples]
    keys = canonical_stage_keys(stage_samples)
    stage_medians = {
        key: _median([float(sample.get(key, 0.0)) for sample in stage_samples])
        for key in keys
    }
    internal_prefill_ms = _median(
        [float(sample["internal_prefill_ms"]) for sample in samples]
    )
    tracked_total_ms = _median(
        [float(sample.get("total_ms", 0.0)) for sample in stage_samples]
    )

    ranking = [
        {
            "stage": key.removesuffix("_ms"),
            "median_ms": value,
            "tracked_fraction": value / tracked_total_ms if tracked_total_ms > 0 else 0.0,
            "prefill_fraction": value / internal_prefill_ms if internal_prefill_ms > 0 else 0.0,
        }
        for key, value in sorted(
            stage_medians.items(), key=lambda item: item[1], reverse=True
        )
    ]

    groups: list[dict[str, Any]] = []
    grouped_keys: set[str] = set()
    for group_name, configured_keys in GROUPS.items():
        present = sorted(configured_keys.intersection(keys))
        grouped_keys.update(present)
        per_sample = [
            sum(float(sample.get(key, 0.0)) for key in present)
            for sample in stage_samples
        ]
        value = _median(per_sample)
        groups.append(
            {
                "group": group_name,
                "median_ms": value,
                "tracked_fraction": (
                    value / tracked_total_ms if tracked_total_ms > 0 else 0.0
                ),
                "prefill_fraction": (
                    value / internal_prefill_ms if internal_prefill_ms > 0 else 0.0
                ),
                "stages": [key.removesuffix("_ms") for key in present],
            }
        )
    remaining = sorted(set(keys).difference(grouped_keys))
    if remaining:
        value = _median(
            [
                sum(float(sample.get(key, 0.0)) for key in remaining)
                for sample in stage_samples
            ]
        )
        groups.append(
            {
                "group": "other_tracked",
                "median_ms": value,
                "tracked_fraction": (
                    value / tracked_total_ms if tracked_total_ms > 0 else 0.0
                ),
                "prefill_fraction": (
                    value / internal_prefill_ms if internal_prefill_ms > 0 else 0.0
                ),
                "stages": [key.removesuffix("_ms") for key in remaining],
            }
        )
    groups.sort(key=lambda row: float(row["median_ms"]), reverse=True)

    top = ranking[0] if ranking else None
    route_audit = audit_production_prefill_routes(samples)
    return {
        "samples": len(samples),
        "engine_prefill_tokens_median": int(
            _median(
                [float(sample["prefill_stage_total_tokens"]) for sample in samples]
            )
        ),
        "internal_prefill_ms_median": internal_prefill_ms,
        "first_token_wall_ms_median": _median(
            [float(sample["wall_ms"]) for sample in samples]
        ),
        "tracked_cuda_ms_median": tracked_total_ms,
        "tracked_fraction_of_internal_prefill": (
            tracked_total_ms / internal_prefill_ms if internal_prefill_ms > 0 else 0.0
        ),
        "unattributed_internal_prefill_ms": internal_prefill_ms - tracked_total_ms,
        "stage_ranking": ranking,
        "groups": groups,
        "next_target": None if top is None else top["stage"],
        "route_audit": route_audit,
        "method": {
            "warmup_instrumentation": "disabled",
            "measurement_instrumentation": "CUDA events enabled",
            "scope": "one generated token; stage ranking describes prefill compute",
            "caveat": (
                "CUDA-event creation adds CPU overhead to internal prefill wall time; "
                "stage event durations measure device work and must not be used as E2E TPS"
            ),
        },
    }


def audit_production_prefill_routes(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prove that the profile measured the promoted E2B/L4 B8 paths."""
    routes = [
        dict(sample.get("prefill_routes") or {})
        for sample in samples
        if sample.get("prefill_routes")
    ]
    errors: list[str] = []
    if len(routes) != len(samples):
        errors.append("one or more samples did not expose prefill route counters")

    def values(key: str) -> list[Any]:
        return [route.get(key) for route in routes]

    expected = {
        "sliding_enabled_layers": 28,
        "full_expand_enabled_layers": 7,
        "fused_attn_prepare_enabled_layers": 15,
        "gated_activation_enabled_layers": 35,
    }
    for key, wanted in expected.items():
        found = values(key)
        if not found or any(int(value or 0) != wanted for value in found):
            errors.append(f"{key}={found}, expected {wanted} in every sample")

    expected_blocks = {"521": 256, "2057": 512}
    found_blocks = values("gated_activation_block_sizes")
    if not found_blocks or any(value != expected_blocks for value in found_blocks):
        errors.append(
            "gated_activation_block_sizes="
            f"{found_blocks}, expected {expected_blocks} in every sample"
        )

    expected_attn_launches = {
        "521x256": [4, 2, True],
        "521x512": [8, 2, True],
        "2057x256": [4, 2, True],
        "2057x512": [4, 2, True],
    }
    found_attn_launches = values("fused_attn_prepare_launches")
    if not found_attn_launches or any(
        value != expected_attn_launches for value in found_attn_launches
    ):
        errors.append(
            "fused_attn_prepare_launches="
            f"{found_attn_launches}, expected {expected_attn_launches} in every sample"
        )

    for key in (
        "sliding_hits",
        "full_expand_hits",
        "fused_attn_prepare_hits",
        "gated_activation_hits",
    ):
        found = [int(value or 0) for value in values(key)]
        if not found or min(found) <= 0:
            errors.append(f"{key} did not prove an active production route: {found}")
        elif len(found) > 1 and max(found) <= min(found):
            errors.append(f"{key} did not advance across samples: {found}")

    path_errors = sorted(
        {
            str(value)
            for value in values("full_expand_error")
            if value
        }
    )
    if path_errors:
        errors.extend(f"full attention path error: {value}" for value in path_errors)

    attention_prepare_failures = sorted(
        {str(value) for value in values("fused_attn_prepare_failure") if value}
    )
    if attention_prepare_failures:
        errors.extend(
            f"attention prepare path error: {value}"
            for value in attention_prepare_failures
        )

    gated_activation_failures = sorted(
        {str(value) for value in values("gated_activation_failure") if value}
    )
    if gated_activation_failures:
        errors.extend(
            f"gated activation path error: {value}"
            for value in gated_activation_failures
        )
    disabled_layers = [
        int(value or 0) for value in values("gated_activation_disabled_layers")
    ]
    if any(disabled_layers):
        errors.append(
            "gated activation runtime-disabled layers were observed: "
            f"{disabled_layers}"
        )

    return {
        "passed": not errors,
        "errors": errors,
        "samples": routes,
        "requirements": expected,
    }


def _measured_sample(runner, prompts: list[str], index: int) -> dict[str, Any]:
    result = runner(prompts, 1)
    expected = len(prompts)
    generated = int(result["generated_tokens"])
    if generated != expected:
        raise RuntimeError(f"generated {generated} tokens; expected {expected}")
    scheduler = dict((result.get("extra") or {}).get("scheduler_stats") or {})
    runtime = dict((result.get("extra") or {}).get("decode_runtime_stats") or {})
    stage = scheduler.get("prefill_stage_timing")
    if not isinstance(stage, dict) or not stage:
        raise RuntimeError(
            "MegaGemm did not expose prefill_stage_timing; instrumentation is inactive"
        )
    return {
        "repeat": index,
        "wall_ms": float(result["elapsed_s"]) * 1000.0,
        "internal_prefill_ms": float(scheduler.get("prefill_time_ms") or 0.0),
        "internal_decode_ms": float(scheduler.get("decode_time_ms") or 0.0),
        "prefill_stage_chunks": int(scheduler.get("prefill_stage_chunks") or 0),
        "prefill_stage_total_tokens": int(
            scheduler.get("prefill_stage_total_tokens") or 0
        ),
        "prefill_stage_timing": {
            key: float(value)
            for key, value in stage.items()
            if key.endswith("_ms")
        },
        "prefill_routes": {
            "sliding_enabled_layers": int(
                runtime.get("gemma4_e2b_l4_sliding_prefill_enabled_layers") or 0
            ),
            "sliding_hits": int(
                runtime.get("gemma4_e2b_l4_sliding_prefill_hits") or 0
            ),
            "full_expand_enabled_layers": int(
                runtime.get("gemma4_e2b_l4_full_prefill_expand_enabled_layers") or 0
            ),
            "full_expand_hits": int(
                runtime.get("gemma4_e2b_l4_full_prefill_expand_hits") or 0
            ),
            "full_expand_error": str(
                runtime.get("gemma4_e2b_l4_full_prefill_expand_error") or ""
            ),
            "fused_attn_prepare_enabled_layers": int(
                runtime.get(
                    "gemma4_e2b_b8_fused_attn_prepare_enabled_layers"
                )
                or 0
            ),
            "fused_attn_prepare_hits": int(
                runtime.get("gemma4_e2b_b8_fused_attn_prepare_hits") or 0
            ),
            "fused_attn_prepare_launches": {
                str(key): list(value)
                for key, value in dict(
                    runtime.get("gemma4_e2b_b8_fused_attn_prepare_launches")
                    or {}
                ).items()
            },
            "fused_attn_prepare_failure": str(
                runtime.get("gemma4_e2b_b8_fused_attn_prepare_failure") or ""
            ),
            "gated_activation_enabled_layers": int(
                runtime.get(
                    "gemma4_e2b_b8_prefill_gated_activation_enabled_layers"
                )
                or 0
            ),
            "gated_activation_block_sizes": {
                str(key): int(value)
                for key, value in dict(
                    runtime.get(
                        "gemma4_e2b_b8_prefill_gated_activation_block_sizes"
                    )
                    or {}
                ).items()
            },
            "gated_activation_hits": int(
                runtime.get("gemma4_e2b_b8_prefill_gated_activation_hits") or 0
            ),
            "gated_activation_disabled_layers": int(
                runtime.get(
                    "gemma4_e2b_b8_prefill_gated_activation_disabled_layers"
                )
                or 0
            ),
            "gated_activation_failure": str(
                runtime.get("gemma4_e2b_b8_prefill_gated_activation_failure")
                or ""
            ),
            "implicit_causal_batches": int(
                runtime.get("gemma4_implicit_causal_prefill_batches") or 0
            ),
            "vectorized_kv_hits": int(
                runtime.get("gemma4_batch_prefill_vectorized_kv_hits") or 0
            ),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import megagemm.models.llama as llama_module

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    gpu_name = torch.cuda.get_device_name(0)
    if "L4" not in gpu_name.upper():
        raise RuntimeError(f"this profile is delimited to NVIDIA L4, got {gpu_name}")

    args.backend = "megagemm"
    # The shared matrix runner namespace carries fields for every backend.
    # This diagnostic always dispatches MegaGemm and never imports a competitor.
    args.vllm_gpu_memory_utilization = 0.0
    profile = _configure_megagemm_profile(args.model)
    runtime_args = _runner_args(args)
    tokenizer = matrix.load_tokenizer(
        args.tokenizer or args.model,
        local_files_only=args.local_files_only,
    )
    prompts, prompt_tokens_actual = matrix.build_prompts(
        tokenizer,
        args.batch_size,
        args.prompt_tokens,
    )

    print("Gemma 4 E2B prefill stage profile", flush=True)
    print(f"  gpu:           {gpu_name}", flush=True)
    print(f"  model:         {args.model}", flush=True)
    print(f"  batch:         {args.batch_size}", flush=True)
    print(f"  prompt:        {args.prompt_tokens} requested", flush=True)
    print(f"  prompt actual: {prompt_tokens_actual} total", flush=True)
    print(f"  warmups:       {args.warmups} (timing disabled)", flush=True)
    print(f"  measurements:  {args.repeats} (CUDA events enabled)", flush=True)

    runner = matrix.make_runner(runtime_args, tokenizer)
    for index in range(1, args.warmups + 1):
        print(f"Warmup {index}/{args.warmups}", flush=True)
        result = runner(prompts, 1)
        if int(result["generated_tokens"]) != args.batch_size:
            raise RuntimeError("warmup did not generate one token per request")

    prior_flag = llama_module._PREFILL_TIMING
    prior_print = llama_module._PREFILL_TIMING_PRINT
    prior_env = os.environ.get("MEGAGEMM_PREFILL_TIMING")
    prior_print_env = os.environ.get("MEGAGEMM_PREFILL_TIMING_PRINT")
    samples: list[dict[str, Any]] = []
    try:
        llama_module._PREFILL_TIMING = True
        llama_module._PREFILL_TIMING_PRINT = False
        os.environ["MEGAGEMM_PREFILL_TIMING"] = "1"
        os.environ["MEGAGEMM_PREFILL_TIMING_PRINT"] = "0"
        for index in range(1, args.repeats + 1):
            print(f"Measure {index}/{args.repeats}", flush=True)
            sample = _measured_sample(runner, prompts, index)
            samples.append(sample)
            print(
                f"  prefill={sample['internal_prefill_ms']:.2f}ms "
                f"tracked={sample['prefill_stage_timing'].get('total_ms', 0.0):.2f}ms "
                f"wall={sample['wall_ms']:.2f}ms",
                flush=True,
            )
    finally:
        llama_module._PREFILL_TIMING = prior_flag
        llama_module._PREFILL_TIMING_PRINT = prior_print
        if prior_env is None:
            os.environ.pop("MEGAGEMM_PREFILL_TIMING", None)
        else:
            os.environ["MEGAGEMM_PREFILL_TIMING"] = prior_env
        if prior_print_env is None:
            os.environ.pop("MEGAGEMM_PREFILL_TIMING_PRINT", None)
        else:
            os.environ["MEGAGEMM_PREFILL_TIMING_PRINT"] = prior_print_env

    summary = summarize_samples(samples)
    payload = {
        "status": "passed" if summary["route_audit"]["passed"] else "invalid",
        "benchmark": "gemma4_e2b_prefill_stage_profile",
        "model": args.model,
        "dtype": "bf16",
        "hardware_label": "1xl4",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "workload": {
            "batch_size": args.batch_size,
            "prompt_tokens_requested_per_request": args.prompt_tokens,
            "prompt_tokens_actual_total": prompt_tokens_actual,
            "max_new_tokens": 1,
        },
        "profile_environment": profile,
        "system": {
            "git": matrix.git_snapshot(),
            "gpu": matrix.gpu_snapshot(),
            "nvidia_smi": matrix.nvidia_smi_snapshot(),
            "packages": matrix.installed_package_versions(),
        },
        "samples": samples,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nGEMMA 4 E2B / L4 / BF16 — PREFILL STAGE PROFILE")
    print("stage                                  median ms   tracked %   prefill %")
    for row in summary["stage_ranking"]:
        print(
            f"{row['stage']:<38} {row['median_ms']:9.2f} "
            f"{row['tracked_fraction'] * 100.0:10.2f} "
            f"{row['prefill_fraction'] * 100.0:10.2f}"
        )
    print("\nGROUPS")
    for row in summary["groups"]:
        print(
            f"{row['group']:<24} {row['median_ms']:9.2f} ms "
            f"({row['prefill_fraction'] * 100.0:6.2f}% of prefill)"
        )
    print(
        "\nTOTAL "
        f"prefill={summary['internal_prefill_ms_median']:.2f}ms "
        f"tracked={summary['tracked_cuda_ms_median']:.2f}ms "
        f"unattributed={summary['unattributed_internal_prefill_ms']:.2f}ms"
    )
    print(f"ENGINE_PREFILL_TOKENS {summary['engine_prefill_tokens_median']}")
    print("ROUTE_AUDIT " + json.dumps(summary["route_audit"], ensure_ascii=False))
    print(f"NEXT_TARGET {summary['next_target']}")
    print("PREFILL_PROFILE " + json.dumps(summary, ensure_ascii=False))
    print(f"Wrote: {args.output}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=2304)
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size != 8 or args.prompt_tokens != 2048:
        raise SystemExit("this profile is fixed to the validated B8/P2048 workload")
    if args.warmups < 1 or args.repeats < 3:
        raise SystemExit("use at least one warmup and three measured samples")
    payload = run(args)
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
