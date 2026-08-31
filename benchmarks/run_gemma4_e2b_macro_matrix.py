"""Full-model regime map for Gemma 4 E2B on one NVIDIA L4.

Unlike the kernel micro-gates, this benchmark loads the model once and rotates
through a matrix of batch sizes, prompt lengths, and generation lengths.  The
1-token sample in every workload pair is used to separate first-token work from
incremental decode without relying exclusively on engine-private timers.  Each
matrix cell is an independent optimization regime; aggregate scores are only a
regression guard and never imply that one kernel/tile must serve every shape.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import benchmark_inference_matrix as matrix


DEFAULT_MODEL = "google/gemma-4-E2B-it"
AUDIT_FRAGMENTS = (
    "gemma4_e2b_l4",
    "gemma4_dense",
    "gemma4_flat",
    "gemma4_ple",
    "gemma4_cublaslt",
    "fused_lm_head",
    "fused_rmsnorm_lm_head",
)


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _geomean(values: list[float]) -> float:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not positive:
        return 0.0
    return float(math.exp(sum(math.log(value) for value in positive) / len(positive)))


def _parse_csv(raw: str) -> list[int]:
    values = matrix.parse_csv_ints(raw)
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate matrix value in {raw!r}")
    if any(value <= 0 for value in values):
        raise ValueError(f"matrix values must be positive: {raw!r}")
    return values


def _scenario_key(batch: int, prompt: int, output: int) -> str:
    return f"b{batch}/p{prompt}/o{output}"


def _pair_key(batch: int, prompt: int, output: int) -> str:
    return f"b{batch}/p{prompt}/o1-o{output}"


def _configure_profile(model: str, profile: str, forced_token_id: int) -> dict[str, str]:
    if profile == "production":
        from benchmarks.run_gemma4_e2b_phase_split import _configure_megagemm_profile

        configured = _configure_megagemm_profile(model)
    else:
        configured = {
            key: value
            for key, value in os.environ.items()
            if key.startswith("MEGAGEMM_")
        }
    os.environ["MEGAGEMM_BENCHMARK_TOKEN_DIGEST"] = "1"
    os.environ["MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID"] = str(forced_token_id)
    configured["MEGAGEMM_BENCHMARK_TOKEN_DIGEST"] = "1"
    configured["MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID"] = str(forced_token_id)
    return configured


def _runner_args(args: argparse.Namespace, batch_sizes: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        backend="megagemm",
        model=args.model,
        tokenizer=args.tokenizer,
        dtype="bf16",
        quantize=None,
        device="cuda",
        max_batch_size=max(batch_sizes),
        max_seq_len=args.max_seq_len,
        num_blocks=0,
        block_size=16,
        kv_alloc="auto",
        kv_offload=False,
        num_cpu_blocks=0,
        gpu_window=64,
        cache_dir=args.cache_dir,
        mgx_prefer_payload_cache=False,
        mgx_payload_cache_dir=None,
        ignore_eos=True,
        local_files_only=args.local_files_only,
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.9,
        vllm_max_model_len=args.max_seq_len,
        vllm_max_num_seqs=max(batch_sizes),
        vllm_max_num_batched_tokens=0,
        vllm_enforce_eager=False,
        vllm_language_model_only=True,
        vllm_disable_prefix_caching=True,
        vllm_disable_cudagraph_memory_profiler=False,
    )


def _audit(extra: dict[str, Any]) -> dict[str, Any]:
    runtime = extra.get("decode_runtime_stats")
    if not isinstance(runtime, dict):
        return {}
    selected: dict[str, Any] = {}
    for key, value in runtime.items():
        if any(fragment in key for fragment in AUDIT_FRAGMENTS):
            if isinstance(value, (str, int, float, bool)) or value is None:
                selected[key] = value
    paged = runtime.get("paged_decode_runtime")
    if isinstance(paged, dict):
        selected["paged_decode_runtime"] = {
            key: value
            for key, value in paged.items()
            if key.endswith("_hits")
            or key.endswith("_disabled")
            or key.endswith("_failure")
        }
    return selected


def _sample(
    runner,
    prompts: list[str],
    *,
    batch: int,
    prompt: int,
    prompt_actual: int,
    output: int,
    repeat: int,
    order_position: int,
) -> dict[str, Any]:
    matrix.sync_cuda()
    result = runner(prompts, output)
    elapsed_s = float(result["elapsed_s"])
    generated = int(result["generated_tokens"])
    expected = batch * output
    if generated != expected:
        raise RuntimeError(
            f"{_scenario_key(batch, prompt, output)} generated {generated}, "
            f"expected {expected} with ignore_eos"
        )
    extra = dict(result.get("extra") or {})
    scheduler = extra.get("scheduler_stats")
    if not isinstance(scheduler, dict):
        scheduler = {}
    prefill_ms = float(scheduler.get("prefill_time_ms") or 0.0)
    decode_ms = float(scheduler.get("decode_time_ms") or 0.0)
    return {
        "key": _scenario_key(batch, prompt, output),
        "batch_size": batch,
        "prompt_tokens_requested_per_request": prompt,
        "prompt_tokens_actual_total": prompt_actual,
        "max_new_tokens_per_request": output,
        "repeat": repeat,
        "order_position": order_position,
        "elapsed_s": elapsed_s,
        "generated_tokens": generated,
        "output_tps": generated / elapsed_s,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "internal_decode_tps": generated / (decode_ms / 1000.0) if decode_ms > 0 else 0.0,
        "generated_token_digest": extra.get("generated_token_digest"),
        "generated_token_lengths": extra.get("generated_token_lengths"),
        "forced_token_id": int(scheduler.get("benchmark_forced_token_id", -1)),
        "audit": _audit(extra),
    }


def summarize_samples(
    samples: list[dict[str, Any]],
    *,
    maximum_spread: float,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    by_workload_repeat: dict[tuple[int, int, int], dict[int, dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(str(sample["key"]), []).append(sample)
        pair_key = (
            int(sample["batch_size"]),
            int(sample["prompt_tokens_requested_per_request"]),
            int(sample["repeat"]),
        )
        by_workload_repeat.setdefault(pair_key, {})[
            int(sample["max_new_tokens_per_request"])
        ] = sample

    scenarios: dict[str, dict[str, Any]] = {}
    for key, rows in sorted(grouped.items()):
        elapsed = [float(row["elapsed_s"]) for row in rows]
        output_tps = [float(row["output_tps"]) for row in rows]
        digests = sorted(
            {str(row["generated_token_digest"]) for row in rows if row.get("generated_token_digest")}
        )
        median_elapsed = _median(elapsed)
        scenarios[key] = {
            "batch_size": int(rows[0]["batch_size"]),
            "prompt_tokens_requested_per_request": int(
                rows[0]["prompt_tokens_requested_per_request"]
            ),
            "prompt_tokens_actual_total": int(rows[0]["prompt_tokens_actual_total"]),
            "max_new_tokens_per_request": int(rows[0]["max_new_tokens_per_request"]),
            "samples": len(rows),
            "median_elapsed_s": median_elapsed,
            "median_output_tps": _median(output_tps),
            "median_prefill_ms": _median([float(row["prefill_ms"]) for row in rows]),
            "median_decode_ms": _median([float(row["decode_ms"]) for row in rows]),
            "median_internal_decode_tps": _median(
                [float(row["internal_decode_tps"]) for row in rows]
            ),
            "spread_ratio": max(elapsed) / min(elapsed) if min(elapsed) > 0 else float("inf"),
            "digests": digests,
            "digest_stable": len(digests) == 1,
            "forced_token_route_active": all(int(row["forced_token_id"]) >= 0 for row in rows),
            "last_audit": rows[-1].get("audit") or {},
            "policy_scope": {
                "kind": "measured_cell_only",
                "batch_size": int(rows[0]["batch_size"]),
                "prompt_tokens_requested_per_request": int(
                    rows[0]["prompt_tokens_requested_per_request"]
                ),
                "max_new_tokens_per_request": int(
                    rows[0]["max_new_tokens_per_request"]
                ),
                "extrapolation_allowed": False,
            },
        }

    paired: dict[str, dict[str, Any]] = {}
    for (batch, prompt, _repeat), output_rows in sorted(by_workload_repeat.items()):
        short = output_rows.get(1)
        if short is None:
            continue
        for output, long in output_rows.items():
            if output <= 1:
                continue
            key = _pair_key(batch, prompt, output)
            delta_s = float(long["elapsed_s"]) - float(short["elapsed_s"])
            paired.setdefault(key, {"deltas": [], "rates": [], "batch": batch, "prompt": prompt, "output": output})
            paired[key]["deltas"].append(delta_s)
            if delta_s > 0.0:
                paired[key]["rates"].append(batch * (output - 1) / delta_s)

    paired_summary: dict[str, dict[str, Any]] = {}
    for key, row in sorted(paired.items()):
        deltas = [float(value) for value in row["deltas"]]
        rates = [float(value) for value in row["rates"]]
        paired_summary[key] = {
            "batch_size": row["batch"],
            "prompt_tokens_requested_per_request": row["prompt"],
            "long_tokens_per_request": row["output"],
            "complete_pairs": len(deltas),
            "positive_pairs": len(rates),
            "median_incremental_decode_ms": _median(deltas) * 1000.0,
            "median_incremental_decode_tps": _median(rates),
        }

    unstable = sorted(
        key for key, row in scenarios.items() if float(row["spread_ratio"]) > maximum_spread
    )
    invalid_digest = sorted(
        key for key, row in scenarios.items() if not row["digest_stable"]
    )
    inactive_route = sorted(
        key for key, row in scenarios.items() if not row["forced_token_route_active"]
    )
    return {
        "scenarios": scenarios,
        "paired_decode": paired_summary,
        "aggregate": {
            "scenario_count": len(scenarios),
            "sample_count": len(samples),
            "measured_wall_s": sum(float(sample["elapsed_s"]) for sample in samples),
            "unstable_scenarios": unstable,
            "invalid_digest_scenarios": invalid_digest,
            "inactive_forced_token_scenarios": inactive_route,
            "valid": not unstable and not invalid_digest and not inactive_route,
        },
    }


def measure(args: argparse.Namespace) -> dict[str, Any]:
    batches = _parse_csv(args.batch_sizes)
    prompts_requested = _parse_csv(args.prompt_tokens)
    outputs = sorted(_parse_csv(args.output_tokens))
    if 1 not in outputs:
        raise ValueError("--output-tokens must include 1 for paired phase attribution")
    if max(prompts_requested) + max(outputs) > args.max_seq_len:
        raise ValueError("largest prompt/output pair exceeds --max-seq-len")

    profile = _configure_profile(args.model, args.profile, args.forced_token_id)
    tokenizer = matrix.load_tokenizer(
        args.tokenizer or args.model,
        local_files_only=args.local_files_only,
    )
    prompt_cache: dict[tuple[int, int], tuple[list[str], int]] = {}
    for batch in batches:
        for prompt in prompts_requested:
            prompt_cache[(batch, prompt)] = matrix.build_prompts(tokenizer, batch, prompt)

    print("Gemma 4 E2B/L4 full-model macro matrix", flush=True)
    print(f"  batches: {batches}", flush=True)
    print(f"  prompts: {prompts_requested}", flush=True)
    print(f"  outputs: {outputs}", flush=True)
    print(f"  scenarios: {len(batches) * len(prompts_requested) * len(outputs)}", flush=True)
    print("  model loads: 1", flush=True)
    runner = matrix.make_runner(_runner_args(args, batches), tokenizer)

    for batch in batches:
        for prompt in prompts_requested:
            prompts, actual = prompt_cache[(batch, prompt)]
            for output in outputs:
                for warmup in range(args.warmups):
                    print(
                        f"Warmup b={batch} p={prompt} o={output} "
                        f"{warmup + 1}/{args.warmups}",
                        flush=True,
                    )
                    _sample(
                        runner,
                        prompts,
                        batch=batch,
                        prompt=prompt,
                        prompt_actual=actual,
                        output=output,
                        repeat=-(warmup + 1),
                        order_position=0,
                    )

    base_schedule = [
        (batch, prompt, output)
        for batch in batches
        for prompt in prompts_requested
        for output in outputs
    ]
    samples: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        schedule = list(base_schedule)
        if repeat % 2 == 0:
            schedule.reverse()
        rotation = (repeat - 1) % len(schedule)
        schedule = schedule[rotation:] + schedule[:rotation]
        for position, (batch, prompt, output) in enumerate(schedule, start=1):
            prompts, actual = prompt_cache[(batch, prompt)]
            print(
                f"Run {position}/{len(schedule)} repeat={repeat}/{args.repeats} "
                f"b={batch} p={prompt} o={output}",
                flush=True,
            )
            sample = _sample(
                runner,
                prompts,
                batch=batch,
                prompt=prompt,
                prompt_actual=actual,
                output=output,
                repeat=repeat,
                order_position=position,
            )
            samples.append(sample)
            print(
                f"  elapsed={sample['elapsed_s']:.4f}s "
                f"output={sample['output_tps']:.2f} tok/s "
                f"prefill={sample['prefill_ms']:.1f}ms "
                f"decode={sample['decode_ms']:.1f}ms",
                flush=True,
            )

    summary = summarize_samples(samples, maximum_spread=args.maximum_spread)
    payload = {
        "benchmark": "gemma4_e2b_l4_macro_matrix",
        "schema_version": 1,
        "label": args.label,
        "model": args.model,
        "dtype": "bf16",
        "hardware_label": "1xl4",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "matrix": {
            "batch_sizes": batches,
            "prompt_tokens": prompts_requested,
            "output_tokens": outputs,
        },
        "method": {
            "model_loads": 1,
            "warmup_per_scenario": args.warmups,
            "repeats": args.repeats,
            "schedule": "alternating-direction rotated full matrix",
            "incremental_decode": "paired wall-time delta against o1 in the same repeat",
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
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nMACRO SUMMARY")
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))
    print(f"Wrote: {args.output}")
    return payload


def compare_payloads(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    minimum_geomean_speedup: float,
    maximum_regression: float,
    final_dispatch_gate: bool = False,
) -> dict[str, Any]:
    for key in ("benchmark", "schema_version", "model", "dtype", "hardware_label", "matrix"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"incomparable {key}: {baseline.get(key)!r} != {candidate.get(key)!r}")
    base_scenarios = baseline["summary"]["scenarios"]
    cand_scenarios = candidate["summary"]["scenarios"]
    if set(base_scenarios) != set(cand_scenarios):
        raise ValueError("baseline and candidate scenario sets differ")

    base_pairs = baseline["summary"]["paired_decode"]
    cand_pairs = candidate["summary"]["paired_decode"]
    rows: list[dict[str, Any]] = []
    digest_mismatches: list[str] = []
    for key in sorted(base_scenarios):
        base = base_scenarios[key]
        cand = cand_scenarios[key]
        output = int(base["max_new_tokens_per_request"])
        output_speedup = float(cand["median_output_tps"]) / float(base["median_output_tps"])
        if output == 1:
            primary_metric = "first_token_wall"
            primary_speedup = float(base["median_elapsed_s"]) / float(cand["median_elapsed_s"])
        else:
            pair_key = _pair_key(
                int(base["batch_size"]),
                int(base["prompt_tokens_requested_per_request"]),
                output,
            )
            base_rate = float(base_pairs[pair_key]["median_incremental_decode_tps"])
            cand_rate = float(cand_pairs[pair_key]["median_incremental_decode_tps"])
            primary_metric = "paired_incremental_decode_tps"
            primary_speedup = cand_rate / base_rate if base_rate > 0.0 else 0.0
        digest_match = base.get("digests") == cand.get("digests") and bool(base.get("digests"))
        if not digest_match:
            digest_mismatches.append(key)
        if not digest_match:
            recommendation = "invalid_result"
        elif primary_speedup >= minimum_geomean_speedup:
            recommendation = "candidate_for_shape_dispatch"
        elif primary_speedup < 1.0 - maximum_regression:
            recommendation = "retain_baseline_for_shape"
        else:
            recommendation = "insufficient_margin"
        rows.append(
            {
                "key": key,
                "batch_size": base["batch_size"],
                "prompt_tokens": base["prompt_tokens_requested_per_request"],
                "output_tokens": output,
                "primary_metric": primary_metric,
                "primary_speedup": primary_speedup,
                "output_speedup": output_speedup,
                "digest_match": digest_match,
                "recommendation": recommendation,
            }
        )

    speedups = [float(row["primary_speedup"]) for row in rows]
    regressions = sorted(
        row["key"] for row in rows if float(row["primary_speedup"]) < 1.0 - maximum_regression
    )
    valid_inputs = bool(
        baseline["summary"]["aggregate"]["valid"]
        and candidate["summary"]["aggregate"]["valid"]
        and not digest_mismatches
        and all(value > 0.0 for value in speedups)
    )
    geometric_speedup = _geomean(speedups)
    full_dispatch_qualifies = bool(
        valid_inputs
        and geometric_speedup >= minimum_geomean_speedup
        and not regressions
    )
    if not valid_inputs:
        decision = "INVALID_MACRO_GATE"
    elif not final_dispatch_gate:
        decision = "RECORD_SCENARIO_RESULTS"
    elif full_dispatch_qualifies:
        decision = "PROMOTE_SHAPE_DISPATCH"
    else:
        decision = "KEEP_CURRENT_DISPATCH"
    candidate_cells = sorted(
        row["key"]
        for row in rows
        if row["recommendation"] == "candidate_for_shape_dispatch"
    )
    baseline_cells = sorted(
        row["key"]
        for row in rows
        if row["recommendation"] == "retain_baseline_for_shape"
    )
    inconclusive_cells = sorted(
        row["key"]
        for row in rows
        if row["recommendation"] == "insufficient_margin"
    )
    return {
        "benchmark": "gemma4_e2b_l4_macro_comparison",
        "baseline_label": baseline.get("label"),
        "candidate_label": candidate.get("label"),
        "decision": decision,
        "decision_mode": "final_dispatch_gate" if final_dispatch_gate else "regime_record",
        "valid": valid_inputs,
        "full_dispatch_qualifies": full_dispatch_qualifies,
        "geometric_mean_speedup": geometric_speedup,
        "median_speedup": _median(speedups),
        "minimum_speedup": min(speedups) if speedups else 0.0,
        "maximum_speedup": max(speedups) if speedups else 0.0,
        "wins": sum(value > 1.0 for value in speedups),
        "ties": sum(value == 1.0 for value in speedups),
        "losses": sum(value < 1.0 for value in speedups),
        "thresholds": {
            "minimum_geometric_mean_speedup": minimum_geomean_speedup,
            "maximum_allowed_regression": maximum_regression,
        },
        "digest_mismatches": digest_mismatches,
        "regressions": regressions,
        "policy_map": {
            "candidate_cells": candidate_cells,
            "baseline_cells": baseline_cells,
            "inconclusive_cells": inconclusive_cells,
            "rule": (
                "results apply only to the exact measured cell; implement an explicit "
                "shape/sequence guard and fall back to the current policy elsewhere"
            ),
        },
        "scenarios": rows,
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = compare_payloads(
        baseline,
        candidate,
        minimum_geomean_speedup=args.minimum_geomean_speedup,
        maximum_regression=args.maximum_regression,
        final_dispatch_gate=args.final_dispatch_gate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nGEMMA 4 E2B/L4 — FULL-MODEL MACRO COMPARISON")
    print(
        f"decision={result['decision']} geomean={result['geometric_mean_speedup']:.4f}x "
        f"min={result['minimum_speedup']:.4f}x wins={result['wins']} losses={result['losses']}"
    )
    print(f"Wrote: {args.output}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    measure_parser = commands.add_parser("measure")
    measure_parser.add_argument("--model", default=DEFAULT_MODEL)
    measure_parser.add_argument("--tokenizer")
    measure_parser.add_argument("--label", default="production")
    measure_parser.add_argument("--profile", choices=("production", "inherited"), default="production")
    measure_parser.add_argument("--batch-sizes", default="1,2,4,8")
    measure_parser.add_argument("--prompt-tokens", default="128,512,2048")
    measure_parser.add_argument("--output-tokens", default="1,16,128")
    measure_parser.add_argument("--warmups", type=int, default=1)
    measure_parser.add_argument("--repeats", type=int, default=3)
    measure_parser.add_argument("--maximum-spread", type=float, default=1.08)
    measure_parser.add_argument("--forced-token-id", type=int, default=42)
    measure_parser.add_argument("--max-seq-len", type=int, default=2304)
    measure_parser.add_argument("--cache-dir")
    measure_parser.add_argument("--local-files-only", action="store_true")
    measure_parser.add_argument("--output", type=Path, required=True)

    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--minimum-geomean-speedup", type=float, default=1.03)
    compare_parser.add_argument("--maximum-regression", type=float, default=0.03)
    compare_parser.add_argument(
        "--final-dispatch-gate",
        action="store_true",
        help=(
            "Evaluate an already-integrated shape-dispatch policy for promotion. "
            "Without this flag the comparison only records per-cell candidates."
        ),
    )
    compare_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "measure":
        if args.repeats < 2:
            raise SystemExit("--repeats must be at least 2")
        if args.warmups < 0:
            raise SystemExit("--warmups must be non-negative")
        measure(args)
    else:
        compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
