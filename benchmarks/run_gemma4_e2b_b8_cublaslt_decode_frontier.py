#!/usr/bin/env python3
"""One-load full-model cuBLASLt frontier for Gemma 4 E2B/L4/B8 decode.

The frontier tunes only the attention/PLE BF16 projections left on torch.mm.
Gate-up and down-projection are intentionally excluded because their exact B8
shapes already lost dedicated full-model gates.  Search and plan creation run
before CUDA Graph capture; the final decision is based on paired natural-token
P2048/O128 full-model measurements, never on the shape screen alone.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Callable


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
from benchmarks.run_gemma4_e2b_b8_compute_frontier import _graph_errors


Shape = tuple[int, int, int]
PROJECTION_ATTRS = (
    ("qkv", "qkv_wt"),
    ("q", "q_wt"),
    ("k", "k_wt"),
    ("v", "v_wt"),
    ("o", "o_wt"),
    ("ple_gate", "ple_gate_wt"),
    ("ple_proj", "ple_proj_wt"),
)


def shape_name(shape: Shape) -> str:
    return f"m{shape[0]}_k{shape[1]}_n{shape[2]}"


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def spread(values: list[float]) -> float:
    return max(values) / min(values) if values and min(values) > 0 else math.inf


def cuda_samples_us(
    fn: Callable[[], Any],
    *,
    warmups: int,
    repeats: int,
) -> list[float]:
    import torch

    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)
    return samples


def collect_shapes(model: Any, rows: int) -> dict[Shape, dict[str, Any]]:
    """Collect one real weight and the dynamic occurrence count per shape."""
    grouped: dict[Shape, dict[str, Any]] = {}
    for layer in getattr(model, "_flat_layer_weights", ()):
        for operation, attr in PROJECTION_ATTRS:
            wt = getattr(layer, attr, None)
            if wt is None:
                continue
            shape = (rows, int(wt.shape[0]), int(wt.shape[1]))
            entry = grouped.setdefault(
                shape,
                {
                    "shape": shape,
                    "wt": wt,
                    "occurrences_per_token": 0,
                    "operations": defaultdict(int),
                },
            )
            entry["occurrences_per_token"] += 1
            entry["operations"][operation] += 1
    return grouped


def numeric_metrics(reference: Any, candidate: Any) -> dict[str, Any]:
    import torch

    ref = reference.float()
    got = candidate.float()
    diff = (got - ref).abs()
    ref_norm = float(torch.linalg.vector_norm(ref).item())
    diff_norm = float(torch.linalg.vector_norm(got - ref).item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            ref.reshape(1, -1), got.reshape(1, -1)
        ).item()
    )
    return {
        "finite": bool(torch.isfinite(got).all().item()),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "relative_l2": diff_norm / max(ref_norm, 1e-12),
        "cosine": cosine,
    }


def tune_shapes(
    model: Any,
    *,
    maximum_algorithms: int,
    warmups: int,
    repeats: int,
    minimum_shape_speedup: float,
    maximum_spread: float,
) -> tuple[dict[Shape, int], dict[str, Any]]:
    import torch
    from megagemm.kernels.mlp_prefill_native import (
        HAS_CUBLASLT_BF16_LINEAR,
        cublaslt_bf16_algorithm_count_cuda,
        cublaslt_bf16_linear_cuda,
    )

    if not HAS_CUBLASLT_BF16_LINEAR:
        raise RuntimeError("focused cuBLASLt BF16 extension is unavailable")
    grouped = collect_shapes(model, 8)
    selected: dict[Shape, int] = {}
    report: dict[str, Any] = {}
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260913)

    for shape in sorted(grouped):
        rows, inner, output = shape
        entry = grouped[shape]
        wt = entry["wt"]
        raw_weight = wt.t()
        if not raw_weight.is_contiguous():
            raise RuntimeError(f"{shape_name(shape)} raw weight is not contiguous")
        x = torch.randn(
            rows,
            inner,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        baseline_out = torch.empty(rows, output, device="cuda", dtype=torch.bfloat16)
        candidate_out = torch.empty_like(baseline_out)

        def baseline() -> Any:
            return torch.mm(x, wt, out=baseline_out)

        with torch.inference_mode():
            baseline()
            reference = baseline_out.clone()
            baseline_before = cuda_samples_us(
                baseline, warmups=warmups, repeats=repeats
            )
            algorithm_count = cublaslt_bf16_algorithm_count_cuda(
                x, raw_weight, maximum_algorithms
            )
            algorithms: list[dict[str, Any]] = []
            for algorithm_index in range(algorithm_count):
                def candidate(index: int = algorithm_index) -> Any:
                    return cublaslt_bf16_linear_cuda(
                        x,
                        raw_weight,
                        None,
                        out=candidate_out,
                        algorithm_index=index,
                    )

                try:
                    candidate()
                    first = candidate_out.clone()
                    candidate()
                    repeat_exact = bool(torch.equal(first, candidate_out))
                    metrics = numeric_metrics(reference, candidate_out)
                    samples = cuda_samples_us(
                        candidate, warmups=warmups, repeats=repeats
                    )
                    error = None
                except Exception as exc:
                    repeat_exact = False
                    metrics = {
                        "finite": False,
                        "max_abs": math.inf,
                        "mean_abs": math.inf,
                        "relative_l2": math.inf,
                        "cosine": 0.0,
                    }
                    samples = []
                    error = f"{type(exc).__name__}: {exc}"
                correct = bool(
                    error is None
                    and repeat_exact
                    and metrics["finite"]
                    and metrics["cosine"] >= 0.9999
                    and metrics["relative_l2"] <= 0.01
                )
                algorithms.append(
                    {
                        "algorithm_index": algorithm_index,
                        "correct": correct,
                        "repeat_exact": repeat_exact,
                        "metrics": metrics,
                        "samples_us": samples,
                        "median_us": median(samples) if samples else math.inf,
                        "spread": spread(samples),
                        "error": error,
                    }
                )
            baseline_after = cuda_samples_us(
                baseline, warmups=warmups, repeats=repeats
            )

        baseline_samples = baseline_before + baseline_after
        baseline_us = min(median(baseline_before), median(baseline_after))
        eligible = [
            result
            for result in algorithms
            if result["correct"]
            and result["spread"] <= maximum_spread
            and result["samples_us"]
        ]
        winner = min(eligible, key=lambda item: item["median_us"]) if eligible else None
        winner_us = float(winner["median_us"]) if winner else math.inf
        winner_speedup = baseline_us / winner_us if winner_us > 0 else 0.0
        conservative_speedup = (
            min(baseline_samples) / max(winner["samples_us"])
            if winner and winner["samples_us"]
            else 0.0
        )
        promote = bool(
            winner
            and winner_speedup >= minimum_shape_speedup
            and conservative_speedup >= 1.0
        )
        if promote:
            selected[shape] = int(winner["algorithm_index"])
        occurrence_count = int(entry["occurrences_per_token"])
        report[shape_name(shape)] = {
            "shape": list(shape),
            "operations": dict(entry["operations"]),
            "occurrences_per_token": occurrence_count,
            "torch_mm_samples_us": baseline_samples,
            "torch_mm_median_us": baseline_us,
            "algorithm_count": algorithm_count,
            "algorithms": algorithms,
            "winner": int(winner["algorithm_index"]) if winner else None,
            "winner_us": winner_us,
            "speedup": winner_speedup,
            "conservative_speedup": conservative_speedup,
            "estimated_savings_us_per_token": (
                max(0.0, baseline_us - winner_us) * occurrence_count
                if promote
                else 0.0
            ),
            "selected": promote,
        }
        print(
            "SHAPE "
            + json.dumps(
                {
                    "name": shape_name(shape),
                    "ops": dict(entry["operations"]),
                    "torch_us": round(baseline_us, 3),
                    "winner": int(winner["algorithm_index"]) if winner else None,
                    "winner_us": round(winner_us, 3) if math.isfinite(winner_us) else None,
                    "speedup": round(winner_speedup, 4),
                    "conservative": round(conservative_speedup, 4),
                    "selected": promote,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    # Prime every selected plan on the exact shape before graph capture.
    for shape, algorithm_index in selected.items():
        entry = grouped[shape]
        wt = entry["wt"]
        x = torch.zeros(shape[0], shape[1], device="cuda", dtype=torch.bfloat16)
        out = torch.empty(shape[0], shape[2], device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            cublaslt_bf16_linear_cuda(
                x,
                wt.t(),
                None,
                out=out,
                algorithm_index=algorithm_index,
            )
    torch.cuda.synchronize()
    return selected, report


def apply_route(model: Any, enabled: bool, algorithms: dict[Shape, int]) -> None:
    model._gemma4_flat_cublaslt_decode_enabled = bool(enabled)
    model._gemma4_flat_cublaslt_decode_algorithms = dict(algorithms) if enabled else {}


def validate_row(
    row: dict[str, Any],
    reference: dict[str, Any],
    output_tokens: int,
) -> list[str]:
    errors: list[str] = []
    if int(row.get("generated_tokens") or 0) != 8 * output_tokens:
        errors.append("wrong generated-token count")
    if row.get("lengths") != [output_tokens] * 8:
        errors.append("wrong per-request output lengths")
    comparison = compare_generated_tokens(row, reference)
    if not comparison["exact_match"]:
        errors.append(
            "natural greedy tokens differ from production: "
            f"agreement={comparison['token_agreement']:.6f} "
            f"first={comparison['first_divergence']}"
        )
    if int((row.get("scheduler_stats") or {}).get("benchmark_forced_token_id", -1)) != -1:
        errors.append("forced tokens must remain disabled")
    if output_tokens > 1:
        errors.extend(_graph_errors(row))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2304)
    parser.add_argument("--maximum-algorithms", type=int, default=16)
    parser.add_argument("--tune-warmups", type=int, default=2)
    parser.add_argument("--tune-repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--minimum-shape-speedup", type=float, default=1.01)
    parser.add_argument("--minimum-decode-speedup", type=float, default=1.02)
    parser.add_argument("--minimum-output-speedup", type=float, default=1.01)
    parser.add_argument("--maximum-spread", type=float, default=1.08)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir")
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 1 or args.repeats < 3 or args.warmups < 1:
        parser.error("max-new-tokens must be >1, repeats >=3 and warmups >=1")

    _configure_environment(args.model, -1)
    import torch
    if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name().upper():
        raise SystemExit("This frontier requires an NVIDIA L4; no model was loaded.")
    from megagemm.engine import InferenceEngine

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
    prompts = matrix.build_prompts(engine.tokenizer, 8, args.prompt_tokens)[0]
    schedulers: dict[tuple[str, int], Any] = {}

    def run_variant(name: str, enabled: bool, algorithms: dict[Shape, int], tokens: int):
        apply_route(model, enabled, algorithms)
        key = (name, tokens)
        engine._last_scheduler = schedulers.get(key)
        row = _run(engine, prompts, tokens)
        schedulers[key] = engine._last_scheduler
        return row

    print("Gemma 4 E2B/L4/B8 cuBLASLt decode frontier", flush=True)
    print("  full-model gate: P2048/O128, natural greedy tokens", flush=True)
    print("  execution fixed: one-step CUDA Graph / scheduler burst 16", flush=True)
    print("  model loads: 1; competing-engine install: disabled", flush=True)

    # Prepare flat weights/buffers and capture immutable production references.
    production_short = run_variant("production", False, {}, 1)
    production_long = run_variant(
        "production", False, {}, args.max_new_tokens
    )
    effective = effective_max_prompt_tokens(production_long, 8)
    if effective + args.max_new_tokens > args.max_seq_len:
        raise RuntimeError(
            "effective prompt plus output exceeds max sequence length: "
            f"{effective}+{args.max_new_tokens}>{args.max_seq_len}"
        )

    selected, shape_report = tune_shapes(
        model,
        maximum_algorithms=args.maximum_algorithms,
        warmups=args.tune_warmups,
        repeats=args.tune_repeats,
        minimum_shape_speedup=args.minimum_shape_speedup,
        maximum_spread=args.maximum_spread,
    )
    selected_json = {shape_name(key): value for key, value in selected.items()}
    print("SELECTED " + json.dumps(selected_json, sort_keys=True), flush=True)

    cases = [("production", False)]
    if selected:
        cases.append(("cublaslt_shape_frontier", True))
    setup: dict[str, Any] = {}
    for name, enabled in cases:
        # Fresh scheduler per route: search and plan creation are already done.
        schedulers.pop((name, 1), None)
        schedulers.pop((name, args.max_new_tokens), None)
        if enabled:
            model._gemma4_flat_cublaslt_decode_hits = {}
            model._gemma4_flat_cublaslt_decode_runtime_disabled = False
            model._gemma4_flat_cublaslt_decode_failure = ""
        short_cold = run_variant(name, enabled, selected, 1)
        long_cold = run_variant(name, enabled, selected, args.max_new_tokens)
        short = long = None
        for _ in range(args.warmups):
            short = run_variant(name, enabled, selected, 1)
            long = run_variant(name, enabled, selected, args.max_new_tokens)
        assert short is not None and long is not None
        errors = validate_row(short, production_short, 1)
        errors.extend(validate_row(long, production_long, args.max_new_tokens))
        repeat_comparison = compare_generated_tokens(long, long_cold)
        if not repeat_comparison["exact_match"]:
            errors.append("candidate is not repeat-deterministic")
        runtime = model.decode_runtime_stats()
        if enabled:
            hits = runtime.get("gemma4_cublaslt_decode_hits") or {}
            if not hits or sum(int(value) for value in hits.values()) <= 0:
                errors.append("selected cuBLASLt route recorded no capture hits")
            if runtime.get("gemma4_cublaslt_decode_runtime_disabled"):
                errors.append(
                    "cuBLASLt route disabled itself: "
                    + str(runtime.get("gemma4_cublaslt_decode_failure") or "unknown")
                )
        setup[name] = {
            "errors": errors,
            "short": short,
            "long": long,
            "cold_long_comparison": repeat_comparison,
            "production_comparison": compare_generated_tokens(long, production_long),
            "runtime": runtime,
        }
        print(
            f"SETUP {name}: " + ("OK" if not errors else json.dumps(errors)),
            flush=True,
        )

    samples: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in cases}
    for repeat in range(args.repeats):
        ordered = cases if repeat % 2 == 0 else list(reversed(cases))
        for name, enabled in ordered:
            short = run_variant(name, enabled, selected, 1)
            long = run_variant(name, enabled, selected, args.max_new_tokens)
            errors = validate_row(short, production_short, 1)
            errors.extend(validate_row(long, production_long, args.max_new_tokens))
            decode_s = float(long["elapsed_s"]) - float(short["elapsed_s"])
            generated_decode = 8 * (args.max_new_tokens - 1)
            sample = {
                "repeat": repeat + 1,
                "short_elapsed_s": short["elapsed_s"],
                "long_elapsed_s": long["elapsed_s"],
                "output_tps": long["output_tps"],
                "incremental_decode_s": decode_s,
                "incremental_decode_tps": generated_decode / decode_s,
                "errors": errors,
                "digest": long["digest"],
                "graph": (long.get("scheduler_stats") or {}).get(
                    "decode_cuda_graphs", {}
                ),
            }
            samples[name].append(sample)
            print(
                f"PAIR {repeat + 1}/{args.repeats} {name}: "
                f"wall={long['elapsed_s']:.4f}s "
                f"output={long['output_tps']:.2f} tok/s "
                f"decode={sample['incremental_decode_tps']:.2f} tok/s "
                f"errors={len(errors)}",
                flush=True,
            )

    summary: dict[str, Any] = {}
    for name, _ in cases:
        rows = samples[name]
        output_values = [float(row["output_tps"]) for row in rows]
        decode_values = [float(row["incremental_decode_tps"]) for row in rows]
        summary[name] = {
            "median_output_tps": median(output_values),
            "median_incremental_decode_tps": median(decode_values),
            "output_spread": spread(output_values),
            "decode_spread": spread(decode_values),
            "errors": [error for row in rows for error in row["errors"]],
        }

    production = summary["production"]
    candidate = summary.get("cublaslt_shape_frontier")
    if candidate is None:
        decision = {
            "promote": False,
            "decision": "KEEP_PRODUCTION_NO_SHAPE_WINNERS",
            "decode_speedup": 1.0,
            "output_speedup": 1.0,
            "valid": not production["errors"],
        }
    else:
        decode_speedup = (
            candidate["median_incremental_decode_tps"]
            / production["median_incremental_decode_tps"]
        )
        output_speedup = (
            candidate["median_output_tps"] / production["median_output_tps"]
        )
        valid = bool(
            not setup["production"]["errors"]
            and not setup["cublaslt_shape_frontier"]["errors"]
            and not production["errors"]
            and not candidate["errors"]
            and production["decode_spread"] <= args.maximum_spread
            and candidate["decode_spread"] <= args.maximum_spread
        )
        promote = bool(
            valid
            and decode_speedup >= args.minimum_decode_speedup
            and output_speedup >= args.minimum_output_speedup
        )
        decision = {
            "promote": promote,
            "decision": (
                "PROMOTE_CUBLASLT_SHAPE_FRONTIER"
                if promote
                else "KEEP_PRODUCTION"
            ),
            "decode_speedup": decode_speedup,
            "output_speedup": output_speedup,
            "minimum_decode_speedup": args.minimum_decode_speedup,
            "minimum_output_speedup": args.minimum_output_speedup,
            "valid": valid,
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
            "prompt_tokens": args.prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "natural_greedy_tokens": True,
            "execution": "E2B/L4 B8 RuntimePolicy graph burst16",
            "excluded": ["gate_up", "down_proj"],
        },
        "selected_algorithms": selected_json,
        "shape_frontier": shape_report,
        "setup": setup,
        "samples": samples,
        "summary": summary,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    print("DECISION " + json.dumps(decision, sort_keys=True), flush=True)
    print(f"Wrote: {args.output}", flush=True)
    return 0 if decision["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
