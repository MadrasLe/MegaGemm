#!/usr/bin/env python3
"""One-load full-model LM-head backend gate for Gemma 4 E2B/L4/B8.

Compares the promoted fused RMSNorm+argmax path against Tensor Core full-logit
GEMM with either the exact PyTorch softcap/argmax contract or the existing
fused softcap reduction.  Every route owns a separate CUDA Graph and is judged
on paired P2048/O1-O128 natural-greedy full-model measurements.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
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
from benchmarks.run_gemma4_e2b_b8_compute_frontier import _graph_errors


@dataclass(frozen=True)
class Case:
    name: str
    batch_cublas: bool = False
    fused_softcap_argmax: bool = False


CASES = (
    Case("production_fused"),
    Case("tensorcore_full_logits", batch_cublas=True),
    Case(
        "tensorcore_fused_softcap_argmax",
        batch_cublas=True,
        fused_softcap_argmax=True,
    ),
)


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def spread(values: list[float]) -> float:
    return max(values) / min(values) if values and min(values) > 0 else math.inf


def apply_case(llama_module: Any, case: Case) -> None:
    # The generic batch-cuBLAS switch remains required by the established A4B
    # path; this exact E2B/L4/B8 switch is experimental and disabled by default.
    llama_module._GEMMA4_BATCH_CUBLAS_LM_HEAD = True
    llama_module._GEMMA4_E2B_L4_B8_BATCH_CUBLAS_LM_HEAD_EXPERIMENT = bool(
        case.batch_cublas
    )
    llama_module._GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX = bool(
        case.fused_softcap_argmax
    )


def validate_row(
    row: dict[str, Any],
    reference: dict[str, Any],
    output_tokens: int,
) -> list[str]:
    errors: list[str] = []
    expected = 8 * output_tokens
    if int(row.get("generated_tokens") or 0) != expected:
        errors.append(f"generated {row.get('generated_tokens')}, expected {expected}")
    if row.get("lengths") != [output_tokens] * 8:
        errors.append("wrong per-request output lengths")
    if row.get("engine_prompt_lengths") != reference.get("engine_prompt_lengths"):
        errors.append("effective prompts differ from production")
    comparison = compare_generated_tokens(row, reference)
    if not comparison["exact_match"]:
        errors.append(
            "natural greedy tokens differ from production: "
            f"agreement={comparison['token_agreement']:.6f} "
            f"first={comparison['first_divergence']}"
        )
    if int((row.get("scheduler_stats") or {}).get("benchmark_forced_token_id", -1)) != -1:
        errors.append("forced-token benchmarking must remain disabled")
    if output_tokens > 1:
        errors.extend(_graph_errors(row))
    if not math.isfinite(float(row.get("elapsed_s") or 0.0)):
        errors.append("wall time is not finite")
    return errors


def route_errors(case: Case, runtime: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    cublas_hits = int(runtime.get("gemma4_batch_cublas_lm_head_hits", 0) or 0)
    softcap_hits = int(
        runtime.get("gemma4_batch_fused_softcap_argmax_hits", 0) or 0
    )
    fused_hits = int(runtime.get("fused_rmsnorm_lm_head_argmax_hits", 0) or 0)
    if case.batch_cublas:
        if cublas_hits <= 0:
            errors.append("Tensor Core full-logit route recorded no capture hits")
        if not runtime.get(
            "gemma4_e2b_l4_b8_batch_cublas_lm_head_experiment", False
        ):
            errors.append("E2B/L4/B8 batch-cuBLAS experiment is disabled")
        if case.fused_softcap_argmax:
            if softcap_hits <= 0:
                errors.append("fused softcap+argmax recorded no capture hits")
            if runtime.get("gemma4_batch_fused_softcap_argmax_disabled"):
                errors.append(
                    "fused softcap+argmax disabled itself: "
                    + str(
                        runtime.get("gemma4_batch_fused_softcap_argmax_error")
                        or "unknown"
                    )
                )
        elif softcap_hits:
            errors.append("unexpected fused softcap+argmax hits")
    else:
        if cublas_hits:
            errors.append("production unexpectedly used full-logit batch cuBLAS")
        if fused_hits <= 0 and not runtime.get("fused_lm_head_argmax_use", False):
            errors.append("production fused LM-head recorded no capture evidence")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2304)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--minimum-decode-speedup", type=float, default=1.02)
    parser.add_argument("--minimum-output-speedup", type=float, default=1.01)
    parser.add_argument("--maximum-spread", type=float, default=1.08)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir")
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 1 or args.warmups < 1 or args.repeats < 3:
        parser.error("max-new-tokens >1, warmups >=1 and repeats >=3 are required")

    _configure_environment(args.model, -1)
    import torch
    if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name().upper():
        raise SystemExit("This gate requires an NVIDIA L4; no model was loaded.")
    import megagemm.models.llama as llama_module
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

    def run_case(case: Case, tokens: int) -> dict[str, Any]:
        apply_case(llama_module, case)
        key = (case.name, tokens)
        engine._last_scheduler = schedulers.get(key)
        row = _run(engine, prompts, tokens)
        schedulers[key] = engine._last_scheduler
        return row

    print("Gemma 4 E2B/L4/B8 LM-head backend gate", flush=True)
    print("  workload: P2048/O1-O128, BF16, natural greedy tokens", flush=True)
    print("  execution: separate one-step CUDA Graph burst16 per route", flush=True)
    print("  cases: " + ", ".join(case.name for case in CASES), flush=True)
    print("  model loads: 1; competing-engine install: disabled", flush=True)

    production_case = CASES[0]
    production_short = run_case(production_case, 1)
    production_long = run_case(production_case, args.max_new_tokens)
    effective = effective_max_prompt_tokens(production_long, 8)
    if effective + args.max_new_tokens > args.max_seq_len:
        raise RuntimeError(
            "effective prompt plus output exceeds max sequence length: "
            f"{effective}+{args.max_new_tokens}>{args.max_seq_len}"
        )

    setup: dict[str, Any] = {}
    for case in CASES:
        schedulers.pop((case.name, 1), None)
        schedulers.pop((case.name, args.max_new_tokens), None)
        model._gemma4_batch_cublas_lm_head_hits = 0
        model._gemma4_batch_fused_softcap_argmax_hits = 0
        model._gemma4_batch_fused_softcap_argmax_disable = False
        model._gemma4_batch_fused_softcap_argmax_error = ""
        if not case.batch_cublas:
            model._fused_rmsnorm_lm_head_argmax_hits = 0
        short_cold = run_case(case, 1)
        long_cold = run_case(case, args.max_new_tokens)
        short = long = None
        for _ in range(args.warmups):
            short = run_case(case, 1)
            long = run_case(case, args.max_new_tokens)
        assert short is not None and long is not None
        runtime = model.decode_runtime_stats()
        errors = validate_row(short, production_short, 1)
        errors.extend(validate_row(long, production_long, args.max_new_tokens))
        errors.extend(route_errors(case, runtime))
        repeat_comparison = compare_generated_tokens(long, long_cold)
        if not repeat_comparison["exact_match"]:
            errors.append("route is not repeat-deterministic")
        setup[case.name] = {
            "errors": errors,
            "production_comparison": compare_generated_tokens(
                long, production_long
            ),
            "repeat_comparison": repeat_comparison,
            "runtime": runtime,
            "short": short,
            "long": long,
        }
        print(
            f"SETUP {case.name}: "
            + ("OK" if not errors else json.dumps(errors)),
            flush=True,
        )

    samples: dict[str, list[dict[str, Any]]] = {case.name: [] for case in CASES}
    orderings = (
        CASES,
        tuple(reversed(CASES)),
        (CASES[1], CASES[0], CASES[2]),
        (CASES[2], CASES[0], CASES[1]),
        (CASES[0], CASES[2], CASES[1]),
    )
    for repeat in range(args.repeats):
        order = orderings[repeat % len(orderings)]
        for case in order:
            short = run_case(case, 1)
            long = run_case(case, args.max_new_tokens)
            errors = validate_row(short, production_short, 1)
            errors.extend(validate_row(long, production_long, args.max_new_tokens))
            decode_s = float(long["elapsed_s"]) - float(short["elapsed_s"])
            decode_tokens = 8 * (args.max_new_tokens - 1)
            sample = {
                "repeat": repeat + 1,
                "short_elapsed_s": short["elapsed_s"],
                "long_elapsed_s": long["elapsed_s"],
                "output_tps": long["output_tps"],
                "incremental_decode_s": decode_s,
                "incremental_decode_tps": decode_tokens / decode_s,
                "errors": errors,
                "digest": long["digest"],
                "graph": (long.get("scheduler_stats") or {}).get(
                    "decode_cuda_graphs", {}
                ),
            }
            samples[case.name].append(sample)
            print(
                f"PAIR {repeat + 1}/{args.repeats} {case.name}: "
                f"wall={long['elapsed_s']:.4f}s "
                f"output={long['output_tps']:.2f} tok/s "
                f"decode={sample['incremental_decode_tps']:.2f} tok/s "
                f"errors={len(errors)}",
                flush=True,
            )

    summary: dict[str, Any] = {}
    for case in CASES:
        rows = samples[case.name]
        outputs = [float(row["output_tps"]) for row in rows]
        decodes = [float(row["incremental_decode_tps"]) for row in rows]
        summary[case.name] = {
            "median_output_tps": median(outputs),
            "median_incremental_decode_tps": median(decodes),
            "output_spread": spread(outputs),
            "decode_spread": spread(decodes),
            "errors": [error for row in rows for error in row["errors"]],
        }

    production = summary[production_case.name]
    rankings: list[dict[str, Any]] = []
    for case in CASES[1:]:
        result = summary[case.name]
        valid = bool(
            not setup[case.name]["errors"]
            and not result["errors"]
            and result["decode_spread"] <= args.maximum_spread
            and result["output_spread"] <= args.maximum_spread
        )
        rankings.append(
            {
                "case": case.name,
                "valid": valid,
                "decode_speedup": (
                    result["median_incremental_decode_tps"]
                    / production["median_incremental_decode_tps"]
                ),
                "output_speedup": (
                    result["median_output_tps"]
                    / production["median_output_tps"]
                ),
            }
        )
    winner = max(rankings, key=lambda item: item["decode_speedup"])
    promote = bool(
        winner["valid"]
        and winner["decode_speedup"] >= args.minimum_decode_speedup
        and winner["output_speedup"] >= args.minimum_output_speedup
    )
    production_healthy = bool(
        not setup[production_case.name]["errors"]
        and not production["errors"]
        and production["decode_spread"] <= args.maximum_spread
        and production["output_spread"] <= args.maximum_spread
    )
    decision = {
        "decision": (
            f"PROMOTE_{winner['case'].upper()}" if promote else "KEEP_PRODUCTION"
        ),
        "promote": promote,
        "winner": winner,
        "rankings": rankings,
        "minimum_decode_speedup": args.minimum_decode_speedup,
        "minimum_output_speedup": args.minimum_output_speedup,
        "production_healthy": production_healthy,
    }
    payload = {
        "status": "passed" if production_healthy else "failed",
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
            "execution": "separate E2B/L4 B8 CUDA Graph burst16 per route",
        },
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
    # Candidate rejection is a successful gate outcome.  Fail only if the
    # production control itself is invalid or unstable.
    return 0 if production_healthy else 2


if __name__ == "__main__":
    raise SystemExit(main())
