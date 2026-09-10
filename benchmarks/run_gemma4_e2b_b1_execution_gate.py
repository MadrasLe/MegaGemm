#!/usr/bin/env python3
"""One-load small-batch/L4 execution-only A/B; no policy promotion.

Full-model natural greedy O1/O128 pairs are timed without Python instrumentation.
Graphs use decode_multi_step(num_steps=1), not a different LM-head/decode policy.
Cold setup is reported separately. Each case owns its scheduler/graph storage;
no state from a previous Colab session or result file is needed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_gemma4_e2b_b1_decode_frontier_gate import (
    _configure_environment, _run, _capture_lm_state, _counter_snapshot, _counter_delta,
)


@dataclass(frozen=True)
class Case:
    name: str
    reuse: bool = False
    graph: bool = False
    eager_control: bool = False
    unroll: bool = False
    runtime_policy: bool = False
    burst_steps: int = 8


CASES = (
    Case("production"),
    Case("eager_multistep_reuse", reuse=True),
    Case("eager_graph_body", reuse=True, graph=True, eager_control=True),
    Case("one_step_graph", reuse=True, graph=True),
    Case("unrolled_graph8", reuse=True, graph=True, unroll=True),
)

PROMOTION_CASES = (
    CASES[0],
    Case("promoted_b4_policy", reuse=True, graph=True, runtime_policy=True),
)

# B8 has a materially different launch/occupancy balance from B4.  Keep its
# frontier independent and compare only full-model paths, with one model load.
# The one-step graph cases vary the number of tokens returned to Python per
# scheduler iteration; unrolled_graph8 additionally captures all eight
# dependent model steps into one graph replay.
B8_FRONTIER_CASES = (
    Case("production_burst8", burst_steps=8),
    Case("eager_reuse_burst8", reuse=True, burst_steps=8),
    Case("one_step_graph_burst4", reuse=True, graph=True, burst_steps=4),
    Case("one_step_graph_burst8", reuse=True, graph=True, burst_steps=8),
    Case("one_step_graph_burst16", reuse=True, graph=True, burst_steps=16),
    Case("unrolled_graph8", reuse=True, graph=True, unroll=True, burst_steps=8),
)

B8_PROMOTION_CASES = (
    B8_FRONTIER_CASES[0],
    Case(
        "promoted_b8_policy",
        reuse=True,
        graph=True,
        runtime_policy=True,
        burst_steps=16,
    ),
)


def case_environment(case):
    return {
        "MEGAGEMM_REUSE_REQUEST_SCHEDULER": str(int(case.reuse)),
        "MEGAGEMM_DECODE_PREFER_STEP": "0",
        "MEGAGEMM_DECODE_CUDA_GRAPHS": str(int(case.graph)),
        "MEGAGEMM_DECODE_CUDA_GRAPHS_PREFER_STEP": str(int(case.graph)),
        "MEGAGEMM_DECODE_GRAPH_MULTI_STEP_BODY": str(int(case.graph)),
        "MEGAGEMM_DECODE_GRAPH_EAGER_CONTROL": str(int(case.eager_control)),
        "MEGAGEMM_DECODE_UNROLLED_GRAPH_BURST": str(int(case.unroll)),
        "MEGAGEMM_NATIVE_DECODE_GRAPH_BURST": "0",
        "MEGAGEMM_DECODE_CUDA_GRAPHS_SHARED_SHAPE_CACHE": "0",
        "MEGAGEMM_DECODE_CUDA_GRAPHS_SHAPE_CACHE": "1",
        "MEGAGEMM_DECODE_CUDA_GRAPHS_STABLE_MAX_BLOCKS": "1",
        "MEGAGEMM_DECODE_GRAPH_PERSISTENT_TOKEN_FEEDBACK": "1",
        "MEGAGEMM_DECODE_GRAPH_TOKEN_BURST": "1",
        "MEGAGEMM_MULTI_STEP_BURST_BATCH": str(int(case.burst_steps)),
        "MEGAGEMM_GEMMA4_E2B_B1_GRAPH_EXPERIMENT": "1",
        "MEGAGEMM_GEMMA4_E2B_SMALL_BATCH_GRAPH_EXPERIMENT": "1",
        "MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID": "-1",
    }


def activate_case_environment(case):
    controls = case_environment(case)
    if case.runtime_policy:
        # Remove every execution override installed by the experimental gate.
        # Scheduler/model resolution must now come solely from RuntimePolicy.
        for name in controls:
            os.environ.pop(name, None)
        return
    os.environ.update(controls)


def compare_generated_tokens(row, reference):
    """Describe natural-greedy parity without conflating it with runtime health."""
    candidate_rows = row.get("generated_ids")
    reference_rows = reference.get("generated_ids")
    exact_digest = bool(row.get("digest") and row.get("digest") == reference.get("digest"))
    if candidate_rows is None or reference_rows is None:
        return {
            "exact_match": exact_digest,
            "token_agreement": 1.0 if exact_digest else 0.0,
            "matching_tokens": None,
            "compared_tokens": None,
            "first_divergence": None,
            "candidate_digest": row.get("digest"),
            "reference_digest": reference.get("digest"),
        }
    matching = compared = 0
    first = None
    sequence_count = max(len(candidate_rows), len(reference_rows))
    for sequence_index in range(sequence_count):
        candidate = candidate_rows[sequence_index] if sequence_index < len(candidate_rows) else []
        expected = reference_rows[sequence_index] if sequence_index < len(reference_rows) else []
        width = max(len(candidate), len(expected))
        compared += width
        for token_index in range(width):
            actual = candidate[token_index] if token_index < len(candidate) else None
            wanted = expected[token_index] if token_index < len(expected) else None
            if actual == wanted:
                matching += 1
            elif first is None:
                first = {
                    "sequence_index": sequence_index,
                    "token_index": token_index,
                    "reference_token": wanted,
                    "candidate_token": actual,
                }
    return {
        "exact_match": bool(exact_digest and first is None),
        "token_agreement": matching / compared if compared else 1.0,
        "matching_tokens": matching,
        "compared_tokens": compared,
        "first_divergence": first,
        "candidate_digest": row.get("digest"),
        "reference_digest": reference.get("digest"),
    }


def validate_run(row, reference, tokens, *, batch_size=1, require_token_match=True):
    errors = []
    if (row.get("generated_tokens") != tokens * batch_size
            or row.get("lengths") != [tokens] * batch_size):
        errors.append("wrong generated token count")
    if require_token_match and not compare_generated_tokens(row, reference)["exact_match"]:
        errors.append("greedy token sequence differs from production")
    if row.get("engine_prompt_tokens") != reference.get("engine_prompt_tokens"):
        errors.append("effective prompt differs from production")
    if (row.get("engine_prompt_lengths") is not None
            and row.get("engine_prompt_lengths") != reference.get("engine_prompt_lengths")):
        errors.append("per-request effective prompts differ from production")
    if row.get("scheduler_stats", {}).get("benchmark_forced_token_id", -1) != -1:
        errors.append("forced tokens must be disabled")
    if not math.isfinite(row.get("elapsed_s", 0)) or row.get("elapsed_s", 0) <= 0:
        errors.append("invalid wall time")
    return errors


def effective_max_prompt_tokens(row, batch_size):
    """Return the per-sequence prompt length, never the batch aggregate."""
    explicit = row.get("engine_max_prompt_tokens")
    if explicit is not None:
        return int(explicit)
    lengths = row.get("engine_prompt_lengths")
    if lengths:
        return max(int(length) for length in lengths)
    return math.ceil(int(row["engine_prompt_tokens"]) / int(batch_size))


def audit_execution(case, row, *, steady):
    stats = row.get("scheduler_stats", {})
    graph = stats.get("decode_cuda_graphs", {})
    errors = []
    for key in ("failures", "unrolled_token_burst_failures"):
        if graph.get(key, 0):
            errors.append(f"{key}: {graph.get('last_failure') or graph.get('unrolled_token_burst_last_failure')}")
    if graph.get("native_token_bursts", 0):
        errors.append("unexpected native executor")
    if case.graph:
        if int(graph.get("token_burst_size", 0) or 0) != int(case.burst_steps):
            errors.append(
                f"wrong token burst size: {graph.get('token_burst_size')} "
                f"!= {case.burst_steps}"
            )
        if not graph.get("multi_step_body") or not graph.get("persistent_token_feedback_steps", 0):
            errors.append("production multi-step body/token feedback not exercised")
        if bool(graph.get("eager_control")) != case.eager_control:
            errors.append("wrong graph/eager control")
        if case.eager_control:
            if graph.get("captures", 0) or graph.get("replays", 0):
                errors.append("eager control unexpectedly used a graph")
        elif not graph.get("replays", 0):
            errors.append("graph did not replay (silent fallback)")
        if case.unroll and not graph.get("unrolled_token_burst_steps", 0):
            errors.append("unrolled graph did not execute")
        if not case.unroll and graph.get("unrolled_token_burst_steps", 0):
            errors.append("unexpected unrolled graph")
        if steady and any(graph.get(k, 0) for k in ("captures", "unrolled_token_burst_captures", "warmups", "physical_rebinds")):
            errors.append("capture/warmup/rebind occurred inside steady timing")
    else:
        if graph.get("captures", 0) or graph.get("replays", 0):
            errors.append("eager production unexpectedly used a graph")
        if not stats.get("decode_execution", {}).get("multi_step_batches", 0):
            errors.append("eager production multi-step path was not exercised")
    if steady and case.reuse and not graph.get("request_scheduler_reused"):
        errors.append("scheduler/graph owner was not reused")
    return errors


def summarize(samples, rejected, *, repeats, minimum_speedup, maximum_spread,
              cases=CASES, allow_rejected=False, correctness=None):
    correctness = correctness or {}
    rows = []
    for case in cases:
        found = [s for s in samples if s["case"] == case.name]
        clean = len(found) == repeats and not any(s["errors"] for s in found) and case.name not in rejected
        times = [s["incremental_decode_s"] for s in found]
        positive = bool(times) and all(math.isfinite(t) and t > 0 for t in times)
        spread = max(times) / min(times) if positive else None
        measurement_valid = clean and positive and spread <= maximum_spread
        correct = case.name not in correctness
        rows.append({
            "case": case.name, "valid": measurement_valid and correct,
            "measurement_valid": measurement_valid, "correct": correct,
            "status": "rejected" if case.name in rejected else "measured",
            "samples": len(found), "spread": spread,
            "decode_tps": statistics.median(s["decode_tps"] for s in found) if positive else None,
            "output_tps": statistics.median(s["long"]["output_tps"] for s in found) if found else None,
            "total_s": statistics.median(s["long"]["elapsed_s"] for s in found) if found else None,
            "errors": rejected.get(case.name, []) + correctness.get(case.name, [])
                      + [e for s in found for e in s["errors"]],
        })
    baseline = rows[0]
    for row in rows:
        row["speedup"] = row["decode_tps"] / baseline["decode_tps"] if row["decode_tps"] and baseline["decode_tps"] else None
    measured = [r for r in rows if r["measurement_valid"]]
    performance_winner = max(measured, key=lambda r: r["decode_tps"]) if measured else None
    eligible = [r for r in rows if r["valid"]]
    winner = max(eligible, key=lambda r: r["decode_tps"]) if eligible and baseline["valid"] else None
    # Both the isolated decode and the full request must improve.
    complete = bool(baseline["valid"] and all(
        r["measurement_valid"] or (
            allow_rejected and r["case"] != baseline["case"] and r["case"] in rejected
        )
        for r in rows
    ))
    wins = bool(complete and winner and winner["case"] != baseline["case"] and winner["speedup"] >= minimum_speedup
                and winner["output_tps"] / baseline["output_tps"] >= minimum_speedup)
    return {
        "valid": complete,
        "decision": "CANDIDATE_WINS" if wins else ("KEEP_PRODUCTION" if complete else "INVALID_GATE"),
        "winner": winner["case"] if wins else (baseline["case"] if baseline["valid"] else None),
        "performance_winner": performance_winner["case"] if performance_winner else None,
        "production_policy_changed": False, "rows": rows,
    }


@contextmanager
def layer_route_probe(model, signatures):
    """Warmup-only counter signature: Python counters do not increment on replay."""
    original = model._gemma4_flat_decode_layers
    def wrapped(*args, **kwargs):
        if signatures:
            return original(*args, **kwargs)
        before = _counter_snapshot(model)
        result = original(*args, **kwargs)
        signatures.append(_counter_delta(before, _counter_snapshot(model)))
        return result
    model._gemma4_flat_decode_layers = wrapped
    try:
        yield
    finally:
        del model._gemma4_flat_decode_layers


def python_call_audit(run):
    """Diagnostic pass only. Counts Python calls, NOT CUDA launches or speed."""
    counts = Counter()
    def profile(frame, event, arg):
        if event == "call":
            filename = frame.f_code.co_filename.replace("\\", "/")
            if "/megagemm/" in filename:
                counts[filename.split("/megagemm/", 1)[1] + ":" + frame.f_code.co_name] += 1
    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        row = run()
    finally:
        sys.setprofile(previous)
    return {"scope": "whole_request_including_prefill", "timing_usable": False,
            "python_calls": sum(counts.values()), "calls": dict(counts.most_common()),
            "decode_body_python_calls": sum(v for k, v in counts.items() if k.endswith(":_gemma4_flat_decode_layers")),
            "decode_multi_step_python_calls": sum(v for k, v in counts.items() if k.endswith(":decode_multi_step")),
            "generated_token_digest": row["digest"],
            "scheduler_stats": row["scheduler_stats"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2304)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument("--maximum-spread", type=float, default=1.06)
    parser.add_argument("--skip-python-audit", action="store_true")
    parser.add_argument("--kernel-candidates", action="store_true",
                        help="compare five kernel variants, all inside a one-step graph")
    parser.add_argument("--verify-promotion", action="store_true",
                        help="compare eager baseline with the default B4/B8 RuntimePolicy")
    parser.add_argument("--b8-frontier", action="store_true",
                        help="compare B8 full-model graph burst sizes 4/8/16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.warmups < 1 or args.repeats < 2 or args.max_new_tokens < 32
            or args.prompt_tokens < 1 or args.batch_size not in (1, 2, 4, 8)):
        parser.error("requires warmups>=1, repeats>=2, output>=32 and batch in 1,2,4,8")
    if args.kernel_candidates and args.batch_size != 1:
        parser.error("kernel candidates are specialized for batch_size=1")
    if sum(bool(value) for value in (
        args.kernel_candidates, args.verify_promotion, args.b8_frontier
    )) > 1:
        parser.error("kernel, promotion, and B8 frontier modes are separate gates")
    if args.verify_promotion and args.batch_size not in (4, 8):
        parser.error("the promoted RuntimePolicy is specialized for batch_size=4 or 8")
    if args.b8_frontier and args.batch_size != 8:
        parser.error("the B8 frontier requires batch_size=8")
    _configure_environment(args.model, -1)
    os.environ.update(case_environment(CASES[0]))
    from benchmarks import benchmark_inference_matrix as matrix
    import torch
    from megagemm.engine import InferenceEngine
    if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name().upper():
        raise SystemExit("This gate requires NVIDIA L4; no model was loaded.")
    experiment = None
    compiler_report = None
    if args.kernel_candidates:
        from benchmarks.gemma4_b1_graph_kernels import GraphKernelExperiment
        experiment = GraphKernelExperiment()
        compiler_report = experiment.compile_preflight()
        if not compiler_report["all_compiled"]:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"valid": False, "decision": "COMPILER_PREFLIGHT_FAILED",
                                              "model_loads": 0, "compiler_preflight": compiler_report}, indent=2), encoding="utf-8")
            print("Compiler preflight failed before model download:", args.output, flush=True)
            return 2
    cases = (
        experiment.cases
        if experiment
        else (
            (B8_PROMOTION_CASES if args.batch_size == 8 else PROMOTION_CASES)
            if args.verify_promotion
            else (B8_FRONTIER_CASES if args.b8_frontier else CASES)
        )
    )
    baseline_name = cases[0].name
    tokenizer = matrix.load_tokenizer(args.model)
    prompts, _ = matrix.build_prompts(tokenizer, args.batch_size, args.prompt_tokens)
    engine = InferenceEngine(args.model, dtype=torch.bfloat16, device="cuda", max_batch_size=args.batch_size,
                             max_seq_len=args.max_seq_len, num_blocks=0, block_size=16, kv_alloc="auto")
    model = engine.model
    if not model.decode_cuda_graph_eligible(num_seqs=args.batch_size, dtype=torch.bfloat16,
                                           device_type="cuda", device_name=torch.cuda.get_device_name()):
        raise RuntimeError("loaded model is not the exact E2B small-batch graph experiment topology")
    label = (
        "graph kernel"
        if experiment
        else (
            "production-policy verification"
            if args.verify_promotion
            else ("B8 execution frontier" if args.b8_frontier else "execution-only")
        )
    )
    print(f"E2B/L4/B{args.batch_size} {label} gate: {len(cases)} cases, ONE model load, natural greedy tokens", flush=True)
    print("Cold setup is separate; measured pairs contain no profiler or capture.", flush=True)
    samples, cold, rejected, schedulers, routes, audits = [], {}, {}, {}, {}, {}
    correctness, token_parity, case_references = {}, {}, {}
    method = {"model_loads": 1, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "dtype": "bf16", "prompt_tokens": args.prompt_tokens,
              "batch_size": args.batch_size,
              "output_tokens": args.max_new_tokens, "forced_token_id": -1,
              "metric": "paired (O_long - O1) wall time; separate end-to-end output TPS",
              "source": str(ROOT), "profile": {k: v for k, v in os.environ.items() if k.startswith("MEGAGEMM_")},
              "baseline": baseline_name, "kernel_candidates": bool(experiment),
              "verify_promotion": bool(args.verify_promotion),
              "b8_frontier": bool(args.b8_frontier),
              "case_burst_steps": {
                  case.name: int(getattr(case, "burst_steps", 8)) for case in cases
              }}
    def save():
        payload = {"method": method, "cold_setup": cold, "samples": samples, "rejected": rejected,
                   "correctness_failures": correctness, "token_parity": token_parity,
                   "layer_route_signatures": routes, "python_audits": audits,
                   "compiler_preflight": compiler_report,
                   "kernel_preflight": experiment.report if experiment else None,
                   **summarize(samples, rejected, repeats=args.repeats,
                               minimum_speedup=args.minimum_speedup, maximum_spread=args.maximum_spread,
                               cases=cases, allow_rejected=bool(experiment), correctness=correctness)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    def run_case(case, tokens):
        activate_case_environment(case)
        if experiment and experiment.ready:
            experiment.activate(model, case)
            # Make the scheduler's own compatibility signature variant-specific.
            os.environ["MEGAGEMM_B1_GRAPH_KERNEL_CASE"] = case.name
        engine._last_scheduler = schedulers.get(case.name) if case.reuse else None
        row = _run(engine, prompts, tokens)
        if case.reuse:
            schedulers[case.name] = engine._last_scheduler
        return row
    def record_token_parity(case, tokens, row):
        comparison = compare_generated_tokens(row, references[tokens])
        token_parity.setdefault(case.name, {})[str(tokens)] = comparison
        if not comparison["exact_match"]:
            first = comparison.get("first_divergence") or {}
            detail = (
                f"natural greedy diverges at token {first.get('token_index')} "
                f"(production={first.get('reference_token')}, candidate={first.get('candidate_token')}, "
                f"agreement={comparison['token_agreement']:.2%})"
            )
            correctness.setdefault(case.name, [])
            if detail not in correctness[case.name]:
                correctness[case.name].append(detail)
        return comparison
    # Bootstrap production/autopickers before snapshots or graph capture.
    run_case(CASES[0], 2)
    references = {t: run_case(CASES[0], t) for t in (1, args.max_new_tokens)}
    lm_state = _capture_lm_state(model)
    effective_max_prompt = effective_max_prompt_tokens(
        references[args.max_new_tokens], args.batch_size
    )
    if effective_max_prompt + args.max_new_tokens > args.max_seq_len:
        raise RuntimeError(
            "largest per-request prompt plus output exceeds max sequence length: "
            f"{effective_max_prompt}+{args.max_new_tokens}>{args.max_seq_len}"
        )
    if experiment:
        try:
            rejected.update(experiment.prepare(model, lm_state))
        except Exception as exc:
            rejected[baseline_name] = [f"kernel preparation failed: {type(exc).__name__}: {exc}"]
            save()
            raise
        save()
    def state_errors(case):
        if experiment:
            return experiment.audit_state(model, case)
        return [] if _capture_lm_state(model) == lm_state else ["LM-head selection changed"]
    for case in cases:
        if case.name in rejected:
            print(f"REJECT {case.name}: {rejected[case.name]}", flush=True)
            continue
        print(f"SETUP {case.name}", flush=True)
        routes[case.name] = []
        try:
            start = time.perf_counter()
            cold_before = _counter_snapshot(model) if experiment else None
            with layer_route_probe(model, routes[case.name]):
                first = run_case(case, args.max_new_tokens)
            if experiment:
                case_references[case.name] = {args.max_new_tokens: first}
                record_token_parity(case, args.max_new_tokens, first)
            errors = validate_run(
                first,
                references[args.max_new_tokens],
                args.max_new_tokens,
                batch_size=args.batch_size,
                require_token_match=not bool(experiment),
            )
            errors += audit_execution(case, first, steady=False)
            if experiment:
                errors += experiment.audit_routes(case, routes[case.name], routes.get(baseline_name, []))
                errors += experiment.audit_cold(case, _counter_delta(cold_before, _counter_snapshot(model)))
            elif not routes[case.name] or routes[case.name] != routes[baseline_name]:
                errors.append("layer kernel-route signature differs from production")
            errors += state_errors(case)
            cold[case.name] = {"setup_wall_s": time.perf_counter() - start, "first_request": first}
            for _ in range(args.warmups):
                for tokens in (1, args.max_new_tokens):
                    row = run_case(case, tokens)
                    if experiment:
                        own_reference = case_references[case.name].get(tokens)
                        if own_reference is None:
                            case_references[case.name][tokens] = row
                            own_reference = row
                            record_token_parity(case, tokens, row)
                        errors += validate_run(
                            row, own_reference, tokens, batch_size=args.batch_size
                        )
                    else:
                        errors += validate_run(
                            row, references[tokens], tokens, batch_size=args.batch_size
                        )
                    if tokens > 1:
                        errors += audit_execution(case, row, steady=True)
                    errors += state_errors(case)
            if errors:
                rejected[case.name] = sorted(set(errors))
                print(f"REJECT {case.name}: {rejected[case.name]}", flush=True)
        except Exception as exc:
            # CUDA capture failures can corrupt stream/model/KV state. Never
            # silently continue and report another case as a valid winner.
            rejected[case.name] = [f"{type(exc).__name__}: {exc}"]
            save()
            raise
        save()
    for repeat in range(args.repeats):
        order = cases[repeat % len(cases):] + cases[:repeat % len(cases)]
        for case in order:
            if case.name in rejected:
                continue
            measurements = {t: run_case(case, t) for t in ((1, args.max_new_tokens) if repeat % 2 == 0 else (args.max_new_tokens, 1))}
            short, long = measurements[1], measurements[args.max_new_tokens]
            validation_references = case_references[case.name] if experiment else references
            errors = validate_run(
                short, validation_references[1], 1, batch_size=args.batch_size
            ) + validate_run(
                long, validation_references[args.max_new_tokens], args.max_new_tokens,
                batch_size=args.batch_size,
            )
            errors += audit_execution(case, long, steady=True)
            errors += state_errors(case)
            duration = long["elapsed_s"] - short["elapsed_s"]
            if not math.isfinite(duration) or duration <= 0:
                errors.append("non-positive incremental decode time")
            row = {"case": case.name, "repeat": repeat + 1, "short": short, "long": long,
                   "incremental_decode_s": duration,
                   "decode_tps": args.batch_size * (args.max_new_tokens - 1) / duration if duration > 0 else 0,
                   "errors": errors}
            samples.append(row)
            print(f"{case.name} {repeat+1}/{args.repeats}: decode={row['decode_tps']:.2f} tok/s total={long['output_tps']:.2f} tok/s errors={errors}", flush=True)
            save()
    if not args.skip_python_audit:
        for case in cases:
            if case.name not in rejected:
                print(f"PYTHON CALL AUDIT (not a speed sample): {case.name}", flush=True)
                audits[case.name] = python_call_audit(lambda: run_case(case, args.max_new_tokens))
                audit = audits[case.name]
                print(f"  Python calls={audit['python_calls']} decode-body calls={audit['decode_body_python_calls']} "
                      f"multi-step calls={audit['decode_multi_step_python_calls']}", flush=True)
                if audit["generated_token_digest"] != references[args.max_new_tokens]["digest"]:
                    rejected[case.name] = ["Python audit request diverged from production tokens"]
                save()
    result = save()
    print("\nGRAPH KERNEL SUMMARY" if experiment else "\nEXECUTION SUMMARY", flush=True)
    for row in result["rows"]:
        print(json.dumps(row), flush=True)
    if experiment:
        print("TOKEN PARITY", flush=True)
        for case in cases:
            print(json.dumps({"case": case.name, **token_parity.get(case.name, {})}), flush=True)
    print("DECISION:", result["decision"], "winner:", result["winner"], flush=True)
    if experiment:
        print("PERFORMANCE WINNER:", result["performance_winner"], flush=True)
    print("RESULTADO:", args.output, flush=True)
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
