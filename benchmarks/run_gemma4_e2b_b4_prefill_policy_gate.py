#!/usr/bin/env python3
"""Loaded-model B4/P2048 prefill policy gate for Gemma 4 E2B on L4.

The gate keeps the promoted B8 path untouched.  It measures complete model
prefill/first-token latency while selecting a B4-only full-attention policy and
several B4-only sliding-attention launch geometries in one loaded engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
)


CASES: tuple[dict[str, Any], ...] = (
    {"name": "baseline", "full": False, "sliding": False, "config": None},
    {"name": "full_h512_only", "full": True, "sliding": False, "config": None},
    {
        "name": "sliding_h256_only_b8_geometry",
        "full": False,
        "sliding": True,
        "config": (4, 8, 64, 4, 2),
    },
    {
        "name": "combined_g1_bm32_bn64_w8_s2",
        "full": True,
        "sliding": True,
        "config": (1, 32, 64, 8, 2),
    },
    {
        "name": "combined_g2_bm16_bn64_w4_s2",
        "full": True,
        "sliding": True,
        "config": (2, 16, 64, 4, 2),
    },
    {
        "name": "combined_g2_bm16_bn64_w8_s2",
        "full": True,
        "sliding": True,
        "config": (2, 16, 64, 8, 2),
    },
    {
        "name": "combined_g4_bm8_bn64_w4_s2",
        "full": True,
        "sliding": True,
        "config": (4, 8, 64, 4, 2),
    },
    {
        "name": "combined_g4_bm8_bn128_w4_s2",
        "full": True,
        "sliding": True,
        "config": (4, 8, 128, 4, 2),
    },
)


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _attention_modules(engine, *, sliding: bool) -> list[Any]:
    modules = []
    for layer in engine.model.layers:
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            continue
        window = int(getattr(attention, "sliding_window", 0) or 0)
        head_dim = int(getattr(attention, "head_dim", 0) or 0)
        if sliding and window == 512 and head_dim == 256:
            modules.append(attention)
        elif not sliding and window <= 0 and head_dim == 512:
            modules.append(attention)
    return modules


def _set_case(
    full_modules: list[Any],
    sliding_modules: list[Any],
    case: dict[str, Any],
) -> None:
    for module in full_modules:
        module._gemma4_e2b_l4_full_prefill_expand_enabled = bool(case["full"])
        module._gemma4_e2b_l4_full_prefill_expand_error = ""
    for module in sliding_modules:
        module._gemma4_e2b_l4_sliding_prefill_enabled = bool(case["sliding"])
    config = case.get("config")
    if config is None:
        return
    group_heads, block_m, block_n, warps, stages = config
    os.environ.update(
        {
            "MEGAGEMM_GEMMA4_E2B_L4_B4_SLIDING_GROUP_HEADS": str(group_heads),
            "MEGAGEMM_GEMMA4_E2B_L4_B4_SLIDING_BLOCK_M": str(block_m),
            "MEGAGEMM_GEMMA4_E2B_L4_B4_SLIDING_BLOCK_N": str(block_n),
            "MEGAGEMM_GEMMA4_E2B_L4_B4_SLIDING_NUM_WARPS": str(warps),
            "MEGAGEMM_GEMMA4_E2B_L4_B4_SLIDING_NUM_STAGES": str(stages),
        }
    )


def _hits(modules: list[Any], attribute: str) -> int:
    return sum(int(getattr(module, attribute, 0)) for module in modules)


def _token_digest(engine) -> str:
    scheduler = engine._last_scheduler
    completed = sorted(scheduler._completed, key=lambda request: int(request.request_id))
    rows = [[int(token) for token in request.generated_ids] for request in completed]
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _sample(
    engine,
    full_modules: list[Any],
    sliding_modules: list[Any],
    prompts: list[str],
    *,
    case: dict[str, Any],
    repeat: int,
    position: int,
) -> dict[str, Any]:
    import torch
    from megagemm.kernels import paged_attention

    _set_case(full_modules, sliding_modules, case)
    paged_attention._GEMMA4_E2B_L4_SLIDING_PREFILL_FAILURE = ""
    before_full = _hits(full_modules, "_gemma4_e2b_l4_full_prefill_expand_hits")
    before_sliding = _hits(sliding_modules, "_gemma4_e2b_l4_sliding_prefill_hits")
    torch.cuda.synchronize()
    started = time.perf_counter()
    engine.generate_batch(
        prompts,
        max_new_tokens=1,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        ignore_eos=True,
        decode_outputs=False,
    )
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000.0
    scheduler = engine._last_scheduler
    stats = scheduler.get_stats()
    generated = int(stats.get("total_tokens") or 0)
    if generated != len(prompts):
        raise RuntimeError(f"generated {generated} tokens; expected {len(prompts)}")
    full_error = next(
        (
            str(module._gemma4_e2b_l4_full_prefill_expand_error)
            for module in full_modules
            if getattr(module, "_gemma4_e2b_l4_full_prefill_expand_error", "")
        ),
        "",
    )
    return {
        "case": str(case["name"]),
        "repeat": repeat,
        "position": position,
        "config": case.get("config"),
        "wall_ms": wall_ms,
        "prefill_ms": float(stats.get("prefill_time_ms") or 0.0),
        "full_hits_delta": _hits(
            full_modules,
            "_gemma4_e2b_l4_full_prefill_expand_hits",
        )
        - before_full,
        "sliding_hits_delta": _hits(
            sliding_modules,
            "_gemma4_e2b_l4_sliding_prefill_hits",
        )
        - before_sliding,
        "full_error": full_error,
        "sliding_error": str(
            paged_attention._GEMMA4_E2B_L4_SLIDING_PREFILL_FAILURE
        ),
        "token_digest": _token_digest(engine),
    }


def summarize(
    samples: list[dict[str, Any]],
    *,
    minimum_speedup: float,
    maximum_spread: float,
) -> dict[str, Any]:
    cases: dict[str, dict[str, Any]] = {}
    definitions = {str(case["name"]): case for case in CASES}
    for name, definition in definitions.items():
        selected = [row for row in samples if row["case"] == name]
        prefill = [float(row["prefill_ms"]) for row in selected]
        expected_full = 7 if definition["full"] else 0
        expected_sliding = 28 if definition["sliding"] else 0
        exact_hits = bool(
            selected
            and all(
                int(row["full_hits_delta"]) == expected_full
                and int(row["sliding_hits_delta"]) == expected_sliding
                for row in selected
            )
        )
        cases[name] = {
            "config": definition.get("config"),
            "samples": len(selected),
            "prefill_ms_median": _median(prefill),
            "prefill_ms_samples": prefill,
            "spread_ratio": max(prefill) / min(prefill) if prefill else float("inf"),
            "wall_ms_median": _median([float(row["wall_ms"]) for row in selected]),
            "full_hits": sum(int(row["full_hits_delta"]) for row in selected),
            "sliding_hits": sum(int(row["sliding_hits_delta"]) for row in selected),
            "exact_hits": exact_hits,
            "errors": sorted(
                {
                    str(error)
                    for row in selected
                    for error in (row["full_error"], row["sliding_error"])
                    if error
                }
            ),
            "digests": sorted({str(row["token_digest"]) for row in selected}),
        }

    all_digests = {str(row["token_digest"]) for row in samples}
    correct = len(all_digests) == 1
    baseline_ms = float(cases["baseline"]["prefill_ms_median"])
    eligible = [
        (name, row)
        for name, row in cases.items()
        if name.startswith("combined_")
        and row["exact_hits"]
        and not row["errors"]
        and float(row["spread_ratio"]) <= maximum_spread
        and len(row["digests"]) == 1
    ]
    eligible.sort(key=lambda item: float(item[1]["prefill_ms_median"]))
    winner_name, winner = eligible[0] if eligible else (None, None)
    winner_ms = float(winner["prefill_ms_median"]) if winner else 0.0
    speedup = baseline_ms / winner_ms if winner_ms > 0.0 else 0.0
    ready = bool(correct and winner and speedup >= minimum_speedup)
    return {
        "decision": (
            "IMPLEMENT_B4_POLICY_AND_RUN_MACRO_GATE"
            if ready
            else "KEEP_B4_BASELINE"
        ),
        "candidate_ready": ready,
        "winner": winner_name,
        "winner_config": winner.get("config") if winner else None,
        "baseline_prefill_ms": baseline_ms,
        "winner_prefill_ms": winner_ms,
        "speedup": speedup,
        "saved_ms": baseline_ms - winner_ms if winner else 0.0,
        "token_digest_exact": correct,
        "minimum_speedup": minimum_speedup,
        "maximum_spread": maximum_spread,
        "cases": cases,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from megagemm.engine import InferenceEngine

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    gpu = torch.cuda.get_device_name(0)
    if "L4" not in gpu.upper():
        raise RuntimeError(f"this gate requires NVIDIA L4, found {gpu}")

    profile = _configure_megagemm_profile(args.model)
    os.environ["MEGAGEMM_GEMMA4_E2B_L4_B4_PREFILL_EXPERIMENT"] = "1"
    os.environ["MEGAGEMM_BENCHMARK_TOKEN_DIGEST"] = "1"
    engine = InferenceEngine(
        args.model,
        dtype=torch.bfloat16,
        device="cuda",
        max_batch_size=4,
        max_seq_len=2304,
        num_blocks=0,
        block_size=16,
        kv_alloc="auto",
        cache_dir=args.cache_dir,
    )
    prompts, actual_tokens = matrix.build_prompts(engine.tokenizer, 4, 2048)
    full_modules = _attention_modules(engine, sliding=False)
    sliding_modules = _attention_modules(engine, sliding=True)
    if len(full_modules) != 7 or len(sliding_modules) != 28:
        raise RuntimeError(
            f"expected 7 full and 28 sliding layers, found "
            f"{len(full_modules)} and {len(sliding_modules)}"
        )

    print("Gemma 4 E2B/L4 B4/P2048 full-model prefill policy gate")
    print(f"  gpu: {gpu}")
    print(f"  model: {args.model}")
    print(f"  prompt tokens: {actual_tokens} total")
    print(f"  cases: {len(CASES)}")
    print("  model loads: 1")

    for case in CASES:
        for warmup in range(1, args.warmups + 1):
            print(f"Warmup case={case['name']} {warmup}/{args.warmups}", flush=True)
            _sample(
                engine,
                full_modules,
                sliding_modules,
                prompts,
                case=case,
                repeat=-warmup,
                position=0,
            )

    samples: list[dict[str, Any]] = []
    base_order = list(CASES)
    for repeat in range(1, args.repeats + 1):
        order = list(base_order if repeat % 2 else reversed(base_order))
        rotation = (repeat - 1) % len(order)
        order = order[rotation:] + order[:rotation]
        for position, case in enumerate(order, start=1):
            print(
                f"Measure repeat={repeat}/{args.repeats} "
                f"position={position}/{len(order)} case={case['name']}",
                flush=True,
            )
            row = _sample(
                engine,
                full_modules,
                sliding_modules,
                prompts,
                case=case,
                repeat=repeat,
                position=position,
            )
            samples.append(row)
            print(
                f"  prefill={row['prefill_ms']:.2f}ms wall={row['wall_ms']:.2f}ms "
                f"full_hits={row['full_hits_delta']} "
                f"sliding_hits={row['sliding_hits_delta']}",
                flush=True,
            )

    summary = summarize(
        samples,
        minimum_speedup=args.minimum_speedup,
        maximum_spread=args.maximum_spread,
    )
    payload = {
        "benchmark": "gemma4_e2b_l4_b4_prefill_policy_gate",
        "model": args.model,
        "gpu": gpu,
        "torch": torch.__version__,
        "profile_environment": profile,
        "workload": {
            "batch_size": 4,
            "prompt_tokens_requested": 2048,
            "prompt_tokens_actual_total": actual_tokens,
            "max_new_tokens": 1,
            "dtype": "bf16",
        },
        "system": {
            "git": matrix.git_snapshot(),
            "gpu": matrix.gpu_snapshot(),
            "packages": matrix.installed_package_versions(),
        },
        "samples": samples,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nB4/P2048 FULL-MODEL PREFILL POLICY")
    print(f"{'case':<42} {'prefill ms':>12} {'speedup':>10} {'full':>7} {'slide':>7}")
    baseline_ms = float(summary["baseline_prefill_ms"])
    for name, row in summary["cases"].items():
        case_ms = float(row["prefill_ms_median"])
        print(
            f"{name:<42} {case_ms:>12.2f} "
            f"{(baseline_ms / case_ms if case_ms else 0.0):>10.3f} "
            f"{row['full_hits']:>7} {row['sliding_hits']:>7}"
        )
    print("DECISION " + json.dumps(summary, separators=(",", ":")))
    print(f"Wrote: {args.output}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--minimum-speedup", type=float, default=1.05)
    parser.add_argument("--maximum-spread", type=float, default=1.08)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.warmups < 1 or args.repeats < 3:
        raise SystemExit("use at least one warmup and three repeats")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
