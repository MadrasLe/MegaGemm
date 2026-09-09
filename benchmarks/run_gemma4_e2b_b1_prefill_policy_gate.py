#!/usr/bin/env python3
"""Loaded-model B1/P2048 prefill policy gate for Gemma 4 E2B on L4.

This runner reuses the full-model policy-gate implementation and can replay
the B1 launch-geometry decision independently from the production default.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from typing import Any, Iterator

from benchmarks import run_gemma4_e2b_b2_prefill_policy_gate as common


# (group_heads, block_m, block_n, num_warps, num_stages)
CASES: tuple[dict[str, Any], ...] = (
    {"name": "baseline", "full": False, "sliding": False, "config": None},
    {"name": "full_h512_only", "full": True, "sliding": False, "config": None},
    {
        "name": "sliding_h256_only_g1_bm8_bn64_w4_s2",
        "full": False,
        "sliding": True,
        "config": (1, 8, 64, 4, 2),
    },
    {
        "name": "combined_g1_bm8_bn32_w4_s2",
        "full": True,
        "sliding": True,
        "config": (1, 8, 32, 4, 2),
    },
    {
        "name": "combined_g1_bm8_bn64_w4_s2",
        "full": True,
        "sliding": True,
        "config": (1, 8, 64, 4, 2),
    },
    {
        "name": "combined_g1_bm8_bn64_w8_s2",
        "full": True,
        "sliding": True,
        "config": (1, 8, 64, 8, 2),
    },
    {
        "name": "combined_g2_bm4_bn64_w4_s2",
        "full": True,
        "sliding": True,
        "config": (2, 4, 64, 4, 2),
    },
    {
        "name": "combined_g2_bm4_bn64_w8_s2",
        "full": True,
        "sliding": True,
        "config": (2, 4, 64, 8, 2),
    },
    {
        "name": "combined_g4_bm4_bn64_w4_s2",
        "full": True,
        "sliding": True,
        "config": (4, 4, 64, 4, 2),
    },
    {
        "name": "combined_g1_bm16_bn64_w4_s2",
        "full": True,
        "sliding": True,
        "config": (1, 16, 64, 4, 2),
    },
    {
        "name": "combined_g1_bm16_bn64_w8_s2",
        "full": True,
        "sliding": True,
        "config": (1, 16, 64, 8, 2),
    },
    {
        "name": "combined_g2_bm8_bn64_w4_s2",
        "full": True,
        "sliding": True,
        "config": (2, 8, 64, 4, 2),
    },
    {
        "name": "combined_g1_bm32_bn64_w4_s2",
        "full": True,
        "sliding": True,
        "config": (1, 32, 64, 4, 2),
    },
    {
        "name": "combined_g2_bm16_bn64_w4_s2",
        "full": True,
        "sliding": True,
        "config": (2, 16, 64, 4, 2),
    },
    {
        "name": "combined_g1_bm8_bn128_w4_s2",
        "full": True,
        "sliding": True,
        "config": (1, 8, 128, 4, 2),
    },
)


_OVERRIDES = {
    "BATCH_SIZE": 1,
    "PROMPT_TOKENS": 2048,
    "EXPERIMENT_FLAG": None,
    "TILE_PREFIX": "MEGAGEMM_GEMMA4_E2B_L4_B1_SLIDING_",
    "DISABLED_ATTR": "_GEMMA4_E2B_L4_B1_SLIDING_PREFILL_DISABLED",
    "BATCH_LABEL": "B1",
    "BENCHMARK_NAME": "gemma4_e2b_l4_b1_prefill_policy_gate",
    "READY_DECISION": "IMPLEMENT_B1_POLICY_AND_RUN_TARGETED_MACRO_GATE",
    "KEEP_DECISION": "KEEP_B1_BASELINE",
    "CASES": CASES,
}


@contextmanager
def _b1_configuration() -> Iterator[None]:
    previous = {name: getattr(common, name) for name in _OVERRIDES}
    try:
        for name, value in _OVERRIDES.items():
            setattr(common, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(common, name, value)


def summarize(
    samples: list[dict[str, Any]],
    *,
    minimum_speedup: float,
    maximum_spread: float,
) -> dict[str, Any]:
    with _b1_configuration():
        return common.summarize(
            samples,
            minimum_speedup=minimum_speedup,
            maximum_spread=maximum_spread,
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    with _b1_configuration():
        return common.run(args)


def build_parser() -> argparse.ArgumentParser:
    return common.build_parser()


def main() -> int:
    args = build_parser().parse_args()
    if args.warmups < 1 or args.repeats < 3:
        raise SystemExit("use at least one warmup and three repeats")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
