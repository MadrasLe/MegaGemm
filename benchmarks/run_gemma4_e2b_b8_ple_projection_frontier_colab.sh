#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
RUN_ID="${RUN_ID:-gemma4_e2b_b8_ple_projection_$(date -u +%Y%m%dT%H%M%SZ)}"

REPO="$REPO" \
RUN_ID="$RUN_ID" \
SCREEN_CASE_NAMES="${SCREEN_CASE_NAMES:-ple_proj_bn32_bk32_w4_s2,ple_proj_bn64_bk32_w4_s2,ple_proj_bn128_bk32_w4_s2,ple_proj_bn64_bk64_w4_s2,ple_proj_bn64_bk32_w8_s2,ple_proj_bn64_bk32_w4_s3}" \
SCREEN_PROMPT_TOKENS="${SCREEN_PROMPT_TOKENS:-2048}" \
SCREEN_OUTPUT_TOKENS="${SCREEN_OUTPUT_TOKENS:-128}" \
SCREEN_REPEATS="${SCREEN_REPEATS:-2}" \
FINAL_REPEATS="${FINAL_REPEATS:-3}" \
WARMUPS="${WARMUPS:-1}" \
OUT="$REPO/bench_results/gemma4_e2b_b8_ple_projection/$RUN_ID/decision.json" \
bash "$REPO/benchmarks/run_gemma4_e2b_b8_compute_frontier_colab.sh"
