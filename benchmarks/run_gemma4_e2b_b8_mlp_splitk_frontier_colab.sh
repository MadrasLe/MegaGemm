#!/usr/bin/env bash
set -euo pipefail

# Scenario-specific B8/O128 decode gate.  Production and all split-K candidates
# share one E2B model load; prefill policy and CUDA Graph burst remain fixed.
REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
export SCREEN_CASE_NAMES="${SCREEN_CASE_NAMES:-mlp_tc_bn64_bk32_w4_s3_sk2,mlp_tc_bn64_bk32_w4_s3_sk4,mlp_tc_bn64_bk32_w4_s3_sk8,mlp_tc_bn64_bk64_w4_s2_sk2,mlp_tc_bn64_bk64_w4_s2_sk4,mlp_tc_bn64_bk64_w4_s2_sk8}"
export SCREEN_PROMPT_TOKENS="${SCREEN_PROMPT_TOKENS:-512,2048}"
export SCREEN_OUTPUT_TOKENS="${SCREEN_OUTPUT_TOKENS:-128}"
export SCREEN_REPEATS="${SCREEN_REPEATS:-2}"
export FINAL_REPEATS="${FINAL_REPEATS:-3}"
export WARMUPS="${WARMUPS:-1}"
export RUN_ID="${RUN_ID:-gemma4_e2b_b8_mlp_splitk_$(date -u +%Y%m%dT%H%M%SZ)}"
export OUT="${OUT:-$REPO/bench_results/gemma4_e2b_b8_mlp_splitk/$RUN_ID/decision.json}"

bash "$REPO/benchmarks/run_gemma4_e2b_b8_compute_frontier_colab.sh"
