#!/usr/bin/env bash
set -euo pipefail

# Full-model only: one E2B load, natural greedy output, production CUDA Graph
# execution fixed.  This wrapper never installs or imports a competing engine.
REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
export SCREEN_CASE_NAMES="${SCREEN_CASE_NAMES:-mlp_gated_act_bs128,mlp_gated_act_bs256,mlp_gated_act_bs512,mlp_gated_act_bs1024,mlp_tc_bn32_bk32_w4_s2,mlp_tc_bn64_bk32_w4_s2,mlp_tc_bn128_bk32_w4_s2,mlp_tc_bn64_bk64_w4_s2,mlp_tc_bn64_bk32_w8_s2,mlp_tc_bn64_bk32_w4_s3,ple_conditioned_bs128,ple_conditioned_bs256,ple_conditioned_bs512,ple_conditioned_bs1024}"
export SCREEN_PROMPT_TOKENS="${SCREEN_PROMPT_TOKENS:-512,2048}"
export SCREEN_OUTPUT_TOKENS="${SCREEN_OUTPUT_TOKENS:-128}"
export SCREEN_REPEATS="${SCREEN_REPEATS:-2}"
export FINAL_REPEATS="${FINAL_REPEATS:-3}"
export WARMUPS="${WARMUPS:-1}"
export RUN_ID="${RUN_ID:-gemma4_e2b_b8_mlp_chain_$(date -u +%Y%m%dT%H%M%SZ)}"
export OUT="${OUT:-$REPO/bench_results/gemma4_e2b_b8_mlp_chain/$RUN_ID/decision.json}"

bash "$REPO/benchmarks/run_gemma4_e2b_b8_compute_frontier_colab.sh"
