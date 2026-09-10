#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
export REPO
export RUN_ID="${RUN_ID:-gemma4_e2b_b8_lm_head_frontier_$(date -u +%Y%m%dT%H%M%SZ)}"
export OUT="${OUT:-$REPO/bench_results/gemma4_e2b_b8_lm_head_frontier/$RUN_ID/decision.json}"

# BN64/BK128/W4/S2 with Triton reduction is production. Screen one LM-head
# dimension at a time on both contexts with long generation, then validate the
# winner against production on P512/P2048 x O16/O128 in the same model load.
export SCREEN_CASE_NAMES="${SCREEN_CASE_NAMES:-production,lm_head_bn32,lm_head_bn128,lm_head_bk64,lm_head_bk256,lm_head_w2,lm_head_w8,lm_head_s1,lm_head_s3,lm_head_torch_reduce}"
export SCREEN_PROMPT_TOKENS="${SCREEN_PROMPT_TOKENS:-512,2048}"
export SCREEN_OUTPUT_TOKENS="${SCREEN_OUTPUT_TOKENS:-128}"
export SCREEN_REPEATS="${SCREEN_REPEATS:-2}"
export FINAL_REPEATS="${FINAL_REPEATS:-3}"
export WARMUPS="${WARMUPS:-1}"

bash "$REPO/benchmarks/run_gemma4_e2b_b8_compute_frontier_colab.sh"
