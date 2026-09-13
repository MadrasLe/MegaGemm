#!/usr/bin/env bash
set -euo pipefail

# Unambiguous entry point for the E2B/L4/B8 down-projection search.  The
# shared harness owns the fresh-session dependency setup and ephemeral build.
REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
TARGET=down \
  bash "$REPO/benchmarks/run_gemma4_e2b_prefill_cublaslt_frontier_colab.sh"
