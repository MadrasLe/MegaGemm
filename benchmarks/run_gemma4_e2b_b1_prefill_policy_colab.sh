#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "GPU: $GPU_NAME"
[[ "$GPU_NAME" == *"L4"* ]] || {
  echo "ERRO: este gate exige NVIDIA L4."
  exit 2
}

RUN_ID="${RUN_ID:-gemma4_e2b_b1_prefill_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/bench_results/gemma4_e2b_b1_prefill/$RUN_ID}"
mkdir -p "$OUT"

python benchmarks/run_gemma4_e2b_b1_prefill_policy_gate.py \
  --model "$MODEL" \
  --warmups "${WARMUPS:-1}" \
  --repeats "${REPEATS:-3}" \
  --output "$OUT/decision.json"

echo "Resultado: $OUT/decision.json"
