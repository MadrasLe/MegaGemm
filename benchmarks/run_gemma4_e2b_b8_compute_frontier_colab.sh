#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "GPU: $GPU_NAME"
[[ "$GPU_NAME" == *"L4"* ]] || {
  echo "ERRO: este frontier exige uma NVIDIA L4."
  exit 2
}

python -m pip install -q \
  "transformers==5.14.1" \
  huggingface_hub \
  safetensors \
  sentencepiece \
  psutil \
  tqdm

python - <<'PY'
from pathlib import Path
import megagemm
import torch

repo = Path("/content/drive/MyDrive/mg/MGRrmsnorm").resolve()
source = Path(megagemm.__file__).resolve()
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("MegaGemm:", source)
assert source.is_relative_to(repo), f"import veio de {source}, não de {repo}"
PY

RUN_ID="${RUN_ID:-gemma4_e2b_b8_compute_frontier_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/bench_results/gemma4_e2b_b8_compute_frontier/$RUN_ID/decision.json}"

EXTRA_ARGS=()
if [[ -n "${SCREEN_CASE_NAMES:-}" ]]; then
  EXTRA_ARGS+=(--screen-case-names "$SCREEN_CASE_NAMES")
fi

python -u benchmarks/run_gemma4_e2b_b8_compute_frontier.py \
  --model "$MODEL" \
  --screen-repeats "${SCREEN_REPEATS:-2}" \
  --final-repeats "${FINAL_REPEATS:-3}" \
  --warmups "${WARMUPS:-1}" \
  --screen-prompt-tokens "${SCREEN_PROMPT_TOKENS:-512,2048}" \
  --screen-output-tokens "${SCREEN_OUTPUT_TOKENS:-16,128}" \
  --output "$OUT" \
  "${EXTRA_ARGS[@]}"

echo "RESULTADO: $OUT"
