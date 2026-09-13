#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "GPU: $GPU_NAME"
[[ "$GPU_NAME" == *"L4"* ]] || {
  echo "ERRO: este gate exige NVIDIA L4."
  exit 2
}

test -f megagemm/kernels/gemma4_e2b_prefill_mlp.py || {
  echo "ERRO: falta megagemm/kernels/gemma4_e2b_prefill_mlp.py na pasta do Drive."
  exit 3
}
test -f benchmarks/run_gemma4_e2b_prefill_fused_gateup_frontier.py || {
  echo "ERRO: falta o frontier de prefill na pasta do Drive."
  exit 3
}

# Apenas dependências do MegaGemm. Não instala vLLM e não recompila extensões.
python -m pip install -q \
  "transformers==5.14.1" \
  huggingface_hub safetensors sentencepiece psutil tqdm

python - <<'PY'
import megagemm
import torch
from megagemm.kernels.gemma4_e2b_prefill_mlp import (
    HAS_GEMMA4_E2B_PREFILL_FUSED_GATEUP,
)

print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("MegaGemm:", megagemm.__file__)
assert "/content/drive/MyDrive/mg/MGRrmsnorm/" in megagemm.__file__
assert HAS_GEMMA4_E2B_PREFILL_FUSED_GATEUP
PY

RUN_ID="${RUN_ID:-gemma4_e2b_prefill_fused_gateup_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/bench_results/gemma4_e2b_prefill_fused_gateup/$RUN_ID/decision.json}"

python benchmarks/run_gemma4_e2b_prefill_fused_gateup_frontier.py \
  --model google/gemma-4-E2B-it \
  --batch-size 8 \
  --prompt-tokens 512,2048 \
  --repeats "${REPEATS:-3}" \
  --max-seq-len 2304 \
  --output "$OUT"

echo "RESULTADO: $OUT"
