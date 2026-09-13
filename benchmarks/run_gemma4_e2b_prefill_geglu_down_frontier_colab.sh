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

test -f benchmarks/run_gemma4_e2b_prefill_geglu_down_frontier.py || {
  echo "ERRO: falta o frontier GeGLU+down na pasta do Drive."
  exit 3
}

# Apenas dependências do MegaGemm. Não instala vLLM, não recompila extensões
# e não modifica o repositório do Drive.
python -m pip install -q \
  "transformers==5.14.1" \
  huggingface_hub safetensors sentencepiece psutil tqdm

python - <<'PY'
import megagemm
import torch
from megagemm.kernels.deepfusion_mlp import HAS_DEEPFUSION_MLP

print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("MegaGemm:", megagemm.__file__)
assert "/content/drive/MyDrive/mg/MGRrmsnorm/" in megagemm.__file__
assert HAS_DEEPFUSION_MLP
PY

RUN_ID="${RUN_ID:-gemma4_e2b_prefill_geglu_down_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/bench_results/gemma4_e2b_prefill_geglu_down/$RUN_ID/decision.json}"

python benchmarks/run_gemma4_e2b_prefill_geglu_down_frontier.py \
  --model google/gemma-4-E2B-it \
  --batch-size 8 \
  --prompt-tokens 512,2048 \
  --repeats "${REPEATS:-3}" \
  --max-seq-len 2304 \
  --output "$OUT"

echo "RESULTADO: $OUT"
