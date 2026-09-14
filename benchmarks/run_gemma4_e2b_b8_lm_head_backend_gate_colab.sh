#!/usr/bin/env bash
set -euo pipefail

# Fresh Colab session, fixed Drive source, no native build and no competing
# engine.  All three LM-head routes run in one Python process/model load.
REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
RUN_ID="${RUN_ID:-gemma4_e2b_b8_lm_head_backend_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/bench_results/gemma4_e2b_b8_lm_head_backend/$RUN_ID/decision.json}"

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "GPU: $GPU_NAME"
[[ "$GPU_NAME" == *"L4"* ]] || {
  echo "ERRO: este gate exige NVIDIA L4."
  exit 2
}

python -m pip install -q \
  "transformers==5.14.1" \
  huggingface_hub safetensors sentencepiece psutil tqdm

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

python -u benchmarks/run_gemma4_e2b_b8_lm_head_backend_gate.py \
  --model "$MODEL" \
  --prompt-tokens 2048 \
  --max-new-tokens 128 \
  --warmups "${WARMUPS:-1}" \
  --repeats "${REPEATS:-5}" \
  --output "$OUT"

echo "RESULTADO: $OUT"
