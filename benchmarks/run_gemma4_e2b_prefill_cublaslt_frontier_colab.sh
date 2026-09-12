#!/usr/bin/env bash
set -euo pipefail

# Fresh-session Colab harness.  The source comes directly from the fixed Drive
# path.  The focused cuBLASLt extension is built under /tmp and leaves no native
# build products in the repository or Drive.
REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
RUN_ID="${RUN_ID:-gemma4_e2b_prefill_cublaslt_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/bench_results/gemma4_e2b_prefill_cublaslt/$RUN_ID/frontier.json}"

cd "$REPO"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "GPU: $GPU_NAME"
[[ "$GPU_NAME" == *"L4"* ]] || {
  echo "ERRO: este gate exige NVIDIA L4."
  exit 2
}

python -m pip install -q \
  setuptools wheel ninja \
  "transformers==5.14.1" \
  huggingface_hub safetensors sentencepiece psutil tqdm

TORCH_LIB_DIR="$(python - <<'PY'
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

BUILD_ROOT="$(mktemp -d /tmp/megagemm_cublaslt.XXXXXX)"
MEGAGEMM_BUILD_ONLY_CUBLASLT=1 \
MAX_JOBS="${MAX_JOBS:-2}" \
python setup.py build_ext \
  --build-temp "$BUILD_ROOT/temp" \
  --build-lib "$BUILD_ROOT/lib"

export PYTHONPATH="$BUILD_ROOT/lib:$REPO${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import torch
import megagemm
import megagemm_cublaslt_ops
from megagemm.kernels.mlp_prefill_native import HAS_CUBLASLT_BF16_LINEAR

print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("MegaGemm:", megagemm.__file__)
print("Focused cuBLASLt:", megagemm_cublaslt_ops.__file__)
assert HAS_CUBLASLT_BF16_LINEAR
PY

mkdir -p "$(dirname "$OUT")"
python benchmarks/run_gemma4_e2b_prefill_cublaslt_frontier.py \
  --model google/gemma-4-E2B-it \
  --batch-size 8 \
  --prompt-tokens 512,2048 \
  --maximum-algorithms "${MAXIMUM_ALGORITHMS:-16}" \
  --screen-warmups "${SCREEN_WARMUPS:-2}" \
  --screen-repeats "${SCREEN_REPEATS:-5}" \
  --repeats "${REPEATS:-3}" \
  --max-seq-len 2304 \
  --output "$OUT"

echo "RESULTADO: $OUT"
