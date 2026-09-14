#!/usr/bin/env bash
set -euo pipefail

# Fresh Colab session, fixed Drive source.  The focused extension is compiled
# under /tmp; no build artifact is written into the repository or Drive.
REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
RUN_ID="${RUN_ID:-gemma4_e2b_b8_cublaslt_decode_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/bench_results/gemma4_e2b_b8_cublaslt_decode/$RUN_ID/decision.json}"

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

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

BUILD_ROOT="$(mktemp -d /tmp/megagemm_cublaslt_decode.XXXXXX)"
MEGAGEMM_BUILD_ONLY_CUBLASLT=1 \
MAX_JOBS="${MAX_JOBS:-2}" \
python setup.py build_ext \
  --build-temp "$BUILD_ROOT/temp" \
  --build-lib "$BUILD_ROOT/lib"

export PYTHONPATH="$BUILD_ROOT/lib:$REPO${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
from pathlib import Path
import megagemm
import megagemm_cublaslt_ops
import torch
from megagemm.kernels.mlp_prefill_native import HAS_CUBLASLT_BF16_LINEAR

repo = Path("/content/drive/MyDrive/mg/MGRrmsnorm").resolve()
source = Path(megagemm.__file__).resolve()
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("MegaGemm:", source)
print("Focused cuBLASLt:", megagemm_cublaslt_ops.__file__)
assert source.is_relative_to(repo), f"import veio de {source}, não de {repo}"
assert HAS_CUBLASLT_BF16_LINEAR
PY

python -u benchmarks/run_gemma4_e2b_b8_cublaslt_decode_frontier.py \
  --model "$MODEL" \
  --prompt-tokens 2048 \
  --max-new-tokens 128 \
  --maximum-algorithms "${MAXIMUM_ALGORITHMS:-16}" \
  --tune-warmups "${TUNE_WARMUPS:-2}" \
  --tune-repeats "${TUNE_REPEATS:-5}" \
  --warmups "${WARMUPS:-1}" \
  --repeats "${REPEATS:-5}" \
  --output "$OUT"

echo "RESULTADO: $OUT"
