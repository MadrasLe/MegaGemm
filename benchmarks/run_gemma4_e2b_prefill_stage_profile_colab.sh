#!/usr/bin/env bash
set -euo pipefail

# Fresh-session MegaGemm-only profiler.  The Drive checkout is the source of
# truth; this harness performs no git operation and installs no vLLM.
REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "GPU: $GPU_NAME"
[[ "$GPU_NAME" == *"L4"* ]] || {
  echo "ERRO: este profiler exige NVIDIA L4."
  exit 2
}

if ! python - <<'PY' >/dev/null 2>&1
import huggingface_hub
import psutil
import safetensors
import sentencepiece
import tqdm
import transformers
import triton

assert transformers.__version__ == "5.14.1", transformers.__version__
PY
then
  python -m pip install -q \
    huggingface_hub safetensors "transformers==5.14.1" sentencepiece psutil tqdm
fi

python - <<'PY'
import megagemm
import torch
import transformers
import triton

print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("MegaGemm:", megagemm.__file__)
print("Triton:", triton.__version__)
print("Transformers:", transformers.__version__)
assert torch.cuda.is_available(), "CUDA indisponível"
assert "/content/drive/MyDrive/mg/MGRrmsnorm/" in megagemm.__file__
PY

RUN_ID="${RUN_ID:-gemma4_e2b_prefill_profile_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/bench_results/gemma4_e2b_prefill_profile/$RUN_ID}"
mkdir -p "$OUT"

python benchmarks/run_gemma4_e2b_prefill_stage_profile.py \
  --model "$MODEL" \
  --batch-size 8 \
  --prompt-tokens 2048 \
  --warmups "${WARMUPS:-3}" \
  --repeats "${REPEATS:-5}" \
  --max-seq-len 2304 \
  --output "$OUT/profile.json"

echo "Resultado: $OUT/profile.json"
