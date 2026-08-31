#!/usr/bin/env bash
set -euo pipefail

# The Drive checkout is the source of truth.  This harness performs no git
# operation and installs no competing engine or notebook-global package stack.
REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
LABEL="${LABEL:-production}"
PROFILE="${PROFILE:-production}"

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "GPU: $GPU_NAME"
[[ "$GPU_NAME" == *"L4"* ]] || {
  echo "ERRO: esta matriz exige NVIDIA L4."
  exit 2
}

python - <<'PY'
import megagemm
import torch

print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("MegaGemm:", megagemm.__file__)
assert torch.cuda.is_available()
assert "/content/drive/MyDrive/mg/MGRrmsnorm/" in megagemm.__file__
PY

RUN_ID="${RUN_ID:-gemma4_e2b_macro_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/bench_results/gemma4_e2b_macro/$RUN_ID}"
mkdir -p "$OUT"

python benchmarks/run_gemma4_e2b_macro_matrix.py measure \
  --model "$MODEL" \
  --label "$LABEL" \
  --profile "$PROFILE" \
  --batch-sizes "${BATCH_SIZES:-1,2,4,8}" \
  --prompt-tokens "${PROMPT_TOKENS:-128,512,2048}" \
  --output-tokens "${OUTPUT_TOKENS:-1,16,128}" \
  --warmups "${WARMUPS:-1}" \
  --repeats "${REPEATS:-3}" \
  --max-seq-len "${MAX_SEQ_LEN:-2304}" \
  --output "$OUT/$LABEL.json"

if [[ -n "${COMPARE_WITH:-}" ]]; then
  python benchmarks/run_gemma4_e2b_macro_matrix.py compare \
    --baseline "$COMPARE_WITH" \
    --candidate "$OUT/$LABEL.json" \
    --output "$OUT/comparison.json"
fi

echo "Resultado: $OUT/$LABEL.json"
if [[ -f "$OUT/comparison.json" ]]; then
  echo "Comparação: $OUT/comparison.json"
fi

# Keep the notebook cell successful when measurement completed but the optional
# COMPARE_WITH input was not supplied.
exit 0
