#!/usr/bin/env bash
set -euo pipefail

# Fresh-session runner.  The project source of truth is the user's Drive copy;
# this script performs no git operation and installs no competing engine.
REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
export SUITE="${SUITE:-all}"
GATE="$REPO/benchmarks/run_gemma4_e2b_b1_decode_frontier_gate.py"
if [[ "$SUITE" == "execution" || "$SUITE" == "graph-kernels" || "$SUITE" == "promotion" ]]; then
  GATE="$REPO/benchmarks/run_gemma4_e2b_b1_execution_gate.py"
fi

[[ -d "$REPO/megagemm" ]] || {
  echo "ERRO: pacote MegaGemm não encontrado em $REPO/megagemm"
  exit 2
}
[[ -f "$GATE" ]] || {
  echo "ERRO: gate não encontrado em $GATE"
  echo "Copie a pasta local atualizada para $REPO antes de executar."
  exit 3
}

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "GPU: $GPU_NAME"
[[ "$GPU_NAME" == *"L4"* ]] || {
  echo "ERRO: este gate exige uma NVIDIA L4."
  exit 4
}

if ! python - <<'PY'
import huggingface_hub
import psutil
import safetensors
import sentencepiece
import tqdm
import transformers
import triton

if transformers.__version__ != "5.14.1":
    raise SystemExit(1)
PY
then
  python -m pip install -q \
    "transformers==5.14.1" \
    huggingface_hub safetensors sentencepiece psutil tqdm
fi

REPO="$REPO" python - <<'PY'
import os
from pathlib import Path

import megagemm
import torch

repo = Path(os.environ["REPO"]).resolve()
source = Path(megagemm.__file__).resolve()
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("MegaGemm:", source)
assert source.is_relative_to(repo), (
    f"import veio de {source}, não da pasta do Drive {repo}"
)
if os.environ["SUITE"] in ("execution", "graph-kernels", "promotion"):
    from megagemm.engine.scheduler import Scheduler
    import triton
    if not callable(getattr(Scheduler, "_decode_graph_model_step", None)):
        raise SystemExit("A cópia de scheduler.py não contém o corpo do graph B1.")
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA indisponível nesta sessão.")
    if os.environ["SUITE"] == "execution":
        print("Execution gate: eager / same-body eager / one-step graph / unrolled graph")
    if os.environ["SUITE"] == "promotion":
        print("Promotion gate: eager baseline / default E2B-L4-B4 RuntimePolicy")
    if os.environ["SUITE"] == "graph-kernels":
        from benchmarks.gemma4_b1_graph_kernels import GraphKernelExperiment
        from megagemm.models.llama import MegaGemmLlama
        if not callable(getattr(MegaGemmLlama, "_prepare_gemma4_b1_mlp_gemv_routes", None)):
            raise SystemExit("A cópia de llama.py não contém as rotas GEMV preparadas.")
        print("Graph kernel cases:", ", ".join(c.name for c in GraphKernelExperiment().cases))
if os.environ["SUITE"] in ("mlp-gemv", "mlp-dispatch"):
    from benchmarks.gemma4_b1_mlp_gemv_frontier import build_cases
    from megagemm.models.llama import MegaGemmLlama
    if not callable(getattr(MegaGemmLlama, "_gemma4_b1_mlp_gemv", None)):
        raise SystemExit("A cópia de megagemm/models/llama.py não contém o dispatch GEMV B1.")
    cases = build_cases()
    if os.environ["SUITE"] == "mlp-dispatch":
        from benchmarks.gemma4_b1_mlp_gemv_frontier import build_dispatch_cases
        if not callable(getattr(MegaGemmLlama, "_prepare_gemma4_b1_mlp_gemv_routes", None)):
            raise SystemExit("A cópia de llama.py não contém a preparação do dispatch GEMV.")
        cases = build_dispatch_cases()
    print("MLP GEMV B1: gate e dispatch presentes;", len(cases), "casos")
PY

RUN_ID="${RUN_ID:-gemma4_e2b_b1_decode_frontier_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/bench_results/gemma4_e2b_b1_decode_frontier/$RUN_ID/decision.json}"

if [[ "$SUITE" == "execution" || "$SUITE" == "graph-kernels" || "$SUITE" == "promotion" ]]; then
  EXTRA_ARGS=()
  if [[ "$SUITE" == "graph-kernels" ]]; then
    EXTRA_ARGS+=(--kernel-candidates --skip-python-audit)
  fi
  if [[ "$SUITE" == "promotion" ]]; then
    EXTRA_ARGS+=(--verify-promotion --skip-python-audit)
  fi
  python -u "$GATE" \
    --model "$MODEL" \
    --batch-size "${BATCH_SIZE:-1}" \
    --prompt-tokens "${PROMPT_TOKENS:-2048}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-128}" \
    --warmups "${WARMUPS:-1}" \
    --repeats "${REPEATS:-3}" \
    --minimum-speedup "${MINIMUM_SPEEDUP:-1.02}" \
    --maximum-spread "${MAXIMUM_SPREAD:-1.06}" \
    --output "$OUT" \
    "${EXTRA_ARGS[@]}"
  echo "RESULTADO: $OUT"
  exit 0
fi

python -u "$GATE" \
  --suite "${SUITE:-all}" \
  --model "$MODEL" \
  --prompt-tokens "${PROMPT_TOKENS:-2048}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-128}" \
  --warmups "${WARMUPS:-1}" \
  --repeats "${REPEATS:-3}" \
  --minimum-speedup "${MINIMUM_SPEEDUP:-1.01}" \
  --maximum-spread "${MAXIMUM_SPREAD:-1.06}" \
  --output "$OUT"

echo "RESULTADO: $OUT"
