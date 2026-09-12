"""
🧠 MegaGemm Unified Model
--------------------------
Multi-architecture inference model using MegaGemm kernels:
  - RMSNorm (CUDA)
  - SwiGLU / GeGLU (Triton)
  - RoPE (PyTorch, CUDA-ready)
  - PagedAttention (Triton) for decode

Supports: LLaMA 2/3, TinyLlama, Mistral, CodeLlama,
          Qwen 2.5, Qwen 3, Gemma 2

Author: Gabriel Yogi
"""

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple, List, Union, runtime_checkable

import torch
import torch.nn as nn

from .sparsity import is_semi_structured_weight
from .runtime_policy import policy_bool, policy_rows, resolve_runtime_policy

try:
    from ..kernels.sparse24_mma import (
        sparse24_mma_linear_unchecked as _flat_sparse24_mma_linear,
    )
except Exception:
    _flat_sparse24_mma_linear = None

from ..kernels.paged_attention import (
    paged_attention_decode,
    prefill_attention,
    packed_prefill_attention,
    fused_rope_kv_write,
    gemma4_e2b_l4_sliding_prefill_attention,
    gemma4_long_sliding_prefill_attention,
    gemma4_long_full_prefill_attention,
    paged_decode_runtime_stats,
    _triton_paged_decode_fused,
)
try:
    from ..kernels.fast_gemv import (
        fast_linear,
        HAS_TRITON_FAST_GEMV,
        fast_gemv_prefers_triton_shape,
        fast_gemv_splitk_scratch_shape,
    )
except Exception:
    fast_linear = None
    HAS_TRITON_FAST_GEMV = False
    fast_gemv_prefers_triton_shape = None
    fast_gemv_splitk_scratch_shape = None
from ..kernels.rope import (
    precompute_freqs_cis,
    precompute_proportional_freqs_cis,
    apply_rotary_emb,
)
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False
try:
    from ..kernels.linear_attention import (
        HAS_TRITON_LINEAR_ATTN,
        chunk_interchunk,
        chunk_interchunk_scan,
        chunk_state_projection,
        chunk_state_update,
        recurrent_gated_delta_decode,
        recurrent_gated_delta_decode_from_ab,
        recurrent_gated_delta_prefill,
        solve_chunk_local_attention,
    )
except Exception:
    HAS_TRITON_LINEAR_ATTN = False
    chunk_interchunk = None
    chunk_interchunk_scan = None
    chunk_state_projection = None
    chunk_state_update = None
    recurrent_gated_delta_decode = None
    recurrent_gated_delta_decode_from_ab = None
    recurrent_gated_delta_prefill = None
    solve_chunk_local_attention = None

# Try MegaGemm CUDA RMSNorm, fallback to PyTorch
try:
    from ..kernels.rmsnorm import (
        RMSNormFunction,
        can_use_cuda_rmsnorm_for as _kernel_can_use_cuda_rmsnorm_for,
        rmsnorm_forward,
    )
    _HAS_CUDA_RMSNORM = True
except Exception:
    rmsnorm_forward = None
    _kernel_can_use_cuda_rmsnorm_for = None
    _HAS_CUDA_RMSNORM = False
if os.environ.get("MEGAGEMM_DISABLE_CUDA_RMSNORM", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}:
    rmsnorm_forward = None
    _HAS_CUDA_RMSNORM = False


def _can_use_cuda_rmsnorm_for(x: torch.Tensor, offset: bool = False) -> bool:
    if not (
        _HAS_CUDA_RMSNORM
        and rmsnorm_forward is not None
        and callable(_kernel_can_use_cuda_rmsnorm_for)
    ):
        return False
    return bool(_kernel_can_use_cuda_rmsnorm_for(x, offset))

# Try Triton RMSNorm with offset support
try:
    from ..kernels.rmsnorm_triton import (
        rmsnorm_triton,
        rmsnorm_triton_add,
        rmsnorm_triton_attn_residual_dense,
        rmsnorm_triton_attn_residual_dual,
        rmsnorm_triton_attn_residual_router_bridge,
        rmsnorm_triton_attn_residual_router_bridge_single,
        rmsnorm_triton_dual,
        rmsnorm_triton_no_weight,
        rmsnorm_triton_pair_add_final,
        rmsnorm_triton_pair_add_final_residual,
        rmsnorm_triton_residual_scale_next,
        rmsnorm_triton_scaled_no_weight,
        rmsnorm_triton_weighted_scaled_no_weight_dual,
    )
    _HAS_TRITON_RMSNORM = True
except Exception:
    rmsnorm_triton = None
    rmsnorm_triton_add = None
    rmsnorm_triton_attn_residual_dense = None
    rmsnorm_triton_attn_residual_dual = None
    rmsnorm_triton_attn_residual_router_bridge = None
    rmsnorm_triton_attn_residual_router_bridge_single = None
    rmsnorm_triton_dual = None
    rmsnorm_triton_no_weight = None
    rmsnorm_triton_pair_add_final = None
    rmsnorm_triton_pair_add_final_residual = None
    rmsnorm_triton_residual_scale_next = None
    rmsnorm_triton_scaled_no_weight = None
    rmsnorm_triton_weighted_scaled_no_weight_dual = None
    _HAS_TRITON_RMSNORM = False
try:
    from ..kernels.gemma4_attention_prepare import (
        HAS_GEMMA4_ATTENTION_PREPARE,
        gemma4_prefill_attention_prepare,
    )
except Exception:
    HAS_GEMMA4_ATTENTION_PREPARE = False
    gemma4_prefill_attention_prepare = None

_USE_EXPERIMENTAL_TRITON_OFFSET_RMSNORM = os.environ.get(
    "MEGAGEMM_USE_TRITON_OFFSET_RMSNORM", ""
).strip().lower() in {"1", "true", "yes", "on"}

# Try MegaGemm Triton SwiGLU
try:
    from ..kernels.swiglu import (
        MegaGemmFunction,
        conditioned_gelu_tanh_forward,
        gated_activation_forward,
        swiglu_forward,
    )
    _HAS_TRITON_SWIGLU = True
except Exception:
    conditioned_gelu_tanh_forward = None
    gated_activation_forward = None
    swiglu_forward = None
    _HAS_TRITON_SWIGLU = False
try:
    from ..kernels.mlp_prefill_native import (
        cublaslt_bf16_linear_cuda,
        mlp_prefill_forward_cuda as native_mlp_prefill_forward_cuda,
        HAS_CUBLASLT_BF16_LINEAR,
        HAS_NATIVE_MLP_PREFILL,
    )
except Exception:
    cublaslt_bf16_linear_cuda = None
    native_mlp_prefill_forward_cuda = None
    HAS_CUBLASLT_BF16_LINEAR = False
    HAS_NATIVE_MLP_PREFILL = False

# Try decode DeepFusion MLP (SwiGLU + down_proj fused)
try:
    from ..kernels.deepfusion_mlp import (
        deepfusion_swiglu_down,
        gemma4_e2b_b8_geglu_down_tensorcore,
        HAS_DEEPFUSION_MLP,
        deepfusion_mlp_prefers_triton_shape,
        deepfusion_runtime_config,
    )
except Exception:
    deepfusion_swiglu_down = None
    gemma4_e2b_b8_geglu_down_tensorcore = None
    HAS_DEEPFUSION_MLP = False
    deepfusion_mlp_prefers_triton_shape = None
    deepfusion_runtime_config = None

# Try grouped Qwen3 MoE decode kernels
try:
    from ..kernels.qwen3_moe import (
        qwen3_moe_grouped_decode,
        qwen3_moe_grouped_decode_int8,
        qwen3_moe_topk_softmax_compact_pack,
        qwen3_moe_grouped_prefers_triton_shape,
        qwen3_moe_grouped_runtime_config,
        qwen3_moe_router_topk_softmax,
        qwen3_moe_dominant_expert_padded_bmm_prefill,
        qwen3_moe_padded_bmm_prefill,
        qwen3_moe_prepare_segmented_prefill_graph_workspace,
        qwen3_moe_segmented_prefers_triton_shape,
        qwen3_moe_segmented_prefill,
        qwen3_moe_topk_softmax,
        HAS_QWEN3_MOE_GROUPED,
    )
except Exception:
    qwen3_moe_grouped_decode = None
    qwen3_moe_grouped_decode_int8 = None
    qwen3_moe_topk_softmax_compact_pack = None
    qwen3_moe_grouped_prefers_triton_shape = None
    qwen3_moe_grouped_runtime_config = None
    qwen3_moe_router_topk_softmax = None
    qwen3_moe_dominant_expert_padded_bmm_prefill = None
    qwen3_moe_padded_bmm_prefill = None
    qwen3_moe_prepare_segmented_prefill_graph_workspace = None
    qwen3_moe_segmented_prefers_triton_shape = None
    qwen3_moe_segmented_prefill = None
    qwen3_moe_topk_softmax = None
    HAS_QWEN3_MOE_GROUPED = False

try:
    from ..kernels.gemma4_moe_router import (
        gemma4_moe_prefill_router_prefers_shape,
        gemma4_moe_prefill_router_topk,
    )
except Exception:
    gemma4_moe_prefill_router_prefers_shape = None
    gemma4_moe_prefill_router_topk = None

try:
    from ..kernels.gemma4_grouped_prefill import (
        gemma4_grouped_mm_prefill,
        gemma4_grouped_mm_prefill_prefers_shape,
    )
except Exception:
    gemma4_grouped_mm_prefill = None
    gemma4_grouped_mm_prefill_prefers_shape = None

# Try decode fused RMSNorm + Linear (for input_layernorm + qkv)
try:
    from ..kernels.fused_rmsnorm_linear import (
        fused_rmsnorm_linear,
        HAS_FUSED_RMSNORM_LINEAR,
        fused_rmsnorm_linear_prefers_triton_shape,
        fused_rmsnorm_linear_runtime_config,
    )
except Exception:
    fused_rmsnorm_linear = None
    HAS_FUSED_RMSNORM_LINEAR = False
    fused_rmsnorm_linear_prefers_triton_shape = None
    fused_rmsnorm_linear_runtime_config = None

# Try decode fused LM head + argmax
try:
    from ..kernels.lm_head_argmax import (
        lm_head_argmax,
        lm_head_rmsnorm_argmax,
        logits_softcap_argmax,
        HAS_FUSED_LM_HEAD_ARGMAX,
        HAS_FUSED_SOFTCAP_ARGMAX,
        lm_head_argmax_prefers_triton_shape,
        lm_head_argmax_runtime_config,
    )
except Exception:
    lm_head_argmax = None
    lm_head_rmsnorm_argmax = None
    logits_softcap_argmax = None
    HAS_FUSED_LM_HEAD_ARGMAX = False
    HAS_FUSED_SOFTCAP_ARGMAX = False
    lm_head_argmax_prefers_triton_shape = None
    lm_head_argmax_runtime_config = None

# Try Fused Add + RMSNorm (Triton)
try:
    from ..kernels.fused_add_rmsnorm import fused_add_rmsnorm
    _HAS_FUSED_ADD_RMSNORM = True
except Exception:
    fused_add_rmsnorm = None
    _HAS_FUSED_ADD_RMSNORM = False

# Raw Triton kernels for inline flat-decode (bypass wrapper overhead)
try:
    from ..kernels.fused_add_rmsnorm import _fused_add_rmsnorm_single_pass as _inline_fused_add_norm
    _HAS_INLINE_FUSED_NORM = True
except Exception:
    _inline_fused_add_norm = None
    _HAS_INLINE_FUSED_NORM = False
try:
    from ..kernels.swiglu import _mg_swiglu_fwd_kernel as _inline_swiglu_kernel
    _HAS_INLINE_SWIGLU = True
except Exception:
    _inline_swiglu_kernel = None
    _HAS_INLINE_SWIGLU = False

# INT8 inline decode: import Triton small-M kernel + TC detection
try:
    from ..kernels.int8_gemm import (
        int8_small_m_gemm as _flat_int8_small_m_gemm,
        _get_triton_int8_support as _flat_get_triton_int8_support,
    )
    _HAS_FLAT_INT8_TRITON = True
except Exception:
    _flat_int8_small_m_gemm = None
    _flat_get_triton_int8_support = None
    _HAS_FLAT_INT8_TRITON = False

# W8A16 fused GEMV: the fast path — dequant weights on-the-fly, no activation quant
try:
    from ..kernels.w8a16_gemv import (
        w8a16_gemv as _flat_w8a16_gemv,
        w8a16_gemv_direct as _flat_w8a16_direct,
        precompute_w8a16_grid as _flat_w8a16_grid,
        HAS_W8A16_GEMV,
    )
    _HAS_FLAT_W8A16_GEMV = HAS_W8A16_GEMV
except Exception:
    _flat_w8a16_gemv = None
    _flat_w8a16_direct = None
    _flat_w8a16_grid = None
    _HAS_FLAT_W8A16_GEMV = False

# W4A16 fused GEMV: AWQ INT4 fast path
try:
    from ..kernels.w4a16_gemv import (
        w4a16_gemv_direct as _flat_w4a16_direct,
        precompute_w4a16_grid as _flat_w4a16_grid,
        HAS_W4A16_GEMV,
    )
    _HAS_FLAT_W4A16_GEMV = HAS_W4A16_GEMV
except Exception:
    _flat_w4a16_direct = None
    _flat_w4a16_grid = None
    _HAS_FLAT_W4A16_GEMV = False

try:
    from ..quantization.w8a16 import _check_int_mm_support as _flat_check_int_mm
except Exception:
    _flat_check_int_mm = None

# Try Fused RMSNorm + SiLU gate (Triton)
try:
    from ..kernels.rmsnorm_gated import rmsnorm_gated
    _HAS_RMSNORM_GATED = True
except Exception:
    rmsnorm_gated = None
    _HAS_RMSNORM_GATED = False
try:
    from ..kernels.rmsnorm_gated_linear import (
        rmsnorm_gated_linear_decode,
        rmsnorm_gated_linear_runtime_config,
        HAS_RMSNORM_GATED_LINEAR,
    )
except Exception:
    rmsnorm_gated_linear_decode = None
    rmsnorm_gated_linear_runtime_config = None
    HAS_RMSNORM_GATED_LINEAR = False

# Try Dao-AILab causal-conv1d
try:
    from causal_conv1d import causal_conv1d_fn
    _HAS_CAUSAL_CONV1D = True
except Exception:
    causal_conv1d_fn = None
    _HAS_CAUSAL_CONV1D = False
try:
    import megagemm_decode_ops as _decode_loop_ops
    _HAS_CPP_DECODE_LOOP = True
except Exception:
    _decode_loop_ops = None
    _HAS_CPP_DECODE_LOOP = False
# Fused RoPE + KV write + paged decode.
# Disable via env: MEGAGEMM_FUSED_ROPE_ATTN=0
def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


_HAS_FUSED_ROPE_ATTN = _HAS_TRITON and _env_enabled(
    "MEGAGEMM_FUSED_ROPE_ATTN", default=True
)
_DEBUG_FUSED_ROPE_ATTN = _env_enabled(
    "MEGAGEMM_DEBUG_FUSED_ROPE_ATTN", default=False
)
_USE_FAST_GEMV = HAS_TRITON_FAST_GEMV and _env_enabled(
    "MEGAGEMM_FAST_GEMV", default=True
)
_FAST_GEMV_MAX_ROWS = max(1, int(os.environ.get("MEGAGEMM_FAST_GEMV_MAX_ROWS", "4")))
_FAST_GEMV_KNOWN_OPS = {
    "qkv",
    "o_proj",
    "gate_up",
    "down",
    "lm_head",
    "linear_attn_in",
    "linear_attn_out",
}


def _parse_fast_gemv_ops() -> set:
    raw = os.environ.get("MEGAGEMM_FAST_GEMV_OPS", "").strip().lower()
    if not raw:
        # Hybrid default: keep fast GEMV only where it tends to help E2E decode.
        return {"gate_up", "down"}
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    if "all" in parts:
        return set(_FAST_GEMV_KNOWN_OPS)
    if "none" in parts:
        return set()
    selected = set()
    for p in parts:
        if p in _FAST_GEMV_KNOWN_OPS:
            selected.add(p)
    return selected


_FAST_GEMV_ENABLED_OPS = _parse_fast_gemv_ops()
_BENCHMARK_FORCED_TOKEN_ID = _env_int(
    "MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID", -1
)
_DECODE_TIMING = _env_enabled("MEGAGEMM_DECODE_TIMING", default=False)
_DECODE_TIMING_PRINT = _env_enabled("MEGAGEMM_DECODE_TIMING_PRINT", default=True)
_DECODE_TIMING_DETAIL = _env_enabled("MEGAGEMM_DECODE_TIMING_DETAIL", default=False)
_PREFILL_TIMING = _env_enabled("MEGAGEMM_PREFILL_TIMING", default=False)
_PREFILL_TIMING_PRINT = _env_enabled("MEGAGEMM_PREFILL_TIMING_PRINT", default=True)
_USE_CPP_DECODE_LOOP = _HAS_CPP_DECODE_LOOP and _env_enabled(
    "MEGAGEMM_CPP_DECODE_LOOP", default=True
)
_USE_DECODE_FAST_LINEAR = fast_linear is not None and _env_enabled(
    "MEGAGEMM_DECODE_FAST_LINEAR", default=False
)
_USE_PREFILL_FAST_LINEAR = fast_linear is not None and _env_enabled(
    "MEGAGEMM_PREFILL_FAST_LINEAR", default=False
)
_QWEN35_LINEAR_CORE_FP16_OUT = _env_enabled(
    "MEGAGEMM_QWEN35_LINEAR_CORE_FP16_OUT", default=True
)
_QWEN35_REUSE_LINEAR_DECODE_BUFFERS = _env_enabled(
    "MEGAGEMM_QWEN35_REUSE_LINEAR_DECODE_BUFFERS", default=True
)
_QWEN35_FUSED_NORM_OUT = HAS_RMSNORM_GATED_LINEAR and _env_enabled(
    "MEGAGEMM_QWEN35_FUSED_NORM_OUT", default=True
)
_QWEN35_FUSED_NORM_OUT_MAX_HIDDEN = max(
    0, _env_int("MEGAGEMM_QWEN35_FUSED_NORM_OUT_MAX_HIDDEN", 0)
)
_USE_NATIVE_MLP_PREFILL = HAS_NATIVE_MLP_PREFILL and _env_enabled(
    "MEGAGEMM_NATIVE_MLP_PREFILL", default=False
)
_PREFILL_REUSE_MAX_MB = max(
    0,
    _env_int("MEGAGEMM_PREFILL_REUSE_MAX_MB", 8),
)
_GEMMA4_PREFILL_GRAPH_FUSED_ATTN_FRONTEND = _env_enabled(
    "MEGAGEMM_GEMMA4_PREFILL_GRAPH_FUSED_ATTN_FRONTEND",
    default=False,
)
_USE_DEEPFUSION_MLP = HAS_DEEPFUSION_MLP and _env_enabled(
    "MEGAGEMM_DEEPFUSION_MLP", default=True
)
_USE_DEEPFUSION_MLP_PREFILL = HAS_DEEPFUSION_MLP and _env_enabled(
    "MEGAGEMM_DEEPFUSION_MLP_PREFILL", default=False
)
_USE_QWEN3_MOE_GROUPED_DECODE = HAS_QWEN3_MOE_GROUPED and _env_enabled(
    "MEGAGEMM_QWEN3_MOE_GROUPED_DECODE", default=True
)
_USE_QWEN3_MOE_SEGMENTED_PREFILL = HAS_QWEN3_MOE_GROUPED and _env_enabled(
    "MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL", default=False
)
_QWEN3_MOE_SEGMENTED_PREFILL_MIN_ASSIGNMENTS = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_MIN_ASSIGNMENTS", 4096),
)
_USE_QWEN3_MOE_SORTED_PREFILL = _env_enabled(
    "MEGAGEMM_QWEN3_MOE_SORTED_PREFILL", default=True
)
_QWEN3_MOE_SORTED_PREFILL_MIN_ASSIGNMENTS = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_SORTED_PREFILL_MIN_ASSIGNMENTS", 65),
)
_USE_QWEN3_MOE_BATCHED_PREFILL = _env_enabled(
    "MEGAGEMM_QWEN3_MOE_BATCHED_PREFILL", default=True
)
_QWEN3_MOE_BATCHED_PREFILL_MIN_ASSIGNMENTS = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_BATCHED_PREFILL_MIN_ASSIGNMENTS", 65),
)
_USE_QWEN3_MOE_BUCKETED_PREFILL = _env_enabled(
    "MEGAGEMM_QWEN3_MOE_BUCKETED_PREFILL", default=True
)
_QWEN3_MOE_BUCKETED_PREFILL_MIN_ASSIGNMENTS = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_BUCKETED_PREFILL_MIN_ASSIGNMENTS", 4096),
)
_QWEN3_MOE_BUCKETED_PREFILL_BUCKET_SIZE = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_BUCKETED_PREFILL_BUCKET_SIZE", 512),
)
_USE_QWEN3_MOE_INT8_DEQUANT_PREFILL = _env_enabled(
    "MEGAGEMM_QWEN3_MOE_INT8_DEQUANT_PREFILL", default=True
)
_QWEN3_MOE_INT8_DEQUANT_PREFILL_MIN_ASSIGNMENTS = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_INT8_DEQUANT_PREFILL_MIN_ASSIGNMENTS", 257),
)
_QWEN3_MOE_GROUPED_DEBUG = _env_enabled(
    "MEGAGEMM_QWEN3_MOE_GROUPED_DEBUG", default=False
)
_QWEN3_MOE_GROUPED_LOGGED = False
_USE_FUSED_RMSNORM_QKV_DECODE = HAS_FUSED_RMSNORM_LINEAR and _env_enabled(
    "MEGAGEMM_FUSED_RMSNORM_QKV_DECODE", default=True
)
_DECODE_CUDA_GRAPHS_ENABLED = _env_enabled(
    "MEGAGEMM_DECODE_CUDA_GRAPHS", default=False
)
_FUSED_RMSNORM_QKV_ALLOW_CUDA_GRAPHS = _env_enabled(
    "MEGAGEMM_FUSED_RMSNORM_QKV_ALLOW_CUDA_GRAPHS", default=False
)
_USE_FUSED_RMSNORM_QKV_PREFILL = HAS_FUSED_RMSNORM_LINEAR and _env_enabled(
    "MEGAGEMM_FUSED_RMSNORM_QKV_PREFILL", default=False
)
_USE_FUSED_RMSNORM_GATEUP_DECODE = HAS_FUSED_RMSNORM_LINEAR and _env_enabled(
    "MEGAGEMM_FUSED_RMSNORM_GATEUP_DECODE", default=True
)
_QWEN35_FUSED_RMSNORM_IN_PROJ_DECODE = HAS_FUSED_RMSNORM_LINEAR and _env_enabled(
    "MEGAGEMM_QWEN35_FUSED_RMSNORM_IN_PROJ_DECODE", default=True
)
_QWEN35_FUSED_RMSNORM_IN_PROJ_MAX_HIDDEN = max(
    0, _env_int("MEGAGEMM_QWEN35_FUSED_RMSNORM_IN_PROJ_MAX_HIDDEN", 0)
)
_QWEN35_FLAT_HYBRID_FULL_INLINE_MAX_HIDDEN = max(
    0, _env_int("MEGAGEMM_QWEN35_FLAT_HYBRID_FULL_INLINE_MAX_HIDDEN", 2048)
)
_USE_FUSED_LM_HEAD_ARGMAX_DECODE = HAS_FUSED_LM_HEAD_ARGMAX and _env_enabled(
    "MEGAGEMM_FUSED_LM_HEAD_ARGMAX_DECODE", default=True
)
_USE_FUSED_RMSNORM_LM_HEAD_ARGMAX_DECODE = (
    HAS_FUSED_LM_HEAD_ARGMAX
    and lm_head_rmsnorm_argmax is not None
    and _env_enabled("MEGAGEMM_FUSED_RMSNORM_LM_HEAD_ARGMAX_DECODE", default=True)
)
_GEMMA4_BATCH_CUBLAS_LM_HEAD = _env_enabled(
    "MEGAGEMM_GEMMA4_BATCH_CUBLAS_LM_HEAD",
    # Paid A100/BF16/B16 validation: exact 16x64 token matrix and +2.9%
    # scheduler decode throughput with graph-token burst replay.
    default=True,
)
_GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX = (
    HAS_FUSED_SOFTCAP_ARGMAX
    and logits_softcap_argmax is not None
    and _env_enabled(
        "MEGAGEMM_GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX",
        default=False,
    )
)
_GEMMA4_FUSED_QKV_DECODE = _env_enabled(
    "MEGAGEMM_GEMMA4_FUSED_QKV_DECODE", default=True
)
_GEMMA4_FUSED_QKV_PREFILL = _env_enabled(
    "MEGAGEMM_GEMMA4_FUSED_QKV_PREFILL", default=True
)
_GEMMA4_FUSED_ATTN_PREP_PREFILL = _env_enabled(
    "MEGAGEMM_GEMMA4_FUSED_ATTN_PREP_PREFILL", default=True
)
_GEMMA4_IMPLICIT_CAUSAL_PREFILL = _env_enabled(
    "MEGAGEMM_GEMMA4_IMPLICIT_CAUSAL_PREFILL", default=True
)
_GEMMA4_LONG_SLIDING_PREFILL = _env_enabled(
    "MEGAGEMM_GEMMA4_LONG_SLIDING_PREFILL", default=True
)
_GEMMA4_LONG_FULL_PREFILL = _env_enabled(
    "MEGAGEMM_GEMMA4_LONG_FULL_PREFILL", default=True
)
_GEMMA4_VECTORIZED_PREFILL_KV = _env_enabled(
    "MEGAGEMM_GEMMA4_VECTORIZED_PREFILL_KV", default=True
)
_GEMMA4_FUSED_DUAL_FFN_NORM_PREFILL = _env_enabled(
    "MEGAGEMM_GEMMA4_FUSED_DUAL_FFN_NORM_PREFILL", default=True
)
_GEMMA4_FUSED_ATTN_MOE_BRIDGE_PREFILL = _env_enabled(
    "MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_BRIDGE_PREFILL", default=True
)
_GEMMA4_FUSED_ATTN_MOE_BRIDGE_DECODE = _env_enabled(
    # A100 BF16/B16 gate: bit-exact including residual aliasing, stable, and
    # 1.519x faster (14.275 -> 9.395 us per layer).
    "MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_BRIDGE_DECODE", default=True
)
_GEMMA4_FUSED_ATTN_MOE_ROUTER_BRIDGE_DECODE = _env_enabled(
    # A100 BF16/B16 v110 gate: byte-exact including aliases, stable, and
    # 1.276x faster (13.460 -> 10.552 us per layer).
    "MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_ROUTER_BRIDGE_DECODE", default=True
)
_GEMMA4_FUSED_ATTN_MOE_ROUTER_SINGLE_KERNEL_DECODE = _env_enabled(
    # A100 BF16/B16 v127 gate: byte-exact including aliases, stable, and
    # 1.329x faster (10.629 -> 7.997 us per layer). The loaded-model A/B
    # exercised this path in all 30 layers (60 hits) without fallback.
    "MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_ROUTER_SINGLE_KERNEL_DECODE",
    default=True,
)
_GEMMA4_FUSED_ADD_FFN_NORM_PREFILL = _env_enabled(
    "MEGAGEMM_GEMMA4_FUSED_ADD_FFN_NORM_PREFILL", default=True
)
_GEMMA4_FUSED_POST_FFN_NORMS_PREFILL = _env_enabled(
    "MEGAGEMM_GEMMA4_FUSED_POST_FFN_NORMS_PREFILL", default=True
)
_GEMMA4_FUSED_MOE_ROUTER_PREFILL = _env_enabled(
    "MEGAGEMM_GEMMA4_FUSED_MOE_ROUTER_PREFILL", default=False
)
_GEMMA4_FUSED_MOE_ROUTER_DECODE = _env_enabled(
    "MEGAGEMM_GEMMA4_FUSED_MOE_ROUTER_DECODE", default=False
)
_GEMMA4_FUSED_ROUTER_COMPACT_PACK_DECODE = _env_enabled(
    # Experimental A100 BF16/B16 path. The cheap checkpoint-free gate runs
    # before any model download and promotes only exact, stable wins.
    "MEGAGEMM_GEMMA4_FUSED_ROUTER_COMPACT_PACK_DECODE", default=False
)
_GEMMA4_FUSED_GATEUP_DECODE = _env_enabled(
    "MEGAGEMM_GEMMA4_FUSED_GATEUP_DECODE", default=True
)
_GEMMA4_DEEPFUSION_MLP_DECODE = _env_enabled(
    "MEGAGEMM_GEMMA4_DEEPFUSION_MLP_DECODE", default=True
)
_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE = _env_enabled(
    # Experimental until a same-session E2B/E4B L4 A/B proves wall-time gain.
    "MEGAGEMM_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE", default=False
)
_GEMMA4_DENSE_ATTN_MLP_BRIDGE_DECODE = _env_enabled(
    # Experimental E2B/L4/BF16/B8 candidate.  It must pass both the exact
    # bridge microgate and a same-session full-model A/B before promotion.
    "MEGAGEMM_GEMMA4_DENSE_ATTN_MLP_BRIDGE_DECODE", default=False
)
_GEMMA4_E2B_CUBLASLT_GATEUP_DECODE = _env_enabled(
    # Experimental explicit-algorithm BF16 path.  The native micro-sweep must
    # provide both N=12288 and N=24576 indices; otherwise torch.mm is retained.
    "MEGAGEMM_GEMMA4_E2B_CUBLASLT_GATEUP_DECODE", default=False
)
_GEMMA4_PLE_CONDITIONED_GELU_DECODE = _env_enabled(
    # Experimental until the E2B/L4/BF16/B8 full-model A/B gate passes.
    "MEGAGEMM_GEMMA4_PLE_CONDITIONED_GELU_DECODE", default=False
)
_GEMMA4_PARALLEL_MOE_DECODE = _env_enabled(
    "MEGAGEMM_GEMMA4_PARALLEL_MOE_DECODE", default=True
)
_GEMMA4_PARALLEL_MOE_PREFILL = _env_enabled(
    # A100 BF16 B16x25 gate: exact, stable, and 1.021x faster
    # (1266.48 -> 1240.06 us per layer branch pair).
    "MEGAGEMM_GEMMA4_PARALLEL_MOE_PREFILL", default=True
)
_GEMMA4_FUSED_POST_MOE_NORM_RESIDUAL_PREFILL = _env_enabled(
    "MEGAGEMM_GEMMA4_FUSED_POST_MOE_NORM_RESIDUAL_PREFILL", default=True
)
_GEMMA4_FUSED_POST_MOE_NORM_RESIDUAL_DECODE = _env_enabled(
    # The isolated kernel won its microbench, but the first full B16 graph run
    # later hit an illegal CUDA access. Keep it experimental until the complete
    # capture -> replay -> new-prefill lifecycle is proven.
    "MEGAGEMM_GEMMA4_FUSED_POST_MOE_NORM_RESIDUAL_DECODE", default=False
)
_GEMMA4_FUSED_EXPERT_REDUCE_POST_MOE_DECODE = _env_enabled(
    "MEGAGEMM_GEMMA4_FUSED_EXPERT_REDUCE_POST_MOE_DECODE", default=True
)
_GEMMA4_FUSED_NEXT_ATTN_NORM_DECODE = _env_enabled(
    # v81 proved exact 16x64 greedy tokens, stable graph replay, and a 1.221%
    # end-to-end decode gain on A100 BF16. This is now the B16 baseline.
    "MEGAGEMM_GEMMA4_FUSED_NEXT_ATTN_NORM_DECODE", default=True
)
_GEMMA4_FUSED_ROUTER_EXPERT_INPUT_NORM_DECODE = _env_enabled(
    # Same policy as the post-MoE fusion above: microbench evidence is not
    # sufficient to promote a path that failed the full graph lifecycle.
    "MEGAGEMM_GEMMA4_FUSED_ROUTER_EXPERT_INPUT_NORM_DECODE", default=False
)
_GEMMA4_A100_A4B_TUNED_MLP = _env_enabled(
    "MEGAGEMM_GEMMA4_A100_A4B_TUNED_MLP", default=True
)
_GEMMA4_FORCE_FUSED_GATEUP_USE = _env_enabled(
    "MEGAGEMM_GEMMA4_FORCE_FUSED_GATEUP_USE", default=False
)
_GEMMA4_FORCE_DEEPFUSION_USE = _env_enabled(
    "MEGAGEMM_GEMMA4_FORCE_DEEPFUSION_USE", default=False
)
_GEMMA4_MLP_FUSION_DEBUG = _env_enabled(
    "MEGAGEMM_GEMMA4_MLP_FUSION_DEBUG", default=False
)
_FUSED_RMSNORM_QKV_MIN_GAIN = max(
    0.0, min(0.3, _env_float("MEGAGEMM_FUSED_RMSNORM_QKV_MIN_GAIN", 0.02))
)
_FUSED_RMSNORM_GATEUP_MIN_GAIN = max(
    0.0, min(0.3, _env_float("MEGAGEMM_FUSED_RMSNORM_GATEUP_MIN_GAIN", 0.05))
)
_QWEN35_FUSED_NORM_OUT_MIN_GAIN = max(
    0.0, min(0.3, _env_float("MEGAGEMM_QWEN35_FUSED_NORM_OUT_MIN_GAIN", 0.10))
)
_QWEN35_FUSED_NORM_OUT_GLOBAL_USE_CACHE = {}
_QWEN35_FUSED_RMSNORM_IN_PROJ_MIN_GAIN = max(
    0.0, min(0.3, _env_float("MEGAGEMM_QWEN35_FUSED_RMSNORM_IN_PROJ_MIN_GAIN", 0.10))
)
_QWEN35_FUSED_RMSNORM_IN_PROJ_DECISION_CACHE = {}
_DEEPFUSION_MLP_MIN_GAIN = max(
    0.0, min(0.3, _env_float("MEGAGEMM_DEEPFUSION_MLP_MIN_GAIN", 0.0))
)


def _gemma4_a100_a4b_tuned_mlp_shape(
    rows: int,
    hidden_dim: int,
    intermediate_dim: int,
    dtype: torch.dtype,
    device_name: str,
) -> bool:
    return bool(
        _GEMMA4_A100_A4B_TUNED_MLP
        and int(rows) == 1
        and int(hidden_dim) == 2816
        and int(intermediate_dim) == 2112
        and dtype == torch.bfloat16
        and "A100" in str(device_name).upper()
    )


def _gemma4_a100_a4b_fused_qkv_prefill_shape(
    rows: int,
    hidden_dim: int,
    q_size: int,
    k_size: int,
    v_size: int,
    dtype: torch.dtype,
    device_name: str,
) -> bool:
    qkv_shape = (int(q_size), int(k_size), int(v_size))
    row_count = int(rows)
    return bool(
        _GEMMA4_FUSED_QKV_PREFILL
        and int(hidden_dim) == 2816
        and (
            (
                (0 < row_count <= 32 or row_count in (200, 400))
                and qkv_shape in ((4096, 2048, 2048), (8192, 1024, 1024))
            )
            or (
                row_count in (16_384, 32_768)
                and qkv_shape == (4096, 2048, 2048)
            )
        )
        and dtype == torch.bfloat16
        and "A100" in str(device_name).upper()
    )


def _gemma4_a100_a4b_fused_attn_prepare_shape(
    batch_size: int,
    rows: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    dtype: torch.dtype,
    device_name: str,
) -> bool:
    head_shape = (int(num_q_heads), int(num_kv_heads), int(head_dim))
    batch = int(batch_size)
    seq_len = int(rows)
    return bool(
        _GEMMA4_FUSED_ATTN_PREP_PREFILL
        and (
            (
                (
                    (batch == 1 and 0 < seq_len <= 32)
                    or (batch in (8, 16) and seq_len == 25)
                )
                and head_shape in ((16, 8, 256), (16, 2, 512))
            )
            or (
                batch in (8, 16)
                and seq_len == 2048
                and head_shape in ((16, 8, 256), (16, 2, 512))
            )
        )
        and int(rotary_dim) == int(head_dim)
        and dtype == torch.bfloat16
        and "A100" in str(device_name).upper()
    )


def _gemma4_a100_a4b_long_kv_scatter_tokens_per_program(
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device_name: str,
) -> int:
    if not (
        int(batch_size) in (8, 16)
        and int(seq_len) == 2048
        and dtype == torch.bfloat16
        and "A100" in str(device_name).upper()
    ):
        return 1
    kv_shape = (int(num_kv_heads), int(head_dim))
    if kv_shape == (8, 256):
        return 4
    if kv_shape == (2, 512):
        return 2
    return 1


def _gemma4_a100_a4b_fused_router_prefill_shape(
    rows: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    dtype: torch.dtype,
    device_name: str,
) -> bool:
    return bool(
        _GEMMA4_FUSED_MOE_ROUTER_PREFILL
        and callable(gemma4_moe_prefill_router_prefers_shape)
        and int(rows) in (25, 400)
        and int(hidden_dim) == 2816
        and int(num_experts) == 128
        and int(top_k) == 8
        and dtype == torch.bfloat16
        and "A100" in str(device_name).upper()
    )


def _gemma4_a100_a4b_fused_router_decode_shape(
    rows: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    dtype: torch.dtype,
    device_name: str,
) -> bool:
    return bool(
        _GEMMA4_FUSED_MOE_ROUTER_DECODE
        and callable(gemma4_moe_prefill_router_prefers_shape)
        and 0 < int(rows) <= 32
        and int(hidden_dim) == 2816
        and int(num_experts) == 128
        and int(top_k) == 8
        and dtype == torch.bfloat16
        and "A100" in str(device_name).upper()
    )


def _gemma4_a100_a4b_prefill_graph_shape(
    config,
    *,
    num_seqs: int,
    total_tokens: int,
    dtype: torch.dtype,
    device_type: str,
    device_name: str,
) -> bool:
    """Narrow policy for the graph-safe Gemma 4 A4B prefill rollout."""
    layer_types = list(getattr(config, "layer_types", ()) or ())
    is_moe_layer = getattr(config, "is_moe_layer", None)
    graph_shape = (int(num_seqs), int(total_tokens))
    route_pack_ready = bool(
        (
            graph_shape == (1, 25)
            and _GEMMA4_A4B_SEGMENTED_PREFILL_SHORT_OPTIONS.get(
                "fixed_route_pack", False
            )
        )
        or (
            graph_shape == (16, 400)
            and _GEMMA4_A4B_SEGMENTED_PREFILL_OPTIONS.get(
                "compact_route_pack", False
            )
        )
    )
    return bool(
        str(getattr(config, "model_type", "")) == "gemma4_text"
        and bool(getattr(config, "enable_moe_block", False))
        and int(getattr(config, "hidden_size", 0)) == 2816
        and int(getattr(config, "num_hidden_layers", 0)) == 30
        and int(getattr(config, "num_experts", 0)) == 128
        and int(getattr(config, "num_experts_per_tok", 0)) == 8
        and int(getattr(config, "moe_intermediate_size", 0)) == 704
        and int(getattr(config, "hidden_size_per_layer_input", 0)) == 0
        and int(getattr(config, "num_kv_shared_layers", 0)) == 0
        and len(layer_types) == 30
        and layer_types.count("sliding_attention") == 25
        and layer_types.count("full_attention") == 5
        and callable(is_moe_layer)
        and all(bool(is_moe_layer(layer_idx)) for layer_idx in range(30))
        and graph_shape in ((1, 25), (16, 400))
        and dtype == torch.bfloat16
        and str(device_type) == "cuda"
        and "A100" in str(device_name).upper()
        and route_pack_ready
    )


def _gemma4_a100_a4b_decode_graph_shape(
    config,
    *,
    num_seqs: int,
    dtype: torch.dtype,
    device_type: str,
    device_name: str,
) -> bool:
    layer_types = list(getattr(config, "layer_types", ()) or ())
    is_moe_layer = getattr(config, "is_moe_layer", None)
    return bool(
        str(getattr(config, "model_type", "")) == "gemma4_text"
        and bool(getattr(config, "enable_moe_block", False))
        and int(getattr(config, "hidden_size", 0)) == 2816
        and int(getattr(config, "num_hidden_layers", 0)) == 30
        and int(getattr(config, "num_experts", 0)) == 128
        and int(getattr(config, "num_experts_per_tok", 0)) == 8
        and int(getattr(config, "moe_intermediate_size", 0)) == 704
        and int(getattr(config, "hidden_size_per_layer_input", 0)) == 0
        and int(getattr(config, "num_kv_shared_layers", 0)) == 0
        and len(layer_types) == 30
        and layer_types.count("sliding_attention") == 25
        and layer_types.count("full_attention") == 5
        and callable(is_moe_layer)
        and all(bool(is_moe_layer(layer_idx)) for layer_idx in range(30))
        and int(num_seqs) in (1, 2, 4, 8, 16)
        and dtype == torch.bfloat16
        and str(device_type) == "cuda"
        and "A100" in str(device_name).upper()
    )


def _gemma4_l4_e2b_decode_graph_shape(
    config,
    *,
    num_seqs: int,
    dtype: torch.dtype,
    device_type: str,
    device_name: str,
) -> bool:
    """Exact graph policy for the native Gemma 4 E2B/L4 decode executor."""
    layer_types = list(getattr(config, "layer_types", ()) or ())
    return bool(
        str(getattr(config, "model_type", "")) == "gemma4_text"
        and not bool(getattr(config, "enable_moe_block", False))
        and int(getattr(config, "hidden_size", 0)) == 1536
        and int(getattr(config, "num_hidden_layers", 0)) == 35
        and int(getattr(config, "num_attention_heads", 0)) == 8
        and int(getattr(config, "num_key_value_heads", 0)) == 1
        and int(getattr(config, "num_kv_shared_layers", 0)) == 20
        and len(layer_types) == 35
        and layer_types.count("sliding_attention") == 28
        and layer_types.count("full_attention") == 7
        and (
            int(num_seqs) in (4, 8)
            or (
                int(num_seqs) in (1, 2, 4)
                and os.environ.get(
                    "MEGAGEMM_GEMMA4_E2B_SMALL_BATCH_GRAPH_EXPERIMENT", "0"
                ) == "1"
            )
            or (
                int(num_seqs) == 1
                and os.environ.get("MEGAGEMM_GEMMA4_E2B_B1_GRAPH_EXPERIMENT", "0") == "1"
            )
        )
        and dtype == torch.bfloat16
        and str(device_type) == "cuda"
        and "L4" in str(device_name).upper()
    )


def _gemma4_a100_a4b_tuned_lm_head_shape(
    model_type: str,
    rows: int,
    hidden_dim: int,
    vocab_size: int,
    dtype: torch.dtype,
    device_name: str,
) -> bool:
    return bool(
        str(model_type) == "gemma4_text"
        and int(rows) == 1
        and int(hidden_dim) == 2816
        and int(vocab_size) == 262144
        and dtype == torch.bfloat16
        and "A100" in str(device_name).upper()
    )


def _gemma4_a100_a4b_batch_cublas_lm_head_shape(
    model_type: str,
    rows: int,
    hidden_dim: int,
    vocab_size: int,
    dtype: torch.dtype,
    device_name: str,
) -> bool:
    return bool(
        _GEMMA4_BATCH_CUBLAS_LM_HEAD
        and str(model_type) == "gemma4_text"
        and int(rows) == 16
        and int(hidden_dim) == 2816
        and int(vocab_size) == 262144
        and dtype == torch.bfloat16
        and "A100" in str(device_name).upper()
    )


def _gemma4_a100_a4b_parallel_moe_shape(
    model_type: str,
    rows: int,
    hidden_dim: int,
    shared_intermediate: int,
    expert_intermediate: int,
    dtype: torch.dtype,
    device_name: str,
) -> bool:
    return bool(
        _GEMMA4_PARALLEL_MOE_DECODE
        and str(model_type) == "gemma4_text"
        and int(rows) in (1, 8, 16)
        and int(hidden_dim) == 2816
        and int(shared_intermediate) == 2112
        and int(expert_intermediate) == 704
        and dtype == torch.bfloat16
        and "A100" in str(device_name).upper()
    )


def _gemma4_a100_a4b_parallel_moe_prefill_shape(
    model_type: str,
    rows: int,
    hidden_dim: int,
    shared_intermediate: int,
    expert_intermediate: int,
    dtype: torch.dtype,
    device_name: str,
) -> bool:
    return bool(
        _GEMMA4_PARALLEL_MOE_PREFILL
        and str(model_type) == "gemma4_text"
        and int(rows) == 400
        and int(hidden_dim) == 2816
        and int(shared_intermediate) == 2112
        and int(expert_intermediate) == 704
        and dtype == torch.bfloat16
        and "A100" in str(device_name).upper()
    )


def _gemma4_a100_a4b_fused_moe_prefill_tail_shape(
    model_type: str,
    rows: int,
    hidden_dim: int,
    shared_intermediate: int,
    expert_intermediate: int,
    dtype: torch.dtype,
    device_name: str,
) -> bool:
    return bool(
        _GEMMA4_FUSED_POST_MOE_NORM_RESIDUAL_PREFILL
        and str(model_type) == "gemma4_text"
        and int(rows) == 400
        and int(hidden_dim) == 2816
        and int(shared_intermediate) == 2112
        and int(expert_intermediate) == 704
        and dtype == torch.bfloat16
        and "A100" in str(device_name).upper()
        and callable(rmsnorm_triton_pair_add_final_residual)
    )


def _gemma4_a100_a4b_fused_attn_moe_bridge_prefill_shape(
    model_type: str,
    rows: int,
    hidden_dim: int,
    shared_intermediate: int,
    expert_intermediate: int,
    dtype: torch.dtype,
    device_name: str,
) -> bool:
    return bool(
        _GEMMA4_FUSED_ATTN_MOE_BRIDGE_PREFILL
        and str(model_type) == "gemma4_text"
        and int(rows) == 400
        and int(hidden_dim) == 2816
        and int(shared_intermediate) == 2112
        and int(expert_intermediate) == 704
        and dtype == torch.bfloat16
        and "A100" in str(device_name).upper()
        and callable(rmsnorm_triton_attn_residual_router_bridge)
    )


def _deterministic_moe_reduce_requested(policy_enabled: bool) -> bool:
    return bool(
        policy_enabled and torch.are_deterministic_algorithms_enabled()
    )


_GEMMA4_A4B_SEGMENTED_PREFILL_OPTIONS = {
    "force": True,
    "block_m": max(1, _env_int("MEGAGEMM_GEMMA4_MOE_PREFILL_BLOCK_M", 16)),
    "block_n": max(16, _env_int("MEGAGEMM_GEMMA4_MOE_PREFILL_BLOCK_N", 128)),
    "block_k": max(16, _env_int("MEGAGEMM_GEMMA4_MOE_PREFILL_BLOCK_K", 64)),
    "fused_gate_block_n": max(
        16,
        _env_int("MEGAGEMM_GEMMA4_MOE_PREFILL_FUSED_GATE_BLOCK_N", 64),
    ),
    "num_warps": max(1, _env_int("MEGAGEMM_GEMMA4_MOE_PREFILL_NUM_WARPS", 4)),
    "num_stages": max(1, _env_int("MEGAGEMM_GEMMA4_MOE_PREFILL_NUM_STAGES", 3)),
    "fused_gate": True,
    "dense_grid": False,
    "route_scatter": True,
    "async_tiles_max_assignments": 4096,
    "compact_route_pack": _env_enabled(
        "MEGAGEMM_GEMMA4_MOE_PREFILL_COMPACT_ROUTE_PACK",
        default=True,
    ),
    "single_accumulator": False,
    "group_size_m": 8,
}
_GEMMA4_A4B_SEGMENTED_PREFILL_LARGE_ROWS_MIN = 400
_GEMMA4_A4B_SEGMENTED_PREFILL_LARGE_OPTIONS = {
    "block_m": max(
        1,
        _env_int("MEGAGEMM_GEMMA4_MOE_PREFILL_LARGE_BLOCK_M", 32),
    ),
}
_GEMMA4_A4B_SEGMENTED_PREFILL_LONG_ROWS = 16_384
_GEMMA4_A4B_SEGMENTED_PREFILL_LONG_MAX_ROWS = 32_768
_GEMMA4_A4B_LONG_DOMINANT_EXPERT_PREFILL = _env_enabled(
    "MEGAGEMM_GEMMA4_MOE_LONG_DOMINANT_EXPERT_PREFILL",
    # Paid A100 gates measured 1.105x-1.121x across the checkpoint's
    # 7.64x-15.28x dominant-expert route profiles at B16/C2048. Keep it
    # opt-in until the loaded-checkpoint token and TTFT promotion gate passes.
    default=False,
)
_GEMMA4_A4B_LONG_DOMINANT_EXPERT_MIN_SKEW = max(
    1.0,
    _env_float(
        "MEGAGEMM_GEMMA4_MOE_LONG_DOMINANT_EXPERT_MIN_SKEW",
        7.5,
    ),
)
_GEMMA4_A4B_LONG_DOMINANT_EXPERT_MAX_LIGHT_PADDING_RATIO = max(
    1.0,
    _env_float(
        "MEGAGEMM_GEMMA4_MOE_LONG_DOMINANT_EXPERT_MAX_LIGHT_PADDING_RATIO",
        1.25,
    ),
)
_GEMMA4_A4B_LONG_PADDED_BMM_PREFILL = _env_enabled(
    "MEGAGEMM_GEMMA4_MOE_LONG_PADDED_BMM_PREFILL",
    # v23 measured 7.641x-15.281x global padding on the real checkpoint.
    # Keep the experiment opt-in; production long prefill stays segmented.
    default=False,
)
_GEMMA4_A4B_LONG_PADDED_BMM_MAX_PADDING_RATIO = max(
    1.0,
    _env_float(
        "MEGAGEMM_GEMMA4_MOE_LONG_PADDED_BMM_MAX_PADDING_RATIO",
        2.0,
    ),
)
_GEMMA4_A4B_SEGMENTED_PREFILL_LONG_OPTIONS = {
    # A100 BF16, balanced 128-expert traffic. Measured at B8xC2048:
    # BM64/DN256/GN128/BK64/W4/S3 took 12.177 ms versus 13.703 ms
    # for the promoted BM64/DN128/GN64 baseline (1.125x), repeat-exact.
    "block_m": max(
        1,
        _env_int("MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_BLOCK_M", 64),
    ),
    "block_n": max(
        1,
        _env_int("MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_BLOCK_N", 256),
    ),
    "block_k": max(
        1,
        _env_int("MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_BLOCK_K", 64),
    ),
    "fused_gate_block_n": max(
        1,
        _env_int(
            "MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_FUSED_GATE_BLOCK_N",
            128,
        ),
    ),
    "num_warps": max(
        1,
        _env_int("MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_NUM_WARPS", 4),
    ),
    "num_stages": max(
        1,
        _env_int("MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_NUM_STAGES", 3),
    ),
    "async_tiles_max_assignments": max(
        1,
        _env_int(
            "MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_ASYNC_TILES_MAX_ASSIGNMENTS",
            4096,
        ),
    ),
    "compact_route_pack": _env_enabled(
        "MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_COMPACT_ROUTE_PACK",
        default=False,
    ),
    "sorted_partial": _env_enabled(
        "MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_SORTED_PARTIAL",
        default=True,
    ),
}


def _gemma4_a100_a4b_segmented_prefill_long_shape(
    rows: int,
    dtype: Optional[torch.dtype],
    device_name: str,
) -> bool:
    return bool(
        int(rows)
        in (
            _GEMMA4_A4B_SEGMENTED_PREFILL_LONG_ROWS,
            _GEMMA4_A4B_SEGMENTED_PREFILL_LONG_MAX_ROWS,
        )
        and dtype == torch.bfloat16
        and "A100" in str(device_name).upper()
    )


_GEMMA4_A4B_SEGMENTED_PREFILL_SHORT_ROWS_MAX = 32
_GEMMA4_A4B_SEGMENTED_PREFILL_SHORT_OPTIONS = {
    "block_m": max(
        1,
        _env_int(
            "MEGAGEMM_GEMMA4_MOE_PREFILL_SHORT_BLOCK_M",
            int(_GEMMA4_A4B_SEGMENTED_PREFILL_OPTIONS["block_m"]),
        ),
    ),
    "block_n": max(16, _env_int("MEGAGEMM_GEMMA4_MOE_PREFILL_SHORT_BLOCK_N", 64)),
    "block_k": max(
        16,
        _env_int(
            "MEGAGEMM_GEMMA4_MOE_PREFILL_SHORT_BLOCK_K",
            128,
        ),
    ),
    "num_warps": max(
        1,
        _env_int(
            "MEGAGEMM_GEMMA4_MOE_PREFILL_SHORT_NUM_WARPS",
            8,
        ),
    ),
    "num_stages": max(
        1,
        _env_int(
            "MEGAGEMM_GEMMA4_MOE_PREFILL_SHORT_NUM_STAGES",
            4,
        ),
    ),
    "fixed_route_pack": _env_enabled(
        "MEGAGEMM_GEMMA4_MOE_PREFILL_FIXED_ROUTE_PACK",
        default=True,
    ),
}


def _gemma4_a4b_segmented_prefill_shape(
    model_type: str,
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    top_k: int,
) -> bool:
    return bool(
        str(model_type) == "gemma4_text"
        and int(num_experts) == 128
        and int(hidden_dim) == 2816
        and int(intermediate_dim) == 704
        and int(top_k) == 8
    )
_DEEPFUSION_PREFILL_FORCE_USE = _env_enabled(
    "MEGAGEMM_DEEPFUSION_PREFILL_FORCE_USE", default=False
)
_DEEPFUSION_PREFILL_BENCH_ITERS = max(
    2, _env_int("MEGAGEMM_DEEPFUSION_PREFILL_BENCH_ITERS", 4)
)
_DEEPFUSION_PREFILL_DEBUG = _env_enabled(
    "MEGAGEMM_DEEPFUSION_PREFILL_DEBUG", default=False
)
_FUSED_LM_HEAD_ARGMAX_MIN_GAIN = max(
    0.0, min(0.3, _env_float("MEGAGEMM_FUSED_LM_HEAD_ARGMAX_MIN_GAIN", 0.03))
)
_FUSED_RMSNORM_LM_HEAD_ARGMAX_MIN_GAIN = max(
    0.0, min(0.3, _env_float("MEGAGEMM_FUSED_RMSNORM_LM_HEAD_ARGMAX_MIN_GAIN", 0.02))
)
_FORCE_FUSED_LM_HEAD_ARGMAX_USE = _env_enabled(
    "MEGAGEMM_FUSED_LM_HEAD_ARGMAX_FORCE_USE", default=False
)
_FORCE_FUSED_RMSNORM_LM_HEAD_ARGMAX_USE = _env_enabled(
    "MEGAGEMM_FUSED_RMSNORM_LM_HEAD_ARGMAX_FORCE_USE", default=False
)
_DECODE_LINEAR_PICK_DEBUG = _env_enabled(
    "MEGAGEMM_DECODE_LINEAR_PICK_DEBUG", default=False
)
_GATE_UP_MODE_AUTOPICK = _env_enabled(
    "MEGAGEMM_GATE_UP_MODE_AUTOPICK", default=True
)
_GATE_UP_FAST_TOLERANCE = max(
    1.0, min(1.30, _env_float("MEGAGEMM_GATE_UP_FAST_TOLERANCE", 1.08))
)
_LINEAR_ATTN_FAST_GEMV_TOLERANCE = max(
    0.50, min(0.98, _env_float("MEGAGEMM_LINEAR_ATTN_FAST_GEMV_TOLERANCE", 0.75))
)
_FORCE_GATE_UP_FAST = _env_enabled(
    "MEGAGEMM_FORCE_GATE_UP_FAST", default=True
)
_DECODE_LINEAR_BACKEND_CACHE = {}
_DECODE_LINEAR_MODE_CACHE = {}

# ── Zero-overhead flat decode path ──────────────────────────────────
_USE_FLAT_DECODE = _env_enabled("MEGAGEMM_FLAT_DECODE", default=True)
_USE_FLAT_BATCH_FP16_DEQUANT = _env_enabled(
    "MEGAGEMM_FLAT_BATCH_FP16_DEQUANT", default=False
)
_USE_FLAT_DEEPFUSION_DOWN_DEQUANT = _env_enabled(
    "MEGAGEMM_FLAT_DEEPFUSION_DOWN_DEQUANT", default=False
)
_USE_FLAT_DEEPFUSION_DOWN = _env_enabled(
    "MEGAGEMM_FLAT_DEEPFUSION_DOWN", default=False
)
_USE_FLAT_FAST_DOWN = _env_enabled(
    "MEGAGEMM_FLAT_FAST_DOWN", default=False
)
_FLAT_DEEPFUSION_DOWN_BENCH = _env_enabled(
    "MEGAGEMM_FLAT_DEEPFUSION_DOWN_BENCH", default=True
)
_FLAT_DEEPFUSION_DOWN_BENCH_ITERS = max(
    1, _env_int("MEGAGEMM_FLAT_DEEPFUSION_DOWN_BENCH_ITERS", 4)
)
_FLAT_DEEPFUSION_DOWN_LOG = _env_enabled(
    "MEGAGEMM_FLAT_DEEPFUSION_DOWN_LOG", default=False
)
_FLAT_FP16_DEQUANT_LOG = _env_enabled(
    "MEGAGEMM_FLAT_FP16_DEQUANT_LOG", default=False
) or _env_enabled(
    "MEGAGEMM_FLAT_DEEPFUSION_DOWN_DEQUANT_LOG", default=False
)
_FLAT_FP16_DEQUANT_LOGGED = False
_FLAT_DEEPFUSION_DOWN_DEQUANT_LOGGED = False
_FLAT_BATCH_FP16_DEQUANT_MIN_BATCH = max(
    1, _env_int("MEGAGEMM_FLAT_BATCH_FP16_DEQUANT_MIN_BATCH", 4)
)
_FLAT_BATCH_FP16_DEQUANT_MAX_MB = max(
    0, _env_int("MEGAGEMM_FLAT_BATCH_FP16_DEQUANT_MAX_MB", 0)
)


def _parse_flat_fp16_dequant_ops() -> set:
    raw = os.environ.get("MEGAGEMM_FLAT_BATCH_FP16_DEQUANT_OPS", "").strip().lower()
    if not raw:
        return {"gate_up", "down"}
    parts = {p.strip() for p in raw.replace(";", ",").split(",") if p.strip()}
    known = {"qkv", "o", "gate_up", "down", "down_fused"}
    if "all" in parts:
        return known
    if "none" in parts:
        return set()
    return {p for p in parts if p in known}


_FLAT_BATCH_FP16_DEQUANT_OPS = _parse_flat_fp16_dequant_ops()


class _FlatLayerWeights:
    """Pre-collected weight references for one transformer layer.
    Uses __slots__ so attribute access is a C-level struct lookup, not dict."""
    __slots__ = (
        'layer_type',
        'qkv_wt', 'qkv_bias',
        'o_wt', 'o_bias',
        'gate_up_wt', 'gate_up_weight', 'gate_up_bias',
        'down_wt', 'down_weight', 'down_bias',
        'qkv_mod', 'o_mod', 'gate_up_mod', 'down_mod',
        'v_from_k',
        'norm1_weight', 'norm2_weight',
        'q_norm_weight', 'k_norm_weight',
        'norm_eps',
        # INT8 inline decode: cached weight/scale refs
        'qkv_int8_w', 'qkv_int8_scale',
        'o_int8_w', 'o_int8_scale',
        'gate_up_int8_w', 'gate_up_int8_scale',
        'down_int8_w', 'down_int8_scale',
        # Optional batch decode path: cached FP16 dequantized INT8 weights.
        'qkv_dequant_wt', 'o_dequant_wt',
        'gate_up_dequant_wt', 'down_dequant_wt', 'down_dequant_raw_wt',
        # AWQ INT4 inline decode: cached qweight/scales/qzeros refs
        'qkv_awq_qw', 'qkv_awq_scales', 'qkv_awq_qzeros', 'qkv_awq_gs',
        'o_awq_qw', 'o_awq_scales', 'o_awq_qzeros', 'o_awq_gs',
        'gate_up_awq_qw', 'gate_up_awq_scales', 'gate_up_awq_qzeros', 'gate_up_awq_gs',
        'down_awq_qw', 'down_awq_scales', 'down_awq_qzeros', 'down_awq_gs',
    )

    def __init__(self):
        # Read unconditionally by the common fused-QKV decode loop. Gemma 4,
        # where V may alias K, uses _Gemma4FlatLayerWeights instead.
        self.v_from_k = False


class _Gemma4FlatLayerWeights:
    """Pre-collected Gemma 4 decode weights for the hybrid flat path."""
    __slots__ = (
        'layer_idx', 'layer_type', 'is_kv_shared', 'kv_share_source',
        'sliding_window', 'num_q_heads', 'num_kv_heads', 'head_dim',
        'q_size', 'k_size', 'v_size', 'qkv_size', 'intermediate_size',
        'ple_size', 'scale', 'rotary_dim', 'half_rotate',
        'input_norm_weight', 'post_attn_norm_weight',
        'pre_ff_norm_weight', 'post_ff_norm_weight',
        'is_moe', 'moe_module',
        'pre_expert_norm_weight', 'post_shared_norm_weight', 'post_expert_norm_weight',
        'q_norm_weight', 'k_norm_weight', 'has_v_norm',
        'qkv_wt', 'qkv_bias', 'qkv_mod', 'qkv_int8_w', 'qkv_int8_scale',
        'q_wt', 'q_bias', 'q_mod', 'q_int8_w', 'q_int8_scale',
        'k_wt', 'k_bias', 'k_mod', 'k_int8_w', 'k_int8_scale',
        'v_wt', 'v_bias', 'v_mod', 'v_int8_w', 'v_int8_scale', 'v_from_k',
        'o_wt', 'o_bias', 'o_mod', 'o_int8_w', 'o_int8_scale',
        'gate_up_wt', 'gate_up_weight', 'gate_up_bias', 'gate_up_mod', 'gate_up_int8_w', 'gate_up_int8_scale',
        'down_wt', 'down_weight', 'down_bias', 'down_mod', 'down_int8_w', 'down_int8_scale',
        'ple_gate_wt', 'ple_proj_wt',
        'post_ple_norm_weight', 'layer_scalar',
    )


# ── Fused dynamic quantization Triton kernel for flat INT8 decode ───
# Replaces 5-6 separate PyTorch micro-ops (abs, amax, clamp, div, round, to)
# with a single kernel launch.  For M=1 decode this eliminates ~80% of the
# dynamic-quantization overhead.

if _HAS_TRITON and triton is not None and tl is not None:
    @triton.jit
    def _flat_fused_dyn_quant_kernel(
        x_ptr,          # [M, K] float16 input
        x_int8_ptr,     # [M, K] int8 output
        x_scale_ptr,    # [M, 1] float32 output scale
        K: tl.constexpr,
        stride_xm,
        stride_qm,
        stride_sm,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_k = tl.arange(0, BLOCK_K)
        mask = offs_k < K

        # Load the full row
        x = tl.load(x_ptr + pid_m * stride_xm + offs_k, mask=mask, other=0.0)

        # Per-row absmax -> scale
        amax = tl.max(tl.abs(x))
        amax = tl.maximum(amax, 1e-12)
        scale = (amax / 127.0).to(tl.float32)
        inv_scale = 127.0 / amax

        # Quantize to INT8 in-register
        x_q = x.to(tl.float32) * inv_scale
        x_q = tl.maximum(tl.minimum(x_q + 0.5, 127.0), -128.0).to(tl.int8)

        # Store
        tl.store(x_int8_ptr + pid_m * stride_qm + offs_k, x_q, mask=mask)
        tl.store(x_scale_ptr + pid_m * stride_sm, scale)

    _HAS_FLAT_FUSED_DYN_QUANT = True
else:
    _flat_fused_dyn_quant_kernel = None
    _HAS_FLAT_FUSED_DYN_QUANT = False


def _flat_int8_linear(
    x_2d,                           # [M, K] float16
    weight_int8,                    # [N, K] int8 row-major
    w_scale,                        # [N] float16
    bias,                           # [N] or None
    x_int8_buf,                     # [M, K_max] int8 pre-allocated (W8A8 fallback)
    x_scale_buf,                    # [M, 1] float32 pre-allocated (W8A8 fallback)
    fused_quant_block_k,            # triton.next_power_of_2(K) (W8A8 fallback)
    out=None,                       # [M, N] fp16 pre-allocated output
):
    """
    Inline INT8 linear for flat decode — bypasses Int8Linear.forward() entirely.

    Priority 1: W8A16 fused GEMV (fastest — no activation quantization needed)
      Loads INT8 weights from HBM, dequantizes on-the-fly in registers,
      does FP16 dot product. Reads HALF the memory → ~2x faster than FP16.

    Priority 2: Fused dynamic quant + Triton small-M GEMM (W8A8)
    Priority 3: torch._int_mm with padding
    Priority 4: dequant + F.linear
    """
    M, K = x_2d.shape

    # ─── Priority 1: W8A16 fused GEMV (no activation quantization!) ───
    if _HAS_FLAT_W8A16_GEMV and _flat_w8a16_gemv is not None:
        return _flat_w8a16_gemv(x_2d, weight_int8, w_scale, bias, out=out)

    # ─── Priority 2: Fused dyn quant + small-M INT8 GEMM (W8A8) ───
    # Slice buffers to actual K
    x_int8 = x_int8_buf[:M, :K].contiguous() if x_int8_buf.shape[1] != K else x_int8_buf[:M]

    if _HAS_FLAT_FUSED_DYN_QUANT and fused_quant_block_k > 0:
        _flat_fused_dyn_quant_kernel[(M,)](
            x_2d,
            x_int8,
            x_scale_buf,
            K=K,
            stride_xm=x_2d.stride(0),
            stride_qm=x_int8.stride(0),
            stride_sm=x_scale_buf.stride(0),
            BLOCK_K=fused_quant_block_k,
        )
    else:
        x_amax = x_2d.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-12).float()
        x_scale_buf[:M].copy_(x_amax.div_(127.0))
        x_f = x_2d.float()
        x_f.div_(x_scale_buf[:M]).round_().clamp_(-128, 127)
        x_int8.copy_(x_f.to(torch.int8))

    if _HAS_FLAT_INT8_TRITON and _flat_int8_small_m_gemm is not None:
        out = _flat_int8_small_m_gemm(
            x_int8, x_scale_buf[:M],
            weight_int8, w_scale, bias,
        )
        if out is not None:
            return out

    # ─── Priority 3: torch._int_mm ───
    if hasattr(torch, '_int_mm'):
        pad_m = 0
        x_q = x_int8
        if M < 16:
            pad_m = 16 - M
            x_q = torch.nn.functional.pad(x_q, (0, 0, 0, pad_m))
        wt = weight_int8.t().contiguous()
        out_i32 = torch._int_mm(x_q, wt)
        if pad_m > 0:
            out_i32 = out_i32[:M]
        out = out_i32.float() * x_scale_buf[:M] * w_scale.float().unsqueeze(0)
        out = out.to(x_2d.dtype)
        if bias is not None:
            out = out + bias
        return out

    # ─── Priority 4: dequant + F.linear ───
    w_fp = weight_int8.to(x_2d.dtype) * w_scale.unsqueeze(1).to(x_2d.dtype)
    return torch.nn.functional.linear(x_2d, w_fp, bias)


def _linear_weight_bias(linear):
    """Return (weight, bias) for nn.Linear-like modules, or (None, bias)."""
    weight = getattr(linear, "weight", None)
    # Semi-structured tensors must flow through nn.Linear/F.linear so PyTorch can
    # dispatch to its cuSPARSELt-backed sparse matmul. MegaGemm's dense custom
    # paths transpose/read the weight directly and are therefore not applicable.
    if is_semi_structured_weight(weight):
        return None, getattr(linear, "bias", None)
    # Int8Linear exposes a compatibility `weight` property, but decode hot paths
    # should use module.forward (native INT8) rather than dequantized weights.
    if hasattr(linear, "weight_int8") and hasattr(linear, "scale"):
        return None, getattr(linear, "bias", None)
    return weight, getattr(linear, "bias", None)


def _decode_linear_runtime_sig(x: torch.Tensor, weight: torch.Tensor) -> tuple:
    if x.dim() == 3:
        rows = int(x.shape[0] * x.shape[1])
    elif x.dim() == 2:
        rows = int(x.shape[0])
    else:
        rows = int(x.numel() // max(1, int(x.shape[-1])))
    return (
        rows,
        int(weight.shape[1]),
        int(weight.shape[0]),
        str(x.dtype),
        x.device.type,
        int(x.device.index) if x.device.index is not None else -1,
    )


def _can_use_fast_gemv(x: torch.Tensor) -> bool:
    if not (_USE_FAST_GEMV and fast_linear is not None and x.is_cuda):
        return False
    if torch.is_grad_enabled():
        return False
    if x.dim() == 3:
        rows = x.shape[0] * x.shape[1]
        return x.shape[1] == 1 and rows <= _FAST_GEMV_MAX_ROWS
    if x.dim() == 2:
        return x.shape[0] <= _FAST_GEMV_MAX_ROWS
    return False


def _can_use_fast_gemv_for(
    op_name: str,
    x: torch.Tensor,
    out_features: Optional[int] = None,
) -> bool:
    if op_name not in _FAST_GEMV_ENABLED_OPS:
        return False
    if not _can_use_fast_gemv(x):
        return False
    if out_features is not None and callable(fast_gemv_prefers_triton_shape):
        if x.dim() == 3:
            rows = int(x.shape[0] * x.shape[1])
        else:
            rows = int(x.shape[0])
        if not fast_gemv_prefers_triton_shape(
            int(x.shape[-1]),
            int(out_features),
            rows,
        ):
            return False
    return True


def _can_use_deepfusion_mlp_for(
    gate_up: torch.Tensor,
    down_weight: torch.Tensor,
) -> bool:
    if not (_USE_DEEPFUSION_MLP and deepfusion_swiglu_down is not None):
        return False
    if torch.is_grad_enabled() or not gate_up.is_cuda:
        return False
    if gate_up.dim() == 3:
        rows = int(gate_up.shape[0] * gate_up.shape[1])
    elif gate_up.dim() == 2:
        rows = int(gate_up.shape[0])
    else:
        return False
    i_dim = int(gate_up.shape[-1] // 2)
    h_dim = int(down_weight.shape[0])
    if callable(deepfusion_mlp_prefers_triton_shape):
        if not deepfusion_mlp_prefers_triton_shape(i_dim, h_dim, rows):
            return False
    return True


def _can_use_fused_rmsnorm_qkv_for(
    hidden_states: torch.Tensor,
    out_features: int,
) -> bool:
    if not (_USE_FUSED_RMSNORM_QKV_DECODE and fused_rmsnorm_linear is not None):
        return False
    if torch.is_grad_enabled() or not hidden_states.is_cuda:
        return False
    if hidden_states.dim() == 3:
        rows = int(hidden_states.shape[0] * hidden_states.shape[1])
    elif hidden_states.dim() == 2:
        rows = int(hidden_states.shape[0])
    else:
        return False
    if (
        _DECODE_CUDA_GRAPHS_ENABLED
        and not _FUSED_RMSNORM_QKV_ALLOW_CUDA_GRAPHS
        and rows == 1
        and int(hidden_states.shape[-1]) == 2048
        and int(out_features) == 5120
    ):
        return False
    if callable(fused_rmsnorm_linear_prefers_triton_shape):
        if not fused_rmsnorm_linear_prefers_triton_shape(
            int(hidden_states.shape[-1]),
            int(out_features),
            rows,
        ):
            return False
    return True


def _can_use_fused_rmsnorm_gateup_for(
    hidden_states: torch.Tensor,
    out_features: int,
) -> bool:
    if not (_USE_FUSED_RMSNORM_GATEUP_DECODE and fused_rmsnorm_linear is not None):
        return False
    if torch.is_grad_enabled() or not hidden_states.is_cuda:
        return False
    if hidden_states.dim() == 3:
        rows = int(hidden_states.shape[0] * hidden_states.shape[1])
    elif hidden_states.dim() == 2:
        rows = int(hidden_states.shape[0])
    else:
        return False
    if callable(fused_rmsnorm_linear_prefers_triton_shape):
        if not fused_rmsnorm_linear_prefers_triton_shape(
            int(hidden_states.shape[-1]),
            int(out_features),
            rows,
        ):
            return False
    return True


def _can_use_fused_lm_head_argmax_for(
    hidden_states: torch.Tensor,
    vocab_size: int,
) -> bool:
    if not (_USE_FUSED_LM_HEAD_ARGMAX_DECODE and lm_head_argmax is not None):
        return False
    if torch.is_grad_enabled() or not hidden_states.is_cuda:
        return False
    if hidden_states.dim() == 3:
        rows = int(hidden_states.shape[0] * hidden_states.shape[1])
    elif hidden_states.dim() == 2:
        rows = int(hidden_states.shape[0])
    else:
        return False
    if callable(lm_head_argmax_prefers_triton_shape):
        if not lm_head_argmax_prefers_triton_shape(
            int(hidden_states.shape[-1]),
            int(vocab_size),
            rows,
        ):
            return False
    return True


def _timing_enabled() -> bool:
    return _DECODE_TIMING


def _apply_benchmark_forced_token(
    next_tokens: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Override token feedback only after the real LM-head/argmax work ran."""
    token_id = int(_BENCHMARK_FORCED_TOKEN_ID)
    if token_id < 0:
        return next_tokens
    if token_id >= int(vocab_size):
        raise ValueError(
            "MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID is outside the model vocabulary: "
            f"{token_id} >= {int(vocab_size)}"
        )
    next_tokens.fill_(token_id)
    return next_tokens


def _prefill_timing_enabled() -> bool:
    return _PREFILL_TIMING


def _timing_record_start(enabled: bool):
    if not enabled:
        return None, None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return start, end


def _timing_record_end(events: Optional[dict], name: str, start_end):
    if events is None or start_end is None or start_end[0] is None:
        return
    start, end = start_end
    end.record()
    events.setdefault(name, []).append((start, end))


def _record_gemma4_prefill_finite_trace(
    trace: Optional[dict],
    layer_idx: int,
    stage: str,
    tensor: Optional[torch.Tensor],
) -> None:
    """Synchronously record one diagnostic tensor and stop at the first NaN/Inf."""
    if not trace or not trace.get("enabled") or tensor is None:
        return
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        return

    finite = torch.isfinite(tensor)
    batch_size = int(trace.get("batch_size", 0) or 0)
    seq_len = int(trace.get("seq_len", 0) or 0)
    per_batch = None
    if batch_size > 0 and tensor.ndim > 0:
        if int(tensor.shape[0]) == batch_size:
            per_batch = finite.reshape(batch_size, -1).all(dim=1)
        elif seq_len > 0 and int(tensor.shape[0]) == batch_size * seq_len:
            per_batch = finite.reshape(batch_size, seq_len, -1).all(dim=(1, 2))

    if per_batch is None:
        all_finite = bool(finite.all().item())
        finite_rows = list(range(batch_size)) if all_finite else []
        nonfinite_rows = [] if all_finite else list(range(batch_size))
    else:
        row_flags = [bool(value) for value in per_batch.detach().cpu().tolist()]
        finite_rows = [index for index, value in enumerate(row_flags) if value]
        nonfinite_rows = [index for index, value in enumerate(row_flags) if not value]
        all_finite = not nonfinite_rows

    event = {
        "layer": int(layer_idx),
        "stage": str(stage),
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype),
        "finite_row_count": len(finite_rows),
        "finite_rows": finite_rows,
        "nonfinite_rows": nonfinite_rows,
    }
    if not all_finite:
        event["nonfinite_values"] = int((~finite).sum().item())
    trace.setdefault("events", []).append(event)

    if not all_finite and trace.get("first_bad") is None:
        trace["first_bad"] = dict(event)
        trace["status"] = "NONFINITE"
        if trace.get("stop_on_nonfinite", True):
            raise RuntimeError(
                "Gemma4 prefill finite trace stopped at first nonfinite tensor: "
                f"layer={layer_idx} stage={stage} rows={nonfinite_rows}"
            )


def _gemma4_log_mlp_fusion(owner, kind: str, message: str) -> None:
    if not _GEMMA4_MLP_FUSION_DEBUG:
        return
    seen = getattr(owner, "_gemma4_mlp_fusion_debug_seen", None)
    if seen is None:
        seen = set()
        setattr(owner, "_gemma4_mlp_fusion_debug_seen", seen)
    key = (kind, message)
    if key in seen:
        return
    seen.add(key)
    print(f"[MegaGemm][gemma4-mlp] {kind}: {message}")


def _get_reusable_out(owner, attr_name: str, shape: tuple, ref: torch.Tensor) -> torch.Tensor:
    buf = getattr(owner, attr_name, None)
    if (
        buf is None
        or tuple(buf.shape) != tuple(shape)
        or buf.device != ref.device
        or buf.dtype != ref.dtype
    ):
        buf = torch.empty(shape, device=ref.device, dtype=ref.dtype)
        setattr(owner, attr_name, buf)
    return buf


def _get_reusable_out_decode(owner, attr_name: str, shape: tuple, ref: torch.Tensor) -> torch.Tensor:
    """
    Decode-only reusable output buffer helper.
    Keeps checks minimal because decode path has stable dtype/device/shapes.
    """
    buf = getattr(owner, attr_name, None)
    if buf is None or buf.shape != shape:
        buf = torch.empty(shape, device=ref.device, dtype=ref.dtype)
        setattr(owner, attr_name, buf)
    return buf


def _get_reusable_out_decode_typed(
    owner,
    attr_name: str,
    shape: tuple,
    ref: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    buf = getattr(owner, attr_name, None)
    if buf is None or tuple(buf.shape) != tuple(shape) or buf.dtype != dtype:
        buf = torch.empty(shape, device=ref.device, dtype=dtype)
        setattr(owner, attr_name, buf)
    return buf


def _get_prefill_out(
    owner,
    attr_name: str,
    shape: tuple,
    ref: torch.Tensor,
) -> torch.Tensor:
    """
    Prefill buffer helper.
    Large prefill activations must not be kept as persistent module attributes,
    otherwise each transformer layer pins a huge tensor in VRAM and causes OOM.
    """
    if _PREFILL_REUSE_MAX_MB <= 0:
        return torch.empty(shape, device=ref.device, dtype=ref.dtype)
    elem_size = max(1, int(ref.element_size()))
    total_bytes = elem_size
    for dim in shape:
        total_bytes *= int(dim)
    if total_bytes > (_PREFILL_REUSE_MAX_MB * 1024 * 1024):
        return torch.empty(shape, device=ref.device, dtype=ref.dtype)
    return _get_reusable_out(owner, attr_name, shape, ref)


def _cached_fast_gemv_decision(
    owner,
    key_attr: str,
    value_attr: str,
    op_name: str,
    x: torch.Tensor,
    out_features: int,
) -> bool:
    """
    Cache fast-GEMV dispatch decision for stable decode shapes.
    Avoids repeating Python policy checks on every token.
    """
    if x.dim() == 3:
        rows = int(x.shape[0] * x.shape[1])
    elif x.dim() == 2:
        rows = int(x.shape[0])
    else:
        return False

    key = (
        rows,
        int(x.shape[-1]),
        int(out_features),
        x.dtype,
        x.device.type,
        x.device.index,
    )
    if getattr(owner, key_attr, None) != key:
        setattr(owner, key_attr, key)
        setattr(
            owner,
            value_attr,
            _can_use_fast_gemv_for(op_name, x, out_features=out_features),
        )
    return bool(getattr(owner, value_attr, False))


def _decode_linear_backend_key(op_name: str, x: torch.Tensor, weight: torch.Tensor) -> tuple:
    if x.dim() == 3:
        rows = int(x.shape[0] * x.shape[1])
    elif x.dim() == 2:
        rows = int(x.shape[0])
    else:
        rows = int(x.numel() // max(1, int(x.shape[-1])))
    return (
        op_name,
        rows,
        int(weight.shape[1]),
        int(weight.shape[0]),
        str(x.dtype),
        x.device.type,
        int(x.device.index) if x.device.index is not None else -1,
    )


def _cached_deepfusion_decision(
    owner,
    key_attr: str,
    value_attr: str,
    gate_up: torch.Tensor,
    down_weight: torch.Tensor,
) -> bool:
    if gate_up.dim() == 3:
        rows = int(gate_up.shape[0] * gate_up.shape[1])
    elif gate_up.dim() == 2:
        rows = int(gate_up.shape[0])
    else:
        return False
    key = (
        rows,
        int(gate_up.shape[-1]),
        int(down_weight.shape[0]),
        int(down_weight.shape[1]),
        gate_up.dtype,
        gate_up.device.type,
        gate_up.device.index,
    )
    if getattr(owner, key_attr, None) != key:
        setattr(owner, key_attr, key)
        setattr(
            owner,
            value_attr,
            _can_use_deepfusion_mlp_for(gate_up, down_weight),
        )
    return bool(getattr(owner, value_attr, False))


def _deepfusion_shape_sig(
    gate_up: torch.Tensor,
    down_weight: torch.Tensor,
):
    if gate_up.dim() == 3:
        rows = int(gate_up.shape[0] * gate_up.shape[1])
    elif gate_up.dim() == 2:
        rows = int(gate_up.shape[0])
    else:
        rows = -1
    return (
        rows,
        int(gate_up.shape[-1]),
        int(down_weight.shape[0]),
        int(down_weight.shape[1]),
        gate_up.dtype,
        gate_up.device.type,
        gate_up.device.index,
    )


def _cached_fused_rmsnorm_qkv_decision(
    owner,
    key_attr: str,
    value_attr: str,
    hidden_states: torch.Tensor,
    out_features: int,
) -> bool:
    if hidden_states.dim() == 3:
        rows = int(hidden_states.shape[0] * hidden_states.shape[1])
    elif hidden_states.dim() == 2:
        rows = int(hidden_states.shape[0])
    else:
        return False
    key = (
        rows,
        int(hidden_states.shape[-1]),
        int(out_features),
        hidden_states.dtype,
        hidden_states.device.type,
        hidden_states.device.index,
    )
    if getattr(owner, key_attr, None) != key:
        setattr(owner, key_attr, key)
        setattr(
            owner,
            value_attr,
            _can_use_fused_rmsnorm_qkv_for(hidden_states, out_features),
        )
    return bool(getattr(owner, value_attr, False))


def _cached_fused_rmsnorm_gateup_decision(
    owner,
    key_attr: str,
    value_attr: str,
    hidden_states: torch.Tensor,
    out_features: int,
) -> bool:
    if hidden_states.dim() == 3:
        rows = int(hidden_states.shape[0] * hidden_states.shape[1])
    elif hidden_states.dim() == 2:
        rows = int(hidden_states.shape[0])
    else:
        return False
    key = (
        rows,
        int(hidden_states.shape[-1]),
        int(out_features),
        hidden_states.dtype,
        hidden_states.device.type,
        hidden_states.device.index,
    )
    if getattr(owner, key_attr, None) != key:
        setattr(owner, key_attr, key)
        setattr(
            owner,
            value_attr,
            _can_use_fused_rmsnorm_gateup_for(hidden_states, out_features),
        )
    return bool(getattr(owner, value_attr, False))


def _cached_fused_lm_head_argmax_decision(
    owner,
    key_attr: str,
    value_attr: str,
    hidden_states: torch.Tensor,
    vocab_size: int,
) -> bool:
    if hidden_states.dim() == 3:
        rows = int(hidden_states.shape[0] * hidden_states.shape[1])
    elif hidden_states.dim() == 2:
        rows = int(hidden_states.shape[0])
    else:
        return False
    key = (
        rows,
        int(hidden_states.shape[-1]),
        int(vocab_size),
        hidden_states.dtype,
        hidden_states.device.type,
        hidden_states.device.index,
    )
    if getattr(owner, key_attr, None) != key:
        setattr(owner, key_attr, key)
        setattr(
            owner,
            value_attr,
            _can_use_fused_lm_head_argmax_for(hidden_states, vocab_size),
        )
    return bool(getattr(owner, value_attr, False))


def _decode_linear(
    owner,
    out_attr_name: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast: Optional[bool] = None,
    fast_mode: str = "",
) -> torch.Tensor:
    """
    Decode-only linear helper:
    - Reuses output buffer
    - Uses fast_linear dispatch (Triton-or-cuBLAS) when available
    """
    if is_semi_structured_weight(weight):
        return torch.nn.functional.linear(x, weight, bias)
    out_features = int(weight.shape[0])
    out = _get_reusable_out_decode(owner, out_attr_name, (*x.shape[:-1], out_features), x)

    if use_fast is None:
        use_fast = _USE_DECODE_FAST_LINEAR
    use_fast = bool(use_fast and fast_linear is not None)
    if use_fast:
        scratch = None
        used_splitk = False
        if fast_mode == "splitk" and fast_gemv_splitk_scratch_shape is not None:
            scratch_shape = fast_gemv_splitk_scratch_shape(x, weight)
            if scratch_shape is not None:
                scratch = _get_reusable_out_decode_typed(
                    owner,
                    f"{out_attr_name}_splitk_scratch",
                    scratch_shape,
                    x,
                    torch.float32,
                )
                used_splitk = True
        result = fast_linear(
            x,
            weight,
            bias,
            out=out,
            mode_override=fast_mode,
            scratch=scratch,
        )
        if used_splitk:
            setattr(
                owner,
                "_fast_gemv_splitk_hits",
                int(getattr(owner, "_fast_gemv_splitk_hits", 0)) + 1,
            )
        return result

    # cuBLAS path with preallocated output to avoid per-token allocations.
    x_2d = x if x.ndim == 2 else x.flatten(0, -2)
    out_2d = out if out.ndim == 2 else out.flatten(0, -2)
    wt = weight.transpose(0, 1)
    if bias is None:
        torch.mm(x_2d, wt, out=out_2d)
    else:
        bias_2d = bias.unsqueeze(0).expand(x_2d.shape[0], -1)
        torch.addmm(bias_2d, x_2d, wt, out=out_2d)
    return out


def _prefill_linear(
    owner,
    out_attr_name: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Prefill-oriented linear helper:
    - flattens [B, T, H] -> [B*T, H] to force a plain GEMM path
    - reuses output buffers across chunks
    - optionally routes through fast_linear, which falls back to cuBLAS addmm/mm
      for large-row prefill shapes
    """
    if is_semi_structured_weight(weight):
        return torch.nn.functional.linear(x, weight, bias)
    out_features = int(weight.shape[0])
    out = _get_prefill_out(owner, out_attr_name, (*x.shape[:-1], out_features), x)

    if (
        _USE_PREFILL_FAST_LINEAR
        and fast_linear is not None
        and x.is_cuda
        and weight.is_cuda
        and not torch.is_grad_enabled()
    ):
        return fast_linear(x, weight, bias, out=out)

    x_2d = x if x.ndim == 2 else x.flatten(0, -2)
    out_2d = out if out.ndim == 2 else out.flatten(0, -2)
    wt = weight.transpose(0, 1)
    if bias is None:
        torch.mm(x_2d, wt, out=out_2d)
    else:
        bias_2d = bias.unsqueeze(0).expand(x_2d.shape[0], -1)
        torch.addmm(bias_2d, x_2d, wt, out=out_2d)
    return out


def _pick_gate_up_fast_mode(
    owner,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_attr_name: str,
) -> str:
    """
    Autopick fast GEMV mode for gate_up decode (tile vs row), cached by shape/device.
    """
    if not _GATE_UP_MODE_AUTOPICK:
        return "tile"
    if not (_USE_FAST_GEMV and fast_linear is not None):
        return "tile"
    if not x.is_cuda or torch.is_grad_enabled():
        return "tile"
    if not _can_use_fast_gemv(x):
        return "tile"

    out_features = int(weight.shape[0])
    key = (
        "gate_up_mode",
        int(weight.shape[1]),
        out_features,
        str(x.dtype),
        x.device.type,
        int(x.device.index) if x.device.index is not None else -1,
    )
    cached = _DECODE_LINEAR_MODE_CACHE.get(key)
    if cached is not None:
        return str(cached)

    out = _get_reusable_out_decode(owner, out_attr_name, (*x.shape[:-1], out_features), x)
    try:
        fast_linear(x, weight, bias, out=out, mode_override="tile")
        fast_linear(x, weight, bias, out=out, mode_override="row")
        torch.cuda.synchronize()
        tile_ms = _cuda_bench_ms(
            lambda: fast_linear(x, weight, bias, out=out, mode_override="tile"),
            iters=8,
        )
        row_ms = _cuda_bench_ms(
            lambda: fast_linear(x, weight, bias, out=out, mode_override="row"),
            iters=8,
        )
        mode = "row" if row_ms <= (tile_ms * 0.99) else "tile"
        if _DECODE_LINEAR_PICK_DEBUG:
            print(
                f"[decode-linear-mode] op=gate_up in={int(weight.shape[1])} out={out_features} "
                f"tile_ms={float(tile_ms):.4f} row_ms={float(row_ms):.4f} mode={mode}"
            )
    except Exception:
        mode = "tile"

    _DECODE_LINEAR_MODE_CACHE[key] = mode
    return mode


def _decode_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    offset: bool = False,
    out: Optional[torch.Tensor] = None,
    prefer_triton: bool = False,
) -> torch.Tensor:
    if not prefer_triton and _can_use_cuda_rmsnorm_for(x, offset):
        try:
            if torch.is_grad_enabled():
                result = RMSNormFunction.apply(x, weight, eps)
            else:
                result = rmsnorm_forward(x, weight, eps)
            if out is not None:
                out.copy_(result)
                return out
            return result
        except Exception:
            pass

    if _HAS_TRITON_RMSNORM and x.is_cuda:
        try:
            return rmsnorm_triton(x, weight, eps, offset, out=out)
        except Exception:
            pass

    variance = x.float().pow(2).mean(-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    w = (weight + 1.0) if offset else weight
    result = (normed * w).to(x.dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _decode_rmsnorm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)).to(x.dtype)


def _cuda_bench_ms(fn, iters: int = 8) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / max(1, iters)


def _pick_decode_linear_backend(
    owner,
    op_name: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_attr_name: str,
) -> bool:
    """
    Pick backend once per (op, shape, device, dtype):
    True  -> use fast_linear
    False -> use nn.Linear / F.linear
    """
    if not (_USE_FAST_GEMV and fast_linear is not None):
        return False
    if not x.is_cuda or torch.is_grad_enabled():
        return False

    if op_name not in _FAST_GEMV_ENABLED_OPS:
        return False
    if not _can_use_fast_gemv(x):
        return False

    out_features = int(weight.shape[0])
    key = _decode_linear_backend_key(op_name, x, weight)
    cached = _DECODE_LINEAR_BACKEND_CACHE.get(key)
    if cached is not None:
        return bool(cached)

    out = _get_reusable_out_decode(owner, out_attr_name, (*x.shape[:-1], out_features), x)
    splitk_scratch = None
    splitk_shape = None
    if fast_gemv_splitk_scratch_shape is not None:
        splitk_shape = fast_gemv_splitk_scratch_shape(x, weight)
        if splitk_shape is not None:
            splitk_scratch = _get_reusable_out_decode_typed(
                owner,
                f"{out_attr_name}_splitk_scratch",
                splitk_shape,
                x,
                torch.float32,
            )

    # Warmup
    fast_linear(x, weight, bias, out=out)
    if splitk_scratch is not None:
        try:
            fast_linear(
                x,
                weight,
                bias,
                out=out,
                mode_override="splitk",
                scratch=splitk_scratch,
            )
        except Exception:
            splitk_scratch = None
    torch.nn.functional.linear(x, weight, bias)
    torch.cuda.synchronize()

    try:
        fast_ms = _cuda_bench_ms(lambda: fast_linear(x, weight, bias, out=out), iters=8)
        best_mode = ""
        if splitk_scratch is not None:
            try:
                splitk_ms = _cuda_bench_ms(
                    lambda: fast_linear(
                        x,
                        weight,
                        bias,
                        out=out,
                        mode_override="splitk",
                        scratch=splitk_scratch,
                    ),
                    iters=8,
                )
                if splitk_ms < fast_ms:
                    fast_ms = splitk_ms
                    best_mode = "splitk"
            except Exception:
                splitk_scratch = None
        torch_ms = _cuda_bench_ms(lambda: torch.nn.functional.linear(x, weight, bias), iters=8)
        if op_name == "gate_up":
            # Gate-up often wins end-to-end even when microbench is near-tie/slightly worse.
            # Use a tolerant threshold to avoid flapping due small timing noise.
            use_fast = fast_ms <= (torch_ms * _GATE_UP_FAST_TOLERANCE)
        elif op_name in {"linear_attn_in", "linear_attn_out"}:
            # On T4, isolated GEMV microbench can prefer the Triton path for
            # linear-attn projections while the full decode loop regresses due
            # to launch pressure. Require a much larger local win before opting in.
            use_fast = fast_ms <= (torch_ms * _LINEAR_ATTN_FAST_GEMV_TOLERANCE)
        else:
            use_fast = fast_ms <= (torch_ms * 0.98)
        if op_name == "gate_up" and _FORCE_GATE_UP_FAST:
            use_fast = True
        _DECODE_LINEAR_MODE_CACHE[key] = best_mode if use_fast else ""
    except Exception:
        use_fast = True
        _DECODE_LINEAR_MODE_CACHE[key] = ""

    _DECODE_LINEAR_BACKEND_CACHE[key] = bool(use_fast)
    if _DECODE_LINEAR_PICK_DEBUG:
        try:
            print(
                f"[decode-linear-pick] op={op_name} in={int(weight.shape[1])} out={out_features} "
                f"fast_ms={float(fast_ms):.4f} torch_ms={float(torch_ms):.4f} "
                f"mode={_DECODE_LINEAR_MODE_CACHE.get(key, '') or 'tile'} use_fast={bool(use_fast)}"
            )
        except Exception:
            print(
                f"[decode-linear-pick] op={op_name} in={int(weight.shape[1])} out={out_features} "
                f"use_fast={bool(use_fast)}"
            )
    return bool(use_fast)

__all__ = ['LlamaConfig', 'MegaGemmLlama']

# Supported model types
SUPPORTED_MODELS = {'llama', 'mistral', 'qwen2', 'qwen3', 'qwen3_moe', 'qwen3_5_text', 'gemma2', 'gemma4_text'}


@dataclass
class LlamaConfig:
    """
    Unified model configuration.

    Core fields work for LLaMA/Mistral. Additional flags enable
    Qwen and Gemma specific features. The loader auto-detects
    model_type from HuggingFace config.json.
    """
    # --- Core (all models) ---
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 22
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 64
    vocab_size: int = 32000
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False

    # --- Model type ---
    model_type: str = 'llama'

    # --- Qwen 2.5: QKV bias ---
    attention_bias: bool = False

    # --- Qwen 3: QK-Norm ---
    qk_norm: bool = False
    qk_norm_offset: bool = False

    # --- Qwen 3.5: query output gate + partial rotary ---
    attention_output_gate: bool = False
    partial_rotary_factor: float = 1.0
    rotary_dim: int = 0
    layer_types: List[str] = field(default_factory=list)
    linear_conv_kernel_dim: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0

    # --- Qwen 3 MoE: sparse FFN routing ---
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    decoder_sparse_step: int = 1
    mlp_only_layers: List[int] = field(default_factory=list)
    norm_topk_prob: bool = True
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.0

    # --- Gemma 4 text: hybrid sliding/full attention + PLE + KV sharing ---
    global_head_dim: int = 0
    num_global_key_value_heads: int = 0
    sliding_window: int = 0
    num_kv_shared_layers: int = 0
    first_kv_shared_layer_idx: int = 0
    use_double_wide_mlp: bool = False
    enable_moe_block: bool = False
    attention_k_eq_v: bool = False
    hidden_size_per_layer_input: int = 0
    vocab_size_per_layer_input: int = 0
    per_layer_head_dims: List[int] = field(default_factory=list)
    per_layer_num_kv_heads: List[int] = field(default_factory=list)
    per_layer_rotary_dims: List[int] = field(default_factory=list)
    per_layer_rope_thetas: List[float] = field(default_factory=list)
    gemma4_full_rope_partial_factor: float = 1.0
    gemma4_full_rope_factor: float = 1.0
    kv_share_sources: List[Optional[int]] = field(default_factory=list)
    kv_cache_layer_indices: List[int] = field(default_factory=list)
    mlp_intermediate_sizes: List[int] = field(default_factory=list)

    # --- Gemma 2: RMSNorm +1 offset ---
    norm_offset: bool = False

    # --- Gemma 2: embedding normalization (multiply by √hidden_size) ---
    embed_scale: float = 1.0

    # --- Gemma 2: logit soft capping ---
    attn_logit_softcapping: float = 0.0   # 0 = disabled, Gemma 2: 50.0
    final_logit_softcapping: float = 0.0  # 0 = disabled, Gemma 2: 30.0

    # --- Gemma 2: activation function ---
    hidden_act: str = 'silu'  # 'silu' (LLaMA/Qwen) or 'gelu' (Gemma 2)

    # --- RoPE convention ---
    # False = interleaved (Meta LLaMA: pair dims 0,1 then 2,3...)
    # True  = half-rotate (HF-native: pair dims 0,dim/2 then 1,dim/2+1...)
    rope_half_rotate: bool = False

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Return True when a layer should use routed MoE experts."""
        if self.model_type == 'gemma4_text':
            return (
                bool(self.enable_moe_block)
                and self.num_experts > 0
                and self.num_experts_per_tok > 0
            )
        if self.model_type != 'qwen3_moe':
            return False
        if self.num_experts <= 0 or self.num_experts_per_tok <= 0:
            return False
        if int(layer_idx) in set(self.mlp_only_layers or []):
            return False
        step = max(1, int(self.decoder_sparse_step or 1))
        return (int(layer_idx) + 1) % step == 0

    @classmethod
    def from_dict(cls, d: dict) -> 'LlamaConfig':
        """Create config from HuggingFace config dict. Auto-detects model type."""
        outer_model_type = d.get('model_type', 'llama')
        if outer_model_type in ('qwen3_5', 'gemma4'):
            arch = d.get('text_config', d)
        else:
            arch = d
        rope_cfg = arch.get('rope_parameters', {})

        model_type = arch.get('model_type', outer_model_type)
        hd = arch.get('head_dim', arch['hidden_size'] // arch['num_attention_heads'])

        config = cls(
            hidden_size=arch['hidden_size'],
            intermediate_size=arch['intermediate_size'],
            num_hidden_layers=arch['num_hidden_layers'],
            num_attention_heads=arch['num_attention_heads'],
            num_key_value_heads=arch.get('num_key_value_heads', arch['num_attention_heads']),
            head_dim=hd,
            vocab_size=arch['vocab_size'],
            max_position_embeddings=arch.get('max_position_embeddings', 2048),
            rms_norm_eps=arch.get('rms_norm_eps', 1e-5),
            rope_theta=arch.get('rope_theta', rope_cfg.get('rope_theta', 10000.0)),
            tie_word_embeddings=arch.get('tie_word_embeddings', False),
            model_type=model_type,
            hidden_act=arch.get('hidden_act', arch.get('hidden_activation', 'silu')),
        )

        layer_types = arch.get('layer_types')
        if layer_types is None and model_type == 'qwen3_5_text':
            interval = int(arch.get('full_attention_interval', 4))
            layer_types = [
                'full_attention' if layer_idx % interval == 0 else 'linear_attention'
                for layer_idx in range(config.num_hidden_layers)
            ]
        if layer_types is None and model_type == 'gemma4_text':
            sliding_window_pattern = 6
            layer_types = [
                'sliding_attention' if (layer_idx + 1) % sliding_window_pattern else 'full_attention'
                for layer_idx in range(config.num_hidden_layers)
            ]
            if layer_types and layer_types[-1] != 'full_attention':
                layer_types[-1] = 'full_attention'
        config.layer_types = list(layer_types or [])

        # Auto-configure based on model type
        # NOTE: ALL HF models use half-rotate RoPE (rotate_half style).
        # HF conversion from Meta rearranges Q/K weights accordingly.
        if model_type in ('llama', 'mistral'):
            config.rope_half_rotate = True  # HF weights need half-rotate

        elif model_type == 'qwen2':
            config.attention_bias = arch.get('attention_bias', True)
            config.rope_half_rotate = True

        elif model_type == 'qwen3':
            config.qk_norm = True
            config.attention_bias = arch.get('attention_bias', False)
            config.rope_half_rotate = True

        elif model_type == 'qwen3_moe':
            config.qk_norm = True
            config.attention_bias = arch.get('attention_bias', False)
            config.num_experts = int(arch.get('num_experts', 0) or 0)
            config.num_experts_per_tok = int(arch.get('num_experts_per_tok', 0) or 0)
            config.moe_intermediate_size = int(
                arch.get('moe_intermediate_size', arch.get('intermediate_size', config.intermediate_size))
            )
            config.decoder_sparse_step = int(arch.get('decoder_sparse_step', 1) or 1)
            config.mlp_only_layers = [int(x) for x in (arch.get('mlp_only_layers') or [])]
            config.norm_topk_prob = bool(arch.get('norm_topk_prob', True))
            config.output_router_logits = bool(arch.get('output_router_logits', False))
            config.router_aux_loss_coef = float(arch.get('router_aux_loss_coef', 0.0) or 0.0)
            config.rope_half_rotate = True

        elif model_type == 'qwen3_5_text':
            config.qk_norm = True
            config.qk_norm_offset = True
            config.norm_offset = True
            config.attention_bias = arch.get('attention_bias', False)
            config.attention_output_gate = arch.get(
                'attn_output_gate',
                arch.get('attention_output_gate', True),
            )
            config.partial_rotary_factor = float(
                arch.get('partial_rotary_factor', rope_cfg.get('partial_rotary_factor', 1.0))
            )
            config.linear_conv_kernel_dim = int(arch.get('linear_conv_kernel_dim', 4))
            config.linear_key_head_dim = int(arch.get('linear_key_head_dim', hd))
            config.linear_value_head_dim = int(arch.get('linear_value_head_dim', hd))
            config.linear_num_key_heads = int(arch.get('linear_num_key_heads', config.num_attention_heads))
            config.linear_num_value_heads = int(arch.get('linear_num_value_heads', config.num_attention_heads))
            config.rope_half_rotate = True

        elif model_type == 'gemma2':
            config.norm_offset = True
            config.embed_scale = math.sqrt(arch['hidden_size'])
            config.attn_logit_softcapping = arch.get('attn_logit_softcapping', 50.0)
            config.final_logit_softcapping = arch.get('final_logit_softcapping', 30.0)
            config.hidden_act = arch.get('hidden_act', 'gelu')
            config.rope_half_rotate = True

        elif model_type == 'gemma4_text':
            config.norm_offset = False
            config.embed_scale = math.sqrt(arch['hidden_size'])
            config.attention_bias = arch.get('attention_bias', False)
            config.qk_norm = True
            config.qk_norm_offset = False
            config.hidden_act = arch.get('hidden_activation', arch.get('hidden_act', 'gelu_pytorch_tanh'))
            config.final_logit_softcapping = arch.get('final_logit_softcapping', 0.0) or 0.0
            config.rope_half_rotate = True
            config.global_head_dim = int(arch.get('global_head_dim', hd))
            config.num_global_key_value_heads = int(
                arch.get('num_global_key_value_heads') or arch.get('num_key_value_heads', config.num_key_value_heads)
            )
            config.sliding_window = int(arch.get('sliding_window') or 0)
            config.num_kv_shared_layers = int(arch.get('num_kv_shared_layers') or 0)
            config.first_kv_shared_layer_idx = max(
                0,
                config.num_hidden_layers - config.num_kv_shared_layers,
            )
            config.use_double_wide_mlp = bool(arch.get('use_double_wide_mlp', False))
            config.enable_moe_block = bool(arch.get('enable_moe_block', False))
            if config.enable_moe_block:
                config.num_experts = int(arch.get('num_experts', 0) or 0)
                config.num_experts_per_tok = int(
                    arch.get(
                        'top_k_experts',
                        arch.get(
                            'num_experts_per_tok',
                            arch.get(
                                'num_experts_per_token',
                                arch.get('num_selected_experts', arch.get('moe_top_k', 8)),
                            ),
                        ),
                    )
                    or 0
                )
                config.moe_intermediate_size = int(
                    arch.get(
                        'moe_intermediate_size',
                        arch.get('expert_intermediate_size', config.intermediate_size),
                    )
                    or 0
                )
                config.norm_topk_prob = bool(arch.get('norm_topk_prob', True))
            config.attention_k_eq_v = bool(arch.get('attention_k_eq_v', False))
            config.hidden_size_per_layer_input = int(arch.get('hidden_size_per_layer_input') or 0)
            config.vocab_size_per_layer_input = int(
                arch.get('vocab_size_per_layer_input') or arch.get('vocab_size', config.vocab_size)
            )

            rope_by_type = rope_cfg if isinstance(rope_cfg, dict) else {}
            sliding_rope = rope_by_type.get('sliding_attention', {}) if isinstance(rope_by_type.get('sliding_attention', {}), dict) else {}
            full_rope = rope_by_type.get('full_attention', {}) if isinstance(rope_by_type.get('full_attention', {}), dict) else {}
            sliding_rotary_dim = int(config.head_dim)
            full_partial = float(full_rope.get('partial_rotary_factor', 0.25))
            config.gemma4_full_rope_partial_factor = full_partial
            config.gemma4_full_rope_factor = float(full_rope.get('factor') or 1.0)
            # Gemma 4's proportional RoPE returns full-head cos/sin tables:
            # inactive channels are encoded as zero-frequency pairs rather than
            # being excluded from the rotary transform.
            full_rotary_dim = int(config.global_head_dim)
            full_rotary_dim = max(2, full_rotary_dim)
            if full_rotary_dim % 2 != 0:
                full_rotary_dim -= 1
            config.per_layer_head_dims = []
            config.per_layer_num_kv_heads = []
            config.per_layer_rotary_dims = []
            config.per_layer_rope_thetas = []
            for layer_idx in range(config.num_hidden_layers):
                layer_type = (
                    config.layer_types[layer_idx]
                    if config.layer_types and layer_idx < len(config.layer_types)
                    else 'full_attention'
                )
                if layer_type == 'full_attention':
                    config.per_layer_head_dims.append(config.global_head_dim)
                    if config.attention_k_eq_v:
                        config.per_layer_num_kv_heads.append(config.num_global_key_value_heads)
                    else:
                        config.per_layer_num_kv_heads.append(config.num_key_value_heads)
                    config.per_layer_rotary_dims.append(full_rotary_dim)
                    config.per_layer_rope_thetas.append(float(full_rope.get('rope_theta', 1000000.0)))
                else:
                    config.per_layer_head_dims.append(config.head_dim)
                    config.per_layer_num_kv_heads.append(config.num_key_value_heads)
                    config.per_layer_rotary_dims.append(sliding_rotary_dim)
                    config.per_layer_rope_thetas.append(float(sliding_rope.get('rope_theta', 10000.0)))

            last_unshared_by_type: Dict[str, int] = {}
            config.kv_share_sources = []
            config.kv_cache_layer_indices = []
            config.mlp_intermediate_sizes = []
            for layer_idx in range(config.num_hidden_layers):
                layer_type = (
                    config.layer_types[layer_idx]
                    if config.layer_types and layer_idx < len(config.layer_types)
                    else 'full_attention'
                )
                is_shared = (
                    config.first_kv_shared_layer_idx > 0
                    and layer_idx >= config.first_kv_shared_layer_idx
                )
                if is_shared:
                    source_idx = last_unshared_by_type.get(layer_type)
                    if source_idx is None:
                        raise ValueError(
                            f"Gemma4 KV-shared layer {layer_idx} has no earlier "
                            f"unshared {layer_type} source layer."
                        )
                    config.kv_share_sources.append(source_idx)
                else:
                    config.kv_share_sources.append(None)
                    config.kv_cache_layer_indices.append(layer_idx)
                    last_unshared_by_type[layer_type] = layer_idx
                mlp_mult = 2 if (config.use_double_wide_mlp and is_shared) else 1
                config.mlp_intermediate_sizes.append(config.intermediate_size * mlp_mult)

        rotary_dim = config.head_dim
        if config.partial_rotary_factor < 1.0:
            rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        rotary_dim = max(2, min(config.head_dim, rotary_dim))
        if rotary_dim % 2 != 0:
            rotary_dim -= 1
        config.rotary_dim = rotary_dim
        if not config.layer_types:
            config.layer_types = ['full_attention'] * config.num_hidden_layers
        if not config.per_layer_head_dims:
            config.per_layer_head_dims = [config.head_dim] * config.num_hidden_layers
        if not config.per_layer_num_kv_heads:
            config.per_layer_num_kv_heads = [config.num_key_value_heads] * config.num_hidden_layers
        if not config.per_layer_rotary_dims:
            config.per_layer_rotary_dims = [config.rotary_dim] * config.num_hidden_layers
        if not config.per_layer_rope_thetas:
            config.per_layer_rope_thetas = [config.rope_theta] * config.num_hidden_layers
        if not config.kv_share_sources:
            config.kv_share_sources = [None] * config.num_hidden_layers
        if not config.kv_cache_layer_indices:
            config.kv_cache_layer_indices = list(range(config.num_hidden_layers))
        if not config.mlp_intermediate_sizes:
            config.mlp_intermediate_sizes = [config.intermediate_size] * config.num_hidden_layers

        return config


def _precompute_layer_rope_cache(
    config: LlamaConfig,
    layer_idx: int,
    max_seq_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute the RoPE table matching one decoder layer's attention type."""
    cache_len = int(max_seq_len or config.max_position_embeddings)
    rotary_dim = int(config.per_layer_rotary_dims[layer_idx])
    rope_theta = float(config.per_layer_rope_thetas[layer_idx])
    layer_type = (
        config.layer_types[layer_idx]
        if config.layer_types and layer_idx < len(config.layer_types)
        else 'full_attention'
    )
    if config.model_type == 'gemma4_text' and layer_type == 'full_attention':
        return precompute_proportional_freqs_cis(
            rotary_dim,
            cache_len,
            config.gemma4_full_rope_partial_factor,
            rope_theta,
            factor=config.gemma4_full_rope_factor,
        )
    return precompute_freqs_cis(
        rotary_dim,
        cache_len,
        rope_theta,
    )


def _initial_rope_cache_len(config: LlamaConfig) -> int:
    """Pick an initial RoPE cache length without materializing huge long-context tables."""
    if config.model_type != 'gemma4_text':
        return int(config.max_position_embeddings)
    raw = os.environ.get("MEGAGEMM_ROPE_CACHE_LEN", "4096")
    try:
        requested = int(raw)
    except ValueError:
        requested = 4096
    return max(1, min(int(config.max_position_embeddings), requested))


class MGRMSNorm(nn.Module):
    """
    RMSNorm with CUDA kernel or PyTorch fallback.

    Supports Gemma-style offset: output = x * (1 + weight) instead of x * weight.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        offset: bool = False,
        with_scale: bool = True,
    ):
        super().__init__()
        self.with_scale = with_scale
        self.prefer_triton = False
        if with_scale:
            init = torch.zeros(hidden_size) if offset else torch.ones(hidden_size)
            self.weight = nn.Parameter(init)
        self.eps = eps
        self.hidden_size = hidden_size
        self.offset = offset  # Gemma 2: weight + 1
        self._triton_no_weight_hits = 0
        self._force_pytorch_no_weight = False

    def _pytorch_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.with_scale:
            w = (self.weight + 1.0) if self.offset else self.weight
            x = x * w
        return x.to(orig_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Priority: CUDA kernel > Triton kernel > PyTorch
        if not self.with_scale:
            if (
                not self._force_pytorch_no_weight
                and _HAS_TRITON_RMSNORM
                and x.is_cuda
            ):
                try:
                    out = rmsnorm_triton_no_weight(x, self.eps)
                    self._triton_no_weight_hits += 1
                    return out
                except Exception:
                    pass
            return self._pytorch_forward(x)
        if not self.prefer_triton and _can_use_cuda_rmsnorm_for(x, self.offset):
            try:
                if torch.is_grad_enabled():
                    return RMSNormFunction.apply(x, self.weight, self.eps)
                return rmsnorm_forward(x, self.weight, self.eps)
            except Exception:
                pass
        # Triton kernel: works on NVIDIA (sm_75+) + AMD (ROCm)
        if _HAS_TRITON_RMSNORM and x.is_cuda:
            try:
                return rmsnorm_triton(x, self.weight, self.eps, self.offset)
            except Exception:
                pass
        # PyTorch: always works (CPU, any GPU)
        return self._pytorch_forward(x)


class RMSNormGated(nn.Module):
    """RMSNorm followed by SiLU gate, used by Qwen 3.5 linear attention."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def _pytorch_forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.weight * torch.nn.functional.silu(gate)).to(orig_dtype)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if _HAS_RMSNORM_GATED and x.is_cuda and gate.is_cuda:
            try:
                return rmsnorm_gated(x, gate, self.weight, self.eps)
            except Exception:
                pass
        return self._pytorch_forward(x, gate)


def apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Zero pad positions so linear recurrent states do not leak across left padding."""
    if attention_mask is None:
        return hidden_states
    mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
    return hidden_states * mask.unsqueeze(-1)


_SUFFIX_PREFILL_MASK_CACHE = {}
_CHUNK_RULE_CACHE = {}
_QWEN35_RUNTIME_POLICY_CACHE = {}
_TRITON_CAUSAL_CONV1D_DISABLED = False
_TRITON_LINEAR_GATES_DISABLED = False


_SLIDING_CAUSAL_MASK_CACHE = {}


def _get_sliding_causal_attn_mask(
    q_len: int,
    k_len: int,
    sliding_window: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive local causal mask for Gemma4 sliding attention."""
    if sliding_window <= 0:
        raise ValueError("sliding_window must be positive")
    key = (
        int(q_len),
        int(k_len),
        int(sliding_window),
        device.type,
        int(device.index) if device.index is not None else -1,
        str(dtype),
    )
    cached = _SLIDING_CAUSAL_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    q_pos = torch.arange(k_len - q_len, k_len, device=device)
    k_pos = torch.arange(k_len, device=device)
    allowed = (k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)) & (
        k_pos.unsqueeze(0) >= (q_pos.unsqueeze(1) - sliding_window + 1)
    )
    mask = torch.zeros(q_len, k_len, device=device, dtype=dtype)
    mask.masked_fill_(~allowed, float('-inf'))
    mask = mask.unsqueeze(0).unsqueeze(0)
    _SLIDING_CAUSAL_MASK_CACHE[key] = mask
    return mask


def _get_suffix_prefill_attn_mask(
    prefix_len: int,
    suffix_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive mask for suffix tokens attending over prefix+suffix KV."""
    device_index = -1 if device.index is None else int(device.index)
    key = (int(prefix_len), int(suffix_len), device.type, device_index, dtype)
    cached = _SUFFIX_PREFILL_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    allow_prefix = torch.ones(suffix_len, prefix_len, dtype=torch.bool, device=device)
    allow_suffix = torch.tril(torch.ones(suffix_len, suffix_len, dtype=torch.bool, device=device))
    allowed = torch.cat([allow_prefix, allow_suffix], dim=1)

    mask = torch.zeros(
        1,
        1,
        suffix_len,
        prefix_len + suffix_len,
        device=device,
        dtype=dtype,
    )
    mask.masked_fill_(~allowed.unsqueeze(0).unsqueeze(0), float("-inf"))
    _SUFFIX_PREFILL_MASK_CACHE[key] = mask
    return mask


@runtime_checkable
class BlockManagerLike(Protocol):
    """Minimal BlockManager interface consumed by MegaGemmLlama."""

    def get_linear_state(self, seq_id: int, layer_idx: int, device=None): ...
    def set_linear_state(self, seq_id: int, layer_idx: int, conv_state=None, recurrent_state=None): ...
    def get_linear_state_batch(self, seq_ids: Sequence[int], layer_idx: int, device=None): ...
    def set_linear_state_batch(self, seq_ids: Sequence[int], layer_idx: int, conv_states=None, recurrent_states=None): ...
    def get_block_table_tensor(self, seq_ids: Sequence[int]): ...
    def get_seq_lens_tensor(self, seq_ids: Sequence[int]): ...
    def get_kv_cache(self, layer_idx: int) -> Optional[torch.Tensor]: ...
    def write_kv(self, seq_id: int, layer_idx: int, k: torch.Tensor, v: torch.Tensor): ...
    def write_kv_prefill_packed(
        self,
        seq_ids: Sequence[int],
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        kv_mapping: Optional[Any] = None,
        tokens_per_program: int = 1,
    ): ...
    def compute_kv_mapping(
        self,
        seq_ids: Sequence[int],
        cu_seqlens: torch.Tensor,
        device,
        seq_lengths: Optional[Sequence[int]] = None,
    ): ...
    def advance_seq_len_batch(self, seq_ids: Sequence[int], num_tokens: int = 1): ...
    def advance_seq_len(self, seq_id: int, num_tokens: int = 1): ...


def _get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _get_env_int_optional(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return int(raw)


def _get_env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _get_env_bool_optional(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip().lower()
    if not raw:
        return None
    return raw not in {"0", "false", "no", "off"}


def _cpu_max_seq_len_from_block_manager(
    block_manager: Optional[BlockManagerLike],
    seq_ids: Sequence[int],
) -> Optional[int]:
    seq_lens = getattr(block_manager, "seq_lens", None)
    if not isinstance(seq_lens, dict) or not seq_ids:
        return None
    try:
        return max(int(seq_lens[int(sid)]) for sid in seq_ids)
    except (KeyError, TypeError, ValueError):
        return None


def _decode_blocks_for_seq_len(
    seq_len: Optional[int],
    block_size: int,
    extra_tokens: int = 1,
) -> Optional[int]:
    if seq_len is None:
        return None
    return max(1, (int(seq_len) + int(extra_tokens) + int(block_size) - 1) // int(block_size))


def _resolve_qwen35_runtime_policy(device: torch.device) -> Tuple[int, bool]:
    """Auto-select linear-attention runtime defaults for this device."""
    device_type = device.type
    device_index = -1 if device.index is None else int(device.index)
    key = (device_type, device_index)
    cached = _QWEN35_RUNTIME_POLICY_CACHE.get(key)
    if cached is not None:
        return cached

    # Stable fallback defaults (CPU / no Triton kernels).
    short_prefill_threshold = 160
    enable_chunk_scan = False

    if (
        device_type == "cuda"
        and torch.cuda.is_available()
        and HAS_TRITON_LINEAR_ATTN
        and chunk_interchunk_scan is not None
    ):
        enable_chunk_scan = True
        major, minor = torch.cuda.get_device_capability(device)
        sm = major * 10 + minor
        # Conservative thresholds by GPU generation:
        # keep T4 at current best-known default, allow larger short-prefill on newer GPUs.
        if sm >= 90:
            short_prefill_threshold = 224
        elif sm >= 80:
            short_prefill_threshold = 192
        elif sm >= 75:
            short_prefill_threshold = 160
        else:
            short_prefill_threshold = 128

    cached = (short_prefill_threshold, enable_chunk_scan)
    _QWEN35_RUNTIME_POLICY_CACHE[key] = cached
    return cached


def _get_chunk_rule_constants(chunk_size: int, device: torch.device):
    device_type = device.type
    device_index = -1 if device.index is None else int(device.index)
    key = (chunk_size, device_type, device_index)
    cached = _CHUNK_RULE_CACHE.get(key)
    if cached is not None:
        return cached

    lower_with_diag_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device),
        diagonal=0,
    )
    upper_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device),
        diagonal=1,
    )
    eye = torch.eye(chunk_size, dtype=torch.float32, device=device)
    cached = (lower_with_diag_mask, upper_mask, eye)
    _CHUNK_RULE_CACHE[key] = cached
    return cached


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    denom = torch.linalg.norm(x.float(), dim=dim, keepdim=True).clamp_min(eps)
    return (x.float() / denom).to(x.dtype)


def _can_use_causal_conv1d(x: torch.Tensor, conv1d_weight: torch.Tensor) -> bool:
    return (
        _HAS_CAUSAL_CONV1D
        and x.is_cuda
        and conv1d_weight.is_cuda
        and conv1d_weight.ndim == 2
        and conv1d_weight.shape[-1] in {2, 3, 4}
    )


def _can_use_triton_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    conv1d_weight: torch.Tensor,
) -> bool:
    return (
        _HAS_TRITON
        and hidden_states.is_cuda
        and conv_state.is_cuda
        and conv1d_weight.is_cuda
        and hidden_states.ndim == 3
        and conv_state.ndim == 3
        and conv1d_weight.ndim == 2
        and hidden_states.shape[-1] == 1
        and conv_state.shape[-1] <= 8
        and conv_state.shape[-1] == conv1d_weight.shape[-1]
        and not _get_env_bool("MEGAGEMM_QWEN35_DISABLE_TRITON_CONV_UPDATE", False)
    )


if _HAS_TRITON:
    @triton.jit
    def _causal_conv1d_update_kernel(
        hidden_ptr,
        state_ptr,
        weight_ptr,
        out_ptr,
        stride_h_b, stride_h_c,
        stride_s_b, stride_s_c, stride_s_k,
        stride_w_c, stride_w_k,
        stride_o_b, stride_o_c,
        channels,
        state_len,
        BLOCK_C: tl.constexpr,
        MAX_K: tl.constexpr,
    ):
        bid = tl.program_id(0)
        pid_c = tl.program_id(1)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < channels

        hidden = tl.load(
            hidden_ptr + bid * stride_h_b + offs_c * stride_h_c,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)

        acc = tl.zeros([BLOCK_C], dtype=tl.float32)
        for k in tl.static_range(0, MAX_K):
            valid_k = k < state_len
            if k + 1 < MAX_K:
                shifted = tl.load(
                    state_ptr + bid * stride_s_b + offs_c * stride_s_c + (k + 1) * stride_s_k,
                    mask=mask_c & ((k + 1) < state_len),
                    other=0.0,
                ).to(tl.float32)
                shifted = tl.where((k + 1) < state_len, shifted, hidden)
            else:
                shifted = hidden
            new_state = tl.where(valid_k, shifted, 0.0)
            weight = tl.load(
                weight_ptr + offs_c * stride_w_c + k * stride_w_k,
                mask=mask_c & valid_k,
                other=0.0,
            ).to(tl.float32)
            tl.store(
                state_ptr + bid * stride_s_b + offs_c * stride_s_c + k * stride_s_k,
                new_state,
                mask=mask_c & valid_k,
            )
            acc += new_state * weight

        acc = acc * tl.sigmoid(acc)
        tl.store(
            out_ptr + bid * stride_o_b + offs_c * stride_o_c,
            acc,
            mask=mask_c,
        )


def _triton_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    conv1d_weight: torch.Tensor,
    out: Optional[torch.Tensor] = None,
):
    global _TRITON_CAUSAL_CONV1D_DISABLED
    if _TRITON_CAUSAL_CONV1D_DISABLED:
        return None
    if not _can_use_triton_causal_conv1d_update(hidden_states, conv_state, conv1d_weight):
        return None
    if not conv_state.is_contiguous():
        conv_state = conv_state.contiguous()

    batch, channels, _ = hidden_states.shape
    out_shape = (batch, channels, 1)
    if out is None:
        out = torch.empty(out_shape, device=hidden_states.device, dtype=hidden_states.dtype)
    elif tuple(out.shape) != out_shape or out.device != hidden_states.device or out.dtype != hidden_states.dtype:
        out = torch.empty(out_shape, device=hidden_states.device, dtype=hidden_states.dtype)
    grid = (batch, triton.cdiv(channels, 128))
    try:
        _causal_conv1d_update_kernel[grid](
            hidden_states,
            conv_state,
            conv1d_weight,
            out,
            hidden_states.stride(0), hidden_states.stride(1),
            conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
            conv1d_weight.stride(0), conv1d_weight.stride(1),
            out.stride(0), out.stride(1),
            channels,
            conv_state.shape[-1],
            BLOCK_C=128,
            MAX_K=8,
            num_warps=4,
        )
    except Exception:
        _TRITON_CAUSAL_CONV1D_DISABLED = True
        return None
    return out, conv_state


def _causal_conv1d_silu(
    hidden_states: torch.Tensor,
    conv1d_weight: torch.Tensor,
) -> torch.Tensor:
    if _can_use_causal_conv1d(hidden_states, conv1d_weight):
        return causal_conv1d_fn(
            hidden_states.to(conv1d_weight.dtype),
            conv1d_weight,
            activation="silu",
        ).to(hidden_states.dtype)
    out = torch.nn.functional.conv1d(
        hidden_states.to(conv1d_weight.dtype),
        conv1d_weight.unsqueeze(1),
        padding=0,
        groups=hidden_states.shape[1],
    )
    return torch.nn.functional.silu(out).to(hidden_states.dtype)


if _HAS_TRITON:
    @triton.jit
    def _fused_linear_gates_kernel(
        a_ptr,
        b_ptr,
        a_log_ptr,
        dt_bias_ptr,
        beta_ptr,
        gk_ptr,
        stride_a_b,
        stride_a_s,
        stride_a_h,
        stride_b_b,
        stride_b_s,
        stride_b_h,
        stride_beta_b,
        stride_beta_s,
        stride_beta_h,
        stride_gk_b,
        stride_gk_s,
        stride_gk_h,
        seq_len,
        num_heads,
        total_elems,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total_elems

        head = offs % num_heads
        seq = (offs // num_heads) % seq_len
        batch = offs // (num_heads * seq_len)

        a = tl.load(
            a_ptr + batch * stride_a_b + seq * stride_a_s + head * stride_a_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            b_ptr + batch * stride_b_b + seq * stride_b_s + head * stride_b_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        a_log = tl.load(a_log_ptr + head, mask=mask, other=0.0).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + head, mask=mask, other=0.0).to(tl.float32)

        beta = 1.0 / (1.0 + tl.exp(-b))
        x = a + dt_bias
        softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        gk = -tl.exp(a_log) * softplus_x

        tl.store(
            beta_ptr + batch * stride_beta_b + seq * stride_beta_s + head * stride_beta_h,
            beta,
            mask=mask,
        )
        tl.store(
            gk_ptr + batch * stride_gk_b + seq * stride_gk_s + head * stride_gk_h,
            gk,
            mask=mask,
        )


def _fused_linear_gates(
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    global _TRITON_LINEAR_GATES_DISABLED

    if (
        _TRITON_LINEAR_GATES_DISABLED
        or not _HAS_TRITON
        or not a.is_cuda
        or not b.is_cuda
        or a.ndim != 3
        or b.ndim != 3
        or a.shape != b.shape
    ):
        beta = b.float().sigmoid()
        gk = -a_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias.float())
        return beta, gk

    beta = torch.empty_like(b, dtype=torch.float32)
    gk = torch.empty_like(a, dtype=torch.float32)
    total_elems = a.numel()
    grid = (triton.cdiv(total_elems, 256),)
    try:
        _fused_linear_gates_kernel[grid](
            a,
            b,
            a_log,
            dt_bias,
            beta,
            gk,
            a.stride(0), a.stride(1), a.stride(2),
            b.stride(0), b.stride(1), b.stride(2),
            beta.stride(0), beta.stride(1), beta.stride(2),
            gk.stride(0), gk.stride(1), gk.stride(2),
            a.shape[1],
            a.shape[2],
            total_elems,
            BLOCK=256,
            num_warps=4,
        )
        return beta, gk
    except Exception:
        _TRITON_LINEAR_GATES_DISABLED = True
        beta = b.float().sigmoid()
        gk = -a_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias.float())
        return beta, gk


def torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    conv1d_weight: torch.Tensor,
    out: Optional[torch.Tensor] = None,
):
    """
    Single-step causal conv update.

    hidden_states: [batch, conv_dim, seq_len]
    conv_state: [batch, conv_dim, kernel_size]
    conv1d_weight: [conv_dim, kernel_size]
    """
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    if seq_len == 1:
        triton_out = _triton_causal_conv1d_update(hidden_states, conv_state, conv1d_weight, out=out)
        if triton_out is not None:
            return triton_out
        if not conv_state.is_contiguous():
            conv_state = conv_state.contiguous()
        hidden_states_new = torch.cat([conv_state, hidden_states.to(conv_state.dtype)], dim=-1)
        conv_state.copy_(hidden_states_new[:, :, -state_len:])
        out = _causal_conv1d_silu(hidden_states_new, conv1d_weight)[:, :, -1:]
        return out.to(hidden_states.dtype), conv_state

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(conv1d_weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = _causal_conv1d_silu(hidden_states_new.to(hidden_states.dtype), conv1d_weight)[:, :, -seq_len:]
    return out.to(hidden_states.dtype), conv_state


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    enable_chunk_scan: bool = False,
):
    """Reference PyTorch fallback aligned with HF Qwen3Next/Qwen3.5 chunked delta rule."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, gate = [
        x.transpose(1, 2).to(torch.float32)
        for x in (query, key, value, beta, gate)
    ]

    batch_size, num_heads, sequence_length, key_dim = key.shape
    value_dim = value.shape[-1]
    chunk_size = min(chunk_size, max(sequence_length, 1))
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, key_dim, value_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value)
    )

    # Process full chunks with chunk kernels, then process the tail tokens
    # recurrently. This avoids padding up to the next chunk boundary.
    full_tokens = (sequence_length // chunk_size) * chunk_size
    tail_tokens = sequence_length - full_tokens

    if full_tokens > 0:
        query_main = query[:, :, :full_tokens]
        key_main = key[:, :, :full_tokens]
        value_main = value[:, :, :full_tokens]
        beta_main = beta[:, :, :full_tokens]
        gate_main = gate[:, :, :full_tokens]

        value_beta = value_main * beta_main.unsqueeze(-1)
        key_beta = key_main * beta_main.unsqueeze(-1)

        query_main, key_main, value_main, key_beta, value_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
            for x in (query_main, key_main, value_main, key_beta, value_beta)
        ]
        gate_main = gate_main.reshape(gate_main.shape[0], gate_main.shape[1], -1, chunk_size)

        mask, upper_mask, eye = _get_chunk_rule_constants(chunk_size, query_main.device)
        gate_main = gate_main.cumsum(dim=-1)
        decay_mask = ((gate_main.unsqueeze(-1) - gate_main.unsqueeze(-2)).tril().exp()).tril()

        attn = -((key_beta @ key_main.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        if HAS_TRITON_LINEAR_ATTN and attn.is_cuda and chunk_size <= 64:
            attn = solve_chunk_local_attention(attn)
        else:
            for i in range(1, chunk_size):
                row = attn[..., i, :i].clone()
                sub = attn[..., :i, :i].clone()
                attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
            attn = attn + eye

        value_main = attn @ value_beta
        key_cumdecay = attn @ (key_beta * gate_main.exp().unsqueeze(-1))
        num_chunks = full_tokens // chunk_size

        if enable_chunk_scan and chunk_interchunk_scan is not None:
            value_new, attn_inter, last_recurrent_state = chunk_interchunk_scan(
                query_main,
                key_main,
                key_cumdecay,
                value_main,
                gate_main,
                last_recurrent_state,
            )
            attn_local = (query_main @ key_main.transpose(-1, -2) * decay_mask).masked_fill_(upper_mask, 0)
            core_main = attn_inter + attn_local @ value_new
        else:
            core_main = torch.zeros_like(value_main)
            for i in range(num_chunks):
                query_i = query_main[:, :, i]
                key_i = key_main[:, :, i]
                value_new, attn_inter, last_recurrent_state = chunk_interchunk(
                    query_i,
                    key_i,
                    key_cumdecay[:, :, i],
                    value_main[:, :, i],
                    gate_main[:, :, i],
                    last_recurrent_state,
                )
                attn_local = (query_i @ key_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(upper_mask, 0)
                core_main[:, :, i] = attn_inter + attn_local @ value_new
        core_attn_out = core_main.reshape(batch_size, num_heads, full_tokens, value_dim)
    else:
        core_attn_out = value.new_empty(batch_size, num_heads, 0, value_dim)

    if tail_tokens > 0:
        query_tail = query[:, :, full_tokens:]
        key_tail = key[:, :, full_tokens:]
        value_tail = value[:, :, full_tokens:]
        beta_tail = beta[:, :, full_tokens:]
        gate_tail = gate[:, :, full_tokens:]
        used_triton_tail = False
        if (
            _get_env_bool("MEGAGEMM_QWEN35_TAIL_TRITON_PREFILL", False)
            and
            HAS_TRITON_LINEAR_ATTN
            and recurrent_gated_delta_prefill is not None
            and query_tail.is_cuda
            and tail_tokens <= 64
        ):
            try:
                # query/key already l2-normalized and query already scaled above
                tail_out = recurrent_gated_delta_prefill(
                    query_tail,
                    key_tail,
                    value_tail,
                    gate_tail,
                    beta_tail,
                    last_recurrent_state,
                    num_kv_groups=1,
                    query_scale=1.0,
                    normalize_qk=False,
                )
                used_triton_tail = True
            except Exception:
                used_triton_tail = False

        if not used_triton_tail:
            tail_out = torch.empty(
                batch_size, num_heads, tail_tokens, value_dim, device=value.device, dtype=value.dtype,
            )
            for i in range(tail_tokens):
                query_i = query_tail[:, :, i]
                key_i = key_tail[:, :, i]
                value_i = value_tail[:, :, i]
                gate_i = gate_tail[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
                beta_i = beta_tail[:, :, i].unsqueeze(-1)

                last_recurrent_state = last_recurrent_state * gate_i
                kv_mem = (last_recurrent_state * key_i.unsqueeze(-1)).sum(dim=-2)
                delta = (value_i - kv_mem) * beta_i
                last_recurrent_state = last_recurrent_state + key_i.unsqueeze(-1) * delta.unsqueeze(-2)
                tail_out[:, :, i] = (last_recurrent_state * query_i.unsqueeze(-1)).sum(dim=-2)

        core_attn_out = torch.cat([core_attn_out, tail_out], dim=2)

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Reference PyTorch fallback aligned with HF Qwen3Next/Qwen3.5 recurrent delta rule."""
    initial_dtype = query.dtype
    can_normalize_in_triton = (
        use_qk_l2norm_in_kernel
        and HAS_TRITON_LINEAR_ATTN
        and recurrent_gated_delta_decode is not None
        and query.is_cuda
        and query.shape[1] == 1
    )
    if use_qk_l2norm_in_kernel and not can_normalize_in_triton:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, gate = [
        x.transpose(1, 2).to(torch.float32)
        for x in (query, key, value, beta, gate)
    ]

    batch_size, num_heads, sequence_length, key_dim = key.shape
    value_dim = value.shape[-1]
    scale = 1.0 / (query.shape[-1] ** 0.5)

    if sequence_length == 1:
        if HAS_TRITON_LINEAR_ATTN and query.is_cuda:
            last_recurrent_state = (
                torch.zeros(batch_size, num_heads, key_dim, value_dim, device=value.device, dtype=value.dtype)
                if initial_state is None
                else initial_state.to(value)
            )
            if not output_final_state:
                last_recurrent_state = last_recurrent_state.clone()

            core_attn_out = recurrent_gated_delta_decode(
                query[:, :, 0],
                key[:, :, 0],
                value[:, :, 0],
                gate[:, :, 0],
                beta[:, :, 0],
                last_recurrent_state,
                query_scale=scale,
                normalize_qk=use_qk_l2norm_in_kernel,
                output_dtype=initial_dtype if _QWEN35_LINEAR_CORE_FP16_OUT else None,
            )

            if not output_final_state:
                last_recurrent_state = None

            if not _QWEN35_LINEAR_CORE_FP16_OUT:
                core_attn_out = core_attn_out.to(initial_dtype)
            return core_attn_out.unsqueeze(1), last_recurrent_state

        query_t = query[:, :, 0]
        key_t = key[:, :, 0]
        query_t = query_t * scale
        value_t = value[:, :, 0]
        gate_t = gate[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, 0].unsqueeze(-1)

        last_recurrent_state = (
            torch.zeros(batch_size, num_heads, key_dim, value_dim, device=value.device, dtype=value.dtype)
            if initial_state is None
            else initial_state.to(value)
        )
        last_recurrent_state = last_recurrent_state * gate_t
        kv_mem = (last_recurrent_state * key_t.unsqueeze(-1)).sum(dim=-2)
        delta = (value_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + key_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out = (last_recurrent_state * query_t.unsqueeze(-1)).sum(dim=-2)

        if not output_final_state:
            last_recurrent_state = None

        return core_attn_out.unsqueeze(1).to(initial_dtype), last_recurrent_state

    query = query * scale
    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, value_dim, device=value.device, dtype=value.dtype)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, key_dim, value_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        query_t = query[:, :, i]
        key_t = key[:, :, i]
        value_t = value[:, :, i]
        gate_t = gate[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * gate_t
        kv_mem = (last_recurrent_state * key_t.unsqueeze(-1)).sum(dim=-2)
        delta = (value_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + key_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * query_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype)
    return core_attn_out, last_recurrent_state


def recurrent_gated_delta_decode_step(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    num_kv_groups: int = 1,
    use_qk_l2norm_in_kernel: bool = False,
):
    """
    Decode-only recurrent delta rule without seq_len transposes/copies.

    Shapes:
      query/key: [batch, q_heads, key_dim]
      value: [batch, kv_heads, value_dim]
      gate/beta: [batch, kv_heads]
      initial_state: [batch, kv_heads, key_dim, value_dim]
    """
    initial_dtype = query.dtype
    can_use_triton = (
        HAS_TRITON_LINEAR_ATTN
        and recurrent_gated_delta_decode is not None
        and query.is_cuda
        and not _env_enabled("MEGAGEMM_QWEN35_DISABLE_TRITON_LINEAR_DECODE", default=False)
    )
    can_normalize_in_triton = (
        use_qk_l2norm_in_kernel
        and can_use_triton
    )
    if use_qk_l2norm_in_kernel and not can_normalize_in_triton:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    batch_size, q_heads, key_dim = key.shape
    num_heads = value.shape[1]
    value_dim = value.shape[-1]
    scale = 1.0 / (query.shape[-1] ** 0.5)

    if can_use_triton:
        last_recurrent_state = (
            torch.zeros(batch_size, num_heads, key_dim, value_dim, device=value.device, dtype=torch.float32)
            if initial_state is None
            else initial_state.to(dtype=torch.float32, device=value.device)
        )
        if not output_final_state:
            last_recurrent_state = last_recurrent_state.clone()
        core_attn_out = recurrent_gated_delta_decode(
            query, key, value, gate, beta, last_recurrent_state,
            num_kv_groups=num_kv_groups,
            query_scale=scale,
            normalize_qk=use_qk_l2norm_in_kernel,
            output_dtype=initial_dtype if _QWEN35_LINEAR_CORE_FP16_OUT else None,
        )
        if not output_final_state:
            last_recurrent_state = None
        if not _QWEN35_LINEAR_CORE_FP16_OUT:
            core_attn_out = core_attn_out.to(initial_dtype)
        return core_attn_out, last_recurrent_state

    query = query.float()
    key = key.float()
    value = value.float()
    beta = beta.float()
    gate = gate.float()
    if num_kv_groups > 1:
        query = query.repeat_interleave(num_kv_groups, dim=1)
        key = key.repeat_interleave(num_kv_groups, dim=1)
    query = query * scale
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, key_dim, value_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value)
    )
    gate_t = gate.exp().unsqueeze(-1).unsqueeze(-1)
    beta_t = beta.unsqueeze(-1)
    last_recurrent_state = last_recurrent_state * gate_t
    kv_mem = (last_recurrent_state * key.unsqueeze(-1)).sum(dim=-2)
    delta = (value - kv_mem) * beta_t
    last_recurrent_state = last_recurrent_state + key.unsqueeze(-1) * delta.unsqueeze(-2)
    core_attn_out = (last_recurrent_state * query.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None

    return core_attn_out.to(initial_dtype), last_recurrent_state


def recurrent_gated_delta_prefill_short_sequence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    num_kv_groups: int = 1,
    use_qk_l2norm_in_kernel: bool = False,
):
    """
    Short-sequence prefill path that reuses the decode kernel step-by-step.

    This avoids the chunked fallback overhead for small prompts where TTFT is
    dominated by padding, reshapes, and dense chunk materialization.
    """
    initial_dtype = query.dtype
    batch_size, sequence_length, q_heads, key_dim = key.shape
    num_heads = value.shape[2]
    value_dim = value.shape[-1]

    can_use_triton_prefill = (
        HAS_TRITON_LINEAR_ATTN
        and recurrent_gated_delta_prefill is not None
        and query.is_cuda
    )
    if can_use_triton_prefill:
        state = (
            torch.zeros(batch_size, num_heads, key_dim, value_dim, device=value.device, dtype=torch.float32)
            if initial_state is None
            else initial_state.to(dtype=torch.float32, device=value.device)
        )
        query_kernel = query.transpose(1, 2).contiguous()
        key_kernel = key.transpose(1, 2).contiguous()
        value_kernel = value.transpose(1, 2).contiguous()
        gate_kernel = gate.transpose(1, 2).contiguous()
        beta_kernel = beta.transpose(1, 2).contiguous()
        scale = 1.0 / (query_kernel.shape[-1] ** 0.5)

        if not output_final_state:
            state = state.clone()
        if sequence_length <= 64:
            core_attn_out = recurrent_gated_delta_prefill(
                query_kernel,
                key_kernel,
                value_kernel,
                gate_kernel,
                beta_kernel,
                state,
                num_kv_groups=num_kv_groups,
                query_scale=scale,
                normalize_qk=use_qk_l2norm_in_kernel,
            )
        else:
            outputs = []
            for start in range(0, sequence_length, 64):
                end = min(start + 64, sequence_length)
                outputs.append(
                    recurrent_gated_delta_prefill(
                        query_kernel[:, :, start:end, :],
                        key_kernel[:, :, start:end, :],
                        value_kernel[:, :, start:end, :],
                        gate_kernel[:, :, start:end],
                        beta_kernel[:, :, start:end],
                        state,
                        num_kv_groups=num_kv_groups,
                        query_scale=scale,
                        normalize_qk=use_qk_l2norm_in_kernel,
                    )
                )
            core_attn_out = torch.cat(outputs, dim=2)
        if not output_final_state:
            state = None
        return core_attn_out.transpose(1, 2).to(initial_dtype), state

    state = (
        torch.zeros(batch_size, num_heads, key_dim, value_dim, device=value.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.to(dtype=torch.float32, device=value.device)
    )
    if num_kv_groups > 1:
        query = query.repeat_interleave(num_kv_groups, dim=2)
        key = key.repeat_interleave(num_kv_groups, dim=2)
    outputs = []
    for idx in range(sequence_length):
        step_out, state = recurrent_gated_delta_decode_step(
            query[:, idx],
            key[:, idx],
            value[:, idx],
            gate[:, idx],
            beta[:, idx],
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        outputs.append(step_out.unsqueeze(1))

    core_attn_out = torch.cat(outputs, dim=1)
    if not output_final_state:
        state = None

    return core_attn_out.to(initial_dtype), state


class GatedDeltaNet(nn.Module):
    """PyTorch fallback implementation of Qwen 3.5 linear attention."""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.num_kv_groups = self.num_v_heads // self.num_k_heads
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            padding=self.conv_kernel_size - 1,
            groups=self.conv_dim,
            bias=False,
        )
        self.in_proj_baz = nn.Linear(
            self.hidden_size, self.value_dim + 2 * self.num_v_heads, bias=False,
        )
        self.in_proj_baz_dim = self.value_dim + 2 * self.num_v_heads
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads).uniform_(0.0, 16.0).log_())
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self._short_prefill_threshold_override = _get_env_int_optional("MEGAGEMM_QWEN35_SHORT_PREFILL_THRESHOLD")
        self._enable_chunk_scan_override = _get_env_bool_optional("MEGAGEMM_QWEN35_ENABLE_CHUNK_SCAN")
        self.short_prefill_threshold = self._short_prefill_threshold_override or 160
        self.enable_chunk_scan = self._enable_chunk_scan_override if self._enable_chunk_scan_override is not None else False
        self.chunk_size = min(_get_env_int("MEGAGEMM_QWEN35_CHUNK_SIZE", 64), 64)
        self._in_proj_fused_weight: Optional[torch.Tensor] = None
        self._in_proj_qkv_version: int = -1
        self._in_proj_baz_version: int = -1
        self._runtime_policy_logged = False
        self._fused_rmsnorm_in_proj_key = None
        self._fused_rmsnorm_in_proj_use = False
        self._disable_fused_rmsnorm_in_proj = False
        self._decode_fused_in_proj_hits = 0
        self._decode_ab_fused_disabled = False
        self._decode_ab_fused_hits = 0
        self._decode_fast_in_proj_hits = 0
        self._decode_fast_out_proj_hits = 0
        self._fused_norm_out_key = None
        self._fused_norm_out_use = False
        self._fused_norm_out_disabled = False
        self._fused_norm_out_hits = 0

    def _build_conv_state(self, mixed_qkv: torch.Tensor) -> torch.Tensor:
        if mixed_qkv.shape[-1] >= self.conv_kernel_size:
            return mixed_qkv[..., -self.conv_kernel_size:].contiguous()
        pad = self.conv_kernel_size - mixed_qkv.shape[-1]
        return torch.nn.functional.pad(mixed_qkv, (pad, 0))

    def _get_fused_in_proj_weight(self) -> torch.Tensor:
        qkv_weight = self.in_proj_qkv.weight
        baz_weight = self.in_proj_baz.weight
        qkv_version = qkv_weight._version
        baz_version = baz_weight._version
        fused_weight = self._in_proj_fused_weight
        if (
            fused_weight is None
            or self._in_proj_qkv_version != qkv_version
            or self._in_proj_baz_version != baz_version
            or fused_weight.device != qkv_weight.device
            or fused_weight.dtype != qkv_weight.dtype
        ):
            self._in_proj_fused_weight = torch.cat([qkv_weight, baz_weight], dim=0).contiguous()
            self._in_proj_qkv_version = qkv_version
            self._in_proj_baz_version = baz_version
            fused_weight = self._in_proj_fused_weight
        return fused_weight

    def _runtime_policy(self, device: torch.device) -> Tuple[int, bool]:
        auto_threshold, auto_chunk_scan = _resolve_qwen35_runtime_policy(device)
        threshold = (
            self._short_prefill_threshold_override
            if self._short_prefill_threshold_override is not None
            else auto_threshold
        )
        chunk_scan = (
            self._enable_chunk_scan_override
            if self._enable_chunk_scan_override is not None
            else auto_chunk_scan
        )
        return threshold, chunk_scan

    def _decode_norm_out_baseline(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
        out_weight: torch.Tensor,
        out_bias: Optional[torch.Tensor],
        use_fast_out_proj: bool,
    ) -> torch.Tensor:
        batch_size, seq_len = core_attn_out.shape[:2]
        normed = self.norm(
            core_attn_out.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        )
        normed = normed.reshape(batch_size, seq_len, self.value_dim)
        return _decode_linear(
            self,
            "_linear_attn_out_proj_out",
            normed,
            out_weight,
            out_bias,
            use_fast=use_fast_out_proj,
        )

    def _decode_norm_out_fused(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
        out_weight: torch.Tensor,
        out_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = int(core_attn_out.shape[0])
        out_2d = _get_reusable_out_decode(
            self,
            "_linear_attn_fused_norm_out",
            (batch_size, self.hidden_size),
            core_attn_out,
        )
        out_2d = rmsnorm_gated_linear_decode(
            core_attn_out[:, 0],
            z[:, 0],
            self.norm.weight,
            self.norm.eps,
            out_weight,
            out_bias,
            out=out_2d,
        )
        return out_2d.unsqueeze(1)

    def _decode_in_proj_baseline_from_raw(
        self,
        hidden_states: torch.Tensor,
        input_norm_weight: torch.Tensor,
        input_norm_eps: float,
        input_norm_offset: bool,
        fused_in_proj_weight: torch.Tensor,
    ) -> torch.Tensor:
        normed = _decode_rmsnorm(
            hidden_states,
            input_norm_weight,
            input_norm_eps,
            input_norm_offset,
        )
        use_fast_in_proj = _USE_DECODE_FAST_LINEAR or _pick_decode_linear_backend(
            self,
            "linear_attn_in",
            normed,
            fused_in_proj_weight,
            None,
            "_linear_attn_in_proj_out",
        )
        return _decode_linear(
            self,
            "_linear_attn_in_proj_out",
            normed,
            fused_in_proj_weight,
            None,
            use_fast=use_fast_in_proj,
        )

    def _should_use_fused_norm_in_proj(
        self,
        hidden_states: torch.Tensor,
        input_norm_weight: torch.Tensor,
        input_norm_eps: float,
        input_norm_offset: bool,
        fused_in_proj_weight: torch.Tensor,
    ) -> bool:
        if (
            not _QWEN35_FUSED_RMSNORM_IN_PROJ_DECODE
            or fused_rmsnorm_linear is None
            or self._disable_fused_rmsnorm_in_proj
            or torch.is_grad_enabled()
            or not hidden_states.is_cuda
            or (
                _QWEN35_FUSED_RMSNORM_IN_PROJ_MAX_HIDDEN > 0
                and self.hidden_size > _QWEN35_FUSED_RMSNORM_IN_PROJ_MAX_HIDDEN
            )
        ):
            return False
        if hidden_states.dim() == 3:
            rows = int(hidden_states.shape[0] * hidden_states.shape[1])
        elif hidden_states.dim() == 2:
            rows = int(hidden_states.shape[0])
        else:
            return False
        out_features = int(fused_in_proj_weight.shape[0])
        if callable(fused_rmsnorm_linear_prefers_triton_shape):
            if not fused_rmsnorm_linear_prefers_triton_shape(
                int(hidden_states.shape[-1]),
                out_features,
                rows,
            ):
                return False
        sig = (
            rows,
            int(hidden_states.shape[-1]),
            out_features,
            str(hidden_states.dtype),
            str(fused_in_proj_weight.dtype),
            hidden_states.device.type,
            int(hidden_states.device.index) if hidden_states.device.index is not None else -1,
            bool(input_norm_offset),
        )
        cached = _QWEN35_FUSED_RMSNORM_IN_PROJ_DECISION_CACHE.get(sig)
        if cached is not None:
            self._fused_rmsnorm_in_proj_key = sig
            self._fused_rmsnorm_in_proj_use = bool(cached)
            return bool(cached)
        if self._fused_rmsnorm_in_proj_key == sig:
            return bool(self._fused_rmsnorm_in_proj_use)
        try:
            fused_proj_out = _get_reusable_out_decode(
                self,
                "_linear_attn_in_proj_out",
                (*hidden_states.shape[:-1], out_features),
                hidden_states,
            )
            fused_rmsnorm_linear(
                hidden_states,
                input_norm_weight,
                input_norm_eps,
                fused_in_proj_weight,
                None,
                norm_offset=input_norm_offset,
                out=fused_proj_out,
            )
            self._decode_in_proj_baseline_from_raw(
                hidden_states,
                input_norm_weight,
                input_norm_eps,
                input_norm_offset,
                fused_in_proj_weight,
            )
            torch.cuda.synchronize()
            fused_ms = _cuda_bench_ms(
                lambda: fused_rmsnorm_linear(
                    hidden_states,
                    input_norm_weight,
                    input_norm_eps,
                    fused_in_proj_weight,
                    None,
                    norm_offset=input_norm_offset,
                    out=fused_proj_out,
                ),
                iters=8,
            )
            base_ms = _cuda_bench_ms(
                lambda: self._decode_in_proj_baseline_from_raw(
                    hidden_states,
                    input_norm_weight,
                    input_norm_eps,
                    input_norm_offset,
                    fused_in_proj_weight,
                ),
                iters=8,
            )
            self._fused_rmsnorm_in_proj_use = bool(
                fused_ms <= (base_ms * (1.0 - _QWEN35_FUSED_RMSNORM_IN_PROJ_MIN_GAIN))
            )
            self._fused_rmsnorm_in_proj_key = sig
            _QWEN35_FUSED_RMSNORM_IN_PROJ_DECISION_CACHE[sig] = bool(
                self._fused_rmsnorm_in_proj_use
            )
        except Exception:
            self._disable_fused_rmsnorm_in_proj = True
            self._fused_rmsnorm_in_proj_use = False
            self._fused_rmsnorm_in_proj_key = sig
            _QWEN35_FUSED_RMSNORM_IN_PROJ_DECISION_CACHE[sig] = False
        return bool(self._fused_rmsnorm_in_proj_use)

    def _should_use_fused_norm_out(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
        out_weight: torch.Tensor,
        out_bias: Optional[torch.Tensor],
        use_fast_out_proj: bool,
    ) -> bool:
        if (
            not _QWEN35_FUSED_NORM_OUT
            or rmsnorm_gated_linear_decode is None
            or self._fused_norm_out_disabled
            or torch.is_grad_enabled()
            or not core_attn_out.is_cuda
            or core_attn_out.shape[1] != 1
            or (
                _QWEN35_FUSED_NORM_OUT_MAX_HIDDEN > 0
                and self.hidden_size > _QWEN35_FUSED_NORM_OUT_MAX_HIDDEN
            )
        ):
            return False
        sig = (
            int(core_attn_out.shape[0]),
            int(self.num_v_heads),
            int(self.head_v_dim),
            int(self.hidden_size),
            str(core_attn_out.dtype),
            core_attn_out.device.type,
            int(core_attn_out.device.index) if core_attn_out.device.index is not None else -1,
        )
        if _QWEN35_FUSED_NORM_OUT_GLOBAL_USE_CACHE.get(sig, False):
            self._fused_norm_out_use = True
            self._fused_norm_out_key = sig
            return True
        if self._fused_norm_out_key == sig:
            return bool(self._fused_norm_out_use)
        try:
            self._decode_norm_out_fused(core_attn_out, z, out_weight, out_bias)
            self._decode_norm_out_baseline(core_attn_out, z, out_weight, out_bias, use_fast_out_proj)
            torch.cuda.synchronize()
            fused_ms = _cuda_bench_ms(
                lambda: self._decode_norm_out_fused(core_attn_out, z, out_weight, out_bias),
                iters=8,
            )
            base_ms = _cuda_bench_ms(
                lambda: self._decode_norm_out_baseline(
                    core_attn_out, z, out_weight, out_bias, use_fast_out_proj
                ),
                iters=8,
            )
            self._fused_norm_out_use = bool(
                fused_ms <= (base_ms * (1.0 - _QWEN35_FUSED_NORM_OUT_MIN_GAIN))
            )
            self._fused_norm_out_key = sig
            if self._fused_norm_out_use:
                _QWEN35_FUSED_NORM_OUT_GLOBAL_USE_CACHE[sig] = True
        except Exception:
            self._fused_norm_out_disabled = True
            self._fused_norm_out_use = False
            self._fused_norm_out_key = sig
        return bool(self._fused_norm_out_use)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        recurrent_state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        timing_events: Optional[dict] = None,
        input_is_normed: bool = True,
        input_norm_weight: Optional[torch.Tensor] = None,
        input_norm_eps: float = 1e-6,
        input_norm_offset: bool = False,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        do_timing = (
            timing_events is not None
            and seq_len == 1
            and hidden_states.is_cuda
            and _timing_enabled()
        )
        short_prefill_threshold, enable_chunk_scan = self._runtime_policy(hidden_states.device)
        self.short_prefill_threshold = short_prefill_threshold
        self.enable_chunk_scan = enable_chunk_scan
        use_precomputed_states = (
            use_cache
            and conv_state is not None
            and recurrent_state is not None
            and seq_len == 1
        )
        if not self._runtime_policy_logged and self.layer_idx == 0:
            threshold_src = "env" if self._short_prefill_threshold_override is not None else "auto"
            chunk_src = "env" if self._enable_chunk_scan_override is not None else "auto"
            print(
                f"⚙️  Qwen3.5 runtime policy: short_prefill_threshold={short_prefill_threshold} "
                f"({threshold_src}), chunk_scan={'on' if enable_chunk_scan else 'off'} ({chunk_src})"
            )
            self._runtime_policy_logged = True
        proj_start_end = _timing_record_start(do_timing)
        fused_in_proj_weight = self._get_fused_in_proj_weight()
        fused_proj = None
        use_fused_norm_in_proj = (
            use_precomputed_states
            and not input_is_normed
            and input_norm_weight is not None
            and not torch.is_grad_enabled()
        )
        if use_fused_norm_in_proj and self._should_use_fused_norm_in_proj(
            hidden_states,
            input_norm_weight,
            input_norm_eps,
            input_norm_offset,
            fused_in_proj_weight,
        ):
            try:
                fused_proj_out = _get_reusable_out_decode(
                    self,
                    "_linear_attn_in_proj_out",
                    (*hidden_states.shape[:-1], self.conv_dim + self.in_proj_baz_dim),
                    hidden_states,
                )
                fused_proj = fused_rmsnorm_linear(
                    hidden_states,
                    input_norm_weight,
                    input_norm_eps,
                    fused_in_proj_weight,
                    None,
                    norm_offset=input_norm_offset,
                    out=fused_proj_out,
                )
                self._decode_fused_in_proj_hits += 1
            except Exception:
                self._disable_fused_rmsnorm_in_proj = True
                self._fused_rmsnorm_in_proj_use = False
                fused_proj = None
        if fused_proj is None and not input_is_normed and input_norm_weight is not None:
            hidden_states = _decode_rmsnorm(
                hidden_states,
                input_norm_weight,
                input_norm_eps,
                input_norm_offset,
            )
        if fused_proj is None and use_precomputed_states and not torch.is_grad_enabled():
            use_fast_in_proj = _USE_DECODE_FAST_LINEAR or _pick_decode_linear_backend(
                self,
                "linear_attn_in",
                hidden_states,
                fused_in_proj_weight,
                None,
                "_linear_attn_in_proj_out",
            )
            fused_proj = _decode_linear(
                self,
                "_linear_attn_in_proj_out",
                hidden_states,
                fused_in_proj_weight,
                None,
                use_fast=use_fast_in_proj,
            )
            if use_fast_in_proj:
                self._decode_fast_in_proj_hits += 1
        elif fused_proj is None:
            fused_proj = torch.nn.functional.linear(hidden_states, fused_in_proj_weight)
        _timing_record_end(timing_events, "linear_attn_proj", proj_start_end)
        mixed_qkv, baz = fused_proj.split([self.conv_dim, self.in_proj_baz_dim], dim=-1)
        z, b, a = baz.split([self.value_dim, self.num_v_heads, self.num_v_heads], dim=-1)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        use_short_prefill = (
            not use_precomputed_states
            and seq_len <= self.short_prefill_threshold
            and hidden_states.is_cuda
            and HAS_TRITON_LINEAR_ATTN
        )

        conv_start_end = _timing_record_start(do_timing)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        if use_precomputed_states:
            conv_out = None
            if _QWEN35_REUSE_LINEAR_DECODE_BUFFERS:
                conv_out = _get_reusable_out_decode(
                    self,
                    "_linear_attn_conv_update_out",
                    tuple(mixed_qkv.shape),
                    mixed_qkv,
                )
            mixed_qkv, conv_state = torch_causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                out=conv_out,
            )
        else:
            if use_cache:
                conv_state = self._build_conv_state(mixed_qkv)
            mixed_qkv = torch.nn.functional.silu(self.conv1d(mixed_qkv)[..., :seq_len])
            if use_cache:
                conv_state = conv_state.contiguous()

        mixed_qkv = mixed_qkv.transpose(1, 2)
        _timing_record_end(timing_events, "linear_attn_conv", conv_start_end)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        use_fused_ab_decode = (
            use_precomputed_states
            and recurrent_gated_delta_decode_from_ab is not None
            and not self._decode_ab_fused_disabled
            and not _env_enabled("MEGAGEMM_QWEN35_DISABLE_FUSED_AB_DECODE", default=False)
            and query.is_cuda
            and recurrent_state is not None
            and not torch.is_grad_enabled()
        )
        beta = None
        gk = None
        if not use_fused_ab_decode:
            gates_start_end = _timing_record_start(do_timing)
            beta, gk = _fused_linear_gates(a, b, self.A_log, self.dt_bias)
            _timing_record_end(timing_events, "linear_attn_gates", gates_start_end)

        core_start_end = _timing_record_start(do_timing)
        if use_precomputed_states:
            if use_fused_ab_decode:
                try:
                    recurrent_state = recurrent_state.to(dtype=torch.float32, device=value.device)
                    if not use_cache:
                        recurrent_state = recurrent_state.clone()
                    core_out = None
                    if _QWEN35_REUSE_LINEAR_DECODE_BUFFERS:
                        core_out_dtype = query.dtype if _QWEN35_LINEAR_CORE_FP16_OUT else torch.float32
                        core_out_ref = query if core_out_dtype == query.dtype else recurrent_state
                        core_out = _get_reusable_out(
                            self,
                            "_linear_attn_core_out",
                            (batch_size, self.num_v_heads, self.head_v_dim),
                            core_out_ref,
                        )
                    core_attn_out = recurrent_gated_delta_decode_from_ab(
                        query[:, 0],
                        key[:, 0],
                        value[:, 0],
                        a[:, 0],
                        b[:, 0],
                        self.A_log,
                        self.dt_bias,
                        recurrent_state,
                        num_kv_groups=self.num_kv_groups,
                        query_scale=1.0 / (self.head_k_dim ** 0.5),
                        normalize_qk=True,
                        output_dtype=query.dtype if _QWEN35_LINEAR_CORE_FP16_OUT else None,
                        out=core_out,
                    )
                    if not _QWEN35_LINEAR_CORE_FP16_OUT:
                        core_attn_out = core_attn_out.to(query.dtype)
                    self._decode_ab_fused_hits += 1
                    if not use_cache:
                        recurrent_state = None
                except Exception:
                    self._decode_ab_fused_disabled = True
                    beta, gk = _fused_linear_gates(a, b, self.A_log, self.dt_bias)
                    core_attn_out, recurrent_state = recurrent_gated_delta_decode_step(
                        query[:, 0],
                        key[:, 0],
                        value[:, 0],
                        gk[:, 0],
                        beta[:, 0],
                        initial_state=recurrent_state,
                        output_final_state=use_cache,
                        num_kv_groups=self.num_kv_groups,
                        use_qk_l2norm_in_kernel=True,
                    )
            else:
                core_attn_out, recurrent_state = recurrent_gated_delta_decode_step(
                    query[:, 0],
                    key[:, 0],
                    value[:, 0],
                    gk[:, 0],
                    beta[:, 0],
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    num_kv_groups=self.num_kv_groups,
                    use_qk_l2norm_in_kernel=True,
                )
            core_attn_out = core_attn_out.unsqueeze(1)
        elif use_short_prefill:
            core_attn_out, recurrent_state = recurrent_gated_delta_prefill_short_sequence(
                query,
                key,
                value,
                gk,
                beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                num_kv_groups=self.num_kv_groups,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            if self.num_kv_groups > 1:
                query = query.repeat_interleave(self.num_kv_groups, dim=2)
                key = key.repeat_interleave(self.num_kv_groups, dim=2)
            core_attn_out, recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                gk,
                beta,
                chunk_size=self.chunk_size,
                initial_state=None,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                enable_chunk_scan=enable_chunk_scan,
            )
        _timing_record_end(timing_events, "linear_attn_core", core_start_end)

        out_weight, out_bias = _linear_weight_bias(self.out_proj)
        use_decode_out_proj = (
            use_precomputed_states
            and out_weight is not None
            and not torch.is_grad_enabled()
        )
        use_fast_out_proj = False
        if use_decode_out_proj:
            out_ref = core_attn_out.reshape(batch_size, seq_len, self.value_dim)
            use_fast_out_proj = _USE_DECODE_FAST_LINEAR or _pick_decode_linear_backend(
                self,
                "linear_attn_out",
                out_ref,
                out_weight,
                out_bias,
                "_linear_attn_out_proj_out",
            )

        if (
            use_decode_out_proj
            and self._should_use_fused_norm_out(
                core_attn_out,
                z,
                out_weight,
                out_bias,
                use_fast_out_proj,
            )
        ):
            norm_out_start_end = _timing_record_start(do_timing)
            out = self._decode_norm_out_fused(core_attn_out, z, out_weight, out_bias)
            self._fused_norm_out_hits += 1
            _timing_record_end(timing_events, "linear_attn_norm_out", norm_out_start_end)
        else:
            norm_start_end = _timing_record_start(do_timing)
            core_attn_out = self.norm(
                core_attn_out.reshape(-1, self.head_v_dim),
                z.reshape(-1, self.head_v_dim),
            )
            core_attn_out = core_attn_out.reshape(batch_size, seq_len, self.value_dim)
            _timing_record_end(timing_events, "linear_attn_norm", norm_start_end)

            out_start_end = _timing_record_start(do_timing)
            if use_decode_out_proj:
                out = _decode_linear(
                    self,
                    "_linear_attn_out_proj_out",
                    core_attn_out,
                    out_weight,
                    out_bias,
                    use_fast=use_fast_out_proj,
                )
                if use_fast_out_proj:
                    self._decode_fast_out_proj_hits += 1
            else:
                out = self.out_proj(core_attn_out)
            _timing_record_end(timing_events, "linear_attn_out_proj", out_start_end)

        if attention_mask is not None:
            out = apply_mask_to_padding_states(out, attention_mask)

        return out, conv_state if use_cache else None, recurrent_state if use_cache else None


class LlamaAttention(nn.Module):
    """Multi-head attention with GQA, RoPE, optional QKV bias and QK-Norm."""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = (
            config.layer_types[layer_idx]
            if config.layer_types and layer_idx < len(config.layer_types)
            else 'full_attention'
        )
        self.is_gemma4 = config.model_type == 'gemma4_text'
        self.kv_share_source = (
            config.kv_share_sources[layer_idx]
            if config.kv_share_sources and layer_idx < len(config.kv_share_sources)
            else None
        )
        self.is_kv_shared = self.kv_share_source is not None
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = (
            config.per_layer_num_kv_heads[layer_idx]
            if config.per_layer_num_kv_heads and layer_idx < len(config.per_layer_num_kv_heads)
            else config.num_key_value_heads
        )
        self.head_dim = (
            config.per_layer_head_dims[layer_idx]
            if config.per_layer_head_dims and layer_idx < len(config.per_layer_head_dims)
            else config.head_dim
        )
        self.scale = 1.0 if self.is_gemma4 else 1.0 / math.sqrt(self.head_dim)
        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.rope_half_rotate = config.rope_half_rotate
        self.attention_output_gate = config.attention_output_gate
        self.rotary_dim = (
            config.per_layer_rotary_dims[layer_idx]
            if config.per_layer_rotary_dims and layer_idx < len(config.per_layer_rotary_dims)
            else (config.rotary_dim or self.head_dim)
        )
        self.sliding_window = config.sliding_window if self.layer_type == 'sliding_attention' else 0
        self.attention_k_eq_v = (
            self.is_gemma4
            and self.layer_type == 'full_attention'
            and bool(config.attention_k_eq_v)
        )

        self._q_size = self.num_q_heads * self.head_dim
        self._q_proj_size = self._q_size * (2 if self.attention_output_gate else 1)
        self._k_size = self.num_kv_heads * self.head_dim
        self._v_size = self.num_kv_heads * self.head_dim
        self._awq_separate = False  # Set True by loader for AWQ

        bias = config.attention_bias

        # Fused QKV projection (FP16 path — replaced by separate Q/K/V for AWQ)
        if self.is_gemma4:
            self.q_proj = nn.Linear(config.hidden_size, self._q_proj_size, bias=bias)
            self.k_proj = (
                None
                if self.is_kv_shared
                else nn.Linear(config.hidden_size, self._k_size, bias=bias)
            )
            self.v_proj = (
                None
                if self.is_kv_shared or self.attention_k_eq_v
                else nn.Linear(config.hidden_size, self._v_size, bias=bias)
            )
            self.qkv_proj = None
        else:
            self.qkv_proj = nn.Linear(
                config.hidden_size,
                self._q_proj_size + self._k_size + self._v_size,
                bias=bias,
            )

        # Separate projections (AWQ path — set by loader)
        # self.q_proj = QuantizedLinear(...)  # set by loader
        # self.k_proj = QuantizedLinear(...)  # set by loader
        # self.v_proj = QuantizedLinear(...)  # set by loader

        self.o_proj = nn.Linear(
            self.num_q_heads * self.head_dim,
            config.hidden_size,
            bias=bias if self.is_gemma4 else False,
        )

        # Qwen 3: QK-Norm (RMSNorm on Q and K before RoPE)
        self.q_norm = None
        self.k_norm = None
        self.v_norm = None
        if config.qk_norm:
            self.q_norm = MGRMSNorm(
                self.head_dim, eps=config.rms_norm_eps, offset=config.qk_norm_offset
            )
            if not self.is_kv_shared:
                self.k_norm = MGRMSNorm(
                    self.head_dim, eps=config.rms_norm_eps, offset=config.qk_norm_offset
                )
            if self.is_gemma4 and not self.is_kv_shared:
                self.v_norm = MGRMSNorm(
                    self.head_dim, eps=config.rms_norm_eps, with_scale=False
                )
        self._disable_fused_decode = self.is_gemma4
        self._fused_decode_checked = False
        self._fused_decode_hits = 0
        self._fused_decode_disable_reason: Optional[str] = (
            "Gemma4 uses per-layer RoPE/KV sharing" if self.is_gemma4 else None
        )
        self._fast_qkv_out = None
        self._fast_q_out = None
        self._fast_k_out = None
        self._fast_v_out = None
        self._prefill_qkv_out = None
        self._prefill_q_out = None
        self._prefill_k_out = None
        self._prefill_v_out = None
        self._prefill_o_out = None
        self._fast_o_out = None
        self._fast_attn_out = None
        self._gemma4_qkv_weight = None
        self._gemma4_qkv_bias = None
        self._gemma4_qkv_cache_key = None
        self._gemma4_fused_qkv_prefill_hits = 0
        self._gemma4_fused_qkv_prefill_skip_reason = ""
        self._gemma4_fused_attn_prepare_hits = 0
        self._gemma4_fused_attn_prepare_skip_reason = ""
        self._gemma4_fused_attn_prepare_disabled = False
        self._gemma4_implicit_causal_prefill_hits = 0
        self._gemma4_e2b_l4_sliding_prefill_enabled = False
        self._gemma4_e2b_l4_sliding_prefill_hits = 0
        self._gemma4_e2b_l4_full_prefill_expand_enabled = False
        self._gemma4_e2b_l4_full_prefill_expand_hits = 0
        self._gemma4_e2b_l4_full_prefill_expand_error = ""
        self._gemma4_long_sliding_prefill_hits = 0
        self._gemma4_long_full_prefill_hits = 0
        self._prefill_prepared_q = None
        self._prefill_prepared_k = None
        self._prefill_prepared_v = None
        self._prefill_prepared_k_cache = None
        self._prefill_prepared_v_cache = None
        self._fast_qkv_decode_key = None
        self._fast_qkv_decode_use = False
        self._fast_o_decode_key = None
        self._fast_o_decode_use = False
        self._fused_rmsnorm_qkv_decode_key = None
        self._fused_rmsnorm_qkv_decode_use = False
        self._fused_rmsnorm_qkv_checked = False
        self._disable_fused_rmsnorm_qkv_decode = False
        self._disable_fused_rmsnorm_qkv_prefill = False

    def _gather_prefix_kv_from_cache(
        self,
        kv_cache: torch.Tensor,
        block_table_row: torch.Tensor,
        prefix_len: int,
        target_dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if prefix_len <= 0:
            return None, None

        block_size = int(kv_cache.shape[3])
        used_blocks = (int(prefix_len) + block_size - 1) // block_size
        phys_blocks = block_table_row[:used_blocks].to(device=kv_cache.device, dtype=torch.long)
        prefix_blocks = kv_cache.index_select(0, phys_blocks)

        prefix_k = (
            prefix_blocks[:, 0]
            .permute(1, 0, 2, 3)
            .contiguous()
            .view(self.num_kv_heads, used_blocks * block_size, self.head_dim)
        )
        prefix_v = (
            prefix_blocks[:, 1]
            .permute(1, 0, 2, 3)
            .contiguous()
            .view(self.num_kv_heads, used_blocks * block_size, self.head_dim)
        )

        prefix_k = prefix_k[:, :prefix_len, :].unsqueeze(0).to(dtype=target_dtype)
        prefix_v = prefix_v[:, :prefix_len, :].unsqueeze(0).to(dtype=target_dtype)
        return prefix_k, prefix_v

    def _prefill_attention_with_prefix_cache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run suffix prefill where queries attend to cached prefix KV plus new suffix KV."""
        if block_table.dim() == 1:
            block_table = block_table.unsqueeze(0)
        if seq_lens.dim() == 0:
            seq_lens = seq_lens.unsqueeze(0)

        suffix_len = int(q.shape[2])
        output = torch.empty_like(q)

        for batch_idx in range(q.shape[0]):
            prefix_len = int(seq_lens[batch_idx].item())
            q_i = q[batch_idx : batch_idx + 1]
            k_i = k[batch_idx : batch_idx + 1]
            v_i = v[batch_idx : batch_idx + 1]

            if prefix_len <= 0:
                out_i = prefill_attention(q_i, k_i, v_i, is_causal=True)
            else:
                prefix_k, prefix_v = self._gather_prefix_kv_from_cache(
                    kv_cache,
                    block_table[batch_idx],
                    prefix_len,
                    target_dtype=k_i.dtype,
                )
                combined_k = torch.cat([prefix_k, k_i], dim=2)
                combined_v = torch.cat([prefix_v, v_i], dim=2)

                mask_i = attn_mask
                if (
                    mask_i is None
                    or int(mask_i.shape[-2]) != suffix_len
                    or int(mask_i.shape[-1]) != prefix_len + suffix_len
                ):
                    mask_i = _get_suffix_prefill_attn_mask(
                        prefix_len,
                        suffix_len,
                        q_i.device,
                        q_i.dtype,
                    )

                out_i = prefill_attention(
                    q_i,
                    combined_k,
                    combined_v,
                    is_causal=False,
                    attn_mask=mask_i,
                )

            output[batch_idx : batch_idx + 1].copy_(out_i)

        return output

    def _gemma4_prefill_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        implicit_causal: bool = False,
    ) -> torch.Tensor:
        q_len = int(q.shape[2])
        k_len = int(k.shape[2])
        sliding_is_plain_causal = (
            self.sliding_window > 0
            and q_len == k_len
            and k_len <= int(self.sliding_window)
        )
        if self.sliding_window <= 0:
            e2b_l4_full_batch = int(q.shape[0]) if q.ndim == 4 else 0
            # Loaded-model gates independently validated B1/B2/B4/B8 on L4.
            # The remaining guards keep the promotion on the exact E2B BF16
            # long-context topology.
            e2b_l4_full_batch_allowed = e2b_l4_full_batch in (1, 2, 4, 8)
            use_e2b_l4_expanded_full = bool(
                implicit_causal
                and self._gemma4_e2b_l4_full_prefill_expand_enabled
                and q.ndim == 4
                and k.ndim == 4
                and v.ndim == 4
                and e2b_l4_full_batch_allowed
                and int(q.shape[1]) == 8
                and 2048 <= q_len <= 2304
                and q_len == k_len
                and int(q.shape[3]) == 512
                and tuple(k.shape) == (e2b_l4_full_batch, 1, q_len, 512)
                and tuple(v.shape) == tuple(k.shape)
                and q.dtype == torch.bfloat16
                and k.dtype == q.dtype
                and v.dtype == q.dtype
            )
            if use_e2b_l4_expanded_full:
                try:
                    expanded_k = k.repeat_interleave(8, dim=1)
                    expanded_v = v.repeat_interleave(8, dim=1)
                    expanded_out = prefill_attention(
                        q,
                        expanded_k,
                        expanded_v,
                        is_causal=True,
                        attn_mask=None,
                        scale=self.scale,
                    )
                    self._gemma4_e2b_l4_full_prefill_expand_hits += 1
                    self._gemma4_implicit_causal_prefill_hits += 1
                    return expanded_out
                except Exception as exc:
                    self._gemma4_e2b_l4_full_prefill_expand_enabled = False
                    self._gemma4_e2b_l4_full_prefill_expand_error = (
                        f"{type(exc).__name__}: {exc}"
                    )
            if implicit_causal and _GEMMA4_LONG_FULL_PREFILL:
                long_full_out = gemma4_long_full_prefill_attention(
                    q,
                    k,
                    v,
                    scale=self.scale,
                )
                if long_full_out is not None:
                    self._gemma4_long_full_prefill_hits += 1
                    return long_full_out
            return prefill_attention(
                q,
                k,
                v,
                is_causal=True,
                attn_mask=attn_mask,
                scale=self.scale,
            )

        if sliding_is_plain_causal:
            if _GEMMA4_IMPLICIT_CAUSAL_PREFILL and (
                implicit_causal or attn_mask is None
            ):
                self._gemma4_implicit_causal_prefill_hits += 1
                return prefill_attention(
                    q,
                    k,
                    v,
                    is_causal=True,
                    attn_mask=None,
                    scale=self.scale,
                )
            return prefill_attention(
                q,
                k,
                v,
                is_causal=True,
                attn_mask=attn_mask,
                scale=self.scale,
            )

        if (
            implicit_causal
            and self._gemma4_e2b_l4_sliding_prefill_enabled
        ):
            e2b_l4_out = gemma4_e2b_l4_sliding_prefill_attention(
                q,
                k,
                v,
                sliding_window=int(self.sliding_window),
                scale=self.scale,
            )
            if e2b_l4_out is not None:
                self._gemma4_e2b_l4_sliding_prefill_hits += 1
                return e2b_l4_out

        if implicit_causal and _GEMMA4_LONG_SLIDING_PREFILL:
            long_sliding_out = gemma4_long_sliding_prefill_attention(
                q,
                k,
                v,
                sliding_window=int(self.sliding_window),
                scale=self.scale,
            )
            if long_sliding_out is not None:
                self._gemma4_long_sliding_prefill_hits += 1
                return long_sliding_out

        local_mask = _get_sliding_causal_attn_mask(
            int(q.shape[2]),
            int(k.shape[2]),
            int(self.sliding_window),
            q.device,
            q.dtype,
        )
        # A uniform implicit-causal batch has no padding.  The local mask is
        # already causal, so adding the Bx1xSxS global mask only materializes a
        # redundant batch-expanded tensor (the L4 gate measured it bit-exact).
        if attn_mask is not None and not implicit_causal:
            local_mask = local_mask + attn_mask
        return prefill_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=local_mask,
            scale=self.scale,
        )

    def _gemma4_linear(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        *,
        is_prefill: bool,
        decode_attr: str,
        prefill_attr: str,
        use_fast: Optional[bool] = None,
    ) -> torch.Tensor:
        weight, bias = _linear_weight_bias(module)
        if weight is None:
            return module(hidden_states)
        if is_prefill:
            return _prefill_linear(self, prefill_attr, hidden_states, weight, bias)
        return _decode_linear(
            self,
            decode_attr,
            hidden_states,
            weight,
            bias,
            use_fast=_USE_DECODE_FAST_LINEAR if use_fast is None else use_fast,
        )

    def _gemma4_fused_qkv_weight_bias(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (
            not self.is_gemma4
            or self.is_kv_shared
            or (self.v_proj is None and not self.attention_k_eq_v)
        ):
            return None, None
        q_weight, q_bias = _linear_weight_bias(self.q_proj)
        k_weight, k_bias = _linear_weight_bias(self.k_proj)
        v_from_k = self.v_proj is None and self.attention_k_eq_v
        if v_from_k:
            v_weight, v_bias = None, None
        else:
            v_weight, v_bias = _linear_weight_bias(self.v_proj)
        if q_weight is None or k_weight is None or (not v_from_k and v_weight is None):
            return None, None

        bias_ptrs = tuple(
            0 if bias is None else int(bias.data_ptr())
            for bias in ((q_bias, k_bias) if v_from_k else (q_bias, k_bias, v_bias))
        )
        cache_key = (
            int(q_weight.data_ptr()),
            int(k_weight.data_ptr()),
            0 if v_from_k else int(v_weight.data_ptr()),
            tuple(q_weight.shape),
            tuple(k_weight.shape),
            () if v_from_k else tuple(v_weight.shape),
            q_weight.dtype,
            q_weight.device.type,
            q_weight.device.index,
            bool(v_from_k),
            bias_ptrs,
        )
        if self._gemma4_qkv_cache_key != cache_key:
            weights = [q_weight, k_weight] if v_from_k else [q_weight, k_weight, v_weight]
            qkv_weight = torch.cat(weights, dim=0).contiguous()
            if q_bias is not None or k_bias is not None or (not v_from_k and v_bias is not None):
                zeros_q = q_weight.new_zeros(q_weight.shape[0])
                zeros_k = k_weight.new_zeros(k_weight.shape[0])
                biases = [
                    q_bias if q_bias is not None else zeros_q,
                    k_bias if k_bias is not None else zeros_k,
                ]
                if not v_from_k:
                    zeros_v = v_weight.new_zeros(v_weight.shape[0])
                    biases.append(v_bias if v_bias is not None else zeros_v)
                qkv_bias = torch.cat(biases, dim=0).contiguous()
            else:
                qkv_bias = None
            self._gemma4_qkv_weight = qkv_weight
            self._gemma4_qkv_bias = qkv_bias
            self._gemma4_qkv_cache_key = cache_key
        return self._gemma4_qkv_weight, self._gemma4_qkv_bias

    def _gemma4_fused_qkv(
        self,
        hidden_states: torch.Tensor,
        *,
        is_prefill: bool,
        graph_safe_prefill: bool = False,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if is_prefill:
            if (
                graph_safe_prefill
                and not _GEMMA4_PREFILL_GRAPH_FUSED_ATTN_FRONTEND
            ):
                self._gemma4_fused_qkv_prefill_skip_reason = (
                    "prefill CUDA graph safety guard"
                )
                return None
            rows = int(hidden_states.numel() // max(1, int(hidden_states.shape[-1])))
            if not hidden_states.is_cuda:
                self._gemma4_fused_qkv_prefill_skip_reason = "requires CUDA"
                return None
            if not _gemma4_a100_a4b_fused_qkv_prefill_shape(
                rows,
                int(hidden_states.shape[-1]),
                self._q_proj_size,
                self._k_size,
                self._v_size,
                hidden_states.dtype,
                torch.cuda.get_device_name(hidden_states.device),
            ):
                self._gemma4_fused_qkv_prefill_skip_reason = "shape policy"
                return None
            qkv_weight, qkv_bias = self._gemma4_fused_qkv_weight_bias()
            if qkv_weight is None:
                self._gemma4_fused_qkv_prefill_skip_reason = "weights unavailable"
                return None
            qkv = _prefill_linear(
                self,
                "_prefill_qkv_out",
                hidden_states,
                qkv_weight,
                qkv_bias,
            )
            self._gemma4_fused_qkv_prefill_hits += 1
            self._gemma4_fused_qkv_prefill_skip_reason = ""
            if self.v_proj is None and self.attention_k_eq_v:
                q_raw, k_raw = qkv.split([self._q_proj_size, self._k_size], dim=-1)
                return q_raw, k_raw, k_raw
            return qkv.split([self._q_proj_size, self._k_size, self._v_size], dim=-1)
        if not _GEMMA4_FUSED_QKV_DECODE:
            return None
        qkv_weight, qkv_bias = self._gemma4_fused_qkv_weight_bias()
        if qkv_weight is None:
            return None
        qkv = _decode_linear(
            self,
            "_fast_qkv_out",
            hidden_states,
            qkv_weight,
            qkv_bias,
            use_fast=_USE_DECODE_FAST_LINEAR,
        )
        if self.v_proj is None and self.attention_k_eq_v:
            q_raw, k_raw = qkv.split([self._q_proj_size, self._k_size], dim=-1)
            return q_raw, k_raw, k_raw
        return qkv.split([self._q_proj_size, self._k_size, self._v_size], dim=-1)

    def _gemma4_fused_attention_prepare(
        self,
        q_raw: torch.Tensor,
        k_raw: Optional[torch.Tensor],
        v_raw: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        *,
        graph_safe_prefill: bool = False,
    ) -> Optional[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        if (
            graph_safe_prefill
            and not _GEMMA4_PREFILL_GRAPH_FUSED_ATTN_FRONTEND
        ):
            self._gemma4_fused_attn_prepare_skip_reason = (
                "prefill CUDA graph safety guard"
            )
            return None
        if self._gemma4_fused_attn_prepare_disabled:
            return None
        if not HAS_GEMMA4_ATTENTION_PREPARE or gemma4_prefill_attention_prepare is None:
            self._gemma4_fused_attn_prepare_skip_reason = "kernel unavailable"
            return None
        if k_raw is None or v_raw is None:
            self._gemma4_fused_attn_prepare_skip_reason = "KV unavailable"
            return None
        bsz, seq_len, _ = q_raw.shape
        if not q_raw.is_cuda:
            self._gemma4_fused_attn_prepare_skip_reason = "requires CUDA"
            return None
        if not _gemma4_a100_a4b_fused_attn_prepare_shape(
            bsz,
            seq_len,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rotary_dim,
            q_raw.dtype,
            torch.cuda.get_device_name(q_raw.device),
        ):
            self._gemma4_fused_attn_prepare_skip_reason = "shape policy"
            return None
        if (
            not self.rope_half_rotate
            or self.q_norm is None
            or self.k_norm is None
            or self.v_norm is None
            or self.q_norm.offset
            or self.k_norm.offset
        ):
            self._gemma4_fused_attn_prepare_skip_reason = "unsupported norm/RoPE semantics"
            return None

        q_out = _get_prefill_out(
            self,
            "_prefill_prepared_q",
            (bsz, self.num_q_heads, seq_len, self.head_dim),
            q_raw,
        )
        k_out = _get_prefill_out(
            self,
            "_prefill_prepared_k",
            (bsz, self.num_kv_heads, seq_len, self.head_dim),
            q_raw,
        )
        v_out = _get_prefill_out(
            self,
            "_prefill_prepared_v",
            (bsz, self.num_kv_heads, seq_len, self.head_dim),
            q_raw,
        )
        k_cache = _get_prefill_out(
            self,
            "_prefill_prepared_k_cache",
            (bsz, seq_len, self.num_kv_heads, self.head_dim),
            q_raw,
        )
        v_cache = _get_prefill_out(
            self,
            "_prefill_prepared_v_cache",
            (bsz, seq_len, self.num_kv_heads, self.head_dim),
            q_raw,
        )
        try:
            result = gemma4_prefill_attention_prepare(
                q_raw,
                k_raw,
                v_raw,
                self.q_norm.weight,
                self.k_norm.weight,
                cos,
                sin,
                positions,
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                eps=self.q_norm.eps,
                q_out=q_out,
                k_out=k_out,
                v_out=v_out,
                k_cache=k_cache,
                v_cache=v_cache,
            )
        except Exception as exc:
            self._gemma4_fused_attn_prepare_disabled = True
            self._gemma4_fused_attn_prepare_skip_reason = (
                f"{type(exc).__name__}: {exc}"
            )
            return None
        self._gemma4_fused_attn_prepare_hits += 1
        self._gemma4_fused_attn_prepare_skip_reason = ""
        return result

    def _gemma4_attention(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        seq_lens_kv: Optional[torch.Tensor] = None,
        decode_phys_blocks: Optional[torch.Tensor] = None,
        decode_blk_offsets: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        shared_prefill_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        timing_events: Optional[dict] = None,
        implicit_causal_prefill: bool = False,
        graph_safe_prefill: bool = False,
        prefill_kv_out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        bsz, seq_len, _ = hidden_states.shape
        finite_trace = (
            getattr(self, "_gemma4_prefill_finite_trace", None)
            if is_prefill else None
        )
        do_prefill_stage_timing = bool(
            timing_events is not None
            and is_prefill
            and hidden_states.is_cuda
            and _prefill_timing_enabled()
        )
        prefill_timing_suffix = None
        if do_prefill_stage_timing:
            prefill_timing_suffix = (
                "sliding" if self.sliding_window > 0 else "full"
            )

        use_shared_prefill_kv = self.is_kv_shared and is_prefill and shared_prefill_kv is not None
        if self.is_kv_shared and is_prefill and shared_prefill_kv is None:
            raise RuntimeError(
                "Gemma4 KV-shared prefill requires K/V from the source layer"
            )
        skip_kv_projection = (self.is_kv_shared and not is_prefill) or use_shared_prefill_kv
        qkv_start_end = _timing_record_start(do_prefill_stage_timing)
        fused_qkv = None if skip_kv_projection else self._gemma4_fused_qkv(
            hidden_states,
            is_prefill=is_prefill,
            graph_safe_prefill=graph_safe_prefill,
        )
        if fused_qkv is not None:
            q_raw, k_raw, v_raw = fused_qkv
        else:
            q_raw = self._gemma4_linear(
                self.q_proj,
                hidden_states,
                is_prefill=is_prefill,
                decode_attr="_fast_q_out",
                prefill_attr="_prefill_q_out",
            )
            k_raw = None
            v_raw = None
            if not skip_kv_projection:
                k_raw = self._gemma4_linear(
                    self.k_proj,
                    hidden_states,
                    is_prefill=is_prefill,
                    decode_attr="_fast_k_out",
                    prefill_attr="_prefill_k_out",
                )
                if self.v_proj is None:
                    v_raw = k_raw
                else:
                    v_raw = self._gemma4_linear(
                        self.v_proj,
                        hidden_states,
                        is_prefill=is_prefill,
                        decode_attr="_fast_v_out",
                        prefill_attr="_prefill_v_out",
                    )
        _timing_record_end(timing_events, "qkv", qkv_start_end)
        if prefill_timing_suffix is not None:
            timing_events.setdefault(f"qkv_{prefill_timing_suffix}", []).append(
                qkv_start_end
            )
        if finite_trace is not None:
            _record_gemma4_prefill_finite_trace(
                finite_trace, self.layer_idx, "attention.q_raw", q_raw
            )
            _record_gemma4_prefill_finite_trace(
                finite_trace, self.layer_idx, "attention.k_raw", k_raw
            )
            _record_gemma4_prefill_finite_trace(
                finite_trace, self.layer_idx, "attention.v_raw", v_raw
            )

        attn_prepare_start_end = _timing_record_start(do_prefill_stage_timing)
        fused_prepared = None
        if is_prefill and not self.is_kv_shared:
            fused_prepared = self._gemma4_fused_attention_prepare(
                q_raw,
                k_raw,
                v_raw,
                cos,
                sin,
                positions,
                graph_safe_prefill=graph_safe_prefill,
            )
        if fused_prepared is not None:
            q, k, v, k_cache, v_cache = fused_prepared
            if prefill_kv_out is not None:
                persistent_k, persistent_v = prefill_kv_out
                if (
                    tuple(persistent_k.shape) != tuple(k_cache.shape)
                    or tuple(persistent_v.shape) != tuple(v_cache.shape)
                ):
                    raise RuntimeError(
                        "Gemma4 persistent prefill K/V output shape mismatch"
                    )
                persistent_k.copy_(k_cache)
                persistent_v.copy_(v_cache)
                k_cache = persistent_k
                v_cache = persistent_v
        else:
            q = q_raw
            q = q.view(bsz, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2).contiguous()
            if self.q_norm is not None:
                q = self.q_norm(q)
            q, _ = apply_rotary_emb(
                q,
                q,
                cos,
                sin,
                position_ids=positions,
                half_rotate=self.rope_half_rotate,
                rotary_dim=self.rotary_dim,
            )

            k_cache = None
            v_cache = None
            if use_shared_prefill_kv:
                k = shared_prefill_kv[0].transpose(1, 2)
                v = shared_prefill_kv[1].transpose(1, 2)
            else:
                if self.is_kv_shared and not is_prefill:
                    if kv_cache is None or block_table is None or seq_lens is None:
                        raise RuntimeError("Gemma4 shared decode requires source kv_cache")
                if self.is_kv_shared and not is_prefill:
                    k = None
                    v = None
                else:
                    k = k_raw
                    v = v_raw
                    if k is None or v is None:
                        raise RuntimeError("Gemma4 KV projection was unexpectedly skipped")
                    k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
                    v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
                    if self.k_norm is not None:
                        k = self.k_norm(k)
                    if self.v_norm is not None:
                        v = self.v_norm(v)
                    k, _ = apply_rotary_emb(
                        k,
                        k,
                        cos,
                        sin,
                        position_ids=positions,
                        half_rotate=self.rope_half_rotate,
                        rotary_dim=self.rotary_dim,
                    )
                    if not self.is_kv_shared:
                        if prefill_kv_out is None:
                            k_cache = k.transpose(1, 2).contiguous()
                            v_cache = v.transpose(1, 2).contiguous()
                        else:
                            k_cache, v_cache = prefill_kv_out
                            expected_shape = (
                                bsz,
                                seq_len,
                                self.num_kv_heads,
                                self.head_dim,
                            )
                            if (
                                tuple(k_cache.shape) != expected_shape
                                or tuple(v_cache.shape) != expected_shape
                            ):
                                raise RuntimeError(
                                    "Gemma4 persistent prefill K/V output "
                                    "shape mismatch"
                                )
                            k_cache.copy_(k.transpose(1, 2))
                            v_cache.copy_(v.transpose(1, 2))
                    elif is_prefill:
                        k_cache = None
                        v_cache = None
            if not self.is_kv_shared and k_cache is None:
                k = k_raw
                v = v_raw
                if k is None or v is None:
                    raise RuntimeError("Gemma4 KV projection was unexpectedly skipped")
                k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
                v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
                if self.k_norm is not None:
                    k = self.k_norm(k)
                if self.v_norm is not None:
                    v = self.v_norm(v)
                k, _ = apply_rotary_emb(
                    k,
                    k,
                    cos,
                    sin,
                    position_ids=positions,
                    half_rotate=self.rope_half_rotate,
                    rotary_dim=self.rotary_dim,
                )
                if prefill_kv_out is None:
                    k_cache = k.transpose(1, 2).contiguous()
                    v_cache = v.transpose(1, 2).contiguous()
                else:
                    k_cache, v_cache = prefill_kv_out
                    expected_shape = (
                        bsz,
                        seq_len,
                        self.num_kv_heads,
                        self.head_dim,
                    )
                    if (
                        tuple(k_cache.shape) != expected_shape
                        or tuple(v_cache.shape) != expected_shape
                    ):
                        raise RuntimeError(
                            "Gemma4 persistent prefill K/V output shape mismatch"
                        )
                    k_cache.copy_(k.transpose(1, 2))
                    v_cache.copy_(v.transpose(1, 2))

        _timing_record_end(timing_events, "attn_prepare", attn_prepare_start_end)
        if prefill_timing_suffix is not None:
            timing_events.setdefault(
                f"attn_prepare_{prefill_timing_suffix}", []
            ).append(attn_prepare_start_end)
        if finite_trace is not None:
            _record_gemma4_prefill_finite_trace(
                finite_trace, self.layer_idx, "attention.q_prepared", q
            )
            _record_gemma4_prefill_finite_trace(
                finite_trace, self.layer_idx, "attention.k_prepared", k
            )
            _record_gemma4_prefill_finite_trace(
                finite_trace, self.layer_idx, "attention.v_prepared", v
            )
            _record_gemma4_prefill_finite_trace(
                finite_trace, self.layer_idx, "attention.k_cache", k_cache
            )
            _record_gemma4_prefill_finite_trace(
                finite_trace, self.layer_idx, "attention.v_cache", v_cache
            )
        attn_core_start_end = _timing_record_start(do_prefill_stage_timing)
        if is_prefill:
            attn_out = self._gemma4_prefill_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                implicit_causal=implicit_causal_prefill,
            )
        else:
            if not self.is_kv_shared:
                if kv_cache is None or block_table is None or seq_lens is None:
                    raise RuntimeError("Gemma4 decode requires kv_cache, block_table, seq_lens")
                if decode_phys_blocks is None or decode_blk_offsets is None:
                    block_size = kv_cache.shape[3]
                    blk_ids = seq_lens.long() // block_size
                    decode_blk_offsets = seq_lens.long() % block_size
                    decode_phys_blocks = block_table[
                        torch.arange(bsz, device=block_table.device), blk_ids
                    ]
                kv_cache[decode_phys_blocks, 0, :, decode_blk_offsets, :] = k_cache[:, 0]
                kv_cache[decode_phys_blocks, 1, :, decode_blk_offsets, :] = v_cache[:, 0]

            q_decode = q.squeeze(2)
            decode_seq_lens = seq_lens_kv if seq_lens_kv is not None else (seq_lens + 1)
            attn_out = paged_attention_decode(
                q_decode,
                kv_cache,
                block_table,
                decode_seq_lens,
                self.scale,
                out=None,
                sliding_window=self.sliding_window if self.sliding_window > 0 else None,
            ).unsqueeze(2)
        _timing_record_end(timing_events, "attn_core", attn_core_start_end)
        if prefill_timing_suffix is not None:
            timing_events.setdefault(
                f"attn_core_{prefill_timing_suffix}", []
            ).append(attn_core_start_end)
        if finite_trace is not None:
            _record_gemma4_prefill_finite_trace(
                finite_trace, self.layer_idx, "attention.core_out", attn_out
            )

        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        o_proj_start_end = _timing_record_start(do_prefill_stage_timing)
        output = self._gemma4_linear(
            self.o_proj,
            attn_out,
            is_prefill=is_prefill,
            decode_attr="_fast_o_out",
            prefill_attr="_prefill_o_out",
        )
        _timing_record_end(timing_events, "o_proj", o_proj_start_end)
        if prefill_timing_suffix is not None:
            timing_events.setdefault(f"o_proj_{prefill_timing_suffix}", []).append(
                o_proj_start_end
            )
        if finite_trace is not None:
            _record_gemma4_prefill_finite_trace(
                finite_trace, self.layer_idx, "attention.o_proj", output
            )
        return output, k_cache, v_cache

    def forward(
        self,
        hidden_states: torch.Tensor,   # [batch, seq_len, hidden]
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        seq_lens_kv: Optional[torch.Tensor] = None,
        decode_phys_blocks: Optional[torch.Tensor] = None,
        decode_blk_offsets: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        append_kv_prefix: bool = False,
        input_is_normed: bool = True,
        input_norm_weight: Optional[torch.Tensor] = None,
        input_norm_eps: float = 1e-6,
        input_norm_offset: bool = False,
        timing_events: Optional[dict] = None,
        implicit_causal_prefill: bool = False,
        graph_safe_prefill: bool = False,
        prefill_kv_out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: (output, k_for_cache, v_for_cache)"""
        if self.is_gemma4:
            return self._gemma4_attention(
                hidden_states,
                cos,
                sin,
                positions,
                kv_cache=kv_cache,
                block_table=block_table,
                seq_lens=seq_lens,
                seq_lens_kv=seq_lens_kv,
                decode_phys_blocks=decode_phys_blocks,
                decode_blk_offsets=decode_blk_offsets,
                is_prefill=is_prefill,
                attn_mask=attn_mask,
                shared_prefill_kv=getattr(self, "_prefill_shared_kv", None),
                timing_events=timing_events,
                implicit_causal_prefill=implicit_causal_prefill,
                graph_safe_prefill=graph_safe_prefill,
                prefill_kv_out=prefill_kv_out,
            )
        bsz, seq_len, _ = hidden_states.shape
        do_prefill_stage_timing = (
            timing_events is not None
            and is_prefill
            and hidden_states.is_cuda
            and _prefill_timing_enabled()
        )

        qkv_start_end = _timing_record_start(do_prefill_stage_timing)
        if self._awq_separate:
            if not input_is_normed and input_norm_weight is not None:
                hidden_states = _decode_rmsnorm(
                    hidden_states,
                    input_norm_weight,
                    input_norm_eps,
                    input_norm_offset,
                )
            # AWQ path: 3 separate quantized matmuls
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        else:
            if not input_is_normed and input_norm_weight is not None:
                if is_prefill:
                    qkv = self._prefill_qkv_from_raw_hidden(
                        hidden_states,
                        input_norm_weight,
                        input_norm_eps,
                        input_norm_offset,
                    )
                else:
                    qkv = self._decode_qkv_from_raw_hidden(
                        hidden_states,
                        input_norm_weight,
                        input_norm_eps,
                        input_norm_offset,
                    )
            else:
                # FP16 path: fused QKV → single matmul + split (3x less kernel launches)
                qkv_weight, qkv_bias = _linear_weight_bias(self.qkv_proj)
                if not is_prefill and _can_use_fast_gemv_for(
                    "qkv",
                    hidden_states,
                    out_features=self._q_proj_size + self._k_size + self._v_size,
                ) and qkv_weight is not None:
                    qkv_out = _get_reusable_out(
                        self,
                        "_fast_qkv_out",
                        (*hidden_states.shape[:-1], self._q_proj_size + self._k_size + self._v_size),
                        hidden_states,
                    )
                    qkv = fast_linear(hidden_states, qkv_weight, qkv_bias, out=qkv_out)
                else:
                    qkv = self.qkv_proj(hidden_states)
            q = qkv[..., :self._q_proj_size]
            k = qkv[..., self._q_proj_size:self._q_proj_size + self._k_size]
            v = qkv[..., self._q_proj_size + self._k_size:]
        _timing_record_end(timing_events, "qkv", qkv_start_end)

        q_gate = None
        if self.attention_output_gate:
            q = q.view(bsz, seq_len, self.num_q_heads, 2 * self.head_dim)
            q_gate = q[..., self.head_dim:]
            q = q[..., :self.head_dim]
        else:
            q = q.view(bsz, seq_len, self.num_q_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if q_gate is not None:
            q_gate = q_gate.reshape(bsz, seq_len, self.num_q_heads * self.head_dim)

        # === Fused decode path: QK-Norm + RoPE + KV write + attention ===
        attn_core_start_end = _timing_record_start(do_prefill_stage_timing)
        can_use_fused_decode = (
            not is_prefill
            and _HAS_FUSED_ROPE_ATTN
            and not self._disable_fused_decode
            and q.is_cuda
            and kv_cache is not None
            and block_table is not None
            and seq_lens is not None
        )
        fused_ok = False
        if can_use_fused_decode:
            try:
                # QK-Norm is fused INTO the kernels.
                k_for_kv = k.transpose(1, 2)
                v_for_kv = v.transpose(1, 2)

                if decode_phys_blocks is not None and decode_blk_offsets is not None:
                    phys = decode_phys_blocks
                    blk_offs = decode_blk_offsets
                else:
                    block_size = kv_cache.shape[3]
                    blk_ids = seq_lens.long() // block_size
                    blk_offs = seq_lens.long() % block_size
                    phys = block_table[torch.arange(bsz, device=block_table.device), blk_ids]

                q_nw = self.q_norm.weight if self.q_norm is not None else None
                k_nw = self.k_norm.weight if self.k_norm is not None else None
                neps = self.q_norm.eps if self.q_norm is not None else 1e-6
                pos_1d = positions.squeeze(-1) if positions.dim() > 1 else positions

                wrote = fused_rope_kv_write(
                    k_for_kv[:, 0], v_for_kv[:, 0],
                    kv_cache, cos, sin, pos_1d,
                    phys, blk_offs,
                    half_rotate=self.rope_half_rotate,
                    rotary_dim=self.rotary_dim,
                    k_norm_weight=k_nw, norm_eps=neps,
                )
                if wrote:
                    q_decode = q.squeeze(2)
                    attn_buf = _get_reusable_out(
                        self,
                        "_fast_attn_out",
                        (bsz, self.num_q_heads, self.head_dim),
                        q_decode,
                    )
                    decode_seq_lens = seq_lens_kv if seq_lens_kv is not None else (seq_lens + 1)
                    attn_out = _triton_paged_decode_fused(
                        q_decode, kv_cache, block_table, decode_seq_lens, self.scale,
                        cos, sin, pos_1d,
                        half_rotate=self.rope_half_rotate,
                        rotary_dim=self.rotary_dim,
                        q_norm_weight=q_nw, norm_eps=neps,
                        out=attn_buf,
                    )
                    attn_out = attn_out.unsqueeze(2)
                    k_cache = k_for_kv
                    v_cache = v_for_kv
                    fused_ok = True
                    self._fused_decode_hits += 1
                else:
                    self._disable_fused_decode = True
                    if self._fused_decode_disable_reason is None:
                        self._fused_decode_disable_reason = "fused_rope_kv_write unavailable"
                    if _DEBUG_FUSED_ROPE_ATTN:
                        print(
                            f"[MegaGemm][layer={self.layer_idx}] fused decode disabled: "
                            f"{self._fused_decode_disable_reason}"
                        )
            except Exception:
                self._disable_fused_decode = True
                if self._fused_decode_disable_reason is None:
                    import traceback
                    self._fused_decode_disable_reason = traceback.format_exc(limit=1).strip()
                if _DEBUG_FUSED_ROPE_ATTN:
                    print(
                        f"[MegaGemm][layer={self.layer_idx}] fused decode disabled: "
                        f"{self._fused_decode_disable_reason}"
                    )

        if not fused_ok:
            # === Standard path (prefill or no fused kernel) ===
            # Apply QK-Norm separately
            if self.q_norm is not None:
                q = self.q_norm(q)
                k = self.k_norm(k)

            # Apply RoPE
            pos_ids = positions
            q, k = apply_rotary_emb(
                q, k, cos, sin,
                position_ids=pos_ids,
                half_rotate=self.rope_half_rotate,
                rotary_dim=self.rotary_dim,
            )

            if is_prefill:
                k_cache = k.transpose(1, 2)
                if cu_seqlens is None:
                    k_cache = k_cache.contiguous()
                v_cache = v.transpose(1, 2)
                if cu_seqlens is None:
                    v_cache = v_cache.contiguous()
            else:
                k_cache = k.transpose(1, 2)
                v_cache = v.transpose(1, 2)

            if append_kv_prefix and cu_seqlens is not None:
                raise ValueError("append_kv_prefix does not support packed cu_seqlens prefill")

            if is_prefill and append_kv_prefix:
                if kv_cache is None or block_table is None or seq_lens is None:
                    raise ValueError(
                        "append_kv_prefix prefill requires kv_cache, block_table, and seq_lens"
                    )
                attn_out = self._prefill_attention_with_prefix_cache(
                    q,
                    k,
                    v,
                    kv_cache,
                    block_table,
                    seq_lens,
                    attn_mask=attn_mask,
                )
            elif cu_seqlens is not None and is_prefill:
                # Packed attention: dispatch to best available backend
                # (flash_attn varlen > SDPA per-seq > Triton)
                total_t = q.shape[2]
                q_packed = q[0].permute(1, 0, 2)  # [T, H, D]
                k_packed = k[0].permute(1, 0, 2)
                v_packed = v[0].permute(1, 0, 2)
                out_packed = packed_prefill_attention(q_packed, k_packed, v_packed, cu_seqlens)
                attn_out = out_packed.permute(1, 0, 2).unsqueeze(0)  # [1, H, T, D]
            elif is_prefill:
                attn_out = prefill_attention(q, k, v, is_causal=True, attn_mask=attn_mask)
            else:
                # Fallback decode (no fused kernel)
                if decode_phys_blocks is not None and decode_blk_offsets is not None:
                    phys = decode_phys_blocks
                    blk_offs = decode_blk_offsets
                else:
                    block_size = kv_cache.shape[3]
                    blk_ids = seq_lens.long() // block_size
                    blk_offs = seq_lens.long() % block_size
                    phys = block_table[torch.arange(bsz, device=block_table.device), blk_ids]
                kv_cache[phys, 0, :, blk_offs, :] = k_cache[:, 0]
                kv_cache[phys, 1, :, blk_offs, :] = v_cache[:, 0]

                q_decode = q.squeeze(2)
                attn_buf = _get_reusable_out(
                    self,
                    "_fast_attn_out",
                    (bsz, self.num_q_heads, self.head_dim),
                    q_decode,
                )
                decode_seq_lens = seq_lens_kv if seq_lens_kv is not None else (seq_lens + 1)
                attn_out = paged_attention_decode(
                    q_decode, kv_cache, block_table, decode_seq_lens, self.scale,
                    out=attn_buf,
                )
                attn_out = attn_out.unsqueeze(2)

        _timing_record_end(timing_events, "attn_core", attn_core_start_end)

        # Reshape and project output.
        # Decode seq_len=1 can avoid transpose metadata churn.
        if not is_prefill and seq_len == 1:
            attn_out = attn_out.squeeze(2).reshape(bsz, 1, -1)
        else:
            attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        if q_gate is not None:
            attn_out.mul_(torch.sigmoid(q_gate))
        o_proj_start_end = _timing_record_start(do_prefill_stage_timing)
        o_weight, o_bias = _linear_weight_bias(self.o_proj)
        if not is_prefill and _can_use_fast_gemv_for(
            "o_proj",
            attn_out,
            out_features=self.config.hidden_size,
        ) and o_weight is not None:
            o_out = _get_reusable_out(
                self, "_fast_o_out",
                (*attn_out.shape[:-1], self.config.hidden_size),
                attn_out,
            )
            output = fast_linear(attn_out, o_weight, o_bias, out=o_out)
        else:
            output = self.o_proj(attn_out)
        _timing_record_end(timing_events, "o_proj", o_proj_start_end)

        return output, k_cache, v_cache

    def _decode_qkv_linear(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv_weight, qkv_bias = _linear_weight_bias(self.qkv_proj)
        if qkv_weight is None:
            return self.qkv_proj(hidden_states)

        if _USE_DECODE_FAST_LINEAR:
            return _decode_linear(
                self,
                "_fast_qkv_out",
                hidden_states,
                qkv_weight,
                qkv_bias,
                use_fast=True,
            )
        sig = _decode_linear_runtime_sig(hidden_states, qkv_weight)
        if self._fast_qkv_decode_key != sig:
            self._fast_qkv_decode_use = bool(
                _pick_decode_linear_backend(
                    self,
                    "qkv",
                    hidden_states,
                    qkv_weight,
                    qkv_bias,
                    "_fast_qkv_out",
                )
            )
            self._fast_qkv_decode_key = sig
        if self._fast_qkv_decode_use:
            return _decode_linear(
                self,
                "_fast_qkv_out",
                hidden_states,
                qkv_weight,
                qkv_bias,
                use_fast=True,
            )
        return _decode_linear(
            self,
            "_fast_qkv_out",
            hidden_states,
            qkv_weight,
            qkv_bias,
            use_fast=False,
        )

    def _decode_qkv_from_raw_hidden(
        self,
        hidden_states: torch.Tensor,
        input_norm_weight: torch.Tensor,
        input_norm_eps: float,
        input_norm_offset: bool,
    ) -> torch.Tensor:
        qkv_weight, qkv_bias = _linear_weight_bias(self.qkv_proj)
        if qkv_weight is None:
            normed = _decode_rmsnorm(
                hidden_states,
                input_norm_weight,
                input_norm_eps,
                input_norm_offset,
            )
            return self.qkv_proj(normed)

        out_features = self._q_proj_size + self._k_size + self._v_size
        if (
            not self._disable_fused_rmsnorm_qkv_decode
            and _cached_fused_rmsnorm_qkv_decision(
                self,
                "_fused_rmsnorm_qkv_decode_key",
                "_fused_rmsnorm_qkv_decode_use",
                hidden_states,
                out_features=out_features,
            )
        ):
            if not self._fused_rmsnorm_qkv_checked:
                try:
                    qkv_out = _get_reusable_out_decode(
                        self,
                        "_fast_qkv_out",
                        (*hidden_states.shape[:-1], out_features),
                        hidden_states,
                    )
                    fused_rmsnorm_linear(
                        hidden_states,
                        input_norm_weight,
                        input_norm_eps,
                        qkv_weight,
                        qkv_bias,
                        norm_offset=input_norm_offset,
                        out=qkv_out,
                    )
                    baseline_normed = _decode_rmsnorm(
                        hidden_states,
                        input_norm_weight,
                        input_norm_eps,
                        input_norm_offset,
                    )
                    self._decode_qkv_linear(baseline_normed)
                    torch.cuda.synchronize()
                    fused_ms = _cuda_bench_ms(
                        lambda: fused_rmsnorm_linear(
                            hidden_states,
                            input_norm_weight,
                            input_norm_eps,
                            qkv_weight,
                            qkv_bias,
                            norm_offset=input_norm_offset,
                            out=qkv_out,
                        ),
                        iters=8,
                    )
                    base_ms = _cuda_bench_ms(
                        lambda: self._decode_qkv_linear(
                            _decode_rmsnorm(
                                hidden_states,
                                input_norm_weight,
                                input_norm_eps,
                                input_norm_offset,
                            )
                        ),
                        iters=8,
                    )
                    self._fused_rmsnorm_qkv_decode_use = bool(
                        fused_ms <= (base_ms * (1.0 - _FUSED_RMSNORM_QKV_MIN_GAIN))
                    )
                    self._fused_rmsnorm_qkv_checked = True
                except Exception:
                    self._disable_fused_rmsnorm_qkv_decode = True
                    self._fused_rmsnorm_qkv_decode_use = False

            if self._fused_rmsnorm_qkv_decode_use and not self._disable_fused_rmsnorm_qkv_decode:
                try:
                    qkv_out = _get_reusable_out_decode(
                        self,
                        "_fast_qkv_out",
                        (*hidden_states.shape[:-1], out_features),
                        hidden_states,
                    )
                    return fused_rmsnorm_linear(
                        hidden_states,
                        input_norm_weight,
                        input_norm_eps,
                        qkv_weight,
                        qkv_bias,
                        norm_offset=input_norm_offset,
                        out=qkv_out,
                    )
                except Exception:
                    self._disable_fused_rmsnorm_qkv_decode = True
                    self._fused_rmsnorm_qkv_decode_use = False

        normed = _decode_rmsnorm(
            hidden_states,
            input_norm_weight,
            input_norm_eps,
            input_norm_offset,
        )
        return self._decode_qkv_linear(normed)

    def _prefill_qkv_from_raw_hidden(
        self,
        hidden_states: torch.Tensor,
        input_norm_weight: torch.Tensor,
        input_norm_eps: float,
        input_norm_offset: bool,
    ) -> torch.Tensor:
        qkv_weight, qkv_bias = _linear_weight_bias(self.qkv_proj)
        if qkv_weight is None:
            normed = _decode_rmsnorm(
                hidden_states,
                input_norm_weight,
                input_norm_eps,
                input_norm_offset,
            )
            return self.qkv_proj(normed)

        out_features = self._q_proj_size + self._k_size + self._v_size
        if (
            _USE_FUSED_RMSNORM_QKV_PREFILL
            and not self._disable_fused_rmsnorm_qkv_prefill
            and hidden_states.is_cuda
            and not torch.is_grad_enabled()
        ):
            try:
                qkv_out = _get_reusable_out(
                    self,
                    "_prefill_qkv_out",
                    (*hidden_states.shape[:-1], out_features),
                    hidden_states,
                )
                return fused_rmsnorm_linear(
                    hidden_states,
                    input_norm_weight,
                    input_norm_eps,
                    qkv_weight,
                    qkv_bias,
                    norm_offset=input_norm_offset,
                    out=qkv_out,
                    mode="prefill",
                )
            except Exception:
                self._disable_fused_rmsnorm_qkv_prefill = True

        normed = _decode_rmsnorm(
            hidden_states,
            input_norm_weight,
            input_norm_eps,
            input_norm_offset,
        )
        return self.qkv_proj(normed)

    def _run_fused_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_kv: Optional[torch.Tensor],
        decode_phys_blocks: Optional[torch.Tensor],
        decode_blk_offsets: Optional[torch.Tensor],
    ):
        if decode_phys_blocks is None or decode_blk_offsets is None:
            block_size = kv_cache.shape[3]
            blk_ids = seq_lens.long() // block_size
            decode_blk_offsets = seq_lens.long() % block_size
            decode_phys_blocks = block_table[torch.arange(q.shape[0], device=block_table.device), blk_ids]

        k_for_kv = k.transpose(1, 2)
        v_for_kv = v.transpose(1, 2)
        q_nw = self.q_norm.weight if self.q_norm is not None else None
        k_nw = self.k_norm.weight if self.k_norm is not None else None
        neps = self.q_norm.eps if self.q_norm is not None else 1e-6
        pos_1d = positions.squeeze(-1) if positions.dim() > 1 else positions

        wrote = fused_rope_kv_write(
            k_for_kv[:, 0],
            v_for_kv[:, 0],
            kv_cache,
            cos,
            sin,
            pos_1d,
            decode_phys_blocks,
            decode_blk_offsets,
            half_rotate=self.rope_half_rotate,
            rotary_dim=self.rotary_dim,
            k_norm_weight=k_nw,
            norm_eps=neps,
        )
        if not wrote:
            return None

        q_decode = q.squeeze(2)
        attn_buf = _get_reusable_out_decode(
            self,
            "_fast_attn_out",
            (q.shape[0], self.num_q_heads, self.head_dim),
            q_decode,
        )
        decode_seq_lens = seq_lens_kv if seq_lens_kv is not None else (seq_lens + 1)
        attn_out = _triton_paged_decode_fused(
            q_decode,
            kv_cache,
            block_table,
            decode_seq_lens,
            self.scale,
            cos,
            sin,
            pos_1d,
            half_rotate=self.rope_half_rotate,
            rotary_dim=self.rotary_dim,
            q_norm_weight=q_nw,
            norm_eps=neps,
            out=attn_buf,
        )
        return attn_out.unsqueeze(2), k_for_kv, v_for_kv

    def forward_decode(
        self,
        hidden_states: torch.Tensor,   # [batch, 1, hidden]
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        seq_lens_kv: Optional[torch.Tensor] = None,
        decode_phys_blocks: Optional[torch.Tensor] = None,
        decode_blk_offsets: Optional[torch.Tensor] = None,
        input_is_normed: bool = True,
        input_norm_weight: Optional[torch.Tensor] = None,
        input_norm_eps: float = 1e-6,
        input_norm_offset: bool = False,
        timing_events: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_gemma4:
            return self._gemma4_attention(
                hidden_states,
                cos,
                sin,
                positions,
                kv_cache=kv_cache,
                block_table=block_table,
                seq_lens=seq_lens,
                seq_lens_kv=seq_lens_kv,
                decode_phys_blocks=decode_phys_blocks,
                decode_blk_offsets=decode_blk_offsets,
                is_prefill=False,
                shared_prefill_kv=None,
            )
        bsz, _, _ = hidden_states.shape
        do_timing = (
            timing_events is not None
            and hidden_states.is_cuda
            and _timing_enabled()
        )

        qkv_start_end = _timing_record_start(do_timing)
        if self._awq_separate:
            if not input_is_normed and input_norm_weight is not None:
                hidden_states = _decode_rmsnorm(
                    hidden_states,
                    input_norm_weight,
                    input_norm_eps,
                    input_norm_offset,
                )
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        else:
            if not input_is_normed and input_norm_weight is not None:
                qkv = self._decode_qkv_from_raw_hidden(
                    hidden_states,
                    input_norm_weight,
                    input_norm_eps,
                    input_norm_offset,
                )
            else:
                qkv = self._decode_qkv_linear(hidden_states)
            q = qkv[..., :self._q_proj_size]
            k = qkv[..., self._q_proj_size:self._q_proj_size + self._k_size]
            v = qkv[..., self._q_proj_size + self._k_size:]
        _timing_record_end(timing_events, "attn_qkv", qkv_start_end)

        q_gate = None
        if self.attention_output_gate:
            q = q.view(bsz, 1, self.num_q_heads, 2 * self.head_dim)
            q_gate = q[..., self.head_dim:]
            q = q[..., :self.head_dim]
        else:
            q = q.view(bsz, 1, self.num_q_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.view(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if q_gate is not None:
            q_gate = q_gate.reshape(bsz, 1, self.num_q_heads * self.head_dim)

        fused_out = None
        can_attempt_fused = (
            _HAS_FUSED_ROPE_ATTN
            and not self._disable_fused_decode
            and kv_cache is not None
            and block_table is not None
            and seq_lens is not None
        )
        if can_attempt_fused:
            if self._fused_decode_checked:
                attn_core_start_end = _timing_record_start(do_timing)
                fused_out = self._run_fused_decode(
                    q, k, v, cos, sin, positions, kv_cache, block_table, seq_lens,
                    seq_lens_kv, decode_phys_blocks, decode_blk_offsets,
                )
                _timing_record_end(timing_events, "attn_core", attn_core_start_end)
                if (
                    timing_events is not None
                    and attn_core_start_end is not None
                    and attn_core_start_end[0] is not None
                ):
                    timing_events.setdefault("attn_core_full", []).append(attn_core_start_end)
                if fused_out is None:
                    self._disable_fused_decode = True
                    if self._fused_decode_disable_reason is None:
                        self._fused_decode_disable_reason = "fused_rope_kv_write unavailable"
            else:
                try:
                    attn_core_start_end = _timing_record_start(do_timing)
                    fused_out = self._run_fused_decode(
                        q, k, v, cos, sin, positions, kv_cache, block_table, seq_lens,
                        seq_lens_kv, decode_phys_blocks, decode_blk_offsets,
                    )
                    _timing_record_end(timing_events, "attn_core", attn_core_start_end)
                    if (
                        timing_events is not None
                        and attn_core_start_end is not None
                        and attn_core_start_end[0] is not None
                    ):
                        timing_events.setdefault("attn_core_full", []).append(attn_core_start_end)
                    if fused_out is None:
                        self._disable_fused_decode = True
                        if self._fused_decode_disable_reason is None:
                            self._fused_decode_disable_reason = "fused_rope_kv_write unavailable"
                    else:
                        self._fused_decode_checked = True
                except Exception:
                    self._disable_fused_decode = True
                    if self._fused_decode_disable_reason is None:
                        import traceback
                        self._fused_decode_disable_reason = traceback.format_exc(limit=1).strip()
                    if _DEBUG_FUSED_ROPE_ATTN:
                        print(
                            f"[MegaGemm][layer={self.layer_idx}] fused decode disabled: "
                            f"{self._fused_decode_disable_reason}"
                        )

        if fused_out is not None:
            attn_out, k_cache, v_cache = fused_out
            self._fused_decode_hits += 1
        else:
            attn_norm_rope_start_end = _timing_record_start(do_timing)
            if self.q_norm is not None:
                q = self.q_norm(q)
                k = self.k_norm(k)

            q, k = apply_rotary_emb(
                q,
                k,
                cos,
                sin,
                position_ids=positions,
                half_rotate=self.rope_half_rotate,
                rotary_dim=self.rotary_dim,
            )
            _timing_record_end(timing_events, "attn_norm_rope", attn_norm_rope_start_end)

            k_cache = k.transpose(1, 2)
            v_cache = v.transpose(1, 2)
            if decode_phys_blocks is None or decode_blk_offsets is None:
                if kv_cache is None or block_table is None or seq_lens is None:
                    raise RuntimeError("decode requires kv_cache, block_table, seq_lens")
                block_size = kv_cache.shape[3]
                blk_ids = seq_lens.long() // block_size
                decode_blk_offsets = seq_lens.long() % block_size
                decode_phys_blocks = block_table[torch.arange(bsz, device=block_table.device), blk_ids]

            attn_kv_write_start_end = _timing_record_start(do_timing)
            kv_cache[decode_phys_blocks, 0, :, decode_blk_offsets, :] = k_cache[:, 0]
            kv_cache[decode_phys_blocks, 1, :, decode_blk_offsets, :] = v_cache[:, 0]
            _timing_record_end(timing_events, "attn_kv_write", attn_kv_write_start_end)

            q_decode = q.squeeze(2)
            attn_buf = _get_reusable_out_decode(
                self,
                "_fast_attn_out",
                (bsz, self.num_q_heads, self.head_dim),
                q_decode,
            )
            decode_seq_lens = seq_lens_kv if seq_lens_kv is not None else (seq_lens + 1)
            attn_core_start_end = _timing_record_start(do_timing)
            attn_out = paged_attention_decode(
                q_decode, kv_cache, block_table, decode_seq_lens, self.scale, out=attn_buf,
            ).unsqueeze(2)
            _timing_record_end(timing_events, "attn_core", attn_core_start_end)
            if (
                timing_events is not None
                and attn_core_start_end is not None
                and attn_core_start_end[0] is not None
            ):
                timing_events.setdefault("attn_core_full", []).append(attn_core_start_end)

        attn_out = attn_out.squeeze(2).reshape(bsz, 1, -1)
        if q_gate is not None:
            attn_out.mul_(torch.sigmoid(q_gate))

        attn_o_proj_start_end = _timing_record_start(do_timing)
        o_weight, o_bias = _linear_weight_bias(self.o_proj)
        if o_weight is None:
            output = self.o_proj(attn_out)
            _timing_record_end(timing_events, "attn_o_proj", attn_o_proj_start_end)
            return output, k_cache, v_cache

        if _USE_DECODE_FAST_LINEAR:
            output = _decode_linear(
                self,
                "_fast_o_out",
                attn_out,
                o_weight,
                o_bias,
                use_fast=True,
            )
        elif _pick_decode_linear_backend(
            self,
            "o_proj",
            attn_out,
            o_weight,
            o_bias,
            "_fast_o_out",
        ):
            output = _decode_linear(
                self,
                "_fast_o_out",
                attn_out,
                o_weight,
                o_bias,
                use_fast=True,
            )
        else:
            output = _decode_linear(
                self,
                "_fast_o_out",
                attn_out,
                o_weight,
                o_bias,
                use_fast=False,
            )

        _timing_record_end(timing_events, "attn_o_proj", attn_o_proj_start_end)
        return output, k_cache, v_cache


class LlamaMLP(nn.Module):
    """SwiGLU/GeGLU FFN. Supports fused gate_up_proj (FP16) and separate projections (AWQ)."""

    def __init__(self, config: LlamaConfig, layer_idx: int = 0):
        super().__init__()
        self.intermediate_size = (
            config.mlp_intermediate_sizes[layer_idx]
            if config.mlp_intermediate_sizes and layer_idx < len(config.mlp_intermediate_sizes)
            else config.intermediate_size
        )
        self.hidden_act = config.hidden_act
        self._awq_separate = False  # Set True by loader when using AWQ

        # Fused gate + up projection (2x intermediate) — used for FP16
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, config.hidden_size, bias=False
        )

        # AWQ: separate gate/up projections will be set by loader
        # self.gate_proj = QuantizedLinear(...)  # set by loader
        # self.up_proj = QuantizedLinear(...)    # set by loader
        self._fast_gate_up_out = None
        self._fast_down_out = None
        self._fast_gate_up_decode_key = None
        self._fast_gate_up_decode_use = False
        self._fast_gate_up_mode_key = None
        self._fast_gate_up_mode = "tile"
        self._fast_down_decode_key = None
        self._fast_down_decode_use = False
        self._fused_rmsnorm_gateup_decode_key = None
        self._fused_rmsnorm_gateup_decode_use = False
        self._fused_rmsnorm_gateup_checked = False
        self._disable_fused_rmsnorm_gateup_decode = False
        self._decode_disable_triton_swiglu = False
        self._decode_swiglu_checked = False
        self._deepfusion_out = None
        self._prefill_deepfusion_out = None
        self._prefill_activated_out = None
        self._deepfusion_decode_key = None
        self._deepfusion_decode_use = False
        self._deepfusion_prefill_key = None
        self._deepfusion_prefill_eligible = False
        self._deepfusion_decode_hits = 0
        self._decode_disable_deepfusion = False
        self._decode_deepfusion_checked = False
        self._disable_prefill_deepfusion = False
        self._prefill_deepfusion_use_cache = {}
        self._prefill_deepfusion_bench = {}
        self._disable_native_mlp_prefill = False
        # Shape-specific Gemma 4 E2B/L4 B8 prefill GELU-tanh x value path.
        # A positive scalar block size is an experiment override; production
        # uses the per-sequence map installed by the measured runtime policy.
        self._gemma4_e2b_prefill_gated_activation_enabled = False
        self._gemma4_e2b_prefill_gated_activation_block_size = 0
        self._gemma4_e2b_prefill_gated_activation_block_sizes = {}
        self._gemma4_e2b_prefill_gated_activation_hits = 0
        self._gemma4_e2b_prefill_gated_activation_runtime_disabled = False
        self._gemma4_e2b_prefill_gated_activation_failure = ""
        # Experimental exact-shape cuBLASLt gate-up route.  Algorithm choices
        # are installed only by the one-load full-model frontier; production
        # remains torch.mm until that gate promotes a stable shape map.
        self._gemma4_e2b_prefill_cublaslt_gateup_enabled = False
        self._gemma4_e2b_prefill_cublaslt_gateup_algorithms = {}
        self._gemma4_e2b_prefill_cublaslt_gateup_hits = 0
        self._gemma4_e2b_prefill_cublaslt_gateup_runtime_disabled = False
        self._gemma4_e2b_prefill_cublaslt_gateup_failure = ""

    def _activation(self, gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Apply gated activation: SiLU (LLaMA/Qwen) or GELU (Gemma 2)."""
        if self.hidden_act in ('gelu', 'gelu_pytorch_tanh'):
            activated = torch.nn.functional.gelu(gate, approximate='tanh')
        else:
            activated = torch.nn.functional.silu(gate)
        if torch.is_grad_enabled():
            return activated * value
        return activated.mul_(value)

    def _decode_deepfusion(
        self,
        gate_up: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        down_weight, down_bias = _linear_weight_bias(self.down_proj)
        if down_weight is None:
            activated = self._decode_swiglu_activation(gate_up)
            out = self.down_proj(activated)
            if residual is not None:
                residual.add_(out)
                return residual
            return out

        if residual is not None:
            out = residual
        else:
            out = _get_reusable_out_decode(
                self,
                "_deepfusion_out",
                (*gate_up.shape[:-1], self.down_proj.out_features),
                gate_up,
            )
        return deepfusion_swiglu_down(
            gate_up,
            down_weight,
            down_bias,
            out=out,
            residual=residual,
            activation="gelu_tanh" if self.hidden_act in ('gelu', 'gelu_pytorch_tanh') else "silu",
        )

    def _prefill_deepfusion(self, gate_up: torch.Tensor) -> torch.Tensor:
        down_weight, down_bias = _linear_weight_bias(self.down_proj)
        if down_weight is None:
            return self._prefill_baseline_tail(gate_up)
        out = _get_prefill_out(
            self,
            "_prefill_deepfusion_out",
            (*gate_up.shape[:-1], self.down_proj.out_features),
            gate_up,
        )
        return deepfusion_swiglu_down(
            gate_up,
            down_weight,
            down_bias,
            out=out,
            mode="prefill",
        )

    def _log_prefill_deepfusion(self, message: str) -> None:
        if _DEEPFUSION_PREFILL_DEBUG:
            print(f"DeepFusion prefill: {message}")

    def _prefill_baseline_tail(
        self,
        gate_up: torch.Tensor,
        timing_events: Optional[dict] = None,
        do_prefill_stage_timing: bool = False,
    ) -> torch.Tensor:
        activation_start_end = _timing_record_start(do_prefill_stage_timing)
        if self.hidden_act in ('gelu', 'gelu_pytorch_tanh'):
            sequence_len = int(gate_up.shape[1]) if gate_up.ndim == 3 else 0
            block_size_override = int(
                self._gemma4_e2b_prefill_gated_activation_block_size or 0
            )
            activation_block_size = block_size_override or int(
                self._gemma4_e2b_prefill_gated_activation_block_sizes.get(
                    sequence_len,
                    0,
                )
            )
            use_gated_activation = bool(
                self._gemma4_e2b_prefill_gated_activation_enabled
                and not self._gemma4_e2b_prefill_gated_activation_runtime_disabled
                and callable(gated_activation_forward)
                and gate_up.is_cuda
                and gate_up.dtype == torch.bfloat16
                and gate_up.ndim == 3
                and int(gate_up.shape[0]) == 8
                and int(gate_up.shape[1]) in (521, 2057)
                and activation_block_size in (128, 256, 512, 1024)
                and int(self.down_proj.out_features) == 1536
                and int(self.intermediate_size) in (6144, 12288)
            )
            activated = None
            if use_gated_activation:
                try:
                    activated_out = _get_prefill_out(
                        self,
                        "_prefill_activated_out",
                        (*gate_up.shape[:-1], self.intermediate_size),
                        gate_up,
                    )
                    activated = gated_activation_forward(
                        gate_up,
                        self.intermediate_size,
                        activation="gelu_tanh",
                        out=activated_out,
                        block_size=activation_block_size,
                    )
                except Exception as exc:
                    self._gemma4_e2b_prefill_gated_activation_runtime_disabled = True
                    if not self._gemma4_e2b_prefill_gated_activation_failure:
                        self._gemma4_e2b_prefill_gated_activation_failure = (
                            f"{type(exc).__name__}: {exc}"
                        )
                    activated = None
                else:
                    self._gemma4_e2b_prefill_gated_activation_hits += 1
            if activated is None:
                gate = gate_up[..., :self.intermediate_size]
                value = gate_up[..., self.intermediate_size:]
                activated = self._activation(gate, value)
        elif _HAS_TRITON_SWIGLU:
            try:
                activated_out = None
                if not torch.is_grad_enabled():
                    activated_out = _get_prefill_out(
                        self,
                        "_prefill_activated_out",
                        (*gate_up.shape[:-1], self.intermediate_size),
                        gate_up,
                    )
                if torch.is_grad_enabled():
                    activated = MegaGemmFunction.apply(gate_up, self.intermediate_size)
                else:
                    activated = swiglu_forward(
                        gate_up,
                        self.intermediate_size,
                        out=activated_out,
                    )
            except Exception:
                gate = gate_up[..., :self.intermediate_size]
                value = gate_up[..., self.intermediate_size:]
                activated = self._activation(gate, value)
        else:
            gate = gate_up[..., :self.intermediate_size]
            value = gate_up[..., self.intermediate_size:]
            activated = self._activation(gate, value)
        _timing_record_end(timing_events, "activation", activation_start_end)

        down_weight, down_bias = _linear_weight_bias(self.down_proj)
        down_start_end = _timing_record_start(do_prefill_stage_timing)
        if _can_use_fast_gemv_for(
            "down",
            activated,
            out_features=self.down_proj.out_features,
        ) and down_weight is not None:
            down_out = _get_prefill_out(
                self,
                "_fast_down_out",
                (*activated.shape[:-1], self.down_proj.out_features),
                activated,
            )
            out = fast_linear(activated, down_weight, down_bias, out=down_out)
            _timing_record_end(timing_events, "down_proj", down_start_end)
            return out
        if (
            down_weight is not None
            and activated.is_cuda
            and not torch.is_grad_enabled()
        ):
            out = _prefill_linear(
                self,
                "_fast_down_out",
                activated,
                down_weight,
                down_bias,
            )
            _timing_record_end(timing_events, "down_proj", down_start_end)
            return out
        out = self.down_proj(activated)
        _timing_record_end(timing_events, "down_proj", down_start_end)
        return out

    def _should_use_prefill_deepfusion(
        self,
        gate_up: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> bool:
        if _DEEPFUSION_PREFILL_FORCE_USE:
            return True
        sig = _deepfusion_shape_sig(gate_up, down_weight)
        if sig in self._prefill_deepfusion_use_cache:
            return bool(self._prefill_deepfusion_use_cache[sig])
        try:
            self._prefill_deepfusion(gate_up)
            self._prefill_baseline_tail(gate_up)
            torch.cuda.synchronize()
            deep_ms = _cuda_bench_ms(
                lambda: self._prefill_deepfusion(gate_up),
                iters=_DEEPFUSION_PREFILL_BENCH_ITERS,
            )
            base_ms = _cuda_bench_ms(
                lambda: self._prefill_baseline_tail(gate_up),
                iters=_DEEPFUSION_PREFILL_BENCH_ITERS,
            )
            use = bool(
                deep_ms <= (base_ms * (1.0 - _DEEPFUSION_MLP_MIN_GAIN))
            )
            self._prefill_deepfusion_bench[sig] = {
                "deep_ms": float(deep_ms),
                "base_ms": float(base_ms),
            }
            self._log_prefill_deepfusion(
                f"shape={sig[:4]} deep={deep_ms:.2f}ms base={base_ms:.2f}ms use={int(use)}"
            )
        except Exception as exc:
            self._log_prefill_deepfusion(f"shape={sig[:4]} disabled after error: {exc}")
            self._disable_prefill_deepfusion = True
            use = False
        self._prefill_deepfusion_use_cache[sig] = use
        return use

    def _decode_swiglu_activation(self, gate_up: torch.Tensor) -> torch.Tensor:
        if self.hidden_act in ('gelu', 'gelu_pytorch_tanh'):
            gate = gate_up[..., :self.intermediate_size]
            value = gate_up[..., self.intermediate_size:]
            return self._activation(gate, value)

        if (
            _HAS_TRITON_SWIGLU
            and not _env_enabled("MEGAGEMM_DISABLE_TRITON_SWIGLU", default=False)
            and not self._decode_disable_triton_swiglu
        ):
            if self._decode_swiglu_checked:
                return swiglu_forward(gate_up, self.intermediate_size)
            try:
                activated = swiglu_forward(gate_up, self.intermediate_size)
                self._decode_swiglu_checked = True
                return activated
            except Exception:
                self._decode_disable_triton_swiglu = True

        gate = gate_up[..., :self.intermediate_size]
        value = gate_up[..., self.intermediate_size:]
        return self._activation(gate, value)

    def _decode_down_proj(self, activated: torch.Tensor) -> torch.Tensor:
        down_weight, down_bias = _linear_weight_bias(self.down_proj)
        if down_weight is None:
            return self.down_proj(activated)

        if _USE_DECODE_FAST_LINEAR:
            return _decode_linear(
                self,
                "_fast_down_out",
                activated,
                down_weight,
                down_bias,
                use_fast=True,
            )
        sig = _decode_linear_runtime_sig(activated, down_weight)
        if self._fast_down_decode_key != sig:
            self._fast_down_decode_use = bool(
                _pick_decode_linear_backend(
                    self,
                    "down",
                    activated,
                    down_weight,
                    down_bias,
                    "_fast_down_out",
                )
            )
            self._fast_down_decode_key = sig
        if self._fast_down_decode_use:
            fast_mode = _DECODE_LINEAR_MODE_CACHE.get(
                _decode_linear_backend_key("down", activated, down_weight),
                "",
            )
            return _decode_linear(
                self,
                "_fast_down_out",
                activated,
                down_weight,
                down_bias,
                use_fast=True,
                fast_mode=fast_mode,
            )
        return _decode_linear(
            self,
            "_fast_down_out",
            activated,
            down_weight,
            down_bias,
            use_fast=False,
        )

    def _decode_gate_up_linear(self, x: torch.Tensor) -> torch.Tensor:
        gate_up_weight, gate_up_bias = _linear_weight_bias(self.gate_up_proj)
        if gate_up_weight is None:
            return self.gate_up_proj(x)

        if _USE_DECODE_FAST_LINEAR:
            gate_up_mode = _pick_gate_up_fast_mode(
                self,
                x,
                gate_up_weight,
                gate_up_bias,
                "_fast_gate_up_out",
            )
            return _decode_linear(
                self,
                "_fast_gate_up_out",
                x,
                gate_up_weight,
                gate_up_bias,
                use_fast=True,
                fast_mode=gate_up_mode,
            )
        sig = _decode_linear_runtime_sig(x, gate_up_weight)
        if self._fast_gate_up_decode_key != sig:
            self._fast_gate_up_decode_use = bool(
                _pick_decode_linear_backend(
                    self,
                    "gate_up",
                    x,
                    gate_up_weight,
                    gate_up_bias,
                    "_fast_gate_up_out",
                )
            )
            self._fast_gate_up_decode_key = sig
            self._fast_gate_up_mode_key = None
        if self._fast_gate_up_decode_use:
            if self._fast_gate_up_mode_key != sig:
                self._fast_gate_up_mode = _pick_gate_up_fast_mode(
                    self,
                    x,
                    gate_up_weight,
                    gate_up_bias,
                    "_fast_gate_up_out",
                )
                self._fast_gate_up_mode_key = sig
            return _decode_linear(
                self,
                "_fast_gate_up_out",
                x,
                gate_up_weight,
                gate_up_bias,
                use_fast=True,
                fast_mode=self._fast_gate_up_mode,
            )
        return _decode_linear(
            self,
            "_fast_gate_up_out",
            x,
            gate_up_weight,
            gate_up_bias,
            use_fast=False,
        )

    def _decode_gate_up_from_raw_hidden(
        self,
        hidden_states: torch.Tensor,
        input_norm_weight: torch.Tensor,
        input_norm_eps: float,
        input_norm_offset: bool,
    ) -> torch.Tensor:
        gate_up_weight, gate_up_bias = _linear_weight_bias(self.gate_up_proj)
        if gate_up_weight is None:
            normed = _decode_rmsnorm(
                hidden_states,
                input_norm_weight,
                input_norm_eps,
                input_norm_offset,
            )
            return self.gate_up_proj(normed)

        out_features = 2 * self.intermediate_size
        if (
            not self._disable_fused_rmsnorm_gateup_decode
            and _cached_fused_rmsnorm_gateup_decision(
                self,
                "_fused_rmsnorm_gateup_decode_key",
                "_fused_rmsnorm_gateup_decode_use",
                hidden_states,
                out_features=out_features,
            )
        ):
            if not self._fused_rmsnorm_gateup_checked:
                try:
                    rows = int(hidden_states.numel() // max(1, int(hidden_states.shape[-1])))
                    gate_up_out = _get_reusable_out_decode(
                        self,
                        "_fast_gate_up_out",
                        (*hidden_states.shape[:-1], out_features),
                        hidden_states,
                    )
                    gate_up_inv_rms = _get_reusable_out_decode_typed(
                        self,
                        "_fast_gate_up_inv_rms",
                        (rows,),
                        hidden_states,
                        torch.float32,
                    )
                    fused_rmsnorm_linear(
                        hidden_states,
                        input_norm_weight,
                        input_norm_eps,
                        gate_up_weight,
                        gate_up_bias,
                        norm_offset=input_norm_offset,
                        out=gate_up_out,
                        inv_rms=gate_up_inv_rms,
                    )
                    baseline_normed = _decode_rmsnorm(
                        hidden_states,
                        input_norm_weight,
                        input_norm_eps,
                        input_norm_offset,
                    )
                    self._decode_gate_up_linear(baseline_normed)
                    torch.cuda.synchronize()
                    fused_ms = _cuda_bench_ms(
                        lambda: fused_rmsnorm_linear(
                            hidden_states,
                            input_norm_weight,
                            input_norm_eps,
                            gate_up_weight,
                            gate_up_bias,
                            norm_offset=input_norm_offset,
                            out=gate_up_out,
                            inv_rms=gate_up_inv_rms,
                        ),
                        iters=8,
                    )
                    base_ms = _cuda_bench_ms(
                        lambda: self._decode_gate_up_linear(
                            _decode_rmsnorm(
                                hidden_states,
                                input_norm_weight,
                                input_norm_eps,
                                input_norm_offset,
                            )
                        ),
                        iters=8,
                    )
                    self._fused_rmsnorm_gateup_decode_use = bool(
                        fused_ms <= (base_ms * (1.0 - _FUSED_RMSNORM_GATEUP_MIN_GAIN))
                    )
                    self._fused_rmsnorm_gateup_checked = True
                except Exception:
                    self._disable_fused_rmsnorm_gateup_decode = True
                    self._fused_rmsnorm_gateup_decode_use = False

            if self._fused_rmsnorm_gateup_decode_use and not self._disable_fused_rmsnorm_gateup_decode:
                try:
                    rows = int(hidden_states.numel() // max(1, int(hidden_states.shape[-1])))
                    gate_up_out = _get_reusable_out_decode(
                        self,
                        "_fast_gate_up_out",
                        (*hidden_states.shape[:-1], out_features),
                        hidden_states,
	                    )
                    gate_up_inv_rms = _get_reusable_out_decode_typed(
                        self,
                        "_fast_gate_up_inv_rms",
                        (rows,),
                        hidden_states,
                        torch.float32,
                    )
                    return fused_rmsnorm_linear(
                        hidden_states,
                        input_norm_weight,
                        input_norm_eps,
                        gate_up_weight,
                        gate_up_bias,
                        norm_offset=input_norm_offset,
                        out=gate_up_out,
                        inv_rms=gate_up_inv_rms,
                    )
                except Exception:
                    self._disable_fused_rmsnorm_gateup_decode = True
                    self._fused_rmsnorm_gateup_decode_use = False

        normed = _decode_rmsnorm(
            hidden_states,
            input_norm_weight,
            input_norm_eps,
            input_norm_offset,
        )
        return self._decode_gate_up_linear(normed)

    def forward(
        self,
        x: torch.Tensor,
        timing_events: Optional[dict] = None,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        do_prefill_stage_timing = (
            timing_events is not None
            and x.is_cuda
            and _prefill_timing_enabled()
        )
        if self._awq_separate:
            gate_up_start_end = _timing_record_start(do_prefill_stage_timing)
            # AWQ path: separate gate/up projections (quantized)
            gate = self.gate_proj(x)
            value = self.up_proj(x)
            activated = self._activation(gate, value)
            _timing_record_end(timing_events, "gate_up", gate_up_start_end)
        else:
            if (
                is_prefill
                and not _USE_DEEPFUSION_MLP_PREFILL
                and _USE_NATIVE_MLP_PREFILL
                and not self._disable_native_mlp_prefill
                and native_mlp_prefill_forward_cuda is not None
                and x.is_cuda
                and not torch.is_grad_enabled()
                and self.hidden_act not in ('gelu', 'gelu_pytorch_tanh')
            ):
                gate_up_weight, gate_up_bias = _linear_weight_bias(self.gate_up_proj)
                down_weight, down_bias = _linear_weight_bias(self.down_proj)
                if gate_up_weight is not None and down_weight is not None:
                    native_start_end = _timing_record_start(do_prefill_stage_timing)
                    try:
                        out = native_mlp_prefill_forward_cuda(
                            x,
                            gate_up_weight.contiguous(),
                            gate_up_bias.contiguous() if gate_up_bias is not None else None,
                            down_weight.contiguous(),
                            down_bias.contiguous() if down_bias is not None else None,
                            self.intermediate_size,
                        )
                        _timing_record_end(timing_events, "mlp_native", native_start_end)
                        return out
                    except Exception:
                        self._disable_native_mlp_prefill = True

            # FP16 path: fused gate_up_proj
            gate_up_start_end = _timing_record_start(do_prefill_stage_timing)
            gate_up_weight, gate_up_bias = _linear_weight_bias(self.gate_up_proj)
            if (
                is_prefill
                and gate_up_weight is not None
                and x.is_cuda
                and not torch.is_grad_enabled()
            ):
                gate_up_out_features = int(gate_up_weight.shape[0])
                gate_up_sequence_len = (
                    int(x.shape[1]) if x.ndim == 3 else 0
                )
                gate_up_algorithm_key = (
                    gate_up_sequence_len,
                    gate_up_out_features,
                )
                use_prefill_cublaslt_gateup = bool(
                    self._gemma4_e2b_prefill_cublaslt_gateup_enabled
                    and not self._gemma4_e2b_prefill_cublaslt_gateup_runtime_disabled
                    and HAS_CUBLASLT_BF16_LINEAR
                    and callable(cublaslt_bf16_linear_cuda)
                    and x.dtype == torch.bfloat16
                    and x.ndim == 3
                    and int(x.shape[0]) == 8
                    and gate_up_sequence_len in (521, 2057)
                    and int(x.shape[2]) == 1536
                    and gate_up_out_features in (12288, 24576)
                    and gate_up_algorithm_key
                    in self._gemma4_e2b_prefill_cublaslt_gateup_algorithms
                )
                gate_up = None
                if use_prefill_cublaslt_gateup:
                    try:
                        gate_up_out = _get_prefill_out(
                            self,
                            "_fast_gate_up_out",
                            (*x.shape[:-1], gate_up_out_features),
                            x,
                        )
                        x_2d = x.flatten(0, -2)
                        gate_up_out_2d = gate_up_out.flatten(0, -2)
                        cublaslt_bf16_linear_cuda(
                            x_2d,
                            gate_up_weight,
                            gate_up_bias,
                            out=gate_up_out_2d,
                            algorithm_index=int(
                                self._gemma4_e2b_prefill_cublaslt_gateup_algorithms[
                                    gate_up_algorithm_key
                                ]
                            ),
                        )
                        gate_up = gate_up_out
                    except Exception as exc:
                        self._gemma4_e2b_prefill_cublaslt_gateup_runtime_disabled = True
                        if not self._gemma4_e2b_prefill_cublaslt_gateup_failure:
                            self._gemma4_e2b_prefill_cublaslt_gateup_failure = (
                                f"{type(exc).__name__}: {exc}"
                            )
                        gate_up = None
                    else:
                        self._gemma4_e2b_prefill_cublaslt_gateup_hits += 1
                if gate_up is None:
                    gate_up = _prefill_linear(
                        self,
                        "_fast_gate_up_out",
                        x,
                        gate_up_weight,
                        gate_up_bias,
                    )
            elif _can_use_fast_gemv_for(
                "gate_up",
                x,
                out_features=2 * self.intermediate_size,
            ) and gate_up_weight is not None:
                gate_up_out = _get_reusable_out(
                    self,
                    "_fast_gate_up_out",
                    (*x.shape[:-1], 2 * self.intermediate_size),
                    x,
                )
                gate_up = fast_linear(x, gate_up_weight, gate_up_bias, out=gate_up_out)
            else:
                gate_up = self.gate_up_proj(x)
            _timing_record_end(timing_events, "gate_up", gate_up_start_end)

        down_weight, down_bias = _linear_weight_bias(self.down_proj)
        if (
            not self._awq_separate
            and self.hidden_act not in ('gelu', 'gelu_pytorch_tanh')
            and _USE_DEEPFUSION_MLP_PREFILL
            and not self._disable_prefill_deepfusion
            and down_weight is not None
            and gate_up.is_cuda
            and not torch.is_grad_enabled()
            and _cached_deepfusion_decision(
                self,
                "_deepfusion_prefill_key",
                "_deepfusion_prefill_eligible",
                gate_up,
                down_weight,
            )
        ):
            try:
                if self._should_use_prefill_deepfusion(gate_up, down_weight):
                    down_start_end = _timing_record_start(do_prefill_stage_timing)
                    out = self._prefill_deepfusion(gate_up)
                    _timing_record_end(timing_events, "down_proj", down_start_end)
                    return out
            except Exception:
                self._disable_prefill_deepfusion = True

        return self._prefill_baseline_tail(
            gate_up,
            timing_events=timing_events,
            do_prefill_stage_timing=do_prefill_stage_timing,
        )

    def forward_decode(
        self,
        x: torch.Tensor,
        input_is_normed: bool = True,
        input_norm_weight: Optional[torch.Tensor] = None,
        input_norm_eps: float = 1e-6,
        input_norm_offset: bool = False,
        timing_events: Optional[dict] = None,
    ) -> torch.Tensor:
        if self._awq_separate:
            return self.forward(x)

        if not input_is_normed and input_norm_weight is not None:
            gate_up = self._decode_gate_up_from_raw_hidden(
                x,
                input_norm_weight,
                input_norm_eps,
                input_norm_offset,
            )
        else:
            gate_up = self._decode_gate_up_linear(x)

        down_weight, _ = _linear_weight_bias(self.down_proj)
        if (
            not self._decode_disable_deepfusion
            and down_weight is not None
            and _cached_deepfusion_decision(
                self,
                "_deepfusion_decode_key",
                "_deepfusion_decode_use",
                gate_up,
                down_weight,
            )
        ):
            if not self._decode_deepfusion_checked:
                try:
                    self._decode_deepfusion_checked = True
                    # One-time shape-local benchmark gate:
                    # use deepfusion only if it beats current decode path.
                    self._decode_deepfusion(gate_up)
                    self._decode_down_proj(self._decode_swiglu_activation(gate_up))
                    torch.cuda.synchronize()
                    deep_ms = _cuda_bench_ms(
                        lambda: self._decode_deepfusion(gate_up),
                        iters=8,
                    )
                    base_ms = _cuda_bench_ms(
                        lambda: self._decode_down_proj(self._decode_swiglu_activation(gate_up)),
                        iters=8,
                    )
                    self._deepfusion_decode_use = bool(
                        deep_ms <= (base_ms * (1.0 - _DEEPFUSION_MLP_MIN_GAIN))
                    )
                except Exception:
                    self._decode_disable_deepfusion = True
                    self._deepfusion_decode_use = False
            if self._deepfusion_decode_use and not self._decode_disable_deepfusion:
                try:
                    out = self._decode_deepfusion(gate_up)
                    self._deepfusion_decode_hits += 1
                    return out
                except Exception:
                    self._decode_disable_deepfusion = True
                    self._deepfusion_decode_use = False

        activated = self._decode_swiglu_activation(gate_up)
        return self._decode_down_proj(activated)

    def forward_decode_add_residual(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        input_is_normed: bool = True,
        input_norm_weight: Optional[torch.Tensor] = None,
        input_norm_eps: float = 1e-6,
        input_norm_offset: bool = False,
        timing_events: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Decode-only MLP that writes the residual add in-place when possible.
        Returns the updated residual tensor.
        """
        if self._awq_separate:
            mlp_out = self.forward_decode(
                x,
                input_is_normed=input_is_normed,
                input_norm_weight=input_norm_weight,
                input_norm_eps=input_norm_eps,
                input_norm_offset=input_norm_offset,
            )
            residual.add_(mlp_out)
            return residual

        if not input_is_normed and input_norm_weight is not None:
            gate_up = self._decode_gate_up_from_raw_hidden(
                x,
                input_norm_weight,
                input_norm_eps,
                input_norm_offset,
            )
        else:
            gate_up = self._decode_gate_up_linear(x)

        down_weight, _ = _linear_weight_bias(self.down_proj)
        if (
            not self._decode_disable_deepfusion
            and down_weight is not None
            and _cached_deepfusion_decision(
                self,
                "_deepfusion_decode_key",
                "_deepfusion_decode_use",
                gate_up,
                down_weight,
            )
        ):
            if not self._decode_deepfusion_checked:
                try:
                    self._decode_deepfusion_checked = True
                    self._decode_deepfusion(gate_up)
                    self._decode_down_proj(self._decode_swiglu_activation(gate_up))
                    torch.cuda.synchronize()
                    deep_ms = _cuda_bench_ms(
                        lambda: self._decode_deepfusion(gate_up),
                        iters=8,
                    )
                    base_ms = _cuda_bench_ms(
                        lambda: self._decode_down_proj(self._decode_swiglu_activation(gate_up)),
                        iters=8,
                    )
                    self._deepfusion_decode_use = bool(
                        deep_ms <= (base_ms * (1.0 - _DEEPFUSION_MLP_MIN_GAIN))
                    )
                except Exception:
                    self._decode_disable_deepfusion = True
                    self._deepfusion_decode_use = False
            if self._deepfusion_decode_use and not self._decode_disable_deepfusion:
                try:
                    out = self._decode_deepfusion(gate_up, residual=residual)
                    self._deepfusion_decode_hits += 1
                    return out
                except Exception:
                    self._decode_disable_deepfusion = True
                    self._deepfusion_decode_use = False

        activated = self._decode_swiglu_activation(gate_up)
        mlp_out = self._decode_down_proj(activated)
        residual.add_(mlp_out)
        return residual


class Qwen3MoeTopKRouter(nn.Module):
    """Qwen3 MoE router: softmax over experts, then normalized top-k routing."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.top_k = int(config.num_experts_per_tok)
        self.num_experts = int(config.num_experts)
        self.norm_topk_prob = bool(config.norm_topk_prob)
        self.hidden_dim = int(config.hidden_size)
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        # Checkpoint loading overwrites this tensor, but a directly constructed
        # model must still be numerically well-defined (tests, conversion tools,
        # and synthetic snapshots all exercise that path).
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self._topk_workspace: dict[str, torch.Tensor] = {}
        self._router_logits: Optional[torch.Tensor] = None
        self._fused_router_disabled = False
        self._fused_router_fail_reason = ""
        self._fused_router_hits = 0

    def _logits_buffer(self, hidden_2d: torch.Tensor) -> torch.Tensor:
        shape = (int(hidden_2d.shape[0]), self.num_experts)
        logits = self._router_logits
        if (
            logits is None
            or tuple(logits.shape) != shape
            or logits.device != hidden_2d.device
            or logits.dtype != hidden_2d.dtype
        ):
            logits = torch.empty(shape, device=hidden_2d.device, dtype=hidden_2d.dtype)
            self._router_logits = logits
        return logits

    def forward(self, hidden_states: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        hidden_2d = hidden_states.reshape(-1, self.hidden_dim)
        top_k = min(self.top_k, self.num_experts)
        router_logits = None
        if self.norm_topk_prob:
            # Qwen3 normalizes only the selected experts. Since softmax is monotonic,
            # topk(softmax(logits)) == topk(logits), and the full softmax denominator
            # cancels during top-k renormalization.
            if (
                qwen3_moe_router_topk_softmax is not None
                and not self._fused_router_disabled
                and hidden_2d.is_cuda
                and not torch.is_grad_enabled()
                and hidden_2d.dtype in (torch.float16, torch.bfloat16, torch.float32)
            ):
                try:
                    routing_weights, selected_experts = qwen3_moe_router_topk_softmax(
                        hidden_2d.contiguous(),
                        self.weight.contiguous(),
                        top_k,
                        workspace=self._topk_workspace,
                    )
                    self._fused_router_hits += 1
                    routing_weights = routing_weights.to(hidden_2d.dtype)
                    return router_logits, routing_weights, selected_experts
                except Exception as exc:
                    self._fused_router_disabled = True
                    self._fused_router_fail_reason = str(exc)

            if (
                hidden_2d.is_cuda
                and not torch.is_grad_enabled()
                and hidden_2d.dtype in (torch.float16, torch.bfloat16, torch.float32)
            ):
                router_logits = self._logits_buffer(hidden_2d)
                torch.mm(hidden_2d, self.weight.t(), out=router_logits)
            else:
                router_logits = torch.nn.functional.linear(hidden_2d, self.weight)

            if qwen3_moe_topk_softmax is not None:
                routing_weights, selected_experts = qwen3_moe_topk_softmax(
                    router_logits.contiguous(),
                    top_k,
                    workspace=self._topk_workspace,
                )
            else:
                top_logits, selected_experts = torch.topk(router_logits, top_k, dim=-1)
                routing_weights = torch.nn.functional.softmax(
                    top_logits,
                    dtype=torch.float32,
                    dim=-1,
                )
        else:
            if (
                hidden_2d.is_cuda
                and not torch.is_grad_enabled()
                and hidden_2d.dtype in (torch.float16, torch.bfloat16, torch.float32)
            ):
                router_logits = self._logits_buffer(hidden_2d)
                torch.mm(hidden_2d, self.weight.t(), out=router_logits)
            else:
                router_logits = torch.nn.functional.linear(hidden_2d, self.weight)
            router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float32, dim=-1)
            routing_weights, selected_experts = torch.topk(router_probs, top_k, dim=-1)
        routing_weights = routing_weights.to(router_logits.dtype)
        return router_logits, routing_weights, selected_experts


class Qwen3MoeExperts(nn.Module):
    """Collection of Qwen3 MoE experts stored in HF-compatible 3D tensors."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.num_experts = int(config.num_experts)
        self.hidden_dim = int(config.hidden_size)
        self.intermediate_dim = int(config.moe_intermediate_size or config.intermediate_size)
        self.hidden_act = config.hidden_act
        self._gemma4_a4b_segmented_prefill = _gemma4_a4b_segmented_prefill_shape(
            config.model_type,
            self.num_experts,
            self.hidden_dim,
            self.intermediate_dim,
            config.num_experts_per_tok,
        )
        # Preserve the measured assignment kernel through batch 8, then switch
        # to the graph-safe expert-grouped kernel before the 64-assignment cliff.
        self._gemma4_batch_decode_max_assignments = (
            128 if self._gemma4_a4b_segmented_prefill else None
        )
        self._gemma4_batch_decode_compact = bool(
            self._gemma4_a4b_segmented_prefill
        )
        self._gemma4_batch_decode_use_compact = bool(
            self._gemma4_a4b_segmented_prefill
        )
        self._gemma4_batch_decode_deterministic_reduce = bool(
            self._gemma4_a4b_segmented_prefill
        )
        self._gemma4_batch_decode_compact_min_rows = 9
        self._gemma4_batch_decode_compact_max_rows = 16
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        # These parameters are usually populated by the HF/MGX loaders.  Keep
        # the standalone module valid as well; leaving torch.empty() untouched
        # can surface allocator garbage, including NaNs, in local conversions.
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj, mean=0.0, std=0.02)
        self.register_buffer("gate_up_int8", None)
        self.register_buffer("gate_up_scale", None)
        self.register_buffer("down_int8", None)
        self.register_buffer("down_scale", None)
        self.register_buffer("gate_up_qweight", None)
        self.register_buffer("gate_up_scales", None)
        self.register_buffer("gate_up_qzeros", None)
        self.register_buffer("down_qweight", None)
        self.register_buffer("down_scales", None)
        self.register_buffer("down_qzeros", None)
        self.awq_group_size = 0
        self._grouped_decode_disabled = False
        self._grouped_decode_fail_reason = ""
        self._grouped_decode_hits = 0
        self._grouped_decode_workspace: dict[str, torch.Tensor] = {}
        self._segmented_prefill_disabled = False
        self._segmented_prefill_fail_reason = ""
        self._segmented_prefill_hits = 0
        self._segmented_prefill_residual_fused_hits = 0
        self._segmented_prefill_assignments = 0
        self._segmented_prefill_tiles = 0
        self._segmented_prefill_async_tile_hits = 0
        self._segmented_prefill_max_tiles = 0
        self._segmented_prefill_partial_reduce_hits = 0
        self._segmented_prefill_single_accumulator_hits = 0
        self._segmented_prefill_sorted_partial_hits = 0
        self._segmented_prefill_fixed_route_pack_hits = 0
        self._segmented_prefill_compact_route_pack_hits = 0
        self._segmented_prefill_route_scatter_hits = 0
        self._segmented_prefill_route_argsort_hits = 0
        self._segmented_prefill_route_scatter_fail_reason = ""
        self._segmented_prefill_workspace: dict[str, torch.Tensor] = {}
        self._segmented_prefill_runtime_options_by_rows: dict[int, dict[str, Any]] = {}
        self._gemma4_grouped_mm_prefill_runtime_by_rows: dict[int, bool] = {}
        self._gemma4_grouped_mm_prefill_disabled = False
        self._gemma4_grouped_mm_prefill_fail_reason = ""
        self._gemma4_grouped_mm_prefill_hits = 0
        self._gemma4_grouped_mm_prefill_last_active = False
        self._gemma4_grouped_mm_prefill_workspace: dict[str, Any] = {}
        self._gemma4_long_dominant_expert_prefill_disabled = False
        self._gemma4_long_dominant_expert_prefill_fail_reason = ""
        self._gemma4_long_dominant_expert_prefill_hits = 0
        self._gemma4_long_dominant_expert_prefill_assignments = 0
        self._gemma4_long_dominant_expert_prefill_guard_misses = 0
        self._gemma4_long_dominant_expert_prefill_last_guard_reason = ""
        self._gemma4_long_dominant_expert_prefill_last_active = False
        self._gemma4_long_dominant_expert_prefill_workspace: dict[str, Any] = {}
        self._gemma4_long_padded_bmm_prefill_disabled = False
        self._gemma4_long_padded_bmm_prefill_fail_reason = ""
        self._gemma4_long_padded_bmm_prefill_hits = 0
        self._gemma4_long_padded_bmm_prefill_assignments = 0
        self._gemma4_long_padded_bmm_prefill_last_active = False
        self._gemma4_long_padded_bmm_prefill_workspace: dict[str, Any] = {}
        self._sorted_prefill_disabled = False
        self._sorted_prefill_fail_reason = ""
        self._sorted_prefill_hits = 0
        self._sorted_prefill_workspace: dict[str, torch.Tensor] = {}
        self._batched_prefill_disabled = False
        self._batched_prefill_fail_reason = ""
        self._batched_prefill_hits = 0
        self._batched_prefill_workspace: dict[str, torch.Tensor] = {}
        self._bucketed_prefill_disabled = False
        self._bucketed_prefill_fail_reason = ""
        self._bucketed_prefill_hits = 0
        self._bucketed_prefill_valid_assignments = 0
        self._bucketed_prefill_padded_assignments = 0
        self._bucketed_prefill_bucket_launches = 0
        self._bucketed_prefill_workspace: dict[str, torch.Tensor] = {}
        self._int8_dequant_prefill_disabled = False
        self._int8_dequant_prefill_fail_reason = ""
        self._int8_dequant_prefill_hits = 0

    def _segmented_prefill_kernel_options(
        self,
        rows: Optional[int] = None,
        *,
        dtype: Optional[torch.dtype] = None,
        device_name: Optional[str] = None,
    ) -> dict:
        if not self._gemma4_a4b_segmented_prefill:
            return {}
        options = dict(_GEMMA4_A4B_SEGMENTED_PREFILL_OPTIONS)
        if rows is not None and int(rows) >= _GEMMA4_A4B_SEGMENTED_PREFILL_LARGE_ROWS_MIN:
            options.update(_GEMMA4_A4B_SEGMENTED_PREFILL_LARGE_OPTIONS)
        weight = getattr(self, "gate_up_proj", None)
        if dtype is None:
            dtype = getattr(weight, "dtype", None)
        if device_name is None:
            weight_device = getattr(weight, "device", None)
            device_name = (
                torch.cuda.get_device_name(weight_device)
                if weight_device is not None
                and getattr(weight_device, "type", "") == "cuda"
                else ""
            )
        if rows is not None and _gemma4_a100_a4b_segmented_prefill_long_shape(
            int(rows),
            dtype,
            str(device_name),
        ):
            options.update(_GEMMA4_A4B_SEGMENTED_PREFILL_LONG_OPTIONS)
        if rows is not None and int(rows) <= _GEMMA4_A4B_SEGMENTED_PREFILL_SHORT_ROWS_MAX:
            options.update(_GEMMA4_A4B_SEGMENTED_PREFILL_SHORT_OPTIONS)
        if rows is not None:
            options.update(
                getattr(
                    self,
                    "_segmented_prefill_runtime_options_by_rows",
                    {},
                ).get(int(rows), {})
            )
        return options

    def set_segmented_prefill_runtime_options(
        self,
        rows: int,
        options: dict[str, Any],
    ) -> None:
        allowed = {
            "block_m",
            "block_n",
            "block_k",
            "fused_gate_block_n",
            "num_warps",
            "num_stages",
            "single_accumulator",
            "sorted_partial",
            "group_size_m",
        }
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(
                f"unsupported segmented prefill runtime options: {sorted(unknown)}"
            )
        selected = {
            key: (
                bool(value)
                if key in ("single_accumulator", "sorted_partial")
                else max(1, int(value))
            )
            for key, value in options.items()
        }
        self._segmented_prefill_runtime_options_by_rows[int(rows)] = selected
        # Route and tile buffers are shape/config dependent. Runtime selection is
        # performed before graph capture, so discard any buffers from the gate.
        self._segmented_prefill_workspace.clear()

    def set_gemma4_grouped_mm_prefill_runtime(
        self,
        rows: int,
        enabled: bool,
    ) -> None:
        if enabled and not self._gemma4_a4b_segmented_prefill:
            raise ValueError(
                "Gemma4 grouped-MM prefill is only supported by the A4B expert shape"
            )
        self._gemma4_grouped_mm_prefill_runtime_by_rows[int(rows)] = bool(enabled)
        self._gemma4_grouped_mm_prefill_workspace.clear()
        self._gemma4_grouped_mm_prefill_last_active = False

    def _gemma4_grouped_mm_prefill_is_enabled(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> bool:
        rows = int(hidden_states.shape[0])
        return bool(
            self._gemma4_grouped_mm_prefill_runtime_by_rows.get(rows, False)
            and not self._gemma4_grouped_mm_prefill_disabled
            and callable(gemma4_grouped_mm_prefill)
            and callable(gemma4_grouped_mm_prefill_prefers_shape)
            and gemma4_grouped_mm_prefill_prefers_shape(
                hidden_states,
                self.gate_up_proj,
                self.down_proj,
                selected_experts,
                routing_weights,
            )
        )

    def _forward_gemma4_grouped_mm_prefill(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if gemma4_grouped_mm_prefill is None:
            raise RuntimeError("Gemma4 grouped-MM prefill kernel is unavailable")
        out = gemma4_grouped_mm_prefill(
            hidden_states,
            self.gate_up_proj,
            self.down_proj,
            selected_experts,
            routing_weights,
            out=residual,
            residual=residual,
            workspace=self._gemma4_grouped_mm_prefill_workspace,
        )
        self._gemma4_grouped_mm_prefill_hits += 1
        self._gemma4_grouped_mm_prefill_last_active = True
        return out

    def _gemma4_long_dominant_expert_prefill_is_enabled(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        *,
        graph_safe_prefill: bool = False,
    ) -> bool:
        if not (
            _GEMMA4_A4B_LONG_DOMINANT_EXPERT_PREFILL
            and self._gemma4_a4b_segmented_prefill
            and not self._gemma4_long_dominant_expert_prefill_disabled
            and callable(qwen3_moe_dominant_expert_padded_bmm_prefill)
            and not graph_safe_prefill
            and not torch.is_grad_enabled()
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and tuple(hidden_states.shape)
            == (_GEMMA4_A4B_SEGMENTED_PREFILL_LONG_MAX_ROWS, self.hidden_dim)
            and tuple(selected_experts.shape)
            == (_GEMMA4_A4B_SEGMENTED_PREFILL_LONG_MAX_ROWS, 8)
            and tuple(routing_weights.shape) == tuple(selected_experts.shape)
            and selected_experts.is_cuda
            and routing_weights.is_cuda
            and self.gate_up_proj.is_cuda
            and self.down_proj.is_cuda
            and self.gate_up_proj.dtype == torch.bfloat16
            and self.down_proj.dtype == torch.bfloat16
        ):
            return False
        return "A100" in torch.cuda.get_device_name(hidden_states.device).upper()

    @staticmethod
    def _gemma4_long_dominant_expert_guard_rejection(exc: Exception) -> bool:
        message = str(exc)
        return message.startswith(
            (
                "Dominant-expert guard:",
                "Dominant-expert light-capacity guard:",
                "Dominant-expert path requires non-heavy assignments",
            )
        )

    def _forward_gemma4_long_dominant_expert_prefill(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if qwen3_moe_dominant_expert_padded_bmm_prefill is None:
            raise RuntimeError(
                "Gemma4 long dominant-expert prefill kernel is unavailable"
            )
        self._gemma4_grouped_mm_prefill_last_active = False
        out = qwen3_moe_dominant_expert_padded_bmm_prefill(
            hidden_states,
            self.gate_up_proj,
            self.down_proj,
            selected_experts,
            routing_weights,
            activation=self.hidden_act,
            out=residual,
            residual=residual,
            workspace=self._gemma4_long_dominant_expert_prefill_workspace,
            align_m=16,
            route_pack_block=256,
            activation_block=512,
            reduce_block_n=256,
            reduce_num_warps=4,
            minimum_dominant_skew=(
                _GEMMA4_A4B_LONG_DOMINANT_EXPERT_MIN_SKEW
            ),
            max_light_padding_ratio=(
                _GEMMA4_A4B_LONG_DOMINANT_EXPERT_MAX_LIGHT_PADDING_RATIO
            ),
        )
        # The segmented workspace reports the most recently exercised reduce.
        # Clear it after a successful hybrid call so runtime contracts cannot
        # mistake an excluded reference run for the active request path.
        self._segmented_prefill_workspace.clear()
        self._gemma4_long_dominant_expert_prefill_hits += 1
        self._gemma4_long_dominant_expert_prefill_assignments += int(
            hidden_states.shape[0]
        ) * int(selected_experts.shape[1])
        self._gemma4_long_dominant_expert_prefill_last_guard_reason = ""
        self._gemma4_long_dominant_expert_prefill_last_active = True
        return out

    def _record_gemma4_long_dominant_expert_failure(
        self,
        exc: Exception,
    ) -> bool:
        """Record a route guard miss without disabling the measured kernel."""
        self._gemma4_long_dominant_expert_prefill_last_active = False
        message = str(exc)
        if self._gemma4_long_dominant_expert_guard_rejection(exc):
            self._gemma4_long_dominant_expert_prefill_guard_misses += 1
            self._gemma4_long_dominant_expert_prefill_last_guard_reason = message
            self._gemma4_long_dominant_expert_prefill_workspace.clear()
            return True
        self._gemma4_long_dominant_expert_prefill_disabled = True
        self._gemma4_long_dominant_expert_prefill_fail_reason = message
        self._gemma4_long_dominant_expert_prefill_last_guard_reason = ""
        return False

    def _gemma4_long_padded_bmm_prefill_is_enabled(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        *,
        graph_safe_prefill: bool = False,
    ) -> bool:
        if not (
            _GEMMA4_A4B_LONG_PADDED_BMM_PREFILL
            and self._gemma4_a4b_segmented_prefill
            and not self._gemma4_long_padded_bmm_prefill_disabled
            and callable(qwen3_moe_padded_bmm_prefill)
            and not graph_safe_prefill
            and not torch.is_grad_enabled()
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and tuple(hidden_states.shape)
            == (_GEMMA4_A4B_SEGMENTED_PREFILL_LONG_ROWS, self.hidden_dim)
            and tuple(selected_experts.shape)
            == (_GEMMA4_A4B_SEGMENTED_PREFILL_LONG_ROWS, 8)
            and tuple(routing_weights.shape) == tuple(selected_experts.shape)
            and selected_experts.is_cuda
            and routing_weights.is_cuda
            and self.gate_up_proj.is_cuda
            and self.down_proj.is_cuda
            and self.gate_up_proj.dtype == torch.bfloat16
            and self.down_proj.dtype == torch.bfloat16
        ):
            return False
        return "A100" in torch.cuda.get_device_name(hidden_states.device).upper()

    def _forward_gemma4_long_padded_bmm_prefill(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if qwen3_moe_padded_bmm_prefill is None:
            raise RuntimeError("Gemma4 long padded-BMM prefill kernel is unavailable")
        self._gemma4_grouped_mm_prefill_last_active = False
        deterministic_route_pack = torch.are_deterministic_algorithms_enabled()
        out = qwen3_moe_padded_bmm_prefill(
            hidden_states,
            self.gate_up_proj,
            self.down_proj,
            selected_experts,
            routing_weights,
            activation=self.hidden_act,
            out=residual,
            residual=residual,
            workspace=self._gemma4_long_padded_bmm_prefill_workspace,
            down_output_dtype="fp32",
            align_m=16,
            route_pack="argsort" if deterministic_route_pack else "atomic",
            route_pack_block=256,
            fused_activation=True,
            activation_block=512,
            max_padding_ratio=_GEMMA4_A4B_LONG_PADDED_BMM_MAX_PADDING_RATIO,
        )
        self._gemma4_long_padded_bmm_prefill_hits += 1
        self._gemma4_long_padded_bmm_prefill_assignments += int(
            hidden_states.shape[0]
        ) * int(selected_experts.shape[1])
        self._gemma4_long_padded_bmm_prefill_last_active = True
        return out

    def prepare_segmented_prefill_cuda_graph_workspace(
        self,
        *,
        rows: int,
        top_k: int,
        device: torch.device,
    ) -> torch.Tensor:
        if qwen3_moe_prepare_segmented_prefill_graph_workspace is None:
            raise RuntimeError("Segmented MoE graph workspace helper is unavailable")
        if not self._gemma4_a4b_segmented_prefill:
            raise RuntimeError("Segmented MoE graph workspace is only enabled for Gemma 4 A4B")
        options = self._segmented_prefill_kernel_options(int(rows))
        return qwen3_moe_prepare_segmented_prefill_graph_workspace(
            self._segmented_prefill_workspace,
            assignments=int(rows) * int(top_k),
            hidden_dim=self.hidden_dim,
            device=device,
            num_experts=self.num_experts,
            block_m=int(options.get("block_m", 16)),
            route_dtype=self.gate_up_proj.dtype,
        )

    def _segmented_prefill_is_enabled(self, assignments: int) -> bool:
        return bool(
            self._gemma4_a4b_segmented_prefill
            or (
                _USE_QWEN3_MOE_SEGMENTED_PREFILL
                and assignments >= _QWEN3_MOE_SEGMENTED_PREFILL_MIN_ASSIGNMENTS
            )
        )

    def _segmented_prefill_prefers_shape(self, rows: int, top_k: int) -> bool:
        options = self._segmented_prefill_kernel_options(rows)
        return bool(
            callable(qwen3_moe_segmented_prefers_triton_shape)
            and qwen3_moe_segmented_prefers_triton_shape(
                rows,
                top_k,
                self.hidden_dim,
                self.intermediate_dim,
                force=bool(options.get("force", False)),
                block_m=options.get("block_m"),
                block_n=options.get("block_n"),
                block_k=options.get("block_k"),
            )
        )

    def _grouped_decode_out_buffer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self._grouped_decode_workspace.get("out")
        if (
            out is None
            or tuple(out.shape) != tuple(hidden_states.shape)
            or out.device != hidden_states.device
            or out.dtype != hidden_states.dtype
        ):
            out = torch.empty_like(hidden_states)
            self._grouped_decode_workspace["out"] = out
        return out

    def _has_int8_experts(self) -> bool:
        return (
            getattr(self, "gate_up_int8", None) is not None
            and getattr(self, "gate_up_scale", None) is not None
            and getattr(self, "down_int8", None) is not None
            and getattr(self, "down_scale", None) is not None
        )

    def _has_awq_experts(self) -> bool:
        return (
            int(getattr(self, "awq_group_size", 0) or 0) > 0
            and getattr(self, "gate_up_qweight", None) is not None
            and getattr(self, "gate_up_scales", None) is not None
            and getattr(self, "gate_up_qzeros", None) is not None
            and getattr(self, "down_qweight", None) is not None
            and getattr(self, "down_scales", None) is not None
            and getattr(self, "down_qzeros", None) is not None
        )

    def _dequant_expert_weight(
        self,
        weight_int8: torch.Tensor,
        scale: torch.Tensor,
        expert_idx: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return weight_int8[expert_idx].to(dtype) * scale[expert_idx].unsqueeze(1).to(dtype)

    def _dequant_awq_expert_weight(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        expert_idx: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        from ..quantization.w4a16 import _dequantize_pytorch

        weight = _dequantize_pytorch(
            qweight[expert_idx],
            scales[expert_idx],
            qzeros[expert_idx],
            int(self.awq_group_size),
        )
        return weight.to(dtype)

    def _dequant_all_expert_weights(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        gate_up = (
            self.gate_up_int8.to(dtype)
            * self.gate_up_scale.to(dtype).unsqueeze(-1)
        ).contiguous()
        down = (
            self.down_int8.to(dtype)
            * self.down_scale.to(dtype).unsqueeze(-1)
        ).contiguous()
        return gate_up, down

    def _activation(self, gate: torch.Tensor) -> torch.Tensor:
        if self.hidden_act in ('gelu', 'gelu_pytorch_tanh'):
            return torch.nn.functional.gelu(gate, approximate='tanh')
        return torch.nn.functional.silu(gate)

    def _forward_segmented_prefill(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        gate_up_proj: Optional[torch.Tensor] = None,
        down_proj: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        graph_safe_prefill: bool = False,
    ) -> torch.Tensor:
        if qwen3_moe_segmented_prefill is None:
            raise RuntimeError("Qwen3 MoE segmented prefill kernel is not available")
        self._gemma4_grouped_mm_prefill_last_active = False
        out = qwen3_moe_segmented_prefill(
            hidden_states,
            self.gate_up_proj if gate_up_proj is None else gate_up_proj,
            self.down_proj if down_proj is None else down_proj,
            selected_experts,
            routing_weights,
            activation=self.hidden_act,
            out=residual,
            residual=residual,
            workspace=self._segmented_prefill_workspace,
            graph_safe=graph_safe_prefill,
            deterministic_reduce=torch.are_deterministic_algorithms_enabled(),
            **self._segmented_prefill_kernel_options(
                int(hidden_states.shape[0]),
                dtype=hidden_states.dtype,
                device_name=(
                    torch.cuda.get_device_name(hidden_states.device)
                    if hidden_states.is_cuda
                    else ""
                ),
            ),
        )
        self._segmented_prefill_assignments += int(hidden_states.shape[0]) * int(
            selected_experts.shape[1]
        )
        self._segmented_prefill_tiles += int(
            self._segmented_prefill_workspace.get("segmented_prefill_last_tiles", 0) or 0
        )
        if int(self._segmented_prefill_workspace.get("segmented_prefill_async_tiles", 0) or 0):
            self._segmented_prefill_async_tile_hits += 1
            self._segmented_prefill_max_tiles += int(
                self._segmented_prefill_workspace.get("segmented_prefill_max_tiles", 0) or 0
            )
        if int(self._segmented_prefill_workspace.get("segmented_prefill_partial_reduce", 0) or 0):
            self._segmented_prefill_partial_reduce_hits += 1
        if int(
            self._segmented_prefill_workspace.get(
                "segmented_prefill_sorted_partial",
                0,
            )
            or 0
        ):
            self._segmented_prefill_sorted_partial_hits += 1
        if int(
            self._segmented_prefill_workspace.get(
                "segmented_prefill_single_accumulator",
                0,
            )
            or 0
        ):
            self._segmented_prefill_single_accumulator_hits += 1
        if int(
            self._segmented_prefill_workspace.get(
                "segmented_prefill_fixed_route_pack",
                0,
            )
            or 0
        ):
            self._segmented_prefill_fixed_route_pack_hits += 1
        elif int(
            self._segmented_prefill_workspace.get(
                "segmented_prefill_compact_route_pack",
                0,
            )
            or 0
        ):
            self._segmented_prefill_compact_route_pack_hits += 1
        elif int(self._segmented_prefill_workspace.get("segmented_prefill_route_scatter", 0) or 0):
            self._segmented_prefill_route_scatter_hits += 1
        else:
            self._segmented_prefill_route_argsort_hits += 1
        route_fail = str(
            self._segmented_prefill_workspace.get(
                "segmented_prefill_route_scatter_fail_reason",
                "",
            )
            or ""
        )
        if route_fail:
            self._segmented_prefill_route_scatter_fail_reason = route_fail
        return out

    def forward_prefill_add_residual(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.numel() == 0:
            return residual
        if self._has_int8_experts() or self._has_awq_experts():
            raise RuntimeError("Qwen3 MoE prefill residual fusion is BF16/FP16 expert-only")
        if tuple(residual.shape) != tuple(hidden_states.shape):
            raise RuntimeError("Qwen3 MoE prefill residual fusion shape mismatch")

        assignments = int(hidden_states.shape[0]) * int(selected_experts.shape[1])
        self._gemma4_long_dominant_expert_prefill_last_active = False
        self._gemma4_long_dominant_expert_prefill_last_guard_reason = ""
        if self._gemma4_long_dominant_expert_prefill_is_enabled(
            hidden_states,
            selected_experts,
            routing_weights,
        ):
            try:
                return self._forward_gemma4_long_dominant_expert_prefill(
                    hidden_states,
                    selected_experts,
                    routing_weights,
                    residual=residual,
                )
            except Exception as exc:
                self._record_gemma4_long_dominant_expert_failure(exc)

        if self._gemma4_long_padded_bmm_prefill_is_enabled(
            hidden_states,
            selected_experts,
            routing_weights,
        ):
            try:
                return self._forward_gemma4_long_padded_bmm_prefill(
                    hidden_states,
                    selected_experts,
                    routing_weights,
                    residual=residual,
                )
            except Exception as exc:
                self._gemma4_long_padded_bmm_prefill_disabled = True
                self._gemma4_long_padded_bmm_prefill_fail_reason = str(exc)
                self._gemma4_long_padded_bmm_prefill_last_active = False

        if self._gemma4_grouped_mm_prefill_is_enabled(
            hidden_states,
            selected_experts,
            routing_weights,
        ):
            try:
                return self._forward_gemma4_grouped_mm_prefill(
                    hidden_states,
                    selected_experts,
                    routing_weights,
                    residual=residual,
                )
            except Exception as exc:
                self._gemma4_grouped_mm_prefill_disabled = True
                self._gemma4_grouped_mm_prefill_fail_reason = str(exc)
                self._gemma4_grouped_mm_prefill_last_active = False

        if not (
            self._segmented_prefill_is_enabled(assignments)
            and not self._segmented_prefill_disabled
            and qwen3_moe_segmented_prefill is not None
            and hidden_states.is_cuda
            and residual.is_cuda
            and selected_experts.is_cuda
            and routing_weights.is_cuda
            and not torch.is_grad_enabled()
            and self._segmented_prefill_prefers_shape(
                int(hidden_states.shape[0]),
                int(selected_experts.shape[1]),
            )
        ):
            raise RuntimeError("Qwen3 MoE prefill residual fusion is not eligible")

        try:
            out = self._forward_segmented_prefill(
                hidden_states,
                selected_experts,
                routing_weights,
                residual=residual,
            )
            self._segmented_prefill_hits += 1
            if int(
                self._segmented_prefill_workspace.get(
                    "segmented_prefill_residual_fused",
                    0,
                )
                or 0
            ):
                self._segmented_prefill_residual_fused_hits += 1
            return out
        except Exception as exc:
            self._segmented_prefill_fail_reason = str(exc)
            raise

    def _prefill_token_ids(
        self,
        rows: int,
        top_k: int,
        device: torch.device,
    ) -> torch.Tensor:
        key = "token_ids"
        shape = (int(rows) * int(top_k),)
        token_ids = self._sorted_prefill_workspace.get(key)
        if (
            token_ids is None
            or tuple(token_ids.shape) != shape
            or token_ids.device != device
            or token_ids.dtype != torch.int64
        ):
            token_ids = torch.arange(rows, device=device, dtype=torch.int64).repeat_interleave(top_k)
            self._sorted_prefill_workspace[key] = token_ids
        return token_ids

    def forward_decode_add_residual(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.numel() == 0:
            return residual
        if self._has_int8_experts() or self._has_awq_experts():
            raise RuntimeError("Qwen3 MoE residual fusion is BF16/FP16 expert-only")
        if tuple(residual.shape) != tuple(hidden_states.shape):
            raise RuntimeError("Qwen3 MoE residual fusion shape mismatch")
        if not (
            _USE_QWEN3_MOE_GROUPED_DECODE
            and not self._grouped_decode_disabled
            and qwen3_moe_grouped_decode is not None
            and callable(qwen3_moe_grouped_prefers_triton_shape)
            and hidden_states.is_cuda
            and residual.is_cuda
            and not torch.is_grad_enabled()
            and qwen3_moe_grouped_prefers_triton_shape(
                int(hidden_states.shape[0]),
                int(selected_experts.shape[1]),
                self.hidden_dim,
                self.intermediate_dim,
            )
        ):
            raise RuntimeError("Qwen3 MoE residual fusion grouped decode is not eligible")

        try:
            global _QWEN3_MOE_GROUPED_LOGGED
            out = qwen3_moe_grouped_decode(
                hidden_states,
                self.gate_up_proj,
                self.down_proj,
                selected_experts,
                routing_weights,
                activation=self.hidden_act,
                out=residual,
                residual=residual,
                workspace=self._grouped_decode_workspace,
            )
            if _QWEN3_MOE_GROUPED_DEBUG and not _QWEN3_MOE_GROUPED_LOGGED:
                cfg = qwen3_moe_grouped_runtime_config() if callable(qwen3_moe_grouped_runtime_config) else {}
                print(f"[MegaGemm][Qwen3MoE] grouped decode active: {cfg}")
                _QWEN3_MOE_GROUPED_LOGGED = True
            self._grouped_decode_hits += 1
            return out
        except Exception as exc:
            msg = str(exc)
            if "residual fusion" not in msg:
                self._grouped_decode_disabled = True
                self._grouped_decode_fail_reason = msg
            if _QWEN3_MOE_GROUPED_DEBUG:
                print(f"[MegaGemm][Qwen3MoE] grouped decode residual fusion disabled: {exc}")
            raise

    def _forward_sorted_prefill(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill path grouped by expert without per-expert GPU syncs."""
        rows, top_k = selected_experts.shape
        assignments = int(rows) * int(top_k)
        flat_experts = selected_experts.reshape(-1).to(torch.int64)
        flat_route = routing_weights.reshape(-1)
        token_ids = self._prefill_token_ids(int(rows), int(top_k), hidden_states.device)

        order = torch.argsort(flat_experts)
        counts = torch.bincount(flat_experts, minlength=self.num_experts)
        counts_cpu = counts.detach().cpu().tolist()

        sorted_tokens = token_ids.index_select(0, order)
        sorted_route = flat_route.index_select(0, order)
        final_hidden_states = torch.zeros_like(hidden_states)

        cursor = 0
        for expert_idx, count in enumerate(counts_cpu):
            count = int(count)
            if count <= 0:
                continue
            end = cursor + count
            token_idx = sorted_tokens[cursor:end]
            current_state = hidden_states.index_select(0, token_idx)
            gate_up = torch.nn.functional.linear(
                current_state,
                self.gate_up_proj[expert_idx],
            )
            gate, up = gate_up.chunk(2, dim=-1)
            current_hidden_states = self._activation(gate) * up
            current_hidden_states = torch.nn.functional.linear(
                current_hidden_states,
                self.down_proj[expert_idx],
            )
            current_hidden_states = current_hidden_states * sorted_route[cursor:end, None]
            final_hidden_states.index_add_(
                0,
                token_idx,
                current_hidden_states.to(final_hidden_states.dtype),
            )
            cursor = end

        if cursor != assignments:
            raise RuntimeError(
                f"sorted Qwen3 MoE prefill consumed {cursor} assignments, expected {assignments}"
            )
        return final_hidden_states

    def _forward_batched_prefill(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        gate_up_proj: Optional[torch.Tensor] = None,
        down_proj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Prefill MoE via two strided batched GEMMs over padded expert groups."""
        gate_up_weight = self.gate_up_proj if gate_up_proj is None else gate_up_proj
        down_weight = self.down_proj if down_proj is None else down_proj
        rows, top_k = selected_experts.shape
        assignments = int(rows) * int(top_k)
        flat_experts = selected_experts.reshape(-1).to(torch.int64)
        flat_route = routing_weights.reshape(-1)
        token_ids = self._prefill_token_ids(int(rows), int(top_k), hidden_states.device)

        order = torch.argsort(flat_experts)
        sorted_experts = flat_experts.index_select(0, order)
        sorted_tokens = token_ids.index_select(0, order)
        sorted_route = flat_route.index_select(0, order)

        counts = torch.bincount(flat_experts, minlength=self.num_experts)
        max_count = int(counts.max().item())
        if max_count <= 0:
            return torch.zeros_like(hidden_states)

        starts = torch.cumsum(counts, dim=0) - counts
        assignment_pos = torch.arange(assignments, device=hidden_states.device, dtype=torch.int64)
        ranks = assignment_pos - starts.index_select(0, sorted_experts)
        padded_offsets = sorted_experts * max_count + ranks
        padded_size = int(self.num_experts) * int(max_count)

        token_pad = torch.zeros(padded_size, device=hidden_states.device, dtype=torch.int64)
        route_pad = torch.zeros(padded_size, device=hidden_states.device, dtype=flat_route.dtype)
        token_pad.scatter_(0, padded_offsets, sorted_tokens)
        route_pad.scatter_(0, padded_offsets, sorted_route)

        x = hidden_states.index_select(0, token_pad).reshape(
            self.num_experts,
            max_count,
            self.hidden_dim,
        )
        gate_up = torch.bmm(x, gate_up_weight.transpose(1, 2))
        gate, up = gate_up.chunk(2, dim=-1)
        activated = self._activation(gate) * up
        projected = torch.bmm(activated, down_weight.transpose(1, 2))
        projected.mul_(route_pad.reshape(self.num_experts, max_count, 1).to(projected.dtype))

        projected_valid = projected.reshape(padded_size, self.hidden_dim).index_select(
            0,
            padded_offsets,
        )
        final_hidden_states = torch.zeros_like(hidden_states)
        final_hidden_states.index_add_(
            0,
            sorted_tokens,
            projected_valid.to(final_hidden_states.dtype),
        )
        return final_hidden_states

    def _forward_int8_dequant_prefill(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        if not self._has_int8_experts():
            raise RuntimeError("Qwen3 MoE INT8 expert buffers are not available")
        rows, top_k = selected_experts.shape
        assignments = int(rows) * int(top_k)
        gate_up_weight, down_weight = self._dequant_all_expert_weights(hidden_states.dtype)
        try:
            if (
                _USE_QWEN3_MOE_SEGMENTED_PREFILL
                and assignments >= _QWEN3_MOE_SEGMENTED_PREFILL_MIN_ASSIGNMENTS
                and qwen3_moe_segmented_prefill is not None
                and callable(qwen3_moe_segmented_prefers_triton_shape)
                and qwen3_moe_segmented_prefers_triton_shape(
                    int(hidden_states.shape[0]),
                    int(selected_experts.shape[1]),
                    self.hidden_dim,
                    self.intermediate_dim,
                )
            ):
                self._segmented_prefill_hits += 1
                return self._forward_segmented_prefill(
                    hidden_states,
                    selected_experts,
                    routing_weights,
                    gate_up_proj=gate_up_weight,
                    down_proj=down_weight,
                )

            if (
                _USE_QWEN3_MOE_BATCHED_PREFILL
                and assignments >= _QWEN3_MOE_BATCHED_PREFILL_MIN_ASSIGNMENTS
            ):
                self._batched_prefill_hits += 1
                return self._forward_batched_prefill(
                    hidden_states,
                    selected_experts,
                    routing_weights,
                    gate_up_proj=gate_up_weight,
                    down_proj=down_weight,
                )
        finally:
            del gate_up_weight, down_weight

        raise RuntimeError(
            "Qwen3 MoE INT8 dequant prefill had no eligible segmented/batched path"
        )

    def _forward_bucketed_prefill(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill MoE via padded bmm buckets to reduce max-count waste."""
        rows, top_k = selected_experts.shape
        flat_experts = selected_experts.reshape(-1).to(torch.int64)
        flat_route = routing_weights.reshape(-1)
        token_ids = self._prefill_token_ids(int(rows), int(top_k), hidden_states.device)

        order = torch.argsort(flat_experts)
        sorted_experts = flat_experts.index_select(0, order)
        sorted_tokens = token_ids.index_select(0, order)
        sorted_route = flat_route.index_select(0, order)

        counts = torch.bincount(flat_experts, minlength=self.num_experts)
        counts_cpu = counts.detach().cpu().tolist()
        if not any(int(count) > 0 for count in counts_cpu):
            return torch.zeros_like(hidden_states)

        bucket_size = int(_QWEN3_MOE_BUCKETED_PREFILL_BUCKET_SIZE)
        buckets: dict[int, list[int]] = {}
        for expert_idx, count in enumerate(counts_cpu):
            count = int(count)
            if count <= 0:
                continue
            bucket_cap = ((count + bucket_size - 1) // bucket_size) * bucket_size
            buckets.setdefault(int(bucket_cap), []).append(int(expert_idx))

        starts = torch.cumsum(counts, dim=0) - counts
        final_hidden_states = torch.zeros_like(hidden_states)
        arange_cache: dict[int, torch.Tensor] = {}
        workspace = self._bucketed_prefill_workspace
        padded_total = 0
        valid_total = 0
        bucket_launches = 0
        max_padded_size = max(
            int(len(expert_ids_cpu)) * int(bucket_cap)
            for bucket_cap, expert_ids_cpu in buckets.items()
        )
        token_pad_storage = workspace.get("bucket_token_pad")
        if (
            token_pad_storage is None
            or int(token_pad_storage.numel()) < max_padded_size
            or token_pad_storage.device != hidden_states.device
            or token_pad_storage.dtype != torch.int64
        ):
            token_pad_storage = torch.empty(
                max_padded_size,
                device=hidden_states.device,
                dtype=torch.int64,
            )
            workspace["bucket_token_pad"] = token_pad_storage
        route_pad_storage = workspace.get("bucket_route_pad")
        if (
            route_pad_storage is None
            or int(route_pad_storage.numel()) < max_padded_size
            or route_pad_storage.device != hidden_states.device
            or route_pad_storage.dtype != flat_route.dtype
        ):
            route_pad_storage = torch.empty(
                max_padded_size,
                device=hidden_states.device,
                dtype=flat_route.dtype,
            )
            workspace["bucket_route_pad"] = route_pad_storage

        for bucket_cap, expert_ids_cpu in sorted(buckets.items()):
            group_count = int(len(expert_ids_cpu))
            if group_count <= 0 or bucket_cap <= 0:
                continue
            expert_ids = torch.tensor(
                expert_ids_cpu,
                device=hidden_states.device,
                dtype=torch.int64,
            )
            group_offsets = arange_cache.get(group_count)
            if group_offsets is None:
                group_offsets = torch.arange(
                    group_count,
                    device=hidden_states.device,
                    dtype=torch.int64,
                )
                arange_cache[group_count] = group_offsets

            expert_group_idx = workspace.get("bucket_expert_group_idx")
            if (
                expert_group_idx is None
                or tuple(expert_group_idx.shape) != (self.num_experts,)
                or expert_group_idx.device != hidden_states.device
                or expert_group_idx.dtype != torch.int64
            ):
                expert_group_idx = torch.empty(
                    (self.num_experts,),
                    device=hidden_states.device,
                    dtype=torch.int64,
                )
                workspace["bucket_expert_group_idx"] = expert_group_idx
            expert_group_idx.fill_(-1)
            expert_group_idx.scatter_(0, expert_ids, group_offsets)
            sorted_group_idx = expert_group_idx.index_select(0, sorted_experts)
            bucket_mask = sorted_group_idx.ge(0)
            bucket_positions = bucket_mask.nonzero(as_tuple=False).flatten()
            if bucket_positions.numel() == 0:
                continue

            bucket_experts = sorted_experts.index_select(0, bucket_positions)
            bucket_tokens = sorted_tokens.index_select(0, bucket_positions)
            bucket_route = sorted_route.index_select(0, bucket_positions)
            bucket_group_idx = sorted_group_idx.index_select(0, bucket_positions)
            bucket_ranks = bucket_positions - starts.index_select(0, bucket_experts)
            padded_offsets = bucket_group_idx * int(bucket_cap) + bucket_ranks
            padded_size = group_count * int(bucket_cap)
            valid_total += int(bucket_positions.numel())
            padded_total += int(padded_size)
            bucket_launches += 1

            token_pad = token_pad_storage[:padded_size]
            token_pad.zero_()
            route_pad = route_pad_storage[:padded_size]
            route_pad.zero_()
            token_pad.scatter_(0, padded_offsets, bucket_tokens)
            route_pad.scatter_(0, padded_offsets, bucket_route)

            x = hidden_states.index_select(0, token_pad).reshape(
                group_count,
                int(bucket_cap),
                self.hidden_dim,
            )
            gate_up_weight = self.gate_up_proj.index_select(0, expert_ids)
            down_weight = self.down_proj.index_select(0, expert_ids)
            gate_up = torch.bmm(x, gate_up_weight.transpose(1, 2))
            gate, up = gate_up.chunk(2, dim=-1)
            activated = self._activation(gate) * up
            projected = torch.bmm(activated, down_weight.transpose(1, 2))
            projected.mul_(
                route_pad.reshape(group_count, int(bucket_cap), 1).to(projected.dtype)
            )

            projected_valid = projected.reshape(padded_size, self.hidden_dim).index_select(
                0,
                padded_offsets,
            )
            final_hidden_states.index_add_(
                0,
                bucket_tokens,
                projected_valid.to(final_hidden_states.dtype),
            )

        self._bucketed_prefill_valid_assignments += int(valid_total)
        self._bucketed_prefill_padded_assignments += int(padded_total)
        self._bucketed_prefill_bucket_launches += int(bucket_launches)
        return final_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        use_grouped_decode: bool = False,
        *,
        post_moe_shared: Optional[torch.Tensor] = None,
        post_moe_shared_weight: Optional[torch.Tensor] = None,
        post_moe_expert_weight: Optional[torch.Tensor] = None,
        post_moe_final_weight: Optional[torch.Tensor] = None,
        post_moe_residual: Optional[torch.Tensor] = None,
        post_moe_out: Optional[torch.Tensor] = None,
        post_moe_wait_event: Optional[object] = None,
        post_moe_layer_scalar: Optional[torch.Tensor] = None,
        post_moe_next_norm_weight: Optional[torch.Tensor] = None,
        post_moe_next_norm_out: Optional[torch.Tensor] = None,
        post_moe_write_next_norm: bool = False,
        post_moe_eps: float = 1e-6,
        graph_safe_prefill: bool = False,
        compact_route_prepacked: bool = False,
    ) -> torch.Tensor:
        post_moe_values = (
            post_moe_shared,
            post_moe_shared_weight,
            post_moe_expert_weight,
            post_moe_final_weight,
            post_moe_residual,
        )
        fused_post_moe_requested = any(value is not None for value in post_moe_values)
        if fused_post_moe_requested and not all(
            value is not None for value in post_moe_values
        ):
            raise ValueError("fused Gemma4 post-MoE inputs must be provided together")
        if fused_post_moe_requested and post_moe_out is None:
            raise ValueError("fused Gemma4 post-MoE requires a persistent output buffer")
        next_norm_values = (
            post_moe_layer_scalar,
            post_moe_next_norm_weight,
            post_moe_next_norm_out,
        )
        fused_next_norm_requested = any(
            value is not None for value in next_norm_values
        )
        if fused_next_norm_requested and not all(
            value is not None for value in next_norm_values
        ):
            raise ValueError(
                "fused Gemma4 layer-scalar/next-norm inputs must be provided together"
            )
        if fused_next_norm_requested and not fused_post_moe_requested:
            raise ValueError(
                "fused Gemma4 layer-scalar/next-norm requires fused post-MoE"
            )
        if hidden_states.numel() == 0:
            return torch.zeros_like(hidden_states)

        has_int8_experts = self._has_int8_experts()
        has_awq_experts = self._has_awq_experts()
        grouped_decode_kernel = (
            qwen3_moe_grouped_decode_int8 if has_int8_experts else qwen3_moe_grouped_decode
        )
        gemma4_batch_policy = bool(
            self._gemma4_batch_decode_compact
            and not has_int8_experts
            and not has_awq_experts
        )
        grouped_max_assignments = (
            self._gemma4_batch_decode_max_assignments
            if gemma4_batch_policy
            else None
        )
        deterministic_batch_reduce = _deterministic_moe_reduce_requested(
            gemma4_batch_policy
            and self._gemma4_batch_decode_deterministic_reduce
        )
        if (
            use_grouped_decode
            and _USE_QWEN3_MOE_GROUPED_DECODE
            and not self._grouped_decode_disabled
            and grouped_decode_kernel is not None
            and callable(qwen3_moe_grouped_prefers_triton_shape)
            and hidden_states.is_cuda
            and not torch.is_grad_enabled()
            and qwen3_moe_grouped_prefers_triton_shape(
                int(hidden_states.shape[0]),
                int(selected_experts.shape[1]),
                self.hidden_dim,
                self.intermediate_dim,
                max_assignments=grouped_max_assignments,
            )
        ):
            try:
                global _QWEN3_MOE_GROUPED_LOGGED
                if fused_post_moe_requested and (has_int8_experts or has_awq_experts):
                    raise RuntimeError("fused Gemma4 post-MoE requires BF16 experts")
                final_hidden_states = (
                    post_moe_out
                    if fused_post_moe_requested
                    else self._grouped_decode_out_buffer(hidden_states)
                )
                if has_int8_experts:
                    out = qwen3_moe_grouped_decode_int8(
                        hidden_states,
                        self.gate_up_int8,
                        self.gate_up_scale,
                        self.down_int8,
                        self.down_scale,
                        selected_experts,
                        routing_weights,
                        activation=self.hidden_act,
                        out=final_hidden_states,
                        workspace=self._grouped_decode_workspace,
                    )
                else:
                    out = qwen3_moe_grouped_decode(
                        hidden_states,
                        self.gate_up_proj,
                        self.down_proj,
                        selected_experts,
                        routing_weights,
                        activation=self.hidden_act,
                        out=final_hidden_states,
                        workspace=self._grouped_decode_workspace,
                        max_assignments=grouped_max_assignments,
                        expert_grouped_compact=(
                            self._gemma4_batch_decode_use_compact
                            if gemma4_batch_policy
                            else None
                        ),
                        expert_grouped_min_rows=(
                            self._gemma4_batch_decode_compact_min_rows
                            if gemma4_batch_policy
                            else None
                        ),
                        expert_grouped_max_rows=(
                            self._gemma4_batch_decode_compact_max_rows
                            if gemma4_batch_policy
                            else None
                        ),
                        assignment_partial_reduce=deterministic_batch_reduce,
                        expert_grouped_compact_partial_reduce=(
                            True if deterministic_batch_reduce else None
                        ),
                        compact_route_prepacked=compact_route_prepacked,
                        post_moe_shared=post_moe_shared,
                        post_moe_shared_weight=post_moe_shared_weight,
                        post_moe_expert_weight=post_moe_expert_weight,
                        post_moe_final_weight=post_moe_final_weight,
                        post_moe_residual=post_moe_residual,
                        post_moe_wait_event=post_moe_wait_event,
                        post_moe_layer_scalar=post_moe_layer_scalar,
                        post_moe_next_norm_weight=post_moe_next_norm_weight,
                        post_moe_next_norm_out=post_moe_next_norm_out,
                        post_moe_write_next_norm=post_moe_write_next_norm,
                        post_moe_eps=post_moe_eps,
                    )
                if _QWEN3_MOE_GROUPED_DEBUG and not _QWEN3_MOE_GROUPED_LOGGED:
                    cfg = qwen3_moe_grouped_runtime_config() if callable(qwen3_moe_grouped_runtime_config) else {}
                    print(f"[MegaGemm][Qwen3MoE] grouped decode active: {cfg}")
                    _QWEN3_MOE_GROUPED_LOGGED = True
                self._grouped_decode_hits += 1
                return out
            except Exception as exc:
                self._grouped_decode_disabled = True
                self._grouped_decode_fail_reason = str(exc)
                if _QWEN3_MOE_GROUPED_DEBUG:
                    print(f"[MegaGemm][Qwen3MoE] grouped decode disabled: {exc}")
                if fused_post_moe_requested:
                    raise

        if fused_post_moe_requested:
            reason = self._grouped_decode_fail_reason or "grouped decode is unavailable"
            raise RuntimeError(
                f"fused Gemma4 post-MoE cannot fall back: {reason}"
            )

        assignments = int(hidden_states.shape[0]) * int(selected_experts.shape[1])
        if not use_grouped_decode:
            self._gemma4_long_dominant_expert_prefill_last_active = False
            self._gemma4_long_dominant_expert_prefill_last_guard_reason = ""
        if (
            not use_grouped_decode
            and not has_int8_experts
            and not has_awq_experts
            and self._gemma4_long_dominant_expert_prefill_is_enabled(
                hidden_states,
                selected_experts,
                routing_weights,
                graph_safe_prefill=graph_safe_prefill,
            )
        ):
            try:
                return self._forward_gemma4_long_dominant_expert_prefill(
                    hidden_states,
                    selected_experts,
                    routing_weights,
                )
            except Exception as exc:
                guard_rejected = self._record_gemma4_long_dominant_expert_failure(
                    exc
                )
                if _QWEN3_MOE_GROUPED_DEBUG:
                    status = "guard fallback" if guard_rejected else "disabled"
                    print(
                        "[MegaGemm][Gemma4MoE] long dominant-expert prefill "
                        f"{status}: {exc}"
                    )

        if (
            not use_grouped_decode
            and not has_int8_experts
            and not has_awq_experts
            and self._gemma4_long_padded_bmm_prefill_is_enabled(
                hidden_states,
                selected_experts,
                routing_weights,
                graph_safe_prefill=graph_safe_prefill,
            )
        ):
            try:
                return self._forward_gemma4_long_padded_bmm_prefill(
                    hidden_states,
                    selected_experts,
                    routing_weights,
                )
            except Exception as exc:
                self._gemma4_long_padded_bmm_prefill_disabled = True
                self._gemma4_long_padded_bmm_prefill_fail_reason = str(exc)
                self._gemma4_long_padded_bmm_prefill_last_active = False
                if _QWEN3_MOE_GROUPED_DEBUG:
                    print(
                        "[MegaGemm][Gemma4MoE] long padded-BMM prefill disabled: "
                        f"{exc}"
                    )

        if (
            not use_grouped_decode
            and has_int8_experts
            and _USE_QWEN3_MOE_INT8_DEQUANT_PREFILL
            and not self._int8_dequant_prefill_disabled
            and assignments >= _QWEN3_MOE_INT8_DEQUANT_PREFILL_MIN_ASSIGNMENTS
            and hidden_states.is_cuda
            and selected_experts.is_cuda
            and routing_weights.is_cuda
            and not torch.is_grad_enabled()
        ):
            try:
                out = self._forward_int8_dequant_prefill(
                    hidden_states,
                    selected_experts,
                    routing_weights,
                )
                self._int8_dequant_prefill_hits += 1
                return out
            except Exception as exc:
                self._int8_dequant_prefill_disabled = True
                self._int8_dequant_prefill_fail_reason = str(exc)
                if _QWEN3_MOE_GROUPED_DEBUG:
                    print(f"[MegaGemm][Qwen3MoE] INT8 dequant prefill disabled: {exc}")

        if (
            not use_grouped_decode
            and not graph_safe_prefill
            and not has_int8_experts
            and not has_awq_experts
            and self._gemma4_grouped_mm_prefill_is_enabled(
                hidden_states,
                selected_experts,
                routing_weights,
            )
        ):
            try:
                return self._forward_gemma4_grouped_mm_prefill(
                    hidden_states,
                    selected_experts,
                    routing_weights,
                )
            except Exception as exc:
                self._gemma4_grouped_mm_prefill_disabled = True
                self._gemma4_grouped_mm_prefill_fail_reason = str(exc)
                self._gemma4_grouped_mm_prefill_last_active = False
                if _QWEN3_MOE_GROUPED_DEBUG:
                    print(
                        "[MegaGemm][Gemma4MoE] grouped-MM prefill disabled: "
                        f"{exc}"
                    )

        if (
            not use_grouped_decode
            and not has_int8_experts
            and not has_awq_experts
            and self._segmented_prefill_is_enabled(assignments)
            and not self._segmented_prefill_disabled
            and qwen3_moe_segmented_prefill is not None
            and hidden_states.is_cuda
            and selected_experts.is_cuda
            and routing_weights.is_cuda
            and not torch.is_grad_enabled()
            and self._segmented_prefill_prefers_shape(
                int(hidden_states.shape[0]),
                int(selected_experts.shape[1]),
            )
        ):
            try:
                out = self._forward_segmented_prefill(
                    hidden_states,
                    selected_experts,
                    routing_weights,
                    graph_safe_prefill=graph_safe_prefill,
                )
                self._segmented_prefill_hits += 1
                return out
            except Exception as exc:
                self._segmented_prefill_disabled = True
                self._segmented_prefill_fail_reason = str(exc)
                if _QWEN3_MOE_GROUPED_DEBUG:
                    print(f"[MegaGemm][Qwen3MoE] segmented prefill disabled: {exc}")

        if (
            not use_grouped_decode
            and not has_int8_experts
            and not has_awq_experts
            and _USE_QWEN3_MOE_BUCKETED_PREFILL
            and not self._bucketed_prefill_disabled
            and assignments >= _QWEN3_MOE_BUCKETED_PREFILL_MIN_ASSIGNMENTS
            and hidden_states.is_cuda
            and selected_experts.is_cuda
            and routing_weights.is_cuda
            and not torch.is_grad_enabled()
        ):
            try:
                out = self._forward_bucketed_prefill(
                    hidden_states,
                    selected_experts,
                    routing_weights,
                )
                self._bucketed_prefill_hits += 1
                return out
            except Exception as exc:
                self._bucketed_prefill_disabled = True
                self._bucketed_prefill_fail_reason = str(exc)
                if _QWEN3_MOE_GROUPED_DEBUG:
                    print(f"[MegaGemm][Qwen3MoE] bucketed prefill disabled: {exc}")

        if (
            not use_grouped_decode
            and not has_int8_experts
            and not has_awq_experts
            and _USE_QWEN3_MOE_BATCHED_PREFILL
            and not self._batched_prefill_disabled
            and assignments >= _QWEN3_MOE_BATCHED_PREFILL_MIN_ASSIGNMENTS
            and hidden_states.is_cuda
            and selected_experts.is_cuda
            and routing_weights.is_cuda
            and not torch.is_grad_enabled()
        ):
            try:
                out = self._forward_batched_prefill(
                    hidden_states,
                    selected_experts,
                    routing_weights,
                )
                self._batched_prefill_hits += 1
                return out
            except Exception as exc:
                self._batched_prefill_disabled = True
                self._batched_prefill_fail_reason = str(exc)
                if _QWEN3_MOE_GROUPED_DEBUG:
                    print(f"[MegaGemm][Qwen3MoE] batched prefill disabled: {exc}")

        if (
            not use_grouped_decode
            and not has_int8_experts
            and not has_awq_experts
            and _USE_QWEN3_MOE_SORTED_PREFILL
            and not self._sorted_prefill_disabled
            and assignments >= _QWEN3_MOE_SORTED_PREFILL_MIN_ASSIGNMENTS
            and hidden_states.is_cuda
            and selected_experts.is_cuda
            and routing_weights.is_cuda
            and not torch.is_grad_enabled()
        ):
            try:
                out = self._forward_sorted_prefill(
                    hidden_states,
                    selected_experts,
                    routing_weights,
                )
                self._sorted_prefill_hits += 1
                return out
            except Exception as exc:
                self._sorted_prefill_disabled = True
                self._sorted_prefill_fail_reason = str(exc)
                if _QWEN3_MOE_GROUPED_DEBUG:
                    print(f"[MegaGemm][Qwen3MoE] sorted prefill disabled: {exc}")

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                selected_experts,
                num_classes=self.num_experts,
            ).permute(2, 1, 0)
            expert_hit = torch.nonzero(
                expert_mask.sum(dim=(-1, -2)) > 0,
                as_tuple=False,
            ).flatten().detach().cpu().tolist()

        for expert_idx in expert_hit:
            expert_idx = int(expert_idx)
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            current_state = hidden_states[token_idx]
            if has_int8_experts:
                gate_up_weight = self._dequant_expert_weight(
                    self.gate_up_int8,
                    self.gate_up_scale,
                    expert_idx,
                    current_state.dtype,
                )
            elif has_awq_experts:
                gate_up_weight = self._dequant_awq_expert_weight(
                    self.gate_up_qweight,
                    self.gate_up_scales,
                    self.gate_up_qzeros,
                    expert_idx,
                    current_state.dtype,
                ).t().contiguous()
            else:
                gate_up_weight = self.gate_up_proj[expert_idx]
            gate_up = torch.nn.functional.linear(current_state, gate_up_weight)
            gate, up = gate_up.chunk(2, dim=-1)
            current_hidden_states = self._activation(gate) * up
            if has_int8_experts:
                down_weight = self._dequant_expert_weight(
                    self.down_int8,
                    self.down_scale,
                    expert_idx,
                    current_hidden_states.dtype,
                )
            elif has_awq_experts:
                down_weight = self._dequant_awq_expert_weight(
                    self.down_qweight,
                    self.down_scales,
                    self.down_qzeros,
                    expert_idx,
                    current_hidden_states.dtype,
                ).t().contiguous()
            else:
                down_weight = self.down_proj[expert_idx]
            current_hidden_states = torch.nn.functional.linear(current_hidden_states, down_weight)
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class Qwen3MoeMLP(nn.Module):
    """Sparse Qwen3 MoE FFN block, matching HF Qwen3MoeSparseMoeBlock."""

    is_moe = True

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_dim = int(config.hidden_size)
        self.intermediate_size = int(config.moe_intermediate_size or config.intermediate_size)
        self.hidden_act = config.hidden_act
        self._awq_separate = False
        self.gate = Qwen3MoeTopKRouter(config)
        self.experts = Qwen3MoeExperts(config)

    def forward(
        self,
        x: torch.Tensor,
        timing_events: Optional[dict] = None,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        input_shape = x.shape[:-1]
        hidden_2d = x.reshape(-1, self.hidden_dim)
        do_timing = timing_events is not None and hidden_2d.is_cuda

        router_start_end = _timing_record_start(do_timing)
        _, routing_weights, selected_experts = self.gate(hidden_2d)
        _timing_record_end(timing_events, "moe_router", router_start_end)

        experts_start_end = _timing_record_start(do_timing)
        out = self.experts(
            hidden_2d,
            selected_experts,
            routing_weights,
            use_grouped_decode=not is_prefill,
        )
        _timing_record_end(timing_events, "moe_experts", experts_start_end)
        return out.reshape(*input_shape, self.hidden_dim)

    def forward_decode(
        self,
        x: torch.Tensor,
        input_is_normed: bool = True,
        input_norm_weight: Optional[torch.Tensor] = None,
        input_norm_eps: float = 1e-6,
        input_norm_offset: bool = False,
        timing_events: Optional[dict] = None,
    ) -> torch.Tensor:
        if not input_is_normed and input_norm_weight is not None:
            x = _decode_rmsnorm(
                x,
                input_norm_weight,
                input_norm_eps,
                input_norm_offset,
            )
        return self.forward(x, timing_events=timing_events, is_prefill=False)

    def forward_prefill_add_residual(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        timing_events: Optional[dict] = None,
    ) -> torch.Tensor:
        input_shape = x.shape[:-1]
        hidden_2d = x.reshape(-1, self.hidden_dim)
        residual_2d = residual.reshape(-1, self.hidden_dim)
        do_timing = timing_events is not None and hidden_2d.is_cuda

        router_start_end = _timing_record_start(do_timing)
        _, routing_weights, selected_experts = self.gate(hidden_2d)
        _timing_record_end(timing_events, "moe_router", router_start_end)

        experts_start_end = _timing_record_start(do_timing)
        try:
            out = self.experts.forward_prefill_add_residual(
                hidden_2d,
                selected_experts,
                routing_weights,
                residual_2d,
            )
            _timing_record_end(timing_events, "moe_experts", experts_start_end)
            return out.reshape(*input_shape, self.hidden_dim)
        except Exception:
            _timing_record_end(timing_events, "moe_experts", experts_start_end)

        experts_start_end = _timing_record_start(do_timing)
        mlp_out = self.experts(
            hidden_2d,
            selected_experts,
            routing_weights,
            use_grouped_decode=False,
        )
        _timing_record_end(timing_events, "moe_experts", experts_start_end)
        residual_2d.add_(mlp_out)
        return residual

    def forward_decode_add_residual(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        input_is_normed: bool = True,
        input_norm_weight: Optional[torch.Tensor] = None,
        input_norm_eps: float = 1e-6,
        input_norm_offset: bool = False,
        timing_events: Optional[dict] = None,
    ) -> torch.Tensor:
        if not input_is_normed and input_norm_weight is not None:
            x = _decode_rmsnorm(
                x,
                input_norm_weight,
                input_norm_eps,
                input_norm_offset,
            )

        input_shape = x.shape[:-1]
        hidden_2d = x.reshape(-1, self.hidden_dim)
        residual_2d = residual.reshape(-1, self.hidden_dim)
        do_timing = timing_events is not None and hidden_2d.is_cuda

        router_start_end = _timing_record_start(do_timing)
        _, routing_weights, selected_experts = self.gate(hidden_2d)
        _timing_record_end(timing_events, "moe_router", router_start_end)

        experts_start_end = _timing_record_start(do_timing)
        try:
            out = self.experts.forward_decode_add_residual(
                hidden_2d,
                selected_experts,
                routing_weights,
                residual_2d,
            )
            _timing_record_end(timing_events, "moe_experts", experts_start_end)
            return out.reshape(*input_shape, self.hidden_dim)
        except Exception:
            _timing_record_end(timing_events, "moe_experts", experts_start_end)

        experts_start_end = _timing_record_start(do_timing)
        mlp_out = self.experts(
            hidden_2d,
            selected_experts,
            routing_weights,
            use_grouped_decode=True,
        )
        _timing_record_end(timing_events, "moe_experts", experts_start_end)
        residual_2d.add_(mlp_out)
        return residual


class Gemma4MoeTopKRouter(nn.Module):
    """Gemma 4 MoE router matching the HF text router semantics."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.top_k = int(config.num_experts_per_tok)
        self.num_experts = int(config.num_experts)
        self.hidden_dim = int(config.hidden_size)
        self.norm_topk_prob = bool(config.norm_topk_prob)
        self.input_norm = MGRMSNorm(
            self.hidden_dim,
            eps=config.rms_norm_eps,
            offset=False,
            with_scale=False,
        )
        self.proj = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.scale = nn.Parameter(
            torch.ones(self.hidden_dim, dtype=torch.get_default_dtype())
        )
        self.per_expert_scale = nn.Parameter(
            torch.ones(self.num_experts, dtype=torch.get_default_dtype())
        )
        self.scalar_root = self.hidden_dim ** -0.5
        # CUDA graphs capture raw workspace pointers. Keep one allocation per
        # phase and row shape so a later prefill or batch size cannot free
        # memory still referenced by a cached graph.
        self._decode_topk_workspaces: dict[int, dict[str, torch.Tensor]] = {}
        self._prefill_topk_workspaces: dict[int, dict[str, torch.Tensor]] = {}
        self._decode_router_logits_by_rows: dict[int, torch.Tensor] = {}
        self._prefill_router_logits_by_rows: dict[int, torch.Tensor] = {}
        self._fused_norm_scale_hits = 0
        self._fused_topk_expert_scale_hits = 0
        self._fused_prefill_hits = 0
        self._fused_prefill_checked = False
        self._fused_prefill_checked_by_rows: set[int] = set()
        self._fused_prefill_runtime_by_rows: dict[int, bool] = {}
        self._fused_prefill_disabled = False
        self._fused_prefill_error = ""
        self._fused_decode_hits = 0
        self._fused_decode_selected = bool(_GEMMA4_FUSED_MOE_ROUTER_DECODE)
        self._fused_decode_disabled = False
        self._fused_decode_error = ""
        self._fused_decode_last_path = ""
        self._compact_route_pack_hits = 0
        self._compact_route_pack_disabled = False
        self._compact_route_pack_error = ""
        self._compact_route_pack_last_active = False

    def set_fused_prefill_runtime(self, rows: int, enabled: bool) -> None:
        self._fused_prefill_runtime_by_rows[int(rows)] = bool(enabled)

    def _logits_buffer(
        self,
        hidden_2d: torch.Tensor,
        *,
        is_prefill: bool,
    ) -> torch.Tensor:
        shape = (int(hidden_2d.shape[0]), self.num_experts)
        cache = (
            self._prefill_router_logits_by_rows
            if is_prefill
            else self._decode_router_logits_by_rows
        )
        logits = cache.get(shape[0])
        if (
            logits is None
            or tuple(logits.shape) != shape
            or logits.device != hidden_2d.device
            or logits.dtype != hidden_2d.dtype
        ):
            logits = torch.empty(shape, device=hidden_2d.device, dtype=hidden_2d.dtype)
            cache[shape[0]] = logits
        return logits

    def _topk_workspace_for(
        self,
        hidden_2d: torch.Tensor,
        *,
        is_prefill: bool,
    ) -> dict[str, torch.Tensor]:
        rows = int(hidden_2d.shape[0])
        cache = (
            self._prefill_topk_workspaces
            if is_prefill
            else self._decode_topk_workspaces
        )
        workspace = cache.get(rows)
        if workspace is None:
            workspace = {}
            cache[rows] = workspace
        return workspace

    def _normalized_router_input(
        self,
        hidden_2d: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        if (
            rmsnorm_triton_scaled_no_weight is not None
            and hidden_2d.is_cuda
            and not torch.is_grad_enabled()
            and int(scale.numel()) == self.hidden_dim
        ):
            self._fused_norm_scale_hits += 1
            return rmsnorm_triton_scaled_no_weight(
                hidden_2d,
                scale,
                self.input_norm.eps,
                self.scalar_root,
            )

        normalized = self.input_norm(hidden_2d)
        if int(scale.numel()) == self.hidden_dim:
            normalized = normalized.mul(scale.view(1, -1))
        elif int(scale.numel()) == 1:
            normalized = normalized.mul(scale.reshape(()))
        else:
            raise RuntimeError(
                f"Gemma4 MoE router.scale has unsupported shape {tuple(self.scale.shape)}; "
                f"expected hidden={self.hidden_dim} or scalar."
            )
        return normalized.mul(self.scalar_root)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        is_prefill: bool = False,
        normalized_router_input: Optional[torch.Tensor] = None,
        compact_route_workspace: Optional[dict[str, torch.Tensor]] = None,
        use_compact_route_pack: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_2d = hidden_states.reshape(-1, self.hidden_dim)
        self._compact_route_pack_last_active = False
        top_k = min(max(1, self.top_k), self.num_experts)
        topk_workspace = self._topk_workspace_for(
            hidden_2d,
            is_prefill=is_prefill,
        )
        weight, bias = _linear_weight_bias(self.proj)
        if normalized_router_input is None:
            scale = self.scale.to(
                device=hidden_2d.device,
                dtype=hidden_2d.dtype,
            ).reshape(-1)
            hidden_for_router = self._normalized_router_input(hidden_2d, scale)
        else:
            hidden_for_router = normalized_router_input.reshape(-1, self.hidden_dim)
            if (
                hidden_for_router.shape != hidden_2d.shape
                or hidden_for_router.device != hidden_2d.device
                or hidden_for_router.dtype != hidden_2d.dtype
            ):
                raise ValueError("pre-normalized router input must match hidden states")
        if (
            weight is not None
            and hidden_for_router.is_cuda
            and not torch.is_grad_enabled()
            and hidden_for_router.dtype in (torch.float16, torch.bfloat16, torch.float32)
        ):
            router_logits = self._logits_buffer(
                hidden_for_router,
                is_prefill=is_prefill,
            )
            torch.mm(hidden_for_router, weight.t(), out=router_logits)
        else:
            router_logits = torch.nn.functional.linear(hidden_for_router, weight, bias)

        expert_scale = self.per_expert_scale.to(
            device=router_logits.device,
            dtype=router_logits.dtype,
        ).reshape(-1)
        scale_fused_in_topk = bool(
            self.norm_topk_prob
            and qwen3_moe_topk_softmax is not None
            and int(expert_scale.numel()) == self.num_experts
        )
        use_fused_compact_pack = bool(
            use_compact_route_pack
            and not is_prefill
            and not self._compact_route_pack_disabled
            and self.norm_topk_prob
            and scale_fused_in_topk
            and callable(qwen3_moe_topk_softmax_compact_pack)
            and compact_route_workspace is not None
            and not int(
                compact_route_workspace.get(
                    "expert_grouped_compact_route_prepacked_disabled",
                    0,
                )
                or 0
            )
        )
        if use_fused_compact_pack:
            try:
                routing_weights, selected_experts = (
                    qwen3_moe_topk_softmax_compact_pack(
                        router_logits.contiguous(),
                        top_k,
                        workspace=topk_workspace,
                        compact_workspace=compact_route_workspace,
                        expert_scale=expert_scale,
                    )
                )
                self._compact_route_pack_hits += 1
                self._compact_route_pack_last_active = True
            except Exception as exc:
                self._compact_route_pack_disabled = True
                self._compact_route_pack_error = f"{type(exc).__name__}: {exc}"
        if (
            not self._compact_route_pack_last_active
            and self.norm_topk_prob
            and qwen3_moe_topk_softmax is not None
        ):
            routing_weights, selected_experts = qwen3_moe_topk_softmax(
                router_logits.contiguous(),
                top_k,
                workspace=topk_workspace,
                expert_scale=expert_scale if scale_fused_in_topk else None,
            )
            if scale_fused_in_topk and hidden_2d.is_cuda:
                self._fused_topk_expert_scale_hits += 1
        elif not self._compact_route_pack_last_active:
            router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float32, dim=-1)
            routing_weights, selected_experts = torch.topk(router_probs, top_k, dim=-1)
            if self.norm_topk_prob:
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        elif hidden_2d.is_cuda:
            self._fused_topk_expert_scale_hits += 1
        expert_scale = expert_scale.to(dtype=routing_weights.dtype)
        if int(expert_scale.numel()) == self.num_experts:
            if not scale_fused_in_topk:
                routing_weights = routing_weights * expert_scale[selected_experts]
        elif int(expert_scale.numel()) != 1:
            raise RuntimeError(
                "Gemma4 MoE router.per_expert_scale has unsupported shape "
                f"{tuple(self.per_expert_scale.shape)}; expected experts={self.num_experts} "
                "or scalar."
            )
        else:
            routing_weights = routing_weights * expert_scale.reshape(())
        return router_logits, routing_weights.to(router_logits.dtype), selected_experts

    def route(
        self,
        hidden_states: torch.Tensor,
        *,
        is_prefill: bool = True,
        normalized_router_input: Optional[torch.Tensor] = None,
        compact_route_workspace: Optional[dict[str, torch.Tensor]] = None,
        use_compact_route_pack: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_2d = hidden_states.reshape(-1, self.hidden_dim)
        self._compact_route_pack_last_active = False
        top_k = min(max(1, self.top_k), self.num_experts)
        weight, _ = _linear_weight_bias(self.proj)
        use_fused_prefill = bool(
            is_prefill
            and not self._fused_prefill_disabled
            and self._fused_prefill_runtime_by_rows.get(
                int(hidden_2d.shape[0]),
                int(hidden_2d.shape[0]) <= 32,
            )
            and _gemma4_a100_a4b_fused_router_prefill_shape(
                int(hidden_2d.shape[0]),
                self.hidden_dim,
                self.num_experts,
                top_k,
                hidden_2d.dtype,
                torch.cuda.get_device_name(hidden_2d.device)
                if hidden_2d.is_cuda
                else "",
            )
        )
        use_fused_decode = bool(
            not is_prefill
            and not use_compact_route_pack
            and self._fused_decode_selected
            and not self._fused_decode_disabled
            and _gemma4_a100_a4b_fused_router_decode_shape(
                int(hidden_2d.shape[0]),
                self.hidden_dim,
                self.num_experts,
                top_k,
                hidden_2d.dtype,
                torch.cuda.get_device_name(hidden_2d.device)
                if hidden_2d.is_cuda
                else "",
            )
        )
        if (
            (use_fused_prefill or use_fused_decode)
            and callable(gemma4_moe_prefill_router_topk)
            and weight is not None
            and hidden_2d.is_cuda
            and not torch.is_grad_enabled()
        ):
            scale = self.scale.to(
                device=hidden_2d.device,
                dtype=hidden_2d.dtype,
            ).reshape(-1)
            expert_scale = self.per_expert_scale.to(
                device=hidden_2d.device,
                dtype=hidden_2d.dtype,
            ).reshape(-1)
            try:
                hidden_for_router = (
                    self._normalized_router_input(hidden_2d, scale)
                    if normalized_router_input is None
                    else normalized_router_input.reshape(-1, self.hidden_dim)
                )
                routing_weights, selected_experts = gemma4_moe_prefill_router_topk(
                    hidden_for_router.contiguous(),
                    weight.contiguous(),
                    expert_scale.contiguous(),
                    top_k,
                    workspace=self._topk_workspace_for(
                        hidden_2d,
                        is_prefill=is_prefill,
                    ),
                )
                if (
                    use_fused_prefill
                    and int(hidden_2d.shape[0])
                    not in self._fused_prefill_checked_by_rows
                ):
                    candidate_weights = routing_weights.clone()
                    candidate_experts = selected_experts.clone()
                    _, reference_weights, reference_experts = self.forward(
                        hidden_states,
                        is_prefill=True,
                    )
                    experts_equal = torch.equal(candidate_experts, reference_experts)
                    weights_equal = torch.equal(
                        candidate_weights,
                        reference_weights,
                    )
                    if not (experts_equal and weights_equal):
                        self._fused_prefill_disabled = True
                        self._fused_prefill_error = "reference mismatch"
                        return reference_weights, reference_experts
                    self._fused_prefill_checked = True
                    self._fused_prefill_checked_by_rows.add(
                        int(hidden_2d.shape[0])
                    )
                    routing_weights = candidate_weights
                    selected_experts = candidate_experts
                if use_fused_prefill:
                    self._fused_prefill_hits += 1
                else:
                    self._fused_decode_hits += 1
                    self._fused_decode_last_path = "fused"
                return routing_weights, selected_experts
            except Exception as exc:
                if use_fused_prefill:
                    self._fused_prefill_disabled = True
                    self._fused_prefill_error = f"{type(exc).__name__}: {exc}"
                else:
                    self._fused_decode_disabled = True
                    self._fused_decode_error = f"{type(exc).__name__}: {exc}"

        if not is_prefill:
            self._fused_decode_last_path = "legacy"
        _, routing_weights, selected_experts = self.forward(
            hidden_states,
            is_prefill=is_prefill,
            normalized_router_input=normalized_router_input,
            compact_route_workspace=compact_route_workspace,
            use_compact_route_pack=use_compact_route_pack,
        )
        return routing_weights, selected_experts


class Gemma4MoeMLP(nn.Module):
    """Gemma 4 MoE FFN: dense shared MLP plus routed expert MLP."""

    is_moe = True

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_dim = int(config.hidden_size)
        self.shared_mlp = LlamaMLP(config, layer_idx)
        self.gate = Gemma4MoeTopKRouter(config)
        self.experts = Qwen3MoeExperts(config)

    def _routed(
        self,
        x: torch.Tensor,
        timing_events: Optional[dict] = None,
        *,
        is_prefill: bool,
        selected_experts: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
        route_input: Optional[torch.Tensor] = None,
        graph_safe_prefill: bool = False,
    ) -> torch.Tensor:
        input_shape = x.shape[:-1]
        hidden_2d = x.reshape(-1, self.hidden_dim)
        do_timing = timing_events is not None and hidden_2d.is_cuda
        if selected_experts is None or routing_weights is None:
            router_start_end = _timing_record_start(do_timing)
            router_hidden = hidden_2d if route_input is None else route_input.reshape(-1, self.hidden_dim)
            routing_weights, selected_experts = self.gate.route(
                router_hidden,
                is_prefill=is_prefill,
            )
            _timing_record_end(timing_events, "moe_router", router_start_end)

        experts_start_end = _timing_record_start(do_timing)
        out = self.experts(
            hidden_2d,
            selected_experts,
            routing_weights,
            use_grouped_decode=not is_prefill,
            graph_safe_prefill=graph_safe_prefill,
        )
        _timing_record_end(timing_events, "moe_experts", experts_start_end)
        return out.reshape(*input_shape, self.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        timing_events: Optional[dict] = None,
        is_prefill: bool = True,
        graph_safe_prefill: bool = False,
    ) -> torch.Tensor:
        shared = self.shared_mlp(x, timing_events=timing_events, is_prefill=is_prefill)
        routed = self._routed(
            x,
            timing_events=timing_events,
            is_prefill=is_prefill,
            graph_safe_prefill=graph_safe_prefill,
        )
        return shared + routed

    def forward_decode(
        self,
        x: torch.Tensor,
        input_is_normed: bool = True,
        input_norm_weight: Optional[torch.Tensor] = None,
        input_norm_eps: float = 1e-6,
        input_norm_offset: bool = False,
        timing_events: Optional[dict] = None,
    ) -> torch.Tensor:
        shared = self.shared_mlp.forward_decode(
            x,
            input_is_normed=input_is_normed,
            input_norm_weight=input_norm_weight,
            input_norm_eps=input_norm_eps,
            input_norm_offset=input_norm_offset,
            timing_events=timing_events,
        )
        routed = self._routed(x, timing_events=timing_events, is_prefill=False)
        return shared + routed

    def forward_prefill_add_residual(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        timing_events: Optional[dict] = None,
    ) -> torch.Tensor:
        out = self.forward(x, timing_events=timing_events, is_prefill=True)
        residual.add_(out)
        return residual

    def forward_decode_add_residual(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        input_is_normed: bool = True,
        input_norm_weight: Optional[torch.Tensor] = None,
        input_norm_eps: float = 1e-6,
        input_norm_offset: bool = False,
        timing_events: Optional[dict] = None,
    ) -> torch.Tensor:
        out = self.forward_decode(
            x,
            input_is_normed=input_is_normed,
            input_norm_weight=input_norm_weight,
            input_norm_eps=input_norm_eps,
            input_norm_offset=input_norm_offset,
            timing_events=timing_events,
        )
        residual.add_(out)
        return residual


class LlamaDecoderLayer(nn.Module):
    """Single transformer layer."""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_gemma4 = config.model_type == 'gemma4_text'
        self.is_moe_layer = config.is_moe_layer(layer_idx)
        self.layer_type = (
            config.layer_types[layer_idx]
            if config.layer_types and layer_idx < len(config.layer_types)
            else 'full_attention'
        )
        self.input_layernorm = MGRMSNorm(
            config.hidden_size, config.rms_norm_eps, offset=config.norm_offset
        )
        self.self_attn = None
        self.linear_attn = None
        if self.layer_type == 'linear_attention':
            self.linear_attn = GatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = LlamaAttention(config, layer_idx)
        self.post_attention_layernorm = MGRMSNorm(
            config.hidden_size, config.rms_norm_eps, offset=config.norm_offset
        )
        self.pre_feedforward_layernorm = None
        self.post_feedforward_layernorm = None
        self.pre_feedforward_layernorm_1 = None
        self.pre_feedforward_layernorm_2 = None
        self.post_feedforward_layernorm_1 = None
        self.post_feedforward_layernorm_2 = None
        if self.is_gemma4:
            self.pre_feedforward_layernorm = MGRMSNorm(
                config.hidden_size, config.rms_norm_eps, offset=False
            )
            self.post_feedforward_layernorm = MGRMSNorm(
                config.hidden_size, config.rms_norm_eps, offset=False
            )
            if self.is_moe_layer:
                self.pre_feedforward_layernorm_2 = MGRMSNorm(
                    config.hidden_size, config.rms_norm_eps, offset=False
                )
                self.post_feedforward_layernorm_1 = MGRMSNorm(
                    config.hidden_size, config.rms_norm_eps, offset=False
                )
                self.post_feedforward_layernorm_2 = MGRMSNorm(
                    config.hidden_size, config.rms_norm_eps, offset=False
                )
        self._gemma4_fused_dual_ffn_norm_hits = 0
        self._gemma4_fused_dual_ffn_norm_disabled = False
        self._gemma4_fused_dual_ffn_norm_error = ""
        self._gemma4_fused_add_ffn_norm_hits = 0
        self._gemma4_fused_add_ffn_norm_disabled = False
        self._gemma4_fused_add_ffn_norm_error = ""
        self._gemma4_fused_post_ffn_norm_hits = 0
        self._gemma4_fused_post_ffn_norm_checked = False
        self._gemma4_fused_post_ffn_norm_disabled = False
        self._gemma4_fused_post_ffn_norm_error = ""
        self._gemma4_prefill_attn_moe_bridge_runtime_by_rows: dict[int, bool] = {}
        self._gemma4_prefill_attn_moe_bridge_workspaces: dict[
            int, dict[str, torch.Tensor]
        ] = {}
        self._gemma4_fused_attn_moe_bridge_prefill_hits = 0
        self._gemma4_fused_attn_moe_router_bridge_prefill_hits = 0
        self._gemma4_prefill_attn_moe_bridge_error = ""
        self._gemma4_prefill_moe_tail_runtime_by_rows: dict[int, bool] = {}
        self._gemma4_fused_post_moe_norm_residual_prefill_hits = 0
        self._gemma4_prefill_moe_tail_error = ""
        self._gemma4_parallel_moe_prefill_hits = 0
        if self.is_moe_layer and config.model_type == 'gemma4_text':
            self.mlp = Gemma4MoeMLP(config, layer_idx)
        elif self.is_moe_layer:
            self.mlp = Qwen3MoeMLP(config)
        else:
            self.mlp = LlamaMLP(config, layer_idx)
        self.hidden_size_per_layer_input = (
            config.hidden_size_per_layer_input if self.is_gemma4 else 0
        )
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = nn.Linear(
                config.hidden_size, self.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = nn.Linear(
                self.hidden_size_per_layer_input, config.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = MGRMSNorm(
                config.hidden_size, config.rms_norm_eps, offset=False
            )
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None
        self._fast_ple_gate_out = None
        self._fast_ple_proj_out = None
        self._prefill_ple_gate_out = None
        self._prefill_ple_proj_out = None
        self.register_buffer("layer_scalar", torch.ones(1), persistent=self.is_gemma4)
        self._norm_offset = config.norm_offset
        self._norm_eps = config.rms_norm_eps
        self._use_fused_norm_qkv_decode = (
            _USE_FUSED_RMSNORM_QKV_DECODE
            and self.layer_type == 'full_attention'
            and not self.is_gemma4
        )
        self._use_fused_norm_gateup_decode = (
            _USE_FUSED_RMSNORM_GATEUP_DECODE
            and not self.is_gemma4
            and not self.is_moe_layer
        )

    def _gemma4_ple_linear(
        self,
        module: nn.Module,
        x: torch.Tensor,
        *,
        is_prefill: bool,
        decode_attr: str,
        prefill_attr: str,
    ) -> torch.Tensor:
        weight, bias = _linear_weight_bias(module)
        if weight is None:
            return module(x)
        if is_prefill:
            return _prefill_linear(self, prefill_attr, x, weight, bias)
        return _decode_linear(
            self,
            decode_attr,
            x,
            weight,
            bias,
            use_fast=_USE_DECODE_FAST_LINEAR,
        )

    def set_gemma4_prefill_moe_tail_runtime(
        self,
        rows: int,
        enabled: bool,
    ) -> None:
        self._gemma4_prefill_moe_tail_runtime_by_rows[int(rows)] = bool(enabled)

    def set_gemma4_prefill_attn_moe_bridge_runtime(
        self,
        rows: int,
        enabled: bool,
    ) -> None:
        self._gemma4_prefill_attn_moe_bridge_runtime_by_rows[int(rows)] = bool(
            enabled
        )

    def _gemma4_prefill_attn_moe_bridge_workspace_for(
        self,
        tensor: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        rows = int(tensor.numel()) // int(tensor.shape[-1])
        workspace = self._gemma4_prefill_attn_moe_bridge_workspaces.get(rows)
        expected_shape = tuple(tensor.shape)
        if workspace is None or any(
            tuple(value.shape) != expected_shape
            or value.device != tensor.device
            or value.dtype != tensor.dtype
            for value in workspace.values()
        ):
            workspace = {
                name: torch.empty_like(tensor)
                for name in ("post_norm", "shared", "expert", "router")
            }
            self._gemma4_prefill_attn_moe_bridge_workspaces[rows] = workspace
        return workspace

    def _gemma4_prefill_attn_moe_bridge_is_enabled(
        self,
        rows: int,
        attn_out: torch.Tensor,
        residual: torch.Tensor,
    ) -> bool:
        if not self._gemma4_prefill_attn_moe_bridge_runtime_by_rows.get(
            int(rows), False
        ):
            return False
        if (
            torch.is_grad_enabled()
            or not attn_out.is_cuda
            or not attn_out.is_contiguous()
            or not residual.is_contiguous()
        ):
            return False
        try:
            if torch.cuda.is_current_stream_capturing():
                return False
        except Exception:
            return False
        device_name = torch.cuda.get_device_name(attn_out.device)
        return _gemma4_a100_a4b_fused_attn_moe_bridge_prefill_shape(
            "gemma4_text" if self.is_gemma4 else "",
            rows,
            int(attn_out.shape[-1]),
            int(self.mlp.shared_mlp.intermediate_size),
            int(self.mlp.experts.intermediate_dim),
            attn_out.dtype,
            device_name,
        ) and float(self.mlp.gate.input_norm.eps) == float(
            self.post_attention_layernorm.eps
        ) and float(self.pre_feedforward_layernorm.eps) == float(
            self.post_attention_layernorm.eps
        ) and float(self.pre_feedforward_layernorm_2.eps) == float(
            self.post_attention_layernorm.eps
        )

    def _gemma4_prefill_moe_tail_is_enabled(
        self,
        rows: int,
        hidden_states: torch.Tensor,
        timing_events: Optional[dict],
    ) -> bool:
        if not self._gemma4_prefill_moe_tail_runtime_by_rows.get(int(rows), False):
            return False
        if timing_events is not None or torch.is_grad_enabled() or not hidden_states.is_cuda:
            return False
        try:
            if torch.cuda.is_current_stream_capturing():
                return False
        except Exception:
            return False
        device_name = torch.cuda.get_device_name(hidden_states.device)
        return _gemma4_a100_a4b_fused_moe_prefill_tail_shape(
            "gemma4_text" if self.is_gemma4 else "",
            rows,
            int(hidden_states.shape[-1]),
            int(self.mlp.shared_mlp.intermediate_size),
            int(self.mlp.experts.intermediate_dim),
            hidden_states.dtype,
            device_name,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        kv_cache=None,
        block_table=None,
        seq_lens=None,
        seq_lens_kv=None,
        decode_phys_blocks=None,
        decode_blk_offsets=None,
        is_prefill=True,
        attn_mask=None,
        cu_seqlens=None,
        linear_conv_state=None,
        linear_recurrent_state=None,
        linear_attention_mask=None,
        use_linear_cache=False,
        timing_events: Optional[dict] = None,
        append_kv_prefix: bool = False,
        per_layer_input: Optional[torch.Tensor] = None,
        input_is_normed: bool = True,
        input_norm_weight: Optional[torch.Tensor] = None,
        input_norm_eps: float = 1e-6,
        input_norm_offset: bool = False,
        implicit_causal_prefill: bool = False,
        gemma4_parallel_moe_prefill_stream=None,
        gemma4_parallel_moe_prefill_fork_event=None,
        gemma4_parallel_moe_prefill_join_event=None,
        graph_safe_prefill: bool = False,
        prefill_kv_out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if append_kv_prefix and self.layer_type != 'full_attention':
            raise NotImplementedError(
                "suffix prefill is currently only implemented for full-attention layers"
            )
        if self.is_gemma4:
            if append_kv_prefix:
                raise NotImplementedError("Gemma4 suffix prefill is not implemented yet")
            finite_trace = (
                getattr(self, "_gemma4_prefill_finite_trace", None)
                if is_prefill else None
            )
            if finite_trace is not None:
                _record_gemma4_prefill_finite_trace(
                    finite_trace, self.layer_idx, "layer.input", hidden_states
                )
            next_linear_conv = None
            next_linear_recurrent = None
            do_timing = (
                timing_events is not None
                and not is_prefill
                and hidden_states.is_cuda
                and _timing_enabled()
            )
            do_prefill_stage_timing = bool(
                timing_events is not None
                and is_prefill
                and hidden_states.is_cuda
                and _prefill_timing_enabled()
            )
            do_leaf_timing = bool(do_timing or do_prefill_stage_timing)
            residual = hidden_states
            norm_start_end = _timing_record_start(do_leaf_timing)
            normed = self.input_layernorm(hidden_states)
            _timing_record_end(timing_events, "gemma4_norms", norm_start_end)
            if finite_trace is not None:
                _record_gemma4_prefill_finite_trace(
                    finite_trace, self.layer_idx, "layer.input_norm", normed
                )
            attn_start_end = _timing_record_start(do_timing)
            attn_out, k_cache, v_cache = self.self_attn(
                normed,
                cos,
                sin,
                positions,
                kv_cache,
                block_table,
                seq_lens,
                seq_lens_kv=seq_lens_kv,
                decode_phys_blocks=decode_phys_blocks,
                decode_blk_offsets=decode_blk_offsets,
                is_prefill=is_prefill,
                attn_mask=attn_mask,
                cu_seqlens=cu_seqlens,
                timing_events=timing_events,
                implicit_causal_prefill=implicit_causal_prefill,
                graph_safe_prefill=graph_safe_prefill,
                prefill_kv_out=prefill_kv_out,
            )
            _timing_record_end(timing_events, "attn", attn_start_end)
            if finite_trace is not None:
                _record_gemma4_prefill_finite_trace(
                    finite_trace, self.layer_idx, "layer.attention_out", attn_out
                )

            bridge_shared_in = None
            bridge_expert_in = None
            bridge_router_in = None
            rows = int(attn_out.numel()) // int(attn_out.shape[-1])
            bridge_used = bool(
                is_prefill
                and self.is_moe_layer
                and isinstance(self.mlp, Gemma4MoeMLP)
                and not graph_safe_prefill
                and self._gemma4_prefill_attn_moe_bridge_is_enabled(
                    rows,
                    attn_out,
                    residual,
                )
            )
            if bridge_used:
                bridge_start_end = _timing_record_start(do_leaf_timing)
                try:
                    bridge_workspace = (
                        self._gemma4_prefill_attn_moe_bridge_workspace_for(attn_out)
                    )
                    router_scale = self.mlp.gate.scale.to(
                        device=attn_out.device,
                        dtype=attn_out.dtype,
                    ).reshape(-1)
                    (
                        hidden_states,
                        bridge_shared_in,
                        bridge_expert_in,
                        bridge_router_in,
                    ) = rmsnorm_triton_attn_residual_router_bridge(
                        attn_out,
                        residual,
                        self.post_attention_layernorm.weight,
                        self.pre_feedforward_layernorm.weight,
                        self.pre_feedforward_layernorm_2.weight,
                        router_scale,
                        self.post_attention_layernorm.eps,
                        self.mlp.gate.scalar_root,
                        out_hidden=residual,
                        post_norm_out=bridge_workspace["post_norm"],
                        shared_out=bridge_workspace["shared"],
                        expert_out=bridge_workspace["expert"],
                        router_out=bridge_workspace["router"],
                    )
                    self._gemma4_fused_attn_moe_bridge_prefill_hits += 1
                    self._gemma4_fused_attn_moe_router_bridge_prefill_hits += 1
                except Exception as exc:
                    bridge_used = False
                    self._gemma4_prefill_attn_moe_bridge_runtime_by_rows[rows] = False
                    self._gemma4_prefill_attn_moe_bridge_error = (
                        f"{type(exc).__name__}: {exc}"
                    )
                _timing_record_end(
                    timing_events,
                    "gemma4_norms",
                    bridge_start_end,
                )
            if not bridge_used:
                norm_start_end = _timing_record_start(do_leaf_timing)
                attn_out = self.post_attention_layernorm(attn_out)
                _timing_record_end(timing_events, "gemma4_norms", norm_start_end)
                residual_start_end = _timing_record_start(do_leaf_timing)
                if torch.is_grad_enabled():
                    hidden_states = residual + attn_out
                else:
                    hidden_states = residual.add_(attn_out)
                _timing_record_end(
                    timing_events, "gemma4_residual_scale", residual_start_end
                )
            if finite_trace is not None:
                _record_gemma4_prefill_finite_trace(
                    finite_trace,
                    self.layer_idx,
                    "layer.post_attention_residual",
                    hidden_states,
                )

            residual = hidden_states
            mlp_start_end = _timing_record_start(do_timing)
            mlp_residual_fused = False
            if self.is_moe_layer and isinstance(self.mlp, Gemma4MoeMLP):
                route_input = residual.reshape(-1, residual.shape[-1])
                router_start_end = _timing_record_start(do_leaf_timing)
                routing_weights, selected_experts = self.mlp.gate.route(
                    route_input,
                    is_prefill=is_prefill,
                    normalized_router_input=bridge_router_in,
                )
                _timing_record_end(timing_events, "moe_router", router_start_end)
                if finite_trace is not None:
                    _record_gemma4_prefill_finite_trace(
                        finite_trace,
                        self.layer_idx,
                        "moe.router_input",
                        route_input,
                    )
                    _record_gemma4_prefill_finite_trace(
                        finite_trace,
                        self.layer_idx,
                        "moe.routing_weights",
                        routing_weights,
                    )

                norm_start_end = _timing_record_start(do_leaf_timing)
                shared_in = bridge_shared_in
                expert_in = bridge_expert_in
                use_dual_ffn_norm = bool(
                    shared_in is None
                    and expert_in is None
                    and is_prefill
                    and _GEMMA4_FUSED_DUAL_FFN_NORM_PREFILL
                    and rmsnorm_triton_dual is not None
                    and hidden_states.is_cuda
                    and not torch.is_grad_enabled()
                    and not self._gemma4_fused_dual_ffn_norm_disabled
                )
                if use_dual_ffn_norm:
                    try:
                        shared_in, expert_in = rmsnorm_triton_dual(
                            hidden_states,
                            self.pre_feedforward_layernorm.weight,
                            self.pre_feedforward_layernorm_2.weight,
                            self.pre_feedforward_layernorm.eps,
                        )
                        self._gemma4_fused_dual_ffn_norm_hits += 1
                    except Exception as exc:
                        self._gemma4_fused_dual_ffn_norm_disabled = True
                        self._gemma4_fused_dual_ffn_norm_error = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        shared_in = self.pre_feedforward_layernorm(hidden_states)
                elif shared_in is None:
                    shared_in = self.pre_feedforward_layernorm(hidden_states)
                _timing_record_end(timing_events, "gemma4_norms", norm_start_end)
                if expert_in is None:
                    norm_start_end = _timing_record_start(do_leaf_timing)
                    expert_in = self.pre_feedforward_layernorm_2(hidden_states)
                    _timing_record_end(timing_events, "gemma4_norms", norm_start_end)
                if finite_trace is not None:
                    _record_gemma4_prefill_finite_trace(
                        finite_trace, self.layer_idx, "moe.shared_input", shared_in
                    )
                    _record_gemma4_prefill_finite_trace(
                        finite_trace, self.layer_idx, "moe.expert_input", expert_in
                    )

                rows = int(shared_in.numel()) // int(shared_in.shape[-1])
                fused_prefill_tail = bool(
                    is_prefill
                    and not graph_safe_prefill
                    and self._gemma4_prefill_moe_tail_is_enabled(
                        rows,
                        shared_in,
                        timing_events,
                    )
                )
                parallel_moe_prefill = bool(
                    is_prefill
                    and gemma4_parallel_moe_prefill_stream is not None
                    and gemma4_parallel_moe_prefill_fork_event is not None
                    and gemma4_parallel_moe_prefill_join_event is not None
                )
                if parallel_moe_prefill:
                    main_stream = torch.cuda.current_stream(shared_in.device)
                    gemma4_parallel_moe_prefill_fork_event.record(main_stream)
                    with torch.cuda.stream(gemma4_parallel_moe_prefill_stream):
                        gemma4_parallel_moe_prefill_stream.wait_event(
                            gemma4_parallel_moe_prefill_fork_event
                        )
                        shared_out = self.mlp.shared_mlp(
                            shared_in,
                            timing_events=timing_events,
                            is_prefill=True,
                        )
                        gemma4_parallel_moe_prefill_join_event.record(
                            gemma4_parallel_moe_prefill_stream
                        )
                    expert_out = self.mlp._routed(
                        expert_in,
                        timing_events=timing_events,
                        is_prefill=True,
                        selected_experts=selected_experts,
                        routing_weights=routing_weights,
                        graph_safe_prefill=graph_safe_prefill,
                    )
                    main_stream.wait_event(
                        gemma4_parallel_moe_prefill_join_event
                    )
                    shared_out.record_stream(main_stream)
                    self._gemma4_parallel_moe_prefill_hits += 1
                else:
                    shared_out = self.mlp.shared_mlp(
                        shared_in,
                        timing_events=timing_events,
                        is_prefill=is_prefill,
                    )
                    expert_out = self.mlp._routed(
                        expert_in,
                        timing_events=timing_events,
                        is_prefill=is_prefill,
                        selected_experts=selected_experts,
                        routing_weights=routing_weights,
                        graph_safe_prefill=graph_safe_prefill,
                    )
                if finite_trace is not None:
                    _record_gemma4_prefill_finite_trace(
                        finite_trace, self.layer_idx, "moe.shared_out", shared_out
                    )
                    _record_gemma4_prefill_finite_trace(
                        finite_trace, self.layer_idx, "moe.expert_out", expert_out
                    )
                norm_start_end = _timing_record_start(do_leaf_timing)
                mlp_out = None
                if fused_prefill_tail:
                    try:
                        hidden_states = rmsnorm_triton_pair_add_final_residual(
                            shared_out,
                            expert_out,
                            self.post_feedforward_layernorm_1.weight,
                            self.post_feedforward_layernorm_2.weight,
                            self.post_feedforward_layernorm.weight,
                            residual,
                            self.post_feedforward_layernorm.eps,
                            out=residual,
                        )
                        mlp_residual_fused = True
                        self._gemma4_fused_post_moe_norm_residual_prefill_hits += 1
                    except Exception as exc:
                        self._gemma4_prefill_moe_tail_runtime_by_rows[rows] = False
                        self._gemma4_prefill_moe_tail_error = (
                            f"{type(exc).__name__}: {exc}"
                        )
                use_fused_post_ffn_norms = bool(
                    not mlp_residual_fused
                    and is_prefill
                    and _GEMMA4_FUSED_POST_FFN_NORMS_PREFILL
                    and callable(rmsnorm_triton_pair_add_final)
                    and shared_out.is_cuda
                    and expert_out.is_cuda
                    and not torch.is_grad_enabled()
                    and shared_out.dtype == torch.bfloat16
                    and rows == 25
                    and int(shared_out.shape[-1]) == 2816
                    and "A100" in torch.cuda.get_device_name(shared_out.device).upper()
                    and not self._gemma4_fused_post_ffn_norm_disabled
                )
                if use_fused_post_ffn_norms:
                    try:
                        mlp_out = rmsnorm_triton_pair_add_final(
                            shared_out,
                            expert_out,
                            self.post_feedforward_layernorm_1.weight,
                            self.post_feedforward_layernorm_2.weight,
                            self.post_feedforward_layernorm.weight,
                            self.post_feedforward_layernorm.eps,
                        )
                        if not self._gemma4_fused_post_ffn_norm_checked:
                            reference_shared = self.post_feedforward_layernorm_1(shared_out)
                            reference_expert = self.post_feedforward_layernorm_2(expert_out)
                            reference = rmsnorm_triton_add(
                                reference_shared,
                                reference_expert,
                                self.post_feedforward_layernorm.weight,
                                self.post_feedforward_layernorm.eps,
                            )
                            if not torch.equal(mlp_out, reference):
                                self._gemma4_fused_post_ffn_norm_disabled = True
                                self._gemma4_fused_post_ffn_norm_error = "reference mismatch"
                                mlp_out = reference
                            else:
                                self._gemma4_fused_post_ffn_norm_checked = True
                        if not self._gemma4_fused_post_ffn_norm_disabled:
                            self._gemma4_fused_post_ffn_norm_hits += 1
                    except Exception as exc:
                        self._gemma4_fused_post_ffn_norm_disabled = True
                        self._gemma4_fused_post_ffn_norm_error = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        mlp_out = None
                if mlp_out is None and not mlp_residual_fused:
                    shared_out = self.post_feedforward_layernorm_1(shared_out)
                    expert_out = self.post_feedforward_layernorm_2(expert_out)
                    if finite_trace is not None:
                        _record_gemma4_prefill_finite_trace(
                            finite_trace,
                            self.layer_idx,
                            "moe.shared_post_norm",
                            shared_out,
                        )
                        _record_gemma4_prefill_finite_trace(
                            finite_trace,
                            self.layer_idx,
                            "moe.expert_post_norm",
                            expert_out,
                        )
                    use_add_ffn_norm = bool(
                        is_prefill
                        and _GEMMA4_FUSED_ADD_FFN_NORM_PREFILL
                        and rmsnorm_triton_add is not None
                        and shared_out.is_cuda
                        and expert_out.is_cuda
                        and not torch.is_grad_enabled()
                        and not self._gemma4_fused_add_ffn_norm_disabled
                    )
                    if use_add_ffn_norm:
                        try:
                            mlp_out = rmsnorm_triton_add(
                                shared_out,
                                expert_out,
                                self.post_feedforward_layernorm.weight,
                                self.post_feedforward_layernorm.eps,
                            )
                            self._gemma4_fused_add_ffn_norm_hits += 1
                        except Exception as exc:
                            self._gemma4_fused_add_ffn_norm_disabled = True
                            self._gemma4_fused_add_ffn_norm_error = (
                                f"{type(exc).__name__}: {exc}"
                            )
                            mlp_out = self.post_feedforward_layernorm(
                                shared_out + expert_out
                            )
                    else:
                        mlp_out = self.post_feedforward_layernorm(
                            shared_out + expert_out
                        )
                _timing_record_end(timing_events, "gemma4_norms", norm_start_end)
                if finite_trace is not None:
                    _record_gemma4_prefill_finite_trace(
                        finite_trace, self.layer_idx, "moe.combined_out", mlp_out
                    )
            else:
                norm_start_end = _timing_record_start(do_leaf_timing)
                mlp_in = self.pre_feedforward_layernorm(hidden_states)
                _timing_record_end(timing_events, "gemma4_norms", norm_start_end)
                mlp_out = self.mlp(mlp_in, timing_events=timing_events, is_prefill=is_prefill)
                norm_start_end = _timing_record_start(do_leaf_timing)
                mlp_out = self.post_feedforward_layernorm(mlp_out)
                _timing_record_end(timing_events, "gemma4_norms", norm_start_end)
            _timing_record_end(timing_events, "mlp", mlp_start_end)
            residual_start_end = _timing_record_start(do_leaf_timing)
            if not mlp_residual_fused:
                if torch.is_grad_enabled():
                    hidden_states = residual + mlp_out
                else:
                    hidden_states = residual.add_(mlp_out)
            _timing_record_end(
                timing_events, "gemma4_residual_scale", residual_start_end
            )
            if finite_trace is not None:
                _record_gemma4_prefill_finite_trace(
                    finite_trace,
                    self.layer_idx,
                    "layer.post_moe_residual",
                    hidden_states,
                )

            if per_layer_input is not None and self.per_layer_input_gate is not None:
                ple_start_end = _timing_record_start(do_leaf_timing)
                residual = hidden_states
                ple = self._gemma4_ple_linear(
                    self.per_layer_input_gate,
                    hidden_states,
                    is_prefill=is_prefill,
                    decode_attr="_fast_ple_gate_out",
                    prefill_attr="_prefill_ple_gate_out",
                )
                ple = torch.nn.functional.gelu(ple, approximate='tanh')
                if torch.is_grad_enabled():
                    ple = ple * per_layer_input
                else:
                    ple.mul_(per_layer_input)
                ple = self._gemma4_ple_linear(
                    self.per_layer_projection,
                    ple,
                    is_prefill=is_prefill,
                    decode_attr="_fast_ple_proj_out",
                    prefill_attr="_prefill_ple_proj_out",
                )
                ple = self.post_per_layer_input_norm(ple)
                if torch.is_grad_enabled():
                    hidden_states = residual + ple
                else:
                    hidden_states = residual.add_(ple)
                _timing_record_end(timing_events, "ple", ple_start_end)

            layer_scale = self.layer_scalar.to(dtype=hidden_states.dtype)
            residual_start_end = _timing_record_start(do_leaf_timing)
            if torch.is_grad_enabled():
                hidden_states = hidden_states * layer_scale
            else:
                hidden_states.mul_(layer_scale)
            _timing_record_end(
                timing_events, "gemma4_residual_scale", residual_start_end
            )
            if finite_trace is not None:
                _record_gemma4_prefill_finite_trace(
                    finite_trace, self.layer_idx, "layer.output", hidden_states
                )
            return hidden_states, k_cache, v_cache, next_linear_conv, next_linear_recurrent

        # === Self-attention with residual ===
        # Pre-norm: RMSNorm(hidden) → attention input
        use_fused_prefill_qkv = (
            is_prefill
            and self.layer_type == 'full_attention'
            and _USE_FUSED_RMSNORM_QKV_PREFILL
        )
        use_linear_raw_decode = (
            self.layer_type == 'linear_attention'
            and not is_prefill
            and use_linear_cache
            and linear_conv_state is not None
            and linear_recurrent_state is not None
            and not input_is_normed
            and input_norm_weight is not None
            and not torch.is_grad_enabled()
        )
        if use_fused_prefill_qkv or use_linear_raw_decode:
            normed = hidden_states
        else:
            normed = self.input_layernorm(hidden_states)
        next_linear_conv = None
        next_linear_recurrent = None
        do_timing = (
            timing_events is not None
            and not is_prefill
            and hidden_states.is_cuda
            and _timing_enabled()
        )
        attn_start_end = _timing_record_start(do_timing)
        if self.layer_type == 'linear_attention':
            attn_out, next_linear_conv, next_linear_recurrent = self.linear_attn(
                normed,
                conv_state=linear_conv_state,
                recurrent_state=linear_recurrent_state,
                attention_mask=linear_attention_mask,
                use_cache=use_linear_cache,
                timing_events=timing_events,
                input_is_normed=not use_linear_raw_decode,
                input_norm_weight=input_norm_weight,
                input_norm_eps=input_norm_eps,
                input_norm_offset=input_norm_offset,
            )
            k_cache = None
            v_cache = None
        else:
            attn_out, k_cache, v_cache = self.self_attn(
                normed, cos, sin, positions,
                kv_cache, block_table, seq_lens,
                seq_lens_kv=seq_lens_kv,
                decode_phys_blocks=decode_phys_blocks,
                decode_blk_offsets=decode_blk_offsets,
                is_prefill=is_prefill,
                attn_mask=attn_mask,
                cu_seqlens=cu_seqlens,
                append_kv_prefix=append_kv_prefix,
                input_is_normed=not use_fused_prefill_qkv,
                input_norm_weight=self.input_layernorm.weight,
                input_norm_eps=self._norm_eps,
                input_norm_offset=self._norm_offset,
                timing_events=timing_events,
            )
        _timing_record_end(timing_events, "attn", attn_start_end)

        # Fused add+rmsnorm: residual add + post-attn norm in 1 kernel
        # Saves 2 kernel launches per layer (add + rmsnorm → 1 fused)
        use_fused_norm_gateup = (
            self._use_fused_norm_gateup_decode
            and not is_prefill
            and not torch.is_grad_enabled()
        )
        if use_fused_norm_gateup:
            hidden_states.add_(attn_out)
            mlp_in = hidden_states
        elif _HAS_FUSED_ADD_RMSNORM:
            hidden_states, mlp_in = fused_add_rmsnorm(
                hidden_states, attn_out,
                self.post_attention_layernorm.weight,
                self._norm_eps, self._norm_offset,
            )
        else:
            hidden_states = hidden_states + attn_out
            mlp_in = self.post_attention_layernorm(hidden_states)

        # === MLP with residual ===
        mlp_start_end = _timing_record_start(do_timing)
        if use_fused_norm_gateup:
            hidden_states = self.mlp.forward_decode_add_residual(
                mlp_in,
                hidden_states,
                input_is_normed=False,
                input_norm_weight=self.post_attention_layernorm.weight,
                input_norm_eps=self._norm_eps,
                input_norm_offset=self._norm_offset,
            )
        elif (
            is_prefill
            and self.is_moe_layer
            and not torch.is_grad_enabled()
            and hasattr(self.mlp, "forward_prefill_add_residual")
        ):
            hidden_states = self.mlp.forward_prefill_add_residual(
                mlp_in,
                hidden_states,
                timing_events=timing_events,
            )
        else:
            mlp_out = self.mlp(mlp_in, timing_events=timing_events, is_prefill=is_prefill)
            if torch.is_grad_enabled():
                hidden_states = hidden_states + mlp_out
            else:
                hidden_states.add_(mlp_out)
        _timing_record_end(timing_events, "mlp", mlp_start_end)

        return hidden_states, k_cache, v_cache, next_linear_conv, next_linear_recurrent

    def decode_forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        kv_cache=None,
        block_table=None,
        seq_lens=None,
        seq_lens_kv=None,
        decode_phys_blocks=None,
        decode_blk_offsets=None,
        linear_conv_state=None,
        linear_recurrent_state=None,
        use_linear_cache=False,
        timing_events: Optional[dict] = None,
        per_layer_input: Optional[torch.Tensor] = None,
    ):
        """
        Decode-specialized forward to reduce Python dispatch overhead.
        Falls back to regular path for linear-attention layers.
        """
        if self.is_gemma4:
            residual = hidden_states
            do_timing = (
                timing_events is not None
                and hidden_states.is_cuda
                and _timing_enabled()
            )
            normed = self.input_layernorm(hidden_states)
            attn_start_end = _timing_record_start(do_timing)
            attn_out, _, _ = self.self_attn.forward_decode(
                normed,
                cos,
                sin,
                positions,
                kv_cache=kv_cache,
                block_table=block_table,
                seq_lens=seq_lens,
                seq_lens_kv=seq_lens_kv,
                decode_phys_blocks=decode_phys_blocks,
                decode_blk_offsets=decode_blk_offsets,
                timing_events=timing_events,
            )
            _timing_record_end(timing_events, "attn", attn_start_end)

            attn_out = self.post_attention_layernorm(attn_out)
            hidden_states = residual.add_(attn_out)

            residual = hidden_states
            mlp_in = self.pre_feedforward_layernorm(hidden_states)
            mlp_start_end = _timing_record_start(do_timing)
            mlp_out = self.mlp.forward_decode(mlp_in)
            _timing_record_end(timing_events, "mlp", mlp_start_end)
            mlp_out = self.post_feedforward_layernorm(mlp_out)
            hidden_states = residual.add_(mlp_out)

            if per_layer_input is not None and self.per_layer_input_gate is not None:
                residual = hidden_states
                ple = self._gemma4_ple_linear(
                    self.per_layer_input_gate,
                    hidden_states,
                    is_prefill=False,
                    decode_attr="_fast_ple_gate_out",
                    prefill_attr="_prefill_ple_gate_out",
                )
                ple = torch.nn.functional.gelu(ple, approximate='tanh')
                ple.mul_(per_layer_input)
                ple = self._gemma4_ple_linear(
                    self.per_layer_projection,
                    ple,
                    is_prefill=False,
                    decode_attr="_fast_ple_proj_out",
                    prefill_attr="_prefill_ple_proj_out",
                )
                ple = self.post_per_layer_input_norm(ple)
                hidden_states = residual.add_(ple)

            hidden_states.mul_(self.layer_scalar.to(dtype=hidden_states.dtype))
            return hidden_states, None, None

        if self.layer_type == 'linear_attention':
            hidden_states, _, _, next_linear_conv, next_linear_recurrent = self.forward(
                hidden_states, cos, sin, positions,
                kv_cache, block_table, seq_lens,
                seq_lens_kv=seq_lens_kv,
                decode_phys_blocks=decode_phys_blocks,
                decode_blk_offsets=decode_blk_offsets,
                is_prefill=False,
                linear_conv_state=linear_conv_state,
                linear_recurrent_state=linear_recurrent_state,
                use_linear_cache=use_linear_cache,
                timing_events=timing_events,
                input_is_normed=False,
                input_norm_weight=self.input_layernorm.weight,
                input_norm_eps=self._norm_eps,
                input_norm_offset=self._norm_offset,
            )
            return hidden_states, next_linear_conv, next_linear_recurrent

        use_fused_norm_qkv = self._use_fused_norm_qkv_decode
        if use_fused_norm_qkv:
            attn_in = hidden_states
        else:
            attn_in = self.input_layernorm(hidden_states)
        do_timing = (
            timing_events is not None
            and hidden_states.is_cuda
            and _timing_enabled()
        )
        attn_start_end = _timing_record_start(do_timing)
        attn_out, _, _ = self.self_attn.forward_decode(
            attn_in,
            cos,
            sin,
            positions,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            seq_lens_kv=seq_lens_kv,
            decode_phys_blocks=decode_phys_blocks,
            decode_blk_offsets=decode_blk_offsets,
            input_is_normed=not use_fused_norm_qkv,
            input_norm_weight=self.input_layernorm.weight,
            input_norm_eps=self._norm_eps,
            input_norm_offset=self._norm_offset,
            timing_events=timing_events,
        )
        _timing_record_end(timing_events, "attn", attn_start_end)

        use_fused_norm_gateup = self._use_fused_norm_gateup_decode
        if use_fused_norm_gateup:
            if torch.is_grad_enabled():
                hidden_states = hidden_states + attn_out
            else:
                hidden_states.add_(attn_out)
            mlp_in = hidden_states
        elif _HAS_FUSED_ADD_RMSNORM:
            hidden_states, mlp_in = fused_add_rmsnorm(
                hidden_states, attn_out,
                self.post_attention_layernorm.weight,
                self._norm_eps, self._norm_offset,
            )
        else:
            hidden_states = hidden_states + attn_out
            mlp_in = self.post_attention_layernorm(hidden_states)

        mlp_start_end = _timing_record_start(do_timing)
        if torch.is_grad_enabled():
            if use_fused_norm_gateup:
                mlp_out = self.mlp.forward_decode(
                    mlp_in,
                    input_is_normed=False,
                    input_norm_weight=self.post_attention_layernorm.weight,
                    input_norm_eps=self._norm_eps,
                    input_norm_offset=self._norm_offset,
                )
            else:
                mlp_out = self.mlp.forward_decode(mlp_in)
        else:
            if use_fused_norm_gateup:
                hidden_states = self.mlp.forward_decode_add_residual(
                    mlp_in,
                    hidden_states,
                    input_is_normed=False,
                    input_norm_weight=self.post_attention_layernorm.weight,
                    input_norm_eps=self._norm_eps,
                    input_norm_offset=self._norm_offset,
                )
            else:
                hidden_states = self.mlp.forward_decode_add_residual(
                    mlp_in,
                    hidden_states,
                )
        _timing_record_end(timing_events, "mlp", mlp_start_end)
        if torch.is_grad_enabled():
            hidden_states = hidden_states + mlp_out
        return hidden_states, None, None

    def decode_forward_full_attn(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        kv_cache,
        block_table,
        seq_lens,
        seq_lens_kv,
        decode_phys_blocks,
        decode_blk_offsets,
        timing_events: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Minimal decode path for full-attention layers only.
        Returns only hidden_states to reduce Python/C++ boundary overhead.
        """
        do_timing = (
            timing_events is not None
            and hidden_states.is_cuda
            and _timing_enabled()
        )
        use_fused_norm_qkv = self._use_fused_norm_qkv_decode
        if use_fused_norm_qkv:
            attn_in = hidden_states
        else:
            attn_in = self.input_layernorm(hidden_states)
        attn_start_end = _timing_record_start(do_timing)
        attn_out, _, _ = self.self_attn.forward_decode(
            attn_in,
            cos,
            sin,
            positions,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            seq_lens_kv=seq_lens_kv,
            decode_phys_blocks=decode_phys_blocks,
            decode_blk_offsets=decode_blk_offsets,
            input_is_normed=not use_fused_norm_qkv,
            input_norm_weight=self.input_layernorm.weight,
            input_norm_eps=self._norm_eps,
            input_norm_offset=self._norm_offset,
            timing_events=timing_events,
        )
        _timing_record_end(timing_events, "attn", attn_start_end)
        use_fused_norm_gateup = self._use_fused_norm_gateup_decode
        if use_fused_norm_gateup:
            if torch.is_grad_enabled():
                hidden_states = hidden_states + attn_out
            else:
                hidden_states.add_(attn_out)
            mlp_in = hidden_states
        elif _HAS_FUSED_ADD_RMSNORM:
            hidden_states, mlp_in = fused_add_rmsnorm(
                hidden_states, attn_out,
                self.post_attention_layernorm.weight,
                self._norm_eps, self._norm_offset,
            )
        else:
            hidden_states = hidden_states + attn_out
            mlp_in = self.post_attention_layernorm(hidden_states)
        mlp_start_end = _timing_record_start(do_timing)
        if torch.is_grad_enabled():
            if use_fused_norm_gateup:
                mlp_out = self.mlp.forward_decode(
                    mlp_in,
                    input_is_normed=False,
                    input_norm_weight=self.post_attention_layernorm.weight,
                    input_norm_eps=self._norm_eps,
                    input_norm_offset=self._norm_offset,
                    timing_events=timing_events,
                )
            else:
                mlp_out = self.mlp.forward_decode(
                    mlp_in,
                    timing_events=timing_events,
                )
            hidden_states = hidden_states + mlp_out
        else:
            if use_fused_norm_gateup:
                hidden_states = self.mlp.forward_decode_add_residual(
                    mlp_in,
                    hidden_states,
                    input_is_normed=False,
                    input_norm_weight=self.post_attention_layernorm.weight,
                    input_norm_eps=self._norm_eps,
                    input_norm_offset=self._norm_offset,
                    timing_events=timing_events,
                )
            else:
                hidden_states = self.mlp.forward_decode_add_residual(
                    mlp_in,
                    hidden_states,
                    timing_events=timing_events,
                )
        _timing_record_end(timing_events, "mlp", mlp_start_end)
        return hidden_states

    def decode_forward_full_attn_infer(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        kv_cache,
        block_table,
        seq_lens,
        seq_lens_kv,
        decode_phys_blocks,
        decode_blk_offsets,
    ) -> torch.Tensor:
        """
        Inference-only flat decode path for full-attention layers.
        Removes per-token dynamic branches (grad checks / flag checks) in hot path.
        """
        if self.layer_type != 'full_attention':
            return self.decode_forward_full_attn(
                hidden_states,
                cos,
                sin,
                positions,
                kv_cache,
                block_table,
                seq_lens,
                seq_lens_kv,
                decode_phys_blocks,
                decode_blk_offsets,
            )

        if self._use_fused_norm_qkv_decode:
            attn_in = hidden_states
        else:
            attn_in = self.input_layernorm(hidden_states)

        attn_out, _, _ = self.self_attn.forward_decode(
            attn_in,
            cos,
            sin,
            positions,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            seq_lens_kv=seq_lens_kv,
            decode_phys_blocks=decode_phys_blocks,
            decode_blk_offsets=decode_blk_offsets,
            input_is_normed=not self._use_fused_norm_qkv_decode,
            input_norm_weight=self.input_layernorm.weight,
            input_norm_eps=self._norm_eps,
            input_norm_offset=self._norm_offset,
        )

        if self._use_fused_norm_gateup_decode:
            hidden_states.add_(attn_out)
            mlp_in = hidden_states
            return self.mlp.forward_decode_add_residual(
                mlp_in,
                hidden_states,
                input_is_normed=False,
                input_norm_weight=self.post_attention_layernorm.weight,
                input_norm_eps=self._norm_eps,
                input_norm_offset=self._norm_offset,
            )

        if _HAS_FUSED_ADD_RMSNORM:
            hidden_states, mlp_in = fused_add_rmsnorm(
                hidden_states,
                attn_out,
                self.post_attention_layernorm.weight,
                self._norm_eps,
                self._norm_offset,
            )
        else:
            hidden_states.add_(attn_out)
            mlp_in = self.post_attention_layernorm(hidden_states)

        return self.mlp.forward_decode_add_residual(mlp_in, hidden_states)


class MegaGemmLlama(nn.Module):
    """
    🧠 MegaGemm Unified Model — High performance inference.

    Supports: LLaMA, Mistral, Qwen 2.5, Qwen 3, Gemma 2.
    Uses MegaGemm kernels for RMSNorm, SwiGLU, and PagedAttention.
    Load weights from HuggingFace with `load_from_hf()`.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.runtime_policy = resolve_runtime_policy(config)
        self._gemma4_prefer_triton_rmsnorm = False
        if self.config.rotary_dim == 0:
            self.config.rotary_dim = self.config.head_dim
        if not self.config.layer_types:
            self.config.layer_types = ['full_attention'] * self.config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.hidden_size_per_layer_input = (
            config.hidden_size_per_layer_input if config.model_type == 'gemma4_text' else 0
        )
        if self.hidden_size_per_layer_input:
            total_ple_dim = config.num_hidden_layers * self.hidden_size_per_layer_input
            self.embed_tokens_per_layer = nn.Embedding(
                config.vocab_size_per_layer_input,
                total_ple_dim,
                padding_idx=0,
            )
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                total_ple_dim,
                bias=False,
            )
            self.per_layer_projection_norm = MGRMSNorm(
                self.hidden_size_per_layer_input,
                config.rms_norm_eps,
                offset=False,
            )
            self.register_buffer(
                "embed_scale_per_layer",
                torch.tensor(self.hidden_size_per_layer_input ** 0.5),
                persistent=False,
            )
            self.register_buffer(
                "per_layer_input_scale",
                torch.rsqrt(torch.tensor(2.0)),
                persistent=False,
            )
            self.register_buffer(
                "per_layer_projection_scale",
                torch.tensor(config.hidden_size ** -0.5),
                persistent=False,
            )
        else:
            self.embed_tokens_per_layer = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None
        self.embed_scale = config.embed_scale  # Gemma 2/4: sqrt(hidden_size)
        self.register_buffer(
            "gemma4_embed_scale",
            (
                torch.tensor(config.embed_scale)
                if config.model_type == 'gemma4_text'
                else None
            ),
            persistent=False,
        )

        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self._all_full_attention = (
            config.model_type != 'gemma4_text'
            and all(layer.layer_type == 'full_attention' for layer in self.layers)
        )
        self._layer_decode_full_fns = (
            [layer.decode_forward_full_attn_infer for layer in self.layers]
            if self._all_full_attention else []
        )
        self.norm = MGRMSNorm(
            config.hidden_size, config.rms_norm_eps, offset=config.norm_offset
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.final_logit_softcapping = config.final_logit_softcapping
        self._fast_lm_head_out = None
        self._fast_lm_head_decode_key = None
        self._fast_lm_head_decode_use = False
        self._fused_lm_head_argmax_key = None
        self._fused_lm_head_argmax_use = False
        self._fused_lm_head_argmax_checked = False
        self._fused_lm_head_argmax_disable = False
        self._fused_lm_head_argmax_error = ""
        self._fused_lm_head_argmax_skip_reason = ""
        self._fused_rmsnorm_lm_head_argmax_key = None
        self._fused_rmsnorm_lm_head_argmax_use = False
        self._fused_rmsnorm_lm_head_argmax_checked = False
        self._fused_rmsnorm_lm_head_argmax_disable = False
        self._fused_rmsnorm_lm_head_argmax_hits = 0
        self._fused_rmsnorm_lm_head_argmax_error = ""
        self._fused_rmsnorm_lm_head_argmax_skip_reason = ""
        self._gemma4_batch_cublas_lm_head_hits = 0
        self._gemma4_batch_fused_softcap_argmax_hits = 0
        self._gemma4_batch_fused_softcap_argmax_disable = False
        self._gemma4_batch_fused_softcap_argmax_error = ""
        self._fused_lm_head_tokens = None
        self._fused_lm_head_partial_vals = None
        self._fused_lm_head_partial_idxs = None
        self._last_decode_timing = None
        self._last_prefill_timing = None
        self._prefill_last_token_only_hits = 0
        self._gemma4_batch_prefill_vectorized_kv_hits = 0
        self._gemma4_implicit_causal_prefill_batches = 0
        self._gemma4_parallel_moe_prefill_enabled = False
        self._gemma4_parallel_moe_prefill_policy = {
            "requested": bool(_GEMMA4_PARALLEL_MOE_PREFILL),
            "enabled": False,
        }
        self._gemma4_parallel_moe_prefill_resource_key = None
        self._gemma4_parallel_moe_prefill_stream = None
        self._gemma4_parallel_moe_prefill_fork_events = None
        self._gemma4_parallel_moe_prefill_join_events = None
        self._force_sequential_prefill = False
        self._gemma4_prefill_finite_trace = None
        self._prefill_cuda_graph_store = {
            'block_manager_id': None,
            'buckets': {},
            'warm_keys': set(),
            'failed_keys': {},
            'skips': 0,
            'captures': 0,
            'capture_body_warmups': 0,
            'capture_replays': 0,
            'replays': 0,
            'external_kv_write_replays': 0,
            'warmups': 0,
            'failures': 0,
            'last_failure': "",
        }

        # Layer offloader (set externally, None = no offload)
        self._offloader = None

        # Precompute RoPE frequencies. Gemma 4 advertises 131k context, but
        # precomputing every per-layer RoPE table at that size costs multiple
        # GB of VRAM; the engine shrinks/grows this to the active max_seq_len.
        self._rope_cache_max_seq_len = _initial_rope_cache_len(config)
        self.cos_cache, self.sin_cache = precompute_freqs_cis(
            config.rotary_dim, self._rope_cache_max_seq_len, config.rope_theta
        )
        self.layer_rope_caches = None
        if config.model_type == 'gemma4_text':
            self.layer_rope_caches = [
                _precompute_layer_rope_cache(
                    config,
                    layer_idx,
                    max_seq_len=self._rope_cache_max_seq_len,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        self._prefill_arange_cache = None
        self._prefill_arange_cache_key = None
        self._prefill_causal_mask_cache = None
        self._prefill_causal_mask_cache_key = None
        # CUDA graphs retain tensor addresses. Keep one deferred K/V destination
        # per layer and captured shape instead of returning attention temporaries.
        self._prefill_graph_deferred_kv_buffers = {}
        self._prefill_graph_deferred_kv_copy_dispatches = 0

        # Zero-overhead flat decode state (populated lazily on first decode)
        self._flat_decode_ready = False
        self._flat_decode_failed = False
        self._flat_decode_failed_reason = ""
        self._flat_is_gemma4 = False
        self._gemma4_flat_parallel_moe_enabled = False
        self._gemma4_flat_parallel_moe_hits = 0
        self._gemma4_flat_parallel_moe_policy = {}
        self._gemma4_flat_parallel_moe_stream = None
        self._gemma4_flat_parallel_moe_fork_events = None
        self._gemma4_flat_parallel_moe_join_events = None
        self._gemma4_flat_parallel_shared_norm_bufs = None
        self._gemma4_flat_fused_attn_moe_bridge_enabled = False
        self._gemma4_flat_fused_attn_moe_bridge_hits = 0
        self._gemma4_flat_fused_attn_moe_router_bridge_enabled = False
        self._gemma4_flat_fused_attn_moe_router_bridge_hits = 0
        self._gemma4_flat_fused_attn_moe_router_single_kernel_enabled = False
        self._gemma4_flat_fused_attn_moe_router_single_kernel_hits = 0
        self._gemma4_flat_fused_router_compact_pack_enabled = False
        self._gemma4_flat_fused_router_compact_pack_hits = 0
        self._gemma4_flat_attn_post_norm_bufs = None
        self._gemma4_flat_shared_input_bufs = None
        self._gemma4_flat_fused_post_moe_norm_residual_enabled = False
        self._gemma4_flat_fused_post_moe_norm_residual_hits = 0
        self._gemma4_flat_fused_expert_reduce_post_moe_enabled = False
        self._gemma4_flat_fused_expert_reduce_post_moe_hits = 0
        self._gemma4_flat_post_moe_out_bufs = None
        self._gemma4_flat_fused_next_attn_norm_supported = False
        self._gemma4_flat_fused_next_attn_norm_enabled = False
        self._gemma4_flat_fused_next_attn_norm_hits = 0
        self._gemma4_flat_fused_layer_scalar_hits = 0
        self._gemma4_flat_next_attn_norm_bufs = None
        self._gemma4_flat_dense_post_norm_chain_enabled = False
        self._gemma4_flat_dense_post_norm_chain_hits = 0
        self._gemma4_flat_dense_next_attn_norm_hits = 0
        self._gemma4_flat_dense_next_attn_norm_bufs = None
        self._gemma4_flat_dense_attn_mlp_bridge_enabled = False
        self._gemma4_flat_dense_attn_mlp_bridge_hits = 0
        self._gemma4_flat_dense_attn_mlp_bridge_runtime_disabled = False
        self._gemma4_flat_dense_attn_mlp_bridge_failure = ""
        self._gemma4_flat_dense_attn_mlp_input_bufs = None
        self._gemma4_flat_b1_large_gateup_experiment = _env_enabled(
            "MEGAGEMM_GEMMA4_E2B_L4_B1_LARGE_GATEUP_EXPERIMENT",
            default=False,
        )
        self._gemma4_flat_b1_large_down_experiment = _env_enabled(
            "MEGAGEMM_GEMMA4_E2B_L4_B1_LARGE_DOWN_EXPERIMENT",
            default=False,
        )
        self._gemma4_flat_b1_large_gateup_hits = 0
        self._gemma4_flat_b1_large_down_hits = 0
        self._gemma4_flat_b8_gated_activation_enabled = False
        self._gemma4_flat_b8_gated_activation_block_size = 512
        self._gemma4_flat_b8_gated_activation_hits = 0
        self._gemma4_flat_b8_gated_activation_runtime_disabled = False
        self._gemma4_flat_b8_gated_activation_failure = ""
        self._gemma4_flat_b8_gated_activation_bufs = None
        self._gemma4_flat_b8_tensorcore_down_enabled = False
        self._gemma4_flat_b8_tensorcore_down_config = (64, 32, 4, 2)
        self._gemma4_flat_b8_tensorcore_down_hits = 0
        self._gemma4_flat_b8_tensorcore_down_runtime_disabled = False
        self._gemma4_flat_b8_tensorcore_down_failure = ""
        self._gemma4_flat_ple_conditioned_gelu_block_size = 256
        self._gemma4_flat_cublaslt_gateup_enabled = False
        self._gemma4_flat_cublaslt_gateup_hits = 0
        self._gemma4_flat_cublaslt_gateup_runtime_disabled = False
        self._gemma4_flat_cublaslt_gateup_failure = ""
        self._gemma4_flat_cublaslt_gateup_algorithms = {}
        self._gemma4_flat_policy_fused_gateup_rows = ()
        self._gemma4_flat_policy_deepfusion_rows = ()
        self._gemma4_flat_policy_cublas_gateup_rows = ()
        self._gemma4_flat_policy_cublas_down_rows = ()
        self._gemma4_flat_fused_router_expert_input_norm_enabled = False
        self._gemma4_flat_fused_router_expert_input_norm_hits = 0
        self._gemma4_flat_expert_input_bufs = None
        self._gemma4_flat_router_input_bufs = None
        self._flat_is_hybrid = False
        self._flat_hybrid_hits = 0
        self._flat_hybrid_full_inline_enabled = False
        self._flat_hybrid_full_inline_hits = 0
        self._flat_hybrid_full_fallback_hits = 0

    def _move_rope_to_device(self, device):
        """Move RoPE caches to correct device."""
        if self.cos_cache.device != device:
            self.cos_cache = self.cos_cache.to(device)
            self.sin_cache = self.sin_cache.to(device)
        if self.layer_rope_caches is not None:
            moved = []
            for cos, sin in self.layer_rope_caches:
                if cos.device != device:
                    cos = cos.to(device)
                    sin = sin.to(device)
                moved.append((cos, sin))
            self.layer_rope_caches = moved

    def _prepare_gemma4_parallel_moe_prefill(
        self,
        hidden_states: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        uniform_full_length: bool,
        timing_events: Optional[dict],
    ):
        empty = (None, None, None)
        rows = int(hidden_states.numel()) // max(
            1,
            int(hidden_states.shape[-1]),
        )
        if not (
            _GEMMA4_PARALLEL_MOE_PREFILL
            and int(batch_size) == 16
            and int(seq_len) == 25
            and rows == 400
            and uniform_full_length
            and self.config.model_type == "gemma4_text"
            and hidden_states.is_cuda
            and not torch.is_grad_enabled()
            and timing_events is None
            and self._offloader is None
            and self.layers
            and all(layer.is_moe_layer for layer in self.layers)
            and isinstance(self.layers[0].mlp, Gemma4MoeMLP)
        ):
            return empty

        try:
            if torch.cuda.is_current_stream_capturing():
                return empty
        except Exception:
            return empty

        first_mlp = self.layers[0].mlp
        device_name = torch.cuda.get_device_name(hidden_states.device)
        if not _gemma4_a100_a4b_parallel_moe_prefill_shape(
            self.config.model_type,
            rows,
            int(hidden_states.shape[-1]),
            int(first_mlp.shared_mlp.intermediate_size),
            int(first_mlp.experts.intermediate_dim),
            hidden_states.dtype,
            device_name,
        ):
            return empty

        device_index = (
            int(hidden_states.device.index)
            if hidden_states.device.index is not None
            else int(torch.cuda.current_device())
        )
        resource_key = (
            device_index,
            hidden_states.dtype,
            rows,
            len(self.layers),
        )
        if (
            self._gemma4_parallel_moe_prefill_resource_key != resource_key
            or self._gemma4_parallel_moe_prefill_stream is None
            or self._gemma4_parallel_moe_prefill_fork_events is None
            or self._gemma4_parallel_moe_prefill_join_events is None
        ):
            with torch.cuda.device(hidden_states.device):
                self._gemma4_parallel_moe_prefill_stream = torch.cuda.Stream(
                    device=hidden_states.device
                )
                self._gemma4_parallel_moe_prefill_fork_events = [
                    torch.cuda.Event() for _ in self.layers
                ]
                self._gemma4_parallel_moe_prefill_join_events = [
                    torch.cuda.Event() for _ in self.layers
                ]
            self._gemma4_parallel_moe_prefill_resource_key = resource_key

        self._gemma4_parallel_moe_prefill_enabled = True
        self._gemma4_parallel_moe_prefill_policy = {
            "requested": True,
            "enabled": True,
            "model_type": self.config.model_type,
            "device_name": device_name,
            "dtype": str(hidden_states.dtype),
            "batch_size": int(batch_size),
            "seq_len": int(seq_len),
            "rows": rows,
            "hidden_dim": int(hidden_states.shape[-1]),
            "shared_intermediate": int(first_mlp.shared_mlp.intermediate_size),
            "expert_intermediate": int(first_mlp.experts.intermediate_dim),
            "measured_speedup": 1.021,
            "estimated_savings_ms": 0.793,
        }
        return (
            self._gemma4_parallel_moe_prefill_stream,
            self._gemma4_parallel_moe_prefill_fork_events,
            self._gemma4_parallel_moe_prefill_join_events,
        )

    def set_rope_cache_max_seq_len(self, max_seq_len: int, device=None) -> None:
        """Resize RoPE caches to the active serving context length."""
        target_len = max(1, min(int(self.config.max_position_embeddings), int(max_seq_len)))
        if (
            target_len == getattr(self, "_rope_cache_max_seq_len", None)
            and (device is None or self.cos_cache.device == torch.device(device))
        ):
            return
        self._rope_cache_max_seq_len = target_len
        self.cos_cache, self.sin_cache = precompute_freqs_cis(
            self.config.rotary_dim,
            target_len,
            self.config.rope_theta,
        )
        if self.config.model_type == 'gemma4_text':
            self.layer_rope_caches = [
                _precompute_layer_rope_cache(
                    self.config,
                    layer_idx,
                    max_seq_len=target_len,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        if device is not None:
            self._move_rope_to_device(torch.device(device))

    def _refresh_gemma4_runtime_buffers(self, device=None, dtype=None) -> None:
        """Materialize non-weight Gemma 4 buffers after meta-device loading."""
        if self.config.model_type != 'gemma4_text':
            return
        if device is None:
            device = self.embed_tokens.weight.device
        device = torch.device(device)
        device_name = ""
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(device)
            except Exception:
                device_name = ""
        self.runtime_policy = resolve_runtime_policy(
            self.config,
            device_name=device_name,
        )
        e2b_l4_sliding_prefill = policy_bool(
            self,
            "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_PREFILL",
            "gemma4_e2b_l4_sliding_prefill",
        )
        e2b_l4_full_prefill_expand = policy_bool(
            self,
            "MEGAGEMM_GEMMA4_E2B_L4_FULL_PREFILL_EXPAND",
            "gemma4_e2b_l4_full_prefill_expand",
        )
        e2b_b8_prefill_gated_activation = policy_bool(
            self,
            "MEGAGEMM_GEMMA4_E2B_B8_PREFILL_GATED_ACTIVATION",
            "gemma4_e2b_b8_prefill_gated_activation",
        )
        e2b_b8_prefill_gated_activation_blocks = {
            int(sequence_len): int(block_size)
            for sequence_len, block_size in getattr(
                self.runtime_policy,
                "gemma4_e2b_b8_prefill_gated_activation_blocks",
                (),
            )
        }
        for layer in self.layers:
            attention = getattr(layer, "self_attn", None)
            if attention is not None:
                sliding_window = int(
                    getattr(attention, "sliding_window", 0) or 0
                )
                attention._gemma4_e2b_l4_sliding_prefill_enabled = bool(
                    e2b_l4_sliding_prefill and sliding_window > 0
                )
                attention._gemma4_e2b_l4_full_prefill_expand_enabled = bool(
                    e2b_l4_full_prefill_expand and sliding_window <= 0
                )
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(
                mlp,
                "_gemma4_e2b_prefill_gated_activation_enabled",
            ):
                mlp._gemma4_e2b_prefill_gated_activation_enabled = bool(
                    e2b_b8_prefill_gated_activation
                    and e2b_b8_prefill_gated_activation_blocks
                )
                mlp._gemma4_e2b_prefill_gated_activation_block_size = 0
                mlp._gemma4_e2b_prefill_gated_activation_block_sizes = dict(
                    e2b_b8_prefill_gated_activation_blocks
                )
        explicit_rmsnorm = os.environ.get(
            "MEGAGEMM_DISABLE_CUDA_RMSNORM", ""
        ).strip()
        prefer_triton_rmsnorm = (
            explicit_rmsnorm.lower() in {"1", "true", "yes", "on"}
            if explicit_rmsnorm
            else bool(self.runtime_policy.prefer_triton_rmsnorm)
        )
        self._gemma4_prefer_triton_rmsnorm = prefer_triton_rmsnorm
        for module in self.modules():
            if isinstance(module, MGRMSNorm):
                module.prefer_triton = prefer_triton_rmsnorm
        if dtype is None:
            dtype = (
                self.embed_tokens.weight.dtype
                if self.embed_tokens.weight.device.type != 'meta'
                else torch.get_default_dtype()
            )
        rope_meta = (
            self.cos_cache is None
            or self.cos_cache.device.type == 'meta'
            or (
                self.layer_rope_caches is not None
                and any(
                    cos.device.type == 'meta' or sin.device.type == 'meta'
                    for cos, sin in self.layer_rope_caches
                )
            )
        )
        if rope_meta:
            self.set_rope_cache_max_seq_len(
                getattr(self, "_rope_cache_max_seq_len", self.config.max_position_embeddings),
                device=device,
            )
        else:
            self._move_rope_to_device(device)
        self.gemma4_embed_scale = torch.tensor(
            self.embed_scale,
            device=device,
            dtype=dtype,
        )
        if self.hidden_size_per_layer_input:
            self.embed_scale_per_layer = torch.tensor(
                self.hidden_size_per_layer_input ** 0.5,
                device=device,
                dtype=dtype,
            )
            self.per_layer_input_scale = torch.tensor(
                2.0 ** -0.5,
                device=device,
                dtype=dtype,
            )
            self.per_layer_projection_scale = torch.tensor(
                self.config.hidden_size ** -0.5,
                device=device,
                dtype=dtype,
            )
        for layer in self.layers:
            scalar = getattr(layer, 'layer_scalar', None)
            if scalar is None or scalar.device.type == 'meta':
                layer.layer_scalar = torch.ones(1, device=device, dtype=dtype)

    def _scale_token_embeddings(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.embed_scale == 1.0:
            return hidden
        scale: float | torch.Tensor = self.embed_scale
        if self.config.model_type == 'gemma4_text':
            scale_tensor = self.gemma4_embed_scale
            if (
                scale_tensor is None
                or scale_tensor.device.type == 'meta'
                or scale_tensor.device != hidden.device
                or scale_tensor.dtype != hidden.dtype
            ):
                scale_tensor = torch.tensor(
                    self.embed_scale,
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                self.gemma4_embed_scale = scale_tensor
            scale = scale_tensor
        if torch.is_grad_enabled():
            return hidden * scale
        hidden.mul_(scale)
        return hidden

    def _get_layer_rope(self, layer_idx: int):
        if self.layer_rope_caches is None:
            return self.cos_cache, self.sin_cache
        return self.layer_rope_caches[layer_idx]

    def prefill_cuda_graph_eligible(
        self,
        *,
        num_seqs: int,
        total_tokens: int,
        dtype: torch.dtype,
        device_type: str,
        device_name: str,
    ) -> bool:
        return _gemma4_a100_a4b_prefill_graph_shape(
            self.config,
            num_seqs=num_seqs,
            total_tokens=total_tokens,
            dtype=dtype,
            device_type=device_type,
            device_name=device_name,
        )

    def prepare_prefill_cuda_graph_workspace(
        self,
        *,
        total_tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        """Pin large MoE intermediates whose addresses are captured by CUDA."""
        if self.config.model_type != 'gemma4_text':
            return ()
        refs = []
        top_k = int(self.config.num_experts_per_tok)
        expected_moe_layers = sum(
            bool(self.config.is_moe_layer(layer_idx))
            for layer_idx in range(len(self.layers))
        )
        for layer in self.layers:
            experts = getattr(getattr(layer, "mlp", None), "experts", None)
            if experts is None:
                continue
            refs.append(
                experts.prepare_segmented_prefill_cuda_graph_workspace(
                    rows=int(total_tokens),
                    top_k=top_k,
                    device=device,
                )
            )
        if len(refs) != expected_moe_layers or not refs:
            raise RuntimeError(
                "Gemma4 prefill graph prepared "
                f"{len(refs)}/{expected_moe_layers} MoE layer workspaces"
            )
        self._prefill_cuda_graph_workspace_bytes = sum(
            int(tensor.numel() * tensor.element_size()) for tensor in refs
        )
        return tuple(refs)

    def _get_prefill_graph_deferred_kv_buffers(
        self,
        layer_idx: int,
        shape: tuple[int, ...],
        ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return shape-persistent K/V destinations for CUDA graph replay."""
        shape = tuple(int(dim) for dim in shape)
        key = (
            int(layer_idx),
            ref.device.type,
            ref.device.index,
            ref.dtype,
            shape,
        )
        buffers = self._prefill_graph_deferred_kv_buffers.get(key)
        if buffers is None:
            capturing = False
            if ref.is_cuda:
                try:
                    capturing = bool(torch.cuda.is_current_stream_capturing())
                except Exception:
                    capturing = False
            if capturing:
                raise RuntimeError(
                    "Gemma4 deferred prefill K/V buffers were not prepared "
                    "before CUDA graph capture"
                )
            buffers = (
                torch.empty(shape, device=ref.device, dtype=ref.dtype),
                torch.empty(shape, device=ref.device, dtype=ref.dtype),
            )
            self._prefill_graph_deferred_kv_buffers[key] = buffers
        return buffers

    def decode_cuda_graph_eligible(
        self,
        *,
        num_seqs: int,
        dtype: torch.dtype,
        device_type: str,
        device_name: str,
    ) -> bool:
        return bool(
            _gemma4_l4_e2b_decode_graph_shape(
                self.config,
                num_seqs=num_seqs,
                dtype=dtype,
                device_type=device_type,
                device_name=device_name,
            )
            or _gemma4_a100_a4b_decode_graph_shape(
                self.config,
                num_seqs=num_seqs,
                dtype=dtype,
                device_type=device_type,
                device_name=device_name,
            )
        )

    def decode_unrolled_graph_burst_eligible(
        self,
        *,
        num_seqs: int,
        num_steps: int,
        dtype: torch.dtype,
        device_type: str,
        device_name: str,
    ) -> bool:
        """Gate multi-token graph capture to the measured Gemma 4 E2B/L4 path."""
        return bool(
            1 < int(num_steps) <= 8
            and _gemma4_l4_e2b_decode_graph_shape(
                self.config,
                num_seqs=num_seqs,
                dtype=dtype,
                device_type=device_type,
                device_name=device_name,
            )
        )

    def _compute_per_layer_inputs(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.hidden_size_per_layer_input <= 0:
            return None
        safe_ids = torch.where(
            (input_ids >= 0) & (input_ids < self.config.vocab_size_per_layer_input),
            input_ids,
            torch.zeros_like(input_ids),
        )
        per_layer_embeds = self.embed_tokens_per_layer(safe_ids)
        per_layer_embeds = per_layer_embeds * self.embed_scale_per_layer.to(
            device=per_layer_embeds.device,
            dtype=per_layer_embeds.dtype,
        )
        per_layer_embeds = per_layer_embeds.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection = per_layer_projection * self.per_layer_projection_scale.to(
            device=per_layer_projection.device,
            dtype=per_layer_projection.dtype,
        )
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        scale = self.per_layer_input_scale.to(
            device=per_layer_projection.device,
            dtype=per_layer_projection.dtype,
        )
        return (per_layer_projection + per_layer_embeds) * scale

    def _get_prefill_arange(self, size: int, device: torch.device) -> torch.Tensor:
        key = (
            int(size),
            device.type,
            int(device.index) if device.index is not None else -1,
        )
        if self._prefill_arange_cache_key != key:
            self._prefill_arange_cache = torch.arange(
                size, device=device, dtype=torch.long,
            )
            self._prefill_arange_cache_key = key
        return self._prefill_arange_cache

    def begin_gemma4_prefill_finite_trace(
        self,
        *,
        stop_on_nonfinite: bool = True,
    ) -> dict:
        """Enable the paid-run diagnostic without changing normal prefill policy."""
        if self.config.model_type != "gemma4_text":
            raise RuntimeError("Gemma4 prefill finite tracing requires gemma4_text")
        trace = {
            "enabled": True,
            "stop_on_nonfinite": bool(stop_on_nonfinite),
            "status": "ARMED",
            "batch_size": 0,
            "seq_len": 0,
            "events": [],
            "first_bad": None,
        }
        self._gemma4_prefill_finite_trace = trace
        return trace

    def end_gemma4_prefill_finite_trace(self) -> dict:
        """Disable finite tracing and return the last compact report."""
        trace = self._gemma4_prefill_finite_trace or {
            "enabled": False,
            "status": "NOT_ARMED",
            "events": [],
            "first_bad": None,
        }
        trace["enabled"] = False
        for layer in self.layers:
            layer._gemma4_prefill_finite_trace = None
            if getattr(layer, "self_attn", None) is not None:
                layer.self_attn._gemma4_prefill_finite_trace = None
        self._gemma4_prefill_finite_trace = None
        return trace

    def _get_prefill_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        key = (
            int(size),
            device.type,
            int(device.index) if device.index is not None else -1,
        )
        if self._prefill_causal_mask_cache_key != key:
            self._prefill_causal_mask_cache = torch.ones(
                size, size, device=device, dtype=torch.bool,
            ).tril()
            self._prefill_causal_mask_cache_key = key
        return self._prefill_causal_mask_cache

    def _apply_final_logit_capping(self, logits: torch.Tensor) -> torch.Tensor:
        """Gemma 2: soft-cap logits with tanh."""
        if self.final_logit_softcapping > 0:
            cap = self.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)
        return logits

    def _lm_head_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if _USE_DECODE_FAST_LINEAR:
            return _decode_linear(
                self,
                "_fast_lm_head_out",
                hidden,
                self.lm_head.weight,
                self.lm_head.bias,
                use_fast=True,
            )
        sig = _decode_linear_runtime_sig(hidden, self.lm_head.weight)
        if self._fast_lm_head_decode_key != sig:
            self._fast_lm_head_decode_use = bool(
                _pick_decode_linear_backend(
                    self,
                    "lm_head",
                    hidden,
                    self.lm_head.weight,
                    self.lm_head.bias,
                    "_fast_lm_head_out",
                )
            )
            self._fast_lm_head_decode_key = sig
        if self._fast_lm_head_decode_use:
            return _decode_linear(
                self,
                "_fast_lm_head_out",
                hidden,
                self.lm_head.weight,
                self.lm_head.bias,
                use_fast=True,
            )
        return _decode_linear(
            self,
            "_fast_lm_head_out",
            hidden,
            self.lm_head.weight,
            self.lm_head.bias,
            use_fast=False,
        )

    def _decode_head_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(hidden)
        logits = self._lm_head_forward(hidden)
        return self._apply_final_logit_capping(logits)

    def _get_fused_lm_head_buffers(self, hidden_norm: torch.Tensor):
        rows = int(hidden_norm.shape[0] * hidden_norm.shape[1]) if hidden_norm.dim() == 3 else int(hidden_norm.shape[0])
        vocab_size = int(self.config.vocab_size)
        out_tok = getattr(self, "_fused_lm_head_tokens", None)
        if (
            out_tok is None
            or out_tok.shape[0] < rows
            or out_tok.device != hidden_norm.device
            or out_tok.dtype != torch.long
        ):
            out_tok = torch.empty((rows,), device=hidden_norm.device, dtype=torch.long)
            self._fused_lm_head_tokens = out_tok

        # Workspace upper-bound for Triton block-reduction.
        # Kernel currently uses BLOCK_N >= 16, so this covers forced/runtime configs.
        n_blocks_cap = max(1, (vocab_size + 15) // 16)
        partial_vals = getattr(self, "_fused_lm_head_partial_vals", None)
        if (
            partial_vals is None
            or partial_vals.shape[0] < rows
            or partial_vals.shape[1] < n_blocks_cap
            or partial_vals.device != hidden_norm.device
            or partial_vals.dtype != torch.float32
        ):
            partial_vals = torch.empty((rows, n_blocks_cap), device=hidden_norm.device, dtype=torch.float32)
            self._fused_lm_head_partial_vals = partial_vals

        partial_idxs = getattr(self, "_fused_lm_head_partial_idxs", None)
        if (
            partial_idxs is None
            or partial_idxs.shape[0] < rows
            or partial_idxs.shape[1] < n_blocks_cap
            or partial_idxs.device != hidden_norm.device
            or partial_idxs.dtype != torch.int32
        ):
            partial_idxs = torch.empty((rows, n_blocks_cap), device=hidden_norm.device, dtype=torch.int32)
            self._fused_lm_head_partial_idxs = partial_idxs

        return out_tok[:rows], partial_vals[:rows, :n_blocks_cap], partial_idxs[:rows, :n_blocks_cap]

    def _decode_raw_logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the exact final-norm and LM-head decode pipeline without capping."""
        if self._flat_decode_ready:
            hidden_norm = _decode_rmsnorm(
                hidden,
                self._flat_norm_weight,
                self._flat_norm_eps,
                self._flat_norm_offset,
                prefer_triton=bool(
                    getattr(self, "_gemma4_prefer_triton_rmsnorm", False)
                ),
            )
            hidden_2d = hidden_norm.reshape(-1, hidden_norm.shape[-1])
            logits = torch.mm(hidden_2d, self._flat_lm_head_wt)
            if self._flat_lm_head_bias is not None:
                logits.add_(self._flat_lm_head_bias)
            logits = logits.view(
                hidden_norm.shape[0],
                -1,
                logits.shape[-1],
            )
        else:
            hidden_norm = self.norm(hidden)
            logits = self._lm_head_forward(hidden_norm)
        return logits

    def _decode_logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the exact final-norm, LM-head, and logit-cap decode pipeline."""
        logits = self._decode_raw_logits_from_hidden(hidden)
        return self._apply_final_logit_capping(logits)

    def _decode_next_token_greedy(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Greedy next-token path that can bypass full logits materialization.
        Input hidden is pre-final-norm state from decode loop.
        """
        vocab_size = int(self.config.vocab_size)
        rows = int(hidden.shape[0] * hidden.shape[1]) if hidden.dim() == 3 else int(hidden.shape[0])
        if hidden.is_cuda and _gemma4_a100_a4b_batch_cublas_lm_head_shape(
            self.config.model_type,
            rows,
            int(hidden.shape[-1]),
            vocab_size,
            hidden.dtype,
            torch.cuda.get_device_name(hidden.device),
        ):
            self._gemma4_batch_cublas_lm_head_hits += 1
            # Gemma4's BF16 softcap can collapse distinct logits to the same
            # value. Apply it before argmax so ties match the logits contract.
            raw_logits = self._decode_raw_logits_from_hidden(hidden)
            if (
                _GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX
                and logits_softcap_argmax is not None
                and not self._gemma4_batch_fused_softcap_argmax_disable
                and self.final_logit_softcapping > 0
            ):
                try:
                    out_tok, partial_vals, partial_idxs = (
                        self._get_fused_lm_head_buffers(hidden)
                    )
                    next_tokens = logits_softcap_argmax(
                        raw_logits[:, -1, :],
                        self.final_logit_softcapping,
                        out_tokens=out_tok,
                        partial_vals=partial_vals,
                        partial_idxs=partial_idxs,
                    )
                    self._gemma4_batch_fused_softcap_argmax_hits += 1
                    return next_tokens
                except Exception as exc:
                    self._gemma4_batch_fused_softcap_argmax_disable = True
                    self._gemma4_batch_fused_softcap_argmax_error = (
                        f"{type(exc).__name__}: {exc}"
                    )

            logits = self._apply_final_logit_capping(raw_logits)
            next_tokens = _get_reusable_out_decode_typed(
                self,
                "_gemma4_batch_cublas_lm_head_tokens",
                (rows,),
                hidden,
                torch.long,
            )
            torch.argmax(logits[:, -1, :], dim=-1, out=next_tokens)
            return next_tokens
        can_try_fused_norm_head = (
            _USE_FUSED_RMSNORM_LM_HEAD_ARGMAX_DECODE
            and lm_head_rmsnorm_argmax is not None
            and not self._fused_rmsnorm_lm_head_argmax_disable
            and not torch.is_grad_enabled()
            and hidden.is_cuda
            and getattr(self.norm, "with_scale", False)
            and callable(lm_head_argmax_prefers_triton_shape)
        )
        if not can_try_fused_norm_head:
            if not _USE_FUSED_RMSNORM_LM_HEAD_ARGMAX_DECODE:
                self._fused_rmsnorm_lm_head_argmax_skip_reason = "decode_disabled"
            elif lm_head_rmsnorm_argmax is None:
                self._fused_rmsnorm_lm_head_argmax_skip_reason = "kernel_unavailable"
            elif self._fused_rmsnorm_lm_head_argmax_disable:
                self._fused_rmsnorm_lm_head_argmax_skip_reason = "disabled_after_error"
            elif torch.is_grad_enabled():
                self._fused_rmsnorm_lm_head_argmax_skip_reason = "grad_enabled"
            elif not hidden.is_cuda:
                self._fused_rmsnorm_lm_head_argmax_skip_reason = "hidden_not_cuda"
            elif not getattr(self.norm, "with_scale", False):
                self._fused_rmsnorm_lm_head_argmax_skip_reason = "norm_without_scale"
            elif not callable(lm_head_argmax_prefers_triton_shape):
                self._fused_rmsnorm_lm_head_argmax_skip_reason = "shape_guard_unavailable"
            else:
                self._fused_rmsnorm_lm_head_argmax_skip_reason = "unknown_gate"
        if can_try_fused_norm_head:
            shape_ok = lm_head_argmax_prefers_triton_shape(
                int(hidden.shape[-1]),
                vocab_size,
                rows,
            )
            can_try_fused_norm_head = bool(shape_ok)
            if can_try_fused_norm_head:
                self._fused_rmsnorm_lm_head_argmax_skip_reason = ""
            else:
                self._fused_rmsnorm_lm_head_argmax_skip_reason = (
                    f"shape_rejected hidden={int(hidden.shape[-1])} "
                    f"vocab={vocab_size} rows={rows}"
                )

        if can_try_fused_norm_head:
            sig = (
                tuple(hidden.shape),
                str(hidden.dtype),
                hidden.device.type,
                int(hidden.device.index) if hidden.device.index is not None else -1,
                vocab_size,
                bool(getattr(self.norm, "offset", False)),
            )
            if self._fused_rmsnorm_lm_head_argmax_key != sig:
                self._fused_rmsnorm_lm_head_argmax_key = sig
                self._fused_rmsnorm_lm_head_argmax_checked = False
                self._fused_rmsnorm_lm_head_argmax_use = False
            if not self._fused_rmsnorm_lm_head_argmax_checked:
                try:
                    out_tok, partial_vals, partial_idxs = self._get_fused_lm_head_buffers(hidden)
                    tuned_gemma4 = _gemma4_a100_a4b_tuned_lm_head_shape(
                        self.config.model_type,
                        rows,
                        int(hidden.shape[-1]),
                        vocab_size,
                        hidden.dtype,
                        torch.cuda.get_device_name(hidden.device),
                    )
                    if _FORCE_FUSED_RMSNORM_LM_HEAD_ARGMAX_USE or tuned_gemma4:
                        self._fused_rmsnorm_lm_head_argmax_use = True
                        self._fused_rmsnorm_lm_head_argmax_checked = True
                        self._fused_rmsnorm_lm_head_argmax_error = ""
                    else:
                        fused_tok = lm_head_rmsnorm_argmax(
                            hidden,
                            self.norm.weight,
                            self.norm.eps,
                            self.norm.offset,
                            self.lm_head.weight,
                            self.lm_head.bias,
                            out_tokens=out_tok,
                            partial_vals=partial_vals,
                            partial_idxs=partial_idxs,
                        ).clone()
                        hidden_norm_probe = self.norm(hidden)
                        base_tok = lm_head_argmax(
                            hidden_norm_probe,
                            self.lm_head.weight,
                            self.lm_head.bias,
                        )
                        torch.cuda.synchronize()
                        same_token = bool(torch.equal(fused_tok, base_tok))
                        if same_token:
                            fused_ms = _cuda_bench_ms(
                                lambda: lm_head_rmsnorm_argmax(
                                    hidden,
                                    self.norm.weight,
                                    self.norm.eps,
                                    self.norm.offset,
                                    self.lm_head.weight,
                                    self.lm_head.bias,
                                    out_tokens=out_tok,
                                    partial_vals=partial_vals,
                                    partial_idxs=partial_idxs,
                                ),
                                iters=8,
                            )
                            base_ms = _cuda_bench_ms(
                                lambda: lm_head_argmax(
                                    self.norm(hidden),
                                    self.lm_head.weight,
                                    self.lm_head.bias,
                                ),
                                iters=8,
                            )
                            self._fused_rmsnorm_lm_head_argmax_use = bool(
                                fused_ms <= (
                                    base_ms * (1.0 - _FUSED_RMSNORM_LM_HEAD_ARGMAX_MIN_GAIN)
                                )
                            )
                        else:
                            self._fused_rmsnorm_lm_head_argmax_use = False
                        self._fused_rmsnorm_lm_head_argmax_checked = True
                except Exception as exc:
                    self._fused_rmsnorm_lm_head_argmax_error = f"{type(exc).__name__}: {exc}"
                    self._fused_rmsnorm_lm_head_argmax_disable = True
                    self._fused_rmsnorm_lm_head_argmax_use = False

            if (
                self._fused_rmsnorm_lm_head_argmax_use
                and not self._fused_rmsnorm_lm_head_argmax_disable
            ):
                try:
                    out_tok, partial_vals, partial_idxs = self._get_fused_lm_head_buffers(hidden)
                    next_tokens = lm_head_rmsnorm_argmax(
                        hidden,
                        self.norm.weight,
                        self.norm.eps,
                        self.norm.offset,
                        self.lm_head.weight,
                        self.lm_head.bias,
                        out_tokens=out_tok,
                        partial_vals=partial_vals,
                        partial_idxs=partial_idxs,
                    )
                    self._fused_rmsnorm_lm_head_argmax_hits += 1
                    return next_tokens.view(hidden.shape[0])
                except Exception as exc:
                    self._fused_rmsnorm_lm_head_argmax_error = f"{type(exc).__name__}: {exc}"
                    self._fused_rmsnorm_lm_head_argmax_disable = True
                    self._fused_rmsnorm_lm_head_argmax_use = False

        hidden_norm = self.norm(hidden)
        if self._fused_lm_head_argmax_checked and not self._fused_lm_head_argmax_use:
            logits = self._lm_head_forward(hidden_norm)
            return logits[:, -1, :].argmax(dim=-1)
        can_try_fused = (
            not self._fused_lm_head_argmax_disable
            and _cached_fused_lm_head_argmax_decision(
                self,
                "_fused_lm_head_argmax_key",
                "_fused_lm_head_argmax_use",
                hidden_norm,
                vocab_size=vocab_size,
            )
        )
        if can_try_fused:
            self._fused_lm_head_argmax_skip_reason = ""
        else:
            if self._fused_lm_head_argmax_disable:
                self._fused_lm_head_argmax_skip_reason = "disabled_after_error"
            elif not _USE_FUSED_LM_HEAD_ARGMAX_DECODE:
                self._fused_lm_head_argmax_skip_reason = "decode_disabled"
            elif lm_head_argmax is None:
                self._fused_lm_head_argmax_skip_reason = "kernel_unavailable"
            elif torch.is_grad_enabled():
                self._fused_lm_head_argmax_skip_reason = "grad_enabled"
            elif not hidden_norm.is_cuda:
                self._fused_lm_head_argmax_skip_reason = "hidden_not_cuda"
            elif not callable(lm_head_argmax_prefers_triton_shape):
                self._fused_lm_head_argmax_skip_reason = "shape_guard_unavailable"
            else:
                self._fused_lm_head_argmax_skip_reason = (
                    f"shape_or_cache_rejected hidden={int(hidden_norm.shape[-1])} "
                    f"vocab={vocab_size}"
                )

        if can_try_fused:
            if not self._fused_lm_head_argmax_checked:
                try:
                    out_tok, partial_vals, partial_idxs = self._get_fused_lm_head_buffers(hidden_norm)
                    if _FORCE_FUSED_LM_HEAD_ARGMAX_USE:
                        self._fused_lm_head_argmax_use = True
                        self._fused_lm_head_argmax_checked = True
                        self._fused_lm_head_argmax_error = ""
                    else:
                        lm_head_argmax(
                            hidden_norm,
                            self.lm_head.weight,
                            self.lm_head.bias,
                            out_tokens=out_tok,
                            partial_vals=partial_vals,
                            partial_idxs=partial_idxs,
                        )
                        _ = self._lm_head_forward(hidden_norm)[:, -1, :].argmax(dim=-1)
                        torch.cuda.synchronize()
                        fused_ms = _cuda_bench_ms(
                            lambda: lm_head_argmax(
                                hidden_norm,
                                self.lm_head.weight,
                                self.lm_head.bias,
                                out_tokens=out_tok,
                                partial_vals=partial_vals,
                                partial_idxs=partial_idxs,
                            ),
                            iters=8,
                        )
                        base_ms = _cuda_bench_ms(
                            lambda: self._lm_head_forward(hidden_norm)[:, -1, :].argmax(dim=-1),
                            iters=8,
                        )
                        self._fused_lm_head_argmax_use = bool(
                            fused_ms <= (base_ms * (1.0 - _FUSED_LM_HEAD_ARGMAX_MIN_GAIN))
                        )
                        self._fused_lm_head_argmax_checked = True
                except Exception as exc:
                    self._fused_lm_head_argmax_error = f"{type(exc).__name__}: {exc}"
                    self._fused_lm_head_argmax_disable = True
                    self._fused_lm_head_argmax_use = False

            if self._fused_lm_head_argmax_use and not self._fused_lm_head_argmax_disable:
                try:
                    out_tok, partial_vals, partial_idxs = self._get_fused_lm_head_buffers(hidden_norm)
                    next_tokens = lm_head_argmax(
                        hidden_norm,
                        self.lm_head.weight,
                        self.lm_head.bias,
                        out_tokens=out_tok,
                        partial_vals=partial_vals,
                        partial_idxs=partial_idxs,
                    )
                    return next_tokens.view(hidden_norm.shape[0])
                except Exception as exc:
                    self._fused_lm_head_argmax_error = f"{type(exc).__name__}: {exc}"
                    self._fused_lm_head_argmax_disable = True
                    self._fused_lm_head_argmax_use = False

        logits = self._lm_head_forward(hidden_norm)
        return logits[:, -1, :].argmax(dim=-1)

    def prefers_scheduler_greedy_token_decode(self, num_seqs: int) -> bool:
        weight = self.lm_head.weight
        return bool(
            weight.is_cuda
            and (
                _gemma4_l4_e2b_decode_graph_shape(
                    self.config,
                    num_seqs=int(num_seqs),
                    dtype=weight.dtype,
                    device_type=weight.device.type,
                    device_name=torch.cuda.get_device_name(weight.device),
                )
                or _gemma4_a100_a4b_batch_cublas_lm_head_shape(
                    self.config.model_type,
                    int(num_seqs),
                    int(weight.shape[-1]),
                    int(weight.shape[0]),
                    weight.dtype,
                    torch.cuda.get_device_name(weight.device),
                )
            )
        )

    def _prime_fused_lm_head_argmax(
        self,
        num_rows: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> bool:
        """
        One-time shape calibration for fused lm_head+argmax.
        Returns whether fused path should be used for current decode shape.
        """
        if not (
            _USE_FUSED_LM_HEAD_ARGMAX_DECODE
            or _USE_FUSED_RMSNORM_LM_HEAD_ARGMAX_DECODE
        ):
            return False
        if self._fused_lm_head_argmax_checked or self._fused_rmsnorm_lm_head_argmax_checked:
            return (
                bool(self._fused_lm_head_argmax_use)
                and not self._fused_lm_head_argmax_disable
            ) or (
                bool(self._fused_rmsnorm_lm_head_argmax_use)
                and not self._fused_rmsnorm_lm_head_argmax_disable
            )
        try:
            probe = torch.empty((num_rows, 1, self.config.hidden_size), device=device, dtype=dtype)
            probe.normal_()
            _ = self._decode_next_token_greedy(probe)
        except Exception as exc:
            self._fused_lm_head_argmax_error = f"prime {type(exc).__name__}: {exc}"
            self._fused_lm_head_argmax_disable = True
            self._fused_lm_head_argmax_use = False
        return (
            bool(self._fused_lm_head_argmax_use)
            and not self._fused_lm_head_argmax_disable
        ) or (
            bool(self._fused_rmsnorm_lm_head_argmax_use)
            and not self._fused_rmsnorm_lm_head_argmax_disable
        )

    def _has_linear_layers(self) -> bool:
        return any(layer.layer_type == 'linear_attention' for layer in self.layers)

    def fused_decode_stats(self) -> dict:
        """Return fused decode status per attention layer for diagnostics."""
        total = 0
        enabled = 0
        disabled = []
        hits = 0

        for idx, layer in enumerate(self.layers):
            attn = getattr(layer, 'self_attn', None)
            if attn is None:
                continue
            total += 1
            hits += int(getattr(attn, '_fused_decode_hits', 0))
            if getattr(attn, '_disable_fused_decode', False):
                disabled.append({
                    "layer": idx,
                    "reason": getattr(attn, '_fused_decode_disable_reason', None),
                })
            else:
                enabled += 1

        return {
            "global_fused_flag": _HAS_FUSED_ROPE_ATTN,
            "total_attn_layers": total,
            "enabled_layers": enabled,
            "disabled_layers": len(disabled),
            "total_fused_hits": hits,
            "disabled_detail": disabled,
        }

    def decode_runtime_stats(self) -> dict:
        """Return decode fast-path status for benchmark diagnostics."""
        fused_attn = self.fused_decode_stats()
        first_disabled = ""
        disabled_detail = fused_attn.get("disabled_detail") or []
        if disabled_detail:
            first_disabled = str(disabled_detail[0].get("reason") or "")
        linear_layers = [
            layer.linear_attn
            for layer in self.layers
            if getattr(layer, "linear_attn", None) is not None
        ]
        full_attn_layers = [
            layer.self_attn
            for layer in self.layers
            if getattr(layer, "self_attn", None) is not None
        ]
        linear_ab_hits = sum(int(getattr(layer, "_decode_ab_fused_hits", 0)) for layer in linear_layers)
        linear_ab_disabled = sum(1 for layer in linear_layers if getattr(layer, "_decode_ab_fused_disabled", False))
        linear_fused_in_hits = sum(int(getattr(layer, "_decode_fused_in_proj_hits", 0)) for layer in linear_layers)
        linear_fused_in_used = sum(
            1
            for layer in linear_layers
            if getattr(layer, "_fused_rmsnorm_in_proj_use", False)
            and not getattr(layer, "_disable_fused_rmsnorm_in_proj", False)
        )
        linear_fused_in_disabled = sum(
            1 for layer in linear_layers if getattr(layer, "_disable_fused_rmsnorm_in_proj", False)
        )
        linear_fast_in_hits = sum(int(getattr(layer, "_decode_fast_in_proj_hits", 0)) for layer in linear_layers)
        linear_fast_out_hits = sum(int(getattr(layer, "_decode_fast_out_proj_hits", 0)) for layer in linear_layers)
        linear_norm_out_hits = sum(int(getattr(layer, "_fused_norm_out_hits", 0)) for layer in linear_layers)
        linear_norm_out_used = sum(1 for layer in linear_layers if getattr(layer, "_fused_norm_out_use", False))
        linear_norm_out_disabled = sum(1 for layer in linear_layers if getattr(layer, "_fused_norm_out_disabled", False))
        norm_out_cfg = (
            rmsnorm_gated_linear_runtime_config()
            if callable(rmsnorm_gated_linear_runtime_config)
            else {}
        )
        lm_head_argmax_cfg = (
            lm_head_argmax_runtime_config()
            if callable(lm_head_argmax_runtime_config)
            else {}
        )
        fused_rmsnorm_linear_cfg = (
            fused_rmsnorm_linear_runtime_config()
            if callable(fused_rmsnorm_linear_runtime_config)
            else {}
        )
        mlp_layers = [
            layer.mlp
            for layer in self.layers
            if getattr(layer, "mlp", None) is not None
        ]
        gemma4_e2b_prefill_gated_activation_enabled_layers = sum(
            int(
                bool(
                    getattr(
                        mlp,
                        "_gemma4_e2b_prefill_gated_activation_enabled",
                        False,
                    )
                )
            )
            for mlp in mlp_layers
        )
        gemma4_e2b_prefill_gated_activation_hits = sum(
            int(
                getattr(
                    mlp,
                    "_gemma4_e2b_prefill_gated_activation_hits",
                    0,
                )
            )
            for mlp in mlp_layers
        )
        gemma4_e2b_prefill_gated_activation_disabled_layers = sum(
            int(
                bool(
                    getattr(
                        mlp,
                        "_gemma4_e2b_prefill_gated_activation_runtime_disabled",
                        False,
                    )
                )
            )
            for mlp in mlp_layers
        )
        gemma4_e2b_prefill_gated_activation_block_sizes = {}
        gemma4_e2b_prefill_gated_activation_failure = ""
        for mlp in mlp_layers:
            if getattr(
                mlp,
                "_gemma4_e2b_prefill_gated_activation_enabled",
                False,
            ):
                gemma4_e2b_prefill_gated_activation_block_sizes.update(
                    {
                        int(sequence_len): int(block_size)
                        for sequence_len, block_size in getattr(
                            mlp,
                            "_gemma4_e2b_prefill_gated_activation_block_sizes",
                            {},
                        ).items()
                    }
                )
            failure = str(
                getattr(
                    mlp,
                    "_gemma4_e2b_prefill_gated_activation_failure",
                    "",
                )
                or ""
            )
            if failure and not gemma4_e2b_prefill_gated_activation_failure:
                gemma4_e2b_prefill_gated_activation_failure = failure
        qwen3_moe_experts = [
            mlp.experts
            for mlp in mlp_layers
            if getattr(mlp, "experts", None) is not None
        ]
        qwen3_moe_int8_expert_layers = sum(
            1
            for experts in qwen3_moe_experts
            if callable(getattr(experts, "_has_int8_experts", None))
            and experts._has_int8_experts()
        )
        qwen3_moe_int8_dequant_prefill_hits = sum(
            int(getattr(experts, "_int8_dequant_prefill_hits", 0))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_int8_dequant_prefill_disabled = sum(
            1
            for experts in qwen3_moe_experts
            if getattr(experts, "_int8_dequant_prefill_disabled", False)
        )
        qwen3_moe_int8_dequant_prefill_first_failure = ""
        for experts in qwen3_moe_experts:
            if getattr(experts, "_int8_dequant_prefill_disabled", False):
                qwen3_moe_int8_dequant_prefill_first_failure = str(
                    getattr(experts, "_int8_dequant_prefill_fail_reason", "")
                )
                break
        qwen3_moe_runtime_cfg = (
            qwen3_moe_grouped_runtime_config()
            if callable(qwen3_moe_grouped_runtime_config)
            else {}
        )
        qwen3_moe_segmented_prefill_hits = sum(
            int(getattr(experts, "_segmented_prefill_hits", 0))
            for experts in qwen3_moe_experts
        )
        gemma4_long_dominant_expert_prefill_hits = sum(
            int(
                getattr(
                    experts,
                    "_gemma4_long_dominant_expert_prefill_hits",
                    0,
                )
            )
            for experts in qwen3_moe_experts
        )
        gemma4_long_dominant_expert_prefill_assignments = sum(
            int(
                getattr(
                    experts,
                    "_gemma4_long_dominant_expert_prefill_assignments",
                    0,
                )
            )
            for experts in qwen3_moe_experts
        )
        gemma4_long_dominant_expert_prefill_guard_misses = sum(
            int(
                getattr(
                    experts,
                    "_gemma4_long_dominant_expert_prefill_guard_misses",
                    0,
                )
            )
            for experts in qwen3_moe_experts
        )
        gemma4_long_dominant_expert_prefill_last_active_layers = sum(
            int(
                bool(
                    getattr(
                        experts,
                        "_gemma4_long_dominant_expert_prefill_last_active",
                        False,
                    )
                )
            )
            for experts in qwen3_moe_experts
        )
        gemma4_long_dominant_expert_prefill_guard_miss_layers = sum(
            int(
                bool(
                    getattr(
                        experts,
                        "_gemma4_long_dominant_expert_prefill_last_guard_reason",
                        "",
                    )
                )
            )
            for experts in qwen3_moe_experts
        )
        gemma4_long_dominant_expert_prefill_disabled_layers = sum(
            int(
                bool(
                    getattr(
                        experts,
                        "_gemma4_long_dominant_expert_prefill_disabled",
                        False,
                    )
                )
            )
            for experts in qwen3_moe_experts
        )
        gemma4_long_dominant_expert_prefill_profiles = []
        gemma4_long_dominant_expert_prefill_guard_rejections = []
        gemma4_long_dominant_expert_prefill_failures = []
        for layer_idx, layer in enumerate(self.layers):
            mlp = getattr(layer, "mlp", None)
            experts = getattr(mlp, "experts", None)
            if experts is None:
                continue
            if getattr(
                experts,
                "_gemma4_long_dominant_expert_prefill_last_active",
                False,
            ):
                workspace = getattr(
                    experts,
                    "_gemma4_long_dominant_expert_prefill_workspace",
                    {},
                )
                gemma4_long_dominant_expert_prefill_profiles.append(
                    {
                        "layer_idx": int(layer_idx),
                        "heavy_expert": int(
                            workspace.get("dominant_padded_bmm_heavy_expert", -1)
                        ),
                        "heavy_count": int(
                            workspace.get("dominant_padded_bmm_heavy_count", 0)
                        ),
                        "dominant_skew": float(
                            workspace.get("dominant_padded_bmm_skew", 0.0)
                        ),
                        "light_padding_ratio": float(
                            workspace.get(
                                "dominant_padded_bmm_light_padding_ratio",
                                0.0,
                            )
                        ),
                        "capacity_ratio": float(
                            workspace.get(
                                "dominant_padded_bmm_capacity_ratio",
                                0.0,
                            )
                        ),
                    }
                )
            guard_reason = str(
                getattr(
                    experts,
                    "_gemma4_long_dominant_expert_prefill_last_guard_reason",
                    "",
                )
                or ""
            )
            if guard_reason:
                gemma4_long_dominant_expert_prefill_guard_rejections.append(
                    {"layer_idx": int(layer_idx), "reason": guard_reason}
                )
            if getattr(
                experts,
                "_gemma4_long_dominant_expert_prefill_disabled",
                False,
            ):
                gemma4_long_dominant_expert_prefill_failures.append(
                    {
                        "layer_idx": int(layer_idx),
                        "reason": str(
                            getattr(
                                experts,
                                "_gemma4_long_dominant_expert_prefill_fail_reason",
                                "",
                            )
                            or ""
                        ),
                    }
                )
        gemma4_long_dominant_expert_prefill_first_failure = (
            str(gemma4_long_dominant_expert_prefill_failures[0]["reason"])
            if gemma4_long_dominant_expert_prefill_failures
            else ""
        )
        gemma4_long_padded_bmm_prefill_hits = sum(
            int(getattr(experts, "_gemma4_long_padded_bmm_prefill_hits", 0))
            for experts in qwen3_moe_experts
        )
        gemma4_long_padded_bmm_prefill_assignments = sum(
            int(
                getattr(
                    experts,
                    "_gemma4_long_padded_bmm_prefill_assignments",
                    0,
                )
            )
            for experts in qwen3_moe_experts
        )
        gemma4_long_padded_bmm_prefill_last_active_layers = sum(
            int(
                bool(
                    getattr(
                        experts,
                        "_gemma4_long_padded_bmm_prefill_last_active",
                        False,
                    )
                )
            )
            for experts in qwen3_moe_experts
        )
        gemma4_long_padded_bmm_prefill_disabled_layers = sum(
            int(
                bool(
                    getattr(
                        experts,
                        "_gemma4_long_padded_bmm_prefill_disabled",
                        False,
                    )
                )
            )
            for experts in qwen3_moe_experts
        )
        gemma4_long_padded_bmm_prefill_failures = []
        for layer_idx, layer in enumerate(self.layers):
            mlp = getattr(layer, "mlp", None)
            experts = getattr(mlp, "experts", None)
            if experts is None or not getattr(
                experts,
                "_gemma4_long_padded_bmm_prefill_disabled",
                False,
            ):
                continue
            gemma4_long_padded_bmm_prefill_failures.append(
                {
                    "layer_idx": int(layer_idx),
                    "reason": str(
                        getattr(
                            experts,
                            "_gemma4_long_padded_bmm_prefill_fail_reason",
                            "",
                        )
                        or ""
                    ),
                }
            )
        gemma4_long_padded_bmm_prefill_first_failure = (
            str(gemma4_long_padded_bmm_prefill_failures[0]["reason"])
            if gemma4_long_padded_bmm_prefill_failures
            else ""
        )
        gemma4_grouped_mm_prefill_hits = sum(
            int(getattr(experts, "_gemma4_grouped_mm_prefill_hits", 0))
            for experts in qwen3_moe_experts
        )
        gemma4_grouped_mm_prefill_selected_layers = sum(
            int(
                any(
                    bool(enabled)
                    for enabled in getattr(
                        experts,
                        "_gemma4_grouped_mm_prefill_runtime_by_rows",
                        {},
                    ).values()
                )
            )
            for experts in qwen3_moe_experts
        )
        gemma4_grouped_mm_prefill_last_active_layers = sum(
            int(
                bool(
                    getattr(
                        experts,
                        "_gemma4_grouped_mm_prefill_last_active",
                        False,
                    )
                )
            )
            for experts in qwen3_moe_experts
        )
        gemma4_grouped_mm_prefill_disabled_layers = sum(
            int(
                bool(
                    getattr(
                        experts,
                        "_gemma4_grouped_mm_prefill_disabled",
                        False,
                    )
                )
            )
            for experts in qwen3_moe_experts
        )
        gemma4_grouped_mm_prefill_first_failure = next(
            (
                str(
                    getattr(
                        experts,
                        "_gemma4_grouped_mm_prefill_fail_reason",
                        "",
                    )
                    or ""
                )
                for experts in qwen3_moe_experts
                if getattr(
                    experts,
                    "_gemma4_grouped_mm_prefill_fail_reason",
                    "",
                )
            ),
            "",
        )
        gemma4_a4b_segmented_prefill_layers = sum(
            1
            for experts in qwen3_moe_experts
            if getattr(experts, "_gemma4_a4b_segmented_prefill", False)
        )
        qwen3_moe_segmented_prefill_residual_fused_hits = sum(
            int(getattr(experts, "_segmented_prefill_residual_fused_hits", 0))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_grouped_decode_hits = sum(
            int(getattr(experts, "_grouped_decode_hits", 0))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_grouped_decode_disabled_layers = sum(
            int(bool(getattr(experts, "_grouped_decode_disabled", False)))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_grouped_decode_first_failure = next(
            (
                str(getattr(experts, "_grouped_decode_fail_reason", "") or "")
                for experts in qwen3_moe_experts
                if getattr(experts, "_grouped_decode_fail_reason", "")
            ),
            "",
        )
        gemma4_batch_decode_policy_layers = sum(
            1
            for experts in qwen3_moe_experts
            if bool(getattr(experts, "_gemma4_batch_decode_compact", False))
        )
        gemma4_batch_decode_use_compact_layers = sum(
            1
            for experts in qwen3_moe_experts
            if bool(getattr(experts, "_gemma4_batch_decode_use_compact", False))
        )
        gemma4_batch_decode_compact_hits = sum(
            int(
                getattr(experts, "_grouped_decode_workspace", {}).get(
                    "expert_grouped_compact_decode_hits",
                    0,
                )
                or 0
            )
            for experts in qwen3_moe_experts
        )
        gemma4_batch_decode_compact_disabled_layers = sum(
            1
            for experts in qwen3_moe_experts
            if int(
                getattr(experts, "_grouped_decode_workspace", {}).get(
                    "expert_grouped_compact_decode_disabled",
                    0,
                )
                or 0
            )
        )
        gemma4_batch_decode_compact_first_failure = next(
            (
                str(
                    getattr(experts, "_grouped_decode_workspace", {}).get(
                        "expert_grouped_compact_decode_fail_reason",
                        "",
                    )
                    or ""
                )
                for experts in qwen3_moe_experts
                if getattr(experts, "_grouped_decode_workspace", {}).get(
                    "expert_grouped_compact_decode_fail_reason"
                )
            ),
            "",
        )
        gemma4_batch_decode_last_paths: dict[str, int] = {}
        gemma4_batch_decode_deterministic_reduce_layers = 0
        gemma4_batch_decode_expert_grid_pack_layers = 0
        gemma4_batch_decode_coalesced_weight_layers = 0
        gemma4_batch_decode_fused_post_moe_layers = 0
        gemma4_batch_decode_active_list_layers = 0
        gemma4_batch_decode_active_list_early_exit_layers = 0
        for experts in qwen3_moe_experts:
            workspace = getattr(experts, "_grouped_decode_workspace", {})
            path = str(
                workspace.get("grouped_decode_last_path", "")
                or ""
            )
            if path:
                gemma4_batch_decode_last_paths[path] = (
                    int(gemma4_batch_decode_last_paths.get(path, 0)) + 1
                )
            reduce_key = (
                "expert_grouped_compact_decode_last_partial_reduce"
                if path == "expert_grouped_compact"
                else "grouped_decode_last_partial_reduce"
            )
            gemma4_batch_decode_deterministic_reduce_layers += int(
                bool(workspace.get(reduce_key, 0))
            )
            gemma4_batch_decode_expert_grid_pack_layers += int(
                bool(
                    workspace.get(
                        "expert_grouped_compact_decode_last_expert_grid_pack",
                        0,
                    )
                )
            )
            gemma4_batch_decode_coalesced_weight_layers += int(
                bool(
                    workspace.get(
                        "expert_grouped_compact_decode_last_coalesced_weights",
                        0,
                    )
                )
            )
            gemma4_batch_decode_fused_post_moe_layers += int(
                bool(
                    workspace.get(
                        "expert_grouped_compact_decode_last_fused_post_moe",
                        0,
                    )
                )
            )
            gemma4_batch_decode_active_list_layers += int(
                bool(
                    workspace.get(
                        "expert_grouped_compact_decode_last_active_list",
                        0,
                    )
                )
            )
            gemma4_batch_decode_active_list_early_exit_layers += int(
                bool(
                    workspace.get(
                        "expert_grouped_compact_decode_last_active_list_early_exit",
                        0,
                    )
                )
            )
        qwen3_moe_grouped_dot_disabled = sum(
            1
            for experts in qwen3_moe_experts
            if int(
                getattr(experts, "_grouped_decode_workspace", {}).get(
                    "grouped_dot_disabled",
                    0,
                )
                or 0
            )
        )
        qwen3_moe_expert_grouped_disabled = sum(
            1
            for experts in qwen3_moe_experts
            if int(
                getattr(experts, "_grouped_decode_workspace", {}).get(
                    "expert_grouped_decode_disabled",
                    0,
                )
                or 0
            )
        )
        qwen3_moe_shared_route_disabled = sum(
            1
            for experts in qwen3_moe_experts
            if int(
                getattr(experts, "_grouped_decode_workspace", {}).get(
                    "shared_route_decode_disabled",
                    0,
                )
                or 0
            )
        )
        qwen3_moe_shared_route_last_token_accum_layers = sum(
            int(
                getattr(experts, "_grouped_decode_workspace", {}).get(
                    "shared_route_decode_last_token_accum",
                    0,
                )
                or 0
            )
            for experts in qwen3_moe_experts
        )
        qwen3_moe_shared_route_last_split_gate_layers = sum(
            int(
                getattr(experts, "_grouped_decode_workspace", {}).get(
                    "shared_route_decode_last_split_gate",
                    0,
                )
                or 0
            )
            for experts in qwen3_moe_experts
        )
        qwen3_moe_shared_route_last_partial_reduce_layers = sum(
            int(
                getattr(experts, "_grouped_decode_workspace", {}).get(
                    "shared_route_decode_last_partial_reduce",
                    0,
                )
                or 0
            )
            for experts in qwen3_moe_experts
        )
        qwen3_moe_shared_route_residual_fused_layers = sum(
            int(
                getattr(experts, "_grouped_decode_workspace", {}).get(
                    "shared_route_decode_residual_fused",
                    0,
                )
                or 0
            )
            for experts in qwen3_moe_experts
        )
        qwen3_moe_route_matrix_disabled = sum(
            1
            for experts in qwen3_moe_experts
            if int(
                getattr(experts, "_grouped_decode_workspace", {}).get(
                    "route_matrix_decode_disabled",
                    0,
                )
                or 0
            )
        )
        qwen3_moe_expert_grouped_general_disabled = sum(
            1
            for experts in qwen3_moe_experts
            if int(
                getattr(experts, "_grouped_decode_workspace", {}).get(
                    "expert_grouped_general_decode_disabled",
                    0,
                )
                or 0
            )
        )
        qwen3_moe_expert_grouped_compact_disabled = sum(
            1
            for experts in qwen3_moe_experts
            if int(
                getattr(experts, "_grouped_decode_workspace", {}).get(
                    "expert_grouped_compact_decode_disabled",
                    0,
                )
                or 0
            )
        )
        qwen3_moe_shared_route_first_failure = ""
        for experts in qwen3_moe_experts:
            reason = str(
                getattr(experts, "_grouped_decode_workspace", {}).get(
                    "shared_route_decode_fail_reason",
                    "",
                )
                or ""
            )
            if reason:
                qwen3_moe_shared_route_first_failure = reason
                break
        qwen3_moe_route_matrix_first_failure = ""
        for experts in qwen3_moe_experts:
            reason = str(
                getattr(experts, "_grouped_decode_workspace", {}).get(
                    "route_matrix_decode_fail_reason",
                    "",
                )
                or ""
            )
            if reason:
                qwen3_moe_route_matrix_first_failure = reason
                break
        qwen3_moe_expert_grouped_general_first_failure = ""
        for experts in qwen3_moe_experts:
            reason = str(
                getattr(experts, "_grouped_decode_workspace", {}).get(
                    "expert_grouped_general_decode_fail_reason",
                    "",
                )
                or ""
            )
            if reason:
                qwen3_moe_expert_grouped_general_first_failure = reason
                break
        qwen3_moe_expert_grouped_compact_first_failure = ""
        for experts in qwen3_moe_experts:
            reason = str(
                getattr(experts, "_grouped_decode_workspace", {}).get(
                    "expert_grouped_compact_decode_fail_reason",
                    "",
                )
                or ""
            )
            if reason:
                qwen3_moe_expert_grouped_compact_first_failure = reason
                break
        qwen3_moe_expert_grouped_first_failure = ""
        for experts in qwen3_moe_experts:
            reason = str(
                getattr(experts, "_grouped_decode_workspace", {}).get(
                    "expert_grouped_decode_fail_reason",
                    "",
                )
                or ""
            )
            if reason:
                qwen3_moe_expert_grouped_first_failure = reason
                break
        qwen3_moe_segmented_prefill_disabled = sum(
            1
            for experts in qwen3_moe_experts
            if getattr(experts, "_segmented_prefill_disabled", False)
        )
        qwen3_moe_segmented_prefill_assignments = sum(
            int(getattr(experts, "_segmented_prefill_assignments", 0))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_prefill_tiles = sum(
            int(getattr(experts, "_segmented_prefill_tiles", 0))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_async_tile_hits = sum(
            int(getattr(experts, "_segmented_prefill_async_tile_hits", 0))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_max_tiles = sum(
            int(getattr(experts, "_segmented_prefill_max_tiles", 0))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_partial_reduce_hits = sum(
            int(getattr(experts, "_segmented_prefill_partial_reduce_hits", 0))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_partial_reduce_layers = sum(
            int(
                getattr(experts, "_segmented_prefill_workspace", {}).get(
                    "segmented_prefill_partial_reduce",
                    0,
                )
                or 0
            )
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_sorted_partial_hits = sum(
            int(getattr(experts, "_segmented_prefill_sorted_partial_hits", 0))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_sorted_partial_layers = sum(
            int(
                getattr(experts, "_segmented_prefill_workspace", {}).get(
                    "segmented_prefill_sorted_partial",
                    0,
                )
                or 0
            )
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_deterministic_reduce_layers = sum(
            int(
                getattr(experts, "_segmented_prefill_workspace", {}).get(
                    "segmented_prefill_deterministic_reduce",
                    0,
                )
                or 0
            )
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_atomic_reduce_layers = sum(
            int(
                getattr(experts, "_segmented_prefill_workspace", {}).get(
                    "segmented_prefill_atomic_reduce",
                    0,
                )
                or 0
            )
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_single_accumulator_hits = sum(
            int(
                getattr(
                    experts,
                    "_segmented_prefill_single_accumulator_hits",
                    0,
                )
            )
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_single_accumulator_layers = sum(
            int(
                getattr(experts, "_segmented_prefill_workspace", {}).get(
                    "segmented_prefill_single_accumulator",
                    0,
                )
                or 0
            )
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_fixed_route_pack_hits = sum(
            int(getattr(experts, "_segmented_prefill_fixed_route_pack_hits", 0))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_compact_route_pack_hits = sum(
            int(getattr(experts, "_segmented_prefill_compact_route_pack_hits", 0))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_compact_route_pack_layers = sum(
            int(
                getattr(experts, "_segmented_prefill_workspace", {}).get(
                    "segmented_prefill_compact_route_pack",
                    0,
                )
                or 0
            )
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_compact_route_pack_single_scan_layers = sum(
            int(
                getattr(experts, "_segmented_prefill_workspace", {}).get(
                    "segmented_prefill_compact_route_pack_passes",
                    0,
                )
                == 1
            )
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_graph_route_pack_layers = sum(
            int(
                getattr(experts, "_segmented_prefill_workspace", {}).get(
                    "segmented_prefill_graph_route_pack",
                    0,
                )
                or 0
            )
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_route_scatter_hits = sum(
            int(getattr(experts, "_segmented_prefill_route_scatter_hits", 0))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_route_argsort_hits = sum(
            int(getattr(experts, "_segmented_prefill_route_argsort_hits", 0))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_prefill_residual_fused_layers = sum(
            int(
                getattr(experts, "_segmented_prefill_workspace", {}).get(
                    "segmented_prefill_residual_fused",
                    0,
                )
                or 0
            )
            for experts in qwen3_moe_experts
        )
        qwen3_moe_segmented_prefill_first_failure = ""
        qwen3_moe_segmented_route_scatter_first_failure = ""
        for experts in qwen3_moe_experts:
            if getattr(experts, "_segmented_prefill_disabled", False):
                qwen3_moe_segmented_prefill_first_failure = str(
                    getattr(experts, "_segmented_prefill_fail_reason", "")
                )
                break
        for experts in qwen3_moe_experts:
            route_failure = str(
                getattr(experts, "_segmented_prefill_route_scatter_fail_reason", "")
            )
            if route_failure:
                qwen3_moe_segmented_route_scatter_first_failure = route_failure
                break
        qwen3_moe_bucketed_prefill_hits = sum(
            int(getattr(experts, "_bucketed_prefill_hits", 0))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_batched_prefill_hits = sum(
            int(getattr(experts, "_batched_prefill_hits", 0))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_sorted_prefill_hits = sum(
            int(getattr(experts, "_sorted_prefill_hits", 0))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_bucketed_prefill_disabled = sum(
            1
            for experts in qwen3_moe_experts
            if getattr(experts, "_bucketed_prefill_disabled", False)
        )
        qwen3_moe_bucketed_prefill_valid_assignments = sum(
            int(getattr(experts, "_bucketed_prefill_valid_assignments", 0))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_bucketed_prefill_padded_assignments = sum(
            int(getattr(experts, "_bucketed_prefill_padded_assignments", 0))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_bucketed_prefill_bucket_launches = sum(
            int(getattr(experts, "_bucketed_prefill_bucket_launches", 0))
            for experts in qwen3_moe_experts
        )
        qwen3_moe_bucketed_prefill_pad_waste = 0.0
        if qwen3_moe_bucketed_prefill_padded_assignments > 0:
            qwen3_moe_bucketed_prefill_pad_waste = max(
                0.0,
                1.0
                - (
                    float(qwen3_moe_bucketed_prefill_valid_assignments)
                    / float(qwen3_moe_bucketed_prefill_padded_assignments)
                ),
            )
        qwen3_moe_bucketed_prefill_first_failure = ""
        for experts in qwen3_moe_experts:
            if getattr(experts, "_bucketed_prefill_disabled", False):
                qwen3_moe_bucketed_prefill_first_failure = str(
                    getattr(experts, "_bucketed_prefill_fail_reason", "")
                )
                break
        deepfusion_checked = sum(
            1 for mlp in mlp_layers if getattr(mlp, "_decode_deepfusion_checked", False)
        )
        deepfusion_used = sum(
            1
            for mlp in mlp_layers
            if getattr(mlp, "_deepfusion_decode_use", False)
            and not getattr(mlp, "_decode_disable_deepfusion", False)
        )
        deepfusion_disabled = sum(
            1 for mlp in mlp_layers if getattr(mlp, "_decode_disable_deepfusion", False)
        )
        deepfusion_hits = sum(
            int(getattr(mlp, "_deepfusion_decode_hits", 0)) for mlp in mlp_layers
        )
        fused_gateup_checked = sum(1 for mlp in mlp_layers if getattr(mlp, "_fused_rmsnorm_gateup_checked", False))
        fused_gateup_used = sum(
            1
            for mlp in mlp_layers
            if getattr(mlp, "_fused_rmsnorm_gateup_decode_use", False)
            and not getattr(mlp, "_disable_fused_rmsnorm_gateup_decode", False)
        )
        fused_gateup_disabled = sum(
            1
            for mlp in mlp_layers
            if getattr(mlp, "_disable_fused_rmsnorm_gateup_decode", False)
        )
        fused_qkv_checked = sum(1 for attn in full_attn_layers if getattr(attn, "_fused_rmsnorm_qkv_checked", False))
        fused_qkv_used = sum(
            1
            for attn in full_attn_layers
            if getattr(attn, "_fused_rmsnorm_qkv_decode_use", False)
            and not getattr(attn, "_disable_fused_rmsnorm_qkv_decode", False)
        )
        fused_qkv_disabled = sum(
            1
            for attn in full_attn_layers
            if getattr(attn, "_disable_fused_rmsnorm_qkv_decode", False)
        )
        gemma4_fused_qkv_prefill_hits = sum(
            int(getattr(attn, "_gemma4_fused_qkv_prefill_hits", 0))
            for attn in full_attn_layers
        )
        gemma4_fused_qkv_prefill_skip_reason = ""
        for attn in full_attn_layers:
            reason = str(
                getattr(attn, "_gemma4_fused_qkv_prefill_skip_reason", "") or ""
            )
            if reason:
                gemma4_fused_qkv_prefill_skip_reason = reason
                break
        gemma4_fused_attn_prepare_hits = sum(
            int(getattr(attn, "_gemma4_fused_attn_prepare_hits", 0))
            for attn in full_attn_layers
        )
        gemma4_fused_attn_prepare_disabled_layers = sum(
            1
            for attn in full_attn_layers
            if getattr(attn, "_gemma4_fused_attn_prepare_disabled", False)
        )
        gemma4_fused_attn_prepare_skip_reason = ""
        for attn in full_attn_layers:
            reason = str(
                getattr(attn, "_gemma4_fused_attn_prepare_skip_reason", "") or ""
            )
            if reason:
                gemma4_fused_attn_prepare_skip_reason = reason
                break
        gemma4_implicit_causal_prefill_hits = sum(
            int(getattr(attn, "_gemma4_implicit_causal_prefill_hits", 0))
            for attn in full_attn_layers
        )
        gemma4_e2b_l4_sliding_prefill_hits = sum(
            int(
                getattr(
                    attn,
                    "_gemma4_e2b_l4_sliding_prefill_hits",
                    0,
                )
            )
            for attn in full_attn_layers
        )
        gemma4_e2b_l4_sliding_prefill_enabled_layers = sum(
            int(
                bool(
                    getattr(
                        attn,
                        "_gemma4_e2b_l4_sliding_prefill_enabled",
                        False,
                    )
                )
            )
            for attn in full_attn_layers
        )
        gemma4_e2b_l4_full_prefill_expand_hits = sum(
            int(
                getattr(
                    attn,
                    "_gemma4_e2b_l4_full_prefill_expand_hits",
                    0,
                )
            )
            for attn in full_attn_layers
        )
        gemma4_e2b_l4_full_prefill_expand_enabled_layers = sum(
            int(
                bool(
                    getattr(
                        attn,
                        "_gemma4_e2b_l4_full_prefill_expand_enabled",
                        False,
                    )
                )
            )
            for attn in full_attn_layers
        )
        gemma4_e2b_l4_full_prefill_expand_error = next(
            (
                str(
                    getattr(
                        attn,
                        "_gemma4_e2b_l4_full_prefill_expand_error",
                        "",
                    )
                )
                for attn in full_attn_layers
                if getattr(
                    attn,
                    "_gemma4_e2b_l4_full_prefill_expand_error",
                    "",
                )
            ),
            "",
        )
        gemma4_long_sliding_prefill_hits = sum(
            int(getattr(attn, "_gemma4_long_sliding_prefill_hits", 0))
            for attn in full_attn_layers
        )
        gemma4_long_full_prefill_hits = sum(
            int(getattr(attn, "_gemma4_long_full_prefill_hits", 0))
            for attn in full_attn_layers
        )
        gemma4_parallel_moe_prefill_hits = sum(
            int(getattr(layer, "_gemma4_parallel_moe_prefill_hits", 0))
            for layer in self.layers
        )
        fast_gemv_splitk_hits = sum(
            int(getattr(module, "_fast_gemv_splitk_hits", 0))
            for module in self.modules()
        )
        rmsnorm_no_weight_triton_hits = sum(
            int(getattr(module, "_triton_no_weight_hits", 0))
            for module in self.modules()
        )
        gemma4_router_fused_norm_scale_hits = sum(
            int(getattr(module, "_fused_norm_scale_hits", 0))
            for module in self.modules()
        )
        gemma4_router_fused_topk_scale_hits = sum(
            int(getattr(module, "_fused_topk_expert_scale_hits", 0))
            for module in self.modules()
        )
        gemma4_router_fused_prefill_hits = sum(
            int(getattr(module, "_fused_prefill_hits", 0))
            for module in self.modules()
        )
        gemma4_router_fused_prefill_selected = sum(
            1
            for module in self.modules()
            if any(
                bool(enabled)
                for enabled in getattr(
                    module,
                    "_fused_prefill_runtime_by_rows",
                    {},
                ).values()
            )
        )
        gemma4_router_fused_prefill_disabled = sum(
            1
            for module in self.modules()
            if getattr(module, "_fused_prefill_disabled", False)
        )
        gemma4_router_fused_prefill_error = next(
            (
                str(getattr(module, "_fused_prefill_error", ""))
                for module in self.modules()
                if getattr(module, "_fused_prefill_error", "")
            ),
            "",
        )
        gemma4_router_fused_decode_hits = sum(
            int(getattr(module, "_fused_decode_hits", 0))
            for module in self.modules()
        )
        gemma4_router_fused_decode_selected = sum(
            1
            for module in self.modules()
            if getattr(module, "_fused_decode_selected", False)
        )
        gemma4_router_fused_decode_disabled = sum(
            1
            for module in self.modules()
            if getattr(module, "_fused_decode_disabled", False)
        )
        gemma4_router_fused_decode_error = next(
            (
                str(getattr(module, "_fused_decode_error", ""))
                for module in self.modules()
                if getattr(module, "_fused_decode_error", "")
            ),
            "",
        )
        gemma4_router_compact_pack_disabled = sum(
            1
            for module in self.modules()
            if getattr(module, "_compact_route_pack_disabled", False)
        )
        gemma4_router_compact_pack_error = next(
            (
                str(getattr(module, "_compact_route_pack_error", ""))
                for module in self.modules()
                if getattr(module, "_compact_route_pack_error", "")
            ),
            "",
        )
        gemma4_router_compact_pack_workspace_disabled = sum(
            1
            for experts in qwen3_moe_experts
            if int(
                experts._grouped_decode_workspace.get(
                    "expert_grouped_compact_route_prepacked_disabled",
                    0,
                )
                or 0
            )
        )
        gemma4_router_compact_pack_workspace_error = next(
            (
                str(
                    experts._grouped_decode_workspace.get(
                        "expert_grouped_compact_route_prepacked_fail_reason",
                        "",
                    )
                    or ""
                )
                for experts in qwen3_moe_experts
                if experts._grouped_decode_workspace.get(
                    "expert_grouped_compact_route_prepacked_fail_reason",
                    "",
                )
            ),
            "",
        )
        gemma4_router_fused_decode_last_paths: dict[str, int] = {}
        for module in self.modules():
            path = str(getattr(module, "_fused_decode_last_path", "") or "")
            if path:
                gemma4_router_fused_decode_last_paths[path] = (
                    int(gemma4_router_fused_decode_last_paths.get(path, 0)) + 1
                )
        gemma4_fused_dual_ffn_norm_hits = sum(
            int(getattr(layer, "_gemma4_fused_dual_ffn_norm_hits", 0))
            for layer in self.layers
        )
        gemma4_fused_add_ffn_norm_hits = sum(
            int(getattr(layer, "_gemma4_fused_add_ffn_norm_hits", 0))
            for layer in self.layers
        )
        gemma4_fused_post_ffn_norm_hits = sum(
            int(getattr(layer, "_gemma4_fused_post_ffn_norm_hits", 0))
            for layer in self.layers
        )
        gemma4_fused_attn_moe_bridge_prefill_enabled_layers = sum(
            int(
                any(
                    bool(enabled)
                    for enabled in getattr(
                        layer,
                        "_gemma4_prefill_attn_moe_bridge_runtime_by_rows",
                        {},
                    ).values()
                )
            )
            for layer in self.layers
        )
        gemma4_fused_attn_moe_bridge_prefill_hits = sum(
            int(
                getattr(
                    layer,
                    "_gemma4_fused_attn_moe_bridge_prefill_hits",
                    0,
                )
            )
            for layer in self.layers
        )
        gemma4_fused_attn_moe_router_bridge_prefill_hits = sum(
            int(
                getattr(
                    layer,
                    "_gemma4_fused_attn_moe_router_bridge_prefill_hits",
                    0,
                )
            )
            for layer in self.layers
        )
        gemma4_prefill_attn_moe_bridge_error = next(
            (
                str(
                    getattr(
                        layer,
                        "_gemma4_prefill_attn_moe_bridge_error",
                        "",
                    )
                )
                for layer in self.layers
                if getattr(layer, "_gemma4_prefill_attn_moe_bridge_error", "")
            ),
            "",
        )
        gemma4_fused_post_moe_norm_residual_prefill_enabled_layers = sum(
            int(
                any(
                    bool(enabled)
                    for enabled in getattr(
                        layer,
                        "_gemma4_prefill_moe_tail_runtime_by_rows",
                        {},
                    ).values()
                )
            )
            for layer in self.layers
        )
        gemma4_fused_post_moe_norm_residual_prefill_hits = sum(
            int(
                getattr(
                    layer,
                    "_gemma4_fused_post_moe_norm_residual_prefill_hits",
                    0,
                )
            )
            for layer in self.layers
        )
        gemma4_prefill_moe_tail_error = next(
            (
                str(getattr(layer, "_gemma4_prefill_moe_tail_error", ""))
                for layer in self.layers
                if getattr(layer, "_gemma4_prefill_moe_tail_error", "")
            ),
            "",
        )
        gemma4_fused_dual_ffn_norm_disabled_layers = sum(
            int(bool(getattr(layer, "_gemma4_fused_dual_ffn_norm_disabled", False)))
            for layer in self.layers
        )
        gemma4_fused_add_ffn_norm_disabled_layers = sum(
            int(bool(getattr(layer, "_gemma4_fused_add_ffn_norm_disabled", False)))
            for layer in self.layers
        )
        gemma4_fused_post_ffn_norm_disabled_layers = sum(
            int(bool(getattr(layer, "_gemma4_fused_post_ffn_norm_disabled", False)))
            for layer in self.layers
        )
        gemma4_fused_dual_ffn_norm_error = next(
            (
                str(getattr(layer, "_gemma4_fused_dual_ffn_norm_error", ""))
                for layer in self.layers
                if getattr(layer, "_gemma4_fused_dual_ffn_norm_error", "")
            ),
            "",
        )
        gemma4_fused_add_ffn_norm_error = next(
            (
                str(getattr(layer, "_gemma4_fused_add_ffn_norm_error", ""))
                for layer in self.layers
                if getattr(layer, "_gemma4_fused_add_ffn_norm_error", "")
            ),
            "",
        )
        gemma4_fused_post_ffn_norm_error = next(
            (
                str(getattr(layer, "_gemma4_fused_post_ffn_norm_error", ""))
                for layer in self.layers
                if getattr(layer, "_gemma4_fused_post_ffn_norm_error", "")
            ),
            "",
        )
        fast_gemv_splitk_cached_ops = sorted(
            {
                str(key[0])
                for key, mode in _DECODE_LINEAR_MODE_CACHE.items()
                if mode == "splitk"
                and isinstance(key, tuple)
                and len(key) > 0
                and str(key[0]) != "gate_up_mode"
            }
        )
        fast_gemv_cached_modes = {}
        for key, mode in _DECODE_LINEAR_MODE_CACHE.items():
            if not isinstance(key, tuple) or not key:
                continue
            op_name = str(key[0])
            mode_name = str(mode or "tile")
            if op_name == "gate_up_mode":
                op_name = "gate_up"
            fast_gemv_cached_modes.setdefault(op_name, {})
            fast_gemv_cached_modes[op_name][mode_name] = (
                int(fast_gemv_cached_modes[op_name].get(mode_name, 0)) + 1
            )
        return {
            "paged_decode_runtime": paged_decode_runtime_stats(),
            "runtime_policy": self.runtime_policy.to_dict(),
            "mgx_sparsity_runtime": dict(
                getattr(self, "_mgx_sparsity_runtime", {"active": False, "format": "none"})
            ),
            "flat_decode_ready": bool(getattr(self, "_flat_decode_ready", False)),
            "prefill_last_token_only_hits": int(
                getattr(self, "_prefill_last_token_only_hits", 0)
            ),
            "gemma4_batch_prefill_vectorized_kv_hits": int(
                getattr(self, "_gemma4_batch_prefill_vectorized_kv_hits", 0)
            ),
            "gemma4_prefill_graph_deferred_kv_buffers": int(
                len(getattr(self, "_prefill_graph_deferred_kv_buffers", {}))
            ),
            "gemma4_prefill_graph_deferred_kv_bytes": int(
                sum(
                    tensor.numel() * tensor.element_size()
                    for pair in getattr(
                        self,
                        "_prefill_graph_deferred_kv_buffers",
                        {},
                    ).values()
                    for tensor in pair
                )
            ),
            "gemma4_prefill_graph_deferred_kv_copy_dispatches": int(
                getattr(
                    self,
                    "_prefill_graph_deferred_kv_copy_dispatches",
                    0,
                )
            ),
            "gemma4_implicit_causal_prefill_enabled": bool(
                _GEMMA4_IMPLICIT_CAUSAL_PREFILL
            ),
            "gemma4_implicit_causal_prefill_batches": int(
                getattr(self, "_gemma4_implicit_causal_prefill_batches", 0)
            ),
            "gemma4_implicit_causal_prefill_hits": int(
                gemma4_implicit_causal_prefill_hits
            ),
            "gemma4_e2b_l4_sliding_prefill_enabled": bool(
                gemma4_e2b_l4_sliding_prefill_enabled_layers > 0
            ),
            "gemma4_e2b_l4_sliding_prefill_enabled_layers": int(
                gemma4_e2b_l4_sliding_prefill_enabled_layers
            ),
            "gemma4_e2b_l4_sliding_prefill_hits": int(
                gemma4_e2b_l4_sliding_prefill_hits
            ),
            "gemma4_e2b_l4_full_prefill_expand_enabled": bool(
                gemma4_e2b_l4_full_prefill_expand_enabled_layers > 0
            ),
            "gemma4_e2b_l4_full_prefill_expand_enabled_layers": int(
                gemma4_e2b_l4_full_prefill_expand_enabled_layers
            ),
            "gemma4_e2b_l4_full_prefill_expand_hits": int(
                gemma4_e2b_l4_full_prefill_expand_hits
            ),
            "gemma4_e2b_l4_full_prefill_expand_error": (
                gemma4_e2b_l4_full_prefill_expand_error
            ),
            "gemma4_e2b_b8_prefill_gated_activation_enabled": bool(
                gemma4_e2b_prefill_gated_activation_enabled_layers > 0
            ),
            "gemma4_e2b_b8_prefill_gated_activation_enabled_layers": int(
                gemma4_e2b_prefill_gated_activation_enabled_layers
            ),
            "gemma4_e2b_b8_prefill_gated_activation_block_sizes": dict(
                sorted(gemma4_e2b_prefill_gated_activation_block_sizes.items())
            ),
            "gemma4_e2b_b8_prefill_gated_activation_hits": int(
                gemma4_e2b_prefill_gated_activation_hits
            ),
            "gemma4_e2b_b8_prefill_gated_activation_disabled_layers": int(
                gemma4_e2b_prefill_gated_activation_disabled_layers
            ),
            "gemma4_e2b_b8_prefill_gated_activation_failure": (
                gemma4_e2b_prefill_gated_activation_failure
            ),
            "gemma4_long_sliding_prefill_enabled": bool(
                _GEMMA4_LONG_SLIDING_PREFILL
            ),
            "gemma4_long_sliding_prefill_hits": int(
                gemma4_long_sliding_prefill_hits
            ),
            "gemma4_long_full_prefill_enabled": bool(
                _GEMMA4_LONG_FULL_PREFILL
            ),
            "gemma4_long_full_prefill_hits": int(
                gemma4_long_full_prefill_hits
            ),
            "gemma4_parallel_moe_prefill_enabled": bool(
                getattr(self, "_gemma4_parallel_moe_prefill_enabled", False)
            ),
            "gemma4_parallel_moe_prefill_hits": int(
                gemma4_parallel_moe_prefill_hits
            ),
            "gemma4_parallel_moe_prefill_policy": dict(
                getattr(self, "_gemma4_parallel_moe_prefill_policy", {})
            ),
            "flat_decode_failed": bool(getattr(self, "_flat_decode_failed", False)),
            "flat_decode_failed_reason": str(getattr(self, "_flat_decode_failed_reason", "")),
            "flat_decode_hybrid": bool(getattr(self, "_flat_is_hybrid", False)),
            "flat_decode_hybrid_hits": int(getattr(self, "_flat_hybrid_hits", 0)),
            "flat_decode_hybrid_full_inline_enabled": bool(
                getattr(self, "_flat_hybrid_full_inline_enabled", False)
            ),
            "flat_decode_hybrid_full_inline_max_hidden": int(
                _QWEN35_FLAT_HYBRID_FULL_INLINE_MAX_HIDDEN
            ),
            "flat_decode_hybrid_full_inline_hits": int(
                getattr(self, "_flat_hybrid_full_inline_hits", 0)
            ),
            "flat_decode_hybrid_full_fallback_hits": int(
                getattr(self, "_flat_hybrid_full_fallback_hits", 0)
            ),
            "flat_has_output_gate": bool(getattr(self, "_flat_has_output_gate", False)),
            "fused_rope_attn_available": bool(fused_attn.get("global_fused_flag", False)),
            "fused_rope_attn_enabled_layers": int(fused_attn.get("enabled_layers", 0)),
            "fused_rope_attn_disabled_layers": int(fused_attn.get("disabled_layers", 0)),
            "fused_rope_attn_total_hits": int(fused_attn.get("total_fused_hits", 0)),
            "fused_rope_attn_first_disabled_reason": first_disabled,
            "fused_rmsnorm_qkv_checked_layers": int(fused_qkv_checked),
            "fused_rmsnorm_qkv_used_layers": int(fused_qkv_used),
            "fused_rmsnorm_qkv_disabled_layers": int(fused_qkv_disabled),
            "gemma4_fused_qkv_prefill_enabled": bool(_GEMMA4_FUSED_QKV_PREFILL),
            "gemma4_fused_qkv_prefill_hits": int(gemma4_fused_qkv_prefill_hits),
            "gemma4_fused_qkv_prefill_skip_reason": (
                gemma4_fused_qkv_prefill_skip_reason
            ),
            "gemma4_fused_attn_prepare_enabled": bool(
                _GEMMA4_FUSED_ATTN_PREP_PREFILL
                and HAS_GEMMA4_ATTENTION_PREPARE
            ),
            "gemma4_fused_attn_prepare_hits": int(
                gemma4_fused_attn_prepare_hits
            ),
            "gemma4_fused_attn_prepare_disabled_layers": int(
                gemma4_fused_attn_prepare_disabled_layers
            ),
            "gemma4_fused_attn_prepare_skip_reason": (
                gemma4_fused_attn_prepare_skip_reason
            ),
            "gemma4_router_fused_norm_scale_hits": int(
                gemma4_router_fused_norm_scale_hits
            ),
            "gemma4_router_fused_topk_scale_hits": int(
                gemma4_router_fused_topk_scale_hits
            ),
            "gemma4_router_fused_prefill_enabled": bool(
                _GEMMA4_FUSED_MOE_ROUTER_PREFILL
            ),
            "gemma4_router_fused_prefill_hits": int(
                gemma4_router_fused_prefill_hits
            ),
            "gemma4_router_fused_prefill_selected_layers": int(
                gemma4_router_fused_prefill_selected
            ),
            "gemma4_router_fused_prefill_disabled_layers": int(
                gemma4_router_fused_prefill_disabled
            ),
            "gemma4_router_fused_prefill_error": (
                gemma4_router_fused_prefill_error
            ),
            "gemma4_router_fused_decode_enabled": bool(
                _GEMMA4_FUSED_MOE_ROUTER_DECODE
            ),
            "gemma4_router_fused_decode_hits": int(
                gemma4_router_fused_decode_hits
            ),
            "gemma4_router_fused_decode_selected_layers": int(
                gemma4_router_fused_decode_selected
            ),
            "gemma4_router_fused_decode_disabled_layers": int(
                gemma4_router_fused_decode_disabled
            ),
            "gemma4_router_fused_decode_last_paths": dict(
                gemma4_router_fused_decode_last_paths
            ),
            "gemma4_router_fused_decode_error": (
                gemma4_router_fused_decode_error
            ),
            "gemma4_router_compact_pack_disabled_layers": int(
                gemma4_router_compact_pack_disabled
                + gemma4_router_compact_pack_workspace_disabled
            ),
            "gemma4_router_compact_pack_error": (
                gemma4_router_compact_pack_error
                or gemma4_router_compact_pack_workspace_error
            ),
            "gemma4_router_compact_pack_workspace_disabled_layers": int(
                gemma4_router_compact_pack_workspace_disabled
            ),
            "gemma4_router_compact_pack_workspace_error": (
                gemma4_router_compact_pack_workspace_error
            ),
            "gemma4_fused_dual_ffn_norm_prefill_enabled": bool(
                _GEMMA4_FUSED_DUAL_FFN_NORM_PREFILL
            ),
            "gemma4_fused_dual_ffn_norm_prefill_hits": int(
                gemma4_fused_dual_ffn_norm_hits
            ),
            "gemma4_fused_dual_ffn_norm_prefill_disabled_layers": int(
                gemma4_fused_dual_ffn_norm_disabled_layers
            ),
            "gemma4_fused_dual_ffn_norm_prefill_error": (
                gemma4_fused_dual_ffn_norm_error
            ),
            "gemma4_fused_add_ffn_norm_prefill_enabled": bool(
                _GEMMA4_FUSED_ADD_FFN_NORM_PREFILL
            ),
            "gemma4_fused_add_ffn_norm_prefill_hits": int(
                gemma4_fused_add_ffn_norm_hits
            ),
            "gemma4_fused_add_ffn_norm_prefill_disabled_layers": int(
                gemma4_fused_add_ffn_norm_disabled_layers
            ),
            "gemma4_fused_add_ffn_norm_prefill_error": (
                gemma4_fused_add_ffn_norm_error
            ),
            "gemma4_fused_post_ffn_norm_prefill_enabled": bool(
                _GEMMA4_FUSED_POST_FFN_NORMS_PREFILL
            ),
            "gemma4_fused_post_ffn_norm_prefill_hits": int(
                gemma4_fused_post_ffn_norm_hits
            ),
            "gemma4_fused_post_ffn_norm_prefill_disabled_layers": int(
                gemma4_fused_post_ffn_norm_disabled_layers
            ),
            "gemma4_fused_post_ffn_norm_prefill_error": (
                gemma4_fused_post_ffn_norm_error
            ),
            "gemma4_fused_attn_moe_bridge_prefill_enabled": bool(
                gemma4_fused_attn_moe_bridge_prefill_enabled_layers > 0
            ),
            "gemma4_fused_attn_moe_bridge_prefill_enabled_layers": int(
                gemma4_fused_attn_moe_bridge_prefill_enabled_layers
            ),
            "gemma4_fused_attn_moe_bridge_prefill_hits": int(
                gemma4_fused_attn_moe_bridge_prefill_hits
            ),
            "gemma4_fused_attn_moe_router_bridge_prefill_hits": int(
                gemma4_fused_attn_moe_router_bridge_prefill_hits
            ),
            "gemma4_prefill_attn_moe_bridge_error": (
                gemma4_prefill_attn_moe_bridge_error
            ),
            "gemma4_fused_post_moe_norm_residual_prefill_enabled": bool(
                gemma4_fused_post_moe_norm_residual_prefill_enabled_layers > 0
            ),
            "gemma4_fused_post_moe_norm_residual_prefill_enabled_layers": int(
                gemma4_fused_post_moe_norm_residual_prefill_enabled_layers
            ),
            "gemma4_fused_post_moe_norm_residual_prefill_hits": int(
                gemma4_fused_post_moe_norm_residual_prefill_hits
            ),
            "gemma4_prefill_moe_tail_error": gemma4_prefill_moe_tail_error,
            "fused_rmsnorm_qkv_graph_guard_enabled": bool(
                _DECODE_CUDA_GRAPHS_ENABLED
                and not _FUSED_RMSNORM_QKV_ALLOW_CUDA_GRAPHS
            ),
            "linear_attn_fused_ab_available": bool(recurrent_gated_delta_decode_from_ab is not None),
            "linear_attn_fused_ab_layers": len(linear_layers),
            "linear_attn_fused_ab_disabled_layers": int(linear_ab_disabled),
            "linear_attn_fused_ab_total_hits": int(linear_ab_hits),
            "linear_attn_fused_rmsnorm_in_proj_enabled": bool(_QWEN35_FUSED_RMSNORM_IN_PROJ_DECODE),
            "linear_attn_fused_rmsnorm_in_proj_max_hidden": int(
                _QWEN35_FUSED_RMSNORM_IN_PROJ_MAX_HIDDEN
            ),
            "linear_attn_fused_rmsnorm_in_proj_min_gain": float(
                _QWEN35_FUSED_RMSNORM_IN_PROJ_MIN_GAIN
            ),
            "linear_attn_fused_rmsnorm_in_proj_used_layers": int(linear_fused_in_used),
            "linear_attn_fused_rmsnorm_in_proj_disabled_layers": int(linear_fused_in_disabled),
            "linear_attn_fused_rmsnorm_in_proj_total_hits": int(linear_fused_in_hits),
            "linear_attn_core_fp16_output": bool(_QWEN35_LINEAR_CORE_FP16_OUT),
            "linear_attn_reuse_decode_buffers": bool(_QWEN35_REUSE_LINEAR_DECODE_BUFFERS),
            "linear_attn_fused_norm_out_available": bool(HAS_RMSNORM_GATED_LINEAR),
            "linear_attn_fused_norm_out_enabled": bool(_QWEN35_FUSED_NORM_OUT),
            "linear_attn_fused_norm_out_max_hidden": int(_QWEN35_FUSED_NORM_OUT_MAX_HIDDEN),
            "linear_attn_fused_norm_out_min_gain": float(_QWEN35_FUSED_NORM_OUT_MIN_GAIN),
            "linear_attn_fused_norm_out_block_n": int(norm_out_cfg.get("block_n", 0) or 0),
            "linear_attn_fused_norm_out_used_layers": int(linear_norm_out_used),
            "linear_attn_fused_norm_out_disabled_layers": int(linear_norm_out_disabled),
            "linear_attn_fused_norm_out_total_hits": int(linear_norm_out_hits),
            "linear_attn_fast_in_proj_total_hits": int(linear_fast_in_hits),
            "linear_attn_fast_out_proj_total_hits": int(linear_fast_out_hits),
            "fast_gemv": bool(_USE_FAST_GEMV),
            "fast_gemv_ops": sorted(_FAST_GEMV_ENABLED_OPS),
            "fast_gemv_gate_up_autopick": bool(_GATE_UP_MODE_AUTOPICK),
            "fast_gemv_cached_modes": fast_gemv_cached_modes,
            "fast_gemv_splitk_cached_ops": fast_gemv_splitk_cached_ops,
            "fast_gemv_splitk_total_hits": int(fast_gemv_splitk_hits),
            "rmsnorm_no_weight_triton_hits": int(
                rmsnorm_no_weight_triton_hits
            ),
            "deepfusion_mlp": bool(_USE_DEEPFUSION_MLP),
            "deepfusion_mlp_min_gain": float(_DEEPFUSION_MLP_MIN_GAIN),
            "deepfusion_mlp_checked_layers": int(deepfusion_checked),
            "deepfusion_mlp_used_layers": int(deepfusion_used),
            "deepfusion_mlp_disabled_layers": int(deepfusion_disabled),
            "deepfusion_mlp_total_hits": int(deepfusion_hits),
            "qwen3_moe_grouped_decode_enabled": bool(
                _USE_QWEN3_MOE_GROUPED_DECODE
            ),
            "qwen3_moe_grouped_decode_total_hits": int(
                qwen3_moe_grouped_decode_hits
            ),
            "qwen3_moe_grouped_decode_disabled_layers": int(
                qwen3_moe_grouped_decode_disabled_layers
            ),
            "qwen3_moe_grouped_decode_first_failure": str(
                qwen3_moe_grouped_decode_first_failure
            ),
            "gemma4_batch_moe_decode_policy": {
                "enabled_layers": int(gemma4_batch_decode_policy_layers),
                "compact_path_layers": int(
                    gemma4_batch_decode_use_compact_layers
                ),
                "max_assignments": 128,
                "compact_min_rows": 9,
                "compact_max_rows": 16,
                "deterministic_partial_reduce": True,
            },
            "gemma4_batch_moe_decode_deterministic_reduce_layers": int(
                gemma4_batch_decode_deterministic_reduce_layers
            ),
            "gemma4_batch_moe_decode_expert_grid_pack_layers": int(
                gemma4_batch_decode_expert_grid_pack_layers
            ),
            "gemma4_batch_moe_decode_coalesced_weight_layers": int(
                gemma4_batch_decode_coalesced_weight_layers
            ),
            "gemma4_batch_moe_decode_fused_post_moe_layers": int(
                gemma4_batch_decode_fused_post_moe_layers
            ),
            "gemma4_batch_moe_decode_active_list_layers": int(
                gemma4_batch_decode_active_list_layers
            ),
            "gemma4_batch_moe_decode_active_list_early_exit_layers": int(
                gemma4_batch_decode_active_list_early_exit_layers
            ),
            "gemma4_batch_moe_decode_compact_hits": int(
                gemma4_batch_decode_compact_hits
            ),
            "gemma4_batch_moe_decode_compact_disabled_layers": int(
                gemma4_batch_decode_compact_disabled_layers
            ),
            "gemma4_batch_moe_decode_compact_first_failure": str(
                gemma4_batch_decode_compact_first_failure
            ),
            "gemma4_batch_moe_decode_last_paths": dict(
                gemma4_batch_decode_last_paths
            ),
            "qwen3_moe_fused_router": bool(
                qwen3_moe_runtime_cfg.get("fused_router", False)
            ),
            "qwen3_moe_fused_router_max_rows": int(
                qwen3_moe_runtime_cfg.get("fused_router_max_rows", 0) or 0
            ),
            "qwen3_moe_router_k_splits": int(
                qwen3_moe_runtime_cfg.get("router_k_splits", 1) or 1
            ),
            "qwen3_moe_grouped_decode_token_accum": bool(
                qwen3_moe_runtime_cfg.get("token_accum", False)
            ),
            "qwen3_moe_grouped_decode_fused_gate": bool(
                qwen3_moe_runtime_cfg.get("grouped_fused_gate", False)
            ),
            "qwen3_moe_grouped_decode_dot": bool(
                qwen3_moe_runtime_cfg.get("grouped_dot", False)
            ),
            "qwen3_moe_grouped_decode_dot_requested": bool(
                qwen3_moe_runtime_cfg.get("grouped_dot_requested", False)
            ),
            "qwen3_moe_grouped_decode_dot_graph_disabled": bool(
                qwen3_moe_runtime_cfg.get("grouped_dot_graph_disabled", False)
            ),
            "qwen3_moe_expert_grouped_decode": bool(
                qwen3_moe_runtime_cfg.get("expert_grouped_decode", False)
            ),
            "qwen3_moe_shared_route_decode": bool(
                qwen3_moe_runtime_cfg.get("shared_route_decode", False)
            ),
            "qwen3_moe_shared_route_batch_max_rows": int(
                qwen3_moe_runtime_cfg.get("shared_route_batch_max_rows", 1) or 1
            ),
            "qwen3_moe_shared_route_assume_identical": bool(
                qwen3_moe_runtime_cfg.get("shared_route_assume_identical", False)
            ),
            "qwen3_moe_shared_route_gate_k_splits": int(
                qwen3_moe_runtime_cfg.get("shared_route_gate_k_splits", 1) or 1
            ),
            "qwen3_moe_shared_route_decode_disabled_layers": int(
                qwen3_moe_shared_route_disabled
            ),
            "qwen3_moe_shared_route_decode_last_token_accum_layers": int(
                qwen3_moe_shared_route_last_token_accum_layers
            ),
            "qwen3_moe_shared_route_decode_last_split_gate_layers": int(
                qwen3_moe_shared_route_last_split_gate_layers
            ),
            "qwen3_moe_shared_route_decode_last_partial_reduce_layers": int(
                qwen3_moe_shared_route_last_partial_reduce_layers
            ),
            "qwen3_moe_shared_route_decode_residual_fused_layers": int(
                qwen3_moe_shared_route_residual_fused_layers
            ),
            "qwen3_moe_shared_route_decode_first_failure": (
                qwen3_moe_shared_route_first_failure
            ),
            "qwen3_moe_route_matrix_decode": bool(
                qwen3_moe_runtime_cfg.get("route_matrix_decode", False)
            ),
            "qwen3_moe_route_matrix_decode_max_rows": int(
                qwen3_moe_runtime_cfg.get("route_matrix_max_rows", 1) or 1
            ),
            "qwen3_moe_route_matrix_decode_disabled_layers": int(
                qwen3_moe_route_matrix_disabled
            ),
            "qwen3_moe_route_matrix_decode_first_failure": (
                qwen3_moe_route_matrix_first_failure
            ),
            "qwen3_moe_expert_grouped_general_decode": bool(
                qwen3_moe_runtime_cfg.get("expert_grouped_general_decode", False)
            ),
            "qwen3_moe_expert_grouped_general_decode_disabled_layers": int(
                qwen3_moe_expert_grouped_general_disabled
            ),
            "qwen3_moe_expert_grouped_general_decode_first_failure": (
                qwen3_moe_expert_grouped_general_first_failure
            ),
            "qwen3_moe_expert_grouped_dense_decode": bool(
                qwen3_moe_runtime_cfg.get("expert_grouped_dense_decode", False)
            ),
            "qwen3_moe_expert_grouped_compact_decode": bool(
                qwen3_moe_runtime_cfg.get("expert_grouped_compact_decode", False)
            ),
            "qwen3_moe_expert_grouped_compact_fused_pack": bool(
                qwen3_moe_runtime_cfg.get("expert_grouped_compact_fused_pack", False)
            ),
            "qwen3_moe_expert_grouped_compact_partial_reduce": bool(
                qwen3_moe_runtime_cfg.get("expert_grouped_compact_partial_reduce", False)
            ),
            "qwen3_moe_expert_grouped_compact_active_list": bool(
                qwen3_moe_runtime_cfg.get("expert_grouped_compact_active_list", False)
            ),
            "qwen3_moe_expert_grouped_compact_active_list_early_exit": bool(
                qwen3_moe_runtime_cfg.get(
                    "expert_grouped_compact_active_list_early_exit",
                    False,
                )
            ),
            "qwen3_moe_expert_grouped_compact_expert_grid_pack": bool(
                qwen3_moe_runtime_cfg.get(
                    "expert_grouped_compact_expert_grid_pack",
                    False,
                )
            ),
            "qwen3_moe_expert_grouped_compact_coalesced_weights": bool(
                qwen3_moe_runtime_cfg.get(
                    "expert_grouped_compact_coalesced_weights",
                    False,
                )
            ),
            "qwen3_moe_expert_grouped_compact_token_accum": bool(
                qwen3_moe_runtime_cfg.get("expert_grouped_compact_token_accum", False)
            ),
            "qwen3_moe_expert_grouped_compact_gate_block_n": int(
                qwen3_moe_runtime_cfg.get("expert_grouped_compact_gate_block_n", 0) or 0
            ),
            "qwen3_moe_expert_grouped_compact_down_block_n": int(
                qwen3_moe_runtime_cfg.get("expert_grouped_compact_down_block_n", 0) or 0
            ),
            "qwen3_moe_expert_grouped_compact_num_warps": int(
                qwen3_moe_runtime_cfg.get("expert_grouped_compact_num_warps", 0)
                or 0
            ),
            "qwen3_moe_expert_grouped_compact_num_stages": int(
                qwen3_moe_runtime_cfg.get("expert_grouped_compact_num_stages", 0)
                or 0
            ),
            "qwen3_moe_expert_grouped_compact_gate_num_stages": int(
                qwen3_moe_runtime_cfg.get(
                    "expert_grouped_compact_gate_num_stages",
                    0,
                )
                or 0
            ),
            "qwen3_moe_expert_grouped_compact_down_num_stages": int(
                qwen3_moe_runtime_cfg.get(
                    "expert_grouped_compact_down_num_stages",
                    0,
                )
                or 0
            ),
            "qwen3_moe_expert_grouped_compact_experts_per_program": int(
                qwen3_moe_runtime_cfg.get(
                    "expert_grouped_compact_experts_per_program",
                    0,
                )
                or 0
            ),
            "qwen3_moe_expert_grouped_compact_paired_gate_up_dot": bool(
                qwen3_moe_runtime_cfg.get(
                    "expert_grouped_compact_paired_gate_up_dot",
                    False,
                )
            ),
            "qwen3_moe_expert_grouped_compact_split_gate_up": bool(
                qwen3_moe_runtime_cfg.get(
                    "expert_grouped_compact_split_gate_up",
                    False,
                )
            ),
            "qwen3_moe_expert_grouped_compact_empty_expert_early_exit": bool(
                qwen3_moe_runtime_cfg.get(
                    "expert_grouped_compact_empty_expert_early_exit",
                    False,
                )
            ),
            "qwen3_moe_expert_grouped_compact_l2_grouped_grid": bool(
                qwen3_moe_runtime_cfg.get(
                    "expert_grouped_compact_l2_grouped_grid",
                    False,
                )
            ),
            "qwen3_moe_expert_grouped_compact_l2_group_size": int(
                qwen3_moe_runtime_cfg.get(
                    "expert_grouped_compact_l2_group_size",
                    0,
                )
                or 0
            ),
            "qwen3_moe_expert_grouped_compact_direct_out": bool(
                qwen3_moe_runtime_cfg.get("expert_grouped_compact_direct_out", False)
            ),
            "qwen3_moe_expert_grouped_compact_decode_disabled_layers": int(
                qwen3_moe_expert_grouped_compact_disabled
            ),
            "qwen3_moe_expert_grouped_compact_decode_first_failure": (
                qwen3_moe_expert_grouped_compact_first_failure
            ),
            "qwen3_moe_expert_grouped_decode_disabled_layers": int(
                qwen3_moe_expert_grouped_disabled
            ),
            "qwen3_moe_expert_grouped_decode_first_failure": (
                qwen3_moe_expert_grouped_first_failure
            ),
            "qwen3_moe_expert_grouped_decode_min_rows": int(
                qwen3_moe_runtime_cfg.get("expert_grouped_min_rows", 0) or 0
            ),
            "qwen3_moe_expert_grouped_decode_max_rows": int(
                qwen3_moe_runtime_cfg.get("expert_grouped_max_rows", 0) or 0
            ),
            "qwen3_moe_expert_grouped_decode_block_m": int(
                qwen3_moe_runtime_cfg.get("expert_grouped_block_m", 0) or 0
            ),
            "qwen3_moe_grouped_decode_int8": bool(
                qwen3_moe_runtime_cfg.get("int8_decode", False)
            ),
            "qwen3_moe_expert_int8_layers": int(qwen3_moe_int8_expert_layers),
            "qwen3_moe_int8_dequant_prefill_enabled": bool(
                _USE_QWEN3_MOE_INT8_DEQUANT_PREFILL
            ),
            "qwen3_moe_int8_dequant_prefill_min_assignments": int(
                _QWEN3_MOE_INT8_DEQUANT_PREFILL_MIN_ASSIGNMENTS
            ),
            "qwen3_moe_int8_dequant_prefill_total_hits": int(
                qwen3_moe_int8_dequant_prefill_hits
            ),
            "qwen3_moe_int8_dequant_prefill_disabled_layers": int(
                qwen3_moe_int8_dequant_prefill_disabled
            ),
            "qwen3_moe_int8_dequant_prefill_first_failure": (
                qwen3_moe_int8_dequant_prefill_first_failure
            ),
            "qwen3_moe_grouped_decode_dot_disabled_layers": int(
                qwen3_moe_grouped_dot_disabled
            ),
            "qwen3_moe_grouped_decode_block_n": int(
                qwen3_moe_runtime_cfg.get("block_n", 0) or 0
            ),
            "qwen3_moe_grouped_decode_block_k": int(
                qwen3_moe_runtime_cfg.get("block_k", 0) or 0
            ),
            "qwen3_moe_grouped_decode_num_warps": int(
                qwen3_moe_runtime_cfg.get("num_warps", 0) or 0
            ),
            "qwen3_moe_segmented_prefill_enabled": bool(_USE_QWEN3_MOE_SEGMENTED_PREFILL),
            "gemma4_long_dominant_expert_prefill_enabled": bool(
                _GEMMA4_A4B_LONG_DOMINANT_EXPERT_PREFILL
            ),
            "gemma4_long_dominant_expert_prefill_rows": int(
                _GEMMA4_A4B_SEGMENTED_PREFILL_LONG_MAX_ROWS
            ),
            "gemma4_long_dominant_expert_prefill_down_output_dtype": "fp32",
            "gemma4_long_dominant_expert_prefill_route_pack": "atomic_split",
            "gemma4_long_dominant_expert_prefill_deterministic_reduce": True,
            "gemma4_long_dominant_expert_prefill_route_pack_block": 256,
            "gemma4_long_dominant_expert_prefill_activation_block": 512,
            "gemma4_long_dominant_expert_prefill_reduce_block_n": 256,
            "gemma4_long_dominant_expert_prefill_reduce_num_warps": 4,
            "gemma4_long_dominant_expert_prefill_align_m": 16,
            "gemma4_long_dominant_expert_prefill_minimum_skew": float(
                _GEMMA4_A4B_LONG_DOMINANT_EXPERT_MIN_SKEW
            ),
            "gemma4_long_dominant_expert_prefill_max_light_padding_ratio": float(
                _GEMMA4_A4B_LONG_DOMINANT_EXPERT_MAX_LIGHT_PADDING_RATIO
            ),
            "gemma4_long_dominant_expert_prefill_hits": int(
                gemma4_long_dominant_expert_prefill_hits
            ),
            "gemma4_long_dominant_expert_prefill_assignments": int(
                gemma4_long_dominant_expert_prefill_assignments
            ),
            "gemma4_long_dominant_expert_prefill_last_active_layers": int(
                gemma4_long_dominant_expert_prefill_last_active_layers
            ),
            "gemma4_long_dominant_expert_prefill_guard_misses": int(
                gemma4_long_dominant_expert_prefill_guard_misses
            ),
            "gemma4_long_dominant_expert_prefill_guard_miss_layers": int(
                gemma4_long_dominant_expert_prefill_guard_miss_layers
            ),
            "gemma4_long_dominant_expert_prefill_guard_rejections": list(
                gemma4_long_dominant_expert_prefill_guard_rejections
            ),
            "gemma4_long_dominant_expert_prefill_disabled_layers": int(
                gemma4_long_dominant_expert_prefill_disabled_layers
            ),
            "gemma4_long_dominant_expert_prefill_first_failure": (
                gemma4_long_dominant_expert_prefill_first_failure
            ),
            "gemma4_long_dominant_expert_prefill_failures": list(
                gemma4_long_dominant_expert_prefill_failures
            ),
            "gemma4_long_dominant_expert_prefill_profiles": list(
                gemma4_long_dominant_expert_prefill_profiles
            ),
            "gemma4_long_padded_bmm_prefill_enabled": bool(
                _GEMMA4_A4B_LONG_PADDED_BMM_PREFILL
            ),
            "gemma4_long_padded_bmm_prefill_rows": int(
                _GEMMA4_A4B_SEGMENTED_PREFILL_LONG_ROWS
            ),
            "gemma4_long_padded_bmm_prefill_down_output_dtype": "fp32",
            "gemma4_long_padded_bmm_prefill_route_pack": (
                "argsort"
                if torch.are_deterministic_algorithms_enabled()
                else "atomic"
            ),
            "gemma4_long_padded_bmm_prefill_route_pack_block": 256,
            "gemma4_long_padded_bmm_prefill_max_padding_ratio": float(
                _GEMMA4_A4B_LONG_PADDED_BMM_MAX_PADDING_RATIO
            ),
            "gemma4_long_padded_bmm_prefill_fused_activation": True,
            "gemma4_long_padded_bmm_prefill_activation_block": 512,
            "gemma4_long_padded_bmm_prefill_reduce_block_n": 256,
            "gemma4_long_padded_bmm_prefill_reduce_num_warps": 4,
            "gemma4_long_padded_bmm_prefill_align_m": 16,
            "gemma4_long_padded_bmm_prefill_hits": int(
                gemma4_long_padded_bmm_prefill_hits
            ),
            "gemma4_long_padded_bmm_prefill_assignments": int(
                gemma4_long_padded_bmm_prefill_assignments
            ),
            "gemma4_long_padded_bmm_prefill_last_active_layers": int(
                gemma4_long_padded_bmm_prefill_last_active_layers
            ),
            "gemma4_long_padded_bmm_prefill_disabled_layers": int(
                gemma4_long_padded_bmm_prefill_disabled_layers
            ),
            "gemma4_long_padded_bmm_prefill_first_failure": (
                gemma4_long_padded_bmm_prefill_first_failure
            ),
            "gemma4_long_padded_bmm_prefill_failures": list(
                gemma4_long_padded_bmm_prefill_failures
            ),
            "gemma4_grouped_mm_prefill_selected_layers": int(
                gemma4_grouped_mm_prefill_selected_layers
            ),
            "gemma4_grouped_mm_prefill_last_active_layers": int(
                gemma4_grouped_mm_prefill_last_active_layers
            ),
            "gemma4_grouped_mm_prefill_hits": int(
                gemma4_grouped_mm_prefill_hits
            ),
            "gemma4_grouped_mm_prefill_disabled_layers": int(
                gemma4_grouped_mm_prefill_disabled_layers
            ),
            "gemma4_grouped_mm_prefill_first_failure": (
                gemma4_grouped_mm_prefill_first_failure
            ),
            "gemma4_a4b_segmented_prefill_layers": int(
                gemma4_a4b_segmented_prefill_layers
            ),
            "gemma4_a4b_segmented_prefill_effective": bool(
                gemma4_a4b_segmented_prefill_layers > 0
            ),
            "gemma4_a4b_segmented_prefill_config": dict(
                _GEMMA4_A4B_SEGMENTED_PREFILL_OPTIONS,
                large_rows_min=_GEMMA4_A4B_SEGMENTED_PREFILL_LARGE_ROWS_MIN,
                large=dict(_GEMMA4_A4B_SEGMENTED_PREFILL_LARGE_OPTIONS),
                long_rows=_GEMMA4_A4B_SEGMENTED_PREFILL_LONG_ROWS,
                long_rows_max=_GEMMA4_A4B_SEGMENTED_PREFILL_LONG_MAX_ROWS,
                long=dict(_GEMMA4_A4B_SEGMENTED_PREFILL_LONG_OPTIONS),
                short_rows_max=_GEMMA4_A4B_SEGMENTED_PREFILL_SHORT_ROWS_MAX,
                short=dict(_GEMMA4_A4B_SEGMENTED_PREFILL_SHORT_OPTIONS),
            ),
            "qwen3_moe_segmented_prefill_dense_grid": bool(
                qwen3_moe_runtime_cfg.get("segmented_prefill_dense_grid", False)
            ),
            "qwen3_moe_segmented_prefill_fused_gate": bool(
                qwen3_moe_runtime_cfg.get("segmented_prefill_fused_gate", False)
            ),
            "qwen3_moe_segmented_prefill_async_tiles": bool(
                qwen3_moe_runtime_cfg.get("segmented_prefill_async_tiles", False)
            ),
            "qwen3_moe_segmented_prefill_async_tiles_max_assignments": int(
                qwen3_moe_runtime_cfg.get(
                    "segmented_prefill_async_tiles_max_assignments",
                    0,
                )
                or 0
            ),
            "qwen3_moe_segmented_prefill_partial_reduce": bool(
                qwen3_moe_runtime_cfg.get("segmented_prefill_partial_reduce", False)
            ),
            "qwen3_moe_segmented_prefill_sorted_partial": bool(
                qwen3_moe_runtime_cfg.get("segmented_prefill_sorted_partial", False)
                or _GEMMA4_A4B_SEGMENTED_PREFILL_LONG_OPTIONS.get(
                    "sorted_partial",
                    False,
                )
            ),
            "qwen3_moe_segmented_prefill_partial_reduce_max_assignments": int(
                qwen3_moe_runtime_cfg.get(
                    "segmented_prefill_partial_reduce_max_assignments",
                    0,
                )
                or 0
            ),
            "qwen3_moe_segmented_prefill_partial_cache_max_assignments": int(
                qwen3_moe_runtime_cfg.get(
                    "segmented_prefill_partial_cache_max_assignments",
                    0,
                )
                or 0
            ),
            "qwen3_moe_segmented_prefill_route_scatter": bool(
                qwen3_moe_runtime_cfg.get("segmented_prefill_route_scatter", False)
            ),
            "qwen3_moe_segmented_prefill_fixed_route_pack": bool(
                qwen3_moe_runtime_cfg.get(
                    "segmented_prefill_fixed_route_pack",
                    False,
                )
                or (
                    gemma4_a4b_segmented_prefill_layers > 0
                    and _GEMMA4_A4B_SEGMENTED_PREFILL_SHORT_OPTIONS.get(
                        "fixed_route_pack",
                        False,
                    )
                )
            ),
            "qwen3_moe_segmented_prefill_compact_route_pack": bool(
                qwen3_moe_runtime_cfg.get(
                    "segmented_prefill_compact_route_pack",
                    False,
                )
                or (
                    gemma4_a4b_segmented_prefill_layers > 0
                    and _GEMMA4_A4B_SEGMENTED_PREFILL_OPTIONS.get(
                        "compact_route_pack",
                        False,
                    )
                )
            ),
            "qwen3_moe_segmented_prefill_route_block": int(
                qwen3_moe_runtime_cfg.get("segmented_prefill_route_block", 0) or 0
            ),
            "qwen3_moe_segmented_prefill_min_assignments": int(
                _QWEN3_MOE_SEGMENTED_PREFILL_MIN_ASSIGNMENTS
            ),
            "qwen3_moe_segmented_prefill_total_hits": int(qwen3_moe_segmented_prefill_hits),
            "qwen3_moe_segmented_prefill_residual_fused_hits": int(
                qwen3_moe_segmented_prefill_residual_fused_hits
            ),
            "qwen3_moe_segmented_prefill_residual_fused_layers": int(
                qwen3_moe_segmented_prefill_residual_fused_layers
            ),
            "qwen3_moe_segmented_prefill_disabled_layers": int(
                qwen3_moe_segmented_prefill_disabled
            ),
            "qwen3_moe_segmented_prefill_assignments": int(
                qwen3_moe_segmented_prefill_assignments
            ),
            "qwen3_moe_segmented_prefill_tiles": int(qwen3_moe_segmented_prefill_tiles),
            "qwen3_moe_segmented_prefill_async_tile_hits": int(
                qwen3_moe_segmented_async_tile_hits
            ),
            "qwen3_moe_segmented_prefill_max_tiles": int(
                qwen3_moe_segmented_max_tiles
            ),
            "qwen3_moe_segmented_prefill_partial_reduce_hits": int(
                qwen3_moe_segmented_partial_reduce_hits
            ),
            "qwen3_moe_segmented_prefill_partial_reduce_layers": int(
                qwen3_moe_segmented_partial_reduce_layers
            ),
            "qwen3_moe_segmented_prefill_sorted_partial_hits": int(
                qwen3_moe_segmented_sorted_partial_hits
            ),
            "qwen3_moe_segmented_prefill_sorted_partial_layers": int(
                qwen3_moe_segmented_sorted_partial_layers
            ),
            "qwen3_moe_segmented_prefill_deterministic_reduce_layers": int(
                qwen3_moe_segmented_deterministic_reduce_layers
            ),
            "qwen3_moe_segmented_prefill_atomic_reduce_layers": int(
                qwen3_moe_segmented_atomic_reduce_layers
            ),
            "qwen3_moe_segmented_prefill_single_accumulator_hits": int(
                qwen3_moe_segmented_single_accumulator_hits
            ),
            "qwen3_moe_segmented_prefill_single_accumulator_layers": int(
                qwen3_moe_segmented_single_accumulator_layers
            ),
            "qwen3_moe_segmented_prefill_fixed_route_pack_hits": int(
                qwen3_moe_segmented_fixed_route_pack_hits
            ),
            "qwen3_moe_segmented_prefill_compact_route_pack_hits": int(
                qwen3_moe_segmented_compact_route_pack_hits
            ),
            "qwen3_moe_segmented_prefill_compact_route_pack_layers": int(
                qwen3_moe_segmented_compact_route_pack_layers
            ),
            "qwen3_moe_segmented_prefill_compact_route_pack_single_scan_layers": int(
                qwen3_moe_segmented_compact_route_pack_single_scan_layers
            ),
            "qwen3_moe_segmented_prefill_graph_route_pack_layers": int(
                qwen3_moe_segmented_graph_route_pack_layers
            ),
            "qwen3_moe_segmented_prefill_route_scatter_hits": int(
                qwen3_moe_segmented_route_scatter_hits
            ),
            "qwen3_moe_segmented_prefill_route_argsort_hits": int(
                qwen3_moe_segmented_route_argsort_hits
            ),
            "qwen3_moe_segmented_prefill_route_scatter_first_failure": (
                qwen3_moe_segmented_route_scatter_first_failure
            ),
            "qwen3_moe_segmented_prefill_first_failure": (
                qwen3_moe_segmented_prefill_first_failure
            ),
            "gemma4_flat_fused_gateup_hits": int(
                getattr(self, "_gemma4_flat_fused_gateup_hits", 0)
            ),
            "gemma4_policy_bf16_fused_gateup_rows": list(
                getattr(self, "_gemma4_flat_policy_fused_gateup_rows", ())
            ),
            "gemma4_flat_fused_qkv_layers": int(
                sum(
                    getattr(layer, "qkv_wt", None) is not None
                    for layer in (getattr(self, "_flat_layer_weights", None) or ())
                )
            ),
            "gemma4_flat_fused_gateup_runtime_disabled": bool(
                getattr(self, "_gemma4_flat_fused_gateup_runtime_disabled", False)
            ),
            "gemma4_ple_conditioned_gelu_decode_enabled": bool(
                getattr(
                    self,
                    "_gemma4_flat_ple_conditioned_gelu_enabled",
                    False,
                )
            ),
            "gemma4_ple_conditioned_gelu_decode_hits": int(
                getattr(self, "_gemma4_flat_ple_conditioned_gelu_hits", 0)
            ),
            "gemma4_ple_conditioned_gelu_block_size": int(
                getattr(self, "_gemma4_flat_ple_conditioned_gelu_block_size", 0)
            ),
            "gemma4_ple_conditioned_gelu_runtime_disabled": bool(
                getattr(
                    self,
                    "_gemma4_flat_ple_conditioned_gelu_runtime_disabled",
                    False,
                )
            ),
            "gemma4_ple_conditioned_gelu_first_failure": str(
                getattr(
                    self,
                    "_gemma4_flat_ple_conditioned_gelu_first_failure",
                    "",
                )
            ),
            "gemma4_e2b_b8_gated_activation_enabled": bool(
                getattr(self, "_gemma4_flat_b8_gated_activation_enabled", False)
            ),
            "gemma4_e2b_b8_gated_activation_block_size": int(
                getattr(self, "_gemma4_flat_b8_gated_activation_block_size", 0)
            ),
            "gemma4_e2b_b8_gated_activation_hits": int(
                getattr(self, "_gemma4_flat_b8_gated_activation_hits", 0)
            ),
            "gemma4_e2b_b8_gated_activation_runtime_disabled": bool(
                getattr(
                    self,
                    "_gemma4_flat_b8_gated_activation_runtime_disabled",
                    False,
                )
            ),
            "gemma4_e2b_b8_gated_activation_failure": str(
                getattr(self, "_gemma4_flat_b8_gated_activation_failure", "")
            ),
            "gemma4_e2b_b8_tensorcore_down_enabled": bool(
                getattr(self, "_gemma4_flat_b8_tensorcore_down_enabled", False)
            ),
            "gemma4_e2b_b8_tensorcore_down_config": list(
                getattr(self, "_gemma4_flat_b8_tensorcore_down_config", ())
            ),
            "gemma4_e2b_b8_tensorcore_down_hits": int(
                getattr(self, "_gemma4_flat_b8_tensorcore_down_hits", 0)
            ),
            "gemma4_e2b_b8_tensorcore_down_runtime_disabled": bool(
                getattr(
                    self,
                    "_gemma4_flat_b8_tensorcore_down_runtime_disabled",
                    False,
                )
            ),
            "gemma4_e2b_b8_tensorcore_down_failure": str(
                getattr(self, "_gemma4_flat_b8_tensorcore_down_failure", "")
            ),
            "gemma4_flat_deepfusion_hits": int(
                getattr(self, "_gemma4_flat_deepfusion_hits", 0)
            ),
            "gemma4_policy_bf16_deepfusion_rows": list(
                getattr(self, "_gemma4_flat_policy_deepfusion_rows", ())
            ),
            "gemma4_policy_bf16_cublas_gateup_rows": list(
                getattr(self, "_gemma4_flat_policy_cublas_gateup_rows", ())
            ),
            "gemma4_policy_bf16_cublas_down_rows": list(
                getattr(self, "_gemma4_flat_policy_cublas_down_rows", ())
            ),
            "gemma4_cublaslt_gateup_decode_enabled": bool(
                getattr(self, "_gemma4_flat_cublaslt_gateup_enabled", False)
            ),
            "gemma4_cublaslt_gateup_decode_hits": int(
                getattr(self, "_gemma4_flat_cublaslt_gateup_hits", 0)
            ),
            "gemma4_cublaslt_gateup_algorithms": dict(
                getattr(self, "_gemma4_flat_cublaslt_gateup_algorithms", {})
            ),
            "gemma4_cublaslt_gateup_runtime_disabled": bool(
                getattr(
                    self,
                    "_gemma4_flat_cublaslt_gateup_runtime_disabled",
                    False,
                )
            ),
            "gemma4_cublaslt_gateup_failure": str(
                getattr(self, "_gemma4_flat_cublaslt_gateup_failure", "")
            ),
            "gemma4_dense_post_norm_chain_decode_enabled": bool(
                getattr(
                    self,
                    "_gemma4_flat_dense_post_norm_chain_enabled",
                    False,
                )
            ),
            "gemma4_dense_post_norm_chain_decode_hits": int(
                getattr(self, "_gemma4_flat_dense_post_norm_chain_hits", 0)
            ),
            "gemma4_dense_next_attn_norm_decode_hits": int(
                getattr(self, "_gemma4_flat_dense_next_attn_norm_hits", 0)
            ),
            "gemma4_dense_attn_mlp_bridge_decode_enabled": bool(
                getattr(
                    self,
                    "_gemma4_flat_dense_attn_mlp_bridge_enabled",
                    False,
                )
            ),
            "gemma4_dense_attn_mlp_bridge_decode_hits": int(
                getattr(self, "_gemma4_flat_dense_attn_mlp_bridge_hits", 0)
            ),
            "gemma4_dense_attn_mlp_bridge_runtime_disabled": bool(
                getattr(
                    self,
                    "_gemma4_flat_dense_attn_mlp_bridge_runtime_disabled",
                    False,
                )
            ),
            "gemma4_dense_attn_mlp_bridge_failure": str(
                getattr(
                    self,
                    "_gemma4_flat_dense_attn_mlp_bridge_failure",
                    "",
                )
            ),
            "gemma4_e2b_b1_large_gateup_experiment": bool(
                getattr(
                    self,
                    "_gemma4_flat_b1_large_gateup_experiment",
                    False,
                )
            ),
            "gemma4_e2b_b1_large_gateup_hits": int(
                getattr(self, "_gemma4_flat_b1_large_gateup_hits", 0)
            ),
            "gemma4_e2b_b1_large_down_experiment": bool(
                getattr(
                    self,
                    "_gemma4_flat_b1_large_down_experiment",
                    False,
                )
            ),
            "gemma4_e2b_b1_large_down_hits": int(
                getattr(self, "_gemma4_flat_b1_large_down_hits", 0)
            ),
            "gemma4_b1_mlp_gemv_configs": dict(
                getattr(self, "_gemma4_b1_mlp_gemv_configs", {})
            ),
            "gemma4_b1_mlp_gemv_dispatch": getattr(self, "_gemma4_b1_mlp_gemv_dispatch", "legacy"),
            "gemma4_b1_mlp_prepared_gateup_hits": int(
                getattr(self, "_gemma4_b1_mlp_prepared_hits", {}).get("gateup", 0)
            ),
            "gemma4_b1_mlp_prepared_down_hits": int(
                getattr(self, "_gemma4_b1_mlp_prepared_hits", {}).get("down", 0)
            ),
            "gemma4_b1_mlp_gemv_failures": dict(
                getattr(self, "_gemma4_b1_mlp_gemv_failures", {})
            ),
            "gemma4_b1_mlp_gemv_gateup_hits": int(
                getattr(self, "_gemma4_b1_mlp_gemv_hits", {}).get("gateup", 0)
            ),
            "gemma4_b1_mlp_gemv_down_hits": int(
                getattr(self, "_gemma4_b1_mlp_gemv_hits", {}).get("down", 0)
            ),
            "gemma4_parallel_moe_decode_enabled": bool(
                getattr(self, "_gemma4_flat_parallel_moe_enabled", False)
            ),
            "gemma4_parallel_moe_decode_hits": int(
                getattr(self, "_gemma4_flat_parallel_moe_hits", 0)
            ),
            "gemma4_parallel_moe_decode_policy": dict(
                getattr(self, "_gemma4_flat_parallel_moe_policy", {})
            ),
            "gemma4_fused_attn_moe_bridge_decode_enabled": bool(
                getattr(
                    self,
                    "_gemma4_flat_fused_attn_moe_bridge_enabled",
                    False,
                )
            ),
            "gemma4_fused_attn_moe_bridge_decode_hits": int(
                getattr(self, "_gemma4_flat_fused_attn_moe_bridge_hits", 0)
            ),
            "gemma4_fused_attn_moe_router_bridge_decode_enabled": bool(
                getattr(
                    self,
                    "_gemma4_flat_fused_attn_moe_router_bridge_enabled",
                    False,
                )
            ),
            "gemma4_fused_attn_moe_router_bridge_decode_hits": int(
                getattr(
                    self,
                    "_gemma4_flat_fused_attn_moe_router_bridge_hits",
                    0,
                )
            ),
            "gemma4_fused_attn_moe_router_single_kernel_decode_enabled": bool(
                getattr(
                    self,
                    "_gemma4_flat_fused_attn_moe_router_single_kernel_enabled",
                    False,
                )
            ),
            "gemma4_fused_attn_moe_router_single_kernel_decode_hits": int(
                getattr(
                    self,
                    "_gemma4_flat_fused_attn_moe_router_single_kernel_hits",
                    0,
                )
            ),
            "gemma4_fused_router_compact_pack_decode_enabled": bool(
                getattr(
                    self,
                    "_gemma4_flat_fused_router_compact_pack_enabled",
                    False,
                )
            ),
            "gemma4_fused_router_compact_pack_decode_hits": int(
                getattr(
                    self,
                    "_gemma4_flat_fused_router_compact_pack_hits",
                    0,
                )
            ),
            "gemma4_fused_post_moe_norm_residual_decode_enabled": bool(
                getattr(
                    self,
                    "_gemma4_flat_fused_post_moe_norm_residual_enabled",
                    False,
                )
            ),
            "gemma4_fused_post_moe_norm_residual_decode_hits": int(
                getattr(self, "_gemma4_flat_fused_post_moe_norm_residual_hits", 0)
            ),
            "gemma4_fused_expert_reduce_post_moe_decode_enabled": bool(
                getattr(
                    self,
                    "_gemma4_flat_fused_expert_reduce_post_moe_enabled",
                    False,
                )
            ),
            "gemma4_fused_expert_reduce_post_moe_decode_hits": int(
                getattr(
                    self,
                    "_gemma4_flat_fused_expert_reduce_post_moe_hits",
                    0,
                )
            ),
            "gemma4_fused_next_attn_norm_decode_supported": bool(
                getattr(
                    self,
                    "_gemma4_flat_fused_next_attn_norm_supported",
                    False,
                )
            ),
            "gemma4_fused_next_attn_norm_decode_enabled": bool(
                getattr(
                    self,
                    "_gemma4_flat_fused_next_attn_norm_enabled",
                    False,
                )
            ),
            "gemma4_fused_next_attn_norm_decode_hits": int(
                getattr(self, "_gemma4_flat_fused_next_attn_norm_hits", 0)
            ),
            "gemma4_fused_layer_scalar_decode_hits": int(
                getattr(self, "_gemma4_flat_fused_layer_scalar_hits", 0)
            ),
            "gemma4_fused_router_expert_input_norm_decode_enabled": bool(
                getattr(
                    self,
                    "_gemma4_flat_fused_router_expert_input_norm_enabled",
                    False,
                )
            ),
            "gemma4_fused_router_expert_input_norm_decode_hits": int(
                getattr(self, "_gemma4_flat_fused_router_expert_input_norm_hits", 0)
            ),
            "qwen3_moe_bucketed_prefill_enabled": bool(_USE_QWEN3_MOE_BUCKETED_PREFILL),
            "qwen3_moe_bucketed_prefill_min_assignments": int(
                _QWEN3_MOE_BUCKETED_PREFILL_MIN_ASSIGNMENTS
            ),
            "qwen3_moe_bucketed_prefill_bucket_size": int(
                _QWEN3_MOE_BUCKETED_PREFILL_BUCKET_SIZE
            ),
            "qwen3_moe_bucketed_prefill_total_hits": int(qwen3_moe_bucketed_prefill_hits),
            "qwen3_moe_bucketed_prefill_disabled_layers": int(
                qwen3_moe_bucketed_prefill_disabled
            ),
            "qwen3_moe_bucketed_prefill_valid_assignments": int(
                qwen3_moe_bucketed_prefill_valid_assignments
            ),
            "qwen3_moe_bucketed_prefill_padded_assignments": int(
                qwen3_moe_bucketed_prefill_padded_assignments
            ),
            "qwen3_moe_bucketed_prefill_pad_waste": float(
                qwen3_moe_bucketed_prefill_pad_waste
            ),
            "qwen3_moe_bucketed_prefill_bucket_launches": int(
                qwen3_moe_bucketed_prefill_bucket_launches
            ),
            "qwen3_moe_bucketed_prefill_first_failure": qwen3_moe_bucketed_prefill_first_failure,
            "qwen3_moe_batched_prefill_total_hits": int(qwen3_moe_batched_prefill_hits),
            "qwen3_moe_sorted_prefill_total_hits": int(qwen3_moe_sorted_prefill_hits),
            "fused_rmsnorm_gateup_available": bool(fused_rmsnorm_linear is not None),
            "fused_rmsnorm_gateup_decode_enabled": bool(_USE_FUSED_RMSNORM_GATEUP_DECODE),
            "fused_rmsnorm_linear_two_pass": bool(
                fused_rmsnorm_linear_cfg.get("two_pass", False)
            ),
            "fused_rmsnorm_gateup_checked_layers": int(fused_gateup_checked),
            "fused_rmsnorm_gateup_used_layers": int(fused_gateup_used),
            "fused_rmsnorm_gateup_disabled_layers": int(fused_gateup_disabled),
            "fused_lm_head_argmax_available": bool(HAS_FUSED_LM_HEAD_ARGMAX),
            "fused_lm_head_argmax_decode_enabled": bool(_USE_FUSED_LM_HEAD_ARGMAX_DECODE),
            "fused_lm_head_argmax_triton_reduce": bool(
                lm_head_argmax_cfg.get("triton_reduce", False)
            ),
            "fused_lm_head_argmax_large_k_block_n": int(
                lm_head_argmax_cfg.get("large_vocab_large_k_block_n", 0) or 0
            ),
            "fused_lm_head_argmax_gemma4_e2b_l4_b8_block_n": int(
                lm_head_argmax_cfg.get("gemma4_e2b_l4_b8_block_n", 0) or 0
            ),
            "gemma4_batch_cublas_lm_head_enabled": bool(
                _GEMMA4_BATCH_CUBLAS_LM_HEAD
            ),
            "gemma4_batch_cublas_lm_head_hits": int(
                getattr(self, "_gemma4_batch_cublas_lm_head_hits", 0)
            ),
            "gemma4_batch_fused_softcap_argmax_enabled": bool(
                _GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX
            ),
            "gemma4_batch_fused_softcap_argmax_available": bool(
                HAS_FUSED_SOFTCAP_ARGMAX
            ),
            "gemma4_batch_fused_softcap_argmax_hits": int(
                getattr(self, "_gemma4_batch_fused_softcap_argmax_hits", 0)
            ),
            "gemma4_batch_fused_softcap_argmax_disabled": bool(
                getattr(
                    self,
                    "_gemma4_batch_fused_softcap_argmax_disable",
                    False,
                )
            ),
            "gemma4_batch_fused_softcap_argmax_error": str(
                getattr(self, "_gemma4_batch_fused_softcap_argmax_error", "")
            ),
            "fused_rmsnorm_lm_head_argmax_enabled": bool(_USE_FUSED_RMSNORM_LM_HEAD_ARGMAX_DECODE),
            "fused_rmsnorm_lm_head_argmax_checked": bool(
                getattr(self, "_fused_rmsnorm_lm_head_argmax_checked", False)
            ),
            "fused_rmsnorm_lm_head_argmax_use": bool(
                getattr(self, "_fused_rmsnorm_lm_head_argmax_use", False)
            ),
            "fused_rmsnorm_lm_head_argmax_hits": int(
                getattr(self, "_fused_rmsnorm_lm_head_argmax_hits", 0)
            ),
            "fused_rmsnorm_lm_head_argmax_disabled": bool(
                getattr(self, "_fused_rmsnorm_lm_head_argmax_disable", False)
            ),
            "fused_rmsnorm_lm_head_argmax_error": str(
                getattr(self, "_fused_rmsnorm_lm_head_argmax_error", "")
            ),
            "fused_rmsnorm_lm_head_argmax_skip_reason": str(
                getattr(self, "_fused_rmsnorm_lm_head_argmax_skip_reason", "")
            ),
            "fused_lm_head_argmax_checked": bool(getattr(self, "_fused_lm_head_argmax_checked", False)),
            "fused_lm_head_argmax_use": bool(getattr(self, "_fused_lm_head_argmax_use", False)),
            "fused_lm_head_argmax_disabled": bool(getattr(self, "_fused_lm_head_argmax_disable", False)),
            "fused_lm_head_argmax_error": str(getattr(self, "_fused_lm_head_argmax_error", "")),
            "fused_lm_head_argmax_skip_reason": str(
                getattr(self, "_fused_lm_head_argmax_skip_reason", "")
            ),
        }

    def get_last_decode_timing(self) -> Optional[dict]:
        """Return last decode timing summary when MEGAGEMM_DECODE_TIMING=1."""
        return self._last_decode_timing

    def get_last_prefill_timing(self) -> Optional[dict]:
        """Return last prefill timing summary when MEGAGEMM_PREFILL_TIMING=1."""
        return self._last_prefill_timing

    def get_prefill_cuda_graph_store(self, block_manager=None) -> dict:
        store = self._prefill_cuda_graph_store
        bm_id = id(block_manager) if block_manager is not None else None
        if bm_id is not None and store.get('block_manager_id') != bm_id:
            store['block_manager_id'] = bm_id
            store['buckets'] = {}
            store['warm_keys'] = set()
            store['failed_keys'] = {}
            store['skips'] = 0
            store['captures'] = 0
            store['capture_body_warmups'] = 0
            store['capture_replays'] = 0
            store['replays'] = 0
            store['external_kv_write_replays'] = 0
            store['warmups'] = 0
            store['failures'] = 0
            store['last_failure'] = ""
        return store

    def _finalize_prefill_timing(self, timing_events: Optional[dict], **meta) -> None:
        if not timing_events:
            return
        torch.cuda.synchronize()
        summary = {k: v for k, v in meta.items()}
        total_ms = 0.0
        detail_only = {
            f"{aggregate}_{topology}"
            for aggregate in ("qkv", "attn_prepare", "attn_core", "o_proj")
            for topology in ("sliding", "full")
        }
        for name, pairs in timing_events.items():
            ms = sum(start.elapsed_time(end) for start, end in pairs)
            summary[f"{name}_ms"] = ms
            if name not in detail_only:
                total_ms += ms
        summary["total_ms"] = total_ms
        self._last_prefill_timing = summary
        if _PREFILL_TIMING_PRINT:
            ordered = [
                ("mlp_native_ms", "mlp_native"),
                ("qkv_ms", "qkv"),
                ("qkv_sliding_ms", "qkv_sliding"),
                ("qkv_full_ms", "qkv_full"),
                ("attn_prepare_ms", "attn_prepare"),
                ("attn_prepare_sliding_ms", "prepare_sliding"),
                ("attn_prepare_full_ms", "prepare_full"),
                ("attn_core_ms", "attn"),
                ("attn_core_sliding_ms", "attn_sliding"),
                ("attn_core_full_ms", "attn_full"),
                ("o_proj_ms", "o"),
                ("o_proj_sliding_ms", "o_sliding"),
                ("o_proj_full_ms", "o_full"),
                ("gate_up_ms", "gate_up"),
                ("down_proj_ms", "down"),
                ("moe_router_ms", "moe_router"),
                ("moe_experts_ms", "moe_experts"),
                ("gemma4_norms_ms", "gemma4_norms"),
                ("gemma4_residual_scale_ms", "gemma4_residual"),
                ("ple_ms", "ple"),
                ("prefill_setup_ms", "setup"),
                ("embedding_ms", "embedding"),
                ("per_layer_input_ms", "per_layer_input"),
                ("kv_write_ms", "kv"),
                ("final_norm_ms", "final_norm"),
                ("lm_head_ms", "lm_head"),
                ("logit_cap_ms", "logit_cap"),
            ]
            parts = []
            for key, label in ordered:
                if key in summary:
                    parts.append(f"{label}={summary[key]:.1f}ms")
            for key in ("num_seqs", "total_tokens", "max_len"):
                if key in summary:
                    parts.append(f"{key}={summary[key]}")
            parts.append(f"total={total_ms:.1f}ms")
            print("prefill_timing " + " | ".join(parts))

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,      # [1, seq_len]
        positions: torch.Tensor,       # [1, seq_len]
        block_manager: BlockManagerLike,  # BlockManager
        seq_id: int,
        logit_lens: Union[bool, int] = False,
        last_token_only: bool = False,
    ) -> torch.Tensor:
        """
        Prefill pass: process full prompt and cache KV.
        Returns logits for last token.

        Args:
            logit_lens: Controls Logit Lens probing.
                - False: disabled (default)
                - True: probe ALL layers
                - int N: probe every Nth layer + first + last (stride mode)
                Returns (logits, {layer_idx: probe_logits}) when enabled.
            last_token_only: Project only the final hidden state through the
                vocabulary head. Generation callers should enable this because
                they consume only next-token logits.
        """
        self._move_rope_to_device(input_ids.device)
        if last_token_only:
            self._prefill_last_token_only_hits += 1
        timing_events = {} if (_prefill_timing_enabled() and input_ids.is_cuda) else None

        hidden = self.embed_tokens(input_ids)
        hidden = self._scale_token_embeddings(hidden)
        per_layer_inputs = self._compute_per_layer_inputs(input_ids, hidden)
        shared_prefill_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        offloader = self._offloader
        do_lens = logit_lens is not False and logit_lens is not None
        layer_probes = {} if do_lens else None
        num_layers = len(self.layers)
        stride = logit_lens if isinstance(logit_lens, int) and logit_lens > 1 else 1

        for layer_idx, layer in enumerate(self.layers):
            # Offload: bring current layer to GPU, then prefetch next
            if offloader:
                layer = offloader.get_layer_on_gpu(layer_idx, layer)
                offloader.prefetch_next(layer_idx + 1, self.layers)

            linear_conv_state = None
            linear_recurrent_state = None
            if layer.layer_type == 'linear_attention':
                linear_conv_state, linear_recurrent_state = block_manager.get_linear_state(
                    seq_id, layer_idx, device=input_ids.device,
                )

            cos, sin = self._get_layer_rope(layer_idx)
            if getattr(layer, "self_attn", None) is not None and layer.self_attn.is_kv_shared:
                layer.self_attn._prefill_shared_kv = shared_prefill_kv.get(layer.self_attn.kv_share_source)
            hidden, k_cache, v_cache, next_linear_conv, next_linear_recurrent = layer(
                hidden, cos, sin, positions,
                is_prefill=True,
                # A single prompt is dense and has no padding. Propagate the
                # same causal fact used by uniform prefill_batch so Gemma 4's
                # exact-shape long-prefill dispatchers can serve B1 as well.
                implicit_causal_prefill=(
                    self.config.model_type == 'gemma4_text'
                ),
                linear_conv_state=linear_conv_state,
                linear_recurrent_state=linear_recurrent_state,
                use_linear_cache=True,
                timing_events=timing_events,
                per_layer_input=(
                    per_layer_inputs[:, :, layer_idx, :]
                    if per_layer_inputs is not None else None
                ),
            )
            if k_cache is not None and v_cache is not None:
                shared_prefill_kv[layer_idx] = (k_cache, v_cache)
                kv_write_start_end = _timing_record_start(timing_events is not None)
                block_manager.write_kv(seq_id, layer_idx, k_cache[0], v_cache[0])
                _timing_record_end(timing_events, "kv_write", kv_write_start_end)
            if getattr(layer, "self_attn", None) is not None:
                layer.self_attn._prefill_shared_kv = None
            if next_linear_conv is not None or next_linear_recurrent is not None:
                block_manager.set_linear_state(
                    seq_id,
                    layer_idx,
                    next_linear_conv[0] if next_linear_conv is not None else None,
                    next_linear_recurrent[0] if next_linear_recurrent is not None else None,
                )

            # Logit Lens: probe intermediate hidden state
            # With stride: probe first, last, and every stride-th layer
            if do_lens and (
                layer_idx == 0 or
                layer_idx == num_layers - 1 or
                layer_idx % stride == 0
            ):
                with torch.no_grad():
                    probe_hidden = hidden[:, -1:, :] if last_token_only else hidden
                    probe = self.norm(probe_hidden)
                    probe_logits = self.lm_head(probe)
                    probe_logits = self._apply_final_logit_capping(probe_logits)
                    layer_probes[layer_idx] = probe_logits[:, -1, :].squeeze(0)

            # Offload: release layer back to CPU
            if offloader:
                offloader.release_layer(layer_idx, layer)

        block_manager.advance_seq_len(seq_id, input_ids.shape[1])

        lm_head_start_end = _timing_record_start(timing_events is not None)
        hidden_for_logits = hidden[:, -1:, :] if last_token_only else hidden
        hidden_for_logits = self.norm(hidden_for_logits)
        logits = self.lm_head(hidden_for_logits)
        logits = self._apply_final_logit_capping(logits)
        _timing_record_end(timing_events, "lm_head", lm_head_start_end)
        self._finalize_prefill_timing(
            timing_events,
            num_seqs=1,
            total_tokens=int(input_ids.shape[1]),
            max_len=int(input_ids.shape[1]),
        )

        if do_lens:
            return logits, layer_probes
        return logits

    @torch.inference_mode()
    def prefill_suffix(
        self,
        input_ids: torch.Tensor,      # [1, suffix_len]
        positions: torch.Tensor,      # [1, suffix_len]
        block_manager: BlockManagerLike,
        seq_id: int,
    ) -> torch.Tensor:
        """
        Append a known suffix onto an existing live context using true prefill.

        The suffix tokens attend to cached prefix KV from the live sequence
        instead of being replayed one token at a time through decode.
        """
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("prefill_suffix currently expects input_ids shaped [1, suffix_len]")
        if positions.shape != input_ids.shape:
            raise ValueError("positions must have the same shape as input_ids")
        if self._has_linear_layers():
            raise NotImplementedError(
                "prefill_suffix currently supports full-attention models only"
            )
        if self.config.model_type == 'gemma4_text':
            raise NotImplementedError(
                "Gemma 4 suffix prefill is disabled until sliding/full masks and KV sharing "
                "are validated for prefix-appends."
            )
        if int(input_ids.shape[1]) <= 0:
            raise ValueError("prefill_suffix requires at least one suffix token")

        self._move_rope_to_device(input_ids.device)
        seq_lens = block_manager.get_seq_lens_tensor([seq_id])
        prefix_len = int(seq_lens[0].item())
        if prefix_len <= 0:
            return self.prefill(input_ids, positions, block_manager, seq_id)

        timing_events = {} if (_prefill_timing_enabled() and input_ids.is_cuda) else None
        hidden = self.embed_tokens(input_ids)
        hidden = self._scale_token_embeddings(hidden)

        suffix_len = int(input_ids.shape[1])
        attn_mask = _get_suffix_prefill_attn_mask(
            prefix_len,
            suffix_len,
            input_ids.device,
            hidden.dtype,
        )
        block_table = block_manager.get_block_table_tensor([seq_id])
        offloader = self._offloader

        for layer_idx, layer in enumerate(self.layers):
            if offloader:
                layer = offloader.get_layer_on_gpu(layer_idx, layer)
                offloader.prefetch_next(layer_idx + 1, self.layers)

            hidden, k_cache, v_cache, _, _ = layer(
                hidden,
                self.cos_cache,
                self.sin_cache,
                positions,
                kv_cache=block_manager.get_kv_cache(layer_idx),
                block_table=block_table,
                seq_lens=seq_lens,
                is_prefill=True,
                attn_mask=attn_mask,
                append_kv_prefix=True,
                timing_events=timing_events,
            )
            if k_cache is not None and v_cache is not None:
                kv_write_start_end = _timing_record_start(timing_events is not None)
                block_manager.write_kv(seq_id, layer_idx, k_cache[0], v_cache[0])
                _timing_record_end(timing_events, "kv_write", kv_write_start_end)

            if offloader:
                offloader.release_layer(layer_idx, layer)

        block_manager.advance_seq_len(seq_id, suffix_len)

        hidden_last = self.norm(hidden[:, -1:, :])
        logits = self.lm_head(hidden_last)
        logits = self._apply_final_logit_capping(logits)
        self._finalize_prefill_timing(
            timing_events,
            num_seqs=1,
            total_tokens=suffix_len,
            max_len=prefix_len + suffix_len,
            prefix_len=prefix_len,
            mode="suffix_prefill",
        )
        return logits

    @torch.inference_mode()
    def prefill_batch(
        self,
        input_ids: torch.Tensor,    # [N, max_len] — left-padded with 0
        lengths: torch.Tensor,       # [N] — actual length per sequence
        block_manager: BlockManagerLike,
        seq_ids: List[int],
        prompt_lengths_cpu: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """
        Batched prefill: process N prompts in a single forward pass.

        Sequences are LEFT-padded so that the last token of each sequence
        aligns at position max_len-1. An attention mask prevents attending
        to pad tokens.

        Args:
            input_ids: [N, max_len] left-padded input token IDs
            lengths: [N] actual token count per sequence
            block_manager: KV cache block manager
            seq_ids: list of sequence IDs (one per prompt)
            prompt_lengths_cpu: scheduler-owned lengths. Supplying these avoids
                synchronizing the GPU ``lengths`` tensor back to the host.

        Returns:
            logits: [N, 1, vocab_size] logits for last real token per sequence
        """
        self._move_rope_to_device(input_ids.device)
        timing_events = {} if (_prefill_timing_enabled() and input_ids.is_cuda) else None

        N, max_len = input_ids.shape
        device = input_ids.device
        finite_trace = self._gemma4_prefill_finite_trace
        if finite_trace is not None and finite_trace.get("enabled"):
            finite_trace.update({
                "status": "RUNNING",
                "batch_size": int(N),
                "seq_len": int(max_len),
                "events": [],
                "first_bad": None,
            })
            for layer in self.layers:
                layer._gemma4_prefill_finite_trace = finite_trace
                if getattr(layer, "self_attn", None) is not None:
                    layer.self_attn._gemma4_prefill_finite_trace = finite_trace
        lengths_cpu = (
            [int(length) for length in prompt_lengths_cpu]
            if prompt_lengths_cpu is not None
            else [int(length) for length in lengths.tolist()]
        )
        if len(lengths_cpu) != N:
            raise ValueError(
                f"Expected {N} prompt lengths, received {len(lengths_cpu)}"
            )
        uniform_full_length = all(length == max_len for length in lengths_cpu)

        # --- Build attention mask: causal + padding ---
        setup_start_end = _timing_record_start(timing_events is not None)
        # pad_mask[i, j] = True if position j is a real token for sequence i
        offsets = max_len - lengths  # [N] — how many pad tokens per seq
        pos_range = self._get_prefill_arange(max_len, device)  # [max_len]
        pad_mask = pos_range.unsqueeze(0) >= offsets.unsqueeze(1)  # [N, max_len]
        linear_attn_mask = pad_mask

        # Combined mask: [N, 1, max_len, max_len]
        # causal: position i can attend to j where j <= i
        # padding: can only attend to non-pad positions
        use_implicit_causal = (
            _GEMMA4_IMPLICIT_CAUSAL_PREFILL
            and self.config.model_type == 'gemma4_text'
            and uniform_full_length
        )
        if use_implicit_causal:
            self._gemma4_implicit_causal_prefill_batches += 1
        causal = self._get_prefill_causal_mask(max_len, device)
        key_mask = pad_mask.unsqueeze(1).unsqueeze(2)
        causal_mask = causal.unsqueeze(0).unsqueeze(0)
        combined = causal_mask & key_mask

        # CRITICAL: pad positions have ALL keys masked → softmax(all -inf) = NaN.
        # NaN propagates through residual → NaN K,V in next layer → contaminates
        # real tokens via Q·K_nan = NaN, and NaN + (-inf) = NaN, not -inf!
        # Fix: let every position self-attend (diagonal=True). Pad positions get
        # a defined output (attending to themselves) instead of NaN.
        # Real positions already self-attend via causal mask, so no change for them.
        diag = torch.eye(max_len, device=device, dtype=torch.bool)
        combined = combined | diag.unsqueeze(0).unsqueeze(0)

        # Keep additive-mask form here because on some CUDA/SDPA combinations
        # it preserves the faster backend selection better than a bool mask.
        attn_mask = torch.zeros(
            N, 1, max_len, max_len,
            device=device, dtype=self.embed_tokens.weight.dtype,
        )
        attn_mask.masked_fill_(~combined, float('-inf'))

        # --- Build position IDs (adjusted for left-padding) ---
        # For left-padded seq: positions should be [0,0,...,0,1,2,...,len-1]
        positions = (pos_range.unsqueeze(0) - offsets.unsqueeze(1)).clamp(min=0)  # [N, max_len]
        _timing_record_end(timing_events, "prefill_setup", setup_start_end)

        # --- Forward pass through all layers ---
        embedding_start_end = _timing_record_start(timing_events is not None)
        hidden = self.embed_tokens(input_ids)
        hidden = self._scale_token_embeddings(hidden)
        _timing_record_end(timing_events, "embedding", embedding_start_end)
        if finite_trace is not None:
            _record_gemma4_prefill_finite_trace(
                finite_trace, -1, "model.embedding", hidden
            )
        per_layer_input_start_end = _timing_record_start(timing_events is not None)
        per_layer_inputs = self._compute_per_layer_inputs(input_ids, hidden)
        _timing_record_end(
            timing_events,
            "per_layer_input",
            per_layer_input_start_end,
        )
        shared_prefill_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        (
            parallel_moe_stream,
            parallel_moe_fork_events,
            parallel_moe_join_events,
        ) = self._prepare_gemma4_parallel_moe_prefill(
            hidden,
            batch_size=N,
            seq_len=max_len,
            uniform_full_length=uniform_full_length,
            timing_events=timing_events,
        )

        # Gemma 4 cannot use the generic packed-attention path. Equal-length
        # batches are still contiguous, though, so build one KV mapping and
        # reuse one vectorized scatter in every layer instead of B host loops.
        vectorized_kv = (
            _GEMMA4_VECTORIZED_PREFILL_KV
            and self.config.model_type == 'gemma4_text'
            and uniform_full_length
            and callable(getattr(block_manager, 'write_kv_prefill_packed', None))
            and callable(getattr(block_manager, 'compute_kv_mapping', None))
        )
        prefill_cu_seqlens = None
        prefill_kv_mapping = None
        if vectorized_kv:
            prefill_cu_seqlens = (
                torch.arange(N + 1, dtype=torch.int32, device=device) * int(max_len)
            )
            try:
                prefill_kv_mapping = block_manager.compute_kv_mapping(
                    seq_ids,
                    prefill_cu_seqlens,
                    device,
                    seq_lengths=lengths_cpu,
                )
            except TypeError:
                # Third-party block managers may still expose the older API.
                prefill_kv_mapping = block_manager.compute_kv_mapping(
                    seq_ids, prefill_cu_seqlens, device
                )
        kv_scatter_device_name = (
            torch.cuda.get_device_name(device)
            if vectorized_kv and hidden.is_cuda
            else ""
        )

        offloader = self._offloader

        for layer_idx, layer in enumerate(self.layers):
            if offloader:
                layer = offloader.get_layer_on_gpu(layer_idx, layer)
                offloader.prefetch_next(layer_idx + 1, self.layers)

            linear_conv_state = None
            linear_recurrent_state = None
            if layer.layer_type == 'linear_attention':
                linear_conv_state, linear_recurrent_state = block_manager.get_linear_state_batch(
                    seq_ids, layer_idx, device=device,
                )

            cos, sin = self._get_layer_rope(layer_idx)
            if getattr(layer, "self_attn", None) is not None and layer.self_attn.is_kv_shared:
                layer.self_attn._prefill_shared_kv = shared_prefill_kv.get(layer.self_attn.kv_share_source)
            hidden, k_cache_out, v_cache_out, next_linear_conv, next_linear_recurrent = layer(
                hidden, cos, sin, positions,
                is_prefill=True, attn_mask=attn_mask,
                implicit_causal_prefill=use_implicit_causal,
                gemma4_parallel_moe_prefill_stream=parallel_moe_stream,
                gemma4_parallel_moe_prefill_fork_event=(
                    parallel_moe_fork_events[layer_idx]
                    if parallel_moe_fork_events is not None
                    else None
                ),
                gemma4_parallel_moe_prefill_join_event=(
                    parallel_moe_join_events[layer_idx]
                    if parallel_moe_join_events is not None
                    else None
                ),
                linear_conv_state=linear_conv_state,
                linear_recurrent_state=linear_recurrent_state,
                linear_attention_mask=linear_attn_mask,
                use_linear_cache=True,
                timing_events=timing_events,
                per_layer_input=(
                    per_layer_inputs[:, :, layer_idx, :]
                    if per_layer_inputs is not None else None
                ),
            )

            if k_cache_out is not None and v_cache_out is not None:
                shared_prefill_kv[layer_idx] = (k_cache_out, v_cache_out)
                kv_write_start_end = _timing_record_start(timing_events is not None)
                if vectorized_kv:
                    k_cache_flat = k_cache_out.reshape(
                        -1, *k_cache_out.shape[-2:]
                    )
                    v_cache_flat = v_cache_out.reshape(
                        -1, *v_cache_out.shape[-2:]
                    )
                    tokens_per_program = (
                        _gemma4_a100_a4b_long_kv_scatter_tokens_per_program(
                            N,
                            max_len,
                            int(k_cache_flat.shape[-2]),
                            int(k_cache_flat.shape[-1]),
                            k_cache_flat.dtype,
                            kv_scatter_device_name,
                        )
                    )
                    try:
                        block_manager.write_kv_prefill_packed(
                            seq_ids,
                            layer_idx,
                            k_cache_flat,
                            v_cache_flat,
                            prefill_cu_seqlens,
                            kv_mapping=prefill_kv_mapping,
                            tokens_per_program=tokens_per_program,
                        )
                    except TypeError:
                        block_manager.write_kv_prefill_packed(
                            seq_ids,
                            layer_idx,
                            k_cache_flat,
                            v_cache_flat,
                            prefill_cu_seqlens,
                            kv_mapping=prefill_kv_mapping,
                        )
                    self._gemma4_batch_prefill_vectorized_kv_hits += 1
                else:
                    # Variable lengths need to strip left padding per sequence.
                    for i, seq_id in enumerate(seq_ids):
                        real_len = lengths_cpu[i]
                        offset = max_len - real_len
                        block_manager.write_kv(
                            seq_id,
                            layer_idx,
                            k_cache_out[i, offset:],
                            v_cache_out[i, offset:],
                        )
                _timing_record_end(timing_events, "kv_write", kv_write_start_end)
                if finite_trace is not None:
                    _record_gemma4_prefill_finite_trace(
                        finite_trace,
                        layer_idx,
                        "model.hidden_after_kv_write",
                        hidden,
                    )
            if getattr(layer, "self_attn", None) is not None:
                layer.self_attn._prefill_shared_kv = None

            if next_linear_conv is not None or next_linear_recurrent is not None:
                block_manager.set_linear_state_batch(
                    seq_ids, layer_idx, next_linear_conv, next_linear_recurrent,
                )

            if offloader:
                offloader.release_layer(layer_idx, layer)

        # Advance seq lens for all sequences
        for i, seq_id in enumerate(seq_ids):
            block_manager.advance_seq_len(seq_id, int(lengths_cpu[i]))

        # Left-padding aligns every final real token at -1. Project only those
        # N rows through Gemma's 262k-vocabulary head, never N * max_len rows.
        self._prefill_last_token_only_hits += 1
        final_norm_start_end = _timing_record_start(timing_events is not None)
        last_hidden = self.norm(hidden[:, -1:, :])
        _timing_record_end(timing_events, "final_norm", final_norm_start_end)
        if finite_trace is not None:
            _record_gemma4_prefill_finite_trace(
                finite_trace, -1, "model.final_norm", last_hidden
            )
        lm_head_start_end = _timing_record_start(timing_events is not None)
        last_logits = self.lm_head(last_hidden)
        _timing_record_end(timing_events, "lm_head", lm_head_start_end)
        if finite_trace is not None:
            _record_gemma4_prefill_finite_trace(
                finite_trace, -1, "model.raw_logits", last_logits
            )
        logit_cap_start_end = _timing_record_start(timing_events is not None)
        last_logits = self._apply_final_logit_capping(last_logits)
        _timing_record_end(timing_events, "logit_cap", logit_cap_start_end)
        if finite_trace is not None:
            _record_gemma4_prefill_finite_trace(
                finite_trace, -1, "model.capped_logits", last_logits
            )
            finite_trace["status"] = "PASS"
        self._finalize_prefill_timing(
            timing_events,
            num_seqs=int(N),
            total_tokens=sum(lengths_cpu),
            max_len=int(max_len),
        )

        return last_logits

    @torch.inference_mode()
    def prefill_batch_graph(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        block_manager: BlockManagerLike,
        kv_phys: torch.Tensor,
        kv_offs: torch.Tensor,
        *,
        defer_kv_writes: bool = False,
        graph_safe_prefill: bool = True,
    ):
        """Graph-safe equal-length padded prefill for Gemma 4.

        Gemma 4 attention consumes the batch dimension and an explicit causal
        mask, so its packed representation cannot preserve request boundaries.
        Sequence bookkeeping and token-to-KV mapping are prepared by the
        scheduler before capture and updated after replay. When
        ``defer_kv_writes`` is true, the graph contains only model compute and
        returns stable per-layer K/V tensors for the scheduler to scatter after
        replay. Keeping mutable cache writes outside the graph avoids capturing
        advanced-index writes into the long-lived paged KV allocation.
        """
        if self.config.model_type != 'gemma4_text':
            raise RuntimeError("Padded graph prefill is only implemented for Gemma 4")
        if self.hidden_size_per_layer_input:
            raise RuntimeError(
                "Gemma4 padded graph prefill does not support per-layer inputs"
            )
        if self._offloader is not None:
            raise RuntimeError("Gemma4 padded graph prefill does not support offload")
        if any(
            bool(getattr(getattr(layer, "self_attn", None), "is_kv_shared", False))
            for layer in self.layers
        ):
            raise RuntimeError(
                "Gemma4 padded graph prefill does not support shared KV layers"
            )

        self._move_rope_to_device(input_ids.device)
        num_seqs, max_len = input_ids.shape
        total_tokens = int(num_seqs) * int(max_len)
        if int(cu_seqlens.numel()) != int(num_seqs) + 1:
            raise ValueError("Invalid cu_seqlens shape for padded graph prefill")
        if int(kv_phys.numel()) != total_tokens or int(kv_offs.numel()) != total_tokens:
            raise ValueError("Invalid KV mapping shape for padded graph prefill")

        device = input_ids.device
        pos_range = self._get_prefill_arange(max_len, device)
        positions = pos_range.unsqueeze(0).expand(num_seqs, -1).contiguous()
        use_implicit_causal = bool(_GEMMA4_IMPLICIT_CAUSAL_PREFILL)
        if use_implicit_causal:
            self._gemma4_implicit_causal_prefill_batches += 1
        causal = self._get_prefill_causal_mask(max_len, device)
        combined = causal.unsqueeze(0).unsqueeze(0).expand(
            num_seqs, 1, max_len, max_len
        )
        attn_mask = torch.zeros(
            num_seqs,
            1,
            max_len,
            max_len,
            device=device,
            dtype=self.embed_tokens.weight.dtype,
        )
        attn_mask.masked_fill_(~combined, float('-inf'))

        hidden = self.embed_tokens(input_ids)
        hidden = self._scale_token_embeddings(hidden)

        kv_mapping = (kv_phys, kv_offs)
        deferred_kv = []
        for layer_idx, layer in enumerate(self.layers):
            cos, sin = self._get_layer_rope(layer_idx)
            prefill_kv_out = None
            persistent_k = None
            persistent_v = None
            if defer_kv_writes:
                attn = getattr(layer, "self_attn", None)
                if attn is None:
                    raise RuntimeError(
                        "Gemma4 padded graph prefill requires attention in every layer"
                    )
                flat_cache_shape = (
                    total_tokens,
                    int(attn.num_kv_heads),
                    int(attn.head_dim),
                )
                persistent_k, persistent_v = (
                    self._get_prefill_graph_deferred_kv_buffers(
                        layer_idx,
                        flat_cache_shape,
                        hidden,
                    )
                )
                prefill_kv_out = (
                    persistent_k.view(
                        num_seqs,
                        max_len,
                        int(attn.num_kv_heads),
                        int(attn.head_dim),
                    ),
                    persistent_v.view(
                        num_seqs,
                        max_len,
                        int(attn.num_kv_heads),
                        int(attn.head_dim),
                    ),
                )
            hidden, k_cache_out, v_cache_out, _, _ = layer(
                hidden,
                cos,
                sin,
                positions,
                is_prefill=True,
                attn_mask=attn_mask,
                implicit_causal_prefill=use_implicit_causal,
                use_linear_cache=True,
                timing_events=None,
                graph_safe_prefill=graph_safe_prefill,
                prefill_kv_out=prefill_kv_out,
            )
            if k_cache_out is not None and v_cache_out is not None:
                k_cache_flat = k_cache_out.reshape(
                    -1, *k_cache_out.shape[-2:]
                )
                v_cache_flat = v_cache_out.reshape(
                    -1, *v_cache_out.shape[-2:]
                )
                if defer_kv_writes:
                    if (
                        persistent_k is None
                        or persistent_v is None
                        or int(k_cache_flat.data_ptr()) != int(persistent_k.data_ptr())
                        or int(v_cache_flat.data_ptr()) != int(persistent_v.data_ptr())
                    ):
                        raise RuntimeError(
                            "Gemma4 attention did not write K/V directly to the "
                            "persistent graph outputs"
                        )
                    self._prefill_graph_deferred_kv_copy_dispatches += 1
                    deferred_kv.append((layer_idx, persistent_k, persistent_v))
                else:
                    block_manager.write_kv_prefill_packed(
                        [],
                        layer_idx,
                        k_cache_flat,
                        v_cache_flat,
                        cu_seqlens,
                        kv_mapping=kv_mapping,
                    )
                self._gemma4_batch_prefill_vectorized_kv_hits += 1

        self._prefill_last_token_only_hits += 1
        last_hidden = self.norm(hidden[:, -1:, :])
        last_logits = self.lm_head(last_hidden)
        last_logits = self._apply_final_logit_capping(last_logits)
        if defer_kv_writes:
            return last_logits, tuple(deferred_kv)
        return last_logits

    @torch.inference_mode()
    def prefill_packed(
        self,
        input_ids: torch.Tensor,       # [1, total_tokens] — all seqs concatenated
        cu_seqlens: torch.Tensor,      # [num_seqs + 1], int32
        lengths: torch.Tensor,         # [num_seqs] — token count per seq
        block_manager: BlockManagerLike,
        seq_ids: List[int],
    ) -> torch.Tensor:
        """
        Packed batched prefill: process N prompts in a single forward pass
        WITHOUT padding. Sequences are concatenated into one tensor.

        Args:
            input_ids: [1, total_tokens] all prompt tokens concatenated
            cu_seqlens: [num_seqs + 1] cumulative lengths, e.g. [0, 100, 350]
            lengths: [num_seqs] per-sequence token count
            block_manager: KV cache block manager
            seq_ids: list of sequence IDs (one per prompt)

        Returns:
            logits: [num_seqs, 1, vocab_size] logits for last real token per seq
        """
        use_packed_linear_native = _get_env_bool("MEGAGEMM_QWEN35_PACKED_LINEAR_NATIVE", False)
        if self._has_linear_layers() and not use_packed_linear_native:
            # Stable default: keep padded batch path for linear layers.
            max_len = int(lengths.max().item())
            padded = torch.zeros(
                len(seq_ids), max_len, dtype=input_ids.dtype, device=input_ids.device
            )
            for i in range(len(seq_ids)):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                seq = input_ids[0, start:end]
                padded[i, max_len - seq.numel():] = seq
            return self.prefill_batch(padded, lengths, block_manager, seq_ids)

        if self.config.model_type == 'gemma4_text':
            max_len = int(lengths.max().item())
            padded = torch.zeros(
                len(seq_ids), max_len, dtype=input_ids.dtype, device=input_ids.device
            )
            for i in range(len(seq_ids)):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                seq = input_ids[0, start:end]
                padded[i, max_len - seq.numel():] = seq
            return self.prefill_batch(padded, lengths, block_manager, seq_ids)

        self._move_rope_to_device(input_ids.device)
        timing_events = {} if (_prefill_timing_enabled() and input_ids.is_cuda) else None

        # GPU tuning flags (set once) — same as vLLM/SGLang
        if not hasattr(self, '_gpu_tuned'):
            self._gpu_tuned = True
            # Allow FP16 reduced precision accumulation in cuBLAS
            # This can be 15-25% faster for large FP16 GEMMs
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            # Allow TF32 for any FP32 fallback paths
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Set medium precision (allows more aggressive math optimizations)
            torch.set_float32_matmul_precision('medium')
            # Configure memory allocator for less fragmentation
            import os
            if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        device = input_ids.device
        total_tokens = input_ids.shape[1]
        num_seqs = len(seq_ids)

        # --- Build position IDs: vectorized (zero GPU-CPU sync) ---
        global_idx = self._get_prefill_arange(total_tokens, device)
        seq_idx = torch.searchsorted(cu_seqlens[1:], global_idx, right=True)
        positions = (global_idx - cu_seqlens[seq_idx]).unsqueeze(0)

        # --- Forward pass through all layers ---
        hidden = self.embed_tokens(input_ids)
        hidden = self._scale_token_embeddings(hidden)

        offloader = self._offloader

        needs_cpu_segment_bounds = not hasattr(block_manager, 'write_kv_prefill_packed')
        if needs_cpu_segment_bounds:
            # Only materialize CPU boundaries when we truly need the slow fallback path.
            _cu_starts_cpu = cu_seqlens[:-1].tolist()
            _cu_ends_cpu = cu_seqlens[1:].tolist()

        # Pre-compute KV block mapping ONCE — reused for all 28 layers
        kv_mapping = None
        if hasattr(block_manager, 'compute_kv_mapping'):
            kv_mapping = block_manager.compute_kv_mapping(
                seq_ids, cu_seqlens, device,
            )

        # Profiling: measure where time goes (only first call)
        import os as _os
        _do_profile = _os.environ.get('MEGAGEMM_PROFILE_PREFILL', '') == '1'
        _t_layer = 0.0
        _t_kv = 0.0
        import time as _time

        for layer_idx, layer in enumerate(self.layers):
            if offloader:
                layer = offloader.get_layer_on_gpu(layer_idx, layer)
                offloader.prefetch_next(layer_idx + 1, self.layers)

            if layer.layer_type == 'linear_attention':
                # Process each packed segment independently for linear-attention
                # layers so recurrent state does not leak across sequences.
                for i, seq_id in enumerate(seq_ids):
                    start = int(cu_seqlens[i].item())
                    end = int(cu_seqlens[i + 1].item())
                    hidden_i = hidden[:, start:end]
                    positions_i = positions[:, start:end]
                    cos, sin = self._get_layer_rope(layer_idx)
                    linear_conv_state, linear_recurrent_state = block_manager.get_linear_state(
                        seq_id, layer_idx, device=device,
                    )
                    hidden_i, _, _, next_linear_conv, next_linear_recurrent = layer(
                        hidden_i, cos, sin, positions_i,
                        is_prefill=True,
                        linear_conv_state=linear_conv_state,
                        linear_recurrent_state=linear_recurrent_state,
                        use_linear_cache=True,
                    )
                    if next_linear_conv is not None or next_linear_recurrent is not None:
                        block_manager.set_linear_state(
                            seq_id,
                            layer_idx,
                            next_linear_conv[0] if next_linear_conv is not None else None,
                            next_linear_recurrent[0] if next_linear_recurrent is not None else None,
                        )
                    hidden[:, start:end].copy_(hidden_i)
                if offloader:
                    offloader.release_layer(layer_idx, layer)
                continue

            if _do_profile:
                torch.cuda.synchronize()
                _t0 = _time.perf_counter()
            cos, sin = self._get_layer_rope(layer_idx)
            hidden, k_cache_out, v_cache_out, next_linear_conv, next_linear_recurrent = layer(
                hidden, cos, sin, positions,
                is_prefill=True, cu_seqlens=cu_seqlens,
                use_linear_cache=True,
                timing_events=timing_events,
            )
            if _do_profile:
                torch.cuda.synchronize()
                _t_layer += _time.perf_counter() - _t0
                _t0 = _time.perf_counter()

            # Write KV to cache — vectorized with pre-computed mapping
            # k_cache_out: [1, total_tokens, num_kv_heads, head_dim]
            if k_cache_out is not None and v_cache_out is not None:
                kv_write_start_end = _timing_record_start(timing_events is not None)
                if hasattr(block_manager, 'write_kv_prefill_packed'):
                    block_manager.write_kv_prefill_packed(
                        seq_ids, layer_idx,
                        k_cache_out[0], v_cache_out[0],
                        cu_seqlens, kv_mapping=kv_mapping,
                    )
                else:
                    for i, seq_id in enumerate(seq_ids):
                        k_real = k_cache_out[0, _cu_starts_cpu[i]:_cu_ends_cpu[i]]
                        v_real = v_cache_out[0, _cu_starts_cpu[i]:_cu_ends_cpu[i]]
                        block_manager.write_kv(seq_id, layer_idx, k_real, v_real)
                _timing_record_end(timing_events, "kv_write", kv_write_start_end)
            if _do_profile:
                torch.cuda.synchronize()
                _t_kv += _time.perf_counter() - _t0

            if next_linear_conv is not None or next_linear_recurrent is not None:
                block_manager.set_linear_state_batch(
                    seq_ids, layer_idx, next_linear_conv, next_linear_recurrent,
                )

            if offloader:
                offloader.release_layer(layer_idx, layer)

        # Profiling summary
        if _do_profile:
            _t_total = _t_layer + _t_kv
            print(f"  📊 PREFILL PROFILE ({total_tokens} tokens, {len(seq_ids)} seqs):")
            print(f"     Layer fwd: {_t_layer:.3f}s ({100*_t_layer/_t_total:.0f}%)")
            print(f"     KV write:  {_t_kv:.3f}s ({100*_t_kv/_t_total:.0f}%)")
            _os.environ['MEGAGEMM_PROFILE_PREFILL'] = ''  # only profile first chunk

        # Advance seq lens for all sequences
        lengths_cpu = lengths.tolist()  # single GPU-CPU transfer
        for i, seq_id in enumerate(seq_ids):
            block_manager.advance_seq_len(seq_id, int(lengths_cpu[i]))

        # --- Extract logits for last real token per sequence ---
        # CRITICAL: only project LAST token per seq through lm_head!
        # Before: lm_head(all 10K tokens) = 3 GB allocation (99.7% wasted)
        # After:  lm_head(N last tokens)  = 9 MB allocation
        last_indices = cu_seqlens[1:] - 1  # [num_seqs], on GPU
        hidden_last = hidden[0, last_indices, :]  # [num_seqs, hidden] - tiny!
        hidden_last = self.norm(hidden_last.unsqueeze(0))  # [1, num_seqs, hidden]
        logits = self.lm_head(hidden_last)  # [1, num_seqs, vocab] - only N * vocab * 2 bytes
        logits = self._apply_final_logit_capping(logits)
        last_logits = logits.transpose(0, 1)  # [num_seqs, 1, vocab]
        self._finalize_prefill_timing(
            timing_events,
            num_seqs=int(num_seqs),
            total_tokens=int(total_tokens),
            max_len=int(lengths.max().item()),
        )

        return last_logits

    # ── Zero-overhead flat decode ────────────────────────────────────

    @torch.inference_mode()
    def prefill_packed_graph(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        block_manager: BlockManagerLike,
        kv_phys: torch.Tensor,
        kv_offs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Graph-safe packed prefill path for stable inference shapes.

        This variant intentionally avoids CPU metadata work and timing side
        effects so that the scheduler can capture/replay it with CUDA Graphs.
        Sequence-length bookkeeping stays in the calling engine or scheduler.
        """
        if self.config.model_type == 'gemma4_text':
            if self.hidden_size_per_layer_input:
                raise RuntimeError(
                    "Gemma4 graph prefill does not support per-layer inputs"
                )
            if any(
                bool(getattr(getattr(layer, "self_attn", None), "is_kv_shared", False))
                for layer in self.layers
            ):
                raise RuntimeError(
                    "Gemma4 graph prefill does not support shared KV layers"
                )
        self._move_rope_to_device(input_ids.device)
        device = input_ids.device
        total_tokens = input_ids.shape[1]

        hidden = self.embed_tokens(input_ids)
        hidden = self._scale_token_embeddings(hidden)

        global_idx = self._get_prefill_arange(total_tokens, device)
        seq_idx = torch.searchsorted(cu_seqlens[1:], global_idx, right=True)
        positions = (global_idx - cu_seqlens[seq_idx]).unsqueeze(0)
        kv_mapping = (kv_phys, kv_offs)

        for layer_idx, layer in enumerate(self.layers):
            cos, sin = self._get_layer_rope(layer_idx)
            hidden, k_cache_out, v_cache_out, _, _ = layer(
                hidden,
                cos,
                sin,
                positions,
                is_prefill=True,
                cu_seqlens=cu_seqlens,
                use_linear_cache=True,
                timing_events=None,
            )
            if k_cache_out is not None and v_cache_out is not None:
                block_manager.write_kv_prefill_packed(
                    [],
                    layer_idx,
                    k_cache_out[0],
                    v_cache_out[0],
                    cu_seqlens,
                    kv_mapping=kv_mapping,
                )

        last_indices = cu_seqlens[1:] - 1
        hidden_last = hidden[0, last_indices, :]
        hidden_last = self.norm(hidden_last.unsqueeze(0))
        logits = self.lm_head(hidden_last)
        logits = self._apply_final_logit_capping(logits)
        return logits.transpose(0, 1)

    @staticmethod
    def _flat_fp_linear(
        x: torch.Tensor,
        wt: torch.Tensor,
        bias: Optional[torch.Tensor],
        out: torch.Tensor,
    ) -> torch.Tensor:
        torch.mm(x, wt, out=out)
        if bias is not None:
            out.add_(bias)
        return out

    def _prepare_gemma4_flat_decode(self):
        """Collect Gemma 4 text weights for a single-loop decode path."""
        try:
            cfg = self.config
            layer_weights = []
            max_qkv = 0
            max_q = 0
            max_attn_heads = 0
            max_head_dim = 0
            max_intermediate = 0
            max_ple = 0
            has_any_int8_linear = False

            def _gemma4_flat_linear_params(linear):
                weight, bias = _linear_weight_bias(linear)
                if weight is not None:
                    return weight, bias, None
                if linear is not None and hasattr(linear, "weight_int8") and hasattr(linear, "scale"):
                    return None, getattr(linear, "bias", None), linear
                return None, bias, None

            for layer_idx, layer in enumerate(self.layers):
                attn = layer.self_attn
                mlp = layer.mlp
                is_moe = isinstance(mlp, Gemma4MoeMLP)
                dense_mlp = mlp.shared_mlp if is_moe else mlp
                if (
                    attn._awq_separate
                    or getattr(dense_mlp, "_awq_separate", False)
                    or attn.attention_output_gate
                ):
                    self._flat_decode_failed = True
                    self._flat_decode_failed_reason = (
                        f"Gemma4 flat decode does not support layer {layer_idx} projection layout"
                    )
                    return

                lw = _Gemma4FlatLayerWeights()
                lw.layer_idx = layer_idx
                lw.layer_type = layer.layer_type
                lw.is_kv_shared = bool(attn.is_kv_shared)
                lw.kv_share_source = attn.kv_share_source
                lw.sliding_window = int(attn.sliding_window or 0)
                lw.num_q_heads = int(attn.num_q_heads)
                lw.num_kv_heads = int(attn.num_kv_heads)
                lw.head_dim = int(attn.head_dim)
                lw.q_size = int(attn._q_proj_size)
                lw.k_size = int(attn._k_size)
                lw.v_size = int(attn._v_size)
                lw.intermediate_size = int(dense_mlp.intermediate_size)
                lw.ple_size = int(getattr(layer, "hidden_size_per_layer_input", 0) or 0)
                lw.scale = float(attn.scale)
                lw.rotary_dim = int(attn.rotary_dim)
                lw.half_rotate = bool(attn.rope_half_rotate)

                lw.input_norm_weight = layer.input_layernorm.weight.data
                lw.post_attn_norm_weight = layer.post_attention_layernorm.weight.data
                lw.pre_ff_norm_weight = layer.pre_feedforward_layernorm.weight.data
                lw.post_ff_norm_weight = layer.post_feedforward_layernorm.weight.data
                lw.is_moe = bool(is_moe)
                lw.moe_module = mlp if is_moe else None
                lw.pre_expert_norm_weight = (
                    layer.pre_feedforward_layernorm_2.weight.data if is_moe else None
                )
                lw.post_shared_norm_weight = (
                    layer.post_feedforward_layernorm_1.weight.data if is_moe else None
                )
                lw.post_expert_norm_weight = (
                    layer.post_feedforward_layernorm_2.weight.data if is_moe else None
                )
                lw.q_norm_weight = attn.q_norm.weight.data if attn.q_norm is not None else None
                lw.k_norm_weight = attn.k_norm.weight.data if attn.k_norm is not None else None
                lw.has_v_norm = attn.v_norm is not None

                lw.qkv_wt = None
                lw.qkv_bias = None
                lw.qkv_mod = None
                lw.qkv_int8_w = None
                lw.qkv_int8_scale = None
                lw.q_wt = None
                lw.q_bias = None
                lw.q_mod = None
                lw.q_int8_w = None
                lw.q_int8_scale = None
                lw.k_wt = None
                lw.k_bias = None
                lw.k_mod = None
                lw.k_int8_w = None
                lw.k_int8_scale = None
                lw.v_wt = None
                lw.v_bias = None
                lw.v_mod = None
                lw.v_int8_w = None
                lw.v_int8_scale = None
                lw.v_from_k = False
                lw.o_wt = None
                lw.o_bias = None
                lw.o_mod = None
                lw.o_int8_w = None
                lw.o_int8_scale = None
                lw.gate_up_wt = None
                lw.gate_up_weight = None
                lw.gate_up_bias = None
                lw.gate_up_mod = None
                lw.gate_up_int8_w = None
                lw.gate_up_int8_scale = None
                lw.down_wt = None
                lw.down_weight = None
                lw.down_bias = None
                lw.down_mod = None
                lw.down_int8_w = None
                lw.down_int8_scale = None
                if lw.is_kv_shared:
                    q_weight, q_bias, q_mod = _gemma4_flat_linear_params(attn.q_proj)
                    if q_weight is None and q_mod is None:
                        self._flat_decode_failed = True
                        return
                    lw.q_bias = q_bias.data if q_bias is not None else None
                    if q_weight is not None:
                        lw.q_wt = q_weight.data.t()
                    else:
                        lw.q_mod = q_mod
                        lw.q_int8_w = q_mod.weight_int8.data.contiguous()
                        lw.q_int8_scale = q_mod.scale.data
                        has_any_int8_linear = True
                    lw.qkv_size = 0
                else:
                    qkv_weight, qkv_bias = attn._gemma4_fused_qkv_weight_bias()
                    if qkv_weight is not None:
                        lw.qkv_wt = qkv_weight.data.t()
                        lw.qkv_bias = qkv_bias.data if qkv_bias is not None else None
                        lw.qkv_size = int(qkv_weight.shape[0])
                        lw.v_from_k = bool(
                            attn.attention_k_eq_v and attn.v_proj is None
                        )
                    else:
                        q_weight, q_bias, q_mod = _gemma4_flat_linear_params(attn.q_proj)
                        k_weight, k_bias, k_mod = _gemma4_flat_linear_params(attn.k_proj)
                        v_weight, v_bias, v_mod = _gemma4_flat_linear_params(attn.v_proj)
                        if (q_weight is None and q_mod is None) or (k_weight is None and k_mod is None):
                            self._flat_decode_failed = True
                            return
                        lw.q_bias = q_bias.data if q_bias is not None else None
                        lw.k_bias = k_bias.data if k_bias is not None else None
                        if q_weight is not None:
                            lw.q_wt = q_weight.data.t()
                        else:
                            lw.q_mod = q_mod
                            lw.q_int8_w = q_mod.weight_int8.data.contiguous()
                            lw.q_int8_scale = q_mod.scale.data
                            has_any_int8_linear = True
                        if k_weight is not None:
                            lw.k_wt = k_weight.data.t()
                        else:
                            lw.k_mod = k_mod
                            lw.k_int8_w = k_mod.weight_int8.data.contiguous()
                            lw.k_int8_scale = k_mod.scale.data
                            has_any_int8_linear = True
                        if v_weight is None and v_mod is None:
                            lw.v_from_k = True
                        else:
                            lw.v_bias = v_bias.data if v_bias is not None else None
                            if v_weight is not None:
                                lw.v_wt = v_weight.data.t()
                            else:
                                lw.v_mod = v_mod
                                lw.v_int8_w = v_mod.weight_int8.data.contiguous()
                                lw.v_int8_scale = v_mod.scale.data
                                has_any_int8_linear = True
                        lw.qkv_size = 0

                o_weight, o_bias, o_mod = _gemma4_flat_linear_params(attn.o_proj)
                gate_up_weight, gate_up_bias, gate_up_mod = _gemma4_flat_linear_params(
                    dense_mlp.gate_up_proj
                )
                down_weight, down_bias, down_mod = _gemma4_flat_linear_params(
                    dense_mlp.down_proj
                )
                if (
                    (o_weight is None and o_mod is None)
                    or (gate_up_weight is None and gate_up_mod is None)
                    or (down_weight is None and down_mod is None)
                ):
                    self._flat_decode_failed = True
                    return
                lw.o_bias = o_bias.data if o_bias is not None else None
                lw.gate_up_bias = gate_up_bias.data if gate_up_bias is not None else None
                lw.down_bias = down_bias.data if down_bias is not None else None
                if o_weight is not None:
                    lw.o_wt = o_weight.data.t()
                else:
                    lw.o_mod = o_mod
                    lw.o_int8_w = o_mod.weight_int8.data.contiguous()
                    lw.o_int8_scale = o_mod.scale.data
                    has_any_int8_linear = True
                if gate_up_weight is not None:
                    lw.gate_up_wt = gate_up_weight.data.t()
                    lw.gate_up_weight = gate_up_weight.data
                else:
                    lw.gate_up_weight = None
                    lw.gate_up_mod = gate_up_mod
                    lw.gate_up_int8_w = gate_up_mod.weight_int8.data.contiguous()
                    lw.gate_up_int8_scale = gate_up_mod.scale.data
                    has_any_int8_linear = True
                if down_weight is not None:
                    lw.down_wt = down_weight.data.t()
                    lw.down_weight = down_weight.data
                else:
                    lw.down_weight = None
                    lw.down_mod = down_mod
                    lw.down_int8_w = down_mod.weight_int8.data.contiguous()
                    lw.down_int8_scale = down_mod.scale.data
                    has_any_int8_linear = True

                if lw.ple_size > 0 and layer.per_layer_input_gate is not None:
                    ple_gate_weight, _ = _linear_weight_bias(layer.per_layer_input_gate)
                    ple_proj_weight, _ = _linear_weight_bias(layer.per_layer_projection)
                    if ple_gate_weight is None or ple_proj_weight is None:
                        self._flat_decode_failed = True
                        return
                    lw.ple_gate_wt = ple_gate_weight.data.t()
                    lw.ple_proj_wt = ple_proj_weight.data.t()
                    lw.post_ple_norm_weight = layer.post_per_layer_input_norm.weight.data
                else:
                    lw.ple_gate_wt = None
                    lw.ple_proj_wt = None
                    lw.post_ple_norm_weight = None
                lw.layer_scalar = layer.layer_scalar.data

                max_qkv = max(max_qkv, lw.qkv_size)
                max_q = max(max_q, lw.q_size)
                max_attn_heads = max(max_attn_heads, lw.num_q_heads)
                max_head_dim = max(max_head_dim, lw.head_dim)
                max_intermediate = max(max_intermediate, lw.intermediate_size)
                max_ple = max(max_ple, lw.ple_size)
                layer_weights.append(lw)

            self._flat_is_gemma4 = True
            self._flat_layer_weights = layer_weights
            self._flat_layer_types = [lw.layer_type for lw in layer_weights]
            self._flat_linear_attn_modules = [
                layer.linear_attn if layer.layer_type == 'linear_attention' else None
                for layer in self.layers
            ]
            self._flat_lm_head_wt = self.lm_head.weight.data.t()
            self._flat_lm_head_bias = self.lm_head.bias.data if self.lm_head.bias is not None else None
            self._flat_norm_weight = self.norm.weight.data
            self._flat_norm_eps = cfg.rms_norm_eps
            self._flat_norm_offset = cfg.norm_offset
            self._flat_has_output_gate = False
            self._flat_hidden_size = cfg.hidden_size
            self._flat_gemma4_max_qkv = max_qkv
            self._flat_gemma4_max_q = max_q
            self._flat_gemma4_max_attn_heads = max_attn_heads
            self._flat_gemma4_max_head_dim = max_head_dim
            self._flat_gemma4_max_intermediate = max_intermediate
            self._flat_gemma4_max_ple = max_ple
            self._gemma4_flat_fused_gateup_use_cache = {}
            self._gemma4_flat_deepfusion_use_cache = {}
            self._gemma4_flat_policy_fused_gateup_rows = policy_rows(
                self,
                "MEGAGEMM_GEMMA4_FORCE_FUSED_GATEUP_USE",
                "gemma4_bf16_fused_gateup_rows",
            )
            self._gemma4_flat_policy_deepfusion_rows = policy_rows(
                self,
                "MEGAGEMM_GEMMA4_FORCE_DEEPFUSION_USE",
                "gemma4_bf16_deepfusion_rows",
            )
            self._gemma4_flat_policy_cublas_gateup_rows = policy_rows(
                self,
                "MEGAGEMM_GEMMA4_FORCE_FUSED_GATEUP_USE",
                "gemma4_bf16_cublas_gateup_rows",
            )
            self._gemma4_flat_policy_cublas_down_rows = policy_rows(
                self,
                "MEGAGEMM_GEMMA4_FORCE_DEEPFUSION_USE",
                "gemma4_bf16_cublas_down_rows",
            )
            self._gemma4_flat_fused_gateup_runtime_disabled = False
            self._gemma4_flat_fused_gateup_hits = 0
            self._gemma4_flat_ple_conditioned_gelu_hits = 0
            self._gemma4_flat_ple_conditioned_gelu_runtime_disabled = False
            self._gemma4_flat_ple_conditioned_gelu_first_failure = ""
            self._gemma4_flat_deepfusion_hits = 0
            self._gemma4_mlp_fusion_debug_seen = set()
            self._flat_int8_inline = False
            self._flat_w8a16_ready = False
            if has_any_int8_linear and triton is not None:
                try:
                    max_hidden_k = max(cfg.hidden_size, max_q)
                    self._flat_int8_quant_block_h = triton.next_power_of_2(max_hidden_k)
                    self._flat_int8_quant_block_i = triton.next_power_of_2(max_intermediate)
                    self._flat_int8_inline = True
                except Exception:
                    self._flat_int8_inline = False
            self._flat_bufs_batch = -1
            self._flat_decode_ready = True
        except Exception as exc:
            self._flat_decode_failed = True
            self._flat_decode_failed_reason = (
                f"Gemma4 flat decode preparation failed: {type(exc).__name__}: {exc}"
            )

    def _prepare_flat_decode(self):
        """Pre-transpose weights and collect layer references for zero-overhead decode.
        Called lazily on first decode_step. All errors are caught — falls back to normal path."""
        if self._flat_decode_failed or not _USE_FLAT_DECODE:
            if not _USE_FLAT_DECODE and not self._flat_decode_failed_reason:
                self._flat_decode_failed_reason = "disabled by MEGAGEMM_FLAT_DECODE=0"
            return
        if self.config.model_type == 'gemma4_text':
            if (
                any(getattr(layer, "is_moe_layer", False) for layer in self.layers)
                and qwen3_moe_grouped_decode is None
            ):
                self._flat_decode_failed = True
                self._flat_decode_failed_reason = (
                    "Gemma4 MoE flat decode needs grouped expert kernels"
                )
                return
            self._prepare_gemma4_flat_decode()
            return
        is_qwen3_moe = self.config.model_type == 'qwen3_moe'
        if is_qwen3_moe and qwen3_moe_grouped_decode is None:
            self._flat_decode_failed = True
            self._flat_decode_failed_reason = "Qwen3 MoE flat decode needs grouped expert kernels"
            return
        is_hybrid = not self._all_full_attention
        self._flat_is_hybrid = is_hybrid
        if not _HAS_FUSED_ADD_RMSNORM:
            self._flat_decode_failed = True
            self._flat_decode_failed_reason = "missing fused_add_rmsnorm kernel"
            return
        # Need fused_rope_kv_write and paged_attention_decode
        if not _HAS_FUSED_ROPE_ATTN:
            self._flat_decode_failed = True
            self._flat_decode_failed_reason = "missing fused_rope_attention decode kernels"
            return

        try:
            cfg = self.config
            layer_weights = []
            has_any_int8_linear = False

            def _flat_linear_params(linear):
                weight, bias = _linear_weight_bias(linear)
                if weight is not None:
                    return weight, bias, None
                if (
                    linear is not None
                    and isinstance(getattr(linear, "_mgx_sparse24_native_values", None), torch.Tensor)
                    and isinstance(getattr(linear, "_mgx_sparse24_native_meta", None), torch.Tensor)
                ):
                    return None, bias, linear
                if hasattr(linear, "weight_int8") and hasattr(linear, "scale"):
                    return None, getattr(linear, "bias", None), linear
                # AWQ QuantizedLinear: has qweight/scales/qzeros
                if hasattr(linear, "qweight") and hasattr(linear, "scales"):
                    return None, getattr(linear, "bias", None), linear
                return None, bias, None

            for layer in self.layers:
                attn = layer.self_attn
                mlp = layer.mlp
                is_linear_layer = layer.layer_type == 'linear_attention'
                is_moe_layer = bool(getattr(mlp, "is_moe", False))
                # AWQ with separate Q/K/V not supported in flat decode
                if not is_linear_layer and attn._awq_separate:
                    self._flat_decode_failed = True
                    self._flat_decode_failed_reason = "AWQ separate Q/K/V attention is unsupported by flat decode"
                    return
                if is_linear_layer:
                    qkv_weight = qkv_bias = qkv_mod = None
                    o_weight = o_bias = o_mod = None
                else:
                    qkv_weight, qkv_bias, qkv_mod = _flat_linear_params(attn.qkv_proj)
                    o_weight, o_bias, o_mod = _flat_linear_params(attn.o_proj)
                if is_moe_layer:
                    gate_up_weight = gate_up_bias = gate_up_mod = None
                    down_weight = down_bias = down_mod = None
                else:
                    gate_up_weight, gate_up_bias, gate_up_mod = _flat_linear_params(mlp.gate_up_proj)
                    down_weight, down_bias, down_mod = _flat_linear_params(mlp.down_proj)
                if (
                    (not is_linear_layer and (qkv_weight is None and qkv_mod is None))
                    or (not is_linear_layer and (o_weight is None and o_mod is None))
                    or (not is_moe_layer and (gate_up_weight is None and gate_up_mod is None))
                    or (not is_moe_layer and (down_weight is None and down_mod is None))
                ):
                    self._flat_decode_failed = True
                    self._flat_decode_failed_reason = "missing flat-decodable qkv/o/gate_up/down weights"
                    return
                has_any_int8_linear = has_any_int8_linear or any(
                    mod is not None
                    and (
                        (hasattr(mod, "weight_int8") and hasattr(mod, "scale"))
                        or (hasattr(mod, "qweight") and hasattr(mod, "scales"))
                    )
                    for mod in (qkv_mod, o_mod, gate_up_mod, down_mod)
                )
                lw = _FlatLayerWeights()
                lw.layer_type = layer.layer_type
                # Standard fused QKV projections always materialize V.  Gemma 4
                # has its own flat-weight structure and may explicitly alias V
                # from K, but this field must still exist on the common hot path.
                lw.v_from_k = False
                # Pre-transpose FP weights (zero copy). INT8 modules are called directly.
                lw.qkv_wt = qkv_weight.data.t() if qkv_weight is not None else None
                lw.qkv_bias = qkv_bias.data if qkv_bias is not None else None
                lw.o_wt = o_weight.data.t() if o_weight is not None else None
                lw.o_bias = o_bias.data if o_bias is not None else None
                lw.gate_up_wt = gate_up_weight.data.t() if gate_up_weight is not None else None
                lw.gate_up_weight = gate_up_weight.data if gate_up_weight is not None else None
                lw.gate_up_bias = gate_up_bias.data if gate_up_bias is not None else None
                lw.down_wt = down_weight.data.t() if down_weight is not None else None
                lw.down_weight = down_weight.data if down_weight is not None else None
                lw.down_bias = down_bias.data if down_bias is not None else None
                lw.qkv_mod = qkv_mod
                lw.o_mod = o_mod
                lw.gate_up_mod = gate_up_mod
                lw.down_mod = down_mod
                # INT8 inline: cache weight_int8 and scale refs directly
                # so the hot loop can call _flat_int8_linear() without Python dispatch.
                if qkv_mod is not None and hasattr(qkv_mod, 'weight_int8'):
                    lw.qkv_int8_w = qkv_mod.weight_int8.data.contiguous()
                    lw.qkv_int8_scale = qkv_mod.scale.data
                else:
                    lw.qkv_int8_w = None
                    lw.qkv_int8_scale = None
                lw.qkv_dequant_wt = None
                if o_mod is not None and hasattr(o_mod, 'weight_int8'):
                    lw.o_int8_w = o_mod.weight_int8.data.contiguous()
                    lw.o_int8_scale = o_mod.scale.data
                else:
                    lw.o_int8_w = None
                    lw.o_int8_scale = None
                lw.o_dequant_wt = None
                if gate_up_mod is not None and hasattr(gate_up_mod, 'weight_int8'):
                    lw.gate_up_int8_w = gate_up_mod.weight_int8.data.contiguous()
                    lw.gate_up_int8_scale = gate_up_mod.scale.data
                else:
                    lw.gate_up_int8_w = None
                    lw.gate_up_int8_scale = None
                lw.gate_up_dequant_wt = None
                if down_mod is not None and hasattr(down_mod, 'weight_int8'):
                    lw.down_int8_w = down_mod.weight_int8.data.contiguous()
                    lw.down_int8_scale = down_mod.scale.data
                else:
                    lw.down_int8_w = None
                    lw.down_int8_scale = None
                lw.down_dequant_wt = None
                lw.down_dequant_raw_wt = None
                lw.norm1_weight = layer.input_layernorm.weight.data
                lw.norm2_weight = layer.post_attention_layernorm.weight.data
                lw.q_norm_weight = (
                    attn.q_norm.weight.data
                    if not is_linear_layer and attn.q_norm is not None else None
                )
                lw.k_norm_weight = (
                    attn.k_norm.weight.data
                    if not is_linear_layer and attn.k_norm is not None else None
                )
                lw.norm_eps = cfg.rms_norm_eps
                # AWQ INT4 inline: cache qweight/scales/qzeros refs
                def _cache_awq(prefix, mod, lw_obj):
                    from megagemm.quantization.w4a16 import QuantizedLinear
                    if isinstance(mod, QuantizedLinear):
                        # Transpose in-place: [K, N//8] → [N//8, K] (no extra VRAM)
                        mod.transpose_for_decode()
                        setattr(lw_obj, f'{prefix}_awq_qw', mod.qweight.data)
                        setattr(lw_obj, f'{prefix}_awq_scales', mod.scales.data)
                        setattr(lw_obj, f'{prefix}_awq_qzeros', mod.qzeros.data)
                        setattr(lw_obj, f'{prefix}_awq_gs', mod.group_size)
                        return True
                    else:
                        setattr(lw_obj, f'{prefix}_awq_qw', None)
                        setattr(lw_obj, f'{prefix}_awq_scales', None)
                        setattr(lw_obj, f'{prefix}_awq_qzeros', None)
                        setattr(lw_obj, f'{prefix}_awq_gs', 0)
                        return False
                has_awq = False
                has_awq |= _cache_awq('qkv', qkv_mod, lw)
                has_awq |= _cache_awq('o', o_mod, lw)
                has_awq |= _cache_awq('gate_up', gate_up_mod, lw)
                has_awq |= _cache_awq('down', down_mod, lw)
                if has_awq:
                    has_any_int8_linear = True  # reuse flag to set up inline decode
                layer_weights.append(lw)

            self._flat_layer_weights = layer_weights
            self._flat_sparse24_ready = bool(
                callable(_flat_sparse24_mma_linear)
                and layer_weights
                and not is_hybrid
                and not is_qwen3_moe
                and all(
                    isinstance(getattr(mod, "_mgx_sparse24_native_values", None), torch.Tensor)
                    and isinstance(getattr(mod, "_mgx_sparse24_native_meta", None), torch.Tensor)
                    for lw in layer_weights
                    if lw.layer_type != "linear_attention"
                    for mod in (lw.qkv_mod, lw.o_mod, lw.gate_up_mod, lw.down_mod)
                )
            )
            self._flat_is_qwen3_moe = bool(is_qwen3_moe)
            self._flat_layer_types = [lw.layer_type for lw in layer_weights]
            self._flat_linear_attn_modules = [
                layer.linear_attn if layer.layer_type == 'linear_attention' else None
                for layer in self.layers
            ]
            self._flat_lm_head_wt = self.lm_head.weight.data.t()
            self._flat_lm_head_bias = self.lm_head.bias.data if self.lm_head.bias is not None else None
            self._flat_norm_weight = self.norm.weight.data

            # Model constants cached as locals
            attn0 = next(
                (layer.self_attn for layer in self.layers if layer.self_attn is not None),
                None,
            )
            if attn0 is None:
                self._flat_decode_failed = True
                self._flat_decode_failed_reason = "flat decode requires at least one full-attention layer"
                return
            self._flat_num_q_heads = cfg.num_attention_heads
            self._flat_num_kv_heads = cfg.num_key_value_heads
            self._flat_head_dim = cfg.head_dim
            self._flat_q_size = attn0._q_proj_size
            self._flat_k_size = attn0._k_size
            self._flat_v_size = attn0._v_size
            self._flat_hidden_size = cfg.hidden_size
            self._flat_hybrid_full_inline_enabled = bool(
                is_hybrid
                and _QWEN35_FLAT_HYBRID_FULL_INLINE_MAX_HIDDEN > 0
                and int(cfg.hidden_size) <= _QWEN35_FLAT_HYBRID_FULL_INLINE_MAX_HIDDEN
            )
            self._flat_intermediate_size = cfg.intermediate_size
            self._flat_scale = attn0.scale
            self._flat_half_rotate = attn0.rope_half_rotate
            self._flat_rotary_dim = attn0.rotary_dim
            self._flat_norm_offset = cfg.norm_offset
            self._flat_norm_eps = cfg.rms_norm_eps
            self._flat_has_output_gate = attn0.attention_output_gate
            self._flat_n_layers = len(layer_weights)

            # Determine if fast_linear should be used for gate_up
            self._flat_use_fast_gate_up = bool(
                fast_linear is not None
                and _USE_FAST_GEMV
                and _FORCE_GATE_UP_FAST
                and not has_any_int8_linear
                and not is_qwen3_moe
            )
            self._flat_use_fast_down = bool(
                fast_linear is not None
                and _USE_FAST_GEMV
                and _USE_FLAT_FAST_DOWN
                and not has_any_int8_linear
                and not is_qwen3_moe
            )

            # Pre-compute inline Triton kernel params
            self._flat_inline_kernels = False
            if _HAS_INLINE_FUSED_NORM and _HAS_INLINE_SWIGLU and triton is not None:
                try:
                    N = cfg.hidden_size
                    fb = triton.next_power_of_2(N)
                    if fb <= 4096:
                        self._flat_fused_block = fb
                        self._flat_fused_warps = min(4, max(1, fb // 256))
                        isize = cfg.intermediate_size
                        self._flat_swiglu_block = min(
                            triton.next_power_of_2(isize), 1024
                        )
                        self._flat_inline_kernels = True
                except Exception:
                    pass

            # INT8 inline decode: pre-compute Triton BLOCK_K for fused quant
            self._flat_int8_inline = False
            if has_any_int8_linear and triton is not None:
                try:
                    max_k = max(cfg.hidden_size, cfg.intermediate_size)
                    self._flat_int8_quant_block_h = triton.next_power_of_2(cfg.hidden_size)
                    self._flat_int8_quant_block_i = triton.next_power_of_2(cfg.intermediate_size)
                    self._flat_int8_inline = True
                    # Pre-compute W8A16 GEMV grids (computed ONCE, reused every token)
                    if _HAS_FLAT_W8A16_GEMV and _flat_w8a16_grid is not None:
                        qkv_n = attn0._q_proj_size + attn0._k_size + attn0._v_size
                        o_n = cfg.hidden_size
                        gu_n = 2 * cfg.intermediate_size
                        dn_n = cfg.hidden_size
                        self._flat_w8a16_grid_qkv = _flat_w8a16_grid(qkv_n)
                        self._flat_w8a16_grid_o = _flat_w8a16_grid(o_n)
                        self._flat_w8a16_grid_gu = _flat_w8a16_grid(gu_n)
                        self._flat_w8a16_grid_dn = _flat_w8a16_grid(dn_n)
                        self._flat_w8a16_ready = True
                    else:
                        self._flat_w8a16_ready = False
                except Exception:
                    pass

            # AWQ INT4 inline decode: pre-compute W4A16 GEMV grids
            self._flat_w4a16_ready = False
            lw0 = layer_weights[0] if layer_weights else None
            if (lw0 is not None and lw0.qkv_awq_qw is not None
                    and _HAS_FLAT_W4A16_GEMV and _flat_w4a16_grid is not None):
                try:
                    qkv_n = attn0._q_proj_size + attn0._k_size + attn0._v_size
                    o_n = cfg.hidden_size
                    gu_n = 2 * cfg.intermediate_size
                    dn_n = cfg.hidden_size
                    self._flat_w4a16_grid_qkv = _flat_w4a16_grid(qkv_n)
                    self._flat_w4a16_grid_o = _flat_w4a16_grid(o_n)
                    self._flat_w4a16_grid_gu = _flat_w4a16_grid(gu_n)
                    self._flat_w4a16_grid_dn = _flat_w4a16_grid(dn_n)
                    self._flat_w4a16_ready = True
                    self._flat_int8_inline = True  # reuse flag for hot loop entry
                except Exception:
                    pass

            # Buffer batch tracking
            self._flat_bufs_batch = -1

            self._flat_decode_ready = True
        except Exception:
            self._flat_decode_failed = True
            if not self._flat_decode_failed_reason:
                import traceback
                self._flat_decode_failed_reason = traceback.format_exc(limit=1).strip()

    def _alloc_flat_bufs(self, batch_size: int, device, dtype):
        """Pre-allocate all decode buffers. Called once per batch-size change."""
        if self._flat_bufs_batch == batch_size:
            return
        if getattr(self, "_flat_is_gemma4", False):
            weights = self._flat_layer_weights
            self._gemma4_flat_qkv_bufs = [
                torch.empty(batch_size, lw.qkv_size, device=device, dtype=dtype)
                if lw.qkv_size > 0 else None
                for lw in weights
            ]
            self._gemma4_flat_q_bufs = [
                torch.empty(batch_size, lw.q_size, device=device, dtype=dtype)
                if lw.qkv_size <= 0 else None
                for lw in weights
            ]
            self._gemma4_flat_k_bufs = [
                torch.empty(batch_size, lw.k_size, device=device, dtype=dtype)
                if (lw.qkv_size <= 0 and not lw.is_kv_shared) else None
                for lw in weights
            ]
            self._gemma4_flat_v_bufs = [
                torch.empty(batch_size, lw.v_size, device=device, dtype=dtype)
                if (lw.qkv_size <= 0 and not lw.is_kv_shared and not lw.v_from_k) else None
                for lw in weights
            ]
            self._gemma4_flat_attn_bufs = [
                torch.empty(
                    batch_size, lw.num_q_heads, lw.head_dim,
                    device=device, dtype=dtype,
                )
                for lw in weights
            ]
            self._gemma4_flat_o_bufs = [
                torch.empty(batch_size, self._flat_hidden_size, device=device, dtype=dtype)
                for _ in weights
            ]
            self._gemma4_flat_gate_up_bufs = [
                torch.empty(
                    batch_size, 2 * lw.intermediate_size,
                    device=device, dtype=dtype,
                )
                for lw in weights
            ]
            self._gemma4_flat_down_bufs = [
                torch.empty(batch_size, self._flat_hidden_size, device=device, dtype=dtype)
                for _ in weights
            ]
            self._gemma4_flat_parallel_shared_norm_bufs = [
                torch.empty(batch_size, self._flat_hidden_size, device=device, dtype=dtype)
                for _ in weights
            ]
            moe_layers = [lw for lw in weights if lw.is_moe and lw.moe_module is not None]
            expert_dims = {
                int(lw.moe_module.experts.intermediate_dim) for lw in moe_layers
            }
            expert_intermediate = next(iter(expert_dims)) if len(expert_dims) == 1 else 0
            shared_intermediate = max(
                (int(lw.intermediate_size) for lw in weights), default=0
            )
            flat_device = torch.device(device)
            device_name = (
                torch.cuda.get_device_name(flat_device)
                if flat_device.type == "cuda"
                else str(flat_device)
            )
            all_layers_moe = len(moe_layers) == len(weights)
            self._gemma4_flat_parallel_moe_enabled = bool(
                all_layers_moe
                and _gemma4_a100_a4b_parallel_moe_shape(
                    self.config.model_type,
                    batch_size,
                    self._flat_hidden_size,
                    shared_intermediate,
                    expert_intermediate,
                    dtype,
                    device_name,
                )
            )
            self._gemma4_flat_fused_attn_moe_bridge_enabled = bool(
                self._gemma4_flat_parallel_moe_enabled
                and int(batch_size) == 16
                and not bool(self._flat_norm_offset)
                and _GEMMA4_FUSED_ATTN_MOE_BRIDGE_DECODE
                and callable(rmsnorm_triton_attn_residual_dual)
            )
            self._gemma4_flat_fused_attn_moe_router_bridge_enabled = bool(
                self._gemma4_flat_fused_attn_moe_bridge_enabled
                and _GEMMA4_FUSED_ATTN_MOE_ROUTER_BRIDGE_DECODE
                and callable(rmsnorm_triton_attn_residual_router_bridge)
                and all(
                    float(lw.moe_module.gate.input_norm.eps)
                    == float(self._flat_norm_eps)
                    for lw in moe_layers
                )
            )
            self._gemma4_flat_fused_attn_moe_router_single_kernel_enabled = bool(
                self._gemma4_flat_fused_attn_moe_router_bridge_enabled
                and _GEMMA4_FUSED_ATTN_MOE_ROUTER_SINGLE_KERNEL_DECODE
                and callable(
                    rmsnorm_triton_attn_residual_router_bridge_single
                )
            )
            self._gemma4_flat_fused_router_compact_pack_enabled = bool(
                self._gemma4_flat_parallel_moe_enabled
                and int(batch_size) == 16
                and int(expert_intermediate) == 704
                and _GEMMA4_FUSED_ROUTER_COMPACT_PACK_DECODE
                and callable(qwen3_moe_topk_softmax_compact_pack)
                and all(
                    int(lw.moe_module.gate.num_experts) == 128
                    and int(lw.moe_module.gate.top_k) == 8
                    for lw in moe_layers
                )
            )
            self._gemma4_flat_attn_post_norm_bufs = (
                [
                    torch.empty(
                        batch_size,
                        self._flat_hidden_size,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in weights
                ]
                if self._gemma4_flat_fused_attn_moe_bridge_enabled
                else None
            )
            self._gemma4_flat_shared_input_bufs = (
                [
                    torch.empty(
                        batch_size,
                        self._flat_hidden_size,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in weights
                ]
                if self._gemma4_flat_fused_attn_moe_bridge_enabled
                else None
            )
            self._gemma4_flat_fused_post_moe_norm_residual_enabled = bool(
                self._gemma4_flat_parallel_moe_enabled
                and int(batch_size) == 16
                and _GEMMA4_FUSED_POST_MOE_NORM_RESIDUAL_DECODE
                and callable(rmsnorm_triton_pair_add_final_residual)
            )
            self._gemma4_flat_fused_expert_reduce_post_moe_enabled = bool(
                self._gemma4_flat_parallel_moe_enabled
                and int(batch_size) == 16
                and _GEMMA4_FUSED_EXPERT_REDUCE_POST_MOE_DECODE
                and not self._gemma4_flat_fused_post_moe_norm_residual_enabled
            )
            self._gemma4_flat_post_moe_out_bufs = (
                [
                    torch.empty(
                        batch_size,
                        self._flat_hidden_size,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in weights
                ]
                if self._gemma4_flat_fused_expert_reduce_post_moe_enabled
                else None
            )
            self._gemma4_flat_fused_next_attn_norm_supported = bool(
                self._gemma4_flat_fused_expert_reduce_post_moe_enabled
                and int(batch_size) == 16
                and not bool(self._flat_norm_offset)
                and all(
                    lw.is_moe
                    and int(lw.ple_size) == 0
                    and int(lw.layer_scalar.numel()) == 1
                    for lw in weights
                )
            )
            fused_next_attn_norm_requested = _env_enabled(
                "MEGAGEMM_GEMMA4_FUSED_NEXT_ATTN_NORM_DECODE",
                default=_GEMMA4_FUSED_NEXT_ATTN_NORM_DECODE,
            )
            self._gemma4_flat_fused_next_attn_norm_enabled = bool(
                self._gemma4_flat_fused_next_attn_norm_supported
                and fused_next_attn_norm_requested
            )
            self._gemma4_flat_next_attn_norm_bufs = (
                [
                    torch.empty(
                        batch_size,
                        self._flat_hidden_size,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in weights
                ]
                if self._gemma4_flat_fused_next_attn_norm_supported
                else None
            )
            dense_post_norm_chain_requested = policy_bool(
                self,
                "MEGAGEMM_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE",
                "gemma4_dense_post_norm_chain",
                default=_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE,
            )
            self._gemma4_flat_dense_post_norm_chain_enabled = bool(
                dense_post_norm_chain_requested
                and callable(rmsnorm_triton_residual_scale_next)
                and self.runtime_policy.name == "gemma4-e2b-l4"
                and all(
                    not lw.is_moe and int(lw.layer_scalar.numel()) == 1
                    for lw in weights
                )
            )
            dense_attn_mlp_bridge_requested = policy_bool(
                self,
                "MEGAGEMM_GEMMA4_DENSE_ATTN_MLP_BRIDGE_DECODE",
                "gemma4_e2b_h512_dense_bridge_pair",
                default=_GEMMA4_DENSE_ATTN_MLP_BRIDGE_DECODE,
            )
            dense_attn_mlp_bridge_b1_experiment = _env_enabled(
                "MEGAGEMM_GEMMA4_E2B_L4_B1_DECODE_FRONTIER_EXPERIMENT",
                default=False,
            )
            dense_attn_mlp_bridge_b1_requested = policy_bool(
                self,
                "MEGAGEMM_GEMMA4_E2B_L4_B1_DENSE_ATTN_MLP_BRIDGE_DECODE",
                "gemma4_e2b_b1_dense_bridge",
                default=False,
            )
            self._gemma4_flat_dense_attn_mlp_bridge_enabled = bool(
                dense_attn_mlp_bridge_requested
                and callable(rmsnorm_triton_attn_residual_dense)
                and self.runtime_policy.name == "gemma4-e2b-l4"
                and (
                    int(batch_size) == 8
                    or (
                        int(batch_size) == 1
                        and (
                            dense_attn_mlp_bridge_b1_requested
                            or dense_attn_mlp_bridge_b1_experiment
                        )
                    )
                )
                and dtype == torch.bfloat16
                and not bool(self._flat_norm_offset)
                and all(not lw.is_moe for lw in weights)
            )
            self._gemma4_flat_dense_attn_mlp_input_bufs = (
                [
                    torch.empty(
                        batch_size,
                        self._flat_hidden_size,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in weights
                ]
                if self._gemma4_flat_dense_attn_mlp_bridge_enabled
                else None
            )
            cublaslt_gateup_algorithms = {
                12288: _env_int(
                    "MEGAGEMM_GEMMA4_E2B_CUBLASLT_GATEUP_N12288_ALGO",
                    -1,
                ),
                24576: _env_int(
                    "MEGAGEMM_GEMMA4_E2B_CUBLASLT_GATEUP_N24576_ALGO",
                    -1,
                ),
            }
            self._gemma4_flat_cublaslt_gateup_enabled = bool(
                _GEMMA4_E2B_CUBLASLT_GATEUP_DECODE
                and HAS_CUBLASLT_BF16_LINEAR
                and callable(cublaslt_bf16_linear_cuda)
                and self.runtime_policy.name == "gemma4-e2b-l4"
                and int(batch_size) == 8
                and dtype == torch.bfloat16
                and all(
                    int(lw.gate_up_weight.shape[0])
                    in cublaslt_gateup_algorithms
                    and cublaslt_gateup_algorithms[
                        int(lw.gate_up_weight.shape[0])
                    ]
                    >= 0
                    for lw in weights
                    if lw.gate_up_weight is not None
                )
            )
            self._gemma4_flat_cublaslt_gateup_algorithms = (
                dict(cublaslt_gateup_algorithms)
                if self._gemma4_flat_cublaslt_gateup_enabled
                else {}
            )
            ple_conditioned_gelu_requested = policy_bool(
                self,
                "MEGAGEMM_GEMMA4_PLE_CONDITIONED_GELU_DECODE",
                "gemma4_ple_conditioned_gelu_decode",
                default=_GEMMA4_PLE_CONDITIONED_GELU_DECODE,
            )
            self._gemma4_flat_ple_conditioned_gelu_enabled = bool(
                ple_conditioned_gelu_requested
                and callable(conditioned_gelu_tanh_forward)
                and self.runtime_policy.name == "gemma4-e2b-l4"
                and int(batch_size) == 8
                and dtype == torch.bfloat16
                and int(self.hidden_size_per_layer_input) == 256
                and all(int(lw.ple_size) == 256 for lw in weights)
            )
            self._gemma4_flat_ple_conditioned_gelu_block_size = _env_int(
                "MEGAGEMM_GEMMA4_PLE_CONDITIONED_GELU_BLOCK_SIZE",
                256,
            )
            b8_large_mlp_supported = bool(
                self.runtime_policy.name == "gemma4-e2b-l4"
                and int(batch_size) == 8
                and dtype == torch.bfloat16
                and int(self._flat_hidden_size) == 1536
                and sum(
                    not lw.is_moe and int(lw.intermediate_size) == 12288
                    for lw in weights
                )
                == 20
            )
            self._gemma4_flat_b8_gated_activation_enabled = bool(
                b8_large_mlp_supported
                and callable(gated_activation_forward)
                and policy_bool(
                    self,
                    "MEGAGEMM_GEMMA4_E2B_B8_GATED_ACTIVATION_DECODE",
                    "gemma4_e2b_b8_gated_activation_decode",
                    default=False,
                )
            )
            self._gemma4_flat_b8_gated_activation_block_size = _env_int(
                "MEGAGEMM_GEMMA4_E2B_B8_GATED_ACTIVATION_BLOCK_SIZE",
                512,
            )
            self._gemma4_flat_b8_tensorcore_down_enabled = bool(
                b8_large_mlp_supported
                and callable(gemma4_e2b_b8_geglu_down_tensorcore)
                and policy_bool(
                    self,
                    "MEGAGEMM_GEMMA4_E2B_B8_TENSORCORE_DOWN_DECODE",
                    "gemma4_e2b_b8_tensorcore_down_decode",
                    default=False,
                )
            )
            self._gemma4_flat_b8_tensorcore_down_config = (
                _env_int("MEGAGEMM_GEMMA4_E2B_B8_TC_DOWN_BLOCK_N", 64),
                _env_int("MEGAGEMM_GEMMA4_E2B_B8_TC_DOWN_BLOCK_K", 32),
                _env_int("MEGAGEMM_GEMMA4_E2B_B8_TC_DOWN_WARPS", 4),
                _env_int("MEGAGEMM_GEMMA4_E2B_B8_TC_DOWN_STAGES", 2),
            )
            self._gemma4_flat_dense_next_attn_norm_bufs = (
                [
                    torch.empty(
                        batch_size,
                        self._flat_hidden_size,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in weights
                ]
                if self._gemma4_flat_dense_post_norm_chain_enabled
                else None
            )
            self._gemma4_flat_fused_router_expert_input_norm_enabled = bool(
                self._gemma4_flat_parallel_moe_enabled
                and int(batch_size) == 16
                and _GEMMA4_FUSED_ROUTER_EXPERT_INPUT_NORM_DECODE
                and callable(rmsnorm_triton_weighted_scaled_no_weight_dual)
            )
            self._gemma4_flat_expert_input_bufs = (
                [
                    torch.empty(
                        batch_size,
                        self._flat_hidden_size,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in weights
                ]
                if (
                    self._gemma4_flat_fused_router_expert_input_norm_enabled
                    or self._gemma4_flat_fused_attn_moe_bridge_enabled
                )
                else None
            )
            self._gemma4_flat_router_input_bufs = (
                [
                    torch.empty(
                        batch_size,
                        self._flat_hidden_size,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in weights
                ]
                if (
                    self._gemma4_flat_fused_router_expert_input_norm_enabled
                    or self._gemma4_flat_fused_attn_moe_router_bridge_enabled
                )
                else None
            )
            self._gemma4_flat_parallel_moe_policy = {
                "requested": bool(_GEMMA4_PARALLEL_MOE_DECODE),
                "model_type": str(self.config.model_type),
                "rows": int(batch_size),
                "hidden_dim": int(self._flat_hidden_size),
                "shared_intermediate": int(shared_intermediate),
                "expert_intermediate": int(expert_intermediate),
                "dtype": str(dtype),
                "device_name": str(device_name),
                "moe_layers": int(len(moe_layers)),
                "total_layers": int(len(weights)),
                "all_layers_moe": bool(all_layers_moe),
                "isolated_shared_norm_buffers": bool(
                    self._gemma4_flat_parallel_shared_norm_bufs is not None
                    and len(self._gemma4_flat_parallel_shared_norm_bufs) == len(weights)
                ),
                "fork_before_router": bool(
                    self._gemma4_flat_parallel_moe_enabled
                ),
                "enabled": bool(self._gemma4_flat_parallel_moe_enabled),
                "fused_attn_moe_bridge_requested": bool(
                    _GEMMA4_FUSED_ATTN_MOE_BRIDGE_DECODE
                ),
                "fused_attn_moe_bridge_enabled": bool(
                    self._gemma4_flat_fused_attn_moe_bridge_enabled
                ),
                "fused_attn_moe_router_bridge_requested": bool(
                    _GEMMA4_FUSED_ATTN_MOE_ROUTER_BRIDGE_DECODE
                ),
                "fused_attn_moe_router_bridge_enabled": bool(
                    self._gemma4_flat_fused_attn_moe_router_bridge_enabled
                ),
                "fused_attn_moe_router_single_kernel_requested": bool(
                    _GEMMA4_FUSED_ATTN_MOE_ROUTER_SINGLE_KERNEL_DECODE
                ),
                "fused_attn_moe_router_single_kernel_enabled": bool(
                    self._gemma4_flat_fused_attn_moe_router_single_kernel_enabled
                ),
                "fused_router_compact_pack_requested": bool(
                    _GEMMA4_FUSED_ROUTER_COMPACT_PACK_DECODE
                ),
                "fused_router_compact_pack_enabled": bool(
                    self._gemma4_flat_fused_router_compact_pack_enabled
                ),
                "isolated_attn_moe_bridge_buffers": bool(
                    self._gemma4_flat_attn_post_norm_bufs is not None
                    and self._gemma4_flat_shared_input_bufs is not None
                    and self._gemma4_flat_expert_input_bufs is not None
                    and len(self._gemma4_flat_attn_post_norm_bufs) == len(weights)
                    and len(self._gemma4_flat_shared_input_bufs) == len(weights)
                    and len(self._gemma4_flat_expert_input_bufs) == len(weights)
                ),
                "isolated_attn_moe_router_buffers": bool(
                    self._gemma4_flat_router_input_bufs is not None
                    and len(self._gemma4_flat_router_input_bufs) == len(weights)
                ),
                "fused_post_moe_norm_residual_requested": bool(
                    _GEMMA4_FUSED_POST_MOE_NORM_RESIDUAL_DECODE
                ),
                "fused_post_moe_norm_residual_enabled": bool(
                    self._gemma4_flat_fused_post_moe_norm_residual_enabled
                ),
                "fused_expert_reduce_post_moe_requested": bool(
                    _GEMMA4_FUSED_EXPERT_REDUCE_POST_MOE_DECODE
                ),
                "fused_expert_reduce_post_moe_enabled": bool(
                    self._gemma4_flat_fused_expert_reduce_post_moe_enabled
                ),
                "isolated_post_moe_output_buffers": bool(
                    self._gemma4_flat_post_moe_out_bufs is not None
                    and len(self._gemma4_flat_post_moe_out_bufs) == len(weights)
                ),
                "fused_next_attn_norm_requested": bool(
                    fused_next_attn_norm_requested
                ),
                "fused_next_attn_norm_supported": bool(
                    self._gemma4_flat_fused_next_attn_norm_supported
                ),
                "fused_next_attn_norm_enabled": bool(
                    self._gemma4_flat_fused_next_attn_norm_enabled
                ),
                "isolated_next_attn_norm_buffers": bool(
                    self._gemma4_flat_next_attn_norm_bufs is not None
                    and len(self._gemma4_flat_next_attn_norm_bufs) == len(weights)
                ),
                "fused_router_expert_input_norm_requested": bool(
                    _GEMMA4_FUSED_ROUTER_EXPERT_INPUT_NORM_DECODE
                ),
                "fused_router_expert_input_norm_enabled": bool(
                    self._gemma4_flat_fused_router_expert_input_norm_enabled
                ),
            }
            self._gemma4_flat_parallel_moe_hits = 0
            self._gemma4_flat_fused_attn_moe_bridge_hits = 0
            self._gemma4_flat_fused_attn_moe_router_bridge_hits = 0
            self._gemma4_flat_fused_attn_moe_router_single_kernel_hits = 0
            self._gemma4_flat_fused_router_compact_pack_hits = 0
            self._gemma4_flat_fused_post_moe_norm_residual_hits = 0
            self._gemma4_flat_fused_expert_reduce_post_moe_hits = 0
            self._gemma4_flat_fused_next_attn_norm_hits = 0
            self._gemma4_flat_fused_layer_scalar_hits = 0
            self._gemma4_flat_dense_post_norm_chain_hits = 0
            self._gemma4_flat_dense_next_attn_norm_hits = 0
            self._gemma4_flat_dense_attn_mlp_bridge_hits = 0
            self._gemma4_flat_dense_attn_mlp_bridge_runtime_disabled = False
            self._gemma4_flat_dense_attn_mlp_bridge_failure = ""
            self._gemma4_flat_cublaslt_gateup_hits = 0
            self._gemma4_flat_cublaslt_gateup_runtime_disabled = False
            self._gemma4_flat_cublaslt_gateup_failure = ""
            self._gemma4_flat_b8_gated_activation_hits = 0
            self._gemma4_flat_b8_gated_activation_runtime_disabled = False
            self._gemma4_flat_b8_gated_activation_failure = ""
            self._gemma4_flat_b8_tensorcore_down_hits = 0
            self._gemma4_flat_b8_tensorcore_down_runtime_disabled = False
            self._gemma4_flat_b8_tensorcore_down_failure = ""
            self._gemma4_flat_fused_router_expert_input_norm_hits = 0
            if self._gemma4_flat_parallel_moe_enabled:
                self._gemma4_flat_parallel_moe_stream = torch.cuda.Stream(device=device)
                self._gemma4_flat_parallel_moe_fork_events = [
                    torch.cuda.Event() for _ in weights
                ]
                self._gemma4_flat_parallel_moe_join_events = [
                    torch.cuda.Event() for _ in weights
                ]
            else:
                self._gemma4_flat_parallel_moe_stream = None
                self._gemma4_flat_parallel_moe_fork_events = None
                self._gemma4_flat_parallel_moe_join_events = None
            self._gemma4_flat_ple_gate_bufs = [
                torch.empty(batch_size, lw.ple_size, device=device, dtype=dtype)
                if lw.ple_size > 0 else None
                for lw in weights
            ]
            self._gemma4_flat_ple_proj_bufs = [
                torch.empty(batch_size, self._flat_hidden_size, device=device, dtype=dtype)
                if lw.ple_size > 0 else None
                for lw in weights
            ]
            self._gemma4_flat_b8_gated_activation_bufs = (
                [
                    torch.empty(
                        batch_size,
                        lw.intermediate_size,
                        device=device,
                        dtype=dtype,
                    )
                    if not lw.is_moe and int(lw.intermediate_size) == 12288
                    else None
                    for lw in weights
                ]
                if b8_large_mlp_supported
                else None
            )
            if getattr(self, '_flat_int8_inline', False):
                max_k = max(
                    self._flat_hidden_size,
                    self._flat_gemma4_max_q,
                    self._flat_gemma4_max_intermediate,
                )
                self._flat_int8_x_buf = torch.empty(batch_size, max_k, dtype=torch.int8, device=device)
                self._flat_int8_scale_buf = torch.empty(batch_size, 1, dtype=torch.float32, device=device)
            self._flat_bufs_batch = batch_size
            return
        H = self._flat_hidden_size
        I = self._flat_intermediate_size
        qkv_dim = self._flat_q_size + self._flat_k_size + self._flat_v_size
        self._flat_qkv_buf = torch.empty(batch_size, qkv_dim, device=device, dtype=dtype)
        self._flat_attn_buf = torch.empty(
            batch_size, self._flat_num_q_heads, self._flat_head_dim,
            device=device, dtype=dtype,
        )
        self._flat_o_buf = torch.empty(batch_size, H, device=device, dtype=dtype)
        self._flat_gate_up_buf = torch.empty(batch_size, 2 * I, device=device, dtype=dtype)
        self._flat_down_buf = torch.empty(batch_size, H, device=device, dtype=dtype)
        # Inline kernel output buffers (eliminate ~84 torch.empty per token)
        if getattr(self, '_flat_inline_kernels', False):
            self._flat_hidden_add_buf = torch.empty(batch_size, H, device=device, dtype=dtype)
            self._flat_normed_add_buf = torch.empty(batch_size, H, device=device, dtype=dtype)
            self._flat_swiglu_buf = torch.empty(batch_size, I, device=device, dtype=dtype)
        # INT8 inline decode: allocate dynamic quantization buffers
        if getattr(self, '_flat_int8_inline', False):
            max_k = max(H, I)
            self._flat_int8_x_buf = torch.empty(batch_size, max_k, dtype=torch.int8, device=device)
            self._flat_int8_scale_buf = torch.empty(batch_size, 1, dtype=torch.float32, device=device)
        # Direct GEMV kernels launch over (N tiles, M rows). The row count depends
        # on the active batch, so refresh these grids whenever flat buffers resize.
        if getattr(self, '_flat_w8a16_ready', False) and _flat_w8a16_grid is not None:
            self._flat_w8a16_grid_qkv = _flat_w8a16_grid(qkv_dim, batch_size)
            self._flat_w8a16_grid_o = _flat_w8a16_grid(H, batch_size)
            self._flat_w8a16_grid_gu = _flat_w8a16_grid(2 * I, batch_size)
            self._flat_w8a16_grid_dn = _flat_w8a16_grid(H, batch_size)
        if getattr(self, '_flat_w4a16_ready', False) and _flat_w4a16_grid is not None:
            self._flat_w4a16_grid_qkv = _flat_w4a16_grid(qkv_dim, batch_size)
            self._flat_w4a16_grid_o = _flat_w4a16_grid(H, batch_size)
            self._flat_w4a16_grid_gu = _flat_w4a16_grid(2 * I, batch_size)
            self._flat_w4a16_grid_dn = _flat_w4a16_grid(H, batch_size)
        if (
            _USE_FLAT_BATCH_FP16_DEQUANT
            and batch_size >= _FLAT_BATCH_FP16_DEQUANT_MIN_BATCH
            and dtype in (torch.float16, torch.bfloat16)
            and getattr(self, '_flat_layer_weights', None) is not None
        ):
            global _FLAT_FP16_DEQUANT_LOGGED
            def _dequant_weight(weight_int8, scale):
                if weight_int8 is None or scale is None:
                    return None
                return weight_int8.to(dtype) * scale.unsqueeze(1).to(dtype)

            def _dequant_weight_t(weight_int8, scale):
                weight = _dequant_weight(weight_int8, scale)
                if weight is None:
                    return None
                return weight.t().contiguous()

            try:
                budget_bytes = _FLAT_BATCH_FP16_DEQUANT_MAX_MB * 1024 * 1024
                planned_bytes = 0

                def _fits_budget(weight_int8) -> bool:
                    nonlocal planned_bytes
                    if weight_int8 is None or budget_bytes <= 0:
                        return True
                    extra_bytes = int(weight_int8.numel()) * torch.tensor([], dtype=dtype).element_size()
                    if planned_bytes + extra_bytes > budget_bytes:
                        return False
                    planned_bytes += extra_bytes
                    return True

                for lw in self._flat_layer_weights:
                    if (
                        "qkv" in _FLAT_BATCH_FP16_DEQUANT_OPS
                        and lw.qkv_dequant_wt is None
                        and _fits_budget(lw.qkv_int8_w)
                    ):
                        lw.qkv_dequant_wt = _dequant_weight_t(lw.qkv_int8_w, lw.qkv_int8_scale)
                    if (
                        "o" in _FLAT_BATCH_FP16_DEQUANT_OPS
                        and lw.o_dequant_wt is None
                        and _fits_budget(lw.o_int8_w)
                    ):
                        lw.o_dequant_wt = _dequant_weight_t(lw.o_int8_w, lw.o_int8_scale)
                    if (
                        "gate_up" in _FLAT_BATCH_FP16_DEQUANT_OPS
                        and lw.gate_up_dequant_wt is None
                        and _fits_budget(lw.gate_up_int8_w)
                    ):
                        lw.gate_up_dequant_wt = _dequant_weight_t(
                            lw.gate_up_int8_w, lw.gate_up_int8_scale
                        )
                    if (
                        "down" in _FLAT_BATCH_FP16_DEQUANT_OPS
                        and lw.down_dequant_wt is None
                        and _fits_budget(lw.down_int8_w)
                    ):
                        lw.down_dequant_wt = _dequant_weight_t(lw.down_int8_w, lw.down_int8_scale)
                    if (
                        "down_fused" in _FLAT_BATCH_FP16_DEQUANT_OPS
                        and lw.down_dequant_raw_wt is None
                        and _fits_budget(lw.down_int8_w)
                    ):
                        lw.down_dequant_raw_wt = _dequant_weight(
                            lw.down_int8_w, lw.down_int8_scale
                        )
                if _FLAT_FP16_DEQUANT_LOG and not _FLAT_FP16_DEQUANT_LOGGED:
                    _FLAT_FP16_DEQUANT_LOGGED = True
                    cached = sum(
                        1
                        for lw in self._flat_layer_weights
                        if lw.down_dequant_raw_wt is not None
                    )
                    print(
                        "[MegaGemm] flat FP16 dequant cache active "
                        f"ops={sorted(_FLAT_BATCH_FP16_DEQUANT_OPS)} "
                        f"batch={batch_size} max_mb={_FLAT_BATCH_FP16_DEQUANT_MAX_MB or 'unlimited'} "
                        f"planned_mb={planned_bytes / (1024 ** 2):.0f} "
                        f"down_fused_layers={cached}"
                    )
            except torch.cuda.OutOfMemoryError:
                for lw in self._flat_layer_weights:
                    lw.qkv_dequant_wt = None
                    lw.o_dequant_wt = None
                    lw.gate_up_dequant_wt = None
                    lw.down_dequant_wt = None
                    lw.down_dequant_raw_wt = None
                torch.cuda.empty_cache()
                print("[MegaGemm] flat FP16 dequant cache OOM; falling back to W8A16 direct decode")
        self._flat_bufs_batch = batch_size

    def _gemma4_flat_baseline_down(self, gate_up: torch.Tensor, lw: _Gemma4FlatLayerWeights, layer_idx: int) -> torch.Tensor:
        gate = gate_up[:, :lw.intermediate_size]
        value = gate_up[:, lw.intermediate_size:]
        activated = torch.nn.functional.gelu(gate, approximate='tanh')
        activated.mul_(value)
        if lw.down_wt is None:
            if getattr(self, '_flat_int8_inline', False) and lw.down_int8_w is not None:
                return _flat_int8_linear(
                    activated,
                    lw.down_int8_w,
                    lw.down_int8_scale,
                    lw.down_bias,
                    self._flat_int8_x_buf,
                    self._flat_int8_scale_buf,
                    self._flat_int8_quant_block_i,
                    out=self._gemma4_flat_down_bufs[layer_idx],
                )
            if lw.down_mod is not None:
                return lw.down_mod(activated)
            raise RuntimeError("Gemma4 flat decode is missing down projection weights")
        return self._flat_fp_linear(
            activated,
            lw.down_wt,
            lw.down_bias,
            self._gemma4_flat_down_bufs[layer_idx],
        )

    def _gemma4_flat_rmsnorm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        offset: bool = False,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return _decode_rmsnorm(
            x,
            weight,
            eps,
            offset,
            out=out,
            prefer_triton=bool(
                getattr(self, "_gemma4_prefer_triton_rmsnorm", False)
            ),
        )

    def _gemma4_flat_should_use_fused_gateup(
        self,
        hidden: torch.Tensor,
        lw: _Gemma4FlatLayerWeights,
        layer_idx: int,
    ) -> bool:
        if not _GEMMA4_FUSED_GATEUP_DECODE:
            _gemma4_log_mlp_fusion(self, "gateup", "disabled by MEGAGEMM_GEMMA4_FUSED_GATEUP_DECODE=0")
            return False
        if lw.gate_up_weight is None or fused_rmsnorm_linear is None:
            _gemma4_log_mlp_fusion(self, "gateup", "unavailable (missing raw gate_up weight or fused_rmsnorm_linear)")
            return False
        if self._gemma4_flat_fused_gateup_runtime_disabled:
            return False
        if torch.is_grad_enabled() or not hidden.is_cuda:
            _gemma4_log_mlp_fusion(self, "gateup", "ineligible (requires CUDA inference mode)")
            return False
        out_features = 2 * lw.intermediate_size
        rows = int(hidden.shape[0]) if hidden.dim() == 2 else int(hidden.shape[0] * hidden.shape[1])
        b1_large_gateup_experiment = bool(
            getattr(
                self,
                "_gemma4_flat_b1_large_gateup_experiment",
                False,
            )
            and self.runtime_policy.name == "gemma4-e2b-l4"
            and hidden.dtype == torch.bfloat16
            and rows == 1
            and int(hidden.shape[-1]) == 1536
            and int(out_features) == 24576
        )
        policy_force_fused_gateup = bool(
            hidden.dtype == torch.bfloat16
            and rows
            in getattr(self, "_gemma4_flat_policy_fused_gateup_rows", ())
        )
        policy_cublas_gateup = bool(
            hidden.dtype == torch.bfloat16
            and rows
            in getattr(self, "_gemma4_flat_policy_cublas_gateup_rows", ())
        )
        if policy_cublas_gateup:
            _gemma4_log_mlp_fusion(
                self,
                "gateup",
                f"measured cuBLAS policy retained for BF16 rows={rows}",
            )
            return False
        force_fused_gateup = bool(
            _GEMMA4_FORCE_FUSED_GATEUP_USE
            or policy_force_fused_gateup
            or b1_large_gateup_experiment
        )
        max_rows_override = rows if force_fused_gateup else None
        tuned_a4b = _gemma4_a100_a4b_tuned_mlp_shape(
            rows,
            int(hidden.shape[-1]),
            int(lw.intermediate_size),
            hidden.dtype,
            torch.cuda.get_device_name(hidden.device),
        )
        if tuned_a4b:
            _gemma4_log_mlp_fusion(
                self,
                "gateup",
                "A100/A4B tuned policy: RMSNorm + cuBLAS gate_up retained",
            )
            return False
        if (
            not force_fused_gateup
            and callable(fused_rmsnorm_linear_prefers_triton_shape)
            and not fused_rmsnorm_linear_prefers_triton_shape(
                int(hidden.shape[-1]),
                int(out_features),
                rows,
                mode="decode",
                max_rows_override=max_rows_override,
            )
        ):
            _gemma4_log_mlp_fusion(
                self,
                "gateup",
                f"shape rejected rows={rows} in={int(hidden.shape[-1])} out={int(out_features)}",
            )
            return False

        key = (
            int(hidden.shape[0]),
            int(hidden.shape[-1]),
            int(out_features),
            hidden.dtype,
            hidden.device.type,
            hidden.device.index,
            force_fused_gateup,
            policy_force_fused_gateup,
            b1_large_gateup_experiment,
        )
        cache = self._gemma4_flat_fused_gateup_use_cache
        if key not in cache:
            use = True
            try:
                out = self._gemma4_flat_gate_up_bufs[layer_idx]
                fused_rmsnorm_linear(
                    hidden,
                    lw.pre_ff_norm_weight,
                    self._flat_norm_eps,
                    lw.gate_up_weight,
                    lw.gate_up_bias,
                    norm_offset=False,
                    out=out,
                    max_rows_override=max_rows_override,
                )
                self._flat_fp_linear(
                    self._gemma4_flat_rmsnorm(hidden, lw.pre_ff_norm_weight, self._flat_norm_eps, False),
                    lw.gate_up_wt,
                    lw.gate_up_bias,
                    out,
                )
                torch.cuda.synchronize()
                fused_ms = _cuda_bench_ms(
                    lambda: fused_rmsnorm_linear(
                        hidden,
                        lw.pre_ff_norm_weight,
                        self._flat_norm_eps,
                        lw.gate_up_weight,
                        lw.gate_up_bias,
                        norm_offset=False,
                        out=out,
                        max_rows_override=max_rows_override,
                    ),
                    iters=8,
                )
                base_ms = _cuda_bench_ms(
                    lambda: self._flat_fp_linear(
                        self._gemma4_flat_rmsnorm(hidden, lw.pre_ff_norm_weight, self._flat_norm_eps, False),
                        lw.gate_up_wt,
                        lw.gate_up_bias,
                        out,
                    ),
                    iters=8,
                )
                if force_fused_gateup:
                    use = True
                    _gemma4_log_mlp_fusion(
                        self,
                        "gateup",
                        "force-use enabled "
                        f"source={'policy' if policy_force_fused_gateup else 'env'} "
                        f"fused_ms={fused_ms:.3f} base_ms={base_ms:.3f}",
                    )
                else:
                    use = bool(fused_ms <= (base_ms * (1.0 - _FUSED_RMSNORM_GATEUP_MIN_GAIN)))
                    _gemma4_log_mlp_fusion(
                        self,
                        "gateup",
                        f"bench fused_ms={fused_ms:.3f} base_ms={base_ms:.3f} use={int(use)} "
                        f"threshold={1.0 - _FUSED_RMSNORM_GATEUP_MIN_GAIN:.3f}",
                    )
            except Exception as exc:
                _gemma4_log_mlp_fusion(self, "gateup", f"exception -> fallback: {exc!r}")
                use = False
            cache[key] = use
        elif _GEMMA4_MLP_FUSION_DEBUG:
            _gemma4_log_mlp_fusion(self, "gateup", f"cached use={int(bool(cache[key]))} key={key[:3]}")
        return bool(cache[key])

    def _gemma4_flat_should_use_deepfusion(
        self,
        gate_up: torch.Tensor,
        lw: _Gemma4FlatLayerWeights,
        layer_idx: int,
    ) -> bool:
        if not _GEMMA4_DEEPFUSION_MLP_DECODE:
            _gemma4_log_mlp_fusion(self, "deepfusion", "disabled by MEGAGEMM_GEMMA4_DEEPFUSION_MLP_DECODE=0")
            return False
        if lw.down_weight is None or deepfusion_swiglu_down is None:
            _gemma4_log_mlp_fusion(self, "deepfusion", "unavailable (missing raw down weight or deepfusion kernel)")
            return False
        if torch.is_grad_enabled() or not gate_up.is_cuda:
            _gemma4_log_mlp_fusion(self, "deepfusion", "ineligible (requires CUDA inference mode)")
            return False
        rows = int(gate_up.shape[0]) if gate_up.dim() == 2 else int(gate_up.shape[0] * gate_up.shape[1])
        policy_force_deepfusion = bool(
            gate_up.dtype == torch.bfloat16
            and rows
            in getattr(self, "_gemma4_flat_policy_deepfusion_rows", ())
        )
        policy_cublas_down = bool(
            gate_up.dtype == torch.bfloat16
            and rows
            in getattr(self, "_gemma4_flat_policy_cublas_down_rows", ())
        )
        if policy_cublas_down:
            _gemma4_log_mlp_fusion(
                self,
                "deepfusion",
                f"measured cuBLAS policy retained for BF16 rows={rows}",
            )
            return False
        i_dim = int(gate_up.shape[-1] // 2)
        h_dim = int(lw.down_weight.shape[0])
        b1_large_down_experiment = bool(
            getattr(
                self,
                "_gemma4_flat_b1_large_down_experiment",
                False,
            )
            and self.runtime_policy.name == "gemma4-e2b-l4"
            and gate_up.dtype == torch.bfloat16
            and rows == 1
            and i_dim == 12288
            and h_dim == 1536
        )
        force_deepfusion = bool(
            _GEMMA4_FORCE_DEEPFUSION_USE
            or policy_force_deepfusion
            or b1_large_down_experiment
        )
        tuned_a4b = _gemma4_a100_a4b_tuned_mlp_shape(
            rows,
            h_dim,
            i_dim,
            gate_up.dtype,
            torch.cuda.get_device_name(gate_up.device),
        )
        if tuned_a4b and not force_deepfusion:
            _gemma4_log_mlp_fusion(
                self,
                "deepfusion",
                "A100/A4B tuned policy: GELU+cuBLAS down retained",
            )
            return False
        if (
            not force_deepfusion
            and callable(deepfusion_mlp_prefers_triton_shape)
            and not deepfusion_mlp_prefers_triton_shape(i_dim, h_dim, rows)
        ):
            _gemma4_log_mlp_fusion(
                self,
                "deepfusion",
                f"shape rejected rows={rows} i={i_dim} h={h_dim}",
            )
            return False

        key = (
            int(gate_up.shape[0]),
            int(gate_up.shape[-1]),
            int(lw.down_weight.shape[0]),
            int(lw.down_weight.shape[1]),
            gate_up.dtype,
            gate_up.device.type,
            gate_up.device.index,
            "gelu_tanh",
            force_deepfusion,
            policy_force_deepfusion,
            b1_large_down_experiment,
        )
        cache = self._gemma4_flat_deepfusion_use_cache
        if key not in cache:
            use = True
            try:
                out = self._gemma4_flat_down_bufs[layer_idx]
                deepfusion_swiglu_down(
                    gate_up,
                    lw.down_weight,
                    lw.down_bias,
                    out=out,
                    activation="gelu_tanh",
                )
                self._gemma4_flat_baseline_down(gate_up, lw, layer_idx)
                torch.cuda.synchronize()
                deep_ms = _cuda_bench_ms(
                    lambda: deepfusion_swiglu_down(
                        gate_up,
                        lw.down_weight,
                        lw.down_bias,
                        out=out,
                        activation="gelu_tanh",
                    ),
                    iters=8,
                )
                base_ms = _cuda_bench_ms(
                    lambda: self._gemma4_flat_baseline_down(gate_up, lw, layer_idx),
                    iters=8,
                )
                if force_deepfusion:
                    use = True
                    _gemma4_log_mlp_fusion(
                        self,
                        "deepfusion",
                        "force-use enabled "
                        f"source={'policy' if policy_force_deepfusion else 'env'} "
                        f"deep_ms={deep_ms:.3f} base_ms={base_ms:.3f}",
                    )
                else:
                    use = bool(deep_ms <= (base_ms * (1.0 - _DEEPFUSION_MLP_MIN_GAIN)))
                    _gemma4_log_mlp_fusion(
                        self,
                        "deepfusion",
                        f"bench deep_ms={deep_ms:.3f} base_ms={base_ms:.3f} use={int(use)} "
                        f"threshold={1.0 - _DEEPFUSION_MLP_MIN_GAIN:.3f}",
                    )
            except Exception as exc:
                _gemma4_log_mlp_fusion(self, "deepfusion", f"exception -> fallback: {exc!r}")
                use = False
            cache[key] = use
        elif _GEMMA4_MLP_FUSION_DEBUG:
            _gemma4_log_mlp_fusion(self, "deepfusion", f"cached use={int(bool(cache[key]))} key={key[:4]}")
        return bool(cache[key])

    @torch.inference_mode()
    def _prepare_gemma4_b1_mlp_gemv_routes(self):
        """Resolve exact-shape routes once, before the measured decode loop."""
        from ..kernels.gemma4_b1_mlp_gemv import B1MlpGemvPlan

        self._gemma4_b1_mlp_prepared_routes = None
        configs = self._gemma4_b1_mlp_gemv_configs
        if not configs:
            return
        if self.runtime_policy.name != "gemma4-e2b-l4":
            raise ValueError("prepared B1 GEMV routes require the E2B/L4 policy")
        large = [(i, lw) for i, lw in enumerate(self._flat_layer_weights)
                 if not lw.is_moe and int(lw.intermediate_size) == 12288]
        if len(large) != 20:
            raise ValueError(f"prepared B1 GEMV requires 20 large layers, found {len(large)}")
        plans = getattr(self, "_gemma4_b1_mlp_gemv_plans", {})
        self._gemma4_b1_mlp_gemv_plans = plans
        hits = getattr(self, "_gemma4_b1_mlp_gemv_hits", {})
        self._gemma4_b1_mlp_gemv_hits = hits
        prepared_hits = getattr(self, "_gemma4_b1_mlp_prepared_hits", {})
        self._gemma4_b1_mlp_prepared_hits = prepared_hits
        failures = self._gemma4_b1_mlp_gemv_failures

        def audited(launch, operation):
            def run(x):
                if operation in failures:
                    return None
                try:
                    result = launch(x)
                except Exception as exc:
                    failures[operation] = f"{type(exc).__name__}: {exc}"
                    return None
                hits[operation] = hits.get(operation, 0) + 1
                prepared_hits[operation] = prepared_hits.get(operation, 0) + 1
                return result
            return run

        routes = [(None, None) for _ in self._flat_layer_weights]
        device = None
        for index, lw in large:
            bound = []
            for operation in ("gateup", "down"):
                config = configs.get(operation)
                if not config:
                    bound.append(None)
                    continue
                weight = lw.gate_up_weight if operation == "gateup" else lw.down_weight
                bias = lw.gate_up_bias if operation == "gateup" else lw.down_bias
                out = (self._gemma4_flat_gate_up_bufs if operation == "gateup"
                       else self._gemma4_flat_down_bufs)[index]
                if device is not None and weight.device != device:
                    raise ValueError("prepared GEMV weights must share one CUDA device")
                device = weight.device
                key = (index, operation, config, id(weight), id(bias))
                plan = plans.get(key)
                if plan is None:
                    plan = plans[key] = B1MlpGemvPlan(weight, bias, operation, config)
                bound.append(audited(plan.bind_output(out), operation))
            routes[index] = tuple(bound)
        self._gemma4_b1_mlp_prepared_weights = self._flat_layer_weights
        self._gemma4_b1_mlp_prepared_buffers = (
            self._gemma4_flat_gate_up_bufs, self._gemma4_flat_down_bufs,
        )
        self._gemma4_b1_mlp_prepared_device = device
        self._gemma4_b1_mlp_prepared_routes = routes

    def _gemma4_b1_mlp_routes_for_decode(self, hidden):
        """One boundary guard per token, outside the 35-layer loop."""
        routes = self._gemma4_b1_mlp_prepared_routes
        buffers = self._gemma4_b1_mlp_prepared_buffers
        if (tuple(hidden.shape) != (1, 1536) or hidden.dtype != torch.bfloat16
                or hidden.device != self._gemma4_b1_mlp_prepared_device
                or not hidden.is_cuda or torch.is_grad_enabled()
                or torch.cuda.current_device() != hidden.device.index
                or self._flat_layer_weights is not self._gemma4_b1_mlp_prepared_weights
                or self._gemma4_flat_gate_up_bufs is not buffers[0]
                or self._gemma4_flat_down_bufs is not buffers[1]):
            self._gemma4_b1_mlp_gemv_failures["boundary"] = "prepared GEMV shape/device/buffers changed"
            return None
        return routes

    def _gemma4_b1_mlp_gemv_enabled(self, hidden, lw, operation):
        configs = getattr(self, "_gemma4_b1_mlp_gemv_configs", {})
        return bool(
            configs.get(operation)
            and self.runtime_policy.name == "gemma4-e2b-l4"
            and hidden.is_cuda and hidden.dtype == torch.bfloat16
            and not torch.is_grad_enabled()
            and hidden.dim() == 2 and int(hidden.shape[0]) == 1
            and int(hidden.shape[-1]) == 1536
            and not lw.is_moe and int(lw.intermediate_size) == 12288
        )

    def _gemma4_b1_mlp_gemv(self, x, lw, layer_idx, operation, out):
        """Explicit experimental route; failures remain visible to the gate."""
        failures = getattr(self, "_gemma4_b1_mlp_gemv_failures", None)
        if failures is None:
            failures = self._gemma4_b1_mlp_gemv_failures = {}
        if operation in failures:
            return None
        try:
            config = self._gemma4_b1_mlp_gemv_configs[operation]
            weight = lw.gate_up_weight if operation == "gateup" else lw.down_weight
            bias = lw.gate_up_bias if operation == "gateup" else lw.down_bias
            plans = getattr(self, "_gemma4_b1_mlp_gemv_plans", None)
            if plans is None:
                plans = self._gemma4_b1_mlp_gemv_plans = {}
            key = (layer_idx, operation, config, id(weight), id(bias))
            plan = plans.get(key)
            if plan is None:
                from ..kernels.gemma4_b1_mlp_gemv import B1MlpGemvPlan

                plan = plans[key] = B1MlpGemvPlan(weight, bias, operation, config)
            result = plan(x, out)
        except Exception as exc:
            failures[operation] = f"{type(exc).__name__}: {exc}"
            return None
        hits = getattr(self, "_gemma4_b1_mlp_gemv_hits", None)
        if hits is None:
            hits = self._gemma4_b1_mlp_gemv_hits = {}
        hits[operation] = hits.get(operation, 0) + 1
        return result

    def _gemma4_flat_shared_mlp_decode(
        self,
        hidden: torch.Tensor,
        lw: _Gemma4FlatLayerWeights,
        layer_idx: int,
        timing_events: Optional[dict],
        *,
        normalized_input: Optional[torch.Tensor] = None,
        gemv_routes: Optional[tuple] = None,
    ) -> torch.Tensor:
        """Run the dense shared FFN branch on the current CUDA stream."""
        norm_eps = self._flat_norm_eps
        int8_inline = getattr(self, '_flat_int8_inline', False)
        b1_gemv_gateup = b1_gemv_down = False
        if gemv_routes is not None:
            b1_gemv_gateup, b1_gemv_down = gemv_routes
        elif (getattr(self, "_gemma4_b1_mlp_gemv_configs", None)
              and getattr(self, "_gemma4_b1_mlp_gemv_dispatch", "legacy") == "legacy"):
            b1_gemv_gateup = self._gemma4_b1_mlp_gemv_enabled(hidden, lw, "gateup")
            b1_gemv_down = self._gemma4_b1_mlp_gemv_enabled(hidden, lw, "down")
        mlp_gate_up_start_end = _timing_record_start(timing_events is not None)
        use_fused_gateup = bool(
            not b1_gemv_gateup and normalized_input is None
            and self._gemma4_flat_should_use_fused_gateup(hidden, lw, layer_idx)
        )
        if use_fused_gateup:
            try:
                rows = (
                    int(hidden.shape[0])
                    if hidden.dim() == 2
                    else int(hidden.shape[0] * hidden.shape[1])
                )
                force_fused_gateup = bool(
                    _GEMMA4_FORCE_FUSED_GATEUP_USE
                    or (
                        hidden.dtype == torch.bfloat16
                        and rows
                        in getattr(
                            self,
                            "_gemma4_flat_policy_fused_gateup_rows",
                            (),
                        )
                    )
                    or (
                        getattr(
                            self,
                            "_gemma4_flat_b1_large_gateup_experiment",
                            False,
                        )
                        and self.runtime_policy.name == "gemma4-e2b-l4"
                        and hidden.dtype == torch.bfloat16
                        and rows == 1
                        and int(hidden.shape[-1]) == 1536
                        and int(2 * lw.intermediate_size) == 24576
                    )
                )
                gate_up = fused_rmsnorm_linear(
                    hidden,
                    lw.pre_ff_norm_weight,
                    norm_eps,
                    lw.gate_up_weight,
                    lw.gate_up_bias,
                    norm_offset=False,
                    out=self._gemma4_flat_gate_up_bufs[layer_idx],
                    max_rows_override=(rows if force_fused_gateup else None),
                )
            except Exception as exc:
                self._gemma4_flat_fused_gateup_use_cache.clear()
                self._gemma4_flat_fused_gateup_runtime_disabled = True
                _gemma4_log_mlp_fusion(
                    self, "gateup", f"runtime exception -> fallback: {exc!r}"
                )
                use_fused_gateup = False
            else:
                self._gemma4_flat_fused_gateup_hits += 1
                if (
                    getattr(
                        self,
                        "_gemma4_flat_b1_large_gateup_experiment",
                        False,
                    )
                    and self.runtime_policy.name == "gemma4-e2b-l4"
                    and hidden.dtype == torch.bfloat16
                    and rows == 1
                    and int(hidden.shape[-1]) == 1536
                    and int(2 * lw.intermediate_size) == 24576
                ):
                    self._gemma4_flat_b1_large_gateup_hits += 1
        if not use_fused_gateup:
            mlp_in = normalized_input
            if mlp_in is None:
                mlp_input_norm_start_end = _timing_record_start(
                    timing_events is not None
                )
                mlp_in = self._gemma4_flat_rmsnorm(
                    hidden,
                    lw.pre_ff_norm_weight,
                    norm_eps,
                    False,
                )
                _timing_record_end(
                    timing_events,
                    "mlp_input_norm",
                    mlp_input_norm_start_end,
                )
            use_cublaslt_gateup = bool(
                self._gemma4_flat_cublaslt_gateup_enabled
                and not self._gemma4_flat_cublaslt_gateup_runtime_disabled
                and lw.gate_up_weight is not None
                and int(lw.gate_up_weight.shape[0])
                in self._gemma4_flat_cublaslt_gateup_algorithms
            )
            gate_up = None
            if gemv_routes is not None and b1_gemv_gateup:
                gate_up = b1_gemv_gateup(mlp_in)
            elif b1_gemv_gateup:
                gate_up = self._gemma4_b1_mlp_gemv(
                    mlp_in, lw, layer_idx, "gateup",
                    self._gemma4_flat_gate_up_bufs[layer_idx],
                )
            if use_cublaslt_gateup and gate_up is None:
                try:
                    gate_up = cublaslt_bf16_linear_cuda(
                        mlp_in,
                        lw.gate_up_weight,
                        lw.gate_up_bias,
                        out=self._gemma4_flat_gate_up_bufs[layer_idx],
                        algorithm_index=(
                            self._gemma4_flat_cublaslt_gateup_algorithms[
                                int(lw.gate_up_weight.shape[0])
                            ]
                        ),
                    )
                except Exception as exc:
                    self._gemma4_flat_cublaslt_gateup_runtime_disabled = True
                    if not self._gemma4_flat_cublaslt_gateup_failure:
                        self._gemma4_flat_cublaslt_gateup_failure = (
                            f"{type(exc).__name__}: {exc}"
                        )
                    gate_up = None
                else:
                    self._gemma4_flat_cublaslt_gateup_hits += 1
            if gate_up is None:
                if lw.gate_up_wt is not None:
                    gate_up = self._flat_fp_linear(
                        mlp_in,
                        lw.gate_up_wt,
                        lw.gate_up_bias,
                        self._gemma4_flat_gate_up_bufs[layer_idx],
                    )
                elif int8_inline and lw.gate_up_int8_w is not None:
                    gate_up = _flat_int8_linear(
                        mlp_in,
                        lw.gate_up_int8_w,
                        lw.gate_up_int8_scale,
                        lw.gate_up_bias,
                        self._flat_int8_x_buf,
                        self._flat_int8_scale_buf,
                        self._flat_int8_quant_block_h,
                        out=self._gemma4_flat_gate_up_bufs[layer_idx],
                    )
                else:
                    gate_up = lw.gate_up_mod(mlp_in)
        _timing_record_end(timing_events, "mlp_gate_up", mlp_gate_up_start_end)

        mlp_down_start_end = _timing_record_start(timing_events is not None)
        use_tensorcore_down = bool(
            not b1_gemv_down
            and getattr(
                self,
                "_gemma4_flat_b8_tensorcore_down_enabled",
                False,
            )
            and not getattr(
                self,
                "_gemma4_flat_b8_tensorcore_down_runtime_disabled",
                False,
            )
            and callable(gemma4_e2b_b8_geglu_down_tensorcore)
            and gate_up.dtype == torch.bfloat16
            and tuple(gate_up.shape) == (8, 24576)
            and lw.down_weight is not None
            and tuple(lw.down_weight.shape) == (1536, 12288)
        )
        down_out = None
        if use_tensorcore_down:
            block_n, block_k, num_warps, num_stages = (
                self._gemma4_flat_b8_tensorcore_down_config
            )
            try:
                down_out = gemma4_e2b_b8_geglu_down_tensorcore(
                    gate_up,
                    lw.down_weight,
                    lw.down_bias,
                    out=self._gemma4_flat_down_bufs[layer_idx],
                    block_n=block_n,
                    block_k=block_k,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            except Exception as exc:
                self._gemma4_flat_b8_tensorcore_down_runtime_disabled = True
                if not self._gemma4_flat_b8_tensorcore_down_failure:
                    self._gemma4_flat_b8_tensorcore_down_failure = (
                        f"{type(exc).__name__}: {exc}"
                    )
                use_tensorcore_down = False
                down_out = None
            else:
                self._gemma4_flat_b8_tensorcore_down_hits += 1

        use_deepfusion = bool(
            not use_tensorcore_down
            and not b1_gemv_down
            and self._gemma4_flat_should_use_deepfusion(gate_up, lw, layer_idx)
        )
        if use_deepfusion:
            try:
                down_out = deepfusion_swiglu_down(
                    gate_up,
                    lw.down_weight,
                    lw.down_bias,
                    out=self._gemma4_flat_down_bufs[layer_idx],
                    activation="gelu_tanh",
                )
            except Exception as exc:
                self._gemma4_flat_deepfusion_use_cache.clear()
                _gemma4_log_mlp_fusion(
                    self, "deepfusion", f"runtime exception -> fallback: {exc!r}"
                )
                use_deepfusion = False
            else:
                self._gemma4_flat_deepfusion_hits += 1
                if (
                    getattr(
                        self,
                        "_gemma4_flat_b1_large_down_experiment",
                        False,
                    )
                    and self.runtime_policy.name == "gemma4-e2b-l4"
                    and gate_up.dtype == torch.bfloat16
                    and int(gate_up.shape[0]) == 1
                    and int(gate_up.shape[-1]) == 24576
                    and int(lw.down_weight.shape[0]) == 1536
                ):
                    self._gemma4_flat_b1_large_down_hits += 1
        if not use_tensorcore_down and not use_deepfusion:
            mlp_act_start_end = _timing_record_start(timing_events is not None)
            use_gated_activation = bool(
                not b1_gemv_down
                and getattr(
                    self,
                    "_gemma4_flat_b8_gated_activation_enabled",
                    False,
                )
                and not getattr(
                    self,
                    "_gemma4_flat_b8_gated_activation_runtime_disabled",
                    False,
                )
                and callable(gated_activation_forward)
                and gate_up.dtype == torch.bfloat16
                and tuple(gate_up.shape) == (8, 24576)
                and self._gemma4_flat_b8_gated_activation_bufs is not None
                and self._gemma4_flat_b8_gated_activation_bufs[layer_idx]
                is not None
            )
            activated = None
            if use_gated_activation:
                try:
                    activated = gated_activation_forward(
                        gate_up,
                        lw.intermediate_size,
                        activation="gelu_tanh",
                        out=self._gemma4_flat_b8_gated_activation_bufs[layer_idx],
                        block_size=(
                            self._gemma4_flat_b8_gated_activation_block_size
                        ),
                    )
                except Exception as exc:
                    self._gemma4_flat_b8_gated_activation_runtime_disabled = True
                    if not self._gemma4_flat_b8_gated_activation_failure:
                        self._gemma4_flat_b8_gated_activation_failure = (
                            f"{type(exc).__name__}: {exc}"
                        )
                    use_gated_activation = False
                    activated = None
                else:
                    self._gemma4_flat_b8_gated_activation_hits += 1
            if activated is None:
                gate = gate_up[:, :lw.intermediate_size]
                value = gate_up[:, lw.intermediate_size:]
                activated = torch.nn.functional.gelu(gate, approximate='tanh')
                activated.mul_(value)
            _timing_record_end(timing_events, "mlp_act", mlp_act_start_end)
            if gemv_routes is not None and b1_gemv_down:
                down_out = b1_gemv_down(activated)
            elif b1_gemv_down:
                down_out = self._gemma4_b1_mlp_gemv(
                    activated, lw, layer_idx, "down",
                    self._gemma4_flat_down_bufs[layer_idx],
                )
            if down_out is not None:
                pass
            elif lw.down_wt is not None:
                down_out = self._flat_fp_linear(
                    activated,
                    lw.down_wt,
                    lw.down_bias,
                    self._gemma4_flat_down_bufs[layer_idx],
                )
            elif int8_inline and lw.down_int8_w is not None:
                down_out = _flat_int8_linear(
                    activated,
                    lw.down_int8_w,
                    lw.down_int8_scale,
                    lw.down_bias,
                    self._flat_int8_x_buf,
                    self._flat_int8_scale_buf,
                    self._flat_int8_quant_block_i,
                    out=self._gemma4_flat_down_bufs[layer_idx],
                )
            else:
                down_out = lw.down_mod(activated)
        _timing_record_end(timing_events, "mlp_down", mlp_down_start_end)
        return down_out

    def _gemma4_flat_decode_layers(
        self,
        hidden: torch.Tensor,
        layer_kv_caches: list,
        block_table: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        phys_blocks: torch.Tensor,
        blk_offsets: torch.Tensor,
        pos_1d: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
        timing_events: Optional[dict] = None,
    ) -> torch.Tensor:
        bsz = hidden.shape[0]
        hidden = hidden.reshape(bsz, -1)
        prepared_gemv_routes = None
        if getattr(self, "_gemma4_b1_mlp_prepared_routes", None) is not None:
            prepared_gemv_routes = self._gemma4_b1_mlp_routes_for_decode(hidden)
        pos_ids = pos_1d.reshape(bsz, 1)
        norm_eps = self._flat_norm_eps
        norm_offset = self._flat_norm_offset
        paged_decode_splits = int(
            getattr(self.runtime_policy, "paged_decode_splits", 0) or 0
        )
        paged_decode_gqa2_direct = bool(
            getattr(self.runtime_policy, "paged_decode_gqa2_direct", False)
        )
        paged_decode_warps_h256 = int(
            getattr(self.runtime_policy, "paged_decode_warps_h256", 0) or 0
        )
        int8_inline = getattr(self, '_flat_int8_inline', False)
        if int8_inline:
            int8_x_buf = self._flat_int8_x_buf
            int8_s_buf = self._flat_int8_scale_buf
            int8_bk_h = self._flat_int8_quant_block_h
            int8_bk_i = self._flat_int8_quant_block_i
        fused_next_attn_norm_chain = bool(
            timing_events is None
            and self._gemma4_flat_fused_next_attn_norm_enabled
            and self._gemma4_flat_next_attn_norm_bufs is not None
        )
        dense_post_norm_chain = bool(
            self._gemma4_flat_dense_post_norm_chain_enabled
            and self._gemma4_flat_dense_next_attn_norm_bufs is not None
        )

        for layer_idx, lw in enumerate(self._flat_layer_weights):
            cos, sin = self._get_layer_rope(layer_idx)
            kv_cache = layer_kv_caches[layer_idx]

            residual = hidden
            attn_input_norm_start_end = _timing_record_start(timing_events is not None)
            if dense_post_norm_chain and layer_idx > 0:
                normed = self._gemma4_flat_dense_next_attn_norm_bufs[
                    layer_idx - 1
                ]
            elif fused_next_attn_norm_chain and layer_idx > 0:
                normed = self._gemma4_flat_next_attn_norm_bufs[layer_idx - 1]
            else:
                normed = self._gemma4_flat_rmsnorm(
                    hidden,
                    lw.input_norm_weight,
                    norm_eps,
                    norm_offset,
                )
            _timing_record_end(timing_events, "attn_input_norm", attn_input_norm_start_end)

            attn_qkv_start_end = _timing_record_start(timing_events is not None)
            if lw.qkv_wt is not None:
                qkv = self._flat_fp_linear(
                    normed,
                    lw.qkv_wt,
                    lw.qkv_bias,
                    self._gemma4_flat_qkv_bufs[layer_idx],
                )
                q_raw = qkv[:, :lw.q_size]
                k_raw = qkv[:, lw.q_size:lw.q_size + lw.k_size]
                v_raw = k_raw if lw.v_from_k else qkv[:, lw.q_size + lw.k_size:]
            else:
                if lw.q_wt is not None:
                    q_raw = self._flat_fp_linear(
                        normed,
                        lw.q_wt,
                        lw.q_bias,
                        self._gemma4_flat_q_bufs[layer_idx],
                    )
                elif int8_inline and lw.q_int8_w is not None:
                    q_raw = _flat_int8_linear(
                        normed,
                        lw.q_int8_w,
                        lw.q_int8_scale,
                        lw.q_bias,
                        int8_x_buf,
                        int8_s_buf,
                        int8_bk_h,
                        out=self._gemma4_flat_q_bufs[layer_idx],
                    )
                else:
                    q_raw = lw.q_mod(normed)
                k_raw = None
                v_raw = None
                if not lw.is_kv_shared:
                    if lw.k_wt is not None:
                        k_raw = self._flat_fp_linear(
                            normed,
                            lw.k_wt,
                            lw.k_bias,
                            self._gemma4_flat_k_bufs[layer_idx],
                        )
                    elif int8_inline and lw.k_int8_w is not None:
                        k_raw = _flat_int8_linear(
                            normed,
                            lw.k_int8_w,
                            lw.k_int8_scale,
                            lw.k_bias,
                            int8_x_buf,
                            int8_s_buf,
                            int8_bk_h,
                            out=self._gemma4_flat_k_bufs[layer_idx],
                        )
                    else:
                        k_raw = lw.k_mod(normed)
                    if lw.v_from_k:
                        v_raw = k_raw
                    elif lw.v_wt is not None:
                        v_raw = self._flat_fp_linear(
                            normed,
                            lw.v_wt,
                            lw.v_bias,
                            self._gemma4_flat_v_bufs[layer_idx],
                        )
                    elif int8_inline and lw.v_int8_w is not None:
                        v_raw = _flat_int8_linear(
                            normed,
                            lw.v_int8_w,
                            lw.v_int8_scale,
                            lw.v_bias,
                            int8_x_buf,
                            int8_s_buf,
                            int8_bk_h,
                            out=self._gemma4_flat_v_bufs[layer_idx],
                        )
                    else:
                        v_raw = lw.v_mod(normed)
            _timing_record_end(timing_events, "attn_qkv", attn_qkv_start_end)

            q = q_raw.view(bsz, lw.num_q_heads, lw.head_dim)
            can_use_fused_rope_attn = (
                _HAS_FUSED_ROPE_ATTN
                and hidden.is_cuda
                and lw.rotary_dim == lw.head_dim
            )
            fused_kv_write_ok = False

            if not lw.is_kv_shared and can_use_fused_rope_attn:
                attn_kv_write_start_end = _timing_record_start(timing_events is not None)
                k = k_raw.view(bsz, lw.num_kv_heads, lw.head_dim)
                v = v_raw.view(bsz, lw.num_kv_heads, lw.head_dim)
                fused_kv_write_ok = bool(
                    fused_rope_kv_write(
                        k,
                        v,
                        kv_cache,
                        cos,
                        sin,
                        pos_1d,
                        phys_blocks,
                        blk_offsets,
                        half_rotate=lw.half_rotate,
                        rotary_dim=lw.rotary_dim,
                        k_norm_weight=lw.k_norm_weight,
                        norm_eps=norm_eps,
                        v_norm=lw.has_v_norm,
                    )
                )
                _timing_record_end(timing_events, "attn_kv_write", attn_kv_write_start_end)

            if can_use_fused_rope_attn and (lw.is_kv_shared or fused_kv_write_ok):
                attn_core_start_end = _timing_record_start(timing_events is not None)
                attn = _triton_paged_decode_fused(
                    q,
                    kv_cache,
                    block_table,
                    seq_lens_kv,
                    lw.scale,
                    cos,
                    sin,
                    pos_1d,
                    half_rotate=lw.half_rotate,
                    rotary_dim=lw.rotary_dim,
                    q_norm_weight=lw.q_norm_weight,
                    norm_eps=norm_eps,
                    out=self._gemma4_flat_attn_bufs[layer_idx],
                    sliding_window=lw.sliding_window if lw.sliding_window > 0 else None,
                    split_policy_override=paged_decode_splits or None,
                    gqa2_direct_policy_enabled=paged_decode_gqa2_direct,
                    num_warps_policy_override=(
                        paged_decode_warps_h256
                        if int(lw.head_dim) == 256
                        else None
                    ),
                    e2b_l4_h512_policy_enabled=bool(
                        getattr(
                            self.runtime_policy,
                            "gemma4_e2b_h512_dense_bridge_pair",
                            False,
                        )
                        and self._gemma4_flat_dense_attn_mlp_bridge_enabled
                    ),
                )
                _timing_record_end(timing_events, "attn_core", attn_core_start_end)
                if timing_events is not None and attn_core_start_end is not None and attn_core_start_end[0] is not None:
                    timing_events.setdefault(
                        "attn_core_sliding" if lw.sliding_window > 0 else "attn_core_full",
                        [],
                    ).append(attn_core_start_end)
            else:
                attn_norm_rope_start_end = _timing_record_start(timing_events is not None)
                if lw.q_norm_weight is not None:
                    q = self._gemma4_flat_rmsnorm(q, lw.q_norm_weight, norm_eps, False)
                q4, _ = apply_rotary_emb(
                    q.unsqueeze(2),
                    q.unsqueeze(2),
                    cos,
                    sin,
                    position_ids=pos_ids,
                    half_rotate=lw.half_rotate,
                    rotary_dim=lw.rotary_dim,
                )
                q = q4.squeeze(2)
                _timing_record_end(timing_events, "attn_norm_rope", attn_norm_rope_start_end)

                if not lw.is_kv_shared:
                    attn_kv_write_start_end = _timing_record_start(timing_events is not None)
                    k = k_raw.view(bsz, lw.num_kv_heads, lw.head_dim)
                    v = v_raw.view(bsz, lw.num_kv_heads, lw.head_dim)
                    if lw.k_norm_weight is not None:
                        k = self._gemma4_flat_rmsnorm(k, lw.k_norm_weight, norm_eps, False)
                    if lw.has_v_norm:
                        v = _decode_rmsnorm_no_weight(v, norm_eps)
                    k4, _ = apply_rotary_emb(
                        k.unsqueeze(2),
                        k.unsqueeze(2),
                        cos,
                        sin,
                        position_ids=pos_ids,
                        half_rotate=lw.half_rotate,
                        rotary_dim=lw.rotary_dim,
                    )
                    k = k4.squeeze(2)
                    kv_cache[phys_blocks, 0, :, blk_offsets, :] = k
                    kv_cache[phys_blocks, 1, :, blk_offsets, :] = v
                    _timing_record_end(timing_events, "attn_kv_write", attn_kv_write_start_end)

                attn_core_start_end = _timing_record_start(timing_events is not None)
                attn = paged_attention_decode(
                    q,
                    kv_cache,
                    block_table,
                    seq_lens_kv,
                    lw.scale,
                    out=self._gemma4_flat_attn_bufs[layer_idx],
                    sliding_window=lw.sliding_window if lw.sliding_window > 0 else None,
                    split_policy_override=paged_decode_splits or None,
                )
                _timing_record_end(timing_events, "attn_core", attn_core_start_end)
                if timing_events is not None and attn_core_start_end is not None and attn_core_start_end[0] is not None:
                    timing_events.setdefault(
                        "attn_core_sliding" if lw.sliding_window > 0 else "attn_core_full",
                        [],
                    ).append(attn_core_start_end)
            attn_2d = attn.reshape(bsz, lw.num_q_heads * lw.head_dim)
            attn_o_proj_start_end = _timing_record_start(timing_events is not None)
            if lw.o_wt is not None:
                o_out = self._flat_fp_linear(
                    attn_2d,
                    lw.o_wt,
                    lw.o_bias,
                    self._gemma4_flat_o_bufs[layer_idx],
                )
            elif int8_inline and lw.o_int8_w is not None:
                o_out = _flat_int8_linear(
                    attn_2d,
                    lw.o_int8_w,
                    lw.o_int8_scale,
                    lw.o_bias,
                    int8_x_buf,
                    int8_s_buf,
                    int8_bk_h,
                    out=self._gemma4_flat_o_bufs[layer_idx],
                )
            else:
                o_out = lw.o_mod(attn_2d)
            _timing_record_end(timing_events, "attn_o_proj", attn_o_proj_start_end)
            bridge_shared_in = None
            bridge_expert_in = None
            bridge_router_in = None
            bridge_dense_in = None
            use_attn_moe_bridge = bool(
                timing_events is None
                and lw.is_moe
                and self._gemma4_flat_fused_attn_moe_bridge_enabled
            )
            if use_attn_moe_bridge:
                if self._gemma4_flat_fused_attn_moe_router_bridge_enabled:
                    gate = lw.moe_module.gate
                    router_scale = gate.scale.to(
                        device=residual.device,
                        dtype=residual.dtype,
                    ).reshape(-1)
                    if (
                        self._gemma4_flat_fused_attn_moe_router_single_kernel_enabled
                    ):
                        (
                            hidden,
                            bridge_shared_in,
                            bridge_expert_in,
                            bridge_router_in,
                        ) = rmsnorm_triton_attn_residual_router_bridge_single(
                            o_out,
                            residual,
                            lw.post_attn_norm_weight,
                            lw.pre_ff_norm_weight,
                            lw.pre_expert_norm_weight,
                            router_scale,
                            norm_eps,
                            gate.scalar_root,
                            out_hidden=residual,
                            post_norm_out=(
                                self._gemma4_flat_attn_post_norm_bufs[layer_idx]
                            ),
                            shared_out=(
                                self._gemma4_flat_shared_input_bufs[layer_idx]
                            ),
                            expert_out=(
                                self._gemma4_flat_expert_input_bufs[layer_idx]
                            ),
                            router_out=self._gemma4_flat_router_input_bufs[layer_idx],
                        )
                        self._gemma4_flat_fused_attn_moe_router_single_kernel_hits += 1
                    else:
                        (
                            hidden,
                            bridge_shared_in,
                            bridge_expert_in,
                            bridge_router_in,
                        ) = rmsnorm_triton_attn_residual_router_bridge(
                            o_out,
                            residual,
                            lw.post_attn_norm_weight,
                            lw.pre_ff_norm_weight,
                            lw.pre_expert_norm_weight,
                            router_scale,
                            norm_eps,
                            gate.scalar_root,
                            out_hidden=residual,
                            post_norm_out=(
                                self._gemma4_flat_attn_post_norm_bufs[layer_idx]
                            ),
                            shared_out=(
                                self._gemma4_flat_shared_input_bufs[layer_idx]
                            ),
                            expert_out=(
                                self._gemma4_flat_expert_input_bufs[layer_idx]
                            ),
                            router_out=self._gemma4_flat_router_input_bufs[layer_idx],
                        )
                    self._gemma4_flat_fused_attn_moe_router_bridge_hits += 1
                else:
                    (
                        hidden,
                        bridge_shared_in,
                        bridge_expert_in,
                    ) = rmsnorm_triton_attn_residual_dual(
                        o_out,
                        residual,
                        lw.post_attn_norm_weight,
                        lw.pre_ff_norm_weight,
                        lw.pre_expert_norm_weight,
                        norm_eps,
                        out_hidden=residual,
                        post_norm_out=(
                            self._gemma4_flat_attn_post_norm_bufs[layer_idx]
                        ),
                        shared_out=self._gemma4_flat_shared_input_bufs[layer_idx],
                        expert_out=self._gemma4_flat_expert_input_bufs[layer_idx],
                    )
                self._gemma4_flat_fused_attn_moe_bridge_hits += 1
            elif (
                not lw.is_moe
                and int(bsz) in (1, 8)
                and self._gemma4_flat_dense_attn_mlp_bridge_enabled
                and not self._gemma4_flat_dense_attn_mlp_bridge_runtime_disabled
            ):
                dense_bridge_start_end = _timing_record_start(
                    timing_events is not None
                )
                try:
                    hidden, bridge_dense_in = (
                        rmsnorm_triton_attn_residual_dense(
                            o_out,
                            residual,
                            lw.post_attn_norm_weight,
                            lw.pre_ff_norm_weight,
                            norm_eps,
                            norm_offset=norm_offset,
                            out_hidden=residual,
                            pre_ff_out=(
                                self._gemma4_flat_dense_attn_mlp_input_bufs[
                                    layer_idx
                                ]
                            ),
                        )
                    )
                except Exception as exc:
                    self._gemma4_flat_dense_attn_mlp_bridge_runtime_disabled = True
                    if not self._gemma4_flat_dense_attn_mlp_bridge_failure:
                        self._gemma4_flat_dense_attn_mlp_bridge_failure = (
                            f"{type(exc).__name__}: {exc}"
                        )
                else:
                    self._gemma4_flat_dense_attn_mlp_bridge_hits += 1
                if bridge_dense_in is None:
                    attn_normed = self._gemma4_flat_rmsnorm(
                        o_out,
                        lw.post_attn_norm_weight,
                        norm_eps,
                        norm_offset,
                    )
                    hidden = residual.add_(attn_normed)
                _timing_record_end(
                    timing_events,
                    "attn_mlp_bridge",
                    dense_bridge_start_end,
                )
            else:
                attn_output_norm_start_end = _timing_record_start(
                    timing_events is not None
                )
                attn_normed = self._gemma4_flat_rmsnorm(
                    o_out,
                    lw.post_attn_norm_weight,
                    norm_eps,
                    norm_offset,
                )
                _timing_record_end(
                    timing_events,
                    "attn_output_norm",
                    attn_output_norm_start_end,
                )
                hidden = residual.add_(attn_normed)

            residual = hidden
            selected_experts = None
            routing_weights = None
            parallel_moe = bool(
                lw.is_moe
                and self._gemma4_flat_parallel_moe_enabled
                and timing_events is None
            )
            fused_post_moe_norm_residual = bool(
                parallel_moe
                and self._gemma4_flat_fused_post_moe_norm_residual_enabled
            )
            fused_expert_reduce_post_moe = bool(
                parallel_moe
                and self._gemma4_flat_fused_expert_reduce_post_moe_enabled
            )
            fuse_next_attn_norm = bool(
                fused_expert_reduce_post_moe
                and fused_next_attn_norm_chain
            )
            write_next_attn_norm = bool(
                fuse_next_attn_norm
                and layer_idx + 1 < len(self._flat_layer_weights)
            )
            post_moe_chain_fused = bool(
                fused_post_moe_norm_residual or fused_expert_reduce_post_moe
            )
            fused_router_expert_input_norm = bool(
                parallel_moe
                and bridge_expert_in is None
                and self._gemma4_flat_fused_router_expert_input_norm_enabled
                and float(lw.moe_module.gate.input_norm.eps) == float(norm_eps)
            )
            expert_in = bridge_expert_in
            normalized_router_input = bridge_router_in
            compact_route_prepacked = False
            shared_normed = None
            parallel_main_stream = None
            parallel_join_event = None
            if parallel_moe:
                parallel_main_stream = torch.cuda.current_stream(hidden.device)
                side_stream = self._gemma4_flat_parallel_moe_stream
                fork_event = self._gemma4_flat_parallel_moe_fork_events[layer_idx]
                parallel_join_event = self._gemma4_flat_parallel_moe_join_events[layer_idx]
                fork_event.record(parallel_main_stream)
                with torch.cuda.stream(side_stream):
                    side_stream.wait_event(fork_event)
                    down_out = self._gemma4_flat_shared_mlp_decode(
                        hidden,
                        lw,
                        layer_idx,
                        None,
                        normalized_input=bridge_shared_in,
                    )
                    if not post_moe_chain_fused:
                        shared_normed = self._gemma4_flat_rmsnorm(
                            down_out,
                            lw.post_shared_norm_weight,
                            norm_eps,
                            False,
                            out=self._gemma4_flat_parallel_shared_norm_bufs[layer_idx],
                        )
                    parallel_join_event.record(side_stream)
            if lw.is_moe:
                moe_router_start_end = _timing_record_start(timing_events is not None)
                if fused_router_expert_input_norm:
                    gate = lw.moe_module.gate
                    router_scale = gate.scale.to(
                        device=residual.device,
                        dtype=residual.dtype,
                    ).reshape(-1)
                    expert_in, normalized_router_input = (
                        rmsnorm_triton_weighted_scaled_no_weight_dual(
                            residual,
                            lw.pre_expert_norm_weight,
                            router_scale,
                            norm_eps,
                            gate.scalar_root,
                            weighted_out=self._gemma4_flat_expert_input_bufs[layer_idx],
                            scaled_out=self._gemma4_flat_router_input_bufs[layer_idx],
                        )
                    )
                    self._gemma4_flat_fused_router_expert_input_norm_hits += 1
                compact_route_workspace = lw.moe_module.experts._grouped_decode_workspace
                use_compact_route_pack = bool(
                    self._gemma4_flat_fused_router_compact_pack_enabled
                    and timing_events is None
                    and not int(
                        compact_route_workspace.get(
                            "expert_grouped_compact_route_prepacked_disabled",
                            0,
                        )
                        or 0
                    )
                )
                routing_weights, selected_experts = lw.moe_module.gate.route(
                    residual,
                    is_prefill=False,
                    normalized_router_input=normalized_router_input,
                    compact_route_workspace=compact_route_workspace,
                    use_compact_route_pack=use_compact_route_pack,
                )
                compact_route_prepacked = bool(
                    lw.moe_module.gate._compact_route_pack_last_active
                )
                if compact_route_prepacked:
                    self._gemma4_flat_fused_router_compact_pack_hits += 1
                _timing_record_end(timing_events, "moe_router", moe_router_start_end)
            if not parallel_moe:
                down_out = self._gemma4_flat_shared_mlp_decode(
                    hidden,
                    lw,
                    layer_idx,
                    timing_events,
                    normalized_input=bridge_dense_in,
                    gemv_routes=(prepared_gemv_routes[layer_idx]
                                 if prepared_gemv_routes is not None else None),
                )
            if lw.is_moe:
                mlp_output_norm_start_end = _timing_record_start(timing_events is not None)
                if not parallel_moe:
                    shared_normed = self._gemma4_flat_rmsnorm(
                        down_out,
                        lw.post_shared_norm_weight,
                        norm_eps,
                        False,
                    )
                if expert_in is None:
                    expert_in = self._gemma4_flat_rmsnorm(
                        residual,
                        lw.pre_expert_norm_weight,
                        norm_eps,
                        False,
                    )
                _timing_record_end(timing_events, "mlp_output_norm", mlp_output_norm_start_end)
                moe_experts_start_end = _timing_record_start(timing_events is not None)
                expert_out = lw.moe_module.experts(
                    expert_in,
                    selected_experts,
                    routing_weights,
                    use_grouped_decode=True,
                    post_moe_shared=(down_out if fused_expert_reduce_post_moe else None),
                    post_moe_shared_weight=(
                        lw.post_shared_norm_weight
                        if fused_expert_reduce_post_moe
                        else None
                    ),
                    post_moe_expert_weight=(
                        lw.post_expert_norm_weight
                        if fused_expert_reduce_post_moe
                        else None
                    ),
                    post_moe_final_weight=(
                        lw.post_ff_norm_weight
                        if fused_expert_reduce_post_moe
                        else None
                    ),
                    post_moe_residual=(residual if fused_expert_reduce_post_moe else None),
                    post_moe_out=(
                        self._gemma4_flat_post_moe_out_bufs[layer_idx]
                        if fused_expert_reduce_post_moe
                        else None
                    ),
                    post_moe_wait_event=(
                        parallel_join_event if fused_expert_reduce_post_moe else None
                    ),
                    post_moe_layer_scalar=(
                        lw.layer_scalar if fuse_next_attn_norm else None
                    ),
                    post_moe_next_norm_weight=(
                        (
                            self._flat_layer_weights[layer_idx + 1].input_norm_weight
                            if write_next_attn_norm
                            else self._flat_norm_weight
                        )
                        if fuse_next_attn_norm
                        else None
                    ),
                    post_moe_next_norm_out=(
                        self._gemma4_flat_next_attn_norm_bufs[layer_idx]
                        if fuse_next_attn_norm
                        else None
                    ),
                    post_moe_write_next_norm=write_next_attn_norm,
                    post_moe_eps=norm_eps,
                    compact_route_prepacked=compact_route_prepacked,
                )
                _timing_record_end(timing_events, "moe_experts", moe_experts_start_end)
                mlp_output_norm_start_end = _timing_record_start(timing_events is not None)
                if parallel_moe and not fused_expert_reduce_post_moe:
                    parallel_main_stream.wait_event(parallel_join_event)
                if parallel_moe:
                    self._gemma4_flat_parallel_moe_hits += 1
                if fused_expert_reduce_post_moe:
                    hidden = expert_out
                    self._gemma4_flat_fused_expert_reduce_post_moe_hits += 1
                    if fuse_next_attn_norm:
                        self._gemma4_flat_fused_layer_scalar_hits += 1
                        if write_next_attn_norm:
                            self._gemma4_flat_fused_next_attn_norm_hits += 1
                elif fused_post_moe_norm_residual:
                    hidden = rmsnorm_triton_pair_add_final_residual(
                        down_out,
                        expert_out,
                        lw.post_shared_norm_weight,
                        lw.post_expert_norm_weight,
                        lw.post_ff_norm_weight,
                        residual,
                        norm_eps,
                        out=residual,
                    )
                    self._gemma4_flat_fused_post_moe_norm_residual_hits += 1
                else:
                    expert_normed = self._gemma4_flat_rmsnorm(
                        expert_out,
                        lw.post_expert_norm_weight,
                        norm_eps,
                        False,
                    )
                    shared_normed.add_(expert_normed)
                    down_normed = self._gemma4_flat_rmsnorm(
                        shared_normed,
                        lw.post_ff_norm_weight,
                        norm_eps,
                        False,
                    )
                _timing_record_end(timing_events, "mlp_output_norm", mlp_output_norm_start_end)
            else:
                mlp_output_norm_start_end = _timing_record_start(timing_events is not None)
                has_ple_tail = bool(
                    lw.ple_size > 0 and per_layer_inputs is not None
                )
                if dense_post_norm_chain and not has_ple_tail:
                    down_normed = None
                else:
                    down_normed = self._gemma4_flat_rmsnorm(
                        down_out,
                        lw.post_ff_norm_weight,
                        norm_eps,
                        norm_offset,
                    )
                _timing_record_end(timing_events, "mlp_output_norm", mlp_output_norm_start_end)
            if not post_moe_chain_fused and down_normed is not None:
                hidden = residual.add_(down_normed)

            has_ple_tail = bool(lw.ple_size > 0 and per_layer_inputs is not None)
            ple_proj = None
            if has_ple_tail:
                residual = hidden
                ple_start_end = _timing_record_start(timing_events is not None)
                ple = self._flat_fp_linear(
                    hidden,
                    lw.ple_gate_wt,
                    None,
                    self._gemma4_flat_ple_gate_bufs[layer_idx],
                )
                ple_condition = per_layer_inputs[:, 0, layer_idx, :]
                use_conditioned_gelu = bool(
                    getattr(
                        self,
                        "_gemma4_flat_ple_conditioned_gelu_enabled",
                        False,
                    )
                    and not getattr(
                        self,
                        "_gemma4_flat_ple_conditioned_gelu_runtime_disabled",
                        False,
                    )
                )
                if use_conditioned_gelu:
                    try:
                        ple = conditioned_gelu_tanh_forward(
                            ple,
                            ple_condition,
                            out=ple,
                            block_size=getattr(
                                self,
                                "_gemma4_flat_ple_conditioned_gelu_block_size",
                                256,
                            ),
                        )
                    except Exception as exc:
                        self._gemma4_flat_ple_conditioned_gelu_runtime_disabled = True
                        if not self._gemma4_flat_ple_conditioned_gelu_first_failure:
                            self._gemma4_flat_ple_conditioned_gelu_first_failure = (
                                f"{type(exc).__name__}: {exc}"
                            )
                        use_conditioned_gelu = False
                    else:
                        self._gemma4_flat_ple_conditioned_gelu_hits += 1
                if not use_conditioned_gelu:
                    ple = torch.nn.functional.gelu(ple, approximate='tanh')
                    ple.mul_(ple_condition)
                ple_proj = self._flat_fp_linear(
                    ple,
                    lw.ple_proj_wt,
                    None,
                    self._gemma4_flat_ple_proj_bufs[layer_idx],
                )
                if not dense_post_norm_chain:
                    ple_normed = self._gemma4_flat_rmsnorm(
                        ple_proj,
                        lw.post_ple_norm_weight,
                        norm_eps,
                        False,
                    )
                    hidden = residual.add_(ple_normed)
                _timing_record_end(timing_events, "ple", ple_start_end)

            if dense_post_norm_chain and not lw.is_moe:
                dense_post_norm_chain_start_end = _timing_record_start(
                    timing_events is not None
                )
                write_next_dense_norm = layer_idx + 1 < len(
                    self._flat_layer_weights
                )
                dense_branch = ple_proj if has_ple_tail else down_out
                dense_weight = (
                    lw.post_ple_norm_weight
                    if has_ple_tail
                    else lw.post_ff_norm_weight
                )
                hidden, _ = rmsnorm_triton_residual_scale_next(
                    dense_branch,
                    residual,
                    dense_weight,
                    lw.layer_scalar,
                    (
                        self._flat_layer_weights[layer_idx + 1].input_norm_weight
                        if write_next_dense_norm
                        else None
                    ),
                    norm_eps,
                    norm_offset=False if has_ple_tail else norm_offset,
                    next_norm_offset=norm_offset,
                    out_hidden=residual,
                    next_norm_out=(
                        self._gemma4_flat_dense_next_attn_norm_bufs[layer_idx]
                        if write_next_dense_norm
                        else None
                    ),
                )
                _timing_record_end(
                    timing_events,
                    "dense_post_norm_chain",
                    dense_post_norm_chain_start_end,
                )
                self._gemma4_flat_dense_post_norm_chain_hits += 1
                if write_next_dense_norm:
                    self._gemma4_flat_dense_next_attn_norm_hits += 1
            elif not fuse_next_attn_norm:
                hidden.mul_(lw.layer_scalar.to(device=hidden.device, dtype=hidden.dtype))

        return hidden.unsqueeze(1)

    def _hybrid_flat_decode_layers(
        self,
        hidden: torch.Tensor,
        layer_kv_caches: list,
        cos: torch.Tensor,
        sin: torch.Tensor,
        pos_1d: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: Optional[torch.Tensor],
        seq_lens_kv: torch.Tensor,
        phys_blocks: torch.Tensor,
        blk_offsets: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
        timing_events: Optional[dict] = None,
        linear_state_cache: Optional[dict] = None,
        block_manager: Optional[BlockManagerLike] = None,
        seq_ids: Optional[List[int]] = None,
        max_decode_blocks: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Hybrid Qwen3.5 flat decode.

        Full-attention layers use the minimal full-attn decode path, while
        linear-attention layers reuse the stable cached recurrent path.
        """
        bsz = hidden.shape[0]
        record_flat_timing = timing_events is not None
        positions = pos_1d.unsqueeze(-1) if pos_1d.dim() == 1 else pos_1d
        if seq_lens is None:
            seq_lens = seq_lens_kv - 1
        self._flat_hybrid_hits += 1
        hidden_2d = hidden.reshape(bsz, -1)

        qkv_buf = self._flat_qkv_buf
        attn_buf = self._flat_attn_buf
        o_buf = self._flat_o_buf
        weights = self._flat_layer_weights
        q_proj_size = self._flat_q_size
        q_actual_size = self._flat_num_q_heads * self._flat_head_dim
        k_size = self._flat_k_size
        nqh = self._flat_num_q_heads
        nkvh = self._flat_num_kv_heads
        hdim = self._flat_head_dim
        scale = self._flat_scale
        half_rot = self._flat_half_rotate
        rot_dim = self._flat_rotary_dim
        norm_off = self._flat_norm_offset
        hsize = self._flat_hidden_size
        has_output_gate = self._flat_has_output_gate
        inline = getattr(self, '_flat_inline_kernels', False)
        if inline:
            hidden_add = self._flat_hidden_add_buf
            normed_add = self._flat_normed_add_buf
            fused_block = self._flat_fused_block
            fused_warps = self._flat_fused_warps
            grid_1 = (bsz,)

        for layer_idx, layer in enumerate(self.layers):
            if layer.layer_type == 'linear_attention':
                hidden_3d = hidden_2d.unsqueeze(1)
                if linear_state_cache is not None:
                    linear_conv_state, linear_recurrent_state = linear_state_cache[layer_idx]
                elif block_manager is not None and seq_ids is not None:
                    linear_conv_state, linear_recurrent_state = block_manager.get_linear_state_batch(
                        seq_ids, layer_idx, device=hidden_2d.device,
                    )
                else:
                    linear_conv_state = None
                    linear_recurrent_state = None

                hidden_3d, next_linear_conv, next_linear_recurrent = layer.decode_forward(
                    hidden_3d, *self._get_layer_rope(layer_idx), positions,
                    kv_cache=layer_kv_caches[layer_idx],
                    block_table=block_table,
                    seq_lens=seq_lens,
                    seq_lens_kv=seq_lens_kv,
                    decode_phys_blocks=phys_blocks,
                    decode_blk_offsets=blk_offsets,
                    linear_conv_state=linear_conv_state,
                    linear_recurrent_state=linear_recurrent_state,
                    use_linear_cache=True,
                    timing_events=timing_events,
                    per_layer_input=(
                        per_layer_inputs[:, :, layer_idx, :]
                        if per_layer_inputs is not None else None
                    ),
                )
                if next_linear_conv is not None or next_linear_recurrent is not None:
                    if linear_state_cache is not None:
                        linear_state_cache[layer_idx][0] = next_linear_conv
                        linear_state_cache[layer_idx][1] = next_linear_recurrent
                    elif block_manager is not None and seq_ids is not None:
                        block_manager.set_linear_state_batch(
                            seq_ids, layer_idx, next_linear_conv, next_linear_recurrent,
                        )
                hidden_2d = hidden_3d.reshape(bsz, -1)
            else:
                lw = weights[layer_idx]
                if (
                    not getattr(self, "_flat_hybrid_full_inline_enabled", False)
                    or lw.qkv_wt is None
                    or lw.o_wt is None
                ):
                    self._flat_hybrid_full_fallback_hits += 1
                    hidden_3d = layer.decode_forward_full_attn(
                        hidden_2d.unsqueeze(1), cos, sin, positions,
                        layer_kv_caches[layer_idx],
                        block_table,
                        seq_lens,
                        seq_lens_kv,
                        phys_blocks,
                        blk_offsets,
                        timing_events=timing_events,
                    )
                    hidden_2d = hidden_3d.reshape(bsz, -1)
                    continue
                self._flat_hybrid_full_inline_hits += 1

                flat_start_end = _timing_record_start(True) if record_flat_timing else None
                qkv_out = layer.self_attn._decode_qkv_from_raw_hidden(
                    hidden_2d,
                    lw.norm1_weight,
                    lw.norm_eps,
                    norm_off,
                )
                if qkv_out.dim() != 2:
                    qkv_out = qkv_out.reshape(bsz, -1)
                if record_flat_timing:
                    _timing_record_end(timing_events, "flat_qkv", flat_start_end)

                q_proj = qkv_out[:, :q_proj_size]
                if has_output_gate:
                    q_split = q_proj.view(bsz, nqh, 2 * hdim)
                    q = q_split[:, :, :hdim]
                    q_gate = q_split[:, :, hdim:].reshape(bsz, q_actual_size)
                else:
                    q = q_proj.view(bsz, nqh, hdim)
                    q_gate = None
                k = qkv_out[:, q_proj_size:q_proj_size + k_size].view(bsz, nkvh, hdim)
                v = (
                    k
                    if lw.v_from_k
                    else qkv_out[:, q_proj_size + k_size:].view(bsz, nkvh, hdim)
                )

                flat_start_end = _timing_record_start(True) if record_flat_timing else None
                fused_rope_kv_write(
                    k, v, layer_kv_caches[layer_idx], cos, sin, pos_1d,
                    phys_blocks, blk_offsets,
                    half_rotate=half_rot, rotary_dim=rot_dim,
                    k_norm_weight=lw.k_norm_weight,
                    norm_eps=lw.norm_eps,
                )
                if record_flat_timing:
                    _timing_record_end(timing_events, "flat_rope_kv", flat_start_end)
                flat_start_end = _timing_record_start(True) if record_flat_timing else None
                _triton_paged_decode_fused(
                    q, layer_kv_caches[layer_idx], block_table, seq_lens_kv, scale,
                    cos, sin, pos_1d,
                    half_rotate=half_rot, rotary_dim=rot_dim,
                    q_norm_weight=lw.q_norm_weight,
                    norm_eps=lw.norm_eps,
                    out=attn_buf,
                )
                if record_flat_timing:
                    _timing_record_end(timing_events, "flat_attn_core", flat_start_end)

                attn_2d = attn_buf.reshape(bsz, q_actual_size)
                if q_gate is not None:
                    attn_2d.mul_(torch.sigmoid(q_gate))
                flat_start_end = _timing_record_start(True) if record_flat_timing else None
                torch.mm(attn_2d, lw.o_wt, out=o_buf)
                if lw.o_bias is not None:
                    o_buf.add_(lw.o_bias)
                if record_flat_timing:
                    _timing_record_end(timing_events, "flat_o_proj", flat_start_end)
                flat_start_end = _timing_record_start(True) if record_flat_timing else None
                if inline and _inline_fused_add_norm is not None:
                    _inline_fused_add_norm[grid_1](
                        hidden_2d, o_buf, lw.norm2_weight,
                        hidden_add, normed_add,
                        hsize, hsize,
                        lw.norm_eps,
                        OFFSET=norm_off,
                        BLOCK_SIZE=fused_block,
                        num_warps=fused_warps,
                    )
                    hidden_2d = hidden_add
                    mlp_2d = normed_add
                else:
                    hidden_3d, mlp_3d = fused_add_rmsnorm(
                        hidden_2d.unsqueeze(1),
                        o_buf.unsqueeze(1),
                        lw.norm2_weight,
                        lw.norm_eps,
                        norm_off,
                    )
                    hidden_2d = hidden_3d.squeeze(1)
                    mlp_2d = mlp_3d.reshape(bsz, -1)
                if record_flat_timing:
                    _timing_record_end(timing_events, "flat_resid_norm", flat_start_end)
                is_moe_layer = bool(getattr(layer.mlp, "is_moe", False))
                mlp_start_end = _timing_record_start(
                    timing_events is not None and hidden_2d.is_cuda and _timing_enabled()
                )
                flat_start_end = _timing_record_start(True) if (record_flat_timing and is_moe_layer) else None
                hidden_2d = layer.mlp.forward_decode_add_residual(
                    mlp_2d,
                    hidden_2d,
                    input_is_normed=True,
                    timing_events=timing_events,
                )
                _timing_record_end(timing_events, "mlp", mlp_start_end)
                if record_flat_timing and is_moe_layer:
                    _timing_record_end(timing_events, "flat_moe", flat_start_end)
        return hidden_2d.unsqueeze(1)

    def _flat_should_use_deepfusion_down(
        self,
        gate_up: torch.Tensor,
        lw: _FlatLayerWeights,
        down_buf: torch.Tensor,
    ) -> bool:
        if not _USE_FLAT_DEEPFUSION_DOWN:
            return False
        if deepfusion_swiglu_down is None or lw.down_weight is None or lw.down_wt is None:
            return False
        if not _can_use_deepfusion_mlp_for(gate_up, lw.down_weight):
            return False

        key = _deepfusion_shape_sig(gate_up, lw.down_weight)
        if getattr(self, "_flat_deepfusion_down_key", None) == key:
            return bool(getattr(self, "_flat_deepfusion_down_use", False))

        activation = (
            "gelu_tanh"
            if self.config.hidden_act in ("gelu", "gelu_pytorch_tanh")
            else "silu"
        )
        use = True
        deep_ms = None
        base_ms = None

        if _FLAT_DEEPFUSION_DOWN_BENCH:
            try:
                bsz = int(gate_up.shape[0]) if gate_up.dim() == 2 else int(gate_up.numel() // gate_up.shape[-1])
                isize = int(gate_up.shape[-1] // 2)

                def _deepfusion_call():
                    deepfusion_swiglu_down(
                        gate_up,
                        lw.down_weight,
                        lw.down_bias,
                        out=down_buf,
                        mode="decode",
                        activation=activation,
                    )

                def _baseline_call():
                    if (
                        getattr(self, "_flat_inline_kernels", False)
                        and _inline_swiglu_kernel is not None
                        and hasattr(self, "_flat_swiglu_buf")
                    ):
                        _inline_swiglu_kernel[(bsz,)](
                            gate_up,
                            self._flat_swiglu_buf,
                            isize,
                            BLOCK_SIZE=self._flat_swiglu_block,
                        )
                        activated = self._flat_swiglu_buf
                    else:
                        if swiglu_forward is None:
                            raise RuntimeError("swiglu_forward unavailable for flat DeepFusion benchmark")
                        activated = swiglu_forward(gate_up, isize)
                    torch.mm(activated.reshape(bsz, isize), lw.down_wt, out=down_buf)
                    if lw.down_bias is not None:
                        down_buf.add_(lw.down_bias)

                _deepfusion_call()
                _baseline_call()
                torch.cuda.synchronize()
                deep_ms = _cuda_bench_ms(_deepfusion_call, iters=_FLAT_DEEPFUSION_DOWN_BENCH_ITERS)
                base_ms = _cuda_bench_ms(_baseline_call, iters=_FLAT_DEEPFUSION_DOWN_BENCH_ITERS)
                use = bool(deep_ms <= (base_ms * (1.0 - _DEEPFUSION_MLP_MIN_GAIN)))
            except Exception:
                use = False

        self._flat_deepfusion_down_key = key
        self._flat_deepfusion_down_use = bool(use)
        if _FLAT_DEEPFUSION_DOWN_LOG:
            if deep_ms is None or base_ms is None:
                print(f"[MegaGemm] flat DeepFusion FP16 down use={int(bool(use))} shape={key[:4]}")
            else:
                print(
                    "[MegaGemm] flat DeepFusion FP16 down "
                    f"deep={deep_ms:.3f}ms base={base_ms:.3f}ms use={int(bool(use))} shape={key[:4]}"
                )
        return bool(use)

    def _flat_decode_layers(
        self,
        hidden: torch.Tensor,          # [B, 1, H]
        layer_kv_caches: list,
        cos: torch.Tensor,
        sin: torch.Tensor,
        pos_1d: torch.Tensor,          # [B]
        block_table: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        phys_blocks: torch.Tensor,
        blk_offsets: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
        timing_events: Optional[dict] = None,
        seq_lens: Optional[torch.Tensor] = None,
        linear_state_cache: Optional[dict] = None,
        block_manager: Optional[BlockManagerLike] = None,
        seq_ids: Optional[List[int]] = None,
        max_decode_blocks: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Zero-overhead decode through all layers.
        v2: inline Triton kernels — no wrapper calls, no per-call allocations.
        """
        if getattr(self, "_flat_is_gemma4", False):
            return self._gemma4_flat_decode_layers(
                hidden,
                layer_kv_caches,
                block_table,
                seq_lens_kv,
                phys_blocks,
                blk_offsets,
                pos_1d,
                per_layer_inputs=per_layer_inputs,
                timing_events=timing_events,
            )
        if getattr(self, "_flat_is_hybrid", False):
            return self._hybrid_flat_decode_layers(
                hidden,
                layer_kv_caches,
                cos,
                sin,
                pos_1d,
                block_table,
                seq_lens,
                seq_lens_kv,
                phys_blocks,
                blk_offsets,
                per_layer_inputs=per_layer_inputs,
                timing_events=timing_events,
                linear_state_cache=linear_state_cache,
                block_manager=block_manager,
                seq_ids=seq_ids,
            )
        bsz = hidden.shape[0]
        record_flat_timing = timing_events is not None
        weights = self._flat_layer_weights
        n_layers = self._flat_n_layers

        # Squeeze to 2D — all ops work on [B, H], avoids view_as overhead
        hidden = hidden.reshape(bsz, -1)

        # Pull buffers to locals
        qkv_buf = self._flat_qkv_buf
        attn_buf = self._flat_attn_buf
        o_buf = self._flat_o_buf
        gate_up_buf = self._flat_gate_up_buf
        down_buf = self._flat_down_buf

        # Pull constants to locals
        q_size = self._flat_q_size
        k_size = self._flat_k_size
        nqh = self._flat_num_q_heads
        nkvh = self._flat_num_kv_heads
        hdim = self._flat_head_dim
        scale = self._flat_scale
        half_rot = self._flat_half_rotate
        rot_dim = self._flat_rotary_dim
        norm_off = self._flat_norm_offset
        isize = self._flat_intermediate_size
        hsize = self._flat_hidden_size
        use_fast_gu = self._flat_use_fast_gate_up
        use_fast_down = getattr(self, "_flat_use_fast_down", False)
        inline = getattr(self, '_flat_inline_kernels', False)
        int8_inline = getattr(self, '_flat_int8_inline', False)
        w8a16_ready = getattr(self, '_flat_w8a16_ready', False)
        sparse24_ready = bool(
            getattr(self, '_flat_sparse24_ready', False)
            and callable(_flat_sparse24_mma_linear)
        )
        sparse24_stats = getattr(self, '_mgx_sparsity_runtime', None)

        # INT8/AWQ inline locals (avoid attribute lookup in hot loop)
        w4a16_ready = getattr(self, '_flat_w4a16_ready', False)
        if w4a16_ready and _flat_w4a16_direct is not None:
            w4a16_fn = _flat_w4a16_direct
            w4a16_grid_qkv = self._flat_w4a16_grid_qkv
            w4a16_grid_o = self._flat_w4a16_grid_o
            w4a16_grid_gu = self._flat_w4a16_grid_gu
            w4a16_grid_dn = self._flat_w4a16_grid_dn
        else:
            w4a16_fn = None

        if int8_inline and w8a16_ready and _flat_w8a16_direct is not None:
            # W8A16 direct path: pre-computed grids, direct kernel call
            w8a16_fn = _flat_w8a16_direct
            w8a16_grid_qkv = self._flat_w8a16_grid_qkv
            w8a16_grid_o = self._flat_w8a16_grid_o
            w8a16_grid_gu = self._flat_w8a16_grid_gu
            w8a16_grid_dn = self._flat_w8a16_grid_dn
        elif int8_inline:
            w8a16_fn = None
            int8_x_buf = self._flat_int8_x_buf
            int8_s_buf = self._flat_int8_scale_buf
            int8_bk_h = self._flat_int8_quant_block_h
            int8_bk_i = self._flat_int8_quant_block_i
        else:
            w8a16_fn = None

        if inline:
            hidden_add = self._flat_hidden_add_buf
            normed_add = self._flat_normed_add_buf
            swiglu_out = self._flat_swiglu_buf
            fused_block = self._flat_fused_block
            fused_warps = self._flat_fused_warps
            swiglu_blk = self._flat_swiglu_block
            grid_1 = (bsz,)

        for i in range(n_layers):
            lw = weights[i]
            kv = layer_kv_caches[i]

            # 1. RMSNorm — skip _decode_rmsnorm branches, call CUDA directly
            flat_start_end = _timing_record_start(True) if record_flat_timing else None
            if inline and _can_use_cuda_rmsnorm_for(hidden, norm_off):
                try:
                    normed = rmsnorm_forward(hidden, lw.norm1_weight, lw.norm_eps)
                except Exception:
                    normed = _decode_rmsnorm(hidden, lw.norm1_weight, lw.norm_eps, norm_off)
            else:
                normed = _decode_rmsnorm(hidden, lw.norm1_weight, lw.norm_eps, norm_off)
            if record_flat_timing:
                _timing_record_end(timing_events, "flat_norm1", flat_start_end)

            # 2. QKV projection
            flat_start_end = _timing_record_start(True) if record_flat_timing else None
            if lw.qkv_wt is not None:
                if lw.qkv_bias is not None:
                    torch.addmm(lw.qkv_bias, normed, lw.qkv_wt, out=qkv_buf)
                else:
                    torch.mm(normed, lw.qkv_wt, out=qkv_buf)
                qkv_out = qkv_buf
            elif lw.qkv_dequant_wt is not None:
                if lw.qkv_bias is not None:
                    torch.addmm(lw.qkv_bias, normed, lw.qkv_dequant_wt, out=qkv_buf)
                else:
                    torch.mm(normed, lw.qkv_dequant_wt, out=qkv_buf)
                qkv_out = qkv_buf
            elif sparse24_ready and lw.qkv_mod is not None:
                _flat_sparse24_mma_linear(
                    normed,
                    lw.qkv_mod._mgx_sparse24_native_values,
                    lw.qkv_mod._mgx_sparse24_native_meta,
                    lw.qkv_bias,
                    out=qkv_buf,
                )
                if isinstance(sparse24_stats, dict):
                    sparse24_stats["native_mma_kernel_hits"] += 1
                    sparse24_stats["flat_decode_native_mma_hits"] += 1
                qkv_out = qkv_buf
            elif int8_inline and lw.qkv_int8_w is not None:
                if w8a16_fn is not None:
                    w8a16_fn(
                        normed, lw.qkv_int8_w, lw.qkv_int8_scale,
                        lw.qkv_bias, qkv_buf, w8a16_grid_qkv,
                    )
                else:
                    _flat_int8_linear(
                        normed, lw.qkv_int8_w, lw.qkv_int8_scale,
                        lw.qkv_bias, int8_x_buf, int8_s_buf, int8_bk_h,
                        out=qkv_buf,
                    )
                qkv_out = qkv_buf
            elif w4a16_fn is not None and lw.qkv_awq_qw is not None:
                w4a16_fn(
                    normed, lw.qkv_awq_qw, lw.qkv_awq_scales, lw.qkv_awq_qzeros,
                    lw.qkv_bias, qkv_buf, w4a16_grid_qkv, lw.qkv_awq_gs,
                )
                qkv_out = qkv_buf
            else:
                qkv_out = lw.qkv_mod(normed)
            if record_flat_timing:
                _timing_record_end(timing_events, "flat_qkv", flat_start_end)

            # 3. Split q/k/v
            q = qkv_out[:, :q_size].view(bsz, nqh, hdim)
            k = qkv_out[:, q_size:q_size + k_size].view(bsz, nkvh, hdim)
            v = (
                k
                if lw.v_from_k
                else qkv_out[:, q_size + k_size:].view(bsz, nkvh, hdim)
            )

            # 4. Fused RoPE + KV cache write
            flat_start_end = _timing_record_start(True) if record_flat_timing else None
            fused_rope_kv_write(
                k, v, kv, cos, sin, pos_1d,
                phys_blocks, blk_offsets,
                half_rotate=half_rot, rotary_dim=rot_dim,
                k_norm_weight=lw.k_norm_weight,
                norm_eps=lw.norm_eps,
            )
            if record_flat_timing:
                _timing_record_end(timing_events, "flat_rope_kv", flat_start_end)

            # 5. Paged attention decode + Q RoPE fused
            flat_start_end = _timing_record_start(True) if record_flat_timing else None
            _triton_paged_decode_fused(
                q, kv, block_table, seq_lens_kv, scale,
                cos, sin, pos_1d,
                half_rotate=half_rot, rotary_dim=rot_dim,
                q_norm_weight=lw.q_norm_weight,
                norm_eps=lw.norm_eps,
                out=attn_buf,
                max_blocks_override=max_decode_blocks,
            )
            if record_flat_timing:
                _timing_record_end(timing_events, "flat_attn_core", flat_start_end)

            # 6. O projection
            flat_start_end = _timing_record_start(True) if record_flat_timing else None
            attn_2d = attn_buf.reshape(bsz, nqh * hdim)
            if lw.o_wt is not None:
                torch.mm(attn_2d, lw.o_wt, out=o_buf)
                if lw.o_bias is not None:
                    o_buf.add_(lw.o_bias)
                o_out = o_buf
            elif lw.o_dequant_wt is not None:
                torch.mm(attn_2d, lw.o_dequant_wt, out=o_buf)
                if lw.o_bias is not None:
                    o_buf.add_(lw.o_bias)
                o_out = o_buf
            elif sparse24_ready and lw.o_mod is not None:
                _flat_sparse24_mma_linear(
                    attn_2d,
                    lw.o_mod._mgx_sparse24_native_values,
                    lw.o_mod._mgx_sparse24_native_meta,
                    lw.o_bias,
                    out=o_buf,
                )
                if isinstance(sparse24_stats, dict):
                    sparse24_stats["native_mma_kernel_hits"] += 1
                    sparse24_stats["flat_decode_native_mma_hits"] += 1
                o_out = o_buf
            elif int8_inline and lw.o_int8_w is not None:
                if w8a16_fn is not None:
                    w8a16_fn(
                        attn_2d, lw.o_int8_w, lw.o_int8_scale,
                        lw.o_bias, o_buf, w8a16_grid_o,
                    )
                else:
                    _flat_int8_linear(
                        attn_2d, lw.o_int8_w, lw.o_int8_scale,
                        lw.o_bias, int8_x_buf, int8_s_buf, int8_bk_h,
                        out=o_buf,
                    )
                o_out = o_buf
            elif w4a16_fn is not None and lw.o_awq_qw is not None:
                w4a16_fn(
                    attn_2d, lw.o_awq_qw, lw.o_awq_scales, lw.o_awq_qzeros,
                    lw.o_bias, o_buf, w4a16_grid_o, lw.o_awq_gs,
                )
                o_out = o_buf
            else:
                o_out = lw.o_mod(attn_2d)
            if record_flat_timing:
                _timing_record_end(timing_events, "flat_o_proj", flat_start_end)

            # 7. Fused residual add + RMSNorm
            flat_start_end = _timing_record_start(True) if record_flat_timing else None
            if inline:
                # Direct Triton kernel — no wrapper, no allocation
                _inline_fused_add_norm[grid_1](
                    hidden, o_out, lw.norm2_weight,
                    hidden_add, normed_add,
                    hsize, hsize,
                    lw.norm_eps,
                    OFFSET=norm_off,
                    BLOCK_SIZE=fused_block,
                    num_warps=fused_warps,
                )
                hidden = hidden_add
                mlp_2d = normed_add
            else:
                h3d = hidden.unsqueeze(1)
                hidden_3d, mlp_3d = fused_add_rmsnorm(
                    h3d, o_out.unsqueeze(1),
                    lw.norm2_weight, lw.norm_eps, norm_off,
                )
                hidden = hidden_3d.squeeze(1)
                mlp_2d = mlp_3d.reshape(bsz, -1)
            if record_flat_timing:
                _timing_record_end(timing_events, "flat_resid_norm", flat_start_end)

            if getattr(self, "_flat_is_qwen3_moe", False):
                flat_start_end = _timing_record_start(True) if record_flat_timing else None
                mlp_start_end = _timing_record_start(
                    timing_events is not None and hidden.is_cuda and _timing_enabled()
                )
                hidden = self.layers[i].mlp.forward_decode_add_residual(
                    mlp_2d,
                    hidden,
                    input_is_normed=True,
                    timing_events=timing_events,
                )
                _timing_record_end(timing_events, "mlp", mlp_start_end)
                if record_flat_timing:
                    _timing_record_end(timing_events, "flat_moe", flat_start_end)
                continue

            # 8. Gate+Up projection
            flat_start_end = _timing_record_start(True) if record_flat_timing else None
            if lw.gate_up_wt is not None:
                if use_fast_gu and fast_linear is not None:
                    fast_linear(
                        mlp_2d.unsqueeze(1), lw.gate_up_weight, lw.gate_up_bias,
                        out=gate_up_buf.unsqueeze(1),
                    )
                    gate_up_out = gate_up_buf
                else:
                    torch.mm(mlp_2d, lw.gate_up_wt, out=gate_up_buf)
                    if lw.gate_up_bias is not None:
                        gate_up_buf.add_(lw.gate_up_bias)
                    gate_up_out = gate_up_buf
            elif lw.gate_up_dequant_wt is not None:
                torch.mm(mlp_2d, lw.gate_up_dequant_wt, out=gate_up_buf)
                if lw.gate_up_bias is not None:
                    gate_up_buf.add_(lw.gate_up_bias)
                gate_up_out = gate_up_buf
            elif sparse24_ready and lw.gate_up_mod is not None:
                _flat_sparse24_mma_linear(
                    mlp_2d,
                    lw.gate_up_mod._mgx_sparse24_native_values,
                    lw.gate_up_mod._mgx_sparse24_native_meta,
                    lw.gate_up_bias,
                    out=gate_up_buf,
                )
                if isinstance(sparse24_stats, dict):
                    sparse24_stats["native_mma_kernel_hits"] += 1
                    sparse24_stats["flat_decode_native_mma_hits"] += 1
                gate_up_out = gate_up_buf
            elif int8_inline and lw.gate_up_int8_w is not None:
                if w8a16_fn is not None:
                    w8a16_fn(
                        mlp_2d, lw.gate_up_int8_w, lw.gate_up_int8_scale,
                        lw.gate_up_bias, gate_up_buf, w8a16_grid_gu,
                    )
                else:
                    _flat_int8_linear(
                        mlp_2d, lw.gate_up_int8_w, lw.gate_up_int8_scale,
                        lw.gate_up_bias, int8_x_buf, int8_s_buf, int8_bk_h,
                        out=gate_up_buf,
                    )
                gate_up_out = gate_up_buf
            elif w4a16_fn is not None and lw.gate_up_awq_qw is not None:
                w4a16_fn(
                    mlp_2d, lw.gate_up_awq_qw, lw.gate_up_awq_scales, lw.gate_up_awq_qzeros,
                    lw.gate_up_bias, gate_up_buf, w4a16_grid_gu, lw.gate_up_awq_gs,
                )
                gate_up_out = gate_up_buf
            else:
                gate_up_out = lw.gate_up_mod(mlp_2d)
            if record_flat_timing:
                _timing_record_end(timing_events, "flat_gate_up", flat_start_end)

            # 9. SwiGLU activation
            fused_down_done = False
            flat_start_end = _timing_record_start(True) if record_flat_timing else None
            if (
                _USE_FLAT_DEEPFUSION_DOWN
                and
                lw.down_weight is not None
                and self._flat_should_use_deepfusion_down(gate_up_out, lw, down_buf)
            ):
                deepfusion_swiglu_down(
                    gate_up_out,
                    lw.down_weight,
                    lw.down_bias,
                    out=down_buf,
                    mode="decode",
                    activation=(
                        "gelu_tanh"
                        if self.config.hidden_act in ('gelu', 'gelu_pytorch_tanh')
                        else "silu"
                    ),
                )
                fused_down_done = True
                activated = gate_up_out[:, :isize]
            elif (
                _USE_FLAT_DEEPFUSION_DOWN_DEQUANT
                and lw.down_dequant_raw_wt is not None
                and deepfusion_swiglu_down is not None
                and _can_use_deepfusion_mlp_for(gate_up_out, lw.down_dequant_raw_wt)
            ):
                global _FLAT_DEEPFUSION_DOWN_DEQUANT_LOGGED
                deepfusion_swiglu_down(
                    gate_up_out,
                    lw.down_dequant_raw_wt,
                    lw.down_bias,
                    out=down_buf,
                    mode="decode",
                    activation=(
                        "gelu_tanh"
                        if self.config.hidden_act in ('gelu', 'gelu_pytorch_tanh')
                        else "silu"
                    ),
                )
                fused_down_done = True
                if (
                    _FLAT_FP16_DEQUANT_LOG
                    and not _FLAT_DEEPFUSION_DOWN_DEQUANT_LOGGED
                ):
                    _FLAT_DEEPFUSION_DOWN_DEQUANT_LOGGED = True
                    print(
                        "[MegaGemm] flat DeepFusion down-dequant path active "
                        f"batch={bsz} hidden={hsize} intermediate={isize}"
                    )
                activated = gate_up_out[:, :isize]
            elif inline:
                # Direct Triton kernel — no wrapper, no allocation
                _inline_swiglu_kernel[grid_1](
                    gate_up_out, swiglu_out,
                    isize,
                    BLOCK_SIZE=swiglu_blk,
                )
                activated = swiglu_out
            else:
                activated = swiglu_forward(gate_up_out, isize)
            if record_flat_timing:
                _timing_record_end(timing_events, "flat_swiglu", flat_start_end)

            # 10. Down projection
            flat_start_end = _timing_record_start(True) if record_flat_timing else None
            activated_2d = activated.reshape(bsz, isize)
            if fused_down_done:
                down_out = down_buf
            elif lw.down_wt is not None:
                if use_fast_down and fast_linear is not None and lw.down_weight is not None:
                    fast_linear(
                        activated_2d, lw.down_weight, lw.down_bias,
                        out=down_buf,
                    )
                else:
                    torch.mm(activated_2d, lw.down_wt, out=down_buf)
                    if lw.down_bias is not None:
                        down_buf.add_(lw.down_bias)
                down_out = down_buf
            elif lw.down_dequant_wt is not None:
                torch.mm(activated_2d, lw.down_dequant_wt, out=down_buf)
                if lw.down_bias is not None:
                    down_buf.add_(lw.down_bias)
                down_out = down_buf
            elif sparse24_ready and lw.down_mod is not None:
                _flat_sparse24_mma_linear(
                    activated_2d,
                    lw.down_mod._mgx_sparse24_native_values,
                    lw.down_mod._mgx_sparse24_native_meta,
                    lw.down_bias,
                    out=down_buf,
                )
                if isinstance(sparse24_stats, dict):
                    sparse24_stats["native_mma_kernel_hits"] += 1
                    sparse24_stats["flat_decode_native_mma_hits"] += 1
                down_out = down_buf
            elif int8_inline and lw.down_int8_w is not None:
                if w8a16_fn is not None:
                    w8a16_fn(
                        activated_2d, lw.down_int8_w, lw.down_int8_scale,
                        lw.down_bias, down_buf, w8a16_grid_dn,
                    )
                else:
                    _flat_int8_linear(
                        activated_2d, lw.down_int8_w, lw.down_int8_scale,
                        lw.down_bias, int8_x_buf, int8_s_buf, int8_bk_i,
                        out=down_buf,
                    )
                down_out = down_buf
            elif w4a16_fn is not None and lw.down_awq_qw is not None:
                w4a16_fn(
                    activated_2d, lw.down_awq_qw, lw.down_awq_scales, lw.down_awq_qzeros,
                    lw.down_bias, down_buf, w4a16_grid_dn, lw.down_awq_gs,
                )
                down_out = down_buf
            else:
                down_out = lw.down_mod(activated_2d)
            if record_flat_timing:
                _timing_record_end(timing_events, "flat_down", flat_start_end)

            # 11. Residual add (in-place)
            flat_start_end = _timing_record_start(True) if record_flat_timing else None
            hidden.add_(down_out)
            if record_flat_timing:
                _timing_record_end(timing_events, "flat_residual", flat_start_end)

        return hidden.unsqueeze(1)  # restore [B, 1, H]

    @torch.inference_mode()
    def decode_step(
        self,
        input_ids: torch.Tensor,      # [num_seqs, 1]
        positions: torch.Tensor,       # [num_seqs, 1]
        block_manager: BlockManagerLike,
        seq_ids: List[int],
        logit_lens: Union[bool, int] = False,
        return_next_token: bool = False,
    ) -> torch.Tensor:
        """
        Decode step: process 1 new token per sequence with paged attention.
        Returns logits for next token, or greedy next-token ids when
        return_next_token=True.

        Args:
            logit_lens: Controls Logit Lens probing.
                - False: disabled (default)
                - True: probe ALL layers
                - int N: probe every Nth layer + first + last (stride mode)
                Returns (logits, {layer_idx: probe_logits}) when enabled.
            return_next_token: For greedy decode, skip full-vocab logits materialization
                and return token ids through the fused lm_head+argmax path when possible.
        """
        self._move_rope_to_device(input_ids.device)
        timing_events = None

        block_table = block_manager.get_block_table_tensor(seq_ids)
        seq_lens = block_manager.get_seq_lens_tensor(seq_ids)
        seq_lens_kv = seq_lens + 1
        block_size = block_manager.block_size
        max_decode_blocks = _decode_blocks_for_seq_len(
            _cpu_max_seq_len_from_block_manager(block_manager, seq_ids),
            block_size,
            extra_tokens=1,
        )
        override_max_decode_blocks = getattr(
            block_manager, "get_decode_max_blocks_override", None
        )
        if override_max_decode_blocks is not None:
            max_decode_blocks = override_max_decode_blocks(seq_ids) or max_decode_blocks
        blk_ids = torch.div(seq_lens, block_size, rounding_mode='floor').long()
        decode_blk_offsets = torch.remainder(seq_lens, block_size).long()
        decode_phys_blocks = block_table[torch.arange(len(seq_ids), device=block_table.device), blk_ids]

        hidden = self.embed_tokens(input_ids)
        hidden = self._scale_token_embeddings(hidden)
        per_layer_inputs = self._compute_per_layer_inputs(input_ids, hidden)

        offloader = self._offloader
        do_lens = logit_lens is not False and logit_lens is not None
        layer_probes = {} if do_lens else None
        num_layers = len(self.layers)
        stride = logit_lens if isinstance(logit_lens, int) and logit_lens > 1 else 1
        layer_kv_caches = [block_manager.get_kv_cache(layer_idx) for layer_idx in range(num_layers)]

        # Flat decode avoids per-layer Module dispatch, not all Python execution.
        if not self._flat_decode_ready and not self._flat_decode_failed:
            self._prepare_flat_decode()
        if (
            self._flat_decode_ready
            and offloader is None
            and not do_lens
            and (not self._flat_has_output_gate or getattr(self, "_flat_is_hybrid", False))
        ):
            self._alloc_flat_bufs(
                hidden.shape[0], hidden.device, hidden.dtype,
            )
            pos_1d = positions.squeeze(-1) if positions.dim() > 1 else positions
            hidden = self._flat_decode_layers(
                hidden, layer_kv_caches,
                self.cos_cache, self.sin_cache, pos_1d,
                block_table, seq_lens_kv,
                decode_phys_blocks, decode_blk_offsets,
                per_layer_inputs=per_layer_inputs,
                timing_events=None,
                seq_lens=seq_lens,
                block_manager=block_manager,
                seq_ids=seq_ids,
                max_decode_blocks=max_decode_blocks,
            )
        elif offloader is None and self._all_full_attention and not do_lens and _USE_CPP_DECODE_LOOP:
            hidden = _decode_loop_ops.run_decode_fns_full_attention(
                self._layer_decode_full_fns,
                layer_kv_caches,
                hidden,
                self.cos_cache,
                self.sin_cache,
                positions,
                block_table,
                seq_lens,
                seq_lens_kv,
                decode_phys_blocks,
                decode_blk_offsets,
            )
        elif offloader is None and self._all_full_attention:
            for layer_idx, layer in enumerate(self.layers):
                cos, sin = self._get_layer_rope(layer_idx)
                hidden = layer.decode_forward_full_attn(
                    hidden, cos, sin, positions,
                    layer_kv_caches[layer_idx],
                    block_table,
                    seq_lens,
                    seq_lens_kv,
                    decode_phys_blocks,
                    decode_blk_offsets,
                    timing_events=timing_events,
                )

                if do_lens and (
                    layer_idx == 0 or
                    layer_idx == num_layers - 1 or
                    layer_idx % stride == 0
                ):
                    with torch.no_grad():
                        probe = self.norm(hidden)
                        probe_logits = self.lm_head(probe)
                        probe_logits = self._apply_final_logit_capping(probe_logits)
                        layer_probes[layer_idx] = probe_logits[:, -1, :].squeeze(0)
        else:
            for layer_idx, layer in enumerate(self.layers):
                if offloader:
                    layer = offloader.get_layer_on_gpu(layer_idx, layer)
                    offloader.prefetch_next(layer_idx + 1, self.layers)

                if layer.layer_type == 'linear_attention':
                    linear_conv_state, linear_recurrent_state = block_manager.get_linear_state_batch(
                        seq_ids, layer_idx, device=input_ids.device,
                    )
                    hidden, next_linear_conv, next_linear_recurrent = layer.decode_forward(
                        hidden, *self._get_layer_rope(layer_idx), positions,
                        kv_cache=layer_kv_caches[layer_idx],
                        block_table=block_table,
                        seq_lens=seq_lens,
                        seq_lens_kv=seq_lens_kv,
                        decode_phys_blocks=decode_phys_blocks,
                        decode_blk_offsets=decode_blk_offsets,
                        linear_conv_state=linear_conv_state,
                        linear_recurrent_state=linear_recurrent_state,
                        use_linear_cache=True,
                        per_layer_input=(
                            per_layer_inputs[:, :, layer_idx, :]
                            if per_layer_inputs is not None else None
                        ),
                    )
                    if next_linear_conv is not None or next_linear_recurrent is not None:
                        block_manager.set_linear_state_batch(
                            seq_ids, layer_idx, next_linear_conv, next_linear_recurrent,
                        )
                else:
                    cos, sin = self._get_layer_rope(layer_idx)
                    if layer.is_gemma4:
                        hidden, _, _, _, _ = layer(
                            hidden,
                            cos,
                            sin,
                            positions,
                            kv_cache=layer_kv_caches[layer_idx],
                            block_table=block_table,
                            seq_lens=seq_lens,
                            seq_lens_kv=seq_lens_kv,
                            decode_phys_blocks=decode_phys_blocks,
                            decode_blk_offsets=decode_blk_offsets,
                            is_prefill=False,
                            timing_events=timing_events,
                            per_layer_input=(
                                per_layer_inputs[:, :, layer_idx, :]
                                if per_layer_inputs is not None else None
                            ),
                        )
                    else:
                        hidden = layer.decode_forward_full_attn(
                            hidden, cos, sin, positions,
                            layer_kv_caches[layer_idx],
                            block_table,
                            seq_lens,
                            seq_lens_kv,
                            decode_phys_blocks,
                            decode_blk_offsets,
                            timing_events=timing_events,
                        )

                if do_lens and (
                    layer_idx == 0 or
                    layer_idx == num_layers - 1 or
                    layer_idx % stride == 0
                ):
                    with torch.no_grad():
                        probe = self.norm(hidden)
                        probe_logits = self.lm_head(probe)
                        probe_logits = self._apply_final_logit_capping(probe_logits)
                        layer_probes[layer_idx] = probe_logits[:, -1, :].squeeze(0)

                if offloader:
                    offloader.release_layer(layer_idx, layer)

        block_manager.advance_seq_len_batch(seq_ids, 1)

        if return_next_token and not do_lens:
            next_tokens = self._decode_next_token_greedy(hidden)
            return _apply_benchmark_forced_token(
                next_tokens,
                int(self.config.vocab_size),
            )

        if do_lens:
            hidden = self.norm(hidden)
            logits = self._lm_head_forward(hidden)
            logits = self._apply_final_logit_capping(logits)
            return logits, layer_probes
        return self._decode_logits_from_hidden(hidden)

    @torch.inference_mode()
    def decode_multi_step(
        self,
        input_ids: torch.Tensor,      # [num_seqs, 1] last token per seq
        positions: torch.Tensor,       # [num_seqs, 1] current position per seq
        block_manager: BlockManagerLike,
        seq_ids: List[int],
        num_steps: int = 16,
        stop_token_ids: Optional[set] = None,
        forced_next_token_ids: Optional[torch.Tensor] = None,
        return_final_logits: bool = True,
        return_token_ids: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Multi-step decode: run N decode steps entirely on GPU.

        Avoids N-1 CPU<->GPU roundtrips by keeping the decode loop on GPU.
        Greedy sampling (argmax) happens on GPU between steps.

        When `forced_next_token_ids` is provided, the loop switches to a
        teacher-forced mode after each step: the next token is taken from the
        provided tensor instead of sampling. This is useful for fast KV replay
        of known suffix tokens on top of an already-restored prefix.

        Set ``return_final_logits=False`` for greedy generation paths that only
        need token IDs; this lets fused lm_head+argmax avoid full-vocab logits
        materialization for every step in the burst.

        Set ``return_token_ids=False`` for benchmark paths that only need the
        KV side effects and final sequence lengths. This avoids an otherwise
        useless [batch, steps] token matrix allocation and per-step stores.
        """
        self._move_rope_to_device(input_ids.device)

        num_seqs = input_ids.shape[0]
        offloader = self._offloader

        if forced_next_token_ids is not None:
            if not torch.is_tensor(forced_next_token_ids):
                forced_next_token_ids = torch.tensor(
                    forced_next_token_ids,
                    dtype=torch.long,
                    device=input_ids.device,
                )
            else:
                forced_next_token_ids = forced_next_token_ids.to(
                    device=input_ids.device,
                    dtype=torch.long,
                )
            if forced_next_token_ids.dim() == 1:
                forced_next_token_ids = forced_next_token_ids.unsqueeze(0)
            if forced_next_token_ids.dim() != 2:
                raise ValueError(
                    "forced_next_token_ids must have shape [num_seqs, num_steps - 1]"
                )
            if int(forced_next_token_ids.shape[0]) != num_seqs:
                raise ValueError(
                    "forced_next_token_ids batch dimension must match input_ids"
                )
            max_forced_steps = max(0, int(num_steps) - 1)
            if int(forced_next_token_ids.shape[1]) > max_forced_steps:
                raise ValueError(
                    "forced_next_token_ids cannot contain more than num_steps - 1 columns"
                )
            forced_next_token_ids = forced_next_token_ids.contiguous()
            forced_steps = int(forced_next_token_ids.shape[1])
        else:
            forced_steps = 0

        all_tokens = None
        if return_token_ids:
            all_tokens = torch.empty(
                num_seqs, num_steps, dtype=torch.long, device=input_ids.device,
            )

        cur_ids = input_ids.clone()
        cur_pos = positions.clone()
        final_logits = None
        decode_path_used = "unknown"
        timing_events = {} if (_timing_enabled() and input_ids.is_cuda) else None
        block_table = block_manager.get_block_table_tensor(seq_ids)
        seq_lens = block_manager.get_seq_lens_tensor(seq_ids)
        block_size = block_manager.block_size
        base_max_seq_len = _cpu_max_seq_len_from_block_manager(block_manager, seq_ids)
        override_max_decode_blocks = getattr(
            block_manager, "get_decode_max_blocks_override", None
        )
        stable_max_decode_blocks = None
        if override_max_decode_blocks is not None:
            stable_max_decode_blocks = override_max_decode_blocks(seq_ids)
        batch_indices = torch.arange(num_seqs, device=block_table.device)
        blk_ids = torch.div(seq_lens, block_size, rounding_mode='floor').long()
        decode_blk_offsets = torch.remainder(seq_lens, block_size).long()
        layer_kv_caches = [block_manager.get_kv_cache(layer_idx) for layer_idx in range(len(self.layers))]
        linear_state_cache = {}
        for layer_idx, layer in enumerate(self.layers):
            if layer.layer_type == 'linear_attention':
                linear_state_cache[layer_idx] = list(
                    block_manager.get_linear_state_batch(
                        seq_ids, layer_idx, device=input_ids.device,
                    )
                )

        use_fused_lm_head_argmax = self._prime_fused_lm_head_argmax(
            num_rows=num_seqs,
            device=input_ids.device,
            dtype=self.embed_tokens.weight.dtype,
        )

        # Lazy init flat decode (same as decode_step)
        if not self._flat_decode_ready and not self._flat_decode_failed:
            self._prepare_flat_decode()

        if (
            offloader is None
            and self._all_full_attention
            and _USE_CPP_DECODE_LOOP
            and timing_events is None
            and _decode_loop_ops is not None
            and forced_next_token_ids is None
            and _BENCHMARK_FORCED_TOKEN_ID < 0
            and not self._flat_decode_ready  # flat decode is faster than C++ with Python callbacks
            and return_token_ids
        ):
            # Use local seq-lens tensors in C++ runner; sync BlockManager once at end.
            seq_lens_local = seq_lens.clone()
            blk_ids_local = blk_ids.clone()
            decode_blk_offsets_local = decode_blk_offsets.clone()
            all_tokens, final_logits = _decode_loop_ops.run_decode_steps_full_attention(
                self._layer_decode_full_fns,
                layer_kv_caches,
                self.embed_tokens,
                self._decode_head_forward,
                self._decode_next_token_greedy if use_fused_lm_head_argmax else None,
                cur_ids,
                cur_pos,
                block_table,
                seq_lens_local,
                blk_ids_local,
                decode_blk_offsets_local,
                int(block_size),
                self.cos_cache,
                self.sin_cache,
                int(num_steps),
                float(self.embed_scale),
            )
            block_manager.advance_seq_len_batch(seq_ids, num_steps)
            return all_tokens, final_logits

        for step in range(num_steps):
            embed_start_end = _timing_record_start(timing_events is not None)
            hidden = self.embed_tokens(cur_ids)
            hidden = self._scale_token_embeddings(hidden)
            per_layer_inputs = self._compute_per_layer_inputs(cur_ids, hidden)
            _timing_record_end(timing_events, "embed", embed_start_end)

            decode_phys_blocks = block_table[batch_indices, blk_ids]
            seq_lens_kv = seq_lens + 1
            if stable_max_decode_blocks is not None:
                max_decode_blocks = stable_max_decode_blocks
            else:
                max_decode_blocks = _decode_blocks_for_seq_len(
                    base_max_seq_len,
                    block_size,
                    extra_tokens=step + 1,
                )
            decode_body_start_end = _timing_record_start(timing_events is not None)

            # Flat decode avoids per-layer Module dispatch, not all Python execution.
            if (
                self._flat_decode_ready
                and offloader is None
                and (not self._flat_has_output_gate or getattr(self, "_flat_is_hybrid", False))
            ):
                decode_path_used = "flat"
                self._alloc_flat_bufs(num_seqs, hidden.device, hidden.dtype)
                pos_1d = cur_pos.squeeze(-1) if cur_pos.dim() > 1 else cur_pos
                hidden = self._flat_decode_layers(
                    hidden, layer_kv_caches,
                    self.cos_cache, self.sin_cache, pos_1d,
                    block_table, seq_lens_kv,
                    decode_phys_blocks, decode_blk_offsets,
                    per_layer_inputs=per_layer_inputs,
                    timing_events=timing_events,
                    seq_lens=seq_lens,
                    linear_state_cache=linear_state_cache,
                    max_decode_blocks=max_decode_blocks,
                )
            elif offloader is None and self._all_full_attention and _USE_CPP_DECODE_LOOP:
                decode_path_used = "cpp_layer_fns"
                hidden = _decode_loop_ops.run_decode_fns_full_attention(
                    self._layer_decode_full_fns,
                    layer_kv_caches,
                    hidden,
                    self.cos_cache,
                    self.sin_cache,
                    cur_pos,
                    block_table,
                    seq_lens,
                    seq_lens_kv,
                    decode_phys_blocks,
                    decode_blk_offsets,
                )
            elif offloader is None and self._all_full_attention:
                decode_path_used = "python_full_attn_loop"
                for layer_idx, layer in enumerate(self.layers):
                    cos, sin = self._get_layer_rope(layer_idx)
                    hidden = layer.decode_forward_full_attn(
                        hidden, cos, sin, cur_pos,
                        layer_kv_caches[layer_idx],
                        block_table,
                        seq_lens,
                        seq_lens_kv,
                        decode_phys_blocks,
                        decode_blk_offsets,
                        timing_events=timing_events,
                    )
            else:
                decode_path_used = "mixed_layer_loop"
                for layer_idx, layer in enumerate(self.layers):
                    if offloader:
                        layer = offloader.get_layer_on_gpu(layer_idx, layer)
                        offloader.prefetch_next(layer_idx + 1, self.layers)

                    if layer.layer_type == 'linear_attention':
                        linear_conv_state, linear_recurrent_state = linear_state_cache[layer_idx]
                        hidden, next_linear_conv, next_linear_recurrent = layer.decode_forward(
                            hidden, *self._get_layer_rope(layer_idx), cur_pos,
                            kv_cache=layer_kv_caches[layer_idx],
                            block_table=block_table,
                            seq_lens=seq_lens,
                            seq_lens_kv=seq_lens_kv,
                            decode_phys_blocks=decode_phys_blocks,
                            decode_blk_offsets=decode_blk_offsets,
                            linear_conv_state=linear_conv_state,
                            linear_recurrent_state=linear_recurrent_state,
                            use_linear_cache=True,
                            timing_events=timing_events,
                            per_layer_input=(
                                per_layer_inputs[:, :, layer_idx, :]
                                if per_layer_inputs is not None else None
                            ),
                        )
                        if next_linear_conv is not None or next_linear_recurrent is not None:
                            linear_state_cache[layer_idx][0] = next_linear_conv
                            linear_state_cache[layer_idx][1] = next_linear_recurrent
                    else:
                        cos, sin = self._get_layer_rope(layer_idx)
                        if layer.is_gemma4:
                            hidden, _, _, _, _ = layer(
                                hidden,
                                cos,
                                sin,
                                cur_pos,
                                kv_cache=layer_kv_caches[layer_idx],
                                block_table=block_table,
                                seq_lens=seq_lens,
                                seq_lens_kv=seq_lens_kv,
                                decode_phys_blocks=decode_phys_blocks,
                                decode_blk_offsets=decode_blk_offsets,
                                is_prefill=False,
                                timing_events=timing_events,
                                per_layer_input=(
                                    per_layer_inputs[:, :, layer_idx, :]
                                    if per_layer_inputs is not None else None
                                ),
                            )
                        else:
                            hidden = layer.decode_forward_full_attn(
                                hidden, cos, sin, cur_pos,
                                layer_kv_caches[layer_idx],
                                block_table,
                                seq_lens,
                                seq_lens_kv,
                                decode_phys_blocks,
                                decode_blk_offsets,
                                timing_events=timing_events,
                            )

                    if offloader:
                        offloader.release_layer(layer_idx, layer)

            _timing_record_end(timing_events, "decode_body", decode_body_start_end)

            block_manager.advance_seq_len_batch(seq_ids, 1)
            decode_blk_offsets.add_(1)
            wrapped = decode_blk_offsets.eq(block_size)
            blk_ids.add_(wrapped.to(dtype=blk_ids.dtype))
            decode_blk_offsets.masked_fill_(wrapped, 0)

            forced_next_tokens = None
            if forced_next_token_ids is not None and step < forced_steps:
                forced_next_tokens = forced_next_token_ids[:, step]

            need_logits = (
                (return_final_logits and step == num_steps - 1)
                or (forced_next_tokens is None and not use_fused_lm_head_argmax)
            )

            if need_logits:
                lm_head_start_end = _timing_record_start(timing_events is not None)
                hidden = self.norm(hidden)
                logits = self._lm_head_forward(hidden)
                logits = self._apply_final_logit_capping(logits)
                _timing_record_end(timing_events, "lm_head", lm_head_start_end)
            else:
                lm_head_start_end = _timing_record_start(timing_events is not None)
                next_tokens = self._decode_next_token_greedy(hidden)
                _timing_record_end(timing_events, "lm_head", lm_head_start_end)
                logits = None

            sample_start_end = _timing_record_start(timing_events is not None)
            if forced_next_tokens is not None:
                next_tokens = forced_next_tokens
            elif logits is not None:
                next_tokens = logits[:, -1, :].argmax(dim=-1)
            next_tokens = _apply_benchmark_forced_token(
                next_tokens,
                int(self.config.vocab_size),
            )
            if all_tokens is not None:
                all_tokens[:, step] = next_tokens
            _timing_record_end(timing_events, "sample", sample_start_end)

            cur_ids[:, 0].copy_(next_tokens)
            cur_pos.add_(1)
            final_logits = logits

        for layer_idx, (linear_conv_state, linear_recurrent_state) in linear_state_cache.items():
            block_manager.set_linear_state_batch(
                seq_ids, layer_idx, linear_conv_state, linear_recurrent_state,
            )

        if timing_events:
            torch.cuda.synchronize()
            summary = {"steps": int(num_steps), "decode_path": decode_path_used}
            if decode_path_used == "flat":
                if getattr(self, "_flat_is_gemma4", False):
                    summary["flat_kind"] = "gemma4"
                elif getattr(self, "_flat_is_hybrid", False):
                    summary["flat_kind"] = "hybrid"
                else:
                    summary["flat_kind"] = "dense"
            event_total_ms = 0.0
            derived_timing_keys = {"attn_core_sliding", "attn_core_full"}
            for name, pairs in timing_events.items():
                ms = sum(start.elapsed_time(end) for start, end in pairs)
                summary[f"{name}_ms"] = ms
                if name not in derived_timing_keys:
                    event_total_ms += ms
            if "attn_ms" not in summary:
                attn_keys = (
                    "attn_input_norm_ms",
                    "attn_qkv_ms",
                    "attn_norm_rope_ms",
                    "attn_kv_write_ms",
                    "attn_core_ms",
                    "attn_o_proj_ms",
                    "attn_output_norm_ms",
                    "attn_mlp_bridge_ms",
                )
                attn_total = sum(summary.get(key, 0.0) for key in attn_keys)
                if attn_total > 0.0:
                    summary["attn_ms"] = attn_total
            if "mlp_ms" not in summary:
                mlp_keys = (
                    "mlp_input_norm_ms",
                    "mlp_gate_up_ms",
                    "mlp_act_ms",
                    "mlp_down_ms",
                    "mlp_output_norm_ms",
                    "dense_post_norm_chain_ms",
                )
                mlp_total = sum(summary.get(key, 0.0) for key in mlp_keys)
                if mlp_total > 0.0:
                    summary["mlp_ms"] = mlp_total
            full_attn_keys = (
                "attn_qkv_ms",
                "attn_norm_rope_ms",
                "attn_kv_write_ms",
                "attn_core_ms",
                "attn_o_proj_ms",
            )
            full_attn_total = sum(summary.get(key, 0.0) for key in full_attn_keys)
            if full_attn_total > 0.0:
                summary["full_attn_ms"] = full_attn_total
            linear_attn_keys = (
                "linear_attn_proj_ms",
                "linear_attn_conv_ms",
                "linear_attn_gates_ms",
                "linear_attn_core_ms",
                "linear_attn_norm_ms",
                "linear_attn_out_proj_ms",
                "linear_attn_norm_out_ms",
            )
            linear_attn_total = sum(summary.get(key, 0.0) for key in linear_attn_keys)
            if linear_attn_total > 0.0:
                summary["linear_attn_ms"] = linear_attn_total
            flat_attn_keys = (
                "flat_norm1_ms",
                "flat_qkv_ms",
                "flat_rope_kv_ms",
                "flat_attn_core_ms",
                "flat_o_proj_ms",
                "flat_resid_norm_ms",
            )
            flat_mlp_keys = (
                "flat_gate_up_ms",
                "flat_swiglu_ms",
                "flat_down_ms",
                "flat_moe_ms",
            )
            flat_other_keys = ("flat_residual_ms",)
            flat_attn_total = sum(summary.get(key, 0.0) for key in flat_attn_keys)
            flat_mlp_total = sum(summary.get(key, 0.0) for key in flat_mlp_keys)
            flat_other_total = sum(summary.get(key, 0.0) for key in flat_other_keys)
            flat_total = flat_attn_total + flat_mlp_total + flat_other_total
            if flat_total > 0.0:
                summary["flat_attn_ms"] = flat_attn_total
                summary["flat_mlp_ms"] = flat_mlp_total
                summary["flat_other_ms"] = flat_other_total
                summary["flat_total_ms"] = flat_total
                if "attn_ms" not in summary:
                    summary["attn_ms"] = flat_attn_total
                if "mlp_ms" not in summary:
                    summary["mlp_ms"] = flat_mlp_total
            if "attn_ms" in summary and (full_attn_total > 0.0 or linear_attn_total > 0.0):
                summary["attn_unattributed_ms"] = max(
                    0.0,
                    summary["attn_ms"] - full_attn_total - linear_attn_total,
                )
            if "decode_body_ms" in summary:
                total_ms = sum(
                    summary.get(key, 0.0)
                    for key in ("embed_ms", "decode_body_ms", "lm_head_ms", "sample_ms")
                )
            else:
                total_ms = event_total_ms
            summary["total_ms"] = total_ms
            summary["batch_size"] = int(num_seqs)
            summary["ms_per_step"] = total_ms / max(1, num_steps)
            summary["ms_per_token"] = summary["ms_per_step"]
            summary["ms_per_output_token"] = total_ms / max(1, num_steps * max(1, int(num_seqs)))
            self._last_decode_timing = summary
            if _DECODE_TIMING_PRINT:
                attn_qkv_ms = summary.get("attn_qkv_ms", summary.get("qkv_ms", 0.0))
                attn_norm_rope_ms = summary.get("attn_norm_rope_ms", 0.0)
                attn_kv_write_ms = summary.get("attn_kv_write_ms", 0.0)
                attn_core_ms = summary.get("attn_core_ms", 0.0)
                attn_o_ms = summary.get("attn_o_proj_ms", summary.get("o_proj_ms", 0.0))
                linear_proj_ms = summary.get("linear_attn_proj_ms", 0.0)
                linear_conv_ms = summary.get("linear_attn_conv_ms", 0.0)
                linear_gates_ms = summary.get("linear_attn_gates_ms", 0.0)
                linear_core_ms = summary.get("linear_attn_core_ms", 0.0)
                linear_norm_ms = summary.get("linear_attn_norm_ms", 0.0)
                linear_out_ms = summary.get("linear_attn_out_proj_ms", 0.0)
                linear_norm_out_ms = summary.get("linear_attn_norm_out_ms", 0.0)
                mlp_gate_ms = summary.get("mlp_gate_up_ms", summary.get("gate_up_ms", 0.0))
                mlp_act_ms = summary.get("mlp_act_ms", 0.0)
                mlp_down_ms = summary.get("mlp_down_ms", summary.get("down_proj_ms", 0.0))
                flat_norm_ms = summary.get("flat_norm1_ms", 0.0)
                flat_qkv_ms = summary.get("flat_qkv_ms", 0.0)
                flat_rope_kv_ms = summary.get("flat_rope_kv_ms", 0.0)
                flat_attn_core_ms = summary.get("flat_attn_core_ms", 0.0)
                flat_o_ms = summary.get("flat_o_proj_ms", 0.0)
                flat_resid_norm_ms = summary.get("flat_resid_norm_ms", 0.0)
                flat_gate_up_ms = summary.get("flat_gate_up_ms", 0.0)
                flat_swiglu_ms = summary.get("flat_swiglu_ms", 0.0)
                flat_down_ms = summary.get("flat_down_ms", 0.0)
                flat_moe_ms = summary.get("flat_moe_ms", 0.0)
                dense_post_norm_chain_ms = summary.get(
                    "dense_post_norm_chain_ms", 0.0
                )
                flat_residual_ms = summary.get("flat_residual_ms", 0.0)
                moe_router_ms = summary.get("moe_router_ms", 0.0)
                moe_experts_ms = summary.get("moe_experts_ms", 0.0)
                body_ms = summary.get("decode_body_ms", 0.0)
                mlp_known_ms = max(
                    summary.get("mlp_ms", 0.0),
                    moe_router_ms + moe_experts_ms,
                )
                known_body_ms = (
                    summary.get("attn_ms", 0.0)
                    + mlp_known_ms
                )
                body_other_ms = max(0.0, body_ms - known_body_ms)
                timing_fields = [
                    f"steps={summary['steps']}",
                    f"path={summary.get('decode_path', 'unknown')}",
                ]
                if summary.get("decode_path") == "flat":
                    timing_fields.append(f"flat_kind={summary.get('flat_kind', 'unknown')}")
                if (
                    summary.get("decode_path") != "flat"
                    and getattr(self, "_flat_decode_failed_reason", "")
                ):
                    timing_fields.append(
                        f"flat_fail={getattr(self, '_flat_decode_failed_reason', '')}"
                    )
                timing_fields.extend(
                    [
                        f"ms/token={summary['ms_per_token']:.3f}",
                        f"ms/out_tok={summary.get('ms_per_output_token', 0.0):.3f}",
                        f"embed={summary.get('embed_ms', 0.0):.1f}ms",
                        f"body={summary.get('decode_body_ms', 0.0):.1f}ms",
                        f"attn={summary.get('attn_ms', 0.0):.1f}ms",
                        f"full_attn={summary.get('full_attn_ms', 0.0):.1f}ms",
                        f"full_parts(qkv/rope/kv/core/o)={attn_qkv_ms:.1f}/{attn_norm_rope_ms:.1f}/{attn_kv_write_ms:.1f}/{attn_core_ms:.1f}/{attn_o_ms:.1f}ms",
                        f"flat={summary.get('flat_total_ms', 0.0):.1f}ms",
                        f"flat_parts(norm/qkv/rope_kv/attn/o/rnorm/gu/act/down/resid)={flat_norm_ms:.1f}/{flat_qkv_ms:.1f}/{flat_rope_kv_ms:.1f}/{flat_attn_core_ms:.1f}/{flat_o_ms:.1f}/{flat_resid_norm_ms:.1f}/{flat_gate_up_ms:.1f}/{flat_swiglu_ms:.1f}/{flat_down_ms:.1f}/{flat_residual_ms:.1f}ms",
                        f"flat_moe={flat_moe_ms:.1f}ms",
                        f"dense_post_norm_chain={dense_post_norm_chain_ms:.1f}ms",
                        f"linear_attn={summary.get('linear_attn_ms', 0.0):.1f}ms",
                        f"linear_parts(proj/conv/gates/core/norm/o/fused_no)={linear_proj_ms:.1f}/{linear_conv_ms:.1f}/{linear_gates_ms:.1f}/{linear_core_ms:.1f}/{linear_norm_ms:.1f}/{linear_out_ms:.1f}/{linear_norm_out_ms:.1f}ms",
                        f"attn_other={summary.get('attn_unattributed_ms', 0.0):.1f}ms",
                        f"mlp={summary.get('mlp_ms', 0.0):.1f}ms",
                        f"mlp_parts(gate/act/down)={mlp_gate_ms:.1f}/{mlp_act_ms:.1f}/{mlp_down_ms:.1f}ms",
                        f"moe(router/experts)={moe_router_ms:.1f}/{moe_experts_ms:.1f}ms",
                        f"body_other={body_other_ms:.1f}ms",
                        f"lm_head={summary.get('lm_head_ms', 0.0):.1f}ms",
                        f"sample={summary.get('sample_ms', 0.0):.1f}ms",
                    ]
                )
                print(
                    "decode_timing "
                    + " | ".join(timing_fields)
                )

        return all_tokens, final_logits
