"""
MegaGemm Kernels — Custom CUDA/Triton kernels for LLM inference.

- RMSNorm (CUDA) — FP32/FP16/BF16 vectorized normalization
- SwiGLU (Triton) — Fused gate+up activation
- RoPE (CUDA/PyTorch) — Rotary position embeddings
- PagedAttention (Triton) — Paged KV cache decode
- FusedAddRMSNorm (Triton) — Fused residual add + RMSNorm
"""

from .rmsnorm import RMSNorm, RMSNormFunction
from .rope import RoPE, apply_rotary_emb, precompute_freqs_cis
from .paged_attention import paged_attention_decode, prefill_attention

# SwiGLU may not be available (requires Triton)
try:
    from .swiglu import MegaGemmTriton, MegaGemmFunction
except Exception:
    pass

# Fused Add + RMSNorm (requires Triton)
try:
    from .fused_add_rmsnorm import fused_add_rmsnorm, HAS_FUSED_ADD_RMSNORM
except Exception:
    HAS_FUSED_ADD_RMSNORM = False

# Fused RMSNorm + SiLU gate (requires Triton)
try:
    from .rmsnorm_gated import rmsnorm_gated, HAS_RMSNORM_GATED
except Exception:
    rmsnorm_gated = None
    HAS_RMSNORM_GATED = False

# Decode fused RMSNormGated + Linear for Qwen 3.5 linear attention
try:
    from .rmsnorm_gated_linear import (
        rmsnorm_gated_linear_decode,
        rmsnorm_gated_linear_runtime_config,
        HAS_RMSNORM_GATED_LINEAR,
    )
except Exception:
    rmsnorm_gated_linear_decode = None
    rmsnorm_gated_linear_runtime_config = None
    HAS_RMSNORM_GATED_LINEAR = False

# Standalone Triton RMSNorm with offset support
try:
    from .rmsnorm_triton import (
        rmsnorm_triton,
        rmsnorm_triton_no_weight,
        rmsnorm_triton_scaled_no_weight,
        rmsnorm_triton_residual_scale_next,
        HAS_TRITON_RMSNORM,
    )
except Exception:
    rmsnorm_triton = None
    rmsnorm_triton_no_weight = None
    rmsnorm_triton_scaled_no_weight = None
    rmsnorm_triton_residual_scale_next = None
    HAS_TRITON_RMSNORM = False

# Gemma4 short-prefill Q/K/V norm + RoPE + layout preparation.
try:
    from .gemma4_attention_prepare import (
        gemma4_prefill_attention_prepare,
        HAS_GEMMA4_ATTENTION_PREPARE,
    )
except Exception:
    gemma4_prefill_attention_prepare = None
    HAS_GEMMA4_ATTENTION_PREPARE = False

# INT8 Fused GEMM (requires Triton + sm_80+)
try:
    from .int8_gemm import int8_fused_gemm, HAS_INT8_FUSED_GEMM
except Exception:
    HAS_INT8_FUSED_GEMM = False

# Decode DeepFusion MLP (requires Triton)
try:
    from .deepfusion_mlp import (
        deepfusion_swiglu_down,
        gemma4_e2b_b8_geglu_down_tensorcore,
        deepfusion_mlp_prefers_triton_shape,
        deepfusion_runtime_config,
        HAS_DEEPFUSION_MLP,
    )
except Exception:
    deepfusion_swiglu_down = None
    gemma4_e2b_b8_geglu_down_tensorcore = None
    deepfusion_mlp_prefers_triton_shape = None
    deepfusion_runtime_config = None
    HAS_DEEPFUSION_MLP = False

# Decode fused RMSNorm + Linear (requires Triton)
try:
    from .fused_rmsnorm_linear import (
        fused_rmsnorm_linear,
        fused_rmsnorm_linear_prefers_triton_shape,
        fused_rmsnorm_linear_runtime_config,
        HAS_FUSED_RMSNORM_LINEAR,
    )
except Exception:
    fused_rmsnorm_linear = None
    fused_rmsnorm_linear_prefers_triton_shape = None
    fused_rmsnorm_linear_runtime_config = None
    HAS_FUSED_RMSNORM_LINEAR = False

# Decode fused LM head + argmax (requires Triton)
try:
    from .lm_head_argmax import (
        lm_head_argmax,
        lm_head_rmsnorm_argmax,
        lm_head_argmax_prefers_triton_shape,
        lm_head_argmax_runtime_config,
        HAS_FUSED_LM_HEAD_ARGMAX,
    )
except Exception:
    lm_head_argmax = None
    lm_head_rmsnorm_argmax = None
    lm_head_argmax_prefers_triton_shape = None
    lm_head_argmax_runtime_config = None
    HAS_FUSED_LM_HEAD_ARGMAX = False

# Qwen 3.5 linear attention recurrent decode (requires Triton)
try:
    from .linear_attention import (
        chunk_interchunk,
        chunk_interchunk_scan,
        chunk_state_projection,
        chunk_state_update,
        recurrent_gated_delta_decode,
        recurrent_gated_delta_decode_from_ab,
        recurrent_gated_delta_prefill,
        solve_chunk_local_attention,
        HAS_TRITON_LINEAR_ATTN,
    )
except Exception:
    chunk_interchunk = None
    chunk_interchunk_scan = None
    chunk_state_projection = None
    chunk_state_update = None
    recurrent_gated_delta_decode = None
    recurrent_gated_delta_decode_from_ab = None
    recurrent_gated_delta_prefill = None
    solve_chunk_local_attention = None
    HAS_TRITON_LINEAR_ATTN = False

__all__ = [
    'RMSNorm', 'RMSNormFunction',
    'rmsnorm_triton', 'rmsnorm_triton_no_weight',
    'rmsnorm_triton_scaled_no_weight', 'HAS_TRITON_RMSNORM',
    'rmsnorm_triton_residual_scale_next',
    'gemma4_prefill_attention_prepare', 'HAS_GEMMA4_ATTENTION_PREPARE',
    'MegaGemmTriton', 'MegaGemmFunction',
    'RoPE', 'apply_rotary_emb', 'precompute_freqs_cis',
    'paged_attention_decode', 'prefill_attention',
    'fused_add_rmsnorm', 'HAS_FUSED_ADD_RMSNORM',
    'rmsnorm_gated', 'HAS_RMSNORM_GATED',
    'rmsnorm_gated_linear_decode', 'rmsnorm_gated_linear_runtime_config',
    'HAS_RMSNORM_GATED_LINEAR',
    'int8_fused_gemm', 'HAS_INT8_FUSED_GEMM',
    'deepfusion_swiglu_down', 'deepfusion_mlp_prefers_triton_shape',
    'gemma4_e2b_b8_geglu_down_tensorcore',
    'deepfusion_runtime_config', 'HAS_DEEPFUSION_MLP',
    'fused_rmsnorm_linear', 'fused_rmsnorm_linear_prefers_triton_shape',
    'fused_rmsnorm_linear_runtime_config', 'HAS_FUSED_RMSNORM_LINEAR',
    'lm_head_argmax', 'lm_head_rmsnorm_argmax', 'lm_head_argmax_prefers_triton_shape',
    'lm_head_argmax_runtime_config', 'HAS_FUSED_LM_HEAD_ARGMAX',
    'chunk_interchunk',
    'chunk_interchunk_scan',
    'chunk_state_projection', 'chunk_state_update',
    'recurrent_gated_delta_decode', 'recurrent_gated_delta_decode_from_ab',
    'recurrent_gated_delta_prefill',
    'solve_chunk_local_attention', 'HAS_TRITON_LINEAR_ATTN',
]
