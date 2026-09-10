"""
Decode-oriented fused LM head + argmax.

Computes next token IDs without materializing full logits:
  token = argmax(hidden @ lm_head_weight.T + bias)
"""

from __future__ import annotations

import os
from typing import Optional

import torch

_HAS_TRITON = False
_HAS_LIBDEVICE = False
try:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra import libdevice

        _HAS_LIBDEVICE = True
    except Exception:
        libdevice = None

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    libdevice = None
    _HAS_TRITON = False
    _HAS_LIBDEVICE = False


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


_CFG_SHAPE_GUARD = _env_bool("MEGAGEMM_FUSED_LM_HEAD_ARGMAX_SHAPE_GUARD", True)
_CFG_FORCE_TRITON = _env_bool("MEGAGEMM_FUSED_LM_HEAD_ARGMAX_FORCE_TRITON", False)
_CFG_MAX_ROWS = max(1, _env_int("MEGAGEMM_FUSED_LM_HEAD_ARGMAX_MAX_ROWS", 64))
_CFG_MIN_N = max(1, _env_int("MEGAGEMM_FUSED_LM_HEAD_ARGMAX_MIN_N", 8192))
_CFG_MIN_K = max(1, _env_int("MEGAGEMM_FUSED_LM_HEAD_ARGMAX_MIN_K", 512))
_CFG_FORCED_BN = _env_int("MEGAGEMM_FUSED_LM_HEAD_ARGMAX_BLOCK_N", 0)
_CFG_FORCED_BK = _env_int("MEGAGEMM_FUSED_LM_HEAD_ARGMAX_BLOCK_K", 0)
_CFG_FORCED_WARPS = _env_int("MEGAGEMM_FUSED_LM_HEAD_ARGMAX_NUM_WARPS", 0)
_CFG_FORCED_STAGES = _env_int("MEGAGEMM_FUSED_LM_HEAD_ARGMAX_NUM_STAGES", 0)
_CFG_TRITON_REDUCE = _env_bool("MEGAGEMM_FUSED_LM_HEAD_ARGMAX_TRITON_REDUCE", True)
_CFG_SOFTCAP_BLOCK_N = max(
    128,
    _env_int("MEGAGEMM_LOGITS_SOFTCAP_ARGMAX_BLOCK_N", 1024),
)
_CFG_SOFTCAP_NUM_WARPS = max(
    1,
    _env_int("MEGAGEMM_LOGITS_SOFTCAP_ARGMAX_NUM_WARPS", 8),
)


if _HAS_TRITON:
    @triton.jit
    def _lm_head_block_max_kernel(
        x_ptr,        # [M, K]
        w_ptr,        # [N, K]
        b_ptr,        # [N] or dummy
        out_val_ptr,  # [M, NB]
        out_idx_ptr,  # [M, NB]
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_vm, stride_vn,
        stride_im, stride_in,
        K,
        N,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_nb = tl.program_id(1)

        offs_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        base_x = x_ptr + pid_m * stride_xm
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            x = tl.load(
                base_x + offs_k * stride_xk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w_mask = n_mask[:, None] & k_mask[None, :]
            w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)

        if HAS_BIAS:
            bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc += bias

        neg_inf = -float("inf")
        masked = tl.where(n_mask, acc, neg_inf)
        local_idx = tl.argmax(masked, axis=0)
        local_val = tl.max(masked, axis=0)
        token_idx = pid_nb * BLOCK_N + local_idx

        tl.store(out_val_ptr + pid_m * stride_vm + pid_nb * stride_vn, local_val)
        tl.store(out_idx_ptr + pid_m * stride_im + pid_nb * stride_in, token_idx)


    @triton.jit
    def _lm_head_rmsnorm_block_max_kernel(
        x_ptr,        # [M, K]
        norm_w_ptr,   # [K]
        w_ptr,        # [N, K]
        b_ptr,        # [N] or dummy
        out_val_ptr,  # [M, NB]
        out_idx_ptr,  # [M, NB]
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_vm, stride_vn,
        stride_im, stride_in,
        K,
        N,
        EPS: tl.constexpr,
        NORM_OFFSET: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_nb = tl.program_id(1)

        offs_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        base_x = x_ptr + pid_m * stride_xm

        sumsq = tl.zeros([], dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            x_raw = tl.load(
                base_x + offs_k * stride_xk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            sumsq += tl.sum(x_raw * x_raw, axis=0)

        inv_rms = tl.rsqrt(sumsq / K + EPS)

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            x_raw = tl.load(
                base_x + offs_k * stride_xk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            norm_w = tl.load(norm_w_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            if NORM_OFFSET:
                norm_w += 1.0
            x = x_raw * inv_rms * norm_w
            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w_mask = n_mask[:, None] & k_mask[None, :]
            w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)

        if HAS_BIAS:
            bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc += bias

        neg_inf = -float("inf")
        masked = tl.where(n_mask, acc, neg_inf)
        local_idx = tl.argmax(masked, axis=0)
        local_val = tl.max(masked, axis=0)
        token_idx = pid_nb * BLOCK_N + local_idx

        tl.store(out_val_ptr + pid_m * stride_vm + pid_nb * stride_vn, local_val)
        tl.store(out_idx_ptr + pid_m * stride_im + pid_nb * stride_in, token_idx)


    @triton.jit
    def _lm_head_reduce_kernel(
        partial_val_ptr,  # [M, NB]
        partial_idx_ptr,  # [M, NB]
        out_tok_ptr,      # [M]
        stride_vm, stride_vn,
        stride_im, stride_in,
        stride_tm,
        NB,
        BLOCK_B: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs = tl.arange(0, BLOCK_B)
        mask = offs < NB

        vals = tl.load(
            partial_val_ptr + pid_m * stride_vm + offs * stride_vn,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        best_off = tl.argmax(vals, axis=0)
        tok = tl.load(
            partial_idx_ptr + pid_m * stride_im + best_off * stride_in,
            mask=best_off < NB,
            other=0,
        )
        tl.store(out_tok_ptr + pid_m * stride_tm, tok)


    if _HAS_LIBDEVICE:
        @triton.jit
        def _logits_softcap_block_max_kernel(
            logits_ptr,   # [M, N], BF16 for the promoted Gemma4 path
            out_val_ptr,  # [M, NB]
            out_idx_ptr,  # [M, NB]
            stride_lm, stride_ln,
            stride_vm, stride_vn,
            stride_im, stride_in,
            N,
            CAP: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            """Apply Gemma BF16 softcap and keep one maximum per vocab tile."""
            pid_m = tl.program_id(0)
            pid_nb = tl.program_id(1)
            offs_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = offs_n < N

            raw = tl.load(
                logits_ptr + pid_m * stride_lm + offs_n * stride_ln,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

            # Match `cap * torch.tanh(logits / cap)` on a BF16 tensor.  Each
            # PyTorch pointwise operation stores BF16, so preserve all three
            # rounding boundaries; they affect first-index tie breaking.
            scaled = (raw / CAP).to(tl.bfloat16)
            activated = libdevice.tanh(scaled.to(tl.float32)).to(tl.bfloat16)
            capped = (activated.to(tl.float32) * CAP).to(tl.bfloat16)
            values = tl.where(mask, capped.to(tl.float32), -float("inf"))
            local_idx = tl.argmax(values, axis=0)
            local_val = tl.max(values, axis=0)

            tl.store(
                out_val_ptr + pid_m * stride_vm + pid_nb * stride_vn,
                local_val,
            )
            tl.store(
                out_idx_ptr + pid_m * stride_im + pid_nb * stride_in,
                pid_nb * BLOCK_N + local_idx,
            )

def _pick_cfg(
    k_dim: int,
    n_dim: int,
    *,
    rows: int = 0,
    dtype: Optional[torch.dtype] = None,
    device_name: str = "",
):
    if _CFG_FORCED_BN > 0 and _CFG_FORCED_BK > 0 and _CFG_FORCED_WARPS > 0:
        return _CFG_FORCED_BN, _CFG_FORCED_BK, _CFG_FORCED_WARPS, max(1, _CFG_FORCED_STAGES or 2)

    # Exact full-model promotion: Gemma 4 E2B decode on L4, batch 8.
    # The paired P512/P2048 x O16/O128 gate measured BN64 at +2.19%
    # geometric-mean decode throughput, with all end-to-end scenarios
    # non-regressing. Keep this guard narrower than the generic large-vocab
    # heuristic because occupancy depends on M/K/N and the GPU.
    if (
        int(rows) == 8
        and int(k_dim) == 1536
        and int(n_dim) == 262144
        and dtype == torch.bfloat16
        and "L4" in str(device_name).upper()
    ):
        return 64, 128, 4, 2

    if n_dim >= 65536:
        # For very large vocab projections the hidden width matters a lot.
        # H=1024/2048 tolerates wider vocab tiles; H>=2560 on T4 tends to
        # lose occupancy/register headroom with BLOCK_N=256.
        block_n = 128 if k_dim > 2048 else 256
    elif n_dim >= 16384:
        block_n = 128
    else:
        block_n = 64

    if k_dim >= 2048:
        block_k = 128
    elif k_dim >= 1024:
        block_k = 128
    else:
        block_k = 64

    num_warps = 4
    num_stages = 2
    return block_n, block_k, num_warps, num_stages


def lm_head_argmax_prefers_triton_shape(in_dim: int, vocab_dim: int, rows: int) -> bool:
    if _CFG_FORCE_TRITON:
        return True
    if not _CFG_SHAPE_GUARD:
        return True
    if rows > _CFG_MAX_ROWS:
        return False
    if in_dim < _CFG_MIN_K:
        return False
    if vocab_dim < _CFG_MIN_N:
        return False
    return True


def _flatten_rows(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x if x.is_contiguous() else x.contiguous()
    x_2d = x.flatten(0, -2)
    return x_2d if x_2d.is_contiguous() else x_2d.contiguous()


def logits_softcap_argmax(
    logits: torch.Tensor,
    cap: float,
    out_tokens: Optional[torch.Tensor] = None,
    partial_vals: Optional[torch.Tensor] = None,
    partial_idxs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return argmax after Gemma's BF16 logit softcap without storing capped logits."""
    cap_value = float(cap)
    if cap_value <= 0.0:
        raise ValueError(f"softcap must be positive, got {cap_value}")

    logits_2d = _flatten_rows(logits)
    m_rows = int(logits_2d.shape[0])
    vocab = int(logits_2d.shape[1])
    use_triton = bool(
        _HAS_TRITON
        and _HAS_LIBDEVICE
        and logits_2d.is_cuda
        and logits_2d.dtype == torch.bfloat16
        and not torch.is_grad_enabled()
    )
    if not use_triton:
        capped = cap_value * torch.tanh(logits_2d / cap_value)
        tokens = capped.argmax(dim=-1)
        if out_tokens is not None:
            out_tokens[:m_rows].copy_(tokens)
            return out_tokens[:m_rows]
        return tokens

    block_n = min(65536, triton.next_power_of_2(_CFG_SOFTCAP_BLOCK_N))
    n_blocks = triton.cdiv(vocab, block_n)
    if _CFG_SOFTCAP_NUM_WARPS <= 1:
        num_warps = 1
    elif _CFG_SOFTCAP_NUM_WARPS <= 2:
        num_warps = 2
    elif _CFG_SOFTCAP_NUM_WARPS <= 4:
        num_warps = 4
    else:
        num_warps = 8

    if (
        partial_vals is not None
        and partial_vals.device == logits_2d.device
        and partial_vals.dtype == torch.float32
        and partial_vals.dim() == 2
        and partial_vals.shape[0] >= m_rows
        and partial_vals.shape[1] >= n_blocks
    ):
        partial_vals = partial_vals[:m_rows, :n_blocks]
    else:
        partial_vals = torch.empty(
            (m_rows, n_blocks),
            device=logits_2d.device,
            dtype=torch.float32,
        )
    if (
        partial_idxs is not None
        and partial_idxs.device == logits_2d.device
        and partial_idxs.dtype == torch.int32
        and partial_idxs.dim() == 2
        and partial_idxs.shape[0] >= m_rows
        and partial_idxs.shape[1] >= n_blocks
    ):
        partial_idxs = partial_idxs[:m_rows, :n_blocks]
    else:
        partial_idxs = torch.empty(
            (m_rows, n_blocks),
            device=logits_2d.device,
            dtype=torch.int32,
        )
    if (
        out_tokens is not None
        and out_tokens.device == logits_2d.device
        and out_tokens.dtype == torch.long
        and out_tokens.dim() == 1
        and out_tokens.shape[0] >= m_rows
    ):
        out_tokens = out_tokens[:m_rows]
    else:
        out_tokens = torch.empty((m_rows,), device=logits_2d.device, dtype=torch.long)

    _logits_softcap_block_max_kernel[(m_rows, n_blocks)](
        logits_2d,
        partial_vals,
        partial_idxs,
        logits_2d.stride(0), logits_2d.stride(1),
        partial_vals.stride(0), partial_vals.stride(1),
        partial_idxs.stride(0), partial_idxs.stride(1),
        vocab,
        CAP=cap_value,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=1,
    )
    block_b = triton.next_power_of_2(n_blocks)
    reduce_warps = 4 if block_b >= 1024 else (2 if block_b >= 256 else 1)
    _lm_head_reduce_kernel[(m_rows,)](
        partial_vals,
        partial_idxs,
        out_tokens,
        partial_vals.stride(0), partial_vals.stride(1),
        partial_idxs.stride(0), partial_idxs.stride(1),
        out_tokens.stride(0),
        n_blocks,
        BLOCK_B=block_b,
        num_warps=reduce_warps,
    )
    return out_tokens


def lm_head_argmax(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: Optional[torch.Tensor] = None,
    out_tokens: Optional[torch.Tensor] = None,
    partial_vals: Optional[torch.Tensor] = None,
    partial_idxs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Return argmax token IDs for each hidden row.
    """
    if hidden.shape[-1] != lm_head_weight.shape[-1]:
        raise ValueError(
            f"in_features mismatch: hidden={hidden.shape[-1]} lm_head_weight={lm_head_weight.shape[-1]}"
        )

    x_2d = _flatten_rows(hidden)
    m_rows = int(x_2d.shape[0])
    k_dim = int(x_2d.shape[1])
    vocab = int(lm_head_weight.shape[0])

    use_triton = (
        _HAS_TRITON
        and hidden.is_cuda
        and lm_head_weight.is_cuda
        and not torch.is_grad_enabled()
        and lm_head_argmax_prefers_triton_shape(k_dim, vocab, m_rows)
    )
    if not use_triton:
        logits = torch.nn.functional.linear(x_2d, lm_head_weight, lm_head_bias)
        tok = logits.argmax(dim=-1)
        if out_tokens is not None:
            out_tokens.copy_(tok)
            return out_tokens
        return tok

    w = lm_head_weight if lm_head_weight.is_contiguous() else lm_head_weight.contiguous()
    bias_ptr = lm_head_bias if lm_head_bias is not None else x_2d
    device_name = (
        torch.cuda.get_device_name(x_2d.device)
        if (
            m_rows == 8
            and k_dim == 1536
            and vocab == 262144
            and x_2d.dtype == torch.bfloat16
        )
        else ""
    )
    block_n, block_k, num_warps, num_stages = _pick_cfg(
        k_dim,
        vocab,
        rows=m_rows,
        dtype=x_2d.dtype,
        device_name=device_name,
    )
    n_blocks = triton.cdiv(vocab, block_n)

    if (
        partial_vals is not None
        and partial_vals.device == x_2d.device
        and partial_vals.dtype == torch.float32
        and partial_vals.dim() == 2
        and partial_vals.shape[0] >= m_rows
        and partial_vals.shape[1] >= n_blocks
    ):
        partial_vals = partial_vals[:m_rows, :n_blocks]
    else:
        partial_vals = torch.empty((m_rows, n_blocks), device=x_2d.device, dtype=torch.float32)
    if (
        partial_idxs is not None
        and partial_idxs.device == x_2d.device
        and partial_idxs.dtype == torch.int32
        and partial_idxs.dim() == 2
        and partial_idxs.shape[0] >= m_rows
        and partial_idxs.shape[1] >= n_blocks
    ):
        partial_idxs = partial_idxs[:m_rows, :n_blocks]
    else:
        partial_idxs = torch.empty((m_rows, n_blocks), device=x_2d.device, dtype=torch.int32)

    grid = (m_rows, n_blocks)
    _lm_head_block_max_kernel[grid](
        x_2d,
        w,
        bias_ptr,
        partial_vals,
        partial_idxs,
        x_2d.stride(0), x_2d.stride(1),
        w.stride(0), w.stride(1),
        partial_vals.stride(0), partial_vals.stride(1),
        partial_idxs.stride(0), partial_idxs.stride(1),
        k_dim,
        vocab,
        HAS_BIAS=1 if lm_head_bias is not None else 0,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if (
        out_tokens is not None
        and out_tokens.device == x_2d.device
        and out_tokens.dtype == torch.long
        and out_tokens.dim() == 1
        and out_tokens.shape[0] >= m_rows
    ):
        out_tokens = out_tokens[:m_rows]
    else:
        out_tokens = torch.empty((m_rows,), device=x_2d.device, dtype=torch.long)

    if _CFG_TRITON_REDUCE:
        block_b = triton.next_power_of_2(n_blocks)
        reduce_warps = 4 if block_b >= 1024 else (2 if block_b >= 256 else 1)
        _lm_head_reduce_kernel[(m_rows,)](
            partial_vals,
            partial_idxs,
            out_tokens,
            partial_vals.stride(0), partial_vals.stride(1),
            partial_idxs.stride(0), partial_idxs.stride(1),
            out_tokens.stride(0),
            n_blocks,
            BLOCK_B=block_b,
            num_warps=reduce_warps,
        )
        return out_tokens

    best_block = partial_vals.argmax(dim=-1)  # [M]
    best_idx = torch.gather(
        partial_idxs,
        1,
        best_block.unsqueeze(-1),
    ).squeeze(-1).to(dtype=torch.long)
    out_tokens.copy_(best_idx)
    return out_tokens


def lm_head_rmsnorm_argmax(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    norm_offset: bool,
    lm_head_weight: torch.Tensor,
    lm_head_bias: Optional[torch.Tensor] = None,
    out_tokens: Optional[torch.Tensor] = None,
    partial_vals: Optional[torch.Tensor] = None,
    partial_idxs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Return argmax token IDs for RMSNorm(hidden) @ lm_head_weight.T.
    """
    if hidden.shape[-1] != lm_head_weight.shape[-1]:
        raise ValueError(
            f"in_features mismatch: hidden={hidden.shape[-1]} lm_head_weight={lm_head_weight.shape[-1]}"
        )
    if norm_weight.shape[-1] != hidden.shape[-1]:
        raise ValueError(
            f"norm weight mismatch: hidden={hidden.shape[-1]} norm={norm_weight.shape[-1]}"
        )

    x_2d = _flatten_rows(hidden)
    m_rows = int(x_2d.shape[0])
    k_dim = int(x_2d.shape[1])
    vocab = int(lm_head_weight.shape[0])

    use_triton = (
        _HAS_TRITON
        and hidden.is_cuda
        and norm_weight.is_cuda
        and lm_head_weight.is_cuda
        and not torch.is_grad_enabled()
        and lm_head_argmax_prefers_triton_shape(k_dim, vocab, m_rows)
    )
    if not use_triton:
        rms = torch.rsqrt(x_2d.float().pow(2).mean(dim=-1, keepdim=True) + float(norm_eps))
        scale = norm_weight.float() + 1.0 if norm_offset else norm_weight.float()
        normed = (x_2d * rms * scale).to(dtype=hidden.dtype)
        return lm_head_argmax(
            normed,
            lm_head_weight,
            lm_head_bias,
            out_tokens=out_tokens,
            partial_vals=partial_vals,
            partial_idxs=partial_idxs,
        )

    w = lm_head_weight if lm_head_weight.is_contiguous() else lm_head_weight.contiguous()
    norm_w = norm_weight if norm_weight.is_contiguous() else norm_weight.contiguous()
    bias_ptr = lm_head_bias if lm_head_bias is not None else x_2d
    device_name = (
        torch.cuda.get_device_name(x_2d.device)
        if (
            m_rows == 8
            and k_dim == 1536
            and vocab == 262144
            and x_2d.dtype == torch.bfloat16
        )
        else ""
    )
    block_n, block_k, num_warps, num_stages = _pick_cfg(
        k_dim,
        vocab,
        rows=m_rows,
        dtype=x_2d.dtype,
        device_name=device_name,
    )
    n_blocks = triton.cdiv(vocab, block_n)

    if (
        partial_vals is not None
        and partial_vals.device == x_2d.device
        and partial_vals.dtype == torch.float32
        and partial_vals.dim() == 2
        and partial_vals.shape[0] >= m_rows
        and partial_vals.shape[1] >= n_blocks
    ):
        partial_vals = partial_vals[:m_rows, :n_blocks]
    else:
        partial_vals = torch.empty((m_rows, n_blocks), device=x_2d.device, dtype=torch.float32)
    if (
        partial_idxs is not None
        and partial_idxs.device == x_2d.device
        and partial_idxs.dtype == torch.int32
        and partial_idxs.dim() == 2
        and partial_idxs.shape[0] >= m_rows
        and partial_idxs.shape[1] >= n_blocks
    ):
        partial_idxs = partial_idxs[:m_rows, :n_blocks]
    else:
        partial_idxs = torch.empty((m_rows, n_blocks), device=x_2d.device, dtype=torch.int32)

    _lm_head_rmsnorm_block_max_kernel[(m_rows, n_blocks)](
        x_2d,
        norm_w,
        w,
        bias_ptr,
        partial_vals,
        partial_idxs,
        x_2d.stride(0), x_2d.stride(1),
        w.stride(0), w.stride(1),
        partial_vals.stride(0), partial_vals.stride(1),
        partial_idxs.stride(0), partial_idxs.stride(1),
        k_dim,
        vocab,
        EPS=float(norm_eps),
        NORM_OFFSET=bool(norm_offset),
        HAS_BIAS=1 if lm_head_bias is not None else 0,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if (
        out_tokens is not None
        and out_tokens.device == x_2d.device
        and out_tokens.dtype == torch.long
        and out_tokens.dim() == 1
        and out_tokens.shape[0] >= m_rows
    ):
        out_tokens = out_tokens[:m_rows]
    else:
        out_tokens = torch.empty((m_rows,), device=x_2d.device, dtype=torch.long)

    if _CFG_TRITON_REDUCE:
        block_b = triton.next_power_of_2(n_blocks)
        reduce_warps = 4 if block_b >= 1024 else (2 if block_b >= 256 else 1)
        _lm_head_reduce_kernel[(m_rows,)](
            partial_vals,
            partial_idxs,
            out_tokens,
            partial_vals.stride(0), partial_vals.stride(1),
            partial_idxs.stride(0), partial_idxs.stride(1),
            out_tokens.stride(0),
            n_blocks,
            BLOCK_B=block_b,
            num_warps=reduce_warps,
        )
        return out_tokens

    best_block = partial_vals.argmax(dim=-1)
    best_idx = torch.gather(
        partial_idxs,
        1,
        best_block.unsqueeze(-1),
    ).squeeze(-1).to(dtype=torch.long)
    out_tokens.copy_(best_idx)
    return out_tokens


HAS_FUSED_LM_HEAD_ARGMAX = _HAS_TRITON
HAS_FUSED_SOFTCAP_ARGMAX = bool(_HAS_TRITON and _HAS_LIBDEVICE)


def lm_head_argmax_runtime_config() -> dict:
    return {
        "has_triton": bool(_HAS_TRITON),
        "has_libdevice": bool(_HAS_LIBDEVICE),
        "shape_guard": bool(_CFG_SHAPE_GUARD),
        "force_triton": bool(_CFG_FORCE_TRITON),
        "max_rows": int(_CFG_MAX_ROWS),
        "min_n": int(_CFG_MIN_N),
        "min_k": int(_CFG_MIN_K),
        "forced_block_n": int(_CFG_FORCED_BN),
        "forced_block_k": int(_CFG_FORCED_BK),
        "forced_num_warps": int(_CFG_FORCED_WARPS),
        "forced_num_stages": int(_CFG_FORCED_STAGES),
        "triton_reduce": bool(_CFG_TRITON_REDUCE),
        "large_vocab_large_k_block_n": 128,
        "gemma4_e2b_l4_b8_block_n": 64,
        "softcap_block_n": int(_CFG_SOFTCAP_BLOCK_N),
        "softcap_num_warps": int(_CFG_SOFTCAP_NUM_WARPS),
    }


__all__ = [
    "lm_head_argmax",
    "lm_head_rmsnorm_argmax",
    "logits_softcap_argmax",
    "lm_head_argmax_prefers_triton_shape",
    "lm_head_argmax_runtime_config",
    "HAS_FUSED_LM_HEAD_ARGMAX",
    "HAS_FUSED_SOFTCAP_ARGMAX",
]
