"""Fused Gemma4 prefill Q/K/V RMSNorm, RoPE, and layout preparation."""

from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _gemma4_prefill_attention_prepare_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        positions_ptr,
        q_out_ptr,
        k_out_ptr,
        v_out_ptr,
        k_cache_ptr,
        v_cache_ptr,
        q_stride_batch,
        q_stride_token,
        k_stride_batch,
        k_stride_token,
        v_stride_batch,
        v_stride_token,
        q_out_stride_batch,
        q_out_stride_head,
        q_out_stride_token,
        k_out_stride_batch,
        k_out_stride_head,
        k_out_stride_token,
        v_out_stride_batch,
        v_out_stride_head,
        v_out_stride_token,
        k_cache_stride_batch,
        k_cache_stride_token,
        k_cache_stride_head,
        v_cache_stride_batch,
        v_cache_stride_token,
        v_cache_stride_head,
        cos_stride_pos,
        sin_stride_pos,
        eps,
        NUM_Q_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        SEQ_LEN: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        batch_idx = row_idx // SEQ_LEN
        token_idx = row_idx - batch_idx * SEQ_LEN
        offsets = tl.arange(0, HEAD_DIM)
        half_dim = HEAD_DIM // 2
        pair_offsets = tl.where(offsets < half_dim, offsets + half_dim, offsets - half_dim)
        rope_offsets = tl.where(offsets < half_dim, offsets, offsets - half_dim)

        position = tl.load(positions_ptr + row_idx).to(tl.int64)
        cos = tl.load(cos_ptr + position * cos_stride_pos + rope_offsets)
        sin = tl.load(sin_ptr + position * sin_stride_pos + rope_offsets)
        # The current reference path materializes BF16 RMSNorm output and BF16
        # RoPE tables before rotating. Preserve those rounding points.
        cos = cos.to(tl.bfloat16).to(tl.float32)
        sin = sin.to(tl.bfloat16).to(tl.float32)

        q_base = (
            q_ptr
            + batch_idx * q_stride_batch
            + token_idx * q_stride_token
            + head_idx * HEAD_DIM
        )
        q = tl.load(q_base + offsets).to(tl.float32)
        q_var = tl.sum(q * q, axis=0) / HEAD_DIM
        q_inv = tl.rsqrt(q_var + eps)
        q_weight = tl.load(q_weight_ptr + offsets).to(tl.float32)
        q_norm = (q * q_inv * q_weight).to(tl.bfloat16).to(tl.float32)
        q_pair_raw = tl.load(q_base + pair_offsets).to(tl.float32)
        q_pair_weight = tl.load(q_weight_ptr + pair_offsets).to(tl.float32)
        q_pair = (q_pair_raw * q_inv * q_pair_weight).to(tl.bfloat16).to(tl.float32)
        q_rotated = q_norm * cos + tl.where(offsets < half_dim, -q_pair, q_pair) * sin
        q_out_base = (
            q_out_ptr
            + batch_idx * q_out_stride_batch
            + head_idx * q_out_stride_head
            + token_idx * q_out_stride_token
        )
        tl.store(q_out_base + offsets, q_rotated)

        kv_mask = head_idx < NUM_KV_HEADS
        k_base = (
            k_ptr
            + batch_idx * k_stride_batch
            + token_idx * k_stride_token
            + head_idx * HEAD_DIM
        )
        k = tl.load(k_base + offsets, mask=kv_mask, other=0.0).to(tl.float32)
        k_var = tl.sum(k * k, axis=0) / HEAD_DIM
        k_inv = tl.rsqrt(k_var + eps)
        k_weight = tl.load(k_weight_ptr + offsets, mask=kv_mask, other=0.0).to(tl.float32)
        k_norm = (k * k_inv * k_weight).to(tl.bfloat16).to(tl.float32)
        k_pair_raw = tl.load(k_base + pair_offsets, mask=kv_mask, other=0.0).to(tl.float32)
        k_pair_weight = tl.load(
            k_weight_ptr + pair_offsets, mask=kv_mask, other=0.0
        ).to(tl.float32)
        k_pair = (k_pair_raw * k_inv * k_pair_weight).to(tl.bfloat16).to(tl.float32)
        k_rotated = k_norm * cos + tl.where(offsets < half_dim, -k_pair, k_pair) * sin

        v_base = (
            v_ptr
            + batch_idx * v_stride_batch
            + token_idx * v_stride_token
            + head_idx * HEAD_DIM
        )
        v = tl.load(v_base + offsets, mask=kv_mask, other=0.0).to(tl.float32)
        v_var = tl.sum(v * v, axis=0) / HEAD_DIM
        v_norm = v * tl.rsqrt(v_var + eps)

        k_out_base = (
            k_out_ptr
            + batch_idx * k_out_stride_batch
            + head_idx * k_out_stride_head
            + token_idx * k_out_stride_token
        )
        v_out_base = (
            v_out_ptr
            + batch_idx * v_out_stride_batch
            + head_idx * v_out_stride_head
            + token_idx * v_out_stride_token
        )
        k_cache_base = (
            k_cache_ptr
            + batch_idx * k_cache_stride_batch
            + token_idx * k_cache_stride_token
            + head_idx * k_cache_stride_head
        )
        v_cache_base = (
            v_cache_ptr
            + batch_idx * v_cache_stride_batch
            + token_idx * v_cache_stride_token
            + head_idx * v_cache_stride_head
        )
        tl.store(k_out_base + offsets, k_rotated, mask=kv_mask)
        tl.store(v_out_base + offsets, v_norm, mask=kv_mask)
        tl.store(k_cache_base + offsets, k_rotated, mask=kv_mask)
        tl.store(v_cache_base + offsets, v_norm, mask=kv_mask)


    @triton.jit
    def _gemma4_prefill_q_prepare_kernel(
        q_ptr,
        q_weight_ptr,
        cos_ptr,
        sin_ptr,
        positions_ptr,
        q_out_ptr,
        q_stride_batch,
        q_stride_token,
        q_out_stride_batch,
        q_out_stride_head,
        q_out_stride_token,
        cos_stride_pos,
        sin_stride_pos,
        eps,
        NUM_Q_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        SEQ_LEN: tl.constexpr,
    ):
        """E2B GQA frontend: Q-only half of the split launch."""
        row_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        batch_idx = row_idx // SEQ_LEN
        token_idx = row_idx - batch_idx * SEQ_LEN
        offsets = tl.arange(0, HEAD_DIM)
        half_dim = HEAD_DIM // 2
        pair_offsets = tl.where(
            offsets < half_dim, offsets + half_dim, offsets - half_dim
        )
        rope_offsets = tl.where(offsets < half_dim, offsets, offsets - half_dim)

        position = tl.load(positions_ptr + row_idx).to(tl.int64)
        cos = tl.load(cos_ptr + position * cos_stride_pos + rope_offsets)
        sin = tl.load(sin_ptr + position * sin_stride_pos + rope_offsets)
        cos = cos.to(tl.bfloat16).to(tl.float32)
        sin = sin.to(tl.bfloat16).to(tl.float32)

        q_base = (
            q_ptr
            + batch_idx * q_stride_batch
            + token_idx * q_stride_token
            + head_idx * HEAD_DIM
        )
        q = tl.load(q_base + offsets).to(tl.float32)
        q_var = tl.sum(q * q, axis=0) / HEAD_DIM
        q_inv = tl.rsqrt(q_var + eps)
        q_weight = tl.load(q_weight_ptr + offsets).to(tl.float32)
        q_norm = (q * q_inv * q_weight).to(tl.bfloat16).to(tl.float32)
        q_pair_raw = tl.load(q_base + pair_offsets).to(tl.float32)
        q_pair_weight = tl.load(q_weight_ptr + pair_offsets).to(tl.float32)
        q_pair = (q_pair_raw * q_inv * q_pair_weight).to(tl.bfloat16).to(tl.float32)
        q_rotated = (
            q_norm * cos
            + tl.where(offsets < half_dim, -q_pair, q_pair) * sin
        )
        q_out_base = (
            q_out_ptr
            + batch_idx * q_out_stride_batch
            + head_idx * q_out_stride_head
            + token_idx * q_out_stride_token
        )
        tl.store(q_out_base + offsets, q_rotated)


    @triton.jit
    def _gemma4_prefill_kv_prepare_kernel(
        k_ptr,
        v_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        positions_ptr,
        k_out_ptr,
        v_out_ptr,
        k_cache_ptr,
        v_cache_ptr,
        k_stride_batch,
        k_stride_token,
        v_stride_batch,
        v_stride_token,
        k_out_stride_batch,
        k_out_stride_head,
        k_out_stride_token,
        v_out_stride_batch,
        v_out_stride_head,
        v_out_stride_token,
        k_cache_stride_batch,
        k_cache_stride_token,
        k_cache_stride_head,
        v_cache_stride_batch,
        v_cache_stride_token,
        v_cache_stride_head,
        cos_stride_pos,
        sin_stride_pos,
        eps,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        SEQ_LEN: tl.constexpr,
    ):
        """E2B GQA frontend: execute K/V only for the physical KV heads."""
        row_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        batch_idx = row_idx // SEQ_LEN
        token_idx = row_idx - batch_idx * SEQ_LEN
        offsets = tl.arange(0, HEAD_DIM)
        half_dim = HEAD_DIM // 2
        pair_offsets = tl.where(
            offsets < half_dim, offsets + half_dim, offsets - half_dim
        )
        rope_offsets = tl.where(offsets < half_dim, offsets, offsets - half_dim)

        position = tl.load(positions_ptr + row_idx).to(tl.int64)
        cos = tl.load(cos_ptr + position * cos_stride_pos + rope_offsets)
        sin = tl.load(sin_ptr + position * sin_stride_pos + rope_offsets)
        cos = cos.to(tl.bfloat16).to(tl.float32)
        sin = sin.to(tl.bfloat16).to(tl.float32)

        k_base = (
            k_ptr
            + batch_idx * k_stride_batch
            + token_idx * k_stride_token
            + head_idx * HEAD_DIM
        )
        k = tl.load(k_base + offsets).to(tl.float32)
        k_var = tl.sum(k * k, axis=0) / HEAD_DIM
        k_inv = tl.rsqrt(k_var + eps)
        k_weight = tl.load(k_weight_ptr + offsets).to(tl.float32)
        k_norm = (k * k_inv * k_weight).to(tl.bfloat16).to(tl.float32)
        k_pair_raw = tl.load(k_base + pair_offsets).to(tl.float32)
        k_pair_weight = tl.load(k_weight_ptr + pair_offsets).to(tl.float32)
        k_pair = (k_pair_raw * k_inv * k_pair_weight).to(tl.bfloat16).to(tl.float32)
        k_rotated = (
            k_norm * cos
            + tl.where(offsets < half_dim, -k_pair, k_pair) * sin
        )

        v_base = (
            v_ptr
            + batch_idx * v_stride_batch
            + token_idx * v_stride_token
            + head_idx * HEAD_DIM
        )
        v = tl.load(v_base + offsets).to(tl.float32)
        v_var = tl.sum(v * v, axis=0) / HEAD_DIM
        v_norm = v * tl.rsqrt(v_var + eps)

        k_out_base = (
            k_out_ptr
            + batch_idx * k_out_stride_batch
            + head_idx * k_out_stride_head
            + token_idx * k_out_stride_token
        )
        v_out_base = (
            v_out_ptr
            + batch_idx * v_out_stride_batch
            + head_idx * v_out_stride_head
            + token_idx * v_out_stride_token
        )
        k_cache_base = (
            k_cache_ptr
            + batch_idx * k_cache_stride_batch
            + token_idx * k_cache_stride_token
            + head_idx * k_cache_stride_head
        )
        v_cache_base = (
            v_cache_ptr
            + batch_idx * v_cache_stride_batch
            + token_idx * v_cache_stride_token
            + head_idx * v_cache_stride_head
        )
        tl.store(k_out_base + offsets, k_rotated)
        tl.store(v_out_base + offsets, v_norm)
        tl.store(k_cache_base + offsets, k_rotated)
        tl.store(v_cache_base + offsets, v_norm)


def gemma4_prefill_attention_prepare(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    num_warps: int = 4,
    num_stages: int = 2,
    split_qkv: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare supported Gemma4 prefill tensors with fused or split launches."""
    if not _HAS_TRITON:
        raise RuntimeError("Triton is required for fused Gemma4 attention preparation")
    tensors = (q_raw, k_raw, v_raw, q_weight, k_weight, cos, sin, positions)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("fused Gemma4 attention preparation requires CUDA tensors")
    if q_raw.dtype != torch.bfloat16 or k_raw.dtype != torch.bfloat16 or v_raw.dtype != torch.bfloat16:
        raise ValueError("fused Gemma4 attention preparation currently requires BF16")
    batch_size = int(q_raw.shape[0])
    seq_len = int(q_raw.shape[1])
    if tuple(k_raw.shape[:2]) != (batch_size, seq_len) or tuple(v_raw.shape[:2]) != (
        batch_size,
        seq_len,
    ):
        raise ValueError("Q/K/V batch and sequence dimensions must match")
    if int(q_raw.shape[-1]) != int(num_q_heads) * int(head_dim):
        raise ValueError("Q projection width does not match the requested head layout")
    if int(k_raw.shape[-1]) != int(num_kv_heads) * int(head_dim):
        raise ValueError("K projection width does not match the requested head layout")
    if int(v_raw.shape[-1]) != int(num_kv_heads) * int(head_dim):
        raise ValueError("V projection width does not match the requested head layout")
    if any(tensor.stride(-1) != 1 for tensor in (q_raw, k_raw, v_raw, cos, sin)):
        raise ValueError("fused Gemma4 attention preparation requires contiguous inner dimensions")
    if int(q_weight.numel()) != int(head_dim) or int(k_weight.numel()) != int(head_dim):
        raise ValueError("Q/K RMSNorm weights must match head_dim")
    if int(num_warps) not in (2, 4, 8):
        raise ValueError("num_warps must be one of 2, 4, or 8")
    if int(num_stages) not in (1, 2, 3, 4):
        raise ValueError("num_stages must be between 1 and 4")

    positions_1d = positions.reshape(-1)
    if int(positions_1d.numel()) != batch_size * seq_len:
        raise ValueError("positions must contain exactly one entry per prefill token")

    expected_shapes = (
        (q_out, (batch_size, int(num_q_heads), seq_len, int(head_dim))),
        (k_out, (batch_size, int(num_kv_heads), seq_len, int(head_dim))),
        (v_out, (batch_size, int(num_kv_heads), seq_len, int(head_dim))),
        (k_cache, (batch_size, seq_len, int(num_kv_heads), int(head_dim))),
        (v_cache, (batch_size, seq_len, int(num_kv_heads), int(head_dim))),
    )
    for tensor, shape in expected_shapes:
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != q_raw.dtype
            or not tensor.is_cuda
            or tensor.stride(-1) != 1
        ):
            raise ValueError(f"invalid fused attention output buffer; expected {shape}")

    common_outputs = (
        q_out,
        k_out,
        v_out,
        k_cache,
        v_cache,
    )
    if split_qkv:
        _gemma4_prefill_q_prepare_kernel[
            (batch_size * seq_len, int(num_q_heads))
        ](
            q_raw,
            q_weight,
            cos,
            sin,
            positions_1d,
            q_out,
            q_raw.stride(0),
            q_raw.stride(1),
            q_out.stride(0),
            q_out.stride(1),
            q_out.stride(2),
            cos.stride(0),
            sin.stride(0),
            float(eps),
            NUM_Q_HEADS=int(num_q_heads),
            HEAD_DIM=int(head_dim),
            SEQ_LEN=seq_len,
            num_warps=int(num_warps),
            num_stages=int(num_stages),
        )
        _gemma4_prefill_kv_prepare_kernel[
            (batch_size * seq_len, int(num_kv_heads))
        ](
            k_raw,
            v_raw,
            k_weight,
            cos,
            sin,
            positions_1d,
            k_out,
            v_out,
            k_cache,
            v_cache,
            k_raw.stride(0),
            k_raw.stride(1),
            v_raw.stride(0),
            v_raw.stride(1),
            k_out.stride(0),
            k_out.stride(1),
            k_out.stride(2),
            v_out.stride(0),
            v_out.stride(1),
            v_out.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            cos.stride(0),
            sin.stride(0),
            float(eps),
            NUM_KV_HEADS=int(num_kv_heads),
            HEAD_DIM=int(head_dim),
            SEQ_LEN=seq_len,
            num_warps=int(num_warps),
            num_stages=int(num_stages),
        )
        return common_outputs

    _gemma4_prefill_attention_prepare_kernel[(batch_size * seq_len, int(num_q_heads))](
        q_raw,
        k_raw,
        v_raw,
        q_weight,
        k_weight,
        cos,
        sin,
        positions_1d,
        q_out,
        k_out,
        v_out,
        k_cache,
        v_cache,
        q_raw.stride(0),
        q_raw.stride(1),
        k_raw.stride(0),
        k_raw.stride(1),
        v_raw.stride(0),
        v_raw.stride(1),
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        v_out.stride(0),
        v_out.stride(1),
        v_out.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        cos.stride(0),
        sin.stride(0),
        float(eps),
        NUM_Q_HEADS=int(num_q_heads),
        NUM_KV_HEADS=int(num_kv_heads),
        HEAD_DIM=int(head_dim),
        SEQ_LEN=seq_len,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return common_outputs


HAS_GEMMA4_ATTENTION_PREPARE = _HAS_TRITON


__all__ = [
    "gemma4_prefill_attention_prepare",
    "HAS_GEMMA4_ATTENTION_PREPARE",
]
