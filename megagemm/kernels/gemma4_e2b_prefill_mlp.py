"""Shape-gated Triton gate/up + GeGLU prefill kernel for Gemma 4 E2B.

The production path materializes ``[M, 2I]`` from cuBLAS and then reads it
again in the gated activation kernel.  This experimental kernel computes the
two matrix products in one Triton program and stores only ``[M, I]`` after the
BF16-staged GELU-tanh multiply.  It is intentionally opt-in until a loaded
model gate proves a wall-time win for each sequence shape.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAS_GEMMA4_E2B_PREFILL_FUSED_GATEUP = True
except Exception:  # pragma: no cover - CPU-only installations
    triton = None
    tl = None
    libdevice = None
    HAS_GEMMA4_E2B_PREFILL_FUSED_GATEUP = False


if HAS_GEMMA4_E2B_PREFILL_FUSED_GATEUP:

    @triton.jit
    def _gemma4_e2b_fused_gateup_geglu_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        m_rows: tl.constexpr,
        intermediate: tl.constexpr,
        hidden: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xk: tl.constexpr,
        stride_wn: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_om: tl.constexpr,
        stride_on: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(m_rows, BLOCK_M)
        num_pid_n = tl.cdiv(intermediate, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, hidden, BLOCK_K):
            k = k_start + offs_k
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
                mask=(offs_m[:, None] < m_rows) & (k[None, :] < hidden),
                other=0.0,
            )
            gate_weight = tl.load(
                weight_ptr
                + offs_n[None, :] * stride_wn
                + k[:, None] * stride_wk,
                mask=(offs_n[None, :] < intermediate) & (k[:, None] < hidden),
                other=0.0,
            )
            up_weight = tl.load(
                weight_ptr
                + (offs_n[None, :] + intermediate) * stride_wn
                + k[:, None] * stride_wk,
                mask=(offs_n[None, :] < intermediate) & (k[:, None] < hidden),
                other=0.0,
            )
            gate_acc += tl.dot(x, gate_weight, out_dtype=tl.float32)
            up_acc += tl.dot(x, up_weight, out_dtype=tl.float32)

        # Match the two BF16 materialization boundaries in the production
        # torch.mm -> GELU-tanh -> multiply path.  The GEMM reduction order is
        # allowed to differ and is checked by the full-model frontier.
        gate = gate_acc.to(tl.bfloat16).to(tl.float32)
        up = up_acc.to(tl.bfloat16).to(tl.float32)
        inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
        gelu = (0.5 * gate * (1.0 + libdevice.tanh(inner))).to(tl.bfloat16)
        activated = gelu.to(tl.float32) * up
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            activated,
            mask=(offs_m[:, None] < m_rows) & (offs_n[None, :] < intermediate),
        )


def gemma4_e2b_prefill_fused_gateup_geglu(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    intermediate_size: int,
    out: torch.Tensor | None = None,
    block_m: int = 64,
    block_n: int = 32,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
    group_m: int = 8,
) -> torch.Tensor:
    """Return ``GELU(x@Wg.T) * (x@Wu.T)`` without a ``[M,2I]`` tensor."""
    if not HAS_GEMMA4_E2B_PREFILL_FUSED_GATEUP:
        raise RuntimeError("Triton fused Gemma4 E2B prefill gate-up is unavailable")
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("fused Gemma4 E2B prefill gate-up requires CUDA tensors")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("fused Gemma4 E2B prefill gate-up requires BF16")
    if x.ndim not in (2, 3) or weight.ndim != 2:
        raise ValueError("expected x [M,H] or [B,S,H] and weight [2I,H]")
    intermediate = int(intermediate_size)
    if intermediate <= 0 or tuple(weight.shape) != (2 * intermediate, int(x.shape[-1])):
        raise ValueError("gate-up weight shape must be [2 * intermediate_size, hidden]")
    if x.stride(-1) != 1 or weight.stride(1) != 1:
        raise ValueError("x and gate-up weight must be contiguous in their last dimension")
    valid_tiles = {32, 64, 128}
    if block_m not in valid_tiles or block_n not in valid_tiles or block_k not in (32, 64):
        raise ValueError("unsupported Triton tile")
    if num_warps not in (4, 8) or num_stages not in (2, 3, 4):
        raise ValueError("unsupported Triton launch geometry")

    expected_shape = tuple(x.shape[:-1]) + (intermediate,)
    if out is None:
        out = torch.empty(expected_shape, device=x.device, dtype=x.dtype)
    elif tuple(out.shape) != expected_shape or out.device != x.device or out.dtype != x.dtype:
        raise ValueError("out must match x leading dimensions, device, and dtype")
    if out.stride(-1) != 1:
        raise ValueError("out must be contiguous in its last dimension")

    x_2d = x.reshape(-1, int(x.shape[-1]))
    out_2d = out.reshape(-1, intermediate)
    m_rows = int(x_2d.shape[0])
    hidden = int(x_2d.shape[1])
    grid = (triton.cdiv(m_rows, block_m) * triton.cdiv(intermediate, block_n),)
    _gemma4_e2b_fused_gateup_geglu_kernel[grid](
        x_2d,
        weight,
        out_2d,
        m_rows,
        intermediate,
        hidden,
        x_2d.stride(0),
        x_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        BLOCK_M=int(block_m),
        BLOCK_N=int(block_n),
        BLOCK_K=int(block_k),
        GROUP_M=int(group_m),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out

