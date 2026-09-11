"""
SwiGLU Triton Module
--------------------
High-performance fused SwiGLU activation with Triton.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch.autograd import Function


# =============================================================================
# Triton Kernels
# =============================================================================

@triton.jit
def _mg_swiglu_fwd_kernel(
    input_ptr,       # [Batch*Seq, 2*H] (Gate + Value contíguos)
    output_ptr,      # [Batch*Seq, H]
    n_cols_half,     # H (Hidden Dim)
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    row_input_ptr = input_ptr + pid * (2 * n_cols_half)
    row_output_ptr = output_ptr + pid * n_cols_half

    for off in range(0, n_cols_half, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols_half

        # Load in FP32 for stability
        gate = tl.load(row_input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        val  = tl.load(row_input_ptr + n_cols_half + offsets, mask=mask, other=0.0).to(tl.float32)

        # Fused SiLU(gate) * val
        gate_sig = tl.sigmoid(gate)
        gate_silu = gate * gate_sig
        out = gate_silu * val

        tl.store(row_output_ptr + offsets, out, mask=mask)


@triton.jit
def _mg_gated_activation_fwd_kernel(
    input_ptr,
    output_ptr,
    ELEMENTS: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < ELEMENTS
    rows = offsets // HIDDEN_DIM
    cols = offsets - rows * HIDDEN_DIM
    gate = tl.load(
        input_ptr + rows * (2 * HIDDEN_DIM) + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    value = tl.load(
        input_ptr + rows * (2 * HIDDEN_DIM) + HIDDEN_DIM + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if ACT == 1:
        inner = 0.7978845608028654 * (
            gate + 0.044715 * gate * gate * gate
        )
        # Match PyTorch's `gelu(..., approximate="tanh")` implementation,
        # including the BF16 materialization before the in-place multiply.
        activated = 0.5 * gate * (1.0 + libdevice.tanh(inner))
        activated = activated.to(tl.bfloat16).to(tl.float32)
    else:
        activated = gate * tl.sigmoid(gate)
    tl.store(output_ptr + offsets, activated * value, mask=mask)


@triton.jit
def _mg_conditioned_gelu_tanh_fwd_kernel(
    gate_ptr,
    condition_ptr,
    output_ptr,
    ELEMENTS: tl.constexpr,
    WIDTH: tl.constexpr,
    GATE_STRIDE_ROW: tl.constexpr,
    GATE_STRIDE_COL: tl.constexpr,
    CONDITION_STRIDE_ROW: tl.constexpr,
    CONDITION_STRIDE_COL: tl.constexpr,
    OUTPUT_STRIDE_ROW: tl.constexpr,
    OUTPUT_STRIDE_COL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuse the Gemma 4 PLE GELU-tanh and per-layer conditioning multiply.

    The BF16 cast between GELU and multiply is intentional.  PyTorch's current
    path materializes the GELU result into a BF16 tensor before ``mul_``; doing
    the same cast in-register preserves that operation boundary without the
    extra global-memory round trip or kernel launch.
    """
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < ELEMENTS
    rows = offsets // WIDTH
    cols = offsets - rows * WIDTH
    gate = tl.load(
        gate_ptr + rows * GATE_STRIDE_ROW + cols * GATE_STRIDE_COL,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    condition = tl.load(
        condition_ptr
        + rows * CONDITION_STRIDE_ROW
        + cols * CONDITION_STRIDE_COL,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    activated_bf16 = (
        0.5 * gate * (1.0 + libdevice.tanh(inner))
    ).to(tl.bfloat16)
    tl.store(
        output_ptr + rows * OUTPUT_STRIDE_ROW + cols * OUTPUT_STRIDE_COL,
        activated_bf16.to(tl.float32) * condition,
        mask=mask,
    )


@triton.jit
def _mg_swiglu_bwd_kernel(
    grad_out_ptr,    # [M, H]
    input_ptr,       # [M, 2H]
    grad_input_ptr,  # [M, 2H]
    n_cols_half,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    row_grad_out = grad_out_ptr + pid * n_cols_half
    row_input = input_ptr + pid * (2 * n_cols_half)
    row_grad_input = grad_input_ptr + pid * (2 * n_cols_half)

    for off in range(0, n_cols_half, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols_half

        g_out = tl.load(row_grad_out + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(row_input + offsets, mask=mask, other=0.0).to(tl.float32)
        val  = tl.load(row_input + n_cols_half + offsets, mask=mask, other=0.0).to(tl.float32)

        # Recompute activation
        sig_gate = tl.sigmoid(gate)
        silu_gate = gate * sig_gate

        # Gradients
        d_val = g_out * silu_gate
        term = 1.0 + gate * (1.0 - sig_gate)
        d_silu = sig_gate * term
        d_gate = g_out * val * d_silu

        tl.store(row_grad_input + offsets, d_gate, mask=mask)
        tl.store(row_grad_input + n_cols_half + offsets, d_val, mask=mask)


# =============================================================================
# Autograd Function
# =============================================================================

class MegaGemmFunction(Function):
    """Autograd function for fused SwiGLU Triton kernel."""

    @staticmethod
    def forward(ctx, w12_out: torch.Tensor, hidden_dim: int):
        w12_out = w12_out.contiguous()
        x_flat = w12_out.view(-1, 2 * hidden_dim)
        M = x_flat.shape[0]

        out_flat = torch.empty((M, hidden_dim), device=w12_out.device, dtype=w12_out.dtype)

        ctx.save_for_backward(w12_out)
        ctx.hidden_dim = hidden_dim

        grid = (M,)
        BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 1024)

        _mg_swiglu_fwd_kernel[grid](
            x_flat, out_flat,
            hidden_dim,
            BLOCK_SIZE=BLOCK_SIZE
        )

        return out_flat.view(w12_out.shape[:-1] + (hidden_dim,))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        w12_out, = ctx.saved_tensors
        hidden_dim = ctx.hidden_dim

        grad_out_flat = grad_output.contiguous().view(-1, hidden_dim)
        x_flat = w12_out.contiguous().view(-1, 2 * hidden_dim)
        M = x_flat.shape[0]

        grad_input_flat = torch.empty_like(x_flat)

        grid = (M,)
        BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 1024)

        _mg_swiglu_bwd_kernel[grid](
            grad_out_flat, x_flat, grad_input_flat,
            hidden_dim,
            BLOCK_SIZE=BLOCK_SIZE
        )

        return grad_input_flat.view(w12_out.shape), None


def swiglu_forward(
    w12_out: torch.Tensor,
    hidden_dim: int,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """
    Inference-only fast path that bypasses autograd.Function wrapper overhead.
    """
    w12_out = w12_out.contiguous()
    x_flat = w12_out.view(-1, 2 * hidden_dim)
    m_rows = x_flat.shape[0]
    expected_shape = w12_out.shape[:-1] + (hidden_dim,)
    if out is None:
        out_flat = torch.empty((m_rows, hidden_dim), device=w12_out.device, dtype=w12_out.dtype)
    else:
        if tuple(out.shape) != tuple(expected_shape):
            raise ValueError(
                f"out shape mismatch: got {tuple(out.shape)} expected {tuple(expected_shape)}"
            )
        if out.device != w12_out.device or out.dtype != w12_out.dtype:
            raise ValueError("out must match input device and dtype")
        out_flat = out if out.ndim == 2 else out.view(-1, hidden_dim)

    grid = (m_rows,)
    block_size = min(triton.next_power_of_2(hidden_dim), 1024)
    _mg_swiglu_fwd_kernel[grid](
        x_flat, out_flat,
        hidden_dim,
        BLOCK_SIZE=block_size,
    )
    if out is not None:
        return out
    return out_flat.view(expected_shape)


def gated_activation_forward(
    gate_up: torch.Tensor,
    hidden_dim: int,
    *,
    activation: str = "silu",
    out: torch.Tensor = None,
    block_size: int = 512,
) -> torch.Tensor:
    """Inference-only fused SiLU/GELU-tanh gate multiplied by the up branch."""
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if gate_up.shape[-1] != 2 * hidden_dim:
        raise ValueError(
            "gate_up last dimension mismatch: "
            f"got {gate_up.shape[-1]} expected {2 * hidden_dim}"
        )
    if not gate_up.is_cuda:
        raise ValueError("gated_activation_forward requires a CUDA tensor")
    if block_size not in (128, 256, 512, 1024):
        raise ValueError("block_size must be one of 128, 256, 512, 1024")

    normalized_activation = str(activation).strip().lower()
    if normalized_activation in {"gelu", "gelu_tanh", "gelu_pytorch_tanh", "geglu"}:
        act_id = 1
    elif normalized_activation in {"silu", "swiglu"}:
        act_id = 0
    else:
        raise ValueError(f"unsupported gated activation: {activation!r}")

    gate_up = gate_up.contiguous()
    flat_input = gate_up.view(-1, 2 * hidden_dim)
    expected_shape = gate_up.shape[:-1] + (hidden_dim,)
    if out is None:
        flat_output = torch.empty(
            (flat_input.shape[0], hidden_dim),
            device=gate_up.device,
            dtype=gate_up.dtype,
        )
    else:
        if tuple(out.shape) != tuple(expected_shape):
            raise ValueError(
                f"out shape mismatch: got {tuple(out.shape)} expected {tuple(expected_shape)}"
            )
        if out.device != gate_up.device or out.dtype != gate_up.dtype:
            raise ValueError("out must match input device and dtype")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        flat_output = out.view(-1, hidden_dim)

    elements = int(flat_input.shape[0]) * int(hidden_dim)
    _mg_gated_activation_fwd_kernel[(triton.cdiv(elements, block_size),)](
        flat_input,
        flat_output,
        ELEMENTS=elements,
        HIDDEN_DIM=int(hidden_dim),
        ACT=act_id,
        BLOCK_SIZE=int(block_size),
        num_warps=4,
        num_stages=1,
    )
    if out is not None:
        return out
    return flat_output.view(expected_shape)


def conditioned_gelu_tanh_forward(
    gate: torch.Tensor,
    condition: torch.Tensor,
    *,
    out: torch.Tensor = None,
    block_size: int = 256,
) -> torch.Tensor:
    """Fuse ``GELU-tanh(gate) * condition`` for the Gemma 4 BF16 PLE tail.

    ``condition`` may be a strided ``[batch, width]`` view into Gemma 4's
    ``[batch, 1, layers, width]`` per-layer input tensor.  Supporting its row
    stride directly is what keeps this path copy-free.
    """
    if gate.ndim != 2 or condition.ndim != 2:
        raise ValueError("gate and condition must both be rank-2 tensors")
    if tuple(gate.shape) != tuple(condition.shape):
        raise ValueError(
            "gate and condition shape mismatch: "
            f"got {tuple(gate.shape)} and {tuple(condition.shape)}"
        )
    if not gate.is_cuda or not condition.is_cuda:
        raise ValueError("conditioned_gelu_tanh_forward requires CUDA tensors")
    if gate.device != condition.device:
        raise ValueError("gate and condition must be on the same device")
    if gate.dtype != torch.bfloat16 or condition.dtype != torch.bfloat16:
        raise ValueError("conditioned_gelu_tanh_forward requires BF16 tensors")
    if gate.stride(1) != 1 or condition.stride(1) != 1:
        raise ValueError("gate and condition must have a contiguous last dimension")
    if block_size not in (128, 256, 512, 1024):
        raise ValueError("block_size must be one of 128, 256, 512, 1024")

    if out is None:
        output = torch.empty_like(gate, memory_format=torch.contiguous_format)
    else:
        if tuple(out.shape) != tuple(gate.shape):
            raise ValueError(
                f"out shape mismatch: got {tuple(out.shape)} expected {tuple(gate.shape)}"
            )
        if out.device != gate.device or out.dtype != gate.dtype:
            raise ValueError("out must match gate device and dtype")
        if out.ndim != 2 or out.stride(1) != 1:
            raise ValueError("out must be rank-2 with a contiguous last dimension")
        output = out

    rows, width = map(int, gate.shape)
    elements = rows * width
    if elements == 0:
        return output
    _mg_conditioned_gelu_tanh_fwd_kernel[
        (triton.cdiv(elements, block_size),)
    ](
        gate,
        condition,
        output,
        ELEMENTS=elements,
        WIDTH=width,
        GATE_STRIDE_ROW=int(gate.stride(0)),
        GATE_STRIDE_COL=int(gate.stride(1)),
        CONDITION_STRIDE_ROW=int(condition.stride(0)),
        CONDITION_STRIDE_COL=int(condition.stride(1)),
        OUTPUT_STRIDE_ROW=int(output.stride(0)),
        OUTPUT_STRIDE_COL=int(output.stride(1)),
        BLOCK_SIZE=int(block_size),
        num_warps=4 if block_size >= 256 else 2,
        num_stages=1,
    )
    return output


# =============================================================================
# NN Module
# =============================================================================

class MegaGemmTriton(nn.Module):
    """
    Mega Gemm Triton: High-performance fused SwiGLU.

    Drop-in replacement for standard SwiGLU (gate + value activation) with
    fused Triton kernel that avoids memory copies.

    Args:
        d_model (int): Input/output dimension.
        multiple_of (int): Ensure hidden dim is multiple of this (default 256).
        hidden_multiple (float): Expansion factor (default 5/3 like LLaMA).

    Example:
        >>> swiglu = MegaGemmTriton(4096).cuda()
        >>> x = torch.randn(32, 128, 4096, device='cuda')
        >>> y = swiglu(x)  # [32, 128, 4096]
    """

    def __init__(self, d_model: int, multiple_of: int = 256, hidden_multiple: float = 5/3):
        super().__init__()
        hidden = int(d_model * hidden_multiple)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.hidden = hidden

        # Fused W1+W2 for better Tensor Core utilization
        self.w12 = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w12_out = self.w12(x)
        hidden = MegaGemmFunction.apply(w12_out, self.hidden)
        return self.w3(hidden)

    def extra_repr(self) -> str:
        return f"d_model={self.w12.in_features}, hidden={self.hidden} (Triton Accelerated)"
