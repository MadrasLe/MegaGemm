"""
Decode-oriented DeepFusion MLP kernel.

Fuses gated activation + down projection into a single Triton kernel:
  out = (act(gate) * up) @ down_weight.T + down_bias

Supported activations:
  - silu
  - gelu_tanh

Input `gate_up` is expected as [..., 2 * intermediate_size].
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


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


_CFG_SHAPE_GUARD = _env_bool("MEGAGEMM_DEEPFUSION_SHAPE_GUARD", True)
_CFG_FORCE_TRITON = _env_bool("MEGAGEMM_DEEPFUSION_FORCE_TRITON", False)
_CFG_MAX_ROWS = max(1, _env_int("MEGAGEMM_DEEPFUSION_MAX_ROWS", 8))
_CFG_MIN_I = max(1, _env_int("MEGAGEMM_DEEPFUSION_MIN_I", 1024))
_CFG_MIN_H = max(1, _env_int("MEGAGEMM_DEEPFUSION_MIN_H", 1024))
_CFG_FORCED_BN = _env_int("MEGAGEMM_DEEPFUSION_BLOCK_N", 0)
_CFG_FORCED_BK = _env_int("MEGAGEMM_DEEPFUSION_BLOCK_K", 0)
_CFG_FORCED_WARPS = _env_int("MEGAGEMM_DEEPFUSION_NUM_WARPS", 0)
_CFG_FORCED_STAGES = _env_int("MEGAGEMM_DEEPFUSION_NUM_STAGES", 0)
_CFG_PREFILL_ENABLED = _env_bool("MEGAGEMM_DEEPFUSION_PREFILL", False)
_CFG_PREFILL_FORCE_TRITON = _env_bool("MEGAGEMM_DEEPFUSION_PREFILL_FORCE_TRITON", False)
_CFG_PREFILL_MIN_ROWS = max(1, _env_int("MEGAGEMM_DEEPFUSION_PREFILL_MIN_ROWS", 128))
_CFG_PREFILL_BLOCK_M = _env_int("MEGAGEMM_DEEPFUSION_PREFILL_BLOCK_M", 0)
_CFG_PREFILL_BLOCK_N = _env_int("MEGAGEMM_DEEPFUSION_PREFILL_BLOCK_N", 0)
_CFG_PREFILL_BLOCK_K = _env_int("MEGAGEMM_DEEPFUSION_PREFILL_BLOCK_K", 0)
_CFG_PREFILL_NUM_WARPS = _env_int("MEGAGEMM_DEEPFUSION_PREFILL_NUM_WARPS", 0)
_CFG_PREFILL_NUM_STAGES = _env_int("MEGAGEMM_DEEPFUSION_PREFILL_NUM_STAGES", 0)
_CFG_PREFILL_GROUP_M = max(1, _env_int("MEGAGEMM_DEEPFUSION_PREFILL_GROUP_M", 8))

_ACT_SILU = 0
_ACT_GELU_TANH = 1


def _normalize_activation(activation: str) -> int:
    act = str(activation).strip().lower()
    if act in {"silu", "swiglu"}:
        return _ACT_SILU
    if act in {"gelu", "gelu_tanh", "gelu_pytorch_tanh", "geglu"}:
        return _ACT_GELU_TANH
    raise ValueError(f"Unsupported deepfusion activation: {activation}")


def _apply_activation_torch(gate: torch.Tensor, up: torch.Tensor, activation: str) -> torch.Tensor:
    act_id = _normalize_activation(activation)
    if act_id == _ACT_GELU_TANH:
        return F.gelu(gate, approximate="tanh") * up
    return F.silu(gate) * up


def _prefill_autotune_configs():
    if (
        _CFG_PREFILL_BLOCK_M > 0
        and _CFG_PREFILL_BLOCK_N > 0
        and _CFG_PREFILL_BLOCK_K > 0
        and _CFG_PREFILL_NUM_WARPS > 0
    ):
        return [
            triton.Config(
                {
                    "BLOCK_M": _CFG_PREFILL_BLOCK_M,
                    "BLOCK_N": _CFG_PREFILL_BLOCK_N,
                    "BLOCK_K": _CFG_PREFILL_BLOCK_K,
                    "GROUP_M": _CFG_PREFILL_GROUP_M,
                },
                num_warps=_CFG_PREFILL_NUM_WARPS,
                num_stages=max(1, _CFG_PREFILL_NUM_STAGES or 2),
            )
        ]
    return [
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=8,
            num_stages=3,
        ),
    ]


if _HAS_TRITON:
    @triton.autotune(
        configs=_prefill_autotune_configs(),
        key=["M", "I", "H"],
    )
    @triton.jit
    def _deepfusion_swiglu_down_prefill_kernel(
        gate_up_ptr,  # [M, 2I]
        w_ptr,        # [H, I]
        b_ptr,        # [H] or dummy
        y_ptr,        # [M, H]
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        M,
        I,
        H,
        HAS_BIAS: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(H, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        n_mask = offs_n < H

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, I, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < I

            gate_ptrs = gate_up_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            up_ptrs = gate_up_ptr + offs_m[:, None] * stride_xm + (offs_k + I)[None, :] * stride_xk
            x_mask = m_mask[:, None] & k_mask[None, :]

            gate = tl.load(gate_ptrs, mask=x_mask, other=0.0).to(tl.float32)
            up = tl.load(up_ptrs, mask=x_mask, other=0.0).to(tl.float32)
            if ACT == 1:
                c = 0.7978845608028654
                inner = c * (gate + 0.044715 * gate * gate * gate)
                gate_act = gate * tl.sigmoid(2.0 * inner)
            else:
                gate_act = gate * tl.sigmoid(gate)
            act = (gate_act * up).to(tl.float16)

            w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
            w_mask = k_mask[:, None] & n_mask[None, :]
            w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float16)

            acc += tl.dot(act, w)

        if HAS_BIAS:
            bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc += bias[None, :]

        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        y_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)

    @triton.jit
    def _deepfusion_swiglu_down_kernel(
        gate_up_ptr,  # [M, 2I]
        w_ptr,        # [H, I]
        b_ptr,        # [H] or dummy
        r_ptr,        # [M, H] residual or dummy
        y_ptr,        # [M, H]
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_rm, stride_rn,
        stride_ym, stride_yn,
        I,
        H,
        HAS_BIAS: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < H

        base_row = gate_up_ptr + pid_m * stride_xm
        base_gate = base_row
        base_up = base_row + I * stride_xk

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        for k_start in range(0, I, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < I

            gate = tl.load(
                base_gate + offs_k * stride_xk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                base_up + offs_k * stride_xk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            if ACT == 1:
                c = 0.7978845608028654
                inner = c * (gate + 0.044715 * gate * gate * gate)
                gate_act = gate * tl.sigmoid(2.0 * inner)
            else:
                gate_act = gate * tl.sigmoid(gate)
            act = gate_act * up

            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w_mask = n_mask[:, None] & k_mask[None, :]
            w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

            acc += tl.sum(w * act[None, :], axis=1)

        if HAS_BIAS:
            bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc += bias

        if HAS_RESIDUAL:
            r_ptrs = r_ptr + pid_m * stride_rm + offs_n * stride_rn
            residual = tl.load(r_ptrs, mask=n_mask, other=0.0).to(tl.float32)
            acc += residual

        y_ptrs = y_ptr + pid_m * stride_ym + offs_n * stride_yn
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=n_mask)


    @triton.jit
    def _gemma4_e2b_b8_geglu_down_tensorcore_kernel(
        gate_up_ptr,  # [8, 24576]
        w_ptr,        # [1536, 12288]
        b_ptr,        # [1536] or dummy
        y_ptr,        # [8, 1536]
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_ym,
        stride_yn,
        I: tl.constexpr,
        H: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Exact B8 Gemma 4 E2B GeGLU+down Tensor Core tile.

        BLOCK_M is deliberately fixed at 16: Ampere/Ada Tensor Cores require a
        matrix tile even though only eight rows are live.  The two BF16 casts
        reproduce PyTorch's materialized GELU and multiply boundaries before
        the FP32-accumulating down projection.
        """
        block_m: tl.constexpr = 16
        offs_m = tl.arange(0, block_m)
        offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < 8
        n_mask = offs_n < H
        acc = tl.zeros((block_m, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, I, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            gate = tl.load(
                gate_up_ptr
                + offs_m[:, None] * stride_xm
                + offs_k[None, :] * stride_xk,
                mask=m_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                gate_up_ptr
                + offs_m[:, None] * stride_xm
                + (offs_k + I)[None, :] * stride_xk,
                mask=m_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            inner = 0.7978845608028654 * (
                gate + 0.044715 * gate * gate * gate
            )
            gelu_bf16 = (gate * tl.sigmoid(2.0 * inner)).to(tl.bfloat16)
            activated = (gelu_bf16.to(tl.float32) * up).to(tl.bfloat16)
            weight = tl.load(
                w_ptr
                + offs_k[:, None] * stride_wk
                + offs_n[None, :] * stride_wn,
                mask=n_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(activated, weight)

        if HAS_BIAS:
            bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc += bias[None, :]

        tl.store(
            y_ptr
            + offs_m[:, None] * stride_ym
            + offs_n[None, :] * stride_yn,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


def _pick_cfg(i_dim: int, h_dim: int, rows: int):
    if _CFG_FORCED_BN > 0 and _CFG_FORCED_BK > 0 and _CFG_FORCED_WARPS > 0:
        return _CFG_FORCED_BN, _CFG_FORCED_BK, _CFG_FORCED_WARPS, max(1, _CFG_FORCED_STAGES or 2)

    if i_dim >= 8192:
        block_k = 128
    elif i_dim >= 4096:
        block_k = 128
    else:
        block_k = 64

    if h_dim >= 4096:
        block_n = 16
        num_warps = 4
    elif h_dim >= 1024:
        block_n = 32
        num_warps = 4
    else:
        block_n = 64
        num_warps = 4

    if rows > 1 and block_n < 64:
        block_n = min(64, block_n * 2)

    num_stages = 2
    return block_n, block_k, num_warps, num_stages
def _flatten_rows(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x if x.is_contiguous() else x.contiguous()
    x_2d = x.flatten(0, -2)
    return x_2d if x_2d.is_contiguous() else x_2d.contiguous()


def _fallback_swiglu_down(
    gate_up: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    residual: Optional[torch.Tensor],
    activation: str,
) -> torch.Tensor:
    i_dim = gate_up.shape[-1] // 2
    gate = gate_up[..., :i_dim]
    up = gate_up[..., i_dim:]
    activated = _apply_activation_torch(gate, up, activation)
    projected = F.linear(activated, down_weight, down_bias)
    if residual is not None:
        if out is residual:
            out.add_(projected)
            return out
        if out is None:
            return projected + residual
        out.copy_(projected)
        out.add_(residual)
        return out
    if out is None:
        return projected
    out.copy_(projected)
    return out


def deepfusion_mlp_prefers_triton_shape(
    i_dim: int,
    h_dim: int,
    rows: int,
    mode: str = "decode",
) -> bool:
    if mode == "prefill":
        if _CFG_PREFILL_FORCE_TRITON:
            return True
        if not _CFG_PREFILL_ENABLED:
            return False
        if rows < _CFG_PREFILL_MIN_ROWS:
            return False
        if i_dim < _CFG_MIN_I:
            return False
        if h_dim < _CFG_MIN_H:
            return False
        return True
    if _CFG_FORCE_TRITON:
        return True
    if not _CFG_SHAPE_GUARD:
        return True
    if rows > _CFG_MAX_ROWS:
        return False
    if i_dim < _CFG_MIN_I:
        return False
    if h_dim < _CFG_MIN_H:
        return False
    return True


def deepfusion_swiglu_down(
    gate_up: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    mode: str = "decode",
    activation: str = "silu",
) -> torch.Tensor:
    """
    Fused decode MLP tail:
      out = (act(gate_up[..., :I]) * gate_up[..., I:]) @ down_weight.T + down_bias
    """
    if gate_up.shape[-1] % 2 != 0:
        raise ValueError("gate_up last dim must be 2 * intermediate_size")
    act_id = _normalize_activation(activation)
    i_dim = gate_up.shape[-1] // 2
    if down_weight.shape[-1] != i_dim:
        raise ValueError(
            f"down_weight in_features mismatch: gate_up I={i_dim}, weight K={down_weight.shape[-1]}"
        )

    orig_shape = gate_up.shape
    x_2d = _flatten_rows(gate_up)
    m_rows = x_2d.shape[0]
    h_dim = int(down_weight.shape[0])
    use_triton = (
        _HAS_TRITON
        and gate_up.is_cuda
        and down_weight.is_cuda
        and not torch.is_grad_enabled()
        and deepfusion_mlp_prefers_triton_shape(i_dim, h_dim, m_rows, mode=mode)
    )

    if residual is not None:
        expected = (*orig_shape[:-1], h_dim)
        if tuple(residual.shape) != tuple(expected):
            raise ValueError(f"residual shape mismatch: got {tuple(residual.shape)} expected {tuple(expected)}")
        if residual.device != gate_up.device or residual.dtype != gate_up.dtype:
            raise ValueError("residual must match gate_up device and dtype")

    if out is None and residual is not None:
        out = residual

    if out is None:
        out_2d = torch.empty((m_rows, h_dim), device=gate_up.device, dtype=gate_up.dtype)
    else:
        expected = (*orig_shape[:-1], h_dim)
        if tuple(out.shape) != tuple(expected):
            raise ValueError(f"out shape mismatch: got {tuple(out.shape)} expected {tuple(expected)}")
        if out.device != gate_up.device or out.dtype != gate_up.dtype:
            raise ValueError("out must match gate_up device and dtype")
        out_2d = out if out.ndim == 2 else out.flatten(0, -2)

    if not use_triton:
        return _fallback_swiglu_down(gate_up, down_weight, down_bias, out, residual, activation)

    w = down_weight if down_weight.is_contiguous() else down_weight.contiguous()
    bias_ptr = down_bias if down_bias is not None else x_2d
    if mode == "prefill" and residual is None:
        grid = lambda META: (
            triton.cdiv(m_rows, META["BLOCK_M"]) * triton.cdiv(h_dim, META["BLOCK_N"]),
        )
        _deepfusion_swiglu_down_prefill_kernel[grid](
            x_2d,
            w,
            bias_ptr,
            out_2d,
            x_2d.stride(0), x_2d.stride(1),
            w.stride(0), w.stride(1),
            out_2d.stride(0), out_2d.stride(1),
            m_rows,
            i_dim,
            h_dim,
            HAS_BIAS=1 if down_bias is not None else 0,
            ACT=act_id,
        )
        if out is not None:
            return out
        return out_2d.view(*orig_shape[:-1], h_dim)

    if residual is not None:
        residual_2d = residual if residual.ndim == 2 else residual.flatten(0, -2)
    else:
        residual_2d = x_2d
    block_n, block_k, num_warps, num_stages = _pick_cfg(i_dim, h_dim, m_rows)
    grid = (m_rows, triton.cdiv(h_dim, block_n))
    _deepfusion_swiglu_down_kernel[grid](
        x_2d,
        w,
        bias_ptr,
        residual_2d,
        out_2d,
        x_2d.stride(0), x_2d.stride(1),
        w.stride(0), w.stride(1),
        residual_2d.stride(0), residual_2d.stride(1),
        out_2d.stride(0), out_2d.stride(1),
        i_dim,
        h_dim,
        HAS_BIAS=1 if down_bias is not None else 0,
        HAS_RESIDUAL=1 if residual is not None else 0,
        ACT=act_id,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if out is not None:
        return out
    return out_2d.view(*orig_shape[:-1], h_dim)


def gemma4_e2b_b8_geglu_down_tensorcore(
    gate_up: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: Optional[torch.Tensor] = None,
    *,
    out: torch.Tensor,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Fused Tensor Core GeGLU+down for the exact E2B/L4/B8 large MLP.

    This deliberately has no generic fallback.  Callers must treat rejection
    as an experimental-route failure and retain the existing cuBLAS chain.
    """
    if not _HAS_TRITON:
        raise RuntimeError("Triton is unavailable")
    if torch.is_grad_enabled():
        raise ValueError("Gemma4 E2B B8 Tensor Core MLP requires inference mode")
    if not gate_up.is_cuda or not down_weight.is_cuda or not out.is_cuda:
        raise ValueError("Gemma4 E2B B8 Tensor Core MLP requires CUDA tensors")
    if gate_up.dtype != torch.bfloat16 or down_weight.dtype != torch.bfloat16:
        raise ValueError("Gemma4 E2B B8 Tensor Core MLP requires BF16 inputs")
    if out.dtype != gate_up.dtype or out.device != gate_up.device:
        raise ValueError("out must match gate_up device and dtype")
    if down_weight.device != gate_up.device:
        raise ValueError("gate_up and down_weight must use the same device")
    if tuple(gate_up.shape) != (8, 24576):
        raise ValueError(
            f"gate_up shape must be (8, 24576), got {tuple(gate_up.shape)}"
        )
    if tuple(down_weight.shape) != (1536, 12288):
        raise ValueError(
            "down_weight shape must be (1536, 12288), got "
            f"{tuple(down_weight.shape)}"
        )
    if tuple(out.shape) != (8, 1536):
        raise ValueError(f"out shape must be (8, 1536), got {tuple(out.shape)}")
    if gate_up.stride(1) != 1 or down_weight.stride(1) != 1 or out.stride(1) != 1:
        raise ValueError("all tensors must have a contiguous last dimension")
    if down_bias is not None:
        if tuple(down_bias.shape) != (1536,):
            raise ValueError("down_bias shape must be (1536,)")
        if down_bias.device != gate_up.device or down_bias.dtype != gate_up.dtype:
            raise ValueError("down_bias must match gate_up device and dtype")
    if block_n not in (32, 64, 128):
        raise ValueError("block_n must be one of 32, 64, 128")
    if block_k not in (32, 64):
        raise ValueError("block_k must be one of 32, 64")
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")
    if num_stages not in (2, 3):
        raise ValueError("num_stages must be 2 or 3")

    bias_ptr = down_bias if down_bias is not None else gate_up
    _gemma4_e2b_b8_geglu_down_tensorcore_kernel[
        (triton.cdiv(1536, block_n),)
    ](
        gate_up,
        down_weight,
        bias_ptr,
        out,
        int(gate_up.stride(0)),
        int(gate_up.stride(1)),
        int(down_weight.stride(0)),
        int(down_weight.stride(1)),
        int(out.stride(0)),
        int(out.stride(1)),
        I=12288,
        H=1536,
        HAS_BIAS=down_bias is not None,
        BLOCK_N=int(block_n),
        BLOCK_K=int(block_k),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


HAS_DEEPFUSION_MLP = _HAS_TRITON


def deepfusion_runtime_config() -> dict:
    return {
        "has_triton": bool(_HAS_TRITON),
        "shape_guard": bool(_CFG_SHAPE_GUARD),
        "force_triton": bool(_CFG_FORCE_TRITON),
        "prefill_enabled": bool(_CFG_PREFILL_ENABLED),
        "prefill_force_triton": bool(_CFG_PREFILL_FORCE_TRITON),
        "prefill_min_rows": int(_CFG_PREFILL_MIN_ROWS),
        "prefill_group_m": int(_CFG_PREFILL_GROUP_M),
        "prefill_block_m": int(_CFG_PREFILL_BLOCK_M),
        "prefill_block_n": int(_CFG_PREFILL_BLOCK_N),
        "prefill_block_k": int(_CFG_PREFILL_BLOCK_K),
        "prefill_num_warps": int(_CFG_PREFILL_NUM_WARPS),
        "prefill_num_stages": int(_CFG_PREFILL_NUM_STAGES),
        "max_rows": int(_CFG_MAX_ROWS),
        "min_i": int(_CFG_MIN_I),
        "min_h": int(_CFG_MIN_H),
        "forced_block_n": int(_CFG_FORCED_BN),
        "forced_block_k": int(_CFG_FORCED_BK),
        "forced_num_warps": int(_CFG_FORCED_WARPS),
        "forced_num_stages": int(_CFG_FORCED_STAGES),
    }


__all__ = [
    "deepfusion_swiglu_down",
    "gemma4_e2b_b8_geglu_down_tensorcore",
    "deepfusion_mlp_prefers_triton_shape",
    "deepfusion_runtime_config",
    "HAS_DEEPFUSION_MLP",
]
