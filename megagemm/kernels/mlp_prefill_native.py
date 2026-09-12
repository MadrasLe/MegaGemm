import torch

try:
    import rmsnorm_cuda_ops as _cuda_ops
except Exception:
    _cuda_ops = None

try:
    import megagemm_cublaslt_ops as _focused_cublaslt_ops
except Exception:
    _focused_cublaslt_ops = None

_cublaslt_ops = (
    _cuda_ops
    if _cuda_ops is not None
    and hasattr(_cuda_ops, "cublaslt_bf16_algorithm_count_cuda")
    and hasattr(_cuda_ops, "cublaslt_bf16_linear_cuda")
    else _focused_cublaslt_ops
)


HAS_NATIVE_MLP_PREFILL = bool(
    _cuda_ops is not None
    and hasattr(_cuda_ops, "mlp_prefill_forward_cuda")
)
HAS_CUBLASLT_BF16_LINEAR = bool(
    _cublaslt_ops is not None
    and hasattr(_cublaslt_ops, "cublaslt_bf16_algorithm_count_cuda")
    and hasattr(_cublaslt_ops, "cublaslt_bf16_linear_cuda")
)


def cublaslt_bf16_algorithm_count_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    maximum_algorithms: int = 32,
) -> int:
    if not HAS_CUBLASLT_BF16_LINEAR:
        raise RuntimeError("native cuBLASLt BF16 linear op is unavailable")
    return int(
        _cublaslt_ops.cublaslt_bf16_algorithm_count_cuda(
            x,
            weight,
            int(maximum_algorithms),
        )
    )


def cublaslt_bf16_linear_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out: torch.Tensor | None = None,
    algorithm_index: int = 0,
) -> torch.Tensor:
    """Compute ``x @ weight.T`` with one explicit cuBLASLt heuristic."""
    if not HAS_CUBLASLT_BF16_LINEAR:
        raise RuntimeError("native cuBLASLt BF16 linear op is unavailable")
    return _cublaslt_ops.cublaslt_bf16_linear_cuda(
        x,
        weight,
        bias,
        out,
        int(algorithm_index),
    )


def mlp_prefill_forward_cuda(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_bias: torch.Tensor | None,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor | None,
    intermediate_size: int,
) -> torch.Tensor:
    if not HAS_NATIVE_MLP_PREFILL:
        raise RuntimeError("native MLP prefill CUDA op is unavailable")
    gate_up_bias_arg = gate_up_bias if gate_up_bias is not None else None
    down_bias_arg = down_bias if down_bias is not None else None
    return _cuda_ops.mlp_prefill_forward_cuda(
        x,
        gate_up_weight,
        gate_up_bias_arg,
        down_weight,
        down_bias_arg,
        int(intermediate_size),
    )
