"""Experimental B1 BF16 GEMVs for E2B's twenty large dense MLPs.

Weights retain their existing contiguous [N, K] layout. Each lane accumulates
over K tiles before the one intra-program reduction. Down-projection can split
K across programs, then reduce FP32 partials in a fixed order (no atomics).
This module is loaded lazily; no production policy is changed here.
"""

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


@dataclass(frozen=True)
class GemvConfig:
    block_n: int
    block_k: int
    splits: int
    warps: int


# Separate choices for the wide-output and long-reduction problems.
GATEUP_CONFIGS = {
    "wide_n4_k256_w4": GemvConfig(4, 256, 1, 4),
    "wide_n4_k2048_w4": GemvConfig(4, 2048, 1, 4),
    "wide_n8_k256_w4": GemvConfig(8, 256, 1, 4),
}
DOWN_CONFIGS = {
    "down_n4_k256_s1_w4": GemvConfig(4, 256, 1, 4),
    "down_n4_k512_s4_w4": GemvConfig(4, 512, 4, 4),
    "down_n4_k512_s8_w4": GemvConfig(4, 512, 8, 4),
    "down_n8_k256_s8_w4": GemvConfig(8, 256, 8, 4),
}


def projection_spec(operation, config_name):
    if operation == "gateup":
        shape, configs = (24576, 1536), GATEUP_CONFIGS
    elif operation == "down":
        shape, configs = (1536, 12288), DOWN_CONFIGS
    else:
        raise ValueError(f"Unknown B1 MLP projection: {operation!r}")
    if config_name not in configs:
        raise ValueError(f"Unknown {operation} configuration: {config_name!r}")
    return shape, configs[config_name]


if triton is not None:
    @triton.jit
    def _b1_mlp_partial(X, W, Bias, Out, Partial, N: tl.constexpr,
                        K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        SPLITS: tl.constexpr, HAS_BIAS: tl.constexpr):
        n = tl.program_id(0) * BN + tl.arange(0, BN)
        split = tl.program_id(1)
        # Host-known geometry must stay constexpr: tl.cdiv is a JIT tensor
        # function, whose result must not be wrapped in a local constexpr.
        chunk: tl.constexpr = (K + SPLITS - 1) // SPLITS
        iterations: tl.constexpr = (chunk + BK - 1) // BK
        lane = tl.arange(0, BK)
        acc = tl.full((BN, BK), 0.0, tl.float32)
        for offset in range(iterations):
            local_k = offset * BK + lane
            k = split * chunk + local_k
            mask_k = (local_k < chunk) & (k < K)
            x = tl.load(X + k, mask=mask_k, other=0).to(tl.float32)
            w = tl.load(W + n[:, None] * K + k[None, :],
                        mask=(n[:, None] < N) & mask_k[None, :],
                        other=0).to(tl.float32)
            acc += w * x[None, :]
        value = tl.sum(acc, axis=1)
        if SPLITS == 1:
            # Match torch.mm(..., out=BF16) followed by BF16 bias addition.
            value = value.to(Out.dtype.element_ty)
            if HAS_BIAS:
                value = (value.to(tl.float32) + tl.load(
                    Bias + n, mask=n < N, other=0).to(tl.float32))
            tl.store(Out + n, value, mask=n < N)
        else:
            tl.store(Partial + split * N + n, value, mask=n < N)

    @triton.jit
    def _b1_mlp_reduce(Partial, Bias, Out, N: tl.constexpr,
                       SPLITS: tl.constexpr, BS: tl.constexpr,
                       HAS_BIAS: tl.constexpr, BN: tl.constexpr):
        n = tl.program_id(0) * BN + tl.arange(0, BN)
        s = tl.arange(0, BS)
        values = tl.load(Partial + s[:, None] * N + n[None, :],
                         mask=(s[:, None] < SPLITS) & (n[None, :] < N), other=0)
        value = tl.sum(values, axis=0).to(Out.dtype.element_ty)
        if HAS_BIAS:
            value = value.to(tl.float32) + tl.load(
                Bias + n, mask=n < N, other=0).to(tl.float32)
        tl.store(Out + n, value, mask=n < N)


class B1MlpGemvPlan:
    """One weight/config binding with reusable scratch, on the current stream.

    A plan belongs to one sequential model executor, like its output buffers.
    Unsupported layouts raise, never silently repack or fall back to torch.mm.
    """

    def __init__(self, weight, bias, operation, config_name):
        shape, config = projection_spec(operation, config_name)
        if tuple(weight.shape) != shape:
            raise ValueError(f"{operation} requires weight {shape}, got {tuple(weight.shape)}")
        if not weight.is_cuda or weight.dtype != torch.bfloat16 or not weight.is_contiguous():
            raise ValueError("B1 MLP GEMV requires contiguous CUDA BF16 [N,K] weights")
        if bias is not None and (
            tuple(bias.shape) != (shape[0],) or bias.dtype != weight.dtype
            or bias.device != weight.device or not bias.is_contiguous()
        ):
            raise ValueError("bias must be contiguous BF16 [N] on the weight device")
        if triton is None:
            raise RuntimeError("Triton is unavailable for B1 MLP GEMV")
        self.weight, self.bias, self.config = weight, bias, config
        self.n, self.k = shape
        self.partial = torch.empty(
            (config.splits, self.n), device=weight.device, dtype=torch.float32,
        ) if config.splits > 1 else None

    def __call__(self, x, out):
        if torch.is_grad_enabled():
            raise RuntimeError("B1 MLP GEMV requires inference_mode or no_grad")
        if tuple(x.shape) != (1, self.k) or tuple(out.shape) != (1, self.n):
            raise ValueError("B1 MLP GEMV input/output shape mismatch")
        if any(t.dtype != self.weight.dtype or t.device != self.weight.device
               or not t.is_contiguous() for t in (x, out)):
            raise ValueError("input/output must be contiguous BF16 on the weight device")
        c = self.config
        # Device guard also covers callers whose current CUDA device differs.
        with torch.cuda.device(x.device):
            _b1_mlp_partial[(triton.cdiv(self.n, c.block_n), c.splits)](
                x, self.weight, self.bias, out, self.partial,
                self.n, self.k, c.block_n, c.block_k, c.splits,
                self.bias is not None, num_warps=c.warps, num_stages=1,
            )
            if self.partial is not None:
                _b1_mlp_reduce[(triton.cdiv(self.n, 128),)](
                    self.partial, self.bias, out, self.n, c.splits,
                    triton.next_power_of_2(c.splits), self.bias is not None, 128,
                    num_warps=4, num_stages=1,
                )
        return out

    def bind_output(self, out):
        """Bind a validated engine-owned output buffer outside the decode loop.

        The caller must guard inference shape/device and buffer lifetime once at
        the executor boundary. The returned launcher has no tensor validation,
        device context manager, allocation, cache lookup or grid construction.
        """
        if tuple(out.shape) != (1, self.n) or out.dtype != self.weight.dtype \
                or out.device != self.weight.device or not out.is_contiguous():
            raise ValueError("bound output must be contiguous BF16 [1,N] on the weight device")
        c = self.config
        n, k = self.n, self.k
        weight, bias, partial = self.weight, self.bias, self.partial
        has_bias = bias is not None
        bn, bk, splits, warps = c.block_n, c.block_k, c.splits, c.warps
        launch = _b1_mlp_partial[(triton.cdiv(n, bn), splits)]
        if partial is None:
            def run(x):
                launch(x, weight, bias, out, partial, n, k, bn, bk, splits,
                       has_bias, num_warps=warps, num_stages=1)
                return out
        else:
            reduce = _b1_mlp_reduce[(triton.cdiv(n, 128),)]
            bs = triton.next_power_of_2(splits)

            def run(x):
                launch(x, weight, bias, out, partial, n, k, bn, bk, splits,
                       has_bias, num_warps=warps, num_stages=1)
                reduce(partial, bias, out, n, splits, bs, has_bias, 128,
                       num_warps=4, num_stages=1)
                return out
        return run
