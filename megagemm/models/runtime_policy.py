"""Model- and hardware-specific runtime policy selection.

The policy records measured execution choices without making benchmark runners
responsible for configuring the engine. Environment variables remain explicit
overrides for experiments and regression isolation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimePolicy:
    name: str = "generic"
    hardware: str = "generic"
    prefer_triton_rmsnorm: bool = False
    decode_prefer_step: bool = False
    reuse_request_scheduler: bool = False
    reuse_request_scheduler_batches: tuple[int, ...] = ()
    decode_cuda_graphs: bool = False
    decode_cuda_graph_batches: tuple[int, ...] = ()
    decode_cuda_graph_prefer_step: bool = False
    decode_graph_multi_step_body: bool = False
    decode_graph_persistent_token_feedback: bool = False
    paged_decode_splits: int = 0
    paged_decode_gqa2_direct: bool = False
    paged_decode_warps_h256: int = 0
    gemma4_dense_post_norm_chain: bool = False
    gemma4_e2b_h512_dense_bridge_pair: bool = False
    gemma4_e2b_b1_dense_bridge: bool = False
    gemma4_ple_conditioned_gelu_decode: bool = False
    gemma4_e2b_l4_sliding_prefill: bool = False
    gemma4_e2b_l4_full_prefill_expand: bool = False
    gemma4_bf16_fused_gateup_rows: tuple[int, ...] = ()
    gemma4_bf16_deepfusion_rows: tuple[int, ...] = ()
    gemma4_bf16_cublas_gateup_rows: tuple[int, ...] = ()
    gemma4_bf16_cublas_down_rows: tuple[int, ...] = ()
    reason: str = "no model-specific measured policy"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _config_int(config: Any, name: str) -> int:
    try:
        return int(getattr(config, name, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _gemma4_topology(config: Any) -> tuple[int, int, int, int]:
    return (
        _config_int(config, "num_hidden_layers"),
        _config_int(config, "hidden_size"),
        _config_int(config, "num_attention_heads"),
        _config_int(config, "num_key_value_heads"),
    )


def resolve_runtime_policy(config: Any, device_name: str = "") -> RuntimePolicy:
    """Resolve only choices backed by a model/hardware-specific measurement."""
    model_type = str(getattr(config, "model_type", "") or "").lower()
    normalized_device = str(device_name or "").upper()
    if model_type != "gemma4_text":
        return RuntimePolicy()

    topology = _gemma4_topology(config)
    if "L4" not in normalized_device:
        return RuntimePolicy(
            name="gemma4-generic",
            hardware=normalized_device or "unknown",
            reason="Gemma 4 topology recognized; no hardware-specific policy promoted",
        )

    if topology == (35, 1536, 8, 1):
        return RuntimePolicy(
            name="gemma4-e2b-l4",
            hardware="NVIDIA L4",
            prefer_triton_rmsnorm=True,
            decode_prefer_step=False,
            reuse_request_scheduler=False,
            reuse_request_scheduler_batches=(4,),
            decode_cuda_graphs=True,
            decode_cuda_graph_batches=(4,),
            decode_cuda_graph_prefer_step=True,
            decode_graph_multi_step_body=True,
            decode_graph_persistent_token_feedback=True,
            paged_decode_splits=1,
            paged_decode_gqa2_direct=True,
            paged_decode_warps_h256=2,
            gemma4_dense_post_norm_chain=True,
            gemma4_e2b_h512_dense_bridge_pair=True,
            gemma4_e2b_b1_dense_bridge=True,
            gemma4_e2b_l4_sliding_prefill=True,
            gemma4_e2b_l4_full_prefill_expand=True,
            gemma4_bf16_cublas_gateup_rows=(8,),
            gemma4_bf16_cublas_down_rows=(8,),
            reason=(
                "validated E2B L4 path: Triton RMSNorm, multi-step eager decode, "
                "unsplit two-warp H256 paged attention with the direct GQA2 "
                "kernel, and the "
                "dense post-norm chain; the route-normalized long-context "
                "BF16 batch-8 gate promotes the full-H512 grouped attention "
                "and dense attention-to-MLP bridge only as a coupled path; "
                "the MLP gate "
                "retains cuBLAS for gate-up and down instead of the slower "
                "fused gate-up and deepfusion MLP kernels, while "
                "long sliding prefill uses independently measured B1, B2, B4, "
                "and B8 Q8/KV1/H256/W512 Triton launch geometries; long full-H512 "
                "prefill expands the single KV head once per layer and uses "
                "implicit-causal SDPA. The exact-output B4/P2048 loaded-model "
                "gate promoted G2/BM16/BN64/W4/S2 after reducing prefill from "
                "2180.93 ms to 913.50 ms (2.387x), while the B8 dispatch "
                "remains unchanged. The exact-output B2/P2048 gate independently "
                "selected the same G2/BM16/BN64/W4/S2 geometry and reduced "
                "prefill from 1078.41 ms to 447.18 ms (2.412x). The exact-output "
                "B1/P2048 gate selected G1/BM32/BN64/W4/S2 and reduced prefill "
                "from 582.35 ms to 272.29 ms (2.139x). The paired B1/P2048 "
                "decode gate promoted the dense attention-to-MLP bridge after "
                "raising median decode from 32.94 to 36.79 tok/s (1.117x); "
                "the exact-output B4/P2048/O128 execution gate promotes the "
                "one-step CUDA Graph path after raising median decode from "
                "122.01 to 141.39 tok/s (1.159x) and end-to-end throughput "
                "from 98.05 to 110.72 tok/s (1.129x); the statistically tied "
                "eight-step unrolled graph remains experimental; "
                "the forced fused LM head and experimental large MLP paths "
                "remain unpromoted"
            ),
        )
    if topology == (42, 2560, 8, 2):
        return RuntimePolicy(
            name="gemma4-e4b-l4",
            hardware="NVIDIA L4",
            prefer_triton_rmsnorm=False,
            decode_prefer_step=True,
            reuse_request_scheduler=True,
            reason=(
                "validated E4B L4 path: native RMSNorm, decode_step, and scheduler reuse"
            ),
        )

    return RuntimePolicy(
        name="gemma4-l4-generic",
        hardware="NVIDIA L4",
        reason=f"unrecognized Gemma 4 L4 topology {topology!r}",
    )


def policy_bool(
    model: Any,
    env_name: str,
    policy_field: str,
    default: bool = False,
) -> bool:
    """Read an explicit environment override or fall back to model policy."""
    raw = os.environ.get(env_name, "").strip().lower()
    if raw:
        return raw in _TRUE_VALUES
    policy = getattr(model, "runtime_policy", None)
    return bool(getattr(policy, policy_field, default))


def policy_rows(
    model: Any,
    env_name: str,
    policy_field: str,
) -> tuple[int, ...]:
    """Return promoted row counts unless an explicit force flag overrides them.

    The caller combines these rows with its already-resolved boolean force flag.
    Suppressing the policy rows whenever the environment variable is explicit
    preserves both ``=0`` experiment isolation and ``=1`` global force behavior.
    """
    if os.environ.get(env_name, "").strip():
        return ()
    policy = getattr(model, "runtime_policy", None)
    raw_rows = getattr(policy, policy_field, ())
    try:
        return tuple(sorted({max(1, int(row)) for row in raw_rows}))
    except (TypeError, ValueError):
        return ()


__all__ = [
    "RuntimePolicy",
    "policy_bool",
    "policy_rows",
    "resolve_runtime_policy",
]
