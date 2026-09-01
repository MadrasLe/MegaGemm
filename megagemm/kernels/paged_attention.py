"""
⚡ Paged Attention for MegaGemm
-------------------------------
Triton kernel for decode-phase attention with paged KV cache.
Falls back to PyTorch when Triton is not available (e.g., Windows).
For prefill, delegates to PyTorch SDPA (which uses FlashAttention).

Supports GQA (Grouped Query Attention) natively.

Author: Gabriel Yogi
Reference: PagedAttention (Kwon et al., 2023)
"""

import os
from dataclasses import dataclass
import torch
import math
from typing import Optional

__all__ = ['paged_attention_decode', 'prefill_attention', 'PackedAttentionMetadata',
           'prepare_packed_attention_metadata', 'packed_attention',
           'packed_prefill_attention', 'fused_rope_kv_write',
           'gemma4_long_sliding_prefill_attention',
           'gemma4_long_full_prefill_attention',
           'gemma4_e2b_l4_sliding_prefill_attention',
           'paged_kv_cache_scatter',
           'paged_kv_cache_scatter_token_tiled',
           '_triton_paged_decode_fused',
           '_triton_paged_decode_grouped_segmented',
           '_triton_paged_decode_grouped_segmented_fused',
           'paged_decode_runtime_stats']

# Try to import Triton (Linux only)
_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass


_DECODE_WORKSPACE = {}
_UNIFORM_CU_EXPECTED_CACHE = {}
_DECODE_SHAPE_LOGGED = set()
_GQA2_DECODE_DISABLED = False
_GQA2_DECODE_LOGGED = False
_GQA2_DIRECT_DECODE_HITS = 0
_GENERIC_DIRECT_DECODE_HITS = 0
_GQA4_DECODE_DISABLED = False
_GQA4_DECODE_LOGGED = False
_GQA8_DECODE_DISABLED = False
_GQA8_DECODE_LOGGED = False
_GROUPED_SEGMENTED_DECODE_DISABLED = False
_GROUPED_SEGMENTED_DECODE_LOGGED = False
_GROUPED_SEGMENTED_DECODE_HITS = 0
_GROUPED_SEGMENTED_DECODE_FAILURE = ""
_GROUPED_SEGMENTED_DECODE_SELECTED_SEGMENTS = {}
_GROUPED_SEGMENTED_DECODE_SELECTED_TILE_SIZES = {}
_GEMMA4_LONG_SLIDING_PREFILL_DISABLED = False
_GEMMA4_LONG_SLIDING_PREFILL_FAILURE = ""
_GEMMA4_LONG_SLIDING_PREFILL_LOGGED = False
_GEMMA4_LONG_FULL_PREFILL_DISABLED = False
_GEMMA4_LONG_FULL_PREFILL_FAILURE = ""
_GEMMA4_LONG_FULL_PREFILL_LOGGED = False
_GEMMA4_E2B_L4_SLIDING_PREFILL_DISABLED = False
_GEMMA4_E2B_L4_B2_SLIDING_PREFILL_DISABLED = False
_GEMMA4_E2B_L4_B4_SLIDING_PREFILL_DISABLED = False
_GEMMA4_E2B_L4_SLIDING_PREFILL_FAILURE = ""
_GEMMA4_E2B_L4_SLIDING_PREFILL_LOGGED = False
_GEMMA4_E2B_L4_SLIDING_PREFILL_HITS = 0


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _prefill_gqa_mode() -> str:
    mode = os.environ.get("MEGAGEMM_PREFILL_GQA_MODE", "native").strip().lower()
    if mode not in {"native", "expand"}:
        return "native"
    return mode


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def paged_decode_runtime_stats() -> dict:
    return {
        "gqa2_direct_hits": int(_GQA2_DIRECT_DECODE_HITS),
        "generic_direct_hits": int(_GENERIC_DIRECT_DECODE_HITS),
        "gqa2_disabled": bool(_GQA2_DECODE_DISABLED),
        "grouped_segmented_hits": int(_GROUPED_SEGMENTED_DECODE_HITS),
        "grouped_segmented_disabled": bool(
            _GROUPED_SEGMENTED_DECODE_DISABLED
        ),
        "grouped_segmented_failure": str(
            _GROUPED_SEGMENTED_DECODE_FAILURE
        ),
        "grouped_segmented_selected_segments": dict(
            _GROUPED_SEGMENTED_DECODE_SELECTED_SEGMENTS
        ),
        "grouped_segmented_selected_tile_sizes": dict(
            _GROUPED_SEGMENTED_DECODE_SELECTED_TILE_SIZES
        ),
    }


def _cuda_device_info(device: Optional[torch.device] = None) -> tuple[tuple[int, int], str, int]:
    try:
        if device is not None and getattr(device, "type", None) == "cuda":
            capability = torch.cuda.get_device_capability(device)
            device_name = torch.cuda.get_device_name(device)
            props = torch.cuda.get_device_properties(device)
        elif torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            device_name = torch.cuda.get_device_name()
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
        else:
            return (0, 0), "", 0
        return capability, device_name, int(getattr(props, "multi_processor_count", 0) or 0)
    except Exception:
        return (0, 0), "", 0


def _device_name_tokens(device_name: str) -> list[str]:
    return device_name.lower().replace("-", " ").replace("_", " ").split()


def _decode_num_warps(
    head_dim: int,
    device: Optional[torch.device] = None,
    *,
    num_splits: int = 1,
    policy_override: Optional[int] = None,
) -> int:
    shape_forced = _env_int(
        f"MEGAGEMM_PAGED_DECODE_WARPS_H{int(head_dim)}",
        0,
    )
    forced = (
        shape_forced
        if shape_forced > 0
        else _env_int("MEGAGEMM_PAGED_DECODE_WARPS", 0)
    )
    if forced > 0:
        return max(1, min(forced, 8))
    try:
        promoted = int(policy_override or 0)
    except (TypeError, ValueError):
        promoted = 0
    if promoted > 0:
        return max(1, min(promoted, 8))
    if head_dim < 128:
        return 4

    # Split-K decode already multiplies the number of resident programs. Keeping
    # the per-program core at 8 warps makes A100 long-context decode heavy on
    # scheduler/register pressure without adding useful parallelism inside the
    # tiny BLOCK_SIZE=16 tile. The env override above can force 8 for sweeps.
    if num_splits > 1:
        return 4

    # Turing/T4 and Ada/L4 are sensitive to over-wide decode programs at HD=128.
    # Sweeps on Qwen2.5-3B showed 4 warps beating 8 warps for long-context decode;
    # keep 8-warps on other newer GPUs until we have per-architecture data.
    capability, device_name, _ = _cuda_device_info(device)
    if capability[0] and capability[0] < 8:
        return 4
    if "l4" in _device_name_tokens(device_name):
        return 4
    return 8


def _decode_block_unroll(
    *,
    block_size: int,
    head_dim: int,
    num_splits: int,
    device: Optional[torch.device] = None,
) -> int:
    forced = _env_int("MEGAGEMM_PAGED_DECODE_BLOCK_UNROLL", 0)
    if forced > 0:
        return 2 if forced >= 2 else 1
    if forced < 0:
        return 1
    if block_size != 16 or head_dim != 128 or num_splits != 1:
        return 1
    try:
        if device is not None and getattr(device, "type", None) == "cuda":
            capability = torch.cuda.get_device_capability(device)
        elif torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
        else:
            capability = (0, 0)
    except Exception:
        capability = (0, 0)
    return 2 if capability[0] and capability[0] < 8 else 1


def _decode_reduce_num_warps(head_dim: int, num_splits: int) -> int:
    forced = _env_int("MEGAGEMM_PAGED_DECODE_REDUCE_WARPS", 0)
    if forced > 0:
        return max(1, min(forced, 8))
    # The reduce kernel only combines a tiny [splits, head_dim] partial tensor.
    # Reusing the attention core's 8 warps wastes scheduler/occupancy budget on
    # A100/L4 long-context decode. One warp is enough for the common HD=128,
    # splits<=8 path we benchmark on Qwen-family models.
    if head_dim <= 128 and num_splits <= 8:
        return 1
    return 2


def _resolve_decode_max_blocks(
    table_max_blocks: int,
    max_blocks_override: Optional[int],
) -> int:
    if max_blocks_override is None:
        return int(table_max_blocks)
    try:
        requested = int(max_blocks_override)
    except (TypeError, ValueError):
        return int(table_max_blocks)
    return max(1, min(requested, int(table_max_blocks)))


def _log_decode_shape_once(
    *,
    fused: bool,
    num_seqs: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    table_max_blocks: int,
    loop_max_blocks: int,
    num_splits: int,
    num_warps: int,
    reduce_warps: int,
) -> None:
    if not _env_bool("MEGAGEMM_PAGED_DECODE_LOG", False):
        return
    gqa_ratio = num_q_heads // max(1, num_kv_heads)
    key = (
        fused,
        int(num_seqs),
        int(num_q_heads),
        int(num_kv_heads),
        int(head_dim),
        int(block_size),
        int(table_max_blocks),
        int(loop_max_blocks),
        int(num_splits),
        int(num_warps),
        int(reduce_warps),
    )
    if key in _DECODE_SHAPE_LOGGED:
        return
    _DECODE_SHAPE_LOGGED.add(key)
    print(
        "[MegaGemm] paged decode shape "
        f"fused={fused} seqs={num_seqs} q_heads={num_q_heads} "
        f"kv_heads={num_kv_heads} gqa={gqa_ratio} head_dim={head_dim} "
        f"block_size={block_size} table_blocks={table_max_blocks} "
        f"loop_blocks={loop_max_blocks} splits={num_splits} "
        f"warps={num_warps} reduce_warps={reduce_warps}"
    )


def _use_gqa2_decode(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_splits: int,
) -> bool:
    if _GQA2_DECODE_DISABLED:
        return False
    if not _env_bool("MEGAGEMM_PAGED_DECODE_GQA2_SPLIT", True):
        return False
    if num_splits <= 1 or num_kv_heads <= 0:
        return False
    if head_dim != 128:
        return False
    if num_q_heads % num_kv_heads != 0:
        return False
    gqa_ratio = num_q_heads // num_kv_heads
    return gqa_ratio >= 2 and gqa_ratio % 2 == 0


def _use_gqa4_decode(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_splits: int,
) -> bool:
    if _GQA4_DECODE_DISABLED:
        return False
    if not _env_bool("MEGAGEMM_PAGED_DECODE_GQA4_SPLIT", False):
        return False
    if num_splits <= 1 or num_kv_heads <= 0:
        return False
    if head_dim != 128:
        return False
    if num_q_heads % num_kv_heads != 0:
        return False
    gqa_ratio = num_q_heads // num_kv_heads
    return gqa_ratio >= 4 and gqa_ratio % 4 == 0


def _use_gqa8_decode(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_splits: int,
) -> bool:
    if _GQA8_DECODE_DISABLED:
        return False
    if not _env_bool("MEGAGEMM_PAGED_DECODE_GQA8_SPLIT", False):
        return False
    if num_splits <= 1 or num_kv_heads <= 0:
        return False
    if head_dim != 128:
        return False
    if num_q_heads % num_kv_heads != 0:
        return False
    gqa_ratio = num_q_heads // num_kv_heads
    return gqa_ratio >= 8 and gqa_ratio % 8 == 0


def _planned_fused_decode_program_heads(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    """Estimate q-head programs for split-count planning.

    The runtime split kernel may group adjacent Q heads for GQA. If the split
    heuristic plans with the ungrouped Q-head count, it under-splits exactly
    where long-context GQA decode needs more parallelism.
    """
    if num_kv_heads <= 0 or head_dim != 128:
        return num_q_heads
    if num_q_heads % num_kv_heads != 0:
        return num_q_heads
    gqa_ratio = num_q_heads // num_kv_heads
    if (
        not _GQA8_DECODE_DISABLED
        and _env_bool("MEGAGEMM_PAGED_DECODE_GQA8_SPLIT", False)
        and gqa_ratio >= 8
        and gqa_ratio % 8 == 0
    ):
        return max(1, (num_q_heads + 7) // 8)
    if (
        not _GQA4_DECODE_DISABLED
        and _env_bool("MEGAGEMM_PAGED_DECODE_GQA4_SPLIT", False)
        and gqa_ratio >= 4
        and gqa_ratio % 4 == 0
    ):
        return max(1, (num_q_heads + 3) // 4)
    if (
        not _GQA2_DECODE_DISABLED
        and _env_bool("MEGAGEMM_PAGED_DECODE_GQA2_SPLIT", True)
        and gqa_ratio >= 2
        and gqa_ratio % 2 == 0
    ):
        return max(1, (num_q_heads + 1) // 2)
    return num_q_heads


def _use_gqa2_direct_decode(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_splits: int,
    policy_enabled: Optional[bool] = None,
) -> bool:
    if _GQA2_DECODE_DISABLED:
        return False
    env_value = os.environ.get("MEGAGEMM_PAGED_DECODE_GQA2", "").strip()
    enabled = (
        _env_bool("MEGAGEMM_PAGED_DECODE_GQA2", False)
        if env_value
        else bool(policy_enabled)
    )
    if not enabled:
        return False
    if num_splits != 1 or num_kv_heads <= 0:
        return False
    # Gemma4 sliding attention uses GQA2 with head_dim=256. The direct kernel
    # keeps one K/V load shared by the two Q heads and remains a power-of-two
    # Triton shape at both 128 and 256.
    if head_dim not in (128, 256):
        return False
    if num_q_heads % num_kv_heads != 0:
        return False
    gqa_ratio = num_q_heads // num_kv_heads
    return gqa_ratio >= 2 and gqa_ratio % 2 == 0


def _use_gqa4_direct_decode(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_splits: int,
) -> bool:
    if _GQA4_DECODE_DISABLED:
        return False
    group_raw = os.environ.get("MEGAGEMM_PAGED_DECODE_GQA_GROUP", "").strip()
    if not group_raw:
        return False
    try:
        group = int(group_raw)
    except ValueError:
        return False
    if group != 4:
        return False
    if num_splits != 1 or num_kv_heads <= 0:
        return False
    if head_dim != 128:
        return False
    if num_q_heads % num_kv_heads != 0:
        return False
    gqa_ratio = num_q_heads // num_kv_heads
    return gqa_ratio >= 4 and gqa_ratio % 4 == 0


def _grouped_segmented_decode_topology(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    sliding_window: Optional[int],
    force: bool = False,
    e2b_l4_h512_policy_enabled: Optional[bool] = None,
) -> Optional[str]:
    """Return the measured Gemma4 topology eligible for the new decode core."""
    if _GROUPED_SEGMENTED_DECODE_DISABLED and not force:
        return None
    if (
        not force
        and not _env_bool(
            "MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_DECODE",
            True,
        )
    ):
        return None
    if not (_HAS_TRITON and query.is_cuda and kv_cache.is_cuda):
        return None
    if query.dtype != torch.bfloat16 or kv_cache.dtype != torch.bfloat16:
        return None
    if query.ndim != 3 or kv_cache.ndim != 5 or block_tables.ndim != 2:
        return None

    num_seqs, num_q_heads, head_dim = query.shape
    num_kv_heads = int(kv_cache.shape[2])
    block_size = int(kv_cache.shape[3])
    if block_size != 16:
        return None
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        return None

    _, device_name, _ = _cuda_device_info(query.device)
    device_tokens = _device_name_tokens(device_name)
    window = int(sliding_window or 0)
    gqa_ratio = num_q_heads // num_kv_heads

    # Experimental Gemma4 E2B/L4 full-attention path.  Keep the gate separate
    # from the already-promoted A4B/A100 topology: E2B has half the batch and a
    # single KV head, so copying the A100 policy blindly is not valid.  The
    # checkpoint-free shape gate uses force=True; the loaded model opts in with
    # the dedicated environment flag only after that gate has selected a
    # segment/tile configuration.
    if (
        (
            force
            or _env_bool(
                "MEGAGEMM_GEMMA4_E2B_L4_H512_GROUPED_ATTN_DECODE",
                bool(e2b_l4_h512_policy_enabled),
            )
        )
        and "l4" in device_tokens
        and num_seqs == 8
        and num_q_heads == 8
        and head_dim == 512
        and num_kv_heads == 1
        and gqa_ratio == 8
        and window == 0
    ):
        return "e2b_l4_full_h512_gqa8"

    if "a100" not in device_tokens:
        return None
    if num_seqs != 16 or num_q_heads != 16:
        return None
    if (
        head_dim == 256
        and num_kv_heads == 8
        and gqa_ratio == 2
        and window == 1024
    ):
        return "sliding_h256_gqa2"
    if (
        head_dim == 512
        and num_kv_heads == 2
        and gqa_ratio == 8
        and window == 0
    ):
        return "full_h512_gqa8"
    return None


def _grouped_segmented_decode_num_segments(
    topology: str,
    max_visible_tokens: int,
) -> int:
    """Select only A100 segment counts promoted by paid shape gates."""
    if topology == "e2b_l4_full_h512_gqa8":
        return _env_int(
            "MEGAGEMM_GEMMA4_E2B_L4_H512_ATTN_SEGMENTS",
            32,
        )
    if topology == "sliding_h256_gqa2" and max_visible_tokens >= 1024:
        return 32
    if topology == "full_h512_gqa8" and max_visible_tokens >= 2048:
        return 8
    return 16


def _grouped_segmented_decode_tile_size(
    topology: str,
    max_visible_tokens: int,
) -> int:
    """Select only A100 tile sizes promoted by paid shape gates."""
    if topology == "e2b_l4_full_h512_gqa8":
        return _env_int(
            "MEGAGEMM_GEMMA4_E2B_L4_H512_ATTN_TILE",
            16,
        )
    if topology == "sliding_h256_gqa2" and max_visible_tokens >= 1024:
        return 64
    if topology == "sliding_h256_gqa2":
        return 32
    return 16


@dataclass
class PackedAttentionMetadata:
    cu_seqlens: torch.Tensor
    boundaries: list[int]
    num_seqs: int
    max_seqlen: int
    block_m: int
    block_n: int
    max_k_tiles: int
    tile_seq_start: Optional[torch.Tensor] = None
    tile_seq_len: Optional[torch.Tensor] = None
    tile_local_idx: Optional[torch.Tensor] = None


def _select_packed_block_sizes(head_dim: int) -> tuple[int, int]:
    if head_dim >= 128:
        return 32, 32
    return 64, 64


def prepare_packed_attention_metadata(
    cu_seqlens: torch.Tensor,
    head_dim: int,
) -> PackedAttentionMetadata:
    """
    Precompute reusable metadata for varlen packed attention.

    This is especially helpful for encoder-style models where each layer
    reuses the same cu_seqlens, avoiding repeated host-side tile planning.
    """
    cu = cu_seqlens
    if cu.dtype != torch.int32:
        cu = cu.to(dtype=torch.int32)
    if not cu.is_contiguous():
        cu = cu.contiguous()

    block_m, block_n = _select_packed_block_sizes(head_dim)
    num_seqs = max(0, int(cu.shape[0]) - 1)
    boundaries = cu.tolist()
    max_seqlen = 0
    tile_seq_start_list: list[int] = []
    tile_seq_len_list: list[int] = []
    tile_local_idx_list: list[int] = []

    for i in range(num_seqs):
        seq_start = int(boundaries[i])
        seq_len = int(boundaries[i + 1] - boundaries[i])
        max_seqlen = max(max_seqlen, seq_len)
        num_tiles = (seq_len + block_m - 1) // block_m
        for t in range(num_tiles):
            tile_seq_start_list.append(seq_start)
            tile_seq_len_list.append(seq_len)
            tile_local_idx_list.append(t)

    if tile_seq_start_list:
        tile_seq_start = torch.tensor(tile_seq_start_list, dtype=torch.int32, device=cu.device)
        tile_seq_len = torch.tensor(tile_seq_len_list, dtype=torch.int32, device=cu.device)
        tile_local_idx = torch.tensor(tile_local_idx_list, dtype=torch.int32, device=cu.device)
    else:
        tile_seq_start = None
        tile_seq_len = None
        tile_local_idx = None

    return PackedAttentionMetadata(
        cu_seqlens=cu,
        boundaries=boundaries,
        num_seqs=num_seqs,
        max_seqlen=max_seqlen,
        block_m=block_m,
        block_n=block_n,
        max_k_tiles=(max_seqlen + block_n - 1) // block_n if max_seqlen > 0 else 0,
        tile_seq_start=tile_seq_start,
        tile_seq_len=tile_seq_len,
        tile_local_idx=tile_local_idx,
    )


def _reuse_packed_attention_metadata(
    cu_seqlens: torch.Tensor,
    head_dim: int,
    packed_meta: Optional[PackedAttentionMetadata],
) -> Optional[PackedAttentionMetadata]:
    if packed_meta is None:
        return None

    expected_block_m, expected_block_n = _select_packed_block_sizes(head_dim)
    if packed_meta.block_m != expected_block_m or packed_meta.block_n != expected_block_n:
        return None

    cu = cu_seqlens
    if cu.dtype != torch.int32:
        cu = cu.to(dtype=torch.int32)
    if not cu.is_contiguous():
        cu = cu.contiguous()

    if packed_meta.cu_seqlens.device != cu.device or packed_meta.cu_seqlens.shape != cu.shape:
        return None
    if packed_meta.cu_seqlens.data_ptr() != cu.data_ptr() and not torch.equal(packed_meta.cu_seqlens, cu):
        return None
    return packed_meta


def _get_packed_attention_metadata(
    cu_seqlens: torch.Tensor,
    head_dim: int,
    packed_meta: Optional[PackedAttentionMetadata],
) -> PackedAttentionMetadata:
    cached = _reuse_packed_attention_metadata(cu_seqlens, head_dim, packed_meta)
    if cached is not None:
        return cached
    return prepare_packed_attention_metadata(cu_seqlens, head_dim=head_dim)


def _get_decode_workspace(
    query: torch.Tensor,
    num_seqs: int,
    num_q_heads: int,
    num_splits: int,
    head_dim: int,
):
    """
    Reuse split-decode temporary buffers to avoid per-token allocations.
    """
    key = (
        query.device.type,
        query.device.index if query.device.index is not None else -1,
        num_seqs,
        num_q_heads,
        num_splits,
        head_dim,
    )
    ws = _DECODE_WORKSPACE.get(key)
    if ws is None:
        partial_acc = torch.empty(
            (num_seqs, num_q_heads, num_splits, head_dim),
            device=query.device,
            dtype=torch.float32,
        )
        partial_m = torch.empty(
            (num_seqs, num_q_heads, num_splits),
            device=query.device,
            dtype=torch.float32,
        )
        partial_l = torch.empty_like(partial_m)
        ws = (partial_acc, partial_m, partial_l)
        _DECODE_WORKSPACE[key] = ws
    return ws


def _prepare_decode_output(query: torch.Tensor, out: Optional[torch.Tensor]) -> torch.Tensor:
    if out is None:
        return torch.empty_like(query)
    if out.shape != query.shape:
        raise ValueError(
            f"decode out shape mismatch: got {tuple(out.shape)}, expected {tuple(query.shape)}"
        )
    if out.device != query.device or out.dtype != query.dtype:
        raise ValueError("decode out must match query device and dtype")
    return out


def _get_decode_split_count(
    num_seqs: int,
    num_q_heads: int,
    max_blocks: int,
    *,
    num_warps: int = 4,
    device: Optional[torch.device] = None,
    policy_override: Optional[int] = None,
) -> int:
    """Auto-tune paged decode split parallelism for long-context decode.

    The old heuristic used a fixed resident-program threshold. That is fragile:
    batch=4 x 32 Q heads already gives 128 programs on Qwen3-MoE, but each
    program can still walk 100+ KV blocks serially and leave modern GPUs
    under-occupied. Target warps per SM instead, while keeping an env override
    for sweeps.
    """
    if max_blocks <= 1:
        return 1

    forced = os.environ.get("MEGAGEMM_PAGED_DECODE_SPLITS", "").strip()
    if forced:
        try:
            value = int(forced)
        except ValueError:
            value = 1
        return max(1, min(value, max_blocks))

    if policy_override is not None and int(policy_override) > 0:
        return max(1, min(int(policy_override), max_blocks))

    base_programs = max(1, num_seqs * num_q_heads)
    _, device_name, sm_count = _cuda_device_info(device)
    name_tokens = _device_name_tokens(device_name)
    is_l4 = "l4" in name_tokens
    split_min_blocks_raw = os.environ.get("MEGAGEMM_PAGED_DECODE_SPLIT_MIN_BLOCKS", "").strip()
    try:
        default_min_blocks = 16 if is_l4 else 64
        split_min_blocks = int(split_min_blocks_raw) if split_min_blocks_raw else default_min_blocks
    except ValueError:
        split_min_blocks = 16 if is_l4 else 64

    if max_blocks < max(8, split_min_blocks):
        return 1

    # A100 long-context GQA4 benefits from a small amount of split
    # over-partitioning. At 136 blocks, 40 splits (four blocks in each active
    # split, plus masked tail splits) measured 30% faster than 32 splits under
    # CUDA Graph replay. Keep the older four-block ceiling everywhere else.
    is_a100_long_context = "a100" in name_tokens and max_blocks >= 128
    blocks_per_split_target = 3 if is_a100_long_context else 4
    max_reasonable_splits = max(
        1,
        math.ceil(max_blocks / blocks_per_split_target),
    )

    target_warps_raw = os.environ.get("MEGAGEMM_PAGED_DECODE_TARGET_WARPS_PER_SM", "").strip()
    try:
        target_warps_per_sm = int(target_warps_raw) if target_warps_raw else 32
    except ValueError:
        target_warps_per_sm = 32
    max_splits_raw = os.environ.get("MEGAGEMM_PAGED_DECODE_MAX_SPLITS", "").strip()
    try:
        # A100-80GB batch-1 GQA4 needs more split parallelism than the old
        # eight-split ceiling. CUDA-graph measurements at 136 blocks found
        # 40 splits at 58.2 us/layer versus 83.0 us/layer for 32 splits.
        # Keep the conservative ceiling elsewhere until each architecture is
        # measured independently.
        default_max_splits = 40 if "a100" in name_tokens else 8
        max_auto_splits = int(max_splits_raw) if max_splits_raw else default_max_splits
    except ValueError:
        max_auto_splits = 40 if "a100" in name_tokens else 8
    max_auto_splits = max(1, max_auto_splits)

    if sm_count > 0:
        target_warps = max(1, target_warps_per_sm) * int(sm_count)
        resident_warps = max(1, base_programs * max(1, int(num_warps)))
        splits = (target_warps + resident_warps - 1) // resident_warps
        splits = min(splits, max_auto_splits, max_blocks, max_reasonable_splits)
        return max(1, splits)

    target_raw = os.environ.get("MEGAGEMM_PAGED_DECODE_TARGET_PROGRAMS", "").strip()
    try:
        target_programs = int(target_raw) if target_raw else 256
    except ValueError:
        target_programs = 256
    target_programs = max(64, target_programs)

    splits = (target_programs + base_programs - 1) // base_programs
    splits = min(splits, max_auto_splits, max_blocks)

    # Avoid overly tiny work chunks.
    splits = min(splits, max_reasonable_splits)
    return max(1, splits)


def _plan_decode_sliding_window(
    seq_len: int,
    block_size: int,
    sliding_window: Optional[int],
) -> tuple[int, int, int, int]:
    """
    Plan the local decode window in paged-KV coordinates.

    Returns:
        window_start_token: absolute token index of the first visible token
        first_block: first logical block that intersects the window
        num_window_blocks: number of logical blocks to scan
        trim_in_first_block: number of tokens to trim from the first scanned block
    """
    total_blocks = (seq_len + block_size - 1) // block_size
    if sliding_window is None or sliding_window <= 0 or seq_len <= sliding_window:
        return 0, 0, total_blocks, 0

    window_start = max(0, seq_len - int(sliding_window))
    first_block = window_start // block_size
    trim_in_first_block = window_start - first_block * block_size
    return window_start, first_block, total_blocks - first_block, trim_in_first_block


def _sliding_loop_max_blocks(
    block_size: int,
    max_blocks: int,
    sliding_window: Optional[int],
) -> int:
    """
    Upper bound on how many logical blocks a sliding decode window can touch.
    """
    if sliding_window is None or sliding_window <= 0:
        return max_blocks

    sw = int(sliding_window)
    window_blocks = ((sw + block_size - 2) // block_size) + 1
    return max(1, min(max_blocks, window_blocks))


# =============================================================================
# Triton Kernel (available only on Linux with NVIDIA GPU)
# =============================================================================

if _HAS_TRITON:
    @triton.jit
    def _paged_attn_decode_kernel(
        output_ptr,
        query_ptr,
        kv_cache_ptr,
        block_tables_ptr,
        seq_lens_ptr,
        scale,
        window_size,
        stride_os, stride_oh,
        stride_qs, stride_qh,
        stride_cb, stride_c2, stride_ch, stride_ct, stride_cd,
        stride_bs,
        num_q_heads,
        num_kv_heads,
        max_blocks_per_seq,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HAS_SLIDING_WINDOW: tl.constexpr,
    ):
        """
        Paged Attention Decode Kernel.
        Each program handles one (sequence, query_head) pair.
        Iterates over KV cache blocks using online softmax.
        Grid: (num_seqs, num_q_heads)
        """
        seq_idx = tl.program_id(0)
        q_head_idx = tl.program_id(1)

        # GQA: map query head -> KV head
        gqa_ratio = num_q_heads // num_kv_heads
        kv_head_idx = q_head_idx // gqa_ratio

        # Sequence length
        seq_len = tl.load(seq_lens_ptr + seq_idx)
        num_seq_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        window_start = 0
        first_block = 0
        num_window_blocks = num_seq_blocks
        if HAS_SLIDING_WINDOW:
            window_start = tl.maximum(seq_len - window_size, 0)
            first_block = window_start // BLOCK_SIZE
            num_window_blocks = num_seq_blocks - first_block

        # Load query [HEAD_DIM]
        d_offsets = tl.arange(0, HEAD_DIM)
        q = tl.load(query_ptr + seq_idx * stride_qs + q_head_idx * stride_qh + d_offsets)
        q = q.to(tl.float32) * scale

        # Online softmax state
        m_prev = -1e20
        l_prev = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        t_offsets = tl.arange(0, BLOCK_SIZE)

        for local_block_idx in range(max_blocks_per_seq):
            block_idx = first_block + local_block_idx
            still_valid = local_block_idx < num_window_blocks

            phys_block = tl.load(
                block_tables_ptr + seq_idx * stride_bs + block_idx,
                mask=still_valid, other=0
            )

            block_start = block_idx * BLOCK_SIZE
            token_offsets = block_start + t_offsets
            mask = still_valid & (token_offsets < seq_len)
            if HAS_SLIDING_WINDOW:
                mask = mask & (token_offsets >= window_start)

            # Load K block: [BLOCK_SIZE, HEAD_DIM]
            k_base = phys_block * stride_cb + 0 * stride_c2 + kv_head_idx * stride_ch
            k_ptrs = kv_cache_ptr + k_base + t_offsets[:, None] * stride_ct + d_offsets[None, :] * stride_cd
            k = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

            # QK^T scores: [BLOCK_SIZE]
            qk = q[None, :] * k
            scores = tl.sum(qk, axis=1)
            scores = tl.where(mask, scores, -1e20)

            # Online softmax
            m_cur = tl.max(scores)
            m_new = tl.maximum(m_prev, m_cur)
            alpha = tl.exp(m_prev - m_new)
            exp_scores = tl.exp(scores - m_new)
            exp_scores = tl.where(mask, exp_scores, 0.0)
            l_cur = tl.sum(exp_scores)

            # Load V block: [BLOCK_SIZE, HEAD_DIM]
            v_base = phys_block * stride_cb + 1 * stride_c2 + kv_head_idx * stride_ch
            v_ptrs = kv_cache_ptr + v_base + t_offsets[:, None] * stride_ct + d_offsets[None, :] * stride_cd
            v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

            # Accumulate
            acc = acc * alpha + tl.sum(exp_scores[:, None] * v, axis=0)
            l_prev = l_prev * alpha + l_cur
            m_prev = m_new

        acc = tl.where(l_prev > 0, acc / l_prev, 0.0)
        out_base = seq_idx * stride_os + q_head_idx * stride_oh
        tl.store(output_ptr + out_base + d_offsets, acc.to(output_ptr.dtype.element_ty))

    @triton.jit
    def _paged_attn_decode_split_kernel(
        partial_acc_ptr,
        partial_m_ptr,
        partial_l_ptr,
        query_ptr,
        kv_cache_ptr,
        block_tables_ptr,
        seq_lens_ptr,
        scale,
        window_size,
        stride_pas, stride_pah, stride_pap, stride_pad,
        stride_pms, stride_pmh, stride_pmp,
        stride_qs, stride_qh,
        stride_cb, stride_c2, stride_ch, stride_ct, stride_cd,
        stride_bs,
        num_q_heads,
        num_kv_heads,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
        HAS_SLIDING_WINDOW: tl.constexpr,
    ):
        """
        Split-parallel paged decode (phase 1).
        Each program handles one (sequence, query_head, split) tuple.
        """
        seq_idx = tl.program_id(0)
        q_head_idx = tl.program_id(1)
        split_idx = tl.program_id(2)

        gqa_ratio = num_q_heads // num_kv_heads
        kv_head_idx = q_head_idx // gqa_ratio

        seq_len = tl.load(seq_lens_ptr + seq_idx)
        num_seq_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        window_start = 0
        first_block = 0
        num_window_blocks = num_seq_blocks
        if HAS_SLIDING_WINDOW:
            window_start = tl.maximum(seq_len - window_size, 0)
            first_block = window_start // BLOCK_SIZE
            num_window_blocks = num_seq_blocks - first_block
        split_start = split_idx * BLOCKS_PER_SPLIT

        d_offsets = tl.arange(0, HEAD_DIM)
        q = tl.load(query_ptr + seq_idx * stride_qs + q_head_idx * stride_qh + d_offsets)
        q = q.to(tl.float32) * scale

        m_prev = -1e20
        l_prev = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        t_offsets = tl.arange(0, BLOCK_SIZE)

        for local_block_idx in range(BLOCKS_PER_SPLIT):
            block_offset = split_start + local_block_idx
            block_idx = first_block + block_offset
            still_valid = block_offset < num_window_blocks

            phys_block = tl.load(
                block_tables_ptr + seq_idx * stride_bs + block_idx,
                mask=still_valid, other=0,
            )

            block_start = block_idx * BLOCK_SIZE
            token_offsets = block_start + t_offsets
            mask = still_valid & (token_offsets < seq_len)
            if HAS_SLIDING_WINDOW:
                mask = mask & (token_offsets >= window_start)

            k_base = phys_block * stride_cb + 0 * stride_c2 + kv_head_idx * stride_ch
            k_ptrs = kv_cache_ptr + k_base + t_offsets[:, None] * stride_ct + d_offsets[None, :] * stride_cd
            k = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

            scores = tl.sum(q[None, :] * k, axis=1)
            scores = tl.where(mask, scores, -1e20)

            m_cur = tl.max(scores)
            m_new = tl.maximum(m_prev, m_cur)
            alpha = tl.exp(m_prev - m_new)
            exp_scores = tl.exp(scores - m_new)
            exp_scores = tl.where(mask, exp_scores, 0.0)
            l_cur = tl.sum(exp_scores)

            v_base = phys_block * stride_cb + 1 * stride_c2 + kv_head_idx * stride_ch
            v_ptrs = kv_cache_ptr + v_base + t_offsets[:, None] * stride_ct + d_offsets[None, :] * stride_cd
            v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

            acc = acc * alpha + tl.sum(exp_scores[:, None] * v, axis=0)
            l_prev = l_prev * alpha + l_cur
            m_prev = m_new

        acc_base = seq_idx * stride_pas + q_head_idx * stride_pah + split_idx * stride_pap
        tl.store(partial_acc_ptr + acc_base + d_offsets * stride_pad, acc)

        stat_base = seq_idx * stride_pms + q_head_idx * stride_pmh + split_idx * stride_pmp
        tl.store(partial_m_ptr + stat_base, m_prev)
        tl.store(partial_l_ptr + stat_base, l_prev)

    @triton.jit
    def _paged_attn_decode_split_reduce_kernel(
        output_ptr,
        partial_acc_ptr,
        partial_m_ptr,
        partial_l_ptr,
        stride_os, stride_oh,
        stride_pas, stride_pah, stride_pap, stride_pad,
        stride_pms, stride_pmh, stride_pmp,
        HEAD_DIM: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        """
        Split-parallel paged decode (phase 2).
        Reduces split-local softmax accumulators into final output.
        """
        seq_idx = tl.program_id(0)
        q_head_idx = tl.program_id(1)

        d_offsets = tl.arange(0, HEAD_DIM)
        m_global = -1e20

        for split_idx in range(NUM_SPLITS):
            stat_base = seq_idx * stride_pms + q_head_idx * stride_pmh + split_idx * stride_pmp
            m_val = tl.load(partial_m_ptr + stat_base)
            m_global = tl.maximum(m_global, m_val)

        denom = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        for split_idx in range(NUM_SPLITS):
            stat_base = seq_idx * stride_pms + q_head_idx * stride_pmh + split_idx * stride_pmp
            m_val = tl.load(partial_m_ptr + stat_base)
            l_val = tl.load(partial_l_ptr + stat_base)
            alpha = tl.exp(m_val - m_global)
            denom += l_val * alpha

            acc_base = seq_idx * stride_pas + q_head_idx * stride_pah + split_idx * stride_pap
            part_acc = tl.load(partial_acc_ptr + acc_base + d_offsets * stride_pad).to(tl.float32)
            acc += part_acc * alpha

        out = tl.where(denom > 0, acc / denom, 0.0)
        out_base = seq_idx * stride_os + q_head_idx * stride_oh
        tl.store(output_ptr + out_base + d_offsets, out.to(output_ptr.dtype.element_ty))

    @triton.jit
    def _paged_attn_decode_qnorm_rope_prepare_kernel(
        output_ptr,
        query_ptr,
        cos_ptr,
        sin_ptr,
        pos_ptr,
        norm_weight_ptr,
        norm_eps,
        stride_os,
        stride_oh,
        stride_od,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_cos_p,
        stride_cos_d,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        ROTARY_HALF_DIM: tl.constexpr,
        HALF_ROTATE: tl.constexpr,
        HAS_QK_NORM: tl.constexpr,
    ):
        """Materialize Gemma4 QNorm + RoPE once before segmented attention."""
        seq_idx = tl.program_id(0)
        q_head_idx = tl.program_id(1)
        d_offsets = tl.arange(0, HEAD_DIM)

        q_base = (
            query_ptr
            + seq_idx * stride_qs
            + q_head_idx * stride_qh
        )
        q = tl.load(q_base + d_offsets * stride_qd).to(tl.float32)

        if HAS_QK_NORM:
            variance = tl.sum(q * q) / HEAD_DIM
            rms_scale = tl.rsqrt(variance + norm_eps)
            norm_w = tl.load(norm_weight_ptr + d_offsets).to(tl.float32)
            q = q * rms_scale * norm_w

        if HALF_ROTATE:
            in_rotary = d_offsets < ROTARY_DIM
            is_first = d_offsets < ROTARY_HALF_DIM
            partner = tl.where(
                is_first,
                d_offsets + ROTARY_HALF_DIM,
                d_offsets - ROTARY_HALF_DIM,
            )
            cos_idx = tl.where(
                is_first,
                d_offsets,
                d_offsets - ROTARY_HALF_DIM,
            )
        else:
            in_rotary = d_offsets < ROTARY_DIM
            is_first = (d_offsets % 2) == 0
            partner = tl.where(is_first, d_offsets + 1, d_offsets - 1)
            cos_idx = d_offsets // 2
        partner = tl.where(in_rotary, partner, d_offsets)
        cos_idx = tl.where(in_rotary, cos_idx, 0)

        q_partner = tl.load(
            q_base + partner * stride_qd,
        ).to(tl.float32)
        if HAS_QK_NORM:
            norm_w_partner = tl.load(
                norm_weight_ptr + partner,
            ).to(tl.float32)
            q_partner = q_partner * rms_scale * norm_w_partner

        pos = tl.load(pos_ptr + seq_idx)
        cos_vals = tl.load(
            cos_ptr + pos * stride_cos_p + cos_idx * stride_cos_d,
        ).to(tl.float32)
        sin_vals = tl.load(
            sin_ptr + pos * stride_cos_p + cos_idx * stride_cos_d,
        ).to(tl.float32)
        q_rot = tl.where(
            is_first,
            q * cos_vals - q_partner * sin_vals,
            q * cos_vals + q_partner * sin_vals,
        )
        q = tl.where(in_rotary, q_rot, q)

        out_base = (
            output_ptr
            + seq_idx * stride_os
            + q_head_idx * stride_oh
        )
        tl.store(
            out_base + d_offsets * stride_od,
            q.to(output_ptr.dtype.element_ty),
        )

    @triton.jit
    def _paged_attn_decode_grouped_segment_kernel(
        partial_acc_ptr,
        partial_m_ptr,
        partial_l_ptr,
        query_ptr,
        kv_cache_ptr,
        block_tables_ptr,
        seq_lens_ptr,
        scale,
        window_size,
        stride_pas,
        stride_pah,
        stride_pap,
        stride_pad,
        stride_pms,
        stride_pmh,
        stride_pmp,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_cb,
        stride_c2,
        stride_ch,
        stride_ct,
        stride_cd,
        stride_bs,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        GQA_RATIO: tl.constexpr,
        TILE_SIZE: tl.constexpr,
        NUM_SEGMENTS: tl.constexpr,
        MAX_TILES_PER_SEGMENT: tl.constexpr,
        HAS_SLIDING_WINDOW: tl.constexpr,
    ):
        """Tensor-core GQA decode for one (sequence, KV head, segment)."""
        seq_idx = tl.program_id(0)
        kv_head_idx = tl.program_id(1)
        segment_idx = tl.program_id(2)

        seq_len = tl.load(seq_lens_ptr + seq_idx)
        window_start = 0
        if HAS_SLIDING_WINDOW:
            window_start = tl.maximum(seq_len - window_size, 0)
        visible_tokens = seq_len - window_start
        tiles_per_segment = (
            visible_tokens + NUM_SEGMENTS * TILE_SIZE - 1
        ) // (NUM_SEGMENTS * TILE_SIZE)
        segment_start = (
            window_start
            + segment_idx * tiles_per_segment * TILE_SIZE
        )
        if segment_start >= seq_len:
            return

        m_offsets = tl.arange(0, BLOCK_M)
        d_offsets = tl.arange(0, HEAD_DIM)
        t_offsets = tl.arange(0, TILE_SIZE)
        q_head_offsets = kv_head_idx * GQA_RATIO + m_offsets
        q_mask = m_offsets < GQA_RATIO

        q_ptrs = (
            query_ptr
            + seq_idx * stride_qs
            + q_head_offsets[:, None] * stride_qh
            + d_offsets[None, :] * stride_qd
        )
        q = tl.load(
            q_ptrs,
            mask=q_mask[:, None],
            other=0.0,
        )

        m_prev = tl.full(
            [BLOCK_M],
            value=-1.0e20,
            dtype=tl.float32,
        )
        l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        for tile_idx in range(MAX_TILES_PER_SEGMENT):
            token_offsets = (
                segment_start + tile_idx * TILE_SIZE + t_offsets
            )
            token_mask = (
                (tile_idx < tiles_per_segment)
                & (token_offsets < seq_len)
            )
            logical_blocks = token_offsets // BLOCK_SIZE
            physical_blocks = tl.load(
                block_tables_ptr
                + seq_idx * stride_bs
                + logical_blocks,
                mask=token_mask,
                other=0,
            )
            offsets_in_block = token_offsets % BLOCK_SIZE

            k_ptrs = (
                kv_cache_ptr
                + physical_blocks[None, :] * stride_cb
                + 0 * stride_c2
                + kv_head_idx * stride_ch
                + offsets_in_block[None, :] * stride_ct
                + d_offsets[:, None] * stride_cd
            )
            k = tl.load(
                k_ptrs,
                mask=token_mask[None, :],
                other=0.0,
            )

            scores = tl.dot(
                q,
                k,
                out_dtype=tl.float32,
            ) * scale
            valid = q_mask[:, None] & token_mask[None, :]
            scores = tl.where(valid, scores, -1.0e20)

            m_cur = tl.max(scores, axis=1)
            m_new = tl.maximum(m_prev, m_cur)
            alpha = tl.exp(m_prev - m_new)
            probs = tl.exp(scores - m_new[:, None])
            probs = tl.where(valid, probs, 0.0)
            l_cur = tl.sum(probs, axis=1)

            v_ptrs = (
                kv_cache_ptr
                + physical_blocks[:, None] * stride_cb
                + 1 * stride_c2
                + kv_head_idx * stride_ch
                + offsets_in_block[:, None] * stride_ct
                + d_offsets[None, :] * stride_cd
            )
            v = tl.load(
                v_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )

            acc = acc * alpha[:, None] + tl.dot(
                probs.to(v.dtype),
                v,
                out_dtype=tl.float32,
            )
            l_prev = l_prev * alpha + l_cur
            m_prev = m_new

        acc_ptrs = (
            partial_acc_ptr
            + seq_idx * stride_pas
            + q_head_offsets[:, None] * stride_pah
            + segment_idx * stride_pap
            + d_offsets[None, :] * stride_pad
        )
        tl.store(acc_ptrs, acc, mask=q_mask[:, None])

        stat_ptrs = (
            seq_idx * stride_pms
            + q_head_offsets * stride_pmh
            + segment_idx * stride_pmp
        )
        tl.store(
            partial_m_ptr + stat_ptrs,
            m_prev,
            mask=q_mask,
        )
        tl.store(
            partial_l_ptr + stat_ptrs,
            l_prev,
            mask=q_mask,
        )

    @triton.jit
    def _paged_attn_decode_grouped_segment_reduce_kernel(
        output_ptr,
        partial_acc_ptr,
        partial_m_ptr,
        partial_l_ptr,
        seq_lens_ptr,
        window_size,
        stride_os,
        stride_oh,
        stride_od,
        stride_pas,
        stride_pah,
        stride_pap,
        stride_pad,
        stride_pms,
        stride_pmh,
        stride_pmp,
        HEAD_DIM: tl.constexpr,
        TILE_SIZE: tl.constexpr,
        NUM_SEGMENTS: tl.constexpr,
        HAS_SLIDING_WINDOW: tl.constexpr,
    ):
        """Reduce graph-safe segment partials for one query head."""
        seq_idx = tl.program_id(0)
        q_head_idx = tl.program_id(1)
        segment_offsets = tl.arange(0, NUM_SEGMENTS)
        d_offsets = tl.arange(0, HEAD_DIM)

        seq_len = tl.load(seq_lens_ptr + seq_idx)
        visible_tokens = seq_len
        if HAS_SLIDING_WINDOW:
            visible_tokens = tl.minimum(seq_len, window_size)
        tiles_per_segment = (
            visible_tokens + NUM_SEGMENTS * TILE_SIZE - 1
        ) // (NUM_SEGMENTS * TILE_SIZE)
        tokens_per_segment = tl.maximum(
            tiles_per_segment * TILE_SIZE,
            1,
        )
        active_segments = (
            visible_tokens + tokens_per_segment - 1
        ) // tokens_per_segment
        segment_mask = segment_offsets < active_segments

        stat_ptrs = (
            seq_idx * stride_pms
            + q_head_idx * stride_pmh
            + segment_offsets * stride_pmp
        )
        segment_m = tl.load(
            partial_m_ptr + stat_ptrs,
            mask=segment_mask,
            other=-1.0e20,
        ).to(tl.float32)
        m_global = tl.max(segment_m)
        segment_scale = tl.exp(segment_m - m_global)
        segment_l = tl.load(
            partial_l_ptr + stat_ptrs,
            mask=segment_mask,
            other=0.0,
        ).to(tl.float32)
        denom = tl.sum(segment_l * segment_scale)

        acc_ptrs = (
            partial_acc_ptr
            + seq_idx * stride_pas
            + q_head_idx * stride_pah
            + segment_offsets[:, None] * stride_pap
            + d_offsets[None, :] * stride_pad
        )
        segment_acc = tl.load(
            acc_ptrs,
            mask=segment_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        acc = tl.sum(
            segment_acc * segment_scale[:, None],
            axis=0,
        )
        out = tl.where(denom > 0.0, acc / denom, 0.0)

        out_ptrs = (
            output_ptr
            + seq_idx * stride_os
            + q_head_idx * stride_oh
            + d_offsets * stride_od
        )
        tl.store(out_ptrs, out.to(output_ptr.dtype.element_ty))

    # =========================================================================
    # Fused RoPE + Paged Attention Decode (FlashInfer approach)
    # =========================================================================

    @triton.jit
    def _paged_attn_decode_rope_kernel(
        output_ptr,
        query_ptr,          # [num_seqs, num_q_heads, head_dim] — RAW (pre-RoPE/Norm)
        kv_cache_ptr,
        block_tables_ptr,
        seq_lens_ptr,
        window_size,
        cos_ptr, sin_ptr,   # [max_pos, half_dim]
        pos_ptr,            # [num_seqs] — position for each sequence
        norm_weight_ptr,    # [head_dim] QK-Norm weight, or nullptr
        scale,
        norm_eps,           # RMSNorm epsilon
        stride_os, stride_oh,
        stride_qs, stride_qh,
        stride_cb, stride_c2, stride_ch, stride_ct, stride_cd,
        stride_bs,
        stride_cos_p, stride_cos_d,
        num_q_heads,
        num_kv_heads,
        max_blocks_per_seq,
            BLOCK_SIZE: tl.constexpr,
            HEAD_DIM: tl.constexpr,
            ROTARY_DIM: tl.constexpr,
            ROTARY_HALF_DIM: tl.constexpr,
            HALF_ROTATE: tl.constexpr,
        HAS_QK_NORM: tl.constexpr,
        HAS_SLIDING_WINDOW: tl.constexpr,
        BLOCK_UNROLL: tl.constexpr,
    ):
        """
        Fused QK-Norm + RoPE + Paged Attention Decode.
        Applies optional RMSNorm + RoPE to Q IN REGISTERS before attention.
        Saves up to 3 kernel launches per layer.
        """
        seq_idx = tl.program_id(0)
        q_head_idx = tl.program_id(1)

        gqa_ratio = num_q_heads // num_kv_heads
        kv_head_idx = q_head_idx // gqa_ratio

        seq_len = tl.load(seq_lens_ptr + seq_idx)
        num_seq_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        window_start = 0
        first_block = 0
        num_window_blocks = num_seq_blocks
        if HAS_SLIDING_WINDOW:
            window_start = tl.maximum(seq_len - window_size, 0)
            first_block = window_start // BLOCK_SIZE
            num_window_blocks = num_seq_blocks - first_block

        # Load query [HEAD_DIM]
        d_offsets = tl.arange(0, HEAD_DIM)
        q_base = query_ptr + seq_idx * stride_qs + q_head_idx * stride_qh
        q = tl.load(q_base + d_offsets).to(tl.float32)

        # === Optional QK-Norm: RMSNorm(q) in registers ===
        if HAS_QK_NORM:
            variance = tl.sum(q * q) / HEAD_DIM
            rms_scale = tl.rsqrt(variance + norm_eps)
            norm_w = tl.load(norm_weight_ptr + d_offsets).to(tl.float32)
            q = q * rms_scale * norm_w

        # === Apply RoPE to Q in registers — all ops on [HEAD_DIM] vectors ===
        pos = tl.load(pos_ptr + seq_idx)

        if HALF_ROTATE:
            in_rotary = d_offsets < ROTARY_DIM
            is_first = d_offsets < ROTARY_HALF_DIM
            partner = tl.where(is_first, d_offsets + ROTARY_HALF_DIM, d_offsets - ROTARY_HALF_DIM)
            cos_idx = tl.where(is_first, d_offsets, d_offsets - ROTARY_HALF_DIM)
        else:
            in_rotary = d_offsets < ROTARY_DIM
            is_first = (d_offsets % 2) == 0
            partner = tl.where(is_first, d_offsets + 1, d_offsets - 1)
            cos_idx = d_offsets // 2
        partner = tl.where(in_rotary, partner, d_offsets)
        cos_idx = tl.where(in_rotary, cos_idx, 0)

        # Get partner values (must use normalized partner if QK-Norm)
        if HAS_QK_NORM:
            # Same rms_scale applies to all elements of this head vector
            q_partner_raw = tl.load(q_base + partner).to(tl.float32)
            norm_w_partner = tl.load(norm_weight_ptr + partner).to(tl.float32)
            q_partner = q_partner_raw * rms_scale * norm_w_partner
        else:
            q_partner = tl.load(q_base + partner).to(tl.float32)

        cos_full = tl.load(cos_ptr + pos * stride_cos_p + cos_idx * stride_cos_d).to(tl.float32)
        sin_full = tl.load(sin_ptr + pos * stride_cos_p + cos_idx * stride_cos_d).to(tl.float32)

        q_rot = tl.where(is_first,
                         q * cos_full - q_partner * sin_full,
                         q * cos_full + q_partner * sin_full)
        q = tl.where(in_rotary, q_rot, q)

        q = q * scale

        # === Standard paged attention (identical to non-fused) ===
        m_prev = -1e20
        l_prev = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        t_offsets = tl.arange(0, BLOCK_SIZE)

        if BLOCK_UNROLL == 1:
            for local_block_idx in range(max_blocks_per_seq):
                block_idx = first_block + local_block_idx
                still_valid = local_block_idx < num_window_blocks
                phys_block = tl.load(
                    block_tables_ptr + seq_idx * stride_bs + block_idx,
                    mask=still_valid, other=0
                )
                block_start = block_idx * BLOCK_SIZE
                token_offsets = block_start + t_offsets
                mask = still_valid & (token_offsets < seq_len)
                if HAS_SLIDING_WINDOW:
                    mask = mask & (token_offsets >= window_start)

                k_base = phys_block * stride_cb + 0 * stride_c2 + kv_head_idx * stride_ch
                k_ptrs = kv_cache_ptr + k_base + t_offsets[:, None] * stride_ct + d_offsets[None, :] * stride_cd
                k = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

                scores = tl.sum(q[None, :] * k, axis=1)
                scores = tl.where(mask, scores, -1e20)

                m_cur = tl.max(scores)
                m_new = tl.maximum(m_prev, m_cur)
                alpha = tl.exp(m_prev - m_new)
                exp_scores = tl.exp(scores - m_new)
                exp_scores = tl.where(mask, exp_scores, 0.0)
                l_cur = tl.sum(exp_scores)

                v_base = phys_block * stride_cb + 1 * stride_c2 + kv_head_idx * stride_ch
                v_ptrs = kv_cache_ptr + v_base + t_offsets[:, None] * stride_ct + d_offsets[None, :] * stride_cd
                v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

                acc = acc * alpha + tl.sum(exp_scores[:, None] * v, axis=0)
                l_prev = l_prev * alpha + l_cur
                m_prev = m_new
        else:
            # Long-context decode repeatedly scans ~130+ 16-token blocks on T4.
            # Unroll by two logical blocks to reduce loop-control overhead without
            # changing the per-block online softmax math or increasing tile size.
            for block_pair_idx in range((max_blocks_per_seq + 1) // 2):
                for inner_block in tl.static_range(0, 2):
                    local_block_idx = block_pair_idx * 2 + inner_block
                    block_idx = first_block + local_block_idx
                    still_valid = local_block_idx < num_window_blocks
                    phys_block = tl.load(
                        block_tables_ptr + seq_idx * stride_bs + block_idx,
                        mask=still_valid, other=0
                    )
                    block_start = block_idx * BLOCK_SIZE
                    token_offsets = block_start + t_offsets
                    mask = still_valid & (token_offsets < seq_len)
                    if HAS_SLIDING_WINDOW:
                        mask = mask & (token_offsets >= window_start)

                    k_base = phys_block * stride_cb + 0 * stride_c2 + kv_head_idx * stride_ch
                    k_ptrs = kv_cache_ptr + k_base + t_offsets[:, None] * stride_ct + d_offsets[None, :] * stride_cd
                    k = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

                    scores = tl.sum(q[None, :] * k, axis=1)
                    scores = tl.where(mask, scores, -1e20)

                    m_cur = tl.max(scores)
                    m_new = tl.maximum(m_prev, m_cur)
                    alpha = tl.exp(m_prev - m_new)
                    exp_scores = tl.exp(scores - m_new)
                    exp_scores = tl.where(mask, exp_scores, 0.0)
                    l_cur = tl.sum(exp_scores)

                    v_base = phys_block * stride_cb + 1 * stride_c2 + kv_head_idx * stride_ch
                    v_ptrs = kv_cache_ptr + v_base + t_offsets[:, None] * stride_ct + d_offsets[None, :] * stride_cd
                    v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

                    acc = acc * alpha + tl.sum(exp_scores[:, None] * v, axis=0)
                    l_prev = l_prev * alpha + l_cur
                    m_prev = m_new

        acc = tl.where(l_prev > 0, acc / l_prev, 0.0)
        out_base = seq_idx * stride_os + q_head_idx * stride_oh
        tl.store(output_ptr + out_base + d_offsets, acc.to(output_ptr.dtype.element_ty))

    @triton.jit
    def _paged_attn_decode_rope_gqa2_kernel(
        output_ptr,
        query_ptr,
        kv_cache_ptr,
        block_tables_ptr,
        seq_lens_ptr,
        window_size,
        cos_ptr,
        sin_ptr,
        pos_ptr,
        norm_weight_ptr,
        scale,
        norm_eps,
        stride_os, stride_oh,
        stride_qs, stride_qh,
        stride_cb, stride_c2, stride_ch, stride_ct, stride_cd,
        stride_bs,
        stride_cos_p, stride_cos_d,
        num_q_heads,
        num_kv_heads,
        max_blocks_per_seq,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        ROTARY_HALF_DIM: tl.constexpr,
        HALF_ROTATE: tl.constexpr,
        HAS_QK_NORM: tl.constexpr,
        HAS_SLIDING_WINDOW: tl.constexpr,
    ):
        """
        Direct GQA decode: one program handles two adjacent Q heads that share
        one KV head. Long-context GQA is bandwidth-bound, so this avoids loading
        the same K/V block twice for Qwen-style grouped-query attention.
        """
        seq_idx = tl.program_id(0)
        q_pair_idx = tl.program_id(1)

        gqa_ratio = num_q_heads // num_kv_heads
        q_head0 = q_pair_idx * 2
        q_head1 = q_head0 + 1
        kv_head_idx = q_head0 // gqa_ratio
        has_q1 = (q_head1 < num_q_heads) & ((q_head1 // gqa_ratio) == kv_head_idx)

        seq_len = tl.load(seq_lens_ptr + seq_idx)
        num_seq_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        window_start = 0
        first_block = 0
        num_window_blocks = num_seq_blocks
        if HAS_SLIDING_WINDOW:
            window_start = tl.maximum(seq_len - window_size, 0)
            first_block = window_start // BLOCK_SIZE
            num_window_blocks = num_seq_blocks - first_block

        d_offsets = tl.arange(0, HEAD_DIM)
        q_base0 = query_ptr + seq_idx * stride_qs + q_head0 * stride_qh
        q_base1 = query_ptr + seq_idx * stride_qs + q_head1 * stride_qh
        q0 = tl.load(q_base0 + d_offsets).to(tl.float32)
        q1 = tl.load(q_base1 + d_offsets, mask=has_q1, other=0.0).to(tl.float32)

        if HAS_QK_NORM:
            variance0 = tl.sum(q0 * q0) / HEAD_DIM
            variance1 = tl.sum(q1 * q1) / HEAD_DIM
            rms0 = tl.rsqrt(variance0 + norm_eps)
            rms1 = tl.rsqrt(variance1 + norm_eps)
            norm_w = tl.load(norm_weight_ptr + d_offsets).to(tl.float32)
            q0 = q0 * rms0 * norm_w
            q1 = q1 * rms1 * norm_w

        pos = tl.load(pos_ptr + seq_idx)
        if HALF_ROTATE:
            in_rotary = d_offsets < ROTARY_DIM
            is_first = d_offsets < ROTARY_HALF_DIM
            partner = tl.where(is_first, d_offsets + ROTARY_HALF_DIM, d_offsets - ROTARY_HALF_DIM)
            cos_idx = tl.where(is_first, d_offsets, d_offsets - ROTARY_HALF_DIM)
        else:
            in_rotary = d_offsets < ROTARY_DIM
            is_first = (d_offsets % 2) == 0
            partner = tl.where(is_first, d_offsets + 1, d_offsets - 1)
            cos_idx = d_offsets // 2
        partner = tl.where(in_rotary, partner, d_offsets)
        cos_idx = tl.where(in_rotary, cos_idx, 0)

        q0_partner = tl.load(q_base0 + partner).to(tl.float32)
        q1_partner = tl.load(q_base1 + partner, mask=has_q1, other=0.0).to(tl.float32)
        if HAS_QK_NORM:
            norm_w_partner = tl.load(norm_weight_ptr + partner).to(tl.float32)
            q0_partner = q0_partner * rms0 * norm_w_partner
            q1_partner = q1_partner * rms1 * norm_w_partner

        cos_full = tl.load(cos_ptr + pos * stride_cos_p + cos_idx * stride_cos_d).to(tl.float32)
        sin_full = tl.load(sin_ptr + pos * stride_cos_p + cos_idx * stride_cos_d).to(tl.float32)
        q0_rot = tl.where(
            is_first,
            q0 * cos_full - q0_partner * sin_full,
            q0 * cos_full + q0_partner * sin_full,
        )
        q1_rot = tl.where(
            is_first,
            q1 * cos_full - q1_partner * sin_full,
            q1 * cos_full + q1_partner * sin_full,
        )
        q0 = tl.where(in_rotary, q0_rot, q0) * scale
        q1 = tl.where(in_rotary, q1_rot, q1) * scale

        m0 = -1e20
        m1 = -1e20
        l0 = 0.0
        l1 = 0.0
        acc0 = tl.zeros([HEAD_DIM], dtype=tl.float32)
        acc1 = tl.zeros([HEAD_DIM], dtype=tl.float32)
        t_offsets = tl.arange(0, BLOCK_SIZE)

        for local_block_idx in range(max_blocks_per_seq):
            block_idx = first_block + local_block_idx
            still_valid = local_block_idx < num_window_blocks
            phys_block = tl.load(
                block_tables_ptr + seq_idx * stride_bs + block_idx,
                mask=still_valid, other=0,
            )
            block_start = block_idx * BLOCK_SIZE
            token_offsets = block_start + t_offsets
            mask = still_valid & (token_offsets < seq_len)
            if HAS_SLIDING_WINDOW:
                mask = mask & (token_offsets >= window_start)

            k_base = phys_block * stride_cb + 0 * stride_c2 + kv_head_idx * stride_ch
            k_ptrs = kv_cache_ptr + k_base + t_offsets[:, None] * stride_ct + d_offsets[None, :] * stride_cd
            k = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

            scores0 = tl.sum(q0[None, :] * k, axis=1)
            scores1 = tl.sum(q1[None, :] * k, axis=1)
            scores0 = tl.where(mask, scores0, -1e20)
            scores1 = tl.where(mask & has_q1, scores1, -1e20)

            m0_cur = tl.max(scores0)
            m1_cur = tl.max(scores1)
            m0_new = tl.maximum(m0, m0_cur)
            m1_new = tl.maximum(m1, m1_cur)
            alpha0 = tl.exp(m0 - m0_new)
            alpha1 = tl.exp(m1 - m1_new)
            exp0 = tl.exp(scores0 - m0_new)
            exp1 = tl.exp(scores1 - m1_new)
            exp0 = tl.where(mask, exp0, 0.0)
            exp1 = tl.where(mask & has_q1, exp1, 0.0)
            l0_cur = tl.sum(exp0)
            l1_cur = tl.sum(exp1)

            v_base = phys_block * stride_cb + 1 * stride_c2 + kv_head_idx * stride_ch
            v_ptrs = kv_cache_ptr + v_base + t_offsets[:, None] * stride_ct + d_offsets[None, :] * stride_cd
            v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

            acc0 = acc0 * alpha0 + tl.sum(exp0[:, None] * v, axis=0)
            acc1 = acc1 * alpha1 + tl.sum(exp1[:, None] * v, axis=0)
            l0 = l0 * alpha0 + l0_cur
            l1 = l1 * alpha1 + l1_cur
            m0 = m0_new
            m1 = m1_new

        out0 = tl.where(l0 > 0, acc0 / l0, 0.0)
        out1 = tl.where(l1 > 0, acc1 / l1, 0.0)
        out_base0 = seq_idx * stride_os + q_head0 * stride_oh
        out_base1 = seq_idx * stride_os + q_head1 * stride_oh
        tl.store(output_ptr + out_base0 + d_offsets, out0.to(output_ptr.dtype.element_ty))
        tl.store(output_ptr + out_base1 + d_offsets, out1.to(output_ptr.dtype.element_ty), mask=has_q1)

    @triton.jit
    def _paged_attn_decode_rope_gqa4_kernel(
        output_ptr,
        query_ptr,
        kv_cache_ptr,
        block_tables_ptr,
        seq_lens_ptr,
        window_size,
        cos_ptr,
        sin_ptr,
        pos_ptr,
        norm_weight_ptr,
        scale,
        norm_eps,
        stride_os, stride_oh,
        stride_qs, stride_qh,
        stride_cb, stride_c2, stride_ch, stride_ct, stride_cd,
        stride_bs,
        stride_cos_p, stride_cos_d,
        num_q_heads,
        num_kv_heads,
        max_blocks_per_seq,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        ROTARY_HALF_DIM: tl.constexpr,
        HALF_ROTATE: tl.constexpr,
        HAS_QK_NORM: tl.constexpr,
        HAS_SLIDING_WINDOW: tl.constexpr,
    ):
        """
        Direct GQA4 decode: one program handles four adjacent Q heads that
        share one KV head. This is an opt-in T4/Qwen long-context experiment:
        less repeated K/V traffic, more registers per program.
        """
        seq_idx = tl.program_id(0)
        q_group_idx = tl.program_id(1)

        gqa_ratio = num_q_heads // num_kv_heads
        q_head0 = q_group_idx * 4
        q_head1 = q_head0 + 1
        q_head2 = q_head0 + 2
        q_head3 = q_head0 + 3
        kv_head_idx = q_head0 // gqa_ratio
        has_q0 = q_head0 < num_q_heads
        has_q1 = (q_head1 < num_q_heads) & ((q_head1 // gqa_ratio) == kv_head_idx)
        has_q2 = (q_head2 < num_q_heads) & ((q_head2 // gqa_ratio) == kv_head_idx)
        has_q3 = (q_head3 < num_q_heads) & ((q_head3 // gqa_ratio) == kv_head_idx)

        seq_len = tl.load(seq_lens_ptr + seq_idx)
        num_seq_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        window_start = 0
        first_block = 0
        num_window_blocks = num_seq_blocks
        if HAS_SLIDING_WINDOW:
            window_start = tl.maximum(seq_len - window_size, 0)
            first_block = window_start // BLOCK_SIZE
            num_window_blocks = num_seq_blocks - first_block

        d_offsets = tl.arange(0, HEAD_DIM)
        q_base0 = query_ptr + seq_idx * stride_qs + q_head0 * stride_qh
        q_base1 = query_ptr + seq_idx * stride_qs + q_head1 * stride_qh
        q_base2 = query_ptr + seq_idx * stride_qs + q_head2 * stride_qh
        q_base3 = query_ptr + seq_idx * stride_qs + q_head3 * stride_qh
        q0 = tl.load(q_base0 + d_offsets, mask=has_q0, other=0.0).to(tl.float32)
        q1 = tl.load(q_base1 + d_offsets, mask=has_q1, other=0.0).to(tl.float32)
        q2 = tl.load(q_base2 + d_offsets, mask=has_q2, other=0.0).to(tl.float32)
        q3 = tl.load(q_base3 + d_offsets, mask=has_q3, other=0.0).to(tl.float32)

        if HAS_QK_NORM:
            variance0 = tl.sum(q0 * q0) / HEAD_DIM
            variance1 = tl.sum(q1 * q1) / HEAD_DIM
            variance2 = tl.sum(q2 * q2) / HEAD_DIM
            variance3 = tl.sum(q3 * q3) / HEAD_DIM
            rms0 = tl.rsqrt(variance0 + norm_eps)
            rms1 = tl.rsqrt(variance1 + norm_eps)
            rms2 = tl.rsqrt(variance2 + norm_eps)
            rms3 = tl.rsqrt(variance3 + norm_eps)
            norm_w = tl.load(norm_weight_ptr + d_offsets).to(tl.float32)
            q0 = q0 * rms0 * norm_w
            q1 = q1 * rms1 * norm_w
            q2 = q2 * rms2 * norm_w
            q3 = q3 * rms3 * norm_w

        pos = tl.load(pos_ptr + seq_idx)
        if HALF_ROTATE:
            in_rotary = d_offsets < ROTARY_DIM
            is_first = d_offsets < ROTARY_HALF_DIM
            partner = tl.where(is_first, d_offsets + ROTARY_HALF_DIM, d_offsets - ROTARY_HALF_DIM)
            cos_idx = tl.where(is_first, d_offsets, d_offsets - ROTARY_HALF_DIM)
        else:
            in_rotary = d_offsets < ROTARY_DIM
            is_first = (d_offsets % 2) == 0
            partner = tl.where(is_first, d_offsets + 1, d_offsets - 1)
            cos_idx = d_offsets // 2
        partner = tl.where(in_rotary, partner, d_offsets)
        cos_idx = tl.where(in_rotary, cos_idx, 0)

        q0_partner = tl.load(q_base0 + partner, mask=has_q0, other=0.0).to(tl.float32)
        q1_partner = tl.load(q_base1 + partner, mask=has_q1, other=0.0).to(tl.float32)
        q2_partner = tl.load(q_base2 + partner, mask=has_q2, other=0.0).to(tl.float32)
        q3_partner = tl.load(q_base3 + partner, mask=has_q3, other=0.0).to(tl.float32)
        if HAS_QK_NORM:
            norm_w_partner = tl.load(norm_weight_ptr + partner).to(tl.float32)
            q0_partner = q0_partner * rms0 * norm_w_partner
            q1_partner = q1_partner * rms1 * norm_w_partner
            q2_partner = q2_partner * rms2 * norm_w_partner
            q3_partner = q3_partner * rms3 * norm_w_partner

        cos_full = tl.load(cos_ptr + pos * stride_cos_p + cos_idx * stride_cos_d).to(tl.float32)
        sin_full = tl.load(sin_ptr + pos * stride_cos_p + cos_idx * stride_cos_d).to(tl.float32)
        q0_rot = tl.where(is_first, q0 * cos_full - q0_partner * sin_full, q0 * cos_full + q0_partner * sin_full)
        q1_rot = tl.where(is_first, q1 * cos_full - q1_partner * sin_full, q1 * cos_full + q1_partner * sin_full)
        q2_rot = tl.where(is_first, q2 * cos_full - q2_partner * sin_full, q2 * cos_full + q2_partner * sin_full)
        q3_rot = tl.where(is_first, q3 * cos_full - q3_partner * sin_full, q3 * cos_full + q3_partner * sin_full)
        q0 = tl.where(in_rotary, q0_rot, q0) * scale
        q1 = tl.where(in_rotary, q1_rot, q1) * scale
        q2 = tl.where(in_rotary, q2_rot, q2) * scale
        q3 = tl.where(in_rotary, q3_rot, q3) * scale

        m0 = -1e20
        m1 = -1e20
        m2 = -1e20
        m3 = -1e20
        l0 = 0.0
        l1 = 0.0
        l2 = 0.0
        l3 = 0.0
        acc0 = tl.zeros([HEAD_DIM], dtype=tl.float32)
        acc1 = tl.zeros([HEAD_DIM], dtype=tl.float32)
        acc2 = tl.zeros([HEAD_DIM], dtype=tl.float32)
        acc3 = tl.zeros([HEAD_DIM], dtype=tl.float32)
        t_offsets = tl.arange(0, BLOCK_SIZE)

        for local_block_idx in range(max_blocks_per_seq):
            block_idx = first_block + local_block_idx
            still_valid = local_block_idx < num_window_blocks
            phys_block = tl.load(
                block_tables_ptr + seq_idx * stride_bs + block_idx,
                mask=still_valid, other=0,
            )
            block_start = block_idx * BLOCK_SIZE
            token_offsets = block_start + t_offsets
            mask = still_valid & (token_offsets < seq_len)
            if HAS_SLIDING_WINDOW:
                mask = mask & (token_offsets >= window_start)

            k_base = phys_block * stride_cb + 0 * stride_c2 + kv_head_idx * stride_ch
            k_ptrs = kv_cache_ptr + k_base + t_offsets[:, None] * stride_ct + d_offsets[None, :] * stride_cd
            k = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

            scores0 = tl.sum(q0[None, :] * k, axis=1)
            scores1 = tl.sum(q1[None, :] * k, axis=1)
            scores2 = tl.sum(q2[None, :] * k, axis=1)
            scores3 = tl.sum(q3[None, :] * k, axis=1)
            scores0 = tl.where(mask & has_q0, scores0, -1e20)
            scores1 = tl.where(mask & has_q1, scores1, -1e20)
            scores2 = tl.where(mask & has_q2, scores2, -1e20)
            scores3 = tl.where(mask & has_q3, scores3, -1e20)

            m0_cur = tl.max(scores0)
            m1_cur = tl.max(scores1)
            m2_cur = tl.max(scores2)
            m3_cur = tl.max(scores3)
            m0_new = tl.maximum(m0, m0_cur)
            m1_new = tl.maximum(m1, m1_cur)
            m2_new = tl.maximum(m2, m2_cur)
            m3_new = tl.maximum(m3, m3_cur)
            alpha0 = tl.exp(m0 - m0_new)
            alpha1 = tl.exp(m1 - m1_new)
            alpha2 = tl.exp(m2 - m2_new)
            alpha3 = tl.exp(m3 - m3_new)
            exp0 = tl.exp(scores0 - m0_new)
            exp1 = tl.exp(scores1 - m1_new)
            exp2 = tl.exp(scores2 - m2_new)
            exp3 = tl.exp(scores3 - m3_new)
            exp0 = tl.where(mask & has_q0, exp0, 0.0)
            exp1 = tl.where(mask & has_q1, exp1, 0.0)
            exp2 = tl.where(mask & has_q2, exp2, 0.0)
            exp3 = tl.where(mask & has_q3, exp3, 0.0)
            l0_cur = tl.sum(exp0)
            l1_cur = tl.sum(exp1)
            l2_cur = tl.sum(exp2)
            l3_cur = tl.sum(exp3)

            v_base = phys_block * stride_cb + 1 * stride_c2 + kv_head_idx * stride_ch
            v_ptrs = kv_cache_ptr + v_base + t_offsets[:, None] * stride_ct + d_offsets[None, :] * stride_cd
            v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

            acc0 = acc0 * alpha0 + tl.sum(exp0[:, None] * v, axis=0)
            acc1 = acc1 * alpha1 + tl.sum(exp1[:, None] * v, axis=0)
            acc2 = acc2 * alpha2 + tl.sum(exp2[:, None] * v, axis=0)
            acc3 = acc3 * alpha3 + tl.sum(exp3[:, None] * v, axis=0)
            l0 = l0 * alpha0 + l0_cur
            l1 = l1 * alpha1 + l1_cur
            l2 = l2 * alpha2 + l2_cur
            l3 = l3 * alpha3 + l3_cur
            m0 = m0_new
            m1 = m1_new
            m2 = m2_new
            m3 = m3_new

        out0 = tl.where(l0 > 0, acc0 / l0, 0.0)
        out1 = tl.where(l1 > 0, acc1 / l1, 0.0)
        out2 = tl.where(l2 > 0, acc2 / l2, 0.0)
        out3 = tl.where(l3 > 0, acc3 / l3, 0.0)
        out_base0 = seq_idx * stride_os + q_head0 * stride_oh
        out_base1 = seq_idx * stride_os + q_head1 * stride_oh
        out_base2 = seq_idx * stride_os + q_head2 * stride_oh
        out_base3 = seq_idx * stride_os + q_head3 * stride_oh
        tl.store(output_ptr + out_base0 + d_offsets, out0.to(output_ptr.dtype.element_ty), mask=has_q0)
        tl.store(output_ptr + out_base1 + d_offsets, out1.to(output_ptr.dtype.element_ty), mask=has_q1)
        tl.store(output_ptr + out_base2 + d_offsets, out2.to(output_ptr.dtype.element_ty), mask=has_q2)
        tl.store(output_ptr + out_base3 + d_offsets, out3.to(output_ptr.dtype.element_ty), mask=has_q3)

    @triton.jit
    def _paged_attn_decode_rope_split_kernel(
        partial_acc_ptr,
        partial_m_ptr,
        partial_l_ptr,
        query_ptr,
        kv_cache_ptr,
        block_tables_ptr,
        seq_lens_ptr,
        window_size,
        cos_ptr,
        sin_ptr,
        pos_ptr,
        norm_weight_ptr,
        scale,
        norm_eps,
        stride_pas, stride_pah, stride_pap, stride_pad,
        stride_pms, stride_pmh, stride_pmp,
        stride_qs, stride_qh,
        stride_cb, stride_c2, stride_ch, stride_ct, stride_cd,
        stride_bs,
        stride_cos_p, stride_cos_d,
        num_q_heads,
        num_kv_heads,
        BLOCK_SIZE: tl.constexpr,
            HEAD_DIM: tl.constexpr,
            ROTARY_DIM: tl.constexpr,
            ROTARY_HALF_DIM: tl.constexpr,
            HALF_ROTATE: tl.constexpr,
        HAS_QK_NORM: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
        HAS_SLIDING_WINDOW: tl.constexpr,
    ):
        """
        Fused QK-Norm + RoPE + split-parallel paged decode (phase 1).
        """
        seq_idx = tl.program_id(0)
        q_head_idx = tl.program_id(1)
        split_idx = tl.program_id(2)

        gqa_ratio = num_q_heads // num_kv_heads
        kv_head_idx = q_head_idx // gqa_ratio

        seq_len = tl.load(seq_lens_ptr + seq_idx)
        num_seq_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        window_start = 0
        first_block = 0
        num_window_blocks = num_seq_blocks
        if HAS_SLIDING_WINDOW:
            window_start = tl.maximum(seq_len - window_size, 0)
            first_block = window_start // BLOCK_SIZE
            num_window_blocks = num_seq_blocks - first_block
        split_start = split_idx * BLOCKS_PER_SPLIT

        d_offsets = tl.arange(0, HEAD_DIM)
        q_base = query_ptr + seq_idx * stride_qs + q_head_idx * stride_qh
        q = tl.load(q_base + d_offsets).to(tl.float32)

        if HAS_QK_NORM:
            variance = tl.sum(q * q) / HEAD_DIM
            rms_scale = tl.rsqrt(variance + norm_eps)
            norm_w = tl.load(norm_weight_ptr + d_offsets).to(tl.float32)
            q = q * rms_scale * norm_w

        pos = tl.load(pos_ptr + seq_idx)
        if HALF_ROTATE:
            in_rotary = d_offsets < ROTARY_DIM
            is_first = d_offsets < ROTARY_HALF_DIM
            partner = tl.where(is_first, d_offsets + ROTARY_HALF_DIM, d_offsets - ROTARY_HALF_DIM)
            cos_idx = tl.where(is_first, d_offsets, d_offsets - ROTARY_HALF_DIM)
        else:
            in_rotary = d_offsets < ROTARY_DIM
            is_first = (d_offsets % 2) == 0
            partner = tl.where(is_first, d_offsets + 1, d_offsets - 1)
            cos_idx = d_offsets // 2
        partner = tl.where(in_rotary, partner, d_offsets)
        cos_idx = tl.where(in_rotary, cos_idx, 0)

        if HAS_QK_NORM:
            q_partner_raw = tl.load(q_base + partner).to(tl.float32)
            norm_w_partner = tl.load(norm_weight_ptr + partner).to(tl.float32)
            q_partner = q_partner_raw * rms_scale * norm_w_partner
        else:
            q_partner = tl.load(q_base + partner).to(tl.float32)

        cos_full = tl.load(cos_ptr + pos * stride_cos_p + cos_idx * stride_cos_d).to(tl.float32)
        sin_full = tl.load(sin_ptr + pos * stride_cos_p + cos_idx * stride_cos_d).to(tl.float32)
        q_rot = tl.where(
            is_first,
            q * cos_full - q_partner * sin_full,
            q * cos_full + q_partner * sin_full,
        )
        q = tl.where(in_rotary, q_rot, q)
        q = q * scale

        m_prev = -1e20
        l_prev = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        t_offsets = tl.arange(0, BLOCK_SIZE)

        for local_block_idx in range(BLOCKS_PER_SPLIT):
            block_offset = split_start + local_block_idx
            block_idx = first_block + block_offset
            still_valid = block_offset < num_window_blocks
            phys_block = tl.load(
                block_tables_ptr + seq_idx * stride_bs + block_idx,
                mask=still_valid, other=0,
            )
            block_start = block_idx * BLOCK_SIZE
            token_offsets = block_start + t_offsets
            mask = still_valid & (token_offsets < seq_len)
            if HAS_SLIDING_WINDOW:
                mask = mask & (token_offsets >= window_start)

            k_base = phys_block * stride_cb + 0 * stride_c2 + kv_head_idx * stride_ch
            k_ptrs = kv_cache_ptr + k_base + t_offsets[:, None] * stride_ct + d_offsets[None, :] * stride_cd
            k = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
            scores = tl.sum(q[None, :] * k, axis=1)
            scores = tl.where(mask, scores, -1e20)

            m_cur = tl.max(scores)
            m_new = tl.maximum(m_prev, m_cur)
            alpha = tl.exp(m_prev - m_new)
            exp_scores = tl.exp(scores - m_new)
            exp_scores = tl.where(mask, exp_scores, 0.0)
            l_cur = tl.sum(exp_scores)

            v_base = phys_block * stride_cb + 1 * stride_c2 + kv_head_idx * stride_ch
            v_ptrs = kv_cache_ptr + v_base + t_offsets[:, None] * stride_ct + d_offsets[None, :] * stride_cd
            v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

            acc = acc * alpha + tl.sum(exp_scores[:, None] * v, axis=0)
            l_prev = l_prev * alpha + l_cur
            m_prev = m_new

        acc_base = seq_idx * stride_pas + q_head_idx * stride_pah + split_idx * stride_pap
        tl.store(partial_acc_ptr + acc_base + d_offsets * stride_pad, acc)
        stat_base = seq_idx * stride_pms + q_head_idx * stride_pmh + split_idx * stride_pmp
        tl.store(partial_m_ptr + stat_base, m_prev)
        tl.store(partial_l_ptr + stat_base, l_prev)

    @triton.jit
    def _paged_attn_decode_rope_gqa2_split_kernel(
        partial_acc_ptr,
        partial_m_ptr,
        partial_l_ptr,
        query_ptr,
        kv_cache_ptr,
        block_tables_ptr,
        seq_lens_ptr,
        window_size,
        cos_ptr,
        sin_ptr,
        pos_ptr,
        norm_weight_ptr,
        scale,
        norm_eps,
        stride_pas, stride_pah, stride_pap, stride_pad,
        stride_pms, stride_pmh, stride_pmp,
        stride_qs, stride_qh,
        stride_cb, stride_c2, stride_ch, stride_ct, stride_cd,
        stride_bs,
        stride_cos_p, stride_cos_d,
        num_q_heads,
        num_kv_heads,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        ROTARY_HALF_DIM: tl.constexpr,
        HALF_ROTATE: tl.constexpr,
        HAS_QK_NORM: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
        HAS_SLIDING_WINDOW: tl.constexpr,
    ):
        """
        GQA-specialized split decode: one program handles two adjacent Q heads
        that share the same KV head. This halves KV block loads for Qwen-style
        GQA without changing the public partial/reduce layout.
        """
        seq_idx = tl.program_id(0)
        q_pair_idx = tl.program_id(1)
        split_idx = tl.program_id(2)

        gqa_ratio = num_q_heads // num_kv_heads
        q_head0 = q_pair_idx * 2
        q_head1 = q_head0 + 1
        kv_head_idx = q_head0 // gqa_ratio
        has_q1 = (q_head1 < num_q_heads) & ((q_head1 // gqa_ratio) == kv_head_idx)

        seq_len = tl.load(seq_lens_ptr + seq_idx)
        num_seq_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        window_start = 0
        first_block = 0
        num_window_blocks = num_seq_blocks
        if HAS_SLIDING_WINDOW:
            window_start = tl.maximum(seq_len - window_size, 0)
            first_block = window_start // BLOCK_SIZE
            num_window_blocks = num_seq_blocks - first_block
        split_start = split_idx * BLOCKS_PER_SPLIT

        d_offsets = tl.arange(0, HEAD_DIM)
        q_base0 = query_ptr + seq_idx * stride_qs + q_head0 * stride_qh
        q_base1 = query_ptr + seq_idx * stride_qs + q_head1 * stride_qh
        q0 = tl.load(q_base0 + d_offsets).to(tl.float32)
        q1 = tl.load(q_base1 + d_offsets, mask=has_q1, other=0.0).to(tl.float32)

        if HAS_QK_NORM:
            variance0 = tl.sum(q0 * q0) / HEAD_DIM
            variance1 = tl.sum(q1 * q1) / HEAD_DIM
            rms0 = tl.rsqrt(variance0 + norm_eps)
            rms1 = tl.rsqrt(variance1 + norm_eps)
            norm_w = tl.load(norm_weight_ptr + d_offsets).to(tl.float32)
            q0 = q0 * rms0 * norm_w
            q1 = q1 * rms1 * norm_w

        pos = tl.load(pos_ptr + seq_idx)
        if HALF_ROTATE:
            in_rotary = d_offsets < ROTARY_DIM
            is_first = d_offsets < ROTARY_HALF_DIM
            partner = tl.where(is_first, d_offsets + ROTARY_HALF_DIM, d_offsets - ROTARY_HALF_DIM)
            cos_idx = tl.where(is_first, d_offsets, d_offsets - ROTARY_HALF_DIM)
        else:
            in_rotary = d_offsets < ROTARY_DIM
            is_first = (d_offsets % 2) == 0
            partner = tl.where(is_first, d_offsets + 1, d_offsets - 1)
            cos_idx = d_offsets // 2
        partner = tl.where(in_rotary, partner, d_offsets)
        cos_idx = tl.where(in_rotary, cos_idx, 0)

        q0_partner = tl.load(q_base0 + partner).to(tl.float32)
        q1_partner = tl.load(q_base1 + partner, mask=has_q1, other=0.0).to(tl.float32)
        if HAS_QK_NORM:
            norm_w_partner = tl.load(norm_weight_ptr + partner).to(tl.float32)
            q0_partner = q0_partner * rms0 * norm_w_partner
            q1_partner = q1_partner * rms1 * norm_w_partner

        cos_full = tl.load(cos_ptr + pos * stride_cos_p + cos_idx * stride_cos_d).to(tl.float32)
        sin_full = tl.load(sin_ptr + pos * stride_cos_p + cos_idx * stride_cos_d).to(tl.float32)
        q0_rot = tl.where(
            is_first,
            q0 * cos_full - q0_partner * sin_full,
            q0 * cos_full + q0_partner * sin_full,
        )
        q1_rot = tl.where(
            is_first,
            q1 * cos_full - q1_partner * sin_full,
            q1 * cos_full + q1_partner * sin_full,
        )
        q0 = tl.where(in_rotary, q0_rot, q0) * scale
        q1 = tl.where(in_rotary, q1_rot, q1) * scale

        m0 = -1e20
        m1 = -1e20
        l0 = 0.0
        l1 = 0.0
        acc0 = tl.zeros([HEAD_DIM], dtype=tl.float32)
        acc1 = tl.zeros([HEAD_DIM], dtype=tl.float32)
        t_offsets = tl.arange(0, BLOCK_SIZE)

        for local_block_idx in range(BLOCKS_PER_SPLIT):
            block_offset = split_start + local_block_idx
            block_idx = first_block + block_offset
            still_valid = block_offset < num_window_blocks
            phys_block = tl.load(
                block_tables_ptr + seq_idx * stride_bs + block_idx,
                mask=still_valid, other=0,
            )
            block_start = block_idx * BLOCK_SIZE
            token_offsets = block_start + t_offsets
            mask = still_valid & (token_offsets < seq_len)
            if HAS_SLIDING_WINDOW:
                mask = mask & (token_offsets >= window_start)

            k_base = phys_block * stride_cb + 0 * stride_c2 + kv_head_idx * stride_ch
            k_ptrs = kv_cache_ptr + k_base + t_offsets[:, None] * stride_ct + d_offsets[None, :] * stride_cd
            k = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

            scores0 = tl.sum(q0[None, :] * k, axis=1)
            scores1 = tl.sum(q1[None, :] * k, axis=1)
            scores0 = tl.where(mask, scores0, -1e20)
            scores1 = tl.where(mask & has_q1, scores1, -1e20)

            m0_cur = tl.max(scores0)
            m1_cur = tl.max(scores1)
            m0_new = tl.maximum(m0, m0_cur)
            m1_new = tl.maximum(m1, m1_cur)
            alpha0 = tl.exp(m0 - m0_new)
            alpha1 = tl.exp(m1 - m1_new)
            exp0 = tl.exp(scores0 - m0_new)
            exp1 = tl.exp(scores1 - m1_new)
            exp0 = tl.where(mask, exp0, 0.0)
            exp1 = tl.where(mask & has_q1, exp1, 0.0)
            l0_cur = tl.sum(exp0)
            l1_cur = tl.sum(exp1)

            v_base = phys_block * stride_cb + 1 * stride_c2 + kv_head_idx * stride_ch
            v_ptrs = kv_cache_ptr + v_base + t_offsets[:, None] * stride_ct + d_offsets[None, :] * stride_cd
            v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

            acc0 = acc0 * alpha0 + tl.sum(exp0[:, None] * v, axis=0)
            acc1 = acc1 * alpha1 + tl.sum(exp1[:, None] * v, axis=0)
            l0 = l0 * alpha0 + l0_cur
            l1 = l1 * alpha1 + l1_cur
            m0 = m0_new
            m1 = m1_new

        acc_base0 = seq_idx * stride_pas + q_head0 * stride_pah + split_idx * stride_pap
        stat_base0 = seq_idx * stride_pms + q_head0 * stride_pmh + split_idx * stride_pmp
        tl.store(partial_acc_ptr + acc_base0 + d_offsets * stride_pad, acc0)
        tl.store(partial_m_ptr + stat_base0, m0)
        tl.store(partial_l_ptr + stat_base0, l0)

        acc_base1 = seq_idx * stride_pas + q_head1 * stride_pah + split_idx * stride_pap
        stat_base1 = seq_idx * stride_pms + q_head1 * stride_pmh + split_idx * stride_pmp
        tl.store(partial_acc_ptr + acc_base1 + d_offsets * stride_pad, acc1, mask=has_q1)
        tl.store(partial_m_ptr + stat_base1, m1, mask=has_q1)
        tl.store(partial_l_ptr + stat_base1, l1, mask=has_q1)

    @triton.jit
    def _paged_attn_decode_rope_gqa4_split_kernel(
        partial_acc_ptr,
        partial_m_ptr,
        partial_l_ptr,
        query_ptr,
        kv_cache_ptr,
        block_tables_ptr,
        seq_lens_ptr,
        window_size,
        cos_ptr,
        sin_ptr,
        pos_ptr,
        norm_weight_ptr,
        scale,
        norm_eps,
        stride_pas, stride_pah, stride_pap, stride_pad,
        stride_pms, stride_pmh, stride_pmp,
        stride_qs, stride_qh,
        stride_cb, stride_c2, stride_ch, stride_ct, stride_cd,
        stride_bs,
        stride_cos_p, stride_cos_d,
        num_q_heads,
        num_kv_heads,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        ROTARY_HALF_DIM: tl.constexpr,
        HALF_ROTATE: tl.constexpr,
        HAS_QK_NORM: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
        HAS_SLIDING_WINDOW: tl.constexpr,
    ):
        """
        GQA4 split decode: one program handles four adjacent Q heads sharing
        one KV head. On Qwen3 GQA=8 this halves KV reloads versus GQA2 while
        keeping split-K parallelism for long contexts.
        """
        seq_idx = tl.program_id(0)
        q_group_idx = tl.program_id(1)
        split_idx = tl.program_id(2)

        gqa_ratio = num_q_heads // num_kv_heads
        q_head0 = q_group_idx * 4
        q_head1 = q_head0 + 1
        q_head2 = q_head0 + 2
        q_head3 = q_head0 + 3
        kv_head_idx = q_head0 // gqa_ratio
        has_q0 = q_head0 < num_q_heads
        has_q1 = (q_head1 < num_q_heads) & ((q_head1 // gqa_ratio) == kv_head_idx)
        has_q2 = (q_head2 < num_q_heads) & ((q_head2 // gqa_ratio) == kv_head_idx)
        has_q3 = (q_head3 < num_q_heads) & ((q_head3 // gqa_ratio) == kv_head_idx)

        seq_len = tl.load(seq_lens_ptr + seq_idx)
        num_seq_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        window_start = 0
        first_block = 0
        num_window_blocks = num_seq_blocks
        if HAS_SLIDING_WINDOW:
            window_start = tl.maximum(seq_len - window_size, 0)
            first_block = window_start // BLOCK_SIZE
            num_window_blocks = num_seq_blocks - first_block
        split_start = split_idx * BLOCKS_PER_SPLIT

        d_offsets = tl.arange(0, HEAD_DIM)
        q_base0 = query_ptr + seq_idx * stride_qs + q_head0 * stride_qh
        q_base1 = query_ptr + seq_idx * stride_qs + q_head1 * stride_qh
        q_base2 = query_ptr + seq_idx * stride_qs + q_head2 * stride_qh
        q_base3 = query_ptr + seq_idx * stride_qs + q_head3 * stride_qh
        q0 = tl.load(q_base0 + d_offsets, mask=has_q0, other=0.0).to(tl.float32)
        q1 = tl.load(q_base1 + d_offsets, mask=has_q1, other=0.0).to(tl.float32)
        q2 = tl.load(q_base2 + d_offsets, mask=has_q2, other=0.0).to(tl.float32)
        q3 = tl.load(q_base3 + d_offsets, mask=has_q3, other=0.0).to(tl.float32)

        if HAS_QK_NORM:
            variance0 = tl.sum(q0 * q0) / HEAD_DIM
            variance1 = tl.sum(q1 * q1) / HEAD_DIM
            variance2 = tl.sum(q2 * q2) / HEAD_DIM
            variance3 = tl.sum(q3 * q3) / HEAD_DIM
            rms0 = tl.rsqrt(variance0 + norm_eps)
            rms1 = tl.rsqrt(variance1 + norm_eps)
            rms2 = tl.rsqrt(variance2 + norm_eps)
            rms3 = tl.rsqrt(variance3 + norm_eps)
            norm_w = tl.load(norm_weight_ptr + d_offsets).to(tl.float32)
            q0 = q0 * rms0 * norm_w
            q1 = q1 * rms1 * norm_w
            q2 = q2 * rms2 * norm_w
            q3 = q3 * rms3 * norm_w

        pos = tl.load(pos_ptr + seq_idx)
        if HALF_ROTATE:
            in_rotary = d_offsets < ROTARY_DIM
            is_first = d_offsets < ROTARY_HALF_DIM
            partner = tl.where(is_first, d_offsets + ROTARY_HALF_DIM, d_offsets - ROTARY_HALF_DIM)
            cos_idx = tl.where(is_first, d_offsets, d_offsets - ROTARY_HALF_DIM)
        else:
            in_rotary = d_offsets < ROTARY_DIM
            is_first = (d_offsets % 2) == 0
            partner = tl.where(is_first, d_offsets + 1, d_offsets - 1)
            cos_idx = d_offsets // 2
        partner = tl.where(in_rotary, partner, d_offsets)
        cos_idx = tl.where(in_rotary, cos_idx, 0)

        q0_partner = tl.load(q_base0 + partner, mask=has_q0, other=0.0).to(tl.float32)
        q1_partner = tl.load(q_base1 + partner, mask=has_q1, other=0.0).to(tl.float32)
        q2_partner = tl.load(q_base2 + partner, mask=has_q2, other=0.0).to(tl.float32)
        q3_partner = tl.load(q_base3 + partner, mask=has_q3, other=0.0).to(tl.float32)
        if HAS_QK_NORM:
            norm_w_partner = tl.load(norm_weight_ptr + partner).to(tl.float32)
            q0_partner = q0_partner * rms0 * norm_w_partner
            q1_partner = q1_partner * rms1 * norm_w_partner
            q2_partner = q2_partner * rms2 * norm_w_partner
            q3_partner = q3_partner * rms3 * norm_w_partner

        cos_full = tl.load(cos_ptr + pos * stride_cos_p + cos_idx * stride_cos_d).to(tl.float32)
        sin_full = tl.load(sin_ptr + pos * stride_cos_p + cos_idx * stride_cos_d).to(tl.float32)
        q0_rot = tl.where(is_first, q0 * cos_full - q0_partner * sin_full, q0 * cos_full + q0_partner * sin_full)
        q1_rot = tl.where(is_first, q1 * cos_full - q1_partner * sin_full, q1 * cos_full + q1_partner * sin_full)
        q2_rot = tl.where(is_first, q2 * cos_full - q2_partner * sin_full, q2 * cos_full + q2_partner * sin_full)
        q3_rot = tl.where(is_first, q3 * cos_full - q3_partner * sin_full, q3 * cos_full + q3_partner * sin_full)
        q0 = tl.where(in_rotary, q0_rot, q0) * scale
        q1 = tl.where(in_rotary, q1_rot, q1) * scale
        q2 = tl.where(in_rotary, q2_rot, q2) * scale
        q3 = tl.where(in_rotary, q3_rot, q3) * scale

        m0 = -1e20
        m1 = -1e20
        m2 = -1e20
        m3 = -1e20
        l0 = 0.0
        l1 = 0.0
        l2 = 0.0
        l3 = 0.0
        acc0 = tl.zeros([HEAD_DIM], dtype=tl.float32)
        acc1 = tl.zeros([HEAD_DIM], dtype=tl.float32)
        acc2 = tl.zeros([HEAD_DIM], dtype=tl.float32)
        acc3 = tl.zeros([HEAD_DIM], dtype=tl.float32)
        t_offsets = tl.arange(0, BLOCK_SIZE)

        for local_block_idx in range(BLOCKS_PER_SPLIT):
            block_offset = split_start + local_block_idx
            block_idx = first_block + block_offset
            still_valid = block_offset < num_window_blocks
            phys_block = tl.load(
                block_tables_ptr + seq_idx * stride_bs + block_idx,
                mask=still_valid, other=0,
            )
            block_start = block_idx * BLOCK_SIZE
            token_offsets = block_start + t_offsets
            mask = still_valid & (token_offsets < seq_len)
            if HAS_SLIDING_WINDOW:
                mask = mask & (token_offsets >= window_start)

            k_base = phys_block * stride_cb + 0 * stride_c2 + kv_head_idx * stride_ch
            k_ptrs = kv_cache_ptr + k_base + t_offsets[:, None] * stride_ct + d_offsets[None, :] * stride_cd
            k = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

            scores0 = tl.sum(q0[None, :] * k, axis=1)
            scores1 = tl.sum(q1[None, :] * k, axis=1)
            scores2 = tl.sum(q2[None, :] * k, axis=1)
            scores3 = tl.sum(q3[None, :] * k, axis=1)
            scores0 = tl.where(mask & has_q0, scores0, -1e20)
            scores1 = tl.where(mask & has_q1, scores1, -1e20)
            scores2 = tl.where(mask & has_q2, scores2, -1e20)
            scores3 = tl.where(mask & has_q3, scores3, -1e20)

            m0_cur = tl.max(scores0)
            m1_cur = tl.max(scores1)
            m2_cur = tl.max(scores2)
            m3_cur = tl.max(scores3)
            m0_new = tl.maximum(m0, m0_cur)
            m1_new = tl.maximum(m1, m1_cur)
            m2_new = tl.maximum(m2, m2_cur)
            m3_new = tl.maximum(m3, m3_cur)
            alpha0 = tl.exp(m0 - m0_new)
            alpha1 = tl.exp(m1 - m1_new)
            alpha2 = tl.exp(m2 - m2_new)
            alpha3 = tl.exp(m3 - m3_new)
            exp0 = tl.exp(scores0 - m0_new)
            exp1 = tl.exp(scores1 - m1_new)
            exp2 = tl.exp(scores2 - m2_new)
            exp3 = tl.exp(scores3 - m3_new)
            exp0 = tl.where(mask & has_q0, exp0, 0.0)
            exp1 = tl.where(mask & has_q1, exp1, 0.0)
            exp2 = tl.where(mask & has_q2, exp2, 0.0)
            exp3 = tl.where(mask & has_q3, exp3, 0.0)
            l0_cur = tl.sum(exp0)
            l1_cur = tl.sum(exp1)
            l2_cur = tl.sum(exp2)
            l3_cur = tl.sum(exp3)

            v_base = phys_block * stride_cb + 1 * stride_c2 + kv_head_idx * stride_ch
            v_ptrs = kv_cache_ptr + v_base + t_offsets[:, None] * stride_ct + d_offsets[None, :] * stride_cd
            v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

            acc0 = acc0 * alpha0 + tl.sum(exp0[:, None] * v, axis=0)
            acc1 = acc1 * alpha1 + tl.sum(exp1[:, None] * v, axis=0)
            acc2 = acc2 * alpha2 + tl.sum(exp2[:, None] * v, axis=0)
            acc3 = acc3 * alpha3 + tl.sum(exp3[:, None] * v, axis=0)
            l0 = l0 * alpha0 + l0_cur
            l1 = l1 * alpha1 + l1_cur
            l2 = l2 * alpha2 + l2_cur
            l3 = l3 * alpha3 + l3_cur
            m0 = m0_new
            m1 = m1_new
            m2 = m2_new
            m3 = m3_new

        acc_base0 = seq_idx * stride_pas + q_head0 * stride_pah + split_idx * stride_pap
        stat_base0 = seq_idx * stride_pms + q_head0 * stride_pmh + split_idx * stride_pmp
        tl.store(partial_acc_ptr + acc_base0 + d_offsets * stride_pad, acc0, mask=has_q0)
        tl.store(partial_m_ptr + stat_base0, m0, mask=has_q0)
        tl.store(partial_l_ptr + stat_base0, l0, mask=has_q0)

        acc_base1 = seq_idx * stride_pas + q_head1 * stride_pah + split_idx * stride_pap
        stat_base1 = seq_idx * stride_pms + q_head1 * stride_pmh + split_idx * stride_pmp
        tl.store(partial_acc_ptr + acc_base1 + d_offsets * stride_pad, acc1, mask=has_q1)
        tl.store(partial_m_ptr + stat_base1, m1, mask=has_q1)
        tl.store(partial_l_ptr + stat_base1, l1, mask=has_q1)

        acc_base2 = seq_idx * stride_pas + q_head2 * stride_pah + split_idx * stride_pap
        stat_base2 = seq_idx * stride_pms + q_head2 * stride_pmh + split_idx * stride_pmp
        tl.store(partial_acc_ptr + acc_base2 + d_offsets * stride_pad, acc2, mask=has_q2)
        tl.store(partial_m_ptr + stat_base2, m2, mask=has_q2)
        tl.store(partial_l_ptr + stat_base2, l2, mask=has_q2)

        acc_base3 = seq_idx * stride_pas + q_head3 * stride_pah + split_idx * stride_pap
        stat_base3 = seq_idx * stride_pms + q_head3 * stride_pmh + split_idx * stride_pmp
        tl.store(partial_acc_ptr + acc_base3 + d_offsets * stride_pad, acc3, mask=has_q3)
        tl.store(partial_m_ptr + stat_base3, m3, mask=has_q3)
        tl.store(partial_l_ptr + stat_base3, l3, mask=has_q3)

    @triton.jit
    def _paged_attn_decode_rope_gqa8_split_kernel(
        partial_acc_ptr,
        partial_m_ptr,
        partial_l_ptr,
        query_ptr,
        kv_cache_ptr,
        block_tables_ptr,
        seq_lens_ptr,
        window_size,
        cos_ptr,
        sin_ptr,
        pos_ptr,
        norm_weight_ptr,
        scale,
        norm_eps,
        stride_pas, stride_pah, stride_pap, stride_pad,
        stride_pms, stride_pmh, stride_pmp,
        stride_qs, stride_qh,
        stride_cb, stride_c2, stride_ch, stride_ct, stride_cd,
        stride_bs,
        stride_cos_p, stride_cos_d,
        num_q_heads,
        num_kv_heads,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        ROTARY_HALF_DIM: tl.constexpr,
        HALF_ROTATE: tl.constexpr,
        HAS_QK_NORM: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
        HAS_SLIDING_WINDOW: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
    ):
        """GQA8 split decode with one K/V load shared by eight Q heads."""
        seq_idx = tl.program_id(0)
        q_group_idx = tl.program_id(1)
        split_idx = tl.program_id(2)

        gqa_ratio = num_q_heads // num_kv_heads
        group_offsets = tl.arange(0, GROUP_SIZE)
        q_heads = q_group_idx * GROUP_SIZE + group_offsets
        kv_head_idx = (q_group_idx * GROUP_SIZE) // gqa_ratio
        valid_heads = (q_heads < num_q_heads) & ((q_heads // gqa_ratio) == kv_head_idx)

        seq_len = tl.load(seq_lens_ptr + seq_idx)
        num_seq_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        window_start = 0
        first_block = 0
        num_window_blocks = num_seq_blocks
        if HAS_SLIDING_WINDOW:
            window_start = tl.maximum(seq_len - window_size, 0)
            first_block = window_start // BLOCK_SIZE
            num_window_blocks = num_seq_blocks - first_block
        split_start = split_idx * BLOCKS_PER_SPLIT

        d_offsets = tl.arange(0, HEAD_DIM)
        q_ptrs = (
            query_ptr
            + seq_idx * stride_qs
            + q_heads[:, None] * stride_qh
            + d_offsets[None, :]
        )
        q = tl.load(
            q_ptrs,
            mask=valid_heads[:, None],
            other=0.0,
        ).to(tl.float32)

        if HAS_QK_NORM:
            variance = tl.sum(q * q, axis=1) / HEAD_DIM
            rms = tl.rsqrt(variance + norm_eps)
            norm_w = tl.load(norm_weight_ptr + d_offsets).to(tl.float32)
            q = q * rms[:, None] * norm_w[None, :]

        pos = tl.load(pos_ptr + seq_idx)
        if HALF_ROTATE:
            in_rotary = d_offsets < ROTARY_DIM
            is_first = d_offsets < ROTARY_HALF_DIM
            partner = tl.where(is_first, d_offsets + ROTARY_HALF_DIM, d_offsets - ROTARY_HALF_DIM)
            cos_idx = tl.where(is_first, d_offsets, d_offsets - ROTARY_HALF_DIM)
        else:
            in_rotary = d_offsets < ROTARY_DIM
            is_first = (d_offsets % 2) == 0
            partner = tl.where(is_first, d_offsets + 1, d_offsets - 1)
            cos_idx = d_offsets // 2
        partner = tl.where(in_rotary, partner, d_offsets)
        cos_idx = tl.where(in_rotary, cos_idx, 0)

        q_partner_ptrs = (
            query_ptr
            + seq_idx * stride_qs
            + q_heads[:, None] * stride_qh
            + partner[None, :]
        )
        q_partner = tl.load(
            q_partner_ptrs,
            mask=valid_heads[:, None],
            other=0.0,
        ).to(tl.float32)
        if HAS_QK_NORM:
            norm_w_partner = tl.load(norm_weight_ptr + partner).to(tl.float32)
            q_partner = q_partner * rms[:, None] * norm_w_partner[None, :]

        cos_full = tl.load(
            cos_ptr + pos * stride_cos_p + cos_idx * stride_cos_d
        ).to(tl.float32)
        sin_full = tl.load(
            sin_ptr + pos * stride_cos_p + cos_idx * stride_cos_d
        ).to(tl.float32)
        q_rot = tl.where(
            is_first[None, :],
            q * cos_full[None, :] - q_partner * sin_full[None, :],
            q * cos_full[None, :] + q_partner * sin_full[None, :],
        )
        q = tl.where(in_rotary[None, :], q_rot, q) * scale

        m = tl.full([GROUP_SIZE], -1e20, dtype=tl.float32)
        l = tl.zeros([GROUP_SIZE], dtype=tl.float32)
        acc = tl.zeros([GROUP_SIZE, HEAD_DIM], dtype=tl.float32)
        t_offsets = tl.arange(0, BLOCK_SIZE)

        for local_block_idx in range(BLOCKS_PER_SPLIT):
            block_offset = split_start + local_block_idx
            block_idx = first_block + block_offset
            still_valid = block_offset < num_window_blocks
            phys_block = tl.load(
                block_tables_ptr + seq_idx * stride_bs + block_idx,
                mask=still_valid,
                other=0,
            )
            block_start = block_idx * BLOCK_SIZE
            token_offsets = block_start + t_offsets
            token_mask = still_valid & (token_offsets < seq_len)
            if HAS_SLIDING_WINDOW:
                token_mask = token_mask & (token_offsets >= window_start)

            k_base = phys_block * stride_cb + kv_head_idx * stride_ch
            k_ptrs = (
                kv_cache_ptr
                + k_base
                + t_offsets[:, None] * stride_ct
                + d_offsets[None, :] * stride_cd
            )
            k = tl.load(
                k_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)

            scores = tl.sum(q[:, None, :] * k[None, :, :], axis=2)
            score_mask = valid_heads[:, None] & token_mask[None, :]
            scores = tl.where(score_mask, scores, -1e20)
            m_cur = tl.max(scores, axis=1)
            m_new = tl.maximum(m, m_cur)
            alpha = tl.exp(m - m_new)
            exp_scores = tl.exp(scores - m_new[:, None])
            exp_scores = tl.where(score_mask, exp_scores, 0.0)
            l_cur = tl.sum(exp_scores, axis=1)

            v_base = phys_block * stride_cb + stride_c2 + kv_head_idx * stride_ch
            v_ptrs = (
                kv_cache_ptr
                + v_base
                + t_offsets[:, None] * stride_ct
                + d_offsets[None, :] * stride_cd
            )
            v = tl.load(
                v_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)

            acc = acc * alpha[:, None] + tl.sum(
                exp_scores[:, :, None] * v[None, :, :],
                axis=1,
            )
            l = l * alpha + l_cur
            m = m_new

        acc_ptrs = (
            partial_acc_ptr
            + seq_idx * stride_pas
            + q_heads[:, None] * stride_pah
            + split_idx * stride_pap
            + d_offsets[None, :] * stride_pad
        )
        stat_ptrs = (
            partial_m_ptr
            + seq_idx * stride_pms
            + q_heads * stride_pmh
            + split_idx * stride_pmp
        )
        tl.store(acc_ptrs, acc, mask=valid_heads[:, None])
        tl.store(stat_ptrs, m, mask=valid_heads)
        stat_l_ptrs = (
            partial_l_ptr
            + seq_idx * stride_pms
            + q_heads * stride_pmh
            + split_idx * stride_pmp
        )
        tl.store(stat_l_ptrs, l, mask=valid_heads)

    # =========================================================================
    # Packed prefill K/V scatter
    # =========================================================================

    @triton.jit
    def _paged_kv_cache_scatter_kernel(
        k_ptr,
        v_ptr,
        cache_ptr,
        phys_blocks_ptr,
        block_offsets_ptr,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_cb,
        stride_c2,
        stride_ch,
        stride_ct,
        stride_cd,
        KV_WIDTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        feature_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        feature_mask = feature_offsets < KV_WIDTH
        head_idx = feature_offsets // HEAD_DIM
        dim_idx = feature_offsets - head_idx * HEAD_DIM

        phys_block = tl.load(phys_blocks_ptr + token_idx).to(tl.int64)
        block_offset = tl.load(block_offsets_ptr + token_idx).to(tl.int64)
        k_src = (
            k_ptr
            + token_idx * stride_kt
            + head_idx * stride_kh
            + dim_idx * stride_kd
        )
        v_src = (
            v_ptr
            + token_idx * stride_vt
            + head_idx * stride_vh
            + dim_idx * stride_vd
        )
        cache_base = (
            cache_ptr
            + phys_block * stride_cb
            + head_idx * stride_ch
            + block_offset * stride_ct
            + dim_idx * stride_cd
        )
        k = tl.load(k_src, mask=feature_mask, other=0.0)
        v = tl.load(v_src, mask=feature_mask, other=0.0)
        tl.store(cache_base, k, mask=feature_mask)
        tl.store(cache_base + stride_c2, v, mask=feature_mask)

    @triton.jit
    def _paged_kv_cache_scatter_token_tiled_kernel(
        k_ptr,
        v_ptr,
        cache_ptr,
        phys_blocks_ptr,
        block_offsets_ptr,
        num_tokens,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_cb,
        stride_c2,
        stride_ch,
        stride_ct,
        stride_cd,
        KV_WIDTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        token_offsets = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        feature_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        token_mask = token_offsets < num_tokens
        feature_mask = feature_offsets < KV_WIDTH
        mask = token_mask[:, None] & feature_mask[None, :]

        head_idx = feature_offsets // HEAD_DIM
        dim_idx = feature_offsets - head_idx * HEAD_DIM
        phys_block = tl.load(
            phys_blocks_ptr + token_offsets,
            mask=token_mask,
            other=0,
        ).to(tl.int64)
        block_offset = tl.load(
            block_offsets_ptr + token_offsets,
            mask=token_mask,
            other=0,
        ).to(tl.int64)

        k_src = (
            k_ptr
            + token_offsets[:, None] * stride_kt
            + head_idx[None, :] * stride_kh
            + dim_idx[None, :] * stride_kd
        )
        v_src = (
            v_ptr
            + token_offsets[:, None] * stride_vt
            + head_idx[None, :] * stride_vh
            + dim_idx[None, :] * stride_vd
        )
        cache_base = (
            cache_ptr
            + phys_block[:, None] * stride_cb
            + head_idx[None, :] * stride_ch
            + block_offset[:, None] * stride_ct
            + dim_idx[None, :] * stride_cd
        )
        k = tl.load(k_src, mask=mask, other=0.0)
        v = tl.load(v_src, mask=mask, other=0.0)
        tl.store(cache_base, k, mask=mask)
        tl.store(cache_base + stride_c2, v, mask=mask)

    # =========================================================================
    # Fused RoPE + KV Cache Write
    # =========================================================================

    @triton.jit
    def _fused_rope_kv_write_kernel(
        kv_cache_ptr,
        k_ptr, v_ptr,            # [num_seqs, num_kv_heads, head_dim]
        cos_ptr, sin_ptr,        # [max_pos, half_dim]
        pos_ptr,                 # [num_seqs]
        phys_blocks_ptr,         # [num_seqs] physical block indices
        blk_offsets_ptr,         # [num_seqs] offset within block
        norm_weight_ptr,         # [head_dim] K-Norm weight, or nullptr
        stride_kv_s, stride_kv_h,
        stride_cb, stride_c2, stride_ch, stride_ct, stride_cd,
        stride_cos_p, stride_cos_d,
            norm_eps,
            num_kv_heads,
            ROTARY_DIM: tl.constexpr,
            ROTARY_HALF_DIM: tl.constexpr,
            HEAD_DIM: tl.constexpr,
        HALF_ROTATE: tl.constexpr,
        HAS_QK_NORM: tl.constexpr,
        HAS_V_NORM: tl.constexpr,
    ):
        """
        Fused: Optional QK-Norm + RoPE to K + write K,V to paged KV cache.
        Each program handles one (seq, kv_head) pair.
        Grid: (num_seqs, num_kv_heads)
        """
        seq_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        phys_block = tl.load(phys_blocks_ptr + seq_idx)
        blk_off = tl.load(blk_offsets_ptr + seq_idx)
        pos = tl.load(pos_ptr + seq_idx)

        d_offsets = tl.arange(0, HEAD_DIM)
        if HALF_ROTATE:
            in_rotary = d_offsets < ROTARY_DIM
            is_first = d_offsets < ROTARY_HALF_DIM
            partner = tl.where(is_first, d_offsets + ROTARY_HALF_DIM, d_offsets - ROTARY_HALF_DIM)
            cos_idx = tl.where(is_first, d_offsets, d_offsets - ROTARY_HALF_DIM)
        else:
            in_rotary = d_offsets < ROTARY_DIM
            is_first = (d_offsets % 2) == 0
            partner = tl.where(is_first, d_offsets + 1, d_offsets - 1)
            cos_idx = d_offsets // 2
        partner = tl.where(in_rotary, partner, d_offsets)
        cos_idx = tl.where(in_rotary, cos_idx, 0)

        # Load cos/sin for this position
        cos_vals = tl.load(cos_ptr + pos * stride_cos_p + cos_idx * stride_cos_d).to(tl.float32)
        sin_vals = tl.load(sin_ptr + pos * stride_cos_p + cos_idx * stride_cos_d).to(tl.float32)

        # Load raw K, optionally apply QK-Norm, then apply RoPE
        k_base = k_ptr + seq_idx * stride_kv_s + head_idx * stride_kv_h
        cache_k_base = (kv_cache_ptr + phys_block * stride_cb + 0 * stride_c2
                        + head_idx * stride_ch + blk_off * stride_ct)

        # Compute RMSNorm scale once on full K vector (if QK-Norm)
        if HAS_QK_NORM:
            k_full = tl.load(k_base + d_offsets).to(tl.float32)
            rms_scale = tl.rsqrt(tl.sum(k_full * k_full) / HEAD_DIM + norm_eps)

        if HAS_QK_NORM:
            nw = tl.load(norm_weight_ptr + d_offsets).to(tl.float32)
            nw_partner = tl.load(norm_weight_ptr + partner).to(tl.float32)
            k_val = tl.load(k_base + d_offsets).to(tl.float32) * rms_scale * nw
            k_partner = tl.load(k_base + partner).to(tl.float32) * rms_scale * nw_partner
        else:
            k_val = tl.load(k_base + d_offsets).to(tl.float32)
            k_partner = tl.load(k_base + partner).to(tl.float32)
        k_rot = tl.where(
            is_first,
            k_val * cos_vals - k_partner * sin_vals,
            k_val * cos_vals + k_partner * sin_vals,
        )
        k_out = tl.where(in_rotary, k_rot, k_val)
        tl.store(cache_k_base + d_offsets * stride_cd, k_out.to(kv_cache_ptr.dtype.element_ty))

        # Load V, optionally apply RMSNorm (without scale), and write directly.
        v_base = v_ptr + seq_idx * stride_kv_s + head_idx * stride_kv_h
        if HAS_V_NORM:
            v_raw = tl.load(v_base + d_offsets).to(tl.float32)
            v_rms_scale = tl.rsqrt(tl.sum(v_raw * v_raw) / HEAD_DIM + norm_eps)
            v_raw = v_raw * v_rms_scale
        else:
            v_raw = tl.load(v_base + d_offsets)
        cache_v_ptr = (kv_cache_ptr + phys_block * stride_cb + 1 * stride_c2
                       + head_idx * stride_ch + blk_off * stride_ct + d_offsets * stride_cd)
        tl.store(cache_v_ptr, v_raw)


def paged_kv_cache_scatter(
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    phys_blocks: torch.Tensor,
    block_offsets: torch.Tensor,
) -> bool:
    """Copy packed prefill K/V directly into paged cache in one Triton launch."""
    if not _HAS_TRITON:
        return False
    tensors = (k, v, kv_cache, phys_blocks, block_offsets)
    if not all(tensor.is_cuda for tensor in tensors):
        return False
    if k.ndim != 3 or v.ndim != 3 or kv_cache.ndim != 5:
        raise ValueError("paged K/V scatter expects K/V [T,H,D] and cache [B,2,H,S,D]")
    if tuple(k.shape) != tuple(v.shape):
        raise ValueError("paged K/V scatter requires identical K and V shapes")
    if k.dtype != v.dtype or k.dtype != kv_cache.dtype:
        raise ValueError("paged K/V scatter requires matching K, V, and cache dtypes")
    if any(tensor.device != k.device for tensor in tensors[1:]):
        raise ValueError("paged K/V scatter tensors must share one CUDA device")

    num_tokens, num_kv_heads, head_dim = map(int, k.shape)
    if int(kv_cache.shape[1]) != 2:
        raise ValueError("paged K/V cache must contain K and V planes")
    if (
        int(kv_cache.shape[2]) != num_kv_heads
        or int(kv_cache.shape[4]) != head_dim
    ):
        raise ValueError("packed K/V shape does not match the paged cache layout")
    if int(phys_blocks.numel()) != num_tokens or int(block_offsets.numel()) != num_tokens:
        raise ValueError("paged K/V mapping must contain one entry per token")
    if num_tokens == 0:
        return True

    block_n = min(256, triton.next_power_of_2(head_dim))
    kv_width = num_kv_heads * head_dim
    grid = (num_tokens, triton.cdiv(kv_width, block_n))
    _paged_kv_cache_scatter_kernel[grid](
        k,
        v,
        kv_cache,
        phys_blocks,
        block_offsets,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        kv_cache.stride(4),
        KV_WIDTH=kv_width,
        HEAD_DIM=head_dim,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=1,
    )
    return True


def paged_kv_cache_scatter_token_tiled(
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    phys_blocks: torch.Tensor,
    block_offsets: torch.Tensor,
    *,
    tokens_per_program: int,
) -> bool:
    """Experimental packed K/V scatter with multiple tokens per program."""
    if not _HAS_TRITON:
        return False
    tensors = (k, v, kv_cache, phys_blocks, block_offsets)
    if not all(tensor.is_cuda for tensor in tensors):
        return False
    if k.ndim != 3 or v.ndim != 3 or kv_cache.ndim != 5:
        raise ValueError("paged K/V scatter expects K/V [T,H,D] and cache [B,2,H,S,D]")
    if tuple(k.shape) != tuple(v.shape):
        raise ValueError("paged K/V scatter requires identical K and V shapes")
    if k.dtype != v.dtype or k.dtype != kv_cache.dtype:
        raise ValueError("paged K/V scatter requires matching K, V, and cache dtypes")
    if any(tensor.device != k.device for tensor in tensors[1:]):
        raise ValueError("paged K/V scatter tensors must share one CUDA device")

    num_tokens, num_kv_heads, head_dim = map(int, k.shape)
    if int(kv_cache.shape[1]) != 2:
        raise ValueError("paged K/V cache must contain K and V planes")
    if (
        int(kv_cache.shape[2]) != num_kv_heads
        or int(kv_cache.shape[4]) != head_dim
    ):
        raise ValueError("packed K/V shape does not match the paged cache layout")
    if int(phys_blocks.numel()) != num_tokens or int(block_offsets.numel()) != num_tokens:
        raise ValueError("paged K/V mapping must contain one entry per token")
    if num_tokens == 0:
        return True

    block_t = int(tokens_per_program)
    if block_t not in (2, 4, 8):
        raise ValueError("tokens_per_program must be one of 2, 4, or 8")
    block_n = min(256, triton.next_power_of_2(head_dim))
    kv_width = num_kv_heads * head_dim
    grid = (triton.cdiv(num_tokens, block_t), triton.cdiv(kv_width, block_n))
    _paged_kv_cache_scatter_token_tiled_kernel[grid](
        k,
        v,
        kv_cache,
        phys_blocks,
        block_offsets,
        num_tokens,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        kv_cache.stride(4),
        KV_WIDTH=kv_width,
        HEAD_DIM=head_dim,
        BLOCK_T=block_t,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=1,
    )
    return True


def _triton_paged_decode(
    query,
    kv_cache,
    block_tables,
    seq_lens,
    scale,
    out=None,
    sliding_window: Optional[int] = None,
    max_blocks_override: Optional[int] = None,
    split_policy_override: Optional[int] = None,
):
    """Triton implementation of paged attention decode."""
    num_seqs, num_q_heads, head_dim = query.shape
    num_kv_heads = kv_cache.shape[2]
    block_size = kv_cache.shape[3]
    table_max_blocks = block_tables.shape[1]
    max_blocks = _resolve_decode_max_blocks(table_max_blocks, max_blocks_override)
    has_sliding = sliding_window is not None and sliding_window > 0
    loop_max_blocks = _sliding_loop_max_blocks(block_size, max_blocks, sliding_window)
    num_warps = _decode_num_warps(head_dim, query.device)
    num_splits = _get_decode_split_count(
        num_seqs,
        num_q_heads,
        loop_max_blocks,
        num_warps=num_warps,
        device=query.device,
        policy_override=split_policy_override,
    )
    num_warps = _decode_num_warps(head_dim, query.device, num_splits=num_splits)
    reduce_warps = _decode_reduce_num_warps(head_dim, num_splits)
    window_size = int(sliding_window or 0)
    _log_decode_shape_once(
        fused=False,
        num_seqs=num_seqs,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        table_max_blocks=table_max_blocks,
        loop_max_blocks=loop_max_blocks,
        num_splits=num_splits,
        num_warps=num_warps,
        reduce_warps=reduce_warps,
    )

    output = _prepare_decode_output(query, out)

    if num_splits <= 1:
        grid = (num_seqs, num_q_heads)
        _paged_attn_decode_kernel[grid](
            output, query, kv_cache, block_tables, seq_lens, scale, window_size,
            output.stride(0), output.stride(1),
            query.stride(0), query.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            kv_cache.stride(3), kv_cache.stride(4),
            block_tables.stride(0),
            num_q_heads, num_kv_heads, loop_max_blocks,
            BLOCK_SIZE=block_size, HEAD_DIM=head_dim,
            HAS_SLIDING_WINDOW=1 if has_sliding else 0,
            num_warps=num_warps,
        )
        return output

    blocks_per_split = (loop_max_blocks + num_splits - 1) // num_splits
    partial_acc, partial_m, partial_l = _get_decode_workspace(
        query, num_seqs, num_q_heads, num_splits, head_dim,
    )

    split_grid = (num_seqs, num_q_heads, num_splits)
    _paged_attn_decode_split_kernel[split_grid](
        partial_acc, partial_m, partial_l,
        query, kv_cache, block_tables, seq_lens, scale, window_size,
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        query.stride(0), query.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        kv_cache.stride(3), kv_cache.stride(4),
        block_tables.stride(0),
        num_q_heads, num_kv_heads,
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim,
        BLOCKS_PER_SPLIT=blocks_per_split,
        HAS_SLIDING_WINDOW=1 if has_sliding else 0,
        num_warps=num_warps,
    )

    reduce_grid = (num_seqs, num_q_heads)
    _paged_attn_decode_split_reduce_kernel[reduce_grid](
        output,
        partial_acc, partial_m, partial_l,
        output.stride(0), output.stride(1),
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        HEAD_DIM=head_dim,
        NUM_SPLITS=num_splits,
        num_warps=reduce_warps,
    )
    return output


def _triton_paged_decode_grouped_segmented(
    query,
    kv_cache,
    block_tables,
    seq_lens,
    scale,
    out=None,
    sliding_window: Optional[int] = None,
    max_blocks_override: Optional[int] = None,
    *,
    force: bool = False,
    num_segments_override: Optional[int] = None,
    tile_size_override: Optional[int] = None,
):
    """Gemma4 B16 tensor-core GQA decode with tuned parallel KV segments."""
    topology = _grouped_segmented_decode_topology(
        query,
        kv_cache,
        block_tables,
        sliding_window=sliding_window,
        force=force,
    )
    if topology is None:
        raise RuntimeError(
            "grouped segmented decode is not eligible for this shape"
        )

    num_seqs, num_q_heads, head_dim = query.shape
    num_kv_heads = int(kv_cache.shape[2])
    block_size = int(kv_cache.shape[3])
    table_max_blocks = int(block_tables.shape[1])
    max_blocks = _resolve_decode_max_blocks(
        table_max_blocks,
        max_blocks_override,
    )
    has_sliding = sliding_window is not None and sliding_window > 0
    window_size = int(sliding_window or 0)
    block_m = 16
    max_visible_tokens = max_blocks * block_size
    if has_sliding:
        max_visible_tokens = min(max_visible_tokens, window_size)
    tile_size = (
        _grouped_segmented_decode_tile_size(
            topology,
            max_visible_tokens,
        )
        if tile_size_override is None
        else int(tile_size_override)
    )
    if tile_size not in (16, 32, 64):
        raise ValueError(
            "grouped segmented decode requires tile size 16, 32, or 64 "
            f"(got {tile_size})"
        )
    gqa_ratio = num_q_heads // num_kv_heads
    num_segments = (
        _grouped_segmented_decode_num_segments(
            topology,
            max_visible_tokens,
        )
        if num_segments_override is None
        else int(num_segments_override)
    )
    if num_segments not in (4, 8, 16, 32):
        raise ValueError(
            "grouped segmented decode requires 4, 8, 16, or 32 segments "
            f"(got {num_segments})"
        )
    max_tiles_per_segment = max(
        1,
        math.ceil(
            max_visible_tokens / (num_segments * tile_size)
        ),
    )
    num_warps = max(
        1,
        min(
            _env_int(
                "MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_WARPS",
                4,
            ),
            8,
        ),
    )
    num_stages = max(
        1,
        min(
            _env_int(
                "MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_STAGES",
                3,
            ),
            5,
        ),
    )
    reduce_warps = max(
        1,
        min(
            _env_int(
                "MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_REDUCE_WARPS",
                4,
            ),
            8,
        ),
    )

    output = _prepare_decode_output(query, out)
    partial_acc, partial_m, partial_l = _get_decode_workspace(
        query,
        num_seqs,
        num_q_heads,
        num_segments,
        head_dim,
    )
    partial_acc_strides = tuple(partial_acc.stride())
    partial_stat_strides = tuple(partial_m.stride())
    segment_grid = (num_seqs, num_kv_heads, num_segments)
    _paged_attn_decode_grouped_segment_kernel[segment_grid](
        partial_acc,
        partial_m,
        partial_l,
        query,
        kv_cache,
        block_tables,
        seq_lens,
        scale,
        window_size,
        partial_acc_strides[0],
        partial_acc_strides[1],
        partial_acc_strides[2],
        partial_acc_strides[3],
        partial_stat_strides[0],
        partial_stat_strides[1],
        partial_stat_strides[2],
        query.stride(0),
        query.stride(1),
        query.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        kv_cache.stride(4),
        block_tables.stride(0),
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        GQA_RATIO=gqa_ratio,
        TILE_SIZE=tile_size,
        NUM_SEGMENTS=num_segments,
        MAX_TILES_PER_SEGMENT=max_tiles_per_segment,
        HAS_SLIDING_WINDOW=1 if has_sliding else 0,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    reduce_grid = (num_seqs, num_q_heads)
    _paged_attn_decode_grouped_segment_reduce_kernel[reduce_grid](
        output,
        partial_acc,
        partial_m,
        partial_l,
        seq_lens,
        window_size,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        partial_acc_strides[0],
        partial_acc_strides[1],
        partial_acc_strides[2],
        partial_acc_strides[3],
        partial_stat_strides[0],
        partial_stat_strides[1],
        partial_stat_strides[2],
        HEAD_DIM=head_dim,
        TILE_SIZE=tile_size,
        NUM_SEGMENTS=num_segments,
        HAS_SLIDING_WINDOW=1 if has_sliding else 0,
        num_warps=reduce_warps,
    )

    global _GROUPED_SEGMENTED_DECODE_HITS
    global _GROUPED_SEGMENTED_DECODE_LOGGED
    global _GROUPED_SEGMENTED_DECODE_SELECTED_SEGMENTS
    global _GROUPED_SEGMENTED_DECODE_SELECTED_TILE_SIZES
    _GROUPED_SEGMENTED_DECODE_HITS += 1
    _GROUPED_SEGMENTED_DECODE_SELECTED_SEGMENTS[topology] = int(
        num_segments
    )
    _GROUPED_SEGMENTED_DECODE_SELECTED_TILE_SIZES[topology] = int(tile_size)
    if (
        _env_bool("MEGAGEMM_PAGED_DECODE_LOG", False)
        and not _GROUPED_SEGMENTED_DECODE_LOGGED
    ):
        _GROUPED_SEGMENTED_DECODE_LOGGED = True
        print(
            "[MegaGemm] Gemma4 grouped segmented attention enabled "
            f"topology={topology} segments={num_segments} "
            f"tile={tile_size} warps={num_warps} stages={num_stages}"
        )
    return output


def _triton_paged_decode_grouped_segmented_fused(
    query,
    kv_cache,
    block_tables,
    seq_lens,
    scale,
    cos,
    sin,
    positions,
    half_rotate=False,
    rotary_dim=None,
    q_norm_weight=None,
    norm_eps=1e-6,
    out=None,
    sliding_window: Optional[int] = None,
    max_blocks_override: Optional[int] = None,
    *,
    force: bool = False,
    num_segments_override: Optional[int] = None,
    tile_size_override: Optional[int] = None,
):
    """QNorm/RoPE prepare plus the grouped segmented Gemma4 decode core."""
    topology = _grouped_segmented_decode_topology(
        query,
        kv_cache,
        block_tables,
        sliding_window=sliding_window,
        force=force,
    )
    if topology is None:
        raise RuntimeError(
            "grouped segmented fused decode is not eligible for this shape"
        )

    num_seqs, num_q_heads, head_dim = query.shape
    rotary_dim = head_dim if rotary_dim is None else int(rotary_dim)
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(
            "grouped segmented fused decode requires an even rotary_dim "
            f"inside the head (got {rotary_dim} vs {head_dim})"
        )
    rotary_half_dim = rotary_dim // 2
    has_qk_norm = q_norm_weight is not None
    norm_w_ptr = q_norm_weight if has_qk_norm else query
    output = _prepare_decode_output(query, out)
    prepare_warps = max(
        1,
        min(
            _env_int(
                "MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_Q_PREP_WARPS",
                4,
            ),
            8,
        ),
    )

    prepare_grid = (num_seqs, num_q_heads)
    _paged_attn_decode_qnorm_rope_prepare_kernel[prepare_grid](
        output,
        query,
        cos,
        sin,
        positions,
        norm_w_ptr,
        norm_eps,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        cos.stride(0),
        cos.stride(1),
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        ROTARY_HALF_DIM=rotary_half_dim,
        HALF_ROTATE=1 if half_rotate else 0,
        HAS_QK_NORM=1 if has_qk_norm else 0,
        num_warps=prepare_warps,
    )
    return _triton_paged_decode_grouped_segmented(
        output,
        kv_cache,
        block_tables,
        seq_lens,
        scale,
        out=output,
        sliding_window=sliding_window,
        max_blocks_override=max_blocks_override,
        force=force,
        num_segments_override=num_segments_override,
        tile_size_override=tile_size_override,
    )


def _triton_paged_decode_fused(query, kv_cache, block_tables, seq_lens, scale,
                                cos, sin, positions, half_rotate=False,
                                rotary_dim=None, q_norm_weight=None, norm_eps=1e-6,
                                out=None, sliding_window: Optional[int] = None,
                                max_blocks_override: Optional[int] = None,
                                split_policy_override: Optional[int] = None,
                                gqa2_direct_policy_enabled: Optional[bool] = None,
                                num_warps_policy_override: Optional[int] = None,
                                e2b_l4_h512_policy_enabled: Optional[bool] = None):
    """Triton fused QK-Norm + RoPE + paged attention decode."""
    num_seqs, num_q_heads, head_dim = query.shape
    num_kv_heads = kv_cache.shape[2]
    block_size = kv_cache.shape[3]
    table_max_blocks = block_tables.shape[1]
    max_blocks = _resolve_decode_max_blocks(table_max_blocks, max_blocks_override)
    has_sliding = sliding_window is not None and sliding_window > 0
    loop_max_blocks = _sliding_loop_max_blocks(block_size, max_blocks, sliding_window)
    rotary_dim = head_dim if rotary_dim is None else int(rotary_dim)
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(
            f"fused decode requires 0 < even rotary_dim <= head_dim (got {rotary_dim} vs {head_dim})"
        )
    rotary_half_dim = rotary_dim // 2
    has_qk_norm = q_norm_weight is not None

    global _GQA2_DECODE_DISABLED, _GQA2_DECODE_LOGGED
    global _GQA2_DIRECT_DECODE_HITS, _GENERIC_DIRECT_DECODE_HITS
    global _GQA4_DECODE_DISABLED, _GQA4_DECODE_LOGGED
    global _GQA8_DECODE_DISABLED, _GQA8_DECODE_LOGGED
    global _GROUPED_SEGMENTED_DECODE_DISABLED
    global _GROUPED_SEGMENTED_DECODE_LOGGED
    global _GROUPED_SEGMENTED_DECODE_FAILURE
    output = _prepare_decode_output(query, out)

    candidate_topology = _grouped_segmented_decode_topology(
        query,
        kv_cache,
        block_tables,
        sliding_window=sliding_window,
        e2b_l4_h512_policy_enabled=e2b_l4_h512_policy_enabled,
    )
    if (
        candidate_topology is not None
        and half_rotate
        and rotary_dim == head_dim
        and has_qk_norm
    ):
        try:
            return _triton_paged_decode_grouped_segmented_fused(
                query,
                kv_cache,
                block_tables,
                seq_lens,
                scale,
                cos,
                sin,
                positions,
                half_rotate=half_rotate,
                rotary_dim=rotary_dim,
                q_norm_weight=q_norm_weight,
                norm_eps=norm_eps,
                out=output,
                sliding_window=sliding_window,
                max_blocks_override=max_blocks_override,
                force=True,
            )
        except Exception as exc:
            _GROUPED_SEGMENTED_DECODE_DISABLED = True
            _GROUPED_SEGMENTED_DECODE_FAILURE = (
                f"{type(exc).__name__}: {exc}"
            )
            if (
                _env_bool("MEGAGEMM_PAGED_DECODE_LOG", False)
                and not _GROUPED_SEGMENTED_DECODE_LOGGED
            ):
                _GROUPED_SEGMENTED_DECODE_LOGGED = True
                print(
                    "[MegaGemm] Gemma4 grouped segmented attention "
                    "disabled; falling back to the scalar decode core "
                    f"({_GROUPED_SEGMENTED_DECODE_FAILURE})"
                )

    num_warps = _decode_num_warps(
        head_dim,
        query.device,
        policy_override=num_warps_policy_override,
    )
    planned_q_heads = _planned_fused_decode_program_heads(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    num_splits = _get_decode_split_count(
        num_seqs,
        planned_q_heads,
        loop_max_blocks,
        num_warps=num_warps,
        device=query.device,
        policy_override=split_policy_override,
    )
    num_warps = _decode_num_warps(
        head_dim,
        query.device,
        num_splits=num_splits,
        policy_override=num_warps_policy_override,
    )
    reduce_warps = _decode_reduce_num_warps(head_dim, num_splits)
    window_size = int(sliding_window or 0)
    _log_decode_shape_once(
        fused=True,
        num_seqs=num_seqs,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        table_max_blocks=table_max_blocks,
        loop_max_blocks=loop_max_blocks,
        num_splits=num_splits,
        num_warps=num_warps,
        reduce_warps=reduce_warps,
    )

    # Dummy pointer when no QK-Norm (Triton needs a valid ptr)
    norm_w_ptr = q_norm_weight if has_qk_norm else query

    block_unroll = _decode_block_unroll(
        block_size=block_size,
        head_dim=head_dim,
        num_splits=num_splits,
        device=query.device,
    )

    if num_splits <= 1:
        if _use_gqa4_direct_decode(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_splits=num_splits,
        ):
            try:
                grid_gqa4 = (num_seqs, triton.cdiv(num_q_heads, 4))
                _paged_attn_decode_rope_gqa4_kernel[grid_gqa4](
                    output, query, kv_cache, block_tables, seq_lens, window_size,
                    cos, sin, positions,
                    norm_w_ptr, scale, norm_eps,
                    output.stride(0), output.stride(1),
                    query.stride(0), query.stride(1),
                    kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                    kv_cache.stride(3), kv_cache.stride(4),
                    block_tables.stride(0),
                    cos.stride(0), cos.stride(1),
                    num_q_heads, num_kv_heads, loop_max_blocks,
                    BLOCK_SIZE=block_size, HEAD_DIM=head_dim,
                    ROTARY_DIM=rotary_dim,
                    ROTARY_HALF_DIM=rotary_half_dim,
                    HALF_ROTATE=1 if half_rotate else 0,
                    HAS_QK_NORM=1 if has_qk_norm else 0,
                    HAS_SLIDING_WINDOW=1 if has_sliding else 0,
                    num_warps=num_warps,
                )
            except Exception:
                _GQA4_DECODE_DISABLED = True
            else:
                if _env_bool("MEGAGEMM_PAGED_DECODE_LOG", False) and not _GQA4_DECODE_LOGGED:
                    _GQA4_DECODE_LOGGED = True
                    print("[MegaGemm] paged decode GQA4 direct path enabled")
                return output

        if _use_gqa2_direct_decode(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_splits=num_splits,
            policy_enabled=gqa2_direct_policy_enabled,
        ):
            try:
                grid_gqa2 = (num_seqs, triton.cdiv(num_q_heads, 2))
                _paged_attn_decode_rope_gqa2_kernel[grid_gqa2](
                    output, query, kv_cache, block_tables, seq_lens, window_size,
                    cos, sin, positions,
                    norm_w_ptr, scale, norm_eps,
                    output.stride(0), output.stride(1),
                    query.stride(0), query.stride(1),
                    kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                    kv_cache.stride(3), kv_cache.stride(4),
                    block_tables.stride(0),
                    cos.stride(0), cos.stride(1),
                    num_q_heads, num_kv_heads, loop_max_blocks,
                    BLOCK_SIZE=block_size, HEAD_DIM=head_dim,
                    ROTARY_DIM=rotary_dim,
                    ROTARY_HALF_DIM=rotary_half_dim,
                    HALF_ROTATE=1 if half_rotate else 0,
                    HAS_QK_NORM=1 if has_qk_norm else 0,
                    HAS_SLIDING_WINDOW=1 if has_sliding else 0,
                    num_warps=num_warps,
                )
            except Exception:
                _GQA2_DECODE_DISABLED = True
            else:
                _GQA2_DIRECT_DECODE_HITS += 1
                if _env_bool("MEGAGEMM_PAGED_DECODE_LOG", False) and not _GQA2_DECODE_LOGGED:
                    _GQA2_DECODE_LOGGED = True
                    print("[MegaGemm] paged decode GQA2 direct path enabled")
                return output

        grid = (num_seqs, num_q_heads)
        try:
            _paged_attn_decode_rope_kernel[grid](
                output, query, kv_cache, block_tables, seq_lens, window_size,
                cos, sin, positions,
                norm_w_ptr, scale, norm_eps,
                output.stride(0), output.stride(1),
                query.stride(0), query.stride(1),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                kv_cache.stride(3), kv_cache.stride(4),
                block_tables.stride(0),
                cos.stride(0), cos.stride(1),
                num_q_heads, num_kv_heads, loop_max_blocks,
                BLOCK_SIZE=block_size, HEAD_DIM=head_dim,
                ROTARY_DIM=rotary_dim,
                ROTARY_HALF_DIM=rotary_half_dim,
                HALF_ROTATE=1 if half_rotate else 0,
                HAS_QK_NORM=1 if has_qk_norm else 0,
                HAS_SLIDING_WINDOW=1 if has_sliding else 0,
                BLOCK_UNROLL=block_unroll,
                num_warps=num_warps,
            )
        except Exception:
            if block_unroll <= 1:
                raise
            _paged_attn_decode_rope_kernel[grid](
                output, query, kv_cache, block_tables, seq_lens, window_size,
                cos, sin, positions,
                norm_w_ptr, scale, norm_eps,
                output.stride(0), output.stride(1),
                query.stride(0), query.stride(1),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                kv_cache.stride(3), kv_cache.stride(4),
                block_tables.stride(0),
                cos.stride(0), cos.stride(1),
                num_q_heads, num_kv_heads, loop_max_blocks,
                BLOCK_SIZE=block_size, HEAD_DIM=head_dim,
                ROTARY_DIM=rotary_dim,
                ROTARY_HALF_DIM=rotary_half_dim,
                HALF_ROTATE=1 if half_rotate else 0,
                HAS_QK_NORM=1 if has_qk_norm else 0,
                HAS_SLIDING_WINDOW=1 if has_sliding else 0,
                BLOCK_UNROLL=1,
                num_warps=num_warps,
            )
        _GENERIC_DIRECT_DECODE_HITS += 1
        return output

    blocks_per_split = (loop_max_blocks + num_splits - 1) // num_splits
    partial_acc, partial_m, partial_l = _get_decode_workspace(
        query, num_seqs, num_q_heads, num_splits, head_dim,
    )

    require_gqa8 = _env_bool("MEGAGEMM_PAGED_DECODE_REQUIRE_GQA8", False)
    if _use_gqa8_decode(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_splits=num_splits,
    ):
        try:
            split_grid_gqa8 = (num_seqs, triton.cdiv(num_q_heads, 8), num_splits)
            _paged_attn_decode_rope_gqa8_split_kernel[split_grid_gqa8](
                partial_acc, partial_m, partial_l,
                query, kv_cache, block_tables, seq_lens, window_size,
                cos, sin, positions,
                norm_w_ptr, scale, norm_eps,
                partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
                partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
                query.stride(0), query.stride(1),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                kv_cache.stride(3), kv_cache.stride(4),
                block_tables.stride(0),
                cos.stride(0), cos.stride(1),
                num_q_heads, num_kv_heads,
                BLOCK_SIZE=block_size, HEAD_DIM=head_dim,
                ROTARY_DIM=rotary_dim,
                ROTARY_HALF_DIM=rotary_half_dim,
                HALF_ROTATE=1 if half_rotate else 0,
                HAS_QK_NORM=1 if has_qk_norm else 0,
                BLOCKS_PER_SPLIT=blocks_per_split,
                HAS_SLIDING_WINDOW=1 if has_sliding else 0,
                GROUP_SIZE=8,
                num_warps=num_warps,
            )
        except Exception as exc:
            _GQA8_DECODE_DISABLED = True
            if require_gqa8:
                raise RuntimeError(
                    "MEGAGEMM_PAGED_DECODE_REQUIRE_GQA8=1 but the GQA8 split "
                    "decode kernel failed "
                    f"(q_heads={num_q_heads}, kv_heads={num_kv_heads}, "
                    f"head_dim={head_dim}, splits={num_splits})"
                ) from exc
            if _env_bool("MEGAGEMM_PAGED_DECODE_LOG", False) and not _GQA8_DECODE_LOGGED:
                _GQA8_DECODE_LOGGED = True
                print(
                    "[MegaGemm] paged decode GQA8 path disabled; "
                    f"falling back to GQA4 ({type(exc).__name__}: {exc})"
                )
        else:
            if _env_bool("MEGAGEMM_PAGED_DECODE_LOG", False) and not _GQA8_DECODE_LOGGED:
                _GQA8_DECODE_LOGGED = True
                print("[MegaGemm] paged decode GQA8 path enabled")
            reduce_grid = (num_seqs, num_q_heads)
            _paged_attn_decode_split_reduce_kernel[reduce_grid](
                output,
                partial_acc, partial_m, partial_l,
                output.stride(0), output.stride(1),
                partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
                partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
                HEAD_DIM=head_dim,
                NUM_SPLITS=num_splits,
                num_warps=reduce_warps,
            )
            return output

    if require_gqa8:
        raise RuntimeError(
            "MEGAGEMM_PAGED_DECODE_REQUIRE_GQA8=1 but GQA8 split decode was "
            "not selected "
            f"(enabled={_env_bool('MEGAGEMM_PAGED_DECODE_GQA8_SPLIT', False)}, "
            f"disabled={_GQA8_DECODE_DISABLED}, q_heads={num_q_heads}, "
            f"kv_heads={num_kv_heads}, head_dim={head_dim}, splits={num_splits})"
        )

    if _use_gqa4_decode(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_splits=num_splits,
    ):
        try:
            split_grid_gqa4 = (num_seqs, triton.cdiv(num_q_heads, 4), num_splits)
            _paged_attn_decode_rope_gqa4_split_kernel[split_grid_gqa4](
                partial_acc, partial_m, partial_l,
                query, kv_cache, block_tables, seq_lens, window_size,
                cos, sin, positions,
                norm_w_ptr, scale, norm_eps,
                partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
                partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
                query.stride(0), query.stride(1),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                kv_cache.stride(3), kv_cache.stride(4),
                block_tables.stride(0),
                cos.stride(0), cos.stride(1),
                num_q_heads, num_kv_heads,
                BLOCK_SIZE=block_size, HEAD_DIM=head_dim,
                ROTARY_DIM=rotary_dim,
                ROTARY_HALF_DIM=rotary_half_dim,
                HALF_ROTATE=1 if half_rotate else 0,
                HAS_QK_NORM=1 if has_qk_norm else 0,
                BLOCKS_PER_SPLIT=blocks_per_split,
                HAS_SLIDING_WINDOW=1 if has_sliding else 0,
                num_warps=num_warps,
            )
        except Exception as exc:
            _GQA4_DECODE_DISABLED = True
            if _env_bool("MEGAGEMM_PAGED_DECODE_LOG", False) and not _GQA4_DECODE_LOGGED:
                _GQA4_DECODE_LOGGED = True
                print(f"[MegaGemm] paged decode GQA4 path disabled; falling back to GQA2 ({type(exc).__name__}: {exc})")
        else:
            if _env_bool("MEGAGEMM_PAGED_DECODE_LOG", False) and not _GQA4_DECODE_LOGGED:
                _GQA4_DECODE_LOGGED = True
                print("[MegaGemm] paged decode GQA4 path enabled")
            reduce_grid = (num_seqs, num_q_heads)
            _paged_attn_decode_split_reduce_kernel[reduce_grid](
                output,
                partial_acc, partial_m, partial_l,
                output.stride(0), output.stride(1),
                partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
                partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
                HEAD_DIM=head_dim,
                NUM_SPLITS=num_splits,
                num_warps=reduce_warps,
            )
            return output

    if _use_gqa2_decode(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_splits=num_splits,
    ):
        try:
            split_grid_gqa2 = (num_seqs, triton.cdiv(num_q_heads, 2), num_splits)
            _paged_attn_decode_rope_gqa2_split_kernel[split_grid_gqa2](
                partial_acc, partial_m, partial_l,
                query, kv_cache, block_tables, seq_lens, window_size,
                cos, sin, positions,
                norm_w_ptr, scale, norm_eps,
                partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
                partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
                query.stride(0), query.stride(1),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                kv_cache.stride(3), kv_cache.stride(4),
                block_tables.stride(0),
                cos.stride(0), cos.stride(1),
                num_q_heads, num_kv_heads,
                BLOCK_SIZE=block_size, HEAD_DIM=head_dim,
                ROTARY_DIM=rotary_dim,
                ROTARY_HALF_DIM=rotary_half_dim,
                HALF_ROTATE=1 if half_rotate else 0,
                HAS_QK_NORM=1 if has_qk_norm else 0,
                BLOCKS_PER_SPLIT=blocks_per_split,
                HAS_SLIDING_WINDOW=1 if has_sliding else 0,
                num_warps=num_warps,
            )
        except Exception:
            _GQA2_DECODE_DISABLED = True
        else:
            if _env_bool("MEGAGEMM_PAGED_DECODE_LOG", False) and not _GQA2_DECODE_LOGGED:
                _GQA2_DECODE_LOGGED = True
                print("[MegaGemm] paged decode GQA2 path enabled")
            reduce_grid = (num_seqs, num_q_heads)
            _paged_attn_decode_split_reduce_kernel[reduce_grid](
                output,
                partial_acc, partial_m, partial_l,
                output.stride(0), output.stride(1),
                partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
                partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
                HEAD_DIM=head_dim,
                NUM_SPLITS=num_splits,
                num_warps=reduce_warps,
            )
            return output

    split_grid = (num_seqs, num_q_heads, num_splits)
    _paged_attn_decode_rope_split_kernel[split_grid](
        partial_acc, partial_m, partial_l,
        query, kv_cache, block_tables, seq_lens, window_size,
        cos, sin, positions,
        norm_w_ptr, scale, norm_eps,
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        query.stride(0), query.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        kv_cache.stride(3), kv_cache.stride(4),
        block_tables.stride(0),
        cos.stride(0), cos.stride(1),
        num_q_heads, num_kv_heads,
            BLOCK_SIZE=block_size, HEAD_DIM=head_dim,
            ROTARY_DIM=rotary_dim,
            ROTARY_HALF_DIM=rotary_half_dim,
        HALF_ROTATE=1 if half_rotate else 0,
        HAS_QK_NORM=1 if has_qk_norm else 0,
        BLOCKS_PER_SPLIT=blocks_per_split,
        HAS_SLIDING_WINDOW=1 if has_sliding else 0,
        num_warps=num_warps,
    )

    reduce_grid = (num_seqs, num_q_heads)
    _paged_attn_decode_split_reduce_kernel[reduce_grid](
        output,
        partial_acc, partial_m, partial_l,
        output.stride(0), output.stride(1),
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        HEAD_DIM=head_dim,
        NUM_SPLITS=num_splits,
        num_warps=reduce_warps,
    )
    return output


def fused_rope_kv_write(k_raw, v, kv_cache, cos, sin, positions,
                         phys_blocks, blk_offsets, half_rotate=False,
                         rotary_dim=None, k_norm_weight=None, norm_eps=1e-6,
                         v_norm: bool = False):
    """
    Fused: Optional QK-Norm + RoPE to K + write K,V to paged KV cache.

    Args:
        k_raw: [num_seqs, num_kv_heads, head_dim] — pre-RoPE/Norm keys
        v: [num_seqs, num_kv_heads, head_dim] — values
        kv_cache: [num_blocks, 2, num_kv_heads, block_size, head_dim]
        cos, sin: [max_pos, half_dim]
        positions: [num_seqs]
        phys_blocks: [num_seqs]
        blk_offsets: [num_seqs]
        half_rotate: True for Qwen/Gemma, False for LLaMA
        k_norm_weight: [head_dim] K-Norm weight, or None to skip
        norm_eps: RMSNorm epsilon
        v_norm: Apply RMSNorm without scale to V before writing
    """
    if not (_HAS_TRITON and k_raw.is_cuda):
        return None  # caller should fall back to non-fused path

    num_seqs, num_kv_heads, head_dim = k_raw.shape
    rotary_dim = head_dim if rotary_dim is None else int(rotary_dim)
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        return None
    rotary_half_dim = rotary_dim // 2
    has_qk_norm = k_norm_weight is not None
    norm_w_ptr = k_norm_weight if has_qk_norm else k_raw  # dummy

    grid = (num_seqs, num_kv_heads)
    _fused_rope_kv_write_kernel[grid](
        kv_cache, k_raw, v,
        cos, sin, positions,
        phys_blocks, blk_offsets,
        norm_w_ptr,
        k_raw.stride(0), k_raw.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        kv_cache.stride(3), kv_cache.stride(4),
        cos.stride(0), cos.stride(1),
        norm_eps,
        num_kv_heads,
        ROTARY_DIM=rotary_dim,
        ROTARY_HALF_DIM=rotary_half_dim,
        HEAD_DIM=head_dim,
        HALF_ROTATE=1 if half_rotate else 0,
        HAS_QK_NORM=1 if has_qk_norm else 0,
        HAS_V_NORM=1 if v_norm else 0,
    )
    return True


def _pytorch_paged_decode(
    query,
    kv_cache,
    block_tables,
    seq_lens,
    scale,
    sliding_window: Optional[int] = None,
):
    """
    PyTorch fallback for paged attention decode.
    Slower but works everywhere (CPU, Windows, no Triton).
    """
    num_seqs, num_q_heads, head_dim = query.shape
    num_kv_heads = kv_cache.shape[2]
    block_size = kv_cache.shape[3]
    gqa_ratio = num_q_heads // num_kv_heads

    output = torch.zeros_like(query)

    # Move to CPU once to avoid per-iteration GPU→CPU sync (Fix #5)
    seq_lens_cpu = seq_lens.cpu()
    block_tables_cpu = block_tables.cpu()

    for seq_idx in range(num_seqs):
        seq_len = seq_lens_cpu[seq_idx].item()
        _, first_block, num_blocks, trim_in_first_block = _plan_decode_sliding_window(
            int(seq_len),
            int(block_size),
            sliding_window,
        )

        # Gather K, V from blocks: [seq_len, num_kv_heads, head_dim]
        k_list = []
        v_list = []
        for blk_i in range(num_blocks):
            logical_block_idx = first_block + blk_i
            phys = block_tables_cpu[seq_idx, logical_block_idx].item()
            block_token_start = logical_block_idx * block_size
            tokens_in_block = min(block_size, seq_len - block_token_start)
            k_list.append(kv_cache[phys, 0, :, :tokens_in_block, :])  # [kv_heads, tokens, dim]
            v_list.append(kv_cache[phys, 1, :, :tokens_in_block, :])

        k_full = torch.cat(k_list, dim=1)  # [kv_heads, seq_len, head_dim]
        v_full = torch.cat(v_list, dim=1)
        if trim_in_first_block > 0:
            k_full = k_full[:, trim_in_first_block:, :]
            v_full = v_full[:, trim_in_first_block:, :]

        # Expand for GQA (zero-copy)
        if gqa_ratio > 1:
            kv_h, sl, hd = k_full.shape
            k_full = k_full[:, None, :, :].expand(kv_h, gqa_ratio, sl, hd).reshape(num_q_heads, sl, hd)
            v_full = v_full[:, None, :, :].expand(kv_h, gqa_ratio, sl, hd).reshape(num_q_heads, sl, hd)

        # q: [num_q_heads, head_dim] -> [num_q_heads, 1, head_dim]
        q = query[seq_idx].unsqueeze(1).float()
        k_f = k_full.float()  # [num_q_heads, seq_len, head_dim]
        v_f = v_full.float()

        # Attention: [num_q_heads, 1, seq_len]
        scores = torch.bmm(q, k_f.transpose(1, 2)) * scale
        attn_weights = torch.softmax(scores, dim=-1)

        # Output: [num_q_heads, 1, head_dim]
        attn_out = torch.bmm(attn_weights, v_f).squeeze(1)
        output[seq_idx] = attn_out.to(output.dtype)

    return output


def paged_attention_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    split_policy_override: Optional[int] = None,
) -> torch.Tensor:
    """
    Paged attention for decode step (1 token per sequence).
    Uses Triton kernel on Linux GPU, PyTorch fallback otherwise.

    Args:
        query: [num_seqs, num_q_heads, head_dim]
        kv_cache: [num_blocks, 2, num_kv_heads, block_size, head_dim]
        block_tables: [num_seqs, max_blocks_per_seq]
        seq_lens: [num_seqs]
        scale: 1/sqrt(head_dim)
        out: Optional pre-allocated output buffer [num_seqs, num_q_heads, head_dim]
        sliding_window: Optional local decode window. When set, only the most
            recent tokens participate in attention.
        split_policy_override: Optional model/hardware policy. An explicit
            ``MEGAGEMM_PAGED_DECODE_SPLITS`` value still takes precedence.

    Returns:
        output: [num_seqs, num_q_heads, head_dim]
    """
    head_dim = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    if _HAS_TRITON and query.is_cuda:
        return _triton_paged_decode(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            scale,
            out=out,
            sliding_window=sliding_window,
            split_policy_override=split_policy_override,
        )
    else:
        return _pytorch_paged_decode(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            scale,
            sliding_window=sliding_window,
        )


if _HAS_TRITON:
    @triton.jit
    def _gemma4_long_sliding_prefill_kernel(
        output_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        scale,
        stride_qb,
        stride_qh,
        stride_qs,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vs,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_os,
        stride_od,
        NUM_Q_TILES: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        NUM_Q_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        SLIDING_WINDOW: tl.constexpr,
        MAX_WINDOW_TILES: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        pid_seq_tile = tl.program_id(0)
        q_head_idx = tl.program_id(1)
        batch_idx = pid_seq_tile // NUM_Q_TILES
        q_tile_idx = pid_seq_tile % NUM_Q_TILES
        kv_head_idx = q_head_idx // (NUM_Q_HEADS // NUM_KV_HEADS)

        m_start = q_tile_idx * BLOCK_M
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < SEQ_LEN
        d_offsets = tl.arange(0, HEAD_DIM)

        q_ptrs = (
            q_ptr
            + batch_idx * stride_qb
            + q_head_idx * stride_qh
            + m_offsets[:, None] * stride_qs
            + d_offsets[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

        m_prev = tl.full([BLOCK_M], value=-1.0e20, dtype=tl.float32)
        l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        first_key = tl.maximum(0, m_start - SLIDING_WINDOW + 1)
        first_k_tile = first_key // BLOCK_N
        n_lane = tl.arange(0, BLOCK_N)

        for tile_offset in range(MAX_WINDOW_TILES):
            n_offsets = (first_k_tile + tile_offset) * BLOCK_N + n_lane
            n_mask = n_offsets < SEQ_LEN

            k_ptrs = (
                k_ptr
                + batch_idx * stride_kb
                + kv_head_idx * stride_kh
                + d_offsets[:, None] * stride_kd
                + n_offsets[None, :] * stride_ks
            )
            k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0)
            scores = tl.dot(q, k, out_dtype=tl.float32) * scale

            valid = (
                m_mask[:, None]
                & n_mask[None, :]
                & (n_offsets[None, :] <= m_offsets[:, None])
                & (
                    n_offsets[None, :]
                    >= (m_offsets[:, None] - SLIDING_WINDOW + 1)
                )
            )
            scores = tl.where(valid, scores, -1.0e20)

            m_cur = tl.max(scores, axis=1)
            m_new = tl.maximum(m_prev, m_cur)
            alpha = tl.exp(m_prev - m_new)
            probs = tl.exp(scores - m_new[:, None])
            probs = tl.where(valid, probs, 0.0)
            l_cur = tl.sum(probs, axis=1)

            v_ptrs = (
                v_ptr
                + batch_idx * stride_vb
                + kv_head_idx * stride_vh
                + n_offsets[:, None] * stride_vs
                + d_offsets[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(
                probs.to(v.dtype),
                v,
                out_dtype=tl.float32,
            )
            l_prev = l_prev * alpha + l_cur
            m_prev = m_new

        acc = tl.where(l_prev[:, None] > 0, acc / l_prev[:, None], 0.0)
        out_ptrs = (
            output_ptr
            + batch_idx * stride_ob
            + q_head_idx * stride_oh
            + m_offsets[:, None] * stride_os
            + d_offsets[None, :] * stride_od
        )
        tl.store(
            out_ptrs,
            acc.to(output_ptr.dtype.element_ty),
            mask=m_mask[:, None],
        )


    @triton.jit
    def _gemma4_e2b_l4_sliding_prefill_kernel(
        output_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        scale,
        stride_qb,
        stride_qh,
        stride_qs,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_os,
        stride_od,
        NUM_Q_TILES: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        SLIDING_WINDOW: tl.constexpr,
        MAX_WINDOW_TILES: tl.constexpr,
        GROUP_HEADS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """Gemma 4 E2B/L4 sliding attention with KV1 head reuse.

        A program owns ``GROUP_HEADS`` query heads over one temporal query
        tile.  Flattening those heads into ``BLOCK_ROWS`` keeps the tensor-core
        M dimension useful while reducing causal-boundary waste compared with
        one large temporal tile.  K/V are loaded once for the whole head group.
        """
        pid_seq_tile = tl.program_id(0)
        head_group_idx = tl.program_id(1)
        batch_idx = pid_seq_tile // NUM_Q_TILES
        q_tile_idx = pid_seq_tile % NUM_Q_TILES

        row_offsets = tl.arange(0, BLOCK_ROWS)
        head_lanes = row_offsets // BLOCK_M
        token_lanes = row_offsets - head_lanes * BLOCK_M
        q_head_offsets = head_group_idx * GROUP_HEADS + head_lanes

        m_start = q_tile_idx * BLOCK_M
        m_offsets = m_start + token_lanes
        m_mask = m_offsets < SEQ_LEN
        d_offsets = tl.arange(0, HEAD_DIM)

        q_ptrs = (
            q_ptr
            + batch_idx * stride_qb
            + q_head_offsets[:, None] * stride_qh
            + m_offsets[:, None] * stride_qs
            + d_offsets[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

        m_prev = tl.full([BLOCK_ROWS], value=-1.0e20, dtype=tl.float32)
        l_prev = tl.zeros([BLOCK_ROWS], dtype=tl.float32)
        acc = tl.zeros([BLOCK_ROWS, HEAD_DIM], dtype=tl.float32)

        first_key = tl.maximum(0, m_start - SLIDING_WINDOW + 1)
        first_k_tile = first_key // BLOCK_N
        n_lane = tl.arange(0, BLOCK_N)

        for tile_offset in range(MAX_WINDOW_TILES):
            n_offsets = (first_k_tile + tile_offset) * BLOCK_N + n_lane
            n_mask = n_offsets < SEQ_LEN

            # E2B has exactly one KV head, so every query head in the program
            # consumes the same K/V tile.
            k_ptrs = (
                k_ptr
                + batch_idx * stride_kb
                + d_offsets[:, None] * stride_kd
                + n_offsets[None, :] * stride_ks
            )
            k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0)
            scores = tl.dot(q, k, out_dtype=tl.float32) * scale

            valid = (
                m_mask[:, None]
                & n_mask[None, :]
                & (n_offsets[None, :] <= m_offsets[:, None])
                & (
                    n_offsets[None, :]
                    >= (m_offsets[:, None] - SLIDING_WINDOW + 1)
                )
            )
            scores = tl.where(valid, scores, -1.0e20)

            m_cur = tl.max(scores, axis=1)
            m_new = tl.maximum(m_prev, m_cur)
            alpha = tl.exp(m_prev - m_new)
            probs = tl.exp(scores - m_new[:, None])
            probs = tl.where(valid, probs, 0.0)
            l_cur = tl.sum(probs, axis=1)

            v_ptrs = (
                v_ptr
                + batch_idx * stride_vb
                + n_offsets[:, None] * stride_vs
                + d_offsets[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(
                probs.to(v.dtype),
                v,
                out_dtype=tl.float32,
            )
            l_prev = l_prev * alpha + l_cur
            m_prev = m_new

        acc = tl.where(l_prev[:, None] > 0, acc / l_prev[:, None], 0.0)
        out_ptrs = (
            output_ptr
            + batch_idx * stride_ob
            + q_head_offsets[:, None] * stride_oh
            + m_offsets[:, None] * stride_os
            + d_offsets[None, :] * stride_od
        )
        tl.store(
            out_ptrs,
            acc.to(output_ptr.dtype.element_ty),
            mask=m_mask[:, None],
        )


def gemma4_e2b_l4_sliding_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sliding_window: int,
    scale: Optional[float] = None,
    group_heads: Optional[int] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    force: bool = False,
) -> Optional[torch.Tensor]:
    """Exact-shape production kernel for Gemma 4 E2B on NVIDIA L4.

    This path is deliberately narrower than the older A100/A4B long-prefill
    kernels: BF16, B2/B4/B8, Q8/KV1, S2048..2304, H256, W512, and L4 only in
    production. Every admitted batch uses a loaded-model measured launch
    geometry. The bounded sequence range includes the chat-template tokens
    added to the publication workload. ``force`` exists solely for tuning
    harnesses.
    """
    global _GEMMA4_E2B_L4_SLIDING_PREFILL_DISABLED
    global _GEMMA4_E2B_L4_B2_SLIDING_PREFILL_DISABLED
    global _GEMMA4_E2B_L4_B4_SLIDING_PREFILL_DISABLED
    global _GEMMA4_E2B_L4_SLIDING_PREFILL_FAILURE
    global _GEMMA4_E2B_L4_SLIDING_PREFILL_LOGGED
    global _GEMMA4_E2B_L4_SLIDING_PREFILL_HITS

    if _GEMMA4_E2B_L4_SLIDING_PREFILL_DISABLED and not force:
        return None
    if not (_HAS_TRITON and q.is_cuda and k.is_cuda and v.is_cuda):
        return None
    if q.device != k.device or q.device != v.device:
        return None
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return None
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return None

    batch_size, num_q_heads, seq_len, head_dim = q.shape
    if (
        batch_size == 2
        and _GEMMA4_E2B_L4_B2_SLIDING_PREFILL_DISABLED
        and not force
    ):
        return None
    if (
        batch_size == 4
        and _GEMMA4_E2B_L4_B4_SLIDING_PREFILL_DISABLED
        and not force
    ):
        return None
    if tuple(k.shape) != (batch_size, 1, seq_len, 256):
        return None
    if tuple(v.shape) != tuple(k.shape):
        return None
    if (
        batch_size not in (2, 4, 8)
        or num_q_heads != 8
        or seq_len < 2048
        or seq_len > 2304
        or head_dim != 256
        or int(sliding_window) != 512
    ):
        return None

    _, device_name, _ = _cuda_device_info(q.device)
    if "l4" not in _device_name_tokens(device_name):
        return None

    if batch_size == 2:
        env_prefix = "MEGAGEMM_GEMMA4_E2B_L4_B2_SLIDING_"
        default_group_heads, default_block_m = 2, 16
    elif batch_size == 4:
        env_prefix = "MEGAGEMM_GEMMA4_E2B_L4_B4_SLIDING_"
        default_group_heads, default_block_m = 2, 16
    else:
        env_prefix = "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_"
        default_group_heads, default_block_m = 4, 8
    group_heads = int(
        group_heads
        if group_heads is not None
        else _env_int(env_prefix + "GROUP_HEADS", default_group_heads)
    )
    block_m = int(
        block_m
        if block_m is not None
        else _env_int(env_prefix + "BLOCK_M", default_block_m)
    )
    block_n = int(
        block_n
        if block_n is not None
        else _env_int(env_prefix + "BLOCK_N", 64)
    )
    num_warps = int(
        num_warps
        if num_warps is not None
        else _env_int(env_prefix + "NUM_WARPS", 4)
    )
    num_stages = int(
        num_stages
        if num_stages is not None
        else _env_int(env_prefix + "NUM_STAGES", 2)
    )
    block_rows = int(group_heads * block_m)
    if (
        group_heads not in (1, 2, 4, 8)
        or num_q_heads % group_heads != 0
        or block_m not in (4, 8, 16, 32)
        or block_n not in (32, 64, 128)
        or block_rows not in (16, 32, 64)
        or num_warps not in (4, 8)
        or num_stages not in (2, 3)
    ):
        return None

    num_q_tiles = triton.cdiv(seq_len, block_m)
    max_window_tiles = triton.cdiv(
        int(sliding_window) + block_m + block_n - 2,
        block_n,
    )
    output = torch.empty_like(q)
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    grid = (batch_size * num_q_tiles, num_q_heads // group_heads)

    try:
        _gemma4_e2b_l4_sliding_prefill_kernel[grid](
            output,
            q,
            k,
            v,
            float(scale),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(2),
            v.stride(3),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            NUM_Q_TILES=num_q_tiles,
            SEQ_LEN=seq_len,
            SLIDING_WINDOW=int(sliding_window),
            MAX_WINDOW_TILES=max_window_tiles,
            GROUP_HEADS=group_heads,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_ROWS=block_rows,
            HEAD_DIM=head_dim,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    except Exception as exc:
        _GEMMA4_E2B_L4_SLIDING_PREFILL_FAILURE = (
            f"{type(exc).__name__}: {exc}"
        )
        # B4 and B8 have distinct production geometries. A failure disables
        # only the affected dispatch instead of poisoning the other batch.
        if not force:
            if batch_size == 2:
                _GEMMA4_E2B_L4_B2_SLIDING_PREFILL_DISABLED = True
            elif batch_size == 4:
                _GEMMA4_E2B_L4_B4_SLIDING_PREFILL_DISABLED = True
            else:
                _GEMMA4_E2B_L4_SLIDING_PREFILL_DISABLED = True
        if not _GEMMA4_E2B_L4_SLIDING_PREFILL_LOGGED:
            _GEMMA4_E2B_L4_SLIDING_PREFILL_LOGGED = True
            print(
                "[MegaGemm] Gemma4 E2B/L4 sliding prefill kernel failed; "
                "falling back to SDPA "
                f"({_GEMMA4_E2B_L4_SLIDING_PREFILL_FAILURE})"
            )
        return None

    _GEMMA4_E2B_L4_SLIDING_PREFILL_HITS += 1
    if not _GEMMA4_E2B_L4_SLIDING_PREFILL_LOGGED:
        _GEMMA4_E2B_L4_SLIDING_PREFILL_LOGGED = True
        print(
            "[MegaGemm] Gemma4 E2B/L4 sliding prefill active: "
            f"B{batch_size} S{seq_len} Q8/KV1 H256 W512 group_heads={group_heads} "
            f"block_m={block_m} block_n={block_n} warps={num_warps}"
        )
    return output


def gemma4_long_sliding_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sliding_window: int,
    scale: Optional[float] = None,
    block_m: int = 64,
    num_warps: int = 8,
    num_stages: int = 2,
    force: bool = False,
) -> Optional[torch.Tensor]:
    """Memory-bounded sliding attention for the measured Gemma4 A4B topology."""
    global _GEMMA4_LONG_SLIDING_PREFILL_DISABLED
    global _GEMMA4_LONG_SLIDING_PREFILL_FAILURE
    global _GEMMA4_LONG_SLIDING_PREFILL_LOGGED

    if _GEMMA4_LONG_SLIDING_PREFILL_DISABLED and not force:
        return None
    if not force and not _env_bool(
        "MEGAGEMM_GEMMA4_LONG_SLIDING_PREFILL",
        True,
    ):
        return None
    if not (_HAS_TRITON and q.is_cuda and k.is_cuda and v.is_cuda):
        return None
    if q.device != k.device or q.device != v.device:
        return None
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return None
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return None
    batch_size, num_q_heads, seq_len, head_dim = q.shape
    if tuple(k.shape) != (batch_size, 8, seq_len, head_dim):
        return None
    if tuple(v.shape) != tuple(k.shape):
        return None
    if (
        num_q_heads != 16
        or batch_size not in (8, 16)
        or seq_len != 2048
        or head_dim != 256
        or int(sliding_window) != 1024
        or int(block_m) not in (8, 16, 32, 64, 128)
        or int(num_warps) not in (4, 8)
        or int(num_stages) not in (2, 3)
    ):
        return None
    _, device_name, _ = _cuda_device_info(q.device)
    if "a100" not in _device_name_tokens(device_name):
        return None

    block_m = int(block_m)
    block_n = 64
    num_q_tiles = triton.cdiv(seq_len, block_m)
    max_window_tiles = triton.cdiv(
        int(sliding_window) + block_m + block_n - 2,
        block_n,
    )
    output = torch.empty_like(q)
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    grid = (batch_size * num_q_tiles, num_q_heads)
    try:
        _gemma4_long_sliding_prefill_kernel[grid](
            output,
            q,
            k,
            v,
            float(scale),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            NUM_Q_TILES=num_q_tiles,
            SEQ_LEN=seq_len,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=int(k.shape[1]),
            SLIDING_WINDOW=int(sliding_window),
            MAX_WINDOW_TILES=max_window_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HEAD_DIM=head_dim,
            num_warps=int(num_warps),
            num_stages=int(num_stages),
        )
    except Exception as exc:
        _GEMMA4_LONG_SLIDING_PREFILL_FAILURE = f"{type(exc).__name__}: {exc}"
        if not force:
            _GEMMA4_LONG_SLIDING_PREFILL_DISABLED = True
        if not _GEMMA4_LONG_SLIDING_PREFILL_LOGGED:
            _GEMMA4_LONG_SLIDING_PREFILL_LOGGED = True
            print(
                "[MegaGemm] Gemma4 long sliding prefill kernel failed; "
                f"falling back to SDPA ({_GEMMA4_LONG_SLIDING_PREFILL_FAILURE})"
            )
        return None

    if not _GEMMA4_LONG_SLIDING_PREFILL_LOGGED:
        _GEMMA4_LONG_SLIDING_PREFILL_LOGGED = True
        print(
            "[MegaGemm] Gemma4 long sliding prefill active: "
            f"B={batch_size} S={seq_len} H256 GQA2 window=1024 "
            f"block_m={block_m} warps={num_warps}"
        )
    return output


def gemma4_long_full_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
    block_m: int = 32,
    block_n: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
    force: bool = False,
) -> Optional[torch.Tensor]:
    """Causal full attention for the measured Gemma4 A4B H512 topology."""
    global _GEMMA4_LONG_FULL_PREFILL_DISABLED
    global _GEMMA4_LONG_FULL_PREFILL_FAILURE
    global _GEMMA4_LONG_FULL_PREFILL_LOGGED

    if _GEMMA4_LONG_FULL_PREFILL_DISABLED and not force:
        return None
    if not force and not _env_bool(
        "MEGAGEMM_GEMMA4_LONG_FULL_PREFILL",
        False,
    ):
        return None
    if not (_HAS_TRITON and q.is_cuda and k.is_cuda and v.is_cuda):
        return None
    if q.device != k.device or q.device != v.device:
        return None
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return None
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return None
    batch_size, num_q_heads, seq_len, head_dim = q.shape
    if tuple(k.shape) != (batch_size, 2, seq_len, head_dim):
        return None
    if tuple(v.shape) != tuple(k.shape):
        return None
    if (
        num_q_heads != 16
        or batch_size not in (8, 16)
        or seq_len != 2048
        or head_dim != 512
        or int(block_m) not in (8, 16, 32, 64)
        or int(block_n) not in (32, 64, 128)
        or int(num_warps) not in (4, 8)
        or int(num_stages) not in (2, 3)
    ):
        return None
    _, device_name, _ = _cuda_device_info(q.device)
    if "a100" not in _device_name_tokens(device_name):
        return None

    block_m = int(block_m)
    block_n = int(block_n)
    num_q_tiles = triton.cdiv(seq_len, block_m)
    max_key_tiles = triton.cdiv(seq_len, block_n)
    output = torch.empty_like(q)
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    grid = (batch_size * num_q_tiles, num_q_heads)
    try:
        _gemma4_long_sliding_prefill_kernel[grid](
            output,
            q,
            k,
            v,
            float(scale),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            NUM_Q_TILES=num_q_tiles,
            SEQ_LEN=seq_len,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=int(k.shape[1]),
            SLIDING_WINDOW=seq_len,
            MAX_WINDOW_TILES=max_key_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HEAD_DIM=head_dim,
            num_warps=int(num_warps),
            num_stages=int(num_stages),
        )
    except Exception as exc:
        _GEMMA4_LONG_FULL_PREFILL_FAILURE = f"{type(exc).__name__}: {exc}"
        if not force:
            _GEMMA4_LONG_FULL_PREFILL_DISABLED = True
        if not _GEMMA4_LONG_FULL_PREFILL_LOGGED:
            _GEMMA4_LONG_FULL_PREFILL_LOGGED = True
            print(
                "[MegaGemm] Gemma4 long full prefill kernel failed; "
                f"falling back to SDPA ({_GEMMA4_LONG_FULL_PREFILL_FAILURE})"
            )
        return None

    if not _GEMMA4_LONG_FULL_PREFILL_LOGGED:
        _GEMMA4_LONG_FULL_PREFILL_LOGGED = True
        print(
            "[MegaGemm] Gemma4 long full prefill active: "
            f"B={batch_size} S={seq_len} H512 GQA8 "
            f"block_m={block_m} block_n={block_n} warps={num_warps}"
        )
    return output


def prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Attention for prefill phase using PyTorch SDPA.
    Handles GQA with native SDPA support when available, falling back to KV expansion.

    Args:
        attn_mask: Optional attention mask [batch, 1, seq_len, seq_len].
                   When provided, is_causal is ignored (mask handles both
                   causal masking and padding).
    """
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]

    scale_kwargs = {} if scale is None else {"scale": float(scale)}

    if num_kv_heads < num_q_heads:
        ratio = num_q_heads // num_kv_heads
        if _prefill_gqa_mode() == "expand":
            k = k.repeat_interleave(ratio, dim=1)
            v = v.repeat_interleave(ratio, dim=1)
        else:
            try:
                if attn_mask is not None:
                    return torch.nn.functional.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=attn_mask,
                        is_causal=False,
                        enable_gqa=True,
                        **scale_kwargs,
                    )
                return torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=is_causal,
                    enable_gqa=True,
                    **scale_kwargs,
                )
            except TypeError:
                k = k.repeat_interleave(ratio, dim=1)
                v = v.repeat_interleave(ratio, dim=1)

    if attn_mask is not None:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, **scale_kwargs
        )
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=is_causal, **scale_kwargs
    )

# =============================================================================
# Packed Prefill Attention — Zero-Waste Batched Prefill
# =============================================================================
# Instead of left-padding + O(N²) attention mask, we concatenate all
# sequences into a single packed tensor and use cu_seqlens to track
# boundaries. Each sequence gets independent causal attention.
#
# Layout:
#   packed_q: [total_tokens, num_q_heads, head_dim]
#   cu_seqlens: [num_seqs + 1]  e.g. [0, 320, 512, 900]
#
# Benefits:
#   - Zero padding waste (every FLOPs is useful)
#   - No O(N²) attention mask allocation
#   - Works with GQA natively
# =============================================================================

if _HAS_TRITON:
    @triton.jit
    def _packed_attention_kernel(
        output_ptr,
        q_ptr, k_ptr, v_ptr,
        # Pre-computed per-tile metadata (avoids in-kernel break/loop)
        tile_seq_start_ptr,   # [total_tiles] — global token offset of each tile's seq
        tile_seq_len_ptr,     # [total_tiles] — length of each tile's seq
        tile_local_idx_ptr,   # [total_tiles] — local tile index within the seq
        scale,
        stride_qt, stride_qh, stride_qd,
        stride_kt, stride_kh, stride_kd,
        stride_vt, stride_vh, stride_vd,
        stride_ot, stride_oh, stride_od,
        num_q_heads,
        num_kv_heads,
        MAX_K_TILES: tl.constexpr,  # upper bound for K tile loop
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """
        Packed Prefill Attention Kernel.

        Each program handles one (tile, query_head) pair.
        tile_seq_start/len/local_idx are pre-computed on host to avoid
        in-kernel break statements (not supported by Triton).

        Grid: (total_tiles, num_q_heads)
        """
        pid_tile = tl.program_id(0)
        q_head_idx = tl.program_id(1)

        # GQA head mapping
        gqa_ratio = num_q_heads // num_kv_heads
        kv_head_idx = q_head_idx // gqa_ratio

        # Load pre-computed tile metadata — O(1), no loops
        seq_start = tl.load(tile_seq_start_ptr + pid_tile)   # global token offset
        seq_len = tl.load(tile_seq_len_ptr + pid_tile)       # seq length
        tile_idx = tl.load(tile_local_idx_ptr + pid_tile)    # which tile within seq

        # Query tile positions (local within sequence)
        m_start = tile_idx * BLOCK_M
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < seq_len

        d_offsets = tl.arange(0, HEAD_DIM)

        # Load Q tile: [BLOCK_M, HEAD_DIM]
        q_base = seq_start
        q_ptrs = q_ptr + (q_base + m_offsets[:, None]) * stride_qt + \
                 q_head_idx * stride_qh + d_offsets[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32) * scale

        # Online softmax accumulators
        m_prev = tl.full([BLOCK_M], value=-1e20, dtype=tl.float32)
        l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Causal attention only needs keys up to the current query tile.
        if IS_CAUSAL:
            n_end = tl.minimum(m_start + BLOCK_M, seq_len)
        else:
            n_end = seq_len

        for n_tile in range(MAX_K_TILES):
            n_start = n_tile * BLOCK_N
            # Early exit check (no break needed — just skip via mask)
            still_valid = n_start < n_end

            n_offsets = n_start + tl.arange(0, BLOCK_N)
            n_mask = still_valid & (n_offsets < seq_len)

            # Load K tile: [BLOCK_N, HEAD_DIM]
            k_ptrs = k_ptr + (q_base + n_offsets[:, None]) * stride_kt + \
                     kv_head_idx * stride_kh + d_offsets[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

            # QK^T: [BLOCK_M, BLOCK_N]
            scores = tl.dot(q, tl.trans(k))

            # Validity mask, with optional causal masking.
            valid_mask = m_mask[:, None] & n_mask[None, :]
            if IS_CAUSAL:
                causal_mask = m_offsets[:, None] >= n_offsets[None, :]
                valid_mask = valid_mask & causal_mask
            scores = tl.where(valid_mask, scores, -1e20)

            # Online softmax
            m_cur = tl.max(scores, axis=1)
            m_new = tl.maximum(m_prev, m_cur)
            alpha = tl.exp(m_prev - m_new)
            exp_scores = tl.exp(scores - m_new[:, None])
            exp_scores = tl.where(valid_mask, exp_scores, 0.0)
            l_cur = tl.sum(exp_scores, axis=1)

            # Load V tile: [BLOCK_N, HEAD_DIM]
            v_ptrs = v_ptr + (q_base + n_offsets[:, None]) * stride_vt + \
                     kv_head_idx * stride_vh + d_offsets[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

            # Accumulate
            acc = acc * alpha[:, None] + tl.dot(exp_scores, v)
            l_prev = l_prev * alpha + l_cur
            m_prev = m_new

        # Normalize
        acc = tl.where(l_prev[:, None] > 0, acc / l_prev[:, None], 0.0)

        # Store
        out_ptrs = output_ptr + (q_base + m_offsets[:, None]) * stride_ot + \
                   q_head_idx * stride_oh + d_offsets[None, :] * stride_od
        tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=m_mask[:, None])


_TRITON_PACKED_ATTENTION_DISABLED = set()
_TRITON_PACKED_ATTENTION_FAILURE_LOGGED = set()


def _log_triton_packed_failure(mode: str, exc: Exception) -> None:
    if mode in _TRITON_PACKED_ATTENTION_FAILURE_LOGGED:
        return
    _TRITON_PACKED_ATTENTION_FAILURE_LOGGED.add(mode)
    print(f"  [MegaGemm] Triton packed attention ({mode}) failed, falling back to SDPA: {exc}")


def _triton_packed_attention(
    q,
    k,
    v,
    cu_seqlens,
    scale,
    causal: bool,
    packed_meta: Optional[PackedAttentionMetadata] = None,
):
    """Triton packed attention dispatch for causal and non-causal varlen batches."""
    total_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    mode = "causal" if causal else "noncausal"

    if mode in _TRITON_PACKED_ATTENTION_DISABLED:
        return None

    # Fix #1: Do NOT expand KV heads here — the kernel handles GQA natively
    # via kv_head_idx = q_head_idx // gqa_ratio. Pre-expanding wastes memory
    # and hides the kernel's GQA logic (gqa_ratio becomes 1 = no-op).

    output = torch.empty_like(q)

    packed_meta = _get_packed_attention_metadata(cu_seqlens, head_dim=head_dim, packed_meta=packed_meta)

    tile_seq_start = packed_meta.tile_seq_start
    tile_seq_len = packed_meta.tile_seq_len
    tile_local_idx = packed_meta.tile_local_idx
    if tile_seq_start is None or tile_seq_len is None or tile_local_idx is None:
        return output

    BLOCK_M = packed_meta.block_m
    BLOCK_N = packed_meta.block_n
    total_tiles = int(tile_seq_start.shape[0])
    MAX_K_TILES = packed_meta.max_k_tiles

    # Adaptive num_stages: newer GPUs (Hopper/Ada) benefit from pipelining
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        if cc[0] >= 9:      # Hopper (H100)
            n_stages = 3
        elif cc[0] >= 8:    # Ampere (A100) / Ada (L4, 4090)
            n_stages = 2
        else:               # Turing (T4) and older
            n_stages = 1
    else:
        n_stages = 1

    grid = (total_tiles, num_q_heads)

    try:
        _packed_attention_kernel[grid](
            output,
            q, k, v,
            tile_seq_start, tile_seq_len, tile_local_idx,
            scale,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            num_q_heads, num_kv_heads,
            MAX_K_TILES=MAX_K_TILES,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=head_dim,
            IS_CAUSAL=causal,
            num_warps=8, num_stages=n_stages,
        )
    except Exception as exc:
        _TRITON_PACKED_ATTENTION_DISABLED.add(mode)
        _log_triton_packed_failure(mode, exc)
        return None
    return output


def _triton_packed_prefill(q, k, v, cu_seqlens, scale):
    """Backward-compatible Triton packed prefill dispatch."""
    return _triton_packed_attention(q, k, v, cu_seqlens, scale, causal=True)


def _pytorch_packed_attention(
    q,
    k,
    v,
    cu_seqlens,
    scale,
    causal: bool,
    packed_meta: Optional[PackedAttentionMetadata] = None,
):
    """
    PyTorch SDPA per-sequence fallback.
    Uses FlashAttention CUDA backend internally via torch SDPA.
    Optimized: single GPU->CPU sync, GQA-native SDPA when available.
    """
    total_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    num_seqs = cu_seqlens.shape[0] - 1
    gqa_ratio = num_q_heads // num_kv_heads if num_kv_heads < num_q_heads else 0

    output = torch.empty_like(q)

    # Single GPU->CPU transfer instead of 2*num_seqs .item() calls
    packed_meta = _reuse_packed_attention_metadata(cu_seqlens, head_dim, packed_meta)
    boundaries = packed_meta.boundaries if packed_meta is not None else cu_seqlens.tolist()

    for i in range(num_seqs):
        start = boundaries[i]
        end = boundaries[i + 1]
        if start == end:
            continue

        # [seq_len, heads, dim] → [1, heads, seq_len, dim]
        qi = q[start:end].transpose(0, 1).unsqueeze(0)
        ki = k[start:end].transpose(0, 1).unsqueeze(0)
        vi = v[start:end].transpose(0, 1).unsqueeze(0)

        if gqa_ratio > 0:
            if _prefill_gqa_mode() == "expand":
                ki = ki.repeat_interleave(gqa_ratio, dim=1)
                vi = vi.repeat_interleave(gqa_ratio, dim=1)
                try:
                    oi = torch.nn.functional.scaled_dot_product_attention(
                        qi, ki, vi, is_causal=causal, scale=scale,
                    )
                except TypeError:
                    oi = torch.nn.functional.scaled_dot_product_attention(
                        qi, ki, vi, is_causal=causal,
                    )
            else:
                try:
                    oi = torch.nn.functional.scaled_dot_product_attention(
                        qi, ki, vi, is_causal=causal, enable_gqa=True, scale=scale,
                    )
                except TypeError:
                    ki = ki.repeat_interleave(gqa_ratio, dim=1)
                    vi = vi.repeat_interleave(gqa_ratio, dim=1)
                    try:
                        oi = torch.nn.functional.scaled_dot_product_attention(
                            qi, ki, vi, is_causal=causal, scale=scale,
                        )
                    except TypeError:
                        oi = torch.nn.functional.scaled_dot_product_attention(
                            qi, ki, vi, is_causal=causal,
                        )
        else:
            try:
                oi = torch.nn.functional.scaled_dot_product_attention(
                    qi, ki, vi, is_causal=causal, scale=scale,
                )
            except TypeError:
                oi = torch.nn.functional.scaled_dot_product_attention(
                    qi, ki, vi, is_causal=causal,
                )

        output[start:end] = oi.squeeze(0).transpose(0, 1)

    return output


def _uniform_packed_seq_len(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    num_seqs: int,
) -> Optional[int]:
    if num_seqs <= 1 or total_tokens <= 0 or total_tokens % num_seqs != 0:
        return None
    seq_len = total_tokens // num_seqs
    key = (
        cu_seqlens.device.type,
        cu_seqlens.device.index if cu_seqlens.device.index is not None else -1,
        str(cu_seqlens.dtype),
        int(num_seqs),
        int(seq_len),
    )
    expected = _UNIFORM_CU_EXPECTED_CACHE.get(key)
    if expected is None or expected.device != cu_seqlens.device:
        expected = torch.arange(
            num_seqs + 1,
            dtype=cu_seqlens.dtype,
            device=cu_seqlens.device,
        ) * seq_len
        _UNIFORM_CU_EXPECTED_CACHE[key] = expected
    if torch.equal(cu_seqlens, expected):
        return seq_len
    return None


def _uniform_packed_sdpa_has_headroom(
    q: torch.Tensor,
    num_seqs: int,
    num_q_heads: int,
    seq_len: int,
) -> bool:
    # On Turing/PyTorch, batched SDPA can materialize fp32 attention scores.
    # For batch=8, heads=16, seq_len~=2k this is about 2 GiB. Keep the path
    # behind a guard, but leave enough room for the long-context T4 case to try
    # the uniform batch instead of falling straight back to 8 per-seq SDPA calls.
    score_bytes = int(num_seqs) * int(num_q_heads) * int(seq_len) * int(seq_len) * 4

    max_mb_raw = os.environ.get("MEGAGEMM_PACKED_ATTN_UNIFORM_MAX_SCORE_MB", "").strip()
    try:
        max_score_mb = int(max_mb_raw) if max_mb_raw else 1024
    except ValueError:
        max_score_mb = 1024
    max_score_bytes = max_score_mb * 1024 * 1024
    if max_score_bytes > 0 and score_bytes > max_score_bytes:
        return False

    if not q.is_cuda or not torch.cuda.is_available():
        return True

    reserve_raw = os.environ.get("MEGAGEMM_PACKED_ATTN_UNIFORM_RESERVE_MB", "").strip()
    try:
        reserve_bytes = (int(reserve_raw) if reserve_raw else 512) * 1024 * 1024
    except ValueError:
        reserve_bytes = 512 * 1024 * 1024

    try:
        free_bytes = torch.cuda.mem_get_info(q.device)[0]
    except Exception:
        return True

    # torch.cuda.mem_get_info() does not count PyTorch's reserved-but-unused
    # caching allocator blocks as free. After the first SDPA layer, that cache is
    # often exactly where the next same-shape SDPA allocation can be served from.
    # Treat most of it as reusable headroom while keeping a conservative reserve.
    reusable_bytes = 0
    if _env_bool("MEGAGEMM_PACKED_ATTN_UNIFORM_COUNT_RESERVED", default=True):
        try:
            device = q.device if q.device.index is not None else torch.cuda.current_device()
            reserved = torch.cuda.memory_reserved(device)
            allocated = torch.cuda.memory_allocated(device)
            reusable_bytes = max(0, int(reserved) - int(allocated))
        except Exception:
            reusable_bytes = 0
    return score_bytes <= max(0, free_bytes + reusable_bytes - reserve_bytes)


def _pytorch_uniform_packed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_seqs: int,
    seq_len: int,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    """Run a uniform packed batch as one batched SDPA call instead of per-seq."""
    total_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    gqa_ratio = num_q_heads // num_kv_heads if num_kv_heads < num_q_heads else 0

    qb = q.reshape(num_seqs, seq_len, num_q_heads, head_dim).transpose(1, 2)
    kb = k.reshape(num_seqs, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    vb = v.reshape(num_seqs, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    if gqa_ratio > 0:
        if _prefill_gqa_mode() == "expand":
            kb = kb.repeat_interleave(gqa_ratio, dim=1)
            vb = vb.repeat_interleave(gqa_ratio, dim=1)
            try:
                out = torch.nn.functional.scaled_dot_product_attention(
                    qb, kb, vb, is_causal=causal, scale=scale,
                )
            except TypeError:
                out = torch.nn.functional.scaled_dot_product_attention(
                    qb, kb, vb, is_causal=causal,
                )
        else:
            try:
                out = torch.nn.functional.scaled_dot_product_attention(
                    qb,
                    kb,
                    vb,
                    is_causal=causal,
                    enable_gqa=True,
                    scale=scale,
                )
            except TypeError:
                kb = kb.repeat_interleave(gqa_ratio, dim=1)
                vb = vb.repeat_interleave(gqa_ratio, dim=1)
                try:
                    out = torch.nn.functional.scaled_dot_product_attention(
                        qb, kb, vb, is_causal=causal, scale=scale,
                    )
                except TypeError:
                    out = torch.nn.functional.scaled_dot_product_attention(
                        qb, kb, vb, is_causal=causal,
                    )
    else:
        try:
            out = torch.nn.functional.scaled_dot_product_attention(
                qb, kb, vb, is_causal=causal, scale=scale,
            )
        except TypeError:
            out = torch.nn.functional.scaled_dot_product_attention(
                qb, kb, vb, is_causal=causal,
            )

    return out.transpose(1, 2).reshape(total_tokens, num_q_heads, head_dim)


# --- Detect flash_attn for native varlen support ---
try:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen
    _HAS_FLASH_ATTN = True
except ImportError:
    _flash_attn_varlen = None
    _HAS_FLASH_ATTN = False

_PACKED_ATTN_BACKEND_LOGGED = set()
_PACKED_ATTN_UNIFORM_DISABLED = set()
_PACKED_ATTN_UNIFORM_SUCCESS_SHAPES = set()


def _log_packed_attention_backend(mode: str, message: str) -> None:
    if mode in _PACKED_ATTN_BACKEND_LOGGED:
        return
    _PACKED_ATTN_BACKEND_LOGGED.add(mode)
    print(message)


def packed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = True,
    packed_meta: Optional[PackedAttentionMetadata] = None,
) -> torch.Tensor:
    """
    Packed attention for variable-length batches.

    Uses the best available backend for the requested mode:
    - flash_attn varlen on CUDA when available
    - Triton packed kernel on CUDA for non-causal varlen batches
    - PyTorch SDPA per sequence as a portable fallback
    - Triton packed kernel for causal prefill when SDPA is not preferred
    """
    head_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    num_seqs = cu_seqlens.shape[0] - 1
    mode = "causal" if causal else "noncausal"
    packed_meta = _reuse_packed_attention_metadata(cu_seqlens, head_dim, packed_meta)

    if _HAS_FLASH_ATTN and q.is_cuda:
        _log_packed_attention_backend(
            mode,
            f"  ⚡ Packed attention ({mode}): flash_attn varlen (native CUDA)",
        )
        packed_meta = _get_packed_attention_metadata(cu_seqlens, head_dim, packed_meta)
        max_seqlen = packed_meta.max_seqlen
        cu = packed_meta.cu_seqlens
        return _flash_attn_varlen(
            q, k, v,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            softmax_scale=scale, causal=causal,
        )

    if (
        causal
        and _HAS_TRITON
        and q.is_cuda
        and _env_bool("MEGAGEMM_PACKED_ATTN_PREFER_TRITON", default=False)
    ):
        packed_meta = _get_packed_attention_metadata(cu_seqlens, head_dim, packed_meta)
        triton_out = _triton_packed_attention(
            q,
            k,
            v,
            cu_seqlens,
            scale,
            causal=True,
            packed_meta=packed_meta,
        )
        if triton_out is not None:
            _log_packed_attention_backend(
                mode,
                "  Packed attention (causal): Triton packed kernel",
            )
            return triton_out

    if (
        causal
        and q.is_cuda
        and _env_bool("MEGAGEMM_PACKED_ATTN_UNIFORM_BATCH", default=True)
        and mode not in _PACKED_ATTN_UNIFORM_DISABLED
    ):
        uniform_seq_len = _uniform_packed_seq_len(
            cu_seqlens,
            total_tokens=int(q.shape[0]),
            num_seqs=int(num_seqs),
        )
        if uniform_seq_len is not None:
            uniform_key = (
                mode,
                q.device.type,
                q.device.index if q.device.index is not None else -1,
                str(q.dtype),
                int(num_seqs),
                int(q.shape[1]),
                int(uniform_seq_len),
                int(head_dim),
            )
            has_headroom = (
                uniform_key in _PACKED_ATTN_UNIFORM_SUCCESS_SHAPES
                or _uniform_packed_sdpa_has_headroom(
                    q,
                    num_seqs=int(num_seqs),
                    num_q_heads=int(q.shape[1]),
                    seq_len=int(uniform_seq_len),
                )
            )
            if has_headroom:
                _log_packed_attention_backend(
                    mode,
                    "  Packed attention "
                    f"({mode}): SDPA uniform-batch "
                    f"(PyTorch backend, gqa={_prefill_gqa_mode()})",
                )
                try:
                    out = _pytorch_uniform_packed_attention(
                        q,
                        k,
                        v,
                        num_seqs=int(num_seqs),
                        seq_len=int(uniform_seq_len),
                        scale=float(scale),
                        causal=True,
                    )
                    _PACKED_ATTN_UNIFORM_SUCCESS_SHAPES.add(uniform_key)
                    return out
                except torch.OutOfMemoryError:
                    _PACKED_ATTN_UNIFORM_DISABLED.add(mode)
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    _log_packed_attention_backend(
                        f"{mode}-uniform-oom",
                        "  Packed attention: uniform-batch OOM, falling back to SDPA per-seq",
                    )
            else:
                _log_packed_attention_backend(
                    f"{mode}-uniform-skip",
                    "  Packed attention: SDPA uniform-batch skipped (low VRAM headroom)",
                )

    if not causal and _HAS_TRITON and q.is_cuda:
        packed_meta = _get_packed_attention_metadata(cu_seqlens, head_dim, packed_meta)
        triton_out = _triton_packed_attention(
            q,
            k,
            v,
            cu_seqlens,
            scale,
            causal=False,
            packed_meta=packed_meta,
        )
        if triton_out is not None:
            _log_packed_attention_backend(
                mode,
                "  ⚡ Packed attention (noncausal): Triton packed kernel",
            )
            return triton_out

    if q.is_cuda and num_seqs <= 128:
        _log_packed_attention_backend(
            mode,
            f"  ⚡ Packed attention ({mode}): SDPA per-seq (PyTorch backend)",
        )
        return _pytorch_packed_attention(
            q,
            k,
            v,
            cu_seqlens,
            scale,
            causal,
            packed_meta=packed_meta,
        )

    if causal and _HAS_TRITON and q.is_cuda:
        packed_meta = _get_packed_attention_metadata(cu_seqlens, head_dim, packed_meta)
        triton_out = _triton_packed_attention(
            q,
            k,
            v,
            cu_seqlens,
            scale,
            causal=True,
            packed_meta=packed_meta,
        )
        if triton_out is not None:
            _log_packed_attention_backend(
                mode,
                "  ⚡ Packed attention (causal): Triton packed kernel",
            )
            return triton_out

    return _pytorch_packed_attention(
        q,
        k,
        v,
        cu_seqlens,
        scale,
        causal,
        packed_meta=packed_meta,
    )


def packed_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: Optional[float] = None,
    packed_meta: Optional[PackedAttentionMetadata] = None,
) -> torch.Tensor:
    """
    Packed prefill attention \u2014 zero-waste batched prefill.

    Priority: flash_attn (CUDA) > SDPA per-seq (PyTorch FlashAttn backend) > Triton
    """
    return packed_attention(
        q,
        k,
        v,
        cu_seqlens,
        scale=scale,
        causal=True,
        packed_meta=packed_meta,
    )
