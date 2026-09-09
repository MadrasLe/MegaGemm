"""
📋 Continuous Batching Scheduler for MegaGemm
----------------------------------------------
Iteration-level scheduler that processes multiple requests concurrently.

Each iteration:
1. Admit: batched prefill of N waiting requests (single forward pass)
2. Batch decode: all running requests decoded in ONE model call
3. Sample: next token per sequence
4. Evict: finished requests freed, results returned

Author: Gabriel Yogi
"""

import os
import torch
import time
import traceback
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Sequence, Union

from .kv_cache import BlockManager
from .sampling import sample_logits
from ..models.runtime_policy import policy_bool, policy_rows

try:
    import megagemm_decode_ops as _decode_native_ops
except ImportError:
    _decode_native_ops = None

__all__ = ['Request', 'RequestStatus', 'Scheduler']


def _get_batch_decode_burst() -> int:
    raw = os.environ.get("MEGAGEMM_MULTI_STEP_BURST_BATCH", "").strip()
    if not raw:
        raw = os.environ.get("MEGAGEMM_MULTI_STEP_BURST", "").strip()
    if not raw:
        raw = "8"
    return max(1, int(raw))


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


def _request_scheduler_env_signature() -> tuple:
    """Capture runtime knobs that can change Scheduler/model graph behavior."""
    return tuple(
        sorted(
            (name, value)
            for name, value in os.environ.items()
            if name.startswith("MEGAGEMM_")
        )
    )


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _cuda_context_is_poisoned(exc: BaseException) -> bool:
    markers = (
        "illegal memory access",
        "device-side assert",
        "unspecified launch failure",
        "cudaerrorillegaladdress",
    )
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = f"{type(current).__name__}: {current}".lower()
        if any(marker in message for marker in markers):
            return True
        current = current.__cause__ or current.__context__
    return False


_DECODE_GRAPH_ENV_KEY_PARTS = (
    # These env knobs change which paged-attention decode kernel/grid gets
    # captured. They must be part of the CUDA graph cache key; otherwise a sweep
    # that mutates an env var can silently replay the graph captured for an
    # earlier setting.
    "MEGAGEMM_PAGED_DECODE_SPLITS",
    "MEGAGEMM_PAGED_DECODE_TARGET_WARPS_PER_SM",
    "MEGAGEMM_PAGED_DECODE_TARGET_PROGRAMS",
    "MEGAGEMM_PAGED_DECODE_MAX_SPLITS",
    "MEGAGEMM_PAGED_DECODE_SPLIT_MIN_BLOCKS",
    "MEGAGEMM_PAGED_DECODE_WARPS",
    "MEGAGEMM_PAGED_DECODE_WARPS_H256",
    "MEGAGEMM_PAGED_DECODE_WARPS_H512",
    "MEGAGEMM_PAGED_DECODE_REDUCE_WARPS",
    "MEGAGEMM_PAGED_DECODE_BLOCK_UNROLL",
    "MEGAGEMM_PAGED_DECODE_GQA2",
    "MEGAGEMM_PAGED_DECODE_GQA2_SPLIT",
    "MEGAGEMM_PAGED_DECODE_GQA4_SPLIT",
    "MEGAGEMM_PAGED_DECODE_GQA8_SPLIT",
    "MEGAGEMM_PAGED_DECODE_GQA_GROUP",
    "MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_DECODE",
    "MEGAGEMM_QWEN3_MOE_FUSED_ROUTER",
    "MEGAGEMM_QWEN3_MOE_FUSED_ROUTER_MAX_ROWS",
    "MEGAGEMM_QWEN3_MOE_ROUTER_K_SPLITS",
    "MEGAGEMM_FUSED_RMSNORM_QKV_ALLOW_CUDA_GRAPHS",
    "MEGAGEMM_QWEN3_MOE_GROUPED_FUSED_GATE",
    "MEGAGEMM_QWEN3_MOE_GROUPED_DOT",
    "MEGAGEMM_QWEN3_MOE_GROUPED_DOT_ALLOW_CUDA_GRAPHS",
    "MEGAGEMM_QWEN3_MOE_TOKEN_ACCUM",
    "MEGAGEMM_QWEN3_MOE_TOKEN_ACCUM_MIN_ROWS",
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_DECODE",
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_DENSE_DECODE",
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_GENERAL_DECODE",
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_DECODE",
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST",
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT",
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_GATE_NUM_STAGES",
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES",
    "MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_DECODE",
    "MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_GATE_K_SPLITS",
    "MEGAGEMM_QWEN3_MOE_GROUPED_BLOCK_N",
    "MEGAGEMM_QWEN3_MOE_GROUPED_BLOCK_K",
    "MEGAGEMM_QWEN3_MOE_GROUPED_NUM_WARPS",
    "MEGAGEMM_QWEN3_MOE_GROUPED_NUM_STAGES",
    "MEGAGEMM_QWEN3_MOE_GROUPED_MAX_ASSIGNMENTS",
    "MEGAGEMM_FUSED_LM_HEAD_ARGMAX_DECODE",
    "MEGAGEMM_FUSED_RMSNORM_LM_HEAD_ARGMAX_DECODE",
    "MEGAGEMM_GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX",
    "MEGAGEMM_GEMMA4_FUSED_NEXT_ATTN_NORM_DECODE",
    "MEGAGEMM_DECODE_GRAPH_PERSISTENT_TOKEN_FEEDBACK",
    "MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID",
)


class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    """Tracks state for a single generation request."""
    request_id: int
    seq_id: int
    prompt_ids: List[int]           # tokenized prompt
    generated_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    status: RequestStatus = RequestStatus.WAITING
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    stop_token_ids: set = field(default_factory=set)

    # Timing
    t_start: float = 0.0
    t_prefill_done: float = 0.0
    t_end: float = 0.0

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    @property
    def num_generated(self) -> int:
        return len(self.generated_ids)

    @property
    def current_len(self) -> int:
        return self.prompt_len + self.num_generated

    @property
    def is_done(self) -> bool:
        if self.num_generated >= self.max_new_tokens:
            return True
        if self.generated_ids and self.generated_ids[-1] in self.stop_token_ids:
            return True
        return False


class Scheduler:
    """
    Continuous batching scheduler.

    Manages waiting/running queues and orchestrates iteration-level batching
    where prefill and decode happen concurrently.

    Usage:
        scheduler = Scheduler(model, block_manager, max_batch_size=8)
        scheduler.add_request(prompt_ids, max_new_tokens=100)
        scheduler.add_request(prompt_ids2, max_new_tokens=50)

        while scheduler.has_pending():
            completed = scheduler.step()
            for req in completed:
                print(req.generated_ids)
    """

    def __init__(
        self,
        model,
        block_manager: BlockManager,
        max_batch_size: int = 32,
        device: str = 'cuda',
        prefill_capture_hook: Optional[Callable[[Request, torch.Tensor], None]] = None,
        materialize_generated_tokens: bool = True,
    ):
        self.model = model
        self.block_manager = block_manager
        self.max_batch_size = max_batch_size
        self.device = device
        self._prefill_capture_hook = prefill_capture_hook
        self._materialize_generated_tokens = bool(materialize_generated_tokens)
        reuse_batches = policy_rows(
            model,
            "MEGAGEMM_REUSE_REQUEST_SCHEDULER",
            "reuse_request_scheduler_batches",
        )
        self._request_scheduler_reuse_enabled = policy_bool(
            model,
            "MEGAGEMM_REUSE_REQUEST_SCHEDULER",
            "reuse_request_scheduler",
            default=False,
        ) or bool(reuse_batches)
        self._request_scheduler_env_signature = _request_scheduler_env_signature()
        self._request_scheduler_reused = False
        self._request_scheduler_reuse_count = 0
        self._decode_skip_token_store = _env_bool(
            "MEGAGEMM_DECODE_SKIP_TOKEN_STORE", default=False
        )
        self._decode_multi_step_burst = _get_batch_decode_burst()
        # Select the regular one-token decode path independently from CUDA
        # Graphs.  Some model/GPU combinations are faster in the flat
        # decode_step path even when graph capture is intentionally disabled.
        self._decode_prefer_step = policy_bool(
            model,
            "MEGAGEMM_DECODE_PREFER_STEP",
            "decode_prefer_step",
            default=False,
        )
        self._decode_step_batches = 0
        self._decode_multi_step_batches = 0
        self._decode_graph_token_burst = _env_bool(
            "MEGAGEMM_DECODE_GRAPH_TOKEN_BURST", default=True,
        )
        self._decode_graph_persistent_token_feedback = policy_bool(
            model,
            "MEGAGEMM_DECODE_GRAPH_PERSISTENT_TOKEN_FEEDBACK",
            "decode_graph_persistent_token_feedback",
            default=False,
        )
        self._decode_native_graph_burst = bool(
            _env_bool("MEGAGEMM_NATIVE_DECODE_GRAPH_BURST", default=False)
            and _decode_native_ops is not None
            and hasattr(_decode_native_ops, "run_cuda_graph_token_burst")
        )
        self._decode_unrolled_graph_burst = _env_bool(
            "MEGAGEMM_DECODE_UNROLLED_GRAPH_BURST",
            default=False,
        )
        # CUDA-graph execution keeps the production multi-step model body,
        # including its LM-head policy. Runtime policy can scope it by batch.
        self._decode_graph_multi_step_body = policy_bool(
            model,
            "MEGAGEMM_DECODE_GRAPH_MULTI_STEP_BODY",
            "decode_graph_multi_step_body",
            default=False,
        )
        self._decode_graph_eager_control = _env_bool(
            "MEGAGEMM_DECODE_GRAPH_EAGER_CONTROL", default=False,
        )
        self._benchmark_forced_token_id = _env_int(
            "MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID", -1
        )
        if self._benchmark_forced_token_id >= 0:
            vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0))
            if not 0 <= self._benchmark_forced_token_id < vocab_size:
                raise ValueError(
                    "MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID is outside the model "
                    f"vocabulary: {self._benchmark_forced_token_id} >= {vocab_size}"
                )

        self._waiting: List[Request] = []
        self._running: Dict[int, Request] = {}  # seq_id -> Request
        self._completed: List[Request] = []

        self._req_counter = 0
        self._seq_counter = 0

        # Pre-allocated decode buffers (reused every step, zero allocation)
        self._decode_input_ids = torch.empty(
            max_batch_size, 1, dtype=torch.long, device=device,
        )
        self._decode_positions = torch.empty(
            max_batch_size, 1, dtype=torch.long, device=device,
        )
        self._decode_burst_tokens = torch.empty(
            max_batch_size,
            self._decode_multi_step_burst,
            dtype=torch.long,
            device=device,
        )
        pin_decode_host = bool(device == 'cuda' and torch.cuda.is_available())
        try:
            self._decode_input_ids_host = torch.empty(
                max_batch_size, dtype=torch.long, pin_memory=pin_decode_host,
            )
            self._decode_positions_host = torch.empty(
                max_batch_size, dtype=torch.long, pin_memory=pin_decode_host,
            )
            self._decode_next_tokens_host = torch.empty(
                max_batch_size, dtype=torch.long, pin_memory=pin_decode_host,
            )
            self._decode_burst_tokens_host = torch.empty(
                max_batch_size,
                self._decode_multi_step_burst,
                dtype=torch.long,
                pin_memory=pin_decode_host,
            )
            self._decode_host_buffers_pinned = pin_decode_host
        except RuntimeError:
            self._decode_input_ids_host = torch.empty(max_batch_size, dtype=torch.long)
            self._decode_positions_host = torch.empty(max_batch_size, dtype=torch.long)
            self._decode_next_tokens_host = torch.empty(
                max_batch_size, dtype=torch.long,
            )
            self._decode_burst_tokens_host = torch.empty(
                max_batch_size,
                self._decode_multi_step_burst,
                dtype=torch.long,
            )
            self._decode_host_buffers_pinned = False
        self._batch_changed = True  # track batch membership changes

        self._prefill_chunk_min = max(1, _env_int("MEGAGEMM_PREFILL_CHUNK_MIN", 8))
        self._prefill_chunk_max = max(
            self._prefill_chunk_min,
            _env_int("MEGAGEMM_PREFILL_CHUNK_MAX", 128),
        )
        self._prefill_chunk_reserve_mb = max(
            128,
            _env_int("MEGAGEMM_PREFILL_CHUNK_RESERVE_MB", 512),
        )
        self._prefill_chunk_safety = max(
            1.0,
            min(2.0, _env_float("MEGAGEMM_PREFILL_CHUNK_SAFETY", 1.5)),
        )
        self._prefill_chunk_empty_cache = _env_bool(
            "MEGAGEMM_PREFILL_CHUNK_EMPTY_CACHE", default=True,
        )
        self._prefill_max_batched_tokens = max(
            0,
            _env_int("MEGAGEMM_PREFILL_MAX_BATCHED_TOKENS", 0),
        )
        self._gemma4_deterministic_prefill_max_batched_tokens = max(
            0,
            _env_int(
                "MEGAGEMM_GEMMA4_DETERMINISTIC_PREFILL_MAX_BATCHED_TOKENS",
                16384,
            ),
        )
        self._prefill_pad_waste_threshold = max(
            0.0,
            min(0.50, _env_float("MEGAGEMM_PREFILL_PAD_WASTE_THRESHOLD", 0.03)),
        )
        self._prefill_prefer_padded = _env_bool(
            "MEGAGEMM_PREFILL_PREFER_PADDED", default=False,
        )
        self._prefill_static_buffers = _env_bool(
            "MEGAGEMM_PREFILL_STATIC_BUFFERS", default=True,
        )
        self._prefill_choice_log_count = 0
        self._prefill_packed_input_ids = None
        self._prefill_packed_cu_seqlens = None
        self._prefill_packed_lengths = None
        self._prefill_padded_input_ids = None
        self._prefill_padded_lengths = None

        # Profiling: separate prefill vs decode time
        self._prefill_time = 0.0
        self._decode_time = 0.0
        self._prefill_stage_timing_totals: Dict[str, float] = {}
        self._prefill_stage_timing_chunks = 0
        self._prefill_stage_total_tokens = 0
        self._prefill_stage_total_seqs = 0
        self._prefill_stage_max_len = 0
        self._prefill_last_chunk_plan = None

        # Optional decode CUDA Graphs.
        # Kept conservative on purpose: one stable graph per active batch
        # membership, invalidated as soon as the running set changes.
        self._decode_cuda_graph_policy_batches = policy_rows(
            model,
            "MEGAGEMM_DECODE_CUDA_GRAPHS",
            "decode_cuda_graph_batches",
        )
        self._decode_cuda_graphs = policy_bool(
            model,
            "MEGAGEMM_DECODE_CUDA_GRAPHS",
            "decode_cuda_graphs",
            default=False,
        )
        self._decode_cuda_graph_prefer_step = policy_bool(
            model,
            "MEGAGEMM_DECODE_CUDA_GRAPHS_PREFER_STEP",
            "decode_cuda_graph_prefer_step",
            default=False,
        )
        self._decode_cuda_graph_shape_cache = _env_bool(
            "MEGAGEMM_DECODE_CUDA_GRAPHS_SHAPE_CACHE",
            default=True,
        )
        self._decode_cuda_graph_shared_shape_cache = _env_bool(
            "MEGAGEMM_DECODE_CUDA_GRAPHS_SHARED_SHAPE_CACHE",
            default=False,
        )
        self._decode_cuda_graph_stable_max_blocks = _env_bool(
            "MEGAGEMM_DECODE_CUDA_GRAPHS_STABLE_MAX_BLOCKS",
            default=True,
        )
        self._decode_cuda_graph_min_batch = max(
            1,
            _env_int("MEGAGEMM_DECODE_CUDA_GRAPHS_MIN_BATCH", 1),
        )
        self._decode_cuda_graph_allow_qwen3_moe = _env_bool(
            "MEGAGEMM_DECODE_CUDA_GRAPHS_ALLOW_QWEN3_MOE",
            default=True,
        )
        self._decode_cuda_graph_log_limit = max(
            0,
            _env_int("MEGAGEMM_DECODE_CUDA_GRAPHS_LOG_LIMIT", 6),
        )
        self._decode_graph_state = None
        self._decode_graph_warm_key = None
        self._decode_graph_failed_key = None
        if self._decode_cuda_graph_shared_shape_cache:
            shape_graph_cache = getattr(block_manager, "_decode_graph_shape_cache", None)
            if shape_graph_cache is None:
                shape_graph_cache = {
                    "states": {},
                    "warm_keys": set(),
                    "failed_keys": set(),
                }
                try:
                    setattr(block_manager, "_decode_graph_shape_cache", shape_graph_cache)
                except Exception:
                    pass
            self._decode_graph_shape_states: Dict[tuple, dict] = shape_graph_cache.setdefault(
                "states", {}
            )
            self._decode_graph_shape_warm_keys = shape_graph_cache.setdefault("warm_keys", set())
            self._decode_graph_shape_failed_keys = shape_graph_cache.setdefault("failed_keys", set())
        else:
            self._decode_graph_shape_states: Dict[tuple, dict] = {}
            self._decode_graph_shape_warm_keys = set()
            self._decode_graph_shape_failed_keys = set()
        self._decode_graph_capture_count = 0
        self._decode_graph_replay_count = 0
        self._decode_graph_eager_warmups = 0
        self._decode_graph_failures = 0
        self._decode_graph_physical_rebinds = 0
        self._decode_graph_log_count = 0
        self._decode_graph_last_failure = ""
        self._decode_vectorized_input_updates = 0
        self._decode_batched_token_host_copies = 0
        self._decode_greedy_token_steps = 0
        self._decode_graph_token_bursts = 0
        self._decode_graph_token_burst_steps = 0
        self._decode_graph_token_feedback_copies = 0
        self._decode_graph_persistent_feedback_steps = 0
        self._decode_graph_chain_input_updates_skipped = 0
        self._decode_native_graph_bursts = 0
        self._decode_native_graph_burst_steps = 0
        self._decode_unrolled_graph_burst_captures = 0
        self._decode_unrolled_graph_burst_replays = 0
        self._decode_unrolled_graph_bursts = 0
        self._decode_unrolled_graph_burst_steps = 0
        self._decode_unrolled_graph_burst_failures = 0
        self._decode_unrolled_graph_last_failure = ""
        self._decode_graph_chain_started_keys = set()
        self._decode_graph_last_feedback_persistent = False
        self._prefill_cuda_graphs = _env_bool(
            "MEGAGEMM_PREFILL_CUDA_GRAPHS", default=False,
        )
        self._prefill_cuda_graph_min_reqs = max(
            1,
            _env_int("MEGAGEMM_PREFILL_CUDA_GRAPHS_MIN_REQS", 16),
        )
        self._prefill_cuda_graph_min_free_mb = max(
            0,
            _env_int("MEGAGEMM_PREFILL_CUDA_GRAPHS_MIN_FREE_MB", 512),
        )
        self._prefill_cuda_graph_log_limit = max(
            0,
            _env_int("MEGAGEMM_PREFILL_CUDA_GRAPHS_LOG_LIMIT", 6),
        )
        self._prefill_graph_log_count = 0

    def can_reuse_for_request(
        self,
        *,
        model,
        block_manager: BlockManager,
        max_batch_size: int,
        device: str,
    ) -> bool:
        """Return whether this idle Scheduler can safely own another request.

        Reuse is deliberately narrower than the cross-Scheduler graph cache.  The
        same Scheduler keeps ownership of its graph inputs and scratch buffers,
        while BlockManager's idle reset restores the capture-time physical block
        order.  Any runtime-knob change creates a fresh owner.
        """
        if not self._request_scheduler_reuse_enabled:
            return False
        if model is not self.model or block_manager is not self.block_manager:
            return False
        if int(max_batch_size) != int(self.max_batch_size):
            return False
        if str(device) != str(self.device):
            return False
        if self._waiting or self._running:
            return False
        if getattr(block_manager, "block_tables", None):
            return False
        if getattr(block_manager, "seq_lens", None):
            return False
        if getattr(block_manager, "_decode_metadata_override", None) is not None:
            return False
        return self._request_scheduler_env_signature == _request_scheduler_env_signature()

    def reset_for_request(
        self,
        *,
        prefill_capture_hook: Optional[Callable[[Request, torch.Tensor], None]],
        materialize_generated_tokens: bool,
    ) -> None:
        """Reset request-local state while retaining owned CUDA Graph storage."""
        if not self.can_reuse_for_request(
            model=self.model,
            block_manager=self.block_manager,
            max_batch_size=self.max_batch_size,
            device=self.device,
        ):
            raise RuntimeError("Scheduler is not idle or compatible for request reuse")

        self._waiting.clear()
        self._running.clear()
        self._completed.clear()
        # Reusing the same logical IDs keeps capture-time Python membership and
        # BlockManager's deterministic physical allocation contract identical.
        self._req_counter = 0
        self._seq_counter = 0
        self._prefill_capture_hook = prefill_capture_hook
        self._materialize_generated_tokens = bool(materialize_generated_tokens)
        self._batch_changed = True

        self._prefill_time = 0.0
        self._decode_time = 0.0
        self._prefill_stage_timing_totals.clear()
        self._prefill_stage_timing_chunks = 0
        self._prefill_stage_total_tokens = 0
        self._prefill_stage_total_seqs = 0
        self._prefill_stage_max_len = 0
        self._prefill_last_chunk_plan = None
        self._prefill_choice_log_count = 0
        self._prefill_graph_log_count = 0
        if hasattr(self, "_chunk_log_count"):
            self._chunk_log_count = 0

        # These counters describe the current request.  Shape states, graphs,
        # warm keys, and failed keys intentionally remain owned by this Scheduler.
        self._decode_graph_capture_count = 0
        self._decode_graph_replay_count = 0
        self._decode_graph_eager_warmups = 0
        self._decode_graph_failures = 0
        self._decode_graph_physical_rebinds = 0
        self._decode_graph_log_count = 0
        self._decode_graph_last_failure = ""
        self._decode_vectorized_input_updates = 0
        self._decode_batched_token_host_copies = 0
        self._decode_greedy_token_steps = 0
        self._decode_graph_token_bursts = 0
        self._decode_graph_token_burst_steps = 0
        self._decode_graph_token_feedback_copies = 0
        self._decode_graph_persistent_feedback_steps = 0
        self._decode_graph_chain_input_updates_skipped = 0
        self._decode_native_graph_bursts = 0
        self._decode_native_graph_burst_steps = 0
        self._decode_unrolled_graph_burst_captures = 0
        self._decode_unrolled_graph_burst_replays = 0
        self._decode_unrolled_graph_bursts = 0
        self._decode_unrolled_graph_burst_steps = 0
        self._decode_unrolled_graph_burst_failures = 0
        self._decode_unrolled_graph_last_failure = ""
        self._decode_graph_chain_started_keys.clear()
        self._decode_graph_last_feedback_persistent = False

        self._request_scheduler_reused = True
        self._request_scheduler_reuse_count += 1

    def _next_ids(self):
        self._req_counter += 1
        self._seq_counter += 1
        return self._req_counter, self._seq_counter

    def add_request(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        stop_token_ids: Optional[set] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> int:
        """Add a generation request to the waiting queue. Returns request_id."""
        req_id, seq_id = self._next_ids()
        req = Request(
            request_id=req_id,
            seq_id=seq_id,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_token_ids=stop_token_ids or set(),
            metadata=dict(metadata or {}),
            t_start=time.perf_counter(),
        )
        self._waiting.append(req)
        return req_id

    def has_pending(self) -> bool:
        """Returns True if there are waiting or running requests."""
        return len(self._waiting) > 0 or len(self._running) > 0

    def _abort_prefill_allocations(self, requests: List[Request]) -> None:
        """Release KV/cache state for requests whose prefill failed midway."""
        aborted = {int(req.seq_id) for req in requests}
        if not aborted:
            return
        for req in requests:
            self._running.pop(req.seq_id, None)
            try:
                self.block_manager.free_sequence(req.seq_id)
            except Exception:
                pass
        self._completed = [req for req in self._completed if int(req.seq_id) not in aborted]

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    def _prefers_scheduler_greedy_token_decode(self, num_seqs: int) -> bool:
        prefer = getattr(
            self.model,
            'prefers_scheduler_greedy_token_decode',
            None,
        )
        if not callable(prefer):
            return False
        try:
            return bool(prefer(int(num_seqs)))
        except Exception:
            return False

    def step(self) -> List[Request]:
        """
        Execute one scheduler iteration.

        1. Try to admit MULTIPLE waiting requests (batched prefill)
        2. Batch decode all running requests (multi-step when greedy)
        3. Sample tokens and check completion
        4. Return list of newly completed requests

        Returns:
            List of requests that finished in this step
        """
        newly_completed = []

        # --- PHASE 1: Batched prefill — admit N waiting requests ---
        prefill_completed_start = len(self._completed)
        batch = self._collect_prefill_batch()
        if batch:
            t0 = time.perf_counter()
            if len(batch) == 1:
                self._prefill_request(batch[0])
            else:
                self._prefill_batch(batch)
            self._prefill_time += time.perf_counter() - t0
            newly_completed.extend(self._completed[prefill_completed_start:])

        # --- PHASE 2: Batch decode all running requests ---
        if self._running:
            # Multi-step decode: run N steps on GPU when all seqs are greedy
            # and no pending prefills that need interleaving
            requests = list(self._running.values())
            all_greedy = all(req.temperature == 0.0 for req in requests)
            has_multi_step = hasattr(self.model, 'decode_multi_step')
            no_pending = len(self._waiting) == 0
            no_stop_tokens = all(not req.stop_token_ids for req in requests)
            max_remaining = min(
                req.max_new_tokens - req.num_generated for req in requests
            )

            t0 = time.perf_counter()
            prefer_graph_step = (
                self._decode_cuda_graphs
                and self._decode_cuda_graph_prefer_step
                and self._decode_graph_batch_allowed(len(requests))
            )
            prefer_decode_step = self._decode_prefer_step or prefer_graph_step
            use_graph_token_burst = bool(
                all_greedy
                and no_pending
                and no_stop_tokens
                and prefer_graph_step
                and self._decode_graph_token_burst
                and self._decode_multi_step_burst > 1
                and max_remaining > 1
                and self._prefers_scheduler_greedy_token_decode(len(requests))
            )
            if use_graph_token_burst:
                finished = self._decode_graph_token_burst_batch(
                    self._decode_multi_step_burst,
                )
            elif all_greedy and has_multi_step and no_pending and not prefer_decode_step:
                self._decode_multi_step_batches += 1
                finished = self._decode_multi_step_batch(self._decode_multi_step_burst)
            else:
                self._decode_step_batches += 1
                finished = self._decode_batch()
            self._decode_time += time.perf_counter() - t0
            newly_completed.extend(finished)
            self._completed.extend(finished)
        return newly_completed

    def _collect_prefill_batch(self) -> List[Request]:
        """Collect waiting requests that fit into GPU KV budget.

        Paged attention needs ALL blocks on GPU simultaneously.
        Automatically limits batch based on available GPU blocks.
        """
        available_slots = self.max_batch_size - len(self._running)
        if available_slots <= 0 or not self._waiting:
            return []

        bm = self.block_manager
        bs = bm.block_size
        gpu_cap = getattr(bm, 'num_gpu_blocks', getattr(bm, 'num_blocks', 10**9))

        # Count blocks already committed to running sequences
        committed = sum(len(bm.block_tables.get(s, [])) for s in self._running)

        batch = []
        for req in self._waiting[:available_slots]:
            total_tokens = req.prompt_len + req.max_new_tokens
            need = (total_tokens + bs - 1) // bs

            if not bm.can_allocate(total_tokens):
                break
            if committed + need > gpu_cap:
                break  # would overflow GPU

            batch.append(req)
            committed += need

        for req in batch:
            self._waiting.remove(req)

        return batch

    def _prefill_batch(self, requests: List[Request]):
        """
        TRUE batched prefill: process N prompts in sub-batches.

        Tries packed prefill (zero-waste) first, falls back to left-padded,
        then to sequential if model doesn't support either.
        """
        has_packed = hasattr(self.model, 'prefill_packed')
        has_padded = hasattr(self.model, 'prefill_batch')

        force_sequential = bool(
            getattr(self.model, "_force_sequential_prefill", False)
        )
        if (
            force_sequential
            or (not has_packed and not has_padded)
            or len(requests) <= 1
        ):
            for req in requests:
                self._prefill_request(req)
            return

        # Process in sub-batches to limit memory.
        # Plan chunks by batched tokens (with a per-request KV cost tax) instead
        # of a fixed number of requests. This aligns better with serving engines
        # like SGLang/vLLM, where admission is primarily token-budget based.
        chunk_plan = self._plan_prefill_chunks(requests)
        self._prefill_last_chunk_plan = {
            'strategy': str(chunk_plan.get('strategy') or ''),
            'total_prompt_tokens': int(
                chunk_plan.get('total_prompt_tokens', 0) or 0
            ),
            'num_chunks': int(len(chunk_plan.get('chunks') or ())),
            'max_requests': int(chunk_plan.get('max_requests', 0) or 0),
            'max_batched_tokens': (
                int(chunk_plan['max_batched_tokens'])
                if chunk_plan.get('max_batched_tokens')
                else None
            ),
            'deterministic_moe_token_cap': (
                int(chunk_plan['deterministic_moe_token_cap'])
                if chunk_plan.get('deterministic_moe_token_cap')
                else None
            ),
            'chunk_prompt_tokens': [
                int(meta.get('prompt_tokens', 0) or 0)
                for meta in (chunk_plan.get('chunk_meta') or ())
            ],
            'chunk_request_counts': [
                int(len(chunk)) for chunk in (chunk_plan.get('chunks') or ())
            ],
        }
        if not hasattr(self, '_chunk_log_count'):
            self._chunk_log_count = 0
        if self._chunk_log_count < 5:
            self._chunk_log_count += 1
            parts = [
                f"reqs={len(requests)}",
                f"prompt_tok={chunk_plan['total_prompt_tokens']}",
                f"chunks={len(chunk_plan['chunks'])}",
                f"req_cap={chunk_plan['max_requests']}",
            ]
            if chunk_plan.get('max_batched_tokens'):
                parts.append(f"token_cap={int(chunk_plan['max_batched_tokens'])}")
            if chunk_plan.get('deterministic_moe_token_cap'):
                parts.append(
                    "det_moe_cap="
                    f"{int(chunk_plan['deterministic_moe_token_cap'])}"
                )
            if chunk_plan.get('cost_budget_tokens'):
                parts.append(f"budget≈{int(chunk_plan['cost_budget_tokens'])} tok-eq")
            print(
                f"  📦 Prefill chunk_plan={chunk_plan['strategy']} "
                f"({', '.join(parts)})"
            )
        for chunk, chunk_meta in zip(chunk_plan['chunks'], chunk_plan['chunk_meta']):
            use_padded = self._should_use_padded_prefill(chunk, has_packed, has_padded)
            if self._prefill_choice_log_count < 5 and len(chunk) > 1:
                self._prefill_choice_log_count += 1
                pad_waste = self._prefill_pad_waste_ratio(chunk)
                mode = "padded" if use_padded else ("packed" if has_packed else "padded")
                extra = ""
                if chunk_plan.get('cost_budget_tokens') and chunk_meta.get('cost_tokens'):
                    extra = f", cost≈{int(chunk_meta['cost_tokens'])}"
                print(
                    f"  ⚙️  Prefill mode={mode} "
                    f"(pad_waste={pad_waste * 100:.1f}%, reqs={len(chunk)}, "
                    f"tok={chunk_meta['prompt_tokens']}{extra})"
                )
            if use_padded:
                self._prefill_chunk_padded(chunk)
            elif has_packed:
                self._prefill_chunk_packed(chunk)
            else:
                self._prefill_chunk_padded(chunk)

    def _estimate_prefill_request_cost_tokens(
        self,
        req: Request,
        bytes_per_token: int,
    ) -> float:
        """Approximate one request's VRAM pressure as token-equivalent units."""
        if bytes_per_token <= 0:
            return float(req.prompt_len)
        bm = self.block_manager
        block_size = max(1, int(getattr(bm, 'block_size', 16) or 16))
        total_tokens = req.prompt_len + req.max_new_tokens
        num_blocks = max(1, (total_tokens + block_size - 1) // block_size)
        bytes_per_block = int(getattr(bm, 'bytes_per_block', 0) or 0)
        if bytes_per_block <= 0:
            bytes_per_block = 20 * 1024 * 1024
        kv_bytes = bytes_per_block * num_blocks
        kv_token_equiv = kv_bytes / float(bytes_per_token)
        return float(req.prompt_len) + kv_token_equiv

    def _gemma4_deterministic_prefill_token_cap(self) -> int:
        """Bound exact A4B prefill temporaries without changing other models."""
        cap = int(
            getattr(
                self,
                "_gemma4_deterministic_prefill_max_batched_tokens",
                0,
            )
            or 0
        )
        if cap <= 0 or not torch.are_deterministic_algorithms_enabled():
            return 0

        config = getattr(self.model, "config", None)
        embed_tokens = getattr(self.model, "embed_tokens", None)
        weight = getattr(embed_tokens, "weight", None)
        dtype = getattr(weight, "dtype", None)
        is_a4b = bool(
            str(getattr(config, "model_type", "")) == "gemma4_text"
            and bool(getattr(config, "enable_moe_block", False))
            and int(getattr(config, "hidden_size", 0) or 0) == 2816
            and int(getattr(config, "num_hidden_layers", 0) or 0) == 30
            and int(getattr(config, "num_experts", 0) or 0) == 128
            and int(getattr(config, "num_experts_per_tok", 0) or 0) == 8
            and int(getattr(config, "moe_intermediate_size", 0) or 0) == 704
            and dtype == torch.bfloat16
        )
        return cap if is_a4b else 0

    def _estimate_prefill_chunk_budget(self, requests: List[Request]) -> dict:
        legacy_chunk = os.environ.get("MEGAGEMM_PREFILL_CHUNK", "").strip()
        if legacy_chunk:
            try:
                max_requests = max(
                    self._prefill_chunk_min,
                    min(self._prefill_chunk_max, int(legacy_chunk)),
                )
            except Exception:
                max_requests = self._prefill_chunk_max
            return {
                'strategy': 'legacy_requests',
                'max_requests': max(1, min(len(requests), max_requests)),
                'max_batched_tokens': (
                    self._prefill_max_batched_tokens
                    if self._prefill_max_batched_tokens > 0
                    else None
                ),
                'cost_budget_tokens': None,
                'bytes_per_token': 0,
            }

        if not torch.cuda.is_available():
            return {
                'strategy': 'batched_tokens',
                'max_requests': max(1, min(len(requests), self._prefill_chunk_max)),
                'max_batched_tokens': (
                    self._prefill_max_batched_tokens
                    if self._prefill_max_batched_tokens > 0
                    else None
                ),
                'cost_budget_tokens': None,
                'bytes_per_token': 0,
            }

        try:
            if self._prefill_chunk_empty_cache:
                torch.cuda.empty_cache()
            free_bytes = torch.cuda.mem_get_info()[0]
            cfg = getattr(self.model, "config", None)
            hidden = int(
                getattr(cfg, "hidden_size", getattr(self.model, "hidden_size", 3584))
            )
            intermediate = int(
                getattr(
                    cfg,
                    "intermediate_size",
                    getattr(self.model, "intermediate_size", int(hidden * 5.3)),
                )
            )
            dtype = getattr(self.model.embed_tokens.weight, "dtype", torch.float16)
            dtype_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4
            mlp_ratio = max(1.0, intermediate / max(1, hidden))
            bytes_per_token = max(
                1,
                int(hidden * mlp_ratio * 2 * dtype_bytes * self._prefill_chunk_safety),
            )
            reserve = self._prefill_chunk_reserve_mb * 1024 * 1024
            budget_bytes = max(0, free_bytes - reserve)
            cost_budget_tokens = int(budget_bytes / bytes_per_token)
            if requests:
                cost_budget_tokens = max(
                    max(req.prompt_len for req in requests),
                    cost_budget_tokens,
                )
            return {
                'strategy': 'batched_tokens',
                'max_requests': max(1, min(len(requests), self._prefill_chunk_max)),
                'max_batched_tokens': (
                    self._prefill_max_batched_tokens
                    if self._prefill_max_batched_tokens > 0
                    else None
                ),
                'cost_budget_tokens': cost_budget_tokens if cost_budget_tokens > 0 else None,
                'bytes_per_token': bytes_per_token,
            }
        except Exception:
            return {
                'strategy': 'batched_tokens',
                'max_requests': max(1, min(len(requests), self._prefill_chunk_max)),
                'max_batched_tokens': (
                    self._prefill_max_batched_tokens
                    if self._prefill_max_batched_tokens > 0
                    else None
                ),
                'cost_budget_tokens': None,
                'bytes_per_token': 0,
            }

    def _plan_prefill_chunks(self, requests: List[Request]) -> dict:
        plan = self._estimate_prefill_chunk_budget(requests)
        max_requests = max(1, int(plan.get('max_requests', len(requests)) or len(requests)))
        prompt_budget = int(plan.get('max_batched_tokens') or 0)
        deterministic_moe_token_cap = self._gemma4_deterministic_prefill_token_cap()
        if deterministic_moe_token_cap > 0:
            prompt_budget = (
                min(prompt_budget, deterministic_moe_token_cap)
                if prompt_budget > 0
                else deterministic_moe_token_cap
            )
            plan['max_batched_tokens'] = prompt_budget
            plan['deterministic_moe_token_cap'] = deterministic_moe_token_cap
        cost_budget = float(plan.get('cost_budget_tokens') or 0.0)
        bytes_per_token = int(plan.get('bytes_per_token') or 0)

        chunks: List[List[Request]] = []
        chunk_meta = []
        idx = 0
        while idx < len(requests):
            chunk: List[Request] = []
            prompt_tokens = 0
            cost_tokens = 0.0
            while idx < len(requests) and len(chunk) < max_requests:
                req = requests[idx]
                req_prompt_tokens = req.prompt_len
                req_cost_tokens = self._estimate_prefill_request_cost_tokens(
                    req, bytes_per_token,
                )
                would_exceed_prompt = (
                    prompt_budget > 0
                    and len(chunk) > 0
                    and (prompt_tokens + req_prompt_tokens) > prompt_budget
                )
                would_exceed_cost = (
                    cost_budget > 0
                    and len(chunk) > 0
                    and (cost_tokens + req_cost_tokens) > cost_budget
                )
                if would_exceed_prompt or would_exceed_cost:
                    break
                chunk.append(req)
                prompt_tokens += req_prompt_tokens
                cost_tokens += req_cost_tokens
                idx += 1
            if not chunk:
                req = requests[idx]
                chunk = [req]
                prompt_tokens = req.prompt_len
                cost_tokens = self._estimate_prefill_request_cost_tokens(
                    req, bytes_per_token,
                )
                idx += 1
            chunks.append(chunk)
            chunk_meta.append({
                'prompt_tokens': int(prompt_tokens),
                'cost_tokens': float(cost_tokens),
            })

        plan['chunks'] = chunks
        plan['chunk_meta'] = chunk_meta
        plan['total_prompt_tokens'] = sum(req.prompt_len for req in requests)
        return plan

    def _prefill_pad_waste_ratio(self, requests: List[Request]) -> float:
        if not requests:
            return 0.0
        max_len = max(req.prompt_len for req in requests)
        total_slots = max_len * len(requests)
        if total_slots <= 0:
            return 0.0
        real_tokens = sum(req.prompt_len for req in requests)
        return max(0.0, 1.0 - (real_tokens / total_slots))

    def _should_use_padded_prefill(
        self,
        requests: List[Request],
        has_packed: bool,
        has_padded: bool,
    ) -> bool:
        if not has_padded:
            return False
        if not has_packed:
            return True
        # Gemma 4's packed entry point already converts back to padded tensors
        # to preserve sliding-attention sequence boundaries. Select that native
        # path here and avoid an unnecessary pack, GPU scalar sync, and unpack.
        config = getattr(self.model, "config", None)
        if str(getattr(config, "model_type", "")) == "gemma4_text":
            return True
        if not self._prefill_prefer_padded:
            return False
        has_linear_layers = False
        has_linear_fn = getattr(self.model, "_has_linear_layers", None)
        if callable(has_linear_fn):
            try:
                has_linear_layers = bool(has_linear_fn())
            except Exception:
                has_linear_layers = False
        if has_linear_layers:
            return False
        return self._prefill_pad_waste_ratio(requests) <= self._prefill_pad_waste_threshold

    def _get_packed_prefill_buffers(self, total_tokens: int, num_seqs: int):
        if not self._prefill_static_buffers:
            return (
                torch.empty((1, total_tokens), dtype=torch.long, device=self.device),
                torch.empty((num_seqs + 1,), dtype=torch.int32, device=self.device),
                torch.empty((num_seqs,), dtype=torch.long, device=self.device),
            )
        if (
            self._prefill_packed_input_ids is None
            or self._prefill_packed_input_ids.shape[1] < total_tokens
        ):
            self._prefill_packed_input_ids = torch.empty(
                (1, total_tokens), dtype=torch.long, device=self.device,
            )
        if (
            self._prefill_packed_cu_seqlens is None
            or self._prefill_packed_cu_seqlens.shape[0] < num_seqs + 1
        ):
            self._prefill_packed_cu_seqlens = torch.empty(
                (num_seqs + 1,), dtype=torch.int32, device=self.device,
            )
        if (
            self._prefill_packed_lengths is None
            or self._prefill_packed_lengths.shape[0] < num_seqs
        ):
            self._prefill_packed_lengths = torch.empty(
                (num_seqs,), dtype=torch.long, device=self.device,
            )
        return (
            self._prefill_packed_input_ids[:, :total_tokens],
            self._prefill_packed_cu_seqlens[:num_seqs + 1],
            self._prefill_packed_lengths[:num_seqs],
        )

    def _get_padded_prefill_buffers(self, num_seqs: int, max_len: int):
        if not self._prefill_static_buffers:
            return (
                torch.zeros((num_seqs, max_len), dtype=torch.long, device=self.device),
                torch.empty((num_seqs,), dtype=torch.long, device=self.device),
            )
        if (
            self._prefill_padded_input_ids is None
            or self._prefill_padded_input_ids.shape[0] < num_seqs
            or self._prefill_padded_input_ids.shape[1] < max_len
        ):
            self._prefill_padded_input_ids = torch.empty(
                (num_seqs, max_len), dtype=torch.long, device=self.device,
            )
        if (
            self._prefill_padded_lengths is None
            or self._prefill_padded_lengths.shape[0] < num_seqs
        ):
            self._prefill_padded_lengths = torch.empty(
                (num_seqs,), dtype=torch.long, device=self.device,
            )
        input_ids = self._prefill_padded_input_ids[:num_seqs, :max_len]
        input_ids.zero_()
        return input_ids, self._prefill_padded_lengths[:num_seqs]

    def _record_prefill_stage_timing(self):
        getter = getattr(self.model, "get_last_prefill_timing", None)
        if not callable(getter):
            return
        summary = getter()
        if not summary:
            return
        self._prefill_stage_timing_chunks += 1
        self._prefill_stage_total_tokens += int(summary.get("total_tokens", 0) or 0)
        self._prefill_stage_total_seqs += int(summary.get("num_seqs", 0) or 0)
        self._prefill_stage_max_len = max(
            self._prefill_stage_max_len,
            int(summary.get("max_len", 0) or 0),
        )
        for key, value in summary.items():
            if not key.endswith("_ms"):
                continue
            try:
                self._prefill_stage_timing_totals[key] = (
                    self._prefill_stage_timing_totals.get(key, 0.0) + float(value)
                )
            except Exception:
                continue

    def _maybe_capture_prefill(self, req: Request, pending_logits: torch.Tensor) -> None:
        hook = self._prefill_capture_hook
        if hook is None:
            return
        hook(req, pending_logits)

    def _log_prefill_graph(self, message: str):
        if self._prefill_graph_log_count >= self._prefill_cuda_graph_log_limit:
            return
        self._prefill_graph_log_count += 1
        print(f"  CUDA Graph prefill: {message}")

    def _get_prefill_graph_store(self):
        getter = getattr(self.model, "get_prefill_cuda_graph_store", None)
        if callable(getter):
            try:
                return getter(self.block_manager)
            except Exception:
                return None
        return None

    def _prefill_graph_is_eligible(
        self,
        requests: List[Request],
        *,
        padded: bool = False,
    ) -> bool:
        if not self._prefill_cuda_graphs:
            return False
        if self.device != 'cuda' or not torch.cuda.is_available():
            return False
        model_eligible = False
        checker = getattr(self.model, "prefill_cuda_graph_eligible", None)
        if callable(checker):
            try:
                model_eligible = bool(
                    checker(
                        num_seqs=len(requests),
                        total_tokens=sum(int(req.prompt_len) for req in requests),
                        dtype=self.model.embed_tokens.weight.dtype,
                        device_type="cuda",
                        device_name=torch.cuda.get_device_name(),
                    )
                )
            except Exception:
                model_eligible = False
        if padded and not model_eligible:
            return False
        if len(requests) < self._prefill_cuda_graph_min_reqs and not model_eligible:
            return False
        if type(self.block_manager) is not BlockManager:
            return False
        if getattr(self.model, "_offloader", None) is not None:
            return False
        if not bool(getattr(self.model, "_all_full_attention", False)) and not model_eligible:
            return False
        has_linear_fn = getattr(self.model, "_has_linear_layers", None)
        if callable(has_linear_fn):
            try:
                if bool(has_linear_fn()):
                    return False
            except Exception:
                return False
        if os.environ.get("MEGAGEMM_PREFILL_TIMING", "").strip().lower() in {"1", "true", "yes", "on"}:
            return False
        if os.environ.get("MEGAGEMM_PROFILE_PREFILL", "").strip() == "1":
            return False
        graph_method = "prefill_batch_graph" if padded else "prefill_packed_graph"
        if not hasattr(self.model, graph_method):
            return False
        if not hasattr(self.block_manager, "compute_kv_mapping"):
            return False
        return True

    def _prefill_graph_key(self, num_seqs: int, total_tokens: int):
        # Exact-shape buckets first. Safer than padded token buckets and still
        # useful for repeated workloads (warmup runs, recurring serving shapes).
        return (int(num_seqs), int(total_tokens))

    def _capture_prefill_graph(
        self,
        store: dict,
        key,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        kv_phys: torch.Tensor,
        kv_offs: torch.Tensor,
    ):
        graph_input_ids = torch.empty_like(input_ids)
        graph_cu_seqlens = torch.empty_like(cu_seqlens)
        graph_kv_phys = torch.empty_like(kv_phys)
        graph_kv_offs = torch.empty_like(kv_offs)
        graph_input_ids.copy_(input_ids)
        graph_cu_seqlens.copy_(cu_seqlens)
        graph_kv_phys.copy_(kv_phys)
        graph_kv_offs.copy_(kv_offs)

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            logits = self.model.prefill_packed_graph(
                graph_input_ids,
                graph_cu_seqlens,
                self.block_manager,
                graph_kv_phys,
                graph_kv_offs,
            )

        # CUDA Graph capture records the work; replay once to populate the
        # static logits and KV destinations for this request.
        graph.replay()

        state = {
            'kind': 'packed',
            'graph': graph,
            'input_ids': graph_input_ids,
            'cu_seqlens': graph_cu_seqlens,
            'kv_phys': graph_kv_phys,
            'kv_offs': graph_kv_offs,
            'logits': logits,
        }
        store['buckets'][key] = state
        store['captures'] = int(store.get('captures', 0)) + 1
        store['capture_replays'] = int(store.get('capture_replays', 0)) + 1
        self._log_prefill_graph(
            f"captured bucket=reqs{key[0]} tok{key[1]}",
        )
        return state

    def _capture_prefill_batch_graph(
        self,
        store: dict,
        key,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        kv_phys: torch.Tensor,
        kv_offs: torch.Tensor,
    ):
        graph_input_ids = torch.empty_like(input_ids)
        graph_cu_seqlens = torch.empty_like(cu_seqlens)
        graph_kv_phys = torch.empty_like(kv_phys)
        graph_kv_offs = torch.empty_like(kv_offs)
        graph_input_ids.copy_(input_ids)
        graph_cu_seqlens.copy_(cu_seqlens)
        graph_kv_phys.copy_(kv_phys)
        graph_kv_offs.copy_(kv_offs)

        workspace_refs = ()
        prepare_workspace = getattr(
            self.model,
            "prepare_prefill_cuda_graph_workspace",
            None,
        )
        if callable(prepare_workspace):
            workspace_refs = tuple(
                prepare_workspace(
                    total_tokens=int(input_ids.numel()),
                    device=input_ids.device,
                )
            )

        # Warm the exact body that will be captured. The regular scheduler
        # warmup uses prefill_batch(), while this path uses the graph-specific
        # deferred-KV implementation. Let its Triton/cuBLAS kernels, shape
        # caches, and stable output allocations settle before CUDA capture.
        expected_layers = len(getattr(self.model, "layers", ()))
        warm_deferred_ptrs = None
        for _ in range(2):
            warm_result = self.model.prefill_batch_graph(
                graph_input_ids,
                graph_cu_seqlens,
                self.block_manager,
                graph_kv_phys,
                graph_kv_offs,
                defer_kv_writes=True,
            )
            if not isinstance(warm_result, tuple) or len(warm_result) != 2:
                raise RuntimeError(
                    "Gemma4 padded prefill graph warmup did not return deferred K/V outputs"
                )
            warm_deferred_kv = tuple(warm_result[1])
            if not warm_deferred_kv or (
                expected_layers > 0 and len(warm_deferred_kv) != expected_layers
            ):
                raise RuntimeError(
                    "Gemma4 padded prefill graph warmup returned "
                    f"{len(warm_deferred_kv)}/{expected_layers} deferred K/V layers"
                )
            current_ptrs = tuple(
                (int(layer_idx), int(k_cache.data_ptr()), int(v_cache.data_ptr()))
                for layer_idx, k_cache, v_cache in warm_deferred_kv
            )
            if warm_deferred_ptrs is None:
                warm_deferred_ptrs = current_ptrs
            elif current_ptrs != warm_deferred_ptrs:
                raise RuntimeError(
                    "Gemma4 padded prefill graph warmup returned unstable "
                    "deferred K/V storage"
                )
            torch.cuda.synchronize()
            del warm_result, warm_deferred_kv
        store['capture_body_warmups'] = int(
            store.get('capture_body_warmups', 0)
        ) + 2

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            graph_result = self.model.prefill_batch_graph(
                graph_input_ids,
                graph_cu_seqlens,
                self.block_manager,
                graph_kv_phys,
                graph_kv_offs,
                defer_kv_writes=True,
            )

        if not isinstance(graph_result, tuple) or len(graph_result) != 2:
            raise RuntimeError(
                "Gemma4 padded prefill graph did not return deferred K/V outputs"
            )
        logits, deferred_kv = graph_result
        deferred_kv = tuple(deferred_kv)
        if not deferred_kv or (
            expected_layers > 0 and len(deferred_kv) != expected_layers
        ):
            raise RuntimeError(
                "Gemma4 padded prefill graph returned "
                f"{len(deferred_kv)}/{expected_layers} deferred K/V layers"
            )
        graph_deferred_ptrs = tuple(
            (int(layer_idx), int(k_cache.data_ptr()), int(v_cache.data_ptr()))
            for layer_idx, k_cache, v_cache in deferred_kv
        )
        if graph_deferred_ptrs != warm_deferred_ptrs:
            raise RuntimeError(
                "Gemma4 padded prefill graph capture changed deferred K/V storage"
            )

        graph.replay()
        # Validate model compute before touching the long-lived paged cache, so
        # an asynchronous failure can be attributed to the graph itself.
        try:
            torch.cuda.synchronize()
        except Exception as exc:
            raise RuntimeError(
                "Gemma4 padded prefill compute-graph replay failed before "
                "external K/V writes"
            ) from exc
        self._write_deferred_prefill_batch_kv(
            deferred_kv,
            graph_cu_seqlens,
            graph_kv_phys,
            graph_kv_offs,
        )
        try:
            torch.cuda.synchronize()
        except Exception as exc:
            raise RuntimeError(
                "Gemma4 padded prefill external K/V write validation failed"
            ) from exc
        state = {
            'kind': 'padded',
            'graph': graph,
            'input_ids': graph_input_ids,
            'cu_seqlens': graph_cu_seqlens,
            'kv_phys': graph_kv_phys,
            'kv_offs': graph_kv_offs,
            'logits': logits,
            'deferred_kv': deferred_kv,
            'deferred_kv_storage_stable': True,
            'kv_write_mode': 'external_after_replay',
            'workspace_refs': workspace_refs,
        }
        store['buckets'][key] = state
        store['captures'] = int(store.get('captures', 0)) + 1
        store['capture_replays'] = int(store.get('capture_replays', 0)) + 1
        store['external_kv_write_replays'] = int(
            store.get('external_kv_write_replays', 0)
        ) + 1
        self._log_prefill_graph(
            f"captured padded bucket=reqs{key[0]} tok{key[1]} "
            f"workspace={sum(t.numel() * t.element_size() for t in workspace_refs) / (1024 ** 2):.0f}MB",
        )
        return state

    def _write_deferred_prefill_batch_kv(
        self,
        deferred_kv,
        cu_seqlens: torch.Tensor,
        kv_phys: torch.Tensor,
        kv_offs: torch.Tensor,
    ) -> None:
        kv_mapping = (kv_phys, kv_offs)
        for layer_idx, k_cache, v_cache in deferred_kv:
            self.block_manager.write_kv_prefill_packed(
                [],
                int(layer_idx),
                k_cache,
                v_cache,
                cu_seqlens,
                kv_mapping=kv_mapping,
            )

    def _advance_prefill_seq_lens(
        self,
        seq_ids: List[int],
        lengths: Union[Sequence[int], torch.Tensor],
    ):
        lengths_cpu = lengths.tolist() if torch.is_tensor(lengths) else lengths
        for i, seq_id in enumerate(seq_ids):
            self.block_manager.advance_seq_len(seq_id, int(lengths_cpu[i]))

    def _prefill_graph_has_headroom(self, store: dict, key) -> bool:
        min_free_mb = int(self._prefill_cuda_graph_min_free_mb)
        if min_free_mb <= 0 or not torch.cuda.is_available():
            return True
        try:
            free_bytes = torch.cuda.mem_get_info()[0]
        except Exception:
            return True
        needed_bytes = min_free_mb * 1024 * 1024
        if free_bytes >= needed_bytes:
            return True
        store['skips'] = int(store.get('skips', 0)) + 1
        store['last_failure'] = (
            f"skip_low_vram free={free_bytes / (1024 * 1024):.0f}MB "
            f"< min_free={min_free_mb}MB for bucket=reqs{key[0]} tok{key[1]}"
        )
        self._log_prefill_graph(
            f"skip bucket=reqs{key[0]} tok{key[1]} due to low free VRAM",
        )
        return False

    def _run_prefill_packed_graph_or_eager(
        self,
        requests: List[Request],
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        lengths: torch.Tensor,
        seq_ids: List[int],
    ) -> torch.Tensor:
        if not self._prefill_graph_is_eligible(requests):
            logits = self.model.prefill_packed(
                input_ids, cu_seqlens, lengths, self.block_manager, seq_ids,
            )
            self._record_prefill_stage_timing()
            return logits

        store = self._get_prefill_graph_store()
        if store is None:
            logits = self.model.prefill_packed(
                input_ids, cu_seqlens, lengths, self.block_manager, seq_ids,
            )
            self._record_prefill_stage_timing()
            return logits

        total_tokens = int(input_ids.shape[1])
        key = self._prefill_graph_key(len(seq_ids), total_tokens)
        kv_phys, kv_offs = self.block_manager.compute_kv_mapping(
            seq_ids, cu_seqlens, input_ids.device,
        )

        state = store.get('buckets', {}).get(key)
        if state is not None:
            state['input_ids'].copy_(input_ids)
            state['cu_seqlens'].copy_(cu_seqlens)
            state['kv_phys'].copy_(kv_phys)
            state['kv_offs'].copy_(kv_offs)
            state['graph'].replay()
            store['replays'] = int(store.get('replays', 0)) + 1
            self._advance_prefill_seq_lens(seq_ids, lengths)
            return state['logits']

        failed_keys = store.setdefault('failed_keys', {})
        if failed_keys.get(key):
            logits = self.model.prefill_packed(
                input_ids, cu_seqlens, lengths, self.block_manager, seq_ids,
            )
            self._record_prefill_stage_timing()
            return logits

        warm_keys = store.setdefault('warm_keys', set())
        if key not in warm_keys:
            warm_keys.add(key)
            store['warmups'] = int(store.get('warmups', 0)) + 1
            self._log_prefill_graph(
                f"warmup bucket=reqs{key[0]} tok{key[1]}",
            )
            logits = self.model.prefill_packed(
                input_ids, cu_seqlens, lengths, self.block_manager, seq_ids,
            )
            self._record_prefill_stage_timing()
            return logits

        if not self._prefill_graph_has_headroom(store, key):
            failed_keys[key] = "low_vram"
            logits = self.model.prefill_packed(
                input_ids, cu_seqlens, lengths, self.block_manager, seq_ids,
            )
            self._record_prefill_stage_timing()
            return logits

        try:
            state = self._capture_prefill_graph(
                store, key, input_ids, cu_seqlens, kv_phys, kv_offs,
            )
            self._advance_prefill_seq_lens(seq_ids, lengths)
            return state['logits']
        except Exception as exc:
            failed_keys[key] = True
            store['failures'] = int(store.get('failures', 0)) + 1
            store['last_failure'] = str(exc)
            self._log_prefill_graph(
                f"capture failed for bucket=reqs{key[0]} tok{key[1]}",
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            logits = self.model.prefill_packed(
                input_ids, cu_seqlens, lengths, self.block_manager, seq_ids,
            )
            self._record_prefill_stage_timing()
            return logits

    def _run_prefill_batch_graph_or_eager(
        self,
        requests: List[Request],
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        seq_ids: List[int],
        prompt_lengths: Sequence[int],
    ) -> torch.Tensor:
        def run_eager() -> torch.Tensor:
            logits = self.model.prefill_batch(
                input_ids,
                lengths,
                self.block_manager,
                seq_ids,
                prompt_lengths_cpu=prompt_lengths,
            )
            return logits

        num_seqs, max_len = input_ids.shape
        if (
            not prompt_lengths
            or any(int(length) != int(max_len) for length in prompt_lengths)
            or not self._prefill_graph_is_eligible(requests, padded=True)
        ):
            return run_eager()

        store = self._get_prefill_graph_store()
        if store is None:
            return run_eager()

        total_tokens = int(num_seqs) * int(max_len)
        key = self._prefill_graph_key(num_seqs, total_tokens)
        cu_seqlens = (
            torch.arange(
                int(num_seqs) + 1,
                dtype=torch.int32,
                device=input_ids.device,
            )
            * int(max_len)
        )
        kv_phys, kv_offs = self.block_manager.compute_kv_mapping(
            seq_ids,
            cu_seqlens,
            input_ids.device,
            seq_lengths=[int(length) for length in prompt_lengths],
        )

        state = store.get('buckets', {}).get(key)
        if state is not None:
            if state.get('kind') != 'padded':
                return run_eager()
            state['input_ids'].copy_(input_ids)
            state['cu_seqlens'].copy_(cu_seqlens)
            state['kv_phys'].copy_(kv_phys)
            state['kv_offs'].copy_(kv_offs)
            state['graph'].replay()
            self._write_deferred_prefill_batch_kv(
                state['deferred_kv'],
                state['cu_seqlens'],
                state['kv_phys'],
                state['kv_offs'],
            )
            store['replays'] = int(store.get('replays', 0)) + 1
            store['external_kv_write_replays'] = int(
                store.get('external_kv_write_replays', 0)
            ) + 1
            self._advance_prefill_seq_lens(seq_ids, prompt_lengths)
            return state['logits']

        failed_keys = store.setdefault('failed_keys', {})
        if failed_keys.get(key):
            return run_eager()

        warm_keys = store.setdefault('warm_keys', set())
        if key not in warm_keys:
            warm_keys.add(key)
            store['warmups'] = int(store.get('warmups', 0)) + 1
            self._log_prefill_graph(
                f"warmup padded bucket=reqs{key[0]} tok{key[1]}",
            )
            return run_eager()

        if not self._prefill_graph_has_headroom(store, key):
            failed_keys[key] = "low_vram"
            return run_eager()

        try:
            state = self._capture_prefill_batch_graph(
                store,
                key,
                input_ids,
                cu_seqlens,
                kv_phys,
                kv_offs,
            )
            self._advance_prefill_seq_lens(seq_ids, prompt_lengths)
            return state['logits']
        except Exception as exc:
            failed_keys[key] = True
            store['failures'] = int(store.get('failures', 0)) + 1
            store['last_failure'] = str(exc)
            self._log_prefill_graph(
                f"capture failed for padded bucket=reqs{key[0]} tok{key[1]}",
            )
            if _cuda_context_is_poisoned(exc):
                raise RuntimeError(
                    "Gemma4 padded prefill CUDA graph poisoned the CUDA context; "
                    "eager fallback is unsafe in this process"
                ) from exc
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return run_eager()

    def _prefill_chunk_packed(self, requests: List[Request]):
        """Prefill a chunk using PACKED sequences (zero-waste, no padding)."""
        # 1. Allocate blocks for ALL requests in chunk
        try:
            for req in requests:
                total_tokens = req.prompt_len + req.max_new_tokens
                self.block_manager.allocate_sequence(req.seq_id, total_tokens)
        except Exception:
            self._abort_prefill_allocations(requests)
            raise

        seq_ids = [req.seq_id for req in requests]

        # Ensure blocks on GPU
        if hasattr(self.block_manager, 'ensure_blocks_on_gpu'):
            try:
                self.block_manager.ensure_blocks_on_gpu(seq_ids)
            except Exception:
                self._abort_prefill_allocations(requests)
                raise

        # 2. Build PACKED input: concatenate all tokens, build cu_seqlens
        all_tokens = []
        lengths_list = []
        for req in requests:
            all_tokens.extend(req.prompt_ids)
            lengths_list.append(req.prompt_len)

        total_tokens = len(all_tokens)
        num_seqs = len(lengths_list)
        input_ids, cu_seqlens, lengths = self._get_packed_prefill_buffers(
            total_tokens, num_seqs,
        )
        try:
            input_ids[0].copy_(torch.tensor(all_tokens, dtype=torch.long, device=self.device))
        except Exception:
            self._abort_prefill_allocations(requests)
            raise

        # cu_seqlens: [0, len1, len1+len2, ...]
        cum = [0]
        for l in lengths_list:
            cum.append(cum[-1] + l)
        try:
            cu_seqlens.copy_(torch.tensor(cum, dtype=torch.int32, device=self.device))
            lengths.copy_(torch.tensor(lengths_list, dtype=torch.long, device=self.device))
        except Exception:
            self._abort_prefill_allocations(requests)
            raise

        # 3. ONE forward pass — packed, zero-waste
        try:
            logits = self._run_prefill_packed_graph_or_eager(
                requests, input_ids, cu_seqlens, lengths, seq_ids,
            )
        except Exception:
            self._abort_prefill_allocations(requests)
            raise

        # 4. Evict cold blocks
        if hasattr(self.block_manager, 'evict_cold_blocks'):
            try:
                self.block_manager.evict_cold_blocks(seq_ids)
            except Exception:
                self._abort_prefill_allocations(requests)
                raise

        # 5. Sample first token per request
        for i, req in enumerate(requests):
            self._maybe_capture_prefill(req, logits[i, -1, :])
            next_token = sample_logits(
                logits[i, -1:, :], req.temperature, req.top_k, req.top_p,
            ).item()
            if self._benchmark_forced_token_id >= 0:
                next_token = self._benchmark_forced_token_id

            req.generated_ids.append(next_token)
            req.status = RequestStatus.RUNNING
            req.t_prefill_done = time.perf_counter()

            if req.is_done:
                req.status = RequestStatus.FINISHED
                req.t_end = time.perf_counter()
                self.block_manager.free_sequence(req.seq_id)
                self._completed.append(req)
            else:
                self._running[req.seq_id] = req

    def _prefill_chunk_padded(self, requests: List[Request]):
        """Prefill a chunk using LEFT-PADDED sequences (legacy fallback)."""
        # 1. Allocate blocks for ALL requests in chunk
        try:
            for req in requests:
                total_tokens = req.prompt_len + req.max_new_tokens
                self.block_manager.allocate_sequence(req.seq_id, total_tokens)
        except Exception:
            self._abort_prefill_allocations(requests)
            raise

        seq_ids = [req.seq_id for req in requests]

        # Ensure blocks on GPU
        if hasattr(self.block_manager, 'ensure_blocks_on_gpu'):
            try:
                self.block_manager.ensure_blocks_on_gpu(seq_ids)
            except Exception:
                self._abort_prefill_allocations(requests)
                raise

        # 2. Build LEFT-PADDED input tensor [N, max_len]
        prompt_lengths = [int(req.prompt_len) for req in requests]
        max_len = max(prompt_lengths)
        input_ids, lengths = self._get_padded_prefill_buffers(len(requests), max_len)

        try:
            padded_prompt_ids = [
                ([0] * (max_len - prompt_lengths[i])) + list(req.prompt_ids)
                for i, req in enumerate(requests)
            ]
            input_ids.copy_(
                torch.tensor(padded_prompt_ids, dtype=torch.long, device=self.device)
            )
            lengths.copy_(
                torch.tensor(prompt_lengths, dtype=lengths.dtype, device=self.device)
            )
        except Exception:
            self._abort_prefill_allocations(requests)
            raise

        # 3. ONE forward pass for this chunk
        try:
            logits = self._run_prefill_batch_graph_or_eager(
                requests,
                input_ids,
                lengths,
                seq_ids,
                prompt_lengths,
            )
        except Exception:
            self._abort_prefill_allocations(requests)
            raise
        self._record_prefill_stage_timing()

        # 4. Evict cold blocks
        if hasattr(self.block_manager, 'evict_cold_blocks'):
            try:
                self.block_manager.evict_cold_blocks(seq_ids)
            except Exception:
                self._abort_prefill_allocations(requests)
                raise

        # 5. Sample first token per request
        for i, req in enumerate(requests):
            self._maybe_capture_prefill(req, logits[i, -1, :])
            next_token = sample_logits(
                logits[i, -1:, :], req.temperature, req.top_k, req.top_p,
            ).item()
            if self._benchmark_forced_token_id >= 0:
                next_token = self._benchmark_forced_token_id

            req.generated_ids.append(next_token)
            req.status = RequestStatus.RUNNING
            req.t_prefill_done = time.perf_counter()

            if req.is_done:
                req.status = RequestStatus.FINISHED
                req.t_end = time.perf_counter()
                self.block_manager.free_sequence(req.seq_id)
                self._completed.append(req)
            else:
                self._running[req.seq_id] = req

    def _prefill_request(self, req: Request):
        """Prefill a single request and move it to running."""
        # Pre-allocate KV blocks for full sequence (prompt + output).
        # The model's attention forward writes KV directly via block_table
        # tensor, so blocks must exist before decode starts.
        total_tokens = req.prompt_len + req.max_new_tokens
        try:
            self.block_manager.allocate_sequence(req.seq_id, total_tokens)
        except Exception:
            self._abort_prefill_allocations([req])
            raise

        # Ensure blocks on GPU before prefill (KV offload support)
        if hasattr(self.block_manager, 'ensure_blocks_on_gpu'):
            try:
                self.block_manager.ensure_blocks_on_gpu([req.seq_id])
            except Exception:
                self._abort_prefill_allocations([req])
                raise

        # Run prefill
        try:
            input_ids = torch.tensor(
                [req.prompt_ids], dtype=torch.long, device=self.device
            )
            positions = torch.arange(
                req.prompt_len, device=self.device
            ).unsqueeze(0)
        except Exception:
            self._abort_prefill_allocations([req])
            raise

        try:
            logits = self.model.prefill(
                input_ids, positions, self.block_manager, req.seq_id
            )
        except Exception:
            self._abort_prefill_allocations([req])
            raise
        self._record_prefill_stage_timing()

        # Evict cold blocks after prefill (KV offload support)
        if hasattr(self.block_manager, 'evict_cold_blocks'):
            try:
                self.block_manager.evict_cold_blocks([req.seq_id])
            except Exception:
                self._abort_prefill_allocations([req])
                raise

        # Sample first token
        self._maybe_capture_prefill(req, logits[:, -1, :])
        next_token = sample_logits(
            logits[:, -1, :], req.temperature, req.top_k, req.top_p,
        ).item()
        if self._benchmark_forced_token_id >= 0:
            next_token = self._benchmark_forced_token_id

        req.generated_ids.append(next_token)
        req.status = RequestStatus.RUNNING
        req.t_prefill_done = time.perf_counter()

        # Check if already done (unlikely but possible)
        if req.is_done:
            req.status = RequestStatus.FINISHED
            req.t_end = time.perf_counter()
            self.block_manager.free_sequence(req.seq_id)
            self._completed.append(req)
        else:
            self._running[req.seq_id] = req

    def _log_decode_graph(self, message: str):
        if self._decode_graph_log_count >= self._decode_cuda_graph_log_limit:
            return
        self._decode_graph_log_count += 1
        print(f"  CUDA Graph decode: {message}")

    def _invalidate_decode_graph(self):
        self._decode_graph_state = None
        self._decode_graph_warm_key = None
        self._decode_graph_failed_key = None
        self._decode_graph_chain_started_keys.clear()

    def _decode_graph_batch_allowed(self, batch_size: int) -> bool:
        policy_batches = getattr(self, "_decode_cuda_graph_policy_batches", ())
        return not policy_batches or int(batch_size) in policy_batches

    def _decode_graph_is_eligible(self, seq_ids: List[int]) -> bool:
        if not self._decode_cuda_graphs:
            return False
        if not self._decode_graph_batch_allowed(len(seq_ids)):
            return False
        if self.device != 'cuda' or not torch.cuda.is_available():
            return False
        if len(seq_ids) < self._decode_cuda_graph_min_batch:
            return False
        # Keep the first rollout narrow: dense full-attention decode on a
        # plain GPU BlockManager, no offload, no tiered KV swapping.
        if type(self.block_manager) is not BlockManager:
            return False
        if getattr(self.model, "_offloader", None) is not None:
            return False
        if bool(getattr(self.model, "_all_full_attention", False)):
            return True
        checker = getattr(self.model, "decode_cuda_graph_eligible", None)
        if callable(checker):
            try:
                if checker(
                    num_seqs=len(seq_ids),
                    dtype=self.model.embed_tokens.weight.dtype,
                    device_type="cuda",
                    device_name=torch.cuda.get_device_name(),
                ):
                    return True
            except Exception:
                pass
        cfg = getattr(self.model, "config", None)
        if (
            self._decode_cuda_graph_allow_qwen3_moe
            and getattr(cfg, "model_type", "") == "qwen3_moe"
        ):
            return True
        return False

    def _advance_decode_graph_python_seq_lens(
        self,
        seq_ids: List[int],
        num_tokens: int = 1,
    ) -> None:
        """Keep Python seq_lens in sync after CUDA Graph replay.

        The captured graph records the GPU-side seq_lens tensor increment from
        BlockManager.advance_seq_len_batch(), but replay does not execute that
        Python method. Keep the dict current for later invalidation/rebuilds.
        """
        seq_lens = getattr(self.block_manager, "seq_lens", None)
        if not isinstance(seq_lens, dict):
            return
        for sid in seq_ids:
            sid = int(sid)
            if sid in seq_lens:
                seq_lens[sid] += int(num_tokens)

    def _decode_graph_shape_key(
        self,
        seq_ids: List[int],
        return_next_token: bool = False,
        chain_graph_inputs: bool = False,
    ) -> tuple:
        table_blocks = max(len(self.block_manager.block_tables[int(sid)]) for sid in seq_ids)
        block_size = max(1, int(getattr(self.block_manager, "block_size", 1) or 1))
        if self._decode_cuda_graph_stable_max_blocks:
            loop_blocks = table_blocks
        else:
            max_seq_len = max(int(self.block_manager.seq_lens[int(sid)]) for sid in seq_ids)
            loop_blocks = max(1, (max_seq_len + 1 + block_size - 1) // block_size)
        env_signature = tuple(
            os.environ.get(name, "").strip()
            for name in _DECODE_GRAPH_ENV_KEY_PARTS
        )
        return (
            len(seq_ids),
            int(table_blocks),
            int(loop_blocks),
            env_signature,
            bool(return_next_token),
            bool(chain_graph_inputs),
        )

    def _decode_graph_seq_lens_signature(self, seq_ids: List[int]) -> tuple:
        seq_lens = getattr(self.block_manager, "seq_lens", {})
        return tuple(int(seq_lens[int(sid)]) for sid in seq_ids)

    def _mark_decode_graph_shape_state_synced(
        self,
        state: dict,
        seq_ids: List[int],
    ) -> None:
        state["seq_lens_signature"] = self._decode_graph_seq_lens_signature(seq_ids)

    def _prepare_decode_graph_shape_state(
        self,
        seq_ids: List[int],
        return_next_token: bool = False,
        chain_graph_inputs: bool = False,
    ) -> tuple:
        key = self._decode_graph_shape_key(
            seq_ids,
            return_next_token=return_next_token,
            chain_graph_inputs=chain_graph_inputs,
        )
        num_seqs, table_blocks, loop_blocks = key[:3]
        state = self._decode_graph_shape_states.get(key)
        device = self._decode_input_ids.device
        if state is None:
            with torch.inference_mode(False):
                state = {
                    "key": key,
                    "seq_key": None,
                    "block_signature": None,
                    "block_table": torch.empty(
                        num_seqs, table_blocks, dtype=torch.int32, device=device,
                    ),
                    "seq_lens": torch.empty(num_seqs, dtype=torch.int32, device=device),
                    "graph": None,
                    "logits": None,
                    "num_seqs": int(num_seqs),
                    "table_blocks": int(table_blocks),
                    "max_decode_blocks": int(loop_blocks),
                    "seq_lens_signature": None,
                    "return_next_token": bool(return_next_token),
                    "chain_graph_inputs": bool(chain_graph_inputs),
                    "unrolled_burst_graphs": {},
                }
                if self._decode_cuda_graph_shared_shape_cache:
                    state["input_ids"] = torch.empty(
                        num_seqs, 1, dtype=self._decode_input_ids.dtype, device=device,
                    )
                    state["positions"] = torch.empty(
                        num_seqs, 1, dtype=self._decode_positions.dtype, device=device,
                    )
            self._decode_graph_shape_states[key] = state

        seq_key = tuple(int(sid) for sid in seq_ids)
        seq_lens_signature = self._decode_graph_seq_lens_signature(seq_ids)
        block_signature = tuple(
            tuple(int(block) for block in self.block_manager.block_tables[sid])
            for sid in seq_key
        )
        previous_block_signature = state.get("block_signature")
        physical_rebind = (
            previous_block_signature is not None
            and previous_block_signature != block_signature
        )
        if physical_rebind and state.get("graph") is not None:
            # A shared graph may outlive the Scheduler that captured it.  Keep the
            # mutable metadata tensors, but never replay a graph against a different
            # physical KV layout: several write kernels bind that layout during
            # capture even though the block-table contents are refreshed.
            state["graph"] = None
            state["logits"] = None
            state["unrolled_burst_graphs"] = {}
            state.pop("graph_input_ids", None)
            state.pop("graph_positions", None)
            self._decode_graph_shape_warm_keys.discard(key)
            self._decode_graph_physical_rebinds += 1
            state["physical_rebinds"] = int(state.get("physical_rebinds", 0)) + 1
        metadata_changed = (
            state.get("seq_key") != seq_key
            or state.get("block_signature") != block_signature
            or state.get("seq_lens_signature") != seq_lens_signature
        )
        if metadata_changed:
            table = state["block_table"]
            table.zero_()
            for row, sid in enumerate(seq_key):
                blocks = self.block_manager.block_tables[sid]
                if blocks:
                    table[row, :len(blocks)].copy_(
                        torch.tensor(blocks, dtype=torch.int32, device=table.device)
                    )
            state["seq_key"] = seq_key
            state["block_signature"] = block_signature
            state["seq_lens"].copy_(
                torch.tensor(
                    [self.block_manager.seq_lens[int(sid)] for sid in seq_ids],
                    dtype=torch.int32,
                    device=state["seq_lens"].device,
                )
            )
            state["seq_lens_signature"] = seq_lens_signature
        return key, state

    def _copy_decode_graph_shape_inputs(
        self,
        state: dict,
        buf_ids: torch.Tensor,
        buf_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._decode_cuda_graph_shared_shape_cache:
            return buf_ids, buf_pos
        input_ids = state.get("input_ids")
        positions = state.get("positions")
        if input_ids is None or positions is None:
            return buf_ids, buf_pos
        input_ids.copy_(buf_ids)
        positions.copy_(buf_pos)
        return input_ids, positions

    def _decode_graph_model_step(
        self, input_ids, positions, seq_ids, return_next_token=False,
    ):
        if getattr(self, "_decode_graph_multi_step_body", False):
            if not return_next_token:
                raise RuntimeError("multi-step graph body requires greedy token output")
            tokens, _ = self.model.decode_multi_step(
                input_ids, positions, self.block_manager, seq_ids,
                num_steps=1, return_final_logits=False, return_token_ids=True,
            )
            return tokens.reshape(len(seq_ids))
        return self.model.decode_step(
            input_ids, positions, self.block_manager, seq_ids,
            return_next_token=return_next_token,
        )

    def _run_decode_with_metadata_override(
        self,
        state: dict,
        seq_ids: List[int],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        return_next_token: bool = False,
    ) -> torch.Tensor:
        setter = getattr(self.block_manager, "set_decode_metadata_override", None)
        clearer = getattr(self.block_manager, "clear_decode_metadata_override", None)
        if setter is None or clearer is None:
            return self.model.decode_step(
                input_ids,
                positions,
                self.block_manager,
                seq_ids,
                return_next_token=return_next_token,
            )
        setter(
            state["block_table"],
            state["seq_lens"],
            int(state.get("max_decode_blocks") or state["block_table"].shape[1]),
        )
        try:
            logits = self._decode_graph_model_step(
                input_ids, positions, seq_ids,
                return_next_token=return_next_token,
            )
            self._mark_decode_graph_shape_state_synced(state, seq_ids)
            return logits
        finally:
            clearer()

    def _capture_decode_graph_shape(
        self,
        key: tuple,
        state: dict,
        seq_ids: List[int],
        buf_ids: torch.Tensor,
        buf_pos: torch.Tensor,
        return_next_token: bool = False,
        chain_graph_inputs: bool = False,
    ) -> torch.Tensor:
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        setter = getattr(self.block_manager, "set_decode_metadata_override")
        clearer = getattr(self.block_manager, "clear_decode_metadata_override")
        input_ids = buf_ids
        positions = buf_pos
        python_seq_lens_before = {
            int(sid): int(self.block_manager.seq_lens[int(sid)]) for sid in seq_ids
        }
        setter(
            state["block_table"],
            state["seq_lens"],
            int(state.get("max_decode_blocks") or state["block_table"].shape[1]),
        )
        try:
            with torch.cuda.graph(graph):
                logits = self._decode_graph_model_step(
                    input_ids, positions, seq_ids,
                    return_next_token=return_next_token,
                )
                if chain_graph_inputs:
                    input_ids.copy_(logits.reshape_as(input_ids))
                    positions.add_(1)
        except Exception:
            for sid, length in python_seq_lens_before.items():
                self.block_manager.seq_lens[sid] = length
            raise
        finally:
            clearer()
        state["graph"] = graph
        state["logits"] = logits
        state["graph_input_ids"] = input_ids
        state["graph_positions"] = positions
        # Stream capture records the decode workload but does not execute the
        # first autoregressive step.  Materialize it before returning logits.
        graph.replay()
        self._decode_graph_replay_count += 1
        self._mark_decode_graph_shape_state_synced(state, seq_ids)
        self._decode_graph_capture_count += 1
        self._log_decode_graph(
            f"captured shape batch={key[0]} table_blocks={key[1]} loop_blocks={key[2]}"
        )
        return logits

    def _run_decode_step_shape_graph(
        self,
        seq_ids: List[int],
        buf_ids: torch.Tensor,
        buf_pos: torch.Tensor,
        return_next_token: bool = False,
    ) -> torch.Tensor:
        chain_graph_inputs = bool(
            self._decode_graph_persistent_token_feedback
            and return_next_token
        )
        key, state = self._prepare_decode_graph_shape_state(
            seq_ids,
            return_next_token=return_next_token,
            chain_graph_inputs=chain_graph_inputs,
        )
        graph_input_ids = state.get("graph_input_ids")
        if graph_input_ids is None:
            graph_input_ids = state.get("input_ids")
        graph_positions = state.get("graph_positions")
        if graph_positions is None:
            graph_positions = state.get("positions")
        persistent_replay = bool(
            chain_graph_inputs
            and (state.get("graph") is not None or state.get("eager_control", False))
            and key in self._decode_graph_chain_started_keys
            and state.get("seq_key") == tuple(int(sid) for sid in seq_ids)
            and graph_input_ids is not None
            and graph_positions is not None
        )
        if persistent_replay:
            input_ids, positions = graph_input_ids, graph_positions
            self._decode_graph_chain_input_updates_skipped += 2
        else:
            input_ids, positions = self._copy_decode_graph_shape_inputs(
                state, buf_ids, buf_pos
            )
        self._decode_graph_last_feedback_persistent = False
        if self._decode_graph_eager_control:
            if not self._decode_graph_multi_step_body or not chain_graph_inputs:
                raise RuntimeError(
                    "eager graph control requires the multi-step token-feedback body"
                )
            result = self._run_decode_with_metadata_override(
                state, seq_ids, input_ids, positions, return_next_token=True,
            )
            input_ids.copy_(result.reshape_as(input_ids))
            positions.add_(1)
            state.update(eager_control=True, graph_input_ids=input_ids, graph_positions=positions)
            self._decode_graph_chain_started_keys.add(key)
            self._decode_graph_last_feedback_persistent = True
            self._decode_graph_persistent_feedback_steps += 1
            return result
        if key in self._decode_graph_shape_failed_keys:
            return self._run_decode_with_metadata_override(
                state,
                seq_ids,
                input_ids,
                positions,
                return_next_token=return_next_token,
            )

        if state.get("graph") is not None:
            state["graph"].replay()
            self._decode_graph_replay_count += 1
            self._advance_decode_graph_python_seq_lens(seq_ids, 1)
            self._mark_decode_graph_shape_state_synced(state, seq_ids)
            if chain_graph_inputs:
                self._decode_graph_chain_started_keys.add(key)
                self._decode_graph_last_feedback_persistent = True
                self._decode_graph_persistent_feedback_steps += 1
            return state["logits"]

        if key not in self._decode_graph_shape_warm_keys:
            self._decode_graph_shape_warm_keys.add(key)
            self._decode_graph_eager_warmups += 1
            self._log_decode_graph(
                f"warmup shape batch={key[0]} table_blocks={key[1]} loop_blocks={key[2]}"
            )
            return self._run_decode_with_metadata_override(
                state,
                seq_ids,
                input_ids,
                positions,
                return_next_token=return_next_token,
            )

        try:
            result = self._capture_decode_graph_shape(
                key,
                state,
                seq_ids,
                input_ids,
                positions,
                return_next_token=return_next_token,
                chain_graph_inputs=chain_graph_inputs,
            )
            if chain_graph_inputs:
                self._decode_graph_chain_started_keys.add(key)
                self._decode_graph_last_feedback_persistent = True
                self._decode_graph_persistent_feedback_steps += 1
            return result
        except Exception as exc:
            self._decode_graph_failures += 1
            self._decode_graph_shape_failed_keys.add(key)
            tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            stack = traceback.extract_tb(exc.__traceback__)
            if stack:
                frame = stack[-1]
                tb = f"{tb} at {frame.filename}:{frame.lineno} in {frame.name}"
            self._decode_graph_last_failure = tb
            if self._decode_graph_multi_step_body:
                # Experimental A/B must stop on capture failure, not execute
                # an eager fallback against partially captured mutable state.
                raise RuntimeError(
                    f"multi-step decode graph capture failed: {tb}"
                ) from exc
            self._log_decode_graph(
                f"capture failed for shape batch={key[0]} table_blocks={key[1]} "
                f"loop_blocks={key[2]}: {tb}; "
                "falling back to eager",
            )
            self._decode_graph_last_feedback_persistent = False
            return self._run_decode_with_metadata_override(
                state,
                seq_ids,
                input_ids,
                positions,
                return_next_token=return_next_token,
            )

    def _capture_decode_graph(
        self,
        seq_ids: List[int],
        buf_ids: torch.Tensor,
        buf_pos: torch.Tensor,
        return_next_token: bool = False,
    ) -> torch.Tensor:
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            logits = self.model.decode_step(
                buf_ids, buf_pos, self.block_manager, seq_ids,
                return_next_token=return_next_token,
            )
        graph.replay()
        self._decode_graph_state = {
            'key': (tuple(seq_ids), bool(return_next_token)),
            'graph': graph,
            'logits': logits,
            'num_seqs': len(seq_ids),
            'return_next_token': bool(return_next_token),
        }
        self._decode_graph_capture_count += 1
        self._decode_graph_replay_count += 1
        self._log_decode_graph(f"captured batch={len(seq_ids)}")
        return logits

    def _run_decode_step(
        self,
        seq_ids: List[int],
        buf_ids: torch.Tensor,
        buf_pos: torch.Tensor,
        return_next_token: bool = False,
    ) -> torch.Tensor:
        if not self._decode_graph_is_eligible(seq_ids):
            return self.model.decode_step(
                buf_ids, buf_pos, self.block_manager, seq_ids,
                return_next_token=return_next_token,
            )
        if self._decode_cuda_graph_shape_cache:
            return self._run_decode_step_shape_graph(
                seq_ids,
                buf_ids,
                buf_pos,
                return_next_token=return_next_token,
            )

        key = (tuple(seq_ids), bool(return_next_token))
        state = self._decode_graph_state
        if state is not None and state.get('key') != key:
            self._invalidate_decode_graph()
            state = None

        if self._decode_graph_failed_key == key:
            return self.model.decode_step(
                buf_ids, buf_pos, self.block_manager, seq_ids,
                return_next_token=return_next_token,
            )
        if state is not None:
            state['graph'].replay()
            self._decode_graph_replay_count += 1
            self._advance_decode_graph_python_seq_lens(seq_ids, 1)
            return state['logits']

        # Warm one eager step before capture so the model can finish any lazy
        # one-time setup outside graph capture.
        if self._decode_graph_warm_key != key:
            self._decode_graph_warm_key = key
            self._decode_graph_eager_warmups += 1
            self._log_decode_graph(f"warmup batch={len(seq_ids)}")
            return self.model.decode_step(
                buf_ids, buf_pos, self.block_manager, seq_ids,
                return_next_token=return_next_token,
            )

        try:
            return self._capture_decode_graph(
                seq_ids,
                buf_ids,
                buf_pos,
                return_next_token=return_next_token,
            )
        except Exception as exc:
            self._decode_graph_failures += 1
            self._decode_graph_failed_key = key
            tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            stack = traceback.extract_tb(exc.__traceback__)
            if stack:
                frame = stack[-1]
                tb = f"{tb} at {frame.filename}:{frame.lineno} in {frame.name}"
            self._decode_graph_last_failure = tb
            self._log_decode_graph(
                f"capture failed for batch={len(seq_ids)}: {tb}; "
                "falling back to eager",
            )
            return self.model.decode_step(
                buf_ids, buf_pos, self.block_manager, seq_ids,
                return_next_token=return_next_token,
            )

    def _decode_graph_persistent_chain_ready(self, seq_ids: List[int]) -> bool:
        """Return whether this Scheduler can resume the graph-owned token inputs."""
        if not (
            self._decode_graph_persistent_token_feedback
            and self._decode_cuda_graphs
            and self._decode_cuda_graph_shape_cache
        ):
            return False
        key = self._decode_graph_shape_key(
            seq_ids,
            return_next_token=True,
            chain_graph_inputs=True,
        )
        state = self._decode_graph_shape_states.get(key)
        if (
            state is None
            or (state.get("graph") is None and not state.get("eager_control", False))
            or key not in self._decode_graph_chain_started_keys
            or state.get("seq_key") != tuple(int(sid) for sid in seq_ids)
        ):
            return False
        block_signature = tuple(
            tuple(int(block) for block in self.block_manager.block_tables[int(sid)])
            for sid in seq_ids
        )
        return state.get("block_signature") == block_signature

    def _decode_unrolled_graph_burst_is_eligible(
        self,
        seq_ids: List[int],
        num_steps: int,
    ) -> bool:
        """Keep multi-token graph capture on the audited E2B/L4 shape."""
        if not (
            self._decode_unrolled_graph_burst
            and int(num_steps) > 1
            and self._decode_graph_persistent_chain_ready(seq_ids)
        ):
            return False
        checker = getattr(
            self.model,
            "decode_unrolled_graph_burst_eligible",
            None,
        )
        if not callable(checker):
            return False
        try:
            return bool(
                checker(
                    num_seqs=len(seq_ids),
                    num_steps=int(num_steps),
                    dtype=self.model.embed_tokens.weight.dtype,
                    device_type="cuda",
                    device_name=torch.cuda.get_device_name(),
                )
            )
        except Exception:
            return False

    def _capture_decode_unrolled_graph_burst(
        self,
        *,
        state: dict,
        seq_ids: List[int],
        output_tokens: torch.Tensor,
        num_steps: int,
    ) -> dict:
        """Capture N autoregressive token steps as one CUDA Graph launch."""
        input_ids = state.get("graph_input_ids")
        positions = state.get("graph_positions")
        if input_ids is None or positions is None:
            raise RuntimeError(
                "persistent graph inputs are missing for unrolled burst capture"
            )
        if tuple(output_tokens.shape) != (len(seq_ids), int(num_steps)):
            raise RuntimeError(
                "unrolled graph output must have shape [num_seqs, num_steps]"
            )

        setter = getattr(self.block_manager, "set_decode_metadata_override", None)
        clearer = getattr(self.block_manager, "clear_decode_metadata_override", None)
        if setter is None or clearer is None:
            raise RuntimeError("BlockManager lacks CUDA Graph metadata overrides")

        graph = torch.cuda.CUDAGraph()
        python_seq_lens_before = {
            int(sid): int(self.block_manager.seq_lens[int(sid)])
            for sid in seq_ids
        }
        setter(
            state["block_table"],
            state["seq_lens"],
            int(state.get("max_decode_blocks") or state["block_table"].shape[1]),
        )
        try:
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                for step in range(int(num_steps)):
                    next_tokens = self._decode_graph_model_step(
                        input_ids, positions, seq_ids,
                        return_next_token=True,
                    ).reshape(len(seq_ids))
                    output_tokens[:, step].copy_(next_tokens)
                    # The final feedback is intentional: the next burst resumes
                    # from graph-owned token/position buffers without a host copy.
                    input_ids.copy_(next_tokens.reshape_as(input_ids))
                    positions.add_(1)
        except Exception:
            # decode_step updates the Python sequence-length dictionary while the
            # CUDA operations are captured.  Restore it before surfacing a failed
            # capture; replaying or falling back after a partial capture is unsafe.
            for sid, length in python_seq_lens_before.items():
                self.block_manager.seq_lens[int(sid)] = int(length)
            raise
        finally:
            clearer()

        # Capture recorded all N dependent steps.  One replay performs the
        # actual burst and materializes every output token.
        graph.replay()
        self._decode_graph_replay_count += 1
        self._decode_unrolled_graph_burst_replays += 1

        entry = {
            "graph": graph,
            "output_tokens": output_tokens,
            "num_steps": int(num_steps),
            "seq_key": tuple(int(sid) for sid in seq_ids),
            "block_signature": state.get("block_signature"),
        }
        state.setdefault("unrolled_burst_graphs", {})[int(num_steps)] = entry
        self._mark_decode_graph_shape_state_synced(state, seq_ids)
        self._decode_unrolled_graph_burst_captures += 1
        self._log_decode_graph(
            f"captured unrolled burst batch={len(seq_ids)} steps={int(num_steps)}"
        )
        return entry

    def _run_unrolled_decode_graph_token_burst(
        self,
        seq_ids: List[int],
        burst_tokens: torch.Tensor,
        num_steps: int,
    ) -> bool:
        """Capture or replay one graph containing the complete token burst."""
        if not self._decode_unrolled_graph_burst_is_eligible(seq_ids, num_steps):
            return False
        key = self._decode_graph_shape_key(
            seq_ids,
            return_next_token=True,
            chain_graph_inputs=True,
        )
        state = self._decode_graph_shape_states.get(key)
        if state is None:
            return False

        entries = state.setdefault("unrolled_burst_graphs", {})
        entry = entries.get(int(num_steps))
        try:
            if entry is None:
                entry = self._capture_decode_unrolled_graph_burst(
                    state=state,
                    seq_ids=seq_ids,
                    output_tokens=burst_tokens,
                    num_steps=int(num_steps),
                )
                # Capture executed the device work and decode_step kept the
                # Python sequence-length dictionary synchronized.
            else:
                if entry.get("seq_key") != tuple(int(sid) for sid in seq_ids):
                    raise RuntimeError("unrolled graph sequence membership changed")
                if entry.get("block_signature") != state.get("block_signature"):
                    raise RuntimeError("unrolled graph physical KV layout changed")
                entry["graph"].replay()
                self._decode_graph_replay_count += 1
                self._decode_unrolled_graph_burst_replays += 1
                self._advance_decode_graph_python_seq_lens(seq_ids, int(num_steps))
                self._mark_decode_graph_shape_state_synced(state, seq_ids)

            graph_output = entry["output_tokens"]
            if graph_output.data_ptr() != burst_tokens.data_ptr():
                burst_tokens.copy_(graph_output)
        except Exception as exc:
            self._decode_unrolled_graph_burst_failures += 1
            self._decode_unrolled_graph_last_failure = (
                f"{type(exc).__name__}: {exc}"
            )
            raise

        self._decode_graph_last_feedback_persistent = True
        self._decode_graph_persistent_feedback_steps += int(num_steps)
        self._decode_unrolled_graph_bursts += 1
        self._decode_unrolled_graph_burst_steps += int(num_steps)
        return True

    def _run_native_decode_graph_token_burst(
        self,
        seq_ids: List[int],
        burst_tokens: torch.Tensor,
        num_steps: int,
    ) -> bool:
        """Replay an already-captured persistent graph in one native call.

        This path deliberately requires an established persistent-feedback
        chain.  Graph warmup/capture stays on the ordinary audited path; after
        capture, no model or layer Python callback runs between token steps.
        """
        if not self._decode_native_graph_burst:
            return False
        if not self._decode_graph_persistent_chain_ready(seq_ids):
            return False

        key = self._decode_graph_shape_key(
            seq_ids,
            return_next_token=True,
            chain_graph_inputs=True,
        )
        state = self._decode_graph_shape_states.get(key)
        if state is None or state.get("graph") is None or state.get("logits") is None:
            return False

        # The extension validates every tensor before issuing the first replay.
        # A replay failure leaves graph-owned state potentially advanced, so it
        # must be surfaced instead of silently executing the tokens twice.
        _decode_native_ops.run_cuda_graph_token_burst(
            state["graph"],
            state["logits"],
            burst_tokens,
            int(num_steps),
        )
        self._decode_graph_replay_count += int(num_steps)
        self._advance_decode_graph_python_seq_lens(seq_ids, int(num_steps))
        self._mark_decode_graph_shape_state_synced(state, seq_ids)
        self._decode_graph_last_feedback_persistent = True
        self._decode_graph_persistent_feedback_steps += int(num_steps)
        self._decode_native_graph_bursts += 1
        self._decode_native_graph_burst_steps += int(num_steps)
        return True

    def _decode_batch(self) -> List[Request]:
        """Decode one token for all running requests in a single batch.

        Optimized hot path:
        - Pre-allocated input/position buffers (zero allocation per step)
        - Block table tensor only rebuilt when batch membership changes
        - Batched greedy sampling via single argmax
        """
        seq_ids = list(self._running.keys())
        requests = [self._running[sid] for sid in seq_ids]
        num_seqs = len(seq_ids)

        all_greedy = all(req.temperature == 0.0 for req in requests)
        greedy_token_decode = bool(
            all_greedy
            and self._prefers_scheduler_greedy_token_decode(num_seqs)
        )

        # Stage on the host, then issue two batched H2D copies per decode step.
        buf_ids = self._decode_input_ids[:num_seqs]
        buf_pos = self._decode_positions[:num_seqs]
        host_ids = self._decode_input_ids_host[:num_seqs]
        host_pos = self._decode_positions_host[:num_seqs]
        for i, req in enumerate(requests):
            host_ids[i] = req.generated_ids[-1]
            host_pos[i] = req.current_len - 1
        buf_ids[:, 0].copy_(
            host_ids,
            non_blocking=self._decode_host_buffers_pinned,
        )
        buf_pos[:, 0].copy_(
            host_pos,
            non_blocking=self._decode_host_buffers_pinned,
        )
        self._decode_vectorized_input_updates += 1

        # Only invalidate block_table when batch membership changed
        if self._batch_changed:
            self._invalidate_decode_graph()
            self.block_manager._block_table_tensor = None
            self.block_manager._seq_lens_tensor = None  # also rebuild on batch change
            self._batch_changed = False

        # Ensure all blocks on GPU before decode (KV offload support)
        if hasattr(self.block_manager, 'ensure_blocks_on_gpu'):
            self.block_manager.ensure_blocks_on_gpu(seq_ids)

        # For the gated greedy shape, the graph returns token IDs directly.
        decode_output = self._run_decode_step(
            seq_ids,
            buf_ids,
            buf_pos,
            return_next_token=greedy_token_decode,
        )

        # Evict cold blocks after decode (KV offload support)
        if hasattr(self.block_manager, 'evict_cold_blocks'):
            self.block_manager.evict_cold_blocks(seq_ids)

        # Batched sampling — check if all seqs use same greedy params
        if all_greedy:
            if greedy_token_decode:
                next_tokens = decode_output.reshape(num_seqs)
                self._decode_greedy_token_steps += 1
            else:
                all_logits = decode_output[:, -1, :]
                next_tokens = all_logits.argmax(dim=-1)
            tokens_host = self._decode_next_tokens_host[:num_seqs]
            tokens_host.copy_(next_tokens.detach(), non_blocking=False)
            tokens_cpu = tokens_host.tolist()
            self._decode_batched_token_host_copies += 1
            for req, token_id in zip(requests, tokens_cpu):
                req.generated_ids.append(int(token_id))
        else:
            # Mixed params: per-request sampling
            for i, req in enumerate(requests):
                next_token = sample_logits(
                    decode_output[i:i+1, -1, :],
                    req.temperature, req.top_k, req.top_p,
                ).item()
                req.generated_ids.append(next_token)

        # Check for finished sequences
        finished = []
        for req in requests:
            if req.is_done:
                req.status = RequestStatus.FINISHED
                req.t_end = time.perf_counter()
                self.block_manager.free_sequence(req.seq_id)
                del self._running[req.seq_id]
                self._batch_changed = True  # batch membership changed
                finished.append(req)

        return finished

    def _decode_graph_token_burst_batch(self, num_steps: int = 8) -> List[Request]:
        """Replay exact greedy-token steps and synchronize tokens once per burst."""
        seq_ids = list(self._running.keys())
        requests = [self._running[sid] for sid in seq_ids]
        num_seqs = len(seq_ids)
        max_remaining = min(
            req.max_new_tokens - req.num_generated for req in requests
        )
        actual_steps = min(
            max(1, int(num_steps)),
            max_remaining,
            self._decode_multi_step_burst,
        )
        if actual_steps <= 1 or any(req.stop_token_ids for req in requests):
            return self._decode_batch()

        buf_ids = self._decode_input_ids[:num_seqs]
        buf_pos = self._decode_positions[:num_seqs]
        if self._batch_changed:
            self._invalidate_decode_graph()
            self.block_manager._block_table_tensor = None
            self.block_manager._seq_lens_tensor = None
            self._batch_changed = False

        if hasattr(self.block_manager, 'ensure_blocks_on_gpu'):
            self.block_manager.ensure_blocks_on_gpu(seq_ids)

        resume_persistent_chain = self._decode_graph_persistent_chain_ready(seq_ids)
        if resume_persistent_chain:
            self._decode_graph_chain_input_updates_skipped += 2
        else:
            host_ids = self._decode_input_ids_host[:num_seqs]
            host_pos = self._decode_positions_host[:num_seqs]
            for i, req in enumerate(requests):
                host_ids[i] = req.generated_ids[-1]
                host_pos[i] = req.current_len - 1
            buf_ids[:, 0].copy_(
                host_ids,
                non_blocking=self._decode_host_buffers_pinned,
            )
            buf_pos[:, 0].copy_(
                host_pos,
                non_blocking=self._decode_host_buffers_pinned,
            )
            self._decode_vectorized_input_updates += 1

        burst_tokens = self._decode_burst_tokens[:num_seqs, :actual_steps]
        unrolled_burst = bool(
            resume_persistent_chain
            and self._run_unrolled_decode_graph_token_burst(
                seq_ids,
                burst_tokens,
                actual_steps,
            )
        )
        native_burst = bool(
            not unrolled_burst
            and resume_persistent_chain
            and self._run_native_decode_graph_token_burst(
                seq_ids,
                burst_tokens,
                actual_steps,
            )
        )
        if not unrolled_burst and not native_burst:
            for step in range(actual_steps):
                next_tokens = self._run_decode_step(
                    seq_ids,
                    buf_ids,
                    buf_pos,
                    return_next_token=True,
                ).reshape(num_seqs)
                burst_tokens[:, step].copy_(next_tokens)
                if step + 1 < actual_steps:
                    if not self._decode_graph_last_feedback_persistent:
                        buf_ids[:, 0].copy_(next_tokens)
                        buf_pos.add_(1)
                        self._decode_graph_token_feedback_copies += 1

        if hasattr(self.block_manager, 'evict_cold_blocks'):
            self.block_manager.evict_cold_blocks(seq_ids)

        tokens_host = self._decode_burst_tokens_host[:num_seqs, :actual_steps]
        tokens_host.copy_(burst_tokens.detach(), non_blocking=False)
        token_rows = tokens_host.tolist()
        self._decode_batched_token_host_copies += 1
        self._decode_greedy_token_steps += actual_steps
        self._decode_graph_token_bursts += 1
        self._decode_graph_token_burst_steps += actual_steps

        finished = []
        for req, row in zip(requests, token_rows):
            req.generated_ids.extend(int(token_id) for token_id in row)
            if req.is_done:
                req.status = RequestStatus.FINISHED
                req.t_end = time.perf_counter()
                self.block_manager.free_sequence(req.seq_id)
                del self._running[req.seq_id]
                self._batch_changed = True
                finished.append(req)
        return finished

    def _decode_multi_step_batch(self, num_steps: int = 16) -> List[Request]:
        """Multi-step decode: run N steps on GPU without returning to Python.

        Only used when ALL requests are greedy (temperature=0) and no
        waiting requests need interleaving. Keeps the GPU saturated by
        eliminating N-1 CPU↔GPU roundtrips.
        """
        seq_ids = list(self._running.keys())
        requests = [self._running[sid] for sid in seq_ids]
        num_seqs = len(seq_ids)

        # Compute max steps each seq can take before hitting limit
        max_remaining = min(
            req.max_new_tokens - req.num_generated for req in requests
        )
        actual_steps = min(num_steps, max_remaining)
        if actual_steps <= 0:
            return []

        # Build inputs
        buf_ids = self._decode_input_ids[:num_seqs]
        buf_pos = self._decode_positions[:num_seqs]
        for i, req in enumerate(requests):
            buf_ids[i, 0] = req.generated_ids[-1]
            buf_pos[i, 0] = req.current_len - 1

        # Invalidate cache if batch changed
        if self._batch_changed:
            self._invalidate_decode_graph()
            self.block_manager._block_table_tensor = None
            self.block_manager._seq_lens_tensor = None
            self._batch_changed = False

        # Ensure blocks on GPU (KV offload support)
        if hasattr(self.block_manager, 'ensure_blocks_on_gpu'):
            self.block_manager.ensure_blocks_on_gpu(seq_ids)

        no_stop_tokens = all(not req.stop_token_ids for req in requests)
        all_finish_after_burst = all(
            req.num_generated + actual_steps >= req.max_new_tokens
            for req in requests
        )
        skip_token_materialization = (
            self._decode_skip_token_store
            and not self._materialize_generated_tokens
            and no_stop_tokens
            and all_finish_after_burst
        )

        # Run N decode steps entirely on GPU!
        all_tokens, _ = self.model.decode_multi_step(
            buf_ids, buf_pos, self.block_manager, seq_ids,
            num_steps=actual_steps,
            return_final_logits=False,
            return_token_ids=not skip_token_materialization,
        )

        # Evict cold blocks (KV offload support)
        if hasattr(self.block_manager, 'evict_cold_blocks'):
            self.block_manager.evict_cold_blocks(seq_ids)

        if skip_token_materialization:
            finished = []
            for req in requests:
                take = min(actual_steps, req.max_new_tokens - req.num_generated)
                if take > 0:
                    # Benchmark-only path: preserve token counts without storing
                    # or copying a generated token matrix that nobody will read.
                    req.generated_ids.extend([0] * take)

                if req.is_done:
                    req.status = RequestStatus.FINISHED
                    req.t_end = time.perf_counter()
                    self.block_manager.free_sequence(req.seq_id)
                    del self._running[req.seq_id]
                    self._batch_changed = True
                    finished.append(req)

            return finished

        # Distribute generated tokens back to requests
        # all_tokens: [num_seqs, actual_steps] on GPU
        tokens_cpu = all_tokens.cpu().tolist()

        finished = []
        for i, req in enumerate(requests):
            burst_tokens = tokens_cpu[i]
            take = min(actual_steps, req.max_new_tokens - req.num_generated)
            if req.stop_token_ids:
                for step, token_id in enumerate(burst_tokens[:take]):
                    if token_id in req.stop_token_ids:
                        take = step + 1
                        break
            if take > 0:
                req.generated_ids.extend(burst_tokens[:take])

            if req.is_done:
                req.status = RequestStatus.FINISHED
                req.t_end = time.perf_counter()
                self.block_manager.free_sequence(req.seq_id)
                del self._running[req.seq_id]
                self._batch_changed = True
                finished.append(req)

        return finished

    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        stats = {
            'waiting': self.num_waiting,
            'running': self.num_running,
            'completed': len(self._completed),
        }
        if self._completed:
            total_tokens = sum(r.num_generated for r in self._completed)
            total_time = sum(r.t_end - r.t_start for r in self._completed)
            decode_time = sum(r.t_end - r.t_prefill_done for r in self._completed)
            stats['total_tokens'] = total_tokens
            stats['total_throughput'] = total_tokens / total_time if total_time > 0 else 0
            stats['decode_throughput'] = total_tokens / decode_time if decode_time > 0 else 0
        stats['prefill_time_ms'] = self._prefill_time * 1000
        stats['decode_time_ms'] = self._decode_time * 1000
        stats['benchmark_forced_token_id'] = self._benchmark_forced_token_id
        stats['decode_execution'] = {
            'prefer_step': bool(self._decode_prefer_step),
            'decode_step_batches': int(self._decode_step_batches),
            'multi_step_batches': int(self._decode_multi_step_batches),
        }
        if self._prefill_last_chunk_plan is not None:
            stats['prefill_chunk_plan'] = dict(self._prefill_last_chunk_plan)
        if self._prefill_stage_timing_totals:
            stats['prefill_stage_timing'] = dict(self._prefill_stage_timing_totals)
            stats['prefill_stage_chunks'] = self._prefill_stage_timing_chunks
            stats['prefill_stage_total_tokens'] = self._prefill_stage_total_tokens
            stats['prefill_stage_total_seqs'] = self._prefill_stage_total_seqs
            stats['prefill_stage_max_len'] = self._prefill_stage_max_len
        prefill_graph_store = self._get_prefill_graph_store()
        if (
            self._prefill_cuda_graphs
            or (
                prefill_graph_store
                and (
                    prefill_graph_store.get('captures')
                    or prefill_graph_store.get('replays')
                    or prefill_graph_store.get('failures')
                    or prefill_graph_store.get('warmups')
                )
            )
        ):
            stats['prefill_cuda_graphs'] = {
                'enabled': bool(self._prefill_cuda_graphs),
                'min_reqs': int(self._prefill_cuda_graph_min_reqs),
            }
            if prefill_graph_store:
                prefill_graph_states = list(
                    (prefill_graph_store.get('buckets', {}) or {}).values()
                )
                stats['prefill_cuda_graphs'].update({
                    'captures': int(prefill_graph_store.get('captures', 0) or 0),
                    'capture_body_warmups': int(
                        prefill_graph_store.get('capture_body_warmups', 0) or 0
                    ),
                    'capture_replays': int(
                        prefill_graph_store.get('capture_replays', 0) or 0
                    ),
                    'replays': int(prefill_graph_store.get('replays', 0) or 0),
                    'external_kv_write_replays': int(
                        prefill_graph_store.get('external_kv_write_replays', 0) or 0
                    ),
                    'warmups': int(prefill_graph_store.get('warmups', 0) or 0),
                    'skips': int(prefill_graph_store.get('skips', 0) or 0),
                    'failures': int(prefill_graph_store.get('failures', 0) or 0),
                    'buckets': int(len(prefill_graph_store.get('buckets', {}) or {})),
                    'bucket_kinds': sorted({
                        str(state.get('kind', 'unknown'))
                        for state in prefill_graph_states
                    }),
                    'kv_write_modes': sorted({
                        str(state.get('kv_write_mode', 'captured'))
                        for state in prefill_graph_states
                    }),
                    'deferred_kv_layers': int(sum(
                        len(state.get('deferred_kv', ()) or ())
                        for state in prefill_graph_states
                    )),
                    'workspace_tensors': int(sum(
                        len(state.get('workspace_refs', ()) or ())
                        for state in prefill_graph_states
                    )),
                    'workspace_bytes': int(sum(
                        tensor.numel() * tensor.element_size()
                        for state in prefill_graph_states
                        for tensor in (state.get('workspace_refs', ()) or ())
                    )),
                })
                if prefill_graph_store.get('last_failure'):
                    stats['prefill_cuda_graphs']['last_failure'] = (
                        prefill_graph_store['last_failure']
                    )
        if (
            self._decode_cuda_graphs
            or self._request_scheduler_reuse_enabled
            or self._decode_graph_capture_count
            or self._decode_graph_replay_count
            or self._decode_graph_failures
        ):
            stats['decode_cuda_graphs'] = {
                'enabled': bool(self._decode_cuda_graphs),
                'multi_step_body': bool(self._decode_graph_multi_step_body),
                'eager_control': bool(self._decode_graph_eager_control),
                'prefer_step': bool(self._decode_cuda_graph_prefer_step),
                'token_burst_enabled': bool(self._decode_graph_token_burst),
                'token_burst_size': int(self._decode_multi_step_burst),
                'shape_cache': bool(self._decode_cuda_graph_shape_cache),
                'shared_shape_cache': bool(self._decode_cuda_graph_shared_shape_cache),
                'request_scheduler_reuse_enabled': bool(
                    self._request_scheduler_reuse_enabled
                ),
                'request_scheduler_reused': bool(self._request_scheduler_reused),
                'request_scheduler_reuse_count': int(
                    self._request_scheduler_reuse_count
                ),
                'stable_max_blocks': bool(self._decode_cuda_graph_stable_max_blocks),
                'min_batch': int(self._decode_cuda_graph_min_batch),
                'allow_qwen3_moe': bool(self._decode_cuda_graph_allow_qwen3_moe),
                'captures': int(self._decode_graph_capture_count),
                'replays': int(self._decode_graph_replay_count),
                'warmups': int(self._decode_graph_eager_warmups),
                'failures': int(self._decode_graph_failures),
                'physical_rebinds': int(self._decode_graph_physical_rebinds),
                'shape_graphs': int(len(self._decode_graph_shape_states)),
                'greedy_token_shape_graphs': int(sum(
                    bool(state.get('return_next_token', False))
                    for state in self._decode_graph_shape_states.values()
                )),
                'greedy_token_steps': int(self._decode_greedy_token_steps),
                'token_bursts': int(self._decode_graph_token_bursts),
                'token_burst_steps': int(self._decode_graph_token_burst_steps),
                'token_feedback_copies': int(
                    self._decode_graph_token_feedback_copies
                ),
                'persistent_token_feedback_enabled': bool(
                    self._decode_graph_persistent_token_feedback
                ),
                'persistent_token_feedback_steps': int(
                    self._decode_graph_persistent_feedback_steps
                ),
                'native_token_burst_enabled': bool(
                    self._decode_native_graph_burst
                ),
                'native_token_bursts': int(self._decode_native_graph_bursts),
                'native_token_burst_steps': int(
                    self._decode_native_graph_burst_steps
                ),
                'unrolled_token_burst_enabled': bool(
                    self._decode_unrolled_graph_burst
                ),
                'unrolled_token_burst_captures': int(
                    self._decode_unrolled_graph_burst_captures
                ),
                'unrolled_token_burst_replays': int(
                    self._decode_unrolled_graph_burst_replays
                ),
                'unrolled_token_bursts': int(
                    self._decode_unrolled_graph_bursts
                ),
                'unrolled_token_burst_steps': int(
                    self._decode_unrolled_graph_burst_steps
                ),
                'unrolled_token_burst_failures': int(
                    self._decode_unrolled_graph_burst_failures
                ),
                'chain_input_updates_skipped': int(
                    self._decode_graph_chain_input_updates_skipped
                ),
                'batched_token_host_copies': int(
                    self._decode_batched_token_host_copies
                ),
                'vectorized_input_updates': int(
                    self._decode_vectorized_input_updates
                ),
            }
            if self._decode_graph_last_failure:
                stats['decode_cuda_graphs']['last_failure'] = (
                    self._decode_graph_last_failure
                )
            if self._decode_unrolled_graph_last_failure:
                stats['decode_cuda_graphs']['unrolled_token_burst_last_failure'] = (
                    self._decode_unrolled_graph_last_failure
                )
        if 'decode_cuda_graphs' not in stats:
            # Keep disabled eager runs auditable.  Previously this all-zero
            # block was omitted, which made "disabled" indistinguishable from
            # missing instrumentation in publication artifacts.
            stats['decode_cuda_graphs'] = {
                'enabled': False,
                'prefer_step': bool(self._decode_cuda_graph_prefer_step),
                'request_scheduler_reuse_enabled': bool(
                    self._request_scheduler_reuse_enabled
                ),
                'request_scheduler_reused': False,
                'request_scheduler_reuse_count': 0,
                'captures': 0,
                'replays': 0,
                'warmups': 0,
                'failures': 0,
            }
        return stats
