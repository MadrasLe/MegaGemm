"""
🚀 MegaGemm Inference Engine
-----------------------------
Simple, fast LLM inference with paged attention and continuous batching.

Usage:
    from megagemm.engine import InferenceEngine

    engine = InferenceEngine("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    outputs = engine.generate(["Hello world", "Once upon a time"])

Author: Gabriel Yogi
"""

import os
import json
import torch
import time
import gc
from pathlib import Path
from typing import Any, List, Optional, Union, Dict, Tuple

from ..models.loader import load_from_hf
from ..models.llama import LlamaConfig
from ..models.runtime_policy import policy_bool, policy_rows
from ..models.mgx import (
    MGXFormatError,
    _collect_tokenizer_hashes,
    _sha256_text,
    attach_session_state_to_mgx,
    extract_session_state_from_mgx,
    is_mgx_path,
    load_from_mgx,
)
from .kv_cache import BlockManager, TieredBlockManager
from .sampling import sample_logits
from .scheduler import Scheduler, Request, RequestStatus
from .deterministic import enable_deterministic_mode, is_deterministic
from .xai import (
    XAIReport, GenerationStep, TokenPrediction,
    extract_top_k_predictions, compute_confidence, compute_entropy,
)
from .monitor import InferenceMonitor, RequestRecord
from .dashboard import DashboardServer

__all__ = ['InferenceEngine']


def _get_single_decode_burst() -> int:
    raw = os.environ.get("MEGAGEMM_MULTI_STEP_BURST_SINGLE", "").strip()
    if not raw:
        raw = os.environ.get("MEGAGEMM_MULTI_STEP_BURST", "").strip()
    if not raw:
        raw = "16"
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


def _request_scheduler_reuse_for_batch(model: Any, batch_size: int) -> bool:
    """Resolve global/env reuse while allowing policy-scoped batch sizes."""
    env_name = "MEGAGEMM_REUSE_REQUEST_SCHEDULER"
    if env_name in os.environ:
        return policy_bool(
            model,
            env_name,
            "reuse_request_scheduler",
            default=False,
        )
    if policy_bool(
        model,
        env_name,
        "reuse_request_scheduler",
        default=False,
    ):
        return True
    return int(batch_size) in policy_rows(
        model,
        env_name,
        "reuse_request_scheduler_batches",
    )


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalize_token_id_set(*token_specs: Any) -> set[int]:
    """Normalize int/list/tensor token ids into a hashable stop-token set."""
    normalized: set[int] = set()
    for token_spec in token_specs:
        if token_spec is None:
            continue
        if torch.is_tensor(token_spec):
            if token_spec.numel() == 0:
                continue
            token_spec = token_spec.detach().cpu().view(-1).tolist()
        if isinstance(token_spec, (list, tuple, set)):
            for item in token_spec:
                normalized.update(_normalize_token_id_set(item))
            continue
        try:
            normalized.add(int(token_spec))
        except (TypeError, ValueError):
            continue
    return normalized


def _read_chat_template_for_tokenizer(model_dir: Path) -> Optional[str]:
    template_path = model_dir / "chat_template.jinja"
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    tokenizer_cfg = _read_json_if_exists(model_dir / "tokenizer_config.json")
    template = tokenizer_cfg.get("chat_template")
    return str(template) if template else None


def _tokenizer_bootstrap_from_snapshot(model_dir: Path, mgx_manifest: Optional[dict] = None) -> dict[str, Any]:
    bootstrap = dict((mgx_manifest or {}).get("tokenizer_init") or {})
    if bootstrap:
        return bootstrap

    tokenizer_cfg = _read_json_if_exists(model_dir / "tokenizer_config.json")
    special_tokens_map = _read_json_if_exists(model_dir / "special_tokens_map.json")

    bootstrap = {}
    for key in (
        "bos_token",
        "eos_token",
        "unk_token",
        "pad_token",
        "sep_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ):
        if key in special_tokens_map:
            bootstrap[key] = special_tokens_map[key]
        elif key in tokenizer_cfg:
            bootstrap[key] = tokenizer_cfg[key]
    for key in (
        "model_max_length",
        "padding_side",
        "truncation_side",
        "clean_up_tokenization_spaces",
    ):
        if key in tokenizer_cfg:
            bootstrap[key] = tokenizer_cfg[key]
    chat_template = _read_chat_template_for_tokenizer(model_dir)
    if chat_template is not None:
        bootstrap["chat_template"] = chat_template
    if (model_dir / "tokenizer.json").exists():
        bootstrap["tokenizer_file"] = "tokenizer.json"
    return bootstrap


def _materialize_special_token(token_spec: Any):
    if token_spec is None:
        return None
    if isinstance(token_spec, str):
        return token_spec
    if isinstance(token_spec, dict):
        content = token_spec.get("content")
        if content is None:
            return None
        try:
            from transformers import AddedToken
            return AddedToken(
                str(content),
                single_word=bool(token_spec.get("single_word", False)),
                lstrip=bool(token_spec.get("lstrip", False)),
                rstrip=bool(token_spec.get("rstrip", False)),
                normalized=bool(token_spec.get("normalized", True)),
                special=bool(token_spec.get("special", True)),
            )
        except Exception:
            return str(content)
    return str(token_spec)


def _special_token_text(token_spec: Any) -> Optional[str]:
    if token_spec is None:
        return None
    if isinstance(token_spec, str):
        return token_spec
    if isinstance(token_spec, dict):
        content = token_spec.get("content")
        return None if content is None else str(content)
    content = getattr(token_spec, "content", None)
    if content is not None:
        return str(content)
    return str(token_spec)


def _apply_tokenizer_bootstrap(tokenizer, bootstrap: dict[str, Any]) -> None:
    for key in ("bos_token", "eos_token", "unk_token", "pad_token", "sep_token", "cls_token", "mask_token"):
        if key in bootstrap:
            value = _materialize_special_token(bootstrap.get(key))
            if value is not None:
                setattr(tokenizer, key, value)

    if "additional_special_tokens" in bootstrap:
        tokens = [
            token for token in (
                _materialize_special_token(item)
                for item in bootstrap.get("additional_special_tokens", [])
            )
            if token is not None
        ]
        if tokens:
            tokenizer.additional_special_tokens = tokens

    if "model_max_length" in bootstrap:
        tokenizer.model_max_length = int(bootstrap["model_max_length"])
    if "padding_side" in bootstrap:
        tokenizer.padding_side = str(bootstrap["padding_side"])
    if "truncation_side" in bootstrap:
        tokenizer.truncation_side = str(bootstrap["truncation_side"])
    if "clean_up_tokenization_spaces" in bootstrap:
        tokenizer.clean_up_tokenization_spaces = bool(bootstrap["clean_up_tokenization_spaces"])
    if "chat_template" in bootstrap and bootstrap["chat_template"] is not None:
        tokenizer.chat_template = str(bootstrap["chat_template"])


def _raise_chat_template_exception(message: str):
    raise ValueError(str(message))


class _MGXUltraFastTokenizer:
    def __init__(
        self,
        backend,
        *,
        tokenizer_source: Path,
        bootstrap: dict[str, Any],
    ):
        self._backend = backend
        self.name_or_path = str(tokenizer_source)
        self.model_max_length = int(bootstrap.get("model_max_length", 10**30))
        self.padding_side = str(bootstrap.get("padding_side", "right"))
        self.truncation_side = str(bootstrap.get("truncation_side", "right"))
        self.clean_up_tokenization_spaces = bool(
            bootstrap.get("clean_up_tokenization_spaces", False)
        )
        self.chat_template = (
            str(bootstrap["chat_template"])
            if bootstrap.get("chat_template") is not None
            else None
        )
        self.bos_token = _special_token_text(bootstrap.get("bos_token"))
        self.eos_token = _special_token_text(bootstrap.get("eos_token"))
        self.unk_token = _special_token_text(bootstrap.get("unk_token"))
        self.pad_token = _special_token_text(bootstrap.get("pad_token"))
        self.sep_token = _special_token_text(bootstrap.get("sep_token"))
        self.cls_token = _special_token_text(bootstrap.get("cls_token"))
        self.mask_token = _special_token_text(bootstrap.get("mask_token"))
        self.additional_special_tokens = [
            token
            for token in (
                _special_token_text(item)
                for item in bootstrap.get("additional_special_tokens", [])
            )
            if token is not None
        ]
        self._chat_template_renderer = None
        if self.chat_template:
            from jinja2.sandbox import SandboxedEnvironment

            env = SandboxedEnvironment(
                autoescape=False,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            env.globals["raise_exception"] = _raise_chat_template_exception
            self._chat_template_renderer = env.from_string(self.chat_template)
        self._sync_special_token_ids()

    def _sync_special_token_ids(self) -> None:
        self.bos_token_id = self._lookup_token_id(self.bos_token)
        self.eos_token_id = self._lookup_token_id(self.eos_token)
        self.unk_token_id = self._lookup_token_id(self.unk_token)
        self.pad_token_id = self._lookup_token_id(self.pad_token)
        self.sep_token_id = self._lookup_token_id(self.sep_token)
        self.cls_token_id = self._lookup_token_id(self.cls_token)
        self.mask_token_id = self._lookup_token_id(self.mask_token)
        self.all_special_ids = [
            token_id
            for token_id in (
                self.bos_token_id,
                self.eos_token_id,
                self.unk_token_id,
                self.pad_token_id,
                self.sep_token_id,
                self.cls_token_id,
                self.mask_token_id,
                *(self._lookup_token_id(token) for token in self.additional_special_tokens),
            )
            if token_id is not None
        ]

    def _lookup_token_id(self, token: Optional[str]) -> Optional[int]:
        if token is None:
            return None
        token_id = self._backend.token_to_id(str(token))
        if token_id is None:
            return None
        return int(token_id)

    def encode(self, text, return_tensors=None, add_special_tokens=True):
        ids = self._backend.encode(
            str(text),
            add_special_tokens=bool(add_special_tokens),
        ).ids
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        flat_ids = [int(token_id) for token_id in ids]
        return self._backend.decode(
            flat_ids,
            skip_special_tokens=bool(skip_special_tokens),
        )

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    ):
        if self._chat_template_renderer is None:
            raise ValueError("Tokenizer does not define a chat template.")
        rendered = self._chat_template_renderer.render(
            messages=messages,
            add_generation_prompt=bool(add_generation_prompt),
            bos_token=self.bos_token or "",
            eos_token=self.eos_token or "",
            unk_token=self.unk_token or "",
            pad_token=self.pad_token or "",
            sep_token=self.sep_token or "",
            cls_token=self.cls_token or "",
            mask_token=self.mask_token or "",
            additional_special_tokens=self.additional_special_tokens,
            tools=kwargs.get("tools"),
            documents=kwargs.get("documents"),
        )
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


def _try_load_fast_tokenizer_from_snapshot(
    tokenizer_source: Path,
    bootstrap: dict[str, Any],
):
    tokenizer_file_name = bootstrap.get("tokenizer_file") or "tokenizer.json"
    tokenizer_file = tokenizer_source / tokenizer_file_name
    if not tokenizer_file.exists():
        return None, None

    try:
        from tokenizers import Tokenizer

        backend = Tokenizer.from_file(str(tokenizer_file))
        tokenizer = _MGXUltraFastTokenizer(
            backend,
            tokenizer_source=tokenizer_source,
            bootstrap=bootstrap,
        )
        return tokenizer, "mgx-ultrafast"
    except Exception as exc:
        print(
            "[MegaGemm][warn] MGX ultrafast tokenizer bootstrap failed; "
            f"falling back to PreTrainedTokenizerFast. Reason: {exc}"
        )

    try:
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
        _apply_tokenizer_bootstrap(tokenizer, bootstrap)
        tokenizer.name_or_path = str(tokenizer_source)
        return tokenizer, "mgx-fast"
    except Exception as exc:
        print(
            "[MegaGemm][warn] MGX fast tokenizer bootstrap failed; "
            f"falling back to AutoTokenizer. Reason: {exc}"
        )
        return None, None


_MGX_TOKENIZER_ASSET_PATTERNS = (
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
    "vocab.txt",
    "sentencepiece.bpe.model",
    "*.model",
    "config.json",
)


def _mgx_recorded_snapshot_revision(mgx_manifest: dict) -> Optional[str]:
    """Recover an exact HF commit from the snapshot path stored by MGX v1."""
    raw_path = (
        mgx_manifest.get("tokenizer_source_path")
        or mgx_manifest.get("source_snapshot_path")
    )
    if not raw_path:
        return None
    snapshot_path = Path(str(raw_path))
    revision = snapshot_path.name
    if snapshot_path.parent.name != "snapshots":
        return None
    if len(revision) < 7 or any(char not in "0123456789abcdefABCDEF" for char in revision):
        return None
    return revision


def _restore_mgx_tokenizer_snapshot(mgx_manifest: dict) -> Path:
    """Restore only tokenizer sidecars when an ephemeral HF cache disappeared."""
    source_model_id = str(mgx_manifest.get("source_model_id") or "").strip()
    if not source_model_id or os.path.isdir(source_model_id):
        raise MGXFormatError(
            "MGX tokenizer assets are missing locally and the artifact does not "
            "contain a recoverable Hugging Face model id."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise MGXFormatError(
            "MGX tokenizer assets are missing locally. Install huggingface_hub "
            "so MegaGemm can restore the tokenizer sidecars automatically."
        ) from exc

    revision = _mgx_recorded_snapshot_revision(mgx_manifest)
    try:
        restored = snapshot_download(
            repo_id=source_model_id,
            revision=revision,
            allow_patterns=list(_MGX_TOKENIZER_ASSET_PATTERNS),
        )
    except Exception as exc:
        revision_note = f" at revision {revision}" if revision else ""
        raise MGXFormatError(
            f"MGX tokenizer assets are missing locally and could not be restored "
            f"from {source_model_id}{revision_note}: {exc}"
        ) from exc

    restored_path = Path(restored)
    if not restored_path.is_dir() or not _candidate_tokenizer_files_for_engine(restored_path):
        raise MGXFormatError(
            f"Tokenizer restore for {source_model_id} completed without usable tokenizer assets."
        )
    print(
        "[MegaGemm] Restored MGX tokenizer assets from "
        f"{source_model_id}{f'@{revision}' if revision else ''}."
    )
    return restored_path


def _candidate_tokenizer_files_for_engine(model_dir: Path) -> list[Path]:
    """Return enough evidence that a tokenizer snapshot is locally usable."""
    names = (
        "tokenizer.json",
        "tokenizer.model",
        "sentencepiece.bpe.model",
        "vocab.json",
        "vocab.txt",
    )
    return [model_dir / name for name in names if (model_dir / name).is_file()]


def _load_tokenizer_for_engine(model_name: str, mgx_manifest: Optional[dict] = None):
    if mgx_manifest is not None:
        tokenizer_source = (
            mgx_manifest.get('tokenizer_source_path')
            or mgx_manifest.get('source_snapshot_path')
        )
        if not tokenizer_source or not os.path.isdir(tokenizer_source):
            tokenizer_source_path = _restore_mgx_tokenizer_snapshot(mgx_manifest)
            tokenizer_source = str(tokenizer_source_path)
        else:
            tokenizer_source_path = Path(tokenizer_source)
        bootstrap = _tokenizer_bootstrap_from_snapshot(tokenizer_source_path, mgx_manifest)
        tokenizer, tokenizer_loader_kind = _try_load_fast_tokenizer_from_snapshot(
            tokenizer_source_path,
            bootstrap,
        )
        if tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                raise ImportError(
                    "Install transformers for tokenizer: pip install transformers"
                )
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                local_files_only=True,
            )
            tokenizer_loader_kind = "mgx-auto"
        expected_hash = mgx_manifest.get('tokenizer_hash')
        hash_info = _collect_tokenizer_hashes(tokenizer_source_path) if expected_hash else {}
        actual_hash = hash_info.get("primary")
        actual_bundle_hash = hash_info.get("bundle")
        actual_core_hash = hash_info.get("core")
        expected_scheme = mgx_manifest.get("tokenizer_hash_scheme")
        expected_core_hash = mgx_manifest.get("tokenizer_core_hash")
        expected_bundle_hash = mgx_manifest.get("tokenizer_bundle_hash")
        if expected_hash:
            if expected_scheme == "core-v1":
                if actual_core_hash != expected_hash:
                    raise MGXFormatError(
                        "Tokenizer hash mismatch for MGX artifact. "
                        f"Expected core hash {expected_hash}, got {actual_core_hash}."
                    )
            elif expected_scheme == "bundle-v0":
                if actual_bundle_hash != expected_hash:
                    raise MGXFormatError(
                        "Tokenizer hash mismatch for MGX artifact. "
                        f"Expected bundle hash {expected_hash}, got {actual_bundle_hash}."
                    )
            else:
                expected_hashes = {
                    value for value in (expected_hash, expected_core_hash, expected_bundle_hash)
                    if value
                }
                actual_hashes = {
                    value for value in (actual_hash, actual_core_hash, actual_bundle_hash)
                    if value
                }
                if expected_hashes.isdisjoint(actual_hashes):
                    # Legacy artifacts created before tokenizer hash scheme stabilization
                    # may carry bundle hashes that are too sensitive to auxiliary files.
                    if actual_core_hash is not None:
                        print(
                            "[MegaGemm][warn] Legacy MGX tokenizer hash mismatch detected; "
                            "falling back to the current tokenizer source. Re-export the artifact "
                            "with the current MGX build to persist the stable tokenizer hash scheme."
                        )
                    else:
                        raise MGXFormatError(
                            "Tokenizer hash mismatch for MGX artifact. "
                            f"Expected one of {sorted(expected_hashes)}, "
                            f"got primary={actual_hash}, core={actual_core_hash}, bundle={actual_bundle_hash}."
                        )
        expected_chat_hash = mgx_manifest.get('chat_template_hash')
        actual_chat_hash = _sha256_text(getattr(tokenizer, 'chat_template', None))
        if expected_chat_hash != actual_chat_hash:
            raise MGXFormatError(
                "Chat template hash mismatch for MGX artifact. "
                f"Expected {expected_chat_hash}, got {actual_chat_hash}."
            )
    else:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "Install transformers for tokenizer: pip install transformers"
            )
        local_only = os.path.isdir(model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_only,
        )
        tokenizer_loader_kind = "auto-local" if local_only else "auto-hf"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(tokenizer, "_sync_special_token_ids"):
            tokenizer._sync_special_token_ids()
    return tokenizer, tokenizer_loader_kind


class InferenceEngine:
    """
    🔥 MegaGemm Inference Engine

    Lightweight LLM serving with:
    - Paged KV cache (no memory waste)
    - Continuous batching (concurrent multi-request serving)
    - Triton-accelerated decode attention
    - MegaGemm CUDA kernels (RMSNorm, SwiGLU, RoPE)
    - HuggingFace model loading

    Example:
        engine = InferenceEngine("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        # Single request
        print(engine.generate("The meaning of life is"))

        # Batch (continuous batching!)
        results = engine.generate_batch([
            "The capital of France is",
            "In machine learning,",
            "The theory of relativity",
        ])
    """

    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype = torch.float16,
        device: str = 'cuda',
        num_blocks: int = 0,
        block_size: int = 16,
        max_batch_size: int = 512,
        cache_dir: Optional[str] = None,
        n_gpu_layers: int = -1,
        offload_dir: Optional[str] = None,
        quantize: Optional[str] = None,
        kv_offload: bool = False,
        num_cpu_blocks: int = 0,
        gpu_window: int = 64,
        kv_alloc: str = 'auto',
        max_seq_len: int = 4096,
        monitor: bool = False,
        dashboard: bool = False,
        dashboard_port: int = 8080,
        deterministic: bool = False,
        seed: int = 42,
        mgx_verify_payload: Optional[bool] = None,
        mgx_prefer_payload_cache: Optional[bool] = None,
        mgx_payload_cache_dir: Optional[str] = None,
    ):
        """
        Initialize the inference engine.

        Args:
            model_name: HuggingFace model ID, local snapshot path, or .mgx artifact
            dtype: Model precision (float16 or bfloat16)
            device: Target device
            num_blocks: KV cache blocks on GPU (0=auto based on kv_alloc mode)
            block_size: Tokens per KV cache block
            cache_dir: HuggingFace cache directory
            n_gpu_layers: Layers on GPU (-1=all). Rest offloaded to CPU/disk.
            offload_dir: Directory for disk offload (None=CPU only)
            quantize: Quantization mode. Use 'int8' for streaming INT8 W8A16;
                'fp8' is retained as a legacy alias for that same INT8 path.
            kv_offload: Enable KV cache CPU offloading for longer contexts
            num_cpu_blocks: KV cache blocks on CPU (0=auto: use 50% of free RAM)
            gpu_window: Min blocks to keep on GPU per sequence (when kv_offload=True)
            kv_alloc: KV cache allocation strategy when num_blocks=0:
                'auto'   - Size to max_batch_size × max_seq_len, cap at 70% VRAM (default)
                'greedy' - Use 70% of free VRAM (for high-throughput serving)
            max_seq_len: Max sequence length for 'auto' mode (prompt + generation tokens)
            monitor: Enable inference monitoring
            dashboard: Start live monitoring dashboard HTTP server
            dashboard_port: Port for the dashboard server (default: 8080)
            deterministic: Enable deterministic inference mode. Guarantees bit-exact
                reproducible output across runs on the same hardware. ~10-15% perf overhead.
            seed: Random seed for deterministic mode (default: 42)
            mgx_verify_payload: Override MGX payload hash verification when loading
                compiled artifacts. None keeps the default/env-controlled behavior.
            mgx_prefer_payload_cache: Prefer a reusable extracted safetensors payload
                cache when available for `.mgx` artifacts.
            mgx_payload_cache_dir: Optional directory for MGX payload cache files.
        """
        self.device = device
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self._max_seq_len = max_seq_len
        self._deterministic = deterministic
        self._decode_multi_step_burst = _get_single_decode_burst()
        self._generate_cuda_graphs_explicit = bool(
            "MEGAGEMM_GENERATE_CUDA_GRAPHS" in os.environ
            or "MEGAGEMM_DECODE_CUDA_GRAPHS" in os.environ
        )
        self._generate_cuda_graphs = _env_bool(
            "MEGAGEMM_GENERATE_CUDA_GRAPHS",
            _env_bool("MEGAGEMM_DECODE_CUDA_GRAPHS", False),
        )
        self._prefill_cuda_graphs = _env_bool(
            "MEGAGEMM_PREFILL_CUDA_GRAPHS",
            False,
        )
        self._generate_multi_step_cuda_graphs = _env_bool(
            "MEGAGEMM_GENERATE_MULTI_STEP_CUDA_GRAPHS",
            True,
        )
        self._generate_step_cuda_graphs = _env_bool(
            "MEGAGEMM_GENERATE_STEP_CUDA_GRAPHS",
            True,
        )
        self._generate_gpu_token_chain = _env_bool(
            "MEGAGEMM_GENERATE_GPU_TOKEN_CHAIN",
            True,
        )
        self._generate_gpu_token_chain_allow_qwen3_moe = _env_bool(
            "MEGAGEMM_GENERATE_GPU_TOKEN_CHAIN_ALLOW_QWEN3_MOE",
            False,
        )
        self._generate_skip_token_materialization = _env_bool(
            "MEGAGEMM_GENERATE_SKIP_TOKEN_MATERIALIZATION",
            False,
        )
        self._generate_fused_argmax_step = _env_bool(
            "MEGAGEMM_GENERATE_FUSED_ARGMAX_STEP",
            True,
        )
        self._generate_direct_graph_inputs = _env_bool(
            "MEGAGEMM_GENERATE_DIRECT_GRAPH_INPUTS",
            True,
        )
        self._generate_persistent_step_graph_inputs = _env_bool(
            "MEGAGEMM_GENERATE_PERSISTENT_STEP_GRAPH_INPUTS",
            False,
        )
        self._generate_stable_max_blocks = _env_bool(
            "MEGAGEMM_GENERATE_CUDA_GRAPHS_STABLE_MAX_BLOCKS",
            _env_bool("MEGAGEMM_DECODE_CUDA_GRAPHS_STABLE_MAX_BLOCKS", True),
        )
        self._generate_graph_log_limit = max(
            0,
            _env_int(
                "MEGAGEMM_GENERATE_CUDA_GRAPHS_LOG_LIMIT",
                _env_int("MEGAGEMM_DECODE_CUDA_GRAPHS_LOG_LIMIT", 6),
            ),
        )
        self._generate_graph_log_count = 0
        self._generate_multi_step_graph_states: Dict[tuple, dict] = {}
        self._last_generated_ids: List[int] = []
        self._last_generation_metrics: Dict[str, float] = {}
        init_timing: Dict[str, object] = {
            "model_ref": model_name,
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "quantize": quantize or "none",
            "kv_offload": bool(kv_offload),
            "mgx_verify_payload": mgx_verify_payload,
            "mgx_prefer_payload_cache": mgx_prefer_payload_cache,
            "mgx_payload_cache_dir": mgx_payload_cache_dir,
        }
        total_start = time.perf_counter()

        # Deterministic mode: guarantee bit-exact reproducibility
        if deterministic:
            enable_deterministic_mode(seed)
            print(f"[MegaGemm] Deterministic mode ON (seed={seed}) - bit-exact reproducible output")

        # Load model
        self._model_name = model_name
        mgx_manifest = None
        phase_start = time.perf_counter()
        if is_mgx_path(model_name):
            self.model = load_from_mgx(
                model_name,
                device=device,
                dtype_override=dtype,
                verify_payload_hash=mgx_verify_payload,
                prefer_payload_cache=mgx_prefer_payload_cache,
                payload_cache_dir=mgx_payload_cache_dir,
            )
            mgx_manifest = getattr(self.model, "_mgx_manifest", None)
            init_timing["model_loader_kind"] = "mgx"
        else:
            self.model = load_from_hf(
                model_name, dtype, device, cache_dir,
                n_gpu_layers=n_gpu_layers,
                offload_dir=offload_dir,
                quantize=quantize,
            )
            init_timing["model_loader_kind"] = "hf"
        init_timing["model_load_seconds"] = time.perf_counter() - phase_start
        init_timing["model_loader_timing"] = getattr(self.model, "_load_timing", None)
        self.config = self.model.config
        if (
            not self._generate_cuda_graphs_explicit
            and device == "cuda"
            and getattr(self.config, "model_type", "") == "gemma4_text"
            and bool(getattr(self.config, "enable_moe_block", False))
        ):
            self._generate_cuda_graphs = True
            init_timing["generate_cuda_graphs_auto"] = "gemma4_moe"
        if hasattr(self.model, "set_rope_cache_max_seq_len"):
            phase_start = time.perf_counter()
            self.model.set_rope_cache_max_seq_len(max_seq_len, device=device)
            if device == 'cuda' and torch.cuda.is_available():
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            init_timing["rope_cache_resize_seconds"] = time.perf_counter() - phase_start

        # Load tokenizer
        phase_start = time.perf_counter()
        self.tokenizer, tokenizer_loader_kind = _load_tokenizer_for_engine(model_name, mgx_manifest)
        init_timing["tokenizer_load_seconds"] = time.perf_counter() - phase_start
        init_timing["tokenizer_loader_kind"] = tokenizer_loader_kind

        # ── Auto-compute block counts from available memory ──
        phase_start = time.perf_counter()
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        num_layers = self.config.num_hidden_layers
        layer_types = getattr(self.config, 'layer_types', None)
        per_layer_num_kv_heads = getattr(self.config, 'per_layer_num_kv_heads', None) or None
        per_layer_head_dims = getattr(self.config, 'per_layer_head_dims', None) or None
        kv_layer_sources = None
        kv_layer_indices = None
        if self.config.model_type == 'gemma4_text':
            if kv_offload:
                raise NotImplementedError(
                    "Gemma 4 KV offload is disabled in the first text-only implementation "
                    "because KV layers can have heterogeneous head dimensions and sharing."
                )
            kv_layer_indices = list(getattr(self.config, 'kv_cache_layer_indices', []) or [])
            kv_layer_sources = {
                layer_idx: source_idx
                for layer_idx, source_idx in enumerate(getattr(self.config, 'kv_share_sources', []) or [])
                if source_idx is not None
            }
        elif layer_types and len(layer_types) == num_layers:
            kv_layer_indices = [
                i for i, layer_type in enumerate(layer_types)
                if layer_type != 'linear_attention'
            ]
            if not kv_layer_indices:
                kv_layer_indices = None
        kv_layers_for_cache = (
            len(kv_layer_indices) if kv_layer_indices is not None else num_layers
        )
        if kv_layers_for_cache != num_layers:
            print(
                f"[MegaGemm] KV cache layer-aware mode: {kv_layers_for_cache}/{num_layers} "
                "layers allocate paged KV."
            )
        # Bytes per block = num_layers × 2(K+V) × num_kv_heads × block_size × head_dim × dtype_bytes
        dtype_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4
        if per_layer_num_kv_heads is not None and per_layer_head_dims is not None:
            layer_indices_for_bytes = kv_layer_indices if kv_layer_indices is not None else list(range(num_layers))
            bytes_per_block = sum(
                2
                * int(per_layer_num_kv_heads[layer_idx])
                * block_size
                * int(per_layer_head_dims[layer_idx])
                * dtype_bytes
                for layer_idx in layer_indices_for_bytes
            )
        else:
            bytes_per_block = kv_layers_for_cache * 2 * num_kv_heads * block_size * head_dim * dtype_bytes

        if num_blocks == 0 and device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
            # Weight loading can leave large freed temporaries in PyTorch's CUDA
            # caching allocator. Release them before sizing the KV cache so the
            # driver-level free-memory reading reflects memory the cache could
            # otherwise reuse.
            gc.collect()
            torch.cuda.empty_cache()
            free_vram = torch.cuda.mem_get_info()[0]  # bytes free
            init_timing["cuda_memory_before_kv"] = {
                "free_gb": free_vram / 1e9,
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            }
            max_vram_blocks = max(64, int(free_vram * 0.70) // bytes_per_block)

            if kv_alloc == 'auto':
                # Smart: size to actual workload (batch × max_seq_len)
                import math
                blocks_per_seq = math.ceil(max_seq_len / block_size)
                need_blocks = blocks_per_seq * max_batch_size
                num_blocks = min(need_blocks, max_vram_blocks)
                print(f"[MegaGemm] Auto KV alloc: {blocks_per_seq} blocks/seq x {max_batch_size} batch "
                      f"= {need_blocks} needed, capped to {num_blocks} "
                      f"({num_blocks * bytes_per_block / 1e9:.1f}GB of {free_vram/1e9:.1f}GB free)")
            else:
                # Greedy: grab 70% of free VRAM (high-throughput serving)
                num_blocks = max_vram_blocks
                print(f"[MegaGemm] Greedy KV alloc: {free_vram/1e9:.1f}GB free -> "
                      f"{num_blocks} blocks ({num_blocks * bytes_per_block / 1e9:.1f}GB)")
        elif num_blocks == 0:
            num_blocks = 4096  # CPU-only fallback

        if kv_offload and num_cpu_blocks == 0:
            try:
                import psutil
                free_ram = psutil.virtual_memory().available
            except ImportError:
                # Fallback: read /proc/meminfo on Linux
                try:
                    with open('/proc/meminfo') as f:
                        for line in f:
                            if 'MemAvailable' in line:
                                free_ram = int(line.split()[1]) * 1024  # kB → bytes
                                break
                        else:
                            free_ram = 16 * 1024**3  # 16GB fallback
                except Exception:
                    free_ram = 16 * 1024**3
            # Use 50% of free RAM for CPU KV blocks
            cpu_budget = int(free_ram * 0.50)
            num_cpu_blocks = max(64, cpu_budget // bytes_per_block)
            print(f"[MegaGemm] Auto CPU blocks: {free_ram/1e9:.1f}GB free -> "
                  f"{num_cpu_blocks} blocks ({num_cpu_blocks * bytes_per_block / 1e9:.1f}GB)")
        elif kv_offload and num_cpu_blocks == 0:
            num_cpu_blocks = 2048  # fallback
        init_timing["kv_planning_seconds"] = time.perf_counter() - phase_start

        # Create KV cache block manager
        self.kv_offload = kv_offload
        phase_start = time.perf_counter()
        if kv_offload:
            self.block_manager = TieredBlockManager(
                num_layers=num_layers,
                num_gpu_blocks=num_blocks,
                num_cpu_blocks=num_cpu_blocks,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
                gpu_window=gpu_window,
                kv_layer_indices=kv_layer_indices,
                per_layer_num_kv_heads=per_layer_num_kv_heads,
                per_layer_head_dims=per_layer_head_dims,
                kv_layer_sources=kv_layer_sources,
            )
            print(f"[MegaGemm] Engine ready! KV Cache (tiered): {self.block_manager}")
        else:
            self.block_manager = BlockManager(
                num_layers=num_layers,
                num_blocks=num_blocks,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
                kv_layer_indices=kv_layer_indices,
                per_layer_num_kv_heads=per_layer_num_kv_heads,
                per_layer_head_dims=per_layer_head_dims,
                kv_layer_sources=kv_layer_sources,
            )
            print(f"[MegaGemm] Engine ready! KV Cache: {self.block_manager}")
        init_timing["block_manager_init_seconds"] = time.perf_counter() - phase_start
        init_timing["num_blocks"] = int(getattr(self.block_manager, "num_blocks", num_blocks))
        if kv_offload:
            init_timing["num_cpu_blocks"] = int(num_cpu_blocks)
        self._seq_counter = 0

        # Monitoring
        self._monitor_enabled = monitor or dashboard
        self._monitor = InferenceMonitor() if self._monitor_enabled else None
        self._quantize_mode = quantize or 'fp16'

        # Dashboard
        self._dashboard = None
        phase_start = time.perf_counter()
        if dashboard:
            self._dashboard = DashboardServer(self._monitor, port=dashboard_port)
            self._dashboard.start()
        init_timing["dashboard_init_seconds"] = time.perf_counter() - phase_start
        init_timing["total_seconds"] = time.perf_counter() - total_start
        self._init_timing = init_timing
        self._seq_token_ids: Dict[int, List[int]] = {}
        self._seq_pending_logits: Dict[int, torch.Tensor] = {}

    def _next_seq_id(self) -> int:
        self._seq_counter += 1
        return self._seq_counter

    def get_init_timing(self) -> Dict[str, object]:
        """Return initialization timing captured during engine construction."""
        return dict(getattr(self, "_init_timing", {}))

    def _clear_sequence_runtime_state(self, seq_id: int) -> None:
        self._seq_token_ids.pop(seq_id, None)
        self._seq_pending_logits.pop(seq_id, None)

    def free_sequence(self, seq_id: int) -> None:
        """Free a sequence from the KV cache and drop any continuation metadata."""
        self.block_manager.free_sequence(seq_id)
        self._clear_sequence_runtime_state(seq_id)

    def _clone_pending_logits(self, logits: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if logits is None:
            return None
        pending = logits.detach()
        if pending.ndim == 3:
            pending = pending[:, -1, :]
        if pending.ndim == 2:
            pending = pending[0]
        if pending.ndim != 1:
            raise ValueError(
                f"Expected 1D pending logits after normalization, got shape {tuple(pending.shape)}"
            )
        return pending.to(device="cpu", dtype=self.dtype).contiguous()

    def _set_sequence_runtime_state(
        self,
        seq_id: int,
        *,
        token_ids: Optional[List[int]] = None,
        pending_next_logits: Optional[torch.Tensor] = None,
    ) -> None:
        if token_ids is not None:
            self._seq_token_ids[seq_id] = [int(token_id) for token_id in token_ids]
        if pending_next_logits is not None:
            self._seq_pending_logits[seq_id] = self._clone_pending_logits(pending_next_logits)
        else:
            self._seq_pending_logits.pop(seq_id, None)

    def _prepare_prompt_inputs(self, prompt: str) -> Tuple[str, torch.Tensor]:
        """
        Format a prompt with the tokenizer chat template and return token IDs on engine device.
        """
        formatted_prompt = prompt
        bos = self.tokenizer.bos_token
        already_formatted = bool(bos and prompt.startswith(bos))

        if (
            not already_formatted
            and hasattr(self.tokenizer, 'chat_template')
            and self.tokenizer.chat_template
        ):
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                formatted_prompt = prompt

        add_special = not (bos and formatted_prompt.startswith(bos))
        input_ids = self.tokenizer.encode(
            formatted_prompt,
            return_tensors='pt',
            add_special_tokens=add_special,
        ).to(self.device)
        return formatted_prompt, input_ids

    @torch.inference_mode()
    def prefill_context(
        self,
        prompt: str,
        *,
        seq_id: Optional[int] = None,
        max_new_tokens: int = 0,
    ) -> Dict[str, object]:
        """
        Prefill a prompt into KV cache and retain the pending next-token logits.

        Unlike `generate(...)`, this keeps the sequence alive so it can be continued later
        with `generate_from_context(...)`, saved into MGX, or captured by MGX Prophet.
        """
        if seq_id is None:
            seq_id = self._next_seq_id()

        formatted_prompt, input_ids = self._prepare_prompt_inputs(prompt)
        prompt_len = int(input_ids.shape[1])
        total_tokens = prompt_len + max(1, int(max_new_tokens))
        positions = torch.arange(prompt_len, device=self.device).unsqueeze(0)

        self.block_manager.allocate_sequence(seq_id, total_tokens)
        try:
            if self.kv_offload:
                self.block_manager.ensure_blocks_on_gpu([seq_id])

            prefill_result = self.model.prefill(
                input_ids,
                positions,
                self.block_manager,
                seq_id,
                last_token_only=True,
            )
            logits = prefill_result[0] if isinstance(prefill_result, tuple) else prefill_result

            if self.kv_offload:
                self.block_manager.evict_cold_blocks([seq_id])
        except Exception:
            self.free_sequence(seq_id)
            raise

        self._set_sequence_runtime_state(
            seq_id,
            token_ids=input_ids[0].tolist(),
            pending_next_logits=logits[:, -1, :],
        )

        return {
            "seq_id": seq_id,
            "prompt": prompt,
            "formatted_prompt": formatted_prompt,
            "prompt_len": prompt_len,
            "seq_len": int(self.block_manager.seq_lens[seq_id]),
            "pending_next_logits_shape": list(self._seq_pending_logits[seq_id].shape),
        }

    @torch.inference_mode()
    def _ensure_sequence_capacity(
        self,
        seq_id: int,
        additional_tokens: int,
    ) -> None:
        """
        Ensure a live sequence has enough block-table capacity for more tokens.

        Restore paths usually pre-allocate headroom, but Prophet prefix replay can
        repair a longer prompt than the restored snapshot originally reserved for.
        """
        if seq_id not in self.block_manager.block_tables:
            raise ValueError(f"Sequence {seq_id} not found")

        additional_tokens = int(additional_tokens)
        if additional_tokens <= 0:
            return

        current_len = int(self.block_manager.seq_lens[seq_id])
        target_len = current_len + additional_tokens
        block_size = int(self.block_manager.block_size)
        blocks_needed = max(1, (target_len + block_size - 1) // block_size)
        blocks = self.block_manager.block_tables[seq_id]
        if len(blocks) >= blocks_needed:
            return

        while len(blocks) < blocks_needed:
            self.block_manager.allocate_block(seq_id)
            blocks = self.block_manager.block_tables[seq_id]

        # Rebuild cached block tables on the next decode after changing capacity.
        if hasattr(self.block_manager, "_block_table_tensor"):
            self.block_manager._block_table_tensor = None

    @torch.inference_mode()
    def truncate_context(self, seq_id: int, new_seq_len: int) -> Dict[str, object]:
        """
        Truncate a live/restored context back to a shorter token prefix.

        This is primarily used by MGX Prophet when a restored snapshot shares a
        long common prefix with the incoming prompt but contains a different
        suffix that must be rolled back before replay.
        """
        if seq_id not in self.block_manager.block_tables:
            raise ValueError(f"Sequence {seq_id} not found")

        current_len = int(self.block_manager.seq_lens[seq_id])
        self.block_manager.truncate_sequence(seq_id, int(new_seq_len))

        token_history = self._seq_token_ids.get(seq_id)
        if token_history is not None:
            self._seq_token_ids[seq_id] = token_history[: int(new_seq_len)]
        self._seq_pending_logits.pop(seq_id, None)

        return {
            "seq_id": seq_id,
            "previous_seq_len": current_len,
            "seq_len": int(self.block_manager.seq_lens[seq_id]),
            "pending_next_logits_available": False,
        }

    @torch.inference_mode()
    def replay_tokens_into_context(
        self,
        seq_id: int,
        token_ids: List[int],
    ) -> Dict[str, object]:
        """
        Replay known tokens into an existing context using the best deterministic route.

        Unlike `generate_from_context(...)`, this does not sample. It appends the
        provided token IDs exactly, updating KV cache and the pending next-token
        logits as if the full prompt had been processed in one pass.
        """
        if seq_id not in self.block_manager.block_tables:
            raise ValueError(f"Sequence {seq_id} not found")

        forced_tokens = [int(token_id) for token_id in token_ids]
        token_history = list(self._seq_token_ids.get(seq_id, []))
        pending_logits = self._seq_pending_logits.get(seq_id)

        if not forced_tokens:
            return {
                "seq_id": seq_id,
                "token_ids": [],
                "seq_len": int(self.block_manager.seq_lens[seq_id]),
                "pending_next_logits_available": pending_logits is not None,
            }

        self._ensure_sequence_capacity(seq_id, len(forced_tokens))

        replay_mode = "decode_step_loop"
        if len(forced_tokens) > 1 and hasattr(self.model, "prefill_suffix"):
            suffix_input = torch.tensor(
                [forced_tokens],
                dtype=torch.long,
                device=self.device,
            )
            start_pos = int(self.block_manager.seq_lens[seq_id])
            suffix_positions = torch.arange(
                start_pos,
                start_pos + len(forced_tokens),
                dtype=torch.long,
                device=self.device,
            )
            suffix_positions = suffix_positions.unsqueeze(0)

            logits = None

            if self.kv_offload:
                self.block_manager.ensure_blocks_on_gpu([seq_id])

            try:
                prefill_result = self.model.prefill_suffix(
                    suffix_input,
                    suffix_positions,
                    self.block_manager,
                    seq_id,
                )
                logits = prefill_result[0] if isinstance(prefill_result, tuple) else prefill_result
            except NotImplementedError:
                logits = None

            if self.kv_offload:
                self.block_manager.evict_cold_blocks([seq_id])

            if logits is not None:
                token_history.extend(forced_tokens)
                pending_logits = logits[:, -1, :]
                replay_mode = "suffix_prefill"

        if replay_mode == "decode_step_loop" and len(forced_tokens) > 1 and hasattr(self.model, "decode_multi_step"):
            decode_input = torch.tensor(
                [[forced_tokens[0]]],
                dtype=torch.long,
                device=self.device,
            )
            decode_pos = torch.tensor(
                [[int(self.block_manager.seq_lens[seq_id])]],
                dtype=torch.long,
                device=self.device,
            )
            forced_next = torch.tensor(
                [forced_tokens[1:]],
                dtype=torch.long,
                device=self.device,
            )

            if self.kv_offload:
                self.block_manager.ensure_blocks_on_gpu([seq_id])

            decode_result = self.model.decode_multi_step(
                decode_input,
                decode_pos,
                self.block_manager,
                [seq_id],
                num_steps=len(forced_tokens),
                forced_next_token_ids=forced_next,
            )
            logits = decode_result[1] if isinstance(decode_result, tuple) else None

            if self.kv_offload:
                self.block_manager.evict_cold_blocks([seq_id])

            if logits is None:
                raise RuntimeError(
                    "decode_multi_step replay did not return final logits for pending_next_logits"
                )

            token_history.extend(forced_tokens)
            pending_logits = logits[:, -1, :]
            replay_mode = "teacher_forced_multi_step"

        if replay_mode == "decode_step_loop":
            decode_input = torch.empty(1, 1, dtype=torch.long, device=self.device)
            decode_pos = torch.empty(1, 1, dtype=torch.long, device=self.device)

            for token_id in forced_tokens:
                decode_input.fill_(token_id)
                decode_pos.fill_(int(self.block_manager.seq_lens[seq_id]))

                if self.kv_offload:
                    self.block_manager.ensure_blocks_on_gpu([seq_id])

                decode_result = self.model.decode_step(
                    decode_input,
                    decode_pos,
                    self.block_manager,
                    [seq_id],
                )
                logits = decode_result[0] if isinstance(decode_result, tuple) else decode_result

                if self.kv_offload:
                    self.block_manager.evict_cold_blocks([seq_id])

                token_history.append(token_id)
                pending_logits = logits[:, -1, :]

        self._seq_token_ids[seq_id] = token_history
        if pending_logits is not None:
            self._seq_pending_logits[seq_id] = self._clone_pending_logits(pending_logits)
        else:
            self._seq_pending_logits.pop(seq_id, None)

        return {
            "seq_id": seq_id,
            "token_ids": forced_tokens,
            "seq_len": int(self.block_manager.seq_lens[seq_id]),
            "pending_next_logits_available": seq_id in self._seq_pending_logits,
            "replay_mode": replay_mode,
        }

    @torch.inference_mode()
    def generate_from_context(
        self,
        seq_id: int,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_token_ids: Optional[List[int]] = None,
    ) -> Dict[str, object]:
        """
        Continue generation from a live or restored sequence snapshot.

        Requires the sequence to carry `pending_next_logits`, which are produced by
        `prefill_context(...)` or restored from a snapshot captured from such a context.
        """
        if seq_id not in self.block_manager.block_tables:
            raise ValueError(f"Sequence {seq_id} not found")
        if seq_id not in self._seq_pending_logits:
            raise ValueError(
                f"Sequence {seq_id} is missing pending_next_logits. "
                "Capture it with prefill_context(...) before trying to continue generation."
            )

        token_history = list(self._seq_token_ids.get(seq_id, []))
        pending_logits = self._seq_pending_logits[seq_id].to(self.device).unsqueeze(0)
        generated_ids: List[int] = []
        self._ensure_sequence_capacity(seq_id, max(0, int(max_new_tokens)))

        stop_set = _normalize_token_id_set(stop_token_ids, self.tokenizer.eos_token_id)

        decode_input = torch.empty(1, 1, dtype=torch.long, device=self.device)
        decode_pos = torch.empty(1, 1, dtype=torch.long, device=self.device)
        stopped = False

        for _ in range(max_new_tokens):
            past_tokens = None
            if repetition_penalty != 1.0 and token_history:
                past_tokens = torch.tensor([token_history], dtype=torch.long, device=self.device)

            next_token_id = sample_logits(
                pending_logits.clone(),
                temperature,
                top_k,
                top_p,
                repetition_penalty=repetition_penalty,
                past_tokens=past_tokens,
            ).item()
            generated_ids.append(next_token_id)

            if next_token_id in stop_set:
                stopped = True
                break

            decode_input.fill_(next_token_id)
            decode_pos.fill_(int(self.block_manager.seq_lens[seq_id]))

            if self.kv_offload:
                self.block_manager.ensure_blocks_on_gpu([seq_id])

            decode_result = self.model.decode_step(
                decode_input,
                decode_pos,
                self.block_manager,
                [seq_id],
            )
            logits = decode_result[0] if isinstance(decode_result, tuple) else decode_result

            if self.kv_offload:
                self.block_manager.evict_cold_blocks([seq_id])

            token_history.append(int(next_token_id))
            pending_logits = logits[:, -1, :]

        self._seq_token_ids[seq_id] = token_history
        if stopped:
            self._seq_pending_logits.pop(seq_id, None)
        else:
            self._seq_pending_logits[seq_id] = self._clone_pending_logits(pending_logits)

        return {
            "seq_id": seq_id,
            "text": self.tokenizer.decode(generated_ids, skip_special_tokens=True),
            "token_ids": generated_ids,
            "stopped": stopped,
            "pending_next_logits_available": seq_id in self._seq_pending_logits,
            "seq_len": int(self.block_manager.seq_lens[seq_id]),
        }

    def _log_generate_graph(self, message: str) -> None:
        if self._generate_graph_log_count >= self._generate_graph_log_limit:
            return
        self._generate_graph_log_count += 1
        print(f"  Generate CUDA Graph: {message}")

    def _single_prefill_graph_is_eligible(
        self,
        input_ids: torch.Tensor,
        logit_lens: Union[bool, int],
    ) -> bool:
        if not self._prefill_cuda_graphs or logit_lens:
            return False
        if str(self.device).split(":", 1)[0] != "cuda" or not torch.cuda.is_available():
            return False
        if type(self.block_manager) is not BlockManager or self.kv_offload:
            return False
        if getattr(self.model, "_offloader", None) is not None:
            return False
        if os.environ.get("MEGAGEMM_PREFILL_TIMING", "").strip().lower() in {
            "1", "true", "yes", "on",
        }:
            return False
        if os.environ.get("MEGAGEMM_PROFILE_PREFILL", "").strip() == "1":
            return False
        checker = getattr(self.model, "prefill_cuda_graph_eligible", None)
        if not callable(checker) or not hasattr(self.model, "prefill_packed_graph"):
            return False
        if not hasattr(self.block_manager, "compute_kv_mapping"):
            return False
        try:
            return bool(
                checker(
                    num_seqs=int(input_ids.shape[0]),
                    total_tokens=int(input_ids.shape[1]),
                    dtype=self.model.embed_tokens.weight.dtype,
                    device_type=input_ids.device.type,
                    device_name=torch.cuda.get_device_name(input_ids.device),
                )
            )
        except Exception:
            return False

    def _run_single_prefill_graph_or_eager(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_id: int,
        logit_lens: Union[bool, int],
    ) -> torch.Tensor:
        def eager_prefill():
            return self.model.prefill(
                input_ids,
                positions,
                self.block_manager,
                seq_id,
                logit_lens=logit_lens,
                last_token_only=True,
            )

        if not self._single_prefill_graph_is_eligible(input_ids, logit_lens):
            return eager_prefill()

        getter = getattr(self.model, "get_prefill_cuda_graph_store", None)
        if not callable(getter):
            return eager_prefill()
        store = getter(self.block_manager)
        key = (
            "single_generate",
            int(input_ids.shape[0]),
            int(input_ids.shape[1]),
            str(input_ids.device),
            str(self.model.embed_tokens.weight.dtype),
        )
        failed_keys = store.setdefault("failed_keys", {})
        if failed_keys.get(key):
            return eager_prefill()

        warm_keys = store.setdefault("warm_keys", set())
        if key not in warm_keys:
            warm_keys.add(key)
            store["warmups"] = int(store.get("warmups", 0)) + 1
            self._log_generate_graph(
                f"prefill warmup tokens={int(input_ids.shape[1])}"
            )
            return eager_prefill()

        cu_seqlens = torch.tensor(
            [0, int(input_ids.shape[1])],
            dtype=torch.int32,
            device=input_ids.device,
        )
        kv_phys, kv_offs = self.block_manager.compute_kv_mapping(
            [seq_id], cu_seqlens, input_ids.device,
        )
        state = store.setdefault("buckets", {}).get(key)
        if state is not None:
            state["input_ids"].copy_(input_ids)
            state["cu_seqlens"].copy_(cu_seqlens)
            state["kv_phys"].copy_(kv_phys)
            state["kv_offs"].copy_(kv_offs)
            state["graph"].replay()
            self.block_manager.advance_seq_len(seq_id, int(input_ids.shape[1]))
            store["replays"] = int(store.get("replays", 0)) + 1
            return state["logits"]

        graph_input_ids = input_ids.clone()
        graph_cu_seqlens = cu_seqlens.clone()
        graph_kv_phys = kv_phys.clone()
        graph_kv_offs = kv_offs.clone()
        try:
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
            state = {
                "graph": graph,
                "input_ids": graph_input_ids,
                "cu_seqlens": graph_cu_seqlens,
                "kv_phys": graph_kv_phys,
                "kv_offs": graph_kv_offs,
                "logits": logits,
            }
            # Capture records the CUDA work but does not populate the static
            # outputs or KV cache. Execute it once before exposing this state.
            graph.replay()
            store["buckets"][key] = state
            store["captures"] = int(store.get("captures", 0)) + 1
            store["capture_replays"] = int(store.get("capture_replays", 0)) + 1
            self.block_manager.advance_seq_len(seq_id, int(input_ids.shape[1]))
            self._log_generate_graph(
                f"prefill captured tokens={int(input_ids.shape[1])}"
            )
            return logits
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            failed_keys[key] = failure
            store["failures"] = int(store.get("failures", 0)) + 1
            store["last_failure"] = failure
            self._log_generate_graph("prefill capture failed")
            return eager_prefill()

    def _prepare_generate_graph_capture_stream(
        self,
        state: dict,
        device: torch.device,
    ) -> Optional[torch.cuda.Stream]:
        if not bool(
            getattr(self.model, "_gemma4_flat_parallel_moe_enabled", False)
        ):
            return None
        stream = state.get("capture_stream")
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            state["capture_stream"] = stream
        stream.wait_stream(torch.cuda.current_stream(device))
        return stream

    @staticmethod
    def _replay_generate_graph(state: dict, graph: torch.cuda.CUDAGraph) -> None:
        stream = state.get("capture_stream")
        if stream is None:
            graph.replay()
            return
        device = state["input_ids"].device
        caller = torch.cuda.current_stream(device)
        stream.wait_stream(caller)
        with torch.cuda.stream(stream):
            graph.replay()
        caller.wait_stream(stream)

    def _single_multi_step_graph_key(self, seq_id: int, num_steps: int) -> tuple:
        blocks = self.block_manager.block_tables[int(seq_id)]
        return (
            "single_multi_step",
            int(num_steps),
            int(len(blocks)),
            int(self.block_manager.seq_lens[int(seq_id)]),
            int(getattr(self.block_manager, "block_size", 1) or 1),
            str(getattr(self.model.config, "model_type", "")),
        )

    def _prepare_single_multi_step_graph_state(
        self,
        key: tuple,
        seq_id: int,
        decode_input: torch.Tensor,
        decode_pos: torch.Tensor,
    ) -> dict:
        state = self._generate_multi_step_graph_states.get(key)
        if state is None:
            _, _, table_blocks, _, _, _ = key
            with torch.inference_mode(False):
                state = {
                    "key": key,
                    "input_ids": torch.empty_like(decode_input),
                    "positions": torch.empty_like(decode_pos),
                    "block_table": torch.empty(
                        1,
                        int(table_blocks),
                        dtype=torch.int32,
                        device=decode_input.device,
                    ),
                    "seq_lens": torch.empty(
                        1,
                        dtype=torch.int32,
                        device=decode_input.device,
                    ),
                    "graph": None,
                    "all_tokens": None,
                    "warmed": False,
                    "failed": False,
                    "failure": "",
                }
            self._generate_multi_step_graph_states[key] = state

        state["input_ids"].copy_(decode_input)
        state["positions"].copy_(decode_pos)
        table = state["block_table"]
        table.zero_()
        blocks = self.block_manager.block_tables[int(seq_id)]
        if blocks:
            table[0, : len(blocks)].copy_(
                torch.tensor(blocks, dtype=torch.int32, device=table.device)
            )
        state["seq_lens"].fill_(int(self.block_manager.seq_lens[int(seq_id)]))
        return state

    def _run_single_multi_step_graph_or_eager(
        self,
        seq_id: int,
        decode_input: torch.Tensor,
        decode_pos: torch.Tensor,
        num_steps: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (
            not self._generate_cuda_graphs
            or not self._generate_multi_step_cuda_graphs
            or self.device != "cuda"
            or not torch.cuda.is_available()
            or self.kv_offload
            or num_steps <= 0
            or not hasattr(self.model, "decode_multi_step")
            or type(self.block_manager) is not BlockManager
        ):
            return self.model.decode_multi_step(
                decode_input,
                decode_pos,
                self.block_manager,
                [seq_id],
                num_steps=num_steps,
                return_final_logits=False,
            )

        key = self._single_multi_step_graph_key(seq_id, num_steps)
        state = self._prepare_single_multi_step_graph_state(
            key,
            seq_id,
            decode_input,
            decode_pos,
        )
        if state.get("failed"):
            return self.model.decode_multi_step(
                decode_input,
                decode_pos,
                self.block_manager,
                [seq_id],
                num_steps=num_steps,
                return_final_logits=False,
            )

        if state.get("graph") is not None:
            self._replay_generate_graph(state, state["graph"])
            self.block_manager.seq_lens[int(seq_id)] += int(num_steps)
            self.block_manager._seq_lens_tensor = None
            self.block_manager._seq_lens_seq_key = None
            return state["all_tokens"], None

        if not state.get("warmed", False):
            state["warmed"] = True
            self._log_generate_graph(
                f"warmup multi_step steps={num_steps} table_blocks={state['block_table'].shape[1]}"
            )
            return self.model.decode_multi_step(
                decode_input,
                decode_pos,
                self.block_manager,
                [seq_id],
                num_steps=num_steps,
                return_final_logits=False,
            )

        prepare_flat = getattr(self.model, "_prepare_flat_decode", None)
        if callable(prepare_flat) and not getattr(self.model, "_flat_decode_ready", False):
            try:
                prepare_flat()
            except Exception:
                pass

        setter = getattr(self.block_manager, "set_decode_metadata_override", None)
        clearer = getattr(self.block_manager, "clear_decode_metadata_override", None)
        if setter is None or clearer is None:
            return self.model.decode_multi_step(
                decode_input,
                decode_pos,
                self.block_manager,
                [seq_id],
                num_steps=num_steps,
                return_final_logits=False,
            )

        graph = torch.cuda.CUDAGraph()
        python_seq_len_before_capture = int(self.block_manager.seq_lens[int(seq_id)])
        setter(
            state["block_table"],
            state["seq_lens"],
            int(state["block_table"].shape[1]),
        )
        try:
            torch.cuda.synchronize()
            capture_stream = self._prepare_generate_graph_capture_stream(
                state, state["input_ids"].device
            )
            graph_context = (
                torch.cuda.graph(graph, stream=capture_stream)
                if capture_stream is not None
                else torch.cuda.graph(graph)
            )
            with graph_context:
                all_tokens, final_logits = self.model.decode_multi_step(
                    state["input_ids"],
                    state["positions"],
                    self.block_manager,
                    [seq_id],
                    num_steps=num_steps,
                    return_final_logits=False,
                )
            if capture_stream is not None:
                torch.cuda.current_stream(state["input_ids"].device).wait_stream(
                    capture_stream
                )
        except Exception as exc:
            self.block_manager.seq_lens[int(seq_id)] = python_seq_len_before_capture
            self.block_manager._seq_lens_tensor = None
            self.block_manager._seq_lens_seq_key = None
            state["failed"] = True
            state["failure"] = f"{type(exc).__name__}: {exc}"
            self._log_generate_graph(
                f"multi_step capture failed steps={num_steps}: {state['failure']}; falling back to eager"
            )
            return self.model.decode_multi_step(
                decode_input,
                decode_pos,
                self.block_manager,
                [seq_id],
                num_steps=num_steps,
                return_final_logits=False,
            )
        finally:
            clearer()

        # Stream capture records the operations but does not materialize the
        # captured output buffers. Python sequence lengths were already
        # advanced while decode_multi_step executed under capture, so replay
        # exactly once here without advancing them again.
        self._replay_generate_graph(state, graph)
        self.block_manager._seq_lens_tensor = None
        self.block_manager._seq_lens_seq_key = None
        state["graph"] = graph
        state["all_tokens"] = all_tokens
        self._log_generate_graph(
            f"captured multi_step steps={num_steps} table_blocks={state['block_table'].shape[1]}"
        )
        return all_tokens, final_logits

    def _single_step_graph_key(self, seq_id: int, *, return_next_token: bool = False) -> tuple:
        blocks = self.block_manager.block_tables[int(seq_id)]
        block_size = int(getattr(self.block_manager, "block_size", 1) or 1)
        if self._generate_stable_max_blocks:
            loop_blocks = len(blocks)
        else:
            seq_len = int(self.block_manager.seq_lens[int(seq_id)])
            loop_blocks = max(1, (seq_len + 1 + block_size - 1) // block_size)
        return (
            "single_step",
            int(len(blocks)),
            int(loop_blocks),
            int(block_size),
            str(getattr(self.model.config, "model_type", "")),
            bool(return_next_token),
        )

    def _prepare_single_step_graph_state(
        self,
        key: tuple,
        seq_id: int,
        decode_input: torch.Tensor,
        decode_pos: torch.Tensor,
        *,
        input_token_id: Optional[int] = None,
        input_token_tensor: Optional[torch.Tensor] = None,
        position_id: Optional[int] = None,
        update_step_inputs: bool = True,
        update_seq_len: bool = True,
    ) -> dict:
        state = self._generate_multi_step_graph_states.get(key)
        if state is None:
            _, table_blocks, loop_blocks, _, _, return_next_token = key
            with torch.inference_mode(False):
                state = {
                    "key": key,
                    "return_next_token": bool(return_next_token),
                    "max_decode_blocks": int(loop_blocks),
                    "input_ids": torch.empty_like(decode_input),
                    "positions": torch.empty_like(decode_pos),
                    "block_table": torch.empty(
                        1,
                        int(table_blocks),
                        dtype=torch.int32,
                        device=decode_input.device,
                    ),
                    "seq_lens": torch.empty(
                        1,
                        dtype=torch.int32,
                        device=decode_input.device,
                    ),
                    "block_table_blocks": None,
                    "graph": None,
                    "logits": None,
                    "next_tokens": None,
                    "warmed": False,
                    "failed": False,
                    "failure": "",
                }
            self._generate_multi_step_graph_states[key] = state

        if update_step_inputs:
            if input_token_tensor is not None:
                state["input_ids"].copy_(input_token_tensor.view_as(state["input_ids"]))
            elif input_token_id is not None:
                state["input_ids"].fill_(int(input_token_id))
            else:
                state["input_ids"].copy_(decode_input)

            if position_id is not None:
                state["positions"].fill_(int(position_id))
            else:
                state["positions"].copy_(decode_pos)
        table = state["block_table"]
        blocks = self.block_manager.block_tables[int(seq_id)]
        blocks_tuple = tuple(int(block_id) for block_id in blocks)
        if state.get("block_table_blocks") != blocks_tuple:
            table.zero_()
            if blocks_tuple:
                table[0, : len(blocks_tuple)].copy_(
                    torch.as_tensor(blocks_tuple, dtype=torch.int32, device=table.device)
                )
            state["block_table_blocks"] = blocks_tuple
        if update_seq_len:
            state["seq_lens"].fill_(int(self.block_manager.seq_lens[int(seq_id)]))
        return state

    def _run_single_decode_step_graph_or_eager(
        self,
        seq_id: int,
        decode_input: torch.Tensor,
        decode_pos: torch.Tensor,
        *,
        return_next_token: bool = False,
        input_token_id: Optional[int] = None,
        input_token_tensor: Optional[torch.Tensor] = None,
        position_id: Optional[int] = None,
        chain_graph_inputs: bool = False,
    ) -> torch.Tensor:
        def _materialize_decode_inputs() -> None:
            if input_token_tensor is not None:
                decode_input.copy_(input_token_tensor.view_as(decode_input))
            elif input_token_id is not None:
                decode_input.fill_(int(input_token_id))
            if position_id is not None:
                decode_pos.fill_(int(position_id))

        def _eager_decode_step():
            _materialize_decode_inputs()
            result = self.model.decode_step(
                decode_input,
                decode_pos,
                self.block_manager,
                [seq_id],
                return_next_token=return_next_token,
            )
            return result

        if (
            not self._generate_cuda_graphs
            or not self._generate_step_cuda_graphs
            or self.device != "cuda"
            or not torch.cuda.is_available()
            or self.kv_offload
            or type(self.block_manager) is not BlockManager
        ):
            return _eager_decode_step()

        key = self._single_step_graph_key(
            seq_id,
            return_next_token=return_next_token,
        )
        cached_state = self._generate_multi_step_graph_states.get(key)
        use_persistent_replay_inputs = bool(
            chain_graph_inputs
            and return_next_token
            and cached_state is not None
            and cached_state.get("graph") is not None
            and int(cached_state.get("chain_seq_id", -1)) == int(seq_id)
        )
        state = self._prepare_single_step_graph_state(
            key,
            seq_id,
            decode_input,
            decode_pos,
            input_token_id=input_token_id,
            input_token_tensor=input_token_tensor,
            position_id=position_id,
            update_step_inputs=not use_persistent_replay_inputs,
            update_seq_len=not use_persistent_replay_inputs,
        )
        if state.get("failed"):
            return _eager_decode_step()

        if state.get("graph") is not None:
            self._replay_generate_graph(state, state["graph"])
            self.block_manager.seq_lens[int(seq_id)] += 1
            self.block_manager._seq_lens_tensor = None
            self.block_manager._seq_lens_seq_key = None
            if chain_graph_inputs and return_next_token:
                state["chain_seq_id"] = int(seq_id)
            if return_next_token:
                return state["next_tokens"]
            return state["logits"]

        if not state.get("warmed", False):
            state["warmed"] = True
            self._log_generate_graph(
                "warmup step "
                f"table_blocks={state['block_table'].shape[1]} "
                f"loop_blocks={state.get('max_decode_blocks')}"
            )
            return _eager_decode_step()

        prepare_flat = getattr(self.model, "_prepare_flat_decode", None)
        if callable(prepare_flat) and not getattr(self.model, "_flat_decode_ready", False):
            try:
                prepare_flat()
            except Exception:
                pass

        setter = getattr(self.block_manager, "set_decode_metadata_override", None)
        clearer = getattr(self.block_manager, "clear_decode_metadata_override", None)
        if setter is None or clearer is None:
            return _eager_decode_step()

        graph = torch.cuda.CUDAGraph()
        python_seq_len_before_capture = int(self.block_manager.seq_lens[int(seq_id)])
        setter(
            state["block_table"],
            state["seq_lens"],
            int(state.get("max_decode_blocks") or state["block_table"].shape[1]),
        )
        try:
            torch.cuda.synchronize()
            capture_stream = self._prepare_generate_graph_capture_stream(
                state, state["input_ids"].device
            )
            graph_context = (
                torch.cuda.graph(graph, stream=capture_stream)
                if capture_stream is not None
                else torch.cuda.graph(graph)
            )
            with graph_context:
                if return_next_token:
                    logits = None
                    next_tokens = self.model.decode_step(
                        state["input_ids"],
                        state["positions"],
                        self.block_manager,
                        [seq_id],
                        return_next_token=True,
                    )
                    if chain_graph_inputs:
                        state["input_ids"].copy_(
                            next_tokens.view_as(state["input_ids"])
                        )
                        state["positions"].add_(1)
                else:
                    logits = self.model.decode_step(
                        state["input_ids"],
                        state["positions"],
                        self.block_manager,
                        [seq_id],
                    )
                    next_tokens = None
            if capture_stream is not None:
                torch.cuda.current_stream(state["input_ids"].device).wait_stream(
                    capture_stream
                )
        except Exception as exc:
            self.block_manager.seq_lens[int(seq_id)] = python_seq_len_before_capture
            self.block_manager._seq_lens_tensor = None
            self.block_manager._seq_lens_seq_key = None
            state["failed"] = True
            state["failure"] = f"{type(exc).__name__}: {exc}"
            self._log_generate_graph(
                "step capture failed "
                f"table_blocks={state['block_table'].shape[1]} "
                f"loop_blocks={state.get('max_decode_blocks')}: "
                f"{state['failure']}; falling back to eager"
            )
            return _eager_decode_step()
        finally:
            clearer()

        # The capture call advances Python metadata, while the captured CUDA
        # work still needs one replay to populate logits/tokens and KV state.
        # Do not increment Python sequence lengths here; capture already did.
        self._replay_generate_graph(state, graph)
        self.block_manager._seq_lens_tensor = None
        self.block_manager._seq_lens_seq_key = None
        state["graph"] = graph
        state["logits"] = logits
        state["next_tokens"] = next_tokens
        if chain_graph_inputs and return_next_token:
            state["chain_seq_id"] = int(seq_id)
        else:
            state.pop("chain_seq_id", None)
        self._log_generate_graph(
            "captured step "
            f"table_blocks={state['block_table'].shape[1]} "
            f"loop_blocks={state.get('max_decode_blocks')}"
        )
        if return_next_token:
            return next_tokens
        return logits

    # =========================================================================
    # Single-request generation (simple API)
    # =========================================================================

    @torch.inference_mode()
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_token_ids: Optional[List[int]] = None,
        verbose: bool = False,
        xai: bool = False,
        xai_top_k: int = 10,
        logit_lens: Union[bool, int] = False,
    ):
        """
        Generate text (sequential, one prompt at a time).

        Args:
            xai: Enable XAI interpretability. Returns (text, XAIReport) per prompt.
            xai_top_k: Number of top-K tokens to capture per step (default: 10).
            logit_lens: Enable Logit Lens (per-layer hidden state probing). Requires xai=True.
                - False: disabled (default)
                - True: probe ALL layers
                - int N: probe every Nth layer + first + last (stride mode, smaller output)

        Returns:
            str or List[str] when xai=False.
            (str, XAIReport) or List[(str, XAIReport)] when xai=True.
        """
        single = isinstance(prompts, str)
        if single:
            prompts = [prompts]

        stop_token_ids = list(_normalize_token_id_set(stop_token_ids, self.tokenizer.eos_token_id))

        results = []
        for prompt in prompts:
            result = self._generate_single(
                prompt, max_new_tokens, temperature,
                top_k, top_p, repetition_penalty,
                stop_token_ids, verbose,
                xai=xai, xai_top_k=xai_top_k, logit_lens=logit_lens,
            )
            results.append(result)

        return results[0] if single else results

    def _generate_single(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        stop_token_ids: List[int],
        verbose: bool,
        xai: bool = False,
        xai_top_k: int = 10,
        logit_lens: Union[bool, int] = False,
    ):
        """Generate text for a single prompt."""
        seq_id = self._next_seq_id()
        original_prompt = prompt  # Keep for XAI report

        _, input_ids = self._prepare_prompt_inputs(prompt)
        prompt_len = input_ids.shape[1]

        # Allocate KV cache blocks (extra for generation)
        total_tokens = prompt_len + max_new_tokens
        self.block_manager.allocate_sequence(seq_id, total_tokens)

        # Pre-allocate reusable tensors for decode loop
        decode_input = torch.empty(1, 1, dtype=torch.long, device=self.device)
        decode_pos = torch.empty(1, 1, dtype=torch.long, device=self.device)

        # Convert stop tokens to set for O(1) lookup
        stop_set = set(stop_token_ids)

        # XAI: prepare collection lists
        xai_steps = [] if xai else None
        xai_probs = [] if xai else None  # chosen token probs for confidence
        # use_logit_lens: False when disabled, or the logit_lens value (True/int) when enabled with xai
        use_logit_lens = logit_lens if (xai and logit_lens) else False

        try:
            # === PREFILL ===
            if verbose:
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            positions = torch.arange(
                prompt_len, device=self.device
            ).unsqueeze(0)

            # Ensure all blocks on GPU before prefill
            if self.kv_offload:
                self.block_manager.ensure_blocks_on_gpu([seq_id])

            # Prefill (with optional Logit Lens)
            prefill_result = self._run_single_prefill_graph_or_eager(
                input_ids,
                positions,
                seq_id,
                use_logit_lens,
            )
            if use_logit_lens and isinstance(prefill_result, tuple):
                logits, layer_probes = prefill_result
            else:
                logits = prefill_result
                layer_probes = None

            # Evict cold blocks after prefill to free GPU memory
            if self.kv_offload:
                self.block_manager.evict_cold_blocks([seq_id])

            if verbose:
                torch.cuda.synchronize()
            t_prefill = time.perf_counter()

            # Sample first token
            first_logits = logits[:, -1, :]
            decode_vocab_limit = int(first_logits.shape[-1])
            fallback_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if fallback_token_id is None or not (0 <= int(fallback_token_id) < decode_vocab_limit):
                fallback_token_id = 0
            else:
                fallback_token_id = int(fallback_token_id)
            is_greedy = temperature == 0.0 or temperature is None
            skip_token_materialization = bool(
                self._generate_skip_token_materialization
                and self._generate_cuda_graphs
                and self._generate_step_cuda_graphs
                and self._generate_gpu_token_chain
                and self.device == "cuda"
                and is_greedy
                and repetition_penalty == 1.0
                and not xai
                and not use_logit_lens
                and not stop_set
            )
            next_token_tensor = sample_logits(
                first_logits, temperature, top_k, top_p,
            ).to(device=self.device, dtype=torch.long)
            next_token_id = (
                fallback_token_id
                if skip_token_materialization
                else int(next_token_tensor.item())
            )

            # XAI: capture first token info
            if xai:
                step_info = self._build_xai_step(
                    first_logits[0], next_token_id, 0, xai_top_k,
                    layer_probes=layer_probes,
                )
                xai_steps.append(step_info)
                xai_probs.append(step_info.chosen.probability)

            generated_ids = [next_token_id]

            # Pre-allocate past_tokens buffer with FIXED size (no torch.cat!)
            if repetition_penalty != 1.0:
                _past_buf = torch.empty(1, prompt_len + max_new_tokens, dtype=torch.long, device=self.device)
                _past_buf[0, :prompt_len] = input_ids[0]
                _past_buf[0, prompt_len] = next_token_id
                _past_idx = prompt_len + 1  # next write position
            else:
                _past_buf = None
                _past_idx = 0

            # Hoist kv_offload check out of loop (hot branch elimination)
            _do_offload = self.kv_offload

            # === GREEDY FAST PATH: decode_multi_step (no Python loop) ===
            has_multi = hasattr(self.model, 'decode_multi_step')
            prefer_step_graph = bool(
                self._generate_cuda_graphs
                and self._generate_step_cuda_graphs
                and not xai
                and repetition_penalty == 1.0
            )
            use_gpu_token_chain = bool(
                prefer_step_graph
                and self._generate_gpu_token_chain
                and (
                    str(getattr(self.model.config, "model_type", "")) != "qwen3_moe"
                    or self._generate_gpu_token_chain_allow_qwen3_moe
                    or _env_bool("MEGAGEMM_GENERATE_GPU_TOKEN_CHAIN_UNSAFE", False)
                    or skip_token_materialization
                )
                and is_greedy
                and not stop_set
                and _past_buf is None
                and not use_logit_lens
            )
            if (
                self._generate_gpu_token_chain
                and not use_gpu_token_chain
                and prefer_step_graph
                and str(getattr(self.model.config, "model_type", "")) == "qwen3_moe"
                and not self._generate_gpu_token_chain_allow_qwen3_moe
            ):
                self._log_generate_graph(
                    "gpu token chain disabled for qwen3_moe; use "
                    "MEGAGEMM_GENERATE_GPU_TOKEN_CHAIN_ALLOW_QWEN3_MOE=1 to force it"
                )
            generated_token_buffer = None
            generated_token_count = 0
            if use_gpu_token_chain:
                if not skip_token_materialization:
                    generated_token_buffer = torch.empty(
                        max_new_tokens,
                        dtype=torch.long,
                        device=self.device,
                    )
                    generated_token_buffer[0].copy_(next_token_tensor.reshape(-1)[0])
                generated_token_count = 1
                if skip_token_materialization:
                    self._log_generate_graph(
                        "gpu token chain active (skip token materialization)"
                    )
                else:
                    self._log_generate_graph("gpu token chain active")
            use_fused_argmax_step = bool(
                prefer_step_graph
                and self._generate_fused_argmax_step
                and not use_gpu_token_chain
                and is_greedy
                and _past_buf is None
                and not use_logit_lens
            )
            if use_fused_argmax_step:
                self._log_generate_graph("fused argmax step active")
            can_fast = (
                is_greedy
                and has_multi
                and not xai
                and repetition_penalty == 1.0
                and not prefer_step_graph
            )

            if can_fast:
                remaining = max_new_tokens - 1
                if remaining > 0 and next_token_id not in stop_set:
                    decode_input.fill_(next_token_id)
                    decode_pos.fill_(prompt_len)

                    if _do_offload:
                        self.block_manager.ensure_blocks_on_gpu([seq_id])
                    all_tokens, _ = self._run_single_multi_step_graph_or_eager(
                        seq_id,
                        decode_input,
                        decode_pos,
                        remaining,
                    )

                    if _do_offload:
                        self.block_manager.evict_cold_blocks([seq_id])

                    decoded_tokens = all_tokens[0].tolist()
                    take = len(decoded_tokens)
                    if stop_set:
                        for idx, tok in enumerate(decoded_tokens):
                            if tok in stop_set:
                                take = idx + 1
                                break
                    if take > 0:
                        generated_ids.extend(decoded_tokens[:take])
                        next_token_id = decoded_tokens[take - 1]
            else:
                # === STANDARD PATH: per-token with sampling ===
                for step in range(max_new_tokens - 1):
                    if not use_gpu_token_chain and next_token_id in stop_set:
                        break

                    use_direct_graph_inputs = bool(
                        self._generate_direct_graph_inputs
                        and self._generate_cuda_graphs
                        and self._generate_step_cuda_graphs
                        and not use_logit_lens
                    )
                    graph_input_token_id = None
                    graph_input_token_tensor = None
                    graph_position_id = None
                    if use_direct_graph_inputs:
                        graph_position_id = int(prompt_len + step)
                        if use_gpu_token_chain:
                            graph_input_token_tensor = next_token_tensor
                        else:
                            graph_input_token_id = int(next_token_id)
                    else:
                        # Fill pre-allocated tensors (no allocation!)
                        if use_gpu_token_chain:
                            decode_input.copy_(next_token_tensor.view_as(decode_input))
                        else:
                            decode_input.fill_(next_token_id)
                        decode_pos.fill_(prompt_len + step)

                    if _do_offload:
                        self.block_manager.ensure_blocks_on_gpu([seq_id])

                    # Run decode step (with optional Logit Lens)
                    if (
                        self._generate_cuda_graphs
                        and self._generate_step_cuda_graphs
                        and not use_logit_lens
                    ):
                        decode_result = self._run_single_decode_step_graph_or_eager(
                            seq_id,
                            decode_input,
                            decode_pos,
                            return_next_token=use_gpu_token_chain or use_fused_argmax_step,
                            input_token_id=graph_input_token_id,
                            input_token_tensor=graph_input_token_tensor,
                            position_id=graph_position_id,
                            chain_graph_inputs=(
                                self._generate_persistent_step_graph_inputs
                                and use_gpu_token_chain
                                and skip_token_materialization
                            ),
                        )
                    else:
                        decode_result = self.model.decode_step(
                            decode_input, decode_pos,
                            self.block_manager, [seq_id],
                            logit_lens=use_logit_lens,
                        )
                    if use_gpu_token_chain:
                        logits = None
                        layer_probes = None
                    elif use_fused_argmax_step:
                        logits = None
                        layer_probes = None
                    elif use_logit_lens and isinstance(decode_result, tuple):
                        logits, layer_probes = decode_result
                    else:
                        logits = decode_result
                        layer_probes = None

                    if _do_offload:
                        self.block_manager.evict_cold_blocks([seq_id])

                    # Append token to pre-allocated rep buffer (no torch.cat!)
                    if _past_buf is not None:
                        _past_buf[0, _past_idx] = next_token_id
                        _past_idx += 1

                    # Sample next token with repetition penalty
                    if use_gpu_token_chain:
                        next_token_tensor = decode_result.to(
                            device=self.device,
                            dtype=torch.long,
                        )
                        if generated_token_buffer is not None:
                            generated_token_buffer[generated_token_count].copy_(
                                next_token_tensor.reshape(-1)[0]
                            )
                        generated_token_count += 1
                    elif use_fused_argmax_step:
                        next_token_tensor = decode_result.to(
                            device=self.device,
                            dtype=torch.long,
                        )
                        next_token_id = int(next_token_tensor.reshape(-1)[0].item())
                        if not (0 <= int(next_token_id) < decode_vocab_limit):
                            raise RuntimeError(
                                "fused argmax step produced invalid token id "
                                f"{next_token_id} for vocab={decode_vocab_limit}"
                            )
                    else:
                        # Get logits before sampling modifies them
                        step_logits = logits[:, -1, :]
                        next_token_id = sample_logits(
                            step_logits, temperature, top_k, top_p,
                            repetition_penalty=repetition_penalty,
                            past_tokens=_past_buf[:, :_past_idx] if _past_buf is not None else None,
                        ).item()

                    # XAI: capture step info (using original logits before rep penalty)
                    if xai:
                        step_info = self._build_xai_step(
                            logits[0, -1, :], next_token_id, step + 1, xai_top_k,
                            layer_probes=layer_probes,
                        )
                        xai_steps.append(step_info)
                        xai_probs.append(step_info.chosen.probability)

                    if not use_gpu_token_chain:
                        generated_ids.append(next_token_id)

            if generated_token_buffer is not None:
                generated_ids = (
                    generated_token_buffer[:generated_token_count]
                    .detach()
                    .cpu()
                    .tolist()
                )
                invalid_tokens = sum(
                    1 for tok in generated_ids
                    if int(tok) < 0 or int(tok) >= decode_vocab_limit
                )
                if invalid_tokens:
                    self._log_generate_graph(
                        f"gpu token chain sanitized {invalid_tokens} invalid token ids"
                    )
                    generated_ids = [
                        int(tok) if 0 <= int(tok) < decode_vocab_limit else fallback_token_id
                        for tok in generated_ids
                    ]
                if generated_ids:
                    next_token_id = int(generated_ids[-1])
            elif skip_token_materialization and use_gpu_token_chain:
                generated_ids = [fallback_token_id] * int(generated_token_count)
                next_token_id = fallback_token_id

            self._last_generated_ids = [int(token_id) for token_id in generated_ids]

            if verbose:
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            num_generated = len(generated_ids)
            prefill_ms = (t_prefill - t_start) * 1000.0
            decode_ms = (t_end - t_prefill) * 1000.0
            self._last_generation_metrics = {
                "prefill_ms": float(prefill_ms),
                "decode_ms": float(decode_ms),
                "output_tokens": float(num_generated),
                "decode_tok_s": float(
                    num_generated / (t_end - t_prefill) if t_end > t_prefill else 0.0
                ),
                "synchronized": float(bool(verbose)),
            }

            # Print offload stats
            if self.kv_offload and verbose:
                self.block_manager.print_stats()

            if verbose:
                num_gen = num_generated
                tps = self._last_generation_metrics["decode_tok_s"]
                print(f"[MegaGemm] Prefill: {prefill_ms:.1f}ms ({prompt_len} tokens) | "
                      f"Decode: {decode_ms:.1f}ms ({num_gen} tokens) | "
                      f"Speed: {tps:.1f} tok/s")

            # Decode output. Benchmarks care about timing; do not let a malformed
            # experimental graph token id waste a full GPU run at text decode time.
            if skip_token_materialization and use_gpu_token_chain:
                output_text = ""
            else:
                try:
                    output_text = self.tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )
                except (OverflowError, ValueError):
                    safe_ids = [
                        int(tok) if 0 <= int(tok) < decode_vocab_limit else fallback_token_id
                        for tok in generated_ids
                    ]
                    output_text = self.tokenizer.decode(
                        safe_ids, skip_special_tokens=True
                    )

            # XAI: build report
            if xai:
                model_name = getattr(self, '_model_name', '')
                report = XAIReport(
                    prompt=original_prompt,
                    generated_text=output_text,
                    steps=xai_steps,
                    confidence_score=compute_confidence(xai_probs),
                    model_name=model_name,
                    num_layers=self.config.num_hidden_layers,
                )

                # Monitor: record request with XAI metrics
                if self._monitor is not None:
                    num_gen = len(generated_ids)
                    prefill_ms = (t_prefill - t_start) * 1000
                    decode_ms = (t_end - t_prefill) * 1000
                    total_ms = (t_end - t_start) * 1000
                    tps = num_gen / (t_end - t_prefill) if t_end > t_prefill else 0
                    self._monitor.record(RequestRecord(
                        timestamp=report.timestamp,
                        prompt=original_prompt,
                        generated_text=output_text,
                        num_tokens_input=prompt_len,
                        num_tokens_output=num_gen,
                        prefill_ms=prefill_ms,
                        decode_ms=decode_ms,
                        total_ms=total_ms,
                        ttft_ms=prefill_ms,
                        tokens_per_second=tps,
                        confidence_score=report.confidence_score,
                        mean_entropy=report.mean_entropy,
                        hallucination_risk=report.hallucination_risk,
                        high_entropy_steps=report.high_entropy_steps,
                        model_name=self._model_name,
                        quantization=self._quantize_mode,
                    ))

                return (output_text, report)

            # Monitor: record request without XAI
            if self._monitor is not None:
                import time as _time
                num_gen = len(generated_ids)
                prefill_ms = (t_prefill - t_start) * 1000
                decode_ms = (t_end - t_prefill) * 1000
                total_ms = (t_end - t_start) * 1000
                tps = num_gen / (t_end - t_prefill) if t_end > t_prefill else 0
                self._monitor.record(RequestRecord(
                    timestamp=_time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    prompt=original_prompt,
                    generated_text=output_text,
                    num_tokens_input=prompt_len,
                    num_tokens_output=num_gen,
                    prefill_ms=prefill_ms,
                    decode_ms=decode_ms,
                    total_ms=total_ms,
                    ttft_ms=prefill_ms,
                    tokens_per_second=tps,
                    model_name=self._model_name,
                    quantization=self._quantize_mode,
                ))

            return output_text

        finally:
            # Always free KV cache blocks
            self.free_sequence(seq_id)

    def _build_xai_step(
        self,
        logits_1d: 'torch.Tensor',  # [vocab_size]
        chosen_id: int,
        position: int,
        k: int,
        layer_probes: Optional[Dict[int, 'torch.Tensor']] = None,
    ) -> GenerationStep:
        """
        Build one XAI GenerationStep from logits.

        This is called per decode step only when xai=True.
        """
        # Top-K predictions from raw logits
        top_k_preds = extract_top_k_predictions(
            logits_1d, self.tokenizer, k=k
        )

        # Find chosen token's probability
        probs = torch.softmax(logits_1d.float(), dim=-1)
        chosen_prob = probs[chosen_id].item()
        chosen_str = self.tokenizer.decode([chosen_id])
        chosen = TokenPrediction(
            token_id=chosen_id,
            token_str=chosen_str,
            probability=chosen_prob,
        )

        # Compute Shannon entropy for hallucination detection
        step_entropy = compute_entropy(logits_1d)

        # Logit Lens: per-layer predictions
        lens_data = None
        if layer_probes is not None:
            lens_data = {}
            for layer_idx, probe_logits in layer_probes.items():
                lens_data[layer_idx] = extract_top_k_predictions(
                    probe_logits, self.tokenizer, k=min(k, 5)
                )

        return GenerationStep(
            position=position,
            chosen=chosen,
            top_k=top_k_preds,
            entropy=step_entropy,
            logit_lens=lens_data,
        )

    @torch.inference_mode()
    def profile_decode_breakdown(
        self,
        prompt: Union[str, List[str], List[List[int]]] = "Explain relativity",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        ignore_eos: bool = False,
        record_shapes: bool = False,
    ) -> Dict[str, Any]:
        """
        Profile one generation pass and summarize decode bottlenecks.
        Returns CPU/CUDA ms buckets and total kernel launch calls.
        """
        from torch.profiler import profile, ProfilerActivity
        prompts = list(prompt) if isinstance(prompt, (list, tuple)) else None

        profiler_kwargs = {
            "activities": [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            "record_shapes": bool(record_shapes),
            "with_stack": False,
            "profile_memory": False,
        }
        profiler_scope = "full_generation"
        if prompts is None:
            with profile(**profiler_kwargs) as prof:
                if ignore_eos:
                    self._generate_single(
                        str(prompt),
                        max_new_tokens,
                        temperature,
                        top_k,
                        top_p,
                        repetition_penalty,
                        [],
                        False,
                        xai=False,
                    )
                else:
                    self.generate(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        xai=False,
                    )
        else:
            # Start only after the final request's prefill callback.  This
            # prevents long-context prefill GEMMs from contaminating a decode
            # profile while preserving the normal Scheduler decode route.
            prof = profile(**profiler_kwargs)
            prefill_callbacks = 0
            profiler_started = False

            def _start_after_batched_prefill(_request, _pending_logits):
                nonlocal prefill_callbacks, profiler_started
                prefill_callbacks += 1
                if not profiler_started and prefill_callbacks >= len(prompts):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    prof.start()
                    profiler_started = True

            try:
                self.generate_batch(
                    prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    ignore_eos=ignore_eos,
                    prefill_capture_hook=_start_after_batched_prefill,
                    decode_outputs=False,
                )
            finally:
                if not profiler_started:
                    prof.start()
                prof.stop()
            profiler_scope = "decode_after_batched_prefill"

        rows = prof.key_averages()

        def _sum_cpu_ms(patterns):
            return sum(
                r.self_cpu_time_total for r in rows
                if any(p in r.key for p in patterns)
            ) / 1000.0

        def _sum_cuda_ms(patterns):
            total_us = 0.0
            for r in rows:
                if any(p in r.key for p in patterns):
                    total_us += float(
                        getattr(r, "self_cuda_time_total", getattr(r, "self_device_time_total", 0.0))
                    )
            return total_us / 1000.0

        launch_calls = sum(
            int(r.count) for r in rows
            if ("cudaLaunchKernel" in r.key or "cuLaunchKernelEx" in r.key)
        )
        cuda_rows = []
        for row in rows:
            cuda_us = float(
                getattr(
                    row,
                    "self_cuda_time_total",
                    getattr(row, "self_device_time_total", 0.0),
                )
            )
            if cuda_us > 0:
                cuda_rows.append(
                    {
                        "name": str(row.key),
                        "cuda_ms": cuda_us / 1000.0,
                        "calls": int(row.count),
                    }
                )
        cuda_rows.sort(key=lambda item: item["cuda_ms"], reverse=True)

        def _jsonable_shapes(value):
            if isinstance(value, (list, tuple)):
                return [_jsonable_shapes(item) for item in value]
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(value)

        mm_shape_rows = []
        if record_shapes:
            for row in prof.key_averages(group_by_input_shape=True):
                if str(row.key) not in {"aten::mm", "aten::addmm"}:
                    continue
                cuda_us = float(
                    getattr(
                        row,
                        "self_cuda_time_total",
                        getattr(row, "self_device_time_total", 0.0),
                    )
                )
                if cuda_us <= 0.0:
                    continue
                mm_shape_rows.append(
                    {
                        "name": str(row.key),
                        "input_shapes": _jsonable_shapes(
                            getattr(row, "input_shapes", [])
                        ),
                        "cuda_ms": cuda_us / 1000.0,
                        "calls": int(row.count),
                    }
                )
            mm_shape_rows.sort(
                key=lambda item: item["cuda_ms"],
                reverse=True,
            )

        summary = {
            "cpu_launch_ms": _sum_cpu_ms(["cudaLaunchKernel", "cuLaunchKernelEx"]),
            "cpu_alloc_ms": _sum_cpu_ms(["aten::empty", "aten::empty_strided", "aten::empty_like"]),
            "cpu_view_ms": _sum_cpu_ms(["aten::view", "aten::reshape", "aten::as_strided", "aten::transpose"]),
            "cuda_gemv_ms": _sum_cuda_ms(
                ["_fast_gemv_kernel", "_fast_gemv_row_kernel", "internal::gemvx", "aten::mm", "aten::addmm"]
            ),
            "cuda_fused_norm_qkv_ms": _sum_cuda_ms(["_fused_rmsnorm_linear_kernel"]),
            "cuda_attn_ms": _sum_cuda_ms(["_paged_attn_decode", "fused_rope_kv_write"]),
            "cuda_paged_attn_ms": _sum_cuda_ms(["_paged_attn_decode"]),
            "cuda_rope_kv_ms": _sum_cuda_ms(["_fused_rope_kv_write_kernel"]),
            "cuda_norm_ms": _sum_cuda_ms(["rmsnorm_kernel", "_fused_add_rmsnorm"]),
            "cuda_deepfusion_ms": _sum_cuda_ms(["_deepfusion_swiglu_down_kernel"]),
            "cuda_qwen3_moe_router_ms": _sum_cuda_ms(
                [
                    "_qwen3_moe_router_topk_softmax_kernel",
                    "_qwen3_moe_topk_softmax_kernel",
                ]
            ),
            "cuda_qwen3_moe_gate_ms": _sum_cuda_ms(
                [
                    "_qwen3_moe_shared_route_gate_swiglu_kernel",
                    "_qwen3_moe_shared_route_gate_up_kernel",
                    "_qwen3_moe_shared_route_gate_k_split_kernel",
                    "_qwen3_moe_shared_route_gate_k_reduce_swiglu_kernel",
                    "_qwen3_moe_gate_swiglu_kernel",
                    "_qwen3_moe_gate_up_kernel",
                    "_qwen3_moe_expert_grouped_compact_gate_swiglu_kernel",
                ]
            ),
            "cuda_qwen3_moe_down_ms": _sum_cuda_ms(
                [
                    "_qwen3_moe_shared_route_down_accum_kernel",
                    "_qwen3_moe_shared_route_down_partial_kernel",
                    "_qwen3_moe_shared_route_swiglu_down_accum_kernel",
                    "_qwen3_moe_down_from_act_token_accum_kernel",
                    "_qwen3_moe_swiglu_down",
                    "_qwen3_moe_expert_grouped_compact_down_partial_kernel",
                ]
            ),
            "cuda_qwen3_moe_reduce_ms": _sum_cuda_ms(
                [
                    "_qwen3_moe_assignment_reduce",
                    "_qwen3_moe_expert_grouped_compact_partial_reduce",
                ]
            ),
            "cuda_copy_kernel_ms": _sum_cuda_ms(["direct_copy_kernel_cuda"]),
            "cuda_graph_replay_ms": _sum_cuda_ms(["cudaGraphLaunch", "cudaGraphExec"]),
            "cuda_fused_lm_head_argmax_ms": _sum_cuda_ms(
                [
                    "_lm_head_block_max_kernel",
                    "_lm_head_rmsnorm_block_max_kernel",
                    "_logits_softcap_block_max_kernel",
                    "_lm_head_reduce_kernel",
                ]
            ),
            "cuda_swiglu_ms": _sum_cuda_ms(["_mg_swiglu_fwd_kernel", "MegaGemmFunction"]),
            "launch_calls": float(launch_calls),
            "cuda_total_self_ms": sum(item["cuda_ms"] for item in cuda_rows),
            "profile_scope": profiler_scope,
            "record_shapes": bool(record_shapes),
            "cuda_mm_shapes": mm_shape_rows,
        }
        summary["cuda_top_ops"] = cuda_rows[:20]
        summary["batch_size"] = float(len(prompts) if prompts is not None else 1)
        summary["cuda_qwen3_moe_ms"] = (
            summary["cuda_qwen3_moe_router_ms"]
            + summary["cuda_qwen3_moe_gate_ms"]
            + summary["cuda_qwen3_moe_down_ms"]
            + summary["cuda_qwen3_moe_reduce_ms"]
        )
        # Alias: same kernel is now used for both norm+qkv and norm+gate_up paths.
        summary["cuda_fused_norm_linear_ms"] = summary["cuda_fused_norm_qkv_ms"]

        decode_timing = getattr(self.model, "get_last_decode_timing", None)
        if callable(decode_timing):
            last = decode_timing()
            if isinstance(last, dict):
                for key, value in last.items():
                    if isinstance(value, (int, float)):
                        summary[f"decode_{key}"] = float(value)

        print(
            "profile_decode_breakdown "
            + " | ".join(
                [
                    f"batch={int(summary['batch_size'])}",
                    f"scope={summary['profile_scope']}",
                    f"cpu_launch={summary['cpu_launch_ms']:.1f}ms",
                    f"cpu_alloc={summary['cpu_alloc_ms']:.1f}ms",
                    f"cpu_view={summary['cpu_view_ms']:.1f}ms",
                    f"cuda_gemv={summary['cuda_gemv_ms']:.1f}ms",
                    f"cuda_fused_norm_qkv={summary['cuda_fused_norm_qkv_ms']:.1f}ms",
                    f"cuda_attn={summary['cuda_attn_ms']:.1f}ms",
                    f"cuda_paged_attn={summary['cuda_paged_attn_ms']:.1f}ms",
                    f"cuda_rope_kv={summary['cuda_rope_kv_ms']:.1f}ms",
                    f"cuda_norm={summary['cuda_norm_ms']:.1f}ms",
                    f"cuda_moe={summary['cuda_qwen3_moe_ms']:.1f}ms",
                    f"cuda_moe_router={summary['cuda_qwen3_moe_router_ms']:.1f}ms",
                    f"cuda_moe_gate={summary['cuda_qwen3_moe_gate_ms']:.1f}ms",
                    f"cuda_moe_down={summary['cuda_qwen3_moe_down_ms']:.1f}ms",
                    f"cuda_moe_reduce={summary['cuda_qwen3_moe_reduce_ms']:.1f}ms",
                    f"cuda_copy_kernel={summary['cuda_copy_kernel_ms']:.1f}ms",
                    f"cuda_deepfusion={summary['cuda_deepfusion_ms']:.1f}ms",
                    f"cuda_fused_lm_head_argmax={summary['cuda_fused_lm_head_argmax_ms']:.1f}ms",
                    f"cuda_swiglu={summary['cuda_swiglu_ms']:.1f}ms",
                    f"cuda_graph_replay={summary['cuda_graph_replay_ms']:.1f}ms",
                    f"launch_calls={int(summary['launch_calls'])}",
                ]
            )
        )
        for idx, item in enumerate(summary["cuda_top_ops"][:12], start=1):
            print(
                f"profile_cuda_top {idx:02d} "
                f"{item['cuda_ms']:.3f}ms calls={item['calls']} name={item['name']}"
            )
        for idx, item in enumerate(summary["cuda_mm_shapes"][:20], start=1):
            print(
                f"profile_mm_shape {idx:02d} "
                f"{item['cuda_ms']:.3f}ms calls={item['calls']} "
                f"op={item['name']} inputs={item['input_shapes']}"
            )
        return summary

    # =========================================================================
    # Monitoring API
    # =========================================================================

    def get_monitor_stats(self) -> Dict:
        """Get aggregated monitoring statistics. Requires monitor=True."""
        if self._monitor is None:
            return {"error": "Monitoring not enabled. Use InferenceEngine(..., monitor=True)"}
        return self._monitor.get_stats()

    def export_monitor_log(self, path: str) -> int:
        """Export monitoring log as JSONL. Returns number of records exported."""
        if self._monitor is None:
            return 0
        return self._monitor.export_log(path)

    def monitor_summary(self) -> str:
        """Get human-readable monitoring dashboard string."""
        if self._monitor is None:
            return "[MegaGemm] Monitoring not enabled. Use InferenceEngine(..., monitor=True)"
        return self._monitor.summary()

    def reset_monitor(self) -> None:
        """Reset all monitoring statistics."""
        if self._monitor is not None:
            self._monitor.reset()

    def start_dashboard(self, port: int = 8080) -> None:
        """Start live monitoring dashboard. Auto-enables monitoring if needed."""
        if self._monitor is None:
            self._monitor = InferenceMonitor()
            self._monitor_enabled = True
        if self._dashboard is None:
            self._dashboard = DashboardServer(self._monitor, port=port)
        if not self._dashboard.is_running:
            self._dashboard.start()

    def stop_dashboard(self) -> None:
        """Stop the live monitoring dashboard."""
        if self._dashboard is not None:
            self._dashboard.stop()

    # =========================================================================
    # Batch generation (continuous batching!)
    # =========================================================================

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[Union[str, List[int]]],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        stop_token_ids: Optional[List[int]] = None,
        ignore_eos: bool = False,
        verbose: bool = False,
        prefill_capture_hook=None,
        decode_outputs: bool = True,
        materialize_generated_tokens: Optional[bool] = None,
    ) -> List[str]:
        """
        Generate text for multiple prompts using continuous batching.

        All prompts are processed concurrently: decode steps are batched
        across all active sequences, giving ~linear throughput scaling.

        Args:
            prompts: Input texts or pretokenized prompt ID rows. Pretokenized
                rows bypass chat-template application and tokenization.
            max_new_tokens: Max tokens per prompt
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            stop_token_ids: Additional stop tokens
            verbose: Print per-step stats
            decode_outputs: Decode generated token IDs into returned text.
            materialize_generated_tokens: Copy generated IDs to host request
                state. Defaults to ``decode_outputs`` for backward compatibility.

        Returns:
            List of generated texts (same order as input)
        """
        if not prompts:
            return []

        eos_ids = _normalize_token_id_set(stop_token_ids)
        if not ignore_eos:
            eos_ids.update(_normalize_token_id_set(self.tokenizer.eos_token_id))

        # Create scheduler
        if materialize_generated_tokens is None:
            materialize_generated_tokens = decode_outputs

        scheduler = None
        previous_scheduler = getattr(self, "_last_scheduler", None)
        if _request_scheduler_reuse_for_batch(self.model, len(prompts)):
            can_reuse = getattr(previous_scheduler, "can_reuse_for_request", None)
            reset_for_request = getattr(previous_scheduler, "reset_for_request", None)
            if callable(can_reuse) and callable(reset_for_request):
                if can_reuse(
                    model=self.model,
                    block_manager=self.block_manager,
                    max_batch_size=self.max_batch_size,
                    device=self.device,
                ):
                    if self.device == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    reset_for_request(
                        prefill_capture_hook=prefill_capture_hook,
                        materialize_generated_tokens=materialize_generated_tokens,
                    )
                    scheduler = previous_scheduler
        if scheduler is None:
            # A non-reusable graph belongs to its old Scheduler. Release that
            # owner before prefill allocates request-local temporary buffers.
            self._last_scheduler = None
            scheduler = Scheduler(
                model=self.model,
                block_manager=self.block_manager,
                max_batch_size=self.max_batch_size,
                device=self.device,
                prefill_capture_hook=prefill_capture_hook,
                materialize_generated_tokens=materialize_generated_tokens,
            )

        # Submit all requests (order preserved via request_id)
        req_id_to_idx = {}
        for i, prompt in enumerate(prompts):
            if isinstance(prompt, str):
                raw_prompt = prompt
                formatted_prompt = prompt
                # Auto-apply chat template (same as _generate_single)
                bos = self.tokenizer.bos_token
                already_formatted = bos and formatted_prompt.startswith(bos)

                if (
                    not already_formatted
                    and hasattr(self.tokenizer, "chat_template")
                    and self.tokenizer.chat_template
                ):
                    try:
                        messages = [{"role": "user", "content": formatted_prompt}]
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    except Exception:
                        pass

                # Avoid double-BOS
                add_special = not (
                    bos and formatted_prompt.startswith(bos)
                )
                prompt_ids = self.tokenizer.encode(
                    formatted_prompt,
                    add_special_tokens=add_special,
                )
            else:
                try:
                    prompt_ids = [int(token) for token in prompt]
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        "batch prompts must be strings or integer token-ID rows"
                    ) from exc
                if not prompt_ids:
                    raise ValueError("pretokenized prompt rows cannot be empty")
                raw_prompt = None
                formatted_prompt = None
            req_id = scheduler.add_request(
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_token_ids=eos_ids,
                metadata={
                    "prompt": raw_prompt,
                    "formatted_prompt": formatted_prompt,
                    "pretokenized": not isinstance(prompt, str),
                },
            )
            req_id_to_idx[req_id] = i

        if verbose:
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        # Run scheduler loop
        results: Dict[int, str] = {}
        iteration = 0

        while scheduler.has_pending():
            completed = scheduler.step()
            iteration += 1

            for req in completed:
                idx = req_id_to_idx[req.request_id]
                if decode_outputs:
                    results[idx] = self.tokenizer.decode(
                        req.generated_ids, skip_special_tokens=True
                    )
                else:
                    results[idx] = ""

            if verbose and iteration % 10 == 0:
                stats = scheduler.get_stats()
                print(f"  [iter {iteration}] "
                      f"running={stats['running']} "
                      f"waiting={stats['waiting']} "
                      f"completed={stats['completed']}/{len(prompts)}")

        if verbose:
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        if verbose:
            stats = scheduler.get_stats()
            total_ms = (t_end - t_start) * 1000
            total_tok = stats.get('total_tokens', 0)
            tps = total_tok / (t_end - t_start) if t_end > t_start else 0
            print(f"\n[MegaGemm] Batch complete: {len(prompts)} prompts, "
                  f"{total_tok} tokens in {total_ms:.0f}ms | "
                  f"Throughput: {tps:.1f} tok/s")

            # Print KV cache offload stats
            if self.kv_offload:
                self.block_manager.print_stats()

        # Save scheduler for profiling
        self._last_scheduler = scheduler

        # Return in original order
        return [results[i] for i in range(len(prompts))]

    @torch.inference_mode()
    def generate_batch_stream(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        stop_token_ids: Optional[List[int]] = None,
    ):
        """
        Streaming batch generation — yields (index, text) as each sequence completes.

        Same as generate_batch() but yields results incrementally instead of
        blocking until all prompts finish. Enables real-time progress bars.

        Args:
            prompts: List of input texts
            max_new_tokens: Max tokens per prompt
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            stop_token_ids: Additional stop tokens

        Yields:
            (index, text) tuples where index is the original prompt position
        """
        if not prompts:
            return

        eos_ids = _normalize_token_id_set(self.tokenizer.eos_token_id, stop_token_ids)

        scheduler = None
        previous_scheduler = getattr(self, "_last_scheduler", None)
        if _request_scheduler_reuse_for_batch(self.model, len(prompts)):
            can_reuse = getattr(previous_scheduler, "can_reuse_for_request", None)
            reset_for_request = getattr(previous_scheduler, "reset_for_request", None)
            if callable(can_reuse) and callable(reset_for_request):
                if can_reuse(
                    model=self.model,
                    block_manager=self.block_manager,
                    max_batch_size=self.max_batch_size,
                    device=self.device,
                ):
                    if self.device == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    reset_for_request(
                        prefill_capture_hook=None,
                        materialize_generated_tokens=True,
                    )
                    scheduler = previous_scheduler
        if scheduler is None:
            self._last_scheduler = None
            scheduler = Scheduler(
                model=self.model,
                block_manager=self.block_manager,
                max_batch_size=self.max_batch_size,
                device=self.device,
            )

        req_id_to_idx = {}
        for i, prompt in enumerate(prompts):
            bos = self.tokenizer.bos_token
            already_formatted = bos and prompt.startswith(bos)

            if not already_formatted and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                try:
                    messages = [{"role": "user", "content": prompt}]
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    pass

            add_special = not (bos and prompt.startswith(bos))
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=add_special)
            req_id = scheduler.add_request(
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_token_ids=eos_ids,
            )
            req_id_to_idx[req_id] = i

        while scheduler.has_pending():
            completed = scheduler.step()

            for req in completed:
                text = self.tokenizer.decode(
                    req.generated_ids, skip_special_tokens=True
                )
                idx = req_id_to_idx[req.request_id]
                yield idx, text

        self._last_scheduler = scheduler

    # =========================================================================
    # KV Cache Persistence (save/restore context for agents)
    # =========================================================================

    @torch.inference_mode()
    def extract_embedding(self, text: str) -> torch.Tensor:
        """
        Extract a semantic embedding vector from text.

        Uses the model's token embedding table (embed_tokens) + mean pooling.
        Nearly free: just a table lookup, no inference needed.

        Args:
            text: Text to embed

        Returns:
            Normalized embedding tensor [hidden_size] on CPU
        """
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        token_embeds = self.model.embed_tokens(input_ids)  # [1, seq_len, hidden_size]
        # Mean pool over token dimension → [hidden_size]
        embedding = token_embeds.squeeze(0).mean(dim=0)
        # L2 normalize for cosine similarity
        embedding = embedding / embedding.norm()
        return embedding.cpu().float()

    def save_context(self, seq_id: int, text: str = None) -> dict:
        """
        Save a snapshot of a sequence's KV cache for later restoration.

        The snapshot includes all per-layer KV data, sequence length,
        and model config for validation. If text is provided, also computes
        and stores a semantic embedding for similarity search.

        Args:
            seq_id: Active sequence to snapshot
            text: Optional conversation text (stored as metadata + used for embedding)

        Returns:
            Dict snapshot that can be saved to disk or stored in DB.
            Contains 'embedding' key if text was provided.
        """
        snapshot = self.block_manager.serialize_sequence(seq_id)
        snapshot['model_name'] = self._model_name
        mgx_manifest = getattr(self.model, "_mgx_manifest", None) or {}
        for key in (
            "source_model_hash",
            "tokenizer_hash",
            "chat_template_hash",
            "source_model_id",
            "quantization",
            "dtype",
            "target_backend",
        ):
            value = mgx_manifest.get(key)
            if value is not None:
                snapshot[key] = value
        token_ids = self._seq_token_ids.get(seq_id)
        if token_ids is None and text is not None:
            try:
                _, encoded = self._prepare_prompt_inputs(text)
                token_ids = encoded[0].tolist()
            except Exception:
                token_ids = None
        if token_ids is not None:
            snapshot['token_ids'] = [int(token_id) for token_id in token_ids]
        pending_next_logits = self._seq_pending_logits.get(seq_id)
        if pending_next_logits is not None:
            snapshot['pending_next_logits'] = pending_next_logits.clone()
        if text is not None:
            snapshot['text'] = text
            snapshot['embedding'] = self.extract_embedding(text)
        return snapshot

    def _validate_context_snapshot(self, snapshot: dict) -> None:
        saved_model = snapshot.get('model_name', '')
        saved_source_hash = snapshot.get('source_model_hash')
        saved_tokenizer_hash = snapshot.get('tokenizer_hash')
        saved_chat_hash = snapshot.get('chat_template_hash')
        current_manifest = getattr(self.model, "_mgx_manifest", None) or {}
        current_source_hash = current_manifest.get("source_model_hash")
        current_tokenizer_hash = current_manifest.get("tokenizer_hash")
        current_chat_hash = current_manifest.get("chat_template_hash")

        same_compiled_family = (
            saved_source_hash is not None
            and current_source_hash is not None
            and saved_source_hash == current_source_hash
        )

        if saved_model and saved_model != self._model_name and not same_compiled_family:
            raise ValueError(
                f"Model mismatch: snapshot from '{saved_model}', "
                f"engine has '{self._model_name}'"
            )
        if saved_source_hash and current_source_hash and saved_source_hash != current_source_hash:
            raise ValueError(
                "Model hash mismatch: snapshot source_model_hash does not match the current MGX artifact."
            )
        if saved_tokenizer_hash and current_tokenizer_hash and saved_tokenizer_hash != current_tokenizer_hash:
            raise ValueError(
                "Tokenizer hash mismatch: snapshot tokenizer_hash does not match the current MGX artifact."
            )
        if saved_chat_hash and current_chat_hash and saved_chat_hash != current_chat_hash:
            raise ValueError(
                "Chat template hash mismatch: snapshot chat_template_hash does not match the current MGX artifact."
            )

    def _context_restore_extra_tokens(self, snapshot: dict, max_new_tokens: int = 0) -> int:
        if max_new_tokens == 0:
            extra = getattr(self, '_max_seq_len', 512) - int(snapshot['seq_len'])
            return max(extra, 64)
        return max(0, int(max_new_tokens))

    def restore_context(self, snapshot: dict, seq_id: int = None,
                        max_new_tokens: int = 0) -> int:
        """
        Restore a sequence from a KV cache snapshot (skip prefill!).

        The sequence becomes immediately ready for decode — no need to
        re-process the original conversation tokens.

        Args:
            snapshot: Dict from save_context()
            seq_id: Sequence ID to use (auto-assigned if None)
            max_new_tokens: Extra tokens to pre-allocate for future decode.
                If 0, uses engine's max_seq_len - seq_len as headroom.

        Returns:
            Assigned sequence ID
        """
        self._validate_context_snapshot(snapshot)

        if seq_id is None:
            # Auto-assign: use max existing + 1
            existing = set(self.block_manager.block_tables.keys())
            seq_id = max(existing, default=-1) + 1

        extra = self._context_restore_extra_tokens(snapshot, max_new_tokens)

        self.block_manager.deserialize_sequence(seq_id, snapshot, extra_tokens=extra)
        token_ids = snapshot.get('token_ids')
        pending_next_logits = snapshot.get('pending_next_logits')
        if token_ids is not None or pending_next_logits is not None:
            self._set_sequence_runtime_state(
                seq_id,
                token_ids=token_ids if token_ids is not None else None,
                pending_next_logits=pending_next_logits if torch.is_tensor(pending_next_logits) else None,
            )
        else:
            self._clear_sequence_runtime_state(seq_id)
        return seq_id

    def restore_contexts(
        self,
        snapshots: List[dict],
        *,
        seq_ids: Optional[List[int]] = None,
        max_new_tokens: int = 0,
    ) -> List[int]:
        """Restore multiple context snapshots with one batched KV write path."""
        snapshots = list(snapshots)
        if not snapshots:
            return []
        if seq_ids is None:
            seq_ids = [self._next_seq_id() for _ in snapshots]
        else:
            seq_ids = [int(seq_id) for seq_id in seq_ids]
        if len(seq_ids) != len(snapshots):
            raise ValueError("seq_ids length must match snapshots length")

        restore_items = []
        for seq_id, snapshot in zip(seq_ids, snapshots):
            self._validate_context_snapshot(snapshot)
            restore_items.append(
                {
                    "seq_id": int(seq_id),
                    "snapshot": snapshot,
                    "extra_tokens": self._context_restore_extra_tokens(snapshot, max_new_tokens),
                }
            )

        self.block_manager.deserialize_sequences(restore_items)
        for seq_id, snapshot in zip(seq_ids, snapshots):
            token_ids = snapshot.get('token_ids')
            pending_next_logits = snapshot.get('pending_next_logits')
            if token_ids is not None or pending_next_logits is not None:
                self._set_sequence_runtime_state(
                    int(seq_id),
                    token_ids=token_ids if token_ids is not None else None,
                    pending_next_logits=(
                        pending_next_logits if torch.is_tensor(pending_next_logits) else None
                    ),
                )
            else:
                self._clear_sequence_runtime_state(int(seq_id))
        return seq_ids

    def fork_context_prefix(
        self,
        source_seq_id: int,
        *,
        seq_id: Optional[int] = None,
        max_new_tokens: int = 0,
    ) -> int:
        """
        Fork a live prefilled context for prefix-cache-style benchmark hits.

        The child shares immutable prefix KV blocks with the source and receives
        private tail capacity for decode. Runtime token history and pending logits
        are copied so the scheduler can continue generation without prefill.
        """
        source_seq_id = int(source_seq_id)
        if source_seq_id not in self.block_manager.block_tables:
            raise ValueError(f"Source sequence {source_seq_id} not found")
        if seq_id is None:
            seq_id = self._next_seq_id()
        seq_id = int(seq_id)

        source_len = int(self.block_manager.seq_lens[source_seq_id])
        extra = int(max_new_tokens or 0)
        if extra <= 0:
            extra = max(0, int(self.max_seq_len) - source_len)
        self.block_manager.fork_sequence_prefix(
            source_seq_id,
            seq_id,
            extra_tokens=extra,
        )

        token_ids = self._seq_token_ids.get(source_seq_id)
        pending_next_logits = self._seq_pending_logits.get(source_seq_id)
        self._set_sequence_runtime_state(
            seq_id,
            token_ids=list(token_ids) if token_ids is not None else None,
            pending_next_logits=(
                pending_next_logits.clone()
                if torch.is_tensor(pending_next_logits)
                else None
            ),
        )
        return seq_id

    def save_context_to_file(self, seq_id: int, path: str, text: str = None):
        """
        Save KV cache snapshot to disk (compressed).

        Args:
            seq_id: Active sequence to save
            path: File path (.pt)
            text: Optional conversation text metadata
        """
        snapshot = self.save_context(seq_id, text)
        torch.save(snapshot, path)
        size_mb = sum(d.nelement() * d.element_size() for d in snapshot['kv_data'])
        size_mb /= (1024 * 1024)
        print(f"[MegaGemm] Saved KV context: seq_len={snapshot['seq_len']}, "
              f"size={size_mb:.1f}MB -> {path}")

    def restore_context_from_file(self, path: str, seq_id: int = None,
                                  max_new_tokens: int = 0) -> int:
        """
        Restore KV cache snapshot from disk.

        Args:
            path: File path (.pt) from save_context_to_file()
            seq_id: Sequence ID to use (auto-assigned if None)
            max_new_tokens: Extra tokens to pre-allocate for decode.

        Returns:
            Assigned sequence ID
        """
        snapshot = torch.load(path, weights_only=False)
        seq_id = self.restore_context(snapshot, seq_id, max_new_tokens)
        print(f"[MegaGemm] Restored KV context: seq_len={snapshot['seq_len']}, "
              f"seq_id={seq_id}")
        return seq_id

    def save_context_to_mgx(
        self,
        seq_id: int,
        path: Optional[str] = None,
        *,
        out_path: Optional[str] = None,
        text: str = None,
    ) -> Dict[str, object]:
        """
        Persist the active sequence snapshot into an MGX session_state section.

        If ``path`` is omitted and the current engine was constructed from a `.mgx`
        artifact, the engine's model path is used automatically.
        """
        target_path = path or (self._model_name if is_mgx_path(self._model_name) else None)
        if target_path is None:
            raise ValueError(
                "save_context_to_mgx requires a target .mgx path when the engine was not loaded from MGX."
            )
        snapshot = self.save_context(seq_id, text=text)
        return attach_session_state_to_mgx(
            target_path,
            snapshot,
            out_path=out_path,
        )

    def restore_context_from_mgx(
        self,
        path: Optional[str] = None,
        *,
        seq_id: int = None,
        max_new_tokens: int = 0,
    ) -> int:
        """
        Restore a runtime snapshot embedded inside an MGX session_state section.

        If ``path`` is omitted and the current engine was constructed from a `.mgx`
        artifact, the engine's model path is used automatically.
        """
        target_path = path or (self._model_name if is_mgx_path(self._model_name) else None)
        if target_path is None:
            raise ValueError(
                "restore_context_from_mgx requires a source .mgx path when the engine was not loaded from MGX."
            )
        snapshot = extract_session_state_from_mgx(target_path)
        restored_seq_id = self.restore_context(snapshot, seq_id=seq_id, max_new_tokens=max_new_tokens)
        print(
            f"[MegaGemm] Restored embedded MGX session_state: seq_len={snapshot['seq_len']}, "
            f"seq_id={restored_seq_id}"
        )
        return restored_seq_id

    def prophet_capture(
        self,
        library_dir: str,
        seq_id: int,
        *,
        text: Optional[str] = None,
        label: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Dict[str, object]:
        """Capture the active sequence into an MGX Prophet state library."""
        from .prophet import MGXProphetLibrary

        library = MGXProphetLibrary(library_dir)
        return library.capture(
            self,
            seq_id,
            text=text,
            label=label,
            metadata=metadata,
        )

    def prophet_lookup(
        self,
        library_dir: str,
        text: str,
        *,
        top_k: int = 3,
        min_similarity: float = 0.35,
        prefix_tokens: int = 16,
        require_compatible: bool = True,
    ) -> List[Dict[str, object]]:
        """Return the best Prophet matches for a new prompt."""
        from .prophet import MGXProphetLibrary

        library = MGXProphetLibrary(library_dir)
        return library.lookup(
            self,
            text,
            top_k=top_k,
            min_similarity=min_similarity,
            prefix_tokens=prefix_tokens,
            require_compatible=require_compatible,
        )

    def prophet_restore_best(
        self,
        library_dir: str,
        text: str,
        *,
        seq_id: int = None,
        max_new_tokens: int = 0,
        top_k: int = 3,
        min_similarity: float = 0.35,
        prefix_tokens: int = 16,
        require_compatible: bool = True,
    ) -> Dict[str, object]:
        """
        Restore the best Prophet snapshot for a new prompt when a compatible match exists.
        """
        from .prophet import MGXProphetLibrary

        library = MGXProphetLibrary(library_dir)
        return library.restore_best(
            self,
            text,
            seq_id=seq_id,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            min_similarity=min_similarity,
            prefix_tokens=prefix_tokens,
            require_compatible=require_compatible,
        )

    def prophet_restore_speculative(
        self,
        library_dir: str,
        text: str,
        *,
        seq_id: int = None,
        max_new_tokens: int = 0,
        top_k: int = 3,
        min_similarity: float = 0.35,
        prefix_tokens: int = 16,
        require_compatible: bool = True,
        validation_mode: str = "full_prefill",
        validation_tokens: int = 4,
        agreement_threshold: float = 1.0,
        fallback_to_prefill: bool = True,
        min_prefix_reuse_score: float = 0.55,
        min_prefix_coverage: float = 0.50,
        max_prefix_rollback_ratio: float = 0.35,
        max_prefix_tail_ratio: float = 0.50,
        use_resident_cache: bool = False,
        resident_cache_max_entries: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Speculatively restore a Prophet candidate and validate it before commit.

        `validation_mode="full_prefill"` is correctness-oriented: it builds a fresh
        baseline context for the incoming prompt, compares a short greedy continuation
        window, and commits either the Prophet candidate or the fresh fallback.

        Prefix replay routes are additionally gated by a recovery policy that can be
        tuned through `min_prefix_reuse_score`, `min_prefix_coverage`,
        `max_prefix_rollback_ratio`, and `max_prefix_tail_ratio`.
        """
        from .prophet import MGXProphetLibrary

        library = MGXProphetLibrary(library_dir)
        return library.restore_speculative(
            self,
            text,
            seq_id=seq_id,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            min_similarity=min_similarity,
            prefix_tokens=prefix_tokens,
            require_compatible=require_compatible,
            validation_mode=validation_mode,
            validation_tokens=validation_tokens,
            agreement_threshold=agreement_threshold,
            fallback_to_prefill=fallback_to_prefill,
            min_prefix_reuse_score=min_prefix_reuse_score,
            min_prefix_coverage=min_prefix_coverage,
            max_prefix_rollback_ratio=max_prefix_rollback_ratio,
            max_prefix_tail_ratio=max_prefix_tail_ratio,
            use_resident_cache=use_resident_cache,
            resident_cache_max_entries=resident_cache_max_entries,
        )

    def __repr__(self) -> str:
        return (
            f"InferenceEngine(model={self.config.hidden_size}H"
            f"/{self.config.num_hidden_layers}L, "
            f"dtype={self.dtype}, batch={self.max_batch_size}, "
            f"{self.block_manager})"
        )
