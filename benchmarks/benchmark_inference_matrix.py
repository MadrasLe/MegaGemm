"""General inference benchmark matrix for MegaGemm, MegaMesh, and HF.

Examples:
    python benchmarks/benchmark_inference_matrix.py --backend megagemm \
      --model meta-llama/Meta-Llama-3.1-8B-Instruct --hardware-label 1xl4 \
      --batch-sizes 1,8,16 --prompt-tokens 128 --max-new-tokens 128

    python benchmarks/benchmark_inference_matrix.py --backend hf \
      --model meta-llama/Meta-Llama-3.1-8B-Instruct --hardware-label 1xl4 \
      --batch-sizes 1,8,16 --prompt-tokens 128 --max-new-tokens 128

    python benchmarks/benchmark_inference_matrix.py --backend microgemm \
      --model Qwen/Qwen2.5-0.5B-Instruct --hardware-label colab-xeon \
      --device cpu --batch-sizes 1,2,4,8 --prompt-tokens 256

    python benchmarks/benchmark_inference_matrix.py --backend mesh-shard \
      --model /kaggle/input/models/qwen-lm/qwen-3/transformers/32b/1 \
      --stages "ttp://127.0.0.1:9090#s0,ttp://127.0.0.1:9091#s1" \
      --hardware-label 2xt4 --batch-sizes 1,8 --prompt-tokens 128
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch


def installed_package_versions() -> dict[str, str | None]:
    """Return direct benchmark/runtime package versions without importing them."""
    distributions = (
        "megagemm",
        "torch",
        "triton",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "safetensors",
        "sentencepiece",
        "vllm",
        "flash-attn",
        "causal-conv1d",
        "numpy",
        "psutil",
    )
    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def git_snapshot() -> dict[str, Any]:
    """Best-effort source revision metadata for reproducible reports."""
    snapshot: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        snapshot["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        snapshot["dirty"] = bool(status.strip())
    except Exception:
        pass
    return snapshot


def nvidia_smi_snapshot() -> dict[str, Any]:
    """Capture driver and clock state without making nvidia-smi mandatory."""
    query = (
        "index,name,uuid,driver_version,memory.total,pstate,"
        "clocks.current.sm,clocks.current.memory,power.limit"
    )
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        ).strip()
        return {"available": True, "query": query, "rows": output.splitlines()}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"expected at least one integer in {raw!r}")
    return values


def runtime_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def gpu_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "count": 0, "devices": []}
    devices = []
    for idx in range(torch.cuda.device_count()):
        free = total = None
        try:
            with torch.cuda.device(idx):
                free, total = torch.cuda.mem_get_info()
        except Exception:
            pass
        props = torch.cuda.get_device_properties(idx)
        devices.append(
            {
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "capability": list(torch.cuda.get_device_capability(idx)),
                "total_gb": props.total_memory / 1024**3,
                "free_gb": (free / 1024**3) if free is not None else None,
                "multiprocessors": props.multi_processor_count,
            }
        )
    return {"available": True, "count": len(devices), "devices": devices}


def cpu_snapshot() -> dict[str, Any]:
    row: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
    }
    try:
        import psutil

        mem = psutil.virtual_memory()
        row.update(
            {
                "ram_total_gb": mem.total / 1024**3,
                "ram_available_gb": mem.available / 1024**3,
            }
        )
    except Exception:
        pass
    return row


def set_tokenizer_padding(tokenizer) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass


def load_tokenizer(model_or_tokenizer: str, *, local_files_only: bool = False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_or_tokenizer,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    set_tokenizer_padding(tokenizer)
    return tokenizer


def build_prompt(tokenizer, target_tokens: int, *, prefix: str = "") -> tuple[str, int]:
    target_tokens = max(1, int(target_tokens))
    seed = (
        "Inference benchmark passage. MegaGemm measures decode throughput, "
        "prefill pressure, KV cache behavior, batching, and long context memory. "
    )
    text = (prefix + " " + seed).strip() if prefix else seed
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < target_tokens:
        text = f"{text} {seed}"
        ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:target_tokens]
    prompt = tokenizer.decode(ids, skip_special_tokens=True)
    actual = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt, actual


def build_prompts(tokenizer, batch_size: int, prompt_tokens: int) -> tuple[list[str], int]:
    prompts = []
    actuals = []
    for idx in range(batch_size):
        prompt, actual = build_prompt(
            tokenizer,
            prompt_tokens,
            prefix=f"Request {idx}:",
        )
        prompts.append(prompt)
        actuals.append(actual)
    return prompts, int(sum(actuals))


def count_generated_text_tokens(tokenizer, outputs: list[str]) -> int:
    return sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in outputs)


def current_cuda_memory() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    idx = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info()
    return {
        "device_index": idx,
        "allocated_gb": torch.cuda.memory_allocated(idx) / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved(idx) / 1024**3,
        "peak_allocated_gb": torch.cuda.max_memory_allocated(idx) / 1024**3,
        "free_gb": free / 1024**3,
        "total_gb": total / 1024**3,
    }


def ensure_executable(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        if not path.exists() or path.is_dir():
            return
        path.chmod(path.stat().st_mode | 0o111)
    except OSError:
        return


def run_capture(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    exe = Path(cmd[0])
    ensure_executable(exe)
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except PermissionError as exc:
        ensure_executable(exe)
        try:
            return subprocess.run(
                cmd,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except PermissionError as retry_exc:
            raise RuntimeError(
                "permission denied while executing command after chmod +x retry\n"
                f"cmd: {' '.join(cmd)}\n"
                f"cwd: {cwd or Path.cwd()}\n"
                "If this path is on a noexec mount, copy/build MicroGemm under /content and run from there."
            ) from retry_exc


def is_colab_drive_path(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    return str(resolved).startswith("/content/drive/")


def stage_microgemm_tree(source_micro: Path, cache_dir: Path) -> Path:
    stage_micro = cache_dir / "build" / "microgemm"
    try:
        if stage_micro.resolve() == source_micro.resolve():
            return source_micro
    except OSError:
        pass

    if stage_micro.exists():
        shutil.rmtree(stage_micro)

    def ignore_build_outputs(_dirname: str, names: list[str]) -> set[str]:
        ignored = {"__pycache__", ".pytest_cache", "build", "build-host", "build-android-arm64"}
        ignored.update(
            name
            for name in names
            if (Path(_dirname) / name).is_file()
            and (
                name.endswith((".o", ".obj", ".dll", ".so", ".dylib", ".exe"))
                or name in {"microgemm", "microgemm-convert", "microgemm-text"}
            )
        )
        return ignored

    stage_micro.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_micro, stage_micro, ignore=ignore_build_outputs)
    return stage_micro


def run_checked(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = run_capture(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            "command failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"cwd: {cwd}\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )
    return result


def megagemm_diagnostics(engine) -> dict[str, Any]:
    model = getattr(engine, "model", None)
    diagnostics: dict[str, Any] = {}
    config = getattr(model, "config", None)
    if config is not None:
        layer_types = list(getattr(config, "layer_types", ()) or ())
        kv_cache_layers = list(
            getattr(config, "kv_cache_layer_indices", ()) or ()
        )
        diagnostics["model_topology"] = {
            "model_type": str(getattr(config, "model_type", "") or ""),
            "num_hidden_layers": int(
                getattr(config, "num_hidden_layers", 0) or 0
            ),
            "hidden_size": int(getattr(config, "hidden_size", 0) or 0),
            "num_attention_heads": int(
                getattr(config, "num_attention_heads", 0) or 0
            ),
            "num_key_value_heads": int(
                getattr(config, "num_key_value_heads", 0) or 0
            ),
            "num_kv_shared_layers": int(
                getattr(config, "num_kv_shared_layers", 0) or 0
            ),
            "kv_cache_layers": len(kv_cache_layers),
            "sliding_attention_layers": layer_types.count(
                "sliding_attention"
            ),
            "full_attention_layers": layer_types.count("full_attention"),
            "linear_attention_layers": layer_types.count(
                "linear_attention"
            ),
            "head_dims": sorted(
                {
                    int(value)
                    for value in (
                        getattr(config, "per_layer_head_dims", ()) or ()
                    )
                }
            ),
        }
    runtime_stats = getattr(model, "decode_runtime_stats", None)
    if callable(runtime_stats):
        try:
            diagnostics["decode_runtime_stats"] = runtime_stats()
        except Exception as exc:
            diagnostics["decode_runtime_stats_error"] = str(exc)
    decode_timing = getattr(model, "get_last_decode_timing", None)
    if callable(decode_timing):
        try:
            last_timing = decode_timing()
            if last_timing:
                diagnostics["last_decode_timing"] = last_timing
        except Exception as exc:
            diagnostics["last_decode_timing_error"] = str(exc)
    return diagnostics


def metric_row(
    *,
    args: argparse.Namespace,
    scenario: str,
    batch_size: int,
    prompt_tokens_requested: int,
    prompt_tokens_actual: int,
    max_new_tokens: int,
    elapsed_s: float,
    generated_tokens: int,
    ok: bool,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_tokens = prompt_tokens_actual + generated_tokens
    row = {
        "ok": ok,
        "error": error,
        "backend": args.backend,
        "hardware_label": args.hardware_label,
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "dtype": args.dtype,
        "quantize": args.quantize or "none",
        "scenario": scenario,
        "batch_size": batch_size,
        "prompt_tokens_requested_per_request": prompt_tokens_requested,
        "prompt_tokens_actual_total": prompt_tokens_actual,
        "max_new_tokens_per_request": max_new_tokens,
        "generated_tokens": generated_tokens,
        "elapsed_s": elapsed_s,
        "output_tps": generated_tokens / elapsed_s if elapsed_s > 0 else 0.0,
        "prefill_tps_coarse": prompt_tokens_actual / elapsed_s if elapsed_s > 0 else 0.0,
        "combined_tps_coarse": total_tokens / elapsed_s if elapsed_s > 0 else 0.0,
        "tokens_per_request_s": (
            generated_tokens / elapsed_s / batch_size if elapsed_s > 0 and batch_size else 0.0
        ),
        "kv_offload": bool(args.kv_offload),
        "num_blocks": args.num_blocks,
        "num_cpu_blocks": args.num_cpu_blocks,
        "gpu_window": args.gpu_window,
        "max_seq_len": args.max_seq_len,
        "microbatch_size": args.microbatch_size,
        "mesh_max_batch_size": args.mesh_max_batch_size,
        "hf_mode": args.hf_mode,
        "ignore_eos": bool(args.ignore_eos),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if torch.cuda.is_available():
        row["cuda_memory"] = current_cuda_memory()
    if extra:
        row.update(extra)
    return row


def _is_cache_reuse_sample(row: dict[str, Any]) -> bool:
    """Rows that should be compared against prefix-cache/cache-hit baselines."""
    backend = str(row.get("backend") or "")
    repeat_index = int(row.get("repeat_index") or 0)
    if backend == "megagemm-prophet":
        prophet = row.get("prophet")
        if isinstance(prophet, dict):
            return prophet.get("mode") == "cached_restore_batch_decode"
        return repeat_index > 1
    if backend == "vllm":
        vllm = row.get("vllm")
        prefix_caching = isinstance(vllm, dict) and bool(vllm.get("prefix_caching"))
        return prefix_caching and repeat_index > 1
    return False


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["backend"],
            row["hardware_label"],
            row["scenario"],
            row["batch_size"],
            row["prompt_tokens_requested_per_request"],
            row["max_new_tokens_per_request"],
            row["kv_offload"],
        )
        grouped.setdefault(key, []).append(row)

    summary = []
    for key, samples in grouped.items():
        ok_samples = [row for row in samples if row.get("ok")]
        first_ok_sample = min(
            ok_samples,
            key=lambda row: int(row.get("repeat_index") or 0),
            default=None,
        )
        tps = [float(row["output_tps"]) for row in ok_samples]
        elapsed = [float(row["elapsed_s"]) for row in ok_samples]
        generated = [int(row["generated_tokens"]) for row in ok_samples]
        steady_samples = [row for row in ok_samples if _is_cache_reuse_sample(row)]
        steady_tps = [float(row["output_tps"]) for row in steady_samples]
        steady_elapsed = [float(row["elapsed_s"]) for row in steady_samples]
        scheduler_stats = [
            row.get("scheduler_stats") for row in ok_samples
            if isinstance(row.get("scheduler_stats"), dict)
        ]
        scheduler_decode_tps = [
            float(stats["decode_throughput"]) for stats in scheduler_stats
            if stats.get("decode_throughput") is not None
        ]
        prefill_ms = [
            float(stats["prefill_time_ms"]) for stats in scheduler_stats
            if stats.get("prefill_time_ms") is not None
        ]
        decode_ms = [
            float(stats["decode_time_ms"]) for stats in scheduler_stats
            if stats.get("decode_time_ms") is not None
        ]
        decode_wall_tps = []
        prefill_tps = []
        prophet_restore_ms = [
            float(row["prophet_restore_s"]) * 1000.0 for row in ok_samples
            if row.get("prophet_restore_s") is not None
        ]
        prophet_decode_ms = [
            float(row["prophet_decode_s"]) * 1000.0 for row in ok_samples
            if row.get("prophet_decode_s") is not None
        ]
        prophet_decode_tps = [
            max(
                0.0,
                float(row["generated_tokens"]) - float(row.get("prophet_bootstrap_tokens") or 0.0),
            ) / float(row["prophet_decode_s"])
            for row in ok_samples
            if row.get("prophet_decode_s") is not None
            and float(row.get("prophet_decode_s") or 0.0) > 0.0
        ]
        steady_prophet_restore_ms = [
            float(row["prophet_restore_s"]) * 1000.0 for row in steady_samples
            if row.get("prophet_restore_s") is not None
        ]
        steady_prophet_decode_ms = [
            float(row["prophet_decode_s"]) * 1000.0 for row in steady_samples
            if row.get("prophet_decode_s") is not None
        ]
        steady_prophet_decode_tps = [
            max(
                0.0,
                float(row["generated_tokens"]) - float(row.get("prophet_bootstrap_tokens") or 0.0),
            ) / float(row["prophet_decode_s"])
            for row in steady_samples
            if row.get("prophet_decode_s") is not None
            and float(row.get("prophet_decode_s") or 0.0) > 0.0
        ]
        for row in ok_samples:
            stats = row.get("scheduler_stats")
            if not isinstance(stats, dict):
                continue
            decode_s = float(stats.get("decode_time_ms") or 0.0) / 1000.0
            prefill_s = float(stats.get("prefill_time_ms") or 0.0) / 1000.0
            if decode_s > 0:
                decode_wall_tps.append(float(row["generated_tokens"]) / decode_s)
            if prefill_s > 0:
                prefill_tps.append(float(row["prompt_tokens_actual_total"]) / prefill_s)
        summary.append(
            {
                "backend": key[0],
                "hardware_label": key[1],
                "scenario": key[2],
                "batch_size": key[3],
                "prompt_tokens_requested_per_request": key[4],
                "max_new_tokens_per_request": key[5],
                "kv_offload": key[6],
                "samples": len(samples),
                "ok_samples": len(ok_samples),
                "first_output_tps": (
                    float(first_ok_sample["output_tps"])
                    if first_ok_sample is not None else 0.0
                ),
                "first_elapsed_s": (
                    float(first_ok_sample["elapsed_s"])
                    if first_ok_sample is not None else 0.0
                ),
                "median_output_tps": statistics.median(tps) if tps else 0.0,
                "steady_ok_samples": len(steady_samples),
                "median_steady_output_tps": (
                    statistics.median(steady_tps) if steady_tps else 0.0
                ),
                "median_steady_elapsed_s": (
                    statistics.median(steady_elapsed) if steady_elapsed else 0.0
                ),
                "best_output_tps": max(tps) if tps else 0.0,
                "worst_output_tps": min(tps) if tps else 0.0,
                "median_scheduler_decode_tps": (
                    statistics.median(scheduler_decode_tps) if scheduler_decode_tps else 0.0
                ),
                "median_decode_wall_tps": (
                    statistics.median(decode_wall_tps) if decode_wall_tps else 0.0
                ),
                "median_prefill_tps": statistics.median(prefill_tps) if prefill_tps else 0.0,
                "median_prefill_time_ms": statistics.median(prefill_ms) if prefill_ms else 0.0,
                "median_decode_time_ms": statistics.median(decode_ms) if decode_ms else 0.0,
                "median_prophet_restore_time_ms": (
                    statistics.median(prophet_restore_ms) if prophet_restore_ms else 0.0
                ),
                "median_prophet_decode_time_ms": (
                    statistics.median(prophet_decode_ms) if prophet_decode_ms else 0.0
                ),
                "median_prophet_decode_tps": (
                    statistics.median(prophet_decode_tps) if prophet_decode_tps else 0.0
                ),
                "median_steady_prophet_restore_time_ms": (
                    statistics.median(steady_prophet_restore_ms)
                    if steady_prophet_restore_ms else 0.0
                ),
                "median_steady_prophet_decode_time_ms": (
                    statistics.median(steady_prophet_decode_ms)
                    if steady_prophet_decode_ms else 0.0
                ),
                "median_steady_prophet_decode_tps": (
                    statistics.median(steady_prophet_decode_tps)
                    if steady_prophet_decode_tps else 0.0
                ),
                "median_elapsed_s": statistics.median(elapsed) if elapsed else 0.0,
                "median_generated_tokens": statistics.median(generated) if generated else 0,
                "errors": [row.get("error") for row in samples if not row.get("ok")],
            }
        )
    summary.sort(
        key=lambda row: (
            str(row["backend"]),
            str(row["hardware_label"]),
            str(row["scenario"]),
            int(row["batch_size"]),
            int(row["prompt_tokens_requested_per_request"]),
        )
    )
    return summary


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    stem = f"{run_id}_{args.hardware_label}_{args.backend}"
    raw_path = out_dir / f"{stem}.jsonl"
    summary_path = out_dir / f"{stem}_summary.json"
    csv_path = out_dir / f"{stem}_summary.csv"

    with raw_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_rows = summarize(rows)
    payload = {
        "benchmark": "inference_matrix",
        "args": vars(args),
        "system": {
            "cpu": cpu_snapshot(),
            "gpu": gpu_snapshot(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "packages": installed_package_versions(),
            "git": git_snapshot(),
            "nvidia_smi": nvidia_smi_snapshot(),
            "env": {
                key: os.environ.get(key)
                for key in [
                    "CUDA_VISIBLE_DEVICES",
                    "MEGAGEMM_FLAT_DECODE",
                    "MEGAGEMM_DISABLE_CUDA_RMSNORM",
                    "MEGAGEMM_DECODE_PREFER_STEP",
                    "MEGAGEMM_DECODE_CUDA_GRAPHS",
                    "MEGAGEMM_DECODE_CUDA_GRAPHS_PREFER_STEP",
                    "MEGAGEMM_DECODE_GRAPH_TOKEN_BURST",
                    "MEGAGEMM_DECODE_GRAPH_PERSISTENT_TOKEN_FEEDBACK",
                    "MEGAGEMM_NATIVE_DECODE_GRAPH_BURST",
                    "MEGAGEMM_DECODE_UNROLLED_GRAPH_BURST",
                    "MEGAGEMM_MULTI_STEP_BURST_BATCH",
                    "MEGAGEMM_BENCHMARK_TOKEN_DIGEST",
                    "MEGAGEMM_REUSE_REQUEST_SCHEDULER",
                    "MEGAGEMM_GEMMA4_FUSED_QKV_DECODE",
                    "MEGAGEMM_GEMMA4_FUSED_GATEUP_DECODE",
                    "MEGAGEMM_GEMMA4_DEEPFUSION_MLP_DECODE",
                    "MEGAGEMM_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE",
                    "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_PREFILL",
                    "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_GROUP_HEADS",
                    "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_BLOCK_M",
                    "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_BLOCK_N",
                    "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_NUM_WARPS",
                    "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_NUM_STAGES",
                    "MEGAGEMM_GEMMA4_E2B_B8_PREFILL_GATED_ACTIVATION",
                    "MEGAGEMM_SKIP_CUDA",
                    "MICROGEMM_CACHE_DIR",
                    "OMP_NUM_THREADS",
                    "HF_HOME",
                    "TRANSFORMERS_CACHE",
                ]
            },
        },
        "rows": summary_rows,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if summary_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    return raw_path, summary_path, csv_path


def load_megagemm_runner(args: argparse.Namespace, tokenizer):
    from megagemm.engine import InferenceEngine

    engine = InferenceEngine(
        args.model,
        dtype=runtime_dtype(args.dtype),
        device=args.device,
        quantize=args.quantize,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        kv_alloc=args.kv_alloc,
        kv_offload=args.kv_offload,
        num_cpu_blocks=args.num_cpu_blocks,
        gpu_window=args.gpu_window,
        cache_dir=args.cache_dir,
        mgx_prefer_payload_cache=args.mgx_prefer_payload_cache,
        mgx_payload_cache_dir=args.mgx_payload_cache_dir,
    )

    def run(prompts: list[str], max_new_tokens: int) -> dict[str, Any]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        sync_cuda()
        start = time.perf_counter()
        outputs = engine.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            ignore_eos=args.ignore_eos,
            decode_outputs=False,
        )
        sync_cuda()
        elapsed_s = time.perf_counter() - start
        scheduler = getattr(engine, "_last_scheduler", None)
        scheduler_stats = scheduler.get_stats() if scheduler is not None else {}
        token_digest = None
        generated_token_lengths = None
        if (
            os.environ.get("MEGAGEMM_BENCHMARK_TOKEN_DIGEST", "").strip().lower()
            in {"1", "true", "yes", "on"}
            and scheduler is not None
        ):
            completed = sorted(
                getattr(scheduler, "_completed", ()) or (),
                key=lambda request: int(request.request_id),
            )
            token_rows = [
                [int(token_id) for token_id in request.generated_ids]
                for request in completed
            ]
            encoded = json.dumps(
                token_rows,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
            token_digest = hashlib.sha256(encoded).hexdigest()
            generated_token_lengths = [len(row) for row in token_rows]
        generated_tokens = int(
            scheduler_stats.get("total_tokens")
            or count_generated_text_tokens(tokenizer, list(outputs))
            or (len(prompts) * max_new_tokens)
        )
        return {
            "elapsed_s": elapsed_s,
            "generated_tokens": generated_tokens,
            "extra": {
                "scheduler_stats": scheduler_stats,
                "generated_token_digest": token_digest,
                "generated_token_lengths": generated_token_lengths,
                "engine_init_timing": getattr(engine, "_init_timing", None),
                **megagemm_diagnostics(engine),
            },
        }

    # Benchmark gates that rotate experimental policies inside one loaded model
    # need explicit access without relying on CPython closure internals.
    run._megagemm_engine = engine
    return run


def load_megagemm_prophet_runner(args: argparse.Namespace, tokenizer):
    from megagemm.engine import InferenceEngine
    from megagemm.engine.prophet import MGXProphetLibrary
    from megagemm.engine.scheduler import Request, RequestStatus, Scheduler
    from megagemm.engine.sampling import sample_logits

    prophet_dir = Path(args.prophet_dir or (Path(args.out_dir) / f"{args.run_id or 'run'}_prophet_library"))
    if args.prophet_reset_dir and prophet_dir.exists():
        shutil.rmtree(prophet_dir)
    prophet_dir.mkdir(parents=True, exist_ok=True)

    engine = InferenceEngine(
        args.model,
        dtype=runtime_dtype(args.dtype),
        device=args.device,
        quantize=args.quantize,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        kv_alloc=args.kv_alloc,
        kv_offload=args.kv_offload,
        num_cpu_blocks=args.num_cpu_blocks,
        gpu_window=args.gpu_window,
        cache_dir=args.cache_dir,
        mgx_prefer_payload_cache=args.mgx_prefer_payload_cache,
        mgx_payload_cache_dir=args.mgx_payload_cache_dir,
    )

    captured_prompts: set[str] = set()
    captured_live_prefix_seq_ids: dict[str, int] = {}
    pending_hook_captures: dict[str, dict[str, Any]] = {}
    capture_count = 0
    prophet_seq_counter = 0

    def next_prophet_seq_id() -> int:
        nonlocal prophet_seq_counter
        prophet_seq_counter += 1
        return -prophet_seq_counter

    def capture_prompt(prompt: str, max_new_tokens: int, *, label: str) -> None:
        nonlocal capture_count
        if prompt in captured_prompts:
            return
        seq_id = next_prophet_seq_id()
        keep_live_prefix = False
        prefill = engine.prefill_context(
            prompt,
            seq_id=seq_id,
            max_new_tokens=max_new_tokens,
        )
        try:
            engine.prophet_capture(
                str(prophet_dir),
                seq_id,
                text=prompt,
                label=label,
                metadata={
                    "benchmark": "inference_matrix",
                    "capture_index": capture_count,
                    "prompt_len": int(prefill.get("prompt_len", 0) or 0),
                },
            )
            keep_live_prefix = bool(args.prophet_live_prefix_cache)
        finally:
            if keep_live_prefix:
                captured_live_prefix_seq_ids[prompt] = seq_id
            else:
                engine.free_sequence(seq_id)
        captured_prompts.add(prompt)
        capture_count += 1

    def capture_prefilled_request(req: Request, pending_logits: torch.Tensor) -> None:
        nonlocal capture_count
        if not args.prophet_live_prefix_cache:
            return
        prompt = str((req.metadata or {}).get("prompt") or "")
        if not prompt or prompt in captured_prompts:
            return

        seq_id = next_prophet_seq_id()
        forked = False
        try:
            engine.block_manager.fork_sequence_prefix(
                int(req.seq_id),
                seq_id,
                extra_tokens=0,
            )
            forked = True
            engine._set_sequence_runtime_state(
                seq_id,
                token_ids=list(req.prompt_ids),
                pending_next_logits=pending_logits,
            )
            captured_live_prefix_seq_ids[prompt] = seq_id
            pending_hook_captures[prompt] = {
                "seq_id": seq_id,
                "capture_index": capture_count,
                "prompt_len": int(len(req.prompt_ids)),
                "label": f"matrix-prime-{capture_count}",
            }
            captured_prompts.add(prompt)
            capture_count += 1
        except Exception as exc:
            if args.verbose:
                print(f"  Prophet prefill-hook capture skipped: {type(exc).__name__}: {exc}")
            if forked:
                try:
                    engine.free_sequence(seq_id)
                except Exception:
                    pass

    def flush_prefill_hook_captures() -> None:
        for prompt, capture in list(pending_hook_captures.items()):
            seq_id = int(capture["seq_id"])
            try:
                engine.prophet_capture(
                    str(prophet_dir),
                    seq_id,
                    text=prompt,
                    label=str(capture["label"]),
                    metadata={
                        "benchmark": "inference_matrix",
                        "capture_index": int(capture["capture_index"]),
                        "prompt_len": int(capture["prompt_len"]),
                        "capture_source": "batched_prefill_hook",
                    },
                )
            except Exception as exc:
                if args.verbose:
                    print(f"  Prophet hook snapshot save skipped: {type(exc).__name__}: {exc}")
            finally:
                pending_hook_captures.pop(prompt, None)

    def run_fresh_and_prime(prompts: list[str], max_new_tokens: int) -> dict[str, Any]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        sync_cuda()
        start = time.perf_counter()
        outputs = engine.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            ignore_eos=args.ignore_eos,
            prefill_capture_hook=capture_prefilled_request,
            decode_outputs=False,
        )
        sync_cuda()
        elapsed_s = time.perf_counter() - start
        scheduler = getattr(engine, "_last_scheduler", None)
        scheduler_stats = scheduler.get_stats() if scheduler is not None else {}
        generated_tokens = int(
            scheduler_stats.get("total_tokens")
            or count_generated_text_tokens(tokenizer, list(outputs))
            or (len(prompts) * max_new_tokens)
        )

        flush_prefill_hook_captures()
        for idx, prompt in enumerate(prompts):
            capture_prompt(prompt, max_new_tokens, label=f"matrix-prime-{idx}")

        return {
            "elapsed_s": elapsed_s,
            "generated_tokens": generated_tokens,
            "extra": {
                "scheduler_stats": scheduler_stats,
                "engine_init_timing": getattr(engine, "_init_timing", None),
                **megagemm_diagnostics(engine),
                "prophet": {
                    "mode": "miss_then_prime",
                    "library_dir": str(prophet_dir),
                    "captured_prompts": len(captured_prompts),
                },
            },
        }

    def run_cached(prompts: list[str], max_new_tokens: int) -> dict[str, Any]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if args.prophet_prime_before_measure:
            for idx, prompt in enumerate(prompts):
                capture_prompt(prompt, max_new_tokens, label=f"matrix-prime-{idx}")

        missing = [prompt for prompt in prompts if prompt not in captured_prompts]
        if missing:
            return run_fresh_and_prime(prompts, max_new_tokens)

        eos_ids = set()
        if not args.ignore_eos:
            raw_eos = tokenizer.eos_token_id
            if isinstance(raw_eos, (list, tuple, set)):
                eos_ids.update(int(token_id) for token_id in raw_eos)
            elif raw_eos is not None:
                eos_ids.add(int(raw_eos))

        sync_cuda()
        start = time.perf_counter()
        scheduler = Scheduler(
            model=engine.model,
            block_manager=engine.block_manager,
            max_batch_size=engine.max_batch_size,
            device=engine.device,
            materialize_generated_tokens=False,
        )
        restored_records: list[dict[str, Any]] = []
        seq_ids: list[int] = []
        batch_exact_restore_used = False
        live_prefix_cache_used = False
        restore_start = time.perf_counter()

        def enqueue_restored_request(idx: int, restore: dict[str, Any]) -> None:
            seq_id = int(restore["seq_id"])
            if seq_id not in engine.block_manager.block_tables:
                raise RuntimeError(f"Prophet did not leave seq_id={seq_id} live")
            if seq_id not in engine._seq_pending_logits:
                raise RuntimeError(f"Prophet restore seq_id={seq_id} has no pending logits")

            prompt_ids = list(engine._seq_token_ids.get(seq_id, []))
            if not prompt_ids:
                raise RuntimeError(f"Prophet restore seq_id={seq_id} has no token history")

            pending_logits = engine._seq_pending_logits[seq_id].to(engine.device).unsqueeze(0)
            first_token = int(sample_logits(pending_logits, 0.0, 0, 1.0).item())

            req = Request(
                request_id=idx + 1,
                seq_id=seq_id,
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                stop_token_ids=eos_ids,
                t_start=start,
            )
            req.generated_ids.append(first_token)
            req.status = RequestStatus.RUNNING
            req.t_prefill_done = time.perf_counter()
            seq_ids.append(seq_id)
            restored_records.append(
                {
                    "seq_id": seq_id,
                    "committed_source": restore.get("committed_source"),
                    "reason": restore.get("reason"),
                    "restored": bool(restore.get("restored")),
                    "speculative_accepted": bool(restore.get("speculative_accepted")),
                    "validation_mode": (restore.get("validation") or {}).get("mode"),
                }
            )

            if req.is_done:
                req.status = RequestStatus.FINISHED
                req.t_end = time.perf_counter()
                engine.block_manager.free_sequence(req.seq_id)
                scheduler._completed.append(req)
            else:
                scheduler._running[req.seq_id] = req

        try:
            batch_restores = None
            if args.prophet_live_prefix_cache:
                planned_seq_ids = [next_prophet_seq_id() for _ in prompts]
                batch_restores = []
                for prompt, planned_seq_id in zip(prompts, planned_seq_ids):
                    source_seq_id = captured_live_prefix_seq_ids.get(prompt)
                    if source_seq_id is None:
                        batch_restores = None
                        break
                    restored_seq_id = engine.fork_context_prefix(
                        source_seq_id,
                        seq_id=planned_seq_id,
                        max_new_tokens=max_new_tokens,
                    )
                    batch_restores.append(
                        {
                            "restored": True,
                            "seq_id": restored_seq_id,
                            "committed_source": "prophet_live_prefix_cache",
                            "speculative_accepted": True,
                            "reason": "live_prefix_fork",
                            "validation": {"mode": "none"},
                        }
                    )
                live_prefix_cache_used = batch_restores is not None

            if batch_restores is None and (
                args.prophet_batch_exact_restore
                and args.prophet_validation_mode == "none"
            ):
                planned_seq_ids = [next_prophet_seq_id() for _ in prompts]
                batch_restores = MGXProphetLibrary(prophet_dir).restore_exact_batch(
                    engine,
                    prompts,
                    seq_ids=planned_seq_ids,
                    max_new_tokens=max_new_tokens,
                    top_k=args.prophet_top_k,
                    min_similarity=args.prophet_min_similarity,
                    prefix_tokens=args.prophet_prefix_tokens,
                    require_compatible=True,
                    use_resident_cache=args.prophet_resident_cache,
                    resident_cache_max_entries=args.prophet_resident_cache_max_entries,
                )
                batch_exact_restore_used = batch_restores is not None

            if batch_restores is not None:
                for idx, restore in enumerate(batch_restores):
                    enqueue_restored_request(idx, restore)
            else:
                for idx, prompt in enumerate(prompts):
                    restore = engine.prophet_restore_speculative(
                        str(prophet_dir),
                        prompt,
                        seq_id=next_prophet_seq_id(),
                        max_new_tokens=max_new_tokens,
                        top_k=args.prophet_top_k,
                        min_similarity=args.prophet_min_similarity,
                        prefix_tokens=args.prophet_prefix_tokens,
                        require_compatible=True,
                        validation_mode=args.prophet_validation_mode,
                        validation_tokens=args.prophet_validation_tokens,
                        agreement_threshold=args.prophet_agreement_threshold,
                        fallback_to_prefill=args.prophet_fallback_to_prefill,
                        min_prefix_reuse_score=args.prophet_min_prefix_reuse_score,
                        min_prefix_coverage=args.prophet_min_prefix_coverage,
                        max_prefix_rollback_ratio=args.prophet_max_prefix_rollback_ratio,
                        max_prefix_tail_ratio=args.prophet_max_prefix_tail_ratio,
                        use_resident_cache=args.prophet_resident_cache,
                        resident_cache_max_entries=args.prophet_resident_cache_max_entries,
                    )
                    enqueue_restored_request(idx, restore)

            scheduler._batch_changed = True
            sync_cuda()
            restore_s = time.perf_counter() - restore_start
            decode_start = time.perf_counter()
            while scheduler.has_pending():
                scheduler.step()
            sync_cuda()
            decode_s = time.perf_counter() - decode_start
            elapsed_s = time.perf_counter() - start

            scheduler_stats = scheduler.get_stats()
            generated_tokens = int(
                scheduler_stats.get("total_tokens")
                or sum(req.num_generated for req in scheduler._completed)
                or (len(prompts) * max_new_tokens)
            )
            committed_counts: dict[str, int] = {}
            for record in restored_records:
                source = str(record.get("committed_source"))
                committed_counts[source] = committed_counts.get(source, 0) + 1
            return {
                "elapsed_s": elapsed_s,
                "generated_tokens": generated_tokens,
                "extra": {
                    "scheduler_stats": scheduler_stats,
                    "engine_init_timing": getattr(engine, "_init_timing", None),
                    "prophet_restore_s": restore_s,
                    "prophet_decode_s": decode_s,
                    "prophet_bootstrap_tokens": len(restored_records),
                    **megagemm_diagnostics(engine),
                    "prophet": {
                        "mode": "cached_restore_batch_decode",
                        "library_dir": str(prophet_dir),
                        "captured_prompts": len(captured_prompts),
                        "batch_exact_restore": batch_exact_restore_used,
                        "live_prefix_cache": live_prefix_cache_used,
                        "committed_source_counts": committed_counts,
                        "restore_records": restored_records,
                        "snapshot_cache": MGXProphetLibrary.snapshot_cache_stats(),
                        "resident_cache": MGXProphetLibrary.resident_cache_stats(engine),
                    },
                },
            }
        finally:
            for seq_id in seq_ids:
                try:
                    engine.free_sequence(seq_id)
                except Exception:
                    pass
                try:
                    engine._clear_sequence_runtime_state(seq_id)
                except Exception:
                    pass

    return run_cached


def load_hf_runner(args: argparse.Namespace, tokenizer):
    from transformers import AutoModelForCausalLM

    dtype = runtime_dtype(args.dtype)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.hf_device_map:
        model_kwargs["device_map"] = args.hf_device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if not args.hf_device_map:
        model.to(args.device)
    model.eval()

    def _move_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        target = getattr(model, "device", None)
        if target is None:
            target = torch.device(args.device)
        return {key: value.to(target) for key, value in batch.items()}

    def run_batched(prompts: list[str], max_new_tokens: int) -> dict[str, Any]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        batch = tokenizer(prompts, return_tensors="pt", padding=True)
        input_width = int(batch["input_ids"].shape[1])
        batch = _move_batch(batch)
        sync_cuda()
        start = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=None if args.ignore_eos else tokenizer.eos_token_id,
                use_cache=True,
            )
        sync_cuda()
        elapsed_s = time.perf_counter() - start
        gen = out[:, input_width:]
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            generated_tokens = int((gen != pad_id).sum().item())
        else:
            generated_tokens = int(gen.numel())
        return {"elapsed_s": elapsed_s, "generated_tokens": generated_tokens, "extra": {}}

    def run_sequential(prompts: list[str], max_new_tokens: int) -> dict[str, Any]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        total_generated = 0
        sync_cuda()
        start = time.perf_counter()
        for prompt in prompts:
            batch = tokenizer([prompt], return_tensors="pt", padding=True)
            input_width = int(batch["input_ids"].shape[1])
            batch = _move_batch(batch)
            with torch.inference_mode():
                out = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=None if args.ignore_eos else tokenizer.eos_token_id,
                    use_cache=True,
                )
            gen = out[:, input_width:]
            pad_id = tokenizer.pad_token_id
            if pad_id is not None:
                total_generated += int((gen != pad_id).sum().item())
            else:
                total_generated += int(gen.numel())
        sync_cuda()
        elapsed_s = time.perf_counter() - start
        return {"elapsed_s": elapsed_s, "generated_tokens": total_generated, "extra": {}}

    return run_sequential if args.hf_mode == "sequential" else run_batched


def microgemm_default_cache_dir() -> Path:
    raw = os.environ.get("MICROGEMM_CACHE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    if Path("/content").exists():
        return Path("/content/microgemm_cache")
    return ROOT / ".cache" / "microgemm"


def ensure_microgemm_binaries(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    source_micro_dir = ROOT / "microgemm"
    micro_dir = (
        stage_microgemm_tree(source_micro_dir, microgemm_default_cache_dir())
        if is_colab_drive_path(source_micro_dir)
        else source_micro_dir
    )
    suffix = ".exe" if sys.platform == "win32" else ""
    cli_bin = micro_dir / f"microgemm{suffix}"
    convert_bin = micro_dir / f"microgemm-convert{suffix}"
    text_bin = micro_dir / f"microgemm-text{suffix}"
    bins = (cli_bin, convert_bin, text_bin)
    bins_ready = all(path.exists() for path in bins)

    sources: list[Path] = []
    for dirname in ("src", "include"):
        root = micro_dir / dirname
        if root.exists():
            sources.extend(path for path in root.rglob("*") if path.is_file())
    for filename in ("Makefile", "CMakeLists.txt", "build.ps1"):
        path = micro_dir / filename
        if path.exists():
            sources.append(path)

    needs_build = not bins_ready
    if bins_ready and sources:
        newest_source = max(path.stat().st_mtime for path in sources)
        oldest_bin = min(path.stat().st_mtime for path in bins)
        needs_build = newest_source > oldest_bin

    if not needs_build:
        for path in bins:
            ensure_executable(path)
        return cli_bin, convert_bin, text_bin

    if args.microgemm_no_build:
        missing = [str(path) for path in bins if not path.exists()]
        reason = f"missing: {missing}" if missing else "sources are newer than binaries"
        raise RuntimeError(
            "MicroGemm binaries need to be built but --microgemm-no-build was set "
            f"({reason}). Expected {cli_bin}, {convert_bin}, and {text_bin}."
        )
    if sys.platform == "win32":
        build_script = micro_dir / "build.ps1"
        if not build_script.exists():
            raise RuntimeError(f"missing MicroGemm build script: {build_script}")
        run_checked(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(build_script)],
            cwd=micro_dir,
        )
    else:
        makefile = micro_dir / "Makefile"
        if not makefile.exists():
            raise RuntimeError(f"missing MicroGemm Makefile: {makefile}")
        run_checked(["make", "-j", str(max(1, os.cpu_count() or 1))], cwd=micro_dir)
    if not all(path.exists() for path in bins):
        raise RuntimeError("MicroGemm build finished but expected binaries were not found")
    for path in bins:
        ensure_executable(path)
    return cli_bin, convert_bin, text_bin


def resolve_microgemm_model_dir(args: argparse.Namespace) -> Path:
    if args.microgemm_model_dir:
        model_dir = Path(args.microgemm_model_dir).expanduser().resolve()
        if not model_dir.is_dir():
            raise RuntimeError(f"--microgemm-model-dir is not a directory: {model_dir}")
        return model_dir

    candidate = Path(args.model).expanduser()
    if candidate.is_dir():
        return candidate.resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "MicroGemm backend needs huggingface_hub to download model snapshots. "
            "Install it or pass --microgemm-model-dir."
        ) from exc

    cache_dir = Path(args.microgemm_cache_dir or microgemm_default_cache_dir()).expanduser()
    local_dir = cache_dir / "hf" / args.model.replace("/", "__")
    path = snapshot_download(
        repo_id=args.model,
        local_dir=str(local_dir),
        local_files_only=bool(args.local_files_only),
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors",
            "*.safetensors.index.json",
        ],
    )
    return Path(path).resolve()


def ensure_microgemm_single_safetensors(model_dir: Path) -> None:
    if (model_dir / "model.safetensors").exists():
        return
    shards = sorted(model_dir.glob("*.safetensors"))
    index_files = sorted(model_dir.glob("*.safetensors.index.json"))
    details = "\n".join(f"  - {p.name}" for p in shards[:16]) or "  none"
    if index_files:
        details += "\nindex files:\n" + "\n".join(f"  - {p.name}" for p in index_files)
    raise RuntimeError(
        "MicroGemm converter currently expects a single-file checkpoint named "
        "`model.safetensors`.\n"
        f"model_dir: {model_dir}\n"
        f"found safetensors:\n{details}\n"
        "Use a smaller single-file Qwen2/Qwen2.5 checkpoint or add sharded "
        "safetensors ingestion to microgemm-convert first."
    )


def ensure_microgemm_mgm(
    args: argparse.Namespace,
    convert_bin: Path,
    model_dir: Path,
) -> Path:
    if args.microgemm_mgm:
        mgm_path = Path(args.microgemm_mgm).expanduser().resolve()
        if not mgm_path.exists():
            raise RuntimeError(f"--microgemm-mgm does not exist: {mgm_path}")
        return mgm_path

    cache_dir = Path(args.microgemm_cache_dir or microgemm_default_cache_dir()).expanduser()
    slug = args.model.replace("/", "__") if not args.microgemm_model_dir else model_dir.name
    mgm_path = (cache_dir / "mgm" / slug / "model.mgm").resolve()
    if args.microgemm_force_convert and mgm_path.exists():
        mgm_path.unlink()
    if mgm_path.exists():
        return mgm_path
    mgm_path.parent.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            str(convert_bin),
            "from-dir",
            str(model_dir),
            str(mgm_path),
            "--kv-block-size",
            str(args.microgemm_kv_block_size),
        ],
        cwd=convert_bin.parent,
    )
    return mgm_path


def parse_microgemm_stdout(stdout: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    text_key: str | None = None
    text_lines: list[str] = []
    for line in stdout.splitlines():
        if line in {"generated_text:", "full_text:"}:
            if text_key is not None:
                out[text_key] = "\n".join(text_lines).strip()
            text_key = line[:-1]
            text_lines = []
            continue
        if text_key is not None:
            if ": " in line and not line.startswith(" "):
                out[text_key] = "\n".join(text_lines).strip()
                text_key = None
                text_lines = []
            else:
                text_lines.append(line)
                continue
        if ": " in line:
            key, value = line.split(": ", 1)
            out[key.strip()] = value.strip()
    if text_key is not None:
        out[text_key] = "\n".join(text_lines).strip()

    numeric_keys = {
        "prompt_token_count",
        "generated_token_count",
        "batch_size",
        "finished_request_count",
        "scheduler_iterations",
        "scheduler_outer_threads",
        "scheduler_inner_threads",
        "scheduler_lm_head_threads",
        "native_continuous_batching",
        "batched_decode",
        "batched_decode_calls",
        "batched_decode_tokens",
        "batched_lm_head",
        "batched_lm_head_calls",
        "batched_lm_head_tokens",
        "setup_ms",
        "prefill_ms",
        "decode_ms",
        "batch_profile_calls",
        "batch_profile_tokens",
        "batch_profile_emit_version",
        "batch_profile_total_ms",
        "batch_profile_alloc_ms",
        "batch_profile_embed_ms",
        "batch_profile_input_norm_ms",
        "batch_profile_qkv_ms",
        "batch_profile_rope_kv_ms",
        "batch_profile_attention_ms",
        "batch_profile_o_proj_ms",
        "batch_profile_post_norm_ms",
        "batch_profile_gate_up_ms",
        "batch_profile_gate_up_quant_ms",
        "batch_profile_gate_up_dot_ms",
        "batch_profile_activation_ms",
        "batch_profile_down_proj_ms",
        "batch_profile_final_norm_ms",
        "batch_profile_lm_head_ms",
        "batch_profile_copy_ms",
        "batch_profile_cleanup_ms",
        "total_ms",
        "loaded_model_bytes",
        "workspace_bytes",
        "kv_cache_bytes",
        "runtime_total_bytes",
        "prefill_tps",
        "decode_tps",
        "total_tps",
    }
    for key in list(out):
        if key not in numeric_keys:
            continue
        try:
            if key.endswith("_count") or key.endswith("_bytes") or key in {
                "native_continuous_batching",
                "batched_decode",
                "batched_decode_calls",
                "batched_decode_tokens",
                "batched_lm_head",
                "batched_lm_head_calls",
                "batched_lm_head_tokens",
                "batch_profile_calls",
                "batch_profile_tokens",
                "batch_profile_emit_version",
                "scheduler_outer_threads",
                "scheduler_inner_threads",
                "scheduler_lm_head_threads",
            }:
                out[key] = int(float(out[key]))
            else:
                out[key] = float(out[key])
        except (TypeError, ValueError):
            pass
    return out


def run_microgemm_text_once(
    *,
    text_bin: Path,
    mgm_path: Path,
    tokenizer_json: Path,
    prompt: str,
    max_new_tokens: int,
    max_seq_len: int,
    threads: int,
    seed: int,
    ignore_eos: bool,
) -> dict[str, Any]:
    prompt_file = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            suffix=".txt",
            delete=False,
        ) as fh:
            fh.write(prompt)
            prompt_file = Path(fh.name)

        cmd = [
            str(text_bin),
            "generate",
            str(mgm_path),
            str(tokenizer_json),
            "--prompt-file",
            str(prompt_file),
            "--max-new-tokens",
            str(max_new_tokens),
            "--temperature",
            "0.0",
            "--top-k",
            "0",
            "--top-p",
            "1.0",
            "--seed",
            str(seed),
        ]
        if max_seq_len > 0:
            cmd.extend(["--max-seq-len", str(max_seq_len)])
        if ignore_eos:
            cmd.append("--ignore-eos")

        env = os.environ.copy()
        if threads > 0:
            env["OMP_NUM_THREADS"] = str(threads)
            env.setdefault("OMP_PROC_BIND", "false")

        start = time.perf_counter()
        result = run_capture(cmd, env=env)
        wall_ms = (time.perf_counter() - start) * 1000.0
        row: dict[str, Any] = {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "wall_ms": wall_ms,
        }
        if result.returncode != 0:
            row["stdout"] = result.stdout[-1000:]
            row["stderr"] = result.stderr[-2000:]
            return row
        row.update(parse_microgemm_stdout(result.stdout))
        return row
    finally:
        if prompt_file is not None:
            try:
                prompt_file.unlink()
            except OSError:
                pass


def run_microgemm_batch_once(
    *,
    text_bin: Path,
    mgm_path: Path,
    tokenizer_json: Path,
    prompts: list[str],
    max_new_tokens: int,
    max_seq_len: int,
    threads: int,
    seed: int,
    ignore_eos: bool,
) -> dict[str, Any]:
    prompt_files: list[Path] = []
    try:
        for prompt in prompts:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                suffix=".txt",
                delete=False,
            ) as fh:
                fh.write(prompt)
                prompt_files.append(Path(fh.name))

        cmd = [
            str(text_bin),
            "batch-generate",
            str(mgm_path),
            str(tokenizer_json),
            "--max-new-tokens",
            str(max_new_tokens),
            "--temperature",
            "0.0",
            "--top-k",
            "0",
            "--top-p",
            "1.0",
            "--seed",
            str(seed),
        ]
        for prompt_file in prompt_files:
            cmd.extend(["--prompt-file", str(prompt_file)])
        if max_seq_len > 0:
            cmd.extend(["--max-seq-len", str(max_seq_len)])
        if ignore_eos:
            cmd.append("--ignore-eos")

        env = os.environ.copy()
        if threads > 0:
            if len(prompts) > 1:
                outer_threads = min(len(prompts), threads)
                inner_threads = max(1, threads // outer_threads)
                env["MICROGEMM_BATCH_TOTAL_THREADS"] = str(threads)
                env["MICROGEMM_BATCH_OUTER_THREADS"] = str(outer_threads)
                env["MICROGEMM_BATCH_INNER_THREADS"] = str(inner_threads)
                env["MICROGEMM_BATCH_LM_HEAD_THREADS"] = str(threads)
                env["OMP_NUM_THREADS"] = str(threads)
                env["OMP_MAX_ACTIVE_LEVELS"] = "1"
                env["OMP_NESTED"] = "false"
            else:
                env["OMP_NUM_THREADS"] = str(threads)
            env.setdefault("OMP_PROC_BIND", "false")

        start = time.perf_counter()
        result = run_capture(cmd, env=env)
        wall_ms = (time.perf_counter() - start) * 1000.0
        row: dict[str, Any] = {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "wall_ms": wall_ms,
        }
        if result.returncode != 0:
            row["stdout"] = result.stdout[-1000:]
            row["stderr"] = result.stderr[-2000:]
            return row
        row.update(parse_microgemm_stdout(result.stdout))
        return row
    finally:
        for prompt_file in prompt_files:
            try:
                prompt_file.unlink()
            except OSError:
                pass


def load_microgemm_runner(args: argparse.Namespace, tokenizer):
    if args.device != "cpu":
        raise ValueError("MicroGemm backend expects --device cpu")

    cli_bin, convert_bin, text_bin = ensure_microgemm_binaries(args)
    model_dir = resolve_microgemm_model_dir(args)
    ensure_microgemm_single_safetensors(model_dir)
    mgm_path = ensure_microgemm_mgm(args, convert_bin, model_dir)
    tokenizer_json = model_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        raise RuntimeError(f"MicroGemm tokenizer.json not found: {tokenizer_json}")

    try:
        caps = run_checked([str(cli_bin), "capabilities"], cwd=cli_bin.parent).stdout.strip()
    except Exception as exc:
        caps = f"capabilities unavailable: {type(exc).__name__}: {exc}"

    print("  MicroGemm backend:")
    print(f"    text:       {text_bin}")
    print(f"    model_dir:  {model_dir}")
    print(f"    mgm:        {mgm_path}")
    print(f"    mode:       {args.microgemm_batch_mode}")

    call_counter = 0

    def run(prompts: list[str], max_new_tokens: int) -> dict[str, Any]:
        nonlocal call_counter
        call_counter += 1
        requested_mode = args.microgemm_batch_mode
        effective_mode = requested_mode
        if requested_mode == "adaptive":
            if len(prompts) == 1:
                effective_mode = "continuous"
            elif len(prompts) <= 4:
                effective_mode = "concurrent"
            else:
                effective_mode = "continuous"

        threads_total = int(args.microgemm_threads or (os.cpu_count() or 1))
        if effective_mode == "continuous":
            threads_per_worker = max(1, threads_total)
        elif args.microgemm_threads_per_worker > 0:
            threads_per_worker = int(args.microgemm_threads_per_worker)
        elif effective_mode == "concurrent":
            threads_per_worker = max(1, threads_total // max(1, len(prompts)))
        else:
            threads_per_worker = max(1, threads_total)

        def run_one(idx: int, prompt: str) -> dict[str, Any]:
            return run_microgemm_text_once(
                text_bin=text_bin,
                mgm_path=mgm_path,
                tokenizer_json=tokenizer_json,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                max_seq_len=args.max_seq_len,
                threads=threads_per_worker,
                seed=int(args.microgemm_seed) + call_counter * 1000 + idx,
                ignore_eos=bool(args.ignore_eos),
            )

        continuous_single_fast_path = effective_mode == "continuous" and len(prompts) == 1
        start = time.perf_counter()
        if effective_mode == "continuous":
            if continuous_single_fast_path:
                worker_rows = [run_one(0, prompts[0])]
            else:
                worker_rows = [
                    run_microgemm_batch_once(
                        text_bin=text_bin,
                        mgm_path=mgm_path,
                        tokenizer_json=tokenizer_json,
                        prompts=prompts,
                        max_new_tokens=max_new_tokens,
                        max_seq_len=args.max_seq_len,
                        threads=threads_per_worker,
                        seed=int(args.microgemm_seed) + call_counter * 1000,
                        ignore_eos=bool(args.ignore_eos),
                    )
                ]
            runtime_setup_ms = float(worker_rows[0].get("setup_ms", 0.0) or 0.0)
            runtime_total_ms = float(worker_rows[0].get("total_ms", 0.0) or 0.0)
            runtime_prefill_ms = float(worker_rows[0].get("prefill_ms", 0.0) or 0.0)
            runtime_decode_ms = float(worker_rows[0].get("decode_ms", 0.0) or 0.0)
        elif effective_mode == "serial" or len(prompts) <= 1:
            worker_rows = [run_one(idx, prompt) for idx, prompt in enumerate(prompts)]
            runtime_setup_ms = 0.0
            runtime_total_ms = sum(float(row.get("total_ms", 0.0) or 0.0) for row in worker_rows)
            runtime_prefill_ms = sum(float(row.get("prefill_ms", 0.0) or 0.0) for row in worker_rows)
            runtime_decode_ms = sum(float(row.get("decode_ms", 0.0) or 0.0) for row in worker_rows)
        else:
            worker_rows = []
            with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
                futures = {
                    pool.submit(run_one, idx, prompt): idx
                    for idx, prompt in enumerate(prompts)
                }
                for future in as_completed(futures):
                    worker_rows.append(future.result())
            runtime_setup_ms = 0.0
            runtime_total_ms = max([float(row.get("total_ms", 0.0) or 0.0) for row in worker_rows] or [0.0])
            runtime_prefill_ms = max([float(row.get("prefill_ms", 0.0) or 0.0) for row in worker_rows] or [0.0])
            runtime_decode_ms = max([float(row.get("decode_ms", 0.0) or 0.0) for row in worker_rows] or [0.0])

        wall_elapsed_s = time.perf_counter() - start
        failed = [row for row in worker_rows if not row.get("ok")]
        if failed:
            first = failed[0]
            raise RuntimeError(
                "MicroGemm worker failed "
                f"returncode={first.get('returncode')} "
                f"stderr={str(first.get('stderr') or '')[-1000:]}"
            )

        batched_decode_calls = sum(
            int(row.get("batched_decode_calls", 0) or 0) for row in worker_rows
        )
        batched_decode_tokens = sum(
            int(row.get("batched_decode_tokens", 0) or 0) for row in worker_rows
        )
        batched_lm_head_calls = sum(
            int(row.get("batched_lm_head_calls", 0) or 0) for row in worker_rows
        )
        batched_lm_head_tokens = sum(
            int(row.get("batched_lm_head_tokens", 0) or 0) for row in worker_rows
        )
        scheduler_outer_threads = int(worker_rows[0].get("scheduler_outer_threads", 0) or 0) if worker_rows else 0
        scheduler_inner_threads = int(worker_rows[0].get("scheduler_inner_threads", 0) or 0) if worker_rows else 0
        scheduler_lm_head_threads = int(worker_rows[0].get("scheduler_lm_head_threads", 0) or 0) if worker_rows else 0
        generated_tokens = sum(int(row.get("generated_token_count", 0) or 0) for row in worker_rows)
        if generated_tokens <= 0 and args.ignore_eos:
            generated_tokens = len(prompts) * max_new_tokens

        runtime_elapsed_s = runtime_total_ms / 1000.0 if runtime_total_ms > 0 else wall_elapsed_s
        decode_s = runtime_decode_ms / 1000.0
        scheduler_like_stats = {
            "kind": (
                "microgemm_continuous_batch"
                if effective_mode == "continuous"
                else "microgemm_workers"
            ),
            "total_tokens": generated_tokens,
            "setup_time_ms": runtime_setup_ms,
            "prefill_time_ms": runtime_prefill_ms,
            "decode_time_ms": runtime_decode_ms,
            "decode_throughput": generated_tokens / decode_s if decode_s > 0 else 0.0,
        }
        return {
            "elapsed_s": runtime_elapsed_s,
            "generated_tokens": generated_tokens,
            "extra": {
                "scheduler_stats": scheduler_like_stats,
                "microgemm": {
                    "mode": requested_mode,
                    "effective_mode": effective_mode,
                    "batch_semantics": (
                        "continuous_parallel_active_set"
                        if effective_mode == "continuous"
                        else "concurrent_workers"
                        if effective_mode == "concurrent"
                        else "serial_workers"
                    ),
                    "native_continuous_batching": effective_mode == "continuous"
                    and not continuous_single_fast_path,
                    "continuous_single_fast_path": continuous_single_fast_path,
                    "threads_total": threads_total,
                    "threads_per_worker": threads_per_worker,
                    "scheduler_outer_threads": scheduler_outer_threads,
                    "scheduler_inner_threads": scheduler_inner_threads,
                    "scheduler_lm_head_threads": scheduler_lm_head_threads,
                    "model_dir": str(model_dir),
                    "mgm_path": str(mgm_path),
                    "runtime_setup_ms": runtime_setup_ms,
                    "runtime_total_ms": runtime_total_ms,
                    "wall_elapsed_s": wall_elapsed_s,
                    "batched_decode_calls": batched_decode_calls,
                    "batched_decode_tokens": batched_decode_tokens,
                    "batched_lm_head_calls": batched_lm_head_calls,
                    "batched_lm_head_tokens": batched_lm_head_tokens,
                    "wall_output_tps": (
                        generated_tokens / wall_elapsed_s if wall_elapsed_s > 0 else 0.0
                    ),
                    "worker_rows": worker_rows,
                    "capabilities": caps,
                },
            },
        }

    return run


def vllm_dtype(name: str) -> str:
    if name == "fp16":
        return "float16"
    if name == "bf16":
        return "bfloat16"
    if name == "fp32":
        return "float32"
    return name


def load_vllm_runner(args: argparse.Namespace, tokenizer):
    if args.device != "cuda":
        raise ValueError("vLLM backend in this benchmark expects --device cuda")
    if args.quantize:
        raise ValueError("Use vLLM-native quantized checkpoints for vLLM; --quantize is MegaGemm-only")
    if args.vllm_disable_cudagraph_memory_profiler:
        os.environ.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0")

    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise RuntimeError(
            "vLLM is not installed or failed to import. "
            f"Underlying import error: {type(exc).__name__}: {exc}. "
            "Install a vLLM wheel whose CUDA/PyTorch ABI matches the isolated "
            "benchmark environment."
        ) from exc

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": vllm_dtype(args.dtype),
        "trust_remote_code": True,
        "tensor_parallel_size": args.vllm_tensor_parallel_size,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "max_model_len": args.vllm_max_model_len or args.max_seq_len,
        "enforce_eager": args.vllm_enforce_eager,
        "enable_prefix_caching": not args.vllm_disable_prefix_caching,
        "disable_log_stats": True,
    }
    if args.vllm_max_num_seqs > 0:
        llm_kwargs["max_num_seqs"] = args.vllm_max_num_seqs
    if args.vllm_max_num_batched_tokens > 0:
        llm_kwargs["max_num_batched_tokens"] = args.vllm_max_num_batched_tokens
    if args.vllm_language_model_only:
        llm_kwargs["language_model_only"] = True
    if args.tokenizer:
        llm_kwargs["tokenizer"] = args.tokenizer
    if args.cache_dir:
        llm_kwargs["download_dir"] = args.cache_dir

    try:
        llm = LLM(**llm_kwargs)
    except TypeError:
        # Older vLLM wheels do not accept every modern keyword. Retry with the
        # essentials so the benchmark remains useful across Colab/Kaggle images.
        for key in (
            "disable_log_stats",
            "trust_remote_code",
            "enable_prefix_caching",
            "language_model_only",
        ):
            llm_kwargs.pop(key, None)
        llm = LLM(**llm_kwargs)

    def run(prompts: list[str], max_new_tokens: int) -> dict[str, Any]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        sampling_kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_new_tokens,
            "ignore_eos": bool(args.ignore_eos),
        }
        sampling_params = SamplingParams(**sampling_kwargs)
        sync_cuda()
        start = time.perf_counter()
        try:
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        except TypeError:
            outputs = llm.generate(prompts, sampling_params)
        sync_cuda()
        elapsed_s = time.perf_counter() - start

        generated_tokens = 0
        for output in outputs:
            candidates = getattr(output, "outputs", None) or []
            if not candidates:
                continue
            first = candidates[0]
            token_ids = getattr(first, "token_ids", None)
            if token_ids is not None:
                generated_tokens += len(token_ids)
            else:
                generated_tokens += len(tokenizer.encode(getattr(first, "text", ""), add_special_tokens=False))
        if generated_tokens <= 0 and args.ignore_eos:
            generated_tokens = len(prompts) * max_new_tokens
        return {
            "elapsed_s": elapsed_s,
            "generated_tokens": generated_tokens,
            "extra": {
                "vllm": {
                    "tensor_parallel_size": args.vllm_tensor_parallel_size,
                    "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                    "max_model_len": args.vllm_max_model_len or args.max_seq_len,
                    "max_num_seqs": args.vllm_max_num_seqs or None,
                    "max_num_batched_tokens": args.vllm_max_num_batched_tokens or None,
                    "enforce_eager": args.vllm_enforce_eager,
                    "prefix_caching": not args.vllm_disable_prefix_caching,
                    "cudagraph_memory_profiler": not args.vllm_disable_cudagraph_memory_profiler,
                }
            },
        }

    return run


def load_mesh_runner(args: argparse.Namespace, tokenizer):
    if args.backend == "mesh-replicas":
        from megagemm.mesh import ShardReplicaRouter

        router = ShardReplicaRouter(
            args.replicas,
            model_name=args.model,
            timeout=args.timeout,
            transport=args.transport,
            enable_thinking=False if args.disable_thinking else None,
            remote_chain_loop=not args.no_remote_chain_loop,
        )

        def run(prompts: list[str], max_new_tokens: int) -> dict[str, Any]:
            start = time.perf_counter()
            result = router.generate_batch(
                prompts,
                max_new_tokens=max_new_tokens,
                microbatch_size=args.microbatch_size,
                strategy=args.replica_strategy,
            )
            elapsed_s = time.perf_counter() - start
            return {
                "elapsed_s": elapsed_s,
                "generated_tokens": int(result.get("generated_tokens", 0)),
                "extra": {"mesh_result": result},
            }

        return run

    from megagemm.mesh import ShardPipeline

    pipeline = ShardPipeline(
        args.stages,
        model_name=args.model,
        timeout=args.timeout,
        transport=args.transport,
        enable_thinking=False if args.disable_thinking else None,
        remote_chain_loop=not args.no_remote_chain_loop,
    )

    def run(prompts: list[str], max_new_tokens: int) -> dict[str, Any]:
        start = time.perf_counter()
        if args.backend == "mesh-continuous":
            result = pipeline.generate_continuous(
                prompts,
                max_new_tokens=max_new_tokens,
                microbatch_size=args.microbatch_size,
                max_batch_size=args.mesh_max_batch_size,
            )
        else:
            result = pipeline.generate_batch(
                prompts,
                max_new_tokens=max_new_tokens,
                microbatch_size=args.microbatch_size,
            )
        elapsed_s = time.perf_counter() - start
        return {
            "elapsed_s": elapsed_s,
            "generated_tokens": int(result.get("generated_tokens", 0)),
            "extra": {"mesh_result": result},
        }

    return run


def make_runner(args: argparse.Namespace, tokenizer) -> Callable[[list[str], int], dict[str, Any]]:
    if args.backend == "megagemm":
        return load_megagemm_runner(args, tokenizer)
    if args.backend == "megagemm-prophet":
        return load_megagemm_prophet_runner(args, tokenizer)
    if args.backend == "hf":
        return load_hf_runner(args, tokenizer)
    if args.backend == "vllm":
        return load_vllm_runner(args, tokenizer)
    if args.backend == "microgemm":
        return load_microgemm_runner(args, tokenizer)
    return load_mesh_runner(args, tokenizer)


def scenario_name(batch_size: int, prompt_tokens: int, args: argparse.Namespace) -> str:
    if args.scenario_label:
        return args.scenario_label
    if args.kv_offload:
        return "kv_offload"
    if prompt_tokens >= 2048:
        return "long_context"
    if batch_size == 1:
        return "single"
    return "batch"


def run_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    tokenizer_source = args.tokenizer or args.model
    tokenizer = load_tokenizer(tokenizer_source, local_files_only=args.local_files_only)
    batch_sizes = parse_csv_ints(args.batch_sizes)
    prompt_token_lengths = parse_csv_ints(args.prompt_tokens)

    print("Inference matrix benchmark")
    print(f"  backend:        {args.backend}")
    print(f"  hardware:       {args.hardware_label}")
    print(f"  model:          {args.model}")
    print(f"  dtype:          {args.dtype}")
    print(f"  quantize:       {args.quantize or 'none'}")
    print(f"  batch sizes:    {batch_sizes}")
    print(f"  prompt tokens:  {prompt_token_lengths}")
    print(f"  max new tokens: {args.max_new_tokens}")
    print(f"  repeats:        {args.repeats}")
    print(f"  gpu:            {gpu_snapshot()}")

    rows: list[dict[str, Any]] = []
    try:
        runner = make_runner(args, tokenizer)
    except torch.cuda.OutOfMemoryError as exc:
        cleanup_cuda()
        print(f"  load failed CUDA OOM: {exc}")
        for prompt_tokens in prompt_token_lengths:
            for batch_size in batch_sizes:
                _, prompt_tokens_actual = build_prompts(tokenizer, batch_size, prompt_tokens)
                rows.append(
                    metric_row(
                        args=args,
                        scenario=scenario_name(batch_size, prompt_tokens, args),
                        batch_size=batch_size,
                        prompt_tokens_requested=prompt_tokens,
                        prompt_tokens_actual=prompt_tokens_actual,
                        max_new_tokens=args.max_new_tokens,
                        elapsed_s=0.0,
                        generated_tokens=0,
                        ok=False,
                        error=f"LOAD CUDA OOM: {exc}",
                    )
                )
        return rows
    except Exception as exc:
        cleanup_cuda()
        print(f"  load failed {type(exc).__name__}: {exc}")
        for prompt_tokens in prompt_token_lengths:
            for batch_size in batch_sizes:
                _, prompt_tokens_actual = build_prompts(tokenizer, batch_size, prompt_tokens)
                rows.append(
                    metric_row(
                        args=args,
                        scenario=scenario_name(batch_size, prompt_tokens, args),
                        batch_size=batch_size,
                        prompt_tokens_requested=prompt_tokens,
                        prompt_tokens_actual=prompt_tokens_actual,
                        max_new_tokens=args.max_new_tokens,
                        elapsed_s=0.0,
                        generated_tokens=0,
                        ok=False,
                        error=f"LOAD {type(exc).__name__}: {exc}",
                        extra={"traceback": traceback.format_exc(limit=3)},
                    )
                )
        return rows

    for prompt_tokens in prompt_token_lengths:
        for batch_size in batch_sizes:
            scenario = scenario_name(batch_size, prompt_tokens, args)
            prompts, prompt_tokens_actual = build_prompts(tokenizer, batch_size, prompt_tokens)
            reset_scenario = getattr(runner, "reset_scenario", None)
            if callable(reset_scenario):
                reset_scenario()
            if args.warmup > 0:
                for warmup_idx in range(args.warmup):
                    cleanup_cuda()
                    print(
                        f"Warmup scenario={scenario} batch={batch_size} prompt_tokens={prompt_tokens} "
                        f"repeat={warmup_idx + 1}/{args.warmup}"
                    )
                    try:
                        runner(prompts, args.max_new_tokens)
                    except Exception as exc:
                        print(f"  warmup failed: {type(exc).__name__}: {exc}")
                        break
                cleanup_cuda()
            for repeat_idx in range(args.repeats):
                cleanup_cuda()
                print(
                    f"Run scenario={scenario} batch={batch_size} prompt_tokens={prompt_tokens} "
                    f"repeat={repeat_idx + 1}/{args.repeats}"
                )
                try:
                    result = runner(prompts, args.max_new_tokens)
                    row_extra = dict(result.get("extra") or {})
                    row_extra["repeat_index"] = repeat_idx + 1
                    row_extra["repeat_count"] = args.repeats
                    row = metric_row(
                        args=args,
                        scenario=scenario,
                        batch_size=batch_size,
                        prompt_tokens_requested=prompt_tokens,
                        prompt_tokens_actual=prompt_tokens_actual,
                        max_new_tokens=args.max_new_tokens,
                        elapsed_s=float(result["elapsed_s"]),
                        generated_tokens=int(result["generated_tokens"]),
                        ok=True,
                        extra=row_extra,
                    )
                    suffix_parts: list[str] = []
                    scheduler_stats = row.get("scheduler_stats")
                    if isinstance(scheduler_stats, dict):
                        prefill_ms = float(scheduler_stats.get("prefill_time_ms") or 0.0)
                        decode_ms = float(scheduler_stats.get("decode_time_ms") or 0.0)
                        if prefill_ms > 0.0:
                            prefill_s = prefill_ms / 1000.0
                            prompt_total = float(row.get("prompt_tokens_actual_total") or 0.0)
                            prefill_tps = prompt_total / prefill_s if prefill_s > 0.0 else 0.0
                            suffix_parts.append(
                                f"prefill={prefill_ms:.1f}ms/{prefill_tps:.0f}tok/s"
                            )
                        if decode_ms > 0.0:
                            decode_s = decode_ms / 1000.0
                            decode_tps = (
                                float(row["generated_tokens"]) / decode_s
                                if decode_s > 0.0
                                else 0.0
                            )
                            suffix_parts.append(
                                f"decode={decode_ms:.1f}ms/{decode_tps:.2f}tok/s"
                            )
                        prefill_stage = scheduler_stats.get("prefill_stage_timing")
                        if isinstance(prefill_stage, dict):
                            labels = [
                                ("qkv_ms", "qkv"),
                                ("attn_core_ms", "attn"),
                                ("o_proj_ms", "o"),
                                ("gate_up_ms", "gate_up"),
                                ("down_proj_ms", "down"),
                                ("mlp_native_ms", "mlp_native"),
                                ("kv_write_ms", "kv"),
                            ]
                            stage_parts = [
                                f"{label}={float(prefill_stage[key]):.0f}ms"
                                for key, label in labels
                                if key in prefill_stage
                            ]
                            if stage_parts:
                                suffix_parts.append("prefill_parts(" + ",".join(stage_parts) + ")")
                    if row.get("prophet_decode_s"):
                        prophet_decode_s = float(row.get("prophet_decode_s") or 0.0)
                        prophet_restore_ms = float(row.get("prophet_restore_s") or 0.0) * 1000.0
                        if prophet_decode_s > 0.0:
                            decoded_tokens = max(
                                0.0,
                                float(row["generated_tokens"])
                                - float(row.get("prophet_bootstrap_tokens") or 0.0),
                            )
                            prophet_decode_tps = decoded_tokens / prophet_decode_s
                            suffix_parts.append(
                                f"prophet_decode_tps={prophet_decode_tps:.2f}"
                                f" restore={prophet_restore_ms:.1f}ms"
                            )
                    suffix = f" {' '.join(suffix_parts)}" if suffix_parts else ""
                    print(
                        f"  ok output_tps={row['output_tps']:.2f} "
                        f"elapsed={row['elapsed_s']:.3f}s generated={row['generated_tokens']}"
                        f"{suffix}"
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    cleanup_cuda()
                    row = metric_row(
                        args=args,
                        scenario=scenario,
                        batch_size=batch_size,
                        prompt_tokens_requested=prompt_tokens,
                        prompt_tokens_actual=prompt_tokens_actual,
                        max_new_tokens=args.max_new_tokens,
                        elapsed_s=0.0,
                        generated_tokens=0,
                        ok=False,
                        error=f"CUDA OOM: {exc}",
                    )
                    print(f"  failed CUDA OOM: {exc}")
                except Exception as exc:
                    cleanup_cuda()
                    row = metric_row(
                        args=args,
                        scenario=scenario,
                        batch_size=batch_size,
                        prompt_tokens_requested=prompt_tokens,
                        prompt_tokens_actual=prompt_tokens_actual,
                        max_new_tokens=args.max_new_tokens,
                        elapsed_s=0.0,
                        generated_tokens=0,
                        ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                        extra={"traceback": traceback.format_exc(limit=3)},
                    )
                    print(f"  failed {type(exc).__name__}: {exc}")
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="General inference TPS matrix benchmark")
    parser.add_argument(
        "--backend",
        required=True,
        choices=[
            "megagemm",
            "megagemm-prophet",
            "hf",
            "vllm",
            "microgemm",
            "mesh-shard",
            "mesh-continuous",
            "mesh-replicas",
        ],
    )
    parser.add_argument("--model", required=True, help="HF model id, local snapshot, or MGX path")
    parser.add_argument("--tokenizer", help="Tokenizer path/id when different from --model")
    parser.add_argument("--hardware-label", required=True, help="Example: 1xt4, 2xt4, 1xl4, 4xl4, 1xa100")
    parser.add_argument("--scenario-label", default="", help="Optional scenario name override")
    parser.add_argument("--batch-sizes", default="1,8,16")
    parser.add_argument("--prompt-tokens", default="128")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--ignore-eos", action="store_true", help="Benchmark fixed decode length by not stopping on EOS")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out-dir", default="bench_results")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print extra diagnostic messages")

    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--quantize", choices=["int8", "fp8", "awq"])
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-alloc", default="auto", choices=["auto", "greedy"])
    parser.add_argument("--kv-offload", action="store_true")
    parser.add_argument("--num-cpu-blocks", type=int, default=0)
    parser.add_argument("--gpu-window", type=int, default=64)
    parser.add_argument("--mgx-prefer-payload-cache", action="store_true")
    parser.add_argument("--mgx-payload-cache-dir")
    parser.add_argument("--prophet-dir", default="", help="Prophet library directory for megagemm-prophet")
    parser.add_argument("--prophet-reset-dir", action="store_true")
    parser.add_argument(
        "--prophet-prime-before-measure",
        action="store_true",
        help="Prime Prophet snapshots before timed rows so every measured repeat is cache-hit.",
    )
    parser.add_argument("--prophet-validation-mode", choices=["none", "full_prefill"], default="none")
    parser.add_argument("--prophet-validation-tokens", type=int, default=4)
    parser.add_argument("--prophet-agreement-threshold", type=float, default=1.0)
    parser.add_argument("--prophet-fallback-to-prefill", action="store_true")
    parser.add_argument("--prophet-top-k", type=int, default=3)
    parser.add_argument("--prophet-min-similarity", type=float, default=0.35)
    parser.add_argument("--prophet-prefix-tokens", type=int, default=64)
    parser.add_argument("--prophet-min-prefix-reuse-score", type=float, default=0.55)
    parser.add_argument("--prophet-min-prefix-coverage", type=float, default=0.50)
    parser.add_argument("--prophet-max-prefix-rollback-ratio", type=float, default=0.35)
    parser.add_argument("--prophet-max-prefix-tail-ratio", type=float, default=0.50)
    parser.add_argument(
        "--prophet-batch-exact-restore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Batch exact Prophet snapshot restores into one layer-wise KV write pass.",
    )
    parser.add_argument(
        "--prophet-live-prefix-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep primed Prophet prefix contexts live and fork them with shared "
            "KV blocks for prefix-cache-style repeated-prompt hits."
        ),
    )
    parser.add_argument(
        "--prophet-resident-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep restored Prophet snapshots resident in the engine and fork "
            "future exact/prefix hits instead of re-restoring KV from disk."
        ),
    )
    parser.add_argument(
        "--prophet-resident-cache-max-entries",
        type=int,
        default=16,
        help="Maximum live Prophet resident source sequences per engine.",
    )

    parser.add_argument("--hf-mode", default="batched", choices=["batched", "sequential"])
    parser.add_argument("--hf-device-map", default="", help="Example: auto. Empty means .to(--device).")

    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-max-model-len", type=int, default=0, help="0 means use --max-seq-len")
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=0,
        help="Maximum scheduler sequences; 0 keeps the vLLM default.",
    )
    parser.add_argument(
        "--vllm-max-num-batched-tokens",
        type=int,
        default=0,
        help="Maximum scheduler tokens per iteration; 0 keeps the vLLM default.",
    )
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--vllm-language-model-only", action="store_true")
    parser.add_argument("--vllm-disable-prefix-caching", action="store_true")
    parser.add_argument("--vllm-disable-cudagraph-memory-profiler", action="store_true")

    parser.add_argument(
        "--microgemm-model-dir",
        default="",
        help="Local HF snapshot directory for MicroGemm conversion. Defaults to downloading --model.",
    )
    parser.add_argument(
        "--microgemm-cache-dir",
        default="",
        help="Cache root for MicroGemm HF snapshots and .mgm files.",
    )
    parser.add_argument("--microgemm-mgm", default="", help="Existing .mgm file to run directly.")
    parser.add_argument("--microgemm-force-convert", action="store_true")
    parser.add_argument("--microgemm-no-build", action="store_true")
    parser.add_argument("--microgemm-kv-block-size", type=int, default=16)
    parser.add_argument("--microgemm-threads", type=int, default=0)
    parser.add_argument("--microgemm-threads-per-worker", type=int, default=0)
    parser.add_argument("--microgemm-seed", type=int, default=42)
    parser.add_argument(
        "--microgemm-batch-mode",
        choices=["adaptive", "continuous", "concurrent", "serial"],
        default="adaptive",
        help=(
            "MicroGemm CPU batch mode. adaptive routes low CPU batch sizes to the "
            "fastest known policy, continuous uses one persistent microgemm-text "
            "batch-generate process with per-request KV state, concurrent uses one "
            "worker process per prompt, and serial runs workers one after another."
        ),
    )

    parser.add_argument("--stages", default="", help="Ordered MegaMesh shard stages for mesh-shard/mesh-continuous")
    parser.add_argument("--replicas", default="", help="Semicolon-separated shard replicas for mesh-replicas")
    parser.add_argument("--transport", default="ttp", choices=["ttp", "binary", "json"])
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--mesh-max-batch-size", type=int, default=8)
    parser.add_argument("--replica-strategy", default="round_robin", choices=["round_robin", "chunk"])
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--no-remote-chain-loop", action="store_true")

    args = parser.parse_args()
    if args.backend in {"mesh-shard", "mesh-continuous"} and not args.stages.strip():
        raise SystemExit("--stages is required for mesh-shard/mesh-continuous")
    if args.backend == "mesh-replicas" and not args.replicas.strip():
        raise SystemExit("--replicas is required for mesh-replicas")
    if args.backend == "microgemm":
        if args.device != "cpu":
            raise SystemExit("--backend microgemm requires --device cpu")
        if args.quantize is None:
            args.quantize = "int8"
    if args.backend == "megagemm" and args.device == "cpu":
        print(
            "Warning: MegaGemm CPU is kept as a compatibility/debug fallback. "
            "Use --backend microgemm for CPU throughput benchmarks.",
            flush=True,
        )

    rows = run_matrix(args)
    raw_path, summary_path, csv_path = write_outputs(args, rows)
    print("\nWrote:")
    print(f"  raw:     {raw_path}")
    print(f"  summary: {summary_path}")
    print(f"  csv:     {csv_path}")
    print("\nSummary:")
    for row in summarize(rows):
        prefix = (
            f"  {row['backend']} {row['hardware_label']} {row['scenario']} "
            f"batch={row['batch_size']} prompt={row['prompt_tokens_requested_per_request']} "
        )
        if int(row.get("steady_ok_samples") or 0) > 0:
            print(
                prefix
                + f"first={row['first_output_tps']:.2f} tok/s "
                + f"cached_median={row['median_steady_output_tps']:.2f} tok/s "
                + f"overall_median={row['median_output_tps']:.2f} tok/s "
                + f"ok={row['ok_samples']}/{row['samples']}"
            )
        else:
            print(
                prefix
                + f"median={row['median_output_tps']:.2f} tok/s "
                + f"ok={row['ok_samples']}/{row['samples']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
