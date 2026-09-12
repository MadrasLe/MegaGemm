from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "run_gemma4_e2b_prefill_stage_profile.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_e2b_prefill_stage_profile_colab.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("gemma4_e2b_prefill_profile", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample(repeat: int, offset: float = 0.0):
    return {
        "repeat": repeat,
        "wall_ms": 2800.0 + offset,
        "internal_prefill_ms": 2700.0 + offset,
        "internal_decode_ms": 20.0,
        "prefill_stage_total_tokens": 16456,
        "prefill_stage_timing": {
            "qkv_ms": 300.0 + offset,
            "qkv_sliding_ms": 200.0 + offset,
            "qkv_full_ms": 100.0,
            "attn_core_ms": 900.0,
            "attn_core_sliding_ms": 500.0,
            "attn_core_full_ms": 400.0,
            "gate_up_ms": 600.0,
            "down_proj_ms": 400.0,
            "gemma4_norms_ms": 100.0,
            "total_ms": 2000.0 + offset,
        },
        "prefill_routes": {
            "sliding_enabled_layers": 28,
            "sliding_hits": 28 * repeat,
            "full_expand_enabled_layers": 7,
            "full_expand_hits": 7 * repeat,
            "full_expand_error": "",
            "implicit_causal_batches": repeat,
            "vectorized_kv_hits": 15 * repeat,
        },
    }


def test_summary_excludes_derived_attention_aggregates_from_ranking_and_groups():
    module = _load_module()
    summary = module.summarize_samples(
        [_sample(1, -10.0), _sample(2), _sample(3, 10.0)]
    )
    stages = [row["stage"] for row in summary["stage_ranking"]]
    assert "qkv" not in stages
    assert "attn_core" not in stages
    assert summary["next_target"] == "gate_up"
    assert summary["tracked_cuda_ms_median"] == pytest.approx(2000.0)
    assert summary["unattributed_internal_prefill_ms"] == pytest.approx(700.0)
    groups = {row["group"]: row for row in summary["groups"]}
    assert groups["attention"]["median_ms"] == pytest.approx(1200.0)
    assert groups["mlp"]["median_ms"] == pytest.approx(1000.0)
    assert summary["route_audit"]["passed"] is True


def test_route_audit_rejects_stale_full_attention_path():
    module = _load_module()
    samples = [_sample(1), _sample(2), _sample(3)]
    samples[1]["prefill_routes"]["full_expand_enabled_layers"] = 0
    audit = module.audit_production_prefill_routes(samples)
    assert audit["passed"] is False
    assert any("full_expand_enabled_layers" in error for error in audit["errors"])


def test_colab_harness_uses_drive_directly_without_vllm_or_git_sync():
    source = HARNESS.read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in source
    assert "git pull" not in source
    assert "pip install" in source
    assert "import vllm" not in source
    assert '"vllm' not in source
    assert "--batch-size 8" in source
    assert "--prompt-tokens 2048" in source
    assert "profile.json" in source
    assert "zipfile" not in source


def test_gemma4_prefill_timing_source_splits_attention_topologies():
    source = (ROOT / "megagemm" / "models" / "llama.py").read_text(
        encoding="utf-8"
    )
    assert 'f"attn_core_{prefill_timing_suffix}"' in source
    assert 'for topology in ("sliding", "full")' in source
    assert "if name not in detail_only:" in source
    assert '_timing_record_end(timing_events, "ple", ple_start_end)' in source
    assert '_timing_record_end(timing_events, "lm_head", lm_head_start_end)' in source
