from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "run_gemma4_e2b_b4_prefill_policy_gate.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_e2b_b4_prefill_policy_colab.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("gemma4_e2b_b4_prefill", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_b4_prefill_experiment_is_separate_from_promoted_b8_policy():
    paged = (ROOT / "megagemm" / "kernels" / "paged_attention.py").read_text(
        encoding="utf-8"
    )
    model = (ROOT / "megagemm" / "models" / "llama.py").read_text(
        encoding="utf-8"
    )
    flag = "MEGAGEMM_GEMMA4_E2B_L4_B4_PREFILL_EXPERIMENT"
    assert flag in paged
    assert flag in model
    assert 'if batch_size == 4' in paged
    assert '(batch_size != 8 and not b4_experimental)' in paged
    assert 'tuple(k.shape) != (batch_size, 1, seq_len, 256)' in paged
    assert 'else "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_"' in paged
    assert "e2b_l4_full_batch == 8" in model
    assert "e2b_l4_full_batch == 4" in model


def test_policy_gate_covers_attribution_and_distinct_b4_tiles():
    module = _load_module()
    by_name = {case["name"]: case for case in module.CASES}
    assert by_name["baseline"] == {
        "name": "baseline",
        "full": False,
        "sliding": False,
        "config": None,
    }
    assert by_name["full_h512_only"]["full"] is True
    assert by_name["full_h512_only"]["sliding"] is False
    assert by_name["sliding_h256_only_b8_geometry"]["full"] is False
    combined = [case for case in module.CASES if case["name"].startswith("combined_")]
    assert len(combined) == 5
    assert len({tuple(case["config"]) for case in combined}) == 5
    assert all(case["full"] and case["sliding"] for case in combined)


def test_summary_selects_only_correct_stable_combined_full_model_case():
    module = _load_module()
    samples = []
    for repeat in (1, 2, 3):
        for case in module.CASES:
            combined = case["name"].startswith("combined_")
            prefill = 2000.0
            if case["name"] == "combined_g2_bm16_bn64_w4_s2":
                prefill = 1500.0
            elif combined:
                prefill = 1700.0
            samples.append(
                {
                    "case": case["name"],
                    "prefill_ms": prefill,
                    "wall_ms": prefill + 10.0,
                    "full_hits_delta": 7 if case["full"] else 0,
                    "sliding_hits_delta": 28 if case["sliding"] else 0,
                    "full_error": "",
                    "sliding_error": "",
                    "token_digest": "same",
                }
            )
    result = module.summarize(
        samples,
        minimum_speedup=1.05,
        maximum_spread=1.08,
    )
    assert result["decision"] == "IMPLEMENT_B4_POLICY_AND_RUN_MACRO_GATE"
    assert result["winner"] == "combined_g2_bm16_bn64_w4_s2"
    assert result["winner_config"] == (2, 16, 64, 4, 2)
    assert result["speedup"] == 2000.0 / 1500.0


def test_colab_harness_is_single_load_drive_scoped_and_non_mutating():
    source = HARNESS.read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in source
    assert "git pull" not in source
    assert "pip install" not in source
    assert "run_gemma4_e2b_b4_prefill_policy_gate.py" in source
    gate = SCRIPT.read_text(encoding="utf-8")
    assert gate.count("InferenceEngine(") == 1
    assert 'print("  model loads: 1")' in gate
    assert "MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID" not in gate
    assert "_GEMMA4_E2B_L4_SLIDING_PREFILL_FAILURE" in gate
