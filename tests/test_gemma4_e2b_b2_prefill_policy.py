from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "run_gemma4_e2b_b2_prefill_policy_gate.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_e2b_b2_prefill_policy_colab.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("gemma4_e2b_b2_prefill", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_b2_candidate_is_experimental_and_does_not_expand_production_batches():
    paged = (ROOT / "megagemm" / "kernels" / "paged_attention.py").read_text(
        encoding="utf-8"
    )
    model = (ROOT / "megagemm" / "models" / "llama.py").read_text(
        encoding="utf-8"
    )
    flag = "MEGAGEMM_GEMMA4_E2B_L4_B2_PREFILL_EXPERIMENT"
    assert flag in paged
    assert flag in model
    assert "and not b2_experimental" in paged
    assert 'env_prefix = "MEGAGEMM_GEMMA4_E2B_L4_B2_SLIDING_"' in paged
    assert "e2b_l4_full_batch in (4, 8)" in model
    assert "e2b_l4_full_batch == 2" in model
    assert "_GEMMA4_E2B_L4_B2_SLIDING_PREFILL_DISABLED" in paged


def test_b2_gate_covers_attribution_and_ten_combined_geometries():
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
    sliding_only = by_name["sliding_h256_only_g1_bm16_bn64_w4_s2"]
    assert sliding_only["full"] is False
    assert sliding_only["sliding"] is True
    combined = [case for case in module.CASES if case["name"].startswith("combined_")]
    assert len(combined) == 10
    assert len({tuple(case["config"]) for case in combined}) == 10
    assert all(case["full"] and case["sliding"] for case in combined)


def test_b2_summary_requires_exact_hits_digest_stability_and_gain():
    module = _load_module()
    winner_name = "combined_g1_bm16_bn64_w4_s2"
    samples = []
    for repeat in (1, 2, 3):
        for case in module.CASES:
            prefill = 1100.0
            if case["name"] == winner_name:
                prefill = 700.0
            elif case["name"].startswith("combined_"):
                prefill = 800.0
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
    assert result["decision"] == "IMPLEMENT_B2_POLICY_AND_RUN_TARGETED_MACRO_GATE"
    assert result["winner"] == winner_name
    assert result["winner_config"] == (1, 16, 64, 4, 2)
    assert result["speedup"] == 1100.0 / 700.0


def test_b2_colab_harness_is_single_load_drive_scoped_and_non_mutating():
    source = HARNESS.read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in source
    assert "git pull" not in source
    assert "pip install" not in source
    assert "run_gemma4_e2b_b2_prefill_policy_gate.py" in source
    gate = SCRIPT.read_text(encoding="utf-8")
    assert gate.count("InferenceEngine(") == 1
    assert 'print("  model loads: 1")' in gate
    assert "MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID" not in gate
    assert "len(combined)" not in gate
