from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "run_gemma4_e2b_b1_prefill_policy_gate.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_e2b_b1_prefill_policy_colab.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("gemma4_e2b_b1_prefill", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_b1_candidate_is_experimental_and_production_batches_remain_scoped():
    paged = (ROOT / "megagemm" / "kernels" / "paged_attention.py").read_text(
        encoding="utf-8"
    )
    model = (ROOT / "megagemm" / "models" / "llama.py").read_text(
        encoding="utf-8"
    )
    flag = "MEGAGEMM_GEMMA4_E2B_L4_B1_PREFILL_EXPERIMENT"
    assert flag in paged
    assert flag in model
    assert "and not b1_experimental" in paged
    assert 'env_prefix = "MEGAGEMM_GEMMA4_E2B_L4_B1_SLIDING_"' in paged
    assert "e2b_l4_full_batch in (2, 4, 8)" in model
    assert "e2b_l4_full_batch == 1" in model
    assert "block_rows not in (8, 16, 32, 64)" in paged
    assert "implicit_causal_prefill=(" in model
    assert "self.config.model_type == 'gemma4_text'" in model


def test_b1_gate_has_isolated_attribution_and_twelve_combined_tiles():
    module = _load_module()
    combined = [case for case in module.CASES if case["name"].startswith("combined_")]
    assert len(module.CASES) == 15
    assert len(combined) == 12
    assert len({tuple(case["config"]) for case in combined}) == 12
    assert all(case["full"] and case["sliding"] for case in combined)
    assert module.CASES[0]["name"] == "baseline"
    assert module.CASES[1]["name"] == "full_h512_only"
    assert module.CASES[2]["name"].startswith("sliding_h256_only_")


def test_b1_summary_uses_b1_decisions_without_leaking_common_configuration():
    module = _load_module()
    common_batch = module.common.BATCH_SIZE
    winner_name = "combined_g1_bm8_bn64_w4_s2"
    samples = []
    for repeat in (1, 2, 3):
        for case in module.CASES:
            prefill = 600.0
            if case["name"] == winner_name:
                prefill = 300.0
            elif case["name"].startswith("combined_"):
                prefill = 350.0
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
    assert result["decision"] == "IMPLEMENT_B1_POLICY_AND_RUN_TARGETED_MACRO_GATE"
    assert result["winner"] == winner_name
    assert result["winner_config"] == (1, 8, 64, 4, 2)
    assert module.common.BATCH_SIZE == common_batch


def test_b1_harness_is_drive_scoped_single_load_and_non_mutating():
    source = HARNESS.read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in source
    assert "git pull" not in source
    assert "pip install" not in source
    assert "run_gemma4_e2b_b1_prefill_policy_gate.py" in source
    gate = SCRIPT.read_text(encoding="utf-8")
    assert gate.count("InferenceEngine(") == 0
    assert "common.run(args)" in gate
    assert "MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID" not in gate
