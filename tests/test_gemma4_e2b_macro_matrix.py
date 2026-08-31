from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "run_gemma4_e2b_macro_matrix.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_e2b_macro_matrix_colab.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("gemma4_e2b_macro_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample(batch, prompt, output, repeat, elapsed, digest="same"):
    return {
        "key": f"b{batch}/p{prompt}/o{output}",
        "batch_size": batch,
        "prompt_tokens_requested_per_request": prompt,
        "prompt_tokens_actual_total": batch * prompt,
        "max_new_tokens_per_request": output,
        "repeat": repeat,
        "order_position": 1,
        "elapsed_s": elapsed,
        "generated_tokens": batch * output,
        "output_tps": batch * output / elapsed,
        "prefill_ms": 1000.0,
        "decode_ms": max(0.0, elapsed * 1000.0 - 1000.0),
        "internal_decode_tps": 100.0,
        "generated_token_digest": digest,
        "generated_token_lengths": [output] * batch,
        "forced_token_id": 42,
        "audit": {},
    }


def test_macro_summary_pairs_every_long_case_with_o1():
    module = _load_module()
    samples = []
    for repeat in (1, 2, 3):
        samples.extend(
            [
                _sample(8, 2048, 1, repeat, 2.0),
                _sample(8, 2048, 16, repeat, 2.5),
                _sample(8, 2048, 128, repeat, 5.0),
            ]
        )
    result = module.summarize_samples(samples, maximum_spread=1.08)
    assert result["aggregate"]["scenario_count"] == 3
    assert result["aggregate"]["sample_count"] == 9
    assert result["aggregate"]["valid"] is True
    assert result["paired_decode"]["b8/p2048/o1-o16"][
        "median_incremental_decode_tps"
    ] == pytest.approx(8 * 15 / 0.5)
    assert result["paired_decode"]["b8/p2048/o1-o128"][
        "median_incremental_decode_tps"
    ] == pytest.approx(8 * 127 / 3.0)


def test_macro_comparison_uses_ttft_and_paired_decode():
    module = _load_module()
    base_samples = []
    cand_samples = []
    for repeat in (1, 2, 3):
        base_samples.extend(
            [_sample(8, 2048, 1, repeat, 2.0), _sample(8, 2048, 128, repeat, 5.0)]
        )
        cand_samples.extend(
            [_sample(8, 2048, 1, repeat, 1.8), _sample(8, 2048, 128, repeat, 4.4)]
        )

    def payload(label, samples):
        return {
            "benchmark": "gemma4_e2b_l4_macro_matrix",
            "schema_version": 1,
            "label": label,
            "model": module.DEFAULT_MODEL,
            "dtype": "bf16",
            "hardware_label": "1xl4",
            "matrix": {
                "batch_sizes": [8],
                "prompt_tokens": [2048],
                "output_tokens": [1, 128],
            },
            "summary": module.summarize_samples(samples, maximum_spread=1.08),
        }

    result = module.compare_payloads(
        payload("baseline", base_samples),
        payload("candidate", cand_samples),
        minimum_geomean_speedup=1.03,
        maximum_regression=0.03,
    )
    assert result["decision"] == "RECORD_SCENARIO_RESULTS"
    assert result["decision_mode"] == "regime_record"
    assert result["full_dispatch_qualifies"] is True
    assert result["wins"] == 2
    assert result["digest_mismatches"] == []
    assert result["scenarios"][0]["primary_metric"] == "first_token_wall"
    assert result["scenarios"][1]["primary_metric"] == "paired_incremental_decode_tps"
    assert result["policy_map"]["candidate_cells"] == [
        "b8/p2048/o1",
        "b8/p2048/o128",
    ]


def test_only_an_integrated_dispatch_can_be_promoted():
    module = _load_module()
    base_samples = []
    cand_samples = []
    for repeat in (1, 2, 3):
        base_samples.extend(
            [_sample(8, 2048, 1, repeat, 2.0), _sample(8, 2048, 128, repeat, 5.0)]
        )
        cand_samples.extend(
            [_sample(8, 2048, 1, repeat, 1.8), _sample(8, 2048, 128, repeat, 4.4)]
        )

    def payload(label, samples):
        return {
            "benchmark": "gemma4_e2b_l4_macro_matrix",
            "schema_version": 1,
            "label": label,
            "model": module.DEFAULT_MODEL,
            "dtype": "bf16",
            "hardware_label": "1xl4",
            "matrix": {
                "batch_sizes": [8],
                "prompt_tokens": [2048],
                "output_tokens": [1, 128],
            },
            "summary": module.summarize_samples(samples, maximum_spread=1.08),
        }

    result = module.compare_payloads(
        payload("baseline", base_samples),
        payload("integrated_dispatch", cand_samples),
        minimum_geomean_speedup=1.03,
        maximum_regression=0.03,
        final_dispatch_gate=True,
    )
    assert result["decision"] == "PROMOTE_SHAPE_DISPATCH"
    assert result["decision_mode"] == "final_dispatch_gate"


def test_macro_comparison_rejects_digest_mismatch():
    module = _load_module()
    base = [_sample(1, 128, 1, repeat, 1.0, "base") for repeat in (1, 2)]
    cand = [_sample(1, 128, 1, repeat, 0.5, "candidate") for repeat in (1, 2)]

    def payload(label, samples):
        return {
            "benchmark": "gemma4_e2b_l4_macro_matrix",
            "schema_version": 1,
            "label": label,
            "model": module.DEFAULT_MODEL,
            "dtype": "bf16",
            "hardware_label": "1xl4",
            "matrix": {"batch_sizes": [1], "prompt_tokens": [128], "output_tokens": [1]},
            "summary": module.summarize_samples(samples, maximum_spread=1.08),
        }

    result = module.compare_payloads(
        payload("base", base),
        payload("candidate", cand),
        minimum_geomean_speedup=1.03,
        maximum_regression=0.03,
    )
    assert result["decision"] == "INVALID_MACRO_GATE"
    assert result["digest_mismatches"] == ["b1/p128/o1"]


def test_colab_harness_is_drive_scoped_and_does_not_mutate_environment():
    source = HARNESS.read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in source
    assert "git pull" not in source
    assert "pip install" not in source
    assert "1,2,4,8" in source
    assert "128,512,2048" in source
    assert "1,16,128" in source
    assert "COMPARE_WITH" in source
    assert 'if [[ -f "$OUT/comparison.json" ]]' in source
    assert source.rstrip().endswith("exit 0")


def test_macro_warms_every_scenario_before_measurement():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "for output in outputs:" in source
    assert '"warmup_per_scenario": args.warmups' in source
    assert "warmup_output = max(outputs)" not in source
