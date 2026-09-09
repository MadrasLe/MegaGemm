from __future__ import annotations

from types import SimpleNamespace
import os

import torch

from benchmarks import run_gemma4_e2b_b1_decode_frontier_gate as gate

from benchmarks.run_gemma4_e2b_b1_decode_frontier_gate import (
    ATTENTION_CASES,
    CORE_CASES,
    DecodeCase,
    _audit_pair,
    _summarize_family,
    _validate_lm_head,
)


def _sample(case: str, rate: float, *, error: str = "") -> dict:
    return {
        "case": case,
        "paired_incremental_decode_tps": rate,
        "short": {"digest": "short", "lengths": [1]},
        "long": {"digest": "long", "lengths": [128]},
        "counter_delta": {},
        "errors": [error] if error else [],
    }


def test_core_frontier_is_exact_four_case_factorial():
    assert [case.name for case in CORE_CASES] == [
        "production",
        "bridge_b1",
        "forced_fused_rms_lm_head",
        "bridge_plus_fused_rms_lm_head",
    ]
    assert [(case.bridge, case.force_fused_lm_head) for case in CORE_CASES] == [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
    assert len(ATTENTION_CASES) == 10
    assert len({case.name for case in ATTENTION_CASES}) == len(ATTENTION_CASES)


def test_lm_head_correctness_probe_disables_autograd():
    class FakeModel:
        lm_head = SimpleNamespace(weight=torch.empty((1, 1536), dtype=torch.bfloat16))

        def _decode_next_token_greedy(self, hidden):
            assert torch.is_inference_mode_enabled()
            return torch.tensor([17], device=hidden.device)

    result = _validate_lm_head(FakeModel(), {})
    assert result == {
        "production_token": 17,
        "forced_fused_token": 17,
        "exact_token_match": True,
    }


def test_family_summary_selects_only_a_stable_full_model_win():
    rates = {
        "production": 33.0,
        "bridge_b1": 33.2,
        "forced_fused_rms_lm_head": 34.0,
        "bridge_plus_fused_rms_lm_head": 34.5,
    }
    samples = [
        _sample(case.name, rates[case.name] * scale)
        for scale in (0.998, 1.0, 1.002)
        for case in CORE_CASES
    ]
    result = _summarize_family(
        "core_factorial",
        CORE_CASES,
        samples,
        minimum_speedup=1.01,
        maximum_spread=1.02,
    )
    assert result["valid"] is True
    assert result["route_digest_match"] is True
    assert result["winner"] == "bridge_plus_fused_rms_lm_head"
    assert result["decision"] == "CANDIDATE_WINS_FULL_MODEL"


def test_family_summary_rejects_counter_or_runtime_errors():
    samples = [
        _sample(
            case.name,
            34.0,
            error=("missing required fast-path hit" if case.name == "bridge_b1" else ""),
        )
        for _repeat in range(3)
        for case in CORE_CASES
    ]
    result = _summarize_family(
        "core_factorial",
        CORE_CASES,
        samples,
        minimum_speedup=1.01,
        maximum_spread=1.02,
    )
    assert result["valid"] is False
    assert result["decision"] == "INVALID_GATE"


def test_exact_b1_layer_hit_audit_accepts_the_specialized_frontier():
    steps = 127
    case = DecodeCase(
        "all",
        "candidate",
        bridge=True,
        force_fused_lm_head=True,
        attention_segments=8,
        attention_tile=16,
        large_gateup=True,
        large_down=True,
    )
    delta = {
        "gemma4_dense_attn_mlp_bridge_decode_hits": 35 * steps,
        "fused_rmsnorm_lm_head_argmax_hits": steps,
        "gemma4_flat_fused_gateup_hits": 35 * steps,
        "gemma4_flat_deepfusion_hits": 20 * steps,
        "gemma4_e2b_b1_large_gateup_hits": 20 * steps,
        "gemma4_e2b_b1_large_down_hits": 20 * steps,
        "paged_gqa2_direct_hits": 28 * steps,
        "paged_generic_direct_hits": 0,
        "paged_grouped_segmented_hits": 7 * steps,
    }
    stats = {
        "paged_decode_runtime": {
            "grouped_segmented_disabled": False,
            "grouped_segmented_failure": "",
            "grouped_segmented_selected_segments": {
                "e2b_l4_b1_full_h512_gqa8": 8,
            },
            "grouped_segmented_selected_tile_sizes": {
                "e2b_l4_b1_full_h512_gqa8": 16,
            },
        },
        "gemma4_dense_attn_mlp_bridge_runtime_disabled": False,
        "fused_rmsnorm_lm_head_argmax_disabled": False,
        "gemma4_flat_fused_gateup_runtime_disabled": False,
    }
    assert _audit_pair(case, delta, stats, decode_steps=steps) == []


def test_h512_followup_preserves_the_promoted_bridge_for_every_case():
    assert len(gate.H512_FOLLOWUP_CASES) == 5
    assert gate.H512_FOLLOWUP_CASES[0].name == "production"
    assert all(case.bridge for case in gate.H512_FOLLOWUP_CASES)
    assert not any(case.force_fused_lm_head for case in gate.H512_FOLLOWUP_CASES)
    assert not any(case.large_gateup or case.large_down for case in gate.H512_FOLLOWUP_CASES)


def test_full_length_warmup_detects_a_late_decode_fallback(monkeypatch):
    calls = []

    def fake_pair(*args, **kwargs):
        calls.append(kwargs["long_tokens"])
        return {"errors": ["grouped attention disabled itself at token 100"]}

    monkeypatch.setattr(gate, "_measure_pair", fake_pair)
    result = gate._warmup_pair(None, [], gate.H512_FOLLOWUP_CASES[1], {}, long_tokens=128, warmups=2)
    assert calls == [128]
    assert result == ["grouped attention disabled itself at token 100"]


def test_numerically_failed_candidate_is_rejected_at_either_context():
    case = gate.H512_FOLLOWUP_CASES[1]
    result = gate._preflight_errors(case, {"cases": [
        {"case": case.name, "context": 2058, "correct": True},
        {"case": case.name, "context": 2184, "correct": False, "error": "OutOfResources"},
    ]})
    assert result == ["attention preflight context=2184: OutOfResources"]
    assert gate._preflight_errors(case, {"cases": []}) == ["missing attention correctness preflight"]


def test_rejected_fast_candidate_cannot_win_the_followup():
    cases = gate.H512_FOLLOWUP_CASES[:3]
    samples = [
        _sample(case.name, rate, error=error)
        for _ in range(3)
        for case, rate, error in (
            (cases[0], 37.0, ""),
            (cases[1], 50.0, "wrong grouped attention hits"),
            (cases[2], 39.0, ""),
        )
    ]
    result = gate._summarize_followup(cases, samples, {}, minimum_speedup=1.01, maximum_spread=1.06)
    assert result["valid"] is True
    assert result["winner"] == cases[2].name
    assert result["all_candidates_valid"] is False
    assert cases[1].name in result["rejected_cases"]
    assert cases[1].name not in result["cases"]


def test_followup_requires_a_valid_baseline_and_at_least_one_valid_candidate():
    cases = gate.H512_FOLLOWUP_CASES[:2]
    samples = [_sample("production", 37.0) for _ in range(3)]
    result = gate._summarize_followup(cases, samples, {cases[1].name: ["compile failed"]}, minimum_speedup=1.01, maximum_spread=1.06)
    assert result["decision"] == "NO_VALID_CANDIDATE"
    assert result["valid"] is False
    assert result["baseline_valid"] is True
    assert result["winner"] is None
    result = gate._summarize_followup(cases, [], {"production": ["wrong route"]}, minimum_speedup=1.01, maximum_spread=1.06)
    assert result["valid"] is False
    assert result["winner"] is None
    assert result["decision"] == "INVALID_BASELINE"


def test_greedy_probe_restores_forced_tokens_even_on_failure(monkeypatch):
    monkeypatch.setenv("MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID", "42")
    monkeypatch.setattr(gate, "_apply_case", lambda *args: None)

    def fake_run(*args):
        assert os.environ["MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID"] == "-1"
        raise RuntimeError("probe failed")

    monkeypatch.setattr(gate, "_run", fake_run)
    result = gate._validate_greedy_route(SimpleNamespace(model=None), [], gate.H512_FOLLOWUP_CASES[0], {}, tokens=16)
    assert result == {"valid": False, "error": "RuntimeError: probe failed"}
    assert os.environ["MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID"] == "42"


def test_followup_main_loads_once_and_never_times_a_rejected_candidate(monkeypatch, tmp_path):
    import json
    from benchmarks import benchmark_inference_matrix as matrix
    import megagemm.engine

    rejected_name = gate.H512_FOLLOWUP_CASES[3].name
    calls = []
    model = SimpleNamespace(
        _prepare_flat_decode=lambda: None,
        _flat_decode_ready=True,
        _gemma4_flat_dense_attn_mlp_input_bufs=[],
    )
    loads = []

    def load_engine(*args, **kwargs):
        loads.append(kwargs)
        return SimpleNamespace(model=model)

    def preflight(cases, **kwargs):
        return {"all_correct": False, "shape": kwargs, "cases": [
            {"case": case.name, "context": kwargs["context"],
             "correct": case.name != rejected_name, "error": "resource limit"}
            for case in cases if case.attention_segments
        ]}

    def measure(engine, prompts, case, state, **kwargs):
        assert case.bridge
        assert case.name != rejected_name
        assert kwargs["long_tokens"] == 128
        calls.append((case.name, kwargs["repeat"]))
        return {**_sample(case.name, 37.0 if case.name == "production" else 39.0),
                "family": "h512_followup", "paired_incremental_decode_ms": 3400.0}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA L4")
    monkeypatch.setattr(megagemm.engine, "InferenceEngine", load_engine)
    monkeypatch.setattr(matrix, "load_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr(matrix, "build_prompts", lambda *a: (["prompt"], 2048))
    for name in ("git_snapshot", "gpu_snapshot", "nvidia_smi_snapshot", "installed_package_versions"):
        monkeypatch.setattr(matrix, name, lambda: {})
    monkeypatch.setattr(gate, "_configure_environment", lambda *a: {})
    monkeypatch.setattr(gate, "_run", lambda *a: {"scheduler_stats": {
        "prefill_chunk_plan": {"total_prompt_tokens": 2057},
    }})
    monkeypatch.setattr(gate, "_capture_lm_state", lambda *a: {})
    monkeypatch.setattr(gate, "_apply_case", lambda *a: None)
    monkeypatch.setattr(gate, "_validate_attention_cases", preflight)
    monkeypatch.setattr(gate, "_measure_pair", measure)
    monkeypatch.setattr(gate, "_validate_greedy_route", lambda *a, **k: {"valid": True, "digest": "unforced"})

    def unexpected(*a):
        raise AssertionError("unrelated LM/MLP preflight should not run")

    monkeypatch.setattr(gate, "_validate_lm_head", unexpected)
    monkeypatch.setattr(gate, "_validate_large_mlp", unexpected)
    output = tmp_path / "decision.json"
    monkeypatch.setattr(gate.sys, "argv", ["gate", "--suite", "h512-followup", "--output", str(output)])
    assert gate.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(loads) == 1
    assert len(calls) == 4 * (1 + 3)
    assert payload["valid"] is True
    assert payload["method"]["baseline_bridge_enabled"] is True
    report = payload["families"]["h512_followup"]
    assert rejected_name in report["rejected_cases"]
    assert rejected_name not in report["cases"]
    assert report["decision"] == "CANDIDATE_WINS_FULL_MODEL"
