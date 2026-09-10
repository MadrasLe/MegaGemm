from benchmarks import run_gemma4_e2b_b8_compute_frontier as gate


def _sample(case, workload, decode_tps, output_tps, repeat, errors=None):
    return {
        "case": case,
        "workload": workload,
        "decode_tps": decode_tps,
        "output_tps": output_tps,
        "repeat": repeat,
        "errors": list(errors or ()),
    }


class _LazyFlatModel:
    def __init__(self):
        self._flat_decode_ready = False
        self._flat_decode_failed_reason = ""
        self._gemma4_flat_policy_fused_gateup_rows = ()
        self._gemma4_flat_policy_deepfusion_rows = ()
        self._gemma4_flat_policy_cublas_gateup_rows = ()
        self._gemma4_flat_policy_cublas_down_rows = ()
        self._gemma4_flat_cublaslt_gateup_enabled = False
        self._gemma4_flat_cublaslt_gateup_algorithms = {}

    def _prepare_flat_decode(self):
        self._flat_decode_ready = True
        self._gemma4_flat_policy_cublas_gateup_rows = (8,)
        self._gemma4_flat_fused_gateup_use_cache = {"compiled": True}
        self._gemma4_flat_deepfusion_use_cache = {"compiled": True}


def test_production_mlp_state_is_captured_after_lazy_flat_prepare():
    model = _LazyFlatModel()
    state = gate._prepare_production_mlp_state(model)
    assert state["_gemma4_flat_policy_cublas_gateup_rows"] == (8,)
    gate._restore_mlp_state(model, state)
    assert model._gemma4_flat_fused_gateup_use_cache == {}
    assert model._gemma4_flat_deepfusion_use_cache == {}


def test_restore_mlp_state_accepts_unmaterialized_optional_caches():
    model = _LazyFlatModel()
    gate._restore_mlp_state(model, gate._production_mlp_state(model))


def test_compute_frontier_covers_context_and_generation_shapes():
    workloads = tuple(
        gate.Workload(prompt, output)
        for prompt in (512, 2048)
        for output in (16, 128)
    )
    assert [workload.key for workload in workloads] == [
        "p512_o16",
        "p512_o128",
        "p2048_o16",
        "p2048_o128",
    ]
    assert gate.SCREEN_CASES[0] == gate.PRODUCTION
    assert {case.family for case in gate.SCREEN_CASES} >= {
        "attention",
        "lm_head",
        "mlp",
    }


def test_targeted_screen_always_includes_production_once():
    selected = gate.select_screen_cases("lm_head_bn64,production,lm_head_bn64")
    assert [case.name for case in selected] == ["production", "lm_head_bn64"]


def test_targeted_screen_rejects_unknown_case():
    try:
        gate.select_screen_cases("not_a_real_case")
    except ValueError as exc:
        assert "not_a_real_case" in str(exc)
    else:
        raise AssertionError("unknown case was accepted")


def test_setup_discards_reference_graphs_before_route_audit():
    workloads = (
        gate.Workload(512, 16),
        gate.Workload(512, 128),
        gate.Workload(2048, 16),
        gate.Workload(2048, 128),
    )
    schedulers = {
        ("production", "p512_o1"): object(),
        ("production", "p512_o16"): object(),
        ("production", "p512_o128"): object(),
        ("production", "p2048_o1"): object(),
        ("production", "p2048_o16"): object(),
        ("production", "p2048_o128"): object(),
        ("lm_head_bn64", "p512_o16"): object(),
    }
    gate._discard_case_schedulers(schedulers, gate.PRODUCTION, workloads)
    assert list(schedulers) == [("lm_head_bn64", "p512_o16")]


def test_adaptive_attention_changes_only_short_context():
    candidate = next(
        case for case in gate.SCREEN_CASES
        if case.name == "full_h512_short_seg8"
    )
    assert candidate.h512_segments(512) == 8
    assert candidate.h512_segments(2048) == 32


def test_attention_route_audit_rejects_silent_production_replay():
    case = gate.ComputeCase(
        "short_seg8", "attention", h512_short_segments=8
    )
    runtime = {
        "paged_decode_runtime": {
            "grouped_segmented_selected_segments": {
                "e2b_l4_full_h512_gqa8": 32,
            },
            "grouped_segmented_selected_tile_sizes": {
                "e2b_l4_full_h512_gqa8": 16,
            },
        }
    }
    errors = gate._attention_route_errors(
        case, gate.Workload(512, 128), runtime
    )
    assert errors == ["H512 route selected 32 segments, expected 8"]


def test_screen_selects_family_winners_and_builds_combination():
    cases = (
        gate.PRODUCTION,
        gate.ComputeCase(
            "attention_win", "attention", h512_short_segments=8
        ),
        gate.ComputeCase("lm_loss", "lm_head", lm_block_n=64),
        gate.ComputeCase("mlp_win", "mlp", mlp_mode="fused_gateup"),
    )
    workloads = (gate.Workload(512, 16), gate.Workload(2048, 128))
    samples = []
    for repeat in (1, 2):
        for workload in workloads:
            samples.extend(
                [
                    _sample("production", workload.key, 100, 80, repeat),
                    _sample("attention_win", workload.key, 104, 83, repeat),
                    _sample("lm_loss", workload.key, 99, 79, repeat),
                    _sample("mlp_win", workload.key, 103, 82, repeat),
                ]
            )
    summary = gate.summarize_screen(
        samples, cases, workloads, repeats=2, maximum_spread=1.05
    )
    assert summary["family_winners"] == {
        "attention": "attention_win",
        "lm_head": "production",
        "mlp": "mlp_win",
    }
    combined = gate.combine_family_winners(cases, summary["family_winners"])
    assert combined.h512_short_segments == 8
    assert combined.lm_block_n == 256
    assert combined.mlp_mode == "fused_gateup"


def test_final_decision_requires_decode_and_end_to_end_win():
    summary = {
        "cases": {
            "production": {"valid": True},
            "best_combination": {
                "valid": True,
                "geomean_decode_speedup": 1.03,
                "geomean_output_speedup": 1.02,
                "worst_decode_speedup": 1.0,
                "worst_output_speedup": 1.0,
                "scenarios": {"p512_o16": {"spread": 1.01}},
            },
        }
    }
    decision = gate.final_decision(
        summary,
        minimum_speedup=1.015,
        maximum_spread=1.08,
        policy_changed=True,
    )
    assert decision["apply_change"] is True
    summary["cases"]["best_combination"]["geomean_output_speedup"] = 1.01
    assert gate.final_decision(
        summary,
        minimum_speedup=1.015,
        maximum_spread=1.08,
        policy_changed=True,
    )["apply_change"] is False


def test_final_decision_rejects_one_bad_end_to_end_scenario():
    summary = {
        "cases": {
            "production": {"valid": True},
            "best_combination": {
                "valid": True,
                "geomean_decode_speedup": 1.03,
                "geomean_output_speedup": 1.02,
                "worst_decode_speedup": 1.0,
                "worst_output_speedup": 0.98,
                "scenarios": {"p512_o16": {"spread": 1.01}},
            },
        }
    }
    assert gate.final_decision(
        summary,
        minimum_speedup=1.015,
        maximum_spread=1.08,
        policy_changed=True,
    )["apply_change"] is False


def test_identical_combination_cannot_be_promoted_from_noise():
    assert gate.changes_compute_policy(
        gate.ComputeCase("best_combination", "combined")
    ) is False


def test_unchanged_final_ignores_duplicate_candidate_noise():
    summary = {
        "cases": {
            "production": {"valid": True},
            "best_combination": {"valid": False},
        }
    }
    decision = gate.final_decision(
        summary,
        minimum_speedup=1.015,
        maximum_spread=1.08,
        policy_changed=False,
    )
    assert decision["decision"] == "KEEP_PRODUCTION"
    assert decision["valid"] is True
    assert decision["geomean_decode_speedup"] == 1.0


def test_colab_wrapper_uses_drive_without_git_or_vllm():
    wrapper = (
        gate.ROOT / "benchmarks" /
        "run_gemma4_e2b_b8_compute_frontier_colab.sh"
    ).read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in wrapper
    assert "git pull" not in wrapper
    assert "vllm" not in wrapper.lower()
    assert "SCREEN_REPEATS" in wrapper
    assert "FINAL_REPEATS" in wrapper
    assert "SCREEN_CASE_NAMES" in wrapper
