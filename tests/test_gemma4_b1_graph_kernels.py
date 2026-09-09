from types import SimpleNamespace

import pytest

from benchmarks import gemma4_b1_graph_kernels as kernels
from benchmarks import run_gemma4_e2b_b1_execution_gate as gate


def test_five_candidates_keep_the_same_one_step_executor():
    experiment = kernels.GraphKernelExperiment()
    assert len(experiment.cases) == 5
    assert all(c.graph and c.reuse and not c.unroll and not c.eager_control for c in experiment.cases)
    envs = [gate.case_environment(c) for c in experiment.cases]
    assert all(e == envs[0] for e in envs)
    assert kernels.SPECS["graph_production"] == ({}, False)
    assert kernels.SPECS["graph_fused_lm_head"] == ({}, True)


@pytest.mark.parametrize("case_name", kernels.SPECS)
def test_capture_route_audit_requires_exactly_twenty_large_layer_hits(case_name):
    experiment = kernels.GraphKernelExperiment()
    case = next(c for c in experiment.cases if c.name == case_name)
    baseline = kernels.expected_routes("graph_production", {"bridge": 35, "paged_gqa2_direct_hits": 28})
    expected = kernels.expected_routes(case_name, baseline)
    assert not experiment.audit_routes(case, [expected], [baseline])
    bad = dict(expected)
    bad["bridge"] = 0
    assert experiment.audit_routes(case, [bad], [baseline])
    if kernels.SPECS[case_name][0]:
        op = next(iter(kernels.SPECS[case_name][0]))
        bad = dict(expected)
        bad[f"gemma4_b1_mlp_prepared_{op}_hits"] = 19
        assert experiment.audit_routes(case, [bad], [baseline])


def test_fused_lm_head_requires_actual_hits_not_just_a_flag():
    experiment = kernels.GraphKernelExperiment()
    case = experiment.cases[-1]
    assert experiment.audit_cold(case, {})
    assert not experiment.audit_cold(case, {"fused_rmsnorm_lm_head_argmax_hits": 2})


def make_prepared(monkeypatch, bad_projection=None):
    from benchmarks import gemma4_b1_mlp_gemv_frontier as frontier
    model = SimpleNamespace(
        _gemma4_flat_dense_attn_mlp_bridge_enabled=True,
        _gemma4_flat_dense_attn_mlp_bridge_runtime_disabled=False,
        lm_state={"fused": False}, prepare_count=0,
    )
    def prepare():
        model.prepare_count += 1
        model._gemma4_b1_mlp_prepared_routes = object()
        model._gemma4_b1_mlp_prepared_weights = object()
        model._gemma4_b1_mlp_prepared_buffers = object()
        model._gemma4_b1_mlp_prepared_device = "cuda:0"
    model._prepare_gemma4_b1_mlp_gemv_routes = prepare
    def validate(model, selected_configs):
        assert selected_configs == kernels.SELECTED
        return {"projections": {name: {"correct": name != bad_projection, "checked_inputs": 40}
                                for names in selected_configs.values() for name in names}}
    monkeypatch.setattr(frontier, "validate_projections", validate)
    monkeypatch.setattr(kernels, "_validate_lm_head", lambda *a: {"exact_token_match": True})
    monkeypatch.setattr(kernels, "_capture_lm_state", lambda m: dict(m.lm_state))
    monkeypatch.setattr(kernels, "_restore_lm_state", lambda m, s: setattr(m, "lm_state", dict(s)))
    monkeypatch.setattr(kernels, "_force_fused_rms_lm_head", lambda m, s: setattr(m, "lm_state", {"fused": True}))
    experiment = kernels.GraphKernelExperiment()
    rejected = experiment.prepare(model, model.lm_state)
    return model, experiment, rejected


def test_switching_cases_restores_route_owner_and_does_not_erase_failures(monkeypatch):
    model, experiment, rejected = make_prepared(monkeypatch)
    assert not rejected and model.prepare_count == 3
    routes = {}
    for case in experiment.cases:
        experiment.activate(model, case)
        assert not experiment.audit_state(model, case)
        routes[case.name] = model._gemma4_b1_mlp_prepared_routes
    gateup = experiment.cases[1]
    experiment.activate(model, gateup)
    assert model._gemma4_b1_mlp_prepared_routes is routes[gateup.name]
    model._gemma4_b1_mlp_gemv_failures["gateup"] = "launch failed"
    experiment.activate(model, experiment.cases[0])
    assert model._gemma4_b1_mlp_prepared_routes is None and model.lm_state == {"fused": False}
    experiment.activate(model, gateup)
    assert experiment.audit_state(model, gateup)
    assert model.prepare_count == 3  # no route rebuild between measured requests


def test_projection_rejection_excludes_only_dependent_cases(monkeypatch):
    model, experiment, rejected = make_prepared(monkeypatch, bad_projection=kernels.GATEUP)
    assert set(rejected) == {"graph_gateup", "graph_both"}
    assert "graph_production" in experiment.bindings and "graph_fused_lm_head" in experiment.bindings


def test_nonstandard_baseline_name_in_summary():
    experiment = kernels.GraphKernelExperiment()
    samples = []
    for case in experiment.cases:
        for _ in range(3):
            faster = case.name == "graph_both"
            samples.append({"case": case.name, "incremental_decode_s": 3.0 if faster else 3.3,
                            "decode_tps": 127 / (3.0 if faster else 3.3),
                            "long": {"output_tps": 38.0 if faster else 36.0, "elapsed_s": 3.4 if faster else 3.6},
                            "errors": []})
    result = gate.summarize(samples, {}, repeats=3, minimum_speedup=1.02, maximum_spread=1.06, cases=experiment.cases)
    assert result["valid"] and result["winner"] == "graph_both"
    assert result["rows"][0]["speedup"] == 1


def test_rejected_candidate_is_excluded_without_invalidating_completed_comparison():
    experiment = kernels.GraphKernelExperiment()
    samples = [{"case": "graph_production", "incremental_decode_s": 3.3,
                "decode_tps": 127 / 3.3, "long": {"output_tps": 36, "elapsed_s": 3.55},
                "errors": []} for _ in range(3)]
    rejected = {c.name: ["token parity failed"] for c in experiment.cases[1:]}
    result = gate.summarize(samples, rejected, repeats=3, minimum_speedup=1.02,
                            maximum_spread=1.06, cases=experiment.cases, allow_rejected=True)
    assert result["valid"] and result["winner"] == "graph_production"
    assert all(r["status"] == "rejected" and not r["valid"] for r in result["rows"][1:])
    result = gate.summarize([], rejected, repeats=3, minimum_speedup=1.02,
                            maximum_spread=1.06, cases=experiment.cases, allow_rejected=True)
    assert not result["valid"] and result["winner"] is None


def test_incorrect_candidate_is_measured_but_cannot_win_or_promote():
    experiment = kernels.GraphKernelExperiment()
    samples = []
    for case in experiment.cases:
        for _ in range(3):
            faster = case.name == "graph_both"
            elapsed = 2.7 if faster else 3.3
            samples.append({
                "case": case.name,
                "incremental_decode_s": elapsed,
                "decode_tps": 127 / elapsed,
                "long": {"output_tps": 42.0 if faster else 36.0, "elapsed_s": elapsed + .2},
                "errors": [],
            })
    result = gate.summarize(
        samples, {}, repeats=3, minimum_speedup=1.02, maximum_spread=1.06,
        cases=experiment.cases,
        correctness={"graph_both": ["natural greedy diverged"]},
    )
    both = next(row for row in result["rows"] if row["case"] == "graph_both")
    assert result["valid"] and result["decision"] == "KEEP_PRODUCTION"
    assert result["performance_winner"] == "graph_both"
    assert not both["valid"] and both["measurement_valid"] and not both["correct"]


def test_compiler_preflight_selects_only_the_two_existing_configs(monkeypatch):
    from benchmarks import compile_gemma4_b1_mlp_gemv as compiler
    called = []
    def compile_kernels(selected_configs):
        called.append(selected_configs)
        return {"all_compiled": True}
    monkeypatch.setattr(compiler, "compile_kernels", compile_kernels)
    assert kernels.GraphKernelExperiment().compile_preflight()["all_compiled"]
    assert called == [{"gateup": (kernels.GATEUP,), "down": (kernels.DOWN,)}]
