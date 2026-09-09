from types import SimpleNamespace

import pytest
import torch

from megagemm.kernels import gemma4_b1_mlp_gemv as kernels
from megagemm.models.llama import MegaGemmLlama
from benchmarks import run_gemma4_e2b_b1_decode_frontier_gate as gate
from benchmarks.gemma4_b1_mlp_gemv_frontier import build_cases


def test_candidates_keep_bridge_lm_head_and_attention_constant():
    cases = build_cases()
    assert len(cases) == 11
    assert len({c.name for c in cases}) == len(cases)
    assert all(c.bridge and not c.force_fused_lm_head and not c.attention_segments
               and not c.large_gateup and not c.large_down for c in cases)
    assert cases[0].name == "production"
    assert not cases[0].gemv_gateup and not cases[0].gemv_down
    assert {c.gemv_gateup for c in cases if c.gemv_gateup} == set(kernels.GATEUP_CONFIGS)
    assert {c.gemv_down for c in cases if c.gemv_down} == set(kernels.DOWN_CONFIGS)
    assert sum(bool(c.gemv_gateup and c.gemv_down) for c in cases) == 3


@pytest.mark.parametrize("operation,configs,shape", [
    ("gateup", kernels.GATEUP_CONFIGS, (24576, 1536)),
    ("down", kernels.DOWN_CONFIGS, (1536, 12288)),
])
def test_exact_projection_configs(operation, configs, shape):
    for name, config in configs.items():
        assert kernels.projection_spec(operation, name) == (shape, config)
        assert shape[1] % config.splits == 0
        assert config.block_k & (config.block_k - 1) == 0
    with pytest.raises(ValueError, match="Unknown"):
        kernels.projection_spec(operation, "not-a-config")


def test_shape_validation_precedes_launch_or_allocation():
    with pytest.raises(ValueError, match="requires weight"):
        kernels.B1MlpGemvPlan(torch.empty(4, 4), None, "gateup", "wide_n4_k256_w4")
    weight = torch.empty((24576, 1536), device="meta", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA BF16"):
        kernels.B1MlpGemvPlan(weight, None, "gateup", "wide_n4_k256_w4")


def _hidden(rows=1, width=1536, dtype=torch.bfloat16, cuda=True):
    return SimpleNamespace(shape=(rows, width), dtype=dtype, is_cuda=cuda, dim=lambda: 2)


@pytest.mark.parametrize("changes", [
    {"rows": 2}, {"rows": 8}, {"width": 2560}, {"dtype": torch.float16}, {"cuda": False},
])
def test_dispatch_excludes_other_batch_dtype_device_and_hidden_shapes(changes):
    model = SimpleNamespace(runtime_policy=SimpleNamespace(name="gemma4-e2b-l4"),
                            _gemma4_b1_mlp_gemv_configs={"gateup": "wide_n4_k256_w4"})
    lw = SimpleNamespace(is_moe=False, intermediate_size=12288)
    with torch.inference_mode():
        assert MegaGemmLlama._gemma4_b1_mlp_gemv_enabled(model, _hidden(), lw, "gateup")
        assert not MegaGemmLlama._gemma4_b1_mlp_gemv_enabled(model, _hidden(**changes), lw, "gateup")
        model.runtime_policy.name = "gemma4-e4b-l4"
        assert not MegaGemmLlama._gemma4_b1_mlp_gemv_enabled(model, _hidden(), lw, "gateup")


def test_dispatch_is_opt_in_inference_only_and_excludes_small_mlp():
    model = SimpleNamespace(runtime_policy=SimpleNamespace(name="gemma4-e2b-l4"))
    lw = SimpleNamespace(is_moe=False, intermediate_size=12288)
    check = MegaGemmLlama._gemma4_b1_mlp_gemv_enabled
    with torch.inference_mode():
        assert not check(model, _hidden(), lw, "gateup")
        model._gemma4_b1_mlp_gemv_configs = {"gateup": "wide_n4_k256_w4"}
        assert not check(model, _hidden(), lw, "down")
        lw.intermediate_size = 6144
        assert not check(model, _hidden(), lw, "gateup")
        lw.intermediate_size = 12288
        lw.is_moe = True
        assert not check(model, _hidden(), lw, "gateup")
    lw.is_moe = False
    with torch.enable_grad():
        assert not check(model, _hidden(), lw, "gateup")


def test_dispatch_caches_plans_counts_success_and_records_fallback(monkeypatch):
    constructed = []

    class Plan:
        def __init__(self, weight, bias, operation, config):
            constructed.append((weight, operation, config))

        def __call__(self, x, out):
            if x == "fail":
                raise RuntimeError("launch failure")
            return out

    monkeypatch.setattr(kernels, "B1MlpGemvPlan", Plan)
    model = SimpleNamespace(_gemma4_b1_mlp_gemv_configs={"gateup": "wide_n4_k256_w4"})
    lw = SimpleNamespace(gate_up_weight=object(), gate_up_bias=None)
    call = MegaGemmLlama._gemma4_b1_mlp_gemv
    out = object()
    assert call(model, "ok", lw, 15, "gateup", out) is out
    assert call(model, "ok", lw, 15, "gateup", out) is out
    assert len(constructed) == 1
    assert model._gemma4_b1_mlp_gemv_hits == {"gateup": 2}
    assert call(model, "fail", lw, 15, "gateup", out) is None
    assert call(model, "ok", lw, 15, "gateup", out) is None
    assert model._gemma4_b1_mlp_gemv_hits == {"gateup": 2}
    assert model._gemma4_b1_mlp_gemv_failures == {"gateup": "RuntimeError: launch failure"}
    model._gemma4_b1_mlp_gemv_failures = {}
    lw.gate_up_weight = object()
    call(model, "ok", lw, 15, "gateup", out)
    assert len(constructed) == 2


def test_new_audit_rejects_fallback_wrong_config_and_missing_hits():
    case = build_cases()[-1]
    delta = {key: 0 for key in gate.COUNTER_KEYS}
    delta.update(gemma4_dense_attn_mlp_bridge_decode_hits=35 * 127,
                 gemma4_b1_mlp_gemv_gateup_hits=20 * 127,
                 gemma4_b1_mlp_gemv_down_hits=20 * 127,
                 paged_gqa2_direct_hits=28 * 127, paged_generic_direct_hits=7 * 127,
                 paged_grouped_segmented_hits=0)
    stats = {"gemma4_b1_mlp_gemv_configs": {"gateup": case.gemv_gateup, "down": case.gemv_down}}
    assert gate._audit_pair(case, delta, stats, decode_steps=127) == []
    delta["gemma4_b1_mlp_gemv_down_hits"] -= 1
    stats["gemma4_b1_mlp_gemv_configs"]["gateup"] = "wrong"
    stats["gemma4_b1_mlp_gemv_failures"] = {"down": "resource limit"}
    errors = gate._audit_pair(case, delta, stats, decode_steps=127)
    assert any("hits=2539" in e for e in errors)
    assert any("config='wrong'" in e for e in errors)
    assert any("fallback" in e for e in errors)


def test_preflight_rejects_only_cases_using_failed_projection():
    cases = build_cases()
    projections = {key: {"correct": True} for key in (*kernels.GATEUP_CONFIGS, *kernels.DOWN_CONFIGS)}
    projections[cases[1].gemv_gateup] = {"correct": False, "error": "compile failed"}
    report = {"all_correct": False, "projections": projections}
    assert gate._preflight_errors(cases[0], report) == []
    assert "compile failed" in gate._preflight_errors(cases[1], report)[0]
    assert gate._preflight_errors(cases[4], report) == []
    assert gate._preflight_errors(cases[-3], report)


@pytest.mark.parametrize("gateup,down,fail", [
    (False, False, False), (True, False, False), (False, True, False),
    (True, True, False), (True, True, True),
])
@pytest.mark.parametrize("prepared", [False, True])
@torch.inference_mode()
def test_shared_mlp_uses_bridge_output_and_preserves_activation(monkeypatch, gateup, down, fail, prepared):
    # Small CPU tensors exercise the real integration, not the CUDA shape guard.
    calls = []
    x = torch.tensor([[0.5, -1.0]], dtype=torch.bfloat16)
    gw = torch.arange(12, dtype=torch.bfloat16).reshape(6, 2) / 10
    dw = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3) / 10
    lw = SimpleNamespace(intermediate_size=3, gate_up_weight=gw, gate_up_wt=gw.t(),
                         gate_up_bias=None, down_weight=dw, down_wt=dw.t(), down_bias=None)

    def should_not_run(*args):
        raise AssertionError("bridge-normalized input must not be normalized/fused again")

    def candidate(value, weights, index, operation, out):
        calls.append(operation)
        if fail:
            return None
        weight = weights.gate_up_wt if operation == "gateup" else weights.down_wt
        return torch.mm(value, weight, out=out)

    model = SimpleNamespace(
        _flat_norm_eps=1e-6, _gemma4_b1_mlp_gemv_configs={"testing": True},
        _gemma4_b1_mlp_gemv_enabled=lambda hidden, weights, op: gateup if op == "gateup" else down,
        _gemma4_b1_mlp_gemv=candidate,
        _gemma4_flat_should_use_fused_gateup=should_not_run,
        _gemma4_flat_rmsnorm=should_not_run,
        _gemma4_flat_should_use_deepfusion=lambda *args: False,
        _gemma4_flat_cublaslt_gateup_enabled=False,
        _gemma4_flat_gate_up_bufs=[torch.empty(1, 6, dtype=torch.bfloat16)],
        _gemma4_flat_down_bufs=[torch.empty(1, 2, dtype=torch.bfloat16)],
        _flat_fp_linear=MegaGemmLlama._flat_fp_linear,
    )
    kwargs = {}
    if prepared:
        model._gemma4_b1_mlp_gemv_enabled = should_not_run
        model._gemma4_b1_mlp_gemv = should_not_run
        kwargs["gemv_routes"] = (
            (lambda value: candidate(value, lw, 0, "gateup", model._gemma4_flat_gate_up_bufs[0])) if gateup else None,
            (lambda value: candidate(value, lw, 0, "down", model._gemma4_flat_down_bufs[0])) if down else None,
        )
    result = MegaGemmLlama._gemma4_flat_shared_mlp_decode(
        model, x, lw, 0, None, normalized_input=x, **kwargs,
    )
    gate_up = torch.mm(x, gw.t())
    act = torch.nn.functional.gelu(gate_up[:, :3], approximate="tanh")
    act.mul_(gate_up[:, 3:])
    assert torch.equal(result, torch.mm(act, dw.t()))
    assert calls == (["gateup"] if gateup else []) + (["down"] if down else [])


def test_mlp_main_loads_once_skips_unrelated_tests_and_rejects_only_failed_cases(monkeypatch, tmp_path):
    import json
    import megagemm.engine
    from benchmarks import benchmark_inference_matrix as matrix
    from benchmarks import gemma4_b1_mlp_gemv_frontier as frontier
    from benchmarks import compile_gemma4_b1_mlp_gemv as compiler

    cases = build_cases()
    failed_config = cases[1].gemv_gateup
    rejected = {c.name for c in cases if c.gemv_gateup == failed_config}
    loads, measurements = [], []
    model = SimpleNamespace(_prepare_flat_decode=lambda: None, _flat_decode_ready=True,
                            _gemma4_flat_dense_attn_mlp_input_bufs=[])

    def load_engine(*a, **k):
        loads.append(k)
        return SimpleNamespace(model=model)

    def measure(engine, prompts, case, state, **kwargs):
        assert case.bridge and case.name not in rejected
        measurements.append((case.name, kwargs["repeat"]))
        return {"case": case.name, "family": case.family,
                "paired_incremental_decode_tps": 37.0 if case.name == "production" else 39.0,
                "paired_incremental_decode_ms": 3300.0, "counter_delta": {}, "errors": [],
                "short": {"digest": "short", "lengths": [1]},
                "long": {"digest": "long", "lengths": [128]}}

    def unrelated(*a, **k):
        raise AssertionError("unchanged attention/LM head/old fusion must not be re-tested")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA L4")
    monkeypatch.setattr(megagemm.engine, "InferenceEngine", load_engine)
    monkeypatch.setattr(matrix, "load_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr(matrix, "build_prompts", lambda *a: (["prompt"], 2048))
    for name in ("git_snapshot", "gpu_snapshot", "nvidia_smi_snapshot", "installed_package_versions"):
        monkeypatch.setattr(matrix, name, lambda: {})
    monkeypatch.setattr(gate, "_configure_environment", lambda *a: {})
    monkeypatch.setattr(compiler, "compile_kernels", lambda: {"any_biasless_compiled": True})
    monkeypatch.setattr(gate, "_run", lambda *a: {"scheduler_stats": {
        "prefill_chunk_plan": {"total_prompt_tokens": 2057}}})
    monkeypatch.setattr(gate, "_capture_lm_state", lambda *a: {})
    monkeypatch.setattr(gate, "_apply_case", lambda *a: None)
    for name in ("_validate_lm_head", "_validate_attention_cases", "_validate_large_mlp"):
        monkeypatch.setattr(gate, name, unrelated)
    monkeypatch.setattr(frontier, "validate_projections", lambda *a: {
        "all_correct": False, "projections": {
            config: {"correct": config != failed_config, "error": "compile failed"}
            for config in (*kernels.GATEUP_CONFIGS, *kernels.DOWN_CONFIGS)}})
    monkeypatch.setattr(gate, "_measure_pair", measure)
    monkeypatch.setattr(gate, "_validate_greedy_route", lambda *a, **k: {"valid": True, "digest": "unforced"})
    output = tmp_path / "decision.json"
    monkeypatch.setattr(gate.sys, "argv", ["gate", "--suite", "mlp-gemv", "--output", str(output)])
    assert gate.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(loads) == 1
    assert len(measurements) == (len(cases) - len(rejected)) * 4
    assert payload["valid"] and payload["method"]["baseline_bridge_enabled"]
    report = payload["families"]["mlp_gemv"]
    assert report["family"] == "mlp_gemv"
    assert set(report["rejected_cases"]) == rejected
    assert report["decision"] == "CANDIDATE_WINS_FULL_MODEL"


def test_all_compile_failures_stop_before_tokenizer_and_model_load(monkeypatch, tmp_path):
    import json
    import megagemm.engine
    from benchmarks import benchmark_inference_matrix as matrix
    from benchmarks import compile_gemma4_b1_mlp_gemv as compiler

    def unexpected(*args, **kwargs):
        raise AssertionError("failed compilation must stop before any model/tokenizer load")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA L4")
    monkeypatch.setattr(megagemm.engine, "InferenceEngine", unexpected)
    monkeypatch.setattr(matrix, "load_tokenizer", unexpected)
    monkeypatch.setattr(gate, "_configure_environment", lambda *a: {})
    monkeypatch.setattr(compiler, "compile_kernels", lambda: {"any_biasless_compiled": False})
    output = tmp_path / "decision.json"
    monkeypatch.setattr(gate.sys, "argv", ["gate", "--suite", "mlp-gemv", "--output", str(output)])
    assert gate.main() == 2
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["valid"] is False and result["method"]["model_loads"] == 0
    assert result["decision"] == "NO_COMPILED_CANDIDATE"


@pytest.mark.skipif(kernels.triton is None, reason="Triton compiler required, CUDA hardware not required")
def test_compile_all_sm89_specializations_without_loading_a_model():
    from benchmarks.compile_gemma4_b1_mlp_gemv import compile_kernels

    result = compile_kernels()
    assert result["model_loads"] == 0
    assert len(result["cases"]) == 14
    assert result["all_compiled"], result


GPU_CONFIGS = [(op, name) for op, configs in (("gateup", kernels.GATEUP_CONFIGS),
                                            ("down", kernels.DOWN_CONFIGS)) for name in configs]


@pytest.mark.skipif(not torch.cuda.is_available() or kernels.triton is None,
                    reason="CUDA and Triton required")
@pytest.mark.parametrize("operation,name", GPU_CONFIGS)
@pytest.mark.parametrize("with_bias", [False, True])
@torch.inference_mode()
def test_cuda_exact_shapes_numerics_repeatability_and_reused_buffers(operation, name, with_bias):
    shape, _ = kernels.projection_spec(operation, name)
    generator = torch.Generator(device="cuda").manual_seed(12)
    weight = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.02
    bias = torch.randn(shape[0], device="cuda", dtype=torch.bfloat16, generator=generator) if with_bias else None
    plan = kernels.B1MlpGemvPlan(weight, bias, operation, name)
    out = torch.empty((1, shape[0]), device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        x = torch.randn((1, shape[1]), device="cuda", dtype=torch.bfloat16, generator=generator)
        reference = torch.mm(x, weight.t())
        if bias is not None:
            reference.add_(bias)
        out.fill_(float("nan"))
        if plan.partial is not None:
            plan.partial.fill_(float("nan"))
        assert plan(x, out) is out
        actual = out.clone()
        if plan.partial is not None:
            plan.partial.fill_(float("nan"))
        plan(x, out)
        assert torch.equal(out, actual)
        # Prepared launch must preserve the checked launch's exact arithmetic.
        bound = plan.bind_output(out)
        out.fill_(float("nan"))
        assert bound(x) is out
        assert torch.equal(out, actual)
        assert torch.isfinite(out).all()
        assert torch.linalg.vector_norm(out.float() - reference.float()) / reference.float().norm() < 0.01
        assert torch.nn.functional.cosine_similarity(out.float(), reference.float()).item() > 0.9999


def test_dispatch_cases_are_matched_kernel_pairs():
    from benchmarks.gemma4_b1_mlp_gemv_frontier import build_dispatch_cases
    cases = build_dispatch_cases()
    assert len(cases) == 7 and cases[0].name == "production"
    for legacy, prepared in zip(cases[1::2], cases[2::2]):
        assert legacy.bridge and prepared.bridge
        assert not legacy.prepared_gemv and prepared.prepared_gemv
        assert (legacy.gemv_gateup, legacy.gemv_down) == (prepared.gemv_gateup, prepared.gemv_down)


@pytest.mark.parametrize("splits", [1, 4])
def test_bound_launcher_builds_grid_once_and_never_inspects_input_or_enters_device(monkeypatch, splits):
    grids, calls = [], []

    class Kernel:
        def __getitem__(self, grid):
            grids.append(grid)
            return lambda *args, **kwargs: calls.append((args, kwargs))

    monkeypatch.setattr(kernels, "triton", SimpleNamespace(cdiv=lambda n, d: (n+d-1)//d,
                                                         next_power_of_2=lambda n: n))
    monkeypatch.setattr(kernels, "_b1_mlp_partial", Kernel(), raising=False)
    monkeypatch.setattr(kernels, "_b1_mlp_reduce", Kernel(), raising=False)
    monkeypatch.setattr(torch.cuda, "device", lambda *args: pytest.fail("device context in hot path"))
    plan = kernels.B1MlpGemvPlan.__new__(kernels.B1MlpGemvPlan)
    plan.weight, plan.bias = torch.empty(8, 16, dtype=torch.bfloat16), None
    plan.n, plan.k = 8, 16
    plan.config = kernels.GemvConfig(4, 256, splits, 4)
    plan.partial = torch.empty(splits, 8) if splits > 1 else None
    out = torch.empty(1, 8, dtype=torch.bfloat16)
    bound = plan.bind_output(out)
    launches = 1 if splits == 1 else 2
    assert len(grids) == launches
    for _ in range(3):
        assert bound(object()) is out  # no shape/device lookup on x
    assert len(grids) == launches and len(calls) == 3 * launches
    with pytest.raises(ValueError, match="bound output"):
        plan.bind_output(torch.empty(2, 8, dtype=torch.bfloat16))


@torch.inference_mode()
def test_preparation_reuses_plans_and_guards_executor_boundary(monkeypatch):
    constructed = []

    class Plan:
        def __init__(self, *args):
            constructed.append(args)

        def bind_output(self, out):
            def run(x):
                if x == "fail":
                    raise RuntimeError("launch failed")
                return out
            return run

    monkeypatch.setattr(kernels, "B1MlpGemvPlan", Plan)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    device = torch.device("cuda:0")
    weights = [SimpleNamespace(is_moe=False, intermediate_size=12288 if i >= 15 else 6144,
                               gate_up_weight=SimpleNamespace(device=device), gate_up_bias=None,
                               down_weight=SimpleNamespace(device=device), down_bias=None) for i in range(35)]
    model = SimpleNamespace(runtime_policy=SimpleNamespace(name="gemma4-e2b-l4"),
                            _flat_layer_weights=weights, _gemma4_b1_mlp_gemv_failures={},
                            _gemma4_b1_mlp_gemv_configs={"gateup": "wide_n4_k2048_w4", "down": "down_n4_k256_s1_w4"},
                            _gemma4_flat_gate_up_bufs=[object() for _ in weights],
                            _gemma4_flat_down_bufs=[object() for _ in weights])
    prepare = MegaGemmLlama._prepare_gemma4_b1_mlp_gemv_routes
    prepare(model)
    prepare(model)
    assert len(constructed) == 40  # cached across repeated pairs/cases
    routes = model._gemma4_b1_mlp_prepared_routes
    assert routes[:15] == [(None, None)] * 15
    model._gemma4_b1_mlp_gemv_plans = None  # no cache access during launches
    assert routes[15][0]("ok") is model._gemma4_flat_gate_up_bufs[15]
    assert routes[15][1]("ok") is model._gemma4_flat_down_bufs[15]
    assert model._gemma4_b1_mlp_prepared_hits == {"gateup": 1, "down": 1}
    assert model._gemma4_b1_mlp_gemv_hits == {"gateup": 1, "down": 1}
    assert routes[15][0]("fail") is None
    assert routes[16][0]("ok") is None
    assert "launch failed" in model._gemma4_b1_mlp_gemv_failures["gateup"]
    hidden = SimpleNamespace(shape=(1, 1536), dtype=torch.bfloat16, device=device, is_cuda=True)
    guard = MegaGemmLlama._gemma4_b1_mlp_routes_for_decode
    assert guard(model, hidden) is routes
    hidden.shape = (8, 1536)
    assert guard(model, hidden) is None
    hidden.shape = (1, 1536)
    model._gemma4_flat_down_bufs = list(model._gemma4_flat_down_bufs)
    assert guard(model, hidden) is None
    assert "boundary" in model._gemma4_b1_mlp_gemv_failures
