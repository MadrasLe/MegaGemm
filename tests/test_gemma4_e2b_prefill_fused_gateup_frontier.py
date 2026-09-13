from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "run_gemma4_e2b_prefill_fused_gateup_frontier.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_e2b_prefill_fused_gateup_frontier_colab.sh"
KERNEL = ROOT / "megagemm" / "kernels" / "gemma4_e2b_prefill_mlp.py"


def _load_benchmark():
    spec = spec_from_file_location("prefill_fused_gateup_frontier", BENCHMARK)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kernel_preserves_production_rounding_boundaries():
    source = KERNEL.read_text(encoding="utf-8")
    assert "gate_acc += tl.dot" in source
    assert "up_acc += tl.dot" in source
    assert "gate_acc.to(tl.bfloat16).to(tl.float32)" in source
    assert "up_acc.to(tl.bfloat16).to(tl.float32)" in source
    assert ").to(tl.bfloat16)" in source
    assert "activated = gelu.to(tl.float32) * up" in source


def test_model_route_is_opt_in_exact_shape_and_counted():
    source = (ROOT / "megagemm" / "models" / "llama.py").read_text(
        encoding="utf-8"
    )
    assert "self._gemma4_e2b_prefill_fused_gateup_enabled = False" in source
    assert "and int(x.shape[0]) == 8" in source
    assert "and int(x.shape[1]) in (521, 2057)" in source
    assert "and int(x.shape[2]) == 1536" in source
    assert "and int(self.intermediate_size) in (6144, 12288)" in source
    assert "self._gemma4_e2b_prefill_fused_gateup_hits += 1" in source
    assert '"gate_up_activation_fused"' in source


def test_frontier_has_one_model_load_and_multiple_geometries():
    module = _load_benchmark()
    source = BENCHMARK.read_text(encoding="utf-8")
    assert len(module.CONFIGS) == 6
    assert source.count("matrix.make_runner(") == 1
    assert '"model_loads": 1' in source
    assert "competing engine install: disabled" in source
    assert "20xI12288 + 15xI6144" in source


def test_summary_selects_shape_specific_fused_configs():
    module = _load_benchmark()
    config_a = module.CONFIGS[0]
    config_b = module.CONFIGS[1]
    cases = [("production", None), config_a, config_b]
    samples = []
    for prompt, baseline, winners in (
        (512, 600.0, {config_a[0]: 570.0, config_b[0]: 580.0}),
        (2048, 1800.0, {config_a[0]: 1770.0, config_b[0]: 1680.0}),
    ):
        digest = f"digest-{prompt}"
        for name, config in cases:
            value = baseline if config is None else winners[name]
            for repeat in range(1, 4):
                samples.append(
                    {
                        "case": name,
                        "config": config,
                        "prompt_tokens": prompt,
                        "repeat": repeat,
                        "warmup": False,
                        "prefill_ms": value,
                        "wall_ms": value + 20.0,
                        "generated_token_digest": digest,
                        "candidate_hits": 0 if config is None else 35,
                        "candidate_disabled_layers": 0,
                        "candidate_failures": [],
                    }
                )
    summary = module.summarize(
        samples,
        active_cases=cases,
        prompts=[512, 2048],
        maximum_spread=1.06,
        minimum_prefill_speedup=1.01,
        minimum_wall_speedup=1.005,
    )
    assert summary["decision"] == "PROMOTE_SHAPE_DISPATCH"
    assert summary["shape_policy"]["b8/p512"]["winner"] == config_a[0]
    assert summary["shape_policy"]["b8/p2048"]["winner"] == config_b[0]


def test_colab_harness_targets_drive_without_repo_mutation_or_vllm():
    source = HARNESS.read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in source
    assert "git pull" not in source
    assert "zip" not in source.lower()
    assert "pip install -q vllm" not in source.lower()
    assert "pip install -e" not in source
    assert "gemma4_e2b_prefill_fused_gateup_frontier.py" in source
