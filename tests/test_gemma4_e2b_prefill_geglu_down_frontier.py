from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "run_gemma4_e2b_prefill_geglu_down_frontier.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_e2b_prefill_geglu_down_frontier_colab.sh"


def _load_benchmark():
    spec = spec_from_file_location("prefill_geglu_down_frontier", BENCHMARK)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prefill_kernel_preserves_bf16_geglu_boundaries():
    source = (ROOT / "megagemm" / "kernels" / "deepfusion_mlp.py").read_text(
        encoding="utf-8"
    )
    assert "IS_BF16: tl.constexpr" in source
    assert "gate_act = gate_act.to(tl.bfloat16).to(tl.float32)" in source
    assert "act = (gate_act * up).to(tl.bfloat16)" in source
    assert "IS_BF16=gate_up.dtype == torch.bfloat16" in source


def test_model_route_is_opt_in_exact_shape_and_counted():
    source = (ROOT / "megagemm" / "models" / "llama.py").read_text(
        encoding="utf-8"
    )
    assert "self._gemma4_e2b_prefill_geglu_down_enabled = False" in source
    assert "and int(gate_up.shape[0]) == 8" in source
    assert "and int(gate_up.shape[1]) in (521, 2057)" in source
    assert "and int(self.intermediate_size) in (6144, 12288)" in source
    assert "self._gemma4_e2b_prefill_geglu_down_hits += 1" in source
    assert '"activation_down_fused"' in source
    assert 'mode="prefill"' in source
    assert '"gelu_tanh"' in source


def _samples(production: float, candidate: list[float], prompt: int = 512):
    rows = []
    for repeat, candidate_value in enumerate(candidate, start=1):
        rows.extend(
            [
                {
                    "case": "production",
                    "prompt_tokens": prompt,
                    "repeat": repeat,
                    "prefill_ms": production,
                    "wall_ms": production + 20.0,
                    "generated_token_digest": "same",
                    "candidate_hits": 0,
                    "candidate_disabled_layers": 0,
                    "candidate_failures": [],
                },
                {
                    "case": "geglu_down_fused",
                    "prompt_tokens": prompt,
                    "repeat": repeat,
                    "prefill_ms": candidate_value,
                    "wall_ms": candidate_value + 20.0,
                    "generated_token_digest": "same",
                    "candidate_hits": 35,
                    "candidate_disabled_layers": 0,
                    "candidate_failures": [],
                },
            ]
        )
    return rows


def test_summary_requires_ratio_and_paired_speedup():
    module = _load_benchmark()
    promoted = module.summarize(
        _samples(600.0, [570.0, 570.0, 570.0]),
        prompts=[512],
        maximum_spread=1.06,
        minimum_prefill_speedup=1.01,
        minimum_wall_speedup=1.005,
    )
    assert promoted["decision"] == "PROMOTE_SHAPE_DISPATCH"

    # A favorable ratio of medians alone must not promote when two of three
    # paired observations lose.  This rejects thermal/order drift.
    noisy = _samples(600.0, [610.0, 610.0, 500.0])
    rejected = module.summarize(
        noisy,
        prompts=[512],
        maximum_spread=1.30,
        minimum_prefill_speedup=1.01,
        minimum_wall_speedup=1.005,
    )
    assert rejected["decision"] == "KEEP_PRODUCTION"
    assert rejected["shape_policy"]["b8/p512"]["winner"] == "production"


def test_frontier_is_one_load_full_model_and_natural_tokens():
    source = BENCHMARK.read_text(encoding="utf-8")
    assert source.count("matrix.make_runner(") == 1
    assert '"model_loads": 1' in source
    assert "MEGAGEMM_BENCHMARK_TOKEN_DIGEST" in source
    assert "MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID" in source
    assert "both ratio-of-medians and paired-median gates" in source
    assert "profiler and competing engines: disabled" in source


def test_colab_harness_uses_drive_without_mutation_or_competing_engine():
    source = HARNESS.read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in source
    assert "git pull" not in source
    assert "zip" not in source.lower()
    assert "pip install -q vllm" not in source.lower()
    assert "pip install -e" not in source
    assert "run_gemma4_e2b_prefill_geglu_down_frontier.py" in source
