from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "run_gemma4_e2b_prefill_ple_tail_frontier.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_e2b_prefill_ple_tail_frontier_colab.sh"


def _load_benchmark():
    spec = spec_from_file_location("prefill_ple_tail_frontier", BENCHMARK)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_route_is_opt_in_exact_shape_and_keeps_gemms():
    source = (ROOT / "megagemm" / "models" / "llama.py").read_text(encoding="utf-8")
    assert "self._gemma4_e2b_prefill_ple_tail_enabled = False" in source
    assert "self._gemma4_e2b_prefill_ple_tail_sequences = set()" in source
    assert "and int(ple.shape[0]) == 8" in source
    assert "and int(ple.shape[1]) in (521, 2057)" in source
    assert "and int(ple.shape[2]) == 1536" in source
    assert "self._gemma4_e2b_prefill_ple_tail_hits += 1" in source
    assert "rmsnorm_triton_residual_scale_next(" in source
    assert "if not ple_tail_fused:" in source


def test_runtime_policy_promotes_only_long_shape():
    source = (ROOT / "megagemm" / "models" / "runtime_policy.py").read_text(
        encoding="utf-8"
    )
    assert "gemma4_e2b_b8_prefill_ple_tail=True" in source
    assert "gemma4_e2b_b8_prefill_ple_tail_sequences=(2057,)" in source


def _samples(production: float, candidate: list[float]):
    rows = []
    for repeat, current in enumerate(candidate, start=1):
        rows.extend([
            {"case": "production", "prompt_tokens": 512, "repeat": repeat,
             "prefill_ms": production, "wall_ms": production + 20.0,
             "generated_token_digest": "same", "candidate_hits": 0,
             "candidate_disabled_layers": 0, "candidate_failures": []},
            {"case": "ple_tail_fused", "prompt_tokens": 512, "repeat": repeat,
             "prefill_ms": current, "wall_ms": current + 20.0,
             "generated_token_digest": "same", "candidate_hits": 35,
             "candidate_disabled_layers": 0, "candidate_failures": []},
        ])
    return rows


def test_summary_promotes_only_stable_paired_win():
    module = _load_benchmark()
    promoted = module.summarize(
        _samples(600.0, [570.0, 570.0, 570.0]), prompts=[512],
        maximum_spread=1.06, minimum_prefill_speedup=1.01,
        minimum_wall_speedup=1.005,
    )
    assert promoted["decision"] == "PROMOTE_SHAPE_DISPATCH"
    rejected = module.summarize(
        _samples(600.0, [610.0, 610.0, 500.0]), prompts=[512],
        maximum_spread=1.30, minimum_prefill_speedup=1.01,
        minimum_wall_speedup=1.005,
    )
    assert rejected["decision"] == "KEEP_PRODUCTION"


def test_frontier_is_one_load_full_model_with_natural_tokens():
    source = BENCHMARK.read_text(encoding="utf-8")
    assert source.count("matrix.make_runner(") == 1
    assert '"model_loads": 1' in source
    assert '"gemm_paths": "unchanged"' in source
    assert "MEGAGEMM_BENCHMARK_TOKEN_DIGEST" in source
    assert "both ratio-of-medians and paired-median gates" in source


def test_colab_harness_targets_drive_without_mutation_or_native_build():
    source = HARNESS.read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in source
    assert "git pull" not in source
    assert "zip" not in source.lower()
    assert "pip install -e" not in source
    assert "setup.py" not in source
    assert "run_gemma4_e2b_prefill_ple_tail_frontier.py" in source
