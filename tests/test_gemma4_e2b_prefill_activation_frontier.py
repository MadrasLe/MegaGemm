from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "run_gemma4_e2b_prefill_activation_frontier.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_e2b_prefill_activation_frontier_colab.sh"


def _load_benchmark():
    spec = spec_from_file_location("prefill_activation_frontier", BENCHMARK)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_path_is_exactly_guarded_and_counted():
    source = (ROOT / "megagemm" / "models" / "llama.py").read_text(encoding="utf-8")
    assert "_gemma4_e2b_prefill_gated_activation_enabled" in source
    assert "and int(gate_up.shape[0]) == 8" in source
    assert "and int(gate_up.shape[1]) in (521, 2057)" in source
    assert "and int(self.intermediate_size) in (6144, 12288)" in source
    assert "self._gemma4_e2b_prefill_gated_activation_hits += 1" in source
    assert '"activation", activation_start_end' in source


def test_gate_is_one_model_load_full_model_matrix():
    module = _load_benchmark()
    assert module.CASES == (
        ("production", None),
        ("fused_b128", 128),
        ("fused_b256", 256),
        ("fused_b512", 512),
        ("fused_b1024", 1024),
    )
    source = BENCHMARK.read_text(encoding="utf-8")
    assert source.count("matrix.make_runner(") == 1
    assert '"model_loads": 1' in source
    assert '"natural-token digest plus exact candidate hit audit"' in source
    assert "EXPECTED_DENSE_LAYERS = 35" in source


def test_shape_policy_can_select_different_blocks():
    module = _load_benchmark()
    samples = []
    for prompt, baseline, winners in (
        (512, 500.0, {128: 480.0, 256: 470.0, 512: 475.0, 1024: 490.0}),
        (2048, 1900.0, {128: 1850.0, 256: 1810.0, 512: 1780.0, 1024: 1820.0}),
    ):
        digest = f"digest-{prompt}"
        for case, block in module.CASES:
            value = baseline if block is None else winners[block]
            for repeat in range(1, 4):
                samples.append(
                    {
                        "case": case,
                        "block_size": block,
                        "prompt_tokens": prompt,
                        "repeat": repeat,
                        "prefill_ms": value,
                        "wall_ms": value + 20.0,
                        "generated_token_digest": digest,
                        "candidate_hits": 0 if block is None else 35,
                        "candidate_disabled_layers": 0,
                        "candidate_failures": [],
                        "production_routes": {
                            "sliding_layers": 28,
                            "full_layers": 7,
                            "full_error": "",
                        },
                    }
                )
    summary = module.summarize(
        samples,
        prompts=[512, 2048],
        maximum_spread=1.06,
        minimum_prefill_speedup=1.01,
        minimum_wall_speedup=1.005,
    )
    assert summary["decision"] == "PROMOTE_SHAPE_DISPATCH"
    assert summary["shape_policy"]["b8/p512"]["block_size"] == 256
    assert summary["shape_policy"]["b8/p2048"]["block_size"] == 512


def test_colab_harness_uses_drive_without_repo_update_or_archive():
    source = HARNESS.read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in source
    assert "git pull" not in source
    assert "zip" not in source.lower()
    assert "pip install -q vllm" not in source.lower()
    assert "run_gemma4_e2b_prefill_activation_frontier.py" in source
