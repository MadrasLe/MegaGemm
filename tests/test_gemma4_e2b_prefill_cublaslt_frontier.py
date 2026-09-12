from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "run_gemma4_e2b_prefill_cublaslt_frontier.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_e2b_prefill_cublaslt_frontier_colab.sh"


def _load_benchmark():
    spec = spec_from_file_location("prefill_cublaslt_frontier", BENCHMARK)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_maps_keep_algorithms_shape_specific():
    module = _load_benchmark()
    screen = {}
    for sequence_len in (521, 2057):
        for output_features in (12288, 24576):
            screen[f"s{sequence_len}_n{output_features}"] = {
                "top_algorithms": [
                    {"algorithm_index": 3},
                    {"algorithm_index": 7},
                ]
            }
    maps = module._candidate_maps(screen, 2057)
    assert len(maps) == 4
    assert maps[0] == {(2057, 12288): 3, (2057, 24576): 3}
    assert all(all(key[0] == 2057 for key in row) for row in maps)


def test_model_route_is_exactly_guarded_and_falls_back():
    source = (ROOT / "megagemm" / "models" / "llama.py").read_text(
        encoding="utf-8"
    )
    assert "_gemma4_e2b_prefill_cublaslt_gateup_algorithms" in source
    assert "and int(x.shape[0]) == 8" in source
    assert "and gate_up_sequence_len in (521, 2057)" in source
    assert "and gate_up_out_features in (12288, 24576)" in source
    assert "self._gemma4_e2b_prefill_cublaslt_gateup_hits += 1" in source
    assert "gate_up = _prefill_linear(" in source


def test_focused_extension_can_be_built_without_unrelated_native_modules():
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    binding = (ROOT / "pytorch_binding" / "cublaslt_binding.cpp").read_text(
        encoding="utf-8"
    )
    wrapper = (
        ROOT / "megagemm" / "kernels" / "mlp_prefill_native.py"
    ).read_text(encoding="utf-8")
    assert "MEGAGEMM_BUILD_ONLY_CUBLASLT" in setup
    assert '"megagemm_cublaslt_ops"' in setup
    assert "cublaslt_bf16_linear_cuda" in binding
    assert "_focused_cublaslt_ops" in wrapper


def test_colab_harness_uses_drive_and_ephemeral_build_only():
    source = HARNESS.read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in source
    assert "MEGAGEMM_BUILD_ONLY_CUBLASLT=1" in source
    assert "mktemp -d /tmp/megagemm_cublaslt" in source
    assert "--build-lib \"$BUILD_ROOT/lib\"" in source
    assert "--inplace" not in source
    assert "git pull" not in source
    assert "vllm" not in source.lower()
    assert "zip" not in source.lower()


def test_frontier_loads_one_model_and_uses_full_model_as_promotion_evidence():
    source = BENCHMARK.read_text(encoding="utf-8")
    assert source.count("matrix.make_runner(") == 1
    assert '"model_loads": 1' in source
    assert '"screen_role": "candidate search only"' in source
    assert '"promotion_evidence": "rotated full-model production comparison"' in source
    assert "EXPECTED_DENSE_LAYERS = 35" in source


def test_summary_promotes_only_a_valid_full_model_win():
    module = _load_benchmark()
    algorithms = {(521, 12288): 3, (521, 24576): 7}
    routes = {
        "sliding_layers": 28,
        "full_layers": 7,
        "activation_layers": 35,
        "activation_failure": "",
    }
    samples = []
    for repeat in range(1, 4):
        samples.extend(
            [
                {
                    "case": "production",
                    "prompt_tokens": 512,
                    "repeat": repeat,
                    "algorithms": None,
                    "wall_ms": 660.0,
                    "prefill_ms": 650.0,
                    "generated_token_digest": "same-token",
                    "candidate_hits": 0,
                    "candidate_disabled_layers": 0,
                    "candidate_failures": [],
                    "production_routes": routes,
                },
                {
                    "case": "cublaslt_rank0",
                    "prompt_tokens": 512,
                    "repeat": repeat,
                    "algorithms": {"s521_n12288": 3, "s521_n24576": 7},
                    "wall_ms": 620.0,
                    "prefill_ms": 610.0,
                    "generated_token_digest": "same-token",
                    "candidate_hits": 35,
                    "candidate_disabled_layers": 0,
                    "candidate_failures": [],
                    "production_routes": routes,
                },
            ]
        )
    summary = module._summarize(
        samples,
        cases_by_prompt={
            512: [("production", None), ("cublaslt_rank0", algorithms)]
        },
        maximum_spread=1.06,
        minimum_prefill_speedup=1.01,
        minimum_wall_speedup=1.005,
    )
    assert summary["decision"] == "PROMOTE_SHAPE_DISPATCH"
    assert summary["shape_policy"]["b8/p512"]["algorithms"] == {
        "s521_n12288": 3,
        "s521_n24576": 7,
    }
