from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "run_gemma4_e2b_prefill_dense_bridge_frontier.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_e2b_prefill_dense_bridge_frontier_colab.sh"


def _load_benchmark():
    spec = spec_from_file_location("prefill_dense_bridge_frontier", BENCHMARK)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dense_bridge_shape_guard_is_exact():
    from megagemm.models.llama import _gemma4_e2b_l4_prefill_dense_bridge_shape

    common = dict(
        batch_size=8,
        hidden_dim=1536,
        dtype=torch.bfloat16,
        device_name="NVIDIA L4",
        enabled=True,
    )
    assert _gemma4_e2b_l4_prefill_dense_bridge_shape(seq_len=521, **common)
    assert _gemma4_e2b_l4_prefill_dense_bridge_shape(seq_len=2057, **common)
    assert not _gemma4_e2b_l4_prefill_dense_bridge_shape(
        seq_len=520, **common
    )
    assert not _gemma4_e2b_l4_prefill_dense_bridge_shape(
        seq_len=521, **dict(common, batch_size=4)
    )
    assert not _gemma4_e2b_l4_prefill_dense_bridge_shape(
        seq_len=521, **dict(common, enabled=False)
    )
    assert not _gemma4_e2b_l4_prefill_dense_bridge_shape(
        seq_len=521, **dict(common, device_name="NVIDIA A100")
    )


def test_model_path_fuses_the_exact_dense_chain():
    source = (ROOT / "megagemm" / "models" / "llama.py").read_text(
        encoding="utf-8"
    )
    assert "_gemma4_e2b_prefill_dense_bridge_enabled" in source
    assert "_gemma4_e2b_prefill_dense_bridge_warps_by_sequence" in source
    assert "rmsnorm_triton_attn_residual_dense(" in source
    assert "out_hidden=residual" in source
    assert "pre_ff_out=pre_ff_out" in source
    assert "self._gemma4_e2b_prefill_dense_bridge_hits += 1" in source
    assert "if not bridge_used and not dense_bridge_used:" in source


def test_production_policy_promotes_only_long_context_w4():
    from megagemm.models.runtime_policy import resolve_runtime_policy
    from types import SimpleNamespace

    policy = resolve_runtime_policy(
        SimpleNamespace(
            model_type="gemma4_text",
            num_hidden_layers=35,
            hidden_size=1536,
            num_attention_heads=8,
            num_key_value_heads=1,
        ),
        "NVIDIA L4",
    )
    assert policy.gemma4_e2b_b8_prefill_dense_bridge is True
    assert policy.gemma4_e2b_b8_prefill_dense_bridge_warps == ((2057, 4),)


def test_gate_is_one_model_load_and_three_launch_widths():
    module = _load_benchmark()
    assert module.CASES == (
        ("production", None),
        ("bridge_w2", 2),
        ("bridge_w4", 4),
        ("bridge_w8", 8),
    )
    source = BENCHMARK.read_text(encoding="utf-8")
    assert source.count("matrix.make_runner(") == 1
    assert "EXPECTED_DENSE_LAYERS = 35" in source
    assert "natural-token digest plus exact 35/35 bridge hit audit" in source


def test_summary_selects_warp_count_per_prompt():
    module = _load_benchmark()
    samples = []
    winners = {512: "bridge_w2", 2048: "bridge_w8"}
    for prompt, baseline in ((512, 600.0), (2048, 1700.0)):
        digest = f"digest-{prompt}"
        for case, num_warps in module.CASES:
            value = baseline
            if case == winners[prompt]:
                value *= 0.96
            elif num_warps is not None:
                value *= 0.98
            for repeat in range(1, 4):
                samples.append(
                    {
                        "case": case,
                        "prompt_tokens": prompt,
                        "prefill_ms": value,
                        "wall_ms": value + 15.0,
                        "generated_token_digest": digest,
                        "candidate_hits": 0 if num_warps is None else 35,
                        "candidate_disabled_layers": 0,
                        "candidate_failures": [],
                        "attention_frontend": {
                            "enabled_layers": 15,
                            "hits": 15 * repeat,
                            "failure": "",
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
    assert summary["shape_policy"]["b8/p512"]["winner"] == winners[512]
    assert summary["shape_policy"]["b8/p2048"]["winner"] == winners[2048]


def test_colab_harness_uses_fixed_drive_without_repo_mutation():
    source = HARNESS.read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in source
    assert "git pull" not in source
    assert "zip" not in source.lower()
    assert "vllm" not in source.lower()
    assert "pip install -e" not in source
    assert "run_gemma4_e2b_prefill_dense_bridge_frontier.py" in source
