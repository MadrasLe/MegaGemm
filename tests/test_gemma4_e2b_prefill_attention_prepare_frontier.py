from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "run_gemma4_e2b_prefill_attention_prepare_frontier.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_e2b_prefill_attention_prepare_frontier_colab.sh"


def _load_benchmark():
    spec = spec_from_file_location("prefill_attention_prepare_frontier", BENCHMARK)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e2b_l4_attention_prepare_shape_guard_is_exact():
    from megagemm.models.llama import _gemma4_e2b_l4_fused_attn_prepare_shape

    common = dict(
        batch_size=8,
        num_q_heads=8,
        num_kv_heads=1,
        rotary_dim=256,
        dtype=torch.bfloat16,
        device_name="NVIDIA L4",
        enabled=True,
    )
    assert _gemma4_e2b_l4_fused_attn_prepare_shape(rows=521, head_dim=256, **common)
    assert _gemma4_e2b_l4_fused_attn_prepare_shape(rows=2057, head_dim=256, **common)
    full = dict(common, rotary_dim=512)
    assert _gemma4_e2b_l4_fused_attn_prepare_shape(rows=521, head_dim=512, **full)
    assert _gemma4_e2b_l4_fused_attn_prepare_shape(rows=2057, head_dim=512, **full)
    assert not _gemma4_e2b_l4_fused_attn_prepare_shape(rows=520, head_dim=256, **common)
    assert not _gemma4_e2b_l4_fused_attn_prepare_shape(
        rows=521, head_dim=256, **dict(common, batch_size=4)
    )
    assert not _gemma4_e2b_l4_fused_attn_prepare_shape(
        rows=521, head_dim=256, **dict(common, enabled=False)
    )
    assert not _gemma4_e2b_l4_fused_attn_prepare_shape(
        rows=521, head_dim=256, **dict(common, device_name="NVIDIA A100")
    )


def test_gate_is_one_load_and_audits_all_kv_sources():
    module = _load_benchmark()
    assert len(module.CASES) == 5
    assert module.CASES[0] == ("production", None)
    source = BENCHMARK.read_text(encoding="utf-8")
    assert source.count("matrix.make_runner(") == 1
    assert "EXPECTED_KV_SOURCE_LAYERS = 15" in source
    assert '"model_loads": 1' in source
    assert "natural-token digest plus exact 15/15 KV-source hit audit" in source


def test_summary_selects_launch_policy_per_prompt_shape():
    module = _load_benchmark()
    samples = []
    winners = {512: "split_h256w2_h512w4", 2048: "split_h256w4_h512w8"}
    for prompt, baseline in ((512, 620.0), (2048, 1820.0)):
        digest = f"digest-{prompt}"
        for case, launch in module.CASES:
            value = baseline
            if case == winners[prompt]:
                value *= 0.95
            elif launch is not None:
                value *= 0.98
            for repeat in range(1, 4):
                samples.append(
                    {
                        "case": case,
                        "prompt_tokens": prompt,
                        "prefill_ms": value,
                        "wall_ms": value + 12.0,
                        "generated_token_digest": digest,
                        "candidate_hits": 0 if launch is None else 15,
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
    assert summary["shape_policy"]["b8/p512"]["winner"] == winners[512]
    assert summary["shape_policy"]["b8/p2048"]["winner"] == winners[2048]


def test_colab_harness_uses_fixed_drive_without_repo_mutation():
    source = HARNESS.read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in source
    assert "git pull" not in source
    assert "zip" not in source.lower()
    assert "vllm" not in source.lower()
    assert "pip install -e" not in source
    assert "run_gemma4_e2b_prefill_attention_prepare_frontier.py" in source
