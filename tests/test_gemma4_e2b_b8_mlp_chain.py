from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_tensorcore_mlp_kernel_is_exactly_guarded_to_e2b_b8_large_shape():
    source = (ROOT / "megagemm" / "kernels" / "deepfusion_mlp.py").read_text(
        encoding="utf-8"
    )
    assert "def gemma4_e2b_b8_geglu_down_tensorcore(" in source
    assert "tuple(gate_up.shape) != (8, 24576)" in source
    assert "tuple(down_weight.shape) != (1536, 12288)" in source
    assert "acc += tl.dot(activated, weight)" in source
    assert "This deliberately has no generic fallback" in source


def test_model_large_mlp_chain_has_counted_fallbacks():
    source = (ROOT / "megagemm" / "models" / "llama.py").read_text(
        encoding="utf-8"
    )
    assert "use_tensorcore_down = bool(" in source
    assert "_gemma4_flat_b8_tensorcore_down_runtime_disabled = True" in source
    assert "_gemma4_flat_b8_tensorcore_down_hits += 1" in source
    assert "use_gated_activation = bool(" in source
    assert "_gemma4_flat_b8_gated_activation_runtime_disabled = True" in source
    assert "_gemma4_flat_b8_gated_activation_hits += 1" in source
    assert "activated = torch.nn.functional.gelu" in source
