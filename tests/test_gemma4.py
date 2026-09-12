import torch
import copy
import json
from unittest import mock

from megagemm.engine.engine import _normalize_token_id_set
from megagemm.engine.kv_cache import BlockManager
from megagemm.kernels.deepfusion_mlp import deepfusion_swiglu_down
from megagemm.kernels.fused_rmsnorm_linear import fused_rmsnorm_linear
from megagemm.kernels.lm_head_argmax import logits_softcap_argmax
from megagemm.kernels.paged_attention import (
    paged_attention_decode,
    paged_kv_cache_scatter,
)
from megagemm.kernels.rmsnorm_triton import (
    rmsnorm_triton,
    rmsnorm_triton_add,
    rmsnorm_triton_add_dual,
    rmsnorm_triton_attn_residual_dual,
    rmsnorm_triton_attn_residual_router_bridge,
    rmsnorm_triton_dual,
    rmsnorm_triton_no_weight,
    rmsnorm_triton_pair_add_final,
    rmsnorm_triton_pair_add_final_residual,
    rmsnorm_triton_scaled_no_weight,
    rmsnorm_triton_weighted_scaled_no_weight_dual,
)
from megagemm.models import load_from_hf
from megagemm.models.llama import (
    LlamaConfig,
    MegaGemmLlama,
    _gemma4_a100_a4b_fused_attn_moe_bridge_prefill_shape,
    _gemma4_a100_a4b_prefill_graph_shape,
    _gemma4_a100_a4b_fused_router_decode_shape,
)
from megagemm.models.loader import _is_text_backbone_weight, _normalize_hf_weight_key
from megagemm.quantization.w8a16 import Int8Linear


def test_rmsnorm_no_weight_matches_reference():
    torch.manual_seed(5)
    x = torch.randn(3, 7, 32)
    actual = rmsnorm_triton_no_weight(x, 1e-6)
    expected = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_gemma4_a4b_fused_decode_router_policy_is_narrow():
    args = (2816, 128, 8, torch.bfloat16, "NVIDIA A100-SXM4-80GB")
    assert not _gemma4_a100_a4b_fused_router_decode_shape(16, *args)
    with mock.patch(
        "megagemm.models.llama._GEMMA4_FUSED_MOE_ROUTER_DECODE",
        True,
    ):
        assert _gemma4_a100_a4b_fused_router_decode_shape(16, *args)
        assert not _gemma4_a100_a4b_fused_router_decode_shape(33, *args)
        assert not _gemma4_a100_a4b_fused_router_decode_shape(
            16,
            2816,
            128,
            8,
            torch.float16,
            "NVIDIA A100-SXM4-80GB",
        )
        assert not _gemma4_a100_a4b_fused_router_decode_shape(
            16,
            2816,
            128,
            8,
            torch.bfloat16,
            "NVIDIA H100 80GB HBM3",
        )


def test_rmsnorm_writes_to_caller_owned_output():
    torch.manual_seed(51)
    x = torch.randn(3, 17)
    weight = torch.randn(17)
    out = torch.empty_like(x)

    actual = rmsnorm_triton(x, weight, 1e-6, False, out=out)
    expected = (
        x
        * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        * weight
    ).to(x.dtype)

    assert actual.data_ptr() == out.data_ptr()
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_rmsnorm_handles_strided_batched_last_token_on_cuda():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(52)
    hidden = torch.randn(
        16,
        25,
        2816,
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = hidden[:, -1:, :]
    weight = torch.randn(2816, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6

    assert x.reshape(-1, x.shape[-1]).stride(0) == 25 * 2816
    actual = rmsnorm_triton(x, weight, eps, offset=True)
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    expected = (x * torch.rsqrt(variance + eps) * (weight + 1.0)).to(x.dtype)

    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, atol=0.03125, rtol=0.01)


def test_dual_rmsnorm_matches_two_reference_calls():
    torch.manual_seed(6)
    x = torch.randn(2, 3, 32)
    weight_a = torch.randn(32)
    weight_b = torch.randn(32)
    eps = 1e-6

    actual_a, actual_b = rmsnorm_triton_dual(x, weight_a, weight_b, eps)
    inv = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    expected_a = (x * inv * weight_a).to(x.dtype)
    expected_b = (x * inv * weight_b).to(x.dtype)

    assert torch.allclose(actual_a, expected_a, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_b, expected_b, atol=1e-6, rtol=1e-6)


def test_attn_residual_dual_rmsnorm_preserves_staged_bf16_and_aliasing():
    torch.manual_seed(61)
    attn_out = torch.randn(2, 3, 32, dtype=torch.bfloat16)
    residual = torch.randn_like(attn_out)
    post_weight = torch.randn(32, dtype=torch.bfloat16)
    shared_weight = torch.randn(32, dtype=torch.bfloat16)
    expert_weight = torch.randn(32, dtype=torch.bfloat16)
    eps = 1e-6

    post_norm = rmsnorm_triton(attn_out, post_weight, eps, False)
    expected_hidden = (residual + post_norm).to(residual.dtype)
    expected_shared, expected_expert = rmsnorm_triton_dual(
        expected_hidden,
        shared_weight,
        expert_weight,
        eps,
    )
    aliased_residual = residual.clone()
    post_norm_out = torch.empty_like(attn_out)
    shared_out = torch.empty_like(attn_out)
    expert_out = torch.empty_like(attn_out)
    actual_hidden, actual_shared, actual_expert = (
        rmsnorm_triton_attn_residual_dual(
            attn_out,
            aliased_residual,
            post_weight,
            shared_weight,
            expert_weight,
            eps,
            out_hidden=aliased_residual,
            post_norm_out=post_norm_out,
            shared_out=shared_out,
            expert_out=expert_out,
        )
    )

    assert actual_hidden.data_ptr() == aliased_residual.data_ptr()
    assert actual_shared.data_ptr() == shared_out.data_ptr()
    assert actual_expert.data_ptr() == expert_out.data_ptr()
    assert torch.equal(
        post_norm_out,
        rmsnorm_triton(attn_out, post_weight, eps, False),
    )
    assert torch.equal(actual_hidden, expected_hidden)
    assert torch.equal(actual_shared, expected_shared)
    assert torch.equal(actual_expert, expected_expert)


def test_attn_residual_router_bridge_preserves_all_staged_bf16_outputs():
    torch.manual_seed(611)
    attn_out = torch.randn(2, 3, 32, dtype=torch.bfloat16)
    residual = torch.randn_like(attn_out)
    post_weight = torch.randn(32, dtype=torch.bfloat16)
    shared_weight = torch.randn(32, dtype=torch.bfloat16)
    expert_weight = torch.randn(32, dtype=torch.bfloat16)
    router_scale = torch.randn(32, dtype=torch.bfloat16)
    eps = 1e-6
    output_scale = 32 ** -0.5

    expected_hidden, expected_shared, expected_expert = (
        rmsnorm_triton_attn_residual_dual(
            attn_out,
            residual,
            post_weight,
            shared_weight,
            expert_weight,
            eps,
        )
    )
    expected_router = rmsnorm_triton_scaled_no_weight(
        expected_hidden,
        router_scale,
        eps,
        output_scale,
    )
    aliased_residual = residual.clone()
    post_out = torch.empty_like(attn_out)
    shared_out = torch.empty_like(attn_out)
    expert_out = torch.empty_like(attn_out)
    router_out = torch.empty_like(attn_out)
    actual = rmsnorm_triton_attn_residual_router_bridge(
        attn_out,
        aliased_residual,
        post_weight,
        shared_weight,
        expert_weight,
        router_scale,
        eps,
        output_scale,
        out_hidden=aliased_residual,
        post_norm_out=post_out,
        shared_out=shared_out,
        expert_out=expert_out,
        router_out=router_out,
    )

    assert actual[0].data_ptr() == aliased_residual.data_ptr()
    assert actual[1].data_ptr() == shared_out.data_ptr()
    assert actual[2].data_ptr() == expert_out.data_ptr()
    assert actual[3].data_ptr() == router_out.data_ptr()
    for value, expected in zip(
        actual,
        (expected_hidden, expected_shared, expected_expert, expected_router),
    ):
        assert torch.equal(value, expected)


def test_add_dual_rmsnorm_preserves_materialized_bf16_and_aliasing():
    torch.manual_seed(62)
    lhs = torch.randn(2, 3, 32, dtype=torch.bfloat16)
    rhs = torch.randn_like(lhs)
    shared_weight = torch.randn(32, dtype=torch.bfloat16)
    expert_weight = torch.randn(32, dtype=torch.bfloat16)
    eps = 1e-6

    expected_hidden = (lhs + rhs).to(lhs.dtype)
    expected_shared, expected_expert = rmsnorm_triton_dual(
        expected_hidden,
        shared_weight,
        expert_weight,
        eps,
    )
    aliased_lhs = lhs.clone()
    actual_hidden, actual_shared, actual_expert = rmsnorm_triton_add_dual(
        aliased_lhs,
        rhs,
        shared_weight,
        expert_weight,
        eps,
        out_hidden=aliased_lhs,
    )

    assert actual_hidden.data_ptr() == aliased_lhs.data_ptr()
    assert torch.equal(actual_hidden, expected_hidden)
    assert torch.equal(actual_shared, expected_shared)
    assert torch.equal(actual_expert, expected_expert)


def test_gemma4_a4b_attn_moe_bridge_policy_is_narrow():
    args = (
        "gemma4_text",
        400,
        2816,
        2112,
        704,
        torch.bfloat16,
        "NVIDIA A100-SXM4-80GB",
    )
    assert _gemma4_a100_a4b_fused_attn_moe_bridge_prefill_shape(*args)
    assert not _gemma4_a100_a4b_fused_attn_moe_bridge_prefill_shape(
        "gemma4_text",
        200,
        2816,
        2112,
        704,
        torch.bfloat16,
        "NVIDIA A100-SXM4-80GB",
    )
    assert not _gemma4_a100_a4b_fused_attn_moe_bridge_prefill_shape(
        "gemma4_text",
        400,
        2816,
        2112,
        704,
        torch.float16,
        "NVIDIA A100-SXM4-80GB",
    )


def test_add_rmsnorm_preserves_staged_input_addition():
    torch.manual_seed(7)
    lhs = torch.randn(2, 3, 32, dtype=torch.bfloat16)
    rhs = torch.randn_like(lhs)
    weight = torch.randn(32, dtype=torch.bfloat16)
    eps = 1e-6

    actual = rmsnorm_triton_add(lhs, rhs, weight, eps)
    summed = (lhs + rhs).to(lhs.dtype)
    inv = torch.rsqrt(summed.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    expected = (summed * inv * weight).to(lhs.dtype)

    assert torch.equal(actual, expected)


def test_pair_add_final_rmsnorm_preserves_staged_bf16_semantics():
    torch.manual_seed(8)
    shared = torch.randn(2, 3, 32, dtype=torch.bfloat16)
    expert = torch.randn_like(shared)
    shared_weight = torch.randn(32, dtype=torch.bfloat16)
    expert_weight = torch.randn(32, dtype=torch.bfloat16)
    final_weight = torch.randn(32, dtype=torch.bfloat16)
    eps = 1e-6

    actual = rmsnorm_triton_pair_add_final(
        shared,
        expert,
        shared_weight,
        expert_weight,
        final_weight,
        eps,
    )
    shared_norm = rmsnorm_triton(shared, shared_weight, eps, False)
    expert_norm = rmsnorm_triton(expert, expert_weight, eps, False)
    expected = rmsnorm_triton_add(shared_norm, expert_norm, final_weight, eps)

    assert torch.equal(actual, expected)


def test_pair_add_final_residual_preserves_staged_bf16_semantics_and_aliasing():
    torch.manual_seed(81)
    shared = torch.randn(2, 3, 32, dtype=torch.bfloat16)
    expert = torch.randn_like(shared)
    residual = torch.randn_like(shared)
    shared_weight = torch.randn(32, dtype=torch.bfloat16)
    expert_weight = torch.randn(32, dtype=torch.bfloat16)
    final_weight = torch.randn(32, dtype=torch.bfloat16)
    eps = 1e-6

    final_norm = rmsnorm_triton_pair_add_final(
        shared,
        expert,
        shared_weight,
        expert_weight,
        final_weight,
        eps,
    )
    expected = (residual + final_norm).to(residual.dtype)
    aliased_residual = residual.clone()
    actual = rmsnorm_triton_pair_add_final_residual(
        shared,
        expert,
        shared_weight,
        expert_weight,
        final_weight,
        aliased_residual,
        eps,
        out=aliased_residual,
    )

    assert actual.data_ptr() == aliased_residual.data_ptr()
    assert torch.equal(actual, expected)


def test_weighted_scaled_dual_rmsnorm_preserves_bf16_router_semantics():
    torch.manual_seed(82)
    x = torch.randn(2, 3, 32, dtype=torch.bfloat16)
    weight = torch.randn(32, dtype=torch.bfloat16)
    scale = torch.randn(32, dtype=torch.bfloat16)
    weighted_out = torch.empty_like(x)
    scaled_out = torch.empty_like(x)
    eps = 1e-6
    output_scale = 32 ** -0.5

    expected_weighted = rmsnorm_triton(x, weight, eps, False)
    expected_scaled = rmsnorm_triton_scaled_no_weight(
        x,
        scale,
        eps,
        output_scale,
    )
    actual_weighted, actual_scaled = (
        rmsnorm_triton_weighted_scaled_no_weight_dual(
            x,
            weight,
            scale,
            eps,
            output_scale,
            weighted_out=weighted_out,
            scaled_out=scaled_out,
        )
    )

    assert actual_weighted.data_ptr() == weighted_out.data_ptr()
    assert actual_scaled.data_ptr() == scaled_out.data_ptr()
    assert torch.equal(actual_weighted, expected_weighted)
    assert torch.equal(actual_scaled, expected_scaled)


def _tiny_gemma4_config_dict(**text_overrides):
    text_config = {
        "model_type": "gemma4_text",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "global_head_dim": 16,
        "num_global_key_value_heads": 2,
        "vocab_size": 128,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "hidden_activation": "gelu_pytorch_tanh",
        "layer_types": [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        "sliding_window": 3,
        "num_kv_shared_layers": 1,
        "use_double_wide_mlp": True,
        "hidden_size_per_layer_input": 4,
        "vocab_size_per_layer_input": 128,
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
            },
        },
    }
    text_config.update(text_overrides)
    return {
        "model_type": "gemma4",
        "vision_config": {"model_type": "gemma4_vision"},
        "audio_config": {"model_type": "gemma4_audio"},
        "text_config": text_config,
    }


def _kv_sources(config):
    return {
        layer_idx: source_idx
        for layer_idx, source_idx in enumerate(config.kv_share_sources)
        if source_idx is not None
    }


def _megagemm_gemma4_to_hf_state(
    model: MegaGemmLlama,
    *,
    legacy_moe_norm_names: bool = False,
) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    hf_state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": state["embed_tokens.weight"].clone(),
        "model.norm.weight": state["norm.weight"].clone(),
        "lm_head.weight": state["lm_head.weight"].clone(),
    }
    for local_name, hf_name in (
        ("embed_tokens_per_layer.weight", "model.embed_tokens_per_layer.weight"),
        ("per_layer_model_projection.weight", "model.per_layer_model_projection.weight"),
        ("per_layer_projection_norm.weight", "model.per_layer_projection_norm.weight"),
    ):
        if local_name in state:
            hf_state[hf_name] = state[local_name].clone()

    for layer_idx in range(model.config.num_hidden_layers):
        prefix = f"layers.{layer_idx}"
        hf_prefix = f"model.layers.{layer_idx}"

        hf_state[f"{hf_prefix}.layer_scalar"] = state[f"{prefix}.layer_scalar"].clone()
        hf_state[f"{hf_prefix}.input_layernorm.weight"] = state[f"{prefix}.input_layernorm.weight"].clone()
        hf_state[f"{hf_prefix}.post_attention_layernorm.weight"] = state[
            f"{prefix}.post_attention_layernorm.weight"
        ].clone()
        mlp = model.layers[layer_idx].mlp
        is_moe = getattr(mlp, "is_moe", False) and getattr(mlp, "shared_mlp", None) is not None
        if is_moe:
            for norm_name in (
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ):
                hf_state[f"{hf_prefix}.{norm_name}.weight"] = state[
                    f"{prefix}.{norm_name}.weight"
                ].clone()
            if legacy_moe_norm_names:
                pass
            else:
                for norm_name in (
                    "pre_feedforward_layernorm_2",
                    "post_feedforward_layernorm_1",
                    "post_feedforward_layernorm_2",
                ):
                    hf_state[f"{hf_prefix}.{norm_name}.weight"] = state[
                        f"{prefix}.{norm_name}.weight"
                    ].clone()
        else:
            hf_state[f"{hf_prefix}.pre_feedforward_layernorm.weight"] = state[
                f"{prefix}.pre_feedforward_layernorm.weight"
            ].clone()
            hf_state[f"{hf_prefix}.post_feedforward_layernorm.weight"] = state[
                f"{prefix}.post_feedforward_layernorm.weight"
            ].clone()
        for local_name, hf_name in (
            ("per_layer_input_gate.weight", "per_layer_input_gate.weight"),
            ("per_layer_projection.weight", "per_layer_projection.weight"),
            ("post_per_layer_input_norm.weight", "post_per_layer_input_norm.weight"),
        ):
            key = f"{prefix}.{local_name}"
            if key in state:
                hf_state[f"{hf_prefix}.{hf_name}"] = state[key].clone()

        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            key = f"{prefix}.self_attn.{proj_name}.weight"
            if key in state:
                hf_state[f"{hf_prefix}.self_attn.{proj_name}.weight"] = state[key].clone()
        for norm_name in ("q_norm", "k_norm"):
            key = f"{prefix}.self_attn.{norm_name}.weight"
            if key in state:
                hf_state[f"{hf_prefix}.self_attn.{norm_name}.weight"] = state[key].clone()

        dense_prefix = f"{prefix}.mlp.shared_mlp" if is_moe else f"{prefix}.mlp"
        gate_up = state[f"{dense_prefix}.gate_up_proj.weight"]
        half = gate_up.shape[0] // 2
        hf_state[f"{hf_prefix}.mlp.gate_proj.weight"] = gate_up[:half].clone()
        hf_state[f"{hf_prefix}.mlp.up_proj.weight"] = gate_up[half:].clone()
        hf_state[f"{hf_prefix}.mlp.down_proj.weight"] = state[f"{dense_prefix}.down_proj.weight"].clone()
        if is_moe:
            hf_state[f"{hf_prefix}.router.proj.weight"] = state[
                f"{prefix}.mlp.gate.proj.weight"
            ].clone()
            hf_state[f"{hf_prefix}.router.scale"] = state[f"{prefix}.mlp.gate.scale"].clone()
            hf_state[f"{hf_prefix}.router.per_expert_scale"] = state[
                f"{prefix}.mlp.gate.per_expert_scale"
            ].clone()
            hf_state[f"{hf_prefix}.experts.gate_up_proj"] = state[
                f"{prefix}.mlp.experts.gate_up_proj"
            ].clone()
            hf_state[f"{hf_prefix}.experts.down_proj"] = state[
                f"{prefix}.mlp.experts.down_proj"
            ].clone()

    return hf_state


def _build_local_gemma4_snapshot(root, hf_config=None, *, legacy_moe_norm_names: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    hf_config = hf_config or _tiny_gemma4_config_dict()
    config = LlamaConfig.from_dict(hf_config)
    model = MegaGemmLlama(config).eval()
    if config.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight
    hf_state = _megagemm_gemma4_to_hf_state(
        model,
        legacy_moe_norm_names=legacy_moe_norm_names,
    )
    if config.enable_moe_block:
        for layer_idx in range(config.num_hidden_layers):
            if config.is_moe_layer(layer_idx):
                hf_prefix = f"model.layers.{layer_idx}"
                hidden_scale = torch.ones(config.hidden_size, dtype=torch.get_default_dtype())
                expert_scale = torch.ones(config.num_experts, dtype=torch.get_default_dtype())
                hf_state[f"{hf_prefix}.router.scale"] = hidden_scale.clone()
                hf_state[f"{hf_prefix}.router.per_expert_scale"] = expert_scale.clone()

    with (root / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(hf_config, fh)
    with (root / "tokenizer_config.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "chat_template": "<bos>{{ messages[0]['content'] }}<eos>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
            },
            fh,
        )
    with (root / "tokenizer.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "version": "1.0",
                "model": {
                    "type": "WordLevel",
                    "unk_token": "<unk>",
                    "vocab": {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "a": 4},
                },
            },
            fh,
        )
    with (root / "special_tokens_map.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
            },
            fh,
        )

    from safetensors.torch import save_file

    save_file(hf_state, str(root / "model.safetensors"))


def test_gemma4_config_unwraps_text_config_and_builds_layer_layout():
    config = LlamaConfig.from_dict(_tiny_gemma4_config_dict())

    assert config.model_type == "gemma4_text"
    assert config.layer_types == [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ]
    assert config.per_layer_head_dims == [8, 16, 8, 16]
    assert config.per_layer_rotary_dims == [8, 16, 8, 16]
    assert config.per_layer_rope_thetas == [10000.0, 1000000.0, 10000.0, 1000000.0]
    assert config.gemma4_full_rope_partial_factor == 0.25
    assert config.kv_cache_layer_indices == [0, 1, 2]
    assert config.kv_share_sources == [None, None, None, 1]
    assert config.mlp_intermediate_sizes == [64, 64, 64, 128]


def test_gemma4_embedding_scale_is_rounded_to_model_dtype():
    config = LlamaConfig.from_dict(_tiny_gemma4_config_dict())
    model = MegaGemmLlama(config).to(dtype=torch.bfloat16)
    hidden = torch.linspace(-3.0, 3.0, 1001, dtype=torch.bfloat16)

    actual = model._scale_token_embeddings(hidden.clone())
    official_scale = torch.tensor(config.embed_scale, dtype=torch.bfloat16)
    expected = hidden * official_scale
    old_float_scalar_behavior = hidden * config.embed_scale

    assert model.gemma4_embed_scale.dtype == torch.bfloat16
    assert "gemma4_embed_scale" not in model.state_dict()
    assert torch.equal(actual, expected)
    assert not torch.equal(old_float_scalar_behavior, expected)


def test_gemma4_a100_a4b_tuned_policy_is_exact_shape_only():
    import megagemm.models.llama as llama_module
    from megagemm.models.llama import (
        _gemma4_a100_a4b_fused_router_prefill_shape,
        _gemma4_a100_a4b_parallel_moe_prefill_shape,
        _gemma4_a100_a4b_parallel_moe_shape,
        _gemma4_a100_a4b_tuned_lm_head_shape,
        _gemma4_a100_a4b_tuned_mlp_shape,
    )

    assert _gemma4_a100_a4b_tuned_mlp_shape(
        1, 2816, 2112, torch.bfloat16, "NVIDIA A100-SXM4-80GB"
    )
    assert not _gemma4_a100_a4b_tuned_mlp_shape(
        2, 2816, 2112, torch.bfloat16, "NVIDIA A100-SXM4-80GB"
    )
    assert not _gemma4_a100_a4b_tuned_mlp_shape(
        1, 2816, 2112, torch.float16, "NVIDIA A100-SXM4-80GB"
    )
    assert not _gemma4_a100_a4b_tuned_mlp_shape(
        1, 2816, 2112, torch.bfloat16, "NVIDIA H100 80GB HBM3"
    )
    assert _gemma4_a100_a4b_tuned_lm_head_shape(
        "gemma4_text",
        1,
        2816,
        262144,
        torch.bfloat16,
        "NVIDIA A100-SXM4-80GB",
    )
    assert not _gemma4_a100_a4b_tuned_lm_head_shape(
        "gemma4_text",
        2,
        2816,
        262144,
        torch.bfloat16,
        "NVIDIA A100-SXM4-80GB",
    )
    assert _gemma4_a100_a4b_parallel_moe_shape(
        "gemma4_text", 1, 2816, 2112, 704, torch.bfloat16,
        "NVIDIA A100-SXM4-80GB",
    )
    assert _gemma4_a100_a4b_parallel_moe_shape(
        "gemma4_text", 8, 2816, 2112, 704, torch.bfloat16,
        "NVIDIA A100-SXM4-80GB",
    )
    assert _gemma4_a100_a4b_parallel_moe_shape(
        "gemma4_text", 16, 2816, 2112, 704, torch.bfloat16,
        "NVIDIA A100-SXM4-80GB",
    )
    assert not _gemma4_a100_a4b_parallel_moe_shape(
        "gemma4_text", 2, 2816, 2112, 704, torch.bfloat16,
        "NVIDIA A100-SXM4-80GB",
    )
    assert not _gemma4_a100_a4b_parallel_moe_shape(
        "gemma4_text", 1, 2816, 2112, 704, torch.bfloat16,
        "NVIDIA H100 80GB HBM3",
    )
    assert _gemma4_a100_a4b_parallel_moe_prefill_shape(
        "gemma4_text", 400, 2816, 2112, 704, torch.bfloat16,
        "NVIDIA A100-SXM4-80GB",
    )
    assert not _gemma4_a100_a4b_parallel_moe_prefill_shape(
        "gemma4_text", 399, 2816, 2112, 704, torch.bfloat16,
        "NVIDIA A100-SXM4-80GB",
    )
    assert not _gemma4_a100_a4b_parallel_moe_prefill_shape(
        "gemma4_text", 400, 2816, 2112, 704, torch.float16,
        "NVIDIA A100-SXM4-80GB",
    )
    assert not _gemma4_a100_a4b_parallel_moe_prefill_shape(
        "gemma4_text", 400, 2816, 2112, 704, torch.bfloat16,
        "NVIDIA H100 80GB HBM3",
    )
    original_router_enabled = llama_module._GEMMA4_FUSED_MOE_ROUTER_PREFILL
    llama_module._GEMMA4_FUSED_MOE_ROUTER_PREFILL = True
    try:
        assert _gemma4_a100_a4b_fused_router_prefill_shape(
            25, 2816, 128, 8, torch.bfloat16, "NVIDIA A100-SXM4-80GB"
        )
        assert _gemma4_a100_a4b_fused_router_prefill_shape(
            400, 2816, 128, 8, torch.bfloat16, "NVIDIA A100-SXM4-80GB"
        )
        assert not _gemma4_a100_a4b_fused_router_prefill_shape(
            24, 2816, 128, 8, torch.bfloat16, "NVIDIA A100-SXM4-80GB"
        )
        assert not _gemma4_a100_a4b_fused_router_prefill_shape(
            25, 2816, 128, 8, torch.bfloat16, "NVIDIA H100 80GB HBM3"
        )
    finally:
        llama_module._GEMMA4_FUSED_MOE_ROUTER_PREFILL = original_router_enabled


def test_gemma4_a4b_experts_use_measured_segmented_prefill_config():
    from megagemm.models.llama import (
        _GEMMA4_A4B_SEGMENTED_PREFILL_LARGE_OPTIONS,
        _GEMMA4_A4B_LONG_PADDED_BMM_PREFILL,
        _GEMMA4_A4B_LONG_PADDED_BMM_MAX_PADDING_RATIO,
        _GEMMA4_A4B_SEGMENTED_PREFILL_LONG_OPTIONS,
        _GEMMA4_A4B_SEGMENTED_PREFILL_LONG_MAX_ROWS,
        _GEMMA4_A4B_SEGMENTED_PREFILL_LONG_ROWS,
        _GEMMA4_A4B_SEGMENTED_PREFILL_LARGE_ROWS_MIN,
        _GEMMA4_A4B_SEGMENTED_PREFILL_OPTIONS,
        _GEMMA4_A4B_SEGMENTED_PREFILL_SHORT_OPTIONS,
        _GEMMA4_A4B_SEGMENTED_PREFILL_SHORT_ROWS_MAX,
        _gemma4_a100_a4b_segmented_prefill_long_shape,
        _gemma4_a4b_segmented_prefill_shape,
        Qwen3MoeExperts,
    )

    assert _gemma4_a4b_segmented_prefill_shape(
        "gemma4_text", 128, 2816, 704, 8
    )
    assert not _gemma4_a4b_segmented_prefill_shape(
        "qwen3_moe", 128, 2816, 704, 8
    )
    assert _GEMMA4_A4B_SEGMENTED_PREFILL_OPTIONS == {
        "force": True,
        "block_m": 16,
        "block_n": 128,
        "block_k": 64,
        "fused_gate_block_n": 64,
        "num_warps": 4,
        "num_stages": 3,
        "fused_gate": True,
        "dense_grid": False,
        "route_scatter": True,
        "async_tiles_max_assignments": 4096,
        "compact_route_pack": True,
        "single_accumulator": False,
        "group_size_m": 8,
    }
    assert _GEMMA4_A4B_SEGMENTED_PREFILL_LARGE_ROWS_MIN == 400
    assert _GEMMA4_A4B_SEGMENTED_PREFILL_LARGE_OPTIONS == {
        "block_m": 32,
    }
    assert _GEMMA4_A4B_SEGMENTED_PREFILL_LONG_ROWS == 16_384
    assert _GEMMA4_A4B_SEGMENTED_PREFILL_LONG_MAX_ROWS == 32_768
    # The global padded-BMM route was retired after the segmented candidate
    # proved to be the stable long-prefill path.
    assert not _GEMMA4_A4B_LONG_PADDED_BMM_PREFILL
    assert _GEMMA4_A4B_LONG_PADDED_BMM_MAX_PADDING_RATIO == 2.0
    assert _GEMMA4_A4B_SEGMENTED_PREFILL_LONG_OPTIONS == {
        "block_m": 64,
        "block_n": 256,
        "block_k": 64,
        "fused_gate_block_n": 128,
        "num_warps": 4,
        "num_stages": 3,
        "compact_route_pack": False,
        "async_tiles_max_assignments": 4096,
        "sorted_partial": True,
    }
    assert _gemma4_a100_a4b_segmented_prefill_long_shape(
        16_384,
        torch.bfloat16,
        "NVIDIA A100-SXM4-80GB",
    )
    assert _gemma4_a100_a4b_segmented_prefill_long_shape(
        32_768,
        torch.bfloat16,
        "NVIDIA A100-SXM4-80GB",
    )
    assert not _gemma4_a100_a4b_segmented_prefill_long_shape(
        16_384,
        torch.bfloat16,
        "NVIDIA H100 80GB HBM3",
    )
    assert _GEMMA4_A4B_SEGMENTED_PREFILL_SHORT_ROWS_MAX == 32
    assert _GEMMA4_A4B_SEGMENTED_PREFILL_SHORT_OPTIONS == {
        "block_m": 16,
        "block_n": 64,
        "block_k": 128,
        "num_warps": 8,
        "num_stages": 4,
        "fixed_route_pack": True,
    }

    class FakeExperts:
        _gemma4_a4b_segmented_prefill = True

    short_options = Qwen3MoeExperts._segmented_prefill_kernel_options(
        FakeExperts(), 25
    )
    medium_options = Qwen3MoeExperts._segmented_prefill_kernel_options(
        FakeExperts(), 100
    )
    large_options = Qwen3MoeExperts._segmented_prefill_kernel_options(
        FakeExperts(), 400
    )
    long_options = Qwen3MoeExperts._segmented_prefill_kernel_options(
        FakeExperts(),
        16_384,
        dtype=torch.bfloat16,
        device_name="NVIDIA A100-SXM4-80GB",
    )
    non_a100_long_options = Qwen3MoeExperts._segmented_prefill_kernel_options(
        FakeExperts(),
        16_384,
        dtype=torch.bfloat16,
        device_name="NVIDIA H100 80GB HBM3",
    )
    for key, value in _GEMMA4_A4B_SEGMENTED_PREFILL_SHORT_OPTIONS.items():
        assert short_options[key] == value
    for key in ("block_m", "block_n", "block_k", "num_warps", "num_stages"):
        assert medium_options[key] == _GEMMA4_A4B_SEGMENTED_PREFILL_OPTIONS[key]
    assert large_options["block_m"] == 32
    assert long_options["block_m"] == 64
    assert long_options["block_n"] == 256
    assert long_options["block_k"] == 64
    assert long_options["fused_gate_block_n"] == 128
    assert long_options["num_warps"] == 4
    assert long_options["num_stages"] == 3
    assert long_options["compact_route_pack"] is False
    assert non_a100_long_options["block_m"] == 32
    assert non_a100_long_options["compact_route_pack"] is True


def test_gemma4_long_padded_bmm_runtime_policy_is_exact():
    from megagemm.models import llama as llama_module
    from megagemm.models.llama import Qwen3MoeExperts

    class FakeTensor:
        def __init__(self, shape, dtype=torch.bfloat16):
            self.shape = shape
            self.dtype = dtype
            self.is_cuda = True
            self.device = torch.device("cuda:0")

    class FakeExperts:
        _gemma4_a4b_segmented_prefill = True
        _gemma4_long_padded_bmm_prefill_disabled = False
        hidden_dim = 2816
        gate_up_proj = FakeTensor((128, 1408, 2816))
        down_proj = FakeTensor((128, 2816, 704))

    hidden = FakeTensor((16_384, 2816))
    selected = FakeTensor((16_384, 8), torch.int64)
    routing = FakeTensor((16_384, 8))
    experts = FakeExperts()

    with (
        mock.patch.object(
            llama_module,
            "_GEMMA4_A4B_LONG_PADDED_BMM_PREFILL",
            True,
        ),
        mock.patch.object(torch, "is_grad_enabled", return_value=False),
        mock.patch.object(
            torch.cuda,
            "get_device_name",
            return_value="NVIDIA A100-SXM4-80GB",
        ),
    ):
        eligible = Qwen3MoeExperts._gemma4_long_padded_bmm_prefill_is_enabled
        assert eligible(experts, hidden, selected, routing)
        assert not eligible(
            experts,
            hidden,
            selected,
            routing,
            graph_safe_prefill=True,
        )
        hidden.shape = (16_383, 2816)
        assert not eligible(experts, hidden, selected, routing)


def test_gemma4_long_dominant_expert_runtime_policy_is_exact():
    from megagemm.models import llama as llama_module
    from megagemm.models.llama import Qwen3MoeExperts

    class FakeTensor:
        def __init__(self, shape, dtype=torch.bfloat16):
            self.shape = shape
            self.dtype = dtype
            self.is_cuda = True
            self.device = torch.device("cuda:0")

    class FakeExperts:
        _gemma4_a4b_segmented_prefill = True
        _gemma4_long_dominant_expert_prefill_disabled = False
        hidden_dim = 2816
        gate_up_proj = FakeTensor((128, 1408, 2816))
        down_proj = FakeTensor((128, 2816, 704))

    hidden = FakeTensor((32_768, 2816))
    selected = FakeTensor((32_768, 8), torch.int64)
    routing = FakeTensor((32_768, 8))
    experts = FakeExperts()

    with (
        mock.patch.object(
            llama_module,
            "_GEMMA4_A4B_LONG_DOMINANT_EXPERT_PREFILL",
            True,
        ),
        mock.patch.object(torch, "is_grad_enabled", return_value=False),
        mock.patch.object(
            torch.cuda,
            "get_device_name",
            return_value="NVIDIA A100-SXM4-80GB",
        ),
    ):
        eligible = (
            Qwen3MoeExperts._gemma4_long_dominant_expert_prefill_is_enabled
        )
        assert eligible(experts, hidden, selected, routing)
        assert not eligible(
            experts,
            hidden,
            selected,
            routing,
            graph_safe_prefill=True,
        )
        hidden.shape = (16_384, 2816)
        assert not eligible(experts, hidden, selected, routing)


def test_gemma4_long_dominant_expert_guard_falls_back_without_disabling():
    from megagemm.models.llama import Qwen3MoeExperts

    class FakeExperts:
        _gemma4_long_dominant_expert_guard_rejection = staticmethod(
            Qwen3MoeExperts._gemma4_long_dominant_expert_guard_rejection
        )
        _gemma4_long_dominant_expert_prefill_disabled = False
        _gemma4_long_dominant_expert_prefill_fail_reason = ""
        _gemma4_long_dominant_expert_prefill_guard_misses = 0
        _gemma4_long_dominant_expert_prefill_last_guard_reason = ""
        _gemma4_long_dominant_expert_prefill_last_active = True
        _gemma4_long_dominant_expert_prefill_workspace = {"stale": 1}

    experts = FakeExperts()
    record = Qwen3MoeExperts._record_gemma4_long_dominant_expert_failure
    guarded = record(
        experts,
        RuntimeError(
            "Dominant-expert guard: hottest expert is only 4.000x the mean"
        ),
    )
    assert guarded
    assert not experts._gemma4_long_dominant_expert_prefill_disabled
    assert experts._gemma4_long_dominant_expert_prefill_guard_misses == 1
    assert experts._gemma4_long_dominant_expert_prefill_workspace == {}

    guarded = record(experts, RuntimeError("unexpected kernel failure"))
    assert not guarded
    assert experts._gemma4_long_dominant_expert_prefill_disabled
    assert experts._gemma4_long_dominant_expert_prefill_fail_reason == (
        "unexpected kernel failure"
    )


def test_gemma4_a4b_fused_qkv_prefill_policy_is_exact():
    from megagemm.models.llama import _gemma4_a100_a4b_fused_qkv_prefill_shape

    device = "NVIDIA A100-SXM4-80GB"
    assert _gemma4_a100_a4b_fused_qkv_prefill_shape(
        25, 2816, 4096, 2048, 2048, torch.bfloat16, device
    )
    assert _gemma4_a100_a4b_fused_qkv_prefill_shape(
        25, 2816, 8192, 1024, 1024, torch.bfloat16, device
    )
    assert _gemma4_a100_a4b_fused_qkv_prefill_shape(
        200, 2816, 4096, 2048, 2048, torch.bfloat16, device
    )
    assert _gemma4_a100_a4b_fused_qkv_prefill_shape(
        400, 2816, 8192, 1024, 1024, torch.bfloat16, device
    )
    assert _gemma4_a100_a4b_fused_qkv_prefill_shape(
        16_384, 2816, 4096, 2048, 2048, torch.bfloat16, device
    )
    assert _gemma4_a100_a4b_fused_qkv_prefill_shape(
        32_768, 2816, 4096, 2048, 2048, torch.bfloat16, device
    )
    assert not _gemma4_a100_a4b_fused_qkv_prefill_shape(
        32_768, 2816, 8192, 1024, 1024, torch.bfloat16, device
    )
    assert not _gemma4_a100_a4b_fused_qkv_prefill_shape(
        16_384, 2816, 8192, 1024, 1024, torch.bfloat16, device
    )
    assert not _gemma4_a100_a4b_fused_qkv_prefill_shape(
        33, 2816, 4096, 2048, 2048, torch.bfloat16, device
    )
    assert not _gemma4_a100_a4b_fused_qkv_prefill_shape(
        100, 2816, 4096, 2048, 2048, torch.bfloat16, device
    )
    assert not _gemma4_a100_a4b_fused_qkv_prefill_shape(
        25, 2816, 4096, 2048, 2048, torch.float16, device
    )
    assert not _gemma4_a100_a4b_fused_qkv_prefill_shape(
        25, 2816, 4096, 2048, 2048, torch.bfloat16, "NVIDIA H100 80GB HBM3"
    )


def test_gemma4_k_equals_v_builds_compact_fused_qk_weight():
    config = LlamaConfig.from_dict(
        _tiny_gemma4_config_dict(attention_k_eq_v=True)
    )
    model = MegaGemmLlama(config).float().eval()
    attn = model.layers[1].self_attn

    assert attn.attention_k_eq_v
    assert not attn.is_kv_shared
    assert attn.k_proj is not None
    assert attn.v_proj is None

    fused_weight, fused_bias = attn._gemma4_fused_qkv_weight_bias()
    expected = torch.cat([attn.q_proj.weight, attn.k_proj.weight], dim=0)
    assert fused_bias is None
    assert fused_weight.shape == expected.shape
    assert torch.equal(fused_weight, expected)


def test_gemma4_router_topk_applies_expert_scale_in_topk_path():
    from megagemm.kernels.qwen3_moe import qwen3_moe_topk_softmax

    torch.manual_seed(19)
    logits = torch.randn(5, 16, dtype=torch.float32)
    expert_scale = torch.linspace(0.5, 1.5, 16, dtype=torch.float32)
    actual_weights, actual_experts = qwen3_moe_topk_softmax(
        logits,
        4,
        expert_scale=expert_scale,
    )

    top_logits, expected_experts = torch.topk(logits, 4, dim=-1)
    expected_weights = torch.softmax(top_logits, dim=-1)
    expected_weights = expected_weights * expert_scale[expected_experts]
    assert torch.equal(actual_experts, expected_experts)
    assert torch.allclose(actual_weights, expected_weights, atol=1e-7, rtol=1e-7)


def test_gemma4_router_scaled_rmsnorm_preserves_staged_rounding():
    from megagemm.kernels.rmsnorm_triton import rmsnorm_triton_scaled_no_weight

    torch.manual_seed(23)
    x = torch.randn(3, 32, dtype=torch.bfloat16)
    scale = torch.randn(32, dtype=torch.bfloat16)
    eps = 1e-6
    scalar_root = 32 ** -0.5
    normalized = (
        x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    ).to(x.dtype)
    expected = normalized.mul(scale).mul(scalar_root)
    actual = rmsnorm_triton_scaled_no_weight(x, scale, eps, scalar_root)
    assert torch.equal(actual, expected)


def test_gemma4_router_keeps_decode_and_prefill_workspaces_isolated():
    config = LlamaConfig.from_dict(
        _tiny_gemma4_config_dict(
            enable_moe_block=True,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=8,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            use_double_wide_mlp=False,
            attention_k_eq_v=True,
        )
    )
    router = MegaGemmLlama(config).float().eval().layers[0].mlp.gate
    decode_hidden = torch.randn(4, config.hidden_size)
    prefill_hidden = torch.randn(25, config.hidden_size)

    decode_logits = router._logits_buffer(decode_hidden, is_prefill=False)
    decode_logits.fill_(3.0)
    decode_ptr = decode_logits.data_ptr()
    decode_workspace = router._topk_workspace_for(decode_hidden, is_prefill=False)
    other_decode_logits = router._logits_buffer(
        torch.randn(8, config.hidden_size),
        is_prefill=False,
    )
    other_decode_workspace = router._topk_workspace_for(
        torch.randn(8, config.hidden_size),
        is_prefill=False,
    )
    prefill_logits = router._logits_buffer(prefill_hidden, is_prefill=True)
    prefill_logits.zero_()
    prefill_workspace = router._topk_workspace_for(prefill_hidden, is_prefill=True)
    router._logits_buffer(torch.randn(50, config.hidden_size), is_prefill=True)

    assert router._decode_topk_workspaces is not router._prefill_topk_workspaces
    assert router._decode_router_logits_by_rows is not router._prefill_router_logits_by_rows
    assert router._decode_topk_workspaces[4] is decode_workspace
    assert router._decode_topk_workspaces[8] is other_decode_workspace
    assert decode_workspace is not other_decode_workspace
    assert decode_workspace is not prefill_workspace
    assert router._decode_router_logits_by_rows[4] is decode_logits
    assert router._decode_router_logits_by_rows[4].data_ptr() == decode_ptr
    assert other_decode_logits is router._decode_router_logits_by_rows[8]
    assert decode_logits is not prefill_logits
    assert torch.equal(decode_logits, torch.full_like(decode_logits, 3.0))

    assert router._fused_prefill_runtime_by_rows == {}
    router.set_fused_prefill_runtime(400, True)
    router.set_fused_prefill_runtime(200, False)
    assert router._fused_prefill_runtime_by_rows == {400: True, 200: False}

    _, decode_weights, decode_experts = router(decode_hidden, is_prefill=False)
    _, prefill_weights, prefill_experts = router(decode_hidden, is_prefill=True)
    assert torch.equal(decode_experts, prefill_experts)
    assert torch.allclose(decode_weights, prefill_weights, atol=1e-7, rtol=1e-7)


def test_gemma4_a4b_fused_attention_prepare_policy_is_exact():
    from megagemm.models.llama import (
        _gemma4_a100_a4b_fused_attn_prepare_shape,
    )

    device = "NVIDIA A100-SXM4-80GB"
    assert _gemma4_a100_a4b_fused_attn_prepare_shape(
        1, 25, 16, 8, 256, 256, torch.bfloat16, device
    )
    assert _gemma4_a100_a4b_fused_attn_prepare_shape(
        1, 25, 16, 2, 512, 512, torch.bfloat16, device
    )
    assert _gemma4_a100_a4b_fused_attn_prepare_shape(
        8, 25, 16, 8, 256, 256, torch.bfloat16, device
    )
    assert _gemma4_a100_a4b_fused_attn_prepare_shape(
        16, 25, 16, 2, 512, 512, torch.bfloat16, device
    )
    assert _gemma4_a100_a4b_fused_attn_prepare_shape(
        8, 2048, 16, 2, 512, 512, torch.bfloat16, device
    )
    assert _gemma4_a100_a4b_fused_attn_prepare_shape(
        8, 2048, 16, 8, 256, 256, torch.bfloat16, device
    )
    assert _gemma4_a100_a4b_fused_attn_prepare_shape(
        16, 2048, 16, 2, 512, 512, torch.bfloat16, device
    )
    assert _gemma4_a100_a4b_fused_attn_prepare_shape(
        16, 2048, 16, 8, 256, 256, torch.bfloat16, device
    )
    assert not _gemma4_a100_a4b_fused_attn_prepare_shape(
        8, 1024, 16, 2, 512, 512, torch.bfloat16, device
    )
    assert not _gemma4_a100_a4b_fused_attn_prepare_shape(
        4, 25, 16, 8, 256, 256, torch.bfloat16, device
    )
    assert not _gemma4_a100_a4b_fused_attn_prepare_shape(
        8, 24, 16, 8, 256, 256, torch.bfloat16, device
    )
    assert not _gemma4_a100_a4b_fused_attn_prepare_shape(
        1, 33, 16, 8, 256, 256, torch.bfloat16, device
    )
    assert not _gemma4_a100_a4b_fused_attn_prepare_shape(
        1, 25, 16, 8, 256, 128, torch.bfloat16, device
    )
    assert not _gemma4_a100_a4b_fused_attn_prepare_shape(
        1, 25, 16, 8, 256, 256, torch.float16, device
    )
    assert not _gemma4_a100_a4b_fused_attn_prepare_shape(
        1, 25, 16, 8, 256, 256, torch.bfloat16, "NVIDIA H100 80GB HBM3"
    )
    assert not _gemma4_a100_a4b_fused_attn_prepare_shape(
        8, 2048, 16, 2, 512, 512, torch.float16, device
    )
    assert not _gemma4_a100_a4b_fused_attn_prepare_shape(
        8, 2048, 16, 2, 512, 512, torch.bfloat16, "NVIDIA H100 80GB HBM3"
    )


def test_gemma4_a4b_long_kv_scatter_policy_is_exact():
    from megagemm.models.llama import (
        _gemma4_a100_a4b_long_kv_scatter_tokens_per_program,
    )

    device = "NVIDIA A100-SXM4-80GB"
    policy = _gemma4_a100_a4b_long_kv_scatter_tokens_per_program
    assert policy(8, 2048, 8, 256, torch.bfloat16, device) == 4
    assert policy(8, 2048, 2, 512, torch.bfloat16, device) == 2
    assert policy(16, 2048, 8, 256, torch.bfloat16, device) == 4
    assert policy(16, 2048, 2, 512, torch.bfloat16, device) == 2
    assert policy(8, 1024, 8, 256, torch.bfloat16, device) == 1
    assert policy(8, 2048, 4, 256, torch.bfloat16, device) == 1
    assert policy(8, 2048, 8, 256, torch.float16, device) == 1
    assert policy(8, 2048, 8, 256, torch.bfloat16, "NVIDIA H100 80GB HBM3") == 1


def test_gemma4_prefill_last_token_only_matches_full_logits():
    torch.manual_seed(11)
    config = LlamaConfig.from_dict(
        _tiny_gemma4_config_dict(num_kv_shared_layers=0)
    )
    model = MegaGemmLlama(config).float().eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    positions = torch.arange(4).unsqueeze(0)

    def make_manager():
        manager = BlockManager(
            num_layers=config.num_hidden_layers,
            num_blocks=16,
            block_size=4,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=torch.float32,
            device="cpu",
            kv_layer_indices=config.kv_cache_layer_indices,
            per_layer_num_kv_heads=config.per_layer_num_kv_heads,
            per_layer_head_dims=config.per_layer_head_dims,
            kv_layer_sources=_kv_sources(config),
        )
        manager.allocate_sequence(0, 8)
        return manager

    with torch.inference_mode():
        full_logits = model.prefill(
            input_ids, positions, make_manager(), 0, last_token_only=False
        )
        last_logits = model.prefill(
            input_ids, positions, make_manager(), 0, last_token_only=True
        )

    assert full_logits.shape == (1, 4, config.vocab_size)
    assert last_logits.shape == (1, 1, config.vocab_size)
    assert torch.allclose(last_logits, full_logits[:, -1:, :], atol=1e-5, rtol=1e-5)
    assert torch.equal(last_logits.argmax(dim=-1), full_logits[:, -1:, :].argmax(dim=-1))


def test_gemma4_moe_config_loads_text_backbone_and_runs(tmp_path):
    snapshot = tmp_path / "gemma4-moe-hf"
    _build_local_gemma4_snapshot(
        snapshot,
        _tiny_gemma4_config_dict(
            enable_moe_block=True,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=8,
            hidden_size_per_layer_input=0,
            use_double_wide_mlp=False,
        ),
    )

    model = load_from_hf(str(snapshot), dtype=torch.float16, device="cpu")

    assert model.config.enable_moe_block
    assert model.config.is_moe_layer(0)
    assert model.layers[0].mlp.__class__.__name__ == "Gemma4MoeMLP"
    assert model.layers[0].mlp.shared_mlp.gate_up_proj.weight.shape[0] == 128
    assert model.layers[0].mlp.experts.gate_up_proj.shape == (4, 16, 32)
    assert model.layers[0].mlp.experts.down_proj.shape == (4, 32, 8)
    assert model.layers[0].mlp.gate.proj.weight.shape == (4, 32)
    assert model.layers[0].mlp.gate.scale.shape == (32,)
    assert model.layers[0].mlp.gate.per_expert_scale.shape == (4,)
    assert model.config.num_experts_per_tok == 2
    assert model.layers[0].pre_feedforward_layernorm.weight.shape == (32,)
    assert model.layers[0].pre_feedforward_layernorm_1 is None
    assert model.layers[0].pre_feedforward_layernorm_2.weight.shape == (32,)
    assert model.layers[0].post_feedforward_layernorm.weight.shape == (32,)
    assert model.layers[0].post_feedforward_layernorm_2.weight.shape == (32,)

    block_manager = BlockManager(
        num_layers=model.config.num_hidden_layers,
        num_blocks=16,
        block_size=4,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        dtype=torch.float16,
        device="cpu",
        kv_layer_indices=model.config.kv_cache_layer_indices,
        per_layer_num_kv_heads=model.config.per_layer_num_kv_heads,
        per_layer_head_dims=model.config.per_layer_head_dims,
        kv_layer_sources=_kv_sources(model.config),
    )
    block_manager.allocate_sequence(0, 8)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    lengths = torch.tensor([4], dtype=torch.long)
    logits = model.prefill_batch(input_ids, lengths, block_manager, [0])
    mlp_out = model.layers[0].mlp(torch.randn(2, 1, model.config.hidden_size, dtype=torch.float16))

    assert logits.shape == (1, 1, model.config.vocab_size)
    assert mlp_out.shape == (2, 1, model.config.hidden_size)
    assert bool(torch.isfinite(mlp_out).all())


def test_gemma4_moe_legacy_ffn_norm_names_load(tmp_path):
    snapshot = tmp_path / "gemma4-moe-legacy-norms"
    hf_config = _tiny_gemma4_config_dict(
        enable_moe_block=True,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        hidden_size_per_layer_input=0,
        use_double_wide_mlp=False,
    )
    _build_local_gemma4_snapshot(snapshot, hf_config, legacy_moe_norm_names=True)

    model = load_from_hf(str(snapshot), dtype=torch.float16, device="cpu")

    for layer in model.layers:
        assert layer.pre_feedforward_layernorm.weight.device.type != "meta"
        assert layer.pre_feedforward_layernorm_2.weight.device.type != "meta"
        assert torch.allclose(
            layer.pre_feedforward_layernorm.weight,
            layer.pre_feedforward_layernorm_2.weight,
        )
        assert torch.allclose(
            layer.post_feedforward_layernorm.weight,
            layer.post_feedforward_layernorm_1.weight,
        )
        assert torch.allclose(
            layer.post_feedforward_layernorm.weight,
            layer.post_feedforward_layernorm_2.weight,
        )


def test_gemma4_moe_ffn_matches_official_branch_order():
    class ZeroAttention(torch.nn.Module):
        def forward(self, hidden_states, *args, **kwargs):
            return torch.zeros_like(hidden_states), None, None

    torch.manual_seed(7)
    config = LlamaConfig.from_dict(
        _tiny_gemma4_config_dict(
            enable_moe_block=True,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=8,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            use_double_wide_mlp=False,
            attention_k_eq_v=True,
        )
    )
    model = MegaGemmLlama(config).float().eval()
    layer = model.layers[0]
    layer.self_attn = ZeroAttention()
    hidden_states = torch.randn(1, 3, config.hidden_size)

    residual = hidden_states
    route_input = residual.reshape(-1, residual.shape[-1])
    _, routing_weights, selected_experts = layer.mlp.gate(route_input)
    shared_out = layer.mlp.shared_mlp(
        layer.pre_feedforward_layernorm(residual),
        is_prefill=True,
    )
    shared_out = layer.post_feedforward_layernorm_1(shared_out)
    expert_out = layer.mlp._routed(
        layer.pre_feedforward_layernorm_2(residual),
        is_prefill=True,
        selected_experts=selected_experts,
        routing_weights=routing_weights,
    )
    expert_out = layer.post_feedforward_layernorm_2(expert_out)
    expected = residual + layer.post_feedforward_layernorm(shared_out + expert_out)
    expected = expected * layer.layer_scalar.to(expected.dtype)

    actual, *_ = layer(
        hidden_states,
        torch.ones(1, config.head_dim // 2),
        torch.zeros(1, config.head_dim // 2),
        torch.arange(hidden_states.shape[1]).unsqueeze(0),
        is_prefill=True,
    )

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_gemma4_tiny_prefill_uses_heterogeneous_kv_and_shared_cache():
    torch.manual_seed(0)
    config = LlamaConfig.from_dict(_tiny_gemma4_config_dict())
    model = MegaGemmLlama(config).eval()
    model._refresh_gemma4_runtime_buffers(device="cpu", dtype=torch.float32)
    assert model.layers[3].self_attn.is_kv_shared
    assert model.layers[3].self_attn.k_proj is None
    assert model.layers[3].self_attn.k_norm is None
    full_cos, full_sin = model.layer_rope_caches[1]
    assert torch.allclose(full_cos[:, 2:], torch.ones_like(full_cos[:, 2:]))
    assert torch.allclose(full_sin[:, 2:], torch.zeros_like(full_sin[:, 2:]))

    block_manager = BlockManager(
        num_layers=config.num_hidden_layers,
        num_blocks=16,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
        device="cpu",
        kv_layer_indices=config.kv_cache_layer_indices,
        per_layer_num_kv_heads=config.per_layer_num_kv_heads,
        per_layer_head_dims=config.per_layer_head_dims,
        kv_layer_sources=_kv_sources(config),
    )
    block_manager.allocate_sequence(0, 8)
    block_manager.allocate_sequence(1, 8)

    input_ids = torch.tensor([[1, 2, 3, 4, 5], [0, 0, 6, 7, 8]], dtype=torch.long)
    lengths = torch.tensor([5, 3], dtype=torch.long)
    logits = model.prefill_batch(input_ids, lengths, block_manager, [0, 1])

    assert logits.shape == (2, 1, config.vocab_size)
    assert block_manager.get_kv_cache(1).shape[2:] == (2, 4, 16)
    assert block_manager.get_kv_cache(3) is block_manager.get_kv_cache(1)
    assert model._gemma4_implicit_causal_prefill_batches == 0

    decode_ids = torch.tensor([[9], [10]], dtype=torch.long)
    decode_positions = (
        block_manager.get_seq_lens_tensor([0, 1]).long().unsqueeze(1)
    )
    decode_logits = model.decode_step(
        decode_ids,
        decode_positions,
        block_manager,
        [0, 1],
    )
    assert decode_logits.shape == (2, 1, config.vocab_size)


def test_gemma4_prefill_runtime_flags_are_scoped_by_attention_type():
    config = LlamaConfig.from_dict(_tiny_gemma4_config_dict())
    model = MegaGemmLlama(config).eval()

    promoted_fields = {
        "gemma4_e2b_l4_sliding_prefill",
        "gemma4_e2b_l4_full_prefill_expand",
    }

    def promoted_policy(_model, _env_name, policy_field, default=False):
        return policy_field in promoted_fields

    with mock.patch(
        "megagemm.models.llama.policy_bool",
        side_effect=promoted_policy,
    ):
        model._refresh_gemma4_runtime_buffers(
            device="cpu",
            dtype=torch.float32,
        )

    sliding_flags = [
        layer.self_attn._gemma4_e2b_l4_sliding_prefill_enabled
        for layer in model.layers
    ]
    full_flags = [
        layer.self_attn._gemma4_e2b_l4_full_prefill_expand_enabled
        for layer in model.layers
    ]

    assert sliding_flags == [True, False, True, False]
    assert full_flags == [False, True, False, True]


def test_gemma4_e2b_attention_prepare_policy_enables_only_kv_sources():
    from megagemm.models.runtime_policy import RuntimePolicy

    config = LlamaConfig.from_dict(_tiny_gemma4_config_dict())
    model = MegaGemmLlama(config).eval()
    launches = (
        (521, 256, 4, 2, True),
        (521, 512, 8, 2, True),
        (2057, 256, 4, 2, True),
        (2057, 512, 4, 2, True),
    )
    policy = RuntimePolicy(
        name="gemma4-e2b-l4",
        hardware="NVIDIA L4",
        gemma4_e2b_b8_fused_attn_prepare=True,
        gemma4_e2b_b8_fused_attn_prepare_launches=launches,
    )

    with mock.patch(
        "megagemm.models.llama.resolve_runtime_policy",
        return_value=policy,
    ):
        model._refresh_gemma4_runtime_buffers(device="cpu", dtype=torch.float32)

    attentions = [layer.self_attn for layer in model.layers]
    assert [
        attention._gemma4_e2b_l4_fused_attn_prepare_enabled
        for attention in attentions
    ] == [True, True, True, False]
    expected = {
        (sequence_len, head_dim): (warps, stages, split)
        for sequence_len, head_dim, warps, stages, split in launches
    }
    assert all(
        attention._gemma4_e2b_l4_fused_attn_prepare_launch_by_shape == expected
        for attention in attentions
    )


def test_gemma4_e2b_prefill_dense_bridge_policy_is_sequence_scoped():
    from megagemm.models.runtime_policy import RuntimePolicy

    config = LlamaConfig.from_dict(_tiny_gemma4_config_dict())
    model = MegaGemmLlama(config).eval()
    policy = RuntimePolicy(
        name="gemma4-e2b-l4",
        hardware="NVIDIA L4",
        gemma4_e2b_b8_prefill_dense_bridge=True,
        gemma4_e2b_b8_prefill_dense_bridge_warps=((2057, 4),),
    )

    with mock.patch(
        "megagemm.models.llama.resolve_runtime_policy",
        return_value=policy,
    ):
        model._refresh_gemma4_runtime_buffers(device="cpu", dtype=torch.float32)

    assert all(
        layer._gemma4_e2b_prefill_dense_bridge_enabled
        for layer in model.layers
    )
    assert all(
        layer._gemma4_e2b_prefill_dense_bridge_warps_by_sequence
        == {2057: 4}
        for layer in model.layers
    )

def test_gemma4_uniform_batch_vectorizes_kv_and_projects_only_last_tokens():
    torch.manual_seed(0)
    config = LlamaConfig.from_dict(_tiny_gemma4_config_dict())
    model = MegaGemmLlama(config).eval()
    model._refresh_gemma4_runtime_buffers(device="cpu", dtype=torch.float32)
    block_manager = BlockManager(
        num_layers=config.num_hidden_layers,
        num_blocks=16,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
        device="cpu",
        kv_layer_indices=config.kv_cache_layer_indices,
        per_layer_num_kv_heads=config.per_layer_num_kv_heads,
        per_layer_head_dims=config.per_layer_head_dims,
        kv_layer_sources=_kv_sources(config),
    )
    block_manager.allocate_sequence(0, 8)
    block_manager.allocate_sequence(1, 8)

    lm_head_inputs = []
    hook = model.lm_head.register_forward_pre_hook(
        lambda _module, args: lm_head_inputs.append(tuple(args[0].shape))
    )
    try:
        logits = model.prefill_batch(
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long),
            torch.tensor([4, 4], dtype=torch.long),
            block_manager,
            [0, 1],
            prompt_lengths_cpu=[4, 4],
        )
    finally:
        hook.remove()

    assert logits.shape == (2, 1, config.vocab_size)
    assert lm_head_inputs == [(2, 1, config.hidden_size)]
    assert model._prefill_last_token_only_hits == 1
    assert model._gemma4_batch_prefill_vectorized_kv_hits > 0
    assert model._gemma4_implicit_causal_prefill_batches == 1
    assert block_manager.seq_lens[0] == 4
    assert block_manager.seq_lens[1] == 4


def test_gemma4_prefill_finite_trace_passes_and_pinpoints_first_bad_stage():
    torch.manual_seed(17)
    config = LlamaConfig.from_dict(
        _tiny_gemma4_config_dict(
            enable_moe_block=True,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=8,
            hidden_size_per_layer_input=0,
            vocab_size_per_layer_input=0,
            use_double_wide_mlp=False,
            num_kv_shared_layers=0,
        )
    )

    def make_blocks():
        blocks = BlockManager(
            num_layers=config.num_hidden_layers,
            num_blocks=16,
            block_size=4,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=torch.float32,
            device="cpu",
            kv_layer_indices=config.kv_cache_layer_indices,
            per_layer_num_kv_heads=config.per_layer_num_kv_heads,
            per_layer_head_dims=config.per_layer_head_dims,
            kv_layer_sources=_kv_sources(config),
        )
        blocks.allocate_sequence(0, 8)
        blocks.allocate_sequence(1, 8)
        return blocks

    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    lengths = torch.tensor([4, 4], dtype=torch.long)
    model = MegaGemmLlama(config).float().eval()
    model._refresh_gemma4_runtime_buffers(device="cpu", dtype=torch.float32)
    model.begin_gemma4_prefill_finite_trace()
    try:
        logits = model.prefill_batch(
            input_ids,
            lengths,
            make_blocks(),
            [0, 1],
            prompt_lengths_cpu=[4, 4],
        )
    finally:
        clean_trace = model.end_gemma4_prefill_finite_trace()

    assert bool(torch.isfinite(logits).all())
    assert clean_trace["status"] == "PASS"
    assert clean_trace["first_bad"] is None
    assert clean_trace["events"][0]["stage"] == "model.embedding"
    assert clean_trace["events"][-1]["stage"] == "model.capped_logits"

    original_forward = model.layers[0].input_layernorm.forward

    def poison_second_batch_row(hidden):
        output = original_forward(hidden).clone()
        output[1].fill_(float("nan"))
        return output

    model.begin_gemma4_prefill_finite_trace()
    error = None
    try:
        with mock.patch.object(
            model.layers[0].input_layernorm,
            "forward",
            side_effect=poison_second_batch_row,
        ):
            model.prefill_batch(
                input_ids,
                lengths,
                make_blocks(),
                [0, 1],
                prompt_lengths_cpu=[4, 4],
            )
    except RuntimeError as exc:
        error = str(exc)
    finally:
        bad_trace = model.end_gemma4_prefill_finite_trace()

    assert error is not None and "first nonfinite tensor" in error
    assert bad_trace["status"] == "NONFINITE"
    assert bad_trace["first_bad"]["layer"] == 0
    assert bad_trace["first_bad"]["stage"] == "layer.input_norm"
    assert bad_trace["first_bad"]["finite_rows"] == [0]
    assert bad_trace["first_bad"]["nonfinite_rows"] == [1]


def test_gemma4_uniform_implicit_causal_prefill_matches_explicit_mask():
    torch.manual_seed(7)
    config = LlamaConfig.from_dict(
        _tiny_gemma4_config_dict(
            sliding_window=8,
            hidden_size_per_layer_input=0,
            vocab_size_per_layer_input=0,
        )
    )
    implicit_model = MegaGemmLlama(config).eval()
    implicit_model._refresh_gemma4_runtime_buffers(device="cpu", dtype=torch.float32)
    explicit_model = copy.deepcopy(implicit_model)

    def make_blocks():
        blocks = BlockManager(
            num_layers=config.num_hidden_layers,
            num_blocks=16,
            block_size=4,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=torch.float32,
            device="cpu",
            kv_layer_indices=config.kv_cache_layer_indices,
            per_layer_num_kv_heads=config.per_layer_num_kv_heads,
            per_layer_head_dims=config.per_layer_head_dims,
            kv_layer_sources=_kv_sources(config),
        )
        blocks.allocate_sequence(0, 8)
        blocks.allocate_sequence(1, 8)
        return blocks

    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    lengths = torch.tensor([4, 4], dtype=torch.long)
    with mock.patch(
        "megagemm.models.llama._GEMMA4_IMPLICIT_CAUSAL_PREFILL",
        True,
    ):
        implicit_logits = implicit_model.prefill_batch(
            input_ids,
            lengths,
            make_blocks(),
            [0, 1],
            prompt_lengths_cpu=[4, 4],
        )
    with mock.patch(
        "megagemm.models.llama._GEMMA4_IMPLICIT_CAUSAL_PREFILL",
        False,
    ):
        explicit_logits = explicit_model.prefill_batch(
            input_ids,
            lengths,
            make_blocks(),
            [0, 1],
            prompt_lengths_cpu=[4, 4],
        )

    assert torch.allclose(implicit_logits, explicit_logits, atol=1e-5, rtol=1e-5)
    assert implicit_model._gemma4_implicit_causal_prefill_batches == 1
    assert explicit_model._gemma4_implicit_causal_prefill_batches == 0
    assert sum(
        layer.self_attn._gemma4_implicit_causal_prefill_hits
        for layer in implicit_model.layers
    ) == sum(
        layer_type == "sliding_attention"
        for layer_type in config.layer_types
    )


def test_compute_kv_mapping_accepts_scheduler_owned_lengths_without_offsets():
    block_manager = BlockManager(
        num_layers=1,
        num_blocks=8,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )
    block_manager.allocate_sequence(0, 4)
    block_manager.allocate_sequence(1, 4)

    physical, offsets = block_manager.compute_kv_mapping(
        [0, 1],
        object(),
        torch.device("cpu"),
        seq_lengths=[4, 4],
    )

    assert physical.shape == (8,)
    assert offsets.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]


def test_packed_kv_write_prefers_single_scatter_kernel():
    block_manager = BlockManager(
        num_layers=1,
        num_blocks=8,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )
    block_manager.allocate_sequence(0, 4)
    block_manager.allocate_sequence(1, 4)
    physical, offsets = block_manager.compute_kv_mapping(
        [0, 1],
        object(),
        torch.device("cpu"),
        seq_lengths=[4, 4],
    )
    k = torch.arange(16, dtype=torch.float32).reshape(8, 1, 2)
    v = k + 100.0
    calls = []

    def fake_scatter(k_arg, v_arg, cache_arg, physical_arg, offsets_arg):
        calls.append(1)
        cache_arg[physical_arg, 0, :, offsets_arg, :] = k_arg
        cache_arg[physical_arg, 1, :, offsets_arg, :] = v_arg
        return True

    block_manager._prefill_kv_scatter_requested = True
    with mock.patch(
        "megagemm.engine.kv_cache._paged_kv_cache_scatter",
        side_effect=fake_scatter,
    ):
        block_manager.write_kv_prefill_packed(
            [0, 1],
            0,
            k,
            v,
            torch.tensor([0, 4, 8], dtype=torch.int32),
            kv_mapping=(physical, offsets),
        )

    assert calls == [1]
    assert block_manager.prefill_kv_scatter_stats()["hits"] == 1
    cache = block_manager.get_kv_cache(0)
    assert torch.equal(cache[physical, 0, :, offsets, :], k)
    assert torch.equal(cache[physical, 1, :, offsets, :], v)
    assert not paged_kv_cache_scatter(k, v, cache, physical, offsets)


def test_packed_kv_write_uses_requested_token_tile_before_baseline():
    block_manager = BlockManager(
        num_layers=1,
        num_blocks=8,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )
    block_manager.allocate_sequence(0, 4)
    block_manager.allocate_sequence(1, 4)
    physical, offsets = block_manager.compute_kv_mapping(
        [0, 1],
        object(),
        torch.device("cpu"),
        seq_lengths=[4, 4],
    )
    k = torch.arange(16, dtype=torch.float32).reshape(8, 1, 2)
    v = k + 100.0
    calls = []

    def fake_tiled(
        k_arg,
        v_arg,
        cache_arg,
        physical_arg,
        offsets_arg,
        *,
        tokens_per_program,
    ):
        calls.append(tokens_per_program)
        cache_arg[physical_arg, 0, :, offsets_arg, :] = k_arg
        cache_arg[physical_arg, 1, :, offsets_arg, :] = v_arg
        return True

    block_manager._prefill_kv_scatter_requested = True
    with mock.patch(
        "megagemm.engine.kv_cache._paged_kv_cache_scatter_token_tiled",
        side_effect=fake_tiled,
    ), mock.patch(
        "megagemm.engine.kv_cache._paged_kv_cache_scatter",
        side_effect=AssertionError("baseline scatter must not run"),
    ):
        block_manager.write_kv_prefill_packed(
            [0, 1],
            0,
            k,
            v,
            torch.tensor([0, 4, 8], dtype=torch.int32),
            kv_mapping=(physical, offsets),
            tokens_per_program=4,
        )

    assert calls == [4]
    stats = block_manager.prefill_kv_scatter_stats()
    assert stats["token_tiled_hits"] == 1
    assert stats["hits"] == 0
    cache = block_manager.get_kv_cache(0)
    assert torch.equal(cache[physical, 0, :, offsets, :], k)
    assert torch.equal(cache[physical, 1, :, offsets, :], v)


def test_gemma4_a4b_prefill_graph_policy_is_exact():
    class Config:
        model_type = "gemma4_text"
        enable_moe_block = True
        hidden_size = 2816
        num_hidden_layers = 30
        num_experts = 128
        num_experts_per_tok = 8
        moe_intermediate_size = 704
        hidden_size_per_layer_input = 0
        num_kv_shared_layers = 0
        layer_types = [
            layer_type
            for _ in range(5)
            for layer_type in (
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            )
        ]

        @staticmethod
        def is_moe_layer(layer_idx):
            return 0 <= int(layer_idx) < 30

    kwargs = {
        "num_seqs": 1,
        "total_tokens": 25,
        "dtype": torch.bfloat16,
        "device_type": "cuda",
        "device_name": "NVIDIA A100-SXM4-80GB",
    }
    assert _gemma4_a100_a4b_prefill_graph_shape(Config(), **kwargs)
    assert _gemma4_a100_a4b_prefill_graph_shape(
        Config(), **{**kwargs, "num_seqs": 16, "total_tokens": 400}
    )
    for batch_size in (2, 4, 8):
        assert not _gemma4_a100_a4b_prefill_graph_shape(
            Config(),
            **{
                **kwargs,
                "num_seqs": batch_size,
                "total_tokens": 25 * batch_size,
            },
        )
    assert not _gemma4_a100_a4b_prefill_graph_shape(
        Config(), **{**kwargs, "num_seqs": 3, "total_tokens": 75}
    )
    assert not _gemma4_a100_a4b_prefill_graph_shape(
        Config(), **{**kwargs, "num_seqs": 16, "total_tokens": 399}
    )
    assert not _gemma4_a100_a4b_prefill_graph_shape(
        Config(), **{**kwargs, "total_tokens": 24}
    )
    assert not _gemma4_a100_a4b_prefill_graph_shape(
        Config(), **{**kwargs, "device_name": "NVIDIA H100 80GB HBM3"}
    )


def test_gemma4_a4b_decode_graph_policy_is_exact():
    from megagemm.models.llama import _gemma4_a100_a4b_decode_graph_shape

    class Config:
        model_type = "gemma4_text"
        enable_moe_block = True
        hidden_size = 2816
        num_hidden_layers = 30
        num_experts = 128
        num_experts_per_tok = 8
        moe_intermediate_size = 704
        hidden_size_per_layer_input = 0
        num_kv_shared_layers = 0
        layer_types = [
            layer_type
            for _ in range(5)
            for layer_type in (
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            )
        ]

        @staticmethod
        def is_moe_layer(layer_idx):
            return 0 <= int(layer_idx) < 30

    kwargs = {
        "dtype": torch.bfloat16,
        "device_type": "cuda",
        "device_name": "NVIDIA A100-SXM4-80GB",
    }
    for batch_size in (1, 2, 4, 8, 16):
        assert _gemma4_a100_a4b_decode_graph_shape(
            Config(), num_seqs=batch_size, **kwargs
        )
    assert not _gemma4_a100_a4b_decode_graph_shape(
        Config(), num_seqs=3, **kwargs
    )
    assert not _gemma4_a100_a4b_decode_graph_shape(
        Config(),
        num_seqs=16,
        **{**kwargs, "device_name": "NVIDIA H100 80GB HBM3"},
    )


def test_gemma4_e2b_l4_native_decode_graph_policy_is_exact():
    from types import SimpleNamespace

    from megagemm.models.llama import (
        MegaGemmLlama,
        _gemma4_l4_e2b_decode_graph_shape,
    )

    class Config:
        model_type = "gemma4_text"
        enable_moe_block = False
        hidden_size = 1536
        num_hidden_layers = 35
        num_attention_heads = 8
        num_key_value_heads = 1
        num_kv_shared_layers = 20
        layer_types = [
            "full_attention"
            if layer_idx in {0, 6, 12, 18, 24, 30, 34}
            else "sliding_attention"
            for layer_idx in range(35)
        ]

    kwargs = {
        "num_seqs": 8,
        "dtype": torch.bfloat16,
        "device_type": "cuda",
        "device_name": "NVIDIA L4",
    }
    assert Config.layer_types.count("sliding_attention") == 28
    assert Config.layer_types.count("full_attention") == 7
    assert _gemma4_l4_e2b_decode_graph_shape(Config(), **kwargs)
    assert not _gemma4_l4_e2b_decode_graph_shape(
        Config(), **{**kwargs, "num_seqs": 1}
    )
    assert not _gemma4_l4_e2b_decode_graph_shape(
        Config(), **{**kwargs, "dtype": torch.float16}
    )
    assert not _gemma4_l4_e2b_decode_graph_shape(
        Config(), **{**kwargs, "device_name": "NVIDIA A100-SXM4-80GB"}
    )

    model = SimpleNamespace(config=Config())
    assert MegaGemmLlama.decode_unrolled_graph_burst_eligible(
        model,
        num_steps=8,
        **kwargs,
    )
    assert not MegaGemmLlama.decode_unrolled_graph_burst_eligible(
        model,
        num_steps=1,
        **kwargs,
    )
    assert not MegaGemmLlama.decode_unrolled_graph_burst_eligible(
        model,
        num_steps=9,
        **kwargs,
    )


def test_gemma4_batch_prefill_selects_native_padded_path():
    from megagemm.engine.scheduler import Scheduler

    class Config:
        model_type = "gemma4_text"

    class Model:
        config = Config()

    class FakeScheduler:
        model = Model()
        _prefill_prefer_padded = False

    assert Scheduler._should_use_padded_prefill(
        FakeScheduler(), [object(), object()], has_packed=True, has_padded=True
    )


def test_gemma4_moe_graph_prefill_matches_eager_with_per_layer_rope():
    torch.manual_seed(11)
    config = LlamaConfig.from_dict(
        _tiny_gemma4_config_dict(
            enable_moe_block=True,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=8,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            use_double_wide_mlp=False,
        )
    )
    eager_model = MegaGemmLlama(config).float().eval()
    eager_model._refresh_gemma4_runtime_buffers(device="cpu", dtype=torch.float32)
    graph_model = copy.deepcopy(eager_model).eval()

    def make_manager():
        manager = BlockManager(
            num_layers=config.num_hidden_layers,
            num_blocks=16,
            block_size=4,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=torch.float32,
            device="cpu",
            kv_layer_indices=config.kv_cache_layer_indices,
            per_layer_num_kv_heads=config.per_layer_num_kv_heads,
            per_layer_head_dims=config.per_layer_head_dims,
            kv_layer_sources=_kv_sources(config),
        )
        manager.allocate_sequence(0, 8)
        return manager

    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    cu_seqlens = torch.tensor([0, 5], dtype=torch.int32)
    eager_manager = make_manager()
    graph_manager = make_manager()

    with torch.inference_mode():
        eager_logits = eager_model.prefill(
            input_ids,
            torch.arange(input_ids.shape[1]).unsqueeze(0),
            eager_manager,
            0,
            last_token_only=True,
        )
        kv_phys, kv_offs = graph_manager.compute_kv_mapping(
            [0],
            cu_seqlens,
            input_ids.device,
        )
        graph_logits = graph_model.prefill_packed_graph(
            input_ids,
            cu_seqlens,
            graph_manager,
            kv_phys,
            kv_offs,
        )
        graph_manager.advance_seq_len(0, input_ids.shape[1])

    assert torch.allclose(graph_logits, eager_logits, atol=1e-5, rtol=1e-5)
    assert torch.equal(graph_logits.argmax(dim=-1), eager_logits.argmax(dim=-1))
    assert graph_manager.seq_lens[0] == eager_manager.seq_lens[0] == input_ids.shape[1]
    for eager_cache, graph_cache in zip(
        eager_manager.kv_caches,
        graph_manager.kv_caches,
    ):
        assert torch.allclose(graph_cache, eager_cache, atol=1e-5, rtol=1e-5)


def test_gemma4_moe_padded_batch_graph_prefill_matches_eager():
    torch.manual_seed(12)
    config = LlamaConfig.from_dict(
        _tiny_gemma4_config_dict(
            enable_moe_block=True,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=8,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            use_double_wide_mlp=False,
        )
    )
    eager_model = MegaGemmLlama(config).float().eval()
    eager_model._refresh_gemma4_runtime_buffers(device="cpu", dtype=torch.float32)
    graph_model = copy.deepcopy(eager_model).eval()

    def make_manager():
        manager = BlockManager(
            num_layers=config.num_hidden_layers,
            num_blocks=32,
            block_size=4,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=torch.float32,
            device="cpu",
            kv_layer_indices=config.kv_cache_layer_indices,
            per_layer_num_kv_heads=config.per_layer_num_kv_heads,
            per_layer_head_dims=config.per_layer_head_dims,
            kv_layer_sources=_kv_sources(config),
        )
        manager.allocate_sequence(0, 8)
        manager.allocate_sequence(1, 8)
        return manager

    input_ids = torch.tensor(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
        dtype=torch.long,
    )
    lengths = torch.tensor([5, 5], dtype=torch.long)
    cu_seqlens = torch.tensor([0, 5, 10], dtype=torch.int32)
    eager_manager = make_manager()
    graph_manager = make_manager()

    with torch.inference_mode():
        eager_logits = eager_model.prefill_batch(
            input_ids,
            lengths,
            eager_manager,
            [0, 1],
            prompt_lengths_cpu=[5, 5],
        )
        kv_phys, kv_offs = graph_manager.compute_kv_mapping(
            [0, 1],
            cu_seqlens,
            input_ids.device,
            seq_lengths=[5, 5],
        )
        graph_logits, deferred_kv = graph_model.prefill_batch_graph(
            input_ids,
            cu_seqlens,
            graph_manager,
            kv_phys,
            kv_offs,
            defer_kv_writes=True,
        )
        first_deferred_ptrs = tuple(
            (layer_idx, k_cache.data_ptr(), v_cache.data_ptr())
            for layer_idx, k_cache, v_cache in deferred_kv
        )
        first_deferred_values = tuple(
            (k_cache.clone(), v_cache.clone())
            for _, k_cache, v_cache in deferred_kv
        )
        replay_logits, replay_deferred_kv = graph_model.prefill_batch_graph(
            input_ids,
            cu_seqlens,
            graph_manager,
            kv_phys,
            kv_offs,
            defer_kv_writes=True,
        )
        replay_deferred_ptrs = tuple(
            (layer_idx, k_cache.data_ptr(), v_cache.data_ptr())
            for layer_idx, k_cache, v_cache in replay_deferred_kv
        )
        assert replay_deferred_ptrs == first_deferred_ptrs
        assert len(graph_model._prefill_graph_deferred_kv_buffers) == (
            config.num_hidden_layers
        )
        assert torch.equal(replay_logits, graph_logits)
        for expected, (_, actual_k, actual_v) in zip(
            first_deferred_values,
            replay_deferred_kv,
        ):
            expected_k, expected_v = expected
            assert torch.equal(actual_k, expected_k)
            assert torch.equal(actual_v, expected_v)
        deferred_kv = replay_deferred_kv
        for layer in graph_model.layers:
            assert (
                layer.self_attn._gemma4_fused_qkv_prefill_skip_reason
                == "prefill CUDA graph safety guard"
            )
            assert (
                layer.self_attn._gemma4_fused_attn_prepare_skip_reason
                == "prefill CUDA graph safety guard"
            )
        assert len(deferred_kv) == config.num_hidden_layers
        for layer_idx, k_cache, v_cache in deferred_kv:
            graph_manager.write_kv_prefill_packed(
                [],
                layer_idx,
                k_cache,
                v_cache,
                cu_seqlens,
                kv_mapping=(kv_phys, kv_offs),
            )
        graph_manager.advance_seq_len(0, 5)
        graph_manager.advance_seq_len(1, 5)

    assert torch.allclose(graph_logits, eager_logits, atol=1e-5, rtol=1e-5)
    assert torch.equal(graph_logits.argmax(dim=-1), eager_logits.argmax(dim=-1))
    assert graph_manager.seq_lens == eager_manager.seq_lens
    for eager_cache, graph_cache in zip(
        eager_manager.kv_caches,
        graph_manager.kv_caches,
    ):
        assert torch.allclose(graph_cache, eager_cache, atol=1e-5, rtol=1e-5)


def test_gemma4_flat_decode_matches_regular_decode():
    torch.manual_seed(123)
    config = LlamaConfig.from_dict(_tiny_gemma4_config_dict())
    flat_model = MegaGemmLlama(config).eval()
    flat_model._refresh_gemma4_runtime_buffers(device="cpu", dtype=torch.float32)
    regular_model = copy.deepcopy(flat_model).eval()

    def make_block_manager():
        block_manager = BlockManager(
            num_layers=config.num_hidden_layers,
            num_blocks=16,
            block_size=4,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=torch.float32,
            device="cpu",
            kv_layer_indices=config.kv_cache_layer_indices,
            per_layer_num_kv_heads=config.per_layer_num_kv_heads,
            per_layer_head_dims=config.per_layer_head_dims,
            kv_layer_sources=_kv_sources(config),
        )
        block_manager.allocate_sequence(0, 8)
        block_manager.allocate_sequence(1, 8)
        return block_manager

    flat_blocks = make_block_manager()
    regular_blocks = make_block_manager()
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [0, 0, 6, 7, 8]], dtype=torch.long)
    lengths = torch.tensor([5, 3], dtype=torch.long)
    flat_model.prefill_batch(input_ids, lengths, flat_blocks, [0, 1])
    regular_model.prefill_batch(input_ids, lengths, regular_blocks, [0, 1])

    regular_model._flat_decode_failed = True
    decode_ids = torch.tensor([[9], [10]], dtype=torch.long)
    flat_positions = flat_blocks.get_seq_lens_tensor([0, 1]).long().unsqueeze(1)
    regular_positions = regular_blocks.get_seq_lens_tensor([0, 1]).long().unsqueeze(1)
    flat_logits = flat_model.decode_step(decode_ids, flat_positions, flat_blocks, [0, 1])
    regular_logits = regular_model.decode_step(
        decode_ids,
        regular_positions,
        regular_blocks,
        [0, 1],
    )

    assert flat_model._flat_decode_ready
    assert flat_model._flat_is_gemma4
    assert torch.allclose(flat_logits, regular_logits, atol=1e-5, rtol=1e-5)


def test_gemma4_raw_decode_logits_feed_exact_capped_pipeline():
    torch.manual_seed(124)
    config = LlamaConfig.from_dict(_tiny_gemma4_config_dict())
    model = MegaGemmLlama(config).eval()
    model._refresh_gemma4_runtime_buffers(device="cpu", dtype=torch.float32)
    hidden = torch.randn(2, 1, config.hidden_size)

    with torch.inference_mode():
        raw_logits = model._decode_raw_logits_from_hidden(hidden)
        capped_logits = model._decode_logits_from_hidden(hidden)
        expected_capped = model._apply_final_logit_capping(raw_logits)

    assert torch.allclose(capped_logits, expected_capped, atol=1e-6, rtol=1e-6)


def test_gemma4_bf16_softcap_can_change_argmax_tie_breaking():
    raw_logits = torch.tensor([[90.0, 100.0]], dtype=torch.bfloat16)
    capped_logits = 30.0 * torch.tanh(raw_logits / 30.0)

    assert raw_logits.argmax(dim=-1).item() == 1
    assert capped_logits[0, 0] == capped_logits[0, 1]
    assert capped_logits.argmax(dim=-1).item() == 0


def test_fused_softcap_argmax_fallback_matches_bf16_contract_exactly():
    torch.manual_seed(125)
    logits = torch.randn(4, 257, dtype=torch.bfloat16).mul_(40.0)
    logits[0].fill_(-10.0)
    logits[0, :2] = torch.tensor([90.0, 100.0], dtype=torch.bfloat16)
    out = torch.empty((4,), dtype=torch.long)

    actual = logits_softcap_argmax(logits, 30.0, out_tokens=out)
    expected = (30.0 * torch.tanh(logits / 30.0)).argmax(dim=-1)

    assert actual.data_ptr() == out.data_ptr()
    assert torch.equal(actual, expected)
    assert actual[0].item() == 0


def test_gemma4_moe_flat_decode_matches_regular_decode():
    torch.manual_seed(321)
    config = LlamaConfig.from_dict(
        _tiny_gemma4_config_dict(
            enable_moe_block=True,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=8,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            use_double_wide_mlp=False,
        )
    )
    flat_model = MegaGemmLlama(config).eval()
    flat_model._refresh_gemma4_runtime_buffers(device="cpu", dtype=torch.float32)
    regular_model = copy.deepcopy(flat_model).eval()

    def make_block_manager():
        block_manager = BlockManager(
            num_layers=config.num_hidden_layers,
            num_blocks=16,
            block_size=4,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=torch.float32,
            device="cpu",
            kv_layer_indices=config.kv_cache_layer_indices,
            per_layer_num_kv_heads=config.per_layer_num_kv_heads,
            per_layer_head_dims=config.per_layer_head_dims,
            kv_layer_sources=_kv_sources(config),
        )
        block_manager.allocate_sequence(0, 8)
        block_manager.allocate_sequence(1, 8)
        return block_manager

    flat_blocks = make_block_manager()
    regular_blocks = make_block_manager()
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [0, 0, 6, 7, 8]], dtype=torch.long)
    lengths = torch.tensor([5, 3], dtype=torch.long)
    flat_model.prefill_batch(input_ids, lengths, flat_blocks, [0, 1])
    regular_model.prefill_batch(input_ids, lengths, regular_blocks, [0, 1])

    regular_model._flat_decode_failed = True
    decode_ids = torch.tensor([[9], [10]], dtype=torch.long)
    flat_positions = flat_blocks.get_seq_lens_tensor([0, 1]).long().unsqueeze(1)
    regular_positions = regular_blocks.get_seq_lens_tensor([0, 1]).long().unsqueeze(1)
    flat_logits = flat_model.decode_step(decode_ids, flat_positions, flat_blocks, [0, 1])
    regular_logits = regular_model.decode_step(
        decode_ids,
        regular_positions,
        regular_blocks,
        [0, 1],
    )

    assert flat_model._flat_decode_ready, flat_model._flat_decode_failed_reason
    assert flat_model._flat_is_gemma4
    assert all(layer_weights.is_moe for layer_weights in flat_model._flat_layer_weights)
    assert torch.allclose(flat_logits, regular_logits, atol=1e-5, rtol=1e-5)


def test_gemma4_int8_loader_quantizes_main_projections(tmp_path):
    snapshot = tmp_path / "gemma4-hf"
    _build_local_gemma4_snapshot(snapshot)
    model = load_from_hf(str(snapshot), dtype=torch.float16, device="cpu", quantize="int8")

    assert isinstance(model.layers[0].self_attn.q_proj, Int8Linear)
    assert isinstance(model.layers[0].self_attn.k_proj, Int8Linear)
    assert isinstance(model.layers[0].self_attn.o_proj, Int8Linear)
    assert isinstance(model.layers[0].mlp.gate_up_proj, Int8Linear)
    assert isinstance(model.layers[0].mlp.down_proj, Int8Linear)
    assert model.layers[3].self_attn.k_proj is None
    assert model.layers[3].self_attn.v_proj is None
    assert model.layers[0].per_layer_input_gate.weight.dtype == torch.float16
    assert model.layers[0].per_layer_projection.weight.dtype == torch.float16

    block_manager = BlockManager(
        num_layers=model.config.num_hidden_layers,
        num_blocks=16,
        block_size=4,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        dtype=torch.float16,
        device="cpu",
        kv_layer_indices=model.config.kv_cache_layer_indices,
        per_layer_num_kv_heads=model.config.per_layer_num_kv_heads,
        per_layer_head_dims=model.config.per_layer_head_dims,
        kv_layer_sources=_kv_sources(model.config),
    )
    block_manager.allocate_sequence(0, 8)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    lengths = torch.tensor([4], dtype=torch.long)
    logits = model.prefill_batch(input_ids, lengths, block_manager, [0])
    decode_ids = torch.tensor([[9]], dtype=torch.long)
    decode_positions = block_manager.get_seq_lens_tensor([0]).long().unsqueeze(1)
    decode_logits = model.decode_step(decode_ids, decode_positions, block_manager, [0])
    assert logits.shape == (1, 1, model.config.vocab_size)
    assert decode_logits.shape == (1, 1, model.config.vocab_size)


def test_gemma4_int8_flat_decode_matches_regular_decode():
    torch.manual_seed(321)
    config = LlamaConfig.from_dict(_tiny_gemma4_config_dict())
    flat_model = MegaGemmLlama(config).eval()
    flat_model._refresh_gemma4_runtime_buffers(device="cpu", dtype=torch.float32)

    for layer in flat_model.layers:
        attn = layer.self_attn
        mlp = layer.mlp
        attn.q_proj = Int8Linear.from_linear(attn.q_proj)
        if attn.k_proj is not None:
            attn.k_proj = Int8Linear.from_linear(attn.k_proj)
        if attn.v_proj is not None:
            attn.v_proj = Int8Linear.from_linear(attn.v_proj)
        attn.o_proj = Int8Linear.from_linear(attn.o_proj)
        mlp.gate_up_proj = Int8Linear.from_linear(mlp.gate_up_proj)
        mlp.down_proj = Int8Linear.from_linear(mlp.down_proj)

    regular_model = copy.deepcopy(flat_model).eval()

    def make_block_manager():
        block_manager = BlockManager(
            num_layers=config.num_hidden_layers,
            num_blocks=16,
            block_size=4,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=torch.float32,
            device="cpu",
            kv_layer_indices=config.kv_cache_layer_indices,
            per_layer_num_kv_heads=config.per_layer_num_kv_heads,
            per_layer_head_dims=config.per_layer_head_dims,
            kv_layer_sources=_kv_sources(config),
        )
        block_manager.allocate_sequence(0, 8)
        block_manager.allocate_sequence(1, 8)
        return block_manager

    flat_blocks = make_block_manager()
    regular_blocks = make_block_manager()
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [0, 0, 6, 7, 8]], dtype=torch.long)
    lengths = torch.tensor([5, 3], dtype=torch.long)
    flat_model.prefill_batch(input_ids, lengths, flat_blocks, [0, 1])
    regular_model.prefill_batch(input_ids, lengths, regular_blocks, [0, 1])

    regular_model._flat_decode_failed = True
    decode_ids = torch.tensor([[9], [10]], dtype=torch.long)
    flat_positions = flat_blocks.get_seq_lens_tensor([0, 1]).long().unsqueeze(1)
    regular_positions = regular_blocks.get_seq_lens_tensor([0, 1]).long().unsqueeze(1)
    flat_logits = flat_model.decode_step(decode_ids, flat_positions, flat_blocks, [0, 1])
    regular_logits = regular_model.decode_step(
        decode_ids,
        regular_positions,
        regular_blocks,
        [0, 1],
    )

    assert flat_model._flat_decode_ready
    assert flat_model._flat_is_gemma4
    assert flat_model._flat_layer_weights[0].q_int8_w is not None
    assert flat_model._flat_layer_weights[0].o_int8_w is not None
    assert torch.allclose(flat_logits, regular_logits, atol=1e-4, rtol=1e-4)


def test_gemma4_rope_cache_can_shrink_to_serving_context():
    cfg = _tiny_gemma4_config_dict()
    cfg["text_config"]["max_position_embeddings"] = 8192
    config = LlamaConfig.from_dict(cfg)
    model = MegaGemmLlama(config).eval()

    model.set_rope_cache_max_seq_len(32, device="cpu")

    assert model.cos_cache.shape[0] == 32
    assert model.layer_rope_caches[0][0].shape[0] == 32
    assert model.layer_rope_caches[1][0].shape[0] == 32


def test_gemma4_loader_prefix_normalization_and_text_filter():
    assert (
        _normalize_hf_weight_key("language_model.model.layers.0.self_attn.q_proj.weight")
        == "model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        _normalize_hf_weight_key("model.language_model.lm_head.weight")
        == "lm_head.weight"
    )
    assert _is_text_backbone_weight("language_model.model.embed_tokens_per_layer.weight")
    assert not _is_text_backbone_weight("vision_tower.encoder.layers.0.weight")


def test_deepfusion_geglu_fallback_matches_reference():
    torch.manual_seed(7)
    gate_up = torch.randn(3, 16, dtype=torch.float32)
    down_weight = torch.randn(5, 8, dtype=torch.float32)
    down_bias = torch.randn(5, dtype=torch.float32)

    out = deepfusion_swiglu_down(
        gate_up,
        down_weight,
        down_bias,
        activation="gelu_tanh",
    )

    gate = gate_up[:, :8]
    value = gate_up[:, 8:]
    expected = torch.nn.functional.linear(
        torch.nn.functional.gelu(gate, approximate="tanh") * value,
        down_weight,
        down_bias,
    )
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_fused_rmsnorm_linear_fallback_matches_reference():
    torch.manual_seed(11)
    hidden = torch.randn(4, 8, dtype=torch.float32)
    norm_weight = torch.randn(8, dtype=torch.float32)
    linear_weight = torch.randn(12, 8, dtype=torch.float32)
    linear_bias = torch.randn(12, dtype=torch.float32)

    out = fused_rmsnorm_linear(
        hidden,
        norm_weight,
        1e-6,
        linear_weight,
        linear_bias,
        norm_offset=False,
    )

    variance = hidden.float().pow(2).mean(-1, keepdim=True)
    normed = (hidden * torch.rsqrt(variance + 1e-6)) * norm_weight
    expected = torch.nn.functional.linear(normed.to(hidden.dtype), linear_weight, linear_bias)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_fused_rmsnorm_linear_row_override_is_local_to_the_call(monkeypatch):
    import importlib

    fused_module = importlib.import_module(
        "megagemm.kernels.fused_rmsnorm_linear"
    )
    monkeypatch.setattr(fused_module, "_CFG_FORCE_TRITON", False)
    monkeypatch.setattr(fused_module, "_CFG_SHAPE_GUARD", True)
    monkeypatch.setattr(fused_module, "_CFG_MAX_ROWS", 4)
    monkeypatch.setattr(fused_module, "_CFG_MIN_K", 1024)
    monkeypatch.setattr(fused_module, "_CFG_MIN_N", 1024)

    assert not fused_module.fused_rmsnorm_linear_prefers_triton_shape(
        1536,
        12288,
        8,
        mode="decode",
    )
    assert fused_module.fused_rmsnorm_linear_prefers_triton_shape(
        1536,
        12288,
        8,
        mode="decode",
        max_rows_override=8,
    )
    assert not fused_module.fused_rmsnorm_linear_prefers_triton_shape(
        1536,
        12288,
        9,
        mode="decode",
        max_rows_override=8,
    )


def test_eos_token_id_lists_are_normalized():
    assert _normalize_token_id_set([1, 106], None, torch.tensor([7])) == {1, 7, 106}


def test_paged_decode_sliding_window_matches_manual_reference():
    query = torch.tensor([[[0.5, -0.25]]], dtype=torch.float32)
    kv_cache = torch.zeros(1, 2, 1, 4, 2, dtype=torch.float32)
    keys = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]],
        dtype=torch.float32,
    )
    values = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 1.0]]],
        dtype=torch.float32,
    )
    kv_cache[0, 0] = keys
    kv_cache[0, 1] = values
    block_tables = torch.tensor([[0]], dtype=torch.int32)
    seq_lens = torch.tensor([4], dtype=torch.int32)

    out = paged_attention_decode(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        scale=1.0,
        sliding_window=2,
    )

    k_last = keys[:, -2:, :]
    v_last = values[:, -2:, :]
    scores = torch.bmm(query[0].unsqueeze(1), k_last.transpose(1, 2))
    weights = torch.softmax(scores, dim=-1)
    expected = torch.bmm(weights, v_last).squeeze(1).unsqueeze(0)
    assert torch.allclose(out, expected, atol=1e-6)


def test_paged_decode_sliding_window_crosses_block_boundary():
    query = torch.tensor([[[0.25, 0.75]]], dtype=torch.float32)
    kv_cache = torch.zeros(2, 2, 1, 4, 2, dtype=torch.float32)

    block0_k = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]], dtype=torch.float32)
    block0_v = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]], dtype=torch.float32)
    block1_k = torch.tensor([[[0.0, 2.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
    block1_v = torch.tensor([[[0.0, 4.0], [5.0, 5.0], [0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)

    kv_cache[0, 0] = block0_k
    kv_cache[0, 1] = block0_v
    kv_cache[1, 0] = block1_k
    kv_cache[1, 1] = block1_v

    block_tables = torch.tensor([[0, 1]], dtype=torch.int32)
    seq_lens = torch.tensor([6], dtype=torch.int32)

    out = paged_attention_decode(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        scale=1.0,
        sliding_window=3,
    )

    keys = torch.cat([block0_k[:, 3:, :], block1_k[:, :2, :]], dim=1)
    values = torch.cat([block0_v[:, 3:, :], block1_v[:, :2, :]], dim=1)
    scores = torch.bmm(query[0].unsqueeze(1), keys.transpose(1, 2))
    weights = torch.softmax(scores, dim=-1)
    expected = torch.bmm(weights, values).squeeze(1).unsqueeze(0)
    assert torch.allclose(out, expected, atol=1e-6)
