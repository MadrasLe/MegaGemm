from types import SimpleNamespace

from megagemm.models.runtime_policy import (
    policy_bool,
    policy_rows,
    resolve_runtime_policy,
)


def _config(layers, hidden, q_heads, kv_heads, model_type="gemma4_text"):
    return SimpleNamespace(
        model_type=model_type,
        num_hidden_layers=layers,
        hidden_size=hidden,
        num_attention_heads=q_heads,
        num_key_value_heads=kv_heads,
    )


def test_e2b_l4_policy_preserves_measured_multi_step_triton_path():
    policy = resolve_runtime_policy(_config(35, 1536, 8, 1), "NVIDIA L4")

    assert policy.name == "gemma4-e2b-l4"
    assert policy.prefer_triton_rmsnorm is True
    assert policy.decode_prefer_step is False
    assert policy.reuse_request_scheduler is False
    assert policy.reuse_request_scheduler_batches == (4, 8)
    assert policy.decode_cuda_graphs is True
    assert policy.decode_cuda_graph_batches == (4, 8)
    assert policy.decode_cuda_graph_prefer_step is True
    assert policy.decode_graph_multi_step_body is True
    assert policy.decode_graph_persistent_token_feedback is True
    assert policy.decode_graph_token_burst_by_batch == ((4, 8), (8, 16))
    assert policy.paged_decode_splits == 1
    assert policy.paged_decode_gqa2_direct is True
    assert policy.paged_decode_warps_h256 == 2
    assert policy.gemma4_dense_post_norm_chain is True
    assert policy.gemma4_e2b_h512_dense_bridge_pair is True
    assert policy.gemma4_e2b_b1_dense_bridge is True
    assert policy.gemma4_ple_conditioned_gelu_decode is False
    assert policy.gemma4_e2b_l4_sliding_prefill is True
    assert policy.gemma4_e2b_l4_full_prefill_expand is True
    assert policy.gemma4_e2b_b8_fused_attn_prepare is True
    assert policy.gemma4_e2b_b8_fused_attn_prepare_launches == (
        (521, 256, 4, 2, True),
        (521, 512, 8, 2, True),
        (2057, 256, 4, 2, True),
        (2057, 512, 4, 2, True),
    )
    assert policy.gemma4_e2b_b8_prefill_dense_bridge is True
    assert policy.gemma4_e2b_b8_prefill_dense_bridge_warps == ((2057, 4),)
    assert policy.gemma4_e2b_b8_prefill_gated_activation is True
    assert policy.gemma4_e2b_b8_prefill_gated_activation_blocks == (
        (521, 256),
        (2057, 512),
    )
    assert policy.gemma4_bf16_fused_gateup_rows == ()
    assert policy.gemma4_bf16_deepfusion_rows == ()
    assert policy.gemma4_bf16_cublas_gateup_rows == (8,)
    assert policy.gemma4_bf16_cublas_down_rows == (8,)


def test_e4b_l4_policy_preserves_measured_step_and_reuse_path():
    policy = resolve_runtime_policy(_config(42, 2560, 8, 2), "NVIDIA L4")

    assert policy.name == "gemma4-e4b-l4"
    assert policy.prefer_triton_rmsnorm is False
    assert policy.decode_prefer_step is True
    assert policy.reuse_request_scheduler is True
    assert policy.reuse_request_scheduler_batches == ()
    assert policy.decode_cuda_graphs is False
    assert policy.decode_cuda_graph_batches == ()
    assert policy.decode_cuda_graph_prefer_step is False
    assert policy.decode_graph_multi_step_body is False
    assert policy.decode_graph_persistent_token_feedback is False
    assert policy.decode_graph_token_burst_by_batch == ()
    assert policy.paged_decode_splits == 0
    assert policy.paged_decode_gqa2_direct is False
    assert policy.paged_decode_warps_h256 == 0
    assert policy.gemma4_dense_post_norm_chain is False
    assert policy.gemma4_e2b_h512_dense_bridge_pair is False
    assert policy.gemma4_e2b_b1_dense_bridge is False
    assert policy.gemma4_ple_conditioned_gelu_decode is False
    assert policy.gemma4_e2b_l4_sliding_prefill is False
    assert policy.gemma4_e2b_l4_full_prefill_expand is False
    assert policy.gemma4_e2b_b8_fused_attn_prepare is False
    assert policy.gemma4_e2b_b8_fused_attn_prepare_launches == ()
    assert policy.gemma4_e2b_b8_prefill_dense_bridge is False
    assert policy.gemma4_e2b_b8_prefill_dense_bridge_warps == ()
    assert policy.gemma4_e2b_b8_prefill_gated_activation is False
    assert policy.gemma4_e2b_b8_prefill_gated_activation_blocks == ()
    assert policy.gemma4_bf16_fused_gateup_rows == ()
    assert policy.gemma4_bf16_deepfusion_rows == ()
    assert policy.gemma4_bf16_cublas_gateup_rows == ()
    assert policy.gemma4_bf16_cublas_down_rows == ()


def test_gemma4_policy_is_not_promoted_to_unmeasured_hardware():
    policy = resolve_runtime_policy(_config(35, 1536, 8, 1), "NVIDIA A100")

    assert policy.name == "gemma4-generic"
    assert policy.decode_prefer_step is False
    assert policy.reuse_request_scheduler is False
    assert policy.reuse_request_scheduler_batches == ()
    assert policy.decode_cuda_graphs is False
    assert policy.decode_cuda_graph_batches == ()
    assert policy.decode_cuda_graph_prefer_step is False
    assert policy.decode_graph_multi_step_body is False
    assert policy.decode_graph_persistent_token_feedback is False
    assert policy.decode_graph_token_burst_by_batch == ()
    assert policy.paged_decode_splits == 0
    assert policy.paged_decode_gqa2_direct is False
    assert policy.paged_decode_warps_h256 == 0
    assert policy.gemma4_dense_post_norm_chain is False
    assert policy.gemma4_e2b_h512_dense_bridge_pair is False
    assert policy.gemma4_e2b_b1_dense_bridge is False
    assert policy.gemma4_ple_conditioned_gelu_decode is False
    assert policy.gemma4_e2b_l4_sliding_prefill is False
    assert policy.gemma4_e2b_l4_full_prefill_expand is False
    assert policy.gemma4_e2b_b8_fused_attn_prepare is False
    assert policy.gemma4_e2b_b8_fused_attn_prepare_launches == ()
    assert policy.gemma4_e2b_b8_prefill_dense_bridge is False
    assert policy.gemma4_e2b_b8_prefill_dense_bridge_warps == ()
    assert policy.gemma4_e2b_b8_prefill_gated_activation is False
    assert policy.gemma4_e2b_b8_prefill_gated_activation_blocks == ()
    assert policy.gemma4_bf16_fused_gateup_rows == ()
    assert policy.gemma4_bf16_deepfusion_rows == ()
    assert policy.gemma4_bf16_cublas_gateup_rows == ()
    assert policy.gemma4_bf16_cublas_down_rows == ()


def test_explicit_environment_flag_overrides_model_policy(monkeypatch):
    model = SimpleNamespace(
        runtime_policy=resolve_runtime_policy(
            _config(42, 2560, 8, 2), "NVIDIA L4"
        )
    )

    assert policy_bool(
        model,
        "MEGAGEMM_DECODE_PREFER_STEP",
        "decode_prefer_step",
    ) is True

    monkeypatch.setenv("MEGAGEMM_DECODE_PREFER_STEP", "0")
    assert policy_bool(
        model,
        "MEGAGEMM_DECODE_PREFER_STEP",
        "decode_prefer_step",
    ) is False


def test_e2b_b4_b8_graph_policy_is_batch_scoped_and_env_overridable(monkeypatch):
    monkeypatch.delenv("MEGAGEMM_DECODE_CUDA_GRAPHS", raising=False)
    model = SimpleNamespace(
        runtime_policy=resolve_runtime_policy(
            _config(35, 1536, 8, 1), "NVIDIA L4"
        )
    )

    assert policy_bool(
        model,
        "MEGAGEMM_DECODE_CUDA_GRAPHS",
        "decode_cuda_graphs",
    ) is True
    assert policy_rows(
        model,
        "MEGAGEMM_DECODE_CUDA_GRAPHS",
        "decode_cuda_graph_batches",
    ) == (4, 8)

    monkeypatch.setenv("MEGAGEMM_DECODE_CUDA_GRAPHS", "0")
    assert policy_bool(
        model,
        "MEGAGEMM_DECODE_CUDA_GRAPHS",
        "decode_cuda_graphs",
    ) is False
    assert policy_rows(
        model,
        "MEGAGEMM_DECODE_CUDA_GRAPHS",
        "decode_cuda_graph_batches",
    ) == ()


def test_e2b_request_scheduler_reuse_is_promoted_for_b4_and_b8(monkeypatch):
    from megagemm.engine.engine import _request_scheduler_reuse_for_batch

    monkeypatch.delenv("MEGAGEMM_REUSE_REQUEST_SCHEDULER", raising=False)
    model = SimpleNamespace(
        runtime_policy=resolve_runtime_policy(
            _config(35, 1536, 8, 1), "NVIDIA L4"
        )
    )

    assert not _request_scheduler_reuse_for_batch(model, 1)
    assert not _request_scheduler_reuse_for_batch(model, 2)
    assert _request_scheduler_reuse_for_batch(model, 4)
    assert _request_scheduler_reuse_for_batch(model, 8)

    monkeypatch.setenv("MEGAGEMM_REUSE_REQUEST_SCHEDULER", "1")
    assert _request_scheduler_reuse_for_batch(model, 1)
    monkeypatch.setenv("MEGAGEMM_REUSE_REQUEST_SCHEDULER", "0")
    assert not _request_scheduler_reuse_for_batch(model, 4)


def test_explicit_environment_can_disable_promoted_e2b_dense_chain(monkeypatch):
    model = SimpleNamespace(
        runtime_policy=resolve_runtime_policy(
            _config(35, 1536, 8, 1), "NVIDIA L4"
        )
    )

    assert policy_bool(
        model,
        "MEGAGEMM_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE",
        "gemma4_dense_post_norm_chain",
    ) is True

    monkeypatch.setenv("MEGAGEMM_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE", "0")
    assert policy_bool(
        model,
        "MEGAGEMM_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE",
        "gemma4_dense_post_norm_chain",
    ) is False


def test_explicit_environment_can_disable_promoted_e2b_h512_bridge_pair(
    monkeypatch,
):
    model = SimpleNamespace(
        runtime_policy=resolve_runtime_policy(
            _config(35, 1536, 8, 1), "NVIDIA L4"
        )
    )

    assert policy_bool(
        model,
        "MEGAGEMM_GEMMA4_DENSE_ATTN_MLP_BRIDGE_DECODE",
        "gemma4_e2b_h512_dense_bridge_pair",
    ) is True

    monkeypatch.setenv("MEGAGEMM_GEMMA4_DENSE_ATTN_MLP_BRIDGE_DECODE", "0")
    assert policy_bool(
        model,
        "MEGAGEMM_GEMMA4_DENSE_ATTN_MLP_BRIDGE_DECODE",
        "gemma4_e2b_h512_dense_bridge_pair",
    ) is False


def test_explicit_environment_can_disable_promoted_e2b_sliding_prefill(
    monkeypatch,
):
    model = SimpleNamespace(
        runtime_policy=resolve_runtime_policy(
            _config(35, 1536, 8, 1), "NVIDIA L4"
        )
    )

    assert policy_bool(
        model,
        "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_PREFILL",
        "gemma4_e2b_l4_sliding_prefill",
    ) is True

    monkeypatch.setenv("MEGAGEMM_GEMMA4_E2B_L4_SLIDING_PREFILL", "0")
    assert policy_bool(
        model,
        "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_PREFILL",
        "gemma4_e2b_l4_sliding_prefill",
    ) is False


def test_explicit_environment_can_disable_promoted_e2b_full_prefill_expand(
    monkeypatch,
):
    model = SimpleNamespace(
        runtime_policy=resolve_runtime_policy(
            _config(35, 1536, 8, 1), "NVIDIA L4"
        )
    )

    assert policy_bool(
        model,
        "MEGAGEMM_GEMMA4_E2B_L4_FULL_PREFILL_EXPAND",
        "gemma4_e2b_l4_full_prefill_expand",
    ) is True

    monkeypatch.setenv("MEGAGEMM_GEMMA4_E2B_L4_FULL_PREFILL_EXPAND", "0")
    assert policy_bool(
        model,
        "MEGAGEMM_GEMMA4_E2B_L4_FULL_PREFILL_EXPAND",
        "gemma4_e2b_l4_full_prefill_expand",
    ) is False


def test_explicit_environment_can_disable_promoted_e2b_prefill_activation(
    monkeypatch,
):
    model = SimpleNamespace(
        runtime_policy=resolve_runtime_policy(
            _config(35, 1536, 8, 1), "NVIDIA L4"
        )
    )

    assert policy_bool(
        model,
        "MEGAGEMM_GEMMA4_E2B_B8_PREFILL_GATED_ACTIVATION",
        "gemma4_e2b_b8_prefill_gated_activation",
    ) is True

    monkeypatch.setenv(
        "MEGAGEMM_GEMMA4_E2B_B8_PREFILL_GATED_ACTIVATION",
        "0",
    )
    assert policy_bool(
        model,
        "MEGAGEMM_GEMMA4_E2B_B8_PREFILL_GATED_ACTIVATION",
        "gemma4_e2b_b8_prefill_gated_activation",
    ) is False


def test_explicit_environment_can_disable_promoted_e2b_prefill_dense_bridge(
    monkeypatch,
):
    model = SimpleNamespace(
        runtime_policy=resolve_runtime_policy(
            _config(35, 1536, 8, 1), "NVIDIA L4"
        )
    )

    assert policy_bool(
        model,
        "MEGAGEMM_GEMMA4_E2B_B8_PREFILL_DENSE_BRIDGE",
        "gemma4_e2b_b8_prefill_dense_bridge",
    ) is True

    monkeypatch.setenv(
        "MEGAGEMM_GEMMA4_E2B_B8_PREFILL_DENSE_BRIDGE",
        "0",
    )
    assert policy_bool(
        model,
        "MEGAGEMM_GEMMA4_E2B_B8_PREFILL_DENSE_BRIDGE",
        "gemma4_e2b_b8_prefill_dense_bridge",
    ) is False


def test_e2b_l4_batch8_cublas_mlp_rows_are_policy_scoped_and_env_overridable(
    monkeypatch,
):
    model = SimpleNamespace(
        runtime_policy=resolve_runtime_policy(
            _config(35, 1536, 8, 1), "NVIDIA L4"
        )
    )

    assert policy_rows(
        model,
        "MEGAGEMM_GEMMA4_FORCE_FUSED_GATEUP_USE",
        "gemma4_bf16_cublas_gateup_rows",
    ) == (8,)
    assert policy_rows(
        model,
        "MEGAGEMM_GEMMA4_FORCE_DEEPFUSION_USE",
        "gemma4_bf16_cublas_down_rows",
    ) == (8,)

    monkeypatch.setenv("MEGAGEMM_GEMMA4_FORCE_FUSED_GATEUP_USE", "0")
    monkeypatch.setenv("MEGAGEMM_GEMMA4_FORCE_DEEPFUSION_USE", "0")
    assert policy_rows(
        model,
        "MEGAGEMM_GEMMA4_FORCE_FUSED_GATEUP_USE",
        "gemma4_bf16_cublas_gateup_rows",
    ) == ()
    assert policy_rows(
        model,
        "MEGAGEMM_GEMMA4_FORCE_DEEPFUSION_USE",
        "gemma4_bf16_cublas_down_rows",
    ) == ()
