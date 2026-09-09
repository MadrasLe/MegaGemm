from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from megagemm.kernels import paged_attention
from megagemm.kernels.rmsnorm_triton import (
    rmsnorm_triton,
    rmsnorm_triton_attn_residual_dense,
)


ROOT = Path(__file__).resolve().parents[1]


def _fake_cuda_tensor(shape: tuple[int, ...], ndim: int):
    return SimpleNamespace(
        shape=shape,
        ndim=ndim,
        dtype=torch.bfloat16,
        is_cuda=True,
        device=torch.device("cuda"),
    )


def test_e2b_l4_h512_grouped_topology_is_exact_and_opt_in(monkeypatch):
    monkeypatch.setattr(paged_attention, "_HAS_TRITON", True)
    monkeypatch.setattr(
        paged_attention,
        "_cuda_device_info",
        lambda _device=None: ((8, 9), "NVIDIA L4", 58),
    )
    monkeypatch.setattr(
        paged_attention,
        "_GROUPED_SEGMENTED_DECODE_DISABLED",
        False,
    )
    monkeypatch.delenv(
        "MEGAGEMM_GEMMA4_E2B_L4_H512_GROUPED_ATTN_DECODE",
        raising=False,
    )
    query = _fake_cuda_tensor((8, 8, 512), 3)
    cache = _fake_cuda_tensor((1152, 2, 1, 16, 512), 5)
    tables = _fake_cuda_tensor((8, 144), 2)

    select = paged_attention._grouped_segmented_decode_topology
    assert select(query, cache, tables, sliding_window=None) is None
    assert (
        select(query, cache, tables, sliding_window=None, force=True)
        == "e2b_l4_full_h512_gqa8"
    )
    assert (
        select(
            query,
            cache,
            tables,
            sliding_window=None,
            e2b_l4_h512_policy_enabled=True,
        )
        == "e2b_l4_full_h512_gqa8"
    )
    monkeypatch.setenv(
        "MEGAGEMM_GEMMA4_E2B_L4_H512_GROUPED_ATTN_DECODE",
        "1",
    )
    assert (
        select(query, cache, tables, sliding_window=None)
        == "e2b_l4_full_h512_gqa8"
    )
    assert (
        select(
            _fake_cuda_tensor((16, 8, 512), 3),
            cache,
            _fake_cuda_tensor((16, 144), 2),
            sliding_window=None,
        )
        is None
    )
    assert select(query, cache, tables, sliding_window=512) is None


def test_e2b_l4_b1_h512_grouped_topology_has_independent_policy(monkeypatch):
    monkeypatch.setattr(paged_attention, "_HAS_TRITON", True)
    monkeypatch.setattr(
        paged_attention,
        "_cuda_device_info",
        lambda _device=None: ((8, 9), "NVIDIA L4", 58),
    )
    monkeypatch.setattr(
        paged_attention,
        "_GROUPED_SEGMENTED_DECODE_DISABLED",
        False,
    )
    monkeypatch.delenv(
        "MEGAGEMM_GEMMA4_E2B_L4_B1_H512_GROUPED_ATTN_DECODE",
        raising=False,
    )
    monkeypatch.setenv(
        "MEGAGEMM_GEMMA4_E2B_L4_H512_GROUPED_ATTN_DECODE",
        "1",
    )
    query = _fake_cuda_tensor((1, 8, 512), 3)
    cache = _fake_cuda_tensor((144, 2, 1, 16, 512), 5)
    tables = _fake_cuda_tensor((1, 144), 2)
    select = paged_attention._grouped_segmented_decode_topology

    # The existing B8 switch must never activate the B1 shape.
    assert select(query, cache, tables, sliding_window=None) is None
    monkeypatch.setenv(
        "MEGAGEMM_GEMMA4_E2B_L4_B1_H512_GROUPED_ATTN_DECODE",
        "1",
    )
    assert (
        select(query, cache, tables, sliding_window=None)
        == "e2b_l4_b1_full_h512_gqa8"
    )
    assert select(query, cache, tables, sliding_window=512) is None


def test_e2b_l4_b1_h512_launch_geometry_is_independent(monkeypatch):
    topology = "e2b_l4_b1_full_h512_gqa8"
    monkeypatch.delenv(
        "MEGAGEMM_GEMMA4_E2B_L4_B1_H512_ATTN_SEGMENTS",
        raising=False,
    )
    monkeypatch.delenv(
        "MEGAGEMM_GEMMA4_E2B_L4_B1_H512_ATTN_TILE",
        raising=False,
    )
    assert paged_attention._grouped_segmented_decode_num_segments(topology, 2304) == 8
    assert paged_attention._grouped_segmented_decode_tile_size(topology, 2304) == 16
    monkeypatch.setenv(
        "MEGAGEMM_GEMMA4_E2B_L4_B1_H512_ATTN_SEGMENTS",
        "16",
    )
    monkeypatch.setenv(
        "MEGAGEMM_GEMMA4_E2B_L4_B1_H512_ATTN_TILE",
        "32",
    )
    assert paged_attention._grouped_segmented_decode_num_segments(topology, 2304) == 16
    assert paged_attention._grouped_segmented_decode_tile_size(topology, 2304) == 32


def test_e2b_l4_h512_segment_and_tile_candidates_are_overrideable(monkeypatch):
    topology = "e2b_l4_full_h512_gqa8"
    monkeypatch.delenv(
        "MEGAGEMM_GEMMA4_E2B_L4_H512_ATTN_SEGMENTS",
        raising=False,
    )
    monkeypatch.delenv(
        "MEGAGEMM_GEMMA4_E2B_L4_H512_ATTN_TILE",
        raising=False,
    )
    assert paged_attention._grouped_segmented_decode_num_segments(topology, 2304) == 32
    assert paged_attention._grouped_segmented_decode_tile_size(topology, 2304) == 16
    monkeypatch.setenv("MEGAGEMM_GEMMA4_E2B_L4_H512_ATTN_SEGMENTS", "16")
    monkeypatch.setenv("MEGAGEMM_GEMMA4_E2B_L4_H512_ATTN_TILE", "32")
    assert paged_attention._grouped_segmented_decode_num_segments(topology, 2304) == 16
    assert paged_attention._grouped_segmented_decode_tile_size(topology, 2304) == 32


def test_dense_attention_mlp_bridge_matches_staged_reference_and_aliases():
    generator = torch.Generator().manual_seed(29)
    attn = torch.randn(8, 1536, generator=generator, dtype=torch.bfloat16)
    residual = torch.randn(8, 1536, generator=generator, dtype=torch.bfloat16)
    post_weight = torch.randn(1536, generator=generator, dtype=torch.bfloat16)
    pre_ff_weight = torch.randn(1536, generator=generator, dtype=torch.bfloat16)
    eps = 1.0e-6

    post = rmsnorm_triton(attn, post_weight, eps, False)
    expected_hidden = (residual + post).to(torch.bfloat16)
    expected_pre_ff = rmsnorm_triton(
        expected_hidden,
        pre_ff_weight,
        eps,
        False,
    )
    hidden_out = residual.clone()
    pre_ff_out = torch.empty_like(residual)
    hidden, pre_ff = rmsnorm_triton_attn_residual_dense(
        attn,
        hidden_out,
        post_weight,
        pre_ff_weight,
        eps,
        out_hidden=hidden_out,
        pre_ff_out=pre_ff_out,
    )
    assert hidden.data_ptr() == hidden_out.data_ptr()
    assert pre_ff.data_ptr() == pre_ff_out.data_ptr()
    assert torch.equal(hidden, expected_hidden)
    assert torch.equal(pre_ff, expected_pre_ff)


def test_e2b_candidate_gates_and_native_binding_are_present():
    attention_gate = (
        ROOT / "benchmarks" / "run_gemma4_e2b_h512_attention_gate.py"
    ).read_text(encoding="utf-8")
    bridge_gate = (
        ROOT
        / "benchmarks"
        / "run_gemma4_e2b_dense_attn_mlp_bridge_gate.py"
    ).read_text(encoding="utf-8")
    cublaslt_gate = (
        ROOT / "benchmarks" / "run_gemma4_e2b_cublaslt_gateup_sweep.py"
    ).read_text(encoding="utf-8")
    colab = (
        ROOT
        / "benchmarks"
        / "run_gemma4_e2b_optimization_microgates_colab.sh"
    ).read_text(encoding="utf-8")
    full_model_gate = (
        ROOT
        / "benchmarks"
        / "run_gemma4_e2b_h512_bridge_full_model_gate.py"
    ).read_text(encoding="utf-8")
    full_model_colab = (
        ROOT
        / "benchmarks"
        / "run_gemma4_e2b_h512_bridge_full_model_colab.sh"
    ).read_text(encoding="utf-8")
    b1_frontier_gate = (
        ROOT
        / "benchmarks"
        / "run_gemma4_e2b_b1_decode_frontier_gate.py"
    ).read_text(encoding="utf-8")
    b1_frontier_colab = (
        ROOT
        / "benchmarks"
        / "run_gemma4_e2b_b1_decode_frontier_colab.sh"
    ).read_text(encoding="utf-8")
    binding = (ROOT / "pytorch_binding" / "binding.cpp").read_text(
        encoding="utf-8"
    )
    native = (ROOT / "src" / "mlp_prefill_kernel.cu").read_text(
        encoding="utf-8"
    )
    llama = (ROOT / "megagemm" / "models" / "llama.py").read_text(
        encoding="utf-8"
    )

    assert "e2b_l4_full_h512_gqa8" in attention_gate
    assert "TEST_FULL_MODEL" in attention_gate
    assert "single_launch_dense_bridge" in bridge_gate
    assert "TEST_FULL_MODEL" in bridge_gate
    assert "N12288" in cublaslt_gate and "N24576" in cublaslt_gate
    assert '"apply_change": False' in cublaslt_gate
    assert "cublaslt_bf16_linear_cuda" in binding
    assert "cublasLtMatmulAlgoGetHeuristic" in native
    assert "MEGAGEMM_GEMMA4_E2B_CUBLASLT_GATEUP_DECODE" in llama
    assert "MEGAGEMM_GEMMA4_DENSE_ATTN_MLP_BRIDGE_DECODE" in llama
    assert "MEGAGEMM_GEMMA4_E2B_L4_B1_DENSE_ATTN_MLP_BRIDGE_DECODE" in llama
    assert '"gemma4_e2b_b1_dense_bridge"' in llama
    assert "timing_events is None\n                and not lw.is_moe" not in llama
    assert '"attn_mlp_bridge"' in llama
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in colab
    assert "git pull" not in colab
    assert "pip install" not in colab
    assert "MEGAGEMM_BUILD_ONLY_RMSNORM_CUDA=1" in colab
    assert "python setup.py build_ext --inplace" in colab
    assert 'CASES: tuple[tuple[str, bool, bool], ...]' in full_model_gate
    assert '("baseline", False, False)' in full_model_gate
    assert '("h512_attention", True, False)' in full_model_gate
    assert '("dense_bridge", False, True)' in full_model_gate
    assert '("combined", True, True)' in full_model_gate
    assert '("baseline_recheck", False, False)' in full_model_gate
    assert '"segments": 32' in full_model_gate
    assert '"tile_size": 16' in full_model_gate
    assert "generated-token digests differ" in full_model_gate
    assert '"MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID"' in full_model_gate
    assert 'scheduler.get("benchmark_forced_token_id", -1)' in full_model_gate
    assert 'if args.strict_exit and decision == "INVALID_GATE"' in full_model_gate
    assert "interpolated_baselines" in full_model_gate
    assert "speedups_vs_interpolated_baseline" in full_model_gate
    assert "PROMOTE_H512_AND_BRIDGE_AS_PAIR" in full_model_gate
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in full_model_colab
    assert "git pull" not in full_model_colab
    assert "vllm" not in full_model_colab.lower()
    assert "pip install -q" in full_model_colab
    assert "pip install -q -e" not in full_model_colab
    assert "setup.py build_ext" not in full_model_colab
    assert 'DecodeCase("core_factorial", "production")' in b1_frontier_gate
    assert '"bridge_b1", bridge=True' in b1_frontier_gate
    assert '"forced_fused_rms_lm_head"' in b1_frontier_gate
    assert '"bridge_plus_fused_rms_lm_head"' in b1_frontier_gate
    assert '"e2b_l4_b1_full_h512_gqa8"' in b1_frontier_gate
    assert '"fused_large_gateup"' in b1_frontier_gate
    assert '"deepfusion_large_down"' in b1_frontier_gate
    assert '"model_loads": 1' in b1_frontier_gate
    assert b1_frontier_gate.count("InferenceEngine(") == 1
    assert "subprocess" not in b1_frontier_gate
    assert "and int(out_features) == 24576" in llama
    assert "and i_dim == 12288" in llama
    assert "and h_dim == 1536" in llama
    assert "_gemma4_flat_b1_large_gateup_hits" in llama
    assert "_gemma4_flat_b1_large_down_hits" in llama
    assert "git pull" not in b1_frontier_colab
    assert "pip install -q -e" not in b1_frontier_colab
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in b1_frontier_colab
