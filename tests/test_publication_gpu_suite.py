import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmarks" / "run_publication_gpu_suite.py"


def load_runner():
    name = "publication_gpu_suite_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load publication GPU suite")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_dense_gemma4_models_select_distinct_fast_profiles_automatically():
    runner = load_runner()

    assert (
        runner.resolve_megagemm_profile("auto", "google/gemma-4-E2B-it")
        == "gemma4-e2b-fast"
    )
    assert (
        runner.resolve_megagemm_profile("auto", "google/gemma-4-E4B-it")
        == "gemma4-e4b-fast"
    )
    assert runner.resolve_megagemm_profile("auto", "Qwen/Qwen2.5-3B") == "none"
    assert runner.resolve_megagemm_profile("none", "google/gemma-4-E4B-it") == "none"


def test_fast_profile_is_scoped_and_model_specific(monkeypatch):
    runner = load_runner()
    monkeypatch.delenv("MEGAGEMM_DECODE_CUDA_GRAPHS", raising=False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("MEGAGEMM_GEMMA4_FORCE_FUSED_GATEUP_USE", "1")
    monkeypatch.setenv("MEGAGEMM_GEMMA4_FORCE_DEEPFUSION_USE", "1")

    e4b_env = runner.child_environment(
        runner.VARIANTS["megagemm-bf16"],
        "gemma4-e4b-fast",
        "google/gemma-4-E4B-it",
    )
    e2b_env = runner.child_environment(
        runner.VARIANTS["megagemm-bf16"],
        "gemma4-e2b-fast",
        "google/gemma-4-E2B-it",
    )
    vllm_env = runner.child_environment(
        runner.VARIANTS["vllm-bf16"],
        "gemma4-e4b-fast",
        "google/gemma-4-E4B-it",
    )

    assert e4b_env["MEGAGEMM_FLAT_DECODE"] == "1"
    assert e4b_env["MEGAGEMM_DISABLE_CUDA_RMSNORM"] == "0"
    assert e4b_env["MEGAGEMM_DECODE_CUDA_GRAPHS"] == "0"
    assert e4b_env["MEGAGEMM_DECODE_PREFER_STEP"] == "1"
    assert e4b_env["MEGAGEMM_REUSE_REQUEST_SCHEDULER"] == "1"
    assert e4b_env["MEGAGEMM_GEMMA4_FUSED_GATEUP_DECODE"] == "1"
    assert e4b_env["MEGAGEMM_GEMMA4_DEEPFUSION_MLP_DECODE"] == "1"
    assert "CUBLAS_WORKSPACE_CONFIG" not in e4b_env
    assert "MEGAGEMM_GEMMA4_FORCE_FUSED_GATEUP_USE" not in e4b_env
    assert "MEGAGEMM_GEMMA4_FORCE_DEEPFUSION_USE" not in e4b_env
    assert e2b_env["MEGAGEMM_DISABLE_CUDA_RMSNORM"] == "1"
    assert "MEGAGEMM_DECODE_PREFER_STEP" not in e2b_env
    assert "MEGAGEMM_DECODE_CUDA_GRAPHS" not in e2b_env
    assert "MEGAGEMM_DECODE_CUDA_GRAPHS_PREFER_STEP" not in e2b_env
    assert "MEGAGEMM_REUSE_REQUEST_SCHEDULER" not in e2b_env
    assert "MEGAGEMM_GEMMA4_FORCE_FUSED_GATEUP_USE" not in e2b_env
    assert "MEGAGEMM_GEMMA4_FORCE_DEEPFUSION_USE" not in e2b_env
    assert "MEGAGEMM_DECODE_CUDA_GRAPHS" not in vllm_env


def test_e2b_audit_accepts_only_the_promoted_b4_graph_and_reuse(tmp_path):
    runner = load_runner()
    raw_path = tmp_path / "e2b_b4.jsonl"
    row = {
        "ok": True,
        "model": "google/gemma-4-E2B-it",
        "hardware_label": "1xl4",
        "dtype": "bf16",
        "scenario": "long_context",
        "batch_size": 4,
        "prompt_tokens_requested_per_request": 128,
        "output_tps": 115.17,
        "model_topology": {
            "num_hidden_layers": 35,
            "hidden_size": 1536,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 20,
            "kv_cache_layers": 15,
            "sliding_attention_layers": 28,
            "full_attention_layers": 7,
        },
        "decode_runtime_stats": {
            "flat_decode_ready": True,
            "flat_decode_failed": False,
            "gemma4_dense_post_norm_chain_decode_enabled": True,
            "gemma4_dense_post_norm_chain_decode_hits": 4445,
        },
        "scheduler_stats": {
            "decode_cuda_graphs": {
                "enabled": True,
                "captures": 1,
                "replays": 127,
                "failures": 0,
                "request_scheduler_reuse_count": 3,
                "decode_cuda_graph_policy_batches": [4],
                "request_scheduler_reuse_policy_batches": [4],
            },
            "decode_execution": {
                "prefer_step": False,
                "decode_step_batches": 128,
                "multi_step_batches": 0,
            },
        },
    }
    raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = runner.audit_gemma4_dense_fast_path(
        raw_path, "gemma4-e2b-fast"
    )

    assert report["status"] == "passed"
    assert report["required"]["decode_mode"] == "batch_scoped_cuda_graph"
    assert report["required"]["decode_cuda_graph_policy_batches"] == [4]
    assert report["required"]["request_scheduler_reuse_policy_batches"] == [4]

    row["batch_size"] = 2
    raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    rejected = runner.audit_gemma4_dense_fast_path(
        raw_path, "gemma4-e2b-fast"
    )
    assert rejected["status"] == "failed"
    assert any("outside promoted batches" in error for error in rejected["errors"])


def test_checkpoint_profile_cannot_be_applied_to_the_other_scale():
    runner = load_runner()

    try:
        runner.profile_environment(
            "gemma4-e4b-fast", "google/gemma-4-E2B-it"
        )
    except ValueError as exc:
        assert "not valid" in str(exc)
    else:
        raise AssertionError("E4B profile was accepted for E2B")


def test_fast_path_audit_proves_runtime_and_reports_selected_kernels(tmp_path):
    runner = load_runner()
    raw_path = tmp_path / "mega.jsonl"
    row = {
        "ok": True,
        "model_topology": {
            "num_hidden_layers": 42,
            "hidden_size": 2560,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "num_kv_shared_layers": 18,
            "kv_cache_layers": 24,
            "sliding_attention_layers": 35,
            "full_attention_layers": 7,
        },
        "decode_runtime_stats": {
            "flat_decode_ready": True,
            "flat_decode_failed": False,
            "gemma4_flat_fused_qkv_layers": 42,
            "fused_rope_attn_total_hits": 42,
            "gemma4_fused_qkv_prefill_hits": 3,
            "gemma4_fused_attn_prepare_hits": 2,
            "gemma4_batch_prefill_vectorized_kv_hits": 7,
            "gemma4_e2b_l4_sliding_prefill_hits": 28,
            "gemma4_flat_fused_gateup_hits": 80,
            "gemma4_flat_deepfusion_hits": 70,
            "gemma4_fused_dual_ffn_norm_prefill_hits": 4,
            "gemma4_fused_add_ffn_norm_prefill_hits": 5,
            "gemma4_fused_post_ffn_norm_prefill_hits": 6,
            "paged_decode_runtime": {
                "generic_direct_hits": 100,
                "gqa2_direct_hits": 0,
            },
            "fused_rmsnorm_lm_head_argmax_use": True,
            "fused_lm_head_argmax_use": False,
        },
        "scheduler_stats": {
            "decode_cuda_graphs": {
                "enabled": False,
                "captures": 0,
                "replays": 0,
                "failures": 0,
                "request_scheduler_reuse_count": 3,
            },
            "decode_execution": {
                "prefer_step": True,
                "decode_step_batches": 127,
                "multi_step_batches": 0,
            },
        },
    }
    raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = runner.audit_gemma4_dense_fast_path(raw_path, "gemma4-e4b-fast")

    assert report["status"] == "passed"
    assert report["required"]["decode_mode"] == "flat_single_step_eager"
    assert report["required"]["decode_cuda_graph_replays"] == 0
    assert report["required"]["decode_step_batches"] == 127
    assert report["selected_kernel_counters"]["gemma4_flat_fused_qkv_layers"] == 42
    assert report["selected_kernel_counters"]["gemma4_flat_fused_gateup"] == 80
    assert report["selected_kernel_counters"]["gemma4_flat_deepfusion"] == 70
    assert (
        report["selected_kernel_counters"]["gemma4_e2b_l4_sliding_prefill"]
        == 28
    )
    assert report["selected_kernel_counters"]["paged_attention_generic_direct"] == 100
    assert report["selected_lm_head"]["fused_rmsnorm_lm_head_argmax"] is True


def test_fast_path_audit_rejects_eager_fallback(tmp_path):
    runner = load_runner()
    raw_path = tmp_path / "mega.jsonl"
    row = {
        "ok": True,
        "model_topology": {
            "num_hidden_layers": 42,
            "hidden_size": 2560,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "num_kv_shared_layers": 18,
            "kv_cache_layers": 24,
            "sliding_attention_layers": 35,
            "full_attention_layers": 7,
        },
        "decode_runtime_stats": {
            "flat_decode_ready": False,
            "flat_decode_failed": True,
            "flat_decode_failed_reason": "unsupported shape",
        },
        "scheduler_stats": {
            "decode_cuda_graphs": {
                "enabled": True,
                "captures": 1,
                "replays": 2,
                "failures": 1,
                "last_failure": "capture failed",
            },
            "decode_execution": {
                "prefer_step": False,
                "decode_step_batches": 0,
                "multi_step_batches": 8,
            },
        },
    }
    raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = runner.audit_gemma4_dense_fast_path(raw_path, "gemma4-e4b-fast")

    assert report["status"] == "failed"
    assert any("flat decode not ready" in error for error in report["errors"])
    assert any("unexpectedly enabled" in error for error in report["errors"])
    assert any("does not match" in error for error in report["errors"])
    assert any("capture failed" in error for error in report["errors"])


def test_e2b_audit_rejects_structurally_valid_but_regressed_l4_path(tmp_path):
    runner = load_runner()
    raw_path = tmp_path / "e2b.jsonl"
    row = {
        "ok": True,
        "model": "google/gemma-4-E2B-it",
        "hardware_label": "1xl4",
        "dtype": "bf16",
        "scenario": "single",
        "batch_size": 1,
        "prompt_tokens_requested_per_request": 128,
        "output_tps": 4.03,
        "model_topology": {
            "num_hidden_layers": 35,
            "hidden_size": 1536,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 20,
            "kv_cache_layers": 15,
            "sliding_attention_layers": 28,
            "full_attention_layers": 7,
        },
        "decode_runtime_stats": {
            "flat_decode_ready": True,
            "flat_decode_failed": False,
            "gemma4_dense_post_norm_chain_decode_enabled": True,
            "gemma4_dense_post_norm_chain_decode_hits": 63,
        },
        "scheduler_stats": {
            "decode_cuda_graphs": {
                "enabled": False,
                "captures": 0,
                "replays": 0,
                "failures": 0,
                "request_scheduler_reuse_count": 0,
            },
            "decode_execution": {
                "prefer_step": False,
                "decode_step_batches": 0,
                "multi_step_batches": 128,
            },
        },
    }
    raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = runner.audit_gemma4_dense_fast_path(
        raw_path, "gemma4-e2b-fast"
    )

    assert report["status"] == "failed"
    assert report["performance_gate"]["applicable"] is True
    assert any("regression gate failed" in error for error in report["errors"])


def test_e2b_audit_accepts_disabled_graph_stats_being_omitted(tmp_path):
    runner = load_runner()
    raw_path = tmp_path / "e2b.jsonl"
    row = {
        "ok": True,
        "model": "google/gemma-4-E2B-it",
        "hardware_label": "1xl4",
        "dtype": "bf16",
        "scenario": "single",
        "batch_size": 1,
        "prompt_tokens_requested_per_request": 128,
        "output_tps": 30.01,
        "model_topology": {
            "num_hidden_layers": 35,
            "hidden_size": 1536,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 20,
            "kv_cache_layers": 15,
            "sliding_attention_layers": 28,
            "full_attention_layers": 7,
        },
        "decode_runtime_stats": {
            "flat_decode_ready": True,
            "flat_decode_failed": False,
            "gemma4_dense_post_norm_chain_decode_enabled": True,
            "gemma4_dense_post_norm_chain_decode_hits": 63,
        },
        "scheduler_stats": {
            "decode_execution": {
                "prefer_step": False,
                "decode_step_batches": 0,
                "multi_step_batches": 128,
            },
        },
    }
    raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = runner.audit_gemma4_dense_fast_path(
        raw_path, "gemma4-e2b-fast"
    )

    assert report["status"] == "passed"
    assert report["required"]["decode_cuda_graphs_disabled"] is True
    assert report["required"]["request_scheduler_reuse_count"] == 0
    assert report["required"]["dense_post_norm_chain_required"] is True
    assert report["required"]["dense_post_norm_chain_enabled"] is True
    assert report["required"]["dense_post_norm_chain_hits"] == 63


def test_e2b_audit_proves_batch8_cublas_mlp_policy_and_zero_fusion_hits(tmp_path):
    runner = load_runner()
    raw_path = tmp_path / "e2b_batch8.jsonl"
    row = {
        "ok": True,
        "model": "google/gemma-4-E2B-it",
        "hardware_label": "1xl4",
        "dtype": "bf16",
        "scenario": "batch",
        "batch_size": 8,
        "prompt_tokens_requested_per_request": 128,
        "output_tps": 249.8,
        "model_topology": {
            "num_hidden_layers": 35,
            "hidden_size": 1536,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 20,
            "kv_cache_layers": 15,
            "sliding_attention_layers": 28,
            "full_attention_layers": 7,
        },
        "decode_runtime_stats": {
            "flat_decode_ready": True,
            "flat_decode_failed": False,
            "gemma4_dense_post_norm_chain_decode_enabled": True,
            "gemma4_dense_post_norm_chain_decode_hits": 63,
            "gemma4_policy_bf16_fused_gateup_rows": [],
            "gemma4_policy_bf16_deepfusion_rows": [],
            "gemma4_policy_bf16_cublas_gateup_rows": [8],
            "gemma4_policy_bf16_cublas_down_rows": [8],
            "gemma4_flat_fused_gateup_hits": 0,
            "gemma4_flat_deepfusion_hits": 0,
            "runtime_policy": {
                "gemma4_e2b_h512_dense_bridge_pair": True,
            },
            "gemma4_dense_attn_mlp_bridge_decode_enabled": True,
            "gemma4_dense_attn_mlp_bridge_decode_hits": 4445,
            "gemma4_dense_attn_mlp_bridge_runtime_disabled": False,
            "gemma4_dense_attn_mlp_bridge_failure": "",
            "paged_decode_runtime": {
                "grouped_segmented_hits": 889,
                "grouped_segmented_disabled": False,
                "grouped_segmented_failure": "",
                "grouped_segmented_selected_segments": {
                    "e2b_l4_full_h512_gqa8": 32,
                },
                "grouped_segmented_selected_tile_sizes": {
                    "e2b_l4_full_h512_gqa8": 16,
                },
            },
        },
        "scheduler_stats": {
            "decode_execution": {
                "prefer_step": False,
                "decode_step_batches": 0,
                "multi_step_batches": 128,
            },
        },
    }
    raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = runner.audit_gemma4_dense_fast_path(
        raw_path, "gemma4-e2b-fast"
    )

    assert report["status"] == "passed"
    assert report["required"]["batch8_cublas_mlp_applicable"] is True
    assert report["required"]["batch8_cublas_gateup_policy_rows"] == [[8]]
    assert report["required"]["batch8_cublas_down_policy_rows"] == [[8]]
    assert report["required"]["batch8_fused_gateup_hits"] == 0
    assert report["required"]["batch8_deepfusion_hits"] == 0
    assert report["required"]["e2b_h512_dense_bridge_pair_policy_enabled"] is True
    assert report["required"]["e2b_h512_grouped_attention_segments"] == [32]
    assert report["required"]["e2b_h512_grouped_attention_tiles"] == [16]
    assert report["required"]["e2b_dense_attn_mlp_bridge_hits"] == 4445

    row["decode_runtime_stats"]["gemma4_flat_deepfusion_hits"] = 1
    raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    rejected = runner.audit_gemma4_dense_fast_path(
        raw_path, "gemma4-e2b-fast"
    )
    assert rejected["status"] == "failed"
    assert any(
        "batch-8 deepfusion path unexpectedly ran" in error
        for error in rejected["errors"]
    )


def test_e2b_audit_requires_promoted_l4_long_sliding_prefill_hits(tmp_path):
    runner = load_runner()
    raw_path = tmp_path / "e2b_long_batch8.jsonl"
    row = {
        "ok": True,
        "model": "google/gemma-4-E2B-it",
        "hardware_label": "1xl4",
        "dtype": "bf16",
        "scenario": "long_context",
        "batch_size": 8,
        "prompt_tokens_requested_per_request": 2048,
        "output_tps": 130.48,
        "model_topology": {
            "num_hidden_layers": 35,
            "hidden_size": 1536,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 20,
            "kv_cache_layers": 15,
            "sliding_attention_layers": 28,
            "full_attention_layers": 7,
        },
        "decode_runtime_stats": {
            "flat_decode_ready": True,
            "flat_decode_failed": False,
            "gemma4_dense_post_norm_chain_decode_enabled": True,
            "gemma4_dense_post_norm_chain_decode_hits": 35560,
            "gemma4_policy_bf16_fused_gateup_rows": [],
            "gemma4_policy_bf16_deepfusion_rows": [],
            "gemma4_policy_bf16_cublas_gateup_rows": [8],
            "gemma4_policy_bf16_cublas_down_rows": [8],
            "gemma4_flat_fused_gateup_hits": 0,
            "gemma4_flat_deepfusion_hits": 0,
            "gemma4_e2b_l4_sliding_prefill_enabled": True,
            "gemma4_e2b_l4_sliding_prefill_hits": 224,
            "gemma4_e2b_l4_full_prefill_expand_enabled": True,
            "gemma4_e2b_l4_full_prefill_expand_enabled_layers": 7,
            "gemma4_e2b_l4_full_prefill_expand_hits": 56,
            "gemma4_e2b_l4_full_prefill_expand_error": "",
            "runtime_policy": {
                "gemma4_e2b_h512_dense_bridge_pair": True,
                "gemma4_e2b_l4_full_prefill_expand": True,
            },
            "gemma4_dense_attn_mlp_bridge_decode_enabled": True,
            "gemma4_dense_attn_mlp_bridge_decode_hits": 4445,
            "gemma4_dense_attn_mlp_bridge_runtime_disabled": False,
            "gemma4_dense_attn_mlp_bridge_failure": "",
            "paged_decode_runtime": {
                "grouped_segmented_hits": 889,
                "grouped_segmented_disabled": False,
                "grouped_segmented_failure": "",
                "grouped_segmented_selected_segments": {
                    "e2b_l4_full_h512_gqa8": 32,
                },
                "grouped_segmented_selected_tile_sizes": {
                    "e2b_l4_full_h512_gqa8": 16,
                },
            },
        },
        "scheduler_stats": {
            "decode_execution": {
                "prefer_step": False,
                "decode_step_batches": 0,
                "multi_step_batches": 128,
            },
        },
    }
    raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = runner.audit_gemma4_dense_fast_path(
        raw_path, "gemma4-e2b-fast"
    )

    assert report["status"] == "passed"
    assert report["required"]["e2b_l4_sliding_prefill_applicable"] is True
    assert report["required"]["e2b_l4_sliding_prefill_enabled"] is True
    assert report["required"]["e2b_l4_sliding_prefill_hits"] == 224
    assert report["required"]["e2b_l4_full_prefill_expand_applicable"] is True
    assert report["required"]["e2b_l4_full_prefill_expand_enabled"] is True
    assert report["required"]["e2b_l4_full_prefill_expand_hits"] == 56
    assert report["required"][
        "e2b_l4_full_prefill_expand_enabled_layers"
    ] == [7]
    assert report["required"]["e2b_l4_full_prefill_expand_errors"] == []
    assert report["performance_gate"]["floors"][
        "long_context/b8/p2048"
    ] == 115.0

    row["decode_runtime_stats"]["gemma4_e2b_l4_full_prefill_expand_hits"] = 0
    raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    rejected = runner.audit_gemma4_dense_fast_path(
        raw_path, "gemma4-e2b-fast"
    )
    assert rejected["status"] == "failed"
    assert any(
        "full-H512 expanded implicit-causal prefill path was never exercised"
        in error
        for error in rejected["errors"]
    )

    row["decode_runtime_stats"]["gemma4_e2b_l4_full_prefill_expand_hits"] = 56
    row["decode_runtime_stats"]["gemma4_e2b_l4_sliding_prefill_hits"] = 0
    raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    rejected = runner.audit_gemma4_dense_fast_path(
        raw_path, "gemma4-e2b-fast"
    )
    assert rejected["status"] == "failed"
    assert any(
        "sliding-prefill Triton kernel was never exercised" in error
        for error in rejected["errors"]
    )


def test_resume_rejects_a_summary_from_a_different_workload(tmp_path):
    runner = load_runner()
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "args": {
                    "backend": "megagemm",
                    "model": "google/gemma-4-E2B-it",
                    "hardware_label": "1xl4",
                    "batch_sizes": "1,8",
                    "prompt_tokens": "128,512,2048",
                    "max_new_tokens": 64,
                    "repeats": 5,
                    "warmup": 3,
                    "dtype": "bf16",
                    "quantize": None,
                    "max_seq_len": 2304,
                    "max_batch_size": 8,
                }
            }
        ),
        encoding="utf-8",
    )
    args = runner.argparse.Namespace(
        model="google/gemma-4-E2B-it",
        batch_sizes="1,8",
        prompt_tokens="128,512,2048",
        max_new_tokens=128,
        repeats=5,
        max_batch_size=8,
    )

    try:
        runner.validate_existing_variant_artifacts(
            summary_path,
            args=args,
            variant=runner.VARIANTS["megagemm-bf16"],
            hardware_label="1xl4",
            warmup=3,
            max_seq_len=2304,
        )
    except RuntimeError as exc:
        assert "max_new_tokens" in str(exc)
    else:
        raise AssertionError("stale benchmark summary was reused")


def test_failed_raw_artifact_is_not_reusable(tmp_path):
    runner = load_runner()
    raw_path = tmp_path / "vllm.jsonl"
    raw_path.write_text(
        json.dumps(
            {
                "ok": False,
                "error": "ModuleNotFoundError: No module named 'vllm'",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    errors = runner.raw_artifact_errors(raw_path)

    assert len(errors) == 1
    assert "1/1 benchmark row(s) failed" in errors[0]
    assert "No module named 'vllm'" in errors[0]
