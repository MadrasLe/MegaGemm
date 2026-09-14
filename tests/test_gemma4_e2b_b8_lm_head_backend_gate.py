from pathlib import Path
from types import SimpleNamespace

import torch

from benchmarks import run_gemma4_e2b_b8_lm_head_backend_gate as gate
from megagemm.models import llama


def test_cases_cover_current_and_both_tensorcore_reductions():
    assert [case.name for case in gate.CASES] == [
        "legacy_direct_fused",
        "tensorcore_full_logits",
        "production_tensorcore_fused_softcap",
    ]
    assert not gate.CASES[0].batch_cublas
    assert gate.CASES[1].batch_cublas
    assert not gate.CASES[1].fused_softcap_argmax
    assert gate.CASES[2].batch_cublas
    assert gate.CASES[2].fused_softcap_argmax


def test_promoted_e2b_route_forces_correct_softcap_reduction():
    source = Path(llama.__file__).read_text(encoding="utf-8")
    assert '"MEGAGEMM_GEMMA4_E2B_L4_B8_BATCH_CUBLAS_LM_HEAD"' in source
    assert "default=True" in source
    assert "and _GEMMA4_E2B_L4_B8_FUSED_SOFTCAP_ARGMAX" in source


def test_backend_gate_can_disable_softcap_without_disabling_tensorcore():
    module = SimpleNamespace()
    gate.apply_case(module, gate.CASES[1])
    assert module._GEMMA4_E2B_L4_B8_BATCH_CUBLAS_LM_HEAD is True
    assert module._GEMMA4_E2B_L4_B8_FUSED_SOFTCAP_ARGMAX is False


def test_e2b_l4_b8_batch_cublas_shape_is_promoted_and_exact(monkeypatch):
    check = llama._gemma4_a100_a4b_batch_cublas_lm_head_shape
    monkeypatch.setattr(llama, "_GEMMA4_BATCH_CUBLAS_LM_HEAD", True)
    monkeypatch.setattr(
        llama,
        "_GEMMA4_E2B_L4_B8_BATCH_CUBLAS_LM_HEAD",
        False,
    )
    args = ("gemma4_text", 8, 1536, 262144, torch.bfloat16, "NVIDIA L4")
    assert not check(*args)
    monkeypatch.setattr(
        llama,
        "_GEMMA4_E2B_L4_B8_BATCH_CUBLAS_LM_HEAD",
        True,
    )
    assert check(*args)
    assert not check("gemma4_text", 4, 1536, 262144, torch.bfloat16, "NVIDIA L4")
    assert not check("gemma4_text", 8, 2816, 262144, torch.bfloat16, "NVIDIA L4")
    assert not check("gemma4_text", 8, 1536, 256000, torch.bfloat16, "NVIDIA L4")
    assert not check("gemma4_text", 8, 1536, 262144, torch.float16, "NVIDIA L4")
    assert not check("gemma4_text", 8, 1536, 262144, torch.bfloat16, "Tesla T4")


def test_existing_a4b_shape_remains_available_without_e2b_experiment(monkeypatch):
    monkeypatch.setattr(llama, "_GEMMA4_BATCH_CUBLAS_LM_HEAD", True)
    monkeypatch.setattr(
        llama,
        "_GEMMA4_E2B_L4_B8_BATCH_CUBLAS_LM_HEAD",
        False,
    )
    assert llama._gemma4_a100_a4b_batch_cublas_lm_head_shape(
        "gemma4_text", 16, 2816, 262144, torch.bfloat16, "NVIDIA A100-SXM4-40GB"
    )


def test_route_audit_distinguishes_three_backends():
    production = {
        "gemma4_batch_cublas_lm_head_hits": 0,
        "fused_rmsnorm_lm_head_argmax_hits": 1,
    }
    assert gate.route_errors(gate.CASES[0], production) == []
    full_logits = {
        "gemma4_batch_cublas_lm_head_hits": 1,
        "gemma4_e2b_l4_b8_batch_cublas_lm_head_enabled": True,
        "gemma4_batch_fused_softcap_argmax_hits": 0,
    }
    assert gate.route_errors(gate.CASES[1], full_logits) == []
    fused_softcap = dict(full_logits)
    fused_softcap["gemma4_e2b_l4_b8_fused_softcap_argmax_enabled"] = True
    fused_softcap["gemma4_batch_fused_softcap_argmax_hits"] = 1
    assert gate.route_errors(gate.CASES[2], fused_softcap) == []
    fused_softcap["gemma4_batch_cublas_lm_head_hits"] = 0
    fused_softcap["gemma4_batch_fused_softcap_argmax_disabled"] = True
    errors = gate.route_errors(gate.CASES[2], fused_softcap)
    assert any("no capture hits" in error for error in errors)
    assert any("disabled itself" in error for error in errors)


def test_colab_harness_is_fresh_drive_one_process_without_vllm_or_git():
    script = Path(
        gate.ROOT,
        "benchmarks",
        "run_gemma4_e2b_b8_lm_head_backend_gate_colab.sh",
    ).read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in script
    assert "run_gemma4_e2b_b8_lm_head_backend_gate.py" in script
    assert "git pull" not in script
    assert "pip install -e" not in script
    assert "vllm" not in script.lower()


def test_rejected_cublaslt_candidate_does_not_fail_completed_gate():
    source = Path(
        gate.ROOT,
        "benchmarks",
        "run_gemma4_e2b_b8_cublaslt_decode_frontier.py",
    ).read_text(encoding="utf-8")
    assert "return 0 if gate_completed else 2" in source
