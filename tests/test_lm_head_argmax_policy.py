import importlib

import torch


lm_head = importlib.import_module("megagemm.kernels.lm_head_argmax")


def _clear_forced_config(monkeypatch):
    monkeypatch.setattr(lm_head, "_CFG_FORCED_BN", 0)
    monkeypatch.setattr(lm_head, "_CFG_FORCED_BK", 0)
    monkeypatch.setattr(lm_head, "_CFG_FORCED_WARPS", 0)
    monkeypatch.setattr(lm_head, "_CFG_FORCED_STAGES", 0)


def test_gemma4_e2b_l4_b8_uses_promoted_bn64(monkeypatch):
    _clear_forced_config(monkeypatch)
    assert lm_head._pick_cfg(
        1536,
        262144,
        rows=8,
        dtype=torch.bfloat16,
        device_name="NVIDIA L4",
    ) == (64, 128, 4, 2)


def test_promoted_lm_head_config_is_exact_shape_and_hardware(monkeypatch):
    _clear_forced_config(monkeypatch)
    assert lm_head._pick_cfg(
        1536,
        262144,
        rows=1,
        dtype=torch.bfloat16,
        device_name="NVIDIA L4",
    )[0] == 256
    assert lm_head._pick_cfg(
        1536,
        262144,
        rows=8,
        dtype=torch.float16,
        device_name="NVIDIA L4",
    )[0] == 256
    assert lm_head._pick_cfg(
        1536,
        262144,
        rows=8,
        dtype=torch.bfloat16,
        device_name="Tesla T4",
    )[0] == 256


def test_explicit_lm_head_override_precedes_promoted_policy(monkeypatch):
    monkeypatch.setattr(lm_head, "_CFG_FORCED_BN", 128)
    monkeypatch.setattr(lm_head, "_CFG_FORCED_BK", 64)
    monkeypatch.setattr(lm_head, "_CFG_FORCED_WARPS", 8)
    monkeypatch.setattr(lm_head, "_CFG_FORCED_STAGES", 3)
    assert lm_head._pick_cfg(
        1536,
        262144,
        rows=8,
        dtype=torch.bfloat16,
        device_name="NVIDIA L4",
    ) == (128, 64, 8, 3)


def test_runtime_config_reports_promoted_gemma4_shape():
    assert lm_head.lm_head_argmax_runtime_config()[
        "gemma4_e2b_l4_b8_block_n"
    ] == 64
