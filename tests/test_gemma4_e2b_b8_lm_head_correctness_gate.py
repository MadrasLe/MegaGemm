from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "benchmarks" / "run_gemma4_e2b_b8_lm_head_correctness_gate.py"
WRAPPER = (
    ROOT
    / "benchmarks"
    / "run_gemma4_e2b_b8_lm_head_correctness_gate_colab.sh"
)


def test_gate_uses_actual_hidden_and_canonical_logits_contract():
    source = GATE.read_text(encoding="utf-8")
    assert "_capture_actual_decode_hidden_states" in source
    assert "captured.append(hidden.detach().clone())" in source
    assert "model._decode_raw_logits_from_hidden(hidden)" in source
    assert "model._apply_final_logit_capping(raw_logits)" in source
    assert "torch.argmax(final_logits, dim=-1)" in source


def test_gate_compares_both_fused_reductions_on_identical_hidden():
    source = GATE.read_text(encoding="utf-8")
    assert "logits_softcap_argmax(" in source
    assert "lm_head_rmsnorm_argmax(" in source
    assert "lm_head_argmax(" in source
    assert "canonical_minus_direct_rmsnorm" in source


def test_gate_disables_graphs_only_for_capture_and_does_not_time():
    source = GATE.read_text(encoding="utf-8")
    assert '"MEGAGEMM_DECODE_CUDA_GRAPHS": "0"' in source
    assert '"cuda_graphs_during_capture": False' in source
    assert "minimum-speedup" not in source
    assert "warmups" not in source
    assert "repeats" not in source


def test_gate_accepts_candidate_replacement_only_when_oracle_exact():
    source = GATE.read_text(encoding="utf-8")
    assert 'softcap["mismatches"] == 0 and direct_rms["mismatches"] > 0' in source
    assert '"PROMOTE_TENSORCORE_FUSED_SOFTCAP_CORRECTNESS"' in source
    assert 'return 0 if payload["status"] == "passed" else 2' in source


def test_colab_wrapper_uses_fixed_drive_and_one_python_gate():
    source = WRAPPER.read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in source
    assert "run_gemma4_e2b_b8_lm_head_correctness_gate.py" in source
    assert "CAPTURE_STEPS" in source
    assert "git pull" not in source
    assert "vllm" not in source.lower()
    assert "pip install -e" not in source
