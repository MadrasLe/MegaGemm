import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks import run_gemma4_e2b_b1_execution_gate as gate
from megagemm.engine.scheduler import Scheduler


ROOT = Path(__file__).resolve().parents[1]


def row(tokens=128, elapsed=3.7):
    return {
        "generated_tokens": tokens, "lengths": [tokens], "digest": "natural-tokens",
        "engine_prompt_tokens": 2057, "engine_prompt_lengths": [2057],
        "engine_max_prompt_tokens": 2057,
        "elapsed_s": elapsed, "output_tps": tokens / elapsed,
        "scheduler_stats": {"benchmark_forced_token_id": -1,
                            "decode_execution": {"multi_step_batches": 16},
                            "decode_cuda_graphs": {"request_scheduler_reused": True}},
    }


def test_case_matrix_only_changes_execution():
    assert len(gate.CASES) == 5
    for case in gate.CASES:
        env = gate.case_environment(case)
        assert env["MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID"] == "-1"
        assert env["MEGAGEMM_NATIVE_DECODE_GRAPH_BURST"] == "0"
        assert not any("GEMV" in k or "LM_HEAD" in k or "GATEUP" in k for k in env)
    assert not gate.CASES[0].reuse


def test_b8_frontier_is_full_model_and_burst_specific():
    assert [case.name for case in gate.B8_FRONTIER_CASES] == [
        "production_burst8",
        "eager_reuse_burst8",
        "one_step_graph_burst4",
        "one_step_graph_burst8",
        "one_step_graph_burst16",
        "unrolled_graph8",
    ]
    assert [case.burst_steps for case in gate.B8_FRONTIER_CASES] == [
        8, 8, 4, 8, 16, 8,
    ]
    assert gate.B8_FRONTIER_CASES[-1].unroll
    for case in gate.B8_FRONTIER_CASES:
        env = gate.case_environment(case)
        assert env["MEGAGEMM_MULTI_STEP_BURST_BATCH"] == str(case.burst_steps)
        assert env["MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID"] == "-1"


def test_colab_wrapper_selects_b8_batch_and_result_directory():
    wrapper = (
        ROOT / "benchmarks" / "run_gemma4_e2b_b1_decode_frontier_colab.sh"
    ).read_text(encoding="utf-8")
    assert 'SUITE" == "b8-execution"' in wrapper
    assert "EFFECTIVE_BATCH_SIZE=8" in wrapper
    assert "--b8-frontier --skip-python-audit" in wrapper
    assert "gemma4_e2b_b8_execution_frontier" in wrapper


def test_execution_audit_rejects_wrong_burst_size_for_graph():
    case = gate.B8_FRONTIER_CASES[2]
    result = graph_row(case)
    result["scheduler_stats"]["decode_cuda_graphs"]["token_burst_size"] = 8
    assert gate.audit_execution(case, result, steady=True)


def test_promotion_case_removes_experiment_overrides(monkeypatch):
    promoted = gate.PROMOTION_CASES[1]
    controls = gate.case_environment(promoted)
    for name in controls:
        monkeypatch.setenv(name, controls[name])

    gate.activate_case_environment(promoted)

    assert promoted.runtime_policy
    assert promoted.reuse and promoted.graph
    assert all(name not in __import__("os").environ for name in controls)


@pytest.mark.parametrize("field,value", [("digest", "wrong"), ("digest", None),
                                         ("lengths", [127]), ("engine_prompt_tokens", 2048),
                                         ("elapsed_s", float("nan"))])
def test_token_and_metric_audit_rejects_bad_rows(field, value):
    bad = row()
    bad[field] = value
    assert gate.validate_run(bad, row(), 128)


def test_real_greedy_validation():
    assert not gate.validate_run(row(), row(), 128)
    bad = row()
    bad["scheduler_stats"]["benchmark_forced_token_id"] = 42
    assert gate.validate_run(bad, row(), 128)


def test_token_comparison_reports_first_divergence_and_agreement():
    reference = {"digest": "production", "generated_ids": [[10, 20, 30, 40]]}
    candidate = {"digest": "candidate", "generated_ids": [[10, 20, 31, 40]]}
    result = gate.compare_generated_tokens(candidate, reference)
    assert not result["exact_match"]
    assert result["token_agreement"] == pytest.approx(0.75)
    assert result["first_divergence"] == {
        "sequence_index": 0, "token_index": 2,
        "reference_token": 30, "candidate_token": 31,
    }


def test_runtime_validation_can_separate_shape_health_from_token_parity():
    reference = row()
    candidate = row()
    candidate["digest"] = "different"
    assert gate.validate_run(candidate, reference, 128)
    assert not gate.validate_run(candidate, reference, 128, require_token_match=False)


def test_runtime_validation_counts_all_tokens_in_a_batch():
    reference = row()
    reference.update(generated_tokens=512, lengths=[128] * 4,
                     engine_prompt_tokens=8228,
                     engine_prompt_lengths=[2057] * 4,
                     engine_max_prompt_tokens=2057)
    assert not gate.validate_run(reference, reference, 128, batch_size=4)
    assert gate.validate_run(reference, reference, 128, batch_size=2)


def test_b4_sequence_capacity_uses_per_request_not_aggregate_tokens():
    result = {
        "engine_prompt_tokens": 8228,
        "engine_prompt_lengths": [2057] * 4,
        "engine_max_prompt_tokens": 2057,
    }
    maximum = gate.effective_max_prompt_tokens(result, batch_size=4)
    assert maximum == 2057
    assert maximum + 128 == 2185
    assert maximum + 128 <= 2304


def graph_row(case):
    r = row()
    r["scheduler_stats"]["decode_cuda_graphs"].update(
        multi_step_body=True, persistent_token_feedback_steps=127,
        eager_control=case.eager_control, replays=0 if case.eager_control else 20,
        unrolled_token_burst_steps=119 if case.unroll else 0,
        token_burst_size=case.burst_steps,
    )
    return r


@pytest.mark.parametrize("case", gate.CASES)
def test_execution_audit_accepts_intended_path(case):
    r = graph_row(case) if case.graph else row()
    assert not gate.audit_execution(case, r, steady=True)


@pytest.mark.parametrize("key,value", [("captures", 1), ("physical_rebinds", 1),
                                       ("warmups", 1), ("replays", 0),
                                       ("multi_step_body", False), ("failures", 1),
                                       ("native_token_bursts", 1)])
def test_reject_capture_and_fallback_inside_timed_sample(key, value):
    case = gate.CASES[3]
    r = graph_row(case)
    r["scheduler_stats"]["decode_cuda_graphs"][key] = value
    assert gate.audit_execution(case, r, steady=True)


def test_cold_capture_is_allowed_but_not_silent_fallback():
    case = gate.CASES[3]
    r = graph_row(case)
    r["scheduler_stats"]["decode_cuda_graphs"]["captures"] = 1
    assert not gate.audit_execution(case, r, steady=False)


def test_summary_no_promotion_from_incomplete_or_invalid_result():
    samples = []
    for case in gate.CASES:
        for repeat in range(3):
            samples.append({"case": case.name, "incremental_decode_s": 3.4,
                            "decode_tps": 127 / 3.4, "long": row(), "errors": []})
    result = gate.summarize(samples, {}, repeats=3, minimum_speedup=1.02, maximum_spread=1.06)
    assert result["valid"] and result["decision"] == "KEEP_PRODUCTION"
    samples[0]["errors"] = ["wrong tokens"]
    result = gate.summarize(samples, {}, repeats=3, minimum_speedup=1.02, maximum_spread=1.06)
    assert not result["valid"] and result["winner"] is None


def test_multi_step_graph_body_calls_production_not_alternate_decode_step():
    calls = []
    def multi(ids, positions, blocks, seqs, **kwargs):
        calls.append(kwargs)
        return ids + 1, None
    scheduler = object.__new__(Scheduler)
    scheduler._decode_graph_multi_step_body = True
    scheduler.model = SimpleNamespace(decode_multi_step=multi)
    scheduler.block_manager = object()
    result = scheduler._decode_graph_model_step(torch.tensor([[5]]), torch.tensor([[12]]), [0], True)
    assert result.tolist() == [6]
    assert calls == [{"num_steps": 1, "return_final_logits": False, "return_token_ids": True}]
    result = scheduler._decode_graph_model_step(
        torch.tensor([[5], [7]]), torch.tensor([[12], [12]]), [0, 1], True
    )
    assert result.tolist() == [6, 8]
    with pytest.raises(RuntimeError, match="greedy token output"):
        scheduler._decode_graph_model_step(None, None, [0, 1], False)


def test_existing_graph_body_keeps_decode_step():
    scheduler = object.__new__(Scheduler)
    scheduler.model = SimpleNamespace(decode_step=lambda *a, **k: "old path")
    scheduler.block_manager = object()
    assert scheduler._decode_graph_model_step(None, None, [0], True) == "old path"


def test_b4_and_b8_graph_topologies_are_production_eligible(monkeypatch):
    from megagemm.models.llama import _gemma4_l4_e2b_decode_graph_shape
    cfg = SimpleNamespace(model_type="gemma4_text", enable_moe_block=False, hidden_size=1536,
                          num_hidden_layers=35, num_attention_heads=8, num_key_value_heads=1,
                          num_kv_shared_layers=20, layer_types=["sliding_attention"] * 28 + ["full_attention"] * 7)
    kw = dict(num_seqs=1, dtype=torch.bfloat16, device_type="cuda", device_name="NVIDIA L4")
    monkeypatch.delenv("MEGAGEMM_GEMMA4_E2B_B1_GRAPH_EXPERIMENT", raising=False)
    assert not _gemma4_l4_e2b_decode_graph_shape(cfg, **kw)
    assert not _gemma4_l4_e2b_decode_graph_shape(cfg, **{**kw, "num_seqs": 2})
    assert _gemma4_l4_e2b_decode_graph_shape(cfg, **{**kw, "num_seqs": 4})
    assert _gemma4_l4_e2b_decode_graph_shape(cfg, **{**kw, "num_seqs": 8})
    monkeypatch.setenv("MEGAGEMM_GEMMA4_E2B_B1_GRAPH_EXPERIMENT", "1")
    assert _gemma4_l4_e2b_decode_graph_shape(cfg, **kw)
    for changes in ({"num_seqs": 2}, {"dtype": torch.float16}, {"device_name": "NVIDIA A100"}):
        assert not _gemma4_l4_e2b_decode_graph_shape(cfg, **{**kw, **changes})
    monkeypatch.setenv("MEGAGEMM_GEMMA4_E2B_SMALL_BATCH_GRAPH_EXPERIMENT", "1")
    for batch_size in (1, 2, 4, 8):
        assert _gemma4_l4_e2b_decode_graph_shape(cfg, **{**kw, "num_seqs": batch_size})
    assert not _gemma4_l4_e2b_decode_graph_shape(cfg, **{**kw, "num_seqs": 3})


def test_graph_runtime_policy_does_not_change_non_b4_decode_dispatch():
    scheduler = object.__new__(Scheduler)
    scheduler._decode_cuda_graph_policy_batches = (4,)

    assert not scheduler._decode_graph_batch_allowed(1)
    assert not scheduler._decode_graph_batch_allowed(2)
    assert scheduler._decode_graph_batch_allowed(4)
    assert not scheduler._decode_graph_batch_allowed(8)

    scheduler._decode_cuda_graph_policy_batches = ()
    for batch_size in (1, 2, 4, 8):
        assert scheduler._decode_graph_batch_allowed(batch_size)


def test_python_audit_restores_profiler_on_error():
    import sys
    before = sys.getprofile()
    def fail():
        raise RuntimeError("sentinel")
    with pytest.raises(RuntimeError, match="sentinel"):
        gate.python_call_audit(fail)
    assert sys.getprofile() is before


@pytest.mark.parametrize("kernel_mode", [False, True])
def test_whole_gate_orchestration_one_load_and_no_external_artifacts(monkeypatch, tmp_path, kernel_mode):
    from benchmarks import benchmark_inference_matrix as matrix
    import megagemm.engine
    loads = []
    model = SimpleNamespace(decode_cuda_graph_eligible=lambda **kw: True)
    class Engine:
        def __init__(self, *a, **kw):
            loads.append(kw)
            self.model = model
    monkeypatch.setattr(megagemm.engine, "InferenceEngine", Engine)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "NVIDIA L4")
    monkeypatch.setattr(matrix, "load_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr(matrix, "build_prompts", lambda *a, **k: (["test prompt"], 2048))
    monkeypatch.setattr(gate, "_configure_environment", lambda *a: {})
    monkeypatch.setattr(gate, "_capture_lm_state", lambda *a: {"fixed": True})
    monkeypatch.setattr(gate, "_counter_snapshot", lambda *a: {})
    if kernel_mode:
        from benchmarks.gemma4_b1_graph_kernels import GraphKernelExperiment
        monkeypatch.setattr(GraphKernelExperiment, "compile_preflight", lambda self: {"all_compiled": True})
        def prepare(self, *a):
            self.ready = True
            return {}
        def activate(self, model, case):
            model.kernel_case = case
        monkeypatch.setattr(GraphKernelExperiment, "prepare", prepare)
        monkeypatch.setattr(GraphKernelExperiment, "activate", activate)
        monkeypatch.setattr(GraphKernelExperiment, "audit_routes", lambda *a: [])
        monkeypatch.setattr(GraphKernelExperiment, "audit_cold", lambda *a: [])
        monkeypatch.setattr(GraphKernelExperiment, "audit_state", lambda *a: [])
        monkeypatch.setenv("MEGAGEMM_B1_GRAPH_KERNEL_CASE", "before-test")
    @contextlib.contextmanager
    def probe(model, signatures):
        signatures.append({"bridge": 35})
        yield
    monkeypatch.setattr(gate, "layer_route_probe", probe)
    def run(engine, prompts, tokens):
        import os
        case = getattr(model, "kernel_case", None) or next(
            c for c in gate.CASES if all(os.environ[k] == v for k, v in gate.case_environment(c).items()))
        result = graph_row(case) if case.graph else row()
        result.update(generated_tokens=tokens, lengths=[tokens], elapsed_s=.27 if tokens == 1 else 3.7,
                      output_tps=tokens / (.27 if tokens == 1 else 3.7))
        engine._last_scheduler = object()
        return result
    monkeypatch.setattr(gate, "_run", run)
    # Isolate environment mutations made by the CLI.
    for key in gate.case_environment(gate.CASES[0]):
        monkeypatch.setenv(key, "before-test")
    dest = tmp_path / "result.json"
    cli = ["--output", str(dest), "--skip-python-audit"]
    if kernel_mode:
        cli += ["--kernel-candidates"]
    assert gate.main(cli) == 0
    result = json.loads(dest.read_text())
    assert len(loads) == 1
    assert len(result["samples"]) == 15
    assert result["valid"] and result["decision"] == "KEEP_PRODUCTION"
    assert len(result["cold_setup"]) == 5
    if kernel_mode:
        assert result["method"]["baseline"] == "graph_production"
        assert all(s["case"].startswith("graph_") for s in result["samples"])


def test_eager_graph_control_chains_real_tokens_and_positions_without_capture():
    scheduler = object.__new__(Scheduler)
    ids, pos = torch.tensor([[5]]), torch.tensor([[15]])
    key = (1, 144, 144, True, True)
    state = {"seq_key": (0,), "graph": None}
    scheduler._decode_graph_persistent_token_feedback = True
    scheduler._decode_graph_multi_step_body = True
    scheduler._decode_graph_eager_control = True
    scheduler._decode_graph_chain_started_keys = set()
    scheduler._decode_graph_chain_input_updates_skipped = 0
    scheduler._decode_graph_persistent_feedback_steps = 0
    scheduler._prepare_decode_graph_shape_state = lambda *a, **k: (key, state)
    scheduler._copy_decode_graph_shape_inputs = lambda state, ids, pos: (ids, pos)
    scheduler._run_decode_with_metadata_override = lambda state, seqs, ids, pos, **k: ids.reshape(1) + 1
    first = scheduler._run_decode_step_shape_graph([0], ids, pos, return_next_token=True)
    second = scheduler._run_decode_step_shape_graph([0], ids, pos, return_next_token=True)
    assert first.tolist() == [6] and second.tolist() == [7]
    assert ids.tolist() == [[7]] and pos.tolist() == [[17]]
    assert state["eager_control"] and state["graph"] is None
    assert scheduler._decode_graph_persistent_feedback_steps == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for real graph replay")
def test_cuda_one_step_and_unrolled_replay_update_tokens_positions_and_kv():
    class Blocks:
        seq_lens = {0: 15}
        override = None
        def set_decode_metadata_override(self, table, lengths, max_blocks):
            self.override = lengths
        def clear_decode_metadata_override(self):
            self.override = None
    blocks = Blocks()
    kv = torch.zeros(64, dtype=torch.long, device="cuda")
    class Model:
        calls = 0
        def decode_multi_step(self, ids, pos, bm, seqs, **kw):
            self.calls += 1
            tokens = ids + 1
            kv.index_copy_(0, pos.reshape(-1), tokens.reshape(-1))
            bm.override.add_(1)
            bm.seq_lens[0] += 1
            return tokens, None
    scheduler = object.__new__(Scheduler)
    scheduler.model, scheduler.block_manager = Model(), blocks
    scheduler._decode_graph_multi_step_body = True
    scheduler._decode_graph_capture_count = scheduler._decode_graph_replay_count = 0
    scheduler._decode_unrolled_graph_burst_captures = scheduler._decode_unrolled_graph_burst_replays = 0
    scheduler._log_decode_graph = lambda *a: None
    ids = torch.tensor([[5]], device="cuda")
    pos = torch.tensor([[15]], device="cuda")
    state = {"block_table": torch.zeros((1, 4), dtype=torch.int32, device="cuda"),
             "seq_lens": torch.tensor([15], dtype=torch.int32, device="cuda"),
             "table_blocks": 4, "max_decode_blocks": 4, "unrolled_burst_graphs": {},
             "block_signature": ((0, 1, 2, 3),)}
    # Warm the operator implementations, then restore the initial state.
    blocks.override = state["seq_lens"]
    scheduler.model.decode_multi_step(ids, pos, blocks, [0])
    blocks.override = None
    blocks.seq_lens[0] = 15
    state["seq_lens"].fill_(15)
    kv.zero_()
    with torch.inference_mode():
        scheduler._capture_decode_graph_shape((1, 4, 4), state, [0], ids, pos,
                                              return_next_token=True, chain_graph_inputs=True)
        calls = scheduler.model.calls
        state["graph"].replay()
        scheduler._advance_decode_graph_python_seq_lens([0])
        torch.cuda.synchronize()
        assert scheduler.model.calls == calls  # replay does not execute Python
        assert ids.item() == 7 and pos.item() == 17 and state["seq_lens"].item() == 17
        out = torch.empty((1, 8), dtype=torch.long, device="cuda")
        entry = scheduler._capture_decode_unrolled_graph_burst(state=state, seq_ids=[0], output_tokens=out, num_steps=8)
        torch.cuda.synchronize()
        assert out.tolist() == [list(range(8, 16))]
        calls = scheduler.model.calls
        entry["graph"].replay()
        torch.cuda.synchronize()
        assert scheduler.model.calls == calls
        assert out.tolist() == [list(range(16, 24))]
        assert ids.item() == 23 and pos.item() == 33 and state["seq_lens"].item() == 33
        assert kv[15:33].tolist() == list(range(6, 24))
