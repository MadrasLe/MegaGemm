import contextlib
from types import SimpleNamespace

import torch

from megagemm.engine.scheduler import Request, RequestStatus, Scheduler


class _FakeBlockManager:
    block_size = 16
    num_blocks = 128
    bytes_per_block = 1024

    def __init__(self):
        self.block_tables = {}
        self.seq_lens = {}
        self._next_block = 0

    def can_allocate(self, num_tokens):
        return True

    def allocate_sequence(self, seq_id, num_tokens=0):
        block_count = max(1, (int(num_tokens) + self.block_size - 1) // self.block_size)
        blocks = list(range(self._next_block, self._next_block + block_count))
        self._next_block += block_count
        self.block_tables[seq_id] = blocks
        self.seq_lens[seq_id] = 0
        return blocks

    def free_sequence(self, seq_id):
        self.block_tables.pop(seq_id, None)
        self.seq_lens.pop(seq_id, None)


class _PrefillOnlyModel:
    def prefill(self, input_ids, positions, block_manager, seq_id):
        block_manager.seq_lens[seq_id] = int(input_ids.shape[1])
        logits = torch.zeros(1, input_ids.shape[1], 8)
        logits[:, -1, 3] = 1.0
        return logits


class _PackedPrefillModel:
    def prefill_packed(self, input_ids, cu_seqlens, lengths, block_manager, seq_ids):
        for sid, length in zip(seq_ids, lengths.tolist()):
            block_manager.seq_lens[sid] = int(length)
        logits = torch.zeros(len(seq_ids), 1, 8)
        logits[:, -1, 3] = 1.0
        return logits


class _GreedyTokenDecodeModel:
    def __init__(self):
        self.return_next_token_calls = []
        self.last_input_ids = None
        self.last_positions = None

    @staticmethod
    def prefers_scheduler_greedy_token_decode(num_seqs):
        return int(num_seqs) == 2

    def decode_step(
        self,
        input_ids,
        positions,
        block_manager,
        seq_ids,
        return_next_token=False,
    ):
        del block_manager, seq_ids
        self.return_next_token_calls.append(bool(return_next_token))
        self.last_input_ids = input_ids.clone()
        self.last_positions = positions.clone()
        if return_next_token:
            return torch.tensor([5, 6], dtype=torch.long)
        raise AssertionError("greedy-token contract was not requested")


class _GreedyTokenBurstModel:
    def __init__(self):
        self.inputs = []
        self.positions = []

    @staticmethod
    def prefers_scheduler_greedy_token_decode(num_seqs):
        return int(num_seqs) == 2

    def decode_step(
        self,
        input_ids,
        positions,
        block_manager,
        seq_ids,
        return_next_token=False,
    ):
        assert return_next_token
        self.inputs.append(input_ids.clone())
        self.positions.append(positions.clone())
        for seq_id in seq_ids:
            block_manager.seq_lens[seq_id] += 1
        return input_ids[:, 0] + 1


class _MultiStepCapableModel:
    def decode_multi_step(self, *args, **kwargs):
        raise AssertionError("the route test replaces the scheduler method")


def _add_running_greedy_request(scheduler):
    scheduler._running[0] = Request(
        request_id=0,
        seq_id=0,
        prompt_ids=[1, 2],
        generated_ids=[3],
        status=RequestStatus.RUNNING,
        max_new_tokens=3,
        temperature=0.0,
    )


def test_scheduler_returns_request_completed_during_single_prefill():
    scheduler = Scheduler(
        _PrefillOnlyModel(),
        _FakeBlockManager(),
        max_batch_size=4,
        device="cpu",
    )
    scheduler.add_request([1, 2, 3], max_new_tokens=1, temperature=0.0)

    completed = scheduler.step()

    assert len(completed) == 1
    assert completed[0].generated_ids == [3]
    assert len(scheduler._completed) == 1
    assert scheduler.has_pending() is False


def test_scheduler_can_prefer_eager_decode_step_without_cuda_graphs(monkeypatch):
    monkeypatch.setenv("MEGAGEMM_DECODE_PREFER_STEP", "1")
    monkeypatch.setenv("MEGAGEMM_DECODE_CUDA_GRAPHS", "0")
    scheduler = Scheduler(
        _MultiStepCapableModel(),
        _FakeBlockManager(),
        max_batch_size=1,
        device="cpu",
    )
    _add_running_greedy_request(scheduler)
    routes = []
    monkeypatch.setattr(
        scheduler, "_decode_batch", lambda: routes.append("step") or []
    )
    monkeypatch.setattr(
        scheduler,
        "_decode_multi_step_batch",
        lambda _burst: routes.append("multi") or [],
    )

    scheduler.step()

    assert routes == ["step"]
    stats = scheduler.get_stats()["decode_execution"]
    assert stats == {
        "prefer_step": True,
        "decode_step_batches": 1,
        "multi_step_batches": 0,
    }
    assert scheduler._decode_cuda_graphs is False


def test_scheduler_keeps_multi_step_as_default(monkeypatch):
    monkeypatch.delenv("MEGAGEMM_DECODE_PREFER_STEP", raising=False)
    monkeypatch.setenv("MEGAGEMM_DECODE_CUDA_GRAPHS", "0")
    scheduler = Scheduler(
        _MultiStepCapableModel(),
        _FakeBlockManager(),
        max_batch_size=1,
        device="cpu",
    )
    _add_running_greedy_request(scheduler)
    routes = []
    monkeypatch.setattr(
        scheduler, "_decode_batch", lambda: routes.append("step") or []
    )
    monkeypatch.setattr(
        scheduler,
        "_decode_multi_step_batch",
        lambda _burst: routes.append("multi") or [],
    )

    scheduler.step()

    assert routes == ["multi"]
    stats = scheduler.get_stats()["decode_execution"]
    assert stats == {
        "prefer_step": False,
        "decode_step_batches": 0,
        "multi_step_batches": 1,
    }


def test_scheduler_returns_requests_completed_during_packed_prefill():
    scheduler = Scheduler(
        _PackedPrefillModel(),
        _FakeBlockManager(),
        max_batch_size=4,
        device="cpu",
    )
    scheduler._chunk_log_count = 5
    scheduler._prefill_choice_log_count = 5
    scheduler.add_request([1, 2, 3], max_new_tokens=1, temperature=0.0)
    scheduler.add_request([4, 5], max_new_tokens=1, temperature=0.0)

    completed = scheduler.step()

    assert len(completed) == 2
    assert [req.generated_ids for req in completed] == [[3], [3]]
    assert len(scheduler._completed) == 2
    assert scheduler.has_pending() is False


def test_scheduler_force_sequential_prefill_bypasses_batched_entry_point():
    class _DualPrefillModel(_PrefillOnlyModel):
        _force_sequential_prefill = True

        def __init__(self):
            self.prefill_calls = []
            self.packed_calls = 0

        def prefill(self, input_ids, positions, block_manager, seq_id):
            self.prefill_calls.append(int(seq_id))
            return super().prefill(input_ids, positions, block_manager, seq_id)

        def prefill_packed(
            self,
            input_ids,
            cu_seqlens,
            lengths,
            block_manager,
            seq_ids,
        ):
            self.packed_calls += 1
            logits = torch.zeros(len(seq_ids), 1, 8)
            logits[:, -1, 7] = 1.0
            return logits

    model = _DualPrefillModel()
    scheduler = Scheduler(
        model,
        _FakeBlockManager(),
        max_batch_size=4,
        device="cpu",
    )
    scheduler.add_request([1, 2, 3], max_new_tokens=1, temperature=0.0)
    scheduler.add_request([4, 5], max_new_tokens=1, temperature=0.0)

    completed = scheduler.step()

    assert len(completed) == 2
    assert [req.generated_ids for req in completed] == [[3], [3]]
    assert model.prefill_calls == [int(req.seq_id) for req in completed]
    assert model.packed_calls == 0


def test_scheduler_greedy_token_decode_uses_batched_transfers():
    model = _GreedyTokenDecodeModel()
    blocks = _FakeBlockManager()
    scheduler = Scheduler(model, blocks, max_batch_size=2, device="cpu")
    requests = []
    for seq_id, prompt, first_token in (
        (0, [1, 2, 3], 11),
        (1, [4, 5], 12),
    ):
        blocks.allocate_sequence(seq_id, len(prompt))
        blocks.seq_lens[seq_id] = len(prompt)
        request = Request(
            request_id=seq_id,
            seq_id=seq_id,
            prompt_ids=prompt,
            generated_ids=[first_token],
            status=RequestStatus.RUNNING,
            max_new_tokens=3,
            temperature=0.0,
        )
        scheduler._running[seq_id] = request
        requests.append(request)

    assert scheduler._decode_batch() == []
    assert model.return_next_token_calls == [True]
    assert model.last_input_ids.tolist() == [[11], [12]]
    assert model.last_positions.tolist() == [[3], [2]]
    assert [request.generated_ids for request in requests] == [[11, 5], [12, 6]]
    assert scheduler._decode_greedy_token_steps == 1
    assert scheduler._decode_batched_token_host_copies == 1
    assert scheduler._decode_vectorized_input_updates == 1


def test_scheduler_graph_token_burst_keeps_feedback_on_device():
    model = _GreedyTokenBurstModel()
    blocks = _FakeBlockManager()
    scheduler = Scheduler(model, blocks, max_batch_size=2, device="cpu")
    requests = []
    for seq_id, prompt, first_token in (
        (0, [1, 2, 3], 11),
        (1, [4, 5], 21),
    ):
        blocks.allocate_sequence(seq_id, len(prompt))
        blocks.seq_lens[seq_id] = len(prompt)
        request = Request(
            request_id=seq_id,
            seq_id=seq_id,
            prompt_ids=prompt,
            generated_ids=[first_token],
            status=RequestStatus.RUNNING,
            max_new_tokens=4,
            temperature=0.0,
        )
        scheduler._running[seq_id] = request
        requests.append(request)

    finished = scheduler._decode_graph_token_burst_batch(3)

    assert finished == requests
    assert [request.generated_ids for request in requests] == [
        [11, 12, 13, 14],
        [21, 22, 23, 24],
    ]
    assert [value.tolist() for value in model.inputs] == [
        [[11], [21]],
        [[12], [22]],
        [[13], [23]],
    ]
    assert [value.tolist() for value in model.positions] == [
        [[3], [2]],
        [[4], [3]],
        [[5], [4]],
    ]
    assert scheduler._decode_graph_token_bursts == 1
    assert scheduler._decode_graph_token_burst_steps == 3
    assert scheduler._decode_graph_token_feedback_copies == 2
    assert scheduler._decode_batched_token_host_copies == 1
    assert scheduler._decode_vectorized_input_updates == 1


def test_scheduler_uses_model_policy_for_b8_graph_burst_capacity(monkeypatch):
    from megagemm.models.runtime_policy import resolve_runtime_policy

    monkeypatch.delenv("MEGAGEMM_MULTI_STEP_BURST_BATCH", raising=False)
    monkeypatch.delenv("MEGAGEMM_MULTI_STEP_BURST", raising=False)
    config = SimpleNamespace(
        model_type="gemma4_text",
        num_hidden_layers=35,
        hidden_size=1536,
        num_attention_heads=8,
        num_key_value_heads=1,
    )
    model = SimpleNamespace(
        runtime_policy=resolve_runtime_policy(config, "NVIDIA L4")
    )
    scheduler = Scheduler(
        model, _FakeBlockManager(), max_batch_size=8, device="cpu"
    )

    assert scheduler._decode_multi_step_burst == 8
    assert scheduler._decode_graph_token_burst_by_batch == {4: 8, 8: 16}
    assert scheduler._decode_burst_capacity == 16
    assert scheduler._decode_burst_tokens.shape == (8, 16)


def test_explicit_burst_override_suppresses_model_batch_policy(monkeypatch):
    from megagemm.models.runtime_policy import resolve_runtime_policy

    monkeypatch.setenv("MEGAGEMM_MULTI_STEP_BURST_BATCH", "4")
    config = SimpleNamespace(
        model_type="gemma4_text",
        num_hidden_layers=35,
        hidden_size=1536,
        num_attention_heads=8,
        num_key_value_heads=1,
    )
    model = SimpleNamespace(
        runtime_policy=resolve_runtime_policy(config, "NVIDIA L4")
    )
    scheduler = Scheduler(
        model, _FakeBlockManager(), max_batch_size=8, device="cpu"
    )

    assert scheduler._decode_multi_step_burst == 4
    assert scheduler._decode_graph_token_burst_by_batch == {}
    assert scheduler._decode_burst_capacity == 4


def test_scheduler_native_graph_burst_avoids_python_decode_loop(monkeypatch):
    import importlib

    scheduler_module = importlib.import_module("megagemm.engine.scheduler")
    blocks = _FakeBlockManager()
    scheduler = Scheduler(
        _GreedyTokenBurstModel(), blocks, max_batch_size=2, device="cpu"
    )
    requests = []
    for seq_id, prompt, first_token in (
        (0, [1, 2, 3], 11),
        (1, [4, 5], 21),
    ):
        blocks.allocate_sequence(seq_id, len(prompt))
        blocks.seq_lens[seq_id] = len(prompt)
        request = Request(
            request_id=seq_id,
            seq_id=seq_id,
            prompt_ids=prompt,
            generated_ids=[first_token],
            status=RequestStatus.RUNNING,
            max_new_tokens=5,
            temperature=0.0,
        )
        scheduler._running[seq_id] = request
        requests.append(request)

    graph_tokens = torch.tensor([11, 21], dtype=torch.long)

    class _FakeGraph:
        def replay(self):
            graph_tokens.add_(1)

    class _FakeNativeOps:
        @staticmethod
        def run_cuda_graph_token_burst(graph, tokens, output, num_steps):
            assert tokens is graph_tokens
            for step in range(int(num_steps)):
                graph.replay()
                output[:, step].copy_(tokens)

    seq_ids = [0, 1]
    scheduler._decode_cuda_graphs = True
    scheduler._decode_cuda_graph_shape_cache = True
    scheduler._decode_graph_persistent_token_feedback = True
    scheduler._decode_native_graph_burst = True
    scheduler._batch_changed = False
    key = scheduler._decode_graph_shape_key(
        seq_ids,
        return_next_token=True,
        chain_graph_inputs=True,
    )
    scheduler._decode_graph_shape_states[key] = {
        "graph": _FakeGraph(),
        "logits": graph_tokens,
        "seq_key": tuple(seq_ids),
        "block_signature": tuple(
            tuple(blocks.block_tables[seq_id]) for seq_id in seq_ids
        ),
        "return_next_token": True,
    }
    scheduler._decode_graph_chain_started_keys.add(key)
    monkeypatch.setattr(scheduler_module, "_decode_native_ops", _FakeNativeOps())
    monkeypatch.setattr(
        scheduler,
        "_run_decode_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("native burst returned to the Python decode loop")
        ),
    )

    finished = scheduler._decode_graph_token_burst_batch(3)

    assert finished == []
    assert [request.generated_ids for request in requests] == [
        [11, 12, 13, 14],
        [21, 22, 23, 24],
    ]
    assert blocks.seq_lens == {0: 6, 1: 5}
    assert scheduler._decode_graph_replay_count == 3
    assert scheduler._decode_native_graph_bursts == 1
    assert scheduler._decode_native_graph_burst_steps == 3
    stats = scheduler.get_stats()["decode_cuda_graphs"]
    assert stats["native_token_burst_enabled"] is True
    assert stats["native_token_bursts"] == 1
    assert stats["native_token_burst_steps"] == 3


def test_scheduler_unrolled_graph_capture_chains_tokens_and_metadata(monkeypatch):
    class _CaptureBlockManager(_FakeBlockManager):
        def __init__(self):
            super().__init__()
            self.override = None

        def set_decode_metadata_override(
            self,
            block_table,
            seq_lens,
            max_decode_blocks=None,
        ):
            self.override = {
                "block_table": block_table,
                "seq_lens": seq_lens,
                "max_decode_blocks": max_decode_blocks,
            }

        def clear_decode_metadata_override(self):
            self.override = None

    class _CaptureModel:
        def decode_step(
            self,
            input_ids,
            positions,
            block_manager,
            seq_ids,
            return_next_token=False,
        ):
            assert return_next_token is True
            next_tokens = input_ids.reshape(len(seq_ids)) + 1
            for seq_id in seq_ids:
                block_manager.seq_lens[int(seq_id)] += 1
            block_manager.override["seq_lens"].add_(1)
            return next_tokens

    class _FakeCudaGraph:
        def __init__(self):
            self.replays = 0

        def replay(self):
            self.replays += 1

    monkeypatch.setattr(torch.cuda, "CUDAGraph", _FakeCudaGraph)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "graph",
        lambda graph: contextlib.nullcontext(),
    )

    blocks = _CaptureBlockManager()
    for seq_id, length in ((0, 3), (1, 2)):
        blocks.allocate_sequence(seq_id, length)
        blocks.seq_lens[seq_id] = length
    scheduler = Scheduler(_CaptureModel(), blocks, max_batch_size=2, device="cpu")
    state = {
        "graph_input_ids": torch.tensor([[11], [21]], dtype=torch.long),
        "graph_positions": torch.tensor([[3], [2]], dtype=torch.long),
        "block_table": torch.tensor([[0], [1]], dtype=torch.int32),
        "seq_lens": torch.tensor([3, 2], dtype=torch.int32),
        "max_decode_blocks": 1,
        "block_signature": ((0,), (1,)),
        "unrolled_burst_graphs": {},
    }
    output = torch.empty(2, 3, dtype=torch.long)

    entry = scheduler._capture_decode_unrolled_graph_burst(
        state=state,
        seq_ids=[0, 1],
        output_tokens=output,
        num_steps=3,
    )

    assert output.tolist() == [[12, 13, 14], [22, 23, 24]]
    assert state["graph_input_ids"].tolist() == [[14], [24]]
    assert state["graph_positions"].tolist() == [[6], [5]]
    assert state["seq_lens"].tolist() == [6, 5]
    assert blocks.seq_lens == {0: 6, 1: 5}
    assert blocks.override is None
    assert entry["num_steps"] == 3
    assert entry["graph"].replays == 1
    assert state["unrolled_burst_graphs"][3] is entry
    assert scheduler._decode_unrolled_graph_burst_captures == 1
    assert scheduler._decode_unrolled_graph_burst_replays == 1


def test_scheduler_shape_graph_capture_replays_before_returning(monkeypatch):
    class _CaptureBlockManager(_FakeBlockManager):
        def __init__(self):
            super().__init__()
            self.override = None

        def set_decode_metadata_override(
            self,
            block_table,
            seq_lens,
            max_decode_blocks=None,
        ):
            self.override = {
                "block_table": block_table,
                "seq_lens": seq_lens,
                "max_decode_blocks": max_decode_blocks,
            }

        def clear_decode_metadata_override(self):
            self.override = None

    class _CaptureModel:
        def decode_step(
            self,
            input_ids,
            positions,
            block_manager,
            seq_ids,
            return_next_token=False,
        ):
            assert return_next_token is True
            for seq_id in seq_ids:
                block_manager.seq_lens[int(seq_id)] += 1
            block_manager.override["seq_lens"].add_(1)
            return input_ids.reshape(len(seq_ids)) + 1

    class _FakeCudaGraph:
        def __init__(self):
            self.replays = 0

        def replay(self):
            self.replays += 1

    monkeypatch.setattr(torch.cuda, "CUDAGraph", _FakeCudaGraph)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "graph",
        lambda graph: contextlib.nullcontext(),
    )

    blocks = _CaptureBlockManager()
    for seq_id, length in ((0, 3), (1, 2)):
        blocks.allocate_sequence(seq_id, length)
        blocks.seq_lens[seq_id] = length
    scheduler = Scheduler(_CaptureModel(), blocks, max_batch_size=2, device="cpu")
    state = {
        "block_table": torch.tensor([[0], [1]], dtype=torch.int32),
        "seq_lens": torch.tensor([3, 2], dtype=torch.int32),
        "max_decode_blocks": 1,
    }
    input_ids = torch.tensor([[11], [21]], dtype=torch.long)
    positions = torch.tensor([[3], [2]], dtype=torch.long)

    tokens = scheduler._capture_decode_graph_shape(
        (2, 1, 1),
        state,
        [0, 1],
        input_ids,
        positions,
        return_next_token=True,
        chain_graph_inputs=True,
    )

    assert tokens.tolist() == [12, 22]
    assert state["graph"].replays == 1
    assert scheduler._decode_graph_capture_count == 1
    assert scheduler._decode_graph_replay_count == 1
    assert blocks.override is None


def test_scheduler_unrolled_graph_burst_replays_once_for_all_steps(monkeypatch):
    blocks = _FakeBlockManager()
    scheduler = Scheduler(
        _GreedyTokenBurstModel(), blocks, max_batch_size=2, device="cpu"
    )
    requests = []
    for seq_id, prompt, first_token in (
        (0, [1, 2, 3], 11),
        (1, [4, 5], 21),
    ):
        blocks.allocate_sequence(seq_id, len(prompt))
        blocks.seq_lens[seq_id] = len(prompt)
        request = Request(
            request_id=seq_id,
            seq_id=seq_id,
            prompt_ids=prompt,
            generated_ids=[first_token],
            status=RequestStatus.RUNNING,
            max_new_tokens=5,
            temperature=0.0,
        )
        scheduler._running[seq_id] = request
        requests.append(request)

    graph_output = torch.empty(2, 3, dtype=torch.long)

    class _FakeUnrolledGraph:
        def __init__(self):
            self.replays = 0

        def replay(self):
            self.replays += 1
            graph_output.copy_(torch.tensor([[12, 13, 14], [22, 23, 24]]))

    graph = _FakeUnrolledGraph()
    seq_ids = [0, 1]
    scheduler._decode_cuda_graphs = True
    scheduler._decode_cuda_graph_shape_cache = True
    scheduler._decode_graph_persistent_token_feedback = True
    scheduler._decode_unrolled_graph_burst = True
    scheduler._batch_changed = False
    key = scheduler._decode_graph_shape_key(
        seq_ids,
        return_next_token=True,
        chain_graph_inputs=True,
    )
    scheduler._decode_graph_shape_states[key] = {
        "graph": object(),
        "logits": torch.tensor([11, 21]),
        "seq_key": tuple(seq_ids),
        "block_signature": tuple(
            tuple(blocks.block_tables[seq_id]) for seq_id in seq_ids
        ),
        "return_next_token": True,
        "unrolled_burst_graphs": {
            3: {
                "graph": graph,
                "output_tokens": graph_output,
                "num_steps": 3,
                "seq_key": tuple(seq_ids),
                "block_signature": tuple(
                    tuple(blocks.block_tables[seq_id]) for seq_id in seq_ids
                ),
            }
        },
    }
    scheduler._decode_graph_chain_started_keys.add(key)
    monkeypatch.setattr(
        scheduler,
        "_decode_unrolled_graph_burst_is_eligible",
        lambda seq_ids, num_steps: True,
    )
    monkeypatch.setattr(
        scheduler,
        "_run_decode_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unrolled burst returned to the one-token decode loop")
        ),
    )

    finished = scheduler._decode_graph_token_burst_batch(3)

    assert finished == []
    assert graph.replays == 1
    assert [request.generated_ids for request in requests] == [
        [11, 12, 13, 14],
        [21, 22, 23, 24],
    ]
    assert blocks.seq_lens == {0: 6, 1: 5}
    assert scheduler._decode_graph_replay_count == 1
    assert scheduler._decode_unrolled_graph_burst_replays == 1
    assert scheduler._decode_unrolled_graph_bursts == 1
    assert scheduler._decode_unrolled_graph_burst_steps == 3
    stats = scheduler.get_stats()["decode_cuda_graphs"]
    assert stats["unrolled_token_burst_enabled"] is True
    assert stats["unrolled_token_burst_replays"] == 1
    assert stats["unrolled_token_bursts"] == 1
    assert stats["unrolled_token_burst_steps"] == 3


def test_shared_decode_graph_is_invalidated_on_physical_kv_rebind(monkeypatch):
    monkeypatch.setenv("MEGAGEMM_DECODE_CUDA_GRAPHS_SHARED_SHAPE_CACHE", "1")
    blocks = _FakeBlockManager()
    blocks.allocate_sequence(0, 16)
    blocks.seq_lens[0] = 8
    scheduler = Scheduler(
        _PrefillOnlyModel(), blocks, max_batch_size=1, device="cpu"
    )

    key, state = scheduler._prepare_decode_graph_shape_state([0], True)
    captured_graph = object()
    state["graph"] = captured_graph
    state["logits"] = torch.tensor([3])
    scheduler._decode_graph_shape_warm_keys.add(key)

    blocks.block_tables[0] = [17]
    same_key, rebound_state = scheduler._prepare_decode_graph_shape_state([0], True)

    assert same_key == key
    assert rebound_state is state
    assert rebound_state["graph"] is None
    assert rebound_state["logits"] is None
    assert key not in scheduler._decode_graph_shape_warm_keys
    assert scheduler._decode_graph_physical_rebinds == 1


def test_idle_scheduler_reset_preserves_owned_graph_and_resets_request_state(
    monkeypatch,
):
    monkeypatch.setenv("MEGAGEMM_REUSE_REQUEST_SCHEDULER", "1")
    monkeypatch.setenv("MEGAGEMM_DECODE_CUDA_GRAPHS", "1")
    blocks = _FakeBlockManager()
    model = _PrefillOnlyModel()
    scheduler = Scheduler(model, blocks, max_batch_size=2, device="cpu")
    scheduler._completed.append(
        Request(request_id=1, seq_id=1, prompt_ids=[1], generated_ids=[3])
    )
    scheduler._decode_graph_capture_count = 1
    scheduler._decode_graph_replay_count = 7
    scheduler._decode_graph_shape_states[(2, 1)] = {"graph": object()}
    scheduler._decode_graph_shape_warm_keys.add((2, 1))

    assert scheduler.can_reuse_for_request(
        model=model,
        block_manager=blocks,
        max_batch_size=2,
        device="cpu",
    )
    scheduler.reset_for_request(
        prefill_capture_hook=None,
        materialize_generated_tokens=True,
    )

    assert scheduler._completed == []
    assert scheduler._req_counter == 0
    assert scheduler._seq_counter == 0
    assert scheduler._decode_graph_capture_count == 0
    assert scheduler._decode_graph_replay_count == 0
    assert scheduler._decode_graph_shape_states[(2, 1)]["graph"] is not None
    assert (2, 1) in scheduler._decode_graph_shape_warm_keys
    stats = scheduler.get_stats()["decode_cuda_graphs"]
    assert stats["request_scheduler_reuse_enabled"] is True
    assert stats["request_scheduler_reused"] is True
    assert stats["request_scheduler_reuse_count"] == 1

    blocks.allocate_sequence(1, 1)
    assert not scheduler.can_reuse_for_request(
        model=model,
        block_manager=blocks,
        max_batch_size=2,
        device="cpu",
    )


def test_idle_scheduler_reuse_rejects_runtime_knob_changes(monkeypatch):
    monkeypatch.setenv("MEGAGEMM_REUSE_REQUEST_SCHEDULER", "1")
    blocks = _FakeBlockManager()
    model = _PrefillOnlyModel()
    scheduler = Scheduler(model, blocks, max_batch_size=1, device="cpu")

    monkeypatch.setenv("MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID", "7")

    assert not scheduler.can_reuse_for_request(
        model=model,
        block_manager=blocks,
        max_batch_size=1,
        device="cpu",
    )
