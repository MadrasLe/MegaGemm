from types import SimpleNamespace

import torch

from benchmarks import run_gemma4_e2b_b8_cublaslt_decode_frontier as gate
from megagemm.models import llama
from megagemm.models.llama import MegaGemmLlama


class _FakeTensor:
    def __init__(self, shape, *, contiguous=True):
        self.shape = shape
        self.is_cuda = True
        self.dtype = torch.bfloat16
        self._contiguous = contiguous
        self._transpose = None

    def is_contiguous(self):
        return self._contiguous

    def t(self):
        return self._transpose


def _fixtures():
    x = _FakeTensor((8, 1536))
    wt = _FakeTensor((1536, 2048), contiguous=False)
    raw_weight = _FakeTensor((2048, 1536))
    wt._transpose = raw_weight
    out = _FakeTensor((8, 2048))
    return x, wt, raw_weight, out


def test_exact_shape_dispatch_counts_hits(monkeypatch):
    x, wt, raw_weight, out = _fixtures()
    calls = []

    def native(input_tensor, weight, bias, *, out, algorithm_index):
        calls.append((input_tensor, weight, bias, out, algorithm_index))
        return out

    monkeypatch.setattr(llama, "cublaslt_bf16_linear_cuda", native)
    model = SimpleNamespace(
        _gemma4_flat_cublaslt_decode_enabled=True,
        _gemma4_flat_cublaslt_decode_algorithms={(8, 1536, 2048): 3},
        _gemma4_flat_cublaslt_decode_hits={},
        _gemma4_flat_cublaslt_decode_runtime_disabled=False,
        _gemma4_flat_cublaslt_decode_failure="",
        _flat_fp_linear=lambda *args: "fallback",
    )
    with torch.inference_mode():
        result = MegaGemmLlama._gemma4_flat_fp_linear(model, x, wt, None, out)
    assert result is out
    assert calls == [(x, raw_weight, None, out, 3)]
    assert model._gemma4_flat_cublaslt_decode_hits == {(8, 1536, 2048): 1}


def test_unselected_shape_falls_back_without_native_call(monkeypatch):
    x, wt, _, out = _fixtures()
    monkeypatch.setattr(
        llama,
        "cublaslt_bf16_linear_cuda",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("native called")),
    )
    model = SimpleNamespace(
        _gemma4_flat_cublaslt_decode_enabled=True,
        _gemma4_flat_cublaslt_decode_algorithms={(8, 1536, 1536): 0},
        _gemma4_flat_cublaslt_decode_hits={},
        _gemma4_flat_cublaslt_decode_runtime_disabled=False,
        _flat_fp_linear=lambda *args: "fallback",
    )
    with torch.inference_mode():
        assert MegaGemmLlama._gemma4_flat_fp_linear(model, x, wt, None, out) == "fallback"


def test_native_failure_disables_route_and_falls_back(monkeypatch):
    x, wt, _, out = _fixtures()

    def fail(*args, **kwargs):
        raise RuntimeError("bad heuristic")

    monkeypatch.setattr(llama, "cublaslt_bf16_linear_cuda", fail)
    model = SimpleNamespace(
        _gemma4_flat_cublaslt_decode_enabled=True,
        _gemma4_flat_cublaslt_decode_algorithms={(8, 1536, 2048): 7},
        _gemma4_flat_cublaslt_decode_hits={},
        _gemma4_flat_cublaslt_decode_runtime_disabled=False,
        _gemma4_flat_cublaslt_decode_failure="",
        _flat_fp_linear=lambda *args: "fallback",
    )
    with torch.inference_mode():
        assert MegaGemmLlama._gemma4_flat_fp_linear(model, x, wt, None, out) == "fallback"
    assert model._gemma4_flat_cublaslt_decode_runtime_disabled
    assert model._gemma4_flat_cublaslt_decode_failure == "RuntimeError: bad heuristic"


def test_shape_collection_counts_attention_and_ple_but_not_mlp():
    def weight(k, n):
        return SimpleNamespace(shape=(k, n))

    layers = [
        SimpleNamespace(
            qkv_wt=weight(1536, 2560), q_wt=None, k_wt=None, v_wt=None,
            o_wt=weight(2048, 1536), gate_up_wt=weight(1536, 24576),
            down_wt=weight(12288, 1536), ple_gate_wt=weight(1536, 256),
            ple_proj_wt=weight(256, 1536),
        ),
        SimpleNamespace(
            qkv_wt=None, q_wt=weight(1536, 2048), k_wt=None, v_wt=None,
            o_wt=weight(2048, 1536), gate_up_wt=weight(1536, 24576),
            down_wt=weight(12288, 1536), ple_gate_wt=weight(1536, 256),
            ple_proj_wt=weight(256, 1536),
        ),
    ]
    grouped = gate.collect_shapes(SimpleNamespace(_flat_layer_weights=layers), 8)
    assert (8, 1536, 24576) not in grouped
    assert (8, 12288, 1536) not in grouped
    assert grouped[(8, 2048, 1536)]["occurrences_per_token"] == 2
    assert grouped[(8, 1536, 256)]["operations"] == {"ple_gate": 2}
    assert grouped[(8, 256, 1536)]["operations"] == {"ple_proj": 2}


def test_colab_harness_uses_drive_and_temporary_native_build():
    script = (
        gate.ROOT
        / "benchmarks"
        / "run_gemma4_e2b_b8_cublaslt_decode_frontier_colab.sh"
    )
    source = script.read_text(encoding="utf-8")
    assert "/content/drive/MyDrive/mg/MGRrmsnorm" in source
    assert "mktemp -d /tmp/megagemm_cublaslt_decode" in source
    assert "MEGAGEMM_BUILD_ONLY_CUBLASLT=1" in source
    assert "git pull" not in source
    assert "pip install -e" not in source
    assert "vllm" not in source.lower()
