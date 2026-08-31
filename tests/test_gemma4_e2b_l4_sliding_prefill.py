import importlib
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "megagemm" / "kernels" / "paged_attention.py"
MODEL = ROOT / "megagemm" / "models" / "llama.py"
BENCHMARK = (
    ROOT / "benchmarks" / "run_gemma4_e2b_l4_sliding_prefill_microbench.py"
)
MATRIX = ROOT / "benchmarks" / "benchmark_inference_matrix.py"


def test_e2b_l4_candidate_rejects_non_cuda_without_allocating_output():
    paged_attention = importlib.import_module(
        "megagemm.kernels.paged_attention"
    )
    q = torch.empty(8, 8, 2, 256, dtype=torch.bfloat16)
    k = torch.empty(8, 1, 2, 256, dtype=torch.bfloat16)
    v = torch.empty_like(k)

    assert paged_attention.gemma4_e2b_l4_sliding_prefill_attention(
        q,
        k,
        v,
        sliding_window=512,
        force=True,
    ) is None


def test_e2b_l4_kernel_keeps_b4_and_b8_production_dispatches_exact():
    kernel = KERNEL.read_text(encoding="utf-8")
    model = MODEL.read_text(encoding="utf-8")

    for expected in (
        "def _gemma4_e2b_l4_sliding_prefill_kernel(",
        "def gemma4_e2b_l4_sliding_prefill_attention(",
        "tuple(k.shape) != (batch_size, 1, seq_len, 256)",
        'batch_size not in (4, 8)',
        "num_q_heads != 8",
        "seq_len < 2048",
        "seq_len > 2304",
        "int(sliding_window) != 512",
        'if "l4" not in _device_name_tokens(device_name):',
        "GROUP_HEADS=group_heads",
        "BLOCK_ROWS=block_rows",
        '"MEGAGEMM_GEMMA4_E2B_L4_B4_SLIDING_"',
        '"MEGAGEMM_GEMMA4_E2B_L4_SLIDING_"',
        'default_group_heads = 2 if batch_size == 4 else 4',
        'default_block_m = 16 if batch_size == 4 else 8',
        'env_prefix + "GROUP_HEADS", default_group_heads',
        'env_prefix + "BLOCK_M", default_block_m',
        'env_prefix + "NUM_WARPS", 4',
        '_GEMMA4_E2B_L4_B4_SLIDING_PREFILL_DISABLED',
    ):
        assert expected in kernel

    assert '"gemma4_e2b_l4_sliding_prefill"' in model
    assert "self._gemma4_e2b_l4_sliding_prefill_enabled" in model
    assert "and self._gemma4_e2b_l4_sliding_prefill_enabled" in model
    assert "self._gemma4_e2b_l4_sliding_prefill_hits += 1" in model
    assert "attn_mask is not None and not implicit_causal" in model


def test_e2b_l4_microbench_is_bounded_and_has_a_correctness_gate():
    source = BENCHMARK.read_text(encoding="utf-8")
    compile(source, str(BENCHMARK), "exec")

    for expected in (
        "B{batch_size} Q{q_heads}/KV{kv_heads}",
        "seq_len = int(args.seq_len)",
        "default=2057",
        '"--winner-only"',
        '"g1_bm32_bn64_w8_s2"',
        '"g2_bm16_bn64_w8_s2"',
        '"g4_bm8_bn64_w8_s2"',
        '"g8_bm4_bn64_w8_s2"',
        "repeat_exact",
        "cosine >= 0.9999",
        "max_abs_error <= 0.125",
        '"TEST_FULL_MODEL" if apply_change else "KEEP_SDPA"',
        "estimated_savings_ms_28_layers",
        "conservative_speedup",
        "sample_ranges_dominate",
    ):
        assert expected in source

    assert "from_pretrained" not in source
    assert "snapshot_download" not in source
    assert "pip install" not in source
    assert "import vllm" not in source.lower()


def test_publication_artifact_records_the_experimental_kernel_configuration():
    source = MATRIX.read_text(encoding="utf-8")

    for name in (
        "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_PREFILL",
        "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_GROUP_HEADS",
        "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_BLOCK_M",
        "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_BLOCK_N",
        "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_NUM_WARPS",
        "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_NUM_STAGES",
    ):
        assert f'"{name}"' in source
