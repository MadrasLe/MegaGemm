from __future__ import annotations

from pathlib import Path

from benchmarks import run_gemma4_e2b_b8_compute_frontier as frontier


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "megagemm" / "kernels" / "swiglu.py"
MODEL = ROOT / "megagemm" / "models" / "llama.py"
HARNESS = (
    ROOT
    / "benchmarks"
    / "run_gemma4_e2b_b8_ple_projection_frontier_colab.sh"
)


def test_fused_ple_projection_preserves_bf16_boundaries_and_exact_shape():
    source = KERNEL.read_text(encoding="utf-8")
    for expected in (
        "def _mg_gemma4_e2b_b8_conditioned_gelu_projection_kernel(",
        "def gemma4_e2b_b8_conditioned_gelu_projection(",
        "tuple(gate.shape) != (8, 256)",
        "tuple(weight.shape) != (256, 1536)",
        "activated = (gelu_bf16.to(tl.float32) * condition).to(tl.bfloat16)",
        "tl.dot(activated, weight, out_dtype=tl.float32)",
        "WEIGHT_STRIDE_K",
        "WEIGHT_STRIDE_N",
    ):
        assert expected in source


def test_model_route_is_opt_in_exact_and_has_safe_fallback():
    source = MODEL.read_text(encoding="utf-8")
    for expected in (
        '"MEGAGEMM_GEMMA4_E2B_B8_FUSED_PLE_PROJECTION_DECODE"',
        'self.runtime_policy.name == "gemma4-e2b-l4"',
        "int(batch_size) == 8",
        "tuple(lw.ple_proj_wt.shape) == (256, 1536)",
        "ple_proj = gemma4_e2b_b8_conditioned_gelu_projection(",
        "_gemma4_flat_b8_fused_ple_projection_runtime_disabled = True",
        "if ple_proj is None:",
        "ple_proj = self._gemma4_flat_fp_linear(",
    ):
        assert expected in source


def test_frontier_sweeps_ple_projection_tiles_and_combines_winner():
    cases = {
        case.name: case
        for case in frontier.SCREEN_CASES
        if case.ple_projection
    }
    assert set(cases) == {
        "ple_proj_bn32_bk32_w4_s2",
        "ple_proj_bn64_bk32_w4_s2",
        "ple_proj_bn128_bk32_w4_s2",
        "ple_proj_bn64_bk64_w4_s2",
        "ple_proj_bn64_bk32_w8_s2",
        "ple_proj_bn64_bk32_w4_s3",
    }
    assert {
        (case.ple_projection_block_n, case.ple_projection_block_k)
        for case in cases.values()
    } >= {(32, 32), (64, 32), (128, 32), (64, 64)}
    winner = cases["ple_proj_bn128_bk32_w4_s2"]
    combined = frontier.combine_family_winners(
        (frontier.PRODUCTION, winner),
        {"ple": winner.name},
    )
    assert combined.ple_projection
    assert combined.ple_projection_block_n == 128


def test_colab_gate_uses_drive_and_one_load_full_model_frontier():
    source = HARNESS.read_text(encoding="utf-8")
    assert 'REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"' in source
    assert "run_gemma4_e2b_b8_compute_frontier_colab.sh" in source
    assert "SCREEN_PROMPT_TOKENS" in source
    assert "SCREEN_OUTPUT_TOKENS" in source
    assert "git pull" not in source
    assert "zip" not in source.lower()
