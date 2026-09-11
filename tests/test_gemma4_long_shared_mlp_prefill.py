import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "run_gemma4_long_shared_mlp_prefill_microbench.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_long_shared_mlp_prefill_colab.sh"
KERNEL = ROOT / "megagemm" / "kernels" / "swiglu.py"


class Gemma4LongSharedMlpPrefillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = BENCHMARK.read_text(encoding="utf-8")
        cls.harness = HARNESS.read_text(encoding="utf-8")
        cls.kernel = KERNEL.read_text(encoding="utf-8")

    def test_gate_uses_exact_long_chunk_and_shared_mlp_shape(self):
        for expected in (
            "rows = 16_384",
            "hidden_dim = 2_816",
            "intermediate_dim = 2_112",
            "layers = 30",
            "chunks = 2",
        ):
            self.assertIn(expected, self.benchmark)

    def test_gate_measures_full_current_production_sequence(self):
        for expected in (
            "torch.mm(hidden, gate_up_weight_t, out=gate_up_out)",
            'F.gelu(gate, approximate="tanh")',
            "activated.mul_(up)",
            "torch.mm(activated, down_weight_t, out=down_out)",
            'profile_case("current_recheck", make_current)',
        ):
            self.assertIn(expected, self.benchmark)

    def test_gate_has_one_bounded_candidate_family_and_control(self):
        self.assertIn("for block_size in (128, 256, 512, 1024):", self.benchmark)
        self.assertIn('profile_case("deepfusion_gelu_down", make_deepfusion)', self.benchmark)
        self.assertIn('activation="gelu_tanh"', self.benchmark)
        self.assertIn(
            'os.environ.setdefault("MEGAGEMM_DEEPFUSION_PREFILL_FORCE_TRITON", "1")',
            self.benchmark,
        )
        self.assertIn(
            "export MEGAGEMM_DEEPFUSION_PREFILL_FORCE_TRITON=1",
            self.harness,
        )
        self.assertNotIn("snapshot_download", self.benchmark)

    def test_fused_kernel_uses_exact_gelu_tanh_formula(self):
        for expected in (
            "def _mg_gated_activation_fwd_kernel(",
            "0.7978845608028654",
            "0.044715 * gate * gate * gate",
            "libdevice.tanh(inner)",
            "activated.to(tl.bfloat16).to(tl.float32)",
            "def gated_activation_forward(",
            '"gelu_pytorch_tanh"',
        ):
            self.assertIn(expected, self.kernel)

    def test_promotion_requires_correct_stable_full_path_speedup(self):
        for expected in (
            "repeat_exact",
            "cosine >= 0.9999",
            "max_abs_error <= 0.125",
            'stable_baseline = by_name["current_recheck"]',
            "maximum_baseline_spread",
            "maximum_candidate_spread",
            "speedup >= float(args.minimum_speedup)",
            "near_tied_candidates",
            "estimated_gap_coverage",
        ):
            self.assertIn(expected, self.benchmark)
        self.assertRegex(self.benchmark, r"\n\s+return 0\n")

    def test_harness_is_bounded_and_has_no_paid_setup(self):
        self.assertIn(
            "harness_rev: gemma4-long-shared-mlp-prefill-v1-fused-gelu-mul",
            self.harness,
        )
        self.assertIn('BENCH_TIMEOUT_MIN="${BENCH_TIMEOUT_MIN:-5}"', self.harness)
        self.assertIn("model_download: disabled", self.harness)
        self.assertIn("vllm_install: disabled", self.harness)
        self.assertIn("package_install: disabled", self.harness)
        self.assertNotIn("pip install", self.harness)
        self.assertNotIn("snapshot_download", self.harness)
        self.assertNotIn("run_gemma4_long_context_vs_vllm", self.harness)

    def test_embedded_python_compiles(self):
        blocks = re.findall(r"<<'PY'\r?\n(.*?)\r?\nPY", self.harness, re.DOTALL)
        self.assertEqual(len(blocks), 1)
        compile(blocks[0], "long_shared_mlp_preflight.py", "exec")


if __name__ == "__main__":
    unittest.main()
