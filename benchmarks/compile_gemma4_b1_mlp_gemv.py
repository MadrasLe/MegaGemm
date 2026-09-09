"""Compile every B1 MLP specialization for sm_89, without a GPU/model load.

This is a compiler check, not a numerical or performance benchmark. It uses
Triton's explicit target API so GPU execution remains a separate validation.
"""

import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def compile_kernels(selected_configs=None):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from megagemm.kernels import gemma4_b1_mlp_gemv as kernels

    target = GPUTarget("cuda", 89, 32)
    results = []
    for operation, configs in (("gateup", kernels.GATEUP_CONFIGS), ("down", kernels.DOWN_CONFIGS)):
        for name, c in configs.items():
            if selected_configs is not None and name not in selected_configs.get(operation, ()):
                continue
            (n, k), _ = kernels.projection_spec(operation, name)
            for bias in (False, True):
                row = {"config": name, "bias": bias, "compiled": False}
                try:
                    constants = dict(N=n, K=k, BN=c.block_n, BK=c.block_k,
                                     SPLITS=c.splits, HAS_BIAS=bias)
                    signature = dict(X="*bf16", W="*bf16", Bias="*bf16",
                                     Out="*bf16", Partial="*fp32")
                    if not bias:
                        constants["Bias"] = None
                    if c.splits == 1:
                        constants["Partial"] = None
                    signature.update({key: "constexpr" for key in constants})
                    compiled = triton.compile(
                        ASTSource(kernels._b1_mlp_partial, signature, constexprs=constants),
                        target=target, options={"num_warps": c.warps, "num_stages": 1},
                    )
                    row["partial_shared_bytes"] = compiled.metadata.shared
                    if c.splits > 1:
                        constants = dict(N=n, SPLITS=c.splits, BS=triton.next_power_of_2(c.splits),
                                         HAS_BIAS=bias, BN=128)
                        signature = dict(Partial="*fp32", Bias="*bf16", Out="*bf16")
                        if not bias:
                            constants["Bias"] = None
                        signature.update({key: "constexpr" for key in constants})
                        compiled = triton.compile(
                            ASTSource(kernels._b1_mlp_reduce, signature, constexprs=constants),
                            target=target, options={"num_warps": 4, "num_stages": 1},
                        )
                        row["reduce_shared_bytes"] = compiled.metadata.shared
                    row["compiled"] = True
                except Exception:
                    row["error"] = traceback.format_exc()
                results.append(row)
                print("MLP_COMPILE " + json.dumps(row), flush=True)
    return {
        "target": "sm_89", "triton": triton.__version__, "model_loads": 0,
        "all_compiled": all(row["compiled"] for row in results),
        "any_biasless_compiled": any(row["compiled"] and not row["bias"] for row in results),
        "cases": results,
    }


if __name__ == "__main__":
    report = compile_kernels()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["all_compiled"] else 2)
