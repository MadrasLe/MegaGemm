"""Exact-shape GEMV cases and correctness preflight for the one-load gate."""

from dataclasses import asdict
import json

import torch

from megagemm.kernels.gemma4_b1_mlp_gemv import (
    B1MlpGemvPlan, GATEUP_CONFIGS, DOWN_CONFIGS,
)


def build_cases():
    from benchmarks.run_gemma4_e2b_b1_decode_frontier_gate import DecodeCase

    return (
        DecodeCase("mlp_gemv", "production", bridge=True),
        *(DecodeCase("mlp_gemv", name, bridge=True, gemv_gateup=name)
          for name in GATEUP_CONFIGS),
        *(DecodeCase("mlp_gemv", name, bridge=True, gemv_down=name)
          for name in DOWN_CONFIGS),
        *(DecodeCase("mlp_gemv", f"both_{index}", bridge=True,
                     gemv_gateup=gateup, gemv_down=down)
          for index, (gateup, down) in enumerate((
              ("wide_n4_k256_w4", "down_n4_k512_s4_w4"),
              ("wide_n4_k2048_w4", "down_n4_k512_s8_w4"),
              ("wide_n8_k256_w4", "down_n8_k256_s8_w4"),
          ), 1)),
    )


def build_dispatch_cases():
    from benchmarks.run_gemma4_e2b_b1_decode_frontier_gate import DecodeCase

    cases = [DecodeCase("mlp_gemv", "production", bridge=True)]
    for label, gateup, down in (
        ("gateup", "wide_n4_k2048_w4", ""),
        ("down", "", "down_n4_k256_s1_w4"),
        ("both", "wide_n4_k2048_w4", "down_n4_k256_s1_w4"),
    ):
        for prepared in (False, True):
            cases.append(DecodeCase(
                "mlp_gemv", f"{'prepared' if prepared else 'legacy'}_{label}",
                bridge=True, gemv_gateup=gateup, gemv_down=down, prepared_gemv=prepared,
            ))
    return tuple(cases)


@torch.inference_mode()
def validate_projections(model, selected_configs=None):
    """Check both projections of every large layer, without latency selection.

    Two seeded BF16 inputs per weight; down receives the production BF16
    GELU-tanh(gate)*up result. Poisoning output/scratch detects incomplete writes.
    Full-model unforced greedy and timed route audits are performed by the gate.
    """
    from benchmarks.run_gemma4_e2b_b1_decode_frontier_gate import _tensor_error

    large = [(i, lw) for i, lw in enumerate(model._flat_layer_weights)
             if not lw.is_moe and int(lw.intermediate_size) == 12288]
    report = {"large_layer_count": len(large), "projections": {}, "all_correct": False}
    if len(large) != 20:
        report["error"] = f"Expected 20 large layers, found {len(large)}"
        return report
    for operation, configs in (("gateup", GATEUP_CONFIGS), ("down", DOWN_CONFIGS)):
        for name, config in configs.items():
            if selected_configs is not None and name not in selected_configs.get(operation, ()):
                continue
            row = {"correct": True, "config": asdict(config), "operation": operation,
                   "checked_inputs": 0, "max_abs_error": 0.0,
                   "max_relative_l2_error": 0.0, "minimum_cosine": 1.0,
                   "repeat_exact": True, "error": None}
            try:
                for index, lw in large:
                    weight = lw.gate_up_weight if operation == "gateup" else lw.down_weight
                    bias = lw.gate_up_bias if operation == "gateup" else lw.down_bias
                    plan = B1MlpGemvPlan(weight, bias, operation, name)
                    for seed in (20260905, 20260906):
                        generator = torch.Generator(device=weight.device).manual_seed(seed + index)
                        hidden = torch.randn((1, 1536), device=weight.device,
                                             dtype=torch.bfloat16, generator=generator)
                        normalized = model._gemma4_flat_rmsnorm(
                            hidden, lw.pre_ff_norm_weight, model._flat_norm_eps, False,
                        )
                        if operation == "down":
                            gate = torch.mm(normalized, lw.gate_up_wt)
                            if lw.gate_up_bias is not None:
                                gate.add_(lw.gate_up_bias)
                            x = torch.nn.functional.gelu(gate[:, :12288], approximate="tanh")
                            x.mul_(gate[:, 12288:])
                        else:
                            x = normalized
                        reference = torch.mm(x, weight.t())
                        if bias is not None:
                            reference.add_(bias)
                        out = torch.full_like(reference, float("nan"))
                        if plan.partial is not None:
                            plan.partial.fill_(float("nan"))
                        actual = plan(x, out).clone()
                        out.fill_(float("nan"))
                        if plan.partial is not None:
                            plan.partial.fill_(float("nan"))
                        repeated = plan(x, out)
                        metrics = _tensor_error(actual, reference)
                        exact = bool(torch.equal(actual, repeated))
                        correct = bool(metrics["finite"] and exact
                                       and metrics["relative_l2_error"] <= 0.01
                                       and metrics["cosine"] >= 0.9999)
                        row["checked_inputs"] += 1
                        row["repeat_exact"] &= exact
                        row["max_abs_error"] = max(row["max_abs_error"], metrics["max_abs_error"])
                        row["max_relative_l2_error"] = max(
                            row["max_relative_l2_error"], metrics["relative_l2_error"])
                        row["minimum_cosine"] = min(row["minimum_cosine"], metrics["cosine"])
                        if not correct:
                            raise RuntimeError(f"layer={index} seed={seed}: {metrics}, repeat_exact={exact}")
            except Exception as exc:
                row["correct"] = False
                row["error"] = f"{type(exc).__name__}: {exc}"
            report["projections"][name] = row
            print("MLP_CORRECTNESS " + name + " " + json.dumps(row), flush=True)
    report["all_correct"] = all(row["correct"] for row in report["projections"].values())
    return report
