#!/usr/bin/env python3
"""Actual-hidden correctness oracle for the Gemma 4 E2B/L4/B8 LM head.

The full-model backend gate found that both Tensor Core routes generate the
same sequence while the current direct fused LM head generates another one.
Sequence comparison alone cannot identify the correct route because the first
different token changes every later hidden state.  This gate therefore:

1. runs one natural-greedy eager decode and captures its real pre-LM-head
   hidden states;
2. evaluates every selector on each identical captured hidden state;
3. treats the engine's materialized logits + Gemma softcap + torch.argmax path
   as the canonical contract.

There is one model load, no timing sweep, no competing engine and no model
mutation.  The output is a correctness proof suitable for deciding whether the
faster Tensor Core backend can replace the current direct fused selector.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import benchmark_inference_matrix as matrix
from benchmarks.run_gemma4_e2b_b1_decode_frontier_gate import (
    _configure_environment,
    _run,
)


def _as_ints(tensor: Any) -> list[int]:
    return [int(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _capture_actual_decode_hidden_states(
    engine: Any,
    prompts: list[str],
    capture_steps: int,
) -> tuple[dict[str, Any], list[Any]]:
    """Capture only real decode-body outputs, after LM-head priming."""
    import torch

    model = engine.model
    model._prime_fused_lm_head_argmax(
        num_rows=8,
        device=model.embed_tokens.weight.device,
        dtype=model.embed_tokens.weight.dtype,
    )
    model_type = type(model)
    original = model_type._decode_next_token_greedy
    captured: list[Any] = []

    def capture(self: Any, hidden: Any) -> Any:
        if self is model and len(captured) < capture_steps:
            captured.append(hidden.detach().clone())
        return original(self, hidden)

    model_type._decode_next_token_greedy = capture
    try:
        row = _run(engine, prompts, capture_steps + 1)
        torch.cuda.synchronize()
    finally:
        model_type._decode_next_token_greedy = original

    if len(captured) != capture_steps:
        raise RuntimeError(
            f"captured {len(captured)} real hidden states, expected {capture_steps}"
        )
    return row, captured


def _evaluate_hidden(model: Any, hidden: Any, step: int) -> dict[str, Any]:
    """Evaluate all selectors on one identical actual decode hidden state."""
    import torch
    from megagemm.kernels.lm_head_argmax import (
        lm_head_argmax,
        lm_head_rmsnorm_argmax,
        logits_softcap_argmax,
    )

    with torch.inference_mode():
        raw_logits = model._decode_raw_logits_from_hidden(hidden)
        capped_logits = model._apply_final_logit_capping(raw_logits)
        final_logits = capped_logits[:, -1, :]
        raw_last = raw_logits[:, -1, :]

        canonical = torch.argmax(final_logits, dim=-1)
        out_tok, partial_vals, partial_idxs = model._get_fused_lm_head_buffers(hidden)
        softcap_fused = logits_softcap_argmax(
            raw_last,
            model.final_logit_softcapping,
            out_tokens=out_tok,
            partial_vals=partial_vals,
            partial_idxs=partial_idxs,
        ).clone()
        direct_rmsnorm_fused = lm_head_rmsnorm_argmax(
            hidden,
            model.norm.weight,
            model.norm.eps,
            model.norm.offset,
            model.lm_head.weight,
            model.lm_head.bias,
            out_tokens=out_tok,
            partial_vals=partial_vals,
            partial_idxs=partial_idxs,
        ).clone()
        hidden_norm = model.norm(hidden)
        direct_normed_fused = lm_head_argmax(
            hidden_norm,
            model.lm_head.weight,
            model.lm_head.bias,
            out_tokens=out_tok,
            partial_vals=partial_vals,
            partial_idxs=partial_idxs,
        ).clone()
        torch.cuda.synchronize()

        oracle_value = torch.gather(
            final_logits, 1, canonical.unsqueeze(1)
        ).squeeze(1)
        direct_value = torch.gather(
            final_logits, 1, direct_rmsnorm_fused.unsqueeze(1)
        ).squeeze(1)
        softcap_value = torch.gather(
            final_logits, 1, softcap_fused.unsqueeze(1)
        ).squeeze(1)
        normed_value = torch.gather(
            final_logits, 1, direct_normed_fused.unsqueeze(1)
        ).squeeze(1)
        raw_top2 = torch.topk(raw_last.float(), k=2, dim=-1)

    canonical_ids = _as_ints(canonical)
    softcap_ids = _as_ints(softcap_fused)
    rms_ids = _as_ints(direct_rmsnorm_fused)
    normed_ids = _as_ints(direct_normed_fused)
    oracle_values = [float(value) for value in oracle_value.float().cpu().tolist()]
    softcap_values = [float(value) for value in softcap_value.float().cpu().tolist()]
    rms_values = [float(value) for value in direct_value.float().cpu().tolist()]
    normed_values = [float(value) for value in normed_value.float().cpu().tolist()]

    rows: list[dict[str, Any]] = []
    for row in range(len(canonical_ids)):
        rows.append(
            {
                "row": row,
                "canonical_token": canonical_ids[row],
                "tensorcore_softcap_token": softcap_ids[row],
                "direct_rmsnorm_fused_token": rms_ids[row],
                "direct_normed_fused_token": normed_ids[row],
                "softcap_matches_canonical": softcap_ids[row] == canonical_ids[row],
                "direct_rmsnorm_matches_canonical": rms_ids[row] == canonical_ids[row],
                "direct_normed_matches_canonical": normed_ids[row] == canonical_ids[row],
                "canonical_capped_value": oracle_values[row],
                "tensorcore_softcap_capped_value": softcap_values[row],
                "direct_rmsnorm_capped_value": rms_values[row],
                "direct_normed_capped_value": normed_values[row],
                "canonical_minus_tensorcore_softcap": (
                    oracle_values[row] - softcap_values[row]
                ),
                "canonical_minus_direct_rmsnorm": oracle_values[row] - rms_values[row],
                "canonical_minus_direct_normed": oracle_values[row] - normed_values[row],
                "raw_top1_minus_top2": float(
                    raw_top2.values[row, 0].item() - raw_top2.values[row, 1].item()
                ),
            }
        )

    return {
        "step": step,
        "shape": list(hidden.shape),
        "finite": bool(torch.isfinite(hidden).all().item()),
        "canonical_tokens": canonical_ids,
        "tensorcore_softcap_tokens": softcap_ids,
        "direct_rmsnorm_fused_tokens": rms_ids,
        "direct_normed_fused_tokens": normed_ids,
        "rows": rows,
    }


def _agreement(steps: list[dict[str, Any]], key: str) -> dict[str, Any]:
    total = 0
    matches = 0
    first_mismatch = None
    positive_oracle_margin = 0
    tied_capped_value = 0
    for step in steps:
        for row in step["rows"]:
            total += 1
            if row[key]:
                matches += 1
                continue
            margin_key = {
                "softcap_matches_canonical": "canonical_minus_tensorcore_softcap",
                "direct_rmsnorm_matches_canonical": "canonical_minus_direct_rmsnorm",
                "direct_normed_matches_canonical": "canonical_minus_direct_normed",
            }[key]
            margin = float(row.get(margin_key, math.nan))
            if math.isfinite(margin) and margin > 0.0:
                positive_oracle_margin += 1
            elif margin == 0.0:
                tied_capped_value += 1
            if first_mismatch is None:
                first_mismatch = {
                    "step": step["step"],
                    **row,
                }
    return {
        "matches": matches,
        "total": total,
        "agreement": matches / total if total else 0.0,
        "mismatches": total - matches,
        "positive_canonical_margin_mismatches": positive_oracle_margin,
        "equal_capped_value_mismatches": tied_capped_value,
        "first_mismatch": first_mismatch,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--capture-steps", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2304)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir")
    args = parser.parse_args(argv)
    if args.capture_steps < 2:
        parser.error("capture-steps must be at least 2")

    _configure_environment(args.model, -1)
    # This diagnostic must execute Python once per real decode step so the
    # pre-head hidden tensors can be copied.  It intentionally does not time.
    os.environ.update(
        {
            "MEGAGEMM_DECODE_CUDA_GRAPHS": "0",
            "MEGAGEMM_DECODE_CUDA_GRAPHS_PREFER_STEP": "0",
            "MEGAGEMM_REUSE_REQUEST_SCHEDULER": "0",
            "MEGAGEMM_GEMMA4_E2B_L4_B8_BATCH_CUBLAS_LM_HEAD": "0",
            "MEGAGEMM_GEMMA4_E2B_L4_B8_FUSED_SOFTCAP_ARGMAX": "0",
        }
    )

    import torch
    if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name().upper():
        raise SystemExit("This gate requires an NVIDIA L4; no model was loaded.")
    import megagemm.models.llama as llama_module
    from megagemm.engine import InferenceEngine

    llama_module._GEMMA4_E2B_L4_B8_BATCH_CUBLAS_LM_HEAD = False
    llama_module._GEMMA4_E2B_L4_B8_FUSED_SOFTCAP_ARGMAX = False
    llama_module._GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX = False
    engine = InferenceEngine(
        args.model,
        dtype=torch.bfloat16,
        device="cuda",
        max_batch_size=8,
        max_seq_len=args.max_seq_len,
        num_blocks=0,
        block_size=16,
        kv_alloc="auto",
        cache_dir=args.cache_dir,
    )
    model = engine.model
    if getattr(model.runtime_policy, "name", "") != "gemma4-e2b-l4":
        raise RuntimeError("loaded model did not resolve the E2B/L4 RuntimePolicy")
    prompts = matrix.build_prompts(engine.tokenizer, 8, args.prompt_tokens)[0]

    print("Gemma 4 E2B/L4/B8 actual-hidden LM-head correctness gate", flush=True)
    print(
        f"  workload: P{args.prompt_tokens}, {args.capture_steps} real decode hidden states",
        flush=True,
    )
    print("  oracle: materialized logits + BF16 softcap + torch.argmax", flush=True)
    print("  timing: disabled; model loads: 1; competing engine: disabled", flush=True)

    generation, hidden_states = _capture_actual_decode_hidden_states(
        engine, prompts, args.capture_steps
    )
    steps = [
        _evaluate_hidden(model, hidden, step)
        for step, hidden in enumerate(hidden_states)
    ]
    softcap = _agreement(steps, "softcap_matches_canonical")
    direct_rms = _agreement(steps, "direct_rmsnorm_matches_canonical")
    direct_normed = _agreement(steps, "direct_normed_matches_canonical")

    if softcap["mismatches"] == 0 and direct_rms["mismatches"] > 0:
        verdict = "PROMOTE_TENSORCORE_FUSED_SOFTCAP_CORRECTNESS"
    elif softcap["mismatches"] == 0 and direct_rms["mismatches"] == 0:
        verdict = "BOTH_ROUTES_MATCH_CANONICAL"
    elif softcap["mismatches"] > 0:
        verdict = "REJECT_FUSED_SOFTCAP_REDUCTION"
    else:
        verdict = "INCONCLUSIVE"

    decision = {
        "verdict": verdict,
        "canonical_contract": "final_norm -> BF16 torch.mm -> BF16 softcap -> torch.argmax",
        "tensorcore_fused_softcap": softcap,
        "current_direct_rmsnorm_fused": direct_rms,
        "direct_normed_fused": direct_normed,
        "safe_to_replace_current_fused": bool(
            softcap["mismatches"] == 0 and direct_rms["mismatches"] > 0
        ),
    }
    payload = {
        "status": "passed" if softcap["mismatches"] == 0 else "failed",
        "method": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "model": args.model,
            "batch_size": 8,
            "prompt_tokens": args.prompt_tokens,
            "captured_decode_steps": args.capture_steps,
            "captured_rows": args.capture_steps * 8,
            "model_loads": 1,
            "natural_greedy_capture": True,
            "cuda_graphs_during_capture": False,
        },
        "generation": generation,
        "steps": steps,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("DECISION " + json.dumps(decision, sort_keys=True), flush=True)
    for step in steps:
        mismatches = [
            row
            for row in step["rows"]
            if not row["direct_rmsnorm_matches_canonical"]
        ]
        if mismatches:
            print(
                f"STEP {step['step']} direct-fused mismatches: "
                + json.dumps(mismatches, sort_keys=True),
                flush=True,
            )
    print(f"Wrote: {args.output}", flush=True)
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
