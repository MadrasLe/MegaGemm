"""B1 projection experiments under a fixed one-step CUDA Graph executor.

No kernel autotuning/timing here: reuse the exact two prepared configurations
from the eager dispatch gate. Each variant owns its routes and failure state.
"""
from dataclasses import asdict
import json

from benchmarks.run_gemma4_e2b_b1_decode_frontier_gate import (
    _capture_lm_state, _restore_lm_state, _force_fused_rms_lm_head, _validate_lm_head,
)

GATEUP = "wide_n4_k2048_w4"
DOWN = "down_n4_k256_s1_w4"
SELECTED = {"gateup": (GATEUP,), "down": (DOWN,)}
SPECS = {
    "graph_production": ({}, False),
    "graph_gateup": ({"gateup": GATEUP}, False),
    "graph_down": ({"down": DOWN}, False),
    "graph_both": ({"gateup": GATEUP, "down": DOWN}, False),
    "graph_fused_lm_head": ({}, True),
}

ROUTE_ATTRS = (
    "_gemma4_b1_mlp_prepared_routes", "_gemma4_b1_mlp_prepared_weights",
    "_gemma4_b1_mlp_prepared_buffers", "_gemma4_b1_mlp_prepared_device",
)


def expected_routes(case_name, baseline):
    configs, _ = SPECS[case_name]
    expected = dict(baseline)
    for op in ("gateup", "down"):
        expected[f"gemma4_b1_mlp_gemv_{op}_hits"] = 20 if op in configs else 0
        expected[f"gemma4_b1_mlp_prepared_{op}_hits"] = 20 if op in configs else 0
    return expected


class GraphKernelExperiment:
    def __init__(self):
        from benchmarks.run_gemma4_e2b_b1_execution_gate import Case
        self.cases = tuple(Case(name, reuse=True, graph=True) for name in SPECS)
        self.ready = False
        self.bindings = {}
        self.report = {"selected_configs": SELECTED, "cases": [asdict(c) for c in self.cases]}

    def compile_preflight(self):
        from benchmarks.compile_gemma4_b1_mlp_gemv import compile_kernels
        return compile_kernels(selected_configs=SELECTED)

    def prepare(self, model, production_lm_state):
        from benchmarks.gemma4_b1_mlp_gemv_frontier import validate_projections
        self.production_lm_state = dict(production_lm_state)
        self.bridge = bool(model._gemma4_flat_dense_attn_mlp_bridge_enabled)
        if not self.bridge:
            raise RuntimeError("The promoted B1 attention-to-MLP bridge is not active")
        self.report["projections"] = validate_projections(model, selected_configs=SELECTED)
        # Done before ANY graph capture: this helper may allocate LM-head scratch.
        try:
            self.report["lm_head"] = _validate_lm_head(model, production_lm_state)
        finally:
            _restore_lm_state(model, production_lm_state)
        rejected = {}
        for case in self.cases:
            configs, fused = SPECS[case.name]
            errors = []
            for config in configs.values():
                validation = self.report["projections"].get("projections", {}).get(config, {})
                if not validation.get("correct") or validation.get("checked_inputs") != 40:
                    errors.append(f"projection preflight failed: {config}: {validation.get('error')}")
            if fused and not self.report["lm_head"].get("exact_token_match"):
                errors.append("fused LM-head token preflight failed")
            if errors:
                rejected[case.name] = errors
                continue
            # Allocate once per variant. The same weight plans/output buffers may
            # be referenced by multiple graphs, but are only executed sequentially.
            failures = {}
            model._gemma4_b1_mlp_gemv_configs = dict(configs)
            model._gemma4_b1_mlp_gemv_failures = failures
            model._gemma4_b1_mlp_gemv_dispatch = "prepared"
            model._gemma4_b1_mlp_prepared_routes = None
            if configs:
                model._prepare_gemma4_b1_mlp_gemv_routes()
            if fused:
                _force_fused_rms_lm_head(model, production_lm_state)
            else:
                _restore_lm_state(model, production_lm_state)
            self.bindings[case.name] = {
                "configs": dict(configs), "failures": failures,
                "route_attrs": {key: getattr(model, key, None) for key in ROUTE_ATTRS},
                "lm_state": _capture_lm_state(model),
            }
        self.ready = True
        self.activate(model, self.cases[0])
        print("GRAPH_KERNEL_PREFLIGHT " + json.dumps(self.report), flush=True)
        return rejected

    def activate(self, model, case):
        binding = self.bindings[case.name]
        model._gemma4_b1_mlp_gemv_configs = binding["configs"]
        # Preserve failure dictionaries: changing cases must not hide a fallback.
        model._gemma4_b1_mlp_gemv_failures = binding["failures"]
        model._gemma4_b1_mlp_gemv_dispatch = "prepared"
        for key, value in binding["route_attrs"].items():
            setattr(model, key, value)
        _restore_lm_state(model, binding["lm_state"])

    def audit_state(self, model, case):
        binding = self.bindings[case.name]
        errors = []
        if binding["failures"]:
            errors.append(f"GEMV fallback: {binding['failures']}")
        if _capture_lm_state(model) != binding["lm_state"]:
            errors.append("LM-head selection changed or fell back")
        if not model._gemma4_flat_dense_attn_mlp_bridge_enabled or model._gemma4_flat_dense_attn_mlp_bridge_runtime_disabled:
            errors.append("promoted B1 bridge changed or failed")
        if model._gemma4_b1_mlp_prepared_routes is not binding["route_attrs"]["_gemma4_b1_mlp_prepared_routes"]:
            errors.append("prepared route owner changed")
        return errors

    def audit_routes(self, case, observed, baseline):
        if not observed or not baseline:
            return ["missing layer-route capture audit"]
        expected = expected_routes(case.name, baseline[0])
        if observed[0] != expected:
            return [f"unexpected layer routes: expected={expected}, observed={observed[0]}"]
        # These counters reflect Python execution during warmup/capture, NOT
        # CUDA Graph replays. Timed rows independently require real graph replay.
        self.report.setdefault("capture_routes", {})[case.name] = observed[0]
        return []

    def audit_cold(self, case, counters):
        self.report.setdefault("warmup_capture_counters", {})[case.name] = counters
        if SPECS[case.name][1] and counters.get("fused_rmsnorm_lm_head_argmax_hits", 0) <= 0:
            return ["forced fused LM head was not executed during warmup/capture"]
        return []
