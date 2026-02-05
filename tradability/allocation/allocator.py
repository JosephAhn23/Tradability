"""
Core allocation logic: conservative base weights, uncertainty, correlation control.

NO return optimization. Weight ∝ feasible_capacity × regime_confidence ÷ uncertainty_penalty.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import math

from .config import AllocationConfig
from .inputs import StrategyInputs
from .uncertainty import uncertainty_penalty_factor, feasible_capacity_share
from .policies import apply_shutdown_rules, apply_throttles, ShutdownRecord, ThrottleRecord
from .sanitize import sanitize_inputs
from .hazard import HazardContext, EMERGENCY_GROSS, compute_hazard_level


def _apply_final_weight_caps(weights: Dict[str, float], max_w: float) -> Dict[str, float]:
    """Enforce per-strategy cap on final weights. No renormalization (may sum < 1 => cash)."""
    return {s: min(w, max_w) for s, w in weights.items()}


@dataclass
class AllocationResult:
    """Final allocation: weights, amounts, throttle/shutdown, reasons. Optional WW3 audit fields."""

    weights: Dict[str, float]
    amounts: Dict[str, float]
    throttle: Dict[str, bool]
    shutdown: Dict[str, bool]
    reasons: Dict[str, str]
    throttle_records: List[ThrottleRecord]
    shutdown_records: List[ShutdownRecord]
    # WW3 audit (optional)
    hazard_level: Optional[int] = None
    reason_codes: Optional[List[str]] = None
    gross_exposure: Optional[float] = None
    number_halted: Optional[int] = None
    correlation_assumption: Optional[str] = None
    telemetry_integrity: Optional[str] = None
    stale_input_count: Optional[int] = None
    subsystem_failures_count: Optional[int] = None


def compute_allocation(
    inputs: List[StrategyInputs],
    config: AllocationConfig,
    portfolio_drawdown_pct: Optional[float] = None,
    hazard_context: Optional[HazardContext] = None,
) -> AllocationResult:
    """
    Policy-based allocation. Base weight ∝ feasible_capacity × regime_confidence / uncertainty_penalty.
    Apply correlation_group cap, throttles, shutdown; normalize.
    Fail-closed: inputs sanitized; no NaN/Inf in output.
    If hazard_context and hazard_level >= 4: gross exposure forced to EMERGENCY_GROSS (near 0).
    """
    inputs = sanitize_inputs(inputs, config)
    throttle_records = []
    shutdown_records = []
    raw_weights = {}

    for inp in inputs:
        # Shutdown first
        shut = apply_shutdown_rules(inp, config)
        if shut is not None:
            shutdown_records.append(shut)
            raw_weights[inp.strategy_id] = 0.0
            continue

        # Base allocation: feasible_capacity × regime_confidence ÷ uncertainty_penalty
        cap_share = feasible_capacity_share(inp, config.total_capital)
        penalty = uncertainty_penalty_factor(inp, config)
        raw = cap_share * inp.regime_confidence / penalty
        # Only positive net edge gets capital
        if inp.net_edge_bps <= 0:
            raw = 0.0
        else:
            # Scale by edge strength (conservative: cap at 2x base)
            raw *= min(1.0 + inp.net_edge_bps / 100.0, 2.0)

        throttle_mag, thr_recs = apply_throttles(inp, config)
        raw *= throttle_mag
        throttle_records.extend(thr_recs)

        raw = min(raw, config.max_weight_per_strategy)
        if raw < config.min_weight_threshold:
            raw = 0.0
            shutdown_records.append(ShutdownRecord(inp.strategy_id, "below_min_weight_threshold"))
        raw_weights[inp.strategy_id] = max(0.0, raw)

    # Correlation control: cap combined exposure per correlation_group
    groups: Dict[Optional[str], List[str]] = {}
    for inp in inputs:
        g = inp.correlation_group
        if g not in groups:
            groups[g] = []
        groups[g].append(inp.strategy_id)
    for g, members in groups.items():
        if g is None or len(members) <= 1:
            continue
        total_group = sum(raw_weights.get(m, 0) for m in members)
        if total_group > config.max_weight_per_strategy * config.correlation_penalty:
            scale = (config.max_weight_per_strategy * config.correlation_penalty) / total_group
            for m in members:
                raw_weights[m] = raw_weights.get(m, 0) * scale

    # Portfolio drawdown: reduce all
    if portfolio_drawdown_pct is not None and portfolio_drawdown_pct >= 0.10:
        for s in raw_weights:
            raw_weights[s] *= 0.5

    # Normalize
    total = sum(raw_weights.values())
    if total <= 0:
        weights = {s: 0.0 for s in raw_weights}
        amounts = {s: 0.0 for s in raw_weights}
    else:
        weights = {s: raw_weights[s] / total for s in raw_weights}
        # Enforce per-strategy cap on final weights (WW3: no strategy can dominate)
        weights = _apply_final_weight_caps(weights, config.max_weight_per_strategy)
        amounts = {}
        for s, w in weights.items():
            cap = config.total_capital * w
            inp = next((i for i in inputs if i.strategy_id == s), None)
            if inp and inp.capacity_max_aum is not None:
                cap = min(cap, inp.capacity_max_aum)
            amounts[s] = cap
        total_alloc = sum(amounts.values())
        # Only scale down (over-allocation), never scale up when caps forced sum(weights) < 1
        if total_alloc > 0 and total_alloc > config.total_capital and abs(total_alloc - config.total_capital) > 1:
            scale = config.total_capital / total_alloc
            for s in amounts:
                amounts[s] *= scale
            for s in weights:
                weights[s] = amounts[s] / config.total_capital

    throttle = {inp.strategy_id: any(r.strategy_id == inp.strategy_id for r in throttle_records) for inp in inputs}
    shutdown = {inp.strategy_id: any(r.strategy_id == inp.strategy_id for r in shutdown_records) for inp in inputs}
    reasons = {}
    for r in shutdown_records:
        reasons[r.strategy_id] = "shutdown: " + r.reason
    for inp in inputs:
        if inp.strategy_id not in reasons:
            thr = [r for r in throttle_records if r.strategy_id == inp.strategy_id]
            reasons[inp.strategy_id] = "; ".join(r.reason for r in thr) if thr else "ok"

    # Global hazard: cut to EMERGENCY_GROSS when hazard_level >= 4
    gross = sum(weights.values())
    if hazard_context is not None and hazard_context.is_emergency:
        max_gross = hazard_context.max_gross_allowed
        if gross > max_gross and gross > 0:
            scale = max_gross / gross
            for s in weights:
                weights[s] *= scale
            for s in amounts:
                amounts[s] *= scale
        gross = sum(weights.values())
        for sid in list(reasons.keys()):
            if "GLOBAL CATASTROPHE" not in (reasons.get(sid) or ""):
                reasons[sid] = "GLOBAL CATASTROPHE MODE; " + (reasons.get(sid) or "")

    # Fail-closed: no NaN/Inf in output
    for s in list(weights.keys()):
        w = weights[s]
        a = amounts[s]
        if math.isnan(w) or math.isinf(w) or w < 0:
            weights[s] = 0.0
            amounts[s] = 0.0
        if math.isnan(a) or math.isinf(a) or a < 0:
            amounts[s] = 0.0
    gross = sum(weights.values())
    n_halted = sum(1 for v in shutdown.values() if v)

    return AllocationResult(
        weights=weights,
        amounts=amounts,
        throttle=throttle,
        shutdown=shutdown,
        reasons=reasons,
        throttle_records=throttle_records,
        shutdown_records=shutdown_records,
        hazard_level=hazard_context.hazard_level if hazard_context else None,
        reason_codes=hazard_context.reason_codes if hazard_context else None,
        gross_exposure=gross,
        number_halted=n_halted,
        correlation_assumption=hazard_context.correlation_assumption if hazard_context else None,
        telemetry_integrity=hazard_context.telemetry_integrity if hazard_context else None,
        stale_input_count=hazard_context.stale_input_count if hazard_context else None,
        subsystem_failures_count=hazard_context.subsystem_failures_count if hazard_context else None,
    )
