"""
WW4 Survival Controller: safety invariants > performance.
Orchestrates state machine, consensus, risk tokens; enforces monotonicity; produces survival report.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .config import AllocationConfig
from .inputs import StrategyInputs
from .allocator import compute_allocation, AllocationResult
from .hazard import HazardContext, compute_hazard_level
from .ww4_state import (
    SurvivalState,
    WW4StateMachine,
    compute_ww4_state,
    MAX_GROSS_BY_STATE,
    MAX_SINGLE_WEIGHT,
)
from .ww4_consensus import consensus_regime, check_sensor_poisoning, ConsensusResult
from .ww4_tokens import compute_budget, RiskTokenBudget


@dataclass
class WW4Context:
    """Input context for one WW4 allocation step."""
    state_machine: WW4StateMachine
    consensus_regime_result: Optional[ConsensusResult] = None
    consensus_disagree: bool = False
    sensor_poisoning: bool = False
    blackout: bool = False
    risk_budget: Optional[RiskTokenBudget] = None
    module_health: Dict[str, str] = field(default_factory=dict)  # "ok" | "fail"
    exposure_before: float = 0.0
    concentration_before: float = 0.0


@dataclass
class WW4AllocationResult:
    """Allocation result plus WW4 audit fields for survival report."""
    result: AllocationResult
    ww4_state: str
    reason_codes: List[str]
    exposure_before: float
    exposure_after: float
    concentration_before: float
    concentration_after: float
    turnover_throttle_applied: bool
    modules_healthy: Dict[str, str]
    consensus_status: str  # "agree" | "disagree"
    why_not_taking_risk: str = ""


def _hazard_from_ww4_state(sm: WW4StateMachine) -> HazardContext:
    """Map WW4 state to HazardContext for existing allocator."""
    level = min(5, int(sm.state) + 1)  # NORMAL=1, CAUTION=2, DANGER=3, SURVIVAL=4, LOCKDOWN=5
    return HazardContext(
        hazard_level=level,
        reason_codes=sm.reason_codes,
        telemetry_blackout=sm.blackout,
        subsystem_failure=sm.infra_failure,
        correlation_meltdown=sm.correlation_crisis,
        liquidity_shock=sm.liquidity_crisis,
        integrity_failure=sm.integrity_failure or sm.sensor_poisoning,
        stale_input_count=len(sm.reason_codes),
        telemetry_integrity="fail" if (sm.telemetry_fail or sm.blackout) else "ok",
        subsystem_failures_count=1 if sm.infra_failure else 0,
        correlation_assumption="worst_case" if sm.correlation_crisis else "normal",
    )


def _enforce_invariants(
    result: AllocationResult,
    sm: WW4StateMachine,
    risk_budget: Optional[RiskTokenBudget],
) -> AllocationResult:
    """Enforce exposure monotonicity, concentration monotonicity, risk token cap."""
    weights = dict(result.weights)
    amounts = dict(result.amounts)
    gross = sum(weights.values())
    max_single = max(weights.values()) if weights else 0.0
    max_gross = MAX_GROSS_BY_STATE.get(sm.state, 0.0)
    max_single_allowed = MAX_SINGLE_WEIGHT.get(sm.state, 0.4)

    # Cap by state
    if gross > max_gross + 1e-9 and gross > 0:
        scale = max_gross / gross
        for s in weights:
            weights[s] *= scale
        for s in amounts:
            amounts[s] *= scale
        gross = sum(weights.values())

    # Risk token cap
    if risk_budget is not None and gross > risk_budget.max_gross_from_tokens() + 1e-9 and gross > 0:
        scale = risk_budget.max_gross_from_tokens() / gross
        for s in weights:
            weights[s] *= scale
        for s in amounts:
            amounts[s] *= scale
        gross = sum(weights.values())

    # Per-strategy concentration cap by state (no renormalize: sum may be < 1)
    for s in list(weights.keys()):
        if weights[s] > max_single_allowed + 1e-9:
            weights[s] = max_single_allowed
    total_w = sum(weights.values())
    total_a = sum(amounts.values())
    if total_w > 0 and total_a > 0:
        for s in amounts:
            amounts[s] = weights.get(s, 0) / total_w * total_a

    return AllocationResult(
        weights=weights,
        amounts=amounts,
        throttle=result.throttle,
        shutdown=result.shutdown,
        reasons=result.reasons,
        throttle_records=result.throttle_records,
        shutdown_records=result.shutdown_records,
        hazard_level=result.hazard_level,
        reason_codes=result.reason_codes,
        gross_exposure=sum(weights.values()),
        number_halted=result.number_halted,
        correlation_assumption=result.correlation_assumption,
        telemetry_integrity=result.telemetry_integrity,
        stale_input_count=result.stale_input_count,
        subsystem_failures_count=result.subsystem_failures_count,
    )


def compute_allocation_ww4(
    inputs: List[StrategyInputs],
    config: AllocationConfig,
    ww4_context: Optional[WW4Context] = None,
    portfolio_drawdown_pct: Optional[float] = None,
    exposure_before: float = 0.0,
    concentration_before: float = 0.0,
) -> WW4AllocationResult:
    """
    WW4 survival allocation. Uses state machine, consensus, risk tokens;
    enforces invariants; returns result + audit for survival report.
    """
    # Default: no WW4 context => run as NORMAL with existing allocator
    if ww4_context is None:
        res = compute_allocation(inputs, config, portfolio_drawdown_pct=portfolio_drawdown_pct)
        gross = sum(res.weights.values())
        conc = max(res.weights.values()) if res.weights else 0.0
        return WW4AllocationResult(
            result=res,
            ww4_state="NORMAL",
            reason_codes=[],
            exposure_before=exposure_before,
            exposure_after=gross,
            concentration_before=concentration_before,
            concentration_after=conc,
            turnover_throttle_applied=any(res.throttle.values()),
            modules_healthy={},
            consensus_status="agree",
        )

    sm = ww4_context.state_machine
    hazard = _hazard_from_ww4_state(sm)
    res = compute_allocation(
        inputs, config,
        portfolio_drawdown_pct=portfolio_drawdown_pct,
        hazard_context=hazard,
    )

    # Risk budget (2-of-3 in SURVIVAL)
    regime_ok = not ww4_context.consensus_disagree and (ww4_context.consensus_regime_result is None or ww4_context.consensus_regime_result.agree)
    feasibility_ok = ww4_context.module_health.get("feasibility", "ok") == "ok"
    stress_ok = ww4_context.module_health.get("stress", "ok") == "ok"
    risk_budget = compute_budget(sm.state, regime_ok=regime_ok, feasibility_ok=feasibility_ok, stress_ok=stress_ok)
    if ww4_context.risk_budget is not None:
        risk_budget = ww4_context.risk_budget

    res = _enforce_invariants(res, sm, risk_budget)

    gross = sum(res.weights.values())
    conc = max(res.weights.values()) if res.weights else 0.0
    why_not = ""
    if sm.state >= SurvivalState.DANGER:
        why_not = "hazard elevated; reducing risk (WW4 survival mode)."
    if ww4_context.sensor_poisoning or ww4_context.blackout:
        why_not = "sensor poisoning or blackout; zero trust."
    if risk_budget.tokens <= 0 and sm.state >= SurvivalState.SURVIVAL:
        why_not = "risk tokens exhausted or 2-of-3 not met in SURVIVAL."

    return WW4AllocationResult(
        result=res,
        ww4_state=sm.state.name,
        reason_codes=sm.reason_codes,
        exposure_before=exposure_before or ww4_context.exposure_before,
        exposure_after=gross,
        concentration_before=concentration_before or ww4_context.concentration_before,
        concentration_after=conc,
        turnover_throttle_applied=any(res.throttle.values()),
        modules_healthy=ww4_context.module_health,
        consensus_status="disagree" if ww4_context.consensus_disagree else "agree",
        why_not_taking_risk=why_not,
    )


def build_ww4_context(
    telemetry_fail: bool = False,
    correlation_crisis: bool = False,
    liquidity_crisis: bool = False,
    model_drift: bool = False,
    infra_failure: bool = False,
    consensus_disagree: bool = False,
    sensor_poisoning: bool = False,
    blackout: bool = False,
    near_miss_count: int = 0,
    previous_state: Optional[WW4StateMachine] = None,
    regime_estimates: Optional[List[float]] = None,
    exposure_before: float = 0.0,
    concentration_before: float = 0.0,
) -> WW4Context:
    """Build WW4Context from triggers and optional consensus inputs."""
    sm = compute_ww4_state(
        telemetry_fail=telemetry_fail,
        correlation_crisis=correlation_crisis,
        liquidity_crisis=liquidity_crisis,
        model_drift=model_drift,
        infra_failure=infra_failure,
        consensus_disagree=consensus_disagree,
        sensor_poisoning=sensor_poisoning,
        blackout=blackout,
        near_miss_count=near_miss_count,
        previous=previous_state,
    )
    reg_consensus = None
    if regime_estimates and len(regime_estimates) >= 2:
        reg_consensus = consensus_regime(regime_estimates)
        if not reg_consensus.agree:
            consensus_disagree = True
    return WW4Context(
        state_machine=sm,
        consensus_regime_result=reg_consensus,
        consensus_disagree=consensus_disagree,
        sensor_poisoning=sensor_poisoning,
        blackout=blackout,
        module_health={},
        exposure_before=exposure_before,
        concentration_before=concentration_before,
    )
