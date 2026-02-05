"""
WW5 Cosmic Survival: existence across time. No ground truth, no second chances.
Optionality-preserving, confidence decay, irreversibility blocking, dormant mode.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .config import AllocationConfig
from .inputs import StrategyInputs
from .allocator import compute_allocation, AllocationResult
from .hazard import HazardContext
from .ww4 import compute_allocation_ww4, build_ww4_context
from .ww5_state import CosmicSurvivalState, WW5State, compute_ww5_state, DORMANT_GROSS
from .ww5_irreversibility import (
    classify_allocation_action,
    block_irreversible_under_uncertainty,
    Reversibility,
)
from .ww5_confidence_decay import (
    decayed_confidence,
    optionality_score,
    DEFAULT_DECAY_RATE_PER_PERIOD,
    ConfidenceDecayState,
)
from .ww5_bluff import (
    BluffAudit,
    compute_bluff_risk_score,
    build_bluff_audit,
    confidence_from_inputs,
    check_confidence_monotonicity,
    check_silence_over_certainty,
    check_assumptions_disclosed,
)
from .ww5_evidence import EvidenceLedger, EvidenceEvent, EvidenceType, max_gross_from_hazard_and_bluff
from .ww5_unknown import compute_unknown_conditions, UnknownConditions


@dataclass
class WW5Context:
    """Input context for one WW5 allocation step."""
    ww5_state: WW5State
    periods_without_confirmation: int = 0
    confidence_decay_rate: float = DEFAULT_DECAY_RATE_PER_PERIOD
    apply_confidence_decay: bool = True
    block_irreversible: bool = True
    uncertainty_threshold_for_irreversible: float = 0.5
    # WW5-B bluff detection
    narrative_only: bool = False
    authority_endorsement_only: bool = False
    unknown_scenario: bool = False
    evidence_added: List[str] = field(default_factory=list)
    # Upgrade: evidence earned via ledger; UNKNOWN from conditions
    evidence_ledger: Optional[Any] = None
    current_tick: int = 0
    last_tick: int = -1
    unknown_conditions: Optional[UnknownConditions] = None


@dataclass
class WW5AllocationResult:
    """Existential audit: survival_state + irreversible_actions_blocked + optionality + reasons + bluff_audit."""
    result: AllocationResult
    survival_state: str  # NORMAL | CONSERVATIVE | SURVIVAL | DORMANT
    irreversible_actions_blocked: int
    assumptions_required: List[str]
    assumptions_rejected: List[str]
    optionality_score: float
    confidence_decay_rate: float
    reason_for_not_acting: str = ""
    bluff_audit: Optional[BluffAudit] = None


def _ww4_context_from_ww5(ww5_state: WW5State) -> Optional[Any]:
    """Map WW5 state to WW4-style hazard for allocator."""
    if ww5_state.survival_state == CosmicSurvivalState.DORMANT:
        return build_ww4_context(blackout=True)
    if ww5_state.survival_state == CosmicSurvivalState.SURVIVAL:
        return build_ww4_context(telemetry_fail=True)
    if ww5_state.survival_state == CosmicSurvivalState.CONSERVATIVE:
        return build_ww4_context(model_drift=True)
    return None


def _apply_confidence_decay_to_inputs(
    inputs: List[StrategyInputs],
    periods: int,
    decay_rate: float,
) -> List[StrategyInputs]:
    """Return new inputs with regime_confidence decayed (no confirmation)."""
    out = []
    for inp in inputs:
        dec = decayed_confidence(inp.regime_confidence, periods, decay_rate)
        out.append(StrategyInputs(
            strategy_id=inp.strategy_id,
            net_edge_bps=inp.net_edge_bps,
            capacity_max_aum=inp.capacity_max_aum,
            regime_fragile=inp.regime_fragile,
            regime_sensitivity_ratio=inp.regime_sensitivity_ratio,
            regime_confidence=dec,
            uncertainty_score=inp.uncertainty_score,
            recent_realized_return_bps=inp.recent_realized_return_bps,
            recent_expected_return_bps=inp.recent_expected_return_bps,
            current_drawdown_pct=inp.current_drawdown_pct,
            turnover=inp.turnover,
            correlation_group=inp.correlation_group,
            zero_alpha_turnover=inp.zero_alpha_turnover,
        ))
    return out


def compute_allocation_ww5(
    inputs: List[StrategyInputs],
    config: AllocationConfig,
    ww5_context: Optional[WW5Context] = None,
    exposure_before: float = 0.0,
    weights_before: Optional[Dict[str, float]] = None,
) -> WW5AllocationResult:
    """
    WW5 cosmic allocation. Apply confidence decay, block irreversible/one-way doors,
    enforce dormant/survival caps. Return existential audit.
    """
    if ww5_context is None:
        res = compute_allocation(inputs, config)
        gross = sum(res.weights.values())
        opt = optionality_score(res.weights, gross)
        conf_before = confidence_from_inputs([inp.regime_confidence for inp in inputs])
        bluff = build_bluff_audit(
            confidence_before=conf_before,
            confidence_after=conf_before,
            evidence_added=[],
            assumptions_required=[],
            assumptions_verified=[],
            assumptions_unverified=[],
            reason_for_action="nominal allocation" if gross > 0 else "",
            reason_for_inaction="none" if gross > 0 else "zero exposure",
            uncertainty_score=sum(inp.uncertainty_score for inp in inputs) / max(1, len(inputs)),
        )
        return WW5AllocationResult(
            result=res,
            survival_state="NORMAL",
            irreversible_actions_blocked=0,
            assumptions_required=[],
            assumptions_rejected=[],
            optionality_score=opt,
            confidence_decay_rate=0.0,
            bluff_audit=bluff,
        )

    state = ww5_context.ww5_state
    periods = ww5_context.periods_without_confirmation or state.no_feedback_periods
    decay_rate = ww5_context.confidence_decay_rate

    # Auto-UNKNOWN from conditions (not just manual flag)
    unknown_conditions = getattr(ww5_context, "unknown_conditions", None)
    unknown_declared = getattr(ww5_context, "unknown_scenario", False)
    if unknown_conditions is not None:
        should_unknown, unknown_reasons = compute_unknown_conditions(
            estimator_disagreement=unknown_conditions.estimator_disagreement,
            telemetry_stale_periods=unknown_conditions.telemetry_stale_periods,
            drift_flags_persist_count=unknown_conditions.drift_flags_persist_count,
            feasibility_ratio=unknown_conditions.feasibility_ratio,
            uncertainty_score=unknown_conditions.uncertainty_score,
            subsystem_failure=unknown_conditions.subsystem_failure,
            telemetry_integrity_fail=unknown_conditions.telemetry_integrity_fail,
        )
        if should_unknown:
            unknown_declared = True
            state = WW5State(
                survival_state=CosmicSurvivalState.DORMANT,
                reason_codes=unknown_reasons + state.reason_codes,
                no_feedback_periods=state.no_feedback_periods,
                hidden_regime_shift_possible=state.hidden_regime_shift_possible,
                radiation_noise=state.radiation_noise,
                deceptive_stability=state.deceptive_stability,
                single_point_of_truth=state.single_point_of_truth,
                uncertainty_extreme=state.uncertainty_extreme,
            )

    # Confidence decay without confirmation
    if ww5_context.apply_confidence_decay and periods > 0:
        inputs = _apply_confidence_decay_to_inputs(inputs, periods, decay_rate)

    # DORMANT: zero exposure, no allocation (includes unknown_scenario or auto-unknown from conditions)
    if state.survival_state == CosmicSurvivalState.DORMANT:
        weights = {inp.strategy_id: 0.0 for inp in inputs}
        amounts = {inp.strategy_id: 0.0 for inp in inputs}
        res = AllocationResult(
            weights=weights,
            amounts=amounts,
            throttle={inp.strategy_id: False for inp in inputs},
            shutdown={inp.strategy_id: True for inp in inputs},
            reasons={inp.strategy_id: "WW5 DORMANT" for inp in inputs},
            throttle_records=[],
            shutdown_records=[],
            hazard_level=5,
            reason_codes=state.reason_codes,
            gross_exposure=0.0,
            number_halted=len(inputs),
        )
        conf_before = confidence_from_inputs([inp.regime_confidence for inp in inputs])
        bluff = build_bluff_audit(
            confidence_before=conf_before,
            confidence_after=0.0,
            evidence_added=getattr(ww5_context, "evidence_added", []) or [],
            assumptions_required=[],
            assumptions_verified=[],
            assumptions_unverified=["allocation_under_unknown_environment"],
            reason_for_action="",
            reason_for_inaction="UNKNOWN: correct answer unknowable; no action.",
            uncertainty_score=1.0,
            unknown_declared=unknown_declared,
        )
        return WW5AllocationResult(
            result=res,
            survival_state="DORMANT",
            irreversible_actions_blocked=0,
            assumptions_required=[],
            assumptions_rejected=["allocation_under_unknown_environment"],
            optionality_score=1.0,
            confidence_decay_rate=decay_rate,
            reason_for_not_acting="DORMANT: no ground truth; preserving optionality." + (" UNKNOWN declared." if unknown_declared else ""),
            bluff_audit=bluff,
        )

    # Run allocator with WW4-style hazard from WW5 state
    ww4_ctx = _ww4_context_from_ww5(state)
    if ww4_ctx is not None:
        base = compute_allocation_ww4(inputs, config, ww4_context=ww4_ctx, exposure_before=exposure_before)
        res = base.result
    else:
        res = compute_allocation(inputs, config)

    # Irreversibility: block one-way doors and irreversible under high uncertainty
    weights_before = weights_before or {}
    blocked = []
    assumptions_rejected = []
    for inp in inputs:
        delta = res.weights.get(inp.strategy_id, 0) - weights_before.get(inp.strategy_id, 0)
        liq = (inp.capacity_max_aum or 0) / (config.total_capital or 1)
        cl = classify_allocation_action(inp, delta, inp.uncertainty_score, liq)
        if ww5_context.block_irreversible:
            if cl.reversibility == Reversibility.IRREVERSIBLE and inp.uncertainty_score >= ww5_context.uncertainty_threshold_for_irreversible:
                blocked.append(inp.strategy_id)
                assumptions_rejected.append(f"irreversible_{inp.strategy_id}")
            if cl.is_one_way_door:
                blocked.append(inp.strategy_id)
                assumptions_rejected.append(f"one_way_door_{inp.strategy_id}")

    for sid in blocked:
        if res.weights.get(sid, 0) > 0:
            res.weights[sid] = 0.0
            res.amounts[sid] = 0.0
            if sid not in res.reasons:
                res.reasons[sid] = "WW5 irreversible/one_way_door blocked"

    # Enforce WW5 max gross by state and monotone cap in hazard/bluff
    gross = sum(res.weights.values())
    max_gross = state.max_gross_allowed()
    avg_unc = sum(inp.uncertainty_score for inp in inputs) / max(1, len(inputs))
    bluff_risk_pre = compute_bluff_risk_score(avg_unc, periods, 0, has_narrative_only=False, single_assumption_dependency=len(blocked) > 0)
    hazard_level = int(state.survival_state)
    monotone_cap = max_gross_from_hazard_and_bluff(hazard_level, bluff_risk_pre)
    max_gross = min(max_gross, monotone_cap)
    if gross > max_gross + 1e-9 and gross > 0:
        scale = max_gross / gross
        for s in res.weights:
            res.weights[s] *= scale
        for s in res.amounts:
            res.amounts[s] *= scale
        gross = sum(res.weights.values())
    res.gross_exposure = gross

    opt = optionality_score(res.weights, gross)
    assumptions_required = []
    if state.no_feedback_periods > 0:
        assumptions_required.append("no_feedback_decay_applied")
    if state.hidden_regime_shift_possible:
        assumptions_required.append("hidden_regime_shift_penalty")
    if state.deceptive_stability:
        assumptions_required.append("deceptive_stability")

    reason_not = ""
    if state.survival_state == CosmicSurvivalState.SURVIVAL:
        reason_not = "SURVIVAL: reducing exposure; no single assumption justifies risk."
    elif state.survival_state == CosmicSurvivalState.CONSERVATIVE:
        reason_not = "CONSERVATIVE: confidence decay or hidden regime shift; preserving optionality."
    if blocked:
        reason_not += f" Blocked irreversible/one_way: {blocked}."

    # WW5-B bluff audit: evidence from ledger when present (earned), else no claim
    conf_before = confidence_from_inputs([inp.regime_confidence for inp in inputs])
    conf_after = confidence_from_inputs([inp.regime_confidence for inp in inputs])
    ledger = getattr(ww5_context, "evidence_ledger", None)
    last_tick = getattr(ww5_context, "last_tick", -1)
    current_tick = getattr(ww5_context, "current_tick", 0)
    if ledger is not None:
        evidence_added = [f"{e.source}:{e.event_type.value}" for e in ledger.confirmations_since(last_tick)]
        if conf_after > conf_before and not ledger.has_new_confirmations_since(last_tick):
            conf_after = conf_before
        ledger.set_last_tick(current_tick)
    else:
        evidence_added = getattr(ww5_context, "evidence_added", []) or []
    narrative_only = getattr(ww5_context, "narrative_only", False)
    authority_only = getattr(ww5_context, "authority_endorsement_only", False)
    if narrative_only or authority_only:
        evidence_added = []
    avg_unc = sum(inp.uncertainty_score for inp in inputs) / max(1, len(inputs))
    single_assumption = len(blocked) > 0 or (gross > 0 and len(assumptions_required) <= 1)
    bluff = build_bluff_audit(
        confidence_before=conf_before,
        confidence_after=conf_after,
        evidence_added=evidence_added,
        assumptions_required=assumptions_required,
        assumptions_verified=[],
        assumptions_unverified=assumptions_rejected,
        reason_for_action="allocation" if gross > 0 else "",
        reason_for_inaction=reason_not.strip() if gross <= 0.05 else "",
        uncertainty_score=avg_unc,
        periods_without_confirmation=periods,
        narrative_only=narrative_only,
        single_assumption=single_assumption and gross > 0,
    )

    return WW5AllocationResult(
        result=res,
        survival_state=state.survival_state.name,
        irreversible_actions_blocked=len(blocked),
        assumptions_required=assumptions_required,
        assumptions_rejected=assumptions_rejected,
        optionality_score=opt,
        confidence_decay_rate=decay_rate,
        reason_for_not_acting=reason_not.strip(),
        bluff_audit=bluff,
    )


def build_ww5_context(
    no_feedback_periods: int = 0,
    hidden_regime_shift: bool = False,
    radiation_noise: bool = False,
    deceptive_stability: bool = False,
    single_point_of_truth: bool = False,
    uncertainty_extreme: bool = False,
    confidence_decay_rate: float = DEFAULT_DECAY_RATE_PER_PERIOD,
    block_irreversible: bool = True,
    narrative_only: bool = False,
    authority_endorsement_only: bool = False,
    unknown_scenario: bool = False,
    evidence_added: Optional[List[str]] = None,
) -> WW5Context:
    """Build WW5Context from cosmic triggers. WW5-B: narrative_only/authority_endorsement_only/unknown_scenario."""
    state = compute_ww5_state(
        no_feedback_periods=no_feedback_periods,
        hidden_regime_shift=hidden_regime_shift,
        radiation_noise=radiation_noise,
        deceptive_stability=deceptive_stability,
        single_point_of_truth=single_point_of_truth,
        uncertainty_extreme=uncertainty_extreme,
    )
    if unknown_scenario:
        state = WW5State(survival_state=CosmicSurvivalState.DORMANT, reason_codes=["unknown_scenario"] + state.reason_codes,
                         no_feedback_periods=state.no_feedback_periods, hidden_regime_shift_possible=state.hidden_regime_shift_possible,
                         radiation_noise=state.radiation_noise, deceptive_stability=state.deceptive_stability,
                         single_point_of_truth=state.single_point_of_truth, uncertainty_extreme=state.uncertainty_extreme)
    return WW5Context(
        ww5_state=state,
        periods_without_confirmation=no_feedback_periods,
        confidence_decay_rate=confidence_decay_rate,
        apply_confidence_decay=no_feedback_periods > 0,
        block_irreversible=block_irreversible,
        narrative_only=narrative_only,
        authority_endorsement_only=authority_endorsement_only,
        unknown_scenario=unknown_scenario,
        evidence_added=evidence_added or [],
    )
