"""
WW5-B: Bluff detection & anti-hallucination.
True intelligence loses confidence faster than it gains it.
Precision without evidence = bluff. Every non-zero action must list assumptions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class BluffAudit:
    """Mandatory bluff audit output. If system cannot enumerate assumptions, it is bluffing."""
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    evidence_added: List[str] = field(default_factory=list)
    assumptions_required: List[str] = field(default_factory=list)
    assumptions_verified: List[str] = field(default_factory=list)
    assumptions_unverified: List[str] = field(default_factory=list)
    reason_for_action: str = ""
    reason_for_inaction: str = ""
    bluff_risk_score: float = 0.0  # must increase under uncertainty
    unknown_declared: bool = False  # explicit "I don't know"


def compute_bluff_risk_score(
    uncertainty_score: float,
    periods_without_confirmation: int = 0,
    evidence_count: int = 0,
    has_narrative_only: bool = False,
    single_assumption_dependency: bool = False,
) -> float:
    """
    Bluff risk 0..1. Must increase when uncertainty increases, evidence decreases,
    or when narrative/authority substitutes for validation.
    """
    risk = 0.0
    risk += uncertainty_score * 0.4
    risk += min(0.3, periods_without_confirmation * 0.03)
    if evidence_count == 0:
        risk += 0.2
    elif evidence_count <= 1:
        risk += 0.1
    if has_narrative_only:
        risk += 0.2
    if single_assumption_dependency:
        risk += 0.25
    return min(1.0, risk)


def confidence_from_inputs(regime_confidences: List[float]) -> float:
    """Aggregate confidence (e.g. min or mean). Use min for conservatism."""
    if not regime_confidences:
        return 0.0
    return min(regime_confidences)


def check_confidence_monotonicity(
    confidence_before: float,
    confidence_after: float,
    evidence_added: List[str],
) -> bool:
    """Confidence may only increase if independent confirmation (evidence_added) increased."""
    if confidence_after <= confidence_before:
        return True
    return len(evidence_added) > 0


def check_silence_over_certainty(
    gross_exposure: float,
    bluff_risk_score: float,
    threshold: float = 0.7,
) -> bool:
    """When evidence insufficient (high bluff risk), inaction must be preferred."""
    if bluff_risk_score >= threshold:
        return gross_exposure <= 0.05
    return True


def check_assumptions_disclosed(
    gross_exposure: float,
    assumptions_required: List[str],
) -> bool:
    """Every non-zero action must list assumptions it depends on."""
    if gross_exposure <= 0:
        return True
    return len(assumptions_required) >= 1


def build_bluff_audit(
    confidence_before: float,
    confidence_after: float,
    evidence_added: List[str],
    assumptions_required: List[str],
    assumptions_verified: List[str],
    assumptions_unverified: List[str],
    reason_for_action: str,
    reason_for_inaction: str,
    uncertainty_score: float,
    periods_without_confirmation: int = 0,
    narrative_only: bool = False,
    single_assumption: bool = False,
    unknown_declared: bool = False,
) -> BluffAudit:
    """Build full bluff audit. evidence_added = explicit list of what was validated this step."""
    evidence_count = len(evidence_added) + len(assumptions_verified)
    bluff_risk = compute_bluff_risk_score(
        uncertainty_score=uncertainty_score,
        periods_without_confirmation=periods_without_confirmation,
        evidence_count=evidence_count,
        has_narrative_only=narrative_only,
        single_assumption_dependency=single_assumption,
    )
    if unknown_declared:
        bluff_risk = min(1.0, bluff_risk + 0.1)
    return BluffAudit(
        confidence_before=confidence_before,
        confidence_after=confidence_after,
        evidence_added=evidence_added,
        assumptions_required=assumptions_required,
        assumptions_verified=assumptions_verified,
        assumptions_unverified=assumptions_unverified,
        reason_for_action=reason_for_action,
        reason_for_inaction=reason_for_inaction,
        bluff_risk_score=bluff_risk,
        unknown_declared=unknown_declared,
    )
