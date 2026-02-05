"""
WW4: Multi-source consensus (Byzantine tolerance).
If estimators disagree beyond threshold → assume worst-case and flag for hazard.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

# Disagreement threshold: e.g. regime_confidence diff > 0.3 => disagree
REGIME_DISAGREEMENT_THRESHOLD = 0.25
FEASIBILITY_DISAGREEMENT_THRESHOLD = 0.2  # capacity/edge relative
CORRELATION_DISAGREEMENT_THRESHOLD = 0.3


@dataclass
class ConsensusResult:
    """Result of 2-of-N consensus check."""
    agree: bool
    value_used: float  # worst-case when disagree
    values: List[float]
    reason: str = ""


def consensus_regime(estimates: List[float]) -> ConsensusResult:
    """Require agreement on regime_confidence. If disagree → use min (worst-case)."""
    if len(estimates) < 2:
        return ConsensusResult(agree=True, value_used=min(estimates) if estimates else 0.0, values=estimates)
    lo, hi = min(estimates), max(estimates)
    if hi - lo > REGIME_DISAGREEMENT_THRESHOLD:
        return ConsensusResult(
            agree=False,
            value_used=lo,
            values=estimates,
            reason="regime_disagree",
        )
    return ConsensusResult(agree=True, value_used=sum(estimates) / len(estimates), values=estimates)


def consensus_feasibility(estimates: List[float]) -> ConsensusResult:
    """Feasibility/capacity estimates. If disagree → use min (worst-case)."""
    if len(estimates) < 2:
        return ConsensusResult(agree=True, value_used=min(estimates) if estimates else 0.0, values=estimates)
    lo, hi = min(estimates), max(estimates)
    if hi > 0 and (hi - lo) / hi > FEASIBILITY_DISAGREEMENT_THRESHOLD:
        return ConsensusResult(
            agree=False,
            value_used=lo,
            values=estimates,
            reason="feasibility_disagree",
        )
    return ConsensusResult(agree=True, value_used=sum(estimates) / len(estimates), values=estimates)


def consensus_correlation_risk(estimates: List[float]) -> ConsensusResult:
    """Correlation risk (higher = worse). If disagree → use max (worst-case)."""
    if len(estimates) < 2:
        return ConsensusResult(agree=True, value_used=max(estimates) if estimates else 0.0, values=estimates)
    lo, hi = min(estimates), max(estimates)
    if hi - lo > CORRELATION_DISAGREEMENT_THRESHOLD:
        return ConsensusResult(
            agree=False,
            value_used=hi,
            values=estimates,
            reason="correlation_disagree",
        )
    return ConsensusResult(agree=True, value_used=sum(estimates) / len(estimates), values=estimates)


def check_sensor_poisoning(
    regime_confidence: float,
    uncertainty_score: float,
    net_edge_bps: float,
    divergence_bps: Optional[float],
    feasibility_ratio: float,
    turnover: Optional[float],
) -> bool:
    """
    Detect inconsistent (lying) metrics: e.g. high edge + high divergence, low uncertainty + high turnover.
    Returns True if poisoning detected.
    """
    # Shadow divergence contradicts "high edge"
    if divergence_bps is not None and net_edge_bps > 50 and divergence_bps > 100:
        return True
    # Feasibility bound contradicts "high edge"
    if feasibility_ratio < 0.2 and net_edge_bps > 80:
        return True
    # Turnover spikes contradict "low uncertainty"
    if uncertainty_score < 0.2 and turnover is not None and turnover > 2.0:
        return True
    # Regime "perfect" but feasibility near zero
    if regime_confidence > 0.95 and feasibility_ratio < 0.1:
        return True
    return False
