"""
WW5 Upgrade: UNKNOWN must be auto-triggered from conditions, not a flag.
compute_unknown_conditions() -> True when evidence insufficient, estimators disagree,
telemetry stale, drift persists, feasibility near 0 with high uncertainty, or subsystem failure.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

# Thresholds: exceed these -> declare UNKNOWN (DORMANT)
ESTIMATOR_DISAGREEMENT_THRESHOLD = 0.3
TELEMETRY_STALE_THRESHOLD = 5
DRIFT_PERSIST_THRESHOLD = 3
FEASIBILITY_NEAR_ZERO = 0.05
HIGH_UNCERTAINTY = 0.7


@dataclass
class UnknownConditions:
    """Observable inputs for auto-UNKNOWN. All derived from state/telemetry, not caller flag."""
    estimator_disagreement: float = 0.0
    telemetry_stale_periods: int = 0
    drift_flags_persist_count: int = 0
    feasibility_ratio: float = 1.0
    uncertainty_score: float = 0.0
    subsystem_failure: bool = False
    telemetry_integrity_fail: bool = False

    def reasons(self) -> list:
        out = []
        if self.estimator_disagreement >= ESTIMATOR_DISAGREEMENT_THRESHOLD:
            out.append("estimator_disagreement")
        if self.telemetry_stale_periods >= TELEMETRY_STALE_THRESHOLD:
            out.append("telemetry_stale")
        if self.drift_flags_persist_count >= DRIFT_PERSIST_THRESHOLD:
            out.append("drift_persist")
        if self.feasibility_ratio <= FEASIBILITY_NEAR_ZERO and self.uncertainty_score >= HIGH_UNCERTAINTY:
            out.append("feasibility_low_uncertainty_high")
        if self.subsystem_failure:
            out.append("subsystem_failure")
        if self.telemetry_integrity_fail:
            out.append("telemetry_integrity_fail")
        return out


def compute_unknown_conditions(
    estimator_disagreement: float = 0.0,
    telemetry_stale_periods: int = 0,
    drift_flags_persist_count: int = 0,
    feasibility_ratio: float = 1.0,
    uncertainty_score: float = 0.0,
    subsystem_failure: bool = False,
    telemetry_integrity_fail: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Returns (should_declare_unknown, reason_codes).
    UNKNOWN emerges from conditions; no manual override needed for production.
    """
    cond = UnknownConditions(
        estimator_disagreement=estimator_disagreement,
        telemetry_stale_periods=telemetry_stale_periods,
        drift_flags_persist_count=drift_flags_persist_count,
        feasibility_ratio=feasibility_ratio,
        uncertainty_score=uncertainty_score,
        subsystem_failure=subsystem_failure,
        telemetry_integrity_fail=telemetry_integrity_fail,
    )
    reasons = cond.reasons()
    return (len(reasons) > 0, reasons)
