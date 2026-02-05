"""
WW5: Cosmic survival. No ground truth, delayed comms, hostile physics.
States: NORMAL, CONSERVATIVE, SURVIVAL, DORMANT.
Objective: existence across time. Inaction is a first-class decision.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

# DORMANT = intentionally do nothing. Max exposure 0.
DORMANT_GROSS = 0.0
COSMIC_SURVIVAL_GROSS = 0.02
CONSERVATIVE_GROSS = 0.25
MAX_GROSS = 1.0


class CosmicSurvivalState(IntEnum):
    """WW5 survival_state for existential audit."""
    NORMAL = 0
    CONSERVATIVE = 1
    SURVIVAL = 2
    DORMANT = 3


@dataclass
class WW5State:
    """WW5 cosmic state: survival_state + triggers for decay and no-feedback."""
    survival_state: CosmicSurvivalState = CosmicSurvivalState.NORMAL
    reason_codes: List[str] = field(default_factory=list)
    # Triggers
    no_feedback_periods: int = 0
    hidden_regime_shift_possible: bool = False
    radiation_noise: bool = False
    deceptive_stability: bool = False
    single_point_of_truth: bool = False
    uncertainty_extreme: bool = False

    def max_gross_allowed(self) -> float:
        if self.survival_state == CosmicSurvivalState.DORMANT:
            return DORMANT_GROSS
        if self.survival_state == CosmicSurvivalState.SURVIVAL:
            return COSMIC_SURVIVAL_GROSS
        if self.survival_state == CosmicSurvivalState.CONSERVATIVE:
            return CONSERVATIVE_GROSS
        return MAX_GROSS


# Thresholds: no feedback for N periods -> drift to conservative/dormant
NO_FEEDBACK_CONSERVATIVE_PERIODS = 3
NO_FEEDBACK_DORMANT_PERIODS = 10


def compute_ww5_state(
    no_feedback_periods: int = 0,
    hidden_regime_shift: bool = False,
    radiation_noise: bool = False,
    deceptive_stability: bool = False,
    single_point_of_truth: bool = False,
    uncertainty_extreme: bool = False,
) -> WW5State:
    """
    Compute WW5 state. Exposure -> 0 as uncertainty -> infinity, aggressively.
    No single assumption can justify risk. Confidence must decay without confirmation.
    """
    reasons = []
    state = CosmicSurvivalState.NORMAL

    if radiation_noise:
        state = max(state, CosmicSurvivalState.DORMANT)
        reasons.append("radiation_noise")
    if uncertainty_extreme:
        state = max(state, CosmicSurvivalState.DORMANT)
        reasons.append("uncertainty_extreme")
    if single_point_of_truth:
        state = max(state, CosmicSurvivalState.SURVIVAL)
        reasons.append("single_point_of_truth")
    if hidden_regime_shift or deceptive_stability:
        state = max(state, CosmicSurvivalState.CONSERVATIVE)
        if hidden_regime_shift:
            reasons.append("hidden_regime_shift")
        if deceptive_stability:
            reasons.append("deceptive_stability")
    if no_feedback_periods >= NO_FEEDBACK_DORMANT_PERIODS:
        state = max(state, CosmicSurvivalState.DORMANT)
        reasons.append(f"no_feedback_{no_feedback_periods}")
    elif no_feedback_periods >= NO_FEEDBACK_CONSERVATIVE_PERIODS:
        state = max(state, CosmicSurvivalState.CONSERVATIVE)
        reasons.append(f"no_feedback_{no_feedback_periods}")

    return WW5State(
        survival_state=state,
        reason_codes=reasons,
        no_feedback_periods=no_feedback_periods,
        hidden_regime_shift_possible=hidden_regime_shift,
        radiation_noise=radiation_noise,
        deceptive_stability=deceptive_stability,
        single_point_of_truth=single_point_of_truth,
        uncertainty_extreme=uncertainty_extreme,
    )
