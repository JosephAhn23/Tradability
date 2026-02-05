"""
WW4: Risk token budget. In SURVIVAL, tokens near zero.
Spending requires 2-of-3 confirmations (regime, feasibility, stress all agree it's safe).
"""

from dataclasses import dataclass
from typing import Optional

from .ww4_state import SurvivalState, MAX_GROSS_BY_STATE


@dataclass
class RiskTokenBudget:
    """Budget of risk tokens; gross exposure cannot exceed tokens (as fraction of capital)."""
    tokens: float  # 0..1, fraction of max gross allowed
    confirmations: int  # 0, 1, 2, or 3 (2-of-3 required in SURVIVAL)
    regime_ok: bool = False
    feasibility_ok: bool = False
    stress_ok: bool = False

    def max_gross_from_tokens(self) -> float:
        return max(0.0, min(1.0, self.tokens))

    def has_2_of_3(self) -> bool:
        n = sum([self.regime_ok, self.feasibility_ok, self.stress_ok])
        return n >= 2


def tokens_for_state(state: SurvivalState) -> float:
    """Risk tokens allowed in this state. SURVIVAL/LOCKDOWN = near zero."""
    return MAX_GROSS_BY_STATE.get(state, 0.0)


def compute_budget(
    state: SurvivalState,
    regime_ok: bool = False,
    feasibility_ok: bool = False,
    stress_ok: bool = False,
) -> RiskTokenBudget:
    """
    In SURVIVAL/LOCKDOWN, tokens = state limit and spending requires 2-of-3.
    In NORMAL/CAUTION/DANGER, tokens = state limit (no 2-of-3 required for basic allocation).
    """
    tokens = tokens_for_state(state)
    if state >= SurvivalState.SURVIVAL:
        # Require 2-of-3 to use any tokens
        confirmations = sum([regime_ok, feasibility_ok, stress_ok])
        if confirmations < 2:
            tokens = 0.0
    return RiskTokenBudget(
        tokens=tokens,
        confirmations=sum([regime_ok, feasibility_ok, stress_ok]),
        regime_ok=regime_ok,
        feasibility_ok=feasibility_ok,
        stress_ok=stress_ok,
    )
