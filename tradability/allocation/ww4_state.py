"""
WW4: Hazard state machine. Safety invariants > performance.
States: NORMAL, CAUTION, DANGER, SURVIVAL, LOCKDOWN.
Once in SURVIVAL, cannot return to NORMAL without cooldown + revalidation.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

# Gross exposure and max single-weight limits per state (monotonic: worse state = lower cap)
SURVIVAL_GROSS = 0.0
LOCKDOWN_GROSS = 0.0
EMERGENCY_GROSS = 0.05  # DANGER
CAUTION_GROSS = 0.5
MAX_GROSS = 1.0


class SurvivalState(IntEnum):
    NORMAL = 0
    CAUTION = 1
    DANGER = 2
    SURVIVAL = 3
    LOCKDOWN = 4


# Max single-strategy weight per state (concentration monotonicity)
MAX_SINGLE_WEIGHT = {
    SurvivalState.NORMAL: 0.4,
    SurvivalState.CAUTION: 0.3,
    SurvivalState.DANGER: 0.15,
    SurvivalState.SURVIVAL: 0.05,
    SurvivalState.LOCKDOWN: 0.0,
}

# Max gross exposure per state (exposure monotonicity)
MAX_GROSS_BY_STATE = {
    SurvivalState.NORMAL: MAX_GROSS,
    SurvivalState.CAUTION: CAUTION_GROSS,
    SurvivalState.DANGER: EMERGENCY_GROSS,
    SurvivalState.SURVIVAL: SURVIVAL_GROSS + 0.02,  # allow tiny buffer for rounding
    SurvivalState.LOCKDOWN: LOCKDOWN_GROSS,
}


@dataclass
class WW4StateMachine:
    """
    State machine with cooldown. Once in SURVIVAL/LOCKDOWN, cannot go back to NORMAL
    until cooldown_ticks and revalidation_ok.
    """
    state: SurvivalState = SurvivalState.NORMAL
    reason_codes: List[str] = field(default_factory=list)
    cooldown_ticks_remaining: int = 0
    revalidation_ok: bool = False
    # Triggers (inputs)
    telemetry_fail: bool = False
    correlation_crisis: bool = False
    liquidity_crisis: bool = False
    model_drift: bool = False
    infra_failure: bool = False
    near_miss_count: int = 0
    consensus_disagree: bool = False
    sensor_poisoning: bool = False
    blackout: bool = False
    integrity_failure: bool = False

    # Cooldown after leaving SURVIVAL before allowing NORMAL (e.g. 5 ticks)
    COOLDOWN_TICKS = 5

    def max_gross_allowed(self) -> float:
        return MAX_GROSS_BY_STATE[self.state]

    def max_single_weight_allowed(self) -> float:
        return MAX_SINGLE_WEIGHT[self.state]

    def transition(self) -> "WW4StateMachine":
        """Compute next state from triggers. Only allow exit from SURVIVAL after cooldown + revalidation."""
        reasons = []
        proposed = self.state

        # LOCKDOWN: blackout or total sensor poison
        if self.blackout or self.sensor_poisoning:
            proposed = max(proposed, SurvivalState.LOCKDOWN)
            if self.blackout:
                reasons.append("blackout")
            if self.sensor_poisoning:
                reasons.append("sensor_poisoning")

        # SURVIVAL: severe triggers
        if self.telemetry_fail or getattr(self, "integrity_failure", False) or self.consensus_disagree:
            proposed = max(proposed, SurvivalState.SURVIVAL)
            if self.telemetry_fail:
                reasons.append("telemetry_fail")
            if self.integrity_failure:
                reasons.append("integrity_failure")
            if self.consensus_disagree:
                reasons.append("consensus_disagree")

        # DANGER: crisis triggers
        if self.correlation_crisis or self.liquidity_crisis or self.infra_failure:
            proposed = max(proposed, SurvivalState.DANGER)
            if self.correlation_crisis:
                reasons.append("correlation_crisis")
            if self.liquidity_crisis:
                reasons.append("liquidity_crisis")
            if self.infra_failure:
                reasons.append("infra_failure")

        # CAUTION: drift or near-misses
        if self.model_drift or self.near_miss_count >= 2:
            proposed = max(proposed, SurvivalState.CAUTION)
            if self.model_drift:
                reasons.append("model_drift")
            if self.near_miss_count >= 2:
                reasons.append("near_miss")

        # Cannot step down from SURVIVAL/LOCKDOWN without cooldown + revalidation
        if self.state in (SurvivalState.SURVIVAL, SurvivalState.LOCKDOWN):
            if self.cooldown_ticks_remaining > 0:
                return WW4StateMachine(
                    state=self.state,
                    reason_codes=reasons or self.reason_codes,
                    cooldown_ticks_remaining=self.cooldown_ticks_remaining - 1,
                    revalidation_ok=self.revalidation_ok,
                    telemetry_fail=self.telemetry_fail,
                    correlation_crisis=self.correlation_crisis,
                    liquidity_crisis=self.liquidity_crisis,
                    model_drift=self.model_drift,
                    infra_failure=self.infra_failure,
                    near_miss_count=self.near_miss_count,
                    consensus_disagree=self.consensus_disagree,
                    sensor_poisoning=self.sensor_poisoning,
                    blackout=self.blackout,
                    integrity_failure=getattr(self, "integrity_failure", False),
                )
            if not self.revalidation_ok:
                proposed = max(proposed, self.state)
            elif self.cooldown_ticks_remaining == 0 and proposed < self.state:
                proposed = max(proposed, self.state - 1)
                self.cooldown_ticks_remaining = self.COOLDOWN_TICKS

        new_state = SurvivalState(max(proposed, self.state))
        return WW4StateMachine(
            state=new_state,
            reason_codes=reasons or self.reason_codes,
            cooldown_ticks_remaining=self.cooldown_ticks_remaining,
            revalidation_ok=self.revalidation_ok,
            telemetry_fail=self.telemetry_fail,
            correlation_crisis=self.correlation_crisis,
            liquidity_crisis=self.liquidity_crisis,
            model_drift=self.model_drift,
            infra_failure=self.infra_failure,
            near_miss_count=self.near_miss_count,
            consensus_disagree=self.consensus_disagree,
            sensor_poisoning=self.sensor_poisoning,
            blackout=self.blackout,
            integrity_failure=self.integrity_failure,
        )


def compute_ww4_state(
    telemetry_fail: bool = False,
    correlation_crisis: bool = False,
    liquidity_crisis: bool = False,
    model_drift: bool = False,
    infra_failure: bool = False,
    consensus_disagree: bool = False,
    sensor_poisoning: bool = False,
    blackout: bool = False,
    near_miss_count: int = 0,
    previous: Optional[WW4StateMachine] = None,
) -> WW4StateMachine:
    """Compute next WW4 state from triggers and optional previous state (for cooldown)."""
    prev = previous or WW4StateMachine()
    sm = WW4StateMachine(
        state=prev.state,
        reason_codes=prev.reason_codes,
        cooldown_ticks_remaining=prev.cooldown_ticks_remaining,
        revalidation_ok=prev.revalidation_ok,
        telemetry_fail=telemetry_fail or prev.telemetry_fail,
        correlation_crisis=correlation_crisis or prev.correlation_crisis,
        liquidity_crisis=liquidity_crisis or prev.liquidity_crisis,
        model_drift=model_drift or prev.model_drift,
        infra_failure=infra_failure or prev.infra_failure,
        near_miss_count=near_miss_count or prev.near_miss_count,
        consensus_disagree=consensus_disagree or prev.consensus_disagree,
        sensor_poisoning=sensor_poisoning or prev.sensor_poisoning,
        blackout=blackout or prev.blackout,
    )
    sm.integrity_failure = blackout or sensor_poisoning
    return sm.transition()
