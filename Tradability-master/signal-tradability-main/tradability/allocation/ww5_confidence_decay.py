"""
WW5: Confidence must decay without confirmation. Stale certainty = death.
If no new trustworthy evidence arrives, effective confidence decays over time.
"""

from dataclasses import dataclass
from typing import Optional

# Decay rate per period without confirmation (e.g. 5% per period)
DEFAULT_DECAY_RATE_PER_PERIOD = 0.05
# Floor: confidence never below this
CONFIDENCE_FLOOR = 0.05


@dataclass
class ConfidenceDecayState:
    """Tracks periods without confirmation and applies decay."""
    periods_without_confirmation: int = 0
    decay_rate: float = DEFAULT_DECAY_RATE_PER_PERIOD

    @property
    def confidence_decay_rate(self) -> float:
        return self.decay_rate


def decayed_confidence(
    raw_confidence: float,
    periods_without_confirmation: int,
    decay_rate: float = DEFAULT_DECAY_RATE_PER_PERIOD,
) -> float:
    """
    Effective confidence = raw_confidence * (1 - decay_rate)^periods_without_confirmation.
    Aggressive: without confirmation, confidence decays.
    """
    if periods_without_confirmation <= 0:
        return max(CONFIDENCE_FLOOR, min(1.0, raw_confidence))
    factor = (1.0 - decay_rate) ** periods_without_confirmation
    return max(CONFIDENCE_FLOOR, min(1.0, raw_confidence * factor))


def optionality_score(weights: dict, gross: float) -> float:
    """
    Optionality: higher when less concentrated and less committed.
    1 - concentration + cash_share component. Range ~0..1.
    """
    if not weights or gross <= 0:
        return 1.0
    n = len(weights)
    max_w = max(weights.values()) if weights else 0
    concentration = max_w
    cash_share = 1.0 - gross
    # Optionality: low concentration + high cash = high optionality
    return min(1.0, (1.0 - concentration) * 0.5 + cash_share * 0.5 + (1.0 / max(n, 1)) * 0.2)
