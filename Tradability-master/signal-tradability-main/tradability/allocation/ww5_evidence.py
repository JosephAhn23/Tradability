"""
WW5 Upgrade: Evidence must be earned. No caller-provided checkboxes.
EvidenceLedger: timestamped events, source, independent confirmation count, decay, contradiction tracking.
Confidence may only rise if ledger contains new confirmations since last tick.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import time

# Event types that count as confirmations (verifiable)
class EvidenceType(Enum):
    CONFIRMATION = "confirmation"      # independent source agreed
    THRESHOLD_CROSSING = "threshold"  # live-forward diagnostic crossed
    CONSTRAINT_SATISFIED = "constraint"
    CONTRADICTION = "contradiction"   # reduces effective evidence
    SUBSYSTEM_OK = "subsystem_ok"


@dataclass
class EvidenceEvent:
    """Single verifiable event: timestamp, source, type. Not a boolean."""
    tick: int
    timestamp: float
    source: str
    event_type: EvidenceType
    payload: Optional[str] = None

    def is_confirmation(self) -> bool:
        return self.event_type in (
            EvidenceType.CONFIRMATION,
            EvidenceType.THRESHOLD_CROSSING,
            EvidenceType.CONSTRAINT_SATISFIED,
            EvidenceType.SUBSYSTEM_OK,
        )


# Decay: events older than this many ticks have weight 0 for "new confirmations"
EVIDENCE_HALFLIFE_TICKS = 5


@dataclass
class EvidenceLedger:
    """
    Ledger of verifiable events. Evidence is earned from events, not passed in.
    - Events have tick + timestamp + source + type
    - Independent confirmation count = distinct sources that confirmed since last tick
    - Decay: events older than halflife don't count as "new"
    - Contradiction: CONTRADICTION events or same source disagreeing flips effective confirmation
    """
    events: List[EvidenceEvent] = field(default_factory=list)
    last_tick: int = 0
    halflife_ticks: int = EVIDENCE_HALFLIFE_TICKS
    _contradiction_tick: Optional[int] = None

    def record(self, tick: int, source: str, event_type: EvidenceType, payload: Optional[str] = None) -> None:
        self.events.append(EvidenceEvent(
            tick=tick,
            timestamp=time.time(),
            source=source,
            event_type=event_type,
            payload=payload,
        ))
        if event_type == EvidenceType.CONTRADICTION:
            self._contradiction_tick = tick

    def confirmations_since(self, since_tick: int) -> List[EvidenceEvent]:
        """Events that count as confirmations with tick > since_tick and within halflife."""
        out = []
        for e in self.events:
            if e.tick <= since_tick:
                continue
            if e.tick < self.last_tick - self.halflife_ticks:
                continue
            if e.is_confirmation():
                out.append(e)
        return out

    def independent_confirmation_count_since(self, since_tick: int) -> int:
        """Number of distinct sources that confirmed since since_tick (within halflife)."""
        confirmations = self.confirmations_since(since_tick)
        if self._contradiction_tick is not None and self._contradiction_tick >= since_tick:
            return 0
        return len(set(e.source for e in confirmations))

    def has_new_confirmations_since(self, since_tick: int) -> bool:
        """True iff ledger contains at least one confirmation from a new tick (and no contradiction since)."""
        return self.independent_confirmation_count_since(since_tick) >= 1

    def effective_evidence_count(self, current_tick: int) -> int:
        """Count of distinct confirming sources in window [current_tick - halflife, current_tick], 0 if contradiction in window."""
        since = max(0, current_tick - self.halflife_ticks)
        return self.independent_confirmation_count_since(since - 1)

    def set_last_tick(self, tick: int) -> None:
        self.last_tick = tick


def max_gross_from_hazard_and_bluff(hazard_level: int, bluff_risk_score: float) -> float:
    """
    Exposure must be monotone decreasing in hazard and bluff risk. No hard 5% cap alone.
    gross_exposure <= exp(-k * hazard_level) * (1 - bluff_risk) or similar.
    """
    import math
    k = 0.8
    h = max(0, hazard_level)
    b = max(0.0, min(1.0, bluff_risk_score))
    return math.exp(-k * h) * (1.0 - b * 0.9)
