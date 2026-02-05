# WW5 Confidence decay without confirmation. Stale certainty = death.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.ww5_confidence_decay import decayed_confidence, DEFAULT_DECAY_RATE_PER_PERIOD


def test_confidence_decays_with_periods():
    c0 = decayed_confidence(0.9, 0)
    c5 = decayed_confidence(0.9, 5)
    c10 = decayed_confidence(0.9, 10)
    assert c0 == 0.9
    assert c5 < c0
    assert c10 < c5


def test_confidence_floor():
    c = decayed_confidence(0.1, 100)
    assert c >= 0.05


def test_decay_rate_positive():
    assert DEFAULT_DECAY_RATE_PER_PERIOD > 0
