# Exposure must be monotone decreasing in hazard and bluff risk (grid test)
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.ww5_evidence import max_gross_from_hazard_and_bluff


def test_max_gross_decreases_in_hazard():
    for bluff in [0.0, 0.5, 1.0]:
        prev = 1.0
        for h in range(4):
            cap = max_gross_from_hazard_and_bluff(h, bluff)
            assert cap <= prev + 1e-9
            prev = cap


def test_max_gross_decreases_in_bluff_risk():
    for h in range(4):
        prev = 1.0
        for b in [0.0, 0.3, 0.6, 1.0]:
            cap = max_gross_from_hazard_and_bluff(h, b)
            assert cap <= prev + 1e-9
            prev = cap


def test_dormant_near_zero():
    cap = max_gross_from_hazard_and_bluff(3, 0.8)
    assert cap <= 0.1
