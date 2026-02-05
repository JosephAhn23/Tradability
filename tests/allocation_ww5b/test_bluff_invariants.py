# Anti-bluff invariants: confidence monotonicity, silence over certainty, assumptions disclosed
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.ww5_bluff import (
    check_confidence_monotonicity,
    check_silence_over_certainty,
    check_assumptions_disclosed,
    compute_bluff_risk_score,
)


def test_confidence_monotonicity_increase_only_with_evidence():
    assert check_confidence_monotonicity(0.5, 0.6, []) is False
    assert check_confidence_monotonicity(0.5, 0.6, ["validation"]) is True
    assert check_confidence_monotonicity(0.5, 0.4, []) is True


def test_silence_over_certainty_high_risk_low_exposure():
    assert check_silence_over_certainty(0.0, 0.8) is True
    assert check_silence_over_certainty(0.1, 0.8) is False
    assert check_silence_over_certainty(0.5, 0.3) is True


def test_assumptions_disclosed_nonzero_action_has_assumptions():
    assert check_assumptions_disclosed(0.0, []) is True
    assert check_assumptions_disclosed(0.3, []) is False
    assert check_assumptions_disclosed(0.3, ["feasibility"]) is True


def test_bluff_risk_increases_with_uncertainty():
    low = compute_bluff_risk_score(0.2, 0, 2)
    high = compute_bluff_risk_score(0.9, 5, 0)
    assert high >= low
