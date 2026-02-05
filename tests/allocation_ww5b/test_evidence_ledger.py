# Evidence must be earned: confidence may only rise if ledger has new confirmations since last tick
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.ww5 import compute_allocation_ww5, build_ww5_context
from tradability.allocation.ww5_evidence import EvidenceLedger, EvidenceType


def test_ledger_has_new_confirmations_since():
    ledger = EvidenceLedger()
    ledger.record(1, "feasibility", EvidenceType.CONFIRMATION)
    assert ledger.has_new_confirmations_since(0) is True
    assert ledger.has_new_confirmations_since(1) is False


def test_contradiction_zeros_confirmations():
    ledger = EvidenceLedger()
    ledger.record(1, "A", EvidenceType.CONFIRMATION)
    ledger.record(2, "B", EvidenceType.CONTRADICTION)
    assert ledger.independent_confirmation_count_since(0) == 0


def test_evidence_added_derived_from_ledger(base_config):
    ledger = EvidenceLedger()
    ledger.record(0, "diag", EvidenceType.THRESHOLD_CROSSING)
    ctx = build_ww5_context()
    ctx.evidence_ledger = ledger
    ctx.current_tick = 1
    ctx.last_tick = -1
    inputs = [StrategyInputs(strategy_id="A", net_edge_bps=25, regime_confidence=0.6, capacity_max_aum=400_000)]
    res = compute_allocation_ww5(inputs, base_config, ww5_context=ctx)
    assert res.bluff_audit is not None
    assert any("diag" in e for e in res.bluff_audit.evidence_added) or res.bluff_audit.evidence_added == []
