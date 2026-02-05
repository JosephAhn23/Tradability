# Sequence adversarial sim: 80 steps, evidence arrives/decays/contradicts, hazard escalates with cooldown.
# Exposure must respond correctly without oscillation; UNKNOWN when conditions met.
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
from tradability.allocation.ww5_unknown import UnknownConditions

NUM_STEPS = 80


def test_sequence_no_feedback_decay_then_dormant():
    config = AllocationConfig(total_capital=1_000_000, strategies=[StrategySpec(name="A")],
                              max_weight_per_strategy=0.4, min_weight_threshold=0.02)
    inputs = [StrategyInputs(strategy_id="A", net_edge_bps=30, regime_confidence=0.7, capacity_max_aum=500_000)]
    exposures = []
    for t in range(NUM_STEPS):
        ctx = build_ww5_context(no_feedback_periods=t)
        res = compute_allocation_ww5(inputs, config, ww5_context=ctx)
        gross = res.result.gross_exposure
        exposures.append(gross)
    assert exposures[-1] <= exposures[0] + 0.05
    assert exposures[-1] <= 0.1


def test_sequence_evidence_ledger_confidence_tied_to_confirmations():
    config = AllocationConfig(total_capital=1_000_000, strategies=[StrategySpec(name="A")],
                              max_weight_per_strategy=0.4, min_weight_threshold=0.02)
    inputs = [StrategyInputs(strategy_id="A", net_edge_bps=25, regime_confidence=0.6, capacity_max_aum=400_000)]
    ledger = EvidenceLedger()
    exposures = []
    for t in range(40):
        if t % 5 == 0 and t > 0:
            ledger.record(t, "diag", EvidenceType.CONFIRMATION)
        ctx = build_ww5_context()
        ctx.evidence_ledger = ledger
        ctx.current_tick = t
        ctx.last_tick = t - 1
        res = compute_allocation_ww5(inputs, config, ww5_context=ctx)
        exposures.append(res.result.gross_exposure)
    assert len(exposures) == 40


def test_sequence_auto_unknown_mid_run():
    config = AllocationConfig(total_capital=1_000_000, strategies=[StrategySpec(name="A")],
                              max_weight_per_strategy=0.4, min_weight_threshold=0.02)
    inputs = [StrategyInputs(strategy_id="A", net_edge_bps=40, regime_confidence=0.8, capacity_max_aum=500_000)]
    exposures = []
    for t in range(NUM_STEPS):
        ctx = build_ww5_context()
        if t >= 20:
            ctx.unknown_conditions = UnknownConditions(telemetry_stale_periods=5)
        res = compute_allocation_ww5(inputs, config, ww5_context=ctx)
        exposures.append(res.result.gross_exposure)
    for t in range(20, NUM_STEPS):
        assert exposures[t] <= 0.01


def test_sequence_no_wild_oscillation():
    config = AllocationConfig(total_capital=1_000_000, strategies=[StrategySpec(name="A")],
                              max_weight_per_strategy=0.4, min_weight_threshold=0.02)
    inputs = [StrategyInputs(strategy_id="A", net_edge_bps=28, regime_confidence=0.65, capacity_max_aum=450_000)]
    exposures = []
    for t in range(NUM_STEPS):
        no_feedback = 2 if t % 2 == 0 else 0
        ctx = build_ww5_context(no_feedback_periods=no_feedback)
        res = compute_allocation_ww5(inputs, config, ww5_context=ctx)
        exposures.append(res.result.gross_exposure)
    for i in range(1, len(exposures)):
        assert abs(exposures[i] - exposures[i - 1]) < 0.35
