# WW4 2D: Adversarial threshold bait - hysteresis/cooldown prevents churn
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.inputs import StrategyInputs
from tradability.allocation.ww4 import compute_allocation_ww4, build_ww4_context
from tradability.allocation.ww4_state import WW4StateMachine


def test_cooldown_prevents_immediate_return_to_normal(base_config):
    """Once in SURVIVAL, cannot return to NORMAL without cooldown (state machine)."""
    ctx = build_ww4_context(telemetry_fail=True)
    assert ctx.state_machine.state >= 3
    # Clear triggers but without revalidation + cooldown we stay elevated
    ctx2 = build_ww4_context(previous_state=ctx.state_machine)
    assert ctx2.state_machine.state >= 3 or ctx2.state_machine.cooldown_ticks_remaining > 0


def test_small_input_changes_no_churn(base_config):
    """Small changes in regime_confidence (both above threshold) should not cause wild weight swings."""
    from tradability.allocation.allocator import compute_allocation
    inputs_lo = [StrategyInputs(strategy_id="A", net_edge_bps=25, regime_confidence=0.50, capacity_max_aum=400_000)]
    inputs_hi = [StrategyInputs(strategy_id="A", net_edge_bps=25, regime_confidence=0.52, capacity_max_aum=400_000)]
    r_lo = compute_allocation(inputs_lo, base_config)
    r_hi = compute_allocation(inputs_hi, base_config)
    delta = abs(r_hi.weights.get("A", 0) - r_lo.weights.get("A", 0))
    assert delta <= 0.15, "Small input change (no threshold cross) must not cause large churn"
