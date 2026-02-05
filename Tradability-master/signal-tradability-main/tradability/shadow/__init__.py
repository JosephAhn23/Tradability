"""
Shadow Trading (Forward Execution Simulation).

Strategies are evaluated using a forward, walk-forward execution simulator:
orders are generated based only on information available at decision time,
deterministic fills with documented assumptions, realistic cost/impact models,
and full logging. No broker; no identity or compliance dependency.

This mirrors pre-deployment validation used in professional quant research.
"""

from .config import load_config, ShadowConfig
from .signals import generate_target_positions
from .execution import orders_from_targets
from .fill_model import fill_orders_deterministic
from .ledger import ShadowLedger
from .feasibility_check import check_feasibility_halt

__all__ = [
    "load_config",
    "ShadowConfig",
    "generate_target_positions",
    "orders_from_targets",
    "fill_orders_deterministic",
    "ShadowLedger",
    "check_feasibility_halt",
]
