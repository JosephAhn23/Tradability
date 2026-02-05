"""
Paper trading (broker-simulated) pipeline.

Orders are sent to a broker paper account (e.g. Alpaca).
No custom simulator; real order lifecycle and fill timing.
"""

from .broker import AlpacaBroker
from .signals import generate_target_positions
from .execution import compute_orders_from_targets
from .ledger import Ledger
from .diagnostics import DiagnosticsLogger

__all__ = [
    "AlpacaBroker",
    "generate_target_positions",
    "compute_orders_from_targets",
    "Ledger",
    "DiagnosticsLogger",
]
