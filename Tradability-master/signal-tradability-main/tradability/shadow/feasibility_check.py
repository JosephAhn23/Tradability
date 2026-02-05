"""
Halt when feasibility bounds are violated (e.g. net edge <= 0, AUM > capacity).
"""

from typing import Optional

from .config import ShadowConfig


def check_feasibility_halt(
    config: ShadowConfig,
    current_equity: float,
    annual_turnover: float,
    net_edge_bps: Optional[float] = None,
) -> tuple:
    """
    Returns (should_halt: bool, reason: str).
    Halt if enabled and (AUM > max_aum or net_edge_bps < min_net_edge_bps).
    """
    if not config.halt_on_feasibility_violation:
        return False, ""

    if config.max_aum is not None and current_equity > config.max_aum:
        return True, f"AUM {current_equity:.0f} exceeds max_aum {config.max_aum:.0f}"

    if config.min_net_edge_bps is not None and net_edge_bps is not None:
        if net_edge_bps < config.min_net_edge_bps:
            return True, f"Net edge {net_edge_bps:.2f} bps below min {config.min_net_edge_bps} bps"

    return False, ""
