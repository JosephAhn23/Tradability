"""
Alpha Feasibility Bounds: upper-bound feasible alpha and zero-alpha boundary.

Estimates net-edge bounds (not alpha prediction), capacity/turnover sensitivity,
and regime-conditioned feasibility. No paid data; reproducible and auditable.
"""

from .config import load_config, FeasibilityConfig
from .edge import compute_gross_edge_proxy
from .costs import CostModel, compute_total_cost_bps
from .capacity import compute_impact_bps, net_edge_surface, zero_alpha_boundary
from .regime import compute_regimes, net_edge_by_regime
from .report import run_report

__all__ = [
    "load_config",
    "FeasibilityConfig",
    "compute_gross_edge_proxy",
    "CostModel",
    "compute_total_cost_bps",
    "compute_impact_bps",
    "net_edge_surface",
    "zero_alpha_boundary",
    "compute_regimes",
    "net_edge_by_regime",
    "run_report",
]
