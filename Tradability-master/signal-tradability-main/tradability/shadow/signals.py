"""
Shadow signals: target positions from Tradability logic, as_of_date enforced (no lookahead).
"""

from datetime import datetime
from typing import List, Tuple, Optional

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse paper adapter (no broker dependency)
from tradability.paper.signals import generate_target_positions as _generate_target_positions


def generate_target_positions(
    tickers: List[str],
    signal_name: str,
    as_of_date: datetime,
    equity: float,
    lookback_start: Optional[datetime] = None,
    position_sizing: str = "equal_weight",
    fixed_dollar_per_name: Optional[float] = None,
    max_position_pct: float = 0.25,
    quantile: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Target positions as of as_of_date using only data up to that date.
    Returns list of (symbol, target_dollar_value).
    """
    return _generate_target_positions(
        tickers=tickers,
        signal_name=signal_name,
        as_of_date=as_of_date,
        lookback_start=lookback_start,
        position_sizing=position_sizing,
        fixed_dollar_per_name=fixed_dollar_per_name,
        max_position_pct=max_position_pct,
        equity=equity,
        quantile=quantile,
    )
