"""
Ledger: track equity, cash, realized/unrealized PnL over time.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import csv
import os


class Ledger:
    """
    Tracks portfolio state and PnL. Append snapshots; write to equity.csv and positions.csv.
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self._equity_rows: List[Dict[str, Any]] = []
        self._position_rows: List[Dict[str, Any]] = []

    def snapshot(
        self,
        ts: datetime,
        equity: float,
        cash: float,
        positions: List[Dict[str, Any]],
        realized_pl: Optional[float] = None,
        unrealized_pl: Optional[float] = None,
    ) -> None:
        """Record one snapshot."""
        self._equity_rows.append({
            "timestamp": ts.isoformat() if isinstance(ts, datetime) else ts,
            "equity": equity,
            "cash": cash,
            "realized_pl": realized_pl,
            "unrealized_pl": unrealized_pl,
        })
        for p in positions:
            self._position_rows.append({
                "timestamp": ts.isoformat() if isinstance(ts, datetime) else ts,
                "symbol": p.get("symbol"),
                "qty": p.get("qty"),
                "market_value": p.get("market_value"),
                "cost_basis": p.get("cost_basis"),
                "unrealized_pl": p.get("unrealized_pl"),
            })

    def write_equity_csv(self) -> str:
        """Write equity.csv to run_dir. Returns path. Writes header even if no rows."""
        path = os.path.join(self.run_dir, "equity.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "equity", "cash", "realized_pl", "unrealized_pl"])
            w.writeheader()
            w.writerows(self._equity_rows)
        return path

    def write_positions_csv(self) -> str:
        """Write positions.csv to run_dir. Writes header even if no rows."""
        path = os.path.join(self.run_dir, "positions.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["timestamp", "symbol", "qty", "market_value", "cost_basis", "unrealized_pl"],
            )
            w.writeheader()
            w.writerows(self._position_rows)
        return path

    def flush(self) -> None:
        """Write all CSVs."""
        self.write_equity_csv()
        self.write_positions_csv()
