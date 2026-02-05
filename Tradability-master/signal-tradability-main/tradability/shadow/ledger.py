"""
Shadow ledger: positions (shares), cash, equity, realized PnL.
"""

from datetime import datetime
from typing import Any, Dict, List

import csv
import os


class ShadowLedger:
    """Track positions, cash, equity over time. Append-only; write to CSVs."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self._equity_rows: List[Dict[str, Any]] = []
        self._position_rows: List[Dict[str, Any]] = []
        self._order_rows: List[Dict[str, Any]] = []
        self._fill_rows: List[Dict[str, Any]] = []
        self._diagnostic_rows: List[Dict[str, Any]] = []

    def snapshot_equity(self, ts: datetime, equity: float, cash: float, realized_pl: float = 0) -> None:
        self._equity_rows.append({
            "timestamp": ts.isoformat() if isinstance(ts, datetime) else ts,
            "equity": equity,
            "cash": cash,
            "realized_pl": realized_pl,
        })

    def snapshot_positions(self, ts: datetime, positions_shares: Dict[str, float], prices: Dict[str, float]) -> None:
        for sym, qty in positions_shares.items():
            mv = qty * prices.get(sym, 0)
            self._position_rows.append({
                "timestamp": ts.isoformat() if isinstance(ts, datetime) else ts,
                "symbol": sym,
                "qty": qty,
                "market_value": mv,
            })

    def log_orders(self, ts: datetime, orders: List[Dict[str, Any]]) -> None:
        for o in orders:
            self._order_rows.append({
                "timestamp": ts.isoformat() if isinstance(ts, datetime) else ts,
                "symbol": o.get("symbol"),
                "side": o.get("side"),
                "qty": o.get("qty"),
                "notional": o.get("notional"),
                "reason": o.get("reason"),
            })

    def log_fills(self, fills: List[Dict[str, Any]]) -> None:
        for f in fills:
            self._fill_rows.append({
                "fill_date": f.get("fill_date").isoformat() if isinstance(f.get("fill_date"), datetime) else f.get("fill_date"),
                "symbol": f.get("symbol"),
                "side": f.get("side"),
                "qty": f.get("qty"),
                "fill_price": f.get("fill_price"),
                "spread_bps": f.get("spread_bps"),
                "slippage_bps": f.get("slippage_bps"),
            })

    def log_diagnostic(self, ts: datetime, metric: str, value: float, detail: str = "") -> None:
        self._diagnostic_rows.append({
            "timestamp": ts.isoformat() if isinstance(ts, datetime) else ts,
            "metric": metric,
            "value": value,
            "detail": detail,
        })

    def write_all(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        for name, rows, keys in [
            ("equity.csv", self._equity_rows, ["timestamp", "equity", "cash", "realized_pl"]),
            ("positions.csv", self._position_rows, ["timestamp", "symbol", "qty", "market_value"]),
            ("orders.csv", self._order_rows, ["timestamp", "symbol", "side", "qty", "notional", "reason"]),
            ("fills.csv", self._fill_rows, ["fill_date", "symbol", "side", "qty", "fill_price", "spread_bps", "slippage_bps", "impact_bps"]),
            ("diagnostics.csv", self._diagnostic_rows, ["timestamp", "metric", "value", "detail"]),
        ]:
            path = os.path.join(self.run_dir, name)
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
