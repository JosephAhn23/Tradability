"""
Diagnostics: slippage proxy, turnover, fill latency, missing fills.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import csv
import os


class DiagnosticsLogger:
    """
    Log execution diagnostics: orders, fills, slippage proxy, turnover, latency.
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self._orders: List[Dict[str, Any]] = []
        self._fills: List[Dict[str, Any]] = []
        self._diagnostics: List[Dict[str, Any]] = []

    def log_order(self, order: Dict[str, Any], submitted_at: Optional[datetime] = None) -> None:
        self._orders.append({
            "submitted_at": (submitted_at or datetime.utcnow()).isoformat(),
            "id": order.get("id"),
            "symbol": order.get("symbol"),
            "side": order.get("side"),
            "qty": order.get("qty"),
            "status": order.get("status"),
            "type": order.get("type"),
        })

    def log_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        filled_qty: float,
        filled_avg_price: float,
        filled_at: Optional[datetime] = None,
        expected_price: Optional[float] = None,
    ) -> None:
        """Record a fill. If expected_price set, slippage proxy = (filled_avg - expected) / expected."""
        row = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "filled_qty": filled_qty,
            "filled_avg_price": filled_avg_price,
            "filled_at": (filled_at or datetime.utcnow()).isoformat(),
        }
        if expected_price is not None and expected_price != 0:
            row["expected_price"] = expected_price
            row["slippage_bps"] = ((filled_avg_price - expected_price) / expected_price) * 10_000
        self._fills.append(row)

    def log_diagnostic(
        self,
        ts: datetime,
        metric: str,
        value: float,
        detail: Optional[str] = None,
    ) -> None:
        self._diagnostics.append({
            "timestamp": ts.isoformat(),
            "metric": metric,
            "value": value,
            "detail": detail or "",
        })

    def orders_from_broker(self, orders: List[Dict[str, Any]]) -> None:
        """Log orders from broker get_orders (includes fills)."""
        for o in orders:
            self._orders.append({
                "submitted_at": o.get("submitted_at"),
                "id": o.get("id"),
                "symbol": o.get("symbol"),
                "side": o.get("side"),
                "qty": o.get("qty"),
                "filled_qty": o.get("filled_qty"),
                "status": o.get("status"),
                "filled_avg_price": o.get("filled_avg_price"),
                "filled_at": o.get("filled_at"),
            })
            if o.get("filled_qty") and float(o.get("filled_qty", 0)) > 0:
                self._fills.append({
                    "order_id": o.get("id"),
                    "symbol": o.get("symbol"),
                    "side": o.get("side"),
                    "filled_qty": o.get("filled_qty"),
                    "filled_avg_price": o.get("filled_avg_price"),
                    "filled_at": o.get("filled_at"),
                })

    def write_orders_csv(self) -> str:
        path = os.path.join(self.run_dir, "orders.csv")
        keys = ["submitted_at", "id", "symbol", "side", "qty", "filled_qty", "status", "filled_avg_price", "filled_at"] if self._orders else ["submitted_at", "id", "symbol", "side", "qty", "status"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(self._orders)
        return path

    def write_fills_csv(self) -> str:
        path = os.path.join(self.run_dir, "fills.csv")
        keys = list(self._fills[0].keys()) if self._fills else ["order_id", "symbol", "side", "filled_qty", "filled_avg_price", "filled_at"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(self._fills)
        return path

    def write_diagnostics_csv(self) -> str:
        path = os.path.join(self.run_dir, "diagnostics.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "metric", "value", "detail"])
            w.writeheader()
            w.writerows(self._diagnostics)
        return path

    def flush(self) -> None:
        self.write_orders_csv()
        self.write_fills_csv()
        self.write_diagnostics_csv()
