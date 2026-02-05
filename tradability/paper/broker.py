"""
Alpaca paper trading client wrapper.

Submit orders, fetch fills, positions. Uses environment variables for API keys.
"""

import os
from datetime import datetime
from typing import List, Dict, Optional, Any

# Alpaca SDK
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus


def _get_client(paper: bool = True) -> TradingClient:
    """Build Alpaca client from env. Keys must be paper keys for paper=True."""
    key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (or ALPACA_API_KEY / ALPACA_SECRET_KEY) for paper trading."
        )
    return TradingClient(key, secret, paper=paper)


class AlpacaBroker:
    """
    Alpaca client wrapper: submit orders, fetch positions and fills.
    Paper-only by default.
    """

    def __init__(self, paper: bool = True):
        self._paper = paper
        self._client: Optional[TradingClient] = None

    @property
    def client(self) -> TradingClient:
        if self._client is None:
            self._client = _get_client(paper=self._paper)
        return self._client

    def get_account(self) -> Dict[str, Any]:
        """Account summary: equity, cash, buying_power."""
        acc = self.client.get_account()
        return {
            "equity": float(acc.equity or 0),
            "cash": float(acc.cash or 0),
            "buying_power": float(acc.buying_power or 0),
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        """Current positions: symbol, qty, side, market_value, cost_basis, etc."""
        positions = self.client.get_all_positions()
        out = []
        for p in positions:
            out.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": str(p.side) if p.side else "long",
                "market_value": float(p.market_value or 0),
                "cost_basis": float(p.cost_basis or 0),
                "unrealized_pl": float(p.unrealized_pl or 0),
                "current_price": float(p.current_price or 0),
            })
        return out

    def get_orders(
        self,
        status: Optional[str] = None,
        after: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Orders (optionally filtered by status)."""
        status_enum = None
        if status:
            status_enum = getattr(QueryOrderStatus, status.upper(), None) or QueryOrderStatus.ALL
        req = GetOrdersRequest(status=status_enum or QueryOrderStatus.ALL, limit=limit)
        if after:
            req.after = after
        orders = self.client.get_orders(filter=req)
        return [
            {
                "id": o.id,
                "symbol": o.symbol,
                "qty": float(o.qty or 0),
                "filled_qty": float(o.filled_qty or 0),
                "side": str(o.side),
                "type": str(o.type),
                "status": str(o.status),
                "filled_avg_price": float(o.filled_avg_price or 0),
                "submitted_at": o.submitted_at.isoformat() if o.submitted_at else None,
                "filled_at": o.filled_at.isoformat() if o.filled_at else None,
            }
            for o in orders
        ]

    def submit_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        """Submit market order. Returns order info dict."""
        side_enum = OrderSide.BUY if str(side).lower() == "buy" else OrderSide.SELL
        tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC
        req = MarketOrderRequest(symbol=symbol, qty=qty, side=side_enum, time_in_force=tif)
        order = self.client.submit_order(order_data=req)
        return {
            "id": order.id,
            "symbol": order.symbol,
            "qty": float(order.qty or 0),
            "side": str(order.side),
            "type": str(order.type),
            "status": str(order.status),
        }

    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float,
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        """Submit limit order."""
        side_enum = OrderSide.BUY if str(side).lower() == "buy" else OrderSide.SELL
        tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC
        req = LimitOrderRequest(
            symbol=symbol, qty=qty, side=side_enum, time_in_force=tif, limit_price=limit_price
        )
        order = self.client.submit_order(order_data=req)
        return {
            "id": order.id,
            "symbol": order.symbol,
            "qty": float(order.qty or 0),
            "side": str(order.side),
            "type": str(order.type),
            "status": str(order.status),
            "limit_price": limit_price,
        }

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Unified submit: order_type in ('market', 'limit')."""
        qty = round(qty, 6)
        if order_type.lower() == "limit" and limit_price is not None:
            return self.submit_limit_order(symbol, side, qty, limit_price)
        return self.submit_market_order(symbol, side, qty)
