"""
Position diff → orders (rebalance logic).

Compares current positions to target positions and generates orders.
Idempotent: safe to re-run (only submits needed trades).
"""

from typing import List, Tuple, Dict, Any, Optional

from .broker import AlpacaBroker


def current_position_map(positions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Map symbol -> current share quantity (positive = long, negative = short)."""
    out = {}
    for p in positions:
        qty = float(p.get("qty", 0))
        side = (p.get("side") or "long").lower()
        if "short" in side or qty < 0:
            out[p["symbol"]] = -abs(qty)
        else:
            out[p["symbol"]] = abs(qty)
    return out


def target_share_map(
    target_dollars: List[Tuple[str, float]],
    prices: Dict[str, float],
) -> Dict[str, float]:
    """Convert target dollar values to target shares using prices."""
    out = {}
    for sym, dollar in target_dollars:
        if dollar == 0:
            out[sym] = 0.0
            continue
        px = prices.get(sym)
        if px is None or px <= 0:
            continue
        out[sym] = dollar / px
    return out


def compute_orders_from_targets(
    broker: AlpacaBroker,
    target_dollars: List[Tuple[str, float]],
    prices: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute and optionally submit orders to move from current positions to targets.

    If prices not provided, fetches from broker positions (current_price) for held names;
    other symbols need to be provided or we skip (no quote fetch in this minimal impl).

    Returns list of order info dicts (submitted orders). Does NOT submit if broker is None
    or dry_run; then returns planned orders only.
    """
    positions = broker.get_positions()
    current = current_position_map(positions)

    if prices is None:
        prices = {p["symbol"]: p["current_price"] for p in positions if p.get("current_price")}
    if not prices:
        # Cannot size orders without prices
        return []

    targets = target_share_map(target_dollars, prices)
    all_symbols = set(current.keys()) | set(targets.keys())
    orders = []
    for sym in all_symbols:
        cur_qty = current.get(sym, 0.0)
        tgt_qty = targets.get(sym, 0.0)
        diff = tgt_qty - cur_qty
        if abs(diff) < 1e-6:
            continue
        side = "buy" if diff > 0 else "sell"
        qty = abs(diff)
        orders.append({
            "symbol": sym,
            "side": side,
            "qty": qty,
            "reason": "rebalance",
        })
    return orders


def submit_orders(
    broker: AlpacaBroker,
    orders: List[Dict[str, Any]],
    order_type: str = "market",
    limit_prices: Optional[Dict[str, float]] = None,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """
    Submit orders to broker. Returns list of order response dicts.
    If dry_run=True, no submission; returns same structure with status='dry_run'.
    """
    limit_prices = limit_prices or {}
    submitted = []
    for o in orders:
        sym = o["symbol"]
        side = o["side"]
        qty = o["qty"]
        if dry_run:
            submitted.append({
                "symbol": sym,
                "side": side,
                "qty": qty,
                "status": "dry_run",
                "id": None,
            })
            continue
        limit_price = limit_prices.get(sym)
        resp = broker.submit_order(
            symbol=sym,
            side=side,
            qty=qty,
            order_type=order_type,
            limit_price=limit_price,
        )
        submitted.append(resp)
    return submitted
