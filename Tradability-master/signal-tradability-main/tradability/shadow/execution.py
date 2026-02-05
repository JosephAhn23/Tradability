"""
Position diff to orders. No broker; orders are simulated.
"""

from typing import Any, Dict, List, Tuple


def current_positions_to_dollars(
    positions_shares: Dict[str, float],
    prices: Dict[str, float],
) -> Dict[str, float]:
    """Current position in dollars by symbol."""
    return {
        sym: qty * prices.get(sym, 0) for sym, qty in positions_shares.items()
        if prices.get(sym, 0) > 0
    }


def orders_from_targets(
    target_dollars: List[Tuple[str, float]],
    current_positions_dollars: Dict[str, float],
    prices: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Compute orders (in shares) to move from current to target.
    Returns list of {symbol, side, qty, notional, reason}.
    """
    target_d = {sym: amt for sym, amt in target_dollars}
    all_syms = set(target_d.keys()) | set(current_positions_dollars.keys())
    orders = []
    for sym in all_syms:
        tgt = target_d.get(sym, 0.0)
        cur = current_positions_dollars.get(sym, 0.0)
        diff_d = tgt - cur
        px = prices.get(sym)
        if px is None or px <= 0:
            continue
        qty = diff_d / px
        if abs(qty) < 1e-9:
            continue
        side = "buy" if qty > 0 else "sell"
        orders.append({
            "symbol": sym,
            "side": side,
            "qty": abs(qty),
            "notional": abs(diff_d),
            "reason": "rebalance",
        })
    return orders
