"""
Paper trading main loop.

Usage:
  python -m tradability.paper.run --config configs/paper.yaml

Run from the signal-tradability-main directory (repo root for imports).
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure repo root is on path when run as module
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml


def _load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _latest_prices(tickers: list) -> Dict[str, float]:
    """Fetch latest close price for each ticker (yfinance, no paid data)."""
    try:
        import yfinance as yf
    except ImportError:
        return {}
    out = {}
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="5d")
            if hist is not None and not hist.empty and "Close" in hist.columns:
                out[t] = float(hist["Close"].iloc[-1])
        except Exception:
            continue
    return out


def _has_alpaca_keys() -> bool:
    return bool(
        os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    ) and bool(
        os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    )


def run_once(
    config: Dict[str, Any],
    run_dir: str,
    as_of_date: Optional[datetime] = None,
) -> None:
    """
    One rebalance step: generate signals -> target positions -> orders -> submit -> ledger/diagnostics.
    """
    from .broker import AlpacaBroker
    from .signals import generate_target_positions
    from .execution import compute_orders_from_targets, submit_orders
    from .ledger import Ledger
    from .diagnostics import DiagnosticsLogger

    as_of_date = as_of_date or datetime.utcnow()
    tickers = config.get("tickers", ["SPY"])
    signal_name = config.get("signal_name", "momentum_12_1")
    order_type = config.get("order_type", "market")
    position_sizing = config.get("position_sizing", "equal_weight")
    fixed_dollar = config.get("fixed_dollar_per_name")
    max_position_pct = float(config.get("max_position_pct", 0.25))
    dry_run = config.get("dry_run", True)

    # Broker only required when not dry_run or when we need live positions/equity
    broker = None
    account = {"equity": config.get("initial_equity", 100_000.0), "cash": 0.0}
    if _has_alpaca_keys():
        broker = AlpacaBroker(paper=True)
        account = broker.get_account()
    equity = account.get("equity") or 100_000.0

    # 1) Target positions (no lookahead)
    targets = generate_target_positions(
        tickers=tickers,
        signal_name=signal_name,
        as_of_date=as_of_date,
        position_sizing=position_sizing,
        fixed_dollar_per_name=fixed_dollar,
        max_position_pct=max_position_pct,
        equity=equity,
    )
    if not targets:
        return

    # 2) Current prices for order sizing
    symbols = [t[0] for t in targets]
    prices = _latest_prices(symbols)
    if not prices and broker:
        positions = broker.get_positions()
        for p in positions:
            sym = p.get("symbol")
            if p.get("current_price"):
                prices[sym] = float(p["current_price"])
    if not prices:
        return

    # 3) Orders (need broker for current positions; if no broker assume flat)
    if broker is None:
        # No broker: assume we have no positions, so targets become orders
        from .execution import target_share_map
        target_shares = target_share_map(targets, prices)
        orders = [
            {"symbol": s, "side": "buy" if q > 0 else "sell", "qty": abs(q), "reason": "rebalance"}
            for s, q in target_shares.items() if abs(q) >= 1e-6
        ]
    else:
        orders = compute_orders_from_targets(broker, targets, prices=prices)
    if not orders:
        return

    # 4) Submit (or dry run). Need broker to submit; if no broker only log planned orders
    if broker is not None:
        submitted = submit_orders(
            broker, orders, order_type=order_type, dry_run=dry_run
        )
    else:
        submitted = [
            {**o, "status": "dry_run", "id": None} for o in orders
        ]

    # 5) Ledger snapshot
    positions = broker.get_positions() if broker else []
    ledger = Ledger(run_dir)
    ledger.snapshot(
        ts=as_of_date,
        equity=account.get("equity", 0),
        cash=account.get("cash", 0),
        positions=positions,
    )
    ledger.flush()

    # 6) Diagnostics
    diag = DiagnosticsLogger(run_dir)
    for s in submitted:
        diag.log_order(s)
    if broker and not dry_run:
        broker_orders = broker.get_orders(limit=50)
        diag.orders_from_broker(broker_orders)
    diag.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper trading (broker-simulated) pipeline")
    parser.add_argument("--config", default="configs/paper.yaml", help="Config YAML path")
    parser.add_argument("--run-dir", default=None, help="Override run output directory")
    parser.add_argument("--once", action="store_true", help="Run one rebalance then exit")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        # Relative to repo root
        config_path = os.path.join(_REPO_ROOT, config_path)
    if not os.path.isfile(config_path):
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = _load_config(config_path)
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or os.path.join(_REPO_ROOT, "paper_runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    if args.once:
        run_once(config, run_dir)
        # Write minimal summary
        summary_path = os.path.join(run_dir, "summary.md")
        with open(summary_path, "w") as f:
            f.write("# Paper run summary\n\n")
            f.write(f"Run ID: {run_id}\n")
            f.write(f"Config: {args.config}\n")
            f.write(f"Dry run: {config.get('dry_run', True)}\n\n")
            f.write("## Where paper trading diverged from backtests\n\n")
            f.write("- (Fill in: missed fills, delayed execution, higher turnover, worse realized prices, regime-specific failures.)\n")
        print(f"Run dir: {run_dir}")
        return

    # Scheduled loop: daily or weekly
    freq = config.get("rebalance_frequency", "daily")
    delta = timedelta(days=7) if "week" in freq.lower() else timedelta(days=1)
    next_ts = datetime.utcnow()
    while True:
        run_once(config, run_dir, as_of_date=next_ts)
        next_ts += delta
        try:
            import time
            time.sleep(max(1, delta.total_seconds()))
        except KeyboardInterrupt:
            break

    summary_path = os.path.join(run_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("# Paper run summary\n\n")
        f.write(f"Run ID: {run_id}\n")
        f.write("## Where paper trading diverged from backtests\n\n")
        f.write("- (Fill in after run.)\n")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
