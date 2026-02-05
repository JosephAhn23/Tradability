"""
Level-4 allocation: capital allocation under uncertainty.

Usage:
  python -m tradability.allocation.run --config configs/allocation.yaml
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import Optional

from .config import load_config, AllocationConfig
from .inputs import load_inputs
from .allocator import compute_allocation
from .stress import run_stress_tests
from .report import write_report


def run_allocation(
    config: AllocationConfig,
    run_dir: str,
    feasibility_run_dir: Optional[str] = None,
    regime_fragile_overrides: Optional[dict] = None,
    portfolio_drawdown_pct: Optional[float] = None,
) -> None:
    """Run allocation; write allocations.csv, throttles.csv, stress_results.csv, report.md."""

    inputs = load_inputs(
        config,
        feasibility_run_dir=feasibility_run_dir,
        regime_fragile_overrides=regime_fragile_overrides,
    )
    result = compute_allocation(inputs, config, portfolio_drawdown_pct=portfolio_drawdown_pct)

    os.makedirs(run_dir, exist_ok=True)

    # allocations.csv: strategy, weight, amount, status (throttle/shutdown), reason
    import csv
    alloc_path = os.path.join(run_dir, "allocations.csv")
    with open(alloc_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "weight", "amount", "throttle", "shutdown", "reason"])
        for spec in config.strategies:
            name = spec.name
            w.writerow([
                name,
                result.weights.get(name, 0),
                result.amounts.get(name, 0),
                result.throttle.get(name, False),
                result.shutdown.get(name, False),
                result.reasons.get(name, ""),
            ])

    # throttles.csv: strategy, reason, magnitude
    thr_path = os.path.join(run_dir, "throttles.csv")
    with open(thr_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "reason", "magnitude"])
        for r in result.throttle_records:
            w.writerow([r.strategy_id, r.reason, r.magnitude])

    # stress_results.csv
    stress_df = run_stress_tests(inputs, config, result)
    stress_path = os.path.join(run_dir, "stress_results.csv")
    stress_df.to_csv(stress_path, index=False)

    # report.md
    write_report(run_dir, config, result, stress_df=stress_df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Level-4 capital allocation under uncertainty (policy-based)"
    )
    parser.add_argument("--config", default="configs/allocation.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--feasibility-dir", default=None, help="Feasibility run for net_edge/capacity")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_REPO_ROOT, config_path)
    if not os.path.isfile(config_path):
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or os.path.join(_REPO_ROOT, "runs", "allocation", run_id)

    run_allocation(config, run_dir, feasibility_run_dir=args.feasibility_dir)
    print(f"Allocation run written to: {run_dir}")


if __name__ == "__main__":
    main()
