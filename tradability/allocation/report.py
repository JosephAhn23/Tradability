"""
Explainable allocation decisions: report.md with required sections.
"""

import os
from typing import List

from .config import AllocationConfig
from .allocator import AllocationResult
from .policies import ThrottleRecord, ShutdownRecord


def write_report(
    run_dir: str,
    config: AllocationConfig,
    result: AllocationResult,
    stress_df=None,
) -> None:
    """Write report.md with: why conservative, what we assumed wrong, under-allocated, what causes shutdown."""
    path = os.path.join(run_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Capital allocation under uncertainty — Report\n\n")
        f.write("Policy-based allocation. No return maximization; survivability under model error.\n\n")

        f.write("## Why capital was allocated conservatively\n\n")
        f.write("- Base weight is proportional to **feasible capacity × regime_confidence ÷ uncertainty_penalty**, not to expected return.\n")
        f.write("- Higher uncertainty score or low regime confidence reduces allocation.\n")
        f.write("- Strategies in the same correlation group have combined exposure capped.\n")
        f.write("- We prefer under-allocation to overconfidence; min weight threshold and shutdown rules enforce this.\n\n")

        f.write("## What we assumed might be wrong\n\n")
        f.write("- **Net edge estimates** may be noisy or decay; feasibility bounds may be optimistic.\n")
        f.write("- **Regime confidence** is estimated from historical regimes; future regimes may differ.\n")
        f.write("- **Correlations** may spike in stress; we apply correlation penalty and stress-test correlation → 1.\n")
        f.write("- **Liquidity** may disappear when needed; we stress-test capacity halved.\n")
        f.write("- **Parameter instability** and backtest vs forward divergence are penalized via uncertainty and throttle rules.\n\n")

        f.write("## Which strategies were intentionally under-allocated\n\n")
        for spec in config.strategies:
            name = spec.name
            thr = [r for r in result.throttle_records if r.strategy_id == name]
            if result.throttle.get(name) or thr:
                f.write(f"- **{name}**: ")
                f.write("; ".join(r.reason + f" (magnitude {r.magnitude})" for r in thr))
                f.write("\n")
        if not any(result.throttle.get(s.name) for s in config.strategies):
            f.write("- None; no throttles applied in this run.\n")
        f.write("\n")

        f.write("## What would cause immediate shutdown\n\n")
        f.write("A strategy is **halted** (zero allocation) if any of:\n")
        f.write("- Feasibility bound (net edge) ≤ 0\n")
        f.write(f"- Regime confidence < {config.min_regime_confidence}\n")
        f.write(f"- Drawdown exceeds {config.max_drawdown_pct:.0%}\n")
        f.write(f"- Realized vs expected divergence ≥ {config.divergence_shutdown_bps} bps\n")
        f.write("- Raw weight below min weight threshold\n\n")
        if result.shutdown_records:
            f.write("**Shutdowns in this run:**\n")
            for r in result.shutdown_records:
                f.write(f"- {r.strategy_id}: {r.reason}\n")
        f.write("\n")

        f.write("## Allocation summary\n\n")
        f.write("| Strategy | Weight | Amount | Throttle | Shutdown | Reason |\n")
        f.write("|----------|--------|--------|----------|----------|--------|\n")
        for spec in config.strategies:
            name = spec.name
            f.write(f"| {name} | {result.weights.get(name, 0):.4f} | {result.amounts.get(name, 0):.0f} | ")
            f.write(f"{result.throttle.get(name, False)} | {result.shutdown.get(name, False)} | {result.reasons.get(name, '')} |\n")
        f.write("\n")

        if stress_df is not None and not stress_df.empty:
            f.write("## Stress test summary\n\n")
            f.write("How allocations change under: 2× estimation error, correlation→1, liquidity shock.\n\n")
            f.write(stress_df.to_string(index=False))
            f.write("\n")
