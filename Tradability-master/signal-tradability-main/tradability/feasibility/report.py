"""
Report: CSVs, report.md, and matplotlib plots.
"""

import os
from typing import Any, Dict, List

import pandas as pd


def write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


def write_report_md(
    run_dir: str,
    gross_edge_bps: float,
    info: Dict[str, Any],
    cost_model: Any,
    zero_boundary: pd.DataFrame,
    assumptions: List[str],
    limitations: List[str],
) -> None:
    path = os.path.join(run_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Alpha Feasibility Bounds — Report\n\n")
        f.write("## Assumptions\n\n")
        for a in assumptions:
            f.write(f"- {a}\n")
        f.write("\n## Gross edge proxy\n\n")
        f.write(f"- Gross edge bound (bps, annualized): **{gross_edge_bps:.2f}**\n")
        f.write(f"- Type: {info.get('type', 'ic')}\n")
        f.write(f"- IC: {info.get('ic', 'N/A')}\n")
        f.write(f"- Sample count: {info.get('sample_count', 'N/A')}\n")
        f.write(f"- IC-to-return scale (conservative): {info.get('scale', 'N/A')}\n")
        f.write("\n## Cost model\n\n")
        f.write(f"- Fee: {cost_model.fee_bps} bps × turnover\n")
        f.write(f"- Spread: {cost_model.spread_bps} bps × turnover\n")
        f.write(f"- Slippage: {cost_model.slippage_bps_per_turnover} bps × turnover\n")
        f.write(f"- Delay: {cost_model.delay_bps} bps × turnover\n")
        f.write(f"- Impact: {cost_model.impact_type}, k={cost_model.impact_k}\n")
        f.write("\n## Zero-alpha boundary (where net edge crosses 0)\n\n")
        f.write(zero_boundary.to_string(index=False))
        f.write("\n\n## Limitations\n\n")
        for L in limitations:
            f.write(f"- {L}\n")
        f.write("\n## Where alpha must disappear\n\n")
        f.write("Net edge bound crosses zero at the AUM/turnover combinations above. ")
        f.write("Beyond that boundary, expected net edge is non-positive under these assumptions.\n")


def plot_net_edge_vs_aum(
    surface: pd.DataFrame,
    turnover_levels: List[float],
    run_dir: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for t in turnover_levels:
        sub = surface.loc[surface["turnover"] == t]
        if sub.empty:
            continue
        sub = sub.sort_values("aum")
        ax.plot(sub["aum"] / 1e6, sub["net_edge_bps"], label=f"Turnover {t:.1f}x")
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("AUM ($M)")
    ax.set_ylabel("Net edge (bps)")
    ax.legend()
    ax.set_title("Net edge vs AUM by turnover level")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "net_edge_vs_aum.png"), dpi=150)
    plt.close(fig)


def plot_net_edge_vs_turnover(
    surface: pd.DataFrame,
    aum_levels: List[float],
    run_dir: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for a in aum_levels:
        sub = surface.loc[surface["aum"] == a]
        if sub.empty:
            continue
        sub = sub.sort_values("turnover")
        ax.plot(sub["turnover"], sub["net_edge_bps"], label=f"AUM ${a/1e6:.1f}M")
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Turnover (annual)")
    ax.set_ylabel("Net edge (bps)")
    ax.legend()
    ax.set_title("Net edge vs turnover by AUM level")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "net_edge_vs_turnover.png"), dpi=150)
    plt.close(fig)


def plot_zero_alpha_boundary(zero_boundary: pd.DataFrame, run_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = zero_boundary.dropna(subset=["turnover_at_zero"]).sort_values("aum")
    if df.empty:
        return
    fig, ax = plt.subplots()
    ax.plot(df["aum"] / 1e6, df["turnover_at_zero"], marker="o", markersize=4)
    ax.set_xlabel("AUM ($M)")
    ax.set_ylabel("Turnover at zero net edge")
    ax.set_title("Zero-alpha boundary (turnover at which net edge = 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "zero_alpha_boundary.png"), dpi=150)
    plt.close(fig)


def run_report(
    run_dir: str,
    surface: pd.DataFrame,
    zero_boundary: pd.DataFrame,
    regime_table: pd.DataFrame,
    gross_edge_bps: float,
    info: Dict[str, Any],
    cost_model: Any,
    aum_grid: List[float],
    turnover_grid: List[float],
) -> None:
    os.makedirs(run_dir, exist_ok=True)
    write_csv(surface, os.path.join(run_dir, "net_edge_surface.csv"))
    write_csv(zero_boundary, os.path.join(run_dir, "zero_alpha_boundary.csv"))
    write_csv(regime_table, os.path.join(run_dir, "regime_table.csv"))

    assumptions = [
        "Gross edge proxy uses only past information (signal at D, return D to D+1).",
        "IC-to-return scaling is conservative (scale < 1).",
        "Costs: spread, fee, slippage, delay scale with turnover; impact scales with sqrt(notional/ADV).",
        "No lookahead in signal or return alignment.",
    ]
    limitations = [
        "Gross edge is an upper-bound proxy, not a forecast.",
        "Impact model is simplified (single sqrt form).",
        "Regime buckets depend on in-sample vol/liquidity quantiles.",
    ]
    write_report_md(run_dir, gross_edge_bps, info, cost_model, zero_boundary, assumptions, limitations)

    # Plots: sample a few levels
    turnover_levels = [turnover_grid[i] for i in [0, len(turnover_grid) // 2, -1] if 0 <= i < len(turnover_grid)]
    if not turnover_levels:
        turnover_levels = list(surface["turnover"].unique())[:3]
    aum_levels = [aum_grid[i] for i in [0, len(aum_grid) // 2, -1] if 0 <= i < len(aum_grid)]
    if not aum_levels:
        aum_levels = list(surface["aum"].unique())[:3]
    plot_net_edge_vs_aum(surface, turnover_levels, run_dir)
    plot_net_edge_vs_turnover(surface, aum_levels, run_dir)
    plot_zero_alpha_boundary(zero_boundary, run_dir)
