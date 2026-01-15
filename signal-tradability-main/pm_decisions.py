"""
PM Decision Framework: What Would You Do Differently?

Every serious reviewer asks silently:
"If I had capital, what would I do differently after reading this?"

This module answers that explicitly.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from signals import get_signal
from data_utils import load_price_data, compute_forward_returns, align_signals_and_returns
from decay_analysis import compute_returns, compute_performance_metrics
from tradability_analysis import analyze_tradability, compute_positions_from_returns
from formal_definitions import compute_statistical_edge, compute_economic_edge, identify_edge_mismatch


@dataclass
class PMDecision:
    """A specific decision recommendation for portfolio managers."""
    decision: str
    rationale: str
    evidence: Dict
    action_items: List[str]
    priority: str  # 'high', 'medium', 'low'


def analyze_signal_for_pm_decision(signal_name: str,
                                   ticker: str = 'SPY',
                                   start_date: datetime = datetime(2000, 1, 1),
                                   end_date: datetime = datetime(2020, 12, 31),
                                   commission_per_trade: float = 0.005,
                                   half_spread: float = 0.001) -> List[PMDecision]:
    """
    Analyze a signal and generate PM decision recommendations.
    
    Returns:
        List of PMDecision objects
    """
    # Load data
    prices, volumes = load_price_data(ticker, start_date, end_date)
    forward_returns = compute_forward_returns(prices)
    volatility = prices.pct_change().rolling(20).std() * np.sqrt(252)
    
    # Compute signal
    signal_def = get_signal(signal_name)
    signal_values = signal_def.compute(prices, **signal_def.default_params())
    aligned_signals, aligned_returns = align_signals_and_returns(signal_values, forward_returns)
    gross_returns = compute_returns(aligned_signals, aligned_returns, quantile=0.5)
    positions = compute_positions_from_returns(gross_returns, aligned_signals)
    
    # Analyze tradability
    tradability = analyze_tradability(
        gross_returns=gross_returns,
        signals=aligned_signals,
        volatility=volatility,
        volumes=volumes,
        prices=prices,
        commission_per_trade=commission_per_trade,
        half_spread=half_spread,
        periods_per_year=252
    )
    
    # Compute edges
    stat_edge = compute_statistical_edge(aligned_signals, aligned_returns)
    net_returns_series = gross_returns - (tradability.cost_drag / 252)
    econ_edge = compute_economic_edge(
        net_returns_series,
        gross_returns,
        tradability.cost_drag,
        tradability.break_even_cost,
        tradability.max_viable_capacity or 0
    )
    
    mismatch = identify_edge_mismatch(stat_edge, econ_edge)
    
    decisions = []
    
    # Decision 1: Size down high-turnover signals
    if tradability.annual_turnover > 3.0:
        decisions.append(PMDecision(
            decision="Size down or avoid high-turnover signals",
            rationale=(
                f"Signal has annual turnover {tradability.annual_turnover:.1f}x, which requires "
                f"transaction costs <{tradability.break_even_cost*100:.2f}% per trade to preserve edge. "
                f"Your execution costs ({commission_per_trade*100:.2f}% + {half_spread*100:.2f}% = "
                f"{(commission_per_trade + half_spread)*100:.2f}% per trade) exceed break-even cost."
            ),
            evidence={
                'annual_turnover': tradability.annual_turnover,
                'break_even_cost': tradability.break_even_cost,
                'your_costs': commission_per_trade + half_spread,
                'net_sharpe': econ_edge.net_sharpe,
            },
            action_items=[
                f"Reduce position size by {(1 - tradability.break_even_cost / (commission_per_trade + half_spread)) * 100:.0f}% to account for cost drag",
                "Consider longer-horizon signals (lower turnover)",
                "Avoid short-horizon mean reversion (high turnover, low break-even cost)",
            ],
            priority='high'
        ))
    
    # Decision 2: Require stronger validation
    if stat_edge.has_statistical_edge and not econ_edge.has_economic_edge:
        decisions.append(PMDecision(
            decision="Require net Sharpe >0.5 after realistic costs",
            rationale=(
                f"Signal has statistical edge (hit rate {stat_edge.hit_rate:.1%}) but no economic edge "
                f"(net Sharpe {econ_edge.net_sharpe:.3f}). Statistical edge without economic edge is not tradable."
            ),
            evidence={
                'statistical_edge': stat_edge.has_statistical_edge,
                'economic_edge': econ_edge.has_economic_edge,
                'hit_rate': stat_edge.hit_rate,
                'net_sharpe': econ_edge.net_sharpe,
                'gross_sharpe': tradability.gross_metrics.sharpe_ratio if tradability.gross_metrics else 0,
            },
            action_items=[
                "Don't trust gross Sharpe alone - always check net Sharpe",
                "Require break-even cost > your execution costs",
                "Reject signals with net Sharpe <0.5 after realistic costs",
            ],
            priority='high'
        ))
    
    # Decision 3: Trade less frequently
    if tradability.annual_turnover > 2.0 and econ_edge.net_sharpe < 0.5:
        decisions.append(PMDecision(
            decision="Reduce trading frequency to preserve economic edge",
            rationale=(
                f"High turnover ({tradability.annual_turnover:.1f}x) creates cost drag "
                f"({tradability.cost_drag*100:.2f}% annualized) that overwhelms residual alpha. "
                f"Reducing turnover can preserve economic edge."
            ),
            evidence={
                'annual_turnover': tradability.annual_turnover,
                'cost_drag': tradability.cost_drag,
                'net_sharpe': econ_edge.net_sharpe,
            },
            action_items=[
                "Consider longer-horizon signals (lower turnover)",
                "Reduce rebalancing frequency",
                "Use wider thresholds for position changes",
            ],
            priority='medium'
        ))
    
    # Decision 4: Stop earlier
    if not econ_edge.has_economic_edge:
        decisions.append(PMDecision(
            decision="Do not trade signals with net Sharpe <0",
            rationale=(
                f"Signal has net Sharpe {econ_edge.net_sharpe:.3f} after costs. "
                f"Even if statistical edge exists (hit rate {stat_edge.hit_rate:.1%}), "
                f"economic edge is eliminated by costs."
            ),
            evidence={
                'net_sharpe': econ_edge.net_sharpe,
                'statistical_edge': stat_edge.has_statistical_edge,
                'hit_rate': stat_edge.hit_rate,
            },
            action_items=[
                "Stop allocation if net Sharpe <0 at realistic costs",
                "Don't trade based on hit rate alone - check net returns",
                "Reject signals where costs exceed gross returns",
            ],
            priority='high'
        ))
    
    # Decision 5: Model capacity explicitly
    if tradability.max_viable_capacity and tradability.max_viable_capacity < 100e6:
        decisions.append(PMDecision(
            decision="Scale down if capacity constraints bind",
            rationale=(
                f"Signal has maximum viable capacity ${tradability.max_viable_capacity/1e6:.1f}M. "
                f"If your AUM exceeds this, market impact will overwhelm edge."
            ),
            evidence={
                'max_viable_capacity': tradability.max_viable_capacity,
                'capacity_pct_of_volume': tradability.max_viable_capacity / (volumes.mean() if volumes is not None else 1) * 100,
            },
            action_items=[
                f"Limit AUM to ${tradability.max_viable_capacity/1e6:.1f}M for this signal",
                "Use conservative participation rates (1% of volume)",
                "Monitor market impact as you scale",
            ],
            priority='medium'
        ))
    
    return decisions


def generate_pm_summary_report(signal_names: List[str],
                                ticker: str = 'SPY',
                                start_date: datetime = datetime(2000, 1, 1),
                                end_date: datetime = datetime(2020, 12, 31),
                                commission_per_trade: float = 0.005,
                                half_spread: float = 0.001) -> str:
    """
    Generate a summary report for portfolio managers.
    
    Returns:
        Formatted string report
    """
    print("=" * 80)
    print("PM DECISION FRAMEWORK")
    print("What Would You Do Differently With Capital?")
    print("=" * 80)
    
    all_decisions = []
    
    for signal_name in signal_names:
        print(f"\nAnalyzing {signal_name}...")
        decisions = analyze_signal_for_pm_decision(
            signal_name, ticker, start_date, end_date,
            commission_per_trade, half_spread
        )
        all_decisions.extend(decisions)
    
    # Group by priority
    high_priority = [d for d in all_decisions if d.priority == 'high']
    medium_priority = [d for d in all_decisions if d.priority == 'medium']
    low_priority = [d for d in all_decisions if d.priority == 'low']
    
    # Generate report
    report = []
    report.append("=" * 80)
    report.append("PORTFOLIO MANAGER DECISION SUMMARY")
    report.append("=" * 80)
    report.append("\nQuestion: If I had capital, what would I do differently after reading this?")
    report.append("\n" + "-" * 80)
    
    if high_priority:
        report.append("\n🔴 HIGH PRIORITY DECISIONS:")
        report.append("-" * 80)
        for i, decision in enumerate(high_priority, 1):
            report.append(f"\n{i}. {decision.decision}")
            report.append(f"   Rationale: {decision.rationale}")
            report.append(f"   Action Items:")
            for item in decision.action_items:
                report.append(f"     • {item}")
    
    if medium_priority:
        report.append("\n🟡 MEDIUM PRIORITY DECISIONS:")
        report.append("-" * 80)
        for i, decision in enumerate(medium_priority, 1):
            report.append(f"\n{i}. {decision.decision}")
            report.append(f"   Rationale: {decision.rationale}")
            report.append(f"   Action Items:")
            for item in decision.action_items:
                report.append(f"     • {item}")
    
    if low_priority:
        report.append("\n🟢 LOW PRIORITY DECISIONS:")
        report.append("-" * 80)
        for i, decision in enumerate(low_priority, 1):
            report.append(f"\n{i}. {decision.decision}")
            report.append(f"   Rationale: {decision.rationale}")
    
    report.append("\n" + "=" * 80)
    report.append("KEY TAKEAWAYS:")
    report.append("=" * 80)
    report.append("1. Size down high-turnover signals (>3x annual turnover)")
    report.append("2. Require net Sharpe >0.5 after realistic costs")
    report.append("3. Trade less frequently to preserve economic edge")
    report.append("4. Stop earlier - don't trade if net Sharpe <0")
    report.append("5. Model capacity explicitly before scaling")
    
    report_text = "\n".join(report)
    print(report_text)
    
    return report_text


if __name__ == "__main__":
    # Generate PM summary report
    signals = ['momentum_12_1', 'mean_reversion']
    generate_pm_summary_report(signals)



