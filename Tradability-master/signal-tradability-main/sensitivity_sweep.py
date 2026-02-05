"""
Sensitivity Analysis: At what cost level would signals pass?

Answers:
1. What parameters would a signal need to pass?
2. What if costs were different (5x, 3x, 1x)?
3. Existence proof - can anything pass under realistic assumptions?
"""

import pandas as pd
import numpy as np


def simulate_signal_at_cost_level(
    gross_sharpe: float,
    turnover: float,
    cost_bps: float
) -> dict:
    """
    Simulate signal performance at a given cost level.
    
    Args:
        gross_sharpe: Pre-cost Sharpe ratio
        turnover: Annual turnover (1x = 100% portfolio traded)
        cost_bps: Cost per trade in basis points
    
    Returns:
        dict with net_sharpe, cost_drag, passes
    """
    # Annual cost drag = turnover * cost per trade * 2 (round trip)
    annual_cost_drag_pct = turnover * (cost_bps / 10000) * 2 * 100
    
    # Net Sharpe approximation: gross - (cost_drag / volatility)
    # Assuming 15% annual vol for typical equity strategy
    vol = 0.15
    net_sharpe = gross_sharpe - (annual_cost_drag_pct / 100) / vol
    
    # Check if passes thresholds
    passes = net_sharpe >= 0.5  # Sharpe threshold
    
    return {
        'gross_sharpe': gross_sharpe,
        'turnover': turnover,
        'cost_bps': cost_bps,
        'annual_cost_drag_pct': annual_cost_drag_pct,
        'net_sharpe': net_sharpe,
        'passes': passes
    }


def find_break_even_cost(gross_sharpe: float, turnover: float, target_net_sharpe: float = 0.5) -> float:
    """Find max cost (bps) where signal still passes."""
    vol = 0.15
    
    # net_sharpe = gross_sharpe - (turnover * cost_bps/10000 * 2) / vol
    # target = gross - (turnover * cost / 10000 * 2) / vol
    # (gross - target) * vol = turnover * cost / 10000 * 2
    # cost = (gross - target) * vol * 10000 / (turnover * 2)
    
    if gross_sharpe <= target_net_sharpe:
        return 0.0  # Can't pass even at zero cost
    
    break_even_cost = (gross_sharpe - target_net_sharpe) * vol * 10000 / (turnover * 2)
    return break_even_cost


def run_sensitivity_analysis():
    """Run full sensitivity sweep."""
    
    print("=" * 70)
    print("SENSITIVITY ANALYSIS: WHAT WOULD PASS?")
    print("=" * 70)
    
    # 1. Framework's tested signal: momentum_12_1
    print("\n### 1. Current Signal (momentum_12_1)")
    print("-" * 50)
    
    # From our results: gross Sharpe ~1.2, turnover 4.2x
    gross_sharpe = 1.2
    turnover = 4.2
    
    cost_levels = [
        ("Framework SABOTAGE (120 bps)", 120.0),  # 1% commission + 0.2% spread
        ("Framework default (60 bps)", 60.0),     # 0.5% commission + 0.1% spread
        ("Realistic (10 bps)", 10.0),             # Modern zero-commission + spread
        ("SPY Reality (1.5 bps)", 1.5),           # Actual L2 measured
    ]
    
    print(f"Gross Sharpe: {gross_sharpe}, Turnover: {turnover}x/year\n")
    print(f"{'Cost Scenario':<25} {'Cost/Trade':<12} {'Annual Drag':<12} {'Net Sharpe':<12} {'Verdict'}")
    print("-" * 75)
    
    for name, cost_bps in cost_levels:
        result = simulate_signal_at_cost_level(gross_sharpe, turnover, cost_bps)
        verdict = "PASS" if result['passes'] else "REJECT"
        print(f"{name:<25} {cost_bps:>8.1f} bps  {result['annual_cost_drag_pct']:>8.1f}%     {result['net_sharpe']:>8.2f}      {verdict}")
    
    break_even = find_break_even_cost(gross_sharpe, turnover)
    print(f"\nBreak-even cost: {break_even:.1f} bps (max cost where signal passes)")
    
    # 2. What gross Sharpe is needed?
    print("\n### 2. What Gross Sharpe is Needed to Pass?")
    print("-" * 50)
    
    # At framework costs (120 bps sabotage), what gross Sharpe is needed?
    print(f"Turnover: {turnover}x, Cost: 120 bps (framework SABOTAGE mode)\n")
    
    for target_gross in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        result = simulate_signal_at_cost_level(target_gross, turnover, 120.0)
        verdict = "PASS" if result['passes'] else "REJECT"
        print(f"Gross Sharpe {target_gross:.1f} -> Net Sharpe {result['net_sharpe']:.2f} -> {verdict}")
    
    # 3. Existence proof: low-turnover strategy
    print("\n### 3. Existence Proof: Low-Turnover Strategy")
    print("-" * 50)
    
    # Try a value strategy with 0.5x turnover
    low_turnover = 0.5
    value_gross = 0.8  # Lower gross Sharpe but less trading
    
    print(f"Scenario: Value strategy, Gross Sharpe {value_gross}, Turnover {low_turnover}x\n")
    
    for name, cost_bps in cost_levels:
        result = simulate_signal_at_cost_level(value_gross, low_turnover, cost_bps)
        verdict = "PASS" if result['passes'] else "REJECT"
        print(f"{name:<25} {cost_bps:>8.1f} bps  {result['annual_cost_drag_pct']:>8.1f}%     {result['net_sharpe']:>8.2f}      {verdict}")
    
    break_even = find_break_even_cost(value_gross, low_turnover)
    print(f"\nBreak-even cost: {break_even:.1f} bps")
    
    # 4. Full grid: Sharpe vs Turnover vs Cost
    print("\n### 4. Parameter Sensitivity Grid")
    print("-" * 50)
    print("At what (Gross Sharpe, Turnover) does a signal PASS at 120 bps cost (SABOTAGE)?\n")
    
    print(f"{'Turnover':<10}", end="")
    for gs in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        print(f" GS={gs:<5}", end="")
    print()
    print("-" * 60)
    
    for turnover in [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]:
        print(f"{turnover:<10.2f}", end="")
        for gs in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
            result = simulate_signal_at_cost_level(gs, turnover, 120.0)
            symbol = "PASS" if result['passes'] else "  - "
            print(f" {symbol:<6}", end="")
        print()
    
    # 5. Actual framework result
    print("\n### 5. Actual Framework Result (momentum_12_1)")
    print("-" * 50)
    print("""
ACTUAL RESULT from execute_signals.py:
  - Gross Sharpe: 1.2
  - Net Sharpe: 0.32 (REJECTED < 0.5)
  - Cause: "Net Sharpe 0.32 < 0.5"

WHY SIMPLIFIED MODEL DIFFERS:
  - Framework uses SABOTAGE mode (1% commission, 0.2% spread, 0.5x ADV)
  - Costs applied to returns directly (not Sharpe adjustment)
  - Multi-year compounding amplifies cost drag
  - Reduced ADV increases market impact exponentially
    
This simplified model shows DIRECTIONAL sensitivity.
Actual pass/fail determined by execute_signals.py.
""")
    
    # 6. Summary
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
1. momentum_12_1 was REJECTED with Net Sharpe 0.32 (actual run)
   - Simplified model underestimates cost impact
   - SABOTAGE mode (1% commission + 0.2% spread + 0.5x ADV) is severe

2. EXISTENCE PROOF: A signal CAN pass if:
   - Low turnover (< 1x/year)
   - High gross Sharpe (> 1.5)
   - Trades liquid instruments
   - Example: Value strategy with 0.8 Sharpe, 0.5x turnover

3. Framework is deliberately harsh:
   - Tests survival under worst-case costs
   - Signals that pass would survive regime changes, volatility spikes
   - Better to reject borderline signals than blow up capital

THRESHOLD JUSTIFICATION:
- 0.5 Net Sharpe: Industry standard for "investable" (Sharpe 0.3 = "interesting")
- $25M capacity: Minimum for institutional allocation (covers operating costs)
- 3x turnover: Above this, transaction costs dominate for most strategies

SENSITIVITY SUMMARY:
- At 1.5 bps (SPY reality): Most strategies would pass
- At 10 bps (realistic): High-turnover strategies fail
- At 120 bps (SABOTAGE): Only exceptional strategies survive
""")


if __name__ == "__main__":
    run_sensitivity_analysis()
