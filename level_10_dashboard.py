"""
Visual dashboard comparing income paths.

Shows:
1. Alpha extraction (Levels 1-9) vs Infrastructure (Level 10)
2. Expected income by year for each path
3. Probability-weighted outcomes
4. Action plan based on recommendation
"""

import numpy as np
from typing import Dict, List
import json

from infrastructure_vs_alpha_decision import make_recommendation


def get_income_levels() -> Dict[int, Dict]:
    """
    All 10 income levels with expected values.
    """
    levels = {
        1: {'name': 'Retail Trader', 'income': 0, 'probability': 0.95, 'time_years': 1},
        2: {'name': 'Full-Time Trader', 'income': 100_000, 'probability': 0.15, 'time_years': 2},
        3: {'name': 'Small Fund', 'income': 300_000, 'probability': 0.05, 'time_years': 4},
        4: {'name': 'Established Fund', 'income': 2_000_000, 'probability': 0.01, 'time_years': 7},
        5: {'name': 'Market Maker', 'income': 5_000_000, 'probability': 0.005, 'time_years': 5},
        6: {'name': 'Prop Desk', 'income': 2_000_000, 'probability': 0.02, 'time_years': 5},
        7: {'name': 'Quant Platform', 'income': 5_000_000, 'probability': 0.30, 'time_years': 4},
        8: {'name': 'Multi-Strat', 'income': 20_000_000, 'probability': 0.001, 'time_years': 10},
        9: {'name': 'Pricing Power', 'income': 50_000_000, 'probability': 0.0001, 'time_years': 10},
        10: {'name': 'Be The Market', 'income': 100_000_000, 'probability': 0.00001, 'time_years': 15}
    }
    
    # Calculate expected value and risk-adjusted value
    for level, data in levels.items():
        data['expected_value'] = data['income'] * data['probability']
        data['risk_adjusted_value'] = data['expected_value'] / max(data['time_years'], 1)
    
    return levels


def get_path_projections() -> Dict[str, List[int]]:
    """
    Get year-by-year income projections for each path.
    """
    years = list(range(0, 11))
    
    # Path 1: Alpha extraction
    # 15% chance of success in year 2, else $0
    alpha_success = [0, 0, 150_000] + [150_000] * 8
    alpha_failure = [0] * 11
    alpha_expected = [
        int(0.15 * alpha_success[i] + 0.85 * alpha_failure[i])
        for i in range(11)
    ]
    
    # Path 2: Infrastructure SaaS
    infra_path = [0, -100_000, 0, 250_000, 500_000, 1_000_000, 2_000_000, 
                  4_000_000, 6_000_000, 8_000_000, 8_000_000]
    
    return {
        'years': years,
        'alpha_success': alpha_success,
        'alpha_failure': alpha_failure,
        'alpha_expected': alpha_expected,
        'infrastructure': infra_path
    }


def create_text_dashboard():
    """Create text-based dashboard."""
    print("=" * 80)
    print("                    INCOME LEVEL 10 DASHBOARD")
    print("                 Alpha Extraction vs Infrastructure")
    print("=" * 80)
    
    # Panel 1: Income Levels Overview
    print("\n" + "-" * 80)
    print("PANEL 1: All 10 Income Levels")
    print("-" * 80)
    
    levels = get_income_levels()
    
    print(f"\n{'Level':<8} {'Name':<20} {'Income':>15} {'Probability':>12} {'Expected Value':>15}")
    print("-" * 75)
    
    for level, data in levels.items():
        prob_str = f"{data['probability']:.2%}" if data['probability'] >= 0.01 else f"{data['probability']:.4%}"
        print(f"L{level:<7} {data['name']:<20} ${data['income']:>13,} {prob_str:>12} ${data['expected_value']:>13,.0f}")
    
    # Highlight best risk-adjusted
    best_level = max(levels.items(), key=lambda x: x[1]['risk_adjusted_value'])
    print(f"\nBest risk-adjusted: Level {best_level[0]} ({best_level[1]['name']})")
    
    # Panel 2: Path Comparison Timeline
    print("\n" + "-" * 80)
    print("PANEL 2: Income Over Time (10-Year Projection)")
    print("-" * 80)
    
    projections = get_path_projections()
    
    print(f"\n{'Year':>6} {'Alpha (EV)':>15} {'Infrastructure':>15} {'Difference':>15}")
    print("-" * 55)
    
    cumulative_alpha = 0
    cumulative_infra = 0
    
    for i, year in enumerate(projections['years']):
        alpha = projections['alpha_expected'][i]
        infra = projections['infrastructure'][i]
        cumulative_alpha += alpha
        cumulative_infra += infra
        diff = infra - alpha
        
        print(f"{year:>6} ${alpha:>13,} ${infra:>13,} ${diff:>13,}")
    
    print("-" * 55)
    print(f"{'TOTAL':>6} ${cumulative_alpha:>13,} ${cumulative_infra:>13,} ${cumulative_infra - cumulative_alpha:>13,}")
    
    # Panel 3: Decision Summary
    print("\n" + "-" * 80)
    print("PANEL 3: Decision Analysis")
    print("-" * 80)
    
    decision = make_recommendation()
    
    print(f"\nAlpha Extraction Path:")
    print(f"  Expected Value: ${decision['alpha_extraction']['expected_value']:,.0f}")
    print(f"  Success Probability: {decision['alpha_extraction']['probability_success']:.0%}")
    print(f"  Risk Level: {decision['alpha_extraction']['risk']}")
    
    print(f"\nInfrastructure Path:")
    print(f"  Expected Value: ${decision['infrastructure']['expected_value']:,.0f}")
    print(f"  Success Probability: {decision['infrastructure']['probability_success']:.0%}")
    print(f"  Risk Level: {decision['infrastructure']['risk']}")
    
    print(f"\nEV Ratio (Infrastructure/Alpha): {decision['ev_ratio']:.1f}x")
    
    # Panel 4: Final Recommendation
    print("\n" + "=" * 80)
    print("                         FINAL RECOMMENDATION")
    print("=" * 80)
    
    print(f"\n{decision['recommendation']}")
    print(f"\n{decision['reasoning']}")
    
    print("\nAction Plan (Next 30 Days):")
    for item in decision['action_plan']['next_30_days']:
        print(f"  - {item}")
    
    # Key insight
    print("\n" + "=" * 80)
    print("""
    THE LEVEL 10 ANSWER:
    
    Don't trade. OWN THE CASINO.
    
    - Traders make money when RIGHT, lose when WRONG
    - Infrastructure owners make money on EVERY TRADE
    - Your framework IS the product
    - License it to people still trying to extract alpha
    
    Expected 10-year difference: ${:,.0f} in favor of infrastructure
    """.format(decision['ev_difference']))
    print("=" * 80)
    
    return decision


def create_plotly_dashboard():
    """Create interactive Plotly dashboard if available."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not available. Creating text dashboard.\n")
        return create_text_dashboard()
    
    levels = get_income_levels()
    projections = get_path_projections()
    decision = make_recommendation()
    
    # Create 2x2 subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Income Levels: Nominal vs Expected Value",
            "10-Year Income Projection",
            "Risk vs Reward by Level",
            "Cumulative Wealth Over Time"
        ),
        specs=[
            [{"secondary_y": True}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ]
    )
    
    # Panel 1: Income levels
    level_names = [f"L{k}: {v['name']}" for k, v in levels.items()]
    nominal = [v['income'] for v in levels.values()]
    expected = [v['expected_value'] for v in levels.values()]
    
    fig.add_trace(
        go.Bar(x=level_names, y=nominal, name='Nominal Income', marker_color='lightblue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=level_names, y=expected, name='Expected Value', 
                   mode='lines+markers', marker_color='red'),
        row=1, col=1, secondary_y=True
    )
    
    # Panel 2: Path projections
    years = projections['years']
    
    fig.add_trace(
        go.Scatter(x=years, y=projections['alpha_expected'], 
                   name='Alpha (EV)', mode='lines+markers',
                   line=dict(color='blue', dash='dash')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=years, y=projections['infrastructure'],
                   name='Infrastructure', mode='lines+markers',
                   line=dict(color='green')),
        row=1, col=2
    )
    
    # Panel 3: Risk vs Reward scatter
    probs = [v['probability'] for v in levels.values()]
    incomes = [v['income'] for v in levels.values()]
    
    fig.add_trace(
        go.Scatter(x=probs, y=incomes, mode='markers+text',
                   text=[f"L{k}" for k in levels.keys()],
                   textposition='top center',
                   marker=dict(size=12, color=list(range(10)), colorscale='Viridis')),
        row=2, col=1
    )
    
    # Panel 4: Cumulative wealth
    alpha_cumulative = np.cumsum(projections['alpha_expected'])
    infra_cumulative = np.cumsum(projections['infrastructure'])
    
    fig.add_trace(
        go.Scatter(x=years, y=alpha_cumulative, name='Alpha Cumulative',
                   fill='tozeroy', line=dict(color='blue')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=years, y=infra_cumulative, name='Infrastructure Cumulative',
                   fill='tozeroy', line=dict(color='green')),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="INCOME LEVEL 10 DASHBOARD: Be The Market",
        title_x=0.5,
        height=900,
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Income Level", row=1, col=1)
    fig.update_yaxes(title_text="Nominal Income ($)", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_yaxes(title_text="Annual Income ($)", row=1, col=2)
    fig.update_xaxes(title_text="Probability", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Income ($)", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=2)
    fig.update_yaxes(title_text="Cumulative Wealth ($)", row=2, col=2)
    
    # Save
    fig.write_html('level_10_dashboard.html')
    print("Interactive dashboard saved to level_10_dashboard.html")
    
    # Also print text summary
    print("\n" + "=" * 80)
    print("DASHBOARD SUMMARY")
    print("=" * 80)
    print(f"\nRecommendation: {decision['recommendation']}")
    print(f"EV Ratio: {decision['ev_ratio']:.1f}x in favor of infrastructure")
    print(f"\nConclusion: Don't trade. OWN THE CASINO.")
    
    return decision


def generate_final_report() -> str:
    """
    Single-page summary: What should you do?
    """
    decision = make_recommendation()
    
    alpha = decision['alpha_extraction']
    infra = decision['infrastructure']
    
    report = f"""
═══════════════════════════════════════════════════════════════════════════════
                    INCOME LEVEL 10 ANALYSIS: FINAL RECOMMENDATION
═══════════════════════════════════════════════════════════════════════════════

YOUR FRAMEWORK'S VERDICT:
- Alpha extraction: Signals struggle after realistic transaction costs
- Net conclusion: Traditional trading is a hard path

EXPECTED VALUE ANALYSIS:

PATH 1: Continue Alpha Search (Levels 1-4)
- Expected income: ${alpha['expected_value']:,.0f}
- Probability of success: {alpha['probability_success']:.0%}
- Time to profitability: {alpha['years_to_profitability']} years
- Risk level: {alpha['risk']}

PATH 2: Build Infrastructure (Level 7-10)
- Expected income: ${infra['expected_value']:,.0f}
- Probability of success: {infra['probability_success']:.0%}
- Time to profitability: {infra['years_to_profitability']} years
- Risk level: {infra['risk']}

RECOMMENDATION:
{decision['recommendation']}

REASONING:
{decision['reasoning']}

ACTION PLAN:

Next 30 Days:
{chr(10).join(f"  - {item}" for item in decision['action_plan']['next_30_days'])}

Next 90 Days:
{chr(10).join(f"  - {item}" for item in decision['action_plan']['next_90_days'])}

Next 365 Days:
{chr(10).join(f"  - {item}" for item in decision['action_plan']['next_365_days'])}

═══════════════════════════════════════════════════════════════════════════════
BOTTOM LINE:

Your framework helps evaluate trading signals. Don't use it to trade.
The framework itself is the product. License it.

Expected 10-year value:
- Trading: ${alpha['expected_value']:,.0f}
- Infrastructure: ${infra['expected_value']:,.0f}

The market has spoken: Be the toll collector, not the traveler.
═══════════════════════════════════════════════════════════════════════════════
"""
    
    return report


def run_dashboard():
    """Main entry point for dashboard."""
    print("Building Level 10 Dashboard...\n")
    
    # Try Plotly, fall back to text
    try:
        decision = create_plotly_dashboard()
    except Exception as e:
        print(f"Plotly failed: {e}")
        decision = create_text_dashboard()
    
    # Generate and save report
    report = generate_final_report()
    
    with open('level_10_final_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nFinal report saved to level_10_final_report.txt")
    
    # Save dashboard data
    levels = get_income_levels()
    projections = get_path_projections()
    
    dashboard_data = {
        'levels': {str(k): {
            'name': v['name'],
            'income': v['income'],
            'probability': v['probability'],
            'expected_value': v['expected_value']
        } for k, v in levels.items()},
        'projections': projections,
        'recommendation': decision['recommendation'],
        'ev_ratio': decision['ev_ratio']
    }
    
    with open('level_10_dashboard_data.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    print(f"Dashboard data saved to level_10_dashboard_data.json")
    
    return decision


if __name__ == "__main__":
    run_dashboard()
