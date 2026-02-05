"""
The ultimate decision framework:

Should you extract alpha (Level 1-9) or own infrastructure (Level 10)?
"""

from typing import Dict, List
import json


def expected_value_alpha_extraction() -> Dict:
    """
    What's the EV of continuing to search for tradable signals?
    """
    # Based on framework results: signals struggle after costs
    # Probability you find a viable signal after extensive testing: Low
    
    assumptions = {
        'probability_find_viable_signal': 0.15,  # 15% after testing 100+ signals
        'years_searching': 2,
        'income_if_found_annually': 150_000,  # Level 2 (full-time trader)
        'income_if_not_found': 0,
        'opportunity_cost_per_year': 100_000,  # What you could earn elsewhere
        'capital_at_risk': 100_000,  # Initial trading capital
        'probability_lose_capital': 0.50,  # 50% chance of significant loss
    }
    
    # Success path: Find signal, trade for 8 years
    years_trading = 10 - assumptions['years_searching']
    ev_success = (
        assumptions['probability_find_viable_signal'] *
        assumptions['income_if_found_annually'] *
        years_trading
    )
    
    # Failure path: No income + lose capital
    ev_failure = (
        (1 - assumptions['probability_find_viable_signal']) *
        (assumptions['income_if_not_found'] - 
         assumptions['probability_lose_capital'] * assumptions['capital_at_risk'])
    )
    
    # Opportunity cost
    opportunity_cost = assumptions['opportunity_cost_per_year'] * assumptions['years_searching']
    
    net_ev = ev_success + ev_failure - opportunity_cost
    
    return {
        'expected_value': net_ev,
        'probability_success': assumptions['probability_find_viable_signal'],
        'years_to_profitability': assumptions['years_searching'],
        'risk': 'HIGH (85% chance of failure)',
        'certainty_of_income': 'LOW (binary: win big or lose everything)',
        'capital_required': assumptions['capital_at_risk'],
        'worst_case': -assumptions['capital_at_risk'] - opportunity_cost,
        'best_case': assumptions['income_if_found_annually'] * years_trading,
        'assumptions': assumptions
    }


def expected_value_infrastructure() -> Dict:
    """
    What's the EV of building market infrastructure product?
    """
    assumptions = {
        'probability_reach_10_customers': 0.70,  # 70% - you have working product
        'probability_reach_50_customers': 0.40,  # 40% - sales execution risk
        'probability_exit_8x_revenue': 0.30,     # 30% - market timing risk
        'year_3_profit': 250_000,
        'year_5_profit': 1_700_000,
        'year_10_profit': 8_000_000,
        'exit_value': 80_000_000,
        'equity_stake_at_exit': 0.50,  # After dilution
        'capital_required': 500_000,
        'years_to_breakeven': 3
    }
    
    # Modest success: reach 10 customers, operate 7 years
    ev_modest = (
        assumptions['probability_reach_10_customers'] *
        assumptions['year_3_profit'] *
        7  # Years 3-10
    )
    
    # Major success: reach 50 customers
    ev_major = (
        assumptions['probability_reach_50_customers'] *
        assumptions['year_10_profit']
    )
    
    # Exit value
    ev_exit = (
        assumptions['probability_exit_8x_revenue'] *
        assumptions['exit_value'] *
        assumptions['equity_stake_at_exit']
    )
    
    total_ev = ev_modest + ev_major + ev_exit
    
    return {
        'expected_value': total_ev,
        'probability_success': assumptions['probability_reach_10_customers'],
        'years_to_profitability': assumptions['years_to_breakeven'],
        'risk': 'MEDIUM (40-70% success range)',
        'certainty_of_income': 'HIGH (recurring revenue, not binary)',
        'capital_required': assumptions['capital_required'],
        'worst_case': -assumptions['capital_required'],
        'best_case': assumptions['exit_value'] * assumptions['equity_stake_at_exit'],
        'assumptions': assumptions
    }


def generate_action_plan(recommendation: str) -> Dict:
    """
    Based on recommendation, what should you do in next 30/90/365 days?
    """
    if 'infrastructure' in recommendation.lower():
        return {
            'next_30_days': [
                'Package framework as Python library with clean API',
                'Build simple web UI for framework (Streamlit/Gradio)',
                'Validate pricing with 5 potential customers (hedge funds)',
                'Create pitch deck for institutional sales'
            ],
            'next_90_days': [
                'Close first 3 paying customers at $50k/year each',
                'Build enterprise features (multi-user, audit logs, custom reports)',
                'Hire first sales/customer success person',
                'Raise $500k seed funding or bootstrap from revenue'
            ],
            'next_365_days': [
                'Reach 15-20 customers ($1.5M ARR)',
                'Expand product: add more asset classes, factor models, regime detection',
                'Build channel partnerships with prime brokers, consultants',
                'Achieve profitability or raise Series A'
            ]
        }
    else:
        return {
            'next_30_days': [
                'Test 50 additional signals across quality spectrum',
                'Expand to 10 assets (IWM, EEM, bonds, commodities)',
                'Calibrate cost model to realistic assumptions',
                'Find at least 3 signals that pass framework'
            ],
            'next_90_days': [
                'Paper trade best signals for 90 days',
                'Validate predicted vs actual costs',
                'Raise $50k-$100k for live trading capital',
                'Build execution infrastructure'
            ],
            'next_365_days': [
                'Deploy capital to best strategies',
                'Track record: target Sharpe > 1.5',
                'Scale to $500k-$1M AUM',
                'Decide: continue trading or pivot to fund raising'
            ]
        }


def make_recommendation() -> Dict:
    """
    Compare EV of both paths and recommend action.
    """
    alpha_ev = expected_value_alpha_extraction()
    infra_ev = expected_value_infrastructure()
    
    ev_diff = infra_ev['expected_value'] - alpha_ev['expected_value']
    ev_ratio = infra_ev['expected_value'] / max(alpha_ev['expected_value'], 1)
    
    # Decision logic
    if infra_ev['expected_value'] > alpha_ev['expected_value'] * 2:
        recommendation = "STRONG RECOMMENDATION: Build infrastructure product"
        reasoning = f"Infrastructure EV (${infra_ev['expected_value']:,.0f}) is {ev_ratio:.1f}x higher than alpha extraction (${alpha_ev['expected_value']:,.0f})."
    elif infra_ev['expected_value'] > alpha_ev['expected_value']:
        recommendation = "MODERATE RECOMMENDATION: Pursue infrastructure"
        reasoning = "Infrastructure has better risk-adjusted returns, but alpha extraction still viable."
    else:
        recommendation = "CONTINUE ALPHA SEARCH"
        reasoning = "Despite current results, EV of finding signal exceeds infrastructure opportunity."
    
    # Risk comparison
    if infra_ev['risk'] < alpha_ev['risk']:
        better_risk = 'Infrastructure'
    else:
        better_risk = 'Alpha'
    
    # Speed to profit
    if infra_ev['years_to_profitability'] < alpha_ev['years_to_profitability']:
        faster = 'Infrastructure'
    else:
        faster = 'Alpha'
    
    action_plan = generate_action_plan(recommendation)
    
    return {
        'recommendation': recommendation,
        'reasoning': reasoning,
        'alpha_extraction': alpha_ev,
        'infrastructure': infra_ev,
        'ev_difference': ev_diff,
        'ev_ratio': ev_ratio,
        'better_risk_profile': better_risk,
        'faster_to_profit': faster,
        'action_plan': action_plan
    }


def run_decision_analysis():
    """Main analysis: Alpha vs Infrastructure decision."""
    print("=" * 80)
    print("INFRASTRUCTURE VS ALPHA EXTRACTION DECISION")
    print("=" * 80)
    
    result = make_recommendation()
    
    # Alpha extraction analysis
    print("\n### PATH 1: ALPHA EXTRACTION (Levels 1-4) ###\n")
    alpha = result['alpha_extraction']
    print(f"  Expected Value: ${alpha['expected_value']:,.0f}")
    print(f"  Probability of Success: {alpha['probability_success']:.0%}")
    print(f"  Time to Profitability: {alpha['years_to_profitability']} years")
    print(f"  Risk Level: {alpha['risk']}")
    print(f"  Income Certainty: {alpha['certainty_of_income']}")
    print(f"  Capital Required: ${alpha['capital_required']:,}")
    print(f"  Worst Case: ${alpha['worst_case']:,}")
    print(f"  Best Case: ${alpha['best_case']:,}")
    
    # Infrastructure analysis
    print("\n### PATH 2: INFRASTRUCTURE (Level 7-10) ###\n")
    infra = result['infrastructure']
    print(f"  Expected Value: ${infra['expected_value']:,.0f}")
    print(f"  Probability of Success: {infra['probability_success']:.0%}")
    print(f"  Time to Profitability: {infra['years_to_profitability']} years")
    print(f"  Risk Level: {infra['risk']}")
    print(f"  Income Certainty: {infra['certainty_of_income']}")
    print(f"  Capital Required: ${infra['capital_required']:,}")
    print(f"  Worst Case: ${infra['worst_case']:,}")
    print(f"  Best Case: ${infra['best_case']:,}")
    
    # Comparison
    print("\n### COMPARISON ###\n")
    print(f"  EV Difference: ${result['ev_difference']:,.0f}")
    print(f"  EV Ratio (Infra/Alpha): {result['ev_ratio']:.1f}x")
    print(f"  Better Risk Profile: {result['better_risk_profile']}")
    print(f"  Faster to Profit: {result['faster_to_profit']}")
    
    # Recommendation
    print("\n" + "=" * 80)
    print(f"RECOMMENDATION: {result['recommendation']}")
    print("=" * 80)
    print(f"\n{result['reasoning']}")
    
    # Action Plan
    print("\n### ACTION PLAN ###")
    
    print("\nNext 30 Days:")
    for item in result['action_plan']['next_30_days']:
        print(f"  - {item}")
    
    print("\nNext 90 Days:")
    for item in result['action_plan']['next_90_days']:
        print(f"  - {item}")
    
    print("\nNext 365 Days:")
    for item in result['action_plan']['next_365_days']:
        print(f"  - {item}")
    
    print("\n" + "=" * 80)
    print("BOTTOM LINE")
    print("=" * 80)
    print(f"""
    Your framework rejected trading signals. Don't reject the framework.
    The framework itself is the product. License it.
    
    Expected 10-year value:
    - Trading: ${alpha['expected_value']:,.0f}
    - Infrastructure: ${infra['expected_value']:,.0f}
    
    The market has spoken: Be the toll collector, not the traveler.
    """)
    print("=" * 80)
    
    # Save results
    with open('alpha_vs_infrastructure_decision.json', 'w') as f:
        # Remove non-serializable items
        output = {
            'recommendation': result['recommendation'],
            'reasoning': result['reasoning'],
            'ev_difference': result['ev_difference'],
            'ev_ratio': result['ev_ratio'],
            'better_risk_profile': result['better_risk_profile'],
            'faster_to_profit': result['faster_to_profit'],
            'alpha_ev': alpha['expected_value'],
            'infra_ev': infra['expected_value'],
            'action_plan': result['action_plan']
        }
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to alpha_vs_infrastructure_decision.json")
    
    return result


if __name__ == "__main__":
    run_decision_analysis()
