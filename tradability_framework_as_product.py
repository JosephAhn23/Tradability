"""
YOUR FRAMEWORK IS THE PRODUCT.

You built a cost model that institutions need. Don't trade with it - SELL it.

This file values your framework as a licensable product.
"""

from typing import Dict, List
import numpy as np


def identify_customers() -> Dict[str, Dict]:
    """
    Who would PAY for a cost model framework?
    """
    return {
        'hedge_funds': {
            'pain_point': 'Need to validate strategies before deploying capital',
            'willingness_to_pay_min': 50_000,
            'willingness_to_pay_max': 200_000,
            'total_addressable_market': 3_000,  # ~3,000 quant funds globally
            'realistic_penetration': 0.01,  # 1% market share
            'revenue_potential': 3_000 * 100_000 * 0.01  # $3M at 1%
        },
        
        'prop_trading_firms': {
            'pain_point': 'Evaluate trader PnL: skill or luck?',
            'willingness_to_pay_min': 100_000,
            'willingness_to_pay_max': 500_000,
            'total_addressable_market': 500,  # ~500 major prop shops
            'realistic_penetration': 0.20,  # 20% penetration
            'revenue_potential': 500 * 200_000 * 0.20  # $20M at 20%
        },
        
        'family_offices': {
            'pain_point': 'Due diligence on external managers',
            'willingness_to_pay_min': 25_000,
            'willingness_to_pay_max': 100_000,
            'total_addressable_market': 5_000,  # ~5,000 family offices with $100M+
            'realistic_penetration': 0.05,  # 5% penetration
            'revenue_potential': 5_000 * 50_000 * 0.05  # $12.5M at 5%
        },
        
        'pension_funds_endowments': {
            'pain_point': 'Evaluate active manager fees vs value',
            'willingness_to_pay_min': 50_000,
            'willingness_to_pay_max': 250_000,
            'total_addressable_market': 2_000,  # ~2,000 institutions
            'realistic_penetration': 0.10,  # 10% penetration
            'revenue_potential': 2_000 * 100_000 * 0.10  # $20M at 10%
        },
        
        'regulators': {
            'pain_point': 'Detect market manipulation vs legitimate trading',
            'willingness_to_pay_min': 500_000,
            'willingness_to_pay_max': 5_000_000,
            'total_addressable_market': 20,  # SEC, CFTC, FCA, MAS, etc.
            'realistic_penetration': 0.25,  # Win 5 contracts
            'revenue_potential': 20 * 1_000_000 * 0.25  # $5M
        },
        
        'broker_dealers': {
            'pain_point': 'Transaction cost analysis (TCA) for clients',
            'willingness_to_pay_min': 200_000,
            'willingness_to_pay_max': 1_000_000,
            'total_addressable_market': 100,  # ~100 major brokers
            'realistic_penetration': 0.50,  # 50% penetration
            'revenue_potential': 100 * 500_000 * 0.50  # $25M at 50%
        },
        
        'academic_institutions': {
            'pain_point': 'Research on market microstructure',
            'willingness_to_pay_min': 10_000,
            'willingness_to_pay_max': 50_000,
            'total_addressable_market': 200,  # ~200 business schools
            'realistic_penetration': 0.20,  # 20% penetration
            'revenue_potential': 200 * 25_000 * 0.20  # $1M
        }
    }


def value_framework_as_saas() -> Dict:
    """
    What's the framework worth as a SaaS product?
    
    Returns NPV of business over 10 years.
    """
    discount_rate = 0.15  # 15% - riskier than index, safer than trading
    
    # Year-by-year projections
    projections = {
        1: {'revenue': 0, 'costs': -250_000, 'note': 'Build product, initial sales'},
        2: {'revenue': 0, 'costs': -250_000, 'note': 'Continue development, find PMF'},
        3: {'revenue': 1_000_000, 'costs': -750_000, 'note': '10 customers @ $100k'},
        4: {'revenue': 2_000_000, 'costs': -1_200_000, 'note': '20 customers'},
        5: {'revenue': 3_500_000, 'costs': -1_800_000, 'note': '35 customers'},
        6: {'revenue': 5_000_000, 'costs': -2_000_000, 'note': '50 customers'},
        7: {'revenue': 7_000_000, 'costs': -2_500_000, 'note': '70 customers'},
        8: {'revenue': 8_500_000, 'costs': -2_800_000, 'note': '85 customers'},
        9: {'revenue': 10_000_000, 'costs': -3_000_000, 'note': '100 customers'},
        10: {'revenue': 10_000_000, 'costs': -2_000_000, 'note': 'Mature, high margin'}
    }
    
    # Calculate cash flows and NPV
    cash_flows = []
    cumulative_profit = 0
    
    for year, data in projections.items():
        profit = data['revenue'] + data['costs']
        cumulative_profit += profit
        discounted = profit / ((1 + discount_rate) ** year)
        cash_flows.append({
            'year': year,
            'revenue': data['revenue'],
            'costs': data['costs'],
            'profit': profit,
            'cumulative': cumulative_profit,
            'discounted': discounted
        })
    
    npv_cash_flows = sum(cf['discounted'] for cf in cash_flows)
    
    # Terminal value: SaaS companies sell for 5-10x revenue
    exit_multiple = 8
    year_10_revenue = projections[10]['revenue']
    terminal_value = year_10_revenue * exit_multiple
    npv_terminal = terminal_value / ((1 + discount_rate) ** 10)
    
    total_value = npv_cash_flows + npv_terminal
    
    return {
        'npv_cash_flows': npv_cash_flows,
        'terminal_value': terminal_value,
        'npv_terminal': npv_terminal,
        'total_business_value': total_value,
        'year_10_revenue': year_10_revenue,
        'year_10_profit': projections[10]['revenue'] + projections[10]['costs'],
        'exit_multiple': exit_multiple,
        'implied_valuation': terminal_value,
        'projections': cash_flows,
        'breakeven_year': next((cf['year'] for cf in cash_flows if cf['cumulative'] > 0), None)
    }


def compare_to_trading() -> Dict:
    """
    Trading: Make $0/year (framework says signals don't work)
    vs
    SaaS: Make $10M/year revenue, $8M profit by year 10
    
    Which would you rather do?
    """
    trading_income = {
        'year_1_10_total': 0,  # Your framework says you can't make money
        'probability_success': 0.05,  # 5% chance you find viable signal
        'time_required_weekly': 60,  # hours
        'stress_level': 'Extreme',
        'exit_value': 0,  # Can't sell a trading track record
        'certainty': 'Binary (win big or lose everything)'
    }
    
    saas_income = {
        'year_1_10_total': 25_450_000,  # Sum of profits
        'probability_success': 0.60,  # 60% chance of reaching 10 customers
        'time_required_weekly': 50,  # Initially, 20 at scale
        'stress_level': 'Medium (normal startup stress)',
        'exit_value': 80_000_000,  # 8x revenue
        'certainty': 'Recurring (monthly/annual contracts)'
    }
    
    # Expected value calculation
    ev_trading = trading_income['year_1_10_total'] * trading_income['probability_success']
    ev_saas = saas_income['year_1_10_total'] * saas_income['probability_success']
    
    # Risk-adjusted (include exit probability)
    exit_probability = 0.30  # 30% chance of successful exit
    ev_saas_with_exit = ev_saas + (saas_income['exit_value'] * exit_probability * 0.5)  # 50% stake after dilution
    
    return {
        'trading': trading_income,
        'saas': saas_income,
        'ev_trading': ev_trading,
        'ev_saas': ev_saas,
        'ev_saas_with_exit': ev_saas_with_exit,
        'ev_difference': ev_saas_with_exit - ev_trading,
        'winner': 'SaaS' if ev_saas_with_exit > ev_trading else 'Trading',
        'recommendation': 'Build the product, not the trade'
    }


def run_product_analysis():
    """Main analysis: Value framework as product."""
    print("=" * 80)
    print("FRAMEWORK AS PRODUCT ANALYSIS")
    print("=" * 80)
    
    # Customer analysis
    print("\n### CUSTOMER SEGMENTS ###\n")
    customers = identify_customers()
    
    total_tam = 0
    total_revenue_potential = 0
    
    print(f"{'Segment':<25} {'TAM':>8} {'Avg Price':>12} {'Penetration':>12} {'Revenue Pot.':>15}")
    print("-" * 75)
    
    for segment, data in customers.items():
        avg_price = (data['willingness_to_pay_min'] + data['willingness_to_pay_max']) / 2
        total_tam += data['total_addressable_market']
        total_revenue_potential += data['revenue_potential']
        print(f"{segment:<25} {data['total_addressable_market']:>8,} ${avg_price:>10,.0f} {data['realistic_penetration']:>11.0%} ${data['revenue_potential']:>13,.0f}")
    
    print("-" * 75)
    print(f"{'TOTAL':<25} {total_tam:>8,} {'':<12} {'':<12} ${total_revenue_potential:>13,.0f}")
    
    # SaaS valuation
    print("\n### SAAS BUSINESS VALUATION ###\n")
    valuation = value_framework_as_saas()
    
    print("10-Year Financial Projections:")
    print(f"{'Year':>6} {'Revenue':>12} {'Costs':>12} {'Profit':>12} {'Cumulative':>12}")
    print("-" * 60)
    
    for cf in valuation['projections']:
        print(f"{cf['year']:>6} ${cf['revenue']:>10,} ${cf['costs']:>10,} ${cf['profit']:>10,} ${cf['cumulative']:>10,}")
    
    print()
    print(f"NPV of Cash Flows: ${valuation['npv_cash_flows']:,.0f}")
    print(f"Terminal Value ({valuation['exit_multiple']}x revenue): ${valuation['terminal_value']:,.0f}")
    print(f"NPV of Terminal Value: ${valuation['npv_terminal']:,.0f}")
    print(f"TOTAL BUSINESS VALUE: ${valuation['total_business_value']:,.0f}")
    print(f"Breakeven Year: {valuation['breakeven_year']}")
    
    # Comparison to trading
    print("\n### TRADING VS SAAS COMPARISON ###\n")
    comparison = compare_to_trading()
    
    print("Trading Path:")
    print(f"  10-year income: ${comparison['trading']['year_1_10_total']:,}")
    print(f"  Probability of success: {comparison['trading']['probability_success']:.0%}")
    print(f"  Expected value: ${comparison['ev_trading']:,.0f}")
    print(f"  Exit value: ${comparison['trading']['exit_value']:,}")
    
    print("\nSaaS Path:")
    print(f"  10-year income: ${comparison['saas']['year_1_10_total']:,}")
    print(f"  Probability of success: {comparison['saas']['probability_success']:.0%}")
    print(f"  Expected value (ops): ${comparison['ev_saas']:,.0f}")
    print(f"  Expected value (with exit): ${comparison['ev_saas_with_exit']:,.0f}")
    print(f"  Exit value: ${comparison['saas']['exit_value']:,}")
    
    print(f"\nWINNER: {comparison['winner']}")
    print(f"EV Difference: ${comparison['ev_difference']:,.0f}")
    print(f"RECOMMENDATION: {comparison['recommendation']}")
    
    print("\n" + "=" * 80)
    
    return {
        'customers': customers,
        'valuation': valuation,
        'comparison': comparison
    }


if __name__ == "__main__":
    run_product_analysis()
