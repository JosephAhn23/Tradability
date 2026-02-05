"""
Model the economics of being a market infrastructure provider.

Key insight: Traders make money when right, lose when wrong.
Infrastructure providers make money EVERY TRADE, win or lose.
"""

from typing import Dict
import numpy as np


class TollboothEconomics:
    """
    Calculate revenue potential from charging fees on trading activity.
    """
    
    def __init__(self):
        self.market_stats = self.get_market_statistics()
    
    def get_market_statistics(self) -> Dict:
        """
        Global trading volume and potential revenue pools.
        """
        return {
            'us_equity_daily_volume': 500_000_000_000,  # $500B/day
            'us_equity_annual_volume': 125_000_000_000_000,  # $125T/year
            'global_equity_volume': 300_000_000_000_000,  # $300T/year
            'total_trading_costs': 102_000_000_000,  # $102B/year (French 2008)
            'number_of_trades_annually': 50_000_000_000,  # 50 billion trades
            'number_of_active_traders': 10_000_000,  # 10M active retail + institutional
            'number_of_institutions': 50_000,  # Funds, banks, etc.
        }
    
    def model_data_provider_revenue(self, subscribers: int, price_per_sub_monthly: float) -> Dict:
        """
        SaaS model: recurring revenue from data/tools.
        
        Example: 1,000 subscribers at $500/month = $6M/year
        """
        monthly_revenue = subscribers * price_per_sub_monthly
        annual_revenue = monthly_revenue * 12
        
        # Cost breakdown
        costs = {
            'data_acquisition': 0.30,  # 30%
            'infrastructure': 0.20,    # 20%
            'sales_marketing': 0.25,   # 25%
            'r_and_d': 0.15            # 15%
        }
        total_cost_ratio = sum(costs.values())
        gross_margin = 1 - total_cost_ratio
        net_profit = annual_revenue * gross_margin
        
        # Subscribers needed for $1M profit
        subs_for_1m = 1_000_000 / (price_per_sub_monthly * 12 * gross_margin)
        
        # Valuation at 8x revenue
        valuation = annual_revenue * 8
        
        return {
            'model': 'Subscription SaaS',
            'subscribers': subscribers,
            'price_monthly': price_per_sub_monthly,
            'annual_revenue': annual_revenue,
            'gross_margin': gross_margin,
            'net_profit': net_profit,
            'subscribers_for_1M_profit': int(subs_for_1m),
            'valuation_8x': valuation
        }
    
    def model_per_trade_fees(self, trades_per_year: int, fee_per_trade: float) -> Dict:
        """
        Transaction model: charge per trade.
        
        Example: Capture 0.1% of US equity trades at $0.01/trade
        """
        annual_revenue = trades_per_year * fee_per_trade
        
        # Costs: technology-heavy
        costs = {
            'technology': 0.40,
            'operations': 0.20,
            'sales': 0.10
        }
        total_cost_ratio = sum(costs.values())
        gross_margin = 1 - total_cost_ratio
        net_profit = annual_revenue * gross_margin
        
        market_share = trades_per_year / self.market_stats['number_of_trades_annually']
        
        return {
            'model': 'Per-Trade Fees',
            'trades_per_year': trades_per_year,
            'fee_per_trade': fee_per_trade,
            'annual_revenue': annual_revenue,
            'gross_margin': gross_margin,
            'net_profit': net_profit,
            'market_share': market_share,
            'valuation_6x': annual_revenue * 6
        }
    
    def model_basis_points_on_aum(self, aum_tracking: float, bps_fee: float) -> Dict:
        """
        Index/ETF model: charge basis points on assets.
        
        Example: $10B tracks your index, you charge 2 bps
        """
        annual_revenue = aum_tracking * (bps_fee / 10000)
        
        # Costs: marketing-heavy
        costs = {
            'marketing': 0.30,
            'operations': 0.10,
            'r_and_d': 0.20
        }
        total_cost_ratio = sum(costs.values())
        gross_margin = 1 - total_cost_ratio
        net_profit = annual_revenue * gross_margin
        
        aum_for_10m = 10_000_000 / ((bps_fee / 10000) * gross_margin)
        
        return {
            'model': 'Basis Points on AUM',
            'aum_tracking': aum_tracking,
            'bps_fee': bps_fee,
            'annual_revenue': annual_revenue,
            'gross_margin': gross_margin,
            'net_profit': net_profit,
            'aum_for_10M_profit': aum_for_10m,
            'valuation_12x': annual_revenue * 12
        }
    
    def model_enterprise_licensing(self, clients: int, annual_license_fee: float) -> Dict:
        """
        Enterprise licensing model: annual contracts with institutions.
        
        Example: 30 hedge funds at $100k/year = $3M/year
        """
        annual_revenue = clients * annual_license_fee
        
        # Costs: sales-heavy
        costs = {
            'sales': 0.35,
            'support': 0.15,
            'r_and_d': 0.20,
            'operations': 0.10
        }
        total_cost_ratio = sum(costs.values())
        gross_margin = 1 - total_cost_ratio
        net_profit = annual_revenue * gross_margin
        
        clients_for_1m = 1_000_000 / (annual_license_fee * gross_margin)
        
        return {
            'model': 'Enterprise Licensing',
            'clients': clients,
            'annual_license_fee': annual_license_fee,
            'annual_revenue': annual_revenue,
            'gross_margin': gross_margin,
            'net_profit': net_profit,
            'clients_for_1M_profit': int(clients_for_1m),
            'valuation_8x': annual_revenue * 8
        }
    
    def compare_revenue_models(self) -> Dict:
        """
        Which tollbooth has best economics?
        """
        scenarios = {
            'backtesting_saas': self.model_data_provider_revenue(
                subscribers=500,
                price_per_sub_monthly=200
            ),
            'enterprise_risk_tool': self.model_data_provider_revenue(
                subscribers=50,
                price_per_sub_monthly=5000
            ),
            'order_routing': self.model_per_trade_fees(
                trades_per_year=1_000_000_000,  # 1B trades
                fee_per_trade=0.001  # $0.001 per trade
            ),
            'smart_beta_index': self.model_basis_points_on_aum(
                aum_tracking=5_000_000_000,  # $5B AUM
                bps_fee=10  # 10 bps
            ),
            'your_framework_licensed': self.model_enterprise_licensing(
                clients=30,  # 30 hedge funds
                annual_license_fee=100_000  # $100k/year
            )
        }
        
        # Rank by net profit
        ranked = sorted(
            scenarios.items(),
            key=lambda x: x[1]['net_profit'],
            reverse=True
        )
        
        return {
            'scenarios': scenarios,
            'ranked_by_profit': [(name, data['net_profit']) for name, data in ranked],
            'best_opportunity': ranked[0][0],
            'best_profit': ranked[0][1]['net_profit']
        }


def run_tollbooth_analysis():
    """Main analysis of tollbooth economics."""
    print("=" * 80)
    print("TOLLBOOTH REVENUE MODEL ANALYSIS")
    print("=" * 80)
    
    economics = TollboothEconomics()
    
    # Market statistics
    print("\n### GLOBAL MARKET STATISTICS ###\n")
    stats = economics.market_stats
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value:,.0f}")
    
    # Compare revenue models
    print("\n### REVENUE MODEL COMPARISON ###\n")
    comparison = economics.compare_revenue_models()
    
    for name, scenario in comparison['scenarios'].items():
        print(f"\n{name.upper().replace('_', ' ')}")
        print(f"  Model: {scenario['model']}")
        print(f"  Annual Revenue: ${scenario['annual_revenue']:,.0f}")
        print(f"  Gross Margin: {scenario['gross_margin']:.0%}")
        print(f"  Net Profit: ${scenario['net_profit']:,.0f}")
        
        # Model-specific metrics
        if 'subscribers' in scenario:
            print(f"  Subscribers: {scenario['subscribers']:,}")
            print(f"  Price/Month: ${scenario['price_monthly']:,.0f}")
        if 'clients' in scenario:
            print(f"  Clients: {scenario['clients']:,}")
            print(f"  License Fee: ${scenario['annual_license_fee']:,.0f}/year")
        if 'aum_tracking' in scenario:
            print(f"  AUM Tracking: ${scenario['aum_tracking']:,.0f}")
            print(f"  BPS Fee: {scenario['bps_fee']} bps")
    
    # Rankings
    print("\n### RANKING BY NET PROFIT ###\n")
    for i, (name, profit) in enumerate(comparison['ranked_by_profit'], 1):
        print(f"  #{i}: {name.replace('_', ' ').title()} - ${profit:,.0f}")
    
    print(f"\n  BEST OPPORTUNITY: {comparison['best_opportunity'].replace('_', ' ').title()}")
    print(f"  BEST PROFIT: ${comparison['best_profit']:,.0f}/year")
    
    # Key insight
    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print("""
    Traders make money when RIGHT, lose when WRONG.
    Infrastructure providers make money on EVERY TRADE.
    
    Your framework licensing can generate:
    - $600k/year with just 30 clients
    - $3M/year at scale (150 clients)
    - $0 risk of losing principal
    - Recurring, predictable revenue
    
    Compare to trading:
    - $0/year (framework says signals don't work)
    - Risk of losing 100% of capital
    - Binary outcomes
    
    BE THE TOLLBOOTH.
    """)
    print("=" * 80)
    
    return comparison


if __name__ == "__main__":
    run_tollbooth_analysis()
