"""
Analyze opportunities to own market infrastructure instead of trading.

The insight: If your framework says alpha extraction is hard,
maybe you should charge fees to people still trying.

Market infrastructure = unavoidable tollbooths.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import json


@dataclass
class InfrastructureOpportunity:
    name: str
    description: str
    examples: List[str]
    revenue_model: str
    moat: str
    capital_required_min: int
    capital_required_max: int
    time_to_revenue_months_min: int
    time_to_revenue_months_max: int
    income_potential_min: int
    income_potential_max: int
    required_skills: List[str]
    competition_score: float  # 0 = impossible, 1 = wide open


def enumerate_tollbooths() -> Dict[str, InfrastructureOpportunity]:
    """
    Map every point in the trading lifecycle where you can extract rent.
    
    Returns dict of infrastructure types and revenue models.
    """
    return {
        'data_provision': InfrastructureOpportunity(
            name='Data Provision',
            description='Sell market data, alternative data, signals',
            examples=['Bloomberg Terminal', 'Refinitiv', 'Quandl'],
            revenue_model='Subscription ($2k-$25k/user/month)',
            moat='Network effects (more users = more valuable data)',
            capital_required_min=500_000,
            capital_required_max=5_000_000,
            time_to_revenue_months_min=12,
            time_to_revenue_months_max=24,
            income_potential_min=1_000_000,
            income_potential_max=50_000_000,
            required_skills=['coding', 'data_engineering', 'sales', 'api_design'],
            competition_score=0.3
        ),
        
        'execution_infrastructure': InfrastructureOpportunity(
            name='Execution Infrastructure',
            description='Provide trading infrastructure, APIs, prime brokerage',
            examples=['Interactive Brokers', 'Alpaca', 'DriveWealth'],
            revenue_model='Per-trade fees + payment for order flow',
            moat='Regulatory barriers + switching costs',
            capital_required_min=10_000_000,
            capital_required_max=100_000_000,
            time_to_revenue_months_min=24,
            time_to_revenue_months_max=36,
            income_potential_min=10_000_000,
            income_potential_max=500_000_000,
            required_skills=['trading', 'regulatory', 'capital_markets', 'sales'],
            competition_score=0.4
        ),
        
        'backtesting_platforms': InfrastructureOpportunity(
            name='Backtesting Platforms',
            description='SaaS platforms for strategy testing',
            examples=['QuantConnect', 'Quantopian (RIP)', 'Backtrader'],
            revenue_model='Freemium ($0-$200/month per user)',
            moat='Community lock-in + data moat',
            capital_required_min=200_000,
            capital_required_max=2_000_000,
            time_to_revenue_months_min=6,
            time_to_revenue_months_max=18,
            income_potential_min=500_000,
            income_potential_max=10_000_000,
            required_skills=['coding', 'quant_finance', 'product_management'],
            competition_score=0.6
        ),
        
        'risk_management_tools': InfrastructureOpportunity(
            name='Risk Management Tools',
            description='Portfolio analytics, risk systems for institutions',
            examples=['FactSet', 'Axioma', 'Barra'],
            revenue_model='Enterprise licensing ($50k-$500k/client/year)',
            moat='Enterprise sales relationships + integration costs',
            capital_required_min=2_000_000,
            capital_required_max=10_000_000,
            time_to_revenue_months_min=18,
            time_to_revenue_months_max=36,
            income_potential_min=5_000_000,
            income_potential_max=100_000_000,
            required_skills=['finance', 'coding', 'enterprise_sales', 'statistics'],
            competition_score=0.5
        ),
        
        'index_construction': InfrastructureOpportunity(
            name='Index Construction',
            description='Create indices, license them to ETF providers',
            examples=['MSCI', 'FTSE Russell', 'S&P Dow Jones'],
            revenue_model='Basis points on AUM tracking your index',
            moat='First-mover advantage + benchmark status',
            capital_required_min=5_000_000,
            capital_required_max=20_000_000,
            time_to_revenue_months_min=24,
            time_to_revenue_months_max=48,
            income_potential_min=10_000_000,
            income_potential_max=1_000_000_000,
            required_skills=['academic', 'finance', 'marketing', 'regulatory'],
            competition_score=0.2
        ),
        
        'order_flow_aggregation': InfrastructureOpportunity(
            name='Order Flow Aggregation',
            description='Payment for order flow (be the Robinhood/Citadel middleman)',
            examples=['Citadel Securities', 'Virtu', 'Two Sigma Securities'],
            revenue_model='Fractions of pennies per share x billions of shares',
            moat='Scale + technology + relationships',
            capital_required_min=50_000_000,
            capital_required_max=500_000_000,
            time_to_revenue_months_min=36,
            time_to_revenue_months_max=60,
            income_potential_min=100_000_000,
            income_potential_max=1_000_000_000,
            required_skills=['hft', 'market_making', 'capital', 'regulatory'],
            competition_score=0.1
        ),
        
        'regulatory_compliance_tools': InfrastructureOpportunity(
            name='Regulatory Compliance Tools',
            description='Sell compliance software to funds (reporting, monitoring)',
            examples=['Workiva', 'ACA Compliance', 'ComplySci'],
            revenue_model='Enterprise SaaS ($20k-$200k/client/year)',
            moat='Regulatory complexity + switching costs',
            capital_required_min=1_000_000,
            capital_required_max=10_000_000,
            time_to_revenue_months_min=12,
            time_to_revenue_months_max=24,
            income_potential_min=2_000_000,
            income_potential_max=50_000_000,
            required_skills=['regulatory', 'coding', 'legal', 'enterprise_sales'],
            competition_score=0.7
        ),
        
        'smart_beta_factory': InfrastructureOpportunity(
            name='Smart Beta Factory',
            description='Create and license factor indices (value, momentum, quality)',
            examples=['AQR', 'Research Affiliates', 'WisdomTree'],
            revenue_model='Management fees on ETFs (0.15% - 0.75% of AUM)',
            moat='Academic credibility + distribution partnerships',
            capital_required_min=5_000_000,
            capital_required_max=50_000_000,
            time_to_revenue_months_min=24,
            time_to_revenue_months_max=36,
            income_potential_min=10_000_000,
            income_potential_max=500_000_000,
            required_skills=['academic', 'finance', 'marketing', 'etf_distribution'],
            competition_score=0.5
        ),
        
        'exchange_ownership': InfrastructureOpportunity(
            name='Exchange Ownership',
            description='Own the venue where trading happens',
            examples=['CME Group', 'Nasdaq', 'ICE'],
            revenue_model='Transaction fees + market data fees + listing fees',
            moat='Network effects + regulatory barriers (near impossible)',
            capital_required_min=100_000_000,
            capital_required_max=10_000_000_000,
            time_to_revenue_months_min=0,
            time_to_revenue_months_max=0,
            income_potential_min=100_000_000,
            income_potential_max=5_000_000_000,
            required_skills=['capital', 'regulatory', 'technology', 'market_structure'],
            competition_score=0.0
        ),
        
        'proprietary_cost_model_licensing': InfrastructureOpportunity(
            name='Proprietary Cost Model Licensing',
            description='Your framework itself - license to institutions',
            examples=['Axioma (risk models)', 'ITG (TCA models)'],
            revenue_model='Annual licensing ($50k-$500k/client)',
            moat='Academic validation + track record',
            capital_required_min=500_000,
            capital_required_max=2_000_000,
            time_to_revenue_months_min=12,
            time_to_revenue_months_max=18,
            income_potential_min=1_000_000,
            income_potential_max=20_000_000,
            required_skills=['academic', 'quant_finance', 'sales', 'research'],
            competition_score=0.8  # YOUR opportunity - under-served
        )
    }


def evaluate_opportunity(
    opportunity: InfrastructureOpportunity,
    your_capital: int,
    your_skills: List[str]
) -> Dict:
    """
    Given your resources, evaluate viability of this infrastructure play.
    
    Args:
        opportunity: The tollbooth to evaluate
        your_capital: How much money you can deploy
        your_skills: List of your skills
    
    Returns:
        Viability analysis with scores and next steps
    """
    # Capital feasibility
    capital_score = min(your_capital / opportunity.capital_required_min, 1.0)
    
    # Skill match
    required = set(opportunity.required_skills)
    yours = set(your_skills)
    skill_overlap = len(required & yours)
    skill_score = skill_overlap / len(required) if required else 0
    
    # Competition (higher = less competition = better)
    competition_score = opportunity.competition_score
    
    # Time to revenue (faster = better)
    avg_months = (opportunity.time_to_revenue_months_min + opportunity.time_to_revenue_months_max) / 2
    time_score = max(0, 1 - (avg_months / 48))  # Normalize: 0 months = 1.0, 48 months = 0
    
    # Weighted viability
    viability = (
        capital_score * 0.3 +
        skill_score * 0.3 +
        competition_score * 0.2 +
        time_score * 0.2
    )
    
    # Skill gaps
    skill_gaps = list(required - yours)
    
    # Capital gap
    capital_gap = max(0, opportunity.capital_required_min - your_capital)
    
    # Generate action plan
    if viability > 0.7:
        priority = "HIGH"
        action = f"Start building {opportunity.name} immediately. You have the capital and skills."
    elif viability > 0.4:
        priority = "MEDIUM"
        action = f"{opportunity.name} is viable but requires skill/capital acquisition first."
    else:
        priority = "LOW"
        action = f"{opportunity.name} not viable with current resources. Partner or choose different opportunity."
    
    return {
        'opportunity': opportunity.name,
        'viability_score': round(viability, 2),
        'capital_score': round(capital_score, 2),
        'skill_score': round(skill_score, 2),
        'competition_score': round(competition_score, 2),
        'time_score': round(time_score, 2),
        'capital_gap': capital_gap,
        'skill_gaps': skill_gaps,
        'income_potential': f"${opportunity.income_potential_min:,} - ${opportunity.income_potential_max:,}/year",
        'time_to_revenue': f"{opportunity.time_to_revenue_months_min}-{opportunity.time_to_revenue_months_max} months",
        'priority': priority,
        'action': action
    }


def rate_moat(moat_description: str) -> str:
    """Rate moat strength based on description."""
    moat_lower = moat_description.lower()
    
    if 'network effects' in moat_lower or 'regulatory' in moat_lower:
        return "STRONG"
    elif 'switching costs' in moat_lower or 'integration' in moat_lower:
        return "MEDIUM"
    else:
        return "WEAK"


def run_infrastructure_analysis(your_capital: int, your_skills: List[str]):
    """
    Run full analysis of all infrastructure opportunities.
    """
    print("=" * 80)
    print("MARKET INFRASTRUCTURE MOAT ANALYSIS")
    print("=" * 80)
    print(f"\nYour capital: ${your_capital:,}")
    print(f"Your skills: {', '.join(your_skills)}")
    print()
    
    opportunities = enumerate_tollbooths()
    results = []
    
    for name, opp in opportunities.items():
        result = evaluate_opportunity(opp, your_capital, your_skills)
        results.append(result)
    
    # Sort by viability
    results.sort(key=lambda x: x['viability_score'], reverse=True)
    
    print("### OPPORTUNITY RANKING (by viability score) ###\n")
    print(f"{'Opportunity':<35} {'Viability':>10} {'Priority':>10} {'Income Potential':<30}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['opportunity']:<35} {r['viability_score']:>10.2f} {r['priority']:>10} {r['income_potential']:<30}")
    
    print("\n### TOP 3 OPPORTUNITIES - DETAILED ###\n")
    
    for i, r in enumerate(results[:3], 1):
        print(f"#{i}: {r['opportunity']}")
        print(f"    Viability Score: {r['viability_score']}")
        print(f"    Capital Score: {r['capital_score']} (gap: ${r['capital_gap']:,})")
        print(f"    Skill Score: {r['skill_score']} (gaps: {r['skill_gaps'] or 'None'})")
        print(f"    Competition Score: {r['competition_score']} (higher = less competition)")
        print(f"    Time to Revenue: {r['time_to_revenue']}")
        print(f"    Income Potential: {r['income_potential']}")
        print(f"    ACTION: {r['action']}")
        print()
    
    # Save results
    with open('infrastructure_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Full analysis saved to infrastructure_analysis.json")
    
    return results


if __name__ == "__main__":
    # Example: Analyze with $500k capital and quant/coding skills
    your_capital = 500_000
    your_skills = ['coding', 'quant_finance', 'research', 'statistics', 'api_design']
    
    run_infrastructure_analysis(your_capital, your_skills)
