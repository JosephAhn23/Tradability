"""
Generalization Analysis: Empirical Claims Across Signal Types

Makes and defends a general claim about tradability.
Not just about two signals - about a class of signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class GeneralizationClaim:
    """
    A formal generalization claim about signal tradability.
    """
    claim: str
    condition: str  # Mathematical condition
    supporting_evidence: Dict
    counterexamples: List[str]  # Signals that violate the claim
    scope: str  # Scope of applicability


def analyze_turnover_threshold_hypothesis(signal_results: Dict[str, Dict]) -> GeneralizationClaim:
    """
    Test the hypothesis: "Signals with turnover above X per year are not economically viable below Y cost"
    
    Args:
        signal_results: Dictionary mapping signal names to analysis results
    
    Returns:
        GeneralizationClaim object
    """
    # Extract data
    turnovers = []
    break_even_costs = []
    net_sharpes = []
    signal_names = []
    
    for name, results in signal_results.items():
        if 'annual_turnover' in results and 'break_even_cost' in results:
            turnovers.append(results['annual_turnover'])
            break_even_costs.append(results['break_even_cost'])
            net_sharpes.append(results.get('net_sharpe', 0))
            signal_names.append(name)
    
    if len(turnovers) < 2:
        return GeneralizationClaim(
            claim="Insufficient data for generalization",
            condition="N/A",
            supporting_evidence={},
            counterexamples=[],
            scope="N/A"
        )
    
    # Find threshold: turnover level where break-even cost becomes very low (< 0.1%)
    high_turnover_signals = [i for i, t in enumerate(turnovers) if t > np.median(turnovers)]
    low_break_even = [break_even_costs[i] for i in high_turnover_signals]
    
    if len(low_break_even) > 0:
        threshold_turnover = np.median([turnovers[i] for i in high_turnover_signals])
        threshold_cost = np.percentile(low_break_even, 75)  # 75th percentile
        
        # Test the claim
        violating_signals = []
        supporting_signals = []
        
        for i, name in enumerate(signal_names):
            if turnovers[i] > threshold_turnover:
                if break_even_costs[i] > threshold_cost:
                    violating_signals.append(name)
                else:
                    supporting_signals.append(name)
        
        claim = (
            f"Signals with annual turnover above {threshold_turnover:.1f}x "
            f"require transaction costs below {threshold_cost*100:.2f}% per trade "
            f"to preserve economic viability. Above this turnover threshold, "
            f"cost drag overwhelms residual alpha even when statistical edge persists."
        )
        
        condition = f"Turnover > {threshold_turnover:.1f} -> BreakEvenCost < {threshold_cost*100:.2f}%"
        
        return GeneralizationClaim(
            claim=claim,
            condition=condition,
            supporting_evidence={
                'threshold_turnover': threshold_turnover,
                'threshold_cost': threshold_cost,
                'supporting_signals': supporting_signals,
                'n_signals_tested': len(signal_names),
                'n_violations': len(violating_signals),
                'caveat': 'Thresholds are empirical estimates from observed signals; may vary with signal type, market conditions, and cost structure'
            },
            counterexamples=violating_signals,
            scope="Simple technical signals with fixed parameters, tested on liquid equity indices (SPY) during 2000-2020"
        )
    else:
        return GeneralizationClaim(
            claim="Cannot establish turnover threshold",
            condition="N/A",
            supporting_evidence={},
            counterexamples=[],
            scope="N/A"
        )


def analyze_horizon_viability_hypothesis(signal_results: Dict[str, Dict]) -> GeneralizationClaim:
    """
    Test the hypothesis: "Short-horizon mean reversion is fundamentally untradable outside privileged execution"
    
    This is a more specific claim about signal class (mean reversion) and horizon.
    """
    mean_reversion_signals = {k: v for k, v in signal_results.items() if 'mean_reversion' in k.lower() or 'reversion' in k.lower()}
    
    if len(mean_reversion_signals) == 0:
        return GeneralizationClaim(
            claim="No mean reversion signals to analyze",
            condition="N/A",
            supporting_evidence={},
            counterexamples=[],
            scope="N/A"
        )
    
    # Check if mean reversion signals show economic failure
    all_untradable = True
    for name, results in mean_reversion_signals.items():
        net_sharpe = results.get('net_sharpe', 0)
        if net_sharpe > 0:
            all_untradable = False
            break
    
    if all_untradable:
        claim = (
            "Short-horizon mean reversion signals are fundamentally untradable "
            "at realistic transaction cost levels, even when they exhibit statistical edge. "
            "The combination of high turnover and small expected edge makes cost drag "
            "insurmountable without privileged execution infrastructure."
        )
        
        return GeneralizationClaim(
            claim=claim,
            condition="Mean reversion + Short horizon -> Net Sharpe < 0 at realistic costs",
            supporting_evidence={
                'mean_reversion_signals': list(mean_reversion_signals.keys()),
                'all_show_net_negative': True
            },
            counterexamples=[],
            scope="Mean reversion signals with holding periods < 20 days"
        )
    else:
        return GeneralizationClaim(
            claim="Mean reversion signals show mixed tradability",
            condition="N/A",
            supporting_evidence={},
            counterexamples=list(mean_reversion_signals.keys()),
            scope="N/A"
        )


def analyze_hit_rate_orthogonality_hypothesis(signal_results: Dict[str, Dict]) -> GeneralizationClaim:
    """
    Test the hypothesis: "Hit rate persistence is orthogonal to economic viability"
    
    This means: signals can maintain hit rate > 50% while becoming unprofitable.
    """
    hit_rate_survivals = []
    net_sharpes = []
    signal_names = []
    
    for name, results in signal_results.items():
        if 'hit_rate_survival' in results and 'net_sharpe' in results:
            hit_rate_survivals.append(results['hit_rate_survival'])
            net_sharpes.append(results['net_sharpe'])
            signal_names.append(name)
    
    if len(hit_rate_survivals) < 2:
        return GeneralizationClaim(
            claim="Insufficient data for generalization",
            condition="N/A",
            supporting_evidence={},
            counterexamples=[],
            scope="N/A"
        )
    
    # Test orthogonality: correlation between hit_rate_survival and net_sharpe should be low
    correlation = np.corrcoef(hit_rate_survivals, net_sharpes)[0, 1]
    
    # Find signals with high hit rate survival but negative net Sharpe
    orthogonal_signals = []
    for i, name in enumerate(signal_names):
        if hit_rate_survivals[i] > 0.95 and net_sharpes[i] < 0:
            orthogonal_signals.append(name)
    
    if abs(correlation) < 0.3 and len(orthogonal_signals) > 0:
        claim = (
            "Hit rate persistence is orthogonal to economic viability. "
            "Signals can maintain directional correctness (hit rate > 50%) "
            "while becoming economically unviable (negative net Sharpe). "
            "This demonstrates that directional accuracy does not imply profitability."
        )
        
        return GeneralizationClaim(
            claim=claim,
            condition="HitRateSurvival orthogonal to NetSharpe (low correlation)",
            supporting_evidence={
                'correlation': correlation,
                'orthogonal_examples': orthogonal_signals,
                'n_signals_tested': len(signal_names)
            },
            counterexamples=[],
            scope="All signal types"
        )
    else:
        return GeneralizationClaim(
            claim="Hit rate and economic viability show correlation",
            condition=f"Correlation = {correlation:.3f}",
            supporting_evidence={'correlation': correlation},
            counterexamples=[],
            scope="N/A"
        )


def select_best_generalization(signal_results: Dict[str, Dict]) -> GeneralizationClaim:
    """
    Select the best-supported generalization claim.
    
    Returns the claim with strongest empirical support.
    """
    claims = [
        analyze_turnover_threshold_hypothesis(signal_results),
        analyze_horizon_viability_hypothesis(signal_results),
        analyze_hit_rate_orthogonality_hypothesis(signal_results),
    ]
    
    # Score each claim by evidence strength
    scored_claims = []
    for claim in claims:
        if claim.claim.startswith("Insufficient"):
            score = 0
        else:
            # Score based on:
            # - Number of supporting signals
            # - Lack of counterexamples
            # - Clear scope
            n_supporting = len(claim.supporting_evidence.get('supporting_signals', []))
            n_counterexamples = len(claim.counterexamples)
            score = n_supporting - n_counterexamples * 2
        
        scored_claims.append((score, claim))
    
    # Return highest scoring claim
    scored_claims.sort(key=lambda x: x[0], reverse=True)
    return scored_claims[0][1] if scored_claims else claims[0]

