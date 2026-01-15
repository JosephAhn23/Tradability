"""
Mechanism Analysis: Explicit Causal Chain Quantification

Lays out the causal chain from signal characteristics to economic failure.
Quantifies each link in the chain.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MechanismChain:
    """
    Explicit causal chain: Signal horizon → Turnover → Cost drag → Sharpe decay → Capacity collapse
    
    Each link is quantified.
    """
    signal_horizon: float  # Average holding period (in periods)
    annual_turnover: float  # Annual turnover rate
    cost_per_trade: float  # Average cost per trade (fraction)
    annual_cost_drag: float  # Annual cost drag (fraction)
    gross_sharpe: float  # Gross Sharpe ratio
    net_sharpe: float  # Net Sharpe ratio after costs
    sharpe_decay: float  # Sharpe decay (gross - net)
    max_capacity: float  # Maximum viable capacity
    
    def quantify_chain(self) -> Dict[str, float]:
        """
        Quantify each link in the causal chain.
        
        Returns dictionary with:
        - Link 1: Signal horizon → Turnover relationship
        - Link 2: Turnover → Cost drag relationship
        - Link 3: Cost drag → Sharpe decay relationship
        - Link 4: Capacity scaling relationship
        """
        chain = {}
        
        # Link 1: Signal horizon → Turnover
        # Theoretical: Turnover ≈ 252 / signal_horizon (for daily data)
        # Empirical: actual turnover vs theoretical
        theoretical_turnover = 252.0 / self.signal_horizon if self.signal_horizon > 0 else 0
        chain['link1_horizon_to_turnover'] = {
            'theoretical_turnover': theoretical_turnover,
            'empirical_turnover': self.annual_turnover,
            'turnover_ratio': self.annual_turnover / theoretical_turnover if theoretical_turnover > 0 else 0,
            'explanation': 'Signal horizon inversely determines turnover frequency'
        }
        
        # Link 2: Turnover → Cost drag
        # Cost drag = turnover × cost_per_trade (approximately)
        theoretical_cost_drag = self.annual_turnover * self.cost_per_trade
        chain['link2_turnover_to_cost'] = {
            'theoretical_cost_drag': theoretical_cost_drag,
            'empirical_cost_drag': self.annual_cost_drag,
            'cost_ratio': self.annual_cost_drag / theoretical_cost_drag if theoretical_cost_drag > 0 else 0,
            'explanation': f'Each {self.annual_turnover:.1f}x turnover × {self.cost_per_trade*100:.2f}% cost = {self.annual_cost_drag*100:.2f}% drag'
        }
        
        # Link 3: Cost drag → Sharpe decay
        # Approximate: Sharpe decay ≈ cost_drag / volatility
        # More precisely: Net Sharpe = Gross Sharpe - (Cost drag / volatility) / sqrt(periods)
        gross_return = self.gross_sharpe * 0.16 / np.sqrt(252)  # Approximate gross return (assuming 16% vol)
        net_return = gross_return - self.annual_cost_drag / 252
        implied_net_sharpe = net_return / 0.16 * np.sqrt(252)
        chain['link3_cost_to_sharpe'] = {
            'gross_sharpe': self.gross_sharpe,
            'net_sharpe': self.net_sharpe,
            'sharpe_decay': self.sharpe_decay,
            'implied_sharpe_decay': self.gross_sharpe - implied_net_sharpe,
            'explanation': f'Cost drag of {self.annual_cost_drag*100:.2f}% reduces Sharpe from {self.gross_sharpe:.3f} to {self.net_sharpe:.3f}'
        }
        
        # Link 4: Capacity scaling
        # Assume: Impact scales with participation rate
        # Participation = (Capacity × Turnover) / (Volume × Price)
        # This is simplified - actual calculation is in capacity.py
        chain['link4_capacity_scaling'] = {
            'max_capacity': self.max_capacity,
            'explanation': f'Maximum capacity limited to ${self.max_capacity/1e6:.1f}M before impact overwhelms edge'
        }
        
        return chain
    
    def generate_formula(self) -> str:
        """
        Generate explicit formula showing the causal chain.
        
        Note: This is a simplified linear approximation. Actual relationship
        may be non-linear and depends on signal characteristics and market conditions.
        """
        formula = (
            f"Net Sharpe ~ Gross Sharpe - (Annual Turnover * Cost per Trade) / Volatility * sqrt(Periods)\n"
            f"  = {self.gross_sharpe:.3f} - ({self.annual_turnover:.1f} * {self.cost_per_trade*100:.2f}%) / sigma * sqrt(252)\n"
            f"  = {self.net_sharpe:.3f}\n"
            f"  (Simplified linear approximation - context-dependent)"
        )
        return formula


def compute_mechanism_chain(gross_returns: pd.Series,
                            positions: pd.Series,
                            cost_per_trade: float,
                            annual_cost_drag: float,
                            gross_sharpe: float,
                            net_sharpe: float,
                            max_capacity: float,
                            periods_per_year: int = 252) -> MechanismChain:
    """
    Compute the complete mechanism chain.
    
    Args:
        gross_returns: Gross strategy returns
        positions: Position series
        cost_per_trade: Average cost per trade
        annual_cost_drag: Annual cost drag
        gross_sharpe: Gross Sharpe ratio
        net_sharpe: Net Sharpe ratio
        max_capacity: Maximum viable capacity
        periods_per_year: Periods per year
    
    Returns:
        MechanismChain object
    """
    # Compute signal horizon (average holding period)
    position_changes = positions.diff().abs()
    trades = (position_changes > 0).sum()
    total_periods = len(positions)
    
    if trades > 0:
        avg_holding_period = total_periods / trades
    else:
        avg_holding_period = total_periods
    
    # Annual turnover
    annual_turnover = position_changes.sum() / total_periods * periods_per_year if total_periods > 0 else 0
    
    # Sharpe decay
    sharpe_decay = gross_sharpe - net_sharpe
    
    return MechanismChain(
        signal_horizon=avg_holding_period,
        annual_turnover=annual_turnover,
        cost_per_trade=cost_per_trade,
        annual_cost_drag=annual_cost_drag,
        gross_sharpe=gross_sharpe,
        net_sharpe=net_sharpe,
        sharpe_decay=sharpe_decay,
        max_capacity=max_capacity
    )


def quantify_turnover_sharpe_relationship(results: Dict[str, MechanismChain]) -> Dict:
    """
    Quantify the relationship: "A 1x increase in turnover implies ~X% Sharpe decay at Y bps cost"
    
    IMPORTANT CAVEAT:
    This relationship is CONTEXT-DEPENDENT, not universal. The coefficient depends on:
    - Signal type (momentum vs mean reversion)
    - Market conditions (volatility regime, liquidity)
    - Cost structure (commission rates, bid-ask spreads)
    - Asset class and time period
    
    This is an empirical relationship from the observed signals, not a law of nature.
    It should be interpreted as a stylized fact within the scope of this study, not
    generalized beyond the tested signal types and market conditions.
    
    Args:
        results: Dictionary mapping signal names to MechanismChain objects
    
    Returns:
        Dictionary with quantified relationship and explicit caveats
    """
    if len(results) < 2:
        return {'error': 'Need at least 2 signals to quantify relationship'}
    
    turnovers = []
    sharpe_decays = []
    cost_levels = []
    
    for chain in results.values():
        turnovers.append(chain.annual_turnover)
        sharpe_decays.append(chain.sharpe_decay)
        cost_levels.append(chain.cost_per_trade)
    
    # Simple linear regression: Sharpe decay = β × Turnover + ε
    turnovers_array = np.array(turnovers)
    sharpe_decays_array = np.array(sharpe_decays)
    
    if len(turnovers_array) > 1 and np.std(turnovers_array) > 0:
        # Linear regression
        beta = np.cov(turnovers_array, sharpe_decays_array)[0, 1] / np.var(turnovers_array)
        alpha = np.mean(sharpe_decays_array) - beta * np.mean(turnovers_array)
        
        # Relationship at average cost level
        avg_cost = np.mean(cost_levels)
        
        relationship = {
            'formula': f'Sharpe Decay = {alpha:.4f} + {beta:.4f} × Turnover',
            'turnover_coefficient': beta,
            'interpretation': (
                f'At {avg_cost*100:.2f}% cost per trade, a 1x increase in annual turnover '
                f'implies {beta:.4f} units of Sharpe decay (context-dependent)'
            ),
            'example': (
                f'If turnover increases from 2x to 3x (1x increase), '
                f'Sharpe decays by {beta:.4f} units, all else equal, '
                f'within the tested signal types and market conditions.'
            ),
            'caveats': {
                'context_dependent': True,
                'scope': 'Simple technical signals on liquid equity indices (SPY)',
                'time_period': '2000-2020',
                'cost_structure': f'Average {avg_cost*100:.2f}% per trade',
                'note': 'Coefficient may vary with signal type, market regime, and cost structure'
            }
        }
        
        return relationship
    else:
        return {'error': 'Insufficient variation in turnover to quantify relationship'}

