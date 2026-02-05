"""
Experiment Configuration System

Separates engine (reusable backtest/evaluation code) from analysis (notebooks).
This enables:
- Config-driven experiments
- Reproducible research
- Professional maturity

Industry pattern: engine → experiment → analysis
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class CostConfig:
    """Transaction cost configuration."""
    commission_per_trade: float = 0.005  # 0.5% per trade
    half_spread: float = 0.001  # 0.1% half-spread
    vol_impact_coefficient: float = 0.1
    vol_impact_coefficient2: float = 0.0001
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'commission_per_trade': self.commission_per_trade,
            'half_spread': self.half_spread,
            'vol_impact_coefficient': self.vol_impact_coefficient,
            'vol_impact_coefficient2': self.vol_impact_coefficient2,
        }


@dataclass
class CapacityConfig:
    """Capacity estimation configuration."""
    participation_rate: float = 0.01  # 1% of average daily volume
    sharpe_threshold: float = 0.5  # Minimum Sharpe for capacity analysis
    linear_impact: bool = True  # Use linear impact model (simplification)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'participation_rate': self.participation_rate,
            'sharpe_threshold': self.sharpe_threshold,
            'linear_impact': self.linear_impact,
        }


@dataclass
class SignalConfig:
    """Signal configuration."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    quantile: float = 0.5  # Threshold for long/short positions
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'params': self.params,
            'quantile': self.quantile,
        }


@dataclass
class DataConfig:
    """Data configuration."""
    ticker: str = 'SPY'
    start_date: datetime = field(default_factory=lambda: datetime(2000, 1, 1))
    end_date: datetime = field(default_factory=lambda: datetime(2020, 12, 31))
    periods_per_year: int = 252
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticker': self.ticker,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'periods_per_year': self.periods_per_year,
        }


@dataclass
class DiscoveryConfig:
    """Discovery proxy configuration."""
    proxy_name: str = 'conservative'  # Options: 'academic', 'book', 'blog', 'conservative', 'aggressive'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'proxy_name': self.proxy_name,
        }


@dataclass
class UncertaintyConfig:
    """Uncertainty quantification configuration."""
    bootstrap_samples: int = 1000  # Number of bootstrap samples
    confidence_level: float = 0.95  # 95% confidence intervals
    sensitivity_cost_range: tuple = (0.0, 0.02)  # Cost range for sensitivity analysis
    sensitivity_cost_steps: int = 50  # Number of cost levels to test
    rolling_window_size: int = 252  # Rolling window for time-varying analysis
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'bootstrap_samples': self.bootstrap_samples,
            'confidence_level': self.confidence_level,
            'sensitivity_cost_range': self.sensitivity_cost_range,
            'sensitivity_cost_steps': self.sensitivity_cost_steps,
            'rolling_window_size': self.rolling_window_size,
        }


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.
    
    This separates the engine (reusable code) from the analysis (notebooks).
    Configs drive experiments; notebooks only interpret results.
    """
    signals: List[SignalConfig] = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    capacity: CapacityConfig = field(default_factory=CapacityConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    
    # Experiment metadata
    experiment_name: str = "default"
    description: str = ""
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'notes': self.notes,
            'signals': [s.to_dict() for s in self.signals],
            'data': self.data.to_dict(),
            'costs': self.costs.to_dict(),
            'capacity': self.capacity.to_dict(),
            'discovery': self.discovery.to_dict(),
            'uncertainty': self.uncertainty.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        signals = [SignalConfig(**s) for s in config_dict.get('signals', [])]
        data = DataConfig(**config_dict.get('data', {}))
        costs = CostConfig(**config_dict.get('costs', {}))
        capacity = CapacityConfig(**config_dict.get('capacity', {}))
        discovery = DiscoveryConfig(**config_dict.get('discovery', {}))
        uncertainty = UncertaintyConfig(**config_dict.get('uncertainty', {}))
        
        return cls(
            signals=signals,
            data=data,
            costs=costs,
            capacity=capacity,
            discovery=discovery,
            uncertainty=uncertainty,
            experiment_name=config_dict.get('experiment_name', 'default'),
            description=config_dict.get('description', ''),
            notes=config_dict.get('notes', ''),
        )


# Predefined experiment configurations

def get_default_config() -> ExperimentConfig:
    """Default configuration for standard tradability analysis."""
    return ExperimentConfig(
        signals=[
            SignalConfig(name='momentum_12_1', params={}, quantile=0.5),
            SignalConfig(name='mean_reversion', params={}, quantile=0.5),
        ],
        data=DataConfig(
            ticker='SPY',
            start_date=datetime(2000, 1, 1),
            end_date=datetime(2020, 12, 31),
        ),
        costs=CostConfig(
            commission_per_trade=0.005,
            half_spread=0.001,
        ),
        experiment_name="default_tradability",
        description="Standard tradability analysis with momentum and mean reversion signals",
    )


def get_sensitivity_config() -> ExperimentConfig:
    """Configuration for cost sensitivity analysis."""
    config = get_default_config()
    config.experiment_name = "cost_sensitivity"
    config.description = "Cost sensitivity analysis across multiple cost levels"
    config.uncertainty.sensitivity_cost_range = (0.0, 0.02)
    config.uncertainty.sensitivity_cost_steps = 100
    return config


def get_uncertainty_config() -> ExperimentConfig:
    """Configuration for uncertainty quantification."""
    config = get_default_config()
    config.experiment_name = "uncertainty_quantification"
    config.description = "Bootstrap confidence intervals and sensitivity analysis"
    config.uncertainty.bootstrap_samples = 1000
    config.uncertainty.confidence_level = 0.95
    return config


def get_counterexample_config() -> ExperimentConfig:
    """Configuration for testing counterexamples (when claims don't hold)."""
    config = get_default_config()
    config.experiment_name = "counterexamples"
    config.description = "Testing when claims do NOT hold (low turnover, high capacity, etc.)"
    # Add low-turnover signals
    config.signals.append(SignalConfig(name='ma_crossover', params={}, quantile=0.5))
    # Use lower costs to test break-even
    config.costs.commission_per_trade = 0.001  # 0.1% (very low)
    config.costs.half_spread = 0.0005  # 0.05% (very low)
    return config



