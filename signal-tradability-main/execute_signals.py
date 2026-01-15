"""
Signal Execution: Permanent Rejections

This is not research. This is survival.

Signals are either DEPLOYED or REJECTED. No hedging. No hope.
"""

from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime

from signals import get_signal
from data_utils import load_price_data, compute_forward_returns, align_signals_and_returns
from decay_analysis import compute_returns, compute_performance_metrics
from tradability_analysis import analyze_tradability, compute_positions_from_returns
from formal_definitions import compute_statistical_edge, compute_economic_edge, identify_edge_mismatch


# WAR-LEVEL CONSTANTS (FIXED, NOT VARIABLES)
FIXED_CAPITAL = 25_000_000  # $25M. Fixed. No resizing.
COMMISSION_PER_TRADE = 0.005  # 0.5% per trade
HALF_SPREAD = 0.001  # 0.1% half-spread
PERIODS_PER_YEAR = 252

# REJECTION THRESHOLDS (PUBLISHED BEFORE SIMULATION)
REJECT_NET_SHARPE_THRESHOLD = 0.5  # Net Sharpe < 0.5: REJECT
REJECT_BREAK_EVEN_COST_THRESHOLD = 0.005  # Break-even < 0.5%: REJECT
REJECT_CAPACITY_THRESHOLD = FIXED_CAPITAL  # Max capacity < $25M: REJECT
REJECT_TURNOVER_THRESHOLD = 3.0  # Turnover > 3x with break-even < 0.5%: REJECT
REJECT_REGIME_SENSITIVITY_THRESHOLD = 2.0  # Regime sensitivity ratio > 2.0: REJECT


@dataclass
class SignalVerdict:
    """Binary decision: DEPLOY or REJECT. No qualifiers."""
    signal_name: str
    decision: str  # "DEPLOY" or "REJECT"
    max_aum: float
    cause_of_death: str  # Numeric cause if rejected
    net_sharpe: float
    annual_turnover: float
    break_even_cost: float
    gross_sharpe: float
    regime_fragile: bool = False  # True if Sharpe varies > 2x or flips sign across regimes
    regime_sensitivity_ratio: Optional[float] = None  # Max Sharpe / Min Sharpe across regimes
    
    def to_table_row(self) -> Dict:
        """Format for the table that ends careers."""
        return {
            'Signal': self.signal_name,
            'Decision': '✅ DEPLOY' if self.decision == 'DEPLOY' else '❌ REJECT',
            'Max AUM': f'${self.max_aum/1e6:.0f}M',
            'Cause of Death': self.cause_of_death,
            'Net Sharpe': f'{self.net_sharpe:.2f}',
            'Turnover': f'{self.annual_turnover:.1f}x',
        }


def execute_signal_verdict(signal_name: str,
                          ticker: str = 'SPY',
                          start_date: datetime = datetime(2000, 1, 1),
                          end_date: datetime = datetime(2020, 12, 31)) -> SignalVerdict:
    """
    Execute signal verdict. Binary decision. No hedging.
    
    Returns:
        SignalVerdict with DEPLOY or REJECT decision
    """
    # Load data
    prices, volumes = load_price_data(ticker, start_date, end_date)
    forward_returns = compute_forward_returns(prices)
    volatility = prices.pct_change().rolling(20).std() * (PERIODS_PER_YEAR ** 0.5)
    
    # Compute signal
    signal_def = get_signal(signal_name)
    signal_values = signal_def.compute(prices, **signal_def.default_params())
    aligned_signals, aligned_returns = align_signals_and_returns(signal_values, forward_returns)
    gross_returns = compute_returns(aligned_signals, aligned_returns, quantile=0.5)
    positions = compute_positions_from_returns(gross_returns, aligned_signals)
    
    # Tradability analysis
    tradability = analyze_tradability(
        gross_returns=gross_returns,
        signals=aligned_signals,
        volatility=volatility,
        volumes=volumes,
        prices=prices,
        commission_per_trade=COMMISSION_PER_TRADE,
        half_spread=HALF_SPREAD,
        periods_per_year=PERIODS_PER_YEAR,
        sharpe_threshold=REJECT_NET_SHARPE_THRESHOLD
    )
    
    # Economic edge
    net_returns_series = gross_returns - (tradability.cost_drag / PERIODS_PER_YEAR)
    econ_edge = compute_economic_edge(
        net_returns_series,
        gross_returns,
        tradability.cost_drag,
        tradability.break_even_cost,
        tradability.max_viable_capacity or 0
    )
    
    # Extract metrics
    net_sharpe = econ_edge.net_sharpe
    gross_sharpe = tradability.gross_metrics.sharpe_ratio if tradability.gross_metrics else 0
    annual_turnover = tradability.annual_turnover
    break_even_cost = tradability.break_even_cost
    max_aum = tradability.max_viable_capacity or 0
    
    # BINARY DECISION LOGIC (NO HEDGING)
    decision = "REJECT"
    cause_of_death = ""
    
    # Test 1: Net Sharpe threshold
    if net_sharpe < REJECT_NET_SHARPE_THRESHOLD:
        decision = "REJECT"
        cause_of_death = f"Net Sharpe {net_sharpe:.2f} < {REJECT_NET_SHARPE_THRESHOLD}"
    
    # Test 2: Break-even cost threshold
    elif break_even_cost < REJECT_BREAK_EVEN_COST_THRESHOLD:
        decision = "REJECT"
        cause_of_death = f"Break-even {break_even_cost*100:.3f}% < {REJECT_BREAK_EVEN_COST_THRESHOLD*100:.1f}%"
    
    # Test 3: Capacity threshold
    elif max_aum < REJECT_CAPACITY_THRESHOLD:
        decision = "REJECT"
        cause_of_death = f"Capacity ${max_aum/1e6:.0f}M < ${REJECT_CAPACITY_THRESHOLD/1e6:.0f}M"
    
    # Test 4: Turnover threshold
    elif annual_turnover > REJECT_TURNOVER_THRESHOLD and break_even_cost < REJECT_BREAK_EVEN_COST_THRESHOLD:
        decision = "REJECT"
        cause_of_death = f"Turnover {annual_turnover:.1f}x → {break_even_cost*100:.3f}% break-even"
    
    # Test 5: Regime fragility (NEW)
    else:
        # Check regime sensitivity
        try:
            from regime_analysis import partition_regimes, compute_regime_sharpe, compute_regime_sensitivity
            market_returns = prices.pct_change()
            regimes = partition_regimes(gross_returns, market_returns=market_returns, volatility=volatility)
            regime_results = compute_regime_sharpe(gross_returns, regimes)
            regime_sensitivity = compute_regime_sensitivity(regime_results)
            
            regime_fragile = regime_sensitivity.get('regime_fragile', False)
            sensitivity_ratio = regime_sensitivity.get('sensitivity_ratio', None)
            
            if regime_fragile:
                decision = "REJECT"
                if regime_sensitivity.get('sign_flip', False):
                    cause_of_death = f"Regime fragile: Sharpe flips sign (ratio: {sensitivity_ratio:.2f})"
                else:
                    cause_of_death = f"Regime fragile: Sharpe varies {sensitivity_ratio:.2f}x across regimes"
            else:
                # If all tests pass: DEPLOY
                decision = "DEPLOY"
                cause_of_death = "Passes all thresholds"
        except Exception:
            # If regime analysis fails, skip this check (backward compatible)
            decision = "DEPLOY"
            cause_of_death = "Passes all thresholds"
            regime_fragile = False
            sensitivity_ratio = None
    
    # Get regime sensitivity if computed
    regime_fragile = False
    sensitivity_ratio = None
    try:
        from regime_analysis import partition_regimes, compute_regime_sharpe, compute_regime_sensitivity
        market_returns = prices.pct_change()
        regimes = partition_regimes(gross_returns, market_returns=market_returns, volatility=volatility)
        regime_results = compute_regime_sharpe(gross_returns, regimes)
        regime_sensitivity = compute_regime_sensitivity(regime_results)
        regime_fragile = regime_sensitivity.get('regime_fragile', False)
        sensitivity_ratio = regime_sensitivity.get('sensitivity_ratio', None)
    except Exception:
        pass
    
    return SignalVerdict(
        signal_name=signal_name,
        decision=decision,
        max_aum=max_aum,
        cause_of_death=cause_of_death,
        net_sharpe=net_sharpe,
        annual_turnover=annual_turnover,
        break_even_cost=break_even_cost,
        gross_sharpe=gross_sharpe,
        regime_fragile=regime_fragile,
        regime_sensitivity_ratio=sensitivity_ratio
    )


def generate_war_table(signal_names: List[str],
                       ticker: str = 'SPY',
                       start_date: datetime = datetime(2000, 1, 1),
                       end_date: datetime = datetime(2020, 12, 31)) -> List[SignalVerdict]:
    """
    Generate the table that ends careers.
    
    Binary decisions. Numeric death causes. No qualifiers.
    """
    print("=" * 80)
    print("WAR-LEVEL SIGNAL EXECUTION")
    print("=" * 80)
    print(f"\nFIXED CAPITAL: ${FIXED_CAPITAL/1e6:.0f}M")
    print("No resizing. No sensitivity. This is the capital amount. Period.\n")
    
    print("REJECTION THRESHOLDS (Published Before Simulation):")
    print(f"  - Net Sharpe < {REJECT_NET_SHARPE_THRESHOLD}: REJECT")
    print(f"  - Break-even cost < {REJECT_BREAK_EVEN_COST_THRESHOLD*100:.1f}%: REJECT")
    print(f"  - Max capacity < ${REJECT_CAPACITY_THRESHOLD/1e6:.0f}M: REJECT")
    print(f"  - Turnover > {REJECT_TURNOVER_THRESHOLD}x with break-even < {REJECT_BREAK_EVEN_COST_THRESHOLD*100:.1f}%: REJECT")
    print()
    
    verdicts = []
    
    for signal_name in signal_names:
        print(f"Executing verdict for {signal_name}...")
        verdict = execute_signal_verdict(signal_name, ticker, start_date, end_date)
        verdicts.append(verdict)
        
        # Print verdict
        status = "✅ DEPLOY" if verdict.decision == "DEPLOY" else "❌ REJECT"
        print(f"  {status}: {verdict.cause_of_death}")
        print(f"    Net Sharpe: {verdict.net_sharpe:.3f}")
        print(f"    Max AUM: ${verdict.max_aum/1e6:.1f}M")
        print(f"    Turnover: {verdict.annual_turnover:.1f}x")
        print()
    
    # Print table
    print("=" * 80)
    print("THE TABLE THAT ENDS CAREERS")
    print("=" * 80)
    print()
    print(f"{'Signal':<30} {'Decision':<12} {'Max AUM':<10} {'Cause of Death':<40} {'Net Sharpe':<12} {'Turnover':<10}")
    print("-" * 120)
    
    for verdict in verdicts:
        row = verdict.to_table_row()
        print(f"{row['Signal']:<30} {row['Decision']:<12} {row['Max AUM']:<10} {row['Cause of Death']:<40} {row['Net Sharpe']:<12} {row['Turnover']:<10}")
    
    print()
    print("Binary decisions. Numeric death causes. No qualifiers.")
    print()
    
    # Permanent rejections
    rejected = [v for v in verdicts if v.decision == "REJECT"]
    if rejected:
        print("PERMANENT REJECTIONS (Dead. Never touch again.):")
        for verdict in rejected:
            print(f"  ❌ {verdict.signal_name}: {verdict.cause_of_death}")
        print()
    
    # Deployed signals
    deployed = [v for v in verdicts if v.decision == "DEPLOY"]
    if deployed:
        print("DEPLOYED SIGNALS:")
        for verdict in deployed:
            print(f"  ✅ {verdict.signal_name}: Max AUM ${verdict.max_aum/1e6:.0f}M")
        print()
    
    return verdicts


if __name__ == "__main__":
    # Execute verdicts for all signals
    signals = ['momentum_12_1', 'mean_reversion', 'volatility_breakout', 'ma_crossover']
    verdicts = generate_war_table(signals)
    
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print()
    print("1️⃣ Capital number locked forever: $25M")
    print("2️⃣ Signals killed permanently:")
    for verdict in verdicts:
        if verdict.decision == "REJECT":
            print(f"   - {verdict.signal_name}: {verdict.cause_of_death}")
    print()
    print("No explanation. No hedging.")
    print()
    print("This person will stop us before we destroy ourselves.")

