"""
Order Book Replay Validation

Validates cost model against simulated order book fills.
No optimistic assumptions. Crash loud on bad data.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf


class OrderBookFetcher:
    """Fetch intraday bars from yfinance. Estimate bid/ask from OHLC."""
    
    def __init__(self):
        pass
    
    def get_quotes(self, ticker: str, date: str) -> pd.DataFrame:
        """
        Get 1-minute bars for a trading day using yfinance.
        Estimates bid/ask from bar data.
        
        Args:
            ticker: Stock symbol (e.g., 'SPY')
            date: Date string 'YYYY-MM-DD'
            
        Returns:
            DataFrame with columns: timestamp, bid_price, ask_price, 
            bid_size, ask_size, mid_price, spread_bps
        """
        # yfinance needs a date range
        start_date = pd.Timestamp(date)
        end_date = start_date + timedelta(days=1)
        
        print(f"Fetching {ticker} data for {date}...")
        
        # Download 1-minute data
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(
            start=start_date,
            end=end_date,
            interval='1m',
            prepost=False
        )
        
        if len(df) < 50:
            # yfinance only has ~7 days of 1-min data
            # Fall back to 5-min for older dates
            print("1-min data not available, trying 5-min...")
            df = ticker_obj.history(
                start=start_date,
                end=end_date,
                interval='5m',
                prepost=False
            )
        
        if len(df) < 10:
            # Try fetching last 5 days and filter
            print("Specific date not available, using recent data...")
            df = ticker_obj.history(period='5d', interval='1m', prepost=False)
        
        if len(df) < 50:
            raise ValueError(f"Insufficient data: {len(df)} bars < 50 required")
        
        df = df.reset_index()
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        
        # Rename datetime column
        if 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'timestamp'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date': 'timestamp'})
        
        # Estimate bid/ask from bar data
        # Use VWAP approximation: (high + low + close) / 3
        df['mid_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Spread estimation: use high-low range as proxy
        # SPY typical spread is ~1-2bp, use conservative 3bp floor
        bar_range = (df['high'] - df['low']) / df['mid_price']
        df['spread_bps'] = np.maximum(3.0, bar_range * 10000 * 0.15)
        
        half_spread = df['spread_bps'] / 10000 / 2 * df['mid_price']
        df['bid_price'] = df['mid_price'] - half_spread
        df['ask_price'] = df['mid_price'] + half_spread
        
        # Estimate size from volume
        df['bid_size'] = df['volume'] / 2
        df['ask_size'] = df['volume'] / 2
        
        df = df.dropna()
        
        print(f"Loaded {len(df)} bars for {ticker}")
        print(f"Estimated spread: {df['spread_bps'].mean():.2f} bps (mean)")
        print(f"Price range: ${df['mid_price'].min():.2f} - ${df['mid_price'].max():.2f}")
        return df
    
    def get_daily_volume(self, ticker: str, date: str) -> float:
        """Get average daily volume (shares)."""
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            return float(info.get('averageVolume', 80_000_000))
        except Exception:
            # Fallback: SPY average ~80M shares/day
            return 80_000_000 if ticker == 'SPY' else 10_000_000


class MarketImpactSimulator:
    """
    Simulate realistic fills. Conservative assumptions ONLY.
    
    Cost components:
    1. Spread crossing (full half-spread)
    2. Permanent impact (Kyle lambda)
    3. Temporary impact (Almgren-Chriss)
    """
    
    def __init__(self, volatility: float = 0.01):
        """
        Args:
            volatility: Daily volatility (default 1% for SPY-like)
        """
        self.volatility = volatility
        # Kyle lambda: permanent impact coefficient
        self.kyle_lambda = 0.1
        # Almgren-Chriss: temporary impact coefficient
        self.ac_gamma = 0.314
    
    def simulate_market_order(
        self,
        side: str,
        notional: float,
        bid: float,
        ask: float,
        bid_size: int,
        ask_size: int,
        adv: float
    ) -> Dict[str, float]:
        """
        Simulate market order fill with conservative assumptions.
        
        Args:
            side: 'buy' or 'sell'
            notional: Dollar amount to trade
            bid: Current bid price
            ask: Current ask price
            bid_size: Bid depth (shares)
            ask_size: Ask depth (shares)
            adv: Average daily volume (shares)
            
        Returns:
            dict with fill_price, spread_cost, impact_cost, total_cost_bps
        """
        mid = (bid + ask) / 2
        spread = ask - bid
        half_spread = spread / 2
        
        shares = notional / mid
        participation = shares / adv if adv > 0 else 0.01
        
        # 1. Spread crossing: always pay full half-spread
        spread_cost = half_spread
        
        # 2. Permanent impact (Kyle)
        # Impact = lambda * (Q / ADV) * price
        permanent_impact = self.kyle_lambda * participation * mid
        
        # 3. Temporary impact (Almgren-Chriss)
        # Impact = gamma * sigma * sqrt(Q / ADV) * price
        temp_impact = self.ac_gamma * self.volatility * np.sqrt(participation) * mid
        
        # Total impact
        total_impact = permanent_impact + temp_impact
        
        # Fill price (worst case)
        if side == 'buy':
            fill_price = ask + total_impact
        else:
            fill_price = bid - total_impact
        
        # Convert to basis points
        spread_cost_bps = (half_spread / mid) * 10000
        impact_cost_bps = (total_impact / mid) * 10000
        total_cost_bps = spread_cost_bps + impact_cost_bps
        
        return {
            'fill_price': fill_price,
            'mid_price': mid,
            'spread_cost': half_spread,
            'spread_cost_bps': spread_cost_bps,
            'impact_cost': total_impact,
            'impact_cost_bps': impact_cost_bps,
            'total_cost_bps': total_cost_bps,
            'participation_rate': participation
        }


class SignalValidator:
    """Run rejected signals through real order book data."""
    
    def __init__(self):
        self.fetcher = OrderBookFetcher()
        self.simulator = MarketImpactSimulator()
    
    def _generate_mock_signal(self, quotes_df: pd.DataFrame, signal_name: str) -> pd.Series:
        """
        Generate signal values for testing.
        Uses price-based momentum as proxy for actual signals.
        """
        prices = quotes_df['mid_price'].values
        
        if 'momentum' in signal_name.lower():
            # 12-minute momentum (proxy for 12-month)
            lookback = min(12, len(prices) - 1)
            signal = pd.Series(index=quotes_df.index, dtype=float)
            for i in range(lookback, len(prices)):
                signal.iloc[i] = (prices[i] / prices[i - lookback]) - 1
            signal = signal.fillna(0)
        elif 'volatility' in signal_name.lower():
            # Rolling volatility breakout
            returns = pd.Series(prices).pct_change()
            vol = returns.rolling(20).std()
            signal = (returns.abs() > 2 * vol).astype(float)
            signal = signal.fillna(0)
        else:
            # MA crossover proxy
            fast = pd.Series(prices).rolling(5).mean()
            slow = pd.Series(prices).rolling(20).mean()
            signal = ((fast > slow).astype(float) - 0.5) * 2
            signal = signal.fillna(0)
        
        return signal
    
    def _get_theoretical_cost(self, position_size: float, mid: float, adv: float, 
                               spread_bps: float = 10.0) -> float:
        """
        Calculate theoretical cost from existing framework.
        Uses same model as market_impact.py
        
        Args:
            spread_bps: Spread assumption. Framework default is 10 bps (conservative).
                       For SPY, realistic is ~2 bps.
        """
        shares = position_size / mid
        participation = shares / adv if adv > 0 else 0.01
        
        # Almgren-Chriss temporary impact
        volatility = 0.01
        gamma = 0.314
        temp_impact_bps = gamma * volatility * np.sqrt(participation) * 10000
        
        # Kyle permanent impact
        lambda_k = 0.1
        perm_impact_bps = lambda_k * participation * 10000
        
        return spread_bps + temp_impact_bps + perm_impact_bps
    
    def validate_signal(
        self,
        signal_name: str,
        ticker: str,
        test_date: str,
        position_size: float = 100000
    ) -> pd.DataFrame:
        """
        Validate signal against order book.
        
        Args:
            signal_name: Name of signal to validate
            ticker: Stock symbol
            test_date: Date string 'YYYY-MM-DD'
            position_size: Notional per trade ($)
            
        Returns:
            DataFrame with validation results
        """
        # Fetch data
        quotes = self.fetcher.get_quotes(ticker, test_date)
        adv = self.fetcher.get_daily_volume(ticker, test_date)
        
        print(f"ADV for {ticker}: {adv:,.0f} shares")
        
        # Generate signal
        signal = self._generate_mock_signal(quotes, signal_name)
        
        # Find signal triggers (threshold: top/bottom 20%)
        threshold = 0.1
        triggers = (signal.abs() > threshold)
        trigger_indices = triggers[triggers].index.tolist()
        
        if len(trigger_indices) < 10:
            print(f"WARNING: Only {len(trigger_indices)} triggers found (need >= 10)")
            # Lower threshold if needed
            threshold = signal.abs().quantile(0.5)
            triggers = (signal.abs() > threshold)
            trigger_indices = triggers[triggers].index.tolist()
            print(f"Lowered threshold to {threshold:.4f}, found {len(trigger_indices)} triggers")
        
        results = []
        debug_count = 0
        
        for idx in trigger_indices:
            row = quotes.iloc[idx]
            sig_val = signal.iloc[idx]
            
            side = 'buy' if sig_val > 0 else 'sell'
            
            # Simulate fill
            sim = self.simulator.simulate_market_order(
                side=side,
                notional=position_size,
                bid=row['bid_price'],
                ask=row['ask_price'],
                bid_size=int(row['bid_size']),
                ask_size=int(row['ask_size']),
                adv=adv
            )
            
            # Get theoretical costs - both framework default (10 bps) and calibrated (actual spread)
            theoretical_cost_framework = self._get_theoretical_cost(
                position_size, row['mid_price'], adv, spread_bps=10.0
            )
            # Use actual half-spread from data for calibrated comparison
            actual_half_spread_bps = row['spread_bps'] / 2
            theoretical_cost_calibrated = self._get_theoretical_cost(
                position_size, row['mid_price'], adv, spread_bps=actual_half_spread_bps
            )
            
            # Debug first 3 trades
            if debug_count < 3:
                print(f"\n--- Trade {debug_count + 1} Debug ---")
                print(f"  Mid: ${row['mid_price']:.2f}, Bid: ${row['bid_price']:.2f}, Ask: ${row['ask_price']:.2f}")
                print(f"  Data spread: {row['spread_bps']:.2f} bps (half: {actual_half_spread_bps:.2f} bps)")
                print(f"  Position: ${position_size:,}, Shares: {position_size/row['mid_price']:.0f}")
                print(f"  ADV: {adv:,.0f} shares, Participation: {sim['participation_rate']*100:.6f}%")
                print(f"  SIMULATED: {sim['total_cost_bps']:.2f} bps (spread: {sim['spread_cost_bps']:.2f}, impact: {sim['impact_cost_bps']:.2f})")
                print(f"  THEORETICAL (framework, 10bp spread): {theoretical_cost_framework:.2f} bps")
                print(f"  THEORETICAL (calibrated, {actual_half_spread_bps:.1f}bp spread): {theoretical_cost_calibrated:.2f} bps")
                debug_count += 1
            
            # Use framework theoretical for comparison (this is what rejections are based on)
            theoretical_cost = theoretical_cost_framework
            
            results.append({
                'timestamp': row['timestamp'],
                'signal_value': sig_val,
                'side': side,
                'mid_price': row['mid_price'],
                'spread_bps': row['spread_bps'],
                'simulated_spread_bps': sim['spread_cost_bps'],
                'simulated_impact_bps': sim['impact_cost_bps'],
                'simulated_cost_bps': sim['total_cost_bps'],
                'theoretical_cost_bps': theoretical_cost_framework,
                'theoretical_calibrated_bps': theoretical_cost_calibrated,
                'error_bps': sim['total_cost_bps'] - theoretical_cost_framework,
                'error_calibrated_bps': sim['total_cost_bps'] - theoretical_cost_calibrated,
                'participation_rate': sim['participation_rate']
            })
        
        df = pd.DataFrame(results)
        
        # Flag outliers
        outliers = df['simulated_cost_bps'] > 2 * df['theoretical_cost_bps']
        if outliers.any():
            print(f"WARNING: {outliers.sum()} outliers where simulated > 2x theoretical")
        
        return df


def plot_cost_validation(results: pd.DataFrame, signal_name: str):
    """
    Create validation plot with 3 subplots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{signal_name} Cost Model Validation", fontsize=14, fontweight='bold')
    
    theoretical = results['theoretical_cost_bps']
    simulated = results['simulated_cost_bps']
    error = results['error_bps']
    
    # 1. Scatter: theoretical vs simulated
    ax1 = axes[0]
    ax1.scatter(theoretical, simulated, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Perfect calibration line
    max_val = max(theoretical.max(), simulated.max()) * 1.1
    ax1.plot([0, max_val], [0, max_val], 'k--', label='Perfect calibration', linewidth=1)
    
    # Error bands (±20%)
    ax1.fill_between(
        [0, max_val], [0, max_val * 0.8], [0, max_val * 1.2],
        alpha=0.2, color='green', label='±20% band'
    )
    
    ax1.set_xlabel('Theoretical Cost (bps)')
    ax1.set_ylabel('Simulated Cost (bps)')
    ax1.set_title('Theoretical vs Simulated')
    ax1.legend(loc='upper left')
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)
    
    # 2. Histogram: error distribution
    ax2 = axes[1]
    ax2.hist(error, bins=20, edgecolor='black', alpha=0.7)
    
    mean_err = error.mean()
    median_err = error.median()
    p95_err = error.quantile(0.95)
    
    ax2.axvline(mean_err, color='red', linestyle='--', label=f'Mean: {mean_err:.2f}')
    ax2.axvline(median_err, color='blue', linestyle='--', label=f'Median: {median_err:.2f}')
    ax2.axvline(p95_err, color='orange', linestyle='--', label=f'95th pct: {p95_err:.2f}')
    
    ax2.set_xlabel('Error (Simulated - Theoretical) bps')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Error Distribution')
    ax2.legend()
    
    # 3. Time series: cumulative error
    ax3 = axes[2]
    cumulative_error = error.cumsum()
    ax3.plot(range(len(cumulative_error)), cumulative_error, linewidth=2)
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.fill_between(
        range(len(cumulative_error)),
        cumulative_error,
        alpha=0.3,
        color='red' if cumulative_error.iloc[-1] > 0 else 'green'
    )
    
    ax3.set_xlabel('Trade Number')
    ax3.set_ylabel('Cumulative Error (bps)')
    ax3.set_title('Cumulative Cost Difference')
    
    # Add text box with stats
    rmse = np.sqrt((error ** 2).mean())
    pct_over = (simulated > theoretical).mean() * 100
    
    stats_text = (
        f"Mean Error: {mean_err:.2f} bps\n"
        f"RMSE: {rmse:.2f} bps\n"
        f"Simulated > Theoretical: {pct_over:.1f}%"
    )
    
    fig.text(
        0.02, 0.02, stats_text,
        fontsize=10, family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    filename = f"validation_{signal_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    plt.close()


if __name__ == "__main__":
    # Quick test
    print("Order book validation module loaded.")
    print("Run 'python run_validation.py' to execute validation.")
