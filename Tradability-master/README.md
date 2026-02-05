# Signal Tradability Framework

Framework for testing if trading signals actually work after costs. Spoiler: most don't.

## The Problem

Backtests lie. A signal with 1.2 Sharpe looks great until you account for:
- Spread costs (you're always buying high, selling low)
- Market impact (your order moves the price against you)  
- Slippage (the price you wanted vs the price you got)

This framework rejects signals that can't survive execution costs. Binary decisions only.

---

## Quick Start

```bash
git clone https://github.com/JosephAhn23/Tradability.git
cd Tradability
pip install -r requirements.txt

python execute_signals.py      # run rejection framework
python run_validation.py       # validate against L2 data
```

---

## Rejection Criteria

A signal gets killed if:

| Test | Threshold | Why |
|------|-----------|-----|
| Net Sharpe | < 0.5 | Below investable threshold |
| Capacity | < $25M | Can't cover fund overhead |
| Turnover | > 3x/year | Costs eat the alpha |
| Regime sensitivity | > 2x variation | Breaks when you need it most |

No hedging. Pass or reject.

---

## Cost Model

```
Total Cost = Commission + Spread + Permanent Impact + Temporary Impact
```

Conservative defaults:
- Commission: 0.5% per trade
- Spread: 10 bps (way above SPY's 1.5 bps, intentionally harsh)
- Impact: Kyle lambda (permanent) + Almgren-Chriss (temporary)

Why so conservative? If a signal dies at 10 bps, it'll definitely die when spreads widen 3x in a vol spike.

---

## Validation

![Execution Cost Model Validation](assets/execution_cost_model_validation.png)

Tested the cost model against L2 order book data (SPY, 676 trades, $100k positions):

| What | Assumed | Actual | Margin |
|------|---------|--------|--------|
| Spread | 10 bps | 1.5 bps | 6.5x |
| Total cost | 10.04 bps | 1.55 bps | 6.5x |

Framework overestimates by 6.5x. That's the point - it's a stress test, not a prediction.

---

## Results

Tested 4 signals. All rejected.

Example - momentum_12_1:
- Backtest Sharpe: 1.2
- After-cost Sharpe: 0.32
- Backtest return: 18%/year
- After-cost return: -1.7%/year

The "alpha" was entirely consumed by 19.7% cost drag from 4x annual turnover.

---

## What Would Pass?

Low turnover + high gross Sharpe + liquid instruments:

| Gross Sharpe | Turnover | At 10 bps | At 120 bps |
|--------------|----------|-----------|------------|
| 1.0 | 4.0x | PASS | REJECT |
| 1.5 | 4.0x | PASS | PASS |
| 0.8 | 0.5x | PASS | PASS |

Run `python sensitivity_sweep.py` for full analysis.

---

## Usage

```python
from execute_signals import execute_signal_verdict
from datetime import datetime

verdict = execute_signal_verdict(
    signal_name='momentum_12_1',
    ticker='SPY',
    start_date=datetime(2000, 1, 1),
    end_date=datetime(2020, 12, 31)
)

print(verdict.decision)        # REJECT
print(verdict.net_sharpe)      # 0.32
print(verdict.cause_of_death)  # Net Sharpe 0.32 < 0.5
```

---

## Limitations

- Small signal universe (4 signals, SPY only)
- No live trading validation
- Conservative assumptions may reject borderline signals
- Static thresholds

---

## References

- Almgren & Chriss (2000) - Optimal execution
- Kyle (1985) - Market impact model  
- Novy-Marx & Velikov (2016) - Trading costs kill anomalies

---

MIT License
