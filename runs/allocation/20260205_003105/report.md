# Capital allocation under uncertainty — Report

Policy-based allocation. No return maximization; survivability under model error.

## Why capital was allocated conservatively

- Base weight is proportional to **feasible capacity × regime_confidence ÷ uncertainty_penalty**, not to expected return.
- Higher uncertainty score or low regime confidence reduces allocation.
- Strategies in the same correlation group have combined exposure capped.
- We prefer under-allocation to overconfidence; min weight threshold and shutdown rules enforce this.

## What we assumed might be wrong

- **Net edge estimates** may be noisy or decay; feasibility bounds may be optimistic.
- **Regime confidence** is estimated from historical regimes; future regimes may differ.
- **Correlations** may spike in stress; we apply correlation penalty and stress-test correlation → 1.
- **Liquidity** may disappear when needed; we stress-test capacity halved.
- **Parameter instability** and backtest vs forward divergence are penalized via uncertainty and throttle rules.

## Which strategies were intentionally under-allocated

- **volatility_breakout**: low_regime_confidence (magnitude 0.7)

## What would cause immediate shutdown

A strategy is **halted** (zero allocation) if any of:
- Feasibility bound (net edge) ≤ 0
- Regime confidence < 0.2
- Drawdown exceeds 20%
- Realized vs expected divergence ≥ 150.0 bps
- Raw weight below min weight threshold

**Shutdowns in this run:**
- momentum_12_1: below_min_weight_threshold
- volatility_breakout: below_min_weight_threshold
- ma_crossover: below_min_weight_threshold

## Allocation summary

| Strategy | Weight | Amount | Throttle | Shutdown | Reason |
|----------|--------|--------|----------|----------|--------|
| momentum_12_1 | 0.0000 | 0 | False | True | shutdown: below_min_weight_threshold |
| volatility_breakout | 0.0000 | 0 | True | True | shutdown: below_min_weight_threshold |
| ma_crossover | 0.0000 | 0 | False | True | shutdown: below_min_weight_threshold |

## Stress test summary

How allocations change under: 2× estimation error, correlation→1, liquidity shock.

           scenario            strategy  weight_base  weight_stress  amount_base  amount_stress
2x_estimation_error       momentum_12_1          0.0            0.0          0.0            0.0
2x_estimation_error volatility_breakout          0.0            0.0          0.0            0.0
2x_estimation_error        ma_crossover          0.0            0.0          0.0            0.0
 correlation_to_one       momentum_12_1          0.0            0.0          0.0            0.0
 correlation_to_one volatility_breakout          0.0            0.0          0.0            0.0
 correlation_to_one        ma_crossover          0.0            0.0          0.0            0.0
    liquidity_shock       momentum_12_1          0.0            0.0          0.0            0.0
    liquidity_shock volatility_breakout          0.0            0.0          0.0            0.0
    liquidity_shock        ma_crossover          0.0            0.0          0.0            0.0
