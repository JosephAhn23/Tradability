# Level 4: Capital allocation under uncertainty

**Allocation is a policy (rules), not a single optimal number.**

## Decisions

| Strategy | Weight | Amount | Throttle | Shutdown | Reason |
|----------|--------|--------|----------|----------|--------|
| momentum_12_1 | 0.0000 | 0 | True | True | regime_fragile; non_positive_net_edge; below_min_weight |
| volatility_breakout | 0.0000 | 0 | True | True | regime_fragile; non_positive_net_edge; below_min_weight |
| ma_crossover | 0.0000 | 0 | True | True | regime_fragile; non_positive_net_edge; below_min_weight |

## Policy

- Higher uncertainty / regime fragility → less capital.
- Divergence (realized vs expected) → throttle or shutdown.
- Drawdown exceeds threshold → freeze or reduce.
- Net edge below zero-alpha boundary → no allocation.
