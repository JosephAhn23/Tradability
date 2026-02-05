# Alpha Feasibility Bounds — Report

## Assumptions

- Gross edge proxy uses only past information (signal at D, return D to D+1).
- IC-to-return scaling is conservative (scale < 1).
- Costs: spread, fee, slippage, delay scale with turnover; impact scales with sqrt(notional/ADV).
- No lookahead in signal or return alignment.

## Gross edge proxy

- Gross edge bound (bps, annualized): **-0.13**
- Type: ic
- IC: -0.0020045958296722685
- Sample count: 6672
- IC-to-return scale (conservative): 0.5

## Cost model

- Fee: 5.0 bps × turnover
- Spread: 10.0 bps × turnover
- Slippage: 5.0 bps × turnover
- Delay: 2.0 bps × turnover
- Impact: sqrt, k=10.0

## Zero-alpha boundary (where net edge crosses 0)

     aum  turnover_at_zero
  100000               0.0
  250000               0.0
  500000               0.0
 1000000               0.0
 2500000               0.0
 5000000               0.0
10000000               0.0

## Limitations

- Gross edge is an upper-bound proxy, not a forecast.
- Impact model is simplified (single sqrt form).
- Regime buckets depend on in-sample vol/liquidity quantiles.

## Where alpha must disappear

Net edge bound crosses zero at the AUM/turnover combinations above. Beyond that boundary, expected net edge is non-positive under these assumptions.
