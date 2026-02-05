# Shadow trading run summary

**Forward walk-forward execution simulation.** No broker; deterministic fills.

## Fill assumption

Orders generated at close(D) are filled at open(D+1). Fill price = open(D+1) × (1 ± (spread_bps + slippage_bps)/10000).

## Feasibility

Halt when feasibility bounds are violated (config: halt_on_feasibility_violation).
