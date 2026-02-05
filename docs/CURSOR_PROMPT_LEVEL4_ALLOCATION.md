# Cursor prompt: Level-4 allocation extension

Paste this into Cursor to extend or refine the Level-4 allocator.

---

You are a senior quant / PM.

**Context:** This repo has:
- **Tradability**: signal rejection (cost, capacity, regime fragility).
- **Feasibility**: net-edge bounds, zero-alpha boundary, regime table.
- **Shadow trading**: forward execution simulation.
- **Allocation** (Level 4): multi-strategy capital allocation under uncertainty — policy-based weights, throttles, shutdowns.

**Task:** Extend the allocation module so it is *production-grade* for "capital allocation under uncertainty."

**Requirements:**

1. **Inputs**
   - Consume feasibility run outputs: `net_edge_surface.csv`, `zero_alpha_boundary.csv`, `regime_table.csv`.
   - Optionally consume shadow run outputs: realized returns, drawdown, so that *divergence* (realized vs expected) and *current drawdown* are real, not placeholders.
   - Support regime fragility from existing `regime_analysis` (e.g. `compute_regime_sensitivity` → sign_flip, sensitivity_ratio) so each strategy can be tagged fragile.

2. **Policy**
   - Keep allocation as a **policy** (rules), not mean-variance optimization.
   - Rules must include:
     - Net edge below zero-alpha boundary → 0 weight.
     - Capacity cap per strategy (from feasibility or config).
     - Regime fragile → lower max weight (e.g. half).
     - Divergence (|realized - expected|) above threshold → throttle (e.g. half weight) or shutdown (0) above higher threshold.
     - Drawdown above X% → freeze (no new capital) or reduce all weights.
   - Document every rule in `summary.md` and in code.

3. **Outputs**
   - `allocation.csv`: strategy, weight, amount, throttle, shutdown, reason.
   - `summary.md`: decisions table + short "Policy" section (what rules were applied).
   - Optional: `policy.yaml` or `policy.md` that can be versioned (the actual rules and thresholds).

4. **CLI**
   - `python -m tradability.allocation.run --config configs/allocation.yaml [--feasibility-dir DIR] [--shadow-dir DIR]`
   - If `--shadow-dir` is provided, use it to compute realized returns and drawdown for divergence/drawdown rules.

5. **Tests**
   - At least one test: when net_edge_bps < min_net_edge_bps, strategy gets 0 weight.
   - At least one test: when divergence_bps >= shutdown threshold, strategy is shutdown.
   - No lookahead: inputs (e.g. realized return) must be computed only from data up to a given date.

**Constraints**
- No paid data.
- Do not replace the policy-based design with unconstrained mean-variance or Kelly.
- Keep feasibility and shadow modules as-is; only extend allocation and its inputs.

**Deliverables**
- Extended `tradability/allocation/` (inputs from feasibility + optional shadow).
- Updated `configs/allocation.yaml` with all policy thresholds.
- Tests under `tests/test_allocation.py`.
- Short README subsection or doc update for Level-4 allocation.

---

**One sentence to remember:** *"Capital allocation is the problem of deciding how wrong I'm allowed to be."*

---

## How to explain Level 4 in interviews

- **"I don't just ask which strategy works. I ask: given many fragile strategies, how do I allocate so the *system* survives?"**
- **"Allocation is a policy — rules like 'if realized diverges from expected, throttle; if drawdown exceeds X, freeze' — not a single optimal number."**
- **"I use feasibility bounds and regime confidence so higher uncertainty means *less* capital, not more. It's error control, not alpha chasing."**
- **"The goal is least-fragile allocation: survival probability, worst-case drawdown, ability to reallocate when wrong."**
