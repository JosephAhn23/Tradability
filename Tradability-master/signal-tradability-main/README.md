# Signal Tradability Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)](https://github.com/JosephAhn23/Tradability)

> **"Real quant research isn't about discovering alpha. It's about knowing exactly why it won't survive."**

A rigorous quantitative research framework that evaluates whether trading signals are **economically tradable** (profitable after real-world costs), not just statistically significant.

## 🎯 What Problem Does This Solve?

Most quant research focuses on finding signals with good **statistical properties** (high Sharpe, positive returns). But in reality, many signals that look great on paper are **untradable** because:

- Transaction costs eat all the profits
- Market impact prevents scaling to meaningful capital
- Signals only work in specific market regimes
- Hidden costs (slippage, spreads) kill profitability

This framework **kills bad ideas before they kill capital**.

## 🎯 Core Purpose

This framework answers the critical question: **"If I had $25M capital, which signals should I actually trade?"**

It rejects signals that:
- Look good statistically but fail after transaction costs
- Can't scale to meaningful capital
- Are too fragile across market regimes
- Have hidden costs that kill profitability

## ✨ Key Features

### 1. **War-Level Testing**
Rigorous adversarial tests that make binary DEPLOY/REJECT decisions:
- Fixed capital ($25M, no resizing)
- Pre-declared thresholds (no optimization after the fact)
- Constraint-based decisions (not performance metrics)
- Automatic failure detection (assertions that exit non-zero)

### 2. **Proper Market Impact Models**
- **Almgren-Chriss (2000)**: Temporary + permanent impact with citations
- **Kyle (1985)**: Lambda model for market impact
- Realistic cost modeling (commissions, spreads, slippage)

### 3. **Statistical Rigor**
- Confidence intervals on all Sharpe ratios (Lo 2002)
- Information Coefficient (IC) with p-values
- Multiple testing correction (Bonferroni, FDR)
- Out-of-sample validation (10-year test period)

### 4. **Comprehensive Analysis**
- **Drawdown Analysis**: Cost-adjusted, recovery time, CVaR
- **Sharpe vs AUM Curves**: Optimal AUM calculation
- **Regime Analysis**: 4 regimes (Bull/Bear × High/Low Vol)
- **Sensitivity Analysis**: 11 scenarios (costs ±50%, spreads +500%, etc.)

### 5. **Cryptographic Attestation**
- SHA256 hashes of all code files
- Data fingerprints
- Output hashes
- Git commit tracking
- Proves reproducibility

## 📁 Project Structure

```
├── war_test_ii.py              # War-level testing framework
├── execute_signals.py          # Binary DEPLOY/REJECT decisions
├── tradability_analysis.py     # Core tradability analysis
├── market_impact.py            # Almgren-Chriss, Kyle models
├── statistical_rigor.py        # CI, IC, multiple testing
├── drawdown_analysis.py         # Drawdown metrics
├── sharpe_vs_aum.py            # Sharpe vs AUM curves
├── regime_analysis.py          # Regime partitioning
├── comprehensive_validation.py # Complete validation suite
├── create_attestation.py       # Cryptographic attestation
└── ...
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/JosephAhn23/Tradability.git
cd Tradability

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

**Run war-level tests:**
```bash
python war_test_ii.py
```

**Execute signal verdicts:**
```bash
python execute_signals.py
```

**Run comprehensive validation:**
```bash
python comprehensive_validation.py
```

**Generate attestation:**
```bash
python create_attestation.py
```

### Example: Evaluate a Signal

```python
from execute_signals import execute_signal_verdict
from datetime import datetime

# Evaluate momentum signal
verdict = execute_signal_verdict(
    signal_name='momentum_12_1',
    ticker='SPY',
    start_date=datetime(2000, 1, 1),
    end_date=datetime(2020, 12, 31)
)

print(f"Decision: {verdict.decision}")
print(f"Max AUM: ${verdict.max_aum/1e6:.0f}M")
print(f"Cause: {verdict.cause_of_death}")
```

### Shadow Trading (Forward Execution Simulation)

**Strategies are evaluated using a forward, walk-forward execution simulator** that generates orders based only on information available at decision time, applies realistic cost and impact models, and logs realized performance.

This mirrors **pre-deployment validation used in professional quant research**, where strategies are run in shadow mode before any capital is allocated.

- **No broker, no identity dependency** — deterministic execution simulation only.
- **No lookahead** — `as_of_date` enforced; signals and orders use only past data.
- **Documented fill assumption** — e.g. orders at close(D) filled at open(D+1) with spread/slippage/impact.
- **Full logging** — orders, fills, equity, positions, diagnostics under `runs/shadow/<run_id>/`.
- **Feasibility halt** — run can stop when feasibility bounds are violated (e.g. AUM > capacity, net edge below threshold).

**Goal:** Validate tradability and rejection criteria under live-like constraints, not to demonstrate profitability.

**Run:**

```bash
cd signal-tradability-main
python -m tradability.shadow.run --config configs/shadow.yaml
```

Output: `orders.csv`, `fills.csv`, `positions.csv`, `equity.csv`, `diagnostics.csv`, `summary.md` in `runs/shadow/<run_id>/`.

*Optional: The repo also includes a broker paper-trading path (`tradability.paper`) for those who want to send orders to a broker’s paper account; the canonical execution validation for this framework is shadow trading.*

### Alpha Feasibility Bounds

The **Alpha Feasibility Bounds** module estimates **upper-bound feasible alpha** and the **zero-alpha boundary** (where expected net edge crosses zero). It does **not** predict alpha; it answers:

- Is the net-edge bound positive under realistic cost assumptions?
- At what turnover does net edge cross zero?
- At what AUM does net edge cross zero? (capacity boundary)
- How does this vary by regime (vol/liquidity bins)?

**What it is:** A conservative bound: `net_edge_bound ≈ gross_edge_proxy - (spread + fee + slippage + impact + delay)`. Uses free OHLCV (yfinance); no lookahead in the gross-edge proxy (signal at D, return D→D+1). Fully reproducible and auditable.

**How to run:**

```bash
cd signal-tradability-main
python -m tradability.feasibility.run --config configs/feasibility_example.yaml
```

**Example outputs** (under `runs/feasibility/<run_id>/`):

- `net_edge_surface.csv` — net edge (bps) for each (AUM, turnover) pair
- `zero_alpha_boundary.csv` — for each AUM, turnover at which net edge = 0
- `regime_table.csv` — net edge and sample days per regime bucket
- `report.md` — assumptions, cost model, limitations, “where alpha must disappear”
- Plots: net edge vs AUM (by turnover), net edge vs turnover (by AUM), zero-alpha boundary curve

You get a defensible statement: *“Net edge bound crosses zero at ~X AUM or ~Y turnover under these assumptions.”*

### Level 4: Capital allocation under uncertainty

If Levels 1–3 ask *"does this strategy work?"*, **Level 4 asks:** *"Given many fragile strategies, how do I allocate capital so the system survives?"*

**A policy-based capital allocator** that sizes and throttles strategies using feasibility bounds, regime confidence, and model uncertainty — **prioritizing survivability over return maximization.** No mean-variance or Kelly; allocation is a *policy* (throttles, caps, shutdown rules).

- **Base weight:** ∝ feasible capacity × regime_confidence ÷ uncertainty_penalty (no return optimization).
- **Uncertainty:** reduces allocation when feasibility is near zero, divergence is high, or uncertainty_score is high.
- **Correlation control:** strategies in the same `correlation_group` have combined exposure capped.
- **Shutdown (logged):** feasibility ≤ 0, regime_confidence &lt; min, drawdown &gt; limit, divergence &gt; threshold.
- **Stress tests:** 2× estimation error, correlation→1, liquidity shock; report how allocations change.

**Run:** `python -m tradability.allocation.run --config configs/allocation.yaml`  
Optional: `--feasibility-dir runs/feasibility/<run_id>` so net edge/capacity come from feasibility.

**Outputs** (in `runs/allocation/<run_id>/`): `allocations.csv`, `throttles.csv`, `stress_results.csv`, `report.md` (with sections: why conservative, what we assumed wrong, under-allocated, what causes shutdown).

**WW4 survival controller (safety-critical mode):** When reality is hostile (sensor poisoning, blackout, infra failure), the allocator can run in **survival mode** with a state machine (NORMAL → CAUTION → DANGER → SURVIVAL → LOCKDOWN), multi-source consensus (disagree ⇒ worst-case), and risk-token budget (2-of-3 in SURVIVAL). Invariants: exposure and concentration are non-increasing as hazard increases. Tests: `pytest tests/allocation_ww4/`. Report: `python tests/allocation_ww4/ww4_survival_report.py` → `runs/allocation_ww4/ww4_survival_report.md`.

**WW5 cosmic survival (existential mode):** No ground truth, delayed comms, hostile physics. Objective: **existence across time** — optionality preserved, confidence decays without confirmation, irreversible actions blocked, one-way doors forbidden. States: NORMAL / CONSERVATIVE / SURVIVAL / **DORMANT** (intentional inaction). Outputs: `survival_state`, `irreversible_actions_blocked`, `assumptions_required` / `assumptions_rejected`, `optionality_score`, `confidence_decay_rate`, `reason_for_not_acting`. Tests: `pytest tests/allocation_ww5/`. Report: `python tests/allocation_ww5/ww5_report.py` → `runs/allocation_ww5/ww5_existential_audit.md`.

**WW5-B bluff detection (anti-hallucination):** Species-survival grade. Bluffing = pretending error does not exist. **Invariants:** confidence may only increase with new evidence; silence over certainty when evidence insufficient; every non-zero action lists assumptions; unknowable → explicit UNKNOWN, no action. **Bluff audit:** `confidence_before`/`after`, `evidence_added`, `assumptions_required`/`verified`/`unverified`, `reason_for_action`/`reason_for_inaction`, `bluff_risk_score` (increases with uncertainty). **Upgrades:** (1) **EvidenceLedger** — evidence earned from timestamped events (confirmations, threshold crossings, contradictions); confidence can only rise if ledger has new confirmations since last tick. (2) **Auto-UNKNOWN** — `compute_unknown_conditions()` triggers DORMANT from telemetry stale, estimator disagreement, drift persist, feasibility low + uncertainty high, subsystem failure (no manual flag in production). (3) **Exposure monotone** in hazard and bluff risk: `max_gross_from_hazard_and_bluff(hazard_level, bluff_risk)`; grid tests. (4) **Sequence adversarial sim** — 80-step runs with evidence decay/contradiction and auto-UNKNOWN; no wild oscillation. **Tests:** precision trap, narrative/authority, say-I-don’t-know, counterfactual, last-chance, auto-unknown, evidence ledger, exposure monotone, sequence. `pytest tests/allocation_ww5b/`. Report: `python tests/allocation_ww5b/ww5b_report.py` → `runs/allocation_ww5b/ww5b_bluff_audit.md`.

## 📊 Example Output

### War Table (Binary Decisions)

| Signal | Decision | Max AUM | Cause of Death | Net Sharpe | Turnover |
|--------|----------|---------|----------------|------------|----------|
| momentum_12_1 | ❌ REJECT | $6M | Net Sharpe 0.32 < 0.5 | 0.32 | 2.1x |
| volatility_breakout | ❌ REJECT | $0M | Capacity $0M < $25M | -0.82 | 4.5x |
| ma_crossover | ❌ REJECT | $14M | Break-even 0.78% < 1.0% | 0.45 | 3.2x |

### Validation Results

**Out-of-Sample (10-Year: 2015-2024):**
- ✅ Accuracy: **100%** (4/4 signals correctly rejected)
- ✅ Test Sharpe with CI: -0.50 [-0.54, -0.45]
- ✅ 2,200+ observations per signal

**Regime Analysis:**
- ⚠️ momentum_12_1: Sign flip detected (Sharpe -0.77 to +1.10)
- ⚠️ Regime fragile: TRUE (works only in Bear/Low Vol)

### Validation Results

**Out-of-Sample (10-Year: 2015-2024):**
- Accuracy: 100% (4/4 signals correctly rejected)
- Test Sharpe with CI: -0.50 [-0.54, -0.45]
- 2,200+ observations per signal

**Regime Analysis:**
- momentum_12_1: Sign flip detected (Sharpe -0.77 to +1.10)
- Regime fragile: TRUE (works only in Bear/Low Vol)

## 🔬 Methodology

### Decision Framework

Signals are **REJECTED** if:
- Break-even cost < 1.0%
- Max capacity < $25M
- Turnover > 3x
- Regime sensitivity > 2x (or sign flip)
- Cost drag > 5%

**No hedging. No qualifiers. Binary decisions only.**

### Cost Model

1. **Explicit Costs:**
   - Commission: 0.5% per trade
   - Bid-ask spread: 0.1% half-spread

2. **Market Impact:**
   - Almgren-Chriss: Temporary + permanent impact
   - Kyle: Lambda model
   - Participation rate: 1% of daily volume

3. **Slippage:**
   - Volatility-adjusted
   - Time-of-day effects

### Validation

- **Out-of-Sample**: Train on 2000-2014, test on 2015-2024
- **Regime Analysis**: 4 regimes (Bull/Bear × High/Low Vol)
- **Sensitivity**: 11 scenarios (costs, spreads, liquidity)
- **Integrity Tests**: Cost linearity, turnover identity, shuffle test

## 📈 Results

### Current Status

- ✅ All 17 requested features implemented
- ✅ 100% accuracy on 10-year out-of-sample test
- ✅ All signals correctly rejected (none are tradable)
- ✅ Regime analysis reveals critical fragility
- ✅ Production-ready for signal rejection decisions

### Key Findings

1. **All signals are untradable** under realistic cost assumptions
2. **Regime fragility** is a critical issue (signals work in some regimes, fail in others)
3. **Break-even costs** are often below realistic transaction costs
4. **Capacity constraints** limit scalability to meaningful capital

### Performance Metrics

| Metric | Value |
|--------|-------|
| Out-of-Sample Accuracy | 100% |
| Test Period | 10 years (2015-2024) |
| Signals Tested | 4 |
| Signals Rejected | 4 |
| Regime Fragile Signals | 1 (momentum_12_1) |

## 🛠️ Requirements

- **Python 3.8+**
- pandas
- numpy
- scipy
- yfinance (for data)

See `requirements.txt` for full list.

### Dependencies

```bash
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0
yfinance>=0.1.70
```

## 📝 Philosophy

This framework is designed to:
- **Kill bad ideas before they kill capital**
- Use **fixed thresholds** (no optimization)
- Make **binary decisions** (no hedging)
- Prove **reproducibility** (cryptographic attestation)
- Test **adversarially** (war-level tests)

It's designed to be **wrong loudly, not quietly**.

## 🔒 Reproducibility

The framework includes cryptographic attestation:
- SHA256 hashes of all code files
- Data fingerprints
- Output hashes
- Git commit tracking

Run `python create_attestation.py` to generate `ATTESTATION.json`.

## 📚 References

- **Almgren & Chriss (2000)**: "Optimal execution of portfolio transactions"
- **Kyle (1985)**: "Continuous auctions and insider trading"
- **Lo (2002)**: "The statistics of Sharpe ratios"

## ⚠️ Limitations

1. **Production Validation**: Paper trading (broker-simulated) is supported; live capital is not.
2. **Parameter Calibration**: Parameters not calibrated to current markets.
3. **Regime Adaptation**: Framework rejects regime-fragile signals but doesn't adapt (deploy only in favorable regimes).

## 🤝 Contributing

This is a research framework. Contributions should focus on:
- Improving cost models
- Adding new validation tests
- Extending regime analysis
- Production trading validation

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Roadmap

- [ ] Production trading validation
- [ ] Parameter calibration to current markets
- [ ] Regime-aware deployment logic
- [ ] Additional asset classes (bonds, commodities, FX)
- [ ] Machine learning signal support

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Joseph Ahn**

- GitHub: [@JosephAhn23](https://github.com/JosephAhn23)
- Repository: [Tradability](https://github.com/JosephAhn23/Tradability)

## 🙏 Acknowledgments

Built with the philosophy: **"Real quant research isn't about discovering alpha. It's about knowing exactly why it won't survive."**

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=JosephAhn23/Tradability&type=Date)](https://star-history.com/#JosephAhn23/Tradability&Date)

---

**Status**: Production-ready for signal rejection decisions. Framework is rigorous, validated, and honest about limitations.

## 📞 Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**Built with the philosophy**: *"Real quant research isn't about discovering alpha. It's about knowing exactly why it won't survive."*

