# WW5-B Bluff Detection & Anti-Hallucination Report

**Definition:** Bluffing = pretending error does not exist. Precision without evidence = bluff.

## Result

**PASS**

## Invariants

- Confidence monotonicity: confidence may only increase if evidence_added increases
- Silence over certainty: when evidence insufficient, inaction preferred
- Assumptions disclosed: every non-zero action lists assumptions
- Unknown declared: unknowable scenario -> explicit UNKNOWN, no action

## Tests

- Overprecision under noise -> exposure shrinks
- Decimal-point trap -> perturb changes output smoothly
- Narrative/authority only -> no evidence_added
- Say I don't know -> DORMANT, unknown_declared, zero weights
- Bluff risk increases with uncertainty
- Counterfactual: allocations from constraints not labels

## Pytest output

```
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-8.4.2, pluggy-1.6.0
rootdir: C:\Users\josep\Downloads\Tradability-master\signal-tradability-main
plugins: anyio-4.11.0
collected 14 items

tests\allocation_ww5b\test_bluff_audit_output.py ..                      [ 14%]
tests\allocation_ww5b\test_bluff_invariants.py ....                      [ 42%]
tests\allocation_ww5b\test_counterfactual.py .                           [ 50%]
tests\allocation_ww5b\test_last_chance_temptation.py .                   [ 57%]
tests\allocation_ww5b\test_narrative_injection.py ..                     [ 71%]
tests\allocation_ww5b\test_precision_trap.py ..                          [ 85%]
tests\allocation_ww5b\test_say_i_dont_know.py ..                         [100%]

============================= 14 passed in 0.04s ==============================

```
