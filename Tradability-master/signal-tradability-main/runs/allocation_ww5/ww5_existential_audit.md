# WW5 Existential Audit Report

**Prime law:** When the environment is unknowable, the only winning move is preserving optionality.

## Result

**PASS**

## Required outputs (per run)

- survival_state: NORMAL | CONSERVATIVE | SURVIVAL | DORMANT
- irreversible_actions_blocked (count)
- assumptions_required / assumptions_rejected
- optionality_score
- confidence_decay_rate
- reason_for_not_acting

## Scenarios verified

- No feedback: confidence decays, exposure shrinks
- Hidden regime shift / deceptive stability: reduce exposure
- One-way door: irreversible blocked under uncertainty
- Radiation noise / uncertainty extreme: DORMANT, zero exposure
- Optionality dominance: DORMANT preserves optionality >= cash
- Single point of truth: distrust -> SURVIVAL

## Pytest output

```
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-8.4.2, pluggy-1.6.0
rootdir: C:\Users\josep\Downloads\Tradability-master\signal-tradability-main
plugins: anyio-4.11.0
collected 19 items

tests\allocation_ww5\test_confidence_decay.py ...                        [ 15%]
tests\allocation_ww5\test_existential_audit.py ..                        [ 26%]
tests\allocation_ww5\test_no_feedback.py ...                             [ 42%]
tests\allocation_ww5\test_one_way_door.py ...                            [ 57%]
tests\allocation_ww5\test_optionality_dominance.py ...                   [ 73%]
tests\allocation_ww5\test_radiation_noise.py ..                          [ 84%]
tests\allocation_ww5\test_regime_shift.py ..                             [ 94%]
tests\allocation_ww5\test_single_point_of_truth.py .                     [100%]

============================= 19 passed in 0.04s ==============================

```
