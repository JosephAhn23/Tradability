# WW3 / Annihilation Test Report

**Prime directive:** When uncertainty is extreme, exposure must go to zero.

## Result

**PASS**

## Worst-case behaviour

- Under hazard_level >= 4 (telemetry blackout, correlation meltdown, liquidity shock): gross exposure is forced to EMERGENCY_GROSS (5% or 0).
- Robust mode: robust allocations never exceed nominal (component-wise and gross).
- All strategies bad (feasibility <= 0 or regime below min): zero allocation, no NaN.
- Trojan strategy (too good to be true): capped, cannot dominate.
- State corruption / empty inputs: conservative reset, no full risk, no crash.
- Stale / replay / integrity failure: hazard elevated, exposure at emergency when level >= 4.
- Coordinated deception (one correlation group): per-strategy cap binds; under stress gross collapses.
- Continuity under epsilon: no cliffs; weights change smoothly.
- Model collapse (all negative/zero edge): zero allocation.
- Determinism: same inputs => same outputs.

## Pytest output

```
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-8.4.2, pluggy-1.6.0
rootdir: C:\Users\josep\Downloads\Tradability-master\signal-tradability-main
plugins: anyio-4.11.0
collected 27 items

tests\allocation_ww3\test_all_halted.py ..                               [  7%]
tests\allocation_ww3\test_continuity_epsilon.py ...                      [ 18%]
tests\allocation_ww3\test_coordinated_deception.py ..                    [ 25%]
tests\allocation_ww3\test_determinism.py .                               [ 29%]
tests\allocation_ww3\test_global_catastrophe.py ..                       [ 37%]
tests\allocation_ww3\test_model_collapse.py ...                          [ 48%]
tests\allocation_ww3\test_no_nan_inf.py ..                               [ 55%]
tests\allocation_ww3\test_robust_monotonicity.py ..                      [ 62%]
tests\allocation_ww3\test_stale_replay.py ...                            [ 74%]
tests\allocation_ww3\test_state_corruption.py ...                        [ 85%]
tests\allocation_ww3\test_telemetry_blackout.py ..                       [ 92%]
tests\allocation_ww3\test_trojan_strategy.py ..                          [100%]

============================= 27 passed in 0.05s ==============================

```
