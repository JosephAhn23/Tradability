# WW4 Survival Report

**Prime law:** When you cannot trust the world, you cannot take risk.

## Result

**PASS**

## Required report fields (per run)

- **hazard_level and state:** NORMAL | CAUTION | DANGER | SURVIVAL | LOCKDOWN
- **reason_codes:** triggers (telemetry_fail, blackout, sensor_poisoning, etc.)
- **exposure before/after:** gross exposure monotonicity under hazard
- **concentration before/after:** max single-strategy weight monotonicity
- **turnover throttle applied:** from allocation result
- **modules healthy/unhealthy:** feasibility, stress, regime
- **consensus status:** agree | disagree (disagree => worst-case)
- **"why not taking risk" narrative:** in DANGER/SURVIVAL/LOCKDOWN

## Invariants verified

- Exposure monotonicity: hazard up => exposure never increases
- Concentration monotonicity: hazard up => max single weight never increases
- Sensor poisoning / blackout => SURVIVAL or LOCKDOWN, exposure near zero
- Consensus disagree => worst-case, low exposure
- Permutation invariance; scale invariance in SURVIVAL
- Chaos (infra failure) => safe shutdown, no crash, audit trail

## Pytest output

```
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-8.4.2, pluggy-1.6.0
rootdir: C:\Users\josep\Downloads\Tradability-master\signal-tradability-main
plugins: anyio-4.11.0
collected 21 items

tests\allocation_ww4\test_blackout_noise.py ..                           [  9%]
tests\allocation_ww4\test_chaos.py ...                                   [ 23%]
tests\allocation_ww4\test_consensus.py ..                                [ 33%]
tests\allocation_ww4\test_dominance_vs_cash.py ..                        [ 42%]
tests\allocation_ww4\test_forced_liquidation.py ..                       [ 52%]
tests\allocation_ww4\test_invariants.py ...                              [ 66%]
tests\allocation_ww4\test_permutation_scale.py ..                        [ 76%]
tests\allocation_ww4\test_sensor_poisoning.py ...                        [ 90%]
tests\allocation_ww4\test_threshold_bait.py ..                           [100%]

============================= 21 passed in 0.05s ==============================

```
