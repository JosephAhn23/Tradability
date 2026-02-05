"""
Generate WW4 survival report: run pytest for allocation_ww4, then write ww4_survival_report.md.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    os.chdir(ROOT)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/allocation_ww4/", "-v", "--tb=short", "-q"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = result.stdout + result.stderr
    passed = result.returncode == 0

    report_path = ROOT / "runs" / "allocation_ww4" / "ww4_survival_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# WW4 Survival Report\n\n")
        f.write("**Prime law:** When you cannot trust the world, you cannot take risk.\n\n")
        f.write("## Result\n\n")
        f.write("**PASS**\n\n" if passed else "**FAIL**\n\n")
        f.write("## Required report fields (per run)\n\n")
        f.write("- **hazard_level and state:** NORMAL | CAUTION | DANGER | SURVIVAL | LOCKDOWN\n")
        f.write("- **reason_codes:** triggers (telemetry_fail, blackout, sensor_poisoning, etc.)\n")
        f.write("- **exposure before/after:** gross exposure monotonicity under hazard\n")
        f.write("- **concentration before/after:** max single-strategy weight monotonicity\n")
        f.write("- **turnover throttle applied:** from allocation result\n")
        f.write("- **modules healthy/unhealthy:** feasibility, stress, regime\n")
        f.write("- **consensus status:** agree | disagree (disagree => worst-case)\n")
        f.write('- **"why not taking risk" narrative:** in DANGER/SURVIVAL/LOCKDOWN\n\n')
        f.write("## Invariants verified\n\n")
        f.write("- Exposure monotonicity: hazard up => exposure never increases\n")
        f.write("- Concentration monotonicity: hazard up => max single weight never increases\n")
        f.write("- Sensor poisoning / blackout => SURVIVAL or LOCKDOWN, exposure near zero\n")
        f.write("- Consensus disagree => worst-case, low exposure\n")
        f.write("- Permutation invariance; scale invariance in SURVIVAL\n")
        f.write("- Chaos (infra failure) => safe shutdown, no crash, audit trail\n\n")
        f.write("## Pytest output\n\n```\n")
        f.write(out[:10000] + ("\n... (truncated)" if len(out) > 10000 else ""))
        f.write("\n```\n")

    print(f"WW4 survival report written to: {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
