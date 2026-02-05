"""
WW3 report generator: run pytest for allocation_ww3, then write ww3_report.md.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    os.chdir(ROOT)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/allocation_ww3/", "-v", "--tb=short", "-q"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = result.stdout + result.stderr
    passed = result.returncode == 0

    report_path = ROOT / "runs" / "allocation_ww3" / "ww3_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# WW3 / Annihilation Test Report\n\n")
        f.write("**Prime directive:** When uncertainty is extreme, exposure must go to zero.\n\n")
        f.write("## Result\n\n")
        f.write("**PASS**\n\n" if passed else "**FAIL**\n\n")
        f.write("## Worst-case behaviour\n\n")
        f.write("- Under hazard_level >= 4 (telemetry blackout, correlation meltdown, liquidity shock): ")
        f.write("gross exposure is forced to EMERGENCY_GROSS (5% or 0).\n")
        f.write("- Robust mode: robust allocations never exceed nominal (component-wise and gross).\n")
        f.write("- All strategies bad (feasibility <= 0 or regime below min): zero allocation, no NaN.\n")
        f.write("- Trojan strategy (too good to be true): capped, cannot dominate.\n")
        f.write("- State corruption / empty inputs: conservative reset, no full risk, no crash.\n")
        f.write("- Stale / replay / integrity failure: hazard elevated, exposure at emergency when level >= 4.\n")
        f.write("- Coordinated deception (one correlation group): per-strategy cap binds; under stress gross collapses.\n")
        f.write("- Continuity under epsilon: no cliffs; weights change smoothly.\n")
        f.write("- Model collapse (all negative/zero edge): zero allocation.\n")
        f.write("- Determinism: same inputs => same outputs.\n\n")
        f.write("## Pytest output\n\n```\n")
        f.write(out[:8000] + ("\n... (truncated)" if len(out) > 8000 else ""))
        f.write("\n```\n")

    print(f"WW3 report written to: {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
