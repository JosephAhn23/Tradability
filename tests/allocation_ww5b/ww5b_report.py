"""Generate WW5-B bluff detection report."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    os.chdir(ROOT)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/allocation_ww5b/", "-v", "--tb=short", "-q"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = result.stdout + result.stderr
    passed = result.returncode == 0
    report_path = ROOT / "runs" / "allocation_ww5b" / "ww5b_bluff_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# WW5-B Bluff Detection & Anti-Hallucination Report\n\n")
        f.write("**Definition:** Bluffing = pretending error does not exist. Precision without evidence = bluff.\n\n")
        f.write("## Result\n\n")
        f.write("**PASS**\n\n" if passed else "**FAIL**\n\n")
        f.write("## Invariants\n\n")
        f.write("- Confidence monotonicity: confidence may only increase if evidence_added increases\n")
        f.write("- Silence over certainty: when evidence insufficient, inaction preferred\n")
        f.write("- Assumptions disclosed: every non-zero action lists assumptions\n")
        f.write("- Unknown declared: unknowable scenario -> explicit UNKNOWN, no action\n\n")
        f.write("## Tests\n\n")
        f.write("- Overprecision under noise -> exposure shrinks\n")
        f.write("- Decimal-point trap -> perturb changes output smoothly\n")
        f.write("- Narrative/authority only -> no evidence_added\n")
        f.write("- Say I don't know -> DORMANT, unknown_declared, zero weights\n")
        f.write("- Bluff risk increases with uncertainty\n")
        f.write("- Counterfactual: allocations from constraints not labels\n\n")
        f.write("## Pytest output\n\n```\n")
        f.write(out[:10000] + ("\n... (truncated)" if len(out) > 10000 else ""))
        f.write("\n```\n")
    print(f"WW5-B report written to: {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
