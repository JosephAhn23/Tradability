"""Generate WW5 existential audit report."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    os.chdir(ROOT)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/allocation_ww5/", "-v", "--tb=short", "-q"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = result.stdout + result.stderr
    passed = result.returncode == 0
    report_path = ROOT / "runs" / "allocation_ww5" / "ww5_existential_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# WW5 Existential Audit Report\n\n")
        f.write("**Prime law:** When the environment is unknowable, the only winning move is preserving optionality.\n\n")
        f.write("## Result\n\n")
        f.write("**PASS**\n\n" if passed else "**FAIL**\n\n")
        f.write("## Required outputs (per run)\n\n")
        f.write("- survival_state: NORMAL | CONSERVATIVE | SURVIVAL | DORMANT\n")
        f.write("- irreversible_actions_blocked (count)\n")
        f.write("- assumptions_required / assumptions_rejected\n")
        f.write("- optionality_score\n")
        f.write("- confidence_decay_rate\n")
        f.write("- reason_for_not_acting\n\n")
        f.write("## Scenarios verified\n\n")
        f.write("- No feedback: confidence decays, exposure shrinks\n")
        f.write("- Hidden regime shift / deceptive stability: reduce exposure\n")
        f.write("- One-way door: irreversible blocked under uncertainty\n")
        f.write("- Radiation noise / uncertainty extreme: DORMANT, zero exposure\n")
        f.write("- Optionality dominance: DORMANT preserves optionality >= cash\n")
        f.write("- Single point of truth: distrust -> SURVIVAL\n\n")
        f.write("## Pytest output\n\n```\n")
        f.write(out[:10000] + ("\n... (truncated)" if len(out) > 10000 else ""))
        f.write("\n```\n")
    print(f"WW5 report written to: {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
