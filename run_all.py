"""Rebuild every result in this repo from the raw inputs, in dependency order.

Usage:
    python run_all.py                     # uses the export pinned in data_paths.py
    python run_all.py --data path/to.csv  # run against a different REDCap export

Each script's console output (the numbers behind every chart and slide) is also
saved to logs/<script>.log. logs/ is gitignored because it is derived from
patient data.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"

# dataframes.py must run first: it writes NHANES/df_Hispanic_NHANES.csv,
# which analyses.py reads.
STEPS = [
    ("dataframes.py",                "NHANES Hispanic reference -> NHANES/df_Hispanic_NHANES.csv"),
    ("analyses.py",                  "FWFHP vs NHANES comparison + logistic regression"),
    ("co_occurences.py",             "Co-occurring diagnoses, binary PHQ-2 (Fisher + FDR)"),
    ("co_occurrences_continuous.py", "Co-occurring diagnoses, continuous PHQ-2 (logistic + FDR)"),
    ("co_occurrence_sensitivity.py", "Sensitivity diagnostics for the co-occurrence models"),
    ("care_gap.py",                  "Screen-positive -> mental health diagnosis"),
]

# Inputs that must exist before anything runs. NHANES files ship with the repo.
NHANES_INPUTS = ["DPQ_L.xpt", "DEMO_L.xpt"]


def run_step(script, env):
    """Run one script, echoing its output to the console and to logs/<script>.log."""
    log_path = LOG_DIR / f"{Path(script).stem}.log"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, script], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path,
                        help="REDCap export CSV to use instead of the pinned default")
    args = parser.parse_args()

    sys.stdout.reconfigure(errors="replace")
    env = dict(os.environ)
    if args.data:
        export = args.data.resolve()
        env["FWFHP_EXPORT"] = str(export)
    else:
        from data_paths import FWFHP_EXPORT as export
    env["MPLBACKEND"] = "Agg"            # save charts without opening windows
    env["PYTHONIOENCODING"] = "utf-8"    # scripts print ≤, ∞, α
    # plt.show() is a no-op under Agg; silence the warning it raises for each chart.
    env["PYTHONWARNINGS"] = "ignore:FigureCanvasAgg is non-interactive:UserWarning"

    missing = [p for p in [export, *(ROOT / f for f in NHANES_INPUTS)] if not p.is_file()]
    if missing:
        print("Missing input file(s):")
        for p in missing:
            print(f"  {p}")
        print("\nThe FWFHP export is not in the repo (patient data). "
              "See README.md -> 'Getting the data from REDCap'.")
        sys.exit(1)

    print(f"FWFHP export: {export}")
    print(f"Python {sys.version.split()[0]} ({sys.executable})")
    LOG_DIR.mkdir(exist_ok=True)
    (ROOT / "charts").mkdir(exist_ok=True)
    (ROOT / "NHANES").mkdir(exist_ok=True)

    for script, desc in STEPS:
        print(f"\n{'=' * 78}\n{script}: {desc}\n{'=' * 78}", flush=True)
        code = run_step(script, env)
        if code != 0:
            print(f"\n{script} failed with exit code {code}. "
                  f"Full output: {LOG_DIR / Path(script).stem}.log")
            sys.exit(code)

    print(f"\nAll {len(STEPS)} steps finished. Charts: charts/  Logs: logs/")


if __name__ == "__main__":
    main()
