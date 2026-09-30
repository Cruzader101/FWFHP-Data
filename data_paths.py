"""Location of the FWFHP REDCap export shared by every analysis script.

The export is patient data: it lives in `Night Camp Data/`, which is gitignored and
must never be committed. See README.md for how to download it from REDCap.

The default below pins the exact export the current results were produced from.
To run against a different export, set the FWFHP_EXPORT environment variable
(or pass --data to run_all.py).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Night Camp Data"

DEFAULT_EXPORT = DATA_DIR / "FarmWorkerFamilyHeal-2426FWFHPDataAnalysi_DATA_LABELS_2026-07-06_1745.csv"

FWFHP_EXPORT = Path(os.environ.get("FWFHP_EXPORT", DEFAULT_EXPORT))
