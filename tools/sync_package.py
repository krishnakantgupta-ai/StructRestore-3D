#!/usr/bin/env python3
"""Copy the CURRENT local research scripts into the installable Python package.

Run from anywhere. The original scripts remain unchanged.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts"
TARGET = ROOT / "src" / "structrestore3d"
REQUIRED = {
    "train_baseline.py", "train_fullres.py", "structrestore_25d.py",
    "infer_baseline.py", "train_v1_audited.py", "audit_masking.py",
    "compare_v0_v1.py", "diagnose_v1_inference.py", "validate_25d.py",
}

def main():
    if not SOURCE.is_dir():
        sys.exit("ERROR: scripts/ folder not found. Unzip the kit into the existing StructRestore-3D project folder.")
    current = {p.name for p in SOURCE.glob("*.py")}
    missing = sorted(REQUIRED - current)
    if missing:
        sys.exit("ERROR: Missing expected research scripts: " + ", ".join(missing))
    TARGET.mkdir(parents=True, exist_ok=True)
    for script in sorted(SOURCE.glob("*.py")):
        shutil.copy2(script, TARGET / script.name)
        print(f"Packaged: {script.name}")
    print("Package source synchronized. Re-run this tool after editing any original script.")

if __name__ == "__main__":
    main()
