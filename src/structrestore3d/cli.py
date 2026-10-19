"""Command-line dispatcher for the existing StructRestore-3D research scripts.

All data and checkpoint paths are relative to the shell working directory.
"""
from pathlib import Path
import runpy
import sys

from . import __version__

# Command -> packaged script name, positional arguments prepended for existing CLI.
COMMANDS = {
    "audit-v1": ("train_v1_audited", ["audit"]),
    "train-v1": ("train_v1_audited", ["train"]),
    "compare": ("compare_v0_v1", []),
    "diagnose": ("diagnose_v1_inference", []),
    "mask-audit": ("audit_masking", []),
    "train-single-v0": ("train_fullres", []),
    "train-25d-v0": ("structrestore_25d", []),
    "train-baseline": ("train_baseline", []),
    "infer-baseline": ("infer_baseline", []),
    "validate-25d": ("validate_25d", []),
    "register-probe": ("step29_registration_probe", []),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(f"StructRestore-3D {__version__} (experimental research software)")
        print("Usage: structrestore3d COMMAND [OPTIONS]")
        print("Commands:")
        for command in COMMANDS:
            print(f"  {command}")
        print("\nExample: structrestore3d audit-v1")
        print("Run 'structrestore3d train-v1 --help' for training arguments.")
        print("All image data and trained model checkpoints must be supplied separately.")
        return 0
    if sys.argv[1] in ("-V", "--version", "version"):
        print(__version__)
        return 0
    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command!r}; run structrestore3d --help", file=sys.stderr)
        return 2
    script, prefix = COMMANDS[command]
    script_path = Path(__file__).resolve().parent / f"{script}.py"
    if not script_path.is_file():
        print(f"The packaged script {script} is missing. Run python tools/sync_package.py before building.", file=sys.stderr)
        return 1
    # Original research scripts import their siblings as top-level modules.
    # Keep backward compatibility without altering the user's scripts.
    package_dir = str(script_path.parent)
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)
    arguments = sys.argv[2:]
    if command == "mask-audit" and "--script" not in arguments:
        arguments += ["--script", str(script_path.parent / "structrestore_25d.py")]
    sys.argv = [str(script_path), *prefix, *arguments]
    runpy.run_path(str(script_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
