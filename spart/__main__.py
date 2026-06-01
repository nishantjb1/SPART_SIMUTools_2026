"""
spart/__main__.py
CLI entry point: python -m spart <command> [options]

Commands
--------
  demo          Run the synthetic demo (no dataset needed).
                Produces demo_animation.gif + demo_pointcloud.csv in results/synthetic_demo/.
                Completes in < 60 s including JIT warmup.

  check         Pre-flight sanity check on one INTERACTION CSV.
                Verifies ego-trajectory sharing, frame-list consistency, geometric
                agreement (FP=0, range_error=0), and data-race fix.

  version       Print SPART version and dependency information.

Examples
--------
  python -m spart demo
  python -m spart check --csv vehicle_tracks_000.csv
  python -m spart version
"""

import argparse
import os
import subprocess
import sys

import spart as _spart_pkg

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts")


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------

def cmd_demo(args):
    """Run the synthetic demo (scripts/exp5_synthetic_demo.py)."""
    script = os.path.join(_SCRIPTS, "exp5_synthetic_demo.py")
    if not os.path.exists(script):
        print(f"[ERROR] Demo script not found: {script}")
        print("        Run from the project root directory.")
        sys.exit(1)
    print("[SPART] Running synthetic demo (no dataset required) ...")
    print(f"        Output will appear in  results/synthetic_demo/")
    ret = subprocess.run([sys.executable, script])
    sys.exit(ret.returncode)


def cmd_check(args):
    """Run pre-flight sanity check (scripts/sanity_check.py)."""
    if not args.csv:
        print("[ERROR] --csv is required for the 'check' command.")
        print("  Usage: python -m spart check --csv vehicle_tracks_000.csv")
        sys.exit(1)
    script = os.path.join(_SCRIPTS, "sanity_check.py")
    if not os.path.exists(script):
        print(f"[ERROR] Sanity check script not found: {script}")
        sys.exit(1)
    ret = subprocess.run([sys.executable, script, "--csv", args.csv])
    sys.exit(ret.returncode)


def cmd_version(args):
    """Print version and dependency information."""
    print(f"SPART version {_spart_pkg.__version__}")
    try:
        import numpy as np
        print(f"  numpy   {np.__version__}")
    except ImportError:
        print("  numpy   NOT FOUND")
    try:
        import numba
        print(f"  numba   {numba.__version__}")
    except ImportError:
        print("  numba   NOT FOUND")
    try:
        import matplotlib
        print(f"  matplotlib {matplotlib.__version__}")
    except ImportError:
        print("  matplotlib NOT FOUND")
    try:
        import pandas as pd
        print(f"  pandas  {pd.__version__}")
    except ImportError:
        print("  pandas  NOT FOUND")
    try:
        import PIL
        print(f"  pillow  {PIL.__version__}")
    except ImportError:
        print("  pillow  NOT FOUND (GIF output disabled)")
    try:
        import psutil
        print(f"  psutil  {psutil.__version__}")
    except ImportError:
        print("  psutil  NOT FOUND (memory tracking disabled)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="python -m spart",
        description="SPART — 2-D LiDAR emulator for AV trajectory-dataset benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # demo
    p_demo = sub.add_parser("demo",
        help="Run synthetic demo (no dataset needed). Outputs GIF + point cloud.")

    # check
    p_check = sub.add_parser("check",
        help="Pre-flight sanity check on one INTERACTION CSV.")
    p_check.add_argument("--csv", required=False, default=None,
                         metavar="PATH",
                         help="Path to vehicle_tracks_*.csv")

    # version
    p_ver = sub.add_parser("version", help="Print version and dependency info.")

    args = parser.parse_args()

    if args.command == "demo":
        cmd_demo(args)
    elif args.command == "check":
        cmd_check(args)
    elif args.command == "version":
        cmd_version(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
