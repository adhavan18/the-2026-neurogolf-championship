#!/usr/bin/env python3
"""Build a NeuroGolf submission: solve tasks, write ONNX files, zip them up.

Usage:
    python scripts/build_submission.py                # all tasks in data/
    python scripts/build_submission.py --tasks 16 276 # a subset
    python scripts/build_submission.py --out-dir submission --zip submission.zip

Reads task JSON from ``$NEUROGOLF_DATA`` (default: ./data).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurogolf.data import available_task_nums  # noqa: E402
from neurogolf.pipeline import package, run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", type=int, nargs="*", default=None, help="task numbers (default: all)")
    ap.add_argument("--out-dir", default="submission", help="output directory for ONNX files")
    ap.add_argument("--zip", default="submission.zip", help="output zip path")
    ap.add_argument("--no-zip", action="store_true", help="skip zipping")
    args = ap.parse_args()

    nums = args.tasks if args.tasks else available_task_nums()
    if not nums:
        print("No task JSON found. Put taskNNN.json in ./data or set $NEUROGOLF_DATA.")
        raise SystemExit(1)

    print(f"Solving {len(nums)} task(s) -> {args.out_dir}/")
    run(nums, out_dir=args.out_dir)

    if not args.no_zip:
        path = package(args.out_dir, args.zip)
        size = os.path.getsize(path)
        print(f"\nWrote {path} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
