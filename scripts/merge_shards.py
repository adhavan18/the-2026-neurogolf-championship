#!/usr/bin/env python3
"""Merge sharded pipeline outputs into one submission directory + zip.

Usage:
    python scripts/merge_shards.py SHARD_DIR [SHARD_DIR ...] --out submission --zip submission.zip

Each shard directory contains taskNNN.onnx files and a manifest.json (a list of
per-task results).  ONNX files are copied into the output directory; manifests
are concatenated sorted by task number.  If several shards contain the same
task, the one with the lower cost wins.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shards", nargs="+", help="shard output directories")
    ap.add_argument("--out", default="submission")
    ap.add_argument("--zip", default="submission.zip")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    best: dict[int, dict] = {}
    src_dir: dict[int, str] = {}
    for shard in args.shards:
        with open(os.path.join(shard, "manifest.json")) as f:
            for entry in json.load(f):
                num = entry["task"]
                cur = best.get(num)
                take = cur is None or (
                    entry["solved"]
                    and (not cur["solved"] or (entry["cost"] or 0) < (cur["cost"] or 0))
                )
                if take:
                    best[num] = entry
                    src_dir[num] = shard

    solved = 0
    points = 0.0
    for num, entry in sorted(best.items()):
        if not entry["solved"]:
            continue
        solved += 1
        points += entry["points"]
        name = f"task{num:03d}.onnx"
        shutil.copy2(os.path.join(src_dir[num], name), os.path.join(args.out, name))

    manifest = [best[n] for n in sorted(best)]
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(os.listdir(args.out)):
            if name.endswith(".onnx"):
                zf.write(os.path.join(args.out, name), arcname=name)

    print(f"Merged {len(args.shards)} shard(s): {solved}/{len(best)} solved, {points:.3f} points")
    print(f"Wrote {args.zip} ({os.path.getsize(args.zip)/1024:.1f} KB)")

    by_solver: dict[str, list] = {}
    for e in manifest:
        if e["solved"]:
            by_solver.setdefault(e["solver"], []).append(e["points"])
    for s in sorted(by_solver):
        pts = by_solver[s]
        print(f"  {s:12s}: {len(pts):3d} tasks, {sum(pts):8.3f} pts")


if __name__ == "__main__":
    main()
