#!/usr/bin/env python3
"""Verify pipeline source checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="optional RL4PPT checkout; never opens private configuration")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    failures = []
    checked = 0
    reference_checked = 0
    for record in manifest["files"]:
        local = ROOT / record["path"]
        if not local.is_file() or sha256(local) != record["sha256"]:
            failures.append({"path": record["path"], "reason": "packaged source differs or is missing"})
        if args.source_root is not None and record.get("source"):
            reference_checked += 1
            source = args.source_root.expanduser().resolve() / record["source"]
            if not source.is_file() or sha256(source) != record["source_sha256"]:
                failures.append({"path": record["source"], "reason": "original source differs from the recorded snapshot"})
        checked += 1
    print(json.dumps({"status": "passed" if not failures else "failed", "files_checked": checked,
                      "reference_files_checked": reference_checked, "failures": failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
