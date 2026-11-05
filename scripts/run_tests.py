#!/usr/bin/env python3
"""Run both inherited suites in separate processes with public fixture settings."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = {"text-first": ("text_first", "validate_quality_fixes.py"), "topology-first": ("topology_first", "verify_quality_repairs.py")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", choices=["all", *RUNNERS], default="all")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "outputs" / "validation" / "tests")
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "rl4ppt-final":
        parser.error("activate rl4ppt-final first")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in RUNNERS if args.path == "all" else [args.path]:
        folder, runner = RUNNERS[name]
        route = ROOT / "paths" / folder
        log = args.log_dir / f"{name}.log"
        print(f"Running {name}; log: {log}", flush=True)
        with log.open("w") as handle:
            result = subprocess.run([sys.executable, str(route / "scripts" / runner), "--tests"], cwd=route, stdout=handle, stderr=subprocess.STDOUT)
        output = log.read_text()
        match = re.search(r"Ran (\d+) tests? in", output)
        record = {"path": name, "exit_code": result.returncode, "tests": int(match.group(1)) if match else None, "log": str(log.resolve())}
        records.append(record)
        print(json.dumps(record), flush=True)
        if result.returncode:
            print(output[-12000:], file=sys.stderr)
    summary = {"status": "passed" if all(r["exit_code"] == 0 for r in records) else "failed", "suites": records, "tests": sum(r["tests"] or 0 for r in records), "configuration": "public offline fixtures; no private configuration loaded"}
    (args.log_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return int(summary["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
