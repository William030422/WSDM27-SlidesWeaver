#!/usr/bin/env python3
"""Select an isolated, unchanged SlidesWeaver generation pipeline."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PATHS = {"text-first": "text_first", "topology-first": "topology_first"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="All other options go to the selected pipeline. Use --pipeline-help for its options.",
    )
    parser.add_argument("--path", choices=PATHS, required=True)
    parser.add_argument("--config", type=Path, help="explicit configuration file (e.g. configs/local.conf)")
    parser.add_argument("--pipeline-help", action="store_true")
    args, forwarded = parser.parse_known_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "rl4ppt-final":
        parser.error("activate the rl4ppt-final environment first: conda activate rl4ppt-final")
    route = ROOT / "paths" / PATHS[args.path]
    if args.pipeline_help:
        forwarded = ["--help"]
    else:
        # The original runners resolve relative output paths against their own roots.
        # Resolve user paths here so this entry point consistently uses the caller's cwd.
        for flag in ("--input", "--output-dir"):
            for index, token in enumerate(forwarded):
                if token == flag and index + 1 < len(forwarded):
                    forwarded[index + 1] = str(Path(forwarded[index + 1]).expanduser().resolve())
                elif token.startswith(flag + "="):
                    forwarded[index] = flag + "=" + str(Path(token.split("=", 1)[1]).expanduser().resolve())
        if not any(token == "--output-dir" or token.startswith("--output-dir=") for token in forwarded):
            from datetime import datetime
            forwarded += ["--output-dir", str(ROOT / "outputs" / f"{args.path}_{datetime.now():%Y%m%d_%H%M%S_%f}")]
    config = args.config
    if config is None and "--mock-api" in forwarded:
        config = ROOT / "configs" / "mock.conf"
    if config is None:
        config = ROOT / "configs" / "local.conf"
    config = config.expanduser().resolve()
    if not args.pipeline_help and not config.is_file():
        parser.error(f"configuration missing: {config}; copy configs/example.conf to configs/local.conf and fill in your settings")
    command = [sys.executable, str(ROOT / "scripts" / "_run_path.py"), str(route), str(config), *forwarded]
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
