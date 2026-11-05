#!/usr/bin/env python3
"""Private bootstrap: configure one route, then run its original CLI."""
from __future__ import annotations

from pathlib import Path
import runpy
import sys


def main() -> None:
    route, config_file = (Path(value).resolve() for value in sys.argv[1:3])
    forwarded = sys.argv[3:]
    sys.path[:0] = [str(route), str(route / "src")]
    # This only selects the configuration file. Parsing, defaults, validation,
    # API payloads, prompts, rendering and multiprocessing remain route-owned.
    from integration import config
    original = config.load_config

    def configured(project_root, *, env_path=None):
        return original(project_root, env_path=env_path if env_path is not None else config_file)

    config.load_config = configured
    sys.argv = [str(route / "run_pipeline.py"), *forwarded]
    runpy.run_path(str(route / "run_pipeline.py"), run_name="__main__")


if __name__ == "__main__":
    main()
