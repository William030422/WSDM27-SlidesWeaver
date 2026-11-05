#!/usr/bin/env python3
"""Check the shared release environment without opening configuration files."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="check mock/test requirements; skip CUDA and model weights")
    parser.add_argument("--model-path", type=Path, help="local Z-Image-Turbo directory (or export DIFFUSION_MODEL_PATH)")
    args = parser.parse_args()
    failures = []
    if os.environ.get("CONDA_DEFAULT_ENV") != "rl4ppt-final":
        failures.append("activate rl4ppt-final")
    if sys.version_info[:2] != (3, 11):
        failures.append("Python 3.11 is required")
    packages = {}
    modules = ["pydantic", "PIL", "numpy", "cv2", "playwright"]
    if not args.offline:
        modules += ["torch", "diffusers", "transformers", "accelerate"]
    for name in modules:
        try:
            module = importlib.import_module(name)
            packages[name] = getattr(module, "__version__", "installed")
        except Exception as exc:
            failures.append(f"cannot import {name}: {exc}")
    binaries = {}
    for name in ("node", "npm", "libreoffice", "pdftoppm", "pdftotext", "fc-match"):
        binaries[name] = shutil.which(name)
        if not binaries[name]:
            failures.append(f"missing binary: {name}")
    if binaries["node"]:
        result = subprocess.run(["node", "-e", "require('pptxgenjs'); require('skia-canvas'); require('fontkit'); require('linebreak');"], cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            failures.append("Node dependencies unavailable; run npm ci in SlidesWeaver")
    if binaries["fc-match"]:
        for style in ("Regular", "Bold"):
            result = subprocess.run(["fc-match", "-f", "%{family}\n%{style}\n%{file}", f"Noto Sans:style={style}"], capture_output=True, text=True)
            lines = result.stdout.strip().splitlines()
            if len(lines) < 3 or "Noto Sans" not in {v.strip() for v in lines[0].split(",")} or (style == "Bold" and "bold" not in lines[1].lower()):
                failures.append(f"install Noto Sans {style}; substitution changes layout")
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as playwright:
            if not Path(playwright.chromium.executable_path).is_file():
                failures.append("Chromium missing; run python -m playwright install chromium")
    except Exception as exc:
        failures.append(f"cannot inspect Chromium: {exc}")
    model = args.model_path or Path(os.environ.get("DIFFUSION_MODEL_PATH", "~/models/Tongyi-MAI_Z-Image-Turbo"))
    model = model.expanduser().resolve()
    if not args.offline:
        try:
            import torch
            from diffusers import ZImagePipeline  # noqa: F401
            if not torch.cuda.is_available():
                failures.append("CUDA is unavailable to PyTorch")
        except Exception as exc:
            failures.append(f"Z-Image stack unavailable: {exc}")
        if not (model / "model_index.json").is_file():
            failures.append(f"model_index.json missing in {model}")
    print(json.dumps({"status": "failed" if failures else "ready", "mode": "offline" if args.offline else "live", "environment": os.environ.get("CONDA_DEFAULT_ENV"), "python": sys.version.split()[0], "packages": packages, "binaries": binaries, "model_path": None if args.offline else str(model), "failures": failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
