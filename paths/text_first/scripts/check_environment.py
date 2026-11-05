#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path.home() / "models" / "Tongyi-MAI_Z-Image-Turbo"
sys.path.insert(0, str(ROOT))


def main() -> int:
    failures: list[str] = []
    if os.environ.get("CONDA_DEFAULT_ENV") != "rl4ppt-final":
        failures.append("activate the rl4ppt-final Conda environment")
    if sys.version_info[:2] != (3, 11):
        failures.append(f"Python 3.11 is required; found {sys.version.split()[0]}")

    versions = _import_versions(failures)
    binaries = _binary_versions(failures)
    _check_node_modules(failures)
    deck_font = _check_deck_font(failures)
    _check_cuda(failures)
    model_path = Path(os.environ.get("DIFFUSION_MODEL_PATH", DEFAULT_MODEL_PATH)).expanduser()
    if not (model_path / "model_index.json").is_file():
        failures.append(f"Z-Image model is missing: {model_path / 'model_index.json'}")

    report = {
        "status": "failed" if failures else "ready",
        "environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": sys.version.split()[0],
        "packages": versions,
        "binaries": binaries,
        "deck_font": deck_font,
        "diffusion_model": str(model_path),
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


def _import_versions(failures: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    modules = {
        "accelerate": "accelerate",
        "cv2": "opencv",
        "diffusers": "diffusers",
        "numpy": "numpy",
        "PIL": "Pillow",
        "pydantic": "pydantic",
        "playwright": "playwright",
        "torch": "torch",
        "transformers": "transformers",
    }
    for module_name, label in modules.items():
        try:
            module = importlib.import_module(module_name)
            versions[label] = str(getattr(module, "__version__", "installed"))
        except Exception as exc:
            failures.append(f"cannot import {module_name}: {exc}")
    try:
        from diffusers import ZImagePipeline  # noqa: F401
    except Exception as exc:
        failures.append(f"cannot import diffusers.ZImagePipeline: {exc}")
    return versions


def _binary_versions(failures: list[str]) -> dict[str, str]:
    commands = {
        "node": ["node", "--version"],
        "npm": ["npm", "--version"],
        "libreoffice": ["libreoffice", "--version"],
        "pdftoppm": ["pdftoppm", "-v"],
    }
    versions: dict[str, str] = {}
    for name, command in commands.items():
        binary_path = shutil.which(command[0])
        if binary_path is None:
            failures.append(f"missing required binary: {command[0]}")
            continue
        if name in {"node", "npm", "pdftoppm"}:
            _require_conda_binary(name, Path(binary_path), failures)
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        output = (result.stdout or result.stderr).strip().splitlines()
        if result.returncode != 0 or not output:
            failures.append(f"cannot run {' '.join(command)}")
            continue
        versions[name] = output[0]
    return versions


def _require_conda_binary(name: str, path: Path, failures: list[str]) -> None:
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix is None:
        failures.append("the rl4ppt-final Conda environment is not active")
        return
    try:
        path.resolve().relative_to(Path(prefix).resolve())
    except ValueError:
        failures.append(f"{name} is outside the active Conda environment: {path}")


def _check_node_modules(failures: list[str]) -> None:
    result = subprocess.run(
        [
            "node",
            "-e",
            "require('pptxgenjs'); require('skia-canvas');",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        failures.append("Node dependencies are incomplete; run npm ci")


def _check_deck_font(failures: list[str]) -> dict[str, object]:
    from integration.config import DECK_FONT, validate_deck_font

    try:
        variants = validate_deck_font(DECK_FONT)
    except Exception as exc:
        failures.append(str(exc))
        variants = {}
    return {"family": DECK_FONT, "variants": variants}


def _check_cuda(failures: list[str]) -> None:
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        failures.append("CUDA is unavailable to PyTorch")

if __name__ == "__main__":
    raise SystemExit(main())
