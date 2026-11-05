#!/usr/bin/env python3
"""Compare original and packaged renderers using the same accepted slide plans.

This local-only check never loads pipeline configuration, calls a model, or
changes the source run. It verifies renderer equivalence, not slide quality or
the reproducibility of a new stochastic generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from PIL import Image, ImageChops, ImageDraw, ImageStat


ROOT = Path(__file__).resolve().parents[1]
NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}
ROUTES = {
    "text-first": ("text_first", "final-demo-diffusion-planning-reuse", "text_parallel"),
    "topology-first": ("topology_first", "engineering-fix-clean", "topology_parallel_confirmed"),
}


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_command(command: list[str], *, cwd: Path, log: Path) -> None:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=300)
    log.write_text(
        json.dumps({"command": command, "cwd": str(cwd)}, ensure_ascii=False) + "\n"
        + result.stdout + "\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")


def compile_and_render(project: Path, manifest: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    pptx = output / "deck.pptx"
    run_command(
        ["node", str(project / "compile_deck.mjs"), str(manifest), str(pptx),
         str(output / "text_fit_report.json")],
        cwd=project, log=output / "compile.log",
    )
    with tempfile.TemporaryDirectory(prefix="slidesweaver-parity-office-") as profile:
        run_command(
            ["libreoffice", f"-env:UserInstallation={Path(profile).as_uri()}",
             "--headless", "--convert-to", "pdf", "--outdir", str(output), str(pptx)],
            cwd=output, log=output / "pdf.log",
        )
    pdf = output / "deck.pdf"
    if not pdf.is_file():
        raise RuntimeError(f"LibreOffice did not create {pdf}; see {output / 'pdf.log'}")
    previews = output / "rendered_slides"
    previews.mkdir()
    run_command(
        ["pdftoppm", "-png", "-r", "120", str(pdf), str(previews / "slide")],
        cwd=output, log=output / "previews.log",
    )


def slide_xml(pptx: Path) -> dict[str, bytes]:
    with ZipFile(pptx) as archive:
        names = sorted(
            (name for name in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)),
            key=lambda name: int(re.search(r"(\d+)\.xml$", name).group(1)),
        )
        return {name: archive.read(name) for name in names}


def native_text(xml: bytes) -> list[dict]:
    records = []
    for shape in ET.fromstring(xml).findall(".//p:sp", NS):
        words = [node.text or "" for node in shape.findall(".//a:t", NS)]
        if not any(text.strip() for text in words):
            continue
        records.append({
            "text_runs": words,
            "latin_fonts": [node.attrib for node in shape.findall(".//a:latin", NS)],
            "cjk_fonts": [node.attrib for node in shape.findall(".//a:ea", NS)],
            "run_properties": [node.attrib for node in shape.findall(".//a:rPr", NS)],
            "transforms": [ET.tostring(node, encoding="unicode")
                           for node in shape.findall(".//a:xfrm", NS)],
        })
    return records


def preview_paths(output: Path) -> list[Path]:
    return sorted((output / "rendered_slides").glob("slide-*.png"),
                  key=lambda item: int(item.stem.rsplit("-", 1)[1]))


def normalize_output_paths(value: object, output: Path) -> object:
    """Ignore only the distinct directories chosen for the two check outputs."""
    if isinstance(value, dict):
        return {key: normalize_output_paths(item, output) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_output_paths(item, output) for item in value]
    if isinstance(value, str):
        return value.replace(str(output), "<RENDER_OUTPUT>")
    return value


def compare_route(route: str, source_root: Path, source_run: Path, output: Path) -> dict:
    package_dir, original_dir, _ = ROUTES[route]
    original, packaged = source_root / original_dir, ROOT / "paths" / package_dir
    manifest = source_run / "assembly" / "deck_compilation.json"
    for required in (manifest, original / "compile_deck.mjs", packaged / "compile_deck.mjs"):
        if not required.is_file():
            raise FileNotFoundError(required)
    compilation = json.loads(manifest.read_text(encoding="utf-8"))
    expected = len(compilation.get("slides", []))
    if not expected:
        raise ValueError(f"Saved compilation contains no slides: {manifest}")
    for slide in compilation["slides"]:
        if not Path(slide["plan"]).is_absolute() or not Path(slide["plan"]).is_file():
            raise ValueError(f"Saved compilation must reference existing absolute plan paths: {slide['plan']}")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    output.mkdir()
    left, right = output / "original", output / "packaged"
    print(f"{route}: compiling and rendering original accepted plans", flush=True)
    compile_and_render(original, manifest, left)
    print(f"{route}: compiling and rendering packaged accepted plans", flush=True)
    compile_and_render(packaged, manifest, right)
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != manifest_hash:
        raise RuntimeError("The source compilation manifest changed during verification")

    xml_left, xml_right = slide_xml(left / "deck.pptx"), slide_xml(right / "deck.pptx")
    xml_records = []
    for name in sorted(set(xml_left) | set(xml_right)):
        first, second = xml_left.get(name), xml_right.get(name)
        xml_records.append({
            "part": name,
            "xml_identical": first == second,
            "native_text_fonts_geometry_identical": (
                first is not None and second is not None and native_text(first) == native_text(second)),
            "native_text_objects": len(native_text(first)) if first is not None else 0,
        })
    fit_left = json.loads((left / "text_fit_report.json").read_text(encoding="utf-8"))
    fit_right = json.loads((right / "text_fit_report.json").read_text(encoding="utf-8"))
    fits_identical = normalize_output_paths(fit_left, left) == normalize_output_paths(fit_right, right)
    overflow = [
        {"slide_id": slide["slide_id"], "region_ids": slide.get("overflow_region_ids", [])}
        for slide in fit_left.get("slides", []) if slide.get("overflow_region_ids")
    ]
    previews_left, previews_right = preview_paths(left), preview_paths(right)
    page_records = []
    width, height, label_height, gap = 640, 360, 28, 12
    paired = Image.new("RGB", (width * 2 + gap, expected * (height + label_height + gap)), "#eeeeee")
    draw = ImageDraw.Draw(paired)
    for index in range(max(len(previews_left), len(previews_right))):
        if index >= min(len(previews_left), len(previews_right)):
            page_records.append({"page": index + 1, "pixel_identical": False, "missing_page": True})
            continue
        with Image.open(previews_left[index]) as opened:
            first = opened.convert("RGB")
        with Image.open(previews_right[index]) as opened:
            second = opened.convert("RGB")
        same_size = first.size == second.size
        difference = ImageChops.difference(first, second) if same_size else None
        page_records.append({
            "page": index + 1, "same_size": same_size,
            "pixel_identical": difference is not None and difference.getbbox() is None,
            "mean_absolute_difference_rgb": ImageStat.Stat(difference).mean if difference is not None else None,
        })
        y = index * (height + label_height + gap)
        draw.text((8, y + 6), f"{route} / {index + 1} / Original", fill="black")
        draw.text((width + gap + 8, y + 6), f"{route} / {index + 1} / SlidesWeaver", fill="black")
        paired.paste(first.resize((width, height)), (0, y + label_height))
        paired.paste(second.resize((width, height)), (width + gap, y + label_height))
    montage = output / "paired_montage.png"
    paired.save(montage)
    counts_match = len(xml_left) == len(xml_right) == len(previews_left) == len(previews_right) == expected
    equivalent = counts_match and fits_identical and all(item["xml_identical"] for item in xml_records) \
        and all(item["pixel_identical"] for item in page_records)
    result = {
        "route": route, "status": "equivalent" if equivalent else "different",
        "source_project": str(original), "packaged_project": str(packaged),
        "source_run": str(source_run), "shared_compilation_manifest": str(manifest),
        "compilation_manifest_sha256": manifest_hash,
        "model_calls": 0, "expected_slides": expected, "page_counts_match": counts_match,
        "native_text_objects": sum(item["native_text_objects"] for item in xml_records),
        "slide_xml": xml_records, "text_fit_report_identical": fits_identical,
        "preserved_overflow_regions": overflow, "pages": page_records,
        "paired_montage": str(montage),
        "quality_assessment": "Not assessed; existing overflow and visual limitations are preserved.",
    }
    write_json(output / "parity.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", choices=("all", *ROUTES), default="all")
    parser.add_argument("--source-root", type=Path, required=True, help="Original RL4PPT workspace root")
    parser.add_argument("--text-run", type=Path, help="Saved Text First run containing assembly/deck_compilation.json")
    parser.add_argument("--topology-run", type=Path, help="Saved Topology First run containing assembly/deck_compilation.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "parity",
                        help="Parent directory; each invocation creates a new timestamped child")
    args = parser.parse_args()
    for binary in ("node", "libreoffice", "pdftoppm"):
        if shutil.which(binary) is None:
            parser.error(f"Missing {binary}; activate the configured environment and install host dependencies")
    source_root = args.source_root.expanduser().resolve()
    if not source_root.is_dir():
        parser.error(f"Source root does not exist: {source_root}")
    base = args.output_dir.expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = base / f"run_{stamp}"
    output.mkdir(exist_ok=False)
    routes = list(ROUTES) if args.path == "all" else [args.path]
    reports = []
    for route in routes:
        _, _, default_run = ROUTES[route]
        override = args.text_run if route == "text-first" else args.topology_run
        source_run = (override or source_root / "case_study" / "page_parallelism_20261102" / "live" / default_run)
        try:
            reports.append(compare_route(route, source_root, source_run.expanduser().resolve(), output / route))
        except Exception as exc:
            log = output / f"{route}_failure.log"
            log.write_text(traceback.format_exc(), encoding="utf-8")
            reports.append({"route": route, "status": "failed", "error": str(exc), "log": str(log)})
            print(f"{route}: failed; see {log}", file=sys.stderr, flush=True)
    equivalent = all(report["status"] == "equivalent" for report in reports)
    report = {
        "status": "equivalent" if equivalent else "failed_or_different",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output), "model_calls": 0,
        "scope": "Original and packaged compilers receive the same saved accepted plans and assets.",
        "limitations": [
            "This verifies deterministic rendering equivalence, not a new model generation.",
            "Existing visual defects and overflow are preserved; equivalence is not a quality pass.",
            "PPTX ZIP timestamps and document metadata are excluded; every slide XML part is compared.",
            "Pixel equality requires both renders to use the same fonts, LibreOffice and Poppler installation.",
        ],
        "routes": reports,
    }
    write_json(output / "render_parity.json", report)
    print(json.dumps({"status": report["status"], "report": str(output / "render_parity.json"),
                      "routes": [{"route": item["route"], "status": item["status"]} for item in reports]}, indent=2), flush=True)
    return 0 if equivalent else 1


if __name__ == "__main__":
    raise SystemExit(main())
