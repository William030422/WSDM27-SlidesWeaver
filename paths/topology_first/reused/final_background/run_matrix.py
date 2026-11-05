from __future__ import annotations

import argparse
import colorsys
import json
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat


ROOT = Path(__file__).resolve().parent
RL4PPT_ROOT = ROOT.parents[4]
V9_ROOT = RL4PPT_ROOT / "demo-v9-step-by-step-improvement"
BACKGROUND_PATH = (
    V9_ROOT
    / "outputs"
    / "grid_energy_binding_e2e_20260714-171758"
    / "step2"
    / "background_raw.png"
)
LAYOUT_PATH = (
    ROOT.parent
    / "planning_deck_v2_postprocessed"
    / "layouts"
    / "slide_004.json"
)
BRIEFS_PATH = ROOT / "component_briefs.json"
INPUT_DIR = ROOT / "inputs"
PROMPT_DIR = ROOT / "prompts"
OUTPUT_DIR = ROOT / "outputs"
WHITE_REFERENCE = INPUT_DIR / "reference_white_frames.png"
COLOR_REFERENCE = INPUT_DIR / "reference_blurred_color_fields.png"
CANVAS = (1536, 864)
FRAME_WIDTH = 6
COLOR_ALPHA = 0.60
COLOR_BLUR_RADIUS = 40

VARIANTS = (
    {
        "slug": "soft_prompt",
        "reference": WHITE_REFERENCE,
        "marker": "three hollow white rectangles",
        "prompt_mode": "soft",
    },
    {
        "slug": "color_control",
        "reference": COLOR_REFERENCE,
        "marker": "three soft blurred color fields",
        "prompt_mode": "strict",
    },
    {
        "slug": "combined",
        "reference": COLOR_REFERENCE,
        "marker": "three soft blurred color fields",
        "prompt_mode": "soft",
    },
)

MODELS = (
    {
        "slug": "gpt_image_2_official",
        "model": "gpt-image-2-official",
        "size": "1536x864",
        "resolution": "1k",
        "quality": "high",
        "output_format": "png",
        "reference_transport": "base64",
    },
    {
        "slug": "nano_banana",
        "model": "gemini-3-pro-image-preview",
        "size": "16:9",
        "resolution": "1K",
        "reference_transport": "base64",
    },
    {
        "slug": "doubao_seedream_5",
        "model": "doubao-seedream-5-0-pro",
        "size": "16:9",
        "resolution": "1K",
        "output_format": "png",
        "watermark": False,
        "reference_transport": "base64",
    },
    {
        "slug": "qwen_image_2_0",
        "model": "qwen-image-2.0",
        "size": "16:9",
        "resolution": "1K",
        "negative_prompt": (
            "hard rectangular panels, frames, borders, cards, UI, charts, readable text, "
            "labels, captions, logos, watermarks, harsh cutout edges"
        ),
        "reference_transport": "public_url",
    },
)

sys.path.insert(0, str(V9_ROOT / "src"))
from v9_deck_planning.apimart import (  # noqa: E402
    ApiMartClient,
    extract_image_url,
    image_to_data_uri,
    redacted_image_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 4-model x 3-variant soft-boundary test.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--resume-missing",
        action="store_true",
        help="Reuse successful outputs from the previous manifest and submit only missing items.",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_regions() -> dict[str, list[int]]:
    payload = json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))
    if payload.get("canvas") != {"width": CANVAS[0], "height": CANVAS[1]}:
        raise ValueError(f"unexpected layout canvas: {payload.get('canvas')}")
    regions = {
        entity["region_id"]: entity["postprocessed_bbox_px"]
        for entity in payload["entities"]
        if str(entity.get("region_id", "")).startswith("visual_")
    }
    expected = {"visual_001", "visual_002", "visual_003"}
    if set(regions) != expected:
        raise ValueError(f"expected {sorted(expected)}, got {sorted(regions)}")
    return regions


def load_briefs(region_ids: set[str]) -> list[dict[str, str]]:
    payload = json.loads(BRIEFS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("component_briefs.json must contain a JSON array")
    expected_keys = {"component_id", "position", "visual_brief"}
    for item in payload:
        if not isinstance(item, dict) or set(item) != expected_keys:
            raise ValueError(f"each brief must contain exactly {sorted(expected_keys)}")
    component_ids = [item["component_id"] for item in payload]
    if len(component_ids) != len(set(component_ids)) or set(component_ids) != region_ids:
        raise ValueError("component briefs and visual regions must be one-to-one")
    return payload


def _bbox_xyxy(bbox: list[int]) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    return x, y, x + width - 1, y + height - 1


def render_white_reference(regions: dict[str, list[int]]) -> None:
    with Image.open(BACKGROUND_PATH).convert("RGB") as background:
        if background.size != CANVAS:
            raise ValueError(f"background must be {CANVAS}, got {background.size}")
        reference = background.copy()
    draw = ImageDraw.Draw(reference)
    for region_id in sorted(regions):
        draw.rectangle(
            _bbox_xyxy(regions[region_id]),
            outline=(255, 255, 255),
            width=FRAME_WIDTH,
        )
    WHITE_REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    reference.save(WHITE_REFERENCE)


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    channels = []
    for value in rgb:
        channel = value / 255
        channels.append(channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast_ratio(first: tuple[int, int, int], second: tuple[int, int, int]) -> float:
    high, low = sorted((_relative_luminance(first), _relative_luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def choose_contrast_color(average: tuple[int, int, int]) -> tuple[int, int, int]:
    red, green, blue = (value / 255 for value in average)
    hue, _, saturation = colorsys.rgb_to_hls(red, green, blue)
    hue = (hue + 0.5) % 1.0
    saturation = max(saturation, 0.65)
    candidates = []
    for lightness in (0.22, 0.78):
        candidate = tuple(
            round(value * 255) for value in colorsys.hls_to_rgb(hue, lightness, saturation)
        )
        candidates.append(candidate)
    return max(candidates, key=lambda color: _contrast_ratio(average, color))


def render_color_reference(regions: dict[str, list[int]]) -> list[dict[str, Any]]:
    with Image.open(BACKGROUND_PATH).convert("RGB") as background:
        reference = background.copy()
        source = background.copy()
    report = []
    peak_alpha = round(255 * COLOR_ALPHA)
    for region_id in sorted(regions):
        bbox = regions[region_id]
        average = tuple(round(value) for value in ImageStat.Stat(source.crop(_bbox_xyxy(bbox))).mean)
        color = choose_contrast_color(average)
        mask = Image.new("L", CANVAS, 0)
        ImageDraw.Draw(mask).rectangle(_bbox_xyxy(bbox), fill=peak_alpha)
        mask = mask.filter(ImageFilter.GaussianBlur(COLOR_BLUR_RADIUS))
        color_layer = Image.new("RGB", CANVAS, color)
        reference = Image.composite(color_layer, reference, mask)
        report.append(
            {
                "region_id": region_id,
                "bbox_px": bbox,
                "background_average_rgb": list(average),
                "overlay_rgb": list(color),
                "contrast_ratio": round(_contrast_ratio(average, color), 3),
                "peak_opacity": COLOR_ALPHA,
                "gaussian_blur_radius_px": COLOR_BLUR_RADIUS,
            }
        )
    COLOR_REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    reference.save(COLOR_REFERENCE)
    write_json(INPUT_DIR / "blurred_color_fields_report.json", report)
    return report


def compile_prompts(briefs: list[dict[str, str]]) -> dict[str, str]:
    mapping = "\n".join(
        f"{item['component_id']} {item['position']}: {item['visual_brief']}." for item in briefs
    )
    prompts = {}
    for variant in VARIANTS:
        marker = variant["marker"]
        if variant["prompt_mode"] == "strict":
            prompt = (
                f"Edit this single reference. The {marker} mark exact target regions.\n"
                f"One-to-one mapping:\n{mapping}\n"
                "Remove the guide tint. Keep each concept inside its field and preserve everything outside. Generate exactly "
                "one concept per field; no swaps, merges, duplicates, or omissions. Match the clean graphic-novel ink/halftone "
                "landscape. No panels, borders, cards, UI, charts, labels, text, logos, or watermarks."
            )
        else:
            prompt = (
                f"Edit this single reference. The {marker} are approximate semantic anchors, not clipping boxes.\n"
                f"One-to-one mapping:\n{mapping}\n"
                "Erase the guides. Center each concept near its anchor, but allow forms, light, and haze to extend past it "
                "and feather into the landscape. Use soft local contrast to separate meaning from background without hard "
                "edges. Preserve the background and match the graphic-novel ink/halftone style. "
                "No panels, borders, cards, UI, charts, labels, text, logos, or watermarks."
            )
        if len(prompt) > 800:
            raise ValueError(f"{variant['slug']} prompt exceeds Qwen's 800-character limit: {len(prompt)}")
        prompt_path = PROMPT_DIR / f"{variant['slug']}.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        prompts[variant["slug"]] = prompt
    return prompts


def _validate_public_url(url: str) -> None:
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"}, method="GET")
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status not in {200, 206}:
            raise RuntimeError(f"public reference URL returned HTTP {response.status}: {url}")


def _upload_tmpfiles(path: Path) -> str:
    result = subprocess.run(
        ["curl", "-fsS", "-F", f"file=@{path}", "https://tmpfiles.org/api/v1/upload"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    payload = json.loads(result.stdout)
    url = str(payload["data"]["url"])
    return url.replace("https://tmpfiles.org/", "https://tmpfiles.org/dl/", 1)


def _upload_litterbox(path: Path) -> str:
    result = subprocess.run(
        [
            "curl",
            "-fsS",
            "-F",
            "reqtype=fileupload",
            "-F",
            "time=24h",
            "-F",
            f"fileToUpload=@{path}",
            "https://litterbox.catbox.moe/resources/internals/api.php",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    url = result.stdout.strip()
    if not url.startswith("https://"):
        raise RuntimeError(f"unexpected Litterbox response: {url[:200]}")
    return url


def upload_public_reference(path: Path) -> dict[str, str]:
    errors = []
    for provider, uploader in (("tmpfiles.org", _upload_tmpfiles), ("litterbox", _upload_litterbox)):
        try:
            url = uploader(path)
            _validate_public_url(url)
            return {"provider": provider, "url": url}
        except Exception as exc:
            errors.append(f"{provider}: {exc}")
    raise RuntimeError("could not upload Qwen reference image: " + " | ".join(errors))


def prepare_public_references(resume_missing: bool) -> dict[str, dict[str, str]]:
    record_path = ROOT / "public_references.json"
    previous: dict[str, dict[str, str]] = {}
    if resume_missing and record_path.exists():
        loaded = json.loads(record_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            previous = loaded
    references = {}
    for path in (WHITE_REFERENCE, COLOR_REFERENCE):
        key = str(path)
        record = previous.get(key)
        if isinstance(record, dict) and isinstance(record.get("url"), str):
            try:
                _validate_public_url(record["url"])
                references[key] = record
                continue
            except Exception:
                pass
        references[key] = upload_public_reference(path)
    write_json(record_path, references)
    return references


def load_previous_successes() -> list[dict[str, Any]]:
    manifest_path = OUTPUT_DIR / "run_manifest.json"
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    successes = []
    for item in payload.get("results", []):
        output = Path(str(item.get("output") or ""))
        if item.get("status") == "succeeded" and output.is_file():
            successes.append(item)
    return successes


def build_payload(
    model_spec: dict[str, Any], prompt: str, reference_path: Path, public_url: str | None
) -> dict[str, Any]:
    if model_spec["reference_transport"] == "public_url":
        if not public_url:
            raise ValueError("Qwen requires a public reference URL")
        reference_value = public_url
    else:
        reference_value = image_to_data_uri(reference_path)
    payload: dict[str, Any] = {
        "model": model_spec["model"],
        "prompt": prompt,
        "size": model_spec["size"],
        "resolution": model_spec["resolution"],
        "n": 1,
        "image_urls": [reference_value],
    }
    for key in ("quality", "output_format", "watermark", "negative_prompt"):
        if key in model_spec:
            payload[key] = model_spec[key]
    if len(payload["image_urls"]) != 1 or "mask_url" in payload:
        raise AssertionError("every request must contain one reference image and no mask_url")
    return payload


def save_as_png(download_path: Path, output_path: Path) -> dict[str, Any]:
    with Image.open(download_path) as source:
        source_format = source.format
        source_mode = source.mode
        size = source.size
        source.convert("RGB").save(output_path, format="PNG")
    download_path.unlink()
    return {
        "source_format": source_format,
        "source_mode": source_mode,
        "output_mode": "RGB",
        "output_size": list(size),
    }


def run_call(
    model_spec: dict[str, Any],
    variant: dict[str, Any],
    prompt: str,
    public_references: dict[str, dict[str, str]],
) -> dict[str, Any]:
    model_slug = model_spec["slug"]
    variant_slug = variant["slug"]
    reference_path = Path(variant["reference"])
    model_dir = OUTPUT_DIR / model_slug
    request_path = model_dir / f"{variant_slug}.request.json"
    task_path = model_dir / f"{variant_slug}.task.json"
    output_path = model_dir / f"{variant_slug}.png"
    download_path = model_dir / f".{variant_slug}.download"
    public_record = public_references.get(str(reference_path))
    public_url = public_record["url"] if public_record else None
    payload = build_payload(model_spec, prompt, reference_path, public_url)
    write_json(
        request_path,
        {
            "model_slug": model_slug,
            "variant": variant_slug,
            "prompt_mode": variant["prompt_mode"],
            "input_reference": str(reference_path),
            "input_reference_count": 1,
            "reference_transport": model_spec["reference_transport"],
            "public_reference_url": public_url,
            "mask_url_present": False,
            "prompt_path": str(PROMPT_DIR / f"{variant_slug}.txt"),
            "prompt_character_count": len(prompt),
            "data_uris_redacted": True,
            "payload": redacted_image_payload(payload),
        },
    )

    client = ApiMartClient(env_path=V9_ROOT / ".env", poll_timeout_seconds=1800)
    task_id = client.submit_image_task(payload)
    print(f"submitted {model_slug}/{variant_slug}: {task_id}", flush=True)
    write_json(task_path, {"status": "submitted", "task_id": task_id})
    task = client.poll_task(task_id)
    write_json(task_path, task)
    client.download(extract_image_url(task), download_path)
    image_metadata = save_as_png(download_path, output_path)
    print(f"completed {model_slug}/{variant_slug}", flush=True)
    return {
        "status": "succeeded",
        "model_slug": model_slug,
        "model": model_spec["model"],
        "variant": variant_slug,
        "task_id": task_id,
        "output": str(output_path),
        "image_call_count": client.image_call_count,
        **image_metadata,
    }


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def render_contact_sheet(results: list[dict[str, Any]]) -> Path:
    thumb_size = (480, 270)
    label_height = 34
    left_margin = 190
    top_margin = 56
    sheet = Image.new(
        "RGB",
        (left_margin + thumb_size[0] * len(VARIANTS), top_margin + (thumb_size[1] + label_height) * len(MODELS)),
        (244, 246, 248),
    )
    draw = ImageDraw.Draw(sheet)
    header_font = _font(20)
    label_font = _font(18)
    for column, variant in enumerate(VARIANTS):
        draw.text(
            (left_margin + column * thumb_size[0] + 12, 16),
            variant["slug"],
            fill=(28, 33, 39),
            font=header_font,
        )
    by_key = {(item.get("model_slug"), item.get("variant")): item for item in results}
    for row, model in enumerate(MODELS):
        y = top_margin + row * (thumb_size[1] + label_height)
        draw.text((12, y + 120), model["slug"], fill=(28, 33, 39), font=label_font)
        for column, variant in enumerate(VARIANTS):
            x = left_margin + column * thumb_size[0]
            item = by_key.get((model["slug"], variant["slug"]), {})
            if item.get("status") == "succeeded":
                with Image.open(item["output"]).convert("RGB") as image:
                    image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                    tile = Image.new("RGB", thumb_size, (225, 228, 232))
                    tile.paste(
                        image,
                        ((thumb_size[0] - image.width) // 2, (thumb_size[1] - image.height) // 2),
                    )
            else:
                tile = Image.new("RGB", thumb_size, (225, 228, 232))
                ImageDraw.Draw(tile).text((20, 120), "FAILED", fill=(160, 35, 35), font=header_font)
            sheet.paste(tile, (x, y))
            draw.text((x + 10, y + thumb_size[1] + 7), variant["prompt_mode"], fill=(70, 76, 84), font=label_font)
    path = OUTPUT_DIR / "comparison_contact_sheet.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def main() -> None:
    args = parse_args()
    regions = load_regions()
    briefs = load_briefs(set(regions))
    render_white_reference(regions)
    color_report = render_color_reference(regions)
    prompts = compile_prompts(briefs)

    preparation = {
        "background": str(BACKGROUND_PATH),
        "layout": str(LAYOUT_PATH),
        "regions": regions,
        "white_reference": str(WHITE_REFERENCE),
        "color_reference": str(COLOR_REFERENCE),
        "color_report": color_report,
        "prompts": {
            slug: {"path": str(PROMPT_DIR / f"{slug}.txt"), "character_count": len(prompt)}
            for slug, prompt in prompts.items()
        },
    }
    write_json(ROOT / "preparation_manifest.json", preparation)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", **preparation}, ensure_ascii=False, indent=2))
        return

    public_references = prepare_public_references(args.resume_missing)
    previous_successes = load_previous_successes() if args.resume_missing else []
    results_by_key = {
        (item["model_slug"], item["variant"]): item for item in previous_successes
    }
    scheduled_count = 0
    for variant in VARIANTS:
        pending_models = [
            model
            for model in MODELS
            if (model["slug"], variant["slug"]) not in results_by_key
        ]
        if not pending_models:
            print(f"skipped completed variant {variant['slug']}", flush=True)
            continue
        scheduled_count += len(pending_models)
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    run_call,
                    model,
                    variant,
                    prompts[variant["slug"]],
                    public_references,
                ): model
                for model in pending_models
            }
            for future in as_completed(futures):
                model = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "status": "failed",
                        "model_slug": model["slug"],
                        "model": model["model"],
                        "variant": variant["slug"],
                        "error": str(exc),
                    }
                    print(f"failed {model['slug']}/{variant['slug']}: {exc}", flush=True)
                results_by_key[(model["slug"], variant["slug"])] = result

    results = [
        results_by_key[(model["slug"], variant["slug"])]
        for variant in VARIANTS
        for model in MODELS
    ]

    contact_sheet = render_contact_sheet(results)
    failures = [item for item in results if item["status"] != "succeeded"]
    manifest = {
        "status": "failed" if failures else "succeeded",
        "expected_image_calls": len(MODELS) * len(VARIANTS),
        "api_submission_attempts_this_run": scheduled_count,
        "reused_successful_outputs": len(previous_successes),
        "successful_image_calls": len(results) - len(failures),
        "failed_image_calls": len(failures),
        "single_reference_per_request": True,
        "mask_url_present": False,
        "gpt_model": "gpt-image-2-official",
        "gpt_size": "1536x864",
        "preparation": preparation,
        "public_references": public_references,
        "results": results,
        "contact_sheet": str(contact_sheet),
    }
    write_json(OUTPUT_DIR / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(results)} generation calls failed")


if __name__ == "__main__":
    main()
