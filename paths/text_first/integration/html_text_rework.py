from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

from v9_deck_planning.json_utils import extract_json, write_json, write_text


CANVAS = (1536, 864)
TEXT_PART_KINDS = {"lead", "body", "bullet"}
MAX_PARTS_PER_BLOCK = 5
MAX_RUNS_PER_PART = 8
MAX_BULLETS_PER_BLOCK = 4
READABLE_ROLE_MIN_PX = {"title": 40.0, "subtitle": 24.0, "body": 20.0}
ROLE_FONT_WEIGHT = {"title": 700, "subtitle": 400}


def inspect_body_blocks(html_path: Path) -> list[dict[str, Any]]:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": CANVAS[0], "height": CANVAS[1]})
        page.goto(html_path.resolve().as_uri(), wait_until="load")
        page.evaluate("document.fonts.ready")
        blocks = page.evaluate(
            """() => [...document.querySelectorAll('[data-content-key][data-role="body"]')]
              .map(node => {
                const rect = node.getBoundingClientRect();
                const style = getComputedStyle(node);
                return {
                  content_key: node.dataset.contentKey || '',
                  region_id: node.dataset.regionId || '',
                  text: (node.innerText ?? node.textContent ?? '').replace(/\s+/g, ' ').trim(),
                  bbox_px: [rect.x, rect.y, rect.width, rect.height],
                  font_size_px: parseFloat(style.fontSize) || 16,
                  line_height_px: parseFloat(style.lineHeight) || null,
                  overflow: node.scrollWidth > node.clientWidth + 1 || node.scrollHeight > node.clientHeight + 1
                };
              })"""
        )
        browser.close()
    return blocks


def build_text_rework_prompt(
    *,
    slide_id: str,
    text_tree: dict[str, Any],
    current_blocks: list[dict[str, Any]],
) -> str:
    authority = {
        "slide_id": slide_id,
        "source_text_tree": text_tree,
        "current_body_blocks": current_blocks,
    }
    return f"""Improve only the internal presentation of selected body-text blocks on slide {slide_id}.
Return JSON only:
{{"slide_id":"{slide_id}","blocks":[{{"content_key":"text_01","preserved_meaning":true,"parts":[{{"kind":"lead|body|bullet","runs":[{{"text":"...","bold":false}}]}}]}}]}}

Rules:
- Keep every fact, number, entity, relationship, and conclusion. Add no claim and change no meaning.
- Do not edit title or subtitle. Omit a body block when it is already presentation-friendly.
- Useful edits: bold a few exact phrases or split the existing wording into 2-4 bullets. Preserve the committed words,
  numbers, signs, units, punctuation, and their order. Do not add a lead or takeaway unless those words already occur there.
- A lead is at most 12 words. Use at most 5 parts and 4 bullets per block.
- Never bold a whole multi-sentence block. Bold only the lead or a few short phrases.
- Keep the result no longer than the current block. Prefer shorter lines; never expand into an essay.
- Do not return HTML, CSS, layout, coordinates, font sizes, colors, notes, or unregistered text.

Semantic and space authority:
{json.dumps(authority, ensure_ascii=False, separators=(',', ':'))}
"""


def parse_text_rework_response(
    raw: str,
    *,
    slide_id: str,
    allowed_blocks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = extract_json(raw)
    if not isinstance(payload, dict):
        raise ValueError("text rework response must be one JSON object")
    if payload.get("slide_id") != slide_id:
        raise ValueError("text rework slide_id does not match")
    raw_blocks = payload.get("blocks")
    if not isinstance(raw_blocks, list):
        raise ValueError("text rework blocks must be an array")

    blocks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_block in raw_blocks:
        if not isinstance(raw_block, dict):
            raise ValueError("each text rework block must be an object")
        content_key = str(raw_block.get("content_key") or "").strip()
        if content_key in seen or content_key not in allowed_blocks:
            raise ValueError(f"unknown or duplicate text rework content_key: {content_key}")
        if raw_block.get("preserved_meaning") is not True:
            raise ValueError(f"{content_key} must affirm preserved_meaning")
        parts = _normalize_parts(raw_block.get("parts"), content_key)
        _validate_reworked_copy(
            content_key=content_key,
            source_text=str(allowed_blocks[content_key].get("text") or ""),
            parts=parts,
        )
        blocks.append({"content_key": content_key, "parts": parts})
        seen.add(content_key)
    return {"schema_version": "1.0", "slide_id": slide_id, "blocks": blocks}


def apply_text_rework_plan(
    *, html_path: Path, plan: dict[str, Any]
) -> dict[str, Any]:
    original_document = html_path.read_text(encoding="utf-8")
    requested = [str(item["content_key"]) for item in plan.get("blocks", [])]
    if not requested:
        return {
            "status": "unchanged",
            "accepted_content_keys": [],
            "rejected_content_keys": [],
        }

    first = _render_text_plan(original_document, plan)
    rejected = set(first["overflow_content_keys"])
    if first["geometry_changed"]:
        rejected.update(requested)
    accepted = [key for key in requested if key not in rejected]
    accepted_plan = {
        **plan,
        "blocks": [
            item for item in plan["blocks"] if item["content_key"] in accepted
        ],
    }
    final = _render_text_plan(original_document, accepted_plan)
    if final["overflow_content_keys"] or final["geometry_changed"]:
        accepted = []
        rejected = set(requested)
        final = _render_text_plan(original_document, {**plan, "blocks": []})
    write_text(html_path, final["document"])
    return {
        "status": "accepted" if accepted else "rejected",
        "accepted_content_keys": accepted,
        "rejected_content_keys": sorted(rejected),
        "overflow_content_keys": final["overflow_content_keys"],
        "geometry_changed": final["geometry_changed"],
    }


def build_typography_profile_prompt(metrics: dict[str, Any]) -> str:
    return f"""Choose one shared title font size and one shared subtitle font size for this deck.
Return JSON only: {{"title_font_size_px":40,"subtitle_font_size_px":20}}

Rules:
- Change font size only. Title weight 700 and subtitle weight 400 are owned by the renderer.
  Do not propose layout, wording, weight, color, or per-slide exceptions.
- Every title must use exactly the same size. Every subtitle must use exactly the same size.
- Stay at or below each role's max_safe_shared_px so no text overflows.
- Prefer a clear presentation hierarchy without making either role unnecessarily small.
- Prefer at least 40px titles and 24px subtitles. If a role cannot fit this readable floor,
  preserve it with an explicit unresolved finding rather than shrinking the whole deck.

Measured deck authority:
{json.dumps(metrics, ensure_ascii=False, separators=(',', ':'))}
"""


def inspect_deck_typography(html_by_slide: dict[str, Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": CANVAS[0], "height": CANVAS[1]})
        for slide_id, html_path in html_by_slide.items():
            page.goto(html_path.resolve().as_uri(), wait_until="load")
            page.evaluate("document.fonts.ready")
            slide_records = page.evaluate(
                """() => [...document.querySelectorAll('[data-content-key][data-role="title"], [data-content-key][data-role="subtitle"]')]
                  .map(node => {
                    const rect = node.getBoundingClientRect();
                    const style = getComputedStyle(node);
                    const original = node.style.fontSize;
                    const originalWeight = node.style.fontWeight;
                    const cssHeight = parseFloat(style.height);
                    const usedHeight = cssHeight + (style.boxSizing === 'border-box' ? 0 :
                      ['Top','Bottom'].reduce((s,side)=>s+(parseFloat(style[`padding${side}`])||0)+(parseFloat(style[`border${side}Width`])||0),0));
                    const scaleY = usedHeight > 0 ? rect.height / usedHeight : 1;
                    const weight = node.dataset.role === 'title' ? 700 : 400;
                    node.style.setProperty('font-weight', String(weight), 'important');
                    for (const child of node.querySelectorAll('*')) {
                      child.style.setProperty('font-weight', String(weight), 'important');
                      child.style.setProperty('font-size', 'inherit', 'important');
                    }
                    let low = 1;
                    let high = 96;
                    let best = 1;
                    for (let i = 0; i < 12; i += 1) {
                      const mid = (low + high) / 2;
                      node.style.setProperty('font-size', `${mid / Math.max(scaleY, 0.001)}px`, 'important');
                      const fits = node.scrollWidth <= node.clientWidth + 1 && node.scrollHeight <= node.clientHeight + 1;
                      if (fits) { best = mid; low = mid; } else { high = mid; }
                    }
                    if (original) node.style.fontSize = original; else node.style.removeProperty('font-size');
                    if (originalWeight) node.style.fontWeight = originalWeight; else node.style.removeProperty('font-weight');
                    return {
                      content_key: node.dataset.contentKey || '',
                      role: node.dataset.role || '',
                      text: (node.innerText ?? node.textContent ?? '').replace(/\s+/g, ' ').trim(),
                      bbox_px: [rect.x, rect.y, rect.width, rect.height],
                      current_font_size_px: (parseFloat(style.fontSize) || 16) * scaleY,
                      max_safe_px: Math.floor(best * 10) / 10
                    };
                  })"""
            )
            for record in slide_records:
                records.append({"slide_id": slide_id, **record})
        browser.close()

    role_limits = {}
    for role in ("title", "subtitle"):
        role_records = [item for item in records if item["role"] == role]
        role_limits[role] = (
            round(min(item["max_safe_px"] for item in role_records), 1)
            if role_records
            else None
        )
    return {
        "elements": records,
        "max_safe_shared_px": role_limits,
    }


def parse_typography_profile(raw: str, metrics: dict[str, Any]) -> dict[str, Any]:
    payload = extract_json(raw)
    if not isinstance(payload, dict):
        raise ValueError("typography profile response must be one JSON object")
    profile: dict[str, Any] = {"schema_version": "1.0"}
    for role, field in (
        ("title", "title_font_size_px"),
        ("subtitle", "subtitle_font_size_px"),
    ):
        limit = metrics["max_safe_shared_px"].get(role)
        if limit is None:
            profile[field] = None
            continue
        try:
            requested = float(payload[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"typography profile requires numeric {field}") from exc
        if not math.isfinite(requested) or requested <= 0:
            raise ValueError(f"typography profile {field} must be positive")
        profile[field] = round(min(requested, float(limit)), 1)
    return profile


def apply_deck_typography_profile(
    *, html_by_slide: dict[str, Path], profile: dict[str, Any]
) -> dict[str, Any]:
    requested = {
        "title": profile.get("title_font_size_px"),
        "subtitle": profile.get("subtitle_font_size_px"),
    }
    original_documents = {
        slide_id: path.read_text(encoding="utf-8")
        for slide_id, path in html_by_slide.items()
    }
    documents = dict(original_documents)
    resolved: dict[str, float | None] = {"title": None, "subtitle": None}
    role_results: dict[str, dict[str, Any]] = {}
    # The two role decisions are independent. An auto-height subtitle must not
    # discard a safe, common title size on every other slide.
    for role, value in requested.items():
        if value is None:
            role_results[role] = {"status": "skipped"}
            continue
        value = float(value)
        floor = READABLE_ROLE_MIN_PX[role]
        attempts = []
        # A bounded half-pixel search preserves geometry and committed copy.
        # Do not search down to 1px to turn an impossible layout into a "pass".
        for step in range(13):
            candidate = round(value - step * 0.5, 1)
            if candidate < floor:
                break
            results = {
                slide_id: _render_typography_profile(document, {role: candidate})
                for slide_id, document in documents.items()
            }
            geometry_changed = any(result["geometry_changed"] for result in results.values())
            overflow = any(role in result["overflow_roles"] for result in results.values())
            attempts.append({"font_size_px": candidate, "geometry_changed": geometry_changed, "overflow": overflow})
            if not geometry_changed and not overflow:
                documents = {slide_id: result["document"] for slide_id, result in results.items()}
                resolved[role] = candidate
                role_results[role] = {"status": "accepted", "font_size_px": candidate,
                                      "font_weight": ROLE_FONT_WEIGHT[role], "attempts": attempts}
                break
            if geometry_changed:
                break
        if role not in role_results:
            role_results[role] = {"status": "rejected", "reason": "no_readable_geometry_preserving_profile",
                                  "minimum_font_size_px": floor, "attempts": attempts}

    for slide_id, document in documents.items():
        if document != original_documents[slide_id]:
            write_text(html_by_slide[slide_id], document)
    accepted = [role for role, result in role_results.items() if result["status"] == "accepted"]
    rejected = [role for role, result in role_results.items() if result["status"] == "rejected"]
    return {
        "status": "partial" if accepted and rejected else "rejected" if rejected else "accepted",
        "requested": requested,
        "resolved": resolved,
        "role_results": role_results,
        "unresolved_roles": rejected,
        "overflow_roles": rejected,
    }


def create_preview_montage(previews: list[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in previews]
    try:
        if not images:
            raise ValueError("typography montage requires at least one preview")
        thumb_w = 768
        thumb_h = 432
        gap = 16
        columns = min(2, len(images))
        rows = math.ceil(len(images) / columns)
        canvas = Image.new(
            "RGB",
            (columns * thumb_w + (columns + 1) * gap, rows * (thumb_h + 28) + (rows + 1) * gap),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for index, source in enumerate(images):
            image = source.resize((thumb_w, thumb_h))
            column = index % columns
            row = index // columns
            x = gap + column * (thumb_w + gap)
            y = gap + row * (thumb_h + 28 + gap)
            canvas.paste(image, (x, y))
            draw.text((x, y + thumb_h + 4), f"Slide {index + 1}", fill="black")
        canvas.save(output_path)
    finally:
        for image in images:
            image.close()


def write_text_rework_artifacts(
    *, slide_dir: Path, prompt: str, raw: str, plan: dict[str, Any], report: dict[str, Any]
) -> None:
    write_text(slide_dir / "text_rework_prompt.txt", prompt)
    write_text(slide_dir / "text_rework_raw_response.txt", raw)
    write_json(slide_dir / "text_rework_plan.json", plan)
    write_json(slide_dir / "text_rework_report.json", report)


def _normalize_parts(raw_parts: Any, content_key: str) -> list[dict[str, Any]]:
    if not isinstance(raw_parts, list) or not 1 <= len(raw_parts) <= MAX_PARTS_PER_BLOCK:
        raise ValueError(f"{content_key} parts must contain 1-{MAX_PARTS_PER_BLOCK} items")
    parts = []
    bullet_count = 0
    for raw_part in raw_parts:
        if not isinstance(raw_part, dict):
            raise ValueError(f"{content_key} parts must be objects")
        kind = str(raw_part.get("kind") or "").strip().lower()
        if kind not in TEXT_PART_KINDS:
            raise ValueError(f"{content_key} has invalid part kind: {kind}")
        if kind == "bullet":
            bullet_count += 1
        raw_runs = raw_part.get("runs")
        if not isinstance(raw_runs, list) or not 1 <= len(raw_runs) <= MAX_RUNS_PER_PART:
            raise ValueError(f"{content_key} part runs must contain 1-{MAX_RUNS_PER_PART} items")
        runs = []
        for raw_run in raw_runs:
            if not isinstance(raw_run, dict):
                raise ValueError(f"{content_key} runs must be objects")
            text = re.sub(r"\s+", " ", str(raw_run.get("text") or "")).strip()
            if not text:
                raise ValueError(f"{content_key} contains an empty run")
            runs.append({"text": text, "bold": bool(raw_run.get("bold", False))})
        parts.append({"kind": kind, "runs": runs})
    if bullet_count not in {0, 2, 3, 4} or bullet_count > MAX_BULLETS_PER_BLOCK:
        raise ValueError(f"{content_key} must use either no bullets or 2-4 bullets")
    if sum(part["kind"] == "lead" for part in parts) > 1:
        raise ValueError(f"{content_key} may contain at most one lead")
    return parts


def canonical_committed_tokens(value: str) -> list[str]:
    """Ignore layout whitespace/list markers, never facts or lexical content."""
    lines = [
        re.sub(r"^\s*(?:[•●▪◦‣⁃*+-]\s+|\d+[.)、]\s+)", "", line)
        for line in str(value).splitlines()
    ]
    return re.findall(r"[A-Za-z0-9_]+|[^\s]", "\n".join(lines))


def _validate_reworked_copy(
    *, content_key: str, source_text: str, parts: list[dict[str, Any]]
) -> None:
    source = re.sub(r"\s+", " ", source_text).strip()
    output = " ".join(
        run["text"] for part in parts for run in part["runs"]
    ).strip()
    if not source or not output:
        raise ValueError(f"{content_key} must keep non-empty copy")
    max_length = max(len(source) + 8, math.ceil(len(source) * 1.05))
    if len(output) > max_length:
        raise ValueError(f"{content_key} text rework expanded the block")
    if len(output) < max(8, math.floor(len(source) * 0.6)):
        raise ValueError(f"{content_key} text rework removed too much source content")
    number_pattern = r"[+−-]?\d+(?:[.,]\d+)*(?:\s*%)?"
    source_numbers = re.findall(number_pattern, source)
    output_numbers = re.findall(number_pattern, output)
    if source_numbers != output_numbers:
        raise ValueError(f"{content_key} text rework changed numeric facts")
    if canonical_committed_tokens(source_text) != canonical_committed_tokens(output):
        raise ValueError(f"{content_key} text rework changed committed copy")
    bold_chars = sum(
        len(run["text"])
        for part in parts
        for run in part["runs"]
        if run["bold"] or part["kind"] == "lead"
    )
    if len(output) > 24 and bold_chars >= len(output) * 0.55:
        raise ValueError(f"{content_key} text rework bolded too much of the block")
    for part in parts:
        if part["kind"] == "lead":
            lead = " ".join(run["text"] for run in part["runs"])
            if len(lead.split()) > 12:
                raise ValueError(f"{content_key} lead exceeds 12 words")


def _render_text_plan(document: str, plan: dict[str, Any]) -> dict[str, Any]:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": CANVAS[0], "height": CANVAS[1]})
        page.set_content(document, wait_until="load")
        page.evaluate("document.fonts.ready")
        before = _geometry_snapshot(page)
        page.evaluate(
            """payload => {
              const appendRuns = (parent, runs) => {
                let previous = '';
                runs.forEach((run, index) => {
                  const node = run.bold ? document.createElement('strong') : document.createElement('span');
                  const text = String(run.text || '');
                  const punctuation = /^[,.;:!?%\)\]\}]/.test(text);
                  const openEnded = /[\(\[\{\/-]$/.test(previous);
                  node.textContent = `${index && !punctuation && !openEnded ? ' ' : ''}${text}`;
                  parent.appendChild(node);
                  previous = text;
                });
              };
              for (const block of payload.blocks || []) {
                const node = document.querySelector(`[data-content-key="${CSS.escape(block.content_key)}"]`);
                if (!node || node.dataset.role !== 'body') continue;
                node.replaceChildren();
                node.dataset.textReworked = 'true';
                for (const part of block.parts) {
                  // A span remains valid inside model-authored <p> regions; display:block
                  // supplies paragraph behavior without browser reparsing it outside the block.
                  const line = document.createElement('span');
                  line.dataset.textPart = part.kind;
                  line.style.display = 'block';
                  line.style.letterSpacing = '0';
                  line.style.margin = part.kind === 'lead' ? '0 0 0.28em 0' : '0 0 0.18em 0';
                  if (part.kind === 'lead') {
                    line.style.fontSize = '1.08em';
                    line.style.fontWeight = '700';
                    line.style.lineHeight = '1.16';
                  }
                  if (part.kind === 'bullet') {
                    line.style.paddingLeft = '1em';
                    line.style.textIndent = '-0.78em';
                    const marker = document.createElement('span');
                    marker.dataset.generatedBullet = 'true';
                    marker.textContent = '• ';
                    line.appendChild(marker);
                  }
                  appendRuns(line, part.runs);
                  node.appendChild(line);
                }
              }
            }""",
            plan,
        )
        page.evaluate("document.fonts.ready")
        after = _geometry_snapshot(page)
        overflow = page.evaluate(
            """() => [...document.querySelectorAll('[data-text-reworked="true"]')]
              .filter(node => node.scrollWidth > node.clientWidth + 1 || node.scrollHeight > node.clientHeight + 1)
              .map(node => node.dataset.contentKey || '')"""
        )
        serialized = "<!doctype html>\n" + page.locator("html").evaluate("node => node.outerHTML") + "\n"
        page.set_content(serialized, wait_until="load")
        page.evaluate("document.fonts.ready")
        reloaded = _geometry_snapshot(page)
        overflow = page.evaluate(
            """() => [...document.querySelectorAll('[data-text-reworked="true"]')]
              .filter(node => node.scrollWidth > node.clientWidth + 1 || node.scrollHeight > node.clientHeight + 1)
              .map(node => node.dataset.contentKey || '')"""
        )
        browser.close()
    return {
        "document": serialized,
        "overflow_content_keys": sorted(set(overflow)),
        "geometry_changed": (
            not _same_geometry(before, after) or not _same_geometry(before, reloaded)
        ),
    }


def _render_typography_profile(document: str, sizes: dict[str, Any]) -> dict[str, Any]:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": CANVAS[0], "height": CANVAS[1]})
        page.set_content(document, wait_until="load")
        page.evaluate("document.fonts.ready")
        before = _geometry_snapshot(page)
        page.evaluate(
            """sizes => {
              for (const role of ['title', 'subtitle']) {
                const value = sizes[role];
                if (value == null) continue;
                for (const node of document.querySelectorAll(`[data-content-key][data-role="${role}"]`)) {
                  const style = getComputedStyle(node);
                  const rect = node.getBoundingClientRect();
                  const cssHeight = parseFloat(style.height);
                  const usedHeight = cssHeight + (style.boxSizing === 'border-box' ? 0 :
                    ['Top','Bottom'].reduce((s,side)=>s+(parseFloat(style[`padding${side}`])||0)+(parseFloat(style[`border${side}Width`])||0),0));
                  const scaleY = usedHeight > 0 ? rect.height / usedHeight : 1;
                  for (const property of [
                    'margin-top', 'margin-right', 'margin-bottom', 'margin-left',
                    'padding-top', 'padding-right', 'padding-bottom', 'padding-left'
                  ]) {
                    node.style.setProperty(property, style.getPropertyValue(property), 'important');
                  }
                  // Reserve the original physical text box before changing
                  // typography. Auto-sized subtitles otherwise veto the deck
                  // simply because their line box grows by a pixel.
                  node.style.setProperty('width', style.width, 'important');
                  node.style.setProperty('height', style.height, 'important');
                  node.style.setProperty('font-size', `${value / Math.max(scaleY, 0.001)}px`, 'important');
                  node.style.setProperty('font-weight', role === 'title' ? '700' : '400', 'important');
                  for (const child of node.querySelectorAll('*')) {
                    child.style.setProperty('font-weight', role === 'title' ? '700' : '400', 'important');
                    child.style.setProperty('font-size', 'inherit', 'important');
                  }
                  node.dataset.deckTypographyProfile = 'true';
                }
              }
            }""",
            sizes,
        )
        page.evaluate("document.fonts.ready")
        after = _geometry_snapshot(page)
        overflow_roles = page.evaluate(
            """() => [...document.querySelectorAll('[data-deck-typography-profile="true"]')]
              .filter(node => node.scrollWidth > node.clientWidth + 1 || node.scrollHeight > node.clientHeight + 1)
              .map(node => node.dataset.role || '')"""
        )
        serialized = "<!doctype html>\n" + page.locator("html").evaluate("node => node.outerHTML") + "\n"
        browser.close()
    return {
        "document": serialized,
        "overflow_roles": sorted(set(overflow_roles)),
        "geometry_changed": not _same_geometry(before, after),
    }


def _geometry_snapshot(page: Any) -> dict[str, list[float]]:
    return page.evaluate(
        """() => Object.fromEntries(
          [...document.querySelectorAll('[data-region-id], [data-module-id], [data-holder-id]')]
            .map((node, index) => {
              const rect = node.getBoundingClientRect();
              const key = node.dataset.regionId ? `region:${node.dataset.regionId}` :
                node.dataset.moduleId ? `module:${node.dataset.moduleId}` :
                node.dataset.holderId ? `holder:${node.dataset.holderId}` : `node:${index}`;
              return [key, [rect.x, rect.y, rect.width, rect.height]];
            })
        )"""
    )


def _same_geometry(
    before: dict[str, list[float]], after: dict[str, list[float]], tolerance: float = 1.0
) -> bool:
    if set(before) != set(after):
        return False
    return all(
        all(abs(left - right) <= tolerance for left, right in zip(before[key], after[key]))
        for key in before
    )
