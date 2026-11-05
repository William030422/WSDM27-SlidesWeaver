from __future__ import annotations

import html as html_lib
import hashlib
import json
import re
import time
import urllib.error
from pathlib import Path
from typing import Any, Callable, Protocol

from v9_deck_planning.apimart import (
    ApiMartClient,
    _extract_text_response,
    _extract_usage,
    image_to_data_uri,
)
from v9_deck_planning.json_utils import (
    call_json_with_retries,
    read_json,
    write_json,
    write_text,
)
from v9_deck_planning.models import DeckPlan

from .config import PipelineConfig
from .html_text_rework import (
    canonical_committed_tokens as _committed_copy_text,
    apply_deck_typography_profile,
    apply_text_rework_plan,
    READABLE_ROLE_MIN_PX,
    build_text_rework_prompt,
    build_typography_profile_prompt,
    create_preview_montage,
    inspect_body_blocks,
    inspect_deck_typography,
    parse_text_rework_response,
    parse_typography_profile,
    write_text_rework_artifacts,
)
from .text_generation import TEXT_FIELD_PATTERN, parse_text_plan


CANVAS = (1536, 864)
MIN_MEANINGFUL_VISUAL_WIDTH_PX = 280
MIN_MEANINGFUL_VISUAL_HEIGHT_PX = 180
MIN_MODULE_COVERAGE = 0.62


class HtmlPlanningClient(Protocol):
    last_usage: dict[str, Any] | None

    def chat_html(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
    ) -> str: ...

    def vlm_html(
        self,
        *,
        model: str,
        prompt: str,
        image_path: Path,
        max_tokens: int,
    ) -> str: ...

    def vlm_json(
        self,
        *,
        model: str,
        prompt: str,
        image_path: Path,
        max_tokens: int,
    ) -> str: ...


class JournaledHtmlPlanningClient:
    """Reuse completed page-local model calls after an interrupted refinement.

    The identity includes the image bytes, rather than only its mutable path.
    An ambiguous POST outcome is kept explicit so a restart cannot silently pay
    for the same request again. Completed raw answers remain available even if
    later parsing, browser work, or the final deck export fails.
    """

    def __init__(self, client: HtmlPlanningClient, directory: Path) -> None:
        self.client, self.directory = client, directory
        self.last_usage: dict[str, Any] | None = None
        self.executed_calls = 0
        self.reused_calls = 0

    def _call(self, method: str, kwargs: dict[str, Any]) -> str:
        from .page_runtime import atomic_write_json
        identity = {"schema": 1, "method": method, **kwargs}
        image_path = identity.pop("image_path", None)
        if image_path is not None:
            identity["image_sha256"] = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
        key = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        path = self.directory / f"{key}.json"
        previous = read_json(path) if path.is_file() else {}
        if previous.get("status") == "completed":
            self.last_usage = previous.get("usage")
            self.reused_calls += 1
            return str(previous["response"])
        if previous.get("status") in {"requesting", "outcome_unknown"}:
            raise RuntimeError("Previous HTML/VLM request outcome is unknown; reconcile its journal before retrying")
        record = {"status": "requesting", "request_sha256": key, "method": method}
        atomic_write_json(path, record)
        try:
            answer = getattr(self.client, method)(**kwargs)
        except Exception as exc:
            cause: BaseException | None = exc
            ambiguous = False
            while cause is not None:
                if isinstance(cause, (TimeoutError, ConnectionError)) or (
                    isinstance(cause, urllib.error.URLError) and not isinstance(cause, urllib.error.HTTPError)
                ):
                    ambiguous = True
                cause = cause.__cause__
            record.update(status="outcome_unknown" if ambiguous else "failed", error_type=type(exc).__name__)
            atomic_write_json(path, record)
            raise
        self.last_usage = self.client.last_usage
        self.executed_calls += 1
        record.update(status="completed", response=answer, usage=self.last_usage)
        atomic_write_json(path, record)
        return answer

    def chat_html(self, **kwargs: Any) -> str:
        return self._call("chat_html", kwargs)

    def vlm_html(self, **kwargs: Any) -> str:
        return self._call("vlm_html", kwargs)

    def vlm_json(self, **kwargs: Any) -> str:
        return self._call("vlm_json", kwargs)


class ApiMartHtmlPlanningClient:
    def __init__(self, *, config: PipelineConfig) -> None:
        self.client = ApiMartClient(
            api_root=config.api_root,
            api_key_env=config.api_key_env,
            env_path=config.env_path,
            timeout_seconds=config.timeout_seconds,
        )
        self.last_usage: dict[str, Any] | None = None

    def chat_html(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are a presentation information designer and expert "
                                "HTML/CSS author. Return only the requested complete HTML "
                                "document."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
            "max_tokens": max_tokens,
            "stream": False,
        }
        response = self.client._post_json("/responses", payload)
        self.last_usage = _extract_usage(response)
        text = _extract_text_response(response)
        if not text.strip():
            raise RuntimeError(f"APIMART HTML planning returned empty content: {response}")
        return text

    def vlm_html(
        self,
        *,
        model: str,
        prompt: str,
        image_path: Path,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are a presentation information designer and expert "
                                "HTML/CSS author. Return only the requested complete HTML "
                                "document."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_text",
                            "text": f"Visual reference for {image_path.stem}:",
                        },
                        {
                            "type": "input_image",
                            "image_url": image_to_data_uri(image_path),
                        },
                    ],
                },
            ],
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            response = self.client._post_json("/responses", payload)
        except RuntimeError as exc:
            message = str(exc)
            if (
                "convert_request_failed" not in message
                and "not implemented" not in message
            ):
                raise
            response = self.client._post_json(
                "/chat/completions",
                {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a presentation information designer and expert "
                                "HTML/CSS author. Return only the requested complete HTML "
                                "document."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "text",
                                    "text": f"Visual reference for {image_path.stem}:",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_to_data_uri(image_path)
                                    },
                                },
                            ],
                        },
                    ],
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
        self.last_usage = _extract_usage(response)
        text = _extract_text_response(response)
        if not text.strip():
            raise RuntimeError(f"APIMART HTML planning returned empty content: {response}")
        return text

    def vlm_json(
        self,
        *,
        model: str,
        prompt: str,
        image_path: Path,
        max_tokens: int,
    ) -> str:
        text = self.client.vlm_json(
            model=model,
            prompt=prompt,
            image_path=image_path,
            max_tokens=max_tokens,
        )
        self.last_usage = self.client.last_usage
        return text


def build_html_planning_prompt(
    *,
    slide_id: str,
    expression_logic: str,
    text_tree: dict[str, Any],
    frozen_content_by_key: dict[str, str],
    deck_font: str = "Noto Sans",
    deck_context: dict[str, Any] | None = None,
) -> str:
    authority = {
        "slide_id": slide_id,
        "text_tree": text_tree,
        "expression_logic": expression_logic,
        "source_content_by_key": frozen_content_by_key,
    }
    if deck_context:
        authority["deck_context"] = deck_context
    return f"""Co-plan editable text and generated imagery for one rich 16:9 slide. Treat the attached diffusion image as
a layout skeleton: adapt its visual masses, rhythm, hierarchy, and flow to the actual content without tracing pixels or
falling back to a generic template.

Content
- The source text is frozen display copy. Represent every data-content-key exactly once. Preserve its words, facts,
  signs, units, and punctuation. You may format the committed text into paragraphs or lists and add inline emphasis;
  do not rewrite, shorten, reorder, or add a lead/takeaway that changes the committed wording.
- Use at least 40px titles, 24px subtitles, and 20px body text on this 1536x864 canvas. These are readable floors,
  not per-slide size targets. Use one shared title size/weight (700) and subtitle size/weight (400) across the deck.
  Allocate text space before decorative borders; never fit copy by making it tiny. Use font-family:{json.dumps(deck_font)},sans-serif
  for all text so the preview and editable PPTX use the same installed font. Do not add facts or unregistered visible text.
- Keep multi-sentence body regions regular-weight. Do not bold an entire body block; fine-grained emphasis is handled by
  the dedicated text rework after geometry is final.
- Use compact but comfortable body leading, about 1.15-1.3, rather than large line spacing that consumes a small text area.

Layout
- Create 2 to 6 top-level, non-nested modules with unique data-module-id values. Keep all registered regions inside one
  module, prevent overlap, use about 48-72px outer margins and 20-36px gaps, and cover at least 62% of the canvas.
- Let the logic and diffusion skeleton decide the structure. When they show distinct stages, branches, or comparisons,
  prefer 2-3 substantial picture-text pairs instead of collapsing everything into one large image. Use one hero visual
  only when the content and skeleton genuinely form one scene. Keep pairs nearby and preserve a clear reading flow.
- Avoid scattered badges, tiny assets, repeated mini-cards, and decoration around every sentence.
- A large holder does not make small text readable: give its registered text a generous quiet inner rectangle and
  use the available space. Avoid very large empty frames whose copy is trapped in a small corner.

Holders
- On a slide with body content, use 1-3 meaningful text holders for key explanations or takeaways; do not frame every
  sentence. A holder is a top-level text-only module whose module element also carries
  data-holder-id="holder_<region-id>" and contains exactly one registered text region marked data-text-surface="holder".
- Draw the planning holder only as a simple wireframe: background:transparent; border:2px solid #777; box-shadow:none.
  No texture, fill, or decoration. Its exact module rectangle is final downstream geometry. Other modules stay transparent.

Visuals and output
- Use 1 to 3 substantial empty visual regions, each at least 280x180px and mapping 1:1 to one generated asset. Each has
  data-region-id, data-role="visual", a detailed text-free data-visual-brief, and its own unique
  data-generation-group="visual_group_<region-id>". Never share a generation group or make one image per bullet.
- Visual regions must stay empty and neutral: no artwork, icons, texture, SVG, canvas, CSS illustration, or asset links.
- Carry the supplied deck style, recognizable character roles, and slide purpose into every visual brief. Do not replace
  a human operator with a robot or replace a required action with unrelated activity. Keep role identities consistent.
- When the committed copy explicitly describes a sequence or feedback relation between visual regions, declare
  data-connect-to="next_region_id" on the source visual; multiple targets are separated by spaces. Use optional
  data-connector-color="#AABBCC" and data-connector-width="2". The exporter draws these arrows as native shapes.
  Declare only real mapped relations; do not invent a cycle or collapse four required stages into three unlabeled pictures.
- Return only one self-contained HTML document beginning with <!doctype html>, with no JavaScript or external resources.
  Create one 1536x864 section.slide with data-slide-id="{slide_id}" and overflow:hidden. Every text region has unique
  data-region-id, data-content-key, data-role="title|subtitle|body", and data-text-surface="plain|holder". Use
  letter-spacing:0, keep text fully visible, and leave at least 8px vertical slack in each text box.

Slide semantic authority:
{json.dumps(authority, ensure_ascii=False, separators=(",", ":"))}
"""


def extract_html_document(raw: str, warnings: list[str] | None = None) -> str:
    warnings = warnings if warnings is not None else []
    text = raw.strip()
    starts = [
        index
        for index in (text.lower().find("<!doctype"), text.lower().find("<html"))
        if index >= 0
    ]
    if not starts:
        warnings.append("response did not contain a complete HTML document; wrapped the returned markup")
        document = f"<!doctype html><html><head><meta charset=\"utf-8\"></head><body>{text}</body></html>\n"
    else:
        start = min(starts)
        end = text.lower().rfind("</html>")
        if end < start:
            warnings.append("response was missing </html>; appended a closing tag")
            document = text[start:].strip() + "\n</html>\n"
        else:
            document = text[start : end + len("</html>")].strip() + "\n"
    forbidden = {
        "script": r"<script(?:\s|>)",
        "iframe": r"<iframe(?:\s|>)",
        "embedded media": r"<(?:img|image|object|embed|link|video|audio|source)(?:\s|>)",
        "external URL": r"https?://",
        "data URI": r"data:[a-z]+/",
        # Inline SVG marker references such as url(#arrowBlue) are local and safe.
        "CSS URL": r"url\s*\(\s*(?!#)",
        "CSS import": r"@import\b",
    }
    for label, pattern in forbidden.items():
        if re.search(pattern, document, flags=re.IGNORECASE):
            warnings.append(f"response contains discouraged {label}; retained for tolerant local compilation")
    generated_text_pattern = r"(content\s*:\s*)[\"'][^\"']+[\"']"
    if re.search(generated_text_pattern, document, flags=re.IGNORECASE):
        warnings.append("response contains visible CSS generated text; removed it before compilation")
        document = re.sub(
            generated_text_pattern,
            r'\1""',
            document,
            flags=re.IGNORECASE,
        )
    return document


def build_html_critic_prompt(
    *,
    slide_id: str,
    text_tree: dict[str, Any],
    frozen_content_by_key: dict[str, str],
    deterministic_findings: list[str] | None = None,
    expression_logic: str = "",
    candidate_visual_groups: list[dict[str, Any]] | None = None,
) -> str:
    del candidate_visual_groups
    return f"""Review the rendered HTML plan for slide {slide_id}. Judge geometry and content mapping, not final artwork.
Return only this JSON shape:
{{"needs_rework":false,"issues":[],"summary":"brief assessment"}}

Request rework for semantic loss or unsupported copy; missing/duplicated keys; unregistered text; overflow, crop, overlap,
invalid modules or weak coverage; excessive decoration; or poor text-image pairing.
Also request rework when the logic and diffusion skeleton suggest several stages, branches, or comparisons but the plan
falls back to one generic oversized visual instead of meaningful picture-text pairs.

Require 1-3 substantial empty visuals with unique generation groups and 1:1 assets. Reject tiny or fragmented visuals,
one image per bullet, non-empty placeholders, and nearly all-image layouts. On slides with body content, require 1-3
important holder modules. Every holder must be a transparent simple solid-line frame, be a top-level text-only module,
carry data-holder-id, contain exactly one registered text region marked data-text-surface="holder", and fit its text.
This is the geography critic. Do not request wording, emphasis, font-size, weight, color, list-format, or decorative text
changes. A separate text rework handles presentation-friendly copy after geometry is stable.

Structured source meaning:
{json.dumps(text_tree, ensure_ascii=False, separators=(',', ':'))}

Source text by semantic key (committed wording must be preserved):
{json.dumps(frozen_content_by_key, ensure_ascii=False, separators=(',', ':'))}

Expression logic:
{expression_logic}

Measured findings from this exact HTML:
{json.dumps(deterministic_findings or [], ensure_ascii=False, separators=(',', ':'))}

Repeat every measured finding with its exact IDs and propose the smallest coherent fix.
"""


def build_html_rework_prompt(
    original_prompt: str, critic: dict[str, Any], current_html: str
) -> str:
    return f"""{original_prompt}

Critic issues:
{json.dumps(critic, ensure_ascii=False, indent=2)}

Revise CURRENT HTML to resolve every geography issue. Preserve all text region IDs, content keys, visible wording,
inline markup, typography, and existing holder IDs exactly. You may reposition or resize modules and may replace, split, or merge visual regions
when needed to follow the logic and diffusion skeleton, provided the final plan has 1-3 substantial 1:1 visual regions.
Keep 2-6 flat modules, compact coverage, clear picture-text pairs, and at least 20px between non-paired modules.

Do not change any visible text, list structure, font size, weight, style, or color. Do not add whole-block bold. A later
dedicated text rework owns leads, bullets, and inline emphasis.

Add or repair holder modules when needed. Each holder must use the exact parseable structure and simple wireframe from
the original contract; keep all non-holder modules transparent. Keep visual regions empty with unique generation groups.
Self-check every measured finding, then return only one complete HTML document.

CURRENT HTML:
{current_html}
"""


def _parse_critic_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("HTML critic response does not contain a JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("HTML critic response must be a JSON object")
    issues = payload.get("issues", [])
    if not isinstance(issues, list):
        issues = [issues]
    normalized_issues = []
    for issue in issues:
        if isinstance(issue, dict):
            parts = []
            if issue.get("region"):
                parts.append(str(issue["region"]))
            if issue.get("issue"):
                parts.append(str(issue["issue"]))
            if issue.get("fix"):
                parts.append(f"fix: {issue['fix']}")
            normalized_issues.append("; ".join(parts) or json.dumps(issue, ensure_ascii=False))
        else:
            normalized_issues.append(str(issue))
    return {
        "needs_rework": bool(payload.get("needs_rework", False)),
        "issues": normalized_issues,
        "summary": str(payload.get("summary", "")),
    }


def _inspect_visual_group_candidates(
    html_path: Path, expression_logic: str
) -> list[dict[str, Any]]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": CANVAS[0], "height": CANVAS[1]})
        page.goto(html_path.resolve().as_uri(), wait_until="load")
        page.evaluate("document.fonts.ready")
        visuals = page.evaluate(
            """() => [...document.querySelectorAll('[data-role="visual"]')].map(node => ({
              region_id: node.dataset.regionId || '',
              generation_group: node.dataset.generationGroup || '',
              visual_brief: node.dataset.visualBrief || ''
            }))"""
        )
        browser.close()
    candidates = []
    for item in visuals:
        region_id = str(item.get("region_id") or "").strip()
        if not region_id:
            continue
        brief = str(item.get("visual_brief") or "").strip()
        candidates.append(
            {
                "group_id": f"visual_group_{region_id}",
                "region_ids": [region_id],
                "relation_type": "single_visual",
                "semantic_description": brief
                or (
                    "Create one integrated detailed text-free explanatory visual "
                    f"for this part of the slide: {expression_logic}"
                ),
                "use_structure_reference": False,
            }
        )
    return candidates


def _render_critic_preview(html_path: Path, output_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": CANVAS[0], "height": CANVAS[1]},
            device_scale_factor=1,
        )
        context.route("http://**/*", lambda route: route.abort())
        context.route("https://**/*", lambda route: route.abort())
        page = context.new_page()
        page.goto(html_path.resolve().as_uri(), wait_until="load")
        page.evaluate("document.fonts.ready")
        for attempt in range(2):
            try:
                page.screenshot(path=str(output_path))
                break
            except Exception:
                if attempt:
                    raise
                page.wait_for_timeout(100)
        context.close()
        browser.close()


def _rework_preserves_structure(
    *, original_html: str, reworked_html: str
) -> tuple[bool, str]:
    original = _inspect_html_structure(original_html)
    reworked = _inspect_html_structure(reworked_html)
    if original["text_region_ids"] != reworked["text_region_ids"]:
        return False, "rework changed the registered text region inventory"
    if original["content_keys"] != reworked["content_keys"]:
        return False, "rework changed the semantic content-key inventory"
    if original["content_by_key"] != reworked["content_by_key"]:
        return False, "geography rework changed visible text"
    if original["typography_by_key"] != reworked["typography_by_key"]:
        return False, "geography rework changed text typography"
    if original["inline_typography_by_key"] != reworked["inline_typography_by_key"]:
        return False, "geography rework changed inline text typography or list structure"
    if reworked["empty_content_keys"]:
        return False, f"rework left empty display text for: {reworked['empty_content_keys']}"
    if not 2 <= len(reworked["module_ids"]) <= 6:
        return False, f"rework has {len(reworked['module_ids'])} modules; expected 2-6"
    if len(reworked["module_ids"]) != len(set(reworked["module_ids"])):
        return False, "rework duplicated data-module-id values"
    if reworked["nested_module_ids"]:
        return False, f"rework nested modules: {reworked['nested_module_ids']}"
    if reworked["unmoduled_region_ids"]:
        return False, f"rework left regions outside modules: {reworked['unmoduled_region_ids']}"
    if not 1 <= len(reworked["visual_region_ids"]) <= 3:
        return False, (
            f"rework has {len(reworked['visual_region_ids'])} visual regions; expected 1-3"
        )
    reworked_degenerate = {
        region_id
        for region_id, bbox in reworked["visual_bboxes"].items()
        if bbox[2] < MIN_MEANINGFUL_VISUAL_WIDTH_PX
        or bbox[3] < MIN_MEANINGFUL_VISUAL_HEIGHT_PX
    }
    if reworked_degenerate:
        return False, (
            "rework left degenerate visual regions: "
            + ", ".join(sorted(reworked_degenerate))
        )
    missing_holders = sorted(set(original["holder_ids"]) - set(reworked["holder_ids"]))
    if missing_holders:
        return False, f"rework removed holder surfaces: {missing_holders}"
    return True, "preserved"


def apply_initial_readability_contract(html_path: Path) -> list[dict[str, Any]]:
    """Set readable role floors before layout commitment, never during export.

    The existing geography critic can then allocate room for the unchanged
    copy. This helper intentionally does not shrink text or move any module.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": CANVAS[0], "height": CANVAS[1]})
        page.set_content(html_path.read_text(encoding="utf-8"), wait_until="load")
        page.evaluate("document.fonts.ready")
        changes = page.evaluate(
            """floors => [...document.querySelectorAll('[data-content-key]')].flatMap(node => {
              const role = node.dataset.role || 'body';
              const floor = floors[role] || floors.body;
              const style = getComputedStyle(node);
              const rect = node.getBoundingClientRect();
              const scale = node.offsetHeight > 0 ? rect.height / node.offsetHeight : 1;
              const current = (parseFloat(style.fontSize) || 16) * scale;
              const changed = [];
              if (current < floor - 0.1) {
                node.style.setProperty('font-size', `${floor / Math.max(scale, 0.001)}px`, 'important');
                changed.push({content_key:node.dataset.contentKey,role,from_font_size_px:current,to_font_size_px:floor});
              }
              let inlineRaised = false;
              for (const child of node.querySelectorAll('*')) {
                if ((parseFloat(getComputedStyle(child).fontSize) || 16) * scale < floor - 0.1) {
                  child.style.setProperty('font-size', `${floor / Math.max(scale, 0.001)}px`, 'important');
                  inlineRaised = true;
                }
              }
              if (inlineRaised) changed.push({content_key:node.dataset.contentKey,role,inline_font_floor_px:floor});
              const currentStyle = getComputedStyle(node);
              const ratio = parseFloat(currentStyle.lineHeight) / parseFloat(currentStyle.fontSize);
              if (role === 'body' && Number.isFinite(ratio) && ratio > 1.3) {
                node.style.setProperty('line-height', '1.3', 'important');
                changed.push({content_key:node.dataset.contentKey,role,line_height_ratio:1.3});
              }
              const weight = role === 'title' ? 700 : role === 'subtitle' ? 400 : null;
              if (weight !== null && Number(style.fontWeight) !== weight) {
                node.style.setProperty('font-weight', String(weight), 'important');
                changed.push({content_key:node.dataset.contentKey,role,font_weight:weight});
              }
              return changed;
            })""",
            READABLE_ROLE_MIN_PX,
        )
        if changes:
            document = "<!doctype html>\n" + page.locator("html").evaluate("node => node.outerHTML") + "\n"
            write_text(html_path, document)
        browser.close()
    return changes


def allocate_holder_text_space(html_path: Path) -> list[dict[str, Any]]:
    """Give a holder's sole text block its existing interior before commitment.

    Keeps the holder/module and artwork geometry fixed; removes stale CSS caps
    that otherwise trap readable text in a small corner of a large frame.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": CANVAS[0], "height": CANVAS[1]})
        page.set_content(html_path.read_text(encoding="utf-8"), wait_until="load")
        page.evaluate("document.fonts.ready")
        changes = page.evaluate("""() => [...document.querySelectorAll('[data-holder-id]')].flatMap(holder => {
          const nodes = [...holder.querySelectorAll('[data-content-key]')];
          if (nodes.length !== 1 || holder.querySelector('[data-role="visual"]')) return [];
          const node = nodes[0], rect = holder.getBoundingClientRect();
          if (holder.clientWidth < 100 || holder.clientHeight < 32) return [];
          const sx = holder.offsetWidth ? rect.width/holder.offsetWidth : 1;
          const sy = holder.offsetHeight ? rect.height/holder.offsetHeight : 1;
          const before = node.getBoundingClientRect();
          const textStyle=getComputedStyle(node);
          const font=(parseFloat(textStyle.fontSize)||16)*sy;
          const insets=['paddingTop','paddingBottom','borderTopWidth','borderBottomWidth'].reduce((s,k)=>s+(parseFloat(textStyle[k])||0),0)*sy;
          const tops=[];const walker=document.createTreeWalker(node,NodeFilter.SHOW_TEXT);
          while(walker.nextNode()){if(!walker.currentNode.textContent.trim())continue;const r=document.createRange();r.selectNodeContents(walker.currentNode);for(const box of r.getClientRects())tops.push(box.top);}
          let rows=0,last=-Infinity;for(const top of tops.sort((a,b)=>a-b)){if(top-last>font*.5){rows++;last=top;}}
          const overflow=node.scrollHeight>node.clientHeight+1||node.scrollWidth>node.clientWidth+1;
          if(!overflow && before.height-insets>=font*(1.5*Math.max(1,rows)+0.4)-1) return [];
          const px = 16/Math.max(sx,0.001), py = (rect.height < 100 ? 2 : 16)/Math.max(sy,0.001);
          for (const [key,value] of Object.entries({position:'absolute',left:`${px}px`,top:`${py}px`,
            right:'auto',bottom:'auto',width:`${holder.clientWidth-2*px}px`,height:`${holder.clientHeight-2*py}px`,
            margin:'0',padding:'0','box-sizing':'border-box','max-width':'none','min-width':'0',
            'max-height':'none','min-height':'0',transform:'none',display:'block'})) node.style.setProperty(key,value,'important');
          const after=node.getBoundingClientRect();
          const a=[before.x,before.y,before.width,before.height], b=[after.x,after.y,after.width,after.height];
          return a.some((v,i)=>Math.abs(v-b[i])>0.1) ? [{content_key:node.dataset.contentKey,before:a,after:b}] : [];
        })""")
        if changes:
            document = "<!doctype html>\n" + page.locator("html").evaluate("node => node.outerHTML") + "\n"
            write_text(html_path, document)
        browser.close()
    return changes


def allocate_readable_text_space(html_path: Path) -> list[dict[str, Any]]:
    """Reserve physical role height and use free plain-text module space.

    A single local pass before commitment; no shrinking, wording changes,
    module relocation, image movement, or inferred semantic relationships.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": CANVAS[0], "height": CANVAS[1]})
        page.set_content(html_path.read_text(encoding="utf-8"), wait_until="load")
        page.evaluate("document.fonts.ready")
        changes = page.evaluate("""() => {
          const changes=[];
          const overlapX=(a,b)=>Math.min(a.right,b.right)-Math.max(a.left,b.left)>1;
          for (const module of document.querySelectorAll('[data-module-id]')) {
            if (module.hasAttribute('data-holder-id')) continue;
            const nodes=[...module.querySelectorAll('[data-content-key]')].filter(n=>n.closest('[data-module-id]')===module);
            if (!nodes.length) continue;
            let mr=module.getBoundingClientRect();const ms=getComputedStyle(module);
            const used=(axis)=>{const sides=axis==='x'?['Left','Right']:['Top','Bottom'];
              return parseFloat(axis==='x'?ms.width:ms.height)+(ms.boxSizing==='border-box'?0:
                sides.reduce((v,k)=>v+(parseFloat(ms[`padding${k}`])||0)+(parseFloat(ms[`border${k}Width`])||0),0));};
            const sx=used('x')>0?mr.width/used('x'):1, sy=used('y')>0?mr.height/used('y'):1;
            const borderX=parseFloat(ms.borderLeftWidth)||0, borderY=parseFloat(ms.borderTopWidth)||0;
            module.style.setProperty('width',ms.width,'important');module.style.setProperty('height',ms.height,'important');
            // Fixed header modules may have a free page gap immediately below
            // them. Reserve the shared role heights there before packing the
            // subtitle; keep all artwork and other modules at their positions.
            const headers=nodes.filter(n=>['title','subtitle'].includes(n.dataset.role))
              .sort((a,b)=>(a.dataset.role==='title'?-1:0)-(b.dataset.role==='title'?-1:0));
            if (headers.length) {
              let needed=mr.top;
              for(const node of headers){const r=node.getBoundingClientRect(),s=getComputedStyle(node);
                const top=node.dataset.role==='subtitle'?Math.max(r.top,Math.ceil(needed+2)):r.top;
                needed=top+Math.ceil(Math.max(r.height,1.5*(parseFloat(s.fontSize)||16)*sy));}
              let limit=864;
              for(const other of document.querySelectorAll('[data-module-id],[data-region-id]')){
                if(other===module||module.contains(other)||other.contains(module))continue;
                const r=other.getBoundingClientRect();if(r.top>=mr.bottom-1&&overlapX(mr,r))limit=Math.min(limit,r.top-2);
              }
              const bottom=Math.min(needed,limit);
              if(bottom>mr.bottom+.1){const before=[mr.x,mr.y,mr.width,mr.height];
                module.style.setProperty('height',`${(bottom-mr.top)/sy}px`,'important');mr=module.getBoundingClientRect();
                changes.push({module_id:module.dataset.moduleId,role:'header',before,after:[mr.x,mr.y,mr.width,mr.height]});}
            }
            const regions=[...module.querySelectorAll('[data-region-id]')];
            let headerBottom=-Infinity;
            for (const node of nodes.sort((a,b)=>(a.dataset.role==='title'?-1:0)-(b.dataset.role==='title'?-1:0))) {
              const before=node.getBoundingClientRect(), style=getComputedStyle(node), role=node.dataset.role;
              const size=(parseFloat(style.fontSize)||16)*sy;
              let top=before.top, bottom=mr.bottom;
              const header=role==='title'||role==='subtitle';
              if (!header) {
                const tops=[];const walker=document.createTreeWalker(node,NodeFilter.SHOW_TEXT);
                while(walker.nextNode()){if(!walker.currentNode.textContent.trim())continue;const r=document.createRange();r.selectNodeContents(walker.currentNode);for(const box of r.getClientRects())tops.push(box.top);}
                let rows=0,last=-Infinity;for(const top of tops.sort((a,b)=>a-b)){if(top-last>size*.5){rows++;last=top;}}
                const insets=['paddingTop','paddingBottom','borderTopWidth','borderBottomWidth'].reduce((s,k)=>s+(parseFloat(style[k])||0),0)*sy;
                if(node.scrollHeight<=node.clientHeight+1&&node.scrollWidth<=node.clientWidth+1&&before.height-insets>=size*(1.5*Math.max(1,rows)+0.4)-1) continue;
              }
              if (role==='subtitle') top=Math.max(top,Math.ceil(headerBottom+2));
              for (const other of regions) {
                if (other===node || node.contains(other) || other.contains(node)) continue;
                if (header && ['title','subtitle'].includes(other.dataset.role)) continue;
                const r=other.getBoundingClientRect();
                if (r.top>=before.bottom-1 && overlapX(before,r)) bottom=Math.min(bottom,r.top-4);
              }
              const available=bottom-top;
              const desired=header?Math.ceil(Math.max(before.height,1.5*size)):Math.max(before.height,available);
              const height=Math.min(desired,available);
              if (height<=0 || height<before.height-1) {
                changes.push({content_key:node.dataset.contentKey,role,status:'unresolved',reason:'no_free_module_text_space'});continue;
              }
              for (const [key,value] of Object.entries({position:'absolute',left:`${(before.left-mr.left)/sx-borderX}px`,
                top:`${(top-mr.top)/sy-borderY}px`,width:`${before.width/sx}px`,height:`${height/sy}px`,
                margin:'0','box-sizing':'border-box','max-height':'none','min-height':'0',transform:'none'})) node.style.setProperty(key,value,'important');
              const after=node.getBoundingClientRect();
              if (header) headerBottom=Math.max(headerBottom,after.bottom);
              const a=[before.x,before.y,before.width,before.height], b=[after.x,after.y,after.width,after.height];
              if (a.some((v,i)=>Math.abs(v-b[i])>0.1)) changes.push({content_key:node.dataset.contentKey,role,before:a,after:b});
            }
          }
          return changes;
        }""")
        if changes:
            document = "<!doctype html>\n" + page.locator("html").evaluate("node => node.outerHTML") + "\n"
            write_text(html_path, document)
        browser.close()
    return changes


def prepare_html_layout(
    html_path: Path,
    *,
    deck_font: str = "Noto Sans",
    source_text: dict[str, Any] | None = None,
    slide_id: str | None = None,
    page: Any | None = None,
) -> list[str]:
    """Repair only invalid layout units before HTML geometry is committed.

    Existing in-bounds modules are untouched. A displaced module moves with its
    text, holder and visual children; a missing or altered text block is restored
    from its own committed source. Quality defects are recorded, never raised.
    """
    from playwright.sync_api import sync_playwright

    document = html_path.read_text(encoding="utf-8")
    font_style = (
        '<style id="rl4ppt-deck-font">'
        '[data-content-key],[data-content-key] *{font-family:'
        + json.dumps(deck_font)
        + ',sans-serif!important}</style>'
    )
    document = re.sub(
        r'<style\b[^>]*\bid=["\']rl4ppt-deck-font["\'][^>]*>.*?</style>',
        "", document, flags=re.DOTALL | re.IGNORECASE,
    )
    if re.search(r'</head\s*>', document, flags=re.IGNORECASE):
        document = re.sub(r'</head\s*>', lambda _: font_style + '</head>', document, count=1, flags=re.IGNORECASE)
    else:
        document = font_style + document
    expected = [
        {**item, "tokens": list(_committed_copy_text(item["text"]))}
        for item in _content_items(source_text)
    ] if source_text is not None else []
    def prepare_page(active_page: Any) -> tuple[list[str], str]:
        active_page.set_content(document, wait_until="load")
        active_page.evaluate("document.fonts.ready")
        repairs = active_page.evaluate(_LOCAL_LAYOUT_REPAIR_SCRIPT, {
            "width": CANVAS[0], "height": CANVAS[1],
            "expected": expected, "slideId": slide_id,
        })
        return repairs, active_page.content() if repairs else document

    if page is not None:
        repairs, document = prepare_page(page)
    else:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                local_page = browser.new_page(viewport={"width": CANVAS[0], "height": CANVAS[1]})
                local_page.route("http://**/*", lambda route: route.abort())
                local_page.route("https://**/*", lambda route: route.abort())
                repairs, document = prepare_page(local_page)
            finally:
                browser.close()
    if document != html_path.read_text(encoding="utf-8"):
        write_text(html_path, document)
    return repairs


_LOCAL_LAYOUT_REPAIR_SCRIPT = r"""({width, height, expected, slideId}) => {
  const repairs = [];
  let slide = document.querySelector('section.slide');
  if (!slide) {
    slide = document.createElement('section');
    slide.className = 'slide';
    [...document.body.childNodes].forEach(node => slide.appendChild(node));
    document.body.appendChild(slide);
    slide.style.cssText = `position:relative;width:${width}px;height:${height}px;overflow:hidden`;
    repairs.push('restored missing section.slide around existing layout');
  }
  if (slideId) slide.dataset.slideId = slideId;
  const root = slide.getBoundingClientRect();
  if (Math.abs(root.width-width) > 1 || Math.abs(root.height-height) > 1) {
    slide.style.setProperty('width', `${width}px`, 'important');
    slide.style.setProperty('height', `${height}px`, 'important');
    repairs.push('restored slide dimensions to the canonical canvas');
  }
  const rectOf = node => {
    const r=node.getBoundingClientRect(), s=slide.getBoundingClientRect();
    return [r.left-s.left,r.top-s.top,r.width,r.height];
  };
  const overlap = (a,b) => Math.max(0,Math.min(a[0]+a[2],b[0]+b[2])-Math.max(a[0],b[0])) *
    Math.max(0,Math.min(a[1]+a[3],b[1]+b[3])-Math.max(a[1],b[1]));
  const pickPosition = (w,h,x,y,others) => {
    const clampX = v => Math.max(0,Math.min(width-w,v));
    const clampY = v => Math.max(0,Math.min(height-h,v));
    const xs=[clampX(x),0,width-w], ys=[clampY(y),0,height-h];
    others.forEach(b=>{xs.push(clampX(b[0]-w-12),clampX(b[0]+b[2]+12));ys.push(clampY(b[1]-h-12),clampY(b[1]+b[3]+12));});
    let best=null;
    for(const px of xs) for(const py of ys) {
      const candidate=[px,py,w,h];
      const collision=others.reduce((sum,b)=>sum+overlap(candidate,b),0);
      const displacement=Math.hypot(px-x,py-y);
      if(!best || collision < best.collision-0.1 || (Math.abs(collision-best.collision)<0.1 && displacement<best.displacement)) {
        best={bbox:candidate,collision,displacement};
      }
    }
    return best.bbox;
  };
  const tokens = text => (String(text).replace(/^\s*(?:[•●▪◦‣⁃*+-]\s+|\d+[.)、]\s+)/gm,'').match(/[A-Za-z0-9_]+|[^\s]/gu) || []);
  for(const item of expected) {
    let node=[...slide.querySelectorAll('[data-content-key]')].find(n=>n.dataset.contentKey===item.content_key);
    if(!node) {
      node=document.createElement('div');
      node.dataset.regionId=`recovered_${item.content_key}`;
      node.dataset.contentKey=item.content_key;
      node.dataset.role=item.role;
      node.dataset.moduleId=`recovered_module_${item.content_key}`;
      node.dataset.textSurface='plain';
      const occupied=[...slide.querySelectorAll('[data-region-id]')].map(rectOf);
      const bbox=pickPosition(640,120,48,48,occupied);
      node.style.cssText=`position:absolute;left:${bbox[0]}px;top:${bbox[1]}px;width:640px;height:120px;font-size:${item.role==='title'?40:28}px;white-space:pre-line`;
      node.textContent=item.text;
      slide.appendChild(node);
      repairs.push(`restored missing committed text ${item.content_key}`);
    } else if(JSON.stringify(tokens(node.innerText ?? node.textContent ?? '')) !== JSON.stringify(item.tokens)) {
      node.textContent=item.text;
      node.style.whiteSpace='pre-line';
      delete node.dataset.textReworked;
      repairs.push(`restored committed wording for ${item.content_key}`);
    }
    if(!node.dataset.regionId) {
      node.dataset.regionId=`recovered_${item.content_key}`;
      repairs.push(`registered missing region ID for ${item.content_key}`);
    }
    node.dataset.role=item.role;
  }
  const units=[...slide.querySelectorAll('[data-module-id]')].filter(n=>!n.parentElement?.closest('[data-module-id]'));
  units.push(...[...slide.querySelectorAll('[data-region-id]')].filter(n=>!n.closest('[data-module-id]')));
  const unionOf = node => {
    const rects=[node,...node.querySelectorAll('[data-region-id],[data-holder-id]')].map(rectOf);
    const x=Math.min(...rects.map(b=>b[0])), y=Math.min(...rects.map(b=>b[1]));
    return [x,y,Math.max(...rects.map(b=>b[0]+b[2]))-x,Math.max(...rects.map(b=>b[1]+b[3]))-y];
  };
  for(const unit of units) {
    const box=unionOf(unit);
    if(box.every(Number.isFinite) && box[2]>0 && box[3]>0 && box[0]>=-0.5 && box[1]>=-0.5 && box[0]+box[2]<=width+0.5 && box[1]+box[3]<=height+0.5) continue;
    if(!box.every(Number.isFinite) || box[2]<=0 || box[3]<=0) {
      unit.style.setProperty('min-width','24px','important');
      unit.style.setProperty('min-height','24px','important');
      repairs.push(`restored positive dimensions for ${unit.dataset.moduleId || unit.dataset.regionId}`);
      continue;
    }
    const originalNodes=[unit,...unit.querySelectorAll('[data-region-id],[data-holder-id]')];
    originalNodes.forEach(n=>{if(!n.dataset.qualityOriginalBbox)n.dataset.qualityOriginalBbox=JSON.stringify(rectOf(n));});
    const scale=Math.min(1,width/box[2],height/box[3]);
    const others=units.filter(n=>n!==unit).map(unionOf).filter(b=>b[0]>=0 && b[1]>=0 && b[0]+b[2]<=width && b[1]+b[3]<=height);
    const target=pickPosition(box[2]*scale,box[3]*scale,box[0],box[1],others);
    const origin=rectOf(unit);
    const dx=target[0]-(origin[0]+(box[0]-origin[0])*scale);
    const dy=target[1]-(origin[1]+(box[1]-origin[1])*scale);
    const existing=getComputedStyle(unit).transform;
    unit.style.setProperty('transform-origin','0 0','important');
    unit.style.setProperty('transform',`translate(${dx}px,${dy}px) scale(${scale}) ${existing==='none'?'':existing}`,'important');
    // A pre-existing scale/rotation may have used a non-zero transform origin.
    // Restore the measured target after composing that matrix at the new origin.
    const realized=unionOf(unit);
    const correctionX=target[0]-realized[0], correctionY=target[1]-realized[1];
    if(Math.abs(correctionX)>0.01 || Math.abs(correctionY)>0.01) {
      const composed=unit.style.transform;
      unit.style.setProperty('transform',`translate(${correctionX}px,${correctionY}px) ${composed}`,'important');
    }
    const name=unit.dataset.moduleId || unit.dataset.regionId;
    unit.dataset.qualityGeometryRepair='true';
    repairs.push(`recovered out-of-canvas unit ${name} without separating its content`);
  }
  return repairs;
}"""


def _find_layout_contract_issues(html_path: Path) -> list[str]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": CANVAS[0], "height": CANVAS[1]})
        page.goto(html_path.resolve().as_uri(), wait_until="load")
        page.evaluate("document.fonts.ready")
        payload = page.evaluate(
            """() => {
              const slideRect = document.querySelector('section.slide')?.getBoundingClientRect() || {x:0,y:0,width:1536,height:864};
              const box = node => {
                const rect = node.getBoundingClientRect();
                return [rect.x-slideRect.x, rect.y-slideRect.y, rect.width, rect.height];
              };
              const alpha = value => {
                const text = String(value || '').trim().toLowerCase();
                if (!text || text === 'transparent') return 0;
                if (text.startsWith('rgba(')) return Number(text.slice(5, -1).split(/[,/]/).at(-1).trim());
                if (text.startsWith('rgb(') && text.includes('/')) return Number(text.slice(4, -1).split('/').at(-1).trim());
                return 1;
              };
              const styledSurface = node => {
                const style = getComputedStyle(node);
                const border = [style.borderTopWidth, style.borderRightWidth,
                  style.borderBottomWidth, style.borderLeftWidth].some(value => parseFloat(value) > 0);
                const rounded = [style.borderTopLeftRadius, style.borderTopRightRadius,
                  style.borderBottomRightRadius, style.borderBottomLeftRadius].some(value => parseFloat(value) > 0);
                return alpha(style.backgroundColor) > 0.05 || border || rounded || style.boxShadow !== 'none';
              };
              const holderRecord = node => {
                const style = getComputedStyle(node);
                const textNodes = [
                  ...(node.matches('[data-content-key]') ? [node] : []),
                  ...node.querySelectorAll('[data-content-key]')
                ];
                const borderWidths = [style.borderTopWidth, style.borderRightWidth,
                  style.borderBottomWidth, style.borderLeftWidth].map(value => parseFloat(value) || 0);
                const borderStyles = [style.borderTopStyle, style.borderRightStyle,
                  style.borderBottomStyle, style.borderLeftStyle];
                return {
                  id: node.dataset.holderId || '',
                  bbox: box(node),
                  is_module: node.hasAttribute('data-module-id'),
                  text_region_ids: textNodes.map(text => text.dataset.regionId || '').filter(Boolean),
                  text_surfaces: textNodes.map(text => text.dataset.textSurface || ''),
                  visual_count: node.querySelectorAll('[data-role="visual"]').length +
                    (node.matches('[data-role="visual"]') ? 1 : 0),
                  simple_outline: alpha(style.backgroundColor) <= 0.05 &&
                    borderWidths.every(width => width >= 1 && width <= 3) &&
                    borderStyles.every(kind => kind === 'solid') && style.boxShadow === 'none'
                };
              };
              const modules = [...document.querySelectorAll('[data-module-id]')];
              return {
                slide_size: [slideRect.width, slideRect.height],
                regions: [...document.querySelectorAll('[data-region-id]')].map(node => ({
                  id: node.dataset.regionId || '', role: node.dataset.role || '',
                  group: node.dataset.equivalentGroup || '',
                  generation_group: node.dataset.generationGroup || '',
                  module_id: node.closest('[data-module-id]')?.dataset.moduleId || '',
                  overflow: node.scrollWidth > node.clientWidth + 1 || node.scrollHeight > node.clientHeight + 1,
                  child_count: node.children.length,
                  text: (node.innerText ?? node.textContent ?? '').trim(),
                  font_size_px: (parseFloat(getComputedStyle(node).fontSize) || 16) *
                    (node.offsetHeight > 0 ? node.getBoundingClientRect().height / node.offsetHeight : 1),
                  bbox: box(node)
                })),
                modules: modules.map(node => ({
                  id: node.dataset.moduleId || '',
                  parent_module_id: node.parentElement?.closest('[data-module-id]')?.dataset.moduleId || '',
                  text_count: node.querySelectorAll('[data-content-key]').length +
                    (node.matches('[data-content-key]') ? 1 : 0),
                  has_visual: Boolean(node.querySelector('[data-role="visual"]')) || node.matches('[data-role="visual"]'),
                  holder_id: node.dataset.holderId || '',
                  styled_surface: styledSurface(node),
                  bbox: box(node)
                })),
                holders: [...document.querySelectorAll('[data-holder-id]')].map(holderRecord),
                extra_text: [...document.querySelectorAll('*')]
                  .filter(node => node.children.length === 0 && (node.innerText ?? node.textContent ?? '').trim())
                  .filter(node => Boolean(node.closest('section.slide')))
                  .filter(node => !node.closest('[data-content-key]'))
                  .map(node => (node.innerText ?? node.textContent ?? '').trim()),
              };
            }"""
        )
        browser.close()
    regions = payload["regions"]
    modules = payload["modules"]
    holders = payload["holders"]
    findings = []
    if any(abs(value - expected) > 1 for value, expected in zip(payload["slide_size"], CANVAS)):
        findings.append(f"slide dimensions differ from the canonical canvas: {payload['slide_size']}")
    canvas_bbox = [0.0, 0.0, float(CANVAS[0]), float(CANVAS[1])]
    for kind, items in (("module", modules), ("region", regions), ("holder", holders)):
        for item in items:
            box = item["bbox"]
            if box[2] <= 0 or box[3] <= 0 or not _bbox_contains(canvas_bbox, box):
                findings.append(f"{kind} {item['id']} extends outside the canvas or has non-positive dimensions")
    if payload["extra_text"]:
        findings.append(
            "unregistered visible text is not carried into assembly: "
            + " | ".join(payload["extra_text"])
        )
    if not 2 <= len(modules) <= 6:
        findings.append(f"module count is {len(modules)}; expected 2-6 flat modules")
    nested = sorted(item["id"] for item in modules if item["parent_module_id"])
    if nested:
        findings.append("nested modules are not allowed: " + ", ".join(nested))
    missing_holders = sorted(
        item["id"]
        for item in modules
        if item["text_count"] == 1
        and not item["has_visual"]
        and item["styled_surface"]
        and not item["holder_id"]
    )
    if missing_holders:
        findings.append(
            "styled single-text modules require explicit data-holder-id: "
            + ", ".join(missing_holders)
        )
    body_regions = [region for region in regions if region["role"] == "body"]
    if body_regions and not holders:
        findings.append("content slides require at least one explicit holder module")
    if len(holders) > 3:
        findings.append(f"holder count is {len(holders)}; expected at most 3 meaningful holders")
    holder_ids = [str(item["id"]).strip() for item in holders]
    if any(not value for value in holder_ids) or len(holder_ids) != len(set(holder_ids)):
        findings.append("data-holder-id values must be non-empty and unique")
    invalid_holder_structure = sorted(
        item["id"] or "<empty>"
        for item in holders
        if not item["is_module"]
        or len(item["text_region_ids"]) != 1
        or item["visual_count"]
        or item["text_surfaces"] != ["holder"]
    )
    if invalid_holder_structure:
        findings.append(
            "holders must be top-level text-only modules with exactly one registered holder text: "
            + ", ".join(invalid_holder_structure)
        )
    non_wireframe_holders = sorted(
        item["id"] or "<empty>" for item in holders if not item["simple_outline"]
    )
    if non_wireframe_holders:
        findings.append(
            "holders must use a transparent simple 1-3px solid outline with no shadow: "
            + ", ".join(non_wireframe_holders)
        )
    module_ids = [str(item["id"]).strip() for item in modules]
    if any(not value for value in module_ids) or len(module_ids) != len(set(module_ids)):
        findings.append("data-module-id values must be non-empty and unique")
    missing_modules = sorted(region["id"] for region in regions if not region["module_id"])
    if missing_modules:
        findings.append("registered regions outside modules: " + ", ".join(missing_modules))
    module_by_id = {item["id"]: item for item in modules}
    outside_bounds = sorted(
        region["id"]
        for region in regions
        if region["module_id"] in module_by_id
        and not _bbox_contains(module_by_id[region["module_id"]]["bbox"], region["bbox"])
    )
    if outside_bounds:
        findings.append(
            "registered regions extend outside their modules: " + ", ".join(outside_bounds)
        )
    for index, left in enumerate(modules):
        for right in modules[index + 1 :]:
            if _bbox_intersection_area(left["bbox"], right["bbox"]) > 1:
                findings.append(f"modules {left['id']} and {right['id']} overlap")
    module_coverage = sum(item["bbox"][2] * item["bbox"][3] for item in modules) / (
        CANVAS[0] * CANVAS[1]
    )
    if module_coverage < MIN_MODULE_COVERAGE:
        findings.append(
            f"module coverage is {module_coverage:.3f}; expected at least {MIN_MODULE_COVERAGE:.2f}"
        )
    visuals = [region for region in regions if region["role"] == "visual"]
    if not 1 <= len(visuals) <= 3:
        findings.append(f"visual region count is {len(visuals)}; expected 1-3 substantial visuals")
    generation_groups = [str(region["generation_group"]).strip() for region in visuals]
    if any(not value for value in generation_groups) or len(generation_groups) != len(
        set(generation_groups)
    ):
        findings.append("every visual must have its own non-empty unique data-generation-group")
    for region in regions:
        if region["role"] != "visual" and region["text"]:
            floor = READABLE_ROLE_MIN_PX.get(region["role"], READABLE_ROLE_MIN_PX["body"])
            if region["font_size_px"] < floor - 0.25:
                findings.append(f"{region['id']} text is below the readable {floor:g}px {region['role']} floor")
        if (
            region["role"] == "visual"
            and (
                region["bbox"][2] < MIN_MEANINGFUL_VISUAL_WIDTH_PX
                or region["bbox"][3] < MIN_MEANINGFUL_VISUAL_HEIGHT_PX
            )
        ):
            findings.append(
                f"{region['id']} is a degenerate visual region at "
                f"{round(region['bbox'][2])}x{round(region['bbox'][3])}px; "
                f"minimum is {MIN_MEANINGFUL_VISUAL_WIDTH_PX}x"
                f"{MIN_MEANINGFUL_VISUAL_HEIGHT_PX}px"
            )
        if region["role"] == "visual" and (region["text"] or region["child_count"]):
            findings.append(f"{region['id']} visual region is not empty")
        if region["role"] != "visual" and region["overflow"]:
            findings.append(f"{region['id']} text overflows its planned box")
    for index, left in enumerate(regions):
        for right in regions[index + 1 :]:
            if _bbox_intersection_area(left["bbox"], right["bbox"]) > 1:
                findings.append(f"{left['id']} overlaps {right['id']}")
    groups: dict[str, list[dict[str, Any]]] = {}
    for region in regions:
        if region["group"]:
            groups.setdefault(region["group"], []).append(region)
    for group_id, members in groups.items():
        if len(members) < 2:
            continue
        widths = [round(item["bbox"][2]) for item in members]
        heights = [round(item["bbox"][3]) for item in members]
        if max(widths) - min(widths) > 2 or max(heights) - min(heights) > 2:
            findings.append(
                f"equivalent group {group_id} has inconsistent sizes: "
                + ", ".join(
                    f"{item['id']}={round(item['bbox'][2])}x{round(item['bbox'][3])}"
                    for item in members
                )
            )
    return findings


def _bbox_intersection_area(left: list[float], right: list[float]) -> float:
    width = max(0.0, min(left[0] + left[2], right[0] + right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[1] + left[3], right[1] + right[3]) - max(left[1], right[1]))
    return width * height


def _bbox_contains(outer: list[float], inner: list[float], tolerance: float = 1.0) -> bool:
    return (
        outer[0] <= inner[0] + tolerance
        and outer[1] <= inner[1] + tolerance
        and outer[0] + outer[2] + tolerance >= inner[0] + inner[2]
        and outer[1] + outer[3] + tolerance >= inner[1] + inner[3]
    )


def _inspect_html_structure(document: str) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": CANVAS[0], "height": CANVAS[1]},
            device_scale_factor=1,
        )
        page.set_content(document, wait_until="load")
        page.evaluate("document.fonts.ready")
        payload = page.evaluate(
            """() => {
              const alpha = value => {
                const text = String(value || '').trim().toLowerCase();
                if (!text || text === 'transparent') return 0;
                if (text.startsWith('rgba(')) {
                  const parts = text.slice(5, -1).split(/[,/]/).map(part => part.trim());
                  return Number(parts.at(-1));
                }
                if (text.startsWith('rgb(') && text.includes('/')) {
                  return Number(text.slice(4, -1).split('/').at(-1).trim());
                }
                return 1;
              };
              const visibleSurface = node => {
                const style = getComputedStyle(node);
                const borders = [
                  [style.borderTopWidth, style.borderTopStyle],
                  [style.borderRightWidth, style.borderRightStyle],
                  [style.borderBottomWidth, style.borderBottomStyle],
                  [style.borderLeftWidth, style.borderLeftStyle]
                ].filter(([width, kind]) => parseFloat(width) > 0 && kind !== 'none').length;
                const rounded = [style.borderTopLeftRadius, style.borderTopRightRadius,
                  style.borderBottomRightRadius, style.borderBottomLeftRadius]
                  .some(value => parseFloat(value) > 0);
                const filled = alpha(style.backgroundColor) > 0.05;
                return borders >= 3 || (filled && (borders >= 1 || rounded || style.boxShadow !== 'none'));
              };
              const holderFor = node => {
                if (visibleSurface(node)) return node;
                let current = node.parentElement;
                while (current && current !== document.body) {
                  const registered = current.querySelectorAll('[data-content-key]').length;
                  if ((current.hasAttribute('data-holder-id') || visibleSurface(current)) &&
                      registered === 1 && !current.querySelector('[data-role="visual"]')) return current;
                  current = current.parentElement;
                }
                return null;
              };
              const visuals = [...document.querySelectorAll('[data-role="visual"]')]
                .map(node => ({id: node.dataset.regionId || '', rect: node.getBoundingClientRect()}));
              const body = [...document.querySelectorAll('[data-content-key][data-role="body"]')]
                .map(node => node.getBoundingClientRect());
              const holderKeys = [...document.querySelectorAll('[data-content-key]')]
                .filter(node => Boolean(holderFor(node)))
                .map(node => node.dataset.contentKey || '');
              return {
                visuals: visuals.map(item => ({
                  id: item.id,
                  x: item.rect.x,
                  y: item.rect.y,
                  height: item.rect.height,
                  width: item.rect.width
                })),
                body: body.map(rect => ({x: rect.x, width: rect.width})),
                holder_keys: holderKeys,
                holder_ids: [...document.querySelectorAll('[data-holder-id]')]
                  .map(node => node.dataset.holderId || '').filter(Boolean),
                module_ids: [...document.querySelectorAll('[data-module-id]')]
                  .map(node => node.dataset.moduleId || '').filter(Boolean),
                nested_module_ids: [...document.querySelectorAll('[data-module-id]')]
                  .filter(node => node.parentElement?.closest('[data-module-id]'))
                  .map(node => node.dataset.moduleId || '').filter(Boolean),
                unmoduled_region_ids: [...document.querySelectorAll('[data-region-id]')]
                  .filter(node => !node.closest('[data-module-id]'))
                  .map(node => node.dataset.regionId || '').filter(Boolean),
                region_ids: [...document.querySelectorAll('[data-region-id]')]
                  .map(node => node.dataset.regionId || '').filter(Boolean),
                text_region_ids: [...document.querySelectorAll('[data-content-key][data-region-id]')]
                  .map(node => node.dataset.regionId || '').filter(Boolean),
                content_keys: [...document.querySelectorAll('[data-content-key]')]
                  .map(node => node.dataset.contentKey || ''),
                empty_content_keys: [...document.querySelectorAll('[data-content-key]')]
                  .filter(node => !(node.innerText ?? node.textContent ?? '').trim())
                  .map(node => node.dataset.contentKey || ''),
                content_by_key: Object.fromEntries(
                  [...document.querySelectorAll('[data-content-key]')]
                    .map(node => [node.dataset.contentKey || '', (node.innerText ?? node.textContent ?? '').replace(/\s+/g, ' ').trim()])
                ),
                typography_by_key: Object.fromEntries(
                  [...document.querySelectorAll('[data-content-key]')]
                    .map(node => {
                      const style = getComputedStyle(node);
                      return [node.dataset.contentKey || '', {
                        font_size: style.fontSize,
                        font_weight: style.fontWeight,
                        font_style: style.fontStyle,
                        color: style.color,
                        line_height: style.lineHeight,
                        text_align: style.textAlign
                      }];
                    })
                ),
                inline_typography_by_key: Object.fromEntries(
                  [...document.querySelectorAll('[data-content-key]')].map(node => {
                    const walker=document.createTreeWalker(node,NodeFilter.SHOW_TEXT);
                    const runs=[];
                    while(walker.nextNode()) {
                      const text=String(walker.currentNode.nodeValue || '').replace(/\s+/g,' ').trim();
                      if(!text) continue;
                      const parent=walker.currentNode.parentElement, style=getComputedStyle(parent);
                      const ancestry=[];
                      let current=parent;
                      while(current && current!==node) {ancestry.push(current.tagName);current=current.parentElement;}
                      runs.push({text,ancestry,font_size:style.fontSize,
                        font_weight:style.fontWeight,font_style:style.fontStyle,color:style.color,line_height:style.lineHeight});
                    }
                    return [node.dataset.contentKey || '',{runs,
                      structure:[...node.querySelectorAll('br,ul,ol,li')].map(child=>child.tagName)}];
                  })
                ),
                visual_briefs: Object.fromEntries(
                  [...document.querySelectorAll('[data-role="visual"]')]
                    .map(node => [node.dataset.regionId || '', node.dataset.visualBrief || ''])
                ),
                generation_groups: Object.fromEntries(
                  [...document.querySelectorAll('[data-role="visual"]')]
                    .map(node => [node.dataset.regionId || '', node.dataset.generationGroup || ''])
                )
              };
            }"""
        )
        browser.close()
    visuals = payload["visuals"]
    if not visuals:
        visual_side = "none"
    else:
        left = min(item["x"] for item in visuals)
        right = max(item["x"] + item["width"] for item in visuals)
        center = (left + right) / 2
        visual_side = "left" if center < CANVAS[0] * 0.42 else (
            "right" if center > CANVAS[0] * 0.58 else "center"
        )
    if not visuals or not payload["body"]:
        visual_text_relation = "unpaired"
    else:
        visual_center = sum(
            item["x"] + item["width"] / 2 for item in visuals
        ) / len(visuals)
        body_center = sum(
            item["x"] + item["width"] / 2 for item in payload["body"]
        ) / len(payload["body"])
        delta = visual_center - body_center
        visual_text_relation = "visual_left_of_text" if delta < -80 else (
            "visual_right_of_text" if delta > 80 else "interleaved"
        )
    return {
        "visual_side": visual_side,
        "visual_text_relation": visual_text_relation,
        "visual_region_ids": sorted(item["id"] for item in visuals),
        "visual_bboxes": {
            item["id"]: [item["x"], item["y"], item["width"], item["height"]]
            for item in visuals
        },
        "holder_content_keys": sorted(set(payload["holder_keys"])),
        "holder_ids": sorted(set(payload["holder_ids"])),
        "module_ids": sorted(payload["module_ids"]),
        "nested_module_ids": sorted(payload["nested_module_ids"]),
        "unmoduled_region_ids": sorted(payload["unmoduled_region_ids"]),
        "region_ids": sorted(payload["region_ids"]),
        "text_region_ids": sorted(payload["text_region_ids"]),
        "content_keys": sorted(payload["content_keys"]),
        "empty_content_keys": sorted(payload["empty_content_keys"]),
        "content_by_key": payload["content_by_key"],
        "typography_by_key": payload["typography_by_key"],
        "inline_typography_by_key": payload["inline_typography_by_key"],
        "visual_briefs": payload["visual_briefs"],
        "generation_groups": payload["generation_groups"],
    }


def run_html_planning(
    *,
    expression_logic_dir: Path,
    diffusion_dir: Path,
    step2_dir: Path,
    text_generation_dir: Path,
    output_dir: Path,
    slide_ids: list[str],
    config: PipelineConfig,
    mock_vlm: bool,
    client_factory: Callable[[], HtmlPlanningClient] | None = None,
    only_slide_id: str | None = None,
    defer_typography: bool = False,
    write_stage_manifest: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    model_call_count = 0
    usage: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    try:
        deck_plan = DeckPlan.model_validate(read_json(step2_dir / "deck_plan.json"))
        rough_by_slide = {slide.slide_id: slide.rough_message for slide in deck_plan.slides}
        if [slide.slide_id for slide in deck_plan.slides][: len(slide_ids)] != slide_ids:
            raise ValueError("HTML planning slide IDs do not match the coarse deck plan")
        text_plan = parse_text_plan(
            json.dumps(
                read_json(text_generation_dir / "text_plan.json"),
                ensure_ascii=False,
            ),
            deck_plan,
        )
        text_by_slide = {slide["slide_id"]: slide for slide in text_plan["slides"]}
        parsed_text_path = text_generation_dir / "parsed_text_plan.json"
        if not parsed_text_path.is_file():
            raise FileNotFoundError(
                f"HTML planning requires parsed text tree: {parsed_text_path}"
            )
        parsed_payload = read_json(parsed_text_path)
        parsed_slides = parsed_payload.get("slides", [])
        if [slide.get("slide_id") for slide in parsed_slides][: len(slide_ids)] != slide_ids:
            raise ValueError("HTML planning text tree IDs do not match slide IDs")
        parsed_text_by_slide = {
            slide["slide_id"]: slide for slide in parsed_slides
        }
        expression_manifest = read_json(expression_logic_dir / "run_manifest.json")
        if (
            expression_manifest.get("stage") != "deck_level_expression_logic_planning"
            or expression_manifest.get("status") != "succeeded"
        ):
            raise ValueError("HTML planning requires succeeded Expression Logic")
        expression_plan = read_json(
            expression_logic_dir / "expression_logic_plan.json"
        )
        expression_slides = expression_plan.get("slides", [])
        if [item.get("slide_id") for item in expression_slides][
            : len(slide_ids)
        ] != slide_ids:
            raise ValueError("HTML planning expression logic IDs do not match slide IDs")
        expression_by_slide = {
            item["slide_id"]: str(item["expression_logic"]).strip()
            for item in expression_slides
        }
        client = None
        if not mock_vlm:
            client = client_factory() if client_factory else ApiMartHtmlPlanningClient(
                config=config
            )
        base_client = client
        html_by_slide: dict[str, Path] = {}
        preview_by_slide: dict[str, Path] = {}

        if only_slide_id is not None and only_slide_id not in slide_ids:
            raise ValueError(f"Unknown slide: {only_slide_id}")
        for slide_id in slide_ids:
            if only_slide_id is not None and slide_id != only_slide_id:
                continue
            reference_path = diffusion_dir / "skeletons" / f"{slide_id}.png"
            if not reference_path.is_file():
                raise FileNotFoundError(
                    f"HTML planning requires diffusion reference: {reference_path}"
                )
            source_text = text_by_slide[slide_id]
            content = _content_items(source_text)
            frozen_content_by_key = {
                item["content_key"]: item["text"] for item in content
            }
            text_tree = parsed_text_by_slide[slide_id]
            expression_logic = expression_by_slide[slide_id]
            prompt = build_html_planning_prompt(
                slide_id=slide_id,
                expression_logic=expression_logic,
                text_tree=text_tree,
                frozen_content_by_key=frozen_content_by_key,
                deck_font=config.deck_font,
                deck_context={"deck_style": deck_plan.style, "deck_goal": deck_plan.deck_goal,
                              "slide_purpose": rough_by_slide[slide_id]},
            )
            slide_dir = output_dir / "slides" / slide_id
            if base_client is not None:
                client = JournaledHtmlPlanningClient(base_client, slide_dir / "model_calls")
            prompt_path = slide_dir / "prompt.txt"
            raw_path = slide_dir / "raw_response.txt"
            html_path = slide_dir / "layout.html"
            write_text(prompt_path, prompt)
            if mock_vlm:
                raw = build_mock_html(
                    slide_id=slide_id,
                    expression_logic=expression_logic,
                    content=content,
                )
            else:
                assert client is not None
                model_call_count += 1
                raw = client.vlm_html(
                    model=config.chat_model,
                    prompt=prompt,
                    image_path=reference_path,
                    max_tokens=config.vlm_max_tokens,
                )
                if client.last_usage is not None:
                    usage.append({"slide_id": slide_id, **client.last_usage})
            write_text(raw_path, raw)
            extraction_warnings: list[str] = []
            try:
                document = extract_html_document(raw, extraction_warnings)
            except (ValueError, TypeError) as exc:
                document = build_mock_html(
                    slide_id=slide_id, expression_logic=expression_logic, content=content
                )
                extraction_warnings.append(
                    f"invalid initial HTML; used deterministic same-stage recovery: {exc}"
                )
            warnings.extend(
                {
                    "slide_id": slide_id,
                    "stage": "html_extraction",
                    "warning": warning,
                }
                for warning in extraction_warnings
            )
            write_text(html_path, document)
            for warning in prepare_html_layout(
                html_path, deck_font=config.deck_font,
                source_text=source_text, slide_id=slide_id,
            ):
                warnings.append({"slide_id": slide_id, "stage": "html_local_recovery", "warning": warning})
            readability = apply_initial_readability_contract(html_path)
            holder_allocation = allocate_holder_text_space(html_path)
            text_allocation = allocate_readable_text_space(html_path)
            document = html_path.read_text(encoding="utf-8")
            initial_html_path = slide_dir / "layout_initial.html"
            write_text(initial_html_path, document)
            candidate_visual_groups = _inspect_visual_group_candidates(
                html_path, expression_logic
            )
            visual_generation_groups = candidate_visual_groups
            critic_record = {
                "status": "skipped" if mock_vlm else "pending",
                "needs_rework": False,
                "rework_attempted": False,
            }
            if not mock_vlm:
                critic_image_path = slide_dir / "critic_input_preview.png"
                try:
                    _render_critic_preview(html_path, critic_image_path)
                    deterministic_findings = _find_layout_contract_issues(html_path)
                    critic_prompt = build_html_critic_prompt(
                        slide_id=slide_id,
                        text_tree=text_tree,
                        frozen_content_by_key=frozen_content_by_key,
                        deterministic_findings=deterministic_findings,
                        expression_logic=expression_logic,
                        candidate_visual_groups=candidate_visual_groups,
                    )
                    def call_critic(active_prompt: str) -> str:
                        nonlocal model_call_count
                        model_call_count += 1
                        return client.vlm_json(
                            model=config.chat_model,
                            prompt=active_prompt,
                            image_path=critic_image_path,
                            max_tokens=config.vlm_max_tokens,
                        )

                    def record_critic_usage(attempt: int, _: str) -> None:
                        if client.last_usage is not None:
                            usage.append(
                                {
                                    "slide_id": slide_id,
                                    "call": "critic",
                                    "attempt": attempt,
                                    **client.last_usage,
                                }
                            )

                    critic, critic_raw = call_json_with_retries(
                        prompt=critic_prompt,
                        call=call_critic,
                        parse=_parse_critic_response,
                        response_path=slide_dir / "critic_raw_response.txt",
                        on_response=record_critic_usage,
                    )
                    critic["issues"] = list(
                        dict.fromkeys([*deterministic_findings, *critic["issues"]])
                    )
                    critic["needs_rework"] = bool(
                        deterministic_findings or critic["needs_rework"]
                    )
                    critic_record = {
                        **critic_record,
                        "status": "succeeded",
                        "needs_rework": critic["needs_rework"],
                        "issues": critic["issues"],
                        "summary": critic["summary"],
                        "visual_generation_groups": candidate_visual_groups,
                    }
                    write_json(slide_dir / "critic_response.json", critic)
                    if critic["needs_rework"]:
                        layout_critic = {
                            "needs_rework": critic["needs_rework"],
                            "issues": critic["issues"],
                            "summary": critic["summary"],
                        }
                        rework_prompt = build_html_rework_prompt(prompt, layout_critic, document)
                        write_text(slide_dir / "rework_prompt.txt", rework_prompt)
                        model_call_count += 1
                        reworked_raw = client.vlm_html(
                            model=config.chat_model,
                            prompt=rework_prompt,
                            image_path=critic_image_path,
                            max_tokens=config.vlm_max_tokens,
                        )
                        if client.last_usage is not None:
                            usage.append({"slide_id": slide_id, "call": "rework", **client.last_usage})
                        write_text(slide_dir / "raw_response_rework.txt", reworked_raw)
                        rework_warnings: list[str] = []
                        reworked_document = extract_html_document(
                            reworked_raw, rework_warnings
                        )
                        warnings.extend(
                            {
                                "slide_id": slide_id,
                                "stage": "html_rework_extraction",
                                "warning": warning,
                            }
                            for warning in rework_warnings
                        )
                        critic_record["rework_attempted"] = True
                        preserves, reason = _rework_preserves_structure(
                            original_html=document,
                            reworked_html=reworked_document,
                        )
                        if preserves:
                            raw = reworked_raw
                            document = reworked_document
                            write_text(raw_path, raw)
                            write_text(html_path, document)
                            post_rework_findings = _find_layout_contract_issues(html_path)
                            if post_rework_findings:
                                second_preview_path = slide_dir / "critic_post_rework_preview.png"
                                _render_critic_preview(html_path, second_preview_path)
                                second_critic = {
                                    "needs_rework": True,
                                    "issues": post_rework_findings,
                                    "summary": (
                                        "The first revision left deterministic contract issues. "
                                        "Fix them without introducing new decoration or geometry errors."
                                    ),
                                }
                                second_prompt = build_html_rework_prompt(
                                    prompt, second_critic, document
                                )
                                write_text(slide_dir / "rework_prompt_2.txt", second_prompt)
                                model_call_count += 1
                                second_raw = client.vlm_html(
                                    model=config.chat_model,
                                    prompt=second_prompt,
                                    image_path=second_preview_path,
                                    max_tokens=config.vlm_max_tokens,
                                )
                                if client.last_usage is not None:
                                    usage.append(
                                        {
                                            "slide_id": slide_id,
                                            "call": "rework_2",
                                            **client.last_usage,
                                        }
                                    )
                                write_text(slide_dir / "raw_response_rework_2.txt", second_raw)
                                second_warnings: list[str] = []
                                second_document = extract_html_document(
                                    second_raw, second_warnings
                                )
                                warnings.extend(
                                    {
                                        "slide_id": slide_id,
                                        "stage": "html_rework_2_extraction",
                                        "warning": warning,
                                    }
                                    for warning in second_warnings
                                )
                                critic_record["second_rework_attempted"] = True
                                second_preserves, second_reason = _rework_preserves_structure(
                                    original_html=document,
                                    reworked_html=second_document,
                                )
                                if second_preserves:
                                    raw = second_raw
                                    document = second_document
                                    write_text(raw_path, raw)
                                    write_text(html_path, document)
                                    post_rework_findings = _find_layout_contract_issues(
                                        html_path
                                    )
                                    critic_record["second_rework_accepted"] = True
                                else:
                                    critic_record["second_rework_accepted"] = False
                                    critic_record["second_rework_rejection_reason"] = (
                                        second_reason
                                    )
                                    warnings.append(
                                        {
                                            "slide_id": slide_id,
                                            "stage": "html_rework_2_structure_guard",
                                            "warning": second_reason,
                                        }
                                    )
                            critic_record["post_rework_findings"] = post_rework_findings
                            if post_rework_findings:
                                warnings.extend(
                                    {
                                        "slide_id": slide_id,
                                        "stage": "html_rework_postcheck",
                                        "warning": finding,
                                    }
                                    for finding in post_rework_findings
                                )
                            visual_generation_groups = _inspect_visual_group_candidates(
                                html_path, expression_logic
                            )
                            critic_record["rework_accepted"] = True
                        else:
                            critic_record["rework_accepted"] = False
                            critic_record["rework_rejection_reason"] = reason
                            warnings.append(
                                {
                                    "slide_id": slide_id,
                                    "stage": "html_rework_structure_guard",
                                    "warning": reason,
                                }
                            )
                    write_json(slide_dir / "critic_response.json", critic_record)
                except Exception as exc:
                    warning = {
                        "slide_id": slide_id,
                        "stage": "critic_or_rework",
                        "error": str(exc),
                    }
                    warnings.append(warning)
                    critic_record = {
                        **critic_record,
                        "status": "warning",
                        "error": str(exc),
                    }
                    # Keep the last valid model HTML; critic failure is non-fatal.
                    write_json(slide_dir / "critic_response.json", critic_record)

            text_rework_record: dict[str, Any] = {
                "status": "skipped" if mock_vlm else "pending",
                "accepted_content_keys": [],
                "rejected_content_keys": [],
            }
            if not mock_vlm:
                assert client is not None
                text_rework_input = slide_dir / "text_rework_input_preview.png"
                try:
                    _render_critic_preview(html_path, text_rework_input)
                    current_blocks = inspect_body_blocks(html_path)
                    text_rework_prompt = build_text_rework_prompt(
                        slide_id=slide_id,
                        text_tree=text_tree,
                        current_blocks=current_blocks,
                    )
                    write_text(slide_dir / "text_rework_prompt.txt", text_rework_prompt)
                    allowed_blocks = {
                        str(item["content_key"]): item for item in current_blocks
                    }

                    def call_text_rework(active_prompt: str) -> str:
                        nonlocal model_call_count
                        model_call_count += 1
                        return client.vlm_json(
                            model=config.chat_model,
                            prompt=active_prompt,
                            image_path=text_rework_input,
                            max_tokens=config.vlm_max_tokens,
                        )

                    def record_text_rework_usage(attempt: int, _: str) -> None:
                        if client.last_usage is not None:
                            usage.append(
                                {
                                    "slide_id": slide_id,
                                    "call": "text_rework",
                                    "attempt": attempt,
                                    **client.last_usage,
                                }
                            )

                    text_rework_plan, text_rework_raw = call_json_with_retries(
                        prompt=text_rework_prompt,
                        call=call_text_rework,
                        parse=lambda raw: parse_text_rework_response(
                            raw,
                            slide_id=slide_id,
                            allowed_blocks=allowed_blocks,
                        ),
                        response_path=slide_dir / "text_rework_raw_response.txt",
                        on_response=record_text_rework_usage,
                    )
                    text_rework_record = apply_text_rework_plan(
                        html_path=html_path,
                        plan=text_rework_plan,
                    )
                    write_text_rework_artifacts(
                        slide_dir=slide_dir,
                        prompt=text_rework_prompt,
                        raw=text_rework_raw,
                        plan=text_rework_plan,
                        report=text_rework_record,
                    )
                    _render_critic_preview(
                        html_path, slide_dir / "text_rework_preview.png"
                    )
                except Exception as exc:
                    text_rework_record = {
                        **text_rework_record,
                        "status": "warning",
                        "error": str(exc),
                    }
                    warnings.append(
                        {
                            "slide_id": slide_id,
                            "stage": "text_rework",
                            "warning": str(exc),
                        }
                    )
                    write_json(
                        slide_dir / "text_rework_report.json", text_rework_record
                    )
            for warning in prepare_html_layout(
                html_path, deck_font=config.deck_font,
                source_text=source_text, slide_id=slide_id,
            ):
                warnings.append({"slide_id": slide_id, "stage": "html_final_local_recovery", "warning": warning})
            # Rework can add a lead or extra paragraph after the first fit pass.
            # Allocate its actual resulting copy before committing HTML geometry.
            post_rework_repairs = {
                "holder_allocation": allocate_holder_text_space(html_path),
                "text_allocation": allocate_readable_text_space(html_path),
            }
            text_rework_record["precommit_text_repairs"] = post_rework_repairs
            write_json(slide_dir / "text_rework_report.json", text_rework_record)
            final_preview_path = slide_dir / "critic_preview.png"
            _render_critic_preview(html_path, final_preview_path)
            html_by_slide[slide_id] = html_path
            preview_by_slide[slide_id] = final_preview_path
            visual_groups_path = slide_dir / "visual_generation_groups.json"
            write_json(
                visual_groups_path,
                {
                    "schema_version": "1.0",
                    "slide_id": slide_id,
                    "geometry_authority": "final_html_region_ids",
                    "groups": visual_generation_groups,
                },
            )
            records.append(
                {
                    "slide_id": slide_id,
                    "expression_logic": expression_logic,
                    "precommit_text_repairs": {"readability": readability, "holder_allocation": holder_allocation,
                                               "text_allocation": text_allocation,
                                               "post_rework": post_rework_repairs},
                    "expression_logic_source": str(
                        (
                            expression_logic_dir
                            / "slides"
                            / slide_id
                            / "expression_logic.json"
                        ).resolve()
                    ),
                    "diffusion_reference": str(reference_path.resolve()),
                    "prompt": str(prompt_path.resolve()),
                    "raw_response": str(raw_path.resolve()),
                    "html": str(html_path.resolve()),
                    "initial_html": str(initial_html_path.resolve()),
                    "final_preview": str(final_preview_path.resolve()),
                    "visual_generation_groups": str(visual_groups_path.resolve()),
                    "content_keys": [item["content_key"] for item in content],
                    "critic": critic_record,
                    "text_rework": text_rework_record,
                }
            )

        typography_result = ({"typography": {"status": "deferred"}, "warnings": [], "usage": [], "model_call_count": 0}
                             if defer_typography else commit_html_typography(
                                 output_dir=output_dir, slide_ids=slide_ids, records=records,
                                 config=config, mock_vlm=mock_vlm,
                                 client_factory=(lambda: base_client) if base_client is not None else None))
        typography_record = typography_result["typography"]
        warnings.extend(typography_result["warnings"])
        usage.extend(typography_result["usage"])
        model_call_count += typography_result["model_call_count"]
        manifest = {
            "stage": "diffusion_inspired_html_planning",
            "status": "succeeded",
            "quality_status": "mock" if mock_vlm else ("degraded" if warnings else "strict_pass"),
            "model": None if mock_vlm else config.chat_model,
            "model_call_count": model_call_count,
            "call_granularity": (
                "planner_plus_geography_critic_and_up_to_two_geography_reworks_"
                "plus_one_text_rework_per_slide_plus_one_deck_typography_critic"
            ),
            "mock_vlm": mock_vlm,
            "canvas": list(CANVAS),
            "slide_count": len(records),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "usage": usage,
            "warnings": warnings,
            "typography": typography_record,
            "output_dir": str(output_dir.resolve()),
            "slides": records,
        }
    except Exception as exc:
        manifest = {
            "stage": "diffusion_inspired_html_planning",
            "status": "failed",
            "error": str(exc),
            "model_call_count": model_call_count,
            "mock_vlm": mock_vlm,
            "slide_count": len(records),
            "output_dir": str(output_dir.resolve()),
            "slides": records,
        }
    if write_stage_manifest:
        write_json(output_dir / "run_manifest.json", manifest)
    if manifest["status"] != "succeeded":
        raise RuntimeError(f"HTML planning failed: {manifest['error']}")
    return manifest


def commit_html_typography(
    *, output_dir: Path, slide_ids: list[str], records: list[dict[str, Any]],
    config: PipelineConfig, mock_vlm: bool,
    client_factory: Callable[[], HtmlPlanningClient] | None = None,
) -> dict[str, Any]:
    """Commit the existing shared role profile while all page owners are paused."""
    html_by_slide = {record["slide_id"]: Path(record["html"]) for record in records}
    preview_by_slide = {record["slide_id"]: Path(record["final_preview"]) for record in records}
    if list(html_by_slide) != slide_ids:
        raise ValueError("Typography requires every selected slide in deck order")
    warnings: list[dict[str, Any]] = []
    usage: list[dict[str, Any]] = []
    model_call_count = 0
    client = None if mock_vlm else JournaledHtmlPlanningClient(
        client_factory() if client_factory else ApiMartHtmlPlanningClient(config=config),
        output_dir / "typography_calls",
    )
    typography_record: dict[str, Any] = {
        "status": "skipped" if mock_vlm else "pending"
    }
    if not mock_vlm:
        assert client is not None
        try:
            typography_metrics = inspect_deck_typography(html_by_slide)
            typography_prompt = build_typography_profile_prompt(typography_metrics)
            typography_montage = output_dir / "typography_input_montage.png"
            create_preview_montage(
                [preview_by_slide[slide_id] for slide_id in slide_ids],
                typography_montage,
            )
            write_text(output_dir / "typography_prompt.txt", typography_prompt)

            def call_typography(active_prompt: str) -> str:
                nonlocal model_call_count
                model_call_count += 1
                return client.vlm_json(
                    model=config.chat_model,
                    prompt=active_prompt,
                    image_path=typography_montage,
                    max_tokens=config.vlm_max_tokens,
                )

            def record_typography_usage(attempt: int, _: str) -> None:
                if client.last_usage is not None:
                    usage.append(
                        {
                            "call": "deck_typography",
                            "attempt": attempt,
                            **client.last_usage,
                        }
                    )

            typography_profile, typography_raw = call_json_with_retries(
                prompt=typography_prompt,
                call=call_typography,
                parse=lambda raw: parse_typography_profile(
                    raw, typography_metrics
                ),
                response_path=output_dir / "typography_raw_response.txt",
                on_response=record_typography_usage,
            )
            typography_record = apply_deck_typography_profile(
                html_by_slide=html_by_slide,
                profile=typography_profile,
            )
            typography_record["metrics"] = typography_metrics
            typography_record["profile"] = typography_profile
            write_json(output_dir / "typography_profile.json", typography_record)
            if typography_record["status"] in {"accepted", "partial"}:
                for record in records:
                    slide_id = record["slide_id"]
                    record["precommit_text_repairs"]["final_role_space"] = allocate_readable_text_space(html_by_slide[slide_id])
                    _render_critic_preview(
                        html_by_slide[slide_id], preview_by_slide[slide_id]
                    )
            for role in typography_record.get("unresolved_roles", []):
                warnings.append({"stage": "deck_typography", "warning": f"{role} retained: no readable geometry-preserving shared profile"})
        except Exception as exc:
            typography_record = {"status": "warning", "error": str(exc)}
            warnings.append(
                {"stage": "deck_typography", "warning": str(exc)}
            )
            write_json(output_dir / "typography_profile.json", typography_record)
    return {"typography": typography_record, "warnings": warnings,
            "usage": usage, "model_call_count": model_call_count, "slides": records}


def _content_items(source_text: dict[str, Any]) -> list[dict[str, str]]:
    items = [
        {
            "content_key": "title",
            "role": "title",
            "text": source_text["title"],
        }
    ]
    if source_text["subtitle"] is not None:
        items.append(
            {
                "content_key": "subtitle",
                "role": "subtitle",
                "text": source_text["subtitle"],
            }
        )
    items.extend(
        {
            "content_key": key,
            "role": "body",
            "text": value,
        }
        for key, value in source_text.items()
        if TEXT_FIELD_PATTERN.fullmatch(key)
    )
    return items


def build_mock_html(
    *, slide_id: str, expression_logic: str, content: list[dict[str, str]]
) -> str:
    title = content[0]
    body = content[1:]
    top = 174
    bottom = 820
    gap = 16
    body_height = (
        (bottom - top - gap * max(0, len(body) - 1)) // max(1, len(body))
    )
    title_region = (
        _mock_text_region(
            item=title,
            region_id="r_title",
            x=0,
            y=0,
            width=1424,
            height=102,
            equivalent_group=None,
        )
    )
    body_regions = []
    for index, item in enumerate(body):
        body_regions.append(
            _mock_text_region(
                item=item,
                region_id=f"r_{item['content_key']}",
                x=0,
                y=index * (body_height + gap),
                width=676,
                height=body_height,
                equivalent_group="copy-stack" if len(body) > 1 else None,
            )
        )
    visual_brief = html_lib.escape(
        f"Create the explanatory visual described by: {expression_logic}", quote=True
    )
    visual_region = f'''<div class="region visual" data-region-id="r_visual_01" data-role="visual"
    data-generation-group="visual_group_r_visual_01" data-visual-brief="{visual_brief}"
    style="left:0;top:0;width:712px;height:646px"></div>'''
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><style>
*{{box-sizing:border-box}}html,body{{margin:0;padding:0;background:#fff}}
.slide{{position:relative;width:1536px;height:864px;overflow:hidden;background:#fff;color:#222;font-family:Arial,sans-serif}}
.module{{position:absolute}}
.region{{position:absolute;overflow:hidden}}
.text{{padding:14px 18px;white-space:pre-line;line-height:1.08;letter-spacing:0}}
.title{{font-size:54.4px;font-weight:700;padding:18px 24px}}
.subtitle{{font-size:35.2px;font-weight:400}}.body{{font-size:28.8px}}
.visual{{border:2px dashed #777;background:#f3f3f3}}
</style></head><body>
  <section class="slide" data-slide-id="{html_lib.escape(slide_id, quote=True)}">
  <div class="module" data-module-id="module_title" style="left:56px;top:40px;width:1424px;height:102px">{title_region}</div>
  <div class="module" data-module-id="module_copy" style="left:56px;top:174px;width:676px;height:646px">{''.join(body_regions)}</div>
  <div class="module" data-module-id="module_visual" style="left:768px;top:174px;width:712px;height:646px">{visual_region}</div>
</section></body></html>
'''


def _mock_text_region(
    *,
    item: dict[str, str],
    region_id: str,
    x: int,
    y: int,
    width: int,
    height: int,
    equivalent_group: str | None,
) -> str:
    group = (
        f' data-equivalent-group="{html_lib.escape(equivalent_group, quote=True)}"'
        if equivalent_group
        else ""
    )
    return (
        f'<div class="region text {item["role"]}" data-region-id="{region_id}" '
        f'data-role="{item["role"]}" data-content-key="{item["content_key"]}" '
        f'data-text-surface="plain"{group} '
        f'style="left:{x}px;top:{y}px;width:{width}px;height:{height}px">'
        f'{html_lib.escape(item["text"])}'
        f'</div>'
    )
