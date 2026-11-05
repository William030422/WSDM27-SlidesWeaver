from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageChops

from integration.config import load_config
from integration.html_compile import (
    _validate_regions,
    run_html_to_coordinates,
)
from integration.html_planning import (
    build_mock_html,
    build_html_critic_prompt,
    build_html_planning_prompt,
    build_html_rework_prompt,
    extract_html_document,
    _rework_preserves_structure,
    _find_layout_contract_issues,
    _inspect_html_structure,
    _parse_critic_response,
    _render_critic_preview,
    run_html_planning,
)
from integration.html_text_rework import (
    apply_deck_typography_profile,
    apply_text_rework_plan,
    build_text_rework_prompt,
    build_typography_profile_prompt,
    inspect_deck_typography,
    parse_text_rework_response,
    parse_typography_profile,
)


ROOT = Path(__file__).resolve().parents[1]


class FlippingReworkClient:
    last_usage = None

    def __init__(self, initial_html: str, reworked_html: str) -> None:
        self.initial_html = initial_html
        self.reworked_html = reworked_html
        self.html_calls = 0
        self.initial_vlm_html_kwargs = None

    def chat_html(self, **kwargs: object) -> str:
        self.chat_html_kwargs = kwargs
        self.html_calls += 1
        return self.initial_html

    def vlm_html(self, **_: object) -> str:
        if self.html_calls == 0:
            self.initial_vlm_html_kwargs = _
            self.html_calls += 1
            return self.initial_html
        self.html_calls += 1
        return self.reworked_html

    def vlm_json(self, **_: object) -> str:
        return json.dumps(
            {
                "needs_rework": True,
                "issues": ["Resolve a local overflow."],
                "summary": "Only a local correction is needed.",
            }
        )


class HtmlPipelineTests(unittest.TestCase):
    def test_critic_parser_normalizes_structured_issues(self) -> None:
        parsed = _parse_critic_response(
            json.dumps(
                {
                    "needs_rework": True,
                    "issues": [
                        {
                            "region": "subtitle/divider",
                            "issue": "The subtitle overlaps the divider.",
                            "fix": "Move the divider down.",
                        }
                    ],
                    "summary": "local",
                }
            )
        )
        self.assertEqual(
            parsed["issues"],
            ["subtitle/divider; The subtitle overlaps the divider.; fix: Move the divider down."],
        )

    def test_prompt_declares_final_text_surface_and_independent_visual_contract(self) -> None:
        text_tree = {
            "slide_id": "slide_001",
            "title": "Learn by trying",
            "subtitle": None,
            "text_blocks": [
                {
                    "text_id": "text_01",
                    "format": "unordered_list",
                    "lead": "Observe and update:",
                    "items": ["Try an action", "Learn from feedback"],
                }
            ],
        }
        frozen = {
            "title": "Learn by trying",
            "text_01": "Observe and update:\n- Try an action\n- Learn from feedback",
        }
        prompt = build_html_planning_prompt(
            slide_id="slide_001",
            expression_logic=(
                "Place one trial visual in the center and attach text_01 to the "
                "outcome it explains."
            ),
            text_tree=text_tree,
            frozen_content_by_key=frozen,
        )

        self.assertIn('data-text-surface="plain|holder"', prompt)
        self.assertIn('data-holder-id="holder_<region-id>"', prompt)
        self.assertIn("source text is frozen display copy", prompt)
        self.assertIn("do not rewrite, shorten, reorder", prompt)
        self.assertIn("2 to 6 top-level, non-nested modules", prompt)
        self.assertIn("mapping 1:1 to one generated asset", prompt)
        self.assertIn("Never share a generation group", prompt)
        self.assertIn("no artwork, icons, texture", prompt)
        self.assertIn("20px body text", prompt)
        self.assertIn("readable floors", prompt)
        self.assertIn("prefer 2-3 substantial picture-text pairs", prompt)
        self.assertIn("use 1-3 meaningful text holders", prompt)
        self.assertIn("background:transparent; border:2px solid #777; box-shadow:none", prompt)
        self.assertNotIn("54.4px", prompt)
        self.assertIn('"expression_logic"', prompt)
        self.assertIn("attached diffusion", prompt.lower())
        self.assertNotIn('"style"', prompt)

        critic_prompt = build_html_critic_prompt(
            slide_id="slide_001",
            text_tree=text_tree,
            frozen_content_by_key=frozen,
            deterministic_findings=["region_text_02 overlaps visual_learned_model"],
        )
        self.assertIn("poor text-image pairing", critic_prompt)
        self.assertIn("region_text_02 overlaps visual_learned_model", critic_prompt)
        self.assertIn("one generic oversized visual", critic_prompt)
        self.assertIn("This is the geography critic", critic_prompt)
        self.assertIn("Do not request wording, emphasis, font-size", critic_prompt)
        self.assertIn("transparent simple solid-line frame", critic_prompt)

        rework_prompt = build_html_rework_prompt(
            prompt,
            {"needs_rework": True, "issues": ["Fix overflow."], "summary": "local"},
            "<!doctype html><section class='slide'></section>",
        )
        self.assertIn("reposition or resize modules", rework_prompt)
        self.assertIn("Preserve all text region IDs, content keys, visible wording", rework_prompt)
        self.assertIn("Do not change any visible text", rework_prompt)
        self.assertIn("replace, split, or merge visual regions", rework_prompt)
        self.assertIn("CURRENT HTML", rework_prompt)

    def test_mock_html_keeps_registered_text_plain(self) -> None:
        mock_html = build_mock_html(
            slide_id="slide_001",
            expression_logic="Pair the text with one explanatory visual.",
            content=[
                {"content_key": "title", "role": "title", "text": "Learn by trying"},
                {"content_key": "text_01", "role": "body", "text": "Observe and update."},
            ],
        )
        self.assertEqual(mock_html.count('data-text-surface="plain"'), 2)

    def test_overflow_keeps_html_text_and_holder_geometry_exact(self) -> None:
        payload = {
            "extra_text": [],
            "regions": [
                self._raw_region(
                    region_id="title_001",
                    role="title",
                    content_key="title",
                    bbox=[40, 30, 600, 70],
                    text="Overflow example",
                ),
                {
                    **self._raw_region(
                        region_id="body_001",
                        role="body",
                        content_key="text_01",
                        bbox=[100, 150, 200, 50],
                        text="The text needs more room.",
                    ),
                    "scroll_width": 250,
                    "scroll_height": 90,
                    "overflow": True,
                    "holder": {
                        "holder_id": "holder_body_001",
                        "bbox_px": [80, 130, 240, 90],
                        "source": "explicit_html_holder",
                    },
                },
            ],
        }

        regions, warnings = _validate_regions(
            payload,
            {
                "slide_id": "slide_001",
                "title": "Overflow example",
                "subtitle": None,
                "text_01": "The text needs more room.",
            },
        )

        body = next(item for item in regions if item["content_key"] == "text_01")
        self.assertEqual(body["text_bbox_px"], [100, 150, 200, 50])
        self.assertEqual(body["holder"]["bbox_px"], [80, 130, 240, 90])
        self.assertEqual(body["holder"]["geometry_authority"], "explicit_html_holder")
        self.assertTrue(body["overflow_detected"])
        self.assertTrue(any("overflows its HTML-planned text box" in item for item in warnings))

    def test_browser_layout_preserves_each_visual_region(self) -> None:
        config = load_config(ROOT)
        with tempfile.TemporaryDirectory(prefix="rl4ppt-html-final-") as temporary:
            root = Path(temporary)
            self._write_compile_fixture(root)
            manifest = run_html_to_coordinates(
                html_planning_dir=root / "html_planning",
                step2_dir=root / "step2",
                text_generation_dir=root / "text_generation",
                output_dir=root / "html_compile",
                slide_ids=["slide_001"],
                config=config,
            )
            layout = json.loads(
                (root / "html_compile/slides/slide_001/html_layout.json").read_text()
            )

        self.assertEqual(manifest["status"], "succeeded")
        self.assertFalse(manifest["topology_generated"])
        self.assertFalse(manifest["calibration_executed"])
        self.assertEqual(
            manifest["slides"][0]["visual_region_count"], 2
        )
        self.assertEqual(
            [item["region_id"] for item in layout["visual_regions"]],
            ["visual_a", "visual_b"],
        )
        self.assertEqual(
            [item["bbox_px"] for item in layout["visual_regions"]],
            [[600, 160, 360, 240], [1000, 440, 320, 220]],
        )
        self.assertNotIn("visual_lump", layout)
        self.assertEqual(
            [item["holder_id"] for item in layout["holders"]],
            ["holder_body_001", "holder_body_003"],
        )
        title = next(item for item in layout["text_regions"] if item["role"] == "title")
        self.assertEqual(title["text_bbox_px"], [60, 40, 700, 80])
        self.assertEqual(title["text_alignment"], "center")
        self.assertIsNone(title["holder"])
        divider = next(item for item in layout["text_regions"] if item["content_key"] == "text_02")
        self.assertEqual(divider["presentation_mode"], "direct_text")
        self.assertIsNone(divider["holder"])
        tinted = next(item for item in layout["text_regions"] if item["content_key"] == "text_03")
        self.assertEqual(tinted["presentation_mode"], "generated_holder_text")
        self.assertEqual(tinted["holder"]["source"], "explicit_html_holder")
        rewritten = next(item for item in layout["text_regions"] if item["content_key"] == "text_01")
        self.assertEqual(rewritten["display_text"], "Observe the outcome.")
        self.assertFalse(rewritten["copy_changed"])
        self.assertEqual(rewritten["typography"]["font_size_pt"], 18.75)

    def test_background_texture_does_not_override_html_title_geometry(self) -> None:
        config = load_config(ROOT)
        with tempfile.TemporaryDirectory(prefix="rl4ppt-title-holder-") as temporary:
            root = Path(temporary)
            self._write_compile_fixture(root)
            background = Image.new("RGB", (1536, 864), "white")
            for x in range(418, 1118):
                for y in range(30, 130):
                    background.putpixel((x, y), (20, 30, 40) if (x // 8) % 2 else (230, 220, 190))
            background.save(root / "step2/canonical_background.png")
            run_html_to_coordinates(
                html_planning_dir=root / "html_planning",
                step2_dir=root / "step2",
                text_generation_dir=root / "text_generation",
                output_dir=root / "html_compile",
                slide_ids=["slide_001"],
                config=config,
            )
            layout = json.loads(
                (root / "html_compile/slides/slide_001/html_layout.json").read_text()
            )
        title = next(item for item in layout["text_regions"] if item["role"] == "title")
        self.assertEqual(title["text_alignment"], "center")
        self.assertEqual(title["text_bbox_px"], [60, 40, 700, 80])
        self.assertIsNone(title["holder"])

    def test_html_compile_extracts_reworked_leads_bullets_and_inline_runs(self) -> None:
        config = load_config(ROOT)
        with tempfile.TemporaryDirectory(prefix="rl4ppt-rich-html-") as temporary:
            root = Path(temporary)
            self._write_compile_fixture(root)
            html_path = root / "html_planning/slides/slide_001/layout.html"
            document = html_path.read_text(encoding="utf-8")
            document = document.replace(
                '>Observe, then learn.</div></div>',
                ''' data-text-reworked="true">
<div data-text-part="lead" style="font-size:1.08em;font-weight:700"><strong>Observe</strong></div>
<div data-text-part="bullet"><span data-generated-bullet="true">• </span><strong>Try</strong><span> an action</span></div>
<div data-text-part="bullet"><span data-generated-bullet="true">• </span><span>Learn from feedback</span></div>
</div></div>''',
            )
            html_path.write_text(document, encoding="utf-8")
            text_plan_path = root / "text_generation/text_plan.json"
            text_plan = json.loads(text_plan_path.read_text(encoding="utf-8"))
            text_plan["slides"][0]["text_01"] = "Observe\n- Try an action\n- Learn from feedback"
            text_plan_path.write_text(json.dumps(text_plan), encoding="utf-8")
            run_html_to_coordinates(
                html_planning_dir=root / "html_planning",
                step2_dir=root / "step2",
                text_generation_dir=root / "text_generation",
                output_dir=root / "html_compile",
                slide_ids=["slide_001"],
                config=config,
            )
            layout = json.loads(
                (root / "html_compile/slides/slide_001/html_layout.json").read_text()
            )
        body = next(
            item for item in layout["text_regions"] if item["content_key"] == "text_01"
        )
        rich = body["display_content"]
        self.assertEqual(rich["format"], "rich_text")
        self.assertEqual([item["kind"] for item in rich["paragraphs"]], ["lead", "bullet", "bullet"])
        self.assertTrue(rich["paragraphs"][0]["runs"][0]["bold"])
        self.assertTrue(rich["paragraphs"][1]["runs"][0]["bold"])
        self.assertGreater(rich["paragraphs"][0]["runs"][0]["font_scale"], 1)

    def test_deterministic_critic_findings_name_overlapping_regions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-html-findings-") as temporary:
            html_path = Path(temporary) / "layout.html"
            html_path.write_text(
                """<!doctype html><style>.a,.b{position:absolute;width:200px;height:100px}.a{left:10px}.b{left:100px}</style>
<div class="a" data-region-id="region_text_02" data-role="body"></div>
<div class="b" data-region-id="visual_learned_model" data-role="visual"></div>""",
                encoding="utf-8",
            )
            findings = _find_layout_contract_issues(html_path)
        self.assertIn("region_text_02 overlaps visual_learned_model", findings)

    def test_deterministic_critic_findings_reject_degenerate_visual_region(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-html-findings-") as temporary:
            html_path = Path(temporary) / "layout.html"
            html_path.write_text(
                """<!doctype html><style>.visual{position:absolute;width:4px;height:4px}</style>
<div class="visual" data-region-id="visual_predicted_state" data-role="visual"></div>""",
                encoding="utf-8",
            )
            findings = _find_layout_contract_issues(html_path)
        self.assertIn(
            "visual_predicted_state is a degenerate visual region at 4x4px; minimum is 280x180px",
            findings,
        )

    def test_deterministic_critic_requires_holder_for_styled_text_module(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-html-holder-finding-") as temporary:
            html_path = Path(temporary) / "layout.html"
            html_path.write_text(
                """<!doctype html><style>
.module{position:absolute;width:700px;height:300px;background:#eee}
.visual{left:760px;width:700px;height:500px;background:transparent}
</style><div class="module" data-module-id="copy"><p data-region-id="copy_text" data-content-key="text_01" data-role="body">Copy</p></div>
<div class="module visual" data-module-id="visual" data-region-id="visual_a" data-role="visual" data-generation-group="visual_group_visual_a"></div>""",
                encoding="utf-8",
            )
            findings = _find_layout_contract_issues(html_path)
        self.assertIn(
            "styled single-text modules require explicit data-holder-id: copy",
            findings,
        )

    def test_deterministic_critic_accepts_parseable_wireframe_holder(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-html-wireframe-holder-") as temporary:
            html_path = Path(temporary) / "layout.html"
            html_path.write_text(
                """<!doctype html><style>
html,body{margin:0}.slide{position:relative;width:1536px;height:864px}
.module{position:absolute;box-sizing:border-box}.title{left:56px;top:40px;width:1424px;height:100px}
.holder{left:56px;top:164px;width:600px;height:640px;background:transparent;border:2px solid #777;box-shadow:none}
.body{position:absolute;left:24px;top:24px;width:548px;height:592px}
.visual{left:680px;top:164px;width:800px;height:640px}
</style><section class="slide">
<div class="module title" data-module-id="title_module"><h1 data-region-id="title" data-content-key="title" data-role="title" data-text-surface="plain">Title</h1></div>
<div class="module holder" data-module-id="holder_module" data-holder-id="holder_body"><p class="body" data-region-id="body" data-content-key="text_01" data-role="body" data-text-surface="holder">A framed explanation.</p></div>
<div class="module visual" data-module-id="visual_module" data-region-id="visual" data-role="visual" data-generation-group="visual_group_visual"></div>
</section>""",
                encoding="utf-8",
            )
            findings = _find_layout_contract_issues(html_path)
        self.assertFalse(any("holder" in finding for finding in findings), findings)

    def test_deterministic_critic_requires_regions_inside_module_bounds(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-html-module-bounds-") as temporary:
            html_path = Path(temporary) / "layout.html"
            html_path.write_text(
                """<!doctype html><style>
.module{position:absolute;left:20px;top:20px;width:700px;height:200px}
.title{position:absolute;left:0;top:0;width:700px;height:204px}
.visual{left:760px;top:20px;width:700px;height:500px}
</style><div class="module" data-module-id="title_module"><h1 class="title" data-region-id="title" data-content-key="title" data-role="title">Title</h1></div>
<div class="module visual" data-module-id="visual_module" data-region-id="visual" data-role="visual" data-generation-group="visual_group_visual"></div>""",
                encoding="utf-8",
            )
            findings = _find_layout_contract_issues(html_path)
        self.assertIn("registered regions extend outside their modules: title", findings)

    def test_deterministic_critic_rejects_unregistered_but_allows_inline_emphasis(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-html-text-contract-") as temporary:
            html_path = Path(temporary) / "layout.html"
            html_path.write_text(
                """<!doctype html><style>
.copy{position:absolute;width:700px;height:300px}.visual{left:760px;width:700px;height:500px}
</style><section class="slide"><div class="copy" data-module-id="copy"><p class="eyebrow">Unregistered label</p><p data-region-id="copy_text" data-content-key="text_01" data-role="body">A <strong>bold phrase</strong>.</p></div>
<div class="copy visual" data-module-id="visual" data-region-id="visual_a" data-role="visual" data-generation-group="visual_group_visual_a"></div></section>""",
                encoding="utf-8",
            )
            findings = _find_layout_contract_issues(html_path)
        self.assertIn(
            "unregistered visible text is not carried into assembly: Unregistered label",
            findings,
        )
        self.assertFalse(any("inline emphasis" in finding for finding in findings))

    def test_rework_guard_allows_recomposition_but_rejects_removed_holder(self) -> None:
        original = """<!doctype html><html><style>
.slide{position:relative;width:1536px;height:864px}.visual{position:absolute;left:80px;top:200px;width:500px;height:400px}
.holder{position:absolute;left:850px;top:250px;width:500px;height:180px;background:#eee;border-left:4px solid #222}
</style><section class="slide"><div class="visual" data-module-id="module_visual" data-region-id="visual_a" data-role="visual"></div>
<div class="holder" data-module-id="module_copy" data-holder-id="holder_body"><p data-content-key="text_01" data-role="body">Framed copy</p></div></section></html>"""
        flipped = original.replace("left:80px", "left:950px")
        accepted, reason = _rework_preserves_structure(
            original_html=original, reworked_html=flipped
        )
        self.assertTrue(accepted, reason)
        changed_copy = flipped.replace("Framed copy", "Clearer framed copy")
        accepted, reason = _rework_preserves_structure(
            original_html=original, reworked_html=changed_copy
        )
        self.assertFalse(accepted)
        self.assertIn("visible text", reason)
        removed = original.replace(' data-holder-id="holder_body"', "")
        accepted, reason = _rework_preserves_structure(
            original_html=original, reworked_html=removed
        )
        self.assertFalse(accepted)
        self.assertIn("removed holder surfaces", reason)

    def test_rework_guard_allows_restoring_a_degenerate_visual(self) -> None:
        original = """<!doctype html><html><style>
.slide{position:relative;width:1536px;height:864px}.visual{position:absolute;width:300px;height:232px}
.current{left:610px;top:217px}.action{left:914px;top:217px}.predicted{left:0;top:0;width:4px;height:4px}
</style><section class="slide">
<div class="visual current" data-module-id="module_current" data-region-id="visual_current" data-role="visual" data-visual-brief="Current" data-generation-group="visual_group_visual_current"></div>
<div class="visual action" data-module-id="module_action" data-region-id="visual_action" data-role="visual" data-visual-brief="Action" data-generation-group="visual_group_visual_action"></div>
<div class="visual predicted" data-module-id="module_predicted" data-region-id="visual_predicted" data-role="visual" data-visual-brief="Predicted" data-generation-group="visual_group_visual_predicted"></div>
</section></html>"""
        repaired = original.replace(
            ".predicted{left:0;top:0;width:4px;height:4px}",
            ".predicted{left:1218px;top:217px}",
        )
        accepted, reason = _rework_preserves_structure(
            original_html=original, reworked_html=repaired
        )
        self.assertTrue(accepted, reason)

    def test_rework_guard_allows_diffusion_inspired_visual_recomposition(self) -> None:
        original = """<!doctype html><html><style>
.slide{position:relative;width:1536px;height:864px}.module{position:absolute}
.title{left:60px;top:40px;width:1416px;height:90px}.body{left:60px;top:160px;width:520px;height:600px}
.visual{left:620px;top:160px;width:856px;height:600px}
</style><section class="slide">
<h1 class="module title" data-module-id="title_module" data-region-id="title" data-content-key="title" data-role="title">Title</h1>
<p class="module body" data-module-id="body_module" data-region-id="body" data-content-key="text_01" data-role="body">Body</p>
<div class="module visual" data-module-id="visual_module" data-region-id="visual_hero" data-role="visual"></div>
</section></html>"""
        reworked = """<!doctype html><html><style>
.slide{position:relative;width:1536px;height:864px}.module{position:absolute}
.title{left:60px;top:40px;width:1416px;height:90px}.body{left:60px;top:160px;width:420px;height:600px}
.visual-a{left:520px;top:160px;width:440px;height:600px}.visual-b{left:996px;top:160px;width:480px;height:600px}
</style><section class="slide">
<h1 class="module title" data-module-id="title_module" data-region-id="title" data-content-key="title" data-role="title">Title</h1>
<p class="module body" data-module-id="body_module" data-region-id="body" data-content-key="text_01" data-role="body">Body</p>
<div class="module visual-a" data-module-id="visual_a_module" data-region-id="visual_stage_a" data-role="visual"></div>
<div class="module visual-b" data-module-id="visual_b_module" data-region-id="visual_stage_b" data-role="visual"></div>
</section></html>"""
        accepted, reason = _rework_preserves_structure(
            original_html=original, reworked_html=reworked
        )
        self.assertTrue(accepted, reason)

    def test_text_rework_is_block_scoped_presentation_friendly_and_geometry_safe(self) -> None:
        source = (
            "Explore or exploit: exploration discovers uncertain options; "
            "exploitation uses known rewards."
        )
        current_blocks = {
            "text_01": {
                "content_key": "text_01",
                "region_id": "body",
                "text": source,
                "bbox_px": [80, 180, 560, 260],
                "font_size_px": 24,
            }
        }
        raw = json.dumps(
            {
                "slide_id": "slide_001",
                "blocks": [
                    {
                        "content_key": "text_01",
                        "preserved_meaning": True,
                        "parts": [
                            {
                                "kind": "lead",
                                "runs": [
                                    {"text": "Explore or exploit", "bold": True},
                                    {"text": ":", "bold": False},
                                ],
                            },
                            {
                                "kind": "bullet",
                                "runs": [
                                    {"text": "exploration", "bold": True},
                                    {"text": "discovers uncertain options;", "bold": False},
                                ],
                            },
                            {
                                "kind": "bullet",
                                "runs": [
                                    {"text": "exploitation", "bold": True},
                                    {"text": "uses known rewards.", "bold": False},
                                ],
                            },
                        ],
                    }
                ],
            }
        )
        plan = parse_text_rework_response(
            raw,
            slide_id="slide_001",
            allowed_blocks=current_blocks,
        )
        prompt = build_text_rework_prompt(
            slide_id="slide_001",
            text_tree={"slide_id": "slide_001", "text_blocks": []},
            current_blocks=list(current_blocks.values()),
        )
        self.assertIn("Keep every fact, number, entity", prompt)
        self.assertIn("Never bold a whole multi-sentence block", prompt)
        with tempfile.TemporaryDirectory(prefix="rl4ppt-text-rework-") as temporary:
            html_path = Path(temporary) / "layout.html"
            html_path.write_text(
                f"""<!doctype html><html><head><style>
.slide{{position:relative;width:1536px;height:864px}}.module{{position:absolute}}
.title{{left:80px;top:40px;width:900px;height:80px;font-size:48px}}
.copy{{left:80px;top:180px;width:560px;height:260px;font-size:24px;line-height:1.25}}
.visual{{left:760px;top:180px;width:600px;height:420px}}
</style></head><body><section class="slide" data-slide-id="slide_001">
<h1 class="module title" data-module-id="title_module" data-region-id="title" data-content-key="title" data-role="title">Choice</h1>
<p class="module copy" data-module-id="copy_module" data-region-id="body" data-content-key="text_01" data-role="body">{source}</p>
<div class="module visual" data-module-id="visual_module" data-region-id="visual" data-role="visual"></div>
</section></body></html>""",
                encoding="utf-8",
            )
            report = apply_text_rework_plan(html_path=html_path, plan=plan)
            document = html_path.read_text(encoding="utf-8")
        self.assertEqual(report["status"], "accepted")
        self.assertEqual(report["accepted_content_keys"], ["text_01"])
        self.assertIn('data-text-reworked="true"', document)
        self.assertIn("<strong>exploration</strong>", document)
        self.assertIn("<strong>Explore or exploit</strong><span>:</span>", document)
        self.assertNotIn("<span> :</span>", document)
        self.assertIn('data-text-part="bullet"', document)

    def test_text_rework_rejects_changed_numbers_and_expansion(self) -> None:
        allowed = {
            "text_01": {
                "content_key": "text_01",
                "text": "Deliver 100 MW for four hours.",
            }
        }
        changed = json.dumps(
            {
                "slide_id": "slide_001",
                "blocks": [
                    {
                        "content_key": "text_01",
                        "preserved_meaning": True,
                        "parts": [
                            {
                                "kind": "body",
                                "runs": [
                                    {"text": "Deliver 200 MW for four hours.", "bold": False}
                                ],
                            }
                        ],
                    }
                ],
            }
        )
        with self.assertRaisesRegex(ValueError, "numeric facts"):
            parse_text_rework_response(
                changed, slide_id="slide_001", allowed_blocks=allowed
            )

    def test_deck_typography_profile_clamps_once_and_changes_only_font_size(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-typography-") as temporary:
            root = Path(temporary)
            html_by_slide = {}
            for index, title in enumerate(("Short title", "A much longer shared title"), start=1):
                path = root / f"slide_{index}.html"
                path.write_text(
                    f"""<!doctype html><html><head><style>
.slide{{position:relative;width:1536px;height:864px}}.module{{position:absolute}}
.title{{left:60px;top:40px;width:900px;height:72px;font-size:{36 + index}px;font-weight:700}}
.subtitle{{left:60px;top:120px;width:900px;height:44px;font-size:{17 + index}px}}
.visual{{left:800px;top:220px;width:500px;height:300px}}
</style></head><body><section class="slide">
<h1 class="module title" data-module-id="title_{index}" data-region-id="title_{index}" data-content-key="title" data-role="title">{title}</h1>
<p class="module subtitle" data-module-id="subtitle_{index}" data-region-id="subtitle_{index}" data-content-key="subtitle" data-role="subtitle">Shared hierarchy</p>
<div class="module visual" data-module-id="visual_{index}" data-region-id="visual_{index}" data-role="visual"></div>
</section></body></html>""",
                    encoding="utf-8",
                )
                html_by_slide[f"slide_{index:03d}"] = path
            metrics = inspect_deck_typography(html_by_slide)
            prompt = build_typography_profile_prompt(metrics)
            profile = parse_typography_profile(
                json.dumps(
                    {"title_font_size_px": 999, "subtitle_font_size_px": 999}
                ),
                metrics,
            )
            report = apply_deck_typography_profile(
                html_by_slide=html_by_slide, profile=profile
            )
            documents = [path.read_text(encoding="utf-8") for path in html_by_slide.values()]
        self.assertIn("Change font size only", prompt)
        self.assertEqual(report["status"], "accepted")
        self.assertEqual(
            profile["title_font_size_px"], metrics["max_safe_shared_px"]["title"]
        )
        self.assertTrue(all('data-deck-typography-profile="true"' in doc for doc in documents))
        self.assertTrue(all("font-weight:700" in doc for doc in documents))

    def test_pipeline_rejects_rework_that_flips_layout_and_keeps_final_preview_current(self) -> None:
        config = load_config(ROOT)
        initial = self._planning_html(visual_left=True, holder=True)
        reworked = self._planning_html(visual_left=False, holder=False)
        client = FlippingReworkClient(initial, reworked)
        with tempfile.TemporaryDirectory(prefix="rl4ppt-rework-guard-") as temporary:
            root = Path(temporary)
            (root / "step2").mkdir()
            (root / "text_generation").mkdir()
            (root / "expression_logic/slides/slide_001").mkdir(parents=True)
            (root / "diffusion/skeletons").mkdir(parents=True)
            (root / "step2/deck_plan.json").write_text(
                json.dumps(
                    {
                        "topic": "Feedback",
                        "audience": "students",
                        "deck_goal": "Explain feedback.",
                        "style": "editorial",
                        "slides": [
                            {
                                "slide_id": "slide_001",
                                "working_title": "Preserve direction",
                                "rough_message": "Keep the visual on the left.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "text_generation/text_plan.json").write_text(
                json.dumps(
                    {
                        "slides": [
                            {
                                "slide_id": "slide_001",
                                "title": "Preserve direction",
                                "subtitle": None,
                                "text_01": "Framed explanation.",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "text_generation/parsed_text_plan.json").write_text(
                json.dumps(
                    {
                        "slides": [
                            {
                                "slide_id": "slide_001",
                                "title": "Preserve direction",
                                "subtitle": None,
                                "text_blocks": [
                                    {
                                        "text_id": "text_01",
                                        "format": "paragraph",
                                        "text": "Framed explanation.",
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "expression_logic/run_manifest.json").write_text(
                json.dumps(
                    {
                        "stage": "deck_level_expression_logic_planning",
                        "status": "succeeded",
                    }
                ),
                encoding="utf-8",
            )
            expression_plan = {
                "slides": [
                    {
                        "slide_id": "slide_001",
                        "expression_logic": (
                            "Keep the explanatory visual on the left and attach "
                            "text_01 on the right."
                        ),
                    }
                ]
            }
            (root / "expression_logic/expression_logic_plan.json").write_text(
                json.dumps(expression_plan), encoding="utf-8"
            )
            (root / "expression_logic/slides/slide_001/expression_logic.json").write_text(
                json.dumps(expression_plan["slides"][0]), encoding="utf-8"
            )
            Image.new("RGB", (1024, 576), "white").save(
                root / "diffusion/skeletons/slide_001.png"
            )
            manifest = run_html_planning(
                expression_logic_dir=root / "expression_logic",
                diffusion_dir=root / "diffusion",
                step2_dir=root / "step2",
                text_generation_dir=root / "text_generation",
                output_dir=root / "html_planning",
                slide_ids=["slide_001"],
                config=config,
                mock_vlm=False,
                client_factory=lambda: client,
            )
            slide_dir = root / "html_planning/slides/slide_001"
            final_html = (slide_dir / "layout.html").read_text()
            expected_preview = root / "expected.png"
            _render_critic_preview(slide_dir / "layout.html", expected_preview)
            with Image.open(expected_preview) as expected, Image.open(
                slide_dir / "critic_preview.png"
            ) as actual:
                preview_difference = ImageChops.difference(expected, actual).getbbox()

        record = manifest["slides"][0]["critic"]
        self.assertTrue(record["rework_attempted"])
        self.assertFalse(record["rework_accepted"])
        self.assertTrue(any(reason in record["rework_rejection_reason"] for reason in ("holder", "typography")))
        self.assertEqual(
            client.initial_vlm_html_kwargs["image_path"],
            root / "diffusion/skeletons/slide_001.png",
        )
        original_structure = _inspect_html_structure(initial)
        final_structure = _inspect_html_structure(final_html)
        for key in ('content_by_key', 'visual_bboxes', 'holder_ids', 'module_ids'):
            self.assertEqual(final_structure[key], original_structure[key])
        self.assertIsNone(preview_difference)

    @staticmethod
    def _planning_html(*, visual_left: bool, holder: bool) -> str:
        visual_x = 70 if visual_left else 930
        text_x = 900 if visual_left else 80
        surface = "background:#eee;border-left:6px solid #333;" if holder else ""
        holder_attr = ' data-holder-id="holder_body"' if holder else ""
        surface_attr = "holder" if holder else "plain"
        return f'''<!doctype html><html><head><style>
html,body{{margin:0}}.slide{{position:relative;width:1536px;height:864px}}
.visual{{position:absolute;left:{visual_x}px;top:180px;width:520px;height:500px;border:2px solid #333}}
.copy{{position:absolute;left:{text_x}px;top:250px;width:500px;height:180px;{surface}}}
</style></head><body><section class="slide" data-slide-id="slide_001">
<h1 data-module-id="module_title" data-region-id="title" data-content-key="title" data-role="title" data-text-surface="plain" style="position:absolute;left:100px;top:40px;width:900px;height:70px;font-size:40px;font-weight:700">Preserve direction</h1>
<div class="visual" data-module-id="module_visual" data-region-id="visual_a" data-role="visual" data-generation-group="visual_group_visual_a" data-visual-brief="Show feedback."></div>
<div class="copy" data-module-id="module_copy"{holder_attr}><p data-region-id="body" data-content-key="text_01" data-role="body" data-text-surface="{surface_attr}" style="font-size:20px">Framed explanation.</p></div>
</section></body></html>'''

    def test_html_extractor_warns_but_keeps_tolerated_markup(self) -> None:
        warnings: list[str] = []
        document = extract_html_document(
            '<!doctype html><html><body><img src="https://example.com/x.png"></body></html>',
            warnings,
        )
        self.assertIn("<img", document)
        self.assertTrue(any("embedded media" in warning for warning in warnings))
        inline = extract_html_document(
            '<!doctype html><html><body><svg><path marker-end="url(#arrowBlue)"/></svg></body></html>'
        )
        self.assertIn("url(#arrowBlue)", inline)
        css_warnings: list[str] = []
        sanitized = extract_html_document(
            '<!doctype html><style>.step::before{content:"01"}</style><div class="step"></div>',
            css_warnings,
        )
        self.assertIn('content:""', sanitized)
        self.assertTrue(any("removed it" in warning for warning in css_warnings))

    @staticmethod
    def _raw_region(
        *, region_id: str, role: str, content_key: str, bbox: list[int], text: str
    ) -> dict:
        return {
            "region_id": region_id,
            "role": role,
            "content_key": content_key,
            "text_surface": "plain",
            "visual_brief": None,
            "equivalent_group": None,
            "bbox_px": bbox,
            "text_content": text,
            "inner_text": text,
            "client_width": bbox[2],
            "client_height": bbox[3],
            "scroll_width": bbox[2],
            "scroll_height": bbox[3],
            "overflow": False,
            "display": "block",
            "visibility": "visible",
            "z_index": "auto",
            "holder": None,
        }

    @staticmethod
    def _write_compile_fixture(root: Path) -> None:
        step2 = root / "step2"
        text_generation = root / "text_generation"
        slide_dir = root / "html_planning/slides/slide_001"
        for directory in (step2, text_generation, slide_dir):
            directory.mkdir(parents=True)
        (step2 / "deck_plan.json").write_text(
            json.dumps(
                {
                    "topic": "Feedback control",
                    "audience": "students",
                    "deck_goal": "Explain feedback.",
                    "style": "clean editorial illustration",
                    "slides": [
                        {
                            "slide_id": "slide_001",
                            "working_title": "A feedback loop",
                            "rough_message": "Observe and update.",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (text_generation / "text_plan.json").write_text(
            json.dumps(
                {
                    "slides": [
                        {
                            "slide_id": "slide_001",
                            "title": "A feedback loop",
                            "subtitle": None,
                            "text_01": "Observe the outcome.",
                            "text_02": "Update the next choice.",
                            "text_03": "Framed conclusion.",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (slide_dir / "layout.html").write_text(
            """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;width:1536px;height:864px} .slide{position:relative;width:1536px;height:864px;overflow:hidden}
[data-module-id],[data-region-id]{position:absolute;box-sizing:border-box}.closed{left:60px;top:160px;width:420px;height:170px;border:3px solid #222;padding:20px}
.divider{left:60px;top:390px;width:420px;height:120px;padding:20px}
.tinted{left:60px;top:550px;width:420px;height:110px;border-left:4px solid #222;background:#eee;padding:20px}
</style></head><body><section class="slide" data-slide-id="slide_001">
<div data-module-id="module_title" style="left:60px;top:40px;width:700px;height:80px"><div data-region-id="title_001" data-content-key="title" data-role="title" data-text-surface="plain" style="inset:0;text-align:center;font-size:48px;font-weight:700">A feedback loop</div></div>
<div class="closed" data-module-id="module_body_1" data-holder-id="holder_body_001"><div data-region-id="body_001" data-content-key="text_01" data-role="body" data-text-surface="holder" style="position:static;width:360px;height:100px;font-size:30px;font-weight:700">Observe, then learn.</div></div>
<div class="divider" data-module-id="module_body_2"><div data-region-id="body_002" data-content-key="text_02" data-role="body" data-text-surface="plain" style="position:static;width:360px;height:80px">Update the next choice.</div></div>
<div class="tinted" data-module-id="module_body_3" data-holder-id="holder_body_003"><div data-region-id="body_003" data-content-key="text_03" data-role="body" data-text-surface="holder" style="position:static;width:360px;height:70px">Framed conclusion.</div></div>
<div data-module-id="module_visuals" style="left:600px;top:160px;width:720px;height:500px">
  <div data-region-id="visual_a" data-role="visual" data-generation-group="visual_group_visual_a" data-visual-brief="Show observation." style="left:0;top:0;width:360px;height:240px"></div>
  <div data-region-id="visual_b" data-role="visual" data-generation-group="visual_group_visual_b" data-visual-brief="Show update." style="left:400px;top:280px;width:320px;height:220px"></div>
</div>
</section></body></html>""",
            encoding="utf-8",
        )
        Image.new("RGB", (1536, 864), "white").save(step2 / "canonical_background.png")


if __name__ == "__main__":
    unittest.main()
