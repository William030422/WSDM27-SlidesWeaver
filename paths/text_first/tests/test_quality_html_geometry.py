from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from playwright.sync_api import sync_playwright

from integration.html_compile import (
    _compile_slide,
    _normalize_bbox,
    _normalize_typography,
    _validate_regions,
)
from integration.html_planning import (
    _find_layout_contract_issues,
    _rework_preserves_structure,
    prepare_html_layout,
)


class HtmlGeometryQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="rl4ppt-html-quality-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.html_path = self.root / "layout.html"

    def _compile(self, document: str, source: dict) -> dict:
        self.html_path.write_text(document, encoding="utf-8")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1536, "height": 864})
                result = _compile_slide(
                    page=page, slide_id="slide_001", html_path=self.html_path,
                    source_text=source, visual_groups_path=self.root / "groups.json",
                    output_dir=self.root / "compiled", deck_font="Noto Sans",
                )
            finally:
                browser.close()
        self.assertEqual(result["status"], "succeeded")
        return json.loads(Path(result["html_layout"]).read_text(encoding="utf-8"))

    def test_font_uses_same_canvas_to_page_scale_as_geometry(self) -> None:
        typography = _normalize_typography({"font_size_px": 32})
        self.assertEqual(typography["font_size_pt"], 20)

    def test_fractional_untransformed_height_does_not_rescale_font(self) -> None:
        layout = self._compile(
            '''<!doctype html><html><head><style>html,body{margin:0}.slide{position:relative;width:1536px;height:864px}.copy{position:absolute;left:60px;top:40px;width:900px;height:64.5px;box-sizing:border-box;font:43px/1.1 "Noto Sans"}</style></head><body><section class="slide" data-slide-id="slide_001"><div class="copy" data-module-id="t" data-region-id="title" data-role="title" data-content-key="title">Shared title</div></section></body></html>''',
            {"slide_id":"slide_001","title":"Shared title","subtitle":None},
        )
        self.assertEqual(layout['text_regions'][0]['typography']['font_size_px'], 43)
        self.assertEqual(layout['text_regions'][0]['typography']['font_size_pt'], 26.875)

    def test_content_box_keeps_asymmetric_padding_and_border(self) -> None:
        layout = self._compile(
            '''<!doctype html><html><head><style>
html,body{margin:0}.slide{position:relative;width:1536px;height:864px}
.copy{position:absolute;left:60px;top:60px;width:400px;height:160px;box-sizing:border-box;
padding:10px 20px 30px 40px;border:2px solid black;font:32px/1.1 Arial}
</style></head><body><section class="slide" data-slide-id="slide_001">
<div class="copy" data-module-id="title_module" data-region-id="title" data-role="title" data-content-key="title">MMMM</div>
</section></body></html>''',
            {"slide_id": "slide_001", "title": "MMMM", "subtitle": None},
        )
        title = layout["text_regions"][0]
        self.assertEqual(title["bbox_px"], [60, 60, 400, 160])
        self.assertEqual(title["text_bbox_px"], [102, 72, 336, 116])
        self.assertEqual(title["text_box_insets_px"], [12, 22, 32, 42])
        self.assertEqual(title["typography"]["font_size_pt"], 20)
        self.assertEqual(title["typography"]["font_family"], "Noto Sans")
        self.assertEqual(title["vertical_alignment"], "top")

    def test_outside_module_moves_as_one_unit_without_collapsing_visual(self) -> None:
        document = self._layout_with_visual(1600)
        self.html_path.write_text(document, encoding="utf-8")
        before = _find_layout_contract_issues(self.html_path)
        self.assertTrue(any("module visual_module extends outside the canvas" in issue for issue in before))
        source = {"slide_id": "slide_001", "title": "Title", "subtitle": None, "text_01": "Body"}
        layout = self._compile(document, source)
        visual = layout["visual_regions"][0]
        self.assertEqual(visual["bbox_px"][2:], [800, 640])
        self.assertLessEqual(visual["bbox_px"][0] + visual["bbox_px"][2], 1536)
        self.assertEqual(visual["raw_bbox_px"], [1600, 164, 800, 640])
        self.assertTrue(layout["geometry_repairs"])
        self.assertEqual(layout["text_regions"][0]["bbox_px"], [60, 40, 700, 80])
        self.assertEqual(layout["text_regions"][1]["bbox_px"], [60, 200, 400, 400])
        after = _find_layout_contract_issues(self.html_path)
        self.assertFalse(any("outside the canvas" in issue for issue in after))

    def test_bbox_recovery_preserves_visible_area_instead_of_one_pixel_strip(self) -> None:
        recovered, warning = _normalize_bbox([1600, 164, 800, 640])
        self.assertEqual(recovered, [736, 164, 800, 640])
        self.assertIsNotNone(warning)

    def test_oversized_unit_scales_font_and_padding_with_its_geometry(self) -> None:
        layout = self._compile(
            '''<!doctype html><html><head><style>
html,body{margin:0}.slide{position:relative;width:1536px;height:864px}
</style></head><body><section class="slide" data-slide-id="slide_001">
<div data-module-id="oversized" style="position:absolute;left:100px;top:200px;width:2000px;height:600px">
<div data-region-id="title" data-role="title" data-content-key="title" style="position:absolute;left:20px;top:20px;width:800px;height:200px;box-sizing:border-box;padding:10px;border:2px solid black;font-size:32px;line-height:1.1">MMMM</div>
</div></section></body></html>''',
            {"slide_id": "slide_001", "title": "MMMM", "subtitle": None},
        )
        title = layout["text_regions"][0]
        self.assertAlmostEqual(title["typography"]["font_size_px"], 32 * 1536 / 2000, places=2)
        self.assertAlmostEqual(title["typography"]["font_size_pt"], 32 * 1536 / 2000 * 0.625, places=2)
        for inset in title["text_box_insets_px"]:
            self.assertAlmostEqual(inset, 12 * 1536 / 2000, places=2)
        self.assertEqual(title["bbox_px"][2:], [614, 154])
        self.assertAlmostEqual(title["typography"]["line_height_ratio"], 1.1, places=2)

    def test_missing_empty_and_duplicate_text_are_recovered_not_fatal(self) -> None:
        regions, warnings = _validate_regions(
            {"extra_text": [], "regions": [
                {"region_id": "same", "content_key": "title", "role": "title", "bbox_px": [48, 48, 600, 120], "inner_text": ""},
                {"region_id": "same", "content_key": None, "role": "visual", "bbox_px": [800, 200, 600, 400]},
            ]},
            {"title": "Frozen title", "subtitle": None, "text_01": "Cost -10%; does not improve."},
        )
        by_key = {item["content_key"]: item for item in regions if item["content_key"]}
        self.assertEqual(by_key["title"]["display_text"], "Frozen title")
        self.assertEqual(by_key["text_01"]["display_text"], "Cost -10%; does not improve.")
        self.assertEqual(len({item["region_id"] for item in regions}), len(regions))
        self.assertTrue(any("missing frozen content" in item for item in warnings))
        self.assertTrue(any("duplicate" in item for item in warnings))

    def test_altered_copy_is_locally_restored_with_original_geometry(self) -> None:
        source = {"slide_id": "slide_001", "title": "Title", "subtitle": None, "text_01": "Cost -10%; does not improve."}
        document = self._layout_with_visual(736).replace(
            '>Body</div>', '>Cost +10; does improve.</div>'
        )
        layout = self._compile(document, source)
        body = next(item for item in layout["text_regions"] if item["content_key"] == "text_01")
        self.assertEqual(body["display_text"], source["text_01"])
        self.assertEqual(body["bbox_px"], [60, 200, 400, 400])

    def test_geography_guard_rejects_inline_emphasis_change(self) -> None:
        original = self._layout_with_visual(736)
        altered = original.replace('>Body</div>', '><strong>Body</strong></div>')
        preserved, reason = _rework_preserves_structure(original_html=original, reworked_html=altered)
        self.assertFalse(preserved)
        self.assertIn("inline", reason)

    def test_preparation_is_idempotent_for_valid_geometry(self) -> None:
        self.html_path.write_text(self._layout_with_visual(736), encoding="utf-8")
        source = {"slide_id": "slide_001", "title": "Title", "subtitle": None, "text_01": "Body"}
        first = prepare_html_layout(self.html_path, source_text=source)
        document = self.html_path.read_text(encoding="utf-8")
        second = prepare_html_layout(self.html_path, source_text=source)
        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(self.html_path.read_text(encoding="utf-8"), document)

    def test_existing_center_origin_scale_is_repaired_once_with_sibling_unchanged(self) -> None:
        self.html_path.write_text(
            '''<!doctype html><html><head><style>
html,body{margin:0}.slide{position:relative;width:1536px;height:864px}
</style></head><body><section class="slide">
<div data-module-id="sibling" style="position:absolute;left:20px;top:20px;width:200px;height:100px"></div>
<div data-module-id="scaled" style="position:absolute;left:1600px;top:100px;width:800px;height:320px;transform:scale(2)">
<div data-region-id="title" data-role="title" data-content-key="title" style="width:800px;height:320px;font-size:32px">Title</div>
</div></section></body></html>''', encoding="utf-8",
        )
        source = {"title": "Title", "subtitle": None}
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1536, "height": 864})
                first = prepare_html_layout(self.html_path, source_text=source, page=page)
                page.goto(self.html_path.as_uri())
                rect = page.locator('[data-module-id="scaled"]').bounding_box()
                sibling = page.locator('[data-module-id="sibling"]').bounding_box()
                self.assertTrue(first)
                self.assertGreaterEqual(rect["x"], -0.1)
                self.assertGreaterEqual(rect["y"], -0.1)
                self.assertLessEqual(rect["x"] + rect["width"], 1536.1)
                self.assertLessEqual(rect["y"] + rect["height"], 864.1)
                self.assertEqual(sibling, {"x": 20, "y": 20, "width": 200, "height": 100})
                document = self.html_path.read_text(encoding="utf-8")
                self.assertEqual(prepare_html_layout(self.html_path, source_text=source, page=page), [])
                self.assertEqual(self.html_path.read_text(encoding="utf-8"), document)
            finally:
                browser.close()

    @staticmethod
    def _layout_with_visual(visual_x: int) -> str:
        return f'''<!doctype html><html><head><style>
html,body{{margin:0}}.slide{{position:relative;width:1536px;height:864px}}
[data-module-id]{{position:absolute;box-sizing:border-box}}
</style></head><body><section class="slide" data-slide-id="slide_001">
<div data-module-id="title_module" data-region-id="title" data-role="title" data-content-key="title" style="left:60px;top:40px;width:700px;height:80px;font-size:40px">Title</div>
<div data-module-id="body_module" data-holder-id="holder_body" data-region-id="body" data-role="body" data-content-key="text_01" data-text-surface="holder" style="left:60px;top:200px;width:400px;height:400px;border:2px solid #777;background:transparent">Body</div>
<div data-module-id="visual_module" style="left:{visual_x}px;top:164px;width:800px;height:640px">
<div data-region-id="visual" data-role="visual" data-generation-group="visual_group_visual" data-visual-brief="An explanatory scene" style="position:absolute;inset:0"></div>
</div></section></body></html>'''


if __name__ == "__main__":
    unittest.main()
