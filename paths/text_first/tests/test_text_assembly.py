from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

from PIL import Image, ImageDraw

from integration.contrast import apply_background_aware_text_colors
from integration.text_assembly import (
    _compile_deck,
    _font_for_role,
    _usable_text_safe_rect,
    _validate_editable_raster_objects,
)


class TextAssemblyTests(unittest.TestCase):
    def test_contrast_solver_samples_each_block_on_the_final_composite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-contrast-") as temporary:
            root = Path(temporary)
            background = root / "background.png"
            holder_path = root / "holder.png"
            Image.new("RGB", (1536, 864), (9, 18, 28)).save(background)
            Image.new("RGBA", (500, 220), (248, 248, 245, 255)).save(holder_path)
            plan = {
                "slide_id": "slide_001",
                "canvas_px": {"width": 1536, "height": 864},
                "background_path": str(background),
                "raster_layers": [
                    {
                        "layer_id": "holder_body",
                        "kind": "holder",
                        "bbox_px": [800, 260, 500, 220],
                        "path": str(holder_path),
                    }
                ],
                "elements": [
                    {
                        "region_id": "direct_body",
                        "content_key": "text_01",
                        "role": "body",
                        "bbox_px": [80, 220, 500, 180],
                        "holder_binding": None,
                        "native_holder": None,
                        "font": {"color": "18232D"},
                    },
                    {
                        "region_id": "holder_body",
                        "content_key": "text_02",
                        "role": "body",
                        "bbox_px": [800, 260, 500, 220],
                        "holder_binding": {
                            "layer_id": "holder_body",
                            "holder_bbox_px": [800, 260, 500, 220],
                            "text_safe_rect": {
                                "x": 0.08,
                                "y": 0.12,
                                "width": 0.84,
                                "height": 0.76,
                            },
                        },
                        "native_holder": None,
                        "font": {"color": "18232D"},
                    },
                ],
            }
            report = apply_background_aware_text_colors(plan)

        by_key = {item["content_key"]: item for item in report["blocks"]}
        self.assertEqual(report["status"], "strict_pass")
        self.assertNotEqual(by_key["text_01"]["final_font_color"], "18232D")
        self.assertEqual(by_key["text_02"]["final_font_color"], "18232D")
        self.assertLess(by_key["text_01"]["background_rgb"][0], 40)
        self.assertGreater(by_key["text_02"]["background_rgb"][0], 240)

    def test_body_font_target_is_15pt(self) -> None:
        self.assertEqual(
            _font_for_role("body", SimpleNamespace(deck_font="Noto Sans")),
            {
                "face": "Noto Sans",
                "size_pt": 15,
                "minimum_pt": 14,
                "weight": "regular",
            },
        )

    def test_near_full_width_holder_rejects_degenerate_low_safe_rect(self) -> None:
        self.assertFalse(
            _usable_text_safe_rect(
                {
                    "x": 0.007184,
                    "y": 0.022727,
                    "width": 0.454741,
                    "height": 0.102273,
                }
            )
        )
        self.assertTrue(
            _usable_text_safe_rect(
                {"x": 0.04, "y": 0.08, "width": 0.92, "height": 0.84}
            )
        )

    def test_background_and_generated_layers_remain_separate_pptx_objects(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-editable-raster-") as temporary:
            root = Path(temporary)
            background = root / "background.png"
            visual = root / "visual_a.png"
            holder = root / "holder_title.png"
            Image.new("RGB", (1536, 864), (235, 238, 240)).save(background)
            Image.new("RGBA", (360, 220), (20, 120, 190, 255)).save(visual)
            holder_image = Image.new("RGBA", (500, 100), (0, 0, 0, 0))
            ImageDraw.Draw(holder_image).rectangle(
                (2, 2, 497, 97), fill=(250, 250, 250, 230), outline=(30, 30, 30, 255)
            )
            holder_image.save(holder)
            plan = {
                "schema_version": "3.0",
                "slide_id": "slide_001",
                "canvas_px": {"width": 1536, "height": 864},
                "background_path": str(background),
                "raster_layers": [
                    {
                        "layer_id": "visual_a",
                        "kind": "visual",
                        "bbox_px": [600, 260, 360, 220],
                        "path": str(visual),
                    },
                    {
                        "layer_id": "holder_region_title",
                        "kind": "holder",
                        "bbox_px": [518, 60, 500, 100],
                        "path": str(holder),
                    },
                ],
                "groups": [],
                "elements": [],
            }
            plan_path = root / "composition_plan.json"
            compilation_path = root / "deck_compilation.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            compilation_path.write_text(
                json.dumps(
                    {
                        "schema_version": "2.0",
                        "slides": [
                            {"slide_id": "slide_001", "plan": str(plan_path)}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            pptx_path = root / "deck.pptx"
            _compile_deck(compilation_path, pptx_path, root / "text_fit.json")

            report = _validate_editable_raster_objects(
                pptx_path,
                [
                    {
                        "slide_id": "slide_001",
                        "object_names": [
                            "background:slide_001",
                            "raster:visual_a",
                            "raster:holder_region_title",
                        ],
                    }
                ],
            )
            with ZipFile(pptx_path) as archive:
                media = [
                    name
                    for name in archive.namelist()
                    if name.startswith("ppt/media/") and not name.endswith("/")
                ]

        self.assertEqual(report["status"], "strict_pass")
        self.assertEqual(report["slides"][0]["expected_object_count"], 3)
        self.assertEqual(report["slides"][0]["picture_object_count"], 3)
        self.assertEqual(len(media), 3)

    def test_rich_text_lead_bullets_and_inline_bold_remain_editable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-rich-text-") as temporary:
            root = Path(temporary)
            background = root / "background.png"
            Image.new("RGB", (1536, 864), "white").save(background)
            plan = {
                "schema_version": "3.0",
                "slide_id": "slide_001",
                "canvas_px": {"width": 1536, "height": 864},
                "background_path": str(background),
                "raster_layers": [],
                "groups": [],
                "elements": [
                    {
                        "region_id": "body",
                        "content_key": "text_01",
                        "role": "body",
                        "bbox_px": [120, 120, 800, 420],
                        "holder_binding": None,
                        "native_holder": None,
                        "text": "Explore or exploit",
                        "bullet_items": [],
                        "rich_text": [
                            {
                                "kind": "lead",
                                "runs": [
                                    {
                                        "text": "Explore or exploit",
                                        "bold": True,
                                        "italic": False,
                                        "font_scale": 1.08,
                                    }
                                ],
                            },
                            {
                                "kind": "bullet",
                                "runs": [
                                    {
                                        "text": "Explore",
                                        "bold": True,
                                        "italic": False,
                                        "font_scale": 1,
                                    },
                                    {
                                        "text": " uncertain options",
                                        "bold": False,
                                        "italic": False,
                                        "font_scale": 1,
                                    },
                                ],
                            },
                            {
                                "kind": "bullet",
                                "runs": [
                                    {
                                        "text": "Exploit known rewards",
                                        "bold": False,
                                        "italic": False,
                                        "font_scale": 1,
                                    }
                                ],
                            },
                        ],
                        "text_alignment": "left",
                        "font": {
                            "face": "Noto Sans",
                            "size_pt": 18,
                            "minimum_pt": 14,
                            "weight": "regular",
                            "color": "111111",
                            "line_height_ratio": 1.15,
                        },
                    }
                ],
            }
            plan_path = root / "composition_plan.json"
            compilation_path = root / "deck_compilation.json"
            pptx_path = root / "deck.pptx"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            compilation_path.write_text(
                json.dumps({"slides": [{"slide_id": "slide_001", "plan": str(plan_path)}]}),
                encoding="utf-8",
            )
            _compile_deck(compilation_path, pptx_path, root / "text_fit.json")
            with ZipFile(pptx_path) as archive:
                slide_xml = archive.read("ppt/slides/slide1.xml").decode("utf-8")

        self.assertIn('b="1"', slide_xml)
        self.assertIn("<a:buChar", slide_xml)
        self.assertIn("Explore or exploit", slide_xml)

    def test_title_and_body_use_exact_html_holder_safe_rect_without_growing_frame(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-holder-fit-") as temporary:
            root = Path(temporary)
            background = root / "background.png"
            title_holder = root / "title_holder.png"
            body_holder = root / "body_holder.png"
            Image.new("RGB", (1536, 864), "white").save(background)
            for path in (title_holder, body_holder):
                holder = Image.new("RGBA", (300, 80), (0, 0, 0, 0))
                ImageDraw.Draw(holder).rectangle(
                    (2, 2, 297, 77), fill=(248, 248, 245, 255), outline=(50, 50, 50, 255), width=3
                )
                holder.save(path)
            safe = {"x": 0.12, "y": 0.2, "width": 0.76, "height": 0.6}
            elements = [
                {
                    "region_id": "title",
                    "role": "title",
                    "bbox_px": [500, 50, 300, 80],
                    "holder_binding": {
                        "layer_id": "holder_title",
                        "holder_bbox_px": [500, 50, 300, 80],
                        "text_safe_rect": safe,
                    },
                    "native_holder": None,
                    "text": "A deliberately long title that needs more space",
                    "bullet_items": [],
                    "text_alignment": "center",
                    "font": {"face": "Noto Sans", "size_pt": 34, "minimum_pt": 25, "weight": "bold"},
                },
                {
                    "region_id": "body",
                    "role": "body",
                    "bbox_px": [100, 250, 300, 80],
                    "holder_binding": {
                        "layer_id": "holder_body",
                        "holder_bbox_px": [100, 250, 300, 80],
                        "text_safe_rect": safe,
                    },
                    "native_holder": None,
                    "text": "A deliberately long body paragraph that cannot fit inside the initial safe rectangle.",
                    "bullet_items": [],
                    "text_alignment": "left",
                    "font": {"face": "Noto Sans", "size_pt": 18, "minimum_pt": 14, "weight": "regular"},
                },
            ]
            plan = {
                "schema_version": "3.0",
                "slide_id": "slide_001",
                "canvas_px": {"width": 1536, "height": 864},
                "background_path": str(background),
                "raster_layers": [
                    {"layer_id": "holder_title", "kind": "holder", "bbox_px": [500, 50, 300, 80], "path": str(title_holder)},
                    {"layer_id": "holder_body", "kind": "holder", "bbox_px": [100, 250, 300, 80], "path": str(body_holder)},
                ],
                "groups": [
                    {"group_id": "singleton:title", "member_region_ids": ["title"]},
                    {"group_id": "singleton:body", "member_region_ids": ["body"]},
                ],
                "elements": elements,
            }
            plan_path = root / "composition_plan.json"
            compilation_path = root / "deck_compilation.json"
            audit_path = root / "text_fit.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            compilation_path.write_text(
                json.dumps({"slides": [{"slide_id": "slide_001", "plan": str(plan_path)}]}),
                encoding="utf-8",
            )
            _compile_deck(compilation_path, root / "deck.pptx", audit_path)
            audit = json.loads(audit_path.read_text())

        records = {item["role"]: item for item in audit["slides"][0]["elements"]}
        self.assertEqual(records["title"]["holder_mode"], "raster_safe_rect")
        self.assertEqual(records["body"]["holder_mode"], "raster_safe_rect")
        self.assertEqual(records["title"]["holder_scale"], 1)
        self.assertEqual(records["body"]["holder_scale"], 1)
        self.assertEqual(records["title"]["resolved_text_bbox_px"], [539, 69, 222, 42])
        self.assertEqual(records["body"]["resolved_text_bbox_px"], [139, 269, 222, 42])

    def test_html_elements_fit_font_locally_and_detected_safe_rect_insets_3px(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-role-font-fit-") as temporary:
            root = Path(temporary)
            background = root / "background.png"
            holder_path = root / "holder.png"
            Image.new("RGB", (1536, 864), "white").save(background)
            Image.new("RGBA", (400, 120), (245, 245, 240, 255)).save(holder_path)
            font = {
                "face": "Noto Sans",
                "size_pt": 15,
                "minimum_pt": 10,
                "weight": "regular",
            }
            plans = [
                {
                    "schema_version": "3.0",
                    "slide_id": "slide_001",
                    "canvas_px": {"width": 1536, "height": 864},
                    "background_path": str(background),
                    "raster_layers": [
                        {
                            "layer_id": "holder_body",
                            "kind": "holder",
                            "bbox_px": [100, 100, 400, 120],
                            "path": str(holder_path),
                        }
                    ],
                    "groups": [],
                    "elements": [
                        {
                            "region_id": "body_short",
                            "role": "body",
                            "bbox_px": [100, 100, 400, 120],
                            "holder_binding": {
                                "layer_id": "holder_body",
                                "holder_bbox_px": [100, 100, 400, 120],
                                "text_safe_rect": {
                                    "x": 0.1,
                                    "y": 0.2,
                                    "width": 0.8,
                                    "height": 0.6,
                                },
                            },
                            "native_holder": None,
                            "text": "Short body.",
                            "bullet_items": [],
                            "text_alignment": "left",
                            "font": font,
                        }
                    ],
                },
                {
                    "schema_version": "3.0",
                    "slide_id": "slide_002",
                    "canvas_px": {"width": 1536, "height": 864},
                    "background_path": str(background),
                    "raster_layers": [],
                    "groups": [],
                    "elements": [
                        {
                            "region_id": "body_long",
                            "role": "body",
                            "bbox_px": [100, 100, 360, 70],
                            "holder_binding": None,
                            "native_holder": None,
                            "text": (
                                "This deliberately long body sentence forces the shared body role "
                                "below its requested font size while the short body would fit unchanged."
                            ),
                            "bullet_items": [],
                            "text_alignment": "left",
                            "font": font,
                        }
                    ],
                },
            ]
            records = []
            for plan in plans:
                plan_path = root / f"{plan['slide_id']}.json"
                plan_path.write_text(json.dumps(plan), encoding="utf-8")
                records.append({"slide_id": plan["slide_id"], "plan": str(plan_path)})
            compilation_path = root / "deck_compilation.json"
            compilation_path.write_text(json.dumps({"slides": records}), encoding="utf-8")
            audit_path = root / "text_fit.json"
            _compile_deck(compilation_path, root / "deck.pptx", audit_path)
            audit = json.loads(audit_path.read_text())

        elements = [item for slide in audit["slides"] for item in slide["elements"]]
        resolved_sizes = {item["resolved_font_size_pt"] for item in elements}
        self.assertEqual(audit["font_resolution_scope"], "per_html_element")
        self.assertGreater(len(resolved_sizes), 1)
        self.assertAlmostEqual(elements[0]["resolved_font_size_pt"], 15, delta=0.1)
        self.assertLess(elements[1]["resolved_font_size_pt"], 15)
        self.assertEqual(elements[0]["text_safe_inset_px"], 3)
        self.assertEqual(elements[0]["resolved_text_bbox_px"], [143, 127, 314, 66])


if __name__ == "__main__":
    unittest.main()
