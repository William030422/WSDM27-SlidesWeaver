from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
import zipfile
from contextlib import chdir
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from integration.text_assembly import (
    ROOT,
    _apply_topology_move,
    _fit_element_deterministically,
    _load_text_placement,
    _topology_move_ranges,
)


class TextPlacementReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        """Resolve the frozen research case locally, independent of checkout path."""
        fixture = Path(__file__).resolve().parent / "fixtures/text_placement"
        case = json.loads((fixture / "case_slide_004.json").read_text())
        for key in ("background_path", "topology_path", "layout_path", "asset_manifest_path"):
            case[key] = str(fixture / case[key])
        temporary = tempfile.TemporaryDirectory(prefix="slidesweaver-case-")
        self.addCleanup(temporary.cleanup)
        self.case_path = Path(temporary.name) / "case_slide_004.json"
        self.case_path.write_text(json.dumps(case), encoding="utf-8")

    def test_bundled_case_matches_fixture_from_another_working_directory(self) -> None:
        module = _load_text_placement()
        bundled_case = ROOT / "reused/text_placement/case_slide_004.json"
        with patch.object(module, "configured_vlm_model", return_value=None):
            expected = module.build_plan(self.case_path)
            with tempfile.TemporaryDirectory(prefix="slidesweaver-case-cwd-") as temporary:
                with chdir(temporary):
                    actual = module.build_plan(bundled_case)
        self.assertEqual(actual, expected)

    def test_direct_text_solver_selects_white_on_dark_background(self) -> None:
        module = _load_text_placement()
        background = np.full((120, 240, 3), (9, 18, 28), dtype=np.uint8)
        metrics = module.analyze_region(background, [0, 0, 240, 120])
        presentation = module.choose_algorithmic_presentation(
            metrics,
            "body",
            {
                "color_complexity_threshold": 5.5,
                "luminance_std_threshold": 9.0,
                "native_shape_opacity": 0.90,
            },
        )

        self.assertEqual(presentation["mode"], "direct_text")
        self.assertEqual(presentation["text_color"], "FFFFFF")
        self.assertGreaterEqual(
            presentation["expected_contrast"], 4.5
        )

    def test_raster_holder_solver_samples_holder_surface(self) -> None:
        module = _load_text_placement()
        with tempfile.TemporaryDirectory(prefix="rl4ppt-holder-contrast-") as temporary:
            root = Path(temporary)
            background_path = root / "background.png"
            holder_path = root / "holder.png"
            background = np.full((140, 240, 3), (9, 18, 28), dtype=np.uint8)
            Image.fromarray(background).save(background_path)
            Image.new("RGBA", (160, 80), (12, 20, 28, 255)).save(holder_path)
            asset = {
                "asset_id": "dark_holder",
                "category": "raster_holder",
                "path": str(holder_path),
                "aspect_ratio": 2.0,
                "text_safe_rect": {
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.8,
                    "height": 0.8,
                },
            }

            presentation, holder_bbox = module._holder_presentation(
                asset,
                [20, 20, 160, 80],
                background,
            )

            self.assertEqual(presentation["mode"], "raster_holder_text")
            self.assertEqual(presentation["text_color"], "FFFFFF")
            self.assertEqual(
                presentation["contrast_source"], "holder_composited_surface"
            )
            self.assertGreaterEqual(presentation["expected_contrast"], 4.5)
            self.assertEqual(holder_bbox, [20.0, 20.0, 160.0, 80.0])

    def test_final_solver_updates_bare_and_holder_text_boxes(self) -> None:
        module = _load_text_placement()
        with tempfile.TemporaryDirectory(prefix="rl4ppt-all-box-contrast-") as temporary:
            root = Path(temporary)
            background_path = root / "background.png"
            holder_path = root / "holder.png"
            Image.new("RGB", (400, 180), (9, 18, 28)).save(background_path)
            Image.new("RGBA", (160, 80), (248, 248, 245, 255)).save(holder_path)
            safe = {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8}
            plan = {
                "background_path": str(background_path),
                "elements": [
                    {
                        "region_id": "bare",
                        "bbox_px": [20, 40, 160, 80],
                        "presentation": {
                            "mode": "direct_text",
                            "text_color": "000000",
                        },
                    },
                    {
                        "region_id": "holder",
                        "bbox_px": [220, 40, 160, 80],
                        "presentation": {
                            "mode": "raster_holder_text",
                            "asset_path": str(holder_path),
                            "text_safe_rect": safe,
                            "text_color": "FFFFFF",
                        },
                        "holder_binding": {
                            "asset_id": "light_holder",
                            "holder_bbox_px": [220, 40, 160, 80],
                            "text_bbox_px": [236, 48, 128, 64],
                        },
                    },
                ],
            }

            module.apply_text_contrast_solver(plan)

            self.assertEqual(
                plan["elements"][0]["presentation"]["text_color"], "FFFFFF"
            )
            self.assertEqual(
                plan["elements"][1]["presentation"]["text_color"], "000000"
            )
            self.assertGreaterEqual(
                plan["elements"][0]["presentation"]["expected_contrast"], 4.5
            )
            self.assertGreaterEqual(
                plan["elements"][1]["presentation"]["expected_contrast"], 4.5
            )

    def test_explicit_holder_decision_recomputes_text_color(self) -> None:
        module = _load_text_placement()
        with tempfile.TemporaryDirectory(prefix="rl4ppt-holder-decision-") as temporary:
            root = Path(temporary)
            background_path = root / "background.png"
            holder_path = root / "holder.png"
            Image.new("RGB", (240, 140), (9, 18, 28)).save(background_path)
            Image.new("RGBA", (160, 80), (12, 20, 28, 255)).save(holder_path)
            safe = {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8}
            bbox = [20.0, 20.0, 160.0, 80.0]
            plan = {
                "background_path": str(background_path),
                "canvas_px": {"width": 240, "height": 140},
                "groups": [
                    {
                        "group_id": "singleton:text",
                        "member_region_ids": ["text"],
                        "arrangement": "freeform",
                        "equal_size": False,
                        "even_spacing": False,
                        "preserve_alignment": False,
                    }
                ],
                "elements": [
                    {
                        "region_id": "text",
                        "role": "body",
                        "bbox_px": bbox,
                        "font": {"size_pt": 18},
                        "presentation": {
                            "mode": "direct_text",
                            "shape": None,
                            "text_color": "FFFFFF",
                        },
                        "content_unit": {
                            "unit_id": "unit_text",
                            "outer_bbox_px": bbox,
                            "holder_asset_id": None,
                            "decoration_asset_id": None,
                            "binding_frozen": False,
                        },
                    }
                ],
                "available_assets": [
                    {
                        "asset_id": "dark_holder",
                        "category": "raster_holder",
                        "path": str(holder_path),
                        "aspect_ratio": 2.0,
                        "text_safe_rect": safe,
                    }
                ],
            }

            updated, outcome = module.apply_asset_decision_non_failing(
                plan,
                {
                    "region_id": "text",
                    "decision": "use_raster_holder",
                    "asset_id": "dark_holder",
                },
            )

            self.assertEqual(outcome["status"], "applied")
            self.assertEqual(
                updated["elements"][0]["presentation"]["text_color"], "FFFFFF"
            )

    def test_title_background_complexity_controls_holder_binding(self) -> None:
        module = _load_text_placement()
        policy = {"title_background_complexity_threshold": 30.0}
        solid = np.full((120, 240, 3), (214, 221, 230), dtype=np.uint8)
        gradient = solid.copy()
        gradient[:, :, 0] = np.linspace(204, 224, gradient.shape[1], dtype=np.uint8)
        high_contrast_gradient = np.repeat(
            np.linspace(20, 235, solid.shape[1], dtype=np.uint8)[None, :, None],
            solid.shape[0],
            axis=0,
        )
        high_contrast_gradient = np.repeat(high_contrast_gradient, 3, axis=2)
        semantic_texture = solid.copy()
        semantic_texture[15:105, 20:220] = np.indices((90, 200)).sum(axis=0)[..., None] % 2 * 190 + 30

        solid_metrics = module.analyze_region(solid, [0, 0, 240, 120])
        gradient_metrics = module.analyze_region(gradient, [0, 0, 240, 120])
        high_gradient_metrics = module.analyze_region(
            high_contrast_gradient, [0, 0, 240, 120]
        )
        texture_metrics = module.analyze_region(semantic_texture, [0, 0, 240, 120])

        self.assertFalse(module._title_background_requires_holder(solid_metrics, policy))
        self.assertFalse(module._title_background_requires_holder(gradient_metrics, policy))
        self.assertTrue(
            module._title_background_requires_holder(high_gradient_metrics, policy)
        )
        self.assertTrue(module._title_background_requires_holder(texture_metrics, policy))
        self.assertLess(gradient_metrics["background_complexity_score"], 30.0)
        self.assertGreater(texture_metrics["background_complexity_score"], 30.0)

    def test_build_plan_only_binds_complex_title_background_to_holder(self) -> None:
        module = _load_text_placement()
        source_case = json.loads(
            (self.case_path).read_text(
                encoding="utf-8"
            )
        )
        with tempfile.TemporaryDirectory(prefix="rl4ppt-title-background-") as temporary:
            temporary_dir = Path(temporary)
            for name, background in (
                (
                    "solid",
                    np.full((864, 1536, 3), (214, 221, 230), dtype=np.uint8),
                ),
                (
                    "complex",
                    np.random.default_rng(7).integers(
                        20, 236, size=(864, 1536, 3), dtype=np.uint8
                    ),
                ),
            ):
                background_path = temporary_dir / f"{name}.png"
                Image.fromarray(background).save(background_path)
                case = {**source_case, "background_path": str(background_path)}
                case["mode_policy"] = {
                    **source_case["mode_policy"],
                    "title_background_complexity_threshold": 30.0,
                }
                case_path = temporary_dir / f"{name}.json"
                case_path.write_text(json.dumps(case), encoding="utf-8")
                plan = module.build_plan(case_path)
                title = next(item for item in plan["elements"] if item["role"] == "title")
                expected = "direct_text" if name == "solid" else "raster_holder_text"
                self.assertEqual(title["presentation"]["mode"], expected)

    def test_bare_title_keeps_available_planning_width(self) -> None:
        module = _load_text_placement()
        plan = module.build_plan(
            self.case_path
        )
        title_index = next(
            index
            for index, item in enumerate(plan["elements"])
            if item["role"] == "title"
        )
        original = json.loads(json.dumps(plan["elements"][title_index]))
        planning_bbox = original["topology_bbox_px"]

        record = _fit_element_deterministically(plan, title_index, module)

        title = plan["elements"][title_index]
        bbox = title["bbox_px"]
        self.assertEqual(title["presentation"]["mode"], "direct_text")
        self.assertEqual(
            title["registration"]["title_text_box_mode"],
            "full_planning_width",
        )
        self.assertEqual(bbox[1], original["bbox_px"][1])
        self.assertEqual(bbox[3], original["bbox_px"][3])
        self.assertEqual(title["font"], original["font"])
        self.assertEqual(bbox[2], planning_bbox[2])
        self.assertAlmostEqual(
            bbox[0] + bbox[2] / 2, planning_bbox[0] + planning_bbox[2] / 2
        )
        self.assertEqual(record["atomic_checks"][0]["line_count"], 1)


    def test_topology_move_changes_exactly_one_axis_inside_original_box(self) -> None:
        module = _load_text_placement()
        plan = module.freeze_content_units(
            module.build_plan(self.case_path)
        )
        target = next(item for item in plan["elements"] if item["region_id"] == "text_001")
        search = target["topology_bbox_px"]
        new_x = search[0]

        moved, outcome = _apply_topology_move(
            plan,
            "text_001",
            {"axis": "x", "position_px": new_x},
            module,
        )

        self.assertEqual(outcome["status"], "applied")
        result = next(item for item in moved["elements"] if item["region_id"] == "text_001")
        self.assertEqual(result["bbox_px"][0], new_x)
        self.assertEqual(result["bbox_px"][1], target["bbox_px"][1])
        self.assertEqual(
            result["holder_binding"]["holder_bbox_px"],
            [float(value) for value in result["bbox_px"]],
        )

    def test_topology_move_rejects_position_that_does_not_fit_search_box(self) -> None:
        module = _load_text_placement()
        plan = module.freeze_content_units(
            module.build_plan(self.case_path)
        )
        target = next(item for item in plan["elements"] if item["region_id"] == "caption_001")
        search = target["topology_bbox_px"]

        unchanged, outcome = _apply_topology_move(
            plan,
            "caption_001",
            {"axis": "x", "position_px": search[0] + search[2]},
            module,
        )

        self.assertEqual(outcome["reason"], "move_does_not_fit_topology_bbox")
        self.assertEqual(unchanged, plan)

    def test_topology_move_absorbs_one_pixel_boundary_rounding(self) -> None:
        module = _load_text_placement()
        plan = module.freeze_content_units(
            module.build_plan(self.case_path)
        )
        maximum_x = _topology_move_ranges(plan, "text_001")["x"][1]

        moved, outcome = _apply_topology_move(
            plan,
            "text_001",
            {"axis": "x", "position_px": maximum_x + 0.5},
            module,
        )

        self.assertEqual(outcome["status"], "applied")
        result = next(item for item in moved["elements"] if item["region_id"] == "text_001")
        self.assertEqual(result["bbox_px"][0], maximum_x)

    def test_topology_move_ranges_intersect_every_group_member_search_box(self) -> None:
        plan = {
            "groups": [
                {
                    "group_id": "rigid",
                    "member_region_ids": ["target", "peer"],
                }
            ],
            "elements": [
                {
                    "region_id": "target",
                    "bbox_px": [100, 50, 40, 20],
                    "topology_bbox_px": [80, 40, 100, 50],
                },
                {
                    "region_id": "peer",
                    "bbox_px": [160, 70, 30, 20],
                    "topology_bbox_px": [150, 60, 60, 40],
                },
            ],
        }

        self.assertEqual(
            _topology_move_ranges(plan, "target"),
            {"x": [90.0, 120.0], "y": [40.0, 60.0]},
        )

    def test_topology_move_ranges_omit_axis_without_meaningful_space(self) -> None:
        plan = {
            "groups": [
                {
                    "group_id": "rigid",
                    "member_region_ids": ["target"],
                }
            ],
            "elements": [
                {
                    "region_id": "target",
                    "bbox_px": [100, 50, 40, 20],
                    "topology_bbox_px": [100, 40, 40, 40],
                }
            ],
        }

        self.assertEqual(
            _topology_move_ranges(plan, "target"),
            {"y": [40.0, 60.0]},
        )
        _, outcome = _apply_topology_move(
            plan,
            "target",
            {"axis": "x", "position_px": 110},
            _load_text_placement(),
        )
        self.assertEqual(outcome["reason"], "axis_not_movable")

    def test_holder_selection_prefers_regular_holder_then_wider_title_holder(self) -> None:
        module = _load_text_placement()
        assets = [
            {
                "asset_id": "holder_wide",
                "category": "raster_holder",
                "aspect_ratio": 2.3,
                "text_safe_rect": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8},
            },
            {
                "asset_id": "title_bar",
                "category": "title_bar",
                "aspect_ratio": 6.2,
                "text_safe_rect": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8},
            },
        ]

        regular = module._select_holder_asset("subtitle", [0, 0, 230, 100], assets)
        wide = module._select_holder_asset("subtitle", [0, 0, 900, 100], assets)
        title = module._select_holder_asset("title", [0, 0, 900, 100], assets)

        self.assertEqual(regular["asset_id"], "holder_wide")
        self.assertEqual(wide["asset_id"], "title_bar")
        self.assertEqual(title["asset_id"], "title_bar")

    def test_deterministic_registration_checks_every_atomic_step(self) -> None:
        module = _load_text_placement()
        plan = {
            "canvas_px": {"width": 400, "height": 240},
            "groups": [
                {
                    "group_id": "singleton:text",
                    "member_region_ids": ["text"],
                    "arrangement": "freeform",
                    "peer_relation": "none",
                    "equal_size": False,
                    "even_spacing": False,
                    "preserve_alignment": False,
                }
            ],
            "elements": [
                {
                    "region_id": "text",
                    "role": "subtitle",
                    "bbox_px": [150, 100, 100, 40],
                    "topology_bbox_px": [50, 100, 300, 40],
                    "anchor_center_px": [200, 120],
                    "text": "A deliberately long line of text that needs deterministic fitting.",
                    "bullet_items": [],
                    "font": {
                        "face": "Aptos",
                        "size_pt": 18,
                        "minimum_pt": 16,
                        "maximum_pt": 20,
                        "weight": "regular",
                    },
                    "presentation": {
                        "mode": "raster_holder_text",
                        "asset_id": "holder",
                        "asset_category": "raster_holder",
                        "text_safe_rect": {
                            "x": 0.1,
                            "y": 0.1,
                            "width": 0.8,
                            "height": 0.8,
                        },
                        "text_color": "000000",
                    },
                }
            ],
        }

        def measurements(current_plan, states, placement):
            # Exercise the complete search without depending on installed font
            # metrics making this particular sentence fit at the initial step.
            return [{"fits": state["element"]["bbox_px"][3] >= 120, "line_count": 3,
                     "required_height_in": 3.75,
                     "available_height_in": state["element"]["bbox_px"][3] / 240 * 7.5}
                    for state in states]
        with patch("integration.text_assembly._measure_registration_states", side_effect=measurements):
            record = _fit_element_deterministically(plan, 0, module)

        checks = record["atomic_checks"]
        horizontal = [item for item in checks if item["op"] == "expand_horizontal"]
        font = [item for item in checks if item["op"] == "decrease_font"]
        vertical = [item for item in checks if item["op"] == "expand_vertical_symmetric"]
        self.assertTrue(horizontal)
        self.assertTrue(all(item["to_width_px"] - item["from_width_px"] <= 4 for item in horizontal))
        self.assertTrue(all(item["from_size_pt"] - item["to_size_pt"] == 1 for item in font))
        self.assertTrue(all(item["to_height_px"] - item["from_height_px"] <= 4 for item in vertical))
        self.assertTrue(all(item["top_delta_px"] == -item["bottom_delta_px"] for item in vertical))
        self.assertTrue(all("fits" in item for item in checks))

    def test_frozen_content_unit_rejects_split_binding_and_holder_distortion(self) -> None:
        module = _load_text_placement()
        plan = module.build_plan(self.case_path)
        plan, outcome = module.apply_asset_decision_non_failing(
            plan,
            {
                "region_id": "caption_001",
                "decision": "use_raster_holder",
                "asset_id": "holder_wide",
            },
        )
        self.assertEqual(outcome["status"], "applied")
        plan = module.freeze_content_units(plan)
        element = next(item for item in plan["elements"] if item["region_id"] == "caption_001")
        unchanged, outcome = module.apply_asset_decision_non_failing(
            plan,
            {
                "region_id": "caption_001",
                "decision": "use_raster_holder",
                "asset_id": "holder_compact",
            },
        )
        self.assertEqual(outcome["reason"], "content_unit_binding_frozen")
        self.assertEqual(unchanged, plan)
        unchanged, outcome = module.apply_patch_non_failing(
            plan,
            {
                "region_id": "caption_001",
                "op": "resize",
                "width_px": element["bbox_px"][2] * 1.04,
                "height_px": element["bbox_px"][3] * 1.01,
            },
        )
        self.assertEqual(outcome["reason"], "holder_resize_must_be_proportional")
        self.assertEqual(unchanged, plan)
        adjusted, outcome = module.apply_unit_delta_non_failing(
            plan,
            {
                "region_id": "caption_001",
                "op": "set_padding",
                "left": 0.02,
                "right": 0.02,
                "top": 0.01,
                "bottom": 0.01,
            },
            "caption_001",
        )
        self.assertEqual(outcome["status"], "applied")
        adjusted_element = next(
            item for item in adjusted["elements"] if item["region_id"] == "caption_001"
        )
        self.assertEqual(adjusted_element["bbox_px"], element["bbox_px"])
        self.assertEqual(
            adjusted_element["holder_binding"]["asset_id"],
            element["holder_binding"]["asset_id"],
        )

    def test_original_case_and_multislide_compiler_remain_compatible(self) -> None:
        module = _load_text_placement()
        case_path = self.case_path
        plan = module.build_plan(case_path)
        self.assertEqual(plan["status"], "ok")
        self.assertEqual(plan["binding"]["unplaced_content_ids"], [])
        self.assertEqual(plan["binding"]["empty_region_ids"], [])
        with tempfile.TemporaryDirectory(prefix="rl4ppt-reuse-compile-") as temporary:
            temporary_dir = Path(temporary)
            plan_path = temporary_dir / "plan.json"
            compilation_path = temporary_dir / "compilation.json"
            output_path = temporary_dir / "deck.pptx"
            audit_path = temporary_dir / "text_fit_report.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            compilation_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "slides": [{"slide_id": plan["slide_id"], "plan": str(plan_path)}],
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    "node",
                    str(ROOT / "compile_deck.mjs"),
                    str(compilation_path),
                    str(output_path),
                    str(audit_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 10_000)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["slides"][0]["slide_id"], plan["slide_id"])
            self.assertTrue(audit["slides"][0]["elements"])

    def test_rewrite_assets_addition_and_role_update_remain_compilable(self) -> None:
        module = _load_text_placement()
        plan = module.build_plan(self.case_path)
        plan, outcome = module.apply_rewrite_non_failing(
            plan,
            {"region_id": "caption_001", "text": "Sense live conditions"},
            {"caption_001"},
        )
        self.assertEqual(outcome["status"], "applied")
        caption = next(
            item for item in plan["elements"] if item["region_id"] == "caption_001"
        )
        plan, outcome = module.apply_patch_non_failing(
            plan,
            {
                "region_id": "caption_001",
                "op": "resize",
                "width_px": caption["bbox_px"][2] * 1.02,
                "height_px": caption["bbox_px"][3] * 1.02,
            },
        )
        self.assertEqual(outcome["status"], "applied")
        plan, outcome = module.apply_asset_decision_non_failing(
            plan,
            {
                "region_id": "caption_001",
                "decision": "use_raster_holder",
                "asset_id": "holder_wide",
            },
        )
        self.assertEqual(outcome["status"], "applied")
        holder_element = next(
            item for item in plan["elements"] if item["region_id"] == "caption_001"
        )
        self.assertEqual(
            holder_element["holder_binding"]["holder_bbox_px"],
            [float(value) for value in holder_element["bbox_px"]],
        )
        self.assertEqual(
            holder_element["holder_binding"]["text_bbox_px"],
            module.derive_holder_text_bbox_px(
                holder_element["bbox_px"],
                holder_element["presentation"]["text_safe_rect"],
            ),
        )
        original_text_bbox = holder_element["holder_binding"]["text_bbox_px"]
        plan, outcome = module.apply_patch_non_failing(
            plan,
            {
                "region_id": "caption_001",
                "op": "resize",
                "width_px": holder_element["bbox_px"][2] * 1.04,
                "height_px": holder_element["bbox_px"][3] * 1.04,
            },
        )
        self.assertEqual(outcome["status"], "applied")
        holder_element = next(
            item for item in plan["elements"] if item["region_id"] == "caption_001"
        )
        self.assertNotEqual(
            holder_element["holder_binding"]["text_bbox_px"], original_text_bbox
        )
        self.assertEqual(
            holder_element["holder_binding"]["text_bbox_px"],
            module.element_text_rect_px(holder_element),
        )
        plan, outcome = module.apply_asset_decision_non_failing(
            plan,
            {
                "region_id": "title_001",
                "decision": "add_badge",
                "asset_id": "badge_emphasis",
            },
        )
        self.assertEqual(outcome["status"], "applied")
        plan, outcome = module.apply_addition_non_failing(
            plan,
            {
                "region_id": "annotation_new_001",
                "text": "Feedback closes the loop.",
                "role": "caption",
                "bbox_px": [1250, 300, 220, 70],
                "asset_id": None,
            },
        )
        self.assertEqual(outcome["status"], "applied")
        plan, outcome = module.apply_role_update_non_failing(
            plan,
            {"role": "caption", "op": "set_font_size", "size_pt": 16},
        )
        self.assertEqual(outcome["status"], "applied")

        with tempfile.TemporaryDirectory(prefix="rl4ppt-style-compile-") as temporary:
            temporary_dir = Path(temporary)
            plan_path = temporary_dir / "plan.json"
            compilation_path = temporary_dir / "compilation.json"
            output_path = temporary_dir / "deck.pptx"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            compilation_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "slides": [{"slide_id": plan["slide_id"], "plan": str(plan_path)}],
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                ["node", str(ROOT / "compile_deck.mjs"), str(compilation_path), str(output_path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
            self.assertGreater(output_path.stat().st_size, 10_000)

    def test_bullet_items_compile_as_left_aligned_pptx_bullets(self) -> None:
        module = _load_text_placement()
        plan = module.build_plan(self.case_path)
        element = next(item for item in plan["elements"] if item["role"] != "title")
        element["text"] = "Observe the state\nChoose an action"
        element["bullet_items"] = ["Observe the state", "Choose an action"]

        with tempfile.TemporaryDirectory(prefix="rl4ppt-bullets-") as temporary:
            temporary_dir = Path(temporary)
            plan_path = temporary_dir / "plan.json"
            compilation_path = temporary_dir / "compilation.json"
            output_path = temporary_dir / "deck.pptx"
            audit_path = temporary_dir / "text_fit_report.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            compilation_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "slides": [{"slide_id": plan["slide_id"], "plan": str(plan_path)}],
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    "node",
                    str(ROOT / "compile_deck.mjs"),
                    str(compilation_path),
                    str(output_path),
                    str(audit_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
            with zipfile.ZipFile(output_path) as archive:
                slide_xml = archive.read("ppt/slides/slide1.xml").decode("utf-8")
            self.assertIn("<a:buChar", slide_xml)
            self.assertIn('algn="l"', slide_xml)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audited = next(
                item
                for item in audit["slides"][0]["elements"]
                if item["region_id"] == element["region_id"]
            )
            self.assertTrue(audited["is_bulleted"])

    def test_native_shape_opacity_never_drops_below_ninety_percent(self) -> None:
        module = _load_text_placement()
        plan = module.build_plan(self.case_path)
        element = plan["elements"][0]
        element["presentation"] = module.native_shape_style(
            element["metrics"], 0.90, tuple(plan["theme_rgb"])
        )
        module._refresh_holder_binding(element)
        module._refresh_content_unit(element)
        native_shapes = [
            item for item in plan["elements"] if item["presentation"]["mode"] == "native_shape_text"
        ]
        self.assertTrue(native_shapes)
        self.assertTrue(
            all(
                item["presentation"]["shape"]["fill_opacity"] >= 0.90
                for item in native_shapes
            )
        )
        role = native_shapes[0]["role"]
        unchanged, outcome = module.apply_role_update_non_failing(
            plan,
            {"role": role, "op": "set_shape_opacity", "opacity": 0.89},
        )
        self.assertEqual(outcome, {"status": "ignored", "reason": "opacity_out_of_range"})
        self.assertEqual(unchanged, plan)


if __name__ == "__main__":
    unittest.main()
