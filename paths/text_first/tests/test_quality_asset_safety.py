from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from integration.config import Size
from integration.contrast import apply_background_aware_text_colors, resolved_text_bbox_px
from integration.final_background import (
    CANVAS,
    _assign_components_to_slots,
    build_asset_board_plan,
    extract_chroma_layers,
    plan_asset_board_slots,
    run_background_set,
    render_mock_asset_board,
)
from integration.holder_safe_rect import detect_text_safe_rect


class AssetSafetyTests(unittest.TestCase):
    def test_distant_fragment_does_not_fill_missing_slot(self) -> None:
        objects = [
            {"object_id": "left", "slot_bbox_px": [20, 20, 150, 100]},
            {"object_id": "missing", "slot_bbox_px": [1200, 650, 100, 100]},
        ]
        records = [
            {"asset_id": "main", "component_bbox": [30, 30, 160, 110], "centroid": [95, 70]},
            {"asset_id": "part", "component_bbox": [40, 130, 80, 150], "centroid": [60, 140]},
        ]
        assignments = _assign_components_to_slots(records, objects)
        self.assertEqual(assignments["missing"], [])
        self.assertEqual([record["asset_id"] for record in assignments["left"]], ["main", "part"])

    def test_reasonable_drift_is_retained(self) -> None:
        objects = [{"object_id": "visual", "slot_bbox_px": [100, 100, 100, 100]}]
        records = [{"asset_id": "near", "component_bbox": [220, 100, 260, 140], "centroid": [240, 120]}]
        self.assertEqual(_assign_components_to_slots(records, objects)["visual"], records)

    def test_fragment_overlapping_occupied_slot_stays_with_its_owner(self) -> None:
        objects = [
            {"object_id": "left", "slot_bbox_px": [100, 100, 200, 100]},
            {"object_id": "missing", "slot_bbox_px": [350, 100, 100, 100]},
        ]
        records = [
            {"asset_id": "main", "component_bbox": [110, 110, 250, 180], "centroid": [180, 145]},
            {"asset_id": "part", "component_bbox": [270, 120, 290, 140], "centroid": [280, 130]},
        ]
        assignments = _assign_components_to_slots(records, objects)
        self.assertEqual(assignments["missing"], [])
        self.assertEqual(len(assignments["left"]), 2)

    def test_safe_rectangle_never_contains_closed_transparent_hole(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Image.new("RGBA", (400, 200), "white")
            ImageDraw.Draw(image).rectangle((198, 98, 200, 100), fill=(0, 0, 0, 0))
            path = Path(temporary) / "holder.png"
            image.save(path)
            safe = detect_text_safe_rect(path)
            self.assertIsNotNone(safe)
            x, y = round(safe["x"] * 400), round(safe["y"] * 200)
            width, height = round(safe["width"] * 400), round(safe["height"] * 200)
            alpha = np.asarray(image)[y:y + height, x:x + width, 3]
            self.assertTrue((alpha >= 224).all())

    def test_mixed_background_gets_native_outline_and_honest_quality_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "background.png"
            background = Image.new("RGB", (400, 200), "white")
            ImageDraw.Draw(background).rectangle((0, 0, 149, 199), fill="black")
            background.save(path)
            plan = self._contrast_plan(path)
            original_bbox = list(plan["elements"][0]["bbox_px"])
            report = apply_background_aware_text_colors(plan)
            self.assertEqual(report["status"], "degraded")
            self.assertLess(report["blocks"][0]["local_contrast_ratio"], 4.5)
            self.assertGreaterEqual(report["blocks"][0]["effective_local_contrast_ratio"], 4.5)
            self.assertEqual(plan["elements"][0]["font"]["outline"], {"color": "FFFFFF", "size": 0.6})
            self.assertEqual(plan["elements"][0]["bbox_px"], original_bbox)

    def test_uniform_background_keeps_planned_color_without_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "background.png"
            Image.new("RGB", (400, 200), "white").save(path)
            plan = self._contrast_plan(path)
            report = apply_background_aware_text_colors(plan)
            self.assertEqual(report["status"], "strict_pass")
            self.assertEqual(plan["elements"][0]["font"]["color"], "000000")
            self.assertNotIn("outline", plan["elements"][0]["font"])

    def test_precomputed_content_box_is_shared_with_export(self) -> None:
        element = {"bbox_px": [0, 0, 400, 200], "resolved_text_bbox_px": [32, 24, 336, 152]}
        self.assertEqual(resolved_text_bbox_px(element), [32, 24, 336, 152])

    def test_empty_matte_preserves_holder_fallback_without_aborting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = root / "generated.png"
            background = root / "background.png"
            Image.new("RGB", CANVAS, (0, 255, 0)).save(generated)
            Image.new("RGB", CANVAS, "white").save(background)
            objects = plan_asset_board_slots([
                {"object_id": "holder", "kind": "holder", "bbox_px": [80, 80, 400, 160], "text_region_id": "text"},
                {"object_id": "visual", "kind": "visual", "bbox_px": [700, 200, 300, 300]},
            ])
            plan = root / "asset_plan.json"
            plan.write_text(json.dumps(build_asset_board_plan(objects, "#00ff00")))
            manifest = extract_chroma_layers(
                generated_path=generated, canonical_background=background, objects=objects,
                asset_plan_path=plan, segmentation_dir=root / "segmentation", extracted_assets_dir=root / "assets", output_dir=root / "layers",
            )
            self.assertEqual([layer["layer_id"] for layer in manifest["layers"]], ["holder"])
            self.assertTrue(manifest["layers"][0]["holder_fallback"])
            self.assertEqual(manifest["missing_object_ids"], ["visual"])
            self.assertEqual(len(manifest["quality_warnings"]), 2)

    def test_mock_missing_assets_do_not_abort_or_create_api_client(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_background_fixture(root)
            config = SimpleNamespace(image_model="gpt-image-2", image_size=Size(1536, 864), image_resolution="1K", image_quality="medium")

            def empty_board(objects, key_color, output_path):
                Image.new("RGB", CANVAS, (0, 255, 0)).save(output_path)

            with patch("integration.final_background.render_mock_asset_board", side_effect=empty_board):
                result = run_background_set(
                    project_root=root, html_compile_dir=root / "html", step2_dir=root / "style", output_dir=root / "out",
                    slide_ids=["slide_001"], config=config, mock_images=True, resume_missing=False,
                    client_factory=lambda: self.fail("mock quality fallback cannot create an API client"),
                )
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["quality_status"], "degraded")
            self.assertEqual(result["failed_slides"], [])
            manifest = json.loads((root / "out/slide_001/layer_manifest.json").read_text())
            self.assertTrue(manifest["layers"][0]["holder_fallback"])
            self.assertIsNone(result["slides"][0]["failure_placeholder"])

    def test_overfull_asset_board_uses_bounded_local_packing(self) -> None:
        objects = [{"object_id": f"v{index}", "kind": "visual", "bbox_px": [0, 0, 1536, 864]} for index in range(20)]
        planned = plan_asset_board_slots(objects)
        self.assertEqual(len(planned), len(objects))
        for original, item in zip(objects, planned, strict=True):
            self.assertEqual(item["bbox_px"], original["bbox_px"])
            x, y, width, height = item["slot_bbox_px"]
            self.assertGreater(width, 0)
            self.assertGreater(height, 0)
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + width, CANVAS[0])
            self.assertLessEqual(y + height, CANVAS[1])

    def test_empty_recovery_result_exhausts_budget_without_aborting_slide(self) -> None:
        class Client:
            def __init__(self):
                self.submitted = 0

            def submit_image_task(self, payload):
                self.submitted += 1
                return f"task_{self.submitted}"

            def poll_task(self, task_id):
                return {"status": "completed", "result": {"images": [{"url": "unused"}]}}

            def download(self, url, output_path):
                Image.new("RGB", CANVAS, (0, 255, 0)).save(output_path)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_background_fixture(root)
            config = SimpleNamespace(image_model="gpt-image-2", image_size=Size(1536, 864), image_resolution="1K", image_quality="medium")
            client = Client()
            result = run_background_set(
                project_root=root, html_compile_dir=root / "html", step2_dir=root / "style", output_dir=root / "out",
                slide_ids=["slide_001"], config=config, mock_images=False, resume_missing=False,
                client_factory=lambda: client,
            )
            self.assertEqual(client.submitted, 2)
            self.assertEqual(result["image_call_count"], 2)
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["quality_status"], "degraded")
            self.assertEqual(result["failed_slides"], [])
            self.assertIsNone(result["slides"][0]["failure_placeholder"])
            manifest = json.loads((root / "out/slide_001/layer_manifest.json").read_text())
            self.assertTrue(manifest["layers"][0]["holder_fallback"])
            self.assertEqual(manifest["base_background"], str((root / "style/canonical_background.png").resolve()))

    def test_downloaded_board_size_is_registered_without_overwriting_original(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            background = root / "background.png"
            generated = root / "generated.png"
            Image.new("RGB", CANVAS, "white").save(background)
            objects = plan_asset_board_slots([
                {"object_id": "visual", "kind": "visual", "bbox_px": [700, 200, 300, 180]},
            ])
            render_mock_asset_board(objects, "#00ff00", generated)
            with Image.open(generated) as full_size:
                half_size = full_size.resize((768, 432), Image.Resampling.LANCZOS)
            half_size.save(generated)
            plan = root / "asset_plan.json"
            plan.write_text(json.dumps(build_asset_board_plan(objects, "#00ff00")))
            manifest = extract_chroma_layers(
                generated_path=generated, canonical_background=background, objects=objects,
                asset_plan_path=plan, segmentation_dir=root / "segmentation", extracted_assets_dir=root / "assets", output_dir=root / "layers",
            )
            with Image.open(generated) as original:
                self.assertEqual(original.size, (768, 432))
            with Image.open(manifest["registered_canvas"]) as registered:
                self.assertEqual(registered.size, CANVAS)
            self.assertEqual(manifest["source_board_size"], [768, 432])
            self.assertEqual(len(manifest["layers"]), 1)
            self.assertEqual(manifest["missing_object_ids"], [])
            self.assertIn("generated_board_size_normalized", [item["reason"] for item in manifest["quality_warnings"]])
            placed = manifest["layers"][0]["bbox_px"]
            self.assertGreaterEqual(placed[0], 700)
            self.assertGreaterEqual(placed[1], 200)
            self.assertLessEqual(placed[0] + placed[2], 1000)
            self.assertLessEqual(placed[1] + placed[3], 380)

    @staticmethod
    def _contrast_plan(background: Path) -> dict:
        return {
            "slide_id": "slide_001", "canvas_px": {"width": 400, "height": 200},
            "background_path": str(background), "raster_layers": [],
            "elements": [{"region_id": "body", "role": "body", "bbox_px": [0, 0, 400, 200], "font": {"color": "000000"}}],
        }

    @staticmethod
    def _write_background_fixture(root: Path) -> None:
        style = root / "style"
        style.mkdir()
        Image.new("RGB", CANVAS, "white").save(style / "canonical_background.png")
        (style / "design_system.json").write_text(json.dumps({"style_prompt": "simple illustration"}))
        slide = root / "html/slides/slide_001"
        slide.mkdir(parents=True)
        (slide / "html_layout.json").write_text(json.dumps({
            "visual_regions": [], "text_regions": [],
            "holders": [{"holder_id": "holder", "bbox_px": [80, 80, 400, 160], "text_region_id": "text"}],
        }))


if __name__ == "__main__":
    unittest.main()
