from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from v9_deck_planning.green_background_removal import extract_assets
from v9_deck_planning.segmented_asset_naming import name_segmented_assets
from v9_deck_planning.style_assets import (
    CANVAS,
    detect_text_safe_rect,
    render_segmented_reference,
)


class Step2AssetPostprocessingTests(unittest.TestCase):
    def test_text_safe_rectangle_excludes_holder_border_and_speech_tail(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-holder-safe-rect-") as temporary:
            path = Path(temporary) / "speech.png"
            image = Image.new("RGBA", (180, 140), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.polygon([(92, 92), (132, 130), (122, 88)], fill=(245, 245, 238, 255))
            draw.rounded_rectangle(
                (12, 10, 168, 100),
                radius=18,
                fill=(245, 245, 238, 255),
                outline=(30, 30, 35, 255),
                width=6,
            )
            image.save(path)

            rectangle = detect_text_safe_rect(path)

            self.assertIsNotNone(rectangle)
            assert rectangle is not None
            self.assertGreater(rectangle["x"], 12 / 180)
            self.assertGreater(rectangle["y"], 10 / 140)
            self.assertLess(rectangle["x"] + rectangle["width"], 168 / 180)
            self.assertLess(rectangle["y"] + rectangle["height"], 100 / 140)

    def test_text_safe_rectangle_excludes_opaque_edge_patterns(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-holder-pattern-") as temporary:
            path = Path(temporary) / "patterned.png"
            image = Image.new("RGBA", (160, 100), (45, 48, 52, 255))
            draw = ImageDraw.Draw(image)
            draw.rectangle((30, 20, 130, 80), fill=(241, 221, 188, 255))
            draw.rectangle((30, 20, 50, 36), fill=(56, 104, 48, 255))
            draw.rectangle((110, 64, 130, 80), fill=(82, 72, 58, 255))
            image.save(path)

            rectangle = detect_text_safe_rect(path)

            self.assertIsNotNone(rectangle)
            assert rectangle is not None
            self.assertGreaterEqual(rectangle["x"], 32 / 160)
            self.assertGreaterEqual(rectangle["y"], 22 / 100)
            self.assertLessEqual(rectangle["x"] + rectangle["width"], 129 / 160)
            self.assertLessEqual(rectangle["y"] + rectangle["height"], 79 / 100)

    def test_assets_remain_atomic_and_reference_uses_postprocessed_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-step2-assets-") as temporary:
            root = Path(temporary)
            raw_path = root / "style_reference_raw.png"
            segmentation_dir = root / "segmentation"
            assets_dir = root / "assets"
            plan_path = root / "style_reference_plan.json"
            reference_path = root / "style_reference.png"

            image = Image.new("RGB", CANVAS, (0, 255, 0))
            draw = ImageDraw.Draw(image)
            draw.ellipse((140, 160, 230, 250), fill=(20, 110, 220), outline=(10, 10, 10), width=5)
            draw.ellipse((250, 150, 430, 300), fill=(20, 110, 220), outline=(10, 10, 10), width=5)
            draw.ellipse((285, 180, 395, 270), fill=(0, 255, 0), outline=(245, 245, 245), width=5)
            image.save(raw_path)

            plan_path.write_text(
                json.dumps(
                    {
                        "board_size": list(CANVAS),
                        "slots": [
                            {
                                "slot_id": "icon_left",
                                "category": "icon",
                                "bbox": [0.08, 0.15, 0.20, 0.45],
                                "enabled": True,
                            },
                            {
                                "slot_id": "holder_right",
                                "category": "raster_holder",
                                "bbox": [0.22, 0.15, 0.34, 0.45],
                                "enabled": True,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            segmentation = extract_assets(raw_path, segmentation_dir, assets_dir=assets_dir)
            naming = name_segmented_assets(
                segmentation_dir / "manifest.json",
                plan_path,
                assets_dir,
            )
            render_segmented_reference(segmentation, segmentation_dir, reference_path)

            self.assertEqual(naming["split_fragment_count"], 0)
            self.assertEqual(naming["postprocessed_asset_count"], 2)
            self.assertTrue(all(record["fragment_count"] <= 1 for record in naming["assets"]))
            self.assertFalse((segmentation_dir / "assets").exists())
            self.assertEqual(
                sorted(path.name for path in assets_dir.glob("*.png")),
                ["asset_001.png", "asset_002.png"],
            )

            right = next(record for record in naming["assets"] if record["asset_id"] == "holder_right")
            right_asset = Image.open(assets_dir / right["path"]).convert("RGBA")
            crop_x0, crop_y0, _, _ = right["crop_bbox"]
            center = (340 - crop_x0, 225 - crop_y0)
            self.assertEqual(right_asset.getpixel(center)[3], 0)
            self.assertEqual(Image.open(reference_path).convert("RGB").getpixel((340, 225)), (246, 247, 245))


if __name__ == "__main__":
    unittest.main()
