from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from integration.background_decomposition import decompose_background


class BackgroundDecompositionTests(unittest.TestCase):
    def test_extracts_region_seeded_modules_without_external_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = np.full((120, 200, 3), 240, dtype=np.uint8)
            final = canonical.copy()
            final[35:85, 25:75] = [210, 35, 40]
            final[30:90, 125:180] = [20, 90, 210]
            canonical_path = root / "canonical.png"
            final_path = root / "final.png"
            Image.fromarray(canonical, mode="RGB").save(canonical_path)
            Image.fromarray(final, mode="RGB").save(final_path)
            coordinates_path = root / "regions.json"
            coordinates_path.write_text(
                json.dumps(
                    {
                        "canvas": [200, 120],
                        "regions": {
                            "A": [20, 30, 60, 60],
                            "B": [120, 25, 65, 70],
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = decompose_background(
                canonical_path=canonical_path,
                final_path=final_path,
                region_coordinates_path=coordinates_path,
                output_dir=root / "output",
            )

            self.assertEqual(result["module_count"], 2)
            self.assertEqual(
                [module["region_label"] for module in result["modules"]],
                ["A", "B"],
            )
            for module in result["modules"]:
                asset_path = Path(module["asset_path"])
                self.assertTrue(asset_path.is_file())
                with Image.open(asset_path) as asset:
                    self.assertEqual(asset.mode, "RGBA")
                    self.assertGreater(asset.getchannel("A").getbbox()[2], 0)

    def test_identical_background_produces_no_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = Image.new("RGB", (200, 120), "white")
            canonical_path = root / "canonical.png"
            final_path = root / "final.png"
            image.save(canonical_path)
            image.save(final_path)
            coordinates_path = root / "regions.json"
            coordinates_path.write_text(
                json.dumps({"regions": {"A": [25, 25, 100, 70]}}),
                encoding="utf-8",
            )

            result = decompose_background(
                canonical_path=canonical_path,
                final_path=final_path,
                region_coordinates_path=coordinates_path,
                output_dir=root / "output",
            )

            self.assertEqual(result["module_count"], 0)


if __name__ == "__main__":
    unittest.main()
