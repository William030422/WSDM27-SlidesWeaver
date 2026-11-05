from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from integration.config import load_config
from v9_deck_planning.models import UserInput
from v9_deck_planning.pipeline import run_deck_planning
from v9_deck_planning.style_pipeline import run_style_asset_generation
from v9_deck_planning.style_image_backend import build_style_image_payload


ROOT = Path(__file__).resolve().parents[1]


class BackgroundFirstStyleAssetsTests(unittest.TestCase):
    def test_background_precedes_original_control_and_mask_in_payload(self) -> None:
        background = Path("background.png")
        control = Path("control.png")
        mask = Path("mask.png")
        with patch(
            "v9_deck_planning.style_image_backend.image_to_data_uri",
            side_effect=lambda path: str(path),
        ):
            payload = build_style_image_payload(
                prompt="asset prompt",
                background_path=background,
                control_path=control,
                mask_path=mask,
                image_model="gpt-image-2-official",
                size="1536x864",
                resolution="1k",
                quality="high",
            )
        self.assertEqual(payload["image_urls"], [str(background), str(control)])
        self.assertEqual(payload["mask_url"], str(mask))

    def test_background_is_the_only_increment_to_v9_asset_generation(self) -> None:
        config = load_config(ROOT)
        with tempfile.TemporaryDirectory(prefix="rl4ppt-reuse-step2-") as temporary:
            run_root = Path(temporary)
            step1 = run_root / "step1"
            step2 = run_root / "step2"
            user_input = UserInput(
                content="Create 5 slides teaching feedback control.",
                style="Exact Doraemon anime teaching style",
            )
            first = run_deck_planning(
                project_root=ROOT,
                user_input=user_input,
                output_dir=step1,
                llm_model=config.chat_model,
                mock_llm=True,
            )
            self.assertEqual(first["status"], "succeeded")
            second = run_style_asset_generation(
                project_root=ROOT,
                source_dir=step1,
                output_dir=step2,
                image_model=config.image_model,
                size=config.image_size.as_text(),
                resolution=config.image_resolution,
                quality=config.image_quality,
                mock_images=True,
            )
            self.assertEqual(second["status"], "succeeded")
            self.assertEqual(
                second["generation_order"],
                ["canonical_background", "style_asset_atlas"],
            )
            self.assertEqual(
                second["post_generation_order"],
                [
                    "segment_assets_and_write_contact_sheet",
                    "remove_enclosed_background_from_all_assets",
                    "compose_style_reference_from_postprocessed_assets",
                ],
            )
            self.assertTrue(second["style_asset_increment"]["original_control_and_mask_logic_preserved"])

            segmentation = json.loads(
                (step2 / "segmentation" / "manifest.json").read_text(encoding="utf-8")
            )
            naming = json.loads(
                (step2 / "assets" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse((step2 / "segmentation" / "assets").exists())
            self.assertEqual(
                len(list((step2 / "assets").glob("*.png"))),
                segmentation["asset_count"],
            )
            self.assertEqual(
                naming["postprocessed_asset_count"],
                segmentation["asset_count"],
            )
            self.assertTrue(
                all(record["path"].startswith("../assets/") for record in segmentation["assets"])
            )

            request = json.loads(
                (step2 / "style_reference.request.json").read_text(encoding="utf-8")
            )
            self.assertTrue(request["previous_generated_image_used"])
            self.assertEqual(
                Path(request["background_path"]).resolve(),
                (step2 / "canonical_background.png").resolve(),
            )
            self.assertEqual(len(request["payload"]["image_urls"]), 2)
            self.assertIn("mask_url", request["payload"])

            prompt = (step2 / "background_prompt.txt").read_text(encoding="utf-8")
            self.assertIn("at least 90 percent", prompt)
            self.assertIn("no more than 10 percent", prompt)
            self.assertIn("Match only the visual language", prompt)


if __name__ == "__main__":
    unittest.main()
