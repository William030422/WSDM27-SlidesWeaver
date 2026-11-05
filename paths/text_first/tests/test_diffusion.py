from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from integration.config import load_config
from integration.diffusion import (
    build_reused_diffusion_prompts,
    run_reused_diffusion,
)


ROOT = Path(__file__).resolve().parents[1]


class DiffusionTests(unittest.TestCase):
    def test_prompt_contains_only_title_purpose_and_visual_logic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-diffusion-") as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            manifest = build_reused_diffusion_prompts(
                step2_dir=root / "step2",
                text_generation_dir=root / "text_generation",
                expression_logic_dir=root / "expression_logic",
                output_dir=root / "diffusion",
            )
            prompt = Path(manifest["slides"][0]["prompt"]).read_text()
            payload = json.loads(
                Path(manifest["slides"][0]["diffusion_input"]).read_text()
            )

        self.assertEqual(
            set(payload), {"slide_id", "title", "purpose", "visual_logic"}
        )
        self.assertIn("A final title", prompt)
        self.assertIn("Explain the decision loop.", prompt)
        self.assertIn("form a circle", prompt)
        self.assertNotIn("Detailed subtitle", prompt)
        self.assertNotIn("Detailed body copy", prompt)
        self.assertEqual(
            manifest["included_fields"], ["title", "purpose", "visual_logic"]
        )

    def test_mock_diffusion_preserves_reused_output_contract(self) -> None:
        config = load_config(ROOT)
        with tempfile.TemporaryDirectory(prefix="rl4ppt-diffusion-mock-") as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            manifest = run_reused_diffusion(
                step2_dir=root / "step2",
                text_generation_dir=root / "text_generation",
                expression_logic_dir=root / "expression_logic",
                output_dir=root / "diffusion",
                config=config,
                mock_diffusion=True,
            )
            with Image.open(root / "diffusion/skeletons/slide_001.png") as image:
                self.assertEqual(image.size, (1024, 576))
                self.assertEqual(image.mode, "RGB")

        self.assertEqual(manifest["status"], "succeeded")
        self.assertTrue(manifest["results"][0]["mock"])

    @staticmethod
    def _write_fixture(root: Path) -> None:
        for directory in (
            root / "step2",
            root / "text_generation",
            root / "expression_logic",
        ):
            directory.mkdir(parents=True)
        (root / "step2/deck_plan.json").write_text(
            json.dumps(
                {
                    "topic": "Feedback",
                    "audience": "students",
                    "deck_goal": "Teach feedback.",
                    "style": "editorial",
                    "slides": [
                        {
                            "slide_id": "slide_001",
                            "working_title": "A working title",
                            "rough_message": "Explain the decision loop.",
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
                            "title": "A final title",
                            "subtitle": "Detailed subtitle that must stay out.",
                            "text_01": "Detailed body copy that must stay out.",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (root / "expression_logic/expression_logic_plan.json").write_text(
            json.dumps(
                {
                    "slides": [
                        {
                            "slide_id": "slide_001",
                            "expression_logic": (
                                "Pair the ideas with visuals and let the pairs form a circle."
                            ),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
