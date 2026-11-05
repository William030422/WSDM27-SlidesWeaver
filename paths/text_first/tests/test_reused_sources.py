from __future__ import annotations

import json
import hashlib
import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from v9_deck_planning.style_pipeline import build_background_prompt, run_style_asset_generation
from v9_deck_planning.models import DeckPlan, UserInput


ROOT = Path(__file__).resolve().parents[1]
REFERENCES = json.loads((ROOT / "tests/reference_sources.json").read_text())


class ArchitectureCleanupTests(unittest.TestCase):
    def test_background_prompt_matches_the_clean_branch_template(self) -> None:
        prompt = build_background_prompt(
            UserInput(content="Explain feedback.", style="Minecraft"),
            DeckPlan(
                topic="Feedback",
                audience="students",
                deck_goal="Explain feedback.",
                style="Minecraft",
                slides=[
                    {
                        "slide_id": "slide_001",
                        "working_title": "Feedback",
                        "rough_message": "Explain feedback.",
                    }
                ],
            ),
        )
        self.assertEqual(
            prompt,
            """Create one neat, reusable 16:9 background for an entire presentation deck.

Match only the visual language of this style -- palette, materials, line work, lighting, and surface treatment:
Minecraft

Deck topic, as a weak atmospheric hint only:
Feedback

This is a background, not an illustration. Keep at least 90 percent of the full canvas as one broad, contiguous,
low-detail negative-space field suitable for later text and teaching visuals. All decoration combined must occupy
no more than 10 percent of the canvas. Keep it at the thin outer perimeter, leaving quiet space across the whole canvas.
Use at most two small abstract edge accents. Do not draw characters, people, rooms, neighborhoods, gadgets, doors,
vehicles, scenes, or other recognizable objects. Any decoration must remain sparse, peripheral, and subordinate.
Keep the composition clean, orderly, quiet, and spacious, with restrained decoration and broad low-detail areas.
Keep the background largely free of semantic information. Do not depict topic-specific objects, teaching entities,
diagrams, charts, data, or semantic relationships.
Do not include text, labels, slide-specific content, fixed cards, panels, placeholders, logos, or watermarks.

Return only the background image.
""",
        )

    def test_removed_architecture_has_no_runtime_entrypoints(self) -> None:
        removed_files = (
            ROOT / "run_content_unit_binding.py",
            ROOT / "integration/topology.py",
            ROOT / "integration/semantic_topology.py",
            ROOT / "integration/region_content.py",
            ROOT / "integration/local_visual_composite.py",
            ROOT / "integration/slide_spec.py",
            ROOT / "integration/visual_prompt_planning.py",
        )
        self.assertTrue(all(not path.exists() for path in removed_files))
        self.assertTrue((ROOT / "integration/diffusion.py").is_file())
        self.assertTrue((ROOT / "reused/diffusion").is_dir())
        for directory in (ROOT / "reused/topology", ROOT / "reused/text_placement"):
            self.assertEqual(list(directory.rglob("*.py")), [])
            self.assertEqual(list(directory.rglob("*.mjs")), [])
            self.assertEqual(list(directory.rglob("*.md")), [])

    def test_step2_generates_only_background_and_style_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-step2-background-") as temporary:
            root = Path(temporary)
            step1 = root / "step1"
            output = root / "step2"
            self._write_step1(step1)
            manifest = run_style_asset_generation(
                project_root=ROOT,
                source_dir=step1,
                output_dir=output,
                mock_images=True,
            )

            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(manifest["stage"], "style_background_generation")
            self.assertEqual(manifest["generation_order"], ["canonical_background"])
            self.assertFalse(manifest["asset_atlas_generated"])
            self.assertFalse(manifest["segmentation_executed"])
            self.assertEqual(manifest["image_call_count"], 0)
            self.assertTrue((output / "canonical_background.png").is_file())
            self.assertTrue((output / "design_system.json").is_file())
            self.assertFalse((output / "asset_manifest.json").exists())
            self.assertFalse((output / "style_reference.png").exists())
            self.assertFalse((output / "segmentation").exists())

    def test_diffusion_runner_preserves_inference_and_uses_portable_model_path(self) -> None:
        local = ROOT / "reused/diffusion/generate_slides.py"
        self.assertEqual(hashlib.sha256(local.read_bytes()).hexdigest(),
                         REFERENCES["reused/diffusion/generate_slides.py"]["sha256"])
        tree = ast.parse(local.read_text())
        model_path = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "MODEL_PATH" for t in node.targets))
        tree.body = [node for node in tree.body if node is not model_path and not (
            isinstance(node, ast.Import) and [alias.name for alias in node.names] == ["os"])]
        algorithm_hash = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
        self.assertEqual(algorithm_hash, REFERENCES["reused/diffusion/generate_slides.py"]["inference_ast_sha256"])
        code = compile(ast.Module(body=[model_path], type_ignores=[]), str(local), "exec")
        with patch.dict(os.environ, {"DIFFUSION_MODEL_PATH": "~/custom-model"}):
            namespace = {"Path": Path, "os": os}
            exec(code, namespace)
            self.assertEqual(namespace["MODEL_PATH"], Path.home() / "custom-model")
        with patch.dict(os.environ, {}, clear=True):
            namespace = {"Path": Path, "os": os}
            exec(code, namespace)
            self.assertEqual(namespace["MODEL_PATH"], Path.home() / "models" / "Tongyi-MAI_Z-Image-Turbo")

    @staticmethod
    def _write_step1(output: Path) -> None:
        output.mkdir(parents=True)
        (output / "run_manifest.json").write_text(
            json.dumps({"stage": "deck_level_planning", "status": "succeeded"}),
            encoding="utf-8",
        )
        (output / "input.json").write_text(
            json.dumps({"content": "Explain feedback control.", "style": "clean editorial illustration"}),
            encoding="utf-8",
        )
        (output / "deck_plan.json").write_text(
            json.dumps(
                {
                    "topic": "Feedback control",
                    "audience": "students",
                    "deck_goal": "Explain a feedback loop.",
                    "style": "clean editorial illustration",
                    "slides": [
                        {
                            "slide_id": "slide_001",
                            "working_title": "Observe and update",
                            "rough_message": "Compare the result and improve the action.",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
