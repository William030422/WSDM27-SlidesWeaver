from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from integration.config import load_config
from integration.diffusion import build_reused_diffusion_prompts, run_reused_diffusion
from integration.final_background import bind_visual_briefs
from integration.slide_spec import run_slide_spec_bridge
from integration.text_assembly import _adapt_region_content


ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self) -> None:
        self.call_count = 0

    def chat_json(self, *, model: str, prompt: str, max_tokens: int) -> str:
        self.call_count += 1
        self.prompt = prompt
        return json.dumps(
            {
                "slide_id": "slide_001",
                "title": "See the error",
                "message": "Feedback starts by comparing target and measurement.",
                "visual_intent": (
                    "A detective-style comparison between a target signal and a measured signal."
                ),
                "editable_content": [
                    {"role": "title", "text": "See the error"},
                    {
                        "role": "body",
                        "text": "Compare target and measurement to find the error.",
                    },
                ],
            }
        )


class SlideSpecDiffusionTests(unittest.TestCase):
    def test_one_lightweight_call_produces_only_slide_spec_fields(self) -> None:
        config = load_config(ROOT)
        with tempfile.TemporaryDirectory(prefix="rl4ppt-slide-spec-") as temporary:
            run_dir = Path(temporary)
            step2 = run_dir / "step2"
            self._write_step2_fixture(step2)
            client = FakeClient()
            manifest = run_slide_spec_bridge(
                project_root=ROOT,
                step2_dir=step2,
                output_dir=run_dir / "slide_specs",
                config=config,
                mock_llm=False,
                client_factory=lambda: client,
            )

            self.assertEqual(client.call_count, 1)
            self.assertEqual(manifest["model_call_count"], 1)
            self.assertIn("visual_intent", client.prompt)
            self.assertIn("open-ended", client.prompt)
            self.assertNotIn("Available Step 2 asset roles", client.prompt)
            self.assertNotIn("Exact shared style", client.prompt)
            spec = json.loads(
                (run_dir / "slide_specs/slides/slide_001/slide_spec.json").read_text()
            )
            self.assertEqual(
                set(spec),
                {"slide_id", "title", "message", "visual_intent"},
            )
            self.assertNotIn("claims", spec)
            self.assertNotIn("relations", spec)
            self.assertEqual(
                manifest["schema_fields"],
                ["slide_id", "title", "message", "visual_intent"],
            )
            self.assertEqual(manifest["quality_status"], "strict_pass")

    def test_diffusion_prompt_and_mock_skeleton_use_slide_spec(self) -> None:
        config = load_config(ROOT)
        with tempfile.TemporaryDirectory(prefix="rl4ppt-diffusion-") as temporary:
            run_dir = Path(temporary)
            step2 = run_dir / "step2"
            self._write_step2_fixture(step2)
            run_slide_spec_bridge(
                project_root=ROOT,
                step2_dir=step2,
                output_dir=run_dir / "slide_specs",
                config=config,
                mock_llm=True,
            )
            prompt_manifest = build_reused_diffusion_prompts(
                slide_spec_dir=run_dir / "slide_specs",
                output_dir=run_dir / "prompts_only",
            )
            prompt = Path(prompt_manifest["slides"][0]["prompt"]).read_text()
            self.assertIn("Finding the control error", prompt)
            self.assertIn("A feedback system starts", prompt)
            self.assertIn("Rough visual intent", prompt)
            self.assertIn("Find a memorable visual way", prompt)
            self.assertIn("Choose any visual grammar", prompt)
            self.assertNotIn("single cohesive editorial illustration", prompt)
            self.assertNotIn("generous whitespace", prompt)
            self.assertNotIn("Ideas available to draw from", prompt)
            self.assertNotIn("model-based RL", prompt)

            manifest = run_reused_diffusion(
                slide_spec_dir=run_dir / "slide_specs",
                output_dir=run_dir / "diffusion",
                config=config,
                mock_diffusion=True,
            )
            self.assertEqual(manifest["status"], "succeeded")
            self.assertTrue(manifest["results"][0]["mock"])
            with Image.open(run_dir / "diffusion/skeletons/slide_001.png") as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (1024, 576))

    def test_downstream_adapters_consume_slide_spec_fields(self) -> None:
        region_content = {
            "slide_id": "slide_001",
            "visual_regions": [
                {"region_id": "visual_1", "semantic_brief": "target and measurement"},
                {"region_id": "visual_2", "semantic_brief": "visible control error"},
            ],
            "text_regions": [
                {"region_id": "title_1", "role": "title", "lines": ["See the error"]},
                {
                    "region_id": "caption_1",
                    "role": "caption",
                    "lines": ["Target minus measurement"],
                },
            ],
        }
        layout = {
            "entities": [
                {"region_id": "visual_1", "postprocessed_bbox_px": [0, 0, 100, 100]},
                {"region_id": "visual_2", "postprocessed_bbox_px": [100, 0, 100, 100]},
            ]
        }

        bindings = bind_visual_briefs(region_content, layout)
        self.assertEqual(bindings[0]["visual_brief"], "target and measurement")
        self.assertEqual(bindings[1]["visual_brief"], "visible control error")
        content = _adapt_region_content(region_content)
        self.assertEqual(
            [item["content_id"] for item in content],
            ["content_title_1", "content_caption_1"],
        )
        self.assertEqual([item["role"] for item in content], ["title", "caption"])
        self.assertEqual(
            [item["target_region_id"] for item in content],
            ["title_1", "caption_1"],
        )

        multi_point = _adapt_region_content(
            {
                "text_regions": [
                    {
                        "region_id": "body_1",
                        "role": "standalone_text",
                        "lines": ["Observe the state", "Choose an action"],
                    }
                ]
            }
        )[0]
        self.assertEqual(multi_point["text"], "Observe the state\nChoose an action")
        self.assertEqual(
            multi_point["bullet_items"], ["Observe the state", "Choose an action"]
        )
        self.assertNotIn("- ", multi_point["text"])
        caption_points = _adapt_region_content(
            {
                "text_regions": [
                    {
                        "region_id": "caption_2",
                        "role": "caption",
                        "lines": ["First observation", "Second observation"],
                    }
                ]
            }
        )[0]
        self.assertEqual(
            caption_points["bullet_items"],
            ["First observation", "Second observation"],
        )

    def _write_step2_fixture(self, step2: Path) -> None:
        step2.mkdir(parents=True)
        payloads = {
            "run_manifest.json": {
                "stage": "style_asset_generation",
                "status": "succeeded",
            },
            "input.json": {
                "content": "Teach feedback control through Doraemon.",
                "style": "Exact Doraemon anime teaching style",
            },
            "deck_plan.json": {
                "stage": "deck_level_planning",
                "topic": "Feedback control",
                "audience": "Beginning engineering students",
                "deck_goal": "Teach the basic feedback loop.",
                "narrative_arc": "Start with error and then close the loop.",
                "style": "Exact Doraemon anime teaching style",
                "slide_count": 1,
                "slides": [
                    {
                        "slide_id": "slide_001",
                        "working_title": "Finding the control error",
                        "purpose": "framing",
                        "rough_message": (
                            "A feedback system starts by comparing a target with a measurement."
                        ),
                        "depends_on": [],
                    }
                ],
            },
            "design_system.json": {
                "style_prompt": "Exact Doraemon anime teaching style",
                "canvas": {"width": 1536, "height": 864},
            },
            "asset_manifest.json": {
                "assets": [
                    {
                        "available": True,
                        "category": "icon",
                        "allowed_roles": ["concept"],
                    }
                ]
            },
        }
        for name, payload in payloads.items():
            (step2 / name).write_text(json.dumps(payload), encoding="utf-8")
        Image.new("RGB", (1536, 864), "white").save(
            step2 / "canonical_background.png"
        )
