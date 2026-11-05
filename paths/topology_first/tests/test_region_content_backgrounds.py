from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageChops

from integration.config import load_config
from integration.final_background import (
    BACKGROUND_VARIANTS,
    COLOR_ALPHA,
    COLOR_FEATHER_PX,
    PROMPT_SCHEMA_VERSION,
    bind_visual_briefs,
    build_image_payload,
    choose_contrast_color,
    compile_generation_prompt,
    compile_region_coordinate_json,
    compile_region_coordinate_map,
    extract_core_style_commands,
    render_control_reference,
    run_background_set,
)
from integration.region_content import run_region_content_planning
from integration.region_content import (
    apply_text_presentation_policy,
    build_text_presentation_policy,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self) -> None:
        self.call_count = 0

    def vlm_json(
        self,
        *,
        model: str,
        prompt: str,
        image_path: Path,
        max_tokens: int,
    ) -> str:
        self.call_count += 1
        self.prompt = prompt
        self.image_path = image_path
        self.assert_image_exists = image_path.is_file()
        return json.dumps(
            {
                "slide_id": "slide_001",
                "visual_regions": [
                    {
                        "region_id": "visual_001",
                        "semantic_brief": "two contrasting temperature indicators with a visible gap",
                    }
                ],
                "text_regions": [
                    {
                        "region_id": "title_001",
                        "role": "title",
                        "lines": ["Find the Control Error"],
                    },
                    {
                        "region_id": "caption_001",
                        "role": "caption",
                        "lines": ["Target minus measurement"],
                    },
                ],
            }
        )


class RegionContentBackgroundTests(unittest.TestCase):
    def test_styled_policy_binds_one_holder_family_before_content(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-policy-") as temporary:
            root = Path(temporary)
            assets = root / "assets"
            assets.mkdir()
            Image.new("RGBA", (400, 200), (255, 255, 255, 255)).save(
                assets / "holder.png"
            )
            manifest_path = root / "asset_manifest.json"
            manifest = {
                "assets": [
                    {
                        "asset_id": "holder_wide",
                        "category": "raster_holder",
                        "path": "assets/holder.png",
                        "available": True,
                        "aspect_ratio": 2.0,
                        "allowed_roles": ["short_explanation"],
                        "text_safe_rect": {
                            "x": 0.1,
                            "y": 0.1,
                            "width": 0.8,
                            "height": 0.8,
                        },
                    }
                ]
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            inventories = {
                "slide_001": [
                    {
                        "region_id": "body_001",
                        "role": "standalone_text",
                        "bbox_px": [100, 100, 600, 240],
                        "canvas_px": [1536, 864],
                    }
                ],
                "slide_002": [
                    {
                        "region_id": "body_002",
                        "role": "standalone_text",
                        "bbox_px": [100, 100, 500, 250],
                        "canvas_px": [1536, 864],
                    }
                ],
            }
            policy = build_text_presentation_policy(
                deck_plan={"text_presentation_policy": "styled"},
                asset_manifest=manifest,
                asset_manifest_path=manifest_path,
                inventories=inventories,
            )
            first = apply_text_presentation_policy(inventories["slide_001"], policy)[0]
            second = apply_text_presentation_policy(inventories["slide_002"], policy)[0]

        self.assertEqual(policy["resolved"], "styled")
        self.assertEqual(
            policy["holder_family_by_role"]["standalone_text"], "holder_wide"
        )
        self.assertEqual(first["presentation_mode"], "raster_holder_text")
        self.assertEqual(
            first["holder_binding"]["holder_family"],
            second["holder_binding"]["holder_family"],
        )
        holder_bbox = first["holder_binding"]["holder_bbox_px"]
        self.assertAlmostEqual(holder_bbox[2] / holder_bbox[3], 2.0, places=3)
        self.assertEqual(first["holder_binding"]["aspect_ratio_change"], 0.0)
        self.assertLess(first["effective_text_bbox_px"][2], holder_bbox[2])

    def test_plans_every_region_once_with_one_model_call(self) -> None:
        config = load_config(ROOT)
        with tempfile.TemporaryDirectory(prefix="rl4ppt-region-content-") as temporary:
            run_dir = Path(temporary)
            self._write_fixture(run_dir)
            client = FakeClient()
            manifest = run_region_content_planning(
                project_root=ROOT,
                slide_spec_dir=run_dir / "slide_specs",
                topology_dir=run_dir / "topology",
                output_dir=run_dir / "region_content",
                slide_ids=["slide_001"],
                config=config,
                mock_llm=False,
                client_factory=lambda: client,
            )
            self.assertEqual(client.call_count, 1)
            self.assertTrue(client.assert_image_exists)
            self.assertEqual(client.image_path.name, "slide_001.png")
            self.assertEqual(client.image_path.parent.name, "postprocessed")
            self.assertEqual(manifest["model_call_count"], 1)
            self.assertIn("structured region inventory", client.prompt)
            self.assertIn("effective_text_bbox_px", client.prompt)
            self.assertIn("capacity_hint", client.prompt)
            self.assertIn("aspect_ratio", client.prompt)
            self.assertIn("observation, mechanism, and consequence", client.prompt)
            self.assertIn("Do not add", client.prompt)
            self.assertNotIn("exact text fitting belongs to Step 4", client.prompt)
            self.assertIn("bbox_px", client.prompt)
            plan = json.loads(
                (run_dir / "region_content/slides/slide_001/region_content_plan.json").read_text()
            )
            self.assertEqual(
                [item["region_id"] for item in plan["visual_regions"]],
                ["visual_001"],
            )
            self.assertEqual(
                {item["region_id"] for item in plan["text_regions"]},
                {"title_001", "caption_001"},
            )
            inventory = json.loads(
                (run_dir / "region_content/slides/slide_001/region_inventory.json").read_text()
            )
            text_inventory = [item for item in inventory if item["role"] != "visual"]
            self.assertTrue(all(item["effective_text_bbox_px"] for item in text_inventory))
            self.assertTrue(all(item["capacity_hint"] in {"compact", "standard", "roomy"} for item in text_inventory))

    def test_model_strategies_keep_planned_text_out_of_image_prompt(self) -> None:
        bindings = [
            {
                "component_id": "visual_001",
                "position": "middle-left",
                "bbox_px": [120, 240, 480, 360],
                "visual_brief": "two contrasting temperature indicators with a visible gap",
            }
        ]
        prompts = {
            item["slug"]: compile_generation_prompt(
                bindings,
                "Doraemon anime style",
                prompt_mode=item["prompt_mode"],
                control_mode=item["control_mode"],
                require_distinct=item["require_distinct"],
            )
            for item in BACKGROUND_VARIANTS
        }
        for prompt in prompts.values():
            self.assertIn("two contrasting temperature indicators", prompt)
            self.assertNotIn("Target minus measurement", prompt)
            self.assertIn("No panels", prompt)
            self.assertIn("SPATIAL REGISTRATION IS THE HIGHEST-PRIORITY", prompt)
            self.assertIn('"regions":{"A":[120,240,480,360]}', prompt)
            self.assertIn("REGION A:", prompt)
            self.assertIn("PRESERVE:", prompt)
            self.assertIn("STYLE:", prompt)
            self.assertNotIn("visual_001 middle-left", prompt)
        self.assertIn("fixed and independent placement regions", prompts["gpt_soft_contrast"])
        self.assertIn(
            "80%-opaque color fields with 2px feathered edges",
            prompts["nano_color_control"],
        )
        self.assertNotIn("blurred color fields", prompts["nano_color_control"])
        self.assertIn("strong local contrast", prompts["doubao_soft_contrast_distinct"])
        self.assertIn("hollow contrast-color rectangles", prompts["gpt_soft_contrast"])
        self.assertNotIn("white rectangles", prompts["gpt_soft_contrast"])

    def test_region_labels_and_compact_coordinates_follow_visual_count(self) -> None:
        for count in (3, 4):
            with self.subTest(count=count):
                plan = {
                    "visual_regions": [
                        {
                            "region_id": f"visual_{index}",
                            "semantic_brief": f"concept {index}",
                        }
                        for index in reversed(range(count))
                    ]
                }
                layout = {
                    "entities": [
                        {
                            "region_id": f"visual_{index}",
                            "postprocessed_bbox_px": [100 * index, 200, 80, 60],
                        }
                        for index in range(count)
                    ]
                }
                bindings = bind_visual_briefs(plan, layout)
                expected_labels = [chr(ord("A") + index) for index in range(count)]

                self.assertEqual(
                    [item["region_label"] for item in bindings], expected_labels
                )
                coordinates = compile_region_coordinate_map(bindings)
                self.assertEqual(list(coordinates["regions"]), expected_labels)
                compact = compile_region_coordinate_json(bindings)
                self.assertNotIn(" ", compact)
                self.assertNotIn("concept", compact)
                self.assertLess(len(compact), 160)
                prompt = compile_generation_prompt(
                    bindings,
                    "Use clean flat colors.",
                    prompt_mode="soft",
                    control_mode="contrast_frames",
                    require_distinct=False,
                )
                for label in expected_labels:
                    self.assertIn(f"REGION {label}:", prompt)
                self.assertNotIn(f"REGION {chr(ord('A') + count)}:", prompt)

    def test_style_core_commands_are_extracted_without_another_model(self) -> None:
        style = (
            "Exact Doraemon anime language: recognizable characters, Nobita's room, "
            "the Anywhere Door, and unrelated franchise scenery. "
            "Use blue, white, red, clean black outlines, light comic panels, and flat cel colors. "
            "Keep poses compact and visually coherent. "
            "No readable text or logos."
        )

        commands = extract_core_style_commands(style)

        self.assertIn("Exact Doraemon anime language", commands)
        self.assertIn("Use blue, white, red", commands)
        self.assertIn("Keep poses compact", commands)
        self.assertIn("No readable text", commands)
        self.assertIn("Nobita's room", commands)
        self.assertIn("Anywhere Door", commands)
        self.assertIn("comic panels", commands)

    def test_mock_background_set_writes_short_coordinate_artifact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-background-prompt-") as temporary:
            run_dir = Path(temporary)
            self._write_fixture(run_dir)
            step2 = run_dir / "step2"
            Image.new("RGB", (1536, 864), "white").save(
                step2 / "canonical_background.png"
            )
            (step2 / "design_system.json").write_text(
                json.dumps({"style_prompt": "Use clean black outlines and flat colors."}),
                encoding="utf-8",
            )
            plan_dir = run_dir / "region_content/slides/slide_001"
            plan_dir.mkdir(parents=True)
            (plan_dir / "region_content_plan.json").write_text(
                json.dumps(
                    {
                        "slide_id": "slide_001",
                        "visual_regions": [
                            {
                                "region_id": "visual_001",
                                "semantic_brief": "one compact control loop",
                            }
                        ],
                        "text_regions": [],
                    }
                ),
                encoding="utf-8",
            )

            manifest = run_background_set(
                project_root=ROOT,
                region_content_dir=run_dir / "region_content",
                step2_dir=step2,
                topology_dir=run_dir / "topology",
                output_dir=run_dir / "final_background",
                slide_ids=["slide_001"],
                config=None,  # type: ignore[arg-type]
                variant=BACKGROUND_VARIANTS[0],
                mock_images=True,
                resume_missing=False,
            )

            coordinate_path = run_dir / "final_background/slide_001/region_coordinates.json"
            coordinates = json.loads(coordinate_path.read_text(encoding="utf-8"))
            request = json.loads(
                (run_dir / "final_background/slide_001/request.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(manifest["image_call_count"], 0)
        self.assertEqual(manifest["prompt_schema_version"], PROMPT_SCHEMA_VERSION)
        self.assertEqual(coordinates["regions"], {"A": [300, 180, 900, 480]})
        self.assertTrue(request["mock"])
        self.assertEqual(request["prompt_schema_version"], PROMPT_SCHEMA_VERSION)

    def test_control_references_and_payloads_match_each_model(self) -> None:
        bindings = [
            {
                "component_id": "visual_001",
                "position": "middle-center",
                "bbox_px": [500, 250, 400, 300],
                "visual_brief": "a control loop",
            }
        ]
        with tempfile.TemporaryDirectory(prefix="rl4ppt-controls-") as temporary:
            root = Path(temporary)
            background = root / "background.png"
            Image.new("RGB", (1536, 864), (230, 230, 230)).save(background)
            contrast = root / "contrast.png"
            color = root / "color.png"
            render_control_reference(background, bindings, contrast, "contrast_frames")
            render_control_reference(background, bindings, color, "color_fields")
            with Image.open(background) as original, Image.open(contrast) as contrast_image:
                self.assertIsNotNone(ImageChops.difference(original, contrast_image).getbbox())
                self.assertEqual(
                    contrast_image.getpixel((500, 250)),
                    choose_contrast_color((230, 230, 230)),
                )
            with Image.open(background) as original, Image.open(color) as color_image:
                self.assertIsNotNone(ImageChops.difference(original, color_image).getbbox())
                overlay = choose_contrast_color((230, 230, 230))
                expected_center = Image.composite(
                    Image.new("RGB", (1, 1), overlay),
                    Image.new("RGB", (1, 1), (230, 230, 230)),
                    Image.new("L", (1, 1), round(255 * COLOR_ALPHA)),
                ).getpixel((0, 0))
                self.assertEqual(color_image.getpixel((700, 400)), expected_center)
                self.assertEqual(color_image.getpixel((497, 400)), (230, 230, 230))
                edge_distances = [
                    sum(abs(channel - 230) for channel in color_image.getpixel((x, 400)))
                    for x in (500, 501, 502)
                ]
                self.assertLess(edge_distances[0], edge_distances[1])
                self.assertLess(edge_distances[1], edge_distances[2])
            self.assertEqual(COLOR_ALPHA, 0.80)
            self.assertEqual(COLOR_FEATHER_PX, 2)

            payloads = {
                variant["model"]: build_image_payload(
                    model=variant["model"],
                    prompt="test",
                    reference_path=contrast,
                )
                for variant in BACKGROUND_VARIANTS
            }
            self.assertEqual(payloads["gpt-image-2-official"]["size"], "1536x864")
            self.assertEqual(payloads["gemini-3-pro-image-preview"]["resolution"], "1K")
            self.assertFalse(payloads["doubao-seedream-5-0-pro"]["watermark"])

    def _write_fixture(self, run_dir: Path) -> None:
        step2 = run_dir / "step2"
        slide_specs = run_dir / "slide_specs/slides/slide_001"
        calibrated = run_dir / "topology/calibrated_topology/slide_001"
        layouts = run_dir / "topology/postprocessed/layouts"
        postprocessed = run_dir / "topology/postprocessed"
        for directory in (step2, slide_specs, calibrated, layouts, postprocessed):
            directory.mkdir(parents=True, exist_ok=True)
        (step2 / "deck_plan.json").write_text(
            json.dumps(
                {
                    "topic": "Feedback control",
                    "audience": "Beginning engineering students",
                    "deck_goal": "Explain a feedback loop.",
                    "slides": [
                        {
                            "slide_id": "slide_001",
                            "rough_message": "Compare target and measurement to find error.",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (slide_specs / "slide_spec.json").write_text(
            json.dumps(
                {
                    "slide_id": "slide_001",
                    "title": "Find the Error",
                    "message": "Compare target and measurement to find error.",
                }
            ),
            encoding="utf-8",
        )
        (calibrated / "calibrated_topology.json").write_text(
            json.dumps(
                {
                    "entities": [
                        {
                            "region_id": "title_001",
                            "canonical_role": "title",
                            "parent_component_id": "c_title",
                        },
                        {
                            "region_id": "visual_001",
                            "canonical_role": "visual",
                            "parent_component_id": "c_visual",
                        },
                        {
                            "region_id": "caption_001",
                            "canonical_role": "caption",
                            "parent_component_id": "c_visual",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        (layouts / "slide_001.json").write_text(
            json.dumps(
                {
                    "entities": [
                        {"region_id": "title_001", "postprocessed_bbox_px": [60, 30, 1400, 100]},
                        {"region_id": "visual_001", "postprocessed_bbox_px": [300, 180, 900, 480]},
                        {"region_id": "caption_001", "postprocessed_bbox_px": [500, 700, 500, 100]},
                    ]
                }
            ),
            encoding="utf-8",
        )
        Image.new("RGB", (1536, 864), "white").save(postprocessed / "slide_001.png")
