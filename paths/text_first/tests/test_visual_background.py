from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from integration.config import load_config
from integration.final_background import (
    BOARD_MARGIN,
    CANVAS,
    SLOT_GUTTER,
    _assign_components_to_slots,
    _fit_bbox_for_aspect,
    _generate_or_resume,
    _plan_recovery_slots,
    _visual_generation_prompt,
    build_asset_board_image_payload,
    build_asset_board_plan,
    build_generation_objects,
    compile_generation_prompt,
    extract_chroma_layers,
    plan_asset_board_slots,
    render_asset_board_controls,
    render_mock_asset_board,
    render_visual_structure_reference_board,
    run_background_set,
    select_key_color,
)
from v9_deck_planning.apimart import ApiMartImageTaskFailure
from v9_deck_planning.green_background_removal import (
    _despill_key_color,
    _key_spill_strength,
)

ROOT = Path(__file__).resolve().parents[1]


class VisualBackgroundTests(unittest.TestCase):
    def test_despill_uses_sampled_key_chroma_without_hardcoded_green(self) -> None:
        key = np.array([250.0, 2.0, 249.0], dtype=np.float32)
        pixels = np.array(
            [[[250.0, 2.0, 249.0], [20.0, 180.0, 30.0], [220.0, 20.0, 20.0]]],
            dtype=np.float32,
        )
        partial = np.ones((1, 3), dtype=bool)

        strength = _key_spill_strength(pixels, key)
        despilled = _despill_key_color(pixels, key, partial)

        self.assertGreater(strength[0, 0], 0.95)
        self.assertEqual(strength[0, 1], 0.0)
        self.assertEqual(strength[0, 2], 0.0)
        self.assertLess(float(np.ptp(despilled[0, 0])), 3.0)
        np.testing.assert_array_equal(despilled[0, 1], pixels[0, 1])
        np.testing.assert_array_equal(despilled[0, 2], pixels[0, 2])

    def test_despill_supports_a_sampled_non_primary_key_color(self) -> None:
        key = np.array([240.0, 224.0, 192.0], dtype=np.float32)
        pixels = np.array(
            [[[240.0, 224.0, 192.0], [25.0, 170.0, 35.0]]],
            dtype=np.float32,
        )

        despilled = _despill_key_color(
            pixels,
            key,
            np.ones((1, 2), dtype=bool),
        )

        self.assertLess(float(np.ptp(despilled[0, 0])), 3.0)
        np.testing.assert_array_equal(despilled[0, 1], pixels[0, 1])

    def test_visual_prompt_removes_placeholder_language(self) -> None:
        prompt = _visual_generation_prompt(
            "Empty neutral visual placeholder for observing a Minecraft world state."
        )
        self.assertIn("actual content-filled, text-free visual", prompt)
        self.assertIn("observing a Minecraft world state", prompt)
        self.assertNotIn("Empty neutral visual placeholder", prompt)
        self.assertIn("Do not generate a blank frame", prompt)

    def test_missing_slot_recovery_uses_a_large_independent_slot(self) -> None:
        planned = _plan_recovery_slots(
            [
                {
                    "object_id": "visual_current_state",
                    "kind": "visual",
                    "bbox_px": [176, 374, 120, 114],
                    "prompt": "Generate the current state.",
                }
            ]
        )
        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0]["object_id"], "visual_current_state")
        self.assertGreater(planned[0]["slot_bbox_px"][2], 400)
        self.assertGreater(planned[0]["slot_bbox_px"][3], 400)
        self.assertAlmostEqual(
            planned[0]["slot_bbox_px"][2] / planned[0]["slot_bbox_px"][3],
            120 / 114,
            delta=0.01,
        )

    def test_excessive_drift_leaves_slot_missing_for_local_recovery(self) -> None:
        objects = [
            {"object_id": "visual_a", "slot_bbox_px": [60, 60, 900, 300]},
            {"object_id": "holder_title", "slot_bbox_px": [60, 430, 900, 90]},
        ]
        records = [
            {
                "asset_id": "asset_visual",
                "component_bbox": [50, 50, 970, 380],
                "centroid": [510, 215],
            },
            {
                "asset_id": "asset_holder",
                "component_bbox": [50, 630, 970, 820],
                "centroid": [510, 725],
            },
        ]

        assignments = _assign_components_to_slots(records, objects)

        self.assertEqual(
            [item["asset_id"] for item in assignments["visual_a"]],
            ["asset_visual"],
        )
        self.assertEqual(
            [item["asset_id"] for item in assignments["holder_title"]],
            [],
        )

    def test_large_overlapping_components_do_not_claim_the_same_slot(self) -> None:
        objects = [
            {"object_id": "holder", "slot_bbox_px": [64, 64, 790, 181]},
            {"object_id": "reward", "slot_bbox_px": [918, 64, 220, 114]},
            {"object_id": "risk", "slot_bbox_px": [1202, 64, 220, 114]},
            {"object_id": "shelter", "slot_bbox_px": [64, 309, 220, 114]},
            {"object_id": "current", "slot_bbox_px": [348, 309, 120, 114]},
            {"object_id": "selected", "slot_bbox_px": [532, 309, 134, 88]},
        ]
        records = [
            {
                "asset_id": "a1",
                "component_bbox": [59, 72, 669, 387],
                "centroid": [363, 229],
            },
            {
                "asset_id": "a2",
                "component_bbox": [713, 77, 1071, 367],
                "centroid": [897, 223],
            },
            {
                "asset_id": "a3",
                "component_bbox": [1106, 78, 1479, 380],
                "centroid": [1297, 226],
            },
            {
                "asset_id": "a4",
                "component_bbox": [54, 481, 491, 760],
                "centroid": [275, 617],
            },
            {
                "asset_id": "a5",
                "component_bbox": [534, 498, 759, 747],
                "centroid": [647, 627],
            },
            {
                "asset_id": "a6",
                "component_bbox": [818, 516, 1026, 740],
                "centroid": [922, 636],
            },
        ]

        assignments = _assign_components_to_slots(records, objects)

        self.assertEqual([item["asset_id"] for item in assignments["holder"]], ["a1"])
        self.assertEqual([item["asset_id"] for item in assignments["reward"]], ["a2"])
        self.assertEqual([item["asset_id"] for item in assignments["risk"]], ["a3"])
        for missing in ("shelter", "current", "selected"):
            self.assertEqual(assignments[missing], [])
        self.assertEqual(
            {
                record["asset_id"]
                for values in assignments.values()
                for record in values
            },
            {"a1", "a2", "a3"},
        )

    def test_visual_is_aspect_fit_inside_its_html_region(self) -> None:
        placed = _fit_bbox_for_aspect((1000, 600), [400, 300, 600, 200])
        self.assertAlmostEqual(placed[2] / placed[3], 1000 / 600, delta=0.01)
        self.assertLessEqual(placed[2], 600)
        self.assertLessEqual(placed[3], 200)

    def test_html_sized_objects_are_repacked_with_wide_non_overlapping_gutters(
        self,
    ) -> None:
        objects = plan_asset_board_slots(
            build_generation_objects(
                {
                    "visual_regions": [
                        {
                            "region_id": "visual_a",
                            "bbox_px": [600, 180, 330, 240],
                            "visual_brief": "Show state.",
                        },
                        {
                            "region_id": "visual_b",
                            "bbox_px": [970, 180, 330, 240],
                            "visual_brief": "Show prediction.",
                        },
                    ],
                    "text_regions": [],
                    "holders": [
                        {
                            "holder_id": "holder_body_001",
                            "bbox_px": [60, 210, 400, 180],
                            "text_region_id": "body_001",
                        },
                        {
                            "holder_id": "holder_body_002",
                            "bbox_px": [60, 450, 400, 180],
                            "text_region_id": "body_002",
                        },
                    ],
                },
            )
        )

        self.assertEqual(
            [item["kind"] for item in objects], ["visual", "visual", "holder", "holder"]
        )
        self.assertEqual(
            [item["object_id"] for item in objects[:2]],
            ["visual_group_visual_a", "visual_group_visual_b"],
        )
        scales = {item["board_scale"] for item in objects}
        self.assertEqual(len(scales), 1)
        for item in objects:
            source_width, source_height = item["bbox_px"][2:]
            slot_width, slot_height = item["slot_bbox_px"][2:]
            self.assertAlmostEqual(
                slot_width / slot_height, source_width / source_height, delta=0.02
            )
            self.assertGreaterEqual(item["slot_bbox_px"][0], BOARD_MARGIN)
            self.assertGreaterEqual(item["slot_bbox_px"][1], BOARD_MARGIN)
        for index, left in enumerate(objects):
            for right in objects[index + 1 :]:
                self.assertTrue(
                    _separated_by(
                        left["slot_bbox_px"], right["slot_bbox_px"], SLOT_GUTTER
                    )
                )

        prompt = compile_generation_prompt(
            objects, "clean editorial illustration", "#00ff00"
        )
        self.assertIn("canonical deck background", prompt)
        self.assertIn("exactly one complete HTML visual region", prompt)
        self.assertIn("one downstream asset", prompt)
        self.assertIn("Never fragment it into disconnected mini-assets", prompt)
        self.assertIn("actual\nsemantic imagery", prompt)
        self.assertNotIn("complete outer silhouette", prompt)
        self.assertIn("creative blank text surface", prompt)
        self.assertIn("Holders must ignore reference image 3", prompt)
        self.assertIn("wide chroma gutters", prompt)
        self.assertIn(
            "Do not let any asset blend into or visually merge with the #00ff00 contrast-color background.",
            prompt,
        )
        self.assertIn("holder_body_001", prompt)

    def test_build_generation_objects_preserves_three_parallel_visual_groups(
        self,
    ) -> None:
        layout = {
            "visual_regions": [
                {
                    "region_id": f"visual_{index}",
                    "bbox_px": [100 + 360 * index, 200, 300, 260],
                    "visual_brief": f"Visual paired only with text {index}.",
                }
                for index in range(3)
            ],
            "visual_generation_groups": [
                {
                    "group_id": f"pair_{index}",
                    "bbox_px": [100 + 360 * index, 200, 300, 260],
                    "semantic_description": f"Visual paired only with text {index}.",
                    "source_region_ids": [f"visual_{index}"],
                    "relation_type": "text_visual_pair",
                    "structure_reference": None,
                }
                for index in range(3)
            ],
            "holders": [],
        }
        objects = plan_asset_board_slots(build_generation_objects(layout))
        self.assertEqual(len(objects), 3)
        self.assertEqual(
            [item["object_id"] for item in objects],
            ["visual_group_visual_0", "visual_group_visual_1", "visual_group_visual_2"],
        )
        self.assertEqual(
            [item["source_region_ids"] for item in objects],
            [["visual_0"], ["visual_1"], ["visual_2"]],
        )

    def test_build_generation_objects_never_merges_declared_visual_group(self) -> None:
        layout = {
            "visual_regions": [
                {
                    "region_id": region_id,
                    "bbox_px": [100 + index * 380, 180, 340, 420],
                    "visual_brief": f"Show {region_id} as a complete visual.",
                }
                for index, region_id in enumerate(("real", "model", "future"))
            ],
            "visual_generation_groups": [
                {
                    "group_id": "flow_main",
                    "bbox_px": [100, 180, 1100, 420],
                    "semantic_description": "Reality leads to a model and an imagined future.",
                    "source_region_ids": ["real", "model", "future"],
                    "relation_type": "progression",
                    "structure_reference": None,
                }
            ],
            "holders": [],
        }
        objects = build_generation_objects(layout)
        self.assertEqual(len(objects), 3)
        self.assertEqual(
            [item["object_id"] for item in objects],
            ["visual_group_real", "visual_group_model", "visual_group_future"],
        )
        self.assertEqual(
            [item["source_region_ids"] for item in objects],
            [["real"], ["model"], ["future"]],
        )

    def test_structure_reference_board_populates_visual_slots_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-structure-board-") as temporary:
            root = Path(temporary)
            source = root / "source.png"
            output = root / "board.png"
            Image.new("RGB", (200, 100), (120, 140, 160)).save(source)
            objects = [
                {
                    "object_id": "flow",
                    "kind": "visual",
                    "slot_bbox_px": [100, 100, 400, 200],
                    "structure_reference": str(source),
                },
                {
                    "object_id": "holder",
                    "kind": "holder",
                    "slot_bbox_px": [700, 100, 400, 200],
                    "structure_reference": str(source),
                },
            ]
            rendered = render_visual_structure_reference_board(objects, output)
            with Image.open(output).convert("RGB") as board:
                visual_pixel = board.getpixel((300, 200))
                holder_pixel = board.getpixel((900, 200))
        self.assertTrue(rendered)
        self.assertNotEqual(visual_pixel, (255, 255, 255))
        self.assertEqual(holder_pixel, (255, 255, 255))

    def test_control_board_uses_key_color_and_html_sized_mask_holes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-control-") as temporary:
            root = Path(temporary)
            control = root / "control.png"
            mask = root / "mask.png"
            objects = [
                {
                    "object_id": "visual_a",
                    "slot_bbox_px": [100, 120, 200, 160],
                }
            ]
            render_asset_board_controls(objects, "#ff00ff", control, mask)
            with Image.open(control) as control_image, Image.open(mask) as mask_image:
                self.assertEqual(control_image.getpixel((0, 0)), (255, 0, 255))
                self.assertEqual(control_image.getpixel((100, 120)), (255, 0, 255))
                self.assertEqual(mask_image.getpixel((99, 119))[3], 255)
                self.assertEqual(mask_image.getpixel((100, 120))[3], 0)
                self.assertEqual(mask_image.getpixel((299, 279))[3], 0)
                self.assertEqual(mask_image.getpixel((300, 280))[3], 255)

    def test_payload_uses_background_as_first_reference_image(self) -> None:
        config = load_config(ROOT)
        with tempfile.TemporaryDirectory(prefix="rl4ppt-payload-") as temporary:
            root = Path(temporary)
            background = root / "background.png"
            control = root / "control.png"
            mask = root / "mask.png"
            Image.new("RGB", CANVAS, (10, 20, 30)).save(background)
            Image.new("RGB", CANVAS, (0, 255, 0)).save(control)
            Image.new("RGBA", CANVAS, (0, 255, 0, 255)).save(mask)
            payload = build_asset_board_image_payload(
                model=config.image_model,
                prompt="Generate an isolated asset.",
                background_path=background,
                control_path=control,
                mask_path=mask,
                config=config,
            )

        self.assertEqual(len(payload["image_urls"]), 2)
        self.assertNotEqual(payload["image_urls"][0], payload["image_urls"][1])
        self.assertTrue(payload["image_urls"][0].startswith("data:image/png;base64,"))
        self.assertIn("mask_url", payload)

    def test_clean_branch_extraction_emits_transparent_centered_layers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rl4ppt-chroma-") as temporary:
            root = Path(temporary)
            background = root / "background.png"
            generated = root / "generated.png"
            Image.new("RGB", CANVAS, "white").save(background)
            objects = plan_asset_board_slots(
                [
                    {
                        "object_id": "visual_a",
                        "kind": "visual",
                        "bbox_px": [500, 200, 420, 260],
                        "prompt": "Create a visual.",
                        "source_region_ids": ["visual_a"],
                    },
                    {
                        "object_id": "holder_body_001",
                        "kind": "holder",
                        "bbox_px": [80, 240, 320, 160],
                        "prompt": "Create a blank holder.",
                        "text_region_id": "body_001",
                    },
                ]
            )
            render_mock_asset_board(objects, "#00ff00", generated)
            plan_path = root / "asset_board_plan.json"
            plan_path.write_text(
                json.dumps(build_asset_board_plan(objects, "#00ff00")), encoding="utf-8"
            )
            manifest = extract_chroma_layers(
                generated_path=generated,
                canonical_background=background,
                objects=objects,
                asset_plan_path=plan_path,
                segmentation_dir=root / "segmentation",
                extracted_assets_dir=root / "extracted_assets",
                output_dir=root / "layers",
            )

            self.assertEqual(
                manifest["extraction"], "clean_branch_slot_aware_chroma_removal"
            )
            self.assertIn(
                "global_lab_dual_threshold", manifest["segmentation_algorithm"]
            )
            self.assertEqual(len(manifest["layers"]), 2)
            native_vs_display = []
            for item, layer in zip(objects, manifest["layers"], strict=True):
                with Image.open(layer["path"]).convert("RGBA") as asset:
                    alpha = asset.getchannel("A")
                    self.assertEqual(alpha.getextrema()[0], 0)
                    self.assertGreater(alpha.getextrema()[1], 0)
                    self.assertEqual(list(asset.size), layer["native_pixel_size"])
                    self.assertEqual(list(asset.size), layer["embedded_pixel_size"])
                self.assertTrue(layer["resolution_preserved"])
                native_vs_display.append(
                    layer["native_pixel_size"] != layer["display_pixel_size"]
                )
                target = item["bbox_px"]
                placed = layer["bbox_px"]
                self.assertAlmostEqual(
                    placed[0] + placed[2] / 2, target[0] + target[2] / 2, delta=1
                )
                self.assertAlmostEqual(
                    placed[1] + placed[3] / 2, target[1] + target[3] / 2, delta=1
                )
                expected_placement = (
                    "fit_exact_html_holder_bbox"
                    if item["kind"] == "holder"
                    else "aspect_fit_inside_html_visual_region"
                )
                self.assertEqual(layer["placement"], expected_placement)
                if item["kind"] == "holder":
                    self.assertEqual(placed, target)
                    self.assertIsNotNone(layer["text_safe_rect"])
            self.assertTrue(any(native_vs_display))

    def test_mock_asset_board_run_never_creates_an_image_client(self) -> None:
        config = load_config(ROOT)
        with tempfile.TemporaryDirectory(prefix="rl4ppt-asset-board-") as temporary:
            root = Path(temporary)
            self._write_run_fixture(root)

            def forbidden_client():
                raise AssertionError("mock run must not construct an API client")

            manifest = run_background_set(
                project_root=ROOT,
                html_compile_dir=root / "html_compile",
                step2_dir=root / "step2",
                output_dir=root / "final_background",
                slide_ids=["slide_001", "slide_002"],
                config=config,
                mock_images=True,
                resume_missing=False,
                client_factory=forbidden_client,
            )
            first = manifest["slides"][0]
            second = manifest["slides"][1]
            request = json.loads(
                (
                    root / "final_background/slide_001/prepared/request_preview.json"
                ).read_text()
            )
            first_layers = json.loads(Path(first["layer_manifest"]).read_text())

        self.assertEqual(manifest["image_call_count"], 0)
        self.assertEqual(manifest["accepted_task_count"], 0)
        self.assertEqual(manifest["background_reference_index"], 0)
        self.assertEqual((first["visual_count"], first["holder_count"]), (2, 1))
        self.assertTrue(first["image_requested"])
        self.assertFalse(second["image_requested"])
        self.assertEqual(
            request["reference_image_order"],
            ["canonical_background", "chroma_control_canvas"],
        )
        self.assertEqual(
            first_layers["extraction"], "clean_branch_slot_aware_chroma_removal"
        )

    def test_minecraft_uses_magenta_key_instead_of_green(self) -> None:
        self.assertEqual(select_key_color("Exact Minecraft visual language"), "#ff00ff")

    def test_changed_bundle_submits_new_task_instead_of_reusing_old_task_id(
        self,
    ) -> None:
        class Client:
            def __init__(self) -> None:
                self.submitted = 0
                self.polled = []

            def submit_image_task(self, _payload: dict) -> str:
                self.submitted += 1
                return "task_new"

            def poll_task(self, task_id: str) -> dict:
                self.polled.append(task_id)
                return {
                    "status": "completed",
                    "result": {"images": [{"url": "unused"}]},
                }

            def download(self, _url: str, output: Path) -> None:
                output.write_bytes(b"new-image")

        with tempfile.TemporaryDirectory(prefix="rl4ppt-bundle-resume-") as temporary:
            live_dir = Path(temporary)
            (live_dir / "task.json").write_text(
                json.dumps({"task_id": "task_old", "bundle_sha256": "old"}),
                encoding="utf-8",
            )
            client = Client()
            result = _generate_or_resume(
                client=client,
                payload={"model": "test"},
                bundle_sha256="new",
                live_dir=live_dir,
            )

        self.assertEqual(client.submitted, 1)
        self.assertEqual(client.polled, ["task_new"])
        self.assertEqual(result["task_id"], "task_new")

    def test_failed_image_task_is_resubmitted_once(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.submitted = 0
                self.polled = []

            def submit_image_task(self, _payload: dict) -> str:
                self.submitted += 1
                return f"task_{self.submitted}"

            def poll_task(self, task_id: str) -> dict:
                self.polled.append(task_id)
                if task_id == "task_1":
                    raise ApiMartImageTaskFailure(task_id, {"type": "task_failed"})
                return {
                    "status": "completed",
                    "result": {"images": [{"url": "unused"}]},
                }

            def download(self, _url: str, output: Path) -> None:
                output.write_bytes(b"image")

        with tempfile.TemporaryDirectory(prefix="rl4ppt-task-resubmit-") as temporary:
            live_dir = Path(temporary)
            client = Client()
            result = _generate_or_resume(
                client=client,
                payload={"model": "test"},
                bundle_sha256="bundle",
                live_dir=live_dir,
            )
            failed_record_exists = (live_dir / "task.attempt_01.failed.json").is_file()

        self.assertEqual(client.submitted, 2)
        self.assertEqual(client.polled, ["task_1", "task_2"])
        self.assertEqual(result["image_calls_during_run"], 2)
        self.assertTrue(failed_record_exists)

    def test_image_task_stops_after_one_resubmission(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.submitted = 0

            def submit_image_task(self, _payload: dict) -> str:
                self.submitted += 1
                return f"task_{self.submitted}"

            def poll_task(self, task_id: str) -> dict:
                raise ApiMartImageTaskFailure(task_id, {"type": "task_failed"})

            def download(self, _url: str, output: Path) -> None:
                raise AssertionError("failed tasks must not download")

        with tempfile.TemporaryDirectory(prefix="rl4ppt-task-stop-") as temporary:
            client = Client()
            with self.assertRaises(ApiMartImageTaskFailure) as raised:
                _generate_or_resume(
                    client=client,
                    payload={"model": "test"},
                    bundle_sha256="bundle",
                    live_dir=Path(temporary),
                )

        self.assertEqual(client.submitted, 2)
        self.assertEqual(raised.exception.image_calls_during_run, 2)

    def test_background_stage_continues_with_failed_slide_placeholder(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.submitted = 0

            def submit_image_task(self, _payload: dict) -> str:
                self.submitted += 1
                return f"task_{self.submitted}"

            def poll_task(self, task_id: str) -> dict:
                raise ApiMartImageTaskFailure(task_id, {"type": "task_failed"})

            def download(self, _url: str, output: Path) -> None:
                raise AssertionError("failed tasks must not download")

        config = load_config(ROOT)
        with tempfile.TemporaryDirectory(prefix="rl4ppt-failed-slide-") as temporary:
            root = Path(temporary)
            self._write_run_fixture(root)
            client = Client()
            manifest = run_background_set(
                project_root=ROOT,
                html_compile_dir=root / "html_compile",
                step2_dir=root / "step2",
                output_dir=root / "final_background",
                slide_ids=["slide_001", "slide_002"],
                config=config,
                mock_images=False,
                resume_missing=False,
                client_factory=lambda: client,
            )
            failed_layer_manifest = json.loads(
                (root / "final_background/slide_001/layer_manifest.json").read_text()
            )

        self.assertEqual(client.submitted, 2)
        self.assertEqual(manifest["failed_slides"], [1])
        self.assertEqual(manifest["slides"][0]["status"], "failed_placeholder")
        self.assertEqual(
            failed_layer_manifest["failure_placeholder"]["slide_number"], 1
        )

    def test_failed_missing_object_recovery_marks_only_that_slide(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.submitted = 0
                self.polled = []

            def submit_image_task(self, _payload: dict) -> str:
                self.submitted += 1
                return f"task_{self.submitted}"

            def poll_task(self, task_id: str) -> dict:
                self.polled.append(task_id)
                if task_id != "task_1":
                    raise ApiMartImageTaskFailure(
                        task_id, {"type": "task_failed", "stage": "recovery"}
                    )
                return {
                    "status": "completed",
                    "result": {"images": [{"url": "unused"}]},
                }

            def download(self, _url: str, output: Path) -> None:
                Image.new("RGB", CANVAS, (0, 255, 0)).save(output)

        config = load_config(ROOT)
        with tempfile.TemporaryDirectory(prefix="rl4ppt-failed-recovery-") as temporary:
            root = Path(temporary)
            self._write_run_fixture(root)
            client = Client()
            manifest = run_background_set(
                project_root=ROOT,
                html_compile_dir=root / "html_compile",
                step2_dir=root / "step2",
                output_dir=root / "final_background",
                slide_ids=["slide_001", "slide_002"],
                config=config,
                mock_images=False,
                resume_missing=False,
                client_factory=lambda: client,
            )
            failed_layer_manifest = json.loads(
                (root / "final_background/slide_001/layer_manifest.json").read_text()
            )
            recovery_result = json.loads(
                (
                    root / "final_background/slide_001/recovery/live/result.json"
                ).read_text()
            )

        first = manifest["slides"][0]
        second = manifest["slides"][1]
        self.assertEqual(client.submitted, 3)
        self.assertEqual(client.polled, ["task_1", "task_2", "task_3"])
        self.assertEqual(manifest["image_call_count"], 3)
        self.assertEqual(manifest["failed_slides"], [1])
        self.assertEqual(first["status"], "failed_placeholder")
        self.assertEqual(first["image_calls_during_run"], 3)
        self.assertEqual(first["recovery_task_id"], "task_3")
        self.assertEqual(second["status"], "succeeded")
        self.assertEqual(failed_layer_manifest["layers"], [])
        self.assertEqual(
            failed_layer_manifest["failure_placeholder"]["reason"],
            "missing_object_recovery_failed_after_one_resubmission",
        )
        self.assertEqual(recovery_result["status"], "failed_placeholder")

    @staticmethod
    def _write_run_fixture(root: Path) -> None:
        step2 = root / "step2"
        step2.mkdir(parents=True)
        Image.new("RGB", CANVAS, (240, 240, 240)).save(
            step2 / "canonical_background.png"
        )
        (step2 / "design_system.json").write_text(
            json.dumps({"style_prompt": "clean editorial illustration"}),
            encoding="utf-8",
        )
        fixtures = {
            "slide_001": {
                "layout": {
                    "visual_regions": [
                        {
                            "region_id": "visual_a",
                            "bbox_px": [600, 180, 330, 500],
                            "visual_brief": "Show state.",
                        },
                        {
                            "region_id": "visual_b",
                            "bbox_px": [970, 180, 330, 500],
                            "visual_brief": "Show prediction.",
                        },
                    ],
                    "text_regions": [],
                    "holders": [
                        {
                            "holder_id": "holder_body_001",
                            "bbox_px": [60, 210, 400, 180],
                            "text_region_id": "body_001",
                        }
                    ],
                },
            },
            "slide_002": {
                "layout": {"visual_regions": [], "text_regions": [], "holders": []}
            },
        }
        for slide_id, fixture in fixtures.items():
            compile_dir = root / "html_compile/slides" / slide_id
            compile_dir.mkdir(parents=True)
            (compile_dir / "html_layout.json").write_text(
                json.dumps(fixture["layout"]), encoding="utf-8"
            )


def _separated_by(left: list[int], right: list[int], gutter: int) -> bool:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    return (
        lx + lw + gutter <= rx
        or rx + rw + gutter <= lx
        or ly + lh + gutter <= ry
        or ry + rh + gutter <= ly
    )


if __name__ == "__main__":
    unittest.main()
