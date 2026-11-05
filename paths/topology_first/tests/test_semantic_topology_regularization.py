from __future__ import annotations

import unittest

from integration.region_content import attach_semantic_context, build_region_inventory
from integration.semantic_topology import (
    normalize_semantic_topology,
    regularize_semantic_topology,
)


class SemanticTopologyRegularizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = {
            "slide_id": "slide_001",
            "elements": [
                {"element_id": "title_1", "role": "title", "bbox": [4, 2, 56, 4], "confidence": 0.9},
                {"element_id": "intro_1", "role": "standalone_text", "bbox": [4, 7, 56, 3], "confidence": 0.8},
                {"element_id": "label_1", "role": "label", "bbox": [5, 12, 24, 2], "confidence": 0.7},
                {"element_id": "visual_1", "role": "visual", "bbox": [5, 14, 24, 12], "confidence": 0.9},
                {"element_id": "caption_1", "role": "caption", "bbox": [5, 27, 24, 2], "confidence": 0.7},
                {"element_id": "visual_2", "role": "visual", "bbox": [35, 14, 24, 14], "confidence": 0.9},
            ],
        }
        self.payload = {
            "slide_id": "slide_001",
            "blocks": [
                {
                    "block_id": "b_title",
                    "role": "title",
                    "semantic_role": "title",
                    "source_region_ids": ["title_1"],
                    "bbox": [4, 2, 56, 4],
                    "confidence": 0.9,
                },
                {
                    "block_id": "b_intro",
                    "role": "body_text",
                    "semantic_role": "intro",
                    "source_region_ids": ["intro_1"],
                    "bbox": [4, 7, 56, 3],
                    "confidence": 0.8,
                },
                {
                    "block_id": "b_first",
                    "role": "semantic_stage",
                    "semantic_role": "first stage",
                    "source_region_ids": ["label_1", "visual_1", "caption_1"],
                    "bbox": [5, 12, 24, 17],
                    "confidence": 0.9,
                },
                {
                    "block_id": "b_second",
                    "role": "semantic_stage",
                    "semantic_role": "second stage",
                    "source_region_ids": ["visual_2"],
                    "bbox": [35, 14, 24, 14],
                    "confidence": 0.9,
                },
            ],
            "connections": [
                {
                    "source_block_id": "b_first",
                    "target_block_id": "b_second",
                    "relation": "sequence_next",
                }
            ],
            "ignored_region_ids": [],
            "reading_order": ["b_title", "b_intro", "b_first", "b_second"],
        }
        self.source_roles = {
            item["element_id"]: item["role"] for item in self.identity["elements"]
        }

    def test_regularizes_peers_and_preserves_top_and_bottom_text_slots(self) -> None:
        semantic = normalize_semantic_topology(
            self.payload,
            slide_id="slide_001",
            source_ids={item["element_id"] for item in self.identity["elements"]},
            source_roles=self.source_roles,
        )

        compiled, report = regularize_semantic_topology(
            semantic=semantic, identity=self.identity
        )

        content = [
            item for item in report["blocks"] if item["block_id"] in {"b_first", "b_second"}
        ]
        self.assertEqual(
            [item["container_bbox"] for item in content],
            [[5, 12, 24, 17], [35, 14, 24, 14]],
        )
        self.assertEqual(report["overlap_count"], 0)
        self.assertEqual(report["layout_family"], "row")
        self.assertEqual(
            {item["position"] for item in content[0]["text_slots"]},
            {"top", "bottom"},
        )
        self.assertTrue(
            all(
                slot["bbox"][2] == content[0]["container_bbox"][2]
                for slot in content[0]["text_slots"]
            )
        )
        self.assertEqual(report["dual_text_slot_block_count"], 1)
        content_group = next(
            item for item in compiled["groups"] if item["group_id"] == "g_content"
        )
        self.assertEqual(content_group["peer_relation"], "sequential")
        self.assertEqual(content_group["arrangement"], "row")
        self.assertEqual(
            content_group["calibration_policy"],
            {"size_policy": "equal_size", "alignment": "top", "spacing": "even"},
        )

    def test_semantic_context_reaches_generation_unit_inventory(self) -> None:
        semantic = normalize_semantic_topology(
            self.payload,
            slide_id="slide_001",
            source_ids={item["element_id"] for item in self.identity["elements"]},
            source_roles=self.source_roles,
        )
        compiled, _ = regularize_semantic_topology(
            semantic=semantic, identity=self.identity
        )
        layout = {
            "canvas": {"width": 1536, "height": 864},
            "entities": [
                {
                    "region_id": item["region_id"],
                    "postprocessed_bbox_px": [value * 24 for value in item["calibrated_bbox"]],
                }
                for item in compiled["entities"]
            ],
        }

        inventory = attach_semantic_context(
            build_region_inventory(compiled, layout), semantic
        )
        first = next(item for item in inventory if item["region_id"] == "c_first")

        self.assertEqual(first["semantic_role"], "first stage")
        self.assertEqual(first["semantic_modality"], "mixed")
        self.assertEqual(first["semantic_connections"][0]["relation"], "sequence_next")
        self.assertEqual(first["semantic_connections"][0]["other_semantic_role"], "second stage")

    def test_text_only_stages_do_not_create_visuals_and_preserve_macro_layout(self) -> None:
        identity = {
            "slide_id": "slide_004",
            "elements": [
                {"element_id": "title_001", "role": "title", "bbox": [3, 3, 40, 7]},
                {"element_id": "text_001", "role": "standalone_text", "bbox": [4, 14, 18, 3]},
                {"element_id": "text_002", "role": "standalone_text", "bbox": [4, 19, 14, 3]},
                {"element_id": "text_003", "role": "standalone_text", "bbox": [4, 23, 28, 9]},
                {"element_id": "visual_001", "role": "visual", "bbox": [33, 13, 28, 20]},
                {"element_id": "label_001", "role": "label", "bbox": [46, 10, 12, 5]},
                {"element_id": "caption_001", "role": "caption", "bbox": [44, 27, 16, 5]},
            ],
        }
        payload = {
            "slide_id": "slide_004",
            "blocks": [
                {"block_id": "b_title", "role": "title", "semantic_role": "title", "source_region_ids": ["title_001"], "bbox": [3, 3, 40, 7]},
                {"block_id": "b_plan", "role": "semantic_stage", "semantic_role": "planning", "source_region_ids": ["text_001"], "bbox": [4, 14, 18, 3]},
                {"block_id": "b_act", "role": "semantic_stage", "semantic_role": "action", "source_region_ids": ["text_002"], "bbox": [4, 19, 14, 3]},
                {"block_id": "b_check", "role": "semantic_stage", "semantic_role": "replan", "source_region_ids": ["text_003"], "bbox": [4, 23, 28, 9]},
                {"block_id": "b_scene", "role": "semantic_visual", "semantic_role": "prediction scene", "source_region_ids": ["visual_001", "label_001", "caption_001"], "bbox": [33, 10, 28, 23]},
            ],
            "connections": [
                {"source_block_id": "b_plan", "target_block_id": "b_act", "relation": "sequence_next"},
                {"source_block_id": "b_act", "target_block_id": "b_check", "relation": "sequence_next"},
            ],
            "ignored_region_ids": [],
            "reading_order": ["b_title", "b_plan", "b_act", "b_check", "b_scene"],
        }
        source_roles = {
            item["element_id"]: item["role"] for item in identity["elements"]
        }

        semantic = normalize_semantic_topology(
            payload,
            slide_id="slide_004",
            source_ids=set(source_roles),
            source_roles=source_roles,
        )
        compiled, report = regularize_semantic_topology(
            semantic=semantic, identity=identity
        )

        modalities = {
            item["block_id"]: item["modality"] for item in semantic["blocks"]
        }
        self.assertEqual(modalities["b_plan"], "text_only")
        self.assertEqual(modalities["b_act"], "text_only")
        self.assertEqual(modalities["b_check"], "text_only")
        self.assertEqual(modalities["b_scene"], "mixed")
        self.assertEqual(
            [
                item["region_id"]
                for item in compiled["entities"]
                if item["canonical_role"] == "visual"
            ],
            ["b_scene_visual"],
        )
        boxes = {
            item["region_id"]: item["calibrated_bbox"] for item in compiled["entities"]
        }
        self.assertEqual(boxes["b_plan"], [4, 14, 18, 3])
        self.assertEqual(boxes["b_act"], [4, 19, 14, 3])
        self.assertEqual(boxes["b_check"], [4, 23, 28, 9])
        scene = next(
            item for item in report["blocks"] if item["block_id"] == "b_scene"
        )
        self.assertEqual(scene["container_bbox"], [33, 10, 28, 23])
        self.assertEqual(boxes["b_scene_visual"][0], 33)
        self.assertEqual(boxes["b_scene_visual"][2], 28)
        self.assertEqual(report["layout_family"], "freeform")

    def test_multiple_body_blocks_keep_their_role(self) -> None:
        identity = {
            "slide_id": "slide_text",
            "elements": [
                {"element_id": "title", "role": "title", "bbox": [2, 2, 40, 4]},
                {"element_id": "text_1", "role": "standalone_text", "bbox": [3, 8, 20, 4]},
                {"element_id": "text_2", "role": "standalone_text", "bbox": [3, 14, 20, 4]},
                {"element_id": "visual", "role": "visual", "bbox": [30, 8, 28, 20]},
            ],
        }
        payload = {
            "blocks": [
                {"block_id": "b_title", "role": "title", "source_region_ids": ["title"], "bbox": [2, 2, 40, 4]},
                {"block_id": "b_text_1", "role": "body_text", "source_region_ids": ["text_1"], "bbox": [3, 8, 20, 4]},
                {"block_id": "b_text_2", "role": "body_text", "source_region_ids": ["text_2"], "bbox": [3, 14, 20, 4]},
                {"block_id": "b_visual", "role": "semantic_visual", "source_region_ids": ["visual"], "bbox": [30, 8, 28, 20]},
            ],
            "connections": [],
            "ignored_region_ids": [],
            "reading_order": ["b_title", "b_text_1", "b_text_2", "b_visual"],
        }
        source_roles = {
            item["element_id"]: item["role"] for item in identity["elements"]
        }

        semantic = normalize_semantic_topology(
            payload,
            slide_id="slide_text",
            source_ids=set(source_roles),
            source_roles=source_roles,
        )
        compiled, _ = regularize_semantic_topology(
            semantic=semantic, identity=identity
        )

        body_blocks = [item for item in semantic["blocks"] if item["role"] == "body_text"]
        self.assertEqual(len(body_blocks), 2)
        self.assertTrue(all(item["modality"] == "text_only" for item in body_blocks))
        self.assertIn("b_text_2", {item["region_id"] for item in compiled["entities"]})


if __name__ == "__main__":
    unittest.main()
