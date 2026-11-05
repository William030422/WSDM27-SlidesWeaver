from __future__ import annotations

import sys
import unittest
from pathlib import Path

from integration.final_background import bind_visual_briefs
from integration.region_content import build_generation_units, build_region_inventory


ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY_SOURCE = ROOT / "reused" / "topology"
if str(TOPOLOGY_SOURCE) not in sys.path:
    sys.path.insert(0, str(TOPOLOGY_SOURCE))

from extract_calibrated_topology_with_vlm import CalibratedTopologyResult  # noqa: E402
from postprocess_calibrated_planning_margins import (  # noqa: E402
    compile_group_policies,
    union_bbox,
)


class GenerationUnitTests(unittest.TestCase):
    def test_overlay_group_is_one_generation_unit(self) -> None:
        topology = {
            "entities": [
                {"region_id": "hero", "canonical_role": "visual"},
                {"region_id": "path", "canonical_role": "visual"},
                {"region_id": "stage_1", "canonical_role": "visual"},
                {"region_id": "stage_2", "canonical_role": "visual"},
                {"region_id": "stage_3", "canonical_role": "visual"},
            ],
            "components": [
                {"component_id": "c_hero", "source_region_ids": ["hero"]},
                {"component_id": "c_path", "source_region_ids": ["path"]},
                {"component_id": "c_stage_1", "source_region_ids": ["stage_1"]},
                {"component_id": "c_stage_2", "source_region_ids": ["stage_2"]},
                {"component_id": "c_stage_3", "source_region_ids": ["stage_3"]},
            ],
            "groups": [
                {
                    "group_id": "g_stages",
                    "arrangement": "row",
                    "children": ["c_stage_1", "c_stage_2", "c_stage_3"],
                },
                {
                    "group_id": "g_diagram",
                    "arrangement": "overlay",
                    "children": ["c_path", "g_stages"],
                },
                {
                    "group_id": "g_root",
                    "arrangement": "row",
                    "children": ["c_hero", "g_diagram"],
                },
            ],
        }
        layout = {
            "entities": [
                {"region_id": "hero", "postprocessed_bbox_px": [40, 300, 300, 450]},
                {"region_id": "path", "postprocessed_bbox_px": [400, 420, 1000, 300]},
                {"region_id": "stage_1", "postprocessed_bbox_px": [460, 360, 160, 160]},
                {"region_id": "stage_2", "postprocessed_bbox_px": [820, 360, 160, 160]},
                {"region_id": "stage_3", "postprocessed_bbox_px": [1180, 360, 160, 160]},
            ]
        }

        units = build_generation_units(topology, layout)

        self.assertEqual({item["unit_id"] for item in units}, {"c_hero", "g_diagram"})
        diagram = next(item for item in units if item["unit_id"] == "g_diagram")
        self.assertEqual(
            set(diagram["source_region_ids"]),
            {"path", "stage_1", "stage_2", "stage_3"},
        )
        inventory = build_region_inventory(topology, layout)
        self.assertEqual(
            {item["region_id"] for item in inventory if item["role"] == "visual"},
            {"c_hero", "g_diagram"},
        )
        bindings = bind_visual_briefs(
            {
                "visual_regions": [
                    {"region_id": "g_diagram", "semantic_brief": "one shared path scene"}
                ]
            },
            layout,
            topology,
        )
        self.assertEqual(bindings[0]["component_id"], "g_diagram")

    def test_components_are_generation_units_without_overlay(self) -> None:
        topology = {
            "entities": [
                {"region_id": "visual_1a", "canonical_role": "visual"},
                {"region_id": "visual_1b", "canonical_role": "visual"},
                {"region_id": "visual_2", "canonical_role": "visual"},
            ],
            "components": [
                {
                    "component_id": "c_stage_1",
                    "source_region_ids": ["visual_1a", "visual_1b"],
                },
                {"component_id": "c_stage_2", "source_region_ids": ["visual_2"]},
            ],
            "groups": [
                {
                    "group_id": "g_sequence",
                    "arrangement": "row",
                    "children": ["c_stage_1", "c_stage_2"],
                }
            ],
        }
        layout = {
            "entities": [
                {"region_id": "visual_1a", "postprocessed_bbox_px": [100, 200, 100, 80]},
                {"region_id": "visual_1b", "postprocessed_bbox_px": [100, 280, 160, 120]},
                {"region_id": "visual_2", "postprocessed_bbox_px": [400, 200, 160, 200]},
            ]
        }

        units = build_generation_units(topology, layout)

        self.assertEqual([item["unit_id"] for item in units], ["c_stage_1", "c_stage_2"])
        self.assertEqual(units[0]["bbox_px"], [100, 200, 160, 200])


class GroupPolicyTests(unittest.TestCase):
    def test_equal_size_even_sequential_row_is_compiled(self) -> None:
        result = CalibratedTopologyResult.model_validate(
            {
                "slide_id": "slide_001",
                "groups": [
                    {
                        "group_id": "g_sequence",
                        "role": "sequence",
                        "scope": "top_level",
                        "arrangement": "row",
                        "children": ["c1", "c2", "c3", "c4"],
                        "peer_relation": "sequential",
                        "calibration_policy": {
                            "size_policy": "equal_size",
                            "alignment": "baseline",
                            "spacing": "even",
                        },
                    }
                ],
                "components": [
                    {
                        "component_id": f"c{index}",
                        "component_role": "mixed",
                        "source_region_ids": [f"visual_{index}", f"caption_{index}"],
                    }
                    for index in range(1, 5)
                ],
                "entities": [
                    *[
                        {
                            "region_id": f"visual_{index}",
                            "canonical_role": "visual",
                            "parent_component_id": f"c{index}",
                            "calibrated_bbox": [index * 2, 1, 1, 1],
                        }
                        for index in range(1, 5)
                    ],
                    *[
                        {
                            "region_id": f"caption_{index}",
                            "canonical_role": "caption",
                            "parent_component_id": f"c{index}",
                            "calibrated_bbox": [index * 2, 2, 1, 1],
                        }
                        for index in range(1, 5)
                    ],
                ],
                "reading_order": [
                    *[f"visual_{index}" for index in range(1, 5)],
                    *[f"caption_{index}" for index in range(1, 5)],
                ],
            }
        )
        boxes = {
            "visual_1": (100, 200, 100, 120),
            "caption_1": (90, 320, 120, 80),
            "visual_2": (350, 180, 180, 100),
            "caption_2": (340, 300, 200, 100),
            "visual_3": (520, 160, 300, 140),
            "caption_3": (560, 310, 220, 90),
            "visual_4": (900, 120, 160, 180),
            "caption_4": (880, 300, 220, 100),
        }

        compiled = compile_group_policies(result, boxes)
        component_boxes = [
            union_bbox([compiled[f"visual_{index}"], compiled[f"caption_{index}"]])
            for index in range(1, 5)
        ]

        self.assertEqual(len({box[2] for box in component_boxes}), 1)
        self.assertEqual(len({box[3] for box in component_boxes}), 1)
        self.assertEqual(len({box[1] + box[3] for box in component_boxes}), 1)
        gaps = [
            component_boxes[index + 1][0]
            - (component_boxes[index][0] + component_boxes[index][2])
            for index in range(3)
        ]
        self.assertEqual(len(set(gaps)), 1)
        self.assertGreaterEqual(gaps[0], 0)
        captions = [compiled[f"caption_{index}"] for index in range(1, 5)]
        self.assertEqual(len({box[1] for box in captions}), 1)
        self.assertEqual(len({box[2] for box in captions}), 1)
        self.assertEqual(len({box[3] for box in captions}), 1)


if __name__ == "__main__":
    unittest.main()
