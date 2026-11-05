from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PIL import Image

from integration.semantic_topology import fit_slot_heights, regularize_semantic_topology
from integration.text_assembly import (
    _bbox_intersection,
    _fit_element_deterministically,
    _geometry_quality_findings,
    _hard_rejection_reasons,
    _load_text_placement,
    _render_and_validate,
    _run_content_unit_fit,
)
from v9_deck_planning.green_background_removal import _despill_key_color, _key_spill_strength
from v9_deck_planning.style_assets import detect_text_safe_rect
from integration.background_decomposition import decompose_background


class QualityTopologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.placement = _load_text_placement()

    def plan(self, *, title: bool = False) -> dict:
        def element(region_id, bbox):
            return {
                "region_id": region_id, "role": "title" if title else "caption",
                "bbox_px": bbox, "topology_bbox_px": list(bbox), "text": "Required copy",
                "font": {"face": "Noto Sans", "size_pt": 18, "minimum_pt": 18},
                "presentation": {"mode": "direct_text", "text_color": "111111"},
            }
        return {
            "canvas_px": {"width": 1536, "height": 864}, "groups": [],
            "elements": [element("subject", [100, 380, 300, 48]), element("neighbor", [100, 450, 300, 48])],
        }

    @staticmethod
    def measurements(plan, states, placement):
        return [
            {"fits": state["element"]["bbox_px"][3] >= 200, "line_count": 4,
             "required_height_in": 2.0, "available_height_in": state["element"]["bbox_px"][3] / 100}
            for state in states
        ]

    def test_fit_retains_usable_text_without_covering_neighbor(self) -> None:
        plan = self.plan()
        neighbor = copy.deepcopy(plan["elements"][1])
        with patch("integration.text_assembly._measure_registration_states", side_effect=self.measurements):
            record = _fit_element_deterministically(plan, 0, self.placement)
        self.assertEqual(record["status"], "unresolved")
        self.assertEqual(_bbox_intersection(plan["elements"][0]["bbox_px"], neighbor["bbox_px"]), 0)
        self.assertEqual(plan["elements"][1], neighbor)
        self.assertEqual(plan["elements"][0]["text"], "Required copy")
        self.assertLess(plan["elements"][0]["bbox_px"][3], 200)

    def test_free_space_still_allows_metric_expansion(self) -> None:
        plan = self.plan()
        plan["elements"] = plan["elements"][:1]
        with patch("integration.text_assembly._measure_registration_states", side_effect=self.measurements):
            record = _fit_element_deterministically(plan, 0, self.placement)
        self.assertEqual(record["status"], "fit")
        self.assertEqual(plan["elements"][0]["bbox_px"][3], 200)

    def test_protected_visual_cannot_be_consumed_by_text_fit(self) -> None:
        plan = self.plan()
        visual = plan["elements"].pop()
        plan["protected_visual_regions"] = [visual]
        with patch("integration.text_assembly._measure_registration_states", side_effect=self.measurements):
            _fit_element_deterministically(plan, 0, self.placement)
        self.assertEqual(_bbox_intersection(plan["elements"][0]["bbox_px"], visual["bbox_px"]), 0)

    def test_title_one_line_but_too_tall_is_unresolved(self) -> None:
        plan = self.plan(title=True)
        measurement = {"fits": False, "line_count": 1, "tight_width_px": 180,
                       "required_height_in": 1.0, "available_height_in": 0.1}
        with patch("integration.text_assembly._measure_registration_states", side_effect=lambda p, s, t: [measurement for _ in s]):
            record = _fit_element_deterministically(plan, 0, self.placement)
        self.assertEqual(record["status"], "unresolved")

    def test_unsafe_title_recentering_keeps_original_safe_geometry(self) -> None:
        plan = self.plan(title=True)
        original = copy.deepcopy(plan["elements"][0])
        plan["elements"][0]["topology_bbox_px"] = [400, 380, 300, 48]
        plan["elements"][1]["bbox_px"] = [450, 380, 200, 48]
        measurement = {"fits": True, "line_count": 1, "tight_width_px": 180,
                       "required_height_in": 0.2, "available_height_in": 0.4}
        with patch("integration.text_assembly._measure_registration_states", side_effect=lambda p, s, t: [measurement for _ in s]):
            _fit_element_deterministically(plan, 0, self.placement)
        self.assertEqual(plan["elements"][0]["bbox_px"], original["bbox_px"])
        self.assertEqual(plan["elements"][0]["registration"]["title_text_box_mode"], "original_geometry_fallback")
        self.assertEqual(_bbox_intersection(plan["elements"][0]["bbox_px"], plan["elements"][1]["bbox_px"]), 0)

    def test_all_unsafe_body_candidates_retain_original_geometry(self) -> None:
        plan = self.plan()
        original = list(plan["elements"][0]["bbox_px"])
        plan["elements"][0]["topology_bbox_px"] = [400, 380, 300, 48]
        plan["elements"][1]["bbox_px"] = [450, 380, 200, 48]
        with patch("integration.text_assembly._measure_registration_states", side_effect=self.measurements):
            record = _fit_element_deterministically(plan, 0, self.placement)
        self.assertEqual(record["status"], "unresolved")
        self.assertEqual(plan["elements"][0]["bbox_px"], original)

    def test_geometry_report_detects_baseline_visual_collision(self) -> None:
        plan = self.plan()
        plan["protected_visual_regions"] = [{"region_id": "visual", "bbox_px": [110, 390, 100, 20]}]
        self.assertTrue(any(item["kind"] == "text_visual_collision" for item in _geometry_quality_findings(plan)))
        plan["intentional_visual_overlaps"] = ["subject::visual"]
        self.assertFalse(any(item["kind"] == "text_visual_collision" for item in _geometry_quality_findings(plan)))

    def test_quality_findings_and_freezing_do_not_abort_for_overlap(self) -> None:
        plan = self.plan()
        plan["elements"][1]["bbox_px"] = [100, 390, 300, 48]
        self.assertTrue(_geometry_quality_findings(plan))
        for element in plan["elements"]:
            self.placement._refresh_content_unit(element)
        frozen = self.placement.freeze_content_units(plan)
        self.assertEqual(len(frozen["elements"]), 2)

    def test_refinement_cannot_worsen_an_already_overlapping_pair(self) -> None:
        plan = self.plan()
        plan["elements"][1]["bbox_px"] = [100, 420, 300, 48]
        candidate = copy.deepcopy(plan)
        candidate["elements"][0]["bbox_px"][3] += 20
        for current in (plan, candidate):
            for element in current["elements"]:
                self.placement._refresh_content_unit(element)
        reasons = _hard_rejection_reasons(plan, candidate, self.placement)
        self.assertTrue(any(reason.startswith("increased_collision:") for reason in reasons))

    def test_invalid_holder_uses_native_fallback_without_losing_copy(self) -> None:
        element = self.plan()["elements"][0]
        original_bbox = list(element["bbox_px"])
        element["presentation"] = {"mode": "raster_holder_text", "asset_id": "broken", "text_safe_rect": None}
        self.placement._refresh_holder_binding(element)
        self.assertEqual(element["presentation"]["mode"], "native_shape_text")
        self.assertEqual(element["text"], "Required copy")
        self.assertEqual(element["bbox_px"], original_bbox)

    def test_blank_render_is_reported_without_interrupting_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rendered = root / "rendered"
            def fake_render(command, **kwargs):
                if command[0] == "libreoffice":
                    (root / "deck.pdf").write_bytes(b"PDF test stub")
                else:
                    Image.new("RGB", (160, 90), "white").save(rendered / "slide-1.png")
                return SimpleNamespace(returncode=0)
            with patch("integration.text_assembly.subprocess.run", side_effect=fake_render):
                _render_and_validate(root / "deck.pptx", root / "deck.pdf", rendered, 1)
            report = json.loads((rendered / "render_quality.json").read_text())
            self.assertEqual(report["status"], "degraded")
            self.assertEqual(report["warnings"][0]["kind"], "blank_rendered_slide")

    def test_invalid_bounds_repaired_locally_and_required_copy_kept(self) -> None:
        plan = self.plan()
        plan["elements"][0]["bbox_px"] = [1500, 380, 300, 48]
        plan["elements"][0]["topology_bbox_px"] = [1500, 380, 300, 48]
        neighbor = copy.deepcopy(plan["elements"][1])
        def measurements_with_valid_neighbor(current_plan, states, placement):
            results = self.measurements(current_plan, states, placement)
            for state, result in zip(states, results):
                if state["element"]["region_id"] == "neighbor":
                    result["fits"] = True
            return results
        with tempfile.TemporaryDirectory() as temporary:
            with patch("integration.text_assembly._measure_registration_states", side_effect=measurements_with_valid_neighbor):
                _run_content_unit_fit(output_dir=Path(temporary), slide_ids=["slide_001"],
                                      plans_by_slide={"slide_001": plan}, text_placement=self.placement)
        self.assertEqual(plan["elements"][0]["text"], "Required copy")
        x, y, width, height = plan["elements"][0]["bbox_px"]
        self.assertLessEqual(x + width, 1536)
        self.assertEqual(plan["elements"][1]["bbox_px"], neighbor["bbox_px"])
        self.assertTrue(plan["quality_warnings"])

    def test_small_mixed_block_is_positive_in_bounds_without_overlap(self) -> None:
        identity = {"elements": [
            {"element_id": "title", "role": "title", "bbox": [3, 2, 58, 4]},
            {"element_id": "label", "role": "label", "bbox": [5, 32, 24, 1]},
            {"element_id": "visual", "role": "visual", "bbox": [5, 33, 24, 1]},
            {"element_id": "caption", "role": "caption", "bbox": [5, 35, 24, 1]},
        ]}
        semantic = {"slide_id": "slide_001", "blocks": [
            {"block_id": "b_title", "role": "title", "bbox": [3, 2, 58, 4], "source_region_ids": ["title"]},
            {"block_id": "b_content", "role": "semantic_stage", "bbox": [5, 32, 24, 4],
             "source_region_ids": ["label", "visual", "caption"]},
        ], "connections": [], "reading_order": ["b_title", "b_content"]}
        compiled, report = regularize_semantic_topology(semantic=semantic, identity=identity)
        for entity in compiled["entities"]:
            x, y, width, height = entity["calibrated_bbox"]
            self.assertGreater(width, 0)
            self.assertGreater(height, 0)
            self.assertLessEqual(x + width, 64)
            self.assertLessEqual(y + height, 36)
        self.assertEqual(report["overlap_count"], 0)

    def test_feasible_existing_slot_sizes_remain_unchanged(self) -> None:
        self.assertEqual(fit_slot_heights(target_height=12, top_height=6, bottom_height=6), (3, 3))
        self.assertEqual(fit_slot_heights(target_height=17, top_height=6, bottom_height=6), (4, 4))

    def test_holder_safe_rectangle_excludes_raw_transparent_hole(self) -> None:
        rgba = np.zeros((100, 200, 4), dtype=np.uint8)
        rgba[10:90, 10:190] = [235, 235, 235, 255]
        rgba[49:52, 99:102] = 0
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "holder.png"
            Image.fromarray(rgba).save(path)
            safe = detect_text_safe_rect(path)
        self.assertIsNotNone(safe)
        x, y, width, height = [round(safe[key] * extent) for key, extent in
                               (("x", 200), ("y", 100), ("width", 200), ("height", 100))]
        self.assertTrue((rgba[y:y + height, x:x + width, 3] > 0).all())

    def test_magenta_key_does_not_treat_green_subject_as_spill(self) -> None:
        key = np.array([255, 0, 255], dtype=np.float32)
        pixels = np.array([[[10, 220, 10], [245, 15, 245]]], dtype=np.float32)
        strength = _key_spill_strength(pixels, key)
        self.assertEqual(strength[0, 0], 0)
        self.assertGreater(strength[0, 1], 0.8)
        corrected = _despill_key_color(pixels, key, np.array([[True, True]]))
        np.testing.assert_array_equal(corrected[0, 0], pixels[0, 0])
        self.assertLess(corrected[0, 1, 0], pixels[0, 1, 0])

    def test_companion_preserves_thin_detail_in_residual_without_changing_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = np.full((100, 200, 3), 240, dtype=np.uint8)
            final = canonical.copy()
            final[49:51, 30:170] = [20, 30, 40]
            Image.fromarray(canonical).save(root / "canonical.png")
            Image.fromarray(final).save(root / "final.png")
            result = decompose_background(canonical_path=root / "canonical.png", final_path=root / "final.png",
                                          region_coordinates_path=root / "missing.json", output_dir=root / "layers")
            self.assertGreater(result["residual_detail_pixel_count"], 0)
            self.assertEqual(result["reconstruction_max_channel_error"], 0)
            np.testing.assert_array_equal(np.asarray(Image.open(root / "canonical.png")), canonical)


if __name__ == "__main__":
    unittest.main()
