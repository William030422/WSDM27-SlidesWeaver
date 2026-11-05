from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

from integration.config import PipelineConfig, Size
from integration.page_pipeline import _page_outputs, _resume_stage_summary, prepare_slide, run_parallel_page_pipeline
from integration.page_runtime import PagePhaseError, atomic_write_json, run_checkpoint
from integration.api_journal import journal_response
from integration.shared_checkpoint import checkpoint_shared_stage
from integration.slide_spec import run_slide_spec_bridge
from integration.text_assembly import _load_text_placement, _run_critic_rounds

ROOT = Path(__file__).resolve().parents[1]


def offline_config() -> PipelineConfig:
    """An explicit public fixture: tests never read the project's .env."""
    return PipelineConfig(project_root=ROOT, env_path=Path("/nonexistent/offline.env"),
        api_root="https://api.invalid/v1", api_key_env="OFFLINE_UNUSED_API_KEY", chat_model="offline-chat",
        image_model="gpt-image-2-official", image_size=Size(1536, 864), image_resolution="1k",
        image_quality="high", image_max_attempts=2, chat_max_tokens=4096, vlm_max_tokens=4096,
        timeout_seconds=10, poll_initial_delay_seconds=0, poll_interval_seconds=0,
        poll_timeout_seconds=10, diffusion_model_path=Path("/nonexistent/offline-model"),
        diffusion_size=Size(1024, 576), diffusion_steps=5, diffusion_guidance_scale=0,
        diffusion_seed=19, title_font="Noto Sans", body_font="Noto Sans", cjk_font="Noto Sans")


def write_deck(root: Path, count: int = 2) -> None:
    step2 = root / "step2"
    step2.mkdir(parents=True)
    style = "Clean technical diagrams"
    atomic_write_json(step2 / "input.json", {"content": "Teach feedback control", "style": style})
    atomic_write_json(step2 / "deck_plan.json", {"topic": "Feedback", "audience": "Students",
        "deck_goal": "Explain the control error", "style": style, "text_presentation_policy": "light",
        "slides": [{"slide_id": f"slide_{index:03d}", "working_title": f"Control {index}",
                    "rough_message": "Compare target and measurement."} for index in range(1, count + 1)]})
    atomic_write_json(step2 / "run_manifest.json", {"stage": "style_asset_generation", "status": "succeeded"})
    atomic_write_json(step2 / "design_system.json", {"style_prompt": style, "canvas": {"width": 1536, "height": 864}})
    atomic_write_json(step2 / "asset_manifest.json", {"assets": []})
    Image.new("RGB", (1536, 864), "white").save(step2 / "canonical_background.png")


class PagePipelineTests(unittest.TestCase):
    def test_reconstructs_completed_stage_summary_after_later_page_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = [{'slide_id': 'slide_001'}, {'slide_id': 'slide_002'}]
            for job in jobs:
                atomic_write_json(root / 'pages' / job['slide_id'] / 'checkpoints/region_content.json',
                    {'status': 'completed', 'result': {'status': 'succeeded', 'stage': 'region_content',
                      'slides': [{'slide_id': job['slide_id']}], 'model_call_count': 1}})
            summary = _resume_stage_summary('region_content', root, jobs)
            self.assertEqual(summary['slide_count'], 2)
            self.assertEqual([s['slide_id'] for s in summary['slides']], ['slide_001', 'slide_002'])
            self.assertTrue((root / 'region_content/run_manifest.json').is_file())

    def test_late_resume_does_not_rewrite_title_only_topology(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_deck(root, 1)
            top = root / "topology/calibrated_topology/slide_001/calibrated_topology.json"
            layout = root / "topology/postprocessed/layouts/slide_001.json"
            atomic_write_json(top, {"entities": [{"region_id": "title", "canonical_role": "title"}]})
            atomic_write_json(layout, {"entities": [{"region_id": "title", "postprocessed_bbox_px": [50, 50, 1200, 80]}]})
            before = (top.read_bytes(), layout.read_bytes())
            for stage in ("background", "assembly"):
                with patch("integration.page_pipeline._inventory", side_effect=AssertionError("frozen topology was reopened")):
                    result = prepare_slide({"output_dir": str(root), "slide_id": "slide_001", "original_index": 0,
                        "config": offline_config(), "start_at": stage, "stop_after": "assembly"})
                self.assertIsNone(result["inventory"])
                self.assertEqual((top.read_bytes(), layout.read_bytes()), before)

    def test_completed_slide_response_survives_local_stage_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_deck(root, 1)
            client = SimpleNamespace(calls=0)
            def chat(**kwargs):
                client.calls += 1
                return json.dumps({"title": "Control", "message": "Compare signals", "visual_intent": "Two signals"})
            client.chat_json = chat
            args = dict(project_root=ROOT, step2_dir=root / "step2", output_dir=root / "slide_specs",
                        config=offline_config(), mock_llm=False, client_factory=lambda: client)
            with patch("integration.slide_spec.finalize_slide_spec", side_effect=OSError("local disk interruption")):
                with self.assertRaises(RuntimeError):
                    run_slide_spec_bridge(**args)
            run_slide_spec_bridge(**args)
            self.assertEqual(client.calls, 1)

    def test_unknown_model_submission_is_not_repeated_and_changed_input_invalidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "response.json"
            calls = []
            def interrupted():
                calls.append(True)
                raise TimeoutError("response outcome unknown")
            with self.assertRaises(TimeoutError):
                journal_response(path=path, request={"prompt": "one"}, invoke=interrupted)
            with self.assertRaisesRegex(RuntimeError, "outcome is unknown"):
                journal_response(path=path, request={"prompt": "one"}, invoke=interrupted)
            self.assertEqual(len(calls), 1)
            first = journal_response(path=path, request={"prompt": "changed"}, invoke=lambda: "completed")
            replay = journal_response(path=path, request={"prompt": "changed"}, invoke=interrupted)
            self.assertFalse(first["reused"])
            self.assertTrue(replay["reused"])
            self.assertEqual(replay["raw"], "completed")

    def test_shared_planning_checkpoint_reuses_complete_matching_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = []
            def plan():
                calls.append(True)
                for name in ("input.json", "deck_plan.json", "run_manifest.json"):
                    atomic_write_json(root / "step1" / name, {"status": "succeeded"})
                return {"status": "succeeded"}
            args = dict(run_dir=root, stage="step1", source_inputs=[], settings={"input": "frozen"}, fn=plan)
            checkpoint_shared_stage(**args)
            checkpoint_shared_stage(**args)
            self.assertEqual(len(calls), 1)
            checkpoint_shared_stage(**{**args, "settings": {"input": "changed"}})
            self.assertEqual(len(calls), 2)

    def test_real_page_chains_keep_owners_and_resume_across_worker_counts(self):
        with tempfile.TemporaryDirectory(prefix="topology-page-chain-") as temporary:
            root = Path(temporary)
            write_deck(root)
            manifests = run_parallel_page_pipeline(output_dir=root, config=offline_config(), workers=2,
                                                    mock_api=True, mock_diffusion=True)
            self.assertEqual(manifests[-1]["slide_count"], 2)
            with zipfile.ZipFile(root / "assembly/deck.pptx") as archive:
                slides = [name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")]
                self.assertEqual(len(slides), 2)
                self.assertIn("Control 1", archive.read("ppt/slides/slide1.xml").decode())
                self.assertIn("Control 2", archive.read("ppt/slides/slide2.xml").decode())
            self.assertTrue((root / "assembly/deck.pdf").is_file())
            ownership = json.loads((root / "page_results.json").read_text())["phases"]
            before = {item["slide_id"]: item["owner_pid"] for item in ownership["inventory"]}
            after = {item["slide_id"]: item["owner_pid"] for item in ownership["page_complete"]}
            self.assertEqual(before, after)
            self.assertEqual(len(set(before.values())), 2)
            self.assertNotIn(os.getpid(), before.values())
            accepted = root / "assembly/pages/slide_001/composition_plan.json"
            baseline = root / "assembly/pages/slide_001/refinement/rounds/base/deck.pdf"
            self.assertTrue(baseline.is_file())
            stamp = accepted.stat().st_mtime_ns
            with patch("integration.text_assembly._prepare_assembly_pages", side_effect=AssertionError("final assembly rebuilt pages")):
                run_parallel_page_pipeline(output_dir=root, config=offline_config(), workers=1,
                                           mock_api=True, mock_diffusion=True)
            self.assertEqual(accepted.stat().st_mtime_ns, stamp)
            execution = json.loads((root / "page_execution.json").read_text())
            self.assertEqual(execution["checkpoint_executed"], 0)
            self.assertGreater(execution["checkpoint_reused"], 0)

    def test_failed_page_does_not_discard_healthy_page_or_repeat_it_on_resume(self):
        with tempfile.TemporaryDirectory(prefix="topology-page-failure-") as temporary:
            root = Path(temporary)
            write_deck(root)
            spec = {"slide_id": "slide_001", "title": "Control", "message": "Compare signals.",
                    "visual_intent": "Two signals"}
            atomic_write_json(root / "slide_specs/slides/slide_001/slide_spec.json", spec)
            atomic_write_json(root / "slide_specs/run_manifest.json", {"status": "succeeded"})
            with self.assertRaises(PagePhaseError) as caught:
                run_parallel_page_pipeline(output_dir=root, config=offline_config(), workers=2,
                    start_at="diffusion", stop_after="diffusion", mock_api=True, mock_diffusion=True)
            self.assertEqual(caught.exception.errors[0]["slide_id"], "slide_002")
            failure = json.loads((root / "page_failures.json").read_text())
            self.assertEqual(failure["healthy_page_ids"], ["slide_001"])
            healthy = root / "diffusion/skeletons/slide_001.png"
            stamp = healthy.stat().st_mtime_ns
            atomic_write_json(root / "slide_specs/slides/slide_002/slide_spec.json", {**spec, "slide_id": "slide_002"})
            run_parallel_page_pipeline(output_dir=root, config=offline_config(), workers=1,
                start_at="diffusion", stop_after="diffusion", mock_api=True, mock_diffusion=True)
            self.assertEqual(healthy.stat().st_mtime_ns, stamp)
            self.assertTrue((root / "diffusion/skeletons/slide_002.png").is_file())
            self.assertEqual(json.loads((root / "run_manifest.json").read_text())["status"], "succeeded")

    def test_saved_critic_is_reused_after_later_local_interruption(self):
        placement = _load_text_placement()
        placement.VLM_MODEL = "offline-chat"
        plan = {"slide_id": "slide_001", "background_path": "/unused.png",
            "canvas_px": {"width": 1536, "height": 864}, "groups": [{"group_id": "g", "member_region_ids": ["body"]}],
            "elements": [{"region_id": "body", "role": "body", "text": "Frozen text", "bbox_px": [100, 200, 500, 200],
                          "font": {"face": "Noto Sans", "size_pt": 18, "minimum_pt": 18, "maximum_pt": 24},
                          "presentation": {"mode": "direct_text", "text_color": "111111"}}]}
        client = SimpleNamespace(call_count=0)
        def review(**kwargs):
            client.call_count += 1
            return {"issues": [], "patches": []}, None
        with tempfile.TemporaryDirectory() as temporary, patch.object(placement, "call_vlm_json", side_effect=review):
            args = dict(output_dir=Path(temporary), work_dir=Path(temporary), slide_ids=["slide_001"],
                initial_rendered={"slide_001": Path("/unused.png")}, text_placement=placement,
                client=client, max_tokens=1800, deck_plan={"slides": []})
            _run_critic_rounds(plans_by_slide={"slide_001": copy.deepcopy(plan)}, **args)
            _run_critic_rounds(plans_by_slide={"slide_001": copy.deepcopy(plan)}, **args)
            self.assertEqual(client.call_count, 1)

    def test_missing_decorated_asset_invalidates_page_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = root / "decorated_holders/slide_001/frame.png"
            calls = []
            def prepare():
                calls.append(True)
                atomic_write_json(root / "composition_plan.json", {"background_path": str(root / "background.png"),
                    "elements": [{"presentation": {"quiet_frame_path": str(frame)}}]})
                atomic_write_json(root / "page_result.json", {})
                for path in (frame, root / "background.png", root / "rounds/accepted/deck.pptx",
                             root / "rounds/accepted/deck.pdf", root / "rounds/accepted/slide_001/preview.png"):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"fixture")
                return {"slide_id": "slide_001"}
            args = dict(output_dir=root, step="page", fingerprint="frozen-inputs",
                        required_outputs=lambda result: _page_outputs(root, "slide_001"), fn=prepare)
            run_checkpoint(**args)
            run_checkpoint(**args)
            self.assertEqual(len(calls), 1)
            frame.unlink()
            run_checkpoint(**args)
            self.assertEqual(len(calls), 2)
            self.assertTrue(frame.is_file())


if __name__ == "__main__":
    unittest.main()
