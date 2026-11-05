from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from integration.config import PipelineConfig, Size
from integration.final_background import _generate_or_resume, render_mock_asset_board, run_background_set
from integration.html_planning import JournaledHtmlPlanningClient, commit_html_typography
from integration.page_pipeline import complete_slide, run_page_pipeline
from integration.page_runtime import atomic_write_json
from integration.text_generation import build_parsed_text_plan
from v9_deck_planning.apimart import ApiMartImageTaskFailure

ROOT = Path(__file__).resolve().parents[1]


def offline_config() -> PipelineConfig:
    return PipelineConfig(
        project_root=ROOT, env_path=ROOT / "NO_CONFIGURATION_USED", api_root="https://offline.invalid/v1",
        api_key_env="NO_KEY", chat_model="offline", image_model="gpt-image-2-official",
        image_size=Size(1536, 864), image_resolution="1k", image_quality="medium", image_max_attempts=2,
        chat_max_tokens=4096, vlm_max_tokens=4096, timeout_seconds=1, poll_initial_delay_seconds=0,
        poll_interval_seconds=0, poll_timeout_seconds=1, diffusion_model_path=Path("/unused/mock-model"),
        diffusion_size=Size(1024, 576), diffusion_steps=5, diffusion_guidance_scale=0,
        diffusion_seed=42, deck_font="Noto Sans",
    )


def write_fixture(root: Path, count: int) -> list[str]:
    ids = [f"slide_{i:03d}" for i in range(1, count + 1)]
    deck = {"topic": "Feedback", "audience": "students", "deck_goal": "Explain feedback.",
            "style": "clean editorial illustration", "slides": [
                {"slide_id": sid, "working_title": f"Feedback {index}",
                 "rough_message": "Observe the outcome and update the next choice."}
                for index, sid in enumerate(ids, 1)]}
    text = {"slides": [{"slide_id": sid, "title": f"Feedback {index}", "subtitle": None,
                         "text_01": "Observe the outcome. Update the next choice."}
                        for index, sid in enumerate(ids, 1)]}
    atomic_write_json(root / "step2/deck_plan.json", deck)
    atomic_write_json(root / "step2/design_system.json", {"style_prompt": deck["style"]})
    Image.new("RGB", (1536, 864), (245, 244, 240)).save(root / "step2/canonical_background.png")
    atomic_write_json(root / "text_generation/text_plan.json", text)
    atomic_write_json(root / "text_generation/parsed_text_plan.json", build_parsed_text_plan(text))
    atomic_write_json(root / "expression_logic/run_manifest.json", {
        "stage": "deck_level_expression_logic_planning", "status": "succeeded"})
    logic = {"slides": [{"slide_id": sid, "expression_logic": "Pair the explanation with one substantial feedback scene."}
                         for sid in ids]}
    atomic_write_json(root / "expression_logic/expression_logic_plan.json", logic)
    for item in logic["slides"]:
        atomic_write_json(root / "expression_logic/slides" / item["slide_id"] / "expression_logic.json", item)
    return ids


class PagePipelineTests(unittest.TestCase):
    def test_ordinary_cli_restart_reuses_all_shared_planning_calls(self) -> None:
        import run_pipeline
        with tempfile.TemporaryDirectory(prefix="tf-shared-planning-resume-") as temporary:
            root = Path(temporary)
            args = Namespace(output_dir=root, input=None, content="Create a 3-slide deck explaining feedback.",
                style="clean editorial illustration", start_at="step1", stop_after="expression_logic",
                mock_api=True, mock_images=False, mock_diffusion=True, slide_limit=None,
                page_workers=5, gpu_devices=None)
            with patch.object(run_pipeline, "parse_args", return_value=args), \
                 patch.object(run_pipeline, "load_config", return_value=offline_config()), \
                 patch.object(run_pipeline, "run_deck_planning", wraps=run_pipeline.run_deck_planning) as step1, \
                 patch.object(run_pipeline, "run_text_generation", wraps=run_pipeline.run_text_generation) as text, \
                 patch.object(run_pipeline, "run_style_asset_generation", wraps=run_pipeline.run_style_asset_generation) as style, \
                 patch.object(run_pipeline, "run_expression_logic_planning", wraps=run_pipeline.run_expression_logic_planning) as logic, \
                 patch("builtins.print"):
                self.assertEqual(run_pipeline.main(), 0)
                paths = [root / "step1/deck_plan.json", root / "text_generation/text_plan.json",
                         root / "step2/canonical_background.png", root / "expression_logic/expression_logic_plan.json"]
                mtimes = [path.stat().st_mtime_ns for path in paths]
                self.assertEqual(run_pipeline.main(), 0)
                self.assertEqual([path.stat().st_mtime_ns for path in paths], mtimes)
                self.assertEqual([call.call_count for call in (step1, text, style, logic)], [1, 1, 1, 1])
                # A missing canonical image invalidates just that global stage.
                (root / "step2/canonical_background.png").unlink()
                self.assertEqual(run_pipeline.main(), 0)
                self.assertEqual([call.call_count for call in (step1, text, style, logic)], [1, 1, 2, 1])
                self.assertTrue((root / "step2/canonical_background.png").is_file())

    def test_main_failure_replaces_stale_success_and_preserves_previous_context(self) -> None:
        import run_pipeline
        with tempfile.TemporaryDirectory(prefix="tf-root-failure-") as temporary:
            root = Path(temporary)
            previous = {"status": "succeeded", "slide_count": 5}
            atomic_write_json(root / "run_manifest.json", previous)
            with patch.object(run_pipeline, "parse_args", return_value=Namespace(output_dir=root)), \
                 patch.object(run_pipeline, "_run", side_effect=RuntimeError("final assembly failed")):
                with self.assertRaisesRegex(RuntimeError, "final assembly failed"):
                    run_pipeline.main()
            failed = json.loads((root / "run_manifest.json").read_text())
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["previous_manifest"], previous)
            self.assertEqual(failed["error_type"], "RuntimeError")

    def test_missing_or_corrupt_derived_assets_rebuild_without_another_image_submission(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tf-derived-layer-repair-") as temporary:
            root = Path(temporary)
            ids = write_fixture(root, 1)
            sid = ids[0]
            atomic_write_json(root / "html_compile/slides" / sid / "html_layout.json", {
                "visual_regions": [{"region_id": "scene", "bbox_px": [700, 180, 600, 450],
                                    "visual_brief": "Show one substantial feedback scene."}],
                "text_regions": [], "holders": [],
            })
            class Client:
                submissions = 0
                downloads = 0
                def submit_image_task(self, payload):
                    self.submissions += 1
                    return "saved_paid_task"
                def poll_task(self, task_id):
                    return {"status": "completed", "result": {"images": [{"url": "unused"}]}}
                def download(self, url, output_path):
                    self.downloads += 1
                    objects = json.loads((root / "final_background" / sid / "prepared/generation_objects.json").read_text())
                    render_mock_asset_board(objects, "#00ff00", output_path)
            client = Client()
            job = dict(output_dir=str(root), slide_id=sid, slide_ids=ids, original_index=0,
                       config=offline_config(), start_at="background", stop_after="background",
                       mock_api=False, mock_images=False, mock_diffusion=False)
            def generate(**kwargs):
                return run_background_set(**kwargs, client_factory=lambda: client)
            layer_manifest = root / "final_background" / sid / "layer_manifest.json"
            with patch("integration.page_pipeline.run_background_set", side_effect=generate):
                complete_slide(job)
                layer = Path(json.loads(layer_manifest.read_text())["layers"][0]["path"])
                raw = root / "final_background" / sid / "live/generated.png"
                raw_mtime = raw.stat().st_mtime_ns
                layer.unlink()
                complete_slide(job)
                self.assertTrue(layer.is_file())
                layer.write_bytes(b"interrupted output")
                complete_slide(job)
                with Image.open(layer) as image:
                    image.verify()
                (root / "final_background" / sid / "run_manifest.json").write_text('{"interrupted":')
                complete_slide(job)
                self.assertEqual(raw.stat().st_mtime_ns, raw_mtime)
                self.assertEqual(client.submissions, 1)
                self.assertEqual(client.downloads, 1)

    def test_explicit_compile_resume_honors_new_html_edits(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tf-explicit-compile-") as temporary:
            root = Path(temporary)
            ids = write_fixture(root, 1)
            html_dir = root / "html_planning/slides" / ids[0]
            html_dir.mkdir(parents=True)
            (html_dir / "layout_committed.html").write_text("previous committed HTML")
            (html_dir / "layout.html").write_text("user's latest HTML edit")
            atomic_write_json(html_dir / "visual_generation_groups.json", {"groups": []})
            job = dict(output_dir=str(root), slide_id=ids[0], slide_ids=ids, original_index=0,
                       config=offline_config(), start_at="html_compile", stop_after="html_compile")
            def compile_page(**kwargs):
                current = html_dir.joinpath("layout.html").read_text()
                layout_dir = root / "html_compile/slides" / ids[0]
                atomic_write_json(layout_dir / "html_layout.json", {"source": current})
                for name in ("layout_preview.png", "background_control.png"):
                    Image.new("RGB", (8, 8), "white").save(layout_dir / name)
                return {"slides": [{"slide_id": ids[0], "source": current}], "status": "succeeded"}
            with patch("integration.page_pipeline.run_html_to_coordinates", side_effect=compile_page) as compiler:
                self.assertEqual(complete_slide(job)["compile"]["slides"][0]["source"], "user's latest HTML edit")
                (html_dir / "layout.html").write_text("another explicit edit")
                self.assertEqual(complete_slide(job)["compile"]["slides"][0]["source"], "another explicit edit")
                self.assertEqual(compiler.call_count, 2)

    def test_interrupted_page_replays_completed_calls_and_binds_image_bytes(self) -> None:
        class Client:
            last_usage = {"total_tokens": 17}
            calls = 0
            def vlm_html(self, **kwargs):
                self.calls += 1
                return "<html>saved planner answer</html>"
            def vlm_json(self, **kwargs):
                self.calls += 1
                return '{"needs_rework":false}'
        with tempfile.TemporaryDirectory(prefix="tf-model-journal-") as temporary:
            root = Path(temporary)
            image = root / "preview.png"
            Image.new("RGB", (16, 16), "white").save(image)
            kwargs = dict(model="test", prompt="frozen prompt", image_path=image, max_tokens=20)
            client = Client()
            first = JournaledHtmlPlanningClient(client, root / "page_1/model_calls")
            saved_html = first.vlm_html(**kwargs)
            saved_review = first.vlm_json(**kwargs)
            # Browser/patch work can fail here; a new process reconstructs the wrapper.
            resumed = JournaledHtmlPlanningClient(client, root / "page_1/model_calls")
            self.assertEqual(resumed.vlm_html(**kwargs), saved_html)
            self.assertEqual(resumed.vlm_json(**kwargs), saved_review)
            self.assertEqual(client.calls, 2)
            self.assertEqual(resumed.last_usage, client.last_usage)
            self.assertEqual(resumed.reused_calls, 2)
            Image.new("RGB", (16, 16), "black").save(image)
            resumed.vlm_html(**kwargs)
            self.assertEqual(client.calls, 3)
            # Page 2 has its own paid-call authority even for an identical prompt.
            JournaledHtmlPlanningClient(client, root / "page_2/model_calls").vlm_html(**kwargs)
            self.assertEqual(client.calls, 4)

    def test_unknown_vlm_post_is_not_repeated(self) -> None:
        class Client:
            last_usage = None
            calls = 0
            def chat_html(self, **kwargs):
                self.calls += 1
                raise TimeoutError("unknown remote outcome")
        with tempfile.TemporaryDirectory(prefix="tf-vlm-unknown-") as temporary:
            client = Client()
            kwargs = dict(model="test", prompt="frozen", max_tokens=10)
            first = JournaledHtmlPlanningClient(client, Path(temporary))
            with self.assertRaises(TimeoutError):
                first.chat_html(**kwargs)
            resumed = JournaledHtmlPlanningClient(client, Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "outcome is unknown"):
                resumed.chat_html(**kwargs)
            self.assertEqual(client.calls, 1)

    def test_typography_reuses_answer_after_local_profile_commit_failure(self) -> None:
        class Client:
            last_usage = {"total_tokens": 7}
            calls = 0
            def vlm_json(self, **kwargs):
                self.calls += 1
                return '{"title_font_size_px":40,"subtitle_font_size_px":null}'
        with tempfile.TemporaryDirectory(prefix="tf-typography-journal-") as temporary:
            root = Path(temporary)
            html = root / "layout.html"
            html.write_text('''<!doctype html><html><body style="margin:0"><section class="slide" style="position:relative;width:1536px;height:864px">
<div data-module-id="header" style="position:absolute;left:60px;top:40px;width:1100px;height:100px">
<h1 data-region-id="title" data-content-key="title" data-role="title" style="margin:0;width:1000px;height:80px;font:700 48px 'Noto Sans'">Feedback</h1></div></section></body></html>''')
            preview = root / "preview.png"
            Image.new("RGB", (1536, 864), "white").save(preview)
            client = Client()
            record = {"slide_id": "slide_001", "html": str(html), "final_preview": str(preview), "precommit_text_repairs": {}}
            kwargs = dict(output_dir=root, slide_ids=["slide_001"], records=[record], config=offline_config(),
                          mock_vlm=False, client_factory=lambda: client)
            with patch("integration.html_planning.apply_deck_typography_profile", side_effect=RuntimeError("local interruption")):
                self.assertEqual(commit_html_typography(**kwargs)["typography"]["status"], "warning")
            result = commit_html_typography(**kwargs)
            self.assertEqual(result["typography"]["status"], "accepted")
            self.assertEqual(client.calls, 1)

    def test_real_page_chains_overlap_and_resume_after_final_assembly_failure(self) -> None:
        # Three pages on two persistent owners also exercises a rendezvous with
        # more pages than process slots; owners must drain their own queue first.
        with tempfile.TemporaryDirectory(prefix="tf-page-chain-") as temporary:
            root = Path(temporary)
            ids = write_fixture(root, 3)
            kwargs = dict(output_dir=root, slide_ids=ids, config=offline_config(), start_at="diffusion",
                          stop_after="assembly", page_workers=2, mock_api=True, mock_diffusion=True)
            with patch("integration.page_pipeline.run_reused_text_assembly", side_effect=RuntimeError("assembly interrupted")):
                with self.assertRaisesRegex(RuntimeError, "assembly interrupted"):
                    run_page_pipeline(**kwargs)
            checkpoint = lambda sid, step: json.loads((root / "pages" / sid / "checkpoints" / f"{step}.json").read_text())
            first_html = checkpoint(ids[0], "html_planning")
            last_diffusion = checkpoint(ids[2], "diffusion")
            self.assertLess(first_html["finished_at"], last_diffusion["started_at"])
            # Page 1 advances to assets before page 3 has even compiled.
            self.assertLess(checkpoint(ids[0], "background")["finished_at"],
                            checkpoint(ids[2], "html_compile")["started_at"])
            owners = json.loads((root / "page_execution.json").read_text())["events"]
            starts = [event for event in owners if event["kind"] == "started"]
            for sid in ids:
                page_events = [event for event in starts if event["slide_id"] == sid]
                self.assertEqual(len({event["pid"] for event in page_events}), 1)
            draft_intervals = [event for event in owners if event["kind"] == "completed" and event["phase"] == "draft"]
            self.assertLess(max(event["started_at"] for event in draft_intervals[:2]),
                            min(event["finished_at"] for event in draft_intervals[:2]))
            protected = [root / "html_planning/slides" / sid / "raw_response.txt" for sid in ids]
            protected += [root / "pages" / sid / "checkpoints" / "diffusion.json" for sid in ids]
            protected += [root / "assembly" / sid / "composition_plan.json" for sid in ids]
            mtimes = {str(path): path.stat().st_mtime_ns for path in protected}
            with patch("integration.page_pipeline.commit_html_typography", side_effect=AssertionError("must reuse typography")):
                manifests = run_page_pipeline(**kwargs)
            self.assertEqual(mtimes, {str(path): path.stat().st_mtime_ns for path in protected})
            self.assertEqual([item["slide_id"] for item in manifests[-1]["slides"]], ids)
            self.assertTrue((root / "assembly/deck.pptx").is_file())
            self.assertTrue((root / "assembly/deck.pdf").is_file())
            self.assertEqual(manifests[-1]["critic_rounds"]["status"], "removed")
            render = json.loads((root / "assembly/pptx_render_check.json").read_text())
            self.assertEqual(render["status"], "strict_pass")

    def test_stop_and_resume_keep_canonical_seed_and_skip_gpu_phase(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tf-page-stop-") as temporary:
            root = Path(temporary)
            ids = write_fixture(root, 2)
            manifests = run_page_pipeline(output_dir=root, slide_ids=ids, config=offline_config(),
                start_at="diffusion", stop_after="diffusion", page_workers=2, mock_api=True, mock_diffusion=True)
            self.assertEqual([item["seed"] for item in manifests[0]["results"]], [42, 43])
            self.assertFalse((root / "html_planning").exists())
            manifests = run_page_pipeline(output_dir=root, slide_ids=ids, config=offline_config(),
                start_at="html_planning", stop_after="html_planning", page_workers=2, mock_api=True, mock_diffusion=True)
            self.assertTrue(manifests[0]["resumed"])
            execution = json.loads((root / "page_execution.json").read_text())
            self.assertEqual(execution["gpu_workers"], 0)
            self.assertTrue(all((root / "html_planning/slides" / sid / "layout_committed.html").is_file() for sid in ids))
            self.assertFalse((root / "html_compile").exists())

    def test_unknown_image_submission_is_not_paid_for_again(self) -> None:
        class Client:
            submissions = 0
            def submit_image_task(self, payload):
                self.submissions += 1
                raise TimeoutError("unknown remote outcome")
        with tempfile.TemporaryDirectory(prefix="tf-submission-unknown-") as temporary:
            client = Client()
            kwargs = dict(client=client, payload={"model": "test"}, bundle_sha256="same", live_dir=Path(temporary))
            with self.assertRaises(TimeoutError):
                _generate_or_resume(**kwargs)
            with self.assertRaisesRegex(RuntimeError, "outcome is unknown"):
                _generate_or_resume(**kwargs)
            self.assertEqual(client.submissions, 1)
            state = json.loads((Path(temporary) / "task.json").read_text())
            self.assertEqual(state["status"], "submission_unknown")

    def test_moderation_terminal_failure_is_not_resubmitted_on_resume(self) -> None:
        class Client:
            submissions = 0
            def submit_image_task(self, payload):
                self.submissions += 1
                return "terminal_task"
            def poll_task(self, task_id):
                raise ApiMartImageTaskFailure(task_id, {"type": "content_review"})
        with tempfile.TemporaryDirectory(prefix="tf-terminal-task-") as temporary:
            client = Client()
            kwargs = dict(client=client, payload={"model": "test"}, bundle_sha256="same", live_dir=Path(temporary))
            for _ in range(2):
                with self.assertRaises(ApiMartImageTaskFailure):
                    _generate_or_resume(**kwargs)
            self.assertEqual(client.submissions, 1)

    def test_real_english_safety_review_error_never_resubmits_on_resume(self) -> None:
        real_error = {"code": "task_failed", "type": "task_failed", "param": "", "message": (
            "Your request did not pass content safety review. Please adjust your prompt or input files and try again. "
            "No image was generated. The model provider still charges for the input it reviewed, so this task is "
            "billed for input only, with no output cost.")}
        cases = [real_error,
                 {"code": "task_failed", "message": "Did NOT pass \n THE\tSAFETY   Review."},
                 {"code": "task_failed", "message": "This task is BILLED  for\n input\tONLY."}]
        for error in cases:
            with self.subTest(error=error), tempfile.TemporaryDirectory(prefix="tf-real-safety-terminal-") as temporary:
                class Client:
                    submissions = 0
                    def submit_image_task(self, payload):
                        self.submissions += 1
                        return "terminal_task"
                    def poll_task(self, task_id):
                        raise ApiMartImageTaskFailure(task_id, error)
                client = Client()
                kwargs = dict(client=client, payload={"model": "test"}, bundle_sha256="same", live_dir=Path(temporary))
                for _ in range(2):
                    with self.assertRaises(ApiMartImageTaskFailure):
                        _generate_or_resume(**kwargs)
                self.assertEqual(client.submissions, 1)


if __name__ == "__main__":
    unittest.main()
