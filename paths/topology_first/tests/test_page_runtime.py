from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from integration.gpu_runtime import generate_diffusion_image
from integration.page_runtime import (
    PageExecutor, PagePhaseError, RunLock, atomic_write_json, current_owner_id,
    fingerprint_paths, run_checkpoint,
    resource_lease,
)


def _probe_job(job):
    time.sleep(job.get("delay", 0))
    if job.get("expect_drafts"):
        count = len(list((Path(job["directory"]) / "draft").glob("*.json")))
        if count != job["expect_drafts"]:
            raise AssertionError(f"barrier released early: {count}")
    result = {"slide_id": job["slide_id"], "index": job["original_index"],
              "pid": os.getpid(), "owner_id": current_owner_id()}
    if job.get("directory"):
        path = Path(job["directory"]) / job.get("phase", "draft") / f"{job['slide_id']}.json"
        atomic_write_json(path, result)
    return result


def _checkpoint_job(job):
    path = Path(job["directory"]) / f"{job['slide_id']}.json"

    def execute():
        result = _probe_job(job)
        atomic_write_json(path, result)
        return result

    return run_checkpoint(Path(job["directory"]) / job["slide_id"], "probe", job["version"], [path], execute)


def _crash_job(job):
    if job["original_index"] == 0:
        if job.get("child_pid_path"):
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            Path(job["child_pid_path"]).write_text(str(child.pid))
        os._exit(17)
    return _probe_job(job)


def _gpu_job(job):
    result = generate_diffusion_image(config=job["config"], prompt="a quiet composition", seed=100 + job["original_index"],
                                      output_path=Path(job["directory"]) / f"{job['slide_id']}.png")
    result["owner_pid"] = os.getpid()
    result["owner_imported_torch"] = "torch" in sys.modules
    return result


def _pause_with_child(job):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    atomic_write_json(Path(job["pid_path"]), {"owner_pid": os.getpid(), "child_pid": child.pid})
    time.sleep(60)


def _oversized_result(job):
    if job["original_index"] == 0:
        payload = "x" * (32 * 1024 * 1024)
        def die():
            time.sleep(0.01)
            os._exit(23)
        threading.Thread(target=die, daemon=True).start()
        return {"payload": payload}
    return _probe_job(job)


def _oversized_request(job):
    if job["original_index"] == 0:
        prompt = "x" * (32 * 1024 * 1024)
        def die():
            time.sleep(0.01)
            os._exit(24)
        threading.Thread(target=die, daemon=True).start()
        return generate_diffusion_image(config=job["config"], prompt=prompt,
            output_path=Path(job["directory"]) / "crashed.png", seed=1)
    return _gpu_job(job)


def _run_oversized_case(kind, directory):
    config = _mock_config() if kind == "request" else None
    with PageExecutor(directory, workers=2, config=config, mock_diffusion=True) as executor:
        try:
            executor.run_phase(_entry(f"_oversized_{kind}"), _jobs(2, directory=directory, config=config))
        except PagePhaseError as exc:
            if exc.results[1] is None or {item["original_index"] for item in exc.errors} != {0}:
                raise AssertionError(f"healthy page lost: {exc.errors}")
        else:
            raise AssertionError("fault injection did not terminate its owner")


def _entry(function):
    return f"{__name__}:{function}"


def _jobs(count, **kwargs):
    return [{"slide_id": f"slide_{index + 1:03}", "original_index": index, **kwargs} for index in range(count)]


def _mock_config():
    return SimpleNamespace(diffusion_model_path=Path("/nonexistent/no-model-loading"),
                           diffusion_size=SimpleNamespace(width=64, height=36),
                           diffusion_steps=5, diffusion_guidance_scale=0.0, gpu_timeout_seconds=10)


def _process_is_live(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
        return state != "Z"
    except (OSError, IndexError):
        # A process may disappear during the read itself, even after a
        # successful existence check. Disappearance is the expected outcome.
        return False


class PageRuntimeTests(unittest.TestCase):
    def test_nested_same_resource_lease_reuses_the_existing_permit(self):
        import integration.page_runtime as runtime

        class Permit:
            held = False
            def acquire(self):
                if self.held:
                    raise AssertionError("nested lease tried to acquire its own permit")
                self.held = True
            def release(self):
                self.held = False

        permit = Permit()
        with mock.patch.dict(runtime._RESOURCE_SEMAPHORES, {"render": permit}):
            with resource_lease("render"):
                with resource_lease("render"):
                    self.assertTrue(permit.held)
            self.assertFalse(permit.held)

    def test_five_owners_overlap_and_match_serial_results(self):
        with tempfile.TemporaryDirectory() as directory:
            runtimes = []
            for workers in (1, 5):
                with PageExecutor(Path(directory) / str(workers), workers=workers) as executor:
                    executor.run_phase(_entry("_probe_job"), _jobs(5), "warmup")
                    started = time.monotonic()
                    result = executor.run_phase(_entry("_probe_job"), _jobs(5, delay=0.25), "measured")
                    runtimes.append(time.monotonic() - started)
                    self.assertEqual([item["index"] for item in result], list(range(5)))
                    self.assertEqual(len({item["pid"] for item in result}), workers)
                metadata = json.loads((Path(directory) / str(workers) / "page_execution.json").read_text())
                self.assertEqual(metadata["actual_page_owners"], workers)
            self.assertLess(runtimes[1], runtimes[0] * 0.8)

    def test_six_and_eleven_pages_reach_barrier_with_stable_owners(self):
        for count in (6, 11):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                with PageExecutor(directory, workers=5) as executor:
                    draft = executor.run_phase(_entry("_probe_job"), _jobs(count, directory=directory), "draft")
                    final = executor.run_phase(_entry("_probe_job"), _jobs(count, directory=directory,
                                                expect_drafts=count, phase="final"), "final")
                self.assertEqual([row["pid"] for row in draft], [row["pid"] for row in final])
                self.assertEqual([row["owner_id"] for row in final], [i % 5 for i in range(count)])
                self.assertEqual(len(list((Path(directory) / "final").glob("*.json"))), count)

    def test_persistent_ready_owner_is_not_killed_by_next_phase_timeout(self):
        with tempfile.TemporaryDirectory() as directory, PageExecutor(directory, workers=1) as executor:
            first = executor.run_phase(_entry("_probe_job"), _jobs(1), "first")
            executor.startup_timeout = 0.01
            second = executor.run_phase(_entry("_probe_job"), _jobs(1, delay=0.1), "second")
            self.assertEqual(first[0]["pid"], second[0]["pid"])

    def test_duplicate_pages_are_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory, PageExecutor(directory) as executor:
            with self.assertRaises(ValueError):
                executor.run_phase(_entry("_probe_job"), _jobs(1) * 2)

    def test_run_lock_prevents_another_writer_and_releases_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            with RunLock(directory):
                with self.assertRaisesRegex(RuntimeError, "already in use"):
                    with RunLock(directory):
                        pass
                with PageExecutor(directory, workers=1, run_lock=False) as executor:
                    self.assertEqual(len(executor.run_phase(_entry("_probe_job"), _jobs(1))), 1)
            with RunLock(directory):
                pass

    def test_checkpoint_reuses_valid_outputs_and_invalidates_changed_or_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "asset.json"
            calls = []

            def execute():
                calls.append(1)
                atomic_write_json(output, {"version": len(calls)})
                return {"asset": str(output), "call": len(calls)}

            def invoke(version):
                return run_checkpoint(root, "asset", version, lambda result: [Path(result["asset"])], execute)

            self.assertEqual(invoke("a"), invoke("a"))
            self.assertEqual(len(calls), 1)
            output.write_text("corrupt")
            self.assertEqual(invoke("a")["call"], 2)
            output.unlink()
            self.assertEqual(invoke("a")["call"], 3)
            self.assertEqual(invoke("b")["call"], 4)
            before = fingerprint_paths([output], extra={"seed": 42})
            output.write_text("changed")
            self.assertNotEqual(before, fingerprint_paths([output], extra={"seed": 42}))

    def test_interrupted_checkpoint_reexecutes_but_completed_one_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoints" / "stage.json"
            atomic_write_json(checkpoint, {"status": "running", "fingerprint": "old"})
            calls = []

            def execute():
                calls.append(1)
                return {"complete": True}

            run_checkpoint(root, "stage", "new", [], execute)
            run_checkpoint(root, "stage", "new", [], execute)
            self.assertEqual(len(calls), 1)

    def test_checkpoint_events_count_this_runs_reuse(self):
        with tempfile.TemporaryDirectory() as directory, PageExecutor(directory, workers=1) as executor:
            jobs = _jobs(1, directory=directory, version="v1")
            first = executor.run_phase(_entry("_checkpoint_job"), jobs, "first")
            second = executor.run_phase(_entry("_checkpoint_job"), jobs, "resume")
            self.assertEqual(first, second)
            metadata = json.loads((Path(directory) / "page_execution.json").read_text())
            self.assertEqual(metadata["checkpoint_executed"], 1)
            self.assertEqual(metadata["checkpoint_reused"], 1)

    def test_owner_crash_preserves_healthy_results_and_can_restart(self):
        with tempfile.TemporaryDirectory() as directory, PageExecutor(directory, workers=2) as executor:
            with self.assertRaises(PagePhaseError) as raised:
                executor.run_phase(_entry("_crash_job"), _jobs(3), "crash")
            self.assertEqual(raised.exception.results[1]["index"], 1)
            self.assertEqual({error["original_index"] for error in raised.exception.errors}, {0, 2})
            result = executor.run_phase(_entry("_probe_job"), _jobs(3), "restart")
            self.assertEqual(len(result), 3)

    def test_owner_crash_stops_only_its_descendants(self):
        with tempfile.TemporaryDirectory() as directory, PageExecutor(directory, workers=1) as executor:
            path = Path(directory) / "child.pid"
            with self.assertRaises(PagePhaseError):
                executor.run_phase(_entry("_crash_job"), _jobs(1, child_pid_path=str(path)))
            child_pid = int(path.read_text())
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if not _process_is_live(child_pid):
                    break
                time.sleep(0.05)
            else:
                self.fail("owner crash left its child running")
            # The coordinator itself was not part of the killed process group.
            self.assertEqual(len(executor.run_phase(_entry("_probe_job"), _jobs(1))), 1)

    def test_one_mock_gpu_is_reused_across_more_than_five_pages_without_torch_in_owners(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _mock_config()
            with PageExecutor(directory, workers=5, config=config, mock_diffusion=True, gpu_devices=["mock:0"]) as executor:
                result = executor.run_phase(_entry("_gpu_job"), _jobs(6, directory=directory, config=config), "gpu")
            self.assertEqual(len({item["gpu_pid"] for item in result}), 1)
            self.assertEqual([item["seed"] for item in result], list(range(100, 106)))
            self.assertTrue(all(item["mock"] for item in result))
            self.assertFalse(any(item["owner_imported_torch"] for item in result))
            from PIL import Image
            for item in result:
                with Image.open(item["output"]) as image:
                    self.assertEqual(image.size, (64, 36))

    def test_gpu_exit_returns_an_error_without_hanging_page(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _mock_config()
            with PageExecutor(directory, workers=1, config=config, mock_diffusion=True) as executor:
                executor._gpu_pool.processes[0].kill()
                executor._gpu_pool.processes[0].join(timeout=2)
                started = time.monotonic()
                with self.assertRaises(PagePhaseError):
                    executor.run_phase(_entry("_gpu_job"), _jobs(1, directory=directory, config=config))
                self.assertLess(time.monotonic() - started, 5)

    def test_one_gpu_exiting_leaves_other_gpu_available_for_healthy_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _mock_config()
            with PageExecutor(directory, workers=5, config=config, mock_diffusion=True,
                              gpu_devices=["mock:0", "mock:1"]) as executor:
                executor._gpu_pool.processes[0].kill()
                executor._gpu_pool.processes[0].join(timeout=2)
                result = executor.run_phase(_entry("_gpu_job"), _jobs(6, directory=directory, config=config))
                self.assertEqual({item["device"] for item in result}, {"mock:1"})

    def test_coordinator_death_stops_owner_and_descendants_and_releases_run_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pids.json"
            job = _jobs(1, pid_path=str(path))
            root = Path(__file__).resolve().parents[1]
            child_paths = [str(root), str(root / "src"), str(root / "tests")]
            code = (
                "import sys\n"
                f"sys.path[:0] = {child_paths!r}\n"
                "from integration.page_runtime import PageExecutor\n"
                f"with PageExecutor({directory!r},workers=1) as executor:\n"
                f" executor.run_phase({_entry('_pause_with_child')!r}, {job!r})\n"
            )
            coordinator = subprocess.Popen([sys.executable, "-c", code])
            try:
                deadline = time.monotonic() + 10
                while not path.is_file() and time.monotonic() < deadline:
                    if coordinator.poll() is not None:
                        self.fail("coordinator exited before starting its page")
                    time.sleep(0.05)
                self.assertTrue(path.is_file())
                pids = json.loads(path.read_text())
                coordinator.kill()
                coordinator.wait(timeout=5)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    live = []
                    for pid in pids.values():
                        if _process_is_live(pid):
                            live.append(pid)
                    if not live:
                        break
                    time.sleep(0.05)
                else:
                    self.fail(f"coordinator death left workers running: {live}")
                with RunLock(directory):
                    pass
            finally:
                if coordinator.poll() is None:
                    coordinator.kill()
                    coordinator.wait(timeout=5)

    def test_owner_killed_mid_oversized_result_or_gpu_request_cannot_block_healthy_page(self):
        root = Path(__file__).resolve().parents[1]
        paths = [str(root), str(root / "src"), str(root / "tests")]
        for kind in ("result", "request"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                code = (f"import sys; sys.path[:0] = {paths!r}\n"
                        f"from {__name__} import _run_oversized_case\n"
                        f"_run_oversized_case({kind!r}, {directory!r})\n")
                result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=12)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
