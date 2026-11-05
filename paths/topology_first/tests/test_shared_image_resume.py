from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from v9_deck_planning.image_task_journal import run_persisted_image_task


class Client:
    def __init__(self):
        self.submits = self.polls = self.downloads = 0
        self.fail_submit = self.fail_poll = self.fail_download = False

    def submit_image_task(self, payload):
        self.submits += 1
        if self.fail_submit:
            raise TimeoutError("unknown submission")
        return "known-task"

    def poll_task(self, task_id):
        self.polls += 1
        if self.fail_poll:
            raise TimeoutError("still processing")
        return {"status": "completed", "result": {"images": [{"url": ["https://offline.invalid/image"]}]}}

    def download(self, url, path):
        self.downloads += 1
        if self.fail_download:
            raise ConnectionError("interrupted download")
        Image.new("RGB", (20, 20), "white").save(path, format="PNG")


class SharedImageResumeTests(unittest.TestCase):
    def test_resume_poll_and_download_never_resubmits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, client = Path(temporary), Client()
            kwargs = dict(client=client, payload={"model": "fixture", "prompt": "same"},
                          task_path=root / "task.json", output_path=root / "raw.png")
            client.fail_poll = True
            with self.assertRaises(TimeoutError):
                run_persisted_image_task(**kwargs)
            client.fail_poll, client.fail_download = False, True
            with self.assertRaises(ConnectionError):
                run_persisted_image_task(**kwargs)
            client.fail_download = False
            run_persisted_image_task(**kwargs)
            self.assertEqual(client.submits, 1)
            run_persisted_image_task(**kwargs)
            self.assertEqual((client.submits, client.downloads), (1, 2))
            (root / "raw.png").unlink()
            client.fail_poll = True
            run_persisted_image_task(**kwargs)
            self.assertEqual(client.submits, 1)
            self.assertEqual(client.polls, 2)

    def test_unknown_post_is_not_blindly_repeated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, client = Path(temporary), Client()
            client.fail_submit = True
            kwargs = dict(client=client, payload={"prompt": "same"}, task_path=root / "task.json", output_path=root / "raw.png")
            with self.assertRaises(TimeoutError):
                run_persisted_image_task(**kwargs)
            with self.assertRaisesRegex(RuntimeError, "unknown"):
                run_persisted_image_task(**kwargs)
            self.assertEqual(client.submits, 1)


if __name__ == "__main__":
    unittest.main()
