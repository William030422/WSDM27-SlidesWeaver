from __future__ import annotations

import socket
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from v9_deck_planning.apimart import ApiMartClient


class ApiMartRetryTests(unittest.TestCase):
    def test_poll_retries_transient_dns_with_bounded_backoff(self) -> None:
        client = self._client()
        responses = [
            self._request_error(socket.EAI_AGAIN, "Temporary failure in name resolution"),
            {"data": {"status": "completed", "result": {}}},
        ]

        with patch.object(client, "_get_json", side_effect=responses) as request, patch(
            "v9_deck_planning.apimart.time.sleep"
        ) as sleep:
            task = client.poll_task("task_123")

        self.assertEqual(task["status"], "completed")
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(1.0)

    def test_poll_stops_after_seven_seconds_of_backoff(self) -> None:
        client = self._client()
        error = self._request_error(
            socket.EAI_AGAIN, "Temporary failure in name resolution"
        )

        with patch.object(client, "_get_json", side_effect=error) as request, patch(
            "v9_deck_planning.apimart.time.sleep"
        ) as sleep, self.assertRaisesRegex(RuntimeError, "Temporary failure"):
            client.poll_task("task_123")

        self.assertEqual(request.call_count, 4)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [1.0, 2.0, 4.0],
        )

    def test_poll_does_not_retry_permanent_dns_error(self) -> None:
        client = self._client()
        error = self._request_error(socket.EAI_NONAME, "Name or service not known")

        with patch.object(client, "_get_json", side_effect=error) as request, patch(
            "v9_deck_planning.apimart.time.sleep"
        ) as sleep, self.assertRaisesRegex(RuntimeError, "Name or service"):
            client.poll_task("task_123")

        request.assert_called_once()
        sleep.assert_not_called()

    def test_download_retries_transient_connection_reset(self) -> None:
        client = self._client()
        with tempfile.TemporaryDirectory(prefix="rl4ppt-apimart-download-") as temporary:
            output = Path(temporary) / "generated.png"
            with patch.object(
                client,
                "_download_bytes",
                side_effect=[ConnectionResetError("reset"), b"image-bytes"],
            ) as download, patch("v9_deck_planning.apimart.time.sleep") as sleep:
                client.download("https://example.invalid/image.png", output)

            self.assertEqual(output.read_bytes(), b"image-bytes")

        self.assertEqual(download.call_count, 2)
        sleep.assert_called_once_with(1.0)

    def test_image_submission_post_is_not_retried(self) -> None:
        client = self._client()
        error = self._request_error(
            socket.EAI_AGAIN, "Temporary failure in name resolution"
        )
        with patch.object(client, "_post_json", side_effect=error) as request, patch(
            "v9_deck_planning.apimart.time.sleep"
        ) as sleep, self.assertRaisesRegex(RuntimeError, "Temporary failure"):
            client.submit_image_task({"model": "test"})

        request.assert_called_once()
        sleep.assert_not_called()

    def test_vlm_json_requests_json_object_output_on_responses_api(self) -> None:
        client = self._client()
        with tempfile.TemporaryDirectory(prefix="rl4ppt-vlm-json-") as temporary:
            image_path = Path(temporary) / "preview.png"
            image_path.write_bytes(b"not-a-real-image")
            with patch.object(
                client,
                "_post_json",
                return_value={"output_text": '{"ok":true}'},
            ) as request:
                result = client.vlm_json(
                    model="test-model",
                    prompt="Return JSON.",
                    image_path=image_path,
                )

        self.assertEqual(result, '{"ok":true}')
        path, payload = request.call_args.args
        self.assertEqual(path, "/responses")
        self.assertEqual(payload["text"], {"format": {"type": "json_object"}})

    @staticmethod
    def _client() -> ApiMartClient:
        return ApiMartClient(
            poll_initial_delay_seconds=0,
            poll_interval_seconds=0,
            poll_timeout_seconds=30,
        )

    @staticmethod
    def _request_error(error_number: int, message: str) -> RuntimeError:
        reason = socket.gaierror(error_number, message)
        url_error = urllib.error.URLError(reason)
        error = RuntimeError(f"APIMART request failed: {url_error}")
        error.__cause__ = url_error
        return error


if __name__ == "__main__":
    unittest.main()
