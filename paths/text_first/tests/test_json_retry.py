from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from v9_deck_planning.json_utils import call_json_with_retries, extract_json


class JsonRetryTests(unittest.TestCase):
    def test_retries_parser_rejection_and_stops_after_success(self) -> None:
        responses = ['{"slides":[]}', '{"slides":[{"slide_id":"slide_001"}]}']
        prompts: list[str] = []

        def call(prompt: str) -> str:
            prompts.append(prompt)
            return responses[len(prompts) - 1]

        def parse(raw: str) -> dict:
            payload = extract_json(raw)
            if not payload.get("slides"):
                raise ValueError("slides must not be empty")
            return payload

        with tempfile.TemporaryDirectory(prefix="rl4ppt-json-retry-") as temporary:
            response_path = Path(temporary) / "raw_response.json"
            parsed, raw = call_json_with_retries(
                prompt="Return the slide JSON.",
                call=call,
                parse=parse,
                response_path=response_path,
            )

            self.assertEqual(len(prompts), 2)
            self.assertIn("previous response was rejected", prompts[1])
            self.assertEqual(parsed["slides"][0]["slide_id"], "slide_001")
            self.assertEqual(raw, responses[1])
            self.assertEqual(response_path.read_text(encoding="utf-8"), responses[1])
            self.assertTrue(
                (Path(temporary) / "raw_response_attempt_01.json").is_file()
            )
            self.assertTrue(
                (Path(temporary) / "raw_response_attempt_02.json").is_file()
            )
            self.assertFalse(
                (Path(temporary) / "raw_response_attempt_03.json").exists()
            )

    def test_stops_after_three_parser_rejections(self) -> None:
        call_count = 0

        def call(_: str) -> str:
            nonlocal call_count
            call_count += 1
            return '{"dummy":null}'

        def parse(_: str) -> dict:
            raise ValueError("required field is missing")

        with tempfile.TemporaryDirectory(prefix="rl4ppt-json-retry-limit-") as temporary:
            with self.assertRaisesRegex(ValueError, "required field"):
                call_json_with_retries(
                    prompt="Return JSON.",
                    call=call,
                    parse=parse,
                    response_path=Path(temporary) / "raw_response.json",
                )

            self.assertEqual(call_count, 3)
            self.assertTrue(
                (Path(temporary) / "raw_response_attempt_03.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
