from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v9_deck_planning.models import UserInput
from v9_deck_planning.pipeline import run_deck_planning
from v9_deck_planning.planner import deck_planning_prompt


ROOT = Path(__file__).resolve().parents[1]


class InvalidPlanClient:
    call_count = 0
    last_usage = None

    def chat_json(self, *, model: str, prompt: str, max_tokens: int) -> str:
        self.call_count += 1
        return '{"topic":"incomplete"}'


class SchemaToleranceTests(unittest.TestCase):
    def test_step1_prompt_omits_orchestration_owned_fields(self) -> None:
        prompt = deck_planning_prompt(
            UserInput(content="Explain feedback control in 2 slides.", style="anime")
        )

        schema = prompt.split("Return only one JSON object matching this schema:", 1)[1]
        self.assertNotIn('"stage"', schema)
        self.assertNotIn('"style"', schema)
        self.assertNotIn('"slide_count"', schema)
        self.assertNotIn('"slide_id"', schema)
        self.assertNotIn('"depends_on"', schema)
        self.assertNotIn('"purpose"', schema)

    def test_invalid_step1_schema_falls_back_after_successful_api_call(self) -> None:
        user_input = UserInput(
            content="Explain feedback control in 2 slides.",
            style="anime",
        )
        client = InvalidPlanClient()
        with tempfile.TemporaryDirectory(prefix="rl4ppt-step1-fallback-") as temporary:
            output_dir = Path(temporary)
            manifest = run_deck_planning(
                project_root=ROOT,
                user_input=user_input,
                output_dir=output_dir,
                client=client,
            )
            plan = json.loads((output_dir / "deck_plan.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["status"], "succeeded")
        self.assertEqual(manifest["model_call_count"], 3)
        self.assertEqual(client.call_count, 3)
        self.assertEqual(manifest["quality_status"], "fallback")
        self.assertTrue(manifest["warnings"])
        self.assertEqual(
            set(plan),
            {
                "topic",
                "audience",
                "deck_goal",
                "style",
                "text_presentation_policy",
                "slides",
            },
        )
        self.assertEqual(plan["text_presentation_policy"], "styled")
        self.assertEqual(len(plan["slides"]), 2)
        self.assertNotIn("purpose", plan["slides"][0])


if __name__ == "__main__":
    unittest.main()
