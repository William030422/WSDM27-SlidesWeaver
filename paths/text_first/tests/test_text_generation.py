from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from integration.config import load_config
from integration.text_generation import (
    build_parsed_text_plan,
    parse_text_plan,
    parse_text_block,
    run_text_generation,
    text_generation_prompt,
    text_generation_template,
)
from v9_deck_planning.models import DeckPlan, RoughSlidePlan, UserInput


ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    last_usage = {"total_tokens": 321}

    def __init__(self, response: dict | list[dict]) -> None:
        self.responses = response if isinstance(response, list) else [response]
        self.call_count = 0
        self.prompt = ""

    def chat_json(self, *, model: str, prompt: str, max_tokens: int) -> str:
        self.call_count += 1
        self.prompt = prompt
        response = self.responses[min(self.call_count - 1, len(self.responses) - 1)]
        return json.dumps(response, ensure_ascii=False)


class TextGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.user_input = UserInput(
            content="Explain feedback control in two slides.",
            style="Clean technical illustration",
        )
        self.deck_plan = DeckPlan(
            topic="Feedback control",
            audience="Beginning engineering students",
            deck_goal="Explain the basic feedback loop.",
            style=self.user_input.style,
            slides=[
                RoughSlidePlan(
                    slide_id="slide_001",
                    working_title="Compare target and measurement",
                    rough_message="Explain how their difference produces the control error.",
                ),
                RoughSlidePlan(
                    slide_id="slide_002",
                    working_title="Close the loop",
                    rough_message="Explain how feedback reduces the error over time.",
                ),
            ],
        )

    def test_prompt_contains_user_input_purposes_and_field_only_template(self) -> None:
        prompt = text_generation_prompt(self.user_input, self.deck_plan)
        template = text_generation_template(self.deck_plan)

        self.assertIn(self.user_input.content, prompt)
        self.assertIn(self.deck_plan.slides[0].rough_message, prompt)
        self.assertIn('"slide_id": "slide_001"', prompt)
        self.assertIn('"slide_id": "slide_002"', prompt)
        self.assertIn("slides[].subtitle", template)
        self.assertIn("slides[].text_01", template)
        self.assertIn("slides[].text_05", template)
        self.assertNotIn("null", template)
        self.assertNotIn("final slide title", template)
        self.assertNotIn("one final paragraph", template)
        self.assertIn("text_02 through text_05", prompt)
        self.assertIn("Do not use a text array", prompt)
        self.assertIn("between 1 and 5 blocks", prompt)
        self.assertIn("An itemized block is one text block", prompt)
        self.assertIn("do not use Unicode arrow symbols", prompt)
        self.assertEqual(
            template.splitlines(),
            [
                "slides[]",
                "slides[].slide_id",
                "slides[].title",
                "slides[].subtitle",
                "slides[].text_01",
                "slides[].text_02",
                "slides[].text_03",
                "slides[].text_04",
                "slides[].text_05",
            ],
        )

    def test_one_deck_level_call_preserves_separate_text_blocks(self) -> None:
        response = {
            "slides": [
                {
                    "slide_id": "slide_001",
                    "title": "Find the error",
                    "subtitle": "Target versus measurement",
                    "text_01": "The target states the desired outcome.",
                    "text_02": "The measurement states the current outcome.",
                    "text_03": "Their difference is the control error.",
                },
                {
                    "slide_id": "slide_002",
                    "title": "Close the loop",
                    "subtitle": "",
                    "text_01": "Feedback updates the action from the observed error.",
                },
            ]
        }
        client = FakeClient(response)
        config = load_config(ROOT)
        with tempfile.TemporaryDirectory(prefix="rl4ppt-text-generation-") as temporary:
            run_dir = Path(temporary)
            source_dir = run_dir / "step1"
            source_dir.mkdir()
            self._write_step1_manifest(source_dir)
            (source_dir / "input.json").write_text(
                self.user_input.model_dump_json(), encoding="utf-8"
            )
            (source_dir / "deck_plan.json").write_text(
                self.deck_plan.model_dump_json(), encoding="utf-8"
            )
            manifest = run_text_generation(
                project_root=ROOT,
                source_dir=source_dir,
                output_dir=run_dir / "text_generation",
                config=config,
                mock_llm=False,
                client=client,
            )
            text_plan = json.loads(
                (run_dir / "text_generation/text_plan.json").read_text(encoding="utf-8")
            )
            parsed_text_plan = json.loads(
                (run_dir / "text_generation/parsed_text_plan.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(client.call_count, 1)
        self.assertEqual(manifest["model_call_count"], 1)
        self.assertEqual(manifest["quality_status"], "strict_pass")
        self.assertEqual(
            list(text_plan["slides"][0]),
            ["slide_id", "title", "subtitle", "text_01", "text_02", "text_03"],
        )
        self.assertIsNone(text_plan["slides"][1]["subtitle"])
        self.assertNotIn("message", text_plan["slides"][0])
        self.assertNotIn("visual_intent", text_plan["slides"][0])
        self.assertEqual(manifest["schema"]["maximum_body_fields"], 5)
        self.assertEqual(
            parsed_text_plan["slides"][0]["text_blocks"][0]["format"],
            "paragraph",
        )

    def test_postprocessor_parses_unordered_and_ordered_itemize_blocks(self) -> None:
        unordered = parse_text_block(
            "text_01",
            "The loop contains:\n• Observe the state\n- Choose an action\n* Read the reward",
        )
        ordered = parse_text_block(
            "text_02",
            "Repeat these steps:\n1. Observe\n2) Act\n3、Update\n   using the reward",
        )
        paragraph = parse_text_block(
            "text_03", "One sentence.\nA second line without list markers."
        )

        self.assertEqual(unordered["format"], "unordered_list")
        self.assertEqual(unordered["lead"], "The loop contains:")
        self.assertEqual(
            unordered["items"],
            ["Observe the state", "Choose an action", "Read the reward"],
        )
        self.assertEqual(ordered["format"], "ordered_list")
        self.assertEqual(ordered["start"], 1)
        self.assertEqual(
            ordered["items"], ["Observe", "Act", "Update using the reward"]
        )
        self.assertEqual(paragraph["format"], "paragraph")

    def test_parser_normalizes_unicode_arrows_to_ascii(self) -> None:
        response = {
            "slides": [
                {
                    "slide_id": "slide_001",
                    "title": "Predict -> act",
                    "subtitle": None,
                    "text_01": "Observe → predict ⇒ act",
                },
                {
                    "slide_id": "slide_002",
                    "title": "Compare paths",
                    "subtitle": None,
                    "text_01": "safe ↔ risky",
                },
            ]
        }

        plan = parse_text_plan(json.dumps(response), self.deck_plan)

        self.assertEqual(plan["slides"][0]["text_01"], "Observe -> predict => act")
        self.assertEqual(plan["slides"][1]["text_01"], "safe <-> risky")

    def test_parsed_plan_keeps_one_itemize_as_one_text_block(self) -> None:
        parsed = build_parsed_text_plan(
            {
                "slides": [
                    {
                        "slide_id": "slide_001",
                        "title": "The loop",
                        "subtitle": None,
                        "text_01": "Four parts:\n• state\n• action\n• reward\n• next state",
                    }
                ]
            }
        )

        blocks = parsed["slides"][0]["text_blocks"]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["format"], "unordered_list")
        self.assertEqual(len(blocks[0]["items"]), 4)

    def test_parser_rejects_invalid_dynamic_text_schema(self) -> None:
        valid_slide_2 = {
            "slide_id": "slide_002",
            "title": "Close the loop",
            "subtitle": None,
            "text_01": "Feedback reduces the error.",
        }
        invalid_slides = {
            "missing_text_01": {
                "slide_id": "slide_001",
                "title": "Find the error",
                "subtitle": None,
            },
            "numbering_gap": {
                "slide_id": "slide_001",
                "title": "Find the error",
                "subtitle": None,
                "text_01": "Target",
                "text_03": "Error",
            },
            "unknown_field": {
                "slide_id": "slide_001",
                "title": "Find the error",
                "subtitle": None,
                "text_01": "Target",
                "notes": "Do not accept me",
            },
            "text_array": {
                "slide_id": "slide_001",
                "title": "Find the error",
                "subtitle": None,
                "text": ["Target", "Measurement"],
            },
            "too_many_blocks": {
                "slide_id": "slide_001",
                "title": "Find the error",
                "subtitle": None,
                **{
                    f"text_{index:02d}": f"Block {index}"
                    for index in range(1, 7)
                },
            },
        }
        for label, invalid_slide in invalid_slides.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                parse_text_plan(
                    json.dumps({"slides": [invalid_slide, valid_slide_2]}),
                    self.deck_plan,
                )

    def test_invalid_model_response_recovers_after_three_attempts(self) -> None:
        client = FakeClient(
            {
                "slides": [
                    {
                        "slide_id": "slide_001",
                        "title": "Incomplete",
                        "subtitle": None,
                        "text_01": "Only one returned slide.",
                    }
                ]
            }
        )
        config = load_config(ROOT)
        with tempfile.TemporaryDirectory(prefix="rl4ppt-text-generation-invalid-") as temporary:
            run_dir = Path(temporary)
            source_dir = run_dir / "step1"
            source_dir.mkdir()
            self._write_step1_manifest(source_dir)
            (source_dir / "input.json").write_text(
                self.user_input.model_dump_json(), encoding="utf-8"
            )
            (source_dir / "deck_plan.json").write_text(
                self.deck_plan.model_dump_json(), encoding="utf-8"
            )
            manifest = run_text_generation(
                project_root=ROOT,
                source_dir=source_dir,
                output_dir=run_dir / "text_generation",
                config=config,
                mock_llm=False,
                client=client,
            )

            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(manifest["quality_status"], "degraded")
            self.assertTrue(manifest["warnings"])
            self.assertEqual(client.call_count, 3)
            self.assertEqual(manifest["model_call_count"], 3)
            self.assertTrue((run_dir / "text_generation/raw_response.json").is_file())
            self.assertTrue(
                (run_dir / "text_generation/raw_response_attempt_03.json").is_file()
            )
            recovered = json.loads((run_dir / "text_generation/text_plan.json").read_text())
            self.assertEqual(recovered["slides"][0]["text_01"], "Only one returned slide.")
            self.assertEqual(len(recovered["slides"]), 2)
            self.assertTrue(
                (run_dir / "text_generation/parsed_text_plan.json").exists()
            )

    def test_invalid_first_response_is_repaired_on_second_attempt(self) -> None:
        valid = {
            "slides": [
                {
                    "slide_id": "slide_001",
                    "title": "Find the error",
                    "subtitle": None,
                    "text_01": "Compare the target with the measurement.",
                },
                {
                    "slide_id": "slide_002",
                    "title": "Close the loop",
                    "subtitle": None,
                    "text_01": "Use feedback to reduce the error.",
                },
            ]
        }
        client = FakeClient([{"slides": []}, valid])
        config = load_config(ROOT)
        with tempfile.TemporaryDirectory(prefix="rl4ppt-text-generation-retry-") as temporary:
            run_dir = Path(temporary)
            source_dir = run_dir / "step1"
            source_dir.mkdir()
            self._write_step1_manifest(source_dir)
            (source_dir / "input.json").write_text(
                self.user_input.model_dump_json(), encoding="utf-8"
            )
            (source_dir / "deck_plan.json").write_text(
                self.deck_plan.model_dump_json(), encoding="utf-8"
            )
            manifest = run_text_generation(
                project_root=ROOT,
                source_dir=source_dir,
                output_dir=run_dir / "text_generation",
                config=config,
                mock_llm=False,
                client=client,
            )

            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(manifest["model_call_count"], 2)
            self.assertEqual(client.call_count, 2)
            self.assertIn("previous response was rejected", client.prompt)
            self.assertTrue(
                (run_dir / "text_generation/raw_response_attempt_01.json").is_file()
            )
            self.assertTrue(
                (run_dir / "text_generation/raw_response_attempt_02.json").is_file()
            )

    def _write_step1_manifest(self, source_dir: Path) -> None:
        (source_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "stage": "deck_level_planning",
                    "status": "succeeded",
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
