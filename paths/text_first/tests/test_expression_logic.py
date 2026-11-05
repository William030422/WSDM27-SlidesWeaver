from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from integration.config import load_config
from integration.expression_logic import (
    build_expression_logic_prompt,
    normalize_expression_logic,
    parse_expression_logic_plan,
    run_expression_logic_planning,
)


ROOT = Path(__file__).resolve().parents[1]


class ExpressionLogicClient:
    last_usage = {"input_tokens": 100, "output_tokens": 30}

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.call_count = 0

    def chat_json(self, **_: object) -> str:
        self.call_count += 1
        return json.dumps(self.response)


class ExpressionLogicTests(unittest.TestCase):
    def test_prompt_is_text_tree_only_and_prioritizes_matching_and_diversity(self) -> None:
        prompt = build_expression_logic_prompt(
            [
                {
                    "slide_id": "slide_001",
                    "title": "Compare choices",
                    "subtitle": None,
                    "text_blocks": [
                        {"text_id": "text_01", "format": "paragraph", "text": "A"},
                        {"text_id": "text_02", "format": "paragraph", "text": "B"},
                    ],
                }
            ]
        )

        self.assertIn("one short, natural-language composition idea", prompt)
        self.assertIn("Never refer to\n  internal identifiers", prompt)
        self.assertIn("creative, and diverse across the deck", prompt)
        self.assertIn("Do not write HTML, CSS, pixel coordinates", prompt)
        self.assertNotIn("style_prompt", prompt)
        self.assertIn("diffusion model", prompt.lower())

    def test_internal_notation_uses_a_non_fatal_natural_language_fallback(self) -> None:
        logic, warning = normalize_expression_logic(
            "Put text01 at x=20px and text_02 on the right.",
            fallback="Pair the ideas with visuals in a clean progression.",
        )
        self.assertEqual(
            logic, "Pair the ideas with visuals in a clean progression."
        )
        self.assertIn("replaced", warning)

    def test_parser_requires_exact_slide_inventory_and_concise_natural_language(self) -> None:
        plan = parse_expression_logic_plan(
            json.dumps(
                {
                    "slides": [
                        {
                            "slide_id": "slide_001",
                            "expression_logic": "  Put four matched ideas in parallel.  ",
                        }
                    ]
                }
            ),
            ["slide_001"],
        )
        self.assertEqual(
            plan.slides[0].expression_logic, "Put four matched ideas in parallel."
        )
        with self.assertRaisesRegex(ValueError, "slide IDs"):
            parse_expression_logic_plan(
                '{"slides":[{"slide_id":"slide_002","expression_logic":"Center it."}]}',
                ["slide_001"],
            )

    def test_runner_uses_one_deck_call_and_writes_per_slide_logic(self) -> None:
        config = load_config(ROOT)
        response = {
            "slides": [
                {
                    "slide_id": "slide_001",
                    "expression_logic": "Use one central process and attach text_01 to its first step.",
                },
                {
                    "slide_id": "slide_002",
                    "expression_logic": "Arrange three matched text and visual pairs in parallel.",
                },
            ]
        }
        client = ExpressionLogicClient(response)
        with tempfile.TemporaryDirectory(prefix="rl4ppt-expression-logic-") as temporary:
            root = Path(temporary)
            text_generation = root / "text_generation"
            text_generation.mkdir()
            (text_generation / "run_manifest.json").write_text(
                json.dumps(
                    {"stage": "deck_level_text_generation", "status": "succeeded"}
                ),
                encoding="utf-8",
            )
            (text_generation / "parsed_text_plan.json").write_text(
                json.dumps(
                    {
                        "slides": [
                            {
                                "slide_id": "slide_001",
                                "title": "A",
                                "subtitle": None,
                                "text_blocks": [
                                    {
                                        "text_id": "text_01",
                                        "format": "paragraph",
                                        "text": "First",
                                    }
                                ],
                            },
                            {
                                "slide_id": "slide_002",
                                "title": "B",
                                "subtitle": None,
                                "text_blocks": [
                                    {
                                        "text_id": "text_01",
                                        "format": "paragraph",
                                        "text": "Second",
                                    }
                                ],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            manifest = run_expression_logic_planning(
                project_root=ROOT,
                text_generation_dir=text_generation,
                output_dir=root / "expression_logic",
                slide_ids=["slide_001", "slide_002"],
                config=config,
                mock_llm=False,
                client=client,
            )
            first = json.loads(
                (
                    root
                    / "expression_logic/slides/slide_001/expression_logic.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["status"], "succeeded")
        self.assertEqual(manifest["model_call_count"], 1)
        self.assertEqual(client.call_count, 1)
        self.assertNotIn("text_01", first["expression_logic"])
        self.assertEqual(manifest["quality_status"], "fallback")
        self.assertTrue(manifest["warnings"])


if __name__ == "__main__":
    unittest.main()
