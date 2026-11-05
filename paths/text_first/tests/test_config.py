from __future__ import annotations

import unittest
from unittest.mock import patch

from integration.config import DECK_FONT, parse_16_9_size, validate_deck_font


class SizeParsingTests(unittest.TestCase):
    def test_accepts_concrete_ascii_size(self) -> None:
        size = parse_16_9_size("1536x864", field_name="SIZE")
        self.assertEqual((size.width, size.height), (1536, 864))

    def test_accepts_unicode_multiplication_sign(self) -> None:
        size = parse_16_9_size("1024\u00d7576", field_name="SIZE")
        self.assertEqual(size.as_text(), "1024x576")

    def test_rejects_ratio_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "concrete WIDTHxHEIGHT"):
            parse_16_9_size("16:9", field_name="SIZE")

    def test_rejects_non_widescreen_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 16:9"):
            parse_16_9_size("1600x1000", field_name="SIZE")

    def test_deck_font_has_exact_regular_and_bold_variants(self) -> None:
        variants = validate_deck_font(DECK_FONT)
        self.assertEqual(set(variants), {"regular", "bold"})

    def test_font_substitution_is_rejected(self) -> None:
        validate_deck_font.cache_clear()
        completed = type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": "Noto Sans\nRegular\n/tmp/substitute.ttf\n",
            },
        )()
        with patch("integration.config.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "substitution is forbidden"):
                validate_deck_font("Aptos")
        validate_deck_font.cache_clear()


if __name__ == "__main__":
    unittest.main()
