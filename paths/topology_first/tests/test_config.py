from __future__ import annotations

import unittest

from integration.config import parse_16_9_size


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


if __name__ == "__main__":
    unittest.main()
