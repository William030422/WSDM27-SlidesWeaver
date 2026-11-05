import unittest
from integration.render_quality import inspect_words
from integration.text_assembly import _resolve_element_text_geometry


class DecorativeRenderQualityTests(unittest.TestCase):
    def test_insufficient_holder_keeps_art_and_committed_text_geometry(self):
        element = {'region_id': 'copy', 'bbox_px': [40, 40, 504, 206],
                   'font': {'color': 'FFFFFF'},
                   'holder_binding': {'holder_bbox_px': [20, 20, 560, 260],
                                      'text_safe_rect': {'x': .3, 'y': .1, 'width': .5, 'height': .8}}}
        _resolve_element_text_geometry(element, [], 'slide_001')
        self.assertIsNotNone(element['holder_binding'])
        self.assertEqual(element['resolved_text_bbox_px'], [40, 40, 504, 206])
        self.assertEqual(element['text_backing']['fill_color'], '14182B')
        self.assertNotIn('native_holder', element)

    def test_pdf_word_overhang_is_distinct_from_an_extra_line(self):
        plan = {'slide_id': 'slide_001', 'elements': [
            {'region_id': 'copy', 'bbox_px': [10, 10, 100, 30], 'text': 'Hello world', 'role': 'body'}]}
        good = [{'text': 'Hello', 'bbox_px': [10, 12, 45, 43]},
                {'text': 'world', 'bbox_px': [50, 12, 90, 43]}]
        self.assertEqual(inspect_words(good, plan)['status'], 'strict_pass')
        bad = good[:1] + [{'text': 'world', 'bbox_px': [10, 44, 60, 70]}]
        self.assertIn('rendered_text_outside_region', inspect_words(bad, plan)['regions'][0]['issues'])

    def test_broken_word_and_title_wrap_are_reported(self):
        plan = {'slide_id': 'slide_001', 'elements': [
            {'region_id': 'title', 'bbox_px': [0, 0, 200, 100], 'text': 'Observe demonstrations', 'role': 'title'}]}
        words = [{'text': 'Observe', 'bbox_px': [0, 0, 80, 20]},
                 {'text': 'demonstration', 'bbox_px': [0, 25, 150, 45]},
                 {'text': 's', 'bbox_px': [0, 50, 10, 70]}]
        issues = inspect_words(words, plan)['regions'][0]['issues']
        self.assertIn('word_broken_across_lines', issues)
        self.assertIn('unexpected_title_wrap', issues)

    def test_pdf_columns_do_not_hide_a_broken_word(self):
        plan = {'slide_id': 'slide_001', 'elements': [{'region_id': 'caption',
                'bbox_px': [0, 0, 200, 100], 'text': 'demonstrations', 'role': 'body'}]}
        words = [{'text': 'demonstration', 'bbox_px': [0, 0, 150, 20]},
                 {'text': 'Other-column', 'bbox_px': [400, 0, 550, 20]},
                 {'text': 's', 'bbox_px': [50, 25, 60, 45]}]
        self.assertIn('word_broken_across_lines', inspect_words(words, plan)['regions'][0]['issues'])

    def test_word_order_and_signed_values_do_not_pass_as_equivalent(self):
        plan = {'slide_id': 'slide_001', 'elements': [{'region_id': 'caption',
                'bbox_px': [0, 0, 300, 60], 'text': 'reward causes action', 'role': 'body'}]}
        words = [{'text': text, 'bbox_px': [i * 80, 0, i * 80 + 75, 20]}
                 for i, text in enumerate(['action', 'causes', 'reward'])]
        self.assertIn('rendered_word_order_changed', inspect_words(words, plan)['regions'][0]['issues'])
        plan['elements'][0]['text'] = 'reward <- behavior at -3'
        words = [{'text': text, 'bbox_px': [i * 50, 0, i * 50 + 45, 20]}
                 for i, text in enumerate(['reward', '->', 'behavior', 'at', '3'])]
        self.assertIn('rendered_words_unverified', inspect_words(words, plan)['regions'][0]['issues'])

    def test_positioned_columns_have_independent_reading_order_and_bounds(self):
        plan = {'slide_id': 'slide_001', 'elements': [{'region_id': 'choices',
                'bbox_px': [0, 0, 300, 70], 'role': 'body', 'rich_text': [
                    {'bbox_px': [0, 0, 90, 70], 'runs': [{'text': 'Collect shells'}]},
                    {'bbox_px': [110, 0, 90, 70], 'runs': [{'text': 'Help friends'}]}]}]}
        words = [{'text': text, 'bbox_px': box} for text, box in [
            ('Collect', [0, 0, 85, 20]), ('Help', [110, 0, 155, 20]),
            ('shells', [0, 25, 65, 45]), ('friends', [110, 25, 185, 45])]]
        report = inspect_words(words, plan)
        self.assertEqual(report['status'], 'strict_pass')
        self.assertEqual(len(report['regions']), 2)
        words[-1]['bbox_px'] = [110, 75, 185, 95]
        self.assertIn('rendered_text_outside_region', inspect_words(words, plan)['regions'][1]['issues'])

    def test_source_hyphen_wrap_does_not_hide_an_arbitrary_word_break(self):
        plan = {'slide_id': 'slide_001', 'elements': [{'region_id': 'caption',
                'bbox_px': [0, 0, 200, 100], 'text': 'EVA-like', 'role': 'body'}]}
        words = [{'text': 'EVA-', 'bbox_px': [0, 0, 70, 20]},
                 {'text': 'like', 'bbox_px': [0, 25, 60, 45]}]
        self.assertEqual(inspect_words(words, plan)['status'], 'strict_pass')
        words[0]['text'], words[1]['text'] = 'EV', 'A-like'
        self.assertIn('word_broken_across_lines', inspect_words(words, plan)['regions'][0]['issues'])


if __name__ == '__main__':
    unittest.main()
