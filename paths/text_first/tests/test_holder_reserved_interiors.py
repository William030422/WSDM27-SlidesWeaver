from __future__ import annotations

import unittest

from integration.final_background import (build_generation_objects, plan_asset_board_slots,
    compile_generation_prompt, _holder_compatibility)


class HolderReservedInteriors(unittest.TestCase):
    def objects(self):
        return build_generation_objects({'visual_regions': [],
            'text_regions': [{'region_id':'copy', 'text_bbox_px':[140,220,320,120]}],
            'holders':[{'holder_id':'frame', 'text_region_id':'copy','bbox_px':[100,200,400,200]}]})

    def test_generation_reserves_the_actual_frozen_text_box(self):
        objects = self.objects()
        self.assertEqual(objects[0]['required_text_rect'], {'x':.1,'y':.1,'width':.8,'height':.6})
        prompt = compile_generation_prompt(plan_asset_board_slots(objects), 'Purple anime', '#00FF00')
        self.assertIn('OUTER FRAME ONLY', prompt)
        self.assertIn('No characters, portraits', prompt)
        self.assertIn('"width":0.8,"height":0.6', prompt)
        self.assertIn('Preserve a rich themed frame', prompt)

    def test_any_safe_rectangle_is_not_enough_to_cover_planned_copy(self):
        item = self.objects()[0]
        narrow = {'x':.3,'y':.1,'width':.4,'height':.8}
        result = _holder_compatibility(item, narrow)
        self.assertEqual(result['status'], 'requires_quiet_backing')
        self.assertTrue(result['keep_decoration'])
        self.assertEqual(result['planned_text_bbox_px'], [140,220,320,120])
        safe = {'x':.05,'y':.05,'width':.9,'height':.9}
        self.assertEqual(_holder_compatibility(item,safe)['status'], 'compatible')

    def test_unsafe_interior_keeps_the_nonempty_frame_for_native_quiet_backing(self):
        result = _holder_compatibility(self.objects()[0], None)
        self.assertEqual(result['status'], 'requires_quiet_backing')
        self.assertTrue(result['keep_decoration'])


if __name__ == '__main__':
    unittest.main()
