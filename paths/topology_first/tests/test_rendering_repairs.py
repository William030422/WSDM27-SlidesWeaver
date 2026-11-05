from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from integration.config import resolve_deck_font, validate_deck_font
from integration.decorative_surface import prepare_decorative_surface, reframe_decorative_surface
from integration.text_assembly import (_apply_review_patches, _load_text_placement,
                                       _run_content_unit_fit, _run_critic_rounds, _bbox_intersection,
                                       _configure_visual_protection, _place_unresolved_group)

ROOT = Path(__file__).resolve().parents[1]


class RenderingRepairTests(unittest.TestCase):
    def test_missing_family_resolves_to_real_regular_and_bold(self):
        self.assertEqual(resolve_deck_font('Uninstalled Test Typeface'), 'Noto Sans')
        files = validate_deck_font('Noto Sans')
        self.assertTrue(all(Path(p).is_file() for p in files.values()))

    def test_quiet_surface_preserves_outer_painted_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'holder.png'
            image = Image.new('RGBA', (400, 180))
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((10, 10, 390, 170), radius=65,
                                   fill='#FFF4C4', outline='#DB922D', width=12)
            draw.ellipse((220, 65, 260, 105), fill='#2D8044')
            image.save(source)
            old = np.asarray(image).copy()
            surface = prepare_decorative_surface(source, root / 'frame.png',
                                                 {'x': .06, 'y': .12, 'width': .88, 'height': .76})
            new = np.asarray(Image.open(surface['frame_path']))
            # Actual orange border pixels survive; only the interior is cleared.
            border = (old[:, :, 0] == 219) & (old[:, :, 1] == 146)
            self.assertGreater(np.count_nonzero(new[border, 3]), np.count_nonzero(border) * .4)
            self.assertTrue(np.array_equal(new[old[:, :, 3] == 0], old[old[:, :, 3] == 0]))
            self.assertEqual(new[90, 240, 3], 0)
            self.assertGreater(len(surface['normalized_polygon']), 5)
            self.assertGreater(surface['text_safe_rect']['width'], .6)
            self.assertTrue(np.array_equal(np.asarray(Image.open(source)), old))

    def test_report_uses_actual_size_not_possible_minimum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new('RGB', (1536, 864), 'white').save(root / 'bg.png')
            element = {'region_id': 'caption', 'role': 'caption',
                       'bbox_px': [100, 100, 400, 65],
                       'text': 'This required sentence fits only at a smaller font size.',
                       'font': {'face': 'Noto Sans', 'size_pt': 30, 'minimum_pt': 10, 'weight': 'regular'},
                       'presentation': {'mode': 'direct_text', 'text_color': '111111'}}
            plan = {'slide_id': 'slide_001', 'canvas_px': {'width': 1536, 'height': 864},
                    'background_path': str(root / 'bg.png'), 'groups': [], 'elements': [element]}
            (root / 'plan.json').write_text(json.dumps(plan))
            (root / 'compilation.json').write_text(json.dumps({'slides': [{'slide_id': 'slide_001', 'plan': str(root / 'plan.json')}]}))
            subprocess.run(['node', str(ROOT / 'compile_deck.mjs'), str(root / 'compilation.json'),
                            str(root / 'deck.pptx'), str(root / 'audit.json')], cwd=ROOT,
                           capture_output=True, text=True, check=True)
            audit = json.loads((root / 'audit.json').read_text())['slides'][0]['elements'][0]
            self.assertTrue(audit['fits_at_minimum'])
            self.assertFalse(audit['fits_at_resolved_size'])
            self.assertTrue(audit['overflow'])

    def test_reframing_preserves_four_colored_edges_and_exact_quiet_area(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'ornament.png'
            image = Image.new('RGBA', (200, 120), '#FFF2CC')
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 199, 23), fill='#F00000')
            draw.rectangle((0, 96, 199, 119), fill='#00D000')
            draw.rectangle((0, 24, 39, 95), fill='#0000E0')
            draw.rectangle((160, 24, 199, 95), fill='#D000D0')
            image.save(source)
            original = np.asarray(image).copy()
            target = {'x': .1, 'y': .1, 'width': .8, 'height': .8}
            surface = reframe_decorative_surface(source, root / 'frame.png',
                {'x': .2, 'y': .2, 'width': .6, 'height': .6}, target)
            actual = np.asarray(Image.open(surface['frame_path']))
            self.assertEqual(actual.shape, original.shape)
            self.assertEqual(tuple(actual[3, 100, :3]), (240, 0, 0))
            self.assertEqual(tuple(actual[116, 100, :3]), (0, 208, 0))
            self.assertEqual(tuple(actual[60, 3, :3]), (0, 0, 224))
            self.assertEqual(tuple(actual[60, 196, :3]), (208, 0, 208))
            self.assertTrue(np.all(actual[12:109, 20:181, 3] == 0))
            self.assertEqual(surface['text_safe_rect'], target)
            self.assertTrue(np.array_equal(np.asarray(Image.open(source)), original))

    def plan(self, sid='slide_001'):
        element = {'region_id': 'body', 'role': 'body', 'text': 'Frozen required copy',
                   'bullet_items': [], 'bbox_px': [100, 200, 500, 200],
                   'topology_bbox_px': [100, 200, 500, 200], 'order': 1,
                   'font': {'face': 'Noto Sans', 'size_pt': 18, 'minimum_pt': 18, 'maximum_pt': 24, 'weight': 'regular'},
                   'presentation': {'mode': 'direct_text', 'text_color': '111111'}}
        placement = _load_text_placement()
        placement._refresh_content_unit(element)
        return {'slide_id': sid, 'background_path': '/unused.png', 'canvas_px': {'width': 1536, 'height': 864},
                'groups': [{'group_id': 'singleton', 'member_region_ids': ['body'],
                            'arrangement': 'freeform', 'equal_size': False, 'even_spacing': False}],
                'elements': [element]}

    def test_critic_is_one_call_per_slide_and_one_batch_render(self):
        plans = {sid: self.plan(sid) for sid in ['slide_001', 'slide_002']}
        placement = _load_text_placement()
        client = SimpleNamespace(call_count=0)
        def review(**kwargs):
            client.call_count += 1
            return {'issues': [], 'patches': [{'region_id': 'body', 'op': 'set_font_size', 'size_pt': 19}]}, None
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(placement, 'call_vlm_json', side_effect=review), \
             patch.object(placement, 'configured_vlm_model', return_value='offline'), \
             patch('integration.text_assembly._write_round') as render:
            result = _run_critic_rounds(output_dir=Path(temporary), work_dir=Path(temporary),
                                       slide_ids=list(plans), plans_by_slide=plans,
                                       initial_rendered={sid: Path('/unused.png') for sid in plans},
                                       text_placement=placement, client=client, max_tokens=1800,
                                       deck_plan={'slides': []})
        self.assertEqual(result['api_call_count'], 2)
        self.assertEqual(result['additional_judge_calls'], 0)
        self.assertEqual(render.call_count, 1)
        self.assertTrue(all(p['elements'][0]['font']['size_pt'] == 19 for p in plans.values()))
        self.assertTrue(all(p['elements'][0]['text'] == 'Frozen required copy' for p in plans.values()))

    def test_critic_cannot_change_copy_or_add_objects(self):
        plan = self.plan()
        original = copy.deepcopy(plan)
        candidate, rejected, applied = _apply_review_patches(plan,
            {'patches': [{'region_id': 'body', 'op': 'rewrite', 'text': 'New claim'},
                         {'region_id': 'unknown', 'op': 'set_font_size', 'size_pt': 19}]},
            _load_text_placement())
        self.assertEqual(applied, 0)
        self.assertEqual(candidate, original)
        self.assertEqual(len(rejected), 2)

    def test_raster_title_keeps_reserved_width_and_decor_at_planned_font(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = Image.new('RGBA', (800, 100), '#FFECC4')
            ImageDraw.Draw(image).rectangle((2, 2, 797, 97), outline='#C88722', width=10)
            asset = root / 'title.png'
            image.save(asset)
            plan = self.plan()
            title = plan['elements'][0]
            title.update({'role': 'title', 'text': 'Krusty Krab Choices Are Demonstrations',
                          'bbox_px': [400, 50, 700, 100], 'topology_bbox_px': [100, 50, 1300, 100]})
            title['font'].update({'size_pt': 34, 'minimum_pt': 25})
            title['presentation'] = {'mode': 'raster_holder_text', 'asset_id': 'title',
                'asset_path': str(asset), 'text_safe_rect': {'x': .12, 'y': .2, 'width': .76, 'height': .6}}
            placement = _load_text_placement()
            _run_content_unit_fit(output_dir=root, slide_ids=['slide_001'],
                                  plans_by_slide={'slide_001': plan}, text_placement=placement)
            repaired = plan['elements'][0]
            self.assertEqual(repaired['bbox_px'], [100, 50, 1300, 100])
            self.assertEqual(repaired['font']['size_pt'], 34)
            self.assertEqual(repaired['registration']['status'], 'fit')
            self.assertTrue(Path(repaired['presentation']['quiet_frame_path']).is_file())
            self.assertTrue(np.array_equal(np.asarray(Image.open(asset)), np.asarray(image)))

    def test_scene_protection_respects_reserved_text_component_but_keeps_other_pixels(self):
        plan = self.plan()
        plan['elements'][0]['topology_bbox_px'] = [100, 200, 500, 200]
        _configure_visual_protection(plan,
            {'entities': [{'region_id': 'scene', 'canonical_role': 'visual', 'parent_component_id': 'c_scene'}]},
            {'entities': [{'region_id': 'scene', 'postprocessed_bbox_px': [50, 50, 1000, 700]}]})
        self.assertEqual(sum(_bbox_intersection([50, 50, 1000, 700], p['bbox_px']) for p in plan['protected_visual_regions']), 600000)
        self.assertEqual(sum(_bbox_intersection([100, 200, 500, 200], p['bbox_px']) for p in plan['protected_visual_regions']), 0)
        self.assertGreater(sum(_bbox_intersection([700, 300, 100, 100], p['bbox_px']) for p in plan['protected_visual_regions']), 0)

    def test_group_capacity_repair_moves_peers_together_without_copy_or_order_changes(self):
        placement = _load_text_placement()
        plan = self.plan()
        first = plan['elements'][0]
        first.update({'bbox_px': [100, 200, 180, 45], 'topology_bbox_px': [100, 200, 180, 45],
                      'text': 'The pilot chooses the goal and keeps authority over the mission.',
                      'registration': {'status': 'unresolved'}})
        second = copy.deepcopy(first)
        second.update({'region_id': 'peer', 'bbox_px': [400, 200, 180, 45], 'topology_bbox_px': [400, 200, 180, 45],
                       'text': 'Automation handles fast corrections and reports uncertainty.'})
        plan['elements'].append(second)
        group = {'group_id': 'peers', 'member_region_ids': ['body', 'peer'], 'arrangement': 'row',
                 'equal_size': True, 'even_spacing': False}
        plan['groups'] = [group]
        before = [e['text'] for e in plan['elements']]
        with tempfile.TemporaryDirectory() as temporary:
            repaired = _place_unresolved_group(plan, group, placement, Path(temporary))
        self.assertIsNotNone(repaired)
        self.assertEqual([e['text'] for e in plan['elements']], before)
        self.assertLess(plan['elements'][0]['bbox_px'][0], plan['elements'][1]['bbox_px'][0])
        self.assertEqual(_bbox_intersection(*(e['bbox_px'] for e in plan['elements'])), 0)
        self.assertIsNone(placement._validate_group_invariants(plan))

    def test_prefreeze_capacity_moves_checklist_to_free_space_with_decorated_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = []
            for name, size in [('banner', (800, 80)), ('compact', (400, 300))]:
                image = Image.new('RGBA', size)
                draw = ImageDraw.Draw(image)
                draw.rounded_rectangle((6, 6, size[0] - 6, size[1] - 6), radius=16,
                                       fill='#15142E', outline='#8873AB', width=5)
                path = root / f'{name}.png'
                image.save(path)
                assets.append({'asset_id': name, 'path': str(path), 'category': 'raster_holder',
                               'allowed_roles': ['short_explanation'], 'aspect_ratio': size[0] / size[1],
                               'text_safe_rect': {'x': .1, 'y': .18, 'width': .8, 'height': .64}})
            placement = _load_text_placement()
            plan = self.plan()
            target = plan['elements'][0]
            target['bbox_px'] = [32, 150, 1472, 100]
            target['topology_bbox_px'] = list(target['bbox_px'])
            target['bullet_items'] = ['Align on the goal', 'Show live feedback', 'Expose uncertainty',
                                      'Define human override', 'Measure the outcome']
            target['text'] = '\n'.join(target['bullet_items'])
            target['presentation'] = {'mode': 'raster_holder_text', 'asset_id': 'banner',
                                      'asset_path': str(root / 'banner.png'), 'asset_category': 'raster_holder',
                                      'text_safe_rect': assets[0]['text_safe_rect'], 'text_color': 'FFFFFF'}
            title = copy.deepcopy(target)
            title.update({'region_id': 'title', 'role': 'title', 'text': 'Shared Control', 'bullet_items': [],
                          'bbox_px': [32, 26, 1472, 80], 'topology_bbox_px': [32, 26, 1472, 80],
                          'presentation': {'mode': 'direct_text', 'text_color': 'FFFFFF'}})
            neighbor = copy.deepcopy(title)
            neighbor.update({'region_id': 'neighbor', 'role': 'caption', 'text': 'Pilot and machine',
                             'bbox_px': [760, 200, 700, 120], 'topology_bbox_px': [760, 200, 700, 120]})
            plan['elements'] = [title, target, neighbor]
            plan['groups'] = [{'group_id': e['region_id'], 'member_region_ids': [e['region_id']],
                               'arrangement': 'freeform', 'equal_size': False, 'even_spacing': False}
                              for e in plan['elements']]
            plan['available_assets'] = assets
            plan['protected_visual_regions'] = [{'region_id': 'scene', 'bbox_px': [760, 360, 700, 450]}]
            copy_before = copy.deepcopy(target['bullet_items'])
            _run_content_unit_fit(output_dir=root, slide_ids=['slide_001'], plans_by_slide={'slide_001': plan},
                                  text_placement=placement)
            repaired = plan['elements'][1]
            self.assertEqual(repaired['bullet_items'], copy_before)
            self.assertEqual(repaired['presentation']['mode'], 'raster_holder_text')
            self.assertEqual(repaired['registration']['status'], 'fit')
            self.assertEqual(_bbox_intersection(repaired['bbox_px'], neighbor['bbox_px']), 0)
            self.assertEqual(_bbox_intersection(repaired['bbox_px'], plan['protected_visual_regions'][0]['bbox_px']), 0)
            self.assertEqual(repaired['font']['size_pt'], 18)
            self.assertTrue(Path(repaired['presentation']['quiet_frame_path']).is_file())
