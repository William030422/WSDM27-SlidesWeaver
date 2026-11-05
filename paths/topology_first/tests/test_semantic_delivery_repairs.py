from __future__ import annotations

import json
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from integration.final_background import (_generate_or_resume, extract_core_style_commands,
    run_background_set, BACKGROUND_VARIANTS)
from integration.region_content import (attach_semantic_context, build_region_inventory,
    parse_region_content, region_content_prompt, ensure_explanatory_overview)
from integration.semantic_topology import normalize_semantic_topology, regularize_semantic_topology
from integration.slide_spec import SlideContentDraft, finalize_slide_spec
from v9_deck_planning.apimart import ApiMartImageTaskFailure


class FakeImages:
    def __init__(self, failures=None, download_failure=False, submit_failure=False):
        self.submits = 0
        self.polls = []
        self.failures = list(failures or [])
        self.download_failure = download_failure
        self.submit_failure = submit_failure

    def submit_image_task(self, payload):
        self.submits += 1
        if self.submit_failure:
            raise TimeoutError("submission outcome unknown")
        return f'task_{self.submits}'

    def poll_task(self, task_id):
        self.polls.append(task_id)
        if self.failures:
            raise self.failures.pop(0)
        return {'status': 'completed', 'result': {'images': [{'url': 'https://image.invalid/result.png'}]}}

    def download(self, url, path):
        if self.download_failure:
            self.download_failure = False
            raise ConnectionError('download interrupted')
        Image.new('RGB', (64, 36), '#ac52e5').save(path, format='PNG')


class SemanticDeliveryRepairs(unittest.TestCase):
    def run_generation(self, root, client, digest='same'):
        return _generate_or_resume(client=client, payload={'prompt': 'benign teaching scene'},
            request_sha256=digest, slide_dir=root, raw_output_path=root / 'background_raw.png')

    def test_poll_timeout_resumes_same_paid_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeImages([TimeoutError('still processing')])
            with self.assertRaises(TimeoutError):
                self.run_generation(root, client)
            self.assertEqual(json.loads((root/'task.json').read_text())['status'], 'poll_pending')
            result = self.run_generation(root, client)
            self.assertEqual(client.submits, 1)
            self.assertEqual(client.polls, ['task_1', 'task_1'])
            self.assertEqual(result['image_calls_during_run'], 0)

    def test_download_interruption_does_not_submit_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeImages(download_failure=True)
            with self.assertRaises(ConnectionError):
                self.run_generation(root, client)
            self.assertEqual(json.loads((root/'task.json').read_text())['status'], 'download_pending')
            self.run_generation(root, client)
            self.assertEqual(client.submits, 1)
            self.run_generation(root, client)
            self.assertEqual(len(client.polls), 2)

    def test_unknown_submission_is_not_blindly_reposted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeImages(submit_failure=True)
            with self.assertRaises(TimeoutError):
                self.run_generation(root, client)
            with self.assertRaisesRegex(RuntimeError, 'outcome is unknown'):
                self.run_generation(root, client)
            self.assertEqual(client.submits, 1)

    def test_provider_failure_budget_and_review_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            errors = [ApiMartImageTaskFailure('task_1', {'code': 'server_error'}),
                      ApiMartImageTaskFailure('task_2', {'code': 'server_error'})]
            client = FakeImages(errors)
            with self.assertRaises(ApiMartImageTaskFailure):
                self.run_generation(root, client)
            with self.assertRaises(ApiMartImageTaskFailure):
                self.run_generation(root, client)
            self.assertEqual(client.submits, 2)
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeImages([ApiMartImageTaskFailure('task_1', {'message': 'Content safety review'})])
            with self.assertRaises(ApiMartImageTaskFailure):
                self.run_generation(Path(tmp), client)
            with self.assertRaises(ApiMartImageTaskFailure):
                self.run_generation(Path(tmp), client)
            self.assertEqual(client.submits, 1)

    def test_changed_prompt_digest_does_not_reuse_old_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeImages()
            self.run_generation(Path(tmp), client)
            self.run_generation(Path(tmp), client, 'changed')
            self.assertEqual(client.submits, 2)

    def test_zero_visual_page_preserves_copy_and_reuses_healthy_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step2 = root / 'step2'
            step2.mkdir()
            Image.new('RGB', (1536, 864), '#101010').save(step2 / 'canonical_background.png')
            (step2 / 'design_system.json').write_text(json.dumps({'style_prompt': 'Purple anime'}))
            for sid in ('slide_001', 'slide_002'):
                sources = {
                    root / 'region_content/slides' / sid / 'region_content_plan.json':
                        {'slide_id': sid, 'text_regions': [{'lines': ['Keep every original teaching word']}], 'visual_regions': []},
                    root / 'topology/calibrated_topology' / sid / 'calibrated_topology.json': {},
                    root / 'topology/postprocessed/layouts' / f'{sid}.json': {},
                }
                for path, data in sources.items():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(data))
            config = SimpleNamespace(api_root='https://offline.invalid', api_key_env='NONE', timeout_seconds=1,
                poll_initial_delay_seconds=0, poll_interval_seconds=0, poll_timeout_seconds=1)
            client = FakeImages()
            bindings = [{'component_id': 'visual', 'position': 'center', 'bbox_px': [300, 200, 400, 300],
                         'visual_brief': 'An existing healthy illustration'}]
            def bind(plan, layout, topology):
                return bindings if plan['slide_id'] == 'slide_002' else []
            args = dict(project_root=root, region_content_dir=root / 'region_content', step2_dir=step2,
                topology_dir=root / 'topology', output_dir=root / 'background', config=config,
                variant=BACKGROUND_VARIANTS[0], mock_images=False, resume_missing=True)
            with patch('integration.final_background.ApiMartClient', return_value=client), \
                 patch('integration.final_background.bind_visual_briefs', side_effect=bind):
                run_background_set(slide_ids=['slide_002'], **args)
                result = run_background_set(slide_ids=['slide_001', 'slide_002'], **args)
            self.assertEqual(client.submits, 1)
            self.assertEqual(result['image_call_count'], 0)
            self.assertEqual(result['quality_status'], 'degraded')
            self.assertEqual(result['slides'][0]['failed_visual']['category'], 'no_visual_regions')
            self.assertIsNone(result['slides'][1]['failed_visual'])
            self.assertFalse((root / 'background/slide_001/task.json').exists())
            self.assertEqual((root / 'background/slide_001/background.png').read_bytes(),
                             (root / 'background/slide_001/background_raw.png').read_bytes())
            self.assertIn('Keep every original teaching word',
                          (root / 'region_content/slides/slide_001/region_content_plan.json').read_text())

    def test_terminal_visual_failure_keeps_other_pages_and_original_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step2 = root/'step2'
            step2.mkdir()
            Image.new('RGB',(1536,864),'#101010').save(step2/'canonical_background.png')
            (step2/'design_system.json').write_text(json.dumps({'style_prompt':'Purple anime'}))
            for sid in ['slide_001','slide_002']:
                paths = {
                    root/'region_content/slides'/sid/'region_content_plan.json':
                        {'text_regions':[{'lines':['All original teaching words']}], 'visual_regions':[]},
                    root/'topology/calibrated_topology'/sid/'calibrated_topology.json': {},
                    root/'topology/postprocessed/layouts'/f'{sid}.json': {},
                }
                for path, data in paths.items():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(data))
            config = SimpleNamespace(api_root='https://offline.invalid',api_key_env='NONE',
                timeout_seconds=1,poll_initial_delay_seconds=0,poll_interval_seconds=0,poll_timeout_seconds=1)
            client = FakeImages([ApiMartImageTaskFailure('task_1', {'message':'Content safety review'})])
            bindings = [{'component_id':'visual','position':'center','bbox_px':[300,200,400,300],
                         'visual_brief':'A benign teaching illustration'}]
            with patch('integration.final_background.ApiMartClient',return_value=client), \
                 patch('integration.final_background.bind_visual_briefs',return_value=bindings):
                result=run_background_set(project_root=root,region_content_dir=root/'region_content',
                    step2_dir=step2,topology_dir=root/'topology',output_dir=root/'background',
                    slide_ids=['slide_001','slide_002'],config=config,variant=BACKGROUND_VARIANTS[0],
                    mock_images=False,resume_missing=True)
            self.assertEqual(result['status'],'succeeded')
            self.assertEqual(result['quality_status'],'degraded')
            self.assertIsNotNone(result['slides'][0]['failed_visual'])
            self.assertIsNone(result['slides'][1]['failed_visual'])
            self.assertEqual(client.submits,2)
            with Image.open(root/'background/slide_001/background.png') as image:
                self.assertEqual(image.getpixel((500,400)),(16,16,16))
            state=json.loads((root/'background/slide_001/task.json').read_text())
            self.assertEqual(state['status'],'failed')
            self.assertIn('All original teaching words',(root/'region_content/slides/slide_001/region_content_plan.json').read_text())

    def test_original_conditions_and_full_style_survive(self):
        brief = 'Define goals, check delay and uncertainty, set override rules, and monitor outcomes.'
        draft = SlideContentDraft(title='Checklist', message='Check uncertainty.', visual_intent='Human and machine')
        spec = finalize_slide_spec('slide_001', draft, brief)
        self.assertEqual(spec.message, brief)
        style = 'EVA anime: purple Unit-01, human pilot Shinji. Deep navy and green.'
        prompt = region_content_prompt(slide_spec={**spec.model_dump(), 'style': style}, inventory=[])
        self.assertIn(brief, prompt)
        self.assertIn(style, prompt)
        self.assertIn('human pilots human', prompt)
        self.assertEqual(extract_core_style_commands(style), style)

    def overview_fixture(self):
        entities = [{'region_id':'title','canonical_role':'title','parent_component_id':'c_title','calibrated_bbox':[3,2,58,4]},
                    {'region_id':'scene','canonical_role':'visual','parent_component_id':'c_scene','calibrated_bbox':[3,8,58,25]}]
        topology = {'entities':entities,'components':[], 'groups':[{'group_id':'g_header','children':['c_title']}],
                    'reading_order':['title','scene']}
        layout = {'canvas':{'width':1536,'height':864},'entities':[
            {'region_id':e['region_id'],'postprocessed_bbox_px':[v*24 for v in e['calibrated_bbox']]}
            for e in entities]}
        return topology, layout

    def test_title_only_explanatory_inventory_keeps_source_message_as_one_overview(self):
        topology, layout = self.overview_fixture()
        brief = 'Cooking and serving are observed actions; goals motivating those actions are hidden.'
        original_visual = list(layout['entities'][1]['postprocessed_bbox_px'])
        result = ensure_explanatory_overview(topology,layout,{'title':'Demonstrations','message':brief})
        self.assertEqual(result['required_text'],brief)
        inventory = build_region_inventory(topology,layout)
        restored = next(e for e in inventory if e['region_id']=='b_semantic_overview')
        self.assertEqual(restored['required_text'],brief)
        current_visual = next(e for e in layout['entities'] if e['region_id']=='scene')['postprocessed_bbox_px']
        self.assertEqual(current_visual[::2],original_visual[::2])  # horizontal position/width stay fixed
        self.assertGreaterEqual(current_visual[1],result['bbox_px'][1]+result['bbox_px'][3])
        self.assertIsNone(ensure_explanatory_overview(topology,layout,{'title':'Demonstrations','message':brief}))
        self.assertEqual(len([e for e in topology['entities'] if e['canonical_role']=='supporting_text']),1)

    def test_intentional_cover_and_title_repetition_do_not_gain_annotations(self):
        for spec in [
            {'title':'Learning Rewards','message':'Learning Rewards'},
            {'title':'Learning Rewards','message':'Overview of learning rewards'},
            {'title':'Learning Rewards','message':'An explanatory introduction to agents, behavior and reward learning.','slide_type':'cover'},
        ]:
            topology,layout=self.overview_fixture()
            before=copy.deepcopy((topology,layout))
            self.assertIsNone(ensure_explanatory_overview(topology,layout,spec))
            self.assertEqual((topology,layout),before)

    def test_visual_only_middle_stage_gets_editable_semantic_label(self):
        identity = {'slide_id': 'slide_001', 'elements': [
            {'element_id': 'title', 'role': 'title', 'bbox': [3,2,58,4]},
            {'element_id': 'a', 'role': 'visual', 'bbox': [3,12,16,18]},
            {'element_id': 'b', 'role': 'visual', 'bbox': [25,12,14,18]},
            {'element_id': 'c', 'role': 'visual', 'bbox': [45,12,16,18]}]}
        blocks = [{'block_id': 'title', 'role': 'title', 'semantic_role': 'Title',
                   'source_region_ids': ['title'], 'bbox': [3,2,58,4]}]
        for key, role in [('a','Observe demonstrations'),('b','Infer a reward'),('c','Learn and test a policy')]:
            source = next(e for e in identity['elements'] if e['element_id'] == key)
            blocks.append({'block_id': f'b_{key}', 'role': 'semantic_stage', 'semantic_role': role,
                           'source_region_ids': [key], 'bbox': source['bbox']})
        semantic = normalize_semantic_topology({'slide_id':'slide_001','blocks':blocks,
            'connections':[{'source_block_id':'b_a','target_block_id':'b_b','relation':'sequence_next'},
                           {'source_block_id':'b_b','target_block_id':'b_c','relation':'sequence_next'}],
            'reading_order':['title','b_a','b_b','b_c']}, slide_id='slide_001',
            source_ids={e['element_id'] for e in identity['elements']},
            source_roles={e['element_id']:e['role'] for e in identity['elements']})
        compiled, report = regularize_semantic_topology(semantic=semantic, identity=identity)
        layout = {'canvas': {'width':1536,'height':864}, 'entities':[
            {'region_id':e['region_id'], 'postprocessed_bbox_px':[v*24 for v in e['calibrated_bbox']]}
            for e in compiled['entities']]}
        inventory = attach_semantic_context(build_region_inventory(compiled, layout), semantic)
        reward = next(e for e in inventory if e['region_id']=='b_b_text_top')
        self.assertEqual(reward['required_text'], 'Infer a reward')
        self.assertEqual(report['overlap_count'], 0)
        payload = {'slide_id':'slide_001', 'visual_regions':[
            {'region_id':e['region_id'],'semantic_brief':'Scene'} for e in inventory if e['role']=='visual'],
            'text_regions':[{'region_id':e['region_id'],'role':e['role'],'lines':['Text']}
                            for e in inventory if e['role']!='visual' and e['region_id']!='b_b_text_top']}
        plan, _ = parse_region_content(json.dumps(payload), 'slide_001', inventory)
        self.assertEqual(next(e.lines for e in plan.text_regions if e.region_id=='b_b_text_top'), ['Infer a reward'])


if __name__ == '__main__':
    unittest.main()
