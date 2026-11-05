from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from playwright.sync_api import sync_playwright

from integration.html_compile import _compile_connectors, _compile_slide
from integration.html_planning import apply_initial_readability_contract, allocate_holder_text_space, allocate_readable_text_space, build_html_planning_prompt
from integration.html_text_rework import apply_deck_typography_profile


class ReadabilityContractTests(unittest.TestCase):
    def test_unsafe_subtitle_does_not_veto_shared_bold_titles(self):
        # Actual EVA failure: an auto-height subtitle grew ~1.3px and rejected
        # safe titles on every page. Tiny subtitle capacity must stay unresolved.
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for i in (1, 2):
                path = Path(tmp) / f'{i}.html'
                path.write_text(f'''<!doctype html><html><head><style>
html,body{{margin:0}}.slide{{position:relative;width:1536px;height:864px;font-family:"Noto Sans"}}
.title{{position:absolute;left:60px;top:40px;width:900px;height:75px;font-size:{45+i}px;font-weight:{400 if i==1 else 700};margin:0}}
.subtitle{{position:absolute;left:60px;top:130px;width:900px;line-height:1.3;font-size:18px;margin:0}}
</style></head><body><section class="slide"><h1 class="title" data-region-id="t" data-content-key="title" data-role="title">Same clear title</h1>
<p class="subtitle" data-region-id="s" data-content-key="subtitle" data-role="subtitle">A fixed supporting sentence</p></section></body></html>''')
                paths[str(i)] = path
            report = apply_deck_typography_profile(html_by_slide=paths,
                                                   profile={'title_font_size_px':43,'subtitle_font_size_px':24})
            self.assertEqual(report['status'], 'partial')
            self.assertEqual(report['role_results']['title']['status'], 'accepted')
            self.assertEqual(report['role_results']['subtitle']['status'], 'rejected')
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                for path in paths.values():
                    page.set_content(path.read_text())
                    self.assertEqual(page.locator('h1').evaluate('n=>getComputedStyle(n).fontSize'), '43px')
                    self.assertEqual(page.locator('h1').evaluate('n=>getComputedStyle(n).fontWeight'), '700')
                    self.assertEqual(page.locator('p').inner_text(), 'A fixed supporting sentence')
                browser.close()

    def test_precommit_floor_preserves_words_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'layout.html'
            path.write_text('''<!doctype html><html><head><style>html,body{margin:0}.slide{width:1536px;height:864px}.copy{font:14px/1.3 "Noto Sans";width:700px;height:300px}</style></head><body><section class="slide"><div class="copy" data-content-key="text_01" data-role="body">Measure delay, uncertainty, and the result.</div></section></body></html>''')
            changes = apply_initial_readability_contract(path)
            self.assertEqual(changes[0]['to_font_size_px'], 20)
            self.assertIn('Measure delay, uncertainty, and the result.', path.read_text())
            self.assertEqual(apply_initial_readability_contract(path), [])

    def test_connectors_require_explicit_mapping_and_avoid_text(self):
        a = {'region_id':'a','role':'visual','bbox_px':[100,200,300,200]}
        b = {'region_id':'b','role':'visual','bbox_px':[700,200,300,200]}
        self.assertEqual(_compile_connectors([a,b]), ([],[]))
        a = {**a, 'connect_to':'b', 'connector_color':'#aabbcc'}
        connectors, warnings = _compile_connectors([a,b])
        self.assertFalse(warnings)
        self.assertEqual(connectors[0]['from_px'], [400,300])
        self.assertEqual(connectors[0]['to_px'], [700,300])
        self.assertEqual(connectors[0]['color'], 'AABBCC')
        text = {'region_id':'explanation','role':'body','content_key':'text_01',
                'bbox_px':[460,260,180,80],'text_bbox_px':[460,260,180,80]}
        connectors, warnings = _compile_connectors([a,b,text])
        self.assertEqual(connectors, [])
        self.assertIn('clearance', warnings[0])
        connectors, warnings = _compile_connectors([{**a,'connect_to':'not_present'},b])
        self.assertEqual(connectors, [])
        self.assertIn('unmapped', warnings[0])

    def test_holder_allocation_removes_stale_caps_without_moving_art(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'layout.html'
            path.write_text('''<!doctype html><html><head><style>html,body{margin:0}.holder{position:absolute;left:60px;top:200px;width:600px;height:70px;box-sizing:border-box;border:2px solid black}.copy{position:absolute;left:30px;top:20px;max-width:200px!important;height:20px!important;font:20px/1.3 "Noto Sans"}.art{position:absolute;left:800px;top:200px;width:400px;height:300px}</style></head><body><section class="slide"><div class="holder" data-module-id="h" data-holder-id="h"><div class="copy" data-content-key="text_01" data-role="body">Keep every required process stage.</div></div><div class="art" data-role="visual"></div></section></body></html>''')
            self.assertTrue(allocate_holder_text_space(path))
            self.assertEqual(allocate_holder_text_space(path), [])
            with sync_playwright() as p:
                browser=p.chromium.launch(headless=True)
                page=browser.new_page();page.set_content(path.read_text())
                box=page.locator('.copy').bounding_box()
                self.assertGreater(box['width'],500)
                self.assertGreater(box['height'],30)
                self.assertEqual(page.locator('.art').bounding_box(),{'x':800,'y':200,'width':400,'height':300})
                self.assertEqual(page.locator('.copy').inner_text(),'Keep every required process stage.')
                browser.close()

    def test_html_prompt_receives_style_and_original_purpose(self):
        context = {'deck_style':'EVA purple/green anime', 'slide_purpose':'Human sets goals; machine executes'}
        prompt = build_html_planning_prompt(slide_id='slide_001', expression_logic='Compare human and machine.',
                                            text_tree={}, frozen_content_by_key={'title':'Shared control'}, deck_context=context)
        self.assertIn(json.dumps(context, separators=(',', ':')), prompt)
        self.assertIn('data-connect-to=', prompt)
        self.assertIn('Do not replace', prompt)

    def test_role_height_and_plain_body_use_free_module_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'layout.html'
            path.write_text('''<!doctype html><html><head><style>html,body{margin:0}.module{position:absolute;width:900px}.header{left:60px;top:30px;height:120px}.title{position:absolute;left:0;top:0;width:800px;height:50px;font:43px/1.1 "Noto Sans";margin:0}.subtitle{position:absolute;left:0;top:65px;width:800px;height:30px;font:24px/1.2 "Noto Sans";margin:0}.content{left:60px;top:200px;height:150px}.copy{position:absolute;left:0;top:0;width:300px;height:30px;font:20px/1.3 "Noto Sans";margin:0}</style></head><body><section class="slide"><div class="module header" data-module-id="header"><h1 class="title" data-region-id="t" data-content-key="title" data-role="title">Shared typography</h1><p class="subtitle" data-region-id="s" data-content-key="subtitle" data-role="subtitle">A readable subtitle</p></div><div class="module content" data-module-id="body"><p class="copy" data-region-id="b" data-content-key="text_01" data-role="body">Preserve the original words.</p><div data-region-id="v" data-role="visual" style="position:absolute;left:400px;top:0;width:400px;height:150px"></div></div></section></body></html>''')
            self.assertTrue(allocate_readable_text_space(path))
            self.assertEqual(allocate_readable_text_space(path), [])
            with sync_playwright() as p:
                browser=p.chromium.launch(headless=True);page=browser.new_page();page.set_content(path.read_text())
                title=page.locator('.title').bounding_box();subtitle=page.locator('.subtitle').bounding_box()
                self.assertGreaterEqual(title['height'],64.5)
                self.assertGreaterEqual(subtitle['height'],36)
                self.assertGreaterEqual(subtitle['y'],title['y']+title['height']+2)
                self.assertEqual(page.locator('.copy').bounding_box()['height'],150)
                self.assertEqual(page.locator('[data-role="visual"]').bounding_box(),{'x':460,'y':200,'width':400,'height':150})
                browser.close()

    def test_browser_connector_metadata_reaches_existing_layout_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'layout.html'
            path.write_text('''<!doctype html><html><head><style>html,body{margin:0}.slide{position:relative;width:1536px;height:864px}.module{position:absolute}</style></head><body><section class="slide" data-slide-id="slide_001">
<div class="module" data-module-id="title_m" style="left:60px;top:30px;width:900px;height:80px"><h1 data-region-id="title" data-content-key="title" data-role="title" style="font:40px/1.1 'Noto Sans';margin:0">A declared progression</h1></div>
<div class="module" data-module-id="a_m" style="left:80px;top:240px;width:300px;height:260px"><div data-region-id="a" data-role="visual" data-visual-brief="Observe" data-generation-group="group_a" data-connect-to="b" data-connector-color="#abcdef" style="width:300px;height:260px"></div></div>
<div class="module" data-module-id="b_m" style="left:800px;top:240px;width:300px;height:260px"><div data-region-id="b" data-role="visual" data-visual-brief="Correct" data-generation-group="group_b" style="width:300px;height:260px"></div></div>
</section></body></html>''')
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                result = _compile_slide(page=browser.new_page(viewport={'width':1536,'height':864}),
                                        slide_id='slide_001', html_path=path,
                                        source_text={'slide_id':'slide_001','title':'A declared progression','subtitle':None},
                                        visual_groups_path=root/'missing.json', output_dir=root/'out',
                                        semantic_context={'deck_style':'EVA'})
                browser.close()
            layout = json.loads(Path(result['html_layout']).read_text())
            self.assertEqual(layout['connectors'][0]['from_px'], [380,370])
            self.assertEqual(layout['connectors'][0]['to_px'], [800,370])
            self.assertEqual(layout['connectors'][0]['color'], 'ABCDEF')
            self.assertEqual(layout['semantic_context']['deck_style'], 'EVA')

    def test_header_reservation_uses_free_gap_without_moving_next_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'layout.html'
            path.write_text('''<html><head><style>html,body{margin:0}.module{position:absolute;left:64px;width:1408px}.header{top:42px;height:112px}.title{position:absolute;top:0;width:900px;height:78px;margin:0;font:700 52px/1.12 "Noto Sans"}.subtitle{position:absolute;top:77px;width:760px;height:35px;margin:0;font:28px/1.2 "Noto Sans"}.art{top:178px;height:270px}</style></head><body><div class="module header" data-module-id="h"><h1 class="title" data-role="title" data-content-key="title" data-region-id="t">Shared title</h1><p class="subtitle" data-role="subtitle" data-content-key="subtitle" data-region-id="s">A separate supporting line</p></div><div class="module art" data-module-id="art"><div data-region-id="v" data-role="visual" style="width:100%;height:100%"></div></div></body></html>''')
            self.assertTrue(allocate_readable_text_space(path))
            with sync_playwright() as p:
                browser=p.chromium.launch(headless=True);page=browser.new_page();page.set_content(path.read_text())
                title=page.locator('.title').bounding_box();sub=page.locator('.subtitle').bounding_box();head=page.locator('.header').bounding_box()
                self.assertGreaterEqual(sub['y'],title['y']+title['height']+2)
                self.assertGreaterEqual(sub['height'],42)
                self.assertLessEqual(sub['y']+sub['height'],head['y']+head['height'])
                self.assertEqual(page.locator('.art').bounding_box()['y'],178)
                browser.close()
            self.assertEqual(allocate_readable_text_space(path),[])


if __name__ == '__main__':
    unittest.main()
