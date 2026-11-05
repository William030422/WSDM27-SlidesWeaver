from __future__ import annotations

import json
import tempfile
import unittest
import subprocess
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

from PIL import Image
from playwright.sync_api import sync_playwright

from integration.html_text_rework import _validate_reworked_copy
from integration.text_generation import recover_text_plan
from integration.text_assembly import _compile_deck, _font_from_html, _render_and_validate, _resolve_element_text_geometry, _validate_final_geometry
from v9_deck_planning.json_utils import call_json_with_retries


class QualityAssemblyTests(unittest.TestCase):
    def test_committed_caption_fit_uses_real_dom_line_count(self):
        from integration.html_compile import _BROWSER_EXTRACTION_SCRIPT, _normalize_typography
        from integration.render_quality import inspect_pdf_text

        text='The cockpit repeatedly compares intention with reality.'
        with tempfile.TemporaryDirectory(prefix='rl4ppt-caption-lines-') as directory:
            root=Path(directory);Image.new('RGB',(1536,864),'white').save(root/'background.png')
            with sync_playwright() as p:
                browser=p.chromium.launch(headless=True);page=browser.new_page(viewport={'width':1536,'height':864})
                page.set_content(f'<html><body style="margin:0"><section class="slide" style="position:relative;width:1536px;height:864px"><p data-content-key="subtitle" data-region-id="caption" data-role="subtitle" style="position:absolute;left:67px;top:140px;width:780px;height:45px;margin:0;font:30px/1.2 Noto Sans">{text}</p></section></body></html>')
                page.evaluate('document.fonts.ready')
                raw=page.eval_on_selector('.slide',_BROWSER_EXTRACTION_SCRIPT)['regions'][0]
                browser.close()
            self.assertAlmostEqual(raw['text_ink_bbox_px'][2],770.56,delta=1)
            typography=_normalize_typography(raw)
            self.assertEqual(typography['line_count'],1)
            element={'region_id':'caption','role':'subtitle','bbox_px':[67,140,780,45],'text':text,
                     'font':_font_from_html({'role':'subtitle','typography':typography},SimpleNamespace(deck_font='Noto Sans'))}
            plan={'slide_id':'slide_001','canvas_px':{'width':1536,'height':864},'background_path':str(root/'background.png'),'raster_layers':[],'elements':[element]}
            (root/'plan.json').write_text(json.dumps(plan));(root/'compilation.json').write_text(json.dumps({'slides':[{'slide_id':'slide_001','plan':str(root/'plan.json')}]}))
            _compile_deck(root/'compilation.json',root/'deck.pptx',root/'fit.json')
            fit=json.loads((root/'fit.json').read_text())['slides'][0]
            self.assertEqual(fit['overflow_region_ids'],[])
            self.assertEqual(fit['elements'][0]['line_count_at_resolved'],1)
            _render_and_validate(root/'deck.pptx',root/'deck.pdf',root/'rendered',1)
            audit=inspect_pdf_text(root/'deck.pdf',{'slide_001':plan})
            self.assertEqual(audit['status'],'strict_pass',audit['warnings'])

    def test_grid_list_keeps_three_native_columns_and_exact_css_leading(self):
        from integration.html_compile import _compile_slide
        from integration.text_assembly import _normalize_rich_text
        from integration.render_quality import inspect_pdf_text

        with tempfile.TemporaryDirectory(prefix="rl4ppt-grid-cells-") as directory:
            root = Path(directory)
            Image.new("RGB", (1536, 864), "white").save(root / "background.png")
            html = root / 'layout.html'
            html.write_text('''<!doctype html><html><head><style>html,body{margin:0}.slide{position:relative;width:1536px;height:864px;font-family:"Noto Sans"}.copy{position:absolute;left:90px;top:159px;width:1356px;height:92px;font:25px/1.24 "Noto Sans"}.lead{display:block;margin-bottom:9px}ul{display:grid;grid-template-columns:repeat(3,1fr);column-gap:34px;margin:0;padding-left:29px}li{margin:0;padding-right:8px}</style></head><body><section class="slide"><h1 data-region-id="title" data-content-key="title" data-role="title" style="position:absolute;left:64px;top:40px;width:900px;height:75px;margin:0;font-size:40px">Three candidate rewards</h1><div class="copy" data-region-id="candidates" data-content-key="text_01" data-role="body"><span class="lead">Consider three candidate rewards:</span><ul><li>Earn profit for the Krusty Krab</li><li>Increase customer happiness</li><li>Help friends</li></ul></div></section></body></html>''')
            with sync_playwright() as p:
                browser=p.chromium.launch(headless=True)
                compiled=_compile_slide(page=browser.new_page(viewport={'width':1536,'height':864}), slide_id='slide_001', html_path=html,
                    source_text={'slide_id':'slide_001','title':'Three candidate rewards','subtitle':None,
                                 'text_01':'Consider three candidate rewards:\nEarn profit for the Krusty Krab\nIncrease customer happiness\nHelp friends'},
                    visual_groups_path=root/'none.json',output_dir=root/'compiled')
                browser.close()
            layout=json.loads(Path(compiled['html_layout']).read_text())
            region=next(r for r in layout['text_regions'] if r['content_key']=='text_01')
            paragraphs=_normalize_rich_text(region['display_content']['paragraphs'])
            self.assertEqual(len(paragraphs),4)
            self.assertEqual(len({p['bbox_px'][1] for p in paragraphs[1:]}),1)
            plan={'slide_id':'slide_001','canvas_px':{'width':1536,'height':864},'background_path':str(root/'background.png'),
                  'raster_layers':[],'elements':[{'region_id':'candidates','role':'body','bbox_px':region['text_bbox_px'],
                  'text':region['display_content']['text'],'rich_text':paragraphs,'font':_font_from_html(region,SimpleNamespace(deck_font='Noto Sans'))}]}
            (root/'plan.json').write_text(json.dumps(plan))
            (root/'compilation.json').write_text(json.dumps({'slides':[{'slide_id':'slide_001','plan':str(root/'plan.json')}]}))
            _compile_deck(root/'compilation.json',root/'deck.pptx',root/'fit.json')
            _render_and_validate(root/'deck.pptx',root/'deck.pdf',root/'rendered',1)
            report=inspect_pdf_text(root/'deck.pdf',{'slide_001':plan})
            self.assertEqual(report['status'],'strict_pass',report['warnings'])
            cells=report['slides'][0]['regions']
            self.assertEqual(len(cells),4)
            self.assertLess(max(c['ink_bbox_px'][1] for c in cells[1:])-min(c['ink_bbox_px'][1] for c in cells[1:]),2)
            import zipfile
            with zipfile.ZipFile(root/'deck.pptx') as deck:
                xml=deck.read('ppt/slides/slide1.xml').decode()
            self.assertIn('<a:spcPts',xml)
            self.assertNotIn('<a:spcPct',xml)

    def test_real_pptx_pdf_font_size_matches_browser_geometry(self):
        from integration.html_compile import _normalize_typography

        with tempfile.TemporaryDirectory(prefix="rl4ppt-font-contract-") as directory:
            root = Path(directory)
            Image.new("RGB", (1536, 864), "white").save(root / "background.png")
            font_size = _normalize_typography({"font_size_px": 32})["font_size_pt"]
            plan = {"slide_id": "probe", "canvas_px": {"width": 1536, "height": 864}, "background_path": str(root / "background.png"), "raster_layers": [], "elements": [{"region_id": "body", "role": "body", "bbox_px": [100, 100, 800, 200], "text": "MMMM", "font": {"face": "Noto Sans", "size_pt": font_size, "minimum_pt": font_size, "weight": "regular", "line_height_ratio": 1.08}}]}
            (root / "plan.json").write_text(json.dumps(plan))
            (root / "compilation.json").write_text(json.dumps({"slides": [{"slide_id": "probe", "plan": str(root / "plan.json") }]}))
            _compile_deck(root / "compilation.json", root / "deck.pptx", root / "fit.json")
            report = _render_and_validate(root / "deck.pptx", root / "deck.pdf", root / "rendered", 1)
            self.assertEqual(report["status"], "strict_pass")
            subprocess.run(["pdftotext", "-bbox", str(root / "deck.pdf"), str(root / "bbox.html")], check=True)
            document = ElementTree.parse(root / "bbox.html")
            namespaces = {"h": "http://www.w3.org/1999/xhtml"}
            page, word = document.find(".//h:page", namespaces), document.find(".//h:word", namespaces)
            self.assertIsNotNone(page)
            self.assertIsNotNone(word)
            pdf_fraction = (float(word.attrib["xMax"]) - float(word.attrib["xMin"])) / float(page.attrib["width"])
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                try:
                    tab = browser.new_page(viewport={"width": 1536, "height": 864})
                    tab.set_content('<div id="text" style="font:32px/1.08 Noto Sans">MMMM</div>')
                    tab.evaluate("document.fonts.ready")
                    html_width = tab.evaluate("() => {const range=document.createRange();range.selectNodeContents(document.querySelector('#text'));return range.getBoundingClientRect().width;}")
                finally:
                    browser.close()
            self.assertAlmostEqual(pdf_fraction / (html_width / 1536), 1.0, delta=.03)

    def test_holder_safety_repairs_interior_without_shrinking_committed_box(self):
        element = {"region_id": "body", "bbox_px": [132, 132, 336, 136], "holder_binding": {"holder_bbox_px": [100, 100, 400, 200], "text_safe_rect": {"x": .1, "y": .1, "width": .8, "height": .8}}}
        _resolve_element_text_geometry(element, [], "slide_001")
        self.assertEqual(element["resolved_text_bbox_px"], [132, 132, 336, 136])
        self.assertTrue(element['text_backing']['overlay'])
        self.assertEqual(element["holder_binding"]["holder_bbox_px"], [100, 100, 400, 200])

    def test_small_holder_margin_keeps_a_nonzero_decorative_rim(self):
        from unittest.mock import patch
        from integration.decorative_surface import reframe_decorative_surface

        with tempfile.TemporaryDirectory(prefix='rl4ppt-small-rim-') as directory:
            root=Path(directory)
            asset=root/'holder.png';Image.new('RGB',(400,200),'purple').save(asset)
            element={'region_id':'body','bbox_px':[120,120,360,172],
                     'holder_binding':{'holder_bbox_px':[100,100,400,200],
                     'text_safe_rect':{'x':.15,'y':.2,'width':.7,'height':.6},
                     'asset_path':str(asset),'repair_interior':True}}
            warnings=[]
            with patch('integration.text_assembly.prepare_decorative_surface',return_value={
                    'text_safe_rect':{'x':.15,'y':.2,'width':.7,'height':.6}}), patch(
                    'integration.text_assembly.reframe_decorative_surface',wraps=reframe_decorative_surface) as reframed:
                _resolve_element_text_geometry(element,warnings,'slide_001',output_dir=root)
            self.assertFalse(warnings)
            quiet=reframed.call_args.args[3]
            self.assertLess(quiet['y']+quiet['height'],1)
            self.assertGreaterEqual((quiet['y']+quiet['height'])*200,192)
            self.assertEqual(element['resolved_text_bbox_px'],[120,120,360,172])
            self.assertFalse(element['text_backing'].get('overlay',False))
            self.assertTrue(Path(element['holder_binding']['quiet_frame_path']).is_file())

    def test_requested_small_html_font_is_not_increased(self):
        font = _font_from_html({"role": "body", "typography": {"font_size_pt": 10, "font_family": "Noto Sans"}}, SimpleNamespace(deck_font="Noto Sans"))
        self.assertEqual(font["size_pt"], 10)

    def test_copy_formatting_keeps_units_signs_and_negation(self):
        for source, output in [("10% to 20%", "10 to 20"), ("-10 points", "+10 points"), ("It does not improve survival.", "It does improve survival.")]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                _validate_reworked_copy(content_key="text_01", source_text=source, parts=[{"kind": "body", "runs": [{"text": output, "bold": False}]}])
        _validate_reworked_copy(content_key="text_01", source_text="The rate is -10% today.", parts=[{"kind": "body", "runs": [{"text": "The rate is", "bold": False}, {"text": "-10%", "bold": True}, {"text": "today.", "bold": False}]}])

    def test_validation_fallback_does_not_catch_transport_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "response.json"
            parsed, _ = call_json_with_retries(prompt="fixture", call=lambda _: "{}", parse=lambda _: (_ for _ in ()).throw(ValueError("quality")), response_path=path, on_validation_failure=lambda _: {"recovered": True})
            self.assertEqual(parsed, {"recovered": True})
            with self.assertRaisesRegex(RuntimeError, "API unavailable"):
                call_json_with_retries(prompt="fixture", call=lambda _: (_ for _ in ()).throw(RuntimeError("API unavailable")), parse=json.loads, response_path=path, on_validation_failure=lambda _: {"recovered": True})

    def test_geometry_findings_are_reported_without_interrupting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps({"slide_id": "slide_001", "canvas_px": {"width": 1536, "height": 864}, "elements": [{"region_id": "a", "bbox_px": [10, 10, 200, 100]}, {"region_id": "b", "bbox_px": [100, 50, 200, 100]}], "raster_layers": []}))
            report = _validate_final_geometry([{"plan": str(path)}], {"slides": []})
        self.assertEqual(report["status"], "degraded")
        self.assertFalse(report["blocking"])
        self.assertEqual(report["issues"][0]["issue"], "text_box_overlap")

    def test_malformed_slide_id_preserves_usable_copy_in_recovery(self):
        deck = SimpleNamespace(slides=[SimpleNamespace(slide_id="slide_001", working_title="Planned", rough_message="Intent")])
        for malformed_id in ([], {}):
            recovered = recover_text_plan(json.dumps({"slides": [{"slide_id": malformed_id, "title": "Usable", "text_01": "Useful factual body"}]}), deck)
            self.assertEqual(recovered["slides"][0]["title"], "Usable")
            self.assertEqual(recovered["slides"][0]["text_01"], "Useful factual body")

    def test_missing_slide_does_not_steal_another_identified_slides_copy(self):
        deck = SimpleNamespace(slides=[SimpleNamespace(slide_id=f"slide_{index:03d}", working_title=f"Plan {index}", rough_message=f"Intent {index}") for index in (1, 2)])
        recovered = recover_text_plan(json.dumps({"slides": [{"slide_id": "slide_002", "title": "Second", "text_01": "Second slide only"}]}), deck)
        self.assertEqual(recovered["slides"][0]["text_01"], "Intent 1")
        self.assertEqual(recovered["slides"][1]["text_01"], "Second slide only")


if __name__ == "__main__":
    unittest.main()
