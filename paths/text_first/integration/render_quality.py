"""One final audit of actual PDF text, using the existing Poppler dependency."""
from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


def _token(value: str) -> str:
    value = value.casefold().replace('→', '->').replace('←', '<-').replace('↔', '<->').replace('−', '-')
    if value in {'->', '<-', '<->', '-', '+', '=', '≠', '≥', '≤'}:
        return value
    if re.search(r'\d', value):
        return re.sub(r'[^\w+%./-]', '', value).rstrip('.,')
    return ''.join(c for c in value if c.isalnum())


def _text(element: dict) -> str:
    rich = element.get('rich_text') or []
    if rich:
        return '\n'.join(''.join(str(r.get('text', '')) for r in p.get('runs', [])) for p in rich)
    items = element.get('bullet_items') or []
    if items:
        return '\n'.join(str(v) for v in items)
    return str(element.get('text', ''))


def _box(element: dict) -> list[float]:
    return list(element.get('resolved_text_bbox_px')
                or (element.get('holder_binding') or {}).get('text_bbox_px')
                or element['bbox_px'])


def _probes(plan: dict):
    """Use the same paragraph cells as the exporter, including grid/flex lists."""
    for element in plan.get('elements', []):
        paragraphs = [p for p in element.get('rich_text', [])
                      if any(str(r.get('text', '')).strip() for r in p.get('runs', []))]
        if paragraphs and all(isinstance(p.get('bbox_px'), list) and len(p['bbox_px']) == 4
                              for p in paragraphs):
            for index, paragraph in enumerate(paragraphs):
                yield {**element, 'region_id': f"{element['region_id']}:paragraph:{index}",
                       'parent_region_id': element['region_id'],
                       'resolved_text_bbox_px': paragraph['bbox_px'],
                       'rich_text': [paragraph]}
        else:
            yield element


def _legal_hyphen_wrap(source: str, words: list[dict]) -> bool:
    """A source hyphen permits a line break there; arbitrary word splits do not."""
    boundaries = {len(_token(source[:match.start()])) for match in re.finditer(r'[-\u2010]', source)}
    if not boundaries:
        return False
    consumed = 0
    for first, second in zip(words, words[1:]):
        consumed += len(_token(first['text']))
        y1 = (first['bbox_px'][1] + first['bbox_px'][3]) / 2
        y2 = (second['bbox_px'][1] + second['bbox_px'][3]) / 2
        if abs(y1 - y2) > 5 and (consumed not in boundaries or not first['text'].endswith(('-', '\u2010'))):
            return False
    return True


def _distance(word: dict, box: list[float]) -> float:
    x, y, w, h = box
    a, b, c, d = word['bbox_px']
    return math.hypot(max(x - c, a - x - w, 0), max(y - d, b - y - h, 0))


def inspect_words(words: list[dict], plan: dict, *, tolerance_px: float = 6) -> dict:
    """Match planned words by spelling and proximity; uncertain matches never pass silently."""
    used: set[int] = set()
    records = []
    warnings = []
    for element in _probes(plan):
        box = _box(element)
        source_words = [t for t in _text(element).split() if _token(t)]
        wanted = [_token(t) for t in source_words]
        matched = []
        matched_positions = []
        missing = []
        splits = []
        for source_word, value in zip(source_words, wanted):
            choices = [(i, [i]) for i, word in enumerate(words)
                       if i not in used and _token(word['text']) == value]
            if not choices:
                for i in range(len(words)):
                    if i in used:
                        continue
                    combined = ''
                    for j in range(i, min(len(words), i + 4)):
                        if j in used:
                            break
                        combined += _token(words[j]['text'])
                        if combined == value:
                            choices.append((i, list(range(i, j + 1))))
                            break
                        if not value.startswith(combined):
                            break
            if not choices:
                # Poppler may interleave another column between two lines.
                # Recover a split word geometrically instead of relying only
                # on document order; never merge words on unrelated regions.
                for i, first in enumerate(words):
                    prefix = _token(first['text'])
                    if i in used or len(prefix) < 3 or prefix == value or not value.startswith(prefix):
                        continue
                    for j, last in enumerate(words):
                        if j in used or j == i or prefix + _token(last['text']) != value:
                            continue
                        if (_distance(first, box) <= 30 and _distance(last, box) <= 30
                                and 0 < last['bbox_px'][1] - first['bbox_px'][1] <= 3 * (first['bbox_px'][3] - first['bbox_px'][1])):
                            choices.append((i, [i, j]))
            if not choices:
                missing.append(value)
                continue
            _, indices = min(choices, key=lambda pair: sum(_distance(words[i], box) for i in pair[1]))
            if min(_distance(words[i], box) for i in indices) > max(120, box[3]):
                missing.append(value)
                continue
            if len(indices) > 1:
                centers = [(words[i]['bbox_px'][1] + words[i]['bbox_px'][3]) / 2 for i in indices]
                if max(centers) - min(centers) > 5 and not _legal_hyphen_wrap(source_word, [words[i] for i in indices]):
                    splits.append(value)
            used.update(indices)
            first = words[indices[0]]['bbox_px']
            matched_positions.append({'order': len(matched_positions), 'x': first[0],
                                      'center_y': (first[1] + first[3]) / 2,
                                      'height': first[3] - first[1]})
            matched.extend(words[i] for i in indices)
        violations = []
        x, y, w, h = box
        for word in matched:
            a, b, c, d = word['bbox_px']
            # Poppler word boxes include font ascender/descender space, not
            # painted pixels. Allow that metric overhang without hiding whole
            # extra lines or horizontal spill.
            vertical_tolerance = max(tolerance_px, (d - b) * .24)
            if a < x - tolerance_px or b < y - vertical_tolerance or c > x + w + tolerance_px or d > y + h + vertical_tolerance:
                violations.append(word['text'])
        issues = []
        if violations:
            issues.append('rendered_text_outside_region')
        if splits:
            issues.append('word_broken_across_lines')
        if missing:
            issues.append('rendered_words_unverified')
        lines = []
        for position in sorted(matched_positions, key=lambda item: item['center_y']):
            if lines and abs(position['center_y'] - lines[-1][0]['center_y']) <= .45 * max(position['height'], lines[-1][0]['height']):
                lines[-1].append(position)
            else:
                lines.append([position])
        geometric_order = [item['order'] for line in lines for item in sorted(line, key=lambda item: item['x'])]
        if geometric_order != sorted(geometric_order):
            issues.append('rendered_word_order_changed')
        if element.get('role') == 'title' and matched:
            line_tops = []
            for word in matched:
                top = word['bbox_px'][1]
                if not any(abs(top - other) <= 5 for other in line_tops):
                    line_tops.append(top)
            expected = int(element.get('expected_line_count') or 1)
            if len(line_tops) > expected:
                issues.append('unexpected_title_wrap')
        record = {'region_id': element['region_id'], 'bbox_px': box,
                  'ink_bbox_px': ([min(v['bbox_px'][0] for v in matched), min(v['bbox_px'][1] for v in matched),
                                   max(v['bbox_px'][2] for v in matched), max(v['bbox_px'][3] for v in matched)] if matched else None),
                  'expected_word_count': len(wanted), 'matched_word_count': len(wanted) - len(missing),
                  'outside_words': violations, 'split_words': splits, 'unverified_words': missing,
                  'issues': issues}
        if element.get('parent_region_id'):
            record['parent_region_id'] = element['parent_region_id']
        records.append(record)
        if issues:
            warnings.append({'slide_id': plan.get('slide_id'), 'region_id': element['region_id'], 'issues': issues})
    return {'slide_id': plan.get('slide_id'), 'status': 'degraded' if warnings else 'strict_pass',
            'warnings': warnings, 'regions': records}


def inspect_pdf_text(pdf_path: Path, plans_by_slide: dict[str, dict], *, font_expectations: dict | None = None) -> dict[str, Any]:
    extracted = subprocess.run(['pdftotext', '-bbox', str(pdf_path), '-'], capture_output=True, text=True, check=True)
    root = ET.fromstring(extracted.stdout)
    pages = [node for node in root.iter() if node.tag.rsplit('}', 1)[-1] == 'page']
    font_result = subprocess.run(['pdffonts', str(pdf_path)], capture_output=True, text=True, check=True)
    fonts = [line.split()[0].split('+')[-1] for line in font_result.stdout.splitlines()[2:] if line.strip()]
    warnings = []
    slides = []
    for index, (slide_id, plan) in enumerate(plans_by_slide.items()):
        if index >= len(pages):
            warnings.append({'slide_id': slide_id, 'issues': ['missing_pdf_page']})
            continue
        page = pages[index]
        sx = plan['canvas_px']['width'] / float(page.attrib['width'])
        sy = plan['canvas_px']['height'] / float(page.attrib['height'])
        words = [{'text': node.text or '', 'bbox_px': [float(node.attrib['xMin']) * sx,
                  float(node.attrib['yMin']) * sy, float(node.attrib['xMax']) * sx, float(node.attrib['yMax']) * sy]}
                 for node in page.iter() if node.tag.rsplit('}', 1)[-1] == 'word']
        report = inspect_words(words, plan)
        slides.append(report)
        warnings.extend(report['warnings'])
    expected_fonts = set((font_expectations or {}).values()) or {
        element['font']['face'] for plan in plans_by_slide.values() for element in plan.get('elements', [])}
    for family in sorted(expected_fonts):
        if not any(_token(family) in _token(font) for font in fonts):
            warnings.append({'font': family, 'issues': ['pdf_font_substitution'], 'actual_fonts': fonts})
    return {'status': 'degraded' if warnings else 'strict_pass', 'blocking': False,
            'warnings': warnings, 'slides': slides, 'fonts': fonts,
            'scope': 'actual_pdf_words_fonts_and_line_breaks', 'tolerance_px': 6}
