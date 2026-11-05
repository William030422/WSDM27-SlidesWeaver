import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const demoRequire = createRequire(path.join(HERE, "package.json"));
const pptxgen = demoRequire("pptxgenjs");
const { calcTextBox } = demoRequire(
  path.join(HERE, "pptx_compiler", "pptxgenjs_helpers", "text.js"),
);
const { imageSizingCrop } = demoRequire(
  path.join(HERE, "pptx_compiler", "pptxgenjs_helpers", "image.js"),
);

const [compilationPath, outputPath, auditPath] = process.argv.slice(2);
if (!compilationPath || !outputPath) {
  throw new Error("usage: node compile_deck.mjs COMPILATION_JSON OUTPUT_PPTX [AUDIT_JSON]");
}
const compilation = JSON.parse(fs.readFileSync(compilationPath, "utf8"));
const planRecords = compilation.slides.map((record) => ({
  record,
  plan: JSON.parse(fs.readFileSync(record.plan, "utf8")),
}));
const pptx = new pptxgen();
pptx.defineLayout({ name: "HTML_FINAL_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "HTML_FINAL_WIDE";
pptx.author = "RL4PPT final HTML geometry assembly";
pptx.subject = "Expression-logic-planned editable presentation";
pptx.lang = "en-US";
const firstPlan = planRecords[0]?.plan || null;
pptx.theme = {
  headFontFace: firstPlan?.elements?.[0]?.font?.face || "Noto Sans",
  bodyFontFace: firstPlan?.elements?.[0]?.font?.face || "Noto Sans",
  lang: "en-US",
};
const slideWidth = 13.333;
const slideHeight = 7.5;
const textSafeInsetPx = 3;
const audit = {
  schema_version: "2.0",
  font_resolution_scope: "per_html_element",
  text_safe_inset_px: textSafeInsetPx,
  slides: [],
};

const textValue = (element, fontSize = element.font.size_pt) => {
  const rich = Array.isArray(element.rich_text) ? element.rich_text : [];
  if (rich.length) {
    const output = [];
    rich.forEach((paragraph, paragraphIndex) => {
      const runs = (paragraph.runs || []).filter((run) => String(run.text || "").trim());
      runs.forEach((run, runIndex) => {
        output.push({
          text: String(run.text),
          options: {
            bold: Boolean(run.bold),
            italic: Boolean(run.italic),
            fontSize: fontSize * Number(run.font_scale || 1),
            bullet: paragraph.kind === "bullet" && runIndex === 0 ? { indent: 14 } : undefined,
            hanging: paragraph.kind === "bullet" && runIndex === 0 ? 3 : 0,
            breakLine: runIndex === runs.length - 1 && paragraphIndex < rich.length - 1,
          },
        });
      });
    });
    return output;
  }
  const items = (element.bullet_items || []).map((item) => String(item).trim()).filter(Boolean);
  if (!items.length) return element.text;
  return items.map((item, index) => ({
    text: item,
    options: {
      bullet: index === 0 && item === element.text ? undefined : { indent: 14 },
      hanging: index === 0 && item === element.text ? 0 : 3,
      breakLine: index < items.length - 1,
    },
  }));
};
const measurementValue = (element, fontSize) => {
  const rich = Array.isArray(element.rich_text) ? element.rich_text : [];
  if (!rich.length) return textValue(element, fontSize);
  return rich.map((paragraph) => {
    const runs = (paragraph.runs || []).filter((run) => String(run.text || "").trim());
    const maxScale = Math.max(1, ...runs.map((run) => Number(run.font_scale || 1)));
    return {
      text: runs.map((run) => String(run.text)).join(""),
      options: {
        fontSize: fontSize * maxScale,
        bold: paragraph.kind === "lead" || runs.some((run) => Boolean(run.bold)),
        bullet: paragraph.kind === "bullet" ? { indent: 14 } : undefined,
        hanging: paragraph.kind === "bullet" ? 3 : 0,
      },
    };
  });
};
const fitAt = (element, fontSize, box) => {
  const lineCount = element.line_count || element.font.line_count;
  const lineHeightPt = element.line_height_pt || element.font.line_height_pt;
  const sameWidth = Math.abs((element.resolved_text_bbox_px || element.bbox_px)[2] - element.bbox_px[2]) < .01;
  if (lineCount && lineHeightPt && sameWidth && Math.abs(fontSize - element.font.size_pt) < .005) {
    const required = lineHeightPt * lineCount / 72;
    return {fits: required <= box.h + 1 / 115.2, required, lines: lineCount};
  }
  const leading = element.font.line_height_ratio || 1.08;
  const layout = calcTextBox(fontSize, {
    text: measurementValue(element, fontSize),
    w: box.w,
    fontFace: element.font.face,
    fontWeight: element.font.weight === "bold" ? "bold" : "normal",
    leading,
    margin: 0,
    padding: 0,
    paraSpaceAfter: element.bullet_items?.length ? 2 : 0,
  });
  return { fits: layout.h <= box.h * 0.97 + 1e-6, required: layout.h, lines: layout.lines };
};
const resolveFontSize = (element, box) => {
  const lower = element.font.minimum_pt;
  const upper = element.font.size_pt;
    // Shared role typography is committed before final HTML geometry. Keep it
    // identical across slides; report a failure rather than shrink one title.
    if (element.role === 'title' || element.role === 'subtitle') return upper;
  if (fitAt(element, upper, box).fits) return upper;
  if (!fitAt(element, lower, box).fits) return lower;
  let lo = lower;
  let hi = upper;
  let best = lower;
  while (hi - lo > 0.05) {
    const mid = (lo + hi) / 2;
    if (fitAt(element, mid, box).fits) {
      best = mid;
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return best;
};
const holderTextBbox = (holderBbox, safe, insetPx = 0) => [
  holderBbox[0] + holderBbox[2] * safe.x + insetPx,
  holderBbox[1] + holderBbox[3] * safe.y + insetPx,
  Math.max(1, holderBbox[2] * safe.width - insetPx * 2),
  Math.max(1, holderBbox[3] * safe.height - insetPx * 2),
];

const preparedPlans = planRecords.map(({ record, plan }) => {
  const canvasWidth = plan.canvas_px.width;
  const canvasHeight = plan.canvas_px.height;
  const bboxToInches = (bbox) => ({
    x: (bbox[0] / canvasWidth) * slideWidth,
    y: (bbox[1] / canvasHeight) * slideHeight,
    w: (bbox[2] / canvasWidth) * slideWidth,
    h: (bbox[3] / canvasHeight) * slideHeight,
  });
  const rasterById = new Map((plan.raster_layers || []).map((layer) => [layer.layer_id, layer]));
  for (const element of plan.elements) {
    if (element.holder_binding) {
      const layer = rasterById.get(element.holder_binding.layer_id);
      const base = element.holder_binding.holder_bbox_px;
      element.holder_base_bbox_px = base;
      if (layer) layer.bbox_px = base;
      else {
        element.native_holder = { bbox_px: base };
        element.holder_binding = null;
      }
      element.resolved_text_bbox_px ||= holderTextBbox(
        base,
        element.holder_binding?.text_safe_rect || { x: 0.07, y: 0.13, width: 0.86, height: 0.74 },
        textSafeInsetPx,
      );
      element.holder_scale = 1;
    } else if (element.native_holder) {
      const base = element.native_holder.bbox_px;
      element.holder_base_bbox_px = base;
      element.resolved_text_bbox_px ||= holderTextBbox(
        base,
        { x: 0.07, y: 0.13, width: 0.86, height: 0.74 },
      );
      element.holder_scale = 1;
    } else {
      element.resolved_text_bbox_px ||= element.bbox_px;
      element.holder_scale = 1;
    }
    const box = bboxToInches(element.resolved_text_bbox_px || element.bbox_px);
    element.resolved_font_size_pt = resolveFontSize(element, box);
  }
  return { record, plan, bboxToInches };
});

const positionedParagraphs = (element, canvasHeight) => {
  const paragraphs = (element.rich_text || []).filter((paragraph) => paragraph.runs?.some((run) => String(run.text || '').trim()));
  if (!paragraphs.length || !paragraphs.every((paragraph) => Array.isArray(paragraph.bbox_px) && paragraph.bbox_px.length === 4)) return [element];
  return paragraphs.map((paragraph, index) => ({
    ...element,
    region_id: `${element.region_id}:paragraph:${index}`,
    parent_region_id: element.region_id,
    bbox_px: paragraph.bbox_px,
    resolved_text_bbox_px: paragraph.bbox_px,
    text_alignment: paragraph.text_alignment || element.text_alignment,
    rich_text: [{...paragraph,kind:'body'}],
    paragraph_kind: paragraph.kind,
    bullet_indent_px: paragraph.bullet_indent_px,
    bullet_items: [],
    // HTML has already fitted and committed this physical paragraph cell.
    // Preserve its requested type rather than shrink a flattened whole block.
    resolved_font_size_pt: element.font.size_pt,
    line_height_px: paragraph.line_height_px,
    line_height_pt: paragraph.line_height_px * slideHeight * 72 / canvasHeight,
    line_count: paragraph.line_count,
  }));
};

for (const { plan, bboxToInches } of preparedPlans) {
  const slide = pptx.addSlide();

  slide.addImage({
    path: plan.background_path,
    objectName: `background:${plan.slide_id}`,
    altText: `Editable background layer for ${plan.slide_id}`,
    ...imageSizingCrop(plan.background_path, 0, 0, slideWidth, slideHeight),
  });
  for (const element of plan.elements) {
    if (!element.text_backing || element.text_backing.overlay) continue;
    const backing = element.text_backing;
    const backingBox = bboxToInches(backing.bbox_px);
    const points = backing.normalized_polygon?.map(([x, y]) => ({x: x * backingBox.w, y: y * backingBox.h}));
    if (points) points.push({close: true});
    slide.addShape(points ? pptx.ShapeType.custGeom : pptx.ShapeType.roundRect, {
      ...backingBox, points, fill: {color: backing.fill_color},
      line: {color: backing.fill_color, transparency: 100, width: 0},
      objectName: `quiet-interior:${element.region_id}`,
    });
  }
  for (const layer of plan.raster_layers || []) {
    slide.addImage({
      path: layer.path,
      objectName: `raster:${layer.layer_id}`,
      altText: `Editable ${layer.kind} layer for ${plan.slide_id}`,
      ...bboxToInches(layer.bbox_px),
    });
  }
  for (const element of plan.elements) {
    if (element.text_backing?.overlay) {
      const backing = element.text_backing;
      slide.addShape(pptx.ShapeType.roundRect, {
        ...bboxToInches(backing.bbox_px), fill: {color: backing.fill_color},
        line: {color: backing.fill_color, transparency: 100, width: 0},
        objectName: `quiet-interior:${element.region_id}`,
      });
    }
    if (!element.native_holder) continue;
    slide.addShape(pptx.ShapeType.roundRect, {
      ...bboxToInches(element.native_holder.bbox_px),
      fill: { color: "F7F7F4", transparency: 2 },
      line: { color: "666666", width: 1 },
    });
  }

  for (const connector of plan.connectors || []) {
    const from = bboxToInches([...connector.from_px, 0, 0]);
    const to = bboxToInches([...connector.to_px, 0, 0]);
    slide.addShape(pptx.ShapeType.line, {x: from.x, y: from.y, w: to.x - from.x, h: to.y - from.y,
      line: {color: connector.color || '607080', width: connector.width_pt || 2,
        endArrowType: connector.arrow === false ? undefined : 'triangle'}});
  }

  const slideAudit = { slide_id: plan.slide_id, overflow_region_ids: [], elements: [] };
  for (const element of plan.elements.flatMap(element => positionedParagraphs(element, plan.canvas_px.height))) {
    const box = bboxToInches(element.resolved_text_bbox_px || element.bbox_px);
    const fontSize = element.resolved_font_size_pt || element.font.size_pt;
    const minimumFit = fitAt(element, element.font.minimum_pt, box);
    const resolvedFit = fitAt(element, fontSize, box);
    const overflow = !resolvedFit.fits;
    if (overflow) slideAudit.overflow_region_ids.push(element.region_id);
    slideAudit.elements.push({
      region_id: element.region_id,
      role: element.role,
      requested_font_size_pt: element.font.size_pt,
      minimum_font_size_pt: element.font.minimum_pt,
      resolved_font_size_pt: fontSize,
      fits_at_minimum: minimumFit.fits,
      fits_at_resolved_size: resolvedFit.fits,
      overflow,
      holder_scale: element.holder_scale || 1,
      holder_mode: element.holder_binding ? "raster_safe_rect" : (element.native_holder ? "native_fallback" : "none"),
      text_safe_inset_px: element.holder_binding ? textSafeInsetPx : 0,
      resolved_text_bbox_px: (element.resolved_text_bbox_px || element.bbox_px).map((value) => Number(value.toFixed(4))),
      required_height_at_minimum_in: Number(minimumFit.required.toFixed(4)),
      required_height_at_resolved_in: Number(resolvedFit.required.toFixed(4)),
      available_height_in: Number(box.h.toFixed(4)),
      line_count_at_minimum: minimumFit.lines,
      line_count_at_resolved: resolvedFit.lines,
    });
    slide.addText(textValue(element, fontSize), {
      ...box,
      fontFace: element.font.face,
      fontSize,
      color: element.font.color || "111111",
      outline: element.font.outline,
      bold: element.font.weight === "bold",
      italic: Boolean(element.font.italic),
      align: element.text_alignment || "left",
      valign: element.vertical_alignment || "top",
      objectName: `text:${element.region_id}`,
      margin: 0,
      paraSpaceAfterPt: 0,
      bullet: element.parent_region_id && element.paragraph_kind === 'bullet'
        ? {indent:(element.bullet_indent_px || 22.4) * slideHeight * 72 / plan.canvas_px.height} : undefined,
      hanging: element.parent_region_id && element.paragraph_kind === 'bullet' ? 3 : 0,
      // CSS line-height is an absolute baseline distance. PowerPoint's
      // multiplier also multiplies font ascender/descender space and inflated
      // multiline copy even when browser geometry fitted it exactly.
      lineSpacing: element.line_height_px
        ? element.line_height_px * slideHeight * 72 / plan.canvas_px.height
        : (element.font.line_height_pt && Math.abs(fontSize - element.font.size_pt) < .005
          ? element.font.line_height_pt : fontSize * (element.font.line_height_ratio || 1.0)),
      isTextBox: true,
    });
  }
  audit.slides.push(slideAudit);
}

await pptx.writeFile({ fileName: outputPath });
if (auditPath) fs.writeFileSync(auditPath, `${JSON.stringify(audit, null, 2)}\n`);
