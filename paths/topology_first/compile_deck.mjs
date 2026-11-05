import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { physicalFont } from "./pptx_compiler/physical_font.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEMO_ROOT = HERE;
const HELPERS = path.join(DEMO_ROOT, "pptx_compiler", "pptxgenjs_helpers");
const demoRequire = createRequire(path.join(DEMO_ROOT, "package.json"));
const pptxgen = demoRequire("pptxgenjs");
const { autoFontSize, calcTextBox } = demoRequire(path.join(HELPERS, "text.js"));
const { imageSizingCrop, imageSizingContain } = demoRequire(path.join(HELPERS, "image.js"));
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = demoRequire(path.join(HELPERS, "layout.js"));

const [compilationPath, outputPath, auditPath] = process.argv.slice(2);
if (!compilationPath || !outputPath) {
  throw new Error("usage: node compile_deck.mjs COMPILATION_JSON OUTPUT_PPTX");
}
const compilation = JSON.parse(fs.readFileSync(compilationPath, "utf8"));
const pptx = new pptxgen();
pptx.defineLayout({ name: "TEXT_PLACEMENT_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "TEXT_PLACEMENT_WIDE";
pptx.author = "RL4PPT reused topology-bound text placement";
pptx.subject = "Diffusion-planned editable presentation";
pptx.lang = "en-US";
const firstPlan = compilation.slides.length
  ? JSON.parse(fs.readFileSync(compilation.slides[0].plan, "utf8"))
  : null;
pptx.theme = {
  headFontFace: physicalFont(firstPlan?.font_roles?.title?.face || "Noto Sans"),
  bodyFontFace: physicalFont(firstPlan?.font_roles?.body?.face || "Noto Sans"),
  lang: "en-US",
};
const textFitAudit = {
  schema_version: "1.0",
  slides: [],
};

for (const record of compilation.slides) {
  const plan = JSON.parse(fs.readFileSync(record.plan, "utf8"));
  for (const element of plan.elements) element.font.face = physicalFont(element.font.face);
  const slide = pptx.addSlide();
  const slideWidth = 13.333;
  const slideHeight = 7.5;
  const canvasWidth = plan.canvas_px.width;
  const canvasHeight = plan.canvas_px.height;
  const bboxToInches = (bbox) => ({
    x: (bbox[0] / canvasWidth) * slideWidth,
    y: (bbox[1] / canvasHeight) * slideHeight,
    w: (bbox[2] / canvasWidth) * slideWidth,
    h: (bbox[3] / canvasHeight) * slideHeight,
  });
  const textBoxFor = (element) => {
    const box = bboxToInches(element.bbox_px);
    let textBox = box;
    if (element.presentation.mode === "native_shape_text") {
      textBox = {
        x: box.x + box.w * 0.06,
        y: box.y + box.h * 0.12,
        w: box.w * 0.88,
        h: box.h * 0.76,
      };
    } else if (element.presentation.mode === "raster_holder_text") {
      if (!element.holder_binding?.text_bbox_px) {
        throw new Error(`${element.region_id}: raster holder is missing text bbox binding`);
      }
      textBox = bboxToInches(element.holder_binding.text_bbox_px);
    }
    if (element.presentation.mode !== "raster_holder_text" && element.internal_padding) {
      const padding = element.internal_padding;
      textBox = {
        x: textBox.x + textBox.w * (padding.left || 0),
        y: textBox.y + textBox.h * (padding.top || 0),
        w: Math.max(0.05, textBox.w * (1 - (padding.left || 0) - (padding.right || 0))),
        h: Math.max(0.05, textBox.h * (1 - (padding.top || 0) - (padding.bottom || 0))),
      };
    }
    if (element.decoration) {
      const decorationWidth = Math.min(box.h * 0.82, box.w * 0.22);
      const gap = Math.min(0.1, box.w * 0.035);
      textBox = {
        x: textBox.x + decorationWidth + gap,
        y: textBox.y,
        w: Math.max(0.05, textBox.w - decorationWidth - gap),
        h: textBox.h,
      };
    }
    return textBox;
  };
  const textForElement = (element) => {
    const items = Array.isArray(element.bullet_items)
      ? element.bullet_items.map((item) => String(item).trim()).filter(Boolean)
      : [];
    if (!items.length) return element.text;
    return items.map((item, index) => ({
      text: item,
      options: {
        bullet: { indent: 14 },
        hanging: 3,
        breakLine: index < items.length - 1,
      },
    }));
  };
  const measureText = (element, fontSize, textBox) => {
    const text = textForElement(element);
    const layout = calcTextBox(fontSize, {
      text,
      w: textBox.w,
      fontFace: element.font.face,
      fontWeight:
        element.font.weight === "bold" || element.font.weight === "semibold"
          ? "bold"
          : "normal",
      leading: 1.08,
      margin: 0,
      padding: 0,
      paraSpaceAfter: Array.isArray(element.bullet_items) && element.bullet_items.length ? 2 : 0,
    });
    return {
      required_height_in: layout.h,
      available_height_in: textBox.h,
      line_count: layout.lines,
      fits: layout.h <= textBox.h * 0.97 + 1e-6,
    };
  };
  const resolvedFontSize = new Map();
  const elementByRegion = new Map(plan.elements.map((element) => [element.region_id, element]));
  for (const group of plan.groups) {
    const members = group.member_region_ids.map((id) => elementByRegion.get(id)).filter(Boolean);
    if (!members.length) continue;
    const fittedSizes = members.map((element) => {
      const textBox = textBoxFor(element);
      return autoFontSize(textForElement(element), element.font.face, {
        ...textBox,
        fontSize: element.font.size_pt,
        minFontSize: element.font.minimum_pt,
        maxFontSize: element.font.size_pt,
        mode: "shrink",
        bold: element.font.weight === "bold" || element.font.weight === "semibold",
        margin: 0,
        leading: 1.08,
        padding: 0,
        paraSpaceAfter: Array.isArray(element.bullet_items) && element.bullet_items.length ? 2 : 0,
      }).fontSize;
    });
    const commonSize = Math.min(...fittedSizes);
    for (const member of members) resolvedFontSize.set(member.region_id, commonSize);
  }
  const slideAudit = {
    slide_id: plan.slide_id,
    overflow_region_ids: [],
    elements: [],
  };
  slide.addImage({
    path: plan.background_path,
    ...imageSizingCrop(plan.background_path, 0, 0, slideWidth, slideHeight),
  });
  // Optional decomposition layer used by the companion combination-background deck.
  // Existing plans omit this field and therefore render exactly as before.
  for (const object of plan.background_objects || []) {
    const objectBox = bboxToInches(object.bbox_px);
    slide.addImage({ path: object.asset_path, ...objectBox });
  }
  for (const accent of plan.accents || []) {
    if (accent.kind === "outline") {
      slide.addShape(pptx.ShapeType.rect, {
        ...bboxToInches(accent.bbox_px),
        fill: { color: accent.color, transparency: 100 },
        line: { color: accent.color, width: accent.width_pt },
      });
    } else if (accent.kind === "connector") {
      const from = bboxToInches([accent.from_px[0], accent.from_px[1], 0, 0]);
      const to = bboxToInches([accent.to_px[0], accent.to_px[1], 0, 0]);
      slide.addShape(pptx.ShapeType.line, {
        x: from.x,
        y: from.y,
        w: to.x - from.x,
        h: to.y - from.y,
        line: { color: accent.color, width: accent.width_pt, endArrowType: "triangle" },
      });
    }
  }
  for (const element of plan.elements) {
    const outerBox = bboxToInches(
      element.holder_binding?.holder_bbox_px || element.bbox_px,
    );
    const textBox = textBoxFor(element);
    const textValue = textForElement(element);
    const fontSize = resolvedFontSize.get(element.region_id) || element.font.size_pt;
    const minimumFit = measureText(element, element.font.minimum_pt, textBox);
    const resolvedFit = measureText(element, fontSize, textBox);
    const overflow = !resolvedFit.fits;
    if (overflow) slideAudit.overflow_region_ids.push(element.region_id);
    slideAudit.elements.push({
      region_id: element.region_id,
      role: element.role,
      is_bulleted: Array.isArray(element.bullet_items) && element.bullet_items.length > 0,
      requested_font_size_pt: element.font.size_pt,
      minimum_font_size_pt: element.font.minimum_pt,
      resolved_font_size_pt: fontSize,
      fits_at_minimum: minimumFit.fits,
      fits_at_resolved_size: resolvedFit.fits,
      overflow,
      required_height_at_minimum_in: Number(minimumFit.required_height_in.toFixed(4)),
      required_height_at_resolved_in: Number(resolvedFit.required_height_in.toFixed(4)),
      available_height_in: Number(textBox.h.toFixed(4)),
      line_count_at_minimum: minimumFit.line_count,
      line_count_at_resolved: resolvedFit.line_count,
    });
    if (element.presentation.mode === "native_shape_text") {
      const shape = element.presentation.shape;
      const fillColor = shape.fill_rgb.map((value) => value.toString(16).padStart(2, "0")).join("").toUpperCase();
      slide.addShape(pptx.ShapeType.roundRect, {
        ...outerBox,
        fill: {
          color: fillColor,
          transparency: Math.round(100 * (1 - Math.max(0.90, shape.fill_opacity))),
        },
        line: { color: fillColor, transparency: 100, width: 0 },
      });
    } else if (element.presentation.mode === "raster_holder_text") {
      if (element.text_backing) {
        const backing = element.text_backing;
        const points = backing.normalized_polygon.map(([x, y]) => ({x: x * outerBox.w, y: y * outerBox.h}));
        points.push({close: true});
        slide.addShape(pptx.ShapeType.custGeom, {
          ...outerBox, points,
          fill: {color: backing.fill_color, transparency: 0},
          line: {color: backing.fill_color, transparency: 100, width: 0},
        });
      }
      // The painted rim stays above the quiet center, and below editable text.
      slide.addImage({ path: element.presentation.quiet_frame_path || element.presentation.asset_path, ...outerBox });
    }
    if (element.decoration) {
      const offsetX = (element.decoration.offset_px?.[0] || 0) / canvasWidth * slideWidth;
      const offsetY = (element.decoration.offset_px?.[1] || 0) / canvasHeight * slideHeight;
      const decorationWidth = Math.min(outerBox.h * 0.82, outerBox.w * 0.22);
      const decorationBox = {
        x: outerBox.x + offsetX,
        y: outerBox.y + outerBox.h * 0.09 + offsetY,
        w: decorationWidth,
        h: outerBox.h * 0.82,
      };
      slide.addImage({
        path: element.decoration.asset_path,
        ...imageSizingContain(
          element.decoration.asset_path,
          decorationBox.x,
          decorationBox.y,
          decorationBox.w,
          decorationBox.h,
        ),
      });
    }
    slide.addText(textValue, {
      ...textBox,
      fontFace: element.font.face,
      fontSize,
      color: element.presentation.text_color,
      bold: element.font.weight === "bold" || element.font.weight === "semibold",
      align:
        Array.isArray(element.bullet_items) && element.bullet_items.length
          ? "left"
          : element.role === "caption"
            ? "center"
            : "left",
      valign: "mid",
      margin: 0,
      breakLine: false,
      paraSpaceAfterPt: 0,
      lineSpacingMultiple: 1.0,
      isTextBox: true,
    });
  }
  for (const accessory of plan.accessories || []) {
    const box = bboxToInches(accessory.bbox_px);
    slide.addImage({
      path: accessory.asset_path,
      ...imageSizingContain(
        accessory.asset_path,
        box.x,
        box.y,
        box.w,
        box.h,
      ),
    });
  }
  textFitAudit.slides.push(slideAudit);
  warnIfSlideHasOverlaps(slide, pptx, { muteContainment: true, ignoreDecorativeShapes: true });
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

await pptx.writeFile({ fileName: outputPath });
if (auditPath) {
  fs.writeFileSync(auditPath, `${JSON.stringify(textFitAudit, null, 2)}\n`, "utf8");
}
console.log(
  JSON.stringify({
    status: "succeeded",
    output: outputPath,
    audit: auditPath || null,
    slide_count: compilation.slides.length,
    overflow_count: textFitAudit.slides.reduce(
      (total, slide) => total + slide.overflow_region_ids.length,
      0,
    ),
  }),
);
