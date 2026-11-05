import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { physicalFont } from "./pptx_compiler/physical_font.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const demoRequire = createRequire(path.join(HERE, "package.json"));
const { calcTextBox } = demoRequire(
  path.join(HERE, "pptx_compiler", "pptxgenjs_helpers", "text.js"),
);

const requestPath = process.argv[2];
if (!requestPath) throw new Error("usage: node measure_text.mjs REQUEST_JSON");
const request = JSON.parse(fs.readFileSync(requestPath, "utf8"));
const canvasWidth = request.canvas_px.width;
const canvasHeight = request.canvas_px.height;

const measureTightWidthPx = (state) => {
  const fontWeight =
    state.font_weight === "bold" || state.font_weight === "semibold"
      ? "bold"
      : "normal";
  const expectedLines = Math.max(1, String(state.text || "").split(/\r?\n/).length);
  let low = 0.01;
  let high = (state.text_bbox_px[2] / canvasWidth) * 13.333;
  const layoutAt = (width) =>
    calcTextBox(state.font_size_pt, {
      text: state.text,
      w: width,
      fontFace: state.font_face,
      fontWeight,
      leading: 1.08,
      margin: 0,
      padding: 0,
    });
  if (layoutAt(high).lines > expectedLines) {
    return state.text_bbox_px[2];
  }
  for (let iteration = 0; iteration < 32; iteration += 1) {
    const middle = (low + high) / 2;
    if (layoutAt(middle).lines <= expectedLines) high = middle;
    else low = middle;
  }
  return Math.min(
    state.text_bbox_px[2],
    (high / 13.333) * canvasWidth + 1,
  );
};

const results = request.states.map((state) => {
  state = { ...state, font_face: physicalFont(state.font_face) };
  const items = Array.isArray(state.bullet_items)
    ? state.bullet_items.map((item) => String(item).trim()).filter(Boolean)
    : [];
  const text = items.length
    ? items.map((item, index) => ({
        text: item,
        options: {
          bullet: { indent: 14 },
          hanging: 3,
          breakLine: index < items.length - 1,
        },
      }))
    : state.text;
  const textBox = {
    w: (state.text_bbox_px[2] / canvasWidth) * 13.333,
    h: (state.text_bbox_px[3] / canvasHeight) * 7.5,
  };
  const layout = calcTextBox(state.font_size_pt, {
    text,
    w: textBox.w,
    fontFace: state.font_face,
    fontWeight:
      state.font_weight === "bold" || state.font_weight === "semibold"
        ? "bold"
        : "normal",
    leading: 1.08,
    margin: 0,
    padding: 0,
    paraSpaceAfter: items.length ? 2 : 0,
  });
  return {
    state_id: state.state_id,
    required_height_in: layout.h,
    available_height_in: textBox.h,
    line_count: layout.lines,
    fits: layout.h <= textBox.h * 0.97 + 1e-6,
    tight_width_px: state.measure_tight_width ? measureTightWidthPx(state) : null,
  };
});

console.log(JSON.stringify({ status: "succeeded", results }));
