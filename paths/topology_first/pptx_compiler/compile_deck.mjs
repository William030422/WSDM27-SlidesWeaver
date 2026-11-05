import fs from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const pptxgen = require("pptxgenjs");
const { imageSizingCrop } = require("./pptxgenjs_helpers/image.js");
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout.js");

const inputPath = process.argv[2];
const outputPath = process.argv[3];
if (!inputPath || !outputPath) {
  console.error("Usage: node compile_deck.mjs <composed-deck.json> <output.pptx>");
  process.exit(2);
}

const deck = JSON.parse(fs.readFileSync(inputPath, "utf8"));
const pptx = new pptxgen();
pptx.defineLayout({ name: "V9_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "V9_WIDE";
pptx.author = "RL4PPT V9 Step 4";
pptx.company = "RL4PPT";
pptx.subject = "Editable teaching slides composed over generated visual backgrounds";
pptx.title = deck.deck_title || "RL4PPT V9 Deck";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};

const SLIDE_W = 13.333;
const SLIDE_H = 7.5;
const GRID_W = 256;
const GRID_H = 144;

function bboxToInches(bbox) {
  return {
    x: (bbox[0] / GRID_W) * SLIDE_W,
    y: (bbox[1] / GRID_H) * SLIDE_H,
    w: (bbox[2] / GRID_W) * SLIDE_W,
    h: (bbox[3] / GRID_H) * SLIDE_H,
  };
}

function addTextElement(slide, element) {
  const box = bboxToInches(element.bbox);
  const textBox = bboxToInches(element.text_bbox);
  if (element.holder_asset_path && element.holder_bbox) {
    // Intentional containment: editable text is bound to this reusable raster holder.
    const holderBox = bboxToInches(element.holder_bbox);
    slide.addImage({ path: element.holder_asset_path, ...holderBox });
  } else if (element.presentation_mode === "translucent_card") {
    // Intentional containment: the native card is the editable text backplate.
    slide.addShape(pptx.ShapeType.roundRect, {
      ...box,
      rectRadius: 0.04,
      fill: {
        color: element.backplate_color,
        transparency: 100 - element.backplate_opacity,
      },
      line: { color: element.text_color, transparency: 82, width: 0.8 },
    });
  }
  if (element.badge_asset_path && element.badge_bbox) {
    const badgeBox = bboxToInches(element.badge_bbox);
    // Intentional overlap: the badge is attached to the holder's upper-left edge.
    slide.addImage({ path: element.badge_asset_path, ...badgeBox });
  }
  slide.addText(element.text, {
    ...textBox,
    fontFace: element.font_face,
    fontSize: element.font_size,
    color: element.text_color,
    bold: element.font_role === "title" || element.font_role === "emphasized_body",
    italic: element.role === "annotation" || element.role === "supporting_note",
    align: element.alignment,
    valign: "mid",
    margin: 0,
    breakLine: false,
    paraSpaceAfterPt: 0,
    lineSpacingMultiple: 1.02,
    isTextBox: true,
    shadow: undefined,
  });
}

for (const plan of deck.slides) {
  const slide = pptx.addSlide();
  slide.background = { color: "FFFFFF" };
  slide.addImage({
    path: plan.background_path,
    ...imageSizingCrop(plan.background_path, 0, 0, SLIDE_W, SLIDE_H),
  });

  const layerItems = [
    ...(plan.assets || []).map((asset) => ({ kind: "asset", z: asset.z_index, value: asset })),
    ...(plan.elements || []).map((element) => ({ kind: "text", z: element.z_index, value: element })),
  ].sort((first, second) => first.z - second.z);

  for (const item of layerItems) {
    if (item.kind === "asset") {
      const box = bboxToInches(item.value.bbox);
      slide.addImage({
        path: item.value.asset_path,
        ...imageSizingCrop(item.value.asset_path, box.x, box.y, box.w, box.h),
      });
    } else {
      addTextElement(slide, item.value);
    }
  }

  warnIfSlideHasOverlaps(slide, pptx, { muteContainment: true });
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

await pptx.writeFile({ fileName: outputPath });
console.log(JSON.stringify({ status: "succeeded", output: outputPath, slide_count: deck.slides.length }));
