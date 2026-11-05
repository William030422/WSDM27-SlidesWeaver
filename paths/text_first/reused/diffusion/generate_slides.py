from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from diffusers import ZImagePipeline
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.environ.get("DIFFUSION_MODEL_PATH", Path.home() / "models" / "Tongyi-MAI_Z-Image-Turbo")
).expanduser()
PROMPT_DIR = ROOT / "prompts"
OUTPUT_DIR = ROOT / "slides"
STEPS = 5
WIDTH = 1024
HEIGHT = 576
SEED = 42


def make_contact_sheet(slides: list[tuple[str, Image.Image]]) -> Image.Image:
    thumb_width = 512
    thumb_height = 288
    label_height = 32
    canvas = Image.new(
        "RGB", (thumb_width * 2, (thumb_height + label_height) * 3), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (slide_id, image) in enumerate(slides):
        x = (index % 2) * thumb_width
        y = (index // 2) * (thumb_height + label_height)
        canvas.paste(image.resize((thumb_width, thumb_height)), (x, y + label_height))
        draw.text((x + 12, y + 10), slide_id, fill="black")
    return canvas


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this experiment")

    prompt_paths = sorted(PROMPT_DIR.glob("slide_*.txt"))
    if len(prompt_paths) != 6:
        raise ValueError(f"expected 6 slide prompts, found {len(prompt_paths)}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipe = ZImagePipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        local_files_only=True,
    )
    pipe.to("cuda")

    results = []
    slides: list[tuple[str, Image.Image]] = []
    for prompt_path in prompt_paths:
        slide_id = prompt_path.stem
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        image = pipe(
            prompt=prompt,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=STEPS,
            guidance_scale=0.0,
            generator=torch.Generator(device="cuda").manual_seed(SEED),
        ).images[0]
        torch.cuda.synchronize()
        output_path = OUTPUT_DIR / f"{slide_id}.png"
        image.save(output_path)
        slides.append((slide_id, image))
        results.append(
            {
                "slide_id": slide_id,
                "prompt": str(prompt_path.resolve()),
                "output": str(output_path.resolve()),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "peak_gpu_memory_gib": round(
                    torch.cuda.max_memory_allocated() / (1024**3), 3
                ),
            }
        )

    contact_sheet_path = OUTPUT_DIR / "contact_sheet.png"
    make_contact_sheet(slides).save(contact_sheet_path)
    manifest = {
        "stage": "local_z_image_turbo_generation",
        "prompt_strategy": "static_curated_message_and_single_editorial_scene",
        "model": str(MODEL_PATH),
        "resolution": [WIDTH, HEIGHT],
        "steps": STEPS,
        "seed": SEED,
        "guidance_scale": 0.0,
        "dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(),
        "slide_count": len(results),
        "results": results,
        "contact_sheet": str(contact_sheet_path.resolve()),
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
