from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from v9_deck_planning.json_utils import read_json, write_json, write_text
from v9_deck_planning.models import DeckPlan

from .config import PipelineConfig


ROOT = Path(__file__).resolve().parents[1]
REUSED_DIR = ROOT / "reused" / "diffusion"


def _load_reused_module(filename: str, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, REUSED_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reused Diffusion module: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_reused_diffusion_prompts(
    *,
    step2_dir: Path,
    text_generation_dir: Path,
    expression_logic_dir: Path,
    output_dir: Path,
    slide_ids: list[str] | None = None,
) -> dict[str, Any]:
    builder = _load_reused_module("build_prompts.py", "reused_build_prompts")
    deck_plan = DeckPlan.model_validate(read_json(step2_dir / "deck_plan.json"))
    text_plan = read_json(text_generation_dir / "text_plan.json")
    titles_by_slide = {
        item["slide_id"]: str(item["title"]).strip()
        for item in text_plan.get("slides", [])
    }
    expression_plan = read_json(
        expression_logic_dir / "expression_logic_plan.json"
    )
    logic_by_slide = {
        item["slide_id"]: str(item["expression_logic"]).strip()
        for item in expression_plan.get("slides", [])
    }
    expected_ids = [slide.slide_id for slide in deck_plan.slides]
    active_ids = slide_ids or expected_ids
    if expected_ids[: len(active_ids)] != active_ids:
        raise ValueError("diffusion slide limit must preserve the Step 2 slide order")
    if list(titles_by_slide) != expected_ids or list(logic_by_slide) != active_ids:
        raise ValueError("diffusion inputs must preserve the selected slide inventory")

    prompt_dir = output_dir / "prompts"
    input_dir = output_dir / "inputs"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for slide in deck_plan.slides[: len(active_ids)]:
        diffusion_input = {
            "slide_id": slide.slide_id,
            "title": titles_by_slide[slide.slide_id],
            "purpose": slide.rough_message,
            "visual_logic": logic_by_slide[slide.slide_id],
        }
        input_path = input_dir / slide.slide_id / "diffusion_input.json"
        prompt_path = prompt_dir / f"{slide.slide_id}.txt"
        write_json(input_path, diffusion_input)
        write_text(prompt_path, builder.build_prompt(diffusion_input))
        records.append(
            {
                "slide_id": slide.slide_id,
                "diffusion_input": str(input_path.resolve()),
                "prompt": str(prompt_path.resolve()),
            }
        )
    manifest = {
        "stage": "expression_logic_to_diffusion_prompt",
        "status": "succeeded",
        "source_template": str((REUSED_DIR / "build_prompts.py").resolve()),
        "template_unchanged": False,
        "included_fields": ["title", "purpose", "visual_logic"],
        "excluded_fields": ["subtitle", "text_blocks", "frozen_body_copy"],
        "slide_count": len(records),
        "output_dir": str(output_dir.resolve()),
        "slides": records,
    }
    write_json(output_dir / "prompt_manifest.json", manifest)
    return manifest


def run_reused_diffusion(
    *,
    step2_dir: Path,
    text_generation_dir: Path,
    expression_logic_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    mock_diffusion: bool,
    slide_ids: list[str] | None = None,
) -> dict[str, Any]:
    prompt_manifest = build_reused_diffusion_prompts(
        step2_dir=step2_dir,
        text_generation_dir=text_generation_dir,
        expression_logic_dir=expression_logic_dir,
        output_dir=output_dir,
        slide_ids=slide_ids,
    )
    skeleton_dir = output_dir / "skeletons"
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    results = []
    warnings = []
    if config.diffusion_size.as_list() != [1024, 576]:
        warnings.append(
            "Diffusion source expects 1024x576; using a mock reference instead"
        )
        mock_diffusion = True
    if mock_diffusion:
        results.extend(_write_mock_skeletons(prompt_manifest["slides"], skeleton_dir, config))
    else:
        try:
            import torch
            from diffusers import ZImagePipeline

            if not torch.cuda.is_available():
                raise RuntimeError("Diffusion generation requires a CUDA GPU")
            pipe = ZImagePipeline.from_pretrained(
                config.diffusion_model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
                local_files_only=True,
            )
            pipe.to("cuda")
            for index, record in enumerate(prompt_manifest["slides"]):
                prompt_path = Path(record["prompt"])
                output_path = skeleton_dir / f"{record['slide_id']}.png"
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                image = pipe(
                    prompt=prompt_path.read_text(encoding="utf-8").strip(),
                    height=config.diffusion_size.height,
                    width=config.diffusion_size.width,
                    num_inference_steps=config.diffusion_steps,
                    guidance_scale=config.diffusion_guidance_scale,
                    generator=torch.Generator(device="cuda").manual_seed(
                        config.diffusion_seed + index
                    ),
                ).images[0]
                torch.cuda.synchronize()
                image.convert("RGB").save(output_path)
                results.append(
                    {
                        "slide_id": record["slide_id"],
                        "output": str(output_path.resolve()),
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "peak_gpu_memory_gib": round(
                            torch.cuda.max_memory_allocated() / (1024**3), 3
                        ),
                        "mock": False,
                    }
                )
        except Exception as exc:
            warnings.append(str(exc))
            completed = {item["slide_id"] for item in results}
            missing = [
                record
                for record in prompt_manifest["slides"]
                if record["slide_id"] not in completed
            ]
            results.extend(_write_mock_skeletons(missing, skeleton_dir, config))
    manifest = {
        "stage": "reused_local_z_image_turbo_generation",
        "status": "succeeded",
        "quality_status": "fallback" if warnings else ("mock" if mock_diffusion else "strict_pass"),
        "warnings": warnings,
        "source_runner": str((REUSED_DIR / "generate_slides.py").resolve()),
        "model": str(config.diffusion_model_path),
        "resolution": config.diffusion_size.as_list(),
        "steps": config.diffusion_steps,
        "seed": config.diffusion_seed,
        "guidance_scale": config.diffusion_guidance_scale,
        "slide_count": len(results),
        "output_dir": str(output_dir.resolve()),
        "results": results,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def _write_mock_skeletons(
    records: list[dict[str, Any]],
    skeleton_dir: Path,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    results = []
    for record in records:
        output_path = skeleton_dir / f"{record['slide_id']}.png"
        image = Image.new(
            "RGB",
            (config.diffusion_size.width, config.diffusion_size.height),
            "white",
        )
        draw = ImageDraw.Draw(image)
        draw.rectangle((48, 48, 976, 528), outline="black", width=4)
        draw.rectangle((80, 96, 464, 176), fill=(225, 225, 225))
        draw.rectangle((560, 128, 928, 448), fill=(200, 200, 200))
        image.save(output_path)
        results.append(
            {
                "slide_id": record["slide_id"],
                "output": str(output_path.resolve()),
                "mock": True,
            }
        )
    return results
