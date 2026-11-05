from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from v9_deck_planning.json_utils import read_json, write_json, write_text

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
    slide_spec_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    builder = _load_reused_module("build_prompts.py", "reused_build_prompts")
    prompt_dir = output_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    records = []
    slide_spec_paths = sorted(
        slide_spec_dir.glob("slides/slide_*/slide_spec.json")
    )
    if not slide_spec_paths:
        raise FileNotFoundError("no slide_spec.json artifacts found")
    for slide_spec_path in slide_spec_paths:
        slide_spec = read_json(slide_spec_path)
        prompt = builder.build_prompt(slide_spec)
        prompt_path = prompt_dir / f"{slide_spec['slide_id']}.txt"
        write_text(prompt_path, prompt)
        records.append(
            {
                "slide_id": slide_spec["slide_id"],
                "slide_spec": str(slide_spec_path.resolve()),
                "prompt": str(prompt_path.resolve()),
            }
        )
    manifest = {
        "stage": "slide_spec_to_diffusion_prompt",
        "source_template": str((REUSED_DIR / "build_prompts.py").resolve()),
        "template_unchanged": False,  # The documented visual_intent adapter modifies the V9-era template.
        "field_adapter": {
            "title": "title",
            "message": "message",
            "visual_intent": "visual_intent",
        },
        "slide_count": len(records),
        "slides": records,
    }
    write_json(output_dir / "prompt_manifest.json", manifest)
    return manifest


def run_reused_diffusion(
    *,
    slide_spec_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    mock_diffusion: bool,
) -> dict[str, Any]:
    prompt_manifest = build_reused_diffusion_prompts(
        slide_spec_dir=slide_spec_dir,
        output_dir=output_dir,
    )
    if config.diffusion_size.as_list() != [1024, 576]:
        raise ValueError("Diffusion skeletons must be exactly 1024x576")
    skeleton_dir = output_dir / "skeletons"
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    results = []
    warnings = []
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
                record for record in prompt_manifest["slides"]
                if record["slide_id"] not in completed
            ]
            results.extend(_write_mock_skeletons(missing, skeleton_dir, config))
    manifest = {
        "stage": "reused_local_z_image_turbo_generation",
        "status": "succeeded",
        "quality_status": "fallback" if warnings else "strict_pass",
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


def run_page_diffusion(
    *, slide_spec_dir: Path, output_dir: Path, slide_id: str,
    original_index: int, config: PipelineConfig, mock_diffusion: bool,
) -> dict[str, Any]:
    """Submit one page to the shared, long-lived GPU owner, without deck writes."""
    from .gpu_runtime import generate_diffusion_image
    builder = _load_reused_module("build_prompts.py", "reused_build_prompts")
    spec_path = slide_spec_dir / "slides" / slide_id / "slide_spec.json"
    prompt = builder.build_prompt(read_json(spec_path))
    prompt_path = output_dir / "prompts" / f"{slide_id}.txt"
    write_text(prompt_path, prompt)
    skeleton_dir = output_dir / "skeletons"
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    output_path = skeleton_dir / f"{slide_id}.png"
    warnings = []
    if mock_diffusion:
        record = _write_mock_skeletons([{"slide_id": slide_id}], skeleton_dir, config)[0]
    else:
        try:
            record = generate_diffusion_image(config=config, prompt=prompt, output_path=output_path,
                                               seed=config.diffusion_seed + original_index)
        except Exception as exc:
            # Failure is page-local; preserve the established explicit fallback.
            warnings.append(str(exc))
            record = _write_mock_skeletons([{"slide_id": slide_id}], skeleton_dir, config)[0]
    return {"stage": "reused_local_z_image_turbo_generation", "status": "succeeded",
            "quality_status": "fallback" if warnings else "mock" if mock_diffusion else "strict_pass", "warnings": warnings,
            "output_dir": str(output_dir.resolve()), "slide_count": 1,
            "results": [{**record, "slide_id": slide_id, "seed": config.diffusion_seed + original_index}],
            "slides": [{"slide_id": slide_id, "slide_spec": str(spec_path.resolve()),
                        "prompt": str(prompt_path.resolve())}]}
