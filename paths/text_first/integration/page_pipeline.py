"""Text First page owners, with one explicit shared-typography rendezvous.

Each owner runs its page's dependency chain. Only the coordinator writes the
stage inventories and final deck; no worker calls a whole-deck stage wrapper.
"""
from __future__ import annotations

import copy
import dataclasses
import os
import shutil
from pathlib import Path
from typing import Any

from v9_deck_planning.json_utils import read_json

from .config import PipelineConfig
from .diffusion import build_reused_diffusion_prompts
from .final_background import run_background_set
from .html_compile import run_html_to_coordinates
from .html_planning import commit_html_typography, run_html_planning
from .page_runtime import (
    PageExecutor, PagePhaseError, RunLock, atomic_write_json, fingerprint_paths,
    resource_lease, run_checkpoint,
)
from .text_assembly import prepare_text_assembly_slide, run_reused_text_assembly

STAGES = ("step1", "text_generation", "step2", "expression_logic", "diffusion",
          "html_planning", "html_compile", "background", "assembly")
ROOT = Path(__file__).resolve().parents[1]
VERSION = "text-first-page-owner-v1"


def _runs(job: dict[str, Any], stage: str) -> bool:
    return STAGES.index(job["start_at"]) <= STAGES.index(stage) <= STAGES.index(job["stop_after"])


def _settings(config: PipelineConfig) -> dict[str, Any]:
    # Paths to config files are selectors only; never inspect their contents.
    return {key: value for key, value in dataclasses.asdict(config).items()
            if key not in {"env_path", "api_key_env"}}


def _fingerprint(job: dict[str, Any], files: list[Path], step: str) -> str:
    return fingerprint_paths(files + [Path(__file__)], extra={
        "version": VERSION, "step": step, "slide_id": job["slide_id"],
        "original_index": job["original_index"], "config": _settings(job["config"]),
        "mock_api": job.get("mock_api", False), "mock_images": job.get("mock_images", False),
        "mock_diffusion": job.get("mock_diffusion", False),
    })


def _checkpoint(job: dict[str, Any], step: str, files: list[Path], outputs: Any, fn: Any) -> dict[str, Any]:
    return run_checkpoint(output_dir=Path(job["output_dir"]) / "pages" / job["slide_id"],
                          step=step, fingerprint=_fingerprint(job, files, step),
                          required_outputs=outputs, fn=fn)


def prepare_slide(job: dict[str, Any]) -> dict[str, Any]:
    """The same owner advances diffusion directly into HTML/refinement."""
    root, sid, config = Path(job["output_dir"]), job["slide_id"], job["config"]
    result: dict[str, Any] = {"slide_id": sid, "owner_pid": os.getpid()}
    skeleton = root / "diffusion" / "skeletons" / f"{sid}.png"
    if _runs(job, "diffusion"):
        from .gpu_runtime import generate_diffusion_image
        prompt_path = Path(job["diffusion_prompt"]["prompt"])
        result["diffusion"] = _checkpoint(
            job, "diffusion", [prompt_path, ROOT / "integration/diffusion.py", ROOT / "integration/gpu_runtime.py"], [skeleton],
            lambda: {"slide_id": sid, **generate_diffusion_image(
                config=config, prompt=prompt_path.read_text(encoding="utf-8"), output_path=skeleton,
                seed=config.diffusion_seed + job["original_index"])},
        )
    if _runs(job, "html_planning"):
        slide_dir = root / "html_planning" / "slides" / sid
        draft = slide_dir / "layout_draft.html"
        draft_preview = slide_dir / "draft_preview.png"
        def plan() -> dict[str, Any]:
            with resource_lease("browser"):
                planned = run_html_planning(
                    expression_logic_dir=root / "expression_logic", diffusion_dir=root / "diffusion",
                    step2_dir=root / "step2", text_generation_dir=root / "text_generation",
                    output_dir=root / "html_planning", slide_ids=job["slide_ids"], config=config,
                    mock_vlm=job["mock_api"], only_slide_id=sid, defer_typography=True,
                    write_stage_manifest=False,
                )
            shutil.copyfile(slide_dir / "layout.html", draft)
            shutil.copyfile(slide_dir / "critic_preview.png", draft_preview)
            return planned
        result["html"] = _checkpoint(job, "html_planning", [
            skeleton, root / "step2/deck_plan.json", root / "text_generation/text_plan.json",
            root / "text_generation/parsed_text_plan.json",
            root / "expression_logic/expression_logic_plan.json",
            ROOT / "integration/html_planning.py", ROOT / "integration/html_text_rework.py",
        ], [draft, draft_preview, slide_dir / "visual_generation_groups.json", slide_dir / "text_rework_report.json"], plan)
    return result


def _plan_outputs(result: dict[str, Any]) -> list[Path]:
    plan_path = Path(result["compilation"]["plan"])
    plan = read_json(plan_path)
    files = [plan_path, Path(plan["background_path"])]
    files.extend(Path(layer["path"]) for layer in plan.get("raster_layers", []) if layer.get("path"))
    return list(dict.fromkeys(files))


def prepare_existing_assembly(job: dict[str, Any]) -> dict[str, Any]:
    """Read existing real assets and commit only the per-page native plan."""
    root, sid = Path(job["output_dir"]), job["slide_id"]
    config = job["config"]
    layout_path = root / "html_compile/slides" / sid / "html_layout.json"
    layer_path = root / "final_background" / sid / "layer_manifest.json"
    layer = read_json(layer_path)
    assets = [Path(layer["base_background"])] + [Path(item["path"]) for item in layer["layers"]
                                                      if item.get("path") and Path(item["path"]).is_file()]
    return _checkpoint(job, "assembly_preparation", [
        layout_path, layer_path, root / "text_generation/parsed_text_plan.json",
        ROOT / "integration/text_assembly.py", ROOT / "integration/contrast.py",
        ROOT / "integration/decorative_surface.py", *assets,
    ], _plan_outputs, lambda: prepare_text_assembly_slide(
        html_compile_dir=root / "html_compile", text_generation_dir=root / "text_generation",
        final_background_dir=root / "final_background", output_dir=root / "assembly",
        slide_id=sid, config=config,
    ))


def complete_slide(job: dict[str, Any]) -> dict[str, Any]:
    """Resume the original page owner after typography has committed."""
    root, sid, config = Path(job["output_dir"]), job["slide_id"], job["config"]
    result: dict[str, Any] = {"slide_id": sid, "owner_pid": os.getpid()}
    html_dir = root / "html_planning/slides" / sid
    if _runs(job, "html_compile"):
        committed = html_dir / "layout_committed.html"
        # Legacy completed HTML has no draft/commit journal; freeze it without a new VLM call.
        if job["start_at"] == "html_compile" or not committed.is_file():
            # Explicit compilation treats the user's current HTML as authority,
            # including edits made after the previous completed run.
            shutil.copyfile(html_dir / "layout.html", committed)
        layout_path = root / "html_compile/slides" / sid / "html_layout.json"
        def compile_page() -> dict[str, Any]:
            shutil.copyfile(committed, html_dir / "layout.html")
            with resource_lease("browser"):
                return run_html_to_coordinates(
                    html_planning_dir=root / "html_planning", step2_dir=root / "step2",
                    text_generation_dir=root / "text_generation", output_dir=root / "html_compile",
                    slide_ids=job["slide_ids"], config=config, only_slide_id=sid,
                    write_stage_manifest=False,
                )
        result["compile"] = _checkpoint(job, "html_compile", [
            committed, html_dir / "visual_generation_groups.json", root / "step2/deck_plan.json",
            root / "step2/canonical_background.png", root / "text_generation/text_plan.json",
            ROOT / "integration/html_compile.py",
        ], [layout_path, layout_path.with_name("layout_preview.png"),
            layout_path.with_name("background_control.png")], compile_page)
    if _runs(job, "background"):
        result["background"] = _checkpoint(job, "background", [
            root / "html_compile/slides" / sid / "html_layout.json",
            root / "step2/canonical_background.png", root / "step2/design_system.json",
            ROOT / "integration/final_background.py",
        ], lambda manifest: _background_outputs(root, sid, manifest), lambda: run_background_set(
            project_root=config.project_root, html_compile_dir=root / "html_compile",
            step2_dir=root / "step2", output_dir=root / "final_background", slide_ids=job["slide_ids"],
            # The checkpoint already validated every derived layer. If it is
            # invalid, re-extract the saved board rather than accepting a stale
            # inner slide manifest; _generate_or_resume retains the paid task.
            config=config, mock_images=job["mock_api"] or job["mock_images"], resume_missing=False,
            only_slide_id=sid, write_stage_manifest=False,
        ))
    if _runs(job, "assembly"):
        result["assembly"] = prepare_existing_assembly(job)
    atomic_write_json(root / "pages" / sid / "page_result.json", result)
    return result


def _background_outputs(root: Path, sid: str, result: dict[str, Any]) -> list[Path]:
    manifest_path = root / "final_background" / sid / "layer_manifest.json"
    layer = read_json(manifest_path)
    return [manifest_path, root / "final_background" / sid / "run_manifest.json",
            root / "final_background" / sid / "background.png", Path(layer["base_background"]),
            Path(result["slides"][0]["generated_canvas"]),
            *[Path(item["path"]) for item in layer["layers"] if item.get("path")]]


def _merge_manifests(manifests: list[dict[str, Any]], *, output_dir: Path, key: str = "slides") -> dict[str, Any]:
    merged = copy.deepcopy(manifests[0])
    merged[key] = [item for manifest in manifests for item in manifest.get(key, [])]
    merged["slide_count"] = len(merged[key])
    merged["output_dir"] = str(output_dir.resolve())
    for field in ("warnings", "usage", "failed_slides"):
        if any(field in manifest for manifest in manifests):
            merged[field] = [item for manifest in manifests for item in manifest.get(field, [])]
    for field in ("model_call_count", "image_call_count", "accepted_task_count"):
        if any(field in manifest for manifest in manifests):
            merged[field] = sum(manifest.get(field, 0) for manifest in manifests)
    if any("elapsed_seconds" in manifest for manifest in manifests):
        merged["cumulative_page_seconds"] = sum(manifest.get("elapsed_seconds", 0) for manifest in manifests)
        merged.pop("elapsed_seconds", None)
    qualities = [manifest.get("quality_status", "strict_pass") for manifest in manifests]
    merged["quality_status"] = ("degraded" if any(q not in {"strict_pass", "mock", "skipped"} for q in qualities)
                                else "mock" if "mock" in qualities else "strict_pass")
    return merged


def _load_stage(directory: Path) -> dict[str, Any]:
    manifest = read_json(directory / "run_manifest.json")
    if manifest.get("status") != "succeeded":
        raise RuntimeError(f"Cannot resume incomplete stage: {directory}")
    return {**manifest, "resumed": True, "output_dir": str(directory.resolve())}


def run_page_pipeline(*, output_dir: Path, slide_ids: list[str], config: PipelineConfig,
                      start_at: str, stop_after: str, page_workers: int = 5,
                      gpu_devices: list[str] | None = None, mock_api: bool = False,
                      mock_images: bool = False, mock_diffusion: bool = False,
                      run_lock: bool = True) -> list[dict[str, Any]]:
    """Run page chains, rendezvous once, then assemble committed native plans."""
    if run_lock:
        with RunLock(output_dir):
            return run_page_pipeline(output_dir=output_dir, slide_ids=slide_ids, config=config,
                start_at=start_at, stop_after=stop_after, page_workers=page_workers,
                gpu_devices=gpu_devices, mock_api=mock_api, mock_images=mock_images,
                mock_diffusion=mock_diffusion, run_lock=False)
    prototype = {"start_at": start_at, "stop_after": stop_after}
    if not slide_ids:
        raise ValueError("At least one page is required")
    prompts = {}
    if _runs(prototype, "diffusion"):
        prompt_manifest = build_reused_diffusion_prompts(
            step2_dir=output_dir / "step2", text_generation_dir=output_dir / "text_generation",
            expression_logic_dir=output_dir / "expression_logic", output_dir=output_dir / "diffusion",
            slide_ids=slide_ids,
        )
        prompts = {item["slide_id"]: item for item in prompt_manifest["slides"]}
    jobs = [{**prototype, "slide_id": sid, "original_index": index, "slide_ids": slide_ids,
             "output_dir": str(output_dir), "config": config, "mock_api": mock_api,
             "mock_images": mock_images, "mock_diffusion": mock_diffusion,
             "diffusion_prompt": prompts.get(sid)} for index, sid in enumerate(slide_ids)]
    stages = []
    try:
        with PageExecutor(output_dir=output_dir, workers=page_workers, gpu_devices=gpu_devices,
                          config=config if _runs(prototype, "diffusion") else None,
                          mock_diffusion=mock_api or mock_diffusion, run_lock=run_lock) as executor:
            drafts = executor.run_phase("integration.page_pipeline:prepare_slide", jobs, "draft")
            if _runs(prototype, "diffusion"):
                diffusion = {"stage": "reused_local_z_image_turbo_generation", "status": "succeeded",
                             "quality_status": "mock" if mock_api or mock_diffusion else "strict_pass",
                             "warnings": [], "model": str(config.diffusion_model_path),
                             "resolution": config.diffusion_size.as_list(), "steps": config.diffusion_steps,
                             "seed": config.diffusion_seed, "guidance_scale": config.diffusion_guidance_scale,
                             "output_dir": str(output_dir / "diffusion"), "slide_count": len(slide_ids),
                             "results": [item["diffusion"] for item in drafts]}
                atomic_write_json(output_dir / "diffusion/run_manifest.json", diffusion)
            else:
                diffusion = _load_stage(output_dir / "diffusion")
            stages.append(diffusion)
            if stop_after == "diffusion":
                return stages
            if _runs(prototype, "html_planning"):
                records = [item["html"]["slides"][0] for item in drafts]
                html_dir = output_dir / "html_planning"
                draft_paths = [html_dir / "slides" / sid / "layout_draft.html" for sid in slide_ids]
                committed_paths = [path.with_name("layout_committed.html") for path in draft_paths]
                def commit() -> dict[str, Any]:
                    for path in draft_paths:
                        shutil.copyfile(path, path.with_name("layout.html"))
                        shutil.copyfile(path.with_name("draft_preview.png"), path.with_name("critic_preview.png"))
                    result = commit_html_typography(output_dir=html_dir, slide_ids=slide_ids,
                                                    records=records, config=config, mock_vlm=mock_api)
                    for source, target in zip(draft_paths, committed_paths):
                        shutil.copyfile(source.with_name("layout.html"), target)
                    atomic_write_json(html_dir / "typography_commit.json", result)
                    return result
                typography = run_checkpoint(output_dir=html_dir, step="typography",
                    fingerprint=fingerprint_paths(draft_paths + [ROOT / "integration/html_planning.py",
                        ROOT / "integration/html_text_rework.py"], extra={"config": _settings(config), "mock": mock_api}),
                    required_outputs=committed_paths + [html_dir / "typography_commit.json"], fn=commit)
                html = _merge_manifests([item["html"] for item in drafts], output_dir=html_dir)
                html["typography"] = typography["typography"]
                html["slides"] = typography["slides"]
                html["model_call_count"] += typography["model_call_count"]
                html["warnings"].extend(typography["warnings"])
                html["usage"].extend(typography["usage"])
                if html["warnings"] and not mock_api:
                    html["quality_status"] = "degraded"
                atomic_write_json(html_dir / "run_manifest.json", html)
            else:
                html = _load_stage(output_dir / "html_planning")
            stages.append(html)
            if stop_after == "html_planning":
                return stages
            completed = executor.run_phase("integration.page_pipeline:complete_slide", jobs, "committed")
            for stage, field, directory in (("html_compile", "compile", "html_compile"),
                                             ("background", "background", "final_background")):
                if STAGES.index(stop_after) < STAGES.index(stage):
                    break
                if _runs(prototype, stage):
                    manifest = _merge_manifests([item[field] for item in completed], output_dir=output_dir / directory)
                    atomic_write_json(output_dir / directory / "run_manifest.json", manifest)
                    if stage == "html_compile":
                        atomic_write_json(output_dir / directory / "html_layout.json", {
                            "schema_version": "2.0", "canvas": {"width": 1536, "height": 864},
                            "slides": [{"slide_id": item["slide_id"], "html_layout": item["html_layout"]}
                                       for item in manifest["slides"]]})
                else:
                    manifest = _load_stage(output_dir / directory)
                stages.append(manifest)
            if stop_after == "assembly":
                assembly = run_reused_text_assembly(
                    html_compile_dir=output_dir / "html_compile", text_generation_dir=output_dir / "text_generation",
                    final_background_dir=output_dir / "final_background", output_dir=output_dir / "assembly",
                    slide_ids=slide_ids, config=config, prepared_pages=[item["assembly"] for item in completed])
                stages.append(assembly)
            return stages
    except PagePhaseError as exc:
        atomic_write_json(output_dir / "page_failures.json", {"status": "failed", "phase": exc.phase,
                          "errors": exc.errors, "completed": [item for item in exc.results if item is not None]})
        raise
