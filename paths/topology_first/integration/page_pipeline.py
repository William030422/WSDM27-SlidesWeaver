"""Persistent page ownership for the topology-first pipeline.

Workers only write page-owned artifacts. The parent freezes the exact existing
deck holder policy and writes stage/deck summaries in original slide order.
"""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Any

from v9_deck_planning.json_utils import read_json
from .config import PipelineConfig
from .diffusion import run_page_diffusion
from .final_background import run_soft_boundary_final_backgrounds
from .page_runtime import PageExecutor, PagePhaseError, RunLock, atomic_write_json, fingerprint_paths, run_checkpoint
from .region_content import (attach_semantic_context, build_region_inventory,
                             build_text_presentation_policy, ensure_explanatory_overview,
                             run_region_content_planning)
from .slide_spec import run_slide_spec_bridge
from .text_assembly import prepare_page_assembly, run_reused_text_assembly
from .topology import run_reused_topology_pipeline

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("step1", "step2", "slide_spec", "diffusion", "topology", "region_content", "background", "assembly")
DIRECTORIES = {"slide_spec": "slide_specs", "diffusion": "diffusion", "topology": "topology",
               "region_content": "region_content", "background": "final_background", "assembly": "assembly"}


def _active(job: dict[str, Any], stage: str) -> bool:
    return STAGES.index(job["start_at"]) <= STAGES.index(stage) <= STAGES.index(job["stop_after"])


def _needed(job: dict[str, Any], stage: str) -> bool:
    return STAGES.index(stage) <= STAGES.index(job["stop_after"])


def _code_paths() -> list[Path]:
    return [*sorted((ROOT / "integration").glob("*.py")), ROOT / "compile_deck.mjs",
            ROOT / "measure_text.mjs", ROOT / "pptx_compiler/physical_font.mjs",
            *sorted((ROOT / "reused/topology").glob("*.py")),
            ROOT / "reused/text_placement/text_placement.py", ROOT / "reused/diffusion/build_prompts.py"]


def _asset_paths(root: Path) -> list[Path]:
    manifest_path = root / "step2/asset_manifest.json"
    manifest = read_json(manifest_path)
    return [manifest_path, root / "step2/canonical_background.png", root / "step2/design_system.json",
            *[manifest_path.parent / asset["path"] for asset in manifest.get("assets", [])
              if asset.get("available") and asset.get("path")]]


def _checkpoint(job: dict[str, Any], stage: str, inputs: list[Path], outputs, fn) -> dict[str, Any]:
    config = job["config"]
    fingerprint = fingerprint_paths([*_code_paths(), *inputs], extra={
        "configuration": asdict(config), "slide_id": job["slide_id"],
        "original_index": job["original_index"], "mock_api": job["mock_api"],
        "mock_images": job["mock_images"], "mock_diffusion": job["mock_diffusion"],
        "reuse_topology_raw": job["reuse_topology_raw"], "stage": stage,
    })
    result = run_checkpoint(output_dir=Path(job["output_dir"]) / "pages" / job["slide_id"],
                            step=stage, fingerprint=fingerprint, required_outputs=outputs, fn=fn)
    return result


def _plan_asset_paths(plan: dict[str, Any]) -> list[Path]:
    paths = [Path(plan["background_path"])]
    for element in plan.get("elements", []):
        presentation = element.get("presentation", {})
        for key in ("asset_path", "quiet_frame_path"):
            if presentation.get(key):
                paths.append(Path(presentation[key]))
        if element.get("decoration", {}).get("asset_path"):
            paths.append(Path(element["decoration"]["asset_path"]))
    for collection in ("background_objects", "accessories"):
        paths.extend(Path(item["asset_path"]) for item in plan.get(collection, []) if item.get("asset_path"))
    return list(dict.fromkeys(paths))


def _background_outputs(root: Path, sid: str) -> list[Path]:
    directory = root / "final_background" / sid
    outputs = [directory / name for name in ("background.png", "background_raw.png", "run_manifest.json",
        "request.json", "control_reference.png", "component_briefs.json", "region_coordinates.json", "prompt.txt")]
    if (directory / "task.json").is_file():
        outputs.append(directory / "task.json")
    return outputs


def _page_outputs(directory: Path, sid: str) -> list[Path]:
    plan = read_json(directory / "composition_plan.json")
    return [directory / "composition_plan.json", directory / "page_result.json",
            directory / "rounds/accepted/deck.pptx", directory / "rounds/accepted/deck.pdf",
            directory / "rounds/accepted" / sid / "preview.png", *_plan_asset_paths(plan)]


def _inventory(root: Path, slide_id: str) -> list[dict[str, Any]]:
    deck = read_json(root / "step2/deck_plan.json")
    rough = next(item for item in deck["slides"] if item["slide_id"] == slide_id)
    spec = read_json(root / "slide_specs/slides" / slide_id / "slide_spec.json")
    spec.update(rough_message=rough.get("rough_message", spec["message"]), style=deck.get("style", ""))
    topology_path = root / "topology/calibrated_topology" / slide_id / "calibrated_topology.json"
    layout_path = root / "topology/postprocessed/layouts" / f"{slide_id}.json"
    topology, layout = read_json(topology_path), read_json(layout_path)
    if ensure_explanatory_overview(topology, layout, spec) is not None:
        atomic_write_json(topology_path, topology)
        atomic_write_json(layout_path, layout)
    inventory = build_region_inventory(topology, layout)
    semantic_path = root / "topology/semantic_topology" / slide_id / "semantic_topology.json"
    if semantic_path.is_file():
        inventory = attach_semantic_context(inventory, read_json(semantic_path))
    return inventory


def prepare_slide(job: dict[str, Any]) -> dict[str, Any]:
    """One persistent owner runs spec → GPU → topology → inventory."""
    root, sid, config = Path(job["output_dir"]), job["slide_id"], job["config"]
    stages: dict[str, Any] = {}
    spec_path = root / "slide_specs/slides" / sid / "slide_spec.json"
    if _active(job, "slide_spec"):
        stages["slide_spec"] = _checkpoint(job, "slide_spec", [root / "step2/deck_plan.json", *_asset_paths(root)],
            [spec_path], lambda: run_slide_spec_bridge(project_root=ROOT, step2_dir=root / "step2",
                output_dir=root / "slide_specs", config=config, mock_llm=job["mock_api"],
                slide_ids=[sid], write_stage_manifest=False))
    if _needed(job, "diffusion"):
        if _active(job, "diffusion"):
            stages["diffusion"] = _checkpoint(job, "diffusion", [spec_path],
                [root / "diffusion/skeletons" / f"{sid}.png", root / "diffusion/prompts" / f"{sid}.txt"],
                lambda: run_page_diffusion(slide_spec_dir=root / "slide_specs", output_dir=root / "diffusion",
                    slide_id=sid, original_index=job["original_index"], config=config,
                    mock_diffusion=job["mock_api"] or job["mock_diffusion"]))
    if _needed(job, "topology"):
        if _active(job, "topology"):
            stages["topology"] = _checkpoint(job, "topology", [root / "diffusion/skeletons" / f"{sid}.png",
                spec_path, root / "step2/deck_plan.json"],
                [root / "topology/calibrated_topology" / sid / "calibrated_topology.json",
                 root / "topology/layout_identity" / sid / "layout_identity.json",
                 root / "topology/semantic_topology" / sid / "semantic_topology.json",
                 root / "topology/postprocessed/layouts" / f"{sid}.json",
                 root / "topology/postprocessed" / f"{sid}.png"],
                lambda: _run_topology_and_inventory(job))
    inventory = _inventory(root, sid) if _active(job, "region_content") else None
    return {"slide_id": sid, "original_index": job["original_index"], "owner_pid": os.getpid(),
            "stages": stages, "inventory": inventory}


def _run_topology_and_inventory(job: dict[str, Any]) -> dict[str, Any]:
    root, sid = Path(job["output_dir"]), job["slide_id"]
    result = run_reused_topology_pipeline(project_root=ROOT, skeleton_dir=root / "diffusion/skeletons",
        output_dir=root / "topology", slide_ids=[sid], config=job["config"],
        reuse_raw=job["reuse_topology_raw"] or job["mock_api"], offline=job["mock_api"],
        write_stage_manifest=False)
    if _active(job, "region_content"):
        # Inventory can add an authoritative overview to this page's topology.
        # Commit that mutation before hashing the topology checkpoint outputs.
        _inventory(root, sid)
    return result


def finish_slide(job: dict[str, Any]) -> dict[str, Any]:
    """The same owner continues content → background → accepted editable page."""
    root, sid, config = Path(job["output_dir"]), job["slide_id"], job["config"]
    stages: dict[str, Any] = {}
    policy_path = root / "region_content/text_presentation_policy.json"
    topology_path = root / "topology/calibrated_topology" / sid / "calibrated_topology.json"
    layout_path = root / "topology/postprocessed/layouts" / f"{sid}.json"
    content_path = root / "region_content/slides" / sid / "region_content_plan.json"
    semantic_path = root / "topology/semantic_topology" / sid / "semantic_topology.json"
    semantic_inputs = [semantic_path] if semantic_path.is_file() else []
    if _active(job, "region_content"):
        stages["region_content"] = _checkpoint(job, "region_content",
            [root / "slide_specs/slides" / sid / "slide_spec.json", root / "step2/deck_plan.json",
             topology_path, layout_path, policy_path, *semantic_inputs, *_asset_paths(root)],
            [content_path, content_path.with_name("region_inventory.json")],
            lambda: run_region_content_planning(project_root=ROOT, step2_dir=root / "step2",
                slide_spec_dir=root / "slide_specs", topology_dir=root / "topology", output_dir=root / "region_content",
                slide_ids=[sid], config=config, mock_llm=job["mock_api"], text_policy=job["text_policy"],
                write_stage_manifest=False))
    if _active(job, "background"):
        stages["background"] = _checkpoint(job, "background",
            [content_path, topology_path, layout_path, *_asset_paths(root)],
            lambda result: _background_outputs(root, sid),
            lambda: run_soft_boundary_final_backgrounds(project_root=ROOT, region_content_dir=root / "region_content",
                step2_dir=root / "step2", topology_dir=root / "topology", output_dir=root / "final_background",
                slide_ids=[sid], config=config, mock_images=job["mock_api"] or job["mock_images"],
                write_stage_manifest=False))
    page = None
    if _active(job, "assembly"):
        page_dir = root / "assembly/pages" / sid
        page = _checkpoint(job, "page_assembly", [content_path, content_path.with_name("region_inventory.json"),
            topology_path, layout_path, policy_path, root / "step2/deck_plan.json",
            root / "final_background" / sid / "background.png",
            root / "final_background" / sid / "component_briefs.json", *_asset_paths(root)],
            lambda result: _page_outputs(page_dir, sid),
            lambda: prepare_page_assembly(region_content_dir=root / "region_content", step2_dir=root / "step2",
                topology_dir=root / "topology", final_background_dir=root / "final_background", output_dir=page_dir,
                slide_id=sid, config=config, mock_vlm=job["mock_api"]))
    return {"slide_id": sid, "original_index": job["original_index"], "owner_pid": os.getpid(),
            "stages": stages, "page": page}


def _summarize(stage: str, results: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    values = [result["stages"][stage] for result in results]
    summary = {**values[0], "slide_count": len(values), "output_dir": str((root / DIRECTORIES[stage]).resolve()),
               "page_parallel": True, "status": "succeeded"}
    for key in ("slides", "results", "warnings", "layout_identity", "semantic_topology", "regularization", "postprocess"):
        if any(key in value for value in values):
            summary[key] = [entry for value in values for entry in value.get(key, [])]
    for key in ("model_call_count", "image_call_count"):
        if any(key in value for value in values):
            summary[key] = sum(value.get(key, 0) for value in values)
    statuses = {value.get("quality_status") for value in values}
    summary["quality_status"] = ("degraded" if statuses - {None, "strict_pass", "mock"}
                                  else "mock" if "mock" in statuses else "strict_pass")
    atomic_write_json(root / DIRECTORIES[stage] / "run_manifest.json", summary)
    if stage == "diffusion":
        atomic_write_json(root / "diffusion/prompt_manifest.json", {"stage": "slide_spec_to_diffusion_prompt",
            "slide_count": len(values), "slides": summary["slides"]})
    return summary


def _resume_stage_summary(stage: str, root: Path, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    path = root / DIRECTORIES[stage] / "run_manifest.json"
    if path.is_file():
        manifest = read_json(path)
        if manifest.get("status") == "succeeded":
            return manifest
    # A later page can fail after every page completed an earlier stage. Their
    # page checkpoints remain authoritative even without the phase summary.
    results = []
    for job in jobs:
        checkpoint = read_json(root / "pages" / job["slide_id"] / "checkpoints" / f"{stage}.json")
        if checkpoint.get("status") != "completed" or checkpoint.get("result", {}).get("status") != "succeeded":
            raise RuntimeError(f"Cannot resume incomplete {stage} for {job['slide_id']}")
        results.append({"stages": {stage: checkpoint["result"]}})
    return _summarize(stage, results, root)


def _run_parallel_page_pipeline(
    *, output_dir: Path, config: PipelineConfig, start_at: str = "slide_spec", stop_after: str = "assembly",
    workers: int = 5, gpu_devices: list[str] | None = None, mock_api: bool = False,
    mock_images: bool = False, mock_diffusion: bool = False, reuse_topology_raw: bool = False,
) -> list[dict[str, Any]]:
    """Run page chains after Step 2; caller owns the run lock and initial planning."""
    root = output_dir.resolve()
    deck = read_json(root / "step2/deck_plan.json")
    jobs = [{"slide_id": slide["slide_id"], "original_index": index, "output_dir": str(root),
             "config": config, "start_at": start_at, "stop_after": stop_after,
             "mock_api": mock_api, "mock_images": mock_images, "mock_diffusion": mock_diffusion,
             "reuse_topology_raw": reuse_topology_raw} for index, slide in enumerate(deck["slides"])]
    manifests: list[dict[str, Any]] = []
    needs_diffusion = STAGES.index(start_at) <= STAGES.index("diffusion") <= STAGES.index(stop_after)
    with PageExecutor(output_dir=root, workers=workers, gpu_devices=gpu_devices, config=config if needs_diffusion else None,
                      mock_diffusion=mock_api or mock_diffusion, run_lock=False) as executor:
        prepared = executor.run_phase(entrypoint="integration.page_pipeline:prepare_slide", jobs=jobs, phase="inventory")
        for stage in ("slide_spec", "diffusion", "topology"):
            if STAGES.index(stage) > STAGES.index(stop_after):
                break
            if STAGES.index(stage) >= STAGES.index(start_at):
                manifests.append(_summarize(stage, prepared, root))
            else:
                manifests.append(_resume_stage_summary(stage, root, jobs))
        if STAGES.index(stop_after) < STAGES.index("region_content"):
            return manifests
        policy_path = root / "region_content/text_presentation_policy.json"
        if STAGES.index(start_at) <= STAGES.index("region_content"):
            manifest_path = root / "step2/asset_manifest.json"
            policy = build_text_presentation_policy(deck_plan=deck, asset_manifest=read_json(manifest_path),
                asset_manifest_path=manifest_path, inventories={item["slide_id"]: item["inventory"] for item in prepared})
            atomic_write_json(policy_path, policy)
        else:
            policy = read_json(policy_path)
        completed = executor.run_phase(entrypoint="integration.page_pipeline:finish_slide",
            jobs=[{**job, "text_policy": policy} for job in jobs], phase="page_complete")
        for stage in ("region_content", "background"):
            if STAGES.index(stage) > STAGES.index(stop_after):
                break
            if STAGES.index(stage) >= STAGES.index(start_at):
                manifests.append(_summarize(stage, completed, root))
            else:
                manifests.append(_resume_stage_summary(stage, root, jobs))
        if stop_after == "assembly":
            pages = [item["page"] for item in completed]
            manifests.append(run_reused_text_assembly(region_content_dir=root / "region_content", step2_dir=root / "step2",
                topology_dir=root / "topology", final_background_dir=root / "final_background", output_dir=root / "assembly",
                slide_ids=[job["slide_id"] for job in jobs], config=config, mock_vlm=mock_api, prepared_pages=pages))
        atomic_write_json(root / "page_results.json", {"workers": workers, "page_count": len(jobs),
            "phases": {"inventory": [{k: item[k] for k in ("slide_id", "original_index", "owner_pid")} for item in prepared],
                       "page_complete": [{k: item[k] for k in ("slide_id", "original_index", "owner_pid")} for item in completed]}})
    return manifests


def run_parallel_page_pipeline(
    *, output_dir: Path, config: PipelineConfig, start_at: str = "slide_spec", stop_after: str = "assembly",
    workers: int = 5, gpu_devices: list[str] | None = None, mock_api: bool = False,
    mock_images: bool = False, mock_diffusion: bool = False, reuse_topology_raw: bool = False,
    run_lock: bool = True,
) -> list[dict[str, Any]]:
    """Run complete page chains; direct callers receive exclusive run ownership."""
    from contextlib import nullcontext
    root = output_dir.resolve()
    with RunLock(root) if run_lock else nullcontext():
        try:
            manifests = _run_parallel_page_pipeline(output_dir=root, config=config, start_at=start_at,
                stop_after=stop_after, workers=workers, gpu_devices=gpu_devices, mock_api=mock_api,
                mock_images=mock_images, mock_diffusion=mock_diffusion, reuse_topology_raw=reuse_topology_raw)
            atomic_write_json(root / "run_manifest.json", {"status": "succeeded", "pipeline": "topology_page_parallelism",
                "start_at": start_at, "stop_after": stop_after,
                "quality_status": "degraded" if any(item.get("quality_status") not in {None, "strict_pass", "mock"}
                                                     for item in manifests) else "mock" if mock_api else "strict_pass",
                "stages": [{"stage": item.get("stage"), "status": item.get("status"),
                            "quality_status": item.get("quality_status"), "output_dir": item.get("output_dir")}
                           for item in manifests]})
            return manifests
        except BaseException as exc:
            errors = exc.errors if isinstance(exc, PagePhaseError) else [{"error_type": type(exc).__name__}]
            report = {"status": "failed", "pipeline": "topology_page_parallelism",
                "phase": getattr(exc, "phase", "final_assembly"),
                "errors": [{key: value for key, value in error.items() if key in {"slide_id", "owner_id", "error_type"}}
                           for error in errors],
                "healthy_page_ids": [item["slide_id"] for item in getattr(exc, "results", []) if item is not None],
                "resume": "Rerun the same stage range; completed page checkpoints are preserved."}
            atomic_write_json(root / "page_failures.json", report)
            atomic_write_json(root / "run_manifest.json", report)
            raise
