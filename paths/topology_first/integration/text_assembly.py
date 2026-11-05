from __future__ import annotations

import importlib.util
import hashlib
import copy
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from v9_deck_planning.apimart import ApiMartClient
from v9_deck_planning.json_utils import read_json, write_json

from .background_decomposition import decompose_background
from .config import PipelineConfig, resolve_deck_font
from .render_quality import inspect_pdf_text


ROOT = Path(__file__).resolve().parents[1]
REUSED_TEXT_PLACEMENT = ROOT / "reused" / "text_placement" / "text_placement.py"
TEXT_MEASURER = ROOT / "measure_text.mjs"


def _load_text_placement() -> Any:
    spec = importlib.util.spec_from_file_location(
        "reused_text_placement", REUSED_TEXT_PLACEMENT
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load reused text placement module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_reused_text_assembly(
    *,
    region_content_dir: Path,
    step2_dir: Path,
    topology_dir: Path,
    final_background_dir: Path,
    output_dir: Path,
    slide_ids: list[str],
    config: PipelineConfig,
    mock_vlm: bool = False,
    prepared_pages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    text_placement = _load_text_placement()
    if prepared_pages is None:
        plans_by_slide, content_unit_fit, critic_log = _prepare_assembly_pages(
            region_content_dir=region_content_dir, step2_dir=step2_dir, topology_dir=topology_dir,
            final_background_dir=final_background_dir, output_dir=output_dir, slide_ids=slide_ids,
            config=config, mock_vlm=mock_vlm,
        )
    else:
        by_id = {page["slide_id"]: page for page in prepared_pages}
        if set(by_id) != set(slide_ids):
            raise ValueError("prepared page IDs do not match deck order")
        plans_by_slide = {sid: read_json(Path(by_id[sid]["composition_plan"])) for sid in slide_ids}
        content_unit_fit = {"status": "completed", "api_call_count": 0, "records": [
            record for sid in slide_ids for record in by_id[sid]["content_unit_fit"].get("records", [])]}
        critic_log = {"status": "completed", "mode": "page_owner_reviews",
            "api_call_count": sum(by_id[sid]["critic"].get("api_call_count", 0) for sid in slide_ids),
            "primary_call_budget": len(slide_ids), "additional_judge_calls": 0,
            "candidate_batch_render_count": sum(by_id[sid]["critic"].get("candidate_batch_render_count", 0) for sid in slide_ids),
            "slides": [record for sid in slide_ids for record in by_id[sid]["critic"].get("slides", [])]}
        write_json(output_dir / "content_unit_fit.json", content_unit_fit)
        write_json(output_dir / "critic_results.json", critic_log)

    compilation_slides = []
    for slide_id in slide_ids:
        slide_dir = output_dir / slide_id
        plan = plans_by_slide[slide_id]
        if prepared_pages is None:
            text_placement.apply_text_contrast_solver(plan)
        slide_dir.mkdir(parents=True, exist_ok=True)
        plan_path = slide_dir / "composition_plan.json"
        write_json(plan_path, plan)
        compilation_slides.append(
            {
                "slide_id": slide_id,
                "plan": str(plan_path.resolve()),
            }
        )
    compilation_path = output_dir / "deck_compilation.json"
    write_json(compilation_path, {"schema_version": "1.0", "slides": compilation_slides})
    pptx_path = output_dir / "deck.pptx"
    text_fit_path = output_dir / "text_fit_report.json"
    _compile_deck(compilation_path, pptx_path, text_fit_path)
    pdf_path = output_dir / "deck.pdf"
    rendered_dir = output_dir / "rendered_slides"
    _render_and_validate(pptx_path, pdf_path, rendered_dir, len(slide_ids))
    rendered_text = inspect_pdf_text(pdf_path, plans_by_slide)
    render_quality = read_json(rendered_dir / "render_quality.json")
    render_quality["text"] = rendered_text
    render_quality["warnings"].extend(rendered_text["warnings"])
    render_quality["status"] = "degraded" if render_quality["warnings"] else "strict_pass"
    write_json(rendered_dir / "render_quality.json", render_quality)
    text_fit = read_json(text_fit_path)
    _attach_text_fit_audit(plans_by_slide, text_fit)
    _copy_rendered_previews(rendered_dir, output_dir, slide_ids)
    for slide_id in slide_ids:
        slide_dir = output_dir / slide_id
        write_json(slide_dir / "composition_plan.json", plans_by_slide[slide_id])
        text_placement.write_report(plans_by_slide[slide_id], slide_dir / "report.txt")
    montage_path = output_dir / "deck_montage.png"
    _create_montage(rendered_dir, montage_path)
    combined_background = build_combination_background_deck(
        output_dir=output_dir,
        step2_dir=step2_dir,
        final_background_dir=final_background_dir,
        slide_ids=slide_ids,
        plans_by_slide=plans_by_slide,
    )
    unresolved_overflow = {
        slide["slide_id"]: slide["overflow_region_ids"]
        for slide in text_fit["slides"]
        if slide["overflow_region_ids"]
    }
    geometry_quality = {
        slide_id: _geometry_quality_findings(plan)
        for slide_id, plan in plans_by_slide.items()
    }
    quality_warnings = [
        {"slide_id": slide_id, "geometry_findings": findings}
        for slide_id, findings in geometry_quality.items() if findings
    ]
    quality_warnings.extend(
        {"slide_id": slide_id, "quality_warnings": plan["quality_warnings"]}
        for slide_id, plan in plans_by_slide.items() if plan.get("quality_warnings")
    )
    quality_warnings.extend(_critic_quality_warnings(critic_log))
    render_quality_path = rendered_dir / "render_quality.json"
    if render_quality_path.is_file():
        render_quality = read_json(render_quality_path)
        if render_quality["warnings"]:
            quality_warnings.append({"render_quality": render_quality["warnings"]})
    manifest = {
        "stage": "reused_topology_bound_text_assembly",
        "status": "succeeded",
        "quality_status": "degraded" if unresolved_overflow or quality_warnings else "strict_pass",
        "warnings": quality_warnings + (
            [{"unresolved_text_overflow": unresolved_overflow}]
            if unresolved_overflow
            else []
        ),
        "source": str(REUSED_TEXT_PLACEMENT.resolve()),
        "slide_count": len(compilation_slides),
        "output_dir": str(output_dir.resolve()),
        "vlm_model": config.chat_model,
        "mock_vlm": mock_vlm,
        "rendered_text_quality": rendered_text,
        "content_unit_fit": content_unit_fit,
        "critic_rounds": critic_log,
        "slides": compilation_slides,
        "artifacts": {
            "pptx": str(pptx_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "rendered_slides": str(rendered_dir.resolve()),
            "montage": str(montage_path.resolve()),
            "text_fit_report": str(text_fit_path.resolve()),
        },
        "combined_background": combined_background,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def _prepare_assembly_pages(
    *, region_content_dir: Path, step2_dir: Path, topology_dir: Path,
    final_background_dir: Path, output_dir: Path, slide_ids: list[str],
    config: PipelineConfig, mock_vlm: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    text_placement = _load_text_placement()
    text_placement.VLM_MODEL = config.chat_model
    text_placement.VLM_ENV_PATH = config.env_path
    client = None
    if not mock_vlm:
        client = ApiMartClient(
            api_root=config.api_root,
            api_key_env=config.api_key_env,
            env_path=config.env_path,
            timeout_seconds=config.timeout_seconds,
            poll_initial_delay_seconds=config.poll_initial_delay_seconds,
            poll_interval_seconds=config.poll_interval_seconds,
            poll_timeout_seconds=config.poll_timeout_seconds,
        )
    policy_path = region_content_dir / "text_presentation_policy.json"
    text_policy = read_json(policy_path) if policy_path.is_file() else {
        "requested": "light",
        "resolved": "light",
        "holder_family_by_role": {},
        "fallbacks": [{"reason": "legacy_region_content_without_policy"}],
    }
    plans_by_slide: dict[str, dict[str, Any]] = {}
    for slide_id in slide_ids:
        slide_dir = output_dir / slide_id
        slide_dir.mkdir(parents=True, exist_ok=True)
        region_content = read_json(
            region_content_dir / "slides" / slide_id / "region_content_plan.json"
        )
        inventory_path = region_content_dir / "slides" / slide_id / "region_inventory.json"
        inventory = read_json(inventory_path) if inventory_path.is_file() else []
        case = {
            "schema_version": "2.0",
            "slide_id": slide_id,
            "background_path": str(
                (final_background_dir / slide_id / "background.png").resolve()
            ),
            "topology_path": str(
                (
                    topology_dir
                    / "calibrated_topology"
                    / slide_id
                    / "calibrated_topology.json"
                ).resolve()
            ),
            "layout_path": str(
                (
                    topology_dir
                    / "postprocessed"
                    / "layouts"
                    / f"{slide_id}.json"
                ).resolve()
            ),
            "asset_manifest_path": str((step2_dir / "asset_manifest.json").resolve()),
            "mode_policy": {
                "color_complexity_threshold": 5.5,
                "luminance_std_threshold": 9.0,
                "title_background_complexity_threshold": 30.0,
                "native_shape_opacity": 0.90,
                "entropy_bins": 0,
            },
            "text_presentation_policy": text_policy,
            "font_roles": _font_roles(config),
            "content": _adapt_region_content(region_content, inventory),
        }
        case_path = slide_dir / "case.json"
        write_json(case_path, case)
        plan = text_placement.build_plan(case_path)
        topology = read_json(Path(case["topology_path"]))
        layout = read_json(Path(case["layout_path"]))
        if plan["binding"]["unplaced_content_ids"]:
            _restore_unplaced_content(plan, case, text_placement)
        _configure_visual_protection(plan, topology, layout)
        plans_by_slide[slide_id] = plan

    content_unit_fit = _run_content_unit_fit(
        output_dir=output_dir,
        slide_ids=slide_ids,
        plans_by_slide=plans_by_slide,
        text_placement=text_placement,
    )
    for slide_id in slide_ids:
        plans_by_slide[slide_id] = text_placement.freeze_content_units(
            plans_by_slide[slide_id]
        )
    work_dir = output_dir / "refinement"
    work_dir.mkdir(parents=True, exist_ok=True)
    base_rendered = _write_round(
        output_dir=work_dir,
        round_name="base",
        slide_ids=slide_ids,
        plans_by_slide=plans_by_slide,
    )
    content_unit_fit["unresolved_after_fit"] = {
        slide_id: _overflow_region_ids(plans_by_slide[slide_id])
        for slide_id in slide_ids
        if _overflow_region_ids(plans_by_slide[slide_id])
    }
    write_json(output_dir / "content_unit_fit.json", content_unit_fit)
    if mock_vlm:
        critic_log = {
            "status": "skipped",
            "reason": "mock_vlm",
            "api_call_count": 0,
        }
    else:
        assert client is not None
        critic_log = _run_critic_rounds(
            output_dir=output_dir,
            work_dir=work_dir,
            slide_ids=slide_ids,
            plans_by_slide=plans_by_slide,
            initial_rendered=base_rendered,
            text_placement=text_placement,
            client=client,
            max_tokens=config.vlm_max_tokens,
            deck_plan=read_json(step2_dir / "deck_plan.json"),
        )

    return plans_by_slide, content_unit_fit, critic_log


def prepare_page_assembly(
    *, region_content_dir: Path, step2_dir: Path, topology_dir: Path,
    final_background_dir: Path, output_dir: Path, slide_id: str,
    config: PipelineConfig, mock_vlm: bool = False,
) -> dict[str, Any]:
    """Finish one page, including its actual native render and bounded review."""
    plans, fit, critic = _prepare_assembly_pages(
        region_content_dir=region_content_dir, step2_dir=step2_dir,
        topology_dir=topology_dir, final_background_dir=final_background_dir,
        output_dir=output_dir, slide_ids=[slide_id], config=config, mock_vlm=mock_vlm,
    )
    placement = _load_text_placement()
    placement.apply_text_contrast_solver(plans[slide_id])
    previews = _write_round(output_dir=output_dir, round_name="accepted",
                            slide_ids=[slide_id], plans_by_slide=plans)
    plan_path = output_dir / "composition_plan.json"
    write_json(plan_path, plans[slide_id])
    warnings = [*_critic_quality_warnings(critic), *plans[slide_id].get("quality_warnings", [])]
    pdf_audit = plans[slide_id].get("rendered_text_audit", {})
    warnings.extend(pdf_audit.get("warnings", []))
    overflows = _overflow_region_ids(plans[slide_id])
    if overflows:
        warnings.append({"unresolved_text_overflow": overflows})
    result = {"slide_id": slide_id, "status": "succeeded",
              "quality_status": "degraded" if warnings else "strict_pass", "warnings": warnings,
              "composition_plan": str(plan_path.resolve()), "preview": str(previews[slide_id].resolve()),
              "content_unit_fit": fit, "critic": critic, "rendered_text_quality": pdf_audit}
    write_json(output_dir / "page_result.json", result)
    return result


def build_combination_background_deck(
    *,
    output_dir: Path,
    step2_dir: Path,
    final_background_dir: Path,
    slide_ids: list[str],
    plans_by_slide: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compile a companion deck with the original background and movable clusters."""
    combined_dir = output_dir / "combined_background"
    combined_dir.mkdir(parents=True, exist_ok=True)
    compilation_slides = []
    decompositions = []
    for slide_id in slide_ids:
        slide_dir = combined_dir / slide_id
        decomposition = decompose_background(
            canonical_path=step2_dir / "canonical_background.png",
            final_path=final_background_dir / slide_id / "background.png",
            region_coordinates_path=(
                final_background_dir / slide_id / "region_coordinates.json"
            ),
            output_dir=slide_dir / "decomposition",
        )
        plan = copy.deepcopy(plans_by_slide[slide_id])
        plan["background_path"] = str(
            (step2_dir / "canonical_background.png").resolve()
        )
        plan["background_objects"] = [
            {
                "module_id": module["module_id"],
                "asset_path": module["asset_path"],
                "bbox_px": module["bbox_px"],
                "region_label": module["region_label"],
            }
            for module in decomposition["modules"]
        ]
        plan_path = slide_dir / "composition_plan.json"
        write_json(plan_path, plan)
        compilation_slides.append(
            {"slide_id": slide_id, "plan": str(plan_path.resolve())}
        )
        decompositions.append(
            {
                "slide_id": slide_id,
                "decomposition": str(
                    (slide_dir / "decomposition" / "decomposition.json").resolve()
                ),
                "module_count": decomposition["module_count"],
            }
        )

    compilation_path = combined_dir / "deck_compilation.json"
    write_json(compilation_path, {"schema_version": "1.0", "slides": compilation_slides})
    pptx_path = combined_dir / "deck.pptx"
    text_fit_path = combined_dir / "text_fit_report.json"
    _compile_deck(compilation_path, pptx_path, text_fit_path)
    pdf_path = combined_dir / "deck.pdf"
    rendered_dir = combined_dir / "rendered_slides"
    _render_and_validate(pptx_path, pdf_path, rendered_dir, len(slide_ids))
    _copy_rendered_previews(rendered_dir, combined_dir, slide_ids)
    montage_path = combined_dir / "deck_montage.png"
    _create_montage(rendered_dir, montage_path)
    result = {
        "stage": "local_combination_background_assembly",
        "status": "succeeded",
        "api_call_count": 0,
        "slide_count": len(slide_ids),
        "module_count": sum(item["module_count"] for item in decompositions),
        "slides": decompositions,
        "artifacts": {
            "pptx": str(pptx_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "rendered_slides": str(rendered_dir.resolve()),
            "montage": str(montage_path.resolve()),
            "text_fit_report": str(text_fit_path.resolve()),
        },
    }
    write_json(combined_dir / "run_manifest.json", result)
    return result


def _write_round(
    *,
    output_dir: Path,
    round_name: str,
    slide_ids: list[str],
    plans_by_slide: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    round_dir = output_dir / "rounds" / round_name
    compilation_slides = []
    for slide_id in slide_ids:
        slide_dir = round_dir / slide_id
        slide_dir.mkdir(parents=True, exist_ok=True)
        plan_path = slide_dir / "composition_plan.json"
        write_json(plan_path, plans_by_slide[slide_id])
        compilation_slides.append({"slide_id": slide_id, "plan": str(plan_path.resolve())})
    compilation_path = round_dir / "deck_compilation.json"
    write_json(
        compilation_path,
        {"schema_version": "1.0", "slides": compilation_slides},
    )
    pptx_path = round_dir / "deck.pptx"
    audit_path = round_dir / "text_fit_report.json"
    _compile_deck(compilation_path, pptx_path, audit_path)
    rendered_dir = round_dir / "rendered_slides"
    _render_and_validate(
        pptx_path,
        round_dir / "deck.pdf",
        rendered_dir,
        len(slide_ids),
    )
    _attach_text_fit_audit(plans_by_slide, read_json(audit_path))
    pdf_audit = inspect_pdf_text(round_dir / "deck.pdf", {sid: plans_by_slide[sid] for sid in slide_ids})
    for report in pdf_audit["slides"]:
        plans_by_slide[report["slide_id"]]["rendered_text_audit"] = report
    rendered_paths = {
        int(path.stem.rsplit("-", 1)[1]): path for path in rendered_dir.glob("slide-*.png")
    }
    preview_paths = []
    for page_number, slide_id in enumerate(slide_ids, start=1):
        plan_path = round_dir / slide_id / "composition_plan.json"
        write_json(plan_path, plans_by_slide[slide_id])
        preview_path = round_dir / slide_id / "preview.png"
        rendered_path = rendered_paths.get(page_number)
        if rendered_path is None:
            _load_text_placement().render_preview(plans_by_slide[slide_id], preview_path)
            plans_by_slide[slide_id].setdefault("quality_warnings", []).append({
                "stage": "render", "warning": "missing_office_preview_local_preview_used"
            })
        else:
            shutil.copyfile(rendered_path, preview_path)
        preview_paths.append(preview_path)
    _create_montage_from_paths(
        preview_paths,
        round_dir / "deck_montage.png",
        cell_width=384,
        cell_height=216,
    )
    return dict(zip(slide_ids, preview_paths, strict=True))


def _run_content_unit_fit(
    *,
    output_dir: Path,
    slide_ids: list[str],
    plans_by_slide: dict[str, dict[str, Any]],
    text_placement: Any,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for slide_id in slide_ids:
        plan = plans_by_slide[slide_id]
        for element in plan["elements"]:
            requested = element["font"]["face"]
            element["font"]["face"] = resolve_deck_font(requested)
            if requested != element["font"]["face"]:
                element["font"]["requested_face"] = requested
        for role in plan.get("font_roles", {}).values():
            role["face"] = resolve_deck_font(role["face"])
        _repair_plan_bounds(plan, text_placement)
        for index in range(len(plan["elements"])):
            element = plan["elements"][index]
            if element["presentation"]["mode"] == "raster_holder_text":
                measured = _measure_registration_states(plan, [{"element": element, "operation": {"op": "capacity_check"}}], text_placement)[0]
                # A proportional image fit must not reduce the reserved text
                # slot. Reframe the same ornament before reducing its copy.
                if element["role"] == "title" or not measured["fits"]:
                    _restore_reserved_holder(plan, index, text_placement, output_dir)
            record = _fit_element_deterministically(plan, index, text_placement)
            repaired = _place_unresolved_unit(plan, index, text_placement, output_dir)
            if repaired:
                record = {**record, "status": "fit", "placement_repair": repaired,
                          "final_bbox_px": list(plan["elements"][index]["bbox_px"])}
            records.append({"slide_id": slide_id, **record})
        records.extend(
            {"slide_id": slide_id, **record}
            for record in _harmonize_registered_groups(plan, text_placement)
        )
        for group in plan.get("groups", []):
            repaired = _place_unresolved_group(plan, group, text_placement, output_dir)
            if repaired:
                records.append({"slide_id": slide_id, **repaired})
        violation = text_placement._validate_group_invariants(plan)
        if violation:
            # Keep the usable content and record the quality limitation. This
            # check is diagnostic, never a reason to discard an entire deck.
            plan.setdefault("quality_warnings", []).append(
                {"stage": "content_unit_fit", "invariant": violation}
            )
        plans_by_slide[slide_id] = plan
    result = {
        "status": "completed",
        "overflow_owner": "content_unit_fit",
        "algorithm": "horizontal_4px_then_font_1pt_then_symmetric_vertical_4px",
        "api_call_count": 0,
        "records": records,
    }
    write_json(output_dir / "content_unit_fit.json", result)
    return result


def _configure_visual_protection(plan: dict, topology: dict, layout: dict) -> None:
    """Protect scene pixels outside the IR's explicitly reserved text slots.

    A scene bounding box is a generation extent, not a foreground segmentation.
    Subtracting planned text components retains that extent everywhere else and
    lets the deterministic renderer use exactly the space already allocated to
    editable copy. This does not grant permission to expand into other artwork.
    """
    by_id = {e["region_id"]: e for e in topology.get("entities", [])}
    reservations = [list(e.get("original_topology_bbox_px", e.get("topology_bbox_px", e["bbox_px"])))
                    for e in plan["elements"]]
    protected = []
    overlaps = []
    for entity in layout.get("entities", []):
        rid = entity["region_id"]
        if by_id.get(rid, {}).get("canonical_role") != "visual":
            continue
        original = entity["postprocessed_bbox_px"]
        pieces = [list(original)]
        for element, reserved in zip(plan["elements"], reservations):
            if _bbox_intersection(original, reserved) > 1:
                overlaps.append(f"{element['region_id']}::{rid}")
            pieces = [part for piece in pieces for part in _subtract_bbox(piece, reserved)]
        protected.extend({"region_id": rid, "component_id": by_id[rid].get("parent_component_id"),
                          "bbox_px": piece} for piece in pieces)
    plan["protected_visual_regions"] = protected
    plan["planned_visual_overlaps"] = overlaps
    plan["intentional_visual_overlaps"] = []


def _subtract_bbox(box: list[float], reserved: list[float]) -> list[list[float]]:
    x, y, w, h = box
    left, top = max(x, reserved[0]), max(y, reserved[1])
    right, bottom = min(x + w, reserved[0] + reserved[2]), min(y + h, reserved[1] + reserved[3])
    if right <= left or bottom <= top:
        return [box]
    return [p for p in [[x, y, w, top - y], [x, bottom, w, y + h - bottom],
                        [x, top, left - x, bottom - top], [right, top, x + w - right, bottom - top]]
            if p[2] > 1 and p[3] > 1]


def _restore_reserved_holder(plan: dict, index: int, placement: Any, output_dir: Path) -> bool:
    from .decorative_surface import reframe_decorative_surface
    source = plan["elements"][index]
    candidate = copy.deepcopy(source)
    candidate["bbox_px"] = list(source.get("topology_bbox_px", source["bbox_px"]))
    if _fit_candidate_issues(plan, index, candidate):
        return False
    safe = candidate["presentation"].get("text_safe_rect")
    asset_path = Path(candidate["presentation"].get("asset_path") or "")
    if not safe or not asset_path.is_file():
        return False
    target = {"x": .04, "y": .10, "width": .92, "height": .80}
    surface = reframe_decorative_surface(asset_path,
        output_dir / "decorated_holders" / plan["slide_id"] / f"{source['region_id']}__{candidate['presentation']['asset_id']}__reserved.png", safe, target)
    candidate["presentation"]["quiet_frame_path"] = surface["frame_path"]
    candidate["text_backing"] = {"mode": "themed_quiet_interior", "fill_color": surface["fill_color"],
        "bbox_px": list(candidate["bbox_px"]), "normalized_polygon": surface["normalized_polygon"],
        "text_safe_rect": surface["text_safe_rect"]}
    placement._refresh_holder_binding(candidate)
    placement._refresh_content_unit(candidate)
    plan["elements"][index] = candidate
    plan.setdefault("repairs", []).append({"region_id": source["region_id"],
        "op": "reframe_reserved_decorative_holder", "from_bbox_px": source["bbox_px"],
        "to_bbox_px": candidate["bbox_px"]})
    return True


def _free_cells(plan: dict, excluded_ids: set[str]) -> list[list[float]]:
    obstacles = [e["bbox_px"] for e in plan["elements"] if e["region_id"] not in excluded_ids]
    obstacles += [e["bbox_px"] for e in plan.get("protected_visual_regions", [])]
    width, height = plan["canvas_px"]["width"], plan["canvas_px"]["height"]
    gap, margin = 12., 32.
    top = max([margin] + [e["bbox_px"][1] + e["bbox_px"][3] + gap
                         for e in plan["elements"] if e["role"] == "title"])
    xs = sorted({margin, *[b[0] + b[2] + gap for b in obstacles]})
    ys = sorted({top, *[b[1] + b[3] + gap for b in obstacles]})
    rights = sorted({width - margin, *[b[0] - gap for b in obstacles]})
    bottoms = sorted({height - margin, *[b[1] - gap for b in obstacles]})
    cells = []
    for x in xs:
        for y in ys:
            for right in rights:
                for bottom in bottoms:
                    box = [x, y, right - x, bottom - y]
                    if x < margin or y < top or right > width - margin or bottom > height - margin or box[2] < 120 or box[3] < 48:
                        continue
                    if not any(_bbox_intersection(box, b) > 1 for b in obstacles):
                        cells.append(box)
    return cells


def _place_unresolved_group(plan: dict, group: dict, placement: Any, output_dir: Path) -> dict | None:
    """Allocate one free cell to a whole peer group before bindings freeze."""
    ids = set(group["member_region_ids"])
    members = [e for e in plan["elements"] if e["region_id"] in ids]
    if len(members) < 2 or not any(e.get("registration", {}).get("status") == "unresolved" for e in members):
        return None
    cells = _free_cells(plan, ids)
    if not cells:
        return None
    axis = 0 if group["arrangement"] == "row" else 1
    if group["arrangement"] == "freeform":
        centers = [placement.bbox_center(e["bbox_px"]) for e in members]
        axis = 0 if max(c[0] for c in centers) - min(c[0] for c in centers) > max(c[1] for c in centers) - min(c[1] for c in centers) else 1
    ordered = sorted(members, key=lambda e: placement.bbox_center(e["bbox_px"])[axis])
    origin = [sum(placement.bbox_center(e["bbox_px"])[a] for e in members) / len(members) for a in (0, 1)]
    original_extent = [max(e["bbox_px"][a] + e["bbox_px"][a + 2] for e in members)
                       - min(e["bbox_px"][a] for e in members) for a in (0, 1)]
    cells.sort(key=lambda b: math.hypot(placement.bbox_center(b)[0] - origin[0], placement.bbox_center(b)[1] - origin[1]))
    templates = []
    for source in ordered:
        candidate = copy.deepcopy(source)
        if candidate["presentation"]["mode"] == "raster_holder_text":
            # Prepare the same ornament once; candidate placements only resize
            # its outer rectangle and keep its editable quiet interior attached.
            temporary_plan = {**plan, "elements": [candidate], "protected_visual_regions": [], "repairs": []}
            candidate.setdefault("original_topology_bbox_px", list(candidate.get("topology_bbox_px", candidate["bbox_px"])))
            candidate["topology_bbox_px"] = list(candidate["bbox_px"])
            _restore_reserved_holder(temporary_plan, 0, placement, output_dir)
            candidate = temporary_plan["elements"][0]
        templates.append(candidate)
    trials, states = [], []
    for cell in cells[:32]:
        for growth in (1., 1.5, 2., 3., 4.):
            extent = [min(cell[a + 2], original_extent[a] * growth) for a in (0, 1)]
            allocated = [min(max(origin[a] - extent[a] / 2, cell[a]), cell[a] + cell[a + 2] - extent[a]) for a in (0, 1)] + extent
            available = allocated[axis + 2] - 12 * (len(members) - 1)
            if available < 48 * len(members):
                continue
            cursor = allocated[axis]
            candidates = []
            for template in templates:
                candidate = copy.deepcopy(template)
                box = list(allocated)
                box[axis], box[axis + 2] = cursor, available / len(members)
                candidate["bbox_px"] = box
                placement._refresh_holder_binding(candidate)
                candidates.append(candidate)
                states.append({"element": candidate, "operation": {"op": "group_free_space_capacity"}})
                cursor += box[axis + 2] + 12
            trials.append((allocated, candidates))
    if not states:
        return None
    measurements = _measure_registration_states(plan, states, placement)
    for trial_index, (cell, candidates) in enumerate(trials):
        results = measurements[trial_index * len(members):(trial_index + 1) * len(members)]
        if not all(r["fits"] for r in results):
            continue
        for candidate, result in zip(candidates, results):
            index = next(i for i, e in enumerate(plan["elements"]) if e["region_id"] == candidate["region_id"])
            candidate["original_topology_bbox_px"] = candidate.get("original_topology_bbox_px", candidate["topology_bbox_px"])
            candidate["topology_bbox_px"] = list(candidate["bbox_px"])
            candidate["anchor_center_px"] = list(placement.bbox_center(candidate["bbox_px"]))
            candidate["registration"] = {"status": "fit", "placement": "prefreeze_group_free_space", "line_count": result["line_count"]}
            placement._refresh_content_unit(candidate)
            plan["elements"][index] = candidate
        record = {"op": "prefreeze_group_free_space", "group_id": group["group_id"], "bbox_px": cell,
                  "member_region_ids": [e["region_id"] for e in ordered]}
        plan.setdefault("repairs", []).append(record)
        return record
    return None


def _place_unresolved_unit(plan: dict[str, Any], index: int, placement: Any, output_dir: Path) -> dict | None:
    """Place an impossible singleton in existing free space before binding freezes.

    Grouped peers and declared labels stay attached. Scene pixels, text and IDs
    are unchanged; role-compatible decorated frames are selected proportionally.
    """
    source = plan["elements"][index]
    group = next((g for g in plan.get("groups", []) if source["region_id"] in g["member_region_ids"]), None)
    if source["role"] == "title" or (group and len(group["member_region_ids"]) > 1):
        return None
    if any(pair.startswith(source["region_id"] + "::") for pair in plan.get("intentional_visual_overlaps", [])):
        return None
    obstacles = [e["bbox_px"] for i, e in enumerate(plan["elements"]) if i != index]
    obstacles += [e["bbox_px"] for e in plan.get("protected_visual_regions", [])]
    collided = any(_bbox_intersection(source["bbox_px"], b) > 1 for b in obstacles)
    if source.get("registration", {}).get("status") != "unresolved" and not collided:
        return None
    width, height = plan["canvas_px"]["width"], plan["canvas_px"]["height"]
    cells = _free_cells(plan, {source["region_id"]})
    if not cells:
        return None
    center = placement.bbox_center(source["bbox_px"])
    cells.sort(key=lambda b: math.hypot(max(b[0] - center[0], center[0] - b[0] - b[2], 0),
                                       2 * max(b[1] - center[1], center[1] - b[1] - b[3], 0)) - b[2] * b[3] / (width * height))
    role_uses = {"subtitle": {"subtitle_background", "short_explanation", "comparison", "definition"},
                 "body": {"short_explanation", "comparison", "definition"},
                 "caption": {"annotation", "short_explanation", "stage_label", "takeaway", "emphasis"}}
    compatible_uses = role_uses.get(source["role"], {source["role"]})
    assets = [a for a in plan.get("available_assets", []) if a.get("category") in {"raster_holder", "subtitle_bar", "title_bar"}
              and a.get("text_safe_rect") and a.get("aspect_ratio", 0) > 0
              and (not a.get("allowed_roles") or compatible_uses.intersection(a["allowed_roles"]))]
    assets.sort(key=lambda a: a["asset_id"] != source["presentation"].get("asset_id"))
    states, seen = [], set()
    for asset in assets[:8]:
        template = copy.deepcopy(source)
        template.pop("text_backing", None)
        template["presentation"] = {**template["presentation"], "mode": "raster_holder_text",
            "asset_id": asset["asset_id"], "asset_path": asset["path"], "asset_category": asset["category"],
            "text_safe_rect": asset["text_safe_rect"], "native_aspect_ratio": asset["aspect_ratio"]}
        template["presentation"].pop("quiet_frame_path", None)
        # Capacity repair keeps the same themed frame; center restoration is a
        # deterministic candidate, not a request for another image or critic.
        placement.restore_quiet_interior(template, output_dir / "decorated_holders" / plan["slide_id"])
        for cell in cells[:16]:
            for scale in (.6, .8, 1.):
                w = cell[2] * scale
                h = w / asset["aspect_ratio"]
                if h > cell[3]:
                    h, w = cell[3], cell[3] * asset["aspect_ratio"]
                x = min(max(source["bbox_px"][0], cell[0]), cell[0] + cell[2] - w)
                y = min(max(source["bbox_px"][1], cell[1]), cell[1] + cell[3] - h)
                box = [x, y, w, h]
                key = (asset["asset_id"], *(round(v, 3) for v in box))
                if key in seen or any(_bbox_intersection(box, b) > 1 for b in obstacles):
                    continue
                seen.add(key)
                candidate = copy.deepcopy(template)
                candidate["bbox_px"] = box
                placement._refresh_holder_binding(candidate)
                states.append({"element": candidate, "operation": {"op": "free_space_capacity"}, "cell": cell})
    if not states:
        return None
    measured = _measure_registration_states(plan, states, placement)
    eligible = [(state, result) for state, result in zip(states, measured) if result["fits"]]
    if not eligible:
        return None
    def score(pair):
        e = pair[0]["element"]
        c = placement.bbox_center(e["bbox_px"])
        return math.hypot(c[0] - center[0], 2 * (c[1] - center[1])) + e["bbox_px"][2] * e["bbox_px"][3] / (width * height) * 150
    state, measurement = min(eligible, key=score)
    candidate = state["element"]
    candidate["original_topology_bbox_px"] = source.get("original_topology_bbox_px", source.get("topology_bbox_px", source["bbox_px"]))
    candidate["topology_bbox_px"] = list(state["cell"])
    candidate["anchor_center_px"] = list(placement.bbox_center(candidate["bbox_px"]))
    candidate["registration"] = {"status": "fit", "placement": "prefreeze_free_space", "line_count": measurement["line_count"]}
    placement._refresh_content_unit(candidate)
    plan["elements"][index] = candidate
    record = {"region_id": source["region_id"], "op": "prefreeze_free_space", "from_bbox_px": source["bbox_px"], "to_bbox_px": candidate["bbox_px"], "holder_asset_id": candidate["presentation"]["asset_id"]}
    plan.setdefault("repairs", []).append(record)
    return record


def _fit_element_deterministically(
    plan: dict[str, Any], element_index: int, text_placement: Any
) -> dict[str, Any]:
    source = plan["elements"][element_index]
    topology_bbox = [
        float(value) for value in source.get("topology_bbox_px", source["bbox_px"])
    ]
    center_x, center_y = text_placement.bbox_center(topology_bbox)
    if (
        source["role"] == "title"
        and source["presentation"]["mode"] == "direct_text"
    ):
        return _fit_direct_title_horizontally(
            plan, element_index, topology_bbox, center_x, text_placement
        )
    current = copy.deepcopy(source)
    current["bbox_px"] = [
        center_x - min(float(current["bbox_px"][2]), topology_bbox[2]) / 2,
        center_y - float(current["bbox_px"][3]) / 2,
        min(float(current["bbox_px"][2]), topology_bbox[2]),
        float(current["bbox_px"][3]),
    ]
    states: list[dict[str, Any]] = []

    def add_state(phase: str, operation: dict[str, Any]) -> None:
        candidate = copy.deepcopy(current)
        text_placement._refresh_holder_binding(candidate)
        states.append(
            {
                "phase": phase,
                "operation": operation,
                "element": candidate,
            }
        )

    add_state("initial", {"op": "initial_check"})
    while current["bbox_px"][2] < topology_bbox[2] - 1e-6:
        before = float(current["bbox_px"][2])
        after = min(topology_bbox[2], before + 4.0)
        current["bbox_px"][0] = center_x - after / 2
        current["bbox_px"][2] = after
        add_state(
            "horizontal",
            {"op": "expand_horizontal", "from_width_px": before, "to_width_px": after},
        )

    minimum_pt = float(current["font"]["minimum_pt"])
    while float(current["font"]["size_pt"]) > minimum_pt + 1e-6:
        before = float(current["font"]["size_pt"])
        after = max(minimum_pt, before - 1.0)
        current["font"]["size_pt"] = after
        add_state(
            "font",
            {"op": "decrease_font", "from_size_pt": before, "to_size_pt": after},
        )

    canvas_height = float(plan["canvas_px"]["height"])
    max_symmetric_height = 2 * min(center_y, canvas_height - center_y)
    while current["bbox_px"][3] < max_symmetric_height - 1e-6:
        before = float(current["bbox_px"][3])
        after = min(max_symmetric_height, before + 4.0)
        current["bbox_px"][1] = center_y - after / 2
        current["bbox_px"][3] = after
        add_state(
            "vertical",
            {
                "op": "expand_vertical_symmetric",
                "from_height_px": before,
                "to_height_px": after,
                "top_delta_px": -(after - before) / 2,
                "bottom_delta_px": (after - before) / 2,
            },
        )

    if not any(not _fit_candidate_issues(plan, element_index, state["element"]) for state in states):
        fallback = copy.deepcopy(source)
        fallback["font"]["size_pt"] = minimum_pt
        text_placement._refresh_holder_binding(fallback)
        states.append({"phase": "original_geometry", "operation": {"op": "retain_original_geometry"}, "element": fallback})
    measurements = _measure_registration_states(plan, states, text_placement)
    safe_indices = [
        index for index, state in enumerate(states)
        if not _fit_candidate_issues(plan, element_index, state["element"])
    ]
    selected_index = safe_indices[-1] if safe_indices else 0
    fit_status = "unresolved"
    for index, (state, measurement) in enumerate(zip(states, measurements, strict=True)):
        one_line_required = source["role"] == "title" and state["phase"] != "vertical"
        if index in safe_indices and measurement["fits"] and (
            not one_line_required or int(measurement["line_count"]) == 1
        ):
            selected_index = index
            fit_status = "fit"
            break

    selected = copy.deepcopy(states[selected_index]["element"])
    checks = []
    for state, measurement in zip(
        states[: selected_index + 1], measurements[: selected_index + 1], strict=True
    ):
        checks.append(
            {
                **state["operation"],
                "fits": bool(measurement["fits"]),
                "line_count": int(measurement["line_count"]),
                "required_height_in": round(float(measurement["required_height_in"]), 4),
                "available_height_in": round(float(measurement["available_height_in"]), 4),
                "geometry_issues": _fit_candidate_issues(plan, element_index, state["element"]),
            }
        )
    selected["registration"] = {
        "status": fit_status,
        "topology_bbox_px": topology_bbox,
        "topology_center_px": [center_x, center_y],
        "atomic_check_count": len(checks),
        "final_font_size_pt": selected["font"]["size_pt"],
    }
    text_placement._refresh_holder_binding(selected)
    text_placement._refresh_content_unit(selected)
    plan["elements"][element_index] = selected
    return {
        "region_id": source["region_id"],
        "owner": "content_unit_fit",
        "status": fit_status,
        "holder_asset_id": selected["presentation"].get("asset_id"),
        "holder_category": selected["presentation"].get("asset_category"),
        "topology_bbox_px": topology_bbox,
        "final_bbox_px": selected["bbox_px"],
        "final_font_size_pt": selected["font"]["size_pt"],
        "atomic_checks": checks,
    }


def _fit_direct_title_horizontally(
    plan: dict[str, Any],
    element_index: int,
    topology_bbox: list[float],
    center_x: float,
    text_placement: Any,
) -> dict[str, Any]:
    source = plan["elements"][element_index]
    initial_state = {
        "phase": "horizontal_title_fit",
        "operation": {"op": "measure_title_width"},
        "element": copy.deepcopy(source),
    }
    initial_measurement = _measure_registration_states(
        plan, [initial_state], text_placement
    )[0]
    selected = copy.deepcopy(source)
    original_x, original_y, original_width, original_height = (
        float(value) for value in source["bbox_px"]
    )
    # Keep the available planning width; measuring a font's exact edge and then
    # narrowing the native frame to that edge was unstable across renderers.
    fitted_width = max(original_width, float(topology_bbox[2]))
    selected["bbox_px"] = [
        center_x - fitted_width / 2,
        original_y,
        fitted_width,
        original_height,
    ]
    text_placement._refresh_holder_binding(selected)
    final_state = {
        "phase": "horizontal_title_fit",
        "operation": {
            "op": "retain_planning_title_width",
            "from_x_px": original_x,
            "from_width_px": original_width,
            "to_x_px": selected["bbox_px"][0],
            "to_width_px": fitted_width,
        },
        "element": selected,
    }
    measurement = _measure_registration_states(plan, [final_state], text_placement)[0]
    geometry_issues = _fit_candidate_issues(plan, element_index, selected)
    if geometry_issues:
        selected = copy.deepcopy(source)
        final_state["element"] = selected
        final_state["operation"] = {
            "op": "retain_title_geometry", "rejected_geometry_issues": geometry_issues,
        }
        measurement = _measure_registration_states(plan, [final_state], text_placement)[0]
    fit_status = "fit" if (
        measurement["fits"] and int(measurement["line_count"]) == 1
        and not _fit_candidate_issues(plan, element_index, selected)
    ) else "unresolved"
    selected["registration"] = {
        "status": fit_status,
        "topology_bbox_px": topology_bbox,
        "topology_center_px": list(text_placement.bbox_center(topology_bbox)),
        "atomic_check_count": 1,
        "final_font_size_pt": selected["font"]["size_pt"],
        "title_text_box_mode": "original_geometry_fallback" if geometry_issues else "full_planning_width",
    }
    text_placement._refresh_content_unit(selected)
    plan["elements"][element_index] = selected
    check = {
        **final_state["operation"],
        "fits": fit_status == "fit",
        "line_count": int(measurement["line_count"]),
        "required_height_in": round(float(measurement["required_height_in"]), 4),
        "available_height_in": round(float(measurement["available_height_in"]), 4),
    }
    return {
        "region_id": source["region_id"],
        "owner": "content_unit_fit",
        "status": fit_status,
        "holder_asset_id": None,
        "holder_category": None,
        "topology_bbox_px": topology_bbox,
        "final_bbox_px": selected["bbox_px"],
        "final_font_size_pt": selected["font"]["size_pt"],
        "atomic_checks": [check],
    }


def _measure_registration_states(
    plan: dict[str, Any], states: list[dict[str, Any]], text_placement: Any
) -> list[dict[str, Any]]:
    request_states = []
    for index, state in enumerate(states):
        element = state["element"]
        request_states.append(
            {
                "state_id": str(index),
                "text": element["text"],
                "bullet_items": element.get("bullet_items", []),
                "text_bbox_px": text_placement.element_text_rect_px(element),
                "font_face": element["font"]["face"],
                "font_weight": element["font"]["weight"],
                "font_size_pt": element["font"]["size_pt"],
                "measure_tight_width": state["operation"]["op"]
                == "measure_title_width",
            }
        )
    request = {"canvas_px": plan["canvas_px"], "states": request_states}
    with tempfile.TemporaryDirectory(prefix="rl4ppt-text-measure-") as temporary:
        request_path = Path(temporary) / "request.json"
        request_path.write_text(
            json.dumps(request, ensure_ascii=False), encoding="utf-8"
        )
        result = subprocess.run(
            ["node", str(TEXT_MEASURER), str(request_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    payload = json.loads(result.stdout)
    measurements = payload.get("results", [])
    if len(measurements) != len(states):
        raise ValueError("text measurer returned an incomplete result")
    return measurements


def _harmonize_registered_groups(
    plan: dict[str, Any], text_placement: Any
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    element_by_region = {
        element["region_id"]: element for element in plan["elements"]
    }
    for group in plan["groups"]:
        members = [
            element_by_region[region_id]
            for region_id in group["member_region_ids"]
            if region_id in element_by_region
        ]
        if len(members) < 2:
            continue
        sizes = {
            (round(member["bbox_px"][2], 6), round(member["bbox_px"][3], 6))
            for member in members
        }
        if group.get("equal_size") and len(sizes) != 1:
            group["equal_size"] = False
            group["size_consistency"] = "approximately_consistent"
        common_font = min(float(member["font"]["size_pt"]) for member in members)
        for member in members:
            if float(member["font"]["size_pt"]) <= common_font + 1e-6:
                continue
            states: list[dict[str, Any]] = []
            current = copy.deepcopy(member)
            while float(current["font"]["size_pt"]) > common_font + 1e-6:
                before = float(current["font"]["size_pt"])
                after = max(common_font, before - 1.0)
                current["font"]["size_pt"] = after
                text_placement._refresh_holder_binding(current)
                states.append(
                    {
                        "phase": "group_font",
                        "operation": {
                            "op": "decrease_group_font",
                            "from_size_pt": before,
                            "to_size_pt": after,
                        },
                        "element": copy.deepcopy(current),
                    }
                )
            measurements = _measure_registration_states(plan, states, text_placement)
            checks = []
            for state, measurement in zip(states, measurements, strict=True):
                checks.append(
                    {
                        **state["operation"],
                        "fits": bool(measurement["fits"]),
                        "line_count": int(measurement["line_count"]),
                    }
                )
            member.update(states[-1]["element"])
            member.setdefault("registration", {})["final_font_size_pt"] = common_font
            text_placement._refresh_holder_binding(member)
            text_placement._refresh_content_unit(member)
            records.append(
                {
                    "region_id": member["region_id"],
                    "owner": "group_font_harmonization",
                    "status": "fit" if all(item["fits"] for item in checks) else "unresolved",
                    "group_id": group["group_id"],
                    "final_font_size_pt": common_font,
                    "atomic_checks": checks,
                }
            )
    return records


def _run_critic_rounds(
    *, output_dir: Path, work_dir: Path, slide_ids: list[str],
    plans_by_slide: dict[str, dict[str, Any]], initial_rendered: dict[str, Path],
    text_placement: Any, client: ApiMartClient, max_tokens: int, deck_plan: dict[str, Any],
) -> dict[str, Any]:
    """One review per slide, one batch candidate render, deterministic acceptance.

    The critic cannot rewrite frozen copy, create annotations, change assets, or
    request another judge. Existing topology/copy/render checks own acceptance.
    """
    claims = {item["slide_id"]: item.get("rough_message", "") for item in deck_plan.get("slides", [])}
    calls_before = client.call_count
    baselines = {sid: copy.deepcopy(plans_by_slide[sid]) for sid in slide_ids}
    candidates = {}
    records = []
    for slide_id in slide_ids:
        plan = baselines[slide_id]
        context = {
            "slide_id": slide_id, "claim": claims.get(slide_id, ""),
            "canvas_px": plan["canvas_px"], "visuals": _visual_context(plan),
            "deterministic_issues": _geometry_quality_findings(plan),
            "rendered_text": plan.get("rendered_text_audit", {}),
            "units": [{"region_id": e["region_id"], "text": e["text"],
                       "bbox_px": e["bbox_px"], "font": e["font"],
                       "legal_moves": _topology_move_ranges(plan, e["region_id"])}
                      for e in plan["elements"]],
        }
        prompt = """Review this exported slide once. Prioritize text clipping/readability,
holders covering faces or key objects, and missing or misleading visual steps
relative to the claim. Report visible issues even when no allowed patch can fix
them; an empty legal move range does not mean the slide has no issue.
Return JSON only:
{"issues":[{"region_id":"existing id","severity":"major or minor","reason":"brief"}],
 "patches":[{"region_id":"existing id","op":"move","axis":"x or y","position_px":0},
 {"region_id":"existing id","op":"set_font_size","size_pt":18},
 {"region_id":"existing id","op":"restore_quiet_interior"}]}
Return only necessary patches, at most five. An empty patches list is valid.
Frozen text, bullets, IDs, semantic relationships, assets and ordering cannot change.
Move only within supplied legal ranges. Keep themed holders; quiet-interior repair
retains their art. Never add decorative badges, annotations, or new content.
Do not propose a whole-slide redesign or another review round.
"""
        # A completed review is a paid artifact. Reuse it after a later local
        # render failure, and never blindly repeat an interrupted submission.
        from .page_runtime import atomic_write_json
        review_prompt = _prompt_with_context(prompt, context)
        image_path = initial_rendered[slide_id]
        digest = hashlib.sha256(review_prompt.encode("utf-8"))
        digest.update(str(getattr(text_placement, "VLM_MODEL", None)).encode("utf-8"))
        digest.update(str(min(max_tokens, 1800)).encode("ascii"))
        if image_path.is_file():
            digest.update(image_path.read_bytes())
        review_fingerprint = digest.hexdigest()
        review_path = output_dir / "critic_checkpoints" / f"{slide_id}.json"
        saved = read_json(review_path) if review_path.is_file() else {}
        if saved.get("fingerprint") == review_fingerprint and saved.get("status") == "completed":
            response, error = saved.get("response"), saved.get("error")
        else:
            if saved.get("fingerprint") == review_fingerprint and saved.get("status") == "submitting":
                raise RuntimeError(f"Review submission outcome unknown for {slide_id}; reconcile before retrying")
            atomic_write_json(review_path, {"status": "submitting", "fingerprint": review_fingerprint})
            response, error = text_placement.call_vlm_json(
                prompt=review_prompt, image_path=image_path,
                client=client, max_tokens=min(max_tokens, 1800),
            )
            atomic_write_json(review_path, {"status": "completed", "fingerprint": review_fingerprint,
                                           "response": response, "error": error})
        record = {"slide_id": slide_id, "issues": (response or {}).get("issues", []), "status": "kept_baseline"}
        if error:
            record["error"] = error
        else:
            candidate, rejected, applied = _apply_review_patches(plan, response or {}, text_placement, output_dir / "decorated_holders" / slide_id)
            record.update({"rejected_patches": rejected, "applied_patch_count": applied})
            if applied:
                candidates[slide_id] = candidate
        records.append(record)
    if candidates:
        _write_round(output_dir=work_dir, round_name="review_candidates",
                     slide_ids=list(candidates), plans_by_slide=candidates)
        for record in records:
            sid = record["slide_id"]
            if sid not in candidates:
                continue
            candidate = candidates[sid]
            reasons = _hard_rejection_reasons(baselines[sid], candidate, text_placement)
            before = {(w.get("region_id"), issue) for w in baselines[sid].get("rendered_text_audit", {}).get("warnings", []) for issue in w.get("issues", [])}
            after = {(w.get("region_id"), issue) for w in candidate.get("rendered_text_audit", {}).get("warnings", []) for issue in w.get("issues", [])}
            if after - before:
                # PDF glyph metrics remain an explicit diagnostic; they do not
                # overrule the geometry/copy/actual-font fit acceptance alone.
                record["rendered_text_warnings"] = sorted(after - before)
            if reasons:
                record["rejected"] = reasons
            else:
                plans_by_slide[sid] = candidate
                record["status"] = "accepted"
    result = {"status": "completed", "mode": "single_review_batch_render",
              "model": text_placement.configured_vlm_model(),
              "api_call_count": client.call_count - calls_before,
              "primary_call_budget": len(slide_ids), "candidate_batch_render_count": int(bool(candidates)),
              "additional_judge_calls": 0, "slides": records}
    result_path = output_dir / "critic_results.json"
    write_json(result_path, result)
    return {**result, "artifact": str(result_path.resolve())}


def _critic_quality_warnings(critic_log: dict[str, Any]) -> list[dict[str, Any]]:
    # Reporting only: findings never veto a deck or start another model round.
    warnings = []
    for review in critic_log.get("slides", []):
        issues = review.get("issues")
        if not isinstance(issues, list):
            continue
        major = [issue for issue in issues if isinstance(issue, dict) and issue.get("severity") == "major"]
        if major:
            warnings.append({"slide_id": review["slide_id"], "critic_major_findings": major})
    return warnings


def _apply_review_patches(plan: dict[str, Any], response: dict[str, Any], text_placement: Any, frame_output_dir: Path | None = None) -> tuple[dict[str, Any], list[dict], int]:
    working = copy.deepcopy(plan)
    rejected = []
    applied = 0
    patches = response.get("patches", [])
    if not isinstance(patches, list):
        return working, [{"reason": "patches_must_be_an_array"}], 0
    for patch in patches[:5]:
        try:
            if not isinstance(patch, dict) or any(k not in {"region_id", "op", "axis", "position_px", "size_pt"} for k in patch):
                raise ValueError("unknown_patch_fields")
            region = patch["region_id"]
            element = next(e for e in working["elements"] if e["region_id"] == region)
            op = patch["op"]
            if op == "move":
                candidate, outcome = _apply_topology_move(working, region, patch, text_placement)
                if outcome["status"] != "applied":
                    raise ValueError(outcome["reason"])
                working = candidate
            elif op == "set_font_size":
                size = float(patch["size_pt"])
                if not math.isfinite(size) or element["role"] == "title":
                    raise ValueError("title_size_is_deck_owned_or_nonfinite")
                group = next(g for g in working["groups"] if region in g["member_region_ids"])
                members = [e for e in working["elements"] if e["region_id"] in group["member_region_ids"]]
                if any(not e["font"]["minimum_pt"] <= size <= e["font"].get("maximum_pt", e["font"]["size_pt"]) for e in members):
                    raise ValueError("font_size_out_of_range")
                for member in members:
                    member["font"]["size_pt"] = size
                    text_placement._refresh_content_unit(member)
            elif op == "restore_quiet_interior":
                if not text_placement.restore_quiet_interior(element, frame_output_dir):
                    raise ValueError("no_decorated_holder")
            else:
                raise ValueError("unsupported_patch_op")
            applied += 1
        except (ValueError, KeyError, StopIteration, TypeError) as exc:
            rejected.append({"patch": patch, "reason": str(exc) or "unknown_region"})
    expected = [(e["region_id"], e["text"], e.get("bullet_items", [])) for e in plan["elements"]]
    actual = [(e["region_id"], e["text"], e.get("bullet_items", [])) for e in working["elements"]]
    if actual != expected:
        return copy.deepcopy(plan), rejected + [{"reason": "frozen_copy_changed"}], 0
    return working, rejected, applied


def _visual_context(plan: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(plan["background_path"]).with_name("component_briefs.json")
    if not path.is_file():
        return []
    return [
        {
            "component_id": item.get("component_id"),
            "bbox_px": item.get("bbox_px"),
            "brief": item.get("visual_brief", ""),
        }
        for item in read_json(path)
    ]


def _topology_move_ranges(
    plan: dict[str, Any], region_id: str
) -> dict[str, list[float]]:
    element_by_region = {item["region_id"]: item for item in plan["elements"]}
    target = element_by_region[region_id]
    group = next(item for item in plan["groups"] if region_id in item["member_region_ids"])
    members = [
        element_by_region[item]
        for item in group["member_region_ids"]
        if item in element_by_region
    ]
    ranges: dict[str, list[float]] = {}
    for axis, coordinate in (("x", 0), ("y", 1)):
        minimum_delta = float("-inf")
        maximum_delta = float("inf")
        for member in members:
            bbox = [float(value) for value in member["bbox_px"]]
            search = [
                float(value)
                for value in member.get("topology_bbox_px", member["bbox_px"])
            ]
            minimum_delta = max(minimum_delta, search[coordinate] - bbox[coordinate])
            maximum_delta = min(
                maximum_delta,
                search[coordinate] + search[coordinate + 2]
                - bbox[coordinate]
                - bbox[coordinate + 2],
            )
        current = float(target["bbox_px"][coordinate])
        if minimum_delta <= maximum_delta and (
            minimum_delta <= -1.0 or maximum_delta >= 1.0
        ):
            ranges[axis] = [
                round(current + minimum_delta, 2),
                round(current + maximum_delta, 2),
            ]
    return ranges


def _apply_topology_move(
    plan: dict[str, Any],
    region_id: str,
    proposal: dict[str, Any],
    text_placement: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    original = copy.deepcopy(plan)
    working = copy.deepcopy(plan)
    try:
        axis = proposal.get("axis")
        if axis not in {"x", "y"}:
            return original, {"status": "ignored", "reason": "no_move"}
        coordinate = 0 if axis == "x" else 1
        element_by_region = {item["region_id"]: item for item in working["elements"]}
        target = element_by_region[region_id]
        position = float(proposal["position_px"])
        legal_positions = _topology_move_ranges(working, region_id)
        if axis not in legal_positions:
            return original, {"status": "ignored", "reason": "axis_not_movable"}
        minimum, maximum = legal_positions[axis]
        if minimum - 1.0 <= position < minimum:
            position = minimum
        elif maximum < position <= maximum + 1.0:
            position = maximum
        elif position < minimum or position > maximum:
            return original, {
                "status": "ignored",
                "reason": "move_does_not_fit_topology_bbox",
            }
        delta = position - float(target["bbox_px"][coordinate])
        if abs(delta) < 1.0:
            return original, {"status": "ignored", "reason": "move_too_small"}
        group = next(
            item for item in working["groups"] if region_id in item["member_region_ids"]
        )
        members = [
            element_by_region[item]
            for item in group["member_region_ids"]
            if item in element_by_region
        ]
        for member in members:
            member["bbox_px"][coordinate] += delta
            search = member.get("topology_bbox_px", member["bbox_px"])
            x, y, width, height = member["bbox_px"]
            if (
                x < search[0] - 1e-6
                or y < search[1] - 1e-6
                or x + width > search[0] + search[2] + 1e-6
                or y + height > search[1] + search[3] + 1e-6
            ):
                return original, {
                    "status": "ignored",
                    "reason": "move_does_not_fit_topology_bbox",
                }
            text_placement._refresh_holder_binding(member)
            text_placement._refresh_content_unit(member)
        violation = text_placement._validate_group_invariants(working)
        if violation:
            return original, {"status": "ignored", "reason": violation}
        return working, {"status": "applied", "reason": ""}
    except Exception as exc:
        return original, {"status": "ignored", "reason": f"invalid_move:{exc}"}


def _collision_pairs(plan: dict[str, Any]) -> set[str]:
    pairs = set()
    for index, left in enumerate(plan["elements"]):
        for right in plan["elements"][index + 1 :]:
            if _bbox_intersection(left["bbox_px"], right["bbox_px"]) > 1.0:
                pairs.add(f"{left['region_id']}::{right['region_id']}")
    return pairs


def _fit_candidate_issues(
    plan: dict[str, Any], element_index: int, candidate: dict[str, Any]
) -> list[str]:
    """Allow metric adaptation only while retaining the original free space."""
    bbox = candidate["bbox_px"]
    if len(bbox) != 4 or not all(math.isfinite(float(value)) for value in bbox):
        return ["non_finite_bbox"]
    x, y, width, height = bbox
    canvas = plan["canvas_px"]
    issues = []
    if width <= 0 or height <= 0 or x < -1e-6 or y < -1e-6 or (
        x + width > canvas["width"] + 1e-6 or y + height > canvas["height"] + 1e-6
    ):
        issues.append("out_of_bounds")
    original = plan["elements"][element_index]["bbox_px"]
    obstacles = [
        item for index, item in enumerate(plan["elements"]) if index != element_index
    ] + plan.get("protected_visual_regions", [])
    for obstacle in obstacles:
        if _bbox_intersection(bbox, obstacle["bbox_px"]) > (
            _bbox_intersection(original, obstacle["bbox_px"]) + 1.0
        ):
            issues.append(f"increased_collision:{obstacle['region_id']}")
    return issues


def _geometry_quality_findings(plan: dict[str, Any]) -> list[dict[str, Any]]:
    findings = [
        {"kind": "text_collision", "pair": pair} for pair in sorted(_collision_pairs(plan))
    ]
    for index, element in enumerate(plan.get("elements", [])):
        issues = _fit_candidate_issues(plan, index, element)
        if issues:
            findings.append({"kind": "geometry", "region_id": element["region_id"], "issues": issues})
        if element.get("registration", {}).get("status") == "unresolved":
            findings.append({"kind": "unresolved_fit", "region_id": element["region_id"]})
        if element.get("quality_warnings"):
            findings.append({"kind": "element_warning", "region_id": element["region_id"], "warnings": element["quality_warnings"]})
        intentional = set(plan.get("intentional_visual_overlaps", []))
        for visual in plan.get("protected_visual_regions", []):
            pair = f"{element['region_id']}::{visual['region_id']}"
            if pair not in intentional and _bbox_intersection(element["bbox_px"], visual["bbox_px"]) > 1.0:
                findings.append({"kind": "text_visual_collision", "pair": pair})
    return findings


def _repair_plan_bounds(plan: dict[str, Any], text_placement: Any) -> None:
    """Contain only invalid units; leave all valid planned geometry unchanged."""
    canvas = plan["canvas_px"]
    def contained(raw: list[float]) -> list[float]:
        defaults = [0.0, 0.0, min(300.0, canvas["width"]), min(60.0, canvas["height"])]
        values = [float(value) for value in raw]
        x, y, width, height = [value if math.isfinite(value) else default for value, default in zip(values, defaults)]
        width = min(max(1.0, width), float(canvas["width"]))
        height = min(max(1.0, height), float(canvas["height"]))
        return [min(max(0.0, x), canvas["width"] - width), min(max(0.0, y), canvas["height"] - height), width, height]
    for element in plan.get("elements", []):
        before = list(element["bbox_px"])
        element["bbox_px"] = contained(before)
        if element.get("topology_bbox_px"):
            element["topology_bbox_px"] = contained(element["topology_bbox_px"])
        if element["bbox_px"] != before:
            plan.setdefault("quality_warnings", []).append({
                "stage": "content_unit_fit", "region_id": element["region_id"],
                "repair": "contained_invalid_unit", "original_bbox_px": before,
            })
        text_placement._refresh_holder_binding(element)
        text_placement._refresh_content_unit(element)


def _restore_unplaced_content(
    plan: dict[str, Any], case: dict[str, Any], text_placement: Any
) -> None:
    """Keep required text when imperfect binding missed an existing region."""
    by_content = {item["content_id"]: item for item in case["content"]}
    inventory_boxes = {
        item["region_id"]: item["postprocessed_bbox_px"]
        for item in read_json(Path(case["layout_path"])).get("entities", [])
    }
    for content_id in list(plan["binding"]["unplaced_content_ids"]):
        content = by_content[content_id]
        region_id = content["target_region_id"]
        bbox = inventory_boxes.get(region_id)
        if bbox is None and plan["elements"]:
            # Preserve content in its closest bound text unit if its source
            # region was absent; do not introduce a second unrelated layout.
            target = min(plan["elements"], key=lambda item: abs(item.get("order", 0) - content["order"]))
            target["text"] += "\n" + content["text"]
            if target.get("bullet_items"):
                target["bullet_items"].extend(content.get("bullet_items") or [content["text"]])
        else:
            bbox = bbox or [48, 48, plan["canvas_px"]["width"] - 96, 120]
            element = copy.deepcopy(plan["elements"][0]) if plan["elements"] else {}
            element.update({
                "region_id": region_id, "role": content["role"], "bbox_px": list(bbox),
                "topology_bbox_px": list(bbox), "content_id": content_id,
                "text": content["text"], "bullet_items": list(content.get("bullet_items", [])),
                "font": copy.deepcopy(case["font_roles"][content["role"]]),
                "presentation": {"mode": "direct_text", "text_color": "111111"},
                "order": content["order"], "anchor_center_px": list(text_placement.bbox_center(bbox)),
                "constraint_group_id": f"singleton:{region_id}", "needs_terra_asset_review": False,
            })
            if "metrics" not in element:
                with Image.open(plan["background_path"]).convert("RGB") as image:
                    element["metrics"] = text_placement.analyze_region(text_placement.np.asarray(image), bbox)
            element.pop("holder_binding", None)
            element.pop("decoration", None)
            element.pop("content_unit", None)
            text_placement._refresh_content_unit(element)
            plan["elements"].append(element)
            plan["groups"].append({
                "group_id": f"singleton:{region_id}", "member_region_ids": [region_id],
                "arrangement": "freeform", "equal_size": False, "even_spacing": False,
            })
        plan.setdefault("quality_warnings", []).append({
            "stage": "binding", "content_id": content_id, "repair": "restored_required_text"
        })
    plan["binding"]["unplaced_content_ids"] = []


def _bbox_intersection(left: list[float], right: list[float]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[0] + left[2], right[0] + right[2])
    y1 = min(left[1] + left[3], right[1] + right[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _hard_rejection_reasons(
    baseline: dict[str, Any], candidate: dict[str, Any], text_placement: Any
) -> list[str]:
    reasons = []
    violation = text_placement._validate_group_invariants(candidate)
    if violation:
        reasons.append(violation)
    new_overflow = set(_overflow_region_ids(candidate)) - set(
        _overflow_region_ids(baseline)
    )
    if new_overflow:
        reasons.append("new_overflow:" + ",".join(sorted(new_overflow)))
    new_collisions = _collision_pairs(candidate) - _collision_pairs(baseline)
    if new_collisions:
        reasons.append("new_collision:" + ",".join(sorted(new_collisions)))
    baseline_by_id = {item["region_id"]: item for item in baseline["elements"]}
    for index, left in enumerate(candidate["elements"]):
        original_left = baseline_by_id.get(left["region_id"])
        if original_left is None:
            continue
        for right in candidate["elements"][index + 1:]:
            original_right = baseline_by_id.get(right["region_id"])
            if original_right is not None and _bbox_intersection(left["bbox_px"], right["bbox_px"]) > (
                _bbox_intersection(original_left["bbox_px"], original_right["bbox_px"]) + 1.0
            ):
                reasons.append(f"increased_collision:{left['region_id']}::{right['region_id']}")
        for visual in baseline.get("protected_visual_regions", []):
            if _bbox_intersection(left["bbox_px"], visual["bbox_px"]) > (
                _bbox_intersection(original_left["bbox_px"], visual["bbox_px"]) + 1.0
            ):
                reasons.append(f"increased_visual_collision:{left['region_id']}::{visual['region_id']}")
    return reasons


def _attach_text_fit_audit(
    plans_by_slide: dict[str, dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    for slide in audit.get("slides", []):
        slide_id = slide.get("slide_id")
        if slide_id in plans_by_slide:
            plans_by_slide[slide_id]["text_fit_audit"] = slide


def _overflow_region_ids(plan: dict[str, Any]) -> list[str]:
    return list(plan.get("text_fit_audit", {}).get("overflow_region_ids", []))


def _prompt_with_context(prompt: str, context: dict[str, Any]) -> str:
    return prompt.rstrip() + "\n\nProgram context:\n" + json.dumps(
        context, ensure_ascii=False, separators=(",", ":")
    )


def _create_montage(rendered_dir: Path, output_path: Path) -> None:
    paths = sorted(rendered_dir.glob("slide-*.png"))
    if not paths:
        preview = Image.new("RGB", (384, 216), "white")
        ImageDraw.Draw(preview).text((16, 100), "Rendered preview unavailable; see quality report", fill="black")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        preview.save(output_path)
        return
    _create_montage_from_paths(paths, output_path, cell_width=384, cell_height=216)


def _create_montage_from_paths(
    paths: list[Path],
    output_path: Path,
    *,
    cell_width: int = 768,
    cell_height: int = 432,
) -> None:
    gap, label_height = 12, 22
    montage = Image.new(
        "RGB",
        (gap + len(paths) * (cell_width + gap), cell_height + label_height + 2 * gap),
        "white",
    )
    draw = ImageDraw.Draw(montage)
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image = ImageOps.contain(image, (cell_width, cell_height))
        x = gap + index * (cell_width + gap) + (cell_width - image.width) // 2
        y = gap + (cell_height - image.height) // 2
        montage.paste(image, (x, y))
        draw.text((gap + index * (cell_width + gap), gap + cell_height + 3), str(index + 1), fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_path)


def _adapt_region_content(
    region_content: dict[str, Any], inventory: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    role_map = {
        "title": "title",
        "standalone_text": "body",
        "supporting_text": "body",
        "caption": "caption",
        "label": "caption",
        "internal_label": "caption",
    }
    inventory_by_region = {
        item["region_id"]: item for item in (inventory or []) if "region_id" in item
    }
    return [
        {
            "content_id": f"content_{item['region_id']}",
            "role": role_map[item["role"]],
            "text": _join_text_lines(item["lines"], item["role"]),
            "bullet_items": _bullet_items(item["lines"], item["role"]),
            "order": index,
            "target_region_id": item["region_id"],
            "source_role": item["role"],
            "semantic_asset_roles": [],
            "provisional_presentation": {
                "mode": inventory_by_region.get(item["region_id"], {}).get(
                    "presentation_mode", "direct_text"
                ),
                "holder_binding": inventory_by_region.get(item["region_id"], {}).get(
                    "holder_binding"
                ),
                "effective_text_bbox_px": inventory_by_region.get(
                    item["region_id"], {}
                ).get("effective_text_bbox_px"),
                "capacity_hint": inventory_by_region.get(item["region_id"], {}).get(
                    "capacity_hint"
                ),
            },
        }
        for index, item in enumerate(region_content["text_regions"])
    ]


def _join_text_lines(lines: list[str], role: str) -> str:
    return "\n".join(lines)


def _bullet_items(lines: list[str], role: str) -> list[str]:
    if len(lines) < 2 or role == "title":
        return []
    return [line.strip() for line in lines if line.strip()]


def _font_roles(config: PipelineConfig) -> dict[str, Any]:
    return {
        "title": {
            "face": config.title_font,
            "size_pt": 34,
            "minimum_pt": 25,
            "maximum_pt": 42,
            "weight": "bold",
        },
        "subtitle": {
            "face": config.body_font,
            "size_pt": 22,
            "minimum_pt": 18,
            "maximum_pt": 28,
            "weight": "regular",
        },
        "body": {
            "face": config.body_font,
            "size_pt": 18,
            "minimum_pt": 18,
            "maximum_pt": 24,
            "weight": "regular",
        },
        "caption": {
            "face": config.body_font,
            "size_pt": 18,
            "minimum_pt": 14,
            "maximum_pt": 20,
            "weight": "semibold",
        },
    }


def _compile_deck(
    compilation_path: Path,
    output_path: Path,
    audit_path: Path | None = None,
) -> None:
    command = [
        "node",
        str((ROOT / "compile_deck.mjs").resolve()),
        str(compilation_path.resolve()),
        str(output_path.resolve()),
    ]
    if audit_path is not None:
        command.append(str(audit_path.resolve()))
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if result.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            "PPTX compilation failed: "
            + (result.stderr.strip() or result.stdout.strip())[:2000]
        )


def _copy_rendered_previews(
    rendered_dir: Path,
    output_dir: Path,
    slide_ids: list[str],
) -> None:
    rendered_paths = {
        int(path.stem.rsplit("-", 1)[1]): path for path in rendered_dir.glob("slide-*.png")
    }
    for page_number, slide_id in enumerate(slide_ids, start=1):
        rendered_path = rendered_paths.get(page_number)
        if rendered_path is None:
            continue
        preview_path = output_dir / slide_id / "preview.png"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rendered_path, preview_path)


def _render_and_validate(
    pptx_path: Path,
    pdf_path: Path,
    rendered_dir: Path,
    slide_count: int,
) -> None:
    from .page_runtime import resource_lease
    with resource_lease("render"):
        _render_and_validate_unlimited(pptx_path, pdf_path, rendered_dir, slide_count)


def _render_and_validate_unlimited(
    pptx_path: Path,
    pdf_path: Path,
    rendered_dir: Path,
    slide_count: int,
) -> None:
    if shutil.which("libreoffice") is None or shutil.which("pdftoppm") is None:
        raise RuntimeError("LibreOffice and pdftoppm are required")
    pdf_path.unlink(missing_ok=True)
    rendered_dir.mkdir(parents=True, exist_ok=True)
    for stale in rendered_dir.glob("slide-*.png"):
        stale.unlink()
    # Concurrent deck checks must not attach to another LibreOffice process's
    # profile and silently convert into that process's output directory.
    with tempfile.TemporaryDirectory(prefix="rl4ppt-lo-profile-") as profile_dir:
        result = subprocess.run(
            [
                "libreoffice",
                f"-env:UserInstallation={Path(profile_dir).as_uri()}",
                "--headless", "--convert-to", "pdf", "--outdir",
                str(pdf_path.parent), str(pptx_path),
            ],
            capture_output=True, text=True, timeout=240,
        )
    if result.returncode != 0 or not pdf_path.is_file():
        raise RuntimeError("LibreOffice failed to render the PPTX")
    result = subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            "120",
            str(pdf_path),
            str(rendered_dir / "slide"),
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    if result.returncode != 0:
        raise RuntimeError("pdftoppm failed to render the PDF")
    rendered = sorted(rendered_dir.glob("slide-*.png"))
    warnings = []
    if len(rendered) != slide_count:
        warnings.append({"kind": "rendered_page_count", "expected": slide_count, "actual": len(rendered)})
    for path in rendered:
        with Image.open(path).convert("RGB") as image:
            if not any(low != high for low, high in image.getextrema()):
                warnings.append({"kind": "blank_rendered_slide", "path": str(path)})
    write_json(rendered_dir / "render_quality.json", {
        "status": "degraded" if warnings else "strict_pass", "warnings": warnings,
    })
