from __future__ import annotations

import importlib
import hashlib
import sys
from pathlib import Path
from typing import Any

from v9_deck_planning.apimart import ApiMartClient
from v9_deck_planning.json_utils import read_json, write_json

from .config import PipelineConfig
from .semantic_topology import (
    extract_semantic_topology,
    fallback_semantic_topology,
    regularize_semantic_topology,
)


ROOT = Path(__file__).resolve().parents[1]
REUSED_DIR = ROOT / "reused" / "topology"


def run_reused_topology_pipeline(
    *,
    project_root: Path,
    skeleton_dir: Path,
    output_dir: Path,
    slide_ids: list[str],
    config: PipelineConfig,
    reuse_raw: bool = False,
    offline: bool = False,
    write_stage_manifest: bool = True,
) -> dict[str, Any]:
    if config.image_size.as_list() != [1536, 864]:
        raise ValueError(
            "the reused topology postprocessor has a tested native canvas of 1536x864"
        )
    if str(REUSED_DIR) not in sys.path:
        sys.path.insert(0, str(REUSED_DIR))
    identity = importlib.import_module("extract_layout_identity_geometry_with_vlm")
    calibration = importlib.import_module("extract_calibrated_topology_with_vlm")
    postprocess = importlib.import_module("postprocess_calibrated_planning_margins")

    identity_dir = output_dir / "layout_identity"
    semantic_dir = output_dir / "semantic_topology"
    calibration_dir = output_dir / "calibrated_topology"
    postprocess_dir = output_dir / "postprocessed"
    identity.V9_ROOT = project_root
    identity.REFERENCE_DIR = skeleton_dir
    identity.MODEL = config.chat_model
    identity.MAX_TOKENS = config.vlm_max_tokens

    identity_results = []
    for slide_id in slide_ids:
        resumed = _resume_succeeded(
            identity_dir,
            slide_id,
            "layout_identity.json",
            skeleton_dir / f"{slide_id}.png",
            config.chat_model,
            not offline and not reuse_raw,
            hashlib.sha256(identity.extraction_prompt(slide_id).encode("utf-8")).hexdigest(),
            config.vlm_max_tokens,
        )
        identity_results.append(
            resumed
            or identity.extract_slide(
                slide_id,
                identity_dir,
                offline or _reuse_raw_if_available(identity_dir, slide_id, reuse_raw),
            )
        )
    identity_failed = [item for item in identity_results if item["status"] == "failed"]

    semantic_client = None
    if not offline:
        semantic_client = ApiMartClient(
            api_root=config.api_root,
            api_key_env=config.api_key_env,
            env_path=config.env_path,
            timeout_seconds=config.timeout_seconds,
            poll_initial_delay_seconds=config.poll_initial_delay_seconds,
            poll_interval_seconds=config.poll_interval_seconds,
            poll_timeout_seconds=config.poll_timeout_seconds,
        )
    semantic_results = []
    if not identity_failed:
        for slide_id in slide_ids:
            semantic_results.append(
                extract_semantic_topology(
                    slide_id=slide_id,
                    skeleton_path=skeleton_dir / f"{slide_id}.png",
                    identity_path=identity_dir / slide_id / "layout_identity.json",
                    identity_overlay_path=identity_dir / slide_id / "layout_identity_overlay.png",
                    output_dir=semantic_dir,
                    client=semantic_client,
                    model=config.chat_model,
                    max_tokens=min(config.vlm_max_tokens, 2048),
                    reuse_raw=_reuse_raw_if_available(semantic_dir, slide_id, reuse_raw),
                    offline=offline,
                )
            )
    semantic_failed = [item for item in semantic_results if item["status"] == "failed"]

    regularization_results = []
    if not identity_failed and not semantic_failed:
        for slide_id in slide_ids:
            try:
                compiled, report = regularize_semantic_topology(
                    semantic=read_json(
                        semantic_dir / slide_id / "semantic_topology.json"
                    ),
                    identity=read_json(
                        identity_dir / slide_id / "layout_identity.json"
                    ),
                )
                compiled = calibration.CalibratedTopologyResult.model_validate(
                    compiled
                ).model_dump(mode="json")
                calibrated_slide_dir = calibration_dir / slide_id
                write_json(
                    calibrated_slide_dir / "calibrated_topology.json", compiled
                )
                regularization_results.append(
                    {
                        "slide_id": slide_id,
                        "status": "succeeded",
                        "layout_family": report["layout_family"],
                        "content_block_count": report["content_block_count"],
                        "top_text_slot_count": report["top_text_slot_count"],
                        "bottom_text_slot_count": report["bottom_text_slot_count"],
                        "dual_text_slot_block_count": report[
                            "dual_text_slot_block_count"
                        ],
                        "overlap_count": report["overlap_count"],
                        "calibrated_topology": str(
                            (
                                calibrated_slide_dir / "calibrated_topology.json"
                            ).resolve()
                        ),
                    }
                )
            except (ValueError, TypeError, KeyError) as exc:
                # Invalid model geometry is a local quality issue. Recover from
                # this slide's existing source regions, preserving other slides.
                identity_data = read_json(identity_dir / slide_id / "layout_identity.json")
                compiled, report = regularize_semantic_topology(
                    semantic=fallback_semantic_topology(identity_data, slide_id),
                    identity=identity_data,
                )
                compiled = calibration.CalibratedTopologyResult.model_validate(compiled).model_dump(mode="json")
                calibrated_slide_dir = calibration_dir / slide_id
                write_json(calibrated_slide_dir / "calibrated_topology.json", compiled)
                regularization_results.append(
                    {"slide_id": slide_id, "status": "succeeded", "quality_status": "fallback",
                     "warning": str(exc), "overlap_count": report["overlap_count"],
                     "calibrated_topology": str((calibrated_slide_dir / "calibrated_topology.json").resolve())}
                )
    regularization_failed = [
        item for item in regularization_results if item["status"] == "failed"
    ]

    postprocess_results = []
    postprocess_warnings = []
    if not identity_failed and not semantic_failed and not regularization_failed:
        for slide_id in slide_ids:
            try:
                postprocess_results.append(
                    postprocess.render_slide(calibration_dir / slide_id, postprocess_dir)
                )
            except Exception as exc:
                postprocess_results.append(
                    _fallback_postprocess(
                        slide_id=slide_id,
                        calibration_dir=calibration_dir,
                        skeleton_dir=skeleton_dir,
                        output_dir=postprocess_dir,
                    )
                )
                postprocess_warnings.append({"slide_id": slide_id, "error": str(exc)})
    failed = [*identity_failed, *semantic_failed, *regularization_failed]
    semantic_warnings = [
        {"slide_id": item["slide_id"], "warning": item.get("warning")}
        for item in semantic_results
        if item.get("quality_status") == "fallback"
    ]
    semantic_warnings.extend(
        {"slide_id": item["slide_id"], "warning": item.get("warning") or "remaining_topology_overlap",
         "overlap_count": item.get("overlap_count", 0)}
        for item in regularization_results
        if item.get("quality_status") == "fallback" or item.get("overlap_count", 0)
    )
    manifest = {
        "stage": "fine_identity_semantic_topology_regularization_and_postprocess",
        "status": "failed" if failed else "succeeded",
        "error": _summarize_failures(
            identity_failed, semantic_failed, regularization_failed
        ),
        "quality_status": (
            "fallback" if postprocess_warnings or semantic_warnings else "strict_pass"
        ),
        "warnings": [*semantic_warnings, *postprocess_warnings],
        "model_call_count": sum(
            item.get("model_call_count", 0)
            for item in [*identity_results, *semantic_results]
        ),
        "sources": {
            "layout_identity": str((REUSED_DIR / "extract_layout_identity_geometry_with_vlm.py").resolve()),
            "semantic_topology": str(
                (Path(__file__).resolve().parent / "semantic_topology.py").resolve()
            ),
            "semantic_schema": str((REUSED_DIR / "extract_calibrated_topology_with_vlm.py").resolve()),
            "postprocess": str((REUSED_DIR / "postprocess_calibrated_planning_margins.py").resolve()),
        },
        "standalone_topology_parser_executed": False,
        "regularization_persistence": "in_memory",
        "slide_count": len(slide_ids),
        "output_dir": str(output_dir.resolve()),
        "layout_identity": identity_results,
        "semantic_topology": semantic_results,
        "regularization": regularization_results,
        "postprocess": postprocess_results,
        "artifacts": {
            "identity_dir": str(identity_dir.resolve()),
            "semantic_topology_dir": str(semantic_dir.resolve()),
            "calibration_dir": str(calibration_dir.resolve()),
            "postprocessed_dir": str(postprocess_dir.resolve()),
            "layouts_dir": str((postprocess_dir / "layouts").resolve()),
        },
    }
    if write_stage_manifest:
        write_json(output_dir / "run_manifest.json", manifest)
    if failed:
        raise RuntimeError(manifest["error"])
    return manifest


def _reuse_raw_if_available(output_dir: Path, slide_id: str, requested: bool) -> bool:
    return requested and (output_dir / slide_id / "vlm_raw_response.json").is_file()


def _resume_succeeded(
    output_dir: Path,
    slide_id: str,
    required_artifact: str,
    reference_path: Path | None = None,
    expected_model: str | None = None,
    require_model_call: bool = False,
    expected_prompt_sha256: str | None = None,
    expected_max_tokens: int | None = None,
) -> dict[str, Any] | None:
    manifest_path = output_dir / slide_id / "run_manifest.json"
    artifact_path = output_dir / slide_id / required_artifact
    if not manifest_path.is_file() or not artifact_path.is_file():
        return None
    manifest = read_json(manifest_path)
    if manifest.get("status") != "succeeded":
        return None
    if expected_model is not None and manifest.get("model") != expected_model:
        return None
    if require_model_call and not manifest.get("raw_response_model_call_count", manifest.get("model_call_count", 0)):
        return None
    if expected_prompt_sha256 is not None and manifest.get("inputs", {}).get("prompt_sha256") != expected_prompt_sha256:
        return None
    if expected_max_tokens is not None and manifest.get("max_tokens") != expected_max_tokens:
        return None
    if reference_path is not None:
        recorded = manifest.get("inputs", {}).get("reference_sha256")
        if recorded != hashlib.sha256(reference_path.read_bytes()).hexdigest():
            return None
    return {**manifest, "resumed": True, "model_call_count": 0}


def _summarize_failures(
    *failed_stages: list[dict[str, Any]],
) -> str | None:
    failures = [
        f"{item.get('slide_id')}: {item.get('error')}"
        for stage in failed_stages
        for item in stage
    ]
    return " | ".join(failures) if failures else None


def _fallback_postprocess(
    *,
    slide_id: str,
    calibration_dir: Path,
    skeleton_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    from PIL import Image

    topology = read_json(
        calibration_dir / slide_id / "calibrated_topology.json"
    )
    entities = [
        {
            "region_id": item["region_id"],
            "input_calibrated_bbox_px": [value * 24 for value in item["calibrated_bbox"]],
            "postprocessed_bbox_px": [value * 24 for value in item["calibrated_bbox"]],
        }
        for item in topology["entities"]
    ]
    layout = {
        "slide_id": slide_id,
        "canvas": {"width": 1536, "height": 864},
        "entities": entities,
    }
    layout_path = output_dir / "layouts" / f"{slide_id}.json"
    write_json(layout_path, layout)
    image_path = output_dir / f"{slide_id}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(skeleton_dir / f"{slide_id}.png") as source:
        source.convert("RGB").resize((1536, 864)).save(image_path)
    return {
        "slide_id": slide_id,
        "output": str(image_path.resolve()),
        "layout": str(layout_path.resolve()),
        "fallback": True,
    }
