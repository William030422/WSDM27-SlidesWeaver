#!/usr/bin/env python3
"""Reassemble saved real Text First outputs without generation/API calls.

Only cloned HTML and fresh output artifacts are changed. The original text,
canonical background, and real generated raster layers are read-only inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from integration.config import PipelineConfig, Size
from integration.html_planning import apply_initial_readability_contract, allocate_holder_text_space, allocate_readable_text_space
from integration.html_text_rework import apply_deck_typography_profile
from integration.html_compile import run_html_to_coordinates
from integration.text_assembly import run_reused_text_assembly


def read(path: Path):
    return json.loads(path.read_text())


def offline_config() -> PipelineConfig:
    # Explicit public settings: no .env loading, credentials, or mock generation.
    return PipelineConfig(project_root=ROOT,env_path=ROOT/'NO_CONFIGURATION_USED',
        api_root='https://offline.invalid/v1',api_key_env='NO_API_KEY_USED',
        chat_model='reused-real-artifacts',image_model='gpt-image-2-official',
        image_size=Size(1536,864),image_resolution='1k',image_quality='medium',image_max_attempts=2,
        chat_max_tokens=4096,vlm_max_tokens=4096,timeout_seconds=0,poll_initial_delay_seconds=0,
        poll_interval_seconds=0,poll_timeout_seconds=0,
        diffusion_model_path=Path(os.environ.get('DIFFUSION_MODEL_PATH', Path.home() / 'models' / 'Tongyi-MAI_Z-Image-Turbo')).expanduser(),
        diffusion_size=Size(1024,576),diffusion_steps=5,diffusion_guidance_scale=0,diffusion_seed=42,
        deck_font='Noto Sans')


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('CONDA_DEFAULT_ENV')!='rl4ppt-final':
        raise RuntimeError('Activate rl4ppt-final before replay')
    source,output=args.source_run.resolve(),args.output_dir.resolve()
    if source==output or source in output.parents:
        raise ValueError('Use an independent output directory; preserve the original run')
    output.mkdir(parents=True,exist_ok=True)
    for name in ('text_generation', 'step2', 'final_background'):
        target = output / name
        if not target.exists():
            target.symlink_to(source / name, target_is_directory=True)
    deck=read(source/'step2/deck_plan.json')
    ids=[s['slide_id'] for s in deck['slides']]
    html_by_slide={}
    repairs={}
    for sid in ids:
        target=output/'html_planning/slides'/sid
        target.mkdir(parents=True,exist_ok=True)
        for name in ['layout.html','visual_generation_groups.json']:
            original=source/'html_planning/slides'/sid/name
            if original.is_file(): shutil.copyfile(original,target/name)
        html_by_slide[sid]=target/'layout.html'
        repairs[sid]={'readability':apply_initial_readability_contract(html_by_slide[sid]),
                      'holder_text_allocation':allocate_holder_text_space(html_by_slide[sid]),
                      'text_box_allocation':allocate_readable_text_space(html_by_slide[sid])}
    previous=read(source/'html_planning/typography_profile.json')
    old=previous.get('profile') or {}
    profile={'title_font_size_px':old.get('title_font_size_px',43),
             'subtitle_font_size_px':old.get('subtitle_font_size_px',24)}
    typography=apply_deck_typography_profile(html_by_slide=html_by_slide,profile=profile)
    for sid in ids:
        repairs[sid]['final_role_space']=allocate_readable_text_space(html_by_slide[sid])
    (output/'html_planning/typography_profile.json').write_text(json.dumps(typography,indent=2)+'\n')
    config=offline_config()
    compile_manifest=run_html_to_coordinates(html_planning_dir=output/'html_planning',
        step2_dir=source/'step2',text_generation_dir=source/'text_generation',
        output_dir=output/'html_compile',slide_ids=ids,config=config)
    assembly=run_reused_text_assembly(html_compile_dir=output/'html_compile',
        text_generation_dir=source/'text_generation',final_background_dir=source/'final_background',
        output_dir=output/'assembly',slide_ids=ids,config=config,mock_vlm=True)
    manifest={'stage':'zero_api_saved_artifact_replay','status':'succeeded',
              'quality_status':assembly.get('quality_status'),'source_run':str(source),
              'output_dir':str(output),'model_calls_during_replay':0,'generated_images_during_replay':0,
              'reused_real_generated_layers':True,'slide_count':len(ids),
              'precommit_repairs':repairs,'typography':typography,
              'html_compile_quality':compile_manifest.get('quality_status'),'assembly':assembly}
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':manifest['status'],'quality_status':manifest['quality_status'],
                      'output_dir':str(output),'slide_count':len(ids),'model_calls':0},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
