# Text First modules

Text First commits the copy before determining final geometry. The pipeline
uses diffusion as composition inspiration, validates the layout in a browser,
and assembles native text with independently selectable raster assets.

| Module | Responsibility |
| --- | --- |
| `src/v9_deck_planning/` | Deck narrative, text-presentation policy, canonical background and design metadata |
| `src/v9_deck_planning/apimart.py` | Chat, vision and image API client |
| `integration/text_generation.py` | Final copy and parsed text trees |
| `integration/expression_logic.py` | Concise visual relationships for each slide |
| `reused/diffusion/` | Local Z-Image-Turbo inference and composition prompts |
| `integration/diffusion.py` | Diffusion inputs, outputs and mock mode |
| `integration/html_planning.py` | HTML planning, geometry review and bounded rework |
| `integration/html_text_rework.py` | Guarded copy refinement and shared typography |
| `integration/html_compile.py` | Browser-derived geometry, rich-text structure and visual regions |
| `integration/final_background.py` | Slot-aware asset boards, transparent extraction and placement |
| `integration/holder_safe_rect.py` | Detection of usable text interiors in generated holders |
| `integration/text_assembly.py` | Layer composition, contrast, text fitting and native PPTX/PDF export |
| `integration/page_pipeline.py` | Per-page dependency chains and the deck typography decision |
| `integration/page_runtime.py`, `integration/gpu_runtime.py` | Persistent page owners, GPU processes and ownership locks |
| `integration/shared_checkpoint.py` | Reusable shared-stage checkpoints |
| `src/v9_deck_planning/image_task_journal.py` | Image task submission and recovery records |
| `compile_deck.mjs`, `pptx_compiler/` | Native PowerPoint text, images and supported geometry |
| `run_pipeline.py` | Stage selection, page execution and final assembly |

Use the [root launcher](../../README.md#quick-start) to choose the path and
configuration. See the [running guide](../../docs/running.md) for valid stages,
recovery and output reports.
