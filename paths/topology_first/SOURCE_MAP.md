# Topology First modules

Topology First grounds a diffusion composition into regions and relationships,
then generates copy that fits those regions. Final scene synthesis is combined
with editable text and bounded content-unit fitting.

| Module | Responsibility |
| --- | --- |
| `src/v9_deck_planning/` | Deck narrative, text-presentation policy, canonical background and reusable style assets |
| `src/v9_deck_planning/apimart.py` | Shared API client; typed task failures distinguish terminal provider results from polling timeouts |
| `integration/slide_spec.py` | Per-slide semantic requests for composition |
| `reused/diffusion/`, `integration/diffusion.py` | Local Z-Image-Turbo inference, prompts and mock mode |
| `reused/topology/`, `integration/topology.py` | Region detection, calibration, regularization and topology previews |
| `integration/semantic_topology.py` | Semantic groups, inventories and generation units |
| `integration/region_content.py` | Capacity-aware copy, visual briefs and holder-family policy |
| `integration/final_background.py` | Scene generation, spatial control and image task recovery |
| `reused/text_placement/text_placement.py` | Holder binding, geometry, contrast and content-unit constraints |
| `reused/text_placement/prompts/` | Bounded layout, movement, copy and acceptance contracts |
| `integration/text_assembly.py` | Fitting, per-slide review, deterministic patch acceptance and native PPTX/PDF export |
| `integration/background_decomposition.py` | Companion background composition with movable raster modules |
| `integration/page_pipeline.py` | Per-page chains and the shared holder-policy decision |
| `integration/page_runtime.py`, `integration/gpu_runtime.py` | Persistent page owners, GPU processes and ownership locks |
| `integration/api_journal.py`, `integration/shared_checkpoint.py` | Saved API responses and shared-stage checkpoints |
| `src/v9_deck_planning/image_task_journal.py` | Image task submission and recovery records |
| `compile_deck.mjs`, `pptx_compiler/` | Native PowerPoint text, images and supported geometry |
| `run_pipeline.py` | Stage selection, page execution and final assembly |

Use the [root launcher](../../README.md#quick-start) to choose the path and
configuration. See the [running guide](../../docs/running.md) for valid stages,
recovery and output reports.
