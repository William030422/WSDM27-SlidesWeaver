# Running SlidesWeaver

[English overview](../README.md) · [中文说明](../README.zh-CN.md) · [Architecture](architecture.md)

Run commands below from the repository root in the `rl4ppt-final` Conda environment.

## Dependencies and preflight

Live generation uses a compatible NVIDIA GPU, a local Z-Image-Turbo model, and APIMART access to the configured chat/vision and image models. Choose model names supported by your endpoint. The local diffusion stack and renderer dependencies are installed by:

```bash
./scripts/setup_environment.sh
conda activate rl4ppt-final
```

The setup script expects Conda to be available. It installs the Python/CUDA stack, Node, Playwright Chromium, and Poppler into the unified environment. Install LibreOffice and fonts on the host separately. For Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y libreoffice-impress fontconfig fonts-noto-core fonts-noto-cjk
```

For other distributions, install equivalent packages with their package manager. Noto Sans Regular/Bold and CJK fonts are part of the rendering contract: font substitution can change text width, line breaks, and PDF results.

Check dependencies for offline validation, or also check the local model for a live run:

```bash
python scripts/check_environment.py --offline
python scripts/check_environment.py --model-path /path/to/Z-Image-Turbo
```

The offline check does not establish live API access or reproduce a GPU generation. Keep all pipeline stages in the same Conda environment.

## Configuration and input

Create a private configuration from the public template:

```bash
cp configs/example.conf configs/local.conf
```

Edit the endpoint, supported model identifiers, and `DIFFUSION_MODEL_PATH`. Supply credentials through your shell:

```bash
export APIMART_API_KEY="your-api-key"
```

The configuration uses `KEY=value` entries. The template documents all required settings; its main groups are:

| Settings | Purpose |
| --- | --- |
| `APIMART_API_ROOT`, `APIMART_CHAT_MODEL`, `APIMART_IMAGE_MODEL` | Provider endpoint and models |
| API token budgets, polling, timeout, and attempt settings | Request and recovery behavior |
| `APIMART_IMAGE_SIZE=1536x864` | Canonical and final raster contract |
| `DIFFUSION_MODEL_PATH` | Local Z-Image-Turbo model directory |
| `DIFFUSION_IMAGE_SIZE=1024x576` | Composition proposal contract |
| Diffusion steps, guidance, and seed | Local composition settings |
| PPT font settings | Installed fonts for editable export |

Do not change the two raster contracts to arbitrary aspect ratios. They are consumed by the existing mask, layout, and assembly code.

Input JSON supports `content` and `style`:

```json
{
  "content": "Create a 4-slide presentation explaining feedback loops to university students. Use English and finish with a concrete worked example.",
  "style": "Editorial diagrams, warm cream background, deep blue and orange accents, generous negative space. Generated artwork contains no readable text."
}
```

Specify slide count, language, audience, required facts, and communication goal in `content`. Legacy keys `user_request` and `style_description` are accepted. Start with [`examples/feedback.json`](../examples/feedback.json).

## Complete runs

```bash
python run.py --path text-first \
  --config configs/local.conf \
  --input examples/feedback.json \
  --output-dir outputs/text_first_live \
  --gpu-devices 0

python run.py --path topology-first \
  --config configs/local.conf \
  --input examples/feedback.json \
  --output-dir outputs/topology_first_live \
  --gpu-devices 0
```

Paths supplied to the root runner are resolved relative to the current working directory. Give different decks and different routes separate output directories.

Five persistent page workers are used by default. `--page-workers 1` selects sequential execution; `--gpu-devices 0,1` selects two GPUs. A GPU owns one resident local model and queues its assigned composition requests. API work can continue while pages wait for local inference.

## Offline and partial mock modes

```bash
python run.py --path text-first \
  --input examples/feedback.json \
  --output-dir outputs/text_first_mock \
  --mock-api --mock-diffusion

python run.py --path topology-first \
  --input examples/feedback.json \
  --output-dir outputs/topology_first_mock \
  --mock-api --mock-diffusion
```

The root runner chooses the bundled mock configuration for these commands. No API key, local model weights, or GPU is required. Node, browser dependencies, fonts, and LibreOffice are still needed for the corresponding layout and export stages.

| Option | Effect |
| --- | --- |
| `--mock-api` | Replace remote model calls with deterministic local responses |
| `--mock-diffusion` | Replace local diffusion inference with a mock composition reference |
| `--mock-images` | Mock remote image generation while leaving chat/VLM calls live |

For a fully offline example, use both `--mock-api` and `--mock-diffusion`. `--mock-images` alone still makes paid chat/VLM calls. Mock outputs test orchestration and editable export; they do not represent live synthesis quality.

## Stages and resume

Valid stages differ between the two paths:

| Path | Stages in execution order |
| --- | --- |
| Text First | `step1` → `text_generation` → `step2` → `expression_logic` → `diffusion` → `html_planning` → `html_compile` → `background` → `assembly` |
| Topology First | `step1` → `step2` → `slide_spec` → `diffusion` → `topology` → `region_content` → `background` → `assembly` |

An ordinary rerun reuses completed shared and page checkpoints when their input fingerprints and required files match. Saved responses and image task IDs also allow recovery inside interrupted stages. Changing worker count does not require regenerating completed pages.

Stop after a stage to inspect its artifacts:

```bash
python run.py --path text-first \
  --config configs/local.conf \
  --input examples/feedback.json \
  --output-dir outputs/text_first_live \
  --stop-after text_generation
```

Resume a run with its existing output directory and earlier artifacts:

```bash
python run.py --path text-first \
  --config configs/local.conf \
  --input examples/feedback.json \
  --output-dir outputs/text_first_live \
  --start-at html_planning

python run.py --path topology-first \
  --config configs/local.conf \
  --input examples/feedback.json \
  --output-dir outputs/topology_first_live \
  --start-at topology
```

Resume requires completed prerequisite stages; the examples assume the relevant earlier artifacts exist. After intentionally editing a Text First `html_planning/slides/<slide_id>/layout.html`, use `--start-at html_compile` to make the edited HTML authoritative. Ordinary recovery uses the saved typography-committed snapshot.

When an image submission has an unknown outcome, inspect its task/request journal before retrying. The existing recovery code records ambiguous outcomes instead of automatically duplicating a paid submission. An output lock prevents cooperating runs from writing to the same directory simultaneously.

## Outputs and diagnostics

| Artifact under `outputs/<run>/` | Meaning |
| --- | --- |
| `assembly/deck.pptx` | Final editable deck |
| `assembly/deck.pdf` | Same-directory LibreOffice PDF render |
| `assembly/deck_montage.png` | Ordered slide preview |
| `assembly/rendered_slides/` | Rasterized PDF pages and rendering evidence |
| `assembly/text_fit_report.json` | Actual font/text fitting diagnostics |
| `assembly/run_manifest.json` | Assembly outputs and quality findings |
| `run_manifest.json` | Whole-run completion, modes, stages, and quality status |
| `page_execution.json` | Owners, timings, and checkpoint reuse |
| `shared_planning/checkpoints/`, `pages/<slide_id>/` | Recovery state |

Text First additionally writes `assembly/pptx_render_check.json`, `contrast_report.json`, `geometry_quality_report.json`, and `raster_editability_report.json`. Topology First stores its render findings in `assembly/rendered_slides/render_quality.json` and its assembly manifest. Consult those paths before treating a successful process exit as a clean result.

| Status | Interpretation |
| --- | --- |
| `status: succeeded` | The requested stages completed; also inspect quality fields |
| `quality_status: strict_pass` | The emitted checks did not report unresolved quality issues |
| `quality_status: degraded` | Output retained with fallback, repair, or unresolved findings recorded |
| Stage `mock` / `fallback` | Synthetic or recovery behavior; inspect mode flags and warnings |
| `status: partial`, `failed_slides`, or page failure records | Some requested pages did not complete normally |

Quality fields report the checks implemented by the pipeline. They do not establish factual correctness or replace visual inspection of the exported deck.

## Validation

```bash
python scripts/run_tests.py --path all
python scripts/verify_provenance.py
```

The first command runs each path's tests in a separate process. The second verifies the source-file checksums. See [validation](../RESULTS.md) for the checks and known limitations.

Live responses depend on sampling and the configured providers, so fresh runs can differ in words and pixels. Exact export comparisons should hold composition plans, images, fonts, and renderer versions fixed.

To compare a reference checkout with this checkout using completed runs:

```bash
python scripts/verify_render_parity.py --path all \
  --source-root /path/to/RL4PPT \
  --text-run /path/to/saved-text-first-run \
  --topology-run /path/to/saved-topology-first-run
```

Each saved run must contain `assembly/deck_compilation.json` with valid absolute plan and asset paths. The check makes no model calls. It compares slide XML, native text/fonts/geometry, text-fit reports, and pixels from real PDF renders, and saves a paired montage. Equality establishes export parity for those fixed inputs; existing quality issues remain visible in both results. This optional comparison requires the reference checkout and its saved runs.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing model or CUDA failure | Use the live preflight; check `DIFFUSION_MODEL_PATH`, visible GPU IDs, driver compatibility, and the active Conda environment |
| API authentication/model error | Check endpoint, supported model IDs, and `APIMART_API_KEY` in the invoking shell |
| Missing Chromium or Node | Rerun setup in this directory and reactivate `rl4ppt-final` |
| Font substitution or changed line breaks | Install the declared Noto fonts and rerun preflight; inspect PDF font diagnostics |
| Missing PDF | Check host LibreOffice and the assembly render report |
| Completed run marked `degraded` | Read stage warnings and fit/render reports; repair or resume the affected stage with its existing artifacts |
| Output directory locked | Let the active writer finish; use separate directories for independent decks |
| Ambiguous image request | Reconcile the saved remote task before any resubmission |
