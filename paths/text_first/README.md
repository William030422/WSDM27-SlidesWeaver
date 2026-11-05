# Text First

**Text-conditioned layout:** commit the copy, use diffusion for composition
inspiration, ground the layout in browser-validated HTML, then synthesize visual
assets and assemble editable text with selectable raster layers.

Use the [main README](../../README.md) and [running guide](../../docs/running.md)
for installation, configuration and complete examples. From the repository root:

```bash
python run.py --path text-first --input examples/feedback.json \
  --output-dir outputs/text_first_mock --mock-api --mock-diffusion
```

Stages: `step1`, `text_generation`, `step2`, `expression_logic`, `diffusion`,
`html_planning`, `html_compile`, `background`, `assembly`.

The [module guide](SOURCE_MAP.md) describes planning, geometry extraction,
asset generation, and assembly. Use the top-level launcher for configuration
and process isolation.
