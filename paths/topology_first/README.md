# Topology First

**Layout-conditioned text:** diffusion proposes the composition; grounded
topology and capacities guide copy generation; a joint visual canvas is combined
with native editable text and bounded content-unit fitting/review.

Use the [main README](../../README.md) and [running guide](../../docs/running.md)
for installation, configuration and complete examples. From the repository root:

```bash
python run.py --path topology-first --input examples/feedback.json \
  --output-dir outputs/topology_first_mock --mock-api --mock-diffusion
```

Stages: `step1`, `step2`, `slide_spec`, `diffusion`, `topology`, `region_content`,
`background`, `assembly`.

The [module guide](SOURCE_MAP.md) describes topology extraction, content
generation, fitting, and assembly. Use the top-level launcher for configuration
and process isolation.
