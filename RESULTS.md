# Validation

Run both regression suites from the repository root:

```bash
python scripts/run_tests.py --path all
python scripts/verify_provenance.py
```

The tests cover model-response handling, geometry, text fitting, asset extraction,
page ownership, recovery, and native PPTX export. The source integrity check
compares files with the recorded checksums.

For an end-to-end check without model calls, run either path with
`--mock-api --mock-diffusion` as shown in the [running guide](docs/running.md).
Both paths compile the PPTX and render it through LibreOffice to PDF. Inspect
the deck preview, `text_fit_report.json`, and the route's render diagnostics.

## Reference checks

The recorded regression and fixed-plan export checks are available as JSON:

- [Regression suites](docs/validation/tests.json)
- [Source integrity](docs/validation/provenance.json)
- [Fixed-plan render comparison](docs/validation/render_parity.json)
- [Offline export checks](docs/validation/smoke.json)

The render comparison uses the same accepted plans, images, fonts and rendering
stack for both exports. It checks native slide XML, text fitting, and PDF page
pixels. The script saves its reports and paired previews under `outputs/parity/`;
these generated files are excluded from version control.

## Interpreting results

A successful export can still contain quality findings. Check `quality_status`
and the stage warnings alongside `status`. A `degraded` result retains the deck
and reports the fallback or unresolved problem.

The recorded Topology First comparison includes one `b_intro` overflow on
slide 2; identical rendering does not resolve it. The default Topology First
mock also reports `b_semantic_overview` overflow. These findings remain in the
reports.

Mock checks validate the software path with synthetic content and images.
They do not establish live image quality or factual correctness. Live model
responses can vary; GPU preflight confirms availability without running
inference. Visual inspection of the final deck remains necessary.
