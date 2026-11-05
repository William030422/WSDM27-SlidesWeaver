#!/usr/bin/env python3
"""Record checksums for pipeline source files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTES = {"text_first": "final-demo-diffusion-planning-reuse", "topology_first": "engineering-fix-clean"}
EXCLUDED = {"__pycache__", "node_modules", ".pytest_cache", ".ruff_cache", ".git", "outputs"}
RUNTIME = {"integration", "src", "reused", "pptx_compiler"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="optional reference checkout for source comparison")
    args = parser.parse_args()
    records = []
    for route, origin in ROUTES.items():
        folder = ROOT / "paths" / route
        for path in sorted(folder.rglob("*")):
            relative = path.relative_to(folder)
            if any(part in EXCLUDED or part.startswith(".env") for part in relative.parts) or not path.is_file() or path.suffix == ".pyc":
                continue
            record = {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}
            source = args.source_root / origin / relative if args.source_root else None
            if source is not None and source.is_file():
                record.update(source=(Path(origin) / relative).as_posix(), source_sha256=digest(source))
            is_runtime = relative.parts[0] in RUNTIME or (len(relative.parts) == 1 and path.suffix in {".py", ".mjs"})
            record["runtime"] = is_runtime
            records.append(record)
    manifest = {"schema_version": 1, "summary": {"files": len(records)}, "files": records}
    (ROOT / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
