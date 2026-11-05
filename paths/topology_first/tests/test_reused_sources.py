from __future__ import annotations

import unittest
import ast
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
REFERENCES = json.loads((ROOT / "tests/reference_sources.json").read_text())


def _ast_hash(node: ast.AST) -> str:
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


class ReusedSourceIntegrityTests(unittest.TestCase):
    def test_v9_step3_and_step4_planning_modules_are_removed(self) -> None:
        local = ROOT / "src" / "v9_deck_planning"
        removed = (
            "step3_models.py",
            "step3_pipeline.py",
            "step3_prompts.py",
            "step3_rendering.py",
            "step4_composition.py",
            "step4_models.py",
            "step4_pipeline.py",
        )
        self.assertTrue(all(not (local / name).exists() for name in removed))

    def test_unmodified_v9_modules_are_byte_identical(self) -> None:
        local = ROOT / "src" / "v9_deck_planning"
        unmodified_modules = {
            "json_utils.py",
            "style_models.py",
            "style_rendering.py",
        }
        for name in unmodified_modules:
            reference = REFERENCES["files"][f"src/v9_deck_planning/{name}"]
            self.assertEqual(
                hashlib.sha256((local / name).read_bytes()).hexdigest(),
                reference["sha256"],
                name,
            )

    def test_apimart_patch_is_limited_to_typed_task_outcomes(self) -> None:
        """Keep transport/payload reuse checked while documenting our adapter patch."""
        local = ROOT / "src/v9_deck_planning/apimart.py"
        patched = ast.parse(local.read_text())
        classes = lambda tree: {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        patched_classes = classes(patched)
        reference = REFERENCES["apimart"]
        self.assertEqual(set(patched_classes) - set(reference["classes"]),
                         {"ApiMartImageTaskFailure", "ApiMartTaskTimeout"})
        methods = lambda node: {item.name: _ast_hash(item)
                                for item in node.body if isinstance(item, ast.FunctionDef)}
        old_methods = reference["methods"]
        new_methods = methods(patched_classes["ApiMartClient"])
        self.assertEqual(set(old_methods), set(new_methods))
        self.assertEqual({name for name in old_methods if old_methods[name] != new_methods[name]}, {"poll_task"})
        functions = lambda tree: {node.name: _ast_hash(node)
                                  for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(reference["functions"], functions(patched))
        poll_method = next(node for node in patched_classes["ApiMartClient"].body
                           if isinstance(node, ast.FunctionDef) and node.name == "poll_task")
        self.assertIn("ApiMartImageTaskFailure", ast.dump(poll_method))
        self.assertIn("ApiMartTaskTimeout", ast.dump(poll_method))
        provenance = (ROOT / "SOURCE_MAP.md").read_text()
        self.assertIn("src/v9_deck_planning/apimart.py", provenance)
        self.assertIn("typed task failures", provenance)

    def test_reused_experiment_files_are_byte_identical(self) -> None:
        for path, reference in REFERENCES["files"].items():
            if not path.startswith("reused/"):
                continue
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                             reference["sha256"], path)

    def test_diffusion_runner_preserves_inference_and_uses_portable_model_path(self) -> None:
        local = ROOT / "reused/diffusion/generate_slides.py"
        tree = ast.parse(local.read_text())
        model_path = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "MODEL_PATH" for t in node.targets))
        tree.body = [node for node in tree.body if node is not model_path and not (
            isinstance(node, ast.Import) and [alias.name for alias in node.names] == ["os"])]
        self.assertEqual(_ast_hash(tree), REFERENCES["files"]["reused/diffusion/generate_slides.py"]["inference_ast_sha256"])
        code = compile(ast.Module(body=[model_path], type_ignores=[]), str(local), "exec")
        with patch.dict(os.environ, {"DIFFUSION_MODEL_PATH": "~/custom-model"}):
            namespace = {"Path": Path, "os": os}
            exec(code, namespace)
            self.assertEqual(namespace["MODEL_PATH"], Path.home() / "custom-model")
        with patch.dict(os.environ, {}, clear=True):
            namespace = {"Path": Path, "os": os}
            exec(code, namespace)
            self.assertEqual(namespace["MODEL_PATH"], Path.home() / "models" / "Tongyi-MAI_Z-Image-Turbo")

    def test_text_placement_prompts_define_bounded_refinement_contract(self) -> None:
        local = ROOT / "reused" / "text_placement" / "prompts"
        prompts = {
            path.name: path.read_text(encoding="utf-8") for path in local.glob("*.md")
        }
        self.assertEqual(
            set(prompts),
            {
                "01_layout.md",
                "02_move.md",
                "03_accept.md",
                "04_badge.md",
                "05_content.md",
            },
        )
        self.assertTrue(all("Model:" not in prompt for prompt in prompts.values()))
        self.assertIn("move_region_ids", prompts["01_layout.md"])
        self.assertIn("search_bbox_px", prompts["02_move.md"])
        self.assertIn("legal_position_ranges_px", prompts["02_move.md"])
        self.assertIn("listed label", prompts["03_accept.md"])
        self.assertIn("at most one", prompts["04_badge.md"])
        self.assertIn("3-14 words", prompts["05_content.md"])


if __name__ == "__main__":
    unittest.main()
