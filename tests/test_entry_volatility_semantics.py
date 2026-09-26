from __future__ import annotations

import ast
from pathlib import Path


from gx1.contracts.entry_model_native_input_normalization_v1 import (
    FIT_POPULATION,
    TRANSFORM,
)
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_BASE_FIELDS


def test_raw_rvol_is_train_normalized_and_no_active_feature_reintroduces_tanh() -> None:
    assert "rvol_20" in MODEL_NATIVE_BASE_FIELDS
    assert "train_only" in TRANSFORM
    assert FIT_POPULATION == "unique_physical_train_rows_entry_exit_union_v2"

    features_root = Path(__file__).resolve().parents[1] / "gx1" / "features"
    violations: list[str] = []
    for source_path in sorted(features_root.glob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            function_name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr
                if isinstance(function, ast.Attribute)
                else ""
            )
            if function_name not in {"tanh", "_tanh"}:
                continue
            if "rvol_20" in ast.unparse(node):
                violations.append(
                    f"{source_path.relative_to(features_root.parent.parent)}:"
                    f"{node.lineno}"
                )
    assert violations == []
