from __future__ import annotations

from pathlib import Path

from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
)
from gx1.models.entry_v10.entry_v10_bundle import (
    _ENTRY_HEAD_STATE_KEYS,
    _MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = ROOT / "gx1"
PRODUCTION_SOURCE_ROOTS = (PRODUCTION_ROOT, ROOT / "scripts")


def test_handcrafted_tf_agreement_surface_is_fully_retired() -> None:
    assert not (PRODUCTION_ROOT / "features" / "tf_agreement_score.py").exists()

    forbidden = (
        "tf_agreement_score",
        "head_tf_agreement",
        "y_tf_agreement_score",
        "tf_agreement_logit",
        "tf_agreement_pred",
    )
    offenders: list[str] = []
    for source_root in PRODUCTION_SOURCE_ROOTS:
        for path in source_root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in source:
                    offenders.append(f"{path.relative_to(ROOT)}:{token}")
    assert offenders == []

    assert "tf_agreement" not in _ENTRY_HEAD_STATE_KEYS
    assert "tf_agreement" not in _MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS
    assert all(
        "tf_agreement" not in field
        for field in MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS
    )


def test_position_size_remains_execution_output_without_direction_authority() -> None:
    assert "position_size" in _ENTRY_HEAD_STATE_KEYS
    assert "position_size" in _MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS
    assert "position_size_logit" in MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS
    assert "position_size_pred" in MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS

    model_source = (
        PRODUCTION_ROOT
        / "models"
        / "entry_v10"
        / "entry_v10_ctx_hybrid_transformer.py"
    ).read_text(encoding="utf-8")
    assert "self.head_position_size" in model_source
    assert '"position_size_logit": self.head_position_size(z)' in model_source
