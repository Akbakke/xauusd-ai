from pathlib import Path

import numpy as np
import pandas as pd

from gx1.scripts.audit_model_native_direction_evidence_v1 import (
    FORBIDDEN_PREDICTION_COLUMNS,
    _chosen_side,
    _selected,
)


def _canonical_predictions() -> pd.DataFrame:
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ],
        dtype=np.float64,
    )
    direction_logits = np.log(probabilities)
    public_logits = np.column_stack(
        [np.max(direction_logits[:, :2], axis=1), direction_logits[:, 2]]
    )
    return pd.DataFrame(
        {
            "selection_score_mode": ["model_direction_argmax"] * 3,
            "pred_direction": [0, 1, 2],
            "direction_logits": direction_logits.tolist(),
            "public_trade_flat_decision_logits": public_logits.tolist(),
            "p_long": probabilities[:, 0],
            "p_short": probabilities[:, 1],
            "p_flat": probabilities[:, 2],
            "public_trade_probability": [8 / 9, 8 / 9, 1 / 9],
            "public_flat_probability": [1 / 9, 1 / 9, 8 / 9],
            "public_trade_flat_margin": [2.0, 2.0, -2.0],
            "public_trade_flat_hard_decision": [0, 0, 1],
        }
    )


def test_selection_is_only_final_model_direction_argmax_surface() -> None:
    frame = _canonical_predictions()

    assert _chosen_side(frame).tolist() == [0, 1, 2]
    assert _selected(frame).tolist() == [True, True, False]


def test_audit_rejects_every_retired_anchor_residual_surface() -> None:
    assert FORBIDDEN_PREDICTION_COLUMNS == {
        "anchor_logits",
        "anchor_gate",
        "anchor_logits_long_minus_short",
        "anchor_gate_long_minus_short",
        "delta_logits",
        "delta_logits_long_minus_short",
    }


def test_source_is_seq513_strict_and_writes_no_duplicate_latest_or_csv() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "gx1/scripts/audit_model_native_direction_evidence_v1.py"
    ).read_text(encoding="utf-8")

    assert "require_model_native_signal_contract(" in source
    assert "MODEL_NATIVE_SIGNAL_DIM" in source
    assert "direction_logits" in source
    assert "write_immutable_json_event(" in source
    assert "atomic_write_parquet_immutable(" in source
    assert "2026-07-08" not in source
    assert "SMART_DIRECTION_CONTRIBUTION" not in source
    assert "_latest.json" not in source
    assert ".to_csv(" not in source
