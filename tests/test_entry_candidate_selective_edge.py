from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
)
from gx1.scripts.evaluate_entry_candidate_selective_edge_v1 import (
    _canonical_live_decision_evidence,
    _concatenate_evidence_chunks,
    _realized_net_policy_pnl,
    _require_entry_q_ssot,
    _require_selective_edge_stage_split,
    _selection_sort_column,
)


def test_entry_q_is_the_only_decision_surface_and_ties_fail_closed() -> None:
    q = torch.tensor(
        [[3.0, -1.0, 0.0], [-2.0, 4.0, 0.0], [-1.0, -2.0, 0.0]],
        dtype=torch.float32,
    )
    assert torch.equal(_require_entry_q_ssot({"entry_action_q_bps": q}), q)
    with pytest.raises(RuntimeError, match="forbidden legacy"):
        _require_entry_q_ssot(
            {"entry_action_q_bps": q, "anchor_logits": torch.zeros_like(q)}
        )
    with pytest.raises(RuntimeError, match="no unique top action"):
        _require_entry_q_ssot(
            {"entry_action_q_bps": torch.tensor([[1.0, 1.0, 0.0]])}
        )


def test_live_decision_evidence_contains_raw_q_argmax_and_no_probabilities() -> None:
    q = torch.tensor(
        [[3.0, -1.0, 0.0], [-2.0, 4.0, 0.0], [-1.0, -2.0, 0.0]],
        dtype=torch.float32,
    )
    evidence = _canonical_live_decision_evidence(
        {"entry_action_q_bps": q}
    )
    assert set(evidence) == {
        "entry_action_q_bps",
        "entry_action_q_margin_bps",
        "model_direction_index",
        "edge_score",
        "selection_score",
    }
    assert evidence["model_direction_index"].tolist() == [0, 1, 2]
    assert np.array_equal(evidence["entry_action_q_bps"], q.numpy())
    assert np.all(evidence["entry_action_q_margin_bps"] > 0.0)


def test_realized_net_policy_pnl_uses_action_side_and_flat_zero() -> None:
    frame = pd.DataFrame(
        {
            "pred_direction": [0, 1, 2],
            "realized_net_long_pnl_bps": [5.0, 99.0, 99.0],
            "realized_net_short_pnl_bps": [99.0, -3.0, 99.0],
        }
    )
    assert _realized_net_policy_pnl(frame).tolist() == [5.0, -3.0, 0.0]
    with pytest.raises(RuntimeError, match="lacks executable net OOS evidence"):
        _realized_net_policy_pnl(frame.drop(columns=["realized_net_long_pnl_bps"]))


def test_selection_sort_is_raw_q_action_value_and_mode_bound() -> None:
    frame = pd.DataFrame(
        {
            "selection_score": [3.0, 2.0],
            "selection_score_mode": [MODEL_DIRECTION_SELECTION_MODE] * 2,
        }
    )
    assert _selection_sort_column(frame) == "selection_score"
    frame.loc[0, "selection_score_mode"] = "probability"
    with pytest.raises(RuntimeError, match="direction mode mismatch"):
        _selection_sort_column(frame)


def test_evidence_chunks_require_exact_row_count_and_shape() -> None:
    combined = _concatenate_evidence_chunks(
        {
            "entry_action_q_bps": [
                np.ones((2, 3), dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32),
            ]
        },
        expected_rows=3,
    )
    assert combined["entry_action_q_bps"].shape == (3, 3)
    with pytest.raises(RuntimeError, match="row mismatch"):
        _concatenate_evidence_chunks(
            {"entry_action_q_bps": [np.ones((2, 3), dtype=np.float32)]},
            expected_rows=3,
        )


@pytest.mark.parametrize(
    ("stage", "split"),
    (("validation_research", "val"), ("runtime_authoritative", "test")),
)
def test_stage_split_contract_is_exact(stage: str, split: str) -> None:
    _require_selective_edge_stage_split(
        evidence_stage=stage, split_spec=split
    )
    wrong = "test" if split == "val" else "val"
    with pytest.raises(RuntimeError, match="SELECTIVE_EDGE_STAGE_SPLIT_INVALID"):
        _require_selective_edge_stage_split(
            evidence_stage=stage, split_spec=wrong
        )
