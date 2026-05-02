import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_extend_r5_2_base_contract_v3_only_if_safe_v1 import OUTPUT_FILES, materialize
from gx1.scripts.train_monday_r6_foundation_score_rebuild_v1 import (
    R5_2_BASE_MEMBERSHIP_CONTRACT_V2,
    R5_2_BASE_MEMBERSHIP_CONTRACT_V3,
    R5_2_BAD_PROB,
    R5_2_RUNNER_PROB,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _frame(*, v3: bool) -> pd.DataFrame:
    rows = []
    for idx in range(86):
        v2_selected = idx < 78
        v3_added = v3 and 78 <= idx < 82
        should = idx < 82
        v3_candidate = 78 <= idx < 82
        rows.append(
            {
                "run_id": "run_0",
                "candidate_uid": f"c{idx:03d}",
                "trade_uid": f"t{idx:03d}",
                "trade_id": f"trade{idx:03d}",
                "decision_timestamp": f"2026-01-01T12:{idx % 60:02d}:00Z",
                "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
                "label_should_not_take_v1": should,
                "tail_10_50_mfe_v1": idx < 49 or idx in {78, 79},
                "take_was_ok_v1": not should,
                "fifty_plus_mfe_v1": False,
                "hundred_plus_mfe_v1": False,
                "two_hundred_plus_mfe_v1": False,
                "strongest_winner_path_v1": False,
                "is_repaired_165_v1": False,
                "r6_label_repaired_165_like_runner_v1": False,
                "r6_label_runner_near_miss_v1": False,
                "pred__entry_r5_should_not_take__prob_true_v1": 0.9 if (v2_selected or v3_candidate) else 0.1,
                "pred__entry_r5_immediate_MAE_risk__prob_true_v1": 0.2,
                "pred__entry_r5_runner_protect__prob_true_v1": 0.2,
                "pred__entry_r5_tail_control_10_50_risk__prob_true_v1": 0.2,
                "pred__entry_r5_bad_trade_but_high_runner_risk__prob_true_v1": 0.2,
                "r5_1_bad_blocker_score_v1": 0.9 if (v2_selected or v3_candidate) else 0.1,
                "r5_1_runner_guard_score_v1": 0.2,
                R5_2_BAD_PROB: 0.4 if (v2_selected or v3_candidate) else 0.1,
                R5_2_RUNNER_PROB: 0.2,
                "r5_2_selected_candidate__block_v1": v2_selected or v3_added,
                "selected_candidate_block_v1": v2_selected,
            }
        )
    return pd.DataFrame(rows)


def _seed_score_dir(path: Path, *, v3: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _frame(v3=v3).to_parquet(path / "monday_r6_foundation_score_frame_v1.parquet", index=False)
    policy = {
        "base_membership_active_contract_id_v1": (
            R5_2_BASE_MEMBERSHIP_CONTRACT_V3["contract_id_v1"] if v3 else R5_2_BASE_MEMBERSHIP_CONTRACT_V2["contract_id_v1"]
        ),
        "v3_contract_applied_v1": v3,
        "metrics_v1": {"bad_blocks_v1": 82 if v3 else 78, "tail_help_v1": 51 if v3 else 49},
    }
    _write_json(path / "score_rebuild_summary_v1.json", {"r5_2_selected_policy_v1": policy})
    _write_json(
        path / "summary_v1.json",
        {
            "decision_v1": "MONDAY_R5_R5_1_R5_2_SCORE_REBUILD_COMPLETED",
            "explicit_score_rebuild_flag_v1": True,
            "r6_heads_trained_v1": False,
        },
    )


def test_v3_materializer_reports_tiny_safe_rebuild(tmp_path: Path) -> None:
    v2_dir = tmp_path / "v2"
    v3_dir = tmp_path / "v3"
    r6_dir = tmp_path / "r6"
    r6_dir.mkdir()
    _seed_score_dir(v2_dir, v3=False)
    _seed_score_dir(v3_dir, v3=True)
    r6 = _frame(v3=False)
    r6["asof_runner_guard_v1"] = False
    r6.to_parquet(r6_dir / "monday_r6_on_foundation_scores_training_frame_v1.parquet", index=False)
    out = tmp_path / "out"

    summary = materialize(reports_root=tmp_path, output_dir=out, v2_score_dir=v2_dir, v3_score_dir=v3_dir, v2_r6_dir=r6_dir)

    assert summary["decision_v1"] == "R5_2_V3_ONLY_TINY_UPLIFT"
    assert summary["v3_implemented_v1"] is True
    assert summary["v3_score_rebuild_bad_blocks_v1"] == 82
    assert summary["v3_score_rebuild_tail_help_v1"] == 51
    assert summary["best_safe_v3_bad_uplift_v1"] == 4
    assert summary["best_safe_v3_tail_uplift_v1"] == 2
    assert summary["r6_retrain_run_v1"] is False
    for filename in OUTPUT_FILES.values():
        assert (out / filename).exists()
    gate = json.loads((out / "v3_gate_v1.json").read_text())
    assert gate["checks_v1"]["contract_ok_v1"] is True
    assert gate["checks_v1"]["safety_ok_v1"] is True
    audit = pd.read_csv(out / "consistency_audit_v1.csv")
    assert set(audit["status_v1"]) == {"PASS"}
