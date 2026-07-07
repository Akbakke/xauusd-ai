import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_fix_r5_2_label_objective_first_v1 import OUTPUT_FILES, materialize
from gx1.scripts.train_monday_r6_explicit_rebuild_from_rehydrated_contract_v1 import (
    R5_2_BAD_PROB,
    R5_2_RUNNER_PROB,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _frames(row_count: int = 1914) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for idx in range(row_count):
        is_strong_bad = idx < 130
        is_tail = 130 <= idx < 330
        is_ambiguous = 330 <= idx < 460
        is_protect = 460 <= idx < 920
        label_bad = is_strong_bad or is_ambiguous
        tail = is_tail
        high_mfe = is_ambiguous or is_protect
        take_ok = is_protect
        base_selected = idx < 20
        rows.append(
            {
                "run_id": "fixture_monday",
                "candidate_uid": f"candidate_{idx:04d}",
                "trade_uid": f"trade_uid_{idx:04d}",
                "trade_id": f"trade_{idx:04d}",
                "decision_timestamp": f"2026-01-05T13:{idx % 60:02d}:00Z",
                "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
                "split_scope_v1": "TRAIN",
                "label_should_not_take_v1": label_bad,
                "tail_10_50_mfe_v1": tail,
                "r6_label_risky_allow_v1": False,
                "r5_2_label_bad_blocker_v1": not is_ambiguous,
                "take_was_ok_v1": take_ok,
                "fifty_plus_mfe_v1": high_mfe,
                "hundred_plus_mfe_v1": False,
                "two_hundred_plus_mfe_v1": False,
                "strongest_winner_path_v1": idx in {700, 701},
                "r6_label_repaired_165_like_runner_v1": idx == 702,
                "r6_label_runner_near_miss_v1": idx == 703,
                "peak_mfe_bps_v1": 75.0 if high_mfe else (25.0 if tail else 7.5),
                "mae_abs_bps_v1": 55.0 if is_strong_bad else 12.0,
                "baseline_realized_pnl_bps_v1": -5.0 if is_strong_bad else 20.0,
                "pred__entry_r5_should_not_take__prob_true_v1": 0.86 if (is_strong_bad or base_selected) else 0.30,
                "pred__entry_r5_immediate_MAE_risk__prob_true_v1": 0.80 if (is_strong_bad or base_selected) else 0.25,
                "pred__entry_r5_runner_protect__prob_true_v1": 0.15 if not high_mfe else 0.70,
                "pred__entry_r5_tail_control_10_50_risk__prob_true_v1": 0.82 if tail else 0.20,
                "pred__entry_r5_bad_trade_but_high_runner_risk__prob_true_v1": 0.15,
                "pred__entry_r5_strong_trade_candidate__prob_true_v1": 0.10,
                "pred__entry_r5_take_was_ok__prob_true_v1": 0.80 if take_ok else 0.10,
                "r5_1_bad_blocker_score_v1": 0.86 if (is_strong_bad or base_selected) else 0.30,
                "r5_1_runner_guard_score_v1": 0.10 if not high_mfe else 0.70,
                R5_2_BAD_PROB: 0.45 if base_selected else (0.20 if high_mfe else 0.30),
                R5_2_RUNNER_PROB: 0.10 if base_selected else (0.70 if high_mfe else 0.40),
                "pred__entry_r6_bad_risk__prob_true_v1": 0.90 if is_strong_bad else 0.20,
                "pred__entry_r6_risky_allow__prob_true_v1": 0.20,
                "pred__entry_r6_tail_control_10_50__prob_true_v1": 0.90 if tail else 0.20,
                "pred__entry_r6_runner_protector__prob_true_v1": 0.80 if high_mfe else 0.10,
                "pred__entry_r6_batch04_blindspot__prob_true_v1": 0.10,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.copy(), frame.copy()


def test_fix_r5_2_label_objective_materializes_contract_and_gate(tmp_path: Path) -> None:
    score_dir = tmp_path / "score"
    r6_dir = tmp_path / "r6"
    investigation_dir = tmp_path / "investigation"
    asset_reuse_dir = tmp_path / "asset_reuse"
    score_dir.mkdir()
    r6_dir.mkdir()
    investigation_dir.mkdir()
    asset_reuse_dir.mkdir()
    score, r6 = _frames()
    score.to_parquet(score_dir / "monday_r6_foundation_score_frame_v1.parquet", index=False)
    r6.to_parquet(r6_dir / "monday_r6_on_foundation_scores_training_frame_v1.parquet", index=False)
    _write_json(
        score_dir / "score_rebuild_summary_v1.json",
        {"r5_2_selected_policy_v1": {"params_v1": {"bad_threshold_v1": 0.35, "runner_max_v1": 0.20}}},
    )
    _write_json(investigation_dir / "summary_v1.json", {"missed_rows_traced_v1": 390})

    out = tmp_path / "out"
    summary = materialize(
        reports_root=tmp_path,
        output_dir=out,
        v3_score_dir=score_dir,
        v3_r6_dir=r6_dir,
        investigation_dir=investigation_dir,
        asset_reuse_dir=asset_reuse_dir,
    )

    assert summary["decision_v1"] == "R5_2_LABEL_OBJECTIVE_READY_FOR_TRUE_REBUILD_SPEC"
    assert summary["next_action_v1"] == "BUILD_TRUE_R5_2_REBUILD_RUNNER_SPEC_NEXT"
    assert summary["label_table_rows_v1"] == 1914
    assert summary["training_started_v1"] is False
    assert summary["no_new_baseline_v1"] is True
    assert summary["ambiguous_high_mfe_bad_positive_count_v1"] == 0
    for filename in OUTPUT_FILES.values():
        assert (out / filename).exists()

    label_table = pd.read_csv(out / "r5_2_pocket_label_table_v1.csv")
    assert {
        "STRONG_BAD_BLOCK_TARGET",
        "TAIL_CONTROL_TARGET",
        "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD",
        "RUNNER_PROTECT_TARGET",
    }.issubset(set(label_table["new_r5_2_label_bucket_v1"]))
    ambiguous = label_table[label_table["new_r5_2_label_bucket_v1"] == "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD"]
    assert not ambiguous["bad_eligibility_target_v1"].astype(bool).any()

    audit = pd.read_csv(out / "consistency_audit_v1.csv")
    assert set(audit["status_v1"]) == {"PASS"}
