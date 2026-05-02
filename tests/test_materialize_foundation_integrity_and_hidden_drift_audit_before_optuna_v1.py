import json
from pathlib import Path

import pandas as pd

import gx1.scripts.materialize_foundation_integrity_and_hidden_drift_audit_before_optuna_v1 as audit
from gx1.scripts.run_r5_2_objective_v3_parallel_rebuild_runner_v1 import V3_SCORE_FIELDS


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _score_frame(rows: int = 1914) -> pd.DataFrame:
    data = {
        "candidate_uid": [f"c{i}" for i in range(rows)],
        "trade_uid": [f"t{i}" for i in range(rows)],
        "trade_id": [f"T{i}" for i in range(rows)],
        "decision_timestamp": [f"2026-01-01T00:{i % 60:02d}:00Z" for i in range(rows)],
        "run_id": [f"W{i % 58}" for i in range(rows)],
        "calendar_quarantine_status_v1": ["ACTIVE_CANDIDATE" if i < 1852 else "QUARANTINED" for i in range(rows)],
        "label_should_not_take_v1": [i % 7 == 0 for i in range(rows)],
        "tail_10_50_mfe_v1": [i % 11 == 0 for i in range(rows)],
        "fifty_plus_mfe_v1": [i % 101 == 0 for i in range(rows)],
        "hundred_plus_mfe_v1": [False] * rows,
        "two_hundred_plus_mfe_v1": [False] * rows,
        "strongest_winner_path_v1": [False] * rows,
        "r6_label_repaired_165_like_runner_v1": [False] * rows,
        "r6_label_runner_near_miss_v1": [False] * rows,
        "r5_2_label_high_mfe_tail_risk_ambiguous_v1": [False] * rows,
        "r5_2_label_runner_protect_v1": [False] * rows,
    }
    for idx in range(88):
        data[f"as_of_feature_{idx:03d}_v1"] = [float((i + idx) % 13) for i in range(rows)]
    for name in [
        "pred__entry_r5_should_not_take__prob_true_v1",
        "pred__entry_r5_immediate_MAE_risk__prob_true_v1",
        "pred__entry_r5_runner_protect__prob_true_v1",
        "pred__entry_r5_tail_control_10_50_risk__prob_true_v1",
        "pred__entry_r5_wait_or_delay_advisory__prob_true_v1",
    ]:
        data[name] = [0.1 + (i % 10) / 20.0 for i in range(rows)]
    data["r5_1_bad_blocker_score_v1"] = [0.2 + (i % 5) / 10.0 for i in range(rows)]
    data["r5_1_runner_guard_score_v1"] = [0.2 for _ in range(rows)]
    data["pred__entry_r5_2_bad_blocker__prob_true_v1"] = [0.3 for _ in range(rows)]
    data["pred__entry_r5_2_runner_protector__prob_true_v1"] = [0.1 for _ in range(rows)]
    data["r5_2_v2_final_base_membership"] = [i % 17 == 0 for i in range(rows)]
    return pd.DataFrame(data)


def _label_table(score: pd.DataFrame) -> pd.DataFrame:
    buckets = (
        ["STRONG_BAD_BLOCK_TARGET"] * 127
        + ["TAIL_CONTROL_TARGET"] * 198
        + ["AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD"] * 130
        + ["RUNNER_PROTECT_TARGET"] * 462
        + ["IGNORE_OR_MONITOR_ONLY"] * 997
    )
    label = score[["candidate_uid", "trade_uid", "trade_id", "decision_timestamp", "run_id", "calendar_quarantine_status_v1"]].copy()
    label["new_r5_2_label_bucket_v1"] = buckets
    label["bad_eligibility_target_v1"] = label["new_r5_2_label_bucket_v1"].eq("STRONG_BAD_BLOCK_TARGET")
    label["tail_eligibility_target_v1"] = label["new_r5_2_label_bucket_v1"].eq("TAIL_CONTROL_TARGET")
    label["runner_protect_target_v1"] = label["new_r5_2_label_bucket_v1"].eq("RUNNER_PROTECT_TARGET")
    label["ambiguous_high_mfe_monitor_v1"] = label["new_r5_2_label_bucket_v1"].eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD")
    label["hundred_plus_mfe_v1"] = False
    label["two_hundred_plus_mfe_v1"] = False
    label["strongest_winner_path_v1"] = False
    label["r6_label_repaired_165_like_runner_v1"] = False
    label["label_should_not_take_v1"] = score["label_should_not_take_v1"]
    label["tail_10_50_mfe_v1"] = score["tail_10_50_mfe_v1"]
    return label


def _base_inputs(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    score = _score_frame()
    label = _label_table(score)
    score_path = tmp_path / "score.parquet"
    label_path = tmp_path / "label.csv"
    score.to_parquet(score_path, index=False)
    label.to_csv(label_path, index=False)
    (tmp_path / "features.csv").write_text("feature_family_v1,status_v1\nAS_OF,REUSE_NOW\n", encoding="utf-8")
    _write_json(tmp_path / "foundation.json", {"row_count_v1": 1914, "as_of_column_count_v1": 109})
    _write_json(tmp_path / "lock.json", {})
    return score, label, score_path, label_path


def _prediction(score: pd.DataFrame) -> pd.DataFrame:
    pred = score[["candidate_uid", "trade_uid", "decision_timestamp", "trade_id", "run_id", "label_should_not_take_v1", "tail_10_50_mfe_v1"]].copy()
    for column in V3_SCORE_FIELDS:
        pred[column] = 0.2
    pred["r5_2_v3_base_membership_pre_veto"] = [i < 10 for i in range(len(pred))]
    pred["r5_2_v3_hard_protection_veto"] = False
    pred["r5_2_v3_final_base_membership"] = [i < 10 for i in range(len(pred))]
    return pred


def _write_v3_root(
    tmp_path: Path,
    root: Path,
    *,
    valid_provenance: bool,
    source_mismatch: bool = False,
    in_sample_flag: bool = False,
    membership_overlap: bool = False,
    invalidated: bool = False,
) -> Path:
    score, _label, score_path, label_path = _base_inputs(tmp_path)
    variant_id = "variant"
    variant_dir = root / "variants" / variant_id
    variant_dir.mkdir(parents=True)
    pred = _prediction(score)
    pred.to_parquet(variant_dir / "prediction_view_v1.parquet", index=False)
    pd.DataFrame([{"variant_id_v1": variant_id, "variant_dir_v1": str(variant_dir)}]).to_csv(root / "v3_variant_outputs_index_v1.csv", index=False)
    _write_json(root / "best_v3_variant_downstream_r6_input_lock_v1.json", {"best_variant_id_v1": variant_id, "ready_for_downstream_r6_v1": False})
    _write_json(
        root / "manifest_v1.json",
        {
            "input_paths_v1": {
                "score_package_v1": str(score_path),
                "label_table_v1": str(label_path),
                "foundation_summary_v1": str(tmp_path / "foundation.json"),
                "feature_inventory_v1": str(tmp_path / "features.csv"),
                "downstream_r6_lock_v1": str(tmp_path / "lock.json"),
            }
        },
    )
    pd.DataFrame(
        [
            {
                "variant_id_v1": variant_id,
                "precision_decision_valid_v1": True,
                "worst_loso_decision_valid_v1": True,
            }
        ]
    ).to_csv(root / "v3_variant_eval_and_safety_gate_v1.csv", index=False)
    if not valid_provenance:
        return root

    provenance_rows = []
    fold_rows = []
    membership_rows = []
    groups = sorted(score["run_id"].astype(str).unique())
    fold_by_group = {group: idx % 5 + 1 for idx, group in enumerate(groups)}
    for field in V3_SCORE_FIELDS:
        for _, row in score.iterrows():
            group = str(row["run_id"])
            fold = fold_by_group[group]
            candidate_uid = "mismatched_candidate" if source_mismatch and field == V3_SCORE_FIELDS[0] and row.name == 0 else row["candidate_uid"]
            identity = {
                "candidate_uid": candidate_uid,
                "trade_uid": row["trade_uid"],
                "decision_timestamp": row["decision_timestamp"],
            }
            fold_rows.append(
                {
                    **identity,
                    "variant_id_v1": variant_id,
                    "score_field_v1": field,
                    "fold_id_v1": fold,
                    "group_key_v1": group,
                    "train_validation_membership_v1": "VALIDATION",
                }
            )
            provenance_rows.append(
                {
                    **identity,
                    "variant_id_v1": variant_id,
                    "score_field_v1": field,
                    "score_head_v1": field.replace("r5_2_v3_", "target_"),
                    "fold_id_v1": fold,
                    "group_key_v1": group,
                    "train_validation_membership_v1": "VALIDATION",
                    "source_model_fold_v1": f"{variant_id}:{field}:fold_{fold:02d}",
                    "model_source_identifier_v1": f"{variant_id}:{field}:fold_{fold:02d}",
                    "score_source_v1": "OOF",
                    "feature_matrix_hash_v1": "f" * 64,
                    "feature_matrix_columns_hash_v1": "c" * 64,
                    "label_table_hash_v1": "l" * 64,
                    "config_hash_v1": "a" * 64,
                    "seed_v1": 20260426 + fold,
                    "decision_valid_v1": not invalidated,
                    "decision_valid_status_v1": "VALID_FOR_PRE_OPTUNA_DECISIONING" if not invalidated else "INVALID_FOR_OPTUNA_DECISIONING",
                    "oof_provenance_status_v1": "PASS",
                    "row_was_in_training_for_source_model_v1": False,
                    "in_sample_score_used_v1": in_sample_flag and field == V3_SCORE_FIELDS[0] and row.name == 0,
                    "fallback_score_used_v1": False,
                    "synthetic_score_used_v1": False,
                }
            )
        for fold in sorted(set(fold_by_group.values())):
            for group in groups:
                membership = "VALIDATION" if fold_by_group[group] == fold else "TRAIN"
                membership_rows.append(
                    {
                        "variant_id_v1": variant_id,
                        "score_field_v1": field,
                        "fold_id_v1": fold,
                        "group_key_v1": group,
                        "source_model_fold_v1": f"{variant_id}:{field}:fold_{fold:02d}",
                        "train_validation_membership_v1": membership,
                    }
                )
                if membership_overlap and field == V3_SCORE_FIELDS[0] and fold == 1 and group == groups[0]:
                    membership_rows.append(
                        {
                            "variant_id_v1": variant_id,
                            "score_field_v1": field,
                            "fold_id_v1": fold,
                            "group_key_v1": group,
                            "source_model_fold_v1": f"{variant_id}:{field}:fold_{fold:02d}",
                            "train_validation_membership_v1": "TRAIN" if membership == "VALIDATION" else "VALIDATION",
                        }
                    )
    pd.DataFrame(provenance_rows).to_csv(root / "v3_oof_score_provenance_v1.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(root / "v3_oof_fold_assignment_v1.csv", index=False)
    pd.DataFrame(membership_rows).to_csv(root / "v3_train_validation_membership_v1.csv", index=False)
    _write_json(
        root / "v3_oof_score_source_manifest_v1.json",
        {
            "layer_name": "V3_OOF_SCORE_SOURCE_MANIFEST_V1",
            "score_source_v1": "OOF",
            "decision_valid_for_pre_optuna_v1": not invalidated,
            "oof_provenance_status_v1": "PASS",
            "scorefield_registry_v1": [
                {
                    "variant_id_v1": variant_id,
                    "score_field_v1": field,
                    "decision_valid_v1": not invalidated,
                    "score_source_v1": "OOF",
                    "oof_provenance_status_v1": "PASS",
                    "metric_denominator_decision_valid_required_v1": True,
                }
                for field in V3_SCORE_FIELDS
            ],
        },
    )
    return root


def _active_selection(path: Path, root: Path, *, policy: str = "EXPLICIT_ONLY_NO_LATEST_GLOB") -> Path:
    _write_json(
        path,
        {
            "contract": "ACTIVE_SCORE_ARTIFACT_SELECTION_V1",
            "decisioning_stage": "PRE_OPTUNA",
            "selection_policy": policy,
            "selected_artifacts": {"v3_oof_scores": str(root)},
            "requirements": {
                "oof_score_provenance_required": True,
                "fold_assignment_required": True,
                "score_source_manifest_required": True,
                "train_validation_membership_required": True,
                "metric_denominator_decision_valid_required": True,
            },
        },
    )
    return path


def _run_audit(tmp_path: Path, root: Path | None, output_name: str, **kwargs: object) -> dict:
    selected = root if root is not None else None
    return audit.materialize(
        reports_root=tmp_path,
        output_dir=tmp_path / output_name,
        v3_oof_dir=root or (tmp_path / "unused"),
        v3_in_sample_dir=root or (tmp_path / "unused_in_sample"),
        optuna_dir=tmp_path / "optuna",
        selected_v3_oof_artifact_root=selected,
        require_explicit_artifact_selection=True,
        reject_invalidated_decision_scorefields=True,
        fail_on_missing_oof_provenance=True,
        fail_on_invalid_metric_denominator=True,
        **kwargs,
    )


def test_old_invalid_v3_only_audit_fails(tmp_path: Path) -> None:
    old_root = _write_v3_root(tmp_path, tmp_path / "old_invalid_v3", valid_provenance=False)
    summary = _run_audit(tmp_path, old_root, "out_old_invalid")
    assert summary["foundation_clean_for_constrained_optuna_v1"] is False
    assert summary["decision_v1"] in {"FIX_SELECTED_V3_OOF_ARTIFACT_FIRST", "FIX_SCORE_PROVENANCE_FIRST"}


def test_old_invalid_v3_plus_new_valid_selected_passes_and_history_is_not_blocker(tmp_path: Path) -> None:
    old_root = tmp_path / "old_invalid_marker"
    _write_json(old_root / "v3_oof_score_provenance_reconstruction_or_invalidation_v1.json", {"reconstruction_status_v1": "INVALID_FOR_OPTUNA_DECISIONING"})
    new_root = _write_v3_root(tmp_path, tmp_path / "new_valid_v3", valid_provenance=True)
    summary = _run_audit(tmp_path, new_root, "out_valid")
    assert summary["decision_v1"] == "FOUNDATION_CLEAN_FOR_CONSTRAINED_OPTUNA"
    assert summary["selected_v3_oof_provenance_status_v1"] == "PASS"
    history = pd.read_csv(tmp_path / "out_valid" / "historical_invalid_v3_artifacts_v1.csv")
    assert "QUARANTINED_NOT_SELECTED_HISTORY_ONLY" in set(history["status_v1"])


def test_new_valid_exists_but_not_explicitly_selected_fails(tmp_path: Path) -> None:
    _write_v3_root(tmp_path, tmp_path / "new_valid_v3", valid_provenance=True)
    summary = audit.materialize(
        reports_root=tmp_path,
        output_dir=tmp_path / "out_missing_selection",
        v3_oof_dir=tmp_path / "new_valid_v3",
        require_explicit_artifact_selection=True,
    )
    assert summary["decision_v1"] == "EXPLICIT_ARTIFACT_SELECTION_REQUIRED"


def test_implicit_latest_glob_policy_fails(tmp_path: Path) -> None:
    root = _write_v3_root(tmp_path, tmp_path / "new_valid_v3", valid_provenance=True)
    contract = _active_selection(tmp_path / "selection.json", root, policy="LATEST_GLOB")
    summary = audit.materialize(
        reports_root=tmp_path,
        output_dir=tmp_path / "out_bad_policy",
        v3_oof_dir=root,
        active_score_artifact_selection=contract,
        require_explicit_artifact_selection=True,
        reject_invalidated_decision_scorefields=True,
        fail_on_missing_oof_provenance=True,
        fail_on_invalid_metric_denominator=True,
    )
    assert summary["decision_v1"] == "FIX_ACTIVE_SCORE_ARTIFACT_SELECTION_FIRST"


def test_selected_v3_missing_provenance_file_fails(tmp_path: Path) -> None:
    root = _write_v3_root(tmp_path, tmp_path / "missing_provenance_v3", valid_provenance=True)
    (root / "v3_oof_score_provenance_v1.csv").unlink()
    summary = _run_audit(tmp_path, root, "out_missing_provenance")
    assert summary["decision_v1"] == "FIX_SELECTED_V3_OOF_ARTIFACT_FIRST"


def test_selected_v3_provenance_row_source_mismatch_fails(tmp_path: Path) -> None:
    root = _write_v3_root(tmp_path, tmp_path / "mismatch_v3", valid_provenance=True, source_mismatch=True)
    summary = _run_audit(tmp_path, root, "out_mismatch")
    assert summary["decision_v1"] == "FIX_SELECTED_V3_OOF_ARTIFACT_FIRST"
    selected = json.loads((tmp_path / "out_mismatch" / "selected_v3_oof_artifact_audit_v1.json").read_text())
    assert any("ROW_SOURCE_MISMATCH" in reason for reason in selected["failure_reasons_v1"])


def test_in_sample_scorefield_marked_oof_fails(tmp_path: Path) -> None:
    root = _write_v3_root(tmp_path, tmp_path / "in_sample_v3", valid_provenance=True, in_sample_flag=True)
    summary = _run_audit(tmp_path, root, "out_in_sample")
    assert summary["decision_v1"] == "FIX_SELECTED_V3_OOF_ARTIFACT_FIRST"


def test_train_validation_overlap_in_membership_fails(tmp_path: Path) -> None:
    root = _write_v3_root(tmp_path, tmp_path / "overlap_v3", valid_provenance=True, membership_overlap=True)
    summary = _run_audit(tmp_path, root, "out_overlap")
    assert summary["decision_v1"] == "FIX_SELECTED_V3_OOF_ARTIFACT_FIRST"


def test_selected_artifact_marked_invalid_for_optuna_fails(tmp_path: Path) -> None:
    root = _write_v3_root(tmp_path, tmp_path / "invalid_selected_v3", valid_provenance=True, invalidated=True)
    summary = _run_audit(tmp_path, root, "out_invalid_selected")
    assert summary["decision_v1"] == "FIX_SELECTED_V3_OOF_ARTIFACT_FIRST"

