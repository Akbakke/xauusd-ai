import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_r6_canonical_line_restore_and_monday_replay_reanchor_v1 import (
    CANONICAL_EXPECTED,
    OLD_MONDAY_BRIDGE,
    OLD_MONDAY_BRIDGE_DIR,
    OLD_MONDAY_EXACT_DIR,
    OLD_MONDAY_EXACT_RAW,
    OLD_MONDAY_R6_ASOF,
    OLD_MONDAY_R6_DIR,
    OLD_MONDAY_R6_LABELS,
    OUTPUT_FILES,
    WEDNESDAY_FREEZE_DIR,
    WEDNESDAY_MANIFEST,
    WEDNESDAY_SNAPSHOT_DIR,
    WEDNESDAY_SUMMARY,
    materialize,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base(candidate_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": ["run"] * len(candidate_ids),
            "candidate_uid": candidate_ids,
            "trade_uid": [f"trade-{candidate}" for candidate in candidate_ids],
            "trade_id": [f"T-{candidate}" for candidate in candidate_ids],
            "decision_timestamp": [f"2026-04-2{idx}T10:00:00Z" for idx, _ in enumerate(candidate_ids)],
        }
    )


def _write_fixture(root: Path) -> None:
    freeze_dir = root / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR
    _write_json(
        freeze_dir / WEDNESDAY_SUMMARY,
        {
            "freeze_id_v1": CANONICAL_EXPECTED["freeze_id_v1"],
            "selected_candidate_id_v1": CANONICAL_EXPECTED["selected_candidate_id_v1"],
            "reports_root_v1": str(root / "missing_wednesday_source"),
            "r6_source_dir_v1": str(root / "missing_wednesday_source" / "r6"),
            "r5_2_freeze_dir_v1": str(root / "missing_wednesday_source" / "r5_2_freeze"),
            "policy_logging_v1": {
                "row_count_v1": CANONICAL_EXPECTED["policy_eval_rows_v1"],
                "hindsight_backfill_rows_v1": CANONICAL_EXPECTED["hindsight_backfill_rows_v1"],
                "mask_mismatch_count_v1": 0,
            },
            "selected_candidate_v1": {
                "should_not_take_block_count_v1": CANONICAL_EXPECTED["bad_blocks_v1"],
                "tail_10_50_help_count_v1": CANONICAL_EXPECTED["tail_help_v1"],
                "should_not_take_precision_v1": CANONICAL_EXPECTED["precision_v1"],
                "worst_loso_precision_v1": CANONICAL_EXPECTED["worst_loso_v1"],
                "repaired_165_block_count_v1": CANONICAL_EXPECTED["repaired_165_damage_v1"],
                "fifty_plus_mfe_block_count_v1": CANONICAL_EXPECTED["fifty_plus_mfe_blocked_v1"],
                "hundred_plus_mfe_block_count_v1": CANONICAL_EXPECTED["hundred_plus_mfe_blocked_v1"],
                "two_hundred_plus_mfe_block_count_v1": CANONICAL_EXPECTED["two_hundred_plus_mfe_blocked_v1"],
                "strongest_winner_path_block_count_v1": CANONICAL_EXPECTED["strongest_winner_damage_v1"],
            },
            "thresholds_v1": {"bad_threshold_v1": 0.95},
        },
    )
    _write_json(
        freeze_dir / WEDNESDAY_MANIFEST,
        {
            "freeze_id_v1": CANONICAL_EXPECTED["freeze_id_v1"],
            "selected_candidate_id_v1": CANONICAL_EXPECTED["selected_candidate_id_v1"],
            "reports_root_v1": str(root / "missing_wednesday_source"),
            "r6_source_dir_v1": str(root / "missing_wednesday_source" / "r6"),
            "as_of_schema_v1": {
                "column_count_v1": CANONICAL_EXPECTED["as_of_column_count_v1"],
                "schema_sha256_v1": "asof",
                "columns_v1": [
                    {"name_v1": "candidate_uid", "dtype_v1": "string"},
                    {"name_v1": "trade_uid", "dtype_v1": "string"},
                    {"name_v1": "decision_timestamp", "dtype_v1": "string"},
                ],
            },
            "hindsight_schema_v1": {
                "column_count_v1": 30,
                "schema_sha256_v1": "hindsight",
                "columns_v1": [{"name_v1": "candidate_uid", "dtype_v1": "string"}],
            },
        },
    )

    exact_dir = root / OLD_MONDAY_EXACT_DIR
    exact_dir.mkdir(parents=True, exist_ok=True)
    _base(["c1"]).to_parquet(exact_dir / OLD_MONDAY_EXACT_RAW, index=False)

    bridge = _base(["c1", "c2"])
    bridge = bridge.drop(columns=["decision_timestamp"])
    bridge["entry_coverage_repair_applied_v1"] = [False, True]
    bridge["bridge_pocket_repaired_165_v1"] = [False, True]
    bridge["bridge_pocket_forensic_repaired_trade_v1"] = [False, True]
    bridge["bridge_pocket_runner_near_miss_v1"] = [False, True]
    bridge["bridge_pocket_fifty_plus_mfe_seed_v1"] = [False, True]
    bridge["bridge_pocket_missed_10_50_tail_control_v1"] = [False, False]
    bridge["bridge_pocket_missed_should_not_take_v1"] = [False, False]
    bridge["bridge_pocket_risky_allow_v1"] = [False, False]
    bridge_dir = root / OLD_MONDAY_BRIDGE_DIR
    bridge_dir.mkdir(parents=True, exist_ok=True)
    bridge.to_parquet(bridge_dir / OLD_MONDAY_BRIDGE, index=False)

    r6_dir = root / OLD_MONDAY_R6_DIR
    r6_dir.mkdir(parents=True, exist_ok=True)
    r6_asof = _base(["c1", "c2"])
    r6_asof.to_parquet(r6_dir / OLD_MONDAY_R6_ASOF, index=False)
    labels = _base(["c1", "c2"])
    labels["r6_label_runner_50_mfe_v1"] = [False, True]
    labels["r6_label_runner_100_mfe_v1"] = [False, True]
    labels["r6_label_runner_200_mfe_v1"] = [False, False]
    labels["r6_label_repaired_165_like_runner_v1"] = [False, True]
    labels["r6_label_runner_near_miss_v1"] = [False, True]
    labels.to_parquet(r6_dir / OLD_MONDAY_R6_LABELS, index=False)


def test_restore_reanchor_materializer_locks_canonical_and_blocks_without_source(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    output_dir = tmp_path / "out"

    summary = materialize(reports_root=tmp_path, output_dir=output_dir)

    assert summary["canonical_wednesday_freeze_id_v1"] == CANONICAL_EXPECTED["freeze_id_v1"]
    assert summary["parity_gate_decision_v1"] == "MONDAY_BLOCKED_BY_MISSING_WEDNESDAY_SOURCE_ARTIFACTS"
    assert summary["hard_recommendation_v1"] == "RESTORE_WEDNESDAY_SOURCE_ARTIFACTS_FIRST"
    for filename in OUTPUT_FILES.values():
        assert (output_dir / filename).exists()

    quarantine = json.loads((output_dir / "bad_monday_narrow_surface_quarantine_v1.json").read_text(encoding="utf-8"))
    assert quarantine["status_v1"] == "QUARANTINED_FROM_CANONICAL_R6_BASELINE"

    row_delta = pd.read_csv(output_dir / "row_delta_explainer_v1.csv")
    delta_rows = row_delta[row_delta["row_type_v1"].eq("DELTA_ROW")]
    assert delta_rows["candidate_uid"].tolist() == ["c2"]
    assert delta_rows["decision_timestamp"].iloc[0] == "2026-04-21T10:00:00Z"
    assert delta_rows["status_v1"].iloc[0] == "RAW_STATE_MISSING"

    next_action = json.loads((output_dir / "next_action_lock_v1.json").read_text(encoding="utf-8"))
    assert "DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE" in next_action["always_blocked_actions_v1"]
