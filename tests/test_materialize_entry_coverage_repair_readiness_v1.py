from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_entry_coverage_repair_readiness_v1 import (
    REPAIR_AUDIT,
    REPAIR_CONSISTENCY_AUDIT,
    REPAIR_SUMMARY,
    R2_AS_OF_TABLE,
    R2_COVERAGE_AUDIT,
    R2_LABEL_TABLE,
    materialize,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _replay_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    times = pd.date_range("2025-01-01T00:00:00Z", periods=10, freq="min")
    for idx, timestamp in enumerate(times):
        open_price = 2600.0 + idx
        close_price = open_price + 0.5
        rows.append(
            {
                "time": timestamp,
                "open": open_price,
                "high": close_price + 1.0,
                "low": open_price - 1.0,
                "close": close_price,
                "spread_bps": 1.5 + idx / 100.0,
                "_v1_body_share_1": 0.2,
                "_v1_clv": 0.4,
                "_v1_cost_bps_dyn": 2.0,
                "_v1_cost_bps_est": 2.1,
                "_v1_bb_squeeze_20_2": 0.1,
                "_v1_bb_bandwidth_delta_10": 0.2,
                "_v1_kama_slope_30": 0.3,
                "micro_momentum_3": 0.4,
                "micro_momentum_5": 0.5,
                "micro_acceleration": 0.6,
                "wick_ratio": 0.7,
                "distance_ema_fast": 0.8,
                "dist_last_swing_high_atr": 0.9,
                "dist_last_swing_low_atr": 1.0,
                "bars_since_swing_high": 2.0,
                "bars_since_swing_low": 3.0,
                "retracement_from_last_impulse": 0.25,
                "minutes_since_session_open": 100.0 + idx,
                "minutes_to_next_session_boundary": 200.0 - idx,
                "session_change_flag": 0,
                "session_tradable": 1,
                "H1_range_compression_ratio": 1.1,
                "M15_range_compression_ratio": 1.2,
                "D1_atr_percentile_252": 0.8,
                "D1_dist_from_ema200_atr": 1.5,
            }
        )
    return rows


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    reports_root = tmp_path / "reports"
    readiness_dir = reports_root / "readiness"
    canonical_dir = reports_root / "canonical"
    extension_dir = reports_root / "repaired"
    run_id = "E2E_SANITY_ORDERFIX_20250101_20250108"
    run_dir = reports_root / "runs" / run_id
    chunk_dir = run_dir / "replay" / "chunk_0"
    for directory in [readiness_dir, canonical_dir, chunk_dir]:
        directory.mkdir(parents=True)

    feature_names = [
        "as_of_hour_utc_v1",
        "as_of_weekday_utc_v1",
        "as_of_candidate_entry_spread_bps_v1",
        "as_of_entry_candidate_margin_v1",
        "as_of_skip_candidate_p_long_v1",
        "as_of_skip_xgb_p_long_v1",
        "as_of_skip_xgb_pred_side_v1",
        "as_of_skip_xgb_has_ctx_v1",
        "as_of_skip_replay_range_bps_v1",
        "as_of_skip_replay_window_ret_1_bps_v1",
        "as_of_skip_replay_window_up_move_15_bps_v1",
    ]
    asof_rows = [
        {
            "run_id": run_id,
            "candidate_uid": "cand-covered",
            "trade_uid": "trade-covered",
            "trade_id": "1",
            "decision_timestamp": "2025-01-01T00:01:00+00:00",
            "used_for_training": True,
            "used_for_validation": False,
            "used_for_holdout": False,
            "entry_observation_present_v1": True,
            "entry_raw_state_present_v1": True,
            "management_observation_present_v1": True,
            **{feature: 1.0 for feature in feature_names if feature != "as_of_skip_xgb_pred_side_v1"},
            "as_of_skip_xgb_pred_side_v1": "LONG",
        },
        {
            "run_id": run_id,
            "candidate_uid": "cand-missing",
            "trade_uid": "trade-missing",
            "trade_id": "2",
            "decision_timestamp": "2025-01-01T00:05:00+00:00",
            "used_for_training": True,
            "used_for_validation": False,
            "used_for_holdout": False,
            "entry_observation_present_v1": False,
            "entry_raw_state_present_v1": False,
            "management_observation_present_v1": True,
            **{feature: pd.NA for feature in feature_names},
        },
    ]
    pd.DataFrame(asof_rows).to_parquet(readiness_dir / R2_AS_OF_TABLE, index=False)
    pd.DataFrame(
        [
            {"candidate_uid": "cand-covered", "label_should_not_take_v1": False, "label_strong_trade_candidate_v1": False},
            {"candidate_uid": "cand-missing", "label_should_not_take_v1": False, "label_strong_trade_candidate_v1": True},
        ]
    ).to_parquet(readiness_dir / R2_LABEL_TABLE, index=False)
    pd.DataFrame(
        [
            {"run_id": run_id, "candidate_uid": "cand-covered", "entry_observation_present_v1": True, "entry_gap_reason_code_v1": "COVERED"},
            {
                "run_id": run_id,
                "candidate_uid": "cand-missing",
                "entry_observation_present_v1": False,
                "entry_gap_reason_code_v1": "missing entry observation",
                "entry_gap_reason_detail_v1": "test gap",
            },
        ]
    ).to_csv(readiness_dir / R2_COVERAGE_AUDIT, index=False)
    _write_json(readiness_dir / "shadow_meta_all_trade_review_harvest_r2_entry_readiness_contract_v1.json", {"as_of_feature_names_v1": feature_names})
    _write_json(readiness_dir / "shadow_meta_all_trade_review_harvest_r2_entry_readiness_summary_v1.json", {"layer_name": "fixture"})
    _write_json(readiness_dir / "shadow_meta_all_trade_review_harvest_r2_entry_readiness_manifest_v1.json", {"layer_name": "fixture"})
    pd.DataFrame([{"candidate_uid": "cand-missing"}]).to_parquet(canonical_dir / "shadow_meta_all_trade_review_as_of_decision_moment_ledger_v1.parquet", index=False)

    pd.DataFrame(_replay_rows()).to_parquet(chunk_dir / "chunk_0_data.parquet", index=False)
    pd.DataFrame(
        [
            {
                "side": "long",
                "session": "US",
                "weekday_utc": 2,
                "hour_utc": 0,
                "atr_bps": 10.0,
                "entry_spread_bps": 1.23,
                "p_long": 0.61,
                "p_short": 0.09,
                "p_flat": 0.30,
                "p_hat": 0.61,
                "margin": 0.31,
                "uncertainty_score": 0.39,
                "tradable_prob": 0.91,
                "mfe_first_n_pred": 2.2,
                "path_quality_pred": 0.71,
                "vol_regime": "HIGH",
                "trend_regime": "TREND_UP",
                "run_id": run_id,
                "candidate_uid": "cand-missing",
                "trade_uid": "trade-missing",
                "trade_id": "2",
                "decision_ts_utc": "2025-01-01T00:05:00+00:00",
                "decision": "LONG",
                "accepted": True,
                "decision_reason": "fixture",
                "policy_hash": "hash",
                "entry_bundle_sha256": "entry",
                "exit_bundle_sha256": "exit",
            }
        ]
    ).to_parquet(run_dir / f"shadow_meta_candidates_{run_id}_MERGED.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ts": "2025-01-01T00:05:00+00:00",
                "head": "OVERLAP",
                "horizon_bars": 24,
                "p_long": 0.61,
                "p_short": 0.09,
                "p_flat": 0.30,
                "p_hat": 0.61,
                "pred_side": "LONG",
                "has_ctx": 1,
            }
        ]
    ).to_parquet(run_dir / f"xgb_multi_horizon_predictions_{run_id}.parquet", index=False)
    return reports_root, readiness_dir, canonical_dir, extension_dir


def test_materialize_entry_coverage_repair_readiness(tmp_path: Path) -> None:
    reports_root, readiness_dir, canonical_dir, extension_dir = _build_fixture(tmp_path)

    result = materialize(
        reports_root,
        readiness_dir=readiness_dir,
        canonical_dir=canonical_dir,
        extension_dir=extension_dir,
        expected_ledger_count=2,
    )

    assert result["status"]["ENTRY_COVERAGE_REPAIR_STATUS"] == "ENTRY_COVERAGE_REPAIRED_READY_FOR_SHADOW_RETRAIN_NOT_LIVE_GATE"
    for artifact in [R2_AS_OF_TABLE, R2_LABEL_TABLE, R2_COVERAGE_AUDIT, REPAIR_AUDIT, REPAIR_CONSISTENCY_AUDIT, REPAIR_SUMMARY]:
        assert (extension_dir / artifact).exists()

    repaired = pd.read_parquet(extension_dir / R2_AS_OF_TABLE).set_index("candidate_uid")
    coverage = pd.read_csv(extension_dir / R2_COVERAGE_AUDIT).set_index("candidate_uid")

    assert bool(repaired.loc["cand-missing", "entry_observation_present_v1"])
    assert bool(repaired.loc["cand-missing", "entry_raw_state_present_v1"])
    assert repaired.loc["cand-missing", "entry_coverage_repair_source_v1"] == "RUN_SHADOW_META_CANDIDATES_PLUS_REPLAY_CHUNK_PLUS_XGB_EXACT"
    assert float(repaired.loc["cand-missing", "as_of_skip_xgb_p_long_v1"]) == 0.61
    assert repaired.loc["cand-missing", "as_of_skip_xgb_pred_side_v1"] == "LONG"
    assert pd.notna(repaired.loc["cand-missing", "as_of_skip_replay_range_bps_v1"])
    assert coverage.loc["cand-missing", "entry_gap_reason_code_v1"] == "covered by exact run-source repair"


def test_materialize_entry_coverage_repair_readiness_monday_top_level_run_dir(tmp_path: Path) -> None:
    reports_root, readiness_dir, canonical_dir, extension_dir = _build_fixture(tmp_path)
    legacy_run_dir = reports_root / "runs" / "E2E_SANITY_ORDERFIX_20250101_20250108"
    monday_run_id = "TRUTH_MONFRI_WEEK_20250106_20250113"
    monday_run_dir = reports_root / monday_run_id
    monday_run_dir.mkdir(parents=True)
    (monday_run_dir / "replay" / "chunk_0").mkdir(parents=True)
    for source, target in [
        (legacy_run_dir / f"shadow_meta_candidates_{legacy_run_dir.name}_MERGED.parquet", monday_run_dir / f"shadow_meta_candidates_{monday_run_id}_MERGED.parquet"),
        (legacy_run_dir / f"xgb_multi_horizon_predictions_{legacy_run_dir.name}.parquet", monday_run_dir / f"xgb_multi_horizon_predictions_{monday_run_id}.parquet"),
        (legacy_run_dir / "replay" / "chunk_0" / "chunk_0_data.parquet", monday_run_dir / "replay" / "chunk_0" / "chunk_0_data.parquet"),
    ]:
        target.write_bytes(source.read_bytes())
    legacy_run_dir.rename(legacy_run_dir.with_name(f"_{legacy_run_dir.name}_moved"))

    asof_df = pd.read_parquet(readiness_dir / R2_AS_OF_TABLE)
    asof_df["run_id"] = monday_run_id
    asof_df.to_parquet(readiness_dir / R2_AS_OF_TABLE, index=False)
    coverage_df = pd.read_csv(readiness_dir / R2_COVERAGE_AUDIT)
    coverage_df["run_id"] = monday_run_id
    coverage_df.to_csv(readiness_dir / R2_COVERAGE_AUDIT, index=False)

    result = materialize(
        reports_root,
        readiness_dir=readiness_dir,
        canonical_dir=canonical_dir,
        extension_dir=extension_dir,
        expected_ledger_count=2,
    )

    assert result["status"]["ENTRY_COVERAGE_REPAIR_STATUS"] == "ENTRY_COVERAGE_REPAIRED_READY_FOR_SHADOW_RETRAIN_NOT_LIVE_GATE"
