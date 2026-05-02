from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gx1.execution.replay_merge import (
    _finalize_shadow_meta_v1,
    _merge_support_table_by_trade_identity,
)


def test_shadow_meta_finalizer_backfills_outcome_only_rows(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "chunk_0"
    chunk_dir.mkdir(parents=True)

    exit_bundle_dir = tmp_path / "exit_bundle"
    exit_bundle_dir.mkdir()
    (exit_bundle_dir / "weights.bin").write_bytes(b"exit-bundle")

    (chunk_dir / "run_header.json").write_text(
        json.dumps(
            {
                "policy_lane": "TEST_POLICY_LANE",
                "artifacts": {
                    "policy": {
                        "sha256": "policy-sha-256",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (chunk_dir / "MODEL_USED_CAPSULE.json").write_text(
        json.dumps({"bundle_sha256": "entry-bundle-sha-256"}),
        encoding="utf-8",
    )
    (chunk_dir / "EXIT_RUNTIME_SOURCE_OF_TRUTH.json").write_text(
        json.dumps({"bundle_path": str(exit_bundle_dir)}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "time": "2025-04-23T12:20:00+00:00",
                "atr_bps": 12.5,
                "trend_regime_id": 2,
                "vol_regime_id": 3,
            }
        ]
    ).to_parquet(chunk_dir / "chunk_0_data.parquet", index=False)

    merged_shadow = pd.DataFrame(
        [
            {
                "candidate_uid": "RUN_X:0:cand::000022:abc123",
                "trade_uid": "RUN_X:0::000001:def456",
                "trade_id": "SIM-000001",
                "decision_ts_utc": "2025-04-23T12:20:00+00:00",
                "source_eval_log": "eval_log.jsonl",
                "source_eval_log_row": 22,
                "decision": "LONG",
                "accepted": True,
                "decision_reason": "pre_quality",
                "p_long": 0.91,
                "p_short": 0.05,
                "p_flat": 0.04,
                "p_hat": 0.91,
                "margin": 0.86,
                "uncertainty_score": 0.09,
                "side_outcome": "long",
                "session_outcome": "OVERLAP",
                "entry_spread_bps_outcome": 1.8,
                "open_ts_utc_outcome": "2025-04-23T12:21:00+00:00",
                "close_ts_utc_outcome": "2025-04-23T13:56:00+00:00",
                "pnl_bps_outcome": 11.5,
                "mfe_bps_outcome": 25.0,
                "mae_bps_outcome": -3.0,
                "duration_bars_outcome": 19,
                "exit_reason_outcome": "THRESHOLD",
            }
        ]
    )

    out = _finalize_shadow_meta_v1(
        merged_shadow,
        run_id="RUN_X",
        chunk_dir=chunk_dir,
        run_header={},
        footer={},
    )
    row = out.iloc[0]

    assert row["open_ts_utc"] == "2025-04-23T12:21:00+00:00"
    assert row["close_ts_utc"] == "2025-04-23T13:56:00+00:00"
    assert row["pnl_bps"] == 11.5
    assert row["mfe_bps"] == 25.0
    assert row["mae_bps"] == -3.0
    assert row["bars_in_trade"] == 19.0
    assert bool(row["trainable_mask_v1"]) is True
    assert bool(row["meta_allow_label_v1"]) is True
    assert bool(row["good_trade_mfe20_mae5_v1"]) is True
    assert abs(float(row["mfe_mae_ratio_v1"]) - (25.0 / 3.0)) < 1e-9


def test_shadow_meta_identity_fallback_joins_journal_by_trade_id_then_outcomes_by_trade_uid(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "chunk_0"
    chunk_dir.mkdir(parents=True)

    exit_bundle_dir = tmp_path / "exit_bundle"
    exit_bundle_dir.mkdir()
    (exit_bundle_dir / "weights.bin").write_bytes(b"exit-bundle")

    (chunk_dir / "run_header.json").write_text(
        json.dumps(
            {
                "policy_lane": "TEST_POLICY_LANE",
                "artifacts": {
                    "policy": {
                        "sha256": "policy-sha-256",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (chunk_dir / "MODEL_USED_CAPSULE.json").write_text(
        json.dumps({"bundle_sha256": "entry-bundle-sha-256"}),
        encoding="utf-8",
    )
    (chunk_dir / "EXIT_RUNTIME_SOURCE_OF_TRUTH.json").write_text(
        json.dumps({"bundle_path": str(exit_bundle_dir)}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "time": "2025-04-23T12:20:00+00:00",
                "atr_bps": 12.5,
                "trend_regime_id": 2,
                "vol_regime_id": 3,
            }
        ]
    ).to_parquet(chunk_dir / "chunk_0_data.parquet", index=False)

    merged_shadow = pd.DataFrame(
        [
            {
                "candidate_uid": "RUN_X:0:cand::000022:abc123",
                "trade_uid": "RUN_X:0::000001:def456",
                "trade_id": "SIM-000001",
                "decision_ts_utc": "2025-04-23T12:20:00+00:00",
                "source_eval_log": "eval_log.jsonl",
                "source_eval_log_row": 22,
                "decision": "LONG",
                "accepted": True,
                "decision_reason": "pre_quality",
                "p_long": 0.91,
                "p_short": 0.05,
                "p_flat": 0.04,
                "p_hat": 0.91,
                "margin": 0.86,
                "uncertainty_score": 0.09,
            }
        ]
    )
    journal_df = pd.DataFrame(
        [
            {
                "trade_uid": None,
                "trade_id": "SIM-000001",
                "close_ts_utc": "2025-04-23T16:51:00+00:00",
                "bars_in_trade": 270,
                "exit_reason": "CATASTROPHIC_GUARD",
            }
        ]
    )
    outcomes_df = pd.DataFrame(
        [
            {
                "trade_uid": "RUN_X:0::000001:def456",
                "candidate_uid": "RUN_X:0:cand::000022:abc123",
                "trade_id": "SIM-000001",
                "open_ts_utc": "2025-04-23T12:21:00+00:00",
                "close_ts_utc": "2026-04-19T14:36:32.368727+00:00",
                "pnl_bps": -81.29519321508268,
                "mfe_bps": 13.944905601403724,
                "mae_bps": -195.980020316285,
                "duration_bars": 270,
                "exit_reason": "CATASTROPHIC_GUARD",
                "side": "long",
                "session": "OVERLAP",
                "entry_spread_bps": 1.8,
            }
        ]
    )

    merged_shadow = _merge_support_table_by_trade_identity(
        merged_shadow,
        journal_df.assign(join_key="SIM-000001"),
        keep_cols=["join_key", "trade_uid", "trade_id", "close_ts_utc", "bars_in_trade", "exit_reason"],
        support_label="journal",
    )
    merged_shadow = _merge_support_table_by_trade_identity(
        merged_shadow,
        outcomes_df.assign(join_key="RUN_X:0::000001:def456"),
        keep_cols=[
            "join_key",
            "trade_uid",
            "candidate_uid",
            "trade_id",
            "open_ts_utc",
            "close_ts_utc",
            "pnl_bps",
            "mfe_bps",
            "mae_bps",
            "duration_bars",
            "exit_reason",
            "side",
            "session",
            "entry_spread_bps",
        ],
        support_label="outcome",
    )

    out = _finalize_shadow_meta_v1(
        merged_shadow,
        run_id="RUN_X",
        chunk_dir=chunk_dir,
        run_header={},
        footer={},
    )
    row = out.iloc[0]

    assert row["open_ts_utc"] == "2025-04-23T12:21:00+00:00"
    assert row["close_ts_utc"] == "2025-04-23T16:51:00+00:00"
    assert row["pnl_bps"] == -81.29519321508268
    assert row["mfe_bps"] == 13.944905601403724
    assert row["mae_bps"] == -195.980020316285
    assert row["bars_in_trade"] == 270.0
    assert row["exit_reason"] == "CATASTROPHIC_GUARD"
    assert bool(row["trainable_mask_v1"]) is True
