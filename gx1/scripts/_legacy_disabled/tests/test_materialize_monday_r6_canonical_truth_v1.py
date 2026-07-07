import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_r6_canonical_truth_v1 import OUTPUT_FILES, materialize


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run(root: Path, run_id: str, *, quarantine_status: str, trade_rows: int) -> None:
    run_dir = root / run_id
    replay_dir = run_dir / "replay" / "chunk_0"
    logs_dir = replay_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    entry_ts = "2026-01-06T10:00:00Z"
    exit_ts = "2026-01-06T10:05:00Z"
    trade_id = "SIM-1"
    trade_uid = f"{run_id}:trade:1"
    candidate_uid = f"{run_id}:cand:1"

    outcomes = pd.DataFrame(
        {
            "trade_id": [trade_id] * trade_rows,
            "trade_uid": [trade_uid] * trade_rows,
            "candidate_uid": [candidate_uid] * trade_rows,
            "entry_time": [entry_ts] * trade_rows,
            "exit_time": [exit_ts] * trade_rows,
            "pnl_bps": [12.5] * trade_rows,
            "mae_bps": [-2.0] * trade_rows,
            "mfe_bps": [75.0] * trade_rows,
            "duration_bars": [5] * trade_rows,
            "side": ["long"] * trade_rows,
            "session": ["OVERLAP"] * trade_rows,
            "exit_reason": ["THRESHOLD"] * trade_rows,
            "early_exit_regret": [False] * trade_rows,
            "early_exit_regret_replay_end_obs": [True] * trade_rows,
            "post_exit_mfe_bps": [20.0] * trade_rows,
        }
    )
    outcomes.to_parquet(run_dir / f"trade_outcomes_{run_id}_MERGED.parquet", index=False)

    journal = pd.DataFrame(
        {
            "trade_id": [trade_id] * trade_rows,
            "trade_uid": [trade_uid] * trade_rows,
            "open_ts_utc": [entry_ts] * trade_rows,
            "close_ts_utc": [exit_ts] * trade_rows,
            "pnl_bps": [12.5] * trade_rows,
            "mae_bps": [-2.0] * trade_rows,
            "mfe_bps": [75.0] * trade_rows,
            "bars_in_trade": [5] * trade_rows,
            "side": ["long"] * trade_rows,
            "session": ["OVERLAP"] * trade_rows,
            "exit_reason": ["THRESHOLD"] * trade_rows,
        }
    )
    journal.to_parquet(run_dir / f"trade_journal_{run_id}_MERGED.parquet", index=False)

    candidate_count = max(1, trade_rows)
    candidates = pd.DataFrame(
        {
            "trade_id": [trade_id] * candidate_count,
            "trade_uid": [trade_uid] * candidate_count,
            "candidate_uid": [candidate_uid] * candidate_count,
            "decision_ts_utc": [entry_ts] * candidate_count,
            "accepted": [trade_rows > 0] * candidate_count,
            "side": ["long"] * candidate_count,
            "session": ["OVERLAP"] * candidate_count,
            "p_hat": [0.82] * candidate_count,
            "margin": [0.2] * candidate_count,
            "policy_hash": ["policy"] * candidate_count,
        }
    )
    candidates.to_parquet(run_dir / f"shadow_meta_candidates_{run_id}_MERGED.parquet", index=False)

    if trade_rows:
        pd.DataFrame(
            {
                "ts": [entry_ts],
                "p_long": [0.8],
                "p_short": [0.1],
                "p_flat": [0.1],
                "p_hat": [0.8],
                "pred_side": ["long"],
                "has_ctx": [True],
                "head": ["r6_head"],
                "horizon_bars": [12],
            }
        ).to_parquet(run_dir / f"xgb_multi_horizon_predictions_{run_id}.parquet", index=False)

        pd.DataFrame(
            {
                "exit_decision": ["threshold", "none"],
                "pnl_bps": [-0.5, 12.5],
                "exit_prob": [0.9, 0.1],
                "exit_threshold": [0.5, 0.5],
                "exit_model_evaluated": [1, 1],
            }
        ).to_csv(replay_dir / "EXIT_EVAL_TRACE.csv", index=False)

    pd.DataFrame({"time": [entry_ts, exit_ts], "asof_feature_a": [1.0, 2.0]}).to_parquet(
        replay_dir / "chunk_0_data.parquet", index=False
    )
    _write_json(replay_dir / "EXIT_FEATURE_VECTOR_PROOF.json", {"feature_list": [{"name": "exit_feat_a", "group": "risk"}]})
    (logs_dir / "replay.log").write_text(
        "EXIT_MODEL_DECIDED_EXIT\nLOSS_CLOSE_NOT_ALLOWED\n[ARB] reject model exit\n",
        encoding="utf-8",
    )
    _write_json(run_dir / "RUN_COMPLETED.json", {"run_id": run_id, "quarantine_status": quarantine_status})


def test_materialize_monday_r6_canonical_truth_builds_one_truth_with_all_available_surfaces(tmp_path: Path) -> None:
    active_run = "TRUTH_MONFRI_WEEK_20260105_20260112"
    quarantine_run = "TRUTH_MONFRI_WEEK_20251201_20251208"
    _write_json(
        tmp_path / "TRUTH_CALENDAR_REORG_MONDAY_WEEK_V1.json",
        {
            "full_monday_week_count": 2,
            "full_monday_weeks": [
                {
                    "run_id": active_run,
                    "quarantine_status": "ACTIVE_CANDIDATE",
                    "calendar_start_utc": "2026-01-05T00:00:00Z",
                    "calendar_end_exclusive_utc": "2026-01-12T00:00:00Z",
                    "friday_flat_cutoff_utc": "2026-01-09T20:55:00Z",
                },
                {
                    "run_id": quarantine_run,
                    "quarantine_status": "QUARANTINE_DIAGNOSTIC",
                    "quarantine_reason": "fixture",
                    "calendar_start_utc": "2025-12-01T00:00:00Z",
                    "calendar_end_exclusive_utc": "2025-12-08T00:00:00Z",
                    "friday_flat_cutoff_utc": "2025-12-05T20:55:00Z",
                },
            ],
        },
    )
    _write_run(tmp_path, active_run, quarantine_status="ACTIVE_CANDIDATE", trade_rows=1)
    _write_run(tmp_path, quarantine_run, quarantine_status="QUARANTINE_DIAGNOSTIC", trade_rows=0)

    output_dir = tmp_path / "out"
    summary = materialize(reports_root=tmp_path, output_dir=output_dir)

    assert summary["status_v1"] == "MONDAY_R6_CANONICAL_TRUTH_BUILT"
    assert summary["coverage_v1"]["trade_truth_rows_v1"] == 1
    assert summary["coverage_v1"]["included_run_count_v1"] == 2
    assert summary["coverage_v1"]["quarantine_marked_run_count_v1"] == 1
    assert summary["coverage_v1"]["zero_trade_run_count_v1"] == 1
    assert summary["coverage_v1"]["entry_bar_exact_rate_v1"] == 1.0

    for filename in OUTPUT_FILES.values():
        assert (output_dir / filename).exists()

    audit = pd.read_csv(output_dir / "consistency_audit_v1.csv")
    assert not audit["status_v1"].eq("FAIL").any()
    assert audit.set_index("check_v1").loc["NOT_1689_EXACT_ONLY", "status_v1"] == "PASS"
    assert audit.set_index("check_v1").loc["NOT_OLD_1852_ONLY", "status_v1"] == "PASS"

    manifest = pd.read_csv(output_dir / "monday_r6_truth_feature_manifest_v1.csv")
    assert set(manifest["surface_v1"]).issuperset(
        {
            "trade_truth",
            "candidate_surface",
            "xgb_signal_surface",
            "exit_eval_trace",
            "bar_feature_surface",
            "exit_transformer_runtime_input",
        }
    )

    conflict = json.loads((output_dir / "monday_r6_exit_conflict_summary_v1.json").read_text(encoding="utf-8"))
    assert conflict["model_would_exit_but_subfloor_rows_v1"] == 1
    assert conflict["log_pattern_counts_v1"]["LOSS_CLOSE_NOT_ALLOWED"] == 2
