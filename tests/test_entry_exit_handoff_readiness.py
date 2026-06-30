import argparse
import json
from pathlib import Path

import pandas as pd

from gx1.scripts.audit_entry_exit_handoff_readiness_v1 import REQUIRED_TRADE_FIELDS, run


def _write_trade_log(path: Path) -> None:
    rows = []
    for idx in range(3):
        row = {field: "x" for field in REQUIRED_TRADE_FIELDS}
        row.update(
            {
                "fold": "2026_TEST",
                "policy_id": "entry_iql_student",
                "session": "EU",
                "vol_regime": "4",
                "entry_time": f"2026-01-01 00:0{idx}:00+00:00",
                "exit_time": f"2026-01-01 01:0{idx}:00+00:00",
                "side": "LONG",
                "p_long": 0.8,
                "p_short": 0.1,
                "p_flat": 0.1,
                "score": 0.8,
                "path_quality_pred": 1.2,
                "bad_path_prob": 0.2,
                "net_pnl_bps": 20.0,
                "mfe_bps": 40.0,
                "mae_bps": 8.0,
                "held_bars": 12,
                "exit_reason": "horizon",
            }
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _args(tmp_path: Path, *, comparison_ready: bool = True) -> argparse.Namespace:
    candidate_trades = tmp_path / "candidate_trades.csv"
    iql_trades = tmp_path / "iql_trades.csv"
    _write_trade_log(candidate_trades)
    _write_trade_log(iql_trades)
    exit_csv = tmp_path / "exit_opportunity.csv"
    pd.DataFrame(
        [
            {
                "model": "iql",
                "scope": "ALL",
                "value": "ALL",
                "n_trades": 3,
                "mean_mfe_capture_ratio": 0.5,
                "p90_giveback_bps": 10.0,
                "peak_oracle_lift_sum_bps": 60.0,
            }
        ]
    ).to_csv(exit_csv, index=False)
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "decision": (
                    "READY_FOR_PROMOTION_REVIEW_VEDTAK"
                    if comparison_ready
                    else "NOT_READY_FOR_PROMOTION_REVIEW"
                ),
                "promotion_shadow_live_allowed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    slice_audit = tmp_path / "slice_audit.json"
    slice_audit.write_text(
        json.dumps(
            {
                "decision": "PASS",
                "candidate_trades_path": str(candidate_trades),
                "iql_trades_path": str(iql_trades),
                "exit_opportunity_csv": str(exit_csv),
                "exit_opportunity_summary": {
                    "iql_all": [
                        {
                            "n_trades": 3,
                            "mean_mfe_capture_ratio": 0.5,
                            "p90_giveback_bps": 10.0,
                            "peak_oracle_lift_sum_bps": 60.0,
                        }
                    ]
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return argparse.Namespace(
        iql_comparison_json=str(comparison),
        iql_slice_audit_json=str(slice_audit),
        legacy_exit_truth_root=str(tmp_path / "missing_truth_e2e_sanity"),
        active_exit_substrate_root=str(tmp_path / "missing_active_exit_substrate"),
        out_dir=str(tmp_path / "out"),
        fail_on_not_ready=False,
        quiet=True,
    )


def test_entry_exit_handoff_blocks_when_exit_substrate_missing(tmp_path: Path) -> None:
    report = run(_args(tmp_path))

    assert report["decision"] == "BLOCKED_BY_MISSING_EXIT_PER_BAR_SUBSTRATE"
    assert report["entry_evidence_ready"] is True
    assert report["exit_per_bar_substrate_ready"] is False
    assert report["exit_training_allowed"] is False
    assert report["exit_iql_allowed"] is False
    assert report["promotion_shadow_live_allowed"] is False
    failed = {row["check"] for row in report["failures"]}
    assert "active Entry-bound exit per-bar substrate is available" in failed
    assert Path(report["json_path"]).exists()
    assert Path(report["md_path"]).exists()


def test_entry_exit_handoff_blocks_when_entry_evidence_missing(tmp_path: Path) -> None:
    report = run(_args(tmp_path, comparison_ready=False))

    assert report["decision"] == "BLOCKED_BY_ENTRY_EVIDENCE"
    assert report["entry_evidence_ready"] is False
    assert report["exit_training_allowed"] is False
    failed = {row["check"] for row in report["failures"]}
    assert "IQL replay comparison is ready" in failed
