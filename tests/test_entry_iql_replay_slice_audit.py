import argparse
import json
from pathlib import Path

import pandas as pd

from gx1.scripts.audit_entry_iql_replay_slices_v1 import run


def _write_fixture(
    root: Path,
    *,
    iql_eu_negative: bool = False,
    iql_stop_loss_positive_mfe: bool = False,
) -> argparse.Namespace:
    candidate_dir = root / "candidate"
    iql_dir = root / "iql"
    out_dir = root / "out"
    candidate_dir.mkdir(parents=True)
    iql_dir.mkdir(parents=True)
    rows = []
    for idx in range(40):
        session = "EU" if idx < 20 else "US"
        vol_regime = "3" if idx % 2 == 0 else "4"
        side = "LONG" if idx % 2 == 0 else "SHORT"
        direction_correct = idx % 4 != 0
        exit_reason = "horizon"
        candidate_pnl = 12.0 if direction_correct else -5.0
        iql_pnl = 18.0 if direction_correct else -4.0
        if iql_eu_negative and session == "EU":
            iql_pnl = -9.0
        base = {
            "fold": "2026_TEST",
            "policy_id": "policy",
            "session": session,
            "vol_regime": vol_regime,
            "entry_day": "2026-01-01",
            "entry_month": "2026-01",
            "entry_time": f"2026-01-01 00:{idx:02d}:00+00:00",
            "exit_time": f"2026-01-01 01:{idx:02d}:00+00:00",
            "side": side,
            "direction_correct": direction_correct,
            "score": 0.8,
            "p_long": 0.8 if side == "LONG" else 0.1,
            "p_short": 0.8 if side == "SHORT" else 0.1,
            "p_flat": 0.1,
            "path_quality_pred": 1.0,
            "bad_path_prob": 0.15 + (idx % 5) * 0.08,
            "mfe_bps": 30.0,
            "mae_bps": 10.0 + idx,
            "horizon_bars": 24,
            "held_bars": 8 + (idx % 10),
            "exit_reason": exit_reason,
        }
        rows.append((base, candidate_pnl, iql_pnl))

    candidate_rows = []
    iql_rows = []
    for idx, (base, candidate_pnl, iql_pnl) in enumerate(rows):
        cand = dict(base)
        cand["net_pnl_bps"] = candidate_pnl
        cand["gross_pnl_bps"] = candidate_pnl
        candidate_rows.append(cand)
        iql = dict(base)
        iql["policy_id"] = "entry_iql_student"
        iql["net_pnl_bps"] = iql_pnl
        iql["gross_pnl_bps"] = iql_pnl
        if iql_stop_loss_positive_mfe and idx < 20:
            iql["exit_reason"] = "stop_loss"
            iql["net_pnl_bps"] = -45.0
            iql["gross_pnl_bps"] = -45.0
            iql["mfe_bps"] = 12.0
            iql["mae_bps"] = 45.0
        iql_rows.append(iql)

    candidate_trades = candidate_dir / "candidate_trades.csv"
    iql_trades = iql_dir / "iql_trades.csv"
    pd.DataFrame(candidate_rows).to_csv(candidate_trades, index=False)
    pd.DataFrame(iql_rows).to_csv(iql_trades, index=False)
    for replay_dir, trades_path in ((candidate_dir, candidate_trades), (iql_dir, iql_trades)):
        (replay_dir / "REPLAY_EVIDENCE_MANIFEST.json").write_text(
            json.dumps(
                {
                    "decision": "PASS",
                    "trades_path": str(trades_path),
                    "promotion_shadow_live_allowed": False,
                    "trainer_started": False,
                    "replay_started": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    comparison = root / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "decision": "READY_FOR_PROMOTION_REVIEW_VEDTAK",
                "promotion_shadow_live_allowed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return argparse.Namespace(
        candidate_replay_dir=str(candidate_dir),
        iql_replay_dir=str(iql_dir),
        comparison_json=str(comparison),
        candidate_trades_path="",
        iql_trades_path="",
        out_dir=str(out_dir),
        min_slice_trades=5,
        min_iql_edge_net_bps=0.0,
        min_iql_edge_profit_factor=1.0,
        max_slice_drawdown_worsening_bps=120.0,
        max_max_loss_worsening_bps=0.0,
        max_diagnostic_mean_degradation_bps=10.0,
        max_tail_p10_degradation_bps=20.0,
        max_diagnostic_max_loss_worsening_bps=10.0,
        max_total_stop_loss_rate=0.25,
        max_supported_slice_stop_loss_rate=0.40,
        max_stop_loss_positive_mfe_rate=0.70,
        max_abs_replay_loss_bps=90.0,
        max_tail_loss_p05_abs_mean_bps=90.0,
        fail_on_not_ready=False,
        quiet=True,
    )


def test_iql_replay_slice_audit_passes_supported_slices(tmp_path: Path) -> None:
    report = run(_write_fixture(tmp_path))

    assert report["decision"] == "PASS"
    assert report["promotion_shadow_live_allowed"] is False
    assert report["trainer_started"] is False
    assert report["replay_started"] is False
    assert report["adapter_built"] is False
    assert report["supported_edge_counts"]["session"] >= 2
    assert report["supported_edge_counts"]["vol_regime"] >= 2
    assert report["supported_edge_counts"]["side"] >= 2
    assert Path(report["slice_metrics_csv"]).exists()
    assert Path(report["slice_comparison_csv"]).exists()
    assert Path(report["exit_opportunity_csv"]).exists()
    assert Path(report["path_signal_calibration"]["csv"]).exists()
    assert report["path_signal_calibration"]["rows"] > 0
    assert report["path_signal_calibration"]["diagnostic_only_not_gate"] is True
    assert report["exit_opportunity_summary"]["iql_all"][0]["peak_oracle_lift_sum_bps"] >= 0.0
    checks = {row["name"]: row["ok"] for row in report["checks"]}
    assert checks["IQL supported edge slices keep positive net/PF/drawdown/max-loss"] is True
    assert checks["IQL diagnostic slices do not materially worsen tails vs candidate"] is True
    assert checks["exit opportunity diagnostics were produced from replay MFE/MAE/held bars"] is True
    assert checks["candidate and IQL tail/path quality hard checks pass"] is True


def test_iql_replay_slice_audit_fails_when_supported_session_loses_edge(tmp_path: Path) -> None:
    report = run(_write_fixture(tmp_path, iql_eu_negative=True))

    assert report["decision"] == "FAIL"
    failed = {row["check"] for row in report["failures"]}
    assert "IQL supported edge slices keep positive net/PF/drawdown/max-loss" in failed
    assert any(row["cube"] == "session" and row["slice"] == "EU" for row in report["edge_failures"])


def test_iql_replay_slice_audit_fails_on_stop_loss_with_positive_mfe_tail_path(
    tmp_path: Path,
) -> None:
    report = run(_write_fixture(tmp_path, iql_stop_loss_positive_mfe=True))

    assert report["decision"] == "FAIL"
    failed = {row["check"] for row in report["failures"]}
    assert "candidate and IQL tail/path quality hard checks pass" in failed
    iql_tail = report["tail_path_quality"]["iql"]
    assert iql_tail["stop_loss_rate"] > 0.25
    assert iql_tail["stop_loss_with_positive_mfe_rate"] > 0.70
    assert any(row["model"] == "iql" for row in report["tail_path_failures"])
