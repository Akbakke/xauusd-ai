from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_zero_trade_should_have_trade_audit_v1 import (
    build_monday_zero_trade_should_have_trade_audit,
    materialize_monday_zero_trade_should_have_trade_audit,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _make_eval_log(run_dir: Path, prices: list[float]) -> None:
    log_path = run_dir / "replay" / "chunk_0" / "logs" / "eval_log_0.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        for idx, price in enumerate(prices):
            payload = {
                "ts_utc": f"2025-01-06T00:{idx:02d}:00Z",
                "price": price,
                "session": "US",
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _make_candidate_frame(rows: int, *, flat_ratio: float = 1.0) -> pd.DataFrame:
    flat_cut = int(rows * flat_ratio)
    reasons = ["flat_dominant"] * flat_cut + ["flat_veto"] * (rows - flat_cut)
    return pd.DataFrame(
        {
            "decision": ["NONE"] * rows,
            "accepted": [False] * rows,
            "decision_reason": reasons,
            "p_long": [0.1] * rows,
            "p_short": [0.01] * rows,
            "p_flat": [0.89] * rows,
            "p_hat": [0.89] * rows,
            "margin": [0.85] * rows,
            "tradable_prob": [0.5] * rows,
            "mfe_first_n_pred": [1.9] * rows,
            "path_quality_pred": [0.72] * rows,
            "session": ["US"] * rows,
            "vol_regime": ["HIGH"] * rows,
            "trend_regime": ["TREND_NEUTRAL"] * rows,
        }
    )


def _make_run(
    reports_root: Path,
    run_id: str,
    *,
    n_trades: int,
    prices: list[float],
    candidate_rows: int,
    candidate_below_threshold: int,
    candidate_flat_veto: int,
) -> None:
    run_dir = reports_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "RUN_COMPLETED.json", {"status": "COMPLETED"})
    _write_json(
        run_dir / f"metrics_{run_id}_MERGED.json",
        {
            "run_id": run_id,
            "n_trades": n_trades,
            "n_model_calls": len(prices),
            "chunk_footer_status": "ok",
        },
    )
    _write_json(
        run_dir / f"trade_report_entry_gates_{run_id}.json",
        {
            "run_id": run_id,
            "n_trades_closed": n_trades,
            "runner_counters": {
                "pregate_passes": len(prices),
                "entry_attempt_long": 0,
                "entry_accept_long": 0,
                "threshold_used": "long=0.42,short=0.42",
                "threshold_source": "canonical",
                "entry_gate_counters": {
                    "candidate_below_threshold": candidate_below_threshold,
                    "candidate_flat_veto": candidate_flat_veto,
                    "pregate_session": 0,
                    "pregate_weekly_entry_window": 0,
                    "pregate_d1_atr_eu": 0,
                    "pregate_regime_filter": 0,
                },
            },
        },
    )
    _make_eval_log(run_dir, prices)
    _make_candidate_frame(candidate_rows).to_parquet(
        run_dir / f"shadow_meta_candidates_{run_id}_MERGED.parquet",
        index=False,
    )


def test_build_monday_zero_trade_should_have_trade_audit_classifies_zero_weeks(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    low_prices = [100.0 + (0.01 * (idx % 5)) for idx in range(300)]
    high_prices = [100.0 + (0.5 * idx) for idx in range(300)]
    mid_prices = [100.0 + (0.15 * idx) for idx in range(300)]

    _make_run(
        reports_root,
        "TRUTH_MONFRI_WEEK_20250106_20250113",
        n_trades=12,
        prices=mid_prices,
        candidate_rows=300,
        candidate_below_threshold=120,
        candidate_flat_veto=90,
    )
    _make_run(
        reports_root,
        "TRUTH_MONFRI_WEEK_20250113_20250120",
        n_trades=8,
        prices=[100.0 + (0.2 * idx) for idx in range(300)],
        candidate_rows=300,
        candidate_below_threshold=140,
        candidate_flat_veto=100,
    )
    _make_run(
        reports_root,
        "TRUTH_MONFRI_WEEK_20250120_20250127",
        n_trades=0,
        prices=low_prices,
        candidate_rows=300,
        candidate_below_threshold=300,
        candidate_flat_veto=300,
    )
    _make_run(
        reports_root,
        "TRUTH_MONFRI_WEEK_20250127_20250203",
        n_trades=0,
        prices=high_prices,
        candidate_rows=300,
        candidate_below_threshold=300,
        candidate_flat_veto=300,
    )

    payload = build_monday_zero_trade_should_have_trade_audit(reports_root, sample_limit=5)
    verdict_df = payload["verdict_df"]
    verdicts = {
        row["run_id"]: row["verdict_v1"]
        for row in verdict_df[["run_id", "verdict_v1"]].to_dict(orient="records")
    }
    hard_status = {
        row["run_id"]: row["hard_status_v1"]
        for row in verdict_df[["run_id", "hard_status_v1"]].to_dict(orient="records")
    }

    assert verdicts["TRUTH_MONFRI_WEEK_20250120_20250127"] == "TRUE_NO_TRADE_REGIME"
    assert hard_status["TRUTH_MONFRI_WEEK_20250120_20250127"] == "BEVIST"
    assert verdicts["TRUTH_MONFRI_WEEK_20250127_20250203"] == "OVERFILTERED_SHOULD_HAVE_TRADED"
    assert hard_status["TRUTH_MONFRI_WEEK_20250127_20250203"] == "INDIKERT"


def test_materialize_monday_zero_trade_should_have_trade_audit_writes_artifacts(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    _make_run(
        reports_root,
        "TRUTH_MONFRI_WEEK_20250106_20250113",
        n_trades=6,
        prices=[100.0 + (0.15 * idx) for idx in range(300)],
        candidate_rows=300,
        candidate_below_threshold=90,
        candidate_flat_veto=90,
    )
    _make_run(
        reports_root,
        "TRUTH_MONFRI_WEEK_20250113_20250120",
        n_trades=0,
        prices=[100.0 + (0.01 * (idx % 3)) for idx in range(300)],
        candidate_rows=300,
        candidate_below_threshold=300,
        candidate_flat_veto=300,
    )

    paths = materialize_monday_zero_trade_should_have_trade_audit(reports_root, sample_limit=3)
    for path in paths.values():
        assert Path(path).exists()

    summary = json.loads(Path(paths["summary_v1"]).read_text(encoding="utf-8"))
    assert summary["completed_zero_trade_runs_v1"] == 1
