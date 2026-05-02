from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.audit_truth_continuous_market_opportunity import build_continuous_market_opportunity_summary


def _write_trade_outcomes(path: Path, trade_count: int) -> None:
    pd.DataFrame({"trade_id": [f"t{i}" for i in range(trade_count)]}).to_parquet(path, index=False)


def _write_eval_log(path: Path, *, prices: list[float], session: str = "OVERLAP") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx, price in enumerate(prices):
            payload = {
                "ts_utc": f"2025-01-01T00:{idx:02d}:00+00:00" if idx < 60 else f"2025-01-01T01:{idx-60:02d}:00+00:00",
                "price": price,
                "session": session,
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def test_build_continuous_market_opportunity_summary_flags_opportunity_rich_zero_trade_run(tmp_path: Path) -> None:
    reports_root = tmp_path / "truth_root"
    runs_root = reports_root / "runs"
    runs_root.mkdir(parents=True)

    zero_low_1 = runs_root / "E2E_SANITY_ORDERFIX_20250101_20250108"
    zero_low_1.mkdir()
    (zero_low_1 / "RUN_COMPLETED.json").write_text("{}\n", encoding="utf-8")
    _write_trade_outcomes(zero_low_1 / "trade_outcomes_E2E_SANITY_ORDERFIX_20250101_20250108_MERGED.parquet", 0)
    _write_eval_log(zero_low_1 / "replay/chunk_0/logs/eval_log_test.jsonl", prices=[100.0 + (i % 2) * 0.01 for i in range(80)])

    zero_high = runs_root / "E2E_SANITY_ORDERFIX_20250108_20250115"
    zero_high.mkdir()
    (zero_high / "RUN_COMPLETED.json").write_text("{}\n", encoding="utf-8")
    _write_trade_outcomes(zero_high / "trade_outcomes_E2E_SANITY_ORDERFIX_20250108_20250115_MERGED.parquet", 0)
    _write_eval_log(zero_high / "replay/chunk_0/logs/eval_log_test.jsonl", prices=[100.0 + i * 0.5 for i in range(80)])

    zero_low_2 = runs_root / "E2E_SANITY_ORDERFIX_20250115_20250122"
    zero_low_2.mkdir()
    (zero_low_2 / "RUN_COMPLETED.json").write_text("{}\n", encoding="utf-8")
    _write_trade_outcomes(zero_low_2 / "trade_outcomes_E2E_SANITY_ORDERFIX_20250115_20250122_MERGED.parquet", 0)
    _write_eval_log(zero_low_2 / "replay/chunk_0/logs/eval_log_test.jsonl", prices=[100.0 + (i % 3) * 0.01 for i in range(80)])

    nonzero_a = runs_root / "E2E_SANITY_ORDERFIX_20250122_20250129"
    nonzero_a.mkdir()
    (nonzero_a / "RUN_COMPLETED.json").write_text("{}\n", encoding="utf-8")
    _write_trade_outcomes(nonzero_a / "trade_outcomes_E2E_SANITY_ORDERFIX_20250122_20250129_MERGED.parquet", 3)
    _write_eval_log(nonzero_a / "replay/chunk_0/logs/eval_log_test.jsonl", prices=[100.0 + i * 0.4 for i in range(80)])

    nonzero_b = runs_root / "E2E_SANITY_ORDERFIX_20250129_20250205"
    nonzero_b.mkdir()
    (nonzero_b / "RUN_COMPLETED.json").write_text("{}\n", encoding="utf-8")
    _write_trade_outcomes(nonzero_b / "trade_outcomes_E2E_SANITY_ORDERFIX_20250129_20250205_MERGED.parquet", 2)
    _write_eval_log(nonzero_b / "replay/chunk_0/logs/eval_log_test.jsonl", prices=[100.0 + i * 0.3 for i in range(80)])

    summary = build_continuous_market_opportunity_summary(reports_root, sample_limit=5)

    assert summary["completed_runs_with_market_data"] == 5
    assert summary["verdicts"]["continuous_market_data_status"] == "PASS"
    assert summary["verdicts"]["zero_trade_opportunity_rich_outlier_status"] == "FAIL"
    assert summary["opportunity_rich_zero_trade_runs_anchor"] == [
        "E2E_SANITY_ORDERFIX_20250108_20250115"
    ]
    assert summary["overall_by_horizon"]["60"]["threshold_rates_bps"]["50"] > 0.0
    assert summary["overall_by_horizon"]["60"]["backward_range_bps"]["mean"] is not None
    assert summary["zero_trade_anchor_comparison_v1"]["anchor_horizon_bars"] == 60


def test_build_continuous_market_opportunity_summary_supports_monday_top_level_runs(tmp_path: Path) -> None:
    reports_root = tmp_path / "truth_root"
    reports_root.mkdir(parents=True)

    zero_run = reports_root / "TRUTH_MONFRI_WEEK_20260406_20260413"
    zero_run.mkdir()
    (zero_run / "RUN_COMPLETED.json").write_text("{}\n", encoding="utf-8")
    _write_trade_outcomes(zero_run / "trade_outcomes_TRUTH_MONFRI_WEEK_20260406_20260413_MERGED.parquet", 0)
    _write_eval_log(zero_run / "replay/chunk_0/logs/eval_log_test.jsonl", prices=[100.0 + i * 0.25 for i in range(80)])

    nonzero_run = reports_root / "TRUTH_MONFRI_WEEK_20260413_20260420"
    nonzero_run.mkdir()
    (nonzero_run / "RUN_COMPLETED.json").write_text("{}\n", encoding="utf-8")
    _write_trade_outcomes(nonzero_run / "trade_outcomes_TRUTH_MONFRI_WEEK_20260413_20260420_MERGED.parquet", 3)
    _write_eval_log(nonzero_run / "replay/chunk_0/logs/eval_log_test.jsonl", prices=[100.0 + i * 0.2 for i in range(80)])

    summary = build_continuous_market_opportunity_summary(reports_root, sample_limit=5)

    assert summary["completed_runs_with_market_data"] == 2
    assert summary["top_zero_trade_runs_anchor"][0]["run_id"] == "TRUTH_MONFRI_WEEK_20260406_20260413"
    assert summary["top_zero_trade_runs_anchor"][0]["start_date"] == "2026-04-06"
