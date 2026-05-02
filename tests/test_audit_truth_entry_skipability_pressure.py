from __future__ import annotations

from pathlib import Path

import pandas as pd

from gx1.scripts.audit_truth_entry_skipability_pressure import build_skipability_pressure_summary


def _write_trade_outcomes(path: Path, trade_ids: list[str]) -> None:
    df = pd.DataFrame({"trade_id": trade_ids})
    df.to_parquet(path, index=False)


def _write_candidates(
    path: Path,
    *,
    sessions: list[str],
    decisions: list[str],
    accepted: list[bool],
    reasons: list[str],
) -> None:
    df = pd.DataFrame(
        {
            "session": sessions,
            "decision": decisions,
            "accepted": accepted,
            "decision_reason": reasons,
        }
    )
    df.to_parquet(path, index=False)


def test_build_skipability_pressure_summary_flags_candidate_rich_zero_trade_runs(tmp_path: Path) -> None:
    reports_root = tmp_path / "truth_root"
    runs_root = reports_root / "runs"
    runs_root.mkdir(parents=True)

    zero_run = runs_root / "E2E_SANITY_ORDERFIX_20250101_20250108"
    zero_run.mkdir()
    (zero_run / "RUN_COMPLETED.json").write_text("{}\n", encoding="utf-8")
    _write_trade_outcomes(
        zero_run / "trade_outcomes_E2E_SANITY_ORDERFIX_20250101_20250108_MERGED.parquet",
        [],
    )
    _write_candidates(
        zero_run / "shadow_meta_candidates_E2E_SANITY_ORDERFIX_20250101_20250108_MERGED.parquet",
        sessions=["EU"] * 1000,
        decisions=["NONE"] * 1000,
        accepted=[False] * 1000,
        reasons=["flat_dominant"] * 900 + ["flat_veto"] * 100,
    )

    nonzero_run = runs_root / "E2E_SANITY_ORDERFIX_20250108_20250115"
    nonzero_run.mkdir()
    (nonzero_run / "RUN_COMPLETED.json").write_text("{}\n", encoding="utf-8")
    _write_trade_outcomes(
        nonzero_run / "trade_outcomes_E2E_SANITY_ORDERFIX_20250108_20250115_MERGED.parquet",
        ["1", "2", "3", "4"],
    )
    _write_candidates(
        nonzero_run / "shadow_meta_candidates_E2E_SANITY_ORDERFIX_20250108_20250115_MERGED.parquet",
        sessions=["EU"] * 100 + ["US"] * 100,
        decisions=["LONG"] * 4 + ["NONE"] * 196,
        accepted=[True] * 4 + [False] * 196,
        reasons=["pre_quality"] * 4 + ["flat_dominant"] * 180 + ["flat_veto"] * 16,
    )

    summary = build_skipability_pressure_summary(reports_root=reports_root, sample_limit=5)

    assert summary["completed_runs"] == 2
    assert summary["completed_zero_trade_runs"] == 1
    assert summary["candidate_rich_zero_trade_runs"] == 1
    assert summary["candidate_rich_zero_trade_run_ids"] == [
        "E2E_SANITY_ORDERFIX_20250101_20250108"
    ]
    assert summary["zero_trade_reason_mix_top8"]["flat_dominant"] == 900
    assert summary["zero_trade_reason_mix_top8"]["flat_veto"] == 100
    assert summary["verdicts"]["zero_trade_candidate_surface_status"] == "FAIL"
    assert summary["verdicts"]["zero_trade_acceptance_status"] == "FAIL"


def test_build_skipability_pressure_summary_supports_monday_run_ids_without_runs_subdir(tmp_path: Path) -> None:
    reports_root = tmp_path / "truth_root"
    reports_root.mkdir(parents=True)

    zero_run = reports_root / "TRUTH_MONFRI_WEEK_20260406_20260413"
    zero_run.mkdir()
    (zero_run / "RUN_COMPLETED.json").write_text("{}\n", encoding="utf-8")
    _write_trade_outcomes(
        zero_run / "trade_outcomes_TRUTH_MONFRI_WEEK_20260406_20260413_MERGED.parquet",
        [],
    )
    _write_candidates(
        zero_run / "shadow_meta_candidates_TRUTH_MONFRI_WEEK_20260406_20260413_MERGED.parquet",
        sessions=["OVERLAP"] * 1200,
        decisions=["NONE"] * 1200,
        accepted=[False] * 1200,
        reasons=["flat_veto"] * 1200,
    )

    nonzero_run = reports_root / "TRUTH_MONFRI_WEEK_20260413_20260420"
    nonzero_run.mkdir()
    (nonzero_run / "RUN_COMPLETED.json").write_text("{}\n", encoding="utf-8")
    _write_trade_outcomes(
        nonzero_run / "trade_outcomes_TRUTH_MONFRI_WEEK_20260413_20260420_MERGED.parquet",
        ["1", "2"],
    )
    _write_candidates(
        nonzero_run / "shadow_meta_candidates_TRUTH_MONFRI_WEEK_20260413_20260420_MERGED.parquet",
        sessions=["OVERLAP"] * 10,
        decisions=["LONG", "LONG"] + ["NONE"] * 8,
        accepted=[True, True] + [False] * 8,
        reasons=["pre_quality", "pre_quality"] + ["flat_dominant"] * 8,
    )

    summary = build_skipability_pressure_summary(reports_root=reports_root, sample_limit=5)

    assert summary["completed_runs"] == 2
    assert summary["completed_zero_trade_runs"] == 1
    assert summary["candidate_rich_zero_trade_run_ids"] == ["TRUTH_MONFRI_WEEK_20260406_20260413"]
    assert summary["zero_trade_runs_detail_sample"][0]["start_date"] == "2026-04-06"
