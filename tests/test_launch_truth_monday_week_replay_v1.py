from __future__ import annotations

import json
from pathlib import Path

from gx1.scripts.launch_truth_monday_week_replay_v1 import _postrun_status


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def test_postrun_status_treats_completed_artifacts_as_completed_even_with_nonzero_returncode(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    run_id = "TRUTH_MONFRI_WEEK_20260316_20260323"
    run_dir = reports_root / run_id
    _write_json(run_dir / "RUN_COMPLETED.json", {"status": "COMPLETED"})
    _write_json(run_dir / "POSTRUN_E2E.json", {"passed": True, "gates_failed": []})
    _write_json(
        run_dir / "replay" / "chunk_0" / "chunk_footer.json",
        {"status": "ok", "n_trades_closed": 54, "bars_processed": 6381, "wall_clock_sec": 5841.0},
    )
    (run_dir / f"trade_outcomes_{run_id}_MERGED.parquet").write_text("", encoding="utf-8")

    status = _postrun_status(reports_root, run_id, returncode=-9)

    assert status["status"] == "COMPLETED"
    assert status["returncode_anomaly"] is True
    assert status["returncode_anomaly_reason"] == "PROCESS_NONZERO_AFTER_COMPLETED_ARTIFACTS"
    assert status["postrun_passed"] is True
    assert status["run_completed_exists"] is True


def test_postrun_status_marks_failed_when_completed_artifacts_are_missing(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    run_id = "TRUTH_MONFRI_WEEK_20250421_20250428"
    run_dir = reports_root / run_id
    _write_json(run_dir / "POSTRUN_E2E.json", {"passed": False, "gates_failed": ["missing"]})

    status = _postrun_status(reports_root, run_id, returncode=-9)

    assert status["status"] == "FAILED_OR_INCOMPLETE"
    assert status["postrun_passed"] is False
    assert status["run_completed_exists"] is False
