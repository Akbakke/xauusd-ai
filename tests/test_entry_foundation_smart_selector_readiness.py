from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_entry_foundation_smart_selector_readiness_v1 import (
    TEST_LOG_NAME,
    VALIDATION_LOG_NAME,
    run,
)


def _write_trades(path: Path, *, split: str, policy_id: str, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    frame["entry_time"] = pd.date_range(
        "2025-01-01T00:00:00Z" if split == "validation" else "2026-01-01T00:00:00Z",
        periods=len(frame),
        freq="5min",
    )
    frame["policy_id"] = policy_id
    frame["student_trade_log_split"] = split
    frame["diagnostic_only_not_replay_evidence"] = split == "validation"
    frame["exit_mode"] = "stop_tp"
    frame["mae_bps"] = 10.0
    frame.to_csv(path, index=False)


def _rows(side: str, session: str, vol_regime: str, pnl: list[float]) -> list[dict]:
    return [
        {
            "side": side,
            "session": session,
            "vol_regime": vol_regime,
            "net_pnl_bps": value,
        }
        for value in pnl
    ]


def test_selector_readiness_uses_validation_only_and_blocks_side_effects(tmp_path: Path) -> None:
    foundation = tmp_path / "foundation"
    smart = tmp_path / "smart"
    foundation.mkdir()
    smart.mkdir()

    _write_trades(
        foundation / VALIDATION_LOG_NAME,
        split="validation",
        policy_id="foundation_policy",
        rows=_rows("LONG", "EU", "3", [5.0] * 4) + _rows("SHORT", "ASIA", "4", [10.0] * 4),
    )
    _write_trades(
        smart / VALIDATION_LOG_NAME,
        split="validation",
        policy_id="smart_policy",
        rows=_rows("LONG", "EU", "3", [20.0] * 4) + _rows("SHORT", "ASIA", "4", [1.0] * 4),
    )
    _write_trades(
        foundation / TEST_LOG_NAME,
        split="test",
        policy_id="foundation_policy",
        rows=_rows("LONG", "EU", "3", [100.0] * 4) + _rows("SHORT", "ASIA", "4", [100.0] * 4),
    )
    _write_trades(
        smart / TEST_LOG_NAME,
        split="test",
        policy_id="smart_policy",
        rows=_rows("LONG", "EU", "3", [-50.0] * 4) + _rows("SHORT", "ASIA", "4", [-50.0] * 4),
    )

    report = run(
        type(
            "Args",
            (),
            {
                "foundation_dir": str(foundation),
                "smart_dir": str(smart),
                "out_dir": str(tmp_path / "out"),
                "cubes": "ALL,side,session,vol_regime",
                "min_validation_slice_trades": 2,
                "min_validation_net_lift_bps": 0.0,
                "fail_on_not_ready": True,
                "quiet": True,
            },
        )()
    )

    assert report["decision"] == "ENTRY_FOUNDATION_SMART_SELECTOR_READINESS_READY_FOR_REVIEW"
    assert report["selector_uses_validation_only"] is True
    assert report["test_diagnostic_only_not_selection_criterion"] is True
    assert report["selector_training_started"] is False
    assert report["replay_started"] is False
    assert report["iql_distillation_started"] is False
    assert report["promotion_shadow_live_allowed"] is False

    wins = {(row["cube"], row["slice"]) for row in report["smart_supported_validation_wins"]}
    assert ("side", "LONG") in wins
    long_row = next(row for row in report["smart_supported_validation_wins"] if row["cube"] == "side" and row["slice"] == "LONG")
    assert long_row["selected_by_validation_only"] == "smart"
    assert long_row["test_diagnostic_only_not_selection_criterion"] is True
    assert long_row["test_diagnostic"]["smart"]["net_sum_bps"] < long_row["test_diagnostic"]["foundation"]["net_sum_bps"]
    assert Path(report["metrics_csv"]).is_file()
    assert Path(report["selector_candidates_json"]).is_file()
