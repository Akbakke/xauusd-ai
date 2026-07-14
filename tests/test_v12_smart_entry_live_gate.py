from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.execution import v12_smart_entry_live as live


REQUIRED_REPAIR_POCKETS = {
    "rising_channel_support_touch",
    "support_retest_continuation",
    "rising_channel_support_continuation",
    "countertrend_short_trap",
    "short_high_mae_low_mfe_early_failure",
    "falling_channel_resistance_touch",
    "resistance_retest_continuation",
    "falling_channel_resistance_continuation",
    "countertrend_long_trap",
    "long_high_mae_low_mfe_early_failure",
}


def _write_gate_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    pockets: dict[str, dict],
    git_commit: str = "unit-clean-commit",
    normalize_pockets: bool = True,
) -> Path:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    rank_ref = tmp_path / "smart520_rank_reference_xau_direction_repair.npz"
    np.savez_compressed(
        rank_ref,
        time_ns=np.asarray([pd.Timestamp("2026-05-21T00:00:00Z").value], dtype=np.int64),
        vol_regime_id=np.asarray([2], dtype=np.int64),
        spread_bucket=np.asarray([0], dtype=np.int64),
        atr_pinned=np.asarray([1.0], dtype=np.float64),
        atr_bps_sorted=np.asarray([10.0], dtype=np.float64),
        spread_bps_sorted=np.asarray([1.0], dtype=np.float64),
    )
    rank_ref_sha = hashlib.sha256(rank_ref.read_bytes()).hexdigest()
    rank_ref.with_suffix(rank_ref.suffix + ".json").write_text(
        json.dumps(
            {
                "schema_version": "smart520_rank_reference_v1",
                "out_npz": str(rank_ref),
                "out_npz_sha256": rank_ref_sha,
                "row_count": 1,
                "time_min": "2026-05-21 00:00:00+00:00",
                "time_max": "2026-05-21 00:00:00+00:00",
                "source_parquet_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    state_contract = {
        "schema_version": "smart520_state_contract_v1",
        "frame_anchor_utc": "2026-05-21 00:00:00+00:00",
        "model_range_start_utc": "2020-11-09 00:00:00+00:00",
        "rank_reference_end_utc": "2026-06-14 23:59:59+00:00",
        "rank_reference_npz": str(rank_ref),
        "rank_reference_npz_sha256": rank_ref_sha,
        "time_split_reference_split": "test",
    }
    (bundle_dir / "bundle_metadata.json").write_text(
        json.dumps({"smart520_state_contract": state_contract}),
        encoding="utf-8",
    )
    now = pd.Timestamp.now(tz="UTC").isoformat()
    parity_path = tmp_path / "SMART520_SERVE_PARITY_latest.json"
    direction_path = tmp_path / "SMART_DIRECTION_LIVE_LIKE_POCKET_AUDIT_latest.json"
    if normalize_pockets:
        pockets = {name: _passing_pocket_metrics(name, row) for name, row in pockets.items()}
    parity_path.write_text(
        json.dumps(
            {
                "decision": "PASS",
                "created_utc": now,
                "live_prebuilt_cutoff": now,
                "bundle_dir": str(bundle_dir),
                "dataset_dir": "/home/andre2/GX1_DATA/runs/v10_dataset_6yr_smartctx_xau_direction_repair",
                "git_commit": git_commit,
                "smart520_state_contract": state_contract,
            }
        ),
        encoding="utf-8",
    )
    direction_path.write_text(
        json.dumps(
            {
                "decision": "PASS",
                "created_utc": now,
                "max_bad_side_rate": 0.35,
                "min_selected_rows": 30,
                "bundle_dir": str(bundle_dir),
                "predictions_parquet": "/home/andre2/GX1_DATA/reports/xau_direction_repair_predictions.parquet",
                "dataset_dir": "/home/andre2/GX1_DATA/runs/v10_dataset_6yr_smartctx_xau_direction_repair",
                "required_selection_score_mode": "expected_utility",
                "observed_selection_score_modes": ["expected_utility"],
                "pockets": pockets,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(live, "SMART_PARITY_GATE_LATEST", parity_path)
    monkeypatch.setattr(live, "SMART_DIRECTION_AUDIT_LATEST", direction_path)
    monkeypatch.setattr(live, "SMART_PARITY_GATE_MAX_AGE_HOURS", 0.0)
    monkeypatch.setattr(live, "SMART_PARITY_GATE_MAX_CUTOFF_LAG_HOURS", 0.0)
    monkeypatch.setattr(live, "SMART_DIRECTION_AUDIT_MAX_AGE_HOURS", 0.0)
    monkeypatch.setattr(live, "SMART_CTX_MAX_STALENESS_M5", 0)
    monkeypatch.setattr(live, "_smart_gate_git_state", lambda: ("unit-clean-commit", False))

    import gx1_guards.artifacts as artifacts

    monkeypatch.setattr(
        artifacts,
        "load_decision_entry",
        lambda name: {
            "path": str(bundle_dir),
            "contract_mode": "smart_seq520_candidate",
            "operating_point": {
                "edge_score_threshold": 0.145,
                "selection_score": "expected_utility",
                "expected_utility_threshold_bps": 0.0,
                "sessions": ["US"],
            },
        },
    )
    return bundle_dir


def _passing_pocket_metrics(name: str, overrides: dict | None = None) -> dict:
    short_bad = {
        "rising_channel_support_touch",
        "support_retest_continuation",
        "rising_channel_support_continuation",
        "countertrend_short_trap",
        "short_high_mae_low_mfe_early_failure",
    }
    long_bad = {
        "falling_channel_resistance_touch",
        "resistance_retest_continuation",
        "falling_channel_resistance_continuation",
        "countertrend_long_trap",
        "long_high_mae_low_mfe_early_failure",
    }
    row = {
        "rows": 40,
        "selected_rows": 30,
        "selected_side_long_rate": 0.20 if name in long_bad else 0.80,
        "selected_side_short_rate": 0.20 if name in short_bad else 0.80,
        "selected_mean_proxy_pnl_bps": 12.0,
    }
    if overrides:
        row.update(overrides)
    return row


def test_smart_serving_gate_rejects_old_direction_audit_without_repair_pockets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_gate_artifacts(
        tmp_path,
        monkeypatch,
        pockets={
            "rising_channel_support_touch": {},
            "falling_channel_resistance_touch": {},
        },
    )

    with pytest.raises(RuntimeError, match="required XAU direction-repair pockets"):
        live.assert_smart_serving_gate()


def test_smart_serving_gate_accepts_direction_audit_with_repair_pockets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _write_gate_artifacts(
        tmp_path,
        monkeypatch,
        pockets={name: {} for name in REQUIRED_REPAIR_POCKETS},
    )

    report = live.assert_smart_serving_gate()

    assert report["decision"] == "PASS"
    assert report["bundle_dir"] == str(bundle_dir)


def test_smart_serving_gate_rejects_empty_direction_pocket_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_gate_artifacts(
        tmp_path,
        monkeypatch,
        pockets={name: {} for name in REQUIRED_REPAIR_POCKETS},
        normalize_pockets=False,
    )

    with pytest.raises(RuntimeError, match="lacks integer rows/selected_rows"):
        live.assert_smart_serving_gate()


def test_smart_serving_gate_rejects_git_commit_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_gate_artifacts(
        tmp_path,
        monkeypatch,
        pockets={name: {} for name in REQUIRED_REPAIR_POCKETS},
        git_commit="old-commit",
    )

    with pytest.raises(RuntimeError, match="git_commit"):
        live.assert_smart_serving_gate()


def test_smart_serving_gate_requires_expected_utility_direction_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_gate_artifacts(
        tmp_path,
        monkeypatch,
        pockets={name: {} for name in REQUIRED_REPAIR_POCKETS},
    )
    data = json.loads(live.SMART_DIRECTION_AUDIT_LATEST.read_text(encoding="utf-8"))
    data["required_selection_score_mode"] = "edge_score"
    live.SMART_DIRECTION_AUDIT_LATEST.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RuntimeError, match="expected_utility"):
        live.assert_smart_serving_gate()


def test_smart_serving_gate_rejects_stale_xau_dataset_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_gate_artifacts(
        tmp_path,
        monkeypatch,
        pockets={name: {} for name in REQUIRED_REPAIR_POCKETS},
    )
    data = json.loads(live.SMART_DIRECTION_AUDIT_LATEST.read_text(encoding="utf-8"))
    data["dataset_dir"] = "/home/andre2/GX1_DATA/runs/v10_dataset_smart_candidate_20260630"
    live.SMART_DIRECTION_AUDIT_LATEST.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RuntimeError, match="stale XAU repair marker"):
        live.assert_smart_serving_gate()


def test_smart_serving_gate_rejects_stale_parity_dataset_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_gate_artifacts(
        tmp_path,
        monkeypatch,
        pockets={name: {} for name in REQUIRED_REPAIR_POCKETS},
    )
    data = json.loads(live.SMART_PARITY_GATE_LATEST.read_text(encoding="utf-8"))
    data["dataset_dir"] = "/home/andre2/GX1_DATA/runs/v10_dataset_smart_candidate_julyext_20260705"
    live.SMART_PARITY_GATE_LATEST.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RuntimeError, match="parity dataset_dir references stale"):
        live.assert_smart_serving_gate()


def test_smart_serving_gate_rejects_stale_rank_reference_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_gate_artifacts(
        tmp_path,
        monkeypatch,
        pockets={name: {} for name in REQUIRED_REPAIR_POCKETS},
    )
    data = json.loads(live.SMART_PARITY_GATE_LATEST.read_text(encoding="utf-8"))
    data["smart520_state_contract"]["rank_reference_npz"] = (
        "/home/andre2/GX1_DATA/models/smart520_rank_reference_julyext_20260708.npz"
    )
    live.SMART_PARITY_GATE_LATEST.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RuntimeError, match="rank_reference_npz references stale marker"):
        live.assert_smart_serving_gate()


def test_smart_serving_gate_requires_zero_context_staleness_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_gate_artifacts(
        tmp_path,
        monkeypatch,
        pockets={name: {} for name in REQUIRED_REPAIR_POCKETS},
    )
    monkeypatch.setattr(live, "SMART_CTX_MAX_STALENESS_M5", 1)

    with pytest.raises(RuntimeError, match="GX1_SMART_CTX_MAX_STALENESS_M5 must be 0"):
        live.assert_smart_serving_gate()


def test_smart_entry_mtf_window_uses_closed_bar_availability_shift(tmp_path: Path) -> None:
    idx = pd.DatetimeIndex(
        [
            "2026-07-08T12:00:00Z",
            "2026-07-08T12:05:00Z",
        ]
    )
    frame = pd.DataFrame(index=idx)
    frame.attrs["ts_int64"] = idx.asi8.astype("int64")
    frame.attrs["feats_np"] = np.asarray([[1.0], [2.0]], dtype=np.float32)
    engine = live.SmartEntryLiveInference(
        bundle_dir=tmp_path,
        operating_point={"edge_score_threshold": 0.10, "sessions": ["US"]},
    )
    engine._per_tf_seq_lens = {"M5": 1}
    engine._multi_tf_shift = {"M5": pd.Timedelta(minutes=5)}

    out = engine._multi_tf_window_tensors(
        pd.Timestamp("2026-07-08T12:05:00Z"),
        multi_tf={"M5": frame},
    )

    assert float(out["seq_m5"][0, 0, 0].item()) == 2.0


def test_smart_decision_preserves_trendline_rail_evidence(tmp_path: Path) -> None:
    engine = live.SmartEntryLiveInference(
        bundle_dir=tmp_path,
        operating_point={
            "edge_score_threshold": 0.10,
            "sessions": ["US"],
            "selection_score": "expected_utility",
            "expected_utility_threshold_bps": 0.0,
        },
    )
    head = {
        "time": pd.Timestamp("2026-07-08T18:00:00Z"),
        "session_id": 3,
        "p_long": 0.21,
        "p_short": 0.62,
        "p_flat": 0.17,
        "edge_score": 0.45,
        "trade_side": 1,
        "path_quality_pred": 1.5,
        "bad_path_prob": 0.08,
        "tradable_prob": 0.73,
        "mfe_first_n_pred": 12.0,
        "side_validity_logit": [2.0, -2.0],
        "long_validity_prob": 0.88,
        "short_validity_prob": 0.12,
        "expected_utility_invalid_side_penalty_bps": 35.0,
        "expected_utility_long_invalid_side_penalty_bps": 4.2,
        "expected_utility_short_invalid_side_penalty_bps": 30.8,
        "expected_utility_long_bps": 12.0,
        "expected_utility_short_bps": -4.0,
        "expected_utility_side": 0,
        "geometry_rising_support_rail_long_pressure": 0.82,
        "geometry_rising_support_rail_short_trap_pressure": 0.76,
        "geometry_falling_resistance_rail_short_pressure": 0.03,
        "geometry_falling_resistance_rail_long_trap_pressure": 0.04,
        "trendline_rail_long_evidence": 0.43,
        "trendline_rail_short_evidence": 0.395,
        "trendline_rail_long_minus_short": 0.035,
        "mtf_trend_evidence": 0.71,
    }

    decision = engine.decide(head, atr_bps=9.0)
    snapshot = decision["_v10_snapshot"]

    assert decision["action"] == "TAKE_LONG_NOW"
    assert decision["selection_score_mode"] == "expected_utility"
    assert decision["legacy_trade_side"] == 1
    assert decision["expected_utility_side"] == 0
    assert decision["selected_side"] == 0
    assert decision["geometry_rising_support_rail_long_pressure"] == 0.82
    assert decision["geometry_rising_support_rail_short_trap_pressure"] == 0.76
    assert decision["trendline_rail_long_minus_short"] == 0.035
    assert decision["long_validity_prob"] == 0.88
    assert decision["short_validity_prob"] == 0.12
    assert decision["expected_utility_long_invalid_side_penalty_bps"] == 4.2
    assert snapshot["geometry_rising_support_rail_long_pressure"] == 0.82
    assert snapshot["trendline_rail_long_minus_short"] == 0.035
    assert snapshot["long_validity_prob"] == 0.88
    assert snapshot["legacy_trade_side"] == 1
    assert snapshot["selected_side"] == 0


def test_smart_decision_recomputes_expected_utility_side_when_missing(tmp_path: Path) -> None:
    engine = live.SmartEntryLiveInference(
        bundle_dir=tmp_path,
        operating_point={
            "edge_score_threshold": 0.10,
            "sessions": ["US"],
            "selection_score": "expected_utility",
            "expected_utility_threshold_bps": 0.0,
        },
    )
    head = {
        "time": pd.Timestamp("2026-07-08T18:00:00Z"),
        "session_id": 3,
        "p_long": 0.55,
        "p_short": 0.45,
        "p_flat": 0.0,
        "edge_score": 0.45,
        "trade_side": 0,
        "path_quality_pred": 1.5,
        "bad_path_prob": 0.08,
        "tradable_prob": 0.73,
        "mfe_first_n_pred": 12.0,
        "expected_utility_long_bps": -2.0,
        "expected_utility_short_bps": 8.0,
    }

    decision = engine.decide(head, atr_bps=9.0)

    assert decision["action"] == "TAKE_SHORT_NOW"
    assert decision["selected_side"] == 1


def test_smart_decision_requires_expected_utility_heads_in_expected_utility_mode(tmp_path: Path) -> None:
    engine = live.SmartEntryLiveInference(
        bundle_dir=tmp_path,
        operating_point={
            "edge_score_threshold": 0.10,
            "sessions": ["US"],
            "selection_score": "expected_utility",
            "expected_utility_threshold_bps": 0.0,
        },
    )
    head = {
        "time": pd.Timestamp("2026-07-08T18:00:00Z"),
        "session_id": 3,
        "p_long": 0.55,
        "p_short": 0.45,
        "p_flat": 0.0,
        "edge_score": 0.45,
        "trade_side": 0,
        "path_quality_pred": 1.5,
        "bad_path_prob": 0.08,
        "tradable_prob": 0.73,
        "mfe_first_n_pred": 12.0,
    }

    with pytest.raises(RuntimeError, match="requires utility heads"):
        engine.decide(head, atr_bps=9.0)


def test_smart_decision_rejects_mismatched_expected_utility_side(tmp_path: Path) -> None:
    engine = live.SmartEntryLiveInference(
        bundle_dir=tmp_path,
        operating_point={
            "edge_score_threshold": 0.10,
            "sessions": ["US"],
            "selection_score": "expected_utility",
            "expected_utility_threshold_bps": 0.0,
        },
    )
    head = {
        "time": pd.Timestamp("2026-07-08T18:00:00Z"),
        "session_id": 3,
        "p_long": 0.55,
        "p_short": 0.45,
        "p_flat": 0.0,
        "edge_score": 0.45,
        "trade_side": 0,
        "path_quality_pred": 1.5,
        "bad_path_prob": 0.08,
        "tradable_prob": 0.73,
        "mfe_first_n_pred": 12.0,
        "expected_utility_long_bps": -2.0,
        "expected_utility_short_bps": 8.0,
        "expected_utility_side": 0,
    }

    with pytest.raises(RuntimeError, match="expected_utility_side mismatch"):
        engine.decide(head, atr_bps=9.0)
