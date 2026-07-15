import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from gx1.scripts import audit_xau_red_slice_separability_v1 as audit


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, *, xau_path: bool = True) -> tuple[Path, Path]:
    stem = "fixture_xau_direction_repair_val" if xau_path else "fixture_eurusd_val"
    parquet = tmp_path / f"{stem}.parquet"
    fields = [
        "p_long",
        "p_short",
        "p_flat",
        "p_hat",
        "uncertainty_score",
        "margin_top1_top2",
        "entropy",
        "chart.geometry_rising_support_rail_long_pressure",
        "chart.geometry_rising_support_rail_short_trap_pressure",
        "chart.geometry_falling_resistance_rail_short_pressure",
        "chart.geometry_falling_resistance_rail_long_trap_pressure",
        "chart.sr_memory_support_respect_pressure_long",
    ]
    snap_rows = [
        [0.0] * 7 + [1.0, 0.1, 0.2, 0.0, 1.0],
        [0.0] * 7 + [1.2, 0.2, 0.1, 0.0, 1.1],
        [0.0] * 7 + [0.1, 1.0, 1.2, 0.1, 0.2],
        [0.0] * 7 + [0.2, 1.1, 1.1, 0.2, 0.1],
        [0.0] * 7 + [0.6, 0.5, 0.4, 0.3, 0.5],
    ]
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC"),
            "snap": snap_rows,
            "ctx_cont": [[0.0] for _ in range(5)],
            "ctx_cat": [[1, 2, 0, 0, 1] for _ in range(5)],
            "y_direction": [0, 0, 1, 1, 2],
            "y_long_path_utility_bps": [10.0, 12.0, -5.0, -6.0, 0.0],
            "y_short_path_utility_bps": [-4.0, -3.0, 8.0, 9.0, 0.0],
            "y_long_bad_path": [0.0, 0.0, 1.0, 1.0, 0.0],
            "y_short_bad_path": [1.0, 1.0, 0.0, 0.0, 0.0],
            "y_rising_channel_support_touch": [1.0, 1.0, 0.0, 0.0, 0.0],
            "y_falling_channel_resistance_touch": [0.0, 0.0, 1.0, 1.0, 0.0],
            "y_countertrend_short_trap": [1.0, 1.0, 0.0, 0.0, 0.0],
            "y_countertrend_long_trap": [0.0, 0.0, 1.0, 1.0, 0.0],
        }
    )
    frame.to_parquet(parquet, index=False)
    _write_json(
        parquet.with_suffix(".manifest.json"),
        {
            "extra": {
                "signal_bridge": {"fields": fields},
                "ctx_contract": {
                    "ctx_cont_names": ["ctx0"],
                    "ctx_cat_names": [
                        "session_id",
                        "vol_regime_id",
                        "atr_bucket",
                        "spread_bucket",
                        "H4_trend_sign_cat",
                    ],
                },
            }
        },
    )
    evidence = tmp_path / "failure_evidence.json"
    _write_json(
        evidence,
        {
            "decision": "FAIL_DIRECTION_SLICE_GUARD",
            "failure_code": "TRAIN_FAIL_DIRECTION_SLICE_GUARD",
            "best_epoch": 1,
            "last_epoch": 1,
            "best_direction_balance_guard_ok": True,
            "best_direction_slice_contract_ok": False,
            "val_data": str(parquet),
            "best_direction_slice_stats": {
                "direction_slice_failure_count": 1,
                "direction_slice_failure_details": [
                    {
                        "ctx_cat_index": 0,
                        "ctx_cat_value": 1,
                        "rows": 5,
                        "label_rates": [0.4, 0.4, 0.2],
                        "pred_rates": [0.1, 0.8, 0.1],
                        "pred_rate_failed_classes": [0],
                    }
                ],
            },
        },
    )
    return evidence, parquet


def test_xau_red_slice_separability_audit_maps_slice_and_required_features(tmp_path: Path) -> None:
    evidence, _ = _fixture(tmp_path)

    report = audit.run(
        argparse.Namespace(
            evidence_json=str(evidence),
            out_dir=str(tmp_path / "reports"),
            top_features_per_slice=4,
            weak_required_feature_std_delta=0.10,
            quiet=True,
            no_fail_on_audit_fail=False,
        )
    )

    assert report["decision"] == "XAU_RED_SLICE_SEPARABILITY_AUDIT_COMPLETE"
    assert report["training_allowed"] is False
    assert report["iql_allowed"] is False
    assert not any(report["side_effects_started"].values())
    slice_report = report["slice_reports"][0]
    assert slice_report["ctx_cat_name"] == "session_id"
    assert slice_report["rows"] == 5
    assert slice_report["label_rates"] == {"LONG": 0.4, "SHORT": 0.4, "FLAT": 0.2}
    required = slice_report["required_feature_summaries"]
    assert required["chart.geometry_rising_support_rail_long_pressure"]["long_minus_short_mean"] > 0
    assert Path(report["json_path"]).exists()


def test_xau_red_slice_separability_audit_rejects_non_xau_val_data(tmp_path: Path) -> None:
    evidence, _ = _fixture(tmp_path, xau_path=False)

    with pytest.raises(RuntimeError, match="XAU direction-repair path"):
        audit.run(
            argparse.Namespace(
                evidence_json=str(evidence),
                out_dir=str(tmp_path / "reports"),
                top_features_per_slice=4,
                weak_required_feature_std_delta=0.10,
                quiet=True,
                no_fail_on_audit_fail=False,
            )
        )
