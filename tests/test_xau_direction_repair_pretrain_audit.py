import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.scripts.audit_xau_direction_repair_pretrain_v1 import (
    DEFAULT_STEM,
    REQUIRED_RAIL_FEATURES,
    run,
    build_parser,
)


def _write_split(
    root: Path,
    split: str,
    *,
    inverted: bool,
    include_rail: bool = True,
    include_inline_seq_structure: bool = True,
    bad_path_mismatch: bool = False,
    anti_short_wrong_side: bool = False,
    alias_mismatch: bool = False,
    tape_root: str = "/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL",
    stem: str = DEFAULT_STEM,
) -> None:
    rank_ref = root / "smart520_rank_reference_xau_direction_repair.npz"
    if not rank_ref.exists():
        np.savez_compressed(
            rank_ref,
            time_ns=np.asarray([pd.Timestamp("2026-01-01T00:00:00Z").value], dtype=np.int64),
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
                    "time_min": "2026-01-01 00:00:00+00:00",
                    "time_max": "2026-01-01 00:00:00+00:00",
                    "source_parquet_sha256": "a" * 64,
                }
            ),
            encoding="utf-8",
        )
    rank_ref_sha = hashlib.sha256(rank_ref.read_bytes()).hexdigest()
    fields = [f"f{i}" for i in range(12)]
    feature_positions = {
        "chart.geometry_support_line_proximity_stack": 1,
        "chart.geometry_resistance_line_proximity_stack": 2,
        "chart.geometry_support_minus_resistance_stack": 3,
        "chart.geometry_channel_position_low_to_high": 4,
    }
    for name, idx in feature_positions.items():
        fields[idx] = name
    if include_rail:
        fields.extend(REQUIRED_RAIL_FEATURES)

    rows = []
    for i in range(80):
        support_dom = i < 40
        flat = i % 10 == 0
        selected_long_bad = 1.0 if support_dom and i % 17 == 1 else 0.0
        selected_short_bad = 1.0 if (not support_dom) and i % 17 == 2 else 0.0
        long_utility = 10.0 + i if support_dom else -10.0 - i
        short_utility = -10.0 - i if support_dom else 10.0 + i
        long_mae = 2.0 + i if support_dom else 3.0 + i
        short_mae = 3.0 + i if support_dom else 2.0 + i
        scalar_bad = 0.0 if flat else selected_long_bad if support_dom else selected_short_bad
        if bad_path_mismatch and i == 1:
            scalar_bad = 1.0 - scalar_bad
        support = 0.90 if support_dom else 0.40
        resistance = 0.40 if support_dom else 0.90
        sr = support - resistance
        if inverted:
            channel_position = 0.75 if support_dom else 0.25
        else:
            channel_position = 0.25 if support_dom else 0.75
        snap = [0.0] * len(fields)
        snap[1] = support
        snap[2] = resistance
        snap[3] = sr
        snap[4] = channel_position
        y_direction = 2 if flat else 0 if support_dom else 1
        y_side = 0 if support_dom else 1
        if anti_short_wrong_side and support_dom and i == 1:
            y_direction = 1
            y_side = 1
            scalar_bad = 1.0
        long_mfe = 20.0 + i if support_dom else 6.0 + i
        long_mae = 2.0 + i if support_dom else 16.0 + i
        short_mfe = 4.0 + i if support_dom else 12.0 + i
        short_mae = 14.0 + i if support_dom else 3.0 + i
        selected_mfe = 0.0 if flat else long_mfe if y_side == 0 else short_mfe
        selected_mae = 0.0 if flat else long_mae if y_side == 0 else short_mae
        row = {
                "time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=5 * i),
                "snap": snap,
                "y_direction": y_direction,
                "y_bad_path": scalar_bad,
                "y_trade": 0.0 if flat else 1.0,
                "y_tradable": 0.0 if flat else 1.0,
                "y_side": y_side,
                "y_side_mask": 0.0 if flat else 1.0,
                "mae_first_n_bps": selected_mae,
                "mfe_first_n_bps": selected_mfe,
                "path_quality_bps": selected_mfe - selected_mae,
                "y_position_size_target": 0.5 if flat else 0.65 if support_dom else 0.35,
                "mfe_long_first_n_bps": long_mfe,
                "mae_long_first_n_bps": long_mae,
                "mfe_short_first_n_bps": short_mfe,
                "mae_short_first_n_bps": short_mae,
                "y_long_path_utility_bps": long_utility,
                "y_short_path_utility_bps": short_utility,
                "y_long_bad_path": selected_long_bad if support_dom else 1.0,
                "y_short_bad_path": 1.0 if support_dom else selected_short_bad,
                "y_long_expected_mae_bps": long_mae,
                "y_short_expected_mae_bps": short_mae,
                "y_rising_channel_support_touch": 1.0 if support_dom else 0.0,
                "y_falling_channel_resistance_touch": 0.0 if support_dom else 1.0,
                "y_support_retest_continuation": 1.0 if support_dom else 0.0,
                "y_resistance_retest_continuation": 0.0 if support_dom else 1.0,
                "y_countertrend_short_trap": 1.0 if support_dom else 0.0,
                "y_countertrend_long_trap": 0.0 if support_dom else 1.0,
                "y_long_high_mae_low_mfe_early_failure": 0.0 if support_dom else 1.0,
                "y_short_high_mae_low_mfe_early_failure": 1.0 if support_dom else 0.0,
        }
        if alias_mismatch:
            row["y_direction_long_score_bps"] = long_utility + (1.0 if i == 1 else 0.0)
            row["y_direction_short_score_bps"] = short_utility
        rows.append(row)

    pd.DataFrame(rows).to_parquet(root / f"{stem}_{split}.parquet")
    manifest = {
        "neutral_xgb_bridge": True,
        "xgb_bridge_source": "neutral_uniform_proba",
        "tape_root": tape_root,
        "extra": {
            "signal_bridge": {
                "neutral_xgb_bridge": True,
                "bridge_source": "neutral_uniform_proba",
                "fields": fields,
                "snap_fields": fields,
                "seq_structure_extension_v1": {
                    "enabled": True,
                    "mode": "inline_from_merged3" if include_inline_seq_structure else "parquet_join",
                },
            },
            "smart520_state_contract": {
                "schema_version": "smart520_state_contract_v1",
                "frame_anchor_utc": "2026-01-01T00:00:00Z",
                "model_range_start_utc": "2020-11-09T00:00:00Z",
                "rank_reference_end_utc": "2026-01-01T06:35:00Z",
                "rank_reference_npz": str(rank_ref),
                "rank_reference_npz_sha256": rank_ref_sha,
                "time_split_reference_split": "test",
            },
        },
        "feature_contract": {"signal_bridge_fields": fields},
    }
    (root / f"{stem}_{split}.manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _args(tmp_path: Path, *extra: str):
    return build_parser().parse_args(
        [
            "--dataset-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "audit"),
            "--data-splits",
            "train,val,test",
            "--max-rows-per-split",
            "80",
            "--max-row-groups-per-split",
            "2",
            *extra,
        ]
    )


def test_xau_direction_repair_pretrain_audit_passes_correct_polarity(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=False)

    report = run(_args(tmp_path))

    assert report["decision"] == "PASS"
    assert report["failures"] == []


def test_xau_direction_repair_pretrain_audit_fails_inverted_channel_position(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=True)

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    latest = json.loads(
        (tmp_path / "audit" / "XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert latest["decision"] == "FAIL"
    assert any("channel_position polarity stale/inverted" in item for item in latest["failures"])


def test_xau_direction_repair_pretrain_audit_requires_rail_features(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=False, include_rail=False)

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    latest = json.loads(
        (tmp_path / "audit" / "XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert latest["decision"] == "FAIL"
    assert any("missing required XAU rail features" in item for item in latest["failures"])


def test_xau_direction_repair_pretrain_audit_requires_inline_seq_structure(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=False, include_inline_seq_structure=False)

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    latest = json.loads(
        (tmp_path / "audit" / "XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert latest["decision"] == "FAIL"
    assert any("requires inline seq-structure" in item for item in latest["failures"])


def test_xau_direction_repair_pretrain_audit_fails_bad_path_side_mismatch(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=False, bad_path_mismatch=(split == "train"))

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    latest = json.loads(
        (tmp_path / "audit" / "XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert latest["decision"] == "FAIL"
    assert any("scalar y_bad_path mismatches" in item for item in latest["failures"])


def test_xau_direction_repair_pretrain_audit_fails_anti_short_wrong_side(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=False, anti_short_wrong_side=(split == "train"))

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    latest = json.loads(
        (tmp_path / "audit" / "XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert latest["decision"] == "FAIL"
    assert any("anti-short structural rows still labeled SHORT" in item for item in latest["failures"])


def test_xau_direction_repair_pretrain_audit_fails_direction_score_alias_mismatch(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=False, alias_mismatch=(split == "train"))

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    latest = json.loads(
        (tmp_path / "audit" / "XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert latest["decision"] == "FAIL"
    assert any("y_direction_long_score_bps mismatches" in item for item in latest["failures"])


def test_xau_direction_repair_pretrain_audit_fails_non_xau_provenance(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        _write_split(
            tmp_path,
            split,
            inverted=False,
            tape_root="/home/andre2/GX1_DATA/data/oanda/canonical/foreign_fx_m5_bid_ask__CANONICAL",
        )

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    latest = json.loads(
        (tmp_path / "audit" / "XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert latest["decision"] == "FAIL"
    assert any("requires XAUUSD tape_root provenance" in item for item in latest["failures"])


def test_xau_direction_repair_pretrain_audit_can_auto_discover_split_stem(tmp_path: Path) -> None:
    stem = "v10_6yr_dataset__HOLD_03B"
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=False, stem=stem)

    report = run(_args(tmp_path, "--stem", "auto"))

    assert report["decision"] == "PASS"
    assert report["requested_stem"] == "auto"
    assert report["stem"] == stem
