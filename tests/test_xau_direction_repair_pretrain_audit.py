import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_model_native_state_v2 import (
    MODEL_NATIVE_HISTORY_MODE,
    MODEL_NATIVE_RANK_TRANSFORM,
    MODEL_NATIVE_STATE_SCHEMA_VERSION,
    MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
)
from gx1.scripts.audit_xau_direction_repair_pretrain_v1 import (
    DEFAULT_STEM,
    REQUIRED_POLARITY_FEATURES,
    REQUIRED_RAIL_FEATURES,
    run,
    build_parser,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields
from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    V12_DIRECTION_UTILITY_MAE_WEIGHT,
    V12_DIRECTION_UTILITY_MFE_WEIGHT,
    V12_DIRECTION_UTILITY_PATH_WEIGHT,
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
    forced_utility: bool = False,
    target_mode_id: int = 1,
    tape_root: str = "/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL",
    stem: str = DEFAULT_STEM,
) -> None:
    rank_ref = root / "model_native_rank_reference_xau_direction_repair.npz"
    if not rank_ref.exists():
        fit_start = pd.Timestamp("2020-11-09T00:00:00Z")
        fit_end = pd.Timestamp("2025-09-30T23:59:59Z")
        np.savez_compressed(
            rank_ref,
            schema_version=np.asarray([MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION]),
            fit_start_ns=np.asarray([fit_start.value], dtype=np.int64),
            fit_end_ns=np.asarray([fit_end.value], dtype=np.int64),
            fit_row_count=np.asarray([1], dtype=np.int64),
            entry_run_id=np.asarray(["MODEL_NATIVE_PRETRAIN_AUDIT_PYTEST"]),
            atr_bps_sorted=np.asarray([10.0], dtype=np.float64),
            spread_bps_sorted=np.asarray([1.0], dtype=np.float64),
        )
        rank_ref_sha = hashlib.sha256(rank_ref.read_bytes()).hexdigest()
        rank_ref.with_suffix(rank_ref.suffix + ".json").write_text(
            json.dumps(
                {
                    "schema_version": MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
                    "fit_scope": "train_only",
                    "rank_transform": MODEL_NATIVE_RANK_TRANSFORM,
                    "row_level_state_present": False,
                    "entry_run_id": "MODEL_NATIVE_PRETRAIN_AUDIT_PYTEST",
                    "out_npz": str(rank_ref.resolve()),
                    "out_npz_sha256": rank_ref_sha,
                    "fit_start_utc": str(fit_start),
                    "fit_end_utc": str(fit_end),
                    "fit_row_count": 1,
                }
            ),
            encoding="utf-8",
        )
    rank_ref_sha = hashlib.sha256(rank_ref.read_bytes()).hexdigest()
    rank_ref_sidecar_sha = hashlib.sha256(
        rank_ref.with_suffix(rank_ref.suffix + ".json").read_bytes()
    ).hexdigest()
    selected_fields = list(REQUIRED_POLARITY_FEATURES)
    if include_rail:
        selected_fields.extend(REQUIRED_RAIL_FEATURES)
    selected_fields = canonical_model_native_selected_fields(
        required_fields=selected_fields,
        remainder_prefix="session_regime.xau_repair_fixture",
    )
    signal_contract = model_native_signal_contract_metadata(selected_fields)
    fields = list(signal_contract["fields"])
    assert tuple(fields[: len(MODEL_NATIVE_BASE_FIELDS)]) == MODEL_NATIVE_BASE_FIELDS
    feature_positions = {name: fields.index(name) for name in REQUIRED_POLARITY_FEATURES}

    rows = []
    for i in range(80):
        support_dom = i < 40
        flat = i % 10 == 0
        selected_long_bad = 1.0 if support_dom and i % 17 == 1 else 0.0
        selected_short_bad = 1.0 if (not support_dom) and i % 17 == 2 else 0.0
        pnl_long = 30.0 + i if support_dom else -30.0 - i
        pnl_short = -30.0 - i if support_dom else 30.0 + i
        if flat:
            pnl_long = -500.0
            pnl_short = -500.0
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
        snap[feature_positions["chart.geometry_support_line_proximity_stack"]] = support
        snap[feature_positions["chart.geometry_resistance_line_proximity_stack"]] = resistance
        snap[feature_positions["chart.geometry_support_minus_resistance_stack"]] = sr
        snap[feature_positions["chart.geometry_channel_position_low_to_high"]] = channel_position
        long_mfe = 20.0 + i if support_dom else 6.0 + i
        long_mae = 2.0 + i if support_dom else 16.0 + i
        short_mfe = 4.0 + i if support_dom else 12.0 + i
        short_mae = 14.0 + i if support_dom else 3.0 + i
        long_utility = np.float32(
            pnl_long
            + V12_DIRECTION_UTILITY_MFE_WEIGHT * long_mfe
            - V12_DIRECTION_UTILITY_MAE_WEIGHT * long_mae
            + V12_DIRECTION_UTILITY_PATH_WEIGHT * (long_mfe - long_mae)
        )
        short_utility = np.float32(
            pnl_short
            + V12_DIRECTION_UTILITY_MFE_WEIGHT * short_mfe
            - V12_DIRECTION_UTILITY_MAE_WEIGHT * short_mae
            + V12_DIRECTION_UTILITY_PATH_WEIGHT * (short_mfe - short_mae)
        )
        y_direction = 2 if flat else 0 if support_dom else 1
        y_side = 0 if support_dom else 1
        if anti_short_wrong_side and support_dom and i == 1:
            y_direction = 1
            y_side = 1
            scalar_bad = 1.0
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
                "bad_path_long_first_n": selected_long_bad if support_dom else 1.0,
                "bad_path_short_first_n": 1.0 if support_dom else selected_short_bad,
                "y_long_final_pnl_at_direction_horizon_bps": pnl_long,
                "y_short_final_pnl_at_direction_horizon_bps": pnl_short,
                "y_direction_target_mode_id": target_mode_id,
                "y_direction_long_score_bps": long_utility,
                "y_direction_short_score_bps": short_utility,
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
        if alias_mismatch and i == 1:
            row["y_direction_long_score_bps"] = long_utility + 1.0
        if forced_utility and i == 1:
            row["y_long_path_utility_bps"] = long_utility - 25.0
            row["y_direction_long_score_bps"] = long_utility - 25.0
        rows.append(row)

    pd.DataFrame(rows).to_parquet(root / f"{stem}_{split}.parquet")
    manifest = {
        "neutral_xgb_bridge": False,
        "xgb_bridge_source": None,
        "tape_root": tape_root,
        "splits": {
            "train": {
                "start": "2020-11-09T00:00:00Z",
                "end": "2025-09-30T23:59:59Z",
            },
            "val": {
                "start": "2025-10-01T00:00:00Z",
                "end": "2025-12-31T23:59:59Z",
            },
            "test": {
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-06-26T03:25:00Z",
            },
        },
        "extra": {
            "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
            "model_native_signal_contract": signal_contract,
            "signal_bridge": {
                "neutral_xgb_bridge": False,
                "bridge_source": None,
                "bridge_dim": 0,
                "fields": fields,
                "snap_fields": fields,
                "seq_structure_extension_v1": {
                    "enabled": True,
                    "mode": (
                        "mandatory_inline_common_causal_history_v1"
                        if include_inline_seq_structure
                        else "parquet_join"
                    ),
                },
            },
            "model_native_state_contract": {
                "schema_version": MODEL_NATIVE_STATE_SCHEMA_VERSION,
                "feature_history_start_utc": "2020-11-01T00:00:00Z",
                "rank_fit_start_utc": "2020-11-09T00:00:00Z",
                "rank_fit_end_utc": "2025-09-30T23:59:59Z",
                "rank_reference_npz": str(rank_ref.resolve()),
                "rank_reference_npz_sha256": rank_ref_sha,
                "rank_reference_sidecar_sha256": rank_ref_sidecar_sha,
                "rank_reference_schema_version": MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
                "normalization_fit_scope": "train_only",
                "rank_transform": MODEL_NATIVE_RANK_TRANSFORM,
                "feature_history_mode": MODEL_NATIVE_HISTORY_MODE,
                "split_reset_allowed": False,
                "post_fit_rows_in_rank_reference": False,
                "runtime_rule_free": True,
                "entry_run_id": "MODEL_NATIVE_PRETRAIN_AUDIT_PYTEST",
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
            "--stem",
            DEFAULT_STEM,
            "--data-splits",
            "train,val,test",
            "--max-rows-per-split",
            "80",
            "--max-row-groups-per-split",
            "2",
            *extra,
        ]
    )


def _read_immutable_audit(tmp_path: Path) -> dict:
    files = sorted((tmp_path / "audit").glob("XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_*.json"))
    assert len(files) == 1
    assert "_latest" not in files[0].name
    return json.loads(files[0].read_text(encoding="utf-8"))


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

    report = _read_immutable_audit(tmp_path)
    assert report["decision"] == "FAIL"
    assert any("channel_position polarity stale/inverted" in item for item in report["failures"])


def test_xau_direction_repair_signal_contract_requires_every_rail_feature() -> None:
    selected = canonical_model_native_selected_fields(
        required_fields=REQUIRED_POLARITY_FEATURES,
        remainder_prefix="session_regime.missing_rail_adversary",
    )
    selected.remove(REQUIRED_RAIL_FEATURES[0])
    selected.append("chart.geometry_replacement_without_registered_rail")

    with pytest.raises(RuntimeError, match="missing_mandatory_full_stack_fields"):
        model_native_signal_contract_metadata(selected)


def test_xau_direction_repair_pretrain_audit_requires_inline_seq_structure(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=False, include_inline_seq_structure=False)

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    report = _read_immutable_audit(tmp_path)
    assert report["decision"] == "FAIL"
    assert any("requires inline seq-structure" in item for item in report["failures"])


def test_xau_direction_repair_pretrain_audit_fails_bad_path_side_mismatch(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=False, bad_path_mismatch=(split == "train"))

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    report = _read_immutable_audit(tmp_path)
    assert report["decision"] == "FAIL"
    assert any("scalar y_bad_path mismatches" in item for item in report["failures"])


def test_xau_direction_repair_pretrain_audit_fails_direction_outcome_mismatch(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=False, anti_short_wrong_side=(split == "train"))

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    report = _read_immutable_audit(tmp_path)
    assert report["decision"] == "FAIL"
    assert any("y_direction mismatches future outcome side selection" in item for item in report["failures"])


def test_xau_direction_repair_pretrain_audit_fails_direction_score_alias_mismatch(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=False, alias_mismatch=(split == "train"))

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    report = _read_immutable_audit(tmp_path)
    assert report["decision"] == "FAIL"
    assert any("y_direction_long_score_bps mismatches" in item for item in report["failures"])


def test_xau_direction_repair_pretrain_audit_rejects_forced_utility(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=False, forced_utility=(split == "train"))

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    report = _read_immutable_audit(tmp_path)
    assert report["decision"] == "FAIL"
    assert any("long utility is not the declared future-outcome formula" in item for item in report["failures"])


def test_xau_direction_repair_pretrain_audit_rejects_legacy_target_mode(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        _write_split(
            tmp_path,
            split,
            inverted=False,
            target_mode_id=0 if split == "train" else 1,
        )

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    report = _read_immutable_audit(tmp_path)
    assert report["decision"] == "FAIL"
    assert any("direction target mode contract is invalid" in item for item in report["failures"])


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

    report = _read_immutable_audit(tmp_path)
    assert report["decision"] == "FAIL"
    assert any("requires XAUUSD tape_root provenance" in item for item in report["failures"])


def test_xau_direction_repair_pretrain_audit_rejects_auto_discovered_stem(tmp_path: Path) -> None:
    stem = "v10_6yr_dataset__HOLD_03B"
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=False, stem=stem)

    with pytest.raises(SystemExit):
        run(_args(tmp_path, "--stem", "auto"))

    report = _read_immutable_audit(tmp_path)
    assert report["decision"] == "FAIL"
    assert any("explicit immutable --stem is required" in item for item in report["failures"])
