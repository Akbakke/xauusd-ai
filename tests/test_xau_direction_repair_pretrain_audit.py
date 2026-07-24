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
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_model_native_state_v2 import (
    MODEL_NATIVE_HISTORY_MODE,
    MODEL_NATIVE_RANK_TRANSFORM,
    MODEL_NATIVE_STATE_SCHEMA_VERSION,
    MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
)
from gx1.contracts.xau_tape_provenance_v1 import (
    BASE_REPAIR_METHOD,
    BASE_REPAIR_SCHEMA,
    CURRENT_SNAPSHOT_METHOD,
    CURRENT_SNAPSHOT_SCHEMA,
    XAU_INSTRUMENT,
    canonical_xau_source_descriptor_v1,
    validate_xau_tape_provenance_v1,
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
from tests.test_oanda_backfill_vedtak_gate import (
    materialize_native_m5_test_bundle,
)


RUN_ID = "MODEL_NATIVE_PRETRAIN_AUDIT_PYTEST"


def _write_xau_tape(root: Path, *, instrument: str) -> Path:
    tape = root / "xauusd_tape_fixture"
    if tape.exists():
        return tape.resolve()
    canonical_sources = {}
    for key, timeframe in (("m5", "M5"), ("m1", "M1")):
        canonical_root = root / f"canonical_{key}"
        if timeframe == "M5":
            materialize_native_m5_test_bundle(canonical_root)
        else:
            canonical_root.mkdir()
        manifest = {
            "instrument": "XAUUSD",
            "timeframe": timeframe,
            "out_root": str(canonical_root.resolve()),
        }
        if timeframe != "M5":
            (canonical_root / "MANIFEST.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
        canonical_sources[key] = canonical_xau_source_descriptor_v1(
            canonical_root.resolve(), timeframe=timeframe
        )
    base = root / "base_tape_fixture"
    base_part = base / "year=2026" / "part-000.parquet"
    base_part.parent.mkdir(parents=True)
    base_part.write_bytes(b"base-xau-tape")
    base_year_hashes = {"year=2026": hashlib.sha256(base_part.read_bytes()).hexdigest()}
    base_manifest_path = base / "REPAIR_MANIFEST.json"
    base_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": BASE_REPAIR_SCHEMA,
                "instrument": XAU_INSTRUMENT,
                "explicit_vedtak_id": RUN_ID,
                "method": BASE_REPAIR_METHOD,
                "geometry_bad_total_after": 0,
                "m5_tape_root": canonical_sources["m5"]["root"],
                "m1_tape_root": canonical_sources["m1"]["root"],
                "canonical_sources": canonical_sources,
                "years": {
                    "year=2026": {"output_sha256": base_year_hashes["year=2026"]}
                },
            }
        ),
        encoding="utf-8",
    )
    current_part = tape / "year=2026" / "part-000.parquet"
    current_part.parent.mkdir(parents=True)
    current_part.write_bytes(b"current-xau-tape")
    snapshot = tape / "collector_snapshot" / "xauusd_m1_fixture.parquet"
    snapshot.parent.mkdir()
    snapshot.write_bytes(b"immutable-xau-m1-snapshot")
    (tape / "REPAIR_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": CURRENT_SNAPSHOT_SCHEMA,
                "instrument": instrument,
                "entry_run_id": RUN_ID,
                "method": CURRENT_SNAPSHOT_METHOD,
                "last_complete_m5_utc": "2026-07-21T00:00:00+00:00",
                "base_tape_root": str(base.resolve()),
                "base_manifest_path": str(base_manifest_path.resolve()),
                "base_manifest_sha256": hashlib.sha256(
                    base_manifest_path.read_bytes()
                ).hexdigest(),
                "base_year_sha256": base_year_hashes,
                "collector_sources": [
                    {
                        "source_path": "/collector/xauusd_m1_fixture.parquet",
                        "snapshot_path": str(snapshot.resolve()),
                        "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                        "rows": 1,
                    }
                ],
                "overlap_exact": True,
                "overlap_proof": {
                    "rows": 1,
                    "max_abs_diff": 0.0,
                    "new_tail_rows": 1,
                },
                "geometry_bad_total_after": 0,
                "years": {
                    "year=2026": {
                        "output_sha256": hashlib.sha256(
                            current_part.read_bytes()
                        ).hexdigest()
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return tape.resolve()


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
    tape_root: str | None = None,
    tape_instrument: str = XAU_INSTRUMENT,
    stem: str = DEFAULT_STEM,
) -> None:
    if tape_root is None:
        tape_root = str(_write_xau_tape(root, instrument=tape_instrument))
    tape_provenance = None
    if tape_instrument == XAU_INSTRUMENT:
        tape_provenance = validate_xau_tape_provenance_v1(
            tape_root,
            expected_run_id=RUN_ID,
            require_current=True,
        )
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
            entry_run_id=np.asarray([RUN_ID]),
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
                    "entry_run_id": RUN_ID,
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
            "xau_tape_provenance": tape_provenance,
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
                "rank_reference_fit_scope": "train_only",
                "rank_transform": MODEL_NATIVE_RANK_TRANSFORM,
                "feature_history_mode": MODEL_NATIVE_HISTORY_MODE,
                "split_reset_allowed": False,
                "post_fit_rows_in_rank_reference": False,
                "runtime_rule_free": True,
                "entry_run_id": RUN_ID,
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


def test_missing_polarity_field_does_not_mask_target_consistency(
    tmp_path: Path,
) -> None:
    missing = "chart.geometry_support_minus_resistance_stack"
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=False)
        manifest_path = tmp_path / f"{DEFAULT_STEM}_{split}.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("fields", "snap_fields"):
            manifest["extra"]["signal_bridge"][key].remove(missing)
        manifest["feature_contract"]["signal_bridge_fields"].remove(missing)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    report = _read_immutable_audit(tmp_path)
    assert report["decision"] == "FAIL"
    assert all(
        row["target_consistency"]["available"] is True
        for row in report["splits"]
    )
    assert any(
        f"missing channel-polarity feature: {missing}" in failure
        for failure in report["failures"]
    )
    assert not any(
        "target consistency audit unavailable" in failure
        for failure in report["failures"]
    )


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
            tape_instrument="INVALID_INSTRUMENT",
        )

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    report = _read_immutable_audit(tmp_path)
    assert report["decision"] == "FAIL"
    assert any("XAU_TAPE_CURRENT_INSTRUMENT_MISMATCH" in item for item in report["failures"])


def test_xau_direction_repair_pretrain_audit_rejects_tape_byte_drift(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=False)
    tape_part = tmp_path / "xauusd_tape_fixture" / "year=2026" / "part-000.parquet"
    tape_part.write_bytes(tape_part.read_bytes() + b"-mutated")

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    report = _read_immutable_audit(tmp_path)
    assert report["decision"] == "FAIL"
    assert any("HASH_MISMATCH" in item for item in report["failures"])


def test_xau_direction_repair_pretrain_audit_rejects_dataset_tape_rebinding(
    tmp_path: Path,
) -> None:
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=False)
    manifest_path = tmp_path / f"{DEFAULT_STEM}_train.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["extra"]["xau_tape_provenance"]["manifest_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    report = _read_immutable_audit(tmp_path)
    assert report["decision"] == "FAIL"
    assert any("tape binding differs" in item for item in report["failures"])


def test_xau_direction_repair_pretrain_audit_rejects_auto_discovered_stem(tmp_path: Path) -> None:
    stem = "v10_6yr_dataset__HOLD_03B"
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=False, stem=stem)

    with pytest.raises(SystemExit):
        run(_args(tmp_path, "--stem", "auto"))

    report = _read_immutable_audit(tmp_path)
    assert report["decision"] == "FAIL"
    assert any("explicit immutable --stem is required" in item for item in report["failures"])
