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
    MODEL_NATIVE_STATE_SCHEMA_VERSION,
)
from gx1.contracts.entry_exit_feature_surface_v1 import (
    ENTRY_M5_FEATURE_SURFACE_CONSUMPTION_MODE,
)
from gx1.contracts.xau_tape_provenance_v1 import (
    BASE_REPAIR_METHOD,
    BASE_REPAIR_SCHEMA,
    CURRENT_SNAPSHOT_METHOD,
    CURRENT_SNAPSHOT_SCHEMA,
    XAU_INSTRUMENT,
    canonical_xau_source_descriptor_v1,
    canonical_json_sha256,
    validate_xau_tape_provenance_v1,
)
from gx1.scripts.audit_xau_direction_repair_pretrain_v1 import (
    DEFAULT_STEM,
    REQUIRED_POLARITY_FEATURES,
    REQUIRED_MANDATORY_LEVEL_FEATURES,
    run,
    build_parser,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields
from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    ENTRY_FITTED_Q_DATASET_STEM_SUFFIX,
    diagnostic_outcome_label_contract,
    entry_fitted_q_dataset_contract,
    representation_auxiliary_outcome_contract,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_AUX_TARGET_COLUMNS,
)
from gx1.contracts.entry_position_size_target_policy_v1 import (
    entry_position_size_target_policy_contract,
    entry_position_size_targets_from_policy,
)
from tests.entry_direction_target_policy_support import (
    entry_direction_target_policy_fixture,
)
from tests.entry_position_size_target_policy_support import (
    entry_position_size_target_policy_fixture,
)
from tests.test_oanda_backfill_vedtak_gate import (
    materialize_native_xau_test_bundle,
)


RUN_ID = "MODEL_NATIVE_PRETRAIN_AUDIT_PYTEST"


def _write_xau_tape(root: Path, *, instrument: str) -> Path:
    tape = root / "xauusd_tape_fixture"
    if tape.exists():
        return tape.resolve()
    canonical_sources = {}
    for key, timeframe in (("m5", "M5"), ("m1", "M1")):
        canonical_root = root / f"canonical_{key}"
        materialize_native_xau_test_bundle(
            canonical_root,
            timeframe=timeframe,
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
    include_mandatory_levels: bool = True,
    include_inline_seq_structure: bool = True,
    bad_path_mismatch: bool = False,
    constant_selected_bad_path: bool = False,
    nonfinite_scalar_bad_path: bool = False,
    dead_side_bad_paths: bool = False,
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
    policy_tape_provenance = (
        tape_provenance
        if tape_provenance is not None
        else {"fixture_instrument": tape_instrument}
    )
    source_parquet_sha256 = "a" * 64
    direction_target_policy = entry_direction_target_policy_fixture(
        source_parquet_sha256=source_parquet_sha256,
        tape_provenance_sha256=canonical_json_sha256(policy_tape_provenance),
        train_start_utc="2020-11-09T00:00:00+00:00",
        train_end_utc="2025-09-30T23:59:59+00:00",
    )
    position_size_target_policy = entry_position_size_target_policy_fixture(
        source_parquet_sha256=source_parquet_sha256,
        tape_provenance_sha256=canonical_json_sha256(policy_tape_provenance),
        train_start_utc="2020-11-09T00:00:00+00:00",
        train_end_utc="2025-09-30T23:59:59+00:00",
    )
    selected_fields = list(REQUIRED_POLARITY_FEATURES)
    if include_mandatory_levels:
        selected_fields.extend(REQUIRED_MANDATORY_LEVEL_FEATURES)
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
        selected_long_bad = (
            0.0
            if constant_selected_bad_path
            else 1.0 if support_dom and i % 17 == 1 else 0.0
        )
        selected_short_bad = (
            0.0
            if constant_selected_bad_path
            else 1.0 if (not support_dom) and i % 17 == 2 else 0.0
        )
        pnl_long = 30.0 + i if support_dom else -30.0 - i
        pnl_short = -30.0 - i if support_dom else 30.0 + i
        if flat:
            pnl_long = -500.0
            pnl_short = -500.0
        long_mae = 2.0 + i if support_dom else 3.0 + i
        short_mae = 3.0 + i if support_dom else 2.0 + i
        scalar_bad = 0.0 if flat else selected_long_bad if support_dom else selected_short_bad
        if nonfinite_scalar_bad_path and i == 1:
            scalar_bad = np.nan
        if bad_path_mismatch and i == 1:
            scalar_bad = 1.0 - scalar_bad
        support = 0.40 if support_dom else 0.90
        resistance = 0.90 if support_dom else 0.40
        if inverted:
            support, resistance = resistance, support
        snap = [0.0] * len(fields)
        snap[feature_positions["level_below_dist_atr"]] = support
        snap[feature_positions["level_above_dist_atr"]] = resistance
        long_mfe = 20.0 + i if support_dom else 6.0 + i
        long_mae = 2.0 + i if support_dom else 16.0 + i
        short_mfe = 4.0 + i if support_dom else 12.0 + i
        short_mae = 14.0 + i if support_dom else 3.0 + i
        long_utility = np.float32(pnl_long)
        short_utility = np.float32(pnl_short)
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
                "y_position_size_target": 0.0,
                "y_position_size_mask": 0.0 if flat else 1.0,
                "mfe_long_first_n_bps": long_mfe,
                "mae_long_first_n_bps": long_mae,
                "mfe_short_first_n_bps": short_mfe,
                "mae_short_first_n_bps": short_mae,
                "bad_path_long_first_n": (
                    0.0 if dead_side_bad_paths
                    else selected_long_bad if support_dom else 1.0
                ),
                "bad_path_short_first_n": (
                    0.0 if dead_side_bad_paths
                    else 1.0 if support_dom else selected_short_bad
                ),
                "y_long_final_pnl_at_direction_horizon_bps": pnl_long,
                "y_short_final_pnl_at_direction_horizon_bps": pnl_short,
                "y_direction_target_mode_id": target_mode_id,
                "y_direction_long_score_bps": long_utility,
                "y_direction_short_score_bps": short_utility,
                "y_long_path_utility_bps": long_utility,
                "y_short_path_utility_bps": short_utility,
                "y_long_bad_path": (
                    0.0 if dead_side_bad_paths
                    else selected_long_bad if support_dom else 1.0
                ),
                "y_short_bad_path": (
                    0.0 if dead_side_bad_paths
                    else 1.0 if support_dom else selected_short_bad
                ),
                "y_long_expected_mae_bps": long_mae,
                "y_short_expected_mae_bps": short_mae,
                # Touch masks fire on a strict subset of rows so the audit's
                # target-liveness gate sees genuine variation; held outcomes
                # vary within the masked rows.
                "y_line_support_touch_held": (
                    1.0 if (support_dom and i % 2 == 0) else 0.0
                ),
                "y_line_support_touch_mask": 1.0 if support_dom else 0.0,
                "y_line_resistance_touch_held": (
                    1.0 if ((not support_dom) and i % 2 == 0) else 0.0
                ),
                "y_line_resistance_touch_mask": 0.0 if support_dom else 1.0,
                "y_support_retest_continuation": 1.0 if support_dom else 0.0,
                "y_resistance_retest_continuation": 0.0 if support_dom else 1.0,
                "y_countertrend_short_trap": 1.0 if support_dom else 0.0,
                "y_countertrend_long_trap": 0.0 if support_dom else 1.0,
                "y_long_high_mae_low_mfe_early_failure": 0.0 if support_dom else 1.0,
                "y_short_high_mae_low_mfe_early_failure": 1.0 if support_dom else 0.0,
        }
        # Every declared aux-v3 target column must exist and be live. The
        # names come from the owner tuple, so a newly declared aux target is
        # covered without restating a column list here.
        for offset, aux_name in enumerate(MODEL_NATIVE_AUX_TARGET_COLUMNS):
            if aux_name in row:
                continue
            row[aux_name] = float(i + 1) * (1.0 + 0.01 * float(offset))
        if alias_mismatch and i == 1:
            row["y_direction_long_score_bps"] = long_utility + 1.0
        if forced_utility and i == 1:
            row["y_long_path_utility_bps"] = long_utility - 25.0
            row["y_direction_long_score_bps"] = long_utility - 25.0
        rows.append(row)

    split_frame = pd.DataFrame(rows)
    sizing = entry_position_size_targets_from_policy(
        policy=position_size_target_policy,
        mfe_first_n_bps=split_frame["mfe_first_n_bps"].to_numpy(float),
        mae_first_n_bps=split_frame["mae_first_n_bps"].to_numpy(float),
        selected_side=np.where(
            split_frame["y_trade"].to_numpy(float) > 0.5,
            split_frame["y_side"].to_numpy(int),
            -1,
        ),
        trade_mask=split_frame["y_trade"].to_numpy(float),
    )
    split_frame["y_position_size_target"] = sizing["target"]
    split_frame["y_position_size_mask"] = sizing["mask"]
    split_frame.to_parquet(root / f"{stem}_{split}.parquet")
    manifest = {
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
            "xau_tape_provenance": policy_tape_provenance,
            **entry_fitted_q_dataset_contract(),
            **diagnostic_outcome_label_contract(direction_target_policy),
            **representation_auxiliary_outcome_contract(),
            **entry_position_size_target_policy_contract(
                position_size_target_policy
            ),
            "early_move_threshold_bps": float(
                direction_target_policy["early_move_threshold_bps"]
            ),
            "source_frame": {
                "mode": "exact_source_parquet",
                "parquet_path": "/fixture/model_native_m5.parquet",
                "parquet_sha256": source_parquet_sha256,
            },
            "strict_entry_labels": {
                **diagnostic_outcome_label_contract(direction_target_policy),
                "direction_follows_tradable_side": True,
                "core_direction_target_provenance": {
                    "source": "future_executable_pnl_outcomes_only",
                    "feature_derived_rewrite_count": 0,
                    "forced_utility_order_count": 0,
                },
            },
            "signal_bridge": {
                "bridge_source": None,
                "bridge_dim": 0,
                "fields": fields,
                "snap_fields": fields,
                "seq_structure_extension_v1": {
                    "enabled": True,
                    "mode": (
                        ENTRY_M5_FEATURE_SURFACE_CONSUMPTION_MODE
                        if include_inline_seq_structure
                        else "parquet_join"
                    ),
                },
            },
            # The retired TRAIN rank-reference fields are forbidden by the
            # state owner; only the exact v7 state contract survives.
            "model_native_state_contract": {
                "schema_version": MODEL_NATIVE_STATE_SCHEMA_VERSION,
                "feature_history_start_utc": "2020-11-01T00:00:00Z",
                "feature_history_mode": MODEL_NATIVE_HISTORY_MODE,
                "split_reset_allowed": False,
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


def test_pretrain_audit_accepts_structurally_constant_selected_bad_path(
    tmp_path: Path,
) -> None:
    for split in ("train", "val", "test"):
        _write_split(
            tmp_path,
            split,
            inverted=False,
            constant_selected_bad_path=True,
        )

    report = run(_args(tmp_path))

    assert report["decision"] == "PASS"
    assert report["failures"] == []
    assert report["target_liveness_policy"]["exempt_columns"]["y_bad_path"]
    for split_report in report["splits"]:
        assert split_report["target_liveness"]["y_bad_path"]["live"] is False
        assert split_report["target_liveness"]["y_bad_path"]["finite_rate"] == 1.0
        assert split_report["target_liveness"]["y_long_bad_path"]["live"] is True
        assert split_report["target_liveness"]["y_short_bad_path"]["live"] is True
        assert (
            split_report["target_consistency"]["bad_path_side_mismatch_count"]
            == 0
        )


def test_pretrain_audit_rejects_nonfinite_selected_bad_path(
    tmp_path: Path,
) -> None:
    for split in ("train", "val", "test"):
        _write_split(
            tmp_path,
            split,
            inverted=False,
            constant_selected_bad_path=True,
            nonfinite_scalar_bad_path=(split == "train"),
        )

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    report = _read_immutable_audit(tmp_path)
    assert report["decision"] == "FAIL"
    assert any(
        "liveness-exempt XAU target column is not fully finite: y_bad_path"
        in failure
        for failure in report["failures"]
    )
    assert any(
        "scalar y_bad_path mismatches" in failure
        for failure in report["failures"]
    )


def test_pretrain_audit_rejects_dead_side_specific_bad_path_sources(
    tmp_path: Path,
) -> None:
    for split in ("train", "val", "test"):
        _write_split(
            tmp_path,
            split,
            inverted=False,
            constant_selected_bad_path=True,
            dead_side_bad_paths=(split == "train"),
        )

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    report = _read_immutable_audit(tmp_path)
    assert report["decision"] == "FAIL"
    assert any(
        "XAU future-outcome target column is not live: y_long_bad_path"
        in failure
        for failure in report["failures"]
    )
    assert any(
        "XAU future-outcome target column is not live: y_short_bad_path"
        in failure
        for failure in report["failures"]
    )


def test_missing_polarity_field_does_not_mask_target_consistency(
    tmp_path: Path,
) -> None:
    missing = "level_above_dist_atr"
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
        f"missing level-distance polarity feature: {missing}" in failure
        for failure in report["failures"]
    )
    assert not any(
        "target consistency audit unavailable" in failure
        for failure in report["failures"]
    )


def test_xau_direction_repair_signal_contract_requires_every_mandatory_level_feature() -> None:
    selected = canonical_model_native_selected_fields(
        required_fields=REQUIRED_POLARITY_FEATURES,
        remainder_prefix="session_regime.missing_geometry_adversary",
    )
    selected.remove(REQUIRED_MANDATORY_LEVEL_FEATURES[0])
    selected.append("level_replacement_without_registered_mandatory")

    with pytest.raises(RuntimeError, match="missing_mandatory_full_stack_fields"):
        model_native_signal_contract_metadata(selected)


def test_xau_direction_repair_pretrain_audit_requires_native_m5_surface(
    tmp_path: Path,
) -> None:
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=False, include_inline_seq_structure=False)

    with pytest.raises(SystemExit):
        run(_args(tmp_path))

    report = _read_immutable_audit(tmp_path)
    assert report["decision"] == "FAIL"
    assert any(
        "requires the exact native M5 feature surface" in item
        for item in report["failures"]
    )


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
    assert any(
        "long score is not executable PnL under the fitted policy" in item
        for item in report["failures"]
    )


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
    # The rebinding is caught through the sizing policy, not the direction
    # policy: `build_entry_v10_ctx_training_dataset_v3` raises
    # MODEL_NATIVE_MANIFEST_RETIRED_DIRECTION_POLICY_PRESENT if a split
    # manifest carries the direction-policy payload at all, so
    # `require_entry_direction_target_manifest_binding` (the only source of
    # ENTRY_TARGET_POLICY_TAPE_MISMATCH) can never run here. The sizing policy
    # pins both the tape hash and the direction policy's own sha256, so the
    # same rebinding still fails closed.
    assert any(
        "ENTRY_POSITION_SIZE_TARGET_POLICY_TAPE_PROVENANCE_SHA256_MISMATCH" in item
        for item in report["failures"]
    )


def test_xau_direction_repair_pretrain_audit_rejects_auto_discovered_stem(tmp_path: Path) -> None:
    # A discoverable-looking stem built from the owner's dataset suffix; the
    # point is that `--stem auto` is rejected whatever is on disk.
    stem = f"v10_6yr_dataset_discovery_adversary{ENTRY_FITTED_Q_DATASET_STEM_SUFFIX}"
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split, inverted=False, stem=stem)

    with pytest.raises(SystemExit):
        run(_args(tmp_path, "--stem", "auto"))

    report = _read_immutable_audit(tmp_path)
    assert report["decision"] == "FAIL"
    assert any("explicit immutable --stem is required" in item for item in report["failures"])
