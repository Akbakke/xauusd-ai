from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION,
    encode_model_native_runtime_head_evidence,
)
from gx1.execution.model_native_entry_replay_v1 import SourceTape
from gx1.exits.contracts.exit_io_v8_regime_m1l512 import (
    EXIT_IO_V8_CONTEXT_CADENCE_CONTRACT,
    EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT,
    EXIT_IO_V8_REGIME_M1L512_FEATURE_NAMES_HASH,
    EXIT_IO_V8_REGIME_M1L512_FEATURES,
    EXIT_IO_V8_REGIME_M1L512_IO_VERSION,
)
from gx1.exits.training.thin_record_dataset import (
    V3_THIN_RECORD_SCHEMA_VERSION,
    V3_TRAINING_DATASET_PRODUCER_CONTRACT,
    V3_TRAINING_LINEAGE_SCHEMA_VERSION,
    V3_TRAINING_PRODUCER_INPUT_NAMES,
    V3_TRAINING_RECORD_EMIT_STRIDE_BARS,
    V3_TRAINING_SOURCE_CODE_FILES,
    V3_TRAINING_TEACHER_HORIZON_BARS,
    build_v3_xgb_bridge_source_identity,
    materialize_model_native_v3_trade_records,
    require_authoritative_v3_training_dataset,
    require_exact_v3_thin_record,
    require_reproducible_v3_training_lineage,
    v3_regular_file_binding,
)
from gx1.features.trade_overlay import (
    N_OVERLAY_COLS,
    OVERLAY_COL_NAMES,
    compute_trade_overlay,
)
from gx1.xgb.multihead.xgb_multihead_model_v1 import (
    proba_to_signal_bridge_v1,
)
from tests.test_entry_model_native_runtime_evidence_contract import _valid_evidence


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_dataset(root: Path, **semantic_updates: object) -> Path:
    root.mkdir(parents=True)
    xgb_bundle = root / "xgb_bundle"
    xgb_bundle.mkdir()
    (xgb_bundle / "model.joblib").write_bytes(b"unit-xgb")
    xgb_feature_contract = xgb_bundle / "xgb_input_features.json"
    xgb_feature_contract.write_text(
        json.dumps({"features": ["feature_a", "feature_b"]}),
        encoding="utf-8",
    )
    xgb_sanitizer = xgb_bundle / "xgb_input_sanitizer.json"
    xgb_sanitizer.write_text(
        json.dumps({"feature_list": ["feature_a", "feature_b"]}),
        encoding="utf-8",
    )
    xgb_identity = build_v3_xgb_bridge_source_identity(
        bundle_dir=xgb_bundle.resolve(),
        feature_contract_path=xgb_feature_contract.resolve(),
        sanitizer_config_path=xgb_sanitizer.resolve(),
    )
    files = {
        "m1_feature_matrix": "m1_feature_matrix.npy",
        "m1_time_ns": "m1_time_ns.npy",
        "trade_state_overlays": "trade_state_overlays.f32",
        "trade_state_overlays_cols": N_OVERLAY_COLS,
        "overlay_index": "overlay_index.parquet",
        "records": "records.jsonl",
    }
    matrix = np.zeros(
        (
            511 + V3_TRAINING_TEACHER_HORIZON_BARS,
            EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT,
        ),
        dtype=np.float32,
    )
    probabilities = np.tile(
        np.asarray([[0.7, 0.2, 0.1]], dtype=np.float32),
        (len(matrix), 1),
    )
    bridge = np.asarray(
        proba_to_signal_bridge_v1(probabilities),
        dtype=np.float32,
    )
    for column, name in enumerate(
        (
            "p_long",
            "p_short",
            "p_flat",
            "p_hat",
            "uncertainty_score",
            "margin_top1_top2",
            "entropy",
        )
    ):
        matrix[:, EXIT_IO_V8_REGIME_M1L512_FEATURES.index(name)] = bridge[
            :,
            column,
        ]
    np.save(root / files["m1_feature_matrix"], matrix, allow_pickle=False)
    times = pd.date_range(
        "2026-07-01T00:00:00Z",
        periods=len(matrix),
        freq="min",
    )
    np.save(
        root / files["m1_time_ns"],
        times.asi8.astype(np.int64),
        allow_pickle=False,
    )
    path_pnl = np.linspace(
        -1.0,
        5.0,
        V3_TRAINING_TEACHER_HORIZON_BARS,
    ).astype(np.float32).astype(np.float64)
    overlay = compute_trade_overlay(
        np.asarray(path_pnl + 1.0, dtype=np.float32).astype(np.float64),
        np.asarray(path_pnl - 2.0, dtype=np.float32).astype(np.float64),
        path_pnl,
        np.full(
            V3_TRAINING_TEACHER_HORIZON_BARS,
            3.0,
            dtype=np.float32,
        ).astype(np.float64),
        {
            "p_long_entry": 0.7,
            "p_hat_entry": 0.7,
            "uncertainty_entry": 0.3,
            "entropy_entry": 0.8,
            "margin_entry": 0.5,
        },
    )
    overlay.tofile(root / files["trade_state_overlays"])
    trade_uid = "UNIT_MODEL_NATIVE_V3:unit-trade:long"
    pd.DataFrame(
        {
            "trade_uid": [trade_uid],
            "overlay_offset": [0],
            "overlay_length": [V3_TRAINING_TEACHER_HORIZON_BARS],
        }
    ).to_parquet(root / files["overlay_index"], index=False)
    scalar_names = (
        "pnl_bps_now",
        "mfe_bps",
        "mae_bps",
        "dd_from_mfe_bps",
        "distance_from_peak_mfe_bps",
        "giveback_ratio",
        "bars_held",
        "time_since_mfe_bars",
        "atr_bps_now",
        "rolling_slope_since_entry",
    )
    entry_index = 511
    entry_fill_time = times[entry_index]
    decision_time = entry_fill_time - pd.Timedelta(minutes=5)
    records = []
    for offset in range(V3_TRAINING_TEACHER_HORIZON_BARS):
        records.append(
            {
                "schema_version": V3_THIN_RECORD_SCHEMA_VERSION,
                "ts": times[entry_index + offset].isoformat(),
                "decision_ts": decision_time.isoformat(),
                "entry_fill_time": entry_fill_time.isoformat(),
                "runtime_head_evidence_sha256": "a" * 64,
                "run_id": "UNIT_MODEL_NATIVE_V3",
                "trade_uid": trade_uid,
                "trade_id": "unit-trade",
                "side": "long",
                "m1_idx_now": entry_index + offset,
                "in_trade_start_in_win": 511 - offset,
                "n_in_trade_bars": offset + 1,
                "overlay_start_row": 0,
                "scalars": {
                    name: float(
                        overlay[offset, OVERLAY_COL_NAMES.index(name)]
                    )
                    for name in scalar_names
                },
                "teacher_final_pnl_bps": float(
                    overlay[-1, OVERLAY_COL_NAMES.index("pnl_bps_now")]
                ),
                "teacher_final_mfe_bps": float(
                    overlay[-1, OVERLAY_COL_NAMES.index("mfe_bps")]
                ),
                "teacher_final_mae_bps": float(
                    overlay[-1, OVERLAY_COL_NAMES.index("mae_bps")]
                ),
                "teacher_duration_bars": V3_TRAINING_TEACHER_HORIZON_BARS,
            }
        )
    (root / files["records"]).write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    producer_inputs = {}
    producer_root = root / "producer_inputs"
    producer_root.mkdir()
    for name in sorted(V3_TRAINING_PRODUCER_INPUT_NAMES):
        path = producer_root / f"{name}.artifact"
        path.write_bytes(f"unit-producer-input:{name}".encode("utf-8"))
        producer_inputs[name] = v3_regular_file_binding(
            path.resolve(),
            context=f"UNIT_PRODUCER_INPUT[{name}]",
        )
    manifest = {
        "producer_contract_v1": V3_TRAINING_DATASET_PRODUCER_CONTRACT,
        "production_allowed_v1": True,
        "model_native_entry_snapshot_v1": True,
        "exact_t5_fill_v1": True,
        "frozen_entry_snapshot_complete_v1": True,
        "canonical_m1_base_mtf_state_complete_v1": True,
        "record_schema_version_v1": V3_THIN_RECORD_SCHEMA_VERSION,
        "teacher_horizon_bars_v1": V3_TRAINING_TEACHER_HORIZON_BARS,
        "emit_stride_bars_v1": V3_TRAINING_RECORD_EMIT_STRIDE_BARS,
        "direction_authority_v1": "runtime_head_calibrated_direction_argmax",
        "flat_handling_v1": "explicit_no_order_no_exit_records",
        "io_context_cadence_contract_v1": EXIT_IO_V8_CONTEXT_CADENCE_CONTRACT,
        "io_version": EXIT_IO_V8_REGIME_M1L512_IO_VERSION,
        "feature_names_hash": EXIT_IO_V8_REGIME_M1L512_FEATURE_NAMES_HASH,
        "trade_state_feature_indices": [
            EXIT_IO_V8_REGIME_M1L512_FEATURES.index(name)
            for name in OVERLAY_COL_NAMES
        ],
        "producer_inputs_v1": producer_inputs,
        "producer_inputs_inventory_sha256_v1": _canonical_sha256(
            producer_inputs
        ),
        "xgb_bridge_source_v1": xgb_identity,
        "input_dim": EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT,
        "window_len": 512,
        "files": files,
    }
    manifest.update(semantic_updates)
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    return root


def _write_bundle(tmp_path: Path) -> tuple[Path, dict, Path]:
    dataset = _write_dataset(tmp_path / "dataset")
    _, dataset_inventory = require_authoritative_v3_training_dataset(dataset)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    state_path = bundle / "exit_transformer_v0.pt"
    state_path.write_bytes(b"unit-state")
    config = {
        "exit_ml_io_version": EXIT_IO_V8_REGIME_M1L512_IO_VERSION,
        "input_dim": 173,
        "window_len": 512,
    }
    config_path = bundle / "transformer_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    m5 = dataset / "producer_inputs" / "canonical_v3.artifact"
    source_inventory = []
    for relative in sorted(V3_TRAINING_SOURCE_CODE_FILES):
        source = bundle / "training_source_v1" / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"unit-source:{relative}".encode("utf-8"))
        source_inventory.append(
            {
                "relative_path": relative,
                **v3_regular_file_binding(
                    source.resolve(),
                    context=f"UNIT_SOURCE[{relative}]",
                ),
            }
        )
    training = {
        "seed": 1337,
        "train_cutoff": "2025-07-01T00:00:00+00:00",
        "val_cutoff": "2026-01-01T00:00:00+00:00",
    }
    lineage = {
        "schema_version": V3_TRAINING_LINEAGE_SCHEMA_VERSION,
        "production_allowed_v1": True,
        "dataset_producer_contract_v1": (
            V3_TRAINING_DATASET_PRODUCER_CONTRACT
        ),
        "dataset_root": str(dataset.resolve()),
        "dataset_files": dataset_inventory,
        "dataset_inventory_sha256": _canonical_sha256(dataset_inventory),
        "m5_prebuilt": v3_regular_file_binding(
            m5.resolve(),
            context="UNIT_M5",
        ),
        "xgb_bridge_source": json.loads(
            json.dumps(
                json.loads(
                    (dataset / "manifest.json").read_text(encoding="utf-8")
                )["xgb_bridge_source_v1"]
            )
        ),
        "source_code_files": source_inventory,
        "source_code_inventory_sha256": _canonical_sha256(source_inventory),
        "split_uid_sha256": {
            "train": hashlib.sha256(b"train").hexdigest(),
            "val": hashlib.sha256(b"val").hexdigest(),
            "test": hashlib.sha256(b"test").hexdigest(),
        },
        "training_recipe_sha256": _canonical_sha256(training),
        "transformer_config_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "initialization": {
            "mode": "cold",
            "source_state_dict": None,
        },
    }
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "exit_io_version": config["exit_ml_io_version"],
                "model_state_dict_sha256": hashlib.sha256(
                    state_path.read_bytes()
                ).hexdigest(),
                "training": training,
                "training_lineage_v1": lineage,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return bundle, config, state_path


def _runtime_head_prediction_row() -> dict[str, object]:
    evidence = _valid_evidence()
    evidence.pop("sizing_authority_contract")
    for field in (
        "decision_available_ts",
        "entry_signal_latency_sec",
        "context_cutoff_ts",
        "context_age_m5_bars",
    ):
        evidence.pop(field, None)
    head = {
        "runtime_head_evidence_schema_version": (
            MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION
        ),
        **evidence,
    }
    payload, payload_sha = encode_model_native_runtime_head_evidence(head)
    return {
        "time": evidence["decision_ts"],
        "pred_direction": evidence["model_direction_index"],
        "runtime_head_evidence_json": payload,
        "runtime_head_evidence_sha256": payload_sha,
    }


def _source_tape() -> SourceTape:
    times = pd.date_range(
        "2026-07-08T09:00:00Z",
        periods=900,
        freq="min",
    )
    ordinal = np.arange(len(times), dtype=np.float64)
    bid_open = 3300.0 + ordinal * 0.01
    bid_close = bid_open + 0.005
    ask_open = bid_open + 0.2
    ask_close = bid_close + 0.2
    return SourceTape(
        source_path=Path("/tmp/unit-source-tape.parquet"),
        source_sha256=hashlib.sha256(b"unit-source-tape").hexdigest(),
        source_size_bytes=1,
        times=times.to_numpy(),
        index=pd.Index(times),
        bid_open=bid_open,
        ask_open=ask_open,
        bid_close=bid_close,
        ask_close=ask_close,
        bid_high=np.maximum(bid_open, bid_close) + 0.1,
        bid_low=np.minimum(bid_open, bid_close) - 0.1,
        ask_high=np.maximum(ask_open, ask_close) + 0.1,
        ask_low=np.minimum(ask_open, ask_close) - 0.1,
    )


def test_reproducible_v3_training_lineage_accepts_exact_self_contained_bytes(
    tmp_path: Path,
) -> None:
    bundle, config, state_path = _write_bundle(tmp_path)

    lineage = require_reproducible_v3_training_lineage(
        bundle_dir=bundle.resolve(),
        config=config,
        state_path=state_path.resolve(),
    )

    assert lineage["production_allowed_v1"] is True
    assert lineage["initialization"]["mode"] == "cold"


def test_v3_trade_materializer_uses_argmax_t5_and_first_fill_bar() -> None:
    tape = _source_tape()

    materialized = materialize_model_native_v3_trade_records(
        prediction_row=_runtime_head_prediction_row(),
        source_tape=tape,
        base_m1_time_ns=tape.index.asi8,
        run_id="UNIT_MODEL_NATIVE_V3",
    )

    assert materialized["status"] == "MATERIALIZED"
    assert materialized["side"] == "long"
    assert materialized["entry_fill_time"] == "2026-07-08T18:00:00+00:00"
    assert len(materialized["records"]) == V3_TRAINING_TEACHER_HORIZON_BARS
    first = materialized["records"][0]
    assert first["ts"] == materialized["entry_fill_time"]
    assert first["n_in_trade_bars"] == 1
    assert first["scalars"]["bars_held"] == 1.0
    assert materialized["records"][-1]["scalars"]["bars_held"] == float(
        V3_TRAINING_TEACHER_HORIZON_BARS
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"production_allowed_v1": 1}, "AUTHORITY_MISSING"),
        ({"exact_t5_fill_v1": False}, "AUTHORITY_MISSING"),
        ({"frozen_entry_snapshot_complete_v1": False}, "AUTHORITY_MISSING"),
        (
            {"canonical_m1_base_mtf_state_complete_v1": False},
            "AUTHORITY_MISSING",
        ),
        ({"model_native_entry_snapshot_v1": False}, "AUTHORITY_MISSING"),
    ],
)
def test_v3_dataset_semantic_authority_is_exact(
    tmp_path: Path,
    updates: dict,
    message: str,
) -> None:
    dataset = _write_dataset(tmp_path / "dataset", **updates)

    with pytest.raises(RuntimeError, match=message):
        require_authoritative_v3_training_dataset(dataset.resolve())


def test_v3_dataset_rejects_nonzero_base_trade_state(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path / "dataset")
    path = dataset / "m1_feature_matrix.npy"
    matrix = np.load(path, allow_pickle=False)
    matrix[0, EXIT_IO_V8_REGIME_M1L512_FEATURES.index("pnl_bps_now")] = 1.0
    np.save(path, matrix, allow_pickle=False)

    with pytest.raises(RuntimeError, match="BASE_MATRIX_TRADE_STATE_NOT_ZERO"):
        require_authoritative_v3_training_dataset(dataset.resolve())


def test_v3_dataset_rejects_record_time_axis_mismatch(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path / "dataset")
    records_path = dataset / "records.jsonl"
    records = records_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(records[0])
    first["m1_idx_now"] += 1
    records[0] = json.dumps(first, sort_keys=True)
    records_path.write_text("\n".join(records) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="RECORD_TIME_INDEX_MISMATCH"):
        require_authoritative_v3_training_dataset(dataset.resolve())


def test_v3_dataset_rehashes_every_bound_producer_input(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path / "dataset")
    (dataset / "producer_inputs" / "source_tape.artifact").write_bytes(
        b"mutated-source-tape"
    )

    with pytest.raises(RuntimeError, match="file bytes differ"):
        require_authoritative_v3_training_dataset(dataset.resolve())


def test_v3_dataset_rejects_invalid_signal_bridge(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path / "dataset")
    path = dataset / "m1_feature_matrix.npy"
    matrix = np.load(path, allow_pickle=False)
    for name in (
        "p_long",
        "p_short",
        "p_flat",
        "p_hat",
        "uncertainty_score",
        "margin_top1_top2",
        "entropy",
    ):
        matrix[:, EXIT_IO_V8_REGIME_M1L512_FEATURES.index(name)] = 0.0
    np.save(path, matrix, allow_pickle=False)

    with pytest.raises(RuntimeError, match="XGB_SIGNAL_BRIDGE_INVALID"):
        require_authoritative_v3_training_dataset(dataset.resolve())


def test_v3_thin_record_rejects_noncanonical_time_and_boolean_numeric(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path / "dataset")
    first = json.loads(
        (dataset / "records.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    naive = dict(first)
    naive["ts"] = str(first["ts"]).removesuffix("+00:00")
    with pytest.raises(RuntimeError, match="not an exact UTC minute"):
        require_exact_v3_thin_record(naive, context="UNIT_NAIVE_TIME")

    boolean = json.loads(json.dumps(first))
    boolean["scalars"]["pnl_bps_now"] = True
    with pytest.raises(RuntimeError, match="cannot be boolean"):
        require_exact_v3_thin_record(boolean, context="UNIT_BOOLEAN_SCALAR")


def test_legacy_v3_manifest_without_lineage_fails_closed(
    tmp_path: Path,
) -> None:
    bundle, config, state_path = _write_bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["training_lineage_v1"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="LINEAGE_MISSING_OR_NONCANONICAL"):
        require_reproducible_v3_training_lineage(
            bundle_dir=bundle.resolve(),
            config=config,
            state_path=state_path.resolve(),
        )


def test_v3_lineage_rehashes_dataset_and_bundle_owned_source_bytes(
    tmp_path: Path,
) -> None:
    bundle, config, state_path = _write_bundle(tmp_path)
    dataset_file = tmp_path / "dataset" / "records.jsonl"
    records = [
        json.loads(line)
        for line in dataset_file.read_text(encoding="utf-8").splitlines()
    ]
    for record in records:
        record["runtime_head_evidence_sha256"] = "b" * 64
    dataset_file.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="DATASET_INVENTORY_MISMATCH"):
        require_reproducible_v3_training_lineage(
            bundle_dir=bundle.resolve(),
            config=config,
            state_path=state_path.resolve(),
        )

    bundle, config, state_path = _write_bundle(tmp_path / "second")
    source_file = (
        bundle
        / "training_source_v1"
        / "gx1/policy/exit_transformer_v0.py"
    )
    source_file.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="file bytes differ"):
        require_reproducible_v3_training_lineage(
            bundle_dir=bundle.resolve(),
            config=config,
            state_path=state_path.resolve(),
        )


def test_v3_lineage_rehashes_xgb_bridge_source_bytes(
    tmp_path: Path,
) -> None:
    bundle, config, state_path = _write_bundle(tmp_path)
    xgb_bundle = tmp_path / "dataset" / "xgb_bundle"
    external_contract = tmp_path / "external_xgb_input_features.json"
    external_contract.write_bytes(
        (xgb_bundle / "xgb_input_features.json").read_bytes()
    )
    with pytest.raises(RuntimeError, match="CONTRACTS_NOT_BUNDLE_OWNED"):
        build_v3_xgb_bridge_source_identity(
            bundle_dir=xgb_bundle.resolve(),
            feature_contract_path=external_contract.resolve(),
            sanitizer_config_path=(
                xgb_bundle / "xgb_input_sanitizer.json"
            ).resolve(),
        )

    (xgb_bundle / "model.joblib").write_bytes(
        b"mutated-xgb"
    )

    with pytest.raises(RuntimeError, match="XGB_BRIDGE_SOURCE"):
        require_reproducible_v3_training_lineage(
            bundle_dir=bundle.resolve(),
            config=config,
            state_path=state_path.resolve(),
        )


def test_v3_parallel_label_spill_preserves_input_uid_order() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "gx1/exits/training/disk_labeled_dataset.py"
    ).read_text(encoding="utf-8")

    assert "pool.imap_unordered" not in source
    assert "pool.imap(_spawn_label_one_trade" in source
