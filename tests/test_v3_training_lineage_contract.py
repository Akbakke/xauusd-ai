from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import replace
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
    V3_TRAINING_DATASET_EVENT_SCHEMA_VERSION,
    V3_TRAINING_DATASET_PRODUCER_CONTRACT,
    V3_TRAINING_LINEAGE_SCHEMA_VERSION,
    V3_TRAINING_PRODUCER_INPUT_NAMES,
    V3_TRAINING_RECORD_EMIT_STRIDE_BARS,
    V3_TRAINING_SOURCE_CODE_FILES,
    V3_TRAINING_TEACHER_HORIZON_BARS,
    build_v3_producer_source_inventory,
    build_v3_xgb_bridge_source_identity,
    materialize_authoritative_v3_training_dataset,
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
        "producer_event": "producer_event.json",
    }
    matrix = np.zeros(
        (
            516 + V3_TRAINING_TEACHER_HORIZON_BARS,
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
        "2026-07-08T09:29:00Z",
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
    np.concatenate([overlay, overlay], axis=0).tofile(
        root / files["trade_state_overlays"]
    )
    run_id = "UNIT_MODEL_NATIVE_V3"
    entry_indices = (511, 516)
    sides = ("long", "short")
    head_rows: list[dict[str, object]] = []
    trade_rows: list[tuple[str, str, str, int]] = []
    for direction, (entry_index, side) in enumerate(
        zip(entry_indices, sides, strict=True)
    ):
        evidence = _valid_evidence()
        evidence.pop("sizing_authority_contract")
        for field in (
            "decision_available_ts",
            "entry_signal_latency_sec",
            "context_cutoff_ts",
            "context_age_m5_bars",
        ):
            evidence.pop(field, None)
        decision_time = times[entry_index] - pd.Timedelta(minutes=5)
        evidence["decision_ts"] = decision_time.isoformat()
        if direction == 1:
            logits = np.asarray([0.2, 2.0, -1.0], dtype=np.float64)
            bias = np.asarray(
                evidence["direction_calibration_bias"],
                dtype=np.float64,
            )
            temperature = float(
                evidence["direction_calibration_temperature"]
            )
            raw_logits = (logits - bias) * temperature
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            evidence["raw_direction_logits"] = raw_logits.tolist()
            evidence["direction_logits"] = logits.tolist()
            evidence["direction_probs"] = probabilities.tolist()
            evidence["model_direction_index"] = 1
            evidence["model_direction"] = "SHORT"
            evidence["selected_side"] = 1
        head = {
            "runtime_head_evidence_schema_version": (
                MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION
            ),
            **evidence,
        }
        payload, payload_sha = encode_model_native_runtime_head_evidence(
            head
        )
        head_rows.append(
            {
                "split": "train",
                "model": "candidate",
                "time": decision_time,
                "pred_direction": direction,
                "runtime_head_evidence_json": payload,
                "runtime_head_evidence_sha256": payload_sha,
            }
        )
        trade_id = _canonical_sha256(
            {
                "run_id": run_id,
                "decision_ts": decision_time.isoformat(),
                "runtime_head_evidence_sha256": payload_sha,
            }
        )
        trade_rows.append((trade_id, payload_sha, side, entry_index))
    prediction_frame = pd.DataFrame(head_rows)
    bundle_root = root / "entry_bundle"
    bundle_root.mkdir()
    state_path = bundle_root / "model_state_dict.pt"
    state_path.write_bytes(b"unit-entry-state")
    state_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
    metadata_path = bundle_root / "bundle_metadata.json"
    metadata_path.write_text(
        json.dumps({"state_dict_sha256": state_sha}),
        encoding="utf-8",
    )
    predictions_path = root / "selective_edge_predictions.parquet"
    prediction_frame.to_parquet(predictions_path, index=False)
    predictions_sha = hashlib.sha256(
        predictions_path.read_bytes()
    ).hexdigest()
    report_path = root / "ENTRY_CANDIDATE_SELECTIVE_EDGE.json"
    evidence_declaration = {
        "schema_version": (
            "entry_candidate_model_direction_prediction_evidence_v3"
        ),
        "authoritative": True,
        "runtime_head_evidence_authoritative": True,
        "path": str(predictions_path.resolve()),
        "sha256": predictions_sha,
        "rows": len(prediction_frame),
        "splits": ["train"],
        "models": ["candidate"],
        "bundle_metadata_path": str(metadata_path.resolve()),
        "bundle_metadata_sha256": hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest(),
        "model_state_dict_sha256": state_sha,
    }
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "entry_candidate_selective_edge_v1",
                "decision": "PASS",
                "failures": [],
                "json_path": str(report_path.resolve()),
                "predictions_path": str(predictions_path.resolve()),
                "prediction_evidence": evidence_declaration,
                "bundle_metadata_sha256": evidence_declaration[
                    "bundle_metadata_sha256"
                ],
                "model_state_dict_sha256": state_sha,
                "splits": ["train"],
                "models": ["candidate"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "trade_uid": f"{run_id}:{trade_id}:{side}",
                "overlay_offset": index
                * V3_TRAINING_TEACHER_HORIZON_BARS,
                "overlay_length": V3_TRAINING_TEACHER_HORIZON_BARS,
            }
            for index, (trade_id, _head_sha, side, _entry_index) in enumerate(
                trade_rows
            )
        ]
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
    records = []
    for trade_id, head_sha, side, entry_index in trade_rows:
        entry_fill_time = times[entry_index]
        decision_time = entry_fill_time - pd.Timedelta(minutes=5)
        trade_uid = f"{run_id}:{trade_id}:{side}"
        for offset in range(V3_TRAINING_TEACHER_HORIZON_BARS):
            records.append(
                {
                    "schema_version": V3_THIN_RECORD_SCHEMA_VERSION,
                    "ts": times[entry_index + offset].isoformat(),
                    "decision_ts": decision_time.isoformat(),
                    "entry_fill_time": entry_fill_time.isoformat(),
                    "runtime_head_evidence_sha256": head_sha,
                    "run_id": run_id,
                    "trade_uid": trade_uid,
                    "trade_id": trade_id,
                    "side": side,
                    "m1_idx_now": entry_index + offset,
                    "in_trade_start_in_win": 511 - offset,
                    "n_in_trade_bars": offset + 1,
                    "overlay_start_row": 0,
                    "scalars": {
                        name: float(
                            overlay[
                                offset,
                                OVERLAY_COL_NAMES.index(name),
                            ]
                        )
                        for name in scalar_names
                    },
                    "teacher_final_pnl_bps": float(
                        overlay[
                            -1,
                            OVERLAY_COL_NAMES.index("pnl_bps_now"),
                        ]
                    ),
                    "teacher_final_mfe_bps": float(
                        overlay[-1, OVERLAY_COL_NAMES.index("mfe_bps")]
                    ),
                    "teacher_final_mae_bps": float(
                        overlay[-1, OVERLAY_COL_NAMES.index("mae_bps")]
                    ),
                    "teacher_duration_bars": (
                        V3_TRAINING_TEACHER_HORIZON_BARS
                    ),
                }
            )
    (root / files["records"]).write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    producer_inputs = {
        "prediction_parquet": v3_regular_file_binding(
            predictions_path.resolve(),
            context="UNIT_PRODUCER_INPUT[prediction_parquet]",
        ),
        "prediction_report": v3_regular_file_binding(
            report_path.resolve(),
            context="UNIT_PRODUCER_INPUT[prediction_report]",
        ),
        "entry_bundle_metadata": v3_regular_file_binding(
            metadata_path.resolve(),
            context="UNIT_PRODUCER_INPUT[entry_bundle_metadata]",
        ),
        "entry_model_state": v3_regular_file_binding(
            state_path.resolve(),
            context="UNIT_PRODUCER_INPUT[entry_model_state]",
        ),
    }
    producer_root = root / "producer_inputs"
    producer_root.mkdir()
    for name in sorted(
        V3_TRAINING_PRODUCER_INPUT_NAMES - set(producer_inputs)
    ):
        path = producer_root / f"{name}.artifact"
        path.write_bytes(f"unit-producer-input:{name}".encode("utf-8"))
        producer_inputs[name] = v3_regular_file_binding(
            path.resolve(),
            context=f"UNIT_PRODUCER_INPUT[{name}]",
        )
    producer_inputs = dict(sorted(producer_inputs.items()))
    producer_sources = build_v3_producer_source_inventory(
        Path(__file__).resolve().parents[1]
    )
    common = {
        "run_id_v1": run_id,
        "producer_event_schema_version_v1": (
            V3_TRAINING_DATASET_EVENT_SCHEMA_VERSION
        ),
        "producer_inputs_v1": producer_inputs,
        "producer_inputs_inventory_sha256_v1": _canonical_sha256(
            producer_inputs
        ),
        "producer_source_files_v1": producer_sources,
        "producer_source_inventory_sha256_v1": _canonical_sha256(
            producer_sources
        ),
        "xgb_bridge_source_v1": xgb_identity,
        "prediction_model_v1": "candidate",
        "prediction_splits_v1": ["train"],
        "prediction_rows_v1": 2,
        "direction_counts_v1": {"LONG": 1, "SHORT": 1, "FLAT": 0},
        "trade_count_v1": 2,
        "record_count_v1": 2 * V3_TRAINING_TEACHER_HORIZON_BARS,
        "first_decision_ts_v1": head_rows[0]["time"].isoformat(),
        "last_decision_ts_v1": head_rows[-1]["time"].isoformat(),
        "first_m1_ts_v1": times[0].isoformat(),
        "last_m1_ts_v1": times[-1].isoformat(),
    }
    member_inventory = []
    for name in sorted(
        {
            "m1_feature_matrix",
            "m1_time_ns",
            "trade_state_overlays",
            "overlay_index",
            "records",
        }
    ):
        binding = v3_regular_file_binding(
            (root / files[name]).resolve(),
            context=f"UNIT_DATASET_MEMBER[{name}]",
        )
        member_inventory.append(
            {
                "name": name,
                "relative_path": files[name],
                "sha256": binding["sha256"],
                "size_bytes": binding["size_bytes"],
            }
        )
    event = {
        "schema_version": V3_TRAINING_DATASET_EVENT_SCHEMA_VERSION,
        "producer_contract_v1": V3_TRAINING_DATASET_PRODUCER_CONTRACT,
        "decision": "PASS",
        "failures": [],
        "created_utc": "2026-07-23T20:00:00+00:00",
        "run_id": run_id,
        "production_allowed_v1": True,
        "io_version": EXIT_IO_V8_REGIME_M1L512_IO_VERSION,
        "input_dim": EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT,
        "window_len": 512,
        "feature_names_hash": EXIT_IO_V8_REGIME_M1L512_FEATURE_NAMES_HASH,
        "dataset_members_v1": member_inventory,
        "dataset_members_inventory_sha256_v1": _canonical_sha256(
            member_inventory
        ),
        **{
            key: value
            for key, value in common.items()
            if key
            not in {
                "run_id_v1",
                "producer_event_schema_version_v1",
            }
        },
    }
    (root / files["producer_event"]).write_text(
        json.dumps(event, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "producer_contract_v1": V3_TRAINING_DATASET_PRODUCER_CONTRACT,
        "production_allowed_v1": True,
        "model_native_entry_snapshot_v1": True,
        "exact_t5_fill_v1": True,
        "frozen_entry_snapshot_complete_v1": True,
        "canonical_m1_base_state_complete_v1": True,
        "multi_tf_training_state_owner_v1": (
            "trainer_recomputes_from_exact_bound_canonical_v3"
        ),
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
        **common,
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


def test_v3_source_inventory_covers_transitive_local_producer_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pending = [
        "gx1/contracts/entry_model_native_bundle_commit_v1.py",
        "gx1/contracts/entry_model_native_runtime_evidence_v1.py",
        "gx1/execution/model_native_entry_replay_v1.py",
        "gx1/execution/v12_m1_to_m5_downsample.py",
        "gx1/execution/v12_state_from_prebuilt.py",
        "gx1/execution/v12_v3_live.py",
        "gx1/execution/v12_xgb_live.py",
        "gx1/exits/training/thin_record_dataset.py",
        "gx1/features/trade_overlay.py",
        "gx1/features/volume_features.py",
        "gx1/scripts/entry_candidate_prediction_evidence_v1.py",
    ]
    observed: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        source_path = repo_root / relative
        if not source_path.is_file():
            continue
        observed.add(relative)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
        for module in modules:
            if not module.startswith(("gx1", "gx1_guards")):
                continue
            module_path = Path(*module.split("."))
            candidate = module_path.with_suffix(".py")
            package_init = module_path / "__init__.py"
            if (repo_root / candidate).is_file():
                pending.append(candidate.as_posix())
            elif (repo_root / package_init).is_file():
                pending.append(package_init.as_posix())

    assert observed.issubset(V3_TRAINING_SOURCE_CODE_FILES)


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


def test_authoritative_v3_dataset_producer_owns_feature_and_record_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_dataset(tmp_path / "input_fixture")
    fixture_manifest = json.loads(
        (fixture / "manifest.json").read_text(encoding="utf-8")
    )
    inputs = fixture_manifest["producer_inputs_v1"]
    xgb_identity = fixture_manifest["xgb_bridge_source_v1"]
    prediction_path = Path(inputs["prediction_parquet"]["path"])
    report_path = Path(inputs["prediction_report"]["path"])
    entry_bundle = Path(inputs["entry_bundle_metadata"]["path"]).parent
    source_path = Path(inputs["source_tape"]["path"])
    prebuilt_manifest = Path(inputs["prebuilt_pair_manifest"]["path"])
    prebuilt_identity = {
        "manifest_path": str(prebuilt_manifest),
        "manifest_sha256": inputs["prebuilt_pair_manifest"]["sha256"],
        "pair_generation_id": "unit-frozen-pair",
        "canonical_v3": {
            "path": inputs["canonical_v3"]["path"],
            "sha256": inputs["canonical_v3"]["sha256"],
            "rows": 289,
            "cols_total": 1,
        },
        "base28": {
            "path": inputs["base_m1"]["path"],
            "sha256": inputs["base_m1"]["sha256"],
            "rows": 900,
            "cols_total": 1,
        },
        "refresh_enabled": False,
    }

    base_index = pd.date_range(
        "2026-07-08T07:54:00Z",
        periods=900,
        freq="min",
    )
    base_m1 = pd.DataFrame({"unit": np.arange(900)}, index=base_index)
    canonical_index = pd.date_range(
        "2026-07-08T00:00:00Z",
        "2026-07-09T00:00:00Z",
        freq="5min",
    )
    canonical_v3 = pd.DataFrame(
        {"unit": np.arange(len(canonical_index))},
        index=canonical_index,
    )
    source = _source_tape()
    tape = SourceTape(
        source_path=source_path,
        source_sha256=inputs["source_tape"]["sha256"],
        source_size_bytes=inputs["source_tape"]["size_bytes"],
        times=source.times,
        index=source.index,
        bid_open=source.bid_open,
        ask_open=source.ask_open,
        bid_close=source.bid_close,
        ask_close=source.ask_close,
        bid_high=source.bid_high,
        bid_low=source.bid_low,
        ask_high=source.ask_high,
        ask_low=source.ask_low,
    )

    from gx1.execution import model_native_entry_replay_v1 as replay_module
    from gx1.execution import v12_state_from_prebuilt as prebuilt_module
    from gx1.execution import v12_v3_live as v3_live_module
    from gx1.execution import v12_xgb_live as xgb_live_module
    from gx1.scripts import (
        entry_candidate_prediction_evidence_v1 as prediction_module,
    )

    monkeypatch.setattr(
        replay_module.SourceTape,
        "load",
        classmethod(lambda _cls, _path: tape),
    )
    monkeypatch.setattr(
        prebuilt_module.PrebuiltStateLoader,
        "load_frozen_pair",
        lambda _self: prebuilt_identity,
    )
    monkeypatch.setattr(
        prebuilt_module.PrebuiltStateLoader,
        "frozen_pair_frames",
        lambda _self: (canonical_v3, base_m1, prebuilt_identity),
    )
    fake_xgb = type(
        "UnitXGB",
        (),
        {"_runtime_identity": xgb_identity},
    )()
    monkeypatch.setattr(
        xgb_live_module.XGBLiveInference,
        "load",
        classmethod(lambda _cls, _path: fake_xgb),
    )

    def _unit_feature_rows(
        *,
        target_m1: pd.DataFrame,
        volume_history_m1: pd.DataFrame,
        canonical_v3: pd.DataFrame,
        xgb_inferer: object,
        feature_names: list[str],
    ) -> np.ndarray:
        assert len(volume_history_m1) == len(target_m1) + 95
        assert canonical_v3 is not None
        assert xgb_inferer is fake_xgb
        assert feature_names == list(EXIT_IO_V8_REGIME_M1L512_FEATURES)
        matrix = np.zeros(
            (len(target_m1), EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT),
            dtype=np.float32,
        )
        probabilities = np.tile(
            np.asarray([[0.7, 0.2, 0.1]], dtype=np.float32),
            (len(target_m1), 1),
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
            matrix[:, EXIT_IO_V8_REGIME_M1L512_FEATURES.index(name)] = (
                bridge[:, column]
            )
        return matrix

    monkeypatch.setattr(
        v3_live_module,
        "build_v3_base_feature_rows",
        _unit_feature_rows,
    )
    monkeypatch.setattr(
        prediction_module,
        "resolve_and_validate_prediction_evidence",
        lambda *_args, **_kwargs: (
            prediction_path,
            {},
            {"models": ["candidate"], "splits": ["train"]},
        ),
    )

    output = tmp_path / "materialized_v3"
    producer_args = {
        "run_id": "UNIT_PRODUCER_V3",
        "prediction_parquet": prediction_path,
        "prediction_report": report_path,
        "entry_bundle_dir": entry_bundle,
        "entry_dataset_dir": fixture,
        "source_tape_path": source_path,
        "xgb_bundle_dir": Path(xgb_identity["bundle_root"]),
        "prebuilt_pair_manifest_path": prebuilt_manifest,
        "prebuilt_generation_root": tmp_path.resolve(),
        "expected_model": "candidate",
        "expected_splits": ["train"],
        "chunk_rows": 512,
    }
    destination, manifest, inventory = (
        materialize_authoritative_v3_training_dataset(
            output_dir=output.resolve(),
            **producer_args,
        )
    )

    assert destination == output.resolve()
    assert manifest["direction_counts_v1"] == {
        "LONG": 1,
        "SHORT": 1,
        "FLAT": 0,
    }
    assert manifest["trade_count_v1"] == 2
    assert manifest["record_count_v1"] == (
        2 * V3_TRAINING_TEACHER_HORIZON_BARS
    )
    assert {item["relative_path"] for item in inventory} == {
        "m1_feature_matrix.npy",
        "m1_time_ns.npy",
        "manifest.json",
        "overlay_index.parquet",
        "producer_event.json",
        "records.jsonl",
        "trade_state_overlays.f32",
    }
    assert not list(tmp_path.glob(".materialized_v3.staging.*"))

    monkeypatch.setattr(
        replay_module.SourceTape,
        "load",
        classmethod(
            lambda _cls, _path: replace(tape, source_sha256="0" * 64)
        ),
    )
    with pytest.raises(RuntimeError, match="SOURCE_TAPE_IDENTITY_MISMATCH"):
        materialize_authoritative_v3_training_dataset(
            output_dir=(tmp_path / "rejected_source_tape").resolve(),
            **producer_args,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"production_allowed_v1": 1}, "AUTHORITY_MISSING"),
        ({"exact_t5_fill_v1": False}, "AUTHORITY_MISSING"),
        ({"frozen_entry_snapshot_complete_v1": False}, "AUTHORITY_MISSING"),
        (
            {"canonical_m1_base_state_complete_v1": False},
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
    records = dataset_file.read_text(encoding="utf-8").splitlines()
    dataset_file.write_text(
        "".join(f"{record} \n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="V3_TRAINING_DATASET_PRODUCER_EVENT_LINEAGE_MISMATCH",
    ):
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
