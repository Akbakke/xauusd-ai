from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    REQUIRED_POSITIVE_LOSS_WEIGHTS,
    training_objective_contract_metadata,
)
from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_DECISION_BAR_SECONDS,
    ENTRY_MTF_CONTEXT_COUNT,
    ENTRY_MTF_CONTEXT_TIMEFRAMES,
    EXIT_DECISION_BAR_SECONDS,
    EXIT_MTF_CONTEXT_COUNT,
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import (
    direction_evidence_fusion_metadata,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    model_native_aux_target_contract_metadata,
)
from gx1.contracts.entry_model_native_learned_component_movement_v1 import (
    COMPONENT_PARAMETERS,
    PARAMETER_SHAPES,
    REFERENCE as MOVEMENT_REFERENCE,
    SCHEMA_VERSION as MOVEMENT_SCHEMA_VERSION,
)
from gx1.contracts.entry_model_native_tf_input_scale_v1 import (
    NEUTRAL_EFFECTIVE_INIT,
    build_tf_input_scale_contract,
    raw_tf_input_scale_from_effective,
)
from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    CORE_ARTIFACTS as BUNDLE_COMMIT_CORE_ARTIFACTS,
    write_bundle_commit_manifest,
)
from gx1.contracts.unified_exit_lifecycle_v1 import (
    UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION,
    UNIFIED_EXIT_M1_AUTHORITY_SCHEMA_VERSION,
    canonical_json_sha256,
)
from gx1.features.htf_features import (
    HTF_V4_MATRIX_CONTRACT,
    MULTI_TF_FEATURE_COUNT_V4,
    MULTI_TF_FEATURE_NAMES_SHA256_V4,
    MULTI_TF_PER_BAR_FEATURES_V4,
    require_multi_tf_resolution_pyramid,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MULTI_TF_SPECIALIST_ROUTING_SCHEMA_VERSION,
    require_multi_tf_specialist_routing_v4,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
    model_direction_decision_contract_metadata,
    unified_entry_exit_contract_metadata,
)
from tests.model_native_input_normalization_support import (
    decision_window_coverage_fixture,
)
from gx1.models.entry_v10.entry_v10_bundle import (
    _ENTRY_HEAD_STATE_KEYS,
    _MODEL_NATIVE_METADATA_ONLY_COMPONENTS,
    _MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS,
    _MODEL_NATIVE_REQUIRED_SPECIALISTS,
    _MODEL_NATIVE_ZERO_INIT_COMPONENT_GROUPS,
    _require_exact_model_native_bundle_metadata,
)
from gx1.scripts import fit_entry_direction_calibration_v1 as calibration
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    build_prediction_evidence_declaration,
    sha256_file,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields
from tests.model_native_turning_point_support import (
    turning_point_prediction_columns,
)
from tests.model_native_offline_rl_support import offline_rl_prediction_columns
from tests.model_native_context_routing_support import (
    context_routing_for_ordered_signal_names,
)
from tests.model_native_input_normalization_support import (
    input_normalization_fit_population_proof_fixture,
    input_normalization_fixture,
)
from tests.model_native_test_seal_support import (
    prefreeze_test_seal_lineage_fixture,
)


SOURCE_STAMP = "20260716T100000123456Z"
PREDICTION_STAMP = "20260716T110000123456Z"
SECOND_PREDICTION_STAMP = "20260716T123000123456Z"
OUTPUT_STAMP = "20260716T120000123456Z"
SECOND_OUTPUT_STAMP = "20260716T130000123456Z"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _source_bundle(
    tmp_path: Path,
    *,
    direction_calibration_present: bool = False,
) -> Path:
    bundle = tmp_path / f"entry_model_native_bundle_{SOURCE_STAMP}"
    bundle.mkdir()
    selected = canonical_model_native_selected_fields(
        remainder_prefix="session_regime.immutable_calibration_fixture"
    )
    signal_contract = model_native_signal_contract_metadata(selected)
    objective = training_objective_contract_metadata(
        {name: 1.0 for name in REQUIRED_POSITIVE_LOSS_WEIGHTS}
    )
    context_routing = context_routing_for_ordered_signal_names(
        list(signal_contract["fields"])
    )
    movement = {
        "schema_version": MOVEMENT_SCHEMA_VERSION,
        "reference": MOVEMENT_REFERENCE,
        "selected_checkpoint_epoch": 1,
        "parameter_deltas": {
            key: {
                "shape": list(shape),
                "max_abs_delta": 0.1,
                "l2_delta": 0.2,
                "changed": True,
            }
            for key, shape in PARAMETER_SHAPES.items()
        },
        "component_changed": {key: True for key in COMPONENT_PARAMETERS},
        "output_rows_distinct": True,
        "decision": "PASS",
    }
    state: dict[str, torch.Tensor] = {}
    active_state_heads = (
        _MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS - _MODEL_NATIVE_METADATA_ONLY_COMPONENTS
    )
    for head in active_state_heads:
        for key in _ENTRY_HEAD_STATE_KEYS[head]:
            state[key] = torch.ones(1, dtype=torch.float32)
    for keys in _MODEL_NATIVE_ZERO_INIT_COMPONENT_GROUPS.values():
        for key in keys:
            state[key] = torch.ones(1, dtype=torch.float32)
    state.update(
        {
            "evidence_fusion_norm.weight": torch.ones(96),
            "evidence_fusion_norm.bias": torch.zeros(96),
            "evidence_fusion_in.weight": torch.ones(128, 96),
            "evidence_fusion_in.bias": torch.zeros(128),
            "evidence_fusion_out.weight": torch.arange(
                3 * 128, dtype=torch.float32
            ).reshape(3, 128) + 1.0,
            "evidence_fusion_out.bias": torch.zeros(3),
        }
    )
    tf_inits = {
        name: NEUTRAL_EFFECTIVE_INIT
        for name in ("m5", "m15", "h1", "h4", "d1")
    }
    learned_tf_raw = {
        name: raw_tf_input_scale_from_effective(value)
        for name, value in tf_inits.items()
    }
    for name, raw in learned_tf_raw.items():
        state[f"tf_input_scale_{name}"] = torch.tensor(raw, dtype=torch.float32)
    mtf_routing = require_multi_tf_specialist_routing_v4(
        MULTI_TF_PER_BAR_FEATURES_V4
    )
    for timeframe in ("m5", "m15", "h1", "h4", "d1"):
        for specialist, indices in mtf_routing.items():
            state[
                "mtf_feature_context_gate."
                f"{timeframe}__{specialist}.weight"
            ] = torch.ones(len(indices), 1, dtype=torch.float32)
    state_path = bundle / "model_state_dict.pt"
    torch.save(state, state_path)
    state_sha = sha256_file(state_path)

    mtf_contract = {
        "enabled": True,
        "v4_mode": True,
        "route_schema_version": "entry_exit_shared_mtf_routes_v1",
        "entry_route_timeframes": list(ENTRY_MTF_CONTEXT_TIMEFRAMES),
        "exit_route_timeframes": list(EXIT_MTF_CONTEXT_TIMEFRAMES),
        "entry_target_availability_shift_minutes": (
            ENTRY_DECISION_BAR_SECONDS / 60.0
        ),
        "exit_target_availability_shift_minutes": (
            EXIT_DECISION_BAR_SECONDS / 60.0
        ),
        "entry_tf_gate_width": ENTRY_MTF_CONTEXT_COUNT,
        "exit_tf_gate_width": EXIT_MTF_CONTEXT_COUNT,
        "entry_family_tf_gate_width": (
            ENTRY_MTF_CONTEXT_COUNT * 8
        ),
        "exit_family_tf_gate_width": EXIT_MTF_CONTEXT_COUNT * 8,
        "shared_cache_identity_sha256": "a" * 64,
        "shared_cache_manifest_sha256": "4" * 64,
        "shared_cache_dir": "/fixture/mtf",
        "shared_cache_manifest_path": "/fixture/mtf/manifest.json",
        "shared_cache_m5_source": "/fixture/xau_m5_full_history.parquet",
        "shared_cache_m5_source_sha256": "c" * 64,
        "m5_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
        "m5_seq_len": 16,
        "m15_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
        "m15_seq_len": 64,
        "h1_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
        "h1_seq_len": 96,
        "h4_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
        "h4_seq_len": 96,
        "d1_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
        "d1_seq_len": 252,
        "multi_tf_num_layers": 1,
        "multi_tf_scale": 0.5,
        "feature_contract": "HTF_V4_EIGHT_FAMILY_CAUSAL_MATRIX_V2",
        "matrix_contract": HTF_V4_MATRIX_CONTRACT,
        "feature_names": list(MULTI_TF_PER_BAR_FEATURES_V4),
        "feature_names_sha256": MULTI_TF_FEATURE_NAMES_SHA256_V4,
        "closed_bar_target_availability": True,
    }
    per_tf_seq_lens = {
        "M5": 16,
        "M15": 64,
        "H1": 96,
        "H4": 96,
        "D1": 252,
    }
    mtf_specialist_indices = {
        name: list(indices)
        for name, indices in require_multi_tf_specialist_routing_v4(
            MULTI_TF_PER_BAR_FEATURES_V4
        ).items()
    }
    mtf_contract.update(
        {
            "resolution_pyramid": require_multi_tf_resolution_pyramid(
                per_tf_seq_lens
            ),
            "decision_window_coverage": decision_window_coverage_fixture(
                per_tf_seq_lens
            ),
            "specialist_routing_schema_version": (
                MULTI_TF_SPECIALIST_ROUTING_SCHEMA_VERSION
            ),
            "specialist_input_indices": mtf_specialist_indices,
            "parameter_family_tf_token_order": [
                f"{tf}:{specialist}"
                for tf in ("m5", "m15", "h1", "h4", "d1")
                for specialist in mtf_specialist_indices
            ],
            "entry_family_tf_token_order": [
                f"{tf.lower()}:{specialist}"
                for tf in ENTRY_MTF_CONTEXT_TIMEFRAMES
                for specialist in mtf_specialist_indices
            ],
            "exit_family_tf_token_order": [
                f"{tf.lower()}:{specialist}"
                for tf in EXIT_MTF_CONTEXT_TIMEFRAMES
                for specialist in mtf_specialist_indices
            ],
        }
    )
    tf_scale_contract = build_tf_input_scale_contract(
        init_effective=tf_inits,
        learned_raw=learned_tf_raw,
    )
    input_normalization = input_normalization_fixture(
        signal_names=list(signal_contract["fields"]),
        mtf_names=list(MULTI_TF_PER_BAR_FEATURES_V4),
        per_tf_seq_lens=per_tf_seq_lens,
        dataset_run_id="MODEL_NATIVE_CALIBRATION_DATASET_PYTEST_V1",
    )
    native_subset_years = {
        "year=2026": {
            "rows": 20,
            "canonical_rows_sha256": "3" * 64,
        }
    }
    m1_authority = {
        "schema_version": UNIFIED_EXIT_M1_AUTHORITY_SCHEMA_VERSION,
        "m1_source_path": "/immutable/pair/base28.parquet",
        "m1_source_sha256": "2" * 64,
        "base28_native_m1_subset_proof": {
            "method": "exact_base28_rows_are_native_m1_subset_v1",
            "rows": 20,
            "years": native_subset_years,
            "proof_sha256": canonical_json_sha256(native_subset_years),
        },
    }
    unified_exit_training_evidence = {
        "schema_version": "gx1_unified_exit_training_evidence_v1",
        "decision": "PASS",
        "shared_model_state_dict": True,
        "entry_representation_surface": "shared_feature_representation",
        "future_outcomes_used_as_model_inputs": False,
        "exit_action_loss_weight": 1.0,
        "lifecycle": {
            "schema_version": UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION,
            "future_outcomes_used_as_model_inputs": False,
            "m1_source_path": m1_authority["m1_source_path"],
            "m1_source_sha256": m1_authority["m1_source_sha256"],
            "m1_authority": m1_authority,
            "m1_authority_sha256": canonical_json_sha256(m1_authority),
        },
        "selected_checkpoint_validation": {
            "unified_exit_action_loss_mean": 0.5,
            "unified_exit_action_rows": 20,
            "unified_exit_hold_rows": 10,
            "unified_exit_now_rows": 10,
            "unified_exit_action_accuracy": 0.6,
        },
        "selected_checkpoint_parameter_movement": {
            "all_exit_components_moved": True,
        },
    }
    shared = {
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": MODEL_NATIVE_SEQ_LEN,
        "dropout": 0.05,
        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "ordered_signal_names": list(signal_contract["fields"]),
        "ordered_ctx_cont_names": list(MODEL_NATIVE_CTX_CONT_FIELDS),
        "ordered_ctx_cat_names": list(MODEL_NATIVE_CTX_CAT_FIELDS),
        "model_native_signal_contract": signal_contract,
        "model_native_training_objective": objective,
        "model_native_direction_evidence_fusion": (
            direction_evidence_fusion_metadata()
        ),
        "model_native_learned_component_movement": movement,
        "context_specialist_routing": context_routing,
        "input_normalization": input_normalization,
        "input_normalization_fit_population_proof": (
            input_normalization_fit_population_proof_fixture(
                input_normalization
            )
        ),
        "multi_tf": mtf_contract,
        "tf_input_scale": tf_scale_contract,
        "aux_head_target_contract": model_native_aux_target_contract_metadata(),
        "run_lineage": {
            "schema_version": "entry_model_native_training_run_lineage_v2",
            "training_run_id": "MODEL_NATIVE_CALIBRATION_TRAIN_PYTEST_V1",
            "dataset_run_id": "MODEL_NATIVE_CALIBRATION_DATASET_PYTEST_V1",
            "training_profile": "candidate",
            "requested_subsample_rows": 0,
            "physical_train_rows": 100,
            "effective_train_rows": 100,
        },
        "prefreeze_test_seal_lineage": prefreeze_test_seal_lineage_fixture(
            dataset_run_id="MODEL_NATIVE_CALIBRATION_DATASET_PYTEST_V1",
            dataset_dir="/immutable/calibration_seq513_dataset",
        ),
        "m1_feature_surface_binding": {
            "parquet_path": "/immutable/calibration_m1_feature_base.parquet",
            "manifest_path": (
                "/immutable/calibration_m1_feature_base.parquet.manifest.json"
            ),
            "dataset_run_id": "MODEL_NATIVE_CALIBRATION_DATASET_PYTEST_V1",
            "pair_generation_id": "MODEL_NATIVE_CALIBRATION_PAIR_PYTEST_V1",
            "parquet_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "feature_field_order_sha256": canonical_json_sha256(
                list(signal_contract["fields"])
            ),
        },
        "unified_entry_exit_contract": unified_entry_exit_contract_metadata(),
        "unified_exit_training_evidence": unified_exit_training_evidence,
    }
    direction_contract = model_direction_decision_contract_metadata()
    lock = {
        **shared,
        "direction_decision_contract": direction_contract,
        "model_path_relative": "model_state_dict.pt",
        "model_sha256": state_sha,
    }
    specialist_indices = {name: [] for name in _MODEL_NATIVE_REQUIRED_SPECIALISTS}
    for index in range(MODEL_NATIVE_SIGNAL_DIM):
        specialist_indices[
            _MODEL_NATIVE_REQUIRED_SPECIALISTS[
                index % len(_MODEL_NATIVE_REQUIRED_SPECIALISTS)
            ]
        ].append(index)
    metadata = {
        **shared,
        "state_dict_sha256": state_sha,
        "supports_context_features": True,
        "anchored_entry_enabled": False,
        "anchor_source": None,
        "anchor_gate": {"enabled": False},
        "direction_decision_contract": direction_contract,
        "train_recipe": {
            "active_heads": sorted(_MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS),
            "aux_regression_positive_only": True,
            "selector_masked_aux": True,
            "symmetric_negatives": True,
            "aux_selector_mode": "long_short_union",
        },
        "enable_pos_enc": True,
        "enable_regime_film": True,
        "hierarchical_entry_heads": {"enabled": True},
        "trendline_rail_head": {
            "enabled": True,
            "output_dim": 6,
            "hand_written_direction_pressure": False,
            "direction_mapping": "direct_learned_evidence_fusion",
        },
        "model_native_state_contract": {
            "decision": "PASS",
            "entry_run_id": "MODEL_NATIVE_CALIBRATION_DATASET_PYTEST_V1",
        },
        "specialist_fusion": {
            "enabled": True,
            "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "input_indices": specialist_indices,
            "context_routing": context_routing,
            "trainable_specialists": list(_MODEL_NATIVE_REQUIRED_SPECIALISTS),
            "num_layers": 1,
            "fusion_scale": 0.25,
            "cross_family_fusion_scale": 0.25,
        },
    }
    if direction_calibration_present:
        metadata["direction_calibration"] = None
    _write_json(bundle / "MASTER_TRANSFORMER_LOCK.json", lock)
    _write_json(bundle / "bundle_metadata.json", metadata)
    write_bundle_commit_manifest(
        bundle_dir=bundle.resolve(),
        artifact_names=BUNDLE_COMMIT_CORE_ARTIFACTS,
        bundle_kind="trained",
        created_at_utc="2026-07-16T10:00:00+00:00",
    )
    return bundle


def _prediction_frame(rows: int = 120) -> pd.DataFrame:
    labels = np.arange(rows, dtype=np.int64) % 3
    probabilities = np.full((rows, 3), 0.25, dtype=np.float64)
    probabilities[np.arange(rows), labels] = 0.5
    logits = np.log(probabilities)
    public_logits = np.column_stack([np.maximum(logits[:, 0], logits[:, 1]), logits[:, 2]])
    public_exp = np.exp(public_logits - public_logits.max(axis=1, keepdims=True))
    public_probabilities = public_exp / public_exp.sum(axis=1, keepdims=True)
    path_pred = np.linspace(-2.0, 2.0, rows)
    bad_labels = np.arange(rows, dtype=np.int64) % 2
    tradable = np.arange(rows) < 100
    selector_long = np.arange(rows) < 100
    selector_short = np.arange(rows) >= 100
    return pd.DataFrame(
        {
            "split": ["val"] * rows,
            "model": ["candidate"] * rows,
            "time": pd.date_range("2026-07-01", periods=rows, freq="5min", tz="UTC"),
            "y_direction": labels,
            "pred_direction": np.argmax(logits, axis=1),
            "p_long": probabilities[:, 0],
            "p_short": probabilities[:, 1],
            "p_flat": probabilities[:, 2],
            "selection_score_mode": [MODEL_DIRECTION_SELECTION_MODE] * rows,
            "public_trade_probability": public_probabilities[:, 0],
            "public_flat_probability": public_probabilities[:, 1],
            "public_trade_flat_margin": public_logits[:, 0] - public_logits[:, 1],
            "public_trade_flat_hard_decision": np.argmax(public_logits, axis=1),
            "direction_logits": [row.tolist() for row in logits],
            "public_trade_flat_decision_logits": [row.tolist() for row in public_logits],
            "path_quality_pred": path_pred,
            "path_quality_bps": 2.0 * path_pred + 3.0,
            "bad_path_prob": np.where(bad_labels == 1, 0.4, 0.2),
            "y_bad_path": bad_labels,
            "y_tradable": tradable.astype(np.int64),
            "y_selector_long_mask": selector_long.astype(np.int64),
            "y_selector_short_mask": selector_short.astype(np.int64),
            **turning_point_prediction_columns(rows),
            **offline_rl_prediction_columns(rows),
        }
    )


def _prediction_event(
    tmp_path: Path,
    bundle: Path,
    *,
    stamp: str = PREDICTION_STAMP,
) -> dict[str, Path]:
    dataset = tmp_path / "entry_model_native_dataset"
    reports = tmp_path / (
        "prediction_events" if stamp == PREDICTION_STAMP else f"prediction_events_{stamp}"
    )
    dataset.mkdir(exist_ok=True)
    reports.mkdir()
    predictions = reports / f"selective_edge_predictions_{stamp}.parquet"
    _prediction_frame().to_parquet(predictions, index=False)
    metadata = json.loads((bundle / "bundle_metadata.json").read_text(encoding="utf-8"))
    evidence = build_prediction_evidence_declaration(
        predictions_path=predictions,
        bundle_dir=bundle,
        bundle_metadata=metadata,
        evidence_stage="pre_calibration",
        requested_splits=["val"],
    )
    report_path = reports / f"ENTRY_CANDIDATE_SELECTIVE_EDGE_{stamp}.json"
    report = {
        "schema_version": "entry_candidate_selective_edge_v1",
        "created_utc": datetime.strptime(stamp, "%Y%m%dT%H%M%S%fZ")
        .replace(tzinfo=timezone.utc)
        .isoformat(),
        "decision": "PASS",
        "failures": [],
        "evidence_stage": "pre_calibration",
        "bundle_dir": str(bundle),
        "dataset_dir": str(dataset),
        "splits": ["val"],
        "models": ["candidate"],
        "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
        "direction_decision_contract": model_direction_decision_contract_metadata(),
        "prediction_evidence": evidence,
        "predictions_path": str(predictions),
        "bundle_metadata_sha256": evidence["bundle_metadata_sha256"],
        "model_state_dict_sha256": evidence["model_state_dict_sha256"],
        "json_path": str(report_path),
    }
    _write_json(report_path, report)
    return {
        "dataset": dataset,
        "predictions": predictions,
        "report": report_path,
        "reports": reports,
    }


def _args(
    bundle: Path,
    event: dict[str, Path],
    output: Path,
    *,
    head: str = "direction",
    execute: bool = True,
) -> list[str]:
    values = [
        "--source-bundle-dir",
        str(bundle),
        "--out-bundle-dir",
        str(output),
        "--predictions-parquet",
        str(event["predictions"]),
        "--prediction-report-json",
        str(event["report"]),
        "--predictions-sha256",
        sha256_file(event["predictions"]),
        "--dataset-dir",
        str(event["dataset"]),
        "--model",
        "candidate",
        "--heads",
        head,
        "--fit-split",
        "val",
        "--run-id",
        "MODEL_NATIVE_CALIBRATION_TEST",
        "--min-fit-rows",
        "90",
    ]
    values.append("--execute" if execute else "--dry-run")
    return values


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_direction_execute_publishes_new_hash_bound_bundle_without_source_mutation(
    tmp_path: Path,
) -> None:
    source = _source_bundle(tmp_path)
    event = _prediction_event(tmp_path, source)
    output = tmp_path / f"entry_model_native_calibrated_{OUTPUT_STAMP}"
    before = _tree_hashes(source)

    assert calibration.main(_args(source, event, output)) == 0

    assert _tree_hashes(source) == before
    assert output.is_dir()
    assert (output / "MASTER_TRANSFORMER_LOCK.json").read_bytes() == (
        source / "MASTER_TRANSFORMER_LOCK.json"
    ).read_bytes()
    assert (output / "model_state_dict.pt").read_bytes() == (
        source / "model_state_dict.pt"
    ).read_bytes()

    source_meta = json.loads((source / "bundle_metadata.json").read_text(encoding="utf-8"))
    output_meta = json.loads((output / "bundle_metadata.json").read_text(encoding="utf-8"))
    assert "direction_calibration" not in source_meta
    assert not {
        "feature_meta_path",
        "seq_scaler_path",
        "snap_scaler_path",
    } & output_meta.keys()
    assert output_meta["direction_calibration"]["version"] == (
        calibration.DIRECTION_CALIBRATION_VERSION
    )
    assert output_meta["direction_calibration"]["fitted_on_split"] == "val"
    assert output_meta["direction_calibration"]["argmax_preserving"] is True
    assert output_meta["direction_calibration"]["tie_policy"] == "fail_closed"
    assert "bias" not in output_meta["direction_calibration"]
    assert (
        output_meta["model_native_training_objective"]
        == source_meta["model_native_training_objective"]
    )

    evidence_path = output / f"{calibration.CALIBRATION_EVENT_PREFIX}{OUTPUT_STAMP}.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["decision"] == "PASS"
    assert evidence["output_bundle"]["lock_and_state_unchanged"] is True
    assert evidence["output_bundle"]["training_objective_unchanged"] is True
    assert evidence["source_bundle"]["artifact_sha256"]["bundle_metadata.json"] == (
        sha256_file(source / "bundle_metadata.json")
    )
    assert evidence["output_bundle"]["artifact_sha256"]["bundle_metadata.json"] == (
        sha256_file(output / "bundle_metadata.json")
    )
    assert evidence["predictions"]["sha256"] == sha256_file(event["predictions"])
    assert evidence["prediction_report"]["sha256"] == sha256_file(event["report"])
    assert evidence["metrics"]["nll_after"] < evidence["metrics"]["nll_before"]
    assert evidence["metrics"]["raw_calibrated_argmax_identical"] is True
    assert evidence["metrics"]["winner_rows_by_class"] == {
        "FLAT": 40,
        "LONG": 40,
        "SHORT": 40,
    }

    output_lock = json.loads(
        (output / "MASTER_TRANSFORMER_LOCK.json").read_text(encoding="utf-8")
    )
    stale_meta = json.loads(json.dumps(output_meta))
    stale_meta["direction_calibration"]["version"] = (
        "entry_model_native_direction_calibration_v1"
    )
    with pytest.raises(RuntimeError, match="enabled/version mismatch"):
        _require_exact_model_native_bundle_metadata(stale_meta, output_lock)

    biased_meta = json.loads(json.dumps(output_meta))
    biased_meta["direction_calibration"]["bias"] = [0.1, -0.1, 0.0]
    with pytest.raises(RuntimeError, match="class bias/remap is forbidden"):
        _require_exact_model_native_bundle_metadata(biased_meta, output_lock)


def test_path_execute_uses_the_same_immutable_contract(tmp_path: Path) -> None:
    source = _source_bundle(tmp_path)
    event = _prediction_event(tmp_path, source)
    output = tmp_path / f"entry_model_native_path_calibrated_{OUTPUT_STAMP}"
    before = _tree_hashes(source)

    assert calibration.main(_args(source, event, output, head="path")) == 0

    assert _tree_hashes(source) == before
    metadata = json.loads((output / "bundle_metadata.json").read_text(encoding="utf-8"))
    assert metadata["path_calibration"]["version"] == calibration.PATH_CALIBRATION_VERSION
    evidence = json.loads(
        (
            output / f"{calibration.CALIBRATION_EVENT_PREFIX}{OUTPUT_STAMP}.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["metrics"]["path_quality_mse_after"] < evidence["metrics"][
        "path_quality_mse_before"
    ]
    assert evidence["metrics"]["bad_path_bce_after"] < evidence["metrics"][
        "bad_path_bce_before"
    ]
    assert evidence["metrics"]["path_quality_support_definition"] == (
        "y_tradable==1"
    )
    assert evidence["metrics"]["path_quality_support_rows"] == 100
    assert evidence["metrics"]["bad_path_support_definition"] == (
        "long_short_union"
    )
    assert evidence["metrics"]["bad_path_support_rows"] == 120


def test_sequential_direction_and_path_calibration_retain_both_events(
    tmp_path: Path,
) -> None:
    source = _source_bundle(tmp_path)
    event = _prediction_event(tmp_path, source)
    direction_bundle = tmp_path / f"entry_model_native_calibrated_{OUTPUT_STAMP}"
    final_bundle = tmp_path / (
        f"entry_model_native_calibrated_{SECOND_OUTPUT_STAMP}"
    )

    assert calibration.main(_args(source, event, direction_bundle)) == 0
    path_event = _prediction_event(
        tmp_path,
        direction_bundle,
        stamp=SECOND_PREDICTION_STAMP,
    )
    assert calibration.main(
        _args(direction_bundle, path_event, final_bundle, head="path")
    ) == 0

    metadata = json.loads(
        (final_bundle / "bundle_metadata.json").read_text(encoding="utf-8")
    )
    commit = json.loads(
        (
            final_bundle / "ENTRY_MODEL_NATIVE_BUNDLE_COMMIT.json"
        ).read_text(encoding="utf-8")
    )
    event_names = sorted(
        name
        for name in commit["artifact_names"]
        if name.startswith(calibration.CALIBRATION_EVENT_PREFIX)
    )
    assert event_names == [
        f"{calibration.CALIBRATION_EVENT_PREFIX}{OUTPUT_STAMP}.json",
        f"{calibration.CALIBRATION_EVENT_PREFIX}{SECOND_OUTPUT_STAMP}.json",
    ]
    assert metadata["direction_calibration"]["fitted_on_split"] == "val"
    assert metadata["path_calibration"]["fitted_on_split"] == "val"


def test_path_fit_is_invariant_to_rows_outside_training_support() -> None:
    frame = _prediction_frame()
    frame.loc[100:, "y_selector_short_mask"] = 0
    baseline, _ = calibration._fit_path(
        frame,
        min_fit_rows=90,
        selector_mode="long_short_union",
    )

    changed = frame.copy()
    changed.loc[100:, "path_quality_pred"] = np.linspace(1000.0, 2000.0, 20)
    changed.loc[100:, "path_quality_bps"] = np.linspace(-5000.0, 9000.0, 20)
    changed.loc[100:, "bad_path_prob"] = np.linspace(0.01, 0.99, 20)
    changed.loc[100:, "y_bad_path"] = 1 - changed.loc[100:, "y_bad_path"]
    observed, _ = calibration._fit_path(
        changed,
        min_fit_rows=90,
        selector_mode="long_short_union",
    )

    assert observed == baseline


def test_output_collision_fails_without_mutating_either_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(tmp_path)
    event = _prediction_event(tmp_path, source)
    output = tmp_path / f"entry_model_native_calibrated_{OUTPUT_STAMP}"
    assert calibration.main(_args(source, event, output)) == 0
    source_before = _tree_hashes(source)
    output_before = _tree_hashes(output)

    assert calibration.main(_args(source, event, output)) == 2

    assert "already exists" in capsys.readouterr().err
    assert _tree_hashes(source) == source_before
    assert _tree_hashes(output) == output_before


def test_existing_selected_calibration_key_is_rejected_even_when_null(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(
        tmp_path,
        direction_calibration_present=True,
    )
    event = _prediction_event(tmp_path, source)
    output = tmp_path / f"entry_model_native_calibrated_{OUTPUT_STAMP}"

    assert calibration.main(_args(source, event, output, execute=False)) == 2

    assert "re-fit is forbidden" in capsys.readouterr().err
    assert not output.exists()


def test_untimestamped_prediction_path_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _source_bundle(tmp_path)
    event = _prediction_event(tmp_path, source)
    mirror = event["reports"] / "selective_edge_predictions.parquet"
    mirror.write_bytes(event["predictions"].read_bytes())
    output = tmp_path / f"entry_model_native_calibrated_{OUTPUT_STAMP}"
    args = _args(source, event, output, execute=False)
    args[args.index(str(event["predictions"]))] = str(mirror)

    assert calibration.main(args) == 2

    assert "not a timestamped authoritative predictions path" in capsys.readouterr().err
    assert not output.exists()


def test_mutable_bundle_alias_is_rejected_before_read(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / f"entry_model_native_latest_{SOURCE_STAMP}"
    source.mkdir()
    output = tmp_path / f"entry_model_native_calibrated_{OUTPUT_STAMP}"
    parser = calibration.build_arg_parser()
    args = parser.parse_args(
        [
            "--source-bundle-dir",
            str(source),
            "--out-bundle-dir",
            str(output),
            "--predictions-parquet",
            str(tmp_path / f"selective_edge_predictions_{PREDICTION_STAMP}.parquet"),
            "--prediction-report-json",
            str(tmp_path / f"ENTRY_CANDIDATE_SELECTIVE_EDGE_{PREDICTION_STAMP}.json"),
            "--predictions-sha256",
            "a" * 64,
            "--dataset-dir",
            str(tmp_path),
            "--model",
            "candidate",
            "--heads",
            "direction",
            "--fit-split",
            "val",
            "--run-id",
            "TEST",
            "--min-fit-rows",
            "10",
            "--dry-run",
        ]
    )

    with pytest.raises(RuntimeError, match="mutable alias"):
        calibration.run(args)


def test_cli_has_no_model_head_split_or_environment_defaults() -> None:
    parser = calibration.build_arg_parser()
    actions = {action.dest: action for action in parser._actions}
    for name in (
        "model",
        "heads",
        "fit_split",
        "min_fit_rows",
        "predictions_sha256",
    ):
        assert actions[name].required is True
    assert "direction_odds_cap" not in actions
    source = Path(calibration.__file__).read_text(encoding="utf-8")
    assert "os.environ" not in source
    assert "foundation" not in source.lower()
    assert "smart520" not in source.lower()
    assert "default=\"candidate\"" not in source
    assert "default=\"val\"" not in source
    assert "default=\"direction\"" not in source


def test_direction_fit_rejects_missing_classes_and_malformed_probabilities() -> None:
    frame = _prediction_frame(12)
    missing = frame[frame["y_direction"] != 2]
    with pytest.raises(RuntimeError, match="missing classes"):
        calibration._fit_direction(missing)

    malformed = frame.copy()
    malformed.loc[0, "p_long"] = 0.9
    with pytest.raises(RuntimeError, match="do not sum to one"):
        calibration._fit_direction(malformed)

    nonfinite = frame.copy()
    nonfinite.loc[0, "p_short"] = np.nan
    with pytest.raises(RuntimeError, match="non-finite or malformed"):
        calibration._fit_direction(nonfinite)


def test_direction_fit_preserves_all_three_argmax_classes_and_rejects_ties() -> None:
    frame = _prediction_frame(12)
    fitted, metrics = calibration._fit_direction(frame)
    raw = np.stack(frame["direction_logits"].to_numpy())
    calibrated = raw / float(fitted["temperature"])

    assert np.array_equal(np.argmax(raw, axis=1), np.argmax(calibrated, axis=1))
    assert set(np.argmax(raw, axis=1).tolist()) == {0, 1, 2}
    assert metrics["raw_calibrated_argmax_identical"] is True
    assert "bias" not in fitted

    tied = frame.copy()
    tied_logits = np.asarray([1.0, 1.0, 0.0], dtype=np.float64)
    tied_exp = np.exp(tied_logits - tied_logits.max())
    tied_probabilities = tied_exp / tied_exp.sum()
    tied.at[0, "direction_logits"] = tied_logits.tolist()
    tied.loc[0, list(calibration.CLASS_COLUMNS)] = tied_probabilities
    with pytest.raises(RuntimeError, match="ties fail closed"):
        calibration._fit_direction(tied)


def test_direction_fit_rejects_unsuccessful_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        calibration,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            message="synthetic optimizer failure",
            x=np.zeros(1),
            nit=0,
        ),
    )

    with pytest.raises(RuntimeError, match="optimizer failed"):
        calibration._fit_direction(_prediction_frame(12))


def test_cli_requires_exactly_one_execution_mode() -> None:
    parser = calibration.build_arg_parser()
    base = [
        "--source-bundle-dir",
        f"/tmp/source_{SOURCE_STAMP}",
        "--out-bundle-dir",
        f"/tmp/output_{OUTPUT_STAMP}",
        "--predictions-parquet",
        f"/tmp/selective_edge_predictions_{PREDICTION_STAMP}.parquet",
        "--prediction-report-json",
        f"/tmp/ENTRY_CANDIDATE_SELECTIVE_EDGE_{PREDICTION_STAMP}.json",
        "--predictions-sha256",
        "a" * 64,
        "--dataset-dir",
        "/tmp",
        "--model",
        "candidate",
        "--heads",
        "direction",
        "--fit-split",
        "val",
        "--run-id",
        "TEST",
        "--min-fit-rows",
        "10",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(base)
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--dry-run", "--execute"])


@pytest.mark.parametrize("split", ["train", "calibration", "test"])
def test_cli_forbids_every_non_val_calibration_split(split: str) -> None:
    parser = calibration.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--source-bundle-dir",
                f"/tmp/source_{SOURCE_STAMP}",
                "--out-bundle-dir",
                f"/tmp/output_{OUTPUT_STAMP}",
                "--predictions-parquet",
                f"/tmp/selective_edge_predictions_{PREDICTION_STAMP}.parquet",
                "--prediction-report-json",
                f"/tmp/ENTRY_CANDIDATE_SELECTIVE_EDGE_{PREDICTION_STAMP}.json",
                "--predictions-sha256",
                "a" * 64,
                "--dataset-dir",
                "/tmp",
                "--model",
                "candidate",
                "--heads",
                "direction",
                "--fit-split",
                split,
                "--run-id",
                "TEST",
                "--min-fit-rows",
                "10",
                "--dry-run",
            ]
        )
