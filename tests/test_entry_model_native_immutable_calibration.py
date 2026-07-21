from __future__ import annotations

import hashlib
import json
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
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
    model_direction_decision_contract_metadata,
)
from gx1.models.entry_v10.entry_v10_bundle import (
    _ENTRY_HEAD_STATE_KEYS,
    _MODEL_NATIVE_METADATA_ONLY_COMPONENTS,
    _MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS,
    _MODEL_NATIVE_REQUIRED_SPECIALISTS,
    _MODEL_NATIVE_ZERO_INIT_COMPONENT_GROUPS,
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


SOURCE_STAMP = "20260716T100000123456Z"
PREDICTION_STAMP = "20260716T110000123456Z"
OUTPUT_STAMP = "20260716T120000123456Z"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _source_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / f"entry_model_native_bundle_{SOURCE_STAMP}"
    bundle.mkdir()
    selected = canonical_model_native_selected_fields(
        remainder_prefix="session_regime.immutable_calibration_fixture"
    )
    signal_contract = model_native_signal_contract_metadata(selected)
    objective = training_objective_contract_metadata(
        {name: 1.0 for name in REQUIRED_POSITIVE_LOSS_WEIGHTS}
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
    state_path = bundle / "model_state_dict.pt"
    torch.save(state, state_path)
    state_sha = sha256_file(state_path)

    shared = {
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": MODEL_NATIVE_SEQ_LEN,
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
        "aux_head_target_contract": model_native_aux_target_contract_metadata(),
    }
    direction_contract = model_direction_decision_contract_metadata()
    lock = {
        **shared,
        "neutral_xgb_bridge": False,
        "xgb_bridge_source": None,
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
        "neutral_xgb_bridge": False,
        "xgb_bridge_source": None,
        "anchored_entry_enabled": False,
        "anchor_source": None,
        "anchor_gate": {"enabled": False},
        "direction_decision_contract": direction_contract,
        "train_recipe": {"active_heads": sorted(_MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS)},
        "multi_tf": {
            "enabled": True,
            "v2_mode": True,
            "m5_seq_dim": 5,
            "m5_seq_len": 96,
            "m15_seq_dim": 5,
            "m15_seq_len": 96,
            "h1_seq_dim": 5,
            "h1_seq_len": 96,
            "h4_seq_dim": 5,
            "h4_seq_len": 96,
            "d1_seq_dim": 5,
            "d1_seq_len": 96,
            "multi_tf_scale": 0.5,
            "closed_bar_target_availability": True,
            "target_availability_shift_minutes": 5.0,
        },
        "enable_pos_enc": True,
        "enable_regime_film": True,
        "tf_input_scale": {
            "enabled": True,
            "init": {name: 1.0 for name in ("m5", "m15", "h1", "h4", "d1")},
        },
        "hierarchical_entry_heads": {"enabled": True},
        "trendline_rail_head": {
            "enabled": True,
            "output_dim": 6,
            "hand_written_direction_pressure": False,
            "direction_mapping": "direct_learned_evidence_fusion",
        },
        "model_native_state_contract": {"decision": "PASS"},
        "specialist_fusion": {
            "enabled": True,
            "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "input_indices": specialist_indices,
            "trainable_specialists": list(_MODEL_NATIVE_REQUIRED_SPECIALISTS),
            "num_layers": 1,
            "fusion_scale": 0.25,
            "cross_family_fusion_scale": 0.25,
        },
        "feature_meta_path": "feature_meta.json",
    }
    _write_json(bundle / "MASTER_TRANSFORMER_LOCK.json", lock)
    _write_json(bundle / "bundle_metadata.json", metadata)
    _write_json(bundle / "feature_meta.json", {"schema_version": "synthetic_feature_meta_v1"})
    (bundle / "unrelated_training_checkpoint.pt").write_bytes(b"must not be copied")
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
            **turning_point_prediction_columns(rows),
            **offline_rl_prediction_columns(rows),
        }
    )


def _prediction_event(tmp_path: Path, bundle: Path) -> dict[str, Path]:
    dataset = tmp_path / "entry_model_native_dataset"
    reports = tmp_path / "prediction_events"
    dataset.mkdir()
    reports.mkdir()
    predictions = reports / f"selective_edge_predictions_{PREDICTION_STAMP}.parquet"
    _prediction_frame().to_parquet(predictions, index=False)
    metadata = json.loads((bundle / "bundle_metadata.json").read_text(encoding="utf-8"))
    evidence = build_prediction_evidence_declaration(
        predictions_path=predictions,
        bundle_dir=bundle,
        bundle_metadata=metadata,
        requested_splits=["val"],
    )
    report_path = reports / f"ENTRY_CANDIDATE_SELECTIVE_EDGE_{PREDICTION_STAMP}.json"
    report = {
        "schema_version": "entry_candidate_selective_edge_v1",
        "created_utc": "2026-07-16T11:00:00.123456+00:00",
        "decision": "PASS",
        "failures": [],
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
    if head == "direction":
        values.extend(["--direction-odds-cap", "2.0"])
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
    assert not (output / "unrelated_training_checkpoint.pt").exists()
    assert (output / "feature_meta.json").is_file()
    assert (output / "MASTER_TRANSFORMER_LOCK.json").read_bytes() == (
        source / "MASTER_TRANSFORMER_LOCK.json"
    ).read_bytes()
    assert (output / "model_state_dict.pt").read_bytes() == (
        source / "model_state_dict.pt"
    ).read_bytes()

    source_meta = json.loads((source / "bundle_metadata.json").read_text(encoding="utf-8"))
    output_meta = json.loads((output / "bundle_metadata.json").read_text(encoding="utf-8"))
    assert "direction_calibration" not in source_meta
    assert output_meta["direction_calibration"]["version"] == (
        calibration.DIRECTION_CALIBRATION_VERSION
    )
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
    source = _source_bundle(tmp_path)
    metadata_path = source / "bundle_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["direction_calibration"] = None
    _write_json(metadata_path, metadata)
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
            "--direction-odds-cap",
            "2.0",
            "--dry-run",
        ]
    )

    with pytest.raises(RuntimeError, match="mutable alias"):
        calibration.run(args)


def test_cli_has_no_model_head_split_or_environment_defaults() -> None:
    parser = calibration.build_arg_parser()
    actions = {action.dest: action for action in parser._actions}
    for name in ("model", "heads", "fit_split", "min_fit_rows"):
        assert actions[name].required is True
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
        calibration._fit_direction(missing, odds_cap=2.0)

    malformed = frame.copy()
    malformed.loc[0, "p_long"] = 0.9
    with pytest.raises(RuntimeError, match="do not sum to one"):
        calibration._fit_direction(malformed, odds_cap=2.0)

    nonfinite = frame.copy()
    nonfinite.loc[0, "p_short"] = np.nan
    with pytest.raises(RuntimeError, match="non-finite or malformed"):
        calibration._fit_direction(nonfinite, odds_cap=2.0)


def test_direction_fit_rejects_unsuccessful_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        calibration,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            message="synthetic optimizer failure",
            x=np.zeros(3),
            nit=0,
        ),
    )

    with pytest.raises(RuntimeError, match="optimizer failed"):
        calibration._fit_direction(_prediction_frame(12), odds_cap=2.0)


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
        "--direction-odds-cap",
        "2.0",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(base)
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--dry-run", "--execute"])


@pytest.mark.parametrize("split", ["train", "test"])
def test_cli_forbids_train_and_test_calibration_splits(split: str) -> None:
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
                "--direction-odds-cap",
                "2.0",
                "--dry-run",
            ]
        )
