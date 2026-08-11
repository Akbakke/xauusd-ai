import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest
import torch

import gx1.scripts.evaluate_entry_candidate_selective_edge_v1 as selective_edge
from gx1.contracts.entry_model_native_calibration_v1 import (
    DIRECTION_CALIBRATION_TIE_POLICY,
    DIRECTION_CALIBRATION_TRANSFORM,
    DIRECTION_CALIBRATION_VERSION,
    PATH_CALIBRATION_VERSION,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_CTX_CONT_INDEX_BY_NAME,
    MODEL_NATIVE_SIGNAL_DIM,
    model_native_signal_contract_metadata,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    required_training_specialists_for_mode,
    specialist_model_contract_for_mode,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
)
from gx1.scripts.evaluate_entry_candidate_selective_edge_v1 import (
    _canonical_live_decision_evidence,
    _concatenate_evidence_chunks,
    _dataset_model_native_contract,
    _derived_serve_parity_outputs,
    _entry_trend_regime_ids_from_ctx_cont,
    _iter_split_chunks,
    _normalize_contract_mode,
    _require_evaluation_lineage,
    _require_model_direction_ssot,
    _require_runtime_authoritative_calibrations,
    _require_selective_edge_stage_split,
    _require_stage_split_bindings,
    _selection_sort_column,
    _specialist_contract_snapshot,
    build_metric_rows,
    build_parser,
    SELECTIVE_EDGE_MAX_STREAM_CHUNK_ROWS,
)
from gx1.execution.v12_smart_entry_live import _direction_ssot_from_logits
from tests.model_native_signal_support import canonical_model_native_selected_fields
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
)


def _signal_contract() -> dict:
    return model_native_signal_contract_metadata(
        canonical_model_native_selected_fields(
            remainder_prefix="session_regime.selective_edge_fixture"
        )
    )


def _bundle_metadata() -> dict:
    specialists = required_training_specialists_for_mode(
        MODEL_NATIVE_CONTRACT_MODE
    )
    return {
        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "model_native_signal_contract": _signal_contract(),
        "specialist_fusion": {
            "enabled": True,
            "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "input_indices": {name: [index] for index, name in enumerate(specialists)},
            "specialist_model_contract": specialist_model_contract_for_mode(
                MODEL_NATIVE_CONTRACT_MODE
            ),
            "specialist_model_contract_valid": True,
            "specialist_model_contract_set_exact": True,
            "specialist_model_contract_owned_objectives_match": True,
            "specialist_model_contract_signal_families_match": True,
            "specialist_model_contract_support_heads_match": True,
            "specialist_model_contract_model_roles_match": True,
        },
    }


def _full_candidate_lineage() -> dict:
    return {
        "training_profile": "candidate",
        "requested_subsample_rows": 0,
        "physical_train_rows": 369_303,
        "effective_train_rows": 369_303,
    }


def _calibration_metadata(
    tmp_path: Path,
    *,
    head: str,
    fitted_on_split: str = "val",
) -> dict:
    stamp = "20260101T000000000000Z"
    metadata = {
        "enabled": True,
        "version": (
            DIRECTION_CALIBRATION_VERSION
            if head == "direction"
            else PATH_CALIBRATION_VERSION
        ),
        "fitted_at_utc": "2026-01-01T00:00:00+00:00",
        "fitted_on_split": fitted_on_split,
        "fitted_rows": 100,
        "model": "candidate",
        "min_fit_rows": 90,
        "run_id": "RUNTIME_TEST_2026",
        "source_bundle_dir": str((tmp_path / "source_bundle").resolve()),
        "source_bundle_metadata_sha256": "a" * 64,
        "predictions_path": str(
            (tmp_path / f"selective_edge_predictions_{stamp}.parquet").resolve()
        ),
        "predictions_sha256": "b" * 64,
        "prediction_report_path": str(
            (
                tmp_path
                / f"ENTRY_CANDIDATE_SELECTIVE_EDGE_{stamp}.json"
            ).resolve()
        ),
        "prediction_report_sha256": "c" * 64,
    }
    if head == "direction":
        metadata.update(
            {
                "temperature": 1.0,
                "class_order": ["LONG", "SHORT", "FLAT"],
                "transform": DIRECTION_CALIBRATION_TRANSFORM,
                "argmax_preserving": True,
                "tie_policy": DIRECTION_CALIBRATION_TIE_POLICY,
            }
        )
    else:
        metadata.update(
            {
                "path_quality_scale": 1.0,
                "path_quality_shift": 0.0,
                "bad_path_temperature": 1.0,
                "bad_path_bias": 0.0,
            }
        )
    return metadata


def _runtime_bundle_metadata(tmp_path: Path) -> dict:
    return {
        "run_lineage": _full_candidate_lineage(),
        "direction_calibration": _calibration_metadata(
            tmp_path,
            head="direction",
        ),
        "path_calibration": _calibration_metadata(
            tmp_path,
            head="path",
        ),
    }


def _runtime_run_args(tmp_path: Path) -> SimpleNamespace:
    bundle_dir = tmp_path / "bundle"
    dataset_dir = tmp_path / "dataset"
    mtf_cache_dir = tmp_path / "mtf_cache"
    bundle_dir.mkdir()
    dataset_dir.mkdir()
    mtf_cache_dir.mkdir()
    m5_prebuilt = tmp_path / "m5.parquet"
    m5_prebuilt.write_bytes(b"m5")
    return SimpleNamespace(
        bundle_dir=str(bundle_dir.resolve()),
        dataset_dir=str(dataset_dir.resolve()),
        splits="test",
        evidence_stage="runtime_authoritative",
        val_manifest_json=None,
        val_manifest_sha256=None,
        val_parquet=None,
        val_parquet_sha256=None,
        test_manifest_json=str(
            (dataset_dir / "native_test.manifest.json").resolve()
        ),
        test_manifest_sha256="d" * 64,
        test_parquet=str((dataset_dir / "native_test.parquet").resolve()),
        test_parquet_sha256="e" * 64,
        device="cpu",
        batch_size=2,
        stream_chunk_rows=4,
        m5_prebuilt_path=str(m5_prebuilt.resolve()),
        multi_tf_cache_dir=str(mtf_cache_dir.resolve()),
        out_dir=str((tmp_path / "out").resolve()),
        quiet=True,
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["val"] * 6 + ["test"] * 6,
            "model": ["candidate"] * 12,
            "time": pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC"),
            "y_direction": [0, 1, 0, 2, 1, 0, 0, 1, 2, 1, 0, 1],
            "pred_direction": [0, 1, 0, 2, 1, 0, 0, 1, 2, 1, 0, 1],
            "trade_side": [0, 1, 0, 2, 1, 0, 0, 1, 2, 1, 0, 1],
            "side": ["LONG", "SHORT", "LONG", "FLAT", "SHORT", "LONG"] * 2,
            "session": ["EU", "EU", "US", "ASIA", "US", "OVERLAP"] * 2,
            "vol_regime": ["1", "1", "2", "0", "2", "1"] * 2,
            "selection_score_mode": [MODEL_DIRECTION_SELECTION_MODE] * 12,
            "selection_score": [
                0.8,
                0.8,
                0.7,
                0.8,
                0.8,
                0.7,
                0.8,
                0.8,
                0.7,
                0.8,
                0.8,
                0.7,
            ],
            "edge_score": [
                0.9,
                0.8,
                0.7,
                -0.2,
                0.6,
                0.5,
                0.95,
                0.85,
                -0.1,
                0.75,
                0.65,
                0.55,
            ],
            "p_long": [0.8, 0.1, 0.7, 0.1, 0.1, 0.7] * 2,
            "p_short": [0.1, 0.8, 0.1, 0.1, 0.8, 0.1] * 2,
            "p_flat": [0.1, 0.1, 0.2, 0.8, 0.1, 0.2] * 2,
            "pnl_proxy_bps": [8.0, 7.0, 6.0, 0.0, 5.0, 4.0] * 2,
            "bad_path_prob": [0.1] * 12,
            "path_quality_pred": [1.0] * 12,
        }
    )


def test_contract_mode_accepts_only_exact_model_native_seq513() -> None:
    assert _normalize_contract_mode(None) == MODEL_NATIVE_CONTRACT_MODE
    assert _normalize_contract_mode(MODEL_NATIVE_CONTRACT_MODE) == MODEL_NATIVE_CONTRACT_MODE
    for retired in ("foundation_seq146", "challenger_seq215", "smart_seq520_candidate"):
        with pytest.raises(RuntimeError, match="retired"):
            _normalize_contract_mode(retired)


def test_specialist_snapshot_requires_all_eight_model_native_specialists() -> None:
    snapshot = _specialist_contract_snapshot(
        _bundle_metadata(), MODEL_NATIVE_CONTRACT_MODE
    )
    assert snapshot["expected_signal_dim"] == MODEL_NATIVE_SIGNAL_DIM
    assert snapshot["bundle_seq_input_dim"] == MODEL_NATIVE_SIGNAL_DIM
    assert snapshot["bundle_snap_input_dim"] == MODEL_NATIVE_SIGNAL_DIM
    assert snapshot["required_specialists_exact"] is True
    assert snapshot["chart_geometry_present"] is True
    assert snapshot["price_action_candle_present"] is True
    assert snapshot["failures"] == []


def test_direction_ssot_is_final_argmax_and_rejects_legacy_outputs() -> None:
    direction = torch.tensor(
        [[3.0, 1.0, 2.0], [0.0, 4.0, 1.0], [0.0, 1.0, 5.0]]
    )
    pair = torch.stack((direction[:, :2].amax(dim=1), direction[:, 2]), dim=1)
    observed_direction, observed_pair = _require_model_direction_ssot(
        {
            "direction_logits": direction,
            "public_trade_flat_decision_logits": pair,
        }
    )
    assert torch.equal(torch.argmax(observed_direction, dim=1), torch.tensor([0, 1, 2]))
    assert torch.equal(torch.argmax(observed_pair, dim=1), torch.tensor([0, 0, 1]))

    with pytest.raises(RuntimeError, match="forbidden legacy"):
        _require_model_direction_ssot(
            {
                "direction_logits": direction,
                "public_trade_flat_decision_logits": pair,
                "anchor_logits": direction,
            }
        )

    tied_direction = torch.tensor([[2.0, 2.0, 0.0]])
    tied_pair = torch.stack(
        (tied_direction[:, :2].amax(dim=1), tied_direction[:, 2]),
        dim=1,
    )
    with pytest.raises(RuntimeError, match="no unique top class"):
        _require_model_direction_ssot(
            {
                "direction_logits": tied_direction,
                "public_trade_flat_decision_logits": tied_pair,
            }
        )
    with pytest.raises(RuntimeError, match="no unique top class"):
        _direction_ssot_from_logits(
            tied_direction[0].numpy(),
            tied_pair[0].numpy(),
            context="test",
        )


def test_candidate_decision_evidence_has_numeric_live_parity() -> None:
    direction = torch.tensor(
        [[3.0, 1.0, 2.0], [0.25, 4.0, 1.0], [0.0, 1.0, 5.0]],
        dtype=torch.float32,
    )
    pair = torch.stack((direction[:, :2].amax(dim=1), direction[:, 2]), dim=1)
    evidence = _canonical_live_decision_evidence(
        {
            "direction_logits": direction,
            "public_trade_flat_decision_logits": pair,
        }
    )
    for row in range(direction.shape[0]):
        live = _direction_ssot_from_logits(
            direction[row].numpy(),
            pair[row].numpy(),
            context="test",
        )
        assert np.allclose(
            evidence["direction_probs"][row],
            live["direction_probs"],
            rtol=1e-6,
            atol=1e-7,
        )
        assert np.allclose(
            evidence["public_trade_flat_decision_probs"][row],
            live["public_trade_flat_decision_probs"],
            rtol=1e-6,
            atol=1e-7,
        )
        direction_index = int(live["model_direction_index"])
        assert evidence["model_direction_index"][row] == direction_index
        assert evidence["selection_score"][row] == pytest.approx(
            float(live["direction_probs"][direction_index]),
            rel=1e-6,
            abs=1e-7,
        )
        assert evidence["p_trade"][row] == pytest.approx(
            float(live["public_trade_flat_decision_probs"][0]),
            rel=1e-6,
            abs=1e-7,
        )
    with pytest.raises(RuntimeError, match="do not match final direction logits"):
        _require_model_direction_ssot(
            {
                "direction_logits": direction[:1],
                "public_trade_flat_decision_logits": torch.tensor([[0.0, 9.0]]),
            }
        )


def test_derived_serve_parity_outputs_are_exact_and_fail_closed() -> None:
    path_log_var = torch.tensor(
        [[0.0], [float(np.log(4.0))]], dtype=torch.float32
    )
    mtf_logits = torch.tensor(
        [[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]], dtype=torch.float32
    )
    observed = _derived_serve_parity_outputs(
        {
            "path_quality_log_var": path_log_var,
            "mtf_dir_logits": mtf_logits,
        },
        path_quality_scale=3.0,
    )

    assert set(observed) == {"path_quality_std", "mtf_dir_probs"}
    assert observed["path_quality_std"].shape == (2,)
    assert observed["path_quality_std"].dtype == np.float32
    assert np.allclose(observed["path_quality_std"], [3.0, 6.0])
    assert observed["mtf_dir_probs"].shape == (2, 3)
    assert observed["mtf_dir_probs"].dtype == np.float32
    assert np.allclose(
        observed["mtf_dir_probs"],
        torch.softmax(mtf_logits, dim=1).numpy(),
        rtol=1e-6,
        atol=1e-7,
    )
    assert np.allclose(observed["mtf_dir_probs"].sum(axis=1), 1.0)

    with pytest.raises(RuntimeError, match="path_quality_log_var invalid"):
        _derived_serve_parity_outputs(
            {
                "path_quality_log_var": torch.tensor([[float("nan")]]),
                "mtf_dir_logits": torch.zeros((1, 3)),
            },
            path_quality_scale=3.0,
        )
    with pytest.raises(RuntimeError, match="mtf_dir_logits invalid"):
        _derived_serve_parity_outputs(
            {
                "path_quality_log_var": torch.zeros((1, 1)),
                "mtf_dir_logits": torch.tensor([[0.0, float("inf"), 0.0]]),
            },
            path_quality_scale=3.0,
        )


def test_dataset_contract_requires_exact_34_plus_479_for_one_stage_split(
    tmp_path: Path,
) -> None:
    contract = _signal_contract()
    assert contract["base_signal_dim"] == 34
    assert contract["selected_feature_count"] == MODEL_NATIVE_SELECTED_FEATURE_COUNT
    bindings: dict[str, dict[str, str]] = {}
    for split in ("val",):
        parquet = tmp_path / f"native_{split}.parquet"
        parquet.write_bytes(f"immutable-{split}".encode())
        manifest = tmp_path / f"native_{split}.manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "output_data_path": str(parquet),
                    "extra": {"model_native_signal_contract": contract},
                }
            ),
            encoding="utf-8",
        )
        bindings[split] = {
            "manifest_path": str(manifest),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "parquet_path": str(parquet),
            "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        }

    observed = _dataset_model_native_contract(
        tmp_path,
        ["val"],
        bindings,
    )
    assert observed["contract"] == contract
    assert {row["seq_input_dim"] for row in observed["splits"].values()} == {MODEL_NATIVE_SIGNAL_DIM}

    broken = json.loads(json.dumps(contract))
    broken["bridge_dim"] = 7
    val_manifest = tmp_path / "native_val.manifest.json"
    val_manifest.write_text(
        json.dumps(
            {
                "output_data_path": str(tmp_path / "native_val.parquet"),
                "extra": {"model_native_signal_contract": broken},
            }
        ),
        encoding="utf-8",
    )
    bindings["val"]["manifest_sha256"] = hashlib.sha256(
        val_manifest.read_bytes()
    ).hexdigest()
    with pytest.raises(RuntimeError, match="MODEL_NATIVE_SIGNAL_CONTRACT_INVALID"):
        _dataset_model_native_contract(tmp_path, ["val"], bindings)
    with pytest.raises(RuntimeError, match="one exact stage split"):
        _dataset_model_native_contract(
            tmp_path,
            ["val", "test"],
            bindings,
        )


def test_selection_metrics_never_accept_session_filter_or_score_fallback() -> None:
    frame = _predictions()
    assert _selection_sort_column(frame) == "selection_score"
    rows = build_metric_rows(frame, top_fracs=[0.05, 0.10])
    assert rows
    assert {row["split"] for row in rows} == {"val", "test"}

    with pytest.raises(RuntimeError, match="session exclusion"):
        build_metric_rows(frame, top_fracs=[0.10], exclude_sessions=("ASIA",))
    with pytest.raises(RuntimeError, match="lacks model-native selection_score"):
        _selection_sort_column(frame.drop(columns=["selection_score"]))


def test_parser_requires_explicit_native_artifact_paths_and_rejects_retired_flags() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    required = [
        "--bundle-dir",
        "/tmp/bundle",
        "--dataset-dir",
        "/tmp/dataset",
        "--splits",
        "test",
        "--evidence-stage",
        "runtime_authoritative",
        "--stream-chunk-rows",
        "4096",
        "--test-manifest-json",
        "/tmp/dataset/native_test.manifest.json",
        "--test-manifest-sha256",
        "a" * 64,
        "--test-parquet",
        "/tmp/dataset/native_test.parquet",
        "--test-parquet-sha256",
        "b" * 64,
        "--m5-prebuilt-path",
        "/tmp/m5.parquet",
        "--multi-tf-cache-dir",
        "/tmp/mtf",
        "--out-dir",
        "/tmp/out",
    ]
    args = parser.parse_args(required)
    assert not hasattr(args, "contract_mode")
    assert args.splits == "test"
    assert args.evidence_stage == "runtime_authoritative"
    assert not hasattr(args, "top_fracs")
    assert not hasattr(args, "model_name")
    assert not hasattr(args, "selection_score_mode")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [*required, "--selection-score-mode", MODEL_DIRECTION_SELECTION_MODE]
        )
    with pytest.raises(SystemExit):
        parser.parse_args([*required, "--smart-seq520"])

    pre_calibration_args = parser.parse_args(
        [
            "--bundle-dir",
            "/tmp/bundle",
            "--dataset-dir",
            "/tmp/dataset",
            "--splits",
            "val",
            "--evidence-stage",
            "pre_calibration",
            "--stream-chunk-rows",
            "4096",
            "--val-manifest-json",
            "/tmp/dataset/native_val.manifest.json",
            "--val-manifest-sha256",
            "c" * 64,
            "--val-parquet",
            "/tmp/dataset/native_val.parquet",
            "--val-parquet-sha256",
            "d" * 64,
            "--m5-prebuilt-path",
            "/tmp/m5.parquet",
            "--multi-tf-cache-dir",
            "/tmp/mtf",
            "--out-dir",
            "/tmp/out",
        ]
    )
    assert pre_calibration_args.splits == "val"
    assert pre_calibration_args.evidence_stage == "pre_calibration"
    with pytest.raises(SystemExit):
        parser.parse_args([*required, "--train-parquet", "/tmp/train.parquet"])
    assert "val,test" not in parser.format_help()
    assert "val,test" not in Path(
        "gx1/scripts/evaluate_entry_candidate_selective_edge_v1.py"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("evidence_stage", "split_spec", "expected"),
    (
        ("pre_calibration", "val", ["val"]),
        ("runtime_authoritative", "test", ["test"]),
    ),
)
def test_selective_edge_stage_machine_accepts_only_exact_stage_split(
    evidence_stage: str,
    split_spec: str,
    expected: list[str],
) -> None:
    assert _require_selective_edge_stage_split(
        evidence_stage=evidence_stage,
        split_spec=split_spec,
    ) == expected


@pytest.mark.parametrize(
    ("evidence_stage", "split_spec"),
    (
        ("pre_calibration", "train"),
        ("pre_calibration", "test"),
        ("pre_calibration", "val,test"),
        ("pre_calibration", "val,"),
        ("runtime_authoritative", "train"),
        ("runtime_authoritative", "val"),
        ("runtime_authoritative", "val,test"),
        ("runtime_authoritative", "test,val"),
    ),
)
def test_selective_edge_stage_machine_rejects_cross_stage_and_subsets(
    evidence_stage: str,
    split_spec: str,
) -> None:
    with pytest.raises(RuntimeError, match="STAGE_SPLIT_INVALID"):
        _require_selective_edge_stage_split(
            evidence_stage=evidence_stage,
            split_spec=split_spec,
        )


def test_selective_edge_stage_bindings_reject_cross_stage_artifacts() -> None:
    args = SimpleNamespace(
        val_manifest_json="/tmp/native_val.manifest.json",
        val_manifest_sha256="a" * 64,
        val_parquet="/tmp/native_val.parquet",
        val_parquet_sha256="b" * 64,
        test_manifest_json=None,
        test_manifest_sha256=None,
        test_parquet=None,
        test_parquet_sha256=None,
    )
    assert set(_require_stage_split_bindings(args, splits=["val"])) == {"val"}

    args.test_parquet = "/tmp/native_test.parquet"
    with pytest.raises(RuntimeError, match="forbidden cross-stage"):
        _require_stage_split_bindings(args, splits=["val"])


def test_runtime_authority_requires_both_frozen_val_calibrations(
    tmp_path: Path,
) -> None:
    metadata = _runtime_bundle_metadata(tmp_path)
    observed = _require_runtime_authoritative_calibrations(metadata)
    assert set(observed) == {"direction", "path"}
    assert {row["fitted_on_split"] for row in observed.values()} == {"val"}

    without_path = dict(metadata)
    without_path.pop("path_calibration")
    with pytest.raises(RuntimeError, match="RUNTIME_PATH_CALIBRATION_INVALID"):
        _require_runtime_authoritative_calibrations(without_path)

    wrong_split = dict(metadata)
    wrong_split["direction_calibration"] = _calibration_metadata(
        tmp_path,
        head="direction",
        fitted_on_split="calibration",
    )
    with pytest.raises(
        RuntimeError,
        match="RUNTIME_DIRECTION_CALIBRATION_INVALID",
    ):
        _require_runtime_authoritative_calibrations(wrong_split)


def test_runtime_bundle_and_calibrations_are_proven_before_test_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _runtime_run_args(tmp_path)
    bundle = SimpleNamespace(metadata=_runtime_bundle_metadata(tmp_path))
    events: list[str] = []
    original_validator = selective_edge.require_model_native_calibration_metadata

    def strict_load(**_kwargs):
        events.append("strict_bundle")
        return bundle

    def validate_calibration(value, *, head, context):
        events.append(head)
        return original_validator(value, head=head, context=context)

    class TestReadReached(RuntimeError):
        pass

    def read_test(*_args, **_kwargs):
        events.append("test_read")
        raise TestReadReached

    monkeypatch.setattr(selective_edge, "load_entry_v10_ctx_bundle", strict_load)
    monkeypatch.setattr(
        selective_edge,
        "require_model_native_calibration_metadata",
        validate_calibration,
    )
    monkeypatch.setattr(
        selective_edge,
        "_dataset_model_native_contract",
        read_test,
    )

    with pytest.raises(TestReadReached):
        selective_edge.run(args)
    assert events == ["strict_bundle", "direction", "path", "test_read"]


def test_runtime_smoke_bundle_fails_before_test_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _runtime_run_args(tmp_path)
    metadata = _runtime_bundle_metadata(tmp_path)
    metadata["run_lineage"] = {
        "training_profile": "smoke",
        "requested_subsample_rows": 10_000,
        "physical_train_rows": 369_303,
        "effective_train_rows": 10_000,
    }
    monkeypatch.setattr(
        selective_edge,
        "load_entry_v10_ctx_bundle",
        lambda **_kwargs: SimpleNamespace(metadata=metadata),
    )
    test_read = False

    def forbidden_test_read(*_args, **_kwargs):
        nonlocal test_read
        test_read = True
        raise AssertionError("TEST must not be read")

    monkeypatch.setattr(
        selective_edge,
        "_dataset_model_native_contract",
        forbidden_test_read,
    )

    with pytest.raises(RuntimeError, match="full-population candidate-profile"):
        selective_edge.run(args)
    assert test_read is False


def test_evaluation_lineage_separates_smoke_from_runtime_authority() -> None:
    smoke = {
        "training_profile": "smoke",
        "requested_subsample_rows": 10_000,
        "physical_train_rows": 369_303,
        "effective_train_rows": 10_000,
    }
    candidate = {
        "training_profile": "candidate",
        "requested_subsample_rows": 0,
        "physical_train_rows": 369_303,
        "effective_train_rows": 369_303,
    }

    _require_evaluation_lineage(smoke, evidence_stage="pre_calibration")
    _require_evaluation_lineage(candidate, evidence_stage="pre_calibration")
    _require_evaluation_lineage(candidate, evidence_stage="runtime_authoritative")

    with pytest.raises(RuntimeError, match="full-population candidate-profile"):
        _require_evaluation_lineage(smoke, evidence_stage="runtime_authoritative")
    with pytest.raises(RuntimeError, match="bounded smoke-profile"):
        _require_evaluation_lineage(
            {**smoke, "effective_train_rows": 369_303},
            evidence_stage="pre_calibration",
        )


def test_evidence_chunk_concatenation_rejects_shape_and_row_drift() -> None:
    combined = _concatenate_evidence_chunks(
        {"position_size_logit": [np.zeros((2, 1)), np.ones((1, 1))]},
        expected_rows=3,
    )
    assert combined["position_size_logit"].shape == (3, 1)

    with pytest.raises(RuntimeError, match="incompatible shapes"):
        _concatenate_evidence_chunks(
            {"position_size_logit": [np.zeros((2, 1)), np.ones(1)]},
            expected_rows=3,
        )
    with pytest.raises(RuntimeError, match="row mismatch"):
        _concatenate_evidence_chunks(
            {"position_size_logit": [np.zeros((2, 1)), np.ones((1, 1))]},
            expected_rows=2,
        )


def test_entry_trend_regime_uses_exact_ctx_cont_d1_source() -> None:
    ctx_cont = np.zeros((3, MODEL_NATIVE_CTX_CONT_DIM), dtype=np.float32)
    ctx_cont[
        :, MODEL_NATIVE_CTX_CONT_INDEX_BY_NAME["D1_dist_from_ema200_atr"]
    ] = (-2.0, 0.0, 2.0)

    assert _entry_trend_regime_ids_from_ctx_cont(ctx_cont).tolist() == [0, 1, 2]

    with pytest.raises(RuntimeError, match="exact context contract"):
        _entry_trend_regime_ids_from_ctx_cont(ctx_cont[:, :-1])
    ctx_cont[
        0, MODEL_NATIVE_CTX_CONT_INDEX_BY_NAME["D1_dist_from_ema200_atr"]
    ] = np.nan
    with pytest.raises(RuntimeError, match="MODEL_NATIVE_CONTEXT_NONFINITE"):
        _entry_trend_regime_ids_from_ctx_cont(ctx_cont)


def test_selective_edge_source_has_no_split_glob_or_imported_split_selector() -> None:
    source = Path(
        "gx1/scripts/evaluate_entry_candidate_selective_edge_v1.py"
    ).read_text(encoding="utf-8")
    assert "_split_file" not in source
    assert 'glob(f"*_{split}' not in source
    assert "source_manifest" not in source
    assert "shutil.copy2(manifest_path, tmp_manifest)" in source


@pytest.mark.parametrize(
    "invalid_rows",
    (0, -1, SELECTIVE_EDGE_MAX_STREAM_CHUNK_ROWS + 1),
)
def test_selective_edge_rejects_unbounded_chunk_sizes(
    tmp_path: Path,
    invalid_rows: int,
) -> None:
    with pytest.raises(RuntimeError, match="STREAM_CHUNK_ROWS_INVALID"):
        list(
            _iter_split_chunks(
                tmp_path / "missing.parquet",
                tmp_path / "missing.json",
                invalid_rows,
            )
        )


def test_selective_edge_rejects_oversized_parquet_row_groups(
    tmp_path: Path,
) -> None:
    path = tmp_path / "split.parquet"
    pd.DataFrame({"value": range(5)}).to_parquet(path, row_group_size=5)

    with pytest.raises(RuntimeError, match="ROW_GROUP_EXCEEDS_MEMORY_CONTRACT"):
        list(_iter_split_chunks(path, tmp_path / "split.json", 4))
