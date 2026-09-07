from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.contracts import entry_exit_production_architecture_v1 as architecture_owner
from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_MTF_CONTEXT_TIMEFRAMES
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS,
)
from gx1.contracts.entry_model_native_joint_task_weighting_v1 import (
    JOINT_TASK_NAMES,
    joint_task_weighting_metadata,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)
from gx1.features.htf_features import HTF_V4_MATRIX_CONTRACT, MULTI_TF_TIMEFRAMES
from gx1.scripts import run_pre_fulltrain_static_preflight_v1 as preflight
from gx1.scripts.run_pre_fulltrain_static_preflight_v1 import (
    PreflightError,
    TEST_BOUNDARY_UTC,
    TRAIN_END_UTC,
    TRAIN_START_UTC,
    VAL_START_UTC,
    inspect_bundle_normalization_binding,
    inspect_dataset_mtf_cache_binding,
    inspect_mtf_cache_test_boundary,
    scan_allowed_split,
)


class _ForbiddenAcceleratorAPI:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"static preflight accessed an accelerator API: {name}")


@pytest.fixture(autouse=True)
def forbid_accelerator_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("cuda", "backends", "accelerator", "xpu", "mps"):
        monkeypatch.setattr(
            preflight.torch, name, _ForbiddenAcceleratorAPI(), raising=False
        )
    monkeypatch.setattr(preflight, "_git_value", lambda *args: None)


def _entry_family_tf_tokens() -> list[str]:
    return [
        f"{timeframe.lower()}:{family}"
        for timeframe in ENTRY_MTF_CONTEXT_TIMEFRAMES
        for family in MODEL_NATIVE_TRAINING_SPECIALISTS
    ]


def _write_split(path: Path, times: list[datetime], horizons: list[int]) -> None:
    pq.write_table(
        pa.table(
            {
                "time": pa.array(times, type=pa.timestamp("ns", tz="UTC")),
                "label_horizon_bars": pa.array(horizons, type=pa.int64()),
                "y_example": pa.array([1.0] * len(times), type=pa.float64()),
            }
        ),
        path,
    )


def test_allowed_val_scan_proves_only_row_bounds(tmp_path: Path) -> None:
    path = tmp_path / "val.parquet"
    _write_split(
        path,
        [
            datetime(2026, 5, 31, 23, 55, tzinfo=timezone.utc),
            datetime(2026, 6, 30, 23, 55, tzinfo=timezone.utc),
        ],
        [12, 96],
    )
    report = scan_allowed_split(
        path,
        label="val",
        nominal_start_utc=VAL_START_UTC,
        nominal_end_utc=TEST_BOUNDARY_UTC,
    )
    assert report["rows"] == 2
    assert report["timestamps_at_or_after_test_boundary"] == 0
    assert report["max_label_horizon_bars_observed"] == 96
    assert report["scope"] == "row_timestamps_and_declared_horizon_domain_only"
    assert report["outcome_containment"] == "UNPROVEN_NOT_INSPECTED"
    assert report["target_values_inspected"] is False


@pytest.mark.parametrize(
    "endpoint_column",
    [
        "fixture_causal_exit_available_at",
        "fixture_aux_outcome_available_at",
        "fixture_exit_state_close_available_at",
    ],
)
@pytest.mark.parametrize(
    "row_time,boundary,endpoint",
    [
        (
            datetime(2026, 6, 30, 23, 55, tzinfo=timezone.utc),
            TEST_BOUNDARY_UTC,
            datetime(2026, 7, 1, 0, 5, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 6, 26, 23, 55, tzinfo=timezone.utc),
            "2026-06-27T00:00:00+00:00",
            datetime(2026, 6, 29, tzinfo=timezone.utc),
        ),
    ],
)
def test_row_scan_cannot_authorize_outcomes_across_boundary_or_closure(
    tmp_path: Path,
    endpoint_column: str,
    row_time: datetime,
    boundary: str,
    endpoint: datetime,
) -> None:
    """Mechanical endpoint fixtures are not production target evidence."""

    path = tmp_path / "val.parquet"
    pq.write_table(
        pa.table(
            {
                "time": pa.array([row_time], type=pa.timestamp("ns", tz="UTC")),
                "label_horizon_bars": pa.array([1], type=pa.int64()),
                endpoint_column: pa.array(
                    [endpoint], type=pa.timestamp("ns", tz="UTC")
                ),
            }
        ),
        path,
    )
    report = scan_allowed_split(
        path,
        label="val",
        nominal_start_utc=VAL_START_UTC,
        nominal_end_utc=boundary,
    )
    assert report["rows"] == 1
    assert report["timestamps_at_or_after_test_boundary"] == 0
    assert report["outcome_containment"] == "UNPROVEN_NOT_INSPECTED"
    assert report["target_values_inspected"] is False


@pytest.mark.parametrize(
    "horizon",
    [
        -1,
        MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS + 1,
        1.5,
        "1",
        True,
        None,
        float("nan"),
        float("inf"),
    ],
)
def test_scan_rejects_invalid_horizon_without_integer_truncation(
    tmp_path: Path, horizon: Any
) -> None:
    path = tmp_path / "val.parquet"
    pq.write_table(
        pa.table(
            {
                "time": pa.array(
                    [datetime(2026, 6, 1, tzinfo=timezone.utc)],
                    type=pa.timestamp("ns", tz="UTC"),
                ),
                "label_horizon_bars": pa.array([horizon]),
            }
        ),
        path,
    )
    with pytest.raises(PreflightError, match="HORIZON_INVALID|SPLIT_INTEGRITY_INVALID"):
        scan_allowed_split(
            path,
            label="val",
            nominal_start_utc=VAL_START_UTC,
            nominal_end_utc=TEST_BOUNDARY_UTC,
        )


def test_scan_rejects_test_timestamp_and_test_like_path(tmp_path: Path) -> None:
    forbidden = tmp_path / "val.parquet"
    _write_split(
        forbidden,
        [datetime(2026, 7, 1, tzinfo=timezone.utc)],
        [1],
    )
    with pytest.raises(PreflightError, match="SPLIT_BOUNDARY_INVALID|SPLIT_INTEGRITY_INVALID"):
        scan_allowed_split(
            forbidden,
            label="val",
            nominal_start_utc=VAL_START_UTC,
            nominal_end_utc=TEST_BOUNDARY_UTC,
        )


def test_mtf_cache_metadata_detects_test_exposure_without_reading_arrays(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "manifest.json"
    cache.write_text(
        """{
          "tfs": {
            "M5": {"last_ts_ns": 1785829800000000000},
            "M15": {"last_ts_ns": 1785828600000000000},
            "H1": {"last_ts_ns": 1785823200000000000},
            "H4": {"last_ts_ns": 1785808800000000000},
            "D1": {"last_ts_ns": 1785708000000000000}
          }
        }""",
        encoding="utf-8",
    )
    report = inspect_mtf_cache_test_boundary(cache)
    assert report["array_bytes_read"] == 0
    assert not report["safe_for_strict_preflight"]
    assert report["test_exposed_timeframes"] == ["D1", "H1", "H4", "M15", "M5"]

    path = tmp_path / "xau_test.parquet"
    _write_split(path, [datetime(2025, 6, 1, tzinfo=timezone.utc)], [1])
    with pytest.raises(PreflightError, match="TEST_PATH_REJECTED"):
        scan_allowed_split(
            path,
            label="val",
            nominal_start_utc=VAL_START_UTC,
            nominal_end_utc=TEST_BOUNDARY_UTC,
        )


def test_static_preflight_uses_the_five_year_recipe_split_boundary() -> None:
    assert TRAIN_START_UTC == "2021-06-01T00:00:00+00:00"
    assert TRAIN_END_UTC == "2026-05-31T23:55:00+00:00"
    assert VAL_START_UTC == TRAIN_END_UTC


def test_dataset_manifest_must_bind_exact_inspected_mtf_cache(tmp_path: Path) -> None:
    manifest = tmp_path / "train.manifest.json"
    manifest.write_text(
        """{
          "extra": {
            "multi_tf_cache_binding": {
              "manifest_sha256": "manifest-new",
              "cache_identity_sha256": "identity-new",
              "m5_prebuilt_source_sha256": "source-new"
            }
          }
        }""",
        encoding="utf-8",
    )
    report = inspect_dataset_mtf_cache_binding(
        manifest,
        expected_manifest_sha256="manifest-new",
        expected_cache_identity_sha256="identity-new",
        expected_source_sha256="source-new",
    )
    assert report["matches_inspected_cache"]
    assert report["array_bytes_read"] == 0
    assert report["test_accessed"] is False

    manifest.write_text(
        """{
          "extra": {
            "multi_tf_cache_binding": {
              "manifest_sha256": "manifest-old",
              "cache_identity_sha256": "identity-old",
              "m5_prebuilt_source_sha256": "source-old"
            }
          }
        }""",
        encoding="utf-8",
    )
    mismatch = inspect_dataset_mtf_cache_binding(
        manifest,
        expected_manifest_sha256="manifest-new",
        expected_cache_identity_sha256="identity-new",
        expected_source_sha256="source-new",
    )
    assert not mismatch["matches_inspected_cache"]
    assert mismatch["mismatched_fields"] == [
        "manifest_sha256",
        "cache_identity_sha256",
        "m5_prebuilt_source_sha256",
    ]


def test_bundle_normalization_must_bind_exact_rebuilt_train_surface(
    tmp_path: Path,
) -> None:
    train_manifest = tmp_path / "train.manifest.json"
    train_manifest.write_text(
        """{
          "extra": {
            "entry_run_id": "PRETEST_RUN",
            "source_frame": {"parquet_sha256": "source-new"}
          }
        }""",
        encoding="utf-8",
    )
    bundle = {
        "run_lineage": {"dataset_run_id": "PRETEST_RUN"},
        "input_normalization": {
            "lineage": {
                "dataset_run_id": "PRETEST_RUN",
                "train_parquet_sha256": "train-new",
                "train_manifest_sha256": hashlib.sha256(
                    train_manifest.read_bytes()
                ).hexdigest(),
                "m5_prebuilt_sha256": "source-new",
                "mtf_cache_manifest_sha256": "cache-new",
            }
        },
    }
    matched = inspect_bundle_normalization_binding(
        bundle,
        train={"sha256": "train-new"},
        train_manifest=train_manifest,
        train_manifest_payload=json.loads(train_manifest.read_text()),
        mtf_cache_manifest_sha256="cache-new",
    )
    assert matched["matches_exact_train_surface"]

    bundle["input_normalization"]["lineage"]["mtf_cache_manifest_sha256"] = "cache-old"
    mismatch = inspect_bundle_normalization_binding(
        bundle,
        train={"sha256": "train-new"},
        train_manifest=train_manifest,
        train_manifest_payload=json.loads(train_manifest.read_text()),
        mtf_cache_manifest_sha256="cache-new",
    )
    assert not mismatch["matches_exact_train_surface"]
    assert mismatch["mismatched_fields"] == ["mtf_cache_manifest_sha256"]


@pytest.fixture
def static_preflight_inputs(tmp_path: Path) -> dict[str, Path]:
    """Build mechanical metadata fixtures without model or training evidence."""

    paths = {
        "train_parquet": tmp_path / "train.parquet",
        "val_parquet": tmp_path / "val.parquet",
        **{
            name: tmp_path / f"{name}.json"
            for name in (
                "train_manifest", "val_manifest", "feature_audit", "target_audit",
                "liveness_audit", "specialist_audit", "execution_audit",
                "bundle_metadata", "multi_tf_cache_manifest",
            )
        },
    }
    _write_split(
        paths["train_parquet"],
        [datetime(2021, 6, 1, tzinfo=timezone.utc)],
        [MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS],
    )
    _write_split(
        paths["val_parquet"],
        [datetime(2026, 6, 1, tzinfo=timezone.utc)],
        [MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS],
    )
    source_sha256 = hashlib.sha256(b"mechanical-source").hexdigest()
    cache_identity = hashlib.sha256(b"mechanical-cache-identity").hexdigest()
    cache = {
        "cache_identity_sha256": cache_identity,
        "m5_prebuilt_source_sha256": source_sha256,
        "tfs": {
            timeframe: {
                "last_ts_ns": int(
                    datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp()
                ) * 1_000_000_000
            }
            for timeframe in MULTI_TF_TIMEFRAMES
        },
    }
    paths["multi_tf_cache_manifest"].write_text(json.dumps(cache), encoding="utf-8")
    cache_manifest_sha256 = hashlib.sha256(
        paths["multi_tf_cache_manifest"].read_bytes()
    ).hexdigest()
    split_manifest = {
        "extra": {
            "entry_run_id": "MECHANICAL_FIXTURE",
            "source_frame": {"parquet_sha256": source_sha256},
            "multi_tf_cache_binding": {
                "manifest_sha256": cache_manifest_sha256,
                "cache_identity_sha256": cache_identity,
                "m5_prebuilt_source_sha256": source_sha256,
            },
        }
    }
    for name in ("train_manifest", "val_manifest"):
        paths[name].write_text(json.dumps(split_manifest), encoding="utf-8")
    for name in (
        "feature_audit", "target_audit", "liveness_audit", "specialist_audit",
        "execution_audit",
    ):
        paths[name].write_text(
            json.dumps({"decision": "PASS", "dataset_sha256": "unrelated-fixture"}),
            encoding="utf-8",
        )
    bundle = {
        "run_lineage": {"dataset_run_id": "MECHANICAL_FIXTURE"},
        "input_normalization": {
            "fit_scope": "train_only",
            "lineage": {
                "dataset_run_id": "MECHANICAL_FIXTURE",
                "train_parquet_sha256": hashlib.sha256(
                    paths["train_parquet"].read_bytes()
                ).hexdigest(),
                "train_manifest_sha256": hashlib.sha256(
                    paths["train_manifest"].read_bytes()
                ).hexdigest(),
                "m5_prebuilt_sha256": source_sha256,
                "mtf_cache_manifest_sha256": cache_manifest_sha256,
                "val_fit_row_count": 0,
                "test_fit_row_count": 0,
            },
        },
        "input_normalization_fit_population_proof": {
            "fixture_scope": "metadata_mechanics_only"
        },
        "multi_tf": {
            "matrix_contract": HTF_V4_MATRIX_CONTRACT,
            "closed_bar_target_availability": True,
            "entry_family_tf_token_order": _entry_family_tf_tokens(),
            "entry_family_tf_gate_width": len(_entry_family_tf_tokens()),
        },
        "model_native_joint_task_weighting": joint_task_weighting_metadata(
            {name: 0.1 for name in JOINT_TASK_NAMES},
            supervision_observed={name: True for name in JOINT_TASK_NAMES},
            gradient_observed={name: True for name in JOINT_TASK_NAMES},
        ),
    }
    paths["bundle_metadata"].write_text(json.dumps(bundle), encoding="utf-8")
    return paths


@pytest.mark.parametrize("cuda_build", [None, "fixture-build-version"])
def test_environment_metadata_is_passive_and_hardware_is_unproven(
    monkeypatch: pytest.MonkeyPatch, cuda_build: str | None
) -> None:
    monkeypatch.setattr(preflight.torch.version, "cuda", cuda_build)
    metadata = preflight._environment_metadata()
    assert metadata["pytorch"] == preflight.torch.__version__
    assert metadata["pytorch_cuda_build"] == cuda_build
    assert metadata["hardware_readiness"] == "UNPROVEN_NOT_PROBED"
    assert not {"cuda_runtime", "cudnn", "cuda_available", "gpu_name"}.intersection(metadata)


def test_full_static_metadata_path_has_owner_parity_and_limited_authority(
    static_preflight_inputs: dict[str, Path],
) -> None:
    report = preflight.build_static_preflight(**static_preflight_inputs)
    assert report["schema_version"] == "gx1_pre_fulltrain_static_preflight_v3"
    assert report["decision"] == "PASS"
    assert report["environment"]["hardware_readiness"] == "UNPROVEN_NOT_PROBED"
    assert report["evidence_scope"] == {
        "pass_meaning": "static_row_boundary_and_metadata_checks_only",
        "training_or_evaluation_authority": False,
        "outcome_containment": "UNPROVEN_NOT_INSPECTED",
        "source_audit_population_bindings": "UNPROVEN_NOT_VALIDATED",
    }
    feature = report["feature_audit"]
    assert feature["semantic_eight_families"] == list(MODEL_NATIVE_TRAINING_SPECIALISTS)
    assert feature["five_timeframes"] == list(MULTI_TF_TIMEFRAMES)
    assert feature["mtf_v4"]["matrix_contract"] == HTF_V4_MATRIX_CONTRACT
    assert feature["mtf_v4"]["entry_timeframes"] == list(ENTRY_MTF_CONTEXT_TIMEFRAMES)
    assert feature["mtf_v4"]["entry_family_tf_token_order"] == _entry_family_tf_tokens()
    assert feature["mtf_v4"]["entry_family_tf_gate_width"] == len(
        _entry_family_tf_tokens()
    )
    assert feature["mtf_cache_test_boundary"]["scope"] == (
        "declared_cache_last_timestamps_only_no_array_validation"
    )
    assert report["tasks"]["names"] == list(JOINT_TASK_NAMES)
    split_audit = report["data_split_audit"]
    assert "purge_embargo" not in split_audit
    assert split_audit["row_boundary_scope"]["purge_embargo_measurement"] == (
        "NOT_PERFORMED"
    )
    assert split_audit["row_boundary_scope"]["outcome_containment"] == (
        "UNPROVEN_NOT_INSPECTED"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("matrix_contract", "unknown-matrix"),
        ("closed_bar_target_availability", False),
        ("entry_family_tf_token_order", _entry_family_tf_tokens()[::-1]),
        (
            "entry_family_tf_token_order",
            ["m15:unknown-family", *_entry_family_tf_tokens()[1:]],
        ),
        ("entry_family_tf_gate_width", len(_entry_family_tf_tokens()) + 1),
        ("entry_family_tf_gate_width", float(len(_entry_family_tf_tokens()))),
        ("entry_family_tf_gate_width", str(len(_entry_family_tf_tokens()))),
        ("entry_family_tf_gate_width", None),
    ],
)
def test_static_preflight_rejects_mutated_bundle_route_metadata(
    static_preflight_inputs: dict[str, Path], field: str, value: Any
) -> None:
    path = static_preflight_inputs["bundle_metadata"]
    bundle = json.loads(path.read_text(encoding="utf-8"))
    bundle["multi_tf"][field] = value
    path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(PreflightError, match="MTF_EIGHT_FAMILY_CONTRACT_INVALID"):
        preflight.build_static_preflight(**static_preflight_inputs)


@pytest.mark.parametrize(
    "field,value",
    [
        (
            "MODEL_NATIVE_TRAINING_SPECIALISTS",
            (*MODEL_NATIVE_TRAINING_SPECIALISTS, "unknown-family"),
        ),
        (
            "MODEL_NATIVE_TRAINING_SPECIALISTS",
            ("unknown-family", *MODEL_NATIVE_TRAINING_SPECIALISTS[1:]),
        ),
        ("ENTRY_MTF_CONTEXT_TIMEFRAMES", ("M5", *ENTRY_MTF_CONTEXT_TIMEFRAMES)),
        ("MULTI_TF_TIMEFRAMES", (*MULTI_TF_TIMEFRAMES, "W1")),
    ],
)
def test_static_preflight_rejects_owner_drift_before_artifact_access(
    monkeypatch: pytest.MonkeyPatch,
    static_preflight_inputs: dict[str, Path],
    field: str,
    value: tuple[str, ...],
) -> None:
    def reject_artifact_access(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("owner drift must fail before artifact access")

    monkeypatch.setattr(architecture_owner, field, value)
    monkeypatch.setattr(preflight, "scan_allowed_split", reject_artifact_access)
    with pytest.raises(PreflightError, match="PRODUCTION_ARCHITECTURE_INVALID"):
        preflight.build_static_preflight(**static_preflight_inputs)


def test_static_preflight_rejects_unknown_task_metadata(
    static_preflight_inputs: dict[str, Path],
) -> None:
    path = static_preflight_inputs["bundle_metadata"]
    bundle = json.loads(path.read_text(encoding="utf-8"))
    bundle["model_native_joint_task_weighting"]["task_names"][0] = "unknown-task"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(PreflightError, match="TASK_WEIGHTING_INVALID"):
        preflight.build_static_preflight(**static_preflight_inputs)


@pytest.mark.parametrize(
    "timeframes",
    [MULTI_TF_TIMEFRAMES[:-1], (*MULTI_TF_TIMEFRAMES, "W1")],
)
def test_cache_metadata_rejects_missing_or_unknown_timeframes(
    tmp_path: Path, timeframes: tuple[str, ...]
) -> None:
    path = tmp_path / "cache_manifest.json"
    path.write_text(
        json.dumps({
            "tfs": {timeframe: {"last_ts_ns": 0} for timeframe in timeframes}
        }),
        encoding="utf-8",
    )
    with pytest.raises(PreflightError, match="MTF_CACHE_TIMEFRAME_MANIFEST_INVALID"):
        inspect_mtf_cache_test_boundary(path)
