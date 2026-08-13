import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.htf_v29_registry_test_support import synthetic_v29_registry_constants

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTEXT_TAG,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_model_native_state_v2 import (
    MODEL_NATIVE_HISTORY_MODE,
    MODEL_NATIVE_RANK_TRANSFORM,
    MODEL_NATIVE_STATE_SCHEMA_VERSION,
    MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
)
from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    _log_label_distribution_proof,
    _model_native_state_contract,
    write_manifest,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields


def _splits() -> dict[str, dict[str, str]]:
    return {
        "train": {
            "start": "2020-11-09 00:00:00+00:00",
            "end": "2025-09-30 23:59:59+00:00",
        },
        "val": {
            "start": "2025-10-01 00:00:00+00:00",
            "end": "2025-12-31 23:59:59+00:00",
        },
        "test": {
            "start": "2026-01-01 00:00:00+00:00",
            "end": "2026-06-26 03:25:00+00:00",
        },
    }


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "canonical_source.parquet"
    source.write_bytes(b"canonical-source-proof")
    return source


def _extra(tmp_path: Path) -> dict:
    signal_contract = model_native_signal_contract_metadata(
        canonical_model_native_selected_fields(
            remainder_prefix="session_regime.split_writer_fixture"
        )
    )
    cache_dir = tmp_path / "mtf_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_manifest = cache_dir / "manifest.json"
    cache_manifest.write_text('{"fixture":"v4"}\n', encoding="utf-8")
    cache_manifest_sha = hashlib.sha256(cache_manifest.read_bytes()).hexdigest()
    return {
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
        "model_native_signal_contract": signal_contract,
        "signal_bridge": {
            "id": signal_contract["schema_version"],
            "fields": list(signal_contract["fields"]),
            "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
            "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
            "bridge_dim": 0,
            "bridge_source": None,
        },
        "ctx_contract": {
            "tag": MODEL_NATIVE_CONTEXT_TAG,
            "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
            "ctx_cat_dim": 5,
            "ctx_cont_names": list(MODEL_NATIVE_CTX_CONT_FIELDS),
            "ctx_cat_names": list(MODEL_NATIVE_CTX_CAT_FIELDS),
        },
        "model_native_state_contract": {
            "schema_version": MODEL_NATIVE_STATE_SCHEMA_VERSION,
            "feature_history_start_utc": "2020-11-01T00:00:00Z",
            "rank_fit_start_utc": "2020-11-09T00:00:00Z",
            "rank_fit_end_utc": "2025-09-30T23:59:59Z",
            "rank_reference_npz": "/immutable/rank_reference.npz",
            "rank_reference_npz_sha256": "a" * 64,
            "rank_reference_sidecar_sha256": "c" * 64,
            "rank_reference_schema_version": MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
            "rank_reference_sidecar_json": "/immutable/rank_reference.npz.json",
            "rank_reference_source_parquet": "/immutable/source.parquet",
            "rank_reference_source_parquet_sha256": "b" * 64,
            "rank_reference_fit_row_count": 123,
            "rank_reference_fit_scope": "train_only",
            "rank_transform": MODEL_NATIVE_RANK_TRANSFORM,
            "feature_history_mode": MODEL_NATIVE_HISTORY_MODE,
            "split_reset_allowed": False,
            "post_fit_rows_in_rank_reference": False,
            "runtime_rule_free": True,
            "entry_run_id": "MODEL_NATIVE_DATASET_BUILD_PYTEST",
        },
        "multi_tf_cache_binding": {
            "cache_dir": str(cache_dir.resolve()),
            "manifest_path": str(cache_manifest.resolve()),
            "manifest_sha256": cache_manifest_sha,
            "cache_identity_sha256": "d" * 64,
            "m5_prebuilt_source": str((tmp_path / "source.parquet").resolve()),
            "m5_prebuilt_source_sha256": "e" * 64,
            # V29 split manifests freeze the TRAIN-fitted registry constants
            # inside the binding; the writer validates them via their owner.
            "v29_registry_constants": synthetic_v29_registry_constants(),
        },
    }


def test_canonical_writer_stamps_exact_seq513_schema_on_all_split_manifests(
    tmp_path: Path,
) -> None:
    for split_name in ("train", "val", "test"):
        output_path = tmp_path / f"model_native_seq513_{split_name}.parquet"
        manifest_path = write_manifest(
            output_path=output_path,
            build_command=[
                "python",
                "-m",
                "gx1.scripts.build_entry_v10_ctx_training_dataset_v3",
            ],
            source_parquet=_source(tmp_path),
            tape_root=tmp_path / "tape",
            splits=_splits(),
            extra=_extra(tmp_path),
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION
        assert manifest["manifest_variant"] == MODEL_NATIVE_CONTRACT_MODE
        assert manifest["expected_seq_snap_width"] == MODEL_NATIVE_SIGNAL_DIM
        assert manifest["output_data_path"] == str(output_path)
        assert manifest["splits"] == _splits()
        assert manifest["inputs"] == {
            "source_parquet": str(_source(tmp_path)),
            "tape_root": str(tmp_path / "tape"),
        }


def test_test_label_distribution_is_withheld_before_final_evaluation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = pd.DataFrame({"y_direction": [0, 0, 1, 2]})

    with caplog.at_level("INFO"):
        _log_label_distribution_proof(frame, split="test")

    assert "withheld_until_final_candidate_evaluation" in caplog.text
    assert "long=2" not in caplog.text


def test_canonical_writer_records_one_exact_source(tmp_path: Path) -> None:
    source = _source(tmp_path)
    manifest_path = write_manifest(
        output_path=tmp_path / "model_native_seq513.parquet",
        build_command=["builder"],
        source_parquet=source,
        tape_root=tmp_path / "tape",
        extra=_extra(tmp_path),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["inputs"]["source_parquet"] == str(source)


def test_canonical_writer_rejects_missing_exact_source(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="MODEL_NATIVE_SOURCE_PARQUET_MISSING"):
        write_manifest(
            output_path=tmp_path / "model_native_seq513.parquet",
            build_command=["builder"],
            source_parquet=tmp_path / "missing.parquet",
            tape_root=tmp_path / "tape",
            extra=_extra(tmp_path),
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda extra: extra.update({"model_native_signal_contract": None}),
            "MODEL_NATIVE_MANIFEST_SIGNAL_CONTRACT_MISSING",
        ),
        (
            lambda extra: extra["signal_bridge"].update({"bridge_dim": 7}),
            "MODEL_NATIVE_MANIFEST_BRIDGE_DIM_INVALID",
        ),
        (
            lambda extra: extra["signal_bridge"].update(
                {"fields": list(reversed(extra["signal_bridge"]["fields"]))}
            ),
            "MODEL_NATIVE_MANIFEST_ORDERED_FIELDS_MISMATCH",
        ),
    ],
)
def test_canonical_split_writer_rejects_soft_or_legacy_contracts(
    tmp_path: Path,
    mutation,
    error: str,
) -> None:
    extra = copy.deepcopy(_extra(tmp_path))
    mutation(extra)

    with pytest.raises(RuntimeError, match=error):
        write_manifest(
            output_path=tmp_path / "model_native_seq513_train.parquet",
            build_command=["builder"],
            source_parquet=_source(tmp_path),
            tape_root=tmp_path / "tape",
            splits=_splits(),
            extra=extra,
        )

    assert not list(tmp_path.glob("*.manifest.json"))


def _rank_reference_fixture(tmp_path: Path) -> tuple[Path, dict, Path]:
    source = tmp_path / "canonical_source.parquet"
    model_source = tmp_path / "model_source.parquet"
    times = pd.to_datetime(
        [
            "2020-11-01T00:00:00Z",
            "2020-11-09T00:00:00Z",
            "2025-09-30T23:59:59Z",
        ],
        utc=True,
    )
    close = np.asarray([1800.0, 1801.0, 2300.0], dtype=np.float64)
    market = pd.DataFrame(
        {
            "time": times,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "bid_close": close - 0.05,
            "ask_close": close + 0.05,
        }
    )
    market.assign(canonical_only=[1.0, 2.0, 3.0]).to_parquet(
        source, index=False
    )
    market.assign(model_only=[4.0, 5.0, 6.0]).to_parquet(
        model_source, index=False
    )
    rank_reference = tmp_path / "model_native_rank_reference.npz"
    fit_start = pd.Timestamp("2020-11-09T00:00:00Z")
    fit_end = pd.Timestamp("2025-09-30T23:59:59Z")
    np.savez_compressed(
        rank_reference,
        schema_version=np.asarray([MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION]),
        fit_start_ns=np.asarray([fit_start.value], dtype=np.int64),
        fit_end_ns=np.asarray([fit_end.value], dtype=np.int64),
        fit_row_count=np.asarray([3], dtype=np.int64),
        entry_run_id=np.asarray(["MODEL_NATIVE_DATASET_BUILD_PYTEST"]),
        atr_bps_sorted=np.asarray([9.0, 10.0, 11.0], dtype=np.float64),
        spread_bps_sorted=np.asarray([0.8, 1.0, 1.2], dtype=np.float64),
    )
    payload = {
        "schema_version": MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
        "fit_scope": "train_only",
        "rank_transform": MODEL_NATIVE_RANK_TRANSFORM,
        "row_level_state_present": False,
        "entry_run_id": "MODEL_NATIVE_DATASET_BUILD_PYTEST",
        "source_parquet": str(source),
        "source_parquet_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "out_npz": str(rank_reference.resolve()),
        "out_npz_sha256": hashlib.sha256(rank_reference.read_bytes()).hexdigest(),
        "history_start_utc": "2020-11-01T00:00:00Z",
        "fit_start_utc": str(fit_start),
        "fit_end_utc": str(fit_end),
        "fit_row_count": 3,
        "fit_time_min": str(fit_start),
        "fit_time_max": str(fit_end),
    }
    rank_reference.with_suffix(".npz.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return rank_reference, payload, model_source


def test_builder_requires_audited_rank_reference_and_source_hashes(
    tmp_path: Path,
) -> None:
    rank_reference, payload, model_source = _rank_reference_fixture(tmp_path)

    contract = _model_native_state_contract(
        args=argparse.Namespace(
            model_native_rank_reference_npz=str(rank_reference),
            source_parquet=str(model_source),
            canonical_v2_parquet=payload["source_parquet"],
            run_id="MODEL_NATIVE_DATASET_BUILD_PYTEST",
        ),
        feature_history_start=pd.Timestamp("2020-11-01T00:00:00Z"),
        train_start=pd.Timestamp("2020-11-09T00:00:00Z"),
        train_end=pd.Timestamp("2025-09-30T23:59:59Z"),
    )

    assert contract["rank_reference_npz_sha256"] == payload["out_npz_sha256"]
    assert (
        contract["rank_reference_source_parquet_sha256"]
        == payload["source_parquet_sha256"]
    )
    assert contract["rank_reference_fit_row_count"] == 3
    assert contract["rank_reference_fit_scope"] == "train_only"
    assert contract["split_reset_allowed"] is False
    assert contract["entry_run_id"] == "MODEL_NATIVE_DATASET_BUILD_PYTEST"


def test_builder_rejects_missing_or_tampered_rank_reference_contract(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="MODEL_NATIVE_RANK_REFERENCE_REQUIRED"):
        _model_native_state_contract(
            args=argparse.Namespace(
                model_native_rank_reference_npz="",
                run_id="MODEL_NATIVE_DATASET_BUILD_PYTEST",
            ),
            feature_history_start=pd.Timestamp("2020-11-01T00:00:00Z"),
            train_start=pd.Timestamp("2020-11-09T00:00:00Z"),
            train_end=pd.Timestamp("2025-09-30T23:59:59Z"),
        )

    rank_reference, payload, model_source = _rank_reference_fixture(tmp_path)
    payload["source_parquet_sha256"] = "0" * 64
    rank_reference.with_suffix(".npz.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="SOURCE_SHA_MISMATCH"):
        _model_native_state_contract(
            args=argparse.Namespace(
                model_native_rank_reference_npz=str(rank_reference),
                source_parquet=str(model_source),
                canonical_v2_parquet=payload["source_parquet"],
                run_id="MODEL_NATIVE_DATASET_BUILD_PYTEST",
            ),
            feature_history_start=pd.Timestamp("2020-11-01T00:00:00Z"),
            train_start=pd.Timestamp("2020-11-09T00:00:00Z"),
            train_end=pd.Timestamp("2025-09-30T23:59:59Z"),
        )
