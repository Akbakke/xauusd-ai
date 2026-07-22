from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_MANDATORY_FAMILY_FEATURES,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    require_model_native_manifest,
)
from gx1.scripts import materialize_entry_model_native_seq513_signal_manifest_v1 as producer
from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_STRUCTURE_FEATURE_NAMES,
    FOUNDATION_STRUCTURE_FEATURE_VERSION,
)
from tests.model_native_rank_reference_support import materialize_test_rank_reference


RUN_ID = "SEQ513_FULL_STACK_RETENTION_TEST"
RANKING_CREATED = datetime(2026, 7, 17, 10, 0, 0, 1, tzinfo=timezone.utc)
OUTPUT_CREATED = datetime(2026, 7, 17, 10, 0, 1, 1, tzinfo=timezone.utc)


def _stamp(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%S%fZ")


def _ranking_payload(tmp_path: Path) -> dict:
    source_path, reference = materialize_test_rank_reference(
        tmp_path / "rank_reference",
        run_id=RUN_ID,
        history_start="2019-12-31T00:00:00Z",
        fit_start="2020-01-01T00:00:00Z",
        fit_end="2025-12-31T23:59:59Z",
    )
    names = [
        MODEL_NATIVE_MANDATORY_SELECTED_FIELDS[0],
        *[
            f"session_regime.rank_candidate_{index:03d}"
            for index in range(MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT)
        ],
    ]
    return {
        "schema_version": producer.TRAIN_FEATURE_RANKING_SCHEMA_VERSION,
        "created_utc": RANKING_CREATED.isoformat(),
        "entry_run_id": RUN_ID,
        "producer": producer.TRAIN_FEATURE_RANKING_PRODUCER,
        "producer_version": producer.TRAIN_FEATURE_RANKING_PRODUCER_VERSION,
        "fit_scope": "train_only",
        "train_start_utc": "2020-01-01T00:00:00+00:00",
        "train_end_utc": "2025-12-31T23:59:59+00:00",
        "source_time_max_utc": "2025-12-31T23:55:00+00:00",
        "target_time_max_utc": "2025-12-31T23:55:00+00:00",
        "source_sha256": str(reference.sidecar["source_parquet_sha256"]),
        "target_sha256": "2" * 64,
        "rank_reference": {
            "path": str(reference.path),
            "sha256": reference.sha256,
            "sidecar_path": str(
                reference.path.with_suffix(reference.path.suffix + ".json")
            ),
            "sidecar_sha256": reference.sidecar_sha256,
            "schema_version": reference.sidecar["schema_version"],
            "entry_run_id": RUN_ID,
            "source_parquet": str(source_path),
            "source_parquet_sha256": str(
                reference.sidecar["source_parquet_sha256"]
            ),
            "history_start_utc": "2019-12-31T00:00:00+00:00",
            "fit_start_utc": "2020-01-01T00:00:00+00:00",
            "fit_end_utc": "2025-12-31T23:59:59+00:00",
            "fit_row_count": reference.fit_row_count,
            "fit_scope": "train_only",
            "rank_transform": reference.sidecar["rank_transform"],
        },
        "ranking_order": dict(producer.TRAIN_FEATURE_RANKING_ORDER),
        "causality_contract": {
            **producer.TRAIN_FEATURE_CAUSALITY_CONTRACT,
            "leakage_columns": [],
        },
        "ranked_features": [
            {"rank": index, "name": name, "score": float(1000 - index)}
            for index, name in enumerate(names, start=1)
        ],
    }


def _write_ranking(tmp_path: Path, payload: dict | None = None) -> Path:
    path = tmp_path / (
        "ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING_"
        f"{_stamp(RANKING_CREATED)}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload or _ranking_payload(tmp_path), allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return path


def _out(tmp_path: Path, *, created: datetime = OUTPUT_CREATED) -> Path:
    return tmp_path / (
        f"{producer.SIGNAL_MANIFEST_EVENT_PREFIX}_{_stamp(created)}.json"
    )


def _args(ranking: Path, out: Path, *, run_id: str = RUN_ID) -> argparse.Namespace:
    return argparse.Namespace(
        feature_ranking_json=str(ranking),
        out=str(out),
        run_id=run_id,
    )


def _freeze_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        producer,
        "_utc_now",
        lambda: datetime(2026, 7, 17, 10, 0, 1, 500_000, tzinfo=timezone.utc),
    )


def test_producer_keeps_all_code_owned_fields_first_and_only_ranks_remainder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_clock(monkeypatch)
    ranking = _write_ranking(tmp_path)
    out = _out(tmp_path)

    manifest = producer.run(_args(ranking, out))

    assert out.is_file()
    assert manifest["selected_feature_count"] == MODEL_NATIVE_SELECTED_FEATURE_COUNT
    mandatory_count = len(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
    assert manifest["selected_features"][:mandatory_count] == list(
        MODEL_NATIVE_MANDATORY_SELECTED_FIELDS
    )
    assert (
        manifest["ranked_remainder_feature_count"]
        == MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT
    )
    assert manifest["selected_features"][mandatory_count:] == [
        f"session_regime.rank_candidate_{index:03d}"
        for index in range(MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT)
    ]
    assert manifest["smart_layer_feature_counts"] == {
        family: len(features)
        for family, features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
    }
    assert manifest["foundation_structure_feature_version"] == (
        FOUNDATION_STRUCTURE_FEATURE_VERSION
    )
    assert manifest["foundation_structure_feature_count"] == len(
        FOUNDATION_STRUCTURE_FEATURE_NAMES
    )
    assert manifest["foundation_structure_missing_feature_count"] == 0
    assert manifest["foundation_structure_all_required_selected"] is True
    assert manifest["feature_ranking"]["sha256"]
    assert manifest[
        "ranking_artifact_is_upstream_prerequisite_not_runtime_authority"
    ] is True
    require_model_native_manifest(manifest, context="PRODUCER_TEST")


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda payload: payload["ranked_features"][-1].update(
                {"name": payload["ranked_features"][-2]["name"]}
            ),
            "DUPLICATE_NAMES",
        ),
        (
            lambda payload: payload["ranked_features"][1].update(
                {"name": "mfe_first_n"}
            ),
            "TARGET_OR_LEAK_FIELD_FORBIDDEN",
        ),
        (
            lambda payload: payload["ranked_features"][1].update(
                {"name": "forward_return_24"}
            ),
            "TARGET_OR_LEAK_FIELD_FORBIDDEN",
        ),
        (
            lambda payload: payload["causality_contract"].update(
                {"test_rows_used": True}
            ),
            "CAUSALITY_CONTRACT_INVALID",
        ),
        (
            lambda payload: payload["rank_reference"].update(
                {"sha256": "f" * 64}
            ),
            "RANK_REFERENCE_METADATA_MISMATCH",
        ),
    ],
)
def test_invalid_ranking_fails_before_output_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    error: str,
) -> None:
    _freeze_clock(monkeypatch)
    payload = _ranking_payload(tmp_path)
    mutate(payload)
    ranking = _write_ranking(tmp_path, payload)
    out = _out(tmp_path)

    with pytest.raises(RuntimeError, match=error):
        producer.run(_args(ranking, out))

    assert not out.exists()


def test_mutated_rank_reference_sidecar_fails_before_output_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_clock(monkeypatch)
    payload = _ranking_payload(tmp_path)
    sidecar_path = Path(payload["rank_reference"]["sidecar_path"])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["created_utc"] = "2026-07-21T00:00:00+00:00"
    sidecar_path.write_text(
        json.dumps(sidecar, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ranking = _write_ranking(tmp_path, payload)
    out = _out(tmp_path)

    with pytest.raises(RuntimeError, match="RANK_REFERENCE_METADATA_MISMATCH"):
        producer.run(_args(ranking, out))

    assert not out.exists()


def test_invalid_run_id_fails_before_output_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_clock(monkeypatch)
    ranking = _write_ranking(tmp_path)
    out = _out(tmp_path)

    with pytest.raises(Exception, match="provide --run-id"):
        producer.run(_args(ranking, out, run_id=""))

    assert not out.exists()


def test_ranking_input_symlink_is_rejected_before_output_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_clock(monkeypatch)
    real = _write_ranking(tmp_path / "real")
    link_dir = tmp_path / "link"
    link_dir.mkdir()
    link = link_dir / real.name
    link.symlink_to(real)
    out = _out(tmp_path)

    with pytest.raises(RuntimeError, match="FEATURE_RANKING_SYMLINK_FORBIDDEN"):
        producer.run(_args(link, out))

    assert not out.exists()


def test_output_symlink_parent_is_rejected_without_writing_real_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_clock(monkeypatch)
    ranking = _write_ranking(tmp_path / "ranking")
    real_out_dir = tmp_path / "real_out"
    real_out_dir.mkdir()
    alias = tmp_path / "out_alias"
    alias.symlink_to(real_out_dir, target_is_directory=True)
    out = _out(alias)

    with pytest.raises(RuntimeError, match="SIGNAL_MANIFEST_OUTPUT_SYMLINK_FORBIDDEN"):
        producer.run(_args(ranking, out))

    assert not list(real_out_dir.iterdir())


def test_backdated_or_future_output_timestamp_is_rejected_without_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranking = _write_ranking(tmp_path)
    monkeypatch.setattr(
        producer,
        "_utc_now",
        lambda: datetime(2026, 7, 17, 11, 0, tzinfo=timezone.utc),
    )
    stale = _out(tmp_path)
    with pytest.raises(RuntimeError, match="TIMESTAMP_STALE"):
        producer.run(_args(ranking, stale))
    assert not stale.exists()

    monkeypatch.setattr(
        producer,
        "_utc_now",
        lambda: datetime(2026, 7, 17, 9, 58, tzinfo=timezone.utc),
    )
    future = _out(tmp_path)
    with pytest.raises(RuntimeError, match="TIMESTAMP_FUTURE"):
        producer.run(_args(ranking, future))
    assert not future.exists()
