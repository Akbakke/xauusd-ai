from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from gx1.scripts import materialize_unified_exit_lifecycle_v2 as producer
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
)
from gx1.contracts.unified_exit_lifecycle_v2 import (
    UNIFIED_EXIT_ECONOMIC_AUTHORITY_SCHEMA_VERSION,
    terminal_state_counts_sha256,
)
from gx1.contracts.unified_exit_lifecycle_v1 import (
    UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION,
)
from gx1.scripts.materialize_unified_exit_lifecycle_v2 import (
    COMPACT_COLUMNS,
    ECONOMIC_COUNTS_SCHEMA_VERSION,
    build_compact_split,
    materialize_compact_train_val_bundle,
    pair_chunk_for_epoch,
    pair_chunk_permutation,
    pair_schedule_coverage,
    require_compact_split,
    scheduled_pair_chunk_pointer,
)


LINEAGE = "4" * 64
M1_SHA = "5" * 64
GAP_SOURCE_SHA = "7" * 64
ENTRY_BINDING_SHA = "8" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _economic_authority(
    tmp_path: Path,
    *,
    split: str,
    dataset_run_id: str,
    entry_rows: int,
    state_counts: list[int | None],
) -> Path:
    mapping = {
        (entry_row, side): state_counts[entry_row * 2 + side]
        for entry_row in range(entry_rows)
        for side in (0, 1)
    }
    counts_path = (tmp_path / f"{split}.economic_counts.json").resolve()
    _write_json(
        counts_path,
        {
            "schema_version": ECONOMIC_COUNTS_SCHEMA_VERSION,
            "decision": "PASS",
            "dataset_run_id": dataset_run_id,
            "split": split,
            "entry_rows": entry_rows,
            "test_data_used": False,
            "terminal_state_counts": [
                {
                    "entry_row_index": entry_row,
                    "side_index": side,
                    "terminal_state_count": mapping[(entry_row, side)],
                }
                for entry_row in range(entry_rows)
                for side in (0, 1)
            ],
        },
    )
    authority_path = (tmp_path / f"{split}.authority.json").resolve()
    _write_json(
        authority_path,
        {
            "schema_version": UNIFIED_EXIT_ECONOMIC_AUTHORITY_SCHEMA_VERSION,
            "decision": "PASS",
            "authority_artifact_path": str(counts_path),
            "authority_artifact_sha256": _sha(counts_path),
            "terminal_state_counts_sha256": terminal_state_counts_sha256(mapping),
            "economic_terminal_definition_sha256": "6" * 64,
            "terminal_event_verifier_schema_version": (
                "gx1_economic_terminal_event_verifier_v1"
            ),
            "terminal_events_recomputed_from_train_val_only": True,
            "test_data_used": False,
        },
    )
    return authority_path


def _inputs(tmp_path: Path) -> dict[str, object]:
    dataset_run_id = "synthetic-train-val"
    m1_path = (tmp_path / "m1.parquet").resolve()
    m1_times = pd.date_range("2020-01-01", periods=3000, freq="min", tz="UTC")
    pd.DataFrame({"time": m1_times}).to_parquet(m1_path, index=False)
    m1_manifest = (tmp_path / "m1.manifest.json").resolve()
    _write_json(
        m1_manifest,
        {
            "timeframe": "M1",
            "quote_complete_m1": True,
            "test_accessed": False,
            "output_parquet": str(m1_path),
            "output_parquet_sha256": _sha(m1_path),
            "row_count": len(m1_times),
        },
    )
    result: dict[str, object] = {
        "output_dir": (tmp_path / "bundle").resolve(),
        "dataset_run_id": dataset_run_id,
        "m1_source_path": m1_path,
        "m1_source_manifest_path": m1_manifest,
        "planned_epochs": 1,
        "claim_full_coverage": False,
    }
    split_rows: dict[str, tuple[Path, list[pd.Timestamp]]] = {}
    for split, starts in {
        "train": [m1_times[0], m1_times[10]],
        "val": [m1_times[20]],
    }.items():
        entry = (tmp_path / f"{split}.entry.parquet").resolve()
        pd.DataFrame({"time": starts}).to_parquet(entry, index=False)
        split_rows[split] = (entry, starts)
        authority = _economic_authority(
            tmp_path,
            split=split,
            dataset_run_id=dataset_run_id,
            entry_rows=len(starts),
            state_counts=[600, 1100] * len(starts),
        )
        result[f"{split}_entry_path"] = entry
        result[f"{split}_economic_authority_path"] = authority
        result[f"{split}_split_end"] = "2020-01-03T00:00:00+00:00"
    lifecycle_dir = (tmp_path / "legacy_lifecycle_authority").resolve()
    lifecycle_dir.mkdir()
    _write_json(
        lifecycle_dir / "UNIFIED_EXIT_LIFECYCLE_MANIFEST.json",
        {
            "schema_version": UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION,
            "decision": "PASS",
            "entry_run_id": dataset_run_id,
            "splits": {
                split: {
                    "entry_dataset_path": str(entry),
                    "entry_dataset_sha256": _sha(entry),
                    "episode_rows": len(starts) * 2,
                }
                for split, (entry, starts) in split_rows.items()
            },
        },
    )
    for split, (entry, starts) in split_rows.items():
        manifest = (tmp_path / f"{split}.entry.manifest.json").resolve()
        _write_json(
            manifest,
            {
                "schema_version": MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
                "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
                "output_data_path": str(entry),
                "extra": {
                    "rows": len(starts),
                    "entry_run_id": dataset_run_id,
                    "pretest_only": True,
                    "pretest_test_guard": {"test_accessed": False},
                    "unified_exit_lifecycle": {
                        "schema_version": (
                            UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION
                        ),
                        "output_dir": str(lifecycle_dir),
                    },
                },
            },
        )
        result[f"{split}_entry_manifest_path"] = manifest
    return result


def _stub_full_admission(inputs: dict[str, object]) -> dict[str, object]:
    root_path = (
        Path(inputs["train_entry_manifest_path"]).parent
        / "legacy_lifecycle_authority"
        / "UNIFIED_EXIT_LIFECYCLE_MANIFEST.json"
    )
    return {
        "root_manifest_path": root_path,
        "root_manifest_sha256": _sha(root_path),
        "splits": {
            split: {"entry_path": Path(inputs[f"{split}_entry_path"])}
            for split in ("train", "val")
        },
        "entry_windows": {
            split: {
                "manifest_sha256": _sha(
                    Path(inputs[f"{split}_entry_manifest_path"])
                )
            }
            for split in ("train", "val")
        },
    }


def test_pair_schedule_is_outcome_blind_deterministic_and_resumable() -> None:
    parameters = set(inspect.signature(pair_chunk_permutation).parameters)
    assert not parameters.intersection({"side", "side_index", "reward", "pnl", "q", "feature"})
    order = pair_chunk_permutation(
        chunk_count=13,
        lineage_sha256=LINEAGE,
        split="train",
        entry_row_index=7,
    )
    assert sorted(order) == list(range(13))
    assert order == pair_chunk_permutation(
        chunk_count=13,
        lineage_sha256=LINEAGE,
        split="train",
        entry_row_index=7,
    )
    before_restart = [
        pair_chunk_for_epoch(
            chunk_count=13,
            epoch_index=epoch,
            lineage_sha256=LINEAGE,
            split="train",
            entry_row_index=7,
        )
        for epoch in range(5)
    ]
    after_restart = [
        pair_chunk_for_epoch(
            chunk_count=13,
            epoch_index=epoch,
            lineage_sha256=LINEAGE,
            split="train",
            entry_row_index=7,
        )
        for epoch in range(5, 13)
    ]
    assert before_restart + after_restart == list(order)


def test_coverage_rejects_false_full_coverage_claim() -> None:
    with pytest.raises(RuntimeError, match="FALSE_FULL_COVERAGE_CLAIM"):
        pair_schedule_coverage(
            chunk_count_by_entry={0: 2, 1: 7},
            planned_epochs=3,
            lineage_sha256=LINEAGE,
            split="train",
            claim_full_coverage=True,
        )
    coverage = pair_schedule_coverage(
        chunk_count_by_entry={0: 2, 1: 7},
        planned_epochs=3,
        lineage_sha256=LINEAGE,
        split="train",
        claim_full_coverage=False,
    )
    assert coverage["covered_unique_chunks"] == 5
    assert coverage["total_chunks"] == 9
    assert coverage["full_coverage"] is False
    assert coverage["resume_cursor"] == {"next_epoch_index": 3}


def test_compact_rows_bind_deep_successors_without_targets() -> None:
    clock = pd.date_range("2020-01-01", periods=2200, freq="min", tz="UTC")
    frame = build_compact_split(
        entry_times=[clock[0]],
        m1_times=clock,
        split="train",
        split_end=clock[-1] + pd.Timedelta(minutes=1),
        terminal_state_count_by_entry_side={(0, 0): 700, (0, 1): 1025},
        m1_source_sha256=M1_SHA,
        gap_classification_source_sha256=GAP_SOURCE_SHA,
        entry_binding_sha256=ENTRY_BINDING_SHA,
    )
    assert len(frame) == 1
    assert tuple(frame.columns) == COMPACT_COLUMNS
    assert frame.loc[0, "long_chunk_count"] == 2
    assert frame.loc[0, "short_chunk_count"] == 3
    assert frame.loc[0, "pair_chunk_count"] == 3
    forbidden = ("target", "reward", "pnl", "_q", "feature")
    assert not any(token in column.lower() for column in frame for token in forbidden)
    pointer = scheduled_pair_chunk_pointer(
        compact_row=frame.iloc[0].to_dict(),
        epoch_index=2,
        lineage_sha256=LINEAGE,
        split="train",
    )
    assert pointer["both_sides_share_timeline"] is True
    assert pointer["selection_uses_outcome_values"] is False
    assert pointer["chunk_start_bars_in_trade"] == pointer["pair_chunk_slot"] * 512
    altered_clock = clock.copy().to_numpy(copy=True)
    altered_clock[1029] = altered_clock[1029] + pd.Timedelta(seconds=1)
    with pytest.raises(RuntimeError, match="M1_CLOCK_BINDING_INVALID"):
        require_compact_split(
            frame,
            split_end=clock[-1] + pd.Timedelta(minutes=1),
            m1_times=altered_clock,
            expected_m1_source_sha256=M1_SHA,
            expected_entry_binding_sha256=ENTRY_BINDING_SHA,
            expected_gap_classification_source_sha256=GAP_SOURCE_SHA,
        )


def test_validate_no_publish_stays_blocked_without_economic_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    monkeypatch.setattr(
        producer,
        "_require_full_v1_admission",
        lambda **_kwargs: _stub_full_admission(inputs),
    )
    report = materialize_compact_train_val_bundle(**inputs, publish=False)
    assert report["mode"] == "validate_no_publish"
    assert report["decision"] == "BLOCKED"
    assert report["published"] is False
    assert not Path(inputs["output_dir"]).exists()
    assert report["blockers"] == [
        "COMPACT_LIFECYCLE_ECONOMIC_TERMINAL_VERIFIER_UNAVAILABLE"
    ]
    with pytest.raises(RuntimeError, match="PUBLISH_BLOCKED"):
        materialize_compact_train_val_bundle(**inputs, publish=True)


def test_swapped_entry_parquet_or_manifest_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    monkeypatch.setattr(
        producer,
        "_require_full_v1_admission",
        lambda **_kwargs: _stub_full_admission(inputs),
    )
    swapped_parquet = dict(inputs)
    swapped_parquet["train_entry_path"] = inputs["val_entry_path"]
    with pytest.raises(RuntimeError, match="ENTRY_ARTIFACT_BINDING_INVALID"):
        materialize_compact_train_val_bundle(**swapped_parquet, publish=False)

    swapped_manifest = dict(inputs)
    swapped_manifest["train_entry_manifest_path"] = inputs[
        "val_entry_manifest_path"
    ]
    with pytest.raises(RuntimeError, match="ENTRY_ARTIFACT_BINDING_INVALID"):
        materialize_compact_train_val_bundle(**swapped_manifest, publish=False)


def test_entry_manifest_schema_and_row_count_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    admission = _stub_full_admission(inputs)
    monkeypatch.setattr(
        producer,
        "_require_full_v1_admission",
        lambda **_kwargs: admission,
    )
    manifest_path = Path(inputs["train_entry_manifest_path"])
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    invalid_schema = json.loads(json.dumps(original))
    invalid_schema["schema_version"] = "wrong_schema"
    _write_json(manifest_path, invalid_schema)
    with pytest.raises(RuntimeError, match="ENTRY_ARTIFACT_BINDING_INVALID"):
        materialize_compact_train_val_bundle(**inputs, publish=False)

    invalid_rows = json.loads(json.dumps(original))
    invalid_rows["extra"]["rows"] += 1
    _write_json(manifest_path, invalid_rows)
    with pytest.raises(RuntimeError, match="ENTRY_ARTIFACT_BINDING_INVALID"):
        materialize_compact_train_val_bundle(**inputs, publish=False)


@pytest.mark.parametrize(
    ("clock", "classification"),
    [
        (
            pd.DatetimeIndex(
                list(pd.date_range("2020-01-03T20:00Z", periods=11, freq="min"))
                + list(pd.date_range("2020-01-05T22:00Z", periods=20, freq="min"))
            ),
            "weekend_source_absence",
        ),
        (
            pd.DatetimeIndex(
                list(pd.date_range("2020-01-07T10:00Z", periods=11, freq="min"))
                + list(pd.date_range("2020-01-07T10:13Z", periods=20, freq="min"))
            ),
            "unknown_source_absence",
        ),
    ],
)
def test_non_m1_transition_censors_before_gap_and_has_no_successor(
    clock: pd.DatetimeIndex, classification: str
) -> None:
    frame = build_compact_split(
        entry_times=[clock[0]],
        m1_times=clock,
        split="train",
        split_end=clock[-1] + pd.Timedelta(minutes=1),
        terminal_state_count_by_entry_side={(0, 0): None, (0, 1): None},
        m1_source_sha256=M1_SHA,
        gap_classification_source_sha256=GAP_SOURCE_SHA,
        entry_binding_sha256=ENTRY_BINDING_SHA,
    )
    row = frame.iloc[0]
    assert row["gap_classification"] == classification
    assert row["available_state_count"] == 6
    assert row["long_lifecycle_state_count"] == 6
    pointer = scheduled_pair_chunk_pointer(
        compact_row=row.to_dict(),
        epoch_index=0,
        lineage_sha256=LINEAGE,
        split="train",
    )
    assert pointer["sides"]["long"]["successor_available"] is False
    assert pointer["sides"]["short"]["successor_available"] is False
    assert pointer["sides"]["long"]["right_censored"] is True
    with pytest.raises(RuntimeError, match="TERMINAL_OUTSIDE_SPLIT"):
        build_compact_split(
            entry_times=[clock[0]],
            m1_times=clock,
            split="train",
            split_end=clock[-1] + pd.Timedelta(minutes=1),
            terminal_state_count_by_entry_side={(0, 0): 7, (0, 1): None},
            m1_source_sha256=M1_SHA,
            gap_classification_source_sha256=GAP_SOURCE_SHA,
            entry_binding_sha256=ENTRY_BINDING_SHA,
        )


def test_compact_validator_uses_external_source_and_entry_identity() -> None:
    clock = pd.date_range("2020-01-01", periods=20, freq="min", tz="UTC")
    frame = build_compact_split(
        entry_times=[clock[0]],
        m1_times=clock,
        split="val",
        split_end=clock[-1] + pd.Timedelta(minutes=1),
        terminal_state_count_by_entry_side={(0, 0): None, (0, 1): None},
        m1_source_sha256=M1_SHA,
        gap_classification_source_sha256=GAP_SOURCE_SHA,
        entry_binding_sha256=ENTRY_BINDING_SHA,
    )
    with pytest.raises(RuntimeError, match="FRAME_INVALID"):
        require_compact_split(
            frame,
            split_end=clock[-1] + pd.Timedelta(minutes=1),
            m1_times=clock,
            expected_m1_source_sha256="9" * 64,
            expected_entry_binding_sha256=ENTRY_BINDING_SHA,
            expected_gap_classification_source_sha256=GAP_SOURCE_SHA,
        )
    with pytest.raises(RuntimeError, match="FRAME_INVALID"):
        require_compact_split(
            frame,
            split_end=clock[-1] + pd.Timedelta(minutes=1),
            m1_times=clock,
            expected_m1_source_sha256=M1_SHA,
            expected_entry_binding_sha256="a" * 64,
            expected_gap_classification_source_sha256=GAP_SOURCE_SHA,
        )


def test_minimal_self_attested_lifecycle_root_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    with pytest.raises(RuntimeError, match="FULL_V1_ADOPTION_INVALID"):
        materialize_compact_train_val_bundle(**inputs, publish=False)
