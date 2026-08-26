from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_exit_production_architecture_v1 import (
    PRODUCTION_MTF_PER_TF_WINDOW_BARS,
)
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.scripts.audit_entry_sequence_roll_v1 import audit_sequence_roll
from gx1.scripts.audit_entry_sequence_source_reconstruction_v1 import (
    audit_sequence_source_reconstruction,
)
from tests.entry_v10_trainer_dataset_support import (
    aux_head_target_contract,
    install_multi_tf_stub,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields


def _write_advanced_parquet(
    path: Path,
    *,
    times: list[str] | None = None,
    rolling_sequence: bool = False,
) -> None:
    rows = 3
    seq_len = MODEL_NATIVE_SEQ_LEN
    signal_dim = MODEL_NATIVE_SIGNAL_DIM
    ctx_cont_dim = MODEL_NATIVE_CTX_CONT_DIM
    ctx_cat_dim = MODEL_NATIVE_CTX_CAT_DIM

    if rolling_sequence:
        chain = np.arange(
            (rows + seq_len - 1) * signal_dim,
            dtype=np.float32,
        ).reshape(rows + seq_len - 1, signal_dim)
        seq = [chain[row : row + seq_len].tolist() for row in range(rows)]
        snap = [chain[row + seq_len - 1].tolist() for row in range(rows)]
    else:
        seq = [
            [[float(row + step + col) for col in range(signal_dim)] for step in range(seq_len)]
            for row in range(rows)
        ]
        snap = [
            [float(row + col) for col in range(signal_dim)] for row in range(rows)
        ]
    ctx_cont = [[float(row + col) for col in range(ctx_cont_dim)] for row in range(rows)]
    ctx_cat = [[int((row + col) % 3) for col in range(ctx_cat_dim)] for row in range(rows)]

    columns = {
            "time": pa.array(
                times or [f"2026-01-01T00:{row * 5:02d}:00Z" for row in range(rows)],
                type=pa.string(),
            ),
            "seq": pa.array(seq, type=pa.list_(pa.list_(pa.float64()))),
            "snap": pa.array(snap, type=pa.list_(pa.float64())),
            "ctx_cont": pa.array(ctx_cont, type=pa.list_(pa.float64())),
            "ctx_cat": pa.array(ctx_cat, type=pa.list_(pa.int64())),
            "mae_first_n_bps": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            "y_early_move": pa.array([0.0, 1.0, 0.0], type=pa.float64()),
            "y_quality_score": pa.array([0.2, 0.4, 0.6], type=pa.float64()),
        }
    for target in trainer._MODEL_NATIVE_ACTIVE_TARGET_COLS:
        values = [0.0, 0.0, 0.0]
        if target == "y_position_size_target":
            values = [0.5, 0.5, 0.5]
        elif target == "y_position_size_mask":
            values = [1.0, 1.0, 0.0]
        columns[target] = pa.array(values)
    table = pa.table(columns)
    pq.write_table(table, path)
    signal_contract = model_native_signal_contract_metadata(
        canonical_model_native_selected_fields(
            remainder_prefix="session_regime.memmap_fixture"
        )
    )
    path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "extra": {
                    "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
                    "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
                    "model_native_signal_contract": signal_contract,
                    "aux_head_target_contract": aux_head_target_contract(),
                    "signal_bridge": {
                        "fields": signal_contract["fields"],
                        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def _write_sequence_roll_proof(parquet_path: Path) -> Path:
    manifest_path = parquet_path.with_suffix(".manifest.json")
    proof_path = parquet_path.with_suffix(".sequence_roll_audit.json")
    proof_path.write_text(
        json.dumps(
            audit_sequence_roll(
                parquet_path=parquet_path,
                manifest_path=manifest_path,
            ),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return proof_path


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_backed_advanced_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    """Create causally filtered rows whose histories come from one M5 surface."""

    width = MODEL_NATIVE_SIGNAL_DIM
    source = np.arange((MODEL_NATIVE_SEQ_LEN + 8) * width, dtype=np.float32).reshape(
        MODEL_NATIVE_SEQ_LEN + 8, width
    )
    source_times = np.datetime64("2026-01-01T00:00:00") + (
        np.arange(len(source), dtype=np.int64) * np.timedelta64(5, "m")
    )
    surface = tmp_path / "m5_feature_base.parquet"
    pq.write_table(
        pa.table(
            {
                "time": pa.array(source_times),
                "signal": pa.array(source.tolist()),
                "ctx_cont": pa.array(
                    np.zeros(
                        (len(source), MODEL_NATIVE_CTX_CONT_DIM), dtype=np.float32
                    ).tolist()
                ),
                "ctx_cat": pa.array(
                    np.zeros(
                        (len(source), MODEL_NATIVE_CTX_CAT_DIM), dtype=np.int64
                    ).tolist()
                ),
            }
        ),
        surface,
    )
    surface_manifest = Path(f"{surface}.manifest.json")
    surface_manifest.write_text(
        json.dumps(
            {
                "schema_version": "gx1_entry_exit_m5_feature_surface_v8",
                "output_parquet": str(surface.resolve()),
                "output_parquet_sha256": _sha256(surface),
                "rows": len(source),
                "signal_dim": width,
                "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
                "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
            }
        ),
        encoding="utf-8",
    )
    positions = np.asarray([95, 97, 101], dtype=np.int64)
    sequence = np.stack(
        [source[position - MODEL_NATIVE_SEQ_LEN + 1 : position + 1] for position in positions]
    )
    parquet_path = tmp_path / "source_backed_train.parquet"
    columns = {
        "time": pa.array(source_times[positions]),
        "seq": pa.array(sequence.tolist()),
        "snap": pa.array(source[positions].tolist()),
        "ctx_cont": pa.array(
            np.zeros((len(positions), MODEL_NATIVE_CTX_CONT_DIM), dtype=np.float32).tolist()
        ),
        "ctx_cat": pa.array(
            np.zeros((len(positions), MODEL_NATIVE_CTX_CAT_DIM), dtype=np.int64).tolist()
        ),
        "mae_first_n_bps": pa.array([1.0, 2.0, 3.0]),
        "y_early_move": pa.array([0.0, 1.0, 0.0]),
        "y_quality_score": pa.array([0.2, 0.4, 0.6]),
    }
    for target in trainer._MODEL_NATIVE_ACTIVE_TARGET_COLS:
        values = [0.0, 0.0, 0.0]
        if target == "y_position_size_target":
            values = [0.5, 0.5, 0.5]
        elif target == "y_position_size_mask":
            values = [1.0, 1.0, 0.0]
        columns[target] = pa.array(values)
    pq.write_table(pa.table(columns), parquet_path)
    signal_contract = model_native_signal_contract_metadata(
        canonical_model_native_selected_fields(
            remainder_prefix="session_regime.source_backed_fixture"
        )
    )
    split_manifest = parquet_path.with_suffix(".manifest.json")
    split_manifest.write_text(
        json.dumps(
            {
                "extra": {
                    "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
                    "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
                    "model_native_signal_contract": signal_contract,
                    "aux_head_target_contract": aux_head_target_contract(),
                    "signal_bridge": {
                        "fields": signal_contract["fields"],
                        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                        "seq_structure_extension_v1": {
                            "feature_surface": {
                                "dataset_run_id": "FIXTURE_DATASET_20260826",
                                "inline_split_recomputation": False,
                                "manifest_path": str(surface_manifest.resolve()),
                                "manifest_sha256": _sha256(surface_manifest),
                                "pair_generation_id": "fixture_generation",
                                "path": str(surface.resolve()),
                                "rows": len(source),
                                "schema_version": "gx1_entry_exit_m5_feature_surface_v8",
                                "sha256": _sha256(surface),
                                "signal_manifest_sha256": "1" * 64,
                                "time_alignment": "exact_entry_m5_source_timeline",
                            }
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    proof_path = parquet_path.with_suffix(".sequence_source_audit.json")
    proof_path.write_text(
        json.dumps(
            audit_sequence_source_reconstruction(
                parquet_path=parquet_path.resolve(), manifest_path=split_manifest.resolve()
            ),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return parquet_path, proof_path, source, positions


def test_advanced_dataset_uses_memmap_when_nested_arrays_exceed_threshold(tmp_path, monkeypatch) -> None:
    parquet_path = tmp_path / "advanced_train.parquet"
    memmap_root = tmp_path / "memmap"
    _write_advanced_parquet(parquet_path)

    monkeypatch.setattr(trainer, "_MEMMAP_MIN_BYTES", 0)
    monkeypatch.setattr(trainer, "_MEMMAP_ROOT", memmap_root)
    assert trainer._NESTED_ARROW_BATCH_ROWS == 512
    assert trainer._MEMMAP_WRITEBACK_ROWS == 2048
    monkeypatch.setattr(trainer, "_NESTED_ARROW_BATCH_ROWS", 1)
    monkeypatch.setattr(trainer, "_MEMMAP_WRITEBACK_ROWS", 2)
    flush_calls = 0
    real_flush = trainer._flush_memmap_pages

    def _counting_flush(*arrays) -> None:
        nonlocal flush_calls
        flush_calls += 1
        real_flush(*arrays)

    monkeypatch.setattr(trainer, "_flush_memmap_pages", _counting_flush)
    m5_path = install_multi_tf_stub(tmp_path, monkeypatch)

    ds = trainer.EntryV10CtxDataset(
        parquet_path,
        seq_len=MODEL_NATIVE_SEQ_LEN,
        m5_prebuilt_path=m5_path,
        per_tf_seq_lens=dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS),
        multi_tf_closed_bar=True,
    )

    assert isinstance(ds._np_seq, np.memmap)
    assert isinstance(ds._np_snap, np.memmap)
    assert isinstance(ds._np_ctx_cont, np.memmap)
    assert isinstance(ds._np_ctx_cat, np.memmap)
    assert ds._np_seq.shape == (3, MODEL_NATIVE_SEQ_LEN, MODEL_NATIVE_SIGNAL_DIM)
    assert ds._np_snap.shape == (3, MODEL_NATIVE_SIGNAL_DIM)
    assert ds._np_ctx_cont.shape == (3, MODEL_NATIVE_CTX_CONT_DIM)
    assert ds._np_ctx_cat.shape == (3, MODEL_NATIVE_CTX_CAT_DIM)
    assert len(ds) == 3

    sample = ds[1]

    assert tuple(sample["seq_x"].shape) == (
        MODEL_NATIVE_SEQ_LEN,
        MODEL_NATIVE_SIGNAL_DIM,
    )
    assert tuple(sample["snap_x"].shape) == (MODEL_NATIVE_SIGNAL_DIM,)
    assert tuple(sample["ctx_cont"].shape) == (MODEL_NATIVE_CTX_CONT_DIM,)
    assert tuple(sample["ctx_cat"].shape) == (MODEL_NATIVE_CTX_CAT_DIM,)
    # The retired `y_direction` -> `y` class-tensor conversion is gone with
    # the hierarchy heads; the row identity is the immutable entry row index
    # and no alias target may be manufactured to satisfy a head check.
    assert int(sample["entry_row_index"].item()) == 1
    for forbidden in ("y", "y_direction", "y_teacher_bad_long", "y_teacher_winner_long"):
        assert forbidden not in sample
    assert memmap_root.exists()
    assert flush_calls == 2


def test_advanced_dataset_does_not_delete_preexisting_memmap_scratch(
    tmp_path,
    monkeypatch,
) -> None:
    parquet_path = tmp_path / "advanced_train.parquet"
    memmap_root = tmp_path / "memmap"
    preserved = memmap_root / "prior_run_999999"
    preserved.mkdir(parents=True)
    sentinel = preserved / "retain-until-approved.bin"
    sentinel.write_bytes(b"retention-owner-must-decide")
    _write_advanced_parquet(parquet_path)

    monkeypatch.setattr(trainer, "_MEMMAP_MIN_BYTES", 0)
    monkeypatch.setattr(trainer, "_MEMMAP_ROOT", memmap_root)
    m5_path = install_multi_tf_stub(tmp_path, monkeypatch)

    ds = trainer.EntryV10CtxDataset(
        parquet_path,
        seq_len=MODEL_NATIVE_SEQ_LEN,
        m5_prebuilt_path=m5_path,
        per_tf_seq_lens=dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS),
        multi_tf_closed_bar=True,
    )

    assert isinstance(ds._np_seq, np.memmap)
    assert sentinel.read_bytes() == b"retention-owner-must-decide"


def test_advanced_dataset_reconstructs_sequence_only_from_exact_roll_proof(
    tmp_path,
    monkeypatch,
) -> None:
    parquet_path = tmp_path / "advanced_train.parquet"
    memmap_root = tmp_path / "memmap"
    _write_advanced_parquet(parquet_path, rolling_sequence=True)
    proof_path = _write_sequence_roll_proof(parquet_path)
    monkeypatch.setattr(trainer, "_MEMMAP_MIN_BYTES", 0)
    monkeypatch.setattr(trainer, "_MEMMAP_ROOT", memmap_root)
    m5_path = install_multi_tf_stub(tmp_path, monkeypatch)

    ds = trainer.EntryV10CtxDataset(
        parquet_path,
        seq_len=MODEL_NATIVE_SEQ_LEN,
        m5_prebuilt_path=m5_path,
        per_tf_seq_lens=dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS),
        multi_tf_closed_bar=True,
        sequence_roll_audit_json=proof_path,
    )

    assert ds._sequence_roll_reconstructed is True
    assert not isinstance(ds._np_seq, np.memmap)
    assert not memmap_root.exists()
    assert ds._np_seq.shape == (3, MODEL_NATIVE_SEQ_LEN, MODEL_NATIVE_SIGNAL_DIM)
    assert np.array_equal(ds._np_seq[:, -1, :], ds._np_snap)
    assert np.array_equal(ds._np_seq[1:, :-1, :], ds._np_seq[:-1, 1:, :])

    ds.indices = np.asarray([0, 2], dtype=np.int64)
    ds.compact_materialized_rows(ds.indices)
    assert ds._sequence_reconstruction_chain is None
    assert ds._np_seq.flags.c_contiguous
    assert ds._np_seq.shape == (2, MODEL_NATIVE_SEQ_LEN, MODEL_NATIVE_SIGNAL_DIM)


def test_advanced_dataset_source_reconstruction_handles_filtered_rows(
    tmp_path,
    monkeypatch,
) -> None:
    parquet_path, proof_path, source, positions = _write_source_backed_advanced_fixture(
        tmp_path
    )
    memmap_root = tmp_path / "memmap"
    monkeypatch.setattr(trainer, "_MEMMAP_MIN_BYTES", 0)
    monkeypatch.setattr(trainer, "_MEMMAP_ROOT", memmap_root)
    m5_path = install_multi_tf_stub(tmp_path, monkeypatch)

    ds = trainer.EntryV10CtxDataset(
        parquet_path,
        seq_len=MODEL_NATIVE_SEQ_LEN,
        m5_prebuilt_path=m5_path,
        per_tf_seq_lens=dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS),
        multi_tf_closed_bar=True,
        sequence_source_audit_json=proof_path,
    )

    assert ds._sequence_source_reconstructed is True
    assert ds._np_seq is None
    assert not memmap_root.exists()
    assert np.array_equal(
        ds.sequence_for_full_row(1),
        source[positions[1] - MODEL_NATIVE_SEQ_LEN + 1 : positions[1] + 1],
    )
    original_mtf_window = ds._get_multi_tf_window

    def _readonly_mtf_window(*args, **kwargs):
        windows = original_mtf_window(*args, **kwargs)
        for value in windows.values():
            value.setflags(write=False)
        return windows

    ds._get_multi_tf_window = _readonly_mtf_window
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="The given NumPy array is not writable",
            category=UserWarning,
        )
        assert np.array_equal(
            ds[1]["seq_x"].numpy(), ds.sequence_for_full_row(1)
        )

    ds.indices = np.asarray([0, 2], dtype=np.int64)
    ds.compact_materialized_rows(ds.indices)
    assert ds._np_seq is None
    assert np.array_equal(ds[1]["seq_x"].numpy(), ds.sequence_for_full_row(2))


def test_sequence_roll_reconstruction_rejects_authoritative_proof_claim(
    tmp_path,
) -> None:
    parquet_path = tmp_path / "advanced_train.parquet"
    _write_advanced_parquet(parquet_path, rolling_sequence=True)
    proof_path = _write_sequence_roll_proof(parquet_path)
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["authority"]["candidate"] = True
    proof_path.write_text(json.dumps(proof), encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="ENTRY_SEQUENCE_ROLL_RECONSTRUCTION_PROOF_INVALID",
    ):
        trainer._require_sequence_roll_audit(
            proof_path,
            parquet_path=parquet_path,
            manifest_path=parquet_path.with_suffix(".manifest.json"),
        )


def test_advanced_dataset_rejects_unsorted_time_rows(tmp_path, monkeypatch) -> None:
    parquet_path = tmp_path / "advanced_train.parquet"
    _write_advanced_parquet(
        parquet_path,
        times=[
            "2026-01-02T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-03T00:00:00Z",
        ],
    )
    m5_path = install_multi_tf_stub(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="ENTRY_V10_CTX_ADVANCED_TIME_ORDER_FAIL"):
        trainer.EntryV10CtxDataset(
            parquet_path,
            seq_len=MODEL_NATIVE_SEQ_LEN,
            m5_prebuilt_path=m5_path,
            per_tf_seq_lens=dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS),
            multi_tf_closed_bar=True,
        )


def test_dataset_rejects_missing_timeframe_length_without_global_fallback(
    tmp_path,
    monkeypatch,
) -> None:
    parquet_path = tmp_path / "advanced_train.parquet"
    _write_advanced_parquet(parquet_path)
    m5_path = install_multi_tf_stub(tmp_path, monkeypatch)

    with pytest.raises(
        RuntimeError,
        match="ENTRY_EXIT_PRODUCTION_ARCHITECTURE_MISMATCH",
    ):
        trainer.EntryV10CtxDataset(
            parquet_path,
            seq_len=MODEL_NATIVE_SEQ_LEN,
            m5_prebuilt_path=m5_path,
            per_tf_seq_lens={"M5": 16, "H4": 96, "D1": 252},
            multi_tf_closed_bar=True,
        )


def test_compact_materialized_rows_preserves_original_row_lookup(tmp_path, monkeypatch) -> None:
    parquet_path = tmp_path / "advanced_train.parquet"
    memmap_root = tmp_path / "memmap"
    _write_advanced_parquet(parquet_path)

    monkeypatch.setattr(trainer, "_MEMMAP_MIN_BYTES", 0)
    monkeypatch.setattr(trainer, "_MEMMAP_ROOT", memmap_root)
    m5_path = install_multi_tf_stub(tmp_path, monkeypatch)
    ds = trainer.EntryV10CtxDataset(
        parquet_path,
        seq_len=MODEL_NATIVE_SEQ_LEN,
        m5_prebuilt_path=m5_path,
        per_tf_seq_lens=dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS),
        multi_tf_closed_bar=True,
    )

    ds.indices = np.asarray([0, 2], dtype=np.int64)
    ds.compact_materialized_rows(ds.indices)

    assert not isinstance(ds._np_seq, np.memmap)
    assert ds._np_seq.shape == (
        2,
        MODEL_NATIVE_SEQ_LEN,
        MODEL_NATIVE_SIGNAL_DIM,
    )
    assert np.array_equal(ds._compact_row_indices, np.asarray([0, 2]))
    sample = ds[1]
    assert int(sample["entry_row_index"].item()) == 2
    np.testing.assert_allclose(
        sample["snap_x"].numpy(),
        np.arange(MODEL_NATIVE_SIGNAL_DIM, dtype=np.float32) + 2.0,
    )


def test_liveness_storage_indices_use_compact_array_coordinates() -> None:
    class CompactDataset:
        indices = np.asarray([3, 10_113, 20_000], dtype=np.int64)
        _compact_row_indices = indices.copy()
        _np_seq = np.empty((3, 0), dtype=np.float32)
        _np_snap = np.empty((3, 0), dtype=np.float32)
        _np_ctx_cont = np.empty((3, 0), dtype=np.float32)

    observed = trainer._deterministic_liveness_storage_indices(
        CompactDataset(),
        sample_rows=3,
    )

    np.testing.assert_array_equal(observed, np.asarray([0, 1, 2]))


def test_liveness_storage_indices_preserve_full_array_coordinates() -> None:
    class FullDataset:
        indices = np.asarray([3, 7, 11], dtype=np.int64)
        _compact_row_indices = None
        _np_seq = np.empty((12, 0), dtype=np.float32)
        _np_snap = np.empty((12, 0), dtype=np.float32)
        _np_ctx_cont = np.empty((12, 0), dtype=np.float32)

    observed = trainer._deterministic_liveness_storage_indices(
        FullDataset(),
        sample_rows=3,
    )

    np.testing.assert_array_equal(observed, np.asarray([3, 7, 11]))
