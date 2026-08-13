from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.contracts.entry_full_input_liveness_v1 import (
    PASS_DECISION,
    SPLITS,
    validate_full_input_liveness_artifact,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTEXT_TAG,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
)
from gx1.scripts.materialize_entry_full_input_liveness_v1 import (
    run,
)
from gx1.features.htf_features import (
    HTF_V4_MATRIX_CONTRACT,
    MULTI_TF_PER_BAR_FEATURES_V4,
    MULTI_TF_RESAMPLE_RULES,
    build_multi_tf_v4_liveness_contract,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields


RUN_ID = "UNIT_LIVENESS_20260717"
STEM = "unit_seq513__DIR_H24B"
OUTPUT_FILENAME = (
    "ENTRY_FULL_INPUT_LIVENESS_CONTRACT_20260717T120000000000Z.json"
)


def _v4_frames() -> dict[str, pd.DataFrame]:
    out = {}
    rows = 32
    row = np.arange(rows, dtype=np.float32)[:, None]
    col = np.arange(
        len(MULTI_TF_PER_BAR_FEATURES_V4), dtype=np.float32
    )[None, :]
    values = row + col * np.float32(0.001)
    for timeframe in MULTI_TF_RESAMPLE_RULES:
        frame = pd.DataFrame(
            values,
            columns=MULTI_TF_PER_BAR_FEATURES_V4,
        )
        frame.attrs["feats_np"] = values.copy()
        frame.attrs["causal_warmup_rows"] = 1
        frame.attrs["htf_feature_contract"] = HTF_V4_MATRIX_CONTRACT
        out[timeframe] = frame
    return out


@pytest.fixture(autouse=True)
def _verified_v4_cache_loader(monkeypatch) -> None:
    def load(_cache_dir):
        return _v4_frames()

    monkeypatch.setattr(
        "gx1.scripts.materialize_entry_full_input_liveness_v1."
        "load_multi_tf_v4_cache",
        load,
    )


def _signal_contract() -> dict:
    required = [
        "candle.pattern_outside_after_inside_bull_breakout_score",
        "candle.pattern_outside_after_inside_bear_breakout_score",
        "ctx_cont.d1_atr14_canon_v2",
        "ctx_cont._v1h4_atr",
    ]
    selected = canonical_model_native_selected_fields(
        required_fields=required,
        remainder_prefix="session_regime.full_input_liveness_fixture",
    )
    return model_native_signal_contract_metadata(selected)


def _write_split(
    dataset_dir: Path,
    *,
    split: str,
    rows: int,
    signal_contract: dict,
    break_seq_snap_parity: bool = False,
) -> None:
    parquet_path = dataset_dir / f"{STEM}_{split}.parquet"
    # Keep the same distribution on every split so the ATR OOD contract is
    # green even though rare-event support uses different row counts.
    row_axis = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None]
    signal_axis = np.arange(MODEL_NATIVE_SIGNAL_DIM, dtype=np.float32)[None, :]
    snap = 1.0 + row_axis * 0.25 + signal_axis * 0.001
    seq = np.repeat(snap[:, None, :], MODEL_NATIVE_SEQ_LEN, axis=1)
    if break_seq_snap_parity:
        seq[0, -1, 0] += np.float32(1.0)
    cont_axis = np.arange(len(MODEL_NATIVE_CTX_CONT_FIELDS), dtype=np.float32)[None, :]
    ctx_cont = 2.0 + row_axis * 0.125 + cont_axis * 0.001
    ctx_cat = np.column_stack(
        [
            (np.arange(rows, dtype=np.int64) + index) % 3
            for index in range(len(MODEL_NATIVE_CTX_CAT_FIELDS))
        ]
    )
    table = pa.table(
        {
            "seq": seq.tolist(),
            "snap": snap.tolist(),
            "ctx_cont": ctx_cont.tolist(),
            "ctx_cat": ctx_cat.tolist(),
        }
    )
    pq.write_table(table, parquet_path)

    ctx_contract = {
        "tag": MODEL_NATIVE_CONTEXT_TAG,
        "ctx_cont_dim": len(MODEL_NATIVE_CTX_CONT_FIELDS),
        "ctx_cat_dim": len(MODEL_NATIVE_CTX_CAT_FIELDS),
        "ctx_cont_names": list(MODEL_NATIVE_CTX_CONT_FIELDS),
        "ctx_cat_names": list(MODEL_NATIVE_CTX_CAT_FIELDS),
    }
    manifest = {
        "schema_version": MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
        "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
        "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
        "output_data_path": str(parquet_path.resolve()),
        "extra": {
            "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
            "entry_run_id": RUN_ID,
            "model_native_state_contract": {"entry_run_id": RUN_ID},
            "model_native_signal_contract": signal_contract,
            "signal_bridge": {
                "id": MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
                "fields": signal_contract["fields"],
                "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                "bridge_dim": 0,
                "bridge_source": None,
            },
            "ctx_contract": ctx_contract,
        },
    }
    (dataset_dir / f"{STEM}_{split}.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_dataset(dataset_dir: Path, *, break_scanned_parity: bool = False) -> None:
    dataset_dir.mkdir(parents=True)
    signal_contract = _signal_contract()
    # Liveness scans only SPLITS; TEST stays sealed, so a corruption that must
    # be caught is injected into the last scanned split.
    corrupt_split = SPLITS[-1]
    for split, rows in (("train", 32), ("val", 8), ("test", 8)):
        _write_split(
            dataset_dir,
            split=split,
            rows=rows,
            signal_contract=signal_contract,
            break_seq_snap_parity=break_scanned_parity and split == corrupt_split,
        )
    build_proof = {
        "entry_run_id": RUN_ID,
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "output_path": str((dataset_dir / f"{STEM}.parquet").resolve()),
        "model_native_signal_contract": signal_contract,
        "ctx_contract": {
            "ctx_cont_names": list(MODEL_NATIVE_CTX_CONT_FIELDS),
            "ctx_cat_names": list(MODEL_NATIVE_CTX_CAT_FIELDS),
        },
        "model_native_state_contract": {"entry_run_id": RUN_ID},
    }
    (dataset_dir / "DATASET_BUILD_PROOF.json").write_text(
        json.dumps(build_proof, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _args(dataset_dir: Path, out_dir: Path) -> argparse.Namespace:
    mtf_cache_dir = dataset_dir.parent / "MULTI_TF_CACHE"
    mtf_cache_dir.mkdir(exist_ok=True)
    (mtf_cache_dir / "manifest.json").write_text(
        json.dumps(
            {
                "cache_identity_sha256": "b" * 64,
                "full_input_liveness": build_multi_tf_v4_liveness_contract(
                    _v4_frames()
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return argparse.Namespace(
        run_id=RUN_ID,
        dataset_dir=str(dataset_dir),
        stem=STEM,
        mtf_cache_dir=str(mtf_cache_dir),
        out_json=str(out_dir / OUTPUT_FILENAME),
        batch_size=16,
        quiet=True,
    )


def test_materializer_fullscans_and_binds_exact_seq513_ctx142_5(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)

    artifact = run(_args(dataset_dir, tmp_path / "audit"))
    artifact_path = tmp_path / "audit" / OUTPUT_FILENAME
    validation = validate_full_input_liveness_artifact(
        artifact_path,
        expected_dataset_dir=dataset_dir,
    )

    assert artifact["decision"] == PASS_DECISION
    assert validation["ok"] is True
    assert validation["field_counts"] == {
        "signal": MODEL_NATIVE_SIGNAL_DIM,
        "ctx_cont": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat": 5,
    }
    assert validation["field_status_row_count"] == len(SPLITS) * (
        MODEL_NATIVE_SIGNAL_DIM + MODEL_NATIVE_CTX_CONT_DIM + 5
    )
    from gx1.features.htf_features import MULTI_TF_FEATURE_COUNT_V4

    assert validation["multi_tf_field_status_row_count"] == (
        5 * MULTI_TF_FEATURE_COUNT_V4
    )
    provenance = artifact["materializer_provenance"]
    assert provenance["entry_run_id"] == RUN_ID
    assert len(provenance["dataset_build_proof"]["sha256"]) == 64
    assert all(
        row["seq_last_exactly_equals_snap"] and row["scan_complete"]
        for row in provenance["semantic_fullscan"].values()
    )


def test_materializer_fails_closed_on_seq_history_not_matching_snap(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, break_scanned_parity=True)

    artifact = run(_args(dataset_dir, tmp_path / "audit"))

    corrupt_split = SPLITS[-1]
    assert artifact["decision"] == "FAIL"
    assert artifact["materializer_provenance"]["semantic_fullscan"][corrupt_split][
        "seq_last_exactly_equals_snap"
    ] is False
    assert any(
        row["code"] == "fullscan_proof_invalid" and row["split"] == corrupt_split
        for row in artifact["failures"]
    )


def test_materializer_rejects_run_id_mismatch_before_writing(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    manifest_path = dataset_dir / f"{STEM}_val.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["extra"]["entry_run_id"] = "DIFFERENT_RUN_ID_20260717"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out_dir = tmp_path / "audit"

    with pytest.raises(RuntimeError, match="SPLIT_RUN_ID_MISMATCH"):
        run(_args(dataset_dir, out_dir))

    assert not (out_dir / OUTPUT_FILENAME).exists()
