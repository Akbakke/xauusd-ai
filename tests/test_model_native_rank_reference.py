from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_state_v2 import (
    FORBIDDEN_ROW_LEVEL_RANK_KEYS,
    MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
    TRAIN_RANK_NPZ_KEYS,
    load_train_rank_reference_v2,
    sha256_file,
)
from gx1.contracts.entry_run_lineage_v1 import EntryRunLineageError
from gx1.scripts.materialize_model_native_train_rank_reference_v2 import run


def _market_frame(*, mutate_post_fit: bool = False) -> pd.DataFrame:
    times = pd.date_range("2026-05-21T00:00:00Z", periods=30, freq="5min")
    close = 2300.0 + np.linspace(0.0, 5.8, len(times)) + np.sin(np.arange(len(times))) * 0.2
    high = close + 0.8
    low = close - 0.7
    bid = close - 0.06
    ask = close + 0.06
    if mutate_post_fit:
        close[20:] += 150.0
        high[20:] = close[20:] + 8.0
        low[20:] = close[20:] - 7.0
        bid[20:] = close[20:] - 2.0
        ask[20:] = close[20:] + 2.0
    return pd.DataFrame(
        {
            "time": times,
            "high": high,
            "low": low,
            "close": close,
            "bid_close": bid,
            "ask_close": ask,
        }
    )


def _materialize(source: Path, out: Path) -> dict:
    return run(
        argparse.Namespace(
            source_parquet=source,
            out=out,
            history_start="2026-05-21T00:00:00Z",
            fit_start="2026-05-21T00:25:00Z",
            fit_end="2026-05-21T01:35:00Z",
            min_rows=1,
            run_id="MODEL_NATIVE_RANK_REFERENCE_PYTEST",
        )
    )


def test_materialize_train_rank_reference_stores_only_train_distributions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fresh_xau_source.parquet"
    _market_frame().to_parquet(source, index=False)
    out = tmp_path / "model_native_train_rank_reference_v2.npz"

    report = _materialize(source, out)

    assert report["schema_version"] == MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION
    assert report["fit_scope"] == "train_only"
    assert report["fit_row_count"] == 15
    assert report["validation_or_test_rows_stored"] is False
    assert report["row_level_state_present"] is False
    assert report["entry_run_id"] == "MODEL_NATIVE_RANK_REFERENCE_PYTEST"
    with np.load(out, allow_pickle=False) as payload:
        assert set(payload.files) == set(TRAIN_RANK_NPZ_KEYS)
        assert not (set(payload.files) & set(FORBIDDEN_ROW_LEVEL_RANK_KEYS))
        assert payload["atr_bps_sorted"].shape == (15,)
        assert payload["spread_bps_sorted"].shape == (15,)
    reference = load_train_rank_reference_v2(out)
    assert reference.fit_row_count == 15
    sidecar = json.loads(out.with_suffix(out.suffix + ".json").read_text(encoding="utf-8"))
    assert sidecar["out_npz"] == str(out.resolve())
    assert sidecar["entry_run_id"] == "MODEL_NATIVE_RANK_REFERENCE_PYTEST"


def test_materialize_train_rank_reference_requires_run_identity_before_writes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    _market_frame().to_parquet(source, index=False)
    out = tmp_path / "rank.npz"

    with pytest.raises(EntryRunLineageError, match="provide --run-id"):
        run(
            argparse.Namespace(
                source_parquet=source,
                out=out,
                history_start="2026-05-21T00:00:00Z",
                fit_start="2026-05-21T00:25:00Z",
                fit_end="2026-05-21T01:35:00Z",
                min_rows=1,
            )
        )

    assert not out.exists()
    assert not out.with_suffix(".npz.json").exists()


def test_post_fit_rows_cannot_change_train_rank_distributions(tmp_path: Path) -> None:
    source_a = tmp_path / "source_a.parquet"
    source_b = tmp_path / "source_b.parquet"
    _market_frame(mutate_post_fit=False).to_parquet(source_a, index=False)
    _market_frame(mutate_post_fit=True).to_parquet(source_b, index=False)
    out_a = tmp_path / "rank_a.npz"
    out_b = tmp_path / "rank_b.npz"

    _materialize(source_a, out_a)
    _materialize(source_b, out_b)
    ref_a = load_train_rank_reference_v2(out_a)
    ref_b = load_train_rank_reference_v2(out_b)

    np.testing.assert_array_equal(ref_a.atr_bps_sorted, ref_b.atr_bps_sorted)
    np.testing.assert_array_equal(ref_a.spread_bps_sorted, ref_b.spread_bps_sorted)


def test_train_rank_loader_rejects_row_level_state_key(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _market_frame().to_parquet(source, index=False)
    out = tmp_path / "rank.npz"
    _materialize(source, out)
    with np.load(out, allow_pickle=False) as payload:
        values = {name: np.asarray(payload[name]) for name in payload.files}
    values["time_ns"] = np.asarray([1], dtype=np.int64)
    np.savez_compressed(out, **values)
    sidecar_path = out.with_suffix(out.suffix + ".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["out_npz_sha256"] = sha256_file(out)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(RuntimeError, match="TRAIN_RANK_KEYS_INVALID"):
        load_train_rank_reference_v2(out)


def test_train_rank_loader_rejects_npz_run_id_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _market_frame().to_parquet(source, index=False)
    out = tmp_path / "rank.npz"
    _materialize(source, out)
    with np.load(out, allow_pickle=False) as payload:
        values = {name: np.asarray(payload[name]) for name in payload.files}
    values["entry_run_id"] = np.asarray(["DIFFERENT_EXPLICIT_RUN_ID"])
    np.savez_compressed(out, **values)
    sidecar_path = out.with_suffix(out.suffix + ".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["out_npz_sha256"] = sha256_file(out)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(RuntimeError, match="NPZ_RUN_ID_MISMATCH"):
        load_train_rank_reference_v2(out)
