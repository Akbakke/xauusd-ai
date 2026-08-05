from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.replay.source_tape_v1 import SourceTape


def _source_frame(times: list[pd.Timestamp]) -> pd.DataFrame:
    values = np.linspace(100.0, 101.0, len(times))
    return pd.DataFrame(
        {
            "time": times,
            "open": values + 0.01,
            "high": values + 0.06,
            "low": values - 0.04,
            "close": values + 0.01,
            "bid_open": values,
            "ask_open": values + 0.02,
            "bid_close": values,
            "ask_close": values + 0.02,
            "bid_high": values + 0.05,
            "bid_low": values - 0.05,
            "ask_high": values + 0.07,
            "ask_low": values - 0.03,
            "volume": np.arange(len(times), dtype=np.int64) + 1,
        }
    )


def test_source_tape_exposes_hash_bound_closed_m1_provider(tmp_path: Path) -> None:
    times = list(pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="min"))
    source_path = tmp_path / "source.parquet"
    _source_frame(times).to_parquet(source_path, index=False)
    tape = SourceTape.load(source_path)

    bar = tape.get_closed_m1_bar(pd.Timestamp(times[1]))
    quote = tape.get_open_quote(pd.Timestamp(times[1]))
    assert tape.source_binding["path"] == str(source_path.resolve())
    assert len(tape.source_binding["sha256"]) == 64
    assert tape.source_binding["size_bytes"] == source_path.stat().st_size
    assert pd.Timestamp(bar["time"]) == pd.Timestamp(times[1])
    assert bar["schema_version"] == "gx1_closed_m1_literal_mba_path_v1"
    assert bar["source_sha256"] == tape.source_sha256
    assert bar["mid_close"] == pytest.approx(tape.mid_close[1])
    assert bar["bid_close"] == pytest.approx(tape.bid_close[1])
    assert bar["volume"] == int(tape.volume[1])
    assert quote["time"] == pd.Timestamp(times[1])
    assert quote["bid"] == pytest.approx(tape.bid_open[1])
    assert quote["ask"] == pytest.approx(tape.ask_open[1])

    missing = pd.Timestamp("2026-01-01T00:10:00Z")
    with pytest.raises(RuntimeError, match="lacks exact closed M1 bar"):
        tape.get_closed_m1_bar(missing)
    with pytest.raises(RuntimeError, match="lacks exact open quote"):
        tape.get_open_quote(missing)


def test_source_tape_resolves_observed_m5_horizon_not_m1_row_count(
    tmp_path: Path,
) -> None:
    times = list(pd.date_range("2026-01-01T00:00:00Z", periods=15, freq="min"))
    source_path = tmp_path / "source.parquet"
    _source_frame(times).to_parquet(source_path, index=False)
    tape = SourceTape.load(source_path)

    decision_indices = tape.indices_for_times(pd.Series([times[0], times[5]]))
    start, end = tape.label_horizon_indices(
        decision_time=pd.Timestamp(times[0]),
        horizon_m5_bars=2,
    )
    assert decision_indices.tolist() == [0, 5]
    assert start == 5
    assert end == 14
    assert pd.Timestamp(tape.times[start]) == pd.Timestamp(times[5])
    assert pd.Timestamp(tape.times[end]) == pd.Timestamp(times[14])


def test_source_tape_rejects_unsorted_rows(tmp_path: Path) -> None:
    times = list(pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="min"))
    source = _source_frame(times).iloc[[1, 0, 2]].reset_index(drop=True)
    source_path = tmp_path / "unsorted.parquet"
    source.to_parquet(source_path, index=False)
    with pytest.raises(RuntimeError, match="not strictly chronological"):
        SourceTape.load(source_path)


def test_source_tape_rejects_price_scale_corruption(tmp_path: Path) -> None:
    times = list(pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="min"))
    source = _source_frame(times)
    price_columns = [name for name in source if name not in {"time", "volume"}]
    source.loc[1:2, price_columns] /= 10.0
    source_path = tmp_path / "scale_corrupt.parquet"
    source.to_parquet(source_path, index=False)
    with pytest.raises(RuntimeError, match="PRICE_SCALE_GLITCH"):
        SourceTape.load(source_path)
