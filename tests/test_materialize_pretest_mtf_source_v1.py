from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.scripts.materialize_pretest_mtf_source_v1 import (
    materialize_pretest_m5_source,
)


def _m1_frame(start: str, periods: int) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="min", tz="UTC")
    return pd.DataFrame(
        {
            "time": index,
            "open": [100.0 + row for row in range(periods)],
            "high": [101.0 + row for row in range(periods)],
            "low": [99.0 + row for row in range(periods)],
            "close": [100.5 + row for row in range(periods)],
            "volume": [1.0] * periods,
        }
    )


def test_materializes_safe_prefix_and_m1_tail_without_test_rows(tmp_path):
    prefix = _m1_frame("2026-06-01T00:00:00Z", 10).iloc[::5].reset_index(drop=True)
    prefix_path = tmp_path / "prefix.parquet"
    pq.write_table(pa.Table.from_pandas(prefix), prefix_path, row_group_size=2)
    tail_dir = tmp_path / "tail"
    tail_dir.mkdir()
    pq.write_table(
        pa.Table.from_pandas(_m1_frame("2026-06-01T00:10:00Z", 10)),
        tail_dir / "xauusd_m1_20260601.parquet",
    )
    output = tmp_path / "safe_m5.parquet"
    manifest = tmp_path / "safe_m5.manifest.json"

    report = materialize_pretest_m5_source(
        prefix_parquet=prefix_path,
        prefix_end_inclusive="2026-06-01T00:05:00Z",
        m1_tail_dir=tail_dir,
        test_start_utc="2026-06-02T00:00:00Z",
        output_parquet=output,
        output_manifest=manifest,
    )

    assert report["test_accessed"] is False
    result = pd.read_parquet(output)
    assert result["time"].max() < pd.Timestamp("2026-06-02T00:00:00Z")
    assert result["time"].tolist() == list(
        pd.date_range("2026-06-01T00:00:00Z", periods=4, freq="5min")
    )


def test_rejects_a_prefix_row_group_that_crosses_the_safe_boundary(tmp_path):
    prefix_path = tmp_path / "crossing.parquet"
    pq.write_table(
        pa.Table.from_pandas(_m1_frame("2026-06-01T00:00:00Z", 15).iloc[::5]),
        prefix_path,
        row_group_size=3,
    )
    tail_dir = tmp_path / "tail"
    tail_dir.mkdir()

    with pytest.raises(RuntimeError, match="PREFIX_ROW_GROUP_CROSSES_BOUNDARY"):
        materialize_pretest_m5_source(
            prefix_parquet=prefix_path,
            prefix_end_inclusive="2026-06-01T00:05:00Z",
            m1_tail_dir=tail_dir,
            test_start_utc="2026-06-02T00:00:00Z",
            output_parquet=tmp_path / "should_not_exist.parquet",
            output_manifest=tmp_path / "should_not_exist.json",
        )
