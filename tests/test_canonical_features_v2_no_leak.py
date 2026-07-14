import numpy as np
import pandas as pd
from pathlib import Path

from gx1.scripts.materialize_build_canonical_features_v2 import (
    compute_d1_features,
    compute_m15_features,
    merge_asof_features,
)


def _m5_frame(start: str, periods: int) -> pd.DataFrame:
    time = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    close = 100.0 + np.arange(periods, dtype=np.float64)
    return pd.DataFrame(
        {
            "time": time,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        }
    )


def test_canonical_v2_htf_feature_times_are_close_times_not_bucket_starts() -> None:
    m5 = _m5_frame("2026-01-01T00:00:00Z", periods=24 * 12 * 3)

    d1 = compute_d1_features(m5)
    m15 = compute_m15_features(m5)

    assert pd.Timestamp(d1["time"].iloc[0]) == pd.Timestamp("2026-01-02T00:00:00Z")
    assert pd.Timestamp(m15["time"].iloc[0]) == pd.Timestamp("2026-01-01T00:15:00Z")


def test_merge_asof_features_only_exposes_closed_htf_rows() -> None:
    base = _m5_frame("2026-01-01T00:00:00Z", periods=4)
    close_time = pd.Timestamp("2026-01-01T00:15:00Z")
    extra = pd.DataFrame(
        {
            "time": [close_time],
            "_time_ns": [int(close_time.value)],
            "m15_marker": [1.0],
        }
    )

    merged = merge_asof_features(base, extra)

    assert merged.loc[0:2, "m15_marker"].isna().all()
    assert merged.loc[3, "m15_marker"] == 1.0


def test_canonical_v2_summary_declares_no_lookahead_contract() -> None:
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "gx1/scripts/materialize_build_canonical_features_v2.py").read_text(encoding="utf-8")

    assert "canonical_features_v2_no_lookahead_close_time_20260713" in text
    assert '"no_lookahead": True' in text
    assert '"d1_feature_time": "bar_close_time"' in text
    assert '"m15_feature_time": "bar_close_time"' in text


def test_canonical_v3_manifest_records_source_v2_no_lookahead_provenance() -> None:
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "gx1/scripts/materialize_canonical_v3_augment.py").read_text(encoding="utf-8")

    assert "source_v2_parquet_sha256" in text
    assert "source_v2_no_lookahead" in text
    assert "source_v2_htf_alignment_contract" in text
