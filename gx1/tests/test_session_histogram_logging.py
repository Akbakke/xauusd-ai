from __future__ import annotations

import pandas as pd

from gx1.scripts.build_entry_v10_ctx_training_dataset import compute_session_histogram


def test_session_histogram_counts_sum_to_n_rows() -> None:
    ts = pd.to_datetime(
        [
            "2025-01-02T01:00:00Z",
            "2025-01-02T02:00:00Z",
            "2025-01-02T10:00:00Z",
            "2025-01-02T16:00:00Z",
        ],
        utc=True,
    )
    df = pd.DataFrame(
        {
            "ts": ts,
            # session_id here is intentionally non-categorical to ensure we fall back to timestamp inference
            "session_id": [0.1, 0.2, 0.3, 0.4],
        }
    )

    h = compute_session_histogram(df, ts_col="ts", session_col="session_id")

    assert "n_rows" in h
    assert "counts" in h
    assert "pct" in h
    assert "counts_sum" in h

    assert h["n_rows"] == len(df)
    assert h["counts_sum"] == len(df)
    assert abs(sum(h["pct"].values()) - 100.0) < 1e-6

