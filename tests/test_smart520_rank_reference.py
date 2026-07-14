from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.scripts.materialize_smart520_rank_reference_v1 import run


def test_materialize_smart520_rank_reference_writes_live_state_npz(tmp_path: Path) -> None:
    source = tmp_path / "FULL_PLUS_CTX_v3src.parquet"
    times = pd.date_range("2026-05-21T00:00:00Z", periods=8, freq="5min")
    pd.DataFrame(
        {
            "time": times,
            "atr": np.linspace(1.0, 1.7, len(times)),
            "atr_bps": np.linspace(10.0, 17.0, len(times)),
            "spread_bps": np.linspace(0.5, 1.2, len(times)),
            "vol_regime_id": [0, 1, 2, 3, 4, 2, 3, 4],
            "spread_bucket": [0, 0, 1, 2, 3, 4, 4, 4],
        }
    ).to_parquet(source, index=False)

    out = tmp_path / "smart520_rank_reference_xau_direction_repair.npz"
    report = run(
        argparse.Namespace(
            source_parquet=source,
            out=out,
            model_range_start="2026-05-21T00:00:00Z",
            reference_end="2026-05-21T00:35:00Z",
            min_rows=1,
        )
    )

    assert report["row_count"] == len(times)
    assert out.is_file()
    ref = np.load(out)
    assert {"time_ns", "vol_regime_id", "spread_bucket", "atr_pinned", "atr_bps_sorted", "spread_bps_sorted"} <= set(ref.files)
    assert ref["time_ns"].shape == (len(times),)
    assert ref["atr_pinned"].shape == (len(times),)
    sidecar = json.loads(out.with_suffix(out.suffix + ".json").read_text(encoding="utf-8"))
    assert sidecar["schema_version"] == "smart520_rank_reference_v1"
    assert sidecar["out_npz"] == str(out)
