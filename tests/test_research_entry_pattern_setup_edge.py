"""Mechanics tests for the rule-based setup-edge evaluator (synthetic, rule 2c)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.scripts import research_entry_pattern_setup_edge_v1 as se


def test_declared_setups_are_unique_and_sided() -> None:
    setups = se.declared_setups()
    names = [s.name for s in setups]
    assert len(names) == len(set(names)) and all(s.side in ("LONG", "SHORT") for s in setups)


def test_score_setup_detects_planted_edge_and_reports_nulls() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    long_all = rng.normal(-1.5, 20.0, n)
    short_all = -long_all - 3.0
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, 200, replace=False)] = True  # 200 rows at random (a periodic mask would let circular shifts land on the planted rows)
    long_all[mask] += 25.0  # planted LONG edge on the setup rows
    short_all[mask] = -long_all[mask] - 3.0
    valid = np.ones(n, dtype=bool)
    row = se.score_setup(name="s", side="LONG", mask=mask, long_all=long_all, short_all=short_all, valid=valid, period="p", horizon=12)
    assert row["n"] == 200 and row["mean_pnl_bps"] > 15 and row["strict_pass"] is True
    # a one-sided setup can never beat "always its own side on the same rows": the best-constant null is a tie by construction
    assert row["best_constant_side_mean_pnl_bps"] == row["mean_pnl_bps"] and row["beats_best_constant"] is False
    assert row["excess_over_best_constant_bps"] == 0.0
    row2 = se.score_setup(name="s", side="SHORT", mask=mask, long_all=long_all, short_all=short_all, valid=valid, period="p", horizon=12)
    assert row2["mean_pnl_bps"] < 0 and row2["strict_pass"] is False and row2["beats_best_constant"] is False
    assert row2["excess_over_best_constant_bps"] < -30
    few = se.score_setup(name="s", side="LONG", mask=mask[:100], long_all=long_all[:100], short_all=short_all[:100], valid=valid[:100], period="p", horizon=12)
    assert few["failure_reason"] == "insufficient_setup_rows"


def test_run_end_to_end_synthetic(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = 12000
    time = pd.date_range("2024-01-01T00:00:00Z", periods=rows, freq="5min")
    rng = np.random.default_rng(2)
    mid = 2000.0 + np.cumsum(rng.normal(0.0, 0.5, rows))
    tape_dir = tmp_path / "tape"
    (tape_dir / "year=2024").mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({"time": time, "close": mid, "bid_close": mid - 0.15, "ask_close": mid + 0.15}), preserve_index=False), tape_dir / "year=2024" / "p.parquet")
    (tape_dir / "MANIFEST.json").write_text("{}", encoding="utf-8")
    ds_dir = tmp_path / "ds"
    ds_dir.mkdir()
    train_end_idx = 10000
    manifest = {"splits": {"train": {"start": str(time[0]), "end": str(time[train_end_idx - 1])}, "val": {"start": str(time[train_end_idx]), "end": str(time[-1] + pd.Timedelta(minutes=5))}}}
    (ds_dir / "entry_dataset__ENTRY_FITTED_Q_train.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    prim = pd.DataFrame({"time": time})
    needed = set()
    for s in se.declared_setups():
        try:
            s.rule(pd.DataFrame({"time": time}))
        except RuntimeError as exc:
            needed.add(str(exc).split(": ")[1])
    # collect every primitive column the rules reference by probing until no error
    cols: dict[str, np.ndarray] = {}
    while True:
        frame = pd.DataFrame({"time": time, **cols})
        missing = None
        for s in se.declared_setups():
            try:
                s.rule(frame)
            except RuntimeError as exc:
                missing = str(exc).split(": ")[1]
                break
        if missing is None:
            break
        cols[missing] = (rng.uniform(size=rows) < 0.02).astype(float) if "event" in missing else rng.choice([-1.0, 0.0, 1.0], size=rows)
    prim = pd.DataFrame({"time": time, **cols})
    prim_path = tmp_path / "prim.parquet"
    prim.to_parquet(prim_path, index=False)
    out = tmp_path / "out"
    args = se.build_parser().parse_args([
        "--pattern-primitives-parquet", str(prim_path), "--dataset-dir", str(ds_dir), "--native-m5-root", str(tape_dir),
        "--horizons", "12", "--period-boundaries", str(time[5000]), str(time[train_end_idx]), "--out-dir", str(out),
    ])
    report = se.run(args)
    metrics = pd.read_csv(out / "metrics.csv")
    assert set(metrics["period"]) == {f"{time[0].date()}..{time[5000].date()}", f"{time[5000].date()}..{time[train_end_idx].date()}", "final_val"}
    assert len(report["setups"]) == len(se.declared_setups()) and (out / "summary.md").is_file()
    assert "strict_pass" in metrics.columns and "beats_best_constant" in metrics.columns
