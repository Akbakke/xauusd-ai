"""Mechanics tests for the Entry direction walk-forward research instrument.

Synthetic fixtures prove fold purge, decision semantics, the evaluator
integration (strict_pass requires mean_pnl_bps > 0) and the fail-closed MTF
join validation.  None of this is market evidence (CLAUDE.md rule 2c).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.features.htf_features import MULTI_TF_SHIFT
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_FLAT_INDEX,
    MODEL_DIRECTION_LONG_INDEX,
    MODEL_DIRECTION_SELECTION_MODE,
    MODEL_DIRECTION_SHORT_INDEX,
)
from gx1.scripts import research_entry_direction_walkforward_v1 as wf
from gx1.scripts.evaluate_entry_candidate_selective_edge_v1 import (
    RESEARCH_LONG_OUTCOME_COLUMN,
    RESEARCH_SHORT_OUTCOME_COLUMN,
)


def _tape(rows: int, *, seed: int = 0) -> wf.Tape:
    rng = np.random.default_rng(seed)
    time = pd.date_range("2024-01-01T00:00:00Z", periods=rows, freq="5min")
    mid = 2000.0 + np.cumsum(rng.normal(0.0, 0.5, rows))
    spread = 0.3
    return wf.Tape(time=pd.DatetimeIndex(time), mid=mid, bid=mid - spread / 2, ask=mid + spread / 2, manifest_sha256="0" * 64, root="synthetic")


def test_executable_targets_are_mirrors_minus_costs_and_nan_off_tape() -> None:
    tape = _tape(200)
    positions = np.arange(0, 200, dtype=np.int64)
    long_bps, short_bps, valid = wf.executable_horizon_targets(tape, positions, 12)
    assert valid.sum() == 188 and np.isnan(long_bps[188:]).all()
    # long buys ask now sells bid later; short mirrors it; both pay the spread
    p, f = 5, 17
    assert long_bps[p] == pytest.approx((tape.bid[f] / tape.ask[p] - 1) * 1e4)
    assert short_bps[p] == pytest.approx((1 - tape.ask[f] / tape.bid[p]) * 1e4)
    assert (long_bps[:188] + short_bps[:188] < 0).all()


def test_fold_purge_keeps_every_fit_outcome_before_first_holdout_row() -> None:
    tape = _tape(3000)
    dataset_time = tape.time
    positions = np.arange(3000, dtype=np.int64)
    boundaries = [pd.Timestamp("2024-01-05T00:00:00Z"), pd.Timestamp("2024-01-08T00:00:00Z")]
    folds = wf.build_folds(boundaries, pd.Timestamp("2024-01-01T00:00:00Z"))
    assert len(folds) == 1
    fit, hold = wf.fold_masks(dataset_time, positions, folds[0], purge_bars=25)
    first_hold = int(positions[hold].min())
    assert (positions[fit] + 25 < first_hold).all()
    assert fit.sum() == first_hold - 25
    assert not (fit & hold).any()


def test_build_folds_rejects_unordered_or_early_boundaries() -> None:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    with pytest.raises(RuntimeError, match="FOLD_BOUNDARIES_INVALID"):
        wf.build_folds([pd.Timestamp("2024-02-01T00:00:00Z"), pd.Timestamp("2024-01-15T00:00:00Z")], start)
    with pytest.raises(RuntimeError, match="FIRST_BOUNDARY_NOT_AFTER_TRAIN_START"):
        wf.build_folds([start, pd.Timestamp("2024-02-01T00:00:00Z")], start)


def test_decisions_argmax_flat_and_contrast_rules() -> None:
    long_pred = np.array([3.0, -1.0, 2.0, 0.5, -2.0])
    short_pred = np.array([1.0, -3.0, 2.0, 0.5, 1.0])
    side, score, contrast = wf.decisions(long_pred, short_pred, "argmax_flat")
    assert side.tolist() == [MODEL_DIRECTION_LONG_INDEX, MODEL_DIRECTION_FLAT_INDEX, MODEL_DIRECTION_FLAT_INDEX, MODEL_DIRECTION_FLAT_INDEX, MODEL_DIRECTION_SHORT_INDEX]
    assert score.tolist() == [3.0, 0.0, 0.0, 0.0, 1.0]
    side2, score2, _ = wf.decisions(long_pred, short_pred, "contrast_always_trade")
    assert side2.tolist() == [MODEL_DIRECTION_LONG_INDEX, MODEL_DIRECTION_LONG_INDEX, MODEL_DIRECTION_FLAT_INDEX, MODEL_DIRECTION_FLAT_INDEX, MODEL_DIRECTION_SHORT_INDEX]
    assert np.allclose(score2, np.abs(contrast))
    with pytest.raises(RuntimeError, match="DECISION_RULE_INVALID"):
        wf.decisions(long_pred, short_pred, "nope")


def test_evaluate_frame_strict_pass_requires_positive_mean() -> None:
    rows = 2000
    rng = np.random.default_rng(1)
    time = pd.date_range("2024-01-01T00:00:00Z", periods=rows, freq="5min")
    # an informative selector on a negative-sum market: excess over coin flip is
    # large and significant (primary_pass) while the realized mean stays < 0.
    signal = rng.normal(0.0, 1.0, rows)
    long_out = 4.0 * signal - 8.0 + rng.normal(0.0, 4.0, rows)
    short_out = -4.0 * signal - 8.0 + rng.normal(0.0, 4.0, rows)
    side = np.where(signal > 0, MODEL_DIRECTION_LONG_INDEX, MODEL_DIRECTION_SHORT_INDEX)
    score = np.abs(signal)
    frame = pd.DataFrame(
        {
            "split": "fold0",
            "model": "informative_negative_sum",
            "time": time,
            "pred_direction": side,
            "selection_score": score,
            "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
            "edge_score": long_out - short_out,
            RESEARCH_LONG_OUTCOME_COLUMN: long_out,
            RESEARCH_SHORT_OUTCOME_COLUMN: short_out,
        }
    )
    fold = wf.Fold(index=0, fit_start=time[0], holdout_start=time[0], holdout_end=time[-1])
    month_sign = pd.Series({pd.Period("2024-01", "M"): -1.0})
    meta = {"target": "t", "horizon_bars": 1, "feature_arm": "a", "target_scaling": "raw", "learner": "ridge", "seed": 0, "decision_rule": "argmax_flat", "fit_rows": 1, "outcome_definition": "synthetic"}
    out = wf.evaluate_frame(frame, fold=fold, month_sign=month_sign, meta=meta)
    assert len(out) == 7
    full = next(r for r in out if r["top_frac"] == 1.0)
    assert bool(full["primary_pass"]) is True
    assert full["mean_advantage_over_coin_bps"] > 2.0
    assert full["mean_pnl_bps"] < 0.0
    assert full["strict_pass"] is False
    assert full["hit_rate_better_side"] > 0.8
    assert full["down_month_rows"] == rows and full["up_month_rows"] == 0
    assert full["flat_share_of_holdout"] == 0.0


def test_mtf_join_fails_closed_on_alias_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = 40
    dataset_time = pd.DatetimeIndex(pd.date_range("2024-01-02T00:00:00Z", periods=rows, freq="5min"))
    feature_names = ["atr_bps_14", "macd_line_atr", "ema20_slope_atr"]
    manifest = {"feature_names": feature_names, "shift_contract": {tf: str(MULTI_TF_SHIFT[tf]) for tf in MULTI_TF_SHIFT}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    ctx_cont_names = ("_v1h1_atr_bps", "_v1h1_ema_diff", "_v1h4_atr_bps", "_v1h4_ema_diff", "d1_atr14_bps_canon_v2", "d1_ema_slope_20_canon_v2")
    ctx_cont = np.zeros((rows, len(ctx_cont_names)), dtype=np.float32)
    # Only the H1 lane is exercised: restrict the Entry timeframes to H1 for the fixture.
    monkeypatch.setattr(wf, "ENTRY_MTF_CONTEXT_TIMEFRAMES", ("H1",))
    ts = pd.date_range("2024-01-01T00:00:00Z", periods=48, freq="1h").asi8
    feats = np.zeros((48, 3), dtype=np.float32)
    feats[:, 0] = np.arange(48, dtype=np.float32)  # atr_bps_14 per H1 bar
    np.save(tmp_path / "H1_ts.npy", ts)
    np.save(tmp_path / "H1_feats.npy", feats)
    # expected last-closed value: cutoff = time + 5min - 1h, last bar label <= cutoff
    cutoff = (dataset_time + pd.Timedelta(minutes=5) - pd.Timedelta(hours=1)).asi8
    expected = np.searchsorted(ts, cutoff, side="right") - 1
    ctx_cont[:, 0] = expected.astype(np.float32)
    matrix, names, report = wf.load_mtf_last_closed(tmp_path, dataset_time, ctx_cont, ctx_cont_names)
    assert matrix.shape == (rows, 3) and names[0] == "H1:atr_bps_14"
    assert report["aliases"]["H1:_v1h1_atr_bps==atr_bps_14"]["mismatch_rows"] == 0
    ctx_cont[3, 0] += 1.0
    with pytest.raises(RuntimeError, match="MTF_JOIN_NOT_EXACT"):
        wf.load_mtf_last_closed(tmp_path, dataset_time, ctx_cont, ctx_cont_names)


def test_one_hot_ctx_cat_uses_owner_domain_and_rejects_unknown_levels() -> None:
    matrix, names = wf.one_hot_ctx_cat(np.array([[0], [3], [1]], dtype=np.int64))
    assert matrix.shape == (3, 4) and names == ("session_id==0", "session_id==1", "session_id==2", "session_id==3")
    assert matrix.sum(axis=1).tolist() == [1.0, 1.0, 1.0]
    with pytest.raises(RuntimeError, match="OUT_OF_DOMAIN"):
        wf.one_hot_ctx_cat(np.array([[7]], dtype=np.int64))


def test_ridge_gram_matches_dense_solution_and_chunks() -> None:
    rng = np.random.default_rng(3)
    n, p = 5000, 7
    X = rng.normal(size=(n, p)).astype(np.float32)
    beta_true = rng.normal(size=p)
    y = X @ beta_true + rng.normal(scale=0.5, size=n)
    X_pred = rng.normal(size=(300, p)).astype(np.float32)
    monkey_chunk = wf.GRAM_CHUNK_ROWS
    wf.GRAM_CHUNK_ROWS = 512  # force many chunks
    try:
        gram = wf.RidgeGram(X, X_pred, inner_fraction=0.2)
        pred, info = gram.fit_predict(y)
    finally:
        wf.GRAM_CHUNK_ROWS = monkey_chunk
    # dense reference with the selected alpha on the full fit fold
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    Xs = ((X - mean) / scale).astype(np.float64)
    yc = y - y.mean()
    beta = np.linalg.solve(Xs.T @ Xs + info["alpha"] * np.eye(p), Xs.T @ yc)
    ref = ((X_pred - mean) / scale).astype(np.float64) @ beta + y.mean()
    assert np.allclose(pred, ref, rtol=1e-4, atol=1e-4)
    assert info["alpha"] in wf.RIDGE_ALPHA_GRID


def _write_synthetic_dataset_and_tape(root: Path, *, train_rows: int, val_rows: int, seed: int = 5) -> tuple[Path, Path]:
    """Tiny dataset dir + native tape with the real column contract (mechanics only, rule 2c)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CTX_CONT_DIM,
        MODEL_NATIVE_CTX_CONT_FIELDS,
        MODEL_NATIVE_SIGNAL_DIM,
    )

    rng = np.random.default_rng(seed)
    total = train_rows + val_rows
    time = pd.date_range("2024-01-01T00:00:00Z", periods=total, freq="5min")
    mid = 2000.0 + np.cumsum(rng.normal(0.0, 0.5, total))
    tape_dir = root / "tape"
    (tape_dir / "year=2024").mkdir(parents=True)
    tape = pd.DataFrame({"time": time, "close": mid, "bid_close": mid - 0.15, "ask_close": mid + 0.15})
    pq.write_table(pa.Table.from_pandas(tape, preserve_index=False), tape_dir / "year=2024" / "part.parquet")
    (tape_dir / "MANIFEST.json").write_text("{}", encoding="utf-8")

    ds_dir = root / "dataset"
    ds_dir.mkdir()
    ctx_names = list(MODEL_NATIVE_CTX_CONT_FIELDS)
    atr_index = ctx_names.index("atr_bps")
    splits = {"train": (0, train_rows), "val": (train_rows, total)}
    split_bounds = {
        "train": {"start": str(time[0]), "end": str(time[train_rows - 1])},
        "val": {"start": str(time[train_rows]), "end": str(time[-1] + pd.Timedelta(minutes=5))},
    }
    for split, (lo, hi) in splits.items():
        n = hi - lo
        snap = rng.normal(size=(n, MODEL_NATIVE_SIGNAL_DIM))
        ctx = rng.normal(size=(n, MODEL_NATIVE_CTX_CONT_DIM))
        ctx[:, atr_index] = 5.0 + rng.uniform(0, 3, n)
        cat = rng.integers(0, 4, size=(n, 1))
        long_out = rng.normal(-2.0, 30.0, n)
        short_out = -long_out - 3.0
        table = pa.table(
            {
                "time": pa.array(time[lo:hi]),
                "snap": pa.array(snap.tolist(), type=pa.list_(pa.float64())),
                "ctx_cont": pa.array(ctx.tolist(), type=pa.list_(pa.float64())),
                "ctx_cat": pa.array(cat.tolist(), type=pa.list_(pa.int64())),
                "label_horizon_bars": pa.array(np.full(n, 19, dtype=np.int32)),
                RESEARCH_LONG_OUTCOME_COLUMN: pa.array(long_out.astype(np.float32)),
                RESEARCH_SHORT_OUTCOME_COLUMN: pa.array(short_out.astype(np.float32)),
            }
        )
        pq.write_table(table, ds_dir / f"entry_dataset__ENTRY_FITTED_Q_{split}.parquet")
        manifest = {
            "feature_contract": {
                "ctx_cont_names": ctx_names,
                "ctx_cat_names": ["session_id"],
                "signal_bridge_fields": [f"f{i}" for i in range(MODEL_NATIVE_SIGNAL_DIM)],
            },
            "splits": split_bounds,
        }
        (ds_dir / f"entry_dataset__ENTRY_FITTED_Q_{split}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ds_dir, tape_dir


def test_run_end_to_end_with_final_val_stage(tmp_path: Path) -> None:
    ds_dir, tape_dir = _write_synthetic_dataset_and_tape(tmp_path, train_rows=6000, val_rows=800)
    out_dir = tmp_path / "out"
    args = wf.build_parser().parse_args(
        [
            "--dataset-dir", str(ds_dir), "--native-m5-root", str(tape_dir), "--out-dir", str(out_dir),
            "--fold-boundaries", "2024-01-11T00:00:00Z", "2024-01-16T00:00:00Z", "2024-01-21T20:00:00Z",
            "--horizons", "12", "--learners", "ridge", "--feature-arms", "snapshot", "--target-scalings", "raw", "atr",
            "--inner-fraction", "0.2", "--max-hgb-iter", "5", "--final-holdout", "val", "--persist-predictions",
            "--decision-rules", "argmax_flat",
        ]
    )
    report = wf.run(args)
    assert report["config"]["final_holdout"] == "val" and report["config"]["final_holdout_rows"] == 800
    assert report["instrument_source_sha256"] and len(report["instrument_source_sha256"]) == 64
    metrics = pd.read_csv(out_dir / "metrics.csv")
    # 2 targets x 2 scalings x (2 folds + final_val) x 7 coverages
    assert len(metrics) == 2 * 2 * 3 * 7
    assert set(metrics["stage"]) == {"fold", "final_val"}
    final = metrics[metrics["stage"] == "final_val"]
    assert (final["holdout_rows"] <= 800).all() and (final["fit_rows"] < 6000).all()
    assert set(final["split"]) == {"final_val"}
    # no fit row's outcome window may reach the first VAL row: fit_rows <= train_rows - purge
    assert final["fit_rows"].max() <= 6000 - (19 + 1)
    preds = sorted((out_dir / "predictions").glob("*.parquet"))
    assert len(preds) == 2 * 2 * 3
    frame = pd.read_parquet(preds[0])
    assert {"time", "pred_long_bps", "pred_short_bps", RESEARCH_LONG_OUTCOME_COLUMN, RESEARCH_SHORT_OUTCOME_COLUMN} <= set(frame.columns)
    assert (out_dir / "summary.md").read_text(encoding="utf-8").startswith("# Entry direction walk-forward research")


def test_fit_hgb_honours_explicit_inputs_and_reports_them() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(1500, 5)).astype(np.float32)
    y = X[:, 0] * 2.0 + rng.normal(scale=0.3, size=1500)
    pred, info = wf.fit_hgb(X, y, X[:50], inner_fraction=0.2, max_iter=20, seed=0, learning_rate=0.03, min_samples_leaf=50)
    assert pred.shape == (50,) and 1 <= info["best_iter"] <= 20
    assert info["learning_rate"] == 0.03 and info["min_samples_leaf"] == 50
    args = wf.build_parser().parse_args(["--dataset-dir", "d", "--native-m5-root", "m", "--out-dir", "o", "--fold-boundaries", "x", "--horizons", "1", "--inner-fraction", "0.2", "--max-hgb-iter", "3"])
    assert args.hgb_learning_rate == wf.HGB_LIBRARY_DEFAULT_LEARNING_RATE and args.hgb_min_samples_leaf == wf.HGB_LIBRARY_DEFAULT_MIN_SAMPLES_LEAF
