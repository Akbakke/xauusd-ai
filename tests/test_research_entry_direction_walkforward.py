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


def test_inner_split_purges_actual_tape_positions() -> None:
    positions = np.arange(1000, dtype=np.int64) * 3
    fit, val = wf._inner_split(1000, 0.2, fit_positions=positions, purge_bars=24)
    assert val.sum() == 200 and fit.sum() == 792
    assert (positions[fit] + 24 < positions[val].min()).all()
    assert (~(fit | val)).sum() == 8


def test_ridge_inner_centering_matches_independent_reference() -> None:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.arange(1000, dtype=np.float32)[:, None]
    y = 3.0 * X[:, 0].astype(np.float64) + 7.0
    gram = wf.RidgeGram(X, X[-10:], inner_fraction=0.2, fit_positions=np.arange(len(X)), purge_bars=24)
    _, info = gram.fit_predict(y)
    errors = []
    for alpha in wf.RIDGE_ALPHA_GRID:
        reference = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        reference.fit(X[gram.inner_fit].astype(np.float64), y[gram.inner_fit])
        errors.append(np.mean((reference.predict(X[gram.inner_val].astype(np.float64)) - y[gram.inner_val]) ** 2))
    assert info["alpha"] == wf.RIDGE_ALPHA_GRID[int(np.argmin(errors))]
    assert info["inner_val_mse"] == pytest.approx(min(errors), rel=1e-5, abs=1e-9)
    assert info["inner_val_mse"] < 0.01
    assert gram.inner_mean[0] == pytest.approx(X[gram.inner_fit].mean())
    assert info["inner_purged_rows"] == 24


def test_hgb_refits_full_fold_after_purged_model_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    import sklearn.ensemble

    fits = []

    class FakeHGB:
        def __init__(self, **params):
            self.params = params

        def fit(self, X, y):
            fits.append((len(X), self.params))
            return self

        def staged_predict(self, X):
            yield np.zeros(len(X))
            yield np.full(len(X), 2.0)
            yield np.full(len(X), 3.0)

        def predict(self, X):
            return np.full(len(X), self.params["max_iter"])

    monkeypatch.setattr(sklearn.ensemble, "HistGradientBoostingRegressor", FakeHGB)
    X = np.zeros((1000, 2), dtype=np.float32)
    prediction, info = wf.fit_hgb(
        X, np.full(1000, 2.0), X[:5], inner_fraction=0.2, max_iter=3, seed=0,
        fit_positions=np.arange(1000), purge_bars=12,
    )
    assert [n for n, _ in fits] == [788, 1000]
    assert fits[-1][1]["max_iter"] == 2 and np.all(prediction == 2)
    assert info["fit_rows"] == 1000 and info["inner_purged_rows"] == 12


def test_tape_filters_before_materialization_and_skips_future_partitions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    (tmp_path / "MANIFEST.json").write_text("{}")
    (tmp_path / "year=2024").mkdir()
    (tmp_path / "year=2025").mkdir()
    (tmp_path / "year=2025" / "future.parquet").write_bytes(b"must not be read")
    times = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
    pq.write_table(pa.table({"time": times, "close": np.full(10, 2000.0),
                            "bid_close": np.full(10, 1999.9), "ask_close": np.full(10, 2000.1)}),
                   tmp_path / "year=2024" / "part.parquet")
    read_table = pq.read_table
    calls = []

    def checked_read(path, **kwargs):
        assert kwargs["filters"] == [("time", "<", times[5].to_pydatetime())]
        table = read_table(path, **kwargs)
        assert len(table) == 5
        calls.append(path)
        return table

    monkeypatch.setattr(wf.pq, "read_table", checked_read)
    tape = wf.load_tape(tmp_path, truncate_before=times[5])
    assert len(calls) == 1 and tape.time.equals(times[:5])


def test_resume_rejects_changed_configuration_source_inputs_and_unbound_cache(tmp_path: Path) -> None:
    (tmp_path / "per_config").mkdir()
    spec = {"config": {"inner_fraction": 0.2}, "source_sha256": "a"}
    inputs = {"arrays": {"X": "x", "y": "y"}}
    digest = wf._check_run_binding(tmp_path, spec, inputs)
    assert digest == wf._check_run_binding(tmp_path, spec, inputs)
    for changed_spec, changed_inputs in [
        ({**spec, "source_sha256": "b"}, inputs),
        ({**spec, "config": {"inner_fraction": 0.3}}, inputs),
        (spec, {"arrays": {"X": "changed", "y": "y"}}),
    ]:
        with pytest.raises(RuntimeError, match="RESUME_BINDING_MISMATCH"):
            wf._check_run_binding(tmp_path, changed_spec, changed_inputs)
    (tmp_path / "RUN_BINDING.json").unlink()
    (tmp_path / "per_config" / "old.json").write_text("{}")
    with pytest.raises(RuntimeError, match="RESUME_UNBOUND_CACHE"):
        wf._check_run_binding(tmp_path, spec)


def test_array_binding_detects_changed_values_and_shape() -> None:
    X = np.arange(10, dtype=np.float32)
    assert wf._array_sha256(X) != wf._array_sha256(X.reshape(5, 2))
    assert wf._array_sha256(X) != wf._array_sha256(X + 1)

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
        gram = wf.RidgeGram(X, X_pred, inner_fraction=0.2, fit_positions=np.arange(len(X)), purge_bars=0)
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
    assert {"time", "pred_long_bps", "pred_short_bps", RESEARCH_LONG_OUTCOME_COLUMN, RESEARCH_SHORT_OUTCOME_COLUMN, "ood_abs_z_mean"} <= set(frame.columns)
    assert "ood_abs_z_mean_mtf" not in frame.columns and np.isfinite(frame["ood_abs_z_mean"]).all() and (frame["ood_abs_z_mean"] > 0).all()
    assert (out_dir / "summary.md").read_text(encoding="utf-8").startswith("# Entry direction walk-forward research")


def test_fit_hgb_honours_explicit_inputs_and_reports_them() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(1500, 5)).astype(np.float32)
    y = X[:, 0] * 2.0 + rng.normal(scale=0.3, size=1500)
    pred, info = wf.fit_hgb(X, y, X[:50], inner_fraction=0.2, max_iter=20, seed=0, learning_rate=0.03, min_samples_leaf=50, fit_positions=np.arange(len(X)), purge_bars=12)
    assert pred.shape == (50,) and 1 <= info["best_iter"] <= 20
    assert info["learning_rate"] == 0.03 and info["min_samples_leaf"] == 50
    args = wf.build_parser().parse_args(["--dataset-dir", "d", "--native-m5-root", "m", "--out-dir", "o", "--fold-boundaries", "x", "--horizons", "1", "--inner-fraction", "0.2", "--max-hgb-iter", "3"])
    assert args.hgb_learning_rate == wf.HGB_LIBRARY_DEFAULT_LEARNING_RATE and args.hgb_min_samples_leaf == wf.HGB_LIBRARY_DEFAULT_MIN_SAMPLES_LEAF


def test_ridge_gram_keep_subset_equals_fresh_fit_on_subset_columns() -> None:
    rng = np.random.default_rng(11)
    n, p = 3000, 9
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = X[:, :4] @ rng.normal(size=4) + rng.normal(scale=0.5, size=n)
    X_pred = rng.normal(size=(200, p)).astype(np.float32)
    keep = np.array([0, 1, 2, 3, 6], dtype=np.int64)
    full = wf.RidgeGram(X, X_pred, inner_fraction=0.2, fit_positions=np.arange(len(X)), purge_bars=0)
    pred_sub, info_sub = full.fit_predict(y, keep=keep)
    fresh = wf.RidgeGram(X[:, keep], X_pred[:, keep], inner_fraction=0.2, fit_positions=np.arange(len(X)), purge_bars=0)
    pred_ref, info_ref = fresh.fit_predict(y)
    assert info_sub["alpha"] == info_ref["alpha"] and info_sub["columns"] == 5
    assert np.allclose(pred_sub, pred_ref, rtol=1e-5, atol=1e-5)
    pred_all, info_all = full.fit_predict(y)
    assert info_all["columns"] == p and not np.allclose(pred_all, pred_sub)


def test_feature_column_groups_map_through_owners() -> None:
    from gx1.features.entry_specialist_feature_groups_v1 import MULTI_TF_SPECIALIST_FEATURE_GROUPS_V4

    swing_field = str(next(iter(MULTI_TF_SPECIALIST_FEATURE_GROUPS_V4["structure_swing_encoder"])))
    trend_field = str(next(iter(MULTI_TF_SPECIALIST_FEATURE_GROUPS_V4["trend_ema_encoder"])))
    names = (
        "f0", "ctx_cont.atr_bps", "session_id==0", "session_id==1",
        f"H1:{swing_field}", f"H4:{swing_field}", f"H1:{trend_field}", "pattern:M5:fvg_bull_event", "pattern:H4:ema_stack",
    )
    groups = wf.feature_column_groups(names)
    assert groups["family:unmapped"].tolist() == [0]
    assert groups["family:vol_compression_encoder"].tolist() == [1]
    assert groups["ctx_cat"].tolist() == [2, 3]
    assert groups["mtf_lane:H1"].tolist() == [4, 6] and groups["mtf_lane:H4"].tolist() == [5]
    assert groups["mtf_family:structure_swing_encoder"].tolist() == [4, 5]
    assert groups["mtf_family:trend_ema_encoder"].tolist() == [6]
    assert groups["patterns:M5"].tolist() == [7] and groups["patterns:H4"].tolist() == [8]
    assert groups["patterns:all"].tolist() == [7, 8] and groups["mtf_lane:all"].tolist() == [4, 5, 6]
    # every column lands in at least one group and the family/lane partitions each cover the MTF columns once
    covered = sorted(set(int(i) for idx in groups.values() for i in idx))
    assert covered == list(range(len(names)))


def test_evaluate_frame_reports_best_constant_side_null() -> None:
    n = 3000  # above the circular-null draw count so the owner null is computable
    rng = np.random.default_rng(4)
    long_out = rng.normal(5.0, 1.0, n)  # LONG always better on these rows
    short_out = -long_out - 3.0
    time = pd.date_range("2024-01-01T00:00:00Z", periods=n, freq="5min")
    side = np.where(np.arange(n) % 2 == 0, MODEL_DIRECTION_LONG_INDEX, MODEL_DIRECTION_SHORT_INDEX)
    frame = pd.DataFrame(
        {
            "split": "fold0", "model": "m", "time": time, "pred_direction": side, "selection_score": np.ones(n),
            "selection_score_mode": wf.MODEL_DIRECTION_SELECTION_MODE, "edge_score": np.ones(n),
            RESEARCH_LONG_OUTCOME_COLUMN: long_out, RESEARCH_SHORT_OUTCOME_COLUMN: short_out,
        }
    )
    fold = wf.Fold(index=0, fit_start=time[0], holdout_start=time[0], holdout_end=time[-1] + pd.Timedelta(minutes=5))
    month_sign = wf.month_drift_sign(wf.Tape(time=pd.DatetimeIndex(time), mid=np.ones(n), bid=np.ones(n), ask=np.ones(n), manifest_sha256="0" * 64, root="synthetic"))
    rows = [r for r in wf.evaluate_frame(frame, fold=fold, month_sign=month_sign, meta={}) if r["top_frac"] == 1.0]
    assert len(rows) == 1
    row = rows[0]
    # half LONG / half SHORT on rows where LONG always wins: always-LONG is the best constant and beats the mixed policy
    assert row["best_constant_side_mean_pnl_bps"] == pytest.approx(float(np.mean(long_out)))
    assert row["excess_over_best_constant_bps"] < 0 and row["beats_best_constant"] is False


def test_load_pattern_primitives_aligns_by_time_and_fails_closed(tmp_path: Path) -> None:
    time = pd.date_range("2024-01-01T00:00:00Z", periods=50, freq="5min")
    prim = pd.DataFrame({"time": time, "M5:x": np.arange(50, dtype=float), "H1:y": np.ones(50)})
    path = tmp_path / "pattern_primitives.parquet"
    prim.to_parquet(path, index=False)
    matrix, names, report = wf.load_pattern_primitives(path, pd.DatetimeIndex(time[10:20]))
    assert names == ("pattern:M5:x", "pattern:H1:y") and matrix.shape == (10, 2) and matrix[:, 0].tolist() == list(range(10, 20))
    assert report["columns"] == 2 and len(report["sha256"]) == 64
    with pytest.raises(RuntimeError, match="WALKFORWARD_PATTERN_ROWS_MISSING"):
        wf.load_pattern_primitives(path, pd.DatetimeIndex([time[-1] + pd.Timedelta(minutes=5)]))


def test_run_end_to_end_with_pattern_arm_and_ablation(tmp_path: Path) -> None:
    ds_dir, tape_dir = _write_synthetic_dataset_and_tape(tmp_path, train_rows=4000, val_rows=400, seed=9)
    rows = 4400
    time = pd.date_range("2024-01-01T00:00:00Z", periods=rows, freq="5min")
    rng = np.random.default_rng(1)
    prim = pd.DataFrame({"time": time, "M5:fvg_bull_event": (rng.uniform(size=rows) < 0.05).astype(float), "H4:ema_stack": rng.choice([-1.0, 0.0, 1.0], size=rows)})
    prim_path = tmp_path / "pattern_primitives.parquet"
    prim.to_parquet(prim_path, index=False)
    out_dir = tmp_path / "out"
    args = wf.build_parser().parse_args(
        [
            "--dataset-dir", str(ds_dir), "--native-m5-root", str(tape_dir), "--out-dir", str(out_dir),
            "--fold-boundaries", "2024-01-08T00:00:00Z", "2024-01-14T21:20:00Z",
            "--horizons", "12", "--learners", "ridge", "--feature-arms", "snapshot_patterns", "--target-scalings", "raw",
            "--inner-fraction", "0.2", "--max-hgb-iter", "5", "--final-holdout", "val", "--decision-rules", "argmax_flat",
            "--targets", "exec_close_h12", "--pattern-primitives-parquet", str(prim_path), "--ablation", "all", "--persist-predictions",
            "--ablation-null-draws", "3",
        ]
    )
    report = wf.run(args)
    persisted = pd.read_parquet(sorted((out_dir / "predictions").glob("*.parquet"))[0])
    assert {"ood_abs_z_mean", "ood_abs_z_mean_patterns"} <= set(persisted.columns) and "ood_abs_z_mean_mtf" not in persisted.columns
    groups = report["config"]["ablation_groups"]["snapshot_patterns"]
    assert groups["patterns:M5"] == 1 and groups["patterns:H4"] == 1 and groups["patterns:all"] == 2 and groups["ctx_cat"] >= 1 and "family:unmapped" in groups
    assert report["config"]["pattern_primitives"]["columns"] == 2
    metrics = pd.read_csv(out_dir / "metrics.csv")
    assert set(metrics["ablation_group"]) == set(groups) | {"none"}
    full = metrics[metrics["ablation_group"] == "none"]
    ablated = metrics[metrics["ablation_group"] != "none"]
    # one full row per (stage-split, coverage) and one ablated row per group for each of them
    assert len(full) == (1 + 1) * 7 and len(ablated) == len(groups) * len(full)
    merged = ablated.merge(full[["split", "top_frac", "mean_pnl_bps"]], on=["split", "top_frac"], suffixes=("", "_full"))
    measurable = merged["mean_pnl_bps"].notna() & merged["mean_pnl_bps_full"].notna()
    # the delta exists exactly where both the full and the ablated cell carry a measurement (rule 2e: no placeholder)
    assert measurable.any() and (merged["delta_mean_pnl_bps_vs_full"].notna() == measurable).all()
    assert np.allclose(merged.loc[measurable, "delta_mean_pnl_bps_vs_full"], merged.loc[measurable, "mean_pnl_bps"] - merged.loc[measurable, "mean_pnl_bps_full"])
    assert {"best_constant_side_mean_pnl_bps", "excess_over_best_constant_bps", "beats_best_constant"} <= set(metrics.columns)
    nulls = ablated[ablated["ablation_null_draws"].notna()]
    assert len(nulls) > 0 and (nulls["ablation_null_draws"] == 3).all()
    assert (nulls["ablation_null_delta_p05_bps"] <= nulls["ablation_null_delta_p95_bps"]).all()
    assert report["config"]["ablation_null_draws"] == 3
    summary = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "| ablation |" in summary


def test_fold_stage_identical_with_and_without_final_holdout(tmp_path: Path) -> None:
    ds_dir, tape_dir = _write_synthetic_dataset_and_tape(tmp_path, train_rows=6000, val_rows=800)
    common = [
        "--dataset-dir", str(ds_dir), "--native-m5-root", str(tape_dir),
        "--fold-boundaries", "2024-01-11T00:00:00Z", "2024-01-16T00:00:00Z", "2024-01-21T20:00:00Z",
        "--horizons", "96", "--learners", "ridge", "--feature-arms", "snapshot", "--target-scalings", "raw",
        "--inner-fraction", "0.2", "--max-hgb-iter", "5", "--decision-rules", "argmax_flat",
    ]
    wf.run(wf.build_parser().parse_args(common + ["--out-dir", str(tmp_path / "a"), "--final-holdout", "none"]))
    wf.run(wf.build_parser().parse_args(common + ["--out-dir", str(tmp_path / "b"), "--final-holdout", "val"]))
    a = pd.read_csv(tmp_path / "a" / "metrics.csv").sort_values(["model", "top_frac"]).reset_index(drop=True)
    b = pd.read_csv(tmp_path / "b" / "metrics.csv")
    b = b[b["stage"] == "fold"].sort_values(["model", "top_frac"]).reset_index(drop=True)
    assert len(a) == len(b) and (a["holdout_rows"].to_numpy() == b["holdout_rows"].to_numpy()).all()
    assert np.allclose(a["mean_pnl_bps"].fillna(0).to_numpy(), b["mean_pnl_bps"].fillna(0).to_numpy())


def test_standardized_abs_z_mean_matches_dense_reference() -> None:
    rng = np.random.default_rng(21)
    X_fit = (rng.normal(size=(3000, 6)) * np.array([1, 2, 3, 4, 5, 6]) + 10).astype(np.float32)
    X_fit[:, 5] = 1.0  # constant column: std 0 -> treated as 1
    X_hold = rng.normal(size=(700, 6)).astype(np.float32)
    monkey = wf.GRAM_CHUNK_ROWS
    wf.GRAM_CHUNK_ROWS = 256
    try:
        out = wf.standardized_abs_z_mean(X_fit, X_hold, {"g": np.array([0, 2], dtype=np.int64)})
    finally:
        wf.GRAM_CHUNK_ROWS = monkey
    mean = X_fit.astype(np.float64).mean(axis=0)
    std = X_fit.astype(np.float64).std(axis=0)
    std[std == 0] = 1.0
    z = np.abs((X_hold.astype(np.float64) - mean) / std)
    assert np.allclose(out["all"], z.mean(axis=1), rtol=1e-5, atol=1e-5)
    assert np.allclose(out["g"], z[:, [0, 2]].mean(axis=1), rtol=1e-5, atol=1e-5)
    with pytest.raises(RuntimeError, match="WALKFORWARD_OOD_DISTANCE_INPUT_INVALID"):
        wf.standardized_abs_z_mean(X_fit[:0], X_hold, {})


def test_selected_mean_pnl_by_coverage_matches_evaluate_frame() -> None:
    n = 2500
    rng = np.random.default_rng(8)
    long_out = rng.normal(0.0, 30.0, n)
    short_out = -long_out - 3.0
    score = rng.normal(size=n)
    score[::7] = score[0]  # ties are broken by time order in both implementations
    side = np.where(rng.uniform(size=n) < 0.2, MODEL_DIRECTION_FLAT_INDEX, np.where(score > 0, MODEL_DIRECTION_LONG_INDEX, MODEL_DIRECTION_SHORT_INDEX))
    time = pd.date_range("2024-01-01T00:00:00Z", periods=n, freq="5min")
    frame = pd.DataFrame(
        {
            "split": "fold0", "model": "m", "time": time, "pred_direction": side, "selection_score": score,
            "selection_score_mode": wf.MODEL_DIRECTION_SELECTION_MODE, "edge_score": score,
            RESEARCH_LONG_OUTCOME_COLUMN: long_out, RESEARCH_SHORT_OUTCOME_COLUMN: short_out,
        }
    )
    fold = wf.Fold(index=0, fit_start=time[0], holdout_start=time[0], holdout_end=time[-1] + pd.Timedelta(minutes=5))
    month_sign = wf.month_drift_sign(wf.Tape(time=pd.DatetimeIndex(time), mid=np.ones(n), bid=np.ones(n), ask=np.ones(n), manifest_sha256="0" * 64, root="synthetic"))
    rows = {r["top_frac"]: r for r in wf.evaluate_frame(frame, fold=fold, month_sign=month_sign, meta={})}
    light = wf.selected_mean_pnl_by_coverage(side, score, long_out, short_out, wf.EVALUATION_COVERAGES)
    assert any(v is None for v in light.values()) and any(v is not None for v in light.values())
    for top_frac, mean_pnl in light.items():
        if "mean_pnl_bps" in rows[top_frac]:
            assert mean_pnl == pytest.approx(rows[top_frac]["mean_pnl_bps"])
        else:
            assert mean_pnl is None


def test_ridge_gram_fixed_alpha_matches_grid_choice() -> None:
    rng = np.random.default_rng(5)
    X = rng.normal(size=(2000, 5)).astype(np.float32)
    y = X[:, 0] * 2 + rng.normal(scale=0.3, size=2000)
    X_pred = rng.normal(size=(100, 5)).astype(np.float32)
    gram = wf.RidgeGram(X, X_pred, inner_fraction=0.2, fit_positions=np.arange(len(X)), purge_bars=0)
    pred, info = gram.fit_predict(y)
    pred_fixed, info_fixed = gram.fit_predict(y, alpha=info["alpha"])
    assert info_fixed["inner_val_mse"] is None and info_fixed["alpha"] == info["alpha"]
    assert np.allclose(pred, pred_fixed)


def _write_cross_fixtures(root: Path, time: pd.DatetimeIndex, *, daily_end: pd.Timestamp, h1_end: pd.Timestamp) -> tuple[Path, Path]:
    rng = np.random.default_rng(3)
    days = pd.date_range((time[0] - pd.Timedelta(days=120)).floor("D"), daily_end.floor("D"), freq="D", tz="UTC")
    daily = pd.DataFrame({f"{inst}_lvl": np.cumsum(rng.normal(0, 0.01, len(days))) + 4.5 for inst in wf.CROSS_DAILY_INSTRUMENTS}, index=days)
    daily_path = root / "macro_features.parquet"
    daily.to_parquet(daily_path)
    hours = pd.date_range((time[0] - pd.Timedelta(days=10)).floor("h"), h1_end.floor("h"), freq="1h", tz="UTC")
    close = 150.0 + np.cumsum(rng.normal(0, 0.05, len(hours)))
    h1 = pd.DataFrame({"time": hours, "close": close, "bid_close": close - 0.01, "ask_close": close + 0.01})
    h1_path = root / "USD_JPY_H1.parquet"
    h1.to_parquet(h1_path, index=False)
    return daily_path, h1_path


def test_cross_asset_block_daily_lag_h1_cutoff_and_validity(tmp_path: Path) -> None:
    time = pd.DatetimeIndex(pd.date_range("2024-03-01T00:00:00Z", periods=3000, freq="5min"))
    daily_path, h1_path = _write_cross_fixtures(tmp_path, time, daily_end=pd.Timestamp("2024-03-06T00:00:00Z"), h1_end=pd.Timestamp("2024-03-05T12:00:00Z"))
    matrix, names, valid, report = wf.load_cross_asset_block(daily_path, h1_path, time)
    daily = pd.read_parquet(daily_path)
    assert matrix.shape == (3000, len(names)) and np.isfinite(matrix).all()
    # decision on 2024-03-03 (any hour) uses the row dated 2024-03-02: chg1d = lvl[03-02] - lvl[03-01]
    i = int(np.flatnonzero(time == pd.Timestamp("2024-03-03T15:00:00Z"))[0])
    col = names.index("cross:daily:dxy_chg1d")
    expected = float(daily.loc["2024-03-02", "dxy_lvl"] - daily.loc["2024-03-01", "dxy_lvl"])
    assert matrix[i, col] == pytest.approx(expected, rel=1e-5) and valid[i]
    # H1: decision 10:55 -> cutoff 10:00 -> bar labelled 10:00 (closing 11:00); ret1 = log(close[10:00]) - log(close[09:00])
    j = int(np.flatnonzero(time == pd.Timestamp("2024-03-02T10:55:00Z"))[0])
    h1 = pd.read_parquet(h1_path).set_index("time")
    r1 = float(np.log(h1.loc["2024-03-02T10:00:00Z", "close"]) - np.log(h1.loc["2024-03-02T09:00:00Z", "close"]))
    assert matrix[j, names.index("cross:usdjpy_h1:ret1")] == pytest.approx(r1, rel=1e-5) and valid[j]
    # after the H1 series ends (+72 h staleness) rows are invalid; after the daily series ends (+1 day lag) too
    assert not valid[time > pd.Timestamp("2024-03-08T13:00:00Z")].any()
    assert valid[time == pd.Timestamp("2024-03-05T12:00:00Z")].all()
    assert report["daily"]["invalid_decision_rows"] + report["daily"]["valid_decision_rows"] == 3000
    assert report["invalid_decision_rows"] == int((~valid).sum())
    with pytest.raises(RuntimeError, match="WALKFORWARD_CROSS_ARM_REQUIRES_INPUT"):
        wf.load_cross_asset_block(None, None, time)


def test_run_end_to_end_cross_arm_drops_invalid_rows(tmp_path: Path) -> None:
    ds_dir, tape_dir = _write_synthetic_dataset_and_tape(tmp_path, train_rows=4000, val_rows=1200, seed=13)  # VAL stays above the circular-null minimum after the drop
    time = pd.DatetimeIndex(pd.date_range("2024-01-01T00:00:00Z", periods=5200, freq="5min"))
    # daily series ends two days before the VAL end (TRAIN ends earlier and stays fully valid), H1 covers everything:
    # VAL loses its last calendar day through the one-day lag
    daily_path, h1_path = _write_cross_fixtures(tmp_path, time, daily_end=time[-1] - pd.Timedelta(days=2), h1_end=time[-1] + pd.Timedelta(hours=2))
    out_dir = tmp_path / "out"
    args = wf.build_parser().parse_args(
        [
            "--dataset-dir", str(ds_dir), "--native-m5-root", str(tape_dir), "--out-dir", str(out_dir),
            "--fold-boundaries", "2024-01-08T00:00:00Z", "2024-01-14T21:20:00Z",
            "--horizons", "12", "--learners", "ridge", "--feature-arms", "snapshot_cross", "--target-scalings", "raw",
            "--inner-fraction", "0.2", "--max-hgb-iter", "5", "--final-holdout", "val", "--decision-rules", "argmax_flat",
            "--targets", "exec_close_h12", "--cross-asset-daily-parquet", str(daily_path), "--cross-asset-h1-parquet", str(h1_path),
            "--ablation", "cross", "--persist-predictions",
        ]
    )
    report = wf.run(args)
    groups = report["config"]["ablation_groups"]["snapshot_cross"]
    assert set(groups) == {"cross:daily", "cross:usdjpy_h1", "cross:all"}
    assert report["config"]["arm_valid_rows"]["snapshot_cross"] == 4000
    assert 0 < report["config"]["val_arm_valid_rows"]["snapshot_cross"] < 1200
    metrics = pd.read_csv(out_dir / "metrics.csv")
    final = metrics[(metrics["stage"] == "final_val") & (metrics["ablation_group"] == "none")]
    assert (final["holdout_rows"] == report["config"]["val_arm_valid_rows"]["snapshot_cross"]).all()
    assert (metrics[metrics["ablation_group"] != "none"]["ablation_group"].isin(set(groups))).all()
    persisted = pd.read_parquet(sorted((out_dir / "predictions").glob("*.parquet"))[-1])  # highest fold index = the final_val stage
    assert len(persisted) == report["config"]["val_arm_valid_rows"]["snapshot_cross"] and np.isfinite(persisted["ood_abs_z_mean"]).all()
