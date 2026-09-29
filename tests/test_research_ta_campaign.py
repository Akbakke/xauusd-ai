"""Synthetic mechanics only; no market outcomes or TEST files."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from gx1.scripts import research_ta_campaign_v1 as ta


def _bars(n=850):
    rng = np.random.default_rng(41)
    opens = pd.date_range("2009-06-01T22:00Z", periods=n, freq="D")
    times = pd.DatetimeIndex(np.column_stack([opens.asi8, (opens + pd.Timedelta(hours=23, minutes=55)).asi8]).ravel(), tz="UTC")
    prices = np.repeat(100 + np.cumsum(rng.normal(0.02, 0.2, n)), 2)
    close = prices + np.tile([0.0, 0.1], n)
    return pd.DataFrame({"open": prices, "high": prices + 0.3, "low": prices - 0.3,
                         "close": close, "bid_open": prices - 0.01, "ask_open": prices + 0.01}, index=times)


def test_daily_panel_waits_for_closed_session_and_uses_next_open_quote():
    frame = _bars(310)
    end = frame.index[-1] + pd.Timedelta(minutes=5)
    panel = ta.daily_panel(frame, end)
    assert len(panel) == 309  # final completed session has no later executable quote
    assert (panel.fill_time >= panel.decision_time).all()
    assert panel.fill_time.iloc[0] == pd.Timestamp("2009-06-02T22:00Z")
    assert panel.fill_mid.iloc[0] == frame.open.iloc[2]
    assert panel.close.iloc[0] == frame.close.iloc[1]
    assert np.isnan(panel[ta.FEATURES].iloc[:252].to_numpy()).any(axis=1).all()
    row = 270
    expected = (panel.close.iloc[row] - panel.close.iloc[row - 252]) / panel.atr14.iloc[row]
    assert panel.momentum_252_atr14.iloc[row] == pytest.approx(expected)
    # Modifying later candles cannot alter an earlier feature vector or quote.
    changed = frame.copy()
    changed.iloc[580:] *= 2
    after = ta.daily_panel(changed, end)
    pd.testing.assert_frame_equal(panel.iloc[:280], after.iloc[:280])


def test_daily_panel_preserves_weekend_gap_without_synthesizing_a_quote():
    frame = _bars(310)
    missing = (frame.index >= pd.Timestamp("2009-06-06T22:00Z")) & (frame.index < pd.Timestamp("2009-06-08T22:00Z"))
    frame = frame.loc[~missing]
    panel = ta.daily_panel(frame, frame.index[-1] + pd.Timedelta(minutes=5))
    row = panel.loc[panel.decision_time == pd.Timestamp("2009-06-06T22:00Z")].iloc[0]
    assert row.fill_time == pd.Timestamp("2009-06-08T22:00Z")
    assert row.fill_time in frame.index


def _spec(tmp_path):
    return {
        "horizons": [20, 5], "primary_horizon": 20, "fold_years": [2010, 2011],
        "min_fit_rows": 100, "min_inner_rows": 20, "inner_fraction": 0.2, "seed": 0,
        "hgb": {"max_iter": 3, "learning_rate": 0.1, "min_samples_leaf": 20},
        "risk": {"lookback": 21, "target_annual_vol": 0.1, "max_gross_leverage": 1.0},
        "initial_equity": 100.0, "periods_per_year": 252,
        "slippage_bps_per_execution": 2.0, "commission_bps_per_execution": 0.0,
        "bootstrap_draws": 39, "mean_block_length": 60, "alpha": 0.05, "desired_power": 0.8,
        "effects": {"mean_delta_bps": [1., 2., 5.], "normalized_delta": [.01, .02, .05],
                    "sharpe_delta": [.1, .2, .3]},
        "tape": {"manifest_sha256": "0" * 64, "root": "synthetic"},
    }


def test_walkforward_and_portfolio_integration_include_full_family_and_purged_fits(tmp_path):
    frame = _bars()
    panel = ta.daily_panel(frame, frame.index[-1] + pd.Timedelta(minutes=5))
    spec = _spec(tmp_path)
    forecasts, fits = ta.fit_predictions(panel, spec)
    complete = [f for f in fits if f["status"] == "FIT"]
    assert complete
    assert all(pd.Timestamp(f["last_fit_outcome_time"]) < pd.Timestamp(f["first_hold_time"]) for f in complete)
    assert all(f["ridge"]["constant_alternative_enabled"] is False for f in complete)
    curve = ta.ResearchFinancingCurve(pd.DatetimeIndex([frame.index[0]]), np.array([0.04]),
                                     frame.index[-1] + pd.Timedelta(days=1), 0.0129, 31557600.)
    result = ta.evaluate(panel, forecasts, curve, spec, tmp_path)
    assert len(result["declared_family"]) == 96
    assert {r["name"] for r in result["endpoints"]} == set(result["declared_family"])
    assert result["evaluation"]["reused_development_history"] is True
    assert set(result["decisions"]) == {"ridge", "hgb"}
    assert all(row["decision"] in {"GO", "NO_GO", "INKONKLUSIV"} for row in result["decisions"].values())
    assert result["portfolios"]["h20:buy_hold:zero"]["execution_count"] == 2
    assert result["portfolios"]["h20:buy_hold:historical_proxy"]["financing_cost_bps"] > 0
    paired = pd.read_parquet(tmp_path / "PAIRED_RETURNS.parquet")
    assert len(paired) == result["evaluation"]["intervals"]
    for name in result["untestable_endpoints"]:
        row = next(r for r in result["endpoints"] if r["name"] == name)
        assert row["effect_verdict"] == "INKONKLUSIV"
        assert "max_t_adjusted_two_sided_p" not in row


def test_loader_never_visits_unauthorized_years_and_checks_file_hash(tmp_path, monkeypatch):
    root = tmp_path / "market"
    allowed = root / "year=2025"
    allowed.mkdir(parents=True)
    frame = _bars(4).reset_index(names="time")
    frame["time"] = pd.date_range("2025-01-01", periods=len(frame), freq="5min", tz="UTC")
    path = allowed / "part-000.parquet"
    frame.to_parquet(path, index=False)
    forbidden = root / "year=2026"
    forbidden.mkdir()
    (forbidden / "part-000.parquet").write_bytes(b"TEST must not be opened")
    manifest = {
        "timestamp_semantics": "bar_start_utc", "decision_available_offset_seconds": 300,
        "market_closure_contract": "oanda_complete_true_source_absence_no_synthesis_v1",
        "year_sha256": {"year=2025": ta.sha(path)},
    }
    manifest_path = root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest))
    spec = {"tape": {"root": str(root), "manifest_sha256": ta.sha(manifest_path)},
            "read_start": "2025-01-01T00:00Z", "read_end_exclusive": "2026-01-01T00:00Z"}
    original = ta.pq.read_table
    calls = []
    def read_only_allowed(path_arg, **kwargs):
        assert Path(path_arg) == path
        calls.append(Path(path_arg))
        assert kwargs["filters"][1] == ("time", "<", pd.Timestamp(spec["read_end_exclusive"]).to_pydatetime())
        return original(path_arg, **kwargs)
    monkeypatch.setattr(ta.pq, "read_table", read_only_allowed)
    got, binding = ta.load_market(spec)
    assert len(got) == len(frame) and calls == [path]
    assert binding["test_accessed"] is False
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="YEAR_HASH"):
        ta.load_market(spec)

def test_run_writes_terminal_bound_inventory_and_predictions(tmp_path, monkeypatch):
    frame = _bars()
    spec = _spec(tmp_path)
    spec.update(feature_names=ta.FEATURES, output_directory=str(tmp_path / "run"),
                read_end_exclusive=str(frame.index[-1] + pd.Timedelta(minutes=5)))
    spec_path = tmp_path / "prereg.json"
    spec_path.write_text(json.dumps(spec))
    curve = ta.ResearchFinancingCurve(pd.DatetimeIndex([frame.index[0]]), np.array([0.04]),
                                     frame.index[-1] + pd.Timedelta(days=1), 0.0129, 31557600.)
    monkeypatch.setattr(ta, "load_funding", lambda spec: curve)
    monkeypatch.setattr(ta, "load_market", lambda spec: (frame, {"test_accessed": False, "fixture": True}))
    result = ta.run_a(spec, spec_path)
    out = Path(spec["output_directory"])
    assert result["status"] == "COMPLETE"
    terminal = json.loads((out / "TERMINAL.json").read_text())
    assert terminal["result_sha256"] == ta.sha(out / "RESULT.json")
    report = json.loads((out / "RESULT.json").read_text())
    assert report["test_accessed"] is False and report["native_training"] is False
    assert "PREDICTIONS.parquet" in report["artifacts"]
    for name, binding in report["artifacts"].items():
        assert binding["sha256"] == ta.sha(out / name)
    with pytest.raises(FileExistsError):
        ta.run_a(spec, spec_path)

def test_funding_fetch_binds_bytes_dates_and_explicit_retry_timeout(tmp_path, monkeypatch):
    import io
    raw = b"observation_date,DFF\n2020-01-01,1.5\n2020-01-02,1.6\n"
    class Response(io.BytesIO):
        url = "https://fred.stlouisfed.org/bound-source"
    observed = []
    def fetch(request, timeout):
        observed.append((request.full_url, timeout))
        return Response(raw)
    monkeypatch.setattr(ta, "urlopen", fetch)
    spec = {"series": "DFF", "url": Response.url, "maximum_bytes": 1000,
            "output_directory": str(tmp_path / "fetch"), "timeout_seconds": 60,
            "start_date": "2020-01-01", "end_date": "2020-01-02"}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(spec))
    receipt = ta.fetch_funding(spec, manifest)
    assert observed == [(Response.url, 60)]
    assert receipt["raw_sha256"] == ta.sha(Path(receipt["raw_path"]))
    assert receipt["rows"] == 2
    assert Path(receipt["raw_path"]).read_bytes() == raw


def test_effr_calendar_exact_coverage_and_piecewise_weekend_funding(tmp_path, monkeypatch):
    import io
    spec = {"series": "EFFR", "rate_field": "percentRate", "start_date": "2021-06-18",
            "end_date": "2021-06-21", "output_directory": str(tmp_path / "effr"),
            "url": "https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json",
            "maximum_bytes": 1000, "timeout_seconds": 60}
    # Saturday Juneteenth does not close the Reserve Bank on Friday.
    records = [{"effectiveDate": "2021-06-21", "type": "EFFR", "percentRate": 2.0},
               {"effectiveDate": "2021-06-18", "type": "EFFR", "percentRate": 1.0}]
    raw = json.dumps({"refRates": records}).encode()
    class Response(io.BytesIO):
        url = spec["url"]
    monkeypatch.setattr(ta, "urlopen", lambda request, timeout: Response(raw))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(spec))
    receipt = ta.fetch_funding(spec, manifest)
    assert receipt["rows"] == 2
    assert Path(receipt["raw_path"]).read_bytes() == raw
    curve = ta.load_funding({"funding": {"manifest": str(manifest),
        "manifest_sha256": ta.sha(manifest), "markup": 0.0129, "seconds_per_year": 31557600.0}})
    assert list(curve.benchmark_annual_rate) == [.01, .02]
    assert list(curve.effective_at) == list(pd.to_datetime(["2021-06-18", "2021-06-21"], utc=True))
    with pytest.raises(RuntimeError, match="TA_FUNDING_COVERAGE"):
        ta.parse_funding(json.dumps({"refRates": records[:1]}).encode(), spec)
    with pytest.raises(RuntimeError, match="TA_FUNDING_COVERAGE"):
        ta.parse_funding(json.dumps({"refRates": records + records[:1]}).encode(), spec)
    spec.update(start_date="2022-06-17", end_date="2022-06-21")
    assert list(ta.funding_dates(spec).strftime("%Y-%m-%d")) == ["2022-06-17", "2022-06-21"]


def test_funding_recovery_reuses_bound_bytes_without_network(tmp_path, monkeypatch):
    raw_path = tmp_path / "original.json"
    raw_path.write_text(json.dumps({"refRates": [
        {"effectiveDate": "2020-01-02", "type": "EFFR", "percentRate": 1.5}]}))
    spec = {"series": "EFFR", "rate_field": "percentRate", "start_date": "2020-01-02",
            "end_date": "2020-01-02", "url": "https://markets.newyorkfed.org/bound",
            "maximum_bytes": 1000, "output_directory": str(tmp_path / "validated"),
            "reuse_download": {"path": str(raw_path), "sha256": ta.sha(raw_path)}}
    monkeypatch.setattr(ta, "urlopen", lambda *a, **k: pytest.fail("must not download again"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(spec))
    receipt = ta.fetch_funding(spec, manifest)
    assert receipt["raw_path"] == str(raw_path)
    assert receipt["rows"] == 1
    spec["output_directory"] = str(tmp_path / "rejected")
    spec["reuse_download"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="TA_FUNDING_REUSED_BYTES_HASH"):
        ta.fetch_funding(spec, manifest)
