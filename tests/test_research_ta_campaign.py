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


def _alfred_form():
    return b"""<select name="form[units]"><option value="lin">Levels</option></select>
    <select name="form[file_type]"><option value="1">Real time</option></select>
    <select name="form[file_format]"><option value="csv">ZIP</option></select>
    <select name="form[selected_vintage_dates][]">
    <option value="2008-12-31">old</option><option value="2009-01-02">first</option>
    <option value="2025-12-31">last</option><option value="2026-01-02">excluded</option></select>"""


def test_alfred_post_binds_vintages_and_preserves_raw_download_without_admitting_inputs(tmp_path, monkeypatch):
    import io, zipfile
    from urllib.parse import parse_qs
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("DFII10.csv", "observation_date,value,realtime_start_date,realtime_end_date\\n")
    class Response(io.BytesIO):
        url = "https://alfred.stlouisfed.org/series/downloaddata?seid=DFII10"
    requests = []
    def fetch(req, timeout):
        requests.append(req)
        return Response(archive.getvalue() if req.data is not None else _alfred_form())
    monkeypatch.setattr(ta, "urlopen", fetch)
    spec = {"series": [{"id": "DFII10", "url": Response.url}], "output_directory": str(tmp_path/"fetch"),
            "timeout_seconds": 60, "maximum_form_bytes": 10000, "maximum_response_bytes": 10000,
            "maximum_uncompressed_bytes": 10000, "vintage_start": "2009-01-01",
            "vintage_end": "2025-12-31", "observation_start": "2009-01-01", "observation_end": "2025-12-31"}
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(spec))
    result = ta.fetch_alfred(spec, manifest)
    assert result["series_status"] == {"DFII10": "RETRIEVED_NOT_ADMITTED"}
    data = parse_qs(requests[1].data.decode(), keep_blank_values=True)
    assert data["form[selected_vintage_dates][]"] == ["2009-01-02", "2025-12-31"]
    assert data["form[file_type]"] == ["1"]
    assert data["form[download_data]"] == [""]
    receipt = json.loads((tmp_path/"fetch"/"DFII10"/"RECEIPT.json").read_text())
    assert receipt["predictor_admitted"] is False
    assert receipt["raw_sha256"] == ta.sha(Path(receipt["raw_path"]))
    assert Path(receipt["raw_path"]).read_bytes() == archive.getvalue()


def test_alfred_rejects_changed_form_and_records_failure(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="TA_ALFRED_FORM_SCHEMA"):
        ta.alfred_form_body(b"<html>unavailable</html>", {})
    def unavailable(*args, **kwargs):
        raise TimeoutError("source timeout")
    monkeypatch.setattr(ta, "urlopen", unavailable)
    spec = {"series": [{"id": "DFII10", "url": "https://alfred.stlouisfed.org/bound"}],
            "output_directory": str(tmp_path/"failed"), "timeout_seconds": 60}
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(spec))
    result = ta.fetch_alfred(spec, manifest)
    assert result["series_status"] == {"DFII10": "FAILED"}
    terminal = json.loads((tmp_path/"failed"/"TERMINAL.json").read_text())
    assert terminal["all_downloads_succeeded"] is False


def _c_market():
    times = pd.date_range("2026-06-01T09:00Z", periods=41, freq="5min")
    mid = 100 + np.arange(len(times)) * .1
    return pd.DataFrame({"open": mid, "high": mid + .3, "low": mid - .3, "close": mid + .05,
                         "volume": 1., "bid_open": mid - .1, "ask_open": mid + .1,
                         "bid_close": mid - .05, "ask_close": mid + .15,
                         "bid_high": mid + .2, "ask_low": mid - .2}, index=times)


def _c_signals(market, indices):
    rows = market.index[indices]
    frame = pd.DataFrame(0, index=rows, columns=ta.C_CELLS)
    frame["known_at"] = rows + pd.Timedelta(minutes=5)
    frame["orb_exit"] = pd.Series(pd.NaT, index=rows, dtype="datetime64[ns, UTC]")
    frame["risk_scale"], frame["atr_bps"] = 1., 20.
    return frame


def test_c_selection_has_closed_bar_clock_conflicts_shared_slots_and_no_future_gap_filter():
    market = _c_market().drop(pd.Timestamp("2026-06-01T10:05Z"))
    signals = _c_signals(market, [0, 1, 2, 13])
    signals[ta.C_CELLS[0]] = [1, 1, 1, -1]
    signals[ta.C_CELLS[1]] = [1, -1, 0, 0]
    spec = {"evaluation_start": "2026-06-01T09:00Z", "read_end_exclusive": "2026-06-02T00:00Z", "hold_bars": 12}
    cohort, counts = ta.c_select(market, signals, spec)
    assert counts == {"conflicting_rows": 1, "same_side_duplicates": 1, "overlap_rows": 1}
    assert len(cohort) == 2
    assert cohort.iloc[0].entry_time == pd.Timestamp("2026-06-01T09:05Z")
    assert cohort.iloc[0].target_time == pd.Timestamp("2026-06-01T10:05Z")
    assert cohort.iloc[0].exit_time == pd.Timestamp("2026-06-01T10:10Z")
    assert cohort.iloc[1].entry_time >= cohort.iloc[0].exit_time
    # A bid-only low cannot fill a passive buy: the ask must reach its bid limit.
    changed = market.copy()
    changed.loc[cohort.iloc[0].entry_time, "ask_low"] = 1000.
    revised, _ = ta.c_select(changed, signals, spec)
    assert not revised.iloc[0].passive_touched
    assert revised.iloc[1].passive_touched
    pd.testing.assert_frame_equal(cohort.drop(columns="passive_touched"),
                                  revised.drop(columns="passive_touched"))


def test_c_book_cash_reconciles_limit_price_costs_signed_funding_and_terminal_mark():
    market = _c_market()
    signals = _c_signals(market, [0, 30])
    signals[ta.C_CELLS[0]] = [1, -1]
    spec = {"evaluation_start": "2026-06-01T09:00Z", "read_end_exclusive": "2026-06-01T12:25Z", "hold_bars": 12}
    cohort, _ = ta.c_select(market, signals, spec)
    assert cohort.iloc[-1].censored_at_end  # still accounted at the last available close
    tape = ta.c_quotes(market, pd.Timestamp(spec["evaluation_start"]))
    curve = ta.ResearchFinancingCurve(pd.DatetimeIndex(["2026-06-01T00:00Z"]), np.array([.04]),
                                     pd.Timestamp("2026-06-02T00:00Z"), .0129, 31557600.)
    for mode in ["active", "long", "passive"]:
        book, outcomes = ta.c_book(tape, cohort, mode, 1., curve, 100.)
        assert book.held_units_after.iloc[-1] == 0
        assert book.equity_liquidation.iloc[-1] - 100. == pytest.approx(outcomes.risk_pnl.sum())
        r = cohort.iloc[0]
        entry = r.passive_limit if mode == "passive" else r.entry_ask
        entry_time = r.passive_fill_time if mode == "passive" else r.entry_time
        duration = (r.exit_time - entry_time).total_seconds()
        expected_cash = 100 / r.decision_mid * (
            r.exit_bid - entry - (0. if mode == "passive" else entry / 1e4)
            - r.exit_bid / 1e4 - entry * (.04 + .0129) * duration / 31557600.)
        assert outcomes.iloc[0].risk_pnl == pytest.approx(expected_cash)
        if mode != "long":
            assert outcomes.iloc[1].financing_bps < 0  # short credit stays signed
        if mode == "passive":
            assert outcomes.iloc[0].spread_bps < outcomes.iloc[0].slippage_bps


def test_c_signal_owner_uses_row_start_plus_five_and_future_mutation_cannot_change_past():
    market = _bars(310)
    market["volume"] = 1.
    spec = {"evaluation_start": str(market.index[560]), "read_end_exclusive": str(market.index[-1] + pd.Timedelta(minutes=5)),
            "round_grid_usd": 50., "initial_equity": 100.,
            "risk": {"periods_per_year": 252., "lookback": 21, "target_annual_vol": .1, "max_gross_leverage": 1.}}
    signals, _ = ta.c_signal_panel(market, spec)
    assert (signals.known_at == signals.index + pd.Timedelta(minutes=5)).all()
    changed = market.copy()
    changed.loc[changed.index >= market.index[600], ["open", "high", "low", "close"]] *= 2
    revised, _ = ta.c_signal_panel(changed, spec)
    pd.testing.assert_frame_equal(signals.loc[signals.index < market.index[598]],
                                  revised.loc[revised.index < market.index[598]])


def test_c_run_includes_nonfills_full_family_and_censored_positions(tmp_path, monkeypatch):
    days = pd.date_range("2026-06-01", "2026-06-30", freq="D", tz="UTC")
    market = pd.concat([_c_market().set_axis(day + (_c_market().index - _c_market().index[0].normalize()))
                        for day in days])
    # Deterministic nonconstant daily prices; no real market bytes.
    multiplier = 1 + np.sin(np.arange(len(market)) / 31) * .01
    price_columns = [c for c in market if c != "volume"]
    market.loc[:, price_columns] = market[price_columns].mul(multiplier, axis=0)
    last = market.iloc[-1:].copy()
    last.index = pd.DatetimeIndex(["2026-06-30T23:55Z"])
    market = pd.concat([market, last])
    signal = _c_signals(market, np.arange(0, len(days) * 41, 41))
    signal[ta.C_CELLS[0]] = np.where(np.arange(len(days)) % 2 == 0, 1, -1)
    # Explicitly prevent half the passive touches, leaving the shared cohort unchanged.
    for i, t in enumerate(signal.known_at):
        if i % 3 == 0:
            market.loc[t, ["ask_low", "bid_high"]] = [market.loc[t, "ask_open"], market.loc[t, "bid_open"]]
    spec = {
        "cells": list(ta.C_CELLS), "hold_bars": 12, "slippage_scenarios": [0., .5, 1., 2.],
        "evaluation_start": "2026-06-01T00:00:00Z", "read_end_exclusive": "2026-07-01T00:00:00Z",
        "output_directory": str(tmp_path / "c"), "initial_equity": 100., "periods_per_year": 365.25,
        "bootstrap_draws": 39, "mean_block_length": 5., "seed": 0, "alpha": .05, "desired_power": .8,
        "effects": {"mean_net_bps": [1., 2., 5.], "normalized_net": [.01, .02, .05], "sharpe_delta": [.1, .2, .3]},
        "limitations": ["synthetic mechanics"],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    curve = ta.ResearchFinancingCurve(pd.DatetimeIndex(["2026-06-01T00:00Z"]), np.array([.04]),
                                     pd.Timestamp("2026-07-01T00:00Z"), .0129, 31557600.)
    monkeypatch.setattr(ta, "load_market", lambda spec, **kwargs: (market, {"test_accessed": False, "synthetic": True}))
    monkeypatch.setattr(ta, "load_funding", lambda spec: curve)
    monkeypatch.setattr(ta, "c_signal_panel", lambda market, spec: (signal, {}))
    got = ta.run_c(spec, spec_path)
    assert got["status"] == "COMPLETE"
    out = Path(spec["output_directory"])
    result = json.loads((out / "RESULT.json").read_text())
    assert len(result["declared_family"]) == len(result["endpoints"]) == 72
    assert result["selection"]["selected"] == 30
    assert result["selection"]["passive_touched"] < result["selection"]["executable"]
    assert result["selection"]["calendar_days"] == 30
    assert len(result["decision"]["required_endpoints"]) == 4
    assert result["test_outcomes_accessed"] is False
    for name, binding in result["artifacts"].items():
        assert ta.sha(out / name) == binding["sha256"]
    terminal = json.loads((out / "TERMINAL.json").read_text())
    assert terminal["result_sha256"] == ta.sha(out / "RESULT.json")


def test_c_passive_terminal_touch_is_marked_even_when_assumed_fill_time_equals_cutoff():
    market = _c_market()
    signal = _c_signals(market, [39])
    signal[ta.C_CELLS[0]] = 1
    spec = {"evaluation_start": "2026-06-01T09:00Z", "read_end_exclusive": "2026-06-01T12:25Z", "hold_bars": 12}
    cohort, _ = ta.c_select(market, signal, spec)
    r = cohort.iloc[0]
    assert r.passive_fill_time == r.exit_time and r.passive_touched and r.censored_at_end
    tape = ta.c_quotes(market, pd.Timestamp(spec["evaluation_start"]))
    book, outcomes = ta.c_book(tape, cohort, "passive", 0., None, 100.)
    assert outcomes.iloc[0].filled
    assert outcomes.iloc[0].risk_pnl == pytest.approx(100 / r.decision_mid * (r.exit_bid - r.passive_limit))
    assert book.held_units_after.iloc[-1] == 0



def test_alfred_version_summary_checks_duplicate_values_and_interval_boundaries():
    rows = [("2020-01-01", "1", "2020-01-02", "2020-01-04"),
            ("2020-01-01", "2", "2020-01-05", "")]
    got = ta.alfred_version_summary(rows + rows[:1])
    assert got["unique_observation_versions"] == 2
    assert got["identical_chunk_duplicates"] == 1
    assert got["revised_observations"] == 1
    assert got["gaps_between_version_intervals"] == 0
    with pytest.raises(ValueError, match="CONFLICTING_DUPLICATE"):
        ta.alfred_version_summary(rows + [("2020-01-01", "3", "2020-01-02", "2020-01-04")])
    with pytest.raises(ValueError, match="OVERLAPPING_INTERVALS"):
        ta.alfred_version_summary([rows[0], ("2020-01-01", "2", "2020-01-04", "")])
    with pytest.raises(ValueError, match="OVERLAPPING_INTERVALS"):
        ta.alfred_version_summary([("2020-01-01", "1", "2020-01-02", ""), rows[1]])
    with pytest.raises(ValueError, match="NONFINITE_VALUE"):
        ta.alfred_version_summary([("2020-01-01", "nan", "2020-01-02", "")])


def test_source_probe_rate_limit_stops_remaining_network_requests(tmp_path, monkeypatch):
    import io
    import runpy
    import socket
    import sys
    from email.message import Message
    from urllib.error import HTTPError
    import urllib.request

    probe = ta.ROOT / "scripts/research_ta_b_source_probe_20260930.py"
    owner = Path(ta.__file__)
    spec = {"probe_sha256": ta.sha(probe), "owner_sha256": ta.sha(owner),
            "transport_hostname": socket.gethostname(),
            "relay_output_directory": str(tmp_path / "receipt"), "timeout_seconds": 2,
            "maximum_metadata_bytes": 1024, "alfred_probe": None,
            "archive_metadata_requests": [{"id": "first", "url": "https://example.test/a"},
                                          {"id": "second", "url": "https://example.test/b"}]}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(spec))
    calls = []
    def limited(req, timeout):
        calls.append(req.full_url)
        headers = Message()
        headers["Retry-After"] = "120"
        raise HTTPError(req.full_url, 429, "rate limited", headers, io.BytesIO(b"rate limited"))
    monkeypatch.setattr(urllib.request, "urlopen", limited)
    monkeypatch.setattr(sys, "argv", [str(probe), str(manifest), ta.sha(manifest), str(owner)])
    runpy.run_path(str(probe), run_name="__main__")
    result = json.loads((tmp_path / "receipt/RESULT.json").read_text())
    assert calls == ["https://example.test/a"]
    first, second = result["archive_requests"]
    assert first["status"] == "FAILED" and first["http_status"] == 429
    assert first["retry_after"] == "120"
    assert second["status"] == "SKIPPED_RATE_LIMIT"
    assert "http_status" not in second



def test_b_macro_publication_requires_a_complete_later_session_and_preserves_prefix():
    opens = pd.date_range("2019-02-01T22:00Z", periods=10, freq="D")
    clocks = pd.DataFrame({"session_open": opens, "decision_time": opens + ta.TRADING_SESSION_DURATION})
    old = [("2009-01-02", "100", "2019-02-04", "2019-02-07")]
    revised = old + [("2009-01-02", "999", "2019-02-08", "")]
    initial = ta.alfred_asof_levels(old, clocks)
    full = ta.alfred_asof_levels(revised, clocks)
    first = pd.Timestamp("2019-02-06T22:00Z")
    revision = pd.Timestamp("2019-02-10T22:00Z")
    assert full.loc[clocks.decision_time < first, "value"].isna().all()
    assert (full.loc[(clocks.decision_time >= first) & (clocks.decision_time < revision), "value"] == 100).all()
    assert (full.loc[clocks.decision_time >= revision, "value"] == 999).all()
    pd.testing.assert_frame_equal(initial.loc[clocks.decision_time < revision],
                                  full.loc[clocks.decision_time < revision])
    # A known future end date never suppresses the currently known value.
    pd.testing.assert_frame_equal(initial, ta.alfred_asof_levels([("2009-01-02", "100", "2019-02-04", "")], clocks))


def test_b_macro_asof_uses_latest_observation_and_removes_missing_revision():
    opens = pd.date_range("2020-01-01T22:00Z", periods=12, freq="D")
    clocks = pd.DataFrame({"session_open": opens, "decision_time": opens + ta.TRADING_SESSION_DURATION})
    rows = [("2020-01-01", "1", "2020-01-02", ""),
            ("2020-01-02", "2", "2020-01-03", "2020-01-06"),
            ("2020-01-02", "", "2020-01-07", "")]
    result = ta.alfred_asof_levels(rows + rows[:1], clocks)
    assert result.loc[clocks.decision_time == pd.Timestamp("2020-01-05T22:00Z"), "value"].item() == 2
    assert result.loc[clocks.decision_time == pd.Timestamp("2020-01-09T22:00Z"), "value"].item() == 1
    assert result.iloc[-1].observation_date == "2020-01-01"
    broken = clocks.copy()
    broken.loc[0, "decision_time"] += pd.Timedelta(minutes=1)
    with pytest.raises(RuntimeError, match="CANONICAL_D1_CLOCK"):
        ta.alfred_asof_levels(rows, broken)
