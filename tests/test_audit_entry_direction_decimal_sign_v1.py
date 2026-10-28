import numpy as np
import pytest

from gx1.scripts.audit_entry_direction_decimal_sign_v1 import (
    confusion_by_horizon,
    decimal_midpoint_numerators,
    exact_signs_for_targets,
    score_versions,
)


def test_decimal_grid_distinguishes_equal_midpoint_and_real_small_moves():
    quotes = [('2300.000', '2300.120'), ('2300.001', '2300.119'),
              ('2300.000', '2300.122'), ('2299.999', '2300.119')]
    numerator, precision = decimal_midpoint_numerators(quotes)
    assert precision == 3
    np.testing.assert_array_equal(numerator, [4600120, 4600120, 4600122, 4600118])
    assert np.sign((float(quotes[1][0]) + float(quotes[1][1])) / 2
                   - (float(quotes[0][0]) + float(quotes[0][1])) / 2) == 1


def test_variable_decimal_precision_is_not_rounded():
    numerator, precision = decimal_midpoint_numerators([('1.2', '1.30'), ('1.201', '1.299'), ('1.201', '1.301')])
    assert precision == 3
    np.testing.assert_array_equal(numerator, [2500, 2500, 2502])


@pytest.mark.parametrize('quotes', [[(1.0, '1.2')], [('NaN', '1.2')], [('-1', '1.2')],
                                   [('1.3', '1.2')], [('1e30', '1e30')], [('1e-30', '1e-30')]])
def test_invalid_or_unrepresentable_price_fails_closed(quotes):
    with pytest.raises(RuntimeError):
        decimal_midpoint_numerators(quotes)


def test_exact_joins_missing_masks_and_confusion():
    source_times = np.array([100, 200, 300], np.int64)
    source_rows = np.array([10, 11, 12], np.int64)
    midpoint, _ = decimal_midpoint_numerators([('2300.000', '2300.120'),
                                               ('2300.001', '2300.119'),
                                               ('2300.000', '2300.122')])
    decisions = np.array([100, 200], np.int64)
    terminal = np.array([[200, 300], [300, 400]], np.int64)
    observed = np.array([[True, True], [True, False]], np.bool_)
    terminal_rows = np.array([[11, 12], [12, -1]], np.int64)
    exact = exact_signs_for_targets(source_times, source_rows, midpoint, decisions=decisions,
        entry_rows=np.array([10, 11], np.int64), terminal_times=terminal,
        observed=observed, terminal_rows=terminal_rows, cutoff_ns=400, block_rows=1)
    np.testing.assert_array_equal(exact, [[0, 1], [1, 0]])
    old = np.array([[1, 1], [1, 0]], np.int8)
    result = confusion_by_horizon(old, exact, observed, terminal < 400)
    assert result[0]['saved_nonzero_to_exact_zero'] == 1
    assert result[0]['saved_by_exact_sign_order_minus_zero_plus'][2] == [0, 1, 1]
    assert result[1]['observed_mature'] == 1
    wrong = terminal_rows.copy(); wrong[0, 0] = 12
    with pytest.raises(RuntimeError, match='TERMINAL_JOIN'):
        exact_signs_for_targets(source_times, source_rows, midpoint, decisions=decisions,
            entry_rows=np.array([10, 11], np.int64), terminal_times=terminal,
            observed=observed, terminal_rows=wrong, cutoff_ns=400)


def test_fixed_score_versions_reproduce_original_and_keep_ties_out():
    logits = np.tile(np.array([[np.log(3)], [-np.log(3)], [np.log(3)]], np.float64), (1, 60))
    stored = np.tile(np.array([[1], [-1], [1]], np.int8), (1, 60))
    exact = stored.copy(); exact[0, :] = 0
    observed = np.ones((3, 60), np.bool_)
    terminal = np.ones((3, 60), np.int64)
    original = {'accuracy': 1.0, 'logloss': -np.log(0.75), 'brier': 0.0625,
                'counts': {'eligible_cells': 3}, 'rank': {'auc': 1.0}}
    result = score_versions(logits, exact, stored, observed, terminal, 2,
                            {'model_metrics': {'per_horizon': [original] * 60}})
    assert len(result) == 60
    assert result[0]['stored']['eligible'] == 3
    assert result[0]['source_decimal']['eligible'] == 2
    assert result[0]['stored']['auc'] == result[0]['source_decimal']['auc'] == 1.0


def test_raw_reader_applies_weekly_closure_and_ignores_post_cutoff_prices(tmp_path):
    import gzip
    import json
    import pandas as pd
    from gx1.contracts.xau_tape_provenance_v1 import scheduled_weekly_closure_mask
    from gx1.scripts.audit_entry_direction_decimal_sign_v1 import _read_raw_quotes

    def candle(time, bid, ask):
        return {'time': time, 'complete': True, 'bid': {'o': bid}, 'ask': {'o': ask}}
    payload = {'instrument': 'XAU_USD', 'granularity': 'M1', 'candles': [
        candle('2024-05-03T20:59:00.000000000Z', '2300.000', '2300.120'),
        candle('2024-05-03T21:00:00.000000000Z', '1', '2'),
        candle('2024-05-05T22:00:00.000000000Z', '2300.001', '2300.119'),
        candle('2026-01-01T00:00:00.000000000Z', 'not-a-price', 'not-a-price'),
    ]}
    path = tmp_path/'source.json.gz'
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        json.dump(payload, f)
    times = pd.to_datetime(['2024-05-03T20:59:00Z', '2024-05-05T22:00:00Z'], utc=True).as_unit('ns').asi8
    source = {'time_ns': times, 'source_row': np.array([10, 11], np.int64),
              'bid_open': np.array([2300.0, 2300.001]), 'ask_open': np.array([2300.12, 2300.119])}
    out_times, rows, mid, precision = _read_raw_quotes([{'path': str(path)}], source, pd=pd,
                                                       closure_mask=scheduled_weekly_closure_mask)
    np.testing.assert_array_equal(out_times, times)
    np.testing.assert_array_equal(rows, [10, 11])
    np.testing.assert_array_equal(mid, [4600120, 4600120])
    assert precision == 3
    bad = dict(source, bid_open=np.array([2300.0, 2300.002]))
    with pytest.raises(RuntimeError, match='VENDOR_CANONICAL_FLOAT64'):
        _read_raw_quotes([{'path': str(path)}], bad, pd=pd, closure_mask=scheduled_weekly_closure_mask)


def test_target_reader_fails_closed_on_domain_and_dtype(tmp_path, monkeypatch):
    import gx1.scripts.audit_entry_direction_decimal_sign_v1 as audit

    monkeypatch.setattr(audit, 'PARENTS', 2)
    decisions = np.array([audit.START_NS + 300 * 10**9, audit.START_NS + 600 * 10**9], np.int64)
    target = {'parent_rows': np.arange(2, dtype=np.int64), 'decision_ns': decisions,
              'bar_start_ns': decisions - 300 * 10**9, 'dynamic_entry_m1_row': np.array([0, 5], np.int64),
              'terminal_time_ns': decisions[:, None] + np.arange(1, 61, dtype=np.int64)[None, :] * 60 * 10**9,
              'terminal_source_row': np.ones((2, 60), np.int64), 'observed': np.ones((2, 60), np.bool_),
              'mid_move': np.ones((2, 60), np.float64), 'mid_return_bps': np.ones((2, 60), np.float64),
              'sign': np.ones((2, 60), np.int8)}
    path = tmp_path/'target.npz'
    np.savez(path, **target)
    assert audit._read_target(path)['sign'].shape == (2, 60)
    bad = dict(target, mid_move=target['mid_move'].copy())
    bad['mid_move'][0, 0] = np.nan
    np.savez(path, **bad)
    with pytest.raises(RuntimeError, match='TARGET_HEADER_CLOCK_SIGN'):
        audit._read_target(path)
    bad = dict(target, sign=target['sign'].astype(np.int64))
    np.savez(path, **bad)
    with pytest.raises(RuntimeError, match='TARGET_DTYPE_SHAPE'):
        audit._read_target(path)


def test_synthetic_read_join_and_scores_under_actual_guards(tmp_path):
    import subprocess
    import sys
    code = r'''
import gzip, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as arrow
import pyarrow.parquet as parquet
import pyarrow.dataset
from gx1.contracts.entry_direction_probability_reference_v1 import direction_rank_diagnostics
from gx1.contracts.entry_native_direction_context_report_v1 import _panel
from gx1.contracts.xau_tape_provenance_v1 import scheduled_weekly_closure_mask
from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
from gx1.scripts import audit_entry_direction_decimal_sign_v1 as audit
out = Path(sys.argv[1]).resolve(); path = out/'synthetic.json.gz'
payload = {'instrument': 'XAU_USD', 'granularity': 'M1', 'candles': [
 {'time':'2024-05-03T20:58:00Z','complete':True,'bid':{'o':'2300.000'},'ask':{'o':'2300.120'}},
 {'time':'2024-05-03T20:59:00Z','complete':True,'bid':{'o':'2300.001'},'ask':{'o':'2300.119'}}]}
with gzip.open(path, 'wt', encoding='utf-8') as f: json.dump(payload, f)
times = pd.to_datetime(['2024-05-03T20:58:00Z','2024-05-03T20:59:00Z'], utc=True).as_unit('ns').asi8
source = {'time_ns':times, 'source_row':np.array([10, 11], np.int64),
 'bid_open':np.array([2300.000,2300.001]), 'ask_open':np.array([2300.120,2300.119])}
sys.dont_write_bytecode = True
sys.addaudithook(candle.make_guard(out, [str(path)], [], synthetic=True))
sys.setprofile(candle.no_queries_or_models)
t, rows, mids, precision = audit._read_raw_quotes([{'path':str(path)}], source, pd=pd,
 closure_mask=scheduled_weekly_closure_mask)
result = audit.exact_signs_for_targets(t, rows, mids, decisions=times[:1], entry_rows=rows[:1],
 terminal_times=times[1:].reshape(1,1), observed=np.ones((1,1),np.bool_),
 terminal_rows=rows[1:].reshape(1,1), cutoff_ns=audit.CUTOFF_NS)
assert result.tolist() == [[0]] and precision == 3
z = np.tile(np.array([[np.log(3)],[-np.log(3)],[np.log(3)]],np.float64),(1,60))
sign = np.tile(np.array([[1],[-1],[1]],np.int8),(1,60)); exact = sign.copy(); exact[0,:] = 0
obs = np.ones((3,60),np.bool_); terminal = np.ones((3,60),np.int64)
original = {'accuracy':1.0,'logloss':-np.log(0.75),'brier':0.0625,
 'counts':{'eligible_cells':3},'rank':{'auc':1.0}}
result = audit.score_versions(z, exact, sign, obs, terminal, 2,
 {'model_metrics':{'per_horizon':[original]*60}})
assert result[0]['source_decimal']['eligible'] == 2
sys.setprofile(None)
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
