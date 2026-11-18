import pytest

from gx1.audit import measure_ta_m5_projected_h4_line_v1 as audit


def test_integer_h4_close_and_fractional_projection():
    assert audit.project_line(2000.0, 100.0, 0.5, 0.25,
                              above=True, fraction=0.0) == 2010.0
    assert audit.project_line(2000.0, 100.0, 0.5, 0.25,
                              above=True, fraction=0.5) == 2012.5
    assert audit.project_line(2000.0, 100.0, 0.5, -0.25,
                              above=False, fraction=0.5) == 1987.5


def test_fractional_projection_rejects_future_bar_and_invalid_atr():
    with pytest.raises(RuntimeError, match='PROJECT_DOMAIN'):
        audit.project_line(2000.0, 100.0, 0.5, 0.25,
                           above=True, fraction=1.0)
    with pytest.raises(RuntimeError, match='PROJECT_DOMAIN'):
        audit.project_line(2000.0, 0.0, 0.5, 0.25,
                           above=True, fraction=0.25)


def test_confirmed_close_crosses_same_frozen_line():
    assert audit.classify_break(99.0, 101.0, [100.0], [100.5]) == audit.UP_BREAK
    assert audit.classify_break(101.0, 99.0, [100.0], [99.5]) == audit.DOWN_BREAK
    assert audit.classify_break(99.0, 100.0, [100.0], [100.5]) == audit.NONE
    assert audit.classify_break(101.0, 100.0, [100.0], [99.5]) == audit.NONE
    assert audit.classify_break(99.0, 100.0, [100.0], [99.0]) == audit.UP_BREAK


def test_line_snapshot_must_be_finite_and_at_most_two_slots():
    with pytest.raises(RuntimeError, match='BREAK_INPUT'):
        audit.classify_break(99.0, 101.0, [100.0], [])
    with pytest.raises(RuntimeError, match='BREAK_LINE'):
        audit.classify_break(99.0, 101.0, [float('nan')], [100.0])


def test_full_h4_clock_and_m5_break_use_only_closed_snapshot(tmp_path, monkeypatch):
    import time
    import numpy as np
    import pandas as pd

    h = pd.Timestamp('2024-06-03 02:00:00', tz='UTC').value
    times = h + np.arange(50, dtype=np.int64)*audit.M5_NS
    closes = np.full(50, 100.0)
    closes[48:] = (101.0, 102.0)
    feats = np.zeros((1, 253), dtype=np.float32)
    feats[0, audit.FIELD_MAP['atr_bps_14']] = 100.0
    feats[0, audit.FIELD_MAP['geomline_above_active_count']] = 1.0
    feats[0, audit.FIELD_MAP['geomline_above_dist_atr']] = 0.5
    feat_path, ts_path = tmp_path/'feats.npy', tmp_path/'ts.npy'
    np.save(feat_path, feats)
    np.save(ts_path, np.array([h], dtype=np.int64))
    monkeypatch.setattr(audit, '_m5_closes', lambda *_: (times, closes, [7]))
    plan = {'h4_caches': {'feats': {'path': str(feat_path)},
                          'ts': {'path': str(ts_path)}},
            'h4_oracle_rows': {'first': 0, 'last_inclusive': 0},
            'enriched': {'path': 'unused'}}
    axes = {'FIT': (np.array([1]), np.array([h+audit.H4_NS+audit.M5_NS])),
            'LATER': (np.array([2]), np.array([h+audit.H4_NS+2*audit.M5_NS]))}
    out = audit._closed_h4_and_event_axes(plan, axes, time.monotonic()+10)
    assert out['code'].tolist() == [audit.UP_BREAK, audit.NONE]
    assert out['complete_h4_bars'] == 1
    assert out['integer_formula_consistency_lines'] == 2
    full_times = h + np.arange(51, dtype=np.int64)*audit.M5_NS
    gap_times = np.delete(full_times, 49)  # 06:05 is absent.
    gap_closes = np.full(len(gap_times), 100.0)
    gap_closes[-2:] = (101.0, 102.0)
    monkeypatch.setattr(audit, '_m5_closes',
                        lambda *_: (gap_times, gap_closes, [7]))
    gap_axes = dict(axes)
    gap_axes['LATER'] = (np.array([2]),
                         np.array([h+audit.H4_NS+3*audit.M5_NS]))
    gap_out = audit._closed_h4_and_event_axes(plan, gap_axes, time.monotonic()+10)
    assert gap_out['code'].tolist() == [audit.UP_BREAK, audit.GAP]
