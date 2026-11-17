"""Synthetic FIT-prefix/clock/terminal checks; no GX1 data, fit admission or job."""
import numpy as np
import pytest

from gx1.audit import qualify_ta_ema_delta_fit_normalization_v1 as audit


def _case(tmp_path, *, bad=None):
    rows = np.arange(10, 15, dtype=np.int64)
    times = 1_700_000_000_000_000_000 + np.arange(5, dtype=np.int64) * 300_000_000_000
    values = np.asarray([1., 2., 5., 8., 13.], dtype=np.float32)
    if bad == 'row':
        rows[1] += 1
    if bad == 'dtype':
        values = values.astype(np.float64)
    sidecar, span = tmp_path / 'EMA.npz', tmp_path / 'SPAN.npz'
    np.savez_compressed(sidecar, row_indices=rows, times_ns=times, delta_5_atr=values)
    np.savez_compressed(span, row_indices=np.arange(10, 15, dtype=np.int64),
                        times_ns=times, native_bar_positions=np.arange(5, dtype=np.int64),
                        anchor_span_bars=np.zeros((5, 2), dtype=np.float32))
    selected_rows = np.arange(10, 13, dtype=np.int64)
    available = times[:3] + 300_000_000_000
    population = {'row_bounds': [10, 13], 'population': {
        'selected_unique_row_count': 3,
        'selected_row_indices_sha256': audit.axes.row_sha(selected_rows, 'local'),
        'available_ns_sha256': audit.axes.axis_sha(available),
    }}
    cutoff = int(available[-1]) + 1
    if bad == 'cutoff':
        cutoff = int(available[-1])
    return {'sidecar': {'path': str(sidecar)}, 'span': {'path': str(span)},
            'population': population, 'parent_rows': 3, 'cutoff_ns': cutoff}


def test_only_exact_fit_prefix_is_normalized(tmp_path):
    stat, proof = audit.fit_surface('signal', **_case(tmp_path))
    assert stat['field_names'] == ['chart.local_ema50_200_spread_delta_5_atr']
    assert stat['fit_row_count'] == proof['rows'] == 3
    assert proof['last_available_ns'] < _case(tmp_path)['cutoff_ns']
    assert proof['statistics_sha256'] == stat['stats_sha256']
    assert proof['raw_values_sha256'] == audit._sha_array(
        np.asarray([1., 2., 5.], dtype=np.float32), '<f4')


@pytest.mark.parametrize('bad', ['row', 'dtype', 'cutoff'])
def test_bad_axes_dtype_or_future_fit_time_fail_closed(tmp_path, bad):
    with pytest.raises(RuntimeError, match='TA_EMA_DELTA_NORMALIZATION_'):
        audit.fit_surface('signal', **_case(tmp_path, bad=bad))


def test_terminal_requires_exact_success_or_failure():
    ref = {'path': '/tmp/plan.json', 'sha256': 'a' * 64}
    ok = audit.terminal_record('COMPLETE', plan_ref=ref, claim=ref, result=ref)
    fail = audit.terminal_record('FAILED_CONSUMED', plan_ref=ref, claim=ref,
                                 error='ValueError:invalid')
    assert ok['exit_code'] == 0 and fail['exit_code'] == 1
    assert ok['remaining_invocations'] == fail['remaining_invocations'] == 0
    with pytest.raises(RuntimeError, match='TERMINAL_STATE'):
        audit.terminal_record('COMPLETE', plan_ref=ref, claim=ref, error='wrong')
