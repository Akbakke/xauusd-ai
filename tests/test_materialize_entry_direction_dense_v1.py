import copy

import numpy as np
import pytest

from gx1.scripts import materialize_entry_direction_dense_v1 as dense


def fixture():
    minute = dense.MINUTE_NS
    base = np.int64(30000 * minute)
    minutes = np.delete(np.arange(66, dtype=np.int64), 7)
    mids = 100.0 + minutes.astype(np.float64) / 10.0
    mids[minutes == 8] = 100.5  # D+3 returns exactly to the first entry midpoint.
    source = {'time_ns': base + minutes * minute, 'source_row': 100 + np.arange(len(minutes), dtype=np.int64),
        'open': mids.copy(), 'bid_open': mids - .01, 'ask_open': mids + .01}
    decisions = base + np.array([5, 10], np.int64) * minute
    cutoff = int(base + 65 * minute)
    parents = np.array([0, 1], np.int64)
    pointers = np.array([source['source_row'][np.flatnonzero(minutes == m)[0]] for m in (5, 10)], np.int64)
    saved = {'parent_rows': parents, 'bar_start_ns': decisions - 5 * minute,
        'decision_ns': decisions, 'dynamic_entry_m1_row': pointers,
        'terminal_time_ns': decisions[:, None] + np.array([1, 5, 15], np.int64) * minute,
        'terminal_source_row': np.full((2, 3), -1, np.int64), 'observed': np.zeros((2, 3), bool),
        'mid_move': np.zeros((2, 3), np.float64), 'mid_return_bps': np.zeros((2, 3), np.float64),
        'sign': np.zeros((2, 3), np.int8)}
    for i, entry_minute in enumerate((5, 10)):
        entry = int(np.flatnonzero(minutes == entry_minute)[0])
        mid0 = (source['bid_open'][entry] + source['ask_open'][entry]) / 2.0
        for j, horizon in enumerate((1, 5, 15)):
            when = entry_minute + horizon
            found = np.flatnonzero(minutes == when)
            if len(found) and base + when * minute < cutoff:
                pos = int(found[0]); midt = (source['bid_open'][pos] + source['ask_open'][pos]) / 2.0
                change = midt - mid0
                saved['terminal_source_row'][i, j] = source['source_row'][pos]
                saved['observed'][i, j] = True
                saved['mid_move'][i, j] = change
                saved['mid_return_bps'][i, j] = 10000.0 * change / mid0
                saved['sign'][i, j] = np.sign(change)
    return source, saved, cutoff, base


def test_dense_times_keep_zero_missing_and_global_cutoff_distinct():
    source, saved, cutoff, base = fixture()
    out = dense.build_dense_targets(source, saved, cutoff_ns=cutoff, block_rows=1)
    assert out['terminal_time_ns'].shape == (2, 60)
    assert out['terminal_time_ns'][0, -1] == cutoff
    assert out['observed'][0, 0] and out['sign'][0, 0] == 1
    assert not out['observed'][0, 1] and out['terminal_source_row'][0, 1] == -1
    assert out['observed'][0, 2] and out['sign'][0, 2] == 0
    assert not out['observed'][0, -1]
    assert out['mid_return_bps'][0, 0] == pytest.approx(10000 * .1 / 100.5)
    assert np.all(out['sign'][~out['observed']] == 0)


def test_fold_cutoff_masks_future_values_without_confusing_zero_and_missing():
    source, saved, cutoff, base = fixture()
    out = dense.build_dense_targets(source, saved, cutoff_ns=cutoff)
    fold = int(base + 9 * dense.MINUTE_NS)
    batch = dense.causal_target_batch(out, np.array([0], np.int64), cutoff_ns=fold)
    assert batch['mature'][0, :4].tolist() == [True, True, True, False]
    assert batch['observed'][0, :4].tolist() == [True, False, True, False]
    assert batch['sign'][0, :4].tolist() == [1, 0, 0, 0]
    changed = {k: v.copy() for k, v in out.items()}
    changed['sign'][0, 3:] = -1
    changed['observed'][0, 3:] = True
    again = dense.causal_target_batch(changed, np.array([0], np.int64), cutoff_ns=fold)
    assert all(np.array_equal(batch[k], again[k]) for k in batch)
    with pytest.raises(RuntimeError, match='BATCH_DECISION_CUTOFF'):
        dense.causal_target_batch(out, np.array([1], np.int64), cutoff_ns=fold)


def test_legacy_anchor_or_original_entry_pointer_mismatch_fails_closed():
    source, saved, cutoff, _ = fixture()
    wrong = copy.deepcopy(saved)
    wrong['sign'][0, 0] *= -1
    with pytest.raises(RuntimeError, match='LEGACY_PARITY:sign'):
        dense.build_dense_targets(source, wrong, cutoff_ns=cutoff)
    wrong = copy.deepcopy(saved)
    wrong['dynamic_entry_m1_row'][0] += 1
    with pytest.raises(RuntimeError, match='ENTRY'):
        dense.build_dense_targets(source, wrong, cutoff_ns=cutoff)


def test_blocking_does_not_change_any_dense_target():
    source, saved, cutoff, _ = fixture()
    a = dense.build_dense_targets(source, saved, cutoff_ns=cutoff, block_rows=1)
    b = dense.build_dense_targets(source, saved, cutoff_ns=cutoff, block_rows=2)
    assert set(a) == set(b) and all(a[k].tobytes() == b[k].tobytes() for k in a)


def test_capped_worker_guard_can_write_and_read_only_the_declared_dense_archive(tmp_path):
    import subprocess
    import sys
    output = tmp_path.parent / 'dense_actual_guard_case'
    program = r'''
import os, sys, zipfile, zlib
from pathlib import Path
import numpy as np
from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
out = Path(sys.argv[1]); out.mkdir(exist_ok=False)
path = out / 'DENSE_TARGETS.npz'
sys.addaudithook(candle.make_guard(out, [], [], output_payload_paths=(str(path),)))
with path.open('xb') as f: np.savez_compressed(f, sign=np.array([-1, 0, 1], dtype=np.int8))
with np.load(path, allow_pickle=False) as saved: assert saved['sign'].tolist() == [-1, 0, 1]
try:
    open(out / 'unlisted.npz', 'wb')
except RuntimeError as e:
    assert 'PAYLOAD' in str(e)
else:
    raise AssertionError('unlisted payload was allowed')
'''
    result = subprocess.run([sys.executable, '-B', '-c', program, str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
