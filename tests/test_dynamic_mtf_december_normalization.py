import hashlib
import json

import numpy as np
import pytest

from gx1.audit.dynamic_mtf_december_normalization import _axis_digest, _closed_rows, _fit_axes


def _ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def test_closed_rows_includes_exact_issue_close_and_keeps_later_values_out():
    starts = np.array([1, 5, 9, 15], dtype=np.int64)
    assert _closed_rows(starts, 3, 12) == 3
    assert _closed_rows(starts, 3, 11) == 2
    with pytest.raises(ValueError, match='SOURCE_CLOCK'):
        _closed_rows(starts[::-1], 3, 12)


def test_fit_axes_hash_raw_and_reserved_clocks_without_decoding_other_members(tmp_path):
    minute = 60_000_000_000
    bar = 5 * minute
    parent = np.arange(8, dtype=np.int64)
    starts = bar * (parent + 1)
    issue = starts + bar
    physical = tmp_path / 'physical.npz'
    # The object-array member would fail if the adapter attempted to decode it.
    np.savez(physical, parent_rows=parent, bar_start_ns=starts, decision_ns=issue,
             outcome=np.array([object()], dtype=object))
    fit_n, reserved_end = 3, 6
    fit = {'parent_range': [0, fit_n], 'parents': fit_n}
    held = {'parent_range': [fit_n, reserved_end], 'parents': reserved_end - fit_n}
    arrays = {}
    for name, values in [('parent_rows', parent), ('bar_start_ns', starts), ('issue_ns', issue)]:
        key = name + '_sha256_le_i64'
        fit[key] = _axis_digest(values[:fit_n])
        held[key] = _axis_digest(values[fit_n:reserved_end])
        if name != 'bar_start_ns':
            leaf = tmp_path / (name + '.npy')
            np.save(leaf, values[:fit_n])
            arrays[name] = _ref(leaf)
    target = tmp_path / 'SERIES.json'
    cut, outer = int(issue[fit_n]), int(issue[reserved_end])
    target.write_text(json.dumps({'metadata': {'parents': fit_n, 'cutoff_ns': cut}, 'arrays': arrays}))
    data, expected, observer = _fit_axes(_ref(physical), _ref(target), fit, cut, held, outer)
    assert all(len(values) == fit_n for values in data.values())
    assert expected['rows'] == fit_n and observer['rows'] == reserved_end
    assert observer['source'] == _ref(physical) and observer['held_out']['cutoff_ns'] == outer
    assert observer['decision_ns_sha256'] == _axis_digest(issue[:reserved_end])
    assert observer['fit']['decision_ns_sha256'] == expected['decision_ns_sha256']
    held['issue_ns_sha256_le_i64'] = fit['issue_ns_sha256_le_i64']
    with pytest.raises(ValueError, match='AXIS_SHA:decision_ns'):
        _fit_axes(_ref(physical), _ref(target), fit, cut, held, outer)
