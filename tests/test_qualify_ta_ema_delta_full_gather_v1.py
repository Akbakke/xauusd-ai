"""Focused checks for exact population selection and streamed full-cell evidence."""
import hashlib
import time
from types import SimpleNamespace

import numpy as np
import pytest

from gx1.audit import qualify_ta_ema_delta_full_gather_v1 as gather


def test_exact_fit_later_ranges_and_batch_tails():
    positions = gather.phase_positions()
    fit, later = positions['FIT'], positions['LATER']
    assert (len(fit), len(later)) == (29574, 17489)
    assert (int(fit[0]), int(fit[-1]), int(later[0]), int(later[-1])) == (
        0, 29573, 29849, 47337)
    assert not np.intersect1d(np.concatenate((fit, later)),
                              np.arange(29574, 29849)).size
    assert len(list(gather.batch_slices('FIT'))) == 1849
    assert len(list(gather.batch_slices('LATER'))) == 1094
    assert list(gather.batch_slices('FIT'))[-1] == slice(29568, 29574)
    assert list(gather.batch_slices('LATER'))[-1] == slice(47337, 47338)


def _fixture(monkeypatch, *, bad_category=False):
    monkeypatch.setattr(gather, 'PHASES', {'FIT': (0, 17)})
    axes = {'parent_rows': np.arange(100, 117, dtype=np.int64),
            'bar_start_ns': np.arange(17, dtype=np.int64) * 300,
            'decision_ns': (np.arange(17, dtype=np.int64) + 1) * 300}
    normal = {'surfaces': {surface: {'field_names': ['f']}
                           for surface in gather.SURFACES.values()}}

    def native(_view, **kwargs):
        rows = kwargs['batch_parent_rows']
        features = {}
        for key in gather.SURFACES:
            shape = gather._shape(key, len(rows), 1)
            features[key] = np.broadcast_to(
                (rows.astype(np.float32) / 1000).reshape(
                    (len(rows),) + (1,) * (len(shape) - 1)), shape).copy()
        cats = (rows % 4).astype(np.int64).reshape(-1, 1)
        if bad_category:
            cats[0, 0] = 4
        features['ctx_cat'] = cats
        return {'witness': {'parent_rows': rows.tolist(),
                            'decision_ns': axes['decision_ns'][rows - 100].tolist()},
                'features': features}

    monkeypatch.setattr(gather.views, 'gather_native_direction_feature_batch', native)
    monkeypatch.setattr(gather.norm, 'apply_surface_normalization',
                        lambda x, _stats: x + np.float32(2))
    return axes, normal


def test_streamed_hashes_cover_every_raw_and_normalized_cell(monkeypatch):
    axes, normal = _fixture(monkeypatch)
    report = gather.gather_phase('FIT', view={'unobserved_units_profile': None},
        normal=normal, axes=axes, seq={}, current={'snap': None,
        'ctx_cont': None, 'ctx_cat': None}, bound=SimpleNamespace(mtf_sources={}),
        started=time.monotonic())
    assert (report['rows'], report['batches'], report['tail_rows'],
            report['normalization_calls']) == (17, 2, 1, 16)
    assert report['category_counts'] == [5, 4, 4, 4]
    for key in gather.SURFACES:
        shape = gather._shape(key, 17, 1)
        values = np.broadcast_to(
            (axes['parent_rows'].astype(np.float32) / 1000).reshape(
                (17,) + (1,) * (len(shape) - 1)), shape)
        expected_raw = hashlib.sha256(np.ascontiguousarray(values, dtype='<f4').tobytes()).hexdigest()
        expected_norm = hashlib.sha256(np.ascontiguousarray(
            values + np.float32(2), dtype='<f4').tobytes()).hexdigest()
        assert report['surfaces'][key]['raw_sha256'] == expected_raw
        assert report['surfaces'][key]['normalized_sha256'] == expected_norm


def test_gather_rejects_category_outside_contract(monkeypatch):
    axes, normal = _fixture(monkeypatch, bad_category=True)
    with pytest.raises(RuntimeError, match='CATEGORY_DOMAIN'):
        gather.gather_phase('FIT', view={'unobserved_units_profile': None},
            normal=normal, axes=axes, seq={}, current={'snap': None,
            'ctx_cont': None, 'ctx_cat': None}, bound=SimpleNamespace(mtf_sources={}),
            started=time.monotonic())
