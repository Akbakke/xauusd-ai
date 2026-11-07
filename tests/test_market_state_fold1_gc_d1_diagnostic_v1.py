import numpy as np
import torch

from gx1.audit.score_market_state_time_query_fold1_oof_v1 import (
    _gc_d1_mask_columns, _paired_m5_summary, _without_gc_d1,
)
from gx1.contracts.entry_gc_mtf_history_v1 import gc_mtf_history_layout
from gx1.contracts.entry_native_direction_prefix_units_v1 import mature_fold1_prefix_units_profile


def test_gc_d1_fit_absence_masks_all_aliases_without_mutating_original():
    columns = _gc_d1_mask_columns(mature_fold1_prefix_units_profile(),
                                  gc_mtf_history_layout())
    widths = {'seq_x': 2725, 'snap_x': 2725, 'ctx_cont': 2542, 'gc_seq_d1': 399}
    shapes = {'seq_x': (1, 2), 'snap_x': (1,), 'ctx_cont': (1,),
              'gc_seq_d1': (1, 3)}
    features = {key: torch.zeros(shapes[key] + (width,))
                for key, width in widths.items()}
    for key, indices in columns.items():
        features[key][..., indices] = 1.0
    original = {key: value.clone() for key, value in features.items()}
    masked = _without_gc_d1(features, columns)
    for key, indices in columns.items():
        assert len(indices) == 102
        assert torch.count_nonzero(masked[key][..., indices]) == 0
        assert torch.equal(features[key], original[key])
    assert torch.equal(masked['seq_x'][:, -1], masked['snap_x'])


def test_paired_summary_uses_ties_and_fixed_reliability_bins():
    t = 1_727_654_400_000_000_000
    rows = [(t + i * 300_000_000_000, original, 0.5, y)
            for i, (original, y) in enumerate(
                ((0.1, 0), (0.9, 1), (0.9, 1), (0.1, 0)))]
    result = _paired_m5_summary(rows, 0.5)
    assert result['count'] == 4
    assert result['original']['auc_tie_aware'] == 1.0
    assert result['fit_absent_gc_d1']['auc_tie_aware'] == 0.5
    assert np.isclose(result['original']['brier_sum'], 0.04)
    bins = result['original']['reliability_bins']
    assert bins[1]['count'] == bins[9]['count'] == 2
    assert sum(item['count'] for item in bins) == 4
    assert len(result['weekly']) == 1 and result['weekly'][0]['count'] == 4
