import numpy as np

from gx1.audit import run_market_state_F_m5_gc_ablation_v1 as audit
from gx1.features.entry_specialist_feature_groups_v1 import compile_compact_entry_execution_profile


def test_gc_mask_covers_actual_compact_snapshot_and_keeps_price_features():
    physical = compile_compact_entry_execution_profile()['physical_field_names']
    names = tuple(['signal.' + name for name in physical['signal']]
                  + ['ctx_cat.' + name for name in physical['ctx_cat']]
                  + ['mtf_' + tf.lower() + '.' + name
                     for tf in ('M5', 'M15', 'H1', 'H4', 'D1')
                     for name in physical['mtf']])
    columns = audit.feature_indices(names)
    assert len(columns['gc']) == 2410
    assert 'signal.ctx_cont.gc_m5_observed_buy_avg_executed_trade_contracts' in (
        names[i] for i in columns['gc'])
    assert 'signal.ema20_slope_atr' in (names[i] for i in columns['without_gc'])
    assert np.array_equal(np.sort(np.r_[columns['gc'], columns['without_gc']]),
                          columns['full'])


def test_paired_score_uses_only_mature_and_known_future_side():
    allowed = np.array([[4, 7, 7, 7, 7], [1, 7, 7, 7, 7],
                        [2, 7, 7, 7, 7], [7, 7, 7, 7, 7]], dtype=np.uint8)
    eligible = np.ones_like(allowed, dtype=np.bool_)
    mature = np.ones_like(allowed, dtype=np.bool_)
    mature[1, 0] = False
    prediction = {'full': np.array([[0.9, 0.8], [0.9, 0.8],
                                    [0.1, 0.8], [0.5, 0.8]]),
                  'without_gc': np.array([[0.9, 0.6], [0.9, 0.6],
                                          [0.1, 0.6], [0.5, 0.6]])}
    scored = audit.score(prediction,
                         {'allowed': allowed, 'eligible': eligible, 'mature': mature},
                         (0.5, 0.5))
    assert scored['activity']['count'] == 2
    assert scored['conditional_side']['count'] == 1
    assert scored['conditional_side']['full_brier_sum'] < scored['conditional_side']['without_gc_brier_sum']
