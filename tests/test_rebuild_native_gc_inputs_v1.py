"""Mechanics/algebra checks, not evidence of historical feature quality."""
import numpy as np

from gx1.contracts import gc_source_disposition_v1 as timing
from gx1.features.gc_order_flow_v1 import L1FeatureCollectorV2, L1_FIELDS_V2
from gx1.scripts.rebuild_native_gc_inputs_v1 import l1_masks, selected_minutes


def test_order_counts_do_not_change_existing_book_features():
    origin = timing.STUDY_START_NS
    events = np.array([[origin, 100, 102, 20, 30, 2, 3],
                       [origin + timing.MINUTE_NS // 2, 100, 102, 20, 30, 4, 6]], np.int64)
    original = L1FeatureCollectorV2(np.array([0]), 1)
    extended = L1FeatureCollectorV2(np.array([0]), 1, include_order_counts=True)
    original.consume(events[:, :5])
    extended.consume(events)
    original.finish(origin + timing.MINUTE_NS)
    extended.finish(origin + timing.MINUTE_NS)
    a, am = original.values([True])
    b, bm = extended.values([True])
    assert np.array_equal(am, bm)
    for key in L1_FIELDS_V2:
        np.testing.assert_array_equal(a[key], b[key])
    values, mask = extended.order_count_values([True])
    assert mask.tolist() == [True]
    assert values['tw_bid_orders'][0] == 3
    assert values['tw_ask_orders'][0] == 4.5
    assert values['tw_bid_contracts_per_order'][0] == 7.5
    assert values['tw_ask_contracts_per_order'][0] == 7.5
    assert values['tw_order_count_imbalance'][0] == -0.2


def test_unavailable_counts_preserve_valid_book_and_mask_counts():
    origin = timing.STUDY_START_NS
    collector = L1FeatureCollectorV2(np.array([0]), 1, include_order_counts=True)
    collector.consume(np.array([[origin, 100, 102, 20, 30, 0, 3]], np.int64))
    collector.finish(origin + timing.MINUTE_NS)
    _, mask = collector.values([True])
    counts, count_mask = collector.order_count_values([True])
    assert mask.tolist() == [True]
    assert count_mask.tolist() == [False]
    assert all(np.isnan(v[0]) for v in counts.values())


def test_roll_is_available_one_minute_after_prior_day():
    daily = np.array([[10, 0, 0], [0, 20, 0]], np.uint64)
    selected, chosen, _ = selected_minutes(daily, [(2024, 6), (2024, 8)])
    day = 24 * 60
    assert chosen.tolist() == [-1, 0, 1]
    assert selected[day] == -1
    assert selected[day + 1] == 0
    assert selected[2 * day] == 0
    assert selected[2 * day + 1] == 1


def test_book_mask_requires_same_contract_clean_prior_and_complete_integral():
    keys = ('bad_bbo', 'gap', 'bad_recv', 'snapshot', 'reset',
            'duplicate_trade_signature', 'file_seam')
    quality = {k: np.zeros(5, np.uint64) for k in keys}
    quality.update(records=np.ones(5, np.uint64), clean_complete_bbo=np.ones(5, np.uint64))
    ofi, bbo = l1_masks(quality, np.array([0, 0, 1, 1, 1]),
                         np.array([timing.MINUTE_NS] * 4 + [0]))
    assert ofi.tolist() == [False, True, False, True, True]
    assert bbo.tolist() == [False, True, False, True, False]
    quality['gap'][3] = 1
    ofi, _ = l1_masks(quality, np.array([0, 0, 1, 1, 1]), np.full(5, timing.MINUTE_NS))
    assert not ofi[3] and not ofi[4]
