import json

import numpy as np

from gx1.audit.dynamic_mtf_gc_d1_bear_trace import diagnostic_trace
from gx1.contracts import gc_source_disposition_v1 as timing
from gx1.features.gc_order_flow_v1 import add_observed_structure


def _surface():
    low = np.array([12, 11, 9, 6, 9, 9, 10, 11, 4, 7, 8, 9, 2, 12, 13, 11, 10, 9.])
    close = low + 1
    opening = close + .25
    opening[8] = low[8] + .25
    n = len(low)
    start = 100 * 1440 * timing.MINUTE_NS + np.arange(n) * 1440 * timing.MINUTE_NS
    valid = np.ones(n, dtype=bool)
    valid[[7, 14, 17]] = False
    contract = np.zeros(n, dtype=np.int64)
    contract[14] = -1
    contract[16:] = 1
    clock = np.full(n, 1440)
    clock[17] = 1200
    accepted = np.full(n, 460)
    accepted[7] = 0
    surface = {'bar_start_ns': start, 'available_ns': start + 1440 * timing.MINUTE_NS,
               'observed_contract_valid': contract >= 0, 'observed_valid': valid, 'contract_index': contract,
               'observed_first_minute_ns': start, 'observed_last_minute_ns': start + 1439 * timing.MINUTE_NS,
               'observed_open_usd': opening, 'observed_high_usd': low + 3, 'observed_low_usd': low,
               'observed_close_usd': close, 'observed_volume': np.arange(n) + 3.,
               'observed_clock_minutes': clock, 'observed_accepted_minutes': accepted}
    add_observed_structure(surface)
    return surface


def test_source_trace_keeps_pivots_across_a_gap_but_ob_origin_is_strictly_earlier():
    surface = _surface()
    summary, rows = diagnostic_trace(surface)
    assert summary['status'] == 'PASS' and summary['mismatch_cells'] == 0
    assert summary['source_rows'] == sum(summary['classification_counts'].values()) == 18
    assert sum(g['rows'] for g in summary['cross_table_observed_candles']) == summary['observed_candles'] == 15
    assert rows[6]['classification'] == 'KNOWN_INACTIVE' and rows[6]['confirmed_low_row'] == 3
    assert rows[7]['classification'] == 'NO_CANDLE_ZERO_ACCEPTED_MINUTES'
    # Row 8 is a bullish candle that crosses the confirmed low. It cannot donate to its own BOS.
    assert rows[8]['classification'] == 'CENSORED_BOS_WITHOUT_PRIOR_BULL'
    assert rows[8]['whole_gap_reset'] and rows[8]['bos_down'] and rows[8]['confirmed_low_row'] == 3
    assert rows[8]['prior_bull_row'] is None and rows[8]['current_candle_bull']
    assert rows[9]['classification'] == 'CENSORED_NO_BOS_WITH_PRIOR_BULL'
    assert rows[12]['prior_bull_row'] == rows[12]['confirmed_low_row'] == 8
    assert rows[12]['classification'] == 'KNOWN_ACTIVE' and rows[12]['bear_created']
    assert rows[13]['classification'] == 'KNOWN_INACTIVE' and rows[13]['bear_created'] is False
    assert rows[14]['classification'] == 'NO_CANDLE_INVALID_CONTRACT'
    assert rows[15]['contract_segment_start'] and rows[15]['confirmed_low_row'] is None
    assert rows[16]['contract_segment_start'] and summary['adjacent_known_contract_changes'] == 1
    assert rows[17]['classification'] == 'NO_CANDLE_INCOMPLETE_CLOCK'
    assert summary['gap_impact']['resets_with_previous_confirmed_low'] == 1
    assert summary['contract_boundary_impact']['previous_low_ended_at_segment_boundary'] == 1
    assert summary['bos_down_with_prior_bull_rows'] == 1
    # A prefix cannot change when suffix data or a later contract are appended.
    prefix = {name: values[:10] for name, values in surface.items()}
    assert diagnostic_trace(prefix)[1] == rows[:10]
    json.dumps([summary, rows], allow_nan=False)


def test_source_trace_reports_storage_mask_and_value_defects_separately():
    surface = _surface()
    surface['observed_proxy_ob_bear_active'][0] = 0.0
    surface['observed_proxy_ob_bear_high'][12] += .25
    summary, rows = diagnostic_trace(surface)
    assert summary['status'] == 'MISMATCH' and summary['mismatch_cells'] == summary['mismatch_rows'] == 2
    assert summary['field_parity']['ob_bear_active']['missing_only_in_replay'] == 1
    assert summary['field_parity']['ob_bear_high']['different_nonmissing_value'] == 1
    assert rows[0]['parity_mismatches'] == ['ob_bear_active']
    assert rows[12]['parity_mismatches'] == ['ob_bear_high']
