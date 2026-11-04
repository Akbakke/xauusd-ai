"""Explain an observed GC D1 bearish order-block proxy on supplied, closed bars."""
from collections import Counter

import numpy as np

from gx1.contracts import gc_source_disposition_v1 as timing
from gx1.features.gc_order_flow_v1 import add_observed_structure
from gx1.features.htf_features import MULTI_TF_SHIFT
from gx1.features.smc_v1 import replay_smc_causal_structure_v2


PREFIX = 'observed_proxy_'
BEAR_FIELDS = ('active', 'created', 'first_retouch', 'low', 'high', 'age_bars', 'origin_volume_contracts')
PARITY_FIELDS = ('skipped_candles', 'bos_down', 'ob_bear_input_valid', 'ob_bear_state_valid',
                 *('ob_bear_' + name for name in BEAR_FIELDS))


def _require(ok, reason):
    if not ok:
        raise ValueError('GC_D1_BEAR_TRACE_' + reason)


def _parity(surface, replay, size):
    fields, errors = {}, [[] for _ in range(size)]
    for name in PARITY_FIELDS:
        stored, actual = np.asarray(surface[PREFIX + name]), replay[PREFIX + name]
        _require(stored.shape == actual.shape == (size,) and stored.dtype.kind in 'biuf', 'STORED_FIELD:' + name)
        absent, expected_absent = np.isnan(stored), np.isnan(actual)
        different_value = ~absent & ~expected_absent & (stored != actual)
        wrong_mask = absent != expected_absent
        mismatches = np.flatnonzero(wrong_mask | different_value)
        fields[name] = {'stored_known': int((~absent).sum()), 'replay_known': int((~expected_absent).sum()),
                        'both_missing': int((absent & expected_absent).sum()),
                        'missing_only_in_storage': int((absent & ~expected_absent).sum()),
                        'missing_only_in_replay': int((~absent & expected_absent).sum()),
                        'different_nonmissing_value': int(different_value.sum()), 'mismatches': len(mismatches)}
        for i in mismatches:
            errors[i].append(name)
    return fields, errors


def diagnostic_trace(surface):
    """Return population counts and a causal row ledger, without copying market prices into the report."""
    replay = {k: np.asarray(v) for k, v in surface.items() if not k.startswith(PREFIX)}
    add_observed_structure(replay)
    valid, contract_valid = replay['observed_valid'], replay['observed_contract_valid']
    opening, close = replay['observed_open_usd'], replay['observed_close_usd']
    starts, ends, contract = replay['bar_start_ns'], replay['available_ns'], replay['contract_index']
    n = len(starts)
    duration = int(MULTI_TF_SHIFT['D1'].value)
    _require(np.all(ends - starts == duration), 'D1_DURATION')
    field_parity, errors = _parity(surface, replay, n)

    coverage_keys = ('observed_clock_minutes', 'observed_accepted_minutes')
    has_coverage = all(k in replay for k in coverage_keys)
    _require(has_coverage or not any(k in replay for k in coverage_keys), 'PARTIAL_COVERAGE_INPUT')
    expected_minutes = duration // timing.MINUTE_NS
    if has_coverage:
        grid, accepted = (replay[k] for k in coverage_keys)
        _require(grid.shape == accepted.shape == (n,) and grid.dtype.kind in 'iu' and accepted.dtype.kind in 'iu'
                 and np.all((accepted >= 0) & (accepted <= grid) & (grid <= expected_minutes))
                 and np.array_equal(valid, (grid == expected_minutes) & contract_valid & (accepted > 0)),
                 'OBSERVED_VALID_OWNER_PARITY')

    # Full-grid boundaries include contract excursions hidden inside missing candles.
    boundary = np.r_[True, (contract[1:] != contract[:-1]) | ~contract_valid[1:] | ~contract_valid[:-1]]
    segment = np.cumsum(boundary)
    rows = []
    for i in range(n):
        no_candle = 'NO_CANDLE_INVALID_CONTRACT' if not contract_valid[i] else 'NO_CANDLE_VALID_CONTRACT'
        if not valid[i] and contract_valid[i] and has_coverage:
            reasons = [reason for condition, reason in ((grid[i] < expected_minutes, 'INCOMPLETE_CLOCK'),
                       (accepted[i] == 0, 'ZERO_ACCEPTED_MINUTES')) if condition]
            no_candle = 'NO_CANDLE_' + '_AND_'.join(reasons)
        row = {'row': i, 'bar_start_ns': int(starts[i]), 'available_ns': int(ends[i]),
               'observed_candle': bool(valid[i]), 'contract_valid': bool(contract_valid[i]),
               'source_contract_index': int(contract[i]), 'grid_contract_boundary': bool(i > 0 and boundary[i]),
               'classification': no_candle if not valid[i] else None, 'parity_mismatches': errors[i]}
        if has_coverage:
            row.update(clock_minutes=int(grid[i]), accepted_minutes=int(accepted[i]))
        rows.append(row)

    kept = np.flatnonzero(valid)
    segment_groups = np.split(kept, np.flatnonzero(np.diff(segment[kept]) != 0) + 1)
    transitions = Counter()
    previous_segment_end = None
    for group in segment_groups:
        if not len(group):
            continue
        if previous_segment_end is not None:
            prior = rows[previous_segment_end]
            transitions['previous_low_ended_at_segment_boundary'] += int(prior['confirmed_low_row'] is not None)
            transitions['previous_known_bear_state_at_segment_boundary'] += int(prior['bear_state_valid'])
            transitions['previous_bull_donor_ended_at_segment_boundary'] += int(
                prior['current_candle_bull'] or prior['prior_bull_row'] is not None)
        structure, _ = replay_smc_causal_structure_v2(replay['observed_high_usd'][group],
                                                       replay['observed_low_usd'][group], close[group])
        prior_bull = None
        zone_history_known = True  # The production proxy starts a fresh contract segment with known absence.
        for j, index in enumerate(group):
            i = int(index)
            skipped = replay[PREFIX + 'skipped_candles'][i]
            _require(bool(np.isnan(skipped)) == (j == 0), 'CONTRACT_SEGMENT_JOIN')
            if j:
                _require(skipped == (starts[i] - starts[group[j-1]]) // duration - 1, 'GAP_JOIN')
            reset = j > 0 and skipped > 0
            if reset:
                transitions['whole_gap_resets'] += 1
                transitions['resets_with_previous_confirmed_low'] += int(structure['last_low_idx'][j-1] >= 0)
                transitions['resets_with_previous_known_bear_state'] += int(replay[PREFIX + 'ob_bear_state_valid'][group[j-1]])
                transitions['resets_with_prior_bull_cleared'] += int(prior_bull is not None)
                zone_history_known = False
                prior_bull = None
            pivot = int(structure['last_low_idx'][j])
            low_ready = pivot >= 0
            bos = bool(structure['bos_down'][j]) if low_ready else None
            origin_ready = prior_bull is not None
            created = bool(bos and origin_ready)
            if created:
                zone_history_known = True
            known = bool(replay[PREFIX + 'ob_bear_state_valid'][i])
            active = bool(replay[PREFIX + 'ob_bear_active'][i]) if known else None
            _require((low_ready and zone_history_known) == known
                     and replay[PREFIX + 'ob_bear_input_valid'][i] == float(low_ready)
                     and (not low_ready or replay[PREFIX + 'bos_down'][i] == float(bos))
                     and (not known or replay[PREFIX + 'ob_bear_created'][i] == float(created)), 'CAUSE_VS_OWNER')
            if not low_ready:
                reason = 'NO_CONFIRMED_LOW'
            elif known:
                reason = 'KNOWN_ACTIVE' if active else 'KNOWN_INACTIVE'
            elif bos:
                reason = 'CENSORED_BOS_WITHOUT_PRIOR_BULL'
            else:
                reason = 'CENSORED_NO_BOS_WITH_PRIOR_BULL' if origin_ready else 'CENSORED_NO_BOS_WITHOUT_PRIOR_BULL'
            rows[i].update(classification=reason, contract_segment_start=(j == 0), whole_gap_reset=bool(reset),
                           skipped_whole_candles=None if j == 0 else int(skipped),
                           confirmed_low_row=int(group[pivot]) if low_ready else None, bos_down=bos,
                           close_below_confirmed_low=bool(close[i] < structure['last_low_price'][j]) if low_ready else None,
                           prior_bull_row=prior_bull, current_candle_bull=bool(close[i] > opening[i]),
                           bear_zone_history_known=zone_history_known, bear_state_valid=known,
                           bear_created=created if known else None, bear_active=active)
            if close[i] > opening[i]:
                prior_bull = i  # Becomes eligible only at a later observed candle.
        previous_segment_end = int(group[-1])

    observed = [row for row in rows if row['observed_candle']]
    cross = Counter((r['confirmed_low_row'] is not None, r['bos_down'], r['prior_bull_row'] is not None,
                     r['whole_gap_reset'], r['bear_state_valid']) for r in observed)
    dimensions = ('confirmed_low', 'bos_down', 'prior_bull_since_reset', 'whole_gap_reset', 'bear_state_valid')
    counts = Counter(r['classification'] for r in rows)
    observed_segments = [r for r in observed if r['contract_segment_start']]
    mismatch_cells = sum(len(v) for v in errors)
    return {'status': 'MISMATCH' if mismatch_cells else 'PASS', 'source_rows': n, 'observed_candles': len(observed),
            'classification_counts': dict(counts), 'coverage_inputs_available': has_coverage,
            'confirmed_low_observed_rows': sum(r['confirmed_low_row'] is not None for r in observed),
            'bos_down_with_prior_bull_rows': sum(r['bos_down'] is True and r['prior_bull_row'] is not None for r in observed),
            'grid_contract_boundary_rows_excluding_first': int(boundary[1:].sum()),
            'adjacent_known_contract_changes': int(np.sum((contract[1:] != contract[:-1]) & contract_valid[1:] & contract_valid[:-1])),
            'observed_contract_segments': len(observed_segments),
            'contract_boundary_impact': {k: transitions[k] for k in ('previous_low_ended_at_segment_boundary',
                'previous_known_bear_state_at_segment_boundary', 'previous_bull_donor_ended_at_segment_boundary')},
            'gap_impact': {k: transitions[k] for k in
                ('whole_gap_resets', 'resets_with_previous_confirmed_low', 'resets_with_previous_known_bear_state',
                 'resets_with_prior_bull_cleared')},
            'cross_table_observed_candles': [{**dict(zip(dimensions, keys)), 'rows': count} for keys, count in cross.items()],
            'field_parity': field_parity, 'mismatch_cells': mismatch_cells,
            'mismatch_rows': sum(bool(v) for v in errors)}, rows
