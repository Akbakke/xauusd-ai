"""Freeze one conditional UP/DOWN candle ranking per native timeframe on the registered FIT prefix."""
import hashlib
from pathlib import Path

import numpy as np

from gx1.audit.dynamic_mtf_december_normalization import _axis_digest, _raw_array
from gx1.audit.dynamic_mtf_train_prefix_inputs import _read, _require
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts.entry_direction_block_rank_v1 import block_rank_pairs
from gx1.features import htf_features as htf

FIELD = 'close_open_atr'
SCHEMA = 'gx1_december_prefix_candle_reference_v1'


def _closed_window_ends(opens, issues, *, shift_ns, window, warmup):
    """Last *actual* closed bar at issue; a calendar hole does not fabricate a candle."""
    _require(opens.dtype == issues.dtype == np.int64 and opens.ndim == issues.ndim == 1
             and len(issues) > 0 and len(opens) > 0 and np.all(np.diff(opens) > 0)
             and np.all(np.diff(issues) > 0) and shift_ns > 0 and window > 0 and warmup >= 0,
             'CANDLE_AXES')
    right = np.searchsorted(opens, issues - shift_ns, side='right')
    _require(np.all(right - window >= warmup), 'CANDLE_WARMUP')
    return right - 1


def _rank_direction(raw, binary, eligible):
    _require(raw.dtype == np.float64 and binary.dtype == np.int8 and eligible.dtype == bool
             and raw.ndim == 1 and raw.shape == binary.shape == eligible.shape and np.isfinite(raw).all()
             and np.array_equal(np.isin(binary, (0, 1)), eligible) and np.all(binary[~eligible] == -1),
             'CANDLE_RANK_DOMAIN')
    y = binary[eligible]
    scores = raw[eligible]
    up, down = int((y == 1).sum()), int((y == 0).sum())
    pairs = up * down
    result = {'up': up, 'down': down, 'pairs': pairs, 'eligible_rows': len(y), 'ties': 0,
              'twice_wins': 0, 'raw_auc': None, 'negative_auc': None, 'orientation': None, 'oriented_auc': None}
    if not pairs:
        return result
    counts = block_rank_pairs(y, scores, np.zeros(len(y), dtype=np.int32))
    wins = int(counts['twice_wins'][0, 0])
    _require(int(counts['positive'][0]) == up and int(counts['negative'][0]) == down
             and 0 <= wins <= 2 * pairs, 'CANDLE_PAIR_COUNT')
    _, groups = np.unique(scores, return_inverse=True)
    k = int(groups.max()) + 1
    ties = int(np.bincount(groups[y == 0], minlength=k) @ np.bincount(groups[y == 1], minlength=k))
    numerator_negative = 2 * pairs - wins
    return {**result, 'ties': ties, 'twice_wins': wins, 'raw_auc': wins / (2 * pairs),
            'negative_auc': numerator_negative / (2 * pairs), 'orientation': 1 if wins >= pairs else -1,
            'oriented_auc': max(wins, numerator_negative) / (2 * pairs)}


def _native_labels(series, cutoff, leads):
    """Use the canonical exact, mature target query. Keep only the five native diagonals."""
    a = series.arrays
    parents, issues = a['parent_rows'], a['issue_ns']
    states = np.full((len(issues), len(target.TIMEFRAMES)), target.UNKNOWN, dtype=np.int8)
    reasons = np.zeros_like(states, dtype=np.uint8)
    obs = np.zeros_like(states, dtype=bool)
    groups = np.zeros_like(states, dtype=np.int8)
    tf = np.arange(len(target.TIMEFRAMES))
    for start in range(0, len(issues), 1024):
        part = slice(start, start + 1024)
        q = target.query_dynamic_mtf_states(series, parent_rows=parents[part], issue_ns=issues[part],
                                            cutoff_ns=cutoff, lead_minutes=leads)
        wanted = issues[part, None] + leads * target.MINUTE_NS
        masks = q['observed'][:, tf, tf]
        _require(np.array_equal(q['maturity_ns'][:, tf, tf], wanted)
                 and np.all(wanted[masks] < cutoff), 'CANDLE_TARGET_MATURITY')
        states[part], reasons[part], obs[part] = q['state'][:, tf, tf], q['reason'][:, tf, tf], masks
        groups[part] = np.where(q['current_observed'], q['current_state'], len(target.STATE_NAMES))
    _require(np.array_equal(obs, np.isin(states, (target.DOWN, target.FLAT, target.UP)))
             and np.all(states[~obs] == target.UNKNOWN), 'CANDLE_TARGET_MASK')
    counts = np.stack([np.bincount(3 * groups[obs[:, j], j] + states[obs[:, j], j], minlength=12).reshape(4, 3)
                       for j in tf])
    return states, reasons, obs, _axis_digest(counts)


def calculate_prefix_candle(policy, stage):
    """Data-only, one-shot caller owns cap and immutable publication. All scored rows precede December."""
    from gx1.contracts.entry_dynamic_mtf_state_report_v1 import verify_prefix_candle_rank_and_clock

    dec = policy['dynamic_mtf_train_prefix_dec01_20261104_001']
    s1 = policy['dynamic_mtf_native_endpoint_loss_step1_20261104_001']
    s2 = policy['dynamic_mtf_masked_december_input_normalization_20261104_001']
    geometry = policy['current_work']['dynamic_mtf_train_gates_20261104']['december_candidate_geometry']
    cutoff = stage['fit_cutoff_ns']
    _require(policy['training_enabled'] is stage['training_enabled'] is stage['model_calls_authorized'] is False
             and stage['candle_reference_authorized'] is True and stage['fit_target_view'] == dec['fit_target_view']
             and stage['candidate_geometry'] == geometry['binding'] and stage['endpoint_result'] == s1['result']
             and stage['normalization_result'] == s2['result'] and stage['feature_view'] == s2['feature_view']
             and stage['prefix_normalization'] == s2['normalization'] and stage['prefix_counts'] == dec['fit_counts']
             and stage['prefix_prior'] == dec['fit_prior'] and stage['prefix_rows'] == dec['prefix_rows'] == 41866
             and cutoff == dec['fit_cutoff_ns'] == 1733011200000000000
             and stage['held_out_rows'] == dec['held_out_rows'] == 5736
             and stage['held_out_cutoff_ns'] == dec['held_out_cutoff_ns'], 'CANDLE_REGISTERED_STAGE')
    with campaign.file_verification_scope():
        endpoint, normalized = _read(stage['endpoint_result']), _read(stage['normalization_result'])
        view = _read(stage['feature_view'])
        qualification = _read(s2['qualification'])
        proof = qualification['proof']
        _require(endpoint['real_fit_data'] == stage['fit_target_view']
                 and normalized['qualification'] == s2['qualification']
                 and normalized['feature_view'] == stage['feature_view']
                 and normalized['normalization'] == stage['prefix_normalization']
                 and view['full_feature_profile'] is True and view['native_endpoint_loss'] is True
                 and proof['prefix'] == view['prefix'] == normalized['fit_prefix']
                 and view['base_lineage'] == proof['base_lineage'], 'CANDLE_QUALIFIED_SOURCES')
        lineage = view['base_lineage']
        cache_ref = {'path': lineage['mtf_cache_manifest_path'], 'sha256': lineage['mtf_cache_manifest_sha256']}
        cache = _read(cache_ref)
        names = cache['feature_names']
        _require(names == list(htf.MULTI_TF_PER_BAR_FEATURES_V4) and names.count(FIELD) == 1
                 and cache['cache_identity_sha256'] == htf.compute_htf_v4_cache_identity(cache), 'CANDLE_CACHE')
        col = names.index(FIELD)
        series = target.load_dynamic_mtf_state_series(
            stage['fit_target_view'], expected_source_refs=s1['target_source_identities'], expected_cutoff_ns=cutoff)
        rows, issues = series.arrays['parent_rows'], series.arrays['issue_ns']
        _require(len(rows) == len(issues) == stage['prefix_rows'] and np.all(issues < cutoff)
                 and _axis_digest(rows) == proof['prefix']['parent_rows_sha256']
                 and _axis_digest(issues) == proof['prefix']['decision_ns_sha256'], 'CANDLE_PARENT_IDENTITY')
        leads = np.array([int(htf.MULTI_TF_SHIFT[tf].value) // target.MINUTE_NS for tf in target.TIMEFRAMES],
                         dtype=np.int64)
        states, reasons, observed, native_digest = _native_labels(series, cutoff, leads)
        _require(native_digest == endpoint['native_count_digest'], 'CANDLE_CURRENT_STATE_OUTCOME_PARITY')
        panels = {}
        for j, tf in enumerate(target.TIMEFRAMES):
            evidence, support = proof['tf'][tf], qualification['support']['mtf_' + tf.lower()]
            info = cache['tfs'][tf]
            root = Path(cache_ref['path']).parent
            _require(info['ts_npy'] == tf + '_ts.npy' and info['feats_npy'] == tf + '_feats.npy'
                     and info['feature_count'] == len(names) and support['field_names'] == names
                     and col not in dict(support['masking_pairs'])
                     and support['observed_counts'][col] == support['population_rows'], 'CANDLE_FIELD:' + tf)
            clock_ref = {'path': str(root / info['ts_npy']), 'sha256': info['ts_npy_sha256']}
            feat_ref = {'path': str(root / info['feats_npy']), 'sha256': info['feats_npy_sha256']}
            stop = evidence['right_index_exclusive']
            _require(0 < stop == evidence['source_row_count'] <= info['n_bars']
                     and info['causal_warmup_rows'] == evidence['source_warmup_rows']
                     and evidence['seq_len'] == view['window_rows'][tf], 'CANDLE_ROW_SCOPE:' + tf)
            opens = _raw_array(clock_ref, shape=(info['n_bars'],), dtype='int64')[:stop]
            shift = int(htf.MULTI_TF_SHIFT[tf].value)
            ids = _closed_window_ends(opens, issues, shift_ns=shift, window=evidence['seq_len'],
                                      warmup=evidence['source_warmup_rows'])
            _require(ids[-1] + 1 == stop and ids[0] + 1 - evidence['seq_len'] == evidence['left_index_inclusive']
                     and np.all(opens[ids] + shift <= issues), 'CANDLE_CLOSED_INPUT:' + tf)
            features = _raw_array(feat_ref, shape=(info['n_bars'], len(names)), dtype='float32')[:stop]
            raw = features[ids, col].astype(np.float64)
            obs = observed[:, j]
            y = states[:, j]
            eligible = obs & (y != target.FLAT)
            binary = np.where(eligible, (y == target.UP).astype(np.int8), np.int8(-1))
            ranks = _rank_direction(raw, binary, eligible)
            other = verify_prefix_candle_rank_and_clock(opens, issues, ids, raw, binary, eligible,
                                                        timeframe=tf, warmup_rows=info['causal_warmup_rows'])
            _require(all(other[k] == ranks[k] for k in ('eligible_rows', 'up', 'down', 'pairs', 'twice_wins',
                     'raw_auc', 'orientation', 'oriented_auc')) and other['selected_rows_sha256_le_i64'] == _axis_digest(ids)
                     and other['latest_closes_sha256_le_i64'] == _axis_digest(opens[ids] + shift),
                     'CANDLE_INDEPENDENT_PARITY:' + tf)
            n = int(obs.sum())
            immature = int(np.count_nonzero((reasons[:, j] & target.IMMATURE) != 0))
            missing = int(np.count_nonzero((reasons[:, j] & target.QUERY_UNAVAILABLE) != 0))
            unresolved = len(y) - n - immature - missing
            _require(unresolved >= 0 and n == endpoint['native_cells_by_tf'][tf]
                     and ranks['eligible_rows'] + np.count_nonzero(y == target.FLAT) == n,
                     'CANDLE_NATIVE_COUNTS:' + tf)
            panels[tf] = {**ranks, 'lead_minutes': int(leads[j]), 'flat': int(np.count_nonzero(y == target.FLAT)),
                          'observed_native': n, 'unknown_or_ineligible': len(y) - n, 'immature': immature,
                          'parents_with_candle_score': len(raw), 'missing_candle_scores': 0,
                          'missing_endpoint': missing, 'unresolved_on_present_endpoint': unresolved,
                          'clock': clock_ref, 'raw_features': feat_ref, 'field_index': col,
                          'source_rows_through_last_closed': stop,
                          'latest_closes_sha256_le_i64': other['latest_closes_sha256_le_i64'],
                          'selected_rows_sha256_le_i64': other['selected_rows_sha256_le_i64'],
                          'eligible_parent_rows_sha256_le_i64': _axis_digest(rows[eligible]),
                          'eligible_issue_ns_sha256_le_i64': _axis_digest(issues[eligible]),
                          'eligible_state_sha256_i8': hashlib.sha256(y[eligible].tobytes()).hexdigest(),
                          'eligible_raw_score_sha256_le_f32': hashlib.sha256(raw[eligible].astype('<f4').tobytes()).hexdigest()}
        campaign.capture_verified_file_snapshot()
    return {'schema_version': SCHEMA,
            'status': ('FROZEN_PREFIX_CANDLE_REFERENCE' if all(p['orientation'] is not None for p in panels.values())
                       else 'UNDETERMINED_PREFIX_CANDLE_REFERENCE'),
            'scope': 'descriptive fit-prefix conditional UP/DOWN rank at one native lead per TF',
            'feature': FIELD, 'positive_class': 'UP', 'fit_parents': len(issues), 'fit_cutoff_ns': cutoff,
            'fit_target_view': stage['fit_target_view'], 'candidate_geometry': stage['candidate_geometry'],
            'feature_view': stage['feature_view'], 'prefix_normalization': stage['prefix_normalization'],
            'endpoint_result': stage['endpoint_result'], 'normalization_result': stage['normalization_result'],
            'cache_manifest': cache_ref, 'native_count_digest': native_digest,
            'fit_issue_ns_sha256_le_i64': _axis_digest(issues),
            'fit_parent_rows_sha256_le_i64': _axis_digest(rows), 'panels': panels, 'model_calls': 0,
            'held_out_outcomes_read': False, 'test_data_used': False, 'independent_edge_proven': False}
