"""Fixed275 causal evaluation strata and pure-direction metrics.

This report owner changes no feature, model, normalization or decision policy.
Contexts describe raw admitted observations, including partial-support proxies;
conditional score differences are not individual-feature causal attribution.
Metadata import is standard-library-only. Numerical dependencies are lazy.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

SCHEMA_VERSION = 'gx1_native_direction_context_map_v1'
TAG_SCHEMA_VERSION = 'gx1_native_direction_context_tags_v1'
REPORT_SCHEMA_VERSION = 'gx1_native_direction_context_report_v1'
HORIZONS_MINUTES = (1, 5, 15)
TIMEFRAMES = ('M5', 'M15', 'H1', 'H4', 'D1')
MINUTE_NS = 60_000_000_000
BAR_NS = 5 * MINUTE_NS
MAX_NS = 2**63 - 1
# Pair side states: unknown / observed zero / observed event. Partial-side
# observations remain distinct; only the explicitly declared joint may collapse.
_PAIR_LABELS = tuple(a + '__' + b for a in ('first_unknown', 'first_zero', 'first_event')
    for b in ('second_unknown', 'second_zero', 'second_event'))
_SIGN_LABELS = ((-2, 'UNKNOWN'), (-1, 'NEGATIVE'), (0, 'OBSERVED_ZERO'), (1, 'POSITIVE'))
_KNOWN_SIGN_LABELS = ('NEGATIVE', 'ZERO', 'POSITIVE')
_JOINT_LABELS = tuple(a + '__' + b for a in _KNOWN_SIGN_LABELS for b in _KNOWN_SIGN_LABELS)
# Immutable specification: names, code/label pairs. These are fixed evaluation
# strata, never train filters, learned thresholds or feature selection.
CONTEXT_SPEC = (
    ('m5_sweep_pair', tuple(enumerate(_PAIR_LABELS))),
    ('m5_ob_retouch_pair', tuple(enumerate(_PAIR_LABELS))),
    ('m5_known_delta_sign', _SIGN_LABELS),
    ('m5_book_ofi_sign', _SIGN_LABELS),
    ('m5_book_mid_response_sign', _SIGN_LABELS),
    ('h1_ema20_slope_atr_sign', _SIGN_LABELS[1:]),
    ('h1_atr_z_sign', _SIGN_LABELS[1:]),
    ('m5_known_delta_mid_response_joint', tuple(enumerate(_JOINT_LABELS + ('UNKNOWN',)))),
    ('m5_ob_retouch_delta_response_joint', ((0, 'NO_RETOUCH'), (1, 'UNKNOWN'))
        + tuple((2 + side * 9 + pair, label + '__' + signs)
            for side, label in enumerate(('BULL', 'BEAR', 'BOTH'))
            for pair, signs in enumerate(_JOINT_LABELS))),
)
# Every selected value uses the registry's own observation field. No local or
# ctx_cont alias is also counted. Source columns and domains are explicit.
_GC = (
    ('sweep_up', 'gc_m5_observed_proxy_sweep_up_event', 'observed_proxy_sweep_up_event', 'observed_valid', True),
    ('sweep_down', 'gc_m5_observed_proxy_sweep_down_event', 'observed_proxy_sweep_down_event', 'observed_valid', True),
    ('ob_bull', 'gc_m5_observed_proxy_ob_bull_first_retouch', 'observed_proxy_ob_bull_first_retouch', 'observed_proxy_ob_bull_state_valid', True),
    ('ob_bear', 'gc_m5_observed_proxy_ob_bear_first_retouch', 'observed_proxy_ob_bear_first_retouch', 'observed_proxy_ob_bear_state_valid', True),
    ('known_delta', 'gc_m5_observed_known_delta_contracts', 'observed_known_delta_contracts', 'observed_valid', False),
    ('book_ofi', 'gc_m5_observed_book_ofi_contracts', 'observed_book_ofi_contracts', 'observed_book_flow_valid', False),
    ('mid_response', 'gc_m5_observed_book_mid_response_ticks', 'observed_book_mid_response_ticks', 'observed_book_valid', False),
)
_H1 = ('ema20_slope_atr', 'atr_z', 'atr_bps_14')


def _require(value, reason):
    if not value:
        raise ValueError('NATIVE_DIRECTION_CONTEXT_' + reason)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(x in '0123456789abcdef' for x in value)


def _spec():
    return [{'name': name, 'categories': [{'code': code, 'label': label} for code, label in labels]}
        for name, labels in CONTEXT_SPEC]


def native_direction_context_map(*, gc_layout, mtf_field_names):
    """Compile a small report map from caller-verified native field metadata.

    ``mtf_field_names`` is the shared native ordered list or an exact five-TF
    mapping with that same list. This does not replace layout/view validation.
    No files, arrays, normalization or targets are read at metadata time.
    """
    _require(isinstance(gc_layout, Mapping) and gc_layout.get('schema_version') == 'gx1_gc_mtf_history_layout_v1', 'GC_LAYOUT')
    _require(_hash(gc_layout.get('contract_sha256')) and gc_layout['contract_sha256']
        == _sha({k: v for k, v in gc_layout.items() if k != 'contract_sha256'}), 'GC_LAYOUT_SEAL')
    if isinstance(mtf_field_names, Mapping):
        _require(set(mtf_field_names) == set(TIMEFRAMES), 'MTF_TIMEFRAMES')
        _require(all(mtf_field_names[tf] == mtf_field_names['H1'] for tf in TIMEFRAMES), 'MTF_SHARED_FIELD_ORDER')
        mtf_field_names = mtf_field_names['H1']
    _require(isinstance(mtf_field_names, (tuple, list)) and len(mtf_field_names) > 0
        and all(isinstance(n, str) and n for n in mtf_field_names)
        and len(set(mtf_field_names)) == len(mtf_field_names) and all(n in mtf_field_names for n in _H1), 'MTF_FIELDS')
    names = gc_layout['field_names']['M5']
    _require(isinstance(names, list) and len(names) == len(set(names)), 'GC_FIELD_ORDER')
    source = {item['name']: item for item in gc_layout['source_fields']['M5']}
    selected = {}
    for key, name, column, validity, binary in _GC:
        mask = name + '_observed'; item = source.get(name, {})
        _require(name in names and mask in names and item.get('column') == column
            and item.get('table') == 'TRADE_FEATURES_M5.parquet'
            and item.get('availability_field') == mask and item.get('validity_column') == validity, 'SOURCE_FIELD:' + name)
        value_index, mask_index = names.index(name), names.index(mask)
        _require([value_index, mask_index] in gc_layout['value_mask_pairs']['M5']
            and (not binary or name in gc_layout['binary_fields']['M5']), 'SOURCE_MASK_DOMAIN:' + name)
        selected[key] = {'name': name, 'value_index': value_index, 'mask': mask,
            'mask_index': mask_index, 'binary': binary, 'source': _copy(item)}
    value = {'schema_version': SCHEMA_VERSION, 'gc_layout_sha256': gc_layout['contract_sha256'],
        'gc_source_registry': _copy(gc_layout['source_registry']),
        'gc_field_names_sha256': {tf: _sha(gc_layout['field_names'][tf]) for tf in TIMEFRAMES},
        'gc_widths': {tf: len(gc_layout['field_names'][tf]) for tf in TIMEFRAMES},
        'window_rows': _copy(gc_layout['window_rows']), 'mtf_field_names_sha256': _sha(list(mtf_field_names)),
        'mtf_width': len(mtf_field_names), 'gc_m5_fields': selected,
        'h1_fields': {n: list(mtf_field_names).index(n) for n in _H1}, 'contexts': _spec(),
        'semantics': {
            'raw_features_only': True, 'gc_slot': 'own_M5_last_native_row', 'h1_slot': 'own_H1_last_native_row',
            'sweep': 'observed confirmed-level wick-through and close-reclaim proxy; first=up,second=down',
            'ob_retouch': 'observed first later wick intersection with causal OB proxy; first=bull,second=bear',
            'known_delta': 'admitted known-buy minus known-sell contracts; unknown aggressor excluded',
            'book_ofi': 'admitted best-quote-size flow proxy, not order identity or deep liquidity',
            'mid_response': 'last minus first admitted BBO mid ticks; partial coverage is not a complete path',
            'ema20_slope_atr': '(EMA20[t]-EMA20[t-5])/positive_Wilder_ATR14 on closed H1 bars',
            'atr_z': '(rolling50(TR)-rolling50_mean(rolling50(TR)))/rolling50_std; closed H1, no epsilon',
            'h1_support': 'finite fully warmed native fields; no fabricated observation mask',
            'joint_unknown': 'delta/response unknown if either own mask false; OB joint also requires both retouch masks',
            'no_retouch': 'both retouch sides observed zero, regardless of pressure availability',
            'interpretation': 'fixed evaluation strata; no model feature/filter/threshold or causal attribution'},
        'source_owners': ['gx1/features/gc_order_flow_v1.py', 'gx1/features/smc_v1.py',
            'gx1/features/htf_features.py', 'gx1/scripts/materialize_build_canonical_features_v1.py']}
    return {**value, 'contract_sha256': _sha(value)}


def _require_map(value):
    _require(isinstance(value, Mapping) and value.get('schema_version') == SCHEMA_VERSION
        and _hash(value.get('contract_sha256'))
        and value['contract_sha256'] == _sha({k: v for k, v in value.items() if k != 'contract_sha256'}), 'MAP_SEAL')
    _require(value.get('contexts') == _spec() and set(value.get('gc_m5_fields', {})) == {x[0] for x in _GC}
        and set(value.get('h1_fields', {})) == set(_H1), 'FIXED_CONTEXT_SPEC')
    return value


def _axis(value, *, size, reason, nonnegative=True):
    import numpy as np
    array = np.asarray(value)
    _require(array.dtype == np.int64 and array.shape == (size,)
        and (not nonnegative or bool((array >= 0).all())), reason)
    return array


def tag_native_direction_contexts(raw_features, *, context_map, witness):
    """Tag a bounded268 raw batch; targets and predicted values are not inputs."""
    import numpy as np
    mapping = _require_map(context_map)
    keys = {'seq_x', 'snap_x', 'ctx_cont', 'ctx_cat'} | {'seq_' + tf.lower() for tf in TIMEFRAMES} | {'gc_seq_' + tf.lower() for tf in TIMEFRAMES}
    _require(isinstance(raw_features, Mapping) and set(raw_features) == keys, 'RAW14_KEYS')
    _require(isinstance(witness, Mapping) and witness.get('raw_values_not_normalized') is True
        and _hash(witness.get('view_sha256')), 'RAW_VIEW_WITNESS')
    local = raw_features['seq_x']
    _require(isinstance(local, np.ndarray) and local.ndim == 3 and local.shape[0] > 0 and local.shape[1] > 0, 'LOCAL_HEADER')
    size = len(local)
    for key, array in raw_features.items():
        rank = 2 if key in ('snap_x', 'ctx_cont', 'ctx_cat') else 3
        _require(isinstance(array, np.ndarray) and array.ndim == rank and len(array) == size
            and array.dtype == (np.int64 if key == 'ctx_cat' else np.float32)
            and all(x > 0 for x in array.shape[1:]), 'RAW_HEADER:' + key)
    _require(raw_features['snap_x'].shape[1] == local.shape[2]
        and np.array_equal(local[:, -1], raw_features['snap_x']), 'LAST_LOCAL_SNAPSHOT')
    parents = _axis(witness['parent_rows'], size=size, reason='PARENT_AXIS')
    starts = _axis(witness['bar_start_ns'], size=size, reason='START_AXIS')
    decisions = _axis(witness['decision_ns'], size=size, reason='DECISION_AXIS')
    _require(len(set(parents.tolist())) == size and bool((starts % BAR_NS == 0).all())
        and bool((starts <= MAX_NS - BAR_NS).all()) and np.array_equal(decisions, starts + BAR_NS), 'ORIGINAL_PARENT_CLOCK')
    _require(isinstance(witness.get('multi_tf'), Mapping) and set(witness['multi_tf']) == set(TIMEFRAMES), 'WITNESS_TIMEFRAMES')
    for tf in TIMEFRAMES:
        rows = mapping['window_rows'][tf]
        _require(raw_features['seq_' + tf.lower()].shape[1:] == (rows, mapping['mtf_width'])
            and raw_features['gc_seq_' + tf.lower()].shape[1:] == (rows, mapping['gc_widths'][tf]), 'TF_HEADERS:' + tf)
        windows = witness['multi_tf'][tf]
        _require(isinstance(windows, list) and len(windows) == size, 'TF_WITNESS_ROWS:' + tf)
        for window, decision in zip(windows, decisions):
            ids = _axis(window['native_row_indices'], size=rows, reason='TF_NATIVE_IDS:' + tf)
            available = _axis(window['available_ns'], size=rows, reason='TF_AVAILABLE:' + tf)
            _require(bool((ids[1:] > ids[:-1]).all()) and bool((available[1:] > available[:-1]).all())
                and bool((available % MINUTE_NS == 0).all()) and bool((available <= decision).all()), 'CAUSAL_TF_WITNESS:' + tf)
    values = {}
    for key, spec in mapping['gc_m5_fields'].items():
        raw = raw_features['gc_seq_m5'][:, -1, spec['value_index']]
        mask = raw_features['gc_seq_m5'][:, -1, spec['mask_index']]
        _require(bool(np.isin(mask, (0, 1)).all()) and bool(np.isfinite(raw).all())
            and bool((raw[mask == 0] == 0).all())
            and (not spec['binary'] or bool(np.isin(raw[mask == 1], (0, 1)).all())), 'GC_VALUE_MASK_DOMAIN:' + key)
        if spec['binary']:
            values[key] = np.where(mask == 0, 0, np.where(raw == 0, 1, 2)).astype(np.int8)
        else:
            values[key] = np.where(mask == 0, -2, np.sign(raw)).astype(np.int8)
    tags = {'m5_sweep_pair': 3 * values['sweep_up'] + values['sweep_down'],
        'm5_ob_retouch_pair': 3 * values['ob_bull'] + values['ob_bear'],
        'm5_known_delta_sign': values['known_delta'], 'm5_book_ofi_sign': values['book_ofi'],
        'm5_book_mid_response_sign': values['mid_response']}
    h1 = raw_features['seq_h1'][:, -1]
    for name, index in mapping['h1_fields'].items():
        _require(bool(np.isfinite(h1[:, index]).all()) and (name != 'atr_bps_14' or bool((h1[:, index] > 0).all())), 'H1_WARMED_DOMAIN:' + name)
    tags['h1_ema20_slope_atr_sign'] = np.sign(h1[:, mapping['h1_fields']['ema20_slope_atr']]).astype(np.int8)
    tags['h1_atr_z_sign'] = np.sign(h1[:, mapping['h1_fields']['atr_z']]).astype(np.int8)
    known = (values['known_delta'] != -2) & (values['mid_response'] != -2)
    joint = np.full(size, 9, dtype=np.int8)
    joint[known] = 3 * (values['known_delta'][known] + 1) + values['mid_response'][known] + 1
    tags['m5_known_delta_mid_response_joint'] = joint
    retouch = tags['m5_ob_retouch_pair']; joined = np.full(size, 1, dtype=np.int8)
    joined[retouch == 4] = 0
    for side, event in enumerate((7, 5, 8)):
        select = (retouch == event) & known
        joined[select] = 2 + side * 9 + joint[select]
    tags['m5_ob_retouch_delta_response_joint'] = joined
    return {'schema_version': TAG_SCHEMA_VERSION, 'context_map_sha256': mapping['contract_sha256'],
        'feature_view_sha256': witness['view_sha256'], 'parent_rows': parents.copy(), 'decision_ns': decisions.copy(),
        'tags': {name: np.asarray(tags[name], dtype=np.int8).copy() for name, _ in CONTEXT_SPEC}}


def _panel(logits, sign, observed, mature):
    import numpy as np
    nonzero = sign != 0; eligible = observed & nonzero & mature
    masks = {'cells': np.ones_like(observed), 'observed_cells': observed, 'missing_cells': ~observed,
        'observed_up_cells': observed & (sign == 1), 'observed_down_cells': observed & (sign == -1),
        'observed_tie_cells': observed & ~nonzero, 'mature_cells': mature, 'immature_cells': ~mature,
        'eligible_cells': eligible, 'eligible_up_cells': eligible & (sign == 1), 'eligible_down_cells': eligible & (sign == -1),
        'immature_observed_nonzero_cells': observed & nonzero & ~mature, 'mature_tie_cells': observed & ~nonzero & mature,
        'immature_tie_cells': observed & ~nonzero & ~mature, 'mature_missing_cells': ~observed & mature,
        'immature_missing_cells': ~observed & ~mature, 'zero_logit_cells': logits == 0,
        'eligible_zero_logit_cells': eligible & (logits == 0)}
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    _require(counts['eligible_cells'] + counts['observed_tie_cells'] + counts['immature_observed_nonzero_cells']
        + counts['missing_cells'] == counts['cells'], 'COUNT_PARTITION')
    z = logits[eligible].astype(np.float64); y = sign[eligible] == 1; predicted = z > 0
    p = np.empty_like(z); positive = z >= 0
    p[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    ez = np.exp(z[~positive]); p[~positive] = ez / (1.0 + ez)
    tp, tn = int((predicted & y).sum()), int((~predicted & ~y).sum())
    fp, fn = int((predicted & ~y).sum()), int((~predicted & y).sum())
    n = len(z); actual_up, actual_down = tp + fn, tn + fp
    losses = np.logaddexp(0.0, np.where(y, -z, z))
    metrics = {'logloss': float(np.sum(losses / n)) if n else None,
        'brier': float(np.mean((p - y) ** 2)) if n else None,
        'accuracy': (tp + tn) / n if n else None,
        'balanced_accuracy': 0.5 * (tp / actual_up + tn / actual_down) if actual_up and actual_down else None}
    _require(all(value is None or bool(np.isfinite(value)) for value in metrics.values()), 'FINITE_METRICS')
    bins = np.minimum((p * 10).astype(np.int64), 9)
    reliability = []
    for index in range(10):
        take = bins == index; count = int(take.sum())
        reliability.append({'bin': index, 'lower': index / 10, 'upper': (index + 1) / 10,
            'upper_inclusive': index == 9, 'count': count,
            'mean_probability_up': float(p[take].mean()) if count else None,
            'observed_up_fraction': float(y[take].mean()) if count else None})
    return {'parent_rows': len(sign), 'counts': counts, **metrics, 'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'up_down_counts': {'actual_up': actual_up, 'actual_down': actual_down,
            'predicted_up': tp + fp, 'predicted_down': tn + fn}, 'reliability_10bin': reliability}


def score_native_direction_logits(logits, *, sign, observed, decision_ns, terminal_time_ns, cutoff_ns, context_tags=None):
    """Score own mature non-tie labels with strict z>0 UP; all others reported.

    Headers/domains are validated even if no label is eligible. Arbitrary
    caller row order is preserved; neither contexts nor logits choose rows.
    """
    import numpy as np
    _require(isinstance(logits, np.ndarray) and logits.dtype in (np.float32, np.float64)
        and logits.ndim == 2 and logits.shape[0] > 0 and logits.shape[1] == 3
        and bool(np.isfinite(logits).all()), 'LOGIT_HEADER_DOMAIN')
    size = len(logits)
    _require(isinstance(sign, np.ndarray) and sign.dtype == np.int8 and sign.shape == (size, 3), 'SIGN_HEADER')
    _require(isinstance(observed, np.ndarray) and observed.dtype == np.bool_ and observed.shape == (size, 3), 'OBSERVED_HEADER')
    _require(isinstance(decision_ns, np.ndarray) and decision_ns.dtype == np.int64 and decision_ns.shape == (size,), 'DECISION_HEADER')
    _require(isinstance(terminal_time_ns, np.ndarray) and terminal_time_ns.dtype == np.int64 and terminal_time_ns.shape == (size, 3), 'TERMINAL_HEADER')
    _require(type(cutoff_ns) is int and 0 < cutoff_ns <= MAX_NS, 'CUTOFF')
    _require(bool(np.isin(sign, (-1, 0, 1)).all()) and bool((sign[~observed] == 0).all()), 'SIGN_DOMAIN_MISSING_STORAGE')
    _require(bool((decision_ns > 0).all()) and bool((decision_ns < cutoff_ns).all())
        and bool((decision_ns <= MAX_NS - 15 * MINUTE_NS).all()), 'DECISION_CUTOFF_OVERFLOW')
    expected = decision_ns[:, None] + np.asarray(HORIZONS_MINUTES, dtype=np.int64)[None, :] * MINUTE_NS
    _require(np.array_equal(terminal_time_ns, expected), 'EXACT_TERMINAL_CLOCK')
    mature = terminal_time_ns < cutoff_ns
    panels = [{ 'horizon_minutes': horizon, **_panel(logits[:, h], sign[:, h], observed[:, h], mature[:, h])}
        for h, horizon in enumerate(HORIZONS_MINUTES)]
    contexts = {}; map_sha = None
    if context_tags is not None:
        _require(isinstance(context_tags, Mapping) and set(context_tags) == {'schema_version', 'context_map_sha256',
            'feature_view_sha256', 'parent_rows', 'decision_ns', 'tags'}
            and context_tags['schema_version'] == TAG_SCHEMA_VERSION
            and _hash(context_tags['context_map_sha256']) and _hash(context_tags['feature_view_sha256']), 'TAG_HEADER')
        parents = context_tags['parent_rows']
        _require(isinstance(parents, np.ndarray) and parents.dtype == np.int64 and parents.shape == (size,)
            and bool((parents >= 0).all()) and len(set(parents.tolist())) == size
            and isinstance(context_tags['decision_ns'], np.ndarray) and context_tags['decision_ns'].dtype == np.int64
            and np.array_equal(context_tags['decision_ns'], decision_ns), 'TAG_ORIGINAL_AXES')
        _require(isinstance(context_tags['tags'], Mapping) and set(context_tags['tags']) == {name for name, _ in CONTEXT_SPEC}, 'TAG_INVENTORY')
        map_sha = context_tags['context_map_sha256']
        for name, labels in CONTEXT_SPEC:
            tag = context_tags['tags'][name]
            _require(isinstance(tag, np.ndarray) and tag.dtype == np.int8 and tag.shape == (size,)
                and bool(np.isin(tag, [code for code, _ in labels]).all()), 'TAG_DOMAIN:' + name)
            categories = []
            for code, label in labels:
                take = tag == code
                categories.append({'code': code, 'label': label, 'parent_rows': int(take.sum()),
                    'per_horizon': [{'horizon_minutes': horizon,
                        **_panel(logits[take, h], sign[take, h], observed[take, h], mature[take, h])}
                        for h, horizon in enumerate(HORIZONS_MINUTES)]})
            contexts[name] = {'categories': categories}
    return {'schema_version': REPORT_SCHEMA_VERSION, 'parent_rows': size, 'horizons_minutes': list(HORIZONS_MINUTES),
        'exclusive_cutoff_ns': cutoff_ns, 'prediction_rule': 'UP iff logit_up > 0; otherwise DOWN',
        'eligibility': 'observed & sign!=0 & terminal_time_ns<exclusive_cutoff_ns',
        'counts': {'parent_rows': size, **{key: sum(row['counts'][key] for row in panels) for key in panels[0]['counts']}},
        'per_horizon': panels, 'context_map_sha256': map_sha, 'contexts': contexts,
        'context_scores_are_causal_attribution': False, 'feature_or_trade_policy_changed': False}
