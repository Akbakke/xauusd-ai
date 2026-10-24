"""Synthetic275 report semantics only; no native/data/fit/edge qualification."""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from gx1.contracts import entry_native_direction_context_report_v1 as owner

MINUTE = 60_000_000_000


def _seal(value):
    value = copy.deepcopy(value)
    value['contract_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return value


def _metadata():
    names, fields, pairs, binary = [], [], [], []
    for _, name, column, validity, is_binary in owner._GC:
        mask = name + '_observed'; pairs.append([len(names), len(names) + 1])
        names.extend((name, mask))
        fields.append({'name': name, 'column': column, 'table': 'TRADE_FEATURES_M5.parquet',
            'availability_field': mask, 'validity_column': validity, 'owner': 'synthetic'})
        if is_binary: binary.append(name)
    per_tf = {tf: (names if tf == 'M5' else [tf + '_value', tf + '_observed']) for tf in owner.TIMEFRAMES}
    layout = _seal({'schema_version': 'gx1_gc_mtf_history_layout_v1', 'field_names': per_tf,
        'source_fields': {'M5': fields}, 'value_mask_pairs': {'M5': pairs}, 'binary_fields': {'M5': binary},
        'source_registry': {'path': '/synthetic/registry.json', 'sha256': 'a' * 64},
        'window_rows': {'M5': 16, 'M15': 64, 'H1': 96, 'H4': 96, 'D1': 252}})
    return layout, ['atr_bps_14', 'ema20_slope_atr', 'atr_z']


def _case(size=3):
    layout, mtf_names = _metadata()
    mapping = owner.native_direction_context_map(gc_layout=layout, mtf_field_names=mtf_names)
    decisions = (1000 * 24 * 60 + np.arange(size, dtype=np.int64) * 5) * MINUTE
    features = {'seq_x': np.zeros((size, 2, 4), dtype=np.float32),
        'snap_x': np.zeros((size, 4), dtype=np.float32), 'ctx_cont': np.zeros((size, 2), dtype=np.float32),
        'ctx_cat': np.zeros((size, 1), dtype=np.int64)}
    windows = {}
    for tf, minutes, phase in (('M5', 5, 0), ('M15', 15, 0), ('H1', 60, 0), ('H4', 240, 120), ('D1', 1440, 1320)):
        rows = mapping['window_rows'][tf]
        features['seq_' + tf.lower()] = np.zeros((size, rows, len(mtf_names)), dtype=np.float32)
        features['gc_seq_' + tf.lower()] = np.zeros((size, rows, mapping['gc_widths'][tf]), dtype=np.float32)
        lane = []
        for decision in decisions:
            step, offset = minutes * MINUTE, phase * MINUTE
            last = (int(decision) - offset) // step * step + offset
            available = last - np.arange(rows - 1, -1, -1, dtype=np.int64) * step
            lane.append({'native_row_indices': list(range(1000, 1000 + rows)), 'available_ns': available.tolist()})
        windows[tf] = lane
    features['seq_h1'][:, :, 0] = 1
    for field in mapping['gc_m5_fields'].values():
        features['gc_seq_m5'][:, -1, field['mask_index']] = 1
    witness = {'parent_rows': list(range(100, 100 + size)), 'bar_start_ns': (decisions - 5 * MINUTE).tolist(),
        'decision_ns': decisions.tolist(), 'source_positions': list(range(1000, 1000 + size)),
        'local_sequence_clock_sha256': 'b' * 64, 'view_sha256': 'c' * 64,
        'multi_tf': windows, 'raw_values_not_normalized': True}
    return mapping, features, witness


def _set(features, mapping, key, raw, observed=1):
    field = mapping['gc_m5_fields'][key]
    features['gc_seq_m5'][:, -1, field['value_index']] = raw
    features['gc_seq_m5'][:, -1, field['mask_index']] = observed


def _targets(decisions, cutoff=None):
    decisions = np.asarray(decisions, dtype=np.int64)
    return {'sign': np.ones((len(decisions), 3), dtype=np.int8),
        'observed': np.ones((len(decisions), 3), dtype=bool), 'decision_ns': decisions,
        'terminal_time_ns': decisions[:, None] + np.array((1, 5, 15), dtype=np.int64)[None, :] * MINUTE,
        'cutoff_ns': int(decisions.max() + 20 * MINUTE) if cutoff is None else cutoff}


def test_metadata_module_has_only_standard_library_top_level_imports():
    tree = ast.parse(Path(owner.__file__).read_text())
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all(not any(x in ast.unparse(node) for x in ('numpy', 'torch', 'gx1.')) for node in imports)
    assert 'sign' not in inspect.signature(owner.tag_native_direction_contexts).parameters
    assert 'observed' not in inspect.signature(owner.tag_native_direction_contexts).parameters


def test_context_map_binds_exact_names_own_masks_and_fixed_75_categories():
    layout, fields = _metadata()
    mapping = owner.native_direction_context_map(gc_layout=layout, mtf_field_names={tf: fields for tf in owner.TIMEFRAMES})
    assert [field['name'] for field in mapping['gc_m5_fields'].values()] == [
        'gc_m5_observed_proxy_sweep_up_event', 'gc_m5_observed_proxy_sweep_down_event',
        'gc_m5_observed_proxy_ob_bull_first_retouch', 'gc_m5_observed_proxy_ob_bear_first_retouch',
        'gc_m5_observed_known_delta_contracts', 'gc_m5_observed_book_ofi_contracts',
        'gc_m5_observed_book_mid_response_ticks']
    assert len(mapping['contexts']) == 9
    assert sum(len(item['categories']) for item in mapping['contexts']) == 75
    for key, name, _, _, _ in owner._GC:
        field = mapping['gc_m5_fields'][key]
        assert layout['field_names']['M5'][field['value_index']] == name
        assert [field['value_index'], field['mask_index']] in layout['value_mask_pairs']['M5']
    mapping['contexts'][0]['categories'].clear()
    assert len(owner.native_direction_context_map(gc_layout=layout, mtf_field_names=fields)['contexts'][0]['categories']) == 9


@pytest.mark.parametrize('fault', ('column', 'mask_pair', 'binary'))
def test_context_map_rejects_wrong_source_domain(fault):
    layout, fields = _metadata(); layout.pop('contract_sha256')
    if fault == 'column': layout['source_fields']['M5'][0]['column'] = 'different_proxy'
    elif fault == 'mask_pair': layout['value_mask_pairs']['M5'][0] = [0, 3]
    else: layout['binary_fields']['M5'].clear()
    with pytest.raises(ValueError, match='SOURCE_'):
        owner.native_direction_context_map(gc_layout=_seal(layout), mtf_field_names=fields)


def test_all_pair_states_preserve_unknown_zero_event_and_both():
    mapping, features, witness = _case(9)
    first, second = np.repeat(np.arange(3), 3), np.tile(np.arange(3), 3)
    for up, down in (('sweep_up', 'sweep_down'), ('ob_bull', 'ob_bear')):
        _set(features, mapping, up, first == 2, first > 0)
        _set(features, mapping, down, second == 2, second > 0)
    output = owner.tag_native_direction_contexts(features, context_map=mapping, witness=witness)
    assert output['tags']['m5_sweep_pair'].tolist() == list(range(9))
    assert output['tags']['m5_ob_retouch_pair'].tolist() == list(range(9))


def test_signs_and_all_29_retouch_delta_response_joint_states():
    mapping, features, witness = _case(29)
    bull, bear = np.zeros(29), np.zeros(29)
    bull[2:11] = 1; bear[11:20] = 1; bull[20:] = 1; bear[20:] = 1
    _set(features, mapping, 'ob_bull', bull, np.r_[1, 0, np.ones(27)])
    _set(features, mapping, 'ob_bear', bear)
    delta = np.r_[0, 0, np.tile(np.repeat((-1, 0, 1), 3), 3)]
    response = np.r_[0, 0, np.tile((-1, 0, 1), 9)]
    _set(features, mapping, 'known_delta', delta)
    _set(features, mapping, 'mid_response', response)
    _set(features, mapping, 'book_ofi', np.r_[0, -1, 0, np.ones(26)], np.r_[0, np.ones(28)])
    output = owner.tag_native_direction_contexts(features, context_map=mapping, witness=witness)
    assert output['tags']['m5_ob_retouch_delta_response_joint'].tolist() == list(range(29))
    assert output['tags']['m5_known_delta_mid_response_joint'][2:11].tolist() == list(range(9))
    assert output['tags']['m5_book_ofi_sign'][:4].tolist() == [-2, -1, 0, 1]


def test_joint_unknown_never_imputes_pressure_and_no_retouch_remains_known():
    mapping, features, witness = _case()
    _set(features, mapping, 'known_delta', 0, 0)
    _set(features, mapping, 'ob_bull', [0, 1, 1])
    _set(features, mapping, 'ob_bear', [0, 0, 0], [1, 1, 0])
    tags = owner.tag_native_direction_contexts(features, context_map=mapping, witness=witness)['tags']
    assert tags['m5_known_delta_mid_response_joint'].tolist() == [9, 9, 9]
    assert tags['m5_ob_retouch_delta_response_joint'].tolist() == [0, 1, 1]
    assert tags['m5_ob_retouch_pair'].tolist() == [4, 7, 6]


def test_tags_preserve_shuffled_axes_nonmidnight_native_clocks_and_copies():
    mapping, features, witness = _case()
    features['seq_h1'][:, -1, mapping['h1_fields']['ema20_slope_atr']] = [-1, 0, 1]
    features['seq_h1'][:, -1, mapping['h1_fields']['atr_z']] = [1, 0, -1]
    order = [2, 0, 1]
    features = {name: value[order].copy() for name, value in features.items()}
    for key in ('parent_rows', 'bar_start_ns', 'decision_ns', 'source_positions'):
        witness[key] = [witness[key][i] for i in order]
    witness['multi_tf'] = {tf: [lane[i] for i in order] for tf, lane in witness['multi_tf'].items()}
    output = owner.tag_native_direction_contexts(features, context_map=mapping, witness=witness)
    assert output['parent_rows'].tolist() == [102, 100, 101]
    assert output['decision_ns'].tolist() == witness['decision_ns']
    assert output['tags']['h1_ema20_slope_atr_sign'].tolist() == [1, -1, 0]
    assert output['tags']['h1_atr_z_sign'].tolist() == [-1, 1, 0]
    output['parent_rows'][0] = 999
    assert witness['parent_rows'][0] == 102
    assert witness['multi_tf']['D1'][0]['available_ns'][-1] % (1440 * MINUTE) == 1320 * MINUTE


@pytest.mark.parametrize('fault', ('mask_fraction', 'masked_value', 'event_two', 'nan', 'h1_nan', 'atr_zero'))
def test_tag_domain_guards(fault):
    mapping, features, witness = _case()
    if fault == 'mask_fraction': _set(features, mapping, 'known_delta', 0, .5)
    elif fault == 'masked_value': _set(features, mapping, 'known_delta', 1, 0)
    elif fault == 'event_two': _set(features, mapping, 'sweep_up', 2)
    elif fault == 'nan': _set(features, mapping, 'known_delta', np.nan)
    elif fault == 'h1_nan': features['seq_h1'][:, -1, mapping['h1_fields']['atr_z']] = np.nan
    else: features['seq_h1'][:, -1, mapping['h1_fields']['atr_bps_14']] = 0
    with pytest.raises(ValueError, match='DOMAIN'):
        owner.tag_native_direction_contexts(features, context_map=mapping, witness=witness)


@pytest.mark.parametrize('fault', ('future', 'duplicate_ids', 'duplicate_clock'))
def test_causal_native_witness_is_checked_without_reselecting_rows(fault):
    mapping, features, witness = _case()
    window = witness['multi_tf']['H1'][0]
    if fault == 'future': window['available_ns'][-1] = witness['decision_ns'][0] + MINUTE
    elif fault == 'duplicate_ids': window['native_row_indices'][-1] = window['native_row_indices'][-2]
    else: window['available_ns'][-1] = window['available_ns'][-2]
    with pytest.raises(ValueError, match='CAUSAL_TF_WITNESS'):
        owner.tag_native_direction_contexts(features, context_map=mapping, witness=witness)


def test_targets_cannot_enter_tagger_and_snapshot_and_raw_map_are_guarded():
    mapping, features, witness = _case()
    with pytest.raises(TypeError):
        owner.tag_native_direction_contexts(features, context_map=mapping, witness=witness, sign=np.ones((3, 3)))
    features['snap_x'][0, 0] = 1
    with pytest.raises(ValueError, match='LAST_LOCAL_SNAPSHOT'):
        owner.tag_native_direction_contexts(features, context_map=mapping, witness=witness)
    features['snap_x'][0, 0] = 0; witness['raw_values_not_normalized'] = False
    with pytest.raises(ValueError, match='RAW_VIEW_WITNESS'):
        owner.tag_native_direction_contexts(features, context_map=mapping, witness=witness)


def test_scores_preserve_maturity_ties_missing_strictzero_and_confusion():
    targets = _targets([105 * MINUTE, 100 * MINUTE, 115 * MINUTE, 110 * MINUTE], cutoff=120 * MINUTE)
    targets['sign'][:] = [[1, -1, 0], [-1, 1, 1], [0, 1, -1], [0, 0, 0]]
    targets['observed'][:] = [[1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1]]
    logits = np.array([[0, -2, 8], [-1000, 2, 1000], [7, 0, -3], [9, -9, 0]], dtype=np.float32)
    report = owner.score_native_direction_logits(logits, **targets)
    assert report['counts']['cells'] == 12 and report['counts']['eligible_cells'] == 5
    assert report['counts']['observed_tie_cells'] == 3 and report['counts']['missing_cells'] == 2
    assert report['counts']['immature_observed_nonzero_cells'] == 2
    assert report['counts']['zero_logit_cells'] == 3 and report['counts']['eligible_zero_logit_cells'] == 1
    h1, h5, h15 = report['per_horizon']
    assert h1['confusion'] == {'tn': 1, 'fp': 0, 'fn': 1, 'tp': 0}
    assert h1['brier'] == .125 and h1['balanced_accuracy'] == .5
    assert abs(h1['logloss'] - np.log(2) / 2) < 1e-15
    assert [p['counts']['eligible_cells'] for p in (h1, h5, h15)] == [2, 2, 1]
    assert h15['balanced_accuracy'] is None and h15['brier'] == h15['logloss'] == 0
    assert all(sum(b['count'] for b in panel['reliability_10bin']) == panel['counts']['eligible_cells'] for panel in report['per_horizon'])
    json.dumps(report, allow_nan=False)


def test_extreme_finite_logits_and_fixed_reliability_endpoints():
    targets = _targets([100 * MINUTE, 105 * MINUTE, 110 * MINUTE])
    logits = np.tile(np.array([-1000, 0, 1000], dtype=np.float64)[:, None], (1, 3))
    report = owner.score_native_direction_logits(logits, **targets)
    for panel in report['per_horizon']:
        assert [b['count'] for b in panel['reliability_10bin']] == [1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        assert panel['confusion'] == {'tn': 0, 'fp': 0, 'fn': 2, 'tp': 1}
    huge = owner.score_native_direction_logits(np.full((3, 3), -1e308), **targets)
    assert all(np.isfinite(p['logloss']) and abs(p['logloss'] / 1e308 - 1) <= 8 * np.finfo(np.float64).eps
        and p['brier'] == 1 for p in huge['per_horizon'])


def test_no_eligible_returns_null_metrics_and_explicit_denominators():
    targets = _targets([100 * MINUTE, 105 * MINUTE]); targets['sign'][:] = 0; targets['observed'][:] = False
    report = owner.score_native_direction_logits(np.zeros((2, 3), np.float32), **targets)
    assert report['counts']['eligible_cells'] == 0 and report['counts']['missing_cells'] == 6
    for panel in report['per_horizon']:
        assert all(panel[name] is None for name in ('logloss', 'brier', 'accuracy', 'balanced_accuracy'))
        assert panel['confusion'] == dict.fromkeys(('tn', 'fp', 'fn', 'tp'), 0)
        assert all(b['count'] == 0 and b['mean_probability_up'] is b['observed_up_fraction'] is None for b in panel['reliability_10bin'])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('fault', ('sign_dtype', 'observed_dtype', 'decision_dtype', 'terminal_dtype', 'nan_logit', 'missing_nonzero'))
def test_empty_eligible_does_not_bypass_headers_or_domains(fault):
    targets = _targets([100 * MINUTE]); targets['sign'][:] = 0; targets['observed'][:] = False
    logits = np.zeros((1, 3), np.float32)
    if fault == 'sign_dtype': targets['sign'] = targets['sign'].astype(np.int64)
    elif fault == 'observed_dtype': targets['observed'] = targets['observed'].astype(np.int8)
    elif fault == 'decision_dtype': targets['decision_ns'] = targets['decision_ns'].astype(np.float64)
    elif fault == 'terminal_dtype': targets['terminal_time_ns'] = targets['terminal_time_ns'].astype(np.float64)
    elif fault == 'nan_logit': logits[0, 0] = np.nan
    else: targets['sign'][0, 0] = 1
    with pytest.raises(ValueError): owner.score_native_direction_logits(logits, **targets)


def test_wrong_terminal_clock_and_bool_cutoff_are_rejected():
    targets = _targets([100 * MINUTE]); logits = np.zeros((1, 3), np.float32)
    targets['terminal_time_ns'][0, 0] += 1
    with pytest.raises(ValueError, match='EXACT_TERMINAL_CLOCK'): owner.score_native_direction_logits(logits, **targets)
    targets['terminal_time_ns'][0, 0] -= 1; targets['cutoff_ns'] = True
    with pytest.raises(ValueError, match='CUTOFF'): owner.score_native_direction_logits(logits, **targets)


def test_context_panels_partition_original_rows_and_do_not_depend_on_targets():
    mapping, features, witness = _case()
    tags = owner.tag_native_direction_contexts(features, context_map=mapping, witness=witness)
    targets = _targets(tags['decision_ns']); logits = np.zeros((3, 3), np.float32)
    report = owner.score_native_direction_logits(logits, **targets, context_tags=tags)
    for context in report['contexts'].values():
        assert sum(c['parent_rows'] for c in context['categories']) == 3
        for h in range(3):
            assert sum(c['per_horizon'][h]['counts']['eligible_cells'] for c in context['categories']) == 3
    targets['sign'] *= -1
    after = owner.tag_native_direction_contexts(features, context_map=mapping, witness=witness)
    assert all(np.array_equal(tags['tags'][name], after['tags'][name]) for name, _ in owner.CONTEXT_SPEC)
    tags['decision_ns'] = tags['decision_ns'][::-1].copy()
    with pytest.raises(ValueError, match='TAG_ORIGINAL_AXES'):
        owner.score_native_direction_logits(logits, **targets, context_tags=tags)
