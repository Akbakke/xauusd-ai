"""Exact additive normalization extension for the qualified two-anchor inputs.

Only the new span statistics are fitted by the caller. Every old per-field
statistic is copied by name. Parent physical lineage remains explicit; this
does not relabel a 2725/253-wide cache as the expanded 2727/255-wide input.
"""
from __future__ import annotations

from collections.abc import Mapping
import json

import numpy as np

from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE

SPAN_FIELDS = ('geomline_above_anchor_span_bars', 'geomline_below_anchor_span_bars')
EMA_DELTA_FIELD = 'ema50_200_spread_delta_5_atr'
EMA_DELTA_SCHEMA = 'gx1_exact_fold1_ema_delta_normalization_extension_v1'
SURFACES = ('signal', 'mtf_m5', 'mtf_m15', 'mtf_h1', 'mtf_h4', 'mtf_d1')
PARENT_BASE_SHA = '3167248a761b06e08a09de3fed3a8ea56a94cd698a52aa3bd9b08e3ff25227f5'
PARENT_ARTIFACT = {
    'path': '/home/andre2/GX1_RUNS/MARKET_STATE_MATURE_FOLD1_NORMALIZATION_20261107_001/PREFIX_NORMALIZATION_FOLD_1.json',
    'sha256': 'e24a2b8d1ae09f6f10c2de7a7097353b52aa2f2118287535e242778272022d54',
}
PHYSICAL_RESULT = {
    'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_INPUTS_20261112_001/RESULT.json',
    'sha256': '8878d17a5bf114fcf9cd5eb0e74792ddc5df068b95b67951b594759d5b2b05bb',
}
FOLD2_PARENT_BASE_SHA = '20123245fb1a7c6d4336c61c3edae1ca695c1b1a75a17f89e8fda13232886f42'
FOLD2_PARENT_ARTIFACT = {
    'path': '/home/andre2/GX1_RUNS/MARKET_STATE_MATURE_PREFIX_NORMALIZATION_20261107_001/PREFIX_NORMALIZATION_FOLD_2.json',
    'sha256': '32f917bb9cf262f57afe2f60d58025b6679a3ffc5874dbe2d3b49382283ff9bc',
}
FOLD2_PHYSICAL_RESULT = {
    'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_SPAN_REUSE_20261113_002/RESULT.json',
    'sha256': '8f2acfafc459cc9a2fbcb7b588c40eb008d1e75c7e47835bbc55f24b68f32cc2',
}
FOLD2_CUTOFF_NS = 1735603200000000000
FOLD3_PARENT_BASE_SHA = '3c49368d96ffa9cb06e3db08dd288632904b7733a4c3891ef95f60214a490c37'
FOLD4_PARENT_BASE_SHA = 'e5358e8d70eb4f72a1e8a4ffb055dc992e14e3051cdf7c3bd24c369a85c5f74b'
FOLD5_PARENT_BASE_SHA = 'da7b9126fcae1a80261c091bdc0a96c18da6ec872e290a6c1886b5721e033fc8'
FIVEFOLD_PHYSICAL_RESULT = {
    'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_FIVEFOLD_SPAN_EXTENSION_20261114_002/RESULT.json',
    'sha256': 'e2309fcdd18100b5ac0b9734161d21ef00ee68d9cd4ada545873f046d88cb30d',
}
FOLD_BINDINGS = {
    PARENT_BASE_SHA: (1, PARENT_ARTIFACT, PHYSICAL_RESULT),
    FOLD2_PARENT_BASE_SHA: (2, FOLD2_PARENT_ARTIFACT, FOLD2_PHYSICAL_RESULT),
    FOLD3_PARENT_BASE_SHA: (3, {
        'path': '/home/andre2/GX1_RUNS/MARKET_STATE_MATURE_PREFIX_NORMALIZATION_20261107_001/PREFIX_NORMALIZATION_FOLD_3.json',
        'sha256': 'd8f41ce422f564f7d747c605f92a4ec83d5d2dc98806bd1b2e1f9d696c90a0cd',
    }, FIVEFOLD_PHYSICAL_RESULT),
    FOLD4_PARENT_BASE_SHA: (4, {
        'path': '/home/andre2/GX1_RUNS/MARKET_STATE_MATURE_PREFIX_NORMALIZATION_20261107_001/PREFIX_NORMALIZATION_FOLD_4.json',
        'sha256': '9cac18f0509d8a343c6a777532650a3caa06cb82bf43ac15c5cc17d292c8f7de',
    }, FIVEFOLD_PHYSICAL_RESULT),
    FOLD5_PARENT_BASE_SHA: (5, {
        'path': '/home/andre2/GX1_RUNS/MARKET_STATE_MATURE_PREFIX_NORMALIZATION_20261107_001/PREFIX_NORMALIZATION_FOLD_5.json',
        'sha256': '36af91debeccd26f7c9411049c296a222840abca563566cd3f29065354d49847',
    }, FIVEFOLD_PHYSICAL_RESULT),
}
STAT_VECTORS = ('center', 'scale', 'binary_mask', 'categorical_mask', 'scale_source',
                'train_transformed_min', 'train_transformed_max')


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('ANCHOR_SPAN_NORMALIZATION_' + reason)


def copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def span_names(surface):
    return [('chart.' if surface == 'signal' else '') + name for name in SPAN_FIELDS]


def ema_delta_names(surface):
    return [('chart.local_' if surface == 'signal' else '') + EMA_DELTA_FIELD]


def _restamp_surface(stats):
    names = stats['field_names']
    stats.update(field_count=len(names), field_names_sha256=norm._field_names_sha256(names),
                 binary_field_count=sum(stats['binary_mask']),
                 categorical_field_count=sum(stats['categorical_mask']))
    stats['stats_sha256'] = norm._stats_sha256(
        field_names=names, center=np.asarray(stats['center'], dtype=np.float32),
        scale=np.asarray(stats['scale'], dtype=np.float32),
        binary_mask=np.asarray(stats['binary_mask'], dtype=np.uint8),
        categorical_mask=np.asarray(stats['categorical_mask'], dtype=np.uint8),
        categorical_domains=stats['categorical_domains'],
        train_transformed_min=np.asarray(stats['train_transformed_min'], dtype=np.float64),
        train_transformed_max=np.asarray(stats['train_transformed_max'], dtype=np.float64),
        unobserved_units_profile_sha256=stats.get('unobserved_units_profile_sha256'),
        observed_fit_counts=stats.get('observed_fit_counts'),
        scale_source=stats['scale_source'])
    return stats


def merge_surface(old, addition, *, field_names, units_profile_sha):
    """Name-ordered union; no fit, clipping, scale invention or new masks."""
    names = list(field_names)
    original = list(old['field_names'])
    extra = list(addition['field_names']) if addition is not None else []
    need(len(names) == len(set(names)) and len(original) == len(set(original))
         and len(extra) == len(set(extra)) and not set(original).intersection(extra)
         and [name for name in names if name not in extra] == original
         and set(names) == set(original) | set(extra), 'EXACT_NAME_ADDITION')
    if addition is not None:
        need(addition['fit_row_count'] == old['fit_row_count']
             and addition['surface'] == old['surface'], 'SAME_PHYSICAL_POPULATION')
    old_index, new_index = ({n: i for i, n in enumerate(a)} for a in (original, extra))
    result = copy(old)
    for key in STAT_VECTORS:
        result[key] = [(old[key][old_index[n]] if n in old_index else addition[key][new_index[n]])
                       for n in names]
    result['field_names'] = names
    result['categorical_domains'] = copy({**old['categorical_domains'],
        **(addition['categorical_domains'] if addition is not None else {})})
    if 'unobserved_units_profile_sha256' in old:
        result['unobserved_units_profile_sha256'] = units_profile_sha
        result['observed_fit_counts'] = [
            old['observed_fit_counts'][old_index[n]] if n in old_index else old['fit_row_count']
            for n in names]
    return _restamp_surface(result)


def old_field_preservation(parent, expanded):
    """Check all old values/domains/support by name, independent of new hashes."""
    need(parent['ctx_cat'] == expanded['ctx_cat'] and parent['lineage'] == expanded['lineage'],
         'PARENT_LINEAGE_AND_CATEGORY_PRESERVED')
    counts = {}
    for surface in norm.EXPECTED_SURFACES:
        old, new = parent['surfaces'][surface], expanded['surfaces'][surface]
        indices = [new['field_names'].index(name) for name in old['field_names']]
        need(old['fit_row_count'] == new['fit_row_count'], 'OLD_FIT_ROWS:' + surface)
        for key in STAT_VECTORS:
            selected = [new[key][i] for i in indices]
            dtype = ('<f4' if key in ('center','scale') else '<f8'
                     if key in ('train_transformed_min','train_transformed_max') else 'u1'
                     if key in ('binary_mask','categorical_mask') else None)
            identical = (old[key] == selected if dtype is None else
                np.asarray(old[key],dtype=dtype).tobytes() == np.asarray(selected,dtype=dtype).tobytes())
            need(identical, 'OLD_FIELD_STAT:' + surface + ':' + key)
        need(old['categorical_domains'] == new['categorical_domains'], 'OLD_DOMAINS:' + surface)
        if 'observed_fit_counts' in old:
            need(old['observed_fit_counts'] == [new['observed_fit_counts'][i] for i in indices]
                 and old['unfitted_estimates'] == new['unfitted_estimates'], 'OLD_SUPPORT:' + surface)
        counts[surface] = len(indices)
    need(len(parent['temporal_aliases']) == len(expanded['temporal_aliases']), 'ALIAS_COUNT')
    for a, b in zip(parent['temporal_aliases'], expanded['temporal_aliases']):
        need({k:v for k,v in a.items() if k not in ('signal_index','ctx_cont_index')} ==
             {k:v for k,v in b.items() if k not in ('signal_index','ctx_cont_index')}, 'ALIAS_METADATA')
        for surface, key in (('signal','signal_index'), ('ctx_cont','ctx_cont_index')):
            need(parent['surfaces'][surface]['field_names'][a[key]] ==
                 expanded['surfaces'][surface]['field_names'][b[key]], 'ALIAS_NAME')
    return dict(old_fields_per_surface=counts, all_old_stat_vectors_exact=True,
                old_category_and_support_values_exact=True, aliases_exact_by_name=True,
                parent_lineage_unchanged=True, old_statistics_refitted=False)


def extend_anchor_span_normalization(parent, new_statistics):
    """Extend one exact mature TRAIN fold; preserve every old statistic."""
    from gx1.features.entry_specialist_feature_groups_v1 import compile_full_ta_entry_execution_profile
    need(isinstance(parent, Mapping) and parent.get('contract_sha256') in FOLD_BINDINGS
         and 'ta_projection' not in parent and 'anchor_span_extension' not in parent, 'EXACT_PARENT')
    need(isinstance(new_statistics, Mapping) and set(new_statistics) == set(SURFACES),
         'SIX_NEW_STAT_SURFACES')
    fields = {key:list(value) for key,value in
              compile_full_ta_entry_execution_profile()['source_field_names'].items()}
    need(fields['signal'].count(ema_delta_names('signal')[0]) == 1
         and fields['mtf'].count(ema_delta_names('mtf_m5')[0]) == 1,
         'LEGACY_FIELD_RECONSTRUCTION')
    fields['signal'].remove(ema_delta_names('signal')[0])
    fields['mtf'].remove(ema_delta_names('mtf_m5')[0])
    full_names = {surface:list(fields['mtf' if surface.startswith('mtf_') else surface])
                  for surface in norm.EXPECTED_SURFACES}
    old_names = {surface:[n for n in names if n not in span_names(surface)]
                 for surface, names in full_names.items()}
    fold_id, parent_artifact, physical_result = FOLD_BINDINGS[parent['contract_sha256']]
    old_units = units.mature_fold1_prefix_units_profile() if fold_id == 1 else None
    new_units = units.anchor_span_fold1_prefix_units_profile() if fold_id == 1 else None
    original = norm.require_input_normalization_contract(
        parent, expected_field_names=old_names, expected_ctx_cat_names=fields['ctx_cat'],
        expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        unobserved_units_profile=old_units)
    additions = {}
    for surface in SURFACES:
        stat = norm.require_surface_normalization(new_statistics[surface], surface=surface,
                                                  field_names=span_names(surface))
        need(set(stat) == set(original['surfaces']['mtf_h1'])
             and stat['fit_row_count'] == original['surfaces'][surface]['fit_row_count']
             and stat['binary_mask'] == stat['categorical_mask'] == [0,0]
             and stat['categorical_domains'] == {}
             and all(x in ('raw_iqr','median_positive_abs_deviation') for x in stat['scale_source']),
             'NEW_STAT_SCOPE:' + surface)
        additions[surface] = stat
    surfaces = {surface:merge_surface(original['surfaces'][surface], additions.get(surface),
        field_names=full_names[surface],
        units_profile_sha=new_units['contract_sha256'] if fold_id == 1 else None)
        for surface in norm.EXPECTED_SURFACES}
    maps = {surface:{i:full_names[surface].index(n)
             for i,n in enumerate(original['surfaces'][surface]['field_names'])}
            for surface in ('signal','ctx_cont')}
    aliases = [{**a,'signal_index':maps['signal'][a['signal_index']],
                'ctx_cont_index':maps['ctx_cont'][a['ctx_cont_index']]}
               for a in original['temporal_aliases']]
    result = norm.build_input_normalization_contract(
        fit_start_utc=original['fit_start_utc'],fit_end_utc=original['fit_end_utc'],
        surfaces=surfaces,ctx_cat=original['ctx_cat'],lineage=original['lineage'],
        temporal_aliases=aliases,entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        unobserved_units_profile=new_units)
    preservation = old_field_preservation(original,result)
    result['anchor_span_extension'] = dict(
        schema_version=f'gx1_exact_fold{fold_id}_anchor_span_normalization_extension_v1',
        parent_artifact=copy(parent_artifact),
        physical_input_result=copy(physical_result),
        source_contract=copy(original),new_field_statistics=copy(additions),
        unobserved_units_profile=copy(new_units),preservation=preservation,
        physical_storage='unchanged_parent_arrays_plus_qualified_two_field_sidecars',
        lineage_scope='parent_physical_arrays_only_new_fields_bound_by_physical_input_result',
        added_statistics_fitted_by_caller_only=True,old_statistics_refitted=False)
    result.pop('contract_sha256')
    result['contract_sha256'] = norm._canonical_sha256(result)
    return result


def extend_ema_delta_normalization(parent, new_statistics, *, physical_result):
    """Add only the restored EMA delta's six FIT statistics to a sealed fold-1 parent."""
    from gx1.features.entry_specialist_feature_groups_v1 import compile_full_ta_entry_execution_profile
    need(isinstance(parent, Mapping) and 'anchor_span_extension' in parent
         and parent['anchor_span_extension'].get('schema_version') ==
             'gx1_exact_fold1_anchor_span_normalization_extension_v1'
         and 'ta_projection' not in parent, 'EMA_DELTA_PARENT')
    need(isinstance(new_statistics, Mapping) and set(new_statistics) == set(SURFACES),
         'EMA_DELTA_SIX_SURFACES')
    need(isinstance(physical_result, Mapping) and set(physical_result) == {'path','sha256'}
         and isinstance(physical_result['path'], str)
         and physical_result['path'].startswith('/home/andre2/GX1_RUNS/')
         and physical_result['path'].endswith('/RESULT.json')
         and isinstance(physical_result['sha256'], str)
         and len(physical_result['sha256']) == 64
         and all(c in '0123456789abcdef' for c in physical_result['sha256']),
         'EMA_DELTA_PHYSICAL_RESULT_REF')
    fields = compile_full_ta_entry_execution_profile()['source_field_names']
    full_names = {surface:list(fields['mtf' if surface.startswith('mtf_') else surface])
                  for surface in norm.EXPECTED_SURFACES}
    old_names = {surface:[name for name in names if name not in ema_delta_names(surface)]
                 for surface,names in full_names.items()}
    old_units = units.anchor_span_fold1_prefix_units_profile()
    new_units = units.ema_delta_fold1_prefix_units_profile()
    original = norm.require_input_normalization_contract(
        parent,expected_field_names=old_names,expected_ctx_cat_names=fields['ctx_cat'],
        expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        unobserved_units_profile=old_units)
    additions = {}
    for surface in SURFACES:
        stat = norm.require_surface_normalization(new_statistics[surface],surface=surface,
                                                  field_names=ema_delta_names(surface))
        need(set(stat) == set(original['surfaces']['mtf_h1'])
             and stat['fit_row_count'] == original['surfaces'][surface]['fit_row_count']
             and stat['binary_mask'] == stat['categorical_mask'] == [0]
             and stat['categorical_domains'] == {}
             and all(source in ('raw_iqr','median_positive_abs_deviation')
                     for source in stat['scale_source']),
             'EMA_DELTA_NEW_STAT_SCOPE:' + surface)
        additions[surface] = stat
    surfaces = {surface:merge_surface(original['surfaces'][surface],additions.get(surface),
        field_names=full_names[surface],units_profile_sha=new_units['contract_sha256'])
        for surface in norm.EXPECTED_SURFACES}
    maps = {surface:{index:full_names[surface].index(name)
             for index,name in enumerate(original['surfaces'][surface]['field_names'])}
            for surface in ('signal','ctx_cont')}
    aliases = [{**alias,'signal_index':maps['signal'][alias['signal_index']],
                'ctx_cont_index':maps['ctx_cont'][alias['ctx_cont_index']]}
               for alias in original['temporal_aliases']]
    result = norm.build_input_normalization_contract(
        fit_start_utc=original['fit_start_utc'],fit_end_utc=original['fit_end_utc'],
        surfaces=surfaces,ctx_cat=original['ctx_cat'],lineage=original['lineage'],
        temporal_aliases=aliases,entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        unobserved_units_profile=new_units)
    preservation = old_field_preservation(original,result)
    result['anchor_span_extension'] = dict(
        schema_version=EMA_DELTA_SCHEMA,source_contract=copy(original),
        physical_input_result=copy(physical_result),new_field_statistics=copy(additions),
        unobserved_units_profile=copy(new_units),preservation=preservation,
        physical_storage='unchanged_parent_arrays_plus_qualified_one_field_sidecars',
        added_statistics_fitted_by_caller_only=True,old_statistics_refitted=False)
    result.pop('contract_sha256')
    result['contract_sha256'] = norm._canonical_sha256(result)
    return result


def require_anchor_span_normalization(value, *, expected_field_names, expected_ctx_cat_names,
                                      expected_entry_training_phase, unobserved_units_profile):
    """Reconstruct the entire extension; resealing cannot alter an old statistic."""
    need(isinstance(value, Mapping), 'CONTRACT')
    provenance = value.get('anchor_span_extension')
    need(isinstance(provenance, Mapping), 'PROVENANCE_AND_UNITS_REQUIRED')
    successor = provenance.get('schema_version') == EMA_DELTA_SCHEMA
    expected = (extend_ema_delta_normalization(
        provenance.get('source_contract'),provenance.get('new_field_statistics'),
        physical_result=provenance.get('physical_input_result')) if successor else
        extend_anchor_span_normalization(
        provenance.get('source_contract'),provenance.get('new_field_statistics')))
    if expected['anchor_span_extension']['unobserved_units_profile'] is None:
        need(unobserved_units_profile is None, 'FOLD2_UNOBSERVED_UNITS_FORBIDDEN')
        profile = None
    else:
        need(isinstance(unobserved_units_profile, Mapping), 'PROVENANCE_AND_UNITS_REQUIRED')
        profile = (units.require_ema_delta_fold1_prefix_units_profile(unobserved_units_profile)
                   if successor else
                   units.require_anchor_span_fold1_prefix_units_profile(unobserved_units_profile))
    need(profile == expected['anchor_span_extension']['unobserved_units_profile']
         and norm._canonical_sha256(value) == norm._canonical_sha256(expected), 'EXACT_EXTENSION')
    fields = {surface:stat['field_names'] for surface,stat in expected['surfaces'].items()}
    need({k:list(v) for k,v in expected_field_names.items()} == fields
         and list(expected_ctx_cat_names) == expected['ctx_cat']['field_names']
         and expected_entry_training_phase in (None,ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE),
         'CONSUMER_BINDING')
    return expected
