"""Independent, admitted comparison of eight immutable first-prefix inputs.

No weights, native forwards, fitting, new labels or calibration fitting. The
initial same-256 bit gate precedes even hashing the two later prediction files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
SCHEMA = 'gx1_direction_saved_bps_comparison_plan_v1'
STAGE = 'independent_direction_bps_comparison_20261027'
SYNTHETIC_STAGE = 'independent_direction_bps_comparison_synthetic_20261027'
TEST_FILE = 'tests/test_compare_entry_direction_saved_bps_v1.py'
DESIGN_REF = {'path': str(ROOT/'DESIGN.json'), 'sha256': 'a6638b593ebe41eebf1b45d60007cd2a52b04ef0a28cdb174782acb4b77b72b6'}
PROTOCOL_REF = {'path': '/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2/DIRECTION_BPS_PREFIX_FIT_FOLD1_20261027_001/COMPARISON_PROTOCOL.json', 'sha256': '25458f5185c55070333729eededff4d33a11dd5b1ee334e74a4358c15a928888'}
INDEPENDENT_PROTOCOL_REF = {'path': str(ROOT/'SAVED_BPS_PROTOCOL_001.json'), 'sha256': '14a7a98be7d5f75fa7d1c0a2534413cc8159cae01d7773144182abcea7fd23eb'}
EXPLORATORY_PROTOCOL_REF = {'path': str(ROOT/'EXPLORATORY_HEAD_RANK_PROTOCOL_001.json'), 'sha256': 'ca4dfbbd0fac33a47fb8b65657ef7ebaaf08126b424e77da5402d91daf1d2b7c'}
PREVIOUS_COMPARISON_REF = {'path': str(ROOT/'SAVED_BPS_COMPARISON_001/OUTPUT/RESULT.json'), 'sha256': '5e8d6a2de22ff9d6d53c814d139ae62b58cdb9925f1bf1b14f40e8ac905f8c21'}
EXPLORATORY_STAGE = 'independent_direction_all_heads_20261027'
EXPLORATORY_SYNTHETIC_STAGE = 'independent_direction_all_heads_synthetic_20261027'
HEADS = ('baseline_logit_up', 'candidate_logit_up', 'candidate_mean_return_bps')
EXPLORATORY_CONTRASTS = ('candidate_mean_return_bps_minus_baseline_logit_up',
    'candidate_mean_return_bps_minus_candidate_logit_up')
EXPLORATORY_ALPHA = 0.05 / 12
HORIZONS = (1, 5, 15)
MINUTE, DAY = 60_000_000_000, 86_400_000_000_000
PREFIX, LATER, PARENTS = 47602, 17288, 118249
TRAIN_CUTOFF, LATER_CUTOFF = 1735689600000000000, 1743465600000000000
FOLD = dict(fold_id=1, prefix_rows=PREFIX, oof_start_row=PREFIX,
    oof_end_row=PREFIX+LATER, train_cutoff_ns=TRAIN_CUTOFF, oof_cutoff_ns=LATER_CUTOFF)
ORDER_SHA = '5a92eba23f101f139b758098e04abdc432e8e3bba032a94a3b0a656a21f0f43f'
REPLICATES, ALPHA = 10000, 0.05 / 6
BANDS = ('observed_exact_zero', '(0,1]', '(1,5]', '(5,10]', '(10,25]', '(25,50]', '(50,+infinity)')
ROLES = ('baseline_initial', 'baseline_final', 'baseline_oof',
    'candidate_initial', 'candidate_final', 'candidate_oof')
INPUT_KEYS = set(ROLES) | {'price_targets', 'bps_scale'}
METADATA_KEYS = set(ROLES) | {arm+'_'+kind for arm in ('baseline', 'candidate')
    for kind in ('recipe', 'input_manifest', 'result', 'qualification')}
SOURCE_PATHS = ('gx1/scripts/compare_entry_direction_saved_bps_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/contracts/entry_direction_probability_reference_v1.py',
    'gx1/contracts/entry_direction_block_rank_v1.py',
    'gx1/contracts/entry_native_direction_bps_v1.py',
    'gx1/contracts/entry_native_direction_context_report_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py')
SHARED = ('parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns', 'observed', 'sign')


def require(ok, reason):
    if not ok:
        raise RuntimeError('SAVED_BPS_COMPARISON_' + reason)


def digest(value, np):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def require_plan(plan):
    fields = {'schema_version', 'mode', 'design', 'protocol', 'independent_protocol',
        'metadata', 'inputs', 'sources', 'out_dir', 'wall_seconds'}
    mode = plan.get('mode')
    optional = {'exploratory_protocol', 'previous_comparison'}
    require(mode in ('actual', 'synthetic') and set(plan) in (fields, fields | optional)
        and plan['schema_version'] == SCHEMA + ('_synthetic' if mode == 'synthetic' else ''), 'PLAN_SCHEMA')
    require(plan['design'] == DESIGN_REF and plan['protocol'] == PROTOCOL_REF
        and plan['independent_protocol'] == INDEPENDENT_PROTOCOL_REF, 'PROTOCOL_BINDINGS')
    if 'exploratory_protocol' in plan:
        require(plan['exploratory_protocol'] == EXPLORATORY_PROTOCOL_REF
            and plan['previous_comparison'] == PREVIOUS_COMPARISON_REF, 'EXPLORATORY_BINDINGS')
    require(set(plan['inputs']) == (INPUT_KEYS if mode == 'actual' else set())
        and set(plan['metadata']) == (METADATA_KEYS if mode == 'actual' else set()), 'INPUT_METADATA_KEYS')
    require(plan['wall_seconds'] == (600 if mode == 'actual' else 180), 'WALL')
    out = Path(plan['out_dir'])
    require(out.is_absolute() and out == out.resolve(strict=False) and out.parent == ROOT
        and ROOT.resolve(strict=True) == ROOT and not out.exists() and not out.is_symlink(), 'OUTPUT_ROOT')
    required_sources = set(SOURCE_PATHS) | ({TEST_FILE} if mode == 'synthetic' else set())
    require(isinstance(plan['sources'], dict) and required_sources <= set(plan['sources']), 'SOURCE_SET')
    for relative, ref in plan['sources'].items():
        require(relative.startswith(('gx1/', 'tests/', 'scripts/')) and '..' not in Path(relative).parts
            and set(ref) == {'path', 'sha256'} and ref['path'] == str(REPO/relative), 'SOURCE_PATH')
    return out


def selected_stage(plan):
    if 'exploratory_protocol' in plan:
        return EXPLORATORY_SYNTHETIC_STAGE if plan['mode'] == 'synthetic' else EXPLORATORY_STAGE
    return SYNTHETIC_STAGE if plan['mode'] == 'synthetic' else STAGE


def require_exploratory_protocol(protocol):
    require(protocol['schema_version'] == 'gx1_independent_exploratory_all_head_rank_protocol_v1'
        and protocol['design'] == DESIGN_REF and protocol['previous_comparison'] == PREVIOUS_COMPARISON_REF
        and protocol['heads'] == list(HEADS) and protocol['horizons_minutes'] == list(HORIZONS)
        and protocol['new_auc_contrasts'] == list(EXPLORATORY_CONTRASTS)
        and protocol['advance_authorized'] is False and protocol['execution_authorized'] is False
        and protocol['model_or_head_selection_authorized'] is False and protocol['test_data_used'] is False,
        'EXPLORATORY_PROTOCOL')
    require(protocol['execution_limits'] == {'fits': 0, 'max_invocations': 1,
        'memory_bytes': 4294967296, 'model_forwards': 0, 'new_input_fields': 0,
        'wall_seconds': 600, 'weight_reads': 0}, 'EXPLORATORY_LIMITS')
    uncertainty = protocol['uncertainty']
    require(uncertainty['day_seed'] == 20261027 and uncertainty['week_seed'] == 20261028
        and uncertainty['groupings'] == ['UTC_DAY', 'UTC_WEEK_MONDAY']
        and uncertainty['interval_count'] == 12 and uncertainty['per_interval_alpha'] == EXPLORATORY_ALPHA
        and uncertainty['min_blocks'] == 8 and uncertainty['replicates'] == REPLICATES
        and uncertainty['shared_draws_across_all_heads_horizons'] is True, 'EXPLORATORY_UNCERTAINTY')


def require_legacy_replay(current, previous):
    # JSON-canonical comparison also preserves signed zero in old point values.
    require(previous['initial_same256_logit_bit_parity'] is True
        and previous['paired_primary']['combined_advance_rule_supported'] is False,
        'ORIGINAL_NEGATIVE_COMPARISON')
    require(json.dumps(current, sort_keys=True, separators=(',', ':'), allow_nan=False)
        == json.dumps(previous, sort_keys=True, separators=(',', ':'), allow_nan=False),
        'EXACT_LEGACY_RESULT_REPLAY')


def require_protocol(protocol, independent):
    require(protocol['schema_version'] == 'gx1_signed_bps_first_prefix_comparison_protocol_v1'
        and protocol['fold'] == FOLD and protocol['horizons_minutes'] == list(HORIZONS)
        and protocol['primary_horizon_minutes'] == 5 and protocol['batch_size'] == 16
        and protocol['seed'] == 20260911 and protocol['train_passes'] == 1
        and protocol['uncertainty']['confirmatory_interval_count'] == 6
        and protocol['uncertainty']['bootstrap_replicates'] == REPLICATES
        and protocol['uncertainty']['per_interval_alpha'] == ALPHA
        and protocol['genuine_execution_authorized_by_protocol'] is False, 'ORIGINAL_PROTOCOL')
    require(independent['schema_version'] == 'gx1_independent_direction_saved_bps_protocol_v1'
        and independent['design'] == DESIGN_REF
        and independent['existing_frozen_statistical_protocol'] == PROTOCOL_REF
        and independent['uncertainty'] == {'corrected_interval_count': 6, 'day_seed': 20261027,
            'min_blocks': 8, 'per_interval_alpha': ALPHA, 'replicates': REPLICATES,
            'shared_weights_across_contrasts': True,
            'undefined_replicates': 'Count; any undefined primary AUC replicate blocks confirmatoryqualification. No replacementdraws/reseeding.',
            'week_seed': 20261028}
        and independent['fits'] == independent['model_forwards'] == independent['weight_reads'] == 0
        and independent['execution_authorized'] is False, 'INDEPENDENT_PROTOCOL')


def require_metadata(meta, refs, inputs, protocol):
    require(meta['candidate_recipe']['fold'] == meta['baseline_recipe']['fold'] == FOLD,
        'RECIPE_FOLD')
    require(protocol['recipe'] == refs['candidate_recipe']
        and protocol['baseline_recipe'] == refs['baseline_recipe']
        and protocol['baseline_qualification'] == refs['baseline_qualification'], 'PROTOCOL_LINEAGE')
    recipes = [meta[arm+'_recipe'] for arm in ('baseline', 'candidate')]
    for arm, recipe in zip(('baseline', 'candidate'), recipes):
        require(recipe['input_manifest'] == refs[arm+'_input_manifest'], 'RECIPE_INPUT_MANIFEST')
        result = meta[arm+'_result']
        require(result['recipe'] == refs[arm+'_recipe'] and result['fold'] == FOLD
            and result['phase_state'] == 'COMPLETE' and result['parent_cursor'] == PREFIX
            and result['oof_cursor'] == LATER and result['optimizer_updates'] == 2976
            and result['parent_order_sha256'] == ORDER_SHA and result['source_unchanged'] is True
            and result['qualification_weights_reused'] is False and result['test_data_used'] is False,
            'COMPLETE_ORIGINAL_RESULT:' + arm)
        for phase, key in (('initial', 'initial_measurement'), ('final', 'final_train_measurement'), ('oof', 'oof_measurement')):
            role = arm+'_'+phase
            record = meta[role]
            require(result[key] == refs[role] and record['predictions'] == inputs[role]
                and record['fold'] == FOLD and record['test_data_used'] is False, 'PREDICTION_LINEAGE:' + role)
    baseline, candidate = (recipe['numerical_recipe'] for recipe in recipes)
    allowed = {'contract_sha256', 'research_objective_sha256', 'signed_bps_supervision'}
    require(candidate.get('signed_bps_supervision') is True
        and baseline.get('signed_bps_supervision', False) is False
        and {k:v for k,v in baseline.items() if k not in allowed}
            == {k:v for k,v in candidate.items() if k not in allowed}, 'MATCHED_NUMERICAL_RECIPE')
    base_manifest, candidate_manifest = (meta[arm+'_input_manifest'] for arm in ('baseline', 'candidate'))
    require(base_manifest['targets'] == candidate_manifest['targets'] == inputs['price_targets']
        and {k:v for k,v in base_manifest.items() if k != 'view_source_bindings'}
            == {k:v for k,v in candidate_manifest.items() if k != 'view_source_bindings'}
        and {k:v for k,v in base_manifest['view_source_bindings'].items() if k != 'direction_component_source'}
            == {k:v for k,v in candidate_manifest['view_source_bindings'].items() if k != 'direction_component_source'},
        'MATCHED_INPUT_MANIFESTS')
    require(meta['baseline_initial']['own_prefix_prior'] == meta['candidate_initial']['own_prefix_prior'],
        'MATCHED_PREFIX_PRIOR')
    require(meta['baseline_qualification']['technical_fit_resume_scoring_qualified'] is True
        and meta['baseline_qualification']['test_data_used'] is False, 'BASELINE_QUALIFICATION')
    proof = meta['candidate_qualification']
    require(proof['schema_version'] == 'gx1_bps_prefix_fit_root_review_v1'
        and proof['status'] == 'COMPLETE_CONSUMED_PAIRED_COMPARISON_PENDING'
        and proof['recipe'] == refs['candidate_recipe'] and proof['original_worker_result'] == refs['candidate_result']
        and proof['own_prefix_bps_scale'] == inputs['bps_scale'] and proof['fold'] == FOLD
        and proof['test_data_used'] is False, 'CANDIDATE_QUALIFICATION')
    require(meta['baseline_initial']['feature_view_identity']['prefix_normalization']
        == meta['candidate_initial']['feature_view_identity']['prefix_normalization'], 'MATCHED_PREFIX_NORMALIZATION')


def read_predictions(path, *, signed_bps, rows, np, context_spec):
    names = set(SHARED) | {'logit_up'} | {'context__'+name for name, labels in context_spec}
    if signed_bps:
        names |= {'mean_return_bps', 'mid_return_bps'}
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(names) and set(archive.files) == names, 'PREDICTION_MEMBERS')
        result = {name: archive[name] for name in archive.files}
    validate_prediction(result, signed_bps=signed_bps, rows=rows, np=np, context_spec=context_spec)
    return result


def validate_prediction(result, *, signed_bps, rows, np, context_spec):
    names = set(SHARED) | {'logit_up'} | {'context__'+name for name, labels in context_spec}
    if signed_bps:
        names |= {'mean_return_bps', 'mid_return_bps'}
    require(set(result) == names, 'PREDICTION_MEMBERS')
    for name, value in result.items():
        shape = (rows,) if name in ('parent_rows', 'bar_start_ns', 'decision_ns') or name.startswith('context__') else (rows, 3)
        dtype = np.int8 if name == 'sign' or name.startswith('context__') else np.bool_ if name == 'observed' else np.float32 if name in ('logit_up', 'mean_return_bps') else np.float64 if name == 'mid_return_bps' else np.int64
        require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == dtype
            and np.isfinite(value).all(), 'PREDICTION_HEADER:' + name)
    require(rows > 0 and len(np.unique(result['parent_rows'])) == rows
        and np.all(result['parent_rows'] >= 0)
        and np.all(result['bar_start_ns'] >= 0)
        and np.all(result['bar_start_ns'] <= np.iinfo(np.int64).max - 20 * MINUTE)
        and np.all(result['bar_start_ns'] % (5*MINUTE) == 0)
        and np.array_equal(result['decision_ns'], result['bar_start_ns'] + 5*MINUTE)
        and np.array_equal(result['terminal_time_ns'], result['decision_ns'][:, None] + np.asarray(HORIZONS)*MINUTE)
        and np.isin(result['sign'], (-1,0,1)).all()
        and np.all(result['sign'][~result['observed']] == 0), 'PREDICTION_CLOCK_DOMAIN')
    for name, labels in context_spec:
        require(np.isin(result['context__'+name], [code for code, label in labels]).all(), 'CONTEXT_DOMAIN')
    if signed_bps:
        require(np.all(result['mid_return_bps'][~result['observed']] == 0)
            and np.array_equal(np.sign(result['mid_return_bps']).astype(np.int8), result['sign']), 'BPS_TARGET_DOMAIN')


def join_targets(prediction, target, *, np):
    parents = prediction['parent_rows']
    require(parents.dtype == np.int64 and parents.ndim == 1 and len(parents) > 0
        and np.all(parents >= 0)
        and np.array_equal(target['parent_rows'], np.arange(len(target['parent_rows']), dtype=np.int64))
        and np.all(parents < len(target['parent_rows'])), 'TARGET_PARENT_JOIN')
    for name in SHARED:
        require(prediction[name].dtype == target[name].dtype
            and prediction[name].tobytes() == target[name][parents].tobytes(), 'TARGET_EXACT_JOIN:' + name)
    if 'mid_return_bps' in prediction:
        require(prediction['mid_return_bps'].tobytes() == target['mid_return_bps'][parents].tobytes(), 'TARGET_BPS_EXACT_JOIN')


def require_pair(baseline, candidate, *, np, initial=False):
    names = set(SHARED) | {name for name in baseline if name.startswith('context__')}
    for name in names:
        require(baseline[name].dtype == candidate[name].dtype
            and baseline[name].tobytes() == candidate[name].tobytes(), 'PAIR_EXACT_JOIN:' + name)
    if initial:
        require(baseline['logit_up'].tobytes() == candidate['logit_up'].tobytes(), 'INITIAL_LOGIT_BIT_PARITY')
        require(np.all(candidate['mean_return_bps'] == 0), 'INITIAL_ZERO_BPS_HEAD')


def band_ids(values, np):
    require(np.isfinite(values).all(), 'BAND_FINITE')
    return np.where(values == 0, 0,
        1 + np.searchsorted(np.asarray([1.,5.,10.,25.,50.]), np.abs(values), side='left'))


def direction_score(sign, logits, mask, *, probability, np):
    labels, z = (sign[mask] == 1).astype(np.int8), logits[mask].astype(np.float64)
    if not len(labels):
        return {'rows': 0, 'logloss': None, 'brier': None, 'auc': None, 'reliability': []}, None
    score = probability.direction_probability_metrics(labels, z)
    rank = probability.direction_rank_diagnostics(labels, z, score_kind='logit_up')
    public = {key:value for key,value in score.items() if key not in ('logloss_by_row', 'brier_by_row')}
    public['auc'] = rank['auc']
    return public, {'labels': labels, 'logit': z,
        'logloss': score['logloss_by_row'], 'brier': score['brier_by_row']}


def point_report(baseline, candidate, target, scale, *, cutoff, probability, bps, np):
    require_pair(baseline, candidate, np=np)
    parents = candidate['parent_rows']; truth = target['mid_return_bps'][parents]
    known = candidate['observed'] & (candidate['terminal_time_ns'] < cutoff)
    eligible = known & (candidate['sign'] != 0)
    returns = bps.score_direction_bps(candidate['mean_return_bps'], mid_return_bps=truth,
        observed=candidate['observed'], terminal_time_ns=candidate['terminal_time_ns'], cutoff_ns=cutoff, scale=scale)
    rows = []
    for j, horizon in enumerate(HORIZONS):
        arms = {name: direction_score(candidate['sign'][:,j], value['logit_up'][:,j], eligible[:,j],
            probability=probability, np=np)[0] for name,value in (('baseline',baseline),('candidate',candidate))}
        bands = band_ids(truth[:,j], np); band_reports = []
        for band, label in enumerate(BANDS):
            selected = known[:,j] & (bands == band)
            directional = eligible[:,j] & (bands == band)
            error = candidate['mean_return_bps'][selected,j].astype(np.float64) - truth[selected,j]
            band_reports.append({'band': label, 'observed_cells': int(selected.sum()),
                'eligible_direction_cells': int(directional.sum()),
                'sse_bps2': float(np.sum(error**2)), 'absolute_error_sum_bps': float(np.abs(error).sum()),
                'mse_bps2': float(np.mean(error**2)) if len(error) else None,
                'mae_bps': float(np.mean(np.abs(error))) if len(error) else None,
                'prefix_scaled_mse': float(np.mean((error/scale['rms_bps'][j])**2)) if len(error) else None,
                'zero_reference_sse_bps2': float(np.sum(truth[selected,j]**2)),
                'own_prefix_mean_reference_sse_bps2': float(np.sum((truth[selected,j]-scale['mean_bps'][j])**2)),
                'direction': {name: direction_score(candidate['sign'][:,j], value['logit_up'][:,j], directional,
                    probability=probability, np=np)[0] for name,value in (('baseline',baseline),('candidate',candidate))}})
        require(sum(row['observed_cells'] for row in band_reports) == int(known[:,j].sum()), 'BAND_PARTITION')
        rows.append({'horizon_minutes': horizon, 'parents': len(parents),
            'observed_mature': int(known[:,j].sum()), 'observed_mature_zero': int((known[:,j] & (candidate['sign'][:,j] == 0)).sum()),
            'mature_missing': int((~candidate['observed'][:,j] & (candidate['terminal_time_ns'][:,j] < cutoff)).sum()),
            'immature': int((candidate['terminal_time_ns'][:,j] >= cutoff).sum()),
            'direction': arms, 'signed_bps': returns['per_horizon'][j], 'fixed_actual_return_bands': band_reports})
    return rows


def block_weights(blocks, *, seed, np, replicates=REPLICATES):
    require(type(blocks) is int and blocks > 0 and type(seed) is int
        and type(replicates) is int and replicates > 0, 'BLOCK_WEIGHTS_SCOPE')
    draws = np.random.default_rng(seed).integers(0, blocks, size=(replicates, blocks))
    weights = np.zeros((replicates, blocks), np.int64)
    np.add.at(weights, (np.repeat(np.arange(replicates), blocks), draws.reshape(-1)), 1)
    return weights


def percentile_interval(values, *, np, alpha=ALPHA):
    finite = np.isfinite(values)
    count = int(finite.sum()); invalid = int((~finite).sum())
    bounds = np.quantile(values[finite], [alpha/2, 1-alpha/2], method='linear') if count else (None, None)
    return {'low': None if bounds[0] is None else float(bounds[0]),
        'high': None if bounds[1] is None else float(bounds[1]),
        'per_interval_alpha': alpha, 'valid_replicates': count, 'undefined_replicates': invalid}


def paired_intervals(baseline, candidate, target, scale, *, probability, rank, np,
        replicates=REPLICATES, min_blocks=8):
    """One draw matrix per grouping; never average per-block AUC or block means."""
    require_pair(baseline, candidate, np=np)
    j = HORIZONS.index(5); parents = candidate['parent_rows']
    truth = target['mid_return_bps'][parents,j]
    known = candidate['observed'][:,j] & (candidate['terminal_time_ns'][:,j] < LATER_CUTOFF)
    eligible = known & (candidate['sign'][:,j] != 0)
    require(known.any() and eligible.any(), 'PRIMARY_OBSERVATIONS')
    labels = (candidate['sign'][eligible,j] == 1).astype(np.int8)
    require(np.unique(labels).size == 2, 'PRIMARY_BOTH_CLASSES')
    predictions = candidate['mean_return_bps'][:,j].astype(np.float64)
    errors = (predictions[known]-truth[known])**2
    deltas = {'mse_candidate_minus_prefix_mean': errors-(truth[known]-scale['mean_bps'][j])**2,
        'mse_candidate_minus_zero': errors-truth[known]**2}
    proper = {arm: probability.direction_probability_metrics(labels, value['logit_up'][eligible,j].astype(np.float64))
        for arm,value in (('baseline',baseline),('candidate',candidate))}
    days = candidate['decision_ns']//DAY; reports = []
    all_supported = True
    for scheme, block_labels, seed in (('UTC_DAY',days,20261027), ('UTC_WEEK_MONDAY',(days+3)//7,20261028)):
        unique, inverse = np.unique(block_labels, return_inverse=True); k = len(unique)
        require(k >= min_blocks, 'MINIMUM_BLOCK_COUNT')
        weights = block_weights(k, seed=seed, np=np, replicates=replicates)
        wf = weights.astype(np.float64)
        counts = np.bincount(inverse[known], minlength=k)
        denominator = wf @ counts
        contrasts = []
        for name, delta in deltas.items():
            sums = np.bincount(inverse[known], weights=delta, minlength=k)
            samples = np.full(replicates, np.nan, np.float64)
            np.divide(wf @ sums, denominator, out=samples, where=denominator > 0)
            interval = percentile_interval(samples, np=np)
            supported = interval['undefined_replicates'] == 0 and interval['high'] is not None and interval['high'] < 0
            all_supported &= supported
            contrasts.append({'contrast': name, 'point_difference': float(delta.mean()),
                'interval': interval, 'supported_improvement': bool(supported)})
        auc = {}
        for arm,value in (('baseline',baseline),('candidate',candidate)):
            # Keep the full parent block vocabulary, even if a block has no
            # eligible direction cells; rank owner requires contiguous IDs.
            used = np.unique(inverse[eligible]); lookup = np.full(k, -1, np.int64)
            lookup[used] = np.arange(len(used), dtype=np.int64)
            summary = rank.block_rank_pairs(labels, value['logit_up'][eligible,j].astype(np.float64), lookup[inverse[eligible]])
            selected_weights = wf[:,used]
            has_cells = selected_weights.sum(axis=1) > 0
            numerator, pairs = np.zeros(replicates), np.zeros(replicates)
            if has_cells.any():
                numerator[has_cells], pairs[has_cells] = rank.weighted_rank_pairs(summary, selected_weights[has_cells])
            auc[arm] = np.full(replicates, np.nan, np.float64)
            np.divide(numerator, 2*pairs, out=auc[arm], where=pairs > 0)
        samples = auc['candidate']-auc['baseline']; interval = percentile_interval(samples, np=np)
        point = {arm: probability.direction_rank_diagnostics(labels, value['logit_up'][eligible,j].astype(np.float64),
            score_kind='logit_up')['auc'] for arm,value in (('baseline',baseline),('candidate',candidate))}
        supported = interval['undefined_replicates'] == 0 and interval['low'] is not None and interval['low'] > 0
        all_supported &= supported
        contrasts.append({'contrast': 'auc_candidate_minus_baseline_logit_up',
            'point_difference': point['candidate']-point['baseline'], 'interval': interval,
            'supported_improvement': bool(supported)})
        descriptive = {}
        direction_counts = np.bincount(inverse[eligible], minlength=k)
        for metric in ('logloss', 'brier'):
            delta = proper['candidate'][metric+'_by_row']-proper['baseline'][metric+'_by_row']
            sums = np.bincount(inverse[eligible], weights=delta, minlength=k)
            denominator = wf @ direction_counts
            samples = np.full(replicates, np.nan, np.float64)
            np.divide(wf @ sums, denominator, out=samples, where=denominator > 0)
            descriptive[metric] = {'point_difference': float(delta.mean()),
                'mean_resampled_difference': float(np.nanmean(samples)) if np.isfinite(samples).any() else None,
                'undefined_replicates': int((~np.isfinite(samples)).sum()),
                'confirmatory_interval': False}
        reports.append({'scheme': scheme, 'blocks': k, 'seed': seed,
            'replicates': replicates, 'weights_sha256': digest(weights, np), 'contrasts': contrasts,
            'proper_score_paired_descriptive': descriptive})
    return {'primary_horizon_minutes': 5, 'family_size': 6, 'groupings': reports,
        'combined_advance_rule_supported': bool(all_supported),
        'undefined_primary_auc_blocks_qualification': any(
            item['contrasts'][2]['interval']['undefined_replicates'] for item in reports),
        'interpretation': 'One reused-development fold only; no generalization, trading or economic qualification.'}


def head_values(baseline, candidate):
    return (baseline['logit_up'], candidate['logit_up'], candidate['mean_return_bps'])


def raw_rank_point(labels, scores, *, rank, np):
    require(labels.dtype == np.int8 and labels.ndim == 1 and scores.dtype == np.float64
        and scores.shape == labels.shape and np.isin(labels, (0, 1)).all()
        and np.isfinite(scores).all(), 'RAW_HEAD_POINT_INPUT')
    down, up = int((labels == 0).sum()), int((labels == 1).sum())
    twice_wins = 0
    if len(labels):
        summary = rank.block_rank_pairs(labels, scores, np.zeros(len(labels), np.int64))
        twice_wins = int(summary['twice_wins'][0, 0])
    confusion = [[int(((scores < 0) & (labels == column)).sum()) for column in (0, 1)],
        [int(((scores == 0) & (labels == column)).sum()) for column in (0, 1)],
        [int(((scores > 0) & (labels == column)).sum()) for column in (0, 1)]]
    correct = confusion[0][0] + confusion[2][1]
    return {'rows': len(labels), 'down_rows': down, 'up_rows': up,
        'auc': twice_wins/(2*down*up) if down and up else None,
        'twice_rank_wins': twice_wins, 'up_down_pairs': down*up,
        'raw_score_sha256': digest(scores, np),
        'zero_threshold': {'predicted_rows': ['negative', 'zero', 'positive'],
            'truth_columns': ['down', 'up'], 'confusion_3x2': confusion,
            'down_recall': confusion[0][0]/down if down else None,
            'up_recall': confusion[2][1]/up if up else None,
            'correct_rows': correct, 'all_rows_accuracy': correct/len(labels) if len(labels) else None,
            'zero_forecasts': sum(confusion[1]), 'trading_actions_defined': False}}


def raw_rank_replicates(labels, scores, parent_block_ids, weights, *, rank, np):
    # Empty/one-class resamples stay undefined. Never redraw or replace them.
    samples = np.full(len(weights), np.nan, np.float64)
    if not len(labels):
        return samples
    used = np.unique(parent_block_ids)
    lookup = np.full(weights.shape[1], -1, np.int64)
    lookup[used] = np.arange(len(used), dtype=np.int64)
    summary = rank.block_rank_pairs(labels, scores, lookup[parent_block_ids])
    selected = weights[:, used]
    has_cells = selected.sum(axis=1) > 0
    numerator, pairs = np.zeros(len(weights)), np.zeros(len(weights))
    if has_cells.any():
        numerator[has_cells], pairs[has_cells] = rank.weighted_rank_pairs(summary, selected[has_cells])
    np.divide(numerator, 2*pairs, out=samples, where=pairs > 0)
    return samples


def all_head_rank_report(baseline, candidate, *, rank, np,
        replicates=REPLICATES, min_blocks=8):
    require_pair(baseline, candidate, np=np)
    require(type(replicates) is int and replicates > 0 and type(min_blocks) is int and min_blocks >= 1,
        'EXPLORATORY_RESAMPLE_SCOPE')
    eligible = (candidate['observed'] & (candidate['sign'] != 0)
        & (candidate['terminal_time_ns'] < LATER_CUTOFF))
    arrays = head_values(baseline, candidate)
    points, panels = [], []
    for j, horizon in enumerate(HORIZONS):
        keep = eligible[:, j]
        labels = (candidate['sign'][keep, j] == 1).astype(np.int8)
        scores = [np.ascontiguousarray(value[keep, j], dtype=np.float64) for value in arrays]
        head_points = [raw_rank_point(labels, value, rank=rank, np=np) for value in scores]
        for head, point in zip(HEADS, head_points):
            points.append({'head': head, 'horizon_minutes': horizon, 'direction_mask_sha256': digest(keep, np), **point})
        panels.append((keep, labels, scores, head_points))
    days = candidate['decision_ns']//DAY
    groupings = []
    for scheme, block_labels, seed in (('UTC_DAY', days, 20261027), ('UTC_WEEK_MONDAY', (days+3)//7, 20261028)):
        unique, inverse = np.unique(block_labels, return_inverse=True); count = len(unique)
        require(count >= min_blocks, 'EXPLORATORY_MINIMUM_BLOCK_COUNT')
        # Exactly one full-parent draw matrix, shared by every head and horizon.
        weights = block_weights(count, seed=seed, np=np, replicates=replicates)
        contrasts = []
        for horizon, (keep, labels, scores, head_points) in zip(HORIZONS, panels):
            samples = [raw_rank_replicates(labels, value, inverse[keep], weights, rank=rank, np=np) for value in scores]
            for name, reference in zip(EXPLORATORY_CONTRASTS, (0, 1)):
                interval = percentile_interval(samples[2]-samples[reference], np=np, alpha=EXPLORATORY_ALPHA)
                mean_auc, reference_auc = head_points[2]['auc'], head_points[reference]['auc']
                supported = (interval['undefined_replicates'] == 0
                    and interval['low'] is not None and interval['low'] > 0)
                contrasts.append({'contrast': name, 'horizon_minutes': horizon,
                    'point_difference': None if mean_auc is None or reference_auc is None else mean_auc-reference_auc,
                    'interval': interval, 'exploratory_positive_difference': bool(supported)})
        groupings.append({'scheme': scheme, 'blocks': count, 'seed': seed,
            'replicates': replicates, 'weights_sha256': digest(weights, np), 'contrasts': contrasts})
    undefined = any(item['interval']['undefined_replicates']
        for grouping in groupings for item in grouping['contrasts'])
    if undefined:
        for grouping in groupings:
            for item in grouping['contrasts']:
                item['exploratory_positive_difference'] = False
    return {'evidence_class': 'Exploratory after original confirmatory failure.',
        'heads': list(HEADS), 'horizons_minutes': list(HORIZONS), 'rank_points': points,
        'family_size': 12, 'groupings': groupings,
        'undefined_draws_prohibit_supported_interpretation': bool(undefined),
        'score_semantics': 'Raw finite head values and exact ties; bps is never a probability or logit.',
        'proper_scores_for_bps_head': False, 'advance_authorized': False,
        'model_or_head_selection_authorized': False, 'economic_edge_qualified': False}


def add_exploratory_report(current, previous, baseline, candidate, *, rank, np,
        replicates=REPLICATES, min_blocks=8):
    require_legacy_replay(current, previous)
    extra = all_head_rank_report(baseline, candidate, rank=rank, np=np,
        replicates=replicates, min_blocks=min_blocks)
    require([(row['scheme'], row['seed'], row['weights_sha256']) for row in extra['groupings']]
        == [(row['scheme'], row['seed'], row['weights_sha256']) for row in current['paired_primary']['groupings']],
        'EXPLORATORY_SHARED_LEGACY_DRAWS')
    # Return a new namespace only after every old result field replayed exactly.
    return {**current, 'exploratory_protocol': EXPLORATORY_PROTOCOL_REF,
        'previous_comparison': PREVIOUS_COMPARISON_REF,
        'exploratory_all_head_rank': extra, 'legacy_result_replayed_exactly': True}


def make_guard(out, inputs, metadata, *, synthetic, source_paths, gate):
    allowed = set(inputs.values()) | set(metadata)
    test_sources = {str(REPO/TEST_FILE)} & set(source_paths)
    later = {inputs[key] for key in ('baseline_oof','candidate_oof') if key in inputs}
    prefix = ('git','-c','core.fsmonitor=false','-c','core.hooksPath=/dev/null','-C',str(REPO))
    tails = {('rev-parse','HEAD'),('branch','--show-current'),('status','--porcelain','--untracked-files=all')}
    def owned(value):
        p = Path(os.fsdecode(value))
        return p.is_absolute() and p == p.resolve(strict=False) and (p == out or out in p.parents) and not any(q.is_symlink() for q in (p,*p.parents))
    def guard(event, args):
        if event == 'subprocess.Popen':
            command = tuple(args[1]); require(command[:len(prefix)] == prefix and command[len(prefix):] in tails, 'PROCESS')
        require(event not in ('os.system','socket.connect','socket.bind','socket.sendto','socket.getaddrinfo',
            'os.exec','os.posix_spawn','os.fork','os.forkpty','pickle.find_class','os.symlink'), 'NETWORK_PROCESS_PICKLE')
        if event == 'import':
            require(args[0].split('.')[0] not in ('sklearn','joblib','tensorflow','jax','keras'), 'MODEL_IMPORT')
        if event == 'open' and isinstance(args[0], (str,bytes,os.PathLike)):
            name = os.fsdecode(args[0]); p = Path(name)
            writing = bool(args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            sealed = any(part.upper() == 'TEST' or part.upper().startswith(('TEST_','TEST.')) for part in p.parts)
            require(not sealed or (synthetic and (owned(name) or (name in test_sources and not writing))), 'TEST_PATH')
            require(name not in later or gate['later_allowed'], 'INITIAL_PARITY_BEFORE_LATER_OPEN')
            if writing: require(owned(name), 'OUTPUT_ONLY_WRITE')
            if '/GX1_DATA/' in name or '/GX1_RUNS/' in name: require(name in allowed or owned(name), 'DECLARED_DATA')
            if p.suffix.lower() in ('.npz','.npy','.parquet','.pt','.pth','.ckpt','.pkl','.pickle','.csv','.gz','.zst','.dbn'):
                require((synthetic and owned(name)) or (name in inputs.values() and not writing), 'PAYLOAD')
        if event in ('os.mkdir','os.remove','os.rmdir'): require(owned(args[0]), 'OUTPUT_ONLY_MUTATION')
        if event in ('os.rename','os.link'): require(owned(args[0]) and owned(args[1]), 'OUTPUT_ONLY_MUTATION')
    return guard


def work_observer(counters):
    def watch(frame, event, arg):
        if event == 'call':
            path, name = frame.f_code.co_filename.replace('\\','/'), frame.f_code.co_name
            forbidden = ('/gx1/models/' in path
                or (path.endswith('/torch/nn/modules/module.py') and name == '__init__')
                or ('/torch/optim/' in path and name in ('__init__','step','_init_group'))
                or ('/torch/cuda/' in path and name == '_lazy_init')
                or name in ('fit_direction_bps_scale','fit_direction_probability_reference',
                    'query_dynamic_m1_prefix','prepare_dynamic_m1_prefix_index','project_direction_price_endpoints'))
            if forbidden:
                counters['native_or_fit_calls'] += 1; require(False, 'NATIVE_FIT_OR_QUERY_CALL')
        if event == 'c_call':
            caller = frame.f_code.co_filename.replace('\\','/')
            direct_torch = ((getattr(arg,'__module__','') or '').startswith('torch')
                and caller.endswith(('/compare_entry_direction_saved_bps_v1.py', '/test_compare_entry_direction_saved_bps_v1.py')))
            if getattr(arg,'__name__','').startswith('_cuda') or direct_torch:
                counters['torch_object_or_cuda_calls'] += 1; require(False, 'TORCH_OBJECT_OR_CUDA_CALL')
    return watch


def run_synthetic(stage, *, publish):
    import pytest
    require(sys.dont_write_bytecode and os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD') == '1'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not os.environ.get('PYTEST_ADDOPTS'), 'SYNTHETIC_ENV')
    class Results:
        def __init__(self): self.nodes=[]; self.collection=[]; self.phases=[]; self.deselected=[]; self.errors=[]
        def pytest_collectreport(self, report): self.collection.append({'nodeid':report.nodeid,'outcome':report.outcome,'longrepr':None if report.passed else str(report.longrepr)})
        def pytest_collection_finish(self, session): self.nodes=[item.nodeid for item in session.items]
        def pytest_runtest_logreport(self, report): self.phases.append({'nodeid':report.nodeid,'when':report.when,'outcome':report.outcome,'longrepr':None if report.passed else str(report.longrepr)})
        def pytest_deselected(self, items): self.deselected.extend(item.nodeid for item in items)
        def pytest_internalerror(self, excrepr, excinfo): self.errors.append(str(excrepr))
    record=Results(); code=None; error=None; cache=stage/'python-cache'; cache.mkdir()
    previous=sys.pycache_prefix
    try:
        sys.pycache_prefix=str(cache)
        code=pytest.main(['-c','/dev/null','--rootdir',str(REPO),'--noconftest','-q','-s',
            '-p','no:cacheprovider','-p','no:logging','--assert=plain','--basetemp',str(stage/'pytest-temp'),TEST_FILE],plugins=[record])
    except Exception as exc: error=exc
    finally:
        sys.pycache_prefix=previous
        publish(stage/'SYNTHETIC_DIAGNOSTICS.json', {'collected_nodeids':record.nodes,
            'collection_reports':record.collection,'runtest_reports':record.phases,
            'deselected':record.deselected,'internal_errors':record.errors,
            'exit_code':None if code is None else int(code),'exception':None if error is None else str(error),
            'bytecode_prefix_restored':sys.pycache_prefix==previous})
    if error is not None: raise error
    require(record.nodes and len(set(record.nodes)) == len(record.nodes)
        and {node.split('::')[0] for node in record.nodes} == {TEST_FILE}
        and not record.deselected and not record.errors and int(code) == 0
        and all(item['outcome']=='passed' for item in record.collection+record.phases), 'SYNTHETIC_ALL_PASSED')
    require(all([item['when'] for item in record.phases if item['nodeid']==node] == ['setup','call','teardown']
        for node in record.nodes) and len(record.phases)==3*len(record.nodes), 'SYNTHETIC_ALL_PHASES')
    return {'status':'PASS_SYNTHETIC_ONLY','cases':len(record.nodes),'nodeids':record.nodes,
        'predictive_value_qualified':False}


def main(argv=None):
    sys.dont_write_bytecode=True
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','plan-sha256','policy-sha256','source-head'): parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv)
    import numpy as np
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.scripts.audit_entry_direction_candle_midpoints_v1 import regular, git_state, read_targets
    initial=git_state(); require(initial==(args.source_head,'work/gx1-current',''), 'CLEAN_HEAD_BRANCH')
    plan_ref={'path':str(regular(args.plan)),'sha256':args.plan_sha256}
    policy_ref={'path':str(REPO/'NEXT_RUN_POLICY.json'),'sha256':args.policy_sha256}
    with binding.fresh_file_verification_scope():
        plan=binding.read_bound_json(Path(plan_ref['path']),plan_ref['sha256']); out=require_plan(plan)
        synthetic=plan['mode']=='synthetic'
        binding.read_bound_json(Path(plan['design']['path']),plan['design']['sha256'])
        protocol=binding.read_bound_json(Path(plan['protocol']['path']),plan['protocol']['sha256'])
        independent=binding.read_bound_json(Path(plan['independent_protocol']['path']),plan['independent_protocol']['sha256'])
        require_protocol(protocol,independent)
        previous = None
        if 'exploratory_protocol' in plan:
            extra=binding.read_bound_json(Path(plan['exploratory_protocol']['path']),plan['exploratory_protocol']['sha256'])
            require_exploratory_protocol(extra)
            previous=binding.read_bound_json(Path(plan['previous_comparison']['path']),plan['previous_comparison']['sha256'])
        policy=binding.read_bound_json(Path(policy_ref['path']),policy_ref['sha256'])
        admitted=policy[selected_stage(plan)]
        require(admitted['status']=='ADMITTED' and admitted['plan']==plan_ref
            and admitted['execution_authorized'] is True and admitted['max_invocations']==1
            and admitted['claimed_invocations']==admitted['completed_invocations']==0
            and admitted['relaunch_allowed'] is False,'ONE_SHOT_POLICY')
        require(all(policy[key] is False for key in ('training_enabled','native_training_enabled',
            'full_epoch_training_allowed','full_val_allowed','native_launch_authorized')),'GLOBAL_GATES')
        for ref in plan['sources'].values(): regular(ref['path']); binding.require_binding(ref,label='source')
        control=binding.capture_verified_file_snapshot()
    proof,lock=cap.require_capped_cpu_audit_execution(),cap.require_capped_lock_ancestry()
    require(lock['scope']==proof['cgroup_path'][len('/sys/fs/cgroup'):],'CAPPED_LOCK')
    out.mkdir()
    def publish(path,value): publication._publish_bytes(path,binding.canonical_bytes(value))
    publish(out/'INVOCATION_CLAIM.json',{'plan':plan_ref,'policy':policy_ref,'source_head':args.source_head,'execution':proof,'lock':lock})
    meta_paths=[plan_ref['path'],policy_ref['path'],plan['design']['path'],plan['protocol']['path'],plan['independent_protocol']['path']]
    meta_paths += [ref['path'] for ref in plan['metadata'].values()]
    meta_paths += [plan[key]['path'] for key in ('exploratory_protocol', 'previous_comparison') if key in plan]
    gate={'later_allowed':False}
    sys.addaudithook(make_guard(out,{key:ref['path'] for key,ref in plan['inputs'].items()},meta_paths,
        synthetic=synthetic,source_paths=[ref['path'] for ref in plan['sources'].values()],gate=gate))
    counters={'native_or_fit_calls':0,'torch_object_or_cuda_calls':0}
    started=time.monotonic(); previous_profile=sys.getprofile(); previous_handler=signal.getsignal(signal.SIGALRM)
    def expired(*unused): require(False,'WALL_LIMIT')
    signal.signal(signal.SIGALRM,expired); signal.alarm(plan['wall_seconds'])
    stage=out/'.OUTPUT.staging'
    try:
        # Torch is imported only because the existing bps score module owns the
        # scale contract and NumPy evaluator; no Torch value is created by us.
        sys.setprofile(work_observer(counters))
        from gx1.contracts import entry_native_direction_bps_v1 as bps
        from gx1.contracts import entry_direction_probability_reference_v1 as probability
        from gx1.contracts import entry_direction_block_rank_v1 as rank
        from gx1.contracts.entry_native_direction_context_report_v1 import CONTEXT_SPEC
        stage.mkdir()
        if synthetic:
            report=run_synthetic(stage,publish=publish); inputs_snapshot=control
        else:
            with binding.fresh_file_verification_scope():
                meta={key:binding.read_bound_json(Path(ref['path']),ref['sha256']) for key,ref in plan['metadata'].items()}
                require_metadata(meta,plan['metadata'],plan['inputs'],protocol)
                def verify(key):
                    ref=plan['inputs'][key]; regular(ref['path']); binding.require_binding(ref,label=key); return ref['path']
                targets=read_targets(verify('price_targets'),np=np)
                scale=bps.require_direction_bps_scale(json.loads(Path(verify('bps_scale')).read_text()))
                require(scale['prefix_rows']==PREFIX and scale['cutoff_ns']==TRAIN_CUTOFF
                    and scale['target_source']==plan['inputs']['price_targets']
                    and scale['parent_rows_sha256']==digest(targets['parent_rows'][:PREFIX],np)
                    and scale['decision_ns_sha256']==digest(targets['decision_ns'][:PREFIX],np),'OWN_PREFIX_SCALE')
                def load(role,rows):
                    pred=read_predictions(verify(role),signed_bps=role.startswith('candidate'),rows=rows,np=np,context_spec=CONTEXT_SPEC)
                    join_targets(pred,targets,np=np); return pred
                baseline_initial,candidate_initial=load('baseline_initial',256),load('candidate_initial',256)
                require_pair(baseline_initial,candidate_initial,np=np,initial=True)
                require(np.all(baseline_initial['parent_rows']<PREFIX),'INITIAL_PREFIX_PARENTS')
                publish(stage/'INITIAL_PARITY.json',{'status':'PASS_BIT_IDENTICAL_INITIAL_SAME256',
                    'parent_rows_sha256':digest(baseline_initial['parent_rows'],np),
                    'logit_up_sha256':digest(baseline_initial['logit_up'],np)})
                gate['later_allowed']=True
                baseline_final,candidate_final=load('baseline_final',256),load('candidate_final',256)
                for initial_pred,final_pred in ((baseline_initial,baseline_final),(candidate_initial,candidate_final)):
                    require_pair(initial_pred,final_pred,np=np)
                baseline_oof,candidate_oof=load('baseline_oof',LATER),load('candidate_oof',LATER)
                require_pair(baseline_oof,candidate_oof,np=np)
                require(np.array_equal(baseline_oof['parent_rows'],np.arange(PREFIX,PREFIX+LATER,dtype=np.int64))
                    and np.all(baseline_oof['decision_ns']>=TRAIN_CUTOFF)
                    and np.all(baseline_oof['decision_ns']<LATER_CUTOFF),'EXACT_LATER_COHORT')
                datasets={'same256_initial':(baseline_initial,candidate_initial,TRAIN_CUTOFF),
                    'same256_final':(baseline_final,candidate_final,TRAIN_CUTOFF),
                    'later_development':(baseline_oof,candidate_oof,LATER_CUTOFF)}
                points={name:point_report(base,candidate,targets,scale,cutoff=cutoff,
                    probability=probability,bps=bps,np=np) for name,(base,candidate,cutoff) in datasets.items()}
                intervals=paired_intervals(baseline_oof,candidate_oof,targets,scale,probability=probability,rank=rank,np=np)
                hashes={name:{key:digest(value,np) for key,value in pred.items()} for name,pred in
                    (('baseline_initial',baseline_initial),('candidate_initial',candidate_initial),
                     ('baseline_final',baseline_final),('candidate_final',candidate_final),
                     ('baseline_oof',baseline_oof),('candidate_oof',candidate_oof))}
                inputs_snapshot=binding.capture_verified_file_snapshot()
            report={'schema_version':'gx1_independent_direction_saved_bps_comparison_v1',
                'initial_same256_logit_bit_parity':True,'fold':FOLD,'points':points,'paired_primary':intervals,
                'prediction_array_sha256':hashes,'inputs':plan['inputs'],'metadata':plan['metadata'],
                'protocol':plan['protocol'],'independent_protocol':plan['independent_protocol'],
                'scope':'One reused-development fold; raw logit direction, no recalibration or head selection.',
                'TRAIN_scope':'Same256 diagnostic rows only; not a whole-TRAIN fit measurement.',
                'economic_edge_qualified':False,'test_data_used':False}
            if previous is not None:
                report=add_exploratory_report(report,previous,baseline_oof,candidate_oof,rank=rank,np=np)
        require(not any(counters.values()),'NO_FORBIDDEN_WORK')
        inputs_snapshot.require_unchanged(); control.require_unchanged()
        require(git_state()==initial and cap.require_capped_lock_ancestry()==lock,'FINAL_SOURCE_LOCK_BOOKENDS')
        publish(stage/'RESULT.json',report)
        publish(stage/'TERMINAL.json',{'status':'COMPLETE','plan':plan_ref,'source_head':args.source_head,
            'counters':counters,'elapsed_seconds':time.monotonic()-started,'test_data_used':False})
        publish_bundle_directory_noreplace(stage,out/'OUTPUT')
        publish(out/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':plan_ref,'result_path':str(out/'OUTPUT/RESULT.json')})
        return 0
    except Exception as exc:
        publish(out/'FAILED_TERMINAL.json',{'status':'FAILED','exit_code':1,'plan':plan_ref,
            'error':type(exc).__name__+': '+str(exc),'counters':counters,
            'elapsed_seconds':time.monotonic()-started,'partial_output_is_invalid':True,'relaunch_allowed':False})
        raise
    finally:
        sys.setprofile(previous_profile); signal.alarm(0); signal.signal(signal.SIGALRM,previous_handler)


if __name__=='__main__':
    raise SystemExit(main())
