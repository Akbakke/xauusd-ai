"""Frozen V3 TRAIN review, using existing binary score math; no IO or execution route.

Callers authenticate raw-file references and decode their permitted arrays once.
Spec construction consumes metadata only. Review consumes fourteen truth arrays
and two existing CPU logit arrays; it never fits, loads weights or calls a model.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import PurePosixPath

import numpy as np

from gx1.audit.native_provider_hlc_learning_v1 import binary_panel
from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
from gx1.contracts import native_future_state_provider_hlc_v1 as hlc

ROWS = 354550
TF = ('M5', 'M15', 'H1', 'H4', 'D1')
YEARS = (2021, 2022, 2023, 2024, 2025)
SPEC_SCHEMA = 'gx1_native_all_tf_provider_hlc_review_spec_v3'
REVIEW_SCHEMA = 'gx1_native_all_tf_provider_hlc_fixed_TRAIN_review_v3'
REFERENCE_KEYS = {'dataset', 'prior', 'initial_result', 'initial_receipt',
                  'initial_acceptance', 'target_acceptance', 'support',
                  'input_binding', 'normalization', 'membership_manifest'}
RULES = {
    'pooled_primary': 'All5TF side NLL/Brier strictly below one frozen V3 prior in row AND equal_target weighting; 20 correlated comparisons. INITIAL is an identity witness of that same prior, not an independent reference.',
    'candidate_verdict': 'Combine pooled row/equal_target proper scores, all5 side rank guards and every fixed timeblock stability verdict; FAIL dominates UNKNOWN then PASS. This remains TRAIN comparison only, never learning/generalization/edge or execution admission.',
    'side_rank_guard': 'Both true classes, nonconstant logits and row AUC>0.5 per TF; one class/empty UNKNOWN. No argmax class quota.',
    'equal_target': 'Within each TF/known task support, average row proper scores inside each target_start_ns, then average target means; never score mean logits or collapse issue-specific labels.',
    'timeblocks': 'Every TF x UTC target-start year 2021..2025, each with row and equal_target weighting; NLL/Brier<=same pooled frozen prior for stability. Empty/one-class UNKNOWN, no omitted blocks. Equal-year macro is descriptive and requires all5 years nonempty.',
    'activity_and_singleton': 'Report activity proper scores and singleton three-class scores separately; side unknown remains unknown, active unknown side is never FLAT.',
    'calibration': 'Side mean probability minus true UP fraction and fixed bins [0,.1,...,1] under row/equal_target weighting. Descriptive only; no calibrator, fitted threshold, or absolute calibration PASS.',
    'dependence': 'Repeated/overlapping targets, TF, years and the prior/INITIAL references are dependent; no independent sample, confidence interval or hypothesis-test claim.',
    'loss_and_order': 'No numeric loss/gradient/task-weight change. Existing source-only order may be reused only with fresh authenticated V3 per-batch activity masks/denominators; never reuse V2 support counts.',
    'scope': 'TRAIN review only; no candidate FINAL exists or is selected by this spec. No fit/admission/extra epochs/HPO/VAL/TEST/edge or chosen correction.',
}
TRUTH_FIELDS = ('issue_ns', 'target_start_ns', 'validity_ns', 'target_present',
                'mature', 'nominal_complete', 'state_window_complete',
                'partial_range_active', 'activity_known', 'activity',
                'side_known', 'side_up', 'provider_hlc_observed', 'state_hlc_observed')


def _need(ok, reason):
    if not bool(ok):
        raise ValueError('ALL_TF_V3_REVIEW_' + reason)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _hash(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _references(refs):
    _need(type(refs) is dict and set(refs) == REFERENCE_KEYS, 'REFERENCE_KEYS')
    for q in refs.values():
        _need(type(q) is dict and set(q) == {'path', 'sha256'} and _hash(q['sha256'])
              and type(q['path']) is str and PurePosixPath(q['path']).is_absolute()
              and str(PurePosixPath(q['path'])) == q['path'] and '..' not in PurePosixPath(q['path']).parts,
              'REFERENCE_BINDING')


def _seal(value):
    return {**value, 'contract_sha256': _sha(value)}


def build_spec(*, metadata, model_contract, prior, initial_result, initial_receipt, references):
    """Metadata-only builder; caller authenticates each referenced JSON's bytes."""
    _references(references)
    target = hlc.require_all_tf_provider_hlc_state_contract(metadata['target_contract'])
    objective.require_native_future_state_train_prior(prior, target_contract=target)
    pop = prior['fit_population']; report = initial_result['report']; initial = report['initial']
    _need(initial_result['status'] == 'COMPLETE_NATIVE_ALL_TF_PROVIDER_HLC_V3_SEGMENTED_WHOLE_TRAIN_INITIAL'
          and initial_result['test_data_used'] is False and report['whole_TRAIN_initialbaseline_qualified'] is True
          and initial['rows'] == pop['issue_rows'] == ROWS and initial['initial_prior_exact'] is True
          and initial['truth_sha256'] == metadata['truth_sha256'] == pop['truth_sha256'], 'OWN_V3_INITIAL')
    _need(initial_receipt['initial_only'] is True and initial_receipt['optimizer_updates'] == 0
          and initial_receipt['model_contract_sha256'] == model_contract['contract_sha256']
          and initial_receipt['target_contract_sha256'] == target['contract_sha256']
          and initial_receipt['model_state_sha256'] == report['initial_state_sha256']
          and initial_receipt['rng_sha256'] == report['initial_rng_sha256']
          and model_contract['target_contract_sha256'] == target['contract_sha256']
          and model_contract['train_prior_sha256'] == prior['contract_sha256']
          and model_contract['objective_contract_sha256'] == prior['objective_contract_sha256']
          and model_contract['timeframes'] == list(TF), 'MODEL_INITIAL_PRIOR_IDENTITY')
    _need(all(metadata[k] == references[r] for k, r in
              [('dataset', 'dataset'), ('prior', 'prior'), ('normalization', 'normalization'),
               ('input_binding_ref', 'input_binding')]), 'METADATA_REFERENCES')
    scores = initial['scores']
    _need(scores['truth_sha256'] == pop['truth_sha256']
          and scores['contract_sha256'] == prior['objective_contract_sha256']
          and scores['target_observation_certificate_ref'] == initial['target_observation_certificate_ref'],
          'INITIAL_SCORE_IDENTITY')
    for task in ('activity', 'side'):
        for tf in TF:
            cell, score = prior['per_task'][task][tf], scores['per_tf'][tf][task]
            _need(cell['count'] == score['count'] and cell['unique_target_starts'] == score['unique_target_starts'],
                  'INITIAL_SUPPORT_IDENTITY')
    return validate_spec(_seal({
        'schema_version': SPEC_SCHEMA, 'references': copy.deepcopy(references),
        'target_contract': copy.deepcopy(target), 'model_contract': copy.deepcopy(model_contract),
        'prior_contract_sha256': prior['contract_sha256'],
        'truth_sha256': pop['truth_sha256'], 'fit_population': copy.deepcopy(pop),
        'populations': copy.deepcopy(prior['per_task']),
        'target_observation_certificate_ref': copy.deepcopy(initial['target_observation_certificate_ref']),
        'initial_state_sha256': report['initial_state_sha256'], 'initial_rng_sha256': report['initial_rng_sha256'],
        'timeframes': list(TF), 'UTC_target_years': list(YEARS), 'rules': copy.deepcopy(RULES),
        'execution_authorized': False, 'training_authorized': False, 'test_authorized': False,
        'main_fit_authorized': False, 'candidate_FINAL_selected': False,
    }))


def validate_spec(spec):
    _need(type(spec) is dict and _hash(spec.get('contract_sha256'))
          and _sha({k: v for k, v in spec.items() if k != 'contract_sha256'}) == spec['contract_sha256'], 'SPEC_SEAL')
    _need(spec['schema_version'] == SPEC_SCHEMA and spec['timeframes'] == list(TF)
          and spec['UTC_target_years'] == list(YEARS) and spec['rules'] == RULES
          and all(spec[k] is False for k in ['execution_authorized', 'training_authorized',
                  'test_authorized', 'main_fit_authorized', 'candidate_FINAL_selected']), 'FROZEN_SCOPE')
    _references(spec['references'])
    hlc.require_all_tf_provider_hlc_state_contract(spec['target_contract'])
    hlc.require_all_tf_provider_hlc_certificate_ref(spec['target_observation_certificate_ref'])
    _need(spec['fit_population']['scope'] == 'TRAIN' and spec['fit_population']['issue_rows'] == ROWS
          and spec['fit_population']['truth_sha256'] == spec['truth_sha256']
          and set(spec['populations']) == {'activity', 'side'}
          and all(set(spec['populations'][task]) == set(TF) for task in ('activity', 'side')), 'FROZEN_POPULATION')
    return spec


def _aggregate(values):
    values = list(values)
    return 'FAIL' if 'FAIL' in values else 'UNKNOWN' if not values or 'UNKNOWN' in values else 'PASS'


def _calibration(logits, labels, mask, starts):
    n = int(mask.sum())
    if not n:
        return {'status': 'UNKNOWN', 'row': None, 'equal_target': None}
    z, y = logits[mask].astype(np.float64), labels[mask].astype(np.float64)
    prob = np.exp(-np.logaddexp(0., -z)); groups, inverse, counts = np.unique(starts[mask], return_inverse=True, return_counts=True)
    index = np.minimum(np.searchsorted(np.linspace(0., 1., 11), prob, side='right') - 1, 9)
    result = {'status': 'DESCRIPTIVE_NOT_CALIBRATION_QUALIFICATION'}
    for name, weights in [('row', np.full(n, 1. / n)),
                          ('equal_target', 1. / (len(groups) * counts[inverse]))]:
        bins = []
        for k in range(10):
            take = index == k; mass = float(weights[take].sum())
            bins.append({'bin': k, 'issue_rows': int(take.sum()), 'weight_mass': mass,
                         'mean_probability': float(np.dot(weights[take], prob[take]) / mass) if mass else None,
                         'true_UP_fraction': float(np.dot(weights[take], y[take]) / mass) if mass else None})
        result[name] = {'mean_probability_minus_truth_pp': float(100 * np.dot(weights, prob - y)), 'bins': bins}
    return result


def _panel(logits, prior_logit, labels, mask, starts, *, side):
    panel = binary_panel(logits, np.full(len(logits), prior_logit, np.float32), labels, mask, starts)
    panel['frozen_V3_prior'] = panel.pop('matched_v2_prior')
    if panel['equal_target_weight'] is not None:
        eq = panel['equal_target_weight']; eq['frozen_V3_prior'] = eq.pop('matched_v2_prior')
    if side:
        panel['calibration'] = _calibration(logits, labels, mask, starts)
        up, down = np.unique(starts[mask & labels]), np.unique(starts[mask & ~labels])
        panel['UP_DOWN_target_support'] = {'UP_unique_targets': len(up), 'DOWN_unique_targets': len(down),
            'targets_in_both': len(np.intersect1d(up, down)), 'independent_outcomes_claimed': False}
    return panel


def _side_rank_guard(panel):
    rank = panel['rank_and_variation']
    if not rank['positive'] or not rank['negative']:
        return 'UNKNOWN'
    return 'PASS' if rank['nonconstant'] and rank['AUC'] is not None and rank['AUC'] > .5 else 'FAIL'


def _metric_verdict(panel, weighting, strict):
    if not panel['true_positive_count'] or not panel['true_negative_count']:
        return 'UNKNOWN'
    values = panel['delta_model_minus_prior'] if weighting == 'row' else panel['equal_target_weight']['delta_model_minus_prior']
    return 'PASS' if all(v < 0 if strict else v <= 0 for v in values.values()) else 'FAIL'


def review_outputs(*, context, outputs, spec, references, expect_initial=False):
    """Pure CPU array review; no payload IO/model/CUDA/fit. Unknowns stay unknown."""
    validate_spec(spec); _references(references)
    _need(references == spec['references'], 'AUTHENTICATED_REFERENCES')
    truth, target, prior = context['truth'], context['target_contract'], context['prior']
    _need(target == spec['target_contract'] and context['truth_sha256'] == spec['truth_sha256']
          and prior['contract_sha256'] == spec['prior_contract_sha256']
          and prior['fit_population'] == spec['fit_population'] and prior['per_task'] == spec['populations'], 'CONTEXT_IDENTITY')
    objective.require_native_future_state_train_prior(prior, target_contract=target)
    _need(context['input_binding']['contract_sha256'] == spec['model_contract']['input_binding_sha256']
          and context['normalization']['contract_sha256'] == spec['model_contract']['input_normalization_sha256'], 'FEATURE_IDENTITY')
    _need({k for k, v in truth.items() if isinstance(v, np.ndarray)} == set(TRUTH_FIELDS)
          and truth['target_observation_certificate_ref'] == spec['target_observation_certificate_ref'], 'TRUTH_INVENTORY_CERTIFICATE')
    issues = truth['issue_ns']; _need(len(issues) == ROWS and np.all(np.diff(issues) > 0)
          and _array_sha(issues) == spec['fit_population']['issue_ns_sha256'], 'ISSUE_AXIS')
    masks = objective._native_future_state_truth(truth, rows=ROWS,
        fold_upper_ns=spec['fit_population']['cutoff_ns'], target_contract=target)
    _need(objective.native_future_state_truth_sha256(truth, target_contract=target) == spec['truth_sha256'], 'TRUTH_SHA')
    _need(set(outputs) == {'activity_logits', 'side_logits'} and all(isinstance(v, np.ndarray)
          and v.dtype == np.float32 and v.shape == (ROWS, 5) and np.isfinite(v).all() for v in outputs.values()), 'LOGIT_HEADER')
    logits = {}
    for task, label in [('activity', 'activity'), ('side', 'side_up')]:
        logits[task] = []
        for j, tf in enumerate(TF):
            mask = masks[task][:, j]; cell = prior['per_task'][task][tf]; y = truth[label][mask, j]
            _need(_array_sha(mask) == cell['mask_sha256'] and int(mask.sum()) == cell['count']
                  and np.bincount(y.astype(np.int64), minlength=2).tolist() == cell['class_counts']
                  and objective._native_prior_array_sha({'issue_ns': issues[mask],
                      'target_start_ns': truth['target_start_ns'][mask, j], 'label': y}) == cell['population_sha256'], 'MASK_POPULATION')
            probabilities = np.asarray(cell['probabilities'], np.float64)
            logits[task].append(np.float32(np.log(probabilities[1]) - np.log(probabilities[0])))
        if expect_initial:
            _need(np.array_equal(outputs[task + '_logits'], np.broadcast_to(np.asarray(logits[task], np.float32), (ROWS, 5))), 'INITIAL_EXACT_PRIOR')
    # Existing objective remains the authoritative partial/singleton score owner.
    import torch
    canonical = objective.native_future_state_score_sums(
        model_output={k: torch.from_numpy(v.copy()) for k, v in outputs.items()},
        truth=truth, fold_upper_ns=spec['fit_population']['cutoff_ns'], target_contract=target)
    pooled, years, outside, support = {}, {}, {}, {}
    for j, tf in enumerate(TF):
        starts = truth['target_start_ns'][:, j]
        year = starts.astype('datetime64[ns]').astype('datetime64[Y]').astype(np.int64) + 1970
        def panel(take):
            return {task: _panel(outputs[task + '_logits'][:, j], logits[task][j], truth[label][:, j],
                        take & masks[task][:, j], starts, side=task == 'side')
                    for task, label in [('activity', 'activity'), ('side', 'side_up')]}
        pooled[tf] = panel(np.ones(ROWS, np.bool_))
        years[tf] = {str(y): panel(year == y) for y in YEARS}
        take = ~np.isin(year, YEARS)
        outside[tf] = {'issue_rows': int(take.sum()), 'activity_known': int((take & masks['activity'][:, j]).sum()),
                       'side_known': int((take & masks['side'][:, j]).sum())}
        support[tf] = {'activity_known': int(masks['activity'][:, j].sum()),
                       'side_known': int(masks['side'][:, j].sum()),
                       'FLAT': int((masks['activity'][:, j] & ~truth['activity'][:, j]).sum()),
                       'ACTIVE_unknown_side': int((masks['activity'][:, j] & truth['activity'][:, j] & ~masks['side'][:, j]).sum()),
                       'unknown_activity': int((~masks['activity'][:, j]).sum()),
                       'mature_unknown_activity': int((truth['mature'][:, j] & ~masks['activity'][:, j]).sum()),
                       'immature': int((~truth['mature'][:, j]).sum())}
    for tf in TF:
        for task in ('activity', 'side'):
            panel, cell = pooled[tf][task], canonical['per_tf'][tf][task]
            _need(panel['count'] == cell['count'] and panel['unique_target_starts'] == cell['unique_target_starts']
                  and all(np.isclose(panel['model'][name] * panel['count'], cell[total], rtol=1e-12, atol=1e-7)
                          for name, total in [('NLL', 'log_loss_sum'), ('brier', 'brier_sum')])
                          if panel['count'] else cell['count'] == 0, 'CANONICAL_ROW_SCORES')
    primary = {tf: {weight: _metric_verdict(pooled[tf]['side'], weight, True)
                   for weight in ('row', 'equal_target')} for tf in TF}
    stability = {tf: {y: {weight: _metric_verdict(years[tf][y]['side'], weight, False)
                   for weight in ('row', 'equal_target')} for y in map(str, YEARS)} for tf in TF}
    macro = {}
    for tf in TF:
        macro[tf] = {}
        for weight in ('row', 'equal_target'):
            panels = [years[tf][str(y)]['side'] for y in YEARS]
            values = [x['model'] if weight == 'row' else (x['equal_target_weight'] or {}).get('model') for x in panels]
            macro[tf][weight] = {'status': 'DESCRIPTIVE' if all(v is not None for v in values) else 'UNKNOWN',
                'model': {metric: float(np.mean([v[metric] for v in values])) for metric in ('NLL', 'brier')}
                         if all(v is not None for v in values) else None}
    proper_verdict = _aggregate(v for x in primary.values() for v in x.values())
    rank_guards = {tf: _side_rank_guard(pooled[tf]['side']) for tf in TF}
    stability_verdict = _aggregate(v for x in stability.values() for y in x.values() for v in y.values())
    comparison_verdict = _aggregate([proper_verdict, *rank_guards.values(), stability_verdict])
    return {'schema_version': REVIEW_SCHEMA, 'spec_sha256': spec['contract_sha256'],
        'truth_sha256': spec['truth_sha256'], 'rows': ROWS, 'expect_initial': expect_initial,
        'output_sha256': {k: _array_sha(v) for k, v in outputs.items()},
        'canonical_scores': canonical,
        'proper_partial_NLL': sum(x['partial_log_loss_sum'] for x in canonical['per_tf'].values()) / max(1, sum(x['activity']['count'] for x in canonical['per_tf'].values())),
        'all_TF': pooled, 'by_UTC_target_year': years, 'equal_year_macro': macro,
        'outside_declared_target_year_counts': outside, 'support': support,
        'primary_proper_score_checks': primary,
        'primary_proper_score_verdict': proper_verdict,
        'side_rank_guard_by_TF': rank_guards,
        'candidate_TRAIN_comparison_verdict': comparison_verdict,
        'primary_metric_witnesses': 20,
        'timeblock_stability': stability,
        'timeblock_stability_verdict': stability_verdict,
        'new_NN_forwards': 0, 'optimizer_updates': 0, 'prior_fits': 0,
        'execution_authorized': False, 'training_authorized': False,
        'learning_qualified': False, 'later_generalization': 'UNKNOWN', 'economic_edge': 'UNKNOWN'}
