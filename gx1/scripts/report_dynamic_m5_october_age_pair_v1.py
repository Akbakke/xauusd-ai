"""Read-only paired October development report. No model call or selection gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from gx1.contracts.entry_dynamic_mtf_state_targets_v1 import (
    FLAT, LOCAL_WINDOW_M1_BARS, STATE_NAMES, TIMEFRAMES,
    load_dynamic_mtf_state_series, query_dynamic_mtf_states,
)
from gx1.contracts.entry_dynamic_mtf_state_report_v1 import StateScoreAccumulator
from gx1.contracts.entry_direction_block_rank_v1 import block_rank_pairs
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

SCHEMA = 'gx1_dynamic_m5_october_age_pair_descriptive_v1'
DAY_NS = 86_400_000_000_000
DRAWS = 10_000
SEED = 20260911  # Existing dynamic-state UTC-day convention, not a new selection threshold.


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError('[OCTOBER_AGE_PAIR_' + code + ']')


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_path(ref: dict) -> Path:
    _require(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'REF_SCHEMA')
    path = Path(ref['path'])
    _require(path.is_absolute() and path.is_file() and not path.is_symlink()
             and _sha(path) == ref['sha256'], 'REF_BYTES')
    return path


def _json(ref: dict) -> dict:
    value = json.loads(_verified_path(ref).read_text())
    _require(type(value) is dict, 'JSON_OBJECT')
    return value


def _array(descriptor: dict, *, parent: Path) -> np.ndarray:
    _require(type(descriptor) is dict and {'path', 'sha256', 'shape', 'dtype'} <= set(descriptor),
             'ARRAY_DESCRIPTOR')
    path = Path(descriptor['path'])
    _require(path.parent == parent and path.suffix == '.npy' and not path.is_symlink()
             and _sha(path) == descriptor['sha256'], 'ARRAY_BYTES')
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    _require(list(value.shape) == descriptor['shape'] and value.dtype.name == descriptor['dtype']
             and not value.flags.writeable, 'ARRAY_HEADER')
    return value


def _losses(probabilities: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int64)
    _require(probabilities.shape == (len(truth), 3) and np.isfinite(probabilities).all()
             and np.all((probabilities >= 0) & (probabilities <= 1))
             and np.all(np.abs(probabilities.sum(axis=1) - 1) <= 3 * np.finfo(np.float32).eps)
             and np.isin(truth, (0, 1, 2)).all(), 'PROBABILITY_DOMAIN')
    one_hot = np.eye(3, dtype=np.float64)[truth]
    with np.errstate(divide='ignore'):
        logloss = -np.log(probabilities[np.arange(len(truth)), truth])
    return {'brier': np.square(probabilities - one_hot).sum(axis=1), 'logloss': logloss}


def _mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if len(values) and np.isfinite(values).all() else None


def _day_interval(delta: np.ndarray, day: np.ndarray) -> dict:
    """Descriptive whole UTC outcome-day bootstrap; no PASS interpretation."""
    delta = np.asarray(delta, dtype=np.float64)
    unique, inverse, count = np.unique(day, return_inverse=True, return_counts=True)
    _require(len(delta) == len(day) and len(delta) > 0, 'DAY_DOMAIN')
    if len(unique) < 2 or not np.isfinite(delta).all():
        return {'days': len(unique), 'row': None, 'equal_day': None}
    total = np.bincount(inverse, weights=delta, minlength=len(unique))
    rng = np.random.Generator(np.random.PCG64(SEED))
    sampled = rng.integers(len(unique), size=(DRAWS, len(unique)))
    row = total[sampled].sum(axis=1) / count[sampled].sum(axis=1)
    equal_day = (total / count)[sampled].mean(axis=1)
    return {'days': len(unique),
            'row': [float(x) for x in np.quantile(row, [.025, .975], method='linear')],
            'equal_day': [float(x) for x in np.quantile(equal_day, [.025, .975], method='linear')]}


def _auc(probabilities: np.ndarray, truth: np.ndarray, keep: np.ndarray) -> float | None:
    y = truth[keep]
    if len(y) < 2 or np.unique(y).size < 2:
        return None
    p = np.asarray(probabilities[keep], dtype=np.float64)
    denom = p[:, 0] + p[:, 2]
    if not np.isfinite(p).all() or np.any(denom <= 0):
        return None
    return float(roc_auc_score(y == 2, p[:, 2] / denom))


def _auc_day_comparison(models: dict, truth: np.ndarray, days: np.ndarray) -> dict:
    keep = truth != FLAT
    y = (truth[keep] == 2).astype(np.int8)
    unique, block, count = np.unique(days[keep], return_inverse=True, return_counts=True)
    if len(unique) < 2 or np.unique(y).size < 2:
        return {'equal_day_auc': {name: None for name in models},
                'candidate_minus_reference_intervals': {name: {'row': None, 'equal_day': None}
                    for name in ('control', 'prior', 'markov')}, 'nonflat_days': len(unique)}
    rng = np.random.Generator(np.random.PCG64(SEED))
    choices = rng.integers(len(unique), size=(DRAWS, len(unique)))
    weights = np.zeros((DRAWS, len(unique)), dtype=np.int16)
    np.add.at(weights, (np.arange(DRAWS)[:, None], choices), 1)
    draws = {'row': {}, 'equal_day': {}}
    points = {}
    for name, probabilities in models.items():
        p = np.asarray(probabilities[keep], dtype=np.float64)
        denom = p[:, 0] + p[:, 2]
        if not np.isfinite(p).all() or np.any(denom <= 0):
            points[name] = None
            continue
        score = p[:, 2] / denom
        summary = block_rank_pairs(y, score, block.astype(np.int64))
        for mode, w in (('row', weights), ('equal_day', weights / count)):
            positive, negative = w @ summary['positive'], w @ summary['negative']
            pairs = positive * negative
            twice_wins = ((w @ summary['twice_wins']) * w).sum(axis=1)
            draws[mode][name] = np.divide(twice_wins, 2 * pairs,
                out=np.full(DRAWS, np.nan), where=pairs > 0)
        unit = 1.0 / count
        pairs = (unit @ summary['positive']) * (unit @ summary['negative'])
        points[name] = (float((unit @ summary['twice_wins'] @ unit) / (2 * pairs))
                        if pairs > 0 else None)
    intervals = {}
    for ref in ('control', 'prior', 'markov'):
        intervals[ref] = {}
        for mode in ('row', 'equal_day'):
            if 'candidate' not in draws[mode] or ref not in draws[mode]:
                intervals[ref][mode] = None
            else:
                delta = draws[mode]['candidate'] - draws[mode][ref]
                intervals[ref][mode] = ([float(x) for x in np.quantile(delta, [.025, .975], method='linear')]
                                        if np.isfinite(delta).all() else None)
    return {'equal_day_auc': points, 'candidate_minus_reference_intervals': intervals,
            'nonflat_days': len(unique)}


def _arm(review_ref: dict) -> dict:
    review = _json(review_ref)
    _require(review.get('schema_version') == 'gx1_october_native_score_terminal_review_v1'
             and review.get('status') == 'COMPLETE_CONSUMED_OCTOBER_DEVELOPMENT_SCORE'
             and review.get('oof_parent_rows') == 6323 and review.get('physical_optimizer_updates') == 0
             and review.get('test_data_used') is False, 'ARM_REVIEW')
    receipt, result, recipe = (_json(review[name]) for name in ('receipt', 'result', 'recipe'))
    _require(receipt['outcome'] == 'COMPLETE' and receipt['guard_decision'] == 'PASS'
             and receipt['trainer_guard_exit_code'] == receipt['progress_observer_exit_code'] == 0
             and result['outcome'] == 'COMPLETE' and result['physical_optimizer_updates'] == 0
             and result['counts']['backward_calls'] == result['counts']['optimizer_step_calls'] == 0
             and result['oof_cursor'] == 6323 and result['test_data_used'] is False
             and recipe['training_enabled'] is False and recipe['test_data_used'] is False,
             'ARM_TERMINAL')
    oof = _json(review['oof'])
    manifest = _json(review['predictions_manifest'])
    _require(manifest.get('schema_version') == 'gx1_dynamic_mtf_state_prediction_arrays_v1'
             and oof.get('schema_version') == 'gx1_dynamic_mtf_state_measurement_v1'
             and oof['predictions'] == review['predictions_manifest']
             and oof['fold'] == recipe['fold']
             and oof['target_manifest'] == manifest['target_manifest'] == recipe['target_manifest']
             and oof['target_sources'] == manifest['target_sources'] == recipe['target_sources']
             and oof['lead_minutes'] == manifest['lead_minutes'] == list(range(1, 1441))
             and oof['timeframes'] == manifest['timeframes'] == list(TIMEFRAMES)
             and manifest['cutoff_ns'] == recipe['fold']['oof_cutoff_ns']
             and oof['test_data_used'] is manifest['test_data_used'] is False,
             'ARM_PREDICTION_BINDING')
    parent = _verified_path(review['predictions_manifest']).parent
    arrays = {name: _array(manifest['arrays'][name], parent=parent)
              for name in ('parent_rows', 'issue_ns', 'probabilities')}
    _require(arrays['parent_rows'].shape == arrays['issue_ns'].shape == (6323,)
             and arrays['probabilities'].shape == (6323, 5, 1440, 3), 'ARM_SHAPES')
    return {'review': review, 'recipe': recipe, 'oof': oof, 'arrays': arrays}


def _compare(plan: dict) -> dict:
    control, candidate = (_arm(plan[name]) for name in ('control_review', 'candidate_review'))
    _require(control['review']['arm'] == 'control' and candidate['review']['arm'] == 'candidate'
             and candidate['review'].get('outcome_values_opened_for_review') is False,
             'MATCHED_ARM_ROLES')
    for key in ('candidate_plan', 'observer_axes', 'target_manifest', 'target_sources',
                'prefix_normalization', 'prefix_prior', 'reference_plan', 'fold'):
        _require(control['recipe'][key] == candidate['recipe'][key], 'MATCHED_RECIPE:' + key)
    fit = {name: _json(arm['recipe']['fit_terminal'])
           for name, arm in (('control', control), ('candidate', candidate))}
    for name, arm in (('control', control), ('candidate', candidate)):
        _require(arm['recipe']['fit_terminal'] == arm['recipe']['score_origin']['fit_terminal_review']
                 and arm['review']['fit_checkpoint'] == arm['recipe']['score_origin']['fit_checkpoint']
                 and fit[name]['arm'] == name and fit[name]['test_data_used'] is False,
                 'FIT_PROVENANCE:' + name)
    _require(fit['control']['train_parent_rows'] == fit['candidate']['training_parent_rows'] == 29849
             and fit['control']['inherited_optimizer_updates'] == fit['candidate']['physical_optimizer_updates'] == 1866
             and fit['control']['new_optimizer_updates'] == 0, 'FIT_MATCHED_EXPOSURE')
    excluded = {'confirmed_run_age_input', 'contract_sha256', 'research_objective_sha256'}
    base = {k: v for k, v in control['recipe']['numerical_recipe'].items() if k not in excluded}
    age = {k: v for k, v in candidate['recipe']['numerical_recipe'].items() if k not in excluded}
    _require(base == age and control['recipe']['numerical_recipe'].get('confirmed_run_age_input') is None
             and candidate['recipe']['numerical_recipe']['confirmed_run_age_input'] is True
             and base['seed'] == 20260911, 'MATCHED_NUMERICAL_RECIPE')
    for key in ('parent_rows', 'issue_ns'):
        _require(np.array_equal(control['arrays'][key], candidate['arrays'][key]), 'MATCHED_AXIS:' + key)
    _require(control['oof']['own_prefix_prior'] == candidate['oof']['own_prefix_prior']
             and control['oof']['lead_minutes'] == candidate['oof']['lead_minutes']
             and control['oof']['timeframes'] == candidate['oof']['timeframes'] == list(TIMEFRAMES),
             'MATCHED_PRIOR_AND_GRID')
    target_ref = control['recipe']['target_manifest']
    target_meta = _json(target_ref)['metadata']
    series = load_dynamic_mtf_state_series(target_ref,
        expected_source_refs=control['recipe']['target_sources'],
        expected_cutoff_ns=target_meta['cutoff_ns'])
    parent_rows = np.asarray(control['arrays']['parent_rows'], dtype=np.int64)
    issue_ns = np.asarray(control['arrays']['issue_ns'], dtype=np.int64)
    design = _json(plan['design'])['population']['development_fold']
    _require(plan['score_cutoff_ns_exclusive'] == design['cutoff_ns_exclusive']
             == control['recipe']['fold']['oof_cutoff_ns']
             and np.array_equal(parent_rows, np.arange(*design['parent_row_range'], dtype=np.int64))
             and len(parent_rows) == design['rows'] == 6323
             and issue_ns[0] == design['first_decision_ns']
             and issue_ns[-1] == design['last_decision_ns']
             and hashlib.sha256(parent_rows.astype('<i8', copy=False).tobytes()).hexdigest()
                 == design['parent_rows_sha256']
             and hashlib.sha256(issue_ns.astype('<i8', copy=False).tobytes()).hexdigest()
                 == design['decision_ns_sha256']
             and np.all(np.diff(issue_ns) > 0), 'OCTOBER_AXIS')
    native_leads = [LOCAL_WINDOW_M1_BARS[tf] for tf in TIMEFRAMES]
    query = query_dynamic_mtf_states(series, parent_rows=parent_rows, issue_ns=issue_ns,
        cutoff_ns=plan['score_cutoff_ns_exclusive'], lead_minutes=native_leads)
    prior = control['oof']['own_prefix_prior']
    scorer = StateScoreAccumulator(prior, lead_minutes=control['oof']['lead_minutes'],
        model_receives_current_state=True)
    by_tf = {}
    for tf_index, tf in enumerate(TIMEFRAMES):
        lead = native_leads[tf_index]
        observed = query['observed'][:, tf_index, tf_index]
        truth = query['state'][:, tf_index, tf_index][observed]
        day = query['validity_ns'][:, tf_index][observed] // DAY_NS
        _require(len(truth) > 0 and np.isin(truth, (0, 1, 2)).all(), 'OBSERVED_TARGET:' + tf)
        for arm in (control, candidate):
            _require(arm['oof']['model_metrics']['per_lead']['observed'][tf_index][lead-1] == len(truth),
                     'SAVED_OBSERVED_COUNT:' + tf)
        model = {name: np.asarray(arm['arrays']['probabilities'][:, tf_index, lead-1, :][observed])
                 for name, arm in (('control', control), ('candidate', candidate))}
        model['prior'] = np.broadcast_to(scorer.prior[tf_index], (len(truth), 3))
        cases = np.where(query['current_observed'][:, tf_index],
                         query['current_state'][:, tf_index], 3)[observed]
        _require(np.isin(cases, (0, 1, 2, 3)).all(), 'CURRENT_STATE:' + tf)
        model['markov'] = scorer.markov_grid[tf_index, cases, lead-1, :]
        losses = {name: _losses(p, truth) for name, p in model.items()}
        saved = {name: arm['oof']['model_metrics']['per_lead']['brier'][tf_index][lead-1]
                 for name, arm in (('control', control), ('candidate', candidate))}
        for name in ('control', 'candidate'):
            _require(abs(_mean(losses[name]['brier']) - saved[name]) < 1e-6,
                     'SAVED_BRIER_PARITY:' + tf + ':' + name)
        for name in ('prior', 'markov'):
            field = name + '_brier'
            recorded = control['oof']['model_metrics']['per_lead'][field][tf_index][lead-1]
            _require(abs(_mean(losses[name]['brier']) - recorded) < 1e-6,
                     'SAVED_REFERENCE_PARITY:' + tf + ':' + name)
        scores = {name: {key: _mean(value) for key, value in loss.items()}
                  for name, loss in losses.items()}
        auc = {name: _auc(p, truth, truth != FLAT) for name, p in model.items()}
        auc_day = _auc_day_comparison(model, truth, day)
        intervals = {ref: {metric: _day_interval(losses['candidate'][metric] - losses[ref][metric], day)
                           for metric in ('brier', 'logloss')}
                     for ref in ('control', 'prior', 'markov')}
        all_days = query['validity_ns'][:, tf_index] // DAY_NS
        all_reason = query['reason'][:, tf_index, tf_index]
        daily = []
        for date in np.unique(all_days):
            on_day = all_days == date
            observed_day = on_day & query['observed'][:, tf_index, tf_index]
            missing_codes, missing_counts = np.unique(all_reason[on_day & ~observed_day], return_counts=True)
            daily.append({'day_since_epoch': int(date), 'parents': int(on_day.sum()),
                'observed': int(observed_day.sum()), 'unobserved': int((on_day & ~observed_day).sum()),
                'observed_class_counts': {name: int((query['state'][:, tf_index, tf_index][observed_day] == i).sum())
                                          for i, name in enumerate(STATE_NAMES)},
                'unobserved_reason_code_counts': {str(int(code)): int(count)
                                                  for code, count in zip(missing_codes, missing_counts)}})
        by_tf[tf] = {'native_lead_minutes': lead, 'observed': len(truth),
            'unobserved': len(parent_rows)-len(truth), 'outcome_utc_days': len(np.unique(day)),
            'class_counts': {name:int((truth == i).sum()) for i,name in enumerate(STATE_NAMES)},
            'coverage_by_outcome_utc_day': daily, 'scores': scores,
            'conditional_up_down_auc': auc, 'conditional_up_down_auc_equal_day': auc_day['equal_day_auc'],
            'conditional_up_down_nonflat_days': auc_day['nonflat_days'],
            'candidate_minus_reference_up_down_auc_utc_day_intervals': auc_day['candidate_minus_reference_intervals'],
            'candidate_minus_reference_utc_day_intervals': intervals,
            'fullgrid_brier_saved': {name: arm['oof']['model_metrics']['per_timeframe']['brier'][tf_index]
                                     for name,arm in (('control',control),('candidate',candidate))},
            'fullgrid_markov_brier_saved': control['oof']['model_metrics']['per_timeframe']['markov_brier'][tf_index],
            'fullgrid_prior_brier_saved': control['oof']['model_metrics']['per_timeframe']['prior_brier'][tf_index]}
    return {'schema_version': SCHEMA, 'status':'DESCRIPTIVE_DEVELOPMENT_ONLY_NO_PASS_GATE',
        'observed_utc':datetime.now(timezone.utc).isoformat(),
        'control_review':plan['control_review'],
        'candidate_review':plan['candidate_review'], 'target_manifest':target_ref,
        'score_cutoff_ns_exclusive':plan['score_cutoff_ns_exclusive'],
        'bootstrap':{'unit':'whole_UTC_outcome_day','draws':DRAWS,'seed':SEED,
                     'interval':'central_two_sided_percentile_95','estimands':['row_mean','equal_day_mean']},
        'timeframes':by_tf, 'age_based_cohort_selection':False,
        'candle_reference':{'status':'UNMEASURED','reason':'No pre-October source-qualified frozen candle reference is bound in the matched score recipes.'},
        'utc_day_interval_limit':'D1 outcome windows overlap UTC days; these intervals remain descriptive.',
        'acceptance_rule_prebound_before_model_score':False,
        'independent_edge':'UNKNOWN','economic_edge':'UNKNOWN','test_data_used':False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    _require(args.plan.is_absolute() and _sha(args.plan) == args.plan_sha256, 'PLAN_BYTES')
    plan = json.loads(args.plan.read_text())
    _require(plan['schema_version'] == SCHEMA + '_plan' and plan['source_file_sha256'] == _sha(Path(__file__))
             and plan['status'] == 'DESCRIPTIVE_PRE_OUTCOME_AFTER_SCORING'
             and plan['decision_authority'] == 'NONE'
             and plan['test_data_used'] is False, 'PLAN_IDENTITY')
    report = _compare(plan)
    report['plan'] = {'path':str(args.plan), 'sha256':args.plan_sha256}
    path = Path(plan['output_path'])
    _require(path.is_absolute() and not path.exists(), 'OUTPUT_NO_REPLACE')
    path.parent.mkdir(parents=True, exist_ok=True)
    _publish_bytes(path, (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
    print(json.dumps({'status': report['status'], 'output_path': str(path),
                      'output_sha256': _sha(path), 'm5_plus5': report['timeframes']['M5']['scores']},
                     sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
