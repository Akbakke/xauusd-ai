"""Focused full-reverse order, B6 scoring and fixed comparison boundaries."""
from copy import deepcopy
import math

import numpy as np
import pytest

from gx1.audit import run_market_state_ta_m5_full_reverse_order_v1 as worker
from gx1.audit import run_market_state_ta_m5_paired_batch_order_v1 as paired
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective

ARGS = dict(model_contract_sha256='1'*64, loss_contract_sha256='2'*64,
            feature_view_sha256='3'*64)


@pytest.fixture
def priors():
    return objective._seal(dict(schema_version='gx1_native_S_F_frozen_endpoint_priors_v1',
        fold_id=1, native_query_k=[1, 3, 12, 48, 288],
        loss_contract_sha256=objective.contract(ta_only=True, dedicated_m5_side=True)['contract_sha256'],
        tasks={task: dict(raw_logits=np.zeros((5, 2)).tolist()) for task in worker.TASKS}))


def arrays(n=22):
    value = dict(parents=np.arange(n, dtype=np.int64)*7+3,
                 issues=np.arange(n, dtype=np.int64)*300_000_000_000)
    for offset, task in enumerate(worker.TASKS):
        value[task+'_query'] = np.tile(np.array([1, 3, 12, 48, 288], np.int64), (n, 1))
        value[task+'_raw'] = np.empty((n, 5, 2), np.float32)
        value[task+'_raw'][..., 0] = np.arange(n, dtype=np.float32)[:, None]/20-offset
        value[task+'_raw'][..., 1] = np.arange(n, dtype=np.float32)[:, None]/9-1+offset/2
        value[task+'_allowed'] = np.broadcast_to(np.where(np.arange(n)[:, None] % 3 == 0, 1, 4), (n, 5)).astype(np.uint8).copy()
        value[task+'_eligible'] = np.ones((n, 5), bool)
        value[task+'_mature'] = np.ones((n, 5), bool)
    return value


def test_reverse_keeps_original_membership_and_tail_without_parent_modulo():
    parents = np.arange(29574, dtype=np.int64)*7+3
    indices = list(reversed(range(1848)))+[1848]
    order, positions = worker.build_reverse_order(parents, indices)
    assert np.array_equal(np.sort(order), parents)
    assert np.array_equal(order[:16], parents[29552:29568])
    assert np.array_equal(order[-22:-6], parents[:16])
    assert np.array_equal(order[-6:], parents[-6:])
    assert positions.dtype == np.int64 and len(np.unique(positions)) == 29574
    for step, original_batch in enumerate(indices):
        assert np.array_equal(order[step*16:min(step*16+16, 29574)],
                              parents[16*original_batch:min(16*original_batch+16, 29574)])


@pytest.mark.parametrize('kind', ['chronological', 'tail_first', 'duplicate', 'float_id', 'bool_id', 'int32', 'duplicate_parent'])
def test_bad_order_or_axis_rejected(kind):
    parents = np.arange(29574, dtype=np.int64)
    indices = list(reversed(range(1848)))+[1848]
    if kind == 'chronological': indices = list(range(1849))
    elif kind == 'tail_first': indices = [1848]+list(reversed(range(1848)))
    elif kind == 'duplicate': indices[1] = indices[0]
    elif kind == 'float_id': indices[0] = float(indices[0])
    elif kind == 'bool_id': indices[-2] = False
    elif kind == 'int32': parents = parents.astype(np.int32)
    elif kind == 'duplicate_parent': parents[1] = parents[0]
    with pytest.raises(RuntimeError):
        worker.build_reverse_order(parents, indices)


def test_tail_scores_use_global_support_but_mean_two_canonical_batches(priors):
    value = arrays()
    value['state_allowed'][16:, 0] = 5
    value['future_eligible'][1, 0] = False
    result = worker.sampled_fit_measurement_with_tail(value, [2., 8.], priors, **ARGS)
    assert result['rows'] == 22 and result['batches'] == 2 and result['last_batch_rows'] == 6
    assert result['mean_canonical_batch_loss'] == 5.  # Not the row-weighted 40/11.
    for task in worker.TASKS:
        known = value[task+'_eligible'][:, 0] & value[task+'_mature'][:, 0] & np.isin(value[task+'_allowed'][:, 0], (1, 4))
        z = value[task+'_raw'][known, 0, 1].astype(np.float64)
        y = value[task+'_allowed'][known, 0] == 4
        p = 1/(1+np.exp(-z))
        expected_brier = math.fsum((float(a)-float(b))**2 for a, b in zip(p, y))/len(y)
        expected_log = math.fsum(float(np.logaddexp(0., -a if b else a)) for a, b in zip(z, y))/len(y)
        stats = result['scores']['model'][task]['M5']['conditional_side']
        assert stats['count'] == len(y)
        assert result['means']['model'][task]['M5']['conditional_side']['brier'] == pytest.approx(expected_brier, abs=1e-15)
        assert result['means']['model'][task]['M5']['conditional_side']['log'] == pytest.approx(expected_log, abs=1e-15)
    assert result['scores']['model']['state']['M5']['conditional_side']['count'] == 16
    assert result['scores']['model']['future']['M5']['conditional_side']['count'] == 21
    assert objective._seal({k: v for k, v in result.items() if k != 'contract_sha256'}) == result


def test_complete_batch_scorer_matches_qualified_paired_math(priors):
    value = arrays(32)
    old = paired.sampled_fit_measurement(value, [1., 2.], priors, **ARGS)
    new = worker.sampled_fit_measurement_with_tail(value, [1., 2.], priors, **ARGS)
    for key in ('scores', 'means', 'observation_chain_sha256', 'mean_canonical_batch_loss', 'canonical_batch_losses'):
        assert new[key] == old[key]


def test_cached_later_uses_native_endpoint_batch_chain_not_train_schema(priors):
    source = arrays(32)
    cache = {k: source[k] for k in ('parents', 'issues')}
    cache['query_native'] = source['future_query']
    for task in worker.TASKS:
        cache[task+'_activity_side_logits'] = source[task+'_raw']
        for key in ('allowed', 'eligible', 'mature'):
            cache[task+'_'+key] = source[task+'_'+key]
    result = worker.cached_endpoint_measurement(cache, priors,
        model_contract_sha256=ARGS['model_contract_sha256'], feature_view_sha256=ARGS['feature_view_sha256'])
    accumulated = None
    for left in (0, 16):
        rows, clocks = cache['parents'][left:left+16], cache['issues'][left:left+16]
        query = cache['query_native'][left:left+16]
        truth = {task: dict(parent_rows=rows, issue_ns=clocks, validity_ns=clocks[:, None]+query*objective.TICK_NS,
                 **{key: cache[task+'_'+key][left:left+16] for key in ('allowed', 'eligible', 'mature')})
                 for task in worker.TASKS}
        measured = objective.endpoint_measurement_sums(state_raw=source['state_raw'][left:left+16],
            future_raw=source['future_raw'][left:left+16], state_truth=truth['state'], future_truth=truth['future'],
            parent_rows=rows, issue_ns=clocks, fold_id=1, partition='score', future_weight=1.,
            model_contract_sha256=ARGS['model_contract_sha256'], feature_view_sha256=ARGS['feature_view_sha256'],
            frozen_priors=priors, ta_only=True, dedicated_m5_side=True)
        accumulated = objective.merge_endpoint_measurements(accumulated, measured)
    assert result == objective.finalize_endpoint_measurement(accumulated)
    assert result['schema_version'] == 'gx1_native_S_F_endpoint_measurement_v1'
    assert result['partition'] == 'score' and result['batches'] == 2
    train = worker.sampled_fit_measurement_with_tail(source, [1., 2.], priors, **ARGS)
    assert result['observation_chain_sha256'] != train['observation_chain_sha256']
    cache['query_native'] = cache['query_native'].copy(); cache['query_native'][0, 1] = 1
    with pytest.raises(RuntimeError, match='ENDPOINT_CACHE_QUERY'):
        worker.cached_endpoint_measurement(cache, priors,
            model_contract_sha256=ARGS['model_contract_sha256'], feature_view_sha256=ARGS['feature_view_sha256'])


def test_preflight_exercises_actual_tail_endpoint_and_report_paths(priors):
    recipe = dict(batch_indices=list(reversed(range(1848)))+[1848], baseline_parity={'score_atol': 1e-6},
                  expected_model_contract_sha256=ARGS['model_contract_sha256'],
                  expected_loss_contract_sha256=ARGS['loss_contract_sha256'],
                  expected_view_sha256=ARGS['feature_view_sha256'])
    assert worker._synthetic_checks(priors, recipe)['model_constructions'] == 0


@pytest.mark.parametrize('n,losses', [(21, [1., 2.]), (22, [1.]), (22, [1., float('nan')]), (0, [])])
def test_wrong_tail_or_loss_count_rejected(priors, n, losses):
    with pytest.raises(RuntimeError, match='FIT_MEASUREMENT_BATCHES'):
        worker.sampled_fit_measurement_with_tail(arrays(n), losses, priors, **ARGS)


def comparison_fixture(priors, no_support=False):
    value = arrays()
    if no_support:
        for task in worker.TASKS: value[task+'_allowed'].fill(5)
    measured = worker.sampled_fit_measurement_with_tail(value, [1., 2.], priors, **ARGS)
    values = {label+'_'+panel: deepcopy(measured) for label in ('INITIAL', 'ORIGINAL_FINAL', 'REVERSED')
              for panel in ('TRAIN', 'LATER')}
    return values


def test_comparison_is_fixed_both_panels_with_distinct_reference_margins(priors):
    values = comparison_fixture(priors)
    assert worker.compare_m5_measurements(values, baseline_score_atol=1e-6)['decision'].startswith('FAIL_')
    for panel in ('TRAIN', 'LATER'):
        for task in worker.TASKS:
            for label, metric in (('INITIAL', 5e-7), ('ORIGINAL_FINAL', 2e-8), ('REVERSED', 0.)):
                values[label+'_'+panel]['means']['model'][task]['M5']['conditional_side'] = {'brier': metric, 'log': metric}
    report = worker.compare_m5_measurements(values, baseline_score_atol=1e-6)
    assert report['reference_margins'] == {'INITIAL': 1e-6, 'ORIGINAL_FINAL': 1e-8, 'PRIOR': 1e-8}
    assert report['decision'].startswith('FAIL_')
    for panel in ('TRAIN', 'LATER'):
        for task in worker.TASKS:
            values['INITIAL_'+panel]['means']['model'][task]['M5']['conditional_side'] = {'brier': 2e-6, 'log': 2e-6}
    assert worker.compare_m5_measurements(values, baseline_score_atol=1e-6)['decision'] == 'PASS_ALL_FOUR_CELLS'
    values['REVERSED_LATER']['means']['model']['future']['M5']['conditional_side']['log'] = 1.
    negative = worker.compare_m5_measurements(values, baseline_score_atol=1e-6)
    assert len(negative['cells']) == 4 and negative['decision'].startswith('FAIL_')


@pytest.mark.parametrize('field', ['observation_chain_sha256', 'frozen_prior_contract_sha256', 'rows'])
def test_mismatched_observations_or_prior_rejected(priors, field):
    values = comparison_fixture(priors)
    values['ORIGINAL_FINAL_TRAIN'][field] = 'bad' if field != 'rows' else 21
    with pytest.raises(RuntimeError):
        worker.compare_m5_measurements(values, baseline_score_atol=1e-6)


def test_zero_side_support_is_negative_not_an_exception(priors):
    report = worker.compare_m5_measurements(comparison_fixture(priors, no_support=True), baseline_score_atol=1e-6)
    assert report['decision'].startswith('FAIL_')
    assert all(cell['count'] == 0 and not cell['passed'] for cell in report['cells'].values())


@pytest.mark.parametrize('atol', [float('inf'), float('nan'), -1., True])
def test_invalid_cached_score_tolerance_rejected(priors, atol):
    with pytest.raises(RuntimeError, match='BASELINE_SCORE_TOLERANCE'):
        worker.compare_m5_measurements(comparison_fixture(priors), baseline_score_atol=atol)
