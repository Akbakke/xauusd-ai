"""TRAIN-only integer counting against direct synthetic queries; no model or backward."""
from types import MappingProxyType

import numpy as np
import pytest

from gx1.contracts import entry_dynamic_mtf_state_report_v1 as report
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
from tests.test_entry_dynamic_mtf_state_targets import START, build, frame_from_closes


def _series():
    minutes = [*range(7), *range(8, 35)]
    prices = [100.0] * 7 + list(np.linspace(100.0, 103.0, 14)) + list(np.linspace(103.0, 99.0, 13))
    frame = frame_from_closes(prices, minutes=minutes)
    frame.loc[10, ['high', 'low', 'close']] = [103.0, 99.0, 101.0]
    parents = np.arange(5, dtype=np.int64)
    issues = START + np.array([5, 10, 15, 20, 25], np.int64) * target.MINUTE_NS
    series = target.bind_dynamic_mtf_state_parents(build(frame), parent_rows=parents, issue_ns=issues)
    return series, parents, issues, int(START + 25 * target.MINUTE_NS)


@pytest.mark.parametrize('horizon', [1, 5, 15, 1441])
def test_weighted_train_counts_match_direct_queries_including_gaps_and_cutoff(horizon):
    series, parents, issues, cutoff = _series()
    leads = np.arange(1, horizon + 1, dtype=np.int64)
    got = report.dense_train_query_counts(series, prefix_rows=4, cutoff_ns=cutoff, lead_minutes=leads)
    q = target.query_dynamic_mtf_states(series, parent_rows=parents[:4], issue_ns=issues[:4],
                                        cutoff_ns=cutoff, lead_minutes=leads)
    direct = report.StatePriorAccumulator(lead_minutes=leads)
    direct.update(q['state'], q['observed'])
    assert got['class_counts'] == direct.counts.tolist()
    assert got['transition_counts'] == direct.transition_counts.tolist()
    assert got['eligible_query_cells_per_tf'] == q['observed'].sum(axis=(0, 2)).tolist()
    assert got['path_observed_query_cells_per_tf'] == q['path_observed'].sum(axis=(0, 2)).tolist()
    assert got['current_path_observed_per_tf'] == q['current_path_observed'].sum(axis=0).tolist()
    expected_current = np.stack([np.count_nonzero(q['current_observed'] & (q['current_state'] == k), axis=0)
                                 for k in range(3)], axis=1)
    assert got['current_state_counts_per_tf'] == expected_current.tolist()
    for name, bit in target.dynamic_mtf_state_target_contract()['reason_bits'].items():
        hits = (q['reason'] & bit) != 0
        assert got['diagnostic_reason_query_counts'][name] == hits.sum(axis=(0, 2)).tolist()
        assert got['excluded_reason_query_counts'][name] == (hits & ~q['observed']).sum(axis=(0, 2)).tolist()
    assert got['later_label_metrics_inspected'] is False and got['test_data_used'] is False


def test_train_counts_do_not_depend_on_later_label_reason_or_path_values():
    series, _, _, cutoff = _series()
    leads = np.arange(1, 11, dtype=np.int64)
    before = report.dense_train_query_counts(series, prefix_rows=4, cutoff_ns=cutoff, lead_minutes=leads)
    after_mask = series.arrays['validity_ns'] >= cutoff
    changed = dict(series.arrays)
    for name, value in [('state', 2), ('observed', True), ('path_observed', True), ('reason', 255)]:
        array = changed[name].copy()
        array[after_mask] = value
        changed[name] = array
    altered = target.DynamicMtfStateSeries(MappingProxyType(changed), series.metadata)
    assert report.dense_train_query_counts(altered, prefix_rows=4, cutoff_ns=cutoff, lead_minutes=leads) == before


def test_train_counts_refuse_inexact_partition_and_sparse_queries():
    series, _, _, cutoff = _series()
    with pytest.raises(RuntimeError, match='TRAIN_EXACT_PREFIX'):
        report.dense_train_query_counts(series, prefix_rows=3, cutoff_ns=cutoff, lead_minutes=[1, 2])
    with pytest.raises(RuntimeError, match='TRAIN_DENSE_GRID'):
        report.dense_train_query_counts(series, prefix_rows=4, cutoff_ns=cutoff, lead_minutes=[1, 3])
    with pytest.raises(RuntimeError, match='TRAIN_BOUNDARY'):
        report.dense_train_query_counts(series, prefix_rows=4, cutoff_ns=cutoff + 1, lead_minutes=[1, 2])


def test_train_prior_from_weighted_counts_matches_direct_unsmoothed_prior():
    series, parents, issues, cutoff = _series()
    arrays = dict(series.arrays)
    n = len(arrays['validity_ns'])
    arrays['state'] = np.broadcast_to((np.arange(n) % 3).astype(np.int8)[:, None], (n, 5)).copy()
    arrays['observed'] = np.ones((n, 5), dtype=bool)
    synthetic = target.DynamicMtfStateSeries(MappingProxyType(arrays), series.metadata)
    leads = np.arange(1, 11, dtype=np.int64)
    counts = report.dense_train_query_counts(synthetic, prefix_rows=4, cutoff_ns=cutoff, lead_minutes=leads)
    result = report.dense_train_prior_from_counts(counts, lead_minutes=leads)
    q = target.query_dynamic_mtf_states(synthetic, parent_rows=parents[:4], issue_ns=issues[:4],
                                        cutoff_ns=cutoff, lead_minutes=leads)
    direct = report.StatePriorAccumulator(lead_minutes=leads)
    direct.update(q['state'], q['observed'])
    assert result == direct.finalize()


def test_train_prior_rejects_wrong_or_overflowed_counts():
    from copy import deepcopy
    series, _, _, cutoff = _series()
    leads = np.arange(1, 11, dtype=np.int64)
    counts = report.dense_train_query_counts(series, prefix_rows=4, cutoff_ns=cutoff, lead_minutes=leads)
    for bad in (True, -1, 2**63):
        altered = deepcopy(counts)
        altered['class_counts'][0][0] = bad
        with pytest.raises(RuntimeError, match='TRAIN_COUNT_ARRAY'):
            report.dense_train_prior_from_counts(altered, lead_minutes=leads)
    altered = deepcopy(counts)
    altered['transition_counts'][0][0][0] = counts['class_counts'][0][0] + 1
    with pytest.raises(RuntimeError, match='TRAIN_COUNT_CONSISTENCY'):
        report.dense_train_prior_from_counts(altered, lead_minutes=leads)
    altered = deepcopy(counts)
    altered['later_label_metrics_inspected'] = True
    with pytest.raises(RuntimeError, match='TRAIN_COUNT_IDENTITY'):
        report.dense_train_prior_from_counts(altered, lead_minutes=leads)


def test_seven_day_weighted_counts_match_direct_queries_with_missing_minutes():
    minutes = np.delete(np.arange(10090), [2000, 5000])
    prices = 100.0 + 2.0 * np.sin(minutes / 30.0)
    frame = frame_from_closes(prices, minutes=minutes)
    source_cutoff = int(START + 10100 * target.MINUTE_NS)
    issue = np.array([START + 5 * target.MINUTE_NS], np.int64)
    parents = np.array([0], np.int64)
    series = target.bind_dynamic_mtf_state_parents(build(frame, cutoff=source_cutoff), parent_rows=parents, issue_ns=issue)
    cutoff = int(START + 10090 * target.MINUTE_NS)
    leads = np.arange(1, 10081, dtype=np.int64)
    got = report.dense_train_query_counts(series, prefix_rows=1, cutoff_ns=cutoff, lead_minutes=leads)
    q = target.query_dynamic_mtf_states(series, parent_rows=parents, issue_ns=issue, cutoff_ns=cutoff, lead_minutes=leads)
    direct = report.StatePriorAccumulator(lead_minutes=leads)
    direct.update(q['state'], q['observed'])
    assert got['class_counts'] == direct.counts.tolist()
    assert got['transition_counts'] == direct.transition_counts.tolist()
    assert got['missing_source_query_cells_per_tf'] == 2
    assert got['immature_query_cells_per_tf'] == 0
    assert got['path_observed_query_cells_per_tf'] == q['path_observed'].sum(axis=(0, 2)).tolist()


def _draft_fixture(monkeypatch):
    from copy import deepcopy
    from gx1.contracts import entry_model_native_train_recipe_v1 as recipes
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as owner
    series, _, _, cutoff = _series()
    arrays = dict(series.arrays)
    n = len(arrays['validity_ns'])
    arrays['state'] = np.broadcast_to((np.arange(n) % 3).astype(np.int8)[:, None], (n, 5)).copy()
    arrays['observed'] = np.ones((n, 5), dtype=bool)
    synthetic = target.DynamicMtfStateSeries(MappingProxyType(arrays), series.metadata)
    fold = {'fold_id': 1, 'prefix_rows': 4, 'oof_start_row': 4, 'oof_end_row': 5,
            'train_cutoff_ns': cutoff, 'oof_cutoff_ns': synthetic.metadata['cutoff_ns']}
    target_ref = {'path': '/synthetic/SERIES.json', 'sha256': 'a' * 64}
    review_ref = {'path': '/synthetic/RULE_REVIEW.json', 'sha256': 'b' * 64}
    sources = deepcopy(synthetic.metadata['source_refs'])
    rule = {'schema_version': 'gx1_dynamic_mtf_state_label_audit_v1', 'support_status': 'PASS', 'blockers': [],
        'target_rule_locked': True, 'test_data_used': False, 'later_metrics_inspected': False,
        'model_forwards': 0, 'optimizer_steps': 0, 'target_manifest': target_ref, 'target_sources': sources,
        'fold': fold, 'target_contract_sha256': synthetic.metadata['definition']['contract_sha256'],
        'independent_rule_review': {'decision': 'PASS_SAVED_TRAIN_RULE_WITNESSES'},
        'integrator_verification': {'ATR_thresholds_and_axes_bitidentical': True,
            'all_TRAIN_classes_and_transition_rows_present': True, 'all_output_refs_and_array_sha256_checked': True,
            'terminal_and_launcher_exit_zero': True}}
    recipe = recipes.native_direction_research_recipe(batch_size=16, max_updates=2, dynamic_state=True,
        current_state_first_loss_weight=1.0, extended_forecast_minutes=1441,
        extended_storage_reserve_bytes=5 * 1024**3)
    monkeypatch.setattr(owner, '_fold', lambda value: dict(value))
    monkeypatch.setattr(owner.private, 'BPS_FOLD', fold)
    monkeypatch.setattr(owner.campaign, 'require_binding', lambda ref, **kwargs: dict(ref))
    monkeypatch.setattr(owner.campaign, 'read_bound_json', lambda *args: rule)
    monkeypatch.setattr(target, 'load_dynamic_mtf_state_series', lambda *args, **kwargs: synthetic)
    args = {'numerical_recipe': recipe, 'target_manifest': target_ref, 'target_sources': sources,
            'fold': fold, 'prior_rule_review': review_ref}
    return owner, synthetic, rule, args


def test_extended_train_audit_is_explicitly_unadmitted_draft(monkeypatch):
    owner, series, _, args = _draft_fixture(monkeypatch)
    draft = owner.build_extended_train_label_audit_draft(**args)
    assert draft['schema_version'] == 'gx1_dynamic_mtf_state_label_audit_draft_v1'
    assert draft['support_status'] == 'DRAFT_NOT_ADMITTED' and draft['native_launch_authorized'] is False
    assert draft['prior']['parents'] == 4 and draft['lead_minutes'][-1] == 1441
    assert draft['objective_sha256'] == args['numerical_recipe']['research_objective_sha256']
    assert draft['later_metrics_inspected'] is False and draft['test_data_used'] is False
    monkeypatch.setattr(owner.campaign, 'read_bound_json', lambda *args: draft)
    monkeypatch.setattr(target, 'read_dynamic_mtf_state_series_metadata', lambda *args, **kwargs:
                        {'metadata': {'parents': 5, 'definition': series.metadata['definition']}})
    launch = {'fold': args['fold'], 'target_sources': args['target_sources'], 'target_manifest': args['target_manifest'],
              'label_audit': {'path': '/synthetic/draft.json', 'sha256': 'c' * 64}, 'numerical_recipe': args['numerical_recipe']}
    with pytest.raises(RuntimeError, match='STATE_TRAIN_LABEL_AUDIT'):
        owner.require_dynamic_state_target_binding(launch)


def test_extended_train_audit_refuses_unreviewed_or_later_selected_rule(monkeypatch):
    owner, _, rule, args = _draft_fixture(monkeypatch)
    rule['later_metrics_inspected'] = True
    monkeypatch.setattr(target, 'load_dynamic_mtf_state_series', lambda *args, **kwargs: pytest.fail('must stop before tape read'))
    with pytest.raises(RuntimeError, match='STATE_TRAIN_DRAFT_RULE_REVIEW'):
        owner.build_extended_train_label_audit_draft(**args)


def test_train_counts_reject_series_that_used_test_data():
    series, _, _, cutoff = _series()
    poisoned = target.DynamicMtfStateSeries(series.arrays, {**series.metadata, 'test_data_used': True})
    with pytest.raises(RuntimeError, match='TRAIN_TEST_CLOSED'):
        report.dense_train_query_counts(poisoned, prefix_rows=4, cutoff_ns=cutoff, lead_minutes=[1, 2])
