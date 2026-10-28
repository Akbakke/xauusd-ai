import copy

import pytest

from gx1.scripts import compare_entry_direction_order_v1 as order


def provenance():
    fold = {'train_cutoff_ns': order.matched.TRAIN_END, 'oof_cutoff_ns': order.matched.LATER_END}
    shared = {'fold': fold, 'feature_view_identity': {'view': 'same'},
        'own_prefix_prior': {'prior': [0.5, 0.5, 0.5]}, 'context_map': {'context': 'same'},
        'fresh_identity_sha256': 'fresh'}
    old = {**shared, 'own_initial_state': {'path': 'old'}, 'test_data_used': False}
    new = {**shared, 'own_initial_state': {'path': 'new'}, 'test_data_used': False}
    prediction = {'sha256': 'f870c6eb0a6000b2be0197bfe650db3f35f41d125d91a90162df2d01aedf112a'}
    old_initial = {**shared, 'own_initial_state': old['own_initial_state'], 'predictions': prediction}
    new_initial = {**shared, 'own_initial_state': new['own_initial_state'], 'predictions': prediction.copy()}
    common_result = {'parent_cursor': 47602, 'optimizer_updates': 2976, 'source_unchanged': True,
        'forbidden_open_paths': []}
    old_result = {**common_result, 'parent_order_sha256': 'mixed'}
    new_result = {**common_result, 'parent_order_sha256': 'chronological'}
    review = {'status': 'COMPLETE_TRAIN_AND_OOF_PAIRED_EVALUATION_PENDING', 'test_data_used': False,
        'initial_state_identity_equal': True, 'initial_prediction_archive_sha256_equal': True}
    return review, old, new, old_initial, new_initial, old_result, new_result


def test_provenance_allows_separate_initial_files_with_identical_bound_state_and_predictions():
    order.require_provenance(*provenance())


@pytest.mark.parametrize('index,key,value,error', [
    (0, 'test_data_used', True, 'REVIEW'),
    (2, 'context_map', {'context': 'different'}, 'IDENTITY'),
    (4, 'own_initial_state', {'path': 'unexpected'}, 'INITIAL_PARITY'),
    (6, 'optimizer_updates', 2975, 'FIT'),
    (6, 'parent_order_sha256', 'mixed', 'FIT'),
])
def test_provenance_rejects_scope_or_comparability_mismatch(index, key, value, error):
    values = list(copy.deepcopy(provenance()))
    values[index][key] = value
    with pytest.raises(RuntimeError, match=error):
        order.require_provenance(*values)


def test_paired_result_names_describe_orders_and_keep_auc_intervals():
    result = {'panels': [{'arms': {'one': {'auc': .51}, 'three': {'auc': .52}},
        'three_minus_one': {'auc': .01}, 'group_intervals': [
            {'three_minus_one': {'auc': {'low': -.01, 'high': .03}},
             'individual_auc_descriptive': {'one': {'low': .49}, 'three': {'low': .50}}}]}],
        'primary_auc_improvement_supported_separately': False,
        'primary_joint_gate_supported': False, 'primary_support': {}}
    transformed = order.label_arms(result)
    panel = transformed['panels'][0]
    assert panel['arms']['mixed']['auc'] == .51
    assert panel['arms']['chronological']['auc'] == .52
    assert panel['chronological_minus_mixed']['auc'] == .01
    assert panel['group_intervals'][0]['individual_auc']['chronological']['low'] == .50
    assert 'three_minus_one' not in panel and not transformed['primary_auc_vs_mixed_supported']


def test_protocol_matches_the_predeclared_four_auc_intervals():
    assert order.PROTOCOL['per_interval_alpha'] == pytest.approx(.05 / 4)
    assert order.PROTOCOL['bootstrap_replicates'] == 10000
    assert order.PROTOCOL['day_seed'] == 20261027 and order.PROTOCOL['week_seed'] == 20261028
    assert len(order.PROTOCOL['primary_auc_intervals']) == 4
