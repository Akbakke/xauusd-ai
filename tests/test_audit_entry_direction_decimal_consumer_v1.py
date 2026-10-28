import copy
from types import SimpleNamespace

import numpy as np
import pytest

from gx1.scripts import audit_entry_direction_decimal_consumer_v1 as audit


def receipt_case():
    decimal = SimpleNamespace(PARENTS=4, FOLD_TRAIN=2, FOLD_LATER_END=4, TRAIN_CUTOFF=100, LATER_CUTOFF=200)
    refs = {name: {'path': '/' + name, 'sha256': name} for name in ('materialization_plan', 'corrected',
        'legacy', 'original', 'audit_result', 'oof_predictions')}
    plan = {'inputs': refs}
    mat = {'schema_version': 'gx1_direction_decimal_targets_result_v1', 'status': 'COMPLETE',
        'plan': refs['materialization_plan'], 'target': refs['corrected'], 'target_identity': 'exact_original_decimal_midpoint_v1',
        'parent_rows': 4, 'horizons_minutes': list(range(1, 61)), 'source_clock_bid_ask_and_decimal_grid_parity': True,
        'independent_audit_confusions_reproduced': True, 'all_false_moves_removed': 976,
        'nonzero_sign_flips': 0, 'test_data_used': False}
    old_plan = {'schema_version': 'gx1_direction_decimal_targets_plan_v1', 'inputs': {
        'legacy_targets': refs['legacy'], 'dense_targets': refs['original'], 'audit_result': refs['audit_result']}}
    signs = {'schema_version': 'gx1_direction_decimal_sign_audit_result_v1', 'status': 'COMPLETE',
        'test_data_used': False, 'vendor_to_canonical_float64_and_clock_parity': True, 'target_source_rows_and_masks_parity': True}
    oof = {'schema_version': 'gx1_native_direction_prefix_measurement_v1', 'role': 'OOF', 'test_data_used': False,
        'predictions': refs['oof_predictions'], 'fold': {'fold_id': 1, 'prefix_rows': 2,
            'oof_start_row': 2, 'oof_end_row': 4, 'train_cutoff_ns': 100, 'oof_cutoff_ns': 200}}
    return plan, mat, old_plan, signs, oof, decimal


def test_receipts_reject_mismatched_identity_or_source_lineage():
    plan, mat, old_plan, signs, oof, decimal = receipt_case()
    audit.require_receipts(plan, mat, old_plan, signs, oof, decimal=decimal)
    for field, value in (('target_identity', 'legacy'), ('nonzero_sign_flips', 1), ('test_data_used', True)):
        changed = copy.deepcopy(mat); changed[field] = value
        with pytest.raises(RuntimeError, match='MATERIALIZATION_RECEIPT'):
            audit.require_receipts(plan, changed, old_plan, signs, oof, decimal=decimal)
    changed = copy.deepcopy(old_plan); changed['inputs']['dense_targets']['sha256'] = 'other'
    with pytest.raises(RuntimeError, match='ORIGINAL_LINEAGE'):
        audit.require_receipts(plan, mat, changed, signs, oof, decimal=decimal)
    changed = copy.deepcopy(oof); changed['fold']['oof_end_row'] += 1
    with pytest.raises(RuntimeError, match='OOF_RECEIPT'):
        audit.require_receipts(plan, mat, old_plan, signs, changed, decimal=decimal)


def test_saved_oof_join_rejects_shifted_parent_or_label():
    one = np.arange(4, dtype=np.int64)
    original = {key: one.copy() for key in ('parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns')}
    original['observed'] = np.array([True, True, True, False], dtype=np.bool_)
    original['sign'] = np.array([0, 1, -1, 0], dtype=np.int8)
    oof = {key: value[2:].copy() for key, value in original.items()}
    assert audit.compare_oof_rows(oof, original, first=2, last=4) == 2
    for key in ('parent_rows', 'sign'):
        wrong = copy.deepcopy(oof); wrong[key][0] += 1
        with pytest.raises(RuntimeError, match='OOF_JOIN'):
            audit.compare_oof_rows(wrong, original, first=2, last=4)


def test_bps_difference_preserves_missing_and_reports_exact_zero():
    old = {'observed': np.array([True, True, False]), 'mid_return_bps': np.array([1., 0., 999.])}
    new = {'observed': old['observed'].copy(), 'mid_return_bps': np.array([1.25, 0., 0.])}
    result = audit.bps_difference(old, new)
    assert result['observed_cells'] == 2 and result['different_float_values'] == 1
    assert result['absolute_difference_bps_quantiles']['1.0'] == .25
    new['observed'][2] = True
    with pytest.raises(RuntimeError, match='BPS_MASK'):
        audit.bps_difference(old, new)
