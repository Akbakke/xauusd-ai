import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gx1.contracts import entry_native_direction_decimal_input_v1 as owner
from gx1.scripts import materialize_entry_direction_dense_v1 as dense
from tests.test_materialize_entry_direction_dense_v1 import decimal_grid_fixture


def make_case(tmp_path, *, invalid=None):
    source, legacy, cutoff, grid = decimal_grid_fixture()
    original = dense.build_dense_targets(source, legacy, cutoff_ns=cutoff)
    corrected = dense.build_decimal_grid_dense_targets(source, legacy, decimal_grid=grid, cutoff_ns=cutoff)
    def ref(path):
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    def archive(name, value):
        path = tmp_path / (name + '.npz'); np.savez_compressed(path, **value); return ref(path)
    def receipt(name, value):
        path = tmp_path / (name + '.json'); path.write_text(json.dumps(value)); return ref(path)
    refs = {'legacy': archive('legacy', legacy), 'original': archive('original', original), 'corrected': archive('corrected', corrected)}
    n = len(legacy['parent_rows']); horizons = list(range(1, 61))
    refs['original_result'] = receipt('original-result', {'schema_version': 'gx1_direction_dense_target_result_v1',
        'status': 'COMPLETE_REAL_TARGETS_ONLY', 'test_data_used': False, 'target': refs['original'],
        'legacy_anchor_byte_parity': {'1': True, '5': True, '15': True}, 'horizons_minutes': horizons, 'parents': n})
    audit = {'schema_version': 'gx1_direction_decimal_sign_audit_result_v1', 'status': 'COMPLETE',
        'test_data_used': False, 'new_model_forwards': 0, 'optimizer_steps': 0,
        'target_source_rows_and_masks_parity': True, 'vendor_to_canonical_float64_and_clock_parity': True}
    if invalid == 'audit': audit['target_source_rows_and_masks_parity'] = False
    refs['audit_result'] = receipt('audit', audit)
    refs['materialization_plan'] = receipt('materialization-plan', {'schema_version': 'gx1_direction_decimal_targets_plan_v1',
        'test_data_used': False, 'inputs': {'dense_targets': refs['original'], 'legacy_targets': refs['legacy'], 'audit_result': refs['audit_result']}})
    result = {'schema_version': 'gx1_direction_decimal_targets_result_v1', 'status': 'COMPLETE', 'test_data_used': False,
        'new_model_forwards': 0, 'optimizer_steps': 0, 'target_identity': owner.DECIMAL_TARGET_IDENTITY,
        'target': refs['corrected'], 'plan': refs['materialization_plan'], 'source_clock_bid_ask_and_decimal_grid_parity': True,
        'independent_audit_confusions_reproduced': True, 'nonzero_sign_flips': 0, 'horizons_minutes': horizons,
        'parent_rows': n, 'all_false_moves_removed': 1}
    if invalid == 'corrected': result['source_clock_bid_ask_and_decimal_grid_parity'] = False
    refs['corrected_result'] = receipt('corrected-result', result)
    inputs = {k: refs[k] for k in ('legacy', 'original', 'corrected', 'audit_result', 'materialization_plan')}
    inputs['materialization_result'] = refs['corrected_result']
    if invalid == 'links': inputs['corrected'] = refs['original']
    refs['consumer_plan'] = receipt('consumer-plan', {'schema_version': 'gx1_direction_decimal_consumer_plan_v1',
        'test_data_used': False, 'inputs': inputs})
    refs['consumer_result'] = receipt('consumer-result', {'schema_version': 'gx1_direction_decimal_consumer_result_v1',
        'status': 'COMPLETE', 'test_data_used': False, 'new_model_forwards': 0, 'optimizer_steps': 0,
        'plan': refs['consumer_plan'], 'target_identity': owner.DECIMAL_TARGET_IDENTITY,
        'corrected_axes_clocks_rows_masks_and_domain_verified': True, 'legacy_anchor_bytes_verified_on_original_only': True,
        'independent_60_horizon_confusion_matrices_reproduced': True, 'nonzero_sign_flips': 0,
        'horizons_minutes': horizons, 'parents': n, 'all_false_moves_removed': 1})
    return refs, {k: legacy[k] for k in owner.AXES}, legacy, corrected


def test_bound_decimal_input_returns_corrected_immutable_targets(tmp_path):
    refs, axes, legacy, corrected = make_case(tmp_path)
    got = owner.load_bound_decimal_direction_targets(refs=refs, axes=axes, legacy=legacy)
    assert np.array_equal(got['sign'], corrected['sign']) and got['sign'][0, 0] == 0
    assert set(got) == {'terminal_time_ns', 'observed', 'sign', 'mid_return_bps'}
    assert all(not value.flags.writeable for value in got.values())


@pytest.mark.parametrize('invalid,reason', [('corrected', 'CORRECTED_EVIDENCE'), ('links', 'CONSUMER_PLAN_LINKS'), ('audit', 'AUDIT_EVIDENCE')])
def test_bound_decimal_input_rejects_incoherent_evidence(tmp_path, invalid, reason):
    refs, axes, legacy, _ = make_case(tmp_path, invalid=invalid)
    with pytest.raises(RuntimeError, match=reason):
        owner.load_bound_decimal_direction_targets(refs=refs, axes=axes, legacy=legacy)


def test_bound_decimal_input_rejects_changed_file_and_in_memory_legacy(tmp_path):
    refs, axes, legacy, _ = make_case(tmp_path)
    altered = copy.deepcopy(legacy); altered['mid_move'][0, 0] += 1
    with pytest.raises(RuntimeError, match='LEGACY_MEMORY:mid_move'):
        owner.load_bound_decimal_direction_targets(refs=refs, axes=axes, legacy=altered)
    Path(refs['corrected']['path']).write_bytes(b'tampered')
    with pytest.raises(owner.binding.RandomAccessCampaignError, match='SHA-256 mismatch'):
        owner.load_bound_decimal_direction_targets(refs=refs, axes=axes, legacy=legacy)


def test_bound_decimal_input_rejects_wrong_physical_axes(tmp_path):
    refs, axes, legacy, _ = make_case(tmp_path)
    wrong = copy.deepcopy(axes); wrong['dynamic_entry_m1_row'][0] += 1
    with pytest.raises(RuntimeError, match='AXIS:dynamic_entry_m1_row'):
        owner.load_bound_decimal_direction_targets(refs=refs, axes=wrong, legacy=legacy)
