"""Frozen actual measurement coordinates, distinct TRAIN/control roles."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.contracts import unified_exit_bounded_val_cohort_v1 as owner
from gx1.contracts.unified_exit_reference_policy_v1 import reference_policy_contract
from gx1.scripts import run_unified_exit_random_access_val_v1 as val
from tests.test_native_chronological_control_source import _binding
from tests.test_chronological_sampled_targets import _sampled_context


def _write(path, value):
    path.write_text(json.dumps(value))
    return _binding(path)


def _fixture(tmp_path):
    n = 700
    times = np.r_[pd.date_range('2026-02-01T00:00Z', periods=300, freq='5min').asi8,
                  pd.date_range('2026-03-01T00:00Z', periods=400, freq='5min').asi8]
    frame = pd.DataFrame({'entry_row_index': np.arange(n, dtype='int64'),
        'parent_entry_row_index': np.arange(n, dtype='int64') + 11,
        'entry_time_ns': times, 'first_state_time_ns': times + 300_000_000_000,
        'parent_m1_start_row': np.arange(n, dtype='int64') * 5 + 479,
        'child_m1_start_row': np.arange(n, dtype='int64') * 5 + 479,
        'successor_transition_count': np.full(n, 600, dtype='int64'),
        'lifecycle_state_count': np.full(n, 601, dtype='int64'),
        'economic_terminal': False, 'right_censored': True, 'entry_bid': 100., 'entry_ask': 100.1})
    index = tmp_path / 'index.parquet'
    frame.to_parquet(index, index=False)
    children = {'train': np.arange(256, dtype='int64')[::-1],
                'control': np.linspace(300, n - 1, 256, dtype='int64')}
    rows = {}
    for role, ids in children.items():
        path = tmp_path / (role + '.npy')
        np.save(path, ids + 11)
        rows[role] = _binding(path)
    design = {'schema_version': 'gx1_frozen_chronological_learning_design_v1',
        'status': 'DESIGN_AND_CONTROL_IDS_FROZEN_NOT_EXECUTABLE',
        'scope': {'test_sealed': True, 'existing_control_periods_are_reused_development': True,
                  'native_launch_authorized': False, 'one_experiment': True},
        'selection': {'control_entries': 256}, 'budget': {'later_control_entries': 256},
        'calendar': {'train_entry_start_inclusive': '2025-06-01T00:00Z',
            'train_control_cutoff': '2026-03-01T00:00Z',
            'development_control_entry_end_exclusive': '2026-06-01T00:00Z',
            'index': _binding(index), 'bindings': {'CONTROL256_PARENT_ROWS': rows['control']}},
        'targets': {'reference_policy': reference_policy_contract()}}
    db = _write(tmp_path / 'design.json', design)
    aux = _write(tmp_path / 'aux.json', {'frozen_design': db,
        'bindings': {'TRAIN256_PROBE_PARENT_ROWS': rows['train']}})
    arrays = {}
    bindings = {}
    for role, ids in children.items():
        offsets = np.tile(np.array([7, 4, 7, 1], dtype='int64'), (256, 1))
        data = {'parent_rows': ids + 11, 'child_rows': ids, 'entry_time_ns': times[ids],
            'sampled_state_indices': offsets,
            'sampled_reference_end_close_ns': times[ids, None] + (offsets + 126) * 60_000_000_000,
            'anchor_reference_end_close_ns': times[ids] + 126 * 60_000_000_000}
        path = tmp_path / (role + '.npz')
        np.savez(path, **data)
        arrays[role] = data
        bindings[role] = _binding(path)
    result = {'schema_version': 'gx1_prefix_measurement_coordinates_v1',
        'decision': 'TRAIN_AND_CONTROL_COORDINATES_FROZEN_NO_MODEL_MEASUREMENTS',
        'design': db, 'source_index': _binding(index), 'population_rows': n,
        'train_samples_exactly_reused': True, 'control_entry_ids_unchanged': True,
        'all_samples_preserved': True, 'test_data_used': False, 'model_forwards': 0,
        'optimizer_steps': 0, 'fits': 0, 'auxiliary_policies': aux, 'coordinates': bindings}
    rb = _write(tmp_path / 'result.json', result)
    return db, rb, result, arrays


@pytest.mark.parametrize('role', ['train', 'control'])
def test_frozen_probe_keeps_native_order_physical_ids_samples_and_correct_cutoff(tmp_path, role):
    db, rb, result, arrays = _fixture(tmp_path)
    cohort = owner.build_chronological_measurement_cohort(db, rb, role=role)
    assert owner.require_bounded_val_cohort(cohort) == cohort
    assert cohort['parent_entry_row_indices'] == arrays[role]['parent_rows'].tolist()
    assert cohort['entry_row_indices'] == arrays[role]['child_rows'].tolist()
    assert cohort['sampled_state_indices'] == arrays[role]['sampled_state_indices'].tolist()
    assert cohort['reference_cutoff_time_ns'] == int(pd.Timestamp(
        '2026-03-01T00:00Z' if role == 'train' else '2026-06-01T00:00Z').value)
    assert cohort['measurement_only'] is True and cohort['source_split'] == 'train'
    assert cohort['split'] == ('train' if role == 'train' else 'val')
    if role == 'train':
        assert cohort['entry_row_indices'][0] > cohort['entry_row_indices'][-1]


@pytest.mark.parametrize('fault', ['reorder', 'child_mapping', 'outside_lifetime', 'future_support',
                                  'shape', 'wrong_design', 'changed_bytes'])
def test_coordinate_binding_rejects_reselection_or_invalid_support(tmp_path, fault):
    db, rb, result, arrays = _fixture(tmp_path)
    data = arrays['train']
    if fault == 'reorder':
        data = {k: v[::-1] for k, v in data.items()}
    elif fault == 'child_mapping':
        data['child_rows'][0] = 299
    elif fault == 'outside_lifetime':
        data['sampled_state_indices'][0, 0] = 600
    elif fault == 'future_support':
        data['sampled_reference_end_close_ns'][0, 0] = int(pd.Timestamp('2026-03-01T00:00Z').value) + 1
    elif fault == 'shape':
        data['sampled_state_indices'] = data['sampled_state_indices'][:, :3]
    elif fault == 'wrong_design':
        result['design'] = {**db, 'sha256': '0' * 64}
    path = Path(result['coordinates']['train']['path'])
    np.savez(path, **data)
    if fault == 'changed_bytes':
        path.write_bytes(path.read_bytes() + b'changed')
    else:
        result['coordinates']['train'] = _binding(path)
    rb = _write(Path(rb['path']), result)
    with pytest.raises(RuntimeError):
        owner.build_chronological_measurement_cohort(db, rb, role='train')


def _args(tmp_path, role):
    coord_dir = tmp_path / 'coords'
    coord_dir.mkdir()
    db, rb, _, _ = _fixture(coord_dir)
    cohort = owner.build_chronological_measurement_cohort(db, rb, role=role)
    args = _sampled_context(tmp_path)
    args.update(parent_rows=cohort['parent_entry_row_indices'],
        candidate_child_rows=cohort['entry_row_indices'], evaluation_cohort=cohort)
    args.pop('candidate_sampled_state_indices')
    factory = args['candidate_state_factory']
    factory.artifact_file_sha256 = {'random_access_index': cohort['source_index']['sha256']}
    if role == 'train':
        factory.times = pd.date_range('2026-02-20T00:00Z', periods=600, freq='min')
    return args


@pytest.mark.parametrize('role', ['train', 'control'])
def test_existing_entry_measurement_automatically_uses_frozen_role_and_samples(tmp_path, monkeypatch, role):
    args = _args(tmp_path, role)
    calls = []
    def forward(model, seq, snap, **kw):
        return {val.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY: seq,
                'entry_action_q_bps': seq.expand(-1, 3).contiguous()}
    def reference(**kw):
        assert kw['reference_cutoff_time_ns'] == args['evaluation_cohort']['reference_cutoff_time_ns']
        offsets = kw.get('start_state_indices')
        calls.append(offsets)
        q = torch.ones(len(kw['child_rows']), 2)
        rows = [{'entry_row_index': i, 'state_index': offset, 'target_hold_bps': x}
            for i, offset, x in zip(kw['child_rows'], offsets or [0] * len(q), q.tolist())]
        return (rows, {'hold_target_bps': q}) if kw.get('return_reference_targets') else rows
    monkeypatch.setattr(val, '_model_forward_fp32', forward)
    monkeypatch.setattr(val, '_multi_tf_kwargs_from_batch', lambda *a: {})
    monkeypatch.setattr(val, '_bounded_reference_exit_observations', reference)
    monkeypatch.setattr(val, '_new_active_head_epoch_accumulator', lambda: {})
    monkeypatch.setattr(val, '_accumulate_active_head_epoch', lambda *a: None)
    monkeypatch.setattr(val, '_active_head_epoch_diagnostics', lambda *a: ({}, None))
    monkeypatch.setattr(val, 'accumulate_route_diagnostics_v1', lambda *a: None)
    monkeypatch.setattr(val, 'finalize_route_diagnostics_v1', lambda *a: {})
    reps, diag, _ = val._entry_representations(**args)
    assert reps[:, 0].tolist() == args['parent_rows']
    assert len(diag['bounded_exit_sampled_observations']) == 1024
    assert all(x == [7, 4, 7, 1] * 64 for x in calls if x is not None)
    assert all(r['target_q_bps'] == [-3., -3., 0.] for r in diag['bounded_entry_observations'])


@pytest.mark.parametrize('fault', ['sample_override', 'source_index'])
def test_measurement_cannot_replace_frozen_samples_or_source_before_forward(tmp_path, monkeypatch, fault):
    args = _args(tmp_path, 'train')
    if fault == 'sample_override':
        args['candidate_sampled_state_indices'] = [[1, 1, 1, 1]] * 256
    else:
        args['candidate_state_factory'].artifact_file_sha256['random_access_index'] = '0' * 64
    def forbidden(*a, **kw):
        raise AssertionError('mismatched binding reached model')
    monkeypatch.setattr(val, '_model_forward_fp32', forbidden)
    with pytest.raises(RuntimeError, match='CHRONOLOGICAL_MEASUREMENT_'):
        val._entry_representations(**args)


@pytest.mark.parametrize('role', ['train', 'control'])
def test_measurement_cohort_cannot_enter_economic_rollout(tmp_path, role):
    db, rb, _, _ = _fixture(tmp_path)
    cohort = owner.build_chronological_measurement_cohort(db, rb, role=role)
    with pytest.raises(RuntimeError, match='DOES_NOT_AUTHORIZE_ROLLOUT'):
        val.evaluate_bound_full_val_v1(model=None, entry_dataset=None, frame=None,
            state_factory=None, checkpoint_binding={}, parent_coordinate_evidence={},
            val_sequence_audit=tmp_path / 'unused', device=torch.device('cpu'), selected_batch_size=16,
            rollout_progress_path=tmp_path / 'unused-progress', result_path=tmp_path / 'unused-result',
            max_forwards_this_invocation=1, progress_interval_forwards=1,
            compute_guard_max_model_forwards=1, compute_guard_max_materialized_state_views=1,
            compute_guard_max_wall_seconds=1., evaluation_cohort=cohort)
