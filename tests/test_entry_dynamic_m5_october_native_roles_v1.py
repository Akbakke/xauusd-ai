from copy import deepcopy

import pytest

from gx1.contracts import entry_dynamic_m5_october_native_roles_v1 as roles


def _case(monkeypatch):
    source = {'m1': {'path': '/synthetic/m1', 'sha256': '1' * 64}}
    axes = {'fit': {'rows': 29849}, 'held_out': {'rows': 6323}}
    control = {'contract_sha256': 'c' * 64, 'prefix_normalization': {},
               'research_objective': {'contract_sha256': '2' * 64},
               'october_observer_axes': axes, 'native_endpoint_loss': True}
    candidate = {**control, 'contract_sha256': 'd' * 64, 'confirmed_run_age_input': True}
    fit = {'metadata': {'parents': 29849, 'cutoff_ns': 1727740800000000000,
                        'test_data_used': False, 'source_refs': source}}
    score = {'metadata': {'parents': 36172, 'cutoff_ns': 1730419200000000000,
                          'test_data_used': False, 'source_refs': source}}
    batch = {'schema_version': 'gx1_dynamic_m5_october_native_batch_binding_plan_v1',
             'control_view': {'path': '/synthetic/control', 'sha256': '3' * 64},
             'candidate_view': {'path': '/synthetic/candidate', 'sha256': '4' * 64},
             'october_series': {'path': '/synthetic/score', 'sha256': '5' * 64},
             'causal_inputs': {'path': '/synthetic/causal', 'sha256': '6' * 64},
             'observer_axes': {'path': '/synthetic/axes', 'sha256': '7' * 64},
             'fit_prefix': {'rows': 29849, 'cutoff_ns': 1727740800000000000},
             'evaluation_axis': {'cutoff_ns': 1730419200000000000},
             'target_source_refs': source}
    reference = {'schema_version': 'gx1_dynamic_m5_october_fit_references_plan_v1',
                 'control_view': batch['control_view'], 'october_series': batch['october_series'],
                 'fit_series': {'path': '/synthetic/fit', 'sha256': '8' * 64},
                 'fit_cutoff_ns': 1727740800000000000,
                 'development_cutoff_ns': 1730419200000000000,
                 'target_source_refs': source}
    objects = {roles.BATCH_PLAN['path']: batch, roles.REFERENCE_PLAN['path']: reference,
               batch['control_view']['path']: control, batch['candidate_view']['path']: candidate,
               reference['fit_series']['path']: fit, reference['october_series']['path']: score,
               batch['observer_axes']['path']: axes}
    monkeypatch.setattr(roles, '_read', lambda ref: deepcopy(objects[ref['path']]))
    monkeypatch.setattr(roles.identity, 'require_checkpoint_view',
                        lambda item: 'control' if item['contract_sha256'] == control['contract_sha256'] else 'candidate')
    return batch, reference, objects


def _launch(batch, reference, *, scope, arm, target):
    return {'run_scope': scope, 'arm': arm, 'input_role': roles.SCOPE_ROLES[scope],
            'candidate_plan': roles.BATCH_PLAN, 'reference_plan': roles.REFERENCE_PLAN,
            'feature_view': batch[arm + '_view'], 'target_manifest': target,
            'causal_inputs': batch['causal_inputs'], 'observer_axes': batch['observer_axes']}


def test_fit_cannot_open_development_target(monkeypatch):
    batch, reference, _ = _case(monkeypatch)
    fit = _launch(batch, reference, scope=roles.FIT_SCOPE, arm='candidate', target=reference['fit_series'])
    assert roles.require_role_binding(fit)['role'] == 'fit'
    fit['target_manifest'] = reference['october_series']
    with pytest.raises(RuntimeError, match='EXACT_ROLE_REFS'):
        roles.require_role_binding(fit)
    fit['input_role'] = 'score'
    with pytest.raises(RuntimeError, match='SCOPE'):
        roles.require_role_binding(fit)


def test_score_is_read_only_role_with_exact_view_and_target(monkeypatch):
    batch, reference, objects = _case(monkeypatch)
    score = _launch(batch, reference, scope=roles.SCORE_SCOPE, arm='control', target=reference['october_series'])
    assert roles.require_role_binding(score)['role'] == 'score'
    score['feature_view'] = batch['candidate_view']
    with pytest.raises(RuntimeError, match='EXACT_ROLE_REFS'):
        roles.require_role_binding(score)
    score['feature_view'] = batch['control_view']
    objects[reference['october_series']['path']]['metadata']['test_data_used'] = True
    with pytest.raises(RuntimeError, match='ROLE_METADATA'):
        roles.require_role_binding(score)


def test_existing_worker_dispatches_october_metadata_and_prepare(monkeypatch):
    from gx1.audit import dynamic_m5_october_native_inputs as adapter
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as worker
    launch = {'run_scope': roles.PRIVATE_SCOPE}
    sentinel = object()
    monkeypatch.setattr(adapter, 'october_native_metadata',
                        lambda incoming, common: sentinel if incoming is launch else None)
    monkeypatch.setattr(adapter, 'prepare_october_native',
                        lambda context, incoming, np: (sentinel, context, incoming, np))
    assert worker._launch_metadata(launch, object()) is sentinel
    assert worker._prepare_dynamic_state(sentinel, launch, 'numpy') == (
        sentinel, sentinel, launch, 'numpy')
