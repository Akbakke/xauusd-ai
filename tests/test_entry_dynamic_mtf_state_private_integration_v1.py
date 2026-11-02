"""Synthetic private state gates and bounded control flow; no native inputs/CUDA."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix_authority
from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.scripts import run_entry_native_direction_dense_qualification_v1 as worker


def _write(path, value):
    path.write_text(json.dumps(value, allow_nan=False))
    return {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def _counts():
    return {'model_constructions':3, 'model_construction_returns':3, 'optimizer_constructions':3,
        'forward_calls':21, 'forward_returns':21, 'backward_calls':4, 'backward_returns':4,
        'optimizer_step_calls':4, 'optimizer_step_returns':4, 'empty_batches':0,
        'staged_state_load_calls':3, 'staged_state_load_returns':3,
        'resume_state_load_calls':1, 'resume_state_load_returns':1, 'forbidden_legacy_calls':0,
        'forward_kind_calls':{'INITIAL_TRAIN':16, 'INITIAL_PARITY_GRAD':1, 'TRAIN':4}}


def _gradients():
    groups = ('state_query_projection', 'state_initial_head', 'state_transition_head',
        'entry_q_joint', 'family_context_global', 'specialist_proj.example', 'mtf_family_proj.example',
        'mtf_feature_context_gate.example', 'gc_mtf_family_proj.example',
        'gc_mtf_feature_context_gate.example', 'specialist_ctx_cont_proj.example')
    return dict.fromkeys(groups, 1)


def test_private_counts_reject_extra_work_and_missing_resume():
    authority.require_dynamic_state_private_counts(_counts())
    for key, value in (('forward_calls',22), ('optimizer_step_returns',5),
            ('resume_state_load_returns',0), ('forbidden_legacy_calls',1)):
        invalid = _counts()
        invalid[key] = value
        with pytest.raises(RuntimeError, match='STATE_EXACT_PRIVATE_WORK'):
            authority.require_dynamic_state_private_counts(invalid)
    invalid = _counts()
    invalid['forward_kind_calls']['INITIAL_PARITY_GRAD'] = True
    with pytest.raises(RuntimeError, match='STATE_EXACT_PRIVATE_WORK'):
        authority.require_dynamic_state_private_counts(invalid)


def test_gradient_gate_checks_both_logical_trajectories():
    steps = [{'trajectory':name, 'logical_step':number, 'gradient_nonzero_counts':_gradients()}
        for name,number in (('reference',1),('reference',2),('interrupted',1),('resumed',2))]
    worker.require_dynamic_state_gradient_contact(steps)
    for item in steps[2:]:
        item['gradient_nonzero_counts']['state_transition_head'] = 0
    with pytest.raises(RuntimeError, match='STATE_GRADIENT_NOT_OBSERVED:state_transition_head'):
        worker.require_dynamic_state_gradient_contact(steps)


@pytest.mark.parametrize('failure', [None, 'missing_d1_flat', 'later_inspection', 'unlocked_audit'])
def test_train_audit_gate_cannot_admit_unsupported_or_later_selected_labels(tmp_path, monkeypatch, failure):
    from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target_owner
    from gx1.contracts import entry_dynamic_mtf_state_report_v1 as report_owner
    leads = list(target_owner.default_trial_lead_minutes())
    prior_counts = np.ones((5,3),np.int64)
    if failure == 'missing_d1_flat':
        prior_counts[4,1] = 0
    transitions = np.ones((5,3,3),np.int64)
    prior = {'schema_version':report_owner.SCHEMA_VERSION, 'timeframes':list(report_owner.TIMEFRAMES),
        'class_names':list(report_owner.STATE_NAMES), 'lead_minutes':leads, 'parents':47602,
        'counts':prior_counts.tolist(), 'probabilities':(prior_counts/prior_counts.sum(axis=1,keepdims=True)).tolist(),
        'transition_counts':transitions.tolist(), 'transition_probabilities':(transitions/3).tolist()}
    target = _write(tmp_path/'SERIES.json', {'synthetic':True})
    sources = {name:{'path':str(tmp_path/name), 'sha256':'a'*64} for name in ('m1','m5_atr')}
    recipe = {'fold':authority.BPS_FOLD, 'target_manifest':target, 'target_sources':sources,
        'numerical_recipe':{'research_objective_sha256':'b'*64}}
    audit = {'schema_version':'gx1_dynamic_mtf_state_label_audit_v1', 'support_status':'PASS',
        'blockers':[], 'target_manifest':target, 'target_sources':sources, 'fold':authority.BPS_FOLD,
        'target_contract_sha256':'c'*64, 'objective_sha256':'b'*64, 'lead_minutes':leads, 'prior':prior,
        'model_forwards':0, 'optimizer_steps':0, 'test_data_used':False, 'later_metrics_inspected':False}
    if failure == 'later_inspection':
        audit['later_metrics_inspected'] = True
    if failure == 'unlocked_audit':
        audit['support_status'] = 'BLOCKED'
    recipe['label_audit'] = _write(tmp_path/'VERIFIED_RESULT.json',audit)
    monkeypatch.setattr(target_owner,'read_dynamic_mtf_state_series_metadata',lambda *args,**kwargs:
        {'metadata':{'parents':64890, 'definition':{'contract_sha256':'c'*64}}})
    if failure:
        with pytest.raises(RuntimeError):
            prefix_authority.require_dynamic_state_target_binding(recipe)
    else:
        _, checked = prefix_authority.require_dynamic_state_target_binding(recipe)
        assert checked['prior'] == prior


def test_dynamic_private_dispatch_preserves_guard_and_context_boundary(monkeypatch, tmp_path):
    from gx1.contracts import gx1_capped_execution_v1 as capped
    calls = []
    monkeypatch.setattr(capped,'require_guarded_cuda_trainer_execution',lambda:calls.append('guard') or {})
    launch = {'run_scope':authority.DYNAMIC_STATE_PRIVATE_SCOPE}
    monkeypatch.setattr(authority,'require_native_direction_research_context',
        lambda *args:calls.append('context') or ({},{},{},None,launch))
    monkeypatch.setattr(worker,'_run_dynamic_state_qualification',lambda **kwargs:calls.append('state') or {'synthetic':True})
    assert worker.run_dense_qualification(policy_path=tmp_path/'policy',policy_file_sha256='0'*64,
        progress_path=tmp_path/'progress') == {'synthetic':True}
    assert calls == ['guard','context','state']


@pytest.mark.parametrize('failure',[None,'prepare','resume_drift','gradient'])
def test_dynamic_private_bounded_control_flow_and_fail_closed_evidence(tmp_path, monkeypatch, failure):
    import torch
    from gx1.contracts import entry_model_native_train_recipe_v1 as recipes
    from gx1.contracts import entry_native_direction_research_state_v1 as state_owner
    from gx1.features import entry_specialist_feature_groups_v1 as groups
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as prefix
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    def ref(name):
        return {'path':str(tmp_path/name), 'sha256':'a'*64}
    numerical = {'dynamic_state_supervision':True, 'batch_size':16, 'max_updates':2}
    monkeypatch.setattr(recipes,'require_native_direction_research_recipe',lambda value:value)
    parents = np.arange(47602,dtype=np.int64)
    ordered = prefix._fixed_order(parents,47602,prefix.SEED,np)
    selected = np.sort(ordered[:32]).tolist()
    launch = {'run_scope':authority.DYNAMIC_STATE_PRIVATE_SCOPE, 'full_feature_profile':True,
        'numerical_recipe':numerical, 'execution_binding_path':str(tmp_path/'binding'),
        'runtime_root':str(tmp_path/'runtime'), 'fold':authority.BPS_FOLD,
        'target_manifest':ref('SERIES.json'), 'target_sources':{}, 'label_audit':ref('audit'),
        'qualification_parent_rows':selected, 'max_forwards':21,
        'max_checkpoint_bytes':1073741824, 'max_published_json_bytes':33554432}
    window = {'recipe':ref('recipe'), 'output_directory':str(tmp_path/'output'),
        'campaign_cursor_path':str(tmp_path/'CURSOR.json')}
    context = {'info':{'metadata':{}}, 'ancestor_plan':{'payloads':{},'budget':{'max_rchar_bytes':137438953472}},
        'physical':SimpleNamespace(release_pages=lambda *args:None), 'ancestor':None,
        'fitted':{'gc_mtf_history':{}}}
    monkeypatch.setattr(common,'_hash_file',lambda path:ref('binding'))
    monkeypatch.setattr(common,'_read',lambda *args:{'plan':ref('plan')})
    monkeypatch.setattr(prefix,'_launch_metadata',lambda *args:context)
    monkeypatch.setattr(common,'_freeze',lambda *args:None)
    monkeypatch.setattr(common,'_install_read_guard',lambda *args:[])
    monkeypatch.setattr(common,'_deny_legacy_work',lambda *args:None)
    monkeypatch.setattr(common,'_io',lambda:{'rchar':0})
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    profile = {'name':'synthetic', 'contract_sha256':'c'*64}
    view = {'entry_execution_profile':profile}
    def prepare(*args):
        if failure == 'prepare':
            raise RuntimeError('synthetic prepare block')
        return {'arrays':{'parent_rows':parents}, 'gc':{'layout':{}}},{},view,None,ordered
    monkeypatch.setattr(prefix,'_prepare',prepare)
    monkeypatch.setattr(groups,'compile_full_gc_mtf_entry_execution_profile',lambda *args:profile)
    monkeypatch.setattr(prefix,'_prefix_prior',lambda *args:{'synthetic':True})
    def batch(prepared,targets,view,selected,np):
        return {'witness':{'parent_rows':selected.tolist()}}, {
            'observed':np.ones((16,5,1),bool), 'maturity_ns':np.zeros((16,5,1),np.int64)}
    monkeypatch.setattr(prefix,'_batch',batch)
    monkeypatch.setattr(prefix,'_device_batch',lambda raw,target,torch,device:({'parents':raw['witness']['parent_rows']},{}))
    monkeypatch.setattr(worker,'_dynamic_state_input_support',lambda raw,*args,**kwargs:raw['witness'])
    class Model:
        feature_view_identity = {'synthetic':True}
        def __init__(self):
            self.value = 0
        def named_parameters(self):
            return []
    def fresh(context,view,recipe,device,counters,**kwargs):
        for key in ('model_constructions','model_construction_returns','optimizer_constructions'):
            counters[key] += 1
        return Model(),object(),{}
    monkeypatch.setattr(common,'_fresh',fresh)
    def capture(model,optimizer,**kwargs):
        state = {'binding':{}, 'model_state':{'value':model.value}, 'buffer_state':{},
            'optimizer_state':{}, 'rng_state':{}, 'module_training':{},
            'cursor':kwargs['cursor'], 'optimizer_updates':kwargs['optimizer_updates']}
        return {**state,'contract_sha256':campaign.canonical_sha256(state)}
    monkeypatch.setattr(state_owner,'capture_native_direction_research_state',capture)
    saved = {}
    def save(path,state,**kwargs):
        for key in ('staged_state_load_calls','staged_state_load_returns'):
            kwargs['counters'][key] += 1
        result = ref(Path(path).name)
        saved[result['path']] = copy.deepcopy(state)
        return result
    def load(reference,**kwargs):
        for key in ('resume_state_load_calls','resume_state_load_returns'):
            kwargs['counters'][key] += 1
        return copy.deepcopy(saved[reference['path']])
    monkeypatch.setattr(common,'_save_state',save)
    monkeypatch.setattr(common,'_load_state',load)
    def restore(state,model,optimizer,**kwargs):
        model.value = state['model_state']['value'] + int(failure == 'resume_drift')
        return {key:state[key] for key in ('cursor','optimizer_updates')}
    monkeypatch.setattr(state_owner,'restore_native_direction_research_state',restore)
    def evaluate(*args,**kwargs):
        counters = kwargs['counters']
        counters['forward_calls'] += 17
        counters['forward_returns'] += 17
        counters['forward_kind_calls'].update(INITIAL_TRAIN=16, INITIAL_PARITY_GRAD=1)
        return {'initial_parity_passed':True,
            'state_artifact':{'arrays':{'probabilities':{'shape':[256,5,1440,3]}}}},256
    monkeypatch.setattr(prefix,'_evaluate',evaluate)
    monkeypatch.setattr(prefix,'_publish_measurement',lambda *args,**kwargs:ref('INITIAL_TRAIN.json'))
    def train(model,optimizer,*args,**kwargs):
        counters = kwargs['counters']
        for key in ('forward_calls','forward_returns','backward_calls','backward_returns',
                'optimizer_step_calls','optimizer_step_returns'):
            counters[key] += 1
        counters['forward_kind_calls']['TRAIN'] = counters['forward_kind_calls'].get('TRAIN',0)+1
        model.value += 1
        evidence = _gradients()
        if failure == 'gradient':
            evidence['state_transition_head'] = 0
        return {'gradient_nonzero_counts':evidence,'per_timeframe_counts':[{'eligible_cells':1}]*5},True
    monkeypatch.setattr(prefix,'_train_batch',train)
    published = {}
    def publish(path,value,*args):
        published[Path(path).name] = value
        return ref(Path(path).name)
    monkeypatch.setattr(common,'_json',publish)
    monkeypatch.setattr(authority,'build_research_cursor',lambda **kwargs:{'outcome':kwargs['outcome']})
    monkeypatch.setattr(authority,'build_research_progress',lambda **kwargs:{'outcome':kwargs['outcome']})
    monkeypatch.setattr(authority,'require_research_progress',lambda *args,**kwargs:None)
    arguments = dict(window=window,plan={'plan_sha256':'d'*64},invocation={'batch_size':16},
        launch=launch,limits={},policy_path=tmp_path/'policy',policy_file_sha256='a'*64,
        progress_path=tmp_path/'PROGRESS.json')
    if failure:
        with pytest.raises(RuntimeError,match='NATIVE_DYNAMIC_STATE_QUALIFICATION_FAILED'):
            worker._run_dynamic_state_qualification(**arguments)
        assert published['RESULT.json']['decision'] == 'FAILED'
        if failure == 'prepare':
            assert published['RESULT.json']['counts']['model_constructions'] == 0
        if failure == 'resume_drift':
            assert published['RESULT.json']['bit_exact_resume'] is False
    else:
        assert worker._run_dynamic_state_qualification(**arguments) == {'outcome':'COMPLETE'}
        result = published['RESULT.json']
        assert result['decision'] == 'PASS_DYNAMIC_STATE_NATIVE_MECHANICS'
        authority.require_dynamic_state_private_counts(result['counts'])
        assert result['private_weights_for_main_fit'] is False
        assert result['bit_exact_resume'] is True
        assert [row for batch in result['batch_input_support'] for row in batch['parent_rows']] == selected
