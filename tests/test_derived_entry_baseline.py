"""Explicit causal Entry baseline; original native evidence stays immutable."""
import copy
import json
from pathlib import Path
import pytest
import torch
from tests.test_native_prefix_learning_measurement import learning_scope, initial_scope, prefix_scope, scope
from tests.test_native_prefix_recipe import native, _write, _bind


@pytest.mark.parametrize('fault',[None,'unbound','control_scope','target','prediction','liquidation','source','policy','origin','exit_target'])
def test_derived_entry_baseline_binding_preserves_origin_and_exact_targets(learning_scope,tmp_path,fault):
    policy,recipe,_,seal=learning_scope
    ab=recipe['chronological_learning_measurement'];audit=json.loads(Path(ab['path']).read_text())
    result=json.loads(Path(audit['result']['path']).read_text());original=result['observations']['train']
    obs=json.loads(Path(original['path']).read_text());p=recipe['exit_reference_policy']
    entry=[{'entry_row_index':i,'parent_entry_row_index':i,'predicted_q_bps':[1.,2.,0.],
        'target_q_bps':[9.,-1.,0.],'target_valid':[True]*3} for i in range(256)]
    anchor=[{'entry_row_index':i,'state_index':0,'target_hold_bps':[10.,-10.],
        'reference_policy_sha256':p['policy_sha256']} for i in range(256)]
    obs['diagnostics']={'bounded_entry_observations':entry,'bounded_exit_anchor_observations':anchor}
    result['observations']['train']=_write(Path(original['path']),obs)
    audit['result']=_write(Path(audit['result']['path']),result)
    recipe['chronological_learning_measurement']=_write(Path(ab['path']),audit)
    policy['chronological_learning_run']['chronological_learning_measurement']=recipe['chronological_learning_measurement']
    target=(torch.tensor([10.,-10.])*(119/120)-1).tolist()+[0.]
    baseline={'schema_version':'gx1_derived_causal_entry_train_baseline_v1','role':'train',
        'source_observation':result['observations']['train'],'initial_measurement_audit':recipe['chronological_learning_measurement'],
        'cohort':obs['cohort'],'model_state_sha256':obs['model_state_sha256'],'target_model_state_sha256':obs['target_model_state_sha256'],
        'target_semantics':'observed_reference_anchor_V_mu_without_hindsight_action','reference_policy':p,
        'canonical_first_liquidation_bps':[[-1.,-1.] for _ in entry],
        'entry_observations':[{**row,'target_q_bps':target.copy()} for row in entry]}
    source={}
    for name in ['gx1/contracts/unified_exit_economic_step_provider_v1.py','gx1/contracts/unified_exit_reference_policy_v1.py',
                 'gx1/scripts/run_unified_exit_random_access_val_v1.py','gx1/contracts/unified_exit_random_access_training_v1.py']:
        path=tmp_path/name;path.parent.mkdir(exist_ok=True,parents=True);path.write_text('synthetic source');source[name]=_bind(path)
    derived={'schema_version':'gx1_derived_causal_entry_train_baseline_result_v1',
        'decision':'CANONICAL_DERIVED_TRAIN_BASELINE_READY_NO_NEW_LEARNING_MEASURED',
        'source_observations':{'initial':result['observations']['train']},'initial_measurement_audit':recipe['chronological_learning_measurement'],
        'old_entry_targets_exactly_reconstructed':True,'new_model_forwards':0,'optimizer_steps':0,
        'test_data_used':False,'control_observations_accessed':False,'source_bindings':source,
        'data_bindings':{k:recipe['files'][k] for k in ['random_access_root','economics_readiness','train_cost_authority']}}
    if fault=='target':baseline['entry_observations'][0]['target_q_bps'][0]+=1
    elif fault=='prediction':baseline['entry_observations'][0]['predicted_q_bps']=[7,2,0]
    elif fault=='liquidation':baseline['canonical_first_liquidation_bps'][0][1]=-2
    elif fault=='source':next(iter(source.values()))['sha256']='c'*64
    elif fault=='policy':baseline['reference_policy']={**p,'hold_probability_numerator':118}
    elif fault=='origin':baseline['source_observation']={'path':'wrong','sha256':'a'*64}
    elif fault=='exit_target':
        obs['diagnostics']['bounded_exit_anchor_observations'][0]['target_hold_bps']=[11,-10]
        # A changed bound original is rejected, not accepted via the new baseline.
        _write(Path(original['path']),obs)
    derived['derived_baseline']=_write(tmp_path/'derived-train.json',baseline)
    rb=_write(tmp_path/'derived-result.json',derived);recipe['chronological_entry_baseline']=rb
    recipe['chronological_train_only_measurement']=True
    policy['chronological_learning_run']['chronological_train_only_measurement']=True
    if fault!='unbound':policy['chronological_learning_run']['chronological_entry_baseline']=rb
    if fault=='control_scope':recipe['chronological_train_only_measurement']=False
    seal()
    if fault:
        with pytest.raises(RuntimeError):native.require_native_run_scope(recipe,invocation_number=1)
    else:
        # TRAIN-only preparation must not read CONTROL outcomes.
        Path(result['observations']['control']['path']).unlink()
        assert native.require_native_run_scope(recipe,invocation_number=1)==256
        loaded=native.require_chronological_learning_measurement(recipe)
        assert loaded['entry_baseline']==baseline and loaded['entry_baseline_result']==rb
        assert loaded['initial_measurement']==result and loaded['artifacts']['initial_measurement_audit']==recipe['chronological_learning_measurement']
