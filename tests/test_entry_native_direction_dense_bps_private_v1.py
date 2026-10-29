"""Synthetic fail-closed metadata and initialization checks; no real native execution."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
from gx1.contracts import entry_native_direction_dense_v1 as dense
from gx1.contracts import entry_native_direction_bps_v1 as bps
from gx1.contracts import gx1_capped_execution_v1 as cap
from gx1.scripts import run_entry_native_direction_dense_qualification_v1 as qualification
from gx1.scripts import run_entry_native_direction_research_window_v1 as runner
from gx1.scripts.run_unified_exit_random_access_full_train_v1 import _dynamic_m5_technical_tree_sha256 as tree_sha
from tests import test_entry_native_direction_research_v1 as model_fixture
from tests.test_entry_native_direction_dense_bps_v1 import _scale
from tests.test_entry_native_direction_research_campaign_v1 import (
    fixture as research_fixture, candidate_price_fixture, reseal, write)


def _recipe_fixture(f, monkeypatch):
    recipe=candidate_price_fixture(f,monkeypatch,'price_only'); runtime=f['runtime'];repo=f['repo']
    recipe['run_scope']=prefix.M5_DENSE_BPS_PRIVATE_SCOPE;recipe['admission_stage']=authority.M5_DENSE_BPS_STAGE
    recipe['max_backwards']=5
    bps_path='gx1/contracts/entry_native_direction_bps_v1.py'
    recipe['source_owners'][bps_path]=write(repo/bps_path,{'synthetic':'bps-owner'})
    numerical=recipe['numerical_recipe'];numerical['dense_signed_bps_supervision']=True
    numerical.pop('contract_sha256');numerical['contract_sha256']=hashlib.sha256(json.dumps(numerical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    audit_value=json.loads(Path(recipe['decimal_input_audit']['path']).read_text())
    prior_audit=write(runtime/'prior-sign-audit.json',{'synthetic':True})
    recipe['decimal_target_refs']['audit_result']=prior_audit
    audit_value['inputs']=recipe['decimal_target_refs']
    audit=write(Path(recipe['decimal_input_audit']['path']),audit_value)
    recipe['decimal_input_audit']=audit;monkeypatch.setattr(authority,'DECIMAL_INPUT_AUDIT',audit)
    scale=_scale();scale.update(prefix_rows=recipe['fold']['prefix_rows'],cutoff_ns=recipe['fold']['train_cutoff_ns'],target_source=recipe['dense_target'])
    scale.pop('contract_sha256');scale['contract_sha256']=bps._sha(scale)
    scale_ref=write(runtime/'dense-scale.json',scale)
    scale_result=write(runtime/'dense-scale-result.json',{'status':'COMPLETE_TRAIN_PREFIX_SCALE_ONLY','scale':scale_ref,
        'model_forwards':0,'optimizer_steps':0,'test_data_used':False,'independent_counts_and_fsum_with_error_bounds':True})
    terminal=write(runtime/'dense-scale-terminal.json',{'status':'COMPLETE','result':scale_result,'scale':scale_ref})
    count=write(runtime/'dense-scale-count.json',{'status':'PASS','scale':scale_ref,'prior_decimal_sign_audit':prior_audit,
        'all_mature_observed_counts_equal':True,'horizons_checked':60,'test_data_used':False})
    initial=write(runtime/'initial-predictions.npz',{'synthetic':'bytes'})
    initial_report=write(runtime/'initial-report.json',{'role':'INITIAL_TRAIN','fold':recipe['fold'],'test_data_used':False,
        'predictions':initial,'preservation':{'model':'a'*64,'all_buffers':'b'*64,'rng':'c'*64}})
    refs={'dense_bps_scale':scale_ref,'dense_bps_scale_result':scale_result,'dense_bps_scale_terminal':terminal,
        'dense_bps_count_review':count,'dense_bps_initial_predictions':initial,'dense_bps_initial_report':initial_report}
    constants={'M5_DENSE_BPS_SCALE':scale_ref,'M5_DENSE_BPS_SCALE_RESULT':scale_result,'M5_DENSE_BPS_SCALE_TERMINAL':terminal,
        'M5_DENSE_BPS_COUNT_REVIEW':count,'M5_DENSE_BPS_BASELINE_INITIAL':initial,'M5_DENSE_BPS_BASELINE_REPORT':initial_report}
    for key,value in constants.items():monkeypatch.setattr(authority,key,value)
    recipe.update(refs);reseal(recipe,'recipe_sha256');return recipe


def test_private_dense_bps_recipe_requires_qualified_scale_and_exact_scope(research_fixture, monkeypatch):
    f=research_fixture;recipe=_recipe_fixture(f,monkeypatch)
    checked=authority.require_research_launch_recipe(write(f['runtime']/'dense-bps-private.json',recipe),source_repo=f['repo'],source_commit='a'*40)
    assert checked['run_scope']==prefix.M5_DENSE_BPS_PRIVATE_SCOPE
    assert checked['candidate_mode']=='price_only' and checked['numerical_recipe']['dense_signed_bps_supervision'] is True
    assert checked['qualification_weights_for_main_fit'] is False and checked['test_data_used'] is False
    for name in ('scale','stage','mode','initial','review'):
        value=copy.deepcopy(recipe)
        if name=='scale':value['dense_bps_scale']['sha256']='0'*64
        elif name=='stage':value['admission_stage']+='-other'
        elif name=='mode':value['candidate_mode']='price_and_m1'
        elif name=='initial':value['dense_bps_initial_predictions']['sha256']='0'*64
        else:value['dense_bps_count_review']['sha256']='0'*64
        reseal(value,'recipe_sha256')
        with pytest.raises(RuntimeError):
            authority.require_research_launch_recipe(write(f['runtime']/('bad-'+name+'.json'),value),source_repo=f['repo'],source_commit='a'*40)


def test_new_private_scope_dispatches_only_to_dense_qualification(monkeypatch, tmp_path):
    monkeypatch.setattr(cap,'require_guarded_cuda_trainer_execution',lambda:{'synthetic':'limits'})
    monkeypatch.setattr(authority,'require_native_direction_research_context',lambda *a:({},{},{},{},{'run_scope':prefix.M5_DENSE_BPS_PRIVATE_SCOPE}))
    seen=[]
    monkeypatch.setattr(qualification,'run_dense_qualification',lambda **kwargs:seen.append(kwargs) or {'decision':'synthetic'})
    p=tmp_path/'window.json';result=runner.run_native_direction_research_window(policy_path=p,policy_file_sha256='a'*64,progress_path=tmp_path/'progress.json')
    assert result=={'decision':'synthetic'} and len(seen)==1 and seen[0]['policy_path']==p


def test_private_initial_parity_checks_all_shared_weights_buffers_rng_and_logits(monkeypatch,tmp_path):
    monkeypatch.setattr(model_fixture.EntryV10CtxHybridTransformer,'forward',model_fixture._native_forward_spy)
    torch.manual_seed(420);baseline=dense.DenseNativeEntryDirectionResearch(model_fixture._native_shell())
    baseline_rng=torch.get_rng_state().clone()
    torch.manual_seed(420);candidate=dense.DenseNativeEntryDirectionResearch(model_fixture._native_shell(),dense_bps_scale=_scale())
    assert torch.equal(torch.get_rng_state(),baseline_rng)
    data={'parent_rows':np.arange(256,dtype=np.int64),'bar_start_ns':np.arange(256,dtype=np.int64),
        'decision_ns':np.arange(256,dtype=np.int64),'logit_up':np.zeros((256,60),np.float32)}
    path=tmp_path/'initial.npz';np.savez(path,**data);initial={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    report={'preservation':{'model':tree_sha(baseline.state_dict()),'all_buffers':tree_sha(dict(baseline.named_buffers())),'rng':'a'*64}}
    measured={'arrays':data,'state_preservation':{'rng':'a'*64}}
    qualification.require_dense_bps_initial_parity(measured,report,initial,candidate,np=np,tree_sha=tree_sha)
    with torch.no_grad():candidate.dense_aux_head.bias[0]+=1
    with pytest.raises(RuntimeError,match='BPS_INITIAL_SHARED_MODEL_BUFFER_RNG_PARITY'):
        qualification.require_dense_bps_initial_parity(measured,report,initial,candidate,np=np,tree_sha=tree_sha)
    with torch.no_grad():candidate.dense_aux_head.bias[0]-=1;candidate.dense_bps_head.bias[0]=1
    with pytest.raises(RuntimeError,match='BPS_INITIAL_ZERO_HEAD'):
        qualification.require_dense_bps_initial_parity(measured,report,initial,candidate,np=np,tree_sha=tree_sha)
