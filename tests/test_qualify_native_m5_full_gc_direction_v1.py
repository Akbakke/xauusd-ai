"""Small synthetic checks of the one-use joint witness authority."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from gx1.audit import qualify_native_m5_full_gc_direction_v1 as owner
from gx1.contracts import market_state_native_m5_feature_inputs_v1 as ta
from gx1.contracts.market_state_native_m5_targets_v1 import contract as target_contract
from gx1.features.gc_native_m5_features_v1 import native_m5_field_catalog


def _fields():
    fields=native_m5_field_catalog()['fields']
    return [next(field for field in fields if field['name']=='gc_'+tf.lower()+'_observed_volume')
            for tf in ta.TIMEFRAMES]


def _provider(rows=400):
    fields=_fields();names={f['timeframe']:[f['name'],f['availability_field']]for f in fields}
    layout={'timeframes':list(ta.TIMEFRAMES),'field_names':names,
        'family_indices':{tf:{'volume':[0,1]}for tf in ta.TIMEFRAMES}}
    return SimpleNamespace(fields=fields,layout=layout,
        gc_surfaces={tf:{'features':np.ones((rows,2),np.float32)}for tf in ta.TIMEFRAMES},
        ta=SimpleNamespace(tf_tails={tf:np.arange(rows,dtype=np.int64)for tf in ta.TIMEFRAMES}))


def test_plan_role_gate_precedes_receipt_io(monkeypatch):
    monkeypatch.setattr(owner,'read',lambda *_:pytest.fail('receipt IO before role gate'))
    with pytest.raises(ValueError,match='INPUT_ROLES'):owner.input_metadata({})
    with pytest.raises(ValueError,match='EXACT_QUALIFIED_TA006'):
        owner.input_metadata({key:{'path':'bad','sha256':'0'*64}for key in owner.INPUT_KEYS})


def test_parent_requires_genuine_success_and_exact_plan_claim_links(monkeypatch):
    rr,tr,pr,cr=({'path':name,'sha256':name*64}for name in 'rtpc')
    result={'plan':pr,'claim':cr,'status':'DONE','stage':'unit','source_unchanged':True,
        'test_data_used':False,'optimizer_steps':0}
    terminal={'plan':pr,'claim':cr,'result':rr,'stage':'unit','status':'COMPLETE','exit_code':0,'test_data_used':False}
    entries={'r':result,'t':terminal,'p':{'stage':'unit'},'c':{'stage':'unit','plan':pr}}
    seen=[]
    def read(ref):seen.append(ref['path']);return entries[ref['path']]
    monkeypatch.setattr(owner,'read',read)
    assert owner.complete_parent(rr,tr,'DONE','unit')[0]is result
    assert seen==list('rtpc')
    entries['t']['status']='FAILED_CONSUMED'
    with pytest.raises(ValueError,match='COMPLETE_PARENT'):owner.complete_parent(rr,tr,'DONE','unit')


def test_input_only_panel_keeps_fit_axis_and_requires_each_supported_route():
    provider=_provider();fit=np.arange(300,360,dtype=np.int64);before=fit.copy()
    for surface in provider.gc_surfaces.values():surface['features'][:320]=0
    selected,report=owner.choose_witness_panel(provider,fit)
    np.testing.assert_array_equal(selected,np.arange(320,336))
    np.testing.assert_array_equal(fit,before)
    assert report['fit_axis_rows_unchanged']==60 and report['future_outcomes_inspected']is False
    assert len(report['per_route'])==5
    provider.gc_surfaces['D1']['features'][:]=0
    with pytest.raises(ValueError,match='INSUFFICIENT_INPUT_ONLY_GC_PANEL'):
        owner.choose_witness_panel(provider,fit)


def test_real_provider_control_is_actual_observed_local_and_own_tf_coordinate(tmp_path,monkeypatch):
    from tests.test_market_state_native_m5_gc_feature_inputs_v1 import _joint
    provider,packet,_,_=_joint(tmp_path,monkeypatch)
    panel=np.arange(200,216,dtype=np.int64)
    choice=owner.choose_control(provider,packet,panel)
    field=next(f for f in provider.fields if f['name']==choice['field_name'])
    assert not field['binary_value'] and not field['structural_mask_value'] and field['transform']is None
    assert choice['delta']==field['unit_scale'] and choice['new']==choice['old']+choice['delta']
    assert choice['source_available_ns']<=provider.issue_ns[panel].max()
    assert choice['source_available_ns'] in provider.local_gc_bucket_available_ns[field['timeframe']]


class _Backbone(torch.nn.Module):
    def __init__(self,provider,leak=False):
        super().__init__();self.leak=leak
        self.gc_mtf_family_proj=torch.nn.ModuleDict({tf.lower()+'__volume':torch.nn.Linear(2,3,bias=False)
            for tf in ta.TIMEFRAMES})
        self._gc_mtf_specialist_index_tuples={key:(0,1)for key in self.gc_mtf_family_proj}
        f=provider.fields[0]
        self._input_normalization_field_names={'signal':[f['name'],f['availability_field']],
            'ctx_cont':['ctx_cont.'+f['name'],'ctx_cont.'+f['availability_field']]}
    def _normalize_gc_history_surface(self,raw,*,timeframe):
        if self.leak:return raw
        return torch.stack((raw[...,0]*raw[...,1],raw[...,1]),dim=-1)
    def _normalize_input_surface(self,raw,*,surface):
        return self._normalize_gc_history_surface(raw,timeframe='M5')


class _Model(torch.nn.Module):
    def __init__(self,provider,leak=False):
        super().__init__();self.backbone=_Backbone(provider,leak)
        self.open_heads=torch.nn.ModuleDict({tf:torch.nn.Linear(3,3)for tf in ta.TIMEFRAMES})
        self.close_heads=torch.nn.ModuleDict({tf:torch.nn.Linear(3,3)for tf in ta.TIMEFRAMES})
    def forward(self,features):
        hidden=[]
        for tf in ta.TIMEFRAMES:
            raw=features['gc_seq_'+tf.lower()]
            normalized=self.backbone._normalize_gc_history_surface(raw,timeframe=tf)
            hidden.append(self.backbone.gc_mtf_family_proj[tf.lower()+'__volume'](normalized).mean(1))
        return {kind+'_logits':torch.stack([getattr(self,kind+'_heads')[tf](h)
            for tf,h in zip(ta.TIMEFRAMES,hidden)],dim=1)for kind in ('open','close')}


def _batch_truth():
    batch={key:np.array([[[0.,0.],[2.,1.]],[[0.,0.],[3.,1.]]],np.float32)
        for key in ('seq_x','seq_m5','seq_m15','seq_h1','seq_h4','seq_d1',
                    'gc_seq_m5','gc_seq_m15','gc_seq_h1','gc_seq_h4','gc_seq_d1')}
    batch.update(snap_x=batch['seq_x'][:,-1].copy(),ctx_cont=batch['seq_x'][:,-1].copy(),
        ctx_cat=np.zeros((2,1),np.int64))
    step=300_000_000_000;issues=np.array([step,2*step],np.int64);shape=(2,5)
    truth={key:np.ones(shape,bool)for key in ('target_present','observed_previous','mature','eligible','open_known_at_issue')}
    truth.update(open_sign=np.zeros(shape,np.int8),close_sign=np.full(shape,1,np.int8),
        target_row=np.broadcast_to(np.arange(2,dtype=np.int64)[:,None],shape).copy(),
        validity_ns=np.broadcast_to((issues+step)[:,None],shape).copy(),issue_ns=issues)
    return batch,truth,4*step


def test_real_loss_single_autograd_witness_checks_weights_unknowns_and_known_open(monkeypatch):
    provider=_provider();torch.manual_seed(7);model=_Model(provider).eval()
    batch,truth,upper=_batch_truth();before={k:v.clone()for k,v in model.state_dict().items()}
    calls=[];original=torch.autograd.grad
    def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(torch.autograd,'grad',counted)
    out,report=owner.gradients(model,batch,truth,target_contract(),upper,provider)
    assert calls==[1] and report['autograd_calls']==1 and report['unknown_value_gradients_exact_zero']
    assert len(report['gc_projections'])==5
    assert all(item['projection_weight_gradient_l2']>0 for item in report['gc_projections'].values())
    assert all(item['observed_nonBoolean_market_input_gradient_l2']>0 for item in report['gc_projections'].values())
    assert all(report['heads']['open_'+tf]['loss_valid_cells']==0 and
        report['heads']['open_'+tf]['weight_gradient_l2']==0 for tf in ta.TIMEFRAMES)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(before[key],value)for key,value in model.state_dict().items())
    assert out['close_logits'].shape==(2,5,3)


def test_unknown_value_leak_fails_actual_returned_gradient_gate():
    provider=_provider();batch,truth,upper=_batch_truth()
    with pytest.raises(ValueError,match='UNKNOWN_MARKET_VALUE_GRADIENT'):
        owner.gradients(_Model(provider,leak=True).eval(),batch,truth,target_contract(),upper,provider)


def test_fourteen_surface_report_uses_fresh_normalizers_and_finite_guard():
    provider=_provider();model=_Model(provider);batch,_,_=_batch_truth()
    tensors={key:torch.from_numpy(value)for key,value in batch.items()}
    report=owner.normalization_report(model,tensors)
    assert len(report)==14 and report['ctx_cat']['shape']==[2,1]
    tensors['seq_x'][0,0,0]=float('nan')
    with pytest.raises(ValueError,match='NORMALIZED_FINITE:seq_x'):owner.normalization_report(model,tensors)


def test_failure_after_claim_is_consumed_and_preclaim_failure_creates_nothing(tmp_path,monkeypatch):
    root=tmp_path/'one_use';monkeypatch.setattr(owner,'ROOT',root)
    monkeypatch.setattr(owner.common,'firewall_state',lambda:{})
    monkeypatch.setattr(owner.common,'require_capped_cpu_research_producer_execution',
        lambda:{'memory_max':12*1024**3,'memory_high':12*1024**3,'swap':512*1024**2,
                'numerical_threads':1,'cgroup_path':'/sys/fs/cgroup/unit'})
    monkeypatch.setattr(owner.common,'require_capped_lock_ancestry',lambda:{'scope':'/unit'})
    def blocked(*args,**kwargs):raise ValueError('preclaim')
    monkeypatch.setattr(owner,'inspect',blocked)
    with pytest.raises(ValueError,match='preclaim'):owner.run('0'*64)
    assert not root.exists()
    plan={'execution':owner.EXECUTION}
    monkeypatch.setattr(owner,'inspect',lambda *args,**kwargs:(plan,{'path':'plan','sha256':'0'*64},{'path':'policy','sha256':'1'*64}))
    def fail(*args):raise ValueError('synthetic physical failure')
    monkeypatch.setattr(owner,'qualify',fail)
    with pytest.raises(ValueError,match='synthetic physical failure'):owner.run('0'*64)
    terminal=json.loads((root/'TERMINAL.json').read_text());original=(root/'CLAIM.json').read_bytes()
    assert terminal['status']=='FAILED_CONSUMED' and terminal['remaining_invocations']==0
    assert terminal['result']is None and terminal['relaunch_allowed']is False
    with pytest.raises(FileExistsError):owner.run('0'*64)
    assert (root/'CLAIM.json').read_bytes()==original

def test_joint_driver_three_genuine_lanes_and_counterfactual_without_old_weights(tmp_path,monkeypatch):
    from gx1.contracts import market_state_native_m5_gc_feature_inputs_v1 as gc
    from gx1.contracts import market_state_ta_integrated_model_v1 as models
    provider=_provider();raw,truth,upper=_batch_truth()
    provider.source_binding={'contract_sha256':'synthetic','nested':{'immutable':True}}
    provider.input_ready=np.ones(2,bool);provider.issue_ns=truth['issue_ns']
    provider.input_binding=lambda norm:{'norm':norm['contract_sha256'],'source':'synthetic'}
    provider.gather=lambda rows,max_rows:{k:v.copy()for k,v in raw.items()}
    provider.support_report=lambda rows:{'rows':len(rows),'unknown_gc_rows_removed':0}
    population={'window_issue_rows':2,'fit_issue_rows':2}
    norm_population={'issue_rows':2,'axis':'synthetic'}
    norm={'contract_sha256':'fresh_synthetic','fit_population':norm_population,
        'gc_mtf_history':{'synthetic':True},'nested':{'fresh':True}}
    meta={'ta_plan':{'fit_start_ns':1},'qualified_fit_population':population,
        'qualified_normalization_population':norm_population,'fold_upper_ns':upper,
        'native_gc_source_binding':provider.source_binding,'target_contract':target_contract(),
        'seed':7,'architecture':{'synthetic':True}}
    plan={'metadata':meta}
    monkeypatch.setattr(owner,'ROOT',tmp_path)
    monkeypatch.setattr(owner.ta_audit,'load_candidate_provider',lambda plan:(provider,{}))
    monkeypatch.setattr(owner.ta_audit,'target_axis',lambda plan,p:(np.array([0,1],np.int64),truth,population))
    monkeypatch.setattr(gc,'load_native_m5_gc_feature_packet',lambda binding:{'synthetic':True})
    monkeypatch.setattr(gc,'NativeM5GCFeatureProvider',lambda ta,p,b:provider)
    fitted=[]
    def fit(p,*,fit_source_rows,cutoff_ns):
        fitted.append((fit_source_rows.copy(),cutoff_ns));return norm
    monkeypatch.setattr(gc,'fit_native_m5_gc_feature_normalization',fit)
    monkeypatch.setattr(owner,'choose_witness_panel',lambda p,rows:(rows.copy(),{'source_only':True}))
    monkeypatch.setattr(owner,'choose_control',lambda *args:{'field_name':'synthetic',
        'source_available_ns':1,'delta':1.})
    after={k:v.copy()for k,v in raw.items()}
    after['gc_seq_m5'][:,-1,0]+=1
    monkeypatch.setattr(gc,'diagnostic_regather_native_m5_gc_primitive',
        lambda *args,**kwargs:{'before':raw,'after':after,'diagnostic':{
            'purpose':'DIAGNOSTIC_INPUT_PRIMITIVE_COUNTERFACTUAL','production_binding_created':False}})
    models_created=[]
    def build(**kwargs):
        assert kwargs['input_normalization']is norm
        model=_Model(provider).eval();model.research_objective={'contract_sha256':'1'*64}
        models_created.append(model);return model
    monkeypatch.setattr(models,'build_native_bar_direction_model',build)
    report=owner.qualify(plan)
    assert len(models_created)==len(fitted)==1 and report['model_forwards']==3
    assert report['autograd_calls']==1 and report['optimizer_steps']==0
    assert report['old_statistics_or_weights_loaded']is False
    assert report['base_issue_coverage']['rows']==2
    assert report['initial_state_sha256']==report['strict_load_state_sha256']
    assert report['control_max_abs_logit_change']['close_logits']>0
    assert all(parameter.grad is None for parameter in models_created[0].parameters())
    with np.load(report['capture']['path'],allow_pickle=False)as archive:
        assert len([key for key in archive.files if key.startswith('raw_')])==14
    assert json.loads((tmp_path/'DIAGNOSTIC_COUNTERFACTUAL.json').read_text())['production_binding_created']is False


def _fit_recipe(mode='mechanics'):
    return {'optimizer':copy.deepcopy(owner.ADAMW_RECIPE),'gradient_clip_norm':1.,'scheduler':None,
        'seed':20260911,'batch_rows':16,'fit_batches':2 if mode=='mechanics'else 8,
        'score_batches_per_split':1 if mode=='mechanics'else None,
        'checkpoint_every_updates':1,'outer_seconds':1800,'internal_seconds':1770,
        'required_windows_boot_id_greater_than':666}


@pytest.mark.parametrize('change',[{'scheduler':{'kind':'cosine'}},{'gradient_clip_norm':2.},
    {'seed':7},{'optimizer':{'lr':.1}},{'internal_seconds':1700}])
def test_native_fit_recipe_rejects_unbound_numeric_changes(change):
    recipe=_fit_recipe();recipe.update(change)
    with pytest.raises(ValueError):owner.fit_recipe(recipe,'mechanics')


def test_native_main_scores_all_input_rows_with_short_and_unknown_tail():
    recipe=_fit_recipe('main');metadata={'mechanics':{'axis':{
        'TRAIN':{'input_ready_rows':24175},'CONTROL':{'input_ready_rows':5789}}}}
    assert owner.fit_recipe(recipe,'main')==recipe
    assert owner.fit_limits(recipe,'main',metadata)['model_forwards']==8+3746
    rows=np.arange(5789,dtype=np.int64);batches=owner.all_score_batches(rows,16)
    np.testing.assert_array_equal(np.concatenate(batches),rows)
    assert len(batches[-1])==13 and batches[-1][-1]==5788
    recipe['score_batches_per_split']=1
    with pytest.raises(ValueError,match='FIT_SCORE_SCOPE'):owner.fit_recipe(recipe,'main')


def test_fit_batch_schedule_is_source_only_nonoverlapping_and_stable():
    rows=1000+np.arange(160,dtype=np.int64)*3
    batches=owner.selected_batches(rows,3)
    assert [int(np.searchsorted(rows,value[0]))for value in batches]==[0,64,144]
    assert len(np.unique(np.concatenate(batches)))==48
    np.testing.assert_array_equal(batches[1],owner.selected_batches(rows,3)[1])


def _cpu_rng(monkeypatch):
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',lambda value:None)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:0)


def _training_model(provider):
    class Stochastic(_Model):
        def __init__(self):
            super().__init__(provider);self.dropout=torch.nn.Dropout(.35)
            self._native_fit_counts={'model_constructions':1,'model_forwards':0,'backward_calls':0,'optimizer_steps':0}
        def forward(self,features):
            self._native_fit_counts['model_forwards']+=1
            return {key:self.dropout(value)for key,value in super().forward(features).items()}
    return Stochastic()


def _training_unit(monkeypatch):
    _cpu_rng(monkeypatch);provider=_provider();batch,truth,_=_batch_truth()
    provider.gather=lambda rows,max_rows:{key:value.copy()for key,value in batch.items()}
    provider.input_binding=lambda norm:{'normalization_sha256':norm['contract_sha256']}
    rows=np.array([0,1],np.int64)
    cohort={'rows':rows,'truth':truth,'fit_rows':rows,'fit_truth':truth,'upper_ns':owner.FIT_END_NS}
    context={'provider':provider,'normalization':{'contract_sha256':'synthetic'},
        'normalization_ref':{'path':'synthetic_norm','sha256':'1'*64},
        'prior_ref':{'path':'synthetic_prior','sha256':'2'*64}}
    plan={'recipe':_fit_recipe(),'metadata':{'target_contract':target_contract(),'architecture':{'synthetic':True}}}
    def fresh():
        model=_training_model(provider)
        kwargs={**owner.ADAMW_RECIPE,'betas':tuple(owner.ADAMW_RECIPE['betas'])}
        return model,torch.optim.AdamW(model.parameters(),**kwargs)
    return provider,cohort,context,plan,fresh


def test_native_dropout_checkpoint_restores_adamw_rng_and_next_update_exactly(tmp_path,monkeypatch):
    import random
    _,cohort,context,plan,fresh=_training_unit(monkeypatch)
    random.seed(7);np.random.seed(7);torch.manual_seed(7)
    model,optimizer=fresh();rows=cohort['fit_rows'];plan_ref={'path':'synthetic_plan','sha256':'3'*64}
    first=owner.fit_step(model,optimizer,context['provider'],cohort,rows,plan)
    assert np.isfinite(first['loss']) and optimizer.state
    artifact=owner.publish_checkpoint(tmp_path/'RESUME.pt',
        owner.fit_checkpoint(model,optimizer,context,plan,plan_ref,1))
    second=owner.fit_step(model,optimizer,context['provider'],cohort,rows,plan)
    expected=owner.fit_state_proof(model,optimizer,2,second)
    cold,cold_optimizer=fresh()
    loaded=torch.load(artifact['path'],map_location='cpu',weights_only=True)
    assert owner.restore_fit_checkpoint(cold,cold_optimizer,context,plan,plan_ref,loaded)==1
    repeated=owner.fit_step(cold,cold_optimizer,context['provider'],cohort,rows,plan)
    assert owner.fit_state_proof(cold,cold_optimizer,2,repeated)==expected
    assert all(parameter.grad is None for parameter in cold.parameters())


def test_native_restore_rejects_wrong_optimizer_shape_before_parameter_copy(tmp_path,monkeypatch):
    _,cohort,context,plan,fresh=_training_unit(monkeypatch)
    model,optimizer=fresh();plan_ref={'path':'synthetic_plan','sha256':'3'*64}
    owner.fit_step(model,optimizer,context['provider'],cohort,cohort['fit_rows'],plan)
    checkpoint=owner.fit_checkpoint(model,optimizer,context,plan,plan_ref,1)
    checkpoint=copy.deepcopy(checkpoint)
    next(iter(checkpoint['optimizer_state']['state'].values()))['exp_avg']=torch.zeros(999)
    cold,cold_optimizer=fresh();before=owner.state_sha(cold.state_dict())
    with pytest.raises(ValueError,match='FIT_OPTIMIZER_STATE'):
        owner.restore_fit_checkpoint(cold,cold_optimizer,context,plan,plan_ref,checkpoint)
    assert owner.state_sha(cold.state_dict())==before


def test_fit_axis_keeps_control_unknown_tail_and_per_head_cutoffs(tmp_path):
    step=300_000_000_000;train=owner.FIT_START_NS+np.arange(20,dtype=np.int64)*step
    control=np.r_[owner.FIT_END_NS+np.arange(19,dtype=np.int64)*step,owner.CONTROL_END_NS-step]
    issues=np.r_[train,control];source=1000+np.arange(40,dtype=np.int64);shape=(40,5)
    truth={key:np.ones(shape,bool)for key in ('target_present','observed_previous','mature','eligible','open_known_at_issue')}
    validity=np.broadcast_to((issues+step)[:,None],shape).copy()
    # One higher-TF label crosses TRAIN cutoff while other heads remain mature.
    validity[0,4]=owner.FIT_END_NS
    truth.update(issue_ns=issues,target_row=np.broadcast_to(np.arange(40)[:,None],shape).copy(),
        validity_ns=validity,open_sign=np.ones(shape,np.int8),close_sign=np.full(shape,-1,np.int8))
    archive=tmp_path/'targets.npz';np.savez(archive,source_row=source,issue_ns=issues,
        **{'future_'+key:value for key,value in truth.items()})
    provider=SimpleNamespace(issue_ns=issues,source_row=source,input_ready=np.ones(40,bool))
    cohorts,report=owner.fit_axes(provider,owner.ref(archive))
    assert report['TRAIN']['base_issue_rows']==20 and report['CONTROL']['input_ready_rows']==20
    assert cohorts['TRAIN']['truth']['eligible'][0,:].tolist()==[True,True,True,True,False]
    assert not cohorts['CONTROL']['truth']['eligible'][-1].any()
    assert cohorts['CONTROL']['rows'][-1]==source[-1]
    np.testing.assert_array_equal(cohorts['CONTROL']['truth']['close_sign'][-1],np.zeros(5,np.int8))


def test_native_fit_scoring_uses_global_unique_targets_and_preserves_rng(monkeypatch):
    from gx1.contracts.market_state_ta_integrated_objective_v1 import fit_native_bar_train_prior
    _cpu_rng(monkeypatch);provider=_provider();step=300_000_000_000;shape=(32,5)
    issues=np.arange(1,33,dtype=np.int64)*step
    truth={key:np.ones(shape,bool)for key in ('target_present','observed_previous','mature','eligible')}
    truth.update(open_known_at_issue=np.zeros(shape,bool),open_sign=np.zeros(shape,np.int8),
        close_sign=np.ones(shape,np.int8),issue_ns=issues,target_row=np.zeros(shape,np.int64),
        validity_ns=np.full(shape,33*step,np.int64))
    rows=np.arange(32,dtype=np.int64);cohort={'rows':rows,'truth':truth,'upper_ns':34*step}
    raw,_,_=_batch_truth()
    provider.gather=lambda rows,max_rows:{key:np.repeat(value[:1],len(rows),axis=0)for key,value in raw.items()}
    prior=fit_native_bar_train_prior(truth=truth,fold_upper_ns=34*step,
        target_contract_sha256=target_contract()['contract_sha256'])
    context={'provider':provider,'cohorts':{'TRAIN':cohort,'CONTROL':cohort},'prior':prior}
    panels={name:owner.all_score_batches(rows,16)for name in context['cohorts']}
    plan={'recipe':_fit_recipe(),'metadata':{'target_contract':target_contract()}}
    model=_training_model(provider).train();before=owner.state_sha(owner.training_rng())
    scores,capture=owner.fit_score(model,context,panels,plan)
    assert model.training and owner.state_sha(owner.training_rng())==before
    for name in ('TRAIN','CONTROL'):
        assert scores[name]['model']['per_task']['close']['D1']['count']==32
        assert scores[name]['model']['per_task']['close']['D1']['unique_target_rows']==1
        assert len(capture[name+'__source_row'])==32

def test_fit_physical_boot_gate_reports_stale_preflight_and_rejects_claim(monkeypatch):
    plan={'mode':'mechanics','recipe':_fit_recipe()}
    monkeypatch.setattr(owner,'physical_windows_boot_id',lambda:666)
    assert owner.fit_boot_state(plan,admitted=False)['fresh']is False
    with pytest.raises(ValueError,match='FIT_FRESH_PHYSICAL_BOOT'):
        owner.fit_boot_state(plan,admitted=True)
    monkeypatch.setattr(owner,'physical_windows_boot_id',lambda:667)
    assert owner.fit_boot_state(plan,admitted=True)['fresh']
    plan={'mode':'main','recipe':_fit_recipe('main'),'metadata':{'mechanics':{'physical_windows_boot_id':667}}}
    assert owner.fit_boot_state(plan,admitted=True)['actual_windows_boot_id']==667


def test_cold_resume_proof_requires_child_zero_forbidden_io():
    proof={'bitexact':True,'parent_pid':123,'clean_process_io_firewall':{
        'denied_attempts':0,'denied_operations':{}}}
    assert owner.require_cold_resume_proof(proof,parent_pid=123)is proof
    for firewall in (None,{'denied_attempts':1,'denied_operations':{'open':1}},
                     {'denied_attempts':0,'denied_operations':{'stat':1}}):
        bad={**proof,'clean_process_io_firewall':firewall}
        with pytest.raises(ValueError,match='FIT_COLD_PROOF'):
            owner.require_cold_resume_proof(bad,parent_pid=123)
