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
        'seed':20260911,'batch_rows':16,'score_batch_rows':64,'fit_batches':2 if mode=='mechanics'else 96,
        'score_batches_per_split':1 if mode=='mechanics'else None,
        'checkpoint_every_updates':1 if mode=='mechanics'else 32,'outer_seconds':1800,'internal_seconds':1770,
        'required_windows_boot_id_greater_than':666}


@pytest.mark.parametrize('change',[{'scheduler':{'kind':'cosine'}},{'gradient_clip_norm':2.},
    {'seed':7},{'optimizer':{'lr':.1}},{'internal_seconds':1700},{'score_batch_rows':16}])
def test_native_fit_recipe_rejects_unbound_numeric_changes(change):
    recipe=_fit_recipe();recipe.update(change)
    with pytest.raises(ValueError):owner.fit_recipe(recipe,'mechanics')


def test_native_main_scores_all_input_rows_with_short_and_unknown_tail():
    recipe=_fit_recipe('main');metadata=_rank_context()[0]
    assert owner.fit_recipe(recipe,'main')==recipe
    assert owner.fit_limits(recipe,'main',metadata)['model_forwards']==160
    rows=np.arange(5789,dtype=np.int64);batches=owner.all_score_batches(rows,64)
    np.testing.assert_array_equal(np.concatenate(batches),rows)
    assert len(batches[-1])==29 and batches[-1][-1]==5788
    train=owner.all_score_batches(np.arange(24175,dtype=np.int64),64)
    assert len(train[-1])==47 and len(train)==378 and len(batches)==91
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


def test_consumed_mechanics001_cli_is_rejected_before_any_receipt_read(monkeypatch):
    monkeypatch.setattr(owner,'read',lambda *_:pytest.fail('consumed plan receipt read'))
    args=SimpleNamespace(plan=str(owner.REPO/'configs/research/NATIVE_M5_FULL_GC_FIT_MECHANICS_20261119_001.json'),
        plan_sha256='0'*64,device='cuda',out_dir='/unused',preflight=False)
    with pytest.raises(ValueError,match='FIT_CLI_SCOPE'):owner.fit_cli(args)


def test_mechanics002_parent_admits_only_original001_statistics_and_no_weights(monkeypatch):
    root='/home/andre2/GX1_RUNS/NATIVE_M5_FULL_GC_FIT_MECHANICS_20261119_001'
    rr={'path':root+'/RESULT.json','sha256':'1a5a3a0ec76400d552df59d2346154bf7442adfa7dba61154a54af6d2d6ebafe'}
    tr={'path':root+'/TERMINAL.json','sha256':'24454ce94896ecc4ccbcf9331a5cba499bf637ef59b43b4ca99479fd5dd533bc'}
    pr={'path':'old_plan','sha256':'1'*64};cr={'path':'old_claim','sha256':'2'*64}
    inputs={'joint_result':{'path':'joint_result','sha256':'3'*64},
        'joint_terminal':{'path':'joint_terminal','sha256':'4'*64},'stats_result':rr,'stats_terminal':tr}
    metadata={'fit_start_ns':owner.FIT_START_NS,'fit_end_ns':owner.FIT_END_NS,
        'control_end_ns':owner.CONTROL_END_NS,'native_gc_source_binding':{'synthetic':True},
        'target_contract':target_contract(),'prior_contract':{'synthetic':True},'architecture':{'synthetic':True}}
    stage='native_m5_full_gc_fit_mechanics_20261119_001'
    report={'cold_resume':{'bitexact':True},'normalization':{'path':root+'/NORMALIZATION.json','sha256':'5'*64},
        'prior':{'path':root+'/TRAIN_PRIOR.json','sha256':'6'*64},'axis':{'synthetic':True},
        'normalization_population':{'synthetic':True},'initial_state_sha256':'7'*64,
        'initial_checkpoint':{'path':root+'/NEVER_READ.pt','sha256':'8'*64}}
    result={'stage':stage,'status':'COMPLETE_NATIVE_M5_FULL_GC_FIT_MECHANICS',
        'plan':pr,'claim':cr,'source_unchanged':True,'test_data_used':False,
        'physical_windows_boot_id':667,'report':report}
    terminal={'stage':stage,'status':'COMPLETE','exit_code':0,'result':rr,'plan':pr,'claim':cr,'test_data_used':False}
    plan={'stage':stage,'metadata':metadata,'inputs':{k:inputs[k]for k in ('joint_result','joint_terminal')},
        'source_base_commit':'9'*40}
    values={rr['path']:result,tr['path']:terminal,pr['path']:plan,cr['path']:{'stage':stage,'plan':pr}}
    seen=[]
    def read(ref):
        assert not ref['path'].endswith('.pt')
        seen.append(ref['path']);return values[ref['path']]
    monkeypatch.setattr(owner,'read',read)
    parent=owner._fit_statistics_parent(inputs,metadata,'mechanics')
    assert parent['role']=='frozen_TRAIN_statistics_only_no_weights'
    assert parent['normalization']==report['normalization'] and parent['prior']==report['prior']
    assert 'checkpoint'not in json.dumps(parent).lower()
    assert seen==[rr['path'],tr['path'],pr['path'],cr['path']]
    bad={**inputs,'stats_result':{**rr,'sha256':'0'*64}}
    with pytest.raises(ValueError,match='FIT_EXACT_STATISTICS_PARENT'):
        owner._fit_statistics_parent(bad,metadata,'mechanics')


def test_main_rejects_old001_as_active_gpu_qualification_before_read(monkeypatch):
    monkeypatch.setattr(owner,'read',lambda *_:pytest.fail('wrong qualification read'))
    inputs={'mechanics_result':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_FULL_GC_FIT_MECHANICS_20261119_001/RESULT.json'},
        'mechanics_terminal':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_FULL_GC_FIT_MECHANICS_20261119_001/TERMINAL.json'}}
    with pytest.raises(ValueError,match='FIT_EXACT_MECHANICS_PARENT'):
        owner._fit_statistics_parent(inputs,{},'main')


def test_mechanics002_reuses_norm_and_prior_without_fitter_copy_or_weight_io(tmp_path,monkeypatch):
    from gx1.contracts import market_state_native_m5_gc_feature_inputs_v1 as gc
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    rows=np.arange(16,dtype=np.int64);axis={'TRAIN':{'input_ready_rows':16},'CONTROL':{'input_ready_rows':16}}
    population={'cutoff_ns':owner.FIT_END_NS,'issue_rows':16,'selected_source_rows_sha256':owner.digest(rows)}
    norm_ref={'path':'original001/NORMALIZATION.json','sha256':'1'*64}
    prior_ref={'path':'original001/TRAIN_PRIOR.json','sha256':'2'*64}
    norm={'fit_population':population};prior={'synthetic_prior':True}
    base=SimpleNamespace(input_ready=np.ones(32,bool))
    provider=SimpleNamespace(input_ready=base.input_ready.copy(),input_binding=lambda value:{'synthetic':True})
    parent={'axis':axis,'normalization':norm_ref,'prior':prior_ref,'normalization_population':population}
    plan={'mode':'mechanics','metadata':{'ta_plan':{},'native_gc_source_binding':{},
        'dependencies':{'target_tape':{'path':'synthetic_targets','sha256':'3'*64}},
        'statistics_parent':parent}}
    monkeypatch.setattr(owner.ta_audit,'load_candidate_provider',lambda plan:(base,{}))
    monkeypatch.setattr(gc,'load_native_m5_gc_feature_packet',lambda binding:{'synthetic':True})
    monkeypatch.setattr(gc,'NativeM5GCFeatureProvider',lambda *args:provider)
    monkeypatch.setattr(owner,'fit_axes',lambda *args:({'TRAIN':{'fit_rows':rows}},axis))
    monkeypatch.setattr(gc,'fit_native_m5_gc_feature_normalization',lambda *a,**kw:pytest.fail('norm refit'))
    monkeypatch.setattr(objective,'fit_native_bar_train_prior',lambda *a,**kw:pytest.fail('prior refit'))
    monkeypatch.setattr(objective,'require_native_bar_train_prior',lambda value:value)
    monkeypatch.setattr(owner,'publish',lambda *args:pytest.fail('stats copy/new artifact'))
    seen=[]
    def read(ref):
        seen.append(ref)
        if ref==norm_ref:return norm
        if ref==prior_ref:return prior
        pytest.fail('unexpected/weight IO')
    monkeypatch.setattr(owner,'read',read)
    context=owner.prepare_native_fit(plan,tmp_path)
    assert context['normalization_ref']==norm_ref and context['prior_ref']==prior_ref
    assert context['normalization']is norm and context['prior']is prior
    assert seen==[norm_ref,prior_ref]
    with pytest.raises(ValueError,match='FIT_EXACT_REUSED_STATISTICS'):
        owner.prepare_native_fit(plan,tmp_path,reload_normalization={'path':'copied_norm','sha256':'4'*64})


def _rank_context():
    step=300_000_000_000
    class ClockOnly(dict):
        def __getitem__(self,key):
            assert key=='issue_ns','outcome read during source selection'
            return super().__getitem__(key)
    cohorts={};axis={}
    for name,n,start,offset in [('TRAIN',24175,owner.FIT_START_NS,1_000_000),
                              ('CONTROL',5789,owner.FIT_END_NS,1_100_000)]:
        rows=offset+np.arange(n,dtype=np.int64);issues=start+np.arange(n,dtype=np.int64)*step
        cohorts[name]={'rows':rows,'truth':ClockOnly(issue_ns=issues)}
        axis[name]={'input_ready_rows':n,'input_not_ready_rows':0,
            'source_rows_sha256':owner.digest(rows),'issue_ns_sha256':owner.digest(issues)}
    metadata={'mechanics':{'result':{'path':'mechanics002/RESULT.json','sha256':'1'*64},'axis':axis}}
    return metadata,{'cohorts':cohorts}


def test_small_source_rank_manifest_freezes_endpoints_without_labels_or_io(monkeypatch):
    metadata,context=_rank_context()
    monkeypatch.setattr(owner,'read',lambda *_:pytest.fail('selection performed receipt/label IO'))
    manifest=owner.small_cohort_manifest(metadata)
    plan={'metadata':metadata,'cohort_selection':manifest}
    selected,report=owner.selected_small_cohorts(plan,context)
    for name,count,blocks,batch in [('FIT',1536,96,16),('CONTROL',512,8,64)]:
        pool=context['cohorts']['TRAIN'if name=='FIT'else'CONTROL']['rows']
        rows=selected[name];ranks=owner.spread_rank_blocks(len(pool),blocks,batch)
        assert len(rows)==count and len(np.unique(rows))==count and np.all(np.diff(rows)>0)
        assert rows[0]==pool[0] and rows[-1]==pool[-1]
        np.testing.assert_array_equal(rows,pool[ranks])
        assert report[name]['first_and_last_pool_rows_included']
        assert owner.digest(ranks)==manifest['cohorts'][name]['selected_rank_sha256']
    bad=copy.deepcopy(plan);bad['cohort_selection']['cohorts']['CONTROL']['selected_rank_sha256']='0'*64
    with pytest.raises(ValueError,match='FIT_FROZEN_SOURCE_SELECTION'):
        owner.selected_small_cohorts(bad,context)


def test_small_source_selection_rejects_runtime_pool_or_overlapping_blocks():
    metadata,context=_rank_context();plan={'metadata':metadata,'cohort_selection':owner.small_cohort_manifest(metadata)}
    context['cohorts']['CONTROL']['rows'][100]+=1
    with pytest.raises(ValueError,match='FIT_RUNTIME_SOURCE_POOL'):owner.selected_small_cohorts(plan,context)
    with pytest.raises(ValueError,match='FIT_SOURCE_RANK_POPULATION'):owner.spread_rank_blocks(127,2,64)


def test_consumed_short_fit001_cli_is_rejected_before_any_receipt_read(monkeypatch):
    monkeypatch.setattr(owner,'read',lambda *_:pytest.fail('consumed main receipt read'))
    args=SimpleNamespace(plan=str(owner.REPO/'configs/research/NATIVE_M5_FULL_GC_SHORT_FIT_20261119_001.json'),
        plan_sha256='0'*64,device='cuda',out_dir='/unused',preflight=False)
    with pytest.raises(ValueError,match='FIT_CLI_SCOPE'):owner.fit_cli(args)


def test_small_source_bridge_allows_only_orchestration_and_exact_four_paths(tmp_path,monkeypatch):
    import hashlib
    driver='gx1/audit/qualify_native_m5_full_gc_direction_v1.py'
    paths=(driver,'tests/test_qualify_native_m5_full_gc_direction_v1.py',
        'scripts/gx1_capped_run.sh','tests/test_gx1_capped_run_contract.py','gx1/contracts/model_math.py')
    original="import os\nFIT_SCOPES={'main':'001'}\nADAMW_RECIPE={'lr':1e-4}\ndef fit_step():return 1\ndef fit_physical():return 1\n"
    changed=original.replace("'001'","'002'").replace("def fit_physical():return 1","def fit_physical():return 96")
    hashes={}
    for path in paths:
        target=tmp_path/path;target.parent.mkdir(parents=True,exist_ok=True)
        body=original if path==driver else'unchanged\n'
        target.write_text(body);hashes[path]=hashlib.sha256(body.encode()).hexdigest()
    (tmp_path/driver).write_text(changed)
    (tmp_path/'scripts/gx1_capped_run.sh').write_text('new exact scope/deadline\n')
    monkeypatch.setattr(owner,'REPO',tmp_path);monkeypatch.setattr(owner,'FIT_SOURCE_PATHS',paths)
    monkeypatch.setattr(owner.subprocess,'check_output',lambda *a,**kw:original.encode())
    parent={'source_owners':hashes,'source_base_commit':'a'*40}
    proof=owner.fit_orchestration_source_bridge(parent)
    assert proof['all_other_source_owners_exact'] and len(proof['allowed_changed_source_paths'])==4
    for wrong in (changed.replace('def fit_step():return 1','def fit_step():return 2'),
                  changed.replace("1e-4","2e-4"),changed.replace('import os','import sys')):
        (tmp_path/driver).write_text(wrong)
        with pytest.raises(ValueError,match='FIT_MECHANICS_NUMERICAL_AST'):
            owner.fit_orchestration_source_bridge(parent)
    (tmp_path/driver).write_text(changed)
    (tmp_path/'gx1/contracts/model_math.py').write_text('wrong numerical source\n')
    with pytest.raises(ValueError,match='FIT_MECHANICS_NUMERICAL_SOURCE'):
        owner.fit_orchestration_source_bridge(parent)


def test_small_matched_prior_uses_exact_fit1536_and_keeps_pool_normalization(tmp_path):
    metadata,context=_rank_context();n=24175;shape=(n,5);step=300_000_000_000
    train=context['cohorts']['TRAIN'];issues=dict.__getitem__(train['truth'],'issue_ns')
    truth={key:np.ones(shape,bool)for key in ('target_present','observed_previous','mature','eligible')}
    truth.update(open_known_at_issue=np.zeros(shape,bool),issue_ns=issues,
        target_row=np.broadcast_to(np.arange(n,dtype=np.int64)[:,None],shape).copy(),
        validity_ns=np.broadcast_to((issues+step)[:,None],shape).copy(),
        open_sign=np.zeros(shape,np.int8),
        close_sign=np.broadcast_to((np.arange(n,dtype=np.int64)%3-1).astype(np.int8)[:,None],shape).copy())
    train.update(truth=truth,fit_rows=train['rows'],fit_truth=truth)
    plan={'metadata':{**metadata,'target_contract':target_contract()},
        'cohort_selection':owner.small_cohort_manifest(metadata)}
    selected,_=owner.selected_small_cohorts(plan,context)
    immutable_norm={'same_frozen_pool_contract':True};context['normalization']=immutable_norm
    context['prior']={'old_pool_prior':True};context['prior_ref']={'path':'old_pool_prior','sha256':'2'*64}
    artifact=owner.prepare_matched_prior(context,selected['FIT'],plan,tmp_path)
    prior=json.loads((tmp_path/'MATCHED_FIT_PRIOR.json').read_text())
    assert context['normalization']is immutable_norm and context['prior_ref']==artifact
    assert prior['fit_population']['issue_rows']==1536
    ranks=owner.spread_rank_blocks(n,96,16)
    expected=np.bincount(truth['close_sign'][ranks,0].astype(np.int64)+1,minlength=3)
    assert prior['per_task']['close']['M5']['class_counts']==expected.tolist()
    assert prior['fit_population']['issue_ns_sha256']==owner.digest(issues[ranks])


def _ta_recipe(mode='ta_mechanics',passes=2):
    recipe=_fit_recipe()
    recipe.update(passes=passes,fit_batches=2 if mode=='ta_mechanics'else passes*96,
        checkpoint_every_updates=1 if mode=='ta_mechanics'else 96,score_batches_per_split=None)
    return recipe


def test_ta_recipe_has_explicit_passes_exact_limits_and_boundary_cursor():
    metadata,context=_rank_context();metadata={'source_pool':metadata['mechanics']}
    recipe=_ta_recipe();assert owner.fit_recipe(recipe,'ta_mechanics')==recipe
    assert owner.fit_limits(recipe,'ta_mechanics',metadata)=={
        'model_constructions':2,'model_forwards':67,'backward_calls':3,'optimizer_steps':3}
    selected,_=owner.selected_small_cohorts({'metadata':metadata,
        'cohort_selection':owner.small_cohort_manifest(metadata)},context)
    plan={'mode':'ta_mechanics','recipe':recipe}
    batches=owner._ta_training_batches(selected,plan)
    assert len(batches)==2
    np.testing.assert_array_equal(batches[0],selected['FIT'][:16])
    np.testing.assert_array_equal(batches[0],batches[1])
    assert owner._ta_schedule_position(plan,1)=={'completed_updates':1,'next_pass':1,
        'next_batch_in_pass':0,'batches_per_pass':1,'declared_passes':2}
    main=_ta_recipe('ta_main',passes=4);assert owner.fit_recipe(main,'ta_main')==main
    assert owner.fit_limits(main,'ta_main',metadata)=={
        'model_constructions':1,'model_forwards':544,'backward_calls':384,'optimizer_steps':384}
    mainplan={'mode':'ta_main','recipe':main};schedule=owner._ta_training_batches(selected,mainplan)
    assert len(schedule)==384
    for at in range(0,len(schedule),96):
        np.testing.assert_array_equal(np.concatenate(schedule[at:at+96]),selected['FIT'])
    assert owner._ta_schedule_position(mainplan,96)['next_pass']==1
    assert owner._ta_schedule_position(mainplan,96)['next_batch_in_pass']==0
    for change in ({'passes':0},{'passes':3},{'score_batches_per_split':1},{'checkpoint_every_updates':2}):
        bad=copy.deepcopy(recipe);bad.update(change)
        with pytest.raises(ValueError):owner.fit_recipe(bad,'ta_mechanics')
    assert owner.FIT_SCOPES['ta_mechanics'][1].name=='NATIVE_M5_TA_PRIOR_MECHANICS_20261119_001.json'
    assert owner.FIT_SCOPES['ta_main'][1].name=='NATIVE_M5_TA_PRIOR_LEARNING_CURVE_20261119_001.json'


def test_ta_role_gates_reject_gc_inputs_before_receipt_or_packet_io(monkeypatch):
    monkeypatch.setattr(owner,'read',lambda *_:pytest.fail('unadmitted metadata IO'))
    for mode in ('ta_mechanics','ta_main'):
        with pytest.raises(ValueError,match='TA_FIT_INPUT_ROLES'):
            owner.ta_fit_metadata({'joint_result':{},'gc_result':{}},mode)
    with pytest.raises(ValueError,match='TA_FIT_EXACT_PARENT'):
        owner.ta_fit_metadata({key:{}for key in ('ta_result','ta_terminal','matched_prior_reference')},'ta_mechanics')


def test_ta_source_rank_axis_reads_only_hash_checked_clock_arrays(tmp_path,monkeypatch):
    step=300_000_000_000
    starts=np.array([owner.FIT_START_NS-step,owner.FIT_START_NS,
        owner.FIT_END_NS-step,owner.FIT_END_NS,owner.CONTROL_END_NS-step],np.int64)
    retained=np.arange(5,dtype=np.int64)
    arrays={'bar_start_ns':starts,'retained_issue_rows':retained}
    path=tmp_path/'clock.npz';np.savez(path,**arrays,signal=np.array([999.],np.float32))
    snapshot={'arrays':{key:{'dtype':value.dtype.str,'shape':list(value.shape),'sha256':owner.digest(value)}
        for key,value in arrays.items()}}
    plan={'inputs':{'candidate_archive':owner.ref(path),'candidate_snapshot':{'path':'snapshot','sha256':'1'*64}}}
    monkeypatch.setattr(owner,'read',lambda ref:snapshot if ref['path']=='snapshot'else pytest.fail('nonclock read'))
    original=owner.np.load
    class ClockOnly:
        def __enter__(self):self.archive=original(path,allow_pickle=False);return self
        def __exit__(self,*args):self.archive.close()
        def __getitem__(self,key):
            assert key in arrays,'feature/label decode before source selection'
            return self.archive[key]
    monkeypatch.setattr(owner.np,'load',lambda *args,**kwargs:ClockOnly())
    axis=owner._ta_pool_axis(plan)
    assert axis['TRAIN']['input_ready_rows']==2 and axis['CONTROL']['input_ready_rows']==2
    assert axis['TRAIN']['source_rows_sha256']==owner.digest(np.array([0,1],np.int64))
    assert axis['CONTROL']['source_rows_sha256']==owner.digest(np.array([2,3],np.int64))
    snapshot['arrays']['bar_start_ns']['sha256']='0'*64
    with pytest.raises(ValueError,match='TA_CLOCK_BYTES'):owner._ta_pool_axis(plan)


def _ta_prior_context():
    from gx1.contracts.market_state_ta_integrated_objective_v1 import fit_native_bar_train_prior
    metadata,context=_rank_context();metadata={'source_pool':metadata['mechanics'],
        'target_contract':target_contract(),'ta_plan':{},'dependencies':{'target_tape':{}}}
    train=context['cohorts']['TRAIN'];n=len(train['rows']);shape=(n,5);step=300_000_000_000
    issues=dict.__getitem__(train['truth'],'issue_ns')
    truth={key:np.ones(shape,bool)for key in ('target_present','observed_previous','mature','eligible')}
    truth.update(issue_ns=issues,open_known_at_issue=np.zeros(shape,bool),
        target_row=np.broadcast_to(np.arange(n,dtype=np.int64)[:,None],shape).copy(),
        validity_ns=np.broadcast_to((issues+step)[:,None],shape).copy(),
        open_sign=np.zeros(shape,np.int8),close_sign=np.zeros(shape,np.int8))
    truth['open_known_at_issue'][:,-1]=True
    train.update(truth=truth,fit_rows=train['rows'],fit_truth=truth,upper_ns=owner.FIT_END_NS)
    plan={'mode':'ta_mechanics','recipe':_ta_recipe(),'metadata':metadata,
        'cohort_selection':owner.small_cohort_manifest(metadata),'inputs':{}}
    selected,_=owner.selected_small_cohorts(plan,context)
    prior=fit_native_bar_train_prior(truth=owner.slice_truth(train,selected['FIT'],fit=True),
        fold_upper_ns=owner.FIT_END_NS,target_contract_sha256=target_contract()['contract_sha256'])
    return plan,context,prior


def test_ta_prepare_fits_only_train_prefix_and_reuses_exact_stats_for_cold_and_main(tmp_path,monkeypatch):
    from gx1.contracts import market_state_native_m5_gc_feature_inputs_v1 as gc
    plan,context,prior=_ta_prior_context();axis=copy.deepcopy(plan['metadata']['source_pool']['axis'])
    # Real target-axis construction normally supplies these complete reports.
    context['cohorts']['CONTROL']['fit_rows']=context['cohorts']['CONTROL']['rows']
    issue=np.zeros(int(context['cohorts']['CONTROL']['rows'][-1])+1,np.int64)
    for cohort in context['cohorts'].values():issue[cohort['rows']]=dict.__getitem__(cohort['truth'],'issue_ns')
    observed=[]
    provider=SimpleNamespace(issue_ns=issue,input_binding=lambda value:observed.append(value['fit_population']['cutoff_ns']))
    monkeypatch.setattr(owner.ta_audit,'load_candidate_provider',lambda _: (provider,{'numeric_arrays_reused':True}))
    monkeypatch.setattr(owner,'fit_axes',lambda *args:(context['cohorts'],axis))
    monkeypatch.setattr(gc,'load_native_m5_gc_feature_packet',lambda *_:pytest.fail('TA opened GC packet'))
    monkeypatch.setattr(gc,'NativeM5GCFeatureProvider',lambda *_:pytest.fail('TA built GC provider'))
    calls=[]
    def fit(p,*,fit_source_rows,cutoff_ns):
        assert p is provider and cutoff_ns==owner.FIT_END_NS
        np.testing.assert_array_equal(fit_source_rows,context['cohorts']['TRAIN']['rows'])
        calls.append(len(fit_source_rows))
        return {'fit_population':{'cutoff_ns':cutoff_ns,'issue_rows':len(fit_source_rows),
            'selected_source_rows_sha256':owner.digest(fit_source_rows)}}
    monkeypatch.setattr(ta,'fit_native_m5_feature_normalization',fit)
    reference=tmp_path/'label_only_prior.json';reference_ref=owner.publish(reference,prior)
    plan['inputs']['matched_prior_reference']=reference_ref
    first=owner.prepare_native_fit(plan,tmp_path)
    assert calls==[24175]and first['normalization_fits']==first['prior_fits']==1
    assert first['prior']==prior and first['prior_ref']['sha256']==reference_ref['sha256']
    stage,path,_=owner.FIT_SCOPES['ta_mechanics']
    monkeypatch.setitem(owner.FIT_SCOPES,'ta_mechanics',(stage,path,tmp_path))
    monkeypatch.setattr(ta,'fit_native_m5_feature_normalization',lambda *a,**k:pytest.fail('stats refitted'))
    cold=owner.prepare_native_fit(plan,tmp_path/'cold',reload_normalization=first['normalization_ref'],reload_prior=first['prior_ref'])
    assert cold['normalization_fits']==cold['prior_fits']==0
    assert cold['normalization']==first['normalization']and cold['prior']==first['prior']
    main=copy.deepcopy(plan);main['mode']='ta_main';main['recipe']=_ta_recipe('ta_main',4)
    main['metadata']['mechanics']={'axis':axis,'normalization':first['normalization_ref'],
        'prior':first['prior_ref'],'normalization_population':first['normalization']['fit_population']}
    reused=owner.prepare_native_fit(main,tmp_path/'main')
    assert reused['normalization_fits']==reused['prior_fits']==0
    assert observed==[owner.FIT_END_NS]*3
    wrong=copy.deepcopy(main);wrong['metadata']['mechanics']['normalization_population']['cutoff_ns']=owner.CONTROL_END_NS
    with pytest.raises(ValueError,match='TA_FIT_IMMUTABLE_STATS'):owner.prepare_native_fit(wrong,tmp_path/'bad')


def test_ta_checkpoint_resume_restores_pass_boundary_rng_optimizer_and_rejects_bad_cursor(tmp_path,monkeypatch):
    provider,cohort,context,plan,fresh=_training_unit(monkeypatch)
    plan.update(mode='ta_mechanics',recipe=_ta_recipe())
    plan_ref={'path':'TAplan','sha256':'3'*64}
    torch.manual_seed(19);model,opt=fresh();model.research_objective={'unit_prior_initialization_sha256':'4'*64}
    owner.fit_step(model,opt,provider,cohort,cohort['rows'],plan)
    checkpoint=copy.deepcopy(owner.fit_checkpoint(model,opt,context,plan,plan_ref,1))
    assert checkpoint['schedule_position']['next_pass']==1 and checkpoint['schedule_position']['next_batch_in_pass']==0
    assert checkpoint['schema_version']=='gx1_native_m5_ta_prior_training_state_v1'
    next_step=owner.fit_step(model,opt,provider,cohort,cohort['rows'],plan)
    expected=owner.fit_state_proof(model,opt,2,next_step)
    resumed,other=fresh();resumed.research_objective=copy.deepcopy(model.research_objective)
    assert owner.restore_fit_checkpoint(resumed,other,context,plan,plan_ref,checkpoint)==1
    actual_step=owner.fit_step(resumed,other,provider,cohort,cohort['rows'],plan)
    assert owner.fit_state_proof(resumed,other,2,actual_step)==expected
    bad=copy.deepcopy(checkpoint);bad['schedule_position']['next_pass']=0
    with pytest.raises(ValueError,match='TA_FIT_CHECKPOINT_PASS_CURSOR'):
        owner.restore_fit_checkpoint(resumed,other,context,plan,plan_ref,bad)


def test_ta_initial_baseline_proves_exact_prior_for_every_selected_row():
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_train_prior_logits
    _,_,prior=_ta_prior_context();captures={}
    for name,n in [('TRAIN',1536),('CONTROL',512)]:
        captures[name+'__source_row']=np.arange(n,dtype=np.int64)
        captures.update({name+'__'+key:value.astype(np.float32)
            for key,value in native_bar_train_prior_logits(prior,rows=n).items()})
    model=SimpleNamespace(research_objective={'initialization':{'prior_sha256':prior['contract_sha256']}})
    proof=owner._ta_initial_prior_proof(model,{'prior':prior},captures)
    assert proof['all_selected_initial_logits_equal_prior']and proof['rows']=={'TRAIN':1536,'CONTROL':512}
    captures['CONTROL__close_logits'][-1,-1,-1]+=np.float32(.1)
    with pytest.raises(ValueError,match='TA_FIT_INITIAL_PRIOR_LOGITS:CONTROL:close_logits'):
        owner._ta_initial_prior_proof(model,{'prior':prior},captures)


def test_ta_learning_curve_scores_every_declared_pass_without_control_selection(tmp_path,monkeypatch):
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_train_prior_logits
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    plan,base,prior=_ta_prior_context();plan['mode']='ta_main';plan['recipe']=_ta_recipe('ta_main',2)
    plan['metadata']['architecture']={'synthetic':True}
    reference=owner.publish(tmp_path/'label_prior.json',prior);plan['inputs']['matched_prior_reference']=reference
    norm={'fit_population':{'cutoff_ns':owner.FIT_END_NS,'issue_rows':24175}}
    normal_ref=owner.publish(tmp_path/'norm.json',norm)
    class Provider:
        surfaces={'source_binding':{'native_only':True}}
        @property
        def source_binding(self):pytest.fail('TA followed GC source-binding interface')
        def input_binding(self,normalization):return {'TA_only':True}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor([0.]))
            self._native_fit_counts={'model_constructions':1,'model_forwards':0,'backward_calls':0,'optimizer_steps':0}
            self.research_objective={'initialization':{'prior_sha256':prior['contract_sha256']}}
        def require_prior_initialization_state(self):assert float(self.weight)==0
    model=Model();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    plan['metadata']['mechanics']={'initial_state_sha256':canonical_model_state_sha256(model.state_dict()),
        'result':{'path':'synthetic_TA_mechanics','sha256':'1'*64}}
    context={**base,'axis':plan['metadata']['source_pool']['axis'],
        'provider':Provider(),'normalization':norm,'normalization_ref':normal_ref,
        'prior':prior,'prior_ref':reference,'feature_requalification':{'same_TA_arrays':True}}
    plan['limits']=owner.fit_limits(plan['recipe'],'ta_main',plan['metadata'])
    monkeypatch.setattr(owner,'prepare_native_fit',lambda *args:context)
    monkeypatch.setattr(owner,'fresh_fit_model',lambda *args,**kwargs:(model,optimizer))
    calls=[];seen=[]
    def score(m,c,panels,p):
        # Deliberately worsening CONTROL cannot shorten/extend the declared loop.
        phase=len(calls);calls.append(phase);captures={};reports={}
        for name,batches in panels.items():
            rows=np.concatenate(batches);captures[name+'__source_row']=rows
            captures.update({name+'__'+key:value.astype(np.float32)
                for key,value in native_bar_train_prior_logits(prior,rows=len(rows)).items()})
            reports[name]={'diagnostic_loss':float(phase+1),'rows':len(rows)}
            m._native_fit_counts['model_forwards']+=len(batches)
        return reports,captures
    def step(m,opt,p,cohort,rows,recipe):
        cursor=len(seen);seen.append(rows.copy())
        with torch.no_grad():m.weight.add_(1)
        for key in ('model_forwards','backward_calls','optimizer_steps'):m._native_fit_counts[key]+=1
        return {'heads_gradient_l2':1.,'backbone_gradient_l2':0. if cursor==0 else 1.}
    monkeypatch.setattr(owner,'fit_score',score);monkeypatch.setattr(owner,'fit_step',step)
    for name in ('synchronize','empty_cache'):monkeypatch.setattr(torch.cuda,name,lambda:None)
    for name in ('max_memory_allocated','max_memory_reserved'):monkeypatch.setattr(torch.cuda,name,lambda:0)
    _cpu_rng(monkeypatch)
    report=owner.fit_physical(plan,{'path':'synthetic_plan','sha256':'2'*64},tmp_path)
    assert calls==[0,1,2]and len(seen)==192 and report['actual_counts']==plan['limits']
    assert [item['completed_updates']for item in report['learning_curve']]==[96,192]
    assert len(report['checkpoints'])==2 and report['declared_passes']==2
    np.testing.assert_array_equal(np.concatenate(seen[:96]),np.concatenate(seen[96:]))
    assert report['initial']['CONTROL']['diagnostic_loss']==1.
    assert report['final']['CONTROL']['diagnostic_loss']==3.
    assert report['prior_initialization']['all_selected_initial_logits_equal_prior']
    assert report['normalization_fits']==report['prior_fits']==0


@pytest.mark.parametrize('mode',['ta_mechanics','ta_main'])
def test_ta_wrong_but_valid_prior_reference_rejected_before_any_receipt_io(tmp_path,monkeypatch,mode):
    from gx1.contracts.market_state_ta_integrated_objective_v1 import require_native_bar_train_prior
    _,_,prior=_ta_prior_context()
    assert require_native_bar_train_prior(prior)['fit_population']['issue_rows']==1536
    assert prior['fit_population']['cutoff_ns']==owner.FIT_END_NS
    wrong_ref=owner.publish(tmp_path/'another_valid_1536_TRAIN_prior.json',prior)
    inputs={'ta_result':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_PIPELINE_20261119_006/RESULT.json',
        'sha256':'5c015b3af8a17c8a16459be0eeba5ca017b7625db91e193aa84b101e664e7bca'},
        'ta_terminal':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_PIPELINE_20261119_006/TERMINAL.json',
        'sha256':'849bcda1483e088ca5fa2ec8671e8ed6b003654331a6556c70af53dcfc11c874'},
        'matched_prior_reference':wrong_ref}
    if mode=='ta_main':inputs.update(mechanics_result={},mechanics_terminal={})
    monkeypatch.setattr(owner,'read',lambda *_:pytest.fail('receipt read before exact prior gate'))
    monkeypatch.setattr(ta,'fit_native_m5_feature_normalization',lambda *a,**kw:pytest.fail('normalization before prior gate'))
    with pytest.raises(ValueError,match='TA_FIT_EXACT_MATCHED_PRIOR'):owner.ta_fit_metadata(inputs,mode)
    # The exact historical path with different valid prior bytes is equally forbidden.
    inputs['matched_prior_reference']={**owner.TA_MATCHED_PRIOR_REFERENCE,'sha256':wrong_ref['sha256']}
    with pytest.raises(ValueError,match='TA_FIT_EXACT_MATCHED_PRIOR'):owner.ta_fit_metadata(inputs,mode)


def _shuffle_unit():
    metadata,context=_rank_context();metadata={'source_pool':metadata['mechanics']}
    plan={'mode':'ta_shuffle','recipe':_ta_recipe('ta_main',4),'metadata':metadata,
        'cohort_selection':owner.small_cohort_manifest(metadata)}
    selected,_=owner.selected_small_cohorts(plan,context)
    plan['training_schedule']=owner.ta_block_schedule_from_rows(selected['FIT'],plan['recipe'],
        plan['cohort_selection']['cohorts']['FIT']['selection_identity_sha256'])
    return plan,selected


def test_whole_block_order_is_prebound_dedicated_rng_and_keeps_every_window_interior():
    plan,selected=_shuffle_unit();before=np.random.get_state()
    schedule=owner.ta_block_schedule_from_rows(selected['FIT'],plan['recipe'],
        plan['cohort_selection']['cohorts']['FIT']['selection_identity_sha256'])
    after=np.random.get_state()
    assert before[0]==after[0]and np.array_equal(before[1],after[1])and before[2:]==after[2:]
    assert schedule==plan['training_schedule']
    assert schedule['block_orders_sha256']=='d5b44663d994aec349eba598892bbbaa21c0a024c86f5748e331e599414b7f39'
    assert owner.fit_recipe(plan['recipe'],'ta_shuffle')==plan['recipe']
    assert owner.fit_limits(plan['recipe'],'ta_shuffle',plan['metadata'])=={
        'model_constructions':1,'model_forwards':544,'backward_calls':384,'optimizer_steps':384}
    batches=owner._ta_training_batches(selected,plan);blocks=selected['FIT'].reshape(96,16)
    for number,order in enumerate(schedule['block_orders']):
        np.testing.assert_array_equal(np.sort(order),np.arange(96))
        assert order!=list(range(96))
        for batch,index in zip(batches[number*96:(number+1)*96],order):
            np.testing.assert_array_equal(batch,blocks[index])
        assert owner.digest(np.concatenate(batches[number*96:(number+1)*96]))==schedule['per_pass_source_rows_sha256'][number]
    assert owner.digest(np.concatenate(batches))==schedule['source_row_schedule_sha256']
    for change in ({'passes':3,'fit_batches':288},{'seed':7}):
        bad=copy.deepcopy(plan['recipe']);bad.update(change)
        with pytest.raises(ValueError):owner.fit_recipe(bad,'ta_shuffle')
    assert owner.FIT_SCOPES['ta_shuffle'][1].name=='NATIVE_M5_TA_PRIOR_BLOCK_ORDER_20261119_001.json'


def test_block_order_schedule_and_checkpoint_cursor_reject_different_order_or_rows():
    plan,selected=_shuffle_unit();position=owner._ta_schedule_position(plan,96);schedule=plan['training_schedule']
    block=schedule['block_orders'][1][0]
    assert position['next_pass']==1 and position['next_batch_in_pass']==0
    assert position['next_source_block']==block
    assert position['next_source_batch_sha256']==owner.digest(selected['FIT'].reshape(96,16)[block])
    assert position['training_schedule_sha256']==schedule['contract_sha256']
    assert owner._ta_schedule_position(plan,384)['next_source_block']is None
    for kind in ('order','hash','source'):
        bad=copy.deepcopy(plan);other=copy.deepcopy(selected)
        if kind=='order':bad['training_schedule']['block_orders'][0][:2]=reversed(bad['training_schedule']['block_orders'][0][:2])
        if kind=='hash':bad['training_schedule']['contract_sha256']='0'*64
        if kind=='source':other['FIT'][0]-=1
        with pytest.raises(ValueError,match='TA_BLOCK_EXACT_SCHEDULE'):owner._ta_training_batches(other,bad)
    bad=copy.deepcopy(plan);bad['training_schedule']['block_orders'][0][0]=96
    with pytest.raises(ValueError,match='TA_BLOCK_CHECKPOINT_SCHEDULE_IDENTITY'):owner._ta_schedule_position(bad,0)


def test_shuffle_dispatch_reuses_identical_ta_statistics_owner_without_mutating_plan(monkeypatch,tmp_path):
    plan,_=_shuffle_unit();before=copy.deepcopy(plan);calls=[]
    def prepare(p,root,**kwargs):calls.append((p,root,kwargs));return {'TA_statistics_reused':True}
    monkeypatch.setattr(owner,'prepare_ta_native_fit',prepare)
    assert owner.prepare_native_fit(plan,tmp_path)=={'TA_statistics_reused':True}
    assert calls[0][0]['mode']=='ta_main'and calls[0][0]['training_schedule']==plan['training_schedule']
    assert calls[0][1]==tmp_path and plan==before


@pytest.mark.parametrize('role',list(owner.TA_BLOCK_ORDER_REFERENCE))
def test_shuffle_exact_chronological_reference_gate_precedes_all_receipt_io(monkeypatch,role):
    inputs={'ta_result':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_PIPELINE_20261119_006/RESULT.json',
        'sha256':'5c015b3af8a17c8a16459be0eeba5ca017b7625db91e193aa84b101e664e7bca'},
        'ta_terminal':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_PIPELINE_20261119_006/TERMINAL.json',
        'sha256':'849bcda1483e088ca5fa2ec8671e8ed6b003654331a6556c70af53dcfc11c874'},
        'matched_prior_reference':copy.deepcopy(owner.TA_MATCHED_PRIOR_REFERENCE),
        'mechanics_result':{},'mechanics_terminal':{},**copy.deepcopy(owner.TA_BLOCK_ORDER_REFERENCE)}
    inputs[role]['sha256']='0'*64
    monkeypatch.setattr(owner,'read',lambda *_:pytest.fail('unbound chronological receipt read'))
    with pytest.raises(ValueError,match='TA_BLOCK_EXACT_CHRONOLOGICAL_REFERENCE'):owner.ta_fit_metadata(inputs,'ta_shuffle')


def test_block_source_bridge_is_narrow_and_rejects_numeric_or_other_owner_changes(tmp_path,monkeypatch):
    leaves=('fresh_fit_model','fit_step','fit_score','prepare_ta_native_fit','fit_axes',
        'seed_native_fit','training_rng','restore_training_rng','state_sha','slice_truth',
        'fit_checkpoint','publish_checkpoint','restore_fit_checkpoint','fit_state_proof',
        '_ta_initial_prior_proof','cold_resume_fit','require_cold_resume_proof')
    original="FIT_SCOPES={'ta_main':'old'}\nADAMW_RECIPE={'lr':1e-4}\n"+''.join('def '+name+'():return 1\n'for name in leaves)
    original+='def _ta_training_batches():return 1\n'
    changed=original.replace("'old'","'new'").replace('def _ta_training_batches():return 1','def _ta_training_batches():return 96')
    driver='gx1/audit/qualify_native_m5_full_gc_direction_v1.py'
    paths=(driver,'tests/test_qualify_native_m5_full_gc_direction_v1.py',
        'scripts/gx1_capped_run.sh','tests/test_gx1_capped_run_contract.py','gx1/contracts/model_numeric.py')
    parent={'mode':'ta_mechanics','source_base_commit':'f'*40,'source_owners':{}}
    for path in paths:
        target=tmp_path/path;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(original if path==driver else'original\n');parent['source_owners'][path]=owner.sha(target)
    (tmp_path/driver).write_text(changed)
    monkeypatch.setattr(owner,'REPO',tmp_path);monkeypatch.setattr(owner,'TA_FIT_SOURCE_PATHS',paths)
    monkeypatch.setattr(owner.subprocess,'check_output',lambda *a,**kw:original.encode())
    proof=owner.ta_block_order_source_bridge(parent)
    assert proof['all_other_source_owners_exact']and len(proof['unchanged_numeric_leaf_ast_sha256'])==18
    assert set(proof['source_changes'])=={driver}
    (tmp_path/driver).write_text(changed.replace('def fit_step():return 1','def fit_step():return 2'))
    with pytest.raises(ValueError,match='TA_BLOCK_UNCHANGED_NUMERICAL_AST'):owner.ta_block_order_source_bridge(parent)
    (tmp_path/driver).write_text(changed);(tmp_path/paths[-1]).write_text('altered numeric model\n')
    with pytest.raises(ValueError,match='TA_BLOCK_UNCHANGED_MODEL_LOSS_STATS_OWNERS'):owner.ta_block_order_source_bridge(parent)


def test_shuffled_pass_boundary_cold_restore_preserves_next_batch_adamw_rng_and_rejects_tampering(monkeypatch):
    _cpu_rng(monkeypatch);plan,selected=_shuffle_unit();batches=owner._ta_training_batches(selected,plan)
    provider=_provider();raw,_,_=_batch_truth();rows=selected['FIT'];n=len(rows);step=300_000_000_000
    def gather(source_rows,*,max_rows):
        assert len(source_rows)<=max_rows
        out={key:np.repeat(value[:1],len(source_rows),axis=0).copy()for key,value in raw.items()}
        for value in out.values():
            if value.ndim==3:value[:,-1,0]=(source_rows%11+1).astype(np.float32)
        return out
    provider.gather=gather;provider.input_binding=lambda norm:{'normalization_sha256':norm['contract_sha256']}
    issues=owner.FIT_START_NS+np.arange(n,dtype=np.int64)*step;shape=(n,5)
    truth={key:np.ones(shape,bool)for key in ('target_present','observed_previous','mature','eligible','open_known_at_issue')}
    truth.update(issue_ns=issues,open_sign=np.zeros(shape,np.int8),
        close_sign=np.broadcast_to((np.arange(n)%3-1).astype(np.int8)[:,None],shape).copy(),
        target_row=np.broadcast_to(np.arange(n,dtype=np.int64)[:,None],shape).copy(),
        validity_ns=np.broadcast_to((issues+step)[:,None],shape).copy())
    cohort={'rows':rows,'fit_rows':rows,'truth':truth,'fit_truth':truth,'upper_ns':owner.FIT_END_NS}
    context={'provider':provider,'normalization':{'contract_sha256':'unit_TA_normalization'},
        'normalization_ref':{'path':'unitnorm','sha256':'1'*64},'prior_ref':{'path':'unitprior','sha256':'2'*64}}
    plan['metadata'].update(target_contract=target_contract(),architecture={'unit':True})
    plan_ref={'path':'unit_shuffle_plan','sha256':'3'*64}
    def fresh():
        model=_training_model(provider);model.research_objective={'unit_prior_initialization_sha256':'4'*64}
        return model,torch.optim.AdamW(model.parameters(),**{**owner.ADAMW_RECIPE,'betas':tuple(owner.ADAMW_RECIPE['betas'])})
    torch.manual_seed(42);model,optimizer=fresh()
    for batch in batches[:96]:owner.fit_step(model,optimizer,provider,cohort,batch,plan)
    checkpoint=copy.deepcopy(owner.fit_checkpoint(model,optimizer,context,plan,plan_ref,96))
    assert checkpoint['schedule_position']['next_pass']==1 and checkpoint['schedule_position']['next_batch_in_pass']==0
    assert checkpoint['schedule_position']['next_source_batch_sha256']==owner.digest(batches[96])
    expected_step=owner.fit_step(model,optimizer,provider,cohort,batches[96],plan)
    expected=owner.fit_state_proof(model,optimizer,97,expected_step)
    resumed,other=fresh();assert owner.restore_fit_checkpoint(resumed,other,context,plan,plan_ref,checkpoint)==96
    actual_step=owner.fit_step(resumed,other,provider,cohort,batches[96],plan)
    actual=owner.fit_state_proof(resumed,other,97,actual_step)
    # The optional first-two-local-call gradient diagnostic is process-local;
    # parameters, optimizer, RNG, logits, loss and the true next source batch are exact.
    assert {key:value for key,value in actual.items()if key!='step'}=={key:value for key,value in expected.items()if key!='step'}
    assert all(actual_step[key]==value for key,value in expected_step.items())
    state_before=owner.state_sha(resumed.state_dict())
    for key in ('cursor','next_source_block','training_schedule_sha256'):
        bad=copy.deepcopy(checkpoint)
        if key=='cursor':bad['cursor']=97
        elif key=='next_source_block':bad['schedule_position'][key]=(bad['schedule_position'][key]+1)%96
        else:bad['schedule_position'][key]='0'*64
        with pytest.raises(ValueError,match='TA_FIT_CHECKPOINT_PASS_CURSOR'):
            owner.restore_fit_checkpoint(resumed,other,context,plan,plan_ref,bad)
        assert owner.state_sha(resumed.state_dict())==state_before


def test_shared_gradient_cross_conflict_is_distinct_from_joint_direction_and_zero_norm():
    report=owner.shared_gradient_measures((torch.tensor([1.,0.]),),(torch.tensor([-.5,1.]),))
    assert report['cross_dot']==-.5 and report['cosine']<0
    assert report['g5_dot_joint']==.5 and report['raw_joint_opposes_M5']is False
    zero=owner.shared_gradient_measures((None,torch.zeros(2)),(torch.ones(1),torch.ones(2)))
    assert zero['g5_l2']==0 and zero['cosine']is None and zero['g5_dot_joint']==0
    assert zero['raw_joint_opposes_M5']is False and zero['unused_M5_parameter_tensors']==1
    with pytest.raises(ValueError,match='SHARED_NONFINITE_GRADIENT'):
        owner.shared_gradient_measures((torch.tensor([float('inf')]),),(None,))


def test_shared_gradient_parent_refs_and_cpu_cli_fail_before_io(monkeypatch):
    monkeypatch.setattr(owner,'read',lambda *_:pytest.fail('unbound parent read'))
    with pytest.raises(ValueError,match='SHARED_EXACT_PARENT_REFS'):owner.shared_gradient_metadata({})
    bad=copy.deepcopy(owner.SHARED_GRADIENT_PARENT);bad['block_result']['sha256']='0'*64
    with pytest.raises(ValueError,match='SHARED_EXACT_PARENT_REFS'):owner.shared_gradient_metadata(bad)
    monkeypatch.setattr(owner,'inspect_shared_gradient',lambda *a,**k:pytest.fail('bad CLI reached metadata gate'))
    args=SimpleNamespace(plan=str(owner.SHARED_GRADIENT_PLAN),out_dir=str(owner.SHARED_GRADIENT_ROOT),
        device='cuda',plan_sha256='0'*64,preflight=True)
    with pytest.raises(ValueError,match='SHARED_CLI_SCOPE'):owner.shared_gradient_cli(args)
    assert owner.SHARED_GRADIENT_BLOCKS==(0,31,63,95)
    assert owner.SHARED_GRADIENT_EXECUTION['internal_seconds']==870
    assert owner.SHARED_GRADIENT_EXECUTION['external_wall_timeout_enforced']is False


def test_shared_gradient_source_bridge_keeps_entire_old_driver_except_cpu_dispatch(tmp_path,monkeypatch):
    leaves=('fresh_fit_model','fit_step','fit_score','prepare_ta_native_fit','fit_axes','seed_native_fit',
        'training_rng','restore_training_rng','state_sha','slice_truth','fit_checkpoint','publish_checkpoint',
        'restore_fit_checkpoint','fit_state_proof','_ta_initial_prior_proof','cold_resume_fit','require_cold_resume_proof')
    original="ADAMW_RECIPE={'lr':1e-4}\n"+''.join('def '+name+'():return 1\n'for name in leaves)
    original+='def main():return 1\n'
    changed=original.replace('def main():return 1','def main():return 2')
    changed+='SHARED_GRADIENT_STAGE="new"\ndef qualify_shared_gradient():return 8\n'
    driver='gx1/audit/qualify_native_m5_full_gc_direction_v1.py';test='tests/test_qualify_native_m5_full_gc_direction_v1.py'
    paths=(driver,test,'gx1/contracts/unchanged_numeric.py');parent={'mode':'ta_shuffle','source_base_commit':'f'*40,'source_owners':{}}
    for name in paths:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(original if name==driver else'original\n')
        parent['source_owners'][name]=owner.sha(path)
    (tmp_path/driver).write_text(changed)
    monkeypatch.setattr(owner,'REPO',tmp_path);monkeypatch.setattr(owner,'TA_FIT_SOURCE_PATHS',paths)
    monkeypatch.setattr(owner.subprocess,'check_output',lambda *a,**kw:original.encode())
    proof=owner.shared_gradient_source_bridge(parent)
    assert proof['all_other_source_owners_exact']and len(proof['numeric_identity']['unchanged_numeric_leaf_ast_sha256'])==18
    (tmp_path/driver).write_text(changed.replace('def fit_score():return 1','def fit_score():return 2'))
    with pytest.raises(ValueError,match='SHARED_EXISTING_DRIVER_AST'):owner.shared_gradient_source_bridge(parent)
    (tmp_path/driver).write_text(changed);(tmp_path/paths[-1]).write_text('changed model/loss/stats\n')
    with pytest.raises(ValueError,match='SHARED_UNCHANGED_MODEL_LOSS_STATS_OWNERS'):owner.shared_gradient_source_bridge(parent)


def test_shared_gradient_qualifier_has_four_train_forwards_eight_derivatives_and_immutable_state(tmp_path,monkeypatch):
    parent,context,prior=_ta_prior_context();parent['mode']='ta_shuffle'
    selected,selection=owner.selected_small_cohorts(parent,context);template,_,_=_batch_truth();seen=[]
    first=selected['FIT'][:16];positions=np.searchsorted(context['cohorts']['TRAIN']['rows'],first)
    context['cohorts']['TRAIN']['truth']['open_known_at_issue'][positions]=True
    provider=_provider();provider.surfaces={'source_binding':{'native_only':True}}
    issue=np.zeros(int(context['cohorts']['CONTROL']['rows'][-1])+1,np.int64)
    for cohort in context['cohorts'].values():issue[cohort['rows']]=dict.__getitem__(cohort['truth'],'issue_ns')
    provider.issue_ns=issue
    def gather(rows,*,max_rows):
        seen.append(rows.copy());assert max_rows==len(rows)==16
        out={key:np.repeat(value[:1],len(rows),axis=0).copy()for key,value in template.items()}
        for value in out.values():
            if value.ndim==3:value[:,-1,0]=(rows%7+1).astype(np.float32)
        return out
    provider.gather=gather
    norm={'unit_TRAIN_only':True};normal_ref=owner.publish(tmp_path/'norm.json',norm);prior_ref=owner.publish(tmp_path/'prior.json',prior)
    context.update(provider=provider,normalization=norm,normalization_ref=normal_ref,prior=prior,prior_ref=prior_ref)
    model=_training_model(provider).eval()
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    before=canonical_model_state_sha256(model.state_dict())
    plan={'metadata':{'parent_fit_plan':parent,'normalization':normal_ref,'prior':prior_ref,
        'selected_cohorts':selection,'final_checkpoint':{'path':'unit_checkpoint','sha256':'1'*64},'final_state_sha256':before}}
    monkeypatch.setattr(owner,'prepare_native_fit',lambda *a,**kw:context)
    monkeypatch.setattr(owner,'shared_gradient_load_model',lambda *args:model)
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**k:pytest.fail('diagnostic constructed optimizer'))
    monkeypatch.setattr(torch.Tensor,'backward',lambda *a,**k:pytest.fail('diagnostic called backward'))
    real=torch.autograd.grad;derivatives=[]
    def grad(*args,**kwargs):
        result=real(*args,**kwargs);derivatives.append(tuple(None if x is None else x.detach().clone()for x in result));return result
    monkeypatch.setattr(torch.autograd,'grad',grad)
    report=owner.qualify_shared_gradient(plan)
    assert report['actual_counts']==owner.SHARED_GRADIENT_LIMITS and len(derivatives)==8
    assert len(seen)==4 and [item['canonical_TRAIN_block']for item in report['batches']]==[0,31,63,95]
    for actual,index in zip(seen,[0,31,63,95]):np.testing.assert_array_equal(actual,selected['FIT'][index*16:(index+1)*16])
    assert report['CONTROL_analysis_rows']==0 and report['shared_loader_may_decode_qualified_CONTROL']is True
    assert report['model_unchanged']and report['CPU_python_numpy_torch_RNG_unchanged']
    assert report['parameter_grad_None_before_and_after']and all(p.grad is None for p in model.parameters())
    assert canonical_model_state_sha256(model.state_dict())==before
    weights=np.array([item['Ntotal']for item in report['batches']],np.float64);assert len(np.unique(weights))>1
    parameters=tuple(p for p in model.backbone.parameters()if p.requires_grad)
    vectors=[]
    for task in (0,1):
        values=[]
        for grads in derivatives[task::2]:
            values.append(np.concatenate([np.zeros(p.numel())if g is None else g.numpy().astype(np.float64).ravel()
                for p,g in zip(parameters,grads)]))
        vectors.append(np.average(np.stack(values),axis=0,weights=weights))
    a,b=vectors;expected_dot=float(a@b)
    assert report['panel']['cross_dot']==pytest.approx(expected_dot,abs=1e-12)
    assert report['panel']['g5_dot_joint']==pytest.approx(float(a@(a+b)),abs=1e-12)
    assert report['panel']['g5_l2']==pytest.approx(float(np.linalg.norm(a)),abs=1e-12)


def test_shared_gradient_controller_preclaim_and_failed_terminal_are_one_use(tmp_path,monkeypatch):
    root=tmp_path/'diagnostic';monkeypatch.setattr(owner,'SHARED_GRADIENT_ROOT',root)
    monkeypatch.setattr(owner.common,'firewall_state',lambda:{'denied_attempts':0})
    cap={'memory_max':12*1024**3,'memory_high':12*1024**3,'swap':512*1024**2,'pids':64,
        'numerical_threads':1,'cgroup_path':'/sys/fs/cgroup/unit'}
    monkeypatch.setattr(owner.common,'require_capped_cpu_research_producer_execution',lambda:cap)
    monkeypatch.setattr(owner.common,'require_capped_lock_ancestry',lambda:{'scope':'/unit'})
    def blocked(*a,**k):raise ValueError('preclaim refused')
    monkeypatch.setattr(owner,'inspect_shared_gradient',blocked)
    with pytest.raises(ValueError,match='preclaim refused'):owner.run_shared_gradient('0'*64)
    assert not root.exists()
    plan_ref={'path':'unitplan','sha256':'1'*64};policy_ref={'path':'unitpolicy','sha256':'2'*64}
    monkeypatch.setattr(owner,'inspect_shared_gradient',lambda *a,**k:({},plan_ref,policy_ref))
    def failed(*args):raise FloatingPointError('nonfinite raw gradient')
    monkeypatch.setattr(owner,'qualify_shared_gradient',failed)
    previous=owner.signal.getsignal(owner.signal.SIGALRM)
    with pytest.raises(FloatingPointError,match='nonfinite raw gradient'):owner.run_shared_gradient('0'*64)
    terminal=json.loads((root/'TERMINAL.json').read_text())
    assert terminal['status']=='FAILED_CONSUMED'and terminal['remaining_invocations']==0
    assert terminal['relaunch_allowed']is False and terminal['test_data_used']is False
    assert (root/'CLAIM.json').exists()and owner.signal.getitimer(owner.signal.ITIMER_REAL)==(0.,0.)
    assert owner.signal.getsignal(owner.signal.SIGALRM)==previous
    # A separate synthetic one-use root exercises the success terminal, never a relaunch.
    complete=tmp_path/'complete';monkeypatch.setattr(owner,'SHARED_GRADIENT_ROOT',complete)
    monkeypatch.setattr(owner,'qualify_shared_gradient',lambda p:{'actual_counts':owner.SHARED_GRADIENT_LIMITS})
    monkeypatch.setattr(owner,'shared_gradient_checked_inputs',lambda p:None)
    monkeypatch.setattr(owner,'shared_gradient_source_clean',lambda p:True)
    real_ref=owner.ref
    monkeypatch.setattr(owner,'ref',lambda p:policy_ref if p==owner.POLICY else plan_ref if p==owner.SHARED_GRADIENT_PLAN else real_ref(p))
    refs=owner.run_shared_gradient('0'*64)
    done=json.loads((complete/'TERMINAL.json').read_text());result=json.loads((complete/'RESULT.json').read_text())
    assert done['status']=='COMPLETE'and done['result']==refs['result']and done['relaunch_allowed']is False
    assert result['training_authorized']is False and result['optimizer_steps']==0
    assert set(path.name for path in complete.iterdir())=={'CLAIM.json','RESULT.json','TERMINAL.json'}


def test_shared_gradient_loads_only_final_model_state_and_rejects_wrong_bindings_before_copy(tmp_path,monkeypatch):
    import io
    from gx1.contracts import market_state_ta_integrated_model_v1 as models
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    parent,_=_shuffle_unit();parent['metadata'].update(architecture={'unit':True},target_contract=target_contract())
    provider=_provider();provider.input_binding=lambda norm:{'unit_normalization':norm['contract_sha256']}
    context={'provider':provider,'normalization':{'contract_sha256':'unit_norm'},'prior':{'unit_prior':True},
        'normalization_ref':{'path':'unitnorm','sha256':'1'*64},'prior_ref':{'path':'unitprior','sha256':'2'*64}}
    def small():
        model=_training_model(provider);model.research_objective={'unit_prior_contract':True};return model
    original=small();parent_ref={'path':'unitparent','sha256':'3'*64};calls=[]
    checkpoint={'schema_version':'gx1_native_m5_ta_prior_training_state_v1','plan':parent_ref,'cursor':384,
        'schedule_position':owner._ta_schedule_position(parent,384),'model_state':original.state_dict(),
        'optimizer_state':{'must_never_be_applied':True},'optimizer_parameter_names':[name for name,_ in original.named_parameters()],
        'bindings':{'input':provider.input_binding(context['normalization']),'normalization':context['normalization_ref'],
            'prior':context['prior_ref'],'target':target_contract(),'architecture':parent['metadata']['architecture'],
            'recipe':parent['recipe'],'model_contract':original.research_objective}}
    def save(name):
        stream=io.BytesIO();torch.save(checkpoint,stream);return owner.publish(tmp_path/name,stream.getvalue())
    plan={'metadata':{'parent_fit_plan':parent,'dependencies':{'parent_plan':parent_ref},
        'final_checkpoint':save('final.pt'),'final_state_sha256':canonical_model_state_sha256(original.state_dict())}}
    def factory(**kwargs):
        assert kwargs['device']==torch.device('cpu')and kwargs['train_prior']is context['prior']
        model=small();load=model.load_state_dict
        def strict(state,*,strict):calls.append(strict);return load(state,strict=strict)
        model.load_state_dict=strict;return model
    monkeypatch.setattr(models,'build_native_bar_direction_model',factory)
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**kw:pytest.fail('optimizer constructed'))
    monkeypatch.setattr(owner,'fresh_fit_model',lambda *a,**kw:pytest.fail('training factory called'))
    loaded=owner.shared_gradient_load_model(plan,context)
    assert calls==[True]and loaded.training is False
    assert canonical_model_state_sha256(loaded.state_dict())==plan['metadata']['final_state_sha256']
    checkpoint['bindings']['prior']={'path':'wrong_prior','sha256':'0'*64}
    plan['metadata']['final_checkpoint']=save('bad.pt')
    with pytest.raises(ValueError,match='SHARED_FINAL_CHECKPOINT_BINDINGS'):owner.shared_gradient_load_model(plan,context)
    assert calls==[True]


def test_shared_gradient_admission_keeps_training_flags_false(tmp_path,monkeypatch):
    root=tmp_path/'not_created';monkeypatch.setattr(owner,'SHARED_GRADIENT_ROOT',root)
    plan={'inputs':{},'metadata':{'dependencies':{}},'source_base_commit':'unit',
        'execution':copy.deepcopy(owner.SHARED_GRADIENT_EXECUTION)}
    plan_ref={'path':str(owner.SHARED_GRADIENT_PLAN),'sha256':'1'*64};policy_ref={'path':str(owner.POLICY),'sha256':'2'*64}
    policy={key:False for key in owner.ta_audit.data.CLOSED_FLAGS}
    policy[owner.SHARED_GRADIENT_STAGE]={'plan':plan_ref,'output_root':str(root),
        'max_invocations':1,'claimed_invocations':0,'completed_invocations':0,'remaining_invocations':1,
        'relaunch_allowed':False,'training_authorized':False,'test_authorized':False,
        'execution':plan['execution'],'status':'ADMITTED_ONE_USE','execution_authorized':True,**owner.SHARED_GRADIENT_LIMITS}
    monkeypatch.setattr(owner.Path,'cwd',lambda:owner.REPO)
    monkeypatch.setattr(owner,'read',lambda binding:plan if binding==plan_ref else policy)
    monkeypatch.setattr(owner,'ref',lambda path:policy_ref)
    monkeypatch.setattr(owner,'build_shared_gradient_plan',lambda **kw:plan)
    monkeypatch.setattr(owner,'shared_gradient_source_clean',lambda p:True)
    assert owner.inspect_shared_gradient('1'*64,admitted=True)[0]is plan
    policy['training_enabled']=True
    with pytest.raises(ValueError,match='SHARED_CLOSED_POLICY'):owner.inspect_shared_gradient('1'*64,admitted=True)
    policy['training_enabled']=False;policy[owner.SHARED_GRADIENT_STAGE]['training_authorized']=True
    with pytest.raises(ValueError,match='SHARED_CLOSED_POLICY'):owner.inspect_shared_gradient('1'*64,admitted=True)
    assert not root.exists()


def test_primary_recipe_keeps_prior_stats_and_block_order_with_explicit_probe_budget():
    plan,selected=_shuffle_unit();plan['mode']='ta_primary_main'
    plan['backbone_gradient_policy']=owner.PRIMARY_BACKBONE_GRADIENT_POLICY
    assert owner.fit_recipe(plan['recipe'],'ta_primary_main')==plan['recipe']
    assert owner.fit_limits(plan['recipe'],'ta_primary_main',plan['metadata'])=={
        'model_constructions':1,'model_forwards':544,'backward_calls':384,'optimizer_steps':384,'autograd_calls':0}
    expected=copy.deepcopy(plan);expected['mode']='ta_shuffle'
    for actual,old in zip(owner._ta_training_batches(selected,plan),owner._ta_training_batches(selected,expected)):
        np.testing.assert_array_equal(actual,old)
    assert owner._ta_schedule_position(plan,96)==owner._ta_schedule_position(expected,96)
    mechanics=copy.deepcopy(plan);mechanics['mode']='ta_primary_mechanics';mechanics['recipe']=_ta_recipe()
    assert owner.fit_recipe(mechanics['recipe'],'ta_primary_mechanics')==mechanics['recipe']
    assert owner.fit_limits(mechanics['recipe'],'ta_primary_mechanics',mechanics['metadata'])=={
        'model_constructions':2,'model_forwards':67,'backward_calls':3,'optimizer_steps':3,'autograd_calls':6}
    batches=owner._ta_training_batches(selected,mechanics)
    assert len(batches)==2;np.testing.assert_array_equal(batches[0],batches[1])
    assert owner._ta_schedule_position(mechanics,1)['next_pass']==1
    assert owner.FIT_SCOPES['ta_primary_mechanics'][1].name=='NATIVE_M5_TA_PRIMARY_BACKBONE_MECHANICS_20261119_001.json'
    assert owner.FIT_SCOPES['ta_primary_main'][1].name=='NATIVE_M5_TA_PRIMARY_BACKBONE_FIT_20261119_001.json'


def test_primary_parent_role_gate_precedes_receipts_and_new_policy_is_explicit(monkeypatch):
    monkeypatch.setattr(owner,'read',lambda *_:pytest.fail('unbound primary receipt read'))
    with pytest.raises(ValueError,match='PRIMARY_EXACT_PARENT_REFS'):owner.primary_backbone_metadata({},'ta_primary_mechanics')
    refs={**copy.deepcopy(owner.SHARED_GRADIENT_PARENT),**copy.deepcopy(owner.PRIMARY_GRADIENT_DESIGN_REFERENCE),
        'matched_prior_reference':copy.deepcopy(owner.TA_MATCHED_PRIOR_REFERENCE)}
    refs['gradient_review']['sha256']='0'*64
    with pytest.raises(ValueError,match='PRIMARY_EXACT_PARENT_REFS'):owner.primary_backbone_metadata(refs,'ta_primary_mechanics')
    assert owner.PRIMARY_BACKBONE_GRADIENT_POLICY=='m5_close_only_backbone_v1'


def test_primary_source_identity_opens_only_readout_policy_and_nonmutating_probes():
    leaves=('fresh_fit_model','fit_step','fit_score','prepare_ta_native_fit','fit_axes','seed_native_fit',
        'training_rng','restore_training_rng','state_sha','slice_truth','fit_checkpoint','publish_checkpoint',
        'restore_fit_checkpoint','fit_state_proof','_ta_initial_prior_proof','cold_resume_fit','require_cold_resume_proof')
    source="ADAMW_RECIPE={'lr':1e-4}\n"+''.join('def '+name+'():return {}\n'for name in leaves)
    changed=source.replace('def fresh_fit_model():return {}',
        'def fresh_fit_model():\n if _primary_fit(plan):kwargs["backbone_gradient_policy"]="m5_close_only_backbone_v1"\n return {}')
    changed=changed.replace('def fit_step():return {}',
        'def fit_step():\n primary_probe=primary_backbone_probe(model,loss,features)if plan.get("mode")=="ta_primary_mechanics"else{}\n return {**primary_probe}')
    assert owner.primary_driver_source_identity(source)==owner.primary_driver_source_identity(changed)
    wrong=changed.replace('def fit_score():return {}','def fit_score():return {"changed_loss_math":1}')
    assert owner.primary_driver_source_identity(source)!=owner.primary_driver_source_identity(wrong)
    old='class NativeM5BarDirectionModel:\n def forward(self):return 1\n def _gc_binding_matches(self):return 1\n'
    new=old.replace('def forward(self):return 1','def forward(self):return 2')+'M5_CLOSE_ONLY_BACKBONE_GRADIENT_POLICY="new"\n'
    assert owner.primary_model_source_identity(old)['unchanged_model_ast_sha256']==owner.primary_model_source_identity(new)['unchanged_model_ast_sha256']
    bad=new.replace('def _gc_binding_matches(self):return 1','def _gc_binding_matches(self):return 2')
    assert owner.primary_model_source_identity(old)['unchanged_model_ast_sha256']!=owner.primary_model_source_identity(bad)['unchanged_model_ast_sha256']


def _primary_training_unit(monkeypatch,leak=False):
    _cpu_rng(monkeypatch);_,truth,_=_batch_truth();rows=np.array([0,1],np.int64)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.backbone=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.Dropout(.25))
            self.open_heads=torch.nn.ModuleDict({tf:torch.nn.Linear(4,3)for tf in ta.TIMEFRAMES})
            self.close_heads=torch.nn.ModuleDict({tf:torch.nn.Linear(4,3)for tf in ta.TIMEFRAMES})
            for heads in (self.open_heads,self.close_heads):
                for head in heads.values():
                    with torch.no_grad():head.weight.zero_();head.bias.copy_(torch.log(torch.tensor([.4,.2,.4])))
            self.research_objective={'backbone_gradient_policy':{'name':owner.PRIMARY_BACKBONE_GRADIENT_POLICY}}
            self._native_fit_counts={'model_constructions':1,'model_forwards':0,'backward_calls':0,'optimizer_steps':0,'autograd_calls':0}
        def forward(self,features):
            self._native_fit_counts['model_forwards']+=1;h=self.backbone(features['x'])
            return {task+'_logits':torch.stack([heads[tf](h if leak or(task=='close'and tf=='M5')else h.detach())
                for tf in ta.TIMEFRAMES],dim=1)for task,heads in (('open',self.open_heads),('close',self.close_heads))}
    provider=SimpleNamespace(gather=lambda r,max_rows:{'x':np.array([[1.,2.],[3.,-1.]],np.float32)},
        input_binding=lambda norm:{'unit_norm':norm['contract_sha256']})
    cohort={'rows':rows,'fit_rows':rows,'truth':truth,'fit_truth':truth,'upper_ns':owner.FIT_END_NS}
    plan={'mode':'ta_primary_mechanics','recipe':_ta_recipe(),'metadata':{'target_contract':target_contract(),'architecture':{'unit':True}},
        'backbone_gradient_policy':owner.PRIMARY_BACKBONE_GRADIENT_POLICY}
    context={'provider':provider,'normalization':{'contract_sha256':'unit_prior_norm'},
        'normalization_ref':{'path':'unitnorm','sha256':'1'*64},'prior_ref':{'path':'unitprior','sha256':'2'*64}}
    def fresh():
        model=Model();return model,torch.optim.AdamW(model.parameters(),**{**owner.ADAMW_RECIPE,'betas':tuple(owner.ADAMW_RECIPE['betas'])})
    return provider,cohort,context,plan,fresh


def test_primary_mechanics_reuses_loss_for_six_probes_and_exact_cold_pass_boundary(monkeypatch):
    provider,cohort,context,plan,fresh=_primary_training_unit(monkeypatch);torch.manual_seed(42);model,optimizer=fresh()
    first=owner.fit_step(model,optimizer,provider,cohort,cohort['rows'],plan)
    route=first['primary_gradient_routing']
    assert route['auxiliary_shared_gradient_l2']==route['M5_shared_gradient_l2']==0
    assert route['auxiliary_head_gradient_l2']>0 and route['M5_head_gradient_l2']>0
    assert route['input_key_gradient_l2']=={'x':0.}
    plan_ref={'path':'primary_plan','sha256':'3'*64}
    checkpoint=copy.deepcopy(owner.fit_checkpoint(model,optimizer,context,plan,plan_ref,1))
    assert checkpoint['schedule_position']['next_pass']==1 and checkpoint['schedule_position']['next_batch_in_pass']==0
    second=owner.fit_step(model,optimizer,provider,cohort,cohort['rows'],plan)
    route=second['primary_gradient_routing'];assert route['auxiliary_shared_gradient_l2']==0
    assert route['M5_shared_gradient_l2']>0 and route['input_key_gradient_l2']['x']>0
    expected=owner.fit_state_proof(model,optimizer,2,second)
    resumed,other=fresh();owner.restore_fit_checkpoint(resumed,other,context,plan,plan_ref,checkpoint)
    actual_step=owner.fit_step(resumed,other,provider,cohort,cohort['rows'],plan)
    assert owner.fit_state_proof(resumed,other,2,actual_step)==expected
    assert model._native_fit_counts['autograd_calls']+resumed._native_fit_counts['autograd_calls']==6
    assert all(p.grad is None for p in model.parameters())and all(p.grad is None for p in resumed.parameters())
    assert first['predictive_counts']==second['predictive_counts']==actual_step['predictive_counts']


def test_primary_probe_rejects_actual_auxiliary_shared_leak_and_main_has_no_probes(monkeypatch):
    provider,cohort,_,plan,fresh=_primary_training_unit(monkeypatch,leak=True);model,optimizer=fresh()
    with torch.no_grad():model.close_heads['H1'].weight[0].fill_(.25)
    with pytest.raises(ValueError,match='PRIMARY_AUXILIARY_ROUTE_OR_GRAD_ACCUMULATION'):
        owner.fit_step(model,optimizer,provider,cohort,cohort['rows'],plan)
    provider,cohort,_,plan,fresh=_primary_training_unit(monkeypatch);plan['mode']='ta_primary_main';model,optimizer=fresh()
    monkeypatch.setattr(owner,'primary_backbone_probe',lambda *a,**kw:pytest.fail('main probed gradients'))
    result=owner.fit_step(model,optimizer,provider,cohort,cohort['rows'],plan)
    assert 'primary_gradient_routing'not in result and model._native_fit_counts['autograd_calls']==0


def test_primary_prepare_reuses_parent_statistics_and_never_initializes_from_old_weights(tmp_path,monkeypatch):
    parent={'mode':'ta_shuffle'};normal={'path':'normal','sha256':'1'*64};prior={'path':'prior','sha256':'2'*64}
    context={'normalization_ref':normal,'prior_ref':prior,'axis':{'unit_TRAIN_and_CONTROL':True},
        'normalization_fits':0,'prior_fits':0};seen=[]
    def prepare(p,root,**kwargs):seen.append((p,root,kwargs));return context
    monkeypatch.setattr(owner,'prepare_native_fit',prepare)
    plan={'mode':'ta_primary_main','metadata':{'parent_fit_plan':parent,
        'mechanics':{'normalization':normal,'prior':prior,'axis':context['axis']}}}
    assert owner.prepare_primary_native_fit(plan,tmp_path,reload_normalization=normal,reload_prior=prior)is context
    assert seen==[(parent,tmp_path,{'reload_normalization':normal,'reload_prior':prior})]
    context['normalization_fits']=1
    with pytest.raises(ValueError,match='PRIMARY_NO_STATISTICS_REFIT'):owner.prepare_primary_native_fit(plan,tmp_path)


def test_noise_fold_cutoff_censors_both_tasks_without_mutating_persisted_truth():
    from tests.test_market_state_native_m5_targets_v1 import build,source
    from gx1.contracts.market_state_ta_integrated_objective_v1 import _native_truth
    from gx1.contracts.market_state_native_m5_targets_v1 import noise_margin_dataset_contract,TICK_NS
    tape=build(*source(1000));issues=tape.issue_ns[500:520]
    original=tape.gather_next_bar_noise_margin(issues,fold_upper_ns=tape.cutoff_ns,margin_atr=.3)
    original.pop('noise_margin_contract');before={key:a.copy()for key,a in original.items()}
    upper=int(issues[-1])+TICK_NS
    actual=owner.noise_fold_truth(original,upper)
    expected=tape.gather_next_bar_noise_margin(issues,fold_upper_ns=upper,margin_atr=.3)
    expected.pop('noise_margin_contract')
    for key in expected:
        np.testing.assert_equal(actual[key],expected[key]);np.testing.assert_equal(original[key],before[key])
    contract=noise_margin_dataset_contract(.3)
    _native_truth(actual,rows=len(issues),fold_upper_ns=upper,
        target_contract_sha256=contract['contract_sha256'],target_contract=contract)
    assert (~actual['mature']).any()
    for task in ('open','close'):
        assert not actual['noise_'+task+'_known'][~actual['mature']].any()
        assert not actual['noise_'+task+'_eligible'][~actual['mature']].any()
        assert np.isnan(actual['noise_'+task+'_delta'][~actual['mature']]).all()


def test_noise_route_requires_own_contract_and_retains_primary_checkpoint_schedule():
    from gx1.contracts.market_state_native_m5_targets_v1 import noise_margin_dataset_contract
    contract=noise_margin_dataset_contract(.3)
    mechanical={'mode':'ta_noise_mechanics','metadata':{'target_contract':contract},
        'recipe':{'fit_batches':2,'passes':2,'batch_rows':16}}
    assert owner._ta_fit(mechanical)and owner._mechanics(mechanical['mode'])and owner._primary_fit(mechanical)
    assert owner.noise_target_kwargs(mechanical)=={'target_contract':contract}
    assert owner.noise_target_kwargs({'mode':'ta_primary_main'})=={}
    selected={'FIT':np.arange(1536,dtype=np.int64)}
    batches=owner._ta_training_batches(selected,mechanical)
    assert len(batches)==2
    np.testing.assert_array_equal(batches[0],batches[1])
    position=owner._ta_schedule_position(mechanical,1)
    assert position['next_pass']==1 and position['next_batch_in_pass']==0
    assert owner._ta_main('ta_noise_main')and not owner._mechanics('ta_noise_main')


def test_noise_prepare_and_fit_role_gates_precede_receipt_or_model_io(monkeypatch):
    monkeypatch.setattr(owner,'read',lambda *_:pytest.fail('IO before role validation'))
    with pytest.raises(ValueError,match='NOISE_PREPARE_INPUT_ROLES'):
        owner.build_noise_prepare_plan(inputs={},margin_atr=.3)
    with pytest.raises(ValueError,match='NOISE_FIT_MODE'):
        owner.build_noise_fit_plan(inputs={},recipe={},mode='ta_primary_main')


def test_noise_persisted_axes_preserve_raw_source_report_and_independent_masks(tmp_path,monkeypatch):
    from tests.test_market_state_native_m5_targets_v1 import build,source
    from gx1.contracts.market_state_native_m5_targets_v1 import noise_margin_dataset_contract
    tape=build(*source(1000));contract=noise_margin_dataset_contract(.3)
    truth=tape.gather_next_bar_noise_margin(tape.issue_ns,fold_upper_ns=tape.cutoff_ns,margin_atr=.3)
    truth.pop('noise_margin_contract');arrays={**truth,'source_row':tape.source_row}
    path=tmp_path/'noise.npz';np.savez_compressed(path,**arrays)
    dataset={'arrays':owner.ref(path),'issue_rows':len(tape.issue_ns),'cutoff_ns':tape.cutoff_ns,'headers':{key:{'dtype':a.dtype.str,'shape':list(a.shape),'sha256':owner.digest(a)}for key,a in arrays.items()}}
    rawpath=tmp_path/'raw.npz';np.savez_compressed(rawpath,source_row=tape.source_row,issue_ns=tape.issue_ns,
        **{'future_'+key:value for key,value in truth.items()if not key.startswith('noise_')})
    provider=SimpleNamespace(issue_ns=tape.issue_ns,source_row=tape.source_row,input_ready=np.ones(len(tape.issue_ns),bool))
    monkeypatch.setattr(owner,'FIT_START_NS',int(tape.issue_ns[500]))
    monkeypatch.setattr(owner,'FIT_END_NS',int(tape.issue_ns[540]))
    monkeypatch.setattr(owner,'CONTROL_END_NS',int(tape.issue_ns[580]))
    actual,report=owner.noise_fit_axes(provider,dataset,contract)
    old,old_report=owner.fit_axes(provider,owner.ref(rawpath))
    for name in('TRAIN','CONTROL'):
        for key,value in old_report[name].items():assert report[name][key]==value
        np.testing.assert_array_equal(actual[name]['rows'],old[name]['rows'])
        assert not actual[name]['truth']['noise_open_eligible'][actual[name]['truth']['open_known_at_issue']].any()
        assert np.all(actual[name]['truth']['noise_close_eligible']<=actual[name]['truth']['eligible'])
    assert any(report['TRAIN']['per_tf_eligible'][i]>report['TRAIN']['per_tf_noise_close_eligible'][i]for i in range(5))
