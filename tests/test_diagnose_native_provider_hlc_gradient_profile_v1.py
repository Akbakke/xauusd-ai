"""Synthetic scalar/axis/parity guards for one fixed native B96 profile; no real payloads."""
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from gx1.audit import diagnose_native_provider_hlc_gradient_profile_v1 as m
from gx1.contracts.native_future_state_provider_hlc_v1 import provider_hlc_state_contract

TF=('M5','M15','H1','H4','D1')
LOSS_TERMS=('canonical','activity_all',*('side_'+tf for tf in TF))


def scalar_fixture(*,all_sides=False):
    n=96;target=provider_hlc_state_contract();tick=300_000_000_000
    issues=int(m.train.START)+np.arange(n,dtype=np.int64)*tick
    duration=np.array([target['tf_duration_ns'][tf]for tf in TF],np.int64)
    origin=np.array([target['tf_origin_ns'][tf]for tf in TF],np.int64)
    start=((issues[:,None]-origin+duration-1)//duration)*duration+origin
    known=np.tile(np.array([[1,1,1,1,1],[1,1,0,1,1],[1,0,1,1,1],[1,1,0,1,1]],bool),(24,1))
    side=known.copy()if all_sides else np.tile(np.array([[1,1,1,1,0],[1,0,0,1,0],[1,0,0,1,0],[1,1,0,0,0]],bool),(24,1))
    truth={'issue_ns':issues,'target_start_ns':start.astype(np.int64),'validity_ns':(start+duration).astype(np.int64),
           **{k:np.ones((n,5),bool)for k in ('target_present','mature','nominal_complete','state_window_complete','state_hlc_observed')},
           **{k:np.zeros((n,5),bool)for k in ('partial_range_active','provider_hlc_observed')},
           'activity_known':known,'activity':known.copy(),'side_known':side,
           'side_up':side&(np.arange(n)[:,None]%2==0),
           'target_observation_certificate_ref':{'path':'/synthetic/certificate.json','sha256':'a'*64,
               'target_contract_sha256':target['contract_sha256'],'source_binding_sha256':target['provider_hlc_source_binding_sha256']}}
    logits=torch.arange(n*5,dtype=torch.float32).reshape(n,5)/1000-.3
    output={'activity_logits':logits.clone().requires_grad_(),'side_logits':(-logits).clone().requires_grad_()}
    return output,truth,target


def test_actual_v2_canonical_components_keep_one_full_activity_denominator():
    output,truth,target=scalar_fixture();terms,support=m.profile_component_terms(output,truth,target)
    assert tuple(terms)==LOSS_TERMS+tuple('mean_q_known_'+tf for tf in TF[:-1])
    D=int(truth['activity_known'].sum());assert D==408
    assert support=={'activity':truth['activity_known'].sum(0).astype(int).tolist(),
                     'side':truth['side_known'].sum(0).astype(int).tolist(),'activity_denominator':D}
    activity=torch.nn.functional.binary_cross_entropy_with_logits(output['activity_logits'],torch.from_numpy(truth['activity']).float(),reduction='none')
    side=torch.nn.functional.binary_cross_entropy_with_logits(output['side_logits'],torch.from_numpy(truth['side_up']).float(),reduction='none')
    assert torch.allclose(terms['activity_all'],activity[torch.from_numpy(truth['activity_known'])].sum()/D)
    for j,tf in enumerate(TF):
        expected=side[:,j][torch.from_numpy(truth['side_known'][:,j])].sum()/D
        assert torch.allclose(terms['side_'+tf],expected)
    independent=terms['activity_all']+sum(terms['side_'+tf]for tf in TF)
    assert torch.allclose(terms['canonical'],independent,atol=1e-6,rtol=1e-5)
    for actual,expected in zip(torch.autograd.grad(terms['canonical'],tuple(output.values()),retain_graph=True),
                               torch.autograd.grad(independent,tuple(output.values()),retain_graph=True)):
        assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-5)


def test_known_mean_q_uses_float64_sigmoid_and_its_own_side_denominator():
    output,truth,target=scalar_fixture();terms,_=m.profile_component_terms(output,truth,target)
    mask=torch.from_numpy(truth['side_known'][:,1]);q=torch.sigmoid(output['side_logits'].double()[:,1])
    value=terms['mean_q_known_M15'];assert value.dtype==torch.float64
    assert torch.equal(value,q[mask].mean()) and mask.sum()==48
    gradient=torch.autograd.grad(value,output['side_logits'],retain_graph=True)[0]
    expected=torch.zeros_like(output['side_logits']);expected[mask,1]=(q[mask]*(1-q[mask])/int(mask.sum())).float()
    assert torch.allclose(gradient,expected,atol=1e-8,rtol=1e-6)
    assert torch.count_nonzero(gradient[:,0])==torch.count_nonzero(gradient[~mask,1])==0


def test_empty_side_has_connected_zero_loss_and_no_invented_mean_q():
    output,truth,target=scalar_fixture();terms,support=m.profile_component_terms(output,truth,target)
    assert support['side'][-1]==0 and 'mean_q_known_D1'not in terms
    assert terms['side_D1'].requires_grad and float(terms['side_D1'])==0
    grad=torch.autograd.grad(terms['side_D1'],output['side_logits'])[0]
    assert torch.equal(grad,torch.zeros_like(grad))


def test_all_known_side_masks_yield_seven_loss_and_five_mean_derivative_terms():
    output,truth,target=scalar_fixture(all_sides=True);terms,support=m.profile_component_terms(output,truth,target)
    assert tuple(terms)==LOSS_TERMS+tuple('mean_q_known_'+tf for tf in TF)
    assert len(terms)==12 and all(n>0 for n in support['side'])


def test_no_activity_support_cannot_create_a_diagnostic_gradient():
    output,truth,target=scalar_fixture()
    for key in ('activity_known','activity','side_known','side_up'):truth[key].fill(False)
    with pytest.raises(ValueError):m.profile_component_terms(output,truth,target)


@pytest.mark.parametrize('task',['activity_logits','side_logits'])
@pytest.mark.parametrize('bad',['shape','nan','inf'])
def test_component_domain_failures_precede_any_derivative(task,bad):
    output,truth,target=scalar_fixture()
    if bad=='shape':output[task]=output[task][:,:4]
    else:
        output[task]=output[task].detach().clone();output[task][0,0]=float(bad);output[task].requires_grad_()
    with pytest.raises(ValueError):m.profile_component_terms(output,truth,target)


def named_fixture():
    return [('backbone.shared',torch.nn.Parameter(torch.zeros(2))),
            ('readout.activity_heads.M5.weight',torch.nn.Parameter(torch.zeros(1))),
            ('readout.side_heads.M5.weight',torch.nn.Parameter(torch.zeros(1)))]


def vectors():
    v=lambda a,b,c:(torch.tensor(a,dtype=torch.float64),None if b is None else torch.tensor([b],dtype=torch.float64),None if c is None else torch.tensor([c],dtype=torch.float64))
    loss={'canonical':v([1,1],2,3),'activity_all':v([1,0],2,None),
          'side_M5':v([0,1],None,3),**{'side_'+tf:(None,None,None)for tf in TF[1:]}}
    q={tf:None for tf in TF};q['M5']=v([1,2],None,1)
    return loss,q


def test_full_named_axis_is_disjoint_backbone_readout_and_named_heads():
    axes=m.profile_parameter_axes(named_fixture())
    assert axes['full']==[0,1,2]and axes['backbone']==[0]and axes['readout']==[1,2]
    assert axes['head:activity.M5']==[1]and axes['head:side.M5']==[2]
    assert sorted(axes['backbone']+axes['readout'])==axes['full']


@pytest.mark.parametrize('bad',['duplicate','foreign'])
def test_axis_does_not_silently_drop_or_duplicate_model_parameters(bad):
    named=named_fixture()
    if bad=='duplicate':named.append(named[0])
    else:named.append(('foreign.weight',torch.nn.Parameter(torch.zeros(1))))
    with pytest.raises(ValueError):m.profile_parameter_axes(named)


def test_gradient_report_preserves_components_axes_and_negative_loss_direction():
    loss,q=vectors();axes=m.profile_parameter_axes(named_fixture());report=m.profile_gradient_report(loss,q,axes)
    assert report['component_identity']is True and report['component_error_L2']==0
    assert report['canonical_L2']==pytest.approx(15**.5)
    tf=report['per_TF']['M5'];assert tf['status']=='MEASURED'
    full=tf['axes']['full'];backbone=tf['axes']['backbone'];readout=tf['axes']['readout']
    assert full['gQ_dot_gLoss']==6 and full['minus_gLoss_directional_derivative']==-6
    assert backbone['gQ_dot_gLoss']==readout['gQ_dot_gLoss']==3
    assert full['gQ_dot_gLoss']==sum(full['gQ_dot_components'].values())
    assert full['gQ_dot_components']['activity_all']==1 and full['gQ_dot_components']['side_M5']==5
    assert full['gQ_L2']==pytest.approx(6**.5)
    for tf in TF[1:]:assert report['per_TF'][tf]['status']=='UNKNOWN_NO_SIDE_SUPPORT'
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize('bad',['component','missing','extra','nan','tolerance'])
def test_broken_component_or_q_proof_cannot_produce_a_direction(bad):
    loss,q=vectors();axes=m.profile_parameter_axes(named_fixture());kwargs={}
    if bad=='component':loss['canonical']=(torch.tensor([2.,1.],dtype=torch.float64),*loss['canonical'][1:])
    elif bad=='missing':del loss['side_H4']
    elif bad=='extra':loss['other']=(None,None,None)
    elif bad=='nan':q['M5'][0][0]=float('nan')
    else:kwargs={'atol':1e-3}
    with pytest.raises(ValueError):m.profile_gradient_report(loss,q,axes,**kwargs)


@pytest.mark.parametrize('side,expected',[([96]*5,12),([0]*5,7),([96,0,2,0,1],10)])
def test_actual_derivative_budget_is_seven_plus_nonempty_known_side_masks(side,expected):
    budget=m.profile_plan_budget(side)
    assert budget['autograd_calls_max']==12 and budget['actual_autograd_calls']==expected


@pytest.mark.parametrize('side',[[0]*4,[0]*6,[True,0,0,0,0],[-1,0,0,0,0],[97,0,0,0,0],[1.,0,0,0,0]])
def test_support_budget_rejects_ambiguous_or_out_of_B96_counts(side):
    with pytest.raises(ValueError):m.profile_plan_budget(side)


@pytest.mark.parametrize('task',['activity_logits','side_logits'])
def test_both_heads_use_the_fixed_original_parity_before_derivatives(task):
    reference={k:np.zeros((96,5),np.float32)for k in ('activity_logits','side_logits')}
    candidate={k:v.copy()for k,v in reference.items()};candidate[task][0,0]=1e-7
    proof=m.profile_parity(reference,candidate);assert all(v['qualified']for v in proof.values())
    candidate[task][0,0]=1e-3
    with pytest.raises(ValueError):m.profile_parity(reference,candidate)


@pytest.mark.parametrize('bad',['nan','shape','dtype'])
def test_parity_domain_corruption_has_no_gradient_fallback(bad):
    reference={k:np.zeros((96,5),np.float32)for k in ('activity_logits','side_logits')};candidate={k:v.copy()for k,v in reference.items()}
    if bad=='nan':candidate['side_logits'][0,0]=float('nan')
    elif bad=='shape':candidate['side_logits']=candidate['side_logits'][:95]
    else:candidate['side_logits']=candidate['side_logits'].astype(np.float64)
    with pytest.raises(ValueError):m.profile_parity(reference,candidate)


def test_runtime_scope_allows_only_grad_not_parameter_grad_accumulation_or_update():
    p=torch.nn.Parameter(torch.tensor(2.));before=torch.get_rng_state().clone();original=torch.autograd.grad
    with m.profile_runtime_blockers():
        grad=torch.autograd.grad(p*p,(p,),create_graph=False)[0]
        assert grad.item()==4 and p.grad is None and torch.autograd.grad is original
    assert p.grad is None and torch.equal(before,torch.get_rng_state())


@pytest.mark.parametrize('action',['tensor_backward','autograd_backward','optimizer','restore','rng_restore','safe_globals'])
def test_runtime_scope_rejects_update_backward_and_restore_routes(action):
    p=torch.nn.Parameter(torch.tensor(2.));original=torch.Tensor.backward
    with m.profile_runtime_blockers():
        with pytest.raises(RuntimeError):
            if action=='tensor_backward':(p*p).backward()
            elif action=='autograd_backward':torch.autograd.backward(p*p)
            elif action=='optimizer':torch.optim.AdamW([p])
            elif action=='restore':m.train.strict_restore()
            elif action=='rng_restore':m.core.restore_training_rng({})
            else:torch.serialization.add_safe_globals([])
    assert p.grad is None and torch.Tensor.backward is original


def test_three_bound_forward_routes_and_IO_budget_are_single_panel_only():
    assert m.FORWARD_PHASES==({'rows':256,'grouped':True,'equal_tf':True,'grad_enabled':False},
                              {'rows':96,'grouped':True,'equal_tf':False,'grad_enabled':False},
                              {'rows':96,'grouped':True,'equal_tf':False,'grad_enabled':True})
    budget=m.profile_plan_budget()
    assert budget['native_model_forwards']==budget['native_backbone_forwards']==budget['native_readout_forwards']==3
    assert budget['native_forward_issue_rows']==448 and budget['autograd_calls_max']==12
    assert (budget['NPZ_archive_loads'],budget['NPZ_arrays_decoded'],budget['NPZ_decoded_bytes'])==(3,44,1500935808)
    assert budget['checkpoint_deserializations']==budget['model_constructions']==budget['model_state_installs']==1
    for name in ('HTTP_calls','RNG_restores','VAL_TEST_rows','backward_calls','new_labels','normalization_fits',
                 'optimizer_constructions','optimizer_restores','optimizer_updates','prior_fits'):
        assert budget[name]==0


def test_physical_uses_one_authentic_state_and_both_parity_gates_before_any_autograd():
    import ast
    source=inspect.getsource(m.physical);tree=ast.parse(source)
    loads=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and ast.unparse(n.func)=='torch.load']
    assert len(loads)==1 and {kw.arg:ast.literal_eval(kw.value)for kw in loads[0].keywords}=={'map_location':'cpu','weights_only':True}
    calls=[ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)]
    assert calls.count('model_owner.build_native_future_state_model')==1
    assert 'readout.validate_final'in calls and 'endpoint.strict_model_install'in calls
    assert calls.count('profile_parity')==5
    assert source.rindex('profile_parity(')<source.index('profile_component_terms(')<source.index('torch.autograd.grad(')
    derivative=next(n for n in ast.walk(tree)if isinstance(n,ast.Call)and ast.unparse(n.func)=='torch.autograd.grad')
    keywords={kw.arg:kw.value for kw in derivative.keywords}
    assert ast.literal_eval(keywords['create_graph'])is False and ast.literal_eval(keywords['allow_unused'])is True
    assert 'len(terms)'in ast.unparse(keywords['retain_graph'])
    assert not any(name.endswith(('.backward','.step','.load_state_dict'))or'restore'in name for name in calls)
    assert "device='cpu',dtype=torch.float64"in source and 'del gradients'in source


@pytest.mark.parametrize('admitted',[False,True])
def test_profile_gate_is_exact_one_use_without_training_or_relaunch(admitted):
    binding={'path':'/synthetic/plan.json','sha256':'a'*64};gate=m.gate_expected(binding,admitted)
    assert gate['plan']==binding and gate['execution_authorized']is admitted and gate['remaining_invocations']==int(admitted)
    assert gate['relaunch_allowed']is gate['training_authorized']is gate['test_authorized']is False


def test_exact_preflight_has_no_payload_model_or_claim(monkeypatch,capsys):
    sha='a'*64;seen=[];binding={'path':'/synthetic/plan.json','sha256':sha};pref={'path':'/synthetic/policy.json','sha256':'b'*64}
    monkeypatch.setattr(m.sys,'argv',['profile','--plan-sha256',sha,'--preflight'])
    monkeypatch.setattr(m,'require_scope',lambda value,**kw:seen.append((value,kw))or{'synthetic_guard':True})
    monkeypatch.setattr(m,'inspect',lambda value,**kw:seen.append(('inspect',value,kw))or({},binding,pref,{'cuda_initialized':False}))
    monkeypatch.setattr(m,'run',lambda *_:pytest.fail('preflight may not run'))
    m.main();report=json.loads(capsys.readouterr().out)
    assert report['claim_created']is report['CUDA_initialized']is False
    assert all(report[k]==0 for k in ('checkpoint_deserializations','NPZ_arrays_decoded','models','NN_forwards'))
    assert seen==[(sha,{'preflight':True}),('inspect',sha,{'admitted':False})]


def lifecycle(monkeypatch,tmp_path):
    repo=tmp_path/'repo';repo.mkdir();runs=tmp_path/'runs';runs.mkdir();planpath=repo/'plan.json';planpath.write_text('{}')
    policy=repo/'NEXT_RUN_POLICY.json';policy.write_text('{}');(repo/'CURRENT_HANDOVER.md').write_text('synthetic frozen handover\n')
    monkeypatch.setattr(m,'REPO',repo);monkeypatch.setattr(m,'ROOT',runs/m.NAME);monkeypatch.setattr(m,'PLAN',planpath);monkeypatch.setattr(m,'POLICY',policy)
    plan={'inputs':{},'input_roles_not_JSON':[]};binding=m.ref(planpath);pref=m.ref(policy);checks=[];timers=[]
    original_read_text=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda path,*a,**kw:'synthetic-wsl-boot'if str(path)=='/proc/sys/kernel/random/boot_id'else original_read_text(path,*a,**kw))
    live={'wsl_boot_id':'synthetic-wsl-boot','physical_windows_boot_id':7}
    monkeypatch.setattr(m,'require_scope',lambda _: {'resource_proof_only':True})
    monkeypatch.setattr(m,'inspect',lambda _,**kw:(plan,binding,pref,live))
    monkeypatch.setattr(m,'checked_inputs',lambda value:checks.append(value))
    monkeypatch.setattr(m.readout,'source_clean',lambda _:True)
    monkeypatch.setattr(m.core,'physical_windows_boot_id',lambda:7)
    monkeypatch.setattr(m.core,'future_cpu_process_identity',lambda:{'synthetic_pid':123})
    monkeypatch.setattr(m.core.common,'firewall_state',lambda:{'denied_attempts':0})
    monkeypatch.setattr(m.signal,'signal',lambda *_:None)
    monkeypatch.setattr(m.signal,'setitimer',lambda *args:timers.append(args))
    return binding,pref,checks,timers


@pytest.mark.parametrize('failure',[False,True])
def test_success_and_failure_terminal_preserve_one_use_and_actual_progress(monkeypatch,tmp_path,failure):
    binding,pref,checks,timers=lifecycle(monkeypatch,tmp_path)
    def physical(plan,live,*,progress):
        progress.update(phase='SYNTHETIC_PARITY_GATE',counts={'native_model_forwards':2,'autograd_calls':0})
        if failure:raise ValueError('synthetic parity failure before derivatives')
        return {'synthetic_report':True,'economic_edge':'UNKNOWN','full20_approved':False}
    monkeypatch.setattr(m,'physical',physical)
    if failure:
        with pytest.raises(ValueError,match='synthetic parity failure'):m.run(binding['sha256'])
    else:
        result=m.run(binding['sha256']);assert result['result']==m.ref(m.ROOT/'RESULT.json')
    terminal=json.loads((m.ROOT/'TERMINAL.json').read_text());claim=json.loads((m.ROOT/'CLAIM.json').read_text())
    assert terminal['status']==('FAILED_CONSUMED'if failure else'COMPLETE')and terminal['exit_code']==int(failure)
    assert terminal['plan']==claim['plan']==binding and claim['policy_at_claim']==pref
    assert terminal['remaining_invocations']==claim['remaining_invocations']==0
    assert terminal['relaunch_allowed']is claim['training_authorized']is False
    assert len(checks)==(1 if failure else 2)and timers[-1][1]==0
    if failure:
        assert terminal['result']is None and terminal['progress']['counts']=={'native_model_forwards':2,'autograd_calls':0}
        assert not (m.ROOT/'RESULT.json').exists()
    before=(m.ROOT/'TERMINAL.json').read_bytes()
    with pytest.raises(FileExistsError):m.run(binding['sha256'])
    assert (m.ROOT/'TERMINAL.json').read_bytes()==before


def test_python_scope_binds_exact_profile_environment_CLI_fresh_root_and_common_guard(monkeypatch,tmp_path):
    repo=tmp_path/'repo';(repo/'configs/research').mkdir(parents=True);runs=tmp_path/'runs';runs.mkdir()
    monkeypatch.setattr(m,'REPO',repo);monkeypatch.setattr(m,'PLAN',repo/'configs/research'/m.PLAN.name);monkeypatch.setattr(m,'ROOT',runs/m.NAME);monkeypatch.chdir(repo)
    seen=[];monkeypatch.setattr(m,'require_guarded_cuda_trainer_execution',lambda:seen.append(True)or{'common_guard':True})
    sha='e'*64;env={k:sha if v=='SHA'else v for k,v in m.ENV.items()};argv=[str(repo/'.venv/bin/python'),'-m',m.MODULE,'--plan-sha256',sha]
    assert m.require_scope(sha,environ=env,argv=argv)=={'common_guard':True}
    assert m.require_scope(sha,preflight=True,environ=env,argv=argv+['--preflight'])=={'common_guard':True}
    for key in ('GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE','GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_PLAN_SHA256','GX1_CAPPED_MEMORY_BYTES','OMP_NUM_THREADS'):
        with pytest.raises(ValueError):m.require_scope(sha,environ={**env,key:env[key]+'wrong'},argv=argv)
    with pytest.raises(ValueError):m.require_scope(sha,environ={**env,'GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_LANE':''},argv=argv)
    for command in (argv+['--lane','main'],argv[:2]+[m.endpoint.MODULE]+argv[3:],argv[:3]+['--plan-sha256='+sha]):
        with pytest.raises(ValueError):m.require_scope(sha,environ=env,argv=command)
    m.ROOT.mkdir()
    with pytest.raises(ValueError):m.require_scope(sha,environ=env,argv=argv)
    assert seen==[True,True]
