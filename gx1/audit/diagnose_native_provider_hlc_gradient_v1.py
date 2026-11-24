"""One frozen FINAL, full-B96 local gradient measurement; never an update."""
from __future__ import annotations
import argparse, gc, json, os, signal, subprocess, sys, time
from pathlib import Path
if __name__ == '__main__':
    from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
    install_native_m5_direction_io_firewall()
import numpy as np
import torch
from gx1.audit import diagnose_native_provider_hlc_readout_v1 as readout
from gx1.audit import native_future_state_provider_hlc_baseline_v1 as adapter
from gx1.contracts import market_state_ta_integrated_model_v1 as model_owner
from gx1.contracts.native_future_state_provider_hlc_v1 import SOURCE_BINDINGS

saved, core, train, objective = readout.saved, readout.core, readout.train, readout.objective
REPO, POLICY = readout.REPO, readout.POLICY
NAME='NATIVE_PROVIDER_HLC_FROZEN_GRADIENT_20261124_001'; STAGE=NAME.lower()
MODULE='gx1.audit.diagnose_native_provider_hlc_gradient_v1'
PLAN=REPO/'configs/research'/(NAME+'.json'); ROOT=Path('/home/andre2/GX1_RUNS')/NAME
SELF='gx1/audit/diagnose_native_provider_hlc_gradient_v1.py'; TEST='tests/test_diagnose_native_provider_hlc_gradient_v1.py'
DESIGN={'path':'/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/PROVIDER_HLC_FROZEN_GRADIENT_DESIGN001.json','sha256':'65e46fbe106ba61e242d912ce24bb76c942e08a91a9a14eb13ecb7b53c042488'}
CACHE_SCOPE={'path':'/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/PROVIDER_HLC_FROZEN_GRADIENT_CACHE_SCOPE001.json','sha256':'536e5bd5672ad33e00f64353272ed64e99473ba5a4064e648001a65266a4db6f'}
ANCHOR='b23a6ade432d34a41675e3eef4a61bbd173a577c'
EXECUTION={'class':'audit','memory':'4G','swap':'512M','numerical_threads':1,'outer_seconds':900,'internal_seconds':870,'kill_after_seconds':10}
CAPTURE=('issue_ns','activity_logits','side_logits'); TERMS=('canonical_total','independent_activity','independent_side','actual_auxiliary')
NOT_JSON={'feature_archive','capture','target_arrays','final_checkpoint','provider_response'}
TOLERANCES={'component_gradient_absolute_L2_floor':1e-12,'component_gradient_global_relative_L2_max':1e-5,'dot_identity_relative_error_max':1e-5,
    'no_post_failure_relaxation':True,'saved_FINAL_logits':{'atol':1e-6,'rtol':1e-5,'source_owner':'THROUGHPUT_EVAL_TOLERANCE; candidate fixed gate, not prequalified CPU/autograd parity'},'sign_margin_relative_to_normS_times_normTotal':1e-6}
ref, read, reference, source_clean = readout.ref, readout.read, readout.reference, readout.source_clean
require_capped_cpu_audit_execution=readout.require_capped_cpu_audit_execution
require_capped_lock_ancestry=readout.require_capped_lock_ancestry


def need(ok,why):
    if not bool(ok): raise ValueError('PROVIDER_HLC_FROZEN_GRADIENT_'+why)


def build_plan(*,source_base_commit=None):
    d=read(DESIGN)
    need(d['schema_version']=='gx1_provider_hlc_frozen_gradient_design_v1' and d['basename']==NAME and d['stage']==STAGE
         and d['module']==MODULE and d['plan_path']==str(PLAN) and d['output_root']==str(ROOT) and d['execution']==EXECUTION
         and d['source_parent_commit']==ANCHOR and d['execution_authorized']is False and d['training_authorized']is False
         and d['test_authorized']is False and d['automatic_retry_or_training']is False and d['tolerances']==TOLERANCES,'LOCKED_DESIGN')
    for binding in d['inputs'].values(): reference(binding)
    rp=read(d['inputs']['qualified_readout_plan'])
    need(rp==readout.build_plan(source_base_commit=rp['source_base_commit']),'QUALIFIED_READOUT_PLAN_UNCHANGED')
    rr=read(d['inputs']['qualified_readout_result']); rt=read(d['inputs']['qualified_readout_terminal']); acceptance=read(d['inputs']['qualified_readout_acceptance'])
    need(rr['status']=='COMPLETE_CONDITIONAL_PROVIDER_HLC_READOUT_ATTRIBUTION' and rr['source_unchanged']is True and rr['test_data_used']is False
         and rr['plan']==rt['plan']==d['inputs']['qualified_readout_plan'] and rt['status']=='COMPLETE' and rt['exit_code']==0 and rt['result']==d['inputs']['qualified_readout_result']
         and {k:acceptance['result'][k]for k in ('path','sha256')}==d['inputs']['qualified_readout_result'] and acceptance['actual_training_cause']=='UNKNOWN'
         and rr['report']['initial_full_tree_model_RNG_and_final_all_ten_proofs_verified']is True and acceptance['all20_approved']is False,'QUALIFIED_READOUT_COMPLETE')
    original=rp['metadata']['original_epoch_plan']; meta=original['metadata']; snapshot=read(d['inputs']['source_snapshot']); dataset=read(rp['inputs']['dataset']); quality=read(rp['inputs']['full_quality'])
    need(d['inputs']['source_snapshot']==meta['source_snapshot'] and d['inputs']['feature_archive']==snapshot['archive']
         and meta['input_binding']==rp['metadata']['initial_plan']['metadata']['input_binding'] and original['model_contract']['backbone_gradient_policy']['policy']=='M5_primary','SAME_ORIGINAL_GENUINE_INPUT_MODEL')
    roles=('original_epoch_plan','completed_model_plan','final_receipt','final_checkpoint','capture','dataset','target_arrays','certificate','prior','full_quality')
    inputs={**{key:rp['inputs'][key]for key in roles},**d['inputs'],'design':DESIGN,'normalization':meta['normalization'],'provider_response':SOURCE_BINDINGS['reference_raw'],'cache_scope':CACHE_SCOPE}
    cache_scope=read(CACHE_SCOPE); axes=snapshot['surfaces_metadata']['axes']; source_spec=snapshot['surfaces_metadata']['source_binding']['source_spec']
    need(cache_scope['schema_version']=='gx1_frozen_gradient_feature_cache_scope_clarification_v1' and cache_scope['design']==DESIGN
         and cache_scope['source_snapshot']==inputs['source_snapshot'] and cache_scope['feature_archive']==inputs['feature_archive']
         and cache_scope['execution_authorized']is False and cache_scope['training_authorized']is False and cache_scope['test_authorized']is False
         and cache_scope['is_budget_or_authority_expansion']is False
         and axes['issue_start_ns']==1609459200000000000 and axes['issue_end_ns']==source_spec['cutoff_ns']==1782864000000000000
         and axes['rows']==1215513 and axes['retained_local_rows']==389579,'QUALIFIED_PRETEST_CACHE_SCOPE')
    fields=train.provider_hlc_truth_fields(rp['target_contract']); headers={key:dataset['array_headers'][key]for key in fields}
    capture_headers={'issue_ns':rp['capture_array_headers']['issue_ns'],**{key:{'shape':[adapter.ROWS,5],'dtype':'<f4','sha256':quality['output_sha256'][key]}for key in CAPTURE[1:]}}
    budget=d['budget']
    need(budget['panel_rows']==budget['parity_forwards']==budget['gradient_forwards']==96 and budget['microbatch_rows']==1
         and budget['NN_forwards_max']==192 and budget['autograd_calls_max']==384 and tuple(budget['autograd_terms'])==TERMS
         and budget['checkpoint_deserializations']==budget['model_constructions']==budget['model_state_install_calls']==budget['full_feature_archive_loads']==budget['target_archive_loads']==budget['capture_archive_loads']==1
         and budget['checkpoint_serialized_bytes']==97914007 and budget['checkpoint_load_policy']=={'map_location':'cpu','weights_only':True,'fallback_or_add_safe_globals':False}
         and budget['feature_arrays_decoded']==len(snapshot['arrays'])==27 and budget['target_arrays_decoded']==len(fields)==14
         and set(budget['target_fields'])==set(fields) and tuple(budget['capture_fields'])==CAPTURE
         and budget['feature_ndarray_bytes_decoded']==sum(int(np.prod(h['shape']))*np.dtype(h['dtype']).itemsize for h in snapshot['arrays'].values())==1433216758
         and budget['target_ndarray_bytes_decoded']==sum(int(np.prod(h['shape']))*np.dtype(h['dtype']).itemsize for h in headers.values())==50700650
         and budget['capture_ndarray_bytes_decoded']==sum(int(np.prod(h['shape']))*np.dtype(h['dtype']).itemsize for h in capture_headers.values())==17018400
         and budget['packet_feature_fields']==9 and budget['packet_feature_bytes']==63407616 and budget['old_context_released_before_model_construction']is True
         and budget['cached_provider_raw_JSON_bytes']==614909 and budget['cached_provider_certificate_bytes']==3460 and budget['cached_provider_helper_array_bytes']==51600
         and budget['cached_provider_candle_records']==1290 and budget['cached_provider_price_scalars']==15480
         and all(budget[key]==0 for key in ('backward_calls','checkpoint_RNG_restore_calls','fits','new_source_HTTP_calls','optimizer_constructions','optimizer_restores','optimizer_updates','prior_refits','VAL_TEST_rows')),'EXACT_PAYLOAD_EXECUTION_BUDGET')
    panel=d['panel']
    need(panel=={'TRAIN_position_start':0,'TRAIN_position_stop_exclusive':96,'drop_or_replace_rows':False,'issue_ns_sha256':'8680d9bf04d705a5a83c5eb4d0ee9e75be0919aadd0f4389311e7e8463606299','logical_batch_index_zero_based':0,'rows':96,'selected_from_result_or_labels':False,'selection':'FIRST_ORIGINAL_CHRONOLOGICAL_B96_BEFORE_GRADIENT_MEASUREMENT','source_rows_sha256':'57b8afa403780c67f135f2f8e39e3066331a0ffaee34e7ad1fed5c921f2647fd'},'EXACT_PREDECLARED_B96')
    need(not subprocess.check_output(['git','diff',ANCHOR,'--','gx1','scripts','tests',':(exclude)'+SELF,':(exclude)'+TEST],cwd=REPO),'ALL_PREVIOUS_TRACKED_SOURCE_BYTES_UNCHANGED')
    for name,digest in rp['source_owners'].items(): need(saved.native.sha(REPO/name)==digest,'PREVIOUS_QUALIFIED_OWNER_UNCHANGED')
    names=set(rp['source_owners'])|{SELF,TEST}
    cp_metadata=readout.checkpoint_zip_metadata(inputs['final_checkpoint']); need(cp_metadata['serialized_bytes']==budget['checkpoint_serialized_bytes'],'EXACT_FINAL_CP_BYTES')
    plan={'schema_version':'gx1_provider_hlc_frozen_gradient_plan_v1','stage':STAGE,'status':'PREPARED_NOT_ADMITTED','plan_path':str(PLAN),'output_root':str(ROOT),
          'source_base_commit':source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'source_owners':{name:saved.native.sha(REPO/name)for name in sorted(names)},
          'cache_scope':cache_scope,'source_proof':{'all_previous_tracked_gx1_scripts_tests_byte_identical_to':ANCHOR,'qualified_readout_source_formula':rp['source_proof']},
          'inputs':inputs,'input_roles_not_JSON':sorted(NOT_JSON),'execution':EXECUTION,'budget':budget,'panel':panel,'tolerances':d['tolerances'],'method':d['method'],'question':d['question'],'limits':d['limits'],
          'metadata':{'original_epoch_plan':original,'trajectory':rp['metadata']['trajectory'],'final_receipt':rp['metadata']['final_receipt'],'adapter_metadata':meta},
          'model_contract':rp['model_contract'],'target_contract':rp['target_contract'],'truth_sha256':rp['truth_sha256'],'TRAIN_issue_ns_sha256':rp['TRAIN_issue_ns_sha256'],
          'feature_array_headers':snapshot['arrays'],'capture_array_headers':capture_headers,'target_array_headers':headers,'target_archive_inventory':list(dataset['array_headers']),
          'checkpoint_zip_metadata':cp_metadata,'runtime_versions':{'numpy':np.__version__,'torch':str(torch.__version__)}}
    checked_inputs(plan); return plan


def checked_inputs(plan):
    for binding in plan['inputs'].values(): reference(binding)
    saved.archive_headers(plan['inputs']['feature_archive'],plan['feature_array_headers'],exact_inventory=list(plan['feature_array_headers']))
    saved.archive_headers(plan['inputs']['capture'],plan['capture_array_headers'],exact_inventory=list(CAPTURE))
    saved.archive_headers(plan['inputs']['target_arrays'],plan['target_array_headers'],exact_inventory=plan['target_archive_inventory'])


def gate_expected(binding,admitted):
    return {'status':'ADMITTED_EXACT_CONDITIONAL_PROVIDER_HLC_FROZEN_GRADIENT'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}


def inspect(sha,*,admitted):
    need(Path.cwd().resolve()==REPO and type(sha)is str and len(sha)==64 and all(c in '0123456789abcdef'for c in sha),'CWD_SHA')
    binding={'path':str(PLAN),'sha256':sha}; plan=read(binding)
    need(plan==build_plan(source_base_commit=plan['source_base_commit']) and source_clean(plan),'EXACT_PLAN_SOURCE')
    pref=ref(POLICY); policy=read(pref)
    need(all(policy.get(key)is False for key in saved.FLAGS) and policy[STAGE]==gate_expected(binding,admitted),'ONE_USE_ALL_TRAINING_CLOSED')
    need(not ROOT.exists() and not ROOT.is_symlink(),'FRESH_ROOT'); return plan,binding,pref


def scaled_canonical_loss(loss,chunk_denominator,full_denominator):
    need(type(chunk_denominator)is int and type(full_denominator)is int and 0<=chunk_denominator<=full_denominator and full_denominator>0,'FULL_PANEL_DENOMINATOR')
    return loss*(chunk_denominator/full_denominator)


def accumulate_gradients(accumulator,gradients):
    need(len(accumulator)==len(gradients),'GRADIENT_PARAMETER_AXIS')
    for j,value in enumerate(gradients):
        if value is None: continue
        need(isinstance(value,torch.Tensor) and value.device.type=='cpu' and bool(torch.isfinite(value).all()),'FINITE_CPU_GRADIENT')
        value=value.detach().to(dtype=torch.float64)
        if accumulator[j]is None: accumulator[j]=value.clone()
        else:
            need(accumulator[j].dtype==torch.float64 and accumulator[j].shape==value.shape,'GRADIENT_ACCUMULATOR_SHAPE')
            accumulator[j].add_(value)
        need(bool(torch.isfinite(accumulator[j]).all()),'FINITE_ACCUMULATED_GRADIENT')
    return accumulator


def gradient_report(total,activity,side,auxiliary,*,tolerances):
    need(tolerances==TOLERANCES and len(total)==len(activity)==len(side)==len(auxiliary)>0,'GRADIENT_REPORT_AXIS_TOLERANCES')
    def combine(*vectors):
        result=[None]*len(total)
        for vector in vectors: accumulate_gradients(result,vector)
        return result
    reconstructed=combine(activity,side,auxiliary); difference=combine(total,[None if value is None else -value for value in reconstructed])
    zero=[None]*len(total); norm=lambda vector:core.shared_gradient_measures(vector,zero)['g5_l2']
    nt,ns,na,nx=map(norm,(total,side,activity,auxiliary)); error=norm(difference); floor=tolerances['component_gradient_absolute_L2_floor']
    need(nx==0.,'ACTUAL_AUXILIARY_SHARED_GRADIENT_NOT_ZERO')
    need(error<=max(floor,tolerances['component_gradient_global_relative_L2_max']*max(nt,norm(reconstructed))),'COMPONENT_GRADIENT_CONTRACT_MISMATCH')
    metrics=core.shared_gradient_measures(side,total); sa=core.shared_gradient_measures(side,activity); ss=ns*ns; dot=metrics['cross_dot']; expected=ss+sa['cross_dot']
    dot_error=abs(dot-expected); dot_scale=max(abs(dot),abs(expected),ns*nt)
    need(dot_error<=max(floor*floor,tolerances['dot_identity_relative_error_max']*dot_scale),'DOT_IDENTITY_CONTRACT_MISMATCH')
    margin=tolerances['sign_margin_relative_to_normS_times_normTotal']*ns*nt
    verdict='LOCAL_CONFLICT'if dot < -margin else'NOT_LOCAL_CONFLICT'if dot > margin else'INDETERMINATE'
    return {'component_gradient_contract_verified':True,'actual_auxiliary_shared_gradient_zero':True,'component_error_L2':error,'component_error_relative_L2':error/max(floor,nt,norm(reconstructed)),
            'L2':{'canonical_total':nt,'independent_activity':na,'independent_side':ns,'actual_auxiliary':nx},'gSide_dot_gTotal':dot,'gSide_dot_gActivity':sa['cross_dot'],'gSide_squared_L2':ss,
            'dot_identity_absolute_error':dot_error,'sign_margin':margin,'local_verdict':verdict,'shared_gradient_measures':sa,'shared_gradient_measure_roles':{'g5':'M5_side','gaux':'M5_activity'},
            'interpretation':'Infinitesimal shared-parameter -gTotal direction on this one frozen eval B96 only; not historical optimizer causation.'}


def measurement_bookend(model,state_before,rng_before):
    need(core.state_sha(model.state_dict())==state_before,'MEASUREMENT_MODEL_STATE_MUTATION')
    need(core.state_sha(core.shared_gradient_rng())==core.state_sha(rng_before),'MEASUREMENT_LOCAL_RNG_MUTATION')
    need(not model.training and all(parameter.grad is None for parameter in model.parameters()) and not torch.cuda.is_initialized(),'MEASUREMENT_MODE_GRAD_CUDA')
    return True


def safe_load_final(plan):
    reference(plan['inputs']['final_checkpoint'])
    cp=torch.load(plan['inputs']['final_checkpoint']['path'],map_location='cpu',weights_only=True)
    m=plan['metadata']; readout.validate_final(cp,m['final_receipt'],m['original_epoch_plan'],plan['inputs']['original_epoch_plan'],m['trajectory'],plan['inputs']['completed_model_plan'])
    return cp


def validate_counts(counts,budget):
    expected={'full_feature_archive_loads':1,'feature_arrays_decoded':27,'target_archive_loads':1,'target_arrays_decoded':14,'capture_archive_loads':1,'capture_arrays_decoded':3,
              'checkpoint_deserializations':1,'model_constructions':1,'model_state_install_calls':1,'parity_forwards':96,'gradient_forwards':96,'NN_forwards':192,'autograd_calls':384,
              'backward_calls':0,'optimizer_constructions':0,'optimizer_restores':0,'optimizer_updates':0,'checkpoint_RNG_restore_calls':0,'fits':0,'prior_refits':0,'new_source_HTTP_calls':0}
    need(counts==expected and all(type(value)is int for value in counts.values()) and counts['NN_forwards']==budget['NN_forwards_max'] and counts['autograd_calls']==budget['autograd_calls_max'],'EXACT_ACTUAL_COUNTS')
    return True


def calculate(plan,*,progress=None):
    torch.set_num_threads(1); need(torch.get_num_threads()==1 and not torch.cuda.is_initialized(),'CPU_THREADS_CUDA')
    progress={}if progress is None else progress; budget=plan['budget']; counts=dict.fromkeys(('full_feature_archive_loads','feature_arrays_decoded','target_archive_loads','target_arrays_decoded','capture_archive_loads','capture_arrays_decoded','checkpoint_deserializations','model_constructions','model_state_install_calls','parity_forwards','gradient_forwards','NN_forwards','autograd_calls','backward_calls','optimizer_constructions','optimizer_restores','optimizer_updates','checkpoint_RNG_restore_calls','fits','prior_refits','new_source_HTTP_calls'),0)
    progress.update(phase='CANONICAL_CONTEXT_AND_EXACT_B96',counts=counts)
    context=adapter.load_context(plan['metadata']['adapter_metadata']); counts.update(full_feature_archive_loads=1,feature_arrays_decoded=27,target_archive_loads=1,target_arrays_decoded=14)
    need(bool(np.all(context['provider'].bar_start_ns<1782864000000000000)),'CACHED_SOURCE_BARS_PRETEST_CUTOFF')
    take=np.arange(96,dtype=np.int64); packet=adapter.packet(context,take)
    need(core.digest(packet['truth']['issue_ns'])==plan['panel']['issue_ns_sha256'] and core.digest(context['source_rows'][take])==plan['panel']['source_rows_sha256']
         and bool(np.all((packet['truth']['issue_ns']>=train.START)&(packet['truth']['issue_ns']<train.UPPER))) and len(packet['truth']['issue_ns'])==96 and len(packet['features'])==9 and sum(v.numel()*v.element_size()for v in packet['features'].values())==budget['packet_feature_bytes'],'EXACT_ORIGINAL_B96_PACKET')
    normalization=context['normalization']; prior=context['prior']; binding=context['input_binding']; target=context['target_contract']; del context; gc.collect()
    progress['phase']='CONTEXT_RELEASED_CAPTURE3'
    capture=saved.load_fields(plan['inputs']['capture'],CAPTURE,plan['capture_array_headers']); counts.update(capture_archive_loads=1,capture_arrays_decoded=3)
    need(sum(v.nbytes for v in capture.values())==budget['capture_ndarray_bytes_decoded'] and core.digest(capture['issue_ns'])==plan['TRAIN_issue_ns_sha256'] and np.array_equal(capture['issue_ns'][:96],packet['truth']['issue_ns']),'SAVED_FINAL_CLOCK_BYTES')
    expected={key:np.ascontiguousarray(capture[key][:96]).copy()for key in CAPTURE[1:]}; del capture; gc.collect()
    progress['phase']='CPU_FACTORY_AND_STRICT_FINAL_LOAD'
    model=model_owner.build_native_future_state_model(input_normalization=normalization,input_binding=binding,architecture=plan['metadata']['original_epoch_plan']['metadata']['architecture'],device=torch.device('cpu'),train_prior=prior,target_contract=target,backbone_gradient_policy='M5_primary'); counts['model_constructions']+=1
    cp=safe_load_final(plan); counts['checkpoint_deserializations']+=1; model.load_state_dict(cp['model_state'],strict=True); counts['model_state_install_calls']+=1
    need(core.state_sha(model.state_dict())==plan['metadata']['final_receipt']['proof']['model_state'],'STRICT_INSTALLED_FINAL_MODEL_PROOF'); del cp; gc.collect(); model.eval()
    state_before=core.state_sha(model.state_dict()); rng_before=core.shared_gradient_rng()
    need(all(p.grad is None and p.device.type=='cpu' and p.dtype==torch.float32 for p in model.parameters()),'FRESH_PARAMETER_GRAD_NONE_CPU_F32')
    def count_forward(*_):
        counts['NN_forwards']+=1; need(counts['NN_forwards']<=budget['NN_forwards_max'],'FORWARD_BUDGET')
    hook=model.register_forward_pre_hook(count_forward); parity={key:0. for key in CAPTURE[1:]}
    def chunk(index):
        return {'features':{key:value[index:index+1]for key,value in packet['features'].items()},'truth':train.slice_provider_hlc_truth(packet['truth'],np.array([index],np.int64),target_contract=target)}
    try:
        with torch.enable_grad():
            progress['phase']='FULL96_AUTOGRAD_ENABLED_EVAL_PARITY_BEFORE_DERIVATIVES'
            for index in range(96):
                small=chunk(index); output=model(**small['features']); counts['parity_forwards']+=1
                for key in CAPTURE[1:]:
                    value=output[key].detach().cpu().numpy(); wanted=expected[key][index:index+1]; delta=float(np.max(np.abs(value-wanted)))
                    parity[key]=max(parity[key],delta); need(np.allclose(value,wanted,atol=TOLERANCES['saved_FINAL_logits']['atol'],rtol=TOLERANCES['saved_FINAL_logits']['rtol']),'SAVED_FINAL_LOGIT_PARITY:'+key)
                del output,small
            measurement_bookend(model,state_before,rng_before)
            progress['phase']='FULL96_CANONICAL_AND_INDEPENDENT_COMPONENT_GRADIENTS'
            named=tuple((name,p)for name,p in model.backbone.named_parameters()if p.requires_grad); params=tuple(p for _,p in named); need(bool(params),'SHARED_PARAMETER_AXIS')
            sums={term:[None]*len(params)for term in TERMS}; denominator=int(packet['truth']['activity_known'].sum()); need(denominator>0,'NONEMPTY_FULL_B96_DENOMINATOR')
            scalar_sums={term:0. for term in TERMS}; support={'activity':packet['truth']['activity_known'].sum(axis=0).astype(int).tolist(),'side':packet['truth']['side_known'].sum(axis=0).astype(int).tolist()}
            for index in range(96):
                small=chunk(index); output=model(**small['features']); counts['gradient_forwards']+=1
                canonical=objective.native_future_state_loss(model_output=output,truth=small['truth'],fold_upper_ns=train.UPPER,target_contract=target)
                values={}
                for name,key,label,mask in (('activity','activity_logits','activity','activity_known'),('side','side_logits','side_up','side_known')):
                    y=torch.as_tensor(small['truth'][label].copy(),dtype=output[key].dtype); m=torch.as_tensor(small['truth'][mask].copy(),dtype=torch.bool)
                    bce=torch.nn.functional.binary_cross_entropy_with_logits(output[key],y,reduction='none'); values[name]=torch.where(m,bce,torch.zeros_like(bce))
                terms=(scaled_canonical_loss(canonical['loss'],canonical['count'],denominator),values['activity'][:,0].sum()/denominator,values['side'][:,0].sum()/denominator,(values['activity'][:,1:].sum()+values['side'][:,1:].sum())/denominator)
                for j,(name,loss)in enumerate(zip(TERMS,terms)):
                    need(bool(torch.isfinite(loss)),'FINITE_COMPONENT_LOSS'); gradients=torch.autograd.grad(loss,params,retain_graph=j<3,allow_unused=True); counts['autograd_calls']+=1
                    need(counts['autograd_calls']<=budget['autograd_calls_max'],'AUTOGRAD_BUDGET'); accumulate_gradients(sums[name],gradients); scalar_sums[name]+=float(loss.detach()); del gradients
                del output,small,canonical,values,terms,loss
            proof=gradient_report(*(sums[name]for name in TERMS),tolerances=plan['tolerances']); del sums
        bookend=measurement_bookend(model,state_before,rng_before); validate_counts(counts,budget)
    finally: hook.remove()
    progress['phase']='COMPLETE_FROZEN_FULL_B96_GRADIENT'
    return {'payload_budget':budget,'counts':counts,'cache_scope':plan['cache_scope'],'panel':plan['panel'],'known_activity_denominator_full_B96':denominator,'per_TF_support':support,'normalized_loss_components':scalar_sums,
            'logit_parity':{'passed':True,'rows':96,'both_heads':True,'autograd_enabled_eval_path':True,'max_absolute_error':parity,'tolerance':plan['tolerances']['saved_FINAL_logits']},
            'gradient_proof':proof,'parameter_axis':{'names':[name for name,_ in named],'tensors':len(named),'aggregation_dtype':'float64','None_is_mathematical_zero_only':True},
            'old_context_released_before_model_construction':True,'final_all_ten_checkpoint_proofs_verified':True,'strict_FINAL_model_state_installed':True,'model_state_and_local_RNG_bookend_passed':bookend,'local_RNG_bookend_scope':['python','numpy','torch_cpu'],'all_parameter_grad_fields_None':True,
            'constructor_RNG_scope':'Constructor consumes diagnostic-local RNG; measurement bookend starts after construction, FINAL load and eval. Checkpoint RNG never restored.',
            'actual_training_cause':'UNKNOWN','selected_causal_fix':'UNKNOWN','later_generalization':'UNKNOWN','economic_edge':'UNKNOWN','all20_approved':False,'limits':plan['limits']}


def publish(name,value):
    path=ROOT/name; saved._publish_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()); return ref(path)


def run(sha):
    began=time.monotonic(); cap=require_capped_cpu_audit_execution(); lock=require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==4*1024**3 and cap['swap']==512*1024**2 and cap['numerical_threads']==1 and cap['pids']==64
         and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):],'CPU_CAPS')
    plan,binding,pref=inspect(sha,admitted=True); checked_inputs(plan); boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(); ROOT.mkdir(parents=True,exist_ok=False)
    handlers={sig:signal.getsignal(sig)for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}; claim=result=None; progress={'phase':'CLAIM'}
    try:
        def expired(*_):raise TimeoutError('PROVIDER_HLC_FROZEN_GRADIENT_WALL')
        def stopped(sig,*_):raise InterruptedError('PROVIDER_HLC_FROZEN_GRADIENT_SIGNAL_'+str(sig))
        signal.signal(signal.SIGALRM,expired); signal.signal(signal.SIGTERM,stopped); signal.signal(signal.SIGINT,stopped)
        claim=publish('CLAIM.json',{'stage':STAGE,'plan':binding,'policy_at_claim':pref,'boot_id':boot,'remaining_invocations':0,'relaunch_allowed':False,'training_authorized':False,'test_data_used':False})
        remaining=EXECUTION['internal_seconds']-(time.monotonic()-began); need(remaining>0,'TIME_BEFORE_DECODE'); signal.setitimer(signal.ITIMER_REAL,remaining)
        allowed={plan['inputs'][role]['path']for role in ('feature_archive','capture','target_arrays','final_checkpoint')}
        def audit(event,args):
            if event in ('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('PROVIDER_HLC_FROZEN_GRADIENT_NETWORK_FORBIDDEN')
            if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
                name=os.fsdecode(args[0])
                if name.endswith(('.npz','.pt','.pth','.parquet','.safetensors','.pkl'))and str(Path(name).resolve())not in allowed:raise RuntimeError('PROVIDER_HLC_FROZEN_GRADIENT_PAYLOAD_FORBIDDEN')
        sys.addaudithook(audit); report=calculate(plan,progress=progress); checked_inputs(plan)
        need(source_clean(plan)and ref(POLICY)==pref and ref(PLAN)==binding and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,'SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND')
        result=publish('RESULT.json',{'status':'COMPLETE_CONDITIONAL_PROVIDER_HLC_FROZEN_GRADIENT','stage':STAGE,'plan':binding,'claim':claim,'policy_at_claim':pref,'boot_id':boot,'report':report,'source_unchanged':True,'test_data_used':False,'economic_edge_proven':False,'elapsed_seconds':time.monotonic()-began})
        terminal=publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'stage':STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False}); return {'result':result,'terminal':terminal}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0); publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'stage':STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'progress':progress,'error':type(error).__name__+':'+str(error)[:2000],'test_data_used':False}); raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in handlers.items():signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--plan-sha256',required=True); parser.add_argument('--preflight',action='store_true'); args=parser.parse_args()
    need(sys.argv[1:]==['--plan-sha256',args.plan_sha256]+(['--preflight']if args.preflight else[]),'EXACT_CLI')
    if args.preflight:
        plan,binding,pref=inspect(args.plan_sha256,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_PROVIDER_HLC_FROZEN_GRADIENT_PASS','plan':binding,'policy':pref,'claim_created':False,'checkpoint_deserializations':0,'NPZ_arrays_decoded':0,'models':0,'NN_forwards':0,'autograd_calls':0,'CUDA_initialized':False}))
    else:print(json.dumps(run(args.plan_sha256),sort_keys=True))


if __name__=='__main__':main()
