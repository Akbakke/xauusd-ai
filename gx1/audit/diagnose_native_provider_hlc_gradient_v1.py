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
                    passed_mask=np.isclose(value,wanted,atol=TOLERANCES['saved_FINAL_logits']['atol'],rtol=TOLERANCES['saved_FINAL_logits']['rtol'])
                    if not bool(passed_mask.all()):
                        absolute_delta=np.abs(value.astype(np.float64)-wanted.astype(np.float64))
                        json_number=lambda number:float(number)if np.isfinite(number)else str(float(number))
                        progress['parity_failure']={'index':index,'issue_ns':int(small['truth']['issue_ns'][0]),'head':key,'timeframes':list(train.TF),
                            'observed_logits':[json_number(number)for number in value[0]],'expected_logits':[json_number(number)for number in wanted[0]],
                            'absolute_delta':[json_number(number)for number in absolute_delta[0]],'max_absolute_delta':json_number(absolute_delta.max()),
                            'nonfinite_values':not bool(np.isfinite(value).all()and np.isfinite(wanted).all()),
                            'fixed_tolerance':dict(TOLERANCES['saved_FINAL_logits']),'passed_mask':passed_mask[0].tolist()}
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


# First-row parity measurement; the consumed gradient route stays unchanged.
PARITY_NAME='NATIVE_PROVIDER_HLC_FIRST_PARITY_20261124_001'; PARITY_STAGE=PARITY_NAME.lower()
PARITY_PLAN=REPO/'configs/research'/(PARITY_NAME+'.json'); PARITY_ROOT=Path('/home/andre2/GX1_RUNS')/PARITY_NAME
PARITY_DESIGN={'path':'/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/PROVIDER_HLC_FIRST_PARITY_DESIGN001.json','sha256':'82721b760e9a6d68b8767dae7da3b6ae123950be08dd58280b4aa2ae4f640800'}
PARITY_PARENT='3231f1d5961e1c8eeb4d14e2b0e17fa48dc9b15b'; PARITY_EXECUTION=dict(EXECUTION)
PARITY_ROUTES=tuple({'name':name,'grad_enabled':grad,'equal_tf':equal,'batch_rows':1,'capture':False,'device':'cpu','grouped':True,'model_training':False}for name,grad,equal in (
    ('CPU_NO_GRAD_EQUAL_TF',False,True),('CPU_NO_GRAD_UNMERGED',False,False),('CPU_GRAD_ENABLED_UNMERGED',True,False)))
PARITY_PAIRS=(('CPU_NO_GRAD_EQUAL_TF','CPU_NO_GRAD_UNMERGED'),('CPU_NO_GRAD_UNMERGED','CPU_GRAD_ENABLED_UNMERGED'),('CPU_NO_GRAD_EQUAL_TF','CPU_GRAD_ENABLED_UNMERGED'))


def first_parity_source_bridge():
    import hashlib
    before=subprocess.check_output(['git','show',PARITY_PARENT+':'+SELF],cwd=REPO); after=(REPO/SELF).read_bytes()
    begin=b'# First-row parity measurement; the consumed gradient route stays unchanged.\n'; end=b'# End first-row parity measurement.\n\n'
    dispatch=b"    if sys.argv[1:2]==['--first-parity-plan-sha256']:\n        return main_first_parity()\n\n"
    need(after.count(begin)==after.count(end)==1,'FIRST_PARITY_SINGLE_ADDITIVE_BLOCK')
    first,tail=after.split(begin);_,last=tail.split(end)
    need(last.count(dispatch)==1 and before==first+last.replace(dispatch,b''),'FIRST_PARITY_OLD_FUNCTIONS_CONSTANTS_MAIN_TAIL_EXACT')
    old_test=subprocess.check_output(['git','show',PARITY_PARENT+':'+TEST],cwd=REPO)
    need((REPO/TEST).read_bytes().startswith(old_test),'FIRST_PARITY_OLD_TEST_BYTES_EXACT')
    need(not subprocess.check_output(['git','diff',PARITY_PARENT,'--','gx1','scripts','tests',':(exclude)'+SELF,':(exclude)'+TEST],cwd=REPO),'FIRST_PARITY_OTHER_TRACKED_SOURCES_EXACT')
    return {'anchor':PARITY_PARENT,'old_owner_sha256':hashlib.sha256(before).hexdigest(),'current_owner_sha256':saved.native.sha(REPO/SELF),
            'old_test_sha256':hashlib.sha256(old_test).hexdigest(),'old_functions_constants_and_main_tail_preserved':True,'old_test_prefix_preserved':True}


def build_first_parity_plan(*,source_base_commit=None):
    d=read(PARITY_DESIGN)
    need(d['schema_version']=='gx1_provider_hlc_first_parity_design_v1' and d['basename']==PARITY_NAME and d['stage']==PARITY_STAGE
         and d['module']==MODULE and d['plan_path']==str(PARITY_PLAN) and d['output_root']==str(PARITY_ROOT) and d['source_parent_commit']==PARITY_PARENT
         and d['CLI_flag']=='--first-parity-plan-sha256' and d['execution']==PARITY_EXECUTION and d['execution_authorized']is False
         and d['training_authorized']is False and d['test_authorized']is False and d['old_gradient_plan_remains_consumed']is True
         and d['original_tolerance_changed']is False and d['fixed_original_tolerance']==TOLERANCES['saved_FINAL_logits']
         and d['routes']==list(PARITY_ROUTES) and d['method']['pairwise_comparisons']==[list(pair)for pair in PARITY_PAIRS]
         and d['method']['finite_tolerance_mismatch_is_measurement_not_technical_failure']is True,'FIRST_PARITY_LOCKED_DESIGN')
    for binding in d['inputs'].values():reference(binding)
    meta={key:read(binding)for key,binding in d['inputs'].items()if key!='torch_transformer_source'}
    base=build_plan(source_base_commit=source_base_commit); old=meta['failed_gradient_plan']; claim=meta['failed_gradient_claim']; terminal=meta['failed_gradient_terminal']; review=meta['failed_gradient_review']
    need(old['schema_version']==base['schema_version'] and old['stage']==STAGE and old['plan_path']==str(PLAN) and old['output_root']==str(ROOT)
         and old['inputs']==base['inputs'] and old['budget']==base['budget'] and old['panel']==base['panel']
         and all(old[key]==base[key]for key in ('model_contract','target_contract','truth_sha256','TRAIN_issue_ns_sha256','metadata','feature_array_headers','capture_array_headers','target_array_headers','target_archive_inventory','cache_scope')),'FIRST_PARITY_SAME_FAILED_FUNCTION_INPUTS')
    need(claim['plan']==terminal['plan']==review['plan']==d['inputs']['failed_gradient_plan'] and terminal['claim']==review['claim']==d['inputs']['failed_gradient_claim']
         and review['terminal']==d['inputs']['failed_gradient_terminal'] and terminal['status']=='FAILED_CONSUMED' and terminal['exit_code']==1 and terminal['result']is None
         and terminal['remaining_invocations']==claim['remaining_invocations']==0 and terminal['relaunch_allowed']is False and claim['relaunch_allowed']is False
         and terminal['error']=='ValueError:PROVIDER_HLC_FROZEN_GRADIENT_SAVED_FINAL_LOGIT_PARITY:activity_logits'
         and terminal['progress']['phase']=='FULL96_AUTOGRAD_ENABLED_EVAL_PARITY_BEFORE_DERIVATIVES' and terminal['progress']['counts']['NN_forwards']==1
         and terminal['progress']['counts']['autograd_calls']==terminal['progress']['counts']['optimizer_updates']==0
         and review['actual_counts']==terminal['progress']['counts'] and review['source_and_all_original_bound_input_bytes_unchanged']is True
         and review['actual_training_cause']==review['selected_causal_fix']=='UNKNOWN' and review['all20_approved']is False,'FIRST_PARITY_OLD_ATTEMPT_CONSUMED_NO_VERDICT')
    backup=meta['failed_gradient_result_backup']; source_backup=meta['failed_gradient_source_plan_backup']; fix=meta['failure_logging_fix']; bridge=first_parity_source_bridge()
    need(backup['status']=='ALL22_ORIGINAL_FAILURE_FILES_SEPARATE_MAC_DISK_SHA_FULLFSYNC_READBACK_VERIFIED' and backup['models_deserialized']==0
         and source_backup['status']=='ALL14_FILES_SEPARATE_MAC_DISK_SHA_FULLFSYNC_READBACK_VERIFIED' and source_backup['models_deserialized']==0
         and fix['after_owner']=={'path':str(REPO/SELF),'sha256':bridge['old_owner_sha256']} and fix['after_test']=={'path':str(REPO/TEST),'sha256':bridge['old_test_sha256']}
         and fix['affected_cases_executed_and_passed']==2 and fix['new_native_model_forwards']==fix['new_real_data_or_checkpoint_decodes']==0
         and fix['concrete_failure_review']==d['inputs']['failed_gradient_review'] and fix['new_plan_required']is True and fix['relaunch_allowed']is False,'FIRST_PARITY_BACKUP_LOGGING_FIX_PROVENANCE')
    expected_budget={**base['budget'],'NN_forwards_max':3,'autograd_calls_max':0,'autograd_terms':[],'gradient_forwards':0,'parity_forwards':3,
        'canonical_loss_calls':0,'forward_rows_per_call':1,'hidden_capture_calls':0,'original_packet_rows':96,'selected_model_row_position':0,'selected_model_unique_rows':1}
    expected_panel={**base['panel'],'full_original_B96_packet_retained_and_authenticated':True,'measurement':'ROW_ZERO_ONLY_FOR_FIRST_FAILED_PARITY_NOT_GRADIENT_QUALIFICATION','selected_model_row_position':0,'selected_model_unique_rows':1}
    need(d['budget']==expected_budget and d['panel']==expected_panel,'FIRST_PARITY_EXACT_BUDGET_PANEL')
    for name,digest in old['source_owners'].items():
        if name not in (SELF,TEST):need(saved.native.sha(REPO/name)==digest,'FIRST_PARITY_PREVIOUS_OWNER_UNCHANGED')
    inputs={**base['inputs'],'gradient_design':base['inputs']['design'],**d['inputs'],'design':PARITY_DESIGN}
    plan={**base,'schema_version':'gx1_provider_hlc_first_parity_plan_v1','stage':PARITY_STAGE,'plan_path':str(PARITY_PLAN),'output_root':str(PARITY_ROOT),
          'inputs':inputs,'input_roles_not_JSON':sorted(NOT_JSON|{'torch_transformer_source'}),'execution':PARITY_EXECUTION,'budget':d['budget'],'panel':d['panel'],
          'source_proof':bridge,'routes':d['routes'],'fixed_original_tolerance':d['fixed_original_tolerance'],'method':d['method'],'limits':d['limits'],
          'question':meta['first_parity_scope']['question'],'failed_attempt_provenance':{'plan':d['inputs']['failed_gradient_plan'],'claim':d['inputs']['failed_gradient_claim'],'terminal':d['inputs']['failed_gradient_terminal']}}
    checked_inputs(plan); return plan


def gate_first_parity_expected(binding,admitted):
    return {'status':'ADMITTED_EXACT_CONDITIONAL_PROVIDER_HLC_FIRST_PARITY'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}


def inspect_first_parity(sha,*,admitted):
    need(Path.cwd().resolve()==REPO and type(sha)is str and len(sha)==64 and all(c in '0123456789abcdef'for c in sha),'FIRST_PARITY_CWD_SHA')
    binding={'path':str(PARITY_PLAN),'sha256':sha}; plan=read(binding)
    need(plan==build_first_parity_plan(source_base_commit=plan['source_base_commit']) and source_clean(plan),'FIRST_PARITY_EXACT_PLAN_SOURCE')
    pref=ref(POLICY); policy=read(pref); old=policy[STAGE]
    need(all(policy.get(key)is False for key in saved.FLAGS) and policy[PARITY_STAGE]==gate_first_parity_expected(binding,admitted)
         and old['plan']==plan['inputs']['failed_gradient_plan'] and old['execution_authorized']is False and old['remaining_invocations']==0
         and old['relaunch_allowed']is False and old['training_authorized']is False and old['test_authorized']is False,'FIRST_PARITY_ONE_USE_OLD_GRADIENT_CLOSED')
    need(not PARITY_ROOT.exists() and not PARITY_ROOT.is_symlink(),'FIRST_PARITY_FRESH_ROOT'); return plan,binding,pref


def first_parity_compare(reference_value,candidate,*,tolerance):
    need(tolerance==TOLERANCES['saved_FINAL_logits'] and all(isinstance(value,np.ndarray)and value.dtype==np.float32 and value.shape==(1,5)and np.isfinite(value).all()for value in (reference_value,candidate)),'FIRST_PARITY_COMPARISON_DOMAIN')
    left=reference_value[0].astype(np.float64); right=candidate[0].astype(np.float64); delta=np.abs(right-left); zero=left==0
    relative=[None if is_zero else float(error/abs(value))for value,error,is_zero in zip(left,delta,zero)]
    passed=np.isclose(candidate,reference_value,atol=tolerance['atol'],rtol=tolerance['rtol'])[0]
    return {'reference_values':left.tolist(),'candidate_values':right.tolist(),'absolute_delta':delta.tolist(),'max_absolute_delta':float(delta.max()),
            'relative_error':relative,'reference_zero_mask':zero.tolist(),'zero_reference_relative_status':[('ZERO_OVER_ZERO'if error==0 else'INFINITE')if is_zero else None for error,is_zero in zip(delta,zero)],
            'passed_mask':passed.tolist(),'all_within_original_tolerance':bool(passed.all()),'fixed_tolerance':dict(tolerance)}


def first_parity_validate_counts(counts,budget):
    expected={'full_feature_archive_loads':1,'feature_arrays_decoded':27,'target_archive_loads':1,'target_arrays_decoded':14,'capture_archive_loads':1,'capture_arrays_decoded':3,
              'checkpoint_deserializations':1,'model_constructions':1,'model_state_install_calls':1,'parity_forwards':3,'gradient_forwards':0,'NN_forwards':3,'autograd_calls':0,
              'backward_calls':0,'optimizer_constructions':0,'optimizer_restores':0,'optimizer_updates':0,'checkpoint_RNG_restore_calls':0,'fits':0,'prior_refits':0,'new_source_HTTP_calls':0,
              'canonical_loss_calls':0,'hidden_capture_calls':0,'grad_enabled_forward_calls':1,'no_grad_forward_calls':2}
    need(counts==expected and all(type(value)is int for value in counts.values()) and counts['NN_forwards']==budget['NN_forwards_max'] and budget['autograd_calls_max']==0,'FIRST_PARITY_EXACT_COUNTS'); return True


def calculate_first_parity(plan,*,progress=None):
    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as body
    torch.set_num_threads(1); need(torch.get_num_threads()==1 and not torch.cuda.is_initialized(),'FIRST_PARITY_CPU_THREADS_CUDA')
    progress={}if progress is None else progress; budget=plan['budget']; counts=dict.fromkeys(('full_feature_archive_loads','feature_arrays_decoded','target_archive_loads','target_arrays_decoded','capture_archive_loads','capture_arrays_decoded','checkpoint_deserializations','model_constructions','model_state_install_calls','parity_forwards','gradient_forwards','NN_forwards','autograd_calls','backward_calls','optimizer_constructions','optimizer_restores','optimizer_updates','checkpoint_RNG_restore_calls','fits','prior_refits','new_source_HTTP_calls','canonical_loss_calls','hidden_capture_calls','grad_enabled_forward_calls','no_grad_forward_calls'),0)
    progress.update(phase='FIRST_PARITY_CANONICAL_CONTEXT_AND_B96',counts=counts,routes=[])
    context=adapter.load_context(plan['metadata']['adapter_metadata']); counts.update(full_feature_archive_loads=1,feature_arrays_decoded=27,target_archive_loads=1,target_arrays_decoded=14)
    need(bool(np.all(context['provider'].bar_start_ns<1782864000000000000)),'FIRST_PARITY_PRETEST_SOURCE_CUTOFF')
    take=np.arange(96,dtype=np.int64); packet=adapter.packet(context,take); selected_source_row=int(context['source_rows'][0])
    need(core.digest(packet['truth']['issue_ns'])==plan['panel']['issue_ns_sha256'] and core.digest(context['source_rows'][take])==plan['panel']['source_rows_sha256']
         and bool(np.all((packet['truth']['issue_ns']>=train.START)&(packet['truth']['issue_ns']<train.UPPER))) and len(packet['truth']['issue_ns'])==96 and len(packet['features'])==9
         and sum(v.numel()*v.element_size()for v in packet['features'].values())==budget['packet_feature_bytes'],'FIRST_PARITY_EXACT_B96_PACKET')
    normalization=context['normalization']; prior=context['prior']; binding=context['input_binding']; target=context['target_contract']; del context; gc.collect()
    progress['phase']='FIRST_PARITY_CONTEXT_RELEASED_CAPTURE3'; capture=saved.load_fields(plan['inputs']['capture'],CAPTURE,plan['capture_array_headers']); counts.update(capture_archive_loads=1,capture_arrays_decoded=3)
    need(sum(v.nbytes for v in capture.values())==budget['capture_ndarray_bytes_decoded'] and core.digest(capture['issue_ns'])==plan['TRAIN_issue_ns_sha256'] and np.array_equal(capture['issue_ns'][:96],packet['truth']['issue_ns']),'FIRST_PARITY_SAVED_CLOCK_BYTES')
    expected={key:np.ascontiguousarray(capture[key][:1]).copy()for key in CAPTURE[1:]}; del capture; gc.collect()
    progress['phase']='FIRST_PARITY_CPU_FACTORY_STRICT_FINAL'
    model=model_owner.build_native_future_state_model(input_normalization=normalization,input_binding=binding,architecture=plan['metadata']['original_epoch_plan']['metadata']['architecture'],device=torch.device('cpu'),train_prior=prior,target_contract=target,backbone_gradient_policy='M5_primary'); counts['model_constructions']+=1
    cp=safe_load_final(plan); counts['checkpoint_deserializations']+=1; model.load_state_dict(cp['model_state'],strict=True); counts['model_state_install_calls']+=1
    need(core.state_sha(model.state_dict())==plan['metadata']['final_receipt']['proof']['model_state'],'FIRST_PARITY_INSTALLED_FINAL_MODEL_PROOF'); del cp; gc.collect(); model.eval()
    state_before=core.state_sha(model.state_dict()); rng_before=core.shared_gradient_rng(); packet_before=core.state_sha(packet); ambient_grad=torch.is_grad_enabled()
    need(all(p.grad is None and p.device.type=='cpu'and p.dtype==torch.float32 for p in model.parameters()),'FIRST_PARITY_CPU_PARAMETER_GRAD_NONE')
    features={key:value[:1]for key,value in packet['features'].items()}
    def record_array(value):return {'values':value[0].astype(np.float64).tolist(),'shape':list(value.shape),'dtype':value.dtype.str,'sha256':core.digest(value)}
    input_proof={key:{'shape':list(value.shape),'dtype':str(value.dtype),'sha256':core.digest(value.detach().cpu().numpy())}for key,value in features.items()}
    def context_bookend():
        need(body._BATCH_EQUAL_LENGTH_MTF_EVAL.get()is False and body._MODEL_FINITE_CHECK_SCOPE.get()is None and torch.is_grad_enabled()==ambient_grad,'FIRST_PARITY_CONTEXT_RESTORE')
    context_bookend()
    progress.update(selected_issue_ns=int(packet['truth']['issue_ns'][0]),selected_source_row=selected_source_row,first_row_feature_proof=input_proof,saved_FINAL={key:record_array(value)for key,value in expected.items()})
    def count_forward(*_):
        counts['NN_forwards']+=1; counts['parity_forwards']+=1; counts['grad_enabled_forward_calls'if torch.is_grad_enabled()else'no_grad_forward_calls']+=1
        need(counts['NN_forwards']<=budget['NN_forwards_max'],'FIRST_PARITY_FORWARD_BUDGET')
    hook=model.register_forward_pre_hook(count_forward); arrays={}; records=progress['routes']
    try:
        for route in plan['routes']:
            context_bookend(); progress.update(phase='FIRST_PARITY_ROUTE',active_route=route['name'])
            try:
                with torch.enable_grad()if route['grad_enabled']else torch.no_grad():
                    output,hidden,activation=core.future_cuda_forward(model,features,grouped=True,equal_tf=route['equal_tf'],capture=False)
                    need(hidden is None and set(output)==set(CAPTURE[1:]),'FIRST_PARITY_OUTPUT_HEADS_NO_HIDDEN_CAPTURE')
                    values={key:value.detach().cpu().numpy().copy()for key,value in output.items()}
                    need(all(value.dtype==np.float32 and value.shape==(1,5)and np.isfinite(value).all()for value in values.values()),'FIRST_PARITY_FINITE_FLOAT32_HEADS')
                    groups={}
                    for tf in train.TF:groups.setdefault(str(features['seq_'+tf.lower()].shape[1]),[]).append(tf)
                    need(activation=={'batch_eval_active':route['equal_tf']and not route['grad_enabled'],'model_training':False,'grad_enabled':route['grad_enabled'],'scope_enabled':route['equal_tf'],'eligible_equal_length_TF_groups':groups},'FIRST_PARITY_ACTUAL_ROUTE_FLAGS')
                    arrays[route['name']]=values; record={'name':route['name'],'requested':dict(route),'activation':activation,'logits':{key:record_array(value)for key,value in values.items()},
                        'vs_saved_FINAL':{key:first_parity_compare(expected[key],value,tolerance=plan['fixed_original_tolerance'])for key,value in values.items()}}
                    records.append(record); del output,hidden,values
            finally:context_bookend()
            measurement_bookend(model,state_before,rng_before)
        pairwise=[{'reference_route':left,'candidate_route':right,'per_head':{key:first_parity_compare(arrays[left][key],arrays[right][key],tolerance=plan['fixed_original_tolerance'])for key in CAPTURE[1:]}}for left,right in PARITY_PAIRS]
        bookend=measurement_bookend(model,state_before,rng_before); context_bookend(); need(core.state_sha(packet)==packet_before,'FIRST_PARITY_INPUT_PACKET_MUTATION'); first_parity_validate_counts(counts,budget)
    finally:hook.remove()
    progress.update(phase='COMPLETE_FIRST_PARITY_MEASUREMENT',pairwise=pairwise)
    return {'payload_budget':budget,'counts':counts,'routes':records,'pairwise':pairwise,'saved_FINAL':{key:record_array(value)for key,value in expected.items()},
            'fixed_original_tolerance':plan['fixed_original_tolerance'],'all_routes_within_saved_original_tolerance':all(comparison['all_within_original_tolerance']for record in records for comparison in record['vs_saved_FINAL'].values()),
            'panel':plan['panel'],'selected_issue_ns':int(packet['truth']['issue_ns'][0]),'selected_source_row':selected_source_row,'first_row_feature_proof':input_proof,'retained_B96_packet_unmutated':True,
            'original_B96_known_activity_denominator_identity_only':int(packet['truth']['activity_known'].sum()),'cache_scope':plan['cache_scope'],
            'old_context_released_before_model_construction':True,'final_all_ten_checkpoint_proofs_verified':True,'strict_FINAL_model_state_installed':True,'model_state_and_local_RNG_bookend_passed':bookend,
            'context_bookends_passed':True,'ambient_grad_mode_restored':True,'all_parameter_grad_fields_None':True,'local_RNG_bookend_scope':['python','numpy','torch_cpu'],
            'count_semantics':'Three parity forwards include two no_grad and one grad-enabled forward. gradient_forwards=0 means zero derivative-phase forwards; autograd/backward calls and parameter updates are zero.',
            'measurement_completed':True,'full96_gradient_qualification':False,'actual_training_cause':'UNKNOWN','selected_causal_fix':'UNKNOWN','later_generalization':'UNKNOWN','economic_edge':'UNKNOWN','all20_approved':False,'limits':plan['limits']}


def parity_publish(name,value):
    path=PARITY_ROOT/name; saved._publish_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()); return ref(path)


def run_first_parity(sha):
    began=time.monotonic(); cap=require_capped_cpu_audit_execution(); lock=require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==4*1024**3 and cap['swap']==512*1024**2 and cap['numerical_threads']==1 and cap['pids']==64
         and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):],'FIRST_PARITY_CPU_CAPS')
    plan,binding,pref=inspect_first_parity(sha,admitted=True); checked_inputs(plan); boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(); PARITY_ROOT.mkdir(parents=True,exist_ok=False)
    handlers={sig:signal.getsignal(sig)for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}; claim=result=None; progress={'phase':'CLAIM'}
    try:
        def expired(*_):raise TimeoutError('PROVIDER_HLC_FIRST_PARITY_WALL')
        def stopped(sig,*_):raise InterruptedError('PROVIDER_HLC_FIRST_PARITY_SIGNAL_'+str(sig))
        signal.signal(signal.SIGALRM,expired); signal.signal(signal.SIGTERM,stopped); signal.signal(signal.SIGINT,stopped)
        claim=parity_publish('CLAIM.json',{'stage':PARITY_STAGE,'plan':binding,'policy_at_claim':pref,'boot_id':boot,'remaining_invocations':0,'relaunch_allowed':False,'training_authorized':False,'test_data_used':False})
        remaining=PARITY_EXECUTION['internal_seconds']-(time.monotonic()-began); need(remaining>0,'FIRST_PARITY_TIME_BEFORE_DECODE'); signal.setitimer(signal.ITIMER_REAL,remaining)
        allowed={plan['inputs'][role]['path']for role in ('feature_archive','capture','target_arrays','final_checkpoint')}
        def audit(event,args):
            if event in ('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('PROVIDER_HLC_FIRST_PARITY_NETWORK_FORBIDDEN')
            if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
                name=os.fsdecode(args[0])
                if name.endswith(('.npz','.pt','.pth','.parquet','.safetensors','.pkl'))and str(Path(name).resolve())not in allowed:raise RuntimeError('PROVIDER_HLC_FIRST_PARITY_PAYLOAD_FORBIDDEN')
        sys.addaudithook(audit); report=calculate_first_parity(plan,progress=progress); checked_inputs(plan)
        need(source_clean(plan)and ref(POLICY)==pref and ref(PARITY_PLAN)==binding and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,'FIRST_PARITY_SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND')
        result=parity_publish('RESULT.json',{'status':'COMPLETE_CONDITIONAL_PROVIDER_HLC_FIRST_PARITY_MEASUREMENT','stage':PARITY_STAGE,'plan':binding,'claim':claim,'policy_at_claim':pref,'boot_id':boot,'report':report,'source_unchanged':True,'test_data_used':False,'economic_edge_proven':False,'elapsed_seconds':time.monotonic()-began})
        terminal=parity_publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'stage':PARITY_STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False}); return {'result':result,'terminal':terminal}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0); parity_publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'stage':PARITY_STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'progress':progress,'error':type(error).__name__+':'+str(error)[:2000],'test_data_used':False}); raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in handlers.items():signal.signal(sig,handler)


# Single B256 geometry measurement; completed parity routes stay unchanged.
BATCH_NAME='NATIVE_PROVIDER_HLC_BATCH_GEOMETRY_20261124_001'; BATCH_STAGE=BATCH_NAME.lower()
BATCH_PLAN=REPO/'configs/research'/(BATCH_NAME+'.json'); BATCH_ROOT=Path('/home/andre2/GX1_RUNS')/BATCH_NAME
BATCH_DESIGN={'path':'/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/PROVIDER_HLC_BATCH_GEOMETRY_DESIGN001.json','sha256':'ee827a739c59f60ab95f04f53d0489ea5ea7343b267e09690cebef6ed139088b'}
BATCH_PARENT='1eae0a9710da279a9a8b81a7cc1147ec86126083'; BATCH_EXECUTION=dict(EXECUTION)
BATCH_ROUTE={'name':'CPU_B256_NO_GRAD_EQUAL_TF','batch_rows':256,'capture':False,'device':'cpu','equal_tf':True,'grad_enabled':False,'grouped':True,'model_training':False}


def build_batch_geometry_plan(*,source_base_commit=None):
    d=read(BATCH_DESIGN)
    need(d['schema_version']=='gx1_provider_hlc_batch_geometry_design_v1' and d['basename']==BATCH_NAME and d['stage']==BATCH_STAGE
         and d['module']==MODULE and d['plan_path']==str(BATCH_PLAN) and d['output_root']==str(BATCH_ROOT) and d['source_parent_commit']==BATCH_PARENT
         and d['CLI']==['--first-parity-plan-sha256','<SHA256>','--batch-geometry'] and d['execution']==BATCH_EXECUTION
         and d['execution_authorized']is False and d['training_authorized']is False and d['test_authorized']is False
         and d['original_tolerance_changed']is False and d['stop_parity_series_after_this_comparison']is True
         and d['fixed_original_tolerance']==TOLERANCES['saved_FINAL_logits'] and d['route']==BATCH_ROUTE,'BATCH_GEOMETRY_LOCKED_DESIGN')
    for binding in d['inputs'].values():reference(binding)
    meta={key:read(binding)for key,binding in d['inputs'].items()}; base=build_first_parity_plan(source_base_commit=source_base_commit)
    old=meta['first_parity_plan']; result=meta['first_parity_result']; terminal=meta['first_parity_terminal']; accepted=meta['first_parity_acceptance']; report=result['report']
    need(old['schema_version']==base['schema_version'] and old['stage']==PARITY_STAGE and old['plan_path']==str(PARITY_PLAN) and old['output_root']==str(PARITY_ROOT)
         and old['inputs']==base['inputs'] and all(old[key]==base[key]for key in ('metadata','model_contract','target_contract','truth_sha256','TRAIN_issue_ns_sha256','budget','panel','routes','fixed_original_tolerance','feature_array_headers','capture_array_headers','target_array_headers','target_archive_inventory','cache_scope')),'BATCH_GEOMETRY_SAME_PRIOR_FUNCTION_INPUTS')
    need(result['status']=='COMPLETE_CONDITIONAL_PROVIDER_HLC_FIRST_PARITY_MEASUREMENT' and result['plan']==terminal['plan']==accepted['plan']==d['inputs']['first_parity_plan']
         and terminal['status']=='COMPLETE' and terminal['exit_code']==0 and terminal['result']==accepted['result']==d['inputs']['first_parity_result']
         and accepted['terminal']==d['inputs']['first_parity_terminal'] and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False
         and result['source_unchanged']is True and result['test_data_used']is False and accepted['actual_counts']==report['counts']
         and accepted['model_input_packet_local_RNG_context_bookends_passed']is True and accepted['source_inputs_plan_policy_WSL_boot_bookends']is True
         and report['measurement_completed']is True and report['full96_gradient_qualification']is False and report['actual_training_cause']==report['selected_causal_fix']=='UNKNOWN'
         and report['all20_approved']is False,'BATCH_GEOMETRY_PRIOR_MEASUREMENT_CONSUMED')
    first_parity_validate_counts(report['counts'],old['budget'])
    for name,digest in old['source_owners'].items():
        if name not in (SELF,TEST):need(saved.native.sha(REPO/name)==digest,'BATCH_GEOMETRY_PREVIOUS_OWNER_UNCHANGED')
    backup=meta['first_parity_result_backup']; source_backup=meta['first_parity_source_plan_backup']
    need(backup['status']=='ALL24_ORIGINAL_FILES_SEPARATE_MAC_DISK_SHA_FULLFSYNC_READBACK_VERIFIED' and backup['models_deserialized']==0
         and source_backup['status']=='ALL14_FILES_SEPARATE_MAC_DISK_SHA_FULLFSYNC_READBACK_VERIFIED' and source_backup['models_deserialized']==0,'BATCH_GEOMETRY_BACKUPS')
    original=meta['original_saved_model_result']; score=original['report']['whole_FINAL_score']; first=score['batches'][0]
    need(original['plan']==base['inputs']['completed_model_plan'] and original['report']['final_logits']==base['inputs']['capture']
         and original['report']['final_checkpoint_receipt']==base['inputs']['final_receipt'] and score['batch_rows']==256
         and score['state_sha256_before_and_after']==base['metadata']['final_receipt']['proof']['model_state']
         and d['original_score_batch']['JSON_pointer']=='/report/whole_FINAL_score/batches/0'
         and {key:first[key]for key in d['original_score_batch']if key!='JSON_pointer'}=={key:value for key,value in d['original_score_batch'].items()if key!='JSON_pointer'} and first['rows']==256 and first['rows_done']==256
         and all(score['output_sha256'][key]==base['capture_array_headers'][key]['sha256']for key in CAPTURE[1:]),'BATCH_GEOMETRY_ORIGINAL_SCORE_BATCH')
    expected_budget={**base['budget'],'NN_forwards_max':1,'forward_rows_per_call':256,'microbatch_rows':256,'original_packet_rows':256,'packet_feature_bytes':169086976,
        'panel_rows':256,'parity_forwards':1,'row_for_prior_CPU_B1_comparison':0,'rows_compared_against_saved_GPU':256,'selected_model_unique_rows':256}
    del expected_budget['selected_model_row_position']
    need(d['budget']==expected_budget and d['panel']=={'TRAIN_position_start':0,'TRAIN_position_stop_exclusive':256,'drop_or_replace_rows':False,
         'new_first256_hashes':'derive and persist only after canonical decode; wholeTRAIN axes+immutable sourcecache+fixedslice bind selection',
         'original_first96_issue_ns_sha256':base['panel']['issue_ns_sha256'],'original_first96_source_rows_sha256':base['panel']['source_rows_sha256'],
         'prior_CPU_B1_route':'CPU_NO_GRAD_EQUAL_TF','prior_CPU_B1_row_position':0,'rows':256,'selected_from_result_quality':False,'selection':'ORIGINAL_FIRST_FULL_FINAL_SCORE_BATCH_BEFORE_NEW_MEASUREMENT'},'BATCH_GEOMETRY_EXACT_BUDGET_PANEL')
    prior=[record for record in report['routes']if record['name']==d['panel']['prior_CPU_B1_route']]
    need(len(prior)==1 and prior[0]['requested']==PARITY_ROUTES[0] and prior[0]['activation']==first['activation']
         and set(report['first_row_feature_proof'])==set(model_owner._native_bar_feature_keys(base['metadata']['adapter_metadata']['input_binding'])),'BATCH_GEOMETRY_PRIOR_CPU_B1')
    inputs={**base['inputs'],'first_parity_design':base['inputs']['design'],**d['inputs'],'design':BATCH_DESIGN}
    plan={**base,'schema_version':'gx1_provider_hlc_batch_geometry_plan_v1','stage':BATCH_STAGE,'plan_path':str(BATCH_PLAN),'output_root':str(BATCH_ROOT),
          'inputs':inputs,'execution':BATCH_EXECUTION,'budget':d['budget'],'panel':d['panel'],'route':d['route'],'method':d['method'],'limits':d['limits'],
          'original_score_batch':d['original_score_batch'],'prior_CPU_B1':{'route':prior[0],'selected_issue_ns':report['selected_issue_ns'],'selected_source_row':report['selected_source_row'],'feature_proof':report['first_row_feature_proof']},
          'source_parent_commit':BATCH_PARENT,'static_first_parity_preservation_required':True,'question':meta['batch_geometry_scope']['question'],'stop_parity_series_after_this_comparison':True}
    checked_inputs(plan); return plan


def gate_batch_geometry_expected(binding,admitted):
    return {'status':'ADMITTED_EXACT_CONDITIONAL_PROVIDER_HLC_BATCH_GEOMETRY'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}


def inspect_batch_geometry(sha,*,admitted):
    need(Path.cwd().resolve()==REPO and type(sha)is str and len(sha)==64 and all(c in '0123456789abcdef'for c in sha),'BATCH_GEOMETRY_CWD_SHA')
    binding={'path':str(BATCH_PLAN),'sha256':sha}; plan=read(binding)
    need(plan==build_batch_geometry_plan(source_base_commit=plan['source_base_commit']) and source_clean(plan),'BATCH_GEOMETRY_EXACT_PLAN_SOURCE')
    pref=ref(POLICY); policy=read(pref)
    need(all(policy.get(key)is False for key in saved.FLAGS) and policy[BATCH_STAGE]==gate_batch_geometry_expected(binding,admitted),'BATCH_GEOMETRY_ONE_USE_FLAGS')
    for stage,role in ((STAGE,'failed_gradient_plan'),(PARITY_STAGE,'first_parity_plan')):
        old=policy[stage]
        need(old['plan']==plan['inputs'][role] and old['execution_authorized']is False and old['remaining_invocations']==0 and old['relaunch_allowed']is False
             and old['training_authorized']is False and old['test_authorized']is False,'BATCH_GEOMETRY_OLD_PLANS_CLOSED')
    need(not BATCH_ROOT.exists() and not BATCH_ROOT.is_symlink(),'BATCH_GEOMETRY_FRESH_ROOT'); return plan,binding,pref


def batch_geometry_compare(reference_value,candidate,*,tolerance):
    need(all(isinstance(value,np.ndarray)and value.dtype==np.float32 and value.shape==(256,5)and np.isfinite(value).all()for value in (reference_value,candidate)),'BATCH_GEOMETRY_COMPARISON_DOMAIN')
    rows=[first_parity_compare(reference_value[index:index+1],candidate[index:index+1],tolerance=tolerance)for index in range(256)]
    passed=sum(sum(row['passed_mask'])for row in rows)
    return {'rows':256,'per_row':rows,'max_absolute_delta':max(row['max_absolute_delta']for row in rows),
            'per_TF_max_absolute_delta':[max(row['absolute_delta'][j]for row in rows)for j in range(5)],'passed_cells':passed,'failed_cells':1280-passed,'all_within_original_tolerance':passed==1280}


def batch_geometry_validate_counts(counts,budget):
    expected={'full_feature_archive_loads':1,'feature_arrays_decoded':27,'target_archive_loads':1,'target_arrays_decoded':14,'capture_archive_loads':1,'capture_arrays_decoded':3,
              'checkpoint_deserializations':1,'model_constructions':1,'model_state_install_calls':1,'parity_forwards':1,'gradient_forwards':0,'NN_forwards':1,'autograd_calls':0,
              'backward_calls':0,'optimizer_constructions':0,'optimizer_restores':0,'optimizer_updates':0,'checkpoint_RNG_restore_calls':0,'fits':0,'prior_refits':0,'new_source_HTTP_calls':0,
              'canonical_loss_calls':0,'hidden_capture_calls':0,'grad_enabled_forward_calls':0,'no_grad_forward_calls':1}
    need(counts==expected and all(type(value)is int for value in counts.values()) and budget['NN_forwards_max']==1 and budget['autograd_calls_max']==0,'BATCH_GEOMETRY_EXACT_COUNTS'); return True


def calculate_batch_geometry(plan,*,progress=None):
    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as body
    torch.set_num_threads(1); need(torch.get_num_threads()==1 and not torch.cuda.is_initialized(),'BATCH_GEOMETRY_CPU_THREADS_CUDA')
    progress={}if progress is None else progress; budget=plan['budget']; counts=dict.fromkeys(('full_feature_archive_loads','feature_arrays_decoded','target_archive_loads','target_arrays_decoded','capture_archive_loads','capture_arrays_decoded','checkpoint_deserializations','model_constructions','model_state_install_calls','parity_forwards','gradient_forwards','NN_forwards','autograd_calls','backward_calls','optimizer_constructions','optimizer_restores','optimizer_updates','checkpoint_RNG_restore_calls','fits','prior_refits','new_source_HTTP_calls','canonical_loss_calls','hidden_capture_calls','grad_enabled_forward_calls','no_grad_forward_calls'),0)
    progress.update(phase='BATCH_GEOMETRY_CANONICAL_CONTEXT_AND_256',counts=counts)
    context=adapter.load_context(plan['metadata']['adapter_metadata']); counts.update(full_feature_archive_loads=1,feature_arrays_decoded=27,target_archive_loads=1,target_arrays_decoded=14)
    need(bool(np.all(context['provider'].bar_start_ns<1782864000000000000)),'BATCH_GEOMETRY_PRETEST_SOURCE_CUTOFF')
    take=np.arange(256,dtype=np.int64); packet=adapter.packet(context,take); selected_source_row=int(context['source_rows'][0])
    need(len(packet['truth']['issue_ns'])==256 and core.digest(packet['truth']['issue_ns'][:96])==plan['panel']['original_first96_issue_ns_sha256']
         and core.digest(context['source_rows'][:96])==plan['panel']['original_first96_source_rows_sha256']
         and bool(np.all((packet['truth']['issue_ns']>=train.START)&(packet['truth']['issue_ns']<train.UPPER))) and len(packet['features'])==9
         and sum(v.numel()*v.element_size()for v in packet['features'].values())==budget['packet_feature_bytes'],'BATCH_GEOMETRY_EXACT_256_PACKET')
    normalization=context['normalization']; prior=context['prior']; binding=context['input_binding']; target=context['target_contract']
    input_identity={'first256_issue_ns_sha256':core.digest(packet['truth']['issue_ns']),'first256_source_rows_sha256':core.digest(context['source_rows'][take]),
                    'original_first96_issue_ns_sha256':core.digest(packet['truth']['issue_ns'][:96]),'original_first96_source_rows_sha256':core.digest(context['source_rows'][:96])}
    del context; gc.collect()
    row0_proof={key:{'shape':list(value[:1].shape),'dtype':str(value.dtype),'sha256':core.digest(value[:1].detach().cpu().numpy())}for key,value in packet['features'].items()}
    prior_b1=plan['prior_CPU_B1']
    need(row0_proof==prior_b1['feature_proof'] and int(packet['truth']['issue_ns'][0])==prior_b1['selected_issue_ns'] and selected_source_row==prior_b1['selected_source_row'],'BATCH_GEOMETRY_ROW0_PRIOR_INPUT_PROOF')
    progress.update(phase='BATCH_GEOMETRY_CONTEXT_RELEASED_CAPTURE3',input_identity=input_identity,row0_feature_proof=row0_proof)
    capture=saved.load_fields(plan['inputs']['capture'],CAPTURE,plan['capture_array_headers']); counts.update(capture_archive_loads=1,capture_arrays_decoded=3)
    need(sum(v.nbytes for v in capture.values())==budget['capture_ndarray_bytes_decoded'] and core.digest(capture['issue_ns'])==plan['TRAIN_issue_ns_sha256'] and np.array_equal(capture['issue_ns'][:256],packet['truth']['issue_ns']),'BATCH_GEOMETRY_SAVED_CLOCK_BYTES')
    expected={key:np.ascontiguousarray(capture[key][:256]).copy()for key in CAPTURE[1:]}; del capture; gc.collect()
    need(all(core.digest(value)==plan['original_score_batch']['output_sha256'][key]for key,value in expected.items()),'BATCH_GEOMETRY_ORIGINAL_CAPTURE256_SHA_BEFORE_FORWARD')
    def record_array(value):return {'values':value.astype(np.float64).tolist(),'shape':list(value.shape),'dtype':value.dtype.str,'sha256':core.digest(value)}
    progress.update(phase='BATCH_GEOMETRY_REFERENCE_AUTHENTICATED_CPU_FACTORY',saved_GPU_reference={key:record_array(value)for key,value in expected.items()},reference_capture256_SHA_authenticated=True)
    model=model_owner.build_native_future_state_model(input_normalization=normalization,input_binding=binding,architecture=plan['metadata']['original_epoch_plan']['metadata']['architecture'],device=torch.device('cpu'),train_prior=prior,target_contract=target,backbone_gradient_policy='M5_primary'); counts['model_constructions']+=1
    cp=safe_load_final(plan); counts['checkpoint_deserializations']+=1; model.load_state_dict(cp['model_state'],strict=True); counts['model_state_install_calls']+=1
    need(core.state_sha(model.state_dict())==plan['metadata']['final_receipt']['proof']['model_state'],'BATCH_GEOMETRY_INSTALLED_FINAL_MODEL_PROOF'); del cp; gc.collect(); model.eval()
    state_before=core.state_sha(model.state_dict()); rng_before=core.shared_gradient_rng(); packet_before=core.state_sha(packet); input_identity['packet_sha256']=packet_before; ambient_grad=torch.is_grad_enabled()
    need(all(p.grad is None and p.device.type=='cpu'and p.dtype==torch.float32 for p in model.parameters()),'BATCH_GEOMETRY_CPU_PARAMETER_GRAD_NONE')
    def context_bookend():need(body._BATCH_EQUAL_LENGTH_MTF_EVAL.get()is False and body._MODEL_FINITE_CHECK_SCOPE.get()is None and torch.is_grad_enabled()==ambient_grad,'BATCH_GEOMETRY_CONTEXT_RESTORE')
    context_bookend()
    def count_forward(*_):
        counts['NN_forwards']+=1; counts['parity_forwards']+=1; counts['grad_enabled_forward_calls'if torch.is_grad_enabled()else'no_grad_forward_calls']+=1
        need(counts['NN_forwards']<=budget['NN_forwards_max'],'BATCH_GEOMETRY_FORWARD_BUDGET')
    hook=model.register_forward_pre_hook(count_forward); progress['phase']='BATCH_GEOMETRY_SINGLE_FORWARD'
    try:
        try:
            with torch.no_grad():
                output,hidden,activation=core.future_cuda_forward(model,packet['features'],grouped=True,equal_tf=True,capture=False)
                need(hidden is None and set(output)==set(CAPTURE[1:]),'BATCH_GEOMETRY_OUTPUT_HEADS_NO_HIDDEN_CAPTURE')
                values={key:value.detach().cpu().numpy().copy()for key,value in output.items()}
                need(all(value.dtype==np.float32 and value.shape==(256,5)and np.isfinite(value).all()for value in values.values()) and activation==plan['original_score_batch']['activation'],'BATCH_GEOMETRY_ACTUAL_FINITE_ROUTE')
                progress.update(outputs={key:record_array(value)for key,value in values.items()},actual_activation=activation)
                del output,hidden
        finally:context_bookend()
        comparisons={key:batch_geometry_compare(expected[key],value,tolerance=plan['fixed_original_tolerance'])for key,value in values.items()}
        row0={}
        for key,value in values.items():
            record=prior_b1['route']['logits'][key]; reference_value=np.array([record['values']],dtype=np.float32)
            need(list(reference_value.shape)==record['shape']==[1,5] and reference_value.dtype.str==record['dtype'] and core.digest(reference_value)==record['sha256'],'BATCH_GEOMETRY_PRIOR_B1_LOGIT_SHA')
            row0[key]=first_parity_compare(reference_value,value[:1],tolerance=plan['fixed_original_tolerance'])
        bookend=measurement_bookend(model,state_before,rng_before); context_bookend(); need(core.state_sha(packet)==packet_before,'BATCH_GEOMETRY_INPUT_PACKET_MUTATION'); batch_geometry_validate_counts(counts,budget)
    finally:hook.remove()
    maxrss=None
    try:
        import resource
        maxrss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError,OSError,AttributeError,ValueError):pass
    progress.update(phase='COMPLETE_BATCH_GEOMETRY_MEASUREMENT',vs_saved_GPU=comparisons,row0_vs_prior_CPU_B1=row0)
    return {'payload_budget':budget,'counts':counts,'route':plan['route'],'actual_activation':activation,'outputs':progress['outputs'],'saved_GPU_reference':progress['saved_GPU_reference'],
            'vs_saved_GPU':comparisons,'row0_vs_prior_CPU_B1':row0,'all_256_within_saved_original_tolerance':all(cell['all_within_original_tolerance']for cell in comparisons.values()),
            'reference_capture256_SHA_authenticated_before_forward':True,'fixed_original_tolerance':plan['fixed_original_tolerance'],'panel':plan['panel'],'input_identity':input_identity,'row0_feature_proof':row0_proof,
            'resource_observation':{'process_maxrss_KiB':maxrss,'scope':'Whole process since startup, including context decoding; not an isolated activation peak or new resource permission.'},
            'cache_scope':plan['cache_scope'],'old_context_released_before_model_construction':True,'final_all_ten_checkpoint_proofs_verified':True,'strict_FINAL_model_state_installed':True,
            'model_state_and_local_RNG_bookend_passed':bookend,'context_bookends_passed':True,'ambient_grad_mode_restored':True,'all_parameter_grad_fields_None':True,'retained_256_packet_unmutated':True,
            'local_RNG_bookend_scope':['python','numpy','torch_cpu'],'measurement_completed':True,'full96_gradient_qualification':False,'stop_parity_series_after_this_comparison':True,
            'actual_training_cause':'UNKNOWN','selected_causal_fix':'UNKNOWN','later_generalization':'UNKNOWN','economic_edge':'UNKNOWN','all20_approved':False,'limits':plan['limits']}


def batch_geometry_publish(name,value):
    path=BATCH_ROOT/name; saved._publish_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()); return ref(path)


def run_batch_geometry(sha):
    began=time.monotonic(); cap=require_capped_cpu_audit_execution(); lock=require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==4*1024**3 and cap['swap']==512*1024**2 and cap['numerical_threads']==1 and cap['pids']==64
         and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):],'BATCH_GEOMETRY_CPU_CAPS')
    plan,binding,pref=inspect_batch_geometry(sha,admitted=True); checked_inputs(plan); boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(); BATCH_ROOT.mkdir(parents=True,exist_ok=False)
    handlers={sig:signal.getsignal(sig)for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}; claim=result=None; progress={'phase':'CLAIM'}
    try:
        def expired(*_):raise TimeoutError('PROVIDER_HLC_BATCH_GEOMETRY_WALL')
        def stopped(sig,*_):raise InterruptedError('PROVIDER_HLC_BATCH_GEOMETRY_SIGNAL_'+str(sig))
        signal.signal(signal.SIGALRM,expired); signal.signal(signal.SIGTERM,stopped); signal.signal(signal.SIGINT,stopped)
        claim=batch_geometry_publish('CLAIM.json',{'stage':BATCH_STAGE,'plan':binding,'policy_at_claim':pref,'boot_id':boot,'remaining_invocations':0,'relaunch_allowed':False,'training_authorized':False,'test_data_used':False})
        remaining=BATCH_EXECUTION['internal_seconds']-(time.monotonic()-began); need(remaining>0,'BATCH_GEOMETRY_TIME_BEFORE_DECODE'); signal.setitimer(signal.ITIMER_REAL,remaining)
        allowed={plan['inputs'][role]['path']for role in ('feature_archive','capture','target_arrays','final_checkpoint')}
        def audit(event,args):
            if event in ('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('PROVIDER_HLC_BATCH_GEOMETRY_NETWORK_FORBIDDEN')
            if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
                name=os.fsdecode(args[0])
                if name.endswith(('.npz','.pt','.pth','.parquet','.safetensors','.pkl'))and str(Path(name).resolve())not in allowed:raise RuntimeError('PROVIDER_HLC_BATCH_GEOMETRY_PAYLOAD_FORBIDDEN')
        sys.addaudithook(audit); report=calculate_batch_geometry(plan,progress=progress); checked_inputs(plan)
        need(source_clean(plan)and ref(POLICY)==pref and ref(BATCH_PLAN)==binding and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,'BATCH_GEOMETRY_SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND')
        result=batch_geometry_publish('RESULT.json',{'status':'COMPLETE_CONDITIONAL_PROVIDER_HLC_BATCH_GEOMETRY_MEASUREMENT','stage':BATCH_STAGE,'plan':binding,'claim':claim,'policy_at_claim':pref,'boot_id':boot,'report':report,'source_unchanged':True,'test_data_used':False,'economic_edge_proven':False,'elapsed_seconds':time.monotonic()-began})
        terminal=batch_geometry_publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'stage':BATCH_STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False}); return {'result':result,'terminal':terminal}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0); batch_geometry_publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'stage':BATCH_STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'progress':progress,'error':type(error).__name__+':'+str(error)[:2000],'test_data_used':False}); raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in handlers.items():signal.signal(sig,handler)


def main_batch_geometry():
    parser=argparse.ArgumentParser(); parser.add_argument('--first-parity-plan-sha256',required=True); parser.add_argument('--batch-geometry',action='store_true',required=True); parser.add_argument('--preflight',action='store_true'); args=parser.parse_args()
    need(sys.argv[1:]==['--first-parity-plan-sha256',args.first_parity_plan_sha256,'--batch-geometry']+(['--preflight']if args.preflight else[]),'BATCH_GEOMETRY_EXACT_CLI')
    if args.preflight:
        plan,binding,pref=inspect_batch_geometry(args.first_parity_plan_sha256,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_PROVIDER_HLC_BATCH_GEOMETRY_PASS','plan':binding,'policy':pref,'claim_created':False,'checkpoint_deserializations':0,'NPZ_arrays_decoded':0,'models':0,'NN_forwards':0,'autograd_calls':0,'CUDA_initialized':False}))
    else:print(json.dumps(run_batch_geometry(args.first_parity_plan_sha256),sort_keys=True))
# End single B256 geometry measurement.

# CPU-conditional B96 gradients; saved-CUDA gate remains failed.
CPU_NAME='NATIVE_PROVIDER_HLC_CPU_CONDITIONAL_GRADIENT_20261124_001'; CPU_STAGE=CPU_NAME.lower()
CPU_PLAN=REPO/'configs/research'/(CPU_NAME+'.json'); CPU_ROOT=Path('/home/andre2/GX1_RUNS')/CPU_NAME
CPU_SCOPE={'path':'/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/PROVIDER_HLC_CPU_CONDITIONAL_GRADIENT_SCOPE001.json','sha256':'a97f2676fc49d0c30c56ce3a9a0df9b966cce0c2d8c54c488154a67663c211ce'}
CPU_PARENT='f9c825571a0ba291364527f1754818fe8e0ab6a7'; CPU_EXECUTION=dict(EXECUTION)
CPU_ROUTE={'name':'CPU_GRAD_ENABLED_UNMERGED','capture':False,'device':'cpu','equal_tf':False,'grad_enabled':True,'grouped':True,'model_training':False,
           'helper':'core.future_cuda_forward under torch.enable_grad; same canonical route as the completed first-row CPU control.'}
CPU_EVIDENCE={
    'batch_geometry_acceptance':{'path':'/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/PROVIDER_HLC_BATCH_GEOMETRY_ACCEPTANCE001.json','sha256':'fb8a2ae908f88062ba7f1826b0942ae3b5c10b2e9c95a1878221e6d45f754735'},
    'batch_geometry_result_backup':{'path':'/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/PROVIDER_HLC_BATCH_GEOMETRY_BACKUP001.json','sha256':'d844a4793d25d9552742400d9865b781b7c57e3e5e5c80fc03a1933323857058'},
    'batch_geometry_source_plan_backup':{'path':'/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/PROVIDER_HLC_BATCH_GEOMETRY_SOURCE_PLAN_BACKUP001.json','sha256':'2a4964c22327d70bcc5ef595e634eb1971a8c211312b3bc9a3f4664b1509bdcd'}}


def build_cpu_conditional_plan(*,source_base_commit=None):
    scope=read(CPU_SCOPE); resources=scope['resource_budget']
    need(scope['schema_version']=='gx1_provider_hlc_CPU_conditional_gradient_scope_v1' and scope['proposed_plan_name']==CPU_NAME and scope['implementation_owner']==SELF
         and scope['execution_authorized']is False and scope['new_model_or_autograd_execution_authorized']is False and scope['training_authorized']is False and scope['test_authorized']is False
         and scope['parity_mini_series_stopped']is True and scope['all20_approved']is False and scope['route']==CPU_ROUTE
         and resources['NN_forwards_max']==192 and resources['autograd_calls_max']==384 and resources['model_constructions']==resources['checkpoint_deserializations']==1
         and resources['class']=='audit' and resources['memory']=='4G' and resources['swap']=='512M' and resources['numerical_threads']==1 and resources['pids']==64
         and resources['outer_timeout_seconds']==900 and resources['inner_timeout_seconds']==870 and resources['kill_after_seconds']==10 and resources['fallback_or_budget_extension']is False
         and all(resources[key]==0 for key in ('backward_calls','optimizer_constructions','optimizer_restores','optimizer_updates','fits','new_source_HTTP_calls','VAL_TEST_selected_rows')),'CPU_CONDITIONAL_LOCKED_SCOPE')
    base=build_batch_geometry_plan(source_base_commit=source_base_commit)
    for binding in CPU_EVIDENCE.values():reference(binding)
    accepted=read(CPU_EVIDENCE['batch_geometry_acceptance']); result=read(scope['final_batch_geometry_result']); report=result['report']; old=read(accepted['plan']); terminal=read(accepted['terminal'])
    need(accepted['schema_version']=='gx1_provider_hlc_batch_geometry_acceptance_v1' and accepted['result']==scope['final_batch_geometry_result'] and accepted['all20_approved']is False
         and result['status']=='COMPLETE_CONDITIONAL_PROVIDER_HLC_BATCH_GEOMETRY_MEASUREMENT' and result['source_unchanged']is True and result['test_data_used']is False
         and result['plan']==terminal['plan']==accepted['plan'] and terminal['status']=='COMPLETE' and terminal['exit_code']==0 and terminal['result']==accepted['result']
         and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False and report['all_256_within_saved_original_tolerance']is False
         and report['full96_gradient_qualification']is False and report['stop_parity_series_after_this_comparison']is True,'CPU_CONDITIONAL_COMPLETED_BATCH_EVIDENCE')
    batch_geometry_validate_counts(report['counts'],old['budget'])
    need(old['inputs']==base['inputs'] and all(old[key]==base[key]for key in ('metadata','model_contract','target_contract','truth_sha256','TRAIN_issue_ns_sha256','budget','panel','route','fixed_original_tolerance','cache_scope'))
         and scope['first_CPU_control_result']==base['inputs']['first_parity_result'] and scope['original_failed_gradient_plan']==base['inputs']['failed_gradient_plan']
         and scope['parent_post_readout_scope']==base['inputs']['post_readout_scope'],'CPU_CONDITIONAL_SAME_FUNCTION_AND_FAILED_GATE')
    for name,digest in old['source_owners'].items():
        if name not in (SELF,TEST):need(saved.native.sha(REPO/name)==digest,'CPU_CONDITIONAL_PREVIOUS_OWNER_UNCHANGED')
    need(read(CPU_EVIDENCE['batch_geometry_result_backup'])['status']=='ALL30_FILES_SEPARATE_MAC_DISK_SHA_FULLFSYNC_READBACK_VERIFIED'
         and read(CPU_EVIDENCE['batch_geometry_source_plan_backup'])['status']=='ALL15_FILES_SEPARATE_MAC_DISK_SHA_FULLFSYNC_READBACK_VERIFIED','CPU_CONDITIONAL_BACKUPS')
    control=read(scope['first_CPU_control_result'])['report']; anchors=[record for record in control['routes']if record['name']==CPU_ROUTE['name']]
    need(len(anchors)==1 and anchors[0]['requested']==PARITY_ROUTES[2] and control['model_state_and_local_RNG_bookend_passed']is True and control['all_parameter_grad_fields_None']is True,'CPU_CONDITIONAL_PRIOR_CPU_ANCHOR')
    original_design=read(DESIGN); budget={**original_design['budget'],'baseline_forwards':96,'baseline_artifact_writes':1,'canonical_loss_calls':96,'independent_BCE_calls':192,'grad_enabled_forward_calls':192,'no_grad_forward_calls':0}
    del budget['parity_forwards']
    inputs={**base['inputs'],'batch_geometry_design':base['inputs']['design'],**CPU_EVIDENCE,'batch_geometry_plan':accepted['plan'],'batch_geometry_result':accepted['result'],'batch_geometry_terminal':accepted['terminal'],
            'scope':CPU_SCOPE,'parent_post_readout_scope':scope['parent_post_readout_scope']}
    plan={**base,'schema_version':'gx1_provider_hlc_CPU_conditional_gradient_plan_v1','stage':CPU_STAGE,'plan_path':str(CPU_PLAN),'output_root':str(CPU_ROOT),'inputs':inputs,
          'execution':CPU_EXECUTION,'budget':budget,'panel':original_design['panel'],'route':CPU_ROUTE,'scope':scope,'question':scope['question'],'source_parent_commit':CPU_PARENT,
          'prior_CPU_anchor':{'route':anchors[0],'selected_issue_ns':control['selected_issue_ns'],'selected_source_row':control['selected_source_row'],'feature_proof':control['first_row_feature_proof']},
          'tolerances':TOLERANCES,'fixed_original_tolerance':TOLERANCES['saved_FINAL_logits'],'original_saved_CUDA_gate':'FAILED','full96_gradient_qualification':False,
          'method':{'CPU_scope':scope,'component_contract':original_design['method']},'limits':[scope['why_new_scope'],scope['decision_before_run']['both_outcomes'],resources['actual_full96_gradient_peak'],
                    'Original B96 CPU-eval local interaction only: at most192 forwards/384 autograd calls; no backward(), optimizer or parameter update, checkpoint RNG restoration, refit, new source HTTP, selected VAL/TEST, Entry/Exit or trading.']}
    checked_inputs(plan); return plan


def gate_cpu_conditional_expected(binding,admitted):
    return {'status':'ADMITTED_EXACT_CPU_CONDITIONAL_PROVIDER_HLC_GRADIENT'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}


def inspect_cpu_conditional(sha,*,admitted):
    need(Path.cwd().resolve()==REPO and type(sha)is str and len(sha)==64 and all(c in '0123456789abcdef'for c in sha),'CPU_CONDITIONAL_CWD_SHA')
    binding={'path':str(CPU_PLAN),'sha256':sha}; plan=read(binding)
    need(plan==build_cpu_conditional_plan(source_base_commit=plan['source_base_commit']) and source_clean(plan),'CPU_CONDITIONAL_EXACT_PLAN_SOURCE')
    pref=ref(POLICY); policy=read(pref)
    need(all(policy.get(key)is False for key in saved.FLAGS) and policy[CPU_STAGE]==gate_cpu_conditional_expected(binding,admitted),'CPU_CONDITIONAL_ONE_USE_FLAGS')
    for stage,role in ((STAGE,'failed_gradient_plan'),(PARITY_STAGE,'first_parity_plan'),(BATCH_STAGE,'batch_geometry_plan')):
        old=policy[stage]
        need(old['plan']==plan['inputs'][role] and old['execution_authorized']is False and old['remaining_invocations']==0 and old['relaunch_allowed']is False
             and old['training_authorized']is False and old['test_authorized']is False,'CPU_CONDITIONAL_OLD_PLANS_CLOSED')
    need(not CPU_ROOT.exists() and not CPU_ROOT.is_symlink(),'CPU_CONDITIONAL_FRESH_ROOT'); return plan,binding,pref


def cpu_conditional_compare(reference_value,candidate,*,tolerance):
    need(all(isinstance(value,np.ndarray)and value.dtype==np.float32 and value.shape==(96,5)and np.isfinite(value).all()for value in (reference_value,candidate)),'CPU_CONDITIONAL_COMPARISON_DOMAIN')
    rows=[first_parity_compare(reference_value[index:index+1],candidate[index:index+1],tolerance=tolerance)for index in range(96)]; passed=sum(sum(row['passed_mask'])for row in rows)
    return {'rows':96,'per_row':rows,'max_absolute_delta':max(row['max_absolute_delta']for row in rows),'per_TF_max_absolute_delta':[max(row['absolute_delta'][j]for row in rows)for j in range(5)],
            'passed_cells':passed,'failed_cells':480-passed,'all_within_original_tolerance':passed==480}


def cpu_conditional_validate_counts(counts,budget):
    expected={'full_feature_archive_loads':1,'feature_arrays_decoded':27,'target_archive_loads':1,'target_arrays_decoded':14,'capture_archive_loads':1,'capture_arrays_decoded':3,
              'checkpoint_deserializations':1,'model_constructions':1,'model_state_install_calls':1,'baseline_forwards':96,'gradient_forwards':96,'NN_forwards':192,'autograd_calls':384,
              'backward_calls':0,'optimizer_constructions':0,'optimizer_restores':0,'optimizer_updates':0,'checkpoint_RNG_restore_calls':0,'fits':0,'prior_refits':0,'new_source_HTTP_calls':0,
              'canonical_loss_calls':96,'independent_BCE_calls':192,'baseline_artifact_writes':1,'grad_enabled_forward_calls':192,'no_grad_forward_calls':0}
    need(counts==expected and all(type(value)is int for value in counts.values()) and budget['NN_forwards_max']==192 and budget['autograd_calls_max']==384,'CPU_CONDITIONAL_EXACT_COUNTS'); return True


def calculate_cpu_conditional(plan,*,progress=None):
    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as body
    torch.set_num_threads(1); need(torch.get_num_threads()==1 and not torch.cuda.is_initialized(),'CPU_CONDITIONAL_THREADS_CUDA')
    progress={}if progress is None else progress; refs=progress.get('execution_refs')
    need(type(refs)is dict and set(refs)=={'plan','claim','policy_at_claim'} and all(type(value)is dict and set(value)=={'path','sha256'}for value in refs.values()),'CPU_CONDITIONAL_EXECUTION_REFS')
    need(refs['plan']['path']==str(CPU_PLAN) and refs['claim']['path']==str(CPU_ROOT/'CLAIM.json') and refs['policy_at_claim']['path']==str(POLICY),'CPU_CONDITIONAL_EXECUTION_REF_PATHS')
    for value in refs.values():reference(value)
    need(read(refs['plan'])==plan,'CPU_CONDITIONAL_EXECUTION_PLAN_BINDING')
    claim=read(refs['claim']); policy=read(refs['policy_at_claim'])
    need(claim['stage']==CPU_STAGE and claim['plan']==refs['plan'] and claim['policy_at_claim']==refs['policy_at_claim'] and claim['remaining_invocations']==0 and claim['relaunch_allowed']is False
         and all(policy.get(key)is False for key in saved.FLAGS) and policy[CPU_STAGE]==gate_cpu_conditional_expected(refs['plan'],True),'CPU_CONDITIONAL_EXECUTION_CLAIM_POLICY_BINDING')
    budget=plan['budget']; counts=dict.fromkeys(('full_feature_archive_loads','feature_arrays_decoded','target_archive_loads','target_arrays_decoded','capture_archive_loads','capture_arrays_decoded','checkpoint_deserializations','model_constructions','model_state_install_calls','baseline_forwards','gradient_forwards','NN_forwards','autograd_calls','backward_calls','optimizer_constructions','optimizer_restores','optimizer_updates','checkpoint_RNG_restore_calls','fits','prior_refits','new_source_HTTP_calls','canonical_loss_calls','independent_BCE_calls','baseline_artifact_writes','grad_enabled_forward_calls','no_grad_forward_calls'),0)
    progress.update(phase='CPU_CONDITIONAL_CANONICAL_CONTEXT_B96',counts=counts,second_pass_rows=[])
    context=adapter.load_context(plan['metadata']['adapter_metadata']); counts.update(full_feature_archive_loads=1,feature_arrays_decoded=27,target_archive_loads=1,target_arrays_decoded=14)
    need(bool(np.all(context['provider'].bar_start_ns<1782864000000000000)),'CPU_CONDITIONAL_PRETEST_SOURCE_CUTOFF')
    take=np.arange(96,dtype=np.int64); packet=adapter.packet(context,take); selected_source_row=int(context['source_rows'][0])
    need(len(packet['truth']['issue_ns'])==96 and core.digest(packet['truth']['issue_ns'])==plan['panel']['issue_ns_sha256'] and core.digest(context['source_rows'][take])==plan['panel']['source_rows_sha256']
         and bool(np.all((packet['truth']['issue_ns']>=train.START)&(packet['truth']['issue_ns']<train.UPPER))) and len(packet['features'])==9
         and sum(v.numel()*v.element_size()for v in packet['features'].values())==budget['packet_feature_bytes'],'CPU_CONDITIONAL_EXACT_B96_PACKET')
    normalization=context['normalization']; prior=context['prior']; binding=context['input_binding']; target=context['target_contract']; del context; gc.collect()
    row0_proof={key:{'shape':list(value[:1].shape),'dtype':str(value.dtype),'sha256':core.digest(value[:1].detach().cpu().numpy())}for key,value in packet['features'].items()}
    anchor=plan['prior_CPU_anchor']; need(row0_proof==anchor['feature_proof'] and int(packet['truth']['issue_ns'][0])==anchor['selected_issue_ns'] and selected_source_row==anchor['selected_source_row'],'CPU_CONDITIONAL_PRIOR_INPUT_IDENTITY')
    progress['phase']='CPU_CONDITIONAL_CONTEXT_RELEASED_CAPTURE3'; capture=saved.load_fields(plan['inputs']['capture'],CAPTURE,plan['capture_array_headers']); counts.update(capture_archive_loads=1,capture_arrays_decoded=3)
    need(sum(v.nbytes for v in capture.values())==budget['capture_ndarray_bytes_decoded'] and core.digest(capture['issue_ns'])==plan['TRAIN_issue_ns_sha256'] and np.array_equal(capture['issue_ns'][:96],packet['truth']['issue_ns']),'CPU_CONDITIONAL_SAVED_CLOCK_BYTES')
    gpu={key:np.ascontiguousarray(capture[key][:96]).copy()for key in CAPTURE[1:]}; del capture; gc.collect()
    progress['phase']='CPU_CONDITIONAL_FACTORY_STRICT_FINAL'
    model=model_owner.build_native_future_state_model(input_normalization=normalization,input_binding=binding,architecture=plan['metadata']['original_epoch_plan']['metadata']['architecture'],device=torch.device('cpu'),train_prior=prior,target_contract=target,backbone_gradient_policy='M5_primary'); counts['model_constructions']+=1
    cp=safe_load_final(plan); counts['checkpoint_deserializations']+=1; model.load_state_dict(cp['model_state'],strict=True); counts['model_state_install_calls']+=1
    need(core.state_sha(model.state_dict())==plan['metadata']['final_receipt']['proof']['model_state'],'CPU_CONDITIONAL_INSTALLED_FINAL_MODEL_PROOF'); del cp; gc.collect(); model.eval()
    state_before=core.state_sha(model.state_dict()); rng_before=core.shared_gradient_rng(); packet_before=core.state_sha(packet); ambient_grad=torch.is_grad_enabled()
    need(all(p.grad is None and p.device.type=='cpu'and p.dtype==torch.float32 for p in model.parameters()),'CPU_CONDITIONAL_PARAMETER_GRAD_NONE')
    denominator=int(packet['truth']['activity_known'].sum()); need(denominator>0,'CPU_CONDITIONAL_FULL_B96_DENOMINATOR')
    identity={'issue_ns_sha256':core.digest(packet['truth']['issue_ns']),'source_rows_sha256':plan['panel']['source_rows_sha256'],'packet_sha256':packet_before,'row0_feature_proof':row0_proof}
    baseline={key:np.empty((96,5),np.float32)for key in CAPTURE[1:]}; replay={key:np.empty((96,5),np.float32)for key in CAPTURE[1:]}; phase='baseline_forwards'
    def record_array(value):return {'values':value.astype(np.float64).tolist(),'shape':list(value.shape),'dtype':value.dtype.str,'sha256':core.digest(value)}
    def context_bookend():need(body._BATCH_EQUAL_LENGTH_MTF_EVAL.get()is False and body._MODEL_FINITE_CHECK_SCOPE.get()is None and torch.is_grad_enabled()==ambient_grad,'CPU_CONDITIONAL_CONTEXT_RESTORE')
    def books():
        context_bookend(); measurement_bookend(model,state_before,rng_before); need(core.state_sha(packet)==packet_before,'CPU_CONDITIONAL_PACKET_MUTATION'); return True
    def count_forward(*_):
        counts['NN_forwards']+=1; counts[phase]+=1; counts['grad_enabled_forward_calls'if torch.is_grad_enabled()else'no_grad_forward_calls']+=1
        need(counts['NN_forwards']<=budget['NN_forwards_max'],'CPU_CONDITIONAL_FORWARD_BUDGET')
    def chunk(index):return {'features':{key:value[index:index+1]for key,value in packet['features'].items()},'truth':train.slice_provider_hlc_truth(packet['truth'],np.array([index],np.int64),target_contract=target)}
    def forward(current):
        output,hidden,activation=core.future_cuda_forward(model,current['features'],grouped=True,equal_tf=False,capture=False)
        groups={}
        for tf in train.TF:groups.setdefault(str(current['features']['seq_'+tf.lower()].shape[1]),[]).append(tf)
        need(hidden is None and set(output)==set(CAPTURE[1:]) and activation=={'batch_eval_active':False,'model_training':False,'grad_enabled':True,'scope_enabled':False,'eligible_equal_length_TF_groups':groups},'CPU_CONDITIONAL_ACTUAL_ROUTE')
        need(all(value.dtype==torch.float32 and value.device.type=='cpu'and tuple(value.shape)==(1,5)and bool(torch.isfinite(value).all())for value in output.values()),'CPU_CONDITIONAL_FINITE_HEADS'); return output
    context_bookend(); hook=model.register_forward_pre_hook(count_forward); anchor_comparison={}
    try:
        progress['phase']='CPU_CONDITIONAL_FULL_BASELINE_BEFORE_DERIVATIVES'
        try:
            with torch.enable_grad():
                for index in range(96):
                    current=chunk(index); output=forward(current)
                    for key,value in output.items():baseline[key][index:index+1]=value.detach().cpu().numpy()
                    if index==0:
                        for key in CAPTURE[1:]:
                            record=anchor['route']['logits'][key]; reference_value=np.array([record['values']],np.float32)
                            need(core.digest(reference_value)==record['sha256'] and list(reference_value.shape)==record['shape']==[1,5] and reference_value.dtype.str==record['dtype'],'CPU_CONDITIONAL_ANCHOR_LOGIT_SHA')
                            anchor_comparison[key]=first_parity_compare(reference_value,baseline[key][:1],tolerance=plan['fixed_original_tolerance'])
                        progress['prior_CPU_anchor_comparison']=anchor_comparison
                        need(all(cell['all_within_original_tolerance']for cell in anchor_comparison.values()),'CPU_CONDITIONAL_PRIOR_CPU_ANCHOR_MISMATCH')
                    del output,current
        finally:context_bookend()
        books(); need(counts['NN_forwards']==counts['baseline_forwards']==96 and counts['autograd_calls']==0,'CPU_CONDITIONAL_BASELINE_BEFORE_GRADIENTS')
        gpu_comparison={key:cpu_conditional_compare(gpu[key],value,tolerance=plan['fixed_original_tolerance'])for key,value in baseline.items()}
        baseline_counts=dict(counts)
        artifact={'status':'CPU_BASELINE_COMPLETE_BEFORE_DERIVATIVES','stage':CPU_STAGE,'plan':refs['plan'],'claim':refs['claim'],'policy_at_claim':refs['policy_at_claim'],'scope':CPU_SCOPE,
                  'route':plan['route'],'panel':plan['panel'],'counts':baseline_counts,'outputs':{key:record_array(value)for key,value in baseline.items()},'saved_CUDA_comparison':gpu_comparison,
                  'prior_CPU_anchor_comparison':anchor_comparison,'input_identity':identity,'known_activity_denominator_full_B96':denominator,'model_state_sha256':state_before,
                  'local_RNG_sha256':core.state_sha(rng_before),'bookends_passed':True,'original_saved_CUDA_gate':'FAILED','full96_gradient_qualification':False,'autograd_calls_before_publication':0}
        baseline_ref=cpu_conditional_publish('CPU_BASELINE.json',artifact); counts['baseline_artifact_writes']+=1; progress.update(cpu_baseline=baseline_ref,baseline_counts=baseline_counts,phase='CPU_CONDITIONAL_REPLAY_AND_COMPONENT_GRADIENTS')
        named=tuple((name,p)for name,p in model.backbone.named_parameters()if p.requires_grad); params=tuple(p for _,p in named); need(bool(params),'CPU_CONDITIONAL_PARAMETER_AXIS')
        sums={term:[None]*len(params)for term in TERMS}; losses={term:0. for term in TERMS}; phase='gradient_forwards'
        try:
            with torch.enable_grad():
                for index in range(96):
                    current=chunk(index); output=forward(current); comparisons={}
                    for key,value in output.items():
                        replay[key][index:index+1]=value.detach().cpu().numpy(); comparisons[key]=first_parity_compare(baseline[key][index:index+1],replay[key][index:index+1],tolerance=plan['fixed_original_tolerance'])
                    progress['second_pass_rows'].append({'index':index,'issue_ns':int(current['truth']['issue_ns'][0]),'outputs':{key:record_array(value[index:index+1])for key,value in replay.items()},'vs_CPU_baseline':comparisons})
                    need(all(cell['all_within_original_tolerance']for cell in comparisons.values()),'CPU_CONDITIONAL_REPLAY_MISMATCH_BEFORE_DERIVATIVES')
                    canonical=objective.native_future_state_loss(model_output=output,truth=current['truth'],fold_upper_ns=train.UPPER,target_contract=target); counts['canonical_loss_calls']+=1
                    values={}
                    for name,key,label,mask in (('activity','activity_logits','activity','activity_known'),('side','side_logits','side_up','side_known')):
                        y=torch.as_tensor(current['truth'][label].copy(),dtype=output[key].dtype); known=torch.as_tensor(current['truth'][mask].copy(),dtype=torch.bool)
                        bce=torch.nn.functional.binary_cross_entropy_with_logits(output[key],y,reduction='none'); counts['independent_BCE_calls']+=1; values[name]=torch.where(known,bce,torch.zeros_like(bce))
                    terms=(scaled_canonical_loss(canonical['loss'],canonical['count'],denominator),values['activity'][:,0].sum()/denominator,values['side'][:,0].sum()/denominator,(values['activity'][:,1:].sum()+values['side'][:,1:].sum())/denominator)
                    for j,(name,loss)in enumerate(zip(TERMS,terms)):
                        need(bool(torch.isfinite(loss)),'CPU_CONDITIONAL_FINITE_COMPONENT_LOSS'); gradients=torch.autograd.grad(loss,params,retain_graph=j<3,allow_unused=True); counts['autograd_calls']+=1
                        need(counts['autograd_calls']<=budget['autograd_calls_max'],'CPU_CONDITIONAL_AUTOGRAD_BUDGET'); accumulate_gradients(sums[name],gradients); losses[name]+=float(loss.detach()); del gradients
                    del output,current,canonical,values,terms,loss
        finally:context_bookend()
        proof=gradient_report(*(sums[name]for name in TERMS),tolerances=plan['tolerances']); del sums
        bookend=books(); cpu_conditional_validate_counts(counts,budget)
    finally:hook.remove()
    progress['phase']='COMPLETE_CPU_CONDITIONAL_GRADIENT_MEASUREMENT'
    return {'payload_budget':budget,'counts':counts,'route':plan['route'],'cpu_baseline':baseline_ref,'baseline_counts':baseline_counts,'cpu_baseline_outputs':artifact['outputs'],
            'cpu_replay_outputs':{key:record_array(value)for key,value in replay.items()},'cpu_replay_comparison':{key:cpu_conditional_compare(baseline[key],value,tolerance=plan['fixed_original_tolerance'])for key,value in replay.items()},
            'saved_CUDA_comparison':gpu_comparison,'original_saved_CUDA_gate':'FAILED','saved_CUDA_comparison_is_informational':True,'full96_gradient_qualification':False,
            'gradient_proof':proof,'known_activity_denominator_full_B96':denominator,'normalized_loss_components':losses,'per_TF_support':{'activity':packet['truth']['activity_known'].sum(axis=0).astype(int).tolist(),'side':packet['truth']['side_known'].sum(axis=0).astype(int).tolist()},
            'parameter_axis':{'names':[name for name,_ in named],'tensors':len(named),'aggregation_dtype':'float64','None_is_mathematical_zero_only':True},'input_identity':identity,'prior_CPU_anchor_comparison':anchor_comparison,
            'CPU_baseline_published_before_derivatives':True,'old_context_released_before_model_construction':True,'final_all_ten_checkpoint_proofs_verified':True,'strict_FINAL_model_state_installed':True,
            'model_state_and_local_RNG_bookend_passed':bookend,'context_bookends_passed':True,'ambient_grad_mode_restored':True,'all_parameter_grad_fields_None':True,'retained_B96_packet_unmutated':True,
            'local_RNG_bookend_scope':['python','numpy','torch_cpu'],'measurement_completed':True,'CPU_conditional_local_interaction_only':True,'actual_training_cause':'UNKNOWN','selected_causal_fix':'UNKNOWN','later_generalization':'UNKNOWN','economic_edge':'UNKNOWN','all20_approved':False,'limits':plan['limits']}


def cpu_conditional_publish(name,value):
    path=CPU_ROOT/name; saved._publish_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()); return ref(path)


def run_cpu_conditional(sha):
    began=time.monotonic(); cap=require_capped_cpu_audit_execution(); lock=require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==4*1024**3 and cap['swap']==512*1024**2 and cap['numerical_threads']==1 and cap['pids']==64
         and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):],'CPU_CONDITIONAL_CAPS')
    plan,binding,pref=inspect_cpu_conditional(sha,admitted=True); checked_inputs(plan); boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(); CPU_ROOT.mkdir(parents=True,exist_ok=False)
    handlers={sig:signal.getsignal(sig)for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}; claim=result=None; progress={'phase':'CLAIM'}
    try:
        def expired(*_):raise TimeoutError('PROVIDER_HLC_CPU_CONDITIONAL_WALL')
        def stopped(sig,*_):raise InterruptedError('PROVIDER_HLC_CPU_CONDITIONAL_SIGNAL_'+str(sig))
        signal.signal(signal.SIGALRM,expired); signal.signal(signal.SIGTERM,stopped); signal.signal(signal.SIGINT,stopped)
        claim=cpu_conditional_publish('CLAIM.json',{'stage':CPU_STAGE,'plan':binding,'policy_at_claim':pref,'boot_id':boot,'remaining_invocations':0,'relaunch_allowed':False,'training_authorized':False,'test_data_used':False}); progress['execution_refs']={'plan':binding,'claim':claim,'policy_at_claim':pref}
        remaining=CPU_EXECUTION['internal_seconds']-(time.monotonic()-began); need(remaining>0,'CPU_CONDITIONAL_TIME_BEFORE_DECODE'); signal.setitimer(signal.ITIMER_REAL,remaining)
        allowed={plan['inputs'][role]['path']for role in ('feature_archive','capture','target_arrays','final_checkpoint')}
        def audit(event,args):
            if event in ('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('PROVIDER_HLC_CPU_CONDITIONAL_NETWORK_FORBIDDEN')
            if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
                name=os.fsdecode(args[0])
                if name.endswith(('.npz','.pt','.pth','.parquet','.safetensors','.pkl'))and str(Path(name).resolve())not in allowed:raise RuntimeError('PROVIDER_HLC_CPU_CONDITIONAL_PAYLOAD_FORBIDDEN')
        sys.addaudithook(audit); report=calculate_cpu_conditional(plan,progress=progress); checked_inputs(plan)
        need(source_clean(plan)and ref(POLICY)==pref and ref(CPU_PLAN)==binding and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,'CPU_CONDITIONAL_SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND')
        result=cpu_conditional_publish('RESULT.json',{'status':'COMPLETE_CONDITIONAL_PROVIDER_HLC_CPU_GRADIENT_MEASUREMENT','stage':CPU_STAGE,'plan':binding,'claim':claim,'policy_at_claim':pref,'boot_id':boot,'report':report,'source_unchanged':True,'test_data_used':False,'economic_edge_proven':False,'elapsed_seconds':time.monotonic()-began})
        terminal=cpu_conditional_publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'stage':CPU_STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False}); return {'result':result,'terminal':terminal}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0); cpu_conditional_publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'stage':CPU_STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'progress':progress,'error':type(error).__name__+':'+str(error)[:2000],'test_data_used':False}); raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in handlers.items():signal.signal(sig,handler)


def main_cpu_conditional():
    parser=argparse.ArgumentParser(); parser.add_argument('--first-parity-plan-sha256',required=True); parser.add_argument('--cpu-conditional-gradient',action='store_true',required=True); parser.add_argument('--preflight',action='store_true'); args=parser.parse_args()
    need(sys.argv[1:]==['--first-parity-plan-sha256',args.first_parity_plan_sha256,'--cpu-conditional-gradient']+(['--preflight']if args.preflight else[]),'CPU_CONDITIONAL_EXACT_CLI')
    if args.preflight:
        plan,binding,pref=inspect_cpu_conditional(args.first_parity_plan_sha256,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_PROVIDER_HLC_CPU_CONDITIONAL_GRADIENT_PASS','plan':binding,'policy':pref,'claim_created':False,'checkpoint_deserializations':0,'NPZ_arrays_decoded':0,'models':0,'NN_forwards':0,'autograd_calls':0,'CUDA_initialized':False}))
    else:print(json.dumps(run_cpu_conditional(args.first_parity_plan_sha256),sort_keys=True))
# End CPU-conditional B96 gradients.

def main_first_parity():
    if sys.argv[3:4]==['--cpu-conditional-gradient']:
        return main_cpu_conditional()

    if sys.argv[3:4]==['--batch-geometry']:
        return main_batch_geometry()

    parser=argparse.ArgumentParser(); parser.add_argument('--first-parity-plan-sha256',required=True); parser.add_argument('--preflight',action='store_true'); args=parser.parse_args()
    need(sys.argv[1:]==['--first-parity-plan-sha256',args.first_parity_plan_sha256]+(['--preflight']if args.preflight else[]),'FIRST_PARITY_EXACT_CLI')
    if args.preflight:
        plan,binding,pref=inspect_first_parity(args.first_parity_plan_sha256,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_PROVIDER_HLC_FIRST_PARITY_PASS','plan':binding,'policy':pref,'claim_created':False,'checkpoint_deserializations':0,'NPZ_arrays_decoded':0,'models':0,'NN_forwards':0,'autograd_calls':0,'CUDA_initialized':False}))
    else:print(json.dumps(run_first_parity(args.first_parity_plan_sha256),sort_keys=True))
# End first-row parity measurement.

def main():
    if sys.argv[1:2]==['--first-parity-plan-sha256']:
        return main_first_parity()

    parser=argparse.ArgumentParser(); parser.add_argument('--plan-sha256',required=True); parser.add_argument('--preflight',action='store_true'); args=parser.parse_args()
    need(sys.argv[1:]==['--plan-sha256',args.plan_sha256]+(['--preflight']if args.preflight else[]),'EXACT_CLI')
    if args.preflight:
        plan,binding,pref=inspect(args.plan_sha256,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_PROVIDER_HLC_FROZEN_GRADIENT_PASS','plan':binding,'policy':pref,'claim_created':False,'checkpoint_deserializations':0,'NPZ_arrays_decoded':0,'models':0,'NN_forwards':0,'autograd_calls':0,'CUDA_initialized':False}))
    else:print(json.dumps(run(args.plan_sha256),sort_keys=True))


if __name__=='__main__':main()
