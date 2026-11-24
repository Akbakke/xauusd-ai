"""One-use CPU attribution of authenticated stored logits; never model execution."""
from __future__ import annotations
import argparse, ast, hashlib, json, os, signal, subprocess, sys, time, zipfile
from pathlib import Path
if __name__ == '__main__':
    from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
    install_native_m5_direction_io_firewall()
import numpy as np
import torch
from gx1.audit import diagnose_native_future_state_alignment_v1 as alignment
from gx1.audit import native_future_state_training_v1 as train
from gx1.contracts import market_state_ta_integrated_objective_v1 as objective

saved, core = alignment.saved, train.core
REPO, POLICY = alignment.REPO, alignment.POLICY
NAME = 'NATIVE_PROVIDER_HLC_READOUT_ATTRIBUTION_20261124_001'
MODULE = 'gx1.audit.diagnose_native_provider_hlc_readout_v1'
STAGE = NAME.lower(); PLAN = REPO/'configs/research'/(NAME+'.json'); ROOT = Path('/home/andre2/GX1_RUNS')/NAME
SELF = 'gx1/audit/diagnose_native_provider_hlc_readout_v1.py'; TEST = 'tests/test_diagnose_native_provider_hlc_readout_v1.py'
DESIGN = {'path': '/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/PROVIDER_HLC_READOUT_ATTRIBUTION_DESIGN001.json',
          'sha256': '7748a1e6a75c21d2f0e99f67da40076ecc46845e2cf7164371a4cf4dfaf6e367'}
EXECUTION = alignment.EXECUTION
TF = train.TF; ROWS = 354550; CAPTURE = ('issue_ns','side_logits'); MASK = ('issue_ns','side_known')
NOT_JSON = {'initial_checkpoint','final_checkpoint','capture','target_arrays','main_goal'}
ref, read = saved.ref, saved.read
reference = alignment.provider_hlc_batch_reference
source_clean = alignment.source_clean
require_capped_cpu_audit_execution = alignment.require_capped_cpu_audit_execution
require_capped_lock_ancestry = alignment.require_capped_lock_ancestry


def need(ok, why):
    if not bool(ok): raise ValueError('PROVIDER_HLC_READOUT_'+why)


def source_proof(design):
    bound = design['source_formula']; path = reference(bound['model_owner'])
    need(bound['side_weight_key']=='readout.side_heads.{TF}.weight' and bound['side_bias_key']=='readout.side_heads.{TF}.bias'
         and bound['weight_shape']==[1,128] and bound['bias_shape']==[1] and bound['dtype']=='float32', 'SOURCE_PARAMETER_ROLES')
    symbols = {node.name: hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
               for node in ast.parse(path.read_text()).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))
               and node.name in bound['symbols_AST_sha256']}
    need(symbols==bound['symbols_AST_sha256'], 'SOURCE_FORMULA_AST')
    return {'model_owner':bound['model_owner'],'symbols_AST_sha256':symbols,
            'formula':'z_tf=nn.Linear(128,1)(h_tf)=w_tf dot h_tf+b_tf; initial w=0 and b=frozen-prior float32 logit',
            'parameter_names':{tf:{'weight':bound['side_weight_key'].format(TF=tf),'bias':bound['side_bias_key'].format(TF=tf)} for tf in TF}}


def checkpoint_zip_metadata(binding):
    path=reference(binding)
    with zipfile.ZipFile(path) as archive:
        records=archive.infolist(); names=[item.filename for item in records]
        need(len(names)==len(set(names)) and all(not Path(name).is_absolute() and '..'not in Path(name).parts for name in names), 'CP_ZIP_DIRECTORY')
        storage=[item for item in records if '/data/'in item.filename and item.filename.rsplit('/',1)[-1].isdigit()]
        need(len([name for name in names if name.endswith('/data.pkl')])==1 and bool(storage), 'CP_ZIP_STORAGE')
        return {'serialized_bytes':path.stat().st_size,'directory_records':len(records),
                'tensor_storage_records':len(storage),'tensor_storage_bytes':sum(item.file_size for item in storage),
                'inspection':'ZIP central directory only; no pickle or tensor payload decoded'}


def build_plan(*,source_base_commit=None):
    d=read(DESIGN)
    need(d['schema_version']=='gx1_provider_hlc_readout_attribution_design_v1' and d['basename']==NAME and d['stage']==STAGE
         and d['plan_path']==str(PLAN) and d['output_root']==str(ROOT) and d['module']==MODULE
         and d['execution']==EXECUTION and d['execution_authorized']is False and d['training_authorized']is False
         and d['test_authorized']is False and d['no_more_epochs']is True and d['model_parameters_changed']is False
         and set(d['input_roles_not_JSON'])==NOT_JSON and len(d['input_roles_not_JSON'])==len(NOT_JSON), 'LOCKED_DESIGN')
    inputs={**d['inputs'],'design':DESIGN}
    for binding in inputs.values(): reference(binding)
    meta={key:read(binding) for key,binding in inputs.items() if key not in NOT_JSON}
    original=meta['original_epoch_plan']; initial=meta['initial_plan']; completed=meta['completed_model_plan']
    result=meta['completed_model_result']; terminal=meta['completed_model_terminal']; report=result['report']; trajectory=report['trajectory']
    ir,fr=meta['initial_receipt'],meta['final_receipt']; initial_report=meta['initial_result']['report']; prior=meta['prior']; dataset=meta['dataset']; quality=meta['full_quality']
    need(terminal['status']=='COMPLETE' and terminal['exit_code']==0 and terminal['result']==inputs['completed_model_result']
         and terminal['plan']==result['plan']==inputs['completed_model_plan'] and result['source_inputs_policy_boot_unchanged']is True
         and result['test_data_used']is False and report['actual_optimizer_updates']==3694 and report['full_TRAIN_rows']==ROWS and report['passes']==1
         and report['original_plan']==inputs['original_epoch_plan'] and report['final_checkpoint_receipt']==inputs['final_receipt']
         and report['final_logits']==inputs['capture'] and report['final_quality']==inputs['full_quality']
         and completed['metadata']['original']==original and trajectory==completed['metadata']['trajectory'], 'COMPLETE_EPOCH_IDENTITIES')
    need(initial['model_contract']==original['model_contract']==trajectory['model_contract']
         and initial['metadata']['input_binding']==original['metadata']['input_binding']
         and initial['metadata']['normalization']==original['metadata']['normalization']
         and initial['metadata']['target_contract']==original['metadata']['target_contract']==dataset['contract']
         and initial['metadata']['prior']==original['metadata']['prior']==inputs['prior']
         and initial['metadata']['dataset']==original['metadata']['dataset']==inputs['dataset']
         and initial['recipe']['seed']==original['recipe']['seed'] and initial['metadata']['TRAIN_rows']==ROWS, 'SAME_QUALIFIED_FUNCTION')
    contract=original['model_contract']; target=original['metadata']['target_contract']; truth_sha=original['metadata']['truth_sha256']; issue_sha=original['schedule']['issue_ns_sha256']
    prior=objective.require_native_future_state_train_prior(prior,target_contract=target)
    need(prior['fit_population']['issue_rows']==ROWS and prior['fit_population']['issue_ns_sha256']==issue_sha==dataset['TRAIN_issue_ns_sha256']
         and prior['fit_population']['truth_sha256']==truth_sha==initial['metadata']['truth_sha256']==report['whole_FINAL_score']['truth_sha256']==quality['truth_sha256']
         and contract['readout']['d_model']==128 and contract['train_prior_sha256']==prior['contract_sha256']==quality['frozen_prior_contract_sha256']
         and report['whole_FINAL_score']['output_sha256']==quality['output_sha256'], 'MATCHED_TRUTH_PRIOR_OUTPUT')
    need(ir['schema_version']=='gx1_native_provider_hlc_fresh_initial_checkpoint_v1' and ir['initial_only']is True and ir['optimizer_updates']==0
         and ir['checkpoint']==inputs['initial_checkpoint'] and ir['plan']==inputs['initial_plan']==meta['initial_result']['plan']
         and ir['model_state_sha256']==initial_report['initial_state_sha256']==report['fresh_initial_state_sha256']
         and ir['model_contract_sha256']==contract['contract_sha256'] and ir['target_contract_sha256']==target['contract_sha256']
         and initial_report['whole_TRAIN_initialbaseline_qualified']is True and initial_report['initial']['initial_prior_exact']is True
         and meta['initial_acceptance']['full_initialbaseline_qualified']is True and meta['initial_acceptance']['result']==inputs['initial_result']
         and meta['initial_acceptance']['initial_receipt']==inputs['initial_receipt'], 'INITIAL_RECEIPT_METADATA')
    train.require_recovery_checkpoint_proof(fr['proof'])
    need(fr['schema_version']=='gx1_native_future_state_training_checkpoint_receipt_v1' and fr['checkpoint']==inputs['final_checkpoint']
         and fr['original_plan']==inputs['original_epoch_plan'] and fr['execution_plan']==inputs['completed_model_plan'] and fr['producer_lane']=='main'
         and type(fr['logical_cursor'])is int and type(fr['optimizer_updates'])is int and fr['logical_cursor']==fr['optimizer_updates']==3694
         and fr['trajectory_sha256']==trajectory['contract_sha256'] and fr['proof']['trajectory']==core.state_sha(trajectory)
         and fr['proof']['position']==core.state_sha(train.position(original['schedule'],3694))
         and fr['proof']['model_state']==report['whole_FINAL_score']['state_sha256_before_and_after']
         and fr['proof']['rng_state']==report['whole_FINAL_score']['rng_sha256_before_and_restored'], 'FINAL_RECEIPT_METADATA')
    batch=meta['batch_risk_result']; batch_terminal=meta['batch_risk_terminal']; batch_plan=meta['batch_risk_plan']; batch_report=batch['report']
    from gx1.audit.native_provider_hlc_learning_v1 import resume_reference
    accepted=meta['batch_risk_acceptance']['result']; resume_reference(accepted,Path(inputs['batch_risk_result']['path']))
    need(batch['status']=='COMPLETE_CONDITIONAL_PROVIDER_HLC_BATCH_ARITHMETIC' and batch['source_unchanged']is True and batch['test_data_used']is False
         and batch['plan']==inputs['batch_risk_plan'] and batch_terminal['status']=='COMPLETE' and batch_terminal['exit_code']==0
         and batch_terminal['result']==inputs['batch_risk_result'] and {key:accepted[key]for key in ('path','sha256')}==inputs['batch_risk_result']
         and batch_report['canonical_v2_truth_prior_and_recorded_scores_verified']is True
         and batch_report['complete_issue_clock_and_original_B96_boundaries_verified']is True
         and all(batch_plan['inputs'][key]==inputs[key] for key in ('capture','dataset','target_arrays','prior','certificate','full_quality'))
         and batch_plan['truth_sha256']==truth_sha and batch_plan['TRAIN_issue_ns_sha256']==issue_sha
         and {key:dataset['target_observation_certificate_ref'][key]for key in ('path','sha256')}==inputs['certificate'], 'REUSED_CERTIFIED_PARTIAL_MASK_PROOF')
    capture_headers={'issue_ns':{'shape':[ROWS],'dtype':'<i8','sha256':issue_sha},
                     'side_logits':{'shape':[ROWS,5],'dtype':'<f4','sha256':quality['output_sha256']['side_logits']}}
    target_headers={key:dataset['array_headers'][key] for key in MASK}; budget=d['budget']
    need(tuple(budget['capture_fields'])==CAPTURE and tuple(budget['target_fields'])==MASK
         and budget['capture_NPZ_decoded_bytes']==9927400 and budget['target_NPZ_decoded_bytes']==4609150
         and sum(int(np.prod(h['shape']))*np.dtype(h['dtype']).itemsize for h in [*capture_headers.values(),*target_headers.values()])==budget['NPZ_ndarray_bytes_decoded']==14536550
         and budget['checkpoint_deserializations']==2 and budget['initial_checkpoint_bytes']==39178552 and budget['final_checkpoint_bytes']==97914007
         and budget['checkpoint_serialized_bytes']==137092559 and budget['checkpoint_load_policy']=={'map_location':'cpu','weights_only':True,'fallback_or_add_safe_globals':False}
         and budget['capture_archive_loads']==budget['target_archive_loads']==1 and budget['rows']==ROWS
         and all(budget[key]==0 for key in ('model_objects_constructed','model_state_install_calls','NN_forwards','backward_calls','autograd_calls','parameter_gradient_calculations','optimizer_constructions','optimizer_restores','optimizer_updates','RNG_restore_calls','feature_archive_loads','source_price_payloads','certificate_reauthentications','fits','prior_refits','HTTP_calls','VAL_TEST_rows','new_labels')), 'EXACT_DECODER_BUDGET')
    zip_metadata={name:checkpoint_zip_metadata(inputs[name+'_checkpoint']) for name in ('initial','final')}
    need(zip_metadata['initial']['serialized_bytes']==budget['initial_checkpoint_bytes'] and zip_metadata['final']['serialized_bytes']==budget['final_checkpoint_bytes'], 'CP_SERIALIZED_SIZES')
    formula=source_proof(d); names=set(batch_plan['source_owners'])|{SELF,TEST}
    need(all(not Path(name).is_absolute() and '..'not in Path(name).parts and Path(name).suffix in ('.py','.sh') for name in names), 'SOURCE_INVENTORY_NAMES')
    for name,digest in batch_plan['source_owners'].items(): need(saved.native.sha(REPO/name)==digest,'PREVIOUS_QUALIFIED_OWNER_UNCHANGED')
    plan={'schema_version':'gx1_provider_hlc_readout_attribution_plan_v1','stage':STAGE,'status':'PREPARED_NOT_ADMITTED','plan_path':str(PLAN),'output_root':str(ROOT),
          'source_base_commit':source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
          'source_owners':{name:saved.native.sha(REPO/name) for name in sorted(names)},'source_proof':formula,'inputs':inputs,'input_roles_not_JSON':sorted(NOT_JSON),
          'execution':EXECUTION,'budget':budget,'checkpoint_zip_metadata':zip_metadata,'capture_array_headers':capture_headers,'target_array_headers':target_headers,
          'capture_archive_inventory':list(alignment.HLC_BATCH_CAPTURE),'target_archive_inventory':list(dataset['array_headers']),
          'metadata':{'initial_plan':initial,'initial_receipt':ir,'final_receipt':fr,'original_epoch_plan':original,'trajectory':trajectory},
          'model_contract':contract,'target_contract':target,'truth_sha256':truth_sha,'TRAIN_issue_ns_sha256':issue_sha,'prior_contract_sha256':prior['contract_sha256'],
          'runtime_versions':{'numpy':np.__version__,'torch':str(torch.__version__)},'question':d['question'],'method':d['method'],'limits':d['limits'],'conditional_scope':d['conditional_scope']}
    checked_inputs(plan)
    return plan


def checked_inputs(plan):
    for binding in plan['inputs'].values(): reference(binding)
    saved.archive_headers(plan['inputs']['capture'],plan['capture_array_headers'],exact_inventory=plan['capture_archive_inventory'])
    saved.archive_headers(plan['inputs']['target_arrays'],plan['target_array_headers'],exact_inventory=plan['target_archive_inventory'])


def gate_expected(binding,admitted):
    return {'status':'ADMITTED_EXACT_CONDITIONAL_PROVIDER_HLC_READOUT_ATTRIBUTION'if admitted else'PREPARED_NOT_ADMITTED',
            'plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}


def inspect(sha,*,admitted):
    need(Path.cwd().resolve()==REPO and type(sha)is str and len(sha)==64 and all(c in '0123456789abcdef'for c in sha),'CWD_SHA')
    binding={'path':str(PLAN),'sha256':sha}; plan=read(binding)
    need(plan==build_plan(source_base_commit=plan['source_base_commit']) and source_clean(plan),'EXACT_PLAN_SOURCE')
    pref=ref(POLICY); policy=read(pref)
    need(all(policy.get(key)is False for key in saved.FLAGS) and policy[STAGE]==gate_expected(binding,admitted),'ONE_USE_ALL_TRAINING_CLOSED')
    need(not ROOT.exists() and not ROOT.is_symlink(),'FRESH_ROOT')
    return plan,binding,pref


def extract_heads(state,*,prior_bias=None):
    from collections.abc import Mapping
    need(isinstance(state,Mapping),'HEAD_STATE_MAPPING')
    if prior_bias is not None: need(isinstance(prior_bias,np.ndarray) and prior_bias.dtype==np.float32 and prior_bias.shape==(5,) and np.isfinite(prior_bias).all(),'INITIAL_PRIOR_BIAS')
    heads={}
    for j,tf in enumerate(TF):
        need(all('readout.side_heads.'+tf+'.'+kind in state for kind in ('weight','bias')),'HEAD_PARAMETER_NAMES')
        tensors={kind:state['readout.side_heads.'+tf+'.'+kind] for kind in ('weight','bias')}
        for kind,shape in (('weight',(1,128)),('bias',(1,))):
            value=tensors[kind]
            need(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.float32 and tuple(value.shape)==shape and not value.requires_grad and bool(torch.isfinite(value).all()),'CPU_HEAD_TENSOR')
        if prior_bias is not None: need(bool((tensors['weight']==0).all()) and float(tensors['bias'][0])==float(prior_bias[j]),'INITIAL_ZERO_WEIGHTS_EXACT_PRIOR_BIAS')
        heads[tf]={'bias':float(tensors['bias'][0]),'weight_sha256':core.state_sha(tensors['weight']),'bias_sha256':core.state_sha(tensors['bias'])}
    return heads


def _identity_buffers(state,contract):
    hashes={'future_state_target_sha256':contract['target_contract_sha256'],'future_state_objective_sha256':contract['objective_contract_sha256'],
            'future_state_prior_sha256':contract['train_prior_sha256'],'future_state_readout_sha256':contract['readout']['contract_sha256']}
    for name,digest in hashes.items():
        value=state['readout.'+name]
        need(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.uint8 and tuple(value.shape)==(32,)
             and value.tolist()==list(bytes.fromhex(digest)),'READOUT_IDENTITY_BUFFER')


def validate_initial(cp,receipt,initial_plan,initial_plan_ref,prior_bias):
    keys={'schema_version','plan','model_contract','input_binding','normalization','prior','dataset','target_contract','truth_sha256','seed','runtime','model_state','model_state_sha256','rng_state','training','optimizer','optimizer_updates','authority'}
    m=initial_plan['metadata']; contract=initial_plan['model_contract']
    need(type(cp)is dict and set(cp)==keys and cp['schema_version']=='gx1_native_provider_hlc_fresh_initial_checkpoint_v1'
         and cp['plan']==receipt['plan']==initial_plan_ref and cp['model_contract']==contract and cp['input_binding']==m['input_binding']
         and cp['normalization']==m['normalization'] and cp['prior']==m['prior'] and cp['dataset']==m['dataset'] and cp['target_contract']==m['target_contract']
         and cp['truth_sha256']==m['truth_sha256'] and cp['seed']==initial_plan['recipe']['seed'] and cp['training']is True
         and cp['optimizer']is None and type(cp['optimizer_updates'])is int and cp['optimizer_updates']==0 and cp['authority']=='INITIAL_ONLY_NO_FIT_OR_RESUME_AUTHORITY'
         and receipt['schema_version']==cp['schema_version'] and receipt['initial_only']is True and type(receipt['optimizer_updates'])is int and receipt['optimizer_updates']==0
         and receipt['model_contract_sha256']==contract['contract_sha256'] and receipt['target_contract_sha256']==m['target_contract']['contract_sha256'],'INITIAL_DISTINCT_SCHEMA_ROLES')
    core.future_cuda_runtime_matches(m['runtime'],cp['runtime'])
    need(core.state_sha(cp)==receipt['checkpoint_tree_sha256'] and core.state_sha(cp['model_state'])==cp['model_state_sha256']==receipt['model_state_sha256']
         and core.state_sha(cp['rng_state'])==receipt['rng_sha256'],'INITIAL_FULL_TREE_MODEL_RNG_PROOF')
    _identity_buffers(cp['model_state'],contract); heads=extract_heads(cp['model_state'],prior_bias=prior_bias)
    need(core.state_sha(cp)==receipt['checkpoint_tree_sha256'],'INITIAL_UNMUTATED_AFTER_EXTRACTION')
    return heads


def validate_final(cp,receipt,original,original_ref,trajectory,execution_ref):
    keys={'schema_version','trajectory','producer','position','model_state','optimizer_state','scheduler_state','rng_state','training','optimizer_parameter_names','steps','scores'}
    train.require_recovery_checkpoint_proof(receipt['proof'])
    need(type(cp)is dict and set(cp)==keys and cp['schema_version']=='gx1_native_future_state_training_checkpoint_v1'
         and cp['trajectory']==trajectory and cp['trajectory']['schedule']==original['schedule']
         and cp['producer']=={'original_plan':original_ref,'lane':'main','runtime':receipt['producer_runtime']}
         and receipt['schema_version']=='gx1_native_future_state_training_checkpoint_receipt_v1' and receipt['producer_lane']=='main'
         and receipt['original_plan']==original_ref and receipt['execution_plan']==execution_ref
         and type(receipt['logical_cursor'])is int and type(receipt['optimizer_updates'])is int and receipt['logical_cursor']==receipt['optimizer_updates']==3694 and cp['position']==train.position(original['schedule'],3694)
         and all(type(value)is int for value in cp['position'].values()) and cp['scheduler_state']is None and cp['training']is True and len(cp['steps'])==3694,
         'FINAL_DISTINCT_SCHEMA_PRODUCER_POSITION')
    need(train.checkpoint_proof(cp)==receipt['proof'],'FINAL_ALL_TEN_PROOFS')
    _identity_buffers(cp['model_state'],original['model_contract']); heads=extract_heads(cp['model_state'])
    need(train.checkpoint_proof(cp)==receipt['proof'],'FINAL_UNMUTATED_AFTER_EXTRACTION')
    return heads


def decompose(logits,known,initial_heads,final_heads):
    need(isinstance(logits,np.ndarray) and logits.dtype==np.float32 and logits.ndim==2 and logits.shape[0]>0 and logits.shape[1]==5 and np.isfinite(logits).all()
         and isinstance(known,np.ndarray) and known.dtype==np.bool_ and known.shape==logits.shape and set(initial_heads)==set(final_heads)==set(TF),'DECOMPOSITION_INPUTS')
    result={}
    for j,tf in enumerate(TF):
        b0,b1=initial_heads[tf]['bias'],final_heads[tf]['bias']; need(np.isfinite(b0) and np.isfinite(b1),'FINITE_BIASES'); db=b1-b0
        panels={}
        for name,mask in (('ALL',np.ones(len(logits),np.bool_)),('KNOWN_SIDE',known[:,j])):
            z=logits[mask,j].astype(np.float64); n=len(z)
            if not n:
                panels[name]={'count':0,'final_mean_logit':None,'mean_logit_displacement':None,'direct_bias_delta':db,'non_bias_residual_mean':None,'identity_abs_error':None,'mean_probability_final':None,'initial_probability':float(np.exp(-np.logaddexp(0.,-b0)))}
                continue
            mean=float(z.mean()); shift=mean-b0; residual=float((z-b1).mean()); error=abs(shift-(db+residual))
            need(np.isclose(shift,db+residual,rtol=0.,atol=1e-12),'FLOAT64_ADDITIVE_IDENTITY')
            panels[name]={'count':n,'final_mean_logit':mean,'mean_logit_displacement':shift,'direct_bias_delta':db,'non_bias_residual_mean':residual,'identity_abs_error':error,
                          'mean_probability_final':float(np.exp(-np.logaddexp(0.,-z)).mean()),'initial_probability':float(np.exp(-np.logaddexp(0.,-b0)))}
        result[tf]={'bias_initial':b0,'bias_final':b1,'bias_delta':db,'uniform_bias_probability_effect_abs_bound':abs(db)/4,**panels}
    return result


def calculate(plan,*,progress=None):
    torch.set_num_threads(1); need(torch.get_num_threads()==1 and not torch.cuda.is_initialized(),'CPU_ONLY_THREADS')
    progress={}if progress is None else progress; progress['phase']='INITIAL_CPU_WEIGHTS_ONLY_PROOF'; counts={'checkpoint_deserializations':0,'capture_archive_loads':0,'target_archive_loads':0}
    prior=objective.require_native_future_state_train_prior(read(plan['inputs']['prior']),target_contract=plan['target_contract'])
    prior_bias=objective.native_future_state_train_prior_logits(prior,rows=1,target_contract=plan['target_contract'])['side_logits'].numpy()[0]
    meta=plan['metadata']; reference(plan['inputs']['initial_checkpoint'])
    cp=torch.load(plan['inputs']['initial_checkpoint']['path'],map_location='cpu',weights_only=True); counts['checkpoint_deserializations']+=1
    initial=validate_initial(cp,meta['initial_receipt'],meta['initial_plan'],plan['inputs']['initial_plan'],prior_bias); del cp
    progress['phase']='FINAL_CPU_WEIGHTS_ONLY_PROOF'; reference(plan['inputs']['final_checkpoint'])
    cp=torch.load(plan['inputs']['final_checkpoint']['path'],map_location='cpu',weights_only=True); counts['checkpoint_deserializations']+=1
    final=validate_final(cp,meta['final_receipt'],meta['original_epoch_plan'],plan['inputs']['original_epoch_plan'],meta['trajectory'],plan['inputs']['completed_model_plan']); del cp
    progress['phase']='CAPTURE2_CERTIFIED_PARTIAL_MASK2'
    capture=saved.load_fields(plan['inputs']['capture'],CAPTURE,plan['capture_array_headers']); counts['capture_archive_loads']+=1
    mask=saved.load_fields(plan['inputs']['target_arrays'],MASK,plan['target_array_headers']); counts['target_archive_loads']+=1
    need(sum(value.nbytes for value in [*capture.values(),*mask.values()])==plan['budget']['NPZ_ndarray_bytes_decoded'] and np.array_equal(capture['issue_ns'],mask['issue_ns'])
         and capture['issue_ns'].shape==(ROWS,) and np.all(np.diff(capture['issue_ns'])>0) and saved.array_sha(capture['issue_ns'])==plan['TRAIN_issue_ns_sha256'],'EXACT_PARTIAL_MASK_CLOCK_BYTES')
    quality=read(plan['inputs']['full_quality']); batch=read(plan['inputs']['batch_risk_result'])['report']
    for j,tf in enumerate(TF):
        cell=prior['per_task']['side'][tf]; col=mask['side_known'][:,j]
        need(int(col.sum())==cell['count']==quality['all_TF'][tf]['side']['count'] and saved.array_sha(col)==cell['mask_sha256'],'REUSED_KNOWN_SIDE_MASK')
    panels=decompose(capture['side_logits'],mask['side_known'],initial,final)
    for j,tf in enumerate(TF):
        cell=panels[tf]['KNOWN_SIDE']; q=cell['mean_probability_final']
        need(np.isclose(q,quality['all_TF'][tf]['side']['model']['mean_positive_probability'],rtol=0.,atol=1e-12)
             and np.isclose(q,batch['per_tf'][tf]['FINAL_mean_q'],rtol=0.,atol=1e-12)
             and np.isclose(cell['initial_probability'],batch['per_tf'][tf]['prior_q'],rtol=0.,atol=1e-12),'QUALIFIED_PROBABILITY_CROSSCHECK')
    need(counts=={'checkpoint_deserializations':2,'capture_archive_loads':1,'target_archive_loads':1} and not torch.cuda.is_initialized(),'EXACT_CPU_DECODER_COUNTS')
    progress.update(phase='COMPLETE_STORED_LOGIT_ATTRIBUTION',**counts)
    return {'per_tf':panels,'parameter_binding':{'initial':initial,'final':final},'decoder_counts':counts,'payload_budget':plan['budget'],
            'count_semantics':'Whole checkpoint and selected NPZ deserializations; bytehash streaming and header/ZIP-only validation are separate I/O, not extra payload decodes.',
            'initial_full_tree_model_RNG_and_final_all_ten_proofs_verified':True,'partial_mask_reused_from_completed_canonical_audit':True,
            'model_objects_constructed':0,'NN_forwards':0,'autograd_calls':0,'parameter_gradient_calculations':0,'optimizer_updates':0,'model_state_install_calls':0,'RNG_restore_calls':0,
            'decomposition':'Mean stored zFINAL minus actual initial bias = actual bias delta + mean(stored zFINAL minus actual final bias). Residual combines projection, representation and floating-point rounding; W/H/history are not separated.',
            'probability_bound':'At fixed non-bias term, magnitude of uniform-bias probability effect is at most abs(delta_b)/4. No additive probability shares or parameter intervention.',
            'actual_training_cause':'UNKNOWN','selected_causal_fix':'UNKNOWN','later_generalization':'UNKNOWN','economic_edge':'UNKNOWN','all20_approved':False,'limits':plan['limits'],'conditional_scope':plan['conditional_scope']}


def publish(name,value):
    path=ROOT/name; saved._publish_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()); return ref(path)


def run(sha):
    began=time.monotonic(); cap=require_capped_cpu_audit_execution(); lock=require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==4*1024**3 and cap['swap']==512*1024**2 and cap['numerical_threads']==1
         and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):],'CPU_CAPS')
    plan,binding,pref=inspect(sha,admitted=True); checked_inputs(plan); boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(); ROOT.mkdir(parents=True,exist_ok=False)
    handlers={sig:signal.getsignal(sig)for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}; claim=result=None; progress={'phase':'CLAIM'}
    try:
        def expired(*_):raise TimeoutError('PROVIDER_HLC_READOUT_WALL')
        def stopped(sig,*_):raise InterruptedError('PROVIDER_HLC_READOUT_SIGNAL_'+str(sig))
        signal.signal(signal.SIGALRM,expired); signal.signal(signal.SIGTERM,stopped); signal.signal(signal.SIGINT,stopped)
        claim=publish('CLAIM.json',{'stage':STAGE,'plan':binding,'policy_at_claim':pref,'boot_id':boot,'remaining_invocations':0,'relaunch_allowed':False,'training_authorized':False,'test_data_used':False})
        remaining=EXECUTION['internal_seconds']-(time.monotonic()-began); need(remaining>0,'TIME_BEFORE_DECODE'); signal.setitimer(signal.ITIMER_REAL,remaining)
        allowed={plan['inputs'][role]['path']for role in ('initial_checkpoint','final_checkpoint','capture','target_arrays')}
        def audit(event,args):
            if event in ('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('PROVIDER_HLC_READOUT_NETWORK_FORBIDDEN')
            if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
                name=os.fsdecode(args[0])
                if name.endswith(('.npz','.pt','.pth','.parquet','.safetensors','.pkl'))and str(Path(name).resolve())not in allowed:raise RuntimeError('PROVIDER_HLC_READOUT_PAYLOAD_FORBIDDEN')
        sys.addaudithook(audit); report=calculate(plan,progress=progress); checked_inputs(plan)
        need(source_clean(plan)and ref(POLICY)==pref and ref(PLAN)==binding and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,'SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND')
        result=publish('RESULT.json',{'status':'COMPLETE_CONDITIONAL_PROVIDER_HLC_READOUT_ATTRIBUTION','stage':STAGE,'plan':binding,'claim':claim,'policy_at_claim':pref,'boot_id':boot,'report':report,'source_unchanged':True,'test_data_used':False,'economic_edge_proven':False,'elapsed_seconds':time.monotonic()-began})
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
        print(json.dumps({'status':'PREFLIGHT_PROVIDER_HLC_READOUT_PASS','plan':binding,'policy':pref,'claim_created':False,'checkpoint_deserializations':0,'NPZ_arrays_decoded':0,'models':0,'NN_forwards':0,'CUDA_initialized':False}))
    else:print(json.dumps(run(args.plan_sha256),sort_keys=True))


if __name__=='__main__':main()
