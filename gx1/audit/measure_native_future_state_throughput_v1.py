"""One source-fixed current-function efficiency diagnostic; never a learning warmstart."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

if __name__ == '__main__':
    from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
    install_native_m5_direction_io_firewall()

import numpy as np
from gx1.audit import qualify_native_m5_full_gc_direction_v1 as c

REPO=c.REPO
POLICY=c.POLICY
STAGE='native_future_state_throughput_20261121_002'
PARENT='native_future_state_train_readiness_20261121_001'
GATE='throughput_002'
PLAN=REPO/'configs/research/NATIVE_FUTURE_STATE_THROUGHPUT_20261121_002.json'
ROOT=Path('/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_THROUGHPUT_20261121_002')
INITIAL=Path('/home/andre2/GX1_RUNS/NATIVE_M5_FUTURE_STATE_CUDA_INITIAL_20261121_001')
DONOR=Path('/home/andre2/GX1_RUNS/NATIVE_M5_FUTURE_STATE_CUDA_MECHANICS_20261121_001/continuous')
PRODUCER_COMMIT='c1cd096cfde51719bb1ba5bbd9036864526081df'
GUARD_PATHS=('scripts/gx1_capped_run.sh','gx1/contracts/gx1_capped_execution_v1.py')
SOURCE_PATHS=tuple(dict.fromkeys((*c.FUTURE_CUDA_SOURCE_PATHS,
    'gx1/audit/measure_native_future_state_throughput_v1.py','tests/test_native_future_state_throughput_v1.py',
    'tests/test_gx1_capped_execution.py','tests/test_gx1_capped_run_native_year_contract.py',
    'gx1/audit/native_future_state_training_v1.py','tests/test_native_future_state_training_v1.py')))
FIXED_INPUTS={
    'initial_result':{'path':str(INITIAL/'RESULT.json'),'sha256':'a1b7716abc27047eaf3d940f83474159ce05abf61396e037690c62529720c1e1'},
    'initial_terminal':{'path':str(INITIAL/'TERMINAL.json'),'sha256':'1a6372bb1356a32eebdeb3d1b0956bd40633d66839de30eabe52afef2ec97419'},
    'mechanics_result':{'path':str(DONOR/'RESULT.json'),'sha256':'f5093b4ddc186747885ed155c63a2c44abbeaf4fdbd8cf58af2f96492825b2e8'},
    'mechanics_terminal':{'path':str(DONOR/'TERMINAL.json'),'sha256':'758e475bdfa83f9a9ef0b426c042916db5b2ae4f8139348702395cb36a7a6b52'}}
EXECUTION=c.future_cuda_execution('mechanics')
LIMITS={'model_constructions':1,'optimizer_constructions':1,'checkpoint_loads':1,'model_state_restores':1,
    'model_forwards':60,'backbone_forwards':60,'backward_calls':42,'autograd_calls':0,
    'gradient_clip_calls':29,'optimizer_steps':0,'normalization_fits':0,'prior_fits':0,
    'feature_cache_payload_loads':1,'target_archive_loads':1,'truth_fields_decoded':12,
    'measurement_writes':14,'allocator_boundary_calls':18,'CONTROL_rows_scored':0,'TEST_rows':0}
TRAIN_ORDER=(64,96,128,128,96,64)
EVAL_ORDER=(64,256,256,64)


def need(ok,reason):
    if not bool(ok):raise ValueError('FUTURE_THROUGHPUT_'+reason)


def read(binding):
    return c.future_cpu_read(binding)


def source_inventory(owners):
    need(type(owners)is dict and set(owners)==set(SOURCE_PATHS),'SOURCE_SET')
    for name,digest in owners.items():
        need(type(name)is str,'SOURCE_SCHEMA');path=Path(name)
        need(type(name)is str and not path.is_absolute()and '..'not in path.parts and str(path)==name
            and path.suffix in('.py','.sh')and type(digest)is str and len(digest)==64
            and all(x in '0123456789abcdef'for x in digest),'SOURCE_SCHEMA')
        actual=REPO/path
        need(actual.is_file()and not actual.is_symlink()and actual.resolve()==actual,'SOURCE_FILE')


def source_bridge(owners):
    from gx1.contracts.gx1_capped_execution_v1 import native_future_state_guard_source_bridge
    c.future_cuda_source_inventory(owners)
    need(len(owners)==58,'OLD58_INVENTORY')
    for name,digest in owners.items():
        if name not in GUARD_PATHS:need(c.sha(REPO/name)==digest,'UNCHANGED_OLD_OWNER:'+name)
    bridge=native_future_state_guard_source_bridge(owners,producer_commit=PRODUCER_COMMIT)
    return {'unchanged_owners':56,'guard_bridge':bridge,'producer_commit':PRODUCER_COMMIT}


def metadata(inputs):
    need(type(inputs)is dict and set(inputs)==set(FIXED_INPUTS)|{'runtime_proof'}
        and all(inputs[k]==v for k,v in FIXED_INPUTS.items()),'INPUT_ROLES')
    initial,_=c.future_cuda_complete(inputs['initial_result'],inputs['initial_terminal'],
        'COMPLETE_NATIVE_FUTURE_STATE_CUDA_INITIAL')
    donor,_=c.future_cuda_complete(inputs['mechanics_result'],inputs['mechanics_terminal'],
        'COMPLETE_NATIVE_FUTURE_STATE_CUDA_MECHANICS')
    ip=read(initial['plan']);dp=read(donor['plan'])
    # The authenticated old inventory is checked before hashing its named paths.
    c.future_cuda_source_inventory(ip['source_owners']);c.future_cuda_source_inventory(dp['source_owners'])
    need(ip['source_owners']==dp['source_owners']and ip['source_base_commit']==dp['source_base_commit']==PRODUCER_COMMIT,
        'DONOR_INITIAL_SOURCE')
    bridge=source_bridge(ip['source_owners'])
    a,b=initial['report'],donor['report']
    need(ip['backbone_gradient_policy']is dp['backbone_gradient_policy']is None
        and ip['metadata']==dp['metadata']and a['whole_TRAIN_initialbaseline_qualified']is True
        and a['initial']['rows']==354550 and a['initial']['initial_prior_exact']is True
        and a['counts']==c.FUTURE_CUDA_LIMITS['initial']and b['counts']==c.FUTURE_CUDA_LIMITS['continuous']
        and a['initial_state_sha256']==b['initial_state_sha256'],'QUALIFIED_CURRENT_FUNCTION')
    receipt=read(b['checkpoint3_receipt'])
    need(receipt['plan']==donor['plan']and receipt['cursor']==3
        and receipt['checkpoint']['path']==str(DONOR/'CURSOR003.pt')
        and set(receipt['checkpoint'])=={'path','sha256'},'DONOR_CURSOR3_METADATA')
    need(Path(receipt['checkpoint']['path']).suffix=='.pt'and len(receipt['checkpoint']['sha256'])==64
        and all(x in '0123456789abcdef'for x in receipt['checkpoint']['sha256']),'DONOR_CHECKPOINT_REFERENCE')
    for key in ('source_snapshot','normalization','input_binding_ref','prior','dataset','native_manifest'):
        read(ip['metadata'][key])
    runtime=c.future_cuda_require_runtime(read(inputs['runtime_proof']))
    old=ip['metadata']['runtime']
    need(all(runtime[k]==old[k]for k in old if k not in('source_commit','observed_utc')),'SAME_BACKEND_BOOT')
    return {'function':ip['metadata'],'runtime':runtime,'donor_plan':donor['plan'],
        'donor_receipt':b['checkpoint3_receipt'],'donor_checkpoint':receipt['checkpoint'],
        'donor_state_proof':receipt['state_proof'],'source_bridge':bridge,
        'whole_TRAIN_INITIAL':inputs['initial_result']}


def build_plan(*,inputs,source_base_commit=None):
    source_inventory(dict.fromkeys(SOURCE_PATHS,'0'*64))
    meta=metadata(inputs);head=source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    need(head==meta['runtime']['source_commit'],'RUNTIME_CURRENT_COMMIT')
    recipe={**read(meta['donor_plan'])['recipe'],'logical_updates':0,'checkpoint_cursor':None}
    return {'schema_version':'gx1_native_future_state_throughput_plan_v1','stage':STAGE,
        'status':'PREPARED_NOT_ADMITTED','output_root':str(ROOT),'source_base_commit':head,
        'source_owners':{name:c.sha(REPO/name)for name in SOURCE_PATHS},'inputs':inputs,'metadata':meta,
        'execution':EXECUTION,'limits':LIMITS,'recipe':recipe,
        'TRAIN_rows':384,'TRAIN_batch_candidates':[64,96,128],'EVAL_batch_candidates':[64,256],
        'TRAIN_timing_order':list(TRAIN_ORDER),'EVAL_timing_order':list(EVAL_ORDER),
        'warmup_batches_per_configuration':1,'timed_passes_per_configuration':2,
        'allocator_policy':'allocator_cold_at_measurement_phase_boundary',
        'allocator_boundary_calls':18,'timing_excludes_allocator_cleanup':True,
        'parity_tolerance':c.THROUGHPUT_EVAL_TOLERANCE,
        'selection':'maximum_total_completed_rows_per_total_synchronized_seconds_under_limits_only',
        'analysis_weights':'authenticated_own_cursor3_not_fresh_INITIAL_or_learning_warmstart',
        'training_authorized':False,'test_data_used':False,'production_recipe_selected':False}


def gate_expected(plan_ref,*,admitted):
    return {'status':'ADMITTED_ONE_USE'if admitted else'PREPARED_NOT_ADMITTED','plan':plan_ref,
        'output_root':str(ROOT),'execution':EXECUTION,'limits':LIMITS,'execution_authorized':admitted,
        'max_invocations':1,'claimed_invocations':0,'completed_invocations':0,
        'remaining_invocations':1 if admitted else 0,'relaunch_allowed':False,
        'training_authorized':False,'test_authorized':False}


def inspect(plan_sha,*,admitted):
    need(Path.cwd().resolve()==REPO and not ROOT.exists()and not ROOT.is_symlink(),'CWD_FRESH_OUTPUT')
    pr={'path':str(PLAN),'sha256':plan_sha};plan=read(pr);source_inventory(plan['source_owners'])
    need(plan==build_plan(inputs=plan['inputs'],source_base_commit=plan['source_base_commit']),'EXACT_PLAN_SOURCE')
    need(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_base_commit']
        and subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current','HEAD_BRANCH')
    policy_ref=c.ref(POLICY);policy=read(policy_ref);parent=policy[PARENT]
    need(parent.get('goal_status')=='ACTIVE'and parent.get('large_training_qualified')is False,'ACTIVE_READINESS')
    need(json.dumps(parent[GATE],sort_keys=True)==json.dumps(gate_expected(pr,admitted=admitted),sort_keys=True),
        'EXACT_ONEUSE_GATE')
    need(all(policy.get(k)is False for k in ('training_enabled','native_training_enabled','native_launch_authorized',
        'full_epoch_training_allowed','full_val_allowed')),'ALL_TRAIN_FLAGS_CLOSED')
    live=c.collect_future_cuda_runtime_metadata();c.future_cuda_runtime_matches(plan['metadata']['runtime'],live)
    return plan,pr,policy_ref,live


def selected_rows(context):
    rows=context['rows'][:384]
    need(len(rows)==384 and np.all(np.diff(rows)>0),'FIRST384')
    return rows


def packet(context,rows):
    import torch
    need(rows.ndim==1 and 0<len(rows)<=256,'PACKET_MAX256')
    at=np.searchsorted(context['rows'],rows)
    need(np.array_equal(context['rows'][at],rows),'SOURCE_TRUTH_JOIN')
    raw=context['provider'].gather(context['provider'].source_row[rows],max_rows=256)
    return {'features':{k:torch.from_numpy(v)for k,v in raw.items()},
        'truth':{k:v[at]for k,v in context['truth'].items()}}


def weighted_numerator(loss,global_known):
    need(type(global_known)is int and global_known>0,'GLOBAL_KNOWN_SUPPORT')
    return (loss['loss_sums']['activity'].sum()+loss['loss_sums']['side'].sum())/global_known


def select_fastest(records,mode):
    grouped={}
    for r in records:
        if r['mode']==mode and r['completed_under_limits']:
            grouped.setdefault(r['batch_rows'],[]).append(r)
    candidates={b:sum(x['rows']for x in runs)/sum(x['wall_seconds']for x in runs)
        for b,runs in grouped.items()if len(runs)==2}
    need(candidates,'NO_COMPLETED_CANDIDATE')
    winner=min(candidates,key=lambda b:(-candidates[b],b))
    return {'batch_rows':winner,'observed_rows_per_second':candidates[winner],
        'all_completed_rows_per_second':candidates,'criterion':'throughput_only_no_loss_accuracy_or_ML_selection'}



def configuration_reports(records,artifacts,mode,parity_refs):
    batches=(64,96,128)if mode=='train'else(64,256)
    output={}
    for batch in batches:
        runs=[r for r in records if r['mode']==mode and r['batch_rows']==batch]
        need(len(runs)==2 and all(r['completed_under_limits']and r['rows']==384 for r in runs),'TWO_FIXED_PASSES')
        seconds=sum(r['wall_seconds']for r in runs)
        output[str(batch)]={'qualified':True,'measured_rows':768,'elapsed_seconds':seconds,
            'rows_per_second':768/seconds,'peak_reserved_bytes':max(r['cuda_peak_reserved_bytes_so_far']for r in runs),
            'peak_reserved_bytes_scope':'process_high_water_not_candidate_specific',
            'per_candidate_memory_peak_measured':False,
            'measurement_refs':[v for v in artifacts if Path(v['path']).name.startswith(mode.upper()+'_B'+str(batch)+'_PASS')],
            'parity_ref':parity_refs[batch]}
    return output


def require_throughput_result(result_ref,terminal_ref):
    result,terminal=c.future_cuda_complete(result_ref,terminal_ref,'COMPLETE_NATIVE_FUTURE_STATE_THROUGHPUT')
    plan=read(result['plan']);report=result['report']
    need(plan['stage']==result['stage']==terminal['stage']==STAGE and plan['limits']==LIMITS
        and report['counts']==LIMITS and report['optimizer_steps']==0
        and report['whole_TRAIN_INITIAL_replaced']is False and report['production_recipe_selected']is False
        and report['allocator_policy']=='allocator_cold_at_measurement_phase_boundary'
        and report['timing_excludes_allocator_cleanup']is True and len(report['allocator_boundaries'])==18,
        'COMPLETE_ZERO_UPDATE_THROUGHPUT')
    for mode,batches in (('train',(64,96,128)),('eval',(64,256))):
        configs=report[mode];need(type(configs)is dict and set(configs)==set(map(str,batches)),'COMPLETE_CANDIDATES')
        for key,value in configs.items():
            need(value['qualified']is True and value['measured_rows']==768 and value['elapsed_seconds']>0
                and np.isfinite(value['elapsed_seconds'])and np.isfinite(value['rows_per_second'])
                and value['rows_per_second']==768/value['elapsed_seconds']
                and value['peak_reserved_bytes']<=12*1024**3 and len(value['measurement_refs'])==2,'QUALIFIED_CANDIDATE')
            measured=[read(binding)for binding in value['measurement_refs']]
            need(all(v['mode']==mode and v['batch_rows']==int(key)and v['rows']==384
                and v['completed_under_limits']is True for v in measured)
                and sum(v['wall_seconds']for v in measured)==value['elapsed_seconds'],'MEASUREMENT_ROLES')
            proof=read(value['parity_ref'])
            if mode=='train':
                need(proof['batch_rows']==int(key)and proof['rows']==384
                    and proof['gradients']['all_qualified']is True
                    and proof['gradients']['nonzero_backbone_coordinates']>0
                    and all(v['qualified']is True for v in proof['arrays'].values()),'TRAIN_PARITY_PROOF')
            else:need(all(v['qualified']is True for v in proof['numerical_parity'].values()),'EVAL_PARITY_PROOF')
        winner=min(batches,key=lambda b:(-configs[str(b)]['rows_per_second'],b))
        need(report['selected_'+mode+'_batch_rows']==winner,'THROUGHPUT_ONLY_WINNER')
    need(report['selection_criterion']=='max_observed_rows_per_second_qualified_under_limits_only','SELECTION_CRITERION')
    return {'train_batch_rows':report['selected_train_batch_rows'],'eval_batch_rows':report['selected_eval_batch_rows'],
        'result':result_ref,'terminal':terminal_ref,'report':report,'plan':result['plan'],
        'function_metadata':plan['metadata']['function'],'source_bridge':plan['metadata']['source_bridge'],
        'runtime':plan['metadata']['runtime']}


def _cohort(model,context,rows,batch_rows,mode,counts,rng,*,phase_sample=False):
    import torch
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_future_state_loss
    need(mode in('eval','parity','train')and batch_rows in(64,96,128,256),'MEASUREMENT_GEOMETRY')
    need(mode=='eval'or batch_rows!=256,'NO_B256_TRAIN')
    c.restore_training_rng(rng);model.zero_grad(set_to_none=True);model.train(mode=='train')
    positions=np.searchsorted(context['rows'],rows);global_known=int(context['truth']['activity_known'][positions].sum())
    need(global_known>0,'OBSERVED_SUPPORT')
    outputs={};hidden=[];support=None;phases={};events=[];device=next(model.parameters()).device
    capture=mode!='train'
    def hook(module,args,out):
        if capture:hidden.append(out['entry_q_timeframe_hidden'].detach())
    h=model.backbone.register_forward_hook(hook)
    def phase(name,fn):
        if not phase_sample:return fn()
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();began=time.perf_counter();value=fn();end.record()
        phases[name]={'cpu_submit_seconds':time.perf_counter()-began};events.append((name,start,end));return value
    torch.cuda.synchronize();began=time.perf_counter()
    try:
        with torch.set_grad_enabled(mode!='eval'):
            for chunk in c.all_score_batches(rows,batch_rows):
                if mode=='train':model.zero_grad(set_to_none=True)
                p=phase('gather',lambda:packet(context,chunk))
                features=phase('host_to_device',lambda:{k:v.to(device)for k,v in p['features'].items()})
                output,_,activation=phase('forward',lambda:c.future_cuda_forward(model,features,equal_tf=mode=='eval'))
                if capture:
                    for k,v in output.items():outputs.setdefault(k,[]).append(v.detach())
                if mode!='eval':
                    loss=phase('loss',lambda:native_future_state_loss(model_output=output,truth=p['truth'],
                        fold_upper_ns=c.FUTURE_CPU_TRAIN_UPPER,target_contract=context['target_contract']))
                    need(not loss['skip_update']and bool(torch.isfinite(loss['loss'])),'FINITE_SUPPORTED_LOSS')
                    values={k:v.detach().cpu().numpy()for k,v in loss['counts'].items()}
                    if support is None:support={k:np.zeros_like(v)for k,v in values.items()}
                    for k,v in values.items():support[k]+=v
                    term=weighted_numerator(loss,global_known)if mode=='parity'else loss['loss']
                    phase('backward',lambda:term.backward());counts['backward_calls']+=1
                    grads=[p.grad for p in model.parameters()if p.grad is not None]
                    phase('finite_gradients',lambda:need(grads and c.finite_all_per_device(grads),'FINITE_GRADIENTS'))
                    if mode=='train':
                        phase('clip_grad_norm',lambda:torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters()if p.grad is not None],1.,error_if_nonfinite=True))
                        counts['gradient_clip_calls']+=1
                phase('finite_parameters',lambda:need(c.finite_all_per_device(model.parameters()),'FINITE_PARAMETERS'))
        torch.cuda.synchronize();elapsed=time.perf_counter()-began
        need(elapsed>0,'POSITIVE_MEASUREMENT_TIME')
        for name,a,b in events:phases[name]['cuda_stream_elapsed_ms']=a.elapsed_time(b)
        arrays={k:torch.cat(v).cpu().numpy()for k,v in outputs.items()}
        if capture:arrays['timeframe_hidden']=torch.cat(hidden).cpu().numpy()
        gradients={n:None if p.grad is None else p.grad.detach().cpu().numpy().copy()
            for n,p in model.named_parameters()}if mode=='parity'else None
        peak=torch.cuda.max_memory_reserved();need(peak<=12*1024**3,'CUDA_RESERVED_LIMIT')
        return {'mode':mode,'batch_rows':batch_rows,'rows':len(rows),'wall_seconds':elapsed,
            'completed_under_limits':True,'cuda_peak_reserved_bytes_so_far':peak,
            'timing_scope':'gather_H2D_forward_finite_parameters; plus_loss_backward_clip_in_train; no_optimizer_step',
            'activation':activation,'support':None if support is None else{k:v.tolist()for k,v in support.items()},
            'global_known_activity_cells':global_known,'phases':phases},arrays,gradients
    finally:
        h.remove();model.zero_grad(set_to_none=True);model.train(True);c.restore_training_rng(rng)


def compare_arrays(reference,candidate):
    need(set(reference)==set(candidate),'PARITY_ARRAY_KEYS')
    values={k:c.future_cuda_array_parity(v,candidate[k])for k,v in reference.items()}
    need(all(v['qualified']for v in values.values()),'BATCH_ARRAY_PARITY')
    return values


def compare_gradients(reference,candidate):
    need(set(reference)==set(candidate),'PARITY_PARAMETER_NAMES')
    values={};nonzero=0
    for name,value in reference.items():
        other=candidate[name]
        if value is None or other is None:need(value is other,'PARITY_UNUSED_PARAMETER');continue
        values[name]=c.future_cuda_array_parity(value,other)
        if name.startswith('backbone.'):nonzero+=int(np.count_nonzero(value))
    need(nonzero>0 and values and all(v['qualified']for v in values.values()),'NONVACUOUS_GRADIENT_PARITY')
    return {'parameters_compared':len(values),'nonzero_backbone_coordinates':nonzero,
        'all_qualified':True,'all_bitexact':all(v['bitexact']for v in values.values()),
        'maximum_absolute_difference':max(v['max_absolute_difference']for v in values.values())}


def publish(name,value):
    import re
    need(name in('CLAIM.json','RESULT.json','TERMINAL.json','PHASE_SAMPLE_TRAIN_B64.json')
        or re.fullmatch(r'(?:EVAL|TRAIN)_B(?:64|96|128|256)_PASS00[12]\.json',name)
        or re.fullmatch(r'BATCH_PARITY_B(?:64|96|128)\.json',name),'OUTPUT_NAME')
    return c.publish(ROOT/name,value)


def physical(plan,plan_ref,counts,artifacts,progress):
    import torch
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
    from gx1.audit.native_future_state_training_v1 import release_phase_allocator_cache
    meta=plan['metadata'];donor_plan=read(meta['donor_plan']);receipt=read(meta['donor_receipt'])
    context=c.future_cuda_load(donor_plan,counts);context['target_contract']=meta['function']['target_contract']
    rows=selected_rows(context);runtime=c.future_cuda_seed_and_runtime({'recipe':plan['recipe'],'metadata':{'runtime':meta['runtime']}})
    fn=meta['function'];model=build_native_future_state_model(input_normalization=context['normalization'],
        input_binding=fn['input_binding'],architecture=fn['architecture'],device=torch.device('cuda'),
        train_prior=context['prior'],target_contract=fn['target_contract'],backbone_gradient_policy=None)
    counts['model_constructions']+=1
    options={**plan['recipe']['optimizer'],'betas':tuple(plan['recipe']['optimizer']['betas'])}
    optimizer=torch.optim.AdamW(model.parameters(),**options);counts['optimizer_constructions']+=1
    need(c.ref(meta['donor_checkpoint']['path'])==meta['donor_checkpoint'],'DONOR_PAYLOAD_HASH')
    checkpoint=torch.load(meta['donor_checkpoint']['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']+=1
    need(c.future_cpu_proof(checkpoint)==receipt['state_proof']==meta['donor_state_proof'],'DONOR_STATE_PROOF')
    c.future_cuda_restore(model,optimizer,checkpoint,donor_plan,meta['donor_plan'],context['order']);counts['model_state_restores']+=1
    del checkpoint
    need(any(bool(torch.count_nonzero(head.weight))for head in model.readout.activity_heads.values()),'DONOR_NONZERO_HEADS')
    frozen={'model':c.state_sha(model.state_dict()),'optimizer':c.state_sha(optimizer.state_dict()),
        'normalization':c.state_sha(context['normalization']),'prior':c.state_sha(context['prior'])}
    rng=c.training_rng();rng_sha=c.state_sha(rng);records=[];parity_refs={};eval_parity_refs={};phase_ref=None;allocator_boundaries=[]
    def count(name):
        def hook(*args):
            counts[name]+=1;need(counts[name]<=LIMITS[name],'FORWARD_BUDGET')
        return hook
    hooks=[model.register_forward_pre_hook(count('model_forwards')),model.backbone.register_forward_pre_hook(count('backbone_forwards'))]
    def measure(mode,batch,warm=False,phase=False):
        progress['stage']=mode+'_B'+str(batch)+('_warmup'if warm else '')
        # No _cohort GPU locals survive this boundary. Cleanup is outside all timers.
        label=str(len(allocator_boundaries)+1).zfill(3)+'_'+progress['stage']
        boundary=release_phase_allocator_cache(model,optimizer,label=label)
        need(type(boundary)is dict,'ALLOCATOR_BOUNDARY_RECORD')
        allocator_boundaries.append(boundary);counts['allocator_boundary_calls']+=1
        need(counts['allocator_boundary_calls']<=18,'ALLOCATOR_BOUNDARY_BUDGET')
        record,arrays,gradients=_cohort(model,context,rows[:batch]if warm else rows,batch,mode,counts,rng,phase_sample=phase)
        record['allocator_boundary']=boundary;record['timing_excludes_allocator_cleanup']=True
        return record,arrays,gradients
    def save(name,value):
        binding=publish(name,value);artifacts.append(binding);counts['measurement_writes']+=1;return binding
    try:
        for b in (64,256):measure('eval',b,warm=True)
        eval_reference=None;visits={64:0,256:0}
        for b in EVAL_ORDER:
            record,arrays,_=measure('eval',b)
            if eval_reference is None:eval_reference=arrays
            record['numerical_parity']=compare_arrays(eval_reference,arrays)
            visits[b]+=1;binding=save('EVAL_B'+str(b)+'_PASS00'+str(visits[b])+'.json',record);eval_parity_refs[b]=binding;records.append(record)
        reference_arrays=None;reference_grads=None;reference_support=None
        for b in (64,96,128):
            record,arrays,grads=measure('parity',b)
            if reference_arrays is None:reference_arrays,reference_grads,reference_support=arrays,grads,record['support']
            need(record['support']==reference_support,'EXACT_SUPPORT_ACROSS_GEOMETRY')
            proof={'batch_rows':b,'rows':384,'global_known_activity_cells':record['global_known_activity_cells'],
                'support':record['support'],'arrays':compare_arrays(reference_arrays,arrays),
                'gradients':compare_gradients(reference_grads,grads),'dropout_disabled_by_eval_mode':True,
                'equalTF_disabled_with_gradients':True,'tolerance':plan['parity_tolerance'],
                'allocator_boundary':record['allocator_boundary'],'timing_excludes_allocator_cleanup':True}
            parity_refs[b]=save('BATCH_PARITY_B'+str(b)+'.json',proof)
        del eval_reference,reference_arrays,reference_grads,arrays,grads
        for b in (64,96,128):
            record,_,_=measure('train',b,warm=True,phase=b==64)
            if b==64:
                record['phase_interpretation']='CPU submit wall and CUDA stream interval; not pure GPU-busy or separate CPU-asinh time'
                phase_ref=save('PHASE_SAMPLE_TRAIN_B64.json',record)
        visits={64:0,96:0,128:0}
        for b in TRAIN_ORDER:
            record,_,_=measure('train',b);visits[b]+=1
            record['dropout_trajectory']='train dropout0.05; no cross-batch bitexact trajectory or ML quality claim'
            save('TRAIN_B'+str(b)+'_PASS00'+str(visits[b])+'.json',record);records.append(record)
        need(counts==LIMITS and len(allocator_boundaries)==18,'EXACT_COUNTS')
        need(c.state_sha(model.state_dict())==frozen['model']and c.state_sha(optimizer.state_dict())==frozen['optimizer']
            and c.state_sha(context['normalization'])==frozen['normalization']and c.state_sha(context['prior'])==frozen['prior']
            and c.state_sha(c.training_rng())==rng_sha and all(p.grad is None for p in model.parameters()),'FROZEN_STATE_RNG_INPUTS')
        selected={m:select_fastest(records,m)for m in('train','eval')}
        return {'counts':counts,'measurements':artifacts,'selection':selected,
            'train':configuration_reports(records,artifacts,'train',parity_refs),
            'eval':configuration_reports(records,artifacts,'eval',eval_parity_refs),
            'selected_train_batch_rows':selected['train']['batch_rows'],
            'selected_eval_batch_rows':selected['eval']['batch_rows'],
            'selection_criterion':'max_observed_rows_per_second_qualified_under_limits_only',
            'phase_sample_ref':phase_ref,'rows':384,
            'allocator_policy':'allocator_cold_at_measurement_phase_boundary',
            'allocator_boundaries':allocator_boundaries,'timing_excludes_allocator_cleanup':True,'source_rows_sha256':c.digest(context['provider'].source_row[rows]),
            'issue_ns_sha256':c.digest(context['provider'].issue_ns[rows]),'state_proof':frozen,
            'rng_sha256':rng_sha,'runtime_flags':runtime,'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'peak_reserved_bytes_scope':'process_high_water_not_candidate_specific',
            'per_candidate_memory_peak_measured':False,
            'guard_log_path':os.environ['GX1_TRAINER_GUARD_LOG_PATH'],
            'guard_log_finalized_by_runner_after_child_exit':True,'optimizer_steps':0,
            'main_warmstart_authorized':False,'whole_TRAIN_INITIAL_replaced':False,'production_recipe_selected':False}
    finally:
        for h in hooks:h.remove()


def bookend(plan,pr,policy_ref):
    return all(c.sha(REPO/k)==v for k,v in plan['source_owners'].items())and c.ref(PLAN)==pr and c.ref(POLICY)==policy_ref


def run(plan_sha):
    from gx1.contracts.gx1_capped_execution_v1 import require_native_future_state_throughput_execution
    began=time.monotonic();guard=require_native_future_state_throughput_execution(plan_sha256=plan_sha)
    plan,pr,policy_ref,live=inspect(plan_sha,admitted=True)
    need(guard['plan']==pr and guard['output_root']==str(ROOT)and guard['outer_seconds']==1800
        and guard['internal_seconds']==1770,'EXACT_GUARD')
    need(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'TIMER_ALREADY_ACTIVE')
    ROOT.mkdir(exist_ok=False);claim=publish('CLAIM.json',{'stage':STAGE,'status':'CLAIMED_ONE_USE','plan':pr,
        'policy_at_claim':policy_ref,'guard':guard,'fresh_live_idle_runtime':live,'remaining_invocations':0,
        'relaunch_allowed':False,'training_authorized':False,'test_data_used':False})
    counts=dict.fromkeys(LIMITS,0);artifacts=[];progress={'stage':'before_payloads'};previous=signal.getsignal(signal.SIGALRM)
    def expired(*args):raise TimeoutError('FUTURE_THROUGHPUT_INTERNAL_1770S')
    try:
        remaining=1770-(time.monotonic()-began);need(remaining>0,'TIME_BEFORE_PAYLOADS')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
        report=physical(plan,pr,counts,artifacts,progress)
        for value in plan['inputs'].values():read(value)
        for key in ('source_snapshot','normalization','input_binding_ref','prior','dataset','native_manifest'):read(plan['metadata']['function'][key])
        snapshot=read(plan['metadata']['function']['source_snapshot']);data=read(plan['metadata']['function']['dataset'])
        need(c.ref(snapshot['archive']['path'])==snapshot['archive']and c.ref(data['arrays']['path'])==data['arrays']
            and c.ref(plan['metadata']['donor_checkpoint']['path'])==plan['metadata']['donor_checkpoint'],'PAYLOAD_BOOKENDS')
        need(bookend(plan,pr,policy_ref)and c.common.firewall_state()['denied_attempts']==0,'SOURCE_POLICY_FIREWALL_BOOKEND')
        need(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id']
            and c.physical_windows_boot_id()==live['physical_windows_boot_id'],'BOOT_BOOKEND')
        result=publish('RESULT.json',{'stage':STAGE,'status':'COMPLETE_NATIVE_FUTURE_STATE_THROUGHPUT',
            'plan':pr,'claim':claim,'report':report,'source_unchanged':True,'training_authorized':False,
            'test_data_used':False,'elapsed_seconds':time.monotonic()-began})
        terminal=publish('TERMINAL.json',{'stage':STAGE,'status':'COMPLETE','exit_code':0,'plan':pr,
            'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,
            'source_unchanged':True,'training_authorized':False,'test_data_used':False})
        return {'result':result,'terminal':terminal}
    except BaseException as error:
        try:unchanged=bookend(plan,pr,policy_ref)
        except BaseException:unchanged=False
        publish('TERMINAL.json',{'stage':STAGE,'status':'FAILED_CONSUMED','exit_code':1,'plan':pr,'claim':claim,
            'counts':counts,'completed_measurements':artifacts,'failed_stage':progress['stage'],
            'error':type(error).__name__+':'+str(error)[:1500],'remaining_invocations':0,'relaunch_allowed':False,
            'source_unchanged':unchanged,'training_authorized':False,'test_data_used':False})
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def main():
    import sys
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();need(sys.argv[1:]==['--plan-sha256',args.plan_sha256]+(['--preflight']if args.preflight else []),'EXACT_CLI')
    if args.preflight:
        _,pr,policy_ref,live=inspect(args.plan_sha256,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_CURRENT_FUNCTION_THROUGHPUT_METADATA_ONLY','plan':pr,
            'policy':policy_ref,'claim_created':False,'model_constructions':0,'checkpoint_loads':0,
            'cached_values_decoded':0,'target_values_decoded':0,'cuda_initialized':False,
            'live_idle_observed_utc':live['observed_utc']}))
    else:print(json.dumps(run(args.plan_sha256),sort_keys=True))


if __name__=='__main__':main()
