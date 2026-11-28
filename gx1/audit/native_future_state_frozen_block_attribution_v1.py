"""One frozen2x2 backbone/readout diagnostic; no trained candidate or update."""
from pathlib import Path
import argparse,copy,hashlib,json,signal,subprocess,sys,time
REPO=Path(__file__).resolve().parents[2]
if str(REPO)not in sys.path:sys.path.insert(0,str(REPO))
from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
_FIREWALL=install_native_m5_direction_io_firewall()
from gx1.audit import native_future_state_frozen_final864_mode_gap_v1 as prior
from gx1.audit import native_future_state_fixed_continuation_v1 as fixed
q=prior.q;o=prior.o;core=prior.core;train=prior.train;guard=prior.guard;recovery=prior.recovery;need=prior.need;read=prior.read;ref=prior.ref;E=o.E
NAME='NATIVE_V3_FROZEN_864_1080_BLOCK_ATTRIBUTION_20261128_001';STAGE=NAME.lower();PLAN=REPO/'configs/research'/(NAME+'.json');ROOT=Path('/home/andre2/GX1_RUNS')/NAME
SNAPSHOT=E/'V3_FROZEN_864_1080_BLOCK_ATTRIBUTION_ADMITTED_POLICY_20261128_001.json'
DESIGN={'path':str(E/'V3_FROZEN_864_1080_BLOCK_ATTRIBUTION_DESIGN_20261128_001.json'),'sha256':'9578cee76b7d0676529bf78a48f90c9d8a6f68bd5e13a914ce9c08c47578d2dd'}
ANCHOR='0c06e9b8a4c2e7becff1989a43d754962933d268';SELF='gx1/audit/native_future_state_frozen_block_attribution_v1.py';TEST='tests/test_native_future_state_frozen_block_attribution_v1.py';GUARD_TEST='tests/test_native_frozen_block_attribution_guard_v1.py'
PROFILE='gx1_v3_frozen_block_attribution_trainer20g_v1';STATUS_PREFLIGHT='PREFLIGHT_V3_FROZEN_BLOCK_ATTRIBUTION_METADATA_ONLY'
INVARIANT_KEYS=frozenset({'future_state_model_sha256'})
merge_inputs=prior.merge_inputs


def gate(binding,admitted=False):
    return {'status':'ADMITTED_EXACT_V3_FROZEN_BLOCK_ATTRIBUTION'if admitted else'PREPARED_CLOSED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False,'spending_authorized':False}


def require_gate(value,binding,admitted=False):
    expected=gate(binding,admitted);need(type(value)is dict and type(value.get('remaining_invocations'))is int and all(type(value.get(k))is bool for k in('execution_authorized','relaunch_allowed','training_authorized','test_authorized','spending_authorized'))and all(value.get(k)==v for k,v in expected.items()),'FROZEN_BLOCK_ATTRIBUTION_TYPED_GATE');return value


def snapshot(binding,pref):
    sr=ref(SNAPSHOT);policy=read(sr);need(pref=={'path':str(o.POLICY),'sha256':sr['sha256']}and all(policy.get(k)is False for k in o.DEV_FLAGS),'FROZEN_BLOCK_ATTRIBUTION_SNAPSHOT_ALL6_CLOSED');require_gate(policy.get(STAGE),binding,True);need(policy[STAGE].get('design')==DESIGN,'FROZEN_BLOCK_ATTRIBUTION_SNAPSHOT_DESIGN');return sr


def operation_limits():
    return {'autograd_calls':0,'backbone_forwards':324,'backward_calls':0,'checkpoint_loads':2,'checkpoint_prepublication_loads':0,'checkpoint_readbacks':0,'checkpoint_receipt_writes':0,'checkpoint_writes':0,'diagnostic_skips':0,'feature_cache_payload_loads':1,'logit_archive_readbacks':4,'logit_archive_writes':4,'model_constructions':1,'model_forwards':324,'model_restores':3,'normalization_fits':0,'observer_records':0,'optimizer_constructions':1,'optimizer_steps':0,'prior_fits':0,'readout_forwards':324,'saved_eval_logit_loads':2,'skipped_batches':0,'target_archive_loads':1,'authentic_checkpoint_prechecks':2,'block_composite_copies':2}


def endpoint_metadata(design,cursor):
    key=str(cursor);parent=read(design['producer'+key+'_plan']);result=read(design['producer'+key+'_result']);terminal=read(design['producer'+key+'_terminal']);claim=read(result['claim']);acceptance=read(design['producer'+key+'_acceptance']);receipt=read(design['authentic'+key+'_receipt']);producer=design['producer'+key+'_plan'];status='COMPLETE_V3_OBSERVED_TRAIN_CONTINUATION_002'if cursor==864 else'COMPLETE_V3_FIXED_CONTINUATION'
    need(result['status']==status and result['error']is None and result['plan']==terminal['plan']==claim['plan']==producer and terminal['result']==design['producer'+key+'_result']and terminal['claim']==result['claim']and terminal['status']==acceptance['status']=='COMPLETE'and terminal['exit_code']==0 and result['report']['ending_cursor']==cursor and set(result['bookends'])=={'plan','sources','inputs','policy','boots','HEAD'}and all(v is True for v in result['bookends'].values()),'FROZEN_BLOCK_ATTRIBUTION_CLOSED_PARENT')
    need(acceptance['plan']==producer and acceptance['result']==design['producer'+key+'_result']and acceptance['terminal']==design['producer'+key+'_terminal']and acceptance['claim']==result['claim']and acceptance['all_bookends_pass']is acceptance['hardware_guard_accepted']is acceptance['actual_process_gone']is True and terminal['remaining_invocations']==claim['remaining_invocations']==0 and terminal['relaunch_allowed']is claim['relaunch_allowed']is False,'FROZEN_BLOCK_ATTRIBUTION_ACCEPTED_PARENT');core.future_cpu_parent_stopped(claim['process_identity'])
    bridge={'producer_plan':producer,'authentic_start_FINAL'+key+'_checkpoint':design['authentic'+key+'_checkpoint'],'authentic_start_FINAL'+key+'_receipt':design['authentic'+key+'_receipt']};fixed.validate_start_receipt(parent,receipt,bridge,result['report'],cursor)
    return parent,receipt,result,claim


def final_metadata():
    design=read(DESIGN);need(design['experiment_id']==NAME and design['stage']==STAGE and design['scope']['total_model_forwards']==324 and design['scope']['passes']==4 and design['scope']['optimizer_steps']==design['scope']['backward_calls']==design['scope']['autograd_calls']==0,'FROZEN_BLOCK_ATTRIBUTION_DESIGN_SCOPE');parents={};receipts={};trajectories={};refs={'frozen_block_attribution_design':DESIGN}
    for cursor in(864,1080):
        key=str(cursor);parent,receipt,result,claim=endpoint_metadata(design,cursor);parents[key]=parent;receipts[key]=receipt;trajectories[key]=parent['metadata']['trajectory'];refs['frozen_block_attribution_'+key+'_claim']=result['claim'];refs['frozen_block_attribution_'+key+'_policy_snapshot']=claim['admitted_policy_snapshot']
    closed,producer,original,fr,original_receipt,identity=q.original_metadata();need(closed['plan']==parents['864']['original_producer_plan']==parents['1080']['original_producer_plan']and all(p['model_contract']==original['model_contract']and p['recipe']==original['recipe']and p['physical_input_identity']==identity for p in parents.values()),'FROZEN_BLOCK_ATTRIBUTION_FRESH_PHYSICAL_IDENTITY');refs['frozen_block_attribution_original_plan']=closed['plan']
    cohort=read(original['inputs']['cohort_prior']);factory=read(original['metadata']['prior']);spec=read(original['inputs']['cohort_spec']);refs.update({'frozen_block_attribution_cohort_prior':original['inputs']['cohort_prior'],'frozen_block_attribution_factory_prior':original['metadata']['prior'],'frozen_block_attribution_numeric_owner_spec':original['inputs']['cohort_spec']})
    scores=copy.deepcopy(closed['value']['report']['scores']);scores['FINAL648']=parents['864']['inputs']['observed_continuation_002_BASELINE648_score'];scores['FINAL864']=design['authentic864_score'];histories={'864':copy.deepcopy(scores)};scores['FINAL1080']=design['authentic1080_score'];histories['1080']=copy.deepcopy(scores)
    for cursor in(864,1080):
        key=str(cursor);numeric=read(design['authentic'+key+'_numeric_owner_score']);wrapper=read(design['authentic'+key+'_score']);review=numeric['review'];need(wrapper['actual_phase']=='FINAL'+key and wrapper['global_cursor']==cursor and wrapper['numeric_owner_score']==design['authentic'+key+'_numeric_owner_score']and wrapper['numeric_owner_logits']==numeric['logits']==design['authentic'+key+'_logits']and numeric['phase']=='PASS2'and numeric['rows']==20727 and numeric['batches']==81 and review['COHORT_score_prior_sha256']==cohort['contract_sha256']and review['factory_prior_sha256']==factory['contract_sha256']and review['spec_sha256']==spec['contract_sha256']and spec['cohort_prior']==cohort and spec['whole_TRAIN_spec']['prior_contract_sha256']==factory['contract_sha256'],'FROZEN_BLOCK_ATTRIBUTION_SAVED_SCORE_CONTRACTS');need(receipts[key]['proof']['scores']==core.state_sha(histories[key]),'FROZEN_BLOCK_ATTRIBUTION_FULL_SCORE_HISTORY');core.future_cuda_runtime_matches(parents[key]['metadata']['runtime'],receipts[key]['producer_runtime'])
    refs.update({'frozen_block_attribution_'+k:v for k,v in design.items()if type(v)is dict and set(v)=={'path','sha256'}});return original,receipts,trajectories,parents,histories,refs,identity


metadata=final_metadata


def build_plan(runtime,source_commit=None):
    original,receipts,trajectories,parents,scores,extra,identity=final_metadata();design=read(DESIGN);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();need(source_commit in(None,head)and runtime['source_commit']==head,'FROZEN_BLOCK_ATTRIBUTION_FROZEN_SOURCE_RUNTIME');core.future_cuda_require_runtime(runtime)
    for trajectory in trajectories.values():train.runtime_compatible(trajectory['runtime_compatibility'],runtime)
    source=read(design['source_reference_plan']);owners={}
    for name,digest in source['source_owners'].items():
        current=ref(REPO/name)['sha256'];need(name in o.GUARDS or current==digest,'FROZEN_BLOCK_ATTRIBUTION_UNCHANGED_SOURCE:'+name);owners[name]=current
    owners.update({n:ref(REPO/n)['sha256']for n in(SELF,TEST,GUARD_TEST)});old={n:hashlib.sha256(subprocess.check_output(['git','show',ANCHOR+':'+n],cwd=REPO)).hexdigest()for n in o.GUARDS};bridge=guard.native_v3_frozen_block_attribution_guard_source_bridge(old,producer_commit=ANCHOR);inputs=merge_inputs(parents['1080']['inputs'],extra);roles=sorted(k for k,v in inputs.items()if Path(v['path']).suffix not in('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl','.pickle','.joblib'));producers={k:design['producer'+k+'_plan']for k in receipts}
    return {'schema_version':'gx1_native_v3_frozen_block_attribution_plan_v1','stage':STAGE,'source_commit':head,'output_root':str(ROOT),'source_owners':owners,'source_reference_plan':design['source_reference_plan'],'guard_source_bridge':bridge,'design':DESIGN,'producer_plans':producers,'original_producer_plan':extra['frozen_block_attribution_original_plan'],'authentic_receipts':{k:design['authentic'+k+'_receipt']for k in receipts},'authentic_checkpoints':{k:v['checkpoint']for k,v in receipts.items()},'metadata':{'original_plan':extra['frozen_block_attribution_original_plan'],'endpoints':{k:{'parent_plan':producers[k],'receipt':receipts[k],'scores':scores[k]}for k in receipts},'runtime':copy.deepcopy(runtime)},'recipe':parents['1080']['recipe'],'model_contract':original['model_contract'],'physical_input_identity':identity,'backbone_gradient_policy':None,'numeric_owner_phase':'PASS2','arms':design['arms_in_fixed_order'],'limits':operation_limits(),'inputs':inputs,'metadata_input_roles':roles,'payload_stat':{k:{'size':Path(v['path']).stat().st_size,'mtime_ns':Path(v['path']).stat().st_mtime_ns}for k,v in inputs.items()if k not in roles},'execution':guard._native_v3_development_execution('warm'),'execution_profile':PROFILE,'scope':design['scope']}


def partition_states(first,second):
    import torch
    need(set(first)==set(second)and first,'FROZEN_BLOCK_ATTRIBUTION_STATE_INVENTORY');groups={'backbone':[],'readout':[],'invariants':[]}
    for name,a in first.items():
        b=second[name];need(isinstance(a,torch.Tensor)and isinstance(b,torch.Tensor)and a.shape==b.shape and a.dtype==b.dtype and bool(torch.isfinite(a).all())and bool(torch.isfinite(b).all()),'FROZEN_BLOCK_ATTRIBUTION_STATE_HEADER')
        if name.startswith('backbone.'):groups['backbone'].append(name)
        elif name.startswith('readout.'):groups['readout'].append(name)
        else:need(name in INVARIANT_KEYS and torch.equal(a,b),'FROZEN_BLOCK_ATTRIBUTION_UNOWNED_OR_CHANGED_INVARIANT');groups['invariants'].append(name)
    need(groups['backbone']and groups['readout']and set(groups['invariants'])==INVARIANT_KEYS,'FROZEN_BLOCK_ATTRIBUTION_COMPLETE_PARTITION');return {k:sorted(v)for k,v in groups.items()}


def composite_state(states,partition,backbone_cursor,readout_cursor):
    b=states[str(backbone_cursor)];r=states[str(readout_cursor)];result={k:b[k]for k in partition['backbone']+partition['invariants']};result.update({k:r[k]for k in partition['readout']});need(set(result)==set(b)==set(r),'FROZEN_BLOCK_ATTRIBUTION_EXHAUSTIVE_COMPOSITE');return result


def require_arm_invariants(before,after,expected):
    checks={'before_expected_full10':before['proof']==expected,'after_expected_full10':after['proof']==expected,'named_modes_unchanged':before['modes']==after['modes'],'grad_none':before['grad_none']and after['grad_none'],'nonpersistent_buffers_unchanged':before['nonpersistent_buffers_sha256']==after['nonpersistent_buffers_sha256']};need(all(checks.values()),'FROZEN_BLOCK_ATTRIBUTION_PER_PASS_MUTATION_BEFORE_NEXT_COPY');return checks



def require_self_parity(outputs,saved,current,saved_numeric):
    need(set(outputs)==set(saved)=={'activity_logits','side_logits'}and all(outputs[k].dtype==saved[k].dtype==o.np.float32 and outputs[k].shape==saved[k].shape and core.digest(outputs[k])==core.digest(saved[k])for k in saved)and current['review']==saved_numeric['review']and current['batch_score_hashes']==saved_numeric['batch_score_hashes'],'FROZEN_BLOCK_ATTRIBUTION_EXACT_SELF_F32_AND_NUMERIC_REVIEW_PARITY');return True


def block_comparisons(reviews):
    names=('SELF864','SELF1080','BACKBONE864_READOUT1080','BACKBONE1080_READOUT864');need(set(reviews)==set(names),'FROZEN_BLOCK_ATTRIBUTION_ALL_FOUR_REVIEWS');base=reviews['SELF864']
    for review in reviews.values():need(all(review[k]==base[k]for k in('truth_sha256','support','selection_ref','COHORT_score_prior_sha256','factory_prior_sha256','spec_sha256')),'FROZEN_BLOCK_ATTRIBUTION_SAME_SCORE_CONTRACTS')
    pairs={'backbone_given_readout864':('SELF864','BACKBONE1080_READOUT864'),'backbone_given_readout1080':('BACKBONE864_READOUT1080','SELF1080'),'readout_given_backbone864':('SELF864','BACKBONE864_READOUT1080'),'readout_given_backbone1080':('BACKBONE1080_READOUT864','SELF1080')};contrasts={}
    for label,(a,b)in pairs.items():
        value=fixed.compare_reviews(reviews[a],reviews[b],a,b);value['actual_after']=value.pop('actual_candidate');value.update(diagnostic_only=True,new_predictor=False,no_new_trained_candidate=True);contrasts[label]=value
    ka=pairs['backbone_given_readout864'][1]+'_minus_'+pairs['backbone_given_readout864'][0];kb=pairs['backbone_given_readout1080'][1]+'_minus_'+pairs['backbone_given_readout1080'][0]
    def interaction(a,b):
        if type(a)is dict:
            if ka in a:
                x,y=a[ka],b[kb];return {'backbone_effect_with_readout864':x,'backbone_effect_with_readout1080':y,'interaction':None if x is None or y is None else y-x,'status':'UNKNOWN'if x is None or y is None else'DESCRIPTIVE'}
            need(set(a)==set(b),'FROZEN_BLOCK_ATTRIBUTION_INTERACTION_METRIC_PATHS');return {k:interaction(v,b[k])for k,v in a.items()}
        need(a==b,'FROZEN_BLOCK_ATTRIBUTION_SAME_SUPPORT_COUNTS');return a
    interactions={key:interaction(contrasts['backbone_given_readout864'][key],contrasts['backbone_given_readout1080'][key])for key in('all_TF','by_UTC_target_year')}
    return {'conditional_block_effects':contrasts,'interaction':interactions,'interaction_formula':'SELF1080 - BACKBONE1080_READOUT864 - BACKBONE864_READOUT1080 + SELF864','equal_target_AUC':'NOT_IMPLEMENTED','diagnostic_only':True,'new_predictor':False,'no_new_trained_candidate':True,'interpretation':'Functional dependence of co-adapted blocks; no optimizer-history, individual-feature causation, patch, selection or edge claim.','later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'}


def physical(plan,binding,root,live):
    import torch
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
    producer=read(plan['metadata']['original_plan']);original=producer['metadata']['original'];counts=dict.fromkeys(plan['limits'],0);context=o.development_nn_context(original);counts['feature_cache_payload_loads']=counts['target_archive_loads']=1;need(o.issue_mix_actual_stream(context,producer)==read(plan['inputs']['observed_continuation_producer_stream_binding']),'FROZEN_BLOCK_ATTRIBUTION_ORIGINAL_SOURCE_TRUTH_STREAM');core.future_cuda_seed_and_runtime({'recipe':original['recipe'],'metadata':{'runtime':live}})
    model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=original['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1;need(model.research_objective==original['model_contract']and core.state_sha(model.state_dict())==original['initial_reference']['model_state_sha256']and core.state_sha(core.training_rng())==original['initial_reference']['rng_sha256'],'FROZEN_BLOCK_ATTRIBUTION_FACTORY_FUNCTION_RNG');optimizer=torch.optim.AdamW(model.parameters(),**original['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'FROZEN_BLOCK_ATTRIBUTION_FRESH_OPTIMIZER');checkpoints={};parents={}
    for key,endpoint in plan['metadata']['endpoints'].items():
        receipt=endpoint['receipt'];parent=read(endpoint['parent_plan']);cpref=plan['authentic_checkpoints'][key];need(cpref==receipt['checkpoint']and ref(cpref['path'])==cpref,'FROZEN_BLOCK_ATTRIBUTION_GENUINE_CP_SHA');cp=torch.load(cpref['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']+=1
        need(cp['schema_version']=='gx1_native_future_state_training_checkpoint_v1'and cp['producer']=={'original_plan':endpoint['parent_plan'],'lane':'continuation','runtime':receipt['producer_runtime']}and cp['trajectory']==parent['metadata']['trajectory']and cp['position']==train.position(parent['schedule'],int(key))and len(cp['steps'])==int(key)and cp['scores']==endpoint['scores']and train.checkpoint_proof(cp)==receipt['proof'],'FROZEN_BLOCK_ATTRIBUTION_BOTH_FULL10_BEFORE_ANY_COPY');counts['authentic_checkpoint_prechecks']+=1;checkpoints[key]=cp;parents[key]=parent
    need(counts['authentic_checkpoint_prechecks']==counts['checkpoint_loads']==2,'FROZEN_BLOCK_ATTRIBUTION_BOTH_AUTHENTIC_BEFORE_COPY');states={k:cp['model_state']for k,cp in checkpoints.items()};partition=partition_states(states['864'],states['1080']);need(set(states['864'])==set(model.state_dict()),'FROZEN_BLOCK_ATTRIBUTION_ACTUAL_REGISTERED_INVENTORY');persistent_keys=set(states['864']);nonpersistent={n:b for n,b in model.named_buffers()if n not in persistent_keys};need(all(n.startswith(('backbone.','readout.'))for n in nonpersistent),'FROZEN_BLOCK_ATTRIBUTION_NONPERSISTENT_BUFFER_OWNER');nonpersistent_sha=core.state_sha(nonpersistent);partition_ref=o.development_publish(root,'STATE_PARTITION.json',{'keys':partition,'registered_nonpersistent_buffers':sorted(nonpersistent),'nonpersistent_constructor_sha256':nonpersistent_sha,'nonpersistent_buffers_copied':False,'state_key_count':len(states['864']),'all_registered_parameters_and_persistent_buffers':True,'invariants_exactly_identical':True});saved={k:o.train_eval_gap_saved_outputs(plan['inputs']['frozen_block_attribution_authentic'+k+'_logits'],context['truth']['issue_ns'],counts)for k in checkpoints};reports=[];reviews={};invariants=[];active='1080';steps=[];scores={};final_proof=None;final_restored=False
    def proof():return train.checkpoint_proof(train.make_checkpoint(model,optimizer,trajectory=parents[active]['metadata']['trajectory'],original_plan_ref=plan['producer_plans'][active],lane='diagnostic',cursor=int(active),steps=steps,scores=scores,producer_runtime=live))
    def capture_state():return {'proof':proof(),'modes':{n:m.training for n,m in model.named_modules()},'grad_none':all(p.grad is None for p in model.parameters()),'nonpersistent_buffers_sha256':core.state_sha({n:b for n,b in model.named_buffers()if n not in persistent_keys})}
    def restore(key):
        nonlocal active,steps,scores
        endpoint=plan['metadata']['endpoints'][key];train.strict_restore(model,optimizer,checkpoints[key],trajectory=parents[key]['metadata']['trajectory'],original_plan_ref=endpoint['parent_plan'],expected_cursor=int(key),current_runtime=live,expected_state_proof=endpoint['receipt']['proof']);counts['model_restores']+=1;active=key;steps=copy.deepcopy(checkpoints[key]['steps']);scores=copy.deepcopy(checkpoints[key]['scores']);need(proof()==endpoint['receipt']['proof'],'FROZEN_BLOCK_ATTRIBUTION_AUTHENTIC_RESTORE_ALL10')
    def counter(key):
        def hook(*_):counts[key]+=1;need(counts[key]<=plan['limits'][key],'FROZEN_BLOCK_ATTRIBUTION_FORWARD_BOUND')
        return hook
    def eval_mode_guard(*_):need(not torch.is_grad_enabled()and all(not m.training for m in model.modules()),'FROZEN_BLOCK_ATTRIBUTION_ACTUAL_EVAL_NO_GRAD')
    hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),model.readout.register_forward_pre_hook(counter('readout_forwards')),model.register_forward_pre_hook(eval_mode_guard),model.register_forward_hook(eval_mode_guard)]
    try:
        for arm in plan['arms']:
            name=arm['name'];self_arm=arm['backbone_cursor']==arm['readout_cursor'];pieces={}
            if self_arm:restore(str(arm['backbone_cursor']));expected=copy.deepcopy(plan['metadata']['endpoints'][active]['receipt']['proof'])
            else:
                need(len(reports)>=2 and all(r['exact_saved_F32_and_numeric_review_parity']for r in reports[:2]),'FROZEN_BLOCK_ATTRIBUTION_BOTH_SELF_PARITIES_BEFORE_CROSS');composite=composite_state(states,partition,arm['backbone_cursor'],arm['readout_cursor']);expected=copy.deepcopy(plan['metadata']['endpoints']['1080']['receipt']['proof']);expected['model_state']=core.state_sha(composite);model.load_state_dict(composite,strict=True);counts['block_composite_copies']+=1
            before=capture_state();need(before['proof']==expected and before['grad_none']and before['nonpersistent_buffers_sha256']==nonpersistent_sha,'FROZEN_BLOCK_ATTRIBUTION_EXPECTED_COMPOSITE_BEFORE_PASS');namespace=root/name;namespace.mkdir(exist_ok=False)
            def capture(module,args,output):
                for k,v in output.items():pieces.setdefault(k,[]).append(v.detach().cpu().numpy().copy())
            handle=model.register_forward_hook(capture)if self_arm else None
            try:numeric_ref=o.development_score(model,context,original,namespace,'PASS2',counts)
            finally:
                if handle is not None:handle.remove()
                after=capture_state();checks={'label':name,'before':before,'after':after,'expected_full10':expected};invariants.append(checks);checks['checks']=require_arm_invariants(before,after,expected)
            numeric=read(numeric_ref);parity=None
            if self_arm:
                outputs={k:o.np.concatenate(v)for k,v in pieces.items()};original_numeric=read(plan['inputs']['frozen_block_attribution_authentic'+active+'_numeric_owner_score']);parity=require_self_parity(outputs,saved[active],numeric,original_numeric)
            reviews[name]=numeric['review'];wrapper=o.development_publish(root,name+'_SCORE.json',{'arm':arm,'numeric_owner_phase':'PASS2','numeric_owner_score':numeric_ref,'numeric_owner_logits':numeric['logits'],'exact_saved_F32_and_numeric_review_parity':parity,'diagnostic_only':True,'new_predictor':False,'no_new_trained_candidate':True,'state_invariants':checks['checks'],'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'});reports.append({'arm':arm,'score':wrapper,'numeric_owner_score':numeric_ref,'logits':numeric['logits'],'exact_saved_F32_and_numeric_review_parity':parity})
        comparison=o.development_publish(root,'BLOCK_CONTRASTS_AND_INTERACTION.json',block_comparisons(reviews));need(len(reports)==4 and all(all(x['checks'].values())for x in invariants),'FROZEN_BLOCK_ATTRIBUTION_ALL_PASSES_IMMUTABLE_BEFORE_FINAL_RESTORE');restore('1080');final_proof=proof();final_snapshot=capture_state();need(final_snapshot['modes']==invariants[1]['before']['modes']and final_snapshot['nonpersistent_buffers_sha256']==nonpersistent_sha and final_snapshot['grad_none'],'FROZEN_BLOCK_ATTRIBUTION_FINAL_NAMED_MODES_BUFFERS_EMPTY_GRAD');need(final_proof==plan['metadata']['endpoints']['1080']['receipt']['proof']and counts==plan['limits']and torch.cuda.max_memory_reserved()<=plan['execution']['cuda_peak_reserved_bytes_max'],'FROZEN_BLOCK_ATTRIBUTION_EXACT_COUNTS_FINAL1080_ALL10_VRAM');final_restored=True
        result_report={'passes':reports,'comparison':comparison,'counts':counts,'state_partition':partition_ref,'state_invariants':{'path':str(root/'STATE_INVARIANTS.json')},'final_state_proof':final_proof,'final_authentic1080_all10_restored':True,'per_pass_mutation_checked_before_next_copy':True,'optimizer_updates':0,'new_checkpoint_writes':0,'diagnostic_only':True,'new_predictor':False,'no_new_trained_candidate':True,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'}
    finally:
        for hook in hooks:hook.remove()
        state_ref=publish_state_invariants(root,counts,plan['limits'],invariants,final_proof,plan['metadata']['endpoints']['1080']['receipt']['proof'],final_restored)
    result_report['state_invariants']=state_ref;return result_report



def publish_state_invariants(root,counts,limits,invariants,final_proof,expected_final,final_restored):
    return o.development_publish(root,'STATE_INVARIANTS.json',{'status':'PASS'if final_restored is True and counts==limits and len(invariants)==4 and all(all(x.get('checks',{}).values())and x.get('checks')for x in invariants)else'FAILED_CONSUMED','counts':dict(counts),'passes':invariants,'final_full10_proof':final_proof,'final_authentic1080_all10_restored':final_proof==expected_final,'final_restore_all_checks_pass':final_restored,'no_restore_after_failed_pass':True})


def payload_allowlist(plan,root):
    return {str(Path(v['path']).resolve())for v in plan['inputs'].values()if Path(v['path']).suffix in('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl','.pickle','.joblib')}|{str((root/arm['name']/'PASS2_LOGITS.npz').resolve())for arm in plan['arms']}

def inspect(sha,admitted=False):
    bound=guard._native_v3_frozen_block_attribution_bound_plan(sha);binding=bound['plan'];plan=read(binding);need(plan==build_plan(plan['metadata']['runtime'],plan['source_commit']),'FROZEN_BLOCK_ATTRIBUTION_EXACT_PLAN');o.recovery_check_inputs(plan);pref=ref(o.POLICY);policy=read(pref);need(all(policy.get(k)is False for k in o.DEV_FLAGS),'FROZEN_BLOCK_ATTRIBUTION_ALL6_GLOBAL_FLAGS_CLOSED');require_gate(policy.get(STAGE),binding,admitted);live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live)
    if admitted:snapshot(binding,pref)
    return plan,binding,pref,live


def bookends(plan,binding,pref,boots):
    observed=recovery.observe_boots();checks={'plan':lambda:ref(PLAN)==binding,'sources':lambda:all(ref(REPO/n)['sha256']==v for n,v in plan['source_owners'].items()),'inputs':lambda:all(ref(v['path'])==v for v in plan['inputs'].values())and snapshot(binding,pref)['sha256']==pref['sha256'],'policy':lambda:ref(o.POLICY)==pref,'boots':lambda:recovery.boot_matches(boots,observed),'HEAD':lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_commit']};book={};errors={}
    for name,check in checks.items():
        try:book[name]=bool(check())
        except BaseException as exc:book[name]=False;errors[name]={'type':type(exc).__name__,'message':str(exc)[:2400]}
    return book,{'claimed_boots':boots,'boot_observation':observed,'check_errors':errors}


def run(sha):
    execution=guard.require_native_v3_frozen_block_attribution_execution(plan_sha256=sha);plan,binding,pref,live=inspect(sha,True);root=ROOT;root.mkdir(exist_ok=False);boots={k:live[k]for k in('physical_windows_boot_id','wsl_boot_id')};started=time.monotonic();claim=o.development_publish(root,'CLAIM.json',{'plan':binding,'run_id':NAME,'policy_at_claim':pref,'admitted_policy_snapshot':snapshot(binding,pref),'execution':execution,'boots':boots,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'relaunch_allowed':False});report=None;error=None
    def stop(*_):raise TimeoutError('FROZEN_BLOCK_ATTRIBUTION_INTERNAL_WALL_OR_SIGNAL')
    handlers={sig:signal.getsignal(sig)for sig in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    for sig in handlers:signal.signal(sig,stop)
    signal.setitimer(signal.ITIMER_REAL,plan['execution']['internal_seconds'])
    try:
        allowed=payload_allowlist(plan,root);sys.addaudithook(lambda e,a:o.development_io_audit(allowed,e,a));o.recovery_check_inputs(plan,physical=True);report=physical(plan,binding,root,live);need(core.common.firewall_state()['denied_attempts']==0,'FROZEN_BLOCK_ATTRIBUTION_ZERO_FIREWALL_DENIALS')
    except BaseException as exc:error=type(exc).__name__+':'+str(exc)[:2400]
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);book,evidence=bookends(plan,binding,pref,boots);error=error or(None if all(book.values())else'FROZEN_BLOCK_ATTRIBUTION_BOOKEND_FAILED');extra={'state_invariants':ref(root/'STATE_INVARIANTS.json')}if(root/'STATE_INVARIANTS.json').is_file()else{};rr=o.development_publish(root,'RESULT.json',{'status':'COMPLETE_V3_FROZEN_BLOCK_ATTRIBUTION'if error is None else'FAILED_CONSUMED','plan':binding,'run_id':NAME,'claim':claim,'report':report,'error':error,'bookends':book,'bookend_evidence':evidence,'elapsed_seconds':time.monotonic()-started,'diagnostic_only':True,'new_predictor':False,'optimizer_updates':0,'new_checkpoint_writes':0,'test_data_used':False,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN',**extra});tr=o.development_publish(root,'TERMINAL.json',{'status':'COMPLETE'if error is None else'FAILED_CONSUMED','exit_code':0 if error is None else 1,'plan':binding,'claim':claim,'result':rr,'remaining_invocations':0,'relaunch_allowed':False})
        for sig,handler in handlers.items():signal.signal(sig,handler)
    if error:raise RuntimeError(error)
    return {'result':rr,'terminal':tr}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--v3-frozen-block-attribution-plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args();sha=args.v3_frozen_block_attribution_plan_sha256;need(sys.argv[1:]==['--v3-frozen-block-attribution-plan-sha256',sha]+(['--preflight']if args.preflight else[]),'FROZEN_BLOCK_ATTRIBUTION_EXACT_CLI')
    if args.preflight:
        guard.require_native_v3_frozen_block_attribution_execution(plan_sha256=sha);plan,binding,pref,live=inspect(sha,False);import torch;need(not torch.cuda.is_initialized(),'FROZEN_BLOCK_ATTRIBUTION_PREFLIGHT_NO_CUDA');print(json.dumps({'status':STATUS_PREFLIGHT,'plan':binding,'policy':pref,'physical_input_identity':plan['physical_input_identity'],'models':0,'payloads_decoded':0,'CUDA_initialized':False,'CLAIM_created':ROOT.exists()}))
    else:print(json.dumps(run(sha),sort_keys=True))

if __name__=='__main__':main()
