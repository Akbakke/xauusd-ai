"""One-use CPU alignment of saved TRAIN forecasts with their unchanged own target."""
from __future__ import annotations
import argparse, json, signal, subprocess, time
from pathlib import Path
if __name__ == '__main__':
    from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
    install_native_m5_direction_io_firewall()
import numpy as np
import torch
from gx1.audit import audit_native_selective_moves_v1 as saved
from gx1.audit.native_future_state_training_v1 import binary_rank_metrics
from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry

REPO, POLICY = saved.REPO, saved.POLICY
STAGE = 'native_future_state_alignment_20261123_001'
BASENAME = 'NATIVE_FUTURE_STATE_ALIGNMENT_20261123_001'
PLAN, ROOT = REPO/'configs/research'/(BASENAME+'.json'), Path('/home/andre2/GX1_RUNS')/BASENAME
SELF, TEST = 'gx1/audit/diagnose_native_future_state_alignment_v1.py', 'tests/test_diagnose_native_future_state_alignment_v1.py'
DESIGN = {'path': '/home/andre2/GX1_RUNS/GX1_NATIVE_FUTURE_STATE_ALIGNMENT_PREPARATION_20261123_001/DESIGN001.json',
          'sha256': '71c8fa5f195d77f3a6fc958c95e0f06cededaee132d5fbcbc1abe035ecb4b64a'}
COVERAGE = {'path':'/home/andre2/GX1_RUNS/GX1_NATIVE_FUTURE_STATE_ALIGNMENT_PREPARATION_20261123_001/COVERAGE_REQUIREMENTS001.json',
            'sha256':'226d82dd109c297f3af127af3d5afee83900ade6430849349e68cc8a2e26a363'}
CAPTURE_ROOT = Path('/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_20261122_001')
ROWS, UPPER, TF = 354550, 1767225600000000000, ('M5','M15','H1','H4','D1')
STRATA = ('ALL','2021','2022','2023','2024','2025')
TRUTH = ('issue_ns','target_start_ns','validity_ns','target_present','mature','nominal_complete',
         'activity_known','activity','side_known','side_up','state_window_complete','partial_range_active')
HEADS = ('activity_logits','side_logits')
EXECUTION = {'class':'audit','memory':'4G','swap':'512M','numerical_threads':1,'outer_seconds':600,'internal_seconds':570,'kill_after_seconds':10}
ref, read = saved.ref, saved.read


def need(ok, why):
    if not bool(ok): raise ValueError('NATIVE_FUTURE_ALIGNMENT_'+why)


def build_plan(*, source_base_commit=None):
    design = read(DESIGN); population = design['population']; method = design['method']
    need(design['schema_version']=='gx1_native_future_state_alignment_design_v1' and design['execution_authorized'] is False
         and population['issue_rows']==ROWS and population['cutoff_ns']==UPPER and population['strata']==list(STRATA)
         and population['timeframes']==list(TF) and population['sampling'] is False
         and all(method[k] is True for k in ('canonical_truth_masks_and_score_owner_required','no_best_year_TF_threshold_or_checkpoint_selection',
         'no_threshold_added','original_target_contract_unchanged','same_truth_population_for_model_and_prior','unknown_not_FLAT')), 'LOCKED_DESIGN')
    coverage=read(COVERAGE)
    need(coverage['schema_version']=='gx1_native_coverage_requirements_v1' and coverage['alignment_design']==DESIGN
         and coverage['execution_authorized'] is False and coverage['all5_global_training_flags_remain_false'] is True
         and coverage['new_model_or_feature_scan_authorized'] is False and coverage['new_price_decode_authorized'] is False,'COVERAGE_REQUIREMENTS')
    prereg = read(saved.PREREG); main_ref = prereg['model_identity']['original_MAIN_plan']; main = read(main_ref); m = main['metadata']
    result_ref = ref(CAPTURE_ROOT/'RESULT.json'); result = read(result_ref); terminal_ref = ref(CAPTURE_ROOT/'TERMINAL.json'); terminal = read(terminal_ref)
    need(terminal['status']=='COMPLETE' and terminal['exit_code']==0 and terminal['result']==result_ref, 'CAPTURE_COMPLETE')
    capture = result['report']['capture']; capture_plan = read(result['plan'])
    saved.validate_capture_metadata(result,capture,capture_plan,prereg)
    dataset = read(m['dataset']); target = m['target_contract']
    need(dataset['contract']==target and dataset['cutoff_ns']==UPPER and dataset['model_input'] is False
         and dataset['TRAIN_issue_ns_sha256']==m['TRAIN_issue_ns_sha256']==main['schedule']['issue_ns_sha256']
         and m['TRAIN_rows']==ROWS, 'ORIGINAL_TARGET_CLOCK')
    prior = objective.require_native_future_state_train_prior(read(m['prior']),target_contract=target)
    need(prior['fit_population']['issue_rows']==ROWS and prior['fit_population']['issue_ns_sha256']==m['TRAIN_issue_ns_sha256'], 'WHOLE_TRAIN_FROZEN_PRIOR')
    headers = {k:dataset['array_headers'][k] for k in TRUTH}
    need(result['report']['capture_array_headers']['issue_ns']==headers['issue_ns'], 'CAPTURE_TARGET_CLOCK_HEADERS')
    names = set(main['source_owners'])|{SELF,TEST,saved.SELF,saved.TEST}
    need(all(not Path(n).is_absolute() and '..' not in Path(n).parts and Path(n).suffix in ('.py','.sh') for n in names), 'SOURCE_NAMES')
    return {'schema_version':'gx1_native_future_state_alignment_plan_v1','stage':STAGE,'status':'PREPARED_NOT_ADMITTED',
        'plan_path':str(PLAN),'output_root':str(ROOT),'source_base_commit':source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'source_owners':{n:saved.native.sha(REPO/n) for n in sorted(names)},'execution':EXECUTION,
        'inputs':{'design':DESIGN,'coverage_requirements':COVERAGE,'main_goal':design['main_goal'],'original_model_identity':saved.PREREG,'main_plan':main_ref,
            'main_result':prereg['model_identity']['original_MAIN_result'],'capture_result':result_ref,'capture_terminal':terminal_ref,
            'capture_plan':result['plan'],'capture':capture,'dataset':m['dataset'],'target_arrays':dataset['arrays'],'prior':m['prior']},
        'target_contract':target,'TRAIN_issue_ns_sha256':m['TRAIN_issue_ns_sha256'],'target_array_headers':headers,
        'capture_array_headers':result['report']['capture_array_headers'],'question':design['question'],'strata':list(STRATA),
        'runtime_versions':{'numpy':np.__version__,'torch':str(torch.__version__)},
        'budget':{'capture_archive_loads':1,'capture_fields':list(saved.CAPTURE_FIELDS),'target_archive_loads':1,'target_fields':list(TRUTH),
                  'canonical_proper_score_calls':12,'models':0,'checkpoint_loads':0,'forwards':0,'backward':0,'fits':0,'optimizer_updates':0,'source_prices':0,'VAL_TEST_rows':0},
        'score_aggregation':'Each nonempty ALL/year stratum is one CPU float32-logit call per model/prior to the canonical owner; owner computes float64 probability/proper-score sums. Not claimed bitidentical to original E256 sums.'}


def source_clean(plan):
    return (subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current'
        and subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_base_commit']
        and not subprocess.check_output(['git','status','--porcelain','--','gx1','scripts','tests'],cwd=REPO,text=True)
        and all(saved.native.sha(REPO/k)==v for k,v in plan['source_owners'].items()))


def gate_expected(binding, admitted):
    return {'status':'ADMITTED_EXACT_CPU_ONLY_FUTURE_STATE_ALIGNMENT' if admitted else 'PREPARED_NOT_ADMITTED','plan':binding,
        'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}


def inspect(plan_sha, *, admitted):
    need(Path.cwd().resolve()==REPO,'CWD'); binding={'path':str(PLAN),'sha256':plan_sha}; plan=read(binding)
    need(plan==build_plan(source_base_commit=plan['source_base_commit']) and source_clean(plan),'EXACT_PLAN_SOURCE')
    pref=ref(POLICY); policy=read(pref)
    need(all(policy.get(k) is False for k in saved.FLAGS) and policy[STAGE]==gate_expected(binding,admitted),'EXACT_ONE_USE_ALL_TRAINING_CLOSED')
    need(not ROOT.exists() and not ROOT.is_symlink(),'FRESH_ROOT')
    return plan,binding,pref


def checked_inputs(plan):
    for binding in plan['inputs'].values(): need(ref(binding['path'])==binding,'INPUT_SHA')
    saved.archive_headers(plan['inputs']['capture'],plan['capture_array_headers'],exact_inventory=saved.CAPTURE_KEYS)
    saved.archive_headers(plan['inputs']['target_arrays'],plan['target_array_headers'])


def support_coverage(truth, masks, log_probs, j):
    present=truth['target_present'][:,j]; starts=truth['target_start_ns'][:,j]; rows=len(starts)
    denominator=int(np.unique(starts[present]).size); ak=masks['activity'][:,j]; sk=masks['side'][:,j]
    tie=(log_probs[:,j]==log_probs[:,j].max(1,keepdims=True)).sum(1)>1
    cells={'target_present':present,'target_MISSING':~present,'mature':truth['mature'][:,j],
        'present_not_mature':present&~truth['mature'][:,j],'nominal_complete':truth['nominal_complete'][:,j],
        'state_window_complete':truth['state_window_complete'][:,j],
        'mature_state_window_incomplete':truth['mature'][:,j]&~truth['state_window_complete'][:,j],
        'partial_range_active':truth['partial_range_active'][:,j],'activity_known':ak,'activity_UNKNOWN':~ak,
        'ACTIVE_known':ak&truth['activity'][:,j],'FLAT_known':ak&~truth['activity'][:,j],
        'side_known':sk,'ACTIVE_side_UNKNOWN':ak&truth['activity'][:,j]&~sk,
        'singleton_known':masks['singleton'][:,j],'singleton_UNKNOWN':~masks['singleton'][:,j],
        'argmax_tie_UNKNOWN':tie,'singleton_argmax_tie_UNKNOWN':tie&masks['singleton'][:,j]}
    report={}
    for name,take in cells.items():
        n=int(take.sum()); u=int(np.unique(starts[take&present]).size); missing=int((take&~present).sum())
        report[name]={'row_count':n,'row_denominator':rows,'row_rate':n/rows,
            'unique_identified_target_count':u,'unique_present_target_denominator':denominator,
            'unique_any_issued_mask_rate':u/denominator if denominator else None,'rows_without_target_identity':missing}
    return {'cells':report,'unique_definition':'Unique present target_start_ns IDs with at least one issued mask true; rates may overlap and are not independent observations.',
        'intrinsic_missing_reasons':'NOT_IDENTIFIABLE_FROM_THE_12_CANONICAL_FIELDS',
        'scope':'Truth/logit observation only; feature/provider coverage, learned utility, independent generalization and economics are not measured here.'}


def verify_prior_support(truth, prior, target):
    masks=objective._native_future_state_truth(truth,rows=len(truth['issue_ns']),fold_upper_ns=UPPER,target_contract=target)
    proof={}
    for task,label in (('activity','activity'),('side','side_up')):
        proof[task]={}
        for j,tf in enumerate(TF):
            take=masks[task][:,j]; cell=prior['per_task'][task][tf]
            actual={'count':int(take.sum()),'class_counts':np.bincount(truth[label][take,j].astype(np.int64),minlength=2).tolist(),
                'unique_target_starts':int(np.unique(truth['target_start_ns'][take,j]).size),'mask_sha256':saved.array_sha(take)}
            need(all(cell[k]==v for k,v in actual.items()),'FROZEN_PRIOR_'+task+'_'+tf+'_SUPPORT')
            proof[task][tf]=actual
    digest=objective._native_prior_array_sha(truth)
    need(prior['fit_population']['truth_sha256']==digest,'FROZEN_PRIOR_TRUTH_HASH')
    return {'all_counts_classes_unique_and_mask_hashes_match':True,'truth_sha256':digest,'per_task':proof,'prior_refitted':False}


def class_metrics(log_probs, truth, masks, j):
    z=log_probs[:,j]; unique=(z==z.max(1,keepdims=True)).sum(1)==1
    predicted=np.where(unique,z.argmax(1),3); known=masks['singleton'][:,j]
    labels=np.where(~truth['activity'][:,j],1,np.where(truth['side_up'][:,j],2,0))
    matrix=np.bincount(labels[known]*4+predicted[known],minlength=12).reshape(3,4)
    counts=matrix.sum(1); names=('DOWN','FLAT','UP')
    return {'row_order':list(names),'column_order':[*names,'TIE_UNKNOWN'],'known_truth_confusion':matrix.tolist(),
        'known_truth_counts':{name:int(counts[k]) for k,name in enumerate(names)},
        'recall':{name:float(matrix[k,k]/counts[k]) if counts[k] else None for k,name in enumerate(names)},
        'unresolved_truth':{'count':int((~known).sum()),'prediction_counts':np.bincount(predicted[~known],minlength=4).tolist()},
        'all_row_prediction_counts':np.bincount(predicted,minlength=4).tolist(),'exact_argmax_ties':int((~unique).sum())}


def score_split(outputs, truth, prior_outputs, target):
    rows=len(truth['issue_ns']); need(rows>0,'NONEMPTY_SPLIT')
    masks=objective._native_future_state_truth(truth,rows=rows,fold_upper_ns=UPPER,target_contract=target)
    tensors={k:torch.from_numpy(outputs[k]) for k in HEADS}; prior_tensors={k:torch.from_numpy(prior_outputs[k]) for k in HEADS}
    with torch.no_grad():
        proper=objective.native_future_state_score_sums(model_output=tensors,truth=truth,fold_upper_ns=UPPER,target_contract=target)
        reference=objective.native_future_state_score_sums(model_output=prior_tensors,truth=truth,fold_upper_ns=UPPER,target_contract=target)
        class_logp=objective.native_future_state_log_probabilities(model_output=tensors,target_contract=target).numpy()
        prior_logp=objective.native_future_state_log_probabilities(model_output=prior_tensors,target_contract=target).numpy()
    side=np.exp(-np.logaddexp(0.,-outputs['side_logits'].astype(np.float64))); activity=np.exp(-np.logaddexp(0.,-outputs['activity_logits'].astype(np.float64)))
    class_p=np.exp(class_logp.astype(np.float64)); report={}; activity_count=int(masks['activity'].sum())
    for j,tf in enumerate(TF):
        sk=masks['side'][:,j]; ak=masks['activity'][:,j]; n=int(sk.sum()); y=truth['side_up'][sk,j]
        model_score=proper['per_tf'][tf]; prior_score=reference['per_tf'][tf]
        need(all(model_score[k]['count']==prior_score[k]['count'] for k in ('activity','side','three_class')),'MATCHED_SCORE_SUPPORT')
        rank=binary_rank_metrics(outputs['side_logits'][sk,j],y); prior_rank=binary_rank_metrics(prior_outputs['side_logits'][sk,j],y)
        side_report={'count':n,'unique_target_starts':model_score['side']['unique_target_starts'],'UP':int(y.sum()),'DOWN':n-int(y.sum()),
            'model_AUC':rank['AUC'],'prior_AUC':prior_rank['AUC'],'AUC_undefined_reason':rank['AUC_undefined_reason']}
        for field,label in (('brier_sum','Brier'),('log_loss_sum','NLL')):
            a=model_score['side'][field]/n if n else None; b=prior_score['side'][field]/n if n else None
            side_report.update({f'model_{label}':a,f'prior_{label}':b,f'model_minus_prior_{label}':a-b if n else None})
        true_masks={'DOWN':sk&~truth['side_up'][:,j],'FLAT':ak&~truth['activity'][:,j],'UP':sk&truth['side_up'][:,j],'UNKNOWN':~masks['singleton'][:,j]}
        means={}
        for name,take in true_masks.items():
            count=int(take.sum()); p=class_p[take,j]
            means[name]={'count':count,'unique_present_target_starts':int(np.unique(truth['target_start_ns'][take&truth['target_present'][:,j],j]).size),
                'mean_p_UP_given_activity':float(side[take,j].mean()) if count else None,
                'mean_p_DOWN_given_activity':float((1-side[take,j]).mean()) if count else None,
                'mean_model_DOWN_FLAT_UP':p.mean(0).tolist() if count else None}
        residual=float((side[sk,j]-y).sum())
        report[tf]={'rows':rows,'activity_known':int(ak.sum()),'activity_UNKNOWN':rows-int(ak.sum()),
            'ACTIVE_known':int((ak&truth['activity'][:,j]).sum()),'ACTIVE_side_UNKNOWN':int((ak&truth['activity'][:,j]&~sk).sum()),
            'mean_activity_probability_all_rows':float(activity[:,j].mean()),'conditional_side':side_report,'truth_strata_probabilities':means,
            'coverage':support_coverage(truth,masks,class_logp,j),
            'model_classification':class_metrics(class_logp,truth,masks,j),'prior_classification':class_metrics(prior_logp,truth,masks,j),
            'canonical_score_sums':{'model':model_score,'frozen_TRAIN_prior':prior_score},
            'fixed_output_side_intercept_derivative':{'sum_sigmoid_minus_known_UP':residual,'all_TF_known_activity_denominator':activity_count,
                'native_partial_NLL_derivative':residual/activity_count if activity_count else None}}
        need(sum(v['count'] for v in means.values())==rows,'ALL_TRUTH_ROWS_ACCOUNTED')
    return {'rows':rows,'per_tf':report}


def analyse(outputs, truth, prior_outputs, target):
    years=truth['issue_ns'].astype('datetime64[ns]').astype('datetime64[Y]').astype(str); report={}
    for split in STRATA:
        take=np.ones(len(years),np.bool_) if split=='ALL' else years==split
        need(take.any(),'ALL_FIVE_TRAIN_YEARS_PRESENT')
        report[split]=score_split({k:v[take] for k,v in outputs.items()},{k:v[take] for k,v in truth.items()},
                                 {k:v[take] for k,v in prior_outputs.items()},target)
    need(sum(report[y]['rows'] for y in STRATA[1:])==report['ALL']['rows'],'COMPLETE_YEAR_PARTITION')
    return report


def calculate(plan):
    torch.set_num_threads(1); need(torch.get_num_threads()==1,'CPU_ONE_THREAD')
    capture=saved.load_fields(plan['inputs']['capture'],saved.CAPTURE_FIELDS,plan['capture_array_headers'])
    truth=saved.load_fields(plan['inputs']['target_arrays'],TRUTH,plan['target_array_headers'])
    issue=capture.pop('issue_ns')
    need(len(issue)==ROWS and np.array_equal(issue,truth['issue_ns']) and np.all(np.diff(issue)>0)
         and saved.array_sha(issue)==plan['TRAIN_issue_ns_sha256'],'FULL_CAPTURE_TRUTH_JOIN')
    need(set(truth)==set(TRUTH) and all(capture[k].dtype==np.float32 and capture[k].shape==(ROWS,5) and np.isfinite(capture[k]).all() for k in HEADS),'EXACT_TRUTH_HEADS')
    prior=objective.require_native_future_state_train_prior(read(plan['inputs']['prior']),target_contract=plan['target_contract'])
    prior_support=verify_prior_support(truth,prior,plan['target_contract'])
    with torch.no_grad(): reference={k:v.numpy() for k,v in objective.native_future_state_train_prior_logits(prior,rows=ROWS,target_contract=plan['target_contract']).items()}
    return {'strata':analyse(capture,truth,reference,plan['target_contract']),'score_aggregation':plan['score_aggregation'],
        'expected_issue_TF_cells':ROWS*len(TF),'retained_issue_TF_cells':ROWS*len(TF),'frozen_prior_support_crosscheck':prior_support,
        'payload_counts':{'capture_archive_loads':1,'capture_fields':list(saved.CAPTURE_FIELDS),'target_archive_loads':1,'target_fields':list(TRUTH)},
        'canonical_proper_score_calls':12,'model_forwards':0,'models':0,'checkpoint_loads':0,'backward':0,'fits':0,'optimizer_updates':0,'source_prices':0,'VAL_TEST_rows':0,
        'limits':['All/year strata reuse the same TRAIN-fitted model and wholeTRAIN prior; not OOF or independent generalization.',
            'Endpoint certificates C-L versus H-C are unchanged, not issue-close direction, first-cross order or precise turn onset.',
            'Derivative is exact algebra for a common side-logit intercept at fixed outputs/support; it is not an AdamW trajectory or evidence of a model implementation bug.',
            'No threshold/year/TF/checkpoint is selected; no calibration fit, trading utility or economic edge is established.']}


def publish(name,value):
    path=ROOT/name; saved._publish_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()); return ref(path)


def run(plan_sha):
    started=time.monotonic(); cap=require_capped_cpu_audit_execution(); lock=require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==4*1024**3 and cap['swap']==512*1024**2 and cap['numerical_threads']==1
         and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):],'CAP_LIMITS')
    plan,binding,pref=inspect(plan_sha,admitted=True); checked_inputs(plan); ROOT.mkdir(parents=True,exist_ok=False)
    claim=publish('CLAIM.json',{'stage':STAGE,'plan':binding,'policy_at_claim':pref,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
    old=signal.getsignal(signal.SIGALRM); result=None
    try:
        def expired(*_): raise TimeoutError('ALIGNMENT_WALL')
        signal.signal(signal.SIGALRM,expired); remaining=EXECUTION['internal_seconds']-(time.monotonic()-started); need(remaining>0,'TIME_BEFORE_CALCULATE')
        signal.setitimer(signal.ITIMER_REAL,remaining); report=calculate(plan); checked_inputs(plan)
        need(source_clean(plan) and ref(POLICY)==pref and ref(PLAN)==binding,'SOURCE_INPUT_PLAN_POLICY_BOOKEND')
        result=publish('RESULT.json',{'status':'COMPLETE_NATIVE_FUTURE_STATE_ALIGNMENT','stage':STAGE,'plan':binding,'claim':claim,'policy_at_claim':pref,
            'design':DESIGN,'coverage_requirements':COVERAGE,'report':report,'source_unchanged':True,'test_data_used':False,'economic_edge_proven':False,'elapsed_seconds':time.monotonic()-started})
        terminal=publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'stage':STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,
            'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False}); return {'result':result,'terminal':terminal}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0); publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'stage':STAGE,'plan':binding,'claim':claim,
            'result':result,'remaining_invocations':0,'relaunch_allowed':False,'error':type(error).__name__+':'+str(error)[:2000],'test_data_used':False}); raise
    finally: signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old)


# Distinct one-use accounting on the original unchanged masks; old alignment remains consumed.
BATCH_STAGE='native_future_state_batch_risk_20261123_001'
BATCH_BASENAME='NATIVE_FUTURE_STATE_BATCH_RISK_20261123_001'
BATCH_PLAN=REPO/'configs/research'/(BATCH_BASENAME+'.json');BATCH_ROOT=Path('/home/andre2/GX1_RUNS')/BATCH_BASENAME
BATCH_EROOT=Path('/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001')
BATCH_CAPTURE=('issue_ns','source_rows','side_logits');BATCH_TRUTH=('issue_ns','activity_known','side_known','side_up')
BATCH_BUDGET={'capture_archive_loads':1,'capture_fields':list(BATCH_CAPTURE),'target_archive_loads':1,'target_fields':list(BATCH_TRUTH),
    'ndarray_bytes_decoded':20918450,'original_batches':3694,'batch_rows':96,'last_batch_rows':22,
    'models':0,'checkpoint_loads':0,'forwards':0,'backward':0,'autograd_calls':0,'optimizer_updates':0,'optimizer_constructions':0,
    'fits':0,'source_prices':0,'features':0,'HTTP_calls':0,'new_labels':0,'VAL_TEST_rows':0}


def batch_risk_statistics(logits,activity_known,side_known,side_up,*,batch_rows,prior_logits):
    n=len(logits);shape=(n,len(TF));need(type(batch_rows)is int and batch_rows>0 and n>0,'BATCH_RISK_GEOMETRY')
    need(logits.dtype==np.float32 and logits.shape==shape and np.isfinite(logits).all()
        and all(a.dtype==np.bool_ and a.shape==shape for a in (activity_known,side_known,side_up))
        and not np.any(side_known&~activity_known)and not np.any(side_up&~side_known),'BATCH_RISK_MASKS_HEAD')
    need(prior_logits.dtype==np.float32 and prior_logits.shape==(len(TF),)and np.isfinite(prior_logits).all(),'BATCH_RISK_PRIOR')
    q=np.exp(-np.logaddexp(0.,-logits.astype(np.float64)));p=np.exp(-np.logaddexp(0.,-prior_logits.astype(np.float64)))
    batches=[];weighted_n=np.zeros(5,np.float64);weighted_up=np.zeros(5,np.float64);weighted_final=np.zeros(5,np.float64);weighted_prior=np.zeros(5,np.float64)
    updates=0;total_D=0
    for start in range(0,n,batch_rows):
        stop=min(start+batch_rows,n);ak=activity_known[start:stop];sk=side_known[start:stop];up=side_up[start:stop]
        ca=ak.sum(0,dtype=np.int64);cs=sk.sum(0,dtype=np.int64);cu=(sk&up).sum(0,dtype=np.int64);D=int(ca.sum());r=np.where(sk,q[start:stop]-up,0.).sum(0)
        rp=p*cs-cu
        if D:
            updates+=1;total_D+=D;weighted_n+=cs/D;weighted_up+=cu/D;weighted_final+=r/D;weighted_prior+=rp/D
        batches.append({'logical_step':len(batches)+1,'rows':stop-start,'start_position':start,'stop_position':stop,'D_activity_all_TF':D,
            'activity_known':ca.tolist(),'side_known':cs.tolist(),'UP':cu.tolist(),'DOWN':(cs-cu).tolist(),
            'FINAL_side_residual_sums':r.tolist(),'prior_side_residual_sums':rp.tolist(),'skip_update':D==0})
    report={}
    for j,tf in enumerate(TF):
        sk=side_known[:,j];count=int(sk.sum());up=int(side_up[sk,j].sum());residual=float((q[sk,j]-side_up[sk,j]).sum())
        report[tf]={'known_side_rows':count,'UP':up,'DOWN':count-up,'constant_row_optimum':up/count if count else None,
            'constant_mean_batch_optimum':float(weighted_up[j]/weighted_n[j])if weighted_n[j]>0 else None,
            'weighted_side_mass':float(weighted_n[j]),'weighted_UP_mass':float(weighted_up[j]),'prior_q':float(p[j]),
            'FINAL_mean_q':(residual+up)/count if count else None,'FINAL_side_residual_sum':residual,
            'prior_gradient_global':float((p[j]*count-up)/total_D)if total_D else None,
            'prior_gradient_mean_batch':float(weighted_prior[j]/updates)if updates else None,
            'FINAL_gradient_global':residual/total_D if total_D else None,
            'FINAL_gradient_mean_batch':float(weighted_final[j]/updates)if updates else None,
            'FINAL_gradient_conditional_side':residual/count if count else None,
            'zero_side_support_batches':sum(b['side_known'][j]==0 for b in batches),
            'support_status':'OBSERVED'if count else'NO_SUPPORT_NO_CONSTANT_OPTIMUM'}
    return {'batches':batches,'per_tf':report,'activity_denominator_total':total_D,'updates':updates,'skipped_batches':len(batches)-updates,
        'constant_reference_status':'Closed-form minimizer of mean supported-batch side BCE only; not a new canonical prior/model/calibrator or Adam convergence prediction.'}


def batch_build_plan(*,design,source_base_commit=None):
    d=read(design);need(Path(design['path'])==BATCH_EROOT/'BATCH_RISK_ACCOUNTING_DESIGN001.json'
        and d['schema_version']=='gx1_native_batch_risk_accounting_design_v1'and d['execution_authorized']is False
        and d['execution']==EXECUTION and d['budget']==BATCH_BUDGET,'BATCH_RISK_LOCKED_DESIGN')
    base=build_plan(source_base_commit=source_base_commit);main=read(base['inputs']['main_plan']);original=read(base['inputs']['main_result'])
    alignment_ref=ref(ROOT/'RESULT.json');alignment=read(alignment_ref);terminal_ref=ref(ROOT/'TERMINAL.json');terminal=read(terminal_ref);old_plan=read(alignment['plan'])
    need(terminal['status']=='COMPLETE'and terminal['exit_code']==0 and terminal['result']==alignment_ref
        and alignment['status']=='COMPLETE_NATIVE_FUTURE_STATE_ALIGNMENT'and alignment['source_unchanged']is True and alignment['test_data_used']is False
        and all(old_plan['inputs'][k]==base['inputs'][k]for k in ('main_plan','capture_result','capture','dataset','target_arrays','prior'))
        and alignment['report']['frozen_prior_support_crosscheck']['all_counts_classes_unique_and_mask_hashes_match']is True,'BATCH_RISK_ORIGINAL_ALIGNMENT')
    schedule=main['schedule'];recipe=main['recipe'];steps=original['report']['steps']
    need(schedule['rows']==ROWS and schedule['batch_rows']==recipe['train_batch_rows']==96 and schedule['batches_per_pass']==schedule['logical_batches']==3694
        and schedule['passes']==1 and schedule['shuffle']is False and schedule['drop_last']is False and schedule['skipped_batches']==0
        and len(steps)==3694 and all(len(schedule[k])==3694 for k in ('known_activity_by_batch','update_flags','batch_source_rows_sha256','batch_issue_ns_sha256'))
        and original['plan']==base['inputs']['main_plan']and original['source_unchanged']is True
        and schedule['known_activity_by_batch']==[sum(x['support']['activity'])for x in steps]
        and all(x is True for x in schedule['update_flags']),'BATCH_RISK_ORIGINAL_SCHEDULE')
    need(base['capture_array_headers']['source_rows']=={'shape':[ROWS],'dtype':'<i8','sha256':schedule['source_rows_sha256']},'BATCH_RISK_GLOBAL_SOURCE_ROWS')
    headers=base['target_array_headers'];ch=base['capture_array_headers'];npbytes=sum(int(np.prod(headers[k]['shape']))*np.dtype(headers[k]['dtype']).itemsize for k in BATCH_TRUTH)
    npbytes+=sum(int(np.prod(ch[k]['shape']))*np.dtype(ch[k]['dtype']).itemsize for k in BATCH_CAPTURE);need(npbytes==20918450,'BATCH_RISK_BYTES')
    for key in ('qualification_requirements','observed_parameter_attribution'):read(d[key])
    inputs={**base['inputs'],'batch_risk_design':design,'alignment_result':alignment_ref,'alignment_terminal':terminal_ref,'alignment_plan':alignment['plan'],
        'qualification_requirements':d['qualification_requirements'],'observed_parameter_attribution':d['observed_parameter_attribution']}
    return {'schema_version':'gx1_native_batch_risk_accounting_plan_v1','stage':BATCH_STAGE,'status':'PREPARED_NOT_ADMITTED','plan_path':str(BATCH_PLAN),'output_root':str(BATCH_ROOT),
        'source_base_commit':base['source_base_commit'],'source_owners':base['source_owners'],'inputs':inputs,'execution':EXECUTION,'budget':BATCH_BUDGET,
        'capture_array_headers':ch,'target_array_headers':{k:headers[k]for k in BATCH_TRUTH},'target_contract':base['target_contract'],
        'TRAIN_issue_ns_sha256':base['TRAIN_issue_ns_sha256'],'runtime_versions':base['runtime_versions'],'question':d['question'],
        'method':'Exact original chronologicalB96 masks; compare mean per-batch native partial-loss side derivatives with globally aggregated partial/conditional-side risk. No new labels or canonical prior.'}


def batch_gate_expected(binding,admitted):
    return {'status':'ADMITTED_EXACT_CPU_ONLY_BATCH_RISK_ACCOUNTING'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,
        'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}


def batch_inspect(plan_sha,*,admitted):
    need(Path.cwd().resolve()==REPO and type(plan_sha)is str and len(plan_sha)==64 and all(c in '0123456789abcdef'for c in plan_sha),'BATCH_RISK_CWD_SHA')
    binding={'path':str(BATCH_PLAN),'sha256':plan_sha};plan=read(binding)
    need(plan==batch_build_plan(design=plan['inputs']['batch_risk_design'],source_base_commit=plan['source_base_commit'])and source_clean(plan),'BATCH_RISK_EXACT_PLAN_SOURCE')
    pref=ref(POLICY);policy=read(pref);need(all(policy.get(k)is False for k in saved.FLAGS)and policy[BATCH_STAGE]==batch_gate_expected(binding,admitted),'BATCH_RISK_EXACT_GATE_ALL_TRAINING_CLOSED')
    need(not BATCH_ROOT.exists()and not BATCH_ROOT.is_symlink(),'BATCH_RISK_FRESH_ROOT');return plan,binding,pref


def batch_checked_inputs(plan):
    for b in plan['inputs'].values():need(ref(b['path'])==b,'BATCH_RISK_INPUT_SHA')
    saved.archive_headers(plan['inputs']['capture'],{k:plan['capture_array_headers'][k]for k in BATCH_CAPTURE},exact_inventory=saved.CAPTURE_KEYS)
    saved.archive_headers(plan['inputs']['target_arrays'],plan['target_array_headers'])


def batch_calculate(plan,*,progress=None):
    torch.set_num_threads(1);need(torch.get_num_threads()==1,'BATCH_RISK_CPU_THREAD');progress={}if progress is None else progress;progress['phase']='CAPTURE3_TARGETS4'
    capture=saved.load_fields(plan['inputs']['capture'],BATCH_CAPTURE,plan['capture_array_headers']);truth=saved.load_fields(plan['inputs']['target_arrays'],BATCH_TRUTH,plan['target_array_headers'])
    progress['capture_fields_decoded']=list(BATCH_CAPTURE);progress['target_fields_decoded']=list(BATCH_TRUTH)
    main=read(plan['inputs']['main_plan']);original=read(plan['inputs']['main_result']);schedule=main['schedule'];issue=capture['issue_ns'];source=capture['source_rows']
    need(issue.shape==source.shape==(ROWS,)and np.array_equal(issue,truth['issue_ns'])and np.all(np.diff(issue)>0)and np.all(np.diff(source)>0)
        and saved.array_sha(issue)==schedule['issue_ns_sha256']and saved.array_sha(source)==schedule['source_rows_sha256'],'BATCH_RISK_AXES')
    prior=objective.require_native_future_state_train_prior(read(plan['inputs']['prior']),target_contract=plan['target_contract'])
    prior_logits=objective.native_future_state_train_prior_logits(prior,rows=1,target_contract=plan['target_contract'])['side_logits'].numpy()[0]
    stats=batch_risk_statistics(capture['side_logits'],truth['activity_known'],truth['side_known'],truth['side_up'],batch_rows=96,prior_logits=prior_logits)
    need(stats['updates']==3694 and stats['skipped_batches']==0 and len(stats['batches'])==3694,'BATCH_RISK_UPDATE_COUNTS');steps=original['report']['steps']
    for b,step,source_hash,issue_hash,D in zip(stats['batches'],steps,schedule['batch_source_rows_sha256'],schedule['batch_issue_ns_sha256'],schedule['known_activity_by_batch']):
        start,stop=b['start_position'],b['stop_position'];need(saved.array_sha(source[start:stop])==source_hash and saved.array_sha(issue[start:stop])==issue_hash
            and b['logical_step']==step['logical_step']and b['D_activity_all_TF']==D and b['activity_known']==step['support']['activity']
            and b['side_known']==step['support']['side']and b['skip_update']==step['skip_update'],'BATCH_RISK_ORIGINAL_BATCH_'+str(b['logical_step']))
    alignment=read(plan['inputs']['alignment_result'])['report']['strata']['ALL']['per_tf']
    for j,tf in enumerate(TF):
        for task,mask in (('activity',truth['activity_known'][:,j]),('side',truth['side_known'][:,j])):
            cell=prior['per_task'][task][tf];need(int(mask.sum())==cell['count']and saved.array_sha(mask)==cell['mask_sha256'],'BATCH_RISK_PRIOR_MASK_'+task+'_'+tf)
        cell=stats['per_tf'][tf];reference=alignment[tf];side=reference['conditional_side'];deriv=reference['fixed_output_side_intercept_derivative'];pc=prior['per_task']['side'][tf]
        need([cell['DOWN'],cell['UP']]==pc['class_counts']==[side['DOWN'],side['UP']]and cell['known_side_rows']==side['count']
            and cell['FINAL_side_residual_sum']==deriv['sum_sigmoid_minus_known_UP']and stats['activity_denominator_total']==deriv['all_TF_known_activity_denominator']
            and cell['FINAL_gradient_global']==deriv['native_partial_NLL_derivative'],'BATCH_RISK_EXISTING_GLOBAL_ACCOUNTING_'+tf)
    progress['phase']='COMPLETE_BATCH_ACCOUNTING'
    return {**stats,'original_schedule_reproduced_exactly':True,'original_prior_masks_classes_and_FINAL_global_derivatives_exact':True,
        'unique_side_targets_from_original_frozen_prior':{tf:prior['per_task']['side'][tf]['unique_target_starts']for tf in TF},
        'counting_unit':'Original issued rows and original B96 batches; repeated TF target labels are not independent targets.',
        'payload_counts':BATCH_BUDGET,'probability_semantics':'q=P(UP|activity) only; not joint a*q',
        'formulae':{'native_batch_loss':'(sum allTF activityBCE + sum allTF sideKnownBCE) / D_b; D_b=sum allTF activityKnown',
            'constant_mean_batch_optimum':'sum_b(UP_bj/D_b) / sum_b(nSide_bj/D_b)',
            'FINAL_mean_batch_gradient':'mean_b(sum_i sideKnown_ij*(sigmoid(z_ij)-y_ij)/D_b)',
            'FINAL_global_gradient':'sum_bi sideKnown_ij*(sigmoid(z_ij)-y_ij) / sum_b D_b'},
        'limits':['Closed-form references summarize original known targets only; no new prior/model/calibrator fit, labels or calendar masks.',
            'Mean-batch risk is not actual AdamW trajectory or its convergence optimum. Dropout-mode gradients, global clipping and momentum/decay are not reconstructed.',
            'SavedFINAL eval outputs are held fixed; do not reuse original training clip norms as if they were FINAL batch gradient norms.',
            'Unknown/calendar/vintage/task-utility/OOS/edge/economic questions remain outside this conditional optimization accounting.']}


def batch_publish(name,value):
    path=BATCH_ROOT/name;saved._publish_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());return ref(path)


def batch_run(plan_sha):
    began=time.monotonic();cap=require_capped_cpu_audit_execution();lock=require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==4*1024**3 and cap['swap']==512*1024**2 and cap['numerical_threads']==1
        and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):],'BATCH_RISK_CAPS')
    plan,binding,pref=batch_inspect(plan_sha,admitted=True);batch_checked_inputs(plan);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();BATCH_ROOT.mkdir(parents=True,exist_ok=False)
    handlers={s:signal.getsignal(s)for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};claim=result=None;progress={'phase':'CLAIM'}
    try:
        def expired(*_):raise TimeoutError('BATCH_RISK_WALL')
        def stopped(s,*_):raise InterruptedError('BATCH_RISK_EXTERNAL_SIGNAL_'+str(s))
        signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
        claim=batch_publish('CLAIM.json',{'stage':BATCH_STAGE,'plan':binding,'policy_at_claim':pref,'boot_id':boot,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        remaining=EXECUTION['internal_seconds']-(time.monotonic()-began);need(remaining>0,'BATCH_RISK_TIME');signal.setitimer(signal.ITIMER_REAL,remaining)
        report=batch_calculate(plan,progress=progress);batch_checked_inputs(plan)
        need(source_clean(plan)and ref(POLICY)==pref and ref(BATCH_PLAN)==binding and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,'BATCH_RISK_SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND')
        result=batch_publish('RESULT.json',{'status':'COMPLETE_NATIVE_BATCH_RISK_ACCOUNTING','stage':BATCH_STAGE,'plan':binding,'claim':claim,'policy_at_claim':pref,'boot_id':boot,
            'report':report,'source_unchanged':True,'test_data_used':False,'economic_edge_proven':False,'elapsed_seconds':time.monotonic()-began})
        terminal=batch_publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'stage':BATCH_STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,
            'source_unchanged':True,'test_data_used':False});return {'result':result,'terminal':terminal}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0);batch_publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'stage':BATCH_STAGE,'plan':binding,'claim':claim,'result':result,
            'remaining_invocations':0,'relaunch_allowed':False,'progress':progress,'error':type(error).__name__+':'+str(error)[:2000],'test_data_used':False});raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan-sha256');mode.add_argument('--batch-risk-plan-sha256');parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.batch_risk_plan_sha256 is not None:
        import sys
        need(sys.argv[1:]==['--batch-risk-plan-sha256',args.batch_risk_plan_sha256]+(['--preflight']if args.preflight else[]),'BATCH_RISK_EXACT_CLI')
        if args.preflight:
            plan,binding,pref=batch_inspect(args.batch_risk_plan_sha256,admitted=False);batch_checked_inputs(plan)
            print(json.dumps({'status':'PREFLIGHT_BATCH_RISK_PASS','plan':binding,'policy':pref,'claim_created':False,'array_payloads_decoded':0,'models':0,'checkpoint_loads':0,'forwards':0,'fits':0}))
        else:print(json.dumps(batch_run(args.batch_risk_plan_sha256)))
    elif args.preflight:
        plan,binding,pref=inspect(args.plan_sha256,admitted=False);checked_inputs(plan)
        print(json.dumps({'status':'PREFLIGHT_PASS','plan':binding,'policy':pref,'claim_created':False,'array_payloads_decoded':0,'models':0,'checkpoint_loads':0,'forwards':0}))
    else:print(json.dumps(run(args.plan_sha256)))

if __name__=='__main__':main()
