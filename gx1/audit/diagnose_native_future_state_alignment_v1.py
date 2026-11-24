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


# Separate CPU-only projection-moment probe; the original frozen study stays consumed.
from gx1.audit import diagnose_native_future_state_bias_v1 as frozen_owner
train,core=frozen_owner.train,frozen_owner.core
MOMENT_STAGE='native_future_state_projection_moment_20261123_002'
MOMENT_BASENAME='NATIVE_FUTURE_STATE_PROJECTION_MOMENT_20261123_002'
MOMENT_PLAN=REPO/'configs/research'/(MOMENT_BASENAME+'.json');MOMENT_ROOT=Path('/home/andre2/GX1_RUNS')/MOMENT_BASENAME
MOMENT_EROOT=Path('/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001')
MOMENT_CAPTURE=('issue_ns','hidden3694','cursor3694__side_logits')
MOMENT_TRUTH=('issue_ns','truth__activity_known','truth__side_known','truth__side_up')
MOMENT_EXECUTION={'class':'audit','memory':'4G','swap':'512M','numerical_threads':1,'outer_seconds':300,'internal_seconds':270,'kill_after_seconds':10}
MOMENT_BUDGET={'checkpoint_loads':1,'checkpoint_bytes':97898135,'capture_archive_loads':1,'capture_fields':list(MOMENT_CAPTURE),
 'cohort_archive_loads':1,'cohort_fields':list(MOMENT_TRUTH),'ndarray_bytes_decoded':54118197,'rows':20727,'timeframes':5,'hidden_width':128,
 'technical_zero_side_gradient_arms':2,'analytical_delta_logit_projections':10,'model_constructions':0,'model_forwards':0,'readout_forwards':0,
 'optimizer_constructions':0,'optimizer_steps':0,'applied_parameter_updates':0,'autograd_calls':0,'fits':0,'calibrator_fits':0,'features':0,'source_prices':0,'new_labels':0,'HTTP_calls':0,'VAL_TEST_rows':0}


def moment_zero_gradient_delta(p,m,v,step,options,*,reset_first_moment=False):
 p,m,v=(np.asarray(x,dtype=np.float64)for x in (p,m,v));need(p.shape==m.shape==v.shape and all(np.isfinite(x).all()for x in (p,m,v))and np.all(v>=0),'MOMENT_ARRAYS')
 need(type(step)is int and step>=0 and type(reset_first_moment)is bool,'MOMENT_STEP')
 lr,eps,decay=options['lr'],options['eps'],options['weight_decay'];beta1,beta2=options['betas']
 need(all(np.isfinite(x)for x in (lr,eps,decay,beta1,beta2))and lr>=0 and eps>0 and decay>=0 and 0<=beta1<1 and 0<=beta2<1
  and options.get('maximize',False)is False and options.get('amsgrad',False)is False,'MOMENT_OPTIONS')
 t=step+1;m_next=np.zeros_like(m)if reset_first_moment else beta1*m;v_next=beta2*v
 decay_delta=-lr*decay*p;moment_delta=-lr*(m_next/(1-beta1**t))/(np.sqrt(v_next/(1-beta2**t))+eps)
 return {'delta':decay_delta+moment_delta,'decay_delta':decay_delta,'moment_delta':moment_delta}


def moment_extract_heads(cp,recipe,*,width=128):
 import torch
 opt=cp['optimizer_state'];need(type(opt)is dict and set(opt)=={'state','param_groups'}and len(opt['param_groups'])==1,'MOMENT_OPTIMIZER_SCHEMA')
 group=opt['param_groups'][0];names=cp['optimizer_parameter_names'];ids=group['params'];need(type(names)is list and all(type(n)is str for n in names)and len(set(names))==len(names)
  and type(ids)is list and len(ids)==len(names)and len(set(ids))==len(ids)and all(type(i)is int for i in ids)
  and set(group)==set(recipe)|{'params'},'MOMENT_PARAMETER_ID_ORDER')
 for k,value in recipe.items():need((tuple(group[k])==tuple(value)if k=='betas'else type(group[k])is type(value)and group[k]==value),'MOMENT_ACTUAL_RECIPE_'+k)
 need(group['amsgrad']is False and group['maximize']is False and cp['scheduler_state']is None,'MOMENT_SUPPORTED_ADAMW')
 options={k:group[k]for k in recipe};heads={};updates=cp['position']['optimizer_updates']
 for tf in train.TF:
  head={'step':None,'options':options.copy(),'parameter_names':{},'parameter_ids':{},'tensor_state_sha256':{}}
  for kind,shape in (('weight',(1,width)),('bias',(1,))):
   name=f'readout.side_heads.{tf}.{kind}';need(name in names and name in cp['model_state'],'MOMENT_ACTUAL_PARAMETER_NAME')
   index=ids[names.index(name)];slot=opt['state'][index];parameter=cp['model_state'][name]
   need(type(slot)is dict and set(slot)=={'step','exp_avg','exp_avg_sq'}and all(isinstance(x,torch.Tensor)and x.device.type=='cpu'and not x.requires_grad and bool(torch.isfinite(x).all())for x in (parameter,*slot.values())),'MOMENT_CPU_TENSORS')
   step=slot['step'];need(step.dtype==torch.float32 and step.numel()==1 and float(step).is_integer()and float(step)==updates and type(updates)is int and updates>0,'MOMENT_ACTUAL_STEP')
   need(parameter.dtype==torch.float32 and tuple(parameter.shape)==shape and all(slot[k].dtype==torch.float32 and slot[k].shape==parameter.shape for k in ('exp_avg','exp_avg_sq'))
    and bool((slot['exp_avg_sq']>=0).all()),'MOMENT_PARAMETER_SLOTS')
   head['step']=int(float(step));head['parameter_names'][kind]=name;head['parameter_ids'][kind]=index
   for label,tensor in ((kind,parameter),(kind+'_exp_avg',slot['exp_avg']),(kind+'_exp_avg_sq',slot['exp_avg_sq'])):
    head[label]=tensor.detach().numpy().astype(np.float64,copy=True).reshape(-1);head['tensor_state_sha256'][label]=core.state_sha(tensor)
  heads[tf]=head
 return heads


def moment_projection_report(hidden,logits,activity_known,side_known,side_up,heads):
 n=len(logits);need(hidden.dtype==np.float32 and hidden.ndim==3 and hidden.shape[:2]==(n,5)and logits.dtype==np.float32 and logits.shape==(n,5)
  and np.isfinite(hidden).all()and np.isfinite(logits).all()and all(a.dtype==np.bool_ and a.shape==(n,5)for a in (activity_known,side_known,side_up))
  and not np.any(side_known&~activity_known)and not np.any(side_up&~side_known),'MOMENT_FIXED_COHORT')
 D=int(activity_known.sum());report={}
 def metrics(z,take,y):
  if not take.any():return {'count':0,'mean_q':None,'actual_UP_fraction':None,'probability_minus_actual':None,'Brier':None,'NLL':None}
  z=z[take];y=y[take].astype(np.float64);q=np.exp(-np.logaddexp(0.,-z))
  return {'count':len(z),'mean_q':float(q.mean()),'actual_UP_fraction':float(y.mean()),'probability_minus_actual':float((q-y).mean()),
   'Brier':float(np.mean((q-y)**2)),'NLL':float(np.mean(np.logaddexp(0.,z)-y*z))}
 for j,tf in enumerate(train.TF):
  h=hidden[:,j].astype(np.float64);z=logits[:,j].astype(np.float64);sk=side_known[:,j];y=side_up[:,j];head=heads[tf]
  need(head['weight'].shape==head['weight_exp_avg'].shape==head['weight_exp_avg_sq'].shape==(h.shape[1],)
   and head['bias'].shape==head['bias_exp_avg'].shape==head['bias_exp_avg_sq'].shape==(1,),'MOMENT_HEAD_WIDTH')
  baseline=metrics(z,sk,y);residual=np.exp(-np.logaddexp(0.,-z[sk]))-y[sk]
  gb=float(residual.sum()/D)if D else 0.;gw=(h[sk].T@residual)/D if D else np.zeros(h.shape[1],np.float64);arms={}
  for label,reset in (('original_m',False),('reset_m',True)):
   dw=moment_zero_gradient_delta(head['weight'],head['weight_exp_avg'],head['weight_exp_avg_sq'],head['step'],head['options'],reset_first_moment=reset)
   db=moment_zero_gradient_delta(head['bias'],head['bias_exp_avg'],head['bias_exp_avg_sq'],head['step'],head['options'],reset_first_moment=reset)
   dz=h@dw['delta']+db['delta'][0];after=metrics(z+dz,sk,y)
   arms[label]={**after,'mean_delta_side_logit_known':float(dz[sk].mean())if sk.any()else None,'mean_delta_side_logit_all_rows':float(dz.mean()),
    'projection_weight_delta_l2':float(np.linalg.norm(dw['delta'])),'bias_delta':float(db['delta'][0]),'bias_decay_delta':float(db['decay_delta'][0]),
    'bias_moment_delta':float(db['moment_delta'][0]),'projection_moment_delta_l2':float(np.linalg.norm(dw['moment_delta'])),
    'native_partial_directional_derivative':float(gw@dw['delta']+gb*db['delta'][0])if D and sk.any()else None,
    'native_partial_moment_directional_derivative':float(gw@dw['moment_delta']+gb*db['moment_delta'][0])if D and sk.any()else None,
    'mean_q_delta':after['mean_q']-baseline['mean_q']if sk.any()else None,'Brier_delta':after['Brier']-baseline['Brier']if sk.any()else None,
    'NLL_delta':after['NLL']-baseline['NLL']if sk.any()else None}
  report[tf]={'baseline':baseline,'arms':arms,'step_before':head['step'],'hypothetical_step':head['step']+1,
   'fixed_cohort_native_partial_bias_derivative':gb if D and sk.any()else None,
   'fixed_cohort_gradient_dot_first_moment':float(gw@head['weight_exp_avg']+gb*head['bias_exp_avg'][0])if D and sk.any()else None,
   'first_moment_effect_original_minus_reset':{k:arms['original_m'][k]-arms['reset_m'][k]if arms['original_m'][k]is not None else None for k in
    ('mean_delta_side_logit_known','mean_delta_side_logit_all_rows','mean_q','Brier','NLL','native_partial_directional_derivative')},
   'support_status':'OBSERVED_FIXED_REFERENCE_COHORT'if sk.any()else'NO_SUPPORT_NOT_A_LOSS_RESULT'}
 return {'per_tf':report,'activity_denominator_all_TF':D,'probability_semantics':'Conditionalq=P(UP|activity) only; no jointa*q attribution',
  'numerical_scope':'Float64 analytic AdamW equations and fixed-h delta-logit projections, not bitexact CUDA/foreach updates or a model forward.'}


def moment_build_plan(*,design,source_base_commit=None):
 import torch,sys
 from gx1.audit import audit_native_selective_moves_v1 as saved
 d=read(design);need(Path(design['path'])==MOMENT_EROOT/'PROJECTION_MOMENT_DESIGN001.json'and d['schema_version']=='gx1_native_projection_moment_design_v1'
  and d['execution_authorized']is False and d['execution']==MOMENT_EXECUTION and d['budget']==MOMENT_BUDGET,'MOMENT_LOCKED_DESIGN')
 frozen_ref=ref(frozen_owner.ROOT/'RESULT.json');frozen=read(frozen_ref);ft_ref=ref(frozen_owner.ROOT/'TERMINAL.json');ft=read(ft_ref);fp=read(frozen['plan'])
 need(ft['status']=='COMPLETE'and ft['exit_code']==0 and ft['result']==frozen_ref and frozen['status']=='COMPLETE_NATIVE_BIAS_FROZEN_DIAGNOSIS'
  and frozen['source_unchanged']is True and frozen['test_data_used']is False and frozen['report']['rows']==frozen_owner.COHORT_ROWS,'MOMENT_AUTHENTIC_FROZEN_COHORT')
 main=read(fp['inputs']['main_plan']);train.require_source_inventory(main['source_owners']);pair=read(fp['inputs']['paired_plan']);frozen_owner.validate_function_metadata(main,pair)
 mt=read(fp['inputs']['main_terminal']);original=read(fp['inputs']['main_result']);receipt_ref=ref(train.MAIN_ROOT/'FINAL.json');receipt=read(receipt_ref)
 need(mt['status']=='COMPLETE'and mt['exit_code']==0 and mt['result']==fp['inputs']['main_result']and original['plan']==fp['inputs']['main_plan']
  and original['source_unchanged']is True and receipt['checkpoint']==original['report']['final_checkpoint']
  and receipt['proof']==original['report']['final_proof']and receipt['original_plan']==receipt['execution_plan']==fp['inputs']['main_plan']
  and fp['checkpoint_receipts'][-1]==receipt_ref and receipt['producer_lane']=='main'and receipt['logical_cursor']==receipt['optimizer_updates']==3694,'MOMENT_AUTHENTIC_MAIN_FINAL')
 need(str(torch.__version__)==receipt['producer_runtime']['library_versions']['torch']and np.__version__==receipt['producer_runtime']['library_versions']['numpy'],'MOMENT_CPU_READER_LIBRARY_IDENTITY')
 cohort_ref=pair['metadata']['cohort_report']['cohort_snapshot'];cohort=read(cohort_ref);ch=frozen['report']['capture_array_headers'];th=cohort['array_headers']
 need(ch['hidden3694']['shape']==[frozen_owner.COHORT_ROWS,5,128]and ch['hidden3694']['dtype']=='<f4'and ch['cursor3694__side_logits']['shape']==[frozen_owner.COHORT_ROWS,5]
  and ch['issue_ns']==th['issue_ns']and main['metadata']['cohort_report']==pair['metadata']['cohort_report'],'MOMENT_SAME_CAPTURE_TRUTH')
 npbytes=sum(int(np.prod(h['shape']))*np.dtype(h['dtype']).itemsize for h in [*(ch[k]for k in MOMENT_CAPTURE),*(th[k]for k in MOMENT_TRUTH)])
 need(npbytes==54118197 and Path(receipt['checkpoint']['path']).stat().st_size==97898135,'MOMENT_PAYLOAD_BUDGET')
 for key in ('observed_parameter_attribution','batch_risk_result','qualification_requirements'):read(d[key])
 batch=read(d['batch_risk_result']);need(batch['status']=='COMPLETE_NATIVE_BATCH_RISK_ACCOUNTING'and batch['source_unchanged']is True and batch['test_data_used']is False,'MOMENT_COMPLETED_BATCH_ACCOUNTING')
 names=set(fp['source_owners'])|{SELF,TEST,saved.SELF,saved.TEST};need(all(not Path(x).is_absolute()and '..'not in Path(x).parts and Path(x).suffix in ('.py','.sh')for x in names),'MOMENT_SOURCE_NAMES')
 inputs={'design':design,**{k:fp['inputs'][k]for k in ('main_plan','paired_plan','main_result','main_terminal')},'final_receipt':receipt_ref,'checkpoint':receipt['checkpoint'],
  'frozen_result':frozen_ref,'frozen_terminal':ft_ref,'frozen_plan':frozen['plan'],'capture':frozen['report']['capture'],'cohort_snapshot':cohort_ref,'cohort_arrays':cohort['arrays'],
  **{k:d[k]for k in ('observed_parameter_attribution','batch_risk_result','qualification_requirements')}}
 return {'schema_version':'gx1_native_projection_moment_plan_v1','stage':MOMENT_STAGE,'status':'PREPARED_NOT_ADMITTED','plan_path':str(MOMENT_PLAN),'output_root':str(MOMENT_ROOT),
  'source_base_commit':source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'source_owners':{n:frozen_owner.sha(REPO/n)for n in sorted(names)},
  'inputs':inputs,'execution':MOMENT_EXECUTION,'budget':MOMENT_BUDGET,'runtime_versions':{'numpy':np.__version__,'torch':str(torch.__version__),'python':sys.version.split()[0]},'GPU_runtime_used':False,'capture_array_headers':ch,'cohort_array_headers':th,'question':d['question'],
  'lr_scope':'Use actual stored terminal optimizer-group lr for a hypotheticalnextg0 step only; terminal training schedule has no admitted next update or defined next training LR.',
  'numeric_scope':'Float64 analytic head deltas on savedFINAL hidden; no actual parameter assignment, optimizer construction, model/readout forward, or CUDA bitexact-update claim.'}


def moment_source_clean(plan):
 return (subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current'
  and subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_base_commit']
  and not subprocess.check_output(['git','status','--porcelain','--','gx1','scripts','tests'],cwd=REPO,text=True)
  and all(frozen_owner.sha(REPO/n)==v for n,v in plan['source_owners'].items()))


def moment_gate_expected(binding,admitted):
 return {'status':'ADMITTED_EXACT_CPU_ONLY_PROJECTION_MOMENT'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,
  'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}


def moment_inspect(plan_sha,*,admitted):
 need(Path.cwd().resolve()==REPO and type(plan_sha)is str and len(plan_sha)==64 and all(c in '0123456789abcdef'for c in plan_sha),'MOMENT_CWD_SHA')
 binding={'path':str(MOMENT_PLAN),'sha256':plan_sha};plan=read(binding)
 need(plan==moment_build_plan(design=plan['inputs']['design'],source_base_commit=plan['source_base_commit'])and moment_source_clean(plan),'MOMENT_EXACT_PLAN_SOURCE')
 pref=ref(POLICY);policy=read(pref);need(all(policy.get(k)is False for k in ('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed'))
  and policy[MOMENT_STAGE]==moment_gate_expected(binding,admitted),'MOMENT_EXACT_GATE_ALL_TRAINING_CLOSED')
 need(not MOMENT_ROOT.exists()and not MOMENT_ROOT.is_symlink(),'MOMENT_FRESH_ROOT');return plan,binding,pref


def moment_checked_inputs(plan):
 from gx1.audit import audit_native_selective_moves_v1 as saved
 for b in plan['inputs'].values():need(ref(b['path'])==b,'MOMENT_INPUT_SHA')
 saved.archive_headers(plan['inputs']['capture'],{k:plan['capture_array_headers'][k]for k in MOMENT_CAPTURE},exact_inventory=plan['capture_array_headers'])
 saved.archive_headers(plan['inputs']['cohort_arrays'],{k:plan['cohort_array_headers'][k]for k in MOMENT_TRUTH},exact_inventory=plan['cohort_array_headers'])


def moment_calculate(plan,*,progress=None):
 import torch
 from gx1.audit import audit_native_selective_moves_v1 as saved
 torch.set_num_threads(1);need(torch.get_num_threads()==1 and not torch.cuda.is_initialized(),'MOMENT_CPU_ONLY');progress={}if progress is None else progress
 main=read(plan['inputs']['main_plan']);original=read(plan['inputs']['main_result']);receipt=read(plan['inputs']['final_receipt']);frozen=read(plan['inputs']['frozen_result'])['report']
 progress['phase']='SAFE_CPU_CHECKPOINT';cp=torch.load(plan['inputs']['checkpoint']['path'],map_location='cpu',weights_only=True);progress['checkpoint_loads']=1
 frozen_owner.validate_checkpoint(cp,receipt,main,original,3694);heads=moment_extract_heads(cp,main['recipe']['optimizer']);need(train.checkpoint_proof(cp)==receipt['proof'],'MOMENT_CHECKPOINT_UNMUTATED');del cp
 progress['phase']='SAVED_HIDDEN_TRUTH';capture=saved.load_fields(plan['inputs']['capture'],MOMENT_CAPTURE,plan['capture_array_headers']);truth=saved.load_fields(plan['inputs']['cohort_arrays'],MOMENT_TRUTH,plan['cohort_array_headers'])
 need(np.array_equal(capture['issue_ns'],truth['issue_ns'])and len(capture['issue_ns'])==frozen_owner.COHORT_ROWS,'MOMENT_SAME_ORIGINAL_COHORT')
 progress['capture_fields_decoded']=list(MOMENT_CAPTURE);progress['cohort_fields_decoded']=list(MOMENT_TRUTH)
 with torch.no_grad():report=moment_projection_report(capture['hidden3694'],capture['cursor3694__side_logits'],truth['truth__activity_known'],truth['truth__side_known'],truth['truth__side_up'],heads)
 reference=frozen['checkpoint_metrics'][-1];need(reference['cursor']==3694,'MOMENT_FINAL_REFERENCE')
 for j,tf in enumerate(train.TF):
  cell=reference['metrics']['ALL'][tf]['side'];actual=report['per_tf'][tf]['baseline'];need(float(heads[tf]['bias'][0])==cell['head_bias']and actual['count']==cell['count']
   and actual['mean_q']==cell['mean_probability']and actual['Brier']==cell['Brier']and actual['NLL']==cell['logloss']
   and report['per_tf'][tf]['fixed_cohort_native_partial_bias_derivative']==cell['native_partial_NLL_intercept_derivative'],'MOMENT_EXISTING_FINAL_METRICS_'+tf)
 need(not torch.cuda.is_initialized(),'MOMENT_NO_CUDA_INITIALIZATION');progress['phase']='COMPLETE_ANALYTIC_PROBE';support=original['report']['steps'][-1]['support']['side']
 return {**report,'parameter_binding':{tf:{k:heads[tf][k]for k in ('step','options','parameter_names','parameter_ids','tensor_state_sha256')}for tf in train.TF},
  'cohort_scope':{'original_issue_rows':frozen_owner.COHORT_ROWS,'whole_TRAIN_issue_rows':354550,'sampling':'Original immutable20727subset, not fullTRAIN; no new labels/calendar masks',
   'unique_known_side_targets_from_original_frozen_report':{tf:reference['metrics']['ALL'][tf]['side']['unique_target_bars']for tf in train.TF}},
  'authentic_full_checkpoint_proof_verified_before_and_after_extraction':True,'stored_terminal_optimizer_lr_used':heads[train.TF[0]]['options']['lr'],
  'next_actual_training_lr':'UNDEFINED_NO_ADMITTED_NEXT_TRAINING_STEP','last_original_batch_side_support':dict(zip(train.TF,support)),
  'last_original_batch_zero_side_support_TFs':[tf for tf,n in zip(train.TF,support)if n==0],
  'gradient_semantics':'Technical zero gradient tensor, not grad=None; grad=None would skip the AdamW parameter update.',
  'all5_probes':'Hypothetical next zero-side-gradient head updates; no actual next batch, parameter update or training trajectory is reconstructed.',
  'payload_counts':MOMENT_BUDGET,'limits':['The large observed term is projection weight dot representation. This probe holds h fixed and isolates only current direct-head first-moment mechanics.',
   'All5g0 probes are reported; H4/D1 also had zero side support in the original final batch. This does not turn hypothetical step3695 into an actual historical step.',
   'Conditionalq only. Activity heads and jointa*q, shared-backbone moments/Jacobians, dropout/clip gradients, full causal origin, generalization and economics are not measured.',
   'Float64 diagnostic equations are not claimed bitexact to actualCUDA/foreach kernels. Original state, targets, capture and prior are never modified.']}


def moment_publish(name,value):
 path=MOMENT_ROOT/name;saved._publish_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());return ref(path)


def moment_run(plan_sha):
 from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution,require_capped_lock_ancestry
 began=time.monotonic();cap=require_capped_cpu_audit_execution();lock=require_capped_lock_ancestry()
 need(cap['memory_max']==cap['memory_high']==4*1024**3 and cap['swap']==512*1024**2 and cap['numerical_threads']==1
  and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):],'MOMENT_CAPS')
 plan,binding,pref=moment_inspect(plan_sha,admitted=True);moment_checked_inputs(plan);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();MOMENT_ROOT.mkdir(parents=True,exist_ok=False)
 handlers={s:signal.getsignal(s)for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};claim=result=None;progress={'phase':'CLAIM'}
 try:
  def expired(*_):raise TimeoutError('PROJECTION_MOMENT_WALL')
  def stopped(s,*_):raise InterruptedError('PROJECTION_MOMENT_EXTERNAL_SIGNAL_'+str(s))
  signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
  claim=moment_publish('CLAIM.json',{'stage':MOMENT_STAGE,'plan':binding,'policy_at_claim':pref,'boot_id':boot,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
  remaining=MOMENT_EXECUTION['internal_seconds']-(time.monotonic()-began);need(remaining>0,'MOMENT_TIME');signal.setitimer(signal.ITIMER_REAL,remaining)
  report=moment_calculate(plan,progress=progress);moment_checked_inputs(plan)
  need(moment_source_clean(plan)and ref(POLICY)==pref and ref(MOMENT_PLAN)==binding and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,'MOMENT_SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND')
  result=moment_publish('RESULT.json',{'status':'COMPLETE_NATIVE_PROJECTION_MOMENT','stage':MOMENT_STAGE,'plan':binding,'claim':claim,'policy_at_claim':pref,'boot_id':boot,
   'report':report,'source_unchanged':True,'test_data_used':False,'economic_edge_proven':False,'elapsed_seconds':time.monotonic()-began})
  terminal=moment_publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'stage':MOMENT_STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,
   'source_unchanged':True,'test_data_used':False});return {'result':result,'terminal':terminal}
 except BaseException as error:
  signal.setitimer(signal.ITIMER_REAL,0);moment_publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'stage':MOMENT_STAGE,'plan':binding,'claim':claim,'result':result,
   'remaining_invocations':0,'relaunch_allowed':False,'progress':progress,'error':type(error).__name__+':'+str(error)[:2000],'test_data_used':False});raise
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  for s,h in handlers.items():signal.signal(s,h)


# Provider-HLC conditional batch arithmetic; existing studies remain consumed.
HLC_BATCH_NAME='NATIVE_PROVIDER_HLC_BATCH_RISK_20261124_001'
HLC_BATCH_STAGE=HLC_BATCH_NAME.lower();HLC_BATCH_PLAN=REPO/'configs/research'/(HLC_BATCH_NAME+'.json');HLC_BATCH_ROOT=Path('/home/andre2/GX1_RUNS')/HLC_BATCH_NAME
HLC_BATCH_DESIGN={'path':str(BATCH_EROOT/'PROVIDER_HLC_BATCH_RISK_DESIGN002.json'),'sha256':'15027441e09cd64d85608b5ff118a4530c75a1e38386d727e8c8df20d89848cd'}
HLC_BATCH_ANCHOR='9f6cfe40469486b549f4430232b042e2439dbf45'
HLC_BATCH_CAPTURE=('issue_ns','activity_logits','side_logits')


def provider_hlc_batch_source_bridge():
    before=subprocess.check_output(['git','show',HLC_BATCH_ANCHOR+':'+SELF],cwd=REPO);after=(REPO/SELF).read_bytes()
    begin=b'# Provider-HLC conditional batch arithmetic; existing studies remain consumed.\n';end=b'# End provider-HLC conditional batch arithmetic.\n\n'
    need(after.count(begin)==after.count(end)==1,'HLC_BATCH_SINGLE_ADDITIVE_BLOCK')
    first,tail=after.split(begin);_,last=tail.split(end)
    dispatch=b"    if __import__('sys').argv[1:2]==['--provider-hlc-batch-risk-plan-sha256']:return provider_hlc_batch_main()\n"
    need(last.count(dispatch)==1 and before==first+last.replace(dispatch,b''),'HLC_BATCH_OLD_BYTES_AST_UNCHANGED')
    return {'anchor':HLC_BATCH_ANCHOR,'old_owner_sha256':__import__('hashlib').sha256(before).hexdigest(),'current_owner_sha256':saved.native.sha(REPO/SELF),'old_functions_and_batch_risk_statistics_preserved':True,'old_main_only_change':'exact_first_token_HLC_batch_dispatch'}


def provider_hlc_batch_reference(binding):
    need(type(binding)is dict and set(binding)=={'path','sha256'}and type(binding['path'])is str,'HLC_BATCH_REFERENCE_SCHEMA')
    path=Path(binding['path'])
    need(path.is_absolute()and str(path)==binding['path']and '..'not in path.parts and path.is_file()and not path.is_symlink()and path.resolve()==path and ref(path)==binding,'HLC_BATCH_CANONICAL_REFERENCE')
    return path


def provider_hlc_batch_build_plan(*,source_base_commit=None):
    from gx1.audit import native_future_state_training_v1 as training
    from gx1.contracts.native_future_state_provider_hlc_v1 import SOURCE_BINDINGS,require_provider_hlc_state_contract,require_provider_hlc_certificate_ref
    d=read(HLC_BATCH_DESIGN)
    need(d['schema_version']=='gx1_provider_hlc_conditional_batch_risk_design_v1'and d['execution_authorized']is False and d['training_authorized']is False and d['test_authorized']is False and d['no_more_epochs']is True and d['model_parameters_changed']is False and d['execution']==EXECUTION and d['basename']==HLC_BATCH_NAME and d['stage']==HLC_BATCH_STAGE and d['plan_path']==str(HLC_BATCH_PLAN)and d['output_root']==str(HLC_BATCH_ROOT),'HLC_BATCH_LOCKED_DESIGN')
    inputs={**d['inputs'],'design':HLC_BATCH_DESIGN}
    for binding in inputs.values():provider_hlc_batch_reference(binding)
    result=read(inputs['completed_model_result']);terminal=read(inputs['completed_model_terminal']);completed=read(inputs['completed_model_plan']);original=read(inputs['original_epoch_plan']);quality=read(inputs['full_quality']);initial=read(inputs['initial_result'])
    report=result['report'];trajectory=report['trajectory'];meta=original['metadata'];schedule=original['schedule'];recipe=original['recipe']
    need(terminal['status']=='COMPLETE'and terminal['exit_code']==0 and terminal['result']==inputs['completed_model_result']and terminal['plan']==result['plan']==inputs['completed_model_plan']and terminal['relaunch_allowed']is False and result['source_inputs_policy_boot_unchanged']is True and result['test_data_used']is False and report['actual_optimizer_updates']==3694 and report['passes']==1 and report['full_TRAIN_rows']==ROWS and report['original_plan']==inputs['original_epoch_plan']and completed['metadata']['original']==original,'HLC_BATCH_COMPLETED_V2_EPOCH')
    need(trajectory['schedule']==schedule and trajectory['recipe']==recipe and trajectory['model_contract']==original['model_contract']and trajectory['target_contract']==meta['target_contract']and schedule['rows']==ROWS and schedule['batch_rows']==recipe['train_batch_rows']==96 and schedule['logical_batches']==schedule['batches_per_pass']==3694 and schedule['passes']==1 and schedule['shuffle']is False and schedule['drop_last']is False and schedule['skipped_batches']==0 and len(schedule['batch_issue_ns_sha256'])==len(schedule['update_flags'])==3694 and all(flag is True for flag in schedule['update_flags']),'HLC_BATCH_ORIGINAL_ISSUE_SCHEDULE')
    target=require_provider_hlc_state_contract(meta['target_contract']);data=read(inputs['dataset']);prior=objective.require_native_future_state_train_prior(read(inputs['prior']),target_contract=target)
    fields=training.provider_hlc_truth_fields(target);budget=d['budget'];certificate=require_provider_hlc_certificate_ref(data['target_observation_certificate_ref'])
    need(data['contract']==target and data['cutoff_ns']==UPPER and data['model_input']is False and data['arrays']==inputs['target_arrays']and meta['dataset']==inputs['dataset']and meta['prior']==inputs['prior']and data['TRAIN_issue_ns_sha256']==meta['TRAIN_issue_ns_sha256']==schedule['issue_ns_sha256']and data['base_dataset']==SOURCE_BINDINGS['base_dataset']and data['base_targets']==SOURCE_BINDINGS['base_targets']and data['native_manifest']==SOURCE_BINDINGS['native_manifest']and inputs['provider_response']==SOURCE_BINDINGS['reference_raw']and {key:certificate[key]for key in ('path','sha256')}==inputs['certificate'],'HLC_BATCH_V2_TARGET_CERTIFICATE_PRIOR')
    truth_sha=meta['truth_sha256'];final=report['whole_FINAL_score'];baseline=initial['report']['initial'];pop=prior['fit_population']
    need(pop['issue_rows']==ROWS and pop['issue_ns_sha256']==schedule['issue_ns_sha256']and pop['cutoff_ns']==UPPER and pop['truth_sha256']==truth_sha==final['truth_sha256']==baseline['truth_sha256']==quality['truth_sha256']and baseline['initial_prior_exact']is True and report['final_logits']==inputs['capture']and report['final_quality']==inputs['full_quality']and final['output_sha256']==quality['output_sha256']and quality['frozen_prior_contract_sha256']==prior['contract_sha256']and quality['rows']==ROWS,'HLC_BATCH_MATCHED_SAVED_IDENTITIES')
    capture_headers={'issue_ns':{'shape':[ROWS],'dtype':'<i8','sha256':schedule['issue_ns_sha256']},**{key:{'shape':[ROWS,5],'dtype':'<f4','sha256':final['output_sha256'][key]}for key in HLC_BATCH_CAPTURE[1:]}}
    headers={key:data['array_headers'][key]for key in fields};size=sum(int(np.prod(h['shape']))*np.dtype(h['dtype']).itemsize for h in [*capture_headers.values(),*headers.values()])
    need(tuple(budget['capture_fields'])==HLC_BATCH_CAPTURE and len(budget['target_fields'])==len(fields)and set(budget['target_fields'])==set(fields) and size==budget['NPZ_ndarray_bytes_decoded']==67719050 and budget['rows']==ROWS and budget['original_batches']==3694 and budget['batch_rows']==96 and budget['last_batch_rows']==22 and budget['capture_archive_loads']==budget['target_archive_loads']==1 and all(budget[key]==0 for key in ('models','checkpoint_loads','NN_forwards','backward_calls','autograd_calls','parameter_gradient_calculations','optimizer_constructions','optimizer_updates','fits','features','HTTP_calls','prior_refits','VAL_TEST_rows','new_labels')),'HLC_BATCH_EXACT_PAYLOAD_BUDGET')
    need(Path(inputs['provider_response']['path']).stat().st_size==budget['cached_provider_response_bytes']==614909 and Path(inputs['certificate']['path']).stat().st_size==budget['certificate_metadata_bytes']==3460 and budget['cached_provider_response_reads']==budget['cached_provider_response_JSON_parses']==budget['certificate_metadata_reads']==budget['certificate_reauthentications']==1 and budget['provider_candle_records']==budget['volume_integers_checked']==budget['source_timestamp_strings_parsed']==1290 and budget['MBA_OHLC_price_scalars_parsed']==15480 and budget['returned_certificate_arrays']=={'target_start_ns_bytes':10320,'validity_ns_bytes':10320,'mid_hlc_bytes':30960,'total_bytes':51600},'HLC_BATCH_EXACT_CACHED_AUTHENTICATION_BUDGET')
    for key in ('full_epoch_review','target_acceptance','previous_causal_design','previous_post_epoch_scope','qualification_requirements'):read(inputs[key])
    names=set(completed['source_owners'])|{SELF,TEST,saved.SELF,saved.TEST}
    need(all(not Path(name).is_absolute()and '..'not in Path(name).parts and Path(name).suffix in ('.py','.sh')for name in names),'HLC_BATCH_SOURCE_NAMES')
    for name,digest in completed['source_owners'].items():
        if name not in (SELF,TEST):need(saved.native.sha(REPO/name)==digest,'HLC_BATCH_UNCHANGED_COMPLETED_SOURCE')
    bridge=provider_hlc_batch_source_bridge()
    plan={'schema_version':'gx1_provider_hlc_conditional_batch_risk_plan_v1','stage':HLC_BATCH_STAGE,'status':'PREPARED_NOT_ADMITTED','plan_path':str(HLC_BATCH_PLAN),'output_root':str(HLC_BATCH_ROOT),'source_base_commit':source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'source_owners':{name:saved.native.sha(REPO/name)for name in sorted(names)},'source_bridge':bridge,'inputs':inputs,'execution':EXECUTION,'budget':budget,'capture_array_headers':capture_headers,'target_array_headers':headers,'target_archive_inventory':list(data['array_headers']),'target_contract':target,'truth_sha256':truth_sha,'prior_contract_sha256':prior['contract_sha256'],'TRAIN_issue_ns_sha256':schedule['issue_ns_sha256'],'schedule':schedule,'runtime_versions':{'numpy':np.__version__,'torch':str(torch.__version__)},'question':d['question'],'method':d['method'],'limits':d['limits'],'conditional_scope':d['new_conditional_scope_preconditions'],'expected_global_decomposition':d['expected_global_decomposition_from_existing_JSON']}
    provider_hlc_batch_checked_inputs(plan)
    return plan


def provider_hlc_batch_gate(binding,admitted):
    return {'status':'ADMITTED_EXACT_CONDITIONAL_PROVIDER_HLC_BATCH_ARITHMETIC'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}


def provider_hlc_batch_inspect(sha,*,admitted):
    need(Path.cwd().resolve()==REPO and type(sha)is str and len(sha)==64 and all(c in '0123456789abcdef'for c in sha),'HLC_BATCH_CWD_SHA')
    binding={'path':str(HLC_BATCH_PLAN),'sha256':sha};plan=read(binding)
    need(plan==provider_hlc_batch_build_plan(source_base_commit=plan['source_base_commit'])and source_clean(plan),'HLC_BATCH_EXACT_PLAN_SOURCE')
    pref=ref(POLICY);policy=read(pref)
    need(all(policy.get(key)is False for key in saved.FLAGS)and policy[HLC_BATCH_STAGE]==provider_hlc_batch_gate(binding,admitted),'HLC_BATCH_ONE_USE_ALL_TRAINING_CLOSED')
    need(not HLC_BATCH_ROOT.exists()and not HLC_BATCH_ROOT.is_symlink(),'HLC_BATCH_FRESH_ROOT')
    return plan,binding,pref


def provider_hlc_batch_checked_inputs(plan):
    for binding in plan['inputs'].values():provider_hlc_batch_reference(binding)
    saved.archive_headers(plan['inputs']['capture'],plan['capture_array_headers'],exact_inventory=HLC_BATCH_CAPTURE)
    saved.archive_headers(plan['inputs']['target_arrays'],plan['target_array_headers'],exact_inventory=plan['target_archive_inventory'])


def provider_hlc_batch_statistics(outputs,truth,*,batch_rows,prior_logits):
    """Fixed-output coordinate arithmetic; original batch denominators survive year splits."""
    issues=truth['issue_ns'];known=truth['activity_known'];side=truth['side_known'];up=truth['side_up'];active=truth['activity'];n=len(issues)
    need(issues.dtype==np.int64 and issues.shape==(n,)and np.all(np.diff(issues)>0)and active.dtype==np.bool_ and active.shape==known.shape and not np.any(active&~known),'HLC_BATCH_STATISTICS_CLOCK_ACTIVITY')
    need(set(outputs)=={'activity_logits','side_logits'}and outputs['activity_logits'].dtype==np.float32 and outputs['activity_logits'].shape==(n,5)and np.isfinite(outputs['activity_logits']).all()and prior_logits['activity_logits'].dtype==np.float32 and prior_logits['activity_logits'].shape==(5,)and np.isfinite(prior_logits['activity_logits']).all(),'HLC_BATCH_ACTIVITY_LOGITS')
    stats=batch_risk_statistics(outputs['side_logits'],known,side,up,batch_rows=batch_rows,prior_logits=prior_logits['side_logits'])
    years=issues.astype('datetime64[ns]').astype('datetime64[Y]').astype(str);names=sorted(set(years.tolist()));whole={name:{task:np.zeros(5,np.float64)for task in ('activity','side')}for name in ('FINAL','prior')};mean={name:{task:np.zeros(5,np.float64)for task in ('activity','side')}for name in whole}
    annual={year:{**{name+'_native_batch_loss_sum':{task:np.zeros(5,np.float64)for task in ('activity','side')}for name in whole},'rows':int((years==year).sum()),'batches_touched':0,'activity_known':np.zeros(5,np.int64),'side_known':np.zeros(5,np.int64),'UP':np.zeros(5,np.int64),'FINAL_side_residual_sums':np.zeros(5,np.float64),'prior_side_residual_sums':np.zeros(5,np.float64),'FINAL_native_batch_gradient_sum':np.zeros(5,np.float64),'prior_native_batch_gradient_sum':np.zeros(5,np.float64),**{name:{task:np.zeros(5,np.float64)for task in ('activity','side')}for name in whole}}for year in names};crossings=[]
    global_D=stats['activity_denominator_total'];updates=stats['updates']
    for batch in stats['batches']:
        left,right,D=batch['start_position'],batch['stop_position'],batch['D_activity_all_TF'];by=years[left:right];masks={'activity':known[left:right],'side':side[left:right]};labels={'activity':active[left:right],'side':up[left:right]};losses={}
        if len(set(by.tolist()))>1:crossings.append(batch['logical_step'])
        for name in whole:
            losses[name]={}
            for task in ('activity','side'):
                z=outputs[task+'_logits'][left:right].astype(np.float64)if name=='FINAL'else np.broadcast_to(prior_logits[task+'_logits'].astype(np.float64),(right-left,5))
                values=np.where(labels[task],np.logaddexp(0.,-z),np.logaddexp(0.,z));values=np.where(masks[task],values,0.);losses[name][task]=values;totals=values.sum(0)
                whole[name][task]+=totals
                if D:mean[name][task]+=totals/D
            batch[name+'_activity_BCE_sums']=losses[name]['activity'].sum(0).tolist();batch[name+'_side_BCE_sums']=losses[name]['side'].sum(0).tolist()
        q=np.exp(-np.logaddexp(0.,-outputs['side_logits'][left:right].astype(np.float64)));p=np.exp(-np.logaddexp(0.,-prior_logits['side_logits'].astype(np.float64)))
        for year in sorted(set(by.tolist())):
            take=by==year;cell=annual[year];cell['batches_touched']+=1;cell['activity_known']+=known[left:right][take].sum(0);cell['side_known']+=side[left:right][take].sum(0);cell['UP']+=(side[left:right][take]&up[left:right][take]).sum(0)
            for name,residual in (('FINAL',np.where(side[left:right],q-up[left:right],0.)),('prior',np.where(side[left:right],p-up[left:right],0.))):
                sums=residual[take].sum(0);cell[name+'_side_residual_sums']+=sums
                if D:cell[name+'_native_batch_gradient_sum']+=sums/D
                for task in ('activity','side'):
                    sums=losses[name][task][take].sum(0);cell[name][task]+=sums
                    if D:cell[name+'_native_batch_loss_sum'][task]+=sums/D
    losses={}
    for name in whole:
        losses[name]={'per_TF_BCE_sums':{task:whole[name][task].tolist()for task in whole[name]},'global_native':{task:float(whole[name][task].sum()/global_D)if global_D else None for task in whole[name]},'mean_supported_B96_native':{task:float(mean[name][task].sum()/updates)if updates else None for task in mean[name]}}
        for aggregation in ('global_native','mean_supported_B96_native'):
            losses[name][aggregation]['total']=sum(losses[name][aggregation][task]for task in ('activity','side'))if global_D else None
    for year,cell in annual.items():
        cell['DOWN']=(cell['side_known']-cell['UP']).tolist()
        for name in ('FINAL','prior'):
            cell[name+'_gradient_global_contribution']=(cell[name+'_side_residual_sums']/global_D).tolist()if global_D else [None]*5
            cell[name+'_gradient_mean_B96_contribution']=(cell[name+'_native_batch_gradient_sum']/updates).tolist()if updates else [None]*5
            weighted=cell.pop(name+'_native_batch_loss_sum');cell[name]={'BCE_sums':{task:cell[name][task].tolist()for task in ('activity','side')},'global_native_contribution':{task:float(cell[name][task].sum()/global_D)if global_D else None for task in ('activity','side')},'mean_supported_B96_contribution':{task:float(weighted[task].sum()/updates)if updates else None for task in ('activity','side')}}
        for key,value in list(cell.items()):
            if isinstance(value,np.ndarray):cell[key]=value.tolist()
    return {**stats,'loss_decomposition':losses,'issue_UTC_years':annual,'year_crossing_logical_batches':crossings,'year_aggregation':'Each year contributes rows within original full B96 batches; D_b remains the full original all-TF activity denominator. Contributions sum to global/all-batch quantities; no rebatched yearly objective.','existing_target_UTC_year_quality_is_separate':True}


def provider_hlc_batch_calculate(plan,*,progress=None):
    from gx1.audit import native_future_state_training_v1 as training
    torch.set_num_threads(1);need(torch.get_num_threads()==1,'HLC_BATCH_CPU_THREAD');progress={}if progress is None else progress;progress['phase']='CAPTURE3_CANONICAL_V2_TARGET14_CACHED_CERTIFICATE_REAUTH'
    capture=saved.load_fields(plan['inputs']['capture'],HLC_BATCH_CAPTURE,plan['capture_array_headers']);truth=training.load_provider_hlc_target_truth(plan['inputs']['dataset'],expected_issue_ns=capture['issue_ns'],target_contract=plan['target_contract'])
    fields=training.provider_hlc_truth_fields(plan['target_contract']);need(sum(capture[key].nbytes for key in HLC_BATCH_CAPTURE)+sum(truth[key].nbytes for key in fields)==plan['budget']['NPZ_ndarray_bytes_decoded'],'HLC_BATCH_ACTUAL_NPZ_BYTES')
    progress['capture_fields_decoded']=list(HLC_BATCH_CAPTURE);progress['target_fields_decoded']=list(fields);progress['certificate_reauthentications']=1
    need(objective.native_future_state_truth_sha256(truth,target_contract=plan['target_contract'])==plan['truth_sha256']and saved.array_sha(capture['issue_ns'])==plan['TRAIN_issue_ns_sha256'],'HLC_BATCH_ACTUAL_TRUTH_CLOCK')
    prior=objective.require_native_future_state_train_prior(read(plan['inputs']['prior']),target_contract=plan['target_contract']);logits=objective.native_future_state_train_prior_logits(prior,rows=1,target_contract=plan['target_contract']);prior_logits={key:value.numpy()[0]for key,value in logits.items()}
    stats=provider_hlc_batch_statistics({key:capture[key]for key in HEADS},truth,batch_rows=plan['budget']['batch_rows'],prior_logits=prior_logits)
    need(stats['updates']==3694 and stats['skipped_batches']==0 and len(stats['batches'])==3694 and stats['batches'][-1]['rows']==22 and set(stats['issue_UTC_years'])==set(STRATA[1:]),'HLC_BATCH_COMPLETE_SCHEDULE')
    for batch,digest,flag in zip(stats['batches'],plan['schedule']['batch_issue_ns_sha256'],plan['schedule']['update_flags']):need(saved.array_sha(capture['issue_ns'][batch['start_position']:batch['stop_position']])==digest and (not batch['skip_update'])is flag,'HLC_BATCH_ISSUE_BATCH_IDENTITY')
    result=read(plan['inputs']['completed_model_result']);initial=read(plan['inputs']['initial_result']);quality=read(plan['inputs']['full_quality']);final=result['report']['whole_FINAL_score'];baseline=initial['report']['initial']
    for j,tf in enumerate(TF):
        for task,mask,label in (('activity',truth['activity_known'][:,j],truth['activity'][:,j]),('side',truth['side_known'][:,j],truth['side_up'][:,j])):
            cell=prior['per_task'][task][tf];need(int(mask.sum())==cell['count']and saved.array_sha(mask)==cell['mask_sha256']and [int((mask&~label).sum()),int((mask&label).sum())]==cell['class_counts'],'HLC_BATCH_FROZEN_PRIOR_SUPPORT')
            for name,reference in (('FINAL',final),('prior',baseline)):
                actual=stats['loss_decomposition'][name]['per_TF_BCE_sums'][task][j];expected=reference['scores']['per_tf'][tf][task]
                need(expected['count']==int(mask.sum())and np.isclose(actual,expected['log_loss_sum'],rtol=1e-12,atol=1e-7),'HLC_BATCH_CANONICAL_BCE_SUM')
        need(np.isclose(stats['per_tf'][tf]['FINAL_mean_q'],quality['all_TF'][tf]['side']['model']['mean_positive_probability'],rtol=1e-12,atol=1e-12),'HLC_BATCH_QUALITY_Q')
    expected=plan['expected_global_decomposition']
    for name,suffix in (('FINAL','final'),('prior','initial')):
        global_loss=stats['loss_decomposition'][name]['global_native']
        for task in ('activity','side','total'):need(np.isclose(global_loss[task],expected[task+'_'+suffix],rtol=1e-12,atol=1e-12),'HLC_BATCH_RECORDED_GLOBAL_DECOMPOSITION')
    progress['phase']='COMPLETE_CONDITIONAL_BATCH_ARITHMETIC'
    return {**stats,'payload_counts':plan['budget'],'payload_count_semantics':'Canonical decoder/reauthentication operations; mandatory SHA streaming and header-only ZIP validation are separate and are not total physical file-open or I/O counts.','complete_issue_clock_and_original_B96_boundaries_verified':True,'physical_source_rows_or_training_step_history_reconstructed':False,'canonical_v2_truth_prior_and_recorded_scores_verified':True,'probability_semantics':'q=P(UP|activity), not joint P(UP).','task_denominator':'D_b=sum all five TF known-activity cells; side BCE only where side_known. Unknown side contributes exact zero coordinate derivative.','analytic_coordinate_derivatives_only':True,'parameter_gradient_calculations':0,'actual_training_cause':'UNKNOWN','selected_causal_fix':'UNKNOWN','later_generalization':'UNKNOWN','economic_edge':'UNKNOWN','all20_approved':False,'limits':plan['limits'],'conditional_scope':plan['conditional_scope']}


def provider_hlc_batch_publish(name,value):
    path=HLC_BATCH_ROOT/name;saved._publish_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());return ref(path)


def provider_hlc_batch_run(sha):
    import os,sys
    began=time.monotonic();cap=require_capped_cpu_audit_execution();lock=require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==4*1024**3 and cap['swap']==512*1024**2 and cap['numerical_threads']==1 and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):],'HLC_BATCH_CAPS')
    plan,binding,pref=provider_hlc_batch_inspect(sha,admitted=True);provider_hlc_batch_checked_inputs(plan);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();HLC_BATCH_ROOT.mkdir(parents=True,exist_ok=False)
    handlers={s:signal.getsignal(s)for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};claim=result=None;progress={'phase':'CLAIM'}
    try:
        def expired(*_):raise TimeoutError('HLC_BATCH_WALL')
        def stopped(s,*_):raise InterruptedError('HLC_BATCH_EXTERNAL_SIGNAL_'+str(s))
        signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
        claim=provider_hlc_batch_publish('CLAIM.json',{'stage':HLC_BATCH_STAGE,'plan':binding,'policy_at_claim':pref,'boot_id':boot,'remaining_invocations':0,'relaunch_allowed':False,'training_authorized':False,'test_data_used':False})
        remaining=EXECUTION['internal_seconds']-(time.monotonic()-began);need(remaining>0,'HLC_BATCH_TIME');signal.setitimer(signal.ITIMER_REAL,remaining)
        allowed={plan['inputs'][key]['path']for key in ('capture','target_arrays')}
        def audit(event,args):
            if event in ('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('HLC_BATCH_NETWORK_FORBIDDEN')
            if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
                name=os.fsdecode(args[0])
                if name.endswith(('.npz','.pt','.pth','.parquet','.safetensors','.pkl'))and str(Path(name).resolve())not in allowed:raise RuntimeError('HLC_BATCH_PAYLOAD_FORBIDDEN')
        sys.addaudithook(audit);report=provider_hlc_batch_calculate(plan,progress=progress);provider_hlc_batch_checked_inputs(plan)
        need(source_clean(plan)and ref(POLICY)==pref and ref(HLC_BATCH_PLAN)==binding and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,'HLC_BATCH_SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND')
        result=provider_hlc_batch_publish('RESULT.json',{'status':'COMPLETE_CONDITIONAL_PROVIDER_HLC_BATCH_ARITHMETIC','stage':HLC_BATCH_STAGE,'plan':binding,'claim':claim,'policy_at_claim':pref,'boot_id':boot,'report':report,'source_unchanged':True,'test_data_used':False,'economic_edge_proven':False,'elapsed_seconds':time.monotonic()-began})
        terminal=provider_hlc_batch_publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'stage':HLC_BATCH_STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False});return {'result':result,'terminal':terminal}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0);provider_hlc_batch_publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'stage':HLC_BATCH_STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'progress':progress,'error':type(error).__name__+':'+str(error)[:2000],'test_data_used':False});raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in handlers.items():signal.signal(sig,handler)


def provider_hlc_batch_main():
    import sys
    parser=argparse.ArgumentParser();parser.add_argument('--provider-hlc-batch-risk-plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    sha=args.provider_hlc_batch_risk_plan_sha256;need(sys.argv[1:]==['--provider-hlc-batch-risk-plan-sha256',sha]+(['--preflight']if args.preflight else[]),'HLC_BATCH_EXACT_CLI')
    if args.preflight:
        plan,binding,pref=provider_hlc_batch_inspect(sha,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_CONDITIONAL_PROVIDER_HLC_BATCH_PASS','plan':binding,'policy':pref,'claim_created':False,'array_payloads_decoded':0,'cached_provider_records_parsed':0,'models':0,'checkpoint_loads':0,'forwards':0,'fits':0}))
    else:print(json.dumps(provider_hlc_batch_run(sha),sort_keys=True))
# End provider-HLC conditional batch arithmetic.

def main():
    if __import__('sys').argv[1:2]==['--provider-hlc-batch-risk-plan-sha256']:return provider_hlc_batch_main()
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan-sha256');mode.add_argument('--batch-risk-plan-sha256');mode.add_argument('--projection-moment-plan-sha256');parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.projection_moment_plan_sha256 is not None:
        import sys
        need(sys.argv[1:]==['--projection-moment-plan-sha256',args.projection_moment_plan_sha256]+(['--preflight']if args.preflight else[]),'MOMENT_EXACT_CLI')
        if args.preflight:
            p,binding,pref=moment_inspect(args.projection_moment_plan_sha256,admitted=False);moment_checked_inputs(p)
            print(json.dumps({'status':'PREFLIGHT_PROJECTION_MOMENT_PASS','plan':binding,'policy':pref,'claim_created':False,'checkpoint_payloads':0,'array_payloads_decoded':0,'models':0,'forwards':0,'optimizer_updates':0}))
        else:print(json.dumps(moment_run(args.projection_moment_plan_sha256),sort_keys=True))
    elif args.batch_risk_plan_sha256 is not None:
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
