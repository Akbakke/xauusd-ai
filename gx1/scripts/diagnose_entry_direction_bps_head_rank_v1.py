"""Exploratory ranking of saved signed-bps predictions; no model execution."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import types

REPO = Path('/home/andre2/src/GX1_CURRENT')
PROGRAM = Path('/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2')
HELPER_SHA = 'a648c47880b6b195d1213741a0826d34c7e475bd75193313a4c26f7723798c7a'
DESIGN_SHA = '11c3f8f115225a950bf6a49043040c64b6147af7b81cfe4f756edf0d6469c32a'
STAGE = 'direction_bps_head_rank_diagnostic_20261027'
DAY = 86400000000000
MINUTE = 60000000000
PREFIX_END = 1735689600000000000
LATER_END = 1743465600000000000
REPLICATES, ALPHA, SEED = 4000, .01, 20261027
BASE = {'parent_rows','bar_start_ns','decision_ns','terminal_time_ns','observed','sign','logit_up'}
np = None


def require(ok, reason):
    if not ok:
        raise RuntimeError('BPS_HEAD_RANK_' + reason)


def verify_pair(candidate, baseline, *, rows, enforce_scope=True):
    require(len(candidate['parent_rows']) == len(baseline['parent_rows']) == rows,
            'PARENT_COUNT')
    for key in baseline:
        if key != 'logit_up':
            require(key in candidate and candidate[key].dtype == baseline[key].dtype
                    and candidate[key].shape == baseline[key].shape
                    and candidate[key].tobytes() == baseline[key].tobytes(),
                    'IDENTITY:' + key)
    require(np.isfinite(candidate['mean_return_bps']).all()
            and np.isfinite(candidate['mid_return_bps']).all()
            and np.array_equal(np.sign(candidate['mid_return_bps']).astype(np.int8),
                               candidate['sign']), 'SIGNED_BPS_AND_LABEL')
    require(np.all(candidate['mid_return_bps'][~candidate['observed']] == 0), 'MISSING_BPS')
    if enforce_scope:
        require(rows == 17288 and np.array_equal(candidate['parent_rows'], np.arange(47602,64890))
                and ((candidate['decision_ns'] >= PREFIX_END)
                     & (candidate['decision_ns'] < LATER_END)).all()
                and np.array_equal(candidate['terminal_time_ns'],
                     candidate['decision_ns'][:,None]+np.asarray([1,5,15],np.int64)[None,:]*MINUTE),
                'LATER_CLOCK')
    return True


def eligible(arrays, j):
    return (arrays['observed'][:,j] & (arrays['sign'][:,j] != 0)
            & (arrays['terminal_time_ns'][:,j] < LATER_END))


def score_vectors(candidate, baseline, j, use):
    return {
        'bps_mean':np.ascontiguousarray(candidate['mean_return_bps'][use,j],dtype=np.float64),
        'candidate_logit':np.ascontiguousarray(candidate['logit_up'][use,j],dtype=np.float64),
        'native276_logit':np.ascontiguousarray(baseline['logit_up'][use,j],dtype=np.float64)}


def direction_labels(arrays, j):
    use=eligible(arrays,j)
    labels=np.ascontiguousarray((arrays['sign'][use,j] == 1).astype(np.int8))
    require(len(labels)>0 and (labels==0).any() and (labels==1).any(), 'BOTH_CLASSES')
    return use,labels


def run_scoring(candidate, baseline, io, rank, paired):
    days = candidate['decision_ns']//DAY
    schemes = []
    for sid, name in enumerate(('UTC_DAY','UTC_WEEK_MONDAY')):
        labels = days if sid == 0 else (days+3)//7
        blocks, inverse = np.unique(labels, return_inverse=True)
        require(len(blocks) >= 8, 'MINIMUM_BLOCKS')
        weights = io.block_weights(int(len(blocks)),replicates=REPLICATES,
                                   seed=np.random.SeedSequence([SEED,sid]))
        schemes.append((name,blocks,inverse,weights))
    results = []
    for j,h in enumerate((1,5,15)):
        use,labels = direction_labels(candidate,j)
        scores = score_vectors(candidate,baseline,j,use)
        points, intervals = {}, []
        for name,blocks,inverse,weights in schemes:
            draws, pairs = {}, None
            for key,z in scores.items():
                values, current = paired.auc_draws(labels,z,inverse[use],weights,rank)
                if pairs is None: pairs=current
                else: require(np.array_equal(pairs,current),'PAIRED_DENOMINATORS')
                draws[key]=values
                single,_=paired.auc_draws(labels,z,inverse[use],
                                         np.ones((1,len(blocks)),np.float64),rank)
                if key in points:
                    require(abs(points[key]-float(single[0]))<1e-14,'SCHEME_POINT_PARITY')
                else: points[key]=float(single[0])
            interval = {'scheme':name,'blocks':len(blocks),
                        'weights_sha256':io.digest(weights,np)}
            for ref in ('native276_logit','candidate_logit'):
                interval['bps_minus_'+ref+'_auc_99_descriptive']=paired.interval(
                    draws['bps_mean']-draws[ref],point=points['bps_mean']-points[ref],
                    alpha=ALPHA,replicates=REPLICATES)
            interval['bps_auc_99_descriptive']=paired.interval(
                draws['bps_mean'],point=points['bps_mean'],alpha=ALPHA,replicates=REPLICATES)
            intervals.append(interval)
        if h == 5:
            require(len(labels)==17197
                    and abs(points['candidate_logit']-.49682269234877807)<1e-12
                    and abs(points['native276_logit']-.5030210874582608)<1e-12,
                    'PRIOR_5M_AUC_PARITY')
        results.append({'horizon_minutes':h,'eligible_nonzero_rows':len(labels),
                        'up_rows':int(labels.sum()),'down_rows':int((labels==0).sum()),
                        'points':points,'intervals':intervals})
    return results


def synthetic(rank,paired,io):
    cases=[]
    y=np.array([0,1,0,1],np.int8)
    ids=np.array([0,0,1,1],np.int64)
    weights=np.array([[1.,1.],[2.,0.]],np.float64)
    a=np.array([0.,0.,-.1,.3],np.float64)
    b=-a
    got,pairs=paired.auc_draws(y,a,ids,weights,rank)
    for i in range(len(weights)):
        ind=np.repeat(np.arange(len(y)),weights[i,ids].astype(int))
        pos=a[ind][y[ind]==1];neg=a[ind][y[ind]==0]
        want=np.mean((pos[:,None]>neg[None,:]).astype(float)+.5*(pos[:,None]==neg[None,:]))
        require(got[i]==want and pairs[i]==len(pos)*len(neg),'SYNTHETIC_TIES_BLOCKS')
    cases.append('paired_block_auc_matches_explicit_duplication_with_ties')
    selection_candidate={'mean_return_bps':np.tile(a[:,None],(1,3)),
                         'logit_up':np.tile(b[:,None],(1,3))}
    selection_baseline={'logit_up':np.tile(b[:,None],(1,3))}
    selected=score_vectors(selection_candidate,selection_baseline,1,np.ones(4,np.bool_))
    ga,_=paired.auc_draws(y,selected['bps_mean'],ids,weights[:1],rank)
    gb,_=paired.auc_draws(y,selected['candidate_logit'],ids,weights[:1],rank)
    require(ga[0]>gb[0] and np.array_equal(selected['native276_logit'],b),
            'SYNTHETIC_OUTPUT_SELECTION')
    cases.append('bps_and_logit_scores_are_not_interchanged')
    signs=np.array([[1,1,1],[-1,0,1],[0,1,-1],[1,1,-1],[0,1,-1]],np.int8)
    observed=np.ones((5,3),np.bool_);observed[2,0]=False
    terminal=np.full((5,3),LATER_END-1,np.int64);terminal[3,0]=LATER_END
    candidate={'parent_rows':np.arange(5,dtype=np.int64),'sign':signs,'observed':observed,
               'terminal_time_ns':terminal,'logit_up':np.zeros((5,3),np.float32),
               'mid_return_bps':signs.astype(np.float64),
               'mean_return_bps':np.zeros((5,3),np.float32)}
    baseline={k:v.copy() for k,v in candidate.items() if k not in ('mid_return_bps','mean_return_bps')}
    require(verify_pair(candidate,baseline,rows=5,enforce_scope=False),'SYNTHETIC_MATCH')
    require(np.array_equal(eligible(candidate,0),np.array([True,True,False,False,False])),
            'SYNTHETIC_MASK')
    cases.append('observed_nonzero_mature_mask')
    baseline['parent_rows'][0]=99
    try:verify_pair(candidate,baseline,rows=5,enforce_scope=False)
    except RuntimeError as exc:require('IDENTITY:parent_rows' in str(exc),'SYNTHETIC_EXPECTED_MISMATCH')
    else:raise RuntimeError('BPS_HEAD_RANK_SYNTHETIC_ACCEPTED_MISMATCH')
    cases.append('mismatched_parents_rejected')
    baseline['parent_rows'][0]=0
    baseline['sign'][1,0]=1
    try:verify_pair(candidate,baseline,rows=5,enforce_scope=False)
    except RuntimeError as exc:require('IDENTITY:sign' in str(exc),'SYNTHETIC_EXPECTED_LABEL_MISMATCH')
    else:raise RuntimeError('BPS_HEAD_RANK_SYNTHETIC_ACCEPTED_LABEL_MISMATCH')
    cases.append('mismatched_direction_label_rejected')
    candidate['sign'][:,0]=1;candidate['observed'][:,0]=True;candidate['terminal_time_ns'][:,0]=LATER_END-1
    try:direction_labels(candidate,0)
    except RuntimeError as exc:require(str(exc).endswith('BOTH_CLASSES'),'SYNTHETIC_EXPECTED_SINGLE_CLASS')
    else:raise RuntimeError('BPS_HEAD_RANK_SYNTHETIC_ACCEPTED_SINGLE_CLASS')
    cases.append('single_class_direction_population_rejected')
    return {'status':'PASS_SYNTHETIC_ONLY','passed':len(cases),'cases':cases,'saved_payload_reads':0}


def run_actual(meta,io,guard,comparison,rank,paired,plan,paths,pins,fdpaths):
    inputs=plan['inputs']
    reports={}
    for name in ('candidate','baseline'):
        item=inputs[name]
        report=meta.read_json(item['report'])
        require(report['role']=='OOF' and report['test_data_used'] is False
                and report['predictions']=={k:item['payload'][k] for k in ('path','sha256')},
                'REPORT_BINDING:'+name)
        reports[name]=report
    contexts=reports['candidate']['context_map']['contexts']
    require(contexts==reports['baseline']['context_map']['contexts'],'CONTEXT_IDENTITY')
    names=BASE|{'context__'+c['name'] for c in contexts}
    def load(ref,keys):
        require(ref['path'] in paths,'DECLARED_PAYLOAD')
        pin=io.pin_artifact(ref);pins[ref['path']]=pin
        fdpaths.add('/proc/self/fd/'+str(pin['fd']))
        return io.read_archive(pin,set(keys),np)
    c=load(inputs['candidate']['payload'],names|{'mid_return_bps','mean_return_bps'})
    b=load(inputs['baseline']['payload'],names)
    require(np.array_equal(c['parent_rows'],b['parent_rows']),'PRESENTATION_IDENTITY')
    c=comparison.canonical_panel(c,contexts,rows=17288,signed_bps=True)
    b=comparison.canonical_panel(b,contexts,rows=17288,signed_bps=False)
    verify_pair(c,b,rows=17288)
    return {'scope':'Reused first TRAIN-development split; no model fitting, forward, checkpoint or raw-data access.',
            'horizons':run_scoring(c,b,io,rank,paired),
            'limitations':['Bps output is an expected signed return, not a calibrated UP probability.',
              'AUC is ranking, not a trading result or path-dependent MFE/MAE.',
              'Intervals are descriptive on reused development data; no independent generalization or economic edge.',
              'Previously fixed advancement criteria remain unchanged.'],
            'model_forwards':0,'fits':0,'optimizer_steps':0,'checkpoint_reads':0,'test_data_used':False}


def main():
    global np
    sys.dont_write_bytecode=True
    helper=PROGRAM/'inspect_direction_gc_parquet_metadata.py'
    raw=helper.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==HELPER_SHA,'METADATA_HELPER')
    meta=types.ModuleType('bps_head_rank_metadata');meta.__file__=str(helper)
    exec(compile(raw,str(helper),'exec'),meta.__dict__)
    pref={'path':sys.argv[1],'sha256':sys.argv[2]}
    plan=meta.read_json(pref)
    require(plan['design']['sha256']==DESIGN_SHA,'DESIGN_SHA')
    design=meta.read_json(plan['design'])
    require(design['primary_horizon_minutes']==5 and design['secondary_horizons_minutes']==[1,15]
            and design['uncertainty']['replicates']==REPLICATES
            and design['uncertainty']['seed']==SEED and design['limits']['test_data_used'] is False,
            'FIXED_DESIGN')
    polref={'path':str(REPO/'NEXT_RUN_POLICY.json'),'sha256':sys.argv[3]}
    policy=meta.read_json(polref);head=sys.argv[4]
    mode=plan['mode'];stage=policy[STAGE]
    require(plan['schema_version']=='gx1_bps_head_rank_plan_v1' and mode in ('synthetic','actual'),'PLAN')
    require(stage['status']=='ADMITTED' and stage['plan']==pref and stage['mode']==mode
            and stage['execution_authorized'] is True and stage['max_invocations']==1
            and stage['claimed_invocations']==stage['completed_invocations']==0
            and stage['saved_payload_reads_authorized'] is (mode=='actual')
            and stage['relaunch_allowed'] is False,'ONE_SHOT')
    require(all(policy[k] is False for k in ('training_enabled','native_training_enabled',
            'native_launch_authorized','full_epoch_training_allowed','full_val_allowed')),'NATIVE_GATES_CLOSED')
    require(meta.git_head()==head and plan['wall_seconds']==120
            and plan['max_rchar_bytes']==128*1024**2,'HEAD_AND_LIMITS')
    if mode=='actual':
        fixed=design['existing_candidate_and_baseline_inputs']
        require(plan['inputs']=={'candidate':fixed['candidate_measurements'],
                                'baseline':fixed['baseline_measurements']},'DESIGN_INPUT_BINDING')
        require(set(plan['payload_paths'])=={fixed['candidate_measurements']['payload']['path'],
                                             fixed['baseline_measurements']['payload']['path']},
                'DESIGN_PATH_BINDING')
    for ref in plan['sources'].values():meta.read_ref(ref)
    require(plan['sources']['worker']['path']==str(Path(__file__).resolve()),'WORKER_PATH')
    admission=meta.read_json(plan['source_admission'])
    require(str(Path(__file__).relative_to(REPO)) in admission['allowed_repo_files'],'SOURCE_ADMISSION')
    cap=meta.source_module('bps_head_rank_cap',plan['sources']['cap'])
    proof,lock=cap.require_capped_cpu_audit_execution(),cap.require_capped_lock_ancestry()
    require(lock['scope']==proof['cgroup_path'][len('/sys/fs/cgroup'):],'LOCK_SCOPE')
    io=meta.source_module('bps_head_rank_io',plan['sources']['io'])
    guard=meta.source_module('bps_head_rank_guard',plan['sources']['read_guard'])
    meta.PUBLISH=meta.source_module('bps_head_rank_publication',plan['sources']['publication'])
    out=Path(plan['out_dir'])
    require(out.parent==PROGRAM and out.parent.resolve(strict=True)==PROGRAM,'OUTPUT_ROOT')
    out.mkdir()
    claim=meta.publish(out,'INVOCATION_CLAIM.json',{'plan':pref,'policy':polref,'source_head':head,
        'mode':mode,'execution':proof,'lock':lock})
    paths=plan['payload_paths']
    require(len(paths)==len(set(paths))==(2 if mode=='actual' else 0),'PAYLOAD_COUNT')
    sys.addaudithook(guard.make_guard(out,paths))
    def expired(*unused):raise RuntimeError('BPS_HEAD_RANK_WALL_LIMIT')
    signal.signal(signal.SIGALRM,expired);signal.alarm(plan['wall_seconds'])
    started,before,previous=time.monotonic(),io.io_state(),sys.getprofile()
    pins,fdpaths,counts={},set(),{'numpy_load_calls':0,'numpy_load_returns':0}
    try:
        sys.path.insert(0,str(REPO))
        import numpy as numpy
        np=numpy;io.np=np
        comparison=meta.source_module('bps_head_rank_comparison',plan['sources']['comparison'])
        comparison.np=np
        rank=meta.source_module('bps_head_rank_rank',plan['sources']['rank'])
        def watch(frame,event,value):
            io.observe_work(frame,event,value)
            if frame.f_code is np.load.__code__ and event in ('call','return'):
                if event=='call':
                    counts['numpy_load_calls']+=1
                    require(counts['numpy_load_calls']<=(2 if mode=='actual' else 0)
                            and frame.f_locals.get('file') in fdpaths
                            and frame.f_locals.get('allow_pickle') is False,'LOAD_CAP')
                elif value is not None:counts['numpy_load_returns']+=1
        sys.setprofile(watch)
        if mode=='synthetic':
            result=synthetic(rank,comparison,io)
        else:
            qualification=meta.read_json(plan['source_qualification'])
            require(qualification['status']=='PASS_SYNTHETIC_ONLY'
                    and qualification['worker']==plan['sources']['worker'],'QUALIFIED_WORKER')
            result=run_actual(meta,io,guard,comparison,rank,comparison,plan,paths,pins,fdpaths)
            require(counts['numpy_load_calls']==counts['numpy_load_returns']==2,'EXACT_LOAD_COUNT')
        require(not any(io.FORBIDDEN.values()),'NO_FORBIDDEN_WORK')
        require(counts['numpy_load_calls']==counts['numpy_load_returns'],'LOAD_RETURN_PARITY')
        io.pinned_bookends(pins)
        used=io.io_state()['rchar']-before['rchar'];require(used<=plan['max_rchar_bytes'],'IO_LIMIT')
        sys.setprofile(previous)
        for path,(identity,digest) in list(meta.PINNED.items()):
            require(meta.regular(path)==identity,'METADATA_BOOKEND')
            meta.read_ref({'path':path,'sha256':digest})
        require(meta.git_head()==head,'HEAD_BOOKEND')
        summary=meta.publish(out,'SUMMARY.json',{'execution_status':'COMPLETE','mode':mode,
            'plan':pref,'source_head':head,**result})
        terminal={'status':'COMPLETE','exit_code':0,'plan':pref,'policy':polref,'claim':claim,
            'summary':summary,'source_head':head,'source_unchanged':True,'inputs_unchanged':True,
            'counters':{**counts,**io.FORBIDDEN},'execution':proof,'lock':lock,
            'rchar_bytes':used,'elapsed_seconds':time.monotonic()-started,'test_data_used':False}
    except Exception as exc:
        sys.setprofile(previous)
        meta.publish(out,'TERMINAL.json',{'status':'FAILED','exit_code':1,'plan':pref,'policy':polref,
            'claim':claim,'source_head':head,'error':type(exc).__name__+': '+str(exc),
            'counters':{**counts,**io.FORBIDDEN},'elapsed_seconds':time.monotonic()-started,'test_data_used':False})
        raise
    finally:
        sys.setprofile(previous);signal.alarm(0)
        for pin in pins.values():os.close(pin['fd'])
    meta.publish(out,'TERMINAL.json',terminal)
    print(json.dumps({'status':'COMPLETE','mode':mode,'summary':summary,'elapsed_seconds':terminal['elapsed_seconds']}))


if __name__=='__main__':
    main()
