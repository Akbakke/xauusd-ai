"""Fixed saved-target bps/loss allocation diagnosis; no fit, model or raw data."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import types

R = Path('/home/andre2/src/GX1_CURRENT')
Q = Path('/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2')
META_SHA = 'a648c47880b6b195d1213741a0826d34c7e475bd75193313a4c26f7723798c7a'
HORIZONS = (1, 5, 15)
MINUTE = 60_000_000_000
START, END = 1714521600000000000, 1767225600000000000
BANDS = ('observed_exact_zero', '(0,1]', '(1,5]', '(5,10]', '(10,25]', '(25,50]', '(50,+infinity)')
FOLDS = ((1,47602,64890,1735689600000000000,1743465600000000000),
         (2,64890,82427,1743465600000000000,1751328000000000000),
         (3,82427,100499,1751328000000000000,1759276800000000000),
         (4,100499,118249,1759276800000000000,1767225600000000000))
np = None


def require(ok, reason):
    if not ok:
        raise RuntimeError('BPS_ALLOCATION_' + reason)


def band_ids(bps):
    require(np.isfinite(bps).all(), 'FINITE_BPS')
    ids = np.searchsorted(np.array([1.,5.,10.,25.,50.]), np.abs(bps), side='left') + 1
    return np.where(bps == 0, 0, ids)


def probability_bins(p):
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'PROBABILITY_DOMAIN')
    return np.minimum((10*p).astype(np.int64), 9)


def masks(t, cutoff):
    mature = t['terminal_time_ns'] < cutoff
    known = mature & t['observed']
    return mature, known, known & (t['sign'] != 0)


def validate_targets(t, dtypes, *, rows, lower_ns=START, cutoff_ns=END):
    require(set(t) == set(dtypes), 'PRICE_MEMBERS')
    for key, dtype in dtypes.items():
        shape = (rows,) if key in ('parent_rows','bar_start_ns','dynamic_entry_m1_row','decision_ns') else (rows,3)
        require(t[key].dtype == np.dtype(dtype) and t[key].shape == shape
                and np.isfinite(t[key]).all(), 'PRICE_HEADER:' + key)
    require(np.array_equal(t['parent_rows'], np.arange(rows))
            and (np.diff(t['bar_start_ns']) > 0).all()
            and ((t['bar_start_ns'] >= lower_ns) & (t['bar_start_ns'] < cutoff_ns)).all()
            and (t['bar_start_ns'] % (5*MINUTE) == 0).all()
            and np.array_equal(t['decision_ns'], t['bar_start_ns']+5*MINUTE)
            and (t['dynamic_entry_m1_row'] >= 0).all(), 'PRICE_CLOCK')
    require(np.array_equal(t['terminal_time_ns'], t['decision_ns'][:,None]+np.array(HORIZONS)*MINUTE)
            and np.array_equal(t['elapsed_wall_seconds'], np.broadcast_to(np.array(HORIZONS)*60,(rows,3))), 'TARGET_CLOCK')
    seen = t['observed']
    require((t['terminal_time_ns'][seen] < cutoff_ns).all()
            and (t['terminal_source_row'][seen] >= 0).all()
            and (t['terminal_source_row'][~seen] == -1).all(), 'SOURCE_ROW_MATURITY')
    require(np.isin(t['sign'], [-1,0,1]).all()
            and np.array_equal(t['sign'][seen], np.sign(t['mid_move'][seen]).astype(np.int8))
            and np.array_equal(t['sign'][seen], np.sign(t['mid_return_bps'][seen]).astype(np.int8)), 'BPS_MOVE_SIGN')
    require(all((t[k][~seen] == 0).all() for k in ('sign','mid_move','mid_return_bps')), 'MISSING_STORAGE')
    require((t['mid_return_bps'][seen] > -10000).all(), 'POSITIVE_PRICE_RETURN_BOUND')


def subset(t, indices):
    return {k: np.ascontiguousarray(v[indices]) for k, v in t.items()}


def join_targets(pred, target):
    rows = pred['parent_rows']
    require(rows.dtype == np.int64 and (np.diff(rows) > 0).all()
            and ((rows >= 0) & (rows < len(target['parent_rows']))).all(), 'JOIN_PARENT')
    selected = subset(target, rows)
    for k in ('parent_rows','bar_start_ns','decision_ns','terminal_time_ns','observed','sign'):
        require(np.array_equal(pred[k], selected[k]), 'TARGET_JOIN:' + k)
    return selected


def describe(values):
    if not len(values):
        return {'rows': 0, 'mean_signed_bps': None, 'mean_absolute_bps': None, 'quantiles_signed_bps_10_50_90': None}
    return {'rows': len(values), 'mean_signed_bps': float(values.mean()),
            'mean_absolute_bps': float(np.abs(values).mean()),
            'quantiles_signed_bps_10_50_90': np.quantile(values,[.1,.5,.9],method='linear').tolist()}


def distribution(t, cutoff):
    mature, known, eligible = masks(t, cutoff)
    bid = band_ids(t['mid_return_bps'])
    result = []
    for j,h in enumerate(HORIZONS):
        counts = {'parents': len(t['parent_rows']), 'immature': int((~mature[:,j]).sum()),
                  'mature_missing': int((mature[:,j] & ~t['observed'][:,j]).sum()),
                  'mature_zero': int((known[:,j] & (t['sign'][:,j] == 0)).sum()),
                  'eligible_nonzero': int(eligible[:,j].sum())}
        require(sum(counts[k] for k in ('immature','mature_missing','mature_zero','eligible_nonzero'))
                == counts['parents'], 'EXCLUSIVE_COUNT_PARTITION')
        bands = []
        for b,name in enumerate(BANDS):
            keep = known[:,j] & (bid[:,j] == b)
            bands.append({'band': name, **describe(t['mid_return_bps'][keep,j]),
                          'up_rows': int(np.sum(keep & (t['sign'][:,j] == 1))),
                          'down_rows': int(np.sum(keep & (t['sign'][:,j] == -1)))})
        require(sum(x['rows'] for x in bands) == int(known[:,j].sum()), 'BAND_PARTITION')
        result.append({'horizon_minutes': h, 'counts': counts, 'bands': bands,
            'small_nonzero_cell_share': bands[1]['rows']/counts['eligible_nonzero'] if counts['eligible_nonzero'] else None})
    return result


def rank_summary(y, scores, owner, kind):
    if not len(y) or not (y == 0).any() or not (y == 1).any():
        return {'auc': None, 'pairs': 0, 'twice_wins': 0, 'undefined_reason': 'empty_or_single_class'}
    a = owner.direction_rank_diagnostics(np.ascontiguousarray(y),np.ascontiguousarray(scores),score_kind=kind)
    return {'auc': a['auc'], 'pairs': a['up_down_pairs'],
            'twice_wins': 2*a['concordant_pairs']+a['tied_pairs'], 'undefined_reason': None}


def score_population(t, logits, cutoff, owner, *, role, fold_id, et_probability=None):
    _, known, eligible = masks(t, cutoff)
    bid = band_ids(t['mid_return_bps'])
    reports, losses = [], {}
    for arm,z in logits.items():
        require(z.shape == t['sign'].shape and z.dtype == np.float64 and np.isfinite(z).all(), 'LOGIT_ARRAY')
        losses[arm] = np.zeros_like(z)
    for j,h in enumerate(HORIZONS):
        use = eligible[:,j]
        y = np.ascontiguousarray((t['sign'][use,j] == 1).astype(np.int8))
        require(len(y) > 0 and (y == 0).any() and (y == 1).any(), 'FULL_COHORT_BOTH_CLASSES')
        arms = {}
        for arm,z in logits.items():
            p = et_probability[:,j] if arm == 'ET252' else owner.expit(z[:,j])
            score = p if arm == 'ET252' else z[:,j]
            kind = 'probability_up' if arm == 'ET252' else 'logit_up'
            m = owner.direction_probability_metrics(y,np.ascontiguousarray(z[use,j]))
            loss = m.pop('logloss_by_row')
            brier = m.pop('brier_by_row')
            losses[arm][use,j] = loss
            full_rank = rank_summary(y,score[use],owner,kind)
            bands = []
            for b,name in enumerate(BANDS):
                local = bid[use,j] == b
                n = int(local.sum())
                yy = np.ascontiguousarray(y[local])
                rank = rank_summary(yy,score[use][local],owner,kind)
                loss_sum = float(loss[local].sum())
                correct = int(np.sum((z[use,j][local] > 0) == (yy == 1)))
                bands.append({'band': name, 'eligible_rows': n, 'up_rows': int(yy.sum()),
                    'logloss_sum': loss_sum, 'brier_sum': float(brier[local].sum()), 'correct': correct,
                    'logloss': loss_sum/n if n else None,
                    'brier': float(brier[local].mean()) if n else None,
                    'accuracy': correct/n if n else None, 'cell_share': n/len(y),
                    'loss_share': loss_sum/float(loss.sum()), **rank})
            require(abs(sum(v['loss_share'] for v in bands)-1) < 1e-12
                    and sum(v['eligible_rows'] for v in bands) == len(y), 'LOSS_PARTITION')
            bins = probability_bins(p)
            probability_rows = []
            for b in range(10):
                selected = known[:,j] & (bins == b)
                binary = eligible[:,j] & (bins == b)
                probability_rows.append({'bin': b, 'lower': b/10, 'upper': (b+1)/10,
                    'last_upper_inclusive': b == 9, **describe(t['mid_return_bps'][selected,j]),
                    'mean_probability_up': float(p[selected].mean()) if selected.any() else None,
                    'binary_eligible_rows': int(binary.sum()),
                    'binary_up_fraction': float(np.mean(t['sign'][binary,j] == 1)) if binary.any() else None,
                    'observed_zero_rows': int(np.sum(selected & (t['sign'][:,j] == 0)))})
            require(sum(v['rows'] for v in probability_rows) == int(known[:,j].sum()), 'PROBABILITY_BIN_PARTITION')
            arms[arm] = {'full': {**m, **full_rank}, 'bands': bands, 'probability_bins': probability_rows}
        reports.append({'role': role, 'fold_id': fold_id, 'horizon_minutes': h,
                        'parents': len(t['parent_rows']), 'eligible_rows': int(use.sum()), 'arms': arms})
    return reports, losses


def frozen_batch_allocation(t, loss, presentation, cutoff, *, batch_size=16):
    n = len(t['parent_rows'])
    require(presentation.dtype == np.int64 and np.array_equal(np.sort(presentation),np.arange(n)), 'PRESENTATION')
    _,_,eligible = masks(t,cutoff)
    batch = presentation//batch_size
    count = np.bincount(batch,weights=eligible.sum(axis=1),minlength=(n+batch_size-1)//batch_size)
    require((count > 0).all(), 'NONEMPTY_BATCH')
    weight = eligible/count[batch,None]/len(count)
    require(abs(float(weight.sum())-1) < 1e-12, 'BATCH_COEFFICIENT_MASS')
    total = float((weight*loss).sum())
    require(total > 0, 'POSITIVE_FROZEN_LOSS')
    bid = band_ids(t['mid_return_bps'])
    rows = []
    for j,h in enumerate(HORIZONS):
        for b,name in enumerate(BANDS):
            selected = bid[:,j] == b
            mass = float(weight[selected,j].sum())
            contribution = float(np.sum(weight[selected,j]*loss[selected,j]))
            rows.append({'horizon_minutes': h, 'band': name, 'coefficient_mass': mass,
                         'loss_contribution': contribution, 'loss_share': contribution/total})
    require(abs(sum(x['coefficient_mass'] for x in rows)-1)<1e-12
            and abs(sum(x['loss_share'] for x in rows)-1)<1e-12, 'BATCH_ALLOCATION_PARTITION')
    last = batch == len(count)-1
    return {'batch_size': batch_size, 'batches': len(count), 'last_batch_parents': int(last.sum()),
        'last_batch_eligible_cells': int(eligible[last].sum()),
        'minimum_eligible_cells_per_batch': int(count.min()), 'maximum_eligible_cells_per_batch': int(count.max()),
        'last_batch_cell_coefficient_relative_to_full48cell_batch': float(48/count[-1]),
        'mean_frozen_batch_loss': total, 'global_cell_mean_frozen_loss': float(loss[eligible].mean()),
        'small_move_coefficient_mass': sum(x['coefficient_mass'] for x in rows if x['band'] == BANDS[1]),
        'small_move_loss_share': sum(x['loss_share'] for x in rows if x['band'] == BANDS[1]), 'allocation': rows,
        'scope': 'Frozen terminal EVAL logits with original batch reduction; not actual optimizer gradients or training-time loss history.'}


def aggregate_later(reports):
    aggregate = []
    for h in HORIZONS:
        members = [x for x in reports if x['role']=='later' and x['horizon_minutes']==h]
        require(len(members)==4, 'FOUR_DISJOINT_FOLDS')
        for b,name in enumerate(BANDS):
            arms = {}
            for arm in ('native','prior','ET252','R1'):
                parts = [x['arms'][arm]['bands'][b] for x in members]
                n = sum(x['eligible_rows'] for x in parts)
                pairs = sum(x['pairs'] for x in parts)
                twice = sum(x['twice_wins'] for x in parts)
                loss = sum(x['logloss_sum'] for x in parts)
                all_n = sum(x['eligible_rows'] for x in members)
                all_loss = sum(x['arms'][arm]['full']['logloss']*x['eligible_rows'] for x in members)
                arms[arm] = {'eligible_rows': n, 'up_rows': sum(x['up_rows'] for x in parts),
                    'within_fold_pairs': pairs, 'within_fold_pair_weighted_auc': twice/(2*pairs) if pairs else None,
                    'logloss': loss/n if n else None, 'logloss_sum': loss,
                    'brier': sum(x['brier_sum'] for x in parts)/n if n else None,
                    'accuracy': sum(x['correct'] for x in parts)/n if n else None,
                    'eligible_cell_share': n/all_n, 'loss_share': loss/all_loss,
                    'folds_with_defined_auc': sum(x['auc'] is not None for x in parts)}
            aggregate.append({'horizon_minutes':h,'band':name,'arms':arms})
    return aggregate


def synthetic(owner, dtypes):
    cases = []
    values = np.array([-50.001,-50.,-25.,-10.,-5.,-1.,-.001,0,.001,1,5,10,25,50,50.001])
    require(np.array_equal(band_ids(values),[6,5,4,3,2,1,1,0,1,1,2,3,4,5,6]), 'SYNTHETIC_EDGES')
    cases.append('all_signed_edges_and_exact_zero')
    require(np.array_equal(probability_bins(np.array([0.,.1,.5,.9,1.])),[0,1,5,9,9]), 'SYNTHETIC_PROB_BINS')
    cases.append('probability_edges_include_one')
    def reject(name, fn, reason):
        try: fn()
        except RuntimeError as exc:
            require(reason in str(exc),'WRONG_REJECTION');cases.append(name)
        else: raise RuntimeError('MISSING_EXPECTED_REJECTION:'+name)
    reject('nonfinite_bps_rejected',lambda:band_ids(np.array([float('nan')])), 'FINITE_BPS')
    reject('invalid_probability_rejected',lambda:probability_bins(np.array([1.1])), 'PROBABILITY_DOMAIN')
    n = 18
    bps = np.full((n,3),.5)
    bps[1::2] *= -1
    bps[-2:] *= 100
    sign = np.sign(bps).astype(np.int8)
    d = START+np.arange(n,dtype=np.int64)*5*MINUTE+5*MINUTE
    t = {'parent_rows':np.arange(n,dtype=np.int64),'decision_ns':d,
         'terminal_time_ns':d[:,None]+np.array(HORIZONS)*MINUTE,'observed':np.ones((n,3),bool),
         'sign':sign,'mid_return_bps':bps,'bar_start_ns':d-5*MINUTE,
         'dynamic_entry_m1_row':np.arange(n,dtype=np.int64)*5+5,
         'terminal_source_row':(np.arange(n,dtype=np.int64)*5+5)[:,None]+np.array(HORIZONS,np.int64),
         'elapsed_wall_seconds':np.broadcast_to(np.array(HORIZONS,np.int64)*60,(n,3)).copy(),
         'mid_move':bps*.2}
    validate_targets(t,dtypes,rows=n)
    cases.append('price_target_clock_sign_and_storage')
    damaged = {k:v.copy() for k,v in t.items()};damaged['mid_return_bps'][0,0]*=-1
    reject('bps_sign_inversion_rejected',lambda:validate_targets(damaged,dtypes,rows=n),'BPS_MOVE_SIGN')
    shifted = {k:v.copy() for k,v in t.items()};shifted['terminal_time_ns'][0,0]+=MINUTE
    reject('target_time_shift_rejected',lambda:validate_targets(shifted,dtypes,rows=n),'TARGET_CLOCK')
    missing = {k:v.copy() for k,v in t.items()};missing['observed'][0,0]=False;missing['terminal_source_row'][0,0]=-1
    reject('nonzero_missing_storage_rejected',lambda:validate_targets(missing,dtypes,rows=n),'MISSING_STORAGE')
    z = np.zeros((n,3))
    reports, losses = score_population(t,{'native':z},END,owner,role='fixture',fold_id=0)
    require(all(abs(x['arms']['native']['bands'][1]['cell_share']-16/18)<1e-15
                and x['arms']['native']['bands'][0]['eligible_rows']==0
                and x['arms']['native']['full']['auc']==.5 for x in reports),'SYNTHETIC_LOSS')
    cases.append('constant_logits_loss_partition_and_ties')
    result = frozen_batch_allocation(t,losses['native'],np.arange(n,dtype=np.int64),END)
    require(result['last_batch_parents']==2 and result['last_batch_eligible_cells']==6
            and result['last_batch_cell_coefficient_relative_to_full48cell_batch']==8
            and abs(result['small_move_coefficient_mass']-.5)<1e-15
            and abs(result['small_move_loss_share']-.5)<1e-15,'SYNTHETIC_BATCH_REDUCTION')
    cases.append('B16_then_B2_equal_batch_means_have_eightfold_cell_weight')
    bad = np.arange(n,dtype=np.int64);bad[1]=bad[0]
    reject('duplicate_presentation_rejected',lambda:frozen_batch_allocation(t,losses['native'],bad,END),'PRESENTATION')
    tc = {k:v.copy() for k,v in t.items()}
    tc['observed'][0,0]=False;tc['sign'][0,0]=0;tc['mid_return_bps'][0,0]=0
    tc['sign'][1,0]=0;tc['mid_return_bps'][1,0]=0
    dist = distribution(tc,int(d[2]+5*MINUTE))
    require(dist[0]['counts']['mature_missing']==1 and dist[0]['counts']['mature_zero']==1
            and dist[1]['counts']['immature']==n-2,'SYNTHETIC_MASKS')
    cases.append('missing_zero_and_cutoff_equality_distinct')
    require(rank_summary(np.ones(3,np.int8),np.zeros(3),owner,'logit_up')['auc'] is None,
            'SYNTHETIC_ONE_CLASS')
    cases.append('one_class_band_auc_explicitly_undefined')
    fixture = dict(t,bar_start_ns=d-5*MINUTE)
    joined = join_targets(fixture,fixture)
    require(np.array_equal(joined['mid_return_bps'],bps),'SYNTHETIC_TARGET_JOIN')
    bad=dict(fixture,sign=sign.copy());bad['sign'][0,0]*=-1
    reject('prediction_label_mismatch_rejected',lambda:join_targets(bad,fixture),'TARGET_JOIN:sign')
    copies=[dict(x,role='later',fold_id=i) for i in range(1,5) for x in reports]
    # Supply the same qualified constant-score fixtures for all named arms.
    for x in copies:
        x['arms']={k:x['arms']['native'] for k in ('native','prior','ET252','R1')}
    combined=aggregate_later(copies)
    require(next(x for x in combined if x['horizon_minutes']==1 and x['band']==BANDS[1])
            ['arms']['native']['within_fold_pair_weighted_auc']==.5,'SYNTHETIC_AGGREGATE')
    cases.append('within_fold_pair_aggregation_not_cross_fold_pooling')
    return {'cases':cases,'case_count':len(cases),'genuine_payloads':0,'fits':0}


def main():
    global np
    sys.dont_write_bytecode=True
    hp=Q/'inspect_direction_gc_parquet_metadata.py';raw=hp.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==META_SHA,'META_SHA')
    meta=types.ModuleType('bps_metadata');meta.__file__=str(hp)
    exec(compile(raw,str(hp),'exec'),meta.__dict__)
    pref={'path':sys.argv[1],'sha256':sys.argv[2]};plan=meta.read_json(pref)
    polref={'path':str(R/'NEXT_RUN_POLICY.json'),'sha256':sys.argv[3]}
    pol=meta.read_json(polref);head=sys.argv[4];mode=plan['mode'];st=pol['direction_bps_allocation']
    require(plan['schema_version']=='gx1_bps_allocation_execution_v1' and mode in ('synthetic','actual'),'PLAN_MODE')
    require(st['status']=='ADMITTED' and st['plan']==pref and st['mode']==mode
            and st['execution_authorized'] is True and st['saved_payload_reads_authorized'] is (mode=='actual')
            and st['max_invocations']==1 and st['claimed_invocations']==st['completed_invocations']==0
            and st['relaunch_allowed'] is False,'ONE_SHOT_SCOPE')
    require(all(pol[k] is False for k in ('training_enabled','native_training_enabled','native_launch_authorized',
             'full_epoch_training_allowed','full_val_allowed')) and meta.git_head()==head,'NATIVE_GATES_HEAD')
    require(plan['wall_seconds']==(120 if mode=='synthetic' else 180),'WALL')
    for rr in plan['sources'].values():meta.read_ref(rr)
    require(plan['sources']['worker']['path']==str(Path(__file__).resolve()),'EXACT_WORKER')
    admission=meta.read_json(plan['source_admission'])
    require(str(Path(__file__).relative_to(R)) in admission['allowed_repo_files'],'SOURCE_ADMISSION')
    design=meta.read_json(plan['design'])
    require(design['horizons_minutes']==list(HORIZONS) and design['absolute_bps_bands']==list(BANDS)
            and design['upper_edges']==[1,5,10,25,50],'FIXED_TARGET_BANDS')
    cap=meta.source_module('bps_cap',plan['sources']['cap'])
    proof,lock=cap.require_capped_cpu_audit_execution(),cap.require_capped_lock_ancestry()
    require(lock['scope']==proof['cgroup_path'][len('/sys/fs/cgroup'):],'LOCK_SCOPE')
    io=meta.source_module('bps_io',plan['sources']['io'])
    guard=meta.source_module('bps_guard',plan['sources']['read_guard'])
    meta.PUBLISH=meta.source_module('bps_publication',plan['sources']['publication'])
    out=Path(plan['out_dir']);require(out.parent==Q and out.parent.resolve(strict=True)==Q,'OUTPUT_ROOT');out.mkdir()
    claim=meta.publish(out,'INVOCATION_CLAIM.json',{'plan':pref,'policy':polref,'source_head':head,
                       'mode':mode,'execution':proof,'lock':lock})
    paths=plan['payload_paths']
    require(len(paths)==len(set(paths))==(9 if mode=='actual' else 0),'NINE_EXACT_PAYLOADS')
    sys.addaudithook(guard.make_guard(out,paths))
    def expired(*args):raise RuntimeError('BPS_ALLOCATION_WALL_LIMIT')
    signal.signal(signal.SIGALRM,expired);signal.alarm(plan['wall_seconds'])
    start,before,previous=time.monotonic(),io.io_state(),sys.getprofile()
    pins,fdpaths,counts={},set(),{}
    try:
        sys.path.insert(0,str(R))
        import numpy as numpy
        np=numpy;io.np=np
        from gx1.contracts import entry_direction_probability_reference_v1 as owner
        helpers=meta.source_module('bps_frozen_helpers',plan['sources']['frozen_helpers']);helpers.np=np
        typed_hash=helpers.typed_hash_owner(meta,plan['sources']['typed_hash'])
        roles={np.load.__code__:'payload_load',owner.direction_probability_metrics.__code__:'proper',
               owner.direction_rank_diagnostics.__code__:'rank'}
        maxima={'payload_load':9 if mode=='actual' else 0,'proper':60,'rank':400}
        counts={k+s:0 for k in maxima for s in ('_calls','_returns')}
        def watch(frame,event,result):
            io.observe_work(frame,event,result)
            role=roles.get(frame.f_code)
            if role and event in ('call','return'):
                if event=='call':
                    counts[role+'_calls']+=1;require(counts[role+'_calls']<=maxima[role],'CALL_CAP:'+role)
                    if role=='payload_load':require(frame.f_locals.get('file') in fdpaths
                        and frame.f_locals.get('allow_pickle') is False,'EXACT_PINNED_LOAD')
                elif result is not None:counts[role+'_returns']+=1
        sys.setprofile(watch)
        if mode=='synthetic':
            result=synthetic(owner, guard.TARGET_DTYPES)
        else:
            qualification=meta.read_json(plan['source_qualification'])
            require(qualification['status']=='PASS_SYNTHETIC_ONLY'
                    and qualification['worker']==plan['sources']['worker'],'QUALIFIED_SOURCE')
            def load(rr,keys):
                require(rr['path'] in paths,'DECLARED_INPUT')
                pin=io.pin_artifact(rr);pins[rr['path']]=pin;fdpaths.add('/proc/self/fd/'+str(pin['fd']))
                return io.read_archive(pin,set(keys),np)
            inputs=plan['inputs']
            tm=meta.read_json(inputs['target_metadata']);tq=meta.read_json(inputs['target_qualification'])
            require(tq['data_qualified'] is True and tq['metadata']==inputs['target_metadata']
                    and tm['price_targets']==inputs['price_targets'] and tm['test_data_used'] is False,'QUALIFIED_TARGET')
            target=load(inputs['price_targets'],guard.TARGET_DTYPES)
            validate_targets(target,guard.TARGET_DTYPES,rows=118249)
            full_dist=distribution(target,END)
            for j,h in enumerate(HORIZONS):
                old=tm['target_report']['label_counts'][j];new=full_dist[j]
                require(old['horizon_minutes']==h and old['observed_unchanged']==new['counts']['mature_zero']
                        and old['missing']==new['counts']['mature_missing']+new['counts']['immature']
                        and old['observed_up']==sum(x['up_rows'] for x in new['bands'])
                        and old['observed_down']==sum(x['down_rows'] for x in new['bands']),'ORIGINAL_LABEL_COUNTS')
            reports=[];prefix_distributions=[];original_reports=[meta.read_json(x['report']) for x in inputs['native']]
            contexts=original_reports[0]['context_map']['contexts']
            keys={'parent_rows','bar_start_ns','decision_ns','terminal_time_ns','observed','sign','logit_up',
                  *('context__'+x['name'] for x in contexts)}
            fp=meta.read_json(inputs['frozen_plan'])
            pieces=[load(x['payload'],keys|{'hidden','presentation_index','head_weight','head_bias'})
                    for x in fp['units'][:3]]
            protocol=meta.read_json(fp['scientific_subquestion'])
            prefix=helpers.assemble_prefix(pieces,contexts,rows=47602,cutoff_ns=FOLDS[0][3],lower_ns=START,
                order_sha=protocol['scope']['original_prefix_presentation_sha256'],typed_hash=typed_hash)
            del pieces
            pt=join_targets(prefix,target)
            prior=np.asarray(original_reports[0]['own_prefix_prior']['logit_up'],np.float64)
            pr,loss=score_population(pt,{'native':prefix['logit_up'].astype(np.float64),
                'prior':np.broadcast_to(prior,(47602,3)).copy()},FOLDS[0][3],owner,role='prefix',fold_id=1)
            old_prefix=meta.read_json(inputs['frozen_root_review'])
            for j,x in enumerate(pr):
                for arm in ('native','prior'):
                    for metric in ('logloss','brier','accuracy','balanced_accuracy','auc'):
                        require(abs(x['arms'][arm]['full'][metric]-old_prefix['comparisons'][j]['prefix'][arm][metric])<1e-14,
                                'ORIGINAL_PREFIX_SCORE:'+metric)
            batch=frozen_batch_allocation(pt,loss['native'],prefix['presentation_index'],FOLDS[0][3])
            reports.extend(pr);del prefix,pt,loss
            cr=meta.read_json(inputs['controls_report']);ct=meta.read_json(inputs['controls_terminal'])
            require(cr['OOF_predictions']==inputs['controls'] and ct['status']=='COMPLETE'
                    and ct['result']==inputs['controls_report'],'QUALIFIED_CONTROLS')
            c=load(inputs['controls'],cr['OOF_array_headers'])
            for k,spec in cr['OOF_array_headers'].items():
                require(c[k].dtype==np.dtype(spec['dtype']) and list(c[k].shape)==spec['shape'],'CONTROL_HEADER:'+k)
            require(np.array_equal(c['parent_rows'],np.arange(29849,118249)),'CONTROL_PARENTS')
            for item,old,(fold,start_row,end_row,cutoff,end) in zip(inputs['native'],original_reports,FOLDS):
                f=old['fold']
                require(item['fold_id']==f['fold_id']==fold and f['prefix_rows']==start_row
                        and f['oof_start_row']==start_row and f['oof_end_row']==end_row
                        and f['train_cutoff_ns']==cutoff and f['oof_cutoff_ns']==end
                        and old['predictions']=={k:item['predictions'][k] for k in ('path','sha256')},'ORIGINAL_FOLD')
                pd=distribution(subset(target,np.arange(start_row)),cutoff)
                prefix_distributions.append({'fold_id':fold,'prefix_rows':start_row,'horizons':pd})
                prior=np.asarray(old['own_prefix_prior']['logit_up'],np.float64)
                for j in range(3):
                    up=sum(b['up_rows'] for b in pd[j]['bands']);down=sum(b['down_rows'] for b in pd[j]['bands'])
                    require(up==old['own_prefix_prior']['eligible_up'][j] and down==old['own_prefix_prior']['eligible_down'][j]
                            and abs(float(np.log(up/down))-prior[j])<1e-15,'PREFIX_PRIOR')
                a=load(item['predictions'],keys)
                require(np.array_equal(a['parent_rows'],np.arange(start_row,end_row)),'NATIVE_PARENTS')
                tt=join_targets(a,target);index=np.searchsorted(c['parent_rows'],a['parent_rows'])
                for k in ('parent_rows','decision_ns','terminal_time_ns','sign','observed'):
                    require(np.array_equal(a[k],c[k][index]),'CONTROL_JOIN:'+k)
                require((c['fold_id'][index]==fold).all() and (c['holdout_end_ns_exclusive'][index]==end).all()
                        and np.array_equal(c['binary_mature'][index],masks(tt,end)[2]),'CONTROL_ELIGIBILITY')
                pp=np.ascontiguousarray(c['saved_ET_FULL_probability_up'][index])
                require(((pp>0)&(pp<1)).all()
                        and np.array_equal((pp>.5).astype(np.int8),c['saved_ET_FULL_direction_class'][index]),'ET_PROBABILITY')
                logits={'native':a['logit_up'].astype(np.float64),'prior':np.broadcast_to(prior,(end_row-start_row,3)).copy(),
                        'R1':np.ascontiguousarray(c['saved_R1_logit_up'][index]),
                        'ET252':np.log(pp)-np.log1p(-pp)}
                rows,_=score_population(tt,logits,end,owner,role='later',fold_id=fold,et_probability=pp)
                for j,x in enumerate(rows):
                    for arm,field in (('native','model_metrics'),('prior','prior_metrics')):
                        original=old[field]['per_horizon'][j]
                        require(original['counts']['eligible_cells']==x['eligible_rows'],'ORIGINAL_LATER_COUNT')
                        for metric in ('logloss','brier','accuracy','balanced_accuracy'):
                            require(abs(x['arms'][arm]['full'][metric]-original[metric])<1e-14,'ORIGINAL_LATER_SCORE:'+metric)
                reports.extend(rows)
            require(counts['payload_load_calls']==counts['payload_load_returns']==9
                    and counts['proper_calls']==counts['proper_returns']==54,'ACTUAL_EXACT_LOAD_PROPER_COUNTS')
            result={'full_TRAIN_label_distribution':full_dist,'prefix_label_distributions':prefix_distributions,
                'prediction_reports':reports,'aggregate_later_by_magnitude':aggregate_later(reports),
                'frozen_prefix_batch_allocation':batch,'original_unstratified_scores_reproduced':True,
                'model_forwards':0,'fits':0,'economic_edge_qualified':False,
                'limits':[design['small_move_premise']['interpretation'],design['batch_reduction']['limit'],
                          design['uncertainty'],'Outcome-magnitude bands are retrospective diagnostics, never executable filters.',
                          'Midpoint bps are not realized PnL and do not include spread/fees/slippage.',
                          'Prefix populations overlap across separate models; do not sum as independent observations.']}
        require(not any(io.FORBIDDEN.values()),'NO_FORBIDDEN_WORK')
        io.pinned_bookends(pins);used=io.io_state()['rchar']-before['rchar'];require(used<=plan['max_rchar_bytes'],'IO_CAP')
        sys.setprofile(previous)
        for path,(identity,digest) in list(meta.PINNED.items()):
            require(meta.regular(path)==identity,'META_BOOKEND');meta.read_ref({'path':path,'sha256':digest})
        require(meta.git_head()==head,'HEAD_BOOKEND')
        data={'status':'COMPLETE','mode':mode,'plan':pref,'source_head':head,'test_data_used':False,**result}
        require(len(meta.json_bytes(data))<=8*1024*1024,'SUMMARY_SIZE')
        summary=meta.publish(out,'SUMMARY.json',data)
        terminal={'status':'COMPLETE','exit_code':0,'plan':pref,'policy':polref,'claim':claim,'summary':summary,
            'source_head':head,'source_unchanged':True,'inputs_unchanged':True,
            'counters':{**counts,**io.FORBIDDEN},'execution':proof,'lock':lock,'rchar_bytes':used,
            'elapsed_seconds':time.monotonic()-start,'test_data_used':False}
    except Exception as exc:
        sys.setprofile(previous)
        meta.publish(out,'TERMINAL.json',{'status':'FAILED','exit_code':1,'plan':pref,'policy':polref,'claim':claim,
            'source_head':head,'error':type(exc).__name__+': '+str(exc),'counters':{**counts,**io.FORBIDDEN},
            'elapsed_seconds':time.monotonic()-start,'test_data_used':False})
        raise
    finally:
        sys.setprofile(previous)
        for pin in pins.values():os.close(pin['fd'])
    answer=meta.publish(out,'TERMINAL.json',terminal);signal.alarm(0);print(json.dumps(answer))


if __name__=='__main__':
    main()
