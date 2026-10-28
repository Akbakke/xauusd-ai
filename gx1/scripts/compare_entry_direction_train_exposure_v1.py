"""Fixed paired TRAIN scoring of saved one- and three-pass direction outputs.

The admitted worker reads four bound prediction archives only. It does not
construct a model, open a checkpoint or market data, fit or choose thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import numpy as np

from gx1.contracts import entry_direction_probability_reference_v1 as probability
from gx1.contracts import gx1_capped_execution_v1 as cap
from gx1.contracts import local_random_access_campaign_v2 as binding
from gx1.contracts.entry_native_direction_context_report_v1 import CONTEXT_SPEC
from gx1.scripts import compare_entry_direction_saved_bps_v1 as saved

REPO = Path('/home/andre2/src/GX1_CURRENT')
PROGRAM = Path('/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2')
ROOT = PROGRAM/'F_FROZEN_EXTRACTION_threepass_terminal_prefix_CAMPAIGN_20261028_001'
LEGACY = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/COMPARE_WHOLE_PREFIX_SIGN_003.py')
LEGACY_SHA = '3be6deb1185bd360b67329252b39277d495709f923268c4d1f20e6c55f197264'
STAGE = 'direction_threepass_train_paired_actual_20261028_001'
ROWS, CUTOFF = 47602, 1735689600000000000
HORIZONS = (1, 5, 15)
RANGES = ((0, 15872), (15872, 31744), (31744, 47602))
CORE = ('parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns', 'observed', 'sign', 'logit_up', 'presentation_index')
SOURCE_PATHS = (
    'gx1/scripts/compare_entry_direction_train_exposure_v1.py',
    'gx1/scripts/compare_entry_direction_saved_bps_v1.py',
    'gx1/contracts/entry_direction_probability_reference_v1.py',
    'gx1/contracts/entry_direction_block_rank_v1.py',
    'gx1/contracts/entry_native_direction_frozen_extraction_v1.py',
    'gx1/contracts/entry_native_direction_context_report_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
)


def require(ok, reason):
    if not ok:
        raise RuntimeError('TRAIN_EXPOSURE_' + reason)


def load_legacy():
    raw = LEGACY.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == LEGACY_SHA, 'LEGACY_SOURCE')
    module_spec = importlib.util.spec_from_loader('qualified_whole_prefix_sign', loader=None)
    module = importlib.util.module_from_spec(module_spec)
    exec(compile(raw, str(LEGACY), 'exec'), module.__dict__)
    require(module.CORE == CORE and module.H == HORIZONS, 'LEGACY_OWNER_SCOPE')
    return module


def pair_arrays(parts, three, *, legacy, ranges=RANGES, rows=ROWS):
    """Reassemble fixed one-pass partitions, then join every outcome/context."""
    one = legacy.assemble(parts, 0, rows, ranges)
    require(set(one) == set(three), 'FIELD_SETS')
    require(np.array_equal(three['parent_rows'], np.arange(rows, dtype=np.int64))
            and three['presentation_index'].dtype == np.int64
            and np.array_equal(np.sort(three['presentation_index']), np.arange(rows, dtype=np.int64)), 'THREE_PARENT_DOMAIN')
    contexts = tuple('context__'+name for name, _ in CONTEXT_SPEC)
    require(set(name for name in one if name.startswith('context__')) == set(contexts), 'CONTEXT_SCHEMA')
    legacy.same(one, three, (*saved.SHARED, 'presentation_index', *contexts))
    for arr in (one, three):
        saved.validate_prediction({k:v for k,v in arr.items() if k != 'presentation_index'},
            signed_bps=False, rows=rows, np=np, context_spec=CONTEXT_SPEC)
        require(np.all(np.diff(arr['decision_ns']) > 0), 'DECISION_ORDER')
    return one, three


def score_train(one, three, prior, *, cutoff=CUTOFF):
    """Fixed in-sample point scores; no resampling or model fitting."""
    require(set(prior) >= {'eligible_up','eligible_down','logit_up','probability_up'}, 'PRIOR_FIELDS')
    panels = []
    for j, horizon in enumerate(HORIZONS):
        mature = one['terminal_time_ns'][:, j] < cutoff
        observed = one['observed'][:, j]
        sign = one['sign'][:, j]
        mask = mature & observed & (sign != 0)
        counts = {'parents': len(sign), 'eligible': int(mask.sum()),
            'observed_mature_zero': int((mature & observed & (sign == 0)).sum()),
            'mature_missing': int((mature & ~observed).sum()), 'immature': int((~mature).sum())}
        require(sum(counts[k] for k in ('eligible','observed_mature_zero','mature_missing','immature')) == len(sign), 'COUNT_PARTITION')
        y = np.ascontiguousarray((sign[mask] == 1).astype(np.int8))
        require(len(y) > 0 and np.unique(y).size == 2, 'BOTH_CLASSES')
        require(int(y.sum()) == prior['eligible_up'][j] and int((1-y).sum()) == prior['eligible_down'][j], 'FIXED_PRIOR_COUNTS')
        p = float(prior['probability_up'][j]); logit = float(prior['logit_up'][j])
        require(np.isfinite(logit) and 0 < p < 1 and np.isclose(p, y.mean(), rtol=0, atol=1e-15)
                and np.isclose(logit, np.log(p/(1-p)), rtol=0, atol=1e-14), 'FIXED_PRIOR_DOMAIN')
        result = {}
        for name, z in (('one_pass',one['logit_up'][mask,j]), ('three_pass',three['logit_up'][mask,j]),
                        ('fixed_prior',np.full(len(y),logit,dtype=np.float64))):
            z = np.ascontiguousarray(z,dtype=np.float64)
            metrics = probability.direction_probability_metrics(y,z)
            rank = probability.direction_rank_diagnostics(y,z,score_kind='logit_up')
            result[name] = {'auc': rank['auc'], 'logloss': metrics['logloss'], 'brier': metrics['brier'],
                'up_rows': metrics['up_rows'], 'down_rows': metrics['down_rows']}
        panels.append({'horizon_minutes':horizon, 'counts':counts, 'arms':result,
            'three_minus_one':{key:result['three_pass'][key]-result['one_pass'][key] for key in ('auc','logloss','brier')},
            'three_minus_prior':{key:result['three_pass'][key]-result['fixed_prior'][key] for key in ('auc','logloss','brier')},
            'role':'primary_descriptive' if horizon == 5 else 'secondary_descriptive'})
    return {'panels':panels,'scope':'Fixed models scored on the same TRAIN observations used for fitting; no independent generalization inference.',
        'test_data_used':False,'economic_edge_qualified':False,'model_forwards':0,'optimizer_steps':0,'fits':0}


def _git_state():
    def git(*args):
        return subprocess.check_output(['git','-C',str(REPO),*args],text=True).strip()
    return git('rev-parse','HEAD'),git('branch','--show-current'),git('status','--porcelain=v1','--untracked-files=all')


def _read_bound(ref):
    require(set(ref) == {'path','sha256'} and Path(ref['path']).suffix in ('.json','.py'), 'METADATA_REF')
    binding.require_binding(ref,label='TRAIN exposure metadata')
    return binding.read_bound_json(Path(ref['path']),ref['sha256']) if ref['path'].endswith('.json') else None


def _install_guard(refs, allowed_json, output):
    allowed = {ref['path'] for ref in refs}; reads = {}; denied = []
    def guard(event,args):
        if event in ('socket.connect','socket.bind','socket.getaddrinfo','os.system','os.exec','os.posix_spawn'):
            raise RuntimeError('TRAIN_EXPOSURE_FORBIDDEN_PROCESS_OR_NETWORK')
        if event == 'subprocess.Popen':
            cmds = [['git','-C',str(REPO),'rev-parse','HEAD'], ['git','-C',str(REPO),'branch','--show-current'],
                    ['git','-C',str(REPO),'status','--porcelain=v1','--untracked-files=all']]
            require(list(args[1]) in cmds,'FORBIDDEN_SUBPROCESS')
        if event == 'import' and args[0].split('.')[0] in ('torch','jax','tensorflow','sklearn'):
            raise RuntimeError('TRAIN_EXPOSURE_FORBIDDEN_MODEL_IMPORT')
        if event != 'open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path = os.path.abspath(os.fsdecode(args[0])); flags = args[2] if len(args)>2 else 0
        write = isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if path.startswith('/home/andre2/GX1_DATA/'):
            raise RuntimeError('TRAIN_EXPOSURE_RAW_DATA')
        if Path(path).suffix.lower() in ('.pt','.pth','.npz','.npy','.parquet','.csv','.pkl') and path not in allowed:
            raise RuntimeError('TRAIN_EXPOSURE_UNBOUND_PAYLOAD')
        if path.startswith('/home/andre2/GX1_RUNS/'):
            if path in allowed:
                require(not write,'ORIGINAL_WRITE');reads[path]=reads.get(path,0)+1
            elif path in allowed_json:
                require(not write,'METADATA_WRITE')
            elif path.startswith(str(output)+'/'):
                require(Path(path).suffix not in ('.pt','.pth','.npz','.npy','.parquet','.csv','.pkl'),'PAYLOAD_OUTPUT')
            else:
                denied.append(path);raise RuntimeError('TRAIN_EXPOSURE_UNBOUND_OPEN')
    sys.addaudithook(guard)
    return reads,denied


def _read_archive(ref):
    path = Path(ref['path']); before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink() and 0 < before.st_size <= 64*1024*1024, 'ARCHIVE_FILE')
    identity = lambda x: (x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    with path.open('rb') as f:
        require(identity(os.fstat(f.fileno())) == identity(before), 'OPEN_IDENTITY')
        raw = f.read(64*1024*1024+1);after = os.fstat(f.fileno())
    require(identity(before) == identity(after) == identity(path.lstat()) and len(raw) == before.st_size
        and hashlib.sha256(raw).hexdigest() == ref['sha256'], 'ARCHIVE_SHA_AND_FREEZE')
    contexts = tuple('context__'+name for name,_ in CONTEXT_SPEC)
    names = set(CORE) | set(contexts) | {'hidden','head_weight','head_bias'}
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        require(len(archive.files)==len(names) and set(archive.files)==names, 'ARCHIVE_MEMBERS')
        return {name:np.ascontiguousarray(archive[name]) for name in (*CORE,*contexts)}


def _write_new(path,value):
    raw = (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    fd = os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
    with os.fdopen(fd,'wb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan','plan-sha256','policy-sha256','source-head'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args(argv);start=time.monotonic()
    scope=cap.require_capped_cpu_audit_execution();lock=cap.require_capped_lock_ancestry()
    require(lock['scope']==scope['cgroup_path'][len('/sys/fs/cgroup'):],'CAPPED_LOCK')
    git=(args.source_head,'work/gx1-current','');require(_git_state()==git,'CLEAN_SOURCE')
    plan_ref={'path':str(Path(args.plan)),'sha256':args.plan_sha256}
    policy_ref={'path':str(REPO/'NEXT_RUN_POLICY.json'),'sha256':args.policy_sha256}
    plan=binding.read_bound_json(Path(plan_ref['path']),plan_ref['sha256'])
    policy=binding.read_bound_json(Path(policy_ref['path']),policy_ref['sha256'])
    fields={'schema_version','stage','extraction_review','one_pass_protocol','one_pass_aggregate','inputs','sources','legacy','output_dir','protocol','test_data_used'}
    require(set(plan)==fields and plan['schema_version']=='gx1_train_exposure_paired_plan_v1'
        and plan['stage']==STAGE and plan['test_data_used'] is False,'PLAN')
    stage=policy[STAGE]
    require(stage['status']=='ADMITTED' and stage['plan']==plan_ref and stage['execution_authorized'] is True
        and stage['genuine_prediction_reads_authorized'] is True and stage['claimed_invocations']==stage['completed_invocations']==0
        and stage['max_invocations']==1 and stage['relaunch_allowed'] is False
        and policy['independent_direction_edge_20261027']['current_unit']==STAGE,'ADMISSION')
    require(all(policy.get(k) is False for k in ('training_enabled','native_training_enabled','full_epoch_training_allowed','full_val_allowed','native_launch_authorized'))
        and all(stage.get(k) is False for k in ('model_forwards_authorized','optimizer_steps_authorized','checkpoint_reads_authorized','test_data_used','broker_access_allowed')),'CLOSED_GATES')
    protocol={'train_parents':47602,'horizons_minutes':[1,5,15],'primary_horizon_minutes':5,
        'metrics':['auc','logloss','brier'],'references':['one_pass','fixed_prefix_prevalence'],
        'identity':'Exact parent, presentation index, decision and terminal clocks, labels, masks and context before filtering.',
        'eligibility':'observed AND sign != 0 AND terminal_time < fixed TRAIN cutoff per horizon',
        'intervals':'No inference intervals: fixed-model TRAIN scores are descriptive in-sample evidence.',
        'no_threshold_or_checkpoint_selection':True,'test_data_used':False}
    require(plan['protocol']==stage['protocol']==protocol,'PROTOCOL')
    require(set(plan['sources'])==set(SOURCE_PATHS),'SOURCE_SET')
    for name,ref in plan['sources'].items():
        require(ref['path']==str(REPO/name),'SOURCE_PATH');_read_bound(ref)
    require(plan['legacy']=={'path':str(LEGACY),'sha256':LEGACY_SHA},'LEGACY_BINDING');_read_bound(plan['legacy'])
    for key in ('extraction_review','one_pass_protocol','one_pass_aggregate'):_read_bound(plan[key])
    legacy=load_legacy()
    review=_read_bound(plan['extraction_review']);old_protocol=_read_bound(plan['one_pass_protocol']);old_aggregate=_read_bound(plan['one_pass_aggregate'])
    require(review['status']=='COMPLETE_FULL_TRAIN_EXTRACTION_PAIRED_SCORING_PENDING' and review['test_data_used'] is False
        and review['parent_rows']==ROWS and review['forwards']==2976 and review['optimizer_steps']==review['reverse_calls']==0
        and review['representations']==plan['inputs']['three'],'THREE_EXTRACTION_PROOF')
    parts=[u for u in old_protocol['units'] if u['unit'] in ('terminal_prefix_part1','terminal_prefix_part2','terminal_prefix_part3')]
    require([u['unit'] for u in parts]==['terminal_prefix_part1','terminal_prefix_part2','terminal_prefix_part3']
        and [u['representation'] for u in parts]==plan['inputs']['one_parts']
        and old_aggregate['train_parents']==ROWS and old_aggregate['initial_terminal_train_pair']=='BYTE_IDENTICAL_LABELS_AND_CONTEXTS'
        and old_aggregate['test_data_used'] is False,'ONE_EXTRACTION_PROOF')
    prior=review['result'];result_meta=_read_bound(prior);new_report=_read_bound(review['report'])
    require(result_meta['representations']==plan['inputs']['three'] and result_meta['own_prefix_prior']==new_report['own_prefix_prior']
        and result_meta['feature_view_identity']==parts[0]['feature_view_identity']
        and result_meta['fresh_identity_sha256']==parts[0]['fresh_identity_sha256'],'MATCHED_STATE_PROOF')
    for item,(start,end) in zip(parts,RANGES):
        result=_read_bound(item['result']);receipt=_read_bound(item['receipt']);pointer=_read_bound(item['pointer'])
        require(result['representations']==item['representation'] and result['partition']==item['partition']
            and item['partition']['presentation_start']==start and item['partition']['presentation_end']==end
            and result['source_unchanged'] is True and result['test_data_used'] is False
            and receipt['receipt_sha256']==binding.canonical_sha256({k:v for k,v in receipt.items() if k!='receipt_sha256'})
            and receipt['outcome']=='COMPLETE' and receipt['guard_decision']=='PASS'
            and receipt['trainer_guard_exit_code']==receipt['progress_observer_exit_code']==0
            and receipt['checkpoint_pointer_after']==item['pointer'] and pointer['result']==item['result']
            and pointer['outcome']=='COMPLETE' and pointer['test_data_used'] is False,'ONE_PART_TERMINAL')
    output=Path(plan['output_dir']);require(output.parent==ROOT and output.is_absolute() and not output.exists() and not output.is_symlink(),'OUTPUT')
    refs=plan['inputs']['one_parts']+[plan['inputs']['three']]
    require(len(refs)==4 and len({r['path'] for r in refs})==4 and all(Path(r['path']).suffix=='.npz' for r in refs),'PAYLOAD_SCOPE')
    output.mkdir(mode=0o700);_write_new(output/'CLAIM.json',{'plan':plan_ref,'source_head':args.source_head,'test_data_used':False})
    allowed_json={plan_ref['path'],plan['extraction_review']['path'],plan['one_pass_protocol']['path'],plan['one_pass_aggregate']['path']}
    reads,denied=_install_guard(refs,allowed_json,output)
    pieces=[_read_archive(ref) for ref in refs]
    require(set(reads)=={ref['path'] for ref in refs} and all(x==1 for x in reads.values()) and not denied,'OPEN_SCOPE')
    one,three=pair_arrays(pieces[:3],pieces[3],legacy=legacy)
    result=score_train(one,three,new_report['own_prefix_prior'])
    for panel,old,new in zip(result['panels'],old_aggregate['train'],new_report['report']['per_horizon']):
        require(panel['horizon_minutes']==old['horizon_minutes']==new['horizon_minutes']
            and panel['counts']['eligible']==old['eligible']==new['counts']['eligible_cells']
            and all(np.isclose(panel['arms']['one_pass'][k],old['terminal'][k],rtol=1e-12,atol=1e-12) for k in ('auc','logloss','brier'))
            and all(np.isclose(panel['arms']['three_pass'][k],new[k],rtol=1e-12,atol=1e-12) for k in ('logloss','brier')),'ORIGINAL_SCORE_REPRODUCTION')
    require(_git_state()==git and cap.require_capped_lock_ancestry()==lock,'FINAL_SOURCE_LOCK')
    result.update(schema_version='gx1_train_exposure_paired_result_v1',status='COMPLETE_TRAIN_FIT_ONLY',plan=plan_ref,
        source_head=args.source_head,matched_parent_presentation_clock_target_mask_contexts=True,
        original_scores_reproduced=True,payload_archives_read=4,checkpoint_reads=0,elapsed_seconds=time.monotonic()-start)
    out_ref=_write_new(output/'RESULT.json',result)
    print(json.dumps({'status':result['status'],'result':out_ref,'primary':result['panels'][1]},sort_keys=True))


if __name__=='__main__':
    main()
