"""Measure later support for GC context coordinates constant in the first TRAIN prefix."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import signal
import sys
import time

import numpy as np

REPO=Path('/home/andre2/src/GX1_CURRENT')
ROOT=Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE='independent_direction_unseen_gc_support_20261028_001'
SOURCE_PATHS=(
 'gx1/scripts/audit_direction_unseen_gc_support_v1.py',
 'gx1/features/gc_native_context_v1.py',
 'gx1/features/micro_structure_v1.py',
 'gx1/features/entry_specialist_feature_groups_v1.py',
 'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
 'gx1/contracts/gx1_capped_execution_v1.py',
 'gx1/contracts/local_random_access_campaign_v2.py',
 'gx1/contracts/immutable_event_authority_v1.py',
 'gx1/contracts/entry_model_native_bundle_commit_v1.py',
 'scripts/gx1_capped_run.sh',
 'tests/test_direction_unseen_gc_support_v1.py',
)


def require(ok,reason):
    if not bool(ok):raise RuntimeError('DIRECTION_UNSEEN_GC_SUPPORT_'+reason)


def normalized_values(values,masks,centers,scales,binary):
    values=np.asarray(values,dtype=np.float32);masks=np.asarray(masks,dtype=np.float32)
    centers=np.asarray(centers,dtype=np.float32);scales=np.asarray(scales,dtype=np.float32);binary=np.asarray(binary,dtype=np.bool_)
    require(values.ndim==2 and masks.shape==values.shape and centers.shape==scales.shape==binary.shape==(values.shape[1],)
        and np.isfinite(values).all() and np.isfinite(centers).all() and np.isfinite(scales).all()
        and np.isin(masks,(0.,1.)).all() and np.isin(values[:,binary],(0.,1.)).all() and np.all(scales>0), 'NORMALIZATION_DOMAIN')
    raw=values.astype(np.float64);center=centers.astype(np.float64);scale=scales.astype(np.float64)
    norm=np.where(binary,raw,np.arcsinh((raw-center)/scale)).astype(np.float32)
    return np.where(masks==1,norm,np.float32(0.0))


def summarize_group(values,masks,centers,scales,binary,names,decisions):
    norm=normalized_values(values,masks,centers,scales,binary);nonzero=norm!=0
    require(len(decisions)==len(values) and len(names)==values.shape[1], 'GROUP_AXES')
    fields=[]
    for j,name in enumerate(names):
        ids=np.flatnonzero(nonzero[:,j]);fields.append({'field':name,'observed_rows':int(np.count_nonzero(masks[:,j]==1)),
            'nonzero_normalized_rows':len(ids),'first_nonzero_decision_ns':int(decisions[ids[0]]) if len(ids) else None,
            'min_normalized':float(norm[:,j].min()),'max_normalized':float(norm[:,j].max())})
    return {'parents':len(values),'parents_with_any_nonzero':int(np.any(nonzero,axis=1).sum()),'fields':fields}


def main(argv=None):
    sys.dont_write_bytecode=True;parser=argparse.ArgumentParser(description=__doc__)
    for option in ('plan','plan-sha256','policy-sha256','source-head'):parser.add_argument('--'+option,required=True)
    args=parser.parse_args(argv)
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.features.gc_native_context_v1 import masking_pairs
    from gx1.features.micro_structure_v1 import range_observation_masking_pairs
    from gx1.features.entry_specialist_feature_groups_v1 import compile_gc_mtf_geometry_entry_execution_profile
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    require(not any(n.split('.')[0] in ('torch','sklearn','joblib') for n in sys.modules),'PRELOADED_MODEL')
    initial=candle.git_state();require(initial==(args.source_head,'work/gx1-current',''),'INITIAL_SOURCE')
    plan_ref={'path':str(candle.regular(args.plan)),'sha256':args.plan_sha256};policy_ref={'path':str(REPO/'NEXT_RUN_POLICY.json'),'sha256':args.policy_sha256}
    with binding.fresh_file_verification_scope():
        plan=binding.read_bound_json(Path(plan_ref['path']),plan_ref['sha256'])
        require(set(plan)=={'schema_version','stage','design','inputs','sources','output_dir','wall_seconds','test_data_used'}
            and plan['schema_version']=='gx1_direction_unseen_gc_support_plan_v1' and plan['stage']==STAGE
            and plan['wall_seconds']==600 and plan['test_data_used'] is False,'PLAN')
        require(plan['design']['path']==str(REPO/'docs/DIRECTION_UNSEEN_GC_SUPPORT_20261028.json'),'DESIGN_PATH')
        design=binding.read_bound_json(Path(plan['design']['path']),plan['design']['sha256'])
        require(design['schema_version']=='gx1_direction_unseen_gc_support_design_v1'
            and design['inputs']==plan['inputs'] and design['test_data_used'] is False,'DESIGN')
        policy=binding.read_bound_json(Path(policy_ref['path']),policy_ref['sha256']);gate=policy[STAGE]
        require(gate['status']=='ADMITTED' and gate['plan']==plan_ref and gate['execution_authorized'] is True
            and gate['max_invocations']==1 and gate['claimed_invocations']==gate['completed_invocations']==0
            and gate['relaunch_allowed'] is False,'ONE_SHOT_POLICY')
        require(all(policy[k] is False for k in ('training_enabled','native_training_enabled','full_epoch_training_allowed','full_val_allowed','native_launch_authorized'))
            and all(gate[k] is False for k in ('model_forwards_authorized','optimizer_steps_authorized','checkpoint_reads_authorized','test_data_used','broker_access_allowed')),'CLOSED_GATES')
        require(set(plan['sources'])==set(SOURCE_PATHS) and all(r['path']==str(REPO/n) for n,r in plan['sources'].items()),'SOURCE_SET')
        for ref in plan['sources'].values():candle.regular(ref['path']);binding.require_binding(ref,label='source')
        keys={'context_mmap','support_plan','decode_binding','prefix_normalization','fit_result','field_audit','corrected_target','sample_pair','fold_design'}
        require(set(plan['inputs'])==keys,'INPUT_SET')
        meta_names=('support_plan','decode_binding','prefix_normalization','fit_result','field_audit','fold_design')
        meta={k:binding.read_bound_json(Path(plan['inputs'][k]['path']),plan['inputs'][k]['sha256']) for k in meta_names}
        support,decoded,norm,fit,field_audit=(meta[k] for k in ('support_plan','decode_binding','prefix_normalization','fit_result','field_audit'))
        mmap_path=plan['inputs']['context_mmap']['path'];record=support['payloads'][mmap_path]
        require(record['verification']=='completed_decode_reuse_readonly' and record['stat']==decoded['scratch_files'][mmap_path]
            and record['stat']['st_size']==118249*2542*4 and not record['stat']['st_mode'] & 0o222
            and fit['feature_view_identity']['prefix_normalization']['binding']==plan['inputs']['prefix_normalization']
            and field_audit['status']=='COMPLETE' and field_audit['test_data_used'] is False
            and field_audit['trained_parent_cursor']==fit['parent_cursor']==47602
            and field_audit['final_updates']==fit['optimizer_updates']==2976,'INPUT_LINEAGE')
        surface=norm['base_normalization']['surfaces']['ctx_cont'];names=surface['field_names'];pairs=dict(masking_pairs(names))
        rows=[r for group in field_audit['field_support']['current_context_unchanged_columns'].values() for r in group]
        wanted=[r['name'] for r in rows];indices=[names.index(name) for name in wanted]
        require(len(wanted)==len(set(wanted))==33 and all(r['train_transformed_min']==r['train_transformed_max']==0 for r in rows)
            and all(i in pairs for i in indices) and not any(i in set(indices) for i,_ in range_observation_masking_pairs(names)), 'FIELD_OWNERS')
        profile=compile_gc_mtf_geometry_entry_execution_profile(norm['gc_mtf_history'])
        require(profile['contract_sha256']==field_audit['profile_sha256'] and set(indices)<=set(profile['ctx_cont_indices'])
            and all(names[i] in profile['gc_context_fields'] for i in indices),'ACTIVE_FIELDS')
        masks=[pairs[i] for i in indices];centers=[surface['center'][i] for i in indices];scales=[surface['scale'][i] for i in indices]
        binary=[bool(surface['binary_mask'][i] or surface['categorical_mask'][i]) for i in indices]
        require(all(surface['train_transformed_min'][i]==surface['train_transformed_max'][i]==0 for i in indices),'TRAIN_SUPPORT_METADATA')
        control_snapshot=binding.capture_verified_file_snapshot()
    proof,lock=cap.require_capped_cpu_audit_execution(),cap.require_capped_lock_ancestry()
    require(lock['scope']==proof['cgroup_path'][len('/sys/fs/cgroup'):],'CAPPED_LOCK_SCOPE')
    out=Path(plan['output_dir']);require(out.parent==ROOT and out.name=='DIRECTION_UNSEEN_GC_SUPPORT_20261028_001'
        and ROOT.resolve(strict=True)==ROOT and not out.exists() and not out.is_symlink(),'OUTPUT_SCOPE');out.mkdir(exist_ok=False)
    def publish(path,value):publication._publish_bytes(path,binding.canonical_bytes(value))
    publish(out/'INVOCATION_CLAIM.json',{'plan':plan_ref,'policy':policy_ref,'source_head':args.source_head,'execution':proof,'lock':lock})
    payloads=[plan['inputs'][k] for k in ('context_mmap','corrected_target','sample_pair')]
    metadata=[plan_ref['path'],policy_ref['path'],plan['design']['path']]+[plan['inputs'][k]['path'] for k in meta_names]
    sys.addaudithook(candle.make_guard(out,[r['path'] for r in payloads],metadata,source_paths=[r['path'] for r in plan['sources'].values()]))
    started=time.monotonic();old_handler=signal.getsignal(signal.SIGALRM)
    def expired(*unused):raise RuntimeError('DIRECTION_UNSEEN_GC_SUPPORT_WALL_LIMIT')
    signal.signal(signal.SIGALRM,expired);signal.alarm(plan['wall_seconds']);stage=out/'.OUTPUT.staging'
    try:
        stage.mkdir(exist_ok=False)
        with binding.fresh_file_verification_scope():
            for ref in payloads:candle.regular(ref['path']);binding.require_binding(ref,label='payload')
            stat=Path(mmap_path).stat();require(all(getattr(stat,key)==value for key,value in record['stat'].items()),'MMAP_ORIGINAL_STAT')
            with np.load(plan['inputs']['corrected_target']['path'],allow_pickle=False) as a:
                parents=a['parent_rows'].copy();starts=a['bar_start_ns'].copy();decisions=a['decision_ns'].copy()
            with np.load(plan['inputs']['sample_pair']['path'],allow_pickle=False) as a:
                sample=a['parent_rows'].copy();sample_decisions=a['decision_ns'].copy()
            require(parents.dtype==starts.dtype==decisions.dtype==sample.dtype==sample_decisions.dtype==np.int64
                and len(parents)==len(starts)==len(decisions)==118249 and np.all(np.diff(parents)>0) and np.all(np.diff(decisions)>0)
                and hashlib.sha256(np.ascontiguousarray(parents).tobytes()).hexdigest()==support['axes']['parent_rows_sha256']
                and hashlib.sha256(np.ascontiguousarray(starts).tobytes()).hexdigest()==support['axes']['bar_start_ns_sha256']
                and np.array_equal(decisions,starts+300*10**9),'EXACT_SOURCE_AXES')
            fold=fit['fold'];later=meta['fold_design']['folds'][1];positions=np.searchsorted(parents,sample)
            require(np.searchsorted(decisions,fold['train_cutoff_ns'])==fold['oof_start_row']
                and np.searchsorted(decisions,fold['oof_cutoff_ns'])==fold['oof_end_row']==later['oof_start_row']
                and np.searchsorted(decisions,later['oof_cutoff_ns'])==later['oof_end_row']
                and len(sample)==1760 and np.all(positions>=later['oof_start_row']) and np.all(positions<later['oof_end_row'])
                and np.array_equal(parents[positions],sample) and np.array_equal(decisions[positions],sample_decisions),'COHORTS')
            mmap=np.memmap(mmap_path,mode='r',dtype=np.float32,shape=(118249,2542));require(not mmap.flags.writeable,'READONLY_MMAP')
            groups={'fit_train':np.arange(fold['oof_start_row'],dtype=np.int64),
                'fit_oof_q1':np.arange(fold['oof_start_row'],fold['oof_end_row'],dtype=np.int64),'intervention_q2_sample':positions}
            summaries={}
            for name,at in groups.items():
                values=mmap[np.ix_(at,indices)];observed=mmap[np.ix_(at,masks)]
                summaries[name]=summarize_group(values,observed,centers,scales,binary,wanted,decisions[at])
                del values,observed
            require(summaries['fit_train']['parents_with_any_nonzero']==0,'TRAIN_ZERO_REPRODUCED')
            del mmap
            payload_snapshot=binding.capture_verified_file_snapshot()
        payload_snapshot.require_unchanged();control_snapshot.require_unchanged()
        require(candle.git_state()==initial and cap.require_capped_lock_ancestry()==lock
            and not any(n.split('.')[0] in ('torch','sklearn','joblib') for n in sys.modules),'FINAL_SOURCE_LOCK')
        result={'schema_version':'gx1_direction_unseen_gc_support_result_v1','status':'COMPLETE','plan':plan_ref,'source_head':args.source_head,
            'fields':33,'train_normalized_zero_reproduced':True,'groups':summaries,'context_mmap':plan['inputs']['context_mmap'],
            'limits':'Previously constant context projection columns and later nonzero support are scoped facts, not proof of output influence or predictive harm. Alias and other routes remain. Q2 uses a fixed development sample.',
            'new_model_forwards':0,'optimizer_steps':0,'test_data_used':False,'economic_edge_qualified':False}
        publish(stage/'RESULT.json',result);publish(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'source_unchanged':True,'inputs_unchanged':True,'elapsed_seconds':time.monotonic()-started,'test_data_used':False})
        publish_bundle_directory_noreplace(stage,out/'OUTPUT');publish(out/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':plan_ref,'result_path':str(out/'OUTPUT/RESULT.json')});return 0
    except Exception as exc:
        publish(out/'FAILED_TERMINAL.json',{'status':'FAILED','exit_code':1,'error':type(exc).__name__+': '+str(exc),'relaunch_allowed':False});raise
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)


if __name__=='__main__':raise SystemExit(main())
