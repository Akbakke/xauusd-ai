"""Data-only check of the new private corrected-target adapter on bound inputs.

The native physical-provider and feature-view are deliberately stubbed; this
checks the actual target owner and fold masks, not native model consumption.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import signal
import sys
import time
from types import ModuleType, SimpleNamespace
import numpy as np

REPO=Path('/home/andre2/src/GX1_CURRENT')
ROOT=Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE='independent_direction_decimal_private_input_20261028_002'
SOURCES=(
    'gx1/scripts/audit_direction_decimal_private_input_v1.py',
    'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py',
    'gx1/scripts/run_entry_native_direction_research_window_v1.py',
    'gx1/scripts/run_entry_native_direction_dense_qualification_v1.py',
    'gx1/contracts/entry_native_direction_research_campaign_v1.py',
    'gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py',
    'gx1/contracts/entry_native_direction_decimal_input_v1.py',
    'gx1/contracts/entry_native_direction_decimal_data_v1.py',
    'gx1/contracts/entry_native_direction_dense_data_v1.py',
    'gx1/contracts/entry_native_direction_dense_v1.py',
    'gx1/features/entry_specialist_feature_groups_v1.py',
    'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_entry_native_direction_decimal_private_integration_v1.py',
    'tests/test_entry_native_direction_dense_qualification_v1.py',
)


def require(ok,reason):
    if not bool(ok):raise RuntimeError('DIRECTION_DECIMAL_PRIVATE_INPUT_AUDIT_'+reason)


def main():
    sys.dont_write_bytecode=True
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('design-sha256','policy-sha256','source-head'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix_authority
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as prefix
    from gx1.scripts import run_entry_native_direction_dense_qualification_v1 as qualification
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    require(not any(k.split('.')[0] in ('torch','sklearn','joblib') for k in sys.modules),'MODEL_PRELOAD')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CUDA_VISIBILITY')
    # The actual dense-data owner imports the horizon constant from the model
    # module. Load it before the audit hook, without constructing a model.
    from gx1.contracts.entry_native_direction_dense_data_v1 import AXES, TARGETS
    from gx1.features import htf_features as htf
    import torch
    require(len(AXES)==4 and len(TARGETS)==6 and not torch.cuda.is_initialized(),'CPU_OWNER_IMPORT')
    initial=candle.git_state();require(initial==(args.source_head,'work/gx1-current',''),'SOURCE')
    design_ref={'path':str(REPO/'docs/DIRECTION_DECIMAL_PRIVATE_INPUT_20261028_002.json'),'sha256':args.design_sha256}
    policy_ref={'path':str(REPO/'NEXT_RUN_POLICY.json'),'sha256':args.policy_sha256}
    with binding.fresh_file_verification_scope():
        design=binding.read_bound_json(Path(design_ref['path']),design_ref['sha256'])
        policy=binding.read_bound_json(Path(policy_ref['path']),policy_ref['sha256']);gate=policy[STAGE]
        require(design['schema_version']=='gx1_direction_decimal_private_input_design_v1' and design['stage']==STAGE
            and design['test_data_used'] is False and design['fold']==authority.BPS_FOLD,'DESIGN')
        require(gate['design']==design_ref and gate['status']=='ADMITTED' and gate['execution_authorized'] is True
            and gate['max_invocations']==1 and gate['claimed_invocations']==gate['completed_invocations']==0
            and gate['relaunch_allowed'] is False and all(gate[k] is False for k in ('model_forwards_authorized',
                'optimizer_steps_authorized','checkpoint_reads_authorized','test_data_used','broker_access_allowed')),'GATE')
        require(all(policy[k] is False for k in ('training_enabled','native_training_enabled','full_epoch_training_allowed',
            'full_val_allowed','native_launch_authorized')),'TRAINING_CLOSED')
        require(set(design['sources'])==set(SOURCES) and design['input_audit']==authority.DECIMAL_INPUT_AUDIT,'SOURCE_SET')
        for path,ref in design['sources'].items():
            require(ref['path']==str(REPO/path),'SOURCE_PATH');binding.require_binding(ref,label='source')
        audit=binding.read_bound_json(Path(design['input_audit']['path']),design['input_audit']['sha256'])
        refs=design['inputs']
        require(audit.get('status')=='COMPLETE' and audit.get('inputs')==refs
            and audit.get('physical_parent_and_bar_start_hashes_verified') is True
            and audit.get('corrected_archive_and_receipt_links_verified') is True
            and audit.get('test_data_used') is False,'INPUT_EVIDENCE')
        control=binding.capture_verified_file_snapshot()
    proof,lock=cap.require_capped_cpu_audit_execution(),cap.require_capped_lock_ancestry()
    require(lock['scope']==proof['cgroup_path'][len('/sys/fs/cgroup'):],'CAPPED_LOCK')
    out=ROOT/'DIRECTION_DECIMAL_PRIVATE_INPUT_20261028_002'
    require(design['output_dir']==str(out) and ROOT.resolve(strict=True)==ROOT and not out.exists() and not out.is_symlink(),'OUTPUT')
    out.mkdir(exist_ok=False)
    def publish(path,value):publication._publish_bytes(path,binding.canonical_bytes(value))
    publish(out/'INVOCATION_CLAIM.json',{'design':design_ref,'policy':policy_ref,'source_head':args.source_head,'execution':proof,'lock':lock})
    payloads=[refs[k]['path'] for k in ('legacy','original','corrected')]
    metadata=[design_ref['path'],policy_ref['path'],design['input_audit']['path']]+[refs[k]['path'] for k in
        ('original_result','corrected_result','consumer_plan','consumer_result','materialization_plan','audit_result')]
    sys.addaudithook(candle.make_guard(out,payloads,metadata,source_paths=[x['path'] for x in design['sources'].values()]))
    stage=out/'.OUTPUT.staging';started=time.monotonic()
    def expired(*unused):raise RuntimeError('DIRECTION_DECIMAL_PRIVATE_INPUT_AUDIT_WALL_LIMIT')
    previous=signal.signal(signal.SIGALRM,expired);signal.alarm(600)
    try:
        stage.mkdir(exist_ok=False)
        with binding.fresh_file_verification_scope():
            for ref in refs.values():binding.require_binding(ref,label='target input')
            with np.load(refs['legacy']['path'],allow_pickle=False) as archive:
                require(archive.files==list(prefix.PRICE_MEMBERS),'LEGACY_MEMBERS')
                arrays={key:archive[key].copy() for key in ('parent_rows','bar_start_ns','decision_ns')}
            prepared={'arrays':arrays,'gc':{'layout':{},'transition':{}},
                'entry_sequence_source':{'times_ns':np.arange(1,dtype=np.int64),'sample_positions':np.arange(1,dtype=np.int64)}}
            physical=SimpleNamespace(prepare_inputs=lambda *_:prepared,release_pages=lambda *_:None)
            source_refs={key:{'path':'/stub/'+key,'sha256':'0'*64} for key in ('raw_gc_layout','raw_gc_transition')}
            manifest={'targets':refs['legacy'],'view_source_bindings':source_refs,
                'prefix_normalization':{'output':{'path':'/stub/prefix','sha256':'0'*64}},
                'constructor_metadata':{'path':'/stub/constructor','sha256':'0'*64}}
            context={'physical':physical,'ancestor':object(),'ancestor_plan':{},'info':{'metadata':{}},'manifest':manifest,
                'fitted':{'base_normalization':{'surfaces':{'mtf_m5':{'field_names':[]}}}},'constructor':{},
                'dense_target':refs['corrected'],'decimal_original':refs['original'],'decimal_target_refs':refs}
            owners={key:design['sources'][key] for key in ('gx1/contracts/entry_native_direction_dense_v1.py',
                'gx1/features/entry_specialist_feature_groups_v1.py','gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py')}
            launch={'run_scope':prefix_authority.DENSE_DECIMAL_PRIVATE_SCOPE,'dense_target':refs['corrected'],
                'decimal_target_refs':refs,'fold':design['fold'],'source_owners':owners}
            view=ModuleType('gx1.contracts.entry_native_direction_feature_view_v1');view.bind_native_direction_feature_view=lambda **kw:kw
            report=ModuleType('gx1.contracts.entry_native_direction_context_report_v1');report.native_direction_context_map=lambda **kw:{}
            stubs={view.__name__:view,report.__name__:report}
            for name,module in stubs.items():require(name not in sys.modules,'PREEXISTING_MODULE:'+name);sys.modules[name]=module
            original_read=common._read;common._read=lambda *_:{}
            original_discard=htf.discard_multi_tf_v4_cache_pages;htf.discard_multi_tf_v4_cache_pages=lambda *_:None
            previous_profile=sys.getprofile();sys.setprofile(candle.no_native_models)
            try:
                got,targets,_view,_map,order=prefix._prepare(context,launch,np,dense_minutes=True)
            finally:
                sys.setprofile(previous_profile);common._read=original_read;htf.discard_multi_tf_v4_cache_pages=original_discard
                for name in stubs:sys.modules.pop(name,None)
            require(got is prepared and set(targets)==set(prefix.TARGET_MEMBERS) and len(order)==design['fold']['prefix_rows']
                and all(not x.flags.writeable for x in targets.values()),'ADAPTER_OUTPUT')
            sign_audit=binding.read_bound_json(Path(refs['audit_result']['path']),refs['audit_result']['sha256'])
            prior=prefix._prefix_prior(arrays,targets,design['fold'],np);qualification.require_decimal_prefix_prior(prior,sign_audit)
            with np.load(refs['corrected']['path'],allow_pickle=False) as archive:
                for key in prefix.TARGET_MEMBERS:require(np.array_equal(targets[key],archive[key]),'CORRECTED_BYTES:'+key)
            with np.load(refs['original']['path'],allow_pickle=False) as archive:
                old=archive['sign'];new=targets['sign'];changed=old!=new
                require(int(changed.sum())==976 and np.all(new[changed]==0) and np.all(old[changed]!=0),'SIGN_REPAIR')
            payload=binding.capture_verified_file_snapshot()
        payload.require_unchanged();control.require_unchanged()
        require(candle.git_state()==initial and cap.require_capped_lock_ancestry()==lock
            and not torch.cuda.is_initialized()
            and not any(k.split('.')[0] in ('sklearn','joblib') for k in sys.modules),'SOURCE_FREEZE_OR_MODEL')
        result={'schema_version':'gx1_direction_decimal_private_input_audit_v1','status':'COMPLETE','design':design_ref,
            'source_head':args.source_head,'input_audit':design['input_audit'],'inputs':refs,'parents':len(arrays['parent_rows']),
            'horizons_minutes':list(range(1,61)),'changed_original_nonzero_to_exact_zero_cells':976,
            'all_60_train_prior_counts_match_independent_source_audit':True,
            'five_minute_train_up':prior['eligible_up'][4],'five_minute_train_down':prior['eligible_down'][4],
            'native_prepare_target_branch_executed':True,'physical_provider_stubbed':True,'feature_view_stubbed':True,
            'torch_preloaded_for_existing_data_owner':True,'cuda_visible_devices':'','cuda_initialized':False,
            'model_calls_profile_guard_active':True,
            'physical_axes_source':'bound legacy targets, previously verified against physical axes',
            'limits':'Torch is imported on CPU for the existing target owner; model and CUDA calls are profiled out. No genuine feature gather, model construction, GPU, forward, optimizer or private mechanics qualification.',
            'model_forwards':0,'optimizer_steps':0,'checkpoint_reads':0,'test_data_used':False,'economic_edge_qualified':False}
        publish(stage/'RESULT.json',result);publish(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,
            'source_unchanged':True,'inputs_unchanged':True,'elapsed_seconds':time.monotonic()-started})
        publish_bundle_directory_noreplace(stage,out/'OUTPUT');publish(out/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'design':design_ref})
    except Exception as exc:
        publish(out/'FAILED_TERMINAL.json',{'status':'FAILED','exit_code':1,'error':type(exc).__name__+': '+str(exc),'relaunch_allowed':False});raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':main()
