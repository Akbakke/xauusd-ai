"""Fit only twelve new TRAIN span statistics; preserve all existing statistics."""
from __future__ import annotations

import argparse
import hashlib
import inspect as signatures
import json
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

from gx1.audit import qualify_market_state_ta_anchor_span_inputs_v1 as io
from gx1.contracts import entry_anchor_span_normalization_v1 as extension
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO, RUNS, POLICY = io.REPO, io.RUNS, io.POLICY
STAGE = 'market_state_ta_trendline_anchor_span_normalization_20261112_001'
ROOT = RUNS / STAGE.upper()
PLAN = REPO / 'configs/research' / (STAGE.upper() + '.json')
SOURCE_BASE = 'f312be89c4417b1d722d2ab91e4cc00c5c6394a1'
TERMINAL = dict(path=str(RUNS/'MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_INPUTS_20261112_001/TERMINAL.json'),
                sha256='9414bc5dd6e07ff565e4cb0f3db7c2364bbea3d954a9708c03fc91c131034007')
SOURCES = (
 'gx1/audit/qualify_market_state_ta_anchor_span_normalization_v1.py',
 'gx1/audit/qualify_market_state_ta_anchor_span_inputs_v1.py',
 'gx1/contracts/entry_anchor_span_normalization_v1.py',
 'gx1/contracts/entry_model_native_input_normalization_v1.py',
 'gx1/contracts/entry_native_direction_prefix_units_v1.py',
 'gx1/contracts/entry_model_native_training_objective_v1.py',
 'gx1/contracts/gx1_capped_execution_v1.py',
 'gx1/contracts/immutable_event_authority_v1.py',
 'gx1/features/entry_specialist_feature_groups_v1.py',
 'gx1/models/entry_v10/entry_v10_input_normalization.py',
 'scripts/gx1_capped_run.sh',
 'tests/test_entry_anchor_span_prefix_units_v1.py',
 'tests/test_entry_anchor_span_normalization_v1.py',
 'tests/test_market_state_ta_anchor_span_normalization_v1.py',
 'tests/test_entry_native_direction_prefix_units.py',
 'tests/test_market_state_mature_fold1_prefix_units.py',
)
SCOPE = dict(fold_id=1, parent_rows=29574, physical_surface_rows=73118, sidecar_archives=6,
             new_field_statistics=12, surface_fit_calls=6, old_field_fit_calls=0,
             old_statistics_refitted=False, target_arrays_decoded=0, model_constructions=0,
             model_forwards=0, checkpoint_reads=0, optimizer_steps=0, test_data_used=False)
WALL_SECONDS = 180


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('ANCHOR_SPAN_NORMALIZATION_AUDIT_' + reason)


def _parent_metadata():
    packet = io.read(extension.PARENT_ARTIFACT)
    parent = packet['base_normalization']
    need(parent['contract_sha256'] == extension.PARENT_BASE_SHA, 'EXACT_PARENT')
    fields = {k:v['field_names'] for k,v in parent['surfaces'].items()}
    norm.require_input_normalization_contract(
        parent, expected_field_names=fields, expected_ctx_cat_names=parent['ctx_cat']['field_names'],
        expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        unobserved_units_profile=units.mature_fold1_prefix_units_profile())
    return parent


def build_plan():
    """Metadata-only binding; no fit or array decode."""
    result = io.read(extension.PHYSICAL_RESULT)
    terminal = io.read(TERMINAL)
    need(result['status'] == 'COMPLETE_INPUT_EXTENSION_ONLY'
         and terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
         and terminal['result'] == extension.PHYSICAL_RESULT
         and result['source_unchanged'] is terminal['source_unchanged'] is True
         and result['normalization_fitted'] is result['fresh_model_baseline_complete'] is False
         and result['test_data_used'] is False, 'PHYSICAL_EXTENSION_COMPLETE')
    physical_plan = io.read(result['plan'])
    parent = _parent_metadata()
    lanes = {('local' if s == 'signal' else s[4:].upper()):s for s in extension.SURFACES}
    need(set(result['surfaces']) == set(lanes), 'SIX_PHYSICAL_SURFACES')
    populations = {}
    for lane,surface in lanes.items():
        item = result['surfaces'][lane]
        need(item['parity']['fields'] == 31 and item['parity']['float32_bit_mismatched_cells'] == 0
             and item['parity']['rows'] == parent['surfaces'][surface]['fit_row_count']
             and item['population'] == physical_plan['populations'][lane], 'PHYSICAL_POPULATION:' + lane)
        populations[surface] = dict(lane=lane, artifact=item['artifact'],
            artifact_stat=io.stat(item['artifact']['path']), population=item['population'],
            row_bounds=physical_plan['scope']['ranges'][lane], new_field_names=extension.span_names(surface))
    need(sum(p['population']['selected_unique_row_count'] for p in populations.values()) == 73118,
         'ROW_COUNT')
    signatures.signature(norm.fit_surface_normalization).bind(None,surface='signal',
        field_names=extension.span_names('signal'),allow_constant_train_fields=False)
    return dict(schema_version='gx1_exact_anchor_span_normalization_plan_v1',stage=STAGE,
        source_base_head=SOURCE_BASE,source_owners={p:io.sha(REPO/p) for p in SOURCES},
        inputs=dict(parent=extension.PARENT_ARTIFACT,physical_result=extension.PHYSICAL_RESULT,
                    physical_terminal=TERMINAL,physical_plan=result['plan']),
        populations=populations,scope=SCOPE,output_root=str(ROOT),
        parent_base_contract_sha256=parent['contract_sha256'],
        new_units_profile=units.anchor_span_fold1_prefix_units_profile(),
        execution=dict(module='gx1.audit.qualify_market_state_ta_anchor_span_normalization_v1',
            job_class='audit',runner='scripts/gx1_capped_run.sh',memory='4G',swap='512M',
            timeout_seconds=WALL_SECONDS,kill_after_seconds=10))


def inspect(plan_sha, *, execute):
    plan_ref=dict(path=str(PLAN),sha256=plan_sha)
    plan=io.read(plan_ref)
    need(plan==build_plan(), 'PLAN_IDENTITY')
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    need(Path.cwd().resolve()==REPO
         and subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='work/gx1-current'
         and subprocess.check_output(['git','status','--porcelain'],text=True)==''
         and subprocess.run(['git','merge-base','--is-ancestor',SOURCE_BASE,head]).returncode==0,'CLEAN_SOURCE')
    need(not ROOT.exists() or (ROOT.is_dir() and not ROOT.is_symlink() and not list(ROOT.iterdir())),
         'UNCONSUMED_ROOT')
    policy_ref=io.ref(POLICY);policy=io.read(policy_ref)
    need(all(policy[k] is False for k in
        ('training_enabled','native_training_enabled','native_launch_authorized','full_val_allowed')),'TRAINING_CLOSED')
    if execute:
        gate=policy.get(STAGE,{})
        need(gate.get('status')=='ADMITTED' and gate.get('plan')==plan_ref
             and gate.get('execution_authorized') is gate.get('data_reads_authorized') is True
             and gate.get('normalization_fits_authorized')==6 and gate.get('training_authorized') is False
             and gate.get('max_invocations')==gate.get('remaining_invocations')==1
             and gate.get('claimed_invocations')==gate.get('completed_invocations')==0
             and gate.get('relaunch_allowed') is False and gate.get('scope')==SCOPE,'EXACT_ADMISSION')
    return plan,plan_ref,policy_ref,head


def publish(name,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    need(len(raw)<=16*1024**2,'JSON_SIZE')
    _publish_bytes(ROOT/name,raw)
    return io.ref(ROOT/name)


def measure(plan):
    parent=_parent_metadata()
    stats,proof={},{}
    for surface,item in plan['populations'].items():
        path=Path(item['artifact']['path'])
        need(io.ref(path)==item['artifact'] and io.stat(path)==item['artifact_stat'],'SIDECAR_BINDING')
        with np.load(path,allow_pickle=False) as z:
            need(set(z.files)=={'row_indices','times_ns','native_bar_positions','anchor_span_bars'},'SIDECAR_MEMBERS')
            rows,times,positions,values=(z[k] for k in
                ('row_indices','times_ns','native_bar_positions','anchor_span_bars'))
        count=item['population']['selected_unique_row_count'];lane=item['lane']
        duration=io.BAR_NS['M5' if lane=='local' else lane]*10**9
        need(rows.dtype==times.dtype==positions.dtype==np.int64
             and rows.shape==times.shape==positions.shape==(count,)
             and values.dtype==np.float32 and values.shape==(count,2)
             and np.isfinite(values).all() and np.all(values>=0) and np.all(values==np.floor(values))
             and np.array_equal(rows,np.arange(*item['row_bounds'],dtype=np.int64))
             and io.row_sha(rows,lane)==item['population']['selected_row_indices_sha256']
             and io.axis_sha(times+duration)==item['population']['available_ns_sha256']
             and np.all(times+duration<io.CUTOFF)
             and np.all(np.diff(times)>0) and np.all(np.diff(positions)>0),'EXACT_FIT_ARRAYS:' + surface)
        stat=norm.fit_surface_normalization(values,surface=surface,field_names=item['new_field_names'],
                                            allow_constant_train_fields=False)
        normalized=norm.apply_surface_normalization(values,stat)
        recovered=np.sinh(normalized.astype(np.float64))*np.asarray(stat['scale'])+np.asarray(stat['center'])
        need(np.isfinite(normalized).all() and np.array_equal(np.rint(recovered),values),
             'INTEGER_SPAN_INVERSE:' + surface)
        for j in range(2):
            _unique,at=np.unique(values[:,j],return_index=True)
            need(len(at)>1 and np.all(np.diff(normalized[at,j])>0),'SPAN_INFORMATION_PRESERVED:' + surface)
        stats[surface]=stat
        proof[surface]=dict(rows=count,new_field_count=2,
            row_indices_sha256=io.row_sha(rows,lane),available_ns_sha256=io.axis_sha(times+duration),
            integer_spans_exact_after_inverse_rounding=True,
            inverse_max_abs_error=float(np.max(np.abs(recovered-values))),
            strictly_increasing_on_every_observed_span=True,
            normalized_values_sha256=hashlib.sha256(normalized.tobytes()).hexdigest(),
            scale_source=stat['scale_source'],center=stat['center'],scale=stat['scale'])
        need(io.stat(path)==item['artifact_stat'],'SIDECAR_BOOKEND')
        publish(surface.upper()+'_NEW_STATS.json',dict(statistics=stat,proof=proof[surface],
            physical_artifact=item['artifact'],normalization_extension_complete=False,test_data_used=False))
    extended=extension.extend_anchor_span_normalization(parent,stats)
    new_units=units.anchor_span_fold1_prefix_units_profile()
    fields={k:s['field_names'] for k,s in extended['surfaces'].items()}
    validated=norm.require_input_normalization_contract(
        extended,expected_field_names=fields,expected_ctx_cat_names=extended['ctx_cat']['field_names'],
        expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        unobserved_units_profile=new_units)
    need(validated==extended,'ACTUAL_EXTENSION_VALIDATION')
    ta=norm.project_ta_input_normalization_contract(extended,unobserved_units_profile=new_units)
    norm.require_input_normalization_contract(
        ta,expected_field_names={k:s['field_names'] for k,s in ta['surfaces'].items()},
        expected_ctx_cat_names=ta['ctx_cat']['field_names'],
        expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    need([ta['surfaces'][k]['field_count'] for k in norm.EXPECTED_SURFACES]==[317,132,255,255,255,255,255],
         'TA_PHYSICAL_WIDTHS')
    artifacts={name:publish(name+'.json',data) for name,data in
        (('NEW_FIELD_STATISTICS',stats),('EXTENDED_BASE_NORMALIZATION',extended),
         ('TA_INPUT_NORMALIZATION',ta),('NEW_UNITS_PROFILE',new_units))}
    return dict(artifacts=artifacts,new_field_fit_proof=proof,
        old_field_preservation=extension.old_field_preservation(parent,extended),
        base_contract_sha256=extended['contract_sha256'],ta_contract_sha256=ta['contract_sha256'],
        units_contract_sha256=new_units['contract_sha256'],
        new_statistics=12,surface_fit_calls=6,old_field_fit_calls=0,
        normalization_qualified=True,full_feature_view_gather_qualified=False,
        actual_model_baseline_complete=False,predictive_value_claimed=False)


def run(plan_sha):
    from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution
    start=time.monotonic();cap=require_capped_cpu_audit_execution()
    plan,plan_ref,policy_ref,head=inspect(plan_sha,execute=True)
    need(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'AMBIENT_TIMER')
    old_alarm=signal.getsignal(signal.SIGALRM)
    ROOT.mkdir(exist_ok=True)
    claim=publish('CLAIM.json',dict(plan=plan_ref,policy=policy_ref,source_head=head,scope=SCOPE,cap=cap,test_data_used=False))
    def bookend():
        need(subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==head
             and subprocess.check_output(['git','status','--porcelain'],text=True)==''
             and io.sha(PLAN)==plan_sha and io.sha(POLICY)==policy_ref['sha256']
             and all(io.sha(REPO/p)==h for p,h in plan['source_owners'].items())
             and all(io.ref(Path(r['path']))==r for r in plan['inputs'].values()),'SOURCE_BOOKEND')
    def expired(*unused):raise TimeoutError('ANCHOR_SPAN_NORMALIZATION_180_SECONDS')
    status,result,failure='FAILED_CONSUMED',None,None
    try:
        signal.signal(signal.SIGALRM,expired)
        signal.setitimer(signal.ITIMER_REAL,max(1,WALL_SECONDS-10-(time.monotonic()-start)))
        measured=measure(plan);bookend()
        result=publish('RESULT.json',dict(status='COMPLETE_NEW_STATISTICS_AND_NORMALIZATION_ONLY',
            plan=plan_ref,claim=claim,policy=policy_ref,source_head=head,source_unchanged=True,
            scope=SCOPE,measurement=measured,elapsed_seconds=time.monotonic()-start,test_data_used=False))
        status='COMPLETE'
    except BaseException as exc:
        failure=publish('FAILED.json',dict(type=type(exc).__name__,error=str(exc)[:4096],plan=plan_ref,test_data_used=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_alarm)
        error=None
        try:bookend()
        except BaseException as exc:
            error=exc;status='FAILED_CONSUMED'
            if failure is None:
                failure=publish('FAILED.json',dict(type=type(exc).__name__,error=str(exc)[:4096],plan=plan_ref,test_data_used=False))
        terminal=io.terminal_record(status=status,plan_ref=plan_ref,result=result,failure=failure,
                                     head=head,elapsed=time.monotonic()-start)
        terminal['source_unchanged']=error is None
        publish('TERMINAL.json',terminal)
        if error is not None:raise RuntimeError('ANCHOR_SPAN_NORMALIZATION_AUDIT_SOURCE_BOOKEND') from error


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--build-plan',action='store_true')
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--plan-sha256')
    args=parser.parse_args()
    if args.build_plan:
        need(not args.preflight and args.plan_sha256 is None and not PLAN.exists(),'BUILD_CLI')
        _publish_bytes(PLAN,(json.dumps(build_plan(),sort_keys=True,indent=2)+'\n').encode())
        print(json.dumps(io.ref(PLAN)))
    else:
        need(args.plan_sha256 is not None,'PLAN_SHA_REQUIRED')
        if args.preflight:
            plan,pref,_policy,head=inspect(args.plan_sha256,execute=False)
            print(json.dumps(dict(status='READY_NO_CLAIM_NO_ARRAY_DECODE_NO_FIT',plan=pref,source_head=head,scope=SCOPE)))
        else:run(args.plan_sha256)


if __name__=='__main__':
    main()
