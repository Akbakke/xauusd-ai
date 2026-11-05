"""One capped, model-free check of actual October native feature/target joins."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.audit import dynamic_mtf_october_feature_delivery as delivery
from gx1.contracts import entry_dynamic_m5_october_causal_inputs_v1 as causal
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as state
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as worker


REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / 'configs/research/DYNAMIC_M5_OCTOBER_NATIVE_BATCH_BINDING_20261106_002.json'
POLICY = REPO / 'NEXT_RUN_POLICY.json'
POLICY_KEY = 'dynamic_m5_october_native_batch_binding_20261106_002'


def _require(ok, reason):
    if not ok:
        raise RuntimeError('OCTOBER_NATIVE_BATCH_BINDING_' + reason)


def _read(ref):
    return delivery._read(ref)


def _ref(path):
    return delivery._ref(path)


def _sha(path):
    return delivery._sha(path)


def _role_prepared(raw, axes, view, scope, tape, role, part, profile):
    observer = views.NativeDirectionObserverAxes(view, source=scope['source'],
        parent_rows=axes['parent_rows'][part], bar_start_ns=axes['bar_start_ns'][part],
        decision_ns=axes['decision_ns'][part], role=role,
        unobserved_units_profile=profile)
    sequence = raw['entry_sequence_source']
    current = {key: value[part] for key, value in raw['current'].items()}
    for value in (*current.values(), sequence['sample_positions'][part]):
        value.setflags(write=False)
    prepared = {'arrays': {'parent_rows': observer.parent_rows,
                           'bar_start_ns': observer.bar_start_ns,
                           'decision_ns': observer.decision_ns},
                'entry_sequence_source': {'signal': sequence['signal'],
                    'times_ns': sequence['times_ns'],
                    'sample_positions': sequence['sample_positions'][part],
                    'audit': sequence['audit']},
                'current': current, 'cache': raw['mtf'],
                'gc': {'surfaces': raw['gc'], 'layout': raw['gc_layout']},
                'observer_axes': observer, 'october_role': role,
                'october_units_profile': profile, 'october_causal_inputs': tape}
    return prepared


def run():
    require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    _require(Path.cwd().resolve() == REPO
             and subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'work/gx1-current'
             and subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == '', 'CLEAN_REPO')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    plan_ref = _ref(PLAN)
    plan = _read(plan_ref)
    policy_bytes = POLICY.read_bytes()
    policy = json.loads(policy_bytes)
    admitted = policy.get(POLICY_KEY)
    _require(plan['schema_version'] == 'gx1_dynamic_m5_october_native_batch_binding_plan_v1'
             and plan['status'] == 'PREPARED_NOT_ADMITTED' and plan['max_invocations'] == 1
             and plan['sample_parent_positions'] == {'fit': [0, 1, 29847, 29848],
                 'development': [29849, 29850, 36170, 36171]}
             and plan['output_root'] == '/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_BATCH_BINDING_20261106_002'
             and plan['execution']['command'] ==
                'bash scripts/gx1_capped_run.sh --class audit --mem 4G --swap 512M -- timeout --signal=TERM --kill-after=10s 1800s .venv/bin/python -m gx1.audit.dynamic_m5_october_native_batch_binding'
             and plan['restrictions'] == {'model_construction': False, 'model_forward': False,
                 'optimizer_updates': False, 'test_split': False, 'broker_or_network': False,
                 'spending_usd': 0, 'post_october_outcomes': False}
             and isinstance(admitted, dict) and admitted.get('status') == 'ADMITTED_ONCE'
             and admitted.get('plan') == plan_ref and admitted.get('operator_approved') is True
             and admitted.get('remaining_invocations') == 1
             and policy['training_enabled'] is False and policy['full_val_allowed'] is False
             and policy['full_epoch_training_allowed'] is False
             and policy['direction_edge_research_goal_20261105']['native_launch_authorized'] is False,
             'ADMISSION')
    _require(plan['standing_authorization'] == policy['no_cost_local_standing_authorization_20261105']['operator_instruction']
             and all(_sha(REPO / path) == digest for path, digest in plan['source_files_sha256'].items()),
             'SOURCE_IDENTITY')
    control = _read(plan['control_view'])
    candidate = _read(plan['candidate_view'])
    scope = _read(plan['observer_axes'])
    previous = _read(plan['prior_input_plan'])
    failed = _read(plan['predecessor_failure'])
    _require(failed['error'] == 'NATIVE_DIRECTION_FEATURE_VIEW_EXPLICIT_UNOBSERVED_UNITS_PROFILE'
             and failed['no_relaunch_under_this_plan'] is True
             and plan['reason_for_successor'] == 'Bind the existing October source-unit profile into the native feature gather call; predecessor 001 is consumed.',
             'PREDECESSOR_FAILURE')
    _require(control['october_observer_axes'] == candidate['october_observer_axes'] == scope
             and control['prefix_normalization'] == candidate['prefix_normalization']
             and previous['full_axes'] == plan['full_axes']
             and previous['fit_prefix'] == plan['fit_prefix']
             and previous['evaluation_axis'] == plan['evaluation_axis'], 'MATCHED_SCOPE')
    profile = units.october_prefix_units_profile()
    views._require_view(control, unobserved_units_profile=profile)
    views._require_view(candidate, unobserved_units_profile=profile)
    root = Path(plan['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    try:
        provider_plan = _read(plan['ancestral_264_plan'])
        provider = delivery.prior._load_module(plan['ancestral_264_worker'], 'gx1_october_batch_264_reader')
        with campaign.file_verification_scope():
            info = provider.validate_metadata(provider_plan)
            axes, full = provider.read_axes(provider_plan, info, np)
            _require(full == plan['full_axes'], 'FULL_AXES')
            delivery.prior._axis_receipt(axes, full, plan['fit_prefix'], plan['evaluation_axis'])
            read_plan = dict(provider_plan, october_gc_transition=plan['current_gc_transition'],
                             october_gc_layout=plan['current_gc_layout'])
            raw = delivery.prior._prepare_prefix_inputs(provider, read_plan, info, axes,
                fit_rows=36172, last_issue_ns=int(axes['decision_ns'][36171]))
            campaign.capture_verified_file_snapshot()
        for value in (raw['entry_sequence_source']['signal'],
                      raw['entry_sequence_source']['times_ns'],
                      raw['entry_sequence_source']['sample_positions'],
                      *raw['current'].values()):
            value.setflags(write=False)
        tape = causal.load_october_causal_inputs(plan['causal_inputs'],
            observer_axes_ref=plan['observer_axes'], candidate_design_ref=plan['candidate_design'],
            target_series_ref=plan['october_series'])
        series = state.load_dynamic_mtf_state_series(plan['october_series'],
            expected_source_refs=plan['target_source_refs'],
            expected_cutoff_ns=plan['evaluation_axis']['cutoff_ns'])
        _require(np.array_equal(tape.arrays['parent_rows'], axes['parent_rows'][:36172])
                 and np.array_equal(tape.arrays['issue_ns'], axes['decision_ns'][:36172]),
                 'CAUSAL_SOURCE_AXES')
        leads = np.asarray(series.metadata['definition']['trial_lead_minutes'], dtype=np.int64)
        _require(len(leads) == 1440 and int(leads[0]) == 1 and int(leads[-1]) == 1440,
                 'EXACT_FIVE_TF_TARGET_DOMAIN')
        targets = {'dynamic_state_series': series,
                   'train_cutoff_ns': scope['fit']['cutoff_ns'], 'lead_minutes': leads}
        base_keys = {'seq_x', 'snap_x', 'ctx_cont', 'ctx_cat',
                     *(f'seq_{tf.lower()}' for tf in state.TIMEFRAMES),
                     *(f'gc_seq_{tf.lower()}' for tf in state.TIMEFRAMES),
                     'current_state', 'current_observed', 'current_path_observed'}
        evidence = {}
        for role, part, selected in (
            ('fit', slice(0, 29849), axes['parent_rows'][[0, 1, 29847, 29848]]),
            ('score', slice(29849, 36172), axes['parent_rows'][[29849, 29850, 36170, 36171]])):
            sampled = np.ascontiguousarray(selected)
            rows = {}
            for name, view in (('control', control), ('candidate', candidate)):
                prepared = _role_prepared(raw, axes, view, scope, tape, role, part, profile)
                features, target = worker._batch(prepared, targets, view, sampled, np,
                    cutoff=scope['held_out']['cutoff_ns'] if role == 'score' else None)
                expected_keys = base_keys | ({'current_confirmed_run_age_features'} if name == 'candidate' else set())
                _require(set(features['features']) == expected_keys
                         and 'state' not in features['features']
                         and target['state'].shape == (4, 5, 1440)
                         and np.array_equal(target['issue_ns'], tape.arrays['issue_ns'][np.searchsorted(
                             tape.arrays['parent_rows'], sampled)]), 'BATCH_SHAPE_AND_INPUT_KEYS')
                rows[name] = (features['features'], target)
            _require(all(np.array_equal(rows['control'][0][key], rows['candidate'][0][key])
                         for key in base_keys)
                     and all(np.array_equal(rows['control'][1][key], rows['candidate'][1][key])
                             for key in rows['control'][1]), 'MATCHED_CONTROL_CANDIDATE')
            evidence[role] = {'sampled_parents': sampled.tolist(),
                'raw_and_current_features_equal': True,
                'candidate_age_input_shape': list(rows['candidate'][0]['current_confirmed_run_age_features'].shape),
                'm5_plus5_observed_in_sample': int(rows['candidate'][1]['observed'][:, 0, 4].sum())}
        _require(_sha(POLICY) == hashlib.sha256(policy_bytes).hexdigest()
                 and _ref(PLAN) == plan_ref
                 and all(_sha(REPO / path) == digest for path, digest in plan['source_files_sha256'].items())
                 and subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == head
                 and subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == '',
                 'SOURCE_OR_POLICY_CHANGED')
        result = delivery._publish(root, 'RESULT.json', {
            'schema_version': 'gx1_dynamic_m5_october_native_batch_binding_result_v1',
            'plan': plan_ref, 'causal_inputs': plan['causal_inputs'], 'samples': evidence,
            'full_target_domain_minutes': len(leads), 'native_feature_groups_including_current': len(base_keys),
            'candidate_only_age_group': 'current_confirmed_run_age_features',
            'model_constructions': 0, 'model_forwards': 0, 'optimizer_updates': 0,
            'test_split_used': False, 'independent_edge': 'UNKNOWN'})
        delivery._publish(root, 'TERMINAL.json', {'plan': plan_ref, 'result': result,
            'exit_code': 0, 'elapsed_seconds': time.monotonic() - began,
            'source_unchanged': True, 'no_relaunch_under_this_plan': True})
        return result
    except Exception as exc:
        delivery._publish(root, 'FAILED.json', {'plan': plan_ref,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'no_relaunch_under_this_plan': True})
        raise


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True))
