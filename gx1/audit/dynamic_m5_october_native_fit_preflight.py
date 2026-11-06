"""One admitted, model-free check of the October native fit-only data adapter."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.audit import dynamic_mtf_october_feature_delivery as delivery
from gx1.audit.dynamic_mtf_train_prefix_inputs import _check_payload_stats
from gx1.contracts import entry_dynamic_m5_october_native_roles_v1 as roles
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as worker
from gx1.scripts import run_entry_native_direction_research_window_v1 as common

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / 'configs/research/DYNAMIC_M5_OCTOBER_NATIVE_FIT_PREFLIGHT_20261106_003.json'
POLICY = REPO / 'NEXT_RUN_POLICY.json'
POLICY_KEY = 'dynamic_m5_october_native_fit_preflight_20261106_003'
ROOT = Path('/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_FIT_PREFLIGHT_20261106_003')
COMMAND = ('bash scripts/gx1_capped_run.sh --class audit --mem 4G --swap 512M -- '
           'timeout --signal=TERM --kill-after=10s 1800s '
           '.venv/bin/python -m gx1.audit.dynamic_m5_october_native_fit_preflight')
PARENTS = [0, 1, 29847, 29848]


def _require(ok, reason):
    if not ok:
        raise RuntimeError('OCTOBER_NATIVE_FIT_PREFLIGHT_' + reason)


def _identity(plan, policy, plan_ref):
    admitted = policy.get(POLICY_KEY)
    _require(plan.get('schema_version') == 'gx1_dynamic_m5_october_native_fit_preflight_plan_v1'
             and plan.get('status') == 'PREPARED_NOT_ADMITTED'
             and plan.get('max_invocations') == 1
             and plan.get('output_root') == str(ROOT)
             and plan.get('execution') == {'command': COMMAND, 'memory': '4G',
                 'swap': '512M', 'timeout_seconds': 1800, 'kill_after_seconds': 10}
             and plan.get('sample_parent_rows') == PARENTS
             and plan.get('arms') == ['control', 'candidate']
             and plan.get('batch_plan') == roles.BATCH_PLAN
             and plan.get('reference_plan') == roles.REFERENCE_PLAN
             and plan.get('restrictions') == {'model_construction': False,
                 'model_forward': False, 'optimizer_updates': False, 'october_outcomes': False,
                 'test_split': False, 'broker_or_network': False, 'spending_usd': 0}
             and plan.get('standing_authorization') == policy['no_cost_local_standing_authorization_20261105']['operator_instruction']
             and isinstance(admitted, dict) and admitted.get('status') == 'ADMITTED_ONCE'
             and admitted.get('plan') == plan_ref and admitted.get('remaining_invocations') == 1
             and admitted.get('operator_approved') is True
             and policy['training_enabled'] is False and policy['full_val_allowed'] is False
             and policy['full_epoch_training_allowed'] is False
             and policy['direction_edge_research_goal_20261105']['native_launch_authorized'] is False,
             'ADMISSION')
    predecessor = policy['dynamic_m5_october_native_fit_preflight_20261106_002']
    _require(predecessor['status'] == 'FAILED_CONSUMED' and predecessor['remaining_invocations'] == 0
             and plan.get('predecessor_failure') == predecessor['failed']
             and plan.get('reason_for_successor') == 'Merge exact control and candidate metadata into the guarded read inventory before source bookends; 002 stopped on CONTROL_VIEW.json.'
             and delivery._read(predecessor['failed'])['error'] ==
                'NATIVE_DIRECTION_RESEARCH_WINDOW_UNBOUND_OPEN:/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_VIEW_BINDING_20261106_001/CONTROL_VIEW.json',
             'PREDECESSOR')
    _require(all(delivery._sha(REPO / path) == digest for path, digest in plan['source_files_sha256'].items()),
             'SOURCE_IDENTITY')


def _launch(batch, refs, arm):
    return {'run_scope': roles.PRIVATE_SCOPE, 'input_role': 'fit', 'arm': arm,
            'candidate_plan': roles.BATCH_PLAN, 'reference_plan': roles.REFERENCE_PLAN,
            'feature_view': batch[arm + '_view'], 'target_manifest': refs['fit_series'],
            'causal_inputs': batch['causal_inputs'], 'observer_axes': batch['observer_axes'],
            'full_feature_profile': True, 'numerical_recipe': {'dynamic_state_supervision': True}}


def _bind_arm_metadata(contexts):
    # Both arms are verified before guard installation; the one guard needs both views.
    candidate = contexts['candidate']['info']['metadata']
    for path, item in contexts['control']['info']['metadata'].items():
        _require(path not in candidate or candidate[path] == item, 'ARM_METADATA_CONFLICT')
        candidate[path] = item


def _bookend(context):
    _check_payload_stats(context['ancestor_plan']['payloads'])
    for path, item in context['info']['metadata'].items():
        _require(common._stat(path) == item['stat']
                 and common._hash_file(path) == item['ref'], 'METADATA_BOOKEND')
    for refs, stats in ((context['state_array_refs'], context['state_array_stats']),
                        (context['october_causal_refs'], context['october_causal_stats'])):
        _require(all(common._stat(ref['path']) == stats[name] for name, ref in refs.items()),
                 'ARRAY_STAT_BOOKEND')


def run():
    require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    _require(Path.cwd().resolve() == REPO
             and subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'work/gx1-current'
             and subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == '',
             'CLEAN_REPO')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    plan_ref = delivery._ref(PLAN)
    plan = delivery._read(plan_ref)
    policy_bytes = POLICY.read_bytes()
    policy = json.loads(policy_bytes)
    _identity(plan, policy, plan_ref)
    batch, refs = roles._read(roles.BATCH_PLAN), roles._read(roles.REFERENCE_PLAN)
    launches = {arm: _launch(batch, refs, arm) for arm in plan['arms']}
    contexts = {arm: worker._launch_metadata(launches[arm], common) for arm in plan['arms']}
    _require(all(contexts[arm]['october_role'] == 'fit'
                 and contexts[arm]['state_target_manifest']['metadata']['parents'] == 29849
                 for arm in plan['arms']), 'FIT_ONLY_METADATA')
    _bind_arm_metadata(contexts)
    ROOT.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    delivery._publish(ROOT, 'CLAIM.json', {'plan': plan_ref, 'source_head': head,
        'policy_sha256': hashlib.sha256(policy_bytes).hexdigest(),
        'sample_parent_rows': PARENTS, 'model_calls': 0})
    try:
        for context in contexts.values():
            context['state_artifact_directory'] = ROOT
        denied = common._install_read_guard(contexts['candidate'], ROOT, ROOT / 'RUNTIME')
        selected = np.asarray(PARENTS, np.int64)
        samples = {}
        for arm in plan['arms']:
            context = contexts[arm]
            prepared = None
            try:
                prepared, targets, view, _, order = worker._prepare_dynamic_state(context, launches[arm], np)
                _require(prepared['october_role'] == 'fit' and len(prepared['arrays']['parent_rows']) == 29849
                         and len(order) == 29849 and len(targets['dynamic_state_series'].arrays['parent_rows']) == 29849,
                         'FIT_ONLY_PREPARED')
                raw, truth = worker._batch(prepared, targets, view, selected, np)
                at = np.searchsorted(prepared['arrays']['parent_rows'], selected)
                _require(np.array_equal(prepared['arrays']['parent_rows'][at], selected)
                         and truth['state'].shape == (4, 5, 1440)
                         and np.array_equal(truth['issue_ns'], prepared['arrays']['decision_ns'][at]),
                         'NATIVE_BATCH_TARGETS')
                samples[arm] = (raw['features'], truth)
            finally:
                if prepared is not None:
                    context['physical'].release_pages(context['ancestor'], prepared, None)
            _bookend(context)
        control, candidate = samples['control'], samples['candidate']
        base = set(control[0])
        _require(set(candidate[0]) == base | {'current_confirmed_run_age_features'}
                 and all(np.array_equal(control[0][name], candidate[0][name]) for name in base)
                 and all(np.array_equal(control[1][name], candidate[1][name]) for name in control[1])
                 and not denied, 'MATCHED_ARMS_AND_GUARD')
        _require(delivery._sha(POLICY) == hashlib.sha256(policy_bytes).hexdigest()
                 and delivery._ref(PLAN) == plan_ref
                 and all(delivery._sha(REPO / path) == digest for path, digest in plan['source_files_sha256'].items())
                 and subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == head
                 and subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == '',
                 'SOURCE_OR_POLICY_CHANGED')
        result = delivery._publish(ROOT, 'RESULT.json', {
            'schema_version': 'gx1_dynamic_m5_october_native_fit_preflight_result_v1',
            'plan': plan_ref, 'sample_parent_rows': PARENTS,
            'fit_target': refs['fit_series'], 'control_view': batch['control_view'],
            'candidate_view': batch['candidate_view'], 'base_feature_groups': sorted(base),
            'candidate_age_group': 'current_confirmed_run_age_features',
            'physical_read_files': len(contexts['candidate']['ancestor_plan']['payloads']),
            'target_leaves': len(contexts['candidate']['state_array_refs']),
            'causal_leaves': len(contexts['candidate']['october_causal_refs']),
            'matched_base_features_and_targets': True, 'denied_reads': denied,
            'model_constructions': 0, 'model_forwards': 0, 'optimizer_updates': 0,
            'october_outcomes_read': False, 'test_split_used': False,
            'independent_edge': 'UNKNOWN'})
        delivery._publish(ROOT, 'TERMINAL.json', {'plan': plan_ref, 'result': result,
            'exit_code': 0, 'elapsed_seconds': time.monotonic() - began,
            'source_unchanged': True, 'no_relaunch_under_this_plan': True})
        return result
    except Exception as exc:
        delivery._publish(ROOT, 'FAILED.json', {'plan': plan_ref,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'no_relaunch_under_this_plan': True})
        raise


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True))
