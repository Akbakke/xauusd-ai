"""One admitted, fit-only target-support check for an October native private cohort."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.audit import dynamic_mtf_october_feature_delivery as delivery
from gx1.contracts import entry_dynamic_m5_october_native_roles_v1 as roles
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as targets
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
from gx1.contracts.entry_dynamic_mtf_december_candidate_plan_v1 import _cohort_counts, _cohort_rows, _axis_digest
from gx1.scripts import run_entry_native_direction_research_window_v1 as common

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / 'configs/research/DYNAMIC_M5_OCTOBER_PRIVATE_COHORT_20261106_001.json'
POLICY = REPO / 'NEXT_RUN_POLICY.json'
POLICY_KEY = 'dynamic_m5_october_private_cohort_20261106_001'
ROOT = Path('/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_PRIVATE_COHORT_20261106_001')
COMMAND = ('bash scripts/gx1_capped_run.sh --class audit --mem 2G --swap 512M -- '
           'timeout --signal=TERM --kill-after=10s 600s '
           '.venv/bin/python -m gx1.audit.dynamic_m5_october_private_cohort')
SEED = 20260911
LEADS = [1, 5, 15, 60, 240, 1440]


def _require(ok, reason):
    if not ok:
        raise RuntimeError('OCTOBER_PRIVATE_COHORT_' + reason)


def _identity(plan, policy, ref):
    admission = policy.get(POLICY_KEY)
    batch = roles._read(roles.BATCH_PLAN)
    refs = roles._read(roles.REFERENCE_PLAN)
    _require(plan.get('schema_version') == 'gx1_dynamic_m5_october_private_cohort_plan_v1'
             and plan.get('status') == 'PREPARED_NOT_ADMITTED'
             and plan.get('max_invocations') == 1
             and plan.get('output_root') == str(ROOT)
             and plan.get('execution') == {'command': COMMAND, 'memory': '2G', 'swap': '512M',
                                            'timeout_seconds': 600, 'kill_after_seconds': 10}
             and plan.get('seed') == SEED and plan.get('lead_minutes') == LEADS
             and plan.get('batch_plan') == roles.BATCH_PLAN
             and plan.get('reference_plan') == roles.REFERENCE_PLAN
             and plan.get('fit_series') == refs['fit_series']
             and plan.get('fit_prefix') == batch['fit_prefix']
             and plan.get('restrictions') == {'model_construction': False,
                 'model_forward': False, 'optimizer_updates': False, 'october_outcomes': False,
                 'test_split': False, 'broker_or_network': False, 'spending_usd': 0}
             and plan.get('standing_authorization') == policy['no_cost_local_standing_authorization_20261105']['operator_instruction']
             and isinstance(admission, dict) and admission.get('status') == 'ADMITTED_ONCE'
             and admission.get('plan') == ref and admission.get('remaining_invocations') == 1
             and admission.get('operator_approved') is True
             and policy['training_enabled'] is False
             and policy['full_epoch_training_allowed'] is False
             and policy['direction_edge_research_goal_20261105']['native_launch_authorized'] is False,
             'ADMISSION')
    _require(all(delivery._sha(REPO / path) == digest
                 for path, digest in plan['source_files_sha256'].items()), 'SOURCE_IDENTITY')
    return batch, refs


def run():
    require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    _require(Path.cwd().resolve() == REPO
             and subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'work/gx1-current'
             and subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == '', 'CLEAN_REPO')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    ref = delivery._ref(PLAN)
    plan = delivery._read(ref)
    policy_bytes = POLICY.read_bytes()
    batch, refs = _identity(plan, json.loads(policy_bytes), ref)
    manifest = delivery._read(refs['fit_series'])
    _require(manifest['metadata']['parents'] == 29849
             and manifest['metadata']['cutoff_ns'] == batch['fit_prefix']['cutoff_ns']
             and manifest['metadata']['source_refs'] == batch['target_source_refs']
             and set(manifest['arrays']) == set(targets.SERIES_ARRAY_DTYPES), 'FIT_TARGET_MANIFEST')
    stats = {name: common._stat(item['path']) for name, item in manifest['arrays'].items()}
    ROOT.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    delivery._publish(ROOT, 'CLAIM.json', {'plan': ref, 'source_head': head,
        'policy_sha256': hashlib.sha256(policy_bytes).hexdigest(), 'model_calls': 0})
    try:
        series = targets.load_dynamic_mtf_state_series(refs['fit_series'],
            expected_source_refs=batch['target_source_refs'],
            expected_cutoff_ns=batch['fit_prefix']['cutoff_ns'])
        rows, issues = series.arrays['parent_rows'], series.arrays['issue_ns']
        _require(np.array_equal(rows, np.arange(29849, dtype=np.int64))
                 and np.all(issues[1:] > issues[:-1])
                 and issues[-1] < batch['fit_prefix']['cutoff_ns']
                 and _axis_digest(rows) == batch['fit_prefix']['parent_rows_sha256']
                 and _axis_digest(issues) == batch['fit_prefix']['decision_ns_sha256'],
                 'FIT_AXES')
        order, private, initial = _cohort_rows(rows, SEED)
        selected = np.unique(np.r_[private, initial])
        query = targets.query_dynamic_mtf_states(series, parent_rows=selected,
            issue_ns=issues[selected].copy(), cutoff_ns=batch['fit_prefix']['cutoff_ns'],
            lead_minutes=np.asarray(LEADS, dtype=np.int64), include_first_flat_activity=True)
        _require(not np.any(query['observed'] & (query['maturity_ns'] >= batch['fit_prefix']['cutoff_ns'])),
                 'STRICT_MATURITY')
        batches = [_cohort_counts(query, np.searchsorted(selected, part)) for part in np.split(private, 2)]
        blockers = [{'batch': i + 1, 'timeframe': tf, 'measure': label}
                    for i, counts in enumerate(batches) for tf, values in counts.items()
                    for label, n in (('native_endpoint', values['native']['observed']),
                                     ('first_state', values['first']['observed']),
                                     ('first_activity', values['first_activity_observed'])) if n == 0]
        _require(all(common._stat(item['path']) == stats[name]
                     for name, item in manifest['arrays'].items())
                 and delivery._sha(POLICY) == hashlib.sha256(policy_bytes).hexdigest()
                 and delivery._ref(PLAN) == ref
                 and all(delivery._sha(REPO / path) == digest
                         for path, digest in plan['source_files_sha256'].items())
                 and subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == head
                 and subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == '',
                 'BOOKENDS')
        result = delivery._publish(ROOT, 'RESULT.json', {
            'schema_version': 'gx1_dynamic_m5_october_private_cohort_result_v1',
            'plan': ref, 'fit_series': refs['fit_series'], 'fit_parent_rows': 29849,
            'fit_cutoff_ns_exclusive': batch['fit_prefix']['cutoff_ns'],
            'seed': SEED, 'lead_minutes': LEADS,
            'one_pass_order_sha256_le_i64': _axis_digest(order),
            'private_parent_rows_sorted': private.tolist(),
            'initial_parent_rows_in_order': initial.tolist(),
            'private_batch_support': batches,
            'initial_support': _cohort_counts(query, np.searchsorted(selected, initial)),
            'private_support_blockers': blockers, 'private_support_pass': not blockers,
            'target_leaves_verified': len(stats), 'model_calls': 0,
            'october_outcomes_read': False, 'test_split_used': False,
            'independent_edge': 'UNKNOWN'})
        delivery._publish(ROOT, 'TERMINAL.json', {'plan': ref, 'result': result,
            'exit_code': 0, 'elapsed_seconds': time.monotonic() - began,
            'source_unchanged': True, 'no_relaunch_under_this_plan': True})
        return result
    except Exception as exc:
        delivery._publish(ROOT, 'FAILED.json', {'plan': ref, 'error_type': type(exc).__name__,
            'error': str(exc)[:2048], 'no_relaunch_under_this_plan': True})
        raise


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True))
