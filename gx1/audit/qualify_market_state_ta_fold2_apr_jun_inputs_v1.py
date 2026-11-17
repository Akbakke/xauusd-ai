"""One-use April-June 2025 TA score sidecars with frozen Fold2 normalization."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

from gx1.audit import qualify_market_state_ta_anchor_span_inputs_v1 as replay
from gx1.audit import qualify_market_state_ta_benchmark_later_inputs_v1 as later
from gx1.contracts import entry_anchor_span_feature_inputs_v1 as overlay
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/TA_FOLD2_APR_JUN_INPUTS_20261117_001')
PLAN = REPO / 'configs/research/TA_FOLD2_APR_JUN_INPUTS_20261117_001.json'
PRIOR = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_LATER_INPUTS_20261113_002/PLAN.json')
VIEW = REPO / 'configs/research/TA_FOLD2_APR_JUN_FEATURE_VIEW_20261117_001.json'
STAGE = 'ta_fold2_apr_jun_inputs_20261117_001'
SOURCES = (
    'gx1/audit/qualify_market_state_ta_fold2_apr_jun_inputs_v1.py',
    'gx1/audit/qualify_market_state_ta_anchor_span_inputs_v1.py',
    'gx1/audit/qualify_market_state_ta_benchmark_later_inputs_v1.py',
    'gx1/contracts/entry_anchor_span_feature_inputs_v1.py',
    'gx1/contracts/entry_anchor_span_feature_view_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/features/htf_features.py',
    'gx1/features/trendline_registry_v1.py',
    'gx1/features/technical_indicators_v1.py',
    'gx1/scripts/audit_direction_d1_trendline_replay_v1.py',
)
MAX_SECONDS = 2700


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_APR_JUN_INPUT_' + reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'REGULAR_FILE')
    return {'path': str(path), 'sha256': sha(path)}


def read(binding):
    need(ref(binding['path']) == binding, 'INPUT_SHA')
    return json.loads(Path(binding['path']).read_text())


def publish(name, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    need(len(raw) < 2 * 1024 * 1024, 'JSON_BUDGET')
    _publish_bytes(ROOT / name, raw)
    return ref(ROOT / name)


def build_plan():
    old_ref = ref(PRIOR)
    old = read(old_ref)
    from gx1.contracts import entry_anchor_span_feature_view_v1 as span
    old_view = read(old['inputs']['view'])
    view_ref = ref(VIEW)
    view = read(view_ref)
    source_refs = deepcopy(old_view['source_bindings'])
    source_refs['anchor_span_inputs_source'] = ref(REPO / 'gx1/contracts/entry_anchor_span_feature_inputs_v1.py')
    expected_view = span.bind_anchor_span_feature_view(
        prefix_normalization=read(old['inputs']['packet']),
        prefix_normalization_ref=old['inputs']['packet'],
        constructor_metadata=read(old['inputs']['constructor']),
        constructor_metadata_ref=old['inputs']['constructor'],
        source_bindings=source_refs, max_batch_rows=16,
        unobserved_units_profile=None, full_feature_profile=True,
        full_ta_feature_profile=True, market_state_time_query_v2=True,
        dedicated_m5_side=True, dedicated_tf_activity=True)
    need(view == expected_view and
         {key for key in view if view[key] != old_view[key]} ==
         {'source_bindings', 'anchor_span_input_extension', 'contract_sha256'},
         'SOURCE_ONLY_VIEW_REBIND')
    panel = overlay.later_panel(fold_id=3)
    need(old['scope']['fold_id'] == 2 and old['scope']['score_end_exclusive_ns'] == panel['score_start_ns']
         and old['replay_plan']['view_sha256'] == old_view['contract_sha256']
         and view['prefix']['rows'] == panel['prefix_rows']
         and view['prefix']['cutoff_ns'] == panel['prefix_cutoff_ns'], 'FOLD2_LINEAGE')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return {'schema_version': 'gx1_ta_fold2_apr_jun_score_inputs_plan_v1',
            'stage': STAGE, 'source_base_commit': head,
            'source_owners': {name: sha(REPO / name) for name in SOURCES},
            'prior_plan': old_ref, 'view': view_ref,
            'scope': {**panel, 'fold_id': 3, 'selected_rows': 17536},
            'output_root': str(ROOT),
            'execution': {'class': 'audit', 'memory': '4G', 'swap': '512M',
                          'max_seconds': MAX_SECONDS, 'model_forwards': 0,
                          'optimizer_steps': 0, 'test_data_used': False}}


def inspect(plan_sha, *, execute):
    plan_ref = ref(PLAN)
    need(plan_ref['sha256'] == plan_sha, 'PLAN_SHA')
    plan = read(plan_ref)
    expected = build_plan()
    # Source base is an ancestor; the later admission edit changes HEAD only.
    expected['source_base_commit'] = plan['source_base_commit']
    need(plan == expected and not ROOT.exists(), 'EXACT_PLAN_AND_UNUSED_ROOT')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    need(subprocess.run(['git', 'merge-base', '--is-ancestor', plan['source_base_commit'], head],
                        cwd=REPO, capture_output=True).returncode == 0, 'SOURCE_ANCESTRY')
    policy_ref = ref(REPO / 'NEXT_RUN_POLICY.json')
    policy = read(policy_ref)
    need(all(policy[key] is False for key in
             ('training_enabled', 'native_training_enabled', 'full_epoch_training_allowed',
              'native_launch_authorized', 'full_val_allowed')), 'TRAINING_CLOSED')
    if execute:
        gate = policy[STAGE]
        need(gate['status'] == 'ADMITTED' and gate['plan'] == plan_ref
             and gate['execution_authorized'] is True and gate['training_authorized'] is False
             and gate['remaining_invocations'] == 1 and gate['relaunch_allowed'] is False,
             'EXACT_ADMISSION')
        need(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True),
             'CLEAN_SOURCE')
    return plan, plan_ref, policy_ref, head


def bookend(plan, plan_ref, policy_ref, head):
    need(ref(PLAN) == plan_ref and ref(REPO / 'NEXT_RUN_POLICY.json') == policy_ref
         and ref(PRIOR) == plan['prior_plan'] and ref(plan['view']['path']) == plan['view']
         and all(sha(REPO / name) == digest for name, digest in plan['source_owners'].items())
         and subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == head
         and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True),
         'SOURCE_BOOKEND')


def run(plan_sha):
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    cap = require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    began = time.monotonic()
    plan, plan_ref, policy_ref, head = inspect(plan_sha, execute=True)
    need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'AMBIENT_TIMER')
    ROOT.mkdir()
    claim = publish('CLAIM.json', {'plan': plan_ref, 'policy': policy_ref,
        'source_commit': head, 'cap': cap, 'claimed_utc': datetime.now(timezone.utc).isoformat()})
    status = 'FAILED_CONSUMED'
    failure = result = coverage = None
    def expire(*_):
        raise TimeoutError('TA_APR_JUN_INPUT_MAX_SECONDS')
    prior_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
        old = read(plan['prior_plan'])
        view = read(plan['view'])
        runtime = deepcopy(old['replay_plan'])
        runtime['view_sha256'] = view['contract_sha256']
        axes, local, clocks = later.read_axes_and_clocks(runtime, fold_id=3)
        bounds, populations, chosen, positions = later.envelopes_from_clocks(
            runtime, axes, local, clocks, fold_id=3)
        need(len(chosen) == plan['scope']['selected_rows'], 'EXACT_SCORE_AXIS')
        runtime['scope'] = {'ranges': bounds,
                            'history_end_exclusive_ns': plan['scope']['score_end_exclusive_ns'],
                            'array_output_budget_bytes': 4 * 1024 * 1024}
        runtime['populations'] = populations
        runtime['output_root'] = str(ROOT)
        del clocks
        reports = replay.measure(runtime, ranges=bounds,
            cutoff=plan['scope']['score_end_exclusive_ns'], output_root=ROOT)
        need(set(reports) == {'local', *replay.TFS}
             and all(item['parity']['fields'] == 31
                     and item['parity']['float32_bit_mismatched_cells'] == 0
                     for item in reports.values()), 'SIX_SURFACE_PARITY')
        counts = later.verify_native_windows(runtime, bounds, reports,
                                              axes, local, chosen, positions)
        bookend(plan, plan_ref, policy_ref, head)
        result = publish('RESULT.json', {'status': 'COMPLETE_APR_JUN_TA_SCORE_INPUTS_NO_MODEL',
            'plan': plan_ref, 'claim': claim, 'source_commit': head,
            'scope': plan['scope'], 'counts': counts, 'surfaces': reports,
            'view': plan['view'], 'source_unchanged': True, 'normalization_fitted': False,
            'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False,
            'economic_edge_proven': False, 'elapsed_seconds': time.monotonic() - began})
        coverage = publish('COVERAGE.json', later.coverage_record(
            runtime, view, axes, populations, result, fold_id=3))
        bookend(plan, plan_ref, policy_ref, head)
        status = 'COMPLETE'
    except BaseException as error:
        failure = publish('FAILED.json', {'type': type(error).__name__, 'error': str(error)[:2048],
            'plan': plan_ref})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
        publish('TERMINAL.json', {'status': status, 'exit_code': 0 if status == 'COMPLETE' else 1,
            'plan': plan_ref, 'result': result, 'coverage': coverage, 'failure': failure,
            'remaining_invocations': 0, 'relaunch_allowed': False,
            'elapsed_seconds': time.monotonic() - began})


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--build-plan', action='store_true')
    modes.add_argument('--preflight', action='store_true')
    modes.add_argument('--run', action='store_true')
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    if args.build_plan:
        need(args.plan_sha256 is None and not PLAN.exists(), 'BUILD_MODE')
        raw = (json.dumps(build_plan(), sort_keys=True, indent=2) + '\n').encode()
        _publish_bytes(PLAN, raw)
        print(json.dumps(ref(PLAN)))
        return
    need(type(args.plan_sha256) is str and len(args.plan_sha256) == 64
         and all(c in '0123456789abcdef' for c in args.plan_sha256), 'PLAN_ID')
    if args.preflight:
        plan, plan_ref, _, head = inspect(args.plan_sha256, execute=False)
        print(json.dumps({'status': 'READY_NO_CLAIM', 'plan': plan_ref,
                          'source_commit': head, 'scope': plan['scope']}))
    else:
        run(args.plan_sha256)


if __name__ == '__main__':
    main()
