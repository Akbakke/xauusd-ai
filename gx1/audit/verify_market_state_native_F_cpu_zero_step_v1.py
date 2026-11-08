"""One capped CPU TRAIN witness for the shared S/F head, without optimizer.

Reuses the completed v2 source/view qualification and two fixed real feature
rows. This worker can never train, read TEST, save weights or contact a broker.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import signal
import subprocess
import time

import numpy as np

from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
from gx1.audit import verify_market_state_time_query_v2_zero_step_v1 as v2
from gx1.audit import verify_market_state_time_query_zero_step_v1 as old
from gx1.contracts import market_state_future_displacement_candidate_v1 as future
from gx1.contracts import market_state_time_query_native_F_model_v1 as dual_model
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective

REPO, RUNS, POLICY = old.REPO, old.RUNS, old.POLICY
STAGE = 'market_state_native_F_cpu_zero_step_20261108_002'
PLAN_PATH = REPO / 'configs/research/MARKET_STATE_NATIVE_F_CPU_ZERO_STEP_20261108_002.json'
OUT_ROOT = RUNS / 'MARKET_STATE_NATIVE_F_CPU_ZERO_STEP_20261108_002'
SOURCE_PATHS = (
    'gx1/audit/verify_market_state_native_F_cpu_zero_step_v1.py',
    'gx1/contracts/market_state_future_displacement_candidate_v1.py',
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/market_state_time_query_loss_v2.py',
    'gx1/contracts/market_state_time_query_native_F_batch_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'scripts/gx1_capped_run.sh',
)
MEMORY_BYTES = 12 * 1024**3
SWAP_BYTES = 512 * 1024**2
WALL_SECONDS = 1800
WITNESS_F_WEIGHT = 1.0  # Prebound equal S/F candidate weight; no fit here.


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('NATIVE_F_CPU_ZERO_STEP_' + reason)


def sha(path: Path) -> str:
    return old._sha(path)


def ref(path: Path) -> dict:
    return {'path': str(path), 'sha256': sha(path)}


def current_state() -> tuple[str, str]:
    head = v2._clean_head()
    branch = subprocess.check_output(('git', 'branch', '--show-current'),
                                     cwd=REPO, text=True).strip()
    need(branch == 'work/gx1-current', 'BRANCH')
    return head, branch


def inspect(plan_ref: dict, *, require_policy: bool) -> tuple[dict, dict, dict, dict, dict, dict]:
    head, _ = current_state()
    need(plan_ref == ref(PLAN_PATH) and not OUT_ROOT.exists(), 'PLAN_OR_OUTPUT')
    plan = json.loads(PLAN_PATH.read_text())
    need(set(plan) == {'schema_version', 'stage', 'status', 'source_base_head',
                       'source_owners', 'inputs', 'scope', 'limits', 'output_root'}
         and plan['schema_version'] == 'gx1_market_state_native_F_cpu_zero_step_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['output_root'] == str(OUT_ROOT)
         and plan['scope'] == {'fold_id': 1, 'partition': 'fit',
                               'sample_parent_rows': v2.PARENT_ROWS,
                               'sample_issue_ns': v2.ISSUE_NS,
                               'pass_index': v2.PASS_INDEX, 'seed': v2.SEED,
                               'physical_model_forwards': 1,
                               'backward_calls': 1, 'optimizer_steps': 0,
                               'witness_F_weight': WITNESS_F_WEIGHT,
                               'training_authorized': False,
                               'test_data_used': False}
         and plan['limits'] == {'memory_max': MEMORY_BYTES,
                                'memory_high': MEMORY_BYTES,
                                'swap': SWAP_BYTES,
                                'wall_seconds': WALL_SECONDS}
         and plan['inputs'] == {'completed_v2_plan': ref(v2.PLAN_PATH),
                                'F_result': {'path': str(future.REFS['result'][0]),
                                             'sha256': future.REFS['result'][1]},
                                'F_tape': {'path': str(future.REFS['tape'][0]),
                                           'sha256': future.REFS['tape'][1]}}
         and subprocess.run(('git', 'merge-base', '--is-ancestor',
                             plan['source_base_head'], head), cwd=REPO,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode == 0,
         'PLAN_SCOPE')
    need(set(plan['source_owners']) == set(SOURCE_PATHS)
         and all(sha(REPO / path) == digest
                 for path, digest in plan['source_owners'].items()),
         'SOURCE_OWNERS')
    v2_plan, _ = v2._checked_plan(v2.PLAN_PATH,
                                  plan['inputs']['completed_v2_plan']['sha256'])
    receipts, view, state_bound = v2._inspect_inputs(v2_plan)
    future_bound = future.load_bound_future_displacement()
    selected = np.asarray(v2.PARENT_ROWS, dtype=np.int64)
    issues = np.asarray(v2.ISSUE_NS, dtype=np.int64)
    truth = future_bound.gather(selected, issues, fold_id=1, partition='fit')
    need(truth['allowed'].tolist() == [[2, 2, 1, 1, 4], [1, 1, 1, 1, 4]]
         and bool(truth['eligible'].all() and truth['mature'].all()),
         'F_TARGET_PAIR')
    policy_ref = ref(POLICY)
    if require_policy:
        policy = json.loads(POLICY.read_text())
        gate = policy.get(STAGE)
        need(policy.get('training_enabled') is False
             and policy.get('native_training_enabled') is False
             and policy.get('native_launch_authorized') is False
             and isinstance(gate, dict)
             and gate.get('status') == 'ADMITTED'
             and gate.get('plan') == plan_ref
             and gate.get('worker') == ref(REPO / SOURCE_PATHS[0])
             and gate.get('max_invocations') == 1
             and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
             and gate.get('execution_authorized') is True
             and gate.get('data_reads_authorized') is True
             and gate.get('model_forwards_authorized') == 1
             and gate.get('backward_calls_authorized') == 1
             and gate.get('optimizer_steps_authorized') == 0
             and gate.get('training_authorized') is False
             and gate.get('relaunch_allowed') is False
             and gate.get('test_data_used') is False,
             'POLICY')
    return plan, policy_ref, receipts, view, state_bound, future_bound


def preflight(plan_ref: dict) -> dict:
    plan, _, receipts, view, _, _ = inspect(plan_ref, require_policy=False)
    return {'status': 'READY_NO_CLAIM_NO_FEATURE_PAYLOAD',
            'plan': plan_ref, 'source_owners_checked': len(plan['source_owners']),
            'view_sha256': view['contract_sha256'],
            'model_contract_sha256': dual_model.contract()['contract_sha256'],
            'objective_contract_sha256': objective.contract()['contract_sha256'],
            'fold1_lineage': receipts['fold1']['lineage'],
            'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False}


def witness(receipts: dict, view: dict, state_bound, future_bound) -> dict:
    import torch
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts.market_state_time_query_native_F_batch_v1 import (
        FEATURE_KEYS, evaluate_fit_batch,
    )
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    chain = receipts['provider_chain']
    provider_plan = chain['provider_plan']
    provider = mature._provider_module()
    info = provider.validate_metadata(provider_plan)
    prepared = mature._prepare_inputs(provider, provider_plan, info, chain['gc_layout'])
    mature._validate_axes(receipts['mature_plan']['cohorts'], prepared['axes'])
    axes = prepared['axes']
    selected = np.asarray(v2.PARENT_ROWS, dtype=np.int64)
    issues = np.asarray(v2.ISSUE_NS, dtype=np.int64)
    at = np.searchsorted(axes['parent_rows'], selected)
    need(np.array_equal(axes['parent_rows'][at], selected)
         and np.array_equal(axes['decision_ns'][at], issues)
         and int(at[-1]) < receipts['fold1']['cohort']['joined_fit_rows'],
         'REAL_FEATURE_ROWS')
    for key in ('parent_rows', 'bar_start_ns', 'decision_ns'):
        axes[key].setflags(write=False)
    sequence = prepared['entry_sequence_source']
    for key in ('signal', 'times_ns', 'sample_positions'):
        sequence[key].setflags(write=False)
    for key in ('snap', 'ctx_cont', 'ctx_cat'):
        prepared['current'][key].setflags(write=False)
    raw = views.gather_native_direction_feature_batch(
        view, parent_rows=axes['parent_rows'],
        bar_start_ns=axes['bar_start_ns'], decision_ns=axes['decision_ns'],
        entry_sequence_source=sequence,
        snap_x=prepared['current']['snap'],
        ctx_cont=prepared['current']['ctx_cont'],
        ctx_cat=prepared['current']['ctx_cat'],
        multi_tf_sources=prepared['cache'],
        gc_surfaces=prepared['gc']['surfaces'],
        gc_layout=prepared['gc']['layout'],
        batch_parent_rows=selected,
        unobserved_units_profile=receipts['fold1']['profile'])
    need(set(raw['features']) == FEATURE_KEYS
         and raw['witness']['parent_rows'] == v2.PARENT_ROWS
         and raw['witness']['decision_ns'] == v2.ISSUE_NS,
         'FEATURE_ONLY_BATCH')
    mature._release_pages(provider, prepared)
    device = torch.device('cpu')
    need(not torch.cuda.is_initialized(), 'NO_CUDA')
    torch.manual_seed(v2.SEED)
    state_model = views.build_native_direction_research_model(
        view, prefix_normalization=receipts['fold1']['normal'],
        constructor_metadata=receipts['constructor'], device=device,
        unobserved_units_profile=receipts['fold1']['profile'])
    need(type(state_model).__name__ == 'FactorizedTimeQueryMarketStateModel'
         and state_model.feature_view_identity['contract_sha256']
             == v2.EXPECTED_VIEW_SHA256, 'FRESH_V2_MODEL')
    model = dual_model.NativeFutureDualModel(state_model)
    model.train()
    initial_state = canonical_model_state_sha256(model.state_dict())
    tensors = {}
    for key, value in raw['features'].items():
        tensor = torch.from_numpy(value).to(device)
        if tensor.is_floating_point():
            tensor.requires_grad_(True)
        tensors[key] = tensor
    captured = []
    handle = model.backbone.register_forward_hook(
        lambda _module, _args, output: captured.append(output['entry_q_joint_hidden']))
    with trainer._training_mha_fastpath_context(model.backbone), \
         trainer._training_autocast_context(), \
         trainer._training_model_finite_check_context():
        outcome = evaluate_fit_batch(
            model, tensors, state_bound, future_bound, selected, issues,
            fold_id=1, pass_index=v2.PASS_INDEX, seed=v2.SEED,
            future_weight=WITNESS_F_WEIGHT)
    handle.remove()
    need(len(captured) == 1 and outcome['state_count'] == 5
         and outcome['future_count'] == 10
         and outcome['state_query_k'].tolist() == v2.EXPECTED_QUERY_K
         and outcome['future_native_k'].tolist() == [1, 3, 12, 48, 288]
         and outcome['future_allowed'].tolist()
             == [[2, 2, 1, 1, 4], [1, 1, 1, 1, 4]]
         and bool(torch.isfinite(outcome['loss'])), 'FORWARD_TARGET_BINDING')
    with torch.no_grad():
        query = torch.from_numpy(outcome['state_query_k']).to(device)
        expected_s = state_model.decode_raw_query_grid(
            captured[0].detach(), query[:, None, :])[:, 0]
        expected_f = model.decode_native_F_raw(captured[0].detach())
    need(torch.equal(outcome['state_activity_side_logits'], expected_s)
         and torch.equal(outcome['future_activity_side_logits'], expected_f),
         'PREFIT_DECODE_PARITY')
    sraw = outcome['state_activity_side_logits']
    fraw = outcome['future_activity_side_logits']
    sraw.retain_grad()
    fraw.retain_grad()
    outcome['loss'].backward()
    sgrad, fgrad = sraw.grad, fraw.grad
    need(sgrad is not None and fgrad is not None
         and bool(torch.isfinite(sgrad).all() and torch.isfinite(fgrad).all())
         and float(sgrad[0, 0, 1]) == 0.0
         and float(sgrad[1, 0, 1]) != 0.0
         and float(fgrad[0, 0, 1]) == 0.0
         and float(fgrad[0, 1, 1]) == 0.0
         and float(fgrad[1, 0, 1]) != 0.0,
         'DIRECT_MASKED_GRADIENT')
    gradients = {
        'state_head': old._gradient_summary(state_model.query_head.weight.grad),
        'future_head': old._gradient_summary(model.future_head.weight.grad),
        'shared_projection': old._gradient_summary(state_model.query_projection[0].weight.grad),
        'backbone': old._gradient_summary(model.backbone.entry_q_joint_in.weight.grad),
    }
    need(all(x['present'] and x['finite'] and x['nonzero_count'] > 0
             for x in gradients.values()), 'BACKWARD_CONSUMPTION')
    final_state = canonical_model_state_sha256(model.state_dict())
    need(initial_state == final_state and not torch.cuda.is_initialized(),
         'NO_UPDATE_OR_CUDA')
    provider.validate_metadata(provider_plan)
    return {'decision': 'PASS_REAL_FEATURE_S_F_FORWARD_BACKWARD_NO_UPDATE',
            'scope_limit': 'Two fixed mature TRAIN rows; no fit, OOF, calibration, direction value, Entry or economic authority.',
            'sample_parent_rows': v2.PARENT_ROWS,
            'sample_issue_ns': v2.ISSUE_NS,
            'state_query_k': outcome['state_query_k'].tolist(),
            'future_native_k': outcome['future_native_k'].tolist(),
            'state_allowed': outcome['state_allowed'].tolist(),
            'future_allowed': outcome['future_allowed'].tolist(),
            'state_count': outcome['state_count'],
            'future_count': outcome['future_count'],
            'loss': float(outcome['loss'].detach()),
            'feature_witness': raw['witness'],
            'feature_shapes': {k: list(v.shape) for k, v in tensors.items()},
            'view_sha256': view['contract_sha256'],
            'model_contract_sha256': outcome['model_contract_sha256'],
            'loss_contract_sha256': outcome['loss_contract_sha256'],
            'initial_state_sha256': initial_state,
            'final_state_sha256': final_state,
            'gradients': gradients,
            'direct_masked_gradients': {
                'state_unknown_M5_side': float(sgrad[0, 0, 1]),
                'state_known_M5_side': float(sgrad[1, 0, 1]),
                'future_flat_M5_side': float(fgrad[0, 0, 1]),
                'future_flat_M15_side': float(fgrad[0, 1, 1]),
                'future_active_M5_side': float(fgrad[1, 0, 1]),
            },
            'cpu_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'model_forwards': 1, 'backward_calls': 1,
            'optimizer_steps': 0, 'checkpoint_loaded': False,
            'cuda_initialized': False, 'test_data_used': False}


def run(plan_ref: dict) -> dict:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_research_producer_execution, require_capped_lock_ancestry,
    )
    began = time.monotonic()
    plan, policy_ref, receipts, view, state_bound, future_bound = inspect(
        plan_ref, require_policy=True)
    cap = require_capped_cpu_research_producer_execution()
    cap['lock'] = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == MEMORY_BYTES
         and cap['swap'] == SWAP_BYTES and cap['pids'] == 64,
         'CAP')
    head, _ = current_state()
    OUT_ROOT.mkdir(parents=True, exist_ok=False)
    claim = None
    try:
        claim = old._write(OUT_ROOT, 'CLAIM.json', {
            'schema_version': 'gx1_market_state_native_F_cpu_zero_step_claim_v1',
            'plan': plan_ref, 'policy': policy_ref,
            'source_head': head, 'test_data_used': False},
            limit=old.MAX_JSON_BYTES)
        old_handler = signal.getsignal(signal.SIGALRM)
        def timeout(_signum, _frame):
            raise TimeoutError('NATIVE_F_CPU_ZERO_STEP_WALL')
        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS - 100)
        try:
            result = witness(receipts, view, state_bound, future_bound)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
        need(current_state()[0] == head
             and ref(PLAN_PATH) == plan_ref
             and ref(POLICY) == policy_ref
             and all(sha(REPO / path) == digest
                     for path, digest in plan['source_owners'].items()),
             'SOURCE_BOOKEND')
        result_ref = old._write(OUT_ROOT, 'RESULT.json', {
            'schema_version': 'gx1_market_state_native_F_cpu_zero_step_result_v1',
            'status': 'COMPLETE', 'plan': plan_ref, 'claim': claim,
            'policy': policy_ref, 'source_head': head,
            'cap': cap, 'elapsed_seconds': time.monotonic()-began,
            **result}, limit=old.MAX_JSON_BYTES)
        terminal = old._write(OUT_ROOT, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_native_F_cpu_zero_step_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0,
            'plan': plan_ref, 'claim': claim, 'result': result_ref,
            'model_forwards': 1, 'backward_calls': 1,
            'optimizer_steps': 0, 'test_data_used': False,
            'no_relaunch_under_this_plan': True}, limit=old.MAX_JSON_BYTES)
        return {'result': result_ref, 'terminal': terminal}
    except BaseException as exc:
        failed = old._write(OUT_ROOT, 'FAILED.json', {
            'schema_version': 'gx1_market_state_native_F_cpu_zero_step_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'elapsed_seconds': time.monotonic()-began,
            'no_relaunch_under_this_plan': True}, limit=old.MAX_JSON_BYTES)
        old._write(OUT_ROOT, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_native_F_cpu_zero_step_terminal_v1',
            'status': 'FAILED', 'exit_code': 1,
            'plan': plan_ref, 'claim': claim, 'failure': failed,
            'model_forwards_upper_bound': 1, 'backward_calls_upper_bound': 1,
            'optimizer_steps': 0, 'test_data_used': False,
            'no_relaunch_under_this_plan': True}, limit=old.MAX_JSON_BYTES)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    plan_ref = {'path': args.plan, 'sha256': args.plan_sha256}
    report = preflight(plan_ref) if args.preflight else run(plan_ref)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
