"""One capped full-GC fold2 TRAIN witness for selected native S/F endpoints.

Reuses the qualified fold2 anchor-span packet, six sidecars and old GC arrays.
One model forward and backward, zero optimizer updates; no TEST or trading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import subprocess
import time

import numpy as np

from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
from gx1.audit import verify_market_state_time_query_zero_step_v1 as old
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as selected_state
from gx1.contracts import market_state_future_displacement_candidate_v1 as future
from gx1.contracts import market_state_time_query_native_F_model_v1 as dual_model
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective

REPO, RUNS, POLICY = old.REPO, old.RUNS, old.POLICY
STAGE = 'market_state_native_F_selected_endpoint_initial_fold2_20261115_002'
PLAN_PATH = REPO / 'configs/research/MARKET_STATE_NATIVE_F_SELECTED_ENDPOINT_INITIAL_FOLD2_20261115_002.json'
OUT_ROOT = RUNS / 'MARKET_STATE_NATIVE_F_SELECTED_ENDPOINT_INITIAL_FOLD2_20261115_002'
WORKER_PATH = 'gx1/audit/verify_market_state_native_F_cpu_zero_step_v1.py'
FOLD2_ROOT = RUNS / 'MARKET_STATE_TA_FOLD2_SPAN_VIEW_20261113_001'
FOLD2_VIEW_PLAN = FOLD2_ROOT / 'PLAN.json'
PARENT_ROWS = [0, 2]
ISSUE_NS = [1714521900000000000, 1714522500000000000]
PASS_INDEX = 0
SEED = 20261107
SOURCE_PATHS = tuple(sorted(mature.SOURCE_PATHS | {
    WORKER_PATH,
    'gx1/audit/verify_market_state_time_query_zero_step_v1.py',
    'gx1/contracts/entry_anchor_span_feature_view_v1.py',
    'gx1/contracts/entry_anchor_span_feature_inputs_v1.py',
    'gx1/contracts/entry_anchor_span_normalization_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/contracts/entry_native_direction_research_v1.py',
    'gx1/contracts/market_state_time_query_binding_v1.py',
    'gx1/contracts/market_state_time_query_model_v2.py',
    'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py',
    'gx1/contracts/market_state_future_displacement_candidate_v1.py',
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/market_state_time_query_loss_v2.py',
    'gx1/contracts/market_state_time_query_side_normalized_loss_v1.py',
    'gx1/contracts/market_state_time_query_native_F_batch_v1.py',
    'gx1/contracts/market_state_time_query_target_v1.py',
    'gx1/contracts/model_state_digest_v1.py',
    'gx1/models/entry_v10/entry_v10_ctx_train_v3.py',
}))
MEMORY_BYTES = 12 * 1024**3
SWAP_BYTES = 512 * 1024**2
WALL_SECONDS = 1800
WITNESS_F_WEIGHT = 1.0


def planned_command(plan_sha256: str) -> list[str]:
    return ['timeout', '--signal=TERM', '--kill-after=10s', '1800s',
            'scripts/gx1_capped_run.sh', '--class', 'producer', '--mem', '12G',
            '--swap', '512M', '--', '.venv/bin/python', '-m',
            'gx1.audit.verify_market_state_native_F_cpu_zero_step_v1',
            '--plan', str(PLAN_PATH), '--plan-sha256', plan_sha256]


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('NATIVE_F_CPU_ZERO_STEP_' + reason)


def sha(path: Path) -> str:
    return old._sha(path)


def ref(path: Path) -> dict:
    return {'path': str(path), 'sha256': sha(path)}


def checked_file(item: dict) -> Path:
    need(type(item) is dict and set(item) == {'path', 'sha256'}, 'FILE_REF_KEYS')
    path = Path(item['path'])
    need(path.is_absolute() and path.is_file() and not path.is_symlink()
         and sha(path) == item['sha256'], 'FILE_REF:' + str(path))
    return path


def checked_json(item: dict) -> dict:
    checked_file(item)
    return old._read_ref(item)


def current_state() -> tuple[str, str]:
    head = old._clean_head()
    branch = subprocess.check_output(('git', 'branch', '--show-current'),
                                     cwd=REPO, text=True).strip()
    need(branch == 'work/gx1-current', 'BRANCH')
    return head, branch


def inspect(plan_ref: dict, *, require_policy: bool) -> dict:
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    head, _ = current_state()
    need(plan_ref == ref(PLAN_PATH) and not OUT_ROOT.exists(), 'PLAN_OR_OUTPUT')
    plan = json.loads(PLAN_PATH.read_text())
    expected_scope = {'fold_id': 2, 'partition': 'fit',
                      'sample_parent_rows': PARENT_ROWS,
                      'sample_issue_ns': ISSUE_NS,
                      'pass_index': PASS_INDEX, 'seed': SEED,
                      'selected_state_endpoints': True,
                      'physical_model_forwards': 1, 'backward_calls': 1,
                      'optimizer_steps': 0, 'witness_F_weight': WITNESS_F_WEIGHT,
                      'training_authorized': False, 'test_data_used': False}
    need(set(plan) == {'schema_version', 'stage', 'status', 'source_base_head',
                       'source_owners', 'inputs', 'scope', 'limits', 'output_root',
                       'command_template', 'expected_view_sha256'}
         and plan['schema_version'] == 'gx1_market_state_native_F_cpu_zero_step_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['output_root'] == str(OUT_ROOT)
         and plan['command_template'] == planned_command('{plan_sha256}')
         and plan['scope'] == expected_scope
         and plan['limits'] == {'memory_max': MEMORY_BYTES,
                                'memory_high': MEMORY_BYTES,
                                'swap': SWAP_BYTES, 'wall_seconds': WALL_SECONDS}
         and subprocess.run(('git', 'merge-base', '--is-ancestor',
                             plan['source_base_head'], head), cwd=REPO,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode == 0,
         'PLAN_SCOPE')
    need(set(plan['source_owners']) == set(SOURCE_PATHS)
         and all(sha(REPO / path) == digest
                 for path, digest in plan['source_owners'].items()),
         'SOURCE_OWNERS')
    inputs = plan['inputs']
    view_plan = checked_json(inputs['fold2_view_plan'])
    need(inputs['fold2_view_plan'] == ref(FOLD2_VIEW_PLAN)
         and inputs['fold2_view_result'] == ref(FOLD2_ROOT / 'RESULT.json')
         and inputs['fold2_view_validation'] == ref(FOLD2_ROOT / 'VALIDATION.json')
         and inputs['fold2_packet'] == view_plan['inputs']['packet']
         and inputs['fold2_constructor'] == view_plan['inputs']['constructor']
         and inputs['fold2_prior_ta_view'] == view_plan['inputs']['view']
         and inputs['fold2_sidecar_result'] == view_plan['inputs']['physical_result']
         and inputs['fold2_sidecar_terminal'] == view_plan['inputs']['physical_terminal']
         and inputs['selected_S_mask'] ==
             {'path': selected_state.REFS['mask'][0],
              'sha256': selected_state.REFS['mask'][1]}
         and inputs['F_result'] ==
             {'path': str(future.REFS['result'][0]),
              'sha256': future.REFS['result'][1]}
         and inputs['F_tape'] ==
             {'path': str(future.REFS['tape'][0]),
              'sha256': future.REFS['tape'][1]},
         'INPUT_REFS')
    completed = checked_json(inputs['fold2_view_result'])
    validation = checked_json(inputs['fold2_view_validation'])
    packet = checked_json(inputs['fold2_packet'])
    constructor = checked_json(inputs['fold2_constructor'])
    prior_view = checked_json(inputs['fold2_prior_ta_view'])
    physical = checked_json(inputs['fold2_sidecar_result'])
    terminal = checked_json(inputs['fold2_sidecar_terminal'])
    physical_plan = checked_json(physical['plan'])
    old_span_plan = checked_json(physical_plan['evidence']['input_plan'])
    provider_plan = checked_json(mature.INPUT_REFS['provider_plan'])
    checked_file(mature.INPUT_REFS['provider_worker'])
    need(completed.get('status') == 'COMPLETE_FULL_FOLD2_TRAIN_INPUT_GATHER'
         and validation.get('status') == 'PASS_FULL_FOLD2_TRAIN_INPUT_GATHER'
         and validation.get('actual_gather_qualified') is True
         and completed.get('test_data_used') is False
         and physical.get('status') == terminal.get('status') == 'COMPLETE'
         and terminal.get('exit_code') == 0
         and physical.get('test_data_used') is False
         and physical_plan['scope']['fold_id'] == 2
         and physical_plan['scope']['parent_rows'] == 47338
         and view_plan['counts']['parent_rows'] == 47338
         and view_plan['inputs']['provider_plan'] == mature.INPUT_REFS['provider_plan']
         and set(physical['surfaces']) == {'local','M5','M15','H1','H4','D1'}
         and all(item['all_overlap_columns_bitexact']
                 for item in physical['surfaces'].values()),
         'FOLD2_INPUT_LINEAGE')
    for lane in ('local','M5','M15','H1','H4','D1'):
        checked_file(physical['surfaces'][lane]['artifact'])
    for lane in ('M5','M15','H1','H4','D1'):
        for kind in ('feats','ts'):
            checked_file(old_span_plan['caches'][lane][kind])
    sources = {key: dict(item) for key, item in prior_view['source_bindings'].items()}
    bridge = old.HISTORICAL_RUNTIME_BRIDGE['gx1/contracts/entry_gc_mtf_history_v1.py']
    need(sources['gc_gather_source']['sha256'] == bridge[0]
         and plan['source_owners']['gx1/contracts/entry_gc_mtf_history_v1.py'] == bridge[1],
         'HISTORICAL_GC_BRIDGE')
    for key, item in sources.items():
        path = Path(item['path'])
        if path.is_relative_to(REPO) and key != 'gc_gather_source':
            relative = str(path.relative_to(REPO))
            need(relative in plan['source_owners'], 'CURRENT_VIEW_SOURCE:' + key)
            sources[key] = {'path': str(path),
                            'sha256': plan['source_owners'][relative]}
        elif key != 'gc_gather_source':
            checked_file(item)
    view = views.bind_native_direction_feature_view(
        prefix_normalization=packet,
        prefix_normalization_ref=inputs['fold2_packet'],
        constructor_metadata=constructor,
        constructor_metadata_ref=inputs['fold2_constructor'],
        source_bindings=sources, max_batch_rows=16,
        full_feature_profile=True, market_state_time_query_v2=True)
    need(view['contract_sha256'] == plan['expected_view_sha256']
         and view['physical_widths'] ==
             {'signal':2727,'ctx_cont':2542,'ctx_cat':1,'mtf':255}
         and 'full_ta_feature_profile' not in view
         and 'unobserved_units_profile' not in view,
         'FULL_GC_VIEW')
    state_bound = selected_state.load_bound_five_endpoint_targets()
    future_bound = future.load_bound_future_displacement()
    selected = np.asarray(PARENT_ROWS, dtype=np.int64)
    issues = np.asarray(ISSUE_NS, dtype=np.int64)
    state_truth = state_bound.gather(selected, issues, fold_id=2, partition='fit')
    future_truth = future_bound.gather(selected, issues, fold_id=2, partition='fit')
    need(state_truth['allowed'].tolist() == [[5,5,1,5,5],[1,1,1,5,5]]
         and future_truth['allowed'].tolist() == [[2,2,1,1,4],[1,1,1,1,4]]
         and bool(state_truth['eligible'].all() and state_truth['mature'].all()
                  and future_truth['eligible'].all() and future_truth['mature'].all()),
         'TARGET_PAIR')
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
             and gate.get('worker') == ref(REPO / WORKER_PATH)
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
    return {'plan':plan,'plan_ref':plan_ref,'policy_ref':policy_ref,
            'view':view,'packet':packet,'constructor':constructor,
            'physical':physical,'physical_plan':physical_plan,
            'old_span_plan':old_span_plan,'provider_plan':provider_plan,
            'state_bound':state_bound,'future_bound':future_bound}


def preflight(plan_ref: dict) -> dict:
    checked = inspect(plan_ref, require_policy=False)
    return {'status': 'READY_NO_CLAIM_NO_FEATURE_ARRAY_DECODE',
            'plan': plan_ref,
            'source_owners_checked': len(checked['plan']['source_owners']),
            'view_sha256': checked['view']['contract_sha256'],
            'model_contract_sha256': dual_model.contract()['contract_sha256'],
            'objective_contract_sha256': objective.contract()['contract_sha256'],
            'fold_id': 2, 'fit_rows': 47338,
            'state_target_mode': 'selected_native_endpoints',
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False}


def witness(checked: dict) -> dict:
    import torch
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts.entry_exit_feature_surface_v1 import load_m1_feature_surface_times
    from gx1.contracts.entry_gc_mtf_history_v1 import load_gc_mtf_history_raw_artifact
    from gx1.contracts.entry_anchor_span_feature_inputs_v1 import (
        build_anchor_span_inputs, TIMEFRAMES,
    )
    from gx1.contracts.market_state_time_query_native_F_batch_v1 import (
        FEATURE_KEYS, evaluate_fit_batch,
    )
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    view, packet = checked['view'], checked['packet']
    provider_plan = checked['provider_plan']
    provider = mature._provider_module()
    info = provider.validate_metadata(provider_plan)
    axes, _ = provider.read_axes(provider_plan, info, np)
    for key in ('parent_rows', 'bar_start_ns', 'decision_ns'):
        values = axes[key]
        need(values.dtype == np.int64 and values.shape == (view['full_axes']['rows'],)
             and hashlib.sha256(values.astype('<i8', copy=False).tobytes()).hexdigest()
                 == view['full_axes'][key + '_sha256']
             and hashlib.sha256(values[:47338].astype('<i8', copy=False).tobytes()).hexdigest()
                 == view['prefix'][key + '_sha256'], 'REAL_AXIS:' + key)
        values.setflags(write=False)
    selected = np.asarray(PARENT_ROWS, dtype=np.int64)
    issues = np.asarray(ISSUE_NS, dtype=np.int64)
    at = np.searchsorted(axes['parent_rows'], selected)
    need(np.array_equal(axes['parent_rows'][at], selected)
         and np.array_equal(axes['decision_ns'][at], issues)
         and int(at[-1]) < 47338
         and bool(np.all(axes['decision_ns'][:47338] < view['prefix']['cutoff_ns'])),
         'REAL_FEATURE_ROWS')
    groups = provider.map_reuse(info['reuse'], np)
    source, current = groups['source050'], groups['current051']
    old_span_plan = checked['old_span_plan']
    need(provider.regular(old_span_plan['local_clock_path'])
         == old_span_plan['local_clock_stat'], 'LOCAL_CLOCK_STAT')
    times = load_m1_feature_surface_times(
        Path(old_span_plan['local_clock_path']),
        context='NATIVE_F_SELECTED_FOLD2_TRAIN_CLOCK_ONLY',
        expected_bar_seconds=300).asi8
    positions = np.searchsorted(times, axes['bar_start_ns']).astype(np.int64)
    need(np.all(positions < len(times))
         and np.array_equal(times[positions], axes['bar_start_ns']),
         'LOCAL_POSITIONS')
    for array in (times, positions, source['signal'], current['snap'],
                  current['ctx_cont'], current['ctx_cat']):
        array.setflags(write=False)
    sequence = dict(signal=source['signal'], times_ns=times,
                    sample_positions=positions, audit=info['audit'])
    physical = checked['physical']
    payloads, values, clocks = {}, {}, {}
    for lane in ('local', *TIMEFRAMES):
        sidecar = physical['surfaces'][lane]['artifact']
        checked_file(sidecar)
        with np.load(sidecar['path'], allow_pickle=False) as archive:
            payloads[lane] = {key: archive[key] for key in archive.files}
        if lane != 'local':
            cache = old_span_plan['caches'][lane]
            for kind in ('feats', 'ts'):
                checked_file(cache[kind])
            values[lane] = np.load(cache['feats']['path'],
                                   mmap_mode='r', allow_pickle=False)
            clocks[lane] = np.load(cache['ts']['path'],
                                   mmap_mode='r', allow_pickle=False)
    parent = packet['parent_prefix_normalization']['base_normalization']
    bound = build_anchor_span_inputs(
        view, entry_sequence_source=sequence,
        snap_x=current['snap'], ctx_cont=current['ctx_cont'],
        ctx_cat=current['ctx_cat'],
        parent_signal_names=parent['surfaces']['signal']['field_names'],
        parent_mtf_names=parent['surfaces']['mtf_m5']['field_names'],
        parent_mtf_values=values, parent_mtf_times=clocks,
        span_payloads=payloads,
        physical_populations={lane: surface['population']
                              for lane, surface in physical['surfaces'].items()},
        unobserved_units_profile=None)
    gc = load_gc_mtf_history_raw_artifact(provider_plan['gc_surfaces'])
    need(gc['layout'] == mature._receipt_chain()['gc_layout']
         and gc['layout']['contract_sha256']
             == view['prefix_normalization']['gc_layout_sha256'],
         'CURRENT_GC_LAYOUT')
    raw = views.gather_native_direction_feature_batch(
        view, parent_rows=axes['parent_rows'],
        bar_start_ns=axes['bar_start_ns'], decision_ns=axes['decision_ns'],
        entry_sequence_source=sequence, snap_x=current['snap'],
        ctx_cont=current['ctx_cont'], ctx_cat=current['ctx_cat'],
        multi_tf_sources=bound.mtf_sources, gc_surfaces=gc['surfaces'],
        gc_layout=gc['layout'], batch_parent_rows=selected,
        unobserved_units_profile=None, anchor_span_inputs=bound)
    need(set(raw['features']) == FEATURE_KEYS
         and raw['witness']['parent_rows'] == PARENT_ROWS
         and raw['witness']['decision_ns'] == ISSUE_NS,
         'FULL_GC_FEATURE_BATCH')
    for array in (*source.values(), *current.values(),
                  *values.values(), *clocks.values()):
        provider.flush_pages(array)
    for surface in gc['surfaces'].values():
        for key in ('native_row_indices', 'available_ns', 'features'):
            provider.flush_pages(surface[key])
    del groups, source, current, values, clocks, gc, payloads, bound, sequence
    device = torch.device('cpu')
    need(not torch.cuda.is_initialized(), 'NO_CUDA')
    torch.manual_seed(SEED)
    state_model = views.build_native_direction_research_model(
        view, prefix_normalization=packet,
        constructor_metadata=checked['constructor'], device=device,
        unobserved_units_profile=None)
    need(type(state_model).__name__ == 'FactorizedTimeQueryMarketStateModel'
         and state_model.feature_view_identity['contract_sha256']
             == view['contract_sha256'], 'FRESH_FULL_GC_FOLD2_MODEL')
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
            model, tensors, checked['state_bound'], checked['future_bound'],
            selected, issues, fold_id=2, pass_index=PASS_INDEX, seed=SEED,
            future_weight=WITNESS_F_WEIGHT, selected_state_endpoints=True)
    handle.remove()
    need(len(captured) == 1 and outcome['state_count'] == 10
         and outcome['future_count'] == 10
         and outcome['state_query_k'].tolist() == [[1, 3, 12, 48, 288]] * 2
         and outcome['state_target_mode'] == 'selected_native_endpoints'
         and outcome['state_allowed'].tolist()
             == [[5, 5, 1, 5, 5], [1, 1, 1, 5, 5]]
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
            'scope_limit': 'Two fixed selected native-endpoint TRAIN rows; no fit, OOF, calibration, direction value, Entry or economic authority.',
            'sample_parent_rows': PARENT_ROWS,
            'sample_issue_ns': ISSUE_NS,
            'state_query_k': outcome['state_query_k'].tolist(),
            'state_target_mode': outcome['state_target_mode'],
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
    checked = inspect(plan_ref, require_policy=True)
    plan, policy_ref = checked['plan'], checked['policy_ref']
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
            result = witness(checked)
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
