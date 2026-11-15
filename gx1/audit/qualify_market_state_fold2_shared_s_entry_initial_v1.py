"""One-use genuine Fold2 B16 shared S/Entry read-only gradient witness.

One native encoder feeds the fixed S query and existing 1..480 minute
two-side Entry heads. Actual cost/path targets are target-only. No optimizer,
checkpoint, TEST, broker, trading or model promotion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from gx1.audit import measure_market_state_fold2_exact_prior_later_v1 as prior
from gx1.audit import qualify_market_state_fold2_multiminute_initial_v1 as initial
from gx1.audit.qualify_market_state_fold2_initial_mask_observation_v1 import classify_pairs
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts import market_state_fold2_selected_native_batches_v1 as batches
from gx1.contracts.entry_causal_m1_outcomes_v1 import query_dynamic_m1_prefix
from gx1.contracts.entry_observed_market_v1 import (
    build_entry_observed_market_targets, read_dynamic_m5_target_source,
    require_entry_observed_market_scope, require_entry_only_gradients,
)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
from gx1.contracts.market_state_fold2_multiminute_path_v1 import (
    QUERY_SHA256, fixed_query_grid,
)
from gx1.contracts.market_state_minute_query_candidate_v1 import decode_raw_minute_grid
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.features.gc_native_context_v1 import masking_pairs
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

REPO = prior.REPO
POLICY = prior.POLICY
STAGE = 'market_state_fold2_shared_s_entry_initial_20261115_001'
PLAN = REPO / 'configs/research/MARKET_STATE_FOLD2_SHARED_S_ENTRY_INITIAL_20261115_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_FOLD2_SHARED_S_ENTRY_INITIAL_20261115_001')
HORIZONS = (1, 5, 15, 60, 240, 480)
CODE = (
    'gx1/audit/qualify_market_state_fold2_shared_s_entry_initial_v1.py',
    'gx1/audit/qualify_market_state_fold2_multiminute_initial_v1.py',
    'gx1/audit/qualify_market_state_fold2_initial_mask_observation_v1.py',
    'gx1/contracts/market_state_fold2_selected_native_batches_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/contracts/entry_causal_m1_outcomes_v1.py',
    'gx1/contracts/entry_observed_market_v1.py',
    'gx1/contracts/market_state_time_query_model_v2.py',
    'gx1/contracts/market_state_minute_query_candidate_v1.py',
    'gx1/contracts/market_state_fold2_multiminute_path_v1.py',
    'gx1/contracts/model_state_digest_v1.py',
    'gx1/features/gc_native_context_v1.py',
    'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py',
    'gx1/models/entry_v10/entry_v10_ctx_train_v3.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'scripts/gx1_capped_run.sh',
)
LIMITS = {'class': 'producer', 'memory': '12G', 'swap': '512M',
          'wall_seconds': 1800, 'max_invocations': 1}
SCOPE = {'fold_id': 2, 'partition': 'fit', 'batch_size': 16, 'batch_index': 0,
         'S_query_sha256': QUERY_SHA256, 'entry_horizon_minutes': list(HORIZONS),
         'native_encoder_forwards': 1, 'S_head_decodes': 1,
         'entry_head_decodes': 1, 'backward_calls': 1,
         'model_constructions': 1, 'optimizer_steps': 0,
         'checkpoint_reads': 0, 'test_data_used': False}


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('FOLD2_SHARED_S_ENTRY_INITIAL_' + reason)


def axis_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def common_target_probe(pred_net: torch.Tensor, pred_mae: torch.Tensor,
                        actual_net: torch.Tensor, actual_mae: torch.Tensor,
                        common: torch.Tensor) -> torch.Tensor:
    need(pred_net.shape == pred_mae.shape == actual_net.shape == actual_mae.shape
         and pred_net.ndim == 3 and pred_net.shape[-1] == 2
         and common.shape == pred_net.shape[:2]
         and bool(common.any()), 'PROBE_SHAPE')
    selected = common.unsqueeze(-1).expand_as(pred_net)
    need(bool(torch.isfinite(pred_net).all() and torch.isfinite(pred_mae).all())
         and bool(torch.isfinite(actual_net[selected]).all())
         and bool(torch.isfinite(actual_mae[selected]).all())
         and bool((actual_mae[selected] >= 0).all()), 'PROBE_DOMAIN')
    return ((pred_net[selected] - actual_net[selected]).square().mean()
            + (pred_mae[selected] - actual_mae[selected]).square().mean()) / 2


def preflight(plan_sha: str) -> tuple:
    need(Path.cwd().resolve() == REPO
         and prior.git('branch', '--show-current') == 'work/gx1-current'
         and prior.git('status', '--porcelain') == '', 'CLEAN_SOURCE')
    head = prior.git('rev-parse', 'HEAD')
    need(not ROOT.exists() and prior.sha(PLAN) == plan_sha, 'PLAN_ROOT')
    plan = json.loads(PLAN.read_text())
    need(plan.get('schema_version') == 'gx1_fold2_shared_s_entry_initial_plan_v1'
         and plan.get('status') == 'ADMITTED_READ_ONLY'
         and plan.get('stage') == STAGE
         and plan.get('source_base_head') == '94132f94283d01eb60d7826e3e5d96d96704d0e0'
         and plan.get('output_root') == str(ROOT)
         and plan.get('scope') == SCOPE and plan.get('limits') == LIMITS
         and plan.get('seed') == 51, 'PLAN')
    need(subprocess.run(('git', 'merge-base', '--is-ancestor',
                         plan['source_base_head'], head), cwd=REPO).returncode == 0,
         'ANCESTOR')
    need(set(plan.get('source_owners', {})) == set(CODE)
         and all(prior.sha(REPO / name) == digest
                 for name, digest in plan['source_owners'].items()), 'SOURCE_HASH')
    need(all(Path(ref['path']).is_file()
             and prior.sha(Path(ref['path'])) == ref['sha256']
             for ref in plan['input_refs'].values()), 'INPUT_HASH')
    initial_result = json.loads(Path(plan['input_refs']['S_initial_result']['path']).read_text())
    mask_result = json.loads(Path(plan['input_refs']['mask_result']['path']).read_text())
    all_side = json.loads(Path(plan['input_refs']['all_side_result']['path']).read_text())
    terminal = json.loads(Path(plan['input_refs']['all_side_terminal']['path']).read_text())
    selected_recipe = json.loads(Path(plan['input_refs']['selected_recipe']['path']).read_text())
    all_side_plan = json.loads(Path(plan['input_refs']['all_side_plan']['path']).read_text())
    need(initial_result['status'] == mask_result['status'] == all_side['status']
         == terminal['status'] == 'COMPLETE'
         and terminal['exit_code'] == 0
         and terminal['result'] == plan['input_refs']['all_side_result']
         and initial_result['batch_parent_rows_sha256']
             == mask_result['batch_parent_rows_sha256']
         and all_side_plan['input_refs']['recipe']
             == plan['input_refs']['cost_recipe']
         and all_side_plan['scope']['horizon_minutes'] == list(HORIZONS)
         and all_side_plan['scope']['fit_rows'] == 47338
         and all_side_plan['scope']['later_rows'] == 4096
         and selected_recipe['fold_id'] == 2
         and selected_recipe['partition'] == 'fit'
         and all(Path(ref['path']).is_file()
                 and prior.sha(Path(ref['path'])) == ref['sha256']
                 for ref in selected_recipe['source_owners'].values()),
         'INPUT_LINEAGE')
    cost_recipe = json.loads(Path(plan['input_refs']['cost_recipe']['path']).read_text())
    cost_scope = require_entry_observed_market_scope(cost_recipe)
    need(cost_scope['economics'] == all_side['economics']
         and cost_scope['training_phase'] == 'entry_only_dynamic_m5_path',
         'COST_SCOPE')
    policy = json.loads(POLICY.read_text())
    expected = {'status': 'ADMITTED_READ_ONLY', 'plan': prior.ref(PLAN),
                'output_root': str(ROOT), 'max_invocations': 1,
                'claimed_invocations': 0, 'completed_invocations': 0,
                'data_reads_authorized': True, 'model_constructions_authorized': 1,
                'encoder_forwards_authorized': 1, 'S_head_decodes_authorized': 1,
                'entry_head_decodes_authorized': 1, 'backward_calls_authorized': 1,
                'optimizer_steps_authorized': 0, 'checkpoint_reads_authorized': 0,
                'training_authorized': False, 'test_authorized': False,
                'broker_access_allowed': False, 'relaunch_allowed': False}
    need(policy['training_enabled'] is False
         and policy['native_training_enabled'] is False
         and policy[STAGE] == expected, 'POLICY')
    return head, prior.ref(POLICY), plan, initial_result, mask_result, selected_recipe, cost_scope


def gather_targets(rows: np.ndarray, issues: np.ndarray, cost_scope: dict) -> tuple:
    source = read_dynamic_m5_target_source(cost_scope, split='train')
    bar_ns = pd.DatetimeIndex(pd.to_datetime(
        source['frame']['time'].iloc[rows], utc=True)).as_unit('ns').asi8
    need(np.array_equal(bar_ns + 300_000_000_000, issues), 'M5_CLOCK')
    net = np.zeros((16, len(HORIZONS), 2), dtype=np.float32)
    mae = np.zeros_like(net)
    common = np.zeros((16, len(HORIZONS)), dtype=bool)
    for j, horizon in enumerate(HORIZONS):
        obs = query_dynamic_m1_prefix(
            source['index'], fill_surface=source['fill'],
            query_entry_rows=rows,
            horizon_minutes=np.full(16, horizon, dtype=np.int64),
            endpoint_kind='terminal_open_inclusive')
        need(np.array_equal(obs['query_entry_rows'], rows)
             and np.array_equal(obs['terminal_time_ns'],
                                issues + horizon * 60_000_000_000)
             and bool(np.all(obs['terminal_time_ns'] < 1735689600000000000)),
             'TARGET_CLOCK')
        term, path = obs['terminal_gross_observed'], obs['path_observed']
        need(term.shape == path.shape == (16, 2)
             and np.array_equal(term[:, 0], term[:, 1])
             and np.array_equal(path[:, 0], path[:, 1]), 'TARGET_MASK')
        valid = term[:, 0] & path[:, 0]
        common[:, j] = valid
        if np.any(valid):
            target = build_entry_observed_market_targets(
                gross_return_bps=torch.as_tensor(
                    obs['terminal_gross_return_bps'][valid], dtype=torch.float64),
                elapsed_wall_clock_seconds=torch.as_tensor(
                    obs['elapsed_wall_seconds'][valid], dtype=torch.int64),
                economics=cost_scope['economics'])['target_bps'].numpy()
            net[valid, j, :] = target[:, :2]
            mae[valid, j, :] = obs['mae_before_first_mfe_upper_bps'][valid]
    need(bool(common.any()) and np.isfinite(net).all()
         and np.isfinite(mae).all() and bool(np.all(mae >= 0)),
         'TARGETS')
    del source
    gc.collect()
    return net, mae, common


def measure(head: str, policy_ref: dict, plan: dict,
            initial_result: dict, mask_result: dict,
            selected_recipe: dict, cost_scope: dict) -> dict:
    view, packet, constructor, raw_batches, reports, provider, provider_plan = (
        batches.load_two_batches(source_owners=selected_recipe['source_owners']))
    need(len(raw_batches) == 2 and len(raw_batches[0][1]) == 16,
         'B16_SOURCE')
    features, rows, issues = raw_batches[0]
    need(axis_sha(rows) == initial_result['batch_parent_rows_sha256']
         and axis_sha(issues) == initial_result['batch_issue_ns_sha256']
         and reports == initial_result['input_reports']
         and view['contract_sha256'] == initial_result['view_sha256'],
         'SAME_B16')
    actual_net, actual_mae, common = gather_targets(rows, issues, cost_scope)
    torch.manual_seed(plan['seed'])
    model = views.build_native_direction_research_model(
        view, prefix_normalization=packet, constructor_metadata=constructor,
        device=torch.device('cpu'), unobserved_units_profile=None)
    need(type(model).__name__ == 'FactorizedTimeQueryMarketStateModel'
         and model.feature_view_identity['contract_sha256'] == view['contract_sha256'],
         'MODEL')
    model.train()
    state_before = canonical_model_state_sha256(model.state_dict())
    need(state_before == initial_result['initial_state_sha256'],
         'SAME_INITIAL_STATE')
    tensors = {}
    for key, value in features.items():
        tensor = torch.from_numpy(value).to(torch.device('cpu'))
        if tensor.is_floating_point():
            tensor.requires_grad_(True)
        tensors[key] = tensor
    query = torch.from_numpy(fixed_query_grid(np.asarray(issues, np.int64)))
    with trainer._training_mha_fastpath_context(model.backbone), \
         trainer._training_autocast_context(), \
         trainer._training_model_finite_check_context():
        hidden = model.encode_features(**tensors)
        S_raw = decode_raw_minute_grid(model, hidden, query)
        entry = model.backbone._decode_dynamic_m5_curves(
            hidden, query_chunk_size=64)
        pred_net = entry['dynamic_terminal_net_markout_bps'][
            :, [h - 1 for h in HORIZONS], :2]
        pred_mae = entry['dynamic_pre_mfe_mae_upper_bps'][
            :, [h - 1 for h in HORIZONS], :]
        probe = common_target_probe(
            pred_net, pred_mae,
            torch.from_numpy(actual_net), torch.from_numpy(actual_mae),
            torch.from_numpy(common))
        probe.backward()
    need(S_raw.shape == (16, 5, 5, 2)
         and pred_net.shape == pred_mae.shape == (16, 6, 2)
         and bool(torch.isfinite(S_raw).all())
         and bool(torch.isfinite(probe)), 'FORWARD_BACKWARD')
    require_entry_only_gradients(
        model.backbone, entry_training_phase='entry_only_dynamic_m5_path')
    state_after = canonical_model_state_sha256(model.state_dict())
    need(state_before == state_after, 'STATE_CHANGED')
    gradients = {key: initial.gradient_summary(value, feature_channels=True)
                 for key, value in tensors.items() if value.is_floating_point()}
    params = {
        'entry_net_head': initial.gradient_summary(
            model.backbone.head_dynamic_terminal_net_markout.weight,
            feature_channels=False),
        'entry_mae_head': initial.gradient_summary(
            model.backbone.head_dynamic_pre_mfe_mae_upper.weight,
            feature_channels=False),
        'backbone_joint': initial.gradient_summary(
            model.backbone.entry_q_joint_in.weight, feature_channels=False),
        'S_query_head': initial.gradient_summary(
            model.query_head.weight, feature_channels=False),
    }
    need(all(v['present'] and v['finite'] and v['nonzero_elements'] > 0
             for k, v in params.items() if k != 'S_query_head')
         and not params['S_query_head']['present'], 'HEAD_GRADIENT')
    fields, _, widths = views._canonical()
    need(widths == view['physical_widths'], 'FIELD_WIDTH')
    layout = packet['gc_mtf_history']['layout']
    mask_checks = {}
    for key, names, pairs in (
            ('seq_x', fields['signal'], masking_pairs(fields['signal'])),
            ('ctx_cont', fields['ctx_cont'], masking_pairs(fields['ctx_cont']))):
        mask_checks[key] = classify_pairs(
            features[key], names, pairs,
            set(gradients[key]['zero_channel_indices']))
    for lane in ('M5', 'M15', 'H1', 'H4', 'D1'):
        key = 'gc_seq_' + lane.lower()
        mask_checks[key] = classify_pairs(
            features[key], layout['field_names'][lane],
            tuple(tuple(pair) for pair in layout['value_mask_pairs'][lane]),
            set(gradients[key]['zero_channel_indices']))
    observed_zero = sum(len(v['zero_gradient_observed_pairs'])
                        for v in mask_checks.values())
    provider.validate_metadata(provider_plan)
    return {
        'schema_version': 'gx1_fold2_shared_s_entry_initial_result_v1',
        'status': 'COMPLETE', 'stage': STAGE, 'source_head': head,
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'plan': prior.ref(PLAN), 'policy': policy_ref,
        'S_initial_result': plan['input_refs']['S_initial_result'],
        'mask_result': plan['input_refs']['mask_result'],
        'batch_parent_rows_sha256': axis_sha(rows),
        'batch_issue_ns_sha256': axis_sha(issues),
        'view_sha256': view['contract_sha256'],
        'initial_state_sha256': state_before,
        'final_state_sha256': state_after,
        'S_query_sha256': QUERY_SHA256,
        'entry_horizon_minutes': list(HORIZONS),
        'common_target_rows_by_horizon': common.sum(axis=0).tolist(),
        'cost_economics_sha256': cost_scope['economics']['economics_sha256'],
        'probe_raw_bps_mse': float(probe.detach().item()),
        'input_gradients': gradients, 'parameter_gradients': params,
        'observed_paired_value_zero_gradient_count': observed_zero,
        'mask_checks': mask_checks,
        'model_constructions': 1, 'native_encoder_forwards': 1,
        'S_head_decodes': 1, 'entry_head_decodes': 1,
        'backward_calls': 1, 'optimizer_steps': 0,
        'checkpoint_reads': 0, 'test_data_used': False,
        'trade_selection': False, 'cuda_initialized': torch.cuda.is_initialized(),
    }


def run(plan_sha: str) -> None:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_research_producer_execution,
        require_capped_lock_ancestry)
    head, policy_ref, plan, initial_result, mask_result, selected_recipe, cost_scope = (
        preflight(plan_sha))
    cap = require_capped_cpu_research_producer_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 12 * 1024**3
         and cap['swap'] == 512 * 1024**2 and lock is not None, 'CAP')
    ROOT.mkdir(parents=False, exist_ok=False)
    _publish_bytes(ROOT / 'CLAIM.json', prior.encode({
        'schema_version': 'gx1_fold2_shared_s_entry_initial_claim_v1',
        'stage': STAGE, 'plan': prior.ref(PLAN),
        'policy': policy_ref, 'source_head': head, 'test_data_used': False}))
    claim = prior.ref(ROOT / 'CLAIM.json')
    started = time.monotonic()
    try:
        result = measure(
            head, policy_ref, plan, initial_result, mask_result,
            selected_recipe, cost_scope)
        result['claim'] = claim
        result['elapsed_seconds'] = time.monotonic() - started
        _publish_bytes(ROOT / 'RESULT.json', prior.encode(result))
        _publish_bytes(ROOT / 'TERMINAL.json', prior.encode({
            'schema_version': 'gx1_fold2_shared_s_entry_initial_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'stage': STAGE,
            'claim': claim, 'result': prior.ref(ROOT / 'RESULT.json'),
            'source_head': head, 'test_data_used': False}))
    except BaseException as exc:
        _publish_bytes(ROOT / 'TERMINAL.json', prior.encode({
            'schema_version': 'gx1_fold2_shared_s_entry_initial_terminal_v1',
            'status': 'FAILED_CONSUMED', 'exit_code': 1, 'stage': STAGE,
            'claim': claim, 'result': None, 'source_head': head,
            'reason': type(exc).__name__ + ':' + str(exc),
            'test_data_used': False}))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha', required=True)
    run(parser.parse_args().plan_sha)


if __name__ == '__main__':
    main()
