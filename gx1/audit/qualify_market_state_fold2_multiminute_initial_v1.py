"""One-use genuine Fold 2 full-GC B16 initial S forward/backward witness.

The fixed five-point path and exact support loss are read-only model mechanics:
no optimizer, checkpoint, TEST, broker, trade or model promotion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from gx1.audit import measure_market_state_fold2_exact_prior_later_v1 as prior
from gx1.audit import build_market_state_time_query_tape_v1 as producer
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts import market_state_fold2_selected_native_batches_v1 as batches
from gx1.contracts.market_state_fold2_multiminute_path_v1 import (
    QUERY_MINUTES, QUERY_SHA256, fixed_query_grid, gather_fixed_path,
    score_fixed_path,
)
from gx1.contracts.market_state_fold2_multiminute_side_loss_v1 import (
    allowed_set_loss, contract as loss_contract,
)
from gx1.contracts.market_state_minute_query_candidate_v1 import (
    MinuteQueryTargetTape, decode_raw_minute_grid,
)
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

REPO = prior.REPO
POLICY = prior.POLICY
STAGE = 'market_state_fold2_multiminute_initial_20261115_001'
PLAN = REPO / 'configs/research/MARKET_STATE_FOLD2_MULTIMINUTE_INITIAL_20261115_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_FOLD2_MULTIMINUTE_INITIAL_20261115_001')
CODE = (
    'gx1/audit/qualify_market_state_fold2_multiminute_initial_v1.py',
    'gx1/contracts/market_state_fold2_multiminute_path_v1.py',
    'gx1/contracts/market_state_fold2_multiminute_side_loss_v1.py',
    'gx1/contracts/market_state_minute_query_candidate_v1.py',
    'gx1/contracts/market_state_fold2_selected_native_batches_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/contracts/market_state_time_query_model_v2.py',
    'gx1/contracts/market_state_time_query_target_v1.py',
    'gx1/contracts/market_state_time_query_scorer_v2.py',
    'gx1/contracts/market_state_five_endpoint_contract_v1.py',
    'gx1/contracts/model_state_digest_v1.py',
    'gx1/audit/build_market_state_time_query_tape_v1.py',
    'gx1/models/entry_v10/entry_v10_ctx_train_v3.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'scripts/gx1_capped_run.sh',
)
LIMITS = {'class': 'producer', 'memory': '12G', 'swap': '512M',
          'wall_seconds': 1800, 'max_invocations': 1}
SCOPE = {
    'fold_id': 2, 'partition': 'fit', 'batch_size': 16, 'batch_index': 0,
    'query_minutes': QUERY_MINUTES.tolist(), 'query_sha256': QUERY_SHA256,
    'backbone_encode_calls': 1, 'head_decode_calls': 1,
    'loss_contract_sha256': loss_contract()['contract_sha256'],
    'model_constructions': 1, 'optimizer_steps': 0, 'checkpoint_reads': 0,
    'input_gradient_channels': True, 'test_data_used': False,
}


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('FOLD2_MULTIMINUTE_INITIAL_' + reason)


def gradient_summary(value: torch.Tensor, *, feature_channels: bool) -> dict:
    grad = value.grad
    if grad is None:
        return {'present': False, 'finite': None, 'nonzero_elements': 0,
                'feature_channels': value.shape[-1] if feature_channels else None,
                'nonzero_channels': 0 if feature_channels else None,
                'zero_channel_indices': list(range(value.shape[-1])) if feature_channels else None}
    finite = bool(torch.isfinite(grad).all())
    need(finite, 'NONFINITE_GRADIENT')
    result = {'present': True, 'finite': True,
              'nonzero_elements': int(torch.count_nonzero(grad).item()),
              'max_abs': float(grad.abs().max().item())}
    if feature_channels:
        active = torch.any(grad != 0, dim=tuple(range(grad.ndim - 1)))
        zero = torch.nonzero(~active).flatten().tolist()
        result.update(feature_channels=value.shape[-1],
                      nonzero_channels=int(active.sum().item()),
                      zero_channel_indices=zero)
    return result


def preflight(plan_sha: str) -> tuple:
    need(Path.cwd().resolve() == REPO
         and prior.git('branch', '--show-current') == 'work/gx1-current'
         and prior.git('status', '--porcelain') == '', 'CLEAN_SOURCE')
    head = prior.git('rev-parse', 'HEAD')
    need(not ROOT.exists() and prior.sha(PLAN) == plan_sha, 'PLAN_ROOT')
    plan = json.loads(PLAN.read_text())
    need(plan.get('schema_version') == 'gx1_fold2_multiminute_initial_plan_v1'
         and plan.get('stage') == STAGE and plan.get('status') == 'ADMITTED_READ_ONLY'
         and plan.get('source_base_head') == '4d583fd91dab29c1e2574cd0d957232dc312005d'
         and plan.get('output_root') == str(ROOT)
         and plan.get('scope') == SCOPE and plan.get('limits') == LIMITS
         and plan.get('seed') == 51, 'PLAN')
    need(subprocess.run(('git', 'merge-base', '--is-ancestor',
                         plan['source_base_head'], head), cwd=REPO).returncode == 0,
         'ANCESTOR')
    need(set(plan.get('source_owners', {})) == set(CODE)
         and all(prior.sha(REPO / path) == digest
                 for path, digest in plan['source_owners'].items()), 'SOURCE_HASH')
    need(all(Path(ref['path']).is_file()
             and prior.sha(Path(ref['path'])) == ref['sha256']
             for ref in plan['input_refs'].values()), 'INPUT_HASH')
    recipe = json.loads(Path(plan['input_refs']['recipe']['path']).read_text())
    need(recipe['run_scope'] == 'market_state_native_F_selected_fold2_two_batch_v1'
         and recipe['fold_id'] == 2 and recipe['partition'] == 'fit'
         and recipe['selected_state_endpoints'] is True
         and recipe['test_data_used'] is False
         and all(Path(ref['path']).is_file()
                 and prior.sha(Path(ref['path'])) == ref['sha256']
                 for ref in recipe['source_owners'].values()), 'RECIPE_SOURCE')
    policy = json.loads(POLICY.read_text())
    need(policy['training_enabled'] is False
         and policy['native_training_enabled'] is False
         and policy[STAGE] == {
             'status': 'ADMITTED_READ_ONLY', 'plan': prior.ref(PLAN),
             'output_root': str(ROOT), 'max_invocations': 1,
             'claimed_invocations': 0, 'completed_invocations': 0,
             'data_reads_authorized': True, 'model_constructions_authorized': 1,
             'backbone_forwards_authorized': 1, 'head_decodes_authorized': 1,
             'backward_calls_authorized': 1,
             'optimizer_steps_authorized': 0, 'checkpoint_reads_authorized': 0,
             'training_authorized': False, 'test_authorized': False,
             'broker_access_allowed': False, 'relaunch_allowed': False},
         'POLICY')
    return head, prior.ref(POLICY), plan, recipe


def measure(head: str, policy_ref: dict, plan: dict, recipe: dict) -> dict:
    need(not torch.cuda.is_initialized(), 'NO_CUDA_BEFORE')
    view, packet, constructor, raw_batches, reports, provider, provider_plan = (
        batches.load_two_batches(source_owners=recipe['source_owners']))
    need(len(raw_batches) == 2 and len(raw_batches[0][1]) == 16,
         'B16_SOURCE')
    features, rows, issues = raw_batches[0]
    issue_ns = np.asarray(issues, np.int64)
    need(len(rows) == 16
         and bool(np.all(issue_ns < 1735689600000000000)), 'FIT_ISSUES')
    torch.manual_seed(plan['seed'])
    model = views.build_native_direction_research_model(
        view, prefix_normalization=packet, constructor_metadata=constructor,
        device=torch.device('cpu'), unobserved_units_profile=None)
    need(type(model).__name__ == 'FactorizedTimeQueryMarketStateModel'
         and model.feature_view_identity['contract_sha256'] == view['contract_sha256'],
         'FRESH_FULL_FEATURE_MODEL')
    model.train()
    initial = canonical_model_state_sha256(model.state_dict())
    tensors = {}
    for key, value in features.items():
        tensor = torch.from_numpy(value).to(torch.device('cpu'))
        if tensor.is_floating_point():
            tensor.requires_grad_(True)
        tensors[key] = tensor
    need(len(tensors) == 14 and 'ctx_cat' in tensors, 'FEATURE_KEYS')
    tape_ref = plan['input_refs']['candidate_tape']
    with np.load(io.BytesIO(Path(tape_ref['path']).read_bytes()),
                 allow_pickle=False) as saved:
        need(set(saved.files) == producer.TAPE_KEYS, 'TAPE_KEYS')
        tape = MinuteQueryTargetTape(
            np.array(saved['validity_ns'], copy=True),
            np.array(saved['m1_source_row'], copy=True),
            np.array(saved['allowed_set'], copy=True),
            np.array(saved['path_observed'], copy=True),
            np.array(saved['gap_kind'], copy=True))
    truth = gather_fixed_path(tape, issue_ns,
                              fold_upper_ns=1735689600000000000)
    query = torch.from_numpy(fixed_query_grid(issue_ns))
    need(bool(np.array_equal(query.numpy(), truth['query_minutes'])),
         'QUERY_TARGET_CLOCK')
    with trainer._training_mha_fastpath_context(model.backbone), \
         trainer._training_autocast_context(), \
         trainer._training_model_finite_check_context():
        hidden = model.encode_features(**tensors)
        predicted = decode_raw_minute_grid(model, hidden, query)
        predicted.retain_grad()
        loss = allowed_set_loss(
            predicted, torch.from_numpy(truth['allowed']),
            torch.from_numpy(truth['eligible']),
            torch.from_numpy(truth['mature']), query)
        loss['loss'].backward()
    need(predicted.shape == (16, 5, 5, 2)
         and bool(torch.isfinite(predicted).all())
         and bool(torch.isfinite(loss['loss']))
         and predicted.grad is not None
         and bool(torch.isfinite(predicted.grad).all()), 'FORWARD_BACKWARD')
    side_grad = predicted.grad[..., 1]
    activity_grad = predicted.grad[..., 0]
    need(not bool(torch.any((side_grad != 0) & ~loss['side_known_mask']))
         and not bool(torch.any((activity_grad != 0) & ~loss['informative_mask'])),
         'UNKNOWN_GRADIENT')
    input_gradients = {
        key: gradient_summary(value, feature_channels=True)
        for key, value in tensors.items() if value.is_floating_point()}
    parameter_gradients = {
        'query_head': gradient_summary(model.query_head.weight, feature_channels=False),
        'query_projection': gradient_summary(
            model.query_projection[0].weight, feature_channels=False),
        'backbone_joint': gradient_summary(
            model.backbone.entry_q_joint_in.weight, feature_channels=False),
        'ctx_cat_embedding': gradient_summary(
            model.backbone.ctx_cat_embeddings[0].weight, feature_channels=False),
    }
    need(all(item['present'] and item['finite'] and item['nonzero_elements'] > 0
             for name, item in parameter_gradients.items()
             if name != 'ctx_cat_embedding'), 'PARAMETER_GRADIENTS')
    scores = score_fixed_path(predicted.detach().numpy(), truth)
    final = canonical_model_state_sha256(model.state_dict())
    need(initial == final and not torch.cuda.is_initialized(),
         'NO_UPDATE_OR_CUDA')
    provider.validate_metadata(provider_plan)
    need(prior.sha(Path(tape_ref['path'])) == tape_ref['sha256']
         and prior.ref(POLICY) == policy_ref
         and prior.git('rev-parse', 'HEAD') == head
         and prior.git('status', '--porcelain') == '', 'BOOKEND')
    return {
        'schema_version': 'gx1_fold2_multiminute_initial_result_v1',
        'status': 'COMPLETE', 'stage': STAGE, 'source_head': head,
        'plan': prior.ref(PLAN), 'policy': policy_ref,
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'view_sha256': view['contract_sha256'],
        'model_contract_sha256': model.research_objective['contract_sha256'],
        'loss_contract_sha256': loss_contract()['contract_sha256'],
        'query_sha256': QUERY_SHA256,
        'initial_state_sha256': initial, 'final_state_sha256': final,
        'batch_parent_rows_sha256': hashlib.sha256(
            np.asarray(rows, '<i8').tobytes()).hexdigest(),
        'batch_issue_ns_sha256': hashlib.sha256(
            issue_ns.astype('<i8', copy=False).tobytes()).hexdigest(),
        'input_reports': reports,
        'target_informative_counts': loss['informative_counts'].tolist(),
        'target_known_side_counts': loss['known_side_counts'].tolist(),
        'raw_side_nonzero_gradient_count': int(torch.count_nonzero(side_grad).item()),
        'raw_activity_nonzero_gradient_count': int(torch.count_nonzero(activity_grad).item()),
        'input_gradients': input_gradients,
        'parameter_gradients': parameter_gradients,
        'initial_loss': float(loss['loss'].detach().item()),
        'initial_marginal_scores': scores,
        'backbone_encode_calls': 1, 'head_decode_calls': 1,
        'backward_calls': 1, 'model_constructions': 1,
        'optimizer_steps': 0, 'checkpoint_reads': 0,
        'cuda_initialized': False, 'test_data_used': False,
        'trade_selection': False,
    }


def run(plan_sha: str) -> None:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_research_producer_execution,
        require_capped_lock_ancestry)
    head, policy_ref, plan, recipe = preflight(plan_sha)
    cap = require_capped_cpu_research_producer_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 12 * 1024**3
         and cap['swap'] == 512 * 1024**2 and lock is not None, 'CAP')
    ROOT.mkdir(parents=False, exist_ok=False)
    _publish_bytes(ROOT / 'CLAIM.json', prior.encode({
        'schema_version': 'gx1_fold2_multiminute_initial_claim_v1',
        'stage': STAGE, 'plan': prior.ref(PLAN),
        'policy': policy_ref, 'source_head': head, 'test_data_used': False}))
    claim = prior.ref(ROOT / 'CLAIM.json')
    started = time.monotonic()
    try:
        result = measure(head, policy_ref, plan, recipe)
        result['claim'] = claim
        result['elapsed_seconds'] = time.monotonic() - started
        _publish_bytes(ROOT / 'RESULT.json', prior.encode(result))
        terminal = {
            'schema_version': 'gx1_fold2_multiminute_initial_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'stage': STAGE,
            'claim': claim, 'result': prior.ref(ROOT / 'RESULT.json'),
            'source_head': head, 'test_data_used': False}
        _publish_bytes(ROOT / 'TERMINAL.json', prior.encode(terminal))
    except BaseException as exc:
        terminal = {
            'schema_version': 'gx1_fold2_multiminute_initial_terminal_v1',
            'status': 'FAILED_CONSUMED', 'exit_code': 1, 'stage': STAGE,
            'claim': claim, 'result': None, 'source_head': head,
            'reason': type(exc).__name__ + ':' + str(exc),
            'test_data_used': False}
        _publish_bytes(ROOT / 'TERMINAL.json', prior.encode(terminal))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha', required=True)
    run(parser.parse_args().plan_sha)


if __name__ == '__main__':
    main()
