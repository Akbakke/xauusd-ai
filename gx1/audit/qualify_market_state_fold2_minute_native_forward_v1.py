"""One-use read-only genuine B16 minute-query native forward witness.

One CPU backbone encode, fixed minute-1 and native-endpoint decodes, then
target/loss/score diagnostics. No optimizer, checkpoint, TEST or trade.
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
from gx1.contracts.market_state_minute_query_candidate_v1 import (
    MinuteQueryTargetTape, NATIVE_MINUTES, decode_raw_minute_grid,
    minute_allowed_set_loss, minute_proper_score_sums,
)
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

REPO = prior.REPO
POLICY = prior.POLICY
STAGE = 'market_state_fold2_minute_native_forward_20261115_001'
PLAN = REPO / 'configs/research/MARKET_STATE_FOLD2_MINUTE_NATIVE_FORWARD_20261115_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_FOLD2_MINUTE_NATIVE_FORWARD_20261115_001')
CODE = (
    'gx1/audit/qualify_market_state_fold2_minute_native_forward_v1.py',
    'gx1/contracts/market_state_minute_query_candidate_v1.py',
    'gx1/contracts/market_state_fold2_selected_native_batches_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/contracts/market_state_time_query_model_v2.py',
    'gx1/contracts/market_state_time_query_target_v1.py',
    'gx1/contracts/market_state_time_query_loss_v2.py',
    'gx1/contracts/market_state_time_query_scorer_v2.py',
    'gx1/contracts/market_state_five_endpoint_contract_v1.py',
    'gx1/contracts/model_state_digest_v1.py',
    'gx1/audit/build_market_state_time_query_tape_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'scripts/gx1_capped_run.sh',
)


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('FOLD2_MINUTE_NATIVE_FORWARD_' + reason)


def preflight(plan_sha: str) -> tuple:
    need(Path.cwd().resolve() == REPO
         and prior.git('branch', '--show-current') == 'work/gx1-current'
         and prior.git('status', '--porcelain') == '', 'CLEAN_SOURCE')
    head = prior.git('rev-parse', 'HEAD')
    need(not ROOT.exists() and prior.sha(PLAN) == plan_sha, 'PLAN_ROOT')
    plan = json.loads(PLAN.read_text())
    need(plan['schema_version'] == 'gx1_fold2_minute_native_forward_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'ADMITTED_READ_ONLY'
         and plan['source_base_head'] == 'a34e12c6010bf9b90a53d2a524493ad29ad737f1'
         and plan['output_root'] == str(ROOT)
         and plan['scope'] == {
             'fold_id': 2, 'partition': 'fit', 'batch_size': 16,
             'batch_index': 0, 'minute_query_all_TF': 1,
             'native_endpoint_minutes': list(NATIVE_MINUTES),
             'backbone_encode_calls': 1, 'head_decode_calls': 3,
             'optimizer_steps': 0, 'checkpoint_reads': 0,
             'test_data_used': False}
         and plan['limits'] == {
             'class': 'producer', 'memory': '12G', 'swap': '512M',
             'wall_seconds': 1800, 'max_invocations': 1}, 'PLAN')
    need(subprocess.run(('git', 'merge-base', '--is-ancestor',
                         plan['source_base_head'], head), cwd=REPO).returncode == 0,
         'ANCESTOR')
    need(set(plan['source_owners']) == set(CODE)
         and all(prior.sha(REPO / path) == digest
                 for path, digest in plan['source_owners'].items()), 'SOURCE_HASH')
    need(all(Path(ref['path']).is_file() and prior.sha(Path(ref['path'])) == ref['sha256']
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
             'backbone_forwards_authorized': 1, 'head_decodes_authorized': 3,
             'optimizer_steps_authorized': 0, 'checkpoint_reads_authorized': 0,
             'training_authorized': False, 'test_authorized': False,
             'broker_access_allowed': False, 'relaunch_allowed': False},
         'POLICY')
    return head, prior.ref(POLICY), plan, recipe


def measure(head: str, policy_ref: dict, plan: dict, recipe: dict) -> dict:
    need(not torch.cuda.is_initialized(), 'NO_CUDA_BEFORE')
    view, packet, constructor, raw, reports, provider, provider_plan = (
        batches.load_two_batches(source_owners=recipe['source_owners']))
    need(len(raw) == 2 and len(raw[0][1]) == 16
         and all(len(x[1]) == 16 for x in raw), 'B16_SOURCE')
    features, rows, issues = raw[0]
    need(all(int(issue) < 1735689600000000000 for issue in issues), 'FIT_ISSUES')
    torch.manual_seed(plan['seed'])
    model = views.build_native_direction_research_model(
        view, prefix_normalization=packet, constructor_metadata=constructor,
        device=torch.device('cpu'), unobserved_units_profile=None)
    need(type(model).__name__ == 'FactorizedTimeQueryMarketStateModel'
         and model.feature_view_identity['contract_sha256'] == view['contract_sha256'],
         'FRESH_FULL_FEATURE_MODEL')
    model.train()
    initial = canonical_model_state_sha256(model.state_dict())
    tensors = {key: torch.from_numpy(value).to(torch.device('cpu'))
               for key, value in features.items()}
    n = len(rows)
    q_old = torch.tensor(NATIVE_K, dtype=torch.int64)[None, None, :].expand(n, 1, -1)
    q_native = torch.tensor(NATIVE_MINUTES, dtype=torch.int64)[None, None, :].expand(n, 1, -1)
    q_minute = torch.ones((n, 1, 5), dtype=torch.int64)
    with torch.no_grad(), trainer._training_mha_fastpath_context(model.backbone), \
         trainer._training_autocast_context(), trainer._training_model_finite_check_context():
        hidden = model.encode_features(**tensors)
        old = model.decode_raw_query_grid(hidden, q_old)
        minute_native = decode_raw_minute_grid(model, hidden, q_native)
        minute_one = decode_raw_minute_grid(model, hidden, q_minute)
    need(torch.equal(old, minute_native)
         and minute_one.shape == old.shape == (16, 1, 5, 2)
         and bool(torch.isfinite(minute_one).all()), 'DECODE_PARITY')
    need(initial == canonical_model_state_sha256(model.state_dict())
         and not torch.cuda.is_initialized(), 'NO_UPDATE_OR_CUDA')
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
    truth = tape.gather_minutes(
        np.asarray(issues, np.int64), np.ones((n, 5), np.int64),
        fold_upper_ns=1735689600000000000)
    allowed = torch.from_numpy(truth['allowed'])
    eligible = torch.from_numpy(truth['eligible'])
    mature = torch.from_numpy(truth['mature'])
    with torch.no_grad():
        loss = minute_allowed_set_loss(
            minute_one[:, 0], allowed, eligible, mature, q_minute[:, 0])
    scores = minute_proper_score_sums(
        minute_one[:, 0].numpy(), truth['allowed'], truth['eligible'],
        truth['mature'], truth['query_minutes'])
    need(bool(torch.isfinite(loss['loss']))
         and scores['query_minutes_sha256']
             == hashlib.sha256(truth['query_minutes'].tobytes()).hexdigest(),
         'LOSS_SCORE')
    provider.validate_metadata(provider_plan)
    need(prior.sha(Path(tape_ref['path'])) == tape_ref['sha256']
         and prior.ref(POLICY) == policy_ref
         and prior.git('rev-parse', 'HEAD') == head
         and prior.git('status', '--porcelain') == '', 'BOOKEND')
    return {
        'schema_version': 'gx1_fold2_minute_native_forward_result_v1',
        'status': 'COMPLETE', 'stage': STAGE, 'source_head': head,
        'plan': prior.ref(PLAN), 'policy': policy_ref,
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'view_sha256': view['contract_sha256'],
        'model_contract_sha256': model.research_objective['contract_sha256'],
        'initial_state_sha256': initial,
        'final_state_sha256': canonical_model_state_sha256(model.state_dict()),
        'batch_parent_rows_sha256': hashlib.sha256(
            np.asarray(rows, '<i8').tobytes()).hexdigest(),
        'batch_issue_ns_sha256': hashlib.sha256(
            np.asarray(issues, '<i8').tobytes()).hexdigest(),
        'feature_keys': len(tensors),
        'input_reports': reports,
        'native_endpoint_raw_bit_equal': True,
        'minute_1_vs_endpoint_M5_max_abs_raw_delta': float(
            (minute_one[:, 0, 0] - old[:, 0, 0]).abs().max()),
        'minute_1_allowed_counts': {str(bit): int(
            np.count_nonzero(truth['allowed'] == bit))
            for bit in (1, 2, 4, 5, 7)},
        'minute_1_eligible_cells': int(truth['eligible'].sum()),
        'minute_1_loss': float(loss['loss']),
        'minute_1_score_sums': scores,
        'backbone_encode_calls': 1, 'head_decode_calls': 3,
        'model_constructions': 1, 'optimizer_steps': 0,
        'checkpoint_reads': 0, 'cuda_initialized': False,
        'test_data_used': False, 'trade_selection': False,
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
        'schema_version': 'gx1_fold2_minute_native_forward_claim_v1',
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
            'schema_version': 'gx1_fold2_minute_native_forward_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'stage': STAGE,
            'claim': claim, 'result': prior.ref(ROOT / 'RESULT.json'),
            'source_head': head, 'test_data_used': False}
        _publish_bytes(ROOT / 'TERMINAL.json', prior.encode(terminal))
    except BaseException as exc:
        terminal = {
            'schema_version': 'gx1_fold2_minute_native_forward_terminal_v1',
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
