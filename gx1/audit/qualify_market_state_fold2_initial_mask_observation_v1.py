"""One-use read-only explanation of first Fold 2 B16 zero input gradients.

Reuses the exact initial gradient receipt and pinned TRAIN source. No model,
forward, optimizer, checkpoint, TEST, broker or trade.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.audit import measure_market_state_fold2_exact_prior_later_v1 as prior
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts import market_state_fold2_selected_native_batches_v1 as batches
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
from gx1.features.gc_native_context_v1 import masking_pairs

REPO = prior.REPO
POLICY = prior.POLICY
STAGE = 'market_state_fold2_initial_mask_observation_20261115_001'
PLAN = REPO / 'configs/research/MARKET_STATE_FOLD2_INITIAL_MASK_OBSERVATION_20261115_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_FOLD2_INITIAL_MASK_OBSERVATION_20261115_001')
CODE = (
    'gx1/audit/qualify_market_state_fold2_initial_mask_observation_v1.py',
    'gx1/audit/measure_market_state_fold2_exact_prior_later_v1.py',
    'gx1/contracts/market_state_fold2_selected_native_batches_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/features/gc_native_context_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'scripts/gx1_capped_run.sh',
)
LIMITS = {'class': 'producer', 'memory': '12G', 'swap': '512M',
          'wall_seconds': 1800, 'max_invocations': 1}
SCOPE = {'fold_id': 2, 'partition': 'fit', 'batch_size': 16, 'batch_index': 0,
         'model_constructions': 0, 'forward_calls': 0, 'backward_calls': 0,
         'optimizer_steps': 0, 'checkpoint_reads': 0, 'test_data_used': False}


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('FOLD2_INITIAL_MASK_OBSERVATION_' + reason)


def classify_pairs(values: np.ndarray, names: list[str], pairs: tuple,
                   zero_indices: set[int]) -> dict:
    need(values.shape[-1] == len(names), 'FIELD_WIDTH')
    need(len({value for value, _ in pairs}) == len(pairs), 'DUPLICATE_VALUE')
    all_values = {int(value) for value, _ in pairs}
    result = {'pair_count': len(pairs), 'zero_gradient_pair_count': 0,
              'zero_gradient_unobserved_pairs': 0,
              'zero_gradient_observed_pairs': [],
              'nonzero_gradient_unobserved_pairs': [],
              'unpaired_zero_gradient_fields': [
                  {'index': int(i), 'name': names[i]}
                  for i in sorted(zero_indices - all_values)]}
    for value, mask in pairs:
        need(0 <= value < len(names) and 0 <= mask < len(names), 'PAIR_INDEX')
        flag = values[..., mask]
        need(bool(np.all((flag == 0) | (flag == 1))), 'NONBINARY_MASK')
        observed = flag == 1
        observed_elements = int(np.count_nonzero(observed))
        if value in zero_indices:
            result['zero_gradient_pair_count'] += 1
            if observed_elements:
                result['zero_gradient_observed_pairs'].append({
                    'index': int(value), 'name': names[value],
                    'observed_elements': observed_elements,
                    'observed_nonzero_value_elements': int(np.count_nonzero(
                        values[..., value][observed]))})
            else:
                result['zero_gradient_unobserved_pairs'] += 1
        elif observed_elements == 0:
            result['nonzero_gradient_unobserved_pairs'].append({
                'index': int(value), 'name': names[value]})
    return result


def preflight(plan_sha: str) -> tuple:
    need(Path.cwd().resolve() == REPO
         and prior.git('branch', '--show-current') == 'work/gx1-current'
         and prior.git('status', '--porcelain') == '', 'CLEAN_SOURCE')
    head = prior.git('rev-parse', 'HEAD')
    need(not ROOT.exists() and prior.sha(PLAN) == plan_sha, 'PLAN_ROOT')
    plan = json.loads(PLAN.read_text())
    need(plan.get('schema_version') == 'gx1_fold2_initial_mask_observation_plan_v1'
         and plan.get('status') == 'ADMITTED_READ_ONLY'
         and plan.get('stage') == STAGE
         and plan.get('source_base_head') == 'f10c279691a0274a6ede738d9f459988b8bf6b4f'
         and plan.get('output_root') == str(ROOT)
         and plan.get('scope') == SCOPE and plan.get('limits') == LIMITS,
         'PLAN')
    need(subprocess.run(('git', 'merge-base', '--is-ancestor',
                         plan['source_base_head'], head), cwd=REPO).returncode == 0,
         'ANCESTOR')
    need(set(plan.get('source_owners', {})) == set(CODE)
         and all(prior.sha(REPO / path) == digest
                 for path, digest in plan['source_owners'].items()), 'SOURCE_HASH')
    need(all(Path(ref['path']).is_file()
             and prior.sha(Path(ref['path'])) == ref['sha256']
             for ref in plan['input_refs'].values()), 'INPUT_HASH')
    prior_result = json.loads(Path(plan['input_refs']['initial_result']['path']).read_text())
    terminal = json.loads(Path(plan['input_refs']['initial_terminal']['path']).read_text())
    recipe = json.loads(Path(plan['input_refs']['recipe']['path']).read_text())
    need(prior_result['status'] == terminal['status'] == 'COMPLETE'
         and terminal['exit_code'] == 0
         and terminal['result'] == plan['input_refs']['initial_result']
         and prior_result['claim'] == terminal['claim']
         and prior_result['source_head'] == terminal['source_head']
         and prior_result['optimizer_steps'] == prior_result['checkpoint_reads'] == 0
         and prior_result['test_data_used'] is False
         and recipe['fold_id'] == 2 and recipe['partition'] == 'fit'
         and all(Path(ref['path']).is_file()
                 and prior.sha(Path(ref['path'])) == ref['sha256']
                 for ref in recipe['source_owners'].values()), 'INPUT_LINEAGE')
    policy = json.loads(POLICY.read_text())
    expected = {'status': 'ADMITTED_READ_ONLY', 'plan': prior.ref(PLAN),
                'output_root': str(ROOT), 'max_invocations': 1,
                'claimed_invocations': 0, 'completed_invocations': 0,
                'data_reads_authorized': True, 'model_constructions_authorized': 0,
                'forward_calls_authorized': 0, 'backward_calls_authorized': 0,
                'optimizer_steps_authorized': 0, 'checkpoint_reads_authorized': 0,
                'training_authorized': False, 'test_authorized': False,
                'broker_access_allowed': False, 'relaunch_allowed': False}
    need(policy['training_enabled'] is False
         and policy['native_training_enabled'] is False
         and policy[STAGE] == expected, 'POLICY')
    return head, prior.ref(POLICY), plan, prior_result, recipe


def measure(head: str, policy_ref: dict, plan: dict,
            prior_result: dict, recipe: dict) -> dict:
    view, packet, _, raw_batches, reports, provider, provider_plan = (
        batches.load_two_batches(source_owners=recipe['source_owners']))
    need(len(raw_batches) == 2 and len(raw_batches[0][1]) == 16,
         'B16_SOURCE')
    features, rows, issues = raw_batches[0]
    digest = lambda array: hashlib.sha256(
        np.asarray(array, dtype='<i8').tobytes()).hexdigest()
    need(digest(rows) == prior_result['batch_parent_rows_sha256']
         and digest(issues) == prior_result['batch_issue_ns_sha256']
         and reports == prior_result['input_reports']
         and view['contract_sha256'] == prior_result['view_sha256'],
         'SAME_B16')
    fields, _, widths = views._canonical()
    need(widths == view['physical_widths']
         and all(views._sha(names) == view['physical_field_names_sha256'][key]
                 for key, names in fields.items()), 'FIELD_IDENTITY')
    layout = packet['gc_mtf_history']['layout']
    gradients = prior_result['input_gradients']
    checks = {}
    for key, names, pairs in (
            ('seq_x', fields['signal'], masking_pairs(fields['signal'])),
            ('ctx_cont', fields['ctx_cont'], masking_pairs(fields['ctx_cont']))):
        zero = set(gradients[key]['zero_channel_indices'])
        checks[key] = classify_pairs(features[key], names, pairs, zero)
        need(len(zero) == len(checks[key]['unpaired_zero_gradient_fields'])
             + checks[key]['zero_gradient_pair_count'], 'ZERO_CLASSIFICATION:' + key)
    for lane in ('M5', 'M15', 'H1', 'H4', 'D1'):
        key = 'gc_seq_' + lane.lower()
        names = layout['field_names'][lane]
        pairs = tuple(tuple(pair) for pair in layout['value_mask_pairs'][lane])
        zero = set(gradients[key]['zero_channel_indices'])
        checks[key] = classify_pairs(features[key], names, pairs, zero)
        need(len(zero) == len(checks[key]['unpaired_zero_gradient_fields'])
             + checks[key]['zero_gradient_pair_count'], 'ZERO_CLASSIFICATION:' + key)
    need(checks['seq_x']['unpaired_zero_gradient_fields']
         == [{'index': 13, 'name': 'smc_swing_state'}]
         and all(not checks[key]['unpaired_zero_gradient_fields']
                 for key in checks if key != 'seq_x'), 'UNPAIRED_ZERO')
    provider.validate_metadata(provider_plan)
    return {'schema_version': 'gx1_fold2_initial_mask_observation_result_v1',
            'status': 'COMPLETE', 'stage': STAGE, 'source_head': head,
            'observed_utc': datetime.now(timezone.utc).isoformat(),
            'plan': prior.ref(PLAN), 'policy': policy_ref,
            'initial_result': plan['input_refs']['initial_result'],
            'batch_parent_rows_sha256': digest(rows),
            'batch_issue_ns_sha256': digest(issues),
            'view_sha256': view['contract_sha256'], 'checks': checks,
            'model_constructions': 0, 'forward_calls': 0, 'backward_calls': 0,
            'optimizer_steps': 0, 'checkpoint_reads': 0,
            'test_data_used': False, 'trade_selection': False}


def run(plan_sha: str) -> None:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_research_producer_execution,
        require_capped_lock_ancestry)
    head, policy_ref, plan, prior_result, recipe = preflight(plan_sha)
    cap = require_capped_cpu_research_producer_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 12 * 1024**3
         and cap['swap'] == 512 * 1024**2 and lock is not None, 'CAP')
    ROOT.mkdir(parents=False, exist_ok=False)
    _publish_bytes(ROOT / 'CLAIM.json', prior.encode({
        'schema_version': 'gx1_fold2_initial_mask_observation_claim_v1',
        'stage': STAGE, 'plan': prior.ref(PLAN),
        'policy': policy_ref, 'source_head': head, 'test_data_used': False}))
    claim = prior.ref(ROOT / 'CLAIM.json')
    started = time.monotonic()
    try:
        result = measure(head, policy_ref, plan, prior_result, recipe)
        result['claim'] = claim
        result['elapsed_seconds'] = time.monotonic() - started
        _publish_bytes(ROOT / 'RESULT.json', prior.encode(result))
        terminal = {
            'schema_version': 'gx1_fold2_initial_mask_observation_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'stage': STAGE,
            'claim': claim, 'result': prior.ref(ROOT / 'RESULT.json'),
            'source_head': head, 'test_data_used': False}
        _publish_bytes(ROOT / 'TERMINAL.json', prior.encode(terminal))
    except BaseException as exc:
        terminal = {
            'schema_version': 'gx1_fold2_initial_mask_observation_terminal_v1',
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
