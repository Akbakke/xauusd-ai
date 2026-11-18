"""One capped real Apr-Jun score batch before a paired TA fit is admitted."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import signal
from pathlib import Path
import subprocess
import time

from gx1.audit.market_state_ta_apr_jun_campaign_inputs_v1 import (
    BASE_VIEW, FIT_REVIEW, ISSUE, MODEL_VIEW, PHYSICAL, REPO, SCORE_ROWS,
    need, prepare, read, ref,
)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

STAGE = 'ta_integrated_apr_jun_paired_preflight_20261117_001'
PLAN = REPO / 'configs/research/TA_INTEGRATED_APR_JUN_PAIRED_PREFLIGHT_20261117_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/TA_INTEGRATED_APR_JUN_PAIRED_PREFLIGHT_20261117_001')
SOURCES = ('gx1/audit/qualify_market_state_ta_apr_jun_paired_preflight_v1.py',
           'gx1/audit/market_state_ta_apr_jun_campaign_inputs_v1.py',
           'gx1/contracts/market_state_ta_integrated_model_v1.py',
           'gx1/contracts/market_state_ta_integrated_paired_objective_v1.py',
           'gx1/contracts/market_state_ta_decoder_view_rebind_v1.py')


def publish(name, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _publish_bytes(ROOT / name, raw)
    return ref(ROOT / name)


def bound_inputs():
    return {'base_view': ref(BASE_VIEW), 'model_view': ref(MODEL_VIEW),
            'physical_result': ref(PHYSICAL / 'RESULT.json'),
            'physical_coverage': ref(PHYSICAL / 'COVERAGE.json'),
            'physical_terminal': ref(PHYSICAL / 'TERMINAL.json'),
            'issue_result': ref(ISSUE / 'RESULT.json'),
            'issue_terminal': ref(ISSUE / 'TERMINAL.json'),
            'fit_review': ref(FIT_REVIEW)}


def inspect(plan_sha):
    plan_ref = ref(PLAN)
    need(plan_ref['sha256'] == plan_sha, 'PREFLIGHT_PLAN_SHA')
    plan = read(plan_ref)
    from gx1.contracts.market_state_ta_integrated_paired_objective_v1 import contract
    need(plan['schema_version'] == 'gx1_ta_integrated_apr_jun_paired_preflight_plan_v1'
         and plan['view_sha256'] == read(ref(MODEL_VIEW))['contract_sha256']
         and plan['paired_objective_sha256'] == contract()['contract_sha256']
         and plan['stage'] == STAGE and plan['output_root'] == str(ROOT)
         and plan['score_rows'] == SCORE_ROWS and plan['batch_positions'] == [0, 16]
         and plan['max_seconds'] == 900
         and plan['model_forwards'] == 1 and plan['optimizer_steps'] == 0
         and plan['test_data_used'] is False
         and plan['inputs'] == bound_inputs()
         and set(plan['source_owners']) == set(SOURCES)
         and all(ref(REPO / p)['sha256'] == plan['source_owners'][p] for p in SOURCES)
         and not ROOT.exists(), 'PREFLIGHT_SCOPE')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    need(subprocess.check_output(['git', 'branch', '--show-current'],
                                 cwd=REPO, text=True).strip() == 'work/gx1-current'
         and subprocess.run(['git', 'merge-base', '--is-ancestor',
                             plan['source_base_commit'], head],
                            cwd=REPO, capture_output=True).returncode == 0
         and not subprocess.check_output(['git', 'status', '--porcelain'],
                                         cwd=REPO, text=True), 'PREFLIGHT_CLEAN_SOURCE')
    policy_ref = ref(REPO / 'NEXT_RUN_POLICY.json')
    policy = read(policy_ref)
    gate = policy[STAGE]
    need(all(policy[k] is False for k in
             ('training_enabled', 'native_training_enabled', 'full_epoch_training_allowed',
              'native_launch_authorized', 'full_val_allowed'))
         and gate['status'] == 'ADMITTED_ONE_USE'
         and gate['plan'] == plan_ref and gate['execution_authorized'] is True
         and gate['training_authorized'] is False and gate['remaining_invocations'] == 1
         and gate['relaunch_allowed'] is False, 'PREFLIGHT_POLICY')
    return plan, plan_ref, policy_ref, head


def run(plan_sha):
    import numpy as np
    import torch
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry,
    )
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import (
        load_bound_five_endpoint_targets,
    )
    from gx1.contracts.market_state_future_displacement_candidate_v1 import (
        load_bound_future_displacement,
    )
    from gx1.contracts.market_state_ta_integrated_model_v1 import IntegratedTAMarketStateModel
    from gx1.contracts.market_state_ta_integrated_paired_objective_v1 import paired_integrated_loss
    from gx1.contracts.market_state_time_query_binding_v1 import (
        load_bound_time_query_targets, sampled_query_k,
    )
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256

    cap = require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    torch.set_num_threads(1)
    plan, plan_ref, policy_ref, head = inspect(plan_sha)
    started = time.monotonic()
    ROOT.mkdir()
    claim = publish('CLAIM.json', {'plan': plan_ref, 'policy': policy_ref,
        'source_commit': head, 'cap': cap, 'claimed_utc': datetime.now(timezone.utc).isoformat(),
        'test_data_used': False})
    status, result, failure = 'FAILED_CONSUMED', None, None
    context = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def expire(*_):
        raise TimeoutError('TA_APR_JUN_PAIRED_PREFLIGHT_DEADLINE')
    try:
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, plan['max_seconds'])
        device = torch.device('cpu')
        context = prepare(device=device)
        axes = context['axes']
        selected = context['selected'][:16]
        rows, issues = axes['parent_rows'][selected], axes['decision_ns'][selected]
        features = context['features']('SCORE', rows, issues)
        need(set(features) == {'seq_x', 'snap_x', 'ctx_cont', 'ctx_cat',
                               'seq_m5', 'seq_m15', 'seq_h1', 'seq_h4', 'seq_d1'},
             'NINE_FEATURE_SURFACES')
        torch.manual_seed(20260911)
        state = views.build_native_direction_research_model(context['view'],
            prefix_normalization=context['packet'],
            constructor_metadata=context['constructor'], device=device,
            unobserved_units_profile=None)
        model = IntegratedTAMarketStateModel(NativeFutureDualModel(state)).eval()
        digest_before = canonical_model_state_sha256(model.state_dict())
        normalizer = model.dual.state_model.backbone
        normalized = {}
        with torch.no_grad():
            for key, surface in [('seq_x', 'signal'), ('snap_x', 'signal'),
                                 ('ctx_cont', 'ctx_cont'),
                                 *((f'seq_{tf}', f'mtf_{tf}') for tf in
                                   ('m5', 'm15', 'h1', 'h4', 'd1'))]:
                value = normalizer._normalize_input_surface(features[key], surface=surface)
                need(value.shape == features[key].shape and bool(torch.isfinite(value).all()),
                     'NORMALIZED_' + key)
                normalized[key] = {'shape': list(value.shape),
                    'nonzero_cells': int(torch.count_nonzero(value)),
                    'sha256': hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()}
            need(features['ctx_cat'].dtype == torch.int64
                 and features['ctx_cat'].shape[0] == 16, 'CATEGORY_SURFACE')
            sampled = sampled_query_k(issues, pass_index=0, seed=20260911)
            native = np.broadcast_to(np.asarray(NATIVE_K, dtype=np.int64), sampled.shape)
            query = torch.from_numpy(np.stack((sampled, native), axis=1).copy())
            tape = context['score_tape']
            output = model(features, query_k=query)
            sampled_truth = load_bound_time_query_targets().gather(
                rows, issues, sampled, fold_id=3, partition='score')
            native_truth = load_bound_five_endpoint_targets().gather(
                rows, issues, fold_id=3, partition='score')
            future_truth = load_bound_future_displacement().gather(
                rows, issues, fold_id=3, partition='score')
            loss = paired_integrated_loss(model_output=output,
                sampled_truth=sampled_truth, native_truth=native_truth,
                future_truth=future_truth, query_grid=query,
                structure_hit=tape['structure_hit'][:16],
                structure_observed=tape['structure_observed'][:16])
        need(canonical_model_state_sha256(model.state_dict()) == digest_before
             and model.research_objective['contract_sha256'] == plan['model_contract_sha256']
             and bool(torch.isfinite(loss['loss']))
             and set(output) == {'state_activity_side_logits', 'state_log_probs',
                                 'state_native_activity_side_logits', 'state_native_log_probs',
                                 'future_activity_side_logits', 'future_log_probs',
                                 'm5_structure_hit_logits'}, 'MODEL_READOUT')
        result = publish('RESULT.json', {'status': 'PASS_ACTUAL_APR_JUN_PAIRED_BATCH_NO_FIT',
            'plan': plan_ref, 'claim': claim, 'source_commit': head,
            'view_sha256': context['view']['contract_sha256'],
            'initial_model_state_sha256_CPU': digest_before,
            'model_contract_sha256': model.research_objective['contract_sha256'],
            'objective_contract_sha256': loss['contract_sha256'],
            'score_parent_rows_sha256': hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest(),
            'score_issue_ns_sha256': hashlib.sha256(issues.astype('<i8').tobytes()).hexdigest(),
            'numeric_normalized_surfaces': normalized,
            'categorical_surface_shape': list(features['ctx_cat'].shape),
            'sampled_S_counts': loss['sampled_S_counts'],
            'native_S_counts': loss['native_S_counts'],
            'native_F_counts': loss['native_F_counts'],
            'structure_observed_counts': loss['structure_observed_counts'].tolist(),
            'paired_loss': float(loss['loss']), 'model_forwards': 1,
            'optimizer_constructions': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'economic_edge_proven': False,
            'elapsed_seconds': time.monotonic() - started})
        need(ref(PLAN) == plan_ref and ref(REPO / 'NEXT_RUN_POLICY.json') == policy_ref
             and subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                         cwd=REPO, text=True).strip() == head
             and not subprocess.check_output(['git', 'status', '--porcelain'],
                                             cwd=REPO, text=True), 'SOURCE_BOOKEND')
        status = 'COMPLETE'
    except BaseException as error:
        failure = publish('FAILED.json', {'type': type(error).__name__,
            'error': str(error)[:2000], 'plan': plan_ref})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        if context is not None:
            context['release']()
        publish('TERMINAL.json', {'status': status,
            'exit_code': 0 if status == 'COMPLETE' else 1,
            'plan': plan_ref, 'result': result, 'failure': failure,
            'remaining_invocations': 0, 'relaunch_allowed': False,
            'elapsed_seconds': time.monotonic() - started})


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    need(len(args.plan_sha256) == 64 and all(c in '0123456789abcdef'
         for c in args.plan_sha256), 'PLAN_SHA_FORMAT')
    run(args.plan_sha256)


if __name__ == '__main__':
    main()
