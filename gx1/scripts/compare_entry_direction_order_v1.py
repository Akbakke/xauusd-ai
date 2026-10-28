"""Compare exactly bound shuffled and chronological saved direction predictions.

One capped CPU audit reads two development OOF archives once. It never opens
market inputs or weights, constructs a model, fits, or selects a threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from gx1.contracts import gx1_capped_execution_v1 as cap
from gx1.contracts import local_random_access_campaign_v2 as binding
from gx1.contracts.entry_native_direction_context_report_v1 import CONTEXT_SPEC
from gx1.scripts import compare_entry_direction_matched_exposure_v1 as matched

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE = 'direction_order_paired_actual_20261028_001'
ALPHA = .05 / 4
PROTOCOL = {'primary_horizon_minutes': 5, 'horizons_minutes': [1, 5, 15],
    'primary_auc_intervals': ['candidate_minus_baseline_day', 'candidate_minus_baseline_week',
        'candidate_auc_day', 'candidate_auc_week'],
    'comparisons': ['auc', 'logloss', 'brier'], 'groupings': ['UTC_DAY', 'UTC_WEEK_MONDAY'],
    'per_interval_alpha': ALPHA, 'quantile_method': 'linear', 'bootstrap_replicates': 10000,
    'day_seed': 20261027, 'week_seed': 20261028, 'minimum_blocks': 8,
    'block_population': 'all fixed later parents before eligibility masks',
    'no_undefined_primary_draws': True, 'no_threshold_or_checkpoint_selection': True}
SOURCE_PATHS = ('gx1/scripts/compare_entry_direction_order_v1.py', *matched.SOURCE_PATHS)


def require(ok, reason):
    if not ok:
        raise RuntimeError('DIRECTION_ORDER_COMPARE_' + reason)


def require_provenance(review, old, new, initial_old, initial_new, old_result, new_result):
    require(review['status'] == 'COMPLETE_TRAIN_AND_OOF_PAIRED_EVALUATION_PENDING'
        and review['test_data_used'] is False
        and review['initial_state_identity_equal'] is True
        and review['initial_prediction_archive_sha256_equal'] is True, 'REVIEW')
    for name in ('fold', 'feature_view_identity', 'own_prefix_prior', 'context_map', 'fresh_identity_sha256'):
        require(old[name] == new[name] == initial_old[name] == initial_new[name], 'IDENTITY:' + name)
    require(old['own_initial_state'] == initial_old['own_initial_state']
        and new['own_initial_state'] == initial_new['own_initial_state']
        and initial_old['predictions']['sha256'] == initial_new['predictions']['sha256']
        and initial_old['predictions']['sha256'] == 'f870c6eb0a6000b2be0197bfe650db3f35f41d125d91a90162df2d01aedf112a',
        'INITIAL_PARITY')
    require(old['fold']['train_cutoff_ns'] == matched.TRAIN_END
        and old['fold']['oof_cutoff_ns'] == matched.LATER_END
        and old['test_data_used'] is new['test_data_used'] is False, 'FOLD')
    require(old_result['parent_cursor'] == new_result['parent_cursor'] == 47602
        and old_result['optimizer_updates'] == new_result['optimizer_updates'] == 2976
        and old_result['parent_order_sha256'] != new_result['parent_order_sha256']
        and old_result['source_unchanged'] is new_result['source_unchanged'] is True
        and not old_result['forbidden_open_paths'] and not new_result['forbidden_open_paths'], 'FIT')


def label_arms(result):
    """The reused numerical engine calls its inputs one/three, not their meaning here."""
    for panel in result['panels']:
        panel['arms'] = {'mixed': panel['arms']['one'], 'chronological': panel['arms']['three']}
        panel['chronological_minus_mixed'] = panel.pop('three_minus_one')
        for group in panel['group_intervals']:
            group['chronological_minus_mixed'] = group.pop('three_minus_one')
            individual = group['individual_auc_descriptive']
            group['individual_auc'] = {'mixed': individual['one'], 'chronological': individual['three']}
            del group['individual_auc_descriptive']
    result['primary_auc_vs_mixed_supported'] = result.pop('primary_auc_improvement_supported_separately')
    del result['primary_joint_gate_supported'], result['primary_support']
    return result


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'plan-sha256', 'policy-sha256', 'source-head'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    start = time.monotonic()
    scope = cap.require_capped_cpu_audit_execution()
    lock = cap.require_capped_lock_ancestry()
    require(lock['scope'] == scope['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK')
    expected_git = (args.source_head, 'work/gx1-current', '')
    require(matched._git_state() == expected_git, 'CLEAN_SOURCE')
    plan_ref = {'path': str(Path(args.plan)), 'sha256': args.plan_sha256}
    plan = binding.read_bound_json(Path(plan_ref['path']), plan_ref['sha256'])
    policy_path = REPO / 'NEXT_RUN_POLICY.json'
    policy = binding.read_bound_json(policy_path, args.policy_sha256)
    fields = {'schema_version', 'stage', 'design', 'review', 'baseline_oof', 'candidate_oof',
        'baseline_initial', 'candidate_initial', 'baseline_result', 'candidate_result',
        'inputs', 'sources', 'output_dir', 'protocol', 'test_data_used'}
    require(set(plan) == fields and plan['schema_version'] == 'gx1_direction_order_paired_plan_v1'
        and plan['stage'] == STAGE and plan['test_data_used'] is False and plan['protocol'] == PROTOCOL, 'PLAN')
    stage = policy[STAGE]
    require(stage['status'] == 'ADMITTED' and stage['plan'] == plan_ref
        and stage['protocol'] == PROTOCOL and stage['execution_authorized'] is True
        and stage['genuine_prediction_reads_authorized'] is True
        and stage['claimed_invocations'] == stage['completed_invocations'] == 0
        and stage['max_invocations'] == 1 and stage['relaunch_allowed'] is False
        and policy['independent_direction_edge_20261027']['current_unit'] == STAGE, 'ADMISSION')
    require(all(policy.get(k) is False for k in ('training_enabled', 'native_training_enabled',
        'full_epoch_training_allowed', 'full_val_allowed', 'native_launch_authorized'))
        and all(stage.get(k) is False for k in ('model_forwards_authorized', 'optimizer_steps_authorized',
            'checkpoint_reads_authorized', 'test_data_used', 'broker_access_allowed')), 'CLOSED_GATES')
    require(set(plan['sources']) == set(SOURCE_PATHS), 'SOURCE_SET')
    for name, ref in plan['sources'].items():
        require(ref['path'] == str(REPO / name), 'SOURCE_PATH')
        matched._metadata_ref(ref)
    meta_names = ('design', 'review', 'baseline_oof', 'candidate_oof', 'baseline_initial',
        'candidate_initial', 'baseline_result', 'candidate_result')
    meta = {}
    for name in meta_names:
        matched._metadata_ref(plan[name])
        meta[name] = binding.read_bound_json(Path(plan[name]['path']), plan[name]['sha256'])
    review = meta['review']
    require(review['baseline_oof'] == plan['baseline_oof'] and review['candidate_oof'] == plan['candidate_oof']
        and review['baseline_result' if 'baseline_result' in review else 'original_result'] == plan['baseline_result']
        and review['candidate_result'] == plan['candidate_result']
        and plan['inputs'] == {'one': review['baseline_predictions'], 'three': review['candidate_predictions']}
        and meta['baseline_oof']['predictions'] == plan['inputs']['one']
        and meta['candidate_oof']['predictions'] == plan['inputs']['three'], 'REVIEW_BINDINGS')
    design = meta['design']['proposed_comparison']
    require('10,000' in design['uncertainty'] and '0.05/4' in design['uncertainty']
        and '5-minute AUC' in design['evaluation'], 'ORIGINAL_DESIGN')
    require_provenance(review, meta['baseline_oof'], meta['candidate_oof'], meta['baseline_initial'],
        meta['candidate_initial'], meta['baseline_result'], meta['candidate_result'])
    output = Path(plan['output_dir'])
    require(output.parent == ROOT and output.is_absolute() and not output.exists() and not output.is_symlink(), 'OUTPUT')
    payload_refs = [plan['inputs']['one'], plan['inputs']['three']]
    require(len({ref['path'] for ref in payload_refs}) == 2
        and all(Path(ref['path']).suffix == '.npz' for ref in payload_refs), 'PAYLOADS')
    output.mkdir(mode=0o700)
    matched._write_new(output / 'CLAIM.json', {'plan': plan_ref, 'source_head': args.source_head, 'test_data_used': False})
    allowed_json = {plan[name]['path'] for name in meta_names} | {plan_ref['path']}
    reads, denials = matched._install_guard(payload_refs, allowed_json, output)
    old = matched._read_prediction(payload_refs[0], context_spec=CONTEXT_SPEC)
    new = matched._read_prediction(payload_refs[1], context_spec=CONTEXT_SPEC)
    require(set(reads) == {ref['path'] for ref in payload_refs}
        and all(count == 1 for count in reads.values()) and not denials, 'OPEN_SCOPE')
    require(np.all(old['decision_ns'] >= matched.TRAIN_END) and np.all(old['decision_ns'] < matched.LATER_END), 'LATER_SPLIT')
    result = label_arms(matched.score_and_compare(old, new, alpha=ALPHA))
    for report, arm in ((meta['baseline_oof'], 'mixed'), (meta['candidate_oof'], 'chronological')):
        for panel, original in zip(result['panels'], report['model_metrics']['per_horizon']):
            require(panel['horizon_minutes'] == original['horizon_minutes']
                and panel['eligible_rows'] == original['counts']['eligible_cells']
                and all(np.isclose(panel['arms'][arm][k], original[k], rtol=1e-12, atol=1e-12)
                    for k in ('logloss', 'brier')), 'ORIGINAL_SCORE_REPRODUCTION')
    require(matched._git_state() == expected_git and cap.require_capped_lock_ancestry() == lock, 'FINAL_SOURCE_LOCK')
    result.update(schema_version='gx1_direction_order_paired_result_v1',
        status='COMPLETE_REUSED_DEVELOPMENT_NO_EDGE_CLAIM', plan=plan_ref, source_head=args.source_head,
        matched_parent_clock_target_mask_contexts=True, original_scores_reproduced=True,
        payload_archives_read=2, checkpoint_reads=0, model_forwards=0, optimizer_steps=0,
        elapsed_seconds=time.monotonic()-start,
        source_revision_limit=review['source_revision_limit'],
        primary_candidate_vs_random_supported=all(g['individual_auc']['chronological']['low'] > .5
            for g in next(p for p in result['panels'] if p['horizon_minutes'] == 5)['group_intervals']))
    result_ref = matched._write_new(output / 'RESULT.json', result)
    print(json.dumps({'status': result['status'], 'result': result_ref,
        'primary': next(p for p in result['panels'] if p['horizon_minutes'] == 5),
        'primary_auc_vs_mixed_supported': result['primary_auc_vs_mixed_supported'],
        'primary_candidate_vs_random_supported': result['primary_candidate_vs_random_supported']}, sort_keys=True))


if __name__ == '__main__':
    main()
