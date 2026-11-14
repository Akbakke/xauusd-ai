"""Read-only fold 2 M5 fit metadata check; never admits or runs training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path('/home/andre2/src/GX1_CURRENT')
REVIEW = REPO / 'docs/TA_FOLD2_INPUT_REVIEW.json'
POLICY = REPO / 'NEXT_RUN_POLICY.json'


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('FOLD2_M5_FIT_PREFLIGHT_' + reason)


def read_ref(binding: dict, *, base: Path) -> dict:
    path = Path(binding['path'])
    need(path.is_absolute() and path.is_file() and not path.is_symlink()
         and path.is_relative_to(base) and set(binding) == {'path', 'sha256'}, 'REFERENCE_PATH')
    need(hashlib.sha256(path.read_bytes()).hexdigest() == binding['sha256'], 'REFERENCE_SHA')
    return json.loads(path.read_text())


def inspect() -> dict:
    need(REPO.is_dir() and subprocess.check_output(
        ['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
         == 'work/gx1-current', 'BRANCH')
    policy = json.loads(POLICY.read_text())
    review = json.loads(REVIEW.read_text())
    h = review['fold2_m5_fit_hypothesis']
    fit = h['fixed_fit']
    resource = h['resource_gate']
    from gx1.contracts.market_state_ta_fold2_m5_fit_v1 import (
        MODEL_CONTRACT_SHA256, require_fixed_design,
    )
    require_fixed_design(h)
    need(policy['status'] == 'TA_FOLD2_M5_CAMPAIGN_ROUTE_SOURCE_REVIEWED_EXECUTION_CLOSED'
         and policy['training_enabled'] is False
         and all(policy.get(k) is False for k in (
             'native_training_enabled', 'native_launch_authorized',
             'full_epoch_training_allowed', 'full_val_allowed'))
         and review['new_job_admitted'] is review['native_training_authorized'] is False
         and h['execution_authorized'] is h['training_authorized'] is False,
         'CLOSED_POLICY')
    need(review['test_data_used'] is h['test_data_used'] is False,
         'TEST_SEAL')
    for name, binding in h['source_owners'].items():
        need(binding['path'] == str(REPO / name), 'SOURCE_PATH')
        path = REPO / name
        need(path.is_file() and not path.is_symlink()
             and hashlib.sha256(path.read_bytes()).hexdigest() == binding['sha256'],
             'SOURCE_SHA')

    from gx1.contracts.market_state_time_query_native_F_objective_v1 import (
        contract, m5_equal_observation_profile,
    )
    profile = m5_equal_observation_profile(fold_id=2)
    loss = contract(ta_only=True, dedicated_m5_side=True,
                    dedicated_tf_activity=True, m5_equal_observation=True,
                    m5_direction_only=True, fold_id=2)
    need(fit['fold_id'] == profile['fold_id'] == 2
         and fit['pass_index'] == profile['pass_index'] == 0
         and fit['seed'] == profile['seed'] == 20260911
         and fit['train_rows'] == profile['fit_rows'] == 47338
         and fit['ordered_batches'] == profile['fit_batches'] == 2959
         and fit['batch_size'] == profile['batch_size'] == 16
         and fit['last_batch_rows'] == 10
         and fit['maximum_optimizer_updates'] == 2959
         and fit['view_contract_sha256'] == profile['feature_view_sha256']
         and fit['m5_equal_observation_profile_sha256'] == profile['contract_sha256']
         and fit['loss_contract_sha256'] == loss['contract_sha256']
         and loss['dual_model_contract_sha256'] == MODEL_CONTRACT_SHA256
         and fit['m5_direction_only'] is True
         and fit['future_weight'] == 1.0,
         'FIT_CONTRACT')
    for task, label in (('state', 'state'), ('future', 'future')):
        need(fit[task + '_m5_known_rows'] == profile[label]['known_side_rows']
             and fit[task + '_nonempty_batches'] == profile[label]['nonempty_batches'],
             'M5_SUPPORT')
    target = read_ref(profile['target_count_result'], base=Path('/home/andre2/GX1_RUNS'))
    target_validation = read_ref(profile['target_count_validation'],
                                 base=Path('/home/andre2/GX1_RUNS'))
    need(target['status'] == 'COMPLETE' and target_validation['status']
         == 'PASS_JSON_PREFIX_MASK_BATCH_AND_PRIOR_JOINS'
         and target['source_unchanged'] is target_validation['source_unchanged'] is True
         and target['test_data_used'] is target_validation['test_data_used'] is False
         and target['all_fit_targets_mature'] is target['S_M5_sampled_endpoint_parity'] is True
         and (target['fold_id'], target['pass_index'], target['seed'],
              target['fit_rows'], target['batches'], target['tail_rows'])
         == (2, 0, 20260911, 47338, 2959, 10)
         and target['fit_parent_rows_sha256'] == profile['parent_axis_sha256']
         and target['fit_decision_ns_sha256'] == profile['issue_axis_sha256']
         and all(target['tasks'][task] == target_validation['tasks'][task]
                 for task in ('state', 'future')),
         'TARGET_RECEIPTS')

    basis = h['observed_basis']
    initial = read_ref(basis['fresh_fold2_initial_result'],
                       base=Path('/home/andre2/GX1_RUNS'))
    validation = read_ref(basis['fresh_fold2_initial_validation'],
                          base=Path('/home/andre2/GX1_RUNS'))
    auc = initial['readouts']['LATER']['future']['all']['M5']['auc']
    need(initial['status'] == 'COMPLETE_FRESH_INITIAL'
         and initial['source_unchanged'] is True
         and initial['test_data_used'] is validation['test_data_used'] is False
         and initial['model_state_sha256'] == basis['fresh_fold2_initial_model_sha256']
         == fit['fresh_start_requires_exact_initial_model_state_sha256']
         and initial['optimizer_updates'] == validation['training_updates'] == 0
         and all(initial['counts'][k] == 0 for k in (
             'backwards', 'checkpoint_loads', 'optimizer_constructions',
             'optimizer_updates', 'test_rows'))
         and validation['status'] == 'PASS_INDEPENDENT_CAPTURE_READOUT'
         and validation['panel_task_month_tf_cells'] == 130
         and auc == basis['fresh_fold2_later_F_M5_auc'],
         'FRESH_INITIAL')
    need(h['fixed_readout']['train_and_later_panels']
         == review['fresh_initial_design']['panels'], 'UNCHANGED_PANELS')
    from gx1.contracts.market_state_ta_fold2_m5_fit_v1 import (
        INITIAL_PLAN_REF, require_reused_input_refs,
    )
    need(INITIAL_PLAN_REF == review['fresh_initial_execution']['plan'],
         'INITIAL_PLAN_REF')
    require_reused_input_refs(INITIAL_PLAN_REF)
    bootstrap = h['fixed_readout']['bootstrap_binding']
    method = read_ref(bootstrap['method_review'], base=REPO)['method']
    script_ref = bootstrap['source_script']
    script = Path(script_ref['path'])
    need(bootstrap['replicates'] == method['bootstrap_replicates'] == 2000
         and bootstrap['seeds'] == method['bootstrap_seeds']
         == {'UTC_DAY': 20260911, 'UTC_MONDAY_WEEK': 20260912}
         and script.is_file() and not script.is_symlink()
         and script.is_relative_to(Path('/home/andre2/GX1_RUNS'))
         and hashlib.sha256(script.read_bytes()).hexdigest() == script_ref['sha256'],
         'BOOTSTRAP_METHOD')

    previous = read_ref(basis['prior_fold1_full_objective_result'],
                        base=Path('/home/andre2/GX1_RUNS'))
    need(previous['test_data_used'] is False
         and previous['optimizer_updates'] == resource['historical_fold1_optimizer_updates'] == 3698
         and previous['timings']['FULL_PREFIX_seconds']
         == resource['historical_fold1_full_prefix_seconds']
         and previous['timings']['elapsed_seconds']
         == resource['historical_fold1_elapsed_seconds']
         and resource['cuda_producer_default_wall_seconds'] == 3600
         and resource['cuda_producer_fit_allowlisted'] is False
         and resource['canonical_trainer_window_max_invocation_seconds_option'] == 12000,
         'RESOURCE_BASIS')
    proposed = resource['proposed_one_window']
    need(proposed == {
        'max_invocation_seconds': 12000, 'max_model_seconds': 10500,
        'wrapper_wall_seconds': 13800, 'gpu_power_cap_w': 250,
        'checkpoint_writes': 0, 'resume_allowed': False,
        'failure_consumes_one_use_plan': True,
    }, 'PROPOSED_WINDOW')
    return {
        'status': 'PASS_METADATA_ONLY_FIT_UNADMITTED',
        'train_rows': fit['train_rows'], 'batches': fit['ordered_batches'],
        'last_batch_rows': fit['last_batch_rows'], 'loss_contract_sha256': loss['contract_sha256'],
        'initial_model_state_sha256': initial['model_state_sha256'],
        'later_F_M5_initial_auc': auc, 'model_constructions': 0,
        'feature_payload_reads': 0, 'optimizer_updates': 0,
        'training_authorized': False, 'test_data_used': False,
    }


def main() -> None:
    print(json.dumps(inspect(), sort_keys=True, separators=(',', ':')))


if __name__ == '__main__':
    main()
