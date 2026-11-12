"""Prepare one bounded, guarded TRAIN-only fold-1 time-query fit campaign.

Metadata preparation creates no model, data payload, or optimizer state. The
candidate recipe, source, policy, boot, and guard template are checked before
its one-use preparation claim is published.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_time_query_fold1_fit_v1 as fold1
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry)
from gx1.scripts.local_random_access_campaign_v2 import _atomic_new


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise campaign.RandomAccessCampaignError('TIME_QUERY_FOLD1_MATERIALIZER_' + reason)


def _ref(path: Path) -> dict[str, str]:
    return {'path': str(path), 'sha256': campaign.file_sha256(path)}


def _publish(path: Path, value: dict) -> dict[str, str]:
    _atomic_new(path, value)
    path.chmod(0o444)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return _ref(path)


def _seal(value: dict, key: str) -> dict:
    return {**value, key: campaign.canonical_sha256(value)}


def materialize(binding_path: Path) -> dict:
    from gx1.contracts import entry_native_direction_research_campaign_v1 as research
    from gx1.contracts import market_state_native_F_private_campaign_v1 as mechanical
    from gx1.contracts import market_state_ta_m5_short_learning_v1 as short
    from gx1.contracts import market_state_ta_non_m5_short_learning_v1 as non_m5
    from gx1.contracts import market_state_ta_tf_activity_short_learning_v1 as activity
    from gx1.contracts import market_state_ta_anchor_span_short_learning_v1 as anchor_span
    from gx1.contracts import market_state_ta_m5_objective_benchmark_v1 as benchmark
    from gx1.contracts import market_state_ta_coverage_benchmark_v1 as coverage
    ta_coverage = binding_path == coverage.ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    ta_benchmark = binding_path == benchmark.ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    if ta_coverage:
        short = coverage
    elif ta_benchmark:
        short = benchmark
    elif binding_path == anchor_span.ROOT / 'PREPARATION_EXECUTION_BINDING.json':
        short = anchor_span
    elif binding_path == non_m5.ROOT / 'PREPARATION_EXECUTION_BINDING.json':
        short = non_m5
    elif binding_path == activity.ROOT / 'PREPARATION_EXECUTION_BINDING.json':
        short = activity
    repo = fold1.CANONICAL_REPO
    ta_short = binding_path == short.ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    ta_full_reverse = binding_path == mechanical.TA_M5_FULL_REVERSE_ORDER_ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    ta_repeated = binding_path == mechanical.TA_M5_REPEATED_PASS_ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    ta_paired = binding_path == mechanical.TA_M5_PAIRED_BATCH_ORDER_ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    ta_centered = binding_path == mechanical.TA_M5_CENTERED_HEAD_COORDINATES_ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    ta_original_prefix = binding_path == mechanical.TA_M5_ORIGINAL_CUDA_PREFIX_ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    ta_m5_equal_loss = binding_path == mechanical.TA_M5_EQUAL_LOSS_ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    ta_final_import = binding_path == mechanical.TA_M5_FINAL_IMPORT_ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    ta_import = ta_final_import or binding_path == mechanical.TA_M5_IMPORT_ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    import_stage = mechanical.TA_M5_FINAL_IMPORT_STAGE if ta_final_import else mechanical.TA_M5_IMPORT_STAGE
    import_root, import_runtime, import_recipe_path = mechanical.ta_m5_import_paths(import_stage)
    ta_m5 = binding_path == mechanical.TA_M5_ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    ta_mechanical = ta_short or ta_full_reverse or ta_repeated or ta_paired or ta_centered or ta_original_prefix or ta_m5_equal_loss or ta_import or ta_m5 or binding_path == mechanical.TA_ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    ta_sf = binding_path == fold1.TA_SF_RUNTIME_ROOT.parent / 'PREPARATION_EXECUTION_BINDING.json'
    ta_recovery = binding_path == fold1.TA_RECOVERY_RUNTIME_ROOT.parent / 'PREPARATION_EXECUTION_BINDING.json'
    ta_score_recovery = binding_path == fold1.TA_M5_SCORE_RECOVERY_ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    ta_next_native_pass = binding_path == fold1.TA_M5_NEXT_NATIVE_PASS_ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    ta_m5_equal_loss_fit = binding_path == fold1.TA_M5_EQUAL_LOSS_FIT_ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    ta_m5_fit = binding_path == fold1.TA_M5_FIT_ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    runtime = (fold1.TA_M5_SCORE_RECOVERY_RUNTIME_ROOT if ta_score_recovery
               else fold1.TA_M5_NEXT_NATIVE_PASS_RUNTIME_ROOT if ta_next_native_pass
               else fold1.TA_M5_EQUAL_LOSS_FIT_RUNTIME_ROOT if ta_m5_equal_loss_fit
               else fold1.TA_M5_FIT_RUNTIME_ROOT if ta_m5_fit else fold1.TA_RECOVERY_RUNTIME_ROOT if ta_recovery
               else fold1.TA_SF_RUNTIME_ROOT if ta_sf else fold1.RUNTIME_ROOT)
    root = runtime.parent
    recipe_path = (fold1.TA_M5_SCORE_RECOVERY_RECIPE_PATH if ta_score_recovery
                   else fold1.TA_M5_NEXT_NATIVE_PASS_RECIPE_PATH if ta_next_native_pass
                   else fold1.TA_M5_EQUAL_LOSS_FIT_RECIPE_PATH if ta_m5_equal_loss_fit
                   else fold1.TA_M5_FIT_RECIPE_PATH if ta_m5_fit else fold1.TA_RECOVERY_RECIPE_PATH if ta_recovery
                   else fold1.TA_SF_RECIPE_PATH if ta_sf else fold1.RECIPE_PATH)
    stage = (fold1.TA_M5_SCORE_RECOVERY_STAGE if ta_score_recovery
             else fold1.TA_M5_NEXT_NATIVE_PASS_STAGE if ta_next_native_pass
             else fold1.TA_M5_EQUAL_LOSS_FIT_STAGE if ta_m5_equal_loss_fit
             else fold1.TA_M5_FIT_STAGE if ta_m5_fit else fold1.TA_RECOVERY_STAGE if ta_recovery
             else fold1.TA_SF_STAGE if ta_sf else fold1.STAGE)
    run_scope = (fold1.TA_M5_SCORE_RECOVERY_SCOPE if ta_score_recovery
                 else fold1.TA_M5_NEXT_NATIVE_PASS_SCOPE if ta_next_native_pass
                 else fold1.TA_M5_EQUAL_LOSS_FIT_SCOPE if ta_m5_equal_loss_fit
                 else fold1.TA_M5_FIT_SCOPE if ta_m5_fit else fold1.TA_RECOVERY_SCOPE if ta_recovery
                 else fold1.TA_SF_SCOPE if ta_sf else fold1.SCOPE)
    authority = prefix
    windows_count = (fold1.TA_M5_SCORE_RECOVERY_MAX_WINDOWS if ta_score_recovery
                     else fold1.TA_M5_NEXT_NATIVE_PASS_MAX_WINDOWS if ta_next_native_pass
                     else fold1.TA_M5_EQUAL_LOSS_FIT_MAX_WINDOWS if ta_m5_equal_loss_fit else fold1.MAX_WINDOWS)
    cursor_directory, output_directory = 'prefix-cursor', 'prefix-fit'
    if ta_mechanical:
        runtime, root = ((short.RUNTIME_ROOT, short.ROOT) if ta_short
            else (mechanical.TA_M5_FULL_REVERSE_ORDER_RUNTIME_ROOT, mechanical.TA_M5_FULL_REVERSE_ORDER_ROOT) if ta_full_reverse
            else (mechanical.TA_M5_REPEATED_PASS_RUNTIME_ROOT, mechanical.TA_M5_REPEATED_PASS_ROOT) if ta_repeated
            else (mechanical.TA_M5_PAIRED_BATCH_ORDER_RUNTIME_ROOT, mechanical.TA_M5_PAIRED_BATCH_ORDER_ROOT) if ta_paired
            else (mechanical.TA_M5_CENTERED_HEAD_COORDINATES_RUNTIME_ROOT, mechanical.TA_M5_CENTERED_HEAD_COORDINATES_ROOT) if ta_centered
            else (mechanical.TA_M5_ORIGINAL_CUDA_PREFIX_RUNTIME_ROOT, mechanical.TA_M5_ORIGINAL_CUDA_PREFIX_ROOT) if ta_original_prefix
            else (mechanical.TA_M5_EQUAL_LOSS_RUNTIME_ROOT, mechanical.TA_M5_EQUAL_LOSS_ROOT) if ta_m5_equal_loss
            else (import_runtime, import_root) if ta_import
            else (mechanical.TA_M5_RUNTIME_ROOT, mechanical.TA_M5_ROOT) if ta_m5
            else (mechanical.TA_RUNTIME_ROOT, mechanical.TA_ROOT))
        recipe_path, stage, run_scope = (
            (short.RECIPE_PATH,short.STAGE,short.SCOPE) if ta_short else
            (mechanical.TA_M5_FULL_REVERSE_ORDER_RECIPE_PATH, mechanical.TA_M5_FULL_REVERSE_ORDER_STAGE, mechanical.TA_M5_FULL_REVERSE_ORDER_SCOPE)
            if ta_full_reverse else
            (mechanical.TA_M5_REPEATED_PASS_RECIPE_PATH, mechanical.TA_M5_REPEATED_PASS_STAGE, mechanical.TA_M5_REPEATED_PASS_SCOPE)
            if ta_repeated else
            (mechanical.TA_M5_PAIRED_BATCH_ORDER_RECIPE_PATH, mechanical.TA_M5_PAIRED_BATCH_ORDER_STAGE, mechanical.TA_M5_PAIRED_BATCH_ORDER_SCOPE)
            if ta_paired else
            (mechanical.TA_M5_CENTERED_HEAD_COORDINATES_RECIPE_PATH, mechanical.TA_M5_CENTERED_HEAD_COORDINATES_STAGE, mechanical.TA_M5_CENTERED_HEAD_COORDINATES_SCOPE)
            if ta_centered else
            (mechanical.TA_M5_ORIGINAL_CUDA_PREFIX_RECIPE_PATH, mechanical.TA_M5_ORIGINAL_CUDA_PREFIX_STAGE, mechanical.TA_M5_ORIGINAL_CUDA_PREFIX_SCOPE)
            if ta_original_prefix else
            (mechanical.TA_M5_EQUAL_LOSS_RECIPE_PATH, mechanical.TA_M5_EQUAL_LOSS_STAGE, mechanical.TA_M5_EQUAL_LOSS_SCOPE)
            if ta_m5_equal_loss else
            (import_recipe_path, import_stage, mechanical.TA_M5_IMPORT_SCOPE)
            if ta_import else (mechanical.TA_M5_RECIPE_PATH, mechanical.TA_M5_STAGE, mechanical.TA_M5_SCOPE)
            if ta_m5 else (mechanical.TA_RECIPE_PATH, mechanical.TA_STAGE, mechanical.TA_SCOPE))
        authority, windows_count = research, 1
        cursor_directory, output_directory = 'research-cursor', 'private-research'
    _need(binding_path == root / 'PREPARATION_EXECUTION_BINDING.json', 'BINDING_PATH')
    binding = campaign.read_bound_json(binding_path, campaign.file_sha256(binding_path))
    _need(set(binding) == {'materializer', 'recipe', 'policy', 'prepared_boot',
                           'guard_template', 'source_head'}, 'BINDING_FIELDS')
    source = repo / 'gx1/scripts/materialize_market_state_time_query_fold1_fit_campaign_v1.py'
    _need(binding['materializer'] == _ref(source)
          and binding['recipe']['path'] == str(recipe_path)
          and binding['prepared_boot']['path'] == str(root / 'PREPARED_WINDOWS_BOOT.json'),
          'SOURCE_AND_PATHS')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(
        ['git', 'status', '--porcelain=v1', '--untracked-files=all'],
        cwd=repo, text=True)
    _need(head == binding['source_head'] and not dirty, 'CLEAN_SOURCE')
    policy_path = repo / 'NEXT_RUN_POLICY.json'
    _need(binding['policy'] == _ref(policy_path), 'POLICY_BYTES')
    policy = campaign.read_bound_json(policy_path, binding['policy']['sha256'])
    gate = policy.get(stage)
    if ta_import:
        _need(policy.get('training_enabled') is False
              and policy.get('native_training_enabled') is False, 'IMPORT_GLOBAL_TRAINING_CLOSED')
    _need(isinstance(gate, dict)
          and gate.get('campaign_preparation_authorized') is True
          and gate.get('recipe') == binding['recipe']
          and gate.get('prepared_boot') == binding['prepared_boot']
          and gate.get('guard_template') == binding['guard_template']
          and gate.get('materializer') == binding['materializer']
          and gate.get('run_root') == str(runtime)
          and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
          and gate.get('max_invocations') == windows_count
          and gate.get('relaunch_allowed') is False
          and gate.get('test_data_used') is False, 'POLICY_GATE')
    require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    _need(not (root / 'CAMPAIGN').exists()
          and not (root / 'CAMPAIGN_PREPARATION_CLAIM.json').exists()
          and not runtime.exists(), 'GENESIS')
    require_recipe = (research.require_research_launch_recipe if ta_mechanical
                      else prefix.require_prefix_fit_launch_recipe)
    recipe = require_recipe(binding['recipe'], source_repo=repo, source_commit=head)
    _need(recipe['run_scope'] == run_scope
          and recipe['admission_stage'] == stage
          and recipe['runtime_root'] == str(runtime)
          and (ta_mechanical or recipe['max_windows'] == windows_count), 'RECIPE')
    if ta_mechanical:
        (short.require_gate if ta_short else mechanical.require_ta_m5_full_reverse_order_gate if ta_full_reverse
         else mechanical.require_ta_m5_repeated_pass_gate if ta_repeated
         else mechanical.require_ta_m5_paired_batch_order_gate if ta_paired
         else mechanical.require_ta_m5_centered_head_coordinates_gate if ta_centered
         else mechanical.require_ta_m5_original_cuda_prefix_gate if ta_original_prefix
          else mechanical.require_ta_m5_equal_loss_mechanical_gate if ta_m5_equal_loss
         else mechanical.require_ta_m5_import_gate if ta_import
         else mechanical.require_ta_m5_mechanical_gate if ta_m5
         else mechanical.require_ta_sf_mechanical_gate)(gate, recipe)
    elif ta_score_recovery:
        fold1.require_ta_m5_score_recovery_gate(gate, recipe)
    else:
        fold1.require_fit_gate(gate, recipe)
    boot = campaign.require_boot_identity(campaign.read_bound_json(
        Path(binding['prepared_boot']['path']), binding['prepared_boot']['sha256']))
    old = campaign.read_bound_json(Path(binding['guard_template']['path']),
                                   binding['guard_template']['sha256'])
    _need(old['phase'] == authority.PHASE and old['test_data_used'] is False,
          'GUARD_TEMPLATE')
    _publish(root / 'CAMPAIGN_PREPARATION_CLAIM.json', {
        'binding': _ref(binding_path), 'source_head': head,
        'genuine_job_started': False, 'test_data_used': False})
    out = root / 'CAMPAIGN'
    out.mkdir()
    runtime.mkdir()
    invocations = []
    windows = []
    for number in range(1, windows_count + 1):
        window = _seal({
            'schema_version': authority.WINDOW_SCHEMA, 'recipe': binding['recipe'],
            'invocation_number': number, 'max_invocation_seconds': 12000 if ta_benchmark or ta_coverage else 3600 if ta_short else 3000 if ta_paired else 12000,
            'progress_path': str(runtime / 'progress' / f'invocation-{number:04d}.json'),
            'campaign_cursor_path': str(runtime / cursor_directory / 'RESUME_CURSOR.json'),
            'output_directory': str(runtime / output_directory), 'test_data_used': False,
        }, 'policy_sha256')
        window_ref = _publish(out / f'WINDOW_{number:04d}.json', window)
        windows.append(window_ref)
        argv = [str(repo / 'scripts/gx1_capped_run.sh'), '--class', 'trainer',
                '--mem', '20G', '--swap', '512M', '--',
                str(repo / '.venv/bin/python'), '-m', authority.MODULE,
                '--window-policy', window_ref['path'],
                '--window-policy-file-sha256', window_ref['sha256'],
                '--progress-path', window['progress_path']]
        execution = _seal({
            'schema_version': campaign.EXECUTION_MANIFEST_SCHEMA,
            'decision': 'PASS_EXACT_INVOCATION', 'invocation_kind': authority.KIND,
            'python_module': authority.MODULE, 'source_commit': head,
            'launcher_argv_sha256': campaign.canonical_sha256(argv),
            'prelaunch_manifest': binding['recipe'],
            'prelaunch_manifest_sha256': recipe['recipe_sha256'],
            'train_session_manifest': window_ref,
            'train_session_manifest_sha256': window['policy_sha256'],
            'test_data_used': False,
        }, 'artifact_sha256')
        execution_ref = _publish(out / f'EXECUTION_{number:04d}.json', execution)
        invocation = _seal({
            'schema_version': campaign.INVOCATION_SCHEMA, 'decision': 'PASS',
            'invocation_number': number, 'invocation_id': f'invocation-{number:04d}',
            'kind': authority.KIND, 'batch_size': fold1.BATCH_SIZE,
            'epoch_index': None, 'window_index': number - 1,
            'window_count': windows_count,
            'optimizer_step_budget': (0 if ta_import or ta_score_recovery else 768 if ta_coverage else 1536 if ta_benchmark else 1849 if ta_full_reverse else 512 if ta_paired or ta_repeated else 256 if ta_short or ta_original_prefix or ta_centered
                                       else mechanical.MAX_PRIVATE_UPDATES if ta_mechanical
                                      else recipe['numerical_recipe']['max_updates']),
            'expected_success_outcome': 'COMPLETE' if ta_mechanical or ta_score_recovery else 'RESUMABLE_OR_COMPLETE',
            'source_commit': head, 'execution_manifest': execution_ref,
            'launcher_argv': argv,
            'launcher_argv_sha256': campaign.canonical_sha256(argv),
            'progress_path': window['progress_path'],
            'guard_log_path': str(runtime / 'logs' /
                                  f'invocation-{number:04d}.signed-guard.log'),
            'checkpoint': {'pointer_path': window['campaign_cursor_path'],
                           'before_mode': 'GENESIS' if number == 1 else 'PREVIOUS_RECEIPT_AFTER',
                           'predecessor_invocation_number': None if number == 1 else number - 1,
                           'write_mode': 'ATOMIC_UPDATE'},
            'maximum_wall_seconds': 13800 if ta_benchmark or ta_coverage else 3900 if ta_short else 3300 if ta_paired else 13800, 'requires_fresh_windows_boot': True,
            'signed_guard_only': True, 'test_data_used': False,
        }, 'invocation_sha256')
        invocations.append(_publish(out / f'INVOCATION_{number:04d}.json', invocation))
    guards = {key: _ref(Path(item['path']))
              for key, item in old['signed_guard_sources'].items()}
    controllers = {key: _ref(Path(item['path']))
                   for key, item in old['controller_sources'].items()}
    hardware = campaign.campaign_hardware_policy(authority.PHASE)
    hardware['physical_power_limit_w'] = 250
    plan = _seal({
        'schema_version': campaign.PLAN_SCHEMA, 'decision': 'PASS_PREPARED',
        'campaign_id': stage.upper(), 'phase': authority.PHASE,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_repo': str(repo), 'source_commit': head,
        'runtime_root': str(runtime), 'gpu_uuid': old['gpu_uuid'],
        'prepared_windows_boot': boot, 'recipe': binding['recipe'],
        'invocations': invocations, 'signed_guard_sources': guards,
        'controller_sources': controllers, 'policy': hardware,
        'authority': {'test': False, 'promotion': False, 'paper': False,
                      'live': False, 'cloud_spend': False},
        'test_data_used': False,
    }, 'plan_sha256')
    checked = campaign.require_plan(plan, verify_files=True)
    _need(len(checked['checked_invocations']) == windows_count,
          'CHECKED_PLAN')
    plan_ref = _publish(out / 'CAMPAIGN_PLAN.json', plan)
    _publish(runtime / 'EXECUTION_BINDING.json', {
        'plan': plan_ref, 'source_head': head, 'policy': binding['policy']})
    require_ready = (research.require_native_direction_research_readiness if ta_mechanical
                     else prefix.require_native_direction_prefix_fit_readiness)
    ready = require_ready(repo, window_path=Path(windows[0]['path']),
                          window_file_sha256=windows[0]['sha256'])
    _need(ready['decision'] == 'READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES'
          and campaign.next_action(checked, [], current_boot=boot)['decision'] == 'REBOOT_REQUIRED'
          and _ref(policy_path) == binding['policy'], 'REBOOT_GATE')
    os.sync()
    return {'decision': 'PASS_METADATA_PREPARED_REBOOT_REQUIRED',
            'plan': plan_ref, 'windows': windows, 'recipe': binding['recipe'],
            'source_head': head, 'genuine_job_started': False,
            'test_data_used': False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.binding), separators=(',', ':')))


if __name__ == '__main__':
    main()
