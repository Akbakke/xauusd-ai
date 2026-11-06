"""Materialize one admitted, zero-update October control TRAIN-scoring recovery."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from gx1.contracts import entry_dynamic_m5_october_native_campaign_v1 as october
from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry)
from gx1.scripts.materialize_dynamic_m5_october_fit_campaign_v1 import _publish, _ref, _seal


def materialize(binding_path: Path):
    repo = october.REPO
    root = prefix.OCTOBER_CONTROL_SCORING_RECOVERY_ROOT
    runtime = root/'RUNTIME'
    stage = 'dynamic_m5_october_control_final_train_recovery_20261106_001'
    require = prefix._require
    binding = campaign.read_bound_json(binding_path, campaign.file_sha256(binding_path))
    require(binding_path == root/'PREPARATION_EXECUTION_BINDING.json'
        and set(binding) == {'materializer', 'recipe', 'policy', 'prepared_boot',
                             'guard_template', 'source_head'}, 'OCTOBER_RECOVERY_MATERIALIZER_BINDING')
    source = repo/'gx1/scripts/materialize_dynamic_m5_october_scoring_recovery_campaign_v1.py'
    require(binding['materializer'] == _ref(source)
        and binding['recipe']['path'] == str(root/'RECIPE.json')
        and binding['prepared_boot']['path'] == str(root/'PREPARED_WINDOWS_BOOT.json'),
        'OCTOBER_RECOVERY_MATERIALIZER_SOURCES')
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],
                                    cwd=repo, text=True)
    require(head == binding['source_head'] and not dirty,
            'OCTOBER_RECOVERY_MATERIALIZER_CLEAN_SOURCE')
    policy_path = repo/'NEXT_RUN_POLICY.json'
    require(binding['policy'] == _ref(policy_path), 'OCTOBER_RECOVERY_MATERIALIZER_POLICY_BYTES')
    policy = campaign.read_bound_json(policy_path, binding['policy']['sha256'])
    gate = policy.get(stage, {})
    require(gate.get('campaign_preparation_authorized') is True
        and gate.get('recipe') == binding['recipe']
        and gate.get('prepared_boot') == binding['prepared_boot']
        and gate.get('guard_template') == binding['guard_template']
        and gate.get('materializer') == binding['materializer']
        and gate.get('run_root') == str(runtime)
        and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
        and gate.get('max_invocations') == 1 and gate.get('relaunch_allowed') is False
        and gate.get('test_data_used') is False,
        'OCTOBER_RECOVERY_MATERIALIZER_POLICY_GATE')
    require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    require(not (root/'CAMPAIGN').exists() and not (root/'CAMPAIGN_PREPARATION_CLAIM.json').exists()
            and not runtime.exists(), 'OCTOBER_RECOVERY_MATERIALIZER_GENESIS')
    recipe = prefix.require_prefix_fit_launch_recipe(binding['recipe'],
        source_repo=repo, source_commit=head)
    require(recipe['run_scope'] == october.roles.FIT_SCOPE and recipe['arm'] == 'control'
        and recipe['admission_stage'] == stage and recipe['runtime_root'] == str(runtime)
        and recipe['max_windows'] == 1 and prefix.is_scoring_recovery(recipe),
        'OCTOBER_RECOVERY_MATERIALIZER_RECIPE')
    october.require_fit_phase_gate(gate, recipe)
    boot = campaign.require_boot_identity(campaign.read_bound_json(
        Path(binding['prepared_boot']['path']), binding['prepared_boot']['sha256']))
    old = campaign.read_bound_json(Path(binding['guard_template']['path']),
                                   binding['guard_template']['sha256'])
    require(old['phase'] == prefix.PHASE and old['test_data_used'] is False,
            'OCTOBER_RECOVERY_GUARD_TEMPLATE')
    _publish(root/'CAMPAIGN_PREPARATION_CLAIM.json', {
        'binding': _ref(binding_path), 'source_head': head,
        'genuine_job_started': False, 'test_data_used': False})
    out = root/'CAMPAIGN'
    out.mkdir()
    runtime.mkdir()
    window = _seal({
        'schema_version': prefix.WINDOW_SCHEMA, 'recipe': binding['recipe'],
        'invocation_number': 1, 'max_invocation_seconds': 12000,
        'progress_path': str(runtime/'progress/invocation-0001.json'),
        'campaign_cursor_path': str(runtime/'prefix-cursor/RESUME_CURSOR.json'),
        'output_directory': str(runtime/'prefix-fit'), 'test_data_used': False}, 'policy_sha256')
    window_ref = _publish(out/'WINDOW_0001.json', window)
    argv = [str(repo/'scripts/gx1_capped_run.sh'), '--class', 'trainer', '--mem', '20G',
            '--swap', '512M', '--', str(repo/'.venv/bin/python'), '-m', prefix.MODULE,
            '--window-policy', window_ref['path'], '--window-policy-file-sha256',
            window_ref['sha256'], '--progress-path', window['progress_path']]
    execution = _seal({
        'schema_version': campaign.EXECUTION_MANIFEST_SCHEMA,
        'decision': 'PASS_EXACT_INVOCATION', 'invocation_kind': prefix.KIND,
        'python_module': prefix.MODULE, 'source_commit': head,
        'launcher_argv_sha256': campaign.canonical_sha256(argv),
        'prelaunch_manifest': binding['recipe'],
        'prelaunch_manifest_sha256': recipe['recipe_sha256'],
        'train_session_manifest': window_ref,
        'train_session_manifest_sha256': window['policy_sha256'],
        'test_data_used': False}, 'artifact_sha256')
    execution_ref = _publish(out/'EXECUTION_0001.json', execution)
    invocation = _seal({
        'schema_version': campaign.INVOCATION_SCHEMA, 'decision': 'PASS',
        'invocation_number': 1, 'invocation_id': 'invocation-0001',
        'kind': prefix.KIND, 'batch_size': 16, 'epoch_index': None,
        'window_index': 0, 'window_count': 1, 'optimizer_step_budget': 0,
        'expected_success_outcome': 'COMPLETE', 'source_commit': head,
        'execution_manifest': execution_ref, 'launcher_argv': argv,
        'launcher_argv_sha256': campaign.canonical_sha256(argv),
        'progress_path': window['progress_path'],
        'guard_log_path': str(runtime/'logs/invocation-0001.signed-guard.log'),
        'checkpoint': {'pointer_path': window['campaign_cursor_path'],
                       'before_mode': 'GENESIS', 'predecessor_invocation_number': None,
                       'write_mode': 'ATOMIC_UPDATE'},
        'maximum_wall_seconds': 13800, 'requires_fresh_windows_boot': True,
        'signed_guard_only': True, 'test_data_used': False}, 'invocation_sha256')
    invocation_ref = _publish(out/'INVOCATION_0001.json', invocation)
    guards = {key: _ref(Path(item['path'])) for key,item in old['signed_guard_sources'].items()}
    controllers = {key: _ref(Path(item['path'])) for key,item in old['controller_sources'].items()}
    hardware = campaign.campaign_hardware_policy(prefix.PHASE)
    hardware['physical_power_limit_w'] = 250
    plan = _seal({
        'schema_version': campaign.PLAN_SCHEMA, 'decision': 'PASS_PREPARED',
        'campaign_id': 'GX1_DYNAMIC_M5_OCTOBER_CONTROL_FINAL_TRAIN_RECOVERY_20261106_001',
        'phase': prefix.PHASE, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_repo': str(repo), 'source_commit': head,
        'runtime_root': str(runtime), 'gpu_uuid': old['gpu_uuid'],
        'prepared_windows_boot': boot, 'recipe': binding['recipe'],
        'invocations': [invocation_ref], 'signed_guard_sources': guards,
        'controller_sources': controllers, 'policy': hardware,
        'authority': {'test': False, 'promotion': False, 'paper': False,
                      'live': False, 'cloud_spend': False},
        'test_data_used': False}, 'plan_sha256')
    checked = campaign.require_plan(plan, verify_files=True)
    require(len(checked['checked_invocations']) == 1, 'OCTOBER_RECOVERY_CHECKED_PLAN')
    plan_ref = _publish(out/'CAMPAIGN_PLAN.json', plan)
    _publish(runtime/'EXECUTION_BINDING.json', {
        'plan': plan_ref, 'source_head': head, 'policy': binding['policy']})
    ready = prefix.require_native_direction_prefix_fit_readiness(repo,
        window_path=Path(window_ref['path']), window_file_sha256=window_ref['sha256'])
    require(ready['decision'] == 'READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES'
        and campaign.next_action(checked, [], current_boot=boot)['decision'] == 'REBOOT_REQUIRED'
        and _ref(policy_path) == binding['policy'], 'OCTOBER_RECOVERY_REBOOT_GATE')
    os.sync()
    return {'decision': 'PASS_METADATA_PREPARED_REBOOT_REQUIRED', 'plan': plan_ref,
            'recipe': binding['recipe'], 'window': window_ref,
            'source_head': head, 'genuine_job_started': False, 'test_data_used': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.binding), separators=(',', ':')))


if __name__ == '__main__':
    main()
