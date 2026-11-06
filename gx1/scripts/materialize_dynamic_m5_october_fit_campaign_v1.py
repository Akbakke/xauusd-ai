"""Prepare one admitted October TRAIN-prefix fit campaign; no model or data reads."""
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
from gx1.scripts.local_random_access_campaign_v2 import _atomic_new


def _ref(path):
    return {'path': str(path), 'sha256': campaign.file_sha256(path)}


def _publish(path, value):
    _atomic_new(path, value)
    path.chmod(0o444)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return _ref(path)


def _seal(value, key):
    return {**value, key: campaign.canonical_sha256(value)}


def materialize(binding_path: Path):
    repo = october.REPO
    require = october._require
    binding = campaign.read_bound_json(binding_path, campaign.file_sha256(binding_path))
    arm = binding.get('arm')
    require(arm in october.FIT_ROOTS, 'FIT_MATERIALIZER_ARM')
    root = october.FIT_ROOTS[arm].parent
    runtime = october.FIT_ROOTS[arm]
    stage = october.FIT_STAGES[arm]
    require(binding_path == root/'PREPARATION_EXECUTION_BINDING.json'
            and set(binding) == {'arm', 'materializer', 'recipe', 'policy', 'prepared_boot',
                                 'guard_template', 'source_head'}, 'FIT_MATERIALIZER_BINDING')
    source = repo/'gx1/scripts/materialize_dynamic_m5_october_fit_campaign_v1.py'
    require(binding['materializer'] == _ref(source)
            and binding['recipe']['path'] == str(root/'RECIPE.json')
            and binding['prepared_boot']['path'] == str(root/'PREPARED_WINDOWS_BOOT.json'),
            'FIT_MATERIALIZER_SOURCE_AND_PATHS')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain=v1', '--untracked-files=all'],
                                    cwd=repo, text=True)
    require(head == binding['source_head'] and not dirty, 'FIT_MATERIALIZER_CLEAN_SOURCE')
    policy_path = repo/'NEXT_RUN_POLICY.json'
    require(binding['policy'] == _ref(policy_path), 'FIT_MATERIALIZER_POLICY_BYTES')
    policy = campaign.read_bound_json(policy_path, binding['policy']['sha256'])
    gate = policy.get(stage)
    require(gate is not None and gate.get('campaign_preparation_authorized') is True
            and gate.get('arm') == arm and gate.get('recipe') == binding['recipe']
            and gate.get('prepared_boot') == binding['prepared_boot']
            and gate.get('guard_template') == binding['guard_template']
            and gate.get('materializer') == binding['materializer']
            and gate.get('run_root') == str(runtime)
            and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
            and gate.get('max_invocations') == 3 and gate.get('relaunch_allowed') is False
            and gate.get('test_data_used') is False, 'FIT_MATERIALIZER_POLICY_GATE')
    require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    require(not (root/'CAMPAIGN').exists()
            and not (root/'CAMPAIGN_PREPARATION_CLAIM.json').exists()
            and not runtime.exists(), 'FIT_MATERIALIZER_GENESIS')
    recipe = prefix.require_prefix_fit_launch_recipe(binding['recipe'],
        source_repo=repo, source_commit=head)
    require(recipe['run_scope'] == october.roles.FIT_SCOPE and recipe['arm'] == arm
            and recipe['admission_stage'] == stage and recipe['runtime_root'] == str(runtime)
            and recipe['max_windows'] == 3, 'FIT_MATERIALIZER_RECIPE')
    october.require_fit_phase_gate(gate, recipe)
    boot = campaign.require_boot_identity(campaign.read_bound_json(
        Path(binding['prepared_boot']['path']), binding['prepared_boot']['sha256']))
    old = campaign.read_bound_json(Path(binding['guard_template']['path']),
                                   binding['guard_template']['sha256'])
    require(old['phase'] == prefix.PHASE and old['test_data_used'] is False,
            'FIT_MATERIALIZER_GUARD_TEMPLATE')
    _publish(root/'CAMPAIGN_PREPARATION_CLAIM.json', {
        'binding': _ref(binding_path), 'source_head': head,
        'genuine_job_started': False, 'test_data_used': False})
    out = root/'CAMPAIGN'
    out.mkdir()
    runtime.mkdir()
    invocations, windows = [], []
    for number in range(1, recipe['max_windows'] + 1):
        window = _seal({
            'schema_version': prefix.WINDOW_SCHEMA, 'recipe': binding['recipe'],
            'invocation_number': number, 'max_invocation_seconds': 12000,
            'progress_path': str(runtime/'progress'/f'invocation-{number:04d}.json'),
            'campaign_cursor_path': str(runtime/'prefix-cursor/RESUME_CURSOR.json'),
            'output_directory': str(runtime/'prefix-fit'), 'test_data_used': False}, 'policy_sha256')
        window_ref = _publish(out/f'WINDOW_{number:04d}.json', window)
        windows.append(window_ref)
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
        execution_ref = _publish(out/f'EXECUTION_{number:04d}.json', execution)
        invocation = _seal({
            'schema_version': campaign.INVOCATION_SCHEMA, 'decision': 'PASS',
            'invocation_number': number, 'invocation_id': f'invocation-{number:04d}',
            'kind': prefix.KIND, 'batch_size': 16, 'epoch_index': None,
            'window_index': number - 1, 'window_count': recipe['max_windows'],
            'optimizer_step_budget': recipe['numerical_recipe']['max_updates'],
            'expected_success_outcome': 'RESUMABLE_OR_COMPLETE', 'source_commit': head,
            'execution_manifest': execution_ref, 'launcher_argv': argv,
            'launcher_argv_sha256': campaign.canonical_sha256(argv),
            'progress_path': window['progress_path'],
            'guard_log_path': str(runtime/'logs'/f'invocation-{number:04d}.signed-guard.log'),
            'checkpoint': {'pointer_path': window['campaign_cursor_path'],
                           'before_mode': 'GENESIS' if number == 1 else 'PREVIOUS_RECEIPT_AFTER',
                           'predecessor_invocation_number': None if number == 1 else number - 1,
                           'write_mode': 'ATOMIC_UPDATE'},
            'maximum_wall_seconds': 13800, 'requires_fresh_windows_boot': True,
            'signed_guard_only': True, 'test_data_used': False}, 'invocation_sha256')
        invocations.append(_publish(out/f'INVOCATION_{number:04d}.json', invocation))
    guards = {key: _ref(Path(item['path'])) for key, item in old['signed_guard_sources'].items()}
    controllers = {key: _ref(Path(item['path'])) for key, item in old['controller_sources'].items()}
    hardware = campaign.campaign_hardware_policy(prefix.PHASE)
    hardware['physical_power_limit_w'] = 250
    plan = _seal({
        'schema_version': campaign.PLAN_SCHEMA, 'decision': 'PASS_PREPARED',
        'campaign_id': 'GX1_DYNAMIC_M5_OCTOBER_' + arm.upper() + '_FIT_20261106_'
            + ('002' if arm == 'candidate' else '001'),
        'phase': prefix.PHASE, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_repo': str(repo), 'source_commit': head, 'runtime_root': str(runtime),
        'gpu_uuid': old['gpu_uuid'], 'prepared_windows_boot': boot,
        'recipe': binding['recipe'], 'invocations': invocations,
        'signed_guard_sources': guards, 'controller_sources': controllers,
        'policy': hardware,
        'authority': {'test': False, 'promotion': False, 'paper': False,
                      'live': False, 'cloud_spend': False},
        'test_data_used': False}, 'plan_sha256')
    checked = campaign.require_plan(plan, verify_files=True)
    require(len(checked['checked_invocations']) == recipe['max_windows'],
            'FIT_MATERIALIZER_CHECKED_PLAN')
    plan_ref = _publish(out/'CAMPAIGN_PLAN.json', plan)
    _publish(runtime/'EXECUTION_BINDING.json', {
        'plan': plan_ref, 'source_head': head, 'policy': binding['policy']})
    ready = prefix.require_native_direction_prefix_fit_readiness(repo,
        window_path=Path(windows[0]['path']), window_file_sha256=windows[0]['sha256'])
    require(ready['decision'] == 'READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES'
            and campaign.next_action(checked, [], current_boot=boot)['decision'] == 'REBOOT_REQUIRED'
            and _ref(policy_path) == binding['policy'], 'FIT_MATERIALIZER_REBOOT_GATE')
    os.sync()
    return {'decision': 'PASS_METADATA_PREPARED_REBOOT_REQUIRED', 'arm': arm,
            'plan': plan_ref, 'windows': windows, 'recipe': binding['recipe'],
            'source_head': head, 'genuine_job_started': False, 'test_data_used': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.binding), separators=(',', ':')))


if __name__ == '__main__':
    main()
