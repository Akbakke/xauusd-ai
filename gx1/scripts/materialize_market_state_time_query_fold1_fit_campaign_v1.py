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
    repo = fold1.CANONICAL_REPO
    root = fold1.RUNTIME_ROOT.parent
    runtime = fold1.RUNTIME_ROOT
    _need(binding_path == root / 'PREPARATION_EXECUTION_BINDING.json', 'BINDING_PATH')
    binding = campaign.read_bound_json(binding_path, campaign.file_sha256(binding_path))
    _need(set(binding) == {'materializer', 'recipe', 'policy', 'prepared_boot',
                           'guard_template', 'source_head'}, 'BINDING_FIELDS')
    source = repo / 'gx1/scripts/materialize_market_state_time_query_fold1_fit_campaign_v1.py'
    _need(binding['materializer'] == _ref(source)
          and binding['recipe']['path'] == str(fold1.RECIPE_PATH)
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
    gate = policy.get(fold1.STAGE)
    _need(isinstance(gate, dict)
          and gate.get('campaign_preparation_authorized') is True
          and gate.get('recipe') == binding['recipe']
          and gate.get('prepared_boot') == binding['prepared_boot']
          and gate.get('guard_template') == binding['guard_template']
          and gate.get('materializer') == binding['materializer']
          and gate.get('run_root') == str(runtime)
          and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
          and gate.get('max_invocations') == fold1.MAX_WINDOWS
          and gate.get('relaunch_allowed') is False
          and gate.get('test_data_used') is False, 'POLICY_GATE')
    require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    _need(not (root / 'CAMPAIGN').exists()
          and not (root / 'CAMPAIGN_PREPARATION_CLAIM.json').exists()
          and not runtime.exists(), 'GENESIS')
    recipe = prefix.require_prefix_fit_launch_recipe(
        binding['recipe'], source_repo=repo, source_commit=head)
    _need(recipe['run_scope'] == fold1.SCOPE
          and recipe['admission_stage'] == fold1.STAGE
          and recipe['runtime_root'] == str(runtime)
          and recipe['max_windows'] == fold1.MAX_WINDOWS, 'RECIPE')
    fold1.require_fit_gate(gate, recipe)
    boot = campaign.require_boot_identity(campaign.read_bound_json(
        Path(binding['prepared_boot']['path']), binding['prepared_boot']['sha256']))
    old = campaign.read_bound_json(Path(binding['guard_template']['path']),
                                   binding['guard_template']['sha256'])
    _need(old['phase'] == prefix.PHASE and old['test_data_used'] is False,
          'GUARD_TEMPLATE')
    _publish(root / 'CAMPAIGN_PREPARATION_CLAIM.json', {
        'binding': _ref(binding_path), 'source_head': head,
        'genuine_job_started': False, 'test_data_used': False})
    out = root / 'CAMPAIGN'
    out.mkdir()
    runtime.mkdir()
    invocations = []
    windows = []
    for number in range(1, fold1.MAX_WINDOWS + 1):
        window = _seal({
            'schema_version': prefix.WINDOW_SCHEMA, 'recipe': binding['recipe'],
            'invocation_number': number, 'max_invocation_seconds': 12000,
            'progress_path': str(runtime / 'progress' / f'invocation-{number:04d}.json'),
            'campaign_cursor_path': str(runtime / 'prefix-cursor' / 'RESUME_CURSOR.json'),
            'output_directory': str(runtime / 'prefix-fit'), 'test_data_used': False,
        }, 'policy_sha256')
        window_ref = _publish(out / f'WINDOW_{number:04d}.json', window)
        windows.append(window_ref)
        argv = [str(repo / 'scripts/gx1_capped_run.sh'), '--class', 'trainer',
                '--mem', '20G', '--swap', '512M', '--',
                str(repo / '.venv/bin/python'), '-m', prefix.MODULE,
                '--window-policy', window_ref['path'],
                '--window-policy-file-sha256', window_ref['sha256'],
                '--progress-path', window['progress_path']]
        execution = _seal({
            'schema_version': campaign.EXECUTION_MANIFEST_SCHEMA,
            'decision': 'PASS_EXACT_INVOCATION', 'invocation_kind': prefix.KIND,
            'python_module': prefix.MODULE, 'source_commit': head,
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
            'kind': prefix.KIND, 'batch_size': fold1.BATCH_SIZE,
            'epoch_index': None, 'window_index': number - 1,
            'window_count': fold1.MAX_WINDOWS,
            'optimizer_step_budget': recipe['numerical_recipe']['max_updates'],
            'expected_success_outcome': 'RESUMABLE_OR_COMPLETE',
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
            'maximum_wall_seconds': 13800, 'requires_fresh_windows_boot': True,
            'signed_guard_only': True, 'test_data_used': False,
        }, 'invocation_sha256')
        invocations.append(_publish(out / f'INVOCATION_{number:04d}.json', invocation))
    guards = {key: _ref(Path(item['path']))
              for key, item in old['signed_guard_sources'].items()}
    controllers = {key: _ref(Path(item['path']))
                   for key, item in old['controller_sources'].items()}
    hardware = campaign.campaign_hardware_policy(prefix.PHASE)
    hardware['physical_power_limit_w'] = 250
    plan = _seal({
        'schema_version': campaign.PLAN_SCHEMA, 'decision': 'PASS_PREPARED',
        'campaign_id': fold1.STAGE.upper(), 'phase': prefix.PHASE,
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
    _need(len(checked['checked_invocations']) == fold1.MAX_WINDOWS,
          'CHECKED_PLAN')
    plan_ref = _publish(out / 'CAMPAIGN_PLAN.json', plan)
    _publish(runtime / 'EXECUTION_BINDING.json', {
        'plan': plan_ref, 'source_head': head, 'policy': binding['policy']})
    ready = prefix.require_native_direction_prefix_fit_readiness(
        repo, window_path=Path(windows[0]['path']),
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
