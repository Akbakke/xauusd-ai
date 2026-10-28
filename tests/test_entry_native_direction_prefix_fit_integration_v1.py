"""Canonical transport regressions; no model/data or hardware execution."""
from pathlib import Path

import pytest

from gx1.contracts import local_random_access_campaign_v2 as campaign

REPO = Path('/home/andre2/src/GX1_CURRENT')
HEAD = 'a' * 40
KIND = 'native_direction_prefix_fit_window'


def invocation(*, kind=KIND, batch_size=16, outer_class='trainer', maximum_wall_seconds=13800):
    argv = [str(REPO/'scripts/gx1_capped_run.sh'), '--class', outer_class,
        '--mem', '20G', '--swap', '512M', '--', str(REPO/'.venv/bin/python'),
        '-m', 'gx1.scripts.run_unified_exit_native_candidate_window_v1',
        '--window-policy', '/tmp/prefix-fit/window.json', '--window-policy-file-sha256', 'a'*64,
        '--progress-path', '/tmp/prefix-fit/progress/invocation-0001.json']
    value = {'schema_version':campaign.INVOCATION_SCHEMA, 'decision':'PASS',
        'invocation_number':1, 'invocation_id':'invocation-0001', 'kind':kind,
        'batch_size':batch_size, 'epoch_index':None, 'window_index':0, 'window_count':4,
        'optimizer_step_budget':2976, 'expected_success_outcome':'RESUMABLE_OR_COMPLETE',
        'source_commit':HEAD, 'execution_manifest':{'path':'/tmp/prefix-fit/execution.json','sha256':'b'*64},
        'launcher_argv':argv, 'launcher_argv_sha256':campaign.canonical_sha256(argv),
        'progress_path':'/tmp/prefix-fit/progress/invocation-0001.json',
        'guard_log_path':'/tmp/prefix-fit/logs/invocation-0001.signed-guard.log',
        'checkpoint':{'pointer_path':'/tmp/prefix-fit/cursor/RESUME_CURSOR.json','before_mode':'GENESIS',
            'predecessor_invocation_number':None,'write_mode':'ATOMIC_UPDATE'},
        'maximum_wall_seconds':maximum_wall_seconds, 'requires_fresh_windows_boot':True,
        'signed_guard_only':True,'test_data_used':False}
    value['invocation_sha256'] = campaign.canonical_sha256(value)
    return value


def check(value):
    # Header/transport only. Exact recipe and physical source are tested in the owner suite.
    return campaign.require_invocation(value, source_repo=REPO, source_commit=HEAD, verify_files=False)


def test_prefix_header_accepts_exact_b16_native_envelope():
    checked=check(invocation())
    assert checked['kind']==KIND and checked['batch_size']==16
    assert checked['maximum_wall_seconds']==13800 and checked['epoch_index'] is None


@pytest.mark.parametrize('batch',[True,2,4,8,16.0])
def test_prefix_header_rejects_other_or_untyped_batches(batch):
    with pytest.raises(campaign.RandomAccessCampaignError,match='batch size'):
        check(invocation(batch_size=batch))


def test_private_research_header_accepts_b2_and_b16_only():
    for batch in (2,16):
        assert check(invocation(kind='native_direction_research_window',batch_size=batch))['batch_size']==batch
    for batch in (True,4,8,16.0):
        with pytest.raises(campaign.RandomAccessCampaignError,match='batch size'):
            check(invocation(kind='native_direction_research_window',batch_size=batch))


@pytest.mark.parametrize('outer_class',['audit','producer'])
def test_prefix_header_cannot_use_nontrainer_class(outer_class):
    with pytest.raises(campaign.RandomAccessCampaignError,match='trainer class'):
        check(invocation(outer_class=outer_class))


def test_prefix_header_rejects_longer_hard_wall():
    with pytest.raises(campaign.RandomAccessCampaignError,match='wall bound'):
        check(invocation(maximum_wall_seconds=13801))


def test_prefix_hardware_values_match_existing_native():
    assert campaign.campaign_hardware_policy('native_direction_prefix_fit') == campaign.campaign_hardware_policy('native_candidate')
    assert campaign.campaign_hardware_policy('native_direction_prefix_fit') == campaign.campaign_hardware_policy('native_direction_research')


def test_complete_prefix_stops_before_unused_windows(monkeypatch):
    monkeypatch.setattr(campaign,'require_receipt_chain',lambda *a,**k:[{'outcome':'COMPLETE'}])
    plan={'phase':'native_direction_prefix_fit','checked_invocations':[{}]*4}
    assert campaign.next_action(plan=plan,receipts=[],current_boot={})=={'decision':'COMPLETE'}


def test_resumable_prefix_exhaustion_blocks_instead_of_claiming_complete(monkeypatch):
    monkeypatch.setattr(campaign,'require_receipt_chain',lambda *a,**k:[{'outcome':'RESUMABLE'}]*4)
    plan={'phase':'native_direction_prefix_fit','checked_invocations':[{}]*4}
    assert campaign.next_action(plan=plan,receipts=[],current_boot={})=={'decision':'BLOCKED_DIRECTION_PREFIX_WINDOWS_EXHAUSTED'}


def test_failed_prefix_stays_blocked(monkeypatch):
    monkeypatch.setattr(campaign,'require_receipt_chain',lambda *a,**k:[{'outcome':'FAILED'}])
    plan={'phase':'native_direction_prefix_fit','checked_invocations':[{}]*4}
    assert campaign.next_action(plan=plan,receipts=[],current_boot={})=={'decision':'BLOCKED_FAILED_INVOCATION'}


def test_prefix_plan_dispatch_does_not_enter_legacy_recipe(monkeypatch):
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as owner
    sentinel={'checked':'prefix-fit'}
    monkeypatch.setattr(owner,'require_prefix_fit_plan',lambda value,**kwargs:sentinel)
    assert campaign.require_plan({'phase':'native_direction_prefix_fit'}) is sentinel


def test_guard_runs_before_window_context(monkeypatch):
    from gx1.scripts import run_unified_exit_native_candidate_window_v1 as native_window
    def reject(**kwargs):
        raise RuntimeError('expected canonical guard rejection')
    def context(*args,**kwargs):
        raise AssertionError('context reached before guard')
    monkeypatch.setattr(native_window.native.trainer,'_require_cuda_trainer_guard_execution',reject)
    monkeypatch.setattr(native_window,'_context',context)
    with pytest.raises(RuntimeError,match='expected canonical guard rejection'):
        native_window.run_window(policy_path=Path('/tmp/prefix-fit/window.json'),
            policy_file_sha256='a'*64,progress_path=Path('/tmp/prefix-fit/progress.json'))
