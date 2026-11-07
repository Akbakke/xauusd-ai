"""Fold-1 time-query counters must not inherit the legacy prefix geometry."""
from __future__ import annotations

import pytest

from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
from gx1.contracts import market_state_time_query_fold1_fit_v1 as fold1


REF = {'path': str(fold1.RECIPE_PATH), 'sha256': 'a' * 64}
ARTIFACT = {'path': '/tmp/gx1-fold1-campaign-test.json', 'sha256': 'b' * 64}


def _complete_cursor(monkeypatch, **changes):
    monkeypatch.setattr(fold1, 'require_checkpoint_view',
                        lambda value, **kwargs: value)
    args = dict(recipe_ref=REF, fold=fold1.FOLD,
                feature_view_identity={'synthetic': True},
                phase_state='COMPLETE', outcome='COMPLETE',
                optimizer_updates=fold1.MAX_UPDATES,
                parent_cursor=fold1.FIT_ROWS,
                parent_order_sha256='c' * 64,
                full_state=ARTIFACT, own_initial_state=ARTIFACT,
                initial_measurement=ARTIFACT,
                final_train_measurement=ARTIFACT,
                oof_cursor=fold1.SCORE_ROWS, result=ARTIFACT)
    return prefix.build_prefix_fit_cursor(**{**args, **changes})


def test_fold1_cursor_counts_only_later_score_rows(monkeypatch):
    cursor = _complete_cursor(monkeypatch)
    assert cursor['parent_cursor'] == 29574
    assert cursor['oof_cursor'] == 17753
    with pytest.raises(Exception, match='CURSOR_COUNTER:oof_cursor'):
        _complete_cursor(monkeypatch, oof_cursor=18028)
    with pytest.raises(Exception, match='TIME_QUERY_CURSOR_FOLD'):
        _complete_cursor(monkeypatch, fold={**fold1.FOLD,
                                           'oof_start_row': fold1.FIT_ROWS})


def test_fold1_progress_uses_mature_fit_and_gap(monkeypatch):
    cursor = _complete_cursor(monkeypatch)
    invocation = {
        'direction_prefix_fit_recipe': REF,
        'optimizer_step_budget': fold1.MAX_UPDATES,
        'kind': prefix.KIND,
        'invocation_sha256': 'd' * 64,
        'checkpoint': {'pointer_path': '/tmp/gx1-fold1-campaign-cursor.json'},
    }
    window = {'recipe': REF}
    plan = {'plan_sha256': 'e' * 64}
    pointer = {'path': invocation['checkpoint']['pointer_path'], 'sha256': 'f' * 64}
    progress = prefix.build_prefix_fit_progress(
        window=window, plan=plan, invocation=invocation,
        checkpoint_pointer=pointer, cursor=cursor)
    assert progress['total_units'] == 29574
    assert progress['oof_cursor'] == 17753
    assert progress['outcome'] == 'COMPLETE'
    bad = dict(progress, oof_cursor=18028)
    bad.pop('progress_sha256')
    bad['progress_sha256'] = prefix.campaign.canonical_sha256(bad)
    with pytest.raises(Exception, match='PROGRESS_COUNTER:oof_cursor'):
        prefix.require_prefix_fit_progress(
            bad, plan_sha256=plan['plan_sha256'], invocation=invocation,
            expected_selection_receipt_sha256=None, verify_file=False)


def test_one_use_receipt_shapes_accept_honest_complete_and_failed():
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    boot_body = {
        'schema_version': campaign.BOOT_SCHEMA, 'computer_name': 'ANDRE2',
        'last_boot_utc': '2026-11-07T00:00:00Z', 'boot_id': 1,
    }
    boot = {**boot_body, 'identity_sha256': campaign.canonical_sha256(boot_body)}
    pointer = str(fold1.RUNTIME_ROOT / 'prefix-cursor' / 'RESUME_CURSOR.json')
    evidence = (fold1.RUNTIME_ROOT / 'invocation-evidence' /
                'invocation-0001')
    invocation = {
        'invocation_sha256': 'd' * 64,
        'invocation_number': 1,
        'invocation_id': 'invocation-0001',
        'kind': prefix.KIND,
        'requires_fresh_windows_boot': True,
        'expected_success_outcome': 'RESUMABLE_OR_COMPLETE',
        'checkpoint': {'before_mode': 'GENESIS', 'pointer_path': pointer},
        'progress_path': str(fold1.RUNTIME_ROOT / 'progress' /
                             'invocation-0001.json'),
    }
    unsigned = {
        'schema_version': campaign.RECEIPT_SCHEMA,
        'plan_sha256': 'e' * 64,
        'invocation_sha256': invocation['invocation_sha256'],
        'invocation_number': 1,
        'invocation_id': 'invocation-0001',
        'kind': prefix.KIND,
        'selection_receipt_sha256': None,
        'boot': boot,
        'started_utc': '2026-11-07T15:00:00+00:00',
        'finished_utc': '2026-11-07T15:01:00+00:00',
        'outcome': 'COMPLETE',
        'trainer_guard_exit_code': 0,
        'progress_observer_exit_code': 0,
        'pointer_before_sha256': 'GENESIS',
        'checkpoint_pointer_after': {'path': pointer, 'sha256': 'a' * 64},
        'checkpoint_pointer_snapshot': {
            'path': str(evidence / 'CHECKPOINT_POINTER.json'),
            'sha256': 'a' * 64,
        },
        'progress': {'path': str(evidence / 'PROGRESS.json'),
                     'sha256': 'b' * 64},
        'guard_log': {'path': str(evidence / 'SIGNED_GUARD.log'),
                      'sha256': 'c' * 64},
        'guard_decision': 'PASS',
        'signed_guard_telemetry_owner': 'gx1_guarded_trainer_exec.sh',
        'active_marker_sha256': 'f' * 64,
        'test_data_used': False,
    }

    def checked(value):
        sealed = {**value, 'receipt_sha256': campaign.canonical_sha256(value)}
        return campaign.require_receipt(
            sealed, plan_sha256=unsigned['plan_sha256'],
            invocation=invocation, verify_files=False)

    assert checked(unsigned)['outcome'] == 'COMPLETE'
    failed = {**unsigned, 'outcome': 'FAILED',
              'trainer_guard_exit_code': 1, 'guard_decision': 'FAILED'}
    assert checked(failed)['outcome'] == 'FAILED'
    with pytest.raises(Exception, match='failed receipt has successful processes'):
        checked({**failed, 'trainer_guard_exit_code': 0})
