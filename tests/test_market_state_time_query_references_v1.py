"""Focused one-use reference-driver guards; no full TRAIN score or model."""
import hashlib
import json
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

from gx1.audit import verify_market_state_time_query_references_v1 as audit


def test_preflight_checks_plan_and_small_chain_without_target_payload(monkeypatch, tmp_path):
    plan_path, out_dir = tmp_path / 'PLAN.json', tmp_path / 'RUN'
    monkeypatch.setattr(audit, 'PLAN_PATH', plan_path)
    monkeypatch.setattr(audit, 'OUT_ROOT', out_dir)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    monkeypatch.setattr(audit, '_clean_head', lambda: head)
    calls = []
    monkeypatch.setattr(audit, '_small_chain', lambda: calls.append('small'))
    monkeypatch.setattr(audit, '_load_candidate_tape',
                        lambda: pytest.fail('preflight read candidate payload'))
    owners = {name: hashlib.sha256((audit.REPO / name).read_bytes()).hexdigest()
              for name in audit.SOURCE_PATHS}
    plan = {
        'schema_version': audit.SCHEMA, 'stage': audit.STAGE,
        'status': 'PREPARED_NOT_ADMITTED', 'output_root': str(out_dir),
        'max_invocations': 1, 'input_refs': audit.INPUT_REFS,
        'source_owners': owners, 'source_base_head': head,
        'execution': audit.EXECUTION, 'limits': audit.LIMITS,
        'metric_contract': audit.CONTRACT, 'numpy_version': audit.NUMPY_VERSION,
        'expected_model_rows': 118249,
        'expected_fit_rows': list(audit.EXPECTED_FIT_ROWS),
        'expected_score_rows': list(audit.EXPECTED_SCORE_ROWS),
        'command_template': audit.planned_command('{plan_sha256}'),
        'read_only': True, 'test_data_used': False,
    }
    plan_path.write_text(json.dumps(plan))
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    got, ref, got_head, policy = audit.preflight(plan_path, digest, out_dir)
    assert got == plan and ref['sha256'] == digest and got_head == head
    assert policy is None and calls == ['small'] and not out_dir.exists()


def test_bound_fold_axes_and_m5_endpoint_summary(monkeypatch):
    monkeypatch.setattr(audit, 'EXPECTED_FIT_ROWS', (2,) * 5)
    monkeypatch.setattr(audit, 'EXPECTED_SCORE_ROWS', (2,) * 5)
    monkeypatch.setattr(audit, 'EXPECTED_SCORE_TOTAL', 10)

    class Bound:
        decision_ns = np.arange(118249, dtype=np.int64)

        def mature_oof_axes(self, *, fold_id):
            official = audit.binding.FOLD_BOUNDARIES_NS[fold_id - 1]
            next_ = audit.binding.FOLD_BOUNDARIES_NS[fold_id]
            fit = np.array([official - 10 * audit.binding.M5_NS,
                            official - 9 * audit.binding.M5_NS], np.int64)
            score = np.array([official, official + audit.binding.M5_NS], np.int64)
            fp = np.array([11, 12], np.int64)
            sp = np.array([21, 22], np.int64)
            sha = lambda a: hashlib.sha256(a.tobytes()).hexdigest()
            return {'fit_decision_ns': fit, 'score_decision_ns': score,
                    'fit_parent_rows': fp, 'score_parent_rows': sp,
                    'fit_parent_rows_sha256': sha(fp), 'score_parent_rows_sha256': sha(sp),
                    'fit_decision_ns_sha256': sha(fit),
                    'score_decision_ns_sha256': sha(score),
                    'official_fit_cutoff_ns': official,
                    'effective_fit_cutoff_ns': official - 1440 * audit.binding.MINUTE_NS,
                    'score_end_exclusive_ns': next_}

    monkeypatch.setattr(audit.binding, 'load_bound_five_endpoint_targets', Bound)

    def fake_score(_tape, fit, score, **kwargs):
        assert len(fit) == len(score) == 2
        assert kwargs['fold_boundary_ns'] in audit.binding.FOLD_BOUNDARIES_NS
        sha = lambda a: hashlib.sha256(a.tobytes()).hexdigest()
        cell = {'fit': {}, 'prior': {}, 'score': {
            'activity_known': 1, 'side_known': 1, 'singleton': 1,
            'activity_prior_brier_sum': .2, 'side_prior_brier_sum': .3,
            'three_class_prior_brier_sum': .4,
            'activity_persistence_count': 1,
            'activity_persistence_brier_sum': 1,
            'activity_prior_paired_brier_sum': .2,
            'side_persistence_count': 1,
            'side_persistence_brier_sum': 1,
            'side_prior_paired_brier_sum': .3}}
        return {'fit_issues': len(fit), 'score_issues': len(score),
                'fit_issue_axis_sha256': sha(fit), 'score_issue_axis_sha256': sha(score),
                'model_fits': 0, 'test_data_used': False,
                'per_tf_k': {tf: {str(k): cell for k in range(1, int(n) + 1)}
                             for tf, n in zip(audit.core.TIMEFRAMES, audit.core.NATIVE_K)},
                'weekly_by_issue_monday_utc': {tf: {} for tf in audit.core.TIMEFRAMES},
                'score_source_reuse': {tf: {} for tf in audit.core.TIMEFRAMES}}

    monkeypatch.setattr(audit.core, 'score_reference_fold', fake_score)
    got = audit._fold_references(object(), time.monotonic() + 1000)
    assert got['bound_model_issue_rows'] == 118249
    assert got['oof_score_rows'] == 10 and len(got['folds']) == 5
    assert got['m5_k1_totals']['side_known'] == 5
    assert got['m5_k1_totals']['side_prior_brier_mean'] == pytest.approx(.3)


def test_postclaim_failure_writes_failed_terminal_once(monkeypatch, tmp_path):
    out_dir = tmp_path / 'RUN'
    plan_ref = {'path': 'plan', 'sha256': 'a' * 64}
    policy_ref = {'path': 'policy', 'sha256': 'b' * 64}
    monkeypatch.setattr(audit, 'preflight', lambda *args, **kwargs:
                        ({'source_owners': {}}, plan_ref, 'head', policy_ref))
    monkeypatch.setattr(audit, '_load_candidate_tape', lambda: (_ for _ in ()).throw(
                        RuntimeError('synthetic tape blocker')))
    with pytest.raises(RuntimeError, match='synthetic tape blocker'):
        audit.run(audit.PLAN_PATH, 'a' * 64, out_dir)
    assert (out_dir / 'CLAIM.json').exists()
    assert not (out_dir / 'RESULT.json').exists()
    failed = json.loads((out_dir / 'FAILED.json').read_text())
    terminal = json.loads((out_dir / 'TERMINAL.json').read_text())
    assert failed['error'] == 'synthetic tape blocker'
    assert terminal['status'] == 'FAILED' and terminal['exit_code'] == 1
    assert terminal['no_relaunch_under_this_plan'] is True
