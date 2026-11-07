"""Synthetic one-use publication, preclaim, and failure-path checks."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from gx1.audit import run_market_state_tabular_oof_v1 as worker
from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as source


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fake_cohorts():
    result = []
    for i, official in enumerate(worker.target.FOLD_BOUNDARIES_NS[:5], 1):
        result.append({
            'fold_id': i,
            'official_fit_cutoff_ns': official,
            'effective_fit_cutoff_ns': official - source.DAY_NS,
            'joined_fit_rows': source.MATURE_ROWS[i - 1],
            'old_joined_fit_rows': source.OLD_ROWS[i - 1],
            'fit_parent_rows_sha256': str(i) * 64,
            'fit_decision_ns_sha256': str(i) * 64,
        })
    return result


def test_preflight_checks_plan_recipe_and_refs_without_payload_or_claim(monkeypatch, tmp_path):
    cohorts = _fake_cohorts()
    plan_path = tmp_path / 'PLAN.json'
    out_dir = tmp_path / 'RUN'
    monkeypatch.setattr(worker, 'PLAN_PATH', plan_path)
    monkeypatch.setattr(worker, 'OUT_ROOT', out_dir)
    monkeypatch.setattr(worker, 'SOURCE_PATHS', frozenset({
        worker.SOURCE_ADAPTER_PATH, 'gx1/audit/run_market_state_tabular_oof_v1.py'}))
    monkeypatch.setattr(worker, '_reference_chain', lambda: {'cohorts': cohorts})
    monkeypatch.setattr(source, '_provider_module', lambda: SimpleNamespace())
    monkeypatch.setattr(worker.model, '_import_lightgbm', lambda: SimpleNamespace())
    monkeypatch.setattr(worker.target, 'load_bound_five_endpoint_targets',
                        lambda: pytest.fail('preflight read target payload'))
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    monkeypatch.setattr(worker, '_clean_head', lambda: head)
    owners = {path: _sha(worker.REPO / path) for path in worker.SOURCE_PATHS}
    plan = {
        'schema_version': worker.SCHEMA, 'stage': worker.STAGE,
        'status': 'PREPARED_NOT_ADMITTED', 'output_root': str(out_dir),
        'max_invocations': 1, 'fold_ids': worker.FOLD_IDS,
        'score_axes': json.loads(json.dumps(worker.SCORE_AXES)),
        'command_template': worker.planned_command('{plan_sha256}'),
        'input_refs': worker.INPUT_REFS, 'cohorts': cohorts,
        'execution': worker.EXECUTION, 'limits': worker.LIMITS,
        'recipe': worker.model.fixed_recipe(),
        'decision_rule': worker.DECISION_RULE,
        'research_fit_authorized': True,
        'fixed_tabular_fits_authorized': worker.HEAD_COUNT,
        'native_training_authorized': False, 'test_data_used': False,
        'source_base_head': head, 'source_owners': owners,
    }
    plan_path.write_text(json.dumps(plan))
    actual = _sha(plan_path)
    got, ref, observed_head, policy_ref = worker.preflight(
        plan_path, actual, out_dir)
    assert got == plan and ref['sha256'] == actual
    assert observed_head == head and policy_ref is None
    assert not out_dir.exists()
    plan['score_axes'][0]['score_rows'] += 1
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(RuntimeError, match='PLAN_SCOPE'):
        worker.preflight(plan_path, _sha(plan_path), out_dir)


def _run_fixture(monkeypatch, tmp_path, *, fail_fold=None):
    monkeypatch.setattr(worker, 'RUNS', tmp_path)
    out_dir = tmp_path / 'RUN'
    parents = np.arange(6, dtype=np.int64)
    decisions = np.arange(100, 106, dtype=np.int64)
    view = SimpleNamespace(
        parent_rows=parents, decision_ns=decisions, width=2,
        feature_names=('one', 'two'), categorical=(),
        take=lambda positions, *, max_rows:
            np.column_stack((positions, positions + 1)).astype(np.float32))
    axes = {'parent_rows': parents, 'decision_ns': decisions}
    prepared = {'axes': axes,
                'expected_axes': {'parent_rows_sha256': 'a' * 64,
                                  'decision_ns_sha256': 'b' * 64}}
    cohorts = []
    priors = {}
    for i in range(1, 6):
        cohorts.append({
            'fold_id': i, 'official_fit_cutoff_ns': i * 10,
            'effective_fit_cutoff_ns': i * 10 - 1,
            'joined_fit_rows': i, 'old_joined_fit_rows': i,
            'fit_parent_rows_sha256': str(i) * 64,
            'fit_decision_ns_sha256': str(i) * 64})
        priors[i] = {'fold_id': i, 'probabilities': [[0.3, 0.4, 0.3]] * 5}
    score_axes = []
    for i in range(1, 6):
        score_axes.append({'fold_id': i, 'score_rows': 1,
                           'score_parent_rows_sha256': str(i) * 64,
                           'score_decision_ns_sha256': str(i + 1) * 64})
    plan = {'cohorts': cohorts, 'score_axes': score_axes,
            'source_owners': {}, 'decision_rule': worker.DECISION_RULE}
    plan_ref = {'path': 'plan', 'sha256': 'a' * 64}
    policy_ref = {'path': 'policy', 'sha256': 'b' * 64}
    monkeypatch.setattr(worker, 'preflight', lambda *args, **kwargs:
                        (plan, plan_ref, 'head', policy_ref))
    monkeypatch.setattr(worker, '_reference_chain', lambda:
                        {'provider_plan': {}, 'gc_layout': {}})
    provider = SimpleNamespace(validate_metadata=lambda _: {})
    monkeypatch.setattr(source, '_provider_module', lambda: provider)
    monkeypatch.setattr(source, '_prepare_inputs', lambda *args: prepared)
    monkeypatch.setattr(source, '_validate_axes', lambda *_: None)
    monkeypatch.setattr(worker, '_view', lambda _: view)
    bound = SimpleNamespace(parent_rows=parents, decision_ns=decisions)
    monkeypatch.setattr(worker.target, 'load_bound_five_endpoint_targets', lambda: bound)
    def fit_targets(_bound, *, fold_id):
        cohort = dict(cohorts[fold_id - 1])
        cohort.update(fit_parent_rows=parents[:fold_id].copy(),
                      fit_decision_ns=decisions[:fold_id].copy(),
                      score_parent_rows=np.asarray([fold_id], dtype=np.int64),
                      score_decision_ns=np.asarray([100 + fold_id], dtype=np.int64),
                      score_parent_rows_sha256=str(fold_id) * 64,
                      score_decision_ns_sha256=str(fold_id + 1) * 64)
        return {'cohort': cohort, 'prior': priors[fold_id], 'binary': {'fit_fold': fold_id}}
    monkeypatch.setattr(worker, 'mature_fold_training_targets', fit_targets)
    events = []
    class Booster:
        def model_to_string(self):
            return 'synthetic fixed model'
    def fit_model(train, score, binary, *, feature_names, categorical):
        i = binary['fit_fold']
        assert (out_dir / 'PRIOR_FROZEN.json').is_file()
        assert train.shape == (i, 2) and score.shape == (1, 2)
        events.append(i)
        if i == fail_fold:
            raise RuntimeError('observed synthetic fit blocker')
        return {'probabilities': np.tile(np.asarray([0.3, 0.4, 0.3]),
                                         (1, 5, 1)),
                'boosters': {tf + ':' + head: Booster()
                             for tf in worker.features.TIMEFRAMES
                             for head in ('activity', 'conditional_side')},
                'fit_counts': {}, 'recipe': worker.model.fixed_recipe()}
    monkeypatch.setattr(worker.model, 'fit_predict_fold', fit_model)
    monkeypatch.setattr(worker, 'paired_mature_fold_scores',
                        lambda *_args, **kwargs:
                        {'fold_id': kwargs['fold_id'], 'score_rows': 1})
    bookends = []
    monkeypatch.setattr(worker, '_source_bookend', lambda *args: bookends.append(True))
    return out_dir, events, bookends


def test_all_five_folds_freeze_prior_first_and_publish_weights_predictions(monkeypatch, tmp_path):
    out_dir, events, bookends = _run_fixture(monkeypatch, tmp_path)
    receipt = worker.run(worker.PLAN_PATH, 'a' * 64, out_dir)
    assert events == [1, 2, 3, 4, 5] and len(bookends) == 6
    assert (out_dir / 'PRIOR_FROZEN.json').is_file()
    assert (out_dir / 'FEATURE_SCHEMA.json').is_file()
    for i in range(1, 6):
        assert (out_dir / f'FOLD_{i}.json').is_file()
        with np.load(out_dir / f'PREDICTIONS_FOLD_{i}.npz', allow_pickle=False) as saved:
            assert list(saved.files) == ['parent_rows', 'decision_ns', 'probabilities']
            assert saved['parent_rows'].tolist() == [i]
            assert saved['probabilities'].shape == (1, 5, 3)
    assert len(list(out_dir.glob('MODEL_FOLD_*.txt'))) == 50
    result = json.loads((out_dir / 'RESULT.json').read_text())
    terminal = json.loads((out_dir / 'TERMINAL.json').read_text())
    assert result['quality_decision'] == 'PENDING_INDEPENDENT_WEEK_BLOCK_REVIEW'
    assert result['test_data_used'] is False
    assert terminal['status'] == 'COMPLETE' and terminal['fold_count'] == 5
    assert receipt['result']['sha256'] == _sha(out_dir / 'RESULT.json')


def test_second_fold_failure_keeps_first_fold_and_closes_one_use_plan(monkeypatch, tmp_path):
    out_dir, events, _ = _run_fixture(monkeypatch, tmp_path, fail_fold=2)
    with pytest.raises(RuntimeError, match='observed synthetic fit blocker'):
        worker.run(worker.PLAN_PATH, 'a' * 64, out_dir)
    assert events == [1, 2]
    assert (out_dir / 'FOLD_1.json').is_file()
    assert not (out_dir / 'FOLD_2.json').exists()
    assert not (out_dir / 'RESULT.json').exists()
    failed = json.loads((out_dir / 'FAILED.json').read_text())
    terminal = json.loads((out_dir / 'TERMINAL.json').read_text())
    assert len(failed['completed_folds']) == 1
    assert terminal['status'] == 'FAILED' and terminal['fold_count'] == 1
    assert terminal['no_relaunch_under_this_plan'] is True


@pytest.mark.parametrize('field,value', [
    ('training_enabled', True), ('native_training_enabled', True),
    ('native_launch_authorized', True),
])
def test_global_policy_training_flags_stay_closed(field, value, tmp_path):
    ref = {'path': 'plan', 'sha256': 'a' * 64}
    gate = {'status': 'ADMITTED', 'decision': 'ADMITTED',
            'plan': ref, 'execution_authorized': True,
            'data_reads_authorized': True, 'research_fit_authorized': True,
            'fixed_tabular_fits_authorized': 50,
            'native_training_authorized': False,
            'claimed_invocations': 0, 'completed_invocations': 0,
            'max_invocations': 1, 'relaunch_allowed': False,
            'memory_max': '12G', 'swap_max': '512M',
            'wall_seconds': 10800, 'test_data_used': False,
            'checkpoint_reads_authorized': False,
            'broker_access_allowed': False, 'test_authorized': False,
            'output_root': str(tmp_path)}
    policy = {'training_enabled': False, 'native_training_enabled': False,
              'native_launch_authorized': False, 'full_val_allowed': False,
              worker.STAGE: gate}
    worker._policy_gate(policy, ref, tmp_path)
    policy[field] = value
    with pytest.raises(RuntimeError, match='POLICY_GATE'):
        worker._policy_gate(policy, ref, tmp_path)
