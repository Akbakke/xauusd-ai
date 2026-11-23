"""Synthetic metadata/admission/report tests; no payload, native model or CUDA work."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gx1.audit import native_provider_hlc_learning_v1 as m


@pytest.mark.parametrize('admitted', [False, True])
def test_exact_one_epoch_gate_and_training_flags(admitted):
    binding = {'path': '/synthetic/plan.json', 'sha256': 'a' * 64}
    gate = m.gate(binding, admitted)
    assert gate['plan'] == binding and gate['remaining_invocations'] == int(admitted)
    assert gate['execution_authorized'] is admitted and gate['full_epoch_training_allowed'] is admitted
    assert all(gate[key] is False for key in
               ('additional_epochs_allowed', 'test_authorized', 'relaunch_allowed'))
    policy = {key: False for key in m.FLAGS}
    if admitted:
        policy.update(training_enabled=True, native_training_enabled=True,
                      native_launch_authorized=True, full_epoch_training_allowed=True)
    m.validate_flags(policy, admitted)
    for key in m.FLAGS:
        wrong = dict(policy); wrong[key] = not policy[key]
        with pytest.raises(ValueError, match='EXACT_MODE_TRAINING_AUTHORITY'):
            m.validate_flags(wrong, admitted)


def _build_fixture(monkeypatch):
    from gx1.contracts import gx1_capped_execution_v1 as guard
    from gx1.contracts import market_state_ta_integrated_model_v1 as model_owner
    def ref(path):
        return {'path': str(path), 'sha256': 'a' * 64}
    runtime = {'source_commit': '1' * 40, 'gpu': {'uuid': 'synthetic'},
               'library_versions': {'torch': 'synthetic'}}
    recipe = {'passes': 1, 'train_batch_rows': 96, 'eval_batch_rows': 256,
              'checkpoint_every_optimizer_updates': 256, 'seed': 20260911,
              'scheduler': None, 'gradient_clip_norm': 1.,
              'optimizer': {'lr': .0001, 'betas': [.9, .999], 'eps': 1e-8,
                            'weight_decay': .0001, 'amsgrad': False}}
    schedule = {'schema_version': 'synthetic_source_clock_schedule', 'rows': 354550,
                'passes': 1, 'batch_rows': 96, 'batches_per_pass': 3694,
                'logical_batches': 3694, 'optimizer_updates': 3694,
                'skipped_batches': 0, 'shuffle': False, 'drop_last': False,
                'source_rows_sha256': 'b' * 64, 'issue_ns_sha256': 'c' * 64,
                'batch_source_rows_sha256': ['d' * 64] * 3694,
                'batch_issue_ns_sha256': ['e' * 64] * 3694,
                'known_activity_by_batch': [123] * 3694, 'update_flags': [True] * 3694,
                'contract_sha256': 'f' * 64}
    metadata = {'runtime': copy.deepcopy(runtime), 'input_binding': {'synthetic': True},
                'architecture': {'synthetic': True}, 'target_contract': {'synthetic': 'v2'},
                'truth_sha256': '2' * 64, 'prior': ref('/synthetic/prior.json')}
    contract = {'contract_sha256': '3' * 64}
    base = {'metadata': metadata, 'model_contract': contract,
            'initial_reference': {'model_state_sha256': '4' * 64, 'rng_sha256': '5' * 64},
            'inputs': {'baseline_acceptance': ref('/synthetic/initial_acceptance.json'),
                       'original_MAIN_plan': ref('/synthetic/original_main.json'),
                       'baseline_result': ref('/synthetic/initial_result.json')},
            'source_owners': {'synthetic_numeric_owner.py': 'a' * 64}}
    acceptance = {'cold_replay_bit_exact': True, 'optimizer_replay_qualified': True,
                  'matched_full_proof_components': ['proof_' + str(j) for j in range(10)],
                  'source_inputs_policy_both_boot_bookends': True}
    initial = {'full_initialbaseline_qualified': True, 'actual_NN_rows': 354550,
               'all5_TF_initial_logits_and_fullproper_scores_exact_matched_v2_prior': True}
    main = {'recipe': recipe, 'schedule': schedule}
    values = {m.BASE_PLAN['path']: base, m.REPLAY_ACCEPTANCE['path']: acceptance,
              base['inputs']['baseline_acceptance']['path']: initial,
              base['inputs']['original_MAIN_plan']['path']: main,
              metadata['prior']['path']: {'synthetic_prior': True}}
    monkeypatch.setattr(m, 'ref', ref)
    monkeypatch.setattr(m, 'read', lambda binding: values[binding['path']])
    monkeypatch.setattr(m, 'adapter_bridge', lambda: {'synthetic_append_only_proof': True})
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *a, **k: b'synthetic_guard_source')
    monkeypatch.setattr(m.core, 'future_cuda_require_runtime', lambda runtime: True)
    monkeypatch.setattr(guard, 'native_provider_hlc_learning_source_bridge',
                        lambda *a, **k: {'synthetic_guard_bridge': True})
    monkeypatch.setattr(model_owner, 'native_future_state_model_contract',
                        lambda *a, **k: copy.deepcopy(contract))
    return runtime, base, acceptance, initial, main, contract


def test_plan_preserves_recipe_clock_initial_and_excludes_target_cell_count_relabeling(monkeypatch):
    runtime, base, _, _, main, _ = _build_fixture(monkeypatch)
    before = copy.deepcopy(main); plan = m.build_plan(runtime)
    assert main == before and plan['recipe'] == main['recipe']
    assert plan['metadata']['target_contract'] == base['metadata']['target_contract']
    assert plan['initial_reference'] == base['initial_reference']
    assert plan['backbone_gradient_policy'] == 'M5_primary'
    assert plan['schedule']['source_schedule_contract_sha256'] == main['schedule']['contract_sha256']
    assert 'known_activity_by_batch' not in plan['schedule']
    for key in ('source_rows_sha256', 'issue_ns_sha256', 'batch_source_rows_sha256',
                'batch_issue_ns_sha256', 'update_flags', 'logical_batches', 'passes', 'rows'):
        assert plan['schedule'][key] == main['schedule'][key]
    assert plan['limits']['model_forwards'] == 3694 + 1385
    assert plan['limits']['optimizer_steps'] == 3694
    assert plan['limits']['checkpoint_writes'] == 16
    assert plan['limits']['checkpoint_loads'] == plan['limits']['model_restores'] == 0
    assert plan['scientific_scope']['new_epochs_or_selection'] is False
    assert plan['scientific_scope']['chronological_generalization'] == 'UNKNOWN'
    assert plan['scientific_scope']['economic_edge'] == 'UNKNOWN'


@pytest.mark.parametrize('bad', ['replay', 'proof_count', 'initial_rows', 'initial_prior',
                               'passes', 'train_batch', 'eval_batch', 'clock_rows',
                               'clock_updates', 'numeric_owner', 'gpu', 'libraries'])
def test_plan_rejects_unqualified_inputs_or_mismatched_frozen_recipe_and_runtime(monkeypatch, bad):
    runtime, base, replay, initial, main, _ = _build_fixture(monkeypatch)
    if bad == 'replay': replay['cold_replay_bit_exact'] = False
    elif bad == 'proof_count': replay['matched_full_proof_components'].pop()
    elif bad == 'initial_rows': initial['actual_NN_rows'] = 90112
    elif bad == 'initial_prior': initial['all5_TF_initial_logits_and_fullproper_scores_exact_matched_v2_prior'] = False
    elif bad == 'passes': main['recipe']['passes'] = 2
    elif bad == 'train_batch': main['recipe']['train_batch_rows'] = 64
    elif bad == 'eval_batch': main['recipe']['eval_batch_rows'] = 64
    elif bad == 'clock_rows': main['schedule']['rows'] = 354549
    elif bad == 'clock_updates': main['schedule']['update_flags'][5] = False
    elif bad == 'numeric_owner': base['source_owners']['synthetic_numeric_owner.py'] = 'f' * 64
    elif bad == 'gpu': runtime['gpu']['uuid'] = 'foreign'
    else: runtime['library_versions']['torch'] = 'foreign'
    with pytest.raises(ValueError, match='PROVIDER_HLC_LEARNING_'):
        m.build_plan(runtime)


def test_full_clock_geometry_has_no_dropped_tail_and_no_target_dependency():
    axis = np.arange(354550, dtype=np.int64)
    batches = m.core.all_score_batches(axis, 96)
    assert len(batches) == 3694 and len(batches[-1]) == 22
    assert all(len(batch) == 96 for batch in batches[:-1])
    np.testing.assert_array_equal(np.concatenate(batches), axis)
    assert len(m.core.all_score_batches(axis, 256)) == 1385
    assert len(m.core.all_score_batches(axis, 256)[-1]) == 246


def _adapter_fixture(monkeypatch, tmp_path):
    before = b'"""Synthetic old module."""\nVALUE = 1\ndef old_helper():\n    return VALUE\n'
    after = before + b'\n\ndef score_final(model, context):\n    return model, context\n'
    target = tmp_path / m.ADAPTER; target.parent.mkdir(parents=True); target.write_bytes(after)
    monkeypatch.setattr(m, 'REPO', tmp_path)
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *a, **k: before)
    return before, after, target


def test_adapter_bridge_proves_exact_original_bytes_ast_and_only_new_score_function(monkeypatch, tmp_path):
    before, after, _ = _adapter_fixture(monkeypatch, tmp_path)
    proof = m.adapter_bridge()
    assert proof['old_bytes_and_AST_preserved'] is True
    assert proof['old_sha256'] == hashlib.sha256(before).hexdigest()
    assert proof['current_sha256'] == hashlib.sha256(after).hexdigest()
    assert proof['new_authority'] == 'score_final_only'


@pytest.mark.parametrize('bad', ['old_bytes', 'old_whitespace', 'extra_assignment', 'extra_helper'])
def test_adapter_bridge_rejects_old_changes_or_unscoped_new_authority(monkeypatch, tmp_path, bad):
    before, after, target = _adapter_fixture(monkeypatch, tmp_path)
    if bad == 'old_bytes': after = after.replace(b'VALUE = 1', b'VALUE = 2')
    elif bad == 'old_whitespace': after = after.replace(b'VALUE = 1', b'VALUE=1')
    elif bad == 'extra_assignment': after += b'NEW_DEFAULT = 7\n'
    else: after += b'\ndef other_helper():\n    return 1\n'
    target.write_bytes(after)
    with pytest.raises(ValueError, match='PROVIDER_HLC_LEARNING_'):
        m.adapter_bridge()


def _run_fixture(monkeypatch, tmp_path):
    from gx1.contracts import gx1_capped_execution_v1 as guard
    planfile = tmp_path / 'PLAN.json'; policyfile = tmp_path / 'POLICY.json'
    planfile.write_text('{}'); policyfile.write_text('{}')
    monkeypatch.setattr(m, 'PLAN', planfile); monkeypatch.setattr(m, 'POLICY', policyfile)
    monkeypatch.setattr(m, 'ROOT', tmp_path / 'output')
    binding = m.ref(planfile); pref = m.ref(policyfile)
    plan = {'inputs': {'feature_archive': {'path': str(tmp_path / 'features.npz')},
                       'target_arrays': {'path': str(tmp_path / 'targets.npz')}}}
    live = {'physical_windows_boot_id': 1,
            'wsl_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    monkeypatch.setattr(guard, 'require_native_provider_hlc_learning_execution',
                        lambda **kwargs: {'plan': binding, 'output_root': str(m.ROOT)})
    monkeypatch.setattr(m, 'inspect', lambda *a: (plan, binding, pref, live))
    monkeypatch.setattr(m, 'check_sources_inputs', lambda plan: None)
    monkeypatch.setattr(m.core, 'physical_windows_boot_id', lambda: 1)
    monkeypatch.setattr(m.core, 'future_cpu_process_identity', lambda: {'pid': 0})
    monkeypatch.setattr(m.core.common, 'firewall_state', lambda: {'denied_attempts': 0})
    audit_hooks = []; timers = []
    monkeypatch.setattr(m.sys, 'addaudithook', audit_hooks.append)
    monkeypatch.setattr(m.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(m.signal, 'setitimer', lambda *a: timers.append(a))
    monkeypatch.setattr(m, 'physical', lambda *a: {'synthetic_model_calls': 0})
    return plan, binding, pref, live, audit_hooks, timers


@pytest.mark.parametrize('failure', [None, 'physical', 'bookend'])
def test_run_exact_success_or_consumed_failure_reporting_without_payload_or_model(monkeypatch, tmp_path, failure):
    _, binding, pref, _, _, timers = _run_fixture(monkeypatch, tmp_path)
    def physical(*args):
        (m.ROOT / 'SYNTHETIC_CHECKPOINT_MARKER.json').write_text('{"preserved":true}')
        if failure == 'physical': raise ValueError('synthetic_physical_failure')
        return {'synthetic_model_calls': 0}
    monkeypatch.setattr(m, 'physical', physical)
    if failure == 'bookend':
        monkeypatch.setattr(m, 'check_sources_inputs',
                            lambda plan: (_ for _ in ()).throw(ValueError('synthetic_bookend_failure')))
    if failure:
        with pytest.raises(ValueError, match='synthetic_' + failure + '_failure'):
            m.run(binding['sha256'])
    else:
        result = m.run(binding['sha256'])
        assert result['result'] == m.ref(m.ROOT / 'RESULT.json')
        assert result['terminal'] == m.ref(m.ROOT / 'TERMINAL.json')
        payload = json.loads((m.ROOT / 'RESULT.json').read_text())
        assert payload['status'] == 'COMPLETE_ONE_V2_TRAIN_DIAGNOSTIC_EPOCH'
        assert payload['report'] == {'synthetic_model_calls': 0}
        assert payload['source_inputs_policy_boot_unchanged'] is True and payload['test_data_used'] is False
    claim = json.loads((m.ROOT / 'CLAIM.json').read_text())
    terminal = json.loads((m.ROOT / 'TERMINAL.json').read_text())
    assert claim['plan'] == binding and claim['policy_at_claim'] == pref
    assert claim['remaining_invocations'] == 0 and claim['TRAIN_diagnostic_only'] is True
    assert terminal['status'] == ('FAILED_CONSUMED' if failure else 'COMPLETE')
    assert terminal['exit_code'] == int(bool(failure)) and terminal['plan'] == binding
    assert terminal['remaining_invocations'] == 0 and terminal['relaunch_allowed'] is False
    assert terminal['claim'] == m.ref(m.ROOT / 'CLAIM.json')
    assert (m.ROOT / 'RESULT.json').exists() is (failure is None)
    assert (m.ROOT / 'SYNTHETIC_CHECKPOINT_MARKER.json').exists()
    assert timers[-1] == (m.signal.ITIMER_REAL, 0)


def test_run_audit_scope_allows_only_declared_payloads_and_forbids_network(monkeypatch, tmp_path):
    plan, binding, _, _, hooks, _ = _run_fixture(monkeypatch, tmp_path)
    m.run(binding['sha256']); assert len(hooks) == 1
    audit = hooks[0]
    for key in ('feature_archive', 'target_arrays'):
        audit('open', (plan['inputs'][key]['path'],))
    for name in ('INITIAL.pt', 'FINAL.pt', 'FINAL_LOGITS.npz', 'UPDATE000256.pt'):
        audit('open', (str(m.ROOT / name),))
    audit('open', (str(tmp_path / 'plain_metadata.json'),))
    for name in ('foreign.pt', 'foreign.npz', 'foreign.parquet'):
        with pytest.raises(RuntimeError, match='LEARNING_PAYLOAD_PATH_FORBIDDEN'):
            audit('open', (str(tmp_path / name),))
    for event in ('socket.connect', 'socket.getaddrinfo', 'urllib.Request'):
        with pytest.raises(RuntimeError, match='LEARNING_NETWORK_FORBIDDEN'):
            audit(event, ())


@pytest.mark.parametrize('case,verdict', [('good', 'PASS'), ('reversed', 'FAIL'),
                                        ('one_class', 'UNKNOWN'), ('empty', 'UNKNOWN')])
def test_binary_panel_fixed_good_reversed_one_class_and_empty_cases(case, verdict):
    logits = np.array([-2., 2., -1., 1.], np.float32)
    prior = np.zeros(4, np.float32); labels = np.array([False, True, False, True])
    mask = np.ones(4, np.bool_); starts = np.array([10, 10, 20, 20], np.int64)
    if case == 'reversed': logits = -logits
    elif case == 'one_class': labels[:] = True
    elif case == 'empty': mask[:] = False
    panel = m.binary_panel(logits, prior, labels, mask, starts)
    assert panel['binary_fit_verdict'] == verdict
    if case == 'good':
        assert panel['rank_and_variation']['AUC'] == 1
        assert all(panel['binary_fit_criteria'].values())
        assert panel['model']['both_strict_argmax_sides_observed'] is True
        assert panel['delta_model_minus_prior']['brier'] < 0
        assert panel['delta_model_minus_prior']['NLL'] < 0
    elif case == 'reversed':
        assert panel['rank_and_variation']['AUC'] == 0
        assert panel['delta_model_minus_prior']['brier'] > 0
        assert panel['delta_model_minus_prior']['NLL'] > 0
    elif case == 'one_class':
        assert panel['rank_and_variation']['AUC'] is None
        assert panel['verdict_reason'] == 'ONE_TRUE_CLASS_AUC_UNDEFINED'
    else:
        assert panel['count'] == 0 and panel['model'] is None
        assert panel['equal_target_weight'] is None and panel['verdict_reason'] == 'NO_KNOWN_OUTCOMES'


def test_equal_target_scores_average_row_losses_in_each_target_instead_of_mean_logits():
    logits = np.array([4., -4., 0., 1.], np.float32)
    labels = np.array([True, True, True, False]); mask = np.ones(4, np.bool_)
    starts = np.array([10, 10, 10, 20], np.int64)
    panel = m.binary_panel(logits, np.zeros(4, np.float32), labels, mask, starts)
    probability = 1 / (1 + np.exp(-logits.astype(np.float64)))
    brier = (probability - labels) ** 2
    nll = np.where(labels, np.logaddexp(0, -logits.astype(np.float64)),
                   np.logaddexp(0, logits.astype(np.float64)))
    equal = panel['equal_target_weight']
    assert equal['targets'] == 2 and equal['min_known_issue_rows_per_target'] == 1
    assert equal['max_known_issue_rows_per_target'] == 3
    assert panel['model']['brier'] == pytest.approx(brier.mean())
    assert panel['model']['NLL'] == pytest.approx(nll.mean())
    assert equal['model']['brier'] == pytest.approx((brier[:3].mean() + brier[3]) / 2)
    assert equal['model']['NLL'] == pytest.approx((nll[:3].mean() + nll[3]) / 2)
    score_of_mean_logit = ((1 / (1 + np.exp(-logits[:3].mean())) - 1) ** 2 + brier[3]) / 2
    assert not np.isclose(equal['model']['brier'], score_of_mean_logit)


def test_quality_panels_reuse_synthetic_v2_truth_and_split_d1_support_without_old_archive(monkeypatch):
    from tests.test_native_future_state_provider_hlc_baseline_v1 import _setup
    _, metadata, _, truth, _, _ = _setup(monkeypatch)
    context = m.adapter.load_context(metadata)
    outputs = {'activity_logits': np.where(truth['activity'], 2., -2.).astype(np.float32),
               'side_logits': np.where(truth['side_up'], 2., -2.).astype(np.float32)}
    quality = m.quality_panels(context, outputs)
    assert quality['rows'] == 8 and quality['all5_pooled_side_TRAIN_fit'] == 'PASS'
    assert quality['pooled_side_verdict_by_TF'] == dict.fromkeys(m.train.TF, 'PASS')
    assert quality['prior_refitted_per_panel'] is False
    assert quality['D1_support_reconstruction']['old_archive_loaded'] is False
    old = quality['D1_support']['v1_known']['pooled']
    new = quality['D1_support']['newly_known']['pooled']
    assert old['activity']['count'] == 5 and new['activity']['count'] == 2
    assert old['side']['count'] == 0 and new['side']['count'] == 3
    assert old['side']['binary_fit_verdict'] == 'UNKNOWN'
    assert set(quality['by_UTC_target_year']['D1']) == {'2021', '2022', '2023', '2024', '2025'}
    assert quality['by_UTC_target_year']['D1']['2021']['side']['binary_fit_verdict'] == 'UNKNOWN'
    assert quality['all_TF']['D1']['side']['count'] == 3
    assert quality['output_sha256'] == {key: m.core.digest(value) for key, value in outputs.items()}


@pytest.mark.parametrize('bad', ['output_dtype', 'stale_prior', 'observation_algebra'])
def test_quality_panels_reject_changed_shapes_prior_or_observability(monkeypatch, bad):
    from tests.test_native_future_state_provider_hlc_baseline_v1 import _setup
    _, metadata, _, _, _, _ = _setup(monkeypatch); context = m.adapter.load_context(metadata)
    outputs = {key: np.zeros((8, 5), np.float32) for key in ('activity_logits', 'side_logits')}
    if bad == 'output_dtype': outputs['side_logits'] = outputs['side_logits'].astype(np.float64)
    elif bad == 'stale_prior': context['truth_sha256'] = 'f' * 64
    else:
        context['truth'] = dict(context['truth'])
        context['truth']['state_hlc_observed'] = context['truth']['state_hlc_observed'].copy()
        context['truth']['state_hlc_observed'][0, 4] = False
    with pytest.raises(ValueError, match='QUALITY_'):
        m.quality_panels(context, outputs)
