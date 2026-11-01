"""Synthetic authority tests; no genuine archives, models or training."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gx1.scripts import compare_entry_direction_family_saved_v1 as a

DAY = 86_400_000_000_000


def ref(path):
    return {'path': str(path), 'sha256': 'a'*64}


def protocol():
    return {
        'schema_version': 'gx1_current_family_added_value_study_v1',
        'comparison': {'arm_order': list(a.ARMS), 'maximum_refit_arms': 10,
                       'intervention_contract_sha256': 'b'*64,
                       'fixed_training': {'numerical_recipe': {'seed': 20260911, 'max_updates': 2976}}},
        'horizons_minutes': list(range(1, 61)), 'primary_horizon_minutes': 5,
        'uncertainty': {'replicates': 10000, 'seed': 20260911, 'familywise_alpha': .05,
                        'primary_metrics': ['auc', 'logloss', 'brier'],
                        'block_schemes': ['UTC_day', 'UTC_week']},
        'fixed_fold': {'prefix_rows': 47602, 'oof_start_row': 47602, 'oof_end_row': 64890,
                       'train_cutoff_ns': a.matched.TRAIN_END, 'oof_cutoff_ns': a.matched.LATER_END},
        'test_data_used': False}


def plan():
    return {
        'schema_version': 'gx1_family_saved_comparison_plan_v1', 'stage': a.STAGE,
        'source_head': 'f'*40, 'protocol': a.PROTOCOL, 'canonical_axis_source': a.AXIS_SOURCE,
        'sources': {k: ref(a.REPO/k) for k in a.SOURCE_PATHS},
        'arms': {arm: {'terminal': ref(a.STUDY/('FIT_BASELINE_001' if arm == 'baseline' else
                 'FIT_'+arm.upper()+'_001')/('VERIFIED_TERMINAL_002.json' if arm == 'baseline'
                                            else 'VERIFIED_TERMINAL.json'))} for arm in a.ARMS},
        'output_dir': str(a.STUDY/'FAMILY_SAVED_COMPARISON_001/OUTPUT'), 'test_data_used': False}


def policy(p, p_ref):
    stage = {'status': 'ADMITTED', 'plan': p_ref, 'source_head': p['source_head'],
             'max_invocations': 1, 'claimed_invocations': 0, 'completed_invocations': 0,
             'execution_authorized': True, 'genuine_prediction_reads_authorized': True,
             'relaunch_allowed': False, 'source_qualification': ref(a.STUDY/'QUAL/RESULT.json')}
    for key in ('model_forwards_authorized', 'optimizer_steps_authorized', 'checkpoint_reads_authorized',
                'raw_input_reads_authorized', 'test_data_used', 'broker_access_allowed'):
        stage[key] = False
    value = {a.STAGE: stage}
    for key in ('training_enabled', 'native_training_enabled', 'full_epoch_training_allowed',
                'full_val_allowed', 'native_launch_authorized'):
        value[key] = False
    return value


def test_exact_ten_protocol_and_admission():
    p = plan()
    a.require_plan(p)
    a.require_protocol(protocol())
    a.require_admission(policy(p, ref(a.STUDY/'PLAN.json')), p, ref(a.STUDY/'PLAN.json'))


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'duplicate', 'baseline_old', 'foreign_protocol'])
def test_plan_rejects_partial_or_foreign(mutation):
    p = plan()
    if mutation == 'missing':
        del p['arms'][a.ARMS[-1]]
    elif mutation == 'extra':
        p['arms']['extra'] = p['arms'][a.ARMS[-1]]
    elif mutation == 'duplicate':
        p['arms'][a.ARMS[-1]] = p['arms'][a.ARMS[-2]]
    elif mutation == 'baseline_old':
        p['arms']['baseline']['terminal'] = ref(a.STUDY/'FIT_BASELINE_001/VERIFIED_TERMINAL.json')
    else:
        p['protocol'] = {**p['protocol'], 'sha256': '0'*64}
    with pytest.raises(RuntimeError):
        a.require_plan(p)


@pytest.mark.parametrize('key,value', [('replicates', 9999), ('seed', 0), ('familywise_alpha', .1)])
def test_fixed_uncertainty_rejected(key, value):
    p = protocol()
    p['uncertainty'][key] = value
    with pytest.raises(RuntimeError):
        a.require_protocol(p)


@pytest.mark.parametrize('key,value', [('claimed_invocations', 1), ('relaunch_allowed', True),
                                     ('raw_input_reads_authorized', True), ('source_head', '0'*40)])
def test_admission_rejects_consumed_or_expanded(key, value):
    p = plan()
    r = ref(a.STUDY/'PLAN.json')
    pol = policy(p, r)
    pol[a.STAGE][key] = value
    with pytest.raises(RuntimeError):
        a.require_admission(pol, p, r)


def metadata_chain(arm):
    p = plan()
    root = Path(p['arms'][arm]['terminal']['path']).parent
    docs = {}
    def put(name, value):
        r = ref(root/name)
        docs[r['path']] = value
        return r
    proto = protocol()
    payloads = {'train': ref(root/'RUNTIME/prefix-fit/window-0002/FINAL_TRAIN.npz'),
                'later': ref(root/'RUNTIME/prefix-fit/window-0002/OOF.npz')}
    initial = ref(root/'PRIVATE/INITIAL.npz')
    initial_report = ref(root/'PRIVATE/INITIAL.json')
    priv = put('PRIVATE_REVIEW.json', {'family_arm': arm,
        'status': 'PASS_CONSUMED_FAMILY_PRIVATE_NATIVE_MECHANICS',
        'unperturbed_initial_prediction_shared_state_parity': True,
        'initial_predictions': initial, 'initial_report': initial_report,
        'qualification_weights_for_main_fit': False, 'test_data_used': False})
    recipe = {'run_scope': 'single_prefix_m5_family_value_sign_one_pass', 'family_arm': arm,
              'runtime_root': str(root/'RUNTIME'), 'dense_evaluation_protocol': a.PROTOCOL,
              'family_intervention_sha256': proto['comparison']['intervention_contract_sha256'],
              'fold': proto['fixed_fold'], 'numerical_recipe': proto['comparison']['fixed_training']['numerical_recipe'],
              'train_passes': 1, 'final_train_measurement_rows': 47602, 'test_data_used': False,
              'native_dense_qualification': priv, 'decimal_initial_predictions': initial,
              'family_initial_report': initial_report, 'input_manifest': ref(root/'INPUT.json')}
    recipe['recipe_sha256'] = a.matched.binding.canonical_sha256(recipe)
    rr = put('RECIPE.json', recipe)
    camp = put('CAMPAIGN.json', {'recipe': rr, 'source_commit': 'f'*40, 'plan_sha256': 'd'*64})
    ex = put('EXECUTION.json', {'plan': camp, 'source_head': 'f'*40})
    receipts = []
    for n in (1, 2):
        receipts.append(put('RECEIPT_%d.json' % n, {'invocation_number': n, 'plan_sha256': 'd'*64,
            'guard_decision': 'PASS', 'trainer_guard_exit_code': 0, 'progress_observer_exit_code': 0,
            'outcome': 'RESUMABLE' if n == 1 else 'COMPLETE', 'test_data_used': False}))
    reports = {}
    for part in ('train', 'later'):
        reports[part] = put(part.upper()+'.json', {
            'predictions': payloads[part], 'fold': recipe['fold'],
            'role': 'FINAL_TRAIN' if part == 'train' else 'OOF', 'test_data_used': False,
            'model_metrics': {'per_horizon': [{} for _ in range(60)]},
            'feature_view_identity': {}, 'own_prefix_prior': {}, 'context_map': {},
            'own_initial_state': initial, 'fresh_identity_sha256': '1'*64})
    result = put('RESULT.json', {'outcome': 'COMPLETE', 'recipe': rr,
        'source_unchanged': True, 'test_data_used': False,
        'final_train_measurement': reports['train'], 'oof_measurement': reports['later']})
    window_review = put('WINDOW_REVIEW.json', {'receipt': receipts[0]})
    review = {'schema_version': 'gx1_family_baseline_training_window_review_v1' if arm == 'baseline'
              else 'gx1_family_training_window_review_v1', 'family_arm': arm,
              'status': 'PASS_X_ONEPASS_FULLTRAIN_LATER_TERMINAL', 'source_head': 'f'*40,
              'next_action': {'decision': 'COMPLETE'}, 'full_train_scored': True, 'later_scored': True,
              'training_passes': 1, 'optimizer_updates': 2976, 'scoring_optimizer_updates': 0,
              'test_data_used': False, 'full_train_predictions': payloads['train'],
              'later_predictions': payloads['later'], 'execution_binding': ex, 'receipt': receipts[-1],
              'training_window_review': window_review, 'result': result}
    if arm == 'baseline':
        del review['family_arm']
    root_ref = put('ACTUAL_ROOT_REVIEW_001.json', review)
    vp = ref(root/'VERIFY_PLAN.json')
    verification = {'schema_version': ('gx1_family_baseline_saved_verification_result_v1'
        if arm == 'baseline' else 'gx1_family_saved_verification_result_v1'),
        'status': 'PASS', 'source_head': 'f'*40, 'canonical_axis_source': a.AXIS_SOURCE,
        'same_metric_owner_recomputation': True, 'model_or_checkpoint_deserialized': False,
        'test_data_used': False, 'inputs': {r['path']: r for r in [root_ref, *payloads.values()]},
        'plan': vp, 'axis_lineage_manifest': recipe['input_manifest'],
        'panels': {part: {'parents': rows, 'horizons': 60} for part, rows in a.ROWS.items()}}
    vr = put('VERIFY_RESULT.json', verification)
    vt = put('VERIFY_TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'result': vr,
        'plan': vp, 'error': None, 'denied_events': [], 'test_data_used': False})
    terminal = {'schema_version': 'gx1_family_baseline_verified_terminal_v1' if arm == 'baseline'
                else 'gx1_family_verified_terminal_v1', 'status': 'COMPLETE', 'exit_code': 0,
                'family_arm': arm, 'source_head': 'f'*40, 'source_unchanged': True,
                'test_data_used': False, 'optimizer_updates': 2976, 'scoring_optimizer_updates': 0,
                'later_predictions': payloads['later'], 'root_review': root_ref,
                'saved_prediction_verification': vr}
    if arm == 'baseline':
        terminal.update(train_parent_rows=47602, later_parent_rows=17288,
                        train_predictions=payloads['train'], saved_prediction_terminal=vt,
                        original_receipt=receipts[-1])
    else:
        terminal.update(train_parents=47602, later_parents=17288,
                        full_train_predictions=payloads['train'], saved_verification_terminal=vt,
                        original_receipts=receipts)
    docs[p['arms'][arm]['terminal']['path']] = terminal
    return p['arms'][arm], proto, docs


@pytest.mark.parametrize('arm', ['baseline', 'momentum_flow_encoder'])
def test_original_terminal_chain_and_baseline_adapter(arm):
    entry, proto, docs = metadata_chain(arm)
    loaded = a.require_arm_evidence(arm, entry, proto, lambda r: docs[r['path']])
    assert loaded['recipe']['family_arm'] == arm
    assert len(loaded['reports']) == 2


@pytest.mark.parametrize('mutation', ['unfinished', 'wrong_arm', 'failed_guard', 'foreign_archive', 'wrong_recipe', 'missing_verification'])
def test_metadata_chain_rejects_mismatch(mutation):
    arm = 'momentum_flow_encoder'
    entry, proto, docs = metadata_chain(arm)
    root = Path(entry['terminal']['path']).parent
    if mutation == 'unfinished':
        docs[entry['terminal']['path']]['status'] = 'RESUMABLE'
    elif mutation == 'wrong_arm':
        docs[entry['terminal']['path']]['family_arm'] = 'baseline'
    elif mutation == 'failed_guard':
        docs[str(root/'RECEIPT_1.json')]['guard_decision'] = 'FAIL'
    elif mutation == 'foreign_archive':
        docs[str(root/'ACTUAL_ROOT_REVIEW_001.json')]['full_train_predictions'] = ref(root/'TEST.npz')
    elif mutation == 'wrong_recipe':
        docs[str(root/'RECIPE.json')]['train_passes'] = 2
    else:
        docs[str(root/'VERIFY_RESULT.json')]['status'] = 'FAIL'
    with pytest.raises(RuntimeError):
        a.require_arm_evidence(arm, entry, proto, lambda r: docs[r['path']])


def arrays(day, first):
    decisions = np.repeat(np.arange(day, day+56, dtype=np.int64)*DAY, 2)
    decisions[1::2] += 300_000_000_000
    sign = np.tile(np.array([1, -1], np.int8), 56)
    data = {'parent_rows': np.arange(first, first+112, dtype=np.int64),
            'bar_start_ns': decisions-300_000_000_000, 'decision_ns': decisions,
            'terminal_time_ns': decisions[:, None]+np.arange(1, 61, dtype=np.int64)[None, :]*60_000_000_000,
            'observed': np.ones((112, 60), np.bool_),
            'sign': np.repeat(sign[:, None], 60, axis=1)}
    data['logit_up'] = data['sign'].astype(np.float32)*.25
    for key, labels in a.matched.CONTEXT_SPEC:
        data['context__'+key] = np.full(112, labels[0][0], np.int8)
    return data


def synthetic_comparison(monkeypatch):
    monkeypatch.setattr(a, 'ROWS', {'train': 112, 'later': 112})
    datasets = {'train': arrays(10, 0), 'later': arrays(100, 112)}
    evidence = {}
    for arm in a.ARMS:
        reports = {}
        for part in datasets:
            reports[part] = {'model_metrics': {'per_horizon': [
                {'horizon_minutes': j, 'counts': {'eligible_cells': 112},
                 'accuracy': 1., 'logloss': float(np.logaddexp(0., -.25)),
                 'brier': float((1/(1+np.exp(.25)))**2), 'rank': {'auc': 1.}}
                for j in range(1, 61)]}}
        evidence[arm] = {'payloads': {p: {'partition': p, 'arm': arm} for p in datasets}, 'reports': reports}
    axis = {'partitions': {p: {'selected_parent_rows': 112,
            **{k+'_sha256': a.matched.saved.digest(data[k], np) for k in ('parent_rows', 'decision_ns')}}
            for p, data in datasets.items()}}
    proto = {'fixed_fold': {'train_cutoff_ns': 80*DAY, 'oof_cutoff_ns': 170*DAY}}
    original = a.matched.compare_family_value_predictions
    requested = []
    def bounded_synthetic(*args, **kwargs):
        requested.append((kwargs['replicates'], kwargs['seed'], kwargs['familywise_alpha']))
        assert kwargs['replicates'] == 10000 and kwargs['seed'] == 20260911
        # Only the test replaces resampling count. Production exposes no override.
        return original(*args, **{**kwargs, 'replicates': 16})
    monkeypatch.setattr(a.matched, 'compare_family_value_predictions', bounded_synthetic)
    reader = lambda ref, rows: copy.deepcopy(datasets[ref['partition']])
    return evidence, axis, proto, reader, requested


def test_ten_arm_call_54_contrasts_and_shared_weights(monkeypatch):
    evidence, axis, proto, reader, requested = synthetic_comparison(monkeypatch)
    result = a.compare_archives(evidence, axis, proto, reader=reader)
    assert requested == [(10000, 20260911, .05)]
    assert len(result['primary_five_minute']) == 9
    assert result['per_interval_alpha'] == .05/54
    assert all(len(result['panels'][p]) == 60 for p in ('train', 'later'))
    for i in (0, 1):
        assert len({x['intervals'][i]['weights_sha256'] for x in result['primary_five_minute']}) == 1
    assert result['economic_edge_qualified'] is False


def test_missing_arm_fails_before_archive_read(monkeypatch):
    evidence, axis, proto, _, _ = synthetic_comparison(monkeypatch)
    evidence.pop(a.ARMS[-1])
    def no_read(*args):
        pytest.fail('partial comparison opened an archive')
    with pytest.raises(RuntimeError, match='ALL_TEN'):
        a.compare_archives(evidence, axis, proto, reader=no_read)


@pytest.mark.parametrize('mutation', ['canonical_axis', 'pair_target', 'original_score'])
def test_archive_pair_and_report_mismatch(monkeypatch, mutation):
    evidence, axis, proto, reader, _ = synthetic_comparison(monkeypatch)
    if mutation == 'canonical_axis':
        axis['partitions']['train']['parent_rows_sha256'] = '0'*64
    elif mutation == 'original_score':
        evidence[a.ARMS[-1]]['reports']['later']['model_metrics']['per_horizon'][4]['accuracy'] = .5
    original_reader = reader
    def modified(ref, rows):
        value = original_reader(ref, rows)
        if mutation == 'pair_target' and ref['arm'] == a.ARMS[-1]:
            value['sign'][0, 0] *= -1
        return value
    with pytest.raises(RuntimeError):
        a.compare_archives(evidence, axis, proto, reader=modified)


def disk_ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def test_saved_archive_reader_hash_and_shape(tmp_path):
    path = tmp_path/'FINAL_TRAIN.npz'
    np.savez(path, **arrays(10, 0))
    r = disk_ref(path)
    assert a.read_prediction(r, 112)['sign'].shape == (112, 60)
    with pytest.raises(RuntimeError, match='ARRAY_DOMAIN'):
        a.read_prediction(r, 111)
    with pytest.raises(RuntimeError, match='HASH_OR_FREEZE'):
        a.read_prediction({**r, 'sha256': '0'*64}, 112)
    alias = tmp_path/'alias.npz'
    alias.symlink_to(path)
    with pytest.raises(RuntimeError, match='SYMLINK'):
        a.read_prediction({**r, 'path': str(alias)}, 112)


def test_atomic_publication_rejects_rerun(tmp_path):
    path = tmp_path/'RESULT.json'
    r = a.atomic_json(path, {'status': 'COMPLETE'})
    assert r == disk_ref(path)
    with pytest.raises(FileExistsError):
        a.atomic_json(path, {'status': 'OTHER'})
    assert json.loads(path.read_text()) == {'status': 'COMPLETE'}


def test_bound_junit_sources_and_exact_test_modules(tmp_path, monkeypatch):
    monkeypatch.setattr(a, 'REPO', tmp_path/'repo')
    monkeypatch.setattr(a, 'STUDY', tmp_path/'study')
    a.STUDY.mkdir()
    tests = {}
    for name in a.TEST_PATHS:
        path = a.REPO/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic test source\n')
        tests[name] = disk_ref(path)
    junit = a.STUDY/'pytest.xml'
    junit.write_text('<testsuites><testsuite>'+''.join(
        '<testcase classname="'+name[:-3].replace('/', '.')+'" name="test_ok"/>'
        for name in a.TEST_PATHS)+'</testsuite></testsuites>')
    stdout = a.STUDY/'pytest.log'
    stdout.write_text('2 passed\n')
    value = {'status': 'PASS', 'sources': {}, 'tests': tests, 'genuine_payloads_read': 0,
             'test_data_used': False, 'pytest': {'exit_code': 0, 'test_files': list(a.TEST_PATHS),
                 'junit': disk_ref(junit), 'stdout': disk_ref(stdout)}}
    a.require_source_qualification(value, {}, {})
    bad = copy.deepcopy(value)
    bad['pytest']['test_files'].pop()
    with pytest.raises(RuntimeError):
        a.require_source_qualification(bad, {}, {})
    (a.REPO/a.TEST_PATHS[0]).write_text('# changed\n')
    with pytest.raises(RuntimeError):
        a.require_source_qualification(value, {}, {})


def test_existing_guard_is_used_for_sealed_payloads(monkeypatch, tmp_path):
    hooks = []
    monkeypatch.setattr(a.matched.sys, 'addaudithook', hooks.append)
    a.matched._install_guard([], set(), tmp_path)
    for path in ('/home/andre2/GX1_DATA/raw.json', '/tmp/unbound.npz', '/tmp/model.pt'):
        with pytest.raises(RuntimeError):
            hooks[0]('open', (path, 'r', 0))
    with pytest.raises(RuntimeError):
        hooks[0]('import', ('torch',))


def test_cli_writes_bound_result_terminal_and_rejects_relaunch(tmp_path, monkeypatch):
    evidence, axis, proto, reader, _ = synthetic_comparison(monkeypatch)
    monkeypatch.setattr(a, 'REPO', tmp_path/'repo')
    monkeypatch.setattr(a, 'STUDY', tmp_path/'study')
    p = plan()
    output = Path(p['output_dir'])
    output.parent.mkdir(parents=True)
    p_ref = ref(a.STUDY/'PLAN.json')
    pol = policy(p, p_ref)
    for item in evidence.values():
        item['recipe'] = {key: {} for key in ('input_manifest', 'dense_target', 'dense_target_result',
            'dense_source_transition_proof', 'decimal_input_audit', 'decimal_target_refs', 'candidate_mode',
            'parent_order_algorithm', 'fold', 'numerical_recipe')}
        item['payloads'] = {part: {'path': str(a.STUDY/(arm+'_'+part+'.npz')),
            'sha256': 'a'*64, 'partition': part, 'arm': arm}
            for arm in [next(k for k, x in evidence.items() if x is item)] for part in ('train', 'later')}
    metadata = {p_ref['path']: p, str(a.REPO/'NEXT_RUN_POLICY.json'): pol,
                pol[a.STAGE]['source_qualification']['path']: {}, a.PROTOCOL['path']: proto,
                a.AXIS_SOURCE['path']: {**axis, 'status': 'COMPLETE', 'full_population_measured': True,
                                      'source_unchanged': True, 'test_data_used': False}}
    monkeypatch.setattr(a, 'read_metadata', lambda r, tracked: metadata[r['path']])
    monkeypatch.setattr(a, 'require_protocol', lambda _: None)
    monkeypatch.setattr(a, 'require_source_qualification', lambda *args: None)
    monkeypatch.setattr(a, 'require_arm_evidence', lambda arm, *args: evidence[arm])
    monkeypatch.setattr(a, 'read_bytes', lambda _: b'synthetic')
    monkeypatch.setattr(a, 'identity', lambda _: (1, 1, 1, 1, 1))
    monkeypatch.setattr(a.matched, '_git_state', lambda: (p['source_head'], 'work/gx1-current', ''))
    monkeypatch.setattr(a.matched.cap, 'require_capped_cpu_audit_execution',
                        lambda: {'cgroup_path': '/sys/fs/cgroup/synthetic'})
    monkeypatch.setattr(a.matched.cap, 'require_capped_lock_ancestry', lambda: {'scope': '/synthetic'})
    reads = {}
    monkeypatch.setattr(a.matched, '_install_guard', lambda *args: (reads, []))
    original_compare = a.compare_archives
    def compare(items, axes, pro):
        def counted(ref, rows):
            reads[ref['path']] = reads.get(ref['path'], 0)+1
            return reader(ref, rows)
        return original_compare(items, axes, pro, reader=counted)
    monkeypatch.setattr(a, 'compare_archives', compare)
    argv = ['--plan', p_ref['path'], '--plan-sha256', p_ref['sha256'],
            '--policy-sha256', 'a'*64, '--source-head', p['source_head']]
    assert a.main(argv) == 0
    terminal = json.loads((output/'TERMINAL.json').read_text())
    assert terminal['status'] == 'COMPLETE'
    result = json.loads((output/'RESULT.json').read_text())
    assert result['primary_contrasts'] == 54 and result['payload_archives_read'] == 20
    with pytest.raises(RuntimeError, match='UNCLAIMED_OUTPUT'):
        a.main(argv)


@pytest.mark.parametrize('arm', ['baseline', 'momentum_flow_encoder'])
def test_verification_schema_is_role_specific(arm):
    entry, proto, docs = metadata_chain(arm)
    root = Path(entry['terminal']['path']).parent
    docs[str(root/'VERIFY_RESULT.json')]['schema_version'] = (
        'gx1_family_saved_verification_result_v1' if arm == 'baseline'
        else 'gx1_family_baseline_saved_verification_result_v1')
    with pytest.raises(RuntimeError, match='SAVED_VERIFICATION'):
        a.require_arm_evidence(arm, entry, proto, lambda r: docs[r['path']])

def recovered_metadata_chain(arm='session_regime_encoder'):
    entry, proto, docs = metadata_chain(arm)
    root = Path(entry['terminal']['path']).parent
    oldroot = a.STUDY/('FIT_SESSION_REGIME_001' if arm=='session_regime_encoder' else 'FIT_CHART_GEOMETRY_001')
    def put(path, value):
        r = ref(path)
        docs[r['path']] = value
        return r
    terminal = docs[entry['terminal']['path']]
    review = docs[str(root/'ACTUAL_ROOT_REVIEW_001.json')]
    recipe = docs[str(root/'RECIPE.json')]
    campaign = docs[str(root/'CAMPAIGN.json')]
    campaign['runtime_root'] = str(root/'RUNTIME')
    for key in ('dense_target', 'dense_target_result', 'dense_source_transition_proof',
                'decimal_input_audit', 'decimal_target_refs'):
        recipe[key] = ref(oldroot/(key+'.json'))
    recipe.update(candidate_mode='price', parent_order_algorithm='fixed', max_windows=1,
                  training_enabled=False, full_epoch_training_allowed=False)
    old_recipe = copy.deepcopy(recipe)
    old_recipe['runtime_root'] = str(oldroot/'RUNTIME')
    old_recipe.update(max_windows=4, training_enabled=True, full_epoch_training_allowed=True)
    old_recipe['recipe_sha256'] = a.matched.binding.canonical_sha256(
        {k: v for k, v in old_recipe.items() if k != 'recipe_sha256'})
    old_recipe_ref = put(oldroot/'RECIPE.json', old_recipe)
    old_plan_ref = put(oldroot/'CAMPAIGN/CAMPAIGN_PLAN.json', {
        'recipe': old_recipe_ref, 'source_commit': 'e'*40, 'plan_sha256': 'c'*64,
        'runtime_root': str(oldroot/'RUNTIME')})
    state = ref(oldroot/'RUNTIME/prefix-fit/states/TRAINED.pt')
    initial = ref(oldroot/'RUNTIME/prefix-fit/states/INITIAL.pt')
    initial_measurement = ref(oldroot/'RUNTIME/prefix-fit/window-0001/INITIAL_TRAIN.json')
    old_result_ref = put(oldroot/'RUNTIME/prefix-fit/window-0001/RESULT.json', {
        'recipe': old_recipe_ref, 'outcome': 'RESUMABLE', 'phase_state': 'SCORING',
        'error': None, 'source_unchanged': True, 'optimizer_updates': 2976,
        'parent_cursor': 47602, 'test_data_used': False,
        'durable_checkpoint': {'state': state, 'contract_sha256': 'b'*64}})
    cursor_ref = put(oldroot/'RUNTIME/invocation-evidence/invocation-0001/CHECKPOINT_POINTER.json', {
        'recipe': old_recipe_ref, 'phase_state': 'SCORING', 'outcome': 'RESUMABLE',
        'optimizer_updates': 2976, 'parent_cursor': 47602, 'oof_cursor': 0,
        'final_train_measurement': None, 'full_state': state, 'own_initial_state': initial,
        'initial_measurement': initial_measurement, 'result': old_result_ref})
    training_ref = put(oldroot/'RUNTIME/receipts/invocation-0001.json', {
        'invocation_number': 1, 'plan_sha256': 'c'*64, 'guard_decision': 'PASS',
        'trainer_guard_exit_code': 0, 'progress_observer_exit_code': 0,
        'outcome': 'RESUMABLE', 'test_data_used': False,
        'checkpoint_pointer_snapshot': cursor_ref})
    training_review_ref = put(oldroot/'WINDOW_0001_VERIFIED_REVIEW.json', {
        'status': 'PASS_TRAINING_COMPLETE_SCORING_PENDING', 'family_arm': arm,
        'source_head': 'e'*40, 'receipt': training_ref, 'full_state': state,
        'cursor': ref(oldroot/'RUNTIME/prefix-cursor/RESUME_CURSOR.json')})
    active = ref(oldroot/'RUNTIME/ACTIVE_INVOCATION.json')
    claim = ref(oldroot/'RUNTIME/prefix-fit/window-0002/CLAIM.json')
    integrity_ref = put(oldroot/'INTEGRITY.json', {
        'status': 'TRAINING_STATE_BYTES_PRESERVED_SCORING_INTERRUPTED', 'source_head': 'e'*40,
        'training_receipt': training_ref, 'training_window_review': training_review_ref,
        'full_state': state, 'own_initial_state': initial,
        'cursor': ref(oldroot/'RUNTIME/prefix-cursor/RESUME_CURSOR.json'),
        'native_exit_code_known': False, 'completed_training_updates': 2976,
        'training_parents': 47602, 'originals_modified': False, 'test_data_used': False,
        'interrupted_active_marker': active, 'interrupted_claim': claim})
    peer_ref = put(oldroot/'PEER.json', {
        'status': 'PASS_SCOPED_METADATA_WORKER_REVIEW', 'references': {'integrity': integrity_ref},
        'scope': {'checkpoint_or_payload_bytes_read_by_peer': False, 'test_data_used': False}})
    if arm=='chart_geometry_encoder':
        docs[peer_ref['path']].pop('scope')
        docs[peer_ref['path']].update(
            schema_version='gx1_family_interruption_independent_metadata_peer_review_v1',
            family_arm=arm,test_data_used=False,
            original_native_exit_code_known=False,original_native_exit_code=None)
        docs[integrity_ref['path']].update(
            schema_version='gx1_family_interrupted_scoring_integrity_v1',family_arm=arm)
    closure_ref = put(oldroot/'CLOSURE.json', {
        'schema_version': 'gx1_family_interrupted_scoring_closure_v1',
        'status': 'CLOSED_INTERRUPTED_SCORING_ORIGINALS_PRESERVED', 'family_arm': arm,
        'original_source_head': 'e'*40, 'state_integrity': integrity_ref,
        'interruption_review': peer_ref, 'training_window_review': training_review_ref,
        'interrupted_active_marker': active, 'interrupted_claim': claim,
        'original_relaunch_allowed': False, 'interrupted_native_exit_code_known': False,
        'interrupted_native_exit_code': None, 'claimed_invocations': 2,
        'completed_invocations': 1, 'revoked_invocations': [3, 4], 'test_data_used': False})
    origin = {'schema_version': ('gx1_family_session_scoring_recovery_origin_v1' if arm=='session_regime_encoder'
        else 'gx1_family_chart_scoring_recovery_origin_v1'),
        'family_arm': arm, 'original_source_head': 'e'*40, 'original_recipe': old_recipe_ref,
        'original_plan': old_plan_ref, 'training_receipt': training_ref, 'training_cursor': cursor_ref,
        'training_review': training_review_ref, 'state_integrity': integrity_ref,
        'interruption_review': peer_ref, 'closure_review': closure_ref}
    recipe['scoring_recovery_origin'] = origin
    recipe['recipe_sha256'] = a.matched.binding.canonical_sha256(
        {k: v for k, v in recipe.items() if k != 'recipe_sha256'})
    scoring_ref = put(root/'RUNTIME/receipts/invocation-0001.json', {
        'invocation_number': 1, 'plan_sha256': 'd'*64, 'guard_decision': 'PASS',
        'trainer_guard_exit_code': 0, 'progress_observer_exit_code': 0,
        'outcome': 'COMPLETE', 'test_data_used': False})
    terminal.pop('original_receipts')
    terminal.update(schema_version='gx1_family_recovered_scoring_verified_terminal_v1',
        scoring_recovery_origin=origin, exit_code_scope='recovery_scoring_invocation',
        training_source_head='e'*40, training_receipt=training_ref, scoring_receipt=scoring_ref)
    review.update(schema_version='gx1_family_recovered_scoring_review_v1',
        status='PASS_RECOVERED_ONEPASS_FULLTRAIN_LATER_TERMINAL', scoring_recovery_origin=origin,
        training_window_review=training_review_ref, receipt=scoring_ref)
    result = docs[str(root/'RESULT.json')]
    result.update(scoring_recovery_origin=origin, durable_checkpoint={'contract_sha256': 'b'*64},
        own_initial_state=initial, initial_measurement=initial_measurement, optimizer_updates=2976,
        parent_cursor=47602, error=None, qualification_weights_reused=False,
        counts={**{k: 0 for k in ('backward_calls', 'backward_returns', 'optimizer_step_calls',
            'optimizer_step_returns', 'empty_batches', 'forbidden_legacy_calls')},
            'forward_calls': 4057, 'forward_returns': 4057,
            'forward_kind_calls': {'FINAL_TRAIN': 2976, 'OOF': 1081}})
    return entry, proto, docs, origin


@pytest.mark.parametrize('arm',['session_regime_encoder','chart_geometry_encoder'])
def test_recovered_scoring_keeps_two_distinct_real_campaigns(arm):
    entry, proto, docs, origin = recovered_metadata_chain(arm)
    reads = []
    def read(r):
        reads.append(r['path'])
        return docs[r['path']]
    loaded = a.require_arm_evidence(arm, entry, proto, read)
    assert loaded['recipe']['scoring_recovery_origin'] == origin
    assert all(Path(p).suffix == '.json' for p in reads)
    assert origin['training_receipt']['path'] in reads
    assert docs[entry['terminal']['path']]['scoring_receipt']['path'] in reads
    assert origin['training_cursor']['path'] in reads


@pytest.mark.parametrize('mutation', [
    'fake_original_chain', 'wrong_receipt_plan', 'relabel_original_as_scoring',
    'mutable_cursor', 'incomplete_training', 'missing_integrity', 'false_exit',
    'changed_state', 'extra_update', 'partial_scoring', 'changed_numeric_recipe',
    'changed_initial_state', 'foreign_arm', 'wrong_origin_schema'])
@pytest.mark.parametrize('arm',['session_regime_encoder','chart_geometry_encoder'])
def test_recovered_lineage_rejects_unproven_or_expanded_result(mutation,arm):
    entry, proto, docs, origin = recovered_metadata_chain(arm)
    root = Path(entry['terminal']['path']).parent
    terminal = docs[entry['terminal']['path']]
    result = docs[str(root/'RESULT.json')]
    if mutation == 'fake_original_chain':
        terminal['original_receipts'] = [origin['training_receipt'], terminal['scoring_receipt']]
    elif mutation == 'wrong_receipt_plan':
        docs[terminal['scoring_receipt']['path']]['plan_sha256'] = 'c'*64
    elif mutation == 'relabel_original_as_scoring':
        terminal['scoring_receipt'] = origin['training_receipt']
    elif mutation == 'mutable_cursor':
        origin['training_cursor'] = ref(a.STUDY/'FIT_SESSION_REGIME_001/RUNTIME/prefix-cursor/RESUME_CURSOR.json')
    elif mutation == 'incomplete_training':
        docs[origin['training_cursor']['path']]['optimizer_updates'] = 2975
    elif mutation == 'missing_integrity':
        docs[origin['state_integrity']['path']]['full_state'] = ref(root/'FOREIGN.pt')
    elif mutation == 'false_exit':
        docs[origin['closure_review']['path']]['interrupted_native_exit_code'] = 0
    elif mutation == 'changed_state':
        result['durable_checkpoint']['contract_sha256'] = 'c'*64
    elif mutation == 'extra_update':
        result['counts']['optimizer_step_calls'] = 1
    elif mutation == 'partial_scoring':
        result['counts']['forward_calls'] = result['counts']['forward_returns'] = 4056
    elif mutation == 'changed_numeric_recipe':
        docs[str(root/'RECIPE.json')]['numerical_recipe'] = {'seed': 1, 'max_updates': 2976}
    elif mutation == 'changed_initial_state':
        result['own_initial_state'] = ref(root/'FOREIGN.pt')
    elif mutation == 'foreign_arm':
        origin['family_arm'] = 'momentum_flow_encoder'
    else:
        origin['schema_version'] = 'unrecognized'
    # Preserve the top recipe seal: semantic lineage, not a stale seal, must reject.
    recipe = docs[str(root/'RECIPE.json')]
    recipe['recipe_sha256'] = a.matched.binding.canonical_sha256(
        {k: v for k, v in recipe.items() if k != 'recipe_sha256'})
    with pytest.raises(RuntimeError):
        a.require_arm_evidence(arm, entry, proto, lambda r: docs[r['path']])


@pytest.mark.parametrize('arm',['session_regime_encoder','chart_geometry_encoder'])
def test_recovered_known_arm_schema_cannot_be_crossed(arm):
    entry,proto,docs,origin=recovered_metadata_chain(arm)
    origin['schema_version']=('gx1_family_chart_scoring_recovery_origin_v1' if arm=='session_regime_encoder'
        else 'gx1_family_session_scoring_recovery_origin_v1')
    recipe=docs[str(Path(entry['terminal']['path']).parent/'RECIPE.json')]
    recipe['recipe_sha256']=a.matched.binding.canonical_sha256({k:v for k,v in recipe.items() if k!='recipe_sha256'})
    with pytest.raises(RuntimeError,match='RECOVERY_ORIGIN'):
        a.require_arm_evidence(arm,entry,proto,lambda r:docs[r['path']])


@pytest.mark.parametrize('field,value',[
    ('schema_version','gx1_session_interruption_independent_metadata_peer_review_v1'),
    ('family_arm','session_regime_encoder'),('test_data_used',True),
    ('original_native_exit_code_known',True),('original_native_exit_code',0),
])
def test_chart_recovered_peer_uses_actual_top_level_fields(field,value):
    entry,proto,docs,origin=recovered_metadata_chain('chart_geometry_encoder')
    peer=docs[origin['interruption_review']['path']]
    assert 'scope' not in peer
    peer[field]=value
    # A fabricated session-style scope must not authorize a mismatched chart peer.
    peer['scope']={'checkpoint_or_payload_bytes_read_by_peer':False,'test_data_used':False}
    with pytest.raises(RuntimeError,match='RECOVERY_CHART_PEER'):
        a.require_arm_evidence('chart_geometry_encoder',entry,proto,lambda r:docs[r['path']])
