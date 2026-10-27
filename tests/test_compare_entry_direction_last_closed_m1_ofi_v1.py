"""Synthetic phaseB boundaries only; no genuine payloads or forest fits."""
import hashlib
from types import SimpleNamespace
import numpy as np
import pytest

from gx1.scripts import compare_entry_direction_last_closed_m1_ofi_v1 as worker


def _plan(mode='synthetic'):
    protocol = {'schema_version': 'gx1_independent_direction_last_m1_protocol_v1',
        'design': worker.DESIGN, 'fold': {'id': 1, 'train_parents': worker.N,
            'later_parents': worker.LATER, 'train_cutoff_ns': worker.CUT, 'later_cutoff_ns': worker.END},
        'arms': {'baseline_fields': 3345, 'augmented_fields': 3347, 'new_fields': list(worker.NEW_FIELDS)},
        'max_genuine_fits': 6, 'horizons_minutes': [1, 5, 15], 'baseline_before_candidate': True,
        'wall_seconds': 1200, 'memory_bytes': 4294967296}
    plan = {'schema_version': worker.SCHEMA+'_synthetic', 'mode': mode,
        'design': worker.DESIGN, 'protocol': worker.PROTOCOL, 'inputs': {}, 'metadata': {},
        'sources': {name: {'path': str(worker.REPO/name), 'sha256': 'a'*64}
            for name in (*worker.SOURCE_PATHS, *worker.TEST_FILES)},
        'out_dir': str(worker.ROOT/'synthetic_fixture'), 'wall_seconds': 180,
        'tests': list(worker.TEST_FILES)}
    return plan, {}, protocol


def test_synthetic_plan_binds_only_new_targeted_file_and_exact_protocol():
    plan, design, protocol = _plan()
    assert worker.require_plan(plan, design, protocol) == worker.ROOT/'synthetic_fixture'
    assert plan['tests'] == ['tests/test_compare_entry_direction_last_closed_m1_ofi_v1.py']


@pytest.mark.parametrize('fault', ['protocol', 'fold', 'fields', 'fits', 'wall', 'tests', 'metadata', 'input', 'traversal'])
def test_plan_rejects_changed_protocol_scope_or_external_sources(fault):
    plan, design, protocol = _plan()
    if fault == 'protocol': plan['protocol'] = {**plan['protocol'], 'sha256': 'b'*64}
    elif fault == 'fold': protocol['fold']['train_parents'] += 1
    elif fault == 'fields': protocol['arms']['augmented_fields'] += 1
    elif fault == 'fits': protocol['max_genuine_fits'] += 1
    elif fault == 'wall': plan['wall_seconds'] += 1
    elif fault == 'tests': plan['tests'].append('tests/test_entry_direction_snapshot_data_v1.py')
    elif fault == 'metadata': plan['metadata']['extra'] = {'path': '/fixture.json', 'sha256': 'a'*64}
    elif fault == 'input': plan['inputs']['train_m1'] = {'path': '/fixture.npz', 'sha256': 'a'*64}
    else: plan['sources']['gx1/../escape.py'] = {'path': str(worker.REPO/'gx1/../escape.py'), 'sha256': 'a'*64}
    with pytest.raises(RuntimeError): worker.require_plan(plan, design, protocol)


@pytest.mark.parametrize('suffix', ['../escape.npy', 'nested/../../escape.npy'])
def test_owned_guard_rejects_parent_traversal_even_inside_lexical_output(tmp_path, suffix):
    out = tmp_path/'output'; out.mkdir()
    assert not worker.owned(out/suffix, out)
    guard = worker.make_guard(out, [], [], [], synthetic=True)
    with pytest.raises(RuntimeError, match='OUTSIDE_OUTPUT_WRITE'):
        guard('open', (str(out/suffix), 'wb', 1))


def test_guard_blocks_undeclared_market_inputs_and_weights(tmp_path):
    out = tmp_path/'output'; out.mkdir()
    guard = worker.make_guard(out, [], [], [], synthetic=False)
    with pytest.raises(RuntimeError, match='UNDECLARED_DATA'):
        guard('open', ('/home/andre2/GX1_DATA/unbound.npy', 'rb', 0))
    with pytest.raises(RuntimeError, match='FORBIDDEN_PAYLOAD'):
        guard('open', (str(out/'weights.pkl'), 'wb', 1))


def _m1():
    minute = worker.MINUTE
    m = {'bar_start_ns': np.arange(15, dtype=np.int64)*minute,
        'contract_index': np.full(15, 2, np.int16), 'assigned': np.ones(15, bool),
        'ofi_valid': np.ones(15, bool)}
    return m, np.array([5, 10, 15], np.int64)*minute


def test_m5_contract_uses_all_five_minutes_without_earlier_ofi_quality_gate():
    m1, decisions = _m1(); m1['ofi_valid'][:4] = False
    actual = worker.current_m5_contract(m1, decisions)
    assert actual.tolist() == [2, 2, 2] and not actual.flags.writeable


@pytest.mark.parametrize('fault', ['first_contract', 'middle_contract', 'unassigned', 'missing_minute'])
def test_m5_contract_invalid_bucket_cannot_be_tautological_last_minute_check(fault):
    m1, decisions = _m1()
    if fault == 'first_contract': m1['contract_index'][5] = 1
    elif fault == 'middle_contract': m1['contract_index'][7] = -1
    elif fault == 'unassigned': m1['assigned'][7] = False
    else: m1 = {key: np.delete(value, 7) for key, value in m1.items()}
    assert worker.current_m5_contract(m1, decisions).tolist() == [2, -1, 2]



@pytest.mark.parametrize('fault', ['none', 'earlier_contract', 'last_ofi_invalid', 'last_incident', 'missing_last'])
def test_authoritative_bucket_and_existing_projection_integrate_without_valid_fallback(fault):
    from gx1.contracts.entry_direction_snapshot_data_v1 import gather_direction_last_closed_m1_ofi
    m1, decisions = _m1()
    m1['available_ns'] = m1['bar_start_ns'] + worker.MINUTE
    m1['ofi_contracts'] = np.arange(15, dtype=np.float64)
    m1['ofi_contracts'][4] = 0.0  # Observed zero must survive as observed.
    m1['receive_clock_excluded'] = np.zeros(15, bool)
    m1['book_event_state_excluded'] = np.zeros(15, bool)
    m1['ofi_valid'][:4] = False  # Earlier OFI quality does not define M5 contract.
    if fault == 'earlier_contract': m1['contract_index'][7] = 1
    elif fault == 'last_ofi_invalid': m1['ofi_valid'][9] = False
    elif fault == 'last_incident': m1['receive_clock_excluded'][9] = True
    elif fault == 'missing_last': m1 = {key: np.delete(value, 9) for key, value in m1.items()}
    parents = np.array([2, 6, 9], np.int64)
    for value in (*m1.values(), decisions, parents): value.setflags(write=False)
    contract = worker.current_m5_contract(m1, decisions)
    before = {key: value.tobytes() for key, value in m1.items()}
    projection = gather_direction_last_closed_m1_ofi(m1, parent_rows=parents,
        decision_times_ns=decisions, current_m5_contract_index=contract)
    second_seen = fault == 'none'
    assert projection[worker.NEW_FIELDS[0]].tolist() == [0., 9. if second_seen else 0., 14.]
    assert projection[worker.NEW_FIELDS[1]].tolist() == [True, second_seen, True]
    assert projection['parent_rows'].tolist() == parents.tolist()
    assert all(not value.flags.writeable for value in projection.values())
    assert {key: value.tobytes() for key, value in m1.items()} == before


def test_new_normalization_uses_only_observed_prefix_and_retains_binary_missing_mask():
    decisions = worker.CUT + (np.arange(worker.N+2, dtype=np.int64)-worker.N)*5*worker.MINUTE
    values = np.zeros(len(decisions), np.float64); seen = np.zeros(len(decisions), bool)
    values[:3] = [0., 3., 9.]; seen[:3] = True
    values[-2:] = [12., 24.]; seen[-2:] = True
    projection = dict(zip(worker.NEW_FIELDS, (values, seen)))
    encoded, stats = worker.normalize_new(projection, decisions)
    changed = {name: value.copy() for name, value in projection.items()}
    changed[worker.NEW_FIELDS[0]][-2:] = [1e20, -1e20]
    other, other_stats = worker.normalize_new(changed, decisions)
    assert stats == other_stats and stats['fit_row_count'] == 3
    assert stats['binary_mask'] == [0, 1]
    assert encoded[:worker.N].tobytes() == other[:worker.N].tobytes()
    assert encoded[0, 1] == 1 and encoded[0, 0] != 0  # Genuine raw zero, centered at3.
    assert encoded[4].tolist() == [0., 0.]  # Missing is hardzero AFTER the transform.
    assert not encoded.flags.writeable


def test_new_normalization_no_observation_cannot_invent_scale():
    decisions = worker.CUT + (np.arange(worker.N+1, dtype=np.int64)-worker.N)*5*worker.MINUTE
    projection = {worker.NEW_FIELDS[0]: np.zeros(len(decisions), np.float64),
        worker.NEW_FIELDS[1]: np.zeros(len(decisions), bool)}
    with pytest.raises(RuntimeError, match='NEW_PREFIX_OBSERVATIONS'):
        worker.normalize_new(projection, decisions)


def test_append_preserves_every_common_f32_bit_including_signed_zero(tmp_path):
    rows = 3; original = np.arange(rows*3345, dtype=np.float32).reshape(rows, 3345)
    original[0, 0] = -0.0
    path = tmp_path/'baseline.npy'; augmented = tmp_path/'augmented.npy'
    path.write_bytes(original.tobytes()); new = np.array([[0, 1], [0, 0], [-3, 1]], np.float32)
    expected = hashlib.sha256(original.tobytes()).hexdigest()
    assert worker.append_and_check(path, augmented, new, rows=rows) == expected
    result = np.memmap(augmented, mode='r', dtype=np.float32, shape=(rows, 3347))
    assert np.ascontiguousarray(result[:, :3345]).tobytes() == original.tobytes()
    assert result[:, -2:].tolist() == new.tolist()
    assert path.read_bytes() == original.tobytes()
    result._mmap.close()


@pytest.mark.parametrize('fault', ['states', 'probability', 'signed_zero'])
def test_baseline_replay_requires_all_original_state_and_probability_bits(fault):
    states = ['a', 'b', 'c']; expected = states.copy()
    p = np.zeros((worker.LATER, 3), np.float64); old = p.copy()
    if fault == 'states': states[2] = 'changed'
    elif fault == 'probability': p[0, 1] = np.nextafter(0., 1.)
    else: p[0, 1] = -0.0
    with pytest.raises(RuntimeError, match='PARITY'):
        worker.require_baseline_parity(states, p, expected, old)


def test_call_gate_rejects_augmented_fit_until_all_baseline_parity_passes():
    calls = worker.Calls(False); calls.phase = 'B'
    # Real code objects are hashable; this fixture declares only the event identity.
    def pretend_fit(): pass
    calls.codes = {pretend_fit.__code__: 'forest_fit'}
    frame = SimpleNamespace(f_code=pretend_fit.__code__)
    with pytest.raises(RuntimeError, match='UNADMITTED_LEARNING_CALL'):
        calls.watch(frame, 'call', None)
    calls.baseline_passed = True; calls.watch(frame, 'call', None)
    assert calls.counts['forest_fit'] == 1


def _probabilities(y):
    p = np.ascontiguousarray(.25 + .5*y, dtype=np.float64)
    return {'A': p.copy(), 'B': p.copy(), 'prior': np.full(len(y), .5), 'R1': p.copy()}


def test_paired_resampling_uses_fixed_seeds_shared_weights_and_zero_difference():
    y = np.tile(np.array([0, 1], np.int8), 8)
    decisions = np.repeat(np.arange(8, dtype=np.int64)*7*86400000000000, 2)
    result = worker.paired_uncertainty(y, _probabilities(y), decisions)
    for scheme, seed in (('UTC_DAY', 20260911), ('UTC_WEEK_MONDAY', 20260912)):
        value = result[scheme]
        assert value['replicates'] == 2000 and value['blocks'] == 8 and value['seed'] == seed
        assert value['weights_sha256'] == worker.digest(worker.draw_weights(8, seed))
        for metric in ('auc', 'logloss', 'brier'):
            ci = value['contrasts']['B_minus_A'][metric]
            assert ci['low'] == ci['high'] == 0 and ci['undefined_replicates'] == 0
    assert not worker.primary_supported(result)


def test_undefined_auc_resamples_are_counted_without_replacement_or_extra_draws():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1], np.int8)
    decisions = np.arange(8, dtype=np.int64)*7*86400000000000
    result = worker.paired_uncertainty(y, _probabilities(y), decisions)
    for value in result.values():
        ci = value['contrasts']['B_minus_A']['auc']
        assert ci['replicates'] == 2000 and ci['undefined_replicates'] > 0
        assert ci['defined_replicates'] + ci['undefined_replicates'] == 2000
    assert not worker.primary_supported(result)


def _supported():
    return {scheme: {'status': 'PAIRED_BLOCK_BOOTSTRAP', 'contrasts': {'B_minus_A': {
        'auc': {'low': .01, 'undefined_replicates': 0},
        'logloss': {'high': -.001, 'undefined_replicates': 0}}}}
        for scheme in ('UTC_DAY', 'UTC_WEEK_MONDAY')}


@pytest.mark.parametrize('fault', ['auc', 'logloss', 'undefined', 'blocks'])
def test_primary_requires_both_metrics_under_both_calendar_schemes(fault):
    result = _supported(); assert worker.primary_supported(result)
    target = result['UTC_WEEK_MONDAY']
    if fault == 'auc': target['contrasts']['B_minus_A']['auc']['low'] = 0
    elif fault == 'logloss': target['contrasts']['B_minus_A']['logloss']['high'] = 0
    elif fault == 'undefined': target['contrasts']['B_minus_A']['auc']['undefined_replicates'] = 1
    else: target['status'] = 'INSUFFICIENT_BLOCKS'
    assert not worker.primary_supported(result)


def test_main_cannot_hash_sealed_input_before_cap_lock_and_guard(monkeypatch, tmp_path):
    from contextlib import contextmanager
    from pathlib import Path
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication

    events = []; installed = []; hashed = []
    out = tmp_path/'guarded-output'
    source_head = 'a'*40; plan_path = Path('/fixture/plan.json')
    sealed_input = '/home/andre2/GX1_RUNS/TEST/forbidden.npy'
    plan = {'mode': 'actual', 'design': worker.DESIGN, 'protocol': worker.PROTOCOL,
        'inputs': {'train_raw': {'path': sealed_input, 'sha256': 'b'*64}},
        'metadata': {}, 'sources': {}, 'wall_seconds': 1200}
    policy = {key: False for key in ('training_enabled', 'native_training_enabled',
        'full_epoch_training_allowed', 'full_val_allowed', 'native_launch_authorized')}
    plan_ref = {'path': str(plan_path), 'sha256': 'c'*64}
    policy[worker.STAGE] = {'status': 'ADMITTED', 'plan': plan_ref,
        'execution_authorized': True, 'max_invocations': 1, 'claimed_invocations': 0,
        'completed_invocations': 0, 'relaunch_allowed': False}

    @contextmanager
    def verification_scope():
        events.append('scope_enter')
        try: yield
        finally: events.append('scope_exit')

    def read_json(path, sha):
        if path == plan_path: return plan
        if path == worker.REPO/'NEXT_RUN_POLICY.json': return policy
        return {}

    def input_binding(ref, *, label):
        assert label == 'input'
        events.append('input_binding')
        assert events.index('cap') < events.index('lock') < events.index('guard') < events.index('input_binding')
        # Simulate the first attempted input open; the installed guard must stop
        # the sealed path before any bytes can be hashed.
        installed[0]('open', (ref['path'], 'rb', 0))
        hashed.append(ref['path'])

    def install_guard(guard):
        events.append('guard'); installed.append(guard)

    monkeypatch.setattr(worker, 'regular', lambda path: Path(path))
    monkeypatch.setattr(worker, 'git_state', lambda: (source_head, 'work/gx1-current', ''))
    monkeypatch.setattr(worker, 'require_plan', lambda *unused: out)
    monkeypatch.setattr(binding, 'fresh_file_verification_scope', verification_scope)
    monkeypatch.setattr(binding, 'read_bound_json', read_json)
    monkeypatch.setattr(binding, 'require_binding', input_binding)
    monkeypatch.setattr(binding, 'capture_verified_file_snapshot', lambda: SimpleNamespace(require_unchanged=lambda: None))
    monkeypatch.setattr(cap, 'require_capped_cpu_audit_execution',
        lambda: events.append('cap') or {'cgroup_path': '/sys/fs/cgroup/fixture'})
    monkeypatch.setattr(cap, 'require_capped_lock_ancestry',
        lambda: events.append('lock') or {'scope': '/fixture'})
    monkeypatch.setattr(publication, '_publish_bytes', lambda *unused: None)
    monkeypatch.setattr(worker.sys, 'addaudithook', install_guard)
    monkeypatch.setattr(worker.sys, 'setprofile', lambda *unused: None)
    monkeypatch.setattr(worker.signal, 'signal', lambda *unused: None)
    monkeypatch.setattr(worker.signal, 'alarm', lambda *unused: None)
    with pytest.raises(RuntimeError, match='TEST_PATH'):
        worker.main(['--plan', str(plan_path), '--plan-sha256', plan_ref['sha256'],
            '--policy-sha256', 'd'*64, '--source-head', source_head])
    assert events[:5] == ['scope_enter', 'scope_exit', 'cap', 'lock', 'guard']
    assert events[5:] == ['scope_enter', 'input_binding', 'scope_exit']
    assert not hashed and len(installed) == 1
