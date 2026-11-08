"""Critical audit defects: close boundaries, warmup, sparse windows and bit masks."""
import numpy as np
import pytest

from gx1.audit import verify_market_state_ta_native_input_path_v1 as audit


def test_closed_window_includes_boundary_and_never_open_bar():
    clock = np.array([0, 10, 20, 30, 40], dtype=np.int64)
    decisions = np.array([30, 39, 40], dtype=np.int64)
    got = audit.closed_windows(clock, decisions, duration_ns=10, length=2, warmup=0)
    assert got.tolist() == [[1, 2], [1, 2], [2, 3]]
    assert np.all(clock[got] + 10 <= decisions[:, None])


def test_sparse_observed_clock_does_not_invent_gap_rows():
    clock = np.array([0, 10, 50, 60], dtype=np.int64)
    got = audit.closed_windows(clock, np.array([60], np.int64),
                              duration_ns=10, length=3, warmup=0)
    assert clock[got].tolist() == [[0, 10, 50]]


def test_last_endpoint_after_warmup_does_not_make_early_history_valid():
    with pytest.raises(RuntimeError, match='HISTORY_WARMUP'):
        audit.closed_windows(np.array([0, 10, 20, 30], np.int64),
            np.array([40], np.int64), duration_ns=10, length=3, warmup=2)


@pytest.mark.parametrize('clock', ([0, 10, 10], [0, 20, 10]))
def test_nonunique_or_reversed_clock_is_rejected(clock):
    with pytest.raises(RuntimeError, match='CLOCK_INPUT'):
        audit.closed_windows(np.array(clock, np.int64), np.array([30], np.int64),
                             duration_ns=10, length=1, warmup=0)


def test_clock_addition_overflow_is_rejected():
    maximum = np.iinfo(np.int64).max
    with pytest.raises(RuntimeError, match='CLOCK_OVERFLOW'):
        audit.closed_windows(np.array([maximum - 1], np.int64),
            np.array([maximum], np.int64), duration_ns=10, length=1, warmup=0)


def test_history_union_retains_fit_score_gap_and_deduplicates_overlap():
    rows, intervals = audit.interval_union(np.array([4, 5, 5, 12], np.int64), 3)
    assert intervals == [[2, 6], [10, 13]]
    assert rows.tolist() == [2, 3, 4, 5, 10, 11, 12]
    assert [part.tolist() for part in audit.contiguous_parts(rows)] == [[2, 3, 4, 5], [10, 11, 12]]


def test_exact_rejects_zero_mask_changes_and_signed_zero():
    value = np.array([[1., 0., -2.]], np.float32)
    audit.exact(value.copy(), value, label='valid')
    for replacement in (-0., 1.):
        changed = value.copy(); changed[0, 1] = replacement
        with pytest.raises(RuntimeError, match='BIT_PARITY'):
            audit.exact(changed, value, label='mask')


def test_exact_never_treats_matching_missing_values_as_valid_model_input():
    values = np.array([[np.nan]], np.float32)
    with pytest.raises(RuntimeError, match='NONFINITE'):
        audit.exact(values, values.copy(), label='missing')
    with pytest.raises(RuntimeError, match='SHAPE_DTYPE'):
        audit.exact(np.ones((1, 1), np.float64), np.ones((1, 1), np.float32), label='precision')


def test_context_bridge_accepts_separate_rsi_rounding_and_checks_utc_boundaries():
    from types import SimpleNamespace
    from gx1.features import htf_features as htf
    from gx1.features.entry_specialist_feature_groups_v1 import compile_full_ta_entry_execution_profile
    profile = compile_full_ta_entry_execution_profile()
    names = list(profile['physical_field_names']['signal'])
    times = np.array(['2024-11-03T06:55:00', '2024-11-03T23:55:00'], dtype='datetime64[ns]').astype(np.int64)
    local = np.zeros((2, len(names)), np.float32)
    cache = {}
    # Independently rounded raw and centered wires can differ after inversion,
    # most visibly near RSI zero. A generic bit-equality rule would be wrong.
    original_rsi = np.array([1.e-6, 99.999997], np.float64)
    raw = original_rsi.astype(np.float32)
    centered = ((original_rsi - 50.) / 50.).astype(np.float32)
    for tf in audit.TIMEFRAMES:
        columns = list(htf.MULTI_TF_PER_BAR_FEATURES_V4)
        values = np.zeros((2, len(columns)), np.float32)
        values[:, columns.index('rsi14_centered')] = centered
        clock = times + 300_000_000_000 - int(htf.MULTI_TF_SHIFT[tf].value)
        cache[tf] = SimpleNamespace(columns=columns, attrs={'ts_int64': clock,
            'feats_np': values, 'causal_warmup_rows': 0})
        local[:, names.index('ctx_cont.' + tf.lower() + '_rsi14_canon_v2')] = raw
    hour = np.array([7., 0.], np.float32)
    weekday = np.array([6., 0.], np.float32)
    for field, value in (
        ('hour_sin', np.sin(2 * np.pi * hour / 24)),
        ('hour_cos', np.cos(2 * np.pi * hour / 24)),
        ('dow_sin', np.sin(2 * np.pi * weekday / 7)),
        ('minutes_since_session_open', np.array([0., 120.], np.float32)),
    ):
        local[:, names.index('ctx_cont.' + field)] = value
    rows, report = audit._local_context_formula_bridges(local, times, cache, profile)
    assert report['fields'] == 37 and report['checked_values'] == 74
    assert all(values.tolist() == [0, 1] for values in rows.values())
    assert report['per_field']['m5_rsi14_canon_v2']['maximum_absolute_error'] > 0.
    assert all(item['available_rows'] == 2 for item in report['per_field'].values())
    changed = local.copy()
    changed[0, names.index('ctx_cont.m5_rsi14_canon_v2')] = 1.e-4
    with pytest.raises(RuntimeError, match='LOCAL_CONTEXT_RSI_ROUNDING'):
        audit._local_context_formula_bridges(changed, times, cache, profile)
    changed = local.copy()
    changed[1, names.index('ctx_cont.minutes_since_session_open')] = 121.
    with pytest.raises(RuntimeError, match='BIT_PARITY:LOCAL_CONTEXT:minutes_since_session_open'):
        audit._local_context_formula_bridges(changed, times, cache, profile)


def _synthetic_publication_run(monkeypatch, tmp_path, *, fail_audit):
    import hashlib
    import json
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
    from gx1.contracts import gx1_capped_execution_v1 as caps
    from types import SimpleNamespace
    from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
    from gx1.contracts import market_state_future_displacement_candidate_v1 as future
    from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner

    def ref(path):
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    out = tmp_path / 'run'; out.mkdir()
    repo = tmp_path / 'repo'; repo.mkdir()
    source = repo / 'owner.py'; source.write_text('# synthetic owner\n')
    payload = tmp_path / 'input.bin'; payload.write_bytes(b'synthetic input only\n')
    payload.chmod(0o444)
    def payload_identity():
        value = payload.stat()
        return tuple(getattr(value, name) for name in ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))
    original_payload_identity = payload_identity()
    plan_path = tmp_path / 'PLAN.json'; plan_path.write_text('{"synthetic":true}\n')
    policy_path = tmp_path / 'POLICY.json'; policy_path.write_text('{"synthetic":true}\n')
    candidate = out / 'CANDIDATE_VIEW.json'; candidate.write_text('{"synthetic":true}\n')
    plan_ref, policy_ref = ref(plan_path), ref(policy_path)
    plan = {'inputs': {'candidate_view': ref(candidate)},
            'source_owners': {'owner.py': ref(source)['sha256']}, 'max_json_bytes': 1024 * 1024}
    head = 'a' * 40
    provider_plan = {'payloads': {str(payload): {'stat': original_payload_identity}}}
    receipts = {'provider_chain': {'provider_plan': provider_plan}, 'historical_lineage': {'synthetic': True}}
    view, prepared = {'contract_sha256': 'b' * 64}, {'synthetic': True}
    state_bound, future_bound = object(), object()
    rows = np.arange(audit.fold.FIT_ROWS, dtype=np.int64)
    issues = rows * 300_000_000_000 + 1714521900000000000
    calls = {'metadata': 0, 'prepared': 0, 'audit': 0, 'target_loads': 0, 'prior_fit': 0, 'release': 0}
    measurement = {'synthetic_input_audit': True}
    priors = {'synthetic_endpoint_priors': True, 'fit_rows': len(rows),
              'fit_parent_rows_sha256': audit.axis_sha(rows),
              'fit_decision_ns_sha256': audit.axis_sha(issues)}
    def validate_metadata_stub(got_plan):
        assert got_plan is provider_plan
        calls['metadata'] += 1
        if payload_identity() != original_payload_identity:
            raise RuntimeError('synthetic provider payload identity changed')
        assert ref(source)['sha256'] == plan['source_owners']['owner.py']
        return {'metadata': {str(source): {'ref': ref(source)}},
                'reuse': {'scratch_files': {str(payload): original_payload_identity}}}
    provider = SimpleNamespace(validate_metadata=validate_metadata_stub)
    def inspect_stub(got_ref, *, require_policy):
        assert got_ref == plan_ref and type(require_policy) is bool
        return plan, policy_ref, head, receipts, view
    def prepared_stub(got_receipts, got_view, *, np):
        assert got_receipts is receipts and got_view is view
        calls['prepared'] += 1
        # Match the real provider: own validation, no campaign registration.
        provider.validate_metadata(provider_plan)
        return provider, provider_plan, prepared, rows, issues
    def audit_stub(**kwargs):
        assert kwargs == {'plan': plan, 'receipts': receipts, 'view': view, 'prepared': prepared}
        calls['audit'] += 1
        if fail_audit:
            raise RuntimeError('injected input audit failure')
        return measurement
    def state_stub():
        calls['target_loads'] += 1
        return state_bound
    def future_stub():
        calls['target_loads'] += 1
        return future_bound
    def prior_stub(got_state, got_future, *, axes, fold_id, ta_only):
        assert got_state is state_bound and got_future is future_bound
        assert set(axes) == {'fit_parent_rows', 'fit_decision_ns'}
        np.testing.assert_array_equal(axes['fit_parent_rows'], rows)
        np.testing.assert_array_equal(axes['fit_decision_ns'], issues)
        assert fold_id == 1 and ta_only is True
        calls['prior_fit'] += 1
        return priors
    def release_stub(got_provider, got_prepared):
        assert got_provider is provider and got_prepared is prepared
        calls['release'] += 1
    monkeypatch.setattr(audit, 'OUT_ROOT', out)
    monkeypatch.setattr(audit, 'REPO', repo)
    monkeypatch.setattr(audit, 'PLAN_PATH', plan_path)
    monkeypatch.setattr(audit, 'POLICY_PATH', policy_path)
    monkeypatch.setattr(audit, 'inspect', inspect_stub)
    # This is the source-head part of the mocked inspection boundary; all
    # publication, payload stat bookends and plan/policy/source/view hashes stay real.
    monkeypatch.setattr(zero, '_clean_head', lambda: head)
    monkeypatch.setattr(caps, 'require_capped_cpu_research_producer_execution', lambda: dict(audit.LIMITS))
    monkeypatch.setattr(runner, '_prepared_fold1', prepared_stub)
    monkeypatch.setattr(audit, 'audit_inputs', audit_stub)
    monkeypatch.setattr(state, 'load_bound_five_endpoint_targets', state_stub)
    monkeypatch.setattr(future, 'load_bound_future_displacement', future_stub)
    monkeypatch.setattr(objective, 'fit_endpoint_priors', prior_stub)
    monkeypatch.setattr(mature, '_release_pages', release_stub)
    monkeypatch.setattr(mature, '_provider_module', lambda: provider)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    def read_ref(item):
        from pathlib import Path
        path = Path(item['path'])
        assert ref(path) == item
        return json.loads(path.read_text())
    return {'out': out, 'payload': payload, 'plan': plan, 'plan_ref': plan_ref, 'policy_ref': policy_ref,
            'calls': calls, 'measurement': measurement, 'priors': priors,
            'ref': ref, 'read_ref': read_ref}


def _assert_consumed_publications_cannot_be_replaced(context):
    from gx1.contracts.immutable_event_authority_v1 import ImmutableEventAuthorityError
    before = {p.name: p.read_bytes() for p in context['out'].iterdir()}
    calls = dict(context['calls'])
    # Inspection is mocked; the real atomic writer independently rejects a
    # second claim before provider/model/target work can begin.
    with pytest.raises(ImmutableEventAuthorityError, match='immutable event already exists'):
        audit.run(context['plan_ref'])
    assert {p.name: p.read_bytes() for p in context['out'].iterdir()} == before
    assert context['calls'] == calls


def test_run_success_publishes_hash_bound_priors_result_and_terminal(monkeypatch, tmp_path):
    import signal
    context = _synthetic_publication_run(monkeypatch, tmp_path, fail_audit=False)
    alarm = signal.getsignal(signal.SIGALRM)
    published = audit.run(context['plan_ref'])
    assert {p.name for p in context['out'].iterdir()} == {
        'CANDIDATE_VIEW.json', 'CLAIM.json', 'ENDPOINT_PRIORS.json', 'RESULT.json', 'TERMINAL.json'}
    result = context['read_ref'](published['result'])
    terminal = context['read_ref'](published['terminal'])
    claim = context['read_ref'](result['claim'])
    assert context['read_ref'](published['endpoint_priors']) == context['priors']
    assert result['status'] == terminal['status'] == 'COMPLETE'
    assert terminal['exit_code'] == 0 and terminal['no_relaunch_under_this_plan'] is True
    assert result['plan'] == terminal['plan'] == claim['plan'] == context['plan_ref']
    assert result['policy'] == claim['policy'] == context['policy_ref']
    assert result['candidate_view'] == claim['candidate_view'] == context['plan']['inputs']['candidate_view']
    assert terminal['result'] == published['result'] and terminal['claim'] == result['claim']
    assert result['endpoint_priors'] == terminal['endpoint_priors'] == published['endpoint_priors']
    assert result['input_audit'] == context['measurement']
    integrity = result['input_integrity']
    assert integrity['provider_metadata_bookend']['payload_stat_files'] == 1
    assert integrity['provider_metadata_bookend']['metadata_sha_files'] == 1
    assert integrity['provider_metadata_validated_before_prepare_and_after_measurement'] is True
    assert integrity['reused_local_decode_fresh_body_hash_claimed'] is False
    assert 'verified_payload_files' not in result
    assert result['source_unchanged'] is True and result['test_data_used'] is False
    assert context['calls'] == {'metadata': 2, 'prepared': 1, 'audit': 1, 'target_loads': 2, 'prior_fit': 1, 'release': 1}
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.) and signal.getsignal(signal.SIGALRM) == alarm
    _assert_consumed_publications_cannot_be_replaced(context)


def test_run_audit_failure_publishes_failed_consumed_without_priors_or_result(monkeypatch, tmp_path):
    import signal
    context = _synthetic_publication_run(monkeypatch, tmp_path, fail_audit=True)
    alarm = signal.getsignal(signal.SIGALRM)
    with pytest.raises(RuntimeError, match='injected input audit failure'):
        audit.run(context['plan_ref'])
    assert {p.name for p in context['out'].iterdir()} == {
        'CANDIDATE_VIEW.json', 'CLAIM.json', 'FAILED.json', 'TERMINAL.json'}
    terminal = context['read_ref'](context['ref'](context['out'] / 'TERMINAL.json'))
    failure = context['read_ref'](terminal['failure'])
    claim = context['read_ref'](terminal['claim'])
    assert terminal['status'] == 'FAILED_CONSUMED' and terminal['exit_code'] == 1
    assert terminal['no_relaunch_under_this_plan'] is True and terminal['test_data_used'] is False
    assert terminal['plan'] == failure['plan'] == claim['plan'] == context['plan_ref']
    assert failure['claim'] == terminal['claim']
    assert failure['error_type'] == 'RuntimeError' and failure['error'] == 'injected input audit failure'
    assert context['calls'] == {'metadata': 1, 'prepared': 1, 'audit': 1, 'target_loads': 0, 'prior_fit': 0, 'release': 1}
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.) and signal.getsignal(signal.SIGALRM) == alarm
    _assert_consumed_publications_cannot_be_replaced(context)


def test_preflight_validates_provider_metadata_without_campaign_registration_or_payload_setup(monkeypatch, tmp_path):
    context = _synthetic_publication_run(monkeypatch, tmp_path, fail_audit=False)
    result = audit.preflight(context['plan_ref'])
    assert result['status'] == 'READY_NO_CLAIM_NO_FEATURE_PAYLOAD_NO_MODEL'
    assert result['provider_integrity_metadata']['payload_stat_files'] == 1
    assert result['provider_integrity_metadata']['metadata_or_stat_checks_are_not_fresh_payload_hashes'] is True
    assert result['feature_payloads_read'] is False and result['model_constructions'] == 0
    assert context['calls'] == {'metadata': 1, 'prepared': 0, 'audit': 0, 'target_loads': 0, 'prior_fit': 0, 'release': 0}
    assert {p.name for p in context['out'].iterdir()} == {'CANDIDATE_VIEW.json'}


def test_run_provider_bookend_rejects_changed_payload_before_success_publication(monkeypatch, tmp_path):
    context = _synthetic_publication_run(monkeypatch, tmp_path, fail_audit=False)
    initial_audit = audit.audit_inputs
    def mutate_after_measurement(**kwargs):
        measurement = initial_audit(**kwargs)
        context['payload'].chmod(0o644)
        context['payload'].write_bytes(b'changed synthetic input')
        return measurement
    monkeypatch.setattr(audit, 'audit_inputs', mutate_after_measurement)
    with pytest.raises(RuntimeError, match='synthetic provider payload identity changed'):
        audit.run(context['plan_ref'])
    assert {p.name for p in context['out'].iterdir()} == {
        'CANDIDATE_VIEW.json', 'CLAIM.json', 'FAILED.json', 'TERMINAL.json'}
    terminal = context['read_ref'](context['ref'](context['out'] / 'TERMINAL.json'))
    failure = context['read_ref'](terminal['failure'])
    assert terminal['status'] == 'FAILED_CONSUMED' and terminal['exit_code'] == 1
    assert failure['error'] == 'synthetic provider payload identity changed'
    assert context['calls'] == {'metadata': 2, 'prepared': 1, 'audit': 1, 'target_loads': 2, 'prior_fit': 1, 'release': 1}
