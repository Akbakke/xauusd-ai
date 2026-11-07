"""Synthetic parity and censor checks for one-use observed-M1 state audit."""
import hashlib
import json
import subprocess
import time

import numpy as np
import pytest

from gx1.audit import verify_market_state_m1_duration_v1 as audit
from gx1.audit import verify_mtf_state_side_locality_v1 as locality
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target


def test_jit_cold_replay_matches_existing_python_formula_all_windows():
    rng = np.random.default_rng(404)
    n = 1800
    close = np.cumsum(rng.normal(0, 0.16, n)) + 2000
    open_ = np.r_[close[0], close[:-1]].copy()
    high = np.maximum(open_, close) + rng.random(n) * 0.08
    low = np.minimum(open_, close) - rng.random(n) * 0.08
    close[::7] = high[::7]
    close[3::11] = low[3::11]
    threshold = np.full(n, 0.45, np.float64)
    threshold[499:501] = 0
    unknown_gap = np.zeros(n, dtype=np.bool_)
    unknown_gap[[0, 300, 1200]] = True
    for window in (5, 15, 60, 240, 1440):
        rows = np.unique(np.r_[window - 1, window, n - 1,
                               np.linspace(window - 1, n - 1, 9,
                                           dtype=np.int64)])
        actual = audit._cold_replay_many(
            rows, window, open_, high, low, close, threshold, unknown_gap)
        expected = [locality.cold_side(
            int(row), window, open_, high, low, close, threshold,
            unknown_gap, target)[0] for row in rows]
        assert actual.tolist() == expected


def test_gray_promotion_and_local_side_demote_without_redefining_formula(monkeypatch):
    n, tfs = 8, 5
    oh = np.full(n, 10., np.float64)
    fake_m1 = object()
    monkeypatch.setattr(audit.target, 'LOCAL_WINDOW_M1_BARS',
                        {tf: 1 for tf in audit.TIMEFRAMES})
    monkeypatch.setattr(audit.target, '_ohlc', lambda _:
                        {'open': oh, 'high': oh, 'low': oh, 'close': oh})
    monkeypatch.setattr(audit, 'early_active_proof', lambda *args:
                        (np.zeros(n, bool), np.zeros(n, bool)))
    base_bits = np.array([7, 7, 2, 1, 4, 5, 7, 7], dtype=np.uint8)
    monkeypatch.setattr(audit.classify, 'classify_rows', lambda *args:
                        (base_bits.copy(), {}))
    def cold(rows, window, *args):
        assert rows.tolist() == [3, 4]
        return np.array([target.DOWN, target.DOWN], np.int8)
    monkeypatch.setattr(audit, '_cold_in_chunks', cold)
    a05 = {
        'validity_ns': np.arange(n, dtype=np.int64) * audit.MINUTE_NS,
        'threshold': np.ones((n, tfs), np.float64),
        'path_observed': np.ones((n, tfs), bool),
    }
    a10 = {
        'threshold': np.full((n, tfs), 2., np.float64),
        'local_range': np.zeros((n, tfs), np.float64),
        'reason': np.zeros((n, tfs), np.uint16),
        'path_observed': np.ones((n, tfs), bool),
    }
    a10['local_range'][1, :] = 1.5
    bits, path, report = audit._classify_all_m1(
        fake_m1, a05, a10, np.zeros(n, np.uint8), time.monotonic() + 90)
    assert np.array_equal(bits[:, 0], np.array([7, 5, 2, 1, 5, 5, 7, 7], np.uint8))
    assert np.array_equal(path[:, 0], np.array([0, 0, 1, 1, 0, 0, 0, 0], bool))
    assert np.all(bits == bits[:, :1]) and np.all(path == path[:, :1])
    assert report['M5'] == {
        'candidate_certified_side_m1_rows': 2, 'local_same': 1,
        'local_unknown': 0, 'local_opposite': 1,
        'gray_promoted_active_m1_rows': 1,
    }


def test_exact_minute_runs_censor_separate_closure_gap_and_unknown():
    minutes = np.array([0, 1, 2, 3, 4, 10, 11, 20, 21, 22, 23, 24])
    clock = minutes.astype(np.int64) * audit.MINUTE_NS
    bits = np.array([1, 1, 2, 4, 4, 1, 1, 1, 5, 1, 1, 2], np.uint8)
    path = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1], bool)
    gaps = np.zeros(len(bits), np.uint8)
    gaps[5], gaps[7] = 1, 2
    atr = np.zeros(len(bits), np.int64)
    atr[3:] = clock[3]
    result = audit.summarize_m1_states(bits, path, clock, gaps, atr, 5)
    assert result['m1_rows'] == 12
    assert result['label_counts'] == {
        'DOWN': 7, 'FLAT': 2, 'UP': 2, 'ACTIVE': 1, 'ALL': 0}
    assert result['path_confirmed_singleton_rows'] == 8
    runs = result['label_only_runs']
    assert runs['by_state']['DOWN']['runs'] == 4
    assert runs['by_state']['DOWN']['left_boundary']['documented_closure'] == 1
    assert runs['by_state']['DOWN']['right_boundary']['unknown_source_gap'] == 1
    assert runs['by_state']['DOWN']['right_boundary']['unknown_label'] == 1
    assert runs['by_state']['FLAT']['fully_known_bounded_runs'] == 1
    assert runs['by_state']['FLAT']['fully_known_bounded_below_own_lead'] == 1
    assert runs['all_singleton']['certified_m1_rows'] == 11
    assert result['path_confirmed_runs']['all_singleton']['certified_m1_rows'] == 8
    assert result['atr_update_coincident_state_changes'] == {
        'adjacent_comparable_pairs': 7, 'state_changes': 3,
        'changes_with_new_closed_TF_ATR': 1,
        'changes_without_new_closed_TF_ATR': 2,
        'atr_updates_in_comparable_pairs': 1,
    }
    assert result['continuous_between_M1_bars_or_intraminute_order_proven'] is False


def test_every_selected_endpoint_bit_and_source_row_required(monkeypatch):
    n = 118706
    issues = np.arange(n, dtype=np.int64) * 5 * audit.MINUTE_NS
    axis = np.arange((n - 1) * 5 + 1441, dtype=np.int64) * audit.MINUTE_NS
    source_rows = np.arange(len(axis), dtype=np.int64) + 100
    m5_rows = np.arange(n, dtype=np.int64) + 200
    bits = np.full((len(axis), 5), 2, np.uint8)
    bits[240, 3] = 7
    selected_bits = np.full((n, 5), 2, np.uint8)
    selected_bits[0, 3] = 7
    selected_sources = np.stack([
        source_rows[np.arange(n) * 5 + lead]
        for lead in audit.LEADS], axis=1)
    selected = {
        'issue_ns': issues, 'm5_source_row': m5_rows,
        'lead_minutes': np.asarray(audit.LEADS, np.int16),
        'm1_source_row': selected_sources, 'allowed_set': selected_bits,
    }
    monkeypatch.setattr(audit.base, 'pd_ns', lambda _: int(axis[-1] + 10 * audit.MINUTE_NS))
    good = audit._selected_endpoint_parity(
        selected, issues, m5_rows, axis, source_rows, bits)
    assert all(item['issues'] == n and item['bit_mismatches'] == 0
               and item['m1_source_row_mismatches'] == 0
               for item in good.values())
    selected['allowed_set'][0, 3] = 5
    with pytest.raises(RuntimeError, match='SELECTED_ENDPOINT_PARITY:H4:1:0'):
        audit._selected_endpoint_parity(
            selected, issues, m5_rows, axis, source_rows, bits)


def test_preflight_uses_small_refs_and_no_selected_payload(monkeypatch, tmp_path):
    plan_path = tmp_path / 'PLAN.json'
    out_dir = tmp_path / 'RUN'
    monkeypatch.setattr(audit, 'PLAN_PATH', plan_path)
    monkeypatch.setattr(audit, 'OUT_ROOT', out_dir)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    monkeypatch.setattr(audit, '_clean_head', lambda: head)
    calls = []
    monkeypatch.setattr(audit.endpoint, '_small_reference_chain',
                        lambda: calls.append('small'))
    monkeypatch.setattr(audit.binding, '_checked_bytes', lambda *args:
                        pytest.fail('preflight read selected payload'))
    owners = {name: hashlib.sha256((audit.REPO / name).read_bytes()).hexdigest()
              for name in audit.SOURCE_PATHS}
    plan = {
        'schema_version': audit.SCHEMA, 'stage': audit.STAGE,
        'status': 'PREPARED_NOT_ADMITTED', 'output_root': str(out_dir),
        'max_invocations': 1, 'input_refs': audit.INPUT_REFS,
        'source_owners': owners, 'source_base_head': head,
        'execution': audit.EXECUTION, 'limits': audit.LIMITS,
        'metric_contract': audit.METRIC_CONTRACT,
        'numba_version': audit.NUMBA_VERSION,
        'expected_m1_rows': 591682, 'expected_issue_rows': 118706,
        'command_template': audit.planned_command('{plan_sha256}'),
        'read_only': True, 'test_data_used': False,
    }
    plan_path.write_text(json.dumps(plan))
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    observed, ref, got_head, policy = audit.preflight(
        plan_path, digest, out_dir)
    assert observed == plan and ref['sha256'] == digest and got_head == head
    assert policy is None and calls == ['small'] and not out_dir.exists()


def test_postclaim_failure_writes_terminal_without_duration(monkeypatch, tmp_path):
    out_dir = tmp_path / 'RUN'
    plan_ref = {'path': 'plan', 'sha256': 'a' * 64}
    policy_ref = {'path': 'policy', 'sha256': 'b' * 64}
    monkeypatch.setattr(audit, 'preflight', lambda *args, **kwargs:
                        ({'source_owners': {}}, plan_ref, 'head', policy_ref))
    monkeypatch.setattr(audit.endpoint, '_checked_small_json', lambda *args: {})
    monkeypatch.setattr(audit, '_temporary_tapes', lambda *args:
                        (_ for _ in ()).throw(RuntimeError('synthetic blocker')))
    with pytest.raises(RuntimeError, match='synthetic blocker'):
        audit.run(audit.PLAN_PATH, 'a' * 64, out_dir)
    assert (out_dir / 'CLAIM.json').exists()
    assert not (out_dir / 'RESULT.json').exists()
    failed = json.loads((out_dir / 'FAILED.json').read_text())
    terminal = json.loads((out_dir / 'TERMINAL.json').read_text())
    assert failed['error'] == 'synthetic blocker'
    assert failed['no_relaunch_under_this_plan'] is True
    assert terminal['status'] == 'FAILED' and terminal['exit_code'] == 1
    assert terminal['no_relaunch_under_this_plan'] is True
