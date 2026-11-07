"""Small contract tests; no raw M1, model fit, or sealed TEST reads."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from gx1.contracts import market_state_time_query_binding_v1 as binding
from gx1.contracts.market_state_time_query_target_v1 import (
    NATIVE_K, TICK_NS, TimeQueryTargetTape,
)


def _fixture() -> binding.BoundTimeQueryTargets:
    boundary = binding.old.FOLD_BOUNDARIES_NS[0]
    next_boundary = binding.old.FOLD_BOUNDARIES_NS[1]
    issues = np.array([boundary - 289 * TICK_NS, boundary - 288 * TICK_NS,
                       boundary + TICK_NS, next_boundary - TICK_NS], dtype=np.int64)
    parents = np.array([11, 22, 33, 44], dtype=np.int64)
    validity = np.array(sorted({int(issue + k * TICK_NS)
                                for issue in issues for k in NATIVE_K}), dtype=np.int64)
    source = np.arange(100, 100 + len(validity), dtype=np.int64)
    bits = np.full((len(validity), 5), 5, dtype=np.uint8)
    score_first = np.searchsorted(validity, issues[2] + TICK_NS)
    bits[score_first, 2] = 7
    tape = TimeQueryTargetTape(validity, source, bits)
    native = np.broadcast_to(np.asarray(NATIVE_K, dtype=np.int64), (len(issues), 5)).copy()
    old = tape.gather(issues, native, fold_upper_ns=binding.old.FOLD_BOUNDARIES_NS[-1])
    return binding.BoundTimeQueryTargets(
        parents, issues, tape, issues, old['allowed'], old['m1_source_row'],
        train_end_ns=binding.old.FOLD_BOUNDARIES_NS[-1])


def test_exact_parent_join_fit_embargo_and_score_maturity():
    bound = _fixture()
    boundary = binding.old.FOLD_BOUNDARIES_NS[0]
    next_boundary = binding.old.FOLD_BOUNDARIES_NS[1]
    early = np.array([boundary - 289 * TICK_NS], dtype=np.int64)
    near = np.array([boundary - 288 * TICK_NS], dtype=np.int64)
    after = np.array([boundary + TICK_NS], dtype=np.int64)
    late = np.array([next_boundary - TICK_NS], dtype=np.int64)
    ones = np.ones((1, 5), dtype=np.int64)

    fit = bound.gather_fit_sampled(np.array([11], dtype=np.int64), early,
                                   fold_id=1, pass_index=3, seed=19)
    assert np.array_equal(fit['query_k'],
                          binding.sampled_query_k(early, pass_index=3, seed=19))
    assert fit['mature'].all()
    assert np.all(fit['validity_ns'] < boundary)
    with pytest.raises(ValueError, match='ISSUE_PARTITION'):
        bound.gather_fit_sampled(np.array([22], dtype=np.int64), near,
                                 fold_id=1, pass_index=3, seed=19)
    with pytest.raises(ValueError, match='EXACT_PARENT_ISSUE'):
        bound.gather_fit_sampled(np.array([11], dtype=np.int64), after,
                                 fold_id=1, pass_index=3, seed=19)
    wrong_k = fit['query_k'].copy()
    wrong_k[0, 2] = 1 + wrong_k[0, 2] % NATIVE_K[2]
    with pytest.raises(ValueError, match='FIT_SCHEDULE'):
        bound.gather(np.array([11], dtype=np.int64), early, wrong_k,
                     fold_id=1, partition='fit', pass_index=3, seed=19)

    score = bound.gather(np.array([33], dtype=np.int64), after, ones,
                         fold_id=1, partition='score')
    assert score['mature'].all()
    assert np.array_equal(score['allowed'], np.array([[5, 5, 7, 5, 5]], dtype=np.uint8))
    assert np.array_equal(score['eligible'], np.array([[1, 1, 0, 1, 1]], dtype=bool))
    assert np.all(score['m1_source_row'] >= 0)
    late_score = bound.gather(np.array([44], dtype=np.int64), late, ones,
                              fold_id=1, partition='score')
    assert np.array_equal(late_score['validity_ns'], np.full((1, 5), next_boundary))
    assert not late_score['mature'].any()
    assert not late_score['eligible'].any()
    assert np.array_equal(late_score['allowed'], np.full((1, 5), 7))
    assert np.array_equal(late_score['m1_source_row'], np.full((1, 5), -1))
    with pytest.raises(ValueError, match='ISSUE_PARTITION'):
        bound.gather(np.array([22], dtype=np.int64), near, ones,
                     fold_id=1, partition='score')
    with pytest.raises(ValueError, match='ISSUE_PARTITION'):
        bound.gather_fit_sampled(np.array([33], dtype=np.int64), after,
                                 fold_id=1, pass_index=3, seed=19)
    with pytest.raises(ValueError, match='FOLD'):
        bound.gather(np.array([33], dtype=np.int64), after, ones,
                     fold_id=6, partition='score')


def test_endpoint_parity_rejects_any_bit_or_source_change():
    bound = _fixture()
    issues = bound.decision_ns.copy()
    native = np.broadcast_to(np.asarray(NATIVE_K, dtype=np.int64), (len(issues), 5)).copy()
    old = bound.tape.gather(issues, native,
                            fold_upper_ns=binding.old.FOLD_BOUNDARIES_NS[-1])
    assert bound.endpoint_parity['cells'] == 20
    changed_bits = old['allowed'].copy()
    changed_bits[0, 0] = 1 if changed_bits[0, 0] != 1 else 2
    with pytest.raises(ValueError, match='ENDPOINT_PARITY'):
        binding._endpoint_parity(bound.tape, issues, changed_bits,
                                 old['m1_source_row'], binding.old.FOLD_BOUNDARIES_NS[-1])
    changed_source = old['m1_source_row'].copy()
    changed_source[1, 0] += 1
    with pytest.raises(ValueError, match='ENDPOINT_PARITY'):
        binding._endpoint_parity(bound.tape, issues, old['allowed'],
                                 changed_source, binding.old.FOLD_BOUNDARIES_NS[-1])


def test_schedule_is_label_blind_order_invariant_and_native_bounded():
    base = binding.old.TRAIN_START_NS
    issues = base + np.arange(32, dtype=np.int64) * TICK_NS
    first = binding.sampled_query_k(issues, pass_index=3, seed=19)
    assert first.shape == (32, 5)
    assert (first[:, 0] == 1).all()
    assert np.all(first >= 1)
    assert np.all(first <= np.asarray(NATIVE_K)[None, :])
    assert np.array_equal(first[::-1],
                          binding.sampled_query_k(issues[::-1], pass_index=3, seed=19))
    assert np.array_equal(first[7:13],
                          binding.sampled_query_k(issues[7:13], pass_index=3, seed=19))
    assert np.array_equal(first,
                          binding.sampled_query_k(issues, pass_index=3, seed=19))
    assert not np.array_equal(first,
                              binding.sampled_query_k(issues, pass_index=4, seed=19))
    assert not np.array_equal(first,
                              binding.sampled_query_k(issues, pass_index=3, seed=20))
    with pytest.raises(ValueError, match='SCHEDULE_INPUT'):
        binding.sampled_query_k(issues + 1, pass_index=3, seed=19)


def test_failed_terminal_and_hash_are_rejected(tmp_path, monkeypatch):
    original_terminal_ref = binding.CANDIDATE_REFS['terminal']
    terminal = json.loads(original_terminal_ref[0].read_text())
    terminal['status'] = 'FAILED'
    failed = tmp_path / 'TERMINAL.json'
    failed.write_text(json.dumps(terminal))
    sha = hashlib.sha256(failed.read_bytes()).hexdigest()
    refs = dict(binding.CANDIDATE_REFS)
    refs['terminal'] = (failed, sha)
    monkeypatch.setattr(binding, 'CANDIDATE_REFS', refs)
    with pytest.raises(ValueError, match='RECEIPT_CHAIN'):
        binding._candidate_receipts()

    refs['terminal'] = (failed, '0' * 64)
    monkeypatch.setattr(binding, 'CANDIDATE_REFS', refs)
    with pytest.raises(ValueError, match='TERMINAL_HASH'):
        binding._candidate_receipts()

    refs['terminal'] = original_terminal_ref
    bad_tape = tmp_path / 'TRAIN_CANDIDATE_TAPE.npz'
    bad_tape.write_bytes(b'not the published tape')
    refs['tape'] = (bad_tape, binding.CANDIDATE_REFS['tape'][1])
    monkeypatch.setattr(binding, 'CANDIDATE_REFS', refs)
    with pytest.raises(ValueError, match='TAPE_HASH'):
        binding._checked_bytes('tape', 32 * 1024**2)
