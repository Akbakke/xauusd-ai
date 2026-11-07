"""Bind future M5-tick state targets to the pinned TRAIN model axis.

The published M1 tape is a research candidate. This owner checks its COMPLETE
receipt and every old native endpoint before exposing targets for loss/scoring.
Query schedules depend only on issue time, TF, pass and a fixed seed.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import struct

import numpy as np

from gx1.contracts import market_state_five_endpoint_target_binding_v1 as old
from gx1.contracts.market_state_time_query_target_v1 import (
    NATIVE_K, TICK_NS, TIMEFRAMES, TimeQueryTargetTape,
)

ROOT = Path('/home/andre2/src/GX1_CURRENT')
RUNS = Path('/home/andre2/GX1_RUNS')
TAPE_ROOT = RUNS / 'MARKET_STATE_TIME_QUERY_TAPE_20261107_001'
CANDIDATE_REFS = {
    'plan': (ROOT / 'configs/research/MARKET_STATE_TIME_QUERY_TAPE_20261107_001.json',
             '3dcf7c67a03a9fbba3f8074e17210bbcab3b28f8a8193f8ca4cba29d192423f0'),
    'claim': (TAPE_ROOT / 'CLAIM.json',
              '2ddc50df15e1208110050967efe2d93a4ad6e845b085cd51360d471cc983c933'),
    'result': (TAPE_ROOT / 'RESULT.json',
               'd36f84f645a66d208e53583255f6d0dbb688ef154f4994b80cf9a0c14c683a86'),
    'terminal': (TAPE_ROOT / 'TERMINAL.json',
                 '2635f51cdb5c592913cd583c1b5cf73b7bd6e64e8f3089ee3d31026e978e3cff'),
    'tape': (TAPE_ROOT / 'TRAIN_CANDIDATE_TAPE.npz',
             '86aff221bf63532b67efe7c5770daa986bdebe916f149dd99961dacde1043b8b'),
}
TAPE_KEYS = frozenset(('validity_ns', 'm1_source_row', 'allowed_set',
                       'path_observed', 'gap_kind'))
_K = np.asarray(NATIVE_K, dtype=np.int64)


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('MARKET_STATE_TIME_QUERY_BINDING_' + reason)


def _ref(name: str) -> dict:
    path, sha = CANDIDATE_REFS[name]
    return {'path': str(path), 'sha256': sha}


def _checked_bytes(name: str, limit: int) -> bytes:
    path, sha = CANDIDATE_REFS[name]
    size = path.stat().st_size
    _need(0 < size <= limit, name.upper() + '_SIZE')
    raw = path.read_bytes()
    _need(len(raw) == size and hashlib.sha256(raw).hexdigest() == sha,
          name.upper() + '_HASH')
    return raw


def _candidate_receipts() -> dict:
    plan = json.loads(_checked_bytes('plan', 1024**2))
    claim = json.loads(_checked_bytes('claim', 1024**2))
    result = json.loads(_checked_bytes('result', 1024**2))
    terminal = json.loads(_checked_bytes('terminal', 1024**2))
    _need(plan.get('stage') == 'market_state_time_query_tape_20261107_001'
          and plan.get('test_data_used') is False
          and claim.get('plan') == _ref('plan')
          and claim.get('test_data_used') is False
          and result.get('plan') == _ref('plan')
          and result.get('claim', {}).get('path') == _ref('claim')['path']
          and result['claim'].get('sha256') == _ref('claim')['sha256']
          and result.get('status') == 'COMPLETE_CANDIDATE_TAPE_ENDPOINT_PARITY'
          and result.get('candidate_tape', {}).get('path') == _ref('tape')['path']
          and result['candidate_tape'].get('sha256') == _ref('tape')['sha256']
          and result['candidate_tape'].get('bytes') == 4169474
          and result.get('model_fits') == 0
          and result.get('test_data_used') is False
          and result.get('target_selection') is False
          and set(result.get('selected_endpoint_parity', {})) == set(TIMEFRAMES)
          and all(v.get('bit_mismatches') == v.get('m1_source_row_mismatches') == 0
                  and v.get('issues') == 118706
                  for v in result['selected_endpoint_parity'].values())
          and terminal.get('status') == 'COMPLETE'
          and terminal.get('exit_code') == 0
          and terminal.get('result') == _ref('result')
          and terminal.get('candidate_tape') == result['candidate_tape']
          and terminal.get('test_data_used') is False,
          'RECEIPT_CHAIN')
    policy = json.loads((ROOT / 'NEXT_RUN_POLICY.json').read_text())
    stage = policy.get('market_state_time_query_tape_20261107_001', {})
    _need(stage.get('status') == 'CONSUMED_COMPLETE_CANDIDATE_M1_TAPE_ZERO_ENDPOINT_MISMATCH'
          and stage.get('candidate_tape') == result['candidate_tape']
          and stage.get('terminal') == _ref('terminal'), 'POLICY_CONSUMED')
    return result


def _immutable(values: np.ndarray) -> np.ndarray:
    """Own byte-backed arrays so an external view cannot rewrite axis identity."""
    return np.frombuffer(np.ascontiguousarray(values).tobytes(),
                         dtype=values.dtype).reshape(values.shape)


def _endpoint_parity(tape: TimeQueryTargetTape, issues: np.ndarray,
                     old_bits: np.ndarray, old_source: np.ndarray,
                     cutoff: int) -> dict:
    n = len(issues)
    _need(issues.dtype == old_source.dtype == np.int64
          and issues.shape == (n,) and old_source.shape == (n, len(TIMEFRAMES))
          and old_bits.dtype == np.uint8 and old_bits.shape == old_source.shape
          and n > 0, 'ENDPOINT_SHAPE')
    query = np.broadcast_to(_K, (n, len(TIMEFRAMES))).copy()
    got = tape.gather(issues, query, fold_upper_ns=cutoff)
    bit_mismatches = int(np.count_nonzero(got['allowed'] != old_bits))
    source_mismatches = int(np.count_nonzero(got['m1_source_row'] != old_source))
    _need(bit_mismatches == source_mismatches == 0, 'ENDPOINT_PARITY')
    return {'issues': n, 'cells': n * len(TIMEFRAMES),
            'bit_mismatches': 0, 'm1_source_row_mismatches': 0}


class BoundTimeQueryTargets:
    """Only exact model parent/issue pairs and declared fold intervals may query."""

    def __init__(self, parent_rows: np.ndarray, decision_ns: np.ndarray,
                 tape: TimeQueryTargetTape, selected_issue_ns: np.ndarray,
                 selected_allowed: np.ndarray, selected_source: np.ndarray,
                 *, train_end_ns: int):
        _need(isinstance(tape, TimeQueryTargetTape)
              and isinstance(parent_rows, np.ndarray)
              and isinstance(decision_ns, np.ndarray)
              and parent_rows.dtype == decision_ns.dtype == np.int64
              and parent_rows.ndim == decision_ns.ndim == 1
              and len(parent_rows) == len(decision_ns) > 0
              and np.all(np.diff(parent_rows) > 0)
              and np.all(np.diff(decision_ns) > 0)
              and np.all(decision_ns % TICK_NS == 0)
              and type(train_end_ns) is int and train_end_ns > int(decision_ns[-1]),
              'MODEL_AXIS')
        self.parent_rows = _immutable(parent_rows)
        self.decision_ns = _immutable(decision_ns)
        self.tape = tape
        self.endpoint_parity = _endpoint_parity(
            tape, selected_issue_ns, selected_allowed, selected_source,
            train_end_ns)

    def gather(self, parent_rows: np.ndarray, issue_ns: np.ndarray,
               query_k: np.ndarray, *, fold_id: int, partition: str,
               pass_index: int | None = None,
               seed: int | None = None) -> dict[str, np.ndarray]:
        _need(type(fold_id) is int and 1 <= fold_id <= 5
              and partition in ('fit', 'score'), 'FOLD')
        _need(isinstance(parent_rows, np.ndarray) and parent_rows.dtype == np.int64
              and isinstance(issue_ns, np.ndarray) and issue_ns.dtype == np.int64
              and parent_rows.ndim == issue_ns.ndim == 1
              and len(parent_rows) == len(issue_ns) > 0, 'QUERY_AXIS')
        at = np.searchsorted(self.parent_rows, parent_rows)
        _need(bool(np.all(at < len(self.parent_rows)))
              and np.array_equal(self.parent_rows[at], parent_rows)
              and np.array_equal(self.decision_ns[at], issue_ns), 'EXACT_PARENT_ISSUE')
        boundary = old.FOLD_BOUNDARIES_NS[fold_id - 1]
        if partition == 'fit':
            lower, issue_upper = old.TRAIN_START_NS, boundary - 1440 * old.MINUTE_NS
            target_upper = boundary
        else:
            lower, issue_upper = boundary, old.FOLD_BOUNDARIES_NS[fold_id]
            target_upper = issue_upper
        _need(bool(np.all((issue_ns >= lower) & (issue_ns < issue_upper))),
              'ISSUE_PARTITION')
        if partition == 'fit':
            expected = sampled_query_k(issue_ns, pass_index=pass_index, seed=seed)
            _need(np.array_equal(query_k, expected), 'FIT_SCHEDULE')
        got = self.tape.gather(issue_ns, query_k, fold_upper_ns=target_upper)
        return {'parent_rows': parent_rows.copy(), **got}

    def gather_fit_sampled(self, parent_rows: np.ndarray, issue_ns: np.ndarray,
                           *, fold_id: int, pass_index: int, seed: int
                           ) -> dict[str, np.ndarray]:
        """Choose fit queries before reading any target cell."""
        query_k = sampled_query_k(issue_ns, pass_index=pass_index, seed=seed)
        return self.gather(parent_rows, issue_ns, query_k, fold_id=fold_id,
                           partition='fit', pass_index=pass_index, seed=seed)


def sampled_query_k(issue_ns: np.ndarray, *, pass_index: int, seed: int) -> np.ndarray:
    """Label-blind, batch-order-independent k for one training pass."""
    _need(isinstance(issue_ns, np.ndarray) and issue_ns.dtype == np.int64
          and issue_ns.ndim == 1 and len(issue_ns) > 0
          and bool(np.all(issue_ns >= 0))
          and bool(np.all(issue_ns % TICK_NS == 0))
          and type(pass_index) is int and 0 <= pass_index < 2**64
          and type(seed) is int and 0 <= seed < 2**64,
          'SCHEDULE_INPUT')
    result = np.empty((len(issue_ns), len(TIMEFRAMES)), dtype=np.int64)
    for row, issue in enumerate(issue_ns):
        tick = int(issue // TICK_NS)
        for col, limit in enumerate(NATIVE_K):
            packed = struct.pack('<QQQQ', tick, col, pass_index, seed)
            digest = hashlib.blake2b(packed, digest_size=8).digest()
            result[row, col] = 1 + int.from_bytes(digest, 'little') % limit
    return result


def load_bound_time_query_targets() -> BoundTimeQueryTargets:
    """Read pinned COMPLETE candidate and old selected/model axes; no raw M1."""
    result = _candidate_receipts()
    with np.load(io.BytesIO(_checked_bytes('tape', 32 * 1024**2)),
                 allow_pickle=False) as saved:
        _need(set(saved.files) == TAPE_KEYS, 'TAPE_MEMBERS')
        axis = np.array(saved['validity_ns'], copy=True)
        source = np.array(saved['m1_source_row'], copy=True)
        bits = np.array(saved['allowed_set'], copy=True)
        path = np.array(saved['path_observed'], copy=True)
        gap = np.array(saved['gap_kind'], copy=True)
    _need(axis.shape == source.shape == gap.shape == (591682,)
          and bits.shape == path.shape == (591682, 5)
          and axis.dtype == source.dtype == np.int64
          and bits.dtype == gap.dtype == np.uint8
          and path.dtype == np.bool_
          and np.array_equal(source, np.arange(5193205, 5193205 + 591682))
          and np.isin(gap, (0, 1, 2)).all()
          and int((gap == 1).sum()) == 413
          and int((gap == 2).sum()) == 95
          and np.all(path <= np.isin(bits, (1, 2, 4)))
          and result.get('tape_report', {}).get('m1_rows') == 591682,
          'TAPE_SCHEMA')
    tape = TimeQueryTargetTape(axis, source, bits)
    bound = old.load_bound_five_endpoint_targets()
    with np.load(io.BytesIO(old._checked_bytes('mask', 4 * 1024**2)),
                 allow_pickle=False) as selected:
        _need(set(selected.files) == old.MASK_KEYS, 'SELECTED_MEMBERS')
        issue = np.array(selected['issue_ns'], copy=True)
        allowed = np.array(selected['allowed_set'], copy=True)
        endpoint_source = np.array(selected['m1_source_row'], copy=True)
    _need(len(bound.decision_ns) == 118249
          and len(issue) == 118706
          and bound.unjoined_target_rows == 457,
          'BOUND_ROWS')
    return BoundTimeQueryTargets(
        bound.parent_rows, bound.decision_ns, tape, issue, allowed,
        endpoint_source, train_end_ns=old.FOLD_BOUNDARIES_NS[-1])
