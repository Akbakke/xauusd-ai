"""Separate, unadmitted F direction target on pinned TRAIN endpoint prices.

S market state keeps its own labels. F measures displacement from the closed
M5 decision price to an exact later close; neither target enters model forward.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state

ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_DECISION_RETURN_GC_20261108_002')
REFS = {
    'tape': (ROOT / 'DECISION_RETURN_TAPE.npz',
             'f87dcc7c669e86a10325a526023933346926f17037685a3082a875e67a1fb167'),
    'result': (ROOT / 'RESULT.json',
               '0998ed88ae6118293f193ace6d4e05660dad6d7abdb67fada2955730fdc03fc9'),
    'terminal': (ROOT / 'TERMINAL.json',
                 '1f141813ccecdc445e547b98f1c3da0cd3da8c6159555c57ee3642875253bc31'),
}
HALF_ATR_T = 0.5  # Candidate inherited numerically from S; no score-based tuning.
TAPE_KEYS = frozenset({'issue_ns', 'target_valid', 'future_sign',
                       'abs_future_over_atr_t', 'exact_endpoint', 'gap_between'})


def _need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise ValueError('FUTURE_DISPLACEMENT_' + reason)


def contract() -> dict:
    value = {
        'schema_version': 'gx1_market_state_future_displacement_candidate_v1',
        'role': 'separate_research_candidate_no_fit_or_trade_authority',
        'target': 'F_s(t,v)=closed_XAU_M5_price(v)-closed_XAU_M5_price(t)',
        'timeframes': ['M5', 'M15', 'H1', 'H4', 'D1'],
        'native_lead_minutes': list(state.LEAD_MINUTES),
        'magnitude': 'abs(F)/last_completed_native_TF_ATR_at_t',
        'candidate_activity_threshold': HALF_ATR_T,
        'allowed_bits': {'DOWN': 1, 'FLAT_NET': 2, 'UP': 4, 'UNKNOWN': 7},
        'side_rule': 'known_only_when_abs_F_ge_half_ATR_t_and_exact_endpoint_mature',
        'S_state_relabeling': False,
        'intermediate_M5_clock_gap': 'reported_not_a_path_or_hold_label',
        'future_target_as_model_forward_input': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def allowed_bits(sign: np.ndarray, magnitude: np.ndarray,
                 valid: np.ndarray) -> np.ndarray:
    """Pure candidate classification; small net changes teach only F activity."""
    _need(isinstance(sign, np.ndarray) and sign.dtype == np.int8
          and sign.ndim == 2 and sign.shape[1] == 5
          and isinstance(magnitude, np.ndarray) and magnitude.dtype == np.float64
          and magnitude.shape == sign.shape
          and isinstance(valid, np.ndarray) and valid.dtype == np.bool_
          and valid.shape == sign.shape
          and np.isin(sign, (-1, 0, 1)).all()
          and np.isfinite(magnitude).all() and np.all(magnitude >= 0)
          and np.array_equal(sign[valid] == 0, magnitude[valid] == 0)
          and np.all(sign[~valid] == 0) and np.all(magnitude[~valid] == 0),
          'NUMERIC_DOMAIN')
    bits = np.full(sign.shape, 7, dtype=np.uint8)
    bits[valid] = 2
    large = valid & (magnitude >= HALF_ATR_T)
    _need(np.all(sign[large] != 0), 'LARGE_ZERO_SIGN')
    bits[large & (sign < 0)] = 1
    bits[large & (sign > 0)] = 4
    return bits


def _read(name: str, limit: int) -> bytes:
    path, expected = REFS[name]
    _need(path.is_file() and not path.is_symlink(), name.upper() + '_FILE')
    before = path.stat()
    _need(0 < before.st_size <= limit, name.upper() + '_SIZE')
    raw = path.read_bytes()
    after = path.stat()
    _need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
           before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size,
                                   after.st_mtime_ns, after.st_ctime_ns)
          and len(raw) == before.st_size
          and hashlib.sha256(raw).hexdigest() == expected,
          name.upper() + '_HASH')
    return raw


@dataclass(frozen=True)
class BoundFutureDisplacement:
    parent_rows: np.ndarray
    decision_ns: np.ndarray
    valid: np.ndarray
    sign: np.ndarray
    magnitude: np.ndarray
    gap_between: np.ndarray

    def gather(self, parents: np.ndarray, issues: np.ndarray, *,
               fold_id: int, partition: str) -> dict[str, np.ndarray]:
        _need(type(fold_id) is int and 1 <= fold_id <= 5
              and partition in ('fit', 'score')
              and isinstance(parents, np.ndarray) and parents.dtype == np.int64
              and isinstance(issues, np.ndarray) and issues.dtype == np.int64
              and parents.ndim == issues.ndim == 1
              and len(parents) == len(issues) > 0, 'QUERY')
        at = np.searchsorted(self.parent_rows, parents)
        _need(np.all(at < len(self.parent_rows))
              and np.array_equal(self.parent_rows[at], parents)
              and np.array_equal(self.decision_ns[at], issues), 'EXACT_JOIN')
        lower, upper = ((state.TRAIN_START_NS, state.FOLD_BOUNDARIES_NS[fold_id - 1])
                        if partition == 'fit' else
                        (state.FOLD_BOUNDARIES_NS[fold_id - 1],
                         state.FOLD_BOUNDARIES_NS[fold_id]))
        _need(np.all((issues >= lower) & (issues < upper)), 'PARTITION')
        validity_ns = issues[:, None] + np.asarray(state.LEAD_MINUTES, np.int64)[None, :] * state.MINUTE_NS
        mature = validity_ns < upper
        selected = allowed_bits(self.sign[at], self.magnitude[at], self.valid[at])
        return {'parent_rows': parents.copy(), 'issue_ns': issues.copy(),
                'validity_ns': validity_ns, 'allowed': selected,
                'eligible': self.valid[at].copy(), 'mature': mature,
                'gap_between': self.gap_between[at].copy()}


def load_bound_future_displacement() -> BoundFutureDisplacement:
    """Validate the frozen F receipt and join it to existing S model rows."""
    result = json.loads(_read('result', 1024 * 1024))
    terminal = json.loads(_read('terminal', 1024 * 1024))
    refs = {name: {'path': str(path), 'sha256': digest}
            for name, (path, digest) in REFS.items()}
    _need(result.get('schema_version') == 'gx1_market_state_decision_return_gc_result_v1'
          and result.get('status') == 'COMPLETE'
          and result.get('target_tape') == refs['tape']
          and result.get('test_data_used') is False
          and terminal.get('schema_version') == 'gx1_market_state_decision_return_gc_terminal_v1'
          and terminal.get('status') == 'COMPLETE'
          and terminal.get('exit_code') == 0
          and terminal.get('result') == refs['result']
          and terminal.get('target_tape') == refs['tape']
          and terminal.get('test_data_used') is False, 'SOURCE_RECEIPTS')
    source = state.load_bound_five_endpoint_targets()
    with np.load(io.BytesIO(_read('tape', 16 * 1024 * 1024)), allow_pickle=False) as tape:
        _need(set(tape.files) == TAPE_KEYS, 'TAPE_MEMBERS')
        issue = tape['issue_ns']
        valid = tape['target_valid']
        sign = tape['future_sign']
        mag = tape['abs_future_over_atr_t']
        exact = tape['exact_endpoint']
        gap = tape['gap_between']
    n = len(issue)
    _need(issue.dtype == np.int64 and issue.shape == (118706,)
          and np.all(np.diff(issue) > 0)
          and valid.dtype == exact.dtype == gap.dtype == np.bool_
          and sign.dtype == np.int8 and mag.dtype == np.float64
          and valid.shape == sign.shape == mag.shape == exact.shape == gap.shape == (n, 5)
          and np.all(valid <= exact), 'TAPE_SCHEMA')
    allowed_bits(sign, mag, valid)
    at = np.searchsorted(issue, source.decision_ns)
    _need(len(at) == len(source.parent_rows) == 118249
          and np.all(at < n) and np.array_equal(issue[at], source.decision_ns),
          'MODEL_ISSUE_JOIN')
    arrays = (source.parent_rows, source.decision_ns, valid[at].copy(),
              sign[at].copy(), mag[at].copy(), gap[at].copy())
    for array in arrays:
        array.setflags(write=False)
    return BoundFutureDisplacement(*arrays)
