"""Read-only prospective S-side turn score on fixed native-cadence panels."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from gx1.audit.observe_market_state_ta_fold2_fresh_initial_v1 import auc_with_ties
from gx1.audit.score_market_state_ta_fold2_m5_fit_v1 import (
    DAY_NS, interval, weighted_auc,
)
from gx1.contracts.market_state_ta_fold2_m5_fit_v1 import BOOTSTRAP_SEEDS

TF_MINUTES = (('M5', 5), ('M15', 15), ('H1', 60), ('H4', 240), ('D1', 1440))
MINUTE_NS = 60 * 10**9
PANEL_ROWS = 4096
FIELDS = ('parent_rows', 'issue_ns', 'state_raw', 'state_allowed',
          'state_eligible', 'state_mature')


def need(ok: bool, reason: str) -> None:
    if not bool(ok):
        raise ValueError('TA_INTEGRATED_TURN_' + reason)


def read_capture(ref: dict) -> dict:
    """Read one immutable private panel, never a model or target provider."""
    need(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'REF_FIELDS')
    path = Path(ref['path'])
    need(path.is_absolute() and path.is_file() and not path.is_symlink()
         and hashlib.sha256(path.read_bytes()).hexdigest() == ref['sha256'], 'REF_BYTES')
    with np.load(path, allow_pickle=False) as archive:
        need(set(FIELDS).issubset(archive.files), 'CAPTURE_FIELDS')
        data = {key: archive[key].copy() for key in FIELDS}
    require_capture(data)
    return data


def require_capture(data: dict) -> None:
    need(type(data) is dict and set(data) == set(FIELDS), 'CAPTURE_FIELDS')
    rows, issues = data['parent_rows'], data['issue_ns']
    need(rows.shape == issues.shape == (PANEL_ROWS,)
         and rows.dtype == issues.dtype == np.int64
         and bool(np.all(np.diff(rows) > 0)) and bool(np.all(np.diff(issues) > 0))
         and data['state_raw'].shape == (PANEL_ROWS, 5, 2)
         and data['state_raw'].dtype == np.float32
         and bool(np.isfinite(data['state_raw']).all())
         and data['state_allowed'].shape == (PANEL_ROWS, 5)
         and data['state_allowed'].dtype == np.uint8
         and data['state_eligible'].shape == data['state_mature'].shape == (PANEL_ROWS, 5)
         and data['state_eligible'].dtype == data['state_mature'].dtype == np.bool_,
         'CAPTURE_SHAPE')


def native_pairs(data: dict, tf_index: int, gap_minutes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Current side is the previous issue's matured native S endpoint."""
    issues = data['issue_ns']
    prior_issue = issues - gap_minutes * MINUTE_NS
    earlier = np.searchsorted(issues, prior_issue)
    exact = (earlier < len(issues)) & (issues[np.minimum(earlier, len(issues) - 1)] == prior_issue)
    current = np.flatnonzero(exact)
    previous = earlier[current]
    allowed = data['state_allowed'][:, tf_index]
    known = (data['state_eligible'][current, tf_index]
             & data['state_mature'][current, tf_index]
             & data['state_eligible'][previous, tf_index]
             & data['state_mature'][previous, tf_index]
             & np.isin(allowed[current], (1, 4))
             & np.isin(allowed[previous], (1, 4)))
    return current[known], previous[known], (allowed[current[known]] != allowed[previous[known]])


def _phase(data: dict, current: np.ndarray, previous: np.ndarray,
           turns: np.ndarray, tf_index: int) -> tuple[dict, np.ndarray]:
    logits = data['state_raw'][current, tf_index, 1].astype(np.float64)
    p_up = 1. / (1. + np.exp(-np.clip(logits, -80., 80.)))
    was_up = data['state_allowed'][previous, tf_index] == 4
    p_turn = np.where(was_up, 1. - p_up, p_up)
    predicted_turn = (logits >= 0.) != was_up
    count = len(turns)
    return ({'known_pair_count': count, 'actual_switch_count': int(turns.sum()),
             'switch_probability_auc': auc_with_ties(p_turn, turns, np),
             'switch_probability_brier': (float(np.square(p_turn - turns).mean())
                                          if count else None),
             'predicted_switch_count': int(predicted_turn.sum()),
             'true_positive_switch_count': int((predicted_turn & turns).sum())}, p_turn)


def score_fixed_later(initial: dict, final: dict) -> dict:
    """One predeclared M5 primary; other TF point estimates retain unknowns."""
    for data in (initial, final):
        require_capture(data)
    for key in FIELDS:
        if key != 'state_raw':
            need(np.array_equal(initial[key], final[key]), 'PAIRED_AXES_AND_TRUTH')
    months = initial['issue_ns'].astype('datetime64[ns]').astype('datetime64[M]').astype(str)
    need(tuple(np.unique(months)) == ('2025-01', '2025-02', '2025-03'), 'LATER_MONTHS')
    result = {}
    primary = None
    for j, (tf, minutes) in enumerate(TF_MINUTES):
        current, previous, turns = native_pairs(initial, j, minutes)
        scores = {}
        probabilities = {}
        for name, data in (('INITIAL', initial), ('FINAL', final)):
            scores[name], probabilities[name] = _phase(data, current, previous, turns, j)
        result[tf] = {'issue_gap_minutes': minutes, 'scores': scores}
        if j == 0:
            primary = (current, turns, probabilities, scores)
    current, turns, probabilities, scores = primary
    need(len(current) > 0 and turns.any() and (~turns).any(), 'M5_PRIMARY_SUPPORT')
    uncertainty = {}
    days = initial['issue_ns'][current] // DAY_NS
    for scheme, groups in (('UTC_DAY', days), ('UTC_MONDAY_WEEK', (days + 3) // 7)):
        unique, inverse = np.unique(groups, return_inverse=True)
        draws = np.random.default_rng(BOOTSTRAP_SEEDS[scheme]).multinomial(
            len(unique), np.full(len(unique), 1. / len(unique)), size=2000)
        weights = draws[:, inverse].astype(np.float64)
        auc = {name: weighted_auc(p, turns, weights)
               for name, p in probabilities.items()}
        brier = {name: np.divide((weights * np.square(p - turns)).sum(axis=1),
                                weights.sum(axis=1),
                                out=np.full(len(weights), np.nan),
                                where=weights.sum(axis=1) > 0)
                 for name, p in probabilities.items()}
        uncertainty[scheme] = {
            'clusters': len(unique),
            'final_minus_initial_auc': interval(
                auc['FINAL'] - auc['INITIAL'],
                scores['FINAL']['switch_probability_auc']
                - scores['INITIAL']['switch_probability_auc']),
            'final_minus_initial_brier': interval(
                brier['FINAL'] - brier['INITIAL'],
                scores['FINAL']['switch_probability_brier']
                - scores['INITIAL']['switch_probability_brier']),
            'final_auc_minus_half': interval(
                auc['FINAL'] - .5, scores['FINAL']['switch_probability_auc'] - .5),
        }
    return {'schema_version': 'gx1_ta_integrated_fixed_later_turn_readout_v1',
            'primary': 'prospective_M5_S_switch', 'timeframes': result,
            'primary_uncertainty': uncertainty, 'test_data_used': False,
            'economic_edge_proven': False}
