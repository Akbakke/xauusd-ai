"""Score 60 distinct midpoint-direction outputs; no fitting or file access."""
from __future__ import annotations

import numpy as np

from gx1.contracts.entry_direction_probability_reference_v1 import direction_rank_diagnostics
from gx1.contracts.entry_native_direction_context_report_v1 import _panel
from gx1.contracts.entry_native_direction_dense_v1 import HORIZONS_MINUTES

MINUTE_NS = 60 * 10**9
MAX_NS = 2**63 - 1


def _require(ok, reason):
    if not ok:
        raise RuntimeError('NATIVE_DIRECTION_DENSE_REPORT_' + reason)


def score_dense_direction_logits(logits, *, sign, observed, decision_ns, terminal_time_ns, cutoff_ns):
    """Report each own minute and its observed, mature, nonzero row counts.

    AUC is within the supplied cohort and not a confidence interval, trade
    result, selection rule, or evidence of generalization. Rows never move.
    """
    _require(isinstance(logits, np.ndarray) and logits.dtype in (np.float32, np.float64)
        and logits.ndim == 2 and logits.shape[0] > 0 and logits.shape[1] == 60
        and bool(np.isfinite(logits).all()), 'LOGITS')
    size = len(logits)
    _require(isinstance(sign, np.ndarray) and sign.dtype == np.int8 and sign.shape == (size, 60), 'SIGN')
    _require(isinstance(observed, np.ndarray) and observed.dtype == np.bool_ and observed.shape == (size, 60), 'OBSERVED')
    _require(isinstance(decision_ns, np.ndarray) and decision_ns.dtype == np.int64 and decision_ns.shape == (size,), 'DECISION')
    _require(isinstance(terminal_time_ns, np.ndarray) and terminal_time_ns.dtype == np.int64
        and terminal_time_ns.shape == (size, 60), 'TERMINAL')
    _require(type(cutoff_ns) is int and 0 < cutoff_ns <= MAX_NS, 'CUTOFF')
    _require(bool(np.isin(sign, (-1, 0, 1)).all()) and bool((sign[~observed] == 0).all()), 'SIGN_DOMAIN')
    _require(bool((decision_ns > 0).all()) and bool((decision_ns < cutoff_ns).all())
        and bool((decision_ns <= MAX_NS - 60 * MINUTE_NS).all()), 'DECISION_CUTOFF')
    expected = decision_ns[:, None] + np.asarray(HORIZONS_MINUTES, dtype=np.int64)[None, :] * MINUTE_NS
    _require(np.array_equal(terminal_time_ns, expected), 'TERMINAL_CLOCK')
    mature = terminal_time_ns < cutoff_ns
    panels = []
    for i, horizon in enumerate(HORIZONS_MINUTES):
        eligible = observed[:, i] & (sign[:, i] != 0) & mature[:, i]
        labels = np.ascontiguousarray((sign[eligible, i] == 1).astype(np.int8))
        scores = np.ascontiguousarray(logits[eligible, i], dtype=np.float64)
        rank = direction_rank_diagnostics(labels, scores, score_kind='logit_up')
        panels.append({'horizon_minutes': horizon, **_panel(logits[:, i], sign[:, i], observed[:, i], mature[:, i]),
            'rank': rank})
    return {'schema_version': 'gx1_native_direction_dense_report_v1', 'per_horizon': panels,
        'scope': 'unweighted observed mature nonzero midpoint direction; supplied cohort only',
        'economic_edge_qualified': False}
