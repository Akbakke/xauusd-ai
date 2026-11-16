"""Causal five-closed-M1-bar path at one M5 decision clock.

A caller supplies hash-checked M1 OHLC from TRAIN or one serving pair. This
pure lookup neither reads sources nor infers missing minutes or fit authority.
The current M1 bar starts one minute before the M5 decision and closes at it.
"""
from __future__ import annotations

import numpy as np

MINUTE_NS = 60_000_000_000
M5_NS = 5 * MINUTE_NS
OFFSETS = np.arange(5, 0, -1, dtype=np.int64) * MINUTE_NS


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('MARKET_STATE_TA_M1_PATH_' + reason)


class M1IssuePathTape:
    """Own observed M1 mid OHLC, then gather exact past-only M5 windows."""

    def __init__(self, bar_start_ns: np.ndarray, mid_ohlc: np.ndarray):
        _need(isinstance(bar_start_ns, np.ndarray)
              and bar_start_ns.dtype == np.int64 and bar_start_ns.ndim == 1
              and len(bar_start_ns) > 0
              and isinstance(mid_ohlc, np.ndarray)
              and mid_ohlc.dtype == np.float64
              and mid_ohlc.shape == (len(bar_start_ns), 4), 'SOURCE_SHAPE')
        _need(bool(np.all(bar_start_ns >= 0))
              and bool(np.all(bar_start_ns % MINUTE_NS == 0))
              and bool(np.all(np.diff(bar_start_ns) > 0))
              and bool(np.isfinite(mid_ohlc).all())
              and bool(np.all(mid_ohlc > 0))
              and bool(np.all(mid_ohlc[:, 1] >= np.maximum(mid_ohlc[:, 0], mid_ohlc[:, 3])))
              and bool(np.all(mid_ohlc[:, 2] <= np.minimum(mid_ohlc[:, 0], mid_ohlc[:, 3])))
              and bool(np.all(mid_ohlc[:, 1] >= mid_ohlc[:, 2])), 'SOURCE_DOMAIN')
        self.bar_start_ns = np.frombuffer(bar_start_ns.tobytes(), dtype=np.int64)
        self.mid_ohlc = np.frombuffer(mid_ohlc.tobytes(), dtype=np.float64).reshape(-1, 4)

    def gather(self, issue_ns: np.ndarray) -> dict[str, np.ndarray]:
        """Read starts t-5..t-1 only; mask every absent bar without dropping t."""
        _need(isinstance(issue_ns, np.ndarray) and issue_ns.dtype == np.int64
              and issue_ns.ndim == 1 and len(issue_ns) > 0
              and bool(np.all(issue_ns >= M5_NS))
              and bool(np.all(issue_ns % M5_NS == 0)), 'ISSUE_CLOCK')
        wanted = issue_ns[:, None] - OFFSETS[None, :]
        at = np.searchsorted(self.bar_start_ns, wanted)
        present = at < len(self.bar_start_ns)
        present[present] &= self.bar_start_ns[at[present]] == wanted[present]
        values = np.zeros((*wanted.shape, 4), dtype=np.float64)
        values[present] = self.mid_ohlc[at[present]]
        last_present = present[:, -1]
        reference = np.where(last_present, values[:, -1, 3], 1.0)
        usable = present & last_present[:, None]
        bps = np.where(usable[:, :, None],
                       (values / reference[:, None, None] - 1.0) * 10_000.0,
                       0.0).astype(np.float32)
        source_index = np.where(present, at, -1).astype(np.int64)
        return {'issue_ns': issue_ns.copy(), 'm1_bar_start_ns': wanted,
                'mid_ohlc_bps': bps, 'observed': present,
                'full_path': present.all(axis=1), 'last_bar_present': last_present,
                'source_index': source_index}
