#!/usr/bin/env python3
"""V12 live canonical_v3 builder — produces the same canonical_v3 features
the batch pipeline produces, but on a rolling slice of recent M1 data so it
stays fresh as the OANDA collector writes new bars.

Pipeline mirrored from batch builders:
    1. Union of collector M1 parquets + canonical M1 tape over a `lookback_hours`
       window ending at `end_ts`.
    2. Aggregate M1 → M5 (open=first, high=max, low=min, close=last, volume=sum)
       dropping any incomplete final 5-min bucket.
    3. Run gx1.scripts.materialize_build_canonical_features_v2.build_canonical_v2
       on the M5 slice → ~112-118 canonical_v2 columns.
    4. Apply gx1.scripts.materialize_canonical_v3_augment:
         - drop 12 redundant columns (PAIRS_TO_PRUNE)
         - +4 cyclic time features
         - +1 smc_premium_state
         - +1 m5h1_momentum

NOT a complete XGB v5 input — the production XGB v5 contract requires
~32 additional features added by `add_ctx_cont_columns_to_prebuilt.py`
(atr_bps, spread_bps, session_id, trend_regime_id, micro_momentum_*,
swing distances, etc.). Those are deferred to a follow-up module.

Performance: cold call ~3-4 sec on 3-day window (~300 M5 bars), dominated
by canonical_v2 ATR/RSI/SMC rolling computations. M5-bucket caching keeps
warm calls free until a new 5-min bar closes.

Usage:
    builder = LiveCanonicalV3Builder(
        collector_dir=Path("/home/andre2/GX1_DATA/reports/v12_live_data"),
        canonical_m1_dir=Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL"),
    )
    cv3 = builder.compute(end_ts=pd.Timestamp.now(tz="UTC").floor("5min"))
    latest_features = cv3.iloc[-1]   # 112-col canonical_v3 row for freshest M5 bar
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.execution.v12_m1_to_m5_downsample import m1_to_m5
from gx1.scripts.materialize_build_canonical_features_v2 import build_canonical_v2
from gx1.scripts.materialize_canonical_v3_augment import (
    DROP_COLUMNS,
    add_cross_tf_momentum,
    add_cyclic_time_features,
    add_smc_premium_state_interaction,
)

LOG = logging.getLogger("v12_canonical_live")

# Default lookback: 30 days. Empirical equivalence vs batch prebuilt:
#   72h  → 99/112 features exact match  (3-day rolling features stable)
#   336h → 104/112 features exact match (H1/H4 EMAs converge)
#   720h → 108/112 features exact match (D1 ATR14 stable; only year-scale
#                                         features like _v1_range_adr drift)
# Cold-build cost at 720h: ~1.3 sec on 6000+ M5 bars. Acceptable for M5-cadence
# decisions (one cold call every 5 min, cached in-between).
DEFAULT_LOOKBACK_HOURS = 30 * 24  # 30 days
MIN_M5_BARS_REQUIRED = 50         # canonical_v2 needs enough history for rolling features


@dataclass
class _BuildCache:
    last_bucket: pd.Timestamp | None = None
    canonical_v3: pd.DataFrame | None = None


@dataclass
class LiveCanonicalV3Builder:
    collector_dir: Path
    canonical_m1_dir: Path
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS
    _cache: _BuildCache = field(default_factory=_BuildCache)

    # ── data loading ──────────────────────────────────────────────────────

    def _load_m1_window(self, end_ts: pd.Timestamp) -> pd.DataFrame:
        """Union of collector parquets + canonical M1 tape over the lookback window."""
        start_ts = end_ts - pd.Timedelta(hours=self.lookback_hours)
        parts: list[pd.DataFrame] = []

        for fp in sorted(self.collector_dir.glob("xauusd_m1_*.parquet")):
            try:
                df = pd.read_parquet(fp)
            except Exception as exc:
                LOG.warning(f"skipping collector file {fp.name}: {exc}")
                continue
            df["time"] = pd.to_datetime(df["time"], utc=True)
            sub = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)]
            if len(sub) > 0:
                parts.append(sub)

        for yr in range(start_ts.year, end_ts.year + 1):
            fp = self.canonical_m1_dir / f"year={yr}" / "part-000.parquet"
            if not fp.exists():
                continue
            try:
                df = pd.read_parquet(fp)
            except Exception as exc:
                LOG.warning(f"skipping canonical M1 file {fp.name}: {exc}")
                continue
            df["time"] = pd.to_datetime(df["time"], utc=True)
            sub = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)]
            if len(sub) > 0:
                parts.append(sub)

        if not parts:
            return pd.DataFrame()
        return (pd.concat(parts, ignore_index=True)
                .drop_duplicates(subset=["time"], keep="last")
                .sort_values("time")
                .reset_index(drop=True))

    # ── augmentation ──────────────────────────────────────────────────────

    @staticmethod
    def _apply_v3_augment(v2: pd.DataFrame) -> pd.DataFrame:
        """In-memory equivalent of materialize_canonical_v3_augment.main()."""
        v3 = v2.copy()
        if "time" in v3.columns and not isinstance(v3.index, pd.DatetimeIndex):
            v3["time"] = pd.to_datetime(v3["time"], utc=True)
            v3 = v3.set_index("time")
        to_drop = [c for c in DROP_COLUMNS if c in v3.columns]
        v3 = v3.drop(columns=to_drop)
        v3 = add_cyclic_time_features(v3)
        v3 = add_smc_premium_state_interaction(v3)
        v3 = add_cross_tf_momentum(v3)
        return v3

    # ── public API ────────────────────────────────────────────────────────

    def compute(self, end_ts: pd.Timestamp) -> pd.DataFrame:
        """Return canonical_v3 features for all M5 bars up to (and including) `end_ts`.

        The freshest M5 bar is at .iloc[-1]. DataFrame is time-indexed (DatetimeIndex).
        Returns empty DataFrame if M1 history is insufficient.

        Caching: result is cached per 5-min bucket. Calling .compute() twice in
        the same M5 bucket returns the cached result.
        """
        cur_bucket = end_ts.floor("5min")
        if self._cache.last_bucket == cur_bucket and self._cache.canonical_v3 is not None:
            return self._cache.canonical_v3

        t0 = time.perf_counter()
        m1 = self._load_m1_window(end_ts)
        if m1.empty:
            LOG.warning(f"no M1 data for window ending at {end_ts}")
            return pd.DataFrame()
        m5 = m1_to_m5(m1)
        if len(m5) < MIN_M5_BARS_REQUIRED:
            LOG.warning(f"only {len(m5)} M5 bars (need {MIN_M5_BARS_REQUIRED}+); "
                         f"increase lookback_hours or wait for more history")
            return pd.DataFrame()

        v2 = build_canonical_v2(m5)
        v3 = self._apply_v3_augment(v2)
        elapsed = time.perf_counter() - t0
        LOG.info(f"canonical_v3 built: {v3.shape[0]} M5 bars × {v3.shape[1]} cols  "
                  f"in {elapsed*1000:.0f} ms  latest={v3.index[-1]}")

        self._cache.last_bucket = cur_bucket
        self._cache.canonical_v3 = v3
        return v3

    def latest_features(self, end_ts: pd.Timestamp) -> dict[str, Any] | None:
        """Convenience: return the latest M5 bar's features as a dict.

        Returns None if compute() fails (insufficient history, no M1 data, etc.).
        """
        cv3 = self.compute(end_ts)
        if cv3.empty:
            return None
        row = cv3.iloc[-1]
        out: dict[str, Any] = {"time": cv3.index[-1].isoformat()}
        for col, val in row.items():
            try:
                out[col] = float(val) if pd.notna(val) else None
            except (TypeError, ValueError):
                out[col] = str(val) if pd.notna(val) else None
        return out
