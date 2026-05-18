#!/usr/bin/env python3
"""V12 state loader — reads features DIRECTLY from disk prebuilts.

This module replaces v12_canonical_live.py + v12_ctx_augment_live.py.
Those tried to RECOMPUTE features from raw M1 data, which subtly diverged
from training-time distributions (different M5 aggregation, different
rolling-window percentiles for buckets, different bid/ask handling).

The correct architecture: use the EXACT same prebuilt files the cemented
V12.1.1 models were trained on. By construction, live state = batch state,
so XGB / V10 / Entry-IQL / Exit-IQL inference reproduces Phase 6 backtest
output exactly.

Architecture:
    Daily cron (gx1/execution/v12_canonical_rebuild.sh) extends:
        - canonical_v3 prebuilt   ← extended with new M5 bars
        - BASE34_CTX16CAT6 prebuilt ← extended with new ctx features

    Live (every M1 tick):
        - Look up the latest closed M5 bar in canonical_v3 prebuilt
        - Join with BASE28 augmented features at the same timestamp
        - Result: 92-feature XGB input identical to training distribution

The two prebuilts:

  canonical_v3 prebuilt (M5 cadence, 112 cols):
    Path: GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/
            xauusd_m5_CANONICAL_V3_2020_2026.parquet
    Contains: M5 OHLC + 84 canonical_v2 features + 4 cyclic + smc_premium_state
              + m5h1_momentum  (108 features + time + open/high/low/close/volume)
    Built by: materialize_build_canonical_features_v2.py
              + materialize_canonical_v3_augment.py

  BASE28 / BASE34 manifest target (M5 cadence, 57 cols including 32 augmented):
    Path: GX1_DATA/data/data/prebuilt/MONDAY_WEEK_EXTENSION_CANDIDATES/
            monday_week_prebuilt_extension_20260430_123100/
            xauusd_m1_EXPANDED_BASE34_CTX16CAT6_20201109_20260420_RAW_INDEX.parquet
    Contains: 32 augmented features (atr_bps, session_id, regime/bucket,
              HTF, micro_momentum, swing, _v1_is_EU/US, _v1_int_*_us, etc.)
    Built by: add_ctx_cont_columns_to_prebuilt.py

Together they provide all 92 features the cemented XGB v5 needs.

Staleness: BASE28 covers up to 2026-04-20; canonical_v3 covers up to
2026-05-08. For live trading post these dates, the daily rebuild cron
must extend both. Until then, paper-runner can only decide on bars
that exist in both prebuilts (i.e. before the staleness cutoff).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger("v12_state_from_prebuilt")

CANONICAL_V3_PREBUILT = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/"
    "xauusd_m5_CANONICAL_V3_2020_2026.parquet"
)
# Joined prebuilt: canonical_v3 + 32 BASE28 ctx_cont/cat features. Produced
# by add_ctx_cont_columns_to_prebuilt.py on the canonical_v3 prebuilt.
# Preferred over reading two prebuilts separately when it exists.
JOINED_PREBUILT = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/"
    "xauusd_m5_CANONICAL_V3_BASE28_AUGMENTED_2020_2026.parquet"
)
BASE28_MANIFEST_PATH = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/BASE28_CANONICAL/CURRENT_MANIFEST.json"
)


def _resolve_base28_parquet() -> Path:
    """Read the manifest to find the current authoritative BASE28 parquet path."""
    import json
    m = json.loads(BASE28_MANIFEST_PATH.read_text())
    return Path(m["parquet_path"])


@dataclass
class PrebuiltStateLoader:
    canonical_v3_path: Path = CANONICAL_V3_PREBUILT
    base28_path: Path | None = None    # resolved from manifest if None
    _cv3: pd.DataFrame | None = field(default=None, init=False)
    _base28: pd.DataFrame | None = field(default=None, init=False)
    _last_ts: pd.Timestamp | None = field(default=None, init=False)
    # mtime tracking for incremental-updater hot-reload
    _cv3_mtime: float = field(default=0.0, init=False)
    _base28_mtime: float = field(default=0.0, init=False)

    def refresh_if_changed(self) -> bool:
        """Reload prebuilts if their disk mtime has advanced since last load.
        Designed to be called every M1 poll by the live runner — the incremental
        updater writes new bars to disk, and this method detects the change and
        hot-reloads in-memory state. Returns True if any prebuilt was reloaded.
        """
        reloaded = False
        try:
            cv3_mt = self.canonical_v3_path.stat().st_mtime
            if cv3_mt > self._cv3_mtime + 0.01:   # 10ms tolerance for fs precision
                old_cutoff = self._cv3.index[-1] if self._cv3 is not None else None
                self._cv3 = pd.read_parquet(self.canonical_v3_path)
                if "time" in self._cv3.columns:
                    self._cv3["time"] = pd.to_datetime(self._cv3["time"], utc=True)
                    self._cv3 = self._cv3.set_index("time")
                self._cv3 = self._cv3.sort_index()
                self._cv3_mtime = cv3_mt
                new_cutoff = self._cv3.index[-1]
                if old_cutoff is None or new_cutoff > old_cutoff:
                    LOG.info(f"canonical_v3 reloaded: cutoff {old_cutoff} → {new_cutoff}")
                reloaded = True
        except Exception as exc:
            LOG.warning(f"canonical_v3 reload failed: {exc}")

        if self._base28 is not None and self.base28_path is not None:
            try:
                # Refresh-daemon rotates BASE28 to a NEW path every cycle; consult
                # the manifest so we hot-swap when the authoritative path changes.
                try:
                    current_path = _resolve_base28_parquet()
                except Exception:
                    current_path = self.base28_path
                if current_path != self.base28_path:
                    self.base28_path = current_path
                    self._base28_mtime = 0.0  # force reload below
                b28_mt = self.base28_path.stat().st_mtime
                if b28_mt > self._base28_mtime + 0.01:
                    self._base28 = pd.read_parquet(self.base28_path)
                    if not isinstance(self._base28.index, pd.DatetimeIndex):
                        if "time" in self._base28.columns:
                            self._base28["time"] = pd.to_datetime(self._base28["time"], utc=True)
                            self._base28 = self._base28.set_index("time")
                    self._base28 = self._base28.sort_index()
                    self._base28_mtime = b28_mt
                    reloaded = True
            except Exception as exc:
                LOG.warning(f"BASE28 reload failed: {exc}")

        if reloaded and self._cv3 is not None:
            # Update joint cutoff
            if self._base28 is not None:
                self._last_ts = min(self._cv3.index[-1], self._base28.index[-1])
            else:
                self._last_ts = self._cv3.index[-1]
            # V12.2: refresh multi-TF features if they were built
            if self._multi_tf_feats is not None:
                try:
                    self.build_multi_tf_features()
                except Exception as exc:
                    LOG.warning(f"multi-TF refresh failed: {exc} — keeping stale features")
        return reloaded

    def load(self) -> None:
        """Load prebuilt(s). Uses joined prebuilt if it exists (single file with
        all 92 XGB features), else falls back to canonical_v3 + BASE28 split."""

        if JOINED_PREBUILT.exists():
            LOG.info(f"loading JOINED prebuilt: {JOINED_PREBUILT.name}")
            cv3 = pd.read_parquet(JOINED_PREBUILT)
            if "time" in cv3.columns:
                cv3["time"] = pd.to_datetime(cv3["time"], utc=True)
                cv3 = cv3.set_index("time")
            if not isinstance(cv3.index, pd.DatetimeIndex):
                # Already DatetimeIndex from add_ctx_cont
                pass
            cv3 = cv3.sort_index()
            self._cv3 = cv3
            self._base28 = None
            self._last_ts = cv3.index[-1]
            self._cv3_mtime = JOINED_PREBUILT.stat().st_mtime
            LOG.info(f"  joined: {len(cv3):,} rows × {len(cv3.columns)} cols  "
                      f"range {cv3.index[0]} → {cv3.index[-1]}")
            return

        # Fallback: load both separately (legacy path before joined was produced)
        if self.base28_path is None:
            self.base28_path = _resolve_base28_parquet()

        LOG.info(f"loading canonical_v3 prebuilt: {self.canonical_v3_path.name}")
        cv3 = pd.read_parquet(self.canonical_v3_path)
        if "time" in cv3.columns:
            cv3["time"] = pd.to_datetime(cv3["time"], utc=True)
            cv3 = cv3.set_index("time")
        cv3 = cv3.sort_index()
        self._cv3 = cv3
        self._cv3_mtime = self.canonical_v3_path.stat().st_mtime
        LOG.info(f"  canonical_v3: {len(cv3):,} rows × {len(cv3.columns)} cols  "
                  f"range {cv3.index[0]} → {cv3.index[-1]}")

        LOG.info(f"loading BASE28 prebuilt: {self.base28_path.name}")
        base28 = pd.read_parquet(self.base28_path)
        if not isinstance(base28.index, pd.DatetimeIndex):
            if "time" in base28.columns:
                base28["time"] = pd.to_datetime(base28["time"], utc=True)
                base28 = base28.set_index("time")
        base28 = base28.sort_index()
        self._base28 = base28
        self._base28_mtime = self.base28_path.stat().st_mtime
        LOG.info(f"  BASE28: {len(base28):,} rows × {len(base28.columns)} cols  "
                  f"range {base28.index[0]} → {base28.index[-1]}")

        self._last_ts = min(cv3.index[-1], base28.index[-1])
        LOG.info(f"  effective live cutoff (min of both): {self._last_ts}")

    # ── public API ────────────────────────────────────────────────────

    @property
    def cutoff_ts(self) -> pd.Timestamp:
        """Latest M5 bar timestamp available in BOTH prebuilts (joint coverage)."""
        if self._last_ts is None:
            raise RuntimeError("loader not initialized — call .load() first")
        return self._last_ts

    def get_window(self, end_ts: pd.Timestamp, n_bars: int = 96) -> pd.DataFrame:
        """Return the last `n_bars` M5 bars up to and including `end_ts`,
        joined from canonical_v3 + BASE28 prebuilts.

        Latest row is the decision bar (at end_ts.floor('5min')).
        Empty DataFrame if `end_ts` is past the prebuilt coverage.
        """
        if self._cv3 is None or self._base28 is None:
            raise RuntimeError("loader not initialized — call .load() first")

        end_bucket = end_ts.floor("5min")
        if end_bucket > self.cutoff_ts:
            LOG.warning(f"decision_ts {end_bucket} is past prebuilt cutoff {self.cutoff_ts} — "
                         f"run canonical rebuild cron")
            return pd.DataFrame()

        cv3_win = self._cv3.loc[:end_bucket].tail(n_bars)
        if cv3_win.empty:
            return pd.DataFrame()

        # Joined-prebuilt path: all features already in cv3_win
        if self._base28 is None:
            return cv3_win

        # Fallback: join BASE28 augmented columns at exact timestamps
        b28_cols = [c for c in self._base28.columns if c not in cv3_win.columns]
        if b28_cols:
            b28_slice = self._base28.loc[:end_bucket, b28_cols].reindex(
                cv3_win.index, method=None,
            )
            joined = pd.concat([cv3_win, b28_slice], axis=1)
        else:
            joined = cv3_win.copy()
        return joined

    # ── Multi-TF (V12.2) ──────────────────────────────────────────────
    _multi_tf_feats: dict | None = field(default=None, init=False)
    _multi_tf_shift: dict | None = field(default=None, init=False)
    _multi_tf_feat_count: int = field(default=0, init=False)

    def build_multi_tf_features(self) -> None:
        """Build M5/M15/H1/H4/D1 per-bar feature tables from the loaded
        canonical_v3 prebuilt. Called once at runner startup (post-load),
        cached for live inference. Refresh hook below also updates them."""
        if self._cv3 is None:
            raise RuntimeError("call .load() before .build_multi_tf_features()")
        from gx1.features.htf_features import (
            build_multi_tf_per_bar_features, MULTI_TF_SHIFT, MULTI_TF_FEATURE_COUNT,
        )
        # Use canonical_v3's OHLC (it has open/high/low/close + time)
        ohlc_cols = ["open", "high", "low", "close"]
        missing = [c for c in ohlc_cols if c not in self._cv3.columns]
        if missing:
            raise RuntimeError(f"canonical_v3 missing OHLC cols: {missing}")
        m5_ohlc = self._cv3[ohlc_cols].copy()
        LOG.info(f"building multi-TF features (M5/M15/H1/H4/D1) from {len(m5_ohlc):,} M5 bars...")
        self._multi_tf_feats = build_multi_tf_per_bar_features(m5_ohlc)
        self._multi_tf_shift = MULTI_TF_SHIFT
        self._multi_tf_feat_count = int(MULTI_TF_FEATURE_COUNT)
        for tf, feats in self._multi_tf_feats.items():
            LOG.info(f"  multi-TF {tf}: {len(feats):,} bars × {feats.shape[1]} feats")

    def get_multi_tf_windows(self, end_ts: pd.Timestamp, n_bars: int = 96
                              ) -> dict[str, np.ndarray]:
        """Return {seq_m5, seq_m15, seq_h1, seq_h4, seq_d1} arrays at-or-before end_ts.
        Each is (n_bars, n_features) float32. Zero-padded at start if warmup unmet.

        Empty dict if multi-TF features aren't built (caller falls back to v3 path)."""
        if self._multi_tf_feats is None:
            return {}
        from gx1.features.htf_features import get_last_n_at_or_before
        return {
            f"seq_{tf.lower()}": get_last_n_at_or_before(
                feats, end_ts, n=n_bars, tf_shift=self._multi_tf_shift[tf]
            )
            for tf, feats in self._multi_tf_feats.items()
        }

    def get_latest_row(self, end_ts: pd.Timestamp) -> pd.Series | None:
        """Convenience: get only the M5-bucket row at end_ts (joined). None if missing."""
        win = self.get_window(end_ts, n_bars=1)
        if win.empty:
            return None
        return win.iloc[-1]
