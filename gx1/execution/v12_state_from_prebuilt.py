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
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ONE-TRUTH REGIME_V4 per-TF V2 mtf scalar projection (5-TF/9-feat). Shared with the
# canonical_incremental daemon + build so the live serve, the daemon-maintained BASE34,
# and the training BASE34 all project the SAME per-TF regime inputs (no train≠serve).
from gx1.features.htf_features import (
    REGIME_V4_V2_MTF_PER_TF,
    REGIME_V4_V2_MTF_TFS,
    REGIME_V4_V2_MTF_SKIP,
)

LOG = logging.getLogger("v12_state_from_prebuilt")


# ── Top-level subprocess workers for parallel augment (2026-06-01) ──
# These run in subprocesses spawned by _async_full_refresh so V2 mtf and
# GROUP-A — the two heavy augmenters — execute in true parallel rather than
# serializing behind the GIL. Each worker takes a pickled cv3, runs ONE
# augmenter via an ephemeral PrebuiltStateLoader instance, returns only the
# new columns (smaller payload back). On Linux fork() is used so cv3 is
# inherited via copy-on-write — actual pickle cost is only the result.
def _mp_v2_mtf_worker(cv3: pd.DataFrame) -> pd.DataFrame:
    """Run V2 multi-TF augmenter in a subprocess. Returns DataFrame with ONLY
    the new V2 mtf columns (suffix '_v2'), indexed identically to cv3 input.

    NB: the augmenter mutates cv3 in-place and returns the same object, so we
    must snapshot the column set BEFORE the call to compute the new-cols delta.
    """
    cols_before = set(cv3.columns)
    loader = PrebuiltStateLoader()
    augmented = loader._augment_cv3_with_v2_mtf_scalars(cv3)
    new_cols = [c for c in augmented.columns
                if c.endswith("_v2") and c not in cols_before]
    return augmented[new_cols].copy()


def _mp_group_a_worker(cv3: pd.DataFrame) -> pd.DataFrame:
    """Run GROUP-A + DIP/STRUCT augmenter in a subprocess. Returns DataFrame
    with ONLY the new ctx_cont columns, indexed identically to cv3 input.
    """
    from gx1.contracts.signal_bridge_v3 import (
        ORDERED_CTX_CONT_GROUP_A_PARITY as _GROUP_A,
        ORDERED_CTX_CONT_DIP_STRUCT as _DIP_STRUCT,
    )
    cols_before = set(cv3.columns)
    loader = PrebuiltStateLoader()
    augmented = loader._augment_cv3_with_group_a_and_dip_struct(cv3)
    new_cols = [c for c in (list(_GROUP_A) + list(_DIP_STRUCT))
                if c in augmented.columns and c not in cols_before]
    return augmented[new_cols].copy()

CANONICAL_V3_PREBUILT = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/"
    "xauusd_m5_CANONICAL_V3_2020_2026.parquet"
)
# NOTE: a "JOINED_PREBUILT" single-file fast-path (canonical_v3 + BASE28 ctx_cont/cat
# pre-merged) was REMOVED 2026-06-05 — it took an unguarded `.exists()` precedence over
# the manifest-resolved canonical_v3 + BASE28 split with NO freshness check, so a stale
# leftover reappearing on disk would silently poison serve/retrain (audit MISS-3 / R10
# coupling). The split path below is the ONE truth (BASE28 resolved fresh from the manifest).
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
    # 2026-06-01: async refresh — background thread runs the full 5-augmenter
    # pipeline on a LOCAL copy so the M1 exit-path never blocks on cv3 reload.
    # When the thread finishes, it atomically swaps self._cv3 / self._base28 /
    # self._last_ts in. Main loop reads them between calls (Python attribute
    # assignment is atomic under the GIL, so reader sees old or new — never half).
    _refresh_thread: threading.Thread | None = field(default=None, init=False)
    _refresh_enabled: bool = field(default=True, init=False)

    def refresh_if_changed(self) -> bool:
        """Non-blocking: if cv3 / BASE28 mtime advanced and no refresh is in
        progress, schedule a BACKGROUND thread to read + augment + atomic-swap.
        Returns True only on the cycle that starts the refresh (so the caller
        can log it); main loop ignores return value either way.

        Pre-2026-06-01 this was synchronous and blocked the M1 exit-path 3-7 min
        every M5 close. New behaviour: M1 polls continue serving exit decisions
        from the current (slightly stale) cv3 while the new augmented cv3 is
        built in the background. cv3 m5_phase staleness during refresh is
        acceptable for exit context (V3 + M1 is primary; cv3 features are
        secondary asof-fill). Entry decisions are per M5 — the 3-7 min refresh
        fits inside one M5 cycle.
        """
        # Quick check: is anything new on disk?
        try:
            cv3_mt = self.canonical_v3_path.stat().st_mtime
        except Exception:
            return False
        b28_path = self.base28_path
        b28_mt_disk = 0.0
        if b28_path is not None:
            try:
                # Manifest may rotate to a new path.
                try:
                    current_path = _resolve_base28_parquet()
                except Exception:
                    current_path = b28_path
                if current_path is not None:
                    b28_path = current_path
                    b28_mt_disk = b28_path.stat().st_mtime
            except Exception:
                b28_mt_disk = 0.0

        cv3_advanced = cv3_mt > self._cv3_mtime + 0.01
        b28_advanced = (b28_mt_disk > self._base28_mtime + 0.01) if (b28_path is not None) else False
        if not (cv3_advanced or b28_advanced):
            return False

        # If a refresh is already in flight, let it finish.
        if self._refresh_thread is not None and self._refresh_thread.is_alive():
            return False
        if not self._refresh_enabled:
            return False

        # Schedule background refresh.
        t = threading.Thread(
            target=self._async_full_refresh,
            args=(cv3_mt, b28_path, b28_mt_disk, cv3_advanced, b28_advanced),
            daemon=True,
            name="cv3_async_refresh",
        )
        self._refresh_thread = t
        t.start()
        return True

    def _async_full_refresh(self, cv3_mt: float, b28_path: Path | None,
                             b28_mt_disk: float, cv3_advanced: bool,
                             b28_advanced: bool) -> None:
        """Background-thread worker: read new cv3 / BASE28 from disk, run all
        5 augmenters on a LOCAL DataFrame, then atomically swap into self.

        Bit-parity with synchronous path is guaranteed because exactly the same
        augmenter methods are called — only the orchestration is threaded.
        """
        try:
            t_start = pd.Timestamp.utcnow()
            # --- cv3 ---
            new_cv3 = self._cv3
            if cv3_advanced:
                df = pd.read_parquet(self.canonical_v3_path)
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"], utc=True)
                    df = df.set_index("time")
                new_cv3 = df.sort_index()
            # --- base28 ---
            new_b28 = self._base28
            if b28_advanced and b28_path is not None:
                df = pd.read_parquet(b28_path)
                if not isinstance(df.index, pd.DatetimeIndex):
                    if "time" in df.columns:
                        df["time"] = pd.to_datetime(df["time"], utc=True)
                        df = df.set_index("time")
                new_b28 = df.sort_index()

            # --- augment local cv3 (does NOT touch self._cv3 yet) ---
            # 2026-06-01: PARALLEL via multiprocessing for V2 mtf + GROUP-A.
            # Threading first attempt gave 0% speedup (GIL serialized the two
            # heavy augmenters which spend their time in Python-level pandas
            # glue). Multiprocessing with fork() gives true parallelism — cv3
            # is inherited via copy-on-write (zero IPC for input), only the
            # new columns are pickled back (small payload). Falls back to
            # sequential on any multiprocessing error so the runner stays up.
            if new_cv3 is not None and cv3_advanced:
                t_aug = pd.Timestamp.utcnow()
                # Fast augmenters in main thread (~3s total)
                new_cv3 = self._augment_cv3_with_volume_features(new_cv3)
                new_cv3 = self._augment_cv3_with_v1_legacy(new_cv3)

                # Heavy augmenters in 2 subprocesses (~95s wall, was 180s sequential)
                used_mp = False
                try:
                    import multiprocessing as _mp
                    from concurrent.futures import ProcessPoolExecutor
                    fork_ctx = _mp.get_context("fork")
                    with ProcessPoolExecutor(max_workers=2, mp_context=fork_ctx) as pool:
                        f_mtf = pool.submit(_mp_v2_mtf_worker, new_cv3)
                        f_grp = pool.submit(_mp_group_a_worker, new_cv3)
                        mtf_new = f_mtf.result(timeout=600)
                        grp_new = f_grp.result(timeout=600)
                    # Merge new cols back
                    for col in mtf_new.columns:
                        if col not in new_cv3.columns:
                            new_cv3[col] = mtf_new[col].values
                    for col in grp_new.columns:
                        if col not in new_cv3.columns:
                            new_cv3[col] = grp_new[col].values
                    # V1/R10 completion (2026-06-04): the parallel-augment SUCCESS path was missing the
                    # regime_v4 augmenter (only the sync load paths + the sequential fallback had it). At
                    # GX1_REGIME_V4=1 the live async hot-refresh would drop the 16 EXIT_IO_V8 regime cols ->
                    # v12_v3_live.build_window fail-closed guard crashes the regime exit serve on first refresh.
                    # Runs AFTER the mtf merge above (supplies the 12 {tf}_*_v2 sources); joins D1_dist + is a
                    # no-op at flag=0 (cement bit-identical).
                    new_cv3 = self._augment_cv3_with_regime_v4(new_cv3)
                    used_mp = True
                except Exception as exc:
                    LOG.warning(f"[parallel-augment] subprocess augment failed "
                                f"({exc}); falling back to sequential")
                    new_cv3 = self._augment_cv3_with_v2_mtf_scalars(new_cv3)
                    new_cv3 = self._augment_cv3_with_group_a_and_dip_struct(new_cv3)
                    new_cv3 = self._augment_cv3_with_regime_v4(new_cv3)  # V1/R10: after v2_mtf (sources)

                aug_took = (pd.Timestamp.utcnow() - t_aug).total_seconds()
                LOG.info(f"[parallel-augment] full pipeline finished in {aug_took:.1f}s "
                         f"(method={'multiprocessing' if used_mp else 'sequential-fallback'})")

            # Joint cutoff
            if new_cv3 is not None and new_b28 is not None:
                new_last_ts = min(new_cv3.index[-1], new_b28.index[-1])
            elif new_cv3 is not None:
                new_last_ts = new_cv3.index[-1]
            else:
                new_last_ts = self._last_ts

            # --- build multi_tf BEFORE the swap so they stay in lock-step ---
            # 2026-06-01 CRITICAL fix: previously we did atomic swap of self._cv3
            # then called build_multi_tf_features() which mutates self._multi_tf_feats
            # separately. Between those two writes, a reader saw new cv3 with OLD
            # multi_tf — silent V10 feature mismatch. Now we build mtf on the LOCAL
            # new_cv3 and swap cv3 + mtf together (single sequential write block
            # under the GIL).
            new_mtf_bundle = None
            if self._multi_tf_feats is not None and new_cv3 is not None:
                try:
                    new_mtf_bundle = self.build_multi_tf_features(new_cv3)
                except Exception as exc:
                    LOG.warning(f"multi-TF refresh failed: {exc} — keeping stale")
                    new_mtf_bundle = None

            # --- atomic swap (single attribute assignments are GIL-atomic) ---
            old_cutoff = self._cv3.index[-1] if self._cv3 is not None else None
            self._cv3 = new_cv3
            if cv3_advanced:
                self._cv3_mtime = cv3_mt
            if b28_advanced and b28_path is not None:
                self._base28 = new_b28
                self.base28_path = b28_path
                self._base28_mtime = b28_mt_disk
            self._last_ts = new_last_ts
            if new_mtf_bundle is not None:
                self._multi_tf_feats = new_mtf_bundle["feats"]
                self._multi_tf_shift = new_mtf_bundle["shift"]
                self._multi_tf_feat_count = new_mtf_bundle["feat_count"]

            took_sec = (pd.Timestamp.utcnow() - t_start).total_seconds()
            new_cutoff = self._cv3.index[-1] if self._cv3 is not None else None
            LOG.info(f"[async-refresh] cv3 cutoff {old_cutoff} → {new_cutoff} "
                     f"(took {took_sec:.1f}s, main loop never blocked)")
        except Exception as exc:
            LOG.warning(f"async cv3 refresh failed: {exc}")

    def load(self) -> None:
        """Load the canonical_v3 + BASE28 prebuilts (the ONE-truth split path).
        BASE28 is resolved FRESH from its manifest; the 5 augmenters then run on
        cv3. (The unguarded JOINED single-file fast-path was removed 2026-06-05 —
        see the note at BASE28_MANIFEST_PATH.)"""

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
        self._augment_cv3_with_volume_features()
        self._augment_cv3_with_v2_mtf_scalars()
        self._augment_cv3_with_group_a_and_dip_struct()
        self._augment_cv3_with_v1_legacy()
        self._augment_cv3_with_regime_v4()  # V1/R10: after v2_mtf (sources); gated GX1_REGIME_V4

    # ── V2 multi-TF scalar augmentation (XGB v7 base80) ───────────────
    # XGB v7 expects 31 V2-suffixed columns per row (M15/H1/H4/D1 × 7-8 each).
    # ONE TRUTH (2026-06-13): the per-TF V2 mtf projection now lives in
    # gx1.features.htf_features.REGIME_V4_V2_MTF_* and is imported at module top, so the serve,
    # the canonical_incremental daemon (BASE34 ctx recompute), and the build all use the IDENTICAL
    # 5-TF/9-feat regime projection. (m5 ADDED 2026-06-05 — regime ALL-5; trend_age_bars_norm = R2
    # for gx1.features.regime_v4_features.) Was inline here; promoted to kill the duplication.
    _V2_MTF_PER_TF = REGIME_V4_V2_MTF_PER_TF
    _V2_MTF_TFS = REGIME_V4_V2_MTF_TFS
    _V2_MTF_SKIP = REGIME_V4_V2_MTF_SKIP

    # ── V1 / R10 (2026-06-04): REGIME_V4 exit-context augmentation ────────
    def _augment_cv3_with_regime_v4(self, cv3: pd.DataFrame | None = None) -> pd.DataFrame | None:
        """Compute the 16 REGIME_V4 EXIT_IO_V8 features on the FULL cv3 — ONE-truth add_regime_v4_features,
        the SAME call the V3 builder uses (materialize_build_v3_training_dataset_v2.py:330-340). Gated by
        GX1_REGIME_V4 so cement (flag=0 / EXIT_IO_V7) stays bit-identical. MUST run AFTER
        _augment_cv3_with_v2_mtf_scalars (supplies the 12 {tf}_*_v2 source cols) and on the FULL frame
        (F4 d1_dist_roc_288 / F9 bars_since_d1_regime_change need >=288-bar D1 history; a 96-bar window
        would clip them to 0 = a second-order skew). Fail-closed: raises on a missing source col.
        """
        if os.environ.get("GX1_REGIME_V4", "0").strip().lower() not in ("1", "true", "yes", "on"):
            return cv3 if cv3 is not None else self._cv3
        in_place = cv3 is None
        target = cv3 if cv3 is not None else self._cv3
        if target is None:
            return None
        from gx1.features.regime_v4_features import (
            REGIME_V4_FEATURE_NAMES as _RV4_NAMES,
            add_regime_v4_features as _add_rv4,
        )
        if all(c in target.columns for c in _RV4_NAMES):
            return target
        # D1_dist_from_ema200_atr is the ONLY REGIME_V4 source not emitted by _augment_cv3_with_v2_mtf_scalars
        # (which supplies the 12 {tf}_*_v2). Join it from base28 (capital-D1) by timestamp before computing.
        if "D1_dist_from_ema200_atr" not in target.columns:
            if self._base28 is not None and "D1_dist_from_ema200_atr" in self._base28.columns:
                target["D1_dist_from_ema200_atr"] = (
                    self._base28["D1_dist_from_ema200_atr"].reindex(target.index, method="ffill")
                )
            else:
                raise RuntimeError(
                    "[REGIME_V4_SERVE] D1_dist_from_ema200_atr missing from base28 — cannot compute the "
                    "EXIT_IO_V8 REGIME_V4 tail at serve (would be train!=serve). Provide it on the prebuilt."
                )
        # add_regime_v4_features needs time-ascending order (shift/run-length); cv3 is sorted at load.
        _add_rv4(target)  # in-place; one-truth; raises on any missing {tf}_*_v2 source col
        if in_place:
            self._cv3 = target
        return target

    def _augment_cv3_with_group_a_and_dip_struct(self, cv3: pd.DataFrame | None = None) -> pd.DataFrame | None:
        """Add 24 GROUP-A parity + 36 DIP/STRUCT ctx_cont columns to cv3
        (defaults to self._cv3). Returns the augmented DataFrame.

        One-truth: re-uses `attach_group_a_dip_struct_ctx_columns` from
        `gx1.scripts.augment_forward_outcome_v2` — same function the V10 builder
        + V3 exit builder + inference-batch candidate gen use. Train/serve skew
        impossible by construction.

        Cost: ~10-20 min at first load (456K cv3 rows × per-bar augment_candidate).
        Cached in cv3 thereafter; re-runs only on cv3 reload (refresh_if_changed).
        """
        in_place = cv3 is None
        target = cv3 if cv3 is not None else self._cv3
        if target is None:
            return None
        from gx1.contracts.signal_bridge_v3 import (
            ORDERED_CTX_CONT_GROUP_A_PARITY as _GROUP_A,
            ORDERED_CTX_CONT_DIP_STRUCT as _DIP_STRUCT,
        )
        need = list(_GROUP_A) + list(_DIP_STRUCT)
        if all(c in target.columns for c in need):
            return target
        if any(c not in target.columns for c in ("high", "low", "close")):
            LOG.warning("group_a/dip_struct augment skipped: cv3 missing OHLC")
            return target
        LOG.info(f"augmenting canonical_v3 with {len(_GROUP_A)} GROUP-A + "
                 f"{len(_DIP_STRUCT)} DIP/STRUCT ctx_cont cols "
                 f"(~10-20 min for {len(target):,} M5 rows)...")
        from gx1.scripts.augment_forward_outcome_v2 import (
            attach_group_a_dip_struct_ctx_columns,
        )
        from gx1.features.htf_features import build_multi_tf_per_bar_features_v2
        # LIVE serve: build the multi-TF bundle IN-MEMORY from THIS cv3 and pass it, instead
        # of reading the on-disk MULTI_TF_V2_CACHE. The disk cache is only as fresh as its
        # last rebuild — nothing in the live loop advances it, so it eventually trips the
        # 2-day stale guard (live breaks) or, worse, would serve stale HTF context. Building
        # from the live cv3 makes the group-A/dip-struct ctx ALWAYS current and removes the
        # last live disk-cache dependency. Same builder + same bars as the disk cache (which
        # prebuild_multi_tf_cache_v2 also makes via build_multi_tf_per_bar_features_v2), and
        # the features are causal/asof → bit-identical at every decision ts (train==serve).
        try:
            mtf_in_mem = build_multi_tf_per_bar_features_v2(target)
        except Exception as exc:
            LOG.warning(f"group_a/dip_struct: in-memory multi-TF build failed ({exc}); "
                        f"falling back to on-disk cache")
            mtf_in_mem = None
        # The helper mutates in place via the returned DataFrame; rebind to be safe.
        target = attach_group_a_dip_struct_ctx_columns(
            target, journal_label="live_costfix", multi_tf=mtf_in_mem,
        )
        if in_place:
            self._cv3 = target
        LOG.info(f"  group_a/dip_struct augment done: cv3 now {len(target.columns)} cols")
        return target

    # ── Legacy canonical_v1 features Entry-IQL trained on ────────────────
    _V1_LEGACY_FEATURES = (
        "atr",                  # M5 ATR14 (price units)
        "std50",                # M5 50-bar close-return std
        "roc20",                # M5 20-bar rate-of-change
        "_v1_vwap_drift48",     # M5 48-bar VWAP drift fraction
        "_v1h1_vwap_drift",     # H1 VWAP drift fraction (last H1 bar)
    )

    def _augment_cv3_with_v1_legacy(self, cv3: pd.DataFrame | None = None) -> pd.DataFrame | None:
        """Compute 5 legacy canonical_v1 features Entry-IQL was trained on but
        which the canonical_v3 prebuilt does not carry. Same compute as the
        v1 canonical augmenter — values match training within FP noise.
        Idempotent.
        """
        target = cv3 if cv3 is not None else self._cv3
        if target is None:
            return None
        cv3 = target
        if all(c in cv3.columns for c in self._V1_LEGACY_FEATURES):
            return cv3
        if any(c not in cv3.columns for c in ("open", "high", "low", "close", "volume")):
            LOG.warning("v1 legacy augment skipped: cv3 lacks full OHLCV")
            return cv3
        LOG.info(f"augmenting canonical_v3 with {len(self._V1_LEGACY_FEATURES)} "
                 f"legacy canonical_v1 features (atr/std50/roc20/vwap_drifts)")

        close = cv3["close"].astype(np.float64)
        high = cv3["high"].astype(np.float64)
        low = cv3["low"].astype(np.float64)
        volume = cv3["volume"].astype(np.float64).clip(lower=0.0)
        # Tiny-volume safety: where volume is zero, fall back to a constant 1.0
        # so VWAP becomes a simple mean (the v1 canonical pipeline did the same).
        vol_safe = volume.where(volume > 0.0, 1.0)

        # ATR14 (Wilder-smoothed TR over high/low/prev_close).
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        cv3["atr"] = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean().fillna(method="bfill").astype(np.float32)

        # E2 (2026-06-04 train==serve parity): std50 / _v1_vwap_drift48 / _v1h1_vwap_drift MUST equal the
        # BUILD truth (augment_canonical_v3_with_missing_features == v12_canonical_incremental._compute_plus5).
        # The old serve math diverged (std50 min_periods 10 vs 2; vwap48 /close vs /vwap48 + min_periods 12
        # vs 1; h1 calendar-resample vs rolling-288) -> a retrain on the PLUS5 parquet would learn values
        # serve could not reproduce. Byte-aligned to the build formulas below (atr + roc20 already matched).
        rets = close.pct_change()
        cv3["std50"] = rets.rolling(50, min_periods=2).std().fillna(0.0).astype(np.float32)

        # 20-bar rate-of-change (close[t] / close[t-20] - 1).
        cv3["roc20"] = close.pct_change(20).fillna(0.0).astype(np.float32)

        # _v1_vwap_drift48: M5 48-period VWAP drift = (close - vwap48) / vwap48 (build def, min_periods=1).
        pv48 = (close * vol_safe).rolling(48, min_periods=1).sum()
        vv48 = vol_safe.rolling(48, min_periods=1).sum()
        vwap48 = pv48 / vv48.replace(0, 1.0)
        cv3["_v1_vwap_drift48"] = ((close - vwap48) / vwap48.replace(0, 1.0)).fillna(0.0).astype(np.float32)

        # _v1h1_vwap_drift: H1 VWAP drift = (close - vwap_288) / vwap_288, rolling 288 M5 (=24 H1),
        # min_periods=12 (build def — NOT a calendar resample). fillna(0) only touches the <12-bar warmup
        # (training-excluded, never hit live with a full window) -> decision-parity with build.
        pv_h1 = (close * vol_safe).rolling(288, min_periods=12).sum()
        v_h1 = vol_safe.rolling(288, min_periods=12).sum()
        vwap_h1 = pv_h1 / v_h1.replace(0, 1.0)
        cv3["_v1h1_vwap_drift"] = ((close - vwap_h1) / vwap_h1.replace(0, 1.0)).fillna(0.0).astype(np.float32)

        LOG.info(f"  v1 legacy augment done: cv3 now {len(cv3.columns)} cols")
        return cv3

    def _augment_cv3_with_volume_features(self, cv3: pd.DataFrame | None = None) -> pd.DataFrame | None:
        """Add 4 volume features (VOLUME_FEATURE_NAMES) to cv3 (defaults to self._cv3).
        Pure-ish: mutates the cv3 you pass in (or self._cv3 by default) and returns it.
        Idempotent — if all are present, skip.
        """
        target = cv3 if cv3 is not None else self._cv3
        if target is None:
            return None
        from gx1.features.volume_features import (
            compute_volume_features, VOLUME_FEATURE_NAMES,
        )
        if all(c in target.columns for c in VOLUME_FEATURE_NAMES):
            return target
        if "volume" not in target.columns or "close" not in target.columns:
            LOG.warning("volume features augment skipped: cv3 lacks volume/close")
            return target
        LOG.info(f"augmenting canonical_v3 with {len(VOLUME_FEATURE_NAMES)} "
                 f"volume features (vol_z_20, vol_ratio_5_20, ...)")
        feats = compute_volume_features(target[["volume", "close"]])
        for name in VOLUME_FEATURE_NAMES:
            if name not in target.columns and name in feats:
                target[name] = feats[name]
        LOG.info(f"  volume features added: cv3 now {len(target.columns)} cols")
        return target

    def _augment_cv3_with_v2_mtf_scalars(self, cv3: pd.DataFrame | None = None) -> pd.DataFrame | None:
        """Add 31 V2 multi-TF scalar columns to cv3 (defaults to self._cv3).
        Returns the augmented DataFrame. Idempotent — if columns are already
        present (e.g. from a rebuilt cv3 parquet), skip.
        """
        target = cv3 if cv3 is not None else self._cv3
        if target is None:
            return None
        # Idempotency check — any expected col present means we've augmented.
        probe = f"{self._V2_MTF_TFS[0]}_{self._V2_MTF_PER_TF[0][0]}_v2"
        if probe in target.columns:
            return target
        from gx1.features.htf_features import attach_v2_mtf_per_bar_scalars
        cv3 = target
        ohlc = ["open", "high", "low", "close"]
        if any(c not in cv3.columns for c in ohlc):
            LOG.warning("V2 mtf augment skipped: canonical_v3 missing OHLC cols")
            return cv3
        m5_df = cv3[ohlc].copy()
        for c in ohlc:
            m5_df[c] = m5_df[c].astype(np.float64)
        m5_df["volume"] = cv3["volume"].astype(np.float64) if "volume" in cv3.columns else 1.0
        LOG.info(f"augmenting canonical_v3 with V2 multi-TF scalar features "
                 f"(31 cols expected) — {len(m5_df):,} M5 rows...")
        # V2 (2026-06-04 one-truth): per-bar projection now lives in the SHARED
        # htf_features.attach_v2_mtf_per_bar_scalars (build_multi_tf_per_bar_features_v2 +
        # MULTI_TF_SHIFT only-closed-bars searchsorted) so the V3 exit builder produces the
        # IDENTICAL {tf}_*_v2 cols (was an inline loop here -> the builder's stale join drifted to
        # 88-95% match). Byte-identical to the prior loop.
        try:
            cv3_ts_ns = cv3.index.values.astype("datetime64[ns]").astype(np.int64)
            _v2cols = attach_v2_mtf_per_bar_scalars(
                m5_df, cv3_ts_ns, self._V2_MTF_PER_TF, self._V2_MTF_TFS, self._V2_MTF_SKIP,
            )
        except Exception as exc:
            LOG.warning(f"V2 mtf build failed: {exc} — XGB feature gap will surface")
            return cv3
        for _col, _vals in _v2cols.items():
            cv3[_col] = _vals
        LOG.info(f"  V2 mtf augment done: +{len(_v2cols)} cols  "
                 f"(cv3 now {len(cv3.columns)} cols total)")
        return cv3

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
            # 2026-06-15 weekend-open boundary fix: the FIRST M5 bar after a market gap
            # (e.g. Sun 22:00) exists in cv3, but its :00 minute is missing from base28
            # (OANDA's first post-open M1 bar lands at :04), so an exact reindex(method=
            # None) left ALL base28 cols NaN on that ONE context bar -> the fail-closed
            # XGB sanitizer (and session_id.astype(int)) refused EVERY decision for the
            # ~8h that bar stays in the 96-bar window = live fully blocked at every Sunday
            # open. Fill ONLY that gap from the next M1 bar WITHIN THE SAME M5 bucket
            # (tolerance=5min). Verified: exact-matched bars (= every bar training saw)
            # are returned BIT-IDENTICALLY (train==serve preserved); the cap at end_bucket
            # prevents any future leak; the decision bar itself is always present in
            # base28 (cutoff = min(cv3,base28)) so it is never fabricated.
            b28_slice = self._base28.loc[:end_bucket, b28_cols].reindex(
                cv3_win.index, method="bfill", tolerance=pd.Timedelta("5min"),
            )
            joined = pd.concat([cv3_win, b28_slice], axis=1)
        else:
            joined = cv3_win.copy()
        return joined

    # ── Multi-TF (V12.2) ──────────────────────────────────────────────
    _multi_tf_feats: dict | None = field(default=None, init=False)
    _multi_tf_shift: dict | None = field(default=None, init=False)
    _multi_tf_feat_count: int = field(default=0, init=False)

    def build_multi_tf_features(self, cv3: pd.DataFrame | None = None) -> dict | None:
        """Build M5/M15/H1/H4/D1 per-bar feature tables from cv3 (defaults to
        self._cv3). Returns a dict {feats, shift, feat_count} that the caller
        can swap atomically into self.* — required by _async_full_refresh so
        the multi_tf cache is never inconsistent with the cv3 it was built from.

        Called once at runner startup (post-load) AND on every async refresh
        cycle. When `cv3` is None (legacy callers) the result is also assigned
        back to self.* for backward-compat.

        2026-05-29: switched to V2 helper (25 features per TF, includes the
        feature set the COSTFIX V10 + V3 v7 multi-TF were trained on). V1
        helper (17 features per TF) was retired with V12.2 cement.
        2026-06-01: refactored to return dict so async refresh can atomic-swap
        cv3 + multi_tf together (fixes CRITICAL race that produced silent V10
        feature mismatch during cv3 swap window).
        """
        source = cv3 if cv3 is not None else self._cv3
        if source is None:
            raise RuntimeError("call .load() before .build_multi_tf_features()")
        from gx1.features.htf_features import (
            build_multi_tf_per_bar_features_v2, MULTI_TF_SHIFT,
        )
        ohlc_cols = ["open", "high", "low", "close"]
        missing = [c for c in ohlc_cols if c not in source.columns]
        if missing:
            raise RuntimeError(f"canonical_v3 missing OHLC cols: {missing}")
        m5_df = source[ohlc_cols].copy()
        if "volume" in source.columns:
            m5_df["volume"] = source["volume"].astype(np.float64)
        else:
            m5_df["volume"] = 1.0
        LOG.info(f"building multi-TF V2 features (M5/M15/H1/H4/D1, 25 dim each) "
                 f"from {len(m5_df):,} M5 bars...")
        feats_dict = build_multi_tf_per_bar_features_v2(m5_df)
        feat_count = 25
        for tf, feats in feats_dict.items():
            if len(feats) > 0:
                feat_count = int(feats.shape[1])
                break
        for tf, feats in feats_dict.items():
            LOG.info(f"  multi-TF V2 {tf}: {len(feats):,} bars × {feats.shape[1]} feats")
        result = {
            "feats": feats_dict,
            "shift": MULTI_TF_SHIFT,
            "feat_count": feat_count,
        }
        # Backward-compat: legacy callers (load() and direct user calls without
        # cv3 arg) expect side-effect on self.*. When cv3 IS provided the caller
        # is responsible for the atomic swap.
        if cv3 is None:
            self._multi_tf_feats = result["feats"]
            self._multi_tf_shift = result["shift"]
            self._multi_tf_feat_count = result["feat_count"]
        return result

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
