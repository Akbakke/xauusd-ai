#!/usr/bin/env python3
"""LIVE smart520 entry-state builder — serving-wave gap 2 (vedtak SMART_JOINT_POLICY_PROMOTION_20260708).

Builds the per-M5-bar 520-dim smart_seq520 entry state — seq (96, 520) + snap (520)
+ ctx_cont (142) + ctx_cat (5) — from the LIVE cv3/BASE28 prebuilts, bit-parity-exact
with the offline dataset builder that produced the promoted cand#4 bundle's inputs
(gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py, julyext dataset
v10_dataset_smart_candidate_julyext_20260705, TEST-split convention).

Extend-don't-fork note (CLAUDE.md rule 7): the existing live entry state builder
(v12_v10_live.V10LiveInference._build_seq_matrix / _build_ctx_cont) implements the
RETIRED legacy 41-dim contract; the smart 520-dim state does not exist anywhere in
serve — this module is the genuinely-new shared one-truth helper. Every formula is
IMPORTED from the offline one-truth builders; nothing is re-derived here:

  - neutral XGB bridge (7)      -> gx1.xgb.multihead.xgb_multihead_model_v1.proba_to_signal_bridge_v1
  - group-A + dip/struct (60)   -> gx1.scripts.augment_forward_outcome_v2.attach_group_a_dip_struct_ctx_columns
  - volume features (4)         -> gx1.features.volume_features.add_volume_features
  - entry smart ctx (19)        -> gx1.features.entry_smart_context.add_entry_smart_context_features
  - 479 extension signals       -> gx1.scripts.build_entry_v10_ctx_training_dataset_v3._build_inline_seq_structure_extension
  - all other source columns    -> live cv3+BASE28 prebuilts (PrebuiltStateLoader.get_window),
                                   maintained full-history by the canonical_incremental daemon
                                   with the SAME one-truth augmenters the offline chain used
                                   (v12_ctx_augment_live mirrors add_ctx_cont_columns_to_prebuilt).

TRAIN==SERVE FRAME CONVENTION (critical — read before touching):
The offline dataset builder was invoked PER SPLIT with a start/end time filter
(build_entry_v10_ctx_training_dataset_v3.main, --time_split), so every
frame-window-dependent feature in the promoted evidence (test split 2026-01-01 ..
2026-07-03) was computed on a frame ANCHORED at the split start:
  * group-A vol_pct_m5_1yr / vol_pct_h1_1yr: the 365d percentile loop in
    _build_atr_percentile_array never runs on a frame shorter than the window
    -> the ENTIRE promoted test evidence carries the 0.5 neutral value. A live
    full-history computation would emit REAL percentiles = a silent train/serve
    skew vs the replay-proven policy.
  * group-A liquidity/dip lookbacks + struct_smc_swing_x_dip_v3 frame-max
    normalization + volume/entry-smart rolling windows: all clip at the frame
    start.
This builder therefore reproduces that convention EXACTLY: the state frame is the
live joined cv3+BASE28 window from SMART520_STATE_FRAME_ANCHOR_UTC (== the julyext
dataset --test_start) to the decision bar, and the frame-dependent families
(group-A/dip-struct, volume, entry-smart, extension layers) are recomputed on that
anchored frame — overwriting the loader's full-history in-memory values. Columns
whose offline values came from the FULL-history FULL_PLUS_CTX source (micro, swing,
session, HTF ctx, REGIME_V4, per-bar price-state) are taken from the live prebuilts
as-is (both sides full-history + converged; proven by the parity gate).

The anchor is a PROMOTION-COUPLED constant: it must be re-pinned in the same wave
as any future contract flip whose evidence dataset uses a different split anchor.
Parity proof: gx1/scripts/verify_smart520_serve_parity_v1.py (mandatory gate).
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from gx1.contracts.signal_bridge_v3 import (
    ORDERED_BRIDGE_FIELDS_V3,
    ORDERED_CTX_CAT_NAMES_V3,
    ORDERED_CTX_CONT_DIP_STRUCT,
    ORDERED_CTX_CONT_ENTRY_SMART_DERIVED,
    ORDERED_CTX_CONT_GROUP_A_PARITY,
    ORDERED_CTX_CONT_NAMES_V3,
    ORDERED_SEQ_FIELDS_V3,
)
from gx1.features.volume_features import VOLUME_FEATURE_NAMES

LOG = logging.getLogger("v12_smart520_state_live")

# ONE-TRUTH frame anchor = the offline dataset's TEST-split start (see module
# docstring). Source of the value: julyext dataset build command --test_start
# (v10_dataset_smart_candidate_julyext_20260705/*_test.manifest.json) — the split
# whose predictions locked the promoted operating point (RUN_MANIFEST
# joint_smart_policy_replay_20260708). PROMOTION-COUPLED: re-pin on contract flip.
SMART520_STATE_FRAME_ANCHOR_UTC = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")

# Model-range start of the offline FULL_PLUS_CTX frame (v10_6yr_rebuild_20260626.sh
# STAGE3 trims cv3 to >= 2020-11-09 before add_ctx_cont). The ctx_cat bucket
# ranks (vol_regime_id/atr_bucket/spread_bucket) are FRAME-GLOBAL percentile
# ranks over THAT frame — reproduced live over [MODEL_RANGE_START, RANK_REF_END]
# with a FROZEN-reference digitize for bars after RANK_REF_END.
SMART520_MODEL_RANGE_START_UTC = pd.Timestamp("2020-11-09 00:00:00", tz="UTC")

# End of the offline FULL_PLUS_CTX frame (julyext build cutoff == test_end).
# Bucket ranks are computed over EXACTLY [MODEL_RANGE_START, RANK_REF_END] (the
# offline frame) so historical buckets match the promoted evidence BIT-EXACTLY;
# bars AFTER this get percentile-vs-frozen-reference digitize (deterministic,
# no rank drift as live data accumulates). PROMOTION-COUPLED like the anchor.
SMART520_RANK_REFERENCE_END_UTC = pd.Timestamp("2026-07-03 16:55:00", tz="UTC")

SEQ_LEN_SMART520 = 96
SIGNAL_DIM_SMART520 = 520
CTX_CONT_DIM_SMART520 = len(ORDERED_CTX_CONT_NAMES_V3)   # 142
CTX_CAT_DIM_SMART520 = len(ORDERED_CTX_CAT_NAMES_V3)     # 5

# Columns the temp source-parquet must carry for the extension's price/candle
# layers (they read by NAME from the parquet):
#   _build_price_derived_layer  -> time + mid|close + atr|atr50|_v1_atr14
#   _build_candlestick_*        -> time/open/high/low/close
_SOURCE_PARQUET_COLS = ["time", "mid", "close", "atr", "open", "high", "low"]


def _compute_bare_atr14(frame: pd.DataFrame) -> np.ndarray:
    """The FULL_PLUS_CTX 'atr' column (price-unit TR rolling-14 mean), which the
    extension price layer prefers over _v1_atr14. Mirror of the one-truth v1
    canonical-features formula (gx1/scripts/materialize_build_canonical_features_v1.py:136-141)
    — 14-bar bounded lookback, so anchored-frame values == full-history values for
    every consumable row (targets sit >= 95 bars into the frame). The live cv3
    prebuilt does not carry 'atr' (only atr50/_v1_atr14), so it is derived here.
    """
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    high_low = high - low
    high_pc = (high - close.shift(1)).abs()
    low_pc = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=1).mean().astype(np.float32).to_numpy()


def _compute_offline_swing_block(close: pd.Series, high: pd.Series, low: pd.Series) -> dict[str, np.ndarray]:
    """The 5 swing ctx features with the OFFLINE dataset's EXACT convention —
    a vectorized bit-parity mirror of the add_ctx_cont swing block
    (gx1/scripts/add_ctx_cont_columns_to_prebuilt.py:571-637, ctx_cont_dim==16):
    2-2 fractal pivots stamped AT the pivot bar using shift(-1)/shift(-2)
    (skipna at the frame tail), running last-pivot state, TR-rolling-14 ATR
    (min_periods=1) normalization.

    NOT compute_swing_structure_features (confirmation-LAG convention): the
    promoted smart dataset's swing values verifiably came from the add_ctx_cont
    block (bit-exact reproduction checked 2026-07-08 on 2026 Q1; the parity gate
    guards this permanently). The offline convention peeks 2 bars ahead for
    pivot confirmation, so at the LIVE frame tail the last <=2 bars' pivots are
    evaluated on the available bars only — the same skipna semantics the
    offline builder itself applied at ITS frame tail. This mirror exists here
    (instead of importing) because add_ctx_cont's block is inline in its main
    function and that file is owned by the parallel foundation wave — promote
    this into add_ctx_cont as the shared function in the next cleanup wave.
    """
    eps = 1e-9
    close = close.astype(float)
    high = high.astype(float)
    low = low.astype(float)
    prev_close = close.shift(1).fillna(close)
    tr = np.maximum((high - low).abs(),
                    np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    atr_safe = tr.rolling(window=14, min_periods=1).mean().clip(lower=eps)
    pivot_high = (high > pd.concat(
        [high.shift(1), high.shift(2), high.shift(-1), high.shift(-2)], axis=1
    ).max(axis=1, skipna=True)).to_numpy()
    pivot_low = (low < pd.concat(
        [low.shift(1), low.shift(2), low.shift(-1), low.shift(-2)], axis=1
    ).min(axis=1, skipna=True)).to_numpy()
    n = len(close)
    idx = np.arange(n, dtype=np.int64)
    # running last-pivot index; rows before the first pivot -> 0 (loop init
    # last_high=high[0], last_hi_i=0 — identical semantics)
    last_high_idx = np.maximum.accumulate(np.where(pivot_high, idx, 0))
    last_low_idx = np.maximum.accumulate(np.where(pivot_low, idx, 0))
    hi_np, lo_np, cl_np = high.to_numpy(), low.to_numpy(), close.to_numpy()
    last_high_vals = hi_np[last_high_idx]
    last_low_vals = lo_np[last_low_idx]
    bars_since_high = (idx - last_high_idx).astype(np.float32)
    bars_since_low = (idx - last_low_idx).astype(np.float32)
    atr_np = atr_safe.to_numpy()
    dist_high = ((cl_np - last_high_vals) / atr_np).astype(np.float32)
    dist_low = ((cl_np - last_low_vals) / atr_np).astype(np.float32)
    denom = np.maximum(last_high_vals - last_low_vals, eps)
    retracement = np.zeros(n, dtype=np.float64)
    up = last_high_idx > last_low_idx
    dn = last_low_idx > last_high_idx
    retracement[up] = (last_high_vals[up] - cl_np[up]) / denom[up]
    retracement[dn] = (cl_np[dn] - last_low_vals[dn]) / denom[dn]
    retracement = np.clip(retracement, 0.0, 1.0).astype(np.float32)
    return {
        "dist_last_swing_high_atr": dist_high,
        "dist_last_swing_low_atr": dist_low,
        "bars_since_swing_high": bars_since_high,
        "bars_since_swing_low": bars_since_low,
        "retracement_from_last_impulse": retracement,
    }


@dataclass
class Smart520StateBuilder:
    """Builds smart520 entry states from the live anchored cv3+BASE28 frame.

    ordered_signal_names: the ACTIVE bundle's 520 ordered signal names
        (bundle_metadata.ordered_signal_names) — [0:41] MUST equal the
        ORDERED_SEQ_FIELDS_V3 contract; [41:] are the 479 extension names
        computed inline (manifest order == bundle order, verified at init).
    """
    ordered_signal_names: list[str]
    multi_tf: dict | None = None   # in-memory MTF-v2 bundle for group-A recompute
    _ext_names: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if len(self.ordered_signal_names) != SIGNAL_DIM_SMART520:
            raise RuntimeError(
                f"[SMART520_STATE] bundle signal dim {len(self.ordered_signal_names)} "
                f"!= {SIGNAL_DIM_SMART520}"
            )
        base = list(self.ordered_signal_names[: len(ORDERED_SEQ_FIELDS_V3)])
        if base != list(ORDERED_SEQ_FIELDS_V3):
            raise RuntimeError(
                "[SMART520_STATE] bundle ordered_signal_names[:41] != ORDERED_SEQ_FIELDS_V3 "
                f"(first mismatch: {[i for i, (a, b) in enumerate(zip(base, ORDERED_SEQ_FIELDS_V3)) if a != b][:5]})"
            )
        self._ext_names = list(self.ordered_signal_names[len(ORDERED_SEQ_FIELDS_V3):])

    # ── frame preparation (anchored convention) ─────────────────────────────

    def prepare_frame(
        self,
        joined: pd.DataFrame,
        bucket_ctx_cat: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Take the joined cv3+BASE28 anchored window (DatetimeIndex, rows
        [ANCHOR .. decision bar]) and recompute the frame-dependent families
        on it — the offline builder's exact order:
        neutral bridge -> group-A/dip-struct -> volume -> 'atr' -> entry-smart.
        Returns a NEW frame with a 'time' column (offline builders join on it).
        """
        if joined.empty:
            raise RuntimeError("[SMART520_STATE] empty joined frame")
        frame = joined.copy()
        frame.index.name = None   # joined index is named 'time' — avoid label ambiguity
        if "time" not in frame.columns:
            frame.insert(0, "time", frame.index)
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        frame = frame.sort_values("time").reset_index(drop=True)

        # 1) neutral XGB bridge — identical to the dataset's --neutral-xgb-bridge
        #    path (build_entry_v10_ctx_training_dataset_v3.py:1623-1625).
        from gx1.xgb.multihead.xgb_multihead_model_v1 import proba_to_signal_bridge_v1
        neutral = proba_to_signal_bridge_v1(np.full((len(frame), 3), 1.0 / 3.0, dtype=np.float64))
        for i, name in enumerate(ORDERED_BRIDGE_FIELDS_V3):
            frame[name] = neutral[:, i].astype(np.float32)

        # 1b) 'atr' + atr_bps + spread_bps — the OFFLINE derivations (verified
        #     numerically vs FULL_PLUS_CTX 2026-07-08):
        #     * atr        = TR rolling-14 mean (materialize_build_canonical_features_v1.py:136-141)
        #     * atr_bps    = atr / ((high+low)/2) * 1e4 (add_ctx_cont_columns_to_prebuilt.py:431-442)
        #     * spread_bps = one-truth _derive_spread_bps_from_available
        #       ((ask_close-bid_close)/bid_close*1e4 clip>=0; add_ctx_cont_columns_to_prebuilt.py:274-294)
        #     The live cv3/daemon values for these carry a DIFFERENT vintage
        #     (legacy zero-spread rows pre-spreadfix + a different live atr_bps
        #     formula), so they are recomputed here unconditionally — all
        #     lookbacks <=14 bars, anchored == full-history. NOTE: the exit-bound
        #     snapshot atr_bps deliberately keeps the RAW live cv3 row value
        #     (the joint-replay-proven exit convention) — see v12_pipeline.
        frame["atr"] = _compute_bare_atr14(frame)
        mid_hl = (pd.to_numeric(frame["high"], errors="coerce")
                  + pd.to_numeric(frame["low"], errors="coerce")) * 0.5
        frame["atr_bps"] = (
            frame["atr"].astype(np.float64) / np.maximum(mid_hl.to_numpy(np.float64), 1e-9) * 1e4
        ).astype(np.float32)
        from gx1.scripts.add_ctx_cont_columns_to_prebuilt import _derive_spread_bps_from_available
        frame = frame.drop(columns=[c for c in ("spread_bps",) if c in frame.columns])
        frame["spread_bps"] = _derive_spread_bps_from_available(frame).astype(np.float32)

        # 1c) full-frame override columns — ctx_cat buckets (offline
        #     frame-global-rank convention, compute_bucket_ctx_cat_full_frame)
        #     + the 5 long-lookback HTF ctx cols (compute_htf_ctx_full_frame).
        #     Overrides the daemon/B28 live values by time-join: the B28 M1-lane
        #     rows in the daemon's INCREMENTAL region are stamped one M5 bar
        #     behind the offline M5-row-label convention (parity gate finding
        #     2026-07-08), so B28 must never be the state source for these.
        if bucket_ctx_cat is not None:
            aligned = bucket_ctx_cat.reindex(pd.DatetimeIndex(frame["time"]))
            if aligned.isna().any().any():
                raise RuntimeError("[SMART520_STATE] frame_overrides missing rows for anchored window")
            for col in aligned.columns:
                if col in ("vol_regime_id", "atr_bucket", "spread_bucket", "H4_trend_sign_cat"):
                    frame[col] = aligned[col].to_numpy(np.int64)
                else:
                    frame[col] = aligned[col].to_numpy(np.float64)

        # 1d) micro (5) + swing (5) + session (5) ctx — recomputed on the
        #     anchored frame via the ONE-TRUTH live augmenter functions
        #     (v12_ctx_augment_live, the documented bit-parity mirrors of
        #     add_ctx_cont). NEVER sourced from B28 (M1-lane stamping skew,
        #     see 1c). All bounded/fast-converging (shift<=5, ewm span 5,
        #     swing pivots lookback=2) -> anchored == offline full-frame.
        from gx1.execution.v12_ctx_augment_live import (
            _add_micro_features, _add_session_features,
        )
        from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
            MICRO_FEATURE_NAMES, SESSION_CTX_CONT_NAMES,
        )
        _frame_ts = frame.set_index(pd.DatetimeIndex(frame["time"]))
        _add_micro_features(_frame_ts)
        _add_session_features(_frame_ts)
        for col in (list(MICRO_FEATURE_NAMES) + list(SESSION_CTX_CONT_NAMES) + ["session_id"]):
            frame[col] = _frame_ts[col].to_numpy()
        del _frame_ts
        for col, arr in _compute_offline_swing_block(
            frame["close"], frame["high"], frame["low"]
        ).items():
            frame[col] = arr

        # 2) group-A (24) + dip/struct (36) — DROP the loader's full-history
        #    in-memory values and recompute on THIS anchored frame (offline
        #    split-frame convention; see module docstring). attach_... is
        #    idempotent-if-present, so the drop is what forces the recompute.
        ga_cols = list(ORDERED_CTX_CONT_GROUP_A_PARITY) + list(ORDERED_CTX_CONT_DIP_STRUCT)
        frame = frame.drop(columns=[c for c in ga_cols if c in frame.columns])
        from gx1.scripts.augment_forward_outcome_v2 import attach_group_a_dip_struct_ctx_columns
        if self.multi_tf is None:
            raise RuntimeError(
                "[SMART520_STATE] multi_tf bundle required for group-A recompute — "
                "pass the in-memory MTF-v2 dict (build_multi_tf_from_cv3)"
            )
        frame = attach_group_a_dip_struct_ctx_columns(
            frame, multi_tf=self.multi_tf, journal_label="smart520_live",
        )

        # 3) volume features (4) — anchored-frame recompute (offline computed
        #    them on the split frame; rolling <=96 so values converge, but we
        #    mirror the convention exactly). add_volume_features overwrites.
        from gx1.features.volume_features import add_volume_features
        if "volume" not in frame.columns:
            raise RuntimeError("[SMART520_STATE] 'volume' column missing from live frame")
        frame = frame.drop(columns=[c for c in VOLUME_FEATURE_NAMES if c in frame.columns])
        add_volume_features(frame)

        # 4) bare 'atr' for the extension price layer: computed in step 1b.

        # 5) entry smart ctx (19) — anchored recompute AFTER group-A (consumes
        #    dist_to_* / dip_* from step 2), mirroring the offline order
        #    (build_entry_v10_ctx_training_dataset_v3.py:2280-2282).
        from gx1.features.entry_smart_context import add_entry_smart_context_features
        frame = frame.drop(
            columns=[c for c in ORDERED_CTX_CONT_ENTRY_SMART_DERIVED if c in frame.columns]
        )
        add_entry_smart_context_features(frame, strict=True)

        # Contract completeness — fail loud (never zero-fill for decisioning).
        missing_sig = [c for c in ORDERED_SEQ_FIELDS_V3 if c not in frame.columns]
        missing_ctx = [c for c in ORDERED_CTX_CONT_NAMES_V3 if c not in frame.columns]
        missing_cat = [c for c in ORDERED_CTX_CAT_NAMES_V3 if c not in frame.columns]
        if missing_sig or missing_ctx or missing_cat:
            raise RuntimeError(
                f"[SMART520_STATE] contract columns missing from live frame — "
                f"sig={missing_sig[:10]} ctx={missing_ctx[:10]} cat={missing_cat}"
            )
        return frame

    # ── state assembly ──────────────────────────────────────────────────────

    def build_states(
        self,
        frame: pd.DataFrame,
        target_times: Sequence[pd.Timestamp],
    ) -> dict[str, Any]:
        """Compute states for `target_times` (must exist in frame['time']).

        Returns dict with:
          seq      (n, 96, 520) float32
          snap     (n, 520)     float32
          ctx_cont (n, 142)     float32
          ctx_cat  (n, 5)       int64
          times    list[pd.Timestamp]
        Mirrors build_dataset_canonical's emission exactly:
        seq = sig_mat[i-95:i+1]; snap = sig_mat[i] (builder line 3024-3026).
        """
        times = pd.DatetimeIndex(pd.to_datetime(list(target_times), utc=True))
        pos_by_time = pd.Index(frame["time"])
        idxs: list[int] = []
        for ts in times:
            loc = pos_by_time.get_indexer([ts])
            if loc[0] < 0:
                raise RuntimeError(f"[SMART520_STATE] target bar {ts} not in live frame")
            if loc[0] < SEQ_LEN_SMART520 - 1:
                raise RuntimeError(
                    f"[SMART520_STATE] target bar {ts} has only {loc[0]} prior frame bars "
                    f"(needs >= {SEQ_LEN_SMART520 - 1}; anchored frame too short)"
                )
            idxs.append(int(loc[0]))

        # 479 extension signals — offline one-truth inline computation. The
        # price/candle layers read the SOURCE PARQUET by name, so hand them the
        # anchored frame's own price columns via a temp parquet (offline handed
        # FULL_PLUS_CTX; EMA-span features converge on the anchored frame —
        # proven by the parity gate).
        from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
            _build_inline_seq_structure_extension,
        )
        with tempfile.NamedTemporaryFile(suffix="_smart520_src.parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            frame[[c for c in _SOURCE_PARQUET_COLS if c in frame.columns]].to_parquet(
                tmp_path, index=False
            )
            ext_mat, ext_names, _meta = _build_inline_seq_structure_extension(
                frame,
                requested_features=self._ext_names,
                ctx_cont_names=list(ORDERED_CTX_CONT_NAMES_V3),
                ctx_cat_names=list(ORDERED_CTX_CAT_NAMES_V3),
                source_parquet=tmp_path,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
        if list(ext_names) != self._ext_names:
            raise RuntimeError("[SMART520_STATE] extension name order mismatch vs bundle")

        base_mat = frame[list(ORDERED_SEQ_FIELDS_V3)].astype(np.float32).to_numpy()
        sig_mat = np.concatenate([base_mat, ext_mat], axis=1).astype(np.float32, copy=False)
        if sig_mat.shape[1] != SIGNAL_DIM_SMART520:
            raise RuntimeError(
                f"[SMART520_STATE] signal width {sig_mat.shape[1]} != {SIGNAL_DIM_SMART520}"
            )
        ctx_cont_mat = frame[list(ORDERED_CTX_CONT_NAMES_V3)].astype(np.float32).to_numpy()
        ctx_cat_mat = frame[list(ORDERED_CTX_CAT_NAMES_V3)].astype(np.int64).to_numpy()

        n = len(idxs)
        seq = np.empty((n, SEQ_LEN_SMART520, SIGNAL_DIM_SMART520), dtype=np.float32)
        snap = np.empty((n, SIGNAL_DIM_SMART520), dtype=np.float32)
        ctx_cont = np.empty((n, CTX_CONT_DIM_SMART520), dtype=np.float32)
        ctx_cat = np.empty((n, CTX_CAT_DIM_SMART520), dtype=np.int64)
        for k, i in enumerate(idxs):
            seq[k] = sig_mat[i - (SEQ_LEN_SMART520 - 1): i + 1]
            snap[k] = sig_mat[i]
            ctx_cont[k] = ctx_cont_mat[i]
            ctx_cat[k] = ctx_cat_mat[i]
        if not np.isfinite(seq).all() or not np.isfinite(snap).all() or not np.isfinite(ctx_cont).all():
            raise RuntimeError("[SMART520_STATE] non-finite state values — refusing to serve")
        return {
            "seq": seq, "snap": snap, "ctx_cont": ctx_cont, "ctx_cat": ctx_cat,
            "times": [pd.Timestamp(t) for t in times],
        }


def compute_bucket_ctx_cat_full_frame(cv3: pd.DataFrame) -> pd.DataFrame:
    """vol_regime_id / atr_bucket / spread_bucket with the OFFLINE convention:
    FRAME-GLOBAL percentile-rank buckets (one-truth _rank_bucket_0_4 from
    add_ctx_cont_columns_to_prebuilt.py:705-712) over the model-range frame
    [SMART520_MODEL_RANGE_START_UTC, now], on the OFFLINE-derived atr_bps /
    spread_bps (NOT the daemon's frozen-edges live buckets — those carry the
    pre-spreadfix zero-spread history and a different atr formula, observed as
    spread_bucket skew in the parity gate 2026-07-08).

    NOTE the offline frame ended at the dataset build cutoff; live extends to
    'now', so ranks include post-evidence rows — the SAME drift class the
    offline convention itself has at every rebuild. Verified within tolerance
    by the parity gate.
    Returns a DataFrame indexed by cv3 time with the 3 int64 columns.
    """
    from gx1.scripts.add_ctx_cont_columns_to_prebuilt import (
        _derive_spread_bps_from_available,
        _rank_bucket_0_4,
    )
    if not isinstance(cv3.index, pd.DatetimeIndex):
        raise RuntimeError("[SMART520_STATE] cv3 must have a DatetimeIndex for bucket recompute")
    sub = cv3.loc[cv3.index >= SMART520_MODEL_RANGE_START_UTC,
                  [c for c in ("high", "low", "close", "bid_close", "ask_close") if c in cv3.columns]].copy()
    atr = _compute_bare_atr14(sub).astype(np.float64)
    mid = (pd.to_numeric(sub["high"], errors="coerce")
           + pd.to_numeric(sub["low"], errors="coerce")).to_numpy(np.float64) * 0.5
    atr_bps = atr / np.maximum(mid, 1e-9) * 1e4
    spread_bps = np.asarray(
        _derive_spread_bps_from_available(
            sub.drop(columns=[c for c in ("spread_bps",) if c in sub.columns])
        ),
        dtype=float,
    )

    def _bucket_with_frozen_tail(values: np.ndarray, fallback: int) -> np.ndarray:
        """Rank buckets over EXACTLY the offline frame [MODEL_RANGE_START,
        RANK_REF_END] (bit-parity with the promoted evidence); bars after
        RANK_REF_END are digitized against the FROZEN in-frame distribution
        (percentile = right-searchsorted fraction — the continuous-data
        equivalent of rank(pct, method='average'))."""
        in_ref = np.asarray(sub.index <= SMART520_RANK_REFERENCE_END_UTC)
        out = np.full(len(values), int(fallback), dtype=np.int64)
        ref_vals = values[in_ref]
        if ref_vals.size:
            out[in_ref] = _rank_bucket_0_4(ref_vals, fallback=fallback)
            tail = ~in_ref
            if tail.any():
                frozen = np.sort(ref_vals[np.isfinite(ref_vals)])
                pct = np.searchsorted(frozen, values[tail], side="right") / max(len(frozen), 1)
                out[tail] = np.clip(pct * 5.0, 0.0, 4.99).astype(np.int64)
        return out

    vol_regime_id = _bucket_with_frozen_tail(atr_bps, fallback=2)
    spread_bucket = _bucket_with_frozen_tail(spread_bps, fallback=0)
    return pd.DataFrame(
        {
            "vol_regime_id": vol_regime_id.astype(np.int64),
            "atr_bucket": vol_regime_id.astype(np.int64),   # offline: atr_bucket == vol_regime_id
            "spread_bucket": spread_bucket.astype(np.int64),
        },
        index=sub.index,
    )


def compute_htf_ctx_full_frame(cv3: pd.DataFrame) -> pd.DataFrame:
    """The 5 long-lookback HTF ctx columns (D1_dist_from_ema200_atr,
    D1_atr_percentile_252, H1_range_compression_ratio,
    M15_range_compression_ratio, H4_trend_sign_cat) recomputed FRESH over the
    full model-range frame [SMART520_MODEL_RANGE_START_UTC, now] via the
    ONE-TRUTH mirror v12_ctx_augment_live._add_htf_features (== the offline
    add_ctx_cont HTF block). NEVER taken from B28: the daemon's incremental
    M1-lane rows stamp these one M5 bar behind the offline convention (parity
    gate finding 2026-07-08). A bare work-frame is passed so the function's
    preserve-guard cannot short-circuit onto stale values.
    """
    from gx1.execution.v12_ctx_augment_live import _add_htf_features
    if not isinstance(cv3.index, pd.DatetimeIndex):
        raise RuntimeError("[SMART520_STATE] cv3 must have a DatetimeIndex for HTF recompute")
    sub_idx = cv3.index[cv3.index >= SMART520_MODEL_RANGE_START_UTC]
    m5 = cv3.loc[sub_idx, ["open", "high", "low", "close"]].copy()
    work = pd.DataFrame(index=sub_idx)
    _add_htf_features(work, m5)
    cols = [
        "D1_dist_from_ema200_atr", "D1_atr_percentile_252",
        "H1_range_compression_ratio", "M15_range_compression_ratio",
        "H4_trend_sign_cat",
    ]
    missing = [c for c in cols if c not in work.columns]
    if missing:
        raise RuntimeError(f"[SMART520_STATE] HTF recompute missing cols: {missing}")

    # REGIME_V4 DERIVED ctx (8) — one-truth add_regime_v4_features on the full
    # model-range frame, fed the RECOMPUTED D1_dist series (the daemon's cv3
    # carries d1_dist_roc_288_v3 / d1_dist_to_boundary_v3 built from ITS OWN
    # D1_dist history, which skews on weekend-open bars — parity gate finding
    # 2026-07-08). Per-TF sources ({tf}_regime_class_id_v2 / _ema_stack_aligned_v2 /
    # _trend_age_bars_norm_v2) come from cv3 (parity-verified equal offline).
    from gx1.features.regime_v4_features import (
        REGIME_V4_SOURCE_COLS,
        add_regime_v4_features,
    )
    src_cols = [c for c in REGIME_V4_SOURCE_COLS if c != "D1_dist_from_ema200_atr"]
    missing_src = [c for c in src_cols if c not in cv3.columns]
    if missing_src:
        raise RuntimeError(f"[SMART520_STATE] REGIME_V4 source cols missing from cv3: {missing_src}")
    rv_work = cv3.loc[sub_idx, src_cols].copy()
    rv_work["D1_dist_from_ema200_atr"] = work["D1_dist_from_ema200_atr"].to_numpy()
    add_regime_v4_features(rv_work)
    derived = [
        "regime_tf_agreement_v3", "regime_stack_sum_v3", "regime_divergence_flag_v3",
        "d1_dist_roc_288_v3", "d1_dist_to_boundary_v3", "d1_regime_changed_flag_v3",
        "bars_since_d1_regime_change_v3", "d1_trend_age_mature_flag_v3",
    ]
    out = work[cols].copy()
    for c in derived:
        out[c] = rv_work[c].to_numpy()
    return out


def build_multi_tf_from_cv3(cv3: pd.DataFrame) -> dict:
    """In-memory MTF-v2 bundle from the live cv3 frame — float32-cast OHLC(+volume),
    the EXACT dtype convention of both the offline disk cache
    (gx1/scripts/prebuild_multi_tf_cache_v2.py:60-66) and the trainer/eval dataset
    (entry_v10_ctx_train_v3.py:1634-1651). NOTE: PrebuiltStateLoader.build_multi_tf_features
    exists but feeds the EXIT chain with float64 OHLC — the entry parity target is the
    float32 cache convention, hence this thin one-truth wrapper (build fn is shared).
    """
    from gx1.features.htf_features import build_multi_tf_per_bar_features_v2
    cols = ["open", "high", "low", "close"]
    missing = [c for c in cols if c not in cv3.columns]
    if missing:
        raise RuntimeError(f"[SMART520_STATE] cv3 missing OHLC for MTF build: {missing}")
    m5 = cv3[cols].copy()
    for c in cols:
        m5[c] = m5[c].astype(np.float32)
    if "volume" in cv3.columns:
        m5["volume"] = cv3["volume"].astype(np.float32)
    if not isinstance(m5.index, pd.DatetimeIndex):
        raise RuntimeError("[SMART520_STATE] cv3 must have a DatetimeIndex for MTF build")
    return build_multi_tf_per_bar_features_v2(m5)
