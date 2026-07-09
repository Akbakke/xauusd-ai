#!/usr/bin/env python3
"""LIVE smart_seq520 entry adapter — serving-wave gap 4 (vedtak SMART_JOINT_POLICY_PROMOTION_20260708).

Loads the CONTRACT-RESOLVED ACTIVE v10_entry smart_seq520 bundle (cand#4) through
the one-truth offline loader (gx1.models.entry_v10.entry_v10_bundle.
load_entry_v10_ctx_bundle — strict load + direction/path calibration installed
into the forward), forwards it per M5 close on the live smart520 state
(Smart520StateBuilder, gap 2) + live multi-TF windows, and applies the PINNED
operating point read from PROJECT_STATE_artifacts.json (v10_entry.operating_point
— session gate US/OVERLAP + edge_score threshold; ONE truth, never re-declared
here or in the launcher).

Extend-don't-fork note (CLAUDE.md rule 7): the existing live wrapper
v12_v10_live.V10LiveInference implements the RETIRED legacy 41-dim
MASTER_TRANSFORMER_LOCK contract with a hand-built model constructor; the smart
bundle's one-truth load path is load_entry_v10_ctx_bundle (calibration +
specialist fusion + parked-head handling), which the offline evaluator
(evaluate_entry_candidate_selective_edge_v1._predict_bundle) also uses — this
adapter mirrors THAT forward exactly, so serve == the promoted evidence path.

edge_score / side (one-truth mirror of _predict_bundle, evaluate_entry_candidate_
selective_edge_v1.py:716-718):
    probs      = softmax(direction_logits)        # calibrated inside the model
    edge_score = max(p_long, p_short) - p_flat
    side       = LONG if p_long >= p_short else SHORT

Exit-bound snapshot: cand#4 heads -> v10_snapshot keys EXACTLY as the joint-replay
driver proved offline (reports/joint_smart_policy_replay_20260708/scripts/
replay_driver.py build_snapshot):
    direction_probs=[p_long,p_short,p_flat], path_quality=path_quality_pred,
    mfe_first_n=mfe_first_n_pred (raw), tradable_prob, bad_path_prob (carried,
    NOT consumed by the ACTIVE exit state), tf_agreement_pred/path_quality_std/
    position_size_pred = 0.0 (NOT consumed), atr_bps = live cv3 atr_bps at the
    prediction bar T. hold_horizon_bars_pred is DELIBERATELY ABSENT -> TradeState
    keeps the -1 sentinel -> the HOLD_HORIZON_EXPIRED Strategy-F rule stays INERT.
    That delta is live-equivalent BY CONSTRUCTION: cand#4's hold_horizon head is
    BLOCKED (bundle metadata blocked_heads) and the a1/deferral reference replays
    were snapshot-inert on it too — do NOT "fix" this by wiring a substitute value.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gx1.execution.v12_smart520_state_live import (
    SEQ_LEN_SMART520,
    SIGNAL_DIM_SMART520,
    SMART520_STATE_FRAME_ANCHOR_UTC,
    Smart520StateBuilder,
    append_multi_tf_incremental,
    build_multi_tf_from_cv3,
)

LOG = logging.getLogger("v12_smart_entry_live")

SESSION_NAMES = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}
SIDE_ACTION = {0: "TAKE_LONG_NOW", 1: "TAKE_SHORT_NOW"}

SMART_PARITY_GATE_LATEST = Path(
    "/home/andre2/GX1_DATA/reports/smart520_serve_parity_v1/SMART520_SERVE_PARITY_latest.json"
)

# Fail-closed context-staleness cap for LIVE decisions (serving-wave gap 3): when the
# last COMPLETED smart-context snapshot lags the decision bar by MORE than this many
# cv3 M5 bars, entry decisions are SKIPPED (journaled smart_ctx_stale_refresh_pending)
# until the background refresh lands — never decide on rotten context. Steady state is
# age<=1: the ~2-min refresh finishes well inside one M5 cycle.
SMART_CTX_MAX_STALENESS_M5 = int(os.environ.get("GX1_SMART_CTX_MAX_STALENESS_M5", "3"))

# Kill-switch for the (self-test-proven) incremental MTF splice at age>=1;
# 0 falls back to the raw snapshot bundle (staleness stays journaled via
# context_age_m5_bars). See SMART520_MTF_SPLICE_TFS in v12_smart520_state_live.
SMART_CTX_MTF_INCREMENTAL = os.environ.get("GX1_SMART_CTX_MTF_INCREMENTAL", "1") == "1"


class SmartContextStaleError(RuntimeError):
    """Raised by predict_live_bar when the context snapshot is older than
    SMART_CTX_MAX_STALENESS_M5 bars behind the decision bar — the pipeline
    journals it as a SKIP (fail-closed) and retries on the next poll."""

    def __init__(self, age: int, cap: int, ctx_cutoff: pd.Timestamp, end_ts: pd.Timestamp):
        super().__init__(
            f"[SMART_ENTRY] context snapshot {age} M5 bars behind decision bar {end_ts} "
            f"(cutoff {ctx_cutoff}, cap {cap}) — refusing to decide on stale context"
        )
        self.age = int(age)
        self.cap = int(cap)
        self.ctx_cutoff = ctx_cutoff
        self.end_ts = end_ts


@dataclass(frozen=True)
class SmartCtxSnapshot:
    """One COMPLETED smart-context build — swapped in as a single atomic reference
    (the loader's 2026-06-01 async-refresh pattern) so a decision that grabbed the
    snapshot can never observe a half-refreshed context. Immutable by convention:
    the background refresh builds a NEW snapshot and replaces the reference."""
    multi_tf: dict
    frame_overrides: pd.DataFrame       # bucket ctx_cat + HTF/REGIME_V4 override cols
    cv3_cutoff: pd.Timestamp
    built_utc: pd.Timestamp
    build_seconds: float


def assert_smart_serving_gate() -> dict:
    """ONE-TRUTH launch gate for the smart serving path (launcher + runner):
    (1) the TRAIN==SERVE parity gate artifact must be decision=PASS and must
        have been produced for the CONTRACT-ACTIVE v10_entry bundle;
    (2) the contract must be smart_seq520_candidate with a complete
        operating_point.
    Raises RuntimeError on any violation; returns the gate report on success.
    """
    import json
    from gx1_guards.artifacts import load_decision_entry
    if not SMART_PARITY_GATE_LATEST.is_file():
        raise RuntimeError(
            f"[SMART_GATE] parity gate artifact missing: {SMART_PARITY_GATE_LATEST} — run "
            f"gx1.scripts.verify_smart520_serve_parity_v1 (capped) first"
        )
    rep = json.loads(SMART_PARITY_GATE_LATEST.read_text())
    entry = load_decision_entry("v10_entry")
    problems: list[str] = []
    if rep.get("decision") != "PASS":
        problems.append(f"parity decision={rep.get('decision')!r} failures={list(rep.get('failures') or [])[:3]}")
    if str(rep.get("bundle_dir")) != str(entry["path"]):
        problems.append(f"parity bundle {rep.get('bundle_dir')} != contract-ACTIVE {entry['path']}")
    if str(entry.get("contract_mode")) != "smart_seq520_candidate":
        problems.append(f"contract_mode={entry.get('contract_mode')!r}")
    op = entry.get("operating_point")
    if not isinstance(op, dict) or "edge_score_threshold" not in op or "sessions" not in op:
        problems.append("v10_entry.operating_point missing/incomplete")
    if problems:
        raise RuntimeError("[SMART_GATE] LAUNCH BLOCKED: " + " | ".join(problems))
    return rep


@dataclass
class SmartEntryLiveInference:
    bundle_dir: Path
    operating_point: dict[str, Any]
    device: str = "cpu"
    _model: Any = field(default=None)
    _meta: dict = field(default_factory=dict)
    _builder: Smart520StateBuilder | None = field(default=None)
    _per_tf_seq_lens: dict[str, int] = field(default_factory=dict)
    _multi_tf_shift: dict = field(default_factory=dict, repr=False)
    # LAST COMPLETED context snapshot (one atomic reference — loader async pattern)
    # + the in-flight background refresh thread (serving-wave gap 3). The per-M1
    # EXIT path never touches either — no lock exists to starve it.
    _ctx: SmartCtxSnapshot | None = field(default=None, repr=False)
    _ctx_refresh_thread: threading.Thread | None = field(default=None, repr=False)
    # per-decision-bucket cache of the prepared anchored frame states
    _last_state_bucket: pd.Timestamp | None = field(default=None)

    # ── loading ──────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, bundle_dir: Path | None = None, device: str = "cpu") -> "SmartEntryLiveInference":
        from gx1_guards.artifacts import load_decision_entry
        entry = load_decision_entry("v10_entry")
        contract_bundle = Path(entry["path"])
        if bundle_dir is None:
            bundle_dir = contract_bundle
        else:
            bundle_dir = Path(bundle_dir)
            if bundle_dir.resolve() != contract_bundle.resolve():
                raise RuntimeError(
                    f"[SMART_ENTRY] explicit bundle_dir {bundle_dir} != contract-ACTIVE "
                    f"{contract_bundle} — rule 8: serve resolves ONLY through the contract"
                )
        mode = str(entry.get("contract_mode") or "")
        if mode != "smart_seq520_candidate":
            raise RuntimeError(
                f"[SMART_ENTRY] contract v10_entry.contract_mode={mode!r} — this adapter "
                f"serves smart_seq520_candidate only"
            )
        op = entry.get("operating_point")
        if not isinstance(op, dict):
            raise RuntimeError("[SMART_ENTRY] contract v10_entry.operating_point missing — fail-closed")
        for req in ("edge_score_threshold", "sessions"):
            if req not in op:
                raise RuntimeError(f"[SMART_ENTRY] operating_point missing '{req}' — fail-closed")

        from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
        bundle = load_entry_v10_ctx_bundle(bundle_dir=bundle_dir, device=device, xgb_models=None)
        model = bundle.transformer_model
        model.eval()
        meta = dict(bundle.metadata)
        if int(meta.get("seq_input_dim") or 0) != SIGNAL_DIM_SMART520:
            raise RuntimeError(
                f"[SMART_ENTRY] bundle seq_input_dim={meta.get('seq_input_dim')} != {SIGNAL_DIM_SMART520}"
            )
        if int(meta.get("seq_len") or 0) != SEQ_LEN_SMART520:
            raise RuntimeError(f"[SMART_ENTRY] bundle seq_len={meta.get('seq_len')} != {SEQ_LEN_SMART520}")
        if not isinstance(meta.get("direction_calibration"), dict):
            raise RuntimeError(
                "[SMART_ENTRY] bundle lacks direction_calibration — the promoted cand#4 is the "
                "CALIBRATED bundle; refusing an uncalibrated load"
            )
        mtf = meta.get("multi_tf") or {}
        if not bool(mtf.get("enabled")) or not bool(mtf.get("v2_mode")):
            raise RuntimeError("[SMART_ENTRY] bundle must be multi-TF v2 — refusing")
        per_tf = {
            "M5": int(mtf.get("m5_seq_len", 96)),
            "M15": int(mtf.get("m15_seq_len", 96)),
            "H1": int(mtf.get("h1_seq_len", 96)),
            "H4": int(mtf.get("h4_seq_len", 96)),
            "D1": int(mtf.get("d1_seq_len", 96)),
        }
        names = [str(x) for x in (meta.get("ordered_signal_names") or [])]
        builder = Smart520StateBuilder(ordered_signal_names=names)
        LOG.info(
            "[SMART_ENTRY] loaded contract-ACTIVE %s (mode=%s, thr=%.17g, sessions=%s, anchor=%s)",
            bundle_dir.name, mode, float(op["edge_score_threshold"]),
            list(op.get("sessions") or []), SMART520_STATE_FRAME_ANCHOR_UTC,
        )
        return cls(
            bundle_dir=bundle_dir, operating_point=dict(op), device=device,
            _model=model, _meta=meta, _builder=builder, _per_tf_seq_lens=per_tf,
        )

    # ── smart context (in-memory snapshot, refreshed on cv3 cutoff advance) ──
    # The build (~2 min: float32 MTF over full cv3 + frozen-rank buckets + full-
    # frame HTF/REGIME_V4 overrides) ran SYNCHRONOUSLY in the runner loop pre
    # gap-3 — every cv3 cutoff advance starved the per-M1 exit decisions for
    # ~2 min. Now it follows the loader's async-refresh pattern
    # (v12_state_from_prebuilt 2026-06-01): background thread builds a NEW
    # SmartCtxSnapshot on a LOCAL cv3 reference, then swaps ONE attribute
    # (GIL-atomic); decisions read the last completed snapshot and journal
    # context_age_m5_bars. No lock anywhere — the exit path cannot be starved.

    def _build_ctx_snapshot(self, cv3: pd.DataFrame) -> SmartCtxSnapshot:
        """The FULL context build (unchanged math — same one-truth functions the
        blocking path always used). Runs on local state only; safe in a thread."""
        from gx1.execution.v12_smart520_state_live import (
            compute_bucket_ctx_cat_full_frame,
            compute_htf_ctx_full_frame,
        )
        t0 = time.perf_counter()
        cutoff = cv3.index[-1]
        multi_tf = build_multi_tf_from_cv3(cv3)
        # full-frame overrides: ctx_cat buckets (offline frame-global-rank
        # convention) + the 5 long-lookback HTF ctx cols (fresh full-frame
        # recompute; B28's incremental M1-lane stamping is one M5 bar behind
        # the offline convention — parity gate finding 2026-07-08)
        overrides = pd.concat(
            [compute_bucket_ctx_cat_full_frame(cv3), compute_htf_ctx_full_frame(cv3)],
            axis=1,
        )
        return SmartCtxSnapshot(
            multi_tf=multi_tf, frame_overrides=overrides,
            cv3_cutoff=cutoff, built_utc=pd.Timestamp.utcnow(),
            build_seconds=time.perf_counter() - t0,
        )

    def _install_ctx_snapshot(self, snap: SmartCtxSnapshot) -> None:
        """Single-reference swap (GIL-atomic). The builder mirror exists only for
        direct Smart520StateBuilder callers; the live decision path passes the
        snapshot's bundle explicitly so it never races the mirror write."""
        self._ctx = snap
        if self._builder is not None:
            self._builder.multi_tf = snap.multi_tf

    def refresh_multi_tf(self, cv3: pd.DataFrame) -> None:
        """BLOCKING context (re)build when cv3's cutoff advanced — the startup /
        parity-gate / offline-driver path (semantics unchanged from pre-gap-3).
        The live runner path uses maybe_schedule_ctx_refresh + predict_live_bar
        instead and never blocks on this."""
        cutoff = cv3.index[-1]
        ctx = self._ctx
        if ctx is not None and ctx.cv3_cutoff == cutoff:
            return
        from gx1.features.htf_features import MULTI_TF_SHIFT
        LOG.info("[SMART_ENTRY] building smart-context snapshot from cv3 (cutoff=%s, blocking)…", cutoff)
        self._multi_tf_shift = dict(MULTI_TF_SHIFT)
        snap = self._build_ctx_snapshot(cv3)
        self._install_ctx_snapshot(snap)
        LOG.info("[SMART_ENTRY] smart-context snapshot ready (cutoff=%s, %.1fs)",
                 cutoff, snap.build_seconds)

    def maybe_schedule_ctx_refresh(self, cv3: pd.DataFrame) -> bool:
        """NON-BLOCKING: schedule a background context rebuild when cv3's cutoff
        advanced past the snapshot's and no refresh is in flight (the loader's
        refresh_if_changed pattern). Returns True only on the scheduling cycle."""
        ctx = self._ctx
        if ctx is None:
            raise RuntimeError(
                "[SMART_ENTRY] no context snapshot — the initial (blocking) "
                "refresh_multi_tf() at startup is mandatory before live decisions"
            )
        if cv3.index[-1] <= ctx.cv3_cutoff:
            return False
        t = self._ctx_refresh_thread
        if t is not None and t.is_alive():
            return False
        t = threading.Thread(
            target=self._async_ctx_refresh, args=(cv3,), daemon=True,
            name="smart_ctx_async_refresh",
        )
        self._ctx_refresh_thread = t
        t.start()
        return True

    def _async_ctx_refresh(self, cv3: pd.DataFrame) -> None:
        """Background-thread worker: full context build on the cv3 reference
        grabbed at schedule time (the loader swaps — never mutates — its frames,
        so this read is race-free), then one atomic snapshot swap. Fail-SAFE:
        on error the previous snapshot stays live and the staleness cap
        (SMART_CTX_MAX_STALENESS_M5) turns a persistent failure into journaled
        entry SKIPs — exits are never affected."""
        try:
            old = self._ctx
            snap = self._build_ctx_snapshot(cv3)
            self._install_ctx_snapshot(snap)
            LOG.info("[smart-ctx-refresh] snapshot cutoff %s → %s (took %.1fs, decisions never blocked)",
                     old.cv3_cutoff if old is not None else None,
                     snap.cv3_cutoff, snap.build_seconds)
        except Exception as exc:  # noqa: BLE001 — fail-safe: keep prior snapshot
            LOG.error(f"[smart-ctx-refresh] FAILED: {exc} — keeping previous snapshot "
                      f"(staleness cap will SKIP entries if this persists)")

    @staticmethod
    def context_age_m5_bars(cv3: pd.DataFrame, end_ts: pd.Timestamp,
                            ctx: SmartCtxSnapshot) -> int:
        """cv3 M5 bars in (ctx.cv3_cutoff, end_ts] — 0 ⇒ the snapshot covers the
        decision bar (may be negative for historical end_ts, e.g. the parity gate)."""
        idx = cv3.index
        return int(idx.searchsorted(end_ts, side="right")
                   - idx.searchsorted(ctx.cv3_cutoff, side="right"))

    def _effective_context(
        self, cv3: pd.DataFrame, ctx: SmartCtxSnapshot, end_ts: pd.Timestamp,
    ) -> tuple[dict, pd.DataFrame, int, bool]:
        """The snapshot context extended to end_ts (age > 0 = gap bars exist):
          * override tables — CHEAP (~0.6s, gap-3 probe) FULL-frame recompute on
            the current cv3 via the same one-truth functions the snapshot build
            used: causal + frozen-rank digitize, so overlapping rows are
            bit-identical and the gap bars are EXACT by construction (no ffill,
            no staleness).
          * MTF cache — the heavy part (~94s full): self-test-proven incremental
            tail splice (append_multi_tf_incremental) for M5/M15/H1; H4/D1 keep
            snapshot rows (forming-bar staleness only, journaled via
            context_age_m5_bars, capped by SMART_CTX_MAX_STALENESS_M5).
        Returns (multi_tf, frame_overrides, age, mtf_spliced)."""
        age = self.context_age_m5_bars(cv3, end_ts, ctx)
        if age <= 0:
            return ctx.multi_tf, ctx.frame_overrides, age, False
        from gx1.execution.v12_smart520_state_live import (
            compute_bucket_ctx_cat_full_frame,
            compute_htf_ctx_full_frame,
        )
        overrides = pd.concat(
            [compute_bucket_ctx_cat_full_frame(cv3), compute_htf_ctx_full_frame(cv3)],
            axis=1,
        )
        multi_tf, spliced = ctx.multi_tf, False
        if SMART_CTX_MTF_INCREMENTAL:
            multi_tf, spliced = append_multi_tf_incremental(cv3, ctx.multi_tf)
        return multi_tf, overrides, age, spliced

    def _prepare_anchored_frame(
        self, loader, cv3: pd.DataFrame, end_ts: pd.Timestamp,
        overrides: pd.DataFrame, multi_tf: dict,
    ) -> pd.DataFrame:
        """Shared anchored-window build + prepare (ONE truth for the blocking
        gate path and the live async path)."""
        cv3_idx = cv3.index
        n_from_anchor = int(cv3_idx.searchsorted(end_ts, side="right")
                            - cv3_idx.searchsorted(SMART520_STATE_FRAME_ANCHOR_UTC, side="left"))
        if n_from_anchor < SEQ_LEN_SMART520:
            raise RuntimeError(f"[SMART_ENTRY] anchored frame too short: {n_from_anchor} bars")
        joined = loader.get_window(end_ts, n_bars=n_from_anchor)
        if joined.empty or joined.index[0] < SMART520_STATE_FRAME_ANCHOR_UTC:
            raise RuntimeError(
                f"[SMART_ENTRY] anchored window build failed: rows={len(joined)} "
                f"start={joined.index[0] if len(joined) else None}"
            )
        return self._builder.prepare_frame(joined, bucket_ctx_cat=overrides, multi_tf=multi_tf)

    def build_anchored_frame(
        self, loader, end_ts: pd.Timestamp, ctx: SmartCtxSnapshot | None = None,
    ) -> pd.DataFrame:
        """ONE-TRUTH anchored state frame [SMART520_STATE_FRAME_ANCHOR_UTC .. end_ts]
        from the live prebuilt loader (joined cv3+BASE28), prepared with all
        smart520 recomputes. Shared by the parity gate and the live pipeline.
        ctx=None (gate/startup path): BLOCKING refresh first — behavior and
        values identical to the pre-gap-3 synchronous implementation."""
        if self._builder is None:
            raise RuntimeError("[SMART_ENTRY] not loaded")
        if ctx is None:
            self.refresh_multi_tf(loader._cv3)
            ctx = self._ctx
        cv3 = loader._cv3
        multi_tf, overrides, _age, _spliced = self._effective_context(cv3, ctx, end_ts)
        return self._prepare_anchored_frame(loader, cv3, end_ts, overrides, multi_tf)

    def _multi_tf_window_tensors(
        self, ts: pd.Timestamp, multi_tf: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """Per-TF windows at-or-before ts with the BUNDLE's per-TF seq lens —
        the exact offline dataset path (EntryV10CtxDataset._get_multi_tf_window:
        get_last_n_at_or_before(feats, ts, n=per_tf, tf_shift=MULTI_TF_SHIFT)).
        `multi_tf=None` uses the current snapshot (gate/offline callers)."""
        if multi_tf is None:
            ctx = self._ctx
            if ctx is None:
                raise RuntimeError("[SMART_ENTRY] multi-TF not built — call refresh_multi_tf() first")
            multi_tf = ctx.multi_tf
        from gx1.features.htf_features import get_last_n_at_or_before
        out: dict[str, torch.Tensor] = {}
        for tf, feats in multi_tf.items():
            n = int(self._per_tf_seq_lens.get(tf, SEQ_LEN_SMART520))
            arr = get_last_n_at_or_before(feats, ts, n=n, tf_shift=self._multi_tf_shift[tf])
            out[f"seq_{tf.lower()}"] = torch.from_numpy(
                arr.astype(np.float32, copy=False)
            ).unsqueeze(0).to(self.device)
        return out

    # ── forward ───────────────────────────────────────────────────────────────

    def forward_states(
        self, states: dict[str, Any], multi_tf: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Forward pre-built smart520 states (from Smart520StateBuilder) through
        the calibrated model. Mirrors evaluate_entry_candidate_selective_edge_v1
        _predict_bundle head-for-head. Returns one dict per state row.
        `multi_tf=None` uses the current snapshot (gate/offline callers); the
        live path passes the SAME bundle the states were built with."""
        if self._model is None:
            raise RuntimeError("[SMART_ENTRY] not loaded")
        results: list[dict[str, Any]] = []
        n = states["seq"].shape[0]
        with torch.no_grad():
            for k in range(n):
                ts = pd.Timestamp(states["times"][k])
                seq_t = torch.from_numpy(states["seq"][k]).unsqueeze(0).to(self.device)
                snap_t = torch.from_numpy(states["snap"][k]).unsqueeze(0).to(self.device)
                ctx_cont_t = torch.from_numpy(states["ctx_cont"][k]).unsqueeze(0).to(self.device)
                ctx_cat_t = torch.from_numpy(states["ctx_cat"][k]).unsqueeze(0).to(self.device)
                mtf_kwargs = self._multi_tf_window_tensors(ts, multi_tf=multi_tf)
                out = self._model(seq_t, snap_t, ctx_cat=ctx_cat_t, ctx_cont=ctx_cont_t, **mtf_kwargs)
                for key, value in out.items():
                    if torch.is_tensor(value) and not bool(torch.isfinite(value).all().item()):
                        raise RuntimeError(f"[SMART_ENTRY] non-finite model output '{key}' at {ts}")
                probs = torch.softmax(out["direction_logits"], dim=-1).cpu().float().numpy()[0]
                p_long, p_short, p_flat = float(probs[0]), float(probs[1]), float(probs[2])
                edge_score = max(p_long, p_short) - p_flat
                res = {
                    "time": ts,
                    "p_long": p_long, "p_short": p_short, "p_flat": p_flat,
                    "edge_score": float(edge_score),
                    "trade_side": 0 if p_long >= p_short else 1,
                    "session_id": int(states["ctx_cat"][k][0]),
                    "path_quality_pred": float(out["path_quality"].cpu().float().numpy().reshape(-1)[0]),
                    "bad_path_prob": float(torch.sigmoid(out["bad_path_logit"]).cpu().float().numpy().reshape(-1)[0]),
                    "tradable_prob": float(torch.sigmoid(out["tradable_logit"]).cpu().float().numpy().reshape(-1)[0]),
                    "mfe_first_n_pred": float(out["mfe_first_n"].cpu().float().numpy().reshape(-1)[0]),
                }
                results.append(res)
        return results

    # ── live per-M5 forward (async-context path — serving-wave gap 3) ────────

    def predict_live_bar(self, loader, end_ts: pd.Timestamp) -> dict[str, Any]:
        """LIVE per-M5 decision forward: uses the LAST COMPLETED context snapshot
        — NEVER blocks on the ~2-min context refresh (which now runs in a
        background thread, scheduled here on cv3 cutoff advance). One atomic
        snapshot grab keeps state build + model forward internally consistent.

        Fail-closed: raises SmartContextStaleError when the snapshot lags the
        decision bar by more than SMART_CTX_MAX_STALENESS_M5 cv3 bars (the
        pipeline journals the SKIP and retries next poll). Journals staleness on
        every result: context_age_m5_bars / context_cutoff_ts /
        context_refresh_in_flight / context_mtf_incremental.
        """
        if self._builder is None or self._model is None:
            raise RuntimeError("[SMART_ENTRY] not loaded")
        cv3 = loader._cv3
        self.maybe_schedule_ctx_refresh(cv3)
        ctx = self._ctx   # ONE atomic grab — never re-read during this decision
        if ctx is None:
            raise RuntimeError("[SMART_ENTRY] no context snapshot — startup refresh missing")
        age = self.context_age_m5_bars(cv3, end_ts, ctx)
        if age > SMART_CTX_MAX_STALENESS_M5:
            raise SmartContextStaleError(
                age=age, cap=SMART_CTX_MAX_STALENESS_M5,
                ctx_cutoff=ctx.cv3_cutoff, end_ts=end_ts,
            )
        multi_tf, overrides, age, spliced = self._effective_context(cv3, ctx, end_ts)
        frame = self._prepare_anchored_frame(loader, cv3, end_ts, overrides, multi_tf)
        states = self._builder.build_states(frame, [end_ts])
        head = self.forward_states(states, multi_tf=multi_tf)[0]
        t = self._ctx_refresh_thread
        head["context_age_m5_bars"] = int(max(age, 0))
        head["context_cutoff_ts"] = str(ctx.cv3_cutoff)
        head["context_refresh_in_flight"] = bool(t is not None and t.is_alive())
        head["context_mtf_incremental"] = bool(spliced)
        return head

    # ── decision (operating point from the contract — ONE truth) ─────────────

    def decide(self, head_out: dict[str, Any], atr_bps: float) -> dict[str, Any]:
        """Apply the pinned operating point to one forward result. Emits the
        runner-facing decision dict incl. the exit-bound _v10_snapshot."""
        thr = float(self.operating_point["edge_score_threshold"])
        sessions = {str(s) for s in (self.operating_point.get("sessions") or [])}
        session = SESSION_NAMES.get(int(head_out["session_id"]), f"UNKNOWN_{head_out['session_id']}")
        edge = float(head_out["edge_score"])
        take = (session in sessions) and (edge >= thr)
        side_idx = int(head_out["trade_side"])
        action = SIDE_ACTION[side_idx] if take else "SKIP"
        skip_reason = None
        if not take:
            skip_reason = "session_gate" if session not in sessions else "edge_below_threshold"

        # Exit-bound snapshot — replay-driver-proven mapping (module docstring).
        # hold_horizon_bars_pred DELIBERATELY ABSENT (blocked head -> -1 sentinel
        # -> HOLD_HORIZON_EXPIRED inert; live-equivalent to the joint replay).
        snapshot = {
            "decision_ts": str(head_out["time"]),
            "direction_probs": [head_out["p_long"], head_out["p_short"], head_out["p_flat"]],
            "path_quality": head_out["path_quality_pred"],
            "mfe_first_n": head_out["mfe_first_n_pred"],
            "tradable_prob": head_out["tradable_prob"],
            "bad_path_prob": head_out["bad_path_prob"],
            "tf_agreement_pred": 0.0,
            "path_quality_std": 0.0,
            "position_size_pred": 0.0,
            "atr_bps": float(atr_bps),
        }
        out = {
            "action": action,
            "action_id": {"SKIP": 0, "TAKE_LONG_NOW": 1, "TAKE_SHORT_NOW": 2}[action],
            "edge_score": edge,
            "edge_score_threshold": thr,
            "session": session,
            "smart_skip_reason": skip_reason,
            "p_long": head_out["p_long"],
            "p_short": head_out["p_short"],
            "p_flat": head_out["p_flat"],
            "v10_path_quality_pred": head_out["path_quality_pred"],
            "v10_mfe_pred_at_entry": head_out["mfe_first_n_pred"],
            "v10_tradable_prob": head_out["tradable_prob"],
            "v10_bad_path_prob": head_out["bad_path_prob"],
            "decision_ts": str(head_out["time"]),
            "_v10_snapshot": snapshot,
            "policy": "smart_seq520_candidate_v1",
            "stub": False,
        }
        # async-context staleness journal (serving-wave gap 3) — present only on
        # the live predict_live_bar path; the parity gate forwards heads directly.
        for k in ("context_age_m5_bars", "context_cutoff_ts",
                  "context_refresh_in_flight", "context_mtf_incremental"):
            if k in head_out:
                out[k] = head_out[k]
        return out
