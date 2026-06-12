"""Cross-chain FEATURE-LIVENESS audit — "NOTHING gets silently ignored" (user vedtak 2026-06-06).

A consolidated, ALWAYS-RUN check that every input feature the chain consumes is ALIVE (non-constant)
and that the multi-TF block is intact (all 5 TFs present, alive, DISTINCT resolutions). It complements
(does NOT duplicate) the stage-local guards:
  - Entry-IQL build dead-zero guard      gx1/scripts/materialize_build_entry_iql_v2.py:893
  - Entry-IQL serve REQUIRED-feature gate gx1/runtime/entry_iql_v2_adapter.py:64
  - V10 build ORDERED_CTX_CONT fail-close gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py

It FAILS LOUD (raises FeatureLivenessError) when a feature OUTSIDE the documented KNOWN_ALLOWED_DEAD
allowlist is dead — i.e. it catches a NEW regression (a feature that silently went constant), while
tolerating the known structural-dead set (tracked in the dead-feature hygiene wave).

Run automatically: the V10 trainer calls assert_v10_batch_liveness() at post-export.
Run manually:  python -m gx1.audit.feature_liveness --v10-bundle <dir> --test-parquet <pq> --m5-prebuilt <pq>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

DEAD_STD = 1e-4   # std below this over a real batch = constant = "ignored input"

# ── Documented structural/known-dead allowlist (the dead-feature hygiene wave) ──────────────
# Format: bare name  OR  "tf:name" for a per-TF multi-TF feature. Edit ONLY with a documented reason.
KNOWN_ALLOWED_DEAD: Dict[str, str] = {
    # Structural — XAU OHLC has NO bid/ask spread, so any spread/cost-from-spread is const:
    "spread_bps": "XAU OHLC has no bid/ask spread (structural). Hygiene wave: drop or wire a cost proxy.",
    "spread_bucket": "bucketization of the const spread (structural).",
    "_v1_cost_bps_est": "const fallback (no spread). _v1_cost_bps_dyn IS alive.",
    "vol_pct_m5_1yr": "1-year vol-percentile not computed → pinned 0.5. Hygiene wave: compute or drop.",
    "vol_pct_h1_1yr": "ditto (pinned 0.5).",
    # Benign by construction — XGB is SESSION-HEADED so session feats are const within a head:
    "session_id": "0 XGB gain by construction (session-headed model).",
    "is_ASIA": "ditto.", "session_change_flag": "ditto.", "session_tradable": "ditto.",
    "minutes_since_session_open": "ditto.", "minutes_to_next_session_boundary": "ditto.",
    # Known bugs/gaps tracked in the hygiene wave (NOT to be silently forgotten):
    "_v1_atr_regime_id": "BUG-MASK: chained-index BUG → const=1 (basic_v1.py:726). Fix EXISTS behind "
                         "GX1_ATR_REGIME_FIX=1 (bf4a6abd, default OFF — live builds still emit const). REMOVE this "
                         "entry at the first rebuild that enables the gate, or it will mask the then-alive feature.",
    "smc_choch": "BUG-MASK (remove when fixed): too sparse (0.1% nonzero) → 0 gain. Hygiene wave: decay to bars_since_choch.",
    # Multi-TF window-property (NOT a bug): D1 EMA-stack alignment can be const over a calm window:
    "d1:ema_stack_aligned_v2": "D1 regime can be stable over a test window → const there; alive in other TFs.",
    # Provenance one-hot — const BY CONSTRUCTION (verified 2026-06-10 by the full-state re-audit): decision_reason is
    # hardcoded `['v2_inference_batch']*n` at candidate-gen (materialize_inference_batch_candidates_v3_v1.py:502),
    # n_unique=1 → the get_dummies column is constant=1, zero possible signal. Appears in Entry-IQL(197) + Exit-IQL(209)
    # state (double- vs single-underscore get_dummies sep). Harmless dead slot; DROP at the next rebuild.
    "decision_reason__v2_inference_batch": "Entry-IQL: const provenance one-hot (hardcoded reason, n_unique=1). Drop at rebuild.",
    "decision_reason_v2_inference_batch": "Exit-IQL: same const provenance one-hot. Drop at rebuild.",
}

MULTI_TF_NAMES: Sequence[str] = ()
try:
    from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V2 as _MTF
    MULTI_TF_NAMES = tuple(_MTF)
except Exception:  # pragma: no cover
    pass


class FeatureLivenessError(RuntimeError):
    """A feature outside KNOWN_ALLOWED_DEAD went constant — a silent-ignore regression."""


def _dead_cols(arr: np.ndarray, names: Sequence[str], tf: str = "") -> List[str]:
    """Return names whose column std < DEAD_STD and which are NOT on the allowlist."""
    arr = np.asarray(arr, dtype=np.float64)
    flat = arr.reshape(-1, arr.shape[-1])
    std = flat.std(axis=0)
    out: List[str] = []
    for j in range(flat.shape[1]):
        if std[j] >= DEAD_STD:
            continue
        nm = names[j] if j < len(names) else f"[{j}]"
        if nm in KNOWN_ALLOWED_DEAD or (tf and f"{tf}:{nm}" in KNOWN_ALLOWED_DEAD):
            continue
        out.append(f"{tf+':' if tf else ''}{nm} (std={std[j]:.1e})")
    return out


def check_multi_tf_integrity(seq_by_tf: Dict[str, np.ndarray]) -> Dict[str, object]:
    """All 5 TFs present, shape (B,L,25), ≤1 allow-listed dead per TF, DISTINCT resolutions.

    seq_by_tf: {"M5": (B,L,25), "M15": ..., "H1": ..., "H4": ..., "D1": ...}
    Returns a report dict; the caller decides to raise on `new_dead`/`missing`/`duplicate`.
    """
    rep: Dict[str, object] = {"missing": [], "new_dead": [], "duplicate": [], "atr_by_tf": {}}
    names = list(MULTI_TF_NAMES)
    want = ["M5", "M15", "H1", "H4", "D1"]
    rep["missing"] = [tf for tf in want if tf not in seq_by_tf]
    atr_idx = names.index("atr_bps_14") if "atr_bps_14" in names else 0
    ema50_idx = names.index("ema50_dist_atr") if "ema50_dist_atr" in names else None
    for tf, arr in seq_by_tf.items():
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim != 3 or a.shape[-1] != len(names):
            rep["new_dead"].append(f"{tf}: BAD SHAPE {a.shape} (expected (B,L,{len(names)}))")
            continue
        rep["new_dead"].extend(_dead_cols(a, names, tf=tf.lower()))
        # mask zero-padded warmup rows (atr==0) so the ATR-scaling sanity isn't deflated on a skewed batch
        _atr_col = a.reshape(-1, a.shape[-1])[:, atr_idx]; _nz = _atr_col[_atr_col > 0]
        rep["atr_by_tf"][tf] = float(_nz.mean()) if _nz.size else 0.0
    # distinctness: ema50_dist series must not be ~identical across TFs (corr<0.98)
    if ema50_idx is not None and {"M5", "D1"} <= set(seq_by_tf):
        def ser(tf):
            return np.asarray(seq_by_tf[tf], np.float64)[:, :, ema50_idx].reshape(-1)
        for a, b in (("M5", "D1"), ("M5", "H1"), ("H1", "D1")):
            sa, sb = ser(a), ser(b); n = min(len(sa), len(sb))  # truncate so the corr ALWAYS runs (no silent skip)
            if n > 100:
                r = float(np.corrcoef(sa[:n], sb[:n])[0, 1])
                if abs(r) > 0.98:
                    rep["duplicate"].append(f"{a}~{b} ema50_dist corr={r:+.3f} (TFs not distinct!)")
    # ATR-scaling sanity: D1 atr should exceed M5 atr (coarser bars span more)
    atr = rep["atr_by_tf"]
    if "M5" in atr and "D1" in atr and atr["D1"] <= atr["M5"]:
        rep["new_dead"].append(f"ATR-SCALE ANOMALY: D1 atr {atr['D1']:.1f} <= M5 atr {atr['M5']:.1f}")
    return rep


def assert_v10_batch_liveness(batch: dict, *, ctx_cont_names: Optional[Sequence[str]] = None,
                              snap_names: Optional[Sequence[str]] = None, raise_on_fail: bool = True) -> dict:
    """Light check the V10 trainer calls at post-export. `batch` = one EntryV10CtxDataset batch.
    Checks ctx_cont/snap liveness + multi-TF integrity. Raises FeatureLivenessError on new deadness."""
    def to_np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    issues: List[str] = []
    if "ctx_cont" in batch and ctx_cont_names is not None:
        issues += [f"ctx_cont:{d}" for d in _dead_cols(to_np(batch["ctx_cont"]), list(ctx_cont_names))]
    if "snap_x" in batch and snap_names is not None:  # the dataset batch key is snap_x (the 41-dim XGB bridge)
        issues += [f"snap:{d}" for d in _dead_cols(to_np(batch["snap_x"]), list(snap_names))]
    seq_by_tf = {k.replace("seq_", "").upper(): to_np(batch[k])
                 for k in ("seq_m5", "seq_m15", "seq_h1", "seq_h4", "seq_d1") if k in batch}
    mtf = check_multi_tf_integrity(seq_by_tf) if seq_by_tf else {"missing": ["ALL — no multi-TF in batch"], "new_dead": [], "duplicate": []}
    issues += [f"multi_tf.{k}={v}" for k in ("missing", "new_dead", "duplicate") for v in (mtf.get(k) or [])]
    rep = {"ok": not issues, "issues": issues, "multi_tf_atr": mtf.get("atr_by_tf", {})}
    if issues:
        msg = "[FEATURE_LIVENESS_FAIL] features/dependencies went silently dead (not on allowlist):\n  - " + "\n  - ".join(issues)
        if raise_on_fail:
            raise FeatureLivenessError(msg)
        print(msg, file=sys.stderr)
    return rep


def audit_iql_state_liveness(X, names, *, role: str = "iql-state", raise_on_fail: bool = False) -> dict:
    """Liveness check for an already-built Entry/Exit-IQL STATE matrix — closes the rule-9 COVERAGE GAP.

    There was NO liveness check on the Entry-IQL (197) or Exit-IQL (209) STATE vectors — that is how 36
    self-computed dip/struct features stayed CONST-ZERO, un-allowlisted, for months. This is the reusable
    enforcement primitive: the IQL builds pass their built (X, names) in to fail LOUD on any state feature
    that is constant/zero and NOT on KNOWN_ALLOWED_DEAD.

    X: (N, F) state matrix (the 1st return of build_state_matrix). names: F feature names (2nd return).
    Pure wrapper on _dead_cols (the one-truth checker) — takes already-built values, so it adds NO heavy
    import to this light lib (which the V10/XGB trainers import at post-export). raise_on_fail=True at cement.
    """
    arr = np.asarray(X, dtype=np.float64)
    names = list(names)
    if arr.shape[-1] != len(names):
        raise ValueError(f"audit_iql_state_liveness[{role}]: {arr.shape[-1]} cols vs {len(names)} names — mismatch")
    flat = arr.reshape(-1, arr.shape[-1])                 # view, no copy (C-contiguous)
    dead = _dead_cols(arr, names)
    n_zero = int((~flat.any(axis=0)).sum())               # all-zero columns (single pass; callers needn't recompute)
    rep = {"ok": not dead, "role": role, "n_rows": int(flat.shape[0]),
           "n_features": arr.shape[-1], "n_zero": n_zero, "dead": dead}
    if dead:
        msg = (f"[FEATURE_LIVENESS_FAIL] {role}: {len(dead)} state feature(s) constant/zero and NOT on "
               f"KNOWN_ALLOWED_DEAD — a silent-ignore regression (rule 9):\n  - " + "\n  - ".join(dead))
        if raise_on_fail:
            raise FeatureLivenessError(msg)
        print(msg, file=sys.stderr)
    return rep


LIVE_TAIL_ALLOWED_CONST: Dict[str, str] = {
    # Structural — XAU OHLC has no bid/ask spread on the M5-derived path:
    "spread_bps": "XAU OHLC no spread (structural).",
    "spread_bucket": "bucketization of the const spread (structural).",
    # REGIME-SATURATED capped formula, NOT an append freeze (verified 2026-06-11, the live-tail
    # check's first real catch): cost = 12*sess*(0.5+0.5*clip(0.6*atr_pct*1e4+0.4*rng_z, 0, 3))
    # saturates at 24.0 whenever atr_bps >= ~5 — permanently true in the 2026 high-vol regime
    # (16,914/17k values = 24.0 over the last 60d; basic_v1.py:1635-1644). train==serve consistent
    # (batch saturates identically). FIX CANDIDATE at the next feature wave: raise/log-scale the cap
    # so the cost proxy regains information in high-vol regimes.
    "_v1_cost_bps_dyn": "scale_term cap 3.0 saturates in the 2026 high-vol regime → const 24.0. Uncap at next feature wave.",
}
LIVE_TAIL_REF_MIN_NUNIQUE = 4   # was-varying threshold: ref-window nunique >= this => freeze when tail==1
LIVE_BASE34_MANIFEST = "/home/andre2/GX1_DATA/data/data/prebuilt/BASE28_CANONICAL/CURRENT_MANIFEST.json"
LIVE_CV3_PREBUILT = ("/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/"
                     "xauusd_m5_CANONICAL_V3_2020_2026.parquet")


US_MARKET_HOLIDAYS = {  # XAU/CME stengt eller early-close (utvid årlig)
    "2025-01-01","2025-01-20","2025-02-17","2025-04-18","2025-05-26","2025-06-19","2025-07-04",
    "2025-09-01","2025-11-27","2025-11-28","2025-12-24","2025-12-25","2025-12-26","2025-12-31",
    "2026-01-01","2026-01-19","2026-02-16","2026-04-03","2026-05-25","2026-07-03","2026-09-07",
    "2026-11-26","2026-11-27","2026-12-24","2026-12-25","2026-12-31",
}
KNOWN_DATA_GAPS = {  # aksepterte historiske hull (dato → grunn). Repareres via OANDA-backfill når mulig.
    "2025-04-24": "80min utfall (historisk, pre-rule).",
    "2025-12-07": "31min tynn søndagsåpning.",
    "2026-03-27": "24min utfall (historisk).",
    "2026-03-31": "HEL DAG 04-01 mangler — April-repair-grense. BACKFILL-KANDIDAT.",
    "2026-04-11": "lørdagsbars + 38t gap — repair-artefakt. UNDERSØKES.",
    "2026-05-29": "75min utfall.",
    "2026-06-05": "11.4t utfall (maskin nede) — BACKFILL-KANDIDAT.",
    "2026-06-10": "OOM-reboot (2 hull 35+75min) — BACKFILL-KANDIDAT.",
    "2026-06-11": "reboot 15min.",
}


def check_live_continuity(tail_days: int = 10, fresh_fail_hours: int = 48,
                          raise_on_fail: bool = False) -> dict:
    """Rule-9 CONTINUITY check (user-direktiv 2026-06-12: «ALLTID oppdatert, INGEN hull, nøyaktig på
    hver M1 (exit) og M5 (entry)»). Skanner cv3 (M5) + BASE34 (M1) for grid-hull, klassifisert mot
    helg / daglig 21-22Z-pause / US-helligdager / tick-tomme minutter (<=10 min) / KNOWN_DATA_GAPS.
    Et UKJENT hull NYERE enn fresh_fail_hours = FAIL (collector/daemon-utfall pågår eller nettopp
    skjedd); eldre ukjente hull rapporteres for backfill. Sjekker også ferskhet (cutoff-alder)."""
    import pandas as pd
    out: dict = {"ok": True, "fresh_gaps": [], "stale_gaps": [], "freshness_min": {}}
    now = pd.Timestamp.now(tz="UTC")
    man = json.loads(Path(LIVE_BASE34_MANIFEST).read_text())
    frames = []
    cv3 = pd.read_parquet(LIVE_CV3_PREBUILT, columns=["time"])
    frames.append(("CV3-M5", pd.DatetimeIndex(pd.to_datetime(cv3["time"], utc=True)).sort_values(), 5))
    b34 = pd.read_parquet(man["parquet_path"], columns=[])
    frames.append(("BASE34-M1", pd.DatetimeIndex(pd.to_datetime(b34.index, utc=True)).sort_values(), 1))
    for name, idx, step in frames:
        out["freshness_min"][name] = round(float((now - idx.max()).total_seconds() / 60), 1)
        idx = idx[idx >= (now - pd.Timedelta(days=tail_days))]
        if len(idx) < 2:
            out["fresh_gaps"].append(f"{name}: <2 bars i {tail_days}d-vinduet")
            continue
        diffs = pd.Series(idx[1:]) - pd.Series(idx[:-1])
        mask = diffs > pd.Timedelta(minutes=step)
        for s, dlt in zip(idx[:-1][mask], diffs[mask]):
            m = dlt.total_seconds() / 60
            if s.dayofweek == 4 and 2300 <= m <= 3200: continue          # helg
            if s.hour in (20, 21) and 50 <= m <= 75: continue            # daglig pause
            if m <= 10: continue                                          # tick-tomt
            keys = {(s + pd.Timedelta(days=o)).strftime("%Y-%m-%d") for o in (0, 1)}
            if keys & US_MARKET_HOLIDAYS: continue
            if keys & set(KNOWN_DATA_GAPS):
                out["stale_gaps"].append(f"{name}: {s} +{m:.0f}min (kjent: {KNOWN_DATA_GAPS[sorted(keys & set(KNOWN_DATA_GAPS))[0]]})")
                continue
            entry = f"{name}: {s} +{m:.0f}min UKJENT"
            if (now - s) <= pd.Timedelta(hours=fresh_fail_hours):
                out["fresh_gaps"].append(entry)
            else:
                out["stale_gaps"].append(entry)
    out["ok"] = not out["fresh_gaps"]
    if not out["ok"]:
        msg = (f"[RULE9-CONTINUITY] FERSKE UKJENTE HULL i live-prebuilts: {out['fresh_gaps']} — "
               f"collector/daemon-utfall; reparér (OANDA-backfill) før live stoler på lookback.")
        if raise_on_fail:
            raise FeatureLivenessError(msg)
        print(msg, file=sys.stderr)
    return out


def check_live_prebuilt_tail(tail_days: int = 5, ref_days: int = 30,
                             base34_manifest: str = LIVE_BASE34_MANIFEST,
                             cv3_path: str = LIVE_CV3_PREBUILT,
                             raise_on_fail: bool = False) -> dict:
    """Rule-9 LIVE-TAIL check (user vedtak 2026-06-11): detect the FREEZE SIGNATURE on the LIVE
    prebuilts — a column constant over the recent tail that USED to vary in the reference window.

    Training-data audits can never see this class of failure: the BASE34 copy-forward freeze lived
    17 days (2026-05-25→06-11, session pinned US / atr_bps const into the live entry+exit states)
    while every training-side liveness audit was green.

    FAIL signature per column: nunique(tail) == 1 AND nunique(ref) >= LIVE_TAIL_REF_MIN_NUNIQUE
    (was-varying, now-frozen), off LIVE_TAIL_ALLOWED_CONST. Constant in BOTH windows = structural
    (reported, not failed). Also reports cutoff staleness (informational — markets may be closed).
    """
    import pandas as pd
    out: dict = {"ok": True, "frozen": [], "structural_const": [], "checked": {}, "stale_minutes": {}}
    frames: list[tuple[str, "pd.DataFrame"]] = []
    man = json.loads(Path(base34_manifest).read_text())
    b34 = pd.read_parquet(man["parquet_path"])
    b34.index = pd.to_datetime(b34.index, utc=True)
    frames.append(("BASE34", b34))
    cv3 = pd.read_parquet(cv3_path)
    if "time" in cv3.columns:
        cv3["time"] = pd.to_datetime(cv3["time"], utc=True)
        cv3 = cv3.set_index("time")
    frames.append(("CV3", cv3.sort_index()))
    now = pd.Timestamp.now(tz="UTC")
    for name, df in frames:
        cutoff = df.index.max()
        out["stale_minutes"][name] = round(float((now - cutoff).total_seconds() / 60.0), 1)
        tail_start = cutoff - pd.Timedelta(days=tail_days)
        ref_start = tail_start - pd.Timedelta(days=ref_days)
        tail = df[df.index > tail_start]
        ref = df[(df.index > ref_start) & (df.index <= tail_start)]
        n_checked = 0
        for c in df.columns:
            if df[c].dtype.kind not in "fiub":
                continue
            n_checked += 1
            nu_tail = int(tail[c].nunique(dropna=True))
            if nu_tail > 1:
                continue
            nu_ref = int(ref[c].nunique(dropna=True))
            if nu_ref >= LIVE_TAIL_REF_MIN_NUNIQUE and c not in LIVE_TAIL_ALLOWED_CONST:
                out["frozen"].append(f"{name}:{c} (tail const={tail[c].dropna().iloc[-1] if len(tail[c].dropna()) else 'NaN'}, ref nunique={nu_ref})")
            elif nu_ref <= 1:
                out["structural_const"].append(f"{name}:{c}")
        out["checked"][name] = {"cols": n_checked, "tail_rows": len(tail), "ref_rows": len(ref)}
    out["ok"] = not out["frozen"]
    if not out["ok"]:
        msg = (f"[RULE9-LIVE-TAIL] FREEZE SIGNATURE on the live prebuilt(s): {out['frozen']} — "
               f"a was-varying column is now constant on the {tail_days}d tail. Fix the append wiring; "
               f"NEVER let live serve frozen context (the 2026-05-25 BASE34 freeze class).")
        if raise_on_fail:
            raise FeatureLivenessError(msg)
        print(msg, file=sys.stderr)
    return out


def audit_xgb_gain(bundle_dir: str, contract_path: str) -> List[str]:
    """Return base80 features with 0 gain in ALL session heads, excluding the allowlist."""
    from gx1.xgb.multihead.xgb_multihead_model_v1 import XGBMultiheadModel
    feats = json.loads(Path(contract_path).read_text())["features"]
    m = XGBMultiheadModel.load(str(Path(bundle_dir) / "xgb_universal_multihead_v2.joblib"))
    used = set()
    for _, head in m.heads.items():
        b = head.get_booster() if hasattr(head, "get_booster") else head
        for k in b.get_score(importance_type="gain"):
            idx = int(k[1:]) if k.startswith("f") and k[1:].isdigit() else (feats.index(k) if k in feats else None)
            if idx is not None:
                used.add(idx)
    return [feats[i] for i in range(len(feats)) if i not in used and feats[i] not in KNOWN_ALLOWED_DEAD]


def _main() -> int:
    ap = argparse.ArgumentParser(description="Cross-chain feature-liveness audit (nothing ignored).")
    ap.add_argument("--v10-bundle", type=str, default=None)
    ap.add_argument("--test-parquet", type=str, default=None)
    ap.add_argument("--m5-prebuilt", type=str, default=None)
    ap.add_argument("--xgb-bundle", type=str, default=None)
    ap.add_argument("--xgb-contract", type=str, default="gx1/xgb/contracts/xgb_input_features_base80_v1.json")
    ap.add_argument("--strict", action="store_true", help="exit nonzero if any NEW dead feature")
    ap.add_argument("--live-tail", action="store_true",
                    help="rule-9 LIVE-TAIL check: freeze-signature scan of the live cv3+BASE34 prebuilt tails")
    ap.add_argument("--tail-days", type=int, default=5)
    ap.add_argument("--ref-days", type=int, default=30)
    a = ap.parse_args()
    failed = False
    if a.live_tail:
        rep = check_live_prebuilt_tail(tail_days=a.tail_days, ref_days=a.ref_days)
        print(f"[LIVE-TAIL] checked={rep['checked']} stale_min={rep['stale_minutes']}")
        print(f"[LIVE-TAIL] structural-const (info): {len(rep['structural_const'])} cols")
        print(f"[LIVE-TAIL] {'OK ✓ — no freeze signature' if rep['ok'] else 'FROZEN: ' + repr(rep['frozen'])}")
        failed |= not rep["ok"]
        crep = check_live_continuity()
        print(f"[CONTINUITY] freshness_min={crep['freshness_min']}  kjente/gamle hull: {len(crep['stale_gaps'])}")
        print(f"[CONTINUITY] {'OK ✓ — ingen ferske ukjente hull' if crep['ok'] else 'FERSKE HULL: ' + repr(crep['fresh_gaps'])}")
        failed |= not crep["ok"]
    if a.xgb_bundle:
        dead = audit_xgb_gain(a.xgb_bundle, a.xgb_contract)
        print(f"[XGB] new-dead (0 gain, off allowlist): {dead or 'NONE ✓'}")
        failed |= bool(dead)
    if a.v10_bundle and a.test_parquet and a.m5_prebuilt:
        import os, torch
        os.environ.setdefault("GX1_REGIME_V4", "1"); os.environ.setdefault("GX1_TREND_REGIME_FROM_D1", "1")
        from torch.utils.data import DataLoader
        from gx1.models.entry_v10.entry_v10_ctx_train_v3 import EntryV10CtxDataset
        from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
        meta = load_entry_v10_ctx_bundle(bundle_dir=Path(a.v10_bundle), device="cpu", xgb_models=None).metadata
        # Use the FULL one-truth ctx_cont names (123). The bundle metadata truncates to ~21, which
        # would leave indices unnamed → unmatchable against the allowlist → false fails. The loader
        # passes ctx_cont through in ORDERED_CTX_CONT_NAMES_V3 order (verified raw==loader).
        try:
            from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import ORDERED_CTX_CONT_NAMES_V3 as _CCN
            cc = list(_CCN)
        except Exception:
            cc = meta.get("ordered_ctx_cont_names") or None
        ds = EntryV10CtxDataset(parquet_path=Path(a.test_parquet), seq_len=96, allow_constant_labels=True,
                                enable_multi_tf=True, m5_prebuilt_path=Path(a.m5_prebuilt), multi_tf_seq_len=96,
                                per_tf_seq_lens={"H4": 96, "D1": 96})
        # shuffle=True is LOAD-BEARING: a consecutive batch false-flags slowly-varying features
        # (e.g. D1 regime is const within any short window but varies over the period). A shuffled
        # large batch samples across the whole period so only TRULY-constant features show std~0.
        # (Training batches are already shuffled → the trainer-callable is correct without this.)
        batch = next(iter(DataLoader(ds, batch_size=8192, shuffle=True, num_workers=4)))
        from gx1.contracts.signal_bridge_v3 import ORDERED_SNAP_FIELDS_V3 as _SNAP
        rep = assert_v10_batch_liveness(batch, ctx_cont_names=cc, snap_names=list(_SNAP), raise_on_fail=False)
        print(f"[V10] multi-TF atr-by-tf: {rep['multi_tf_atr']}")
        print(f"[V10] {'OK ✓ — nothing ignored' if rep['ok'] else 'ISSUES: ' + repr(rep['issues'])}")
        failed |= not rep["ok"]
    return 1 if (failed and a.strict) else 0


if __name__ == "__main__":
    raise SystemExit(_main())
