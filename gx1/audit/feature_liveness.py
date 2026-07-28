"""Cross-chain FEATURE-LIVENESS audit — "NOTHING gets silently ignored" (user vedtak 2026-06-06).

A consolidated, ALWAYS-RUN check that every input feature the chain consumes is ALIVE (non-constant)
and that the multi-TF block is intact (all 5 TFs present, alive, DISTINCT resolutions). It complements
(does NOT duplicate) the model-native Entry training contracts or the separate
Exit-IQL state-vector coverage guard.

The active V10 batch gate is model-native only: its signal surface must be the
exact 513-field contract, its 142 continuous-context inputs must be present,
and every value on both surfaces must be finite and non-constant. It never
consults the historical ``KNOWN_ALLOWED_DEAD`` diagnostic allowlist. That
allowlist remains only for explicitly legacy IQL/XGB hygiene readers and cannot
turn a retired signal surface into an authoritative PASS.

Run automatically: the V10 trainer calls assert_v10_batch_liveness() at post-export.
Run manually:  python -m gx1.audit.feature_liveness --v10-bundle <dir> --test-parquet <pq> --m5-prebuilt <pq>
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gx1.contracts.entry_model_native_signal_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    MODEL_NATIVE_BASE_SIGNAL_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
    ordered_model_native_signal_fields,
    require_model_native_signal_contract,
)
from gx1.execution.v12_state_from_prebuilt import (
    PREBUILT_PAIR_MANIFEST_PATH,
    PREBUILT_PAIR_ROOT,
    read_prebuilt_pair_manifest,
    verify_prebuilt_pair,
)

DEAD_STD = 1e-4   # std below this over a real batch = constant = "ignored input"

# Escalation resolver: given (surface, field name), return that field's
# (std, nunique) over its COMPLETE declared population, or None when the caller
# cannot take that measurement. A sample can only ever raise the suspicion of
# deadness; the verdict belongs to the full population.
PopulationStats = Callable[[str, str], Optional[Tuple[float, int]]]


def _population_alive(std: float, nunique: int) -> bool:
    """Is a field alive on its complete declared population?

    Deadness has two scales, so the verdict needs both of this owner's existing
    conventions and no third constant:

    ``DEAD_STD`` settles sparse impulses whose magnitude is O(1). A binary
    regime-change flag firing on 0.024% of rows has std 1.5e-02 — far above the
    bar — yet is absent from almost every sample drawn from it.

    ``LIVE_TAIL_REF_MIN_NUNIQUE`` settles richly-varying fields whose natural
    magnitude is small. A readiness score taking 137,844 distinct values inside
    [0, 0.0044] has std 5.6e-05; it is the opposite of constant, and failing it
    on ``DEAD_STD`` measures scale rather than liveness.

    A genuinely dead field — one value for every row — has nunique 1 and std 0
    and fails both. That is the only case this gate exists to catch.
    """
    return bool(std >= DEAD_STD or nunique >= LIVE_TAIL_REF_MIN_NUNIQUE)

# ── Legacy diagnostic structural/known-dead allowlist ──────────────────────────────────────
# Format: bare name OR "tf:name" for a per-TF feature. This is available to
# historical IQL/XGB hygiene readers only. The model-native entry gate below
# explicitly disables it for all 513 signal and 142 ctx-cont inputs.
KNOWN_ALLOWED_DEAD: Dict[str, str] = {
    "vol_pct_m5_1yr": "1-year vol-percentile not computed → pinned 0.5. Hygiene wave: compute or drop.",
    "vol_pct_h1_1yr": "ditto (pinned 0.5).",
    # Ultra-sparse but ALIVE (91 nonzero / 396,681 rows): false-flags as dead below
    # DEAD_STD on typical sample sizes — the documented slow-varying D1 class.
    "d1_regime_changed_flag_v3": "ultra-sparse impulse flag (0.023% nonzero) — alive on full scan 2026-07-05; sibling bars_since_d1_regime_change_v3 carries the signal.",
    # Benign by construction — XGB is SESSION-HEADED so session feats are const within a head:
    "session_id": "0 XGB gain by construction (session-headed model).",
    "is_ASIA": "ditto.", "session_change_flag": "ditto.", "session_tradable": "ditto.",
    "minutes_since_session_open": "ditto.", "minutes_to_next_session_boundary": "ditto.",
    "_v1_is_EU": "legacy baked session one-hot; 0 XGB gain by construction in session-headed model.",
    "_v1_is_US": "legacy baked session one-hot; 0 XGB gain by construction in session-headed model.",
    # Known bugs/gaps tracked in the hygiene wave (NOT to be silently forgotten):
    "_v1_atr_regime_id": "BUG-MASK: chained-index BUG → const=1 (basic_v1.py:726). Fix EXISTS behind "
                         "GX1_ATR_REGIME_FIX=1 (bf4a6abd, default OFF — live builds still emit const). REMOVE this "
                         "entry at the first rebuild that enables the gate, or it will mask the then-alive feature.",
    "smc_choch": "BUG-MASK (remove when fixed): too sparse (0.1% nonzero) → 0 gain. Hygiene wave: decay to bars_since_choch.",
    # Multi-TF window-property (NOT a bug): D1 EMA-stack alignment can be const over a calm window:
    "d1:ema_stack_aligned_v2": "D1 regime can be stable over a test window → const there; alive in other TFs.",
    # Provenance one-hot — constant by construction in the historical Exit-IQL
    # training substrate. Harmless dead slot; drop at the next Exit rebuild.
    "decision_reason_v2_inference_batch": "Exit-IQL: same const provenance one-hot. Drop at rebuild.",
}

MULTI_TF_NAMES: Sequence[str] = ()
try:
    # Names are only used to label a dead column, so carrying the widest
    # declared contract keeps every V2 index valid (V3's first 25 are V2) while
    # naming V3's extra columns instead of falling back to "[index]".
    from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V3 as _MTF
    MULTI_TF_NAMES = tuple(_MTF)
except Exception:  # pragma: no cover
    pass


class FeatureLivenessError(RuntimeError):
    """A contracted feature failed finiteness/liveness or input identity."""


def _dead_cols(
    arr: np.ndarray,
    names: Sequence[str],
    tf: str = "",
    *,
    allow_known_dead: bool = True,
    population_stats: Optional[PopulationStats] = None,
    surface: str = "",
) -> List[str]:
    """Return non-finite or constant columns without an allowed legacy exemption.

    ``allow_known_dead=False`` is load-bearing for the model-native input gate:
    no structural or legacy exemption is permitted on a contracted model input.

    ``population_stats`` replaces exemption with measurement. A sample below
    ``DEAD_STD`` is a suspicion, not a verdict: escalate that one field to its
    complete declared population and let ``_population_alive`` decide. Without a
    resolver the sample verdict stands, so legacy readers are unaffected.
    """
    arr = np.asarray(arr, dtype=np.float64)
    flat = arr.reshape(-1, arr.shape[-1])
    out: List[str] = []
    for j in range(flat.shape[1]):
        nm = names[j] if j < len(names) else f"[{j}]"
        qualified = f"{tf+':' if tf else ''}{nm}"
        finite = np.isfinite(flat[:, j])
        if not bool(finite.all()):
            out.append(f"{qualified} (nonfinite={int((~finite).sum())})")
            continue
        std = float(flat[:, j].std())
        if std >= DEAD_STD:
            continue
        if allow_known_dead and (
            nm in KNOWN_ALLOWED_DEAD or (tf and f"{tf}:{nm}" in KNOWN_ALLOWED_DEAD)
        ):
            continue
        if population_stats is not None:
            measured = population_stats(surface or tf, str(nm))
            if measured is not None:
                pop_std, pop_nunique = float(measured[0]), int(measured[1])
                if _population_alive(pop_std, pop_nunique):
                    continue
                out.append(
                    f"{qualified} (sample_std={std:.1e} population_std={pop_std:.1e} "
                    f"population_nunique={pop_nunique})"
                )
                continue
        out.append(f"{qualified} (std={std:.1e})")
    return out


def _surface_contract_issues(
    value: object,
    names: Sequence[str],
    *,
    surface: str,
    expected_dim: int,
    population_stats: Optional[PopulationStats] = None,
) -> List[str]:
    """Validate one authoritative model-native numeric input surface."""

    issues: List[str] = []
    normalized_names = [str(name).strip() for name in names]
    try:
        arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        arr = np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        return [f"{surface}: non-numeric input ({exc})"]
    if arr.ndim < 2:
        return [f"{surface}: BAD SHAPE {arr.shape} (expected final dim {expected_dim})"]
    if arr.shape[-1] != expected_dim:
        issues.append(f"{surface}: width={arr.shape[-1]} expected={expected_dim}")
    if len(normalized_names) != expected_dim:
        issues.append(
            f"{surface}: name_count={len(normalized_names)} expected={expected_dim}"
        )
    if len(set(normalized_names)) != len(normalized_names):
        issues.append(f"{surface}: duplicate field names")
    if any(not name for name in normalized_names):
        issues.append(f"{surface}: blank field names")
    forbidden = sorted(set(normalized_names) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS))
    if forbidden:
        issues.append(f"{surface}: forbidden legacy bridge fields={forbidden}")
    if arr.shape[-1] == len(normalized_names):
        issues.extend(
            f"{surface}:{detail}"
            for detail in _dead_cols(
                arr,
                normalized_names,
                allow_known_dead=False,
                population_stats=population_stats,
                surface=surface,
            )
        )
    return issues


def _model_native_signal_contract_issues(names: Sequence[str]) -> List[str]:
    """Prove that ``names`` is exactly base34 + manifest-owned selected479."""

    normalized = tuple(str(name).strip() for name in names)
    issues: List[str] = []
    if len(normalized) != MODEL_NATIVE_SIGNAL_DIM:
        issues.append(
            f"signal contract width={len(normalized)} expected={MODEL_NATIVE_SIGNAL_DIM}"
        )
    forbidden = sorted(set(normalized) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS))
    if forbidden:
        issues.append(f"signal contract contains forbidden legacy bridge fields={forbidden}")
    if len(normalized) == MODEL_NATIVE_SIGNAL_DIM:
        try:
            expected = ordered_model_native_signal_fields(
                normalized[MODEL_NATIVE_BASE_SIGNAL_DIM:]
            )
        except RuntimeError as exc:
            issues.append(str(exc))
        else:
            if normalized != expected:
                issues.append("signal contract base34/order mismatch")
    return issues


def check_multi_tf_integrity(
    seq_by_tf: Dict[str, np.ndarray],
    *,
    allow_known_dead: bool = True,
) -> Dict[str, object]:
    """All 5 TFs present, correctly shaped, live and at distinct resolutions.

    seq_by_tf: {"M5": (B,L,25), "M15": ..., "H1": ..., "H4": ..., "D1": ...}
    Returns a report dict; the caller decides to raise on `new_dead`/`missing`/`duplicate`.
    ``allow_known_dead`` exists for historical diagnostics only; the active
    model-native gate always passes ``False``.
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
        rep["new_dead"].extend(
            _dead_cols(
                a,
                names,
                tf=tf.lower(),
                allow_known_dead=allow_known_dead,
            )
        )
        # mask zero-padded warmup rows (atr==0) so the ATR-scaling sanity isn't deflated on a skewed batch
        _atr_col = a.reshape(-1, a.shape[-1])[:, atr_idx]
        _nz = _atr_col[_atr_col > 0]
        rep["atr_by_tf"][tf] = float(_nz.mean()) if _nz.size else 0.0
    # distinctness: ema50_dist series must not be ~identical across TFs (corr<0.98)
    if ema50_idx is not None and {"M5", "D1"} <= set(seq_by_tf):
        def ser(tf):
            return np.asarray(seq_by_tf[tf], np.float64)[:, :, ema50_idx].reshape(-1)
        for a, b in (("M5", "D1"), ("M5", "H1"), ("H1", "D1")):
            sa, sb = ser(a), ser(b)
            n = min(len(sa), len(sb))  # always run correlation on shared rows
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
                              snap_names: Optional[Sequence[str]] = None, raise_on_fail: bool = True,
                              population_stats: Optional[PopulationStats] = None) -> dict:
    """Authoritative post-export gate for exact model-native V10 inputs.

    The gate never infers names, dimensions, bridge compatibility, or constant
    exemptions. Missing surfaces and retired signal contracts are hard failures
    even when ``raise_on_fail=False``; that mode only returns the FAIL report.

    ``population_stats`` lets the caller escalate a sample-flagged field to its
    complete declared population, which is the only place a deadness verdict is
    valid. The caller owns access to its own data; this gate owns the verdict.
    """
    def to_np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    issues: List[str] = []
    if snap_names is None:
        issues.append("signal: exact ordered field names missing")
    else:
        issues.extend(_model_native_signal_contract_issues(snap_names))
    if "seq_x" not in batch:
        issues.append("signal_sequence: seq_x surface missing")
    elif snap_names is not None:
        issues.extend(
            _surface_contract_issues(
                batch["seq_x"],
                snap_names,
                surface="signal_sequence",
                expected_dim=MODEL_NATIVE_SIGNAL_DIM,
                population_stats=population_stats,
            )
        )
    if "snap_x" not in batch:
        issues.append("signal: snap_x surface missing")
    elif snap_names is not None:
        issues.extend(
            _surface_contract_issues(
                batch["snap_x"],
                snap_names,
                surface="signal",
                expected_dim=MODEL_NATIVE_SIGNAL_DIM,
                population_stats=population_stats,
            )
        )
    if ctx_cont_names is None:
        issues.append("ctx_cont: exact ordered field names missing")
    if "ctx_cont" not in batch:
        issues.append("ctx_cont: surface missing")
    elif ctx_cont_names is not None:
        issues.extend(
            _surface_contract_issues(
                batch["ctx_cont"],
                ctx_cont_names,
                surface="ctx_cont",
                expected_dim=142,
                population_stats=population_stats,
            )
        )
    seq_by_tf = {k.replace("seq_", "").upper(): to_np(batch[k])
                 for k in ("seq_m5", "seq_m15", "seq_h1", "seq_h4", "seq_d1") if k in batch}
    mtf = (
        check_multi_tf_integrity(seq_by_tf, allow_known_dead=False)
        if seq_by_tf
        else {
            "missing": ["ALL — no multi-TF in batch"],
            "new_dead": [],
            "duplicate": [],
        }
    )
    issues += [f"multi_tf.{k}={v}" for k in ("missing", "new_dead", "duplicate") for v in (mtf.get(k) or [])]
    rep = {
        "ok": not issues,
        "authoritative": True,
        "contract": "model_native_seq513",
        "issues": issues,
        "multi_tf_atr": mtf.get("atr_by_tf", {}),
    }
    if issues:
        msg = "[FEATURE_LIVENESS_FAIL] features/dependencies went silently dead (not on allowlist):\n  - " + "\n  - ".join(issues)
        if raise_on_fail:
            raise FeatureLivenessError(msg)
        print(msg, file=sys.stderr)
    return rep


def audit_iql_state_liveness(X, names, *, role: str = "iql-state", raise_on_fail: bool = False) -> dict:
    """Liveness check for an already-built Exit-IQL state matrix.

    This reusable enforcement primitive lets Exit-IQL builds pass their actual
    ``(X, names)`` values and fail loudly on any state feature that is constant
    or zero and not on ``KNOWN_ALLOWED_DEAD``.

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


LIVE_TAIL_ALLOWED_CONST: Dict[str, str] = {}
LIVE_TAIL_REF_MIN_NUNIQUE = 4   # was-varying threshold: ref-window nunique >= this => freeze when tail==1
LIVE_PAIR_MANIFEST = str(PREBUILT_PAIR_MANIFEST_PATH)
LIVE_PAIR_GENERATION_ROOT = str(PREBUILT_PAIR_ROOT)


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
    "2026-06-17": "OOM-reboot (full phase6-gate tippet 58G-cap) — ~15-16min hull 14:27-14:43 (collector/daemon starved under RAM-thrash, mistet OANDA M1-vindu). Historisk mutable freshener er fjernet; gapet krever ny immutable native-M1 source-generation. BACKFILL-KANDIDAT.",
    "2026-06-27": "Live-stack nede 26. jun→5. jul (logs/-dir slettet 29. jun + reboot 2. jul → 209/STDOUT). REPARERT 2026-07-05: OANDA-backfill jun26→jul3 + cv3/BASE34 trunker+re-append (backup .bak_hole_repair_20260705). Kun helg/vedlikeholds-gap gjenstår i uken.",
    "2026-07-03": "US Independence Day (4. juli observert): tidlig stenging 16:59Z fredag — OANDA har ingen candles 17:00Z fre → 21:00Z søn. Markeds-kalender, ikke utfall.",
}


def check_live_continuity(
    tail_days: int = 10,
    fresh_fail_hours: int = 48,
    pair_manifest: str = LIVE_PAIR_MANIFEST,
    generation_root: str = LIVE_PAIR_GENERATION_ROOT,
    raise_on_fail: bool = False,
) -> dict:
    """Rule-9 CONTINUITY check (user-direktiv 2026-06-12: «ALLTID oppdatert, INGEN hull, nøyaktig på
    hver M1 (exit) og M5 (entry)»). Skanner cv3 (M5) + BASE34 (M1) for grid-hull, klassifisert mot
    helg / daglig 21-22Z-pause / US-helligdager / tick-tomme minutter (<=10 min) / KNOWN_DATA_GAPS.
    Et UKJENT hull NYERE enn fresh_fail_hours = FAIL (collector/daemon-utfall pågår eller nettopp
    skjedd); eldre ukjente hull rapporteres for backfill. Sjekker også ferskhet (cutoff-alder)."""
    import pandas as pd
    out: dict = {"ok": True, "fresh_gaps": [], "stale_gaps": [], "freshness_min": {}}
    now = pd.Timestamp.now(tz="UTC")
    pair = read_prebuilt_pair_manifest(
        Path(pair_manifest),
        generation_root=Path(generation_root),
    )
    verify_prebuilt_pair(pair)
    frames = []
    cv3 = pd.read_parquet(pair.canonical_v3.parquet_path, columns=["time"])
    frames.append(("CV3-M5", pd.DatetimeIndex(pd.to_datetime(cv3["time"], utc=True)).sort_values(), 5))
    b34 = pd.read_parquet(pair.base28.parquet_path, columns=[])
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
            if s.dayofweek == 4 and 2300 <= m <= 3200:
                continue  # helg
            if s.hour in (20, 21) and 50 <= m <= 75:
                continue  # daglig pause
            if m <= 10:
                continue  # tick-tomt
            keys = {(s + pd.Timedelta(days=o)).strftime("%Y-%m-%d") for o in (0, 1)}
            if keys & US_MARKET_HOLIDAYS:
                continue
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


def check_live_prebuilt_tail(
    tail_days: int = 5,
    ref_days: int = 30,
    pair_manifest: str = LIVE_PAIR_MANIFEST,
    generation_root: str = LIVE_PAIR_GENERATION_ROOT,
    raise_on_fail: bool = False,
) -> dict:
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
    pair = read_prebuilt_pair_manifest(
        Path(pair_manifest),
        generation_root=Path(generation_root),
    )
    verify_prebuilt_pair(pair)
    b34 = pd.read_parquet(pair.base28.parquet_path)
    b34.index = pd.to_datetime(b34.index, utc=True)
    frames.append(("BASE34", b34))
    cv3 = pd.read_parquet(pair.canonical_v3.parquet_path)
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


def audit_xgb_gain(bundle_dir: str) -> List[str]:
    """Return exact bundle features with zero gain in every session head."""
    from gx1.execution.v12_xgb_live import XGBLiveInference

    admitted = XGBLiveInference.load(Path(bundle_dir))
    feats = admitted._features
    m = admitted._model
    if m is None:
        raise FeatureLivenessError("XGB bundle admitted without a loaded model")
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
        dead = audit_xgb_gain(a.xgb_bundle)
        print(f"[XGB] new-dead (0 gain, off allowlist): {dead or 'NONE ✓'}")
        failed |= bool(dead)
    if a.v10_bundle and a.test_parquet and a.m5_prebuilt:
        import os
        os.environ.setdefault("GX1_REGIME_V4", "1")
        os.environ.setdefault("GX1_TREND_REGIME_FROM_D1", "1")
        from torch.utils.data import DataLoader
        from gx1.models.entry_v10.entry_v10_ctx_train_v3 import EntryV10CtxDataset
        from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
        meta = load_entry_v10_ctx_bundle(bundle_dir=Path(a.v10_bundle), device="cpu").metadata
        signal_contract = meta.get("model_native_signal_contract")
        require_model_native_signal_contract(
            signal_contract,
            context="FEATURE_LIVENESS_CLI",
        )
        snap_names = list(signal_contract["fields"])
        cc = list(meta.get("ordered_ctx_cont_names") or ())
        if len(cc) != 142:
            raise FeatureLivenessError(
                "[FEATURE_LIVENESS_CLI_CTX_CONT_CONTRACT_INVALID] "
                f"ordered_ctx_cont_names={len(cc)} expected=142"
            )
        ds = EntryV10CtxDataset(
            parquet_path=Path(a.test_parquet),
            seq_len=96,
            m5_prebuilt_path=Path(a.m5_prebuilt),
            multi_tf_seq_len=96,
            per_tf_seq_lens={"H4": 96, "D1": 96},
        )
        # shuffle=True is LOAD-BEARING: a consecutive batch false-flags slowly-varying features
        # (e.g. D1 regime is const within any short window but varies over the period). A shuffled
        # large batch samples across the whole period so only TRULY-constant features show std~0.
        # (Training batches are already shuffled → the trainer-callable is correct without this.)
        batch = next(iter(DataLoader(ds, batch_size=8192, shuffle=True, num_workers=4)))
        rep = assert_v10_batch_liveness(
            batch,
            ctx_cont_names=cc,
            snap_names=snap_names,
            raise_on_fail=False,
        )
        print(f"[V10] multi-TF atr-by-tf: {rep['multi_tf_atr']}")
        print(f"[V10] {'OK ✓ — nothing ignored' if rep['ok'] else 'ISSUES: ' + repr(rep['issues'])}")
        failed |= not rep["ok"]
    return 1 if (failed and a.strict) else 0


if __name__ == "__main__":
    raise SystemExit(_main())
