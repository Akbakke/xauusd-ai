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
    "_v1_atr_regime_id": "chained-index BUG → const=1 (basic_v1.py:726). Hygiene wave: fix the mask.",
    "smc_choch": "too sparse (0.1% nonzero) → 0 gain. Hygiene wave: decay to bars_since_choch.",
    # Multi-TF window-property (NOT a bug): D1 EMA-stack alignment can be const over a calm window:
    "d1:ema_stack_aligned_v2": "D1 regime can be stable over a test window → const there; alive in other TFs.",
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
        rep["atr_by_tf"][tf] = float(a.reshape(-1, a.shape[-1])[:, atr_idx].mean())
    # distinctness: ema50_dist series must not be ~identical across TFs (corr<0.98)
    if ema50_idx is not None and {"M5", "D1"} <= set(seq_by_tf):
        def ser(tf):
            return np.asarray(seq_by_tf[tf], np.float64)[:, :, ema50_idx].reshape(-1)
        for a, b in (("M5", "D1"), ("M5", "H1"), ("H1", "D1")):
            if len(ser(a)) == len(ser(b)):
                r = float(np.corrcoef(ser(a), ser(b))[0, 1])
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
    if "snap" in batch and snap_names is not None:
        issues += [f"snap:{d}" for d in _dead_cols(to_np(batch["snap"]), list(snap_names))]
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
    a = ap.parse_args()
    failed = False
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
        rep = assert_v10_batch_liveness(batch, ctx_cont_names=cc, raise_on_fail=False)
        print(f"[V10] multi-TF atr-by-tf: {rep['multi_tf_atr']}")
        print(f"[V10] {'OK ✓ — nothing ignored' if rep['ok'] else 'ISSUES: ' + repr(rep['issues'])}")
        failed |= not rep["ok"]
    return 1 if (failed and a.strict) else 0


if __name__ == "__main__":
    raise SystemExit(_main())
