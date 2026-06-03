#!/usr/bin/env python3
"""V12 XGB v7 (base80) live inference wrapper.

Loads the xgb_v7_base80 bundle (80 feats, isotonic-calibrated), runs
predict_proba per-session on augmented canonical_v3 rows, and emits the 7-dim
signal_bridge vector that V10/V3 consume downstream.

Inputs at inference time:
    - Augmented canonical_v3 DataFrame from
      gx1.execution.v12_ctx_augment_live.augment_canonical_v3()
      (must contain all 80 features listed in
       gx1/xgb/contracts/xgb_input_features_base80_v1.json)
    - The bar's session (ASIA / EU / OVERLAP / US) — taken from session_id
      column if not specified.

Outputs per bar:
    p_long, p_short, p_flat   ∈ [0,1], sum ≈ 1
    signal_bridge_v1          7-dim numpy array (consumed by V10 SEQ matrix)

The XGB v5 bundle has 4 session-specific heads — they were trained on
ASIA/EU/OVERLAP/US slices separately. Each prediction routes to the
correct head based on the bar's session_id.

Usage:
    xgb = XGBLiveInference.load_default()
    row = augmented_cv3.iloc[[-1]]                       # latest bar
    out = xgb.predict(row, session="US")                 # or session=None → auto
    print(out["p_long"], out["signal_bridge"])
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Force signal_bridge_v3 — same as inference_batch_v3_v1 default
os.environ.setdefault("GX1_SIGNAL_BRIDGE_VERSION", "3")

from gx1.xgb.multihead.xgb_multihead_model_v1 import XGBMultiheadModel, proba_to_signal_bridge_v1
from gx1.xgb.preprocess.xgb_input_sanitizer import XGBInputSanitizer
from gx1.time.session_detector import get_session_vectorized

LOG = logging.getLogger("v12_xgb_live")

# base80 is the ONE active XGB stack (80 feats incl. hour/dow), calibrated
# (isotonic, bundle-driven at load). The superseded 76-feat / 92-feat bundles
# were DELETED 2026-05-26 (cleanup) — base80 is the only XGB bundle on disk.
# predict() is fail-closed on missing feature / NaN.
DEFAULT_BUNDLE_DIR = Path(
    "/home/andre2/GX1_DATA/models/models/xgb_v7_base80_20260526T052210Z"
)
DEFAULT_SANITIZER_CONFIG = Path(
    "/home/andre2/src/GX1_ENGINE/gx1/xgb/contracts/xgb_input_sanitizer_base80_v1.json"
)
DEFAULT_FEATURE_CONTRACT = Path(
    "/home/andre2/src/GX1_ENGINE/gx1/xgb/contracts/xgb_input_features_base80_v1.json"
)

SESSION_ID_TO_NAME = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}


@dataclass
class XGBLiveInference:
    bundle_dir: Path
    sanitizer_config: Path
    feature_contract: Path
    _model: XGBMultiheadModel | None = field(default=None)
    _sanitizer: XGBInputSanitizer | None = field(default=None)
    _features: list[str] = field(default_factory=list)

    @classmethod
    def load(
        cls,
        bundle_dir: Path = DEFAULT_BUNDLE_DIR,
        sanitizer_config: Path = DEFAULT_SANITIZER_CONFIG,
        feature_contract: Path = DEFAULT_FEATURE_CONTRACT,
    ) -> "XGBLiveInference":
        bundle_dir = Path(bundle_dir)
        # Locate the joblib inside the bundle
        joblib_path = bundle_dir / "xgb_universal_multihead_v2.joblib"
        if not joblib_path.exists():
            raise FileNotFoundError(f"XGB bundle joblib not found: {joblib_path}")
        meta_path = bundle_dir / "xgb_universal_multihead_v2_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"XGB bundle meta not found: {meta_path}")

        LOG.info(f"loading XGB v7 base80 bundle: {bundle_dir.name}")
        model = XGBMultiheadModel.load(str(joblib_path))
        sanitizer = XGBInputSanitizer.from_config(str(sanitizer_config))
        contract = json.loads(feature_contract.read_text())
        features = list(contract["features"])

        # 2026-06-02 fix (audit MEDIUM-#2): cross-check loaded model's trained
        # feature list against the contract. Without this, swapping bundle dir
        # (e.g. older 76-feat bundle vs 80-feat contract) silently NaN-pads the
        # missing columns → 0-fill at inference → bad predictions.
        meta = json.loads(meta_path.read_text())
        model_feature_names = (
            meta.get("feature_names")
            or meta.get("features")
            or meta.get("input_feature_names")
            # B5 fix (2026-06-04): xgb v2 meta stores the list under feature_names_ordered.
            # Without this key the cross-check below silently no-op'd (fell to the warning),
            # leaving predict-time X[:, :expected] free to silently truncate a mismatched bundle.
            or meta.get("feature_names_ordered")
        )
        if model_feature_names:
            model_set = set(model_feature_names)
            contract_set = set(features)
            missing_in_model = contract_set - model_set
            extra_in_model = model_set - contract_set
            if missing_in_model or extra_in_model:
                raise RuntimeError(
                    f"[XGB_CONTRACT_MISMATCH] loaded model ({len(model_feature_names)} feats) "
                    f"vs contract ({len(features)} feats) disagree. "
                    f"missing_in_model={sorted(missing_in_model)[:5]} "
                    f"extra_in_model={sorted(extra_in_model)[:5]}. "
                    f"Refusing to silent-pad. Check bundle_dir + feature_contract paths."
                )
            LOG.info(f"  XGB cross-check OK — model + contract agree on {len(features)} features")
        else:
            LOG.warning(
                f"  XGB meta has no feature_names — cannot cross-check loaded model vs contract. "
                f"Continuing with contract-driven feature list of {len(features)} feats."
            )
        LOG.info(f"  features: {len(features)}  sessions: {list(model.sessions) if hasattr(model, 'sessions') else 'auto'}")

        return cls(
            bundle_dir=bundle_dir,
            sanitizer_config=sanitizer_config,
            feature_contract=feature_contract,
            _model=model,
            _sanitizer=sanitizer,
            _features=features,
        )

    @classmethod
    def load_default(cls) -> "XGBLiveInference":
        return cls.load()

    # ── prediction ────────────────────────────────────────────────────

    def predict(self, augmented_cv3_row: pd.DataFrame,
                session: str | None = None) -> dict[str, Any]:
        """Run XGB v5 on a single-row (or multi-row) augmented canonical_v3
        DataFrame. If `session` is None, infer it from the bar's session_id
        column (or from the timestamp index if no session_id present).

        Returns dict with:
            p_long, p_short, p_flat  (1-d arrays len = n_rows)
            signal_bridge_v1         (n_rows, 7) ndarray
            session                  per-row session name
        """
        if self._model is None or self._sanitizer is None:
            raise RuntimeError("XGBLiveInference not loaded — call .load() first")

        # Check feature coverage
        missing = [c for c in self._features if c not in augmented_cv3_row.columns]
        if missing:
            raise RuntimeError(f"missing {len(missing)} XGB features: {missing[:10]}")

        # Per-bar session resolution
        if session is not None:
            sessions = np.array([session] * len(augmented_cv3_row), dtype=object)
        elif "session_id" in augmented_cv3_row.columns:
            sid = augmented_cv3_row["session_id"].astype(int).to_numpy()
            sessions = np.array([SESSION_ID_TO_NAME.get(int(s), "ASIA") for s in sid], dtype=object)
        else:
            ts = augmented_cv3_row.index if isinstance(augmented_cv3_row.index, pd.DatetimeIndex) \
                 else pd.to_datetime(augmented_cv3_row["time"], utc=True)
            sessions = get_session_vectorized(ts).to_numpy(dtype=object)

        # Sanitize feature matrix. LIVE is fail-closed: respect the sanitizer's
        # hard_fail_on_nan (no silent NaN→0 fill). A NaN feature means corrupt
        # input — better to raise and skip the bar than feed a wrong signal_bridge
        # into V10/V3. (Batch/training paths may still fill.)
        df_feat = augmented_cv3_row[self._features].copy()
        x, _ = self._sanitizer.sanitize(
            df_feat, feature_list=self._features, allow_nan_fill=False, nan_fill_value=0.0,
        )
        df_san = pd.DataFrame(x, columns=self._features, index=df_feat.index)

        # Per-session predict_proba (XGB v5 has 4 session-specific heads)
        n = len(df_san)
        p_long = np.zeros(n, dtype=np.float32)
        p_short = np.zeros(n, dtype=np.float32)
        p_flat = np.zeros(n, dtype=np.float32)
        bridge = np.zeros((n, 7), dtype=np.float32)

        for sess_name in ("ASIA", "EU", "OVERLAP", "US"):
            idx = np.where(sessions == sess_name)[0]
            if idx.size == 0:
                continue
            probs = self._model.predict_proba(
                df_san.iloc[idx], session=sess_name, feature_list=self._features,
            )
            if hasattr(probs, "p_long"):
                pl = np.asarray(probs.p_long, dtype=np.float32)
                ps = np.asarray(probs.p_short, dtype=np.float32)
                pf = np.asarray(probs.p_flat, dtype=np.float32)
            else:
                pl = np.asarray(probs["p_long"], dtype=np.float32)
                ps = np.asarray(probs["p_short"], dtype=np.float32)
                pf = np.asarray(probs["p_flat"], dtype=np.float32)
            p_long[idx] = pl
            p_short[idx] = ps
            p_flat[idx] = pf
            bridge[idx] = proba_to_signal_bridge_v1(np.column_stack([pl, ps, pf])).astype(np.float32)

        return {
            "p_long": p_long,
            "p_short": p_short,
            "p_flat": p_flat,
            "signal_bridge_v1": bridge,
            "session": sessions,
        }
