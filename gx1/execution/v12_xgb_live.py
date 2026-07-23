#!/usr/bin/env python3
"""Fail-closed Exit-XGB live inference wrapper.

Loads one selected bundle with bundle-owned ordered feature and sanitizer
contracts, runs predict_proba per-session on augmented canonical_v3 rows, and
emits the seven-dimensional signal bridge consumed by V10/V3 downstream.

Inputs at inference time:
    - Augmented canonical_v3 DataFrame from
      gx1.execution.v12_ctx_augment_live.augment_canonical_v3()
      (must contain every feature in the selected bundle's exact contract)
    - The bar's session (ASIA / EU / OVERLAP / US) — taken from session_id
      column if not specified.

Outputs per bar:
    p_long, p_short, p_flat   ∈ [0,1], sum ≈ 1
    signal_bridge_v1          7-dim numpy array (consumed by V10 SEQ matrix)

The bundle has four session-specific heads trained on
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.xgb.multihead.xgb_multihead_model_v1 import XGBMultiheadModel, proba_to_signal_bridge_v1
from gx1.xgb.preprocess.xgb_input_sanitizer import XGBInputSanitizer
from gx1.time.session_detector import get_session_vectorized
from gx1.exits.training.thin_record_dataset import (
    V3_XGB_FEATURE_CONTRACT_FILENAME,
    V3_XGB_SANITIZER_CONFIG_FILENAME,
    build_v3_xgb_bridge_source_identity,
)

LOG = logging.getLogger("v12_xgb_live")

# Artifact selection names the only candidate. Admission then requires its
# recursive byte identity and exact model/contract/sanitizer feature order.
# predict() is fail-closed on missing features and NaN.
def _resolve_default_xgb_bundle() -> Path:
    # FG-1 fix (2026-06-06, rule 8): resolve the ACTIVE xgb bundle via the ONE selection
    # contract — NEVER a hardcoded literal (else live keeps serving the OLD xgb after a cement
    # flips the contract → silent train/serve skew). Mirrors v12_v3_live._resolve_default_v3_bundle.
    # Fail-closed: load_decision_artifact raises on missing/ambiguous/non-ACTIVE/non-XAU.
    from gx1_guards.artifacts import load_decision_artifact
    return Path(load_decision_artifact("xgb"))


BUNDLE_SANITIZER_FILENAME = V3_XGB_SANITIZER_CONFIG_FILENAME
BUNDLE_FEATURE_CONTRACT_FILENAME = V3_XGB_FEATURE_CONTRACT_FILENAME

SESSION_ID_TO_NAME = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}


def _require_ordered_xgb_feature_identity(
    *,
    model_feature_names: object,
    contract_features: list[str],
    sanitizer_features: list[str],
) -> None:
    if not isinstance(model_feature_names, list) or not model_feature_names:
        raise RuntimeError(
            "[XGB_CONTRACT_MISMATCH] bundle metadata lacks exact ordered "
            "feature names"
        )
    if model_feature_names != contract_features:
        mismatch = next(
            (
                index
                for index, (model_name, contract_name) in enumerate(
                    zip(model_feature_names, contract_features)
                )
                if model_name != contract_name
            ),
            min(len(model_feature_names), len(contract_features)),
        )
        raise RuntimeError(
            "[XGB_CONTRACT_MISMATCH] loaded model feature order differs "
            f"from contract at index={mismatch}; "
            f"model_count={len(model_feature_names)} "
            f"contract_count={len(contract_features)}"
        )
    if sanitizer_features != contract_features:
        raise RuntimeError(
            "[XGB_CONTRACT_MISMATCH] sanitizer feature order differs "
            "from the exact model contract"
        )


@dataclass
class XGBLiveInference:
    bundle_dir: Path
    sanitizer_config: Path
    feature_contract: Path
    _model: XGBMultiheadModel | None = field(default=None)
    _sanitizer: XGBInputSanitizer | None = field(default=None)
    _features: list[str] = field(default_factory=list)
    _runtime_identity: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        bundle_dir: Path,
    ) -> "XGBLiveInference":
        bundle_dir = Path(bundle_dir).expanduser()
        sanitizer_config = bundle_dir / BUNDLE_SANITIZER_FILENAME
        feature_contract = bundle_dir / BUNDLE_FEATURE_CONTRACT_FILENAME
        for label, path, kind in (
            ("bundle", bundle_dir, "directory"),
            ("sanitizer", sanitizer_config, "file"),
            ("feature contract", feature_contract, "file"),
        ):
            if not path.is_absolute() or path.is_symlink():
                raise RuntimeError(
                    f"XGB {label} path must be absolute and non-symlinked: {path}"
                )
            if (kind == "directory" and not path.is_dir()) or (
                kind == "file" and not path.is_file()
            ):
                raise RuntimeError(f"XGB {label} is missing: {path}")
        bundle_dir = bundle_dir.resolve()
        sanitizer_config = sanitizer_config.resolve()
        feature_contract = feature_contract.resolve()
        runtime_identity = build_v3_xgb_bridge_source_identity(
            bundle_dir=bundle_dir,
            feature_contract_path=feature_contract,
            sanitizer_config_path=sanitizer_config,
        )
        # Locate the joblib inside the bundle
        joblib_path = bundle_dir / "xgb_universal_multihead_v2.joblib"
        if not joblib_path.exists():
            raise FileNotFoundError(f"XGB bundle joblib not found: {joblib_path}")
        meta_path = bundle_dir / "xgb_universal_multihead_v2_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"XGB bundle meta not found: {meta_path}")

        LOG.info("loading exact Exit-XGB bundle: %s", bundle_dir.name)
        model = XGBMultiheadModel.load(str(joblib_path))
        sanitizer = XGBInputSanitizer.from_config(str(sanitizer_config))
        contract = json.loads(feature_contract.read_text())
        features = list(contract["features"])

        # 2026-06-02 fix (audit MEDIUM-#2): cross-check loaded model's trained
        # feature list against the contract. Without this, swapping bundle dir
        # A mismatched bundle could otherwise silently NaN-pad or truncate
        # inputs before inference.
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
        _require_ordered_xgb_feature_identity(
            model_feature_names=model_feature_names,
            contract_features=features,
            sanitizer_features=sanitizer.feature_list,
        )
        admitted_identity = build_v3_xgb_bridge_source_identity(
            bundle_dir=bundle_dir,
            feature_contract_path=feature_contract,
            sanitizer_config_path=sanitizer_config,
        )
        if admitted_identity != runtime_identity:
            raise RuntimeError(
                "[XGB_BUNDLE_CHANGED_DURING_LOAD] exact model or contract "
                "bytes changed between identity admission and deserialization"
            )
        LOG.info(
            "  XGB cross-check OK — model, sanitizer and contract agree on %d "
            "ordered features",
            len(features),
        )
        LOG.info(f"  features: {len(features)}  sessions: {list(model.sessions) if hasattr(model, 'sessions') else 'auto'}")

        return cls(
            bundle_dir=bundle_dir,
            sanitizer_config=sanitizer_config,
            feature_contract=feature_contract,
            _model=model,
            _sanitizer=sanitizer,
            _features=features,
            _runtime_identity=runtime_identity,
        )

    @classmethod
    def load_default(cls) -> "XGBLiveInference":
        return cls.load(_resolve_default_xgb_bundle())

    # ── prediction ────────────────────────────────────────────────────

    def predict(self, augmented_cv3_row: pd.DataFrame,
                session: str | None = None) -> dict[str, Any]:
        """Run Exit-XGB on a single-row (or multi-row) augmented canonical_v3
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
        elif "session_id" in augmented_cv3_row.columns and augmented_cv3_row["session_id"].notna().all():
            sid = augmented_cv3_row["session_id"].astype(int).to_numpy()
            sessions = np.array([SESSION_ID_TO_NAME.get(int(s), "ASIA") for s in sid], dtype=object)
        else:
            # session_id ABSENT or NaN -> derive the session from the bar timestamp (the
            # canonical, always-available source). 2026-06-15 LIVE-BLOCKER fix: at the
            # weekend market-open get_window() joins base28 (M1) cols onto the cv3 (M5)
            # window with reindex(method=None); when cv3 is momentarily ahead of base28
            # (the two prebuilts async-refresh independently), session_id is NaN on the
            # un-joined tail bar(s) and ``.astype(int)`` raised IntCastingNaNError on
            # EVERY poll -> ALL live entries blocked (crash-loop, no journal). Parity
            # verified 1.00000 over a full week: get_session_vectorized(ts) is bit-
            # identical to SESSION_ID_TO_NAME[session_id] (session is a pure function of
            # UTC time), so this fallback is train==serve-exact, never a degraded input.
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

        # Per-session predict_proba across the four exact session heads.
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
