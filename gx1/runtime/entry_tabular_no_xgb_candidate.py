"""Runtime helpers for the tabular no-XGB Entry candidate.

This module is intentionally small and fail-closed. It builds the exact feature
matrix used by the no-XGB research lane: snap fields after the 7-field XGB
signal bridge, followed by ctx_cont and ctx_cat. Any XGB-derived/probability
feature name in the candidate contract is treated as a hard failure.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from gx1.audit.entry_transformer_feature_audit import _stack_list_column
from gx1.contracts.signal_bridge_v3 import ORDERED_SEQ_FIELDS_V3


XGB_SIGNAL_FIELD_COUNT = 7
EXCLUDED_XGB_FIELDS = [
    "p_long",
    "p_short",
    "p_flat",
    "p_hat",
    "uncertainty_score",
    "margin_top1_top2",
    "entropy",
]
SUSPICIOUS_NAME_PARTS = (
    "xgb",
    "p_long",
    "p_short",
    "p_flat",
    "prob",
    "signal_bridge",
    "uncertainty_score",
    "margin_top1_top2",
    "entropy",
)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=json_default, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def feature_contract_hash(feature_names: list[str]) -> str:
    return stable_json_hash({"feature_names": feature_names, "version": "entry_tabular_no_xgb_v1"})


def assert_no_xgb_feature_names(feature_names: list[str]) -> None:
    suspicious = []
    for name in feature_names:
        low = str(name).lower()
        if any(part in low for part in SUSPICIOUS_NAME_PARTS):
            suspicious.append(str(name))
    if suspicious:
        raise RuntimeError(f"NO_XGB_FEATURE_CONTRACT_VIOLATION: {suspicious[:25]}")


def _manifest_for_split(parquet_path: Path) -> Path | None:
    candidate = parquet_path.with_suffix(".manifest.json")
    return candidate if candidate.exists() else None


def _ctx_names_from_manifest(parquet_path: Path) -> tuple[list[str], list[str]]:
    manifest_path = _manifest_for_split(parquet_path)
    if manifest_path is None:
        return [], []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_contract = manifest.get("feature_contract") if isinstance(manifest.get("feature_contract"), dict) else {}
    ctx_contract = None
    if isinstance(manifest.get("extra"), dict):
        ctx_contract = manifest["extra"].get("ctx_contract")
    if isinstance(manifest.get("build_metadata"), dict):
        ctx_contract = ctx_contract or manifest["build_metadata"].get("ctx_contract")
    if not isinstance(ctx_contract, dict):
        ctx_contract = feature_contract
    ctx_cont_names = list(ctx_contract.get("ctx_cont_names") or feature_contract.get("ctx_cont_names") or [])
    ctx_cat_names = list(ctx_contract.get("ctx_cat_names") or feature_contract.get("ctx_cat_names") or [])
    return ctx_cont_names, ctx_cat_names


def selected_feature_names(parquet_path: Path) -> list[str]:
    ctx_cont_names, ctx_cat_names = _ctx_names_from_manifest(parquet_path)
    snap_names = list(ORDERED_SEQ_FIELDS_V3[XGB_SIGNAL_FIELD_COUNT:])
    feature_names = [f"snap.{name}" for name in snap_names]
    if ctx_cont_names:
        feature_names.extend(f"ctx_cont.{name}" for name in ctx_cont_names)
    if ctx_cat_names:
        feature_names.extend(f"ctx_cat.{name}" for name in ctx_cat_names)
    assert_no_xgb_feature_names(feature_names)
    return feature_names


def build_feature_matrix(
    parquet_path: Path,
    *,
    expected_feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    df = pd.read_parquet(parquet_path, columns=["snap", "ctx_cont", "ctx_cat"])
    snap = _stack_list_column(df["snap"], np.float32)[:, XGB_SIGNAL_FIELD_COUNT:]
    ctx_cont = _stack_list_column(df["ctx_cont"], np.float32)
    ctx_cat = _stack_list_column(df["ctx_cat"], np.int64).astype(np.float32)
    x = np.concatenate([snap, ctx_cont, ctx_cat], axis=1).astype(np.float32, copy=False)

    feature_names = selected_feature_names(parquet_path)
    if len(feature_names) != x.shape[1]:
        snap_names = [f"snap.{name}" for name in ORDERED_SEQ_FIELDS_V3[XGB_SIGNAL_FIELD_COUNT:]]
        ctx_cont_names = [f"ctx_cont.{i}" for i in range(ctx_cont.shape[1])]
        ctx_cat_names = [f"ctx_cat.{i}" for i in range(ctx_cat.shape[1])]
        feature_names = snap_names + ctx_cont_names + ctx_cat_names
    assert_no_xgb_feature_names(feature_names)
    if x.shape[1] != len(feature_names):
        raise RuntimeError(f"FEATURE_SHAPE_MISMATCH: x={x.shape[1]} names={len(feature_names)}")
    if expected_feature_names is not None and list(expected_feature_names) != feature_names:
        raise RuntimeError("FEATURE_NAME_ORDER_MISMATCH")
    return x, feature_names


def load_model(model_path: Path) -> Any:
    return joblib.load(model_path)


def predict_proba(model: Any, x: np.ndarray) -> np.ndarray:
    probs = np.asarray(model.predict_proba(x), dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise RuntimeError(f"UNEXPECTED_PROBABILITY_SHAPE: {probs.shape}")
    row_sum = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, np.maximum(row_sum, 1e-12))
    if not np.isfinite(probs).all():
        raise RuntimeError("NONFINITE_CANDIDATE_PROBABILITIES")
    return probs


def score_probabilities(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chosen_side = np.where(probs[:, 0] >= probs[:, 1], 0, 1).astype(np.int64)
    chosen_prob = np.maximum(probs[:, 0], probs[:, 1]).astype(np.float64, copy=False)
    score = chosen_prob - probs[:, 2]
    return chosen_side, chosen_prob, score.astype(np.float64, copy=False)


def feature_vector_from_live_inputs(
    *,
    snap_x: np.ndarray,
    ctx_cont: np.ndarray,
    ctx_cat: np.ndarray,
    expected_feature_names: list[str],
) -> np.ndarray:
    snap = np.asarray(snap_x, dtype=np.float32)
    cont = np.asarray(ctx_cont, dtype=np.float32)
    cat = np.asarray(ctx_cat, dtype=np.float32)
    if snap.ndim == 2:
        if snap.shape[0] != 1:
            raise RuntimeError(f"LIVE_SNAP_BATCH_UNSUPPORTED: {snap.shape}")
        snap = snap[0]
    if cont.ndim == 2:
        if cont.shape[0] != 1:
            raise RuntimeError(f"LIVE_CTX_CONT_BATCH_UNSUPPORTED: {cont.shape}")
        cont = cont[0]
    if cat.ndim == 2:
        if cat.shape[0] != 1:
            raise RuntimeError(f"LIVE_CTX_CAT_BATCH_UNSUPPORTED: {cat.shape}")
        cat = cat[0]
    if snap.ndim != 1 or cont.ndim != 1 or cat.ndim != 1:
        raise RuntimeError(f"LIVE_FEATURE_RANK_MISMATCH: snap={snap.shape} cont={cont.shape} cat={cat.shape}")
    if snap.shape[0] <= XGB_SIGNAL_FIELD_COUNT:
        raise RuntimeError(f"LIVE_SNAP_DIM_TOO_SMALL: {snap.shape[0]}")
    x = np.concatenate([snap[XGB_SIGNAL_FIELD_COUNT:], cont, cat], axis=0).astype(np.float32, copy=False)
    if x.shape[0] != len(expected_feature_names):
        raise RuntimeError(f"LIVE_FEATURE_DIM_MISMATCH: x={x.shape[0]} names={len(expected_feature_names)}")
    return x.reshape(1, -1)


@dataclass(frozen=True)
class EntryTabularNoXGBShadowDecision:
    action: str
    side: str
    score: float
    score_threshold: float
    chosen_prob: float
    p_long: float
    p_short: float
    p_flat: float
    feature_contract_hash: str
    candidate_id: str
    manifest_path: str


class EntryTabularNoXGBShadow:
    """Manifest-resolved, shadow-only tabular Entry scorer."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        candidate_id: str,
        model: Any,
        feature_names: list[str],
        score_threshold: float,
        min_direction_prob: float,
        feature_hash: str,
    ) -> None:
        self.manifest_path = manifest_path
        self.candidate_id = candidate_id
        self.model = model
        self.feature_names = list(feature_names)
        self.score_threshold = float(score_threshold)
        self.min_direction_prob = float(min_direction_prob)
        self.feature_hash = feature_hash

    @classmethod
    def load(cls, *, manifest_path: Path, score_threshold: float) -> "EntryTabularNoXGBShadow":
        manifest_path = Path(manifest_path).expanduser().resolve()
        package_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("package_status") != "NOT_PROMOTED_NOT_LIVE_READY":
            raise RuntimeError("TABULAR_NO_XGB_SHADOW_REQUIRES_NOT_PROMOTED_PACKAGE")
        if bool(manifest.get("promotion_allowed")) or bool(manifest.get("live_ready")):
            raise RuntimeError("TABULAR_NO_XGB_SHADOW_PACKAGE_UNEXPECTEDLY_LIVE_READY")

        feature_manifest = json.loads((package_dir / "feature_manifest.json").read_text(encoding="utf-8"))
        policy_config = json.loads((package_dir / "policy_config.json").read_text(encoding="utf-8"))
        feature_names = [str(item["name"]) for item in feature_manifest.get("features", [])]
        if not feature_names:
            raise RuntimeError("TABULAR_NO_XGB_SHADOW_EMPTY_FEATURE_MANIFEST")
        assert_no_xgb_feature_names(feature_names)
        observed_feature_hash = feature_contract_hash(feature_names)
        expected_feature_hash = str(feature_manifest.get("feature_contract_hash"))
        if observed_feature_hash != expected_feature_hash:
            raise RuntimeError("TABULAR_NO_XGB_SHADOW_FEATURE_HASH_MISMATCH")
        if manifest["feature_contract"]["feature_contract_hash"] != observed_feature_hash:
            raise RuntimeError("TABULAR_NO_XGB_SHADOW_CANDIDATE_FEATURE_HASH_MISMATCH")

        model_path = Path(manifest["model"]["primary_model_path"])
        if sha256_file(model_path) != str(manifest["model"]["primary_model_sha256"]):
            raise RuntimeError("TABULAR_NO_XGB_SHADOW_MODEL_HASH_MISMATCH")
        model = load_model(model_path)
        n_model_features = int(getattr(model, "n_features_in_", len(feature_names)))
        if n_model_features != len(feature_names):
            raise RuntimeError(f"TABULAR_NO_XGB_SHADOW_MODEL_FEATURE_MISMATCH: {n_model_features}!={len(feature_names)}")
        if not np.isfinite(float(score_threshold)):
            raise RuntimeError("TABULAR_NO_XGB_SHADOW_THRESHOLD_NONFINITE")
        return cls(
            manifest_path=manifest_path,
            candidate_id=str(manifest["candidate_id"]),
            model=model,
            feature_names=feature_names,
            score_threshold=float(score_threshold),
            min_direction_prob=float(policy_config.get("min_direction_prob", 0.0)),
            feature_hash=observed_feature_hash,
        )

    def predict_from_live_inputs(
        self,
        *,
        snap_x: np.ndarray,
        ctx_cont: np.ndarray,
        ctx_cat: np.ndarray,
    ) -> EntryTabularNoXGBShadowDecision:
        x = feature_vector_from_live_inputs(
            snap_x=snap_x,
            ctx_cont=ctx_cont,
            ctx_cat=ctx_cat,
            expected_feature_names=self.feature_names,
        )
        probs = predict_proba(self.model, x)
        chosen_side, chosen_prob, score = score_probabilities(probs)
        side_id = int(chosen_side[0])
        score_value = float(score[0])
        chosen_prob_value = float(chosen_prob[0])
        side = "LONG" if side_id == 0 else "SHORT"
        action = "SKIP"
        if score_value >= self.score_threshold and chosen_prob_value >= self.min_direction_prob:
            action = "TAKE_LONG_NOW" if side_id == 0 else "TAKE_SHORT_NOW"
        return EntryTabularNoXGBShadowDecision(
            action=action,
            side=side,
            score=score_value,
            score_threshold=self.score_threshold,
            chosen_prob=chosen_prob_value,
            p_long=float(probs[0, 0]),
            p_short=float(probs[0, 1]),
            p_flat=float(probs[0, 2]),
            feature_contract_hash=self.feature_hash,
            candidate_id=self.candidate_id,
            manifest_path=str(self.manifest_path),
        )
