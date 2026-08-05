#!/usr/bin/env python3
"""Machine-check Entry foundation feature liveness before training."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from gx1.utils.nested_array_columns_v1 import (
    stack_nested_array_column as _stack_list_column,
)
from gx1.contracts.entry_foundation_audit_policy_v1 import (
    FOUNDATION_AUDIT_DATA_SPLITS,
    foundation_audit_policy_binding,
    foundation_audit_policy_enforcement,
    foundation_audit_policy_metadata,
)
from gx1.contracts.entry_dataset_split_artifacts_v1 import (
    ENTRY_DATASET_SPLIT_ARTIFACTS_SCHEMA_VERSION,
    require_dataset_split_artifacts,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_BASE_SIGNAL_DIM,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SIGNAL_DIM,
    require_model_native_manifest,
    require_model_native_signal_contract,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
)
from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_STRUCTURE_SOURCE_FIELDS,
    FOUNDATION_STRUCTURE_FEATURE_NAMES,
    FOUNDATION_STRUCTURE_FEATURE_VERSION,
    missing_foundation_structure_source_fields,
)
KNOWN_SPARSE_SOURCE_FIELDS = {
    "snap.smc_choch",
}

_FEATURE_AUDIT_POLICY = foundation_audit_policy_metadata()["feature_liveness"]
LIVENESS_EPSILON = float(_FEATURE_AUDIT_POLICY["liveness_epsilon"])
NEAR_CONSTANT_STD = float(_FEATURE_AUDIT_POLICY["near_constant_std"])
MIN_REQUIRED_FAMILY_ACTIVE_RATE = float(
    _FEATURE_AUDIT_POLICY["min_required_family_active_rate"]
)
MIN_REQUIRED_OBJECTIVE_ACTIVE_RATE = float(
    _FEATURE_AUDIT_POLICY["min_required_objective_active_rate"]
)
MIN_REQUIRED_SOURCE_ACTIVE_RATE = float(
    _FEATURE_AUDIT_POLICY["min_required_source_active_rate"]
)
MIN_REQUIRED_SOURCE_ACTIVE_COUNT = int(
    _FEATURE_AUDIT_POLICY["min_required_source_active_count"]
)
PARQUET_BATCH_SIZE = int(_FEATURE_AUDIT_POLICY["parquet_batch_size"])
SELECTED_FEATURE_LEARNABILITY_SPLITS = tuple(
    str(split)
    for split in _FEATURE_AUDIT_POLICY["selected_feature_learnability_splits"]
)

REQUIRED_FOUNDATION_LIVENESS_FAMILIES = (
    "foundation_hh_hl_lh_ll",
    "foundation_bos_choch_age",
    "foundation_sweep_reclaim",
    "foundation_compression_expansion",
    "foundation_impulse_pullback",
    "foundation_session_x_structure",
)

_SESSION_X_STRUCTURE_SIGNALS = (
    "hh_state",
    "hl_state",
    "lh_state",
    "ll_state",
    "bos_balance",
    "choch_recent",
    "sweep_reclaim_balance",
)

REQUIRED_FOUNDATION_OBJECTIVE_FEATURES = {
    "hh_hl_lh_ll": (
        "chart.foundation_hh_state",
        "chart.foundation_hl_state",
        "chart.foundation_lh_state",
        "chart.foundation_ll_state",
        "chart.foundation_structure_up_minus_down",
    ),
    "bos_choch_age": (
        "chart.foundation_bos_up_age_bars",
        "chart.foundation_bos_down_age_bars",
        "chart.foundation_bos_up_recent_tau24",
        "chart.foundation_bos_down_recent_tau24",
        "chart.foundation_bos_recent_balance",
        "chart.foundation_choch_age_bars",
        "chart.foundation_choch_recent_tau24",
        "chart.foundation_bars_since_structure_break_min",
    ),
    "sweep_reclaim_false_breakout": (
        "chart.foundation_sweep_low_reclaim_up_proxy",
        "chart.foundation_sweep_high_reclaim_down_proxy",
        "chart.foundation_false_breakout_high_followthrough_down_proxy",
        "chart.foundation_false_breakout_low_followthrough_up_proxy",
        "chart.foundation_sweep_reclaim_balance_proxy",
    ),
    "compression_expansion": (
        "chart.foundation_compression_state",
        "chart.foundation_expansion_state",
        "chart.foundation_compression_release_trigger",
        "chart.foundation_compression_release_up",
        "chart.foundation_compression_release_down",
    ),
    "impulse_pullback_phase": (
        "chart.foundation_impulse_direction",
        "chart.foundation_impulse_age_proxy",
        "chart.foundation_pullback_phase_up",
        "chart.foundation_pullback_phase_down",
        "chart.foundation_pullback_depth_norm",
        "chart.foundation_impulse_pullback_alignment",
    ),
    "session_x_structure": tuple(
        f"chart.foundation_{session}_x_{signal}"
        for session in ("asia", "eu", "us", "overlap")
        for signal in _SESSION_X_STRUCTURE_SIGNALS
    ),
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _feature_family(name: str) -> str:
    n = str(name).lower()
    if n.startswith("chart.foundation_"):
        if any(k in n for k in ("asia_x", "eu_x", "us_x", "overlap_x")):
            return "foundation_session_x_structure"
        if any(k in n for k in ("hh_state", "hl_state", "lh_state", "ll_state", "structure_up_minus_down")):
            return "foundation_hh_hl_lh_ll"
        if "bos_" in n or "choch_" in n or "structure_break" in n:
            return "foundation_bos_choch_age"
        if "sweep" in n or "false_breakout" in n:
            return "foundation_sweep_reclaim"
        if "compression" in n or "expansion" in n:
            return "foundation_compression_expansion"
        if "impulse" in n or "pullback" in n:
            return "foundation_impulse_pullback"
        return "foundation_other"
    if any(k in n for k in ("smc", "swing", "bos", "choch", "structure", "pullback")):
        return "structure_smc_swing"
    if any(k in n for k in ("sweep", "liquidity", "sr_", "pivot", "wick")):
        return "liquidity_sweep"
    if any(k in n for k in ("compression", "squeeze", "atr", "vol", "range", "rvol")):
        return "volatility_compression"
    if any(k in n for k in ("ema", "trend", "momentum", "slope", "ret_", "roc")):
        return "momentum_trend"
    if any(k in n for k in ("session", "hour", "dow", "is_asia", "is_eu", "is_us", "overlap")):
        return "session_time"
    return "other"


def _load_manifest_features(path: Path | None) -> tuple[list[str], dict[str, Any]]:
    if path is None:
        raise RuntimeError("model-native selection manifest is required")
    if not path.is_file():
        raise RuntimeError(f"model-native selection manifest missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError(f"model-native selection manifest root is invalid: {path}")
    contract = require_model_native_manifest(manifest, context="FOUNDATION_FEATURE_AUDIT")
    features = list(contract["selected_fields"])
    mandatory_prefix = features[:MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT]
    ranked_remainder = features[MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT:]
    declared_ranked_remainder = manifest.get("ranked_remainder_features")
    feature_ranking = manifest.get("feature_ranking")
    partition_failures: list[str] = []
    exact_scalars = {
        "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "mandatory_selected_feature_count": (
            MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
        ),
        "ranked_remainder_feature_count": (
            MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT
        ),
    }
    for key, expected in exact_scalars.items():
        if manifest.get(key) != expected:
            partition_failures.append(
                f"{key}={manifest.get(key)!r} expected={expected}"
            )
    if mandatory_prefix != list(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS):
        partition_failures.append("mandatory_selected_fields prefix/order mismatch")
    if len(ranked_remainder) != MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT:
        partition_failures.append(
            "ranked remainder width mismatch: "
            f"observed={len(ranked_remainder)} "
            f"expected={MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT}"
        )
    if declared_ranked_remainder != ranked_remainder:
        partition_failures.append("ranked_remainder_features order mismatch")
    ranked_remainder_sha256 = _sha256_json(ranked_remainder)
    if manifest.get("ranked_remainder_fields_sha256") != ranked_remainder_sha256:
        partition_failures.append("ranked_remainder_fields_sha256 mismatch")
    selected_fields_sha256 = _sha256_json(features)
    if manifest.get("selected_fields_sha256") != selected_fields_sha256:
        partition_failures.append("selected_fields_sha256 mismatch")
    if not isinstance(feature_ranking, dict):
        partition_failures.append("feature_ranking metadata missing")
        feature_ranking = {}
    if feature_ranking.get("fit_scope") != "train_only":
        partition_failures.append(
            f"feature_ranking.fit_scope={feature_ranking.get('fit_scope')!r} "
            "expected='train_only'"
        )
    ranking_sha256 = str(feature_ranking.get("sha256") or "")
    if len(ranking_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in ranking_sha256
    ):
        partition_failures.append("feature_ranking.sha256 is not lowercase sha256")
    if (
        int(feature_ranking.get("eligible_ranked_remainder_count") or 0)
        < MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT
    ):
        partition_failures.append(
            "feature_ranking eligible remainder count is insufficient"
        )
    if manifest.get(
        "ranking_artifact_is_upstream_prerequisite_not_runtime_authority"
    ) is not True:
        partition_failures.append(
            "ranking artifact authority boundary is not explicit"
        )
    if partition_failures:
        raise RuntimeError(
            "[FOUNDATION_FEATURE_AUDIT_MODEL_NATIVE_PARTITION_INVALID] "
            + " | ".join(partition_failures)
        )
    return features, {
        "manifest_path": str(path),
        "selected_features_source": "exact_model_native_selection_manifest",
        "manifest_schema_version": manifest.get("schema_version"),
        "manifest_selected_feature_count": len(features),
        "manifest_foundation_all_required_selected": manifest.get("foundation_structure_all_required_selected"),
        "manifest_mandatory_selected_feature_count": (
            manifest.get("mandatory_selected_feature_count")
        ),
        "manifest_ranked_remainder_feature_count": (
            manifest.get("ranked_remainder_feature_count")
        ),
        "ranked_remainder_fields_sha256": ranked_remainder_sha256,
        "feature_ranking_fit_scope": feature_ranking.get("fit_scope"),
        "feature_ranking_sha256": ranking_sha256,
        "model_native_signal_contract": contract,
    }


def _split_schema(files: dict[str, Path], splits: list[str]) -> dict[str, Any]:
    if tuple(files) != tuple(splits):
        raise RuntimeError("model-native split parquet bindings are not exact")
    out: dict[str, Any] = {}
    for split in splits:
        path = files[split]
        pf = pq.ParquetFile(path)
        first = None
        for batch in pf.iter_batches(batch_size=1, columns=["seq", "snap", "ctx_cont", "ctx_cat"]):
            first = batch.to_pandas().iloc[0]
            break
        if first is None:
            raise RuntimeError(f"empty split parquet: {path}")
        seq0 = _stack_list_column([first["seq"]], np.float32)[0]
        snap0 = _stack_list_column([first["snap"]], np.float32)[0]
        ctx0 = _stack_list_column([first["ctx_cont"]], np.float32)[0]
        cat0 = _stack_list_column([first["ctx_cat"]], np.int64)[0]
        out[split] = {
            "path": str(path),
            "rows": int(pf.metadata.num_rows),
            "seq_shape_observed": [int(x) for x in seq0.shape],
            "snap_dim_observed": int(len(snap0)),
            "ctx_cont_dim_observed": int(len(ctx0)),
            "ctx_cat_dim_observed": int(len(cat0)),
            "model_native_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
            "ctx_cont_contract_dim_v3": MODEL_NATIVE_CTX_CONT_DIM,
            "ctx_cat_contract_dim_v3": MODEL_NATIVE_CTX_CAT_DIM,
        }
    return out


def _load_emitted_contract(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        raise RuntimeError(f"split manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError(f"split manifest root invalid: {manifest_path}")
    extra = manifest.get("extra")
    if not isinstance(extra, dict):
        raise RuntimeError(f"split manifest extra contract missing: {manifest_path}")
    signal_contract = extra.get("model_native_signal_contract")
    if not isinstance(signal_contract, dict):
        raise RuntimeError(f"model-native signal contract missing: {manifest_path}")
    require_model_native_signal_contract(
        signal_contract,
        context="FOUNDATION_FEATURE_AUDIT_SPLIT",
    )
    signal_bridge = extra.get("signal_bridge")
    ctx_contract = extra.get("ctx_contract")
    if not isinstance(signal_bridge, dict) or not isinstance(ctx_contract, dict):
        raise RuntimeError(f"split signal/context surface missing: {manifest_path}")
    signal_fields = list(signal_contract["fields"])
    if list(signal_bridge.get("fields") or []) != signal_fields:
        raise RuntimeError(f"split signal surface order mismatch: {manifest_path}")
    ctx_cont_names = list(ctx_contract.get("ctx_cont_names") or [])
    ctx_cat_names = list(ctx_contract.get("ctx_cat_names") or [])
    if ctx_cont_names != list(MODEL_NATIVE_CTX_CONT_FIELDS):
        raise RuntimeError(f"split ctx_cont order mismatch: {manifest_path}")
    if ctx_cat_names != list(MODEL_NATIVE_CTX_CAT_FIELDS):
        raise RuntimeError(f"split ctx_cat order mismatch: {manifest_path}")
    extension = signal_bridge.get("seq_structure_extension_v1")
    if not isinstance(extension, dict):
        raise RuntimeError(f"split seq513 extension contract missing: {manifest_path}")
    extension_dim = signal_bridge.get("seq_structure_extension_dim")
    if not isinstance(extension_dim, int):
        raise RuntimeError(f"split seq513 extension dim missing: {manifest_path}")
    return {
        "manifest_path": str(manifest_path),
        "signal_fields": signal_fields,
        "ctx_cont_names": ctx_cont_names,
        "ctx_cat_names": ctx_cat_names,
        "seq_input_dim": signal_bridge.get("seq_input_dim"),
        "seq_structure_extension_dim": extension_dim,
        "seq_structure_extension_v1": extension,
        "model_native_signal_contract": signal_contract,
    }


def _audit_features_for_contract(selected_features: list[str], signal_fields: list[str]) -> list[str]:
    missing = [name for name in selected_features if name not in set(signal_fields)]
    if missing:
        raise RuntimeError(
            f"model-native selected audit features missing: {missing[:30]} total={len(missing)}"
        )
    return list(selected_features)


def _contract_features(contract: dict[str, Any]) -> list[str]:
    extension = contract.get("seq_structure_extension_v1") or {}
    return [str(x) for x in extension.get("features", [])]


class _SplitMatrixShapeError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class _StreamingStatsAccumulator:
    def __init__(self, names: list[str], *, liveness_epsilon: float) -> None:
        self.names = list(names)
        self.dim = len(self.names)
        self.liveness_epsilon = float(liveness_epsilon)
        self.n = 0
        self.finite = np.zeros(self.dim, dtype=np.int64)
        self.nonfinite = np.zeros(self.dim, dtype=np.int64)
        self.zero = np.zeros(self.dim, dtype=np.int64)
        self.active = np.zeros(self.dim, dtype=np.int64)
        self.sum = np.zeros(self.dim, dtype=np.float64)
        self.sumsq = np.zeros(self.dim, dtype=np.float64)
        self.min = np.full(self.dim, np.inf, dtype=np.float64)
        self.max = np.full(self.dim, -np.inf, dtype=np.float64)

    def add(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise RuntimeError(f"streaming stats shape mismatch: got={arr.shape} expected=(*,{self.dim})")
        rows = int(arr.shape[0])
        self.n += rows
        if self.dim == 0 or rows == 0:
            return
        finite = np.isfinite(arr)
        clean = np.where(finite, arr, 0.0)
        self.finite += finite.sum(axis=0).astype(np.int64)
        self.nonfinite += (~finite).sum(axis=0).astype(np.int64)
        self.zero += (clean == 0.0).sum(axis=0).astype(np.int64)
        self.active += (np.abs(clean) > self.liveness_epsilon).sum(axis=0).astype(np.int64)
        self.sum += clean.sum(axis=0)
        self.sumsq += (clean * clean).sum(axis=0)
        self.min = np.minimum(self.min, np.min(clean, axis=0))
        self.max = np.maximum(self.max, np.max(clean, axis=0))

    def add_columns(self, columns: list[np.ndarray]) -> None:
        if len(columns) != self.dim:
            raise RuntimeError(f"streaming stats column mismatch: got={len(columns)} expected={self.dim}")
        if self.dim == 0:
            return
        first = np.asarray(columns[0])
        rows = int(first.shape[0])
        self.n += rows
        if rows == 0:
            return
        for i, values in enumerate(columns):
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim != 1 or int(arr.shape[0]) != rows:
                raise RuntimeError(
                    f"streaming stats column shape mismatch: got={arr.shape} expected=({rows},)"
                )
            finite = np.isfinite(arr)
            clean = np.where(finite, arr, 0.0)
            self.finite[i] += int(finite.sum())
            self.nonfinite[i] += int((~finite).sum())
            self.zero[i] += int((clean == 0.0).sum())
            self.active[i] += int((np.abs(clean) > self.liveness_epsilon).sum())
            self.sum[i] += float(clean.sum())
            self.sumsq[i] += float((clean * clean).sum())
            self.min[i] = min(float(self.min[i]), float(clean.min()))
            self.max[i] = max(float(self.max[i]), float(clean.max()))

    def _base_row(self, index: int, *, near_constant_std: float) -> dict[str, Any]:
        if self.n <= 0:
            mean = 0.0
            std = 0.0
            min_value = 0.0
            max_value = 0.0
            finite_rate = 0.0
            zero_rate = 0.0
            active_rate = 0.0
        else:
            mean = float(self.sum[index] / float(self.n))
            variance = max(float(self.sumsq[index] / float(self.n) - mean * mean), 0.0)
            std = float(np.sqrt(variance))
            min_value = float(self.min[index]) if np.isfinite(self.min[index]) else 0.0
            max_value = float(self.max[index]) if np.isfinite(self.max[index]) else 0.0
            finite_rate = float(self.finite[index]) / float(self.n)
            zero_rate = float(self.zero[index]) / float(self.n)
            active_rate = float(self.active[index]) / float(self.n)
        return {
            "n": int(self.n),
            "finite_rate": finite_rate,
            "nonfinite_count": int(self.nonfinite[index]),
            "zero_rate": zero_rate,
            "active_count": int(self.active[index]),
            "active_rate": active_rate,
            "mean": mean,
            "std": std,
            "min": min_value,
            "max": max_value,
            "near_constant": bool(std <= float(near_constant_std)),
        }

    def feature_rows(self, *, split: str, near_constant_std: float) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for i, name in enumerate(self.names):
            base = self._base_row(i, near_constant_std=near_constant_std)
            base.pop("active_count", None)
            rows.append(
                {
                    "split": split,
                    "feature": name,
                    "family": _feature_family(name),
                    **base,
                    "constant_allowed": False,
                }
            )
        return rows

    def source_rows(
        self,
        *,
        split: str,
        near_constant_std: float,
        min_active_rate: float,
        min_active_count: int,
    ) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for i, source_field in enumerate(self.names):
            source_kind, raw_name = str(source_field).split(".", 1)
            base = self._base_row(i, near_constant_std=near_constant_std)
            nonfinite_count = int(base["nonfinite_count"])
            active_count = int(base["active_count"])
            active_rate = float(base["active_rate"])
            near_constant = bool(base["near_constant"])
            rows[source_field] = {
                "split": split,
                "source_field": source_field,
                "source_kind": source_kind,
                "raw_name": raw_name,
                "observed": True,
                "live": bool(
                    nonfinite_count == 0
                    and not near_constant
                    and active_count >= int(min_active_count)
                    and active_rate >= float(min_active_rate)
                ),
                **base,
            }
        return rows


def _missing_source_liveness_row(split: str, source_field: str) -> dict[str, Any]:
    source_kind, raw_name = str(source_field).split(".", 1)
    return {
        "split": split,
        "source_field": source_field,
        "source_kind": source_kind,
        "raw_name": raw_name,
        "observed": False,
        "live": False,
        "n": 0,
        "finite_rate": 0.0,
        "nonfinite_count": 0,
        "zero_rate": 0.0,
        "active_count": 0,
        "active_rate": 0.0,
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 0.0,
        "near_constant": True,
    }


def _batch_column_values(batch: Any, name: str) -> list[Any]:
    idx = batch.schema.get_field_index(name)
    if idx < 0:
        raise RuntimeError(f"batch lacks required column: {name}")
    return batch.column(idx).to_pylist()


def _stream_split_liveness_rows(
    parquet_path: Path,
    *,
    split: str,
    signal_fields: list[str],
    ctx_cont_names: list[str],
    audit_features: list[str],
    batch_size: int,
    liveness_epsilon: float,
    near_constant_std: float,
    min_source_active_rate: float,
    min_source_active_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pf = pq.ParquetFile(parquet_path)
    signal_idx = {str(name): i for i, name in enumerate(signal_fields)}
    ctx_idx = {str(name): i for i, name in enumerate(ctx_cont_names)}
    audit_cols = [signal_idx[str(name)] for name in audit_features]

    source_specs: list[tuple[str, str, int]] = []
    for source_field in FOUNDATION_STRUCTURE_SOURCE_FIELDS:
        source_kind, raw_name = str(source_field).split(".", 1)
        idx = (signal_idx if source_kind == "snap" else ctx_idx).get(raw_name)
        if idx is not None:
            source_specs.append((str(source_field), source_kind, int(idx)))
    source_fields = [field for field, _, _ in source_specs]

    feature_acc = _StreamingStatsAccumulator(
        list(audit_features),
        liveness_epsilon=float(liveness_epsilon),
    )
    source_acc = _StreamingStatsAccumulator(
        source_fields,
        liveness_epsilon=float(liveness_epsilon),
    )

    rows_seen = 0
    for batch in pf.iter_batches(batch_size=max(1, int(batch_size)), columns=["snap", "ctx_cont"]):
        snap = _stack_list_column(_batch_column_values(batch, "snap"), np.float32)
        ctx_cont = _stack_list_column(_batch_column_values(batch, "ctx_cont"), np.float32)
        if snap.ndim != 2 or snap.shape[1] != len(signal_fields):
            raise _SplitMatrixShapeError(
                f"{split}: snap matrix shape {list(snap.shape)} incompatible with emitted signal field count {len(signal_fields)}"
            )
        if ctx_cont.ndim != 2 or ctx_cont.shape[1] != len(ctx_cont_names):
            raise _SplitMatrixShapeError(
                f"{split}: ctx_cont matrix shape {list(ctx_cont.shape)} incompatible with emitted ctx_cont field count {len(ctx_cont_names)}"
            )
        rows_seen += int(snap.shape[0])
        feature_acc.add(snap[:, audit_cols])
        if source_specs:
            source_acc.add_columns(
                [
                    snap[:, idx] if source_kind == "snap" else ctx_cont[:, idx]
                    for _, source_kind, idx in source_specs
                ]
            )

    if rows_seen == 0:
        raise _SplitMatrixShapeError(
            f"{split}: snap matrix shape {[0]} incompatible with emitted signal field count {len(signal_fields)}"
        )

    source_by_field = source_acc.source_rows(
        split=split,
        near_constant_std=float(near_constant_std),
        min_active_rate=float(min_source_active_rate),
        min_active_count=int(min_source_active_count),
    )
    source_rows = [
        source_by_field.get(str(source_field), _missing_source_liveness_row(split, str(source_field)))
        for source_field in FOUNDATION_STRUCTURE_SOURCE_FIELDS
    ]
    return (
        feature_acc.feature_rows(split=split, near_constant_std=float(near_constant_std)),
        source_rows,
    )


def _required_source_field_liveness_failures(
    source_rows: list[dict[str, Any]],
    *,
    splits: list[str],
    required_source_fields: tuple[str, ...],
    min_active_rate: float,
    min_active_count: int,
) -> list[str]:
    by_key = {
        (str(row.get("split")), str(row.get("source_field"))): row
        for row in source_rows
        if isinstance(row, dict)
    }
    failures: list[str] = []
    for split in splits:
        for source_field in required_source_fields:
            row = by_key.get((split, source_field))
            if row is None:
                failures.append(f"{split}: required foundation source-field liveness missing: {source_field}")
                continue
            if not bool(row.get("observed")):
                failures.append(f"{split}: required foundation source field absent from matrix: {source_field}")
            if int(row.get("nonfinite_count") or 0) > 0:
                failures.append(
                    f"{split}: required foundation source field has non-finite values: "
                    f"{source_field} nonfinite={row.get('nonfinite_count')}"
                )
            if bool(row.get("near_constant")):
                failures.append(
                    f"{split}: required foundation source field is near-constant: "
                    f"{source_field} std={row.get('std')}"
                )
            active_count = int(row.get("active_count") or 0)
            if active_count < int(min_active_count):
                failures.append(
                    f"{split}: required foundation source field active count too low: "
                    f"{source_field} active_count={active_count} min={int(min_active_count)}"
                )
            active_rate = float(row.get("active_rate") or 0.0)
            if source_field in KNOWN_SPARSE_SOURCE_FIELDS and active_count >= int(min_active_count):
                continue
            if active_rate < float(min_active_rate):
                failures.append(
                    f"{split}: required foundation source field active rate too low: "
                    f"{source_field} active_rate={active_rate:.8f} min={float(min_active_rate):.8f}"
                )
    return failures


def _family_liveness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["split"]), str(row["family"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (split, family), items in sorted(grouped.items()):
        out.append(
            {
                "split": split,
                "family": family,
                "feature_count": int(len(items)),
                "mean_active_rate": float(np.mean([float(x["active_rate"]) for x in items])),
                "min_active_rate": float(np.min([float(x["active_rate"]) for x in items])),
                "near_constant_count": int(sum(bool(x["near_constant"]) for x in items)),
                "nonfinite_count": int(sum(int(x["nonfinite_count"]) for x in items)),
            }
        )
    return out


def _required_family_liveness_failures(
    family_rows: list[dict[str, Any]],
    *,
    splits: list[str],
    required_families: tuple[str, ...],
    min_mean_active_rate: float,
) -> list[str]:
    by_key = {
        (str(row["split"]), str(row["family"])): row
        for row in family_rows
    }
    failures: list[str] = []
    for split in splits:
        for family in required_families:
            row = by_key.get((split, family))
            if row is None:
                failures.append(f"{split}: required foundation liveness family missing: {family}")
                continue
            if int(row.get("feature_count") or 0) <= 0:
                failures.append(f"{split}: required foundation liveness family has zero features: {family}")
            if int(row.get("nonfinite_count") or 0) > 0:
                failures.append(
                    f"{split}: required foundation liveness family has non-finite values: "
                    f"{family} nonfinite={row.get('nonfinite_count')}"
                )
            if int(row.get("near_constant_count") or 0) > 0:
                failures.append(
                    f"{split}: required foundation liveness family has near-constant features: "
                    f"{family} near_constant={row.get('near_constant_count')}"
                )
            active_rate = float(row.get("mean_active_rate") or 0.0)
            if active_rate < float(min_mean_active_rate):
                failures.append(
                    f"{split}: required foundation liveness family active rate too low: "
                    f"{family} mean_active_rate={active_rate:.8f} min={float(min_mean_active_rate):.8f}"
                )
    return failures


def _objective_coverage(selected_features: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    selected = set(selected_features)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for objective, required in REQUIRED_FOUNDATION_OBJECTIVE_FEATURES.items():
        required_list = list(required)
        missing = [name for name in required_list if name not in selected]
        rows.append(
            {
                "objective": objective,
                "required_count": int(len(required_list)),
                "present_count": int(len(required_list) - len(missing)),
                "missing_count": int(len(missing)),
                "missing": missing,
            }
        )
        if missing:
            failures.append(
                f"foundation objective coverage missing {objective}: "
                f"{missing[:30]} total={len(missing)}"
            )
    return rows, failures


def _objective_liveness(stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_split_feature = {
        (str(row.get("split")), str(row.get("feature"))): row
        for row in stats
        if isinstance(row, dict)
    }
    splits = sorted({split for split, _ in by_split_feature})
    rows: list[dict[str, Any]] = []
    for split in splits:
        for objective, required in REQUIRED_FOUNDATION_OBJECTIVE_FEATURES.items():
            required_list = list(required)
            observed = [by_split_feature.get((split, feature)) for feature in required_list]
            missing = [
                feature
                for feature, row in zip(required_list, observed, strict=False)
                if row is None
            ]
            live_rows = [row for row in observed if row is not None]
            active_rates = [float(row.get("active_rate") or 0.0) for row in live_rows]
            near_constant = [
                str(row.get("feature"))
                for row in live_rows
                if bool(row.get("near_constant")) and not bool(row.get("constant_allowed"))
            ]
            nonfinite = [
                {
                    "feature": str(row.get("feature")),
                    "nonfinite_count": int(row.get("nonfinite_count") or 0),
                }
                for row in live_rows
                if int(row.get("nonfinite_count") or 0) > 0
            ]
            rows.append(
                {
                    "split": split,
                    "objective": objective,
                    "required_count": int(len(required_list)),
                    "observed_count": int(len(live_rows)),
                    "missing_count": int(len(missing)),
                    "missing": missing,
                    "mean_active_rate": float(np.mean(active_rates)) if active_rates else 0.0,
                    "min_active_rate": float(np.min(active_rates)) if active_rates else 0.0,
                    "near_constant_count": int(len(near_constant)),
                    "near_constant": near_constant,
                    "nonfinite_count": int(sum(row["nonfinite_count"] for row in nonfinite)),
                    "nonfinite": nonfinite,
                }
            )
    return rows


def _required_objective_liveness_failures(
    objective_rows: list[dict[str, Any]],
    *,
    splits: list[str],
    required_objectives: tuple[str, ...],
    min_mean_active_rate: float,
) -> list[str]:
    by_key = {
        (str(row.get("split")), str(row.get("objective"))): row
        for row in objective_rows
        if isinstance(row, dict)
    }
    failures: list[str] = []
    for split in splits:
        for objective in required_objectives:
            row = by_key.get((split, objective))
            if row is None:
                failures.append(f"{split}: required foundation objective liveness missing: {objective}")
                continue
            if int(row.get("missing_count") or 0) != 0:
                failures.append(
                    f"{split}: required foundation objective has missing live features: "
                    f"{objective} missing={row.get('missing')}"
                )
            if int(row.get("nonfinite_count") or 0) > 0:
                failures.append(
                    f"{split}: required foundation objective has non-finite values: "
                    f"{objective} nonfinite={row.get('nonfinite_count')}"
                )
            if int(row.get("near_constant_count") or 0) > 0:
                failures.append(
                    f"{split}: required foundation objective has near-constant features: "
                    f"{objective} near_constant={row.get('near_constant')}"
                )
            active_rate = float(row.get("mean_active_rate") or 0.0)
            if active_rate < float(min_mean_active_rate):
                failures.append(
                    f"{split}: required foundation objective active rate too low: "
                    f"{objective} mean_active_rate={active_rate:.8f} min={float(min_mean_active_rate):.8f}"
                )
    return failures


def _drift_rows(stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_feature: dict[str, dict[str, dict[str, Any]]] = {}
    for row in stats:
        by_feature.setdefault(str(row["feature"]), {})[str(row["split"])] = row
    out: list[dict[str, Any]] = []
    for feature, split_rows in sorted(by_feature.items()):
        train = split_rows.get("train")
        if not train:
            continue
        train_std = max(float(train["std"]), 1e-9)
        for split, row in sorted(split_rows.items()):
            if split == "train":
                continue
            std_ratio = float(row["std"]) / train_std
            out.append(
                {
                    "feature": feature,
                    "family": row["family"],
                    "split": split,
                    "mean_shift_abs": abs(float(row["mean"]) - float(train["mean"])),
                    "std_ratio_vs_train": std_ratio,
                    "active_rate_delta_vs_train": float(row["active_rate"]) - float(train["active_rate"]),
                }
            )
    out.sort(key=lambda r: (float(r["mean_shift_abs"]), abs(float(r["std_ratio_vs_train"]) - 1.0)), reverse=True)
    return out


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Feature Foundation Audit",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Selected features: `{report['selected_feature_count']}`",
        f"- Foundation required: `{report['foundation_required_feature_count']}`",
        f"- Failure count: `{len(report['failures'])}`",
        "",
        "## Failures",
        "",
    ]
    if report["failures"]:
        lines.extend([f"- {failure}" for failure in report["failures"]])
    else:
        lines.append("- None")
    lines.extend(["", "## Objective Coverage", ""])
    for row in report["foundation_objective_coverage"]:
        lines.append(
            f"- `{row['objective']}`: present={row['present_count']}/{row['required_count']} "
            f"missing={row['missing_count']}"
        )
    lines.extend(["", "## Objective Liveness", ""])
    for row in report["foundation_objective_liveness"]:
        lines.append(
            f"- `{row['split']}` `{row['objective']}`: observed={row['observed_count']}/{row['required_count']} "
            f"mean_active={row['mean_active_rate']:.6f} near_constant={row['near_constant_count']} "
            f"nonfinite={row['nonfinite_count']}"
        )
    lines.extend(["", "## Source Dependencies", ""])
    for split, contract in sorted((report.get("emitted_contracts") or {}).items()):
        lines.append(
            f"- `{split}`: source_fields={contract.get('foundation_structure_source_field_count')} "
            f"missing={contract.get('foundation_structure_source_missing_count')}"
        )
    lines.extend(["", "## Source Field Liveness", ""])
    for row in report["foundation_source_field_liveness"]:
        lines.append(
            f"- `{row['split']}` `{row['source_field']}`: active={row['active_count']}/{row['n']} "
            f"rate={row['active_rate']:.8f} near_constant={row['near_constant']} "
            f"nonfinite={row['nonfinite_count']}"
        )
    lines.extend(["", "## Family Liveness", ""])
    for row in report["family_liveness"]:
        lines.append(
            f"- `{row['split']}` `{row['family']}`: features={row['feature_count']} "
            f"mean_active={row['mean_active_rate']:.6f} near_constant={row['near_constant_count']} "
            f"nonfinite={row['nonfinite_count']}"
        )
    lines.extend(["", "## Top Drift", ""])
    for row in report["distribution_drift_top20"]:
        lines.append(
            f"- `{row['split']}` `{row['feature']}`: mean_shift={row['mean_shift_abs']:.6f} "
            f"std_ratio={row['std_ratio_vs_train']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    manifest_path = Path(args.seq_structure_manifest).expanduser().resolve() if args.seq_structure_manifest else None
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = list(FOUNDATION_AUDIT_DATA_SPLITS)
    selected_features, manifest_meta = _load_manifest_features(manifest_path)
    foundation_required = list(FOUNDATION_STRUCTURE_FEATURE_NAMES)
    foundation_missing_from_manifest = [name for name in foundation_required if name not in set(selected_features)]

    failures: list[str] = []
    schema: dict[str, Any] = {}
    split_artifacts: dict[str, dict[str, str]] = {}
    try:
        split_artifacts = require_dataset_split_artifacts(
            dataset_dir,
            {
                split: {
                    "manifest_path": getattr(args, f"{split}_manifest_json"),
                    "manifest_sha256": getattr(
                        args,
                        f"{split}_manifest_sha256",
                    ),
                    "parquet_sha256": getattr(args, f"{split}_parquet_sha256"),
                }
                for split in splits
            },
            expected_splits=splits,
            context="FOUNDATION_FEATURE_AUDIT_DATASET",
        )
        schema = _split_schema(
            {
                split: Path(split_artifacts[split]["parquet_path"])
                for split in splits
            },
            splits,
        )
    except Exception as exc:
        failures.append(f"dataset split/schema load failed: {exc}")

    stats: list[dict[str, Any]] = []
    source_field_liveness: list[dict[str, Any]] = []
    missing_by_split: dict[str, list[str]] = {}
    emitted_contracts: dict[str, dict[str, Any]] = {}
    audited_feature_count_by_split: dict[str, int] = {}
    if schema:
        for split in splits:
            split_schema = schema.get(split)
            if not split_schema:
                missing_by_split[split] = list(selected_features)
                failures.append(f"{split}: split schema missing")
                continue
            parquet_path = Path(str(split_schema["path"]))
            try:
                contract = _load_emitted_contract(
                    Path(split_artifacts[split]["manifest_path"])
                )
            except Exception as exc:
                failures.append(f"{split}: emitted signal contract load failed: {exc}")
                missing_by_split[split] = list(selected_features)
                continue
            emitted_contracts[split] = contract
            signal_fields = [str(x) for x in contract["signal_fields"]]
            ctx_cont_names = [str(x) for x in contract.get("ctx_cont_names") or []]
            source_universe = [f"snap.{name}" for name in signal_fields] + [
                f"ctx_cont.{name}" for name in ctx_cont_names
            ]
            source_missing = missing_foundation_structure_source_fields(source_universe)
            contract["foundation_structure_source_fields"] = list(FOUNDATION_STRUCTURE_SOURCE_FIELDS)
            contract["foundation_structure_source_field_count"] = int(len(FOUNDATION_STRUCTURE_SOURCE_FIELDS))
            contract["foundation_structure_source_missing"] = source_missing
            contract["foundation_structure_source_missing_count"] = int(len(source_missing))
            if source_missing:
                failures.append(
                    f"{split}: foundation structure source fields missing from emitted signal/ctx contracts: "
                    f"{source_missing[:30]} total={len(source_missing)}"
                )
            signal_idx = {name: i for i, name in enumerate(signal_fields)}
            contract_features = _contract_features(contract)
            contract_feature_set = set(contract_features)
            seq_extension = (
                contract.get("seq_structure_extension_v1")
                if isinstance(contract.get("seq_structure_extension_v1"), dict)
                else {}
            )
            split_foundation_version = str(seq_extension.get("foundation_structure_feature_version") or "")
            if split_foundation_version != FOUNDATION_STRUCTURE_FEATURE_VERSION:
                failures.append(
                    f"{split}: emitted seq_structure_extension_v1 foundation version mismatch: "
                    f"{split_foundation_version or '<missing>'} != {FOUNDATION_STRUCTURE_FEATURE_VERSION}"
                )
            if not bool(seq_extension.get("foundation_structure_all_required_selected")):
                failures.append(f"{split}: emitted seq_structure_extension_v1 did not preserve all-required foundation selection")
            if int(seq_extension.get("foundation_structure_feature_count") or 0) != len(FOUNDATION_STRUCTURE_FEATURE_NAMES):
                failures.append(
                    f"{split}: emitted seq_structure_extension_v1 foundation feature count "
                    f"{seq_extension.get('foundation_structure_feature_count')} != {len(FOUNDATION_STRUCTURE_FEATURE_NAMES)}"
                )

            if int(split_schema["snap_dim_observed"]) != len(signal_fields):
                failures.append(
                    f"{split}: snap dim {split_schema['snap_dim_observed']} != emitted signal field count {len(signal_fields)}"
                )
            seq_shape = list(split_schema.get("seq_shape_observed") or [])
            if len(seq_shape) != 2 or int(seq_shape[1]) != len(signal_fields):
                failures.append(f"{split}: seq shape {seq_shape} incompatible with emitted signal field count {len(signal_fields)}")
            if int(contract.get("seq_input_dim") or 0) != len(signal_fields):
                failures.append(f"{split}: manifest seq_input_dim {contract.get('seq_input_dim')} != emitted signal field count {len(signal_fields)}")
            if int(contract.get("seq_structure_extension_dim") or 0) != len(selected_features):
                failures.append(
                    f"{split}: seq_structure_extension_dim {contract.get('seq_structure_extension_dim')} != selected feature count {len(selected_features)}"
                )
            contract_missing = [name for name in selected_features if name not in contract_feature_set]
            if contract_missing:
                failures.append(
                    f"{split}: selected features missing from emitted seq_structure_extension_v1: "
                    f"{contract_missing[:30]} total={len(contract_missing)}"
                )

            missing = [name for name in selected_features if name not in signal_idx]
            missing_by_split[split] = missing
            if missing:
                continue
            audit_features = _audit_features_for_contract(selected_features, signal_fields)
            audited_feature_count_by_split[split] = int(len(audit_features))
            audit_missing = [name for name in audit_features if name not in signal_idx]
            if audit_missing:
                failures.append(f"{split}: audit features missing from emitted signal fields: {audit_missing[:30]} total={len(audit_missing)}")
                continue
            try:
                split_stats, split_source_rows = _stream_split_liveness_rows(
                    parquet_path,
                    split=split,
                    signal_fields=signal_fields,
                    ctx_cont_names=ctx_cont_names,
                    audit_features=audit_features,
                    batch_size=PARQUET_BATCH_SIZE,
                    liveness_epsilon=LIVENESS_EPSILON,
                    near_constant_std=NEAR_CONSTANT_STD,
                    min_source_active_rate=MIN_REQUIRED_SOURCE_ACTIVE_RATE,
                    min_source_active_count=MIN_REQUIRED_SOURCE_ACTIVE_COUNT,
                )
            except _SplitMatrixShapeError as exc:
                failures.append(exc.message)
                continue
            except Exception as exc:
                failures.append(f"{split}: snap/ctx_cont load failed: {exc}")
                continue
            source_field_liveness.extend(split_source_rows)
            stats.extend(split_stats)
    else:
        for split in splits:
            missing_by_split[split] = []

    if foundation_missing_from_manifest:
        failures.append(
            "foundation features missing from sequence manifest: "
            f"{foundation_missing_from_manifest[:30]} total={len(foundation_missing_from_manifest)}"
        )
    objective_coverage, objective_coverage_failures = _objective_coverage(selected_features)
    failures.extend(objective_coverage_failures)
    for split, missing in missing_by_split.items():
        if missing:
            failures.append(f"{split}: selected features missing from generated matrix: {missing[:30]} total={len(missing)}")
    for row in stats:
        feature = str(row["feature"])
        if int(row["nonfinite_count"]) > 0:
            failures.append(f"{row['split']}: non-finite values in {feature}: {row['nonfinite_count']}")
        if (
            str(row["split"]) in SELECTED_FEATURE_LEARNABILITY_SPLITS
            and bool(row["near_constant"])
        ):
            failures.append(f"{row['split']}: unexpected near-constant feature {feature} std={row['std']}")

    families_present = {str(row["family"]) for row in stats}
    missing_families = sorted(set(REQUIRED_FOUNDATION_LIVENESS_FAMILIES) - families_present)
    if missing_families:
        failures.append(f"foundation liveness families missing from stats: {missing_families}")

    family_liveness = _family_liveness(stats)
    failures.extend(
        _required_family_liveness_failures(
            family_liveness,
            splits=splits,
            required_families=REQUIRED_FOUNDATION_LIVENESS_FAMILIES,
            min_mean_active_rate=MIN_REQUIRED_FAMILY_ACTIVE_RATE,
        )
    )
    objective_liveness = _objective_liveness(stats)
    objective_liveness_failures = _required_objective_liveness_failures(
        objective_liveness,
        splits=splits,
        required_objectives=tuple(REQUIRED_FOUNDATION_OBJECTIVE_FEATURES),
        min_mean_active_rate=MIN_REQUIRED_OBJECTIVE_ACTIVE_RATE,
    )
    failures.extend(objective_liveness_failures)
    source_liveness_failures = _required_source_field_liveness_failures(
        source_field_liveness,
        splits=splits,
        required_source_fields=FOUNDATION_STRUCTURE_SOURCE_FIELDS,
        min_active_rate=MIN_REQUIRED_SOURCE_ACTIVE_RATE,
        min_active_count=MIN_REQUIRED_SOURCE_ACTIVE_COUNT,
    )
    failures.extend(source_liveness_failures)

    drift = _drift_rows(stats)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    decision = "PASS" if not failures else "FAIL"
    report = {
        "schema_version": "entry_feature_foundation_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        **foundation_audit_policy_binding(),
        "foundation_audit_policy_enforcement": (
            foundation_audit_policy_enforcement("feature")
        ),
        "dataset_dir": str(dataset_dir),
        "data_splits": splits,
        "split_artifacts_schema_version": (
            ENTRY_DATASET_SPLIT_ARTIFACTS_SCHEMA_VERSION
        ),
        "split_artifacts": split_artifacts,
        "model_native_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
        "base_signal_fields": list(MODEL_NATIVE_BASE_FIELDS),
        "ctx_cont_dim_v3": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim_v3": MODEL_NATIVE_CTX_CAT_DIM,
        "selected_feature_count": int(len(selected_features)),
        "mandatory_selected_feature_count": (
            MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
        ),
        "ranked_remainder_feature_count": (
            MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT
        ),
        "foundation_structure_feature_version": FOUNDATION_STRUCTURE_FEATURE_VERSION,
        "foundation_required_feature_count": int(len(foundation_required)),
        "foundation_objective_coverage": objective_coverage,
        "foundation_objective_coverage_all_present": not objective_coverage_failures,
        "foundation_objective_liveness": objective_liveness,
        "foundation_objective_liveness_all_live": not objective_liveness_failures,
        "foundation_source_field_liveness": source_field_liveness,
        "foundation_source_field_liveness_all_live": not source_liveness_failures,
        "foundation_missing_from_manifest_count": int(len(foundation_missing_from_manifest)),
        "foundation_missing_from_manifest": foundation_missing_from_manifest,
        "split_schema": schema,
        "emitted_contracts": emitted_contracts,
        "audited_feature_count_by_split": audited_feature_count_by_split,
        "missing_by_split": missing_by_split,
        "failures": failures,
        "stats": stats,
        "required_foundation_liveness_families": list(REQUIRED_FOUNDATION_LIVENESS_FAMILIES),
        "selected_feature_learnability_splits": list(
            SELECTED_FEATURE_LEARNABILITY_SPLITS
        ),
        "min_required_family_active_rate": MIN_REQUIRED_FAMILY_ACTIVE_RATE,
        "min_required_objective_active_rate": MIN_REQUIRED_OBJECTIVE_ACTIVE_RATE,
        "min_required_source_active_rate": MIN_REQUIRED_SOURCE_ACTIVE_RATE,
        "min_required_source_active_count": MIN_REQUIRED_SOURCE_ACTIVE_COUNT,
        "family_liveness": family_liveness,
        "distribution_drift_top20": drift[:20],
        **manifest_meta,
    }

    json_path = out_dir / f"ENTRY_FEATURE_FOUNDATION_AUDIT_{timestamp}.json"
    md_path = out_dir / f"ENTRY_FEATURE_FOUNDATION_AUDIT_{timestamp}.md"
    if json_path.exists() or md_path.exists():
        raise RuntimeError(
            f"FOUNDATION_AUDIT_IMMUTABLE_EVENT_EXISTS: json={json_path} md={md_path}"
        )
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)

    if not args.quiet:
        print(json.dumps({k: report[k] for k in ["decision", "selected_feature_count", "foundation_required_feature_count", "failures", "json_path", "md_path"]}, indent=2, default=_json_default))
    if failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", required=True)
    for split in FOUNDATION_AUDIT_DATA_SPLITS:
        ap.add_argument(f"--{split}-manifest-json", required=True)
        ap.add_argument(f"--{split}-manifest-sha256", required=True)
        ap.add_argument(f"--{split}-parquet-sha256", required=True)
    ap.add_argument("--seq-structure-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
