#!/usr/bin/env python3
"""Audit the exact model-native seq513 specialist feature contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.utils.nested_array_columns_v1 import (
    stack_nested_array_column as _stack_list_column,
)
from gx1.contracts.entry_foundation_audit_policy_v1 import (
    FOUNDATION_AUDIT_DATA_SPLITS,
    foundation_audit_policy_binding,
    foundation_audit_policy_enforcement,
    foundation_audit_policy_metadata,
)
from gx1.contracts.entry_full_input_liveness_v1 import (
    RARE_EVENT_MINIMUMS,
    canonical_policy as full_input_liveness_policy,
    classify_field_status,
)
from gx1.contracts.entry_dataset_split_artifacts_v1 import (
    ENTRY_DATASET_SPLIT_ARTIFACTS_SCHEMA_VERSION,
    require_dataset_split_artifacts,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_MANDATORY_FAMILY_FEATURES,
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SIGNAL_DIM,
    model_native_mandatory_full_stack_metadata,
    require_model_native_manifest,
    require_model_native_signal_contract,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_SPECIALIST,
    FOUNDATION_OBJECTIVE_SPECIALISTS,
    FOUNDATION_REQUIREMENT_PATTERNS,
    MODEL_NATIVE_EXPECTED_SPECIALIST_FEATURE_COUNT,
    MODEL_NATIVE_SMART_FAMILY_CONTRACT,
    MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT,
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    SPECIALIST_FUSION_ACTIVE_HEADS,
    SPECIALIST_FUSION_BLOCKED_HEADS,
    SPECIALIST_SHARED_REACHABLE_HEADS,
    SPECIALIST_GROUPS,
    classify_entry_specialist_feature,
    group_features_by_specialist,
)
from gx1.scripts.audit_entry_foundation_features_v1 import REQUIRED_FOUNDATION_OBJECTIVE_FEATURES

_SPECIALIST_AUDIT_POLICY = foundation_audit_policy_metadata()[
    "specialist_liveness"
]
MIN_SIGNAL_COUNTS = dict(_SPECIALIST_AUDIT_POLICY["min_signal_counts"])
MIN_LIVE_FEATURE_COUNTS = dict(
    _SPECIALIST_AUDIT_POLICY["min_live_feature_counts"]
)
MIN_SPECIALIST_MEAN_ACTIVE_RATE = float(
    _SPECIALIST_AUDIT_POLICY["min_specialist_mean_active_rate"]
)
MIN_FEATURE_ACTIVE_RATE = float(
    _SPECIALIST_AUDIT_POLICY["min_feature_active_rate"]
)
LIVENESS_EPSILON = float(_SPECIALIST_AUDIT_POLICY["liveness_epsilon"])
NEAR_CONSTANT_STD = float(_SPECIALIST_AUDIT_POLICY["near_constant_std"])
SPECIALIST_CONTRACT_MODE = str(_SPECIALIST_AUDIT_POLICY["contract_mode"])
TRAIN_LIVE_STATUSES = frozenset(
    str(value) for value in _SPECIALIST_AUDIT_POLICY["train_live_statuses"]
)
OOS_OBSERVED_STATUSES = frozenset(
    str(value) for value in _SPECIALIST_AUDIT_POLICY["oos_observed_statuses"]
)
SPECIALIST_RARE_EVENT_MINIMUMS = {
    str(field): int(count)
    for field, count in _SPECIALIST_AUDIT_POLICY[
        "rare_event_minimum_active_count"
    ].items()
}
_CANONICAL_SIGNAL_RARE_EVENT_MINIMUMS = {
    field: int(minimums["train"])
    for (surface, field), minimums in RARE_EVENT_MINIMUMS.items()
    if surface == "signal" and "train" in minimums
}
if SPECIALIST_RARE_EVENT_MINIMUMS != _CANONICAL_SIGNAL_RARE_EVENT_MINIMUMS:
    raise RuntimeError("SPECIALIST_RARE_EVENT_POLICY_DIVERGES_FROM_FULL_INPUT_LIVENESS")
_FULL_INPUT_NUMERIC_POLICY = full_input_liveness_policy()["numeric"]
if (
    float(_FULL_INPUT_NUMERIC_POLICY["active_abs_threshold"])
    != float(LIVENESS_EPSILON)
    or float(_FULL_INPUT_NUMERIC_POLICY["near_constant_std"])
    != float(NEAR_CONSTANT_STD)
    or float(_FULL_INPUT_NUMERIC_POLICY["min_active_rate"])
    != float(MIN_FEATURE_ACTIVE_RATE)
):
    raise RuntimeError("SPECIALIST_NUMERIC_POLICY_DIVERGES_FROM_FULL_INPUT_LIVENESS")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_selected_features(path: Path) -> list[str]:
    manifest = _read_json(path)
    selected = [str(x) for x in manifest.get("selected_features", []) if str(x).strip()]
    if not selected:
        raise RuntimeError(f"sequence structure manifest has no selected_features: {path}")
    return selected


def _load_split_signal_fields(
    split_artifacts: dict[str, dict[str, str]],
    splits: list[str],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for split in splits:
        path = Path(split_artifacts[split]["manifest_path"])
        manifest = _read_json(path)
        extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
        signal_bridge = extra.get("signal_bridge") if isinstance(extra.get("signal_bridge"), dict) else {}
        model_native_signal_contract = (
            extra.get("model_native_signal_contract")
            if isinstance(extra.get("model_native_signal_contract"), dict)
            else None
        )
        if model_native_signal_contract is None:
            raise RuntimeError(f"{split}: split manifest lacks model_native_signal_contract: {path}")
        require_model_native_signal_contract(
            model_native_signal_contract,
            context=f"SPECIALIST_AUDIT_{split.upper()}",
        )
        fields = [str(x) for x in signal_bridge.get("fields", []) if str(x).strip()]
        if fields != list(model_native_signal_contract["fields"]):
            raise RuntimeError(
                f"{split}: signal_bridge fields differ from model_native_signal_contract fields"
            )
        extension = signal_bridge.get("seq_structure_extension_v1") or {}
        extension_features = [str(x) for x in extension.get("features", []) if str(x).strip()]
        out[split] = {
            "manifest_path": str(path),
            "fields": fields,
            "field_count": len(fields),
            "extension_features": extension_features,
            "extension_feature_count": len(extension_features),
            "seq_input_dim": int(signal_bridge.get("seq_input_dim") or 0),
            "snap_input_dim": int(signal_bridge.get("snap_input_dim") or 0),
            "model_native_signal_contract": model_native_signal_contract,
        }
    return out


def _load_split_context_fields(
    split_artifacts: dict[str, dict[str, str]],
    splits: list[str],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for split in splits:
        path = Path(split_artifacts[split]["manifest_path"])
        manifest = _read_json(path)
        ctx_contract = ((manifest.get("extra") or {}).get("ctx_contract") or {})
        ctx_cont_names = [str(x) for x in ctx_contract.get("ctx_cont_names", []) if str(x).strip()]
        ctx_cat_names = [str(x) for x in ctx_contract.get("ctx_cat_names", []) if str(x).strip()]
        out[split] = {
            "manifest_path": str(path),
            "ctx_tag": str(ctx_contract.get("tag") or ""),
            "ctx_cont_names": ctx_cont_names,
            "ctx_cat_names": ctx_cat_names,
            "ctx_cont_dim": int(ctx_contract.get("ctx_cont_dim") or len(ctx_cont_names)),
            "ctx_cat_dim": int(ctx_contract.get("ctx_cat_dim") or len(ctx_cat_names)),
        }
    return out


def _feature_rows(features: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "index": i,
            "feature": feature,
            "specialist": classify_entry_specialist_feature(feature),
        }
        for i, feature in enumerate(features)
    ]


def _count_rows(signal_fields: list[str], selected_features: list[str]) -> list[dict[str, Any]]:
    signal_grouped = group_features_by_specialist(signal_fields)
    selected_set = set(selected_features)
    rows: list[dict[str, Any]] = []
    for group in list(SPECIALIST_GROUPS) + ["unmapped"]:
        features = signal_grouped.get(group, [])
        selected = [f for f in features if f in selected_set]
        rows.append(
            {
                "specialist": group,
                "signal_feature_count": int(len(features)),
                "selected_extension_count": int(len(selected)),
                "signal_features": features,
                "selected_extension_features": selected,
            }
        )
    return rows


def _context_routing_rows(context_contracts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not context_contracts:
        return []
    split = sorted(context_contracts)[0]
    contract = context_contracts[split]
    rows: list[dict[str, Any]] = []
    for scope, names_key in (("ctx_cont", "ctx_cont_names"), ("ctx_cat", "ctx_cat_names")):
        for index, name in enumerate(contract.get(names_key) or []):
            feature = f"{scope}.{name}"
            rows.append(
                {
                    "scope": scope,
                    "index": int(index),
                    "feature": feature,
                    "specialist": classify_entry_specialist_feature(feature),
                }
            )
    return rows


def _context_routing_failures(rows: list[dict[str, Any]]) -> list[str]:
    unmapped = [row for row in rows if str(row.get("specialist")) == "unmapped"]
    if unmapped:
        return [
            f"{MODEL_NATIVE_CONTRACT_MODE} context routing has unmapped fields: "
            + ", ".join(str(row.get("feature")) for row in unmapped[:40])
            + f" total={len(unmapped)}"
        ]
    return []


def _contract_training_surface() -> dict[str, Any]:
    return {
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "registered_for_training_surfaces": True,
        "training_allowed_by_contract_mode": True,
        "training_allowed_by_this_audit": False,
        "training_allowed": False,
        "requires_separate_readiness_gate": True,
    }


def _smart_family_contract_rows(
    selected_features: list[str],
    signal_fields: list[str],
) -> list[dict[str, Any]]:
    selected_set = set(selected_features)
    signal_set = set(signal_fields)
    rows: list[dict[str, Any]] = []
    for (label, expected_features), (contract_label, spec) in zip(
        MODEL_NATIVE_MANDATORY_FAMILY_FEATURES,
        MODEL_NATIVE_SMART_FAMILY_CONTRACT.items(),
        strict=True,
    ):
        if label != contract_label:
            raise RuntimeError(
                "MODEL_NATIVE_SMART_FAMILY_REGISTRY_ORDER_MISMATCH: "
                f"registry={label!r} specialist_contract={contract_label!r}"
            )
        expected_specialist_counts = {
            str(k): int(v)
            for k, v in (spec.get("expected_specialist_counts") or {}).items()
        }
        expected_feature_count = len(expected_features)
        selected_members = [name for name in expected_features if name in selected_set]
        signal_members = [name for name in expected_features if name in signal_set]
        missing_selected = [name for name in expected_features if name not in selected_set]
        missing_signal = [name for name in expected_features if name not in signal_set]
        rows.append(
            {
                "family": label,
                "purpose": spec.get("purpose"),
                "expected_feature_count": expected_feature_count,
                "observed_feature_count": len(selected_members),
                "selected_feature_count": len(selected_members),
                "emitted_signal_feature_count": len(signal_members),
                "missing_selected_features": missing_selected,
                "missing_emitted_signal_features": missing_signal,
                "feature_count_matches": (
                    len(selected_members) == expected_feature_count
                    and len(signal_members) == expected_feature_count
                    and not missing_selected
                    and not missing_signal
                ),
                "expected_specialist_counts": expected_specialist_counts,
                "owned_specialists": list(spec.get("owned_specialists") or ()),
            }
        )
    return rows


def _smart_contract_failures(
    seq_manifest: dict[str, Any],
    *,
    signal_field_count: int,
    selected_feature_count: int,
    smart_family_rows: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    manifest_variant = str(seq_manifest.get("manifest_variant") or "")
    if manifest_variant != MODEL_NATIVE_CONTRACT_MODE:
        failures.append(f"model-native smart manifest_variant mismatch: observed={manifest_variant!r}")
    if signal_field_count != MODEL_NATIVE_SIGNAL_DIM:
        failures.append(
            f"model-native smart signal width mismatch: observed={signal_field_count} "
            f"expected={MODEL_NATIVE_SIGNAL_DIM}"
        )
    if selected_feature_count != MODEL_NATIVE_SELECTED_FEATURE_COUNT:
        failures.append(
            f"model-native smart selected feature count mismatch: observed={selected_feature_count} "
            f"expected={MODEL_NATIVE_SELECTED_FEATURE_COUNT}"
        )
    if seq_manifest.get("smart_layers_included") is not True:
        failures.append("model-native smart manifest does not declare smart_layers_included=true")
    if seq_manifest.get("dataset_rebuild_required_before_training") is not True:
        failures.append("model-native smart manifest must preserve dataset_rebuild_required_before_training=true")
    if seq_manifest.get("training_allowed") is not False:
        failures.append("model-native smart manifest must keep training_allowed=false")
    try:
        require_model_native_manifest(seq_manifest, context="SPECIALIST_AUDIT")
    except RuntimeError as exc:
        failures.append(str(exc))

    source_counts = (
        seq_manifest.get("source_feature_counts")
        if isinstance(seq_manifest.get("source_feature_counts"), dict)
        else {}
    )
    expected_source_counts = {
        "smart_candidate_layers": MODEL_NATIVE_EXPECTED_SPECIALIST_FEATURE_COUNT,
        "mandatory_full_stack": MODEL_NATIVE_EXPECTED_SPECIALIST_FEATURE_COUNT,
        "ranked_remainder": (
            MODEL_NATIVE_SELECTED_FEATURE_COUNT
            - MODEL_NATIVE_EXPECTED_SPECIALIST_FEATURE_COUNT
        ),
    }
    observed_smart_count = int(source_counts.get("smart_candidate_layers") or 0)
    if observed_smart_count != MODEL_NATIVE_EXPECTED_SPECIALIST_FEATURE_COUNT:
        failures.append(
            "model-native smart layer source count mismatch: "
            f"observed={source_counts.get('smart_candidate_layers')} "
            f"expected={MODEL_NATIVE_EXPECTED_SPECIALIST_FEATURE_COUNT}"
        )
    if source_counts != expected_source_counts:
        failures.append(
            "model-native source feature count metadata stale: "
            f"declared={source_counts} expected={expected_source_counts}"
        )
    expected_family_count = len(MODEL_NATIVE_MANDATORY_FAMILY_FEATURES)
    if len(smart_family_rows) != expected_family_count:
        failures.append(
            "model-native smart family contract count mismatch: "
            f"observed={len(smart_family_rows)} expected={expected_family_count}"
        )
    for row in smart_family_rows:
        if row.get("feature_count_matches") is not True:
            failures.append(
                f"model-native smart family count mismatch: {row.get('family')} "
                f"observed={row.get('observed_feature_count')} expected={row.get('expected_feature_count')}"
            )
    recomputed_family_counts = {
        str(row["family"]): int(row["selected_feature_count"])
        for row in smart_family_rows
    }
    expected_family_counts = {
        family: len(features)
        for family, features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
    }
    declared_family_counts = seq_manifest.get("smart_layer_feature_counts")
    if not isinstance(declared_family_counts, dict):
        failures.append("model-native mandatory family count metadata missing")
    elif declared_family_counts != expected_family_counts:
        failures.append(
            "model-native mandatory family count metadata stale: "
            f"declared={declared_family_counts} recomputed={recomputed_family_counts} "
            f"expected={expected_family_counts}"
        )
    declared_mandatory = seq_manifest.get("mandatory_full_stack")
    expected_mandatory = model_native_mandatory_full_stack_metadata()
    if not isinstance(declared_mandatory, dict):
        failures.append("model-native mandatory_full_stack metadata missing")
    elif declared_mandatory != expected_mandatory:
        failures.append("model-native mandatory_full_stack metadata stale")
    observed_mandatory_count = sum(
        int(row["selected_feature_count"]) for row in smart_family_rows
    )
    if observed_mandatory_count != MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT:
        failures.append(
            "model-native mandatory selected feature count mismatch: "
            f"observed={observed_mandatory_count} "
            f"expected={MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT}"
        )
    return failures


def _specialist_input_liveness_rows(
    split_artifacts: dict[str, dict[str, str]],
    splits: list[str],
    signal_fields: list[str],
    required_specialists: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups_by_feature = [classify_entry_specialist_feature(feature) for feature in signal_fields]
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    for split in splits:
        parquet_path = Path(split_artifacts[split]["parquet_path"])
        snap = _stack_list_column(pd.read_parquet(parquet_path, columns=["snap"])["snap"], np.float32)
        if snap.ndim != 2 or snap.shape[1] != len(signal_fields):
            raise RuntimeError(
                f"{split}: snap matrix shape {list(snap.shape)} does not match signal field count {len(signal_fields)}"
            )
        all_finite = np.isfinite(snap)
        all_clean = np.where(all_finite, snap, 0.0)
        all_std = np.std(all_clean, axis=0)
        all_min = np.min(all_clean, axis=0)
        all_max = np.max(all_clean, axis=0)
        all_value_range = all_max - all_min
        all_active_count = np.sum(
            np.abs(all_clean) > float(LIVENESS_EPSILON), axis=0
        )
        all_active_rate = np.mean(
            np.abs(all_clean) > float(LIVENESS_EPSILON), axis=0
        )
        all_finite_by_feature = np.all(all_finite, axis=0)
        all_status: list[str] = []
        all_status_reason: list[str] = []
        all_live_values: list[bool] = []
        accepted_statuses = (
            TRAIN_LIVE_STATUSES if split == "train" else OOS_OBSERVED_STATUSES
        )
        for index, feature in enumerate(signal_fields):
            status, reason = classify_field_status(
                split=split,
                surface="signal",
                field=feature,
                stats={
                    "row_count": int(snap.shape[0]),
                    "finite_count": int(np.sum(all_finite[:, index])),
                    "nonfinite_count": int(np.sum(~all_finite[:, index])),
                    "std": float(all_std[index]),
                    "min": float(all_min[index]),
                    "max": float(all_max[index]),
                    "value_range": float(all_value_range[index]),
                    "active_count": int(all_active_count[index]),
                    "active_rate": float(all_active_rate[index]),
                },
            )
            all_status.append(status)
            all_status_reason.append(reason)
            all_live_values.append(status in accepted_statuses)
        all_live = np.asarray(all_live_values, dtype=bool)
        for index, feature in enumerate(signal_fields):
            feature_rows.append(
                {
                    "split": split,
                    "index": int(index),
                    "feature": feature,
                    "specialist": groups_by_feature[index],
                    "finite": bool(all_finite_by_feature[index]),
                    "std": float(all_std[index]),
                    "value_range": float(all_value_range[index]),
                    "active_count": int(all_active_count[index]),
                    "active_rate": float(all_active_rate[index]),
                    "status": all_status[index],
                    "status_reason": all_status_reason[index],
                    "live": bool(all_live[index]),
                }
            )

        # Exact value duplicates are dead/redundant decision slots even when
        # the names and routing differ. Hash first, then byte-verify collisions.
        digest_groups: dict[str, list[int]] = {}
        for index in range(len(signal_fields)):
            values = np.ascontiguousarray(all_clean[:, index], dtype=np.float32)
            digest = hashlib.sha256(values.tobytes(order="C")).hexdigest()
            digest_groups.setdefault(digest, []).append(index)
        for digest, indices in digest_groups.items():
            if len(indices) < 2:
                continue
            verified = [
                index
                for index in indices
                if np.array_equal(all_clean[:, indices[0]], all_clean[:, index])
            ]
            if len(verified) > 1:
                duplicate_rows.append(
                    {
                        "split": split,
                        "sha256": digest,
                        "indices": verified,
                        "features": [signal_fields[index] for index in verified],
                    }
                )
        for group in required_specialists:
            idx = [i for i, owner in enumerate(groups_by_feature) if owner == group]
            features = [signal_fields[i] for i in idx]
            arr = snap[:, idx] if idx else np.empty((snap.shape[0], 0), dtype=np.float32)
            finite = np.isfinite(arr)
            clean = np.where(finite, arr, 0.0)
            std = np.std(clean, axis=0) if clean.size else np.asarray([], dtype=np.float64)
            active_rate = (
                np.mean(np.abs(clean) > float(LIVENESS_EPSILON), axis=0)
                if clean.size
                else np.asarray([], dtype=np.float64)
            )
            finite_by_feature = np.all(finite, axis=0) if finite.size else np.asarray([], dtype=bool)
            live_mask = all_live[idx] if idx else np.asarray([], dtype=bool)
            live_features = [feature for feature, live in zip(features, live_mask, strict=False) if bool(live)]
            near_constant_features = [
                feature
                for feature, value in zip(features, std, strict=False)
                if float(value) <= float(NEAR_CONSTANT_STD)
            ]
            rows.append(
                {
                    "split": split,
                    "specialist": group,
                    "feature_count": int(len(features)),
                    "live_feature_count": int(np.sum(live_mask)) if live_mask.size else 0,
                    "min_required_live_feature_count": int(MIN_LIVE_FEATURE_COUNTS.get(group, 1)),
                    "mean_active_rate": float(np.mean(active_rate)) if active_rate.size else 0.0,
                    "min_active_rate": float(np.min(active_rate)) if active_rate.size else 0.0,
                    "nonfinite_count": int((~finite).sum()) if finite.size else 0,
                    "near_constant_count": int(len(near_constant_features)),
                    "live_features": live_features,
                    "near_constant_features": near_constant_features,
                }
            )
    return rows, feature_rows, duplicate_rows


def _specialist_liveness_failures(
    rows: list[dict[str, Any]],
    splits: list[str],
    required_specialists: tuple[str, ...],
) -> list[str]:
    by_key = {
        (str(row.get("split")), str(row.get("specialist"))): row
        for row in rows
        if isinstance(row, dict)
    }
    failures: list[str] = []
    for split in splits:
        for specialist in required_specialists:
            row = by_key.get((split, specialist))
            if row is None:
                failures.append(f"{split}: specialist input liveness missing: {specialist}")
                continue
            if int(row.get("nonfinite_count") or 0) > 0:
                failures.append(
                    f"{split}: specialist input contains non-finite values: "
                    f"{specialist} nonfinite={row.get('nonfinite_count')}"
                )
            if int(row.get("live_feature_count") or 0) < int(row.get("min_required_live_feature_count") or 1):
                failures.append(
                    f"{split}: specialist live feature count below minimum: "
                    f"{specialist} live={row.get('live_feature_count')} "
                    f"min={row.get('min_required_live_feature_count')}"
                )
            if float(row.get("mean_active_rate") or 0.0) < float(MIN_SPECIALIST_MEAN_ACTIVE_RATE):
                failures.append(
                    f"{split}: specialist mean active rate too low: "
                    f"{specialist} mean_active_rate={float(row.get('mean_active_rate') or 0.0):.8f} "
                    f"min={MIN_SPECIALIST_MEAN_ACTIVE_RATE:.8f}"
                )
    return failures


def _foundation_requirement_rows(selected_features: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for requirement, spec in FOUNDATION_REQUIREMENT_PATTERNS.items():
        tokens = tuple(str(x) for x in spec["tokens"])
        expected = str(spec["expected_specialist"])
        features = [feature for feature in selected_features if any(token in feature for token in tokens)]
        specialists = Counter(classify_entry_specialist_feature(feature) for feature in features)
        rows.append(
            {
                "requirement": requirement,
                "expected_specialist": expected,
                "feature_count": int(len(features)),
                "features": features,
                "specialist_counts": dict(sorted(specialists.items())),
                "all_mapped_to_expected": bool(features) and set(specialists) == {expected},
            }
        )
    return rows


def _foundation_objective_routing_rows(selected_features: list[str]) -> list[dict[str, Any]]:
    selected = set(selected_features)
    rows: list[dict[str, Any]] = []
    for objective, required_features in REQUIRED_FOUNDATION_OBJECTIVE_FEATURES.items():
        expected = str(FOUNDATION_OBJECTIVE_SPECIALISTS.get(objective) or "")
        feature_rows = []
        missing = []
        misrouted = []
        for feature in required_features:
            present = feature in selected
            specialist = classify_entry_specialist_feature(feature)
            ok = bool(present and specialist == expected)
            feature_rows.append(
                {
                    "feature": feature,
                    "present": present,
                    "specialist": specialist,
                    "expected_specialist": expected,
                    "ok": ok,
                }
            )
            if not present:
                missing.append(feature)
            elif specialist != expected:
                misrouted.append(
                    {
                        "feature": feature,
                        "specialist": specialist,
                        "expected_specialist": expected,
                    }
                )
        rows.append(
            {
                "objective": objective,
                "expected_specialist": expected,
                "required_count": int(len(required_features)),
                "present_count": int(len(required_features) - len(missing)),
                "routed_to_expected_count": int(sum(1 for row in feature_rows if row["ok"])),
                "missing_count": int(len(missing)),
                "misrouted_count": int(len(misrouted)),
                "missing": missing,
                "misrouted": misrouted,
                "features": feature_rows,
                "all_present_and_routed_to_expected": bool(not missing and not misrouted and expected),
            }
        )
    return rows


def _specialist_model_contract_failures(
    contract: dict[str, Any],
    required_specialists: tuple[str, ...],
) -> list[str]:
    failures: list[str] = []
    required = set(required_specialists)
    actual = set(contract)
    if actual != required:
        failures.append(
            "specialist model contract has wrong trainable specialist set: "
            f"missing={sorted(required - actual)} extra={sorted(actual - required)}"
        )
    objective_owner: dict[str, str] = {}
    active_heads = set(SPECIALIST_FUSION_ACTIVE_HEADS)
    for specialist in required_specialists:
        spec = contract.get(specialist) if isinstance(contract.get(specialist), dict) else {}
        if not str(spec.get("model_role") or ""):
            failures.append(f"{specialist}: specialist model contract missing model_role")
        families = tuple(str(x) for x in spec.get("primary_signal_families") or () if str(x))
        if not families:
            failures.append(f"{specialist}: specialist model contract missing primary_signal_families")
        supports_heads = tuple(str(x) for x in spec.get("supports_heads") or () if str(x))
        if not supports_heads:
            failures.append(f"{specialist}: specialist model contract missing supports_heads")
        if supports_heads != SPECIALIST_SHARED_REACHABLE_HEADS:
            failures.append(
                f"{specialist}: specialist reachable-head topology mismatch: "
                f"observed={list(supports_heads)} "
                f"expected={list(SPECIALIST_SHARED_REACHABLE_HEADS)}"
            )
        for objective in tuple(str(x) for x in spec.get("owned_objectives") or () if str(x)):
            expected = FOUNDATION_OBJECTIVE_SPECIALISTS.get(objective)
            if expected is None:
                failures.append(f"{specialist}: specialist model contract owns unknown objective: {objective}")
                continue
            if str(expected) != specialist:
                failures.append(
                    f"{specialist}: specialist model contract owns objective expected for {expected}: {objective}"
                )
            previous = objective_owner.get(objective)
            if previous and previous != specialist:
                failures.append(
                    f"specialist model contract objective has multiple owners: {objective} {previous}/{specialist}"
                )
            objective_owner[objective] = specialist
    for objective, expected_specialist in FOUNDATION_OBJECTIVE_SPECIALISTS.items():
        observed = objective_owner.get(objective)
        if observed != expected_specialist:
            failures.append(
                f"specialist model contract objective owner mismatch: "
                f"{objective} observed={observed} expected={expected_specialist}"
            )
    return failures


def _architecture(signal_fields: list[str]) -> dict[str, Any]:
    rows = _feature_rows(signal_fields)
    by_group: dict[str, list[int]] = {group: [] for group in SPECIALIST_GROUPS}
    by_group["unmapped"] = []
    for row in rows:
        by_group.setdefault(str(row["specialist"]), []).append(int(row["index"]))
    return {
        "input_dim": int(len(signal_fields)),
        "seq_len": 96,
        "specialist_input_indices": by_group,
        "recommended_fusion": {
            "type": "cross_attended_dynamic_gated_specialists_plus_five_tf_cooperation",
            "gate_context": ["session_id", "vol_regime_id", "atr_bucket", "spread_bucket", "H4_trend_sign_cat"],
            "heads": list(SPECIALIST_FUSION_ACTIVE_HEADS),
            "active_heads": list(SPECIALIST_FUSION_ACTIVE_HEADS),
            "blocked_heads": list(SPECIALIST_FUSION_BLOCKED_HEADS),
            "direction_path": "specialist cross-attention -> dynamic specialist gate -> specialist+five-TF cross-attention -> 96-value learned evidence fusion -> calibrated LONG/SHORT/FLAT argmax",
            "independent_timeframe_only_head": "mtf_direction",
        },
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Specialist Feature Group Audit",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Contract mode: `{report['contract_mode']}`",
        f"- Signal dim: `{report['signal_field_count']}`",
        f"- Selected extension features: `{report['selected_feature_count']}`",
        f"- Training allowed by this audit: `{report['training_allowed']}`",
        f"- Failure count: `{len(report['failures'])}`",
        "",
        "## Failures",
        "",
    ]
    if report["failures"]:
        lines.extend([f"- {failure}" for failure in report["failures"]])
    else:
        lines.append("- None")
    lines.extend(["", "## Specialist Counts", ""])
    for row in report["specialist_counts"]:
        lines.append(
            f"- `{row['specialist']}`: signal={row['signal_feature_count']} "
            f"selected={row['selected_extension_count']}"
        )
    lines.extend(["", "## Foundation Requirements", ""])
    for row in report["foundation_requirements"]:
        lines.append(
            f"- `{row['requirement']}` -> `{row['expected_specialist']}`: "
            f"features={row['feature_count']} ok={row['all_mapped_to_expected']}"
        )
    lines.extend(["", "## Exact Objective Routing", ""])
    for row in report["foundation_objective_routing"]:
        lines.append(
            f"- `{row['objective']}` -> `{row['expected_specialist']}`: "
            f"routed={row['routed_to_expected_count']}/{row['required_count']} "
            f"missing={row['missing_count']} misrouted={row['misrouted_count']}"
        )
    lines.extend(["", "## Specialist Model Contract", ""])
    model_contract = report.get("specialist_model_contract") if isinstance(report.get("specialist_model_contract"), dict) else {}
    for specialist, spec in model_contract.items():
        owned = ", ".join(str(x) for x in spec.get("owned_objectives") or []) or "diagnostic/support"
        heads = ", ".join(str(x) for x in spec.get("supports_heads") or [])
        lines.append(
            f"- `{specialist}`: role={spec.get('model_role')} owned={owned} supports={heads}"
        )
    if report.get("smart_family_contract_required"):
        lines.extend(["", "## Smart Family Contract", ""])
        for row in report.get("smart_family_contract_rows") or []:
            lines.append(
                f"- `{row['family']}`: observed={row['observed_feature_count']} "
                f"expected={row['expected_feature_count']} ok={row['feature_count_matches']}"
            )
    lines.extend(["", "## Specialist Input Liveness", ""])
    for row in report["specialist_input_liveness"]:
        lines.append(
            f"- `{row['split']}` `{row['specialist']}`: live={row['live_feature_count']}/"
            f"{row['feature_count']} min={row['min_required_live_feature_count']} "
            f"mean_active={row['mean_active_rate']:.6f} nonfinite={row['nonfinite_count']}"
        )
    lines.extend(["", "## Context Routing", ""])
    context_unmapped = [str(x) for x in report.get("context_routing_unmapped_fields") or []]
    lines.append(f"- Unmapped context fields: `{len(context_unmapped)}`")
    if context_unmapped:
        lines.append(f"- Fields: `{', '.join(context_unmapped[:40])}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    seq_manifest_path = Path(args.seq_structure_manifest).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = list(FOUNDATION_AUDIT_DATA_SPLITS)
    contract_mode = SPECIALIST_CONTRACT_MODE
    required_training_specialists = MODEL_NATIVE_TRAINING_SPECIALISTS
    specialist_model_contract = MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT

    failures: list[str] = []
    split_artifacts = require_dataset_split_artifacts(
        dataset_dir,
        {
            split: {
                "manifest_path": getattr(args, f"{split}_manifest_json"),
                "manifest_sha256": getattr(args, f"{split}_manifest_sha256"),
                "parquet_sha256": getattr(args, f"{split}_parquet_sha256"),
            }
            for split in splits
        },
        expected_splits=splits,
        context="SPECIALIST_FEATURE_AUDIT_DATASET",
    )
    seq_manifest = _read_json(seq_manifest_path)
    selected_features = [str(x) for x in seq_manifest.get("selected_features", []) if str(x).strip()]
    if not selected_features:
        raise RuntimeError(f"sequence structure manifest has no selected_features: {seq_manifest_path}")
    split_contracts = _load_split_signal_fields(split_artifacts, splits)
    context_contracts = _load_split_context_fields(split_artifacts, splits)
    first_split = splits[0]
    signal_fields = list(split_contracts[first_split]["fields"])
    signal_set = set(signal_fields)
    selected_set = set(selected_features)

    for split, contract in split_contracts.items():
        fields = contract["fields"]
        if fields != signal_fields:
            failures.append(f"{split}: emitted signal fields differ from {first_split}")
        if int(contract["seq_input_dim"]) != len(fields) or int(contract["snap_input_dim"]) != len(fields):
            failures.append(
                f"{split}: manifest dims seq/snap={contract['seq_input_dim']}/{contract['snap_input_dim']} "
                f"do not match field_count={len(fields)}"
            )
        missing_ext = [feature for feature in selected_features if feature not in set(fields)]
        if missing_ext:
            failures.append(f"{split}: selected extension features missing from signal fields: {missing_ext[:30]} total={len(missing_ext)}")

    signal_owner_by_field = {
        feature: classify_entry_specialist_feature(feature)
        for feature in signal_fields
    }
    forbidden_bridge_fields = [
        feature
        for feature, owner in signal_owner_by_field.items()
        if owner == FORBIDDEN_LEGACY_BRIDGE_SPECIALIST
    ]
    if forbidden_bridge_fields:
        failures.append(
            "forbidden legacy bridge fields present in model-native signal surface: "
            f"{forbidden_bridge_fields}"
        )
    signal_unmapped_fields = [
        feature
        for feature, owner in signal_owner_by_field.items()
        if owner not in SPECIALIST_GROUPS
    ]
    if signal_unmapped_fields:
        failures.append(f"unmapped signal fields: {signal_unmapped_fields[:30]} total={len(signal_unmapped_fields)}")

    specialist_counts = _count_rows(signal_fields, selected_features)
    count_by_group = {row["specialist"]: int(row["signal_feature_count"]) for row in specialist_counts}
    selected_count_by_group = {row["specialist"]: int(row["selected_extension_count"]) for row in specialist_counts}
    for group in required_training_specialists:
        min_count = int(MIN_SIGNAL_COUNTS.get(group, 1))
        if count_by_group.get(group, 0) < min_count:
            failures.append(f"{group}: signal feature count below minimum {min_count}: {count_by_group.get(group, 0)}")
    for group in ("structure_swing_encoder", "smc_liquidity_encoder", "session_regime_encoder"):
        if selected_count_by_group.get(group, 0) <= 0:
            failures.append(f"{group}: no selected sequence-extension features assigned")

    (
        specialist_input_liveness,
        signal_feature_liveness,
        exact_duplicate_signal_groups,
    ) = _specialist_input_liveness_rows(
        split_artifacts,
        splits,
        signal_fields,
        required_training_specialists,
    )
    specialist_input_liveness_failures = _specialist_liveness_failures(
        specialist_input_liveness,
        splits,
        required_training_specialists,
    )
    failures.extend(specialist_input_liveness_failures)
    train_dead_features = [
        str(row["feature"])
        for row in signal_feature_liveness
        if row.get("split") == "train" and not bool(row.get("live"))
    ]
    if train_dead_features:
        failures.append(
            "train: every one of the 513 signal fields must satisfy canonical "
            "finite/variable/activity support; unlearnable fields="
            f"{train_dead_features[:30]} total={len(train_dead_features)}"
        )
    train_duplicate_groups = [
        row for row in exact_duplicate_signal_groups if row.get("split") == "train"
    ]
    if train_duplicate_groups:
        failures.append(
            "train: exact duplicate signal value columns are forbidden: "
            f"{[row['features'] for row in train_duplicate_groups[:10]]} "
            f"total_groups={len(train_duplicate_groups)}"
        )
    context_routing_rows = _context_routing_rows(context_contracts)
    context_routing_unmapped_fields = [
        str(row.get("feature"))
        for row in context_routing_rows
        if str(row.get("specialist")) == "unmapped"
    ]
    context_routing_failures = _context_routing_failures(context_routing_rows)
    failures.extend(context_routing_failures)

    foundation_rows = _foundation_requirement_rows(selected_features)
    for row in foundation_rows:
        if not row["features"]:
            failures.append(f"foundation requirement has no selected features: {row['requirement']}")
        if not row["all_mapped_to_expected"]:
            failures.append(
                f"foundation requirement {row['requirement']} not fully mapped to {row['expected_specialist']}: "
                f"{row['specialist_counts']}"
            )
    objective_routing_rows = _foundation_objective_routing_rows(selected_features)
    for row in objective_routing_rows:
        if not row["all_present_and_routed_to_expected"]:
            failures.append(
                f"foundation objective {row['objective']} not exactly routed to {row['expected_specialist']}: "
                f"missing={row['missing_count']} misrouted={row['misrouted_count']}"
            )
    specialist_model_contract_failures = _specialist_model_contract_failures(
        specialist_model_contract,
        required_training_specialists,
    )
    failures.extend(specialist_model_contract_failures)
    contract_training_surface = _contract_training_surface()
    smart_family_contract = MODEL_NATIVE_SMART_FAMILY_CONTRACT
    smart_family_rows = _smart_family_contract_rows(
        selected_features,
        signal_fields,
    )
    smart_contract_failures = _smart_contract_failures(
        seq_manifest,
        signal_field_count=len(signal_fields),
        selected_feature_count=len(selected_features),
        smart_family_rows=smart_family_rows,
    )
    failures.extend(smart_contract_failures)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "entry_specialist_feature_group_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS" if not failures else "FAIL",
        **foundation_audit_policy_binding(),
        "foundation_audit_policy_enforcement": (
            foundation_audit_policy_enforcement("specialist")
        ),
        "report_only": True,
        "training_allowed": False,
        "training_allowed_reason": "specialist feature-group audit is report-only; training requires separate readiness gates",
        "dataset_dir": str(dataset_dir),
        "split_artifacts_schema_version": (
            ENTRY_DATASET_SPLIT_ARTIFACTS_SCHEMA_VERSION
        ),
        "split_artifacts": split_artifacts,
        "seq_structure_manifest": str(seq_manifest_path),
        "contract_mode": contract_mode,
        "contract_training_surface": contract_training_surface,
        "data_splits": splits,
        "signal_field_count": int(len(signal_fields)),
        "signal_unmapped_count": int(len(signal_unmapped_fields)),
        "signal_unmapped_fields": signal_unmapped_fields,
        "forbidden_legacy_bridge_field_count": int(len(forbidden_bridge_fields)),
        "forbidden_legacy_bridge_fields": forbidden_bridge_fields,
        "signal_routing_all_mapped": not signal_unmapped_fields,
        "selected_feature_count": int(len(selected_features)),
        "selected_features_present_in_signal_count": int(len(selected_set & signal_set)),
        "required_training_specialists": list(required_training_specialists),
        "specialist_groups": SPECIALIST_GROUPS,
        "specialist_model_contract": specialist_model_contract,
        "specialist_model_contract_valid": not specialist_model_contract_failures,
        "specialist_model_contract_failures": specialist_model_contract_failures,
        "smart_family_contract": smart_family_contract,
        "smart_family_contract_required": bool(smart_family_contract),
        "smart_family_contract_rows": smart_family_rows,
        "smart_family_contract_valid": not smart_contract_failures,
        "smart_family_contract_failures": smart_contract_failures,
        "specialist_counts": specialist_counts,
        "specialist_input_liveness": specialist_input_liveness,
        "specialist_input_liveness_all_live": not specialist_input_liveness_failures,
        "signal_feature_liveness": signal_feature_liveness,
        "train_dead_signal_feature_count": int(len(train_dead_features)),
        "train_dead_signal_features": train_dead_features,
        "every_train_signal_feature_live": not train_dead_features,
        "exact_duplicate_signal_groups": exact_duplicate_signal_groups,
        "train_exact_duplicate_signal_group_count": int(
            len(train_duplicate_groups)
        ),
        "no_train_exact_duplicate_signal_values": not train_duplicate_groups,
        "context_contracts": {
            split: {
                "manifest_path": contract.get("manifest_path"),
                "ctx_tag": contract.get("ctx_tag"),
                "ctx_cont_dim": contract.get("ctx_cont_dim"),
                "ctx_cat_dim": contract.get("ctx_cat_dim"),
            }
            for split, contract in context_contracts.items()
        },
        "context_routing": context_routing_rows,
        "context_routing_unmapped_count": int(len(context_routing_unmapped_fields)),
        "context_routing_unmapped_fields": context_routing_unmapped_fields,
        "context_routing_all_mapped": not context_routing_unmapped_fields,
        "context_routing_failures": context_routing_failures,
        "min_live_feature_counts": MIN_LIVE_FEATURE_COUNTS,
        "min_specialist_mean_active_rate": float(MIN_SPECIALIST_MEAN_ACTIVE_RATE),
        "min_feature_active_rate": float(MIN_FEATURE_ACTIVE_RATE),
        "foundation_requirements": foundation_rows,
        "foundation_objective_routing": objective_routing_rows,
        "foundation_objective_routing_all_present_and_expected": all(
            bool(row["all_present_and_routed_to_expected"])
            for row in objective_routing_rows
        ),
        "feature_rows": _feature_rows(signal_fields),
        "split_contracts": {
            split: {k: v for k, v in contract.items() if k not in {"fields", "extension_features"}}
            for split, contract in split_contracts.items()
        },
        "architecture_contract": _architecture(signal_fields),
        "failures": failures,
    }

    json_path = out_dir / f"ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_{timestamp}.json"
    md_path = out_dir / f"ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_{timestamp}.md"
    if json_path.exists() or md_path.exists():
        raise RuntimeError(
            "SPECIALIST_AUDIT_IMMUTABLE_EVENT_EXISTS: "
            f"json={json_path} md={md_path}"
        )
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "signal_field_count": report["signal_field_count"],
                    "selected_feature_count": report["selected_feature_count"],
                    "failures": report["failures"],
                    "json_path": report["json_path"],
                    "md_path": report["md_path"],
                },
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        )
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
