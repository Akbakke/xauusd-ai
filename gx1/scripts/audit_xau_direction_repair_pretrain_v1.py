#!/usr/bin/env python3
"""Fail-closed XAU future-outcome target audit before model-native training."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gx1.contracts.entry_model_native_smoke_bundle_audit_v1 import (
    PRETRAIN_AUDIT_SCHEMA,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    model_native_signal_contract_failures,
)
from gx1.contracts.entry_model_native_state_v2 import (
    validate_state_contract_metadata_v2,
)
from gx1.contracts.entry_exit_feature_surface_v1 import (
    ENTRY_M5_FEATURE_SURFACE_CONSUMPTION_MODE,
)
from gx1.contracts.xau_tape_provenance_v1 import (
    canonical_json_sha256,
    validate_xau_tape_provenance_v1,
)
from gx1.contracts.entry_direction_target_policy_v1 import (
    apply_entry_direction_target_rule,
)
from gx1.contracts.entry_fitted_q_v1 import (
    require_entry_fitted_q_contract,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_AUX_TARGET_COLUMNS,
)
from gx1.contracts.entry_causal_m1_position_size_target_policy_v1 import (
    require_causal_m1_position_size_target_manifest_binding,
)
from gx1.contracts.entry_position_size_target_policy_v1 import (
    require_entry_position_size_target_manifest_binding,
)
from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    ENTRY_DIRECTION_TARGET_MODE_ID,
    ENTRY_FITTED_Q_DATASET_STEM_SUFFIX,
)
import pyarrow.parquet as pq
from gx1.contracts.entry_pretrain_polarity_signal_v1 import (
    PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS,
    RESISTANCE_DISTANCE_FEATURE,
    SUPPORT_DISTANCE_FEATURE,
)


DEFAULT_STEM = f"v10_6yr_dataset{ENTRY_FITTED_Q_DATASET_STEM_SUFFIX}"
REQUIRED_POLARITY_FEATURES = PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS
# V30 package 7 (2026-08-13): the previous `REQUIRED_RAIL_FEATURES` filtered
# the mandatory geometry tuple on the substring `_rail_`.  Those six fields
# were removed as NAME-ONLY (no rail, no slope, no line), which would have left
# the filter EMPTY and the proof silently vacuous - a fail-open gate.  The proof
# is re-pointed at the complete mandatory geometry tuple, which is the property
# it was really guarding: every code-owned mandatory chart-geometry field is
# present in the split's signal manifest.
REQUIRED_MANDATORY_LEVEL_FEATURES = tuple(REQUIRED_POLARITY_FEATURES)
if not REQUIRED_MANDATORY_LEVEL_FEATURES:
    raise RuntimeError("XAU_PRETRAIN_AUDIT_MANDATORY_LEVEL_FEATURES_EMPTY")
# The columns the restored target-consistency proof re-derives and compares.
# They are still emitted by the builder even though the direction labels are
# now declared diagnostic (`entry_action_authority: False`): a demoted column
# that other audits still read must still be proved consistent, or those
# audits report a wrong number from a silently broken input (rule 25).
DIRECTION_CONSISTENCY_TARGET_COLUMNS = (
    "y_direction",
    "y_trade",
    "y_side",
    "y_bad_path",
    "y_long_bad_path",
    "y_short_bad_path",
    "y_direction_target_mode_id",
    "y_direction_long_score_bps",
    "y_direction_short_score_bps",
    "y_long_path_utility_bps",
    "y_short_path_utility_bps",
    "y_long_final_pnl_at_direction_horizon_bps",
    "y_short_final_pnl_at_direction_horizon_bps",
)
REQUIRED_XAU_TARGET_COLUMNS = (
    "y_long_expected_mae_bps",
    "y_short_expected_mae_bps",
    "y_position_size_target",
    "y_position_size_mask",
    "y_line_support_touch_held",
    "y_line_support_touch_mask",
    "y_line_resistance_touch_held",
    "y_line_resistance_touch_mask",
    "y_countertrend_short_trap",
    "y_countertrend_long_trap",
    *DIRECTION_CONSISTENCY_TARGET_COLUMNS,
    *MODEL_NATIVE_AUX_TARGET_COLUMNS,
)
# These columns are required and audited, but are not required to have two
# distinct values.  This is not a generic constant-column escape hatch:
#
# * ``y_direction_target_mode_id`` is an exact schema identity checked by the
#   target-consistency proof.
# * ``y_bad_path`` is the retired selected-side scalar diagnostic.  The fitted
#   direction policy only selects a side whose spread-aware horizon PnL clears
#   a positive edge floor, while bad-path means that same PnL is negative.
#   Therefore the selected-side scalar is structurally zero when the policy is
#   coherent.  Its two counterfactual source targets remain liveness-required,
#   and the consistency proof requires an exact finite selected-side copy.
#
# Adding another exemption requires an explicit code/schema change and tests.
TARGET_LIVENESS_EXEMPTION_REASONS = {
    "y_direction_target_mode_id": (
        "constant_schema_identity_checked_by_target_consistency"
    ),
    "y_bad_path": (
        "retired_selected_side_diagnostic_checked_as_finite_exact_copy_of_"
        "live_side_specific_targets"
    ),
}
TARGET_LIVENESS_EXEMPT_COLUMNS = frozenset(
    TARGET_LIVENESS_EXEMPTION_REASONS
)
# Tolerances follow the convention this audit already used before the V30
# rewrite: the copy identities compare float32 columns that the builder writes
# from one and the same array, and the policy-derived comparison allows one
# extra float32 arithmetic step.  Neither is a fitted or chosen magnitude.
TARGET_COPY_IDENTITY_TOLERANCE = 1e-5
TARGET_POLICY_DERIVED_TOLERANCE = 1e-4
PREFREEZE_SPLITS = ("train", "val")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj) if np.isfinite(obj) else None
    return str(obj)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_fields(manifest: dict[str, Any]) -> list[str]:
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    bridge = extra.get("signal_bridge") if isinstance(extra.get("signal_bridge"), dict) else {}
    for source in (
        bridge.get("fields"),
        bridge.get("snap_fields"),
        (manifest.get("feature_contract") or {}).get("signal_bridge_fields")
        if isinstance(manifest.get("feature_contract"), dict)
        else None,
    ):
        if isinstance(source, list) and all(isinstance(item, str) for item in source):
            return list(source)
    raise RuntimeError("manifest lacks signal_bridge feature fields")


def _seq_structure_mode(manifest: dict[str, Any]) -> str | None:
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    bridge = extra.get("signal_bridge") if isinstance(extra.get("signal_bridge"), dict) else {}
    for source in (
        bridge.get("seq_structure_extension_v1"),
        extra.get("seq_structure_extension_v1"),
    ):
        if isinstance(source, dict) and str(source.get("mode") or "").strip():
            return str(source.get("mode")).strip()
    return None


def _manifest_provenance(manifest: dict[str, Any]) -> dict[str, Any]:
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
    signal_surface = (
        extra.get("signal_bridge")
        if isinstance(extra.get("signal_bridge"), dict)
        else {}
    )
    state_contract = extra.get("model_native_state_contract")
    return {
        "tape_root": str(
            manifest.get("tape_root")
            or extra.get("tape_root")
            or inputs.get("tape_root")
            or ""
        ),
        "contract_mode": str(extra.get("contract_mode") or ""),
        "direction_logit_mode": str(extra.get("direction_logit_mode") or ""),
        "model_native_signal_contract": extra.get("model_native_signal_contract"),
        "xau_tape_provenance": extra.get("xau_tape_provenance"),
        "entry_fitted_q": extra.get("entry_fitted_q"),
        "entry_position_size_target_policy": extra.get(
            "entry_position_size_target_policy"
        ),
        "entry_position_size_target_policy_sha256": extra.get(
            "entry_position_size_target_policy_sha256"
        ),
        "fields": list(signal_surface.get("fields") or []),
        "splits": manifest.get("splits") if isinstance(manifest.get("splits"), dict) else {},
        "model_native_state_contract": (
            dict(state_contract) if isinstance(state_contract, dict) else {}
        ),
    }


def _position_size_target_policy(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    source_frame = (
        extra.get("source_frame")
        if isinstance(extra.get("source_frame"), dict)
        else {}
    )
    xau_provenance = extra.get("xau_tape_provenance")
    expected_source = str(source_frame.get("parquet_sha256") or "").strip().lower()
    if len(expected_source) != 64:
        raise RuntimeError("manifest source_frame parquet hash missing")
    if not isinstance(xau_provenance, dict):
        raise RuntimeError("manifest XAU tape provenance missing")
    splits = manifest.get("splits")
    train_window = (
        splits.get("train")
        if isinstance(splits, dict) and isinstance(splits.get("train"), dict)
        else {}
    )
    # Both bindings the owner offers are supplied. Passing `None` here would
    # disable a lineage check the manifest actually carries, which is the
    # soft pass-through rule 2 forbids: it is how a sizing policy fitted
    # against a different tape, or against a different direction policy,
    # would reach training unnoticed.
    expected_direction_policy = str(
        extra.get("diagnostic_outcome_policy_sha256") or ""
    ).strip().lower()
    if len(expected_direction_policy) != 64:
        raise RuntimeError(
            "manifest diagnostic_outcome_policy_sha256 missing"
        )
    if "entry_causal_m1_position_size_target_policy" not in extra:
        # Immutable legacy artifacts remain auditable; the independent
        # execution-causality launch gate blocks them from new training.
        return require_entry_position_size_target_manifest_binding(
            extra,
            expected_source_parquet_sha256=expected_source,
            expected_tape_provenance_sha256=canonical_json_sha256(xau_provenance),
            expected_direction_policy_sha256=expected_direction_policy,
            expected_train_start=train_window.get("start"),
            expected_train_end=train_window.get("end"),
        )
    lifecycle = extra.get("unified_exit_lifecycle")
    expected_m1_source = (
        str(lifecycle.get("m1_source_sha256") or "").strip().lower()
        if isinstance(lifecycle, dict)
        else ""
    )
    if len(expected_m1_source) != 64:
        raise RuntimeError("manifest unified_exit_lifecycle M1 source hash missing")
    return require_causal_m1_position_size_target_manifest_binding(
        extra,
        expected_source_parquet_sha256=expected_source,
        expected_tape_provenance_sha256=canonical_json_sha256(xau_provenance),
        expected_m1_source_sha256=expected_m1_source,
        expected_direction_policy_sha256=expected_direction_policy,
        expected_train_start=train_window.get("start"),
        expected_train_end=train_window.get("end"),
    )


def _state_contract_failures(contract: dict[str, Any], *, split: str) -> list[str]:
    if not isinstance(contract, dict) or not contract:
        return [f"{split}: XAU repair requires model_native_state_contract provenance"]
    try:
        validate_state_contract_metadata_v2(contract)
    except (RuntimeError, TypeError, ValueError, OSError) as exc:
        return [f"{split}: model_native_state_contract v2 invalid: {exc}"]
    return []


def _split_artifacts(dataset_dir: Path, stem: str, split: str) -> tuple[Path, Path]:
    parquet_path = dataset_dir / f"{stem}_{split}.parquet"
    manifest_path = dataset_dir / f"{stem}_{split}.manifest.json"
    if not parquet_path.exists():
        raise RuntimeError(f"missing parquet for split={split}: {parquet_path}")
    if not manifest_path.exists():
        raise RuntimeError(f"missing manifest for split={split}: {manifest_path}")
    return parquet_path, manifest_path


def _resolve_stem(dataset_dir: Path, requested_stem: str, splits: list[str]) -> str:
    requested = str(requested_stem or "").strip()
    if not requested or requested.lower() == "auto":
        raise RuntimeError("an explicit immutable --stem is required; discovery is forbidden")
    return requested


def _column_liveness(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    unique_count = int(np.unique(finite).size) if finite.size else 0
    std = float(np.std(finite)) if finite.size else None
    return {
        "finite_rate": float(finite.size / max(arr.size, 1)),
        "unique_count": unique_count,
        "std": std,
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "live": bool(finite.size == arr.size and unique_count >= 2 and std is not None and std > 1e-9),
    }


def _read_sample(
    parquet_path: Path,
    *,
    columns: list[str],
    max_rows: int,
    max_row_groups: int,
) -> dict[str, np.ndarray]:
    pf = pq.ParquetFile(parquet_path)
    if pf.num_row_groups <= 0:
        raise RuntimeError(f"{parquet_path} has no row groups")
    row_groups = list(range(pf.num_row_groups))
    if max_row_groups > 0 and len(row_groups) > max_row_groups:
        positions = np.linspace(0, len(row_groups) - 1, num=max_row_groups)
        row_groups = sorted({int(round(pos)) for pos in positions})

    parts: dict[str, list[np.ndarray]] = {name: [] for name in columns}
    remaining = int(max_rows)
    for rg in row_groups:
        if remaining <= 0:
            break
        table = pf.read_row_group(rg, columns=columns)
        take = min(int(table.num_rows), remaining)
        if take <= 0:
            continue
        if take < table.num_rows:
            table = table.slice(0, take)
        for name in columns:
            col = table[name].combine_chunks()
            if name == "snap":
                parts[name].append(np.asarray(col.to_pylist(), dtype=np.float32))
            else:
                parts[name].append(np.asarray(col.to_numpy(zero_copy_only=False)))
        remaining -= take

    out: dict[str, np.ndarray] = {}
    for name, chunks in parts.items():
        if not chunks:
            out[name] = np.empty((0, 0), dtype=np.float32) if name == "snap" else np.empty((0,), dtype=np.float32)
        else:
            out[name] = np.concatenate(chunks, axis=0)
    return out


def _direction_rule_floors(extra: dict[str, Any]) -> tuple[float, float]:
    """Read the TRAIN-fitted rule floors from the manifest's own projection.

    The split manifest is forbidden by the builder from carrying the direction
    policy payload (`MODEL_NATIVE_MANIFEST_RETIRED_DIRECTION_POLICY_PRESENT`),
    so the two floors it does declare are the only legitimate origin here.
    They are not defaulted: a missing or non-finite floor fails closed.
    """

    floors: list[float] = []
    for key in (
        "diagnostic_tradable_edge_floor_bps",
        "diagnostic_side_margin_floor_bps",
    ):
        value = extra.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"manifest {key} missing or not numeric")
        if not np.isfinite(float(value)):
            raise RuntimeError(f"manifest {key} is not finite")
        floors.append(float(value))
    return floors[0], floors[1]


def _target_consistency(
    sample: dict[str, np.ndarray],
    *,
    extra: dict[str, Any],
    position_size_policy_sha256: str,
) -> dict[str, Any]:
    """Re-derive the direction targets and compare them to the serialized ones.

    The rule is evaluated by the shared owner
    `entry_direction_target_policy_v1.apply_entry_direction_target_rule`, the
    same code path the dataset builder runs through
    `entry_direction_targets_from_policy`, so the audit cannot drift into a
    second label formula.
    """

    lineage = {
        "authority": "none_diagnostic_active_target_liveness_only",
        "entry_q_targets_serialized_in_dataset": False,
        "entry_q_source": "frozen_exit_first_state_target_model",
        "position_size_policy_sha256": position_size_policy_sha256,
    }
    missing = [
        name
        for name in DIRECTION_CONSISTENCY_TARGET_COLUMNS
        if name not in sample
    ]
    if missing:
        return {**lineage, "available": False, "unavailable_columns": missing}
    columns = {
        name: np.asarray(sample[name])
        for name in DIRECTION_CONSISTENCY_TARGET_COLUMNS
    }
    row_counts = sorted({int(arr.shape[0]) for arr in columns.values()})
    if len(row_counts) != 1 or row_counts[0] == 0:
        return {**lineage, "available": False, "row_counts": row_counts}

    edge_floor, margin_floor = _direction_rule_floors(extra)
    derived = apply_entry_direction_target_rule(
        long_executable_pnl_bps=columns[
            "y_long_final_pnl_at_direction_horizon_bps"
        ].astype(np.float32),
        short_executable_pnl_bps=columns[
            "y_short_final_pnl_at_direction_horizon_bps"
        ].astype(np.float32),
        tradable_edge_floor_bps=edge_floor,
        side_margin_floor_bps=margin_floor,
    )

    y_direction = columns["y_direction"].astype(np.int32)
    y_trade = columns["y_trade"].astype(np.float32)
    y_side = columns["y_side"].astype(np.int32)
    long_rows = (y_trade > 0.5) & (y_side == 0)
    short_rows = (y_trade > 0.5) & (y_side == 1)
    expected_scalar_bad = np.zeros(row_counts[0], dtype=np.float32)
    expected_scalar_bad[long_rows] = columns["y_long_bad_path"].astype(
        np.float32
    )[long_rows]
    expected_scalar_bad[short_rows] = columns["y_short_bad_path"].astype(
        np.float32
    )[short_rows]

    # The mode id is an integer schema tag, so it is compared exactly against
    # the constant owned by the module that writes it. No tolerance applies.
    mode_raw = columns["y_direction_target_mode_id"].astype(np.float64)
    mode_int = np.rint(mode_raw)
    mode_invalid = (
        ~np.isfinite(mode_raw)
        | (mode_raw != mode_int)
        | (mode_int != float(ENTRY_DIRECTION_TARGET_MODE_ID))
    )

    def _copy_mismatch(left: str, right: str) -> int:
        lhs = columns[left].astype(np.float32)
        rhs = columns[right].astype(np.float32)
        return int(
            np.sum(
                ~np.isfinite(lhs)
                | ~np.isfinite(rhs)
                | (np.abs(lhs - rhs) > TARGET_COPY_IDENTITY_TOLERANCE)
            )
        )

    def _policy_mismatch(column: str, expected: np.ndarray) -> int:
        observed = columns[column].astype(np.float32)
        expected = np.asarray(expected, dtype=np.float32)
        return int(
            np.sum(
                ~np.isfinite(observed)
                | ~np.isfinite(expected)
                | (np.abs(observed - expected) > TARGET_POLICY_DERIVED_TOLERANCE)
            )
        )

    return {
        **lineage,
        "available": True,
        "rows": row_counts[0],
        "direction_rule_owner": (
            "gx1.contracts.entry_direction_target_policy_v1."
            "apply_entry_direction_target_rule"
        ),
        "tradable_edge_floor_bps": edge_floor,
        "side_margin_floor_bps": margin_floor,
        "direction_outcome_mismatch_count": int(
            np.sum(y_direction != derived["direction"])
        ),
        "bad_path_side_mismatch_count": int(
            np.sum(
                ~np.isfinite(columns["y_bad_path"].astype(np.float32))
                | ~np.isfinite(expected_scalar_bad)
                | (
                    np.abs(
                        columns["y_bad_path"].astype(np.float32)
                        - expected_scalar_bad
                    )
                    > TARGET_COPY_IDENTITY_TOLERANCE
                )
            )
        ),
        "direction_long_score_alias_mismatch_count": _copy_mismatch(
            "y_direction_long_score_bps", "y_long_path_utility_bps"
        ),
        "direction_short_score_alias_mismatch_count": _copy_mismatch(
            "y_direction_short_score_bps", "y_short_path_utility_bps"
        ),
        "long_score_executable_pnl_mismatch_count": _policy_mismatch(
            "y_direction_long_score_bps", derived["long_score_bps"]
        ),
        "short_score_executable_pnl_mismatch_count": _policy_mismatch(
            "y_direction_short_score_bps", derived["short_score_bps"]
        ),
        "target_mode_contract_invalid_count": int(mode_invalid.sum()),
    }


def _audit_split(
    *,
    split: str,
    parquet_path: Path,
    manifest_path: Path,
    max_rows: int,
    max_row_groups: int,
    distance_dominance_margin_atr: float,
    min_pocket_rows: int,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    require_entry_fitted_q_contract(
        extra.get("entry_fitted_q"),
        context=f"XAU_PRETRAIN_{split.upper()}",
    )
    position_size_target_policy = _position_size_target_policy(manifest)
    fields = _feature_fields(manifest)
    feature_index = {name: (fields.index(name) if name in fields else None) for name in REQUIRED_POLARITY_FEATURES}
    schema_names = set(pq.read_schema(parquet_path).names)
    missing_targets = [name for name in REQUIRED_XAU_TARGET_COLUMNS if name not in schema_names]
    sample_columns = (
        ["snap"]
        + [name for name in REQUIRED_XAU_TARGET_COLUMNS if name in schema_names]
    )
    sample = _read_sample(
        parquet_path,
        columns=sample_columns,
        max_rows=max_rows,
        max_row_groups=max_row_groups,
    )
    snap = np.asarray(sample["snap"], dtype=np.float32)
    if snap.ndim != 2:
        raise RuntimeError(f"{parquet_path} snap sample is not 2D: {snap.shape}")
    missing_features = [name for name, idx in feature_index.items() if idx is None or idx >= snap.shape[1]]
    polarity: dict[str, Any] = {"available": False}
    if not missing_features:
        # The pocket is measured directly from the two raw nearest-level
        # distances. Deleted channel/Fibonacci scorebooks are not recreated as
        # zero placeholders or audit-side weighted composites.
        support_distance = snap[:, int(feature_index[SUPPORT_DISTANCE_FEATURE])]
        resistance_distance = snap[:, int(feature_index[RESISTANCE_DISTANCE_FEATURE])]
        margin = float(distance_dominance_margin_atr)
        support_dom = (resistance_distance - support_distance) > margin
        resistance_dom = (support_distance - resistance_distance) > margin
        polarity = {
            "available": True,
            "distance_dominance_margin_atr": margin,
            "pocket_definition": (
                "nearest_opposite_side_distance_minus_nearest_side_distance"
            ),
            "support_dominant_rows": int(support_dom.sum()),
            "resistance_dominant_rows": int(resistance_dom.sum()),
            "enough_pocket_rows": bool(
                int(support_dom.sum()) >= int(min_pocket_rows)
                and int(resistance_dom.sum()) >= int(min_pocket_rows)
            ),
        }

    target_liveness = {
        name: _column_liveness(np.asarray(sample[name]))
        for name in REQUIRED_XAU_TARGET_COLUMNS
        if name in sample and name != "y_direction_target_mode_id"
    }
    target_consistency = _target_consistency(
        sample,
        extra=extra,
        position_size_policy_sha256=position_size_target_policy["policy_sha256"],
    )

    return {
        "split": split,
        "parquet_path": str(parquet_path),
        "manifest_path": str(manifest_path),
        "rows_sampled": int(snap.shape[0]),
        "feature_count": int(len(fields)),
        "seq_structure_extension_mode": _seq_structure_mode(manifest),
        "provenance": _manifest_provenance(manifest),
        "feature_index": feature_index,
        "missing_polarity_features": missing_features,
        "missing_target_columns": missing_targets,
        "polarity": polarity,
        "target_liveness": target_liveness,
        "target_consistency": target_consistency,
        "core_target_policy": "frozen_exit_first_state_fitted_q_only",
        "entry_position_size_target_policy_sha256": position_size_target_policy[
            "policy_sha256"
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = list(PREFREEZE_SPLITS)
    failures: list[str] = []
    requested_stem = str(args.stem)
    stem: str | None = None
    try:
        stem = _resolve_stem(dataset_dir, requested_stem, splits)
    except Exception as exc:
        failures.append(f"stem resolution failed: {exc}")

    split_reports: list[dict[str, Any]] = []
    manifest_fields: list[str] | None = None
    missing_mandatory_level_features: list[str] = []
    if stem is not None:
        for split in splits:
            try:
                parquet_path, manifest_path = _split_artifacts(dataset_dir, stem, split)
                manifest = _load_json(manifest_path)
                fields = _feature_fields(manifest)
                if manifest_fields is None:
                    manifest_fields = fields
                    missing_mandatory_level_features = [
                        name
                        for name in REQUIRED_MANDATORY_LEVEL_FEATURES
                        if name not in fields
                    ]
                split_report = _audit_split(
                    split=split,
                    parquet_path=parquet_path,
                    manifest_path=manifest_path,
                    max_rows=int(args.max_rows_per_split),
                    max_row_groups=int(args.max_row_groups_per_split),
                    distance_dominance_margin_atr=float(
                        args.distance_dominance_margin_atr
                    ),
                    min_pocket_rows=int(args.min_pocket_rows),
                )
                split_reports.append(split_report)
            except Exception as exc:
                failures.append(f"{split}: audit failed: {exc}")

    if missing_mandatory_level_features:
        failures.append(
            "missing required XAU mandatory level features in manifest: "
            f"{missing_mandatory_level_features}"
        )

    stale_markers = ("utilityrepair", "20260710", "smart_candidate_20260630", "julyext")
    dataset_dir_text = str(dataset_dir).lower()
    for marker in stale_markers:
        if marker in dataset_dir_text:
            failures.append(f"dataset_dir contains known stale XAU repair marker {marker!r}: {dataset_dir}")

    tape_provenance_cache: dict[tuple[str, str], dict[str, Any]] = {}
    tape_provenance_by_split: dict[str, dict[str, Any]] = {}
    for row in split_reports:
        split = str(row.get("split"))
        seq_mode = row.get("seq_structure_extension_mode")
        if seq_mode != ENTRY_M5_FEATURE_SURFACE_CONSUMPTION_MODE:
            failures.append(
                f"{split}: XAU repair requires the exact native M5 feature "
                f"surface; observed mode={seq_mode}"
            )
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        if str(provenance.get("contract_mode") or "") != MODEL_NATIVE_CONTRACT_MODE:
            failures.append(f"{split}: contract_mode must be {MODEL_NATIVE_CONTRACT_MODE}")
        if (
            str(provenance.get("direction_logit_mode") or "")
            != MODEL_NATIVE_DIRECTION_LOGIT_MODE
        ):
            failures.append(
                f"{split}: direction_logit_mode must be {MODEL_NATIVE_DIRECTION_LOGIT_MODE}"
            )
        signal_contract = provenance.get("model_native_signal_contract")
        signal_failures = model_native_signal_contract_failures(
            signal_contract if isinstance(signal_contract, dict) else {}
        )
        failures.extend(f"{split}: {failure}" for failure in signal_failures)
        try:
            require_entry_fitted_q_contract(
                provenance.get("entry_fitted_q"),
                context=f"XAU_PRETRAIN_PROVENANCE_{split.upper()}",
            )
        except RuntimeError as exc:
            failures.append(
                f"{split}: Entry fitted-Q contract invalid: {exc}"
            )
        forbidden = sorted(
            set(provenance.get("fields") or ()) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS)
        )
        if forbidden:
            failures.append(f"{split}: forbidden legacy bridge fields: {forbidden}")
        state_contract = provenance.get("model_native_state_contract")
        state_contract = state_contract if isinstance(state_contract, dict) else {}
        failures.extend(_state_contract_failures(state_contract, split=split))
        tape_root = str(provenance.get("tape_root") or "").strip()
        expected_run_id = str(state_contract.get("entry_run_id") or "").strip()
        cache_key = (tape_root, expected_run_id)
        try:
            if cache_key not in tape_provenance_cache:
                tape_provenance_cache[cache_key] = validate_xau_tape_provenance_v1(
                    tape_root,
                    expected_run_id=expected_run_id,
                    require_current=True,
                )
            tape_provenance_by_split[split] = tape_provenance_cache[cache_key]
            declared_tape_provenance = provenance.get("xau_tape_provenance")
            if declared_tape_provenance != tape_provenance_by_split[split]:
                failures.append(
                    f"{split}: dataset manifest XAU_USD tape binding differs from "
                    "the revalidated immutable tape lineage"
                )
        except (RuntimeError, OSError, ValueError) as exc:
            failures.append(f"{split}: immutable XAU_USD tape provenance invalid: {exc}")
        splits_contract = provenance.get("splits")
        train_window = (
            splits_contract.get("train")
            if isinstance(splits_contract, dict)
            and isinstance(splits_contract.get("train"), dict)
            else {}
        )
        if not train_window:
            failures.append(f"{split}: manifest missing exact TRAIN split window")
        # The TRAIN-only rank-fit window check that stood here read
        # `rank_fit_start_utc` / `rank_fit_end_utc` from the state contract.
        # Both names are now in `RETIRED_RANK_STATE_FIELDS`, which the state
        # contract rejects outright, so this audit could never pass: it
        # demanded exactly the fields the surface forbids. There is no fixed
        # TRAIN-ranked selection left to bound — every code-owned candidate is
        # available — so the check has no subject rather than a new owner.
        for name in row.get("missing_polarity_features") or []:
            failures.append(f"{split}: missing level-distance polarity feature: {name}")
        for name in row.get("missing_target_columns") or []:
            failures.append(f"{split}: missing XAU future-outcome target column: {name}")
        for name, live in (row.get("target_liveness") or {}).items():
            if name in TARGET_LIVENESS_EXEMPT_COLUMNS:
                if float(live.get("finite_rate") or 0.0) != 1.0:
                    failures.append(
                        f"{split}: liveness-exempt XAU target column is not "
                        f"fully finite: {name}"
                    )
                continue
            if not bool(live.get("live")):
                failures.append(
                    f"{split}: XAU future-outcome target column is not live: {name}"
                )
        consistency = row.get("target_consistency") if isinstance(row.get("target_consistency"), dict) else {}
        if (
            consistency.get("authority")
            != "none_diagnostic_active_target_liveness_only"
            or consistency.get("entry_q_targets_serialized_in_dataset")
            is not False
            or consistency.get("entry_q_source")
            != "frozen_exit_first_state_target_model"
            or consistency.get("position_size_policy_sha256")
            != row.get("entry_position_size_target_policy_sha256")
        ):
            failures.append(
                f"{split}: fitted-Q/position-size target lineage is invalid"
            )
        # The consistency proof is independent of the polarity features: a
        # missing signal column must never silently disable the target audit.
        if not bool(consistency.get("available")):
            failures.append(
                f"{split}: XAU future-outcome target consistency audit "
                "unavailable"
            )
        else:
            bad_path_mismatches = int(
                consistency.get("bad_path_side_mismatch_count") or 0
            )
            if bad_path_mismatches:
                failures.append(
                    f"{split}: scalar y_bad_path mismatches the side-specific "
                    f"bad-path targets on {bad_path_mismatches} rows"
                )
            for key, message in (
                (
                    "target_mode_contract_invalid_count",
                    "direction target mode contract is invalid/mixed",
                ),
                (
                    "direction_outcome_mismatch_count",
                    "y_direction mismatches future outcome side selection",
                ),
                (
                    "direction_long_score_alias_mismatch_count",
                    "y_direction_long_score_bps mismatches outcome long utility",
                ),
                (
                    "direction_short_score_alias_mismatch_count",
                    "y_direction_short_score_bps mismatches outcome short utility",
                ),
                (
                    "long_score_executable_pnl_mismatch_count",
                    "long score is not executable PnL under the fitted policy",
                ),
                (
                    "short_score_executable_pnl_mismatch_count",
                    "short score is not executable PnL under the fitted policy",
                ),
            ):
                count = int(consistency.get(key) or 0)
                if count:
                    failures.append(f"{split}: {message} on {count} rows")
        polarity = row.get("polarity") if isinstance(row.get("polarity"), dict) else {}
        if not bool(polarity.get("available")):
            continue
        if not bool(polarity.get("enough_pocket_rows")):
            failures.append(
                f"{split}: insufficient support/resistance pocket rows for polarity audit "
                f"(support={polarity.get('support_dominant_rows')} resistance={polarity.get('resistance_dominant_rows')})"
            )
            continue

    sizing_policy_hashes = {
        str(row.get("split")): str(
            row.get("entry_position_size_target_policy_sha256") or ""
        )
        for row in split_reports
    }
    if len(set(sizing_policy_hashes.values())) > 1:
        failures.append(
            "TRAIN/VAL position-size targets do not share one immutable "
            f"TRAIN-only policy: {sizing_policy_hashes}"
        )

    split_state_contracts: dict[str, dict[str, Any]] = {}
    for row in split_reports:
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        contract = provenance.get("model_native_state_contract")
        if isinstance(contract, dict) and contract:
            split_state_contracts[str(row.get("split"))] = contract
    if len(split_state_contracts) > 1:
        baseline_split = next(iter(split_state_contracts))
        baseline = split_state_contracts[baseline_split]
        for split, contract in split_state_contracts.items():
            if contract != baseline:
                failures.append(
                    f"{split}: model_native_state_contract differs from {baseline_split}; "
                    "TRAIN/VAL must share one immutable common-history/TRAIN-rank contract"
                )

    if len(tape_provenance_by_split) > 1:
        baseline_split = next(iter(tape_provenance_by_split))
        baseline = tape_provenance_by_split[baseline_split]
        for split, proof in tape_provenance_by_split.items():
            if proof != baseline:
                failures.append(
                    f"{split}: immutable XAU_USD tape provenance differs from "
                    f"{baseline_split}; TRAIN/VAL must share one exact tape lineage"
                )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report = {
        "schema_version": PRETRAIN_AUDIT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS" if not failures else "FAIL",
        "dataset_dir": str(dataset_dir),
        "requested_stem": requested_stem,
        "stem": str(stem or requested_stem),
        "data_splits": splits,
        "require_mandatory_level_features": True,
        "require_inline_seq_structure": True,
        "require_xau_provenance": True,
        "required_mandatory_level_features": list(
            REQUIRED_MANDATORY_LEVEL_FEATURES
        ),
        "missing_mandatory_level_features": (
            missing_mandatory_level_features
        ),
        "required_xau_target_columns": list(REQUIRED_XAU_TARGET_COLUMNS),
        "target_liveness_policy": {
            "required_live_columns": [
                name
                for name in REQUIRED_XAU_TARGET_COLUMNS
                if name not in TARGET_LIVENESS_EXEMPT_COLUMNS
            ],
            "exempt_columns": dict(TARGET_LIVENESS_EXEMPTION_REASONS),
            "exempt_columns_still_require_full_finiteness": True,
            "selected_side_bad_path_exact_copy_required": True,
            "side_specific_bad_path_liveness_required": [
                "y_long_bad_path",
                "y_short_bad_path",
            ],
        },
        "tape_provenance": tape_provenance_by_split,
        "thresholds": {
            "distance_dominance_margin_atr": float(
                args.distance_dominance_margin_atr
            ),
            "min_pocket_rows": int(args.min_pocket_rows),
        },
        "splits": split_reports,
        "failures": failures,
    }
    json_path = out_dir / f"XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_{timestamp}.json"
    report["json_path"] = str(json_path)
    with json_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n")

    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "failures": failures,
                    "json_path": str(json_path),
                },
                indent=2,
                default=_json_default,
            )
        )
    if failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Explicit exact model-native XAU dataset directory; no stale default is allowed.",
    )
    parser.add_argument("--stem", required=True, help="Explicit immutable dataset split stem; discovery is forbidden.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-rows-per-split", type=int, default=25000)
    parser.add_argument("--max-row-groups-per-split", type=int, default=5)
    parser.add_argument(
        "--distance-dominance-margin-atr",
        type=float,
        default=0.25,
    )
    parser.add_argument("--min-pocket-rows", type=int, default=30)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
