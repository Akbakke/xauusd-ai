#!/usr/bin/env python3
"""Audit Entry seq146 specialist feature grouping before specialist training."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.audit.entry_transformer_feature_audit import _stack_list_column
from gx1.features.entry_specialist_feature_groups_v1 import (
    FOUNDATION_OBJECTIVE_SPECIALISTS,
    FOUNDATION_REQUIREMENT_PATTERNS,
    SMART_SEQ520_EXPECTED_SELECTED_FEATURE_COUNT,
    SMART_SEQ520_EXPECTED_SIGNAL_DIM,
    SMART_SEQ520_EXPECTED_SMART_FEATURE_COUNT,
    SPECIALIST_AUDIT_CONTRACT_MODES,
    SPECIALIST_CONTRACT_MODES,
    SPECIALIST_FUSION_ACTIVE_HEADS,
    SPECIALIST_FUSION_BLOCKED_HEADS,
    SPECIALIST_GROUPS,
    classify_entry_specialist_feature,
    group_features_by_specialist,
    required_training_specialists_for_mode,
    smart_family_contract_for_mode,
    specialist_contract_training_allowed_for_mode,
    specialist_model_contract_for_mode,
)
from gx1.scripts.audit_entry_foundation_features_v1 import REQUIRED_FOUNDATION_OBJECTIVE_FEATURES
from gx1.scripts.verify_entry_foundation_state_v1 import FOUNDATION_DATASET_DIR, REPORTS_ROOT, SEQ_STRUCTURE_MANIFEST


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_specialist_feature_group_audit_20260628_v1"

MIN_SIGNAL_COUNTS = {
    "structure_swing_encoder": 10,
    "smc_liquidity_encoder": 8,
    "trend_ema_encoder": 6,
    "vol_compression_encoder": 6,
    "momentum_flow_encoder": 3,
    "session_regime_encoder": 6,
    "chart_geometry_encoder": 8,
    "price_action_candle_encoder": 6,
}

MIN_LIVE_FEATURE_COUNTS = dict(MIN_SIGNAL_COUNTS)
MIN_SPECIALIST_MEAN_ACTIVE_RATE = 0.01
MIN_FEATURE_ACTIVE_RATE = 0.01
LIVENESS_EPSILON = 1e-7
NEAR_CONSTANT_STD = 1e-9


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


def _parse_csv(raw: str) -> list[str]:
    return [p.strip() for p in str(raw or "").split(",") if p.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_manifest_path(dataset_dir: Path, split: str) -> Path:
    matches = sorted(dataset_dir.glob(f"*_{split}.manifest.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {split} split manifest under {dataset_dir}, got {matches}")
    return matches[0]


def _split_parquet_path(dataset_dir: Path, split: str) -> Path:
    matches = sorted(dataset_dir.glob(f"*_{split}.parquet"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {split} split parquet under {dataset_dir}, got {matches}")
    return matches[0]


def _load_selected_features(path: Path) -> list[str]:
    manifest = _read_json(path)
    selected = [str(x) for x in manifest.get("selected_features", []) if str(x).strip()]
    if not selected:
        raise RuntimeError(f"sequence structure manifest has no selected_features: {path}")
    return selected


def _load_split_signal_fields(dataset_dir: Path, splits: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for split in splits:
        path = _split_manifest_path(dataset_dir, split)
        manifest = _read_json(path)
        signal_bridge = ((manifest.get("extra") or {}).get("signal_bridge") or {})
        fields = [str(x) for x in signal_bridge.get("fields", []) if str(x).strip()]
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
        }
    return out


def _load_split_context_fields(dataset_dir: Path, splits: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for split in splits:
        path = _split_manifest_path(dataset_dir, split)
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


def _context_routing_failures(rows: list[dict[str, Any]], *, contract_mode: str) -> list[str]:
    unmapped = [row for row in rows if str(row.get("specialist")) == "unmapped"]
    if unmapped:
        return [
            f"{contract_mode} context routing has unmapped fields: "
            + ", ".join(str(row.get("feature")) for row in unmapped[:40])
            + f" total={len(unmapped)}"
        ]
    return []


def _contract_training_surface(contract_mode: str) -> dict[str, Any]:
    return {
        "contract_mode": contract_mode,
        "registered_for_training_surfaces": contract_mode in SPECIALIST_CONTRACT_MODES,
        "training_allowed_by_contract_mode": specialist_contract_training_allowed_for_mode(contract_mode),
        "training_allowed_by_this_audit": False,
        "training_allowed": False,
        "requires_separate_readiness_gate": True,
    }


def _smart_family_contract_rows(
    seq_manifest: dict[str, Any],
    *,
    contract_mode: str,
) -> list[dict[str, Any]]:
    family_contract = smart_family_contract_for_mode(contract_mode)
    if not family_contract:
        return []
    manifest_counts = (
        seq_manifest.get("smart_layer_feature_counts")
        if isinstance(seq_manifest.get("smart_layer_feature_counts"), dict)
        else {}
    )
    rows: list[dict[str, Any]] = []
    for label, spec in family_contract.items():
        expected_specialist_counts = {
            str(k): int(v)
            for k, v in (spec.get("expected_specialist_counts") or {}).items()
        }
        expected_feature_count = int(spec.get("expected_feature_count") or sum(expected_specialist_counts.values()))
        observed = manifest_counts.get(label)
        rows.append(
            {
                "family": label,
                "purpose": spec.get("purpose"),
                "expected_feature_count": expected_feature_count,
                "observed_feature_count": int(observed) if observed is not None else None,
                "feature_count_matches": observed is not None and int(observed) == expected_feature_count,
                "expected_specialist_counts": expected_specialist_counts,
                "owned_specialists": list(spec.get("owned_specialists") or ()),
            }
        )
    return rows


def _smart_contract_failures(
    seq_manifest: dict[str, Any],
    *,
    contract_mode: str,
    signal_field_count: int,
    selected_feature_count: int,
    smart_family_rows: list[dict[str, Any]],
) -> list[str]:
    if contract_mode != "smart_seq520_candidate":
        return []

    failures: list[str] = []
    manifest_variant = str(seq_manifest.get("manifest_variant") or "")
    if manifest_variant != "smart_seq520_candidate":
        failures.append(f"smart_seq520 manifest_variant mismatch: observed={manifest_variant!r}")
    if signal_field_count != SMART_SEQ520_EXPECTED_SIGNAL_DIM:
        failures.append(
            f"smart_seq520 signal width mismatch: observed={signal_field_count} "
            f"expected={SMART_SEQ520_EXPECTED_SIGNAL_DIM}"
        )
    if selected_feature_count != SMART_SEQ520_EXPECTED_SELECTED_FEATURE_COUNT:
        failures.append(
            f"smart_seq520 selected feature count mismatch: observed={selected_feature_count} "
            f"expected={SMART_SEQ520_EXPECTED_SELECTED_FEATURE_COUNT}"
        )
    if seq_manifest.get("smart_layers_included") is not True:
        failures.append("smart_seq520 manifest does not declare smart_layers_included=true")
    if seq_manifest.get("dataset_rebuild_required_before_training") is not True:
        failures.append("smart_seq520 manifest must preserve dataset_rebuild_required_before_training=true")
    if seq_manifest.get("training_allowed") is not False:
        failures.append("smart_seq520 manifest must keep training_allowed=false")

    source_counts = (
        seq_manifest.get("source_feature_counts")
        if isinstance(seq_manifest.get("source_feature_counts"), dict)
        else {}
    )
    observed_smart_count = int(source_counts.get("smart_candidate_layers") or 0)
    if observed_smart_count != SMART_SEQ520_EXPECTED_SMART_FEATURE_COUNT:
        failures.append(
            "smart_seq520 smart layer source count mismatch: "
            f"observed={source_counts.get('smart_candidate_layers')} "
            f"expected={SMART_SEQ520_EXPECTED_SMART_FEATURE_COUNT}"
        )
    if len(smart_family_rows) != 10:
        failures.append(f"smart_seq520 smart family contract must have exactly 10 families: {len(smart_family_rows)}")
    for row in smart_family_rows:
        if row.get("feature_count_matches") is not True:
            failures.append(
                f"smart_seq520 family count mismatch: {row.get('family')} "
                f"observed={row.get('observed_feature_count')} expected={row.get('expected_feature_count')}"
            )
    return failures


def _specialist_input_liveness_rows(
    dataset_dir: Path,
    splits: list[str],
    signal_fields: list[str],
    required_specialists: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups_by_feature = [classify_entry_specialist_feature(feature) for feature in signal_fields]
    rows: list[dict[str, Any]] = []
    for split in splits:
        parquet_path = _split_parquet_path(dataset_dir, split)
        snap = _stack_list_column(pd.read_parquet(parquet_path, columns=["snap"])["snap"], np.float32)
        if snap.ndim != 2 or snap.shape[1] != len(signal_fields):
            raise RuntimeError(
                f"{split}: snap matrix shape {list(snap.shape)} does not match signal field count {len(signal_fields)}"
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
            live_mask = (
                finite_by_feature
                & (std > float(NEAR_CONSTANT_STD))
                & (active_rate >= float(MIN_FEATURE_ACTIVE_RATE))
            )
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
    return rows


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
        unsupported_heads = sorted(set(supports_heads) - active_heads)
        if unsupported_heads:
            failures.append(f"{specialist}: specialist model contract references inactive heads: {unsupported_heads}")
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
            "type": "gated_mixture_of_specialist_encoders",
            "gate_context": ["session_id", "vol_regime_id", "atr_bucket", "spread_bucket", "H4_trend_sign_cat"],
            "heads": list(SPECIALIST_FUSION_ACTIVE_HEADS),
            "active_heads": list(SPECIALIST_FUSION_ACTIVE_HEADS),
            "blocked_heads": list(SPECIALIST_FUSION_BLOCKED_HEADS),
            "distillation_path": "specialist_transformer_teacher -> offline replay scoring -> IQL entry policy student",
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
    splits = _parse_csv(args.data_splits)
    contract_mode = str(getattr(args, "contract_mode", "foundation_seq146") or "foundation_seq146").strip()
    required_training_specialists = required_training_specialists_for_mode(contract_mode)
    specialist_model_contract = specialist_model_contract_for_mode(contract_mode)

    failures: list[str] = []
    seq_manifest = _read_json(seq_manifest_path)
    selected_features = [str(x) for x in seq_manifest.get("selected_features", []) if str(x).strip()]
    if not selected_features:
        raise RuntimeError(f"sequence structure manifest has no selected_features: {seq_manifest_path}")
    split_contracts = _load_split_signal_fields(dataset_dir, splits)
    context_contracts = _load_split_context_fields(dataset_dir, splits)
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

    signal_unmapped_fields = [
        feature for feature in signal_fields if classify_entry_specialist_feature(feature) == "unmapped"
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

    specialist_input_liveness = _specialist_input_liveness_rows(
        dataset_dir,
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
    context_routing_rows = _context_routing_rows(context_contracts)
    context_routing_unmapped_fields = [
        str(row.get("feature"))
        for row in context_routing_rows
        if str(row.get("specialist")) == "unmapped"
    ]
    context_routing_failures = _context_routing_failures(context_routing_rows, contract_mode=contract_mode)
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
    contract_training_surface = _contract_training_surface(contract_mode)
    smart_family_contract = smart_family_contract_for_mode(contract_mode)
    smart_family_rows = _smart_family_contract_rows(seq_manifest, contract_mode=contract_mode)
    smart_contract_failures = _smart_contract_failures(
        seq_manifest,
        contract_mode=contract_mode,
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
        "report_only": True,
        "training_allowed": False,
        "training_allowed_with_explicit_vedtak": False,
        "training_allowed_reason": "specialist feature-group audit is report-only; training requires separate readiness gates",
        "dataset_dir": str(dataset_dir),
        "seq_structure_manifest": str(seq_manifest_path),
        "contract_mode": contract_mode,
        "contract_training_surface": contract_training_surface,
        "data_splits": splits,
        "signal_field_count": int(len(signal_fields)),
        "signal_unmapped_count": int(len(signal_unmapped_fields)),
        "signal_unmapped_fields": signal_unmapped_fields,
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
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    latest_json = out_dir / "ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json"
    latest_md = out_dir / "ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

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
    if args.fail_on_audit_fail and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default=str(FOUNDATION_DATASET_DIR))
    ap.add_argument("--seq-structure-manifest", default=str(SEQ_STRUCTURE_MANIFEST))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--data-splits", default="train,val,test")
    ap.add_argument("--contract-mode", choices=SPECIALIST_AUDIT_CONTRACT_MODES, default="foundation_seq146")
    ap.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
