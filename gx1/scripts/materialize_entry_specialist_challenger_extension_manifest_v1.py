#!/usr/bin/env python3
"""Materialize the combined Entry specialist challenger sequence manifest.

Report-only. This joins the active foundation sequence extension with audited
chart-geometry and candlestick challenger features so the next dataset rebuild
can feed the specialist Transformer ensemble with the intended inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gx1.features.entry_chart_geometry_v1 import (
    CHART_GEOMETRY_FEATURE_NAMES,
    CHART_GEOMETRY_FEATURE_VERSION,
)
from gx1.features.entry_candlestick_patterns_v1 import (
    CANDLESTICK_PATTERN_FEATURE_NAMES,
    CANDLESTICK_PATTERN_FEATURE_VERSION,
)
from gx1.features.entry_foundation_structure_v1 import FOUNDATION_STRUCTURE_FEATURE_VERSION
from gx1.features.entry_momentum_flow_v1 import MOMENTUM_FLOW_FEATURE_NAMES, MOMENTUM_FLOW_FEATURE_VERSION
from gx1.features.entry_mtf_confluence_v1 import MTF_CONFLUENCE_FEATURE_NAMES, MTF_CONFLUENCE_FEATURE_VERSION
from gx1.features.entry_session_regime_interactions_v1 import (
    SESSION_REGIME_INTERACTION_FEATURE_NAMES,
    SESSION_REGIME_INTERACTION_FEATURE_VERSION,
)
from gx1.features.entry_smc_liquidity_quality_v1 import (
    SMC_LIQUIDITY_QUALITY_FEATURE_NAMES,
    SMC_LIQUIDITY_QUALITY_FEATURE_VERSION,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    classify_entry_specialist_feature,
    group_features_by_specialist,
    required_training_specialists_for_mode,
    specialist_model_contract_for_mode,
)
from gx1.features.entry_structure_swing_derivations_v1 import (
    STRUCTURE_SWING_DERIVATION_FEATURE_NAMES,
    STRUCTURE_SWING_DERIVATION_FEATURE_VERSION,
)
from gx1.features.entry_support_resistance_memory_v1 import (
    SUPPORT_RESISTANCE_MEMORY_FEATURE_NAMES,
    SUPPORT_RESISTANCE_MEMORY_FEATURE_VERSION,
)
from gx1.features.entry_trend_ema_v1 import TREND_EMA_FEATURE_NAMES, TREND_EMA_FEATURE_VERSION
from gx1.features.entry_vol_compression_v1 import VOL_COMPRESSION_FEATURE_NAMES, VOL_COMPRESSION_FEATURE_VERSION
from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT, REPO, SEQ_STRUCTURE_MANIFEST


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_specialist_challenger_extension_manifest_20260630_v1"
DEFAULT_CHART_GEOMETRY_MANIFEST = (
    REPORTS_ROOT
    / "entry_chart_geometry_challenger_audit_20260630_v1/ENTRY_CHART_GEOMETRY_CHALLENGER_MANIFEST_latest.json"
)
DEFAULT_CANDLESTICK_MANIFEST = (
    REPORTS_ROOT
    / "entry_candlestick_pattern_challenger_audit_20260630_v1/ENTRY_CANDLESTICK_PATTERN_CHALLENGER_MANIFEST_latest.json"
)
ACTIVE_SPECIALIST_CONTRACT_MODE = "foundation_seq146"
TARGET_CHALLENGER_CONTRACT_MODE = "challenger_seq215"
SMART_CANDIDATE_CONTRACT_MODE = "smart_seq520_candidate"
DEFAULT_BASE_SIGNAL_FEATURE_COUNT = 41
AUDITED_SEQ215_CHART_GEOMETRY_FEATURE_COUNT = 41
AUDITED_SEQ215_CANDLESTICK_FEATURE_COUNT = 28
SPECIALIST_CONTRACT_AUTHORITY = (
    "gx1.features.entry_specialist_feature_groups_v1:"
    "specialist_model_contract_for_mode()/required_training_specialists_for_mode()"
)
CHART_GEOMETRY_SMART_FEATURE_NAMES = CHART_GEOMETRY_FEATURE_NAMES[
    AUDITED_SEQ215_CHART_GEOMETRY_FEATURE_COUNT:
]
CANDLESTICK_SMART_FEATURE_NAMES = CANDLESTICK_PATTERN_FEATURE_NAMES[
    AUDITED_SEQ215_CANDLESTICK_FEATURE_COUNT:
]
SMART_LAYER_FEATURES: "OrderedDict[str, tuple[str, tuple[str, ...], str, Path]]" = OrderedDict(
    [
        (
            "trend_ema_smart_layer",
            (
                TREND_EMA_FEATURE_VERSION,
                TREND_EMA_FEATURE_NAMES,
                "gx1.features.entry_trend_ema_v1:build_entry_trend_ema_layer",
                REPO / "gx1/features/entry_trend_ema_v1.py",
            ),
        ),
        (
            "smc_liquidity_quality_layer",
            (
                SMC_LIQUIDITY_QUALITY_FEATURE_VERSION,
                SMC_LIQUIDITY_QUALITY_FEATURE_NAMES,
                "gx1.features.entry_smc_liquidity_quality_v1:build_entry_smc_liquidity_quality_layer",
                REPO / "gx1/features/entry_smc_liquidity_quality_v1.py",
            ),
        ),
        (
            "structure_swing_derivation_layer",
            (
                STRUCTURE_SWING_DERIVATION_FEATURE_VERSION,
                STRUCTURE_SWING_DERIVATION_FEATURE_NAMES,
                "gx1.features.entry_structure_swing_derivations_v1:build_entry_structure_swing_derivation_layer",
                REPO / "gx1/features/entry_structure_swing_derivations_v1.py",
            ),
        ),
        (
            "momentum_flow_smart_layer",
            (
                MOMENTUM_FLOW_FEATURE_VERSION,
                MOMENTUM_FLOW_FEATURE_NAMES,
                "gx1.features.entry_momentum_flow_v1:build_entry_momentum_flow_layer",
                REPO / "gx1/features/entry_momentum_flow_v1.py",
            ),
        ),
        (
            "session_regime_interaction_layer",
            (
                SESSION_REGIME_INTERACTION_FEATURE_VERSION,
                SESSION_REGIME_INTERACTION_FEATURE_NAMES,
                "gx1.features.entry_session_regime_interactions_v1:build_entry_session_regime_interaction_layer",
                REPO / "gx1/features/entry_session_regime_interactions_v1.py",
            ),
        ),
        (
            "vol_compression_smart_layer",
            (
                VOL_COMPRESSION_FEATURE_VERSION,
                VOL_COMPRESSION_FEATURE_NAMES,
                "gx1.features.entry_vol_compression_v1:build_entry_vol_compression_layer",
                REPO / "gx1/features/entry_vol_compression_v1.py",
            ),
        ),
        (
            "chart_geometry_smart2_layer",
            (
                CHART_GEOMETRY_FEATURE_VERSION,
                CHART_GEOMETRY_SMART_FEATURE_NAMES,
                "gx1.features.entry_chart_geometry_v1:build_entry_chart_geometry_layer",
                REPO / "gx1/features/entry_chart_geometry_v1.py",
            ),
        ),
        (
            "price_action_candle_smart3_layer",
            (
                CANDLESTICK_PATTERN_FEATURE_VERSION,
                CANDLESTICK_SMART_FEATURE_NAMES,
                "gx1.features.entry_candlestick_patterns_v1:build_entry_candlestick_pattern_layer",
                REPO / "gx1/features/entry_candlestick_patterns_v1.py",
            ),
        ),
        (
            "support_resistance_memory_layer",
            (
                SUPPORT_RESISTANCE_MEMORY_FEATURE_VERSION,
                SUPPORT_RESISTANCE_MEMORY_FEATURE_NAMES,
                "gx1.features.entry_support_resistance_memory_v1:build_entry_support_resistance_memory_layer",
                REPO / "gx1/features/entry_support_resistance_memory_v1.py",
            ),
        ),
        (
            "mtf_confluence_layer",
            (
                MTF_CONFLUENCE_FEATURE_VERSION,
                MTF_CONFLUENCE_FEATURE_NAMES,
                "gx1.features.entry_mtf_confluence_v1:build_entry_mtf_confluence_layer",
                REPO / "gx1/features/entry_mtf_confluence_v1.py",
            ),
        ),
    ]
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    return str(obj)


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_features(data: dict[str, Any], *, label: str) -> list[str]:
    features = [str(x) for x in data.get("selected_features", []) if str(x).strip()]
    if not features:
        raise RuntimeError(f"{label}: selected_features is empty")
    return features


def _dedupe_preserve_order(items: list[str]) -> tuple[list[str], list[str]]:
    seen: set[str] = set()
    out: list[str] = []
    duplicates: list[str] = []
    for item in items:
        if item in seen:
            duplicates.append(item)
            continue
        seen.add(item)
        out.append(item)
    return out, duplicates


def _source_meta(path: Path, data: dict[str, Any], *, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema_version": data.get("schema_version"),
        "decision": data.get("decision"),
        "manifest_only": bool(data.get("manifest_only")) if "manifest_only" in data else None,
        "dataset_dir": data.get("dataset_dir"),
        "source_parquet": data.get("source_parquet"),
        "selected_feature_count": len(data.get("selected_features", []) or []),
        "dataset_rebuild_required_before_training": data.get("dataset_rebuild_required_before_training"),
        "trainable_in_current_contract": data.get("trainable_in_current_contract"),
    }


def _smart_source_meta(
    *,
    label: str,
    version: str,
    features: tuple[str, ...],
    builder: str,
    source_path: Path,
) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(source_path),
        "sha256": _sha256_file(source_path),
        "schema_version": "entry_dormant_smart_feature_layer_v1",
        "decision": "READY_FOR_SMART_CHALLENGER_MANIFEST_REVIEW",
        "manifest_only": True,
        "feature_version": version,
        "builder": builder,
        "selected_feature_count": int(len(features)),
        "dataset_rebuild_required_before_training": True,
        "trainable_in_current_contract": False,
    }


def _feature_rows(features: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(features):
        rows.append(
            {
                "index": int(index),
                "name": name,
                "specialist": classify_entry_specialist_feature(name),
            }
        )
    return rows


def _counter(features: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(classify_entry_specialist_feature(name) for name in features).items()))


def _mode_specialist_contract(mode: str, *, role: str) -> dict[str, Any]:
    required = tuple(required_training_specialists_for_mode(mode))
    model_contract = specialist_model_contract_for_mode(mode)
    missing_contract_entries = [name for name in required if name not in model_contract]
    extra_contract_entries = [name for name in model_contract if name not in set(required)]
    contract_registered = not missing_contract_entries and not extra_contract_entries
    return {
        "contract_mode": mode,
        "role": role,
        "authority": SPECIALIST_CONTRACT_AUTHORITY,
        "required_training_specialists": list(required),
        "required_training_specialist_count": int(len(required)),
        "specialist_model_contract": model_contract,
        "specialist_model_contract_specialists": list(model_contract.keys()),
        "specialist_model_contract_specialist_count": int(len(model_contract)),
        "specialist_model_contract_set_exact": bool(contract_registered),
        "missing_specialist_model_contract_entries": missing_contract_entries,
        "extra_specialist_model_contract_entries": extra_contract_entries,
        "contract_registered": bool(contract_registered),
        "contract_update_required_before_training": bool(not contract_registered),
    }


def _specialist_contract_provenance(*, target_contract_mode: str = TARGET_CHALLENGER_CONTRACT_MODE) -> dict[str, Any]:
    active = _mode_specialist_contract(ACTIVE_SPECIALIST_CONTRACT_MODE, role="active_foundation")
    target = _mode_specialist_contract(target_contract_mode, role="target_challenger")
    active_required = set(active["required_training_specialists"])
    target_required = set(target["required_training_specialists"])
    target["additional_training_specialists_vs_active_foundation"] = [
        name for name in target["required_training_specialists"] if name not in active_required
    ]
    target["inherits_active_foundation_specialists"] = all(name in target_required for name in active_required)
    target["registered_contract_note"] = (
        f"{target_contract_mode} specialist contract is registered for audit/loader proof; "
        "execution remains closed until later gates and explicit vedtak"
    )
    return {
        "authority": SPECIALIST_CONTRACT_AUTHORITY,
        "active_foundation": active,
        "target_challenger": target,
        "active_vs_target": {
            "active_contract_mode": ACTIVE_SPECIALIST_CONTRACT_MODE,
            "active_required_training_specialist_count": active["required_training_specialist_count"],
            "target_contract_mode": target_contract_mode,
            "target_required_training_specialist_count": target["required_training_specialist_count"],
            "target_additional_training_specialists": target["additional_training_specialists_vs_active_foundation"],
        },
        "contract_update_required_before_training": bool(target["contract_update_required_before_training"]),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    manifest = report["manifest"]
    lines = [
        "# Entry Specialist Challenger Extension Manifest",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Selected features: `{manifest['selected_feature_count']}`",
        f"- Active foundation extension: `{report['counts']['foundation_sequence_extension_features']}`",
        f"- Chart geometry challenger: `{report['counts']['chart_geometry_challenger_features']}`",
        f"- Candlestick challenger: `{report['counts']['candlestick_challenger_features']}`",
        f"- Smart candidate layers: `{report['counts']['smart_candidate_features']}`",
        f"- Expected seq/snap width after rebuild: `{manifest['expected_seq_snap_width']}`",
        f"- Duplicate dropped: `{report['counts']['duplicate_feature_count']}`",
        f"- Failure count: `{len(report['failures'])}`",
        f"- Active contract: `{manifest['specialist_contract_provenance']['active_foundation']['contract_mode']}` "
        f"({manifest['specialist_contract_provenance']['active_foundation']['required_training_specialist_count']} specialists)",
        f"- Target challenger contract: `{manifest['specialist_contract_provenance']['target_challenger']['contract_mode']}` "
        f"({manifest['specialist_contract_provenance']['target_challenger']['required_training_specialist_count']} specialists)",
        f"- Target contract update required: `{manifest['contract_update_required_before_training']}`",
        "",
        "## Specialist Counts",
        "",
    ]
    for name, count in manifest["feature_counts_by_specialist"].items():
        lines.append(f"- `{name}`: `{count}`")
    lines.extend(["", "## Rebuild Command Shape", ""])
    lines.append("```bash")
    lines.append(" ".join(manifest["builder_usage"]["argv_template"]))
    lines.append("```")
    lines.extend(["", "## Failures", ""])
    lines.extend([f"- {failure}" for failure in report["failures"]] or ["- None"])
    lines.extend(["", "## Selected Features", ""])
    lines.extend(f"- `{name}`" for name in manifest["selected_features"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    foundation_path = Path(args.foundation_seq_manifest).expanduser().resolve()
    chart_path = Path(args.chart_geometry_manifest).expanduser().resolve()
    candle_path = Path(args.candlestick_manifest).expanduser().resolve()
    include_smart_layers = bool(getattr(args, "include_smart_layers", False))
    base_signal_feature_count = int(
        getattr(args, "base_signal_feature_count", DEFAULT_BASE_SIGNAL_FEATURE_COUNT)
    )

    foundation = _read_json(foundation_path)
    chart = _read_json(chart_path)
    candle = _read_json(candle_path)
    foundation_features = _selected_features(foundation, label="foundation sequence manifest")
    chart_features = _selected_features(chart, label="chart geometry manifest")
    candle_features = _selected_features(candle, label="candlestick manifest")
    smart_layer_features: "OrderedDict[str, list[str]]" = OrderedDict()
    if include_smart_layers:
        for label, (_, names, _, _) in SMART_LAYER_FEATURES.items():
            smart_layer_features[label] = list(names)
    smart_features = [name for names in smart_layer_features.values() for name in names]

    failures: list[str] = []
    if chart.get("decision") != "READY_FOR_CHALLENGER_DATASET_REBUILD":
        failures.append(f"chart geometry manifest not rebuild-ready: {chart.get('decision')}")
    if candle.get("decision") != "READY_FOR_CHALLENGER_DATASET_REBUILD":
        failures.append(f"candlestick manifest not rebuild-ready: {candle.get('decision')}")
    if foundation.get("foundation_structure_all_required_selected") is not True:
        failures.append("foundation sequence manifest does not prove all required foundation structure features selected")

    combined, duplicates = _dedupe_preserve_order(
        foundation_features + chart_features + candle_features + smart_features
    )
    grouped = group_features_by_specialist(combined)
    unmapped = grouped.get("unmapped", [])
    if unmapped:
        failures.append(f"unmapped combined features: {unmapped[:30]} total={len(unmapped)}")

    target_contract_mode = SMART_CANDIDATE_CONTRACT_MODE if include_smart_layers else TARGET_CHALLENGER_CONTRACT_MODE
    contract_provenance = _specialist_contract_provenance(target_contract_mode=target_contract_mode)
    active_contract = contract_provenance["active_foundation"]
    target_contract = contract_provenance["target_challenger"]
    trainable_challengers = [
        specialist
        for specialist in ("chart_geometry_encoder", "price_action_candle_encoder")
        if specialist in set(active_contract["required_training_specialists"])
    ]
    if trainable_challengers:
        failures.append(f"challenger specialists already trainable before contract update: {trainable_challengers}")
    target_required = set(target_contract["required_training_specialists"])
    missing_target_challengers = [
        specialist
        for specialist in ("chart_geometry_encoder", "price_action_candle_encoder")
        if specialist not in target_required
    ]
    if missing_target_challengers:
        failures.append(f"target challenger contract missing seq215 specialists: {missing_target_challengers}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    expected_seq_snap_width = base_signal_feature_count + int(len(combined))
    manifest_variant = f"smart_seq{expected_seq_snap_width}_candidate" if include_smart_layers else "seq215_challenger"
    decision = (
        "READY_FOR_SMART_CHALLENGER_DATASET_REBUILD_MANIFEST"
        if include_smart_layers and not failures
        else "READY_FOR_CHALLENGER_DATASET_REBUILD_MANIFEST"
        if not failures
        else "FAIL"
    )
    source_manifests = {
        "foundation_sequence_extension": _source_meta(foundation_path, foundation, label="foundation_sequence_extension"),
        "chart_geometry_challenger": _source_meta(chart_path, chart, label="chart_geometry_challenger"),
        "candlestick_challenger": _source_meta(candle_path, candle, label="candlestick_challenger"),
    }
    if smart_layer_features:
        source_manifests["smart_candidate_layers"] = OrderedDict(
            (
                label,
                _smart_source_meta(
                    label=label,
                    version=version,
                    features=features,
                    builder=builder,
                    source_path=source_path,
                ),
            )
            for label, (version, features, builder, source_path) in SMART_LAYER_FEATURES.items()
        )
    manifest = {
        "schema_version": "entry_specialist_challenger_extension_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_variant": manifest_variant,
        "decision": decision,
        "purpose": (
            "manifest-only selected feature order for rebuilding Entry sequence arrays with "
            "foundation structure, numeric chart geometry, closed-bar candlestick pattern inputs"
            + (", and dormant specialist smart-layer candidates" if include_smart_layers else "")
        ),
        "manifest_only": True,
        "selected_features": combined,
        "selected_feature_count": int(len(combined)),
        "base_signal_feature_count": base_signal_feature_count,
        "expected_seq_snap_width": expected_seq_snap_width,
        "feature_counts_by_specialist": _counter(combined),
        "features_by_specialist": grouped,
        "feature_rows": _feature_rows(combined),
        "source_manifests": source_manifests,
        "source_feature_counts": {
            "foundation_sequence_extension": int(len(foundation_features)),
            "chart_geometry_challenger": int(len(chart_features)),
            "candlestick_challenger": int(len(candle_features)),
            "smart_candidate_layers": int(len(smart_features)),
        },
        "smart_layers_included": include_smart_layers,
        "smart_layer_feature_counts": {
            label: int(len(features)) for label, features in smart_layer_features.items()
        },
        "smart_layer_feature_versions": {
            label: version for label, (version, _, _, _) in SMART_LAYER_FEATURES.items()
        }
        if smart_layer_features
        else {},
        "duplicate_features_dropped": duplicates,
        "foundation_structure_feature_version": foundation.get(
            "foundation_structure_feature_version", FOUNDATION_STRUCTURE_FEATURE_VERSION
        ),
        "chart_geometry_feature_version": chart.get(
            "chart_geometry_feature_version", CHART_GEOMETRY_FEATURE_VERSION
        ),
        "candlestick_pattern_feature_version": candle.get(
            "candlestick_pattern_feature_version", CANDLESTICK_PATTERN_FEATURE_VERSION
        ),
        "foundation_structure_feature_count": foundation.get("foundation_structure_feature_count"),
        "foundation_structure_missing_feature_count": foundation.get("foundation_structure_missing_feature_count"),
        "foundation_structure_all_required_selected": foundation.get("foundation_structure_all_required_selected"),
        "dataset_rebuild_required_before_training": True,
        "contract_update_required_before_training": bool(
            target_contract["contract_update_required_before_training"] or missing_target_challengers
        ),
        "training_allowed": False,
        "shadow_live_promotion_allowed": False,
        "specialist_contract_provenance": contract_provenance,
        "active_foundation_contract_mode": ACTIVE_SPECIALIST_CONTRACT_MODE,
        "active_foundation_required_training_specialists": active_contract["required_training_specialists"],
        "active_foundation_specialist_model_contract": active_contract["specialist_model_contract"],
        "target_challenger_contract_mode": target_contract_mode,
        "target_challenger_required_training_specialists": target_contract["required_training_specialists"],
        "target_challenger_specialist_model_contract": target_contract["specialist_model_contract"],
        "target_challenger_contract_update_required_before_training": bool(
            target_contract["contract_update_required_before_training"] or missing_target_challengers
        ),
        "current_specialist_contract_mode": ACTIVE_SPECIALIST_CONTRACT_MODE,
        "current_required_training_specialists": active_contract["required_training_specialists"],
        "current_specialist_model_contract": active_contract["specialist_model_contract"],
        "required_next_specialist_contract_review": {
            "status": (
                "REGISTERED_SMART_SEQ520_CANDIDATE_CONTRACT"
                if include_smart_layers
                else "REGISTERED_CHALLENGER_SEQ215_CONTRACT"
            ),
            "authority": SPECIALIST_CONTRACT_AUTHORITY,
            "contract_update_required_before_training": bool(
                target_contract["contract_update_required_before_training"] or missing_target_challengers
            ),
            "must_decide_exact_trainable_specialists": target_contract["required_training_specialists"],
            "must_update_bundle_audit_contract": False,
            "must_prove_liveness_noncollapse_edge_by_slice": True,
        },
        "builder_usage": {
            "argv_template": [
                ".venv/bin/python",
                "-m",
                "gx1.scripts.build_entry_v10_ctx_training_dataset_v3",
                "--seq-structure-manifest",
                "<this_manifest_json>",
                "--seq-structure-compute-inline",
                "--time_split",
                "--neutral-xgb-bridge",
                "--output",
                "<new_challenger_dataset_dir>/<stem>.parquet",
            ],
            "ram_note": (
                "The combined extension adds "
                f"{len(chart_features) + len(candle_features) + len(smart_features)} challenger/smart features "
                f"on top of the active {len(foundation_features)}-feature foundation extension. "
                "Rebuild under gx1_capped_run with conservative memory and streaming batch settings."
            ),
        },
    }
    report = {
        "schema_version": "entry_specialist_challenger_extension_manifest_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "manifest": manifest,
        "counts": {
            "foundation_sequence_extension_features": int(len(foundation_features)),
            "chart_geometry_challenger_features": int(len(chart_features)),
            "candlestick_challenger_features": int(len(candle_features)),
            "smart_candidate_features": int(len(smart_features)),
            "combined_selected_features": int(len(combined)),
            "base_signal_features": base_signal_feature_count,
            "expected_seq_snap_width": expected_seq_snap_width,
            "duplicate_feature_count": int(len(duplicates)),
        },
        "failures": failures,
        "side_effects_started": {
            "training": False,
            "replay": False,
            "iql_distillation": False,
            "shadow": False,
            "live": False,
            "promotion": False,
        },
        "next_required_gate": "rebuild challenger dataset, then feature audit -> specialist audit -> train-readiness",
        "training_allowed": False,
        "shadow_live_promotion_allowed": False,
    }

    stem = (
        "ENTRY_SPECIALIST_CHALLENGER_SMART_EXTENSION"
        if include_smart_layers
        else "ENTRY_SPECIALIST_CHALLENGER_EXTENSION"
    )
    manifest_json = out_dir / f"{stem}_MANIFEST_{timestamp}.json"
    report_json = out_dir / f"{stem}_REPORT_{timestamp}.json"
    report_md = out_dir / f"{stem}_REPORT_{timestamp}.md"
    manifest["manifest_json_path"] = str(manifest_json)
    report["json_path"] = str(report_json)
    report["md_path"] = str(report_md)

    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(report_md, report)
    (out_dir / f"{stem}_MANIFEST_latest.json").write_text(
        manifest_json.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out_dir / f"{stem}_REPORT_latest.json").write_text(
        report_json.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out_dir / f"{stem}_REPORT_latest.md").write_text(
        report_md.read_text(encoding="utf-8"), encoding="utf-8"
    )
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": decision,
                    "counts": report["counts"],
                    "manifest_json_path": str(manifest_json),
                    "report_json_path": str(report_json),
                    "failures": failures,
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
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--foundation-seq-manifest", default=str(SEQ_STRUCTURE_MANIFEST))
    ap.add_argument("--chart-geometry-manifest", default=str(DEFAULT_CHART_GEOMETRY_MANIFEST))
    ap.add_argument("--candlestick-manifest", default=str(DEFAULT_CANDLESTICK_MANIFEST))
    ap.add_argument("--include-smart-layers", action="store_true")
    ap.add_argument("--base-signal-feature-count", type=int, default=DEFAULT_BASE_SIGNAL_FEATURE_COUNT)
    ap.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
