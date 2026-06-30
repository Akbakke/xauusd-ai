#!/usr/bin/env python3
"""Materialize the active Entry feature/input inventory and AI specialist plan.

This is report-only. It reads manifests and parquet schemas, not full training
data, and ranks the available input families into concrete specialist AI
models that can later feed the Entry Transformer after the normal gates pass.
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
import pyarrow.parquet as pq

from gx1.features.entry_chart_geometry_v1 import (
    CHART_GEOMETRY_FEATURE_NAMES,
    CHART_GEOMETRY_FEATURE_VERSION,
    CHART_GEOMETRY_SOURCE_FIELDS,
    missing_chart_geometry_source_fields,
)
from gx1.features.entry_candlestick_patterns_v1 import (
    CANDLESTICK_PATTERN_FEATURE_NAMES,
    CANDLESTICK_PATTERN_FEATURE_VERSION,
    CANDLESTICK_PATTERN_SOURCE_FIELDS,
    missing_candlestick_source_fields,
)
from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_STRUCTURE_FEATURE_NAMES,
    FOUNDATION_STRUCTURE_FEATURE_VERSION,
    FOUNDATION_STRUCTURE_SOURCE_FIELDS,
)
from gx1.features.entry_momentum_flow_v1 import (
    MOMENTUM_FLOW_OPTIONAL_SOURCE_FIELDS,
    MOMENTUM_FLOW_REQUIRED_SOURCE_FIELDS,
    missing_entry_momentum_flow_source_fields,
)
from gx1.features.entry_mtf_confluence_v1 import (
    MTF_CONFLUENCE_OPTIONAL_SOURCE_FIELDS,
    MTF_CONFLUENCE_SOURCE_FIELDS,
    missing_mtf_confluence_source_fields,
)
from gx1.features.entry_session_regime_interactions_v1 import (
    SESSION_REGIME_INTERACTION_SOURCE_FIELDS,
    missing_session_regime_interaction_source_fields,
)
from gx1.features.entry_smc_liquidity_quality_v1 import (
    SMC_LIQUIDITY_QUALITY_OPTIONAL_FIELDS,
    SMC_LIQUIDITY_QUALITY_SOURCE_FIELDS,
    missing_smc_liquidity_quality_source_fields,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    classify_entry_specialist_feature,
    required_training_specialists_for_mode,
    specialist_model_contract_for_mode,
)
from gx1.features.entry_structure_swing_derivations_v1 import (
    STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS,
    missing_structure_swing_derivation_source_fields,
)
from gx1.features.entry_support_resistance_memory_v1 import (
    SUPPORT_RESISTANCE_MEMORY_OPTIONAL_FIELDS,
    SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS,
    missing_support_resistance_memory_source_fields,
)
from gx1.features.entry_trend_ema_v1 import TREND_EMA_SOURCE_FIELDS, missing_trend_ema_source_fields
from gx1.features.entry_vol_compression_v1 import (
    VOL_COMPRESSION_OPTIONAL_SOURCE_FIELDS,
    VOL_COMPRESSION_REQUIRED_SOURCE_FIELDS,
    missing_entry_vol_compression_source_fields,
)
from gx1.scripts.evaluate_entry_selective_edge_v1 import _split_files
from gx1.scripts.materialize_entry_specialist_challenger_extension_manifest_v1 import (
    DEFAULT_BASE_SIGNAL_FEATURE_COUNT,
    SMART_LAYER_FEATURES,
)
from gx1.scripts.verify_entry_foundation_state_v1 import FOUNDATION_DATASET_DIR, REPORTS_ROOT, SEQ_STRUCTURE_MANIFEST


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_feature_ai_inventory_20260630_v1"
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
SPECIALIST_CONTRACT_AUTHORITY = (
    "gx1.features.entry_specialist_feature_groups_v1:"
    "specialist_model_contract_for_mode()/required_training_specialists_for_mode()"
)
NEXT_REQUIRED_GATE = (
    "seq215 challenger inputs and the 8-specialist contract are materialized and audited; "
    "next boundary is explicit SEQ215 smoke-manifest/smoke-train evidence with edge audit, "
    "then candidate-readiness-seq215. smart_seq495 remains report-only until a separate capped "
    "dataset rebuild, feature audit, specialist audit, liveness/non-collapse proof and "
    "train-readiness gate pass. Do not start candidate training, replay, IQL, shadow or live "
    "from inventory alone."
)

SMART_LAYER_SOURCE_CONTRACTS: "OrderedDict[str, dict[str, Any]]" = OrderedDict(
    [
        (
            "trend_ema_smart_layer",
            {
                "required_source_fields": TREND_EMA_SOURCE_FIELDS,
                "optional_source_fields": (),
                "missing_required_source_fields": missing_trend_ema_source_fields,
            },
        ),
        (
            "smc_liquidity_quality_layer",
            {
                "required_source_fields": SMC_LIQUIDITY_QUALITY_SOURCE_FIELDS,
                "optional_source_fields": SMC_LIQUIDITY_QUALITY_OPTIONAL_FIELDS,
                "missing_required_source_fields": missing_smc_liquidity_quality_source_fields,
            },
        ),
        (
            "structure_swing_derivation_layer",
            {
                "required_source_fields": STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS,
                "optional_source_fields": (),
                "missing_required_source_fields": missing_structure_swing_derivation_source_fields,
            },
        ),
        (
            "momentum_flow_smart_layer",
            {
                "required_source_fields": MOMENTUM_FLOW_REQUIRED_SOURCE_FIELDS,
                "optional_source_fields": MOMENTUM_FLOW_OPTIONAL_SOURCE_FIELDS,
                "missing_required_source_fields": missing_entry_momentum_flow_source_fields,
            },
        ),
        (
            "session_regime_interaction_layer",
            {
                "required_source_fields": SESSION_REGIME_INTERACTION_SOURCE_FIELDS,
                "optional_source_fields": (),
                "missing_required_source_fields": missing_session_regime_interaction_source_fields,
            },
        ),
        (
            "vol_compression_smart_layer",
            {
                "required_source_fields": VOL_COMPRESSION_REQUIRED_SOURCE_FIELDS,
                "optional_source_fields": VOL_COMPRESSION_OPTIONAL_SOURCE_FIELDS,
                "missing_required_source_fields": missing_entry_vol_compression_source_fields,
            },
        ),
        (
            "chart_geometry_smart2_layer",
            {
                "required_source_fields": CHART_GEOMETRY_SOURCE_FIELDS,
                "optional_source_fields": (),
                "missing_required_source_fields": missing_chart_geometry_source_fields,
            },
        ),
        (
            "price_action_candle_smart3_layer",
            {
                "required_source_fields": CANDLESTICK_PATTERN_SOURCE_FIELDS,
                "optional_source_fields": (),
                "missing_required_source_fields": missing_candlestick_source_fields,
            },
        ),
        (
            "support_resistance_memory_layer",
            {
                "required_source_fields": SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS,
                "optional_source_fields": SUPPORT_RESISTANCE_MEMORY_OPTIONAL_FIELDS,
                "missing_required_source_fields": missing_support_resistance_memory_source_fields,
            },
        ),
        (
            "mtf_confluence_layer",
            {
                "required_source_fields": MTF_CONFLUENCE_SOURCE_FIELDS,
                "optional_source_fields": MTF_CONFLUENCE_OPTIONAL_SOURCE_FIELDS,
                "missing_required_source_fields": missing_mtf_confluence_source_fields,
            },
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
    return json.loads(path.read_text(encoding="utf-8"))


def _split_manifest_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(".manifest.json")


def _file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _schema_columns_if_available(path: str | Path | None) -> list[str]:
    if not path:
        return []
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return []
    return [str(name) for name in pq.read_schema(resolved).names]


def _load_split_contract(parquet_path: Path) -> dict[str, Any]:
    manifest_path = _split_manifest_path(parquet_path)
    manifest = _read_json(manifest_path)
    extra = manifest.get("extra") or {}
    signal_bridge = extra.get("signal_bridge") or {}
    ctx_contract = extra.get("ctx_contract") or {}
    fields = [str(x) for x in signal_bridge.get("fields", [])]
    ctx_cont = [str(x) for x in ctx_contract.get("ctx_cont_names", [])]
    ctx_cat = [str(x) for x in ctx_contract.get("ctx_cat_names", [])]
    if not fields:
        raise RuntimeError(f"missing signal_bridge.fields in {manifest_path}")
    if not ctx_cont:
        raise RuntimeError(f"missing ctx_cont_names in {manifest_path}")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "parquet": _file_metadata(parquet_path),
        "rows": int(pq.ParquetFile(parquet_path).metadata.num_rows),
        "signal_fields": fields,
        "ctx_cont_names": ctx_cont,
        "ctx_cat_names": ctx_cat,
        "signal_bridge": signal_bridge,
        "ctx_contract": ctx_contract,
        "schema_columns": list(pq.ParquetFile(parquet_path).schema_arrow.names),
    }


def _load_sequence_features(path: Path) -> tuple[list[str], dict[str, Any]]:
    if not path.exists():
        return [], {"path": str(path), "exists": False}
    data = _read_json(path)
    return [str(x) for x in data.get("selected_features", [])], {
        "path": str(path),
        "exists": True,
        "sha256": _sha256_file(path),
        "schema_version": data.get("schema_version"),
        "selected_feature_count": len(data.get("selected_features", []) or []),
        "foundation_structure_all_required_selected": data.get("foundation_structure_all_required_selected"),
    }


def _load_chart_geometry(path: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    if not path.exists():
        return list(CHART_GEOMETRY_FEATURE_NAMES), list(CHART_GEOMETRY_SOURCE_FIELDS), {
            "path": str(path),
            "exists": False,
            "fallback": "repo constants",
        }
    data = _read_json(path)
    return [str(x) for x in data.get("selected_features", [])], [str(x) for x in data.get("source_fields", [])], {
        "path": str(path),
        "exists": True,
        "sha256": _sha256_file(path),
        "schema_version": data.get("schema_version"),
        "decision": data.get("decision"),
        "trainable_in_current_contract": data.get("trainable_in_current_contract"),
        "dataset_rebuild_required_before_training": data.get("dataset_rebuild_required_before_training"),
    }


def _load_candlestick(path: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    if not path.exists():
        return list(CANDLESTICK_PATTERN_FEATURE_NAMES), list(CANDLESTICK_PATTERN_SOURCE_FIELDS), {
            "path": str(path),
            "exists": False,
            "fallback": "repo constants",
        }
    data = _read_json(path)
    return [str(x) for x in data.get("selected_features", [])], [str(x) for x in data.get("source_fields", [])], {
        "path": str(path),
        "exists": True,
        "sha256": _sha256_file(path),
        "schema_version": data.get("schema_version"),
        "decision": data.get("decision"),
        "source_parquet": data.get("source_parquet"),
        "source_parquet_schema_columns": _schema_columns_if_available(data.get("source_parquet")),
        "trainable_in_current_contract": data.get("trainable_in_current_contract"),
        "dataset_rebuild_required_before_training": data.get("dataset_rebuild_required_before_training"),
    }


def _group(names: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = OrderedDict()
    for name in names:
        group = classify_entry_specialist_feature(name)
        out.setdefault(group, []).append(name)
    return out


def _count_groups(names: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(classify_entry_specialist_feature(name) for name in names).items()))


def _field_variants(name: str) -> set[str]:
    raw = str(name)
    variants = {raw}
    for prefix in ("snap.", "ctx_cont.", "ctx_cat."):
        if raw.startswith(prefix):
            variants.add(raw[len(prefix) :])
            return variants
    variants.add(f"snap.{raw}")
    variants.add(f"ctx_cont.{raw}")
    variants.add(f"ctx_cat.{raw}")
    return variants


def _field_available(name: str, available: set[str]) -> bool:
    return bool(_field_variants(name) & available)


def _available_input_names(
    *,
    signal_fields: list[str],
    ctx_cont: list[str],
    ctx_cat: list[str],
    sequence_features: list[str],
    chart_geometry_features: list[str],
    candlestick_features: list[str],
    smart_features: list[str],
    schema_columns: list[str],
) -> set[str]:
    names: list[str] = []
    for name in signal_fields:
        names.extend([name, f"snap.{name}"])
    for name in ctx_cont:
        names.extend([name, f"ctx_cont.{name}"])
    for name in ctx_cat:
        names.extend([name, f"ctx_cat.{name}"])
    names.extend(sequence_features)
    names.extend(chart_geometry_features)
    names.extend(candlestick_features)
    names.extend(smart_features)
    names.extend(schema_columns)
    return set(dict.fromkeys(str(name) for name in names if str(name).strip()))


def _smart_layer_rows(available: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, (version, features, builder, source_path) in SMART_LAYER_FEATURES.items():
        contract = SMART_LAYER_SOURCE_CONTRACTS[label]
        required_sources = tuple(str(name) for name in contract["required_source_fields"])
        optional_sources = tuple(str(name) for name in contract["optional_source_fields"])
        missing_fn = contract["missing_required_source_fields"]
        missing_required = list(missing_fn(available))
        missing_required_set = set(missing_required)
        optional_missing = [name for name in optional_sources if not _field_available(name, available)]
        feature_counts = Counter(classify_entry_specialist_feature(name) for name in features)
        rows.append(
            {
                "label": label,
                "version": version,
                "builder": builder,
                "source_path": str(source_path),
                "feature_count": int(len(features)),
                "feature_counts_by_specialist": dict(sorted(feature_counts.items())),
                "features": list(features),
                "required_source_fields": list(required_sources),
                "required_source_field_count": int(len(required_sources)),
                "present_required_source_fields": [
                    name for name in required_sources if name not in missing_required_set
                ],
                "missing_required_source_fields": missing_required,
                "missing_required_source_field_count": int(len(missing_required)),
                "optional_source_fields": list(optional_sources),
                "present_optional_source_fields": [
                    name for name in optional_sources if name not in set(optional_missing)
                ],
                "missing_optional_source_fields": optional_missing,
                "missing_optional_source_field_count": int(len(optional_missing)),
                "dataset_rebuild_required_before_training": True,
                "training_allowed": False,
                "rebuild_input_status": (
                    "READY_FOR_CAPPED_REBUILD_SOURCE_REVIEW"
                    if not missing_required
                    else "BLOCKED_BY_MISSING_REQUIRED_SOURCE_FIELDS"
                ),
            }
        )
    return rows


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


def _specialist_contract_provenance() -> dict[str, Any]:
    active = _mode_specialist_contract(ACTIVE_SPECIALIST_CONTRACT_MODE, role="active_foundation")
    target = _mode_specialist_contract(TARGET_CHALLENGER_CONTRACT_MODE, role="target_challenger")
    active_required = set(active["required_training_specialists"])
    target_required = set(target["required_training_specialists"])
    target["additional_training_specialists_vs_active_foundation"] = [
        name for name in target["required_training_specialists"] if name not in active_required
    ]
    target["inherits_active_foundation_specialists"] = all(name in target_required for name in active_required)
    target["registered_contract_note"] = (
        "challenger_seq215 8-specialist contract is registered; no specialist model contract "
        "update is required before seq215 proof/smoke gates"
    )
    return {
        "authority": SPECIALIST_CONTRACT_AUTHORITY,
        "active_foundation": active,
        "target_challenger": target,
        "active_vs_target": {
            "active_contract_mode": ACTIVE_SPECIALIST_CONTRACT_MODE,
            "active_required_training_specialist_count": active["required_training_specialist_count"],
            "target_contract_mode": TARGET_CHALLENGER_CONTRACT_MODE,
            "target_required_training_specialist_count": target["required_training_specialist_count"],
            "target_additional_training_specialists": target["additional_training_specialists_vs_active_foundation"],
        },
        "contract_update_required_before_training": bool(target["contract_update_required_before_training"]),
    }


def _feature_rows(names: list[str], *, input_surface: str, source: str) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "input_surface": input_surface,
            "source": source,
            "specialist": classify_entry_specialist_feature(name),
        }
        for name in names
    ]


def _contains_any(name: str, tokens: tuple[str, ...]) -> bool:
    n = name.lower()
    return any(token in n for token in tokens)


def _pick(names: list[str], tokens: tuple[str, ...]) -> list[str]:
    return [name for name in names if _contains_any(name, tokens)]


def _specialist_plan(
    *,
    signal_fields: list[str],
    ctx_cont: list[str],
    ctx_cat: list[str],
    sequence_features: list[str],
    chart_geometry_features: list[str],
    candlestick_features: list[str],
) -> list[dict[str, Any]]:
    all_named = signal_fields + [f"ctx_cont.{name}" for name in ctx_cont] + [f"ctx_cat.{name}" for name in ctx_cat]
    candle_existing = _pick(
        all_named,
        ("body", "wick", "clv", "range_z", "body_share", "candle", "open", "high", "low", "close"),
    )
    return [
        {
            "rank": 1,
            "model_id": "structure_swing_sequence_ai",
            "status": "active_trainable_now",
            "purpose": "Direction backbone: HH/HL/LH/LL, BOS/CHoCH recency, impulse/pullback phase.",
            "why": "This is closest to manual market-structure reading and already has audited foundation labels/features.",
            "existing_feature_count": len([x for x in all_named + sequence_features if classify_entry_specialist_feature(x) == "structure_swing_encoder"]),
            "example_inputs": _pick(all_named + sequence_features, ("foundation_hh", "foundation_hl", "foundation_lh", "foundation_ll", "bos", "choch", "swing", "pullback"))[:30],
            "missing_dependencies": [],
        },
        {
            "rank": 2,
            "model_id": "chart_geometry_line_fib_pattern_sequence_ai",
            "status": "challenger_ready_for_dataset_rebuild",
            "purpose": "Numeric support/resistance, channels, trendlines, Fibonacci zones, EMA-cross pressure and triangle/flag proxies.",
            "why": "This directly matches the discretionary line-drawing idea and the challenger audit proved source availability.",
            "existing_feature_count": len(chart_geometry_features),
            "example_inputs": chart_geometry_features[:35],
            "missing_dependencies": ["rebuild active dataset/manifest so these generated features become trainable inputs"],
        },
        {
            "rank": 3,
            "model_id": "multi_timeframe_trend_ema_sequence_ai",
            "status": "active_trainable_now",
            "purpose": "M5/M15/H1/H4/D1 trend stack, EMA slope, trend age, regime agreement/divergence.",
            "why": "Directional edge needs timeframe alignment and conflict detection before entry.",
            "existing_feature_count": len(_pick(all_named + sequence_features, ("ema", "trend", "slope", "regime_tf", "regime_stack", "h1_", "h4_", "d1_", "m15_"))),
            "example_inputs": _pick(all_named + sequence_features, ("ema", "trend", "slope", "regime_tf", "regime_stack", "h1_", "h4_", "d1_", "m15_"))[:35],
            "missing_dependencies": [],
        },
        {
            "rank": 4,
            "model_id": "smc_liquidity_sequence_ai",
            "status": "active_trainable_now",
            "purpose": "Sweeps, reclaim/false breakout, premium/discount, level proximity and wick/liquidity context.",
            "why": "This decides whether a structure break is real tradable pressure or a trap.",
            "existing_feature_count": len([x for x in all_named + sequence_features if classify_entry_specialist_feature(x) == "smc_liquidity_encoder"]),
            "example_inputs": _pick(all_named + sequence_features, ("sweep", "liquidity", "premium", "discount", "support", "resistance", "sr_", "dist_to_r", "dist_to_s", "pivot", "wick_level"))[:35],
            "missing_dependencies": [],
        },
        {
            "rank": 5,
            "model_id": "momentum_flow_sequence_ai",
            "status": "active_trainable_now",
            "purpose": "Returns, impulse velocity, micro momentum, acceleration, signed volume/flow and follow-through.",
            "why": "Market direction is usually only tradable when structure has current flow behind it.",
            "existing_feature_count": len([x for x in all_named + sequence_features if classify_entry_specialist_feature(x) == "momentum_flow_encoder"]),
            "example_inputs": _pick(all_named + sequence_features, ("ret_", "mom", "momentum", "acceleration", "signed_vol", "flow", "clv", "rvol"))[:35],
            "missing_dependencies": [],
        },
        {
            "rank": 6,
            "model_id": "volatility_compression_breakout_sequence_ai",
            "status": "active_trainable_now",
            "purpose": "ATR/volatility percentile, squeeze, compression-release and expansion direction.",
            "why": "Breakouts without volatility release fail; volatility also controls stop/MAE risk.",
            "existing_feature_count": len([x for x in all_named + sequence_features if classify_entry_specialist_feature(x) == "vol_compression_encoder"]),
            "example_inputs": _pick(all_named + sequence_features, ("atr", "vol", "range", "squeeze", "compression", "bandwidth", "sigma"))[:35],
            "missing_dependencies": [],
        },
        {
            "rank": 7,
            "model_id": "candlestick_pattern_sequence_ai",
            "status": "challenger_ready_for_dataset_rebuild",
            "purpose": "Single/double/triple candle states: doji, pin/hammer/shooting star, engulfing, inside/outside bars, morning/evening star, three soldiers/crows.",
            "why": "This is the missing human-readable price-action specialist; current inputs had body/wick/range proxies and now have explicit closed-bar pattern scores.",
            "existing_feature_count": len(candle_existing) + len(candlestick_features),
            "example_inputs": (candlestick_features + candle_existing)[:35],
            "missing_dependencies": [
                "rebuild active dataset/manifest so these generated features become trainable inputs",
                "optional later M15/H1 aggregate candle pattern layer",
            ],
        },
        {
            "rank": 8,
            "model_id": "session_regime_context_gater_ai",
            "status": "active_trainable_now",
            "purpose": "Asia/EU/US/overlap, session boundary, volatility/spread buckets and regime conditioning.",
            "why": "This should gate specialists rather than dominate direction: the same pattern behaves differently by session/regime.",
            "existing_feature_count": len([x for x in all_named + sequence_features if classify_entry_specialist_feature(x) == "session_regime_encoder"]),
            "example_inputs": _pick(all_named + sequence_features, ("session", "asia", "eu", "us", "overlap", "hour", "dow", "regime", "bucket", "spread"))[:35],
            "missing_dependencies": [],
        },
        {
            "rank": 9,
            "model_id": "path_tail_risk_veto_ai",
            "status": "head_contract_active_but_input_layer_should_be_expanded",
            "purpose": "Predict bad-path, MAE/tail drawdown and low-quality continuation before entry.",
            "why": "This is not pure direction, but it prevents the ensemble from taking entries that look right but replay badly.",
            "existing_feature_count": len(_pick(all_named + sequence_features, ("mae", "mfe", "tail", "bad_path", "path_quality", "spread", "atr", "wick", "divergence"))),
            "example_inputs": _pick(all_named + sequence_features, ("mae", "mfe", "tail", "bad_path", "path_quality", "spread", "atr", "wick", "divergence"))[:35],
            "missing_dependencies": ["post-train replay/slice labels must remain tied to exact candidate trade logs"],
        },
    ]


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Feature AI Inventory",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Dataset: `{report['dataset_dir']}`",
        f"- Signal seq/snap fields: `{report['counts']['signal_fields']}`",
        f"- Context continuous fields: `{report['counts']['ctx_cont']}`",
        f"- Context categorical fields: `{report['counts']['ctx_cat']}`",
        f"- Sequence extension features: `{report['counts']['sequence_extension_features']}`",
        f"- Chart-geometry challenger features: `{report['counts']['chart_geometry_features']}`",
        f"- Candlestick-pattern challenger features: `{report['counts']['candlestick_pattern_features']}`",
        f"- Smart-layer candidate features: `{report['counts']['smart_layer_features']}`",
        f"- Smart candidate variant: `{report['smart_candidate']['manifest_variant']}`",
        f"- Smart candidate expected signal dim: `{report['smart_candidate']['expected_signal_dim']}`",
        f"- Smart source coverage all required available: "
        f"`{report['smart_candidate']['source_coverage_all_required_available']}`",
        f"- Label/target columns: `{report['counts']['label_or_target_columns']}`",
        f"- Active contract: `{report['specialist_contract_provenance']['active_foundation']['contract_mode']}` "
        f"({report['specialist_contract_provenance']['active_foundation']['required_training_specialist_count']} specialists)",
        f"- Target challenger contract: `{report['specialist_contract_provenance']['target_challenger']['contract_mode']}` "
        f"({report['specialist_contract_provenance']['target_challenger']['required_training_specialist_count']} specialists)",
        f"- Target contract update required: `{report['specialist_contract_provenance']['contract_update_required_before_training']}`",
        "",
        "## Ranked Specialist AI Models",
        "",
    ]
    for row in report["ranked_specialist_models"]:
        lines.append(
            f"{row['rank']}. `{row['model_id']}` - `{row['status']}` - "
            f"{row['purpose']} Existing inputs: `{row['existing_feature_count']}`."
        )
        if row["missing_dependencies"]:
            lines.append(f"   Missing: {', '.join(row['missing_dependencies'])}")
    lines.extend(["", "## Active Input Counts By Current Specialist", ""])
    for group, count in report["active_input_counts_by_specialist"].items():
        lines.append(f"- `{group}`: `{count}`")
    lines.extend(["", "## Smart Candidate Layers", ""])
    for row in report["smart_layer_source_coverage"]:
        lines.append(
            f"- `{row['label']}`: `{row['feature_count']}` features, "
            f"`{row['rebuild_input_status']}`, missing required sources "
            f"`{row['missing_required_source_field_count']}`"
        )
    lines.extend(["", "## Smart Candidate Input Counts By Specialist", ""])
    for group, count in report["smart_candidate_input_counts_by_specialist"].items():
        lines.append(f"- `{group}`: `{count}`")
    lines.extend(["", "## All Signal Seq/Snap Fields", ""])
    lines.extend(f"- `{name}`" for name in report["all_inputs"]["signal_fields"])
    lines.extend(["", "## All Context Continuous Fields", ""])
    lines.extend(f"- `{name}`" for name in report["all_inputs"]["ctx_cont"])
    lines.extend(["", "## All Context Categorical Fields", ""])
    lines.extend(f"- `{name}`" for name in report["all_inputs"]["ctx_cat"])
    lines.extend(["", "## Sequence Extension Features", ""])
    lines.extend(f"- `{name}`" for name in report["all_inputs"]["sequence_extension_features"])
    lines.extend(["", "## Chart Geometry Challenger Features", ""])
    lines.extend(f"- `{name}`" for name in report["all_inputs"]["chart_geometry_features"])
    lines.extend(["", "## Candlestick Pattern Challenger Features", ""])
    lines.extend(f"- `{name}`" for name in report["all_inputs"]["candlestick_pattern_features"])
    lines.extend(["", "## Smart Layer Features", ""])
    for label, names in report["all_inputs"]["smart_layer_features_by_label"].items():
        lines.append(f"### {label}")
        lines.extend(f"- `{name}`" for name in names)
    lines.extend(["", "## Label/Target/Metadata Columns", ""])
    lines.extend(f"- `{name}`" for name in report["all_inputs"]["label_or_target_columns"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = [part.strip() for part in str(args.data_splits).split(",") if part.strip()]
    files = _split_files(dataset_dir, splits)

    split_contracts = {split: _load_split_contract(Path(files[split])) for split in splits}
    ref = split_contracts[splits[0]]
    failures: list[str] = []
    for split, contract in split_contracts.items():
        for key in ("signal_fields", "ctx_cont_names", "ctx_cat_names"):
            if contract[key] != ref[key]:
                failures.append(f"{split}: {key} differs from {splits[0]}")

    sequence_features, sequence_meta = _load_sequence_features(Path(args.seq_structure_manifest).expanduser().resolve())
    chart_features, chart_sources, chart_meta = _load_chart_geometry(Path(args.chart_geometry_manifest).expanduser().resolve())
    candle_features, candle_sources, candle_meta = _load_candlestick(Path(args.candlestick_manifest).expanduser().resolve())
    schema_columns = list(ref["schema_columns"])
    external_source_schema_columns = list(candle_meta.get("source_parquet_schema_columns") or [])
    vector_columns = {"seq", "snap", "ctx_cont", "ctx_cat"}
    label_or_target_columns = [name for name in schema_columns if name not in vector_columns]
    smart_layer_features_by_label: "OrderedDict[str, list[str]]" = OrderedDict(
        (label, list(features)) for label, (_, features, _, _) in SMART_LAYER_FEATURES.items()
    )
    smart_features = [name for features in smart_layer_features_by_label.values() for name in features]

    signal_fields = list(ref["signal_fields"])
    ctx_cont = list(ref["ctx_cont_names"])
    ctx_cat = list(ref["ctx_cat_names"])
    available_inputs_for_smart = _available_input_names(
        signal_fields=signal_fields,
        ctx_cont=ctx_cont,
        ctx_cat=ctx_cat,
        sequence_features=sequence_features,
        chart_geometry_features=chart_features,
        candlestick_features=candle_features,
        smart_features=smart_features,
        schema_columns=schema_columns + external_source_schema_columns,
    )
    smart_layer_rows = _smart_layer_rows(available_inputs_for_smart)
    smart_missing_required_layers = [
        row["label"] for row in smart_layer_rows if int(row["missing_required_source_field_count"]) > 0
    ]
    smart_candidate_extension_feature_count = (
        len(sequence_features) + len(chart_features) + len(candle_features) + len(smart_features)
    )
    smart_candidate_expected_signal_dim = (
        DEFAULT_BASE_SIGNAL_FEATURE_COUNT + smart_candidate_extension_feature_count
    )
    active_named_for_grouping = (
        signal_fields
        + [f"ctx_cont.{name}" for name in ctx_cont]
        + [f"ctx_cat.{name}" for name in ctx_cat]
        + sequence_features
        + chart_features
        + candle_features
    )
    smart_candidate_named_for_grouping = active_named_for_grouping + smart_features
    plan = _specialist_plan(
        signal_fields=signal_fields,
        ctx_cont=ctx_cont,
        ctx_cat=ctx_cat,
        sequence_features=sequence_features,
        chart_geometry_features=chart_features,
        candlestick_features=candle_features,
    )
    contract_provenance = _specialist_contract_provenance()
    active_contract = contract_provenance["active_foundation"]
    target_contract = contract_provenance["target_challenger"]

    decision = "READY_FOR_SPECIALIST_AI_DESIGN_REVIEW" if not failures else "FAIL"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "entry_feature_ai_inventory_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "dataset_dir": str(dataset_dir),
        "data_splits": splits,
        "side_effects_started": {
            "training": False,
            "replay": False,
            "iql_distillation": False,
            "shadow": False,
            "live": False,
            "promotion": False,
        },
        "counts": {
            "signal_fields": len(signal_fields),
            "ctx_cont": len(ctx_cont),
            "ctx_cat": len(ctx_cat),
            "sequence_extension_features": len(sequence_features),
            "foundation_structure_features": len(FOUNDATION_STRUCTURE_FEATURE_NAMES),
            "foundation_structure_source_fields": len(FOUNDATION_STRUCTURE_SOURCE_FIELDS),
            "chart_geometry_features": len(chart_features),
            "chart_geometry_source_fields": len(chart_sources),
            "candlestick_pattern_features": len(candle_features),
            "candlestick_pattern_source_fields": len(candle_sources),
            "smart_layer_count": len(smart_layer_features_by_label),
            "smart_layer_features": len(smart_features),
            "smart_candidate_extension_features": smart_candidate_extension_feature_count,
            "smart_candidate_expected_signal_dim": smart_candidate_expected_signal_dim,
            "smart_layers_missing_required_source_fields": len(smart_missing_required_layers),
            "external_source_schema_columns": len(external_source_schema_columns),
            "label_or_target_columns": len(label_or_target_columns),
            "all_named_inputs_including_generated": len(smart_candidate_named_for_grouping),
        },
        "all_inputs": {
            "signal_fields": signal_fields,
            "ctx_cont": ctx_cont,
            "ctx_cat": ctx_cat,
            "sequence_extension_features": sequence_features,
            "foundation_structure_features": list(FOUNDATION_STRUCTURE_FEATURE_NAMES),
            "foundation_structure_source_fields": list(FOUNDATION_STRUCTURE_SOURCE_FIELDS),
            "chart_geometry_features": chart_features,
            "chart_geometry_source_fields": chart_sources,
            "candlestick_pattern_features": candle_features,
            "candlestick_pattern_source_fields": candle_sources,
            "smart_layer_features_by_label": smart_layer_features_by_label,
            "smart_layer_features": smart_features,
            "external_source_schema_columns": external_source_schema_columns,
            "label_or_target_columns": label_or_target_columns,
        },
        "feature_rows": (
            _feature_rows(signal_fields, input_surface="seq_and_snap_signal", source="active_split_manifest")
            + _feature_rows([f"ctx_cont.{name}" for name in ctx_cont], input_surface="ctx_cont", source="active_split_manifest")
            + _feature_rows([f"ctx_cat.{name}" for name in ctx_cat], input_surface="ctx_cat", source="active_split_manifest")
            + _feature_rows(sequence_features, input_surface="seq_structure_extension", source="sequence_structure_manifest")
            + _feature_rows(chart_features, input_surface="chart_geometry_challenger", source="chart_geometry_manifest")
            + _feature_rows(candle_features, input_surface="candlestick_pattern_challenger", source="candlestick_manifest")
            + _feature_rows(smart_features, input_surface="smart_seq495_candidate", source="smart_layer_manifest")
        ),
        "active_input_counts_by_specialist": _count_groups(active_named_for_grouping),
        "active_inputs_by_specialist": _group(active_named_for_grouping),
        "smart_candidate": {
            "manifest_variant": f"smart_seq{smart_candidate_expected_signal_dim}_candidate",
            "base_signal_features": DEFAULT_BASE_SIGNAL_FEATURE_COUNT,
            "seq215_extension_features": len(sequence_features) + len(chart_features) + len(candle_features),
            "smart_layer_features": len(smart_features),
            "combined_extension_features": smart_candidate_extension_feature_count,
            "expected_signal_dim": smart_candidate_expected_signal_dim,
            "training_allowed": False,
            "dataset_rebuild_required_before_training": True,
            "feature_audit_required_before_training": True,
            "specialist_audit_required_before_training": True,
            "liveness_noncollapse_required_before_training": True,
            "missing_required_source_field_layers": smart_missing_required_layers,
            "source_coverage_all_required_available": not smart_missing_required_layers,
            "external_source_schema_columns_used": len(external_source_schema_columns),
        },
        "smart_layer_source_coverage": smart_layer_rows,
        "smart_candidate_input_counts_by_specialist": _count_groups(smart_candidate_named_for_grouping),
        "smart_candidate_inputs_by_specialist": _group(smart_candidate_named_for_grouping),
        "ranked_specialist_models": plan,
        "specialist_contract_provenance": contract_provenance,
        "active_foundation_contract_mode": ACTIVE_SPECIALIST_CONTRACT_MODE,
        "active_foundation_required_training_specialists": active_contract["required_training_specialists"],
        "active_foundation_specialist_model_contract": active_contract["specialist_model_contract"],
        "target_challenger_contract_mode": TARGET_CHALLENGER_CONTRACT_MODE,
        "target_challenger_required_training_specialists": target_contract["required_training_specialists"],
        "target_challenger_specialist_model_contract": target_contract["specialist_model_contract"],
        "target_challenger_contract_update_required_before_training": target_contract[
            "contract_update_required_before_training"
        ],
        "current_specialist_contract_mode": ACTIVE_SPECIALIST_CONTRACT_MODE,
        "current_required_training_specialists": active_contract["required_training_specialists"],
        "current_specialist_model_contract": active_contract["specialist_model_contract"],
        "foundation_structure_feature_version": FOUNDATION_STRUCTURE_FEATURE_VERSION,
        "chart_geometry_feature_version": CHART_GEOMETRY_FEATURE_VERSION,
        "candlestick_pattern_feature_version": CANDLESTICK_PATTERN_FEATURE_VERSION,
        "sequence_structure_manifest": sequence_meta,
        "chart_geometry_manifest": chart_meta,
        "candlestick_manifest": candle_meta,
        "split_contracts": {
            split: {
                key: value
                for key, value in contract.items()
                if key not in {"signal_fields", "ctx_cont_names", "ctx_cat_names"}
            }
            for split, contract in split_contracts.items()
        },
        "failures": failures,
        "next_required_gate": NEXT_REQUIRED_GATE,
        "training_allowed": False,
        "shadow_live_promotion_allowed": False,
    }
    json_path = out_dir / f"ENTRY_FEATURE_AI_INVENTORY_{timestamp}.json"
    md_path = out_dir / f"ENTRY_FEATURE_AI_INVENTORY_{timestamp}.md"
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_FEATURE_AI_INVENTORY_latest.json").write_text(
        json_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out_dir / "ENTRY_FEATURE_AI_INVENTORY_latest.md").write_text(
        md_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "counts": report["counts"],
                    "json_path": report["json_path"],
                    "md_path": report["md_path"],
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
    ap.add_argument("--dataset-dir", default=str(FOUNDATION_DATASET_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--data-splits", default="train,val,test")
    ap.add_argument("--seq-structure-manifest", default=str(SEQ_STRUCTURE_MANIFEST))
    ap.add_argument("--chart-geometry-manifest", default=str(DEFAULT_CHART_GEOMETRY_MANIFEST))
    ap.add_argument("--candlestick-manifest", default=str(DEFAULT_CANDLESTICK_MANIFEST))
    ap.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
