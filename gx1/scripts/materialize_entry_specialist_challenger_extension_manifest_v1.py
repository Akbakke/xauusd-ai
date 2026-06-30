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

from gx1.features.entry_chart_geometry_v1 import CHART_GEOMETRY_FEATURE_VERSION
from gx1.features.entry_candlestick_patterns_v1 import CANDLESTICK_PATTERN_FEATURE_VERSION
from gx1.features.entry_foundation_structure_v1 import FOUNDATION_STRUCTURE_FEATURE_VERSION
from gx1.features.entry_specialist_feature_groups_v1 import (
    REQUIRED_TRAINING_SPECIALISTS,
    SPECIALIST_MODEL_CONTRACT,
    classify_entry_specialist_feature,
    group_features_by_specialist,
)
from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT, SEQ_STRUCTURE_MANIFEST


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_specialist_challenger_extension_manifest_20260630_v1"
DEFAULT_CHART_GEOMETRY_MANIFEST = (
    REPORTS_ROOT
    / "entry_chart_geometry_challenger_audit_20260630_v1/ENTRY_CHART_GEOMETRY_CHALLENGER_MANIFEST_latest.json"
)
DEFAULT_CANDLESTICK_MANIFEST = (
    REPORTS_ROOT
    / "entry_candlestick_pattern_challenger_audit_20260630_v1/ENTRY_CANDLESTICK_PATTERN_CHALLENGER_MANIFEST_latest.json"
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
        f"- Duplicate dropped: `{report['counts']['duplicate_feature_count']}`",
        f"- Failure count: `{len(report['failures'])}`",
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

    foundation = _read_json(foundation_path)
    chart = _read_json(chart_path)
    candle = _read_json(candle_path)
    foundation_features = _selected_features(foundation, label="foundation sequence manifest")
    chart_features = _selected_features(chart, label="chart geometry manifest")
    candle_features = _selected_features(candle, label="candlestick manifest")

    failures: list[str] = []
    if chart.get("decision") != "READY_FOR_CHALLENGER_DATASET_REBUILD":
        failures.append(f"chart geometry manifest not rebuild-ready: {chart.get('decision')}")
    if candle.get("decision") != "READY_FOR_CHALLENGER_DATASET_REBUILD":
        failures.append(f"candlestick manifest not rebuild-ready: {candle.get('decision')}")
    if foundation.get("foundation_structure_all_required_selected") is not True:
        failures.append("foundation sequence manifest does not prove all required foundation structure features selected")

    combined, duplicates = _dedupe_preserve_order(foundation_features + chart_features + candle_features)
    grouped = group_features_by_specialist(combined)
    unmapped = grouped.get("unmapped", [])
    if unmapped:
        failures.append(f"unmapped combined features: {unmapped[:30]} total={len(unmapped)}")

    trainable_challengers = [
        specialist
        for specialist in ("chart_geometry_encoder", "price_action_candle_encoder")
        if specialist in set(REQUIRED_TRAINING_SPECIALISTS)
    ]
    if trainable_challengers:
        failures.append(f"challenger specialists already trainable before contract update: {trainable_challengers}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    decision = "READY_FOR_CHALLENGER_DATASET_REBUILD_MANIFEST" if not failures else "FAIL"
    manifest = {
        "schema_version": "entry_specialist_challenger_extension_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "purpose": (
            "manifest-only selected feature order for rebuilding Entry sequence arrays with "
            "foundation structure, numeric chart geometry, and closed-bar candlestick pattern inputs"
        ),
        "manifest_only": True,
        "selected_features": combined,
        "selected_feature_count": int(len(combined)),
        "feature_counts_by_specialist": _counter(combined),
        "features_by_specialist": grouped,
        "feature_rows": _feature_rows(combined),
        "source_manifests": {
            "foundation_sequence_extension": _source_meta(foundation_path, foundation, label="foundation_sequence_extension"),
            "chart_geometry_challenger": _source_meta(chart_path, chart, label="chart_geometry_challenger"),
            "candlestick_challenger": _source_meta(candle_path, candle, label="candlestick_challenger"),
        },
        "source_feature_counts": {
            "foundation_sequence_extension": int(len(foundation_features)),
            "chart_geometry_challenger": int(len(chart_features)),
            "candlestick_challenger": int(len(candle_features)),
        },
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
        "contract_update_required_before_training": True,
        "training_allowed": False,
        "shadow_live_promotion_allowed": False,
        "current_required_training_specialists": list(REQUIRED_TRAINING_SPECIALISTS),
        "current_specialist_model_contract": SPECIALIST_MODEL_CONTRACT,
        "required_next_specialist_contract_review": {
            "must_decide_exact_trainable_specialists": [
                "structure_swing_encoder",
                "smc_liquidity_encoder",
                "trend_ema_encoder",
                "vol_compression_encoder",
                "momentum_flow_encoder",
                "session_regime_encoder",
                "chart_geometry_encoder",
                "price_action_candle_encoder",
            ],
            "must_update_bundle_audit_contract": True,
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
                "--output_dir",
                "<new_challenger_dataset_dir>",
            ],
            "ram_note": (
                "The combined extension adds 69 challenger features on top of the active 105-feature "
                "foundation extension. Rebuild under gx1_capped_run with conservative memory and "
                "streaming batch settings."
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
            "combined_selected_features": int(len(combined)),
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

    manifest_json = out_dir / f"ENTRY_SPECIALIST_CHALLENGER_EXTENSION_MANIFEST_{timestamp}.json"
    report_json = out_dir / f"ENTRY_SPECIALIST_CHALLENGER_EXTENSION_REPORT_{timestamp}.json"
    report_md = out_dir / f"ENTRY_SPECIALIST_CHALLENGER_EXTENSION_REPORT_{timestamp}.md"
    manifest["manifest_json_path"] = str(manifest_json)
    report["json_path"] = str(report_json)
    report["md_path"] = str(report_md)

    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(report_md, report)
    (out_dir / "ENTRY_SPECIALIST_CHALLENGER_EXTENSION_MANIFEST_latest.json").write_text(
        manifest_json.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out_dir / "ENTRY_SPECIALIST_CHALLENGER_EXTENSION_REPORT_latest.json").write_text(
        report_json.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out_dir / "ENTRY_SPECIALIST_CHALLENGER_EXTENSION_REPORT_latest.md").write_text(
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
    ap.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
