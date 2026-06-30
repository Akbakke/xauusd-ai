#!/usr/bin/env python3
"""Materialize a report-only Entry trend/EMA extension manifest.

This does not rebuild datasets and does not start training, replay, IQL,
shadow, live or promotion paths. It only records the proposed trend/EMA feature
layer and its source-field/routing contract for later seq215 review.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gx1.features.entry_specialist_feature_groups_v1 import classify_entry_specialist_feature
from gx1.features.entry_trend_ema_v1 import (
    TREND_EMA_FEATURE_DESCRIPTIONS,
    TREND_EMA_FEATURE_NAMES,
    TREND_EMA_FEATURE_VERSION,
    TREND_EMA_SOURCE_FIELDS,
)
from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_trend_ema_extension_manifest_20260630_v1"
EXPECTED_SPECIALIST = "trend_ema_encoder"
NEXT_REQUIRED_GATE = (
    "manual review, then optional seq215 dataset-rebuild plan; any rebuild/audit must remain separate "
    "from smoke training, replay, IQL, shadow, live and promotion gates"
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    return str(obj)


def _feature_rows(features: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "index": int(index),
            "name": name,
            "specialist": classify_entry_specialist_feature(name),
            "description": TREND_EMA_FEATURE_DESCRIPTIONS.get(name.rsplit("ema_", 1)[-1]),
        }
        for index, name in enumerate(features)
    ]


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    manifest = report["manifest"]
    lines = [
        "# Entry Trend/EMA Extension Manifest",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Feature version: `{manifest['trend_ema_feature_version']}`",
        f"- Proposed features: `{manifest['selected_feature_count']}`",
        f"- Source fields: `{len(manifest['source_fields'])}`",
        f"- Expected specialist: `{manifest['expected_specialist']}`",
        f"- All features route to expected specialist: `{manifest['all_features_route_to_expected_specialist']}`",
        f"- Failure count: `{len(report['failures'])}`",
        "",
        "## Proposed Features",
        "",
    ]
    for row in manifest["feature_rows"]:
        lines.append(f"- `{row['name']}`: {row.get('description') or row['specialist']}")
    lines.extend(["", "## Source Fields", ""])
    lines.extend(f"- `{name}`" for name in manifest["source_fields"])
    lines.extend(["", "## Side Effects", ""])
    for key, value in report["side_effects_started"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Failures", ""])
    lines.extend([f"- {failure}" for failure in report["failures"]] or ["- None"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = _feature_rows(TREND_EMA_FEATURE_NAMES)
    failures: list[str] = []
    duplicate_features = sorted({name for name in TREND_EMA_FEATURE_NAMES if TREND_EMA_FEATURE_NAMES.count(name) > 1})
    duplicate_sources = sorted({name for name in TREND_EMA_SOURCE_FIELDS if TREND_EMA_SOURCE_FIELDS.count(name) > 1})
    misrouted = [row for row in feature_rows if row["specialist"] != EXPECTED_SPECIALIST]
    if duplicate_features:
        failures.append(f"duplicate trend/EMA features: {duplicate_features}")
    if duplicate_sources:
        failures.append(f"duplicate trend/EMA source fields: {duplicate_sources}")
    if misrouted:
        failures.append(f"trend/EMA features not routed to {EXPECTED_SPECIALIST}: {misrouted[:10]}")

    decision = "READY_FOR_TREND_EMA_EXTENSION_REVIEW" if not failures else "FAIL"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = OrderedDict(
        [
            ("schema_version", "entry_trend_ema_extension_manifest_v1"),
            ("created_utc", datetime.now(timezone.utc).isoformat()),
            ("decision", decision),
            ("manifest_only", True),
            ("purpose", "proposed causal trend/EMA specialist feature layer for later seq215 review"),
            ("trend_ema_feature_version", TREND_EMA_FEATURE_VERSION),
            ("expected_specialist", EXPECTED_SPECIALIST),
            ("selected_features", list(TREND_EMA_FEATURE_NAMES)),
            ("selected_feature_count", int(len(TREND_EMA_FEATURE_NAMES))),
            ("source_fields", list(TREND_EMA_SOURCE_FIELDS)),
            ("feature_rows", feature_rows),
            ("all_features_route_to_expected_specialist", bool(not misrouted)),
            ("dataset_rebuild_required_before_training", True),
            ("training_allowed", False),
            ("replay_allowed", False),
            ("iql_allowed", False),
            ("shadow_live_promotion_allowed", False),
            ("next_required_gate", NEXT_REQUIRED_GATE),
        ]
    )
    report = {
        "schema_version": "entry_trend_ema_extension_manifest_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "manifest": manifest,
        "counts": {
            "selected_features": int(len(TREND_EMA_FEATURE_NAMES)),
            "source_fields": int(len(TREND_EMA_SOURCE_FIELDS)),
            "misrouted_features": int(len(misrouted)),
            "duplicate_features": int(len(duplicate_features)),
            "duplicate_sources": int(len(duplicate_sources)),
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
        "next_required_gate": NEXT_REQUIRED_GATE,
        "training_allowed": False,
        "shadow_live_promotion_allowed": False,
    }

    manifest_json = out_dir / f"ENTRY_TREND_EMA_EXTENSION_MANIFEST_{timestamp}.json"
    report_json = out_dir / f"ENTRY_TREND_EMA_EXTENSION_REPORT_{timestamp}.json"
    report_md = out_dir / f"ENTRY_TREND_EMA_EXTENSION_REPORT_{timestamp}.md"
    manifest["manifest_json_path"] = str(manifest_json)
    report["json_path"] = str(report_json)
    report["md_path"] = str(report_md)

    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(report_md, report)
    (out_dir / "ENTRY_TREND_EMA_EXTENSION_MANIFEST_latest.json").write_text(
        manifest_json.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out_dir / "ENTRY_TREND_EMA_EXTENSION_REPORT_latest.json").write_text(
        report_json.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out_dir / "ENTRY_TREND_EMA_EXTENSION_REPORT_latest.md").write_text(
        report_md.read_text(encoding="utf-8"), encoding="utf-8"
    )
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": decision,
                    "counts": report["counts"],
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
    ap.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
