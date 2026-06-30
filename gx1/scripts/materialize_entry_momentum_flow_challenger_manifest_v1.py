#!/usr/bin/env python3
"""Materialize a report-only Entry momentum/flow challenger manifest."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gx1.features.entry_momentum_flow_v1 import (
    MOMENTUM_FLOW_FEATURE_NAMES,
    MOMENTUM_FLOW_FEATURE_VERSION,
    MOMENTUM_FLOW_OPTIONAL_SOURCE_FIELDS,
    MOMENTUM_FLOW_REQUIRED_SOURCE_FIELDS,
)
from gx1.features.entry_specialist_feature_groups_v1 import classify_entry_specialist_feature
from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_momentum_flow_challenger_manifest_20260630_v1"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    return str(obj)


def _feature_rows() -> list[dict[str, Any]]:
    return [
        {
            "index": int(i),
            "name": name,
            "specialist": classify_entry_specialist_feature(name),
        }
        for i, name in enumerate(MOMENTUM_FLOW_FEATURE_NAMES)
    ]


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    manifest = report["manifest"]
    lines = [
        "# Entry Momentum/Flow Challenger Manifest",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Feature version: `{manifest['momentum_flow_feature_version']}`",
        f"- Features: `{manifest['selected_feature_count']}`",
        f"- Specialist: `{manifest['specialist']}`",
        f"- Failure count: `{len(report['failures'])}`",
        "",
        "## Required Sources",
        "",
    ]
    lines.extend(f"- `{name}`" for name in manifest["required_source_fields"])
    lines.extend(["", "## Optional Sources", ""])
    lines.extend(f"- `{name}`" for name in manifest["optional_source_fields"])
    lines.extend(["", "## Features", ""])
    lines.extend(f"- `{name}`" for name in manifest["selected_features"])
    lines.extend(["", "## Failures", ""])
    lines.extend([f"- {failure}" for failure in report["failures"]] or ["- None"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _feature_rows()
    failures = [
        f"momentum flow feature misrouted: {row['name']} -> {row['specialist']}"
        for row in rows
        if row["specialist"] != "momentum_flow_encoder"
    ]
    decision = "READY_FOR_MOMENTUM_FLOW_CHALLENGER_REVIEW" if not failures else "FAIL"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "schema_version": "entry_momentum_flow_challenger_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "purpose": (
            "manifest-only candidate feature order for a future audited momentum/flow "
            "extension; this does not rebuild datasets, train, replay, distill, shadow, "
            "live, or promote"
        ),
        "manifest_only": True,
        "specialist": "momentum_flow_encoder",
        "momentum_flow_feature_version": MOMENTUM_FLOW_FEATURE_VERSION,
        "selected_features": list(MOMENTUM_FLOW_FEATURE_NAMES),
        "selected_feature_count": int(len(MOMENTUM_FLOW_FEATURE_NAMES)),
        "feature_rows": rows,
        "required_source_fields": list(MOMENTUM_FLOW_REQUIRED_SOURCE_FIELDS),
        "optional_source_fields": list(MOMENTUM_FLOW_OPTIONAL_SOURCE_FIELDS),
        "dataset_rebuild_required_before_training": True,
        "training_allowed": False,
        "shadow_live_promotion_allowed": False,
        "builder_usage": {
            "module": "gx1.features.entry_momentum_flow_v1:build_entry_momentum_flow_layer",
            "integration_note": (
                "Wire into a future challenger sequence manifest only after feature liveness "
                "and specialist routing audits are updated for the added signal width."
            ),
        },
    }
    report = {
        "schema_version": "entry_momentum_flow_challenger_manifest_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "manifest": manifest,
        "failures": failures,
        "side_effects_started": {
            "training": False,
            "replay": False,
            "iql_distillation": False,
            "shadow": False,
            "live": False,
            "promotion": False,
        },
        "next_required_gate": "wire as challenger-only extension, rebuild candidate dataset, then feature audit -> specialist audit",
        "training_allowed": False,
        "shadow_live_promotion_allowed": False,
    }
    json_path = out_dir / f"ENTRY_MOMENTUM_FLOW_CHALLENGER_MANIFEST_{timestamp}.json"
    md_path = out_dir / f"ENTRY_MOMENTUM_FLOW_CHALLENGER_MANIFEST_{timestamp}.md"
    latest_json = out_dir / "ENTRY_MOMENTUM_FLOW_CHALLENGER_MANIFEST_latest.json"
    latest_md = out_dir / "ENTRY_MOMENTUM_FLOW_CHALLENGER_MANIFEST_latest.md"
    payload = json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n"
    json_path.write_text(payload, encoding="utf-8")
    latest_json.write_text(payload, encoding="utf-8")
    _write_markdown(md_path, report)
    _write_markdown(latest_md, report)
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    report = run(args)
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    if args.fail_on_audit_fail and report["decision"] != "READY_FOR_MOMENTUM_FLOW_CHALLENGER_REVIEW":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
