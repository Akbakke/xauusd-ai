#!/usr/bin/env python3
"""Materialize a report-only manifest for candidate SMC/liquidity quality features."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gx1.features.entry_smc_liquidity_quality_v1 import (
    SMC_LIQUIDITY_QUALITY_FEATURE_NAMES,
    SMC_LIQUIDITY_QUALITY_FEATURE_VERSION,
    SMC_LIQUIDITY_QUALITY_OPTIONAL_FIELDS,
    SMC_LIQUIDITY_QUALITY_SOURCE_FIELDS,
)
from gx1.features.entry_specialist_feature_groups_v1 import classify_entry_specialist_feature
from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_smc_liquidity_quality_manifest_20260630_v1"


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
            "index": int(i),
            "name": name,
            "specialist": classify_entry_specialist_feature(name),
        }
        for i, name in enumerate(features)
    ]


def _write_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Entry SMC Liquidity Quality Manifest",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Feature version: `{report['feature_version']}`",
        f"- Candidate features: `{report['candidate_feature_count']}`",
        f"- Required source fields: `{report['required_source_field_count']}`",
        f"- Optional context fields: `{report['optional_context_field_count']}`",
        f"- Training allowed: `{report['training_allowed']}`",
        f"- Side effects started: `{report['side_effects_started']}`",
        "",
        "## Candidate Features",
        "",
    ]
    lines.extend(f"- `{name}`" for name in report["selected_features"])
    lines.extend(
        [
            "",
            "## Source Fields",
            "",
        ]
    )
    lines.extend(f"- `{name}`" for name in report["source_fields"])
    lines.extend(
        [
            "",
            "## Optional Fields",
            "",
        ]
    )
    lines.extend(f"- `{name}`" for name in report["optional_fields"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    features = tuple(SMC_LIQUIDITY_QUALITY_FEATURE_NAMES)
    rows = _feature_rows(features)
    counts = Counter(row["specialist"] for row in rows)
    failures = [
        f"candidate feature is not routed to smc_liquidity_encoder: {row['name']} -> {row['specialist']}"
        for row in rows
        if row["specialist"] != "smc_liquidity_encoder"
    ]
    report = {
        "schema_version": "entry_smc_liquidity_quality_manifest_v1",
        "created_utc": created,
        "decision": "READY_FOR_FEATURE_AUDIT_REVIEW" if not failures else "BLOCKED_FOR_ROUTING_REVIEW",
        "purpose": (
            "Report-only candidate SMC/liquidity quality layer. It is not part of active seq146 or "
            "challenger seq215 until a future manifest/rebuild/audit gate explicitly adopts it."
        ),
        "feature_version": SMC_LIQUIDITY_QUALITY_FEATURE_VERSION,
        "selected_features": list(features),
        "candidate_feature_count": int(len(features)),
        "source_fields": list(SMC_LIQUIDITY_QUALITY_SOURCE_FIELDS),
        "required_source_field_count": int(len(SMC_LIQUIDITY_QUALITY_SOURCE_FIELDS)),
        "optional_fields": list(SMC_LIQUIDITY_QUALITY_OPTIONAL_FIELDS),
        "optional_context_field_count": int(len(SMC_LIQUIDITY_QUALITY_OPTIONAL_FIELDS)),
        "feature_rows": rows,
        "feature_counts_by_specialist": dict(sorted(counts.items())),
        "failures": failures,
        "training_allowed": False,
        "replay_allowed": False,
        "iql_allowed": False,
        "shadow_live_promotion_allowed": False,
        "side_effects_started": False,
        "next_required_gate": (
            "Wire into a forked sequence-extension manifest, rebuild under capped RAM, then run "
            "entry feature foundation audit and specialist feature group audit before any seq215 smoke gate."
        ),
    }
    json_path = out_dir / f"ENTRY_SMC_LIQUIDITY_QUALITY_MANIFEST_{created}.json"
    md_path = out_dir / f"ENTRY_SMC_LIQUIDITY_QUALITY_MANIFEST_{created}.md"
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    _write_md(report, md_path)
    shutil.copyfile(json_path, out_dir / "ENTRY_SMC_LIQUIDITY_QUALITY_MANIFEST_latest.json")
    shutil.copyfile(md_path, out_dir / "ENTRY_SMC_LIQUIDITY_QUALITY_MANIFEST_latest.md")
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    if not args.quiet:
        print(json.dumps({"decision": report["decision"], "json_path": report["json_path"]}, indent=2))
    if args.fail_on_audit_fail and report["decision"] != "READY_FOR_FEATURE_AUDIT_REVIEW":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
