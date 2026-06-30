#!/usr/bin/env python3
"""Write a report-only manifest for dormant Entry session/regime features."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.features.entry_session_regime_interactions_v1 import (
    SESSION_REGIME_INTERACTION_FEATURE_NAMES,
    SESSION_REGIME_INTERACTION_FEATURE_VERSION,
    SESSION_REGIME_INTERACTION_SOURCE_FIELDS,
)
from gx1.features.entry_specialist_feature_groups_v1 import classify_entry_specialist_feature
from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_session_regime_interaction_manifest_20260630_v1"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Session Regime Interaction Manifest",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Feature version: `{report['feature_version']}`",
        f"- Selected features: `{report['selected_feature_count']}`",
        f"- Failure count: `{len(report['failures'])}`",
        f"- Training allowed: `{report['training_allowed']}`",
        "",
        "## Failures",
        "",
    ]
    lines.extend([f"- {failure}" for failure in report["failures"]] or ["- None"])
    lines.extend(["", "## Selected Features", ""])
    lines.extend(f"- `{name}`" for name in report["selected_features"])
    lines.extend(["", "## Source Fields", ""])
    lines.extend(f"- `{name}`" for name in report["source_fields"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    features = list(SESSION_REGIME_INTERACTION_FEATURE_NAMES)
    routing = {name: classify_entry_specialist_feature(name) for name in features}
    misrouted = {name: group for name, group in routing.items() if group != "session_regime_encoder"}
    failures = []
    if misrouted:
        failures.append(f"session/regime features misrouted: {dict(list(misrouted.items())[:20])} total={len(misrouted)}")

    decision = "READY_FOR_CHALLENGER_MANIFEST_REVIEW" if not failures else "FAIL"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "entry_session_regime_interaction_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "feature_version": SESSION_REGIME_INTERACTION_FEATURE_VERSION,
        "selected_feature_count": int(len(features)),
        "selected_features": features,
        "source_field_count": int(len(SESSION_REGIME_INTERACTION_SOURCE_FIELDS)),
        "source_fields": list(SESSION_REGIME_INTERACTION_SOURCE_FIELDS),
        "specialist": "session_regime_encoder",
        "routing": routing,
        "failures": failures,
        "dataset_rebuild_required_before_training": True,
        "training_allowed": False,
        "replay_allowed": False,
        "iql_allowed": False,
        "shadow_live_promotion_allowed": False,
        "next_required_gate": (
            "Wire these features into a seq215+ challenger manifest, rebuild the dataset, "
            "then rerun feature foundation, specialist-group and smoke-dataset audits before any training."
        ),
    }
    json_path = out_dir / f"ENTRY_SESSION_REGIME_INTERACTION_MANIFEST_{timestamp}.json"
    md_path = out_dir / f"ENTRY_SESSION_REGIME_INTERACTION_MANIFEST_{timestamp}.md"
    latest_json = out_dir / "ENTRY_SESSION_REGIME_INTERACTION_MANIFEST_latest.json"
    latest_md = out_dir / "ENTRY_SESSION_REGIME_INTERACTION_MANIFEST_latest.md"
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    json_text = json.dumps(report, indent=2, sort_keys=True)
    json_path.write_text(json_text + "\n", encoding="utf-8")
    latest_json.write_text(json_text + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    _write_markdown(latest_md, report)
    if not bool(args.quiet):
        print(json.dumps(report, indent=2, sort_keys=True))
    if failures and bool(args.fail_on_audit_fail):
        raise SystemExit(1)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
