#!/usr/bin/env python3
"""Materialize Entry IQL-student replay evidence from an explicit trade log.

This script does not train IQL, run replay, build adapters, promote, shadow or
trade. It only converts a supplied IQL-student replay trade log into the metrics
and manifest artifacts consumed by the post-distillation comparison gate.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.scripts.materialize_entry_candidate_replay_evidence_v1 import (
    _json_default,
    _read_table,
    audit_iql_transition_trades,
    build_replay_tables,
    normalize_trades,
)
from gx1.scripts.materialize_entry_iql_distillation_contract_v1 import (
    DEFAULT_OUT_DIR as DISTILL_CONTRACT_OUT_DIR,
    IQL_DISTILLATION_REQUIRED_ARTIFACT_KEYS,
    _sha256_file,
)
from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


DEFAULT_DISTILL_CONTRACT_JSON = DISTILL_CONTRACT_OUT_DIR / "ENTRY_IQL_DISTILLATION_CONTRACT_latest.json"
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_iql_distillation_replay_20260628_v1"
REQUIRED_DECISION = "ENTRY_IQL_DISTILLATION_CONTRACT_READY"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing IQL distillation contract: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"IQL distillation contract is not an object: {path}")
    return payload


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _distillation_artifact_hash_contract(distillation_contract: dict[str, Any]) -> dict[str, Any]:
    artifact_paths = (
        distillation_contract.get("artifact_paths")
        if isinstance(distillation_contract.get("artifact_paths"), dict)
        else {}
    )
    artifact_sha256 = (
        distillation_contract.get("artifact_sha256")
        if isinstance(distillation_contract.get("artifact_sha256"), dict)
        else {}
    )
    failures: list[str] = []
    checks: dict[str, dict[str, Any]] = {}
    if not artifact_paths:
        failures.append("IQL distillation contract missing artifact_paths")
    if not artifact_sha256:
        failures.append("IQL distillation contract missing artifact_sha256")
    for key in IQL_DISTILLATION_REQUIRED_ARTIFACT_KEYS:
        raw = str(artifact_paths.get(key) or "")
        path = Path(raw).expanduser().resolve() if raw else None
        exists = bool(path and path.is_file())
        expected = str(artifact_sha256.get(key) or "")
        observed = _sha256_file(path) if exists and path is not None else ""
        ok = bool(expected and observed and expected == observed)
        checks[key] = {
            "path": str(path) if path is not None else "",
            "expected": expected,
            "observed": observed,
            "exists": exists,
            "ok": ok,
        }
        if not ok:
            failures.append(f"IQL distillation artifact hash mismatch or missing: {key}")
    return {
        "ready": not failures,
        "required_keys": list(IQL_DISTILLATION_REQUIRED_ARTIFACT_KEYS),
        "checks": checks,
        "failures": failures,
    }


def _distillation_identity_contract(
    *,
    distillation_contract_path: Path,
    distillation_contract: dict[str, Any],
) -> dict[str, Any]:
    identity = (
        distillation_contract.get("evidence_identity")
        if isinstance(distillation_contract.get("evidence_identity"), dict)
        else {}
    )
    candidate_bundle_dir = str(identity.get("candidate_bundle_dir") or "")
    selective_bundle_dir = str(identity.get("selective_edge_bundle_dir") or "")
    replay_bundle_dir = str(identity.get("replay_identity_candidate_bundle_dir") or "")
    candidate_replay_manifest_json = str(identity.get("replay_evidence_manifest_json") or "")
    candidate_replay_manifest_path = Path(candidate_replay_manifest_json).expanduser() if candidate_replay_manifest_json else None
    candidate_replay_manifest = (
        _read_json_if_exists(candidate_replay_manifest_path) if candidate_replay_manifest_path is not None else {}
    )
    candidate_replay_identity = (
        candidate_replay_manifest.get("replay_identity_contract")
        if isinstance(candidate_replay_manifest.get("replay_identity_contract"), dict)
        else {}
    )
    artifact_hash_contract = _distillation_artifact_hash_contract(distillation_contract)
    pretrain_provenance = (
        distillation_contract.get("candidate_pretrain_provenance_contract")
        if isinstance(distillation_contract.get("candidate_pretrain_provenance_contract"), dict)
        else {}
    )
    smoke_dataset_provenance = (
        distillation_contract.get("smoke_dataset_provenance_contract")
        if isinstance(distillation_contract.get("smoke_dataset_provenance_contract"), dict)
        else {}
    )
    specialist_set_provenance = (
        distillation_contract.get("specialist_set_provenance_contract")
        if isinstance(distillation_contract.get("specialist_set_provenance_contract"), dict)
        else {}
    )
    specialist_model_provenance = (
        distillation_contract.get("specialist_model_provenance_contract")
        if isinstance(distillation_contract.get("specialist_model_provenance_contract"), dict)
        else {}
    )
    bundle_specialist_model_provenance = (
        distillation_contract.get("bundle_specialist_model_provenance_contract")
        if isinstance(distillation_contract.get("bundle_specialist_model_provenance_contract"), dict)
        else {}
    )
    replay_artifact_provenance = (
        distillation_contract.get("replay_artifact_provenance_contract")
        if isinstance(distillation_contract.get("replay_artifact_provenance_contract"), dict)
        else {}
    )
    failures: list[str] = []
    if str(distillation_contract.get("decision")) != REQUIRED_DECISION:
        failures.append(f"IQL distillation contract decision is not ready: {distillation_contract.get('decision')}")
    if not bool(pretrain_provenance.get("ok")):
        failures.append("IQL distillation contract did not preserve candidate pretrain provenance")
    if not bool(smoke_dataset_provenance.get("ok")):
        failures.append("IQL distillation contract did not preserve smoke dataset audit provenance")
    if not bool(specialist_set_provenance.get("ok")):
        failures.append("IQL distillation contract did not preserve exact specialist set provenance")
    if not bool(specialist_model_provenance.get("ok")):
        failures.append("IQL distillation contract did not preserve specialist model contract provenance")
    if not bool(bundle_specialist_model_provenance.get("ok")):
        failures.append("IQL distillation contract did not preserve candidate bundle specialist model contract provenance")
    if not bool(replay_artifact_provenance.get("ok")):
        failures.append("IQL distillation contract did not preserve replay artifact provenance")
    if bool(distillation_contract.get("promotion_shadow_live_allowed")) is not False:
        failures.append("IQL distillation contract does not block promotion/shadow/live")
    if not identity:
        failures.append("IQL distillation contract does not carry evidence_identity")
    if not candidate_bundle_dir:
        failures.append("IQL distillation evidence_identity has no candidate_bundle_dir")
    if candidate_bundle_dir and selective_bundle_dir != candidate_bundle_dir:
        failures.append(
            "IQL distillation evidence_identity selective bundle mismatch: "
            f"{selective_bundle_dir} != {candidate_bundle_dir}"
        )
    if candidate_bundle_dir and replay_bundle_dir != candidate_bundle_dir:
        failures.append(
            "IQL distillation evidence_identity replay bundle mismatch: "
            f"{replay_bundle_dir} != {candidate_bundle_dir}"
        )
    if not bool(identity.get("replay_identity_ready")):
        failures.append("IQL distillation evidence_identity replay_identity_ready is false")
    if not candidate_replay_manifest_json:
        failures.append("IQL distillation evidence_identity has no replay_evidence_manifest_json")
    elif candidate_replay_manifest_path is not None and not candidate_replay_manifest_path.exists():
        failures.append(f"candidate replay evidence manifest is missing: {candidate_replay_manifest_path}")
    if candidate_replay_manifest:
        if str(candidate_replay_manifest.get("decision")) != "PASS":
            failures.append(
                "candidate replay evidence manifest decision is not PASS: "
                f"{candidate_replay_manifest.get('decision')}"
            )
        if not bool(candidate_replay_identity.get("ready")):
            failures.append("candidate replay evidence manifest identity contract is not ready")
        candidate_replay_bundle_dir = str(candidate_replay_identity.get("candidate_bundle_dir") or "")
        if candidate_bundle_dir and candidate_replay_bundle_dir != candidate_bundle_dir:
            failures.append(
                "candidate replay evidence manifest bundle mismatch: "
                f"{candidate_replay_bundle_dir} != {candidate_bundle_dir}"
            )
    failures.extend(artifact_hash_contract["failures"])

    return {
        "ready": not failures,
        "distillation_contract_json": str(distillation_contract_path),
        "candidate_bundle_dir": candidate_bundle_dir,
        "selective_edge_bundle_dir": selective_bundle_dir,
        "replay_identity_candidate_bundle_dir": replay_bundle_dir,
        "no_xgb_bundle_dir": str(identity.get("no_xgb_bundle_dir") or ""),
        "candidate_bundle_audit_json": str(identity.get("candidate_bundle_audit_json") or ""),
        "selective_edge_summary_json": str(identity.get("selective_edge_summary_json") or ""),
        "candidate_replay_evidence_manifest_json": candidate_replay_manifest_json,
        "candidate_replay_manifest_decision": str(candidate_replay_manifest.get("decision") or ""),
        "candidate_replay_manifest_identity_ready": bool(candidate_replay_identity.get("ready")),
        "replay_identity_ready": bool(identity.get("replay_identity_ready")),
        "candidate_pretrain_provenance_contract": pretrain_provenance,
        "smoke_dataset_provenance_contract": smoke_dataset_provenance,
        "specialist_set_provenance_contract": specialist_set_provenance,
        "specialist_model_provenance_contract": specialist_model_provenance,
        "bundle_specialist_model_provenance_contract": bundle_specialist_model_provenance,
        "replay_artifact_provenance_contract": replay_artifact_provenance,
        "distillation_artifact_hash_contract": artifact_hash_contract,
        "failures": failures,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry IQL Replay Evidence",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Trades: `{report['n_trades']}`",
        f"- Out dir: `{report['out_dir']}`",
        f"- Promotion/shadow/live allowed: `{report['promotion_shadow_live_allowed']}`",
        "",
        "## Failures",
        "",
    ]
    if report["failures"]:
        lines.extend(f"- {failure}" for failure in report["failures"])
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    trades_path = Path(args.trades_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    distillation_contract_path = Path(args.distillation_contract_json).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    distillation_contract = _read_json(distillation_contract_path)
    identity = _distillation_identity_contract(
        distillation_contract_path=distillation_contract_path,
        distillation_contract=distillation_contract,
    )
    raw = _read_table(trades_path)
    require_year = None if args.require_year <= 0 else int(args.require_year)
    trades, failures = normalize_trades(
        raw,
        policy_id=str(args.policy_id),
        require_year=require_year,
        allow_non_2026=bool(args.allow_non_2026),
    )
    policies = sorted(str(x) for x in trades["policy_id"].unique())
    if bool(args.require_policy_id) and policies != [str(args.policy_id)]:
        failures.append(
            "IQL replay trade log policy_id must match --policy-id exactly: "
            f"expected={[str(args.policy_id)]} observed={policies}"
        )
    failures.extend(identity["failures"])
    metrics, daily, monthly = build_replay_tables(trades)
    iql_transition_audit = audit_iql_transition_trades(trades)
    if bool(args.require_iql_transition_fields):
        failures.extend(iql_transition_audit["failures"])

    best = metrics[metrics["scope"].astype(str).isin(["aggregate", "all", "ALL"])]
    if best.empty:
        failures.append("no aggregate IQL replay metrics were produced")
    else:
        row = best.sort_values("net_sum_bps", ascending=False).iloc[0]
        if int(row.get("n_trades") or 0) <= 0:
            failures.append("aggregate IQL replay metrics have zero trades")

    trades_out = out_dir / "replay_policy_trades.csv"
    metrics_out = out_dir / "replay_policy_metrics.csv"
    daily_out = out_dir / "replay_policy_daily.csv"
    monthly_out = out_dir / "replay_policy_monthly.csv"
    summary_out = out_dir / "summary.json"
    manifest_out = out_dir / "REPLAY_EVIDENCE_MANIFEST.json"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_json = out_dir / f"ENTRY_IQL_REPLAY_EVIDENCE_{timestamp}.json"
    report_md = out_dir / f"ENTRY_IQL_REPLAY_EVIDENCE_{timestamp}.md"

    trades.to_csv(trades_out, index=False)
    metrics.to_csv(metrics_out, index=False)
    daily.to_csv(daily_out, index=False)
    monthly.to_csv(monthly_out, index=False)

    best_row = best.sort_values("net_sum_bps", ascending=False).iloc[0].to_dict() if not best.empty else {}
    report = {
        "schema_version": "entry_iql_replay_evidence_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS" if not failures else "FAIL",
        "trades_path": str(trades_path),
        "out_dir": str(out_dir),
        "distillation_contract_json": str(distillation_contract_path),
        "candidate_bundle_dir": identity["candidate_bundle_dir"],
        "no_xgb_bundle_dir": identity["no_xgb_bundle_dir"],
        "replay_identity_contract": identity,
        "evidence_identity": identity,
        "required_year": require_year,
        "n_trades": int(len(trades)),
        "policies": policies,
        "best_aggregate_row": best_row,
        "trades_csv": str(trades_out),
        "metrics_csv": str(metrics_out),
        "daily_csv": str(daily_out),
        "monthly_csv": str(monthly_out),
        "summary_json": str(summary_out),
        "manifest_json": str(manifest_out),
        "iql_transition_dataset_ready": bool(iql_transition_audit["ready"]),
        "iql_transition_contract": iql_transition_audit,
        "json_path": str(report_json),
        "md_path": str(report_md),
        "trainer_started": False,
        "replay_started": False,
        "adapter_built": False,
        "promotion_shadow_live_allowed": False,
        "failures": failures,
    }
    summary_out.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    manifest_out.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(report_md, report)
    (out_dir / "ENTRY_IQL_REPLAY_EVIDENCE_latest.json").write_text(
        report_json.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_IQL_REPLAY_EVIDENCE_latest.md").write_text(
        report_md.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "failures": failures,
                    "metrics_csv": str(metrics_out),
                    "monthly_csv": str(monthly_out),
                    "json_path": str(report_json),
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
    ap.add_argument("--trades-path", required=True)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--distillation-contract-json", default=str(DEFAULT_DISTILL_CONTRACT_JSON))
    ap.add_argument("--policy-id", default="entry_iql_student")
    ap.add_argument("--require-year", type=int, default=2026)
    ap.add_argument("--allow-non-2026", action="store_true")
    ap.add_argument("--require-policy-id", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--require-iql-transition-fields", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
