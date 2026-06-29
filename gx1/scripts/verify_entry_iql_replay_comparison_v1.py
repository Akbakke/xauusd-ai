#!/usr/bin/env python3
"""Verify post-distillation Entry-IQL replay comparison.

This gate compares an offline IQL-student replay against the candidate replay
baseline. It does not train IQL, run replay, build adapters, promote, shadow or
trade. Passing this gate only opens an explicit promotion-review discussion.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.materialize_entry_iql_distillation_contract_v1 import DEFAULT_OUT_DIR as DISTILL_CONTRACT_OUT_DIR
from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT
from gx1.scripts.verify_entry_replay_readiness_v1 import DEFAULT_REPLAY_DIR as CANDIDATE_REPLAY_DIR
from gx1.scripts.verify_entry_replay_readiness_v1 import _best_replay_row
from gx1.scripts.verify_entry_training_readiness_v1 import _check


DEFAULT_DISTILL_CONTRACT_JSON = DISTILL_CONTRACT_OUT_DIR / "ENTRY_IQL_DISTILLATION_CONTRACT_latest.json"
DEFAULT_IQL_REPLAY_DIR = REPORTS_ROOT / "entry_iql_distillation_replay_20260628_v1"
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_iql_replay_comparison_20260628_v1"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        out = float(obj)
        return out if np.isfinite(out) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _row_value(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not row:
        return float(default)
    value = _float_or_none(row.get(key))
    return float(default) if value is None else float(value)


def _identity_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    for key in ("replay_identity_contract", "evidence_identity"):
        value = manifest.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _identity_bundle_dir(identity: dict[str, Any]) -> str:
    return str(
        identity.get("candidate_bundle_dir")
        or identity.get("replay_identity_candidate_bundle_dir")
        or ""
    )


def _identity_ready(identity: dict[str, Any]) -> bool:
    return bool(identity.get("ready")) or bool(identity.get("replay_identity_ready"))


def _distillation_hash_ready(identity: dict[str, Any], manifest: dict[str, Any]) -> bool:
    for source in (identity, manifest):
        contract = source.get("distillation_artifact_hash_contract") if isinstance(source, dict) else None
        if isinstance(contract, dict):
            return bool(contract.get("ready"))
    return False


def _normal_path_string(value: Any) -> str:
    raw = str(value or "").strip()
    return str(Path(raw).expanduser().resolve()) if raw else ""


def _no_negative_months(monthly: pd.DataFrame) -> bool:
    return not monthly.empty and "net_sum_bps" in monthly.columns and bool(
        (pd.to_numeric(monthly["net_sum_bps"], errors="coerce") > 0.0).all()
    )


def build_comparison_checks(
    *,
    distill_contract: dict[str, Any],
    candidate_replay_manifest: dict[str, Any],
    iql_replay_manifest: dict[str, Any],
    distillation_contract_json: str,
    candidate_replay_manifest_json: str,
    iql_replay_manifest_json: str,
    candidate_metrics: pd.DataFrame,
    candidate_monthly: pd.DataFrame,
    iql_metrics: pd.DataFrame,
    iql_monthly: pd.DataFrame,
    min_net_lift_bps: float,
    min_iql_profit_factor: float,
    min_profit_factor_lift: float,
    max_drawdown_worsening_bps: float,
    max_loss_worsening_bps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_row = _best_replay_row(candidate_metrics)
    iql_row = _best_replay_row(iql_metrics)
    evidence_identity = (
        distill_contract.get("evidence_identity")
        if isinstance(distill_contract.get("evidence_identity"), dict)
        else {}
    )
    pretrain_provenance = (
        distill_contract.get("candidate_pretrain_provenance_contract")
        if isinstance(distill_contract.get("candidate_pretrain_provenance_contract"), dict)
        else {}
    )
    smoke_dataset_provenance = (
        distill_contract.get("smoke_dataset_provenance_contract")
        if isinstance(distill_contract.get("smoke_dataset_provenance_contract"), dict)
        else {}
    )
    specialist_set_provenance = (
        distill_contract.get("specialist_set_provenance_contract")
        if isinstance(distill_contract.get("specialist_set_provenance_contract"), dict)
        else {}
    )
    specialist_model_provenance = (
        distill_contract.get("specialist_model_provenance_contract")
        if isinstance(distill_contract.get("specialist_model_provenance_contract"), dict)
        else {}
    )
    bundle_specialist_model_provenance = (
        distill_contract.get("bundle_specialist_model_provenance_contract")
        if isinstance(distill_contract.get("bundle_specialist_model_provenance_contract"), dict)
        else {}
    )
    replay_artifact_provenance = (
        distill_contract.get("replay_artifact_provenance_contract")
        if isinstance(distill_contract.get("replay_artifact_provenance_contract"), dict)
        else {}
    )
    contract_bundle_dir = _identity_bundle_dir(evidence_identity)
    contract_selective_bundle_dir = str(evidence_identity.get("selective_edge_bundle_dir") or "")
    contract_replay_bundle_dir = str(evidence_identity.get("replay_identity_candidate_bundle_dir") or "")
    expected_candidate_manifest_json = _normal_path_string(evidence_identity.get("replay_evidence_manifest_json"))
    observed_candidate_manifest_json = _normal_path_string(candidate_replay_manifest_json)
    expected_distillation_contract_json = _normal_path_string(distillation_contract_json)
    candidate_manifest_identity = _identity_from_manifest(candidate_replay_manifest)
    iql_manifest_identity = _identity_from_manifest(iql_replay_manifest)
    iql_distillation_hash_contract = (
        iql_manifest_identity.get("distillation_artifact_hash_contract")
        if isinstance(iql_manifest_identity.get("distillation_artifact_hash_contract"), dict)
        else iql_replay_manifest.get("distillation_artifact_hash_contract")
        if isinstance(iql_replay_manifest.get("distillation_artifact_hash_contract"), dict)
        else {}
    )
    candidate_manifest_bundle_dir = _identity_bundle_dir(candidate_manifest_identity)
    iql_manifest_bundle_dir = _identity_bundle_dir(iql_manifest_identity)
    iql_manifest_distillation_contract_json = _normal_path_string(
        iql_replay_manifest.get("distillation_contract_json")
        or iql_manifest_identity.get("distillation_contract_json")
    )
    iql_manifest_candidate_replay_json = _normal_path_string(
        iql_manifest_identity.get("candidate_replay_evidence_manifest_json")
        or iql_manifest_identity.get("replay_evidence_manifest_json")
    )
    cand_net = _row_value(candidate_row, "net_sum_bps")
    iql_net = _row_value(iql_row, "net_sum_bps")
    cand_pf = _row_value(candidate_row, "profit_factor")
    iql_pf = _row_value(iql_row, "profit_factor")
    cand_dd = abs(_row_value(candidate_row, "max_drawdown_bps"))
    iql_dd = abs(_row_value(iql_row, "max_drawdown_bps"))
    cand_loss = _row_value(candidate_row, "max_loss_bps")
    iql_loss = _row_value(iql_row, "max_loss_bps")
    details = {
        "candidate_best_row": candidate_row or {},
        "iql_best_row": iql_row or {},
        "net_lift_bps": iql_net - cand_net,
        "profit_factor_lift": iql_pf - cand_pf,
        "drawdown_delta_bps": iql_dd - cand_dd,
        "max_loss_delta_bps": iql_loss - cand_loss,
        "evidence_identity": evidence_identity,
        "candidate_replay_identity": candidate_manifest_identity,
        "iql_replay_identity": iql_manifest_identity,
        "candidate_replay_manifest_json": observed_candidate_manifest_json,
        "iql_replay_manifest_json": _normal_path_string(iql_replay_manifest_json),
        "distillation_contract_json": expected_distillation_contract_json,
        "iql_distillation_artifact_hash_contract": iql_distillation_hash_contract,
        "candidate_pretrain_provenance_contract": pretrain_provenance,
        "smoke_dataset_provenance_contract": smoke_dataset_provenance,
        "specialist_set_provenance_contract": specialist_set_provenance,
        "specialist_model_provenance_contract": specialist_model_provenance,
        "bundle_specialist_model_provenance_contract": bundle_specialist_model_provenance,
        "replay_artifact_provenance_contract": replay_artifact_provenance,
    }
    checks = [
        _check(
            "IQL distillation contract is ready",
            str(distill_contract.get("decision")) == "ENTRY_IQL_DISTILLATION_CONTRACT_READY",
            {"decision": distill_contract.get("decision")},
        ),
        _check(
            "IQL distillation contract still blocks promotion/shadow/live",
            bool(distill_contract.get("promotion_shadow_live_allowed")) is False,
        ),
        _check(
            "IQL distillation contract preserved candidate pretrain provenance",
            bool(pretrain_provenance.get("ok")),
            {"candidate_pretrain_provenance_contract": pretrain_provenance},
        ),
        _check(
            "IQL distillation contract preserved smoke dataset audit provenance",
            bool(smoke_dataset_provenance.get("ok")),
            {"smoke_dataset_provenance_contract": smoke_dataset_provenance},
        ),
        _check(
            "IQL distillation contract preserved exact specialist set provenance",
            bool(specialist_set_provenance.get("ok")),
            {"specialist_set_provenance_contract": specialist_set_provenance},
        ),
        _check(
            "IQL distillation contract preserved specialist model contract provenance",
            bool(specialist_model_provenance.get("ok")),
            {"specialist_model_provenance_contract": specialist_model_provenance},
        ),
        _check(
            "IQL distillation contract preserved candidate bundle specialist model contract provenance",
            bool(bundle_specialist_model_provenance.get("ok")),
            {"bundle_specialist_model_provenance_contract": bundle_specialist_model_provenance},
        ),
        _check(
            "IQL distillation contract preserved replay artifact provenance",
            bool(replay_artifact_provenance.get("ok")),
            {"replay_artifact_provenance_contract": replay_artifact_provenance},
        ),
        _check(
            "IQL distillation contract carries evidence identity",
            bool(evidence_identity),
            {"evidence_identity": evidence_identity},
        ),
        _check(
            "IQL distillation evidence identity has candidate bundle dir",
            bool(contract_bundle_dir),
            {"evidence_identity": evidence_identity},
        ),
        _check(
            "IQL distillation evidence identity is internally aligned",
            bool(contract_bundle_dir)
            and contract_selective_bundle_dir == contract_bundle_dir
            and contract_replay_bundle_dir == contract_bundle_dir
            and bool(evidence_identity.get("replay_identity_ready")),
            {
                "candidate_bundle_dir": contract_bundle_dir,
                "selective_edge_bundle_dir": contract_selective_bundle_dir,
                "replay_identity_candidate_bundle_dir": contract_replay_bundle_dir,
                "replay_identity_ready": evidence_identity.get("replay_identity_ready"),
            },
        ),
        _check(
            "candidate replay manifest exists",
            bool(candidate_replay_manifest),
        ),
        _check(
            "candidate replay manifest is PASS",
            str(candidate_replay_manifest.get("decision")) == "PASS",
            {"decision": candidate_replay_manifest.get("decision")},
        ),
        _check(
            "candidate replay manifest evidence identity is ready",
            _identity_ready(candidate_manifest_identity),
            {"candidate_replay_identity": candidate_manifest_identity},
        ),
        _check(
            "candidate replay manifest evidence identity matches distillation contract",
            bool(contract_bundle_dir) and candidate_manifest_bundle_dir == contract_bundle_dir,
            {
                "contract_candidate_bundle_dir": contract_bundle_dir,
                "candidate_manifest_bundle_dir": candidate_manifest_bundle_dir,
            },
        ),
        _check(
            "candidate replay manifest path matches distillation evidence identity",
            bool(expected_candidate_manifest_json)
            and observed_candidate_manifest_json == expected_candidate_manifest_json,
            {
                "expected_candidate_replay_manifest_json": expected_candidate_manifest_json,
                "observed_candidate_replay_manifest_json": observed_candidate_manifest_json,
            },
        ),
        _check(
            "IQL replay manifest exists",
            bool(iql_replay_manifest),
        ),
        _check(
            "IQL replay manifest is PASS",
            str(iql_replay_manifest.get("decision")) == "PASS",
            {"decision": iql_replay_manifest.get("decision")},
        ),
        _check(
            "IQL replay manifest evidence identity is ready",
            _identity_ready(iql_manifest_identity),
            {"iql_replay_identity": iql_manifest_identity},
        ),
        _check(
            "IQL replay manifest validated distillation artifact hashes",
            _distillation_hash_ready(iql_manifest_identity, iql_replay_manifest),
            {"distillation_artifact_hash_contract": iql_distillation_hash_contract},
        ),
        _check(
            "IQL replay manifest evidence identity matches distillation contract",
            bool(contract_bundle_dir) and iql_manifest_bundle_dir == contract_bundle_dir,
            {
                "contract_candidate_bundle_dir": contract_bundle_dir,
                "iql_manifest_bundle_dir": iql_manifest_bundle_dir,
            },
        ),
        _check(
            "IQL replay manifest distillation contract matches comparison input",
            bool(expected_distillation_contract_json)
            and iql_manifest_distillation_contract_json == expected_distillation_contract_json,
            {
                "expected_distillation_contract_json": expected_distillation_contract_json,
                "iql_manifest_distillation_contract_json": iql_manifest_distillation_contract_json,
            },
        ),
        _check(
            "IQL replay manifest references candidate replay evidence from distillation contract",
            bool(expected_candidate_manifest_json)
            and iql_manifest_candidate_replay_json == expected_candidate_manifest_json,
            {
                "expected_candidate_replay_manifest_json": expected_candidate_manifest_json,
                "iql_manifest_candidate_replay_json": iql_manifest_candidate_replay_json,
            },
        ),
        _check("candidate replay metrics have rows", not candidate_metrics.empty, {"rows": int(len(candidate_metrics))}),
        _check("candidate replay monthly has rows", not candidate_monthly.empty, {"rows": int(len(candidate_monthly))}),
        _check("IQL replay metrics have rows", not iql_metrics.empty, {"rows": int(len(iql_metrics))}),
        _check("IQL replay monthly has rows", not iql_monthly.empty, {"rows": int(len(iql_monthly))}),
        _check(
            "candidate replay best row has trades",
            candidate_row is not None and int(candidate_row.get("n_trades") or 0) > 0,
            {"candidate_best_row": candidate_row or {}},
        ),
        _check(
            "IQL replay best row has trades",
            iql_row is not None and int(iql_row.get("n_trades") or 0) > 0,
            {"iql_best_row": iql_row or {}},
        ),
        _check(
            "IQL replay net sum beats candidate",
            iql_row is not None and candidate_row is not None and (iql_net - cand_net) > float(min_net_lift_bps),
            {"threshold": min_net_lift_bps, **details},
        ),
        _check(
            "IQL replay profit factor passes absolute threshold",
            iql_row is not None and iql_pf >= float(min_iql_profit_factor),
            {"threshold": min_iql_profit_factor, "iql_profit_factor": iql_pf},
        ),
        _check(
            "IQL replay profit factor does not degrade vs candidate",
            iql_row is not None and candidate_row is not None and (iql_pf - cand_pf) >= float(min_profit_factor_lift),
            {"threshold": min_profit_factor_lift, **details},
        ),
        _check(
            "IQL replay drawdown does not worsen beyond bound",
            iql_row is not None and candidate_row is not None and (iql_dd - cand_dd) <= float(max_drawdown_worsening_bps),
            {"max_worsening_bps": max_drawdown_worsening_bps, **details},
        ),
        _check(
            "IQL replay max loss does not worsen beyond bound",
            iql_row is not None and candidate_row is not None and (cand_loss - iql_loss) <= float(max_loss_worsening_bps),
            {"max_worsening_bps": max_loss_worsening_bps, **details},
        ),
        _check("IQL replay has no negative months", _no_negative_months(iql_monthly), {"rows": int(len(iql_monthly))}),
        _check("gate never trains IQL", True),
        _check("gate never builds adapters", True),
        _check("gate never promotes, shadows, or starts live", True),
    ]
    return checks, details


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry IQL Replay Comparison",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Promotion review allowed with explicit vedtak: `{report['promotion_review_allowed_with_explicit_vedtak']}`",
        f"- Promotion/shadow/live allowed: `{report['promotion_shadow_live_allowed']}`",
        "",
        "## Comparison",
        "",
        f"- Net lift bps: `{report['comparison'].get('net_lift_bps')}`",
        f"- Profit factor lift: `{report['comparison'].get('profit_factor_lift')}`",
        f"- Drawdown delta bps: `{report['comparison'].get('drawdown_delta_bps')}`",
        "",
        "## Failures",
        "",
    ]
    if report["failures"]:
        for failure in report["failures"]:
            lines.append(f"- `{failure['check']}`")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    distill_contract_path = Path(args.distillation_contract_json).expanduser().resolve()
    candidate_replay_dir = Path(args.candidate_replay_dir).expanduser().resolve()
    iql_replay_dir = Path(args.iql_replay_dir).expanduser().resolve()

    distill_contract = _read_json_or_empty(distill_contract_path)
    candidate_replay_manifest_path = candidate_replay_dir / "REPLAY_EVIDENCE_MANIFEST.json"
    iql_replay_manifest_path = iql_replay_dir / "REPLAY_EVIDENCE_MANIFEST.json"
    candidate_replay_manifest = _read_json_or_empty(candidate_replay_manifest_path)
    iql_replay_manifest = _read_json_or_empty(iql_replay_manifest_path)
    candidate_metrics = _read_csv_or_empty(candidate_replay_dir / "replay_policy_metrics.csv")
    candidate_monthly = _read_csv_or_empty(candidate_replay_dir / "replay_policy_monthly.csv")
    iql_metrics = _read_csv_or_empty(iql_replay_dir / "replay_policy_metrics.csv")
    iql_monthly = _read_csv_or_empty(iql_replay_dir / "replay_policy_monthly.csv")

    checks, comparison = build_comparison_checks(
        distill_contract=distill_contract,
        candidate_replay_manifest=candidate_replay_manifest,
        iql_replay_manifest=iql_replay_manifest,
        distillation_contract_json=str(distill_contract_path),
        candidate_replay_manifest_json=str(candidate_replay_manifest_path),
        iql_replay_manifest_json=str(iql_replay_manifest_path),
        candidate_metrics=candidate_metrics,
        candidate_monthly=candidate_monthly,
        iql_metrics=iql_metrics,
        iql_monthly=iql_monthly,
        min_net_lift_bps=float(args.min_net_lift_bps),
        min_iql_profit_factor=float(args.min_iql_profit_factor),
        min_profit_factor_lift=float(args.min_profit_factor_lift),
        max_drawdown_worsening_bps=float(args.max_drawdown_worsening_bps),
        max_loss_worsening_bps=float(args.max_loss_worsening_bps),
    )
    failures = [{"check": check["name"], "details": check.get("details") or {}} for check in checks if not check["ok"]]
    ready = not failures
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "entry_iql_replay_comparison_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "READY_FOR_PROMOTION_REVIEW_VEDTAK" if ready else "NOT_READY_FOR_PROMOTION_REVIEW",
        "promotion_review_allowed_with_explicit_vedtak": bool(ready),
        "promotion_shadow_live_allowed": False,
        "distillation_contract_json": str(distill_contract_path),
        "candidate_replay_dir": str(candidate_replay_dir),
        "iql_replay_dir": str(iql_replay_dir),
        "candidate_replay_manifest_json": str(candidate_replay_manifest_path),
        "iql_replay_manifest_json": str(iql_replay_manifest_path),
        "evidence_identity": comparison.get("evidence_identity") or {},
        "comparison": comparison,
        "checks": checks,
        "failures": failures,
        "next_required_gate": (
            "explicit promotion-review gate; shadow/live remain blocked"
            if ready
            else "run IQL distillation research and materialize IQL replay evidence before promotion review"
        ),
    }
    json_path = out_dir / f"ENTRY_IQL_REPLAY_COMPARISON_{timestamp}.json"
    md_path = out_dir / f"ENTRY_IQL_REPLAY_COMPARISON_{timestamp}.md"
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_IQL_REPLAY_COMPARISON_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_IQL_REPLAY_COMPARISON_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "failures": failures,
                    "json_path": str(json_path),
                    "next_required_gate": report["next_required_gate"],
                },
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        )
    if args.fail_on_not_ready and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distillation-contract-json", default=str(DEFAULT_DISTILL_CONTRACT_JSON))
    ap.add_argument("--candidate-replay-dir", default=str(CANDIDATE_REPLAY_DIR))
    ap.add_argument("--iql-replay-dir", default=str(DEFAULT_IQL_REPLAY_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--min-net-lift-bps", type=float, default=0.0)
    ap.add_argument("--min-iql-profit-factor", type=float, default=1.05)
    ap.add_argument("--min-profit-factor-lift", type=float, default=0.0)
    ap.add_argument("--max-drawdown-worsening-bps", type=float, default=0.0)
    ap.add_argument("--max-loss-worsening-bps", type=float, default=0.0)
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
