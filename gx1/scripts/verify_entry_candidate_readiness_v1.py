#!/usr/bin/env python3
"""Verify Entry readiness for full candidate training after smoke edge evidence.

This gate is intentionally stricter than train-readiness. It does not train. It
requires a real smoke-train bundle audit with edge diagnostics before any full
candidate-training vedtak should be considered.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gx1.scripts.verify_entry_foundation_state_v1 import (
    FOUNDATION_SMOKE_DATASET_DIR,
    REPORTS_ROOT,
    REPO,
    SPECIALIST_AUDIT_LATEST,
)
from gx1.scripts.verify_entry_training_readiness_v1 import (
    DEFAULT_OUT_DIR as TRAIN_READINESS_OUT_DIR,
    EXPECTED_ACTIVE_TRAINING_HEADS,
    EXPECTED_BLOCKED_HEADS,
    SMOKE_BUNDLE_AUDIT_LATEST,
    _artifact_fingerprint_checks,
    _artifact_fingerprints,
    _check,
)
from gx1.scripts.verify_entry_training_readiness_v1 import run as run_train_readiness


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_candidate_readiness_20260628_v1"
REQUIRED_SPECIALIST_GROUPS = (
    "structure_swing_encoder",
    "smc_liquidity_encoder",
    "trend_ema_encoder",
    "vol_compression_encoder",
    "momentum_flow_encoder",
    "session_regime_encoder",
)
REQUIRED_MIN_GATE_ENTROPY = 0.05


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _all_ok(checks: list[dict[str, Any]]) -> bool:
    return all(bool(check.get("ok")) for check in checks)


def _splits(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(split): row
        for split, row in (report.get("splits") or {}).items()
        if isinstance(row, dict)
    }


def _split_names(report: dict[str, Any]) -> list[str]:
    raw = report.get("data_splits")
    if isinstance(raw, list) and raw:
        return [str(x) for x in raw]
    return sorted(_splits(report))


def _all_split_direction_beats_majority(report: dict[str, Any]) -> bool:
    rows = _splits(report)
    names = _split_names(report)
    return bool(names) and all(bool(((rows.get(split) or {}).get("direction") or {}).get("beats_majority_baseline")) for split in names)


def _all_split_bad_path_negative(report: dict[str, Any]) -> bool:
    rows = _splits(report)
    names = _split_names(report)
    values = [
        (((rows.get(split) or {}).get("bad_path") or {}).get("prob_vs_path_quality_spearman"))
        for split in names
    ]
    return bool(values) and all(value is not None and float(value) < 0.0 for value in values)


def _all_split_gate_live(report: dict[str, Any], *, min_active_specialists: int, min_gate_entropy: float) -> bool:
    rows = _splits(report)
    names = _split_names(report)
    if not names:
        return False
    for split in names:
        gate = ((rows.get(split) or {}).get("specialist_gate") or {})
        if not bool(gate.get("finite")):
            return False
        if float(gate.get("row_sum_max_abs_error") or 999.0) > 1e-4:
            return False
        if int(gate.get("active_specialist_count_gt_1pct") or 0) < int(min_active_specialists):
            return False
        entropy_mean = gate.get("entropy_mean")
        if entropy_mean is None or float(entropy_mean) < float(min_gate_entropy):
            return False
    return True


def _all_required_split_gate_weights_live(report: dict[str, Any], *, min_mean_weight: float = 0.01) -> bool:
    rows = _splits(report)
    names = _split_names(report)
    if not names:
        return False
    for split in names:
        gate = ((rows.get(split) or {}).get("specialist_gate") or {})
        mean_weight = gate.get("mean_weight") if isinstance(gate.get("mean_weight"), dict) else {}
        for group in REQUIRED_SPECIALIST_GROUPS:
            if float(mean_weight.get(group) or 0.0) <= float(min_mean_weight):
                return False
    return True


def _specialist_audit_contract_passes(report: dict[str, Any], *, min_active_specialists: int, min_gate_entropy: float) -> bool:
    required = {str(x) for x in report.get("required_training_specialists", []) if str(x)}
    return (
        bool(report.get("require_specialist_fusion"))
        and required == set(REQUIRED_SPECIALIST_GROUPS)
        and int(report.get("min_active_specialists") or 0) >= int(min_active_specialists)
        and float(report.get("min_gate_entropy") or -1.0) >= float(min_gate_entropy)
    )


def _head_contract_passes(report: dict[str, Any]) -> bool:
    contract = report.get("head_contract") if isinstance(report.get("head_contract"), dict) else {}
    active = set(str(x) for x in contract.get("active_training_heads", []) if str(x))
    blocked = set(str(x) for x in contract.get("blocked_heads", []) if str(x))
    return (
        bool(report.get("require_head_contract"))
        and str(contract.get("decision")) == "PASS"
        and not contract.get("failures")
        and active == set(EXPECTED_ACTIVE_TRAINING_HEADS)
        and blocked == set(EXPECTED_BLOCKED_HEADS)
    )


def _bundle_specialist_model_contract_passes(report: dict[str, Any]) -> bool:
    bundle = report.get("bundle_summary") if isinstance(report.get("bundle_summary"), dict) else {}
    contract = (
        report.get("bundle_specialist_model_contract")
        if isinstance(report.get("bundle_specialist_model_contract"), dict)
        else {}
    )
    return (
        bool(bundle.get("specialist_model_contract_declared_valid"))
        and bool(bundle.get("specialist_model_contract_valid"))
        and bool(bundle.get("specialist_model_contract_set_exact"))
        and bool(bundle.get("specialist_model_contract_owned_objectives_match"))
        and bool(bundle.get("specialist_model_contract_support_heads_match"))
        and bool(bundle.get("specialist_model_contract_signal_families_match"))
        and bool(bundle.get("specialist_model_contract_model_roles_match"))
        and str(contract.get("decision")) == "PASS"
        and bool(contract.get("valid"))
        and bool(contract.get("set_exact"))
        and bool(contract.get("owned_objectives_match"))
        and bool(contract.get("support_heads_match"))
        and bool(contract.get("signal_families_match"))
        and bool(contract.get("model_roles_match"))
        and not contract.get("failures")
    )


def _smoke_edge_checks(report: dict[str, Any], *, min_active_specialists: int = 3) -> list[dict[str, Any]]:
    effective_min_active_specialists = max(int(min_active_specialists), len(REQUIRED_SPECIALIST_GROUPS))
    bundle = report.get("bundle_summary") if isinstance(report.get("bundle_summary"), dict) else {}
    split_rows = {split: int((row or {}).get("rows") or 0) for split, row in _splits(report).items()}
    specialist_groups = set(str(x) for x in bundle.get("specialist_groups", []) if str(x))
    bad_path_rho = {
        split: (((row or {}).get("bad_path") or {}).get("prob_vs_path_quality_spearman"))
        for split, row in _splits(report).items()
    }
    direction = {
        split: {
            "accuracy": ((row or {}).get("direction") or {}).get("accuracy"),
            "majority_baseline_accuracy": ((row or {}).get("direction") or {}).get("majority_baseline_accuracy"),
            "beats_majority_baseline": ((row or {}).get("direction") or {}).get("beats_majority_baseline"),
        }
        for split, row in _splits(report).items()
    }
    gate = {
        split: {
            "row_sum_max_abs_error": (((row or {}).get("specialist_gate") or {}).get("row_sum_max_abs_error")),
            "active_specialist_count_gt_1pct": (((row or {}).get("specialist_gate") or {}).get("active_specialist_count_gt_1pct")),
            "entropy_mean": (((row or {}).get("specialist_gate") or {}).get("entropy_mean")),
            "finite": (((row or {}).get("specialist_gate") or {}).get("finite")),
            "mean_weight": (((row or {}).get("specialist_gate") or {}).get("mean_weight")),
        }
        for split, row in _splits(report).items()
    }
    head_contract = report.get("head_contract") if isinstance(report.get("head_contract"), dict) else {}
    pretrain_manifest = (
        report.get("pretrain_manifest_contract")
        if isinstance(report.get("pretrain_manifest_contract"), dict)
        else {}
    )
    active_heads = set(str(x) for x in head_contract.get("active_training_heads", []) if str(x))
    blocked_heads = set(str(x) for x in head_contract.get("blocked_heads", []) if str(x))
    specialist_contract = {
        "require_specialist_fusion": report.get("require_specialist_fusion"),
        "required_training_specialists": report.get("required_training_specialists"),
        "min_active_specialists": report.get("min_active_specialists"),
        "min_gate_entropy": report.get("min_gate_entropy"),
    }
    return [
        _check("smoke bundle audit PASS", str(report.get("decision")) == "PASS", {"failures": report.get("failures")}),
        _check("smoke bundle audit has zero failures", not report.get("failures"), {"failures": report.get("failures")}),
        _check("smoke bundle audit used smoke dataset", str(report.get("dataset_dir")) == str(FOUNDATION_SMOKE_DATASET_DIR)),
        _check("smoke bundle audit is from actual train output, not sanity bundle", not bool(bundle.get("sanity_bundle"))),
        _check("smoke bundle audit was run with require_edge", bool(report.get("require_edge"))),
        _check("smoke bundle audit was run with require_head_contract", bool(report.get("require_head_contract"))),
        _check(
            "smoke bundle audit validated pre-train manifest provenance",
            str(pretrain_manifest.get("decision")) == "PASS"
            and not pretrain_manifest.get("failures")
            and bool(pretrain_manifest.get("feature_objective_coverage_all_present"))
            and bool(pretrain_manifest.get("feature_objective_liveness_all_live"))
            and bool(pretrain_manifest.get("feature_source_field_liveness_all_live"))
            and bool(pretrain_manifest.get("specialist_objective_routing_all_present_and_expected"))
            and bool(pretrain_manifest.get("specialist_input_liveness_all_live"))
            and bool(pretrain_manifest.get("specialist_active_heads_match_target"))
            and bool(pretrain_manifest.get("specialist_blocked_heads_match_target"))
            and bool(pretrain_manifest.get("specialist_model_contract_valid"))
            and bool(pretrain_manifest.get("specialist_model_contract_set_exact"))
            and bool(pretrain_manifest.get("specialist_model_contract_owned_objectives_match"))
            and bool(pretrain_manifest.get("smoke_dataset_audit_provenance_all_artifacts_present"))
            and bool(pretrain_manifest.get("smoke_dataset_audit_provenance_all_artifact_hashes_present"))
            and bool(pretrain_manifest.get("worktree_critical_gate_review_ok")),
            {"pretrain_manifest_contract": pretrain_manifest},
        ),
        _check(
            "smoke bundle head contract PASS",
            _head_contract_passes(report),
            {
                "head_contract": head_contract,
                "expected_active_heads": list(EXPECTED_ACTIVE_TRAINING_HEADS),
                "actual_active_heads": sorted(active_heads),
                "expected_blocked_heads": list(EXPECTED_BLOCKED_HEADS),
                "actual_blocked_heads": sorted(blocked_heads),
            },
        ),
        _check("smoke bundle is seq146", int(bundle.get("seq_input_dim") or 0) == 146 and int(bundle.get("snap_input_dim") or 0) == 146),
        _check("smoke bundle has multi-TF enabled", bool(bundle.get("multi_tf_enabled"))),
        _check("smoke bundle has specialist fusion", bool(bundle.get("specialist_fusion_enabled"))),
        _check(
            "smoke bundle specialist model contract is preserved in bundle metadata",
            _bundle_specialist_model_contract_passes(report),
            {
                "bundle_summary": {
                    "specialist_model_contract_declared_valid": bundle.get("specialist_model_contract_declared_valid"),
                    "specialist_model_contract_valid": bundle.get("specialist_model_contract_valid"),
                    "specialist_model_contract_set_exact": bundle.get("specialist_model_contract_set_exact"),
                    "specialist_model_contract_owned_objectives_match": bundle.get("specialist_model_contract_owned_objectives_match"),
                    "specialist_model_contract_support_heads_match": bundle.get("specialist_model_contract_support_heads_match"),
                    "specialist_model_contract_signal_families_match": bundle.get("specialist_model_contract_signal_families_match"),
                    "specialist_model_contract_model_roles_match": bundle.get("specialist_model_contract_model_roles_match"),
                },
                "bundle_specialist_model_contract": report.get("bundle_specialist_model_contract"),
            },
        ),
        _check(
            "smoke bundle audit was run with specialist-fusion gate contract",
            _specialist_audit_contract_passes(
                report,
                min_active_specialists=effective_min_active_specialists,
                min_gate_entropy=REQUIRED_MIN_GATE_ENTROPY,
            ),
            {
                "required_min_active_specialists": effective_min_active_specialists,
                "required_min_gate_entropy": REQUIRED_MIN_GATE_ENTROPY,
                "specialist_contract": specialist_contract,
            },
        ),
        _check(
            "smoke bundle includes required specialist groups",
            all(group in specialist_groups for group in REQUIRED_SPECIALIST_GROUPS),
            {"specialist_groups": sorted(specialist_groups)},
        ),
        _check(
            "smoke bundle has exact specialist groups",
            specialist_groups == set(REQUIRED_SPECIALIST_GROUPS),
            {
                "expected_specialist_groups": list(REQUIRED_SPECIALIST_GROUPS),
                "actual_specialist_groups": sorted(specialist_groups),
            },
        ),
        _check(
            "smoke audit covered val/test rows",
            split_rows.get("val", 0) > 0 and split_rows.get("test", 0) > 0,
            {"split_rows": split_rows},
        ),
        _check("direction beats majority on all audited splits", _all_split_direction_beats_majority(report), direction),
        _check("bad_path probability ranks worse path quality higher", _all_split_bad_path_negative(report), {"bad_path_rho": bad_path_rho}),
        _check(
            "specialist gate is finite, normalized, non-collapsed, and entropic",
            _all_split_gate_live(
                report,
                min_active_specialists=effective_min_active_specialists,
                min_gate_entropy=REQUIRED_MIN_GATE_ENTROPY,
            ),
            {
                "min_active_specialists": effective_min_active_specialists,
                "min_gate_entropy": REQUIRED_MIN_GATE_ENTROPY,
                "gate": gate,
            },
        ),
        _check(
            "each required specialist has non-collapsed gate weight",
            _all_required_split_gate_weights_live(report, min_mean_weight=0.01),
            {"min_mean_weight": 0.01, "gate": gate},
        ),
    ]


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Candidate Readiness",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Candidate training allowed with explicit vedtak: `{report['candidate_training_allowed_with_explicit_vedtak']}`",
        f"- Promotion/shadow/live allowed: `{report['promotion_shadow_live_allowed']}`",
        f"- Next required gate: `{report['next_required_gate']}`",
        "",
        "## Gates",
        "",
    ]
    for gate in report["gates"]:
        lines.append(f"- `{gate['name']}`: {gate['decision']} ({gate['passed']}/{gate['total']} checks)")
    lines.extend(["", "## Failures", ""])
    if report["failures"]:
        for failure in report["failures"]:
            lines.append(f"- `{failure['gate']}`: {failure['check']}")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke_audit_path = Path(args.smoke_bundle_audit_json).expanduser().resolve()

    train_readiness = run_train_readiness(
        argparse.Namespace(
            audit_doc=str(args.audit_doc),
            out_dir=str(TRAIN_READINESS_OUT_DIR),
            fail_on_not_ready=False,
            quiet=True,
        )
    )
    smoke_audit = _read_json(smoke_audit_path)
    artifacts = {
        "train_readiness": str(train_readiness.get("json_path") or (TRAIN_READINESS_OUT_DIR / "ENTRY_TRAINING_READINESS_latest.json")),
        "smoke_bundle_audit": str(smoke_audit_path),
        "specialist_audit": str(SPECIALIST_AUDIT_LATEST),
    }
    artifact_fingerprints = _artifact_fingerprints(artifacts)
    gate_checks = {
        "train_readiness": [
            _check(
                "foundation train-readiness is green",
                str(train_readiness.get("decision")) == "READY_FOR_VEDTAK_SMOKE_TRAIN",
                {"decision": train_readiness.get("decision"), "failures": train_readiness.get("failures")},
            ),
            _check("train-readiness still blocks candidate training", bool(train_readiness.get("candidate_training_allowed")) is False),
            _check("train-readiness still blocks promotion/shadow/live", bool(train_readiness.get("promotion_shadow_live_allowed")) is False),
        ],
        "smoke_edge_audit": _smoke_edge_checks(smoke_audit, min_active_specialists=int(args.min_active_specialists)),
        "promotion_guard": [
            _check("candidate gate never promotes", True),
            _check("candidate gate never starts shadow/live", True),
        ],
        "artifact_provenance": _artifact_fingerprint_checks(artifact_fingerprints),
    }
    gates = []
    for name, checks in gate_checks.items():
        passed = sum(1 for check in checks if check["ok"])
        gates.append(
            {
                "name": name,
                "decision": "PASS" if _all_ok(checks) else "FAIL",
                "passed": int(passed),
                "total": int(len(checks)),
                "checks": checks,
            }
        )
    failures = [
        {"gate": gate["name"], "check": check["name"], "details": check.get("details") or {}}
        for gate in gates
        for check in gate["checks"]
        if not check["ok"]
    ]
    ready = not failures
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "entry_candidate_readiness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "READY_FOR_CANDIDATE_TRAINING_VEDTAK" if ready else "NOT_READY_FOR_CANDIDATE_TRAINING",
        "candidate_training_allowed_with_explicit_vedtak": bool(ready),
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            "candidate specialist-fusion train wrapper with explicit vedtak and post-train replay gate"
            if ready
            else "run scripts/entry_next_edge_control.sh smoke-train --vedtak <id> --require-edge-audit"
        ),
        "artifacts": artifacts,
        "artifact_fingerprints": artifact_fingerprints,
        "smoke_bundle_audit_json": str(smoke_audit_path),
        "train_readiness_json": train_readiness.get("json_path"),
        "gates": gates,
        "failures": failures,
    }
    json_path = out_dir / f"ENTRY_CANDIDATE_READINESS_{timestamp}.json"
    md_path = out_dir / f"ENTRY_CANDIDATE_READINESS_{timestamp}.md"
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    latest_json = out_dir / "ENTRY_CANDIDATE_READINESS_latest.json"
    latest_md = out_dir / "ENTRY_CANDIDATE_READINESS_latest.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "failures": report["failures"],
                    "json_path": report["json_path"],
                    "md_path": report["md_path"],
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
    ap.add_argument("--audit-doc", default=str(REPO / "docs/ENTRY_FOUNDATION_AUDIT_20260628.md"))
    ap.add_argument("--smoke-bundle-audit-json", default=str(SMOKE_BUNDLE_AUDIT_LATEST))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--min-active-specialists", type=int, default=len(REQUIRED_SPECIALIST_GROUPS))
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
