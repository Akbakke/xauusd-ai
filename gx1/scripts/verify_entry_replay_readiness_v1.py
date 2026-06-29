#!/usr/bin/env python3
"""Verify Entry candidate replay and IQL-distillation readiness.

This gate does not run replay, train IQL, promote, shadow, or trade. It checks
that a post-candidate specialist-fusion bundle has selective-edge evidence and
offline replay evidence before IQL distillation can be considered.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.materialize_entry_candidate_replay_evidence_v1 import IQL_TRANSITION_REQUIRED_COLUMNS
from gx1.scripts.verify_entry_candidate_readiness_v1 import (
    DEFAULT_OUT_DIR as CANDIDATE_READINESS_OUT_DIR,
    _bundle_specialist_model_contract_passes,
    REQUIRED_MIN_GATE_ENTROPY,
    REQUIRED_SPECIALIST_GROUPS,
)
from gx1.scripts.verify_entry_foundation_state_v1 import FOUNDATION_DATASET_DIR, REPORTS_ROOT, REPO
from gx1.scripts.verify_entry_training_readiness_v1 import _check
from gx1.scripts.verify_entry_training_readiness_v1 import (
    EXPECTED_ACTIVE_TRAINING_HEADS,
    EXPECTED_BLOCKED_HEADS,
    _artifact_fingerprint_checks,
    _artifact_fingerprints,
)


CANDIDATE_READINESS_LATEST = CANDIDATE_READINESS_OUT_DIR / "ENTRY_CANDIDATE_READINESS_latest.json"
DEFAULT_SELECTIVE_EDGE_DIR = REPORTS_ROOT / "entry_candidate_selective_edge_20260628_v1"
DEFAULT_REPLAY_DIR = REPORTS_ROOT / "entry_candidate_replay_20260628_v1"
DEFAULT_CANDIDATE_BUNDLE_AUDIT = (
    REPORTS_ROOT / "entry_candidate_bundle_audit_20260628_v1/ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_latest.json"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_replay_readiness_20260628_v1"


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


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _all_ok(checks: list[dict[str, Any]]) -> bool:
    return all(bool(check.get("ok")) for check in checks)


def _same_resolved_path(actual: Any, expected: Path) -> bool:
    if actual in (None, ""):
        return False
    try:
        return Path(str(actual)).resolve(strict=False) == expected.resolve(strict=False)
    except (OSError, RuntimeError):
        return False


def _model_summaries(summary: dict[str, Any], model_name: str) -> list[dict[str, Any]]:
    return [
        row
        for row in summary.get("summaries", [])
        if isinstance(row, dict) and str(row.get("model")) == str(model_name)
    ]


def _summary_by_split(summary: dict[str, Any], model_name: str) -> dict[str, dict[str, Any]]:
    return {str(row.get("split")): row for row in _model_summaries(summary, model_name)}


def _split_metric(summary: dict[str, Any], model_name: str, key: str) -> dict[str, Any]:
    return {split: row.get(key) for split, row in _summary_by_split(summary, model_name).items()}


def _all_split_metric_gt(summary: dict[str, Any], model_name: str, key: str, threshold: float) -> bool:
    values = _split_metric(summary, model_name, key)
    required = [split for split in ("val", "test") if split in values]
    return len(required) == 2 and all(values[split] is not None and float(values[split]) > float(threshold) for split in required)


def _selective_edge_checks(
    summary: dict[str, Any],
    metrics: pd.DataFrame,
    *,
    model_name: str,
    min_top5_mean_pnl_bps: float,
    min_top10_mean_pnl_bps: float,
    require_no_xgb_ablation: bool,
    expected_bundle_dir: str | None = None,
) -> list[dict[str, Any]]:
    splits = set(str(x) for x in summary.get("splits", []))
    models = {str(row.get("model")) for row in summary.get("summaries", []) if isinstance(row, dict)}
    no_xgb_model = f"{model_name}_no_xgb"
    no_xgb_bundle_dir = str(summary.get("no_xgb_bundle_dir") or "")
    summary_bundle_dir = str(summary.get("bundle_dir") or "")
    no_xgb_ablation = summary.get("no_xgb_ablation") if isinstance(summary.get("no_xgb_ablation"), dict) else {}
    no_xgb_mode = str(no_xgb_ablation.get("mode") or "")
    no_xgb_neutralizes_bridge = bool(no_xgb_ablation.get("neutralize_signal_bridge"))
    no_xgb_neutralized_fields = [str(x) for x in no_xgb_ablation.get("neutralized_fields", [])]
    no_xgb_neutral_values = no_xgb_ablation.get("neutral_values", [])
    no_xgb_same_bundle = bool(no_xgb_bundle_dir and summary_bundle_dir and no_xgb_bundle_dir == summary_bundle_dir)
    no_xgb_provenance_ok = True
    if require_no_xgb_ablation:
        if no_xgb_mode == "neutralize_signal_bridge":
            no_xgb_provenance_ok = (
                no_xgb_same_bundle
                and no_xgb_neutralizes_bridge
                and len(no_xgb_neutralized_fields) >= 7
                and len(no_xgb_neutral_values) >= 7
            )
        elif no_xgb_mode == "bundle":
            no_xgb_provenance_ok = bool(no_xgb_bundle_dir) and not no_xgb_same_bundle
        else:
            no_xgb_provenance_ok = False
    candidate_by_split = _summary_by_split(summary, model_name)
    top5 = _split_metric(summary, model_name, "top5_all_mean_pnl_bps")
    top10 = _split_metric(summary, model_name, "top10_all_mean_pnl_bps")
    required_columns = {
        "split",
        "model",
        "scope",
        "top_frac",
        "group",
        "n",
        "mean_pnl_bps",
        "win_rate",
        "direction_precision",
    }
    metric_columns = set(str(c) for c in metrics.columns)
    slice_rows = pd.DataFrame()
    if not metrics.empty and required_columns.issubset(metric_columns):
        slice_rows = metrics[
            (metrics["model"].astype(str) == str(model_name))
            & (metrics["scope"].astype(str) == "top_score")
            & (metrics["group"].astype(str).str.startswith("session="))
        ]
    return [
        _check("selective-edge summary PASS", str(summary.get("decision")) == "PASS", {"failures": summary.get("failures")}),
        _check("selective-edge summary has zero failures", not summary.get("failures"), {"failures": summary.get("failures")}),
        _check("selective-edge summary uses foundation dataset", _same_resolved_path(summary.get("dataset_dir"), FOUNDATION_DATASET_DIR)),
        _check(
            "selective-edge summary matches candidate bundle audit bundle",
            True if expected_bundle_dir is None else str(summary.get("bundle_dir")) == str(expected_bundle_dir),
            {"expected_bundle_dir": expected_bundle_dir, "summary_bundle_dir": summary.get("bundle_dir")},
        ),
        _check("selective-edge summary has val/test", {"val", "test"}.issubset(splits), {"splits": sorted(splits)}),
        _check("selective-edge summary includes candidate model", model_name in models, {"models": sorted(models)}),
        _check(
            "selective-edge summary includes no-XGB ablation",
            (no_xgb_model in models) if require_no_xgb_ablation else True,
            {"required": require_no_xgb_ablation, "models": sorted(models)},
        ),
        _check(
            "selective-edge summary records no-XGB bundle dir",
            bool(no_xgb_bundle_dir) if require_no_xgb_ablation else True,
            {"required": require_no_xgb_ablation, "no_xgb_bundle_dir": no_xgb_bundle_dir},
        ),
        _check(
            "selective-edge no-XGB ablation provenance is explicit",
            no_xgb_provenance_ok,
            {
                "required": require_no_xgb_ablation,
                "mode": no_xgb_mode,
                "neutralize_signal_bridge": no_xgb_neutralizes_bridge,
                "same_bundle_as_candidate": no_xgb_same_bundle,
                "neutralized_fields": no_xgb_neutralized_fields,
            },
        ),
        _check(
            "candidate selective-edge has val/test summaries",
            {"val", "test"}.issubset(set(candidate_by_split)),
            {"candidate_splits": sorted(candidate_by_split)},
        ),
        _check(
            "candidate top5 mean pnl is positive on val/test",
            _all_split_metric_gt(summary, model_name, "top5_all_mean_pnl_bps", min_top5_mean_pnl_bps),
            {"threshold": min_top5_mean_pnl_bps, "top5_all_mean_pnl_bps": top5},
        ),
        _check(
            "candidate top10 mean pnl is positive on val/test",
            _all_split_metric_gt(summary, model_name, "top10_all_mean_pnl_bps", min_top10_mean_pnl_bps),
            {"threshold": min_top10_mean_pnl_bps, "top10_all_mean_pnl_bps": top10},
        ),
        _check("selective-edge metrics CSV exists and has rows", not metrics.empty, {"rows": int(len(metrics))}),
        _check(
            "selective-edge metrics CSV has required columns",
            required_columns.issubset(metric_columns),
            {"missing_columns": sorted(required_columns - metric_columns)},
        ),
        _check(
            "selective-edge metrics include session slices",
            len(slice_rows) > 0,
            {"session_slice_rows": int(len(slice_rows))},
        ),
    ]


def _candidate_bundle_audit_checks(path: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    exists = path.exists()
    bundle = report.get("bundle_summary") if isinstance(report.get("bundle_summary"), dict) else {}
    head_contract = report.get("head_contract") if isinstance(report.get("head_contract"), dict) else {}
    pretrain_manifest = (
        report.get("pretrain_manifest_contract")
        if isinstance(report.get("pretrain_manifest_contract"), dict)
        else {}
    )
    active_heads = set(str(x) for x in head_contract.get("active_training_heads", []) if str(x))
    blocked_heads = set(str(x) for x in head_contract.get("blocked_heads", []) if str(x))
    splits = {
        str(split): row
        for split, row in (report.get("splits") or {}).items()
        if isinstance(row, dict)
    }
    split_rows = {split: int((row or {}).get("rows") or 0) for split, row in splits.items()}
    specialist_groups = set(str(x) for x in bundle.get("specialist_groups", []) if str(x))
    required_specialists = {str(x) for x in report.get("required_training_specialists", []) if str(x)}
    required_gate_live = True
    for row in splits.values():
        gate = (row or {}).get("specialist_gate") if isinstance(row, dict) else {}
        mean_weight = gate.get("mean_weight") if isinstance(gate, dict) and isinstance(gate.get("mean_weight"), dict) else {}
        for group in REQUIRED_SPECIALIST_GROUPS:
            if float(mean_weight.get(group) or 0.0) <= 0.01:
                required_gate_live = False
    return [
        _check("candidate bundle audit exists", exists, {"path": str(path)}),
        _check("candidate bundle audit PASS", exists and str(report.get("decision")) == "PASS", {"failures": report.get("failures")}),
        _check("candidate bundle audit has zero failures", exists and not report.get("failures"), {"failures": report.get("failures")}),
        _check("candidate bundle audit used foundation dataset", exists and _same_resolved_path(report.get("dataset_dir"), FOUNDATION_DATASET_DIR)),
        _check("candidate bundle audit is from actual train output, not sanity bundle", exists and not bool(bundle.get("sanity_bundle"))),
        _check("candidate bundle is seq146", exists and int(bundle.get("seq_input_dim") or 0) == 146 and int(bundle.get("snap_input_dim") or 0) == 146),
        _check("candidate bundle has multi-TF enabled", exists and bool(bundle.get("multi_tf_enabled"))),
        _check("candidate bundle has specialist fusion", exists and bool(bundle.get("specialist_fusion_enabled"))),
        _check(
            "candidate bundle specialist model contract is preserved in bundle metadata",
            exists and _bundle_specialist_model_contract_passes(report),
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
            "candidate bundle includes required specialist groups",
            exists and set(REQUIRED_SPECIALIST_GROUPS).issubset(specialist_groups),
            {"specialist_groups": sorted(specialist_groups)},
        ),
        _check(
            "candidate bundle has exact specialist groups",
            exists and specialist_groups == set(REQUIRED_SPECIALIST_GROUPS),
            {
                "expected_specialist_groups": list(REQUIRED_SPECIALIST_GROUPS),
                "actual_specialist_groups": sorted(specialist_groups),
            },
        ),
        _check(
            "candidate bundle audit was run with specialist-fusion gate contract",
            exists
            and bool(report.get("require_specialist_fusion"))
            and required_specialists == set(REQUIRED_SPECIALIST_GROUPS)
            and int(report.get("min_active_specialists") or 0) >= len(REQUIRED_SPECIALIST_GROUPS)
            and float(report.get("min_gate_entropy") or -1.0) >= float(REQUIRED_MIN_GATE_ENTROPY),
            {
                "required_training_specialists": report.get("required_training_specialists"),
                "min_active_specialists": report.get("min_active_specialists"),
                "min_gate_entropy": report.get("min_gate_entropy"),
            },
        ),
        _check(
            "candidate bundle required specialist gate weights are non-collapsed",
            exists and required_gate_live,
            {"min_mean_weight": 0.01, "required_specialists": list(REQUIRED_SPECIALIST_GROUPS)},
        ),
        _check("candidate bundle audit was run with require_head_contract", exists and bool(report.get("require_head_contract"))),
        _check(
            "candidate bundle head contract PASS",
            exists
            and str(head_contract.get("decision")) == "PASS"
            and not head_contract.get("failures")
            and active_heads == set(EXPECTED_ACTIVE_TRAINING_HEADS)
            and blocked_heads == set(EXPECTED_BLOCKED_HEADS),
            {
                "head_contract": head_contract,
                "expected_active_heads": list(EXPECTED_ACTIVE_TRAINING_HEADS),
                "actual_active_heads": sorted(active_heads),
                "expected_blocked_heads": list(EXPECTED_BLOCKED_HEADS),
                "actual_blocked_heads": sorted(blocked_heads),
            },
        ),
        _check(
            "candidate bundle audit validated pre-train manifest provenance",
            exists
            and str(pretrain_manifest.get("decision")) == "PASS"
            and not pretrain_manifest.get("failures")
            and bool(pretrain_manifest.get("feature_objective_coverage_all_present"))
            and bool(pretrain_manifest.get("feature_objective_liveness_all_live"))
            and bool(pretrain_manifest.get("feature_source_field_liveness_all_live"))
            and bool(pretrain_manifest.get("specialist_objective_routing_all_present_and_expected"))
            and bool(pretrain_manifest.get("specialist_input_liveness_all_live"))
            and bool(pretrain_manifest.get("specialist_active_heads_match_target"))
            and bool(pretrain_manifest.get("specialist_blocked_heads_match_target"))
            and bool(pretrain_manifest.get("specialist_required_training_set_exact"))
            and bool(pretrain_manifest.get("specialist_trainable_set_exact"))
            and bool(pretrain_manifest.get("specialist_model_contract_valid"))
            and bool(pretrain_manifest.get("specialist_model_contract_set_exact"))
            and bool(pretrain_manifest.get("specialist_model_contract_owned_objectives_match"))
            and bool(pretrain_manifest.get("smoke_edge_required_specialists_exact"))
            and bool(pretrain_manifest.get("smoke_edge_specialist_groups_exact"))
            and bool(pretrain_manifest.get("smoke_edge_specialist_model_contract_valid"))
            and bool(pretrain_manifest.get("smoke_edge_specialist_model_contract_set_exact"))
            and bool(pretrain_manifest.get("smoke_edge_specialist_model_contract_owned_objectives_match"))
            and bool(pretrain_manifest.get("smoke_dataset_audit_provenance_all_artifacts_present"))
            and bool(pretrain_manifest.get("smoke_dataset_audit_provenance_all_artifact_hashes_present"))
            and bool(pretrain_manifest.get("smoke_edge_worktree_critical_gate_review_ok")),
            {"pretrain_manifest_contract": pretrain_manifest},
        ),
        _check(
            "candidate bundle audit covered val/test rows",
            exists and split_rows.get("val", 0) > 0 and split_rows.get("test", 0) > 0,
            {"split_rows": split_rows},
        ),
    ]


def _best_replay_row(metrics: pd.DataFrame) -> dict[str, Any] | None:
    if metrics.empty:
        return None
    if "scope" in metrics.columns:
        aggregate = metrics[metrics["scope"].astype(str).isin(["aggregate", "ALL", "overall"])]
        if not aggregate.empty:
            metrics = aggregate
    if "net_sum_bps" in metrics.columns:
        metrics = metrics.sort_values("net_sum_bps", ascending=False)
    return metrics.iloc[0].to_dict()


def _replay_checks(
    replay_dir: Path,
    manifest: dict[str, Any],
    metrics: pd.DataFrame,
    monthly: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    min_net_sum_bps: float,
    min_profit_factor: float,
    max_drawdown_bps: float,
    expected_candidate_bundle_dir: str | None = None,
) -> list[dict[str, Any]]:
    row = _best_replay_row(metrics)
    required_metric_columns = {
        "policy_id",
        "n_trades",
        "net_sum_bps",
        "win_rate",
        "profit_factor",
        "max_drawdown_bps",
        "max_loss_bps",
    }
    metric_columns = set(str(c) for c in metrics.columns)
    monthly_columns = set(str(c) for c in monthly.columns)
    trade_columns = set(str(c) for c in trades.columns)
    iql_missing_columns = sorted(set(IQL_TRANSITION_REQUIRED_COLUMNS) - trade_columns)
    identity = manifest.get("replay_identity_contract") if isinstance(manifest.get("replay_identity_contract"), dict) else {}
    row_details = row or {}
    drawdown = row_details.get("max_drawdown_bps")
    drawdown_ok = drawdown is not None and abs(float(drawdown)) <= float(max_drawdown_bps)
    return [
        _check("offline replay dir exists", replay_dir.exists(), {"replay_dir": str(replay_dir)}),
        _check("offline replay manifest PASS", str(manifest.get("decision")) == "PASS", {"manifest_failures": manifest.get("failures")}),
        _check("offline replay manifest has zero failures", not manifest.get("failures"), {"manifest_failures": manifest.get("failures")}),
        _check("offline replay identity contract ready", bool(identity.get("ready")), {"identity": identity}),
        _check(
            "offline replay identity matches candidate bundle audit",
            True if expected_candidate_bundle_dir is None else str(identity.get("candidate_bundle_dir") or "") == str(expected_candidate_bundle_dir),
            {"expected_candidate_bundle_dir": expected_candidate_bundle_dir, "identity_candidate_bundle_dir": identity.get("candidate_bundle_dir")},
        ),
        _check("offline replay metrics exist and have rows", not metrics.empty, {"rows": int(len(metrics))}),
        _check(
            "offline replay metrics have required columns",
            required_metric_columns.issubset(metric_columns),
            {"missing_columns": sorted(required_metric_columns - metric_columns)},
        ),
        _check("offline replay monthly file has rows", not monthly.empty, {"rows": int(len(monthly))}),
        _check("offline replay monthly file has net_sum_bps", "net_sum_bps" in monthly_columns),
        _check("offline replay trades file has rows", not trades.empty, {"rows": int(len(trades))}),
        _check(
            "offline replay trades have IQL transition columns",
            not iql_missing_columns,
            {"missing_columns": iql_missing_columns, "required_columns": list(IQL_TRANSITION_REQUIRED_COLUMNS)},
        ),
        _check(
            "offline replay best row has enough trades",
            row is not None and int(row_details.get("n_trades") or 0) > 0,
            {"best_row": row_details},
        ),
        _check(
            "offline replay net sum is positive",
            row is not None and float(row_details.get("net_sum_bps") or 0.0) > float(min_net_sum_bps),
            {"threshold": min_net_sum_bps, "best_row": row_details},
        ),
        _check(
            "offline replay profit factor passes",
            row is not None and float(row_details.get("profit_factor") or 0.0) >= float(min_profit_factor),
            {"threshold": min_profit_factor, "best_row": row_details},
        ),
        _check(
            "offline replay drawdown is within bound",
            drawdown_ok,
            {"max_abs_drawdown_bps": max_drawdown_bps, "best_row": row_details},
        ),
        _check(
            "offline replay has no negative months",
            not monthly.empty and "net_sum_bps" in monthly.columns and bool((monthly["net_sum_bps"].astype(float) > 0.0).all()),
            {"monthly_rows": int(len(monthly))},
        ),
    ]


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Replay Readiness",
        "",
        f"- Decision: `{report['decision']}`",
        f"- IQL distillation allowed with explicit vedtak: `{report['iql_distillation_allowed_with_explicit_vedtak']}`",
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
    candidate_readiness_path = Path(args.candidate_readiness_json).expanduser().resolve()
    candidate_bundle_audit_path = Path(args.candidate_bundle_audit_json).expanduser().resolve()
    selective_summary_path = Path(args.selective_edge_summary_json).expanduser().resolve()
    selective_metrics_path = Path(args.selective_edge_metrics_csv).expanduser().resolve()
    replay_dir = Path(args.replay_dir).expanduser().resolve()

    candidate_readiness = _read_json(candidate_readiness_path)
    candidate_bundle_audit = _read_json(candidate_bundle_audit_path) if candidate_bundle_audit_path.exists() else {}
    expected_candidate_bundle_dir = str(candidate_bundle_audit.get("bundle_dir") or "") or None
    selective_summary = _read_json(selective_summary_path) if selective_summary_path.exists() else {}
    selective_metrics = _read_csv_or_empty(selective_metrics_path)
    replay_metrics = _read_csv_or_empty(replay_dir / "replay_policy_metrics.csv")
    replay_monthly = _read_csv_or_empty(replay_dir / "replay_policy_monthly.csv")
    replay_trades = _read_csv_or_empty(replay_dir / "replay_policy_trades.csv")
    replay_manifest = _read_json(replay_dir / "REPLAY_EVIDENCE_MANIFEST.json") if (replay_dir / "REPLAY_EVIDENCE_MANIFEST.json").exists() else {}
    replay_identity = replay_manifest.get("replay_identity_contract") if isinstance(replay_manifest.get("replay_identity_contract"), dict) else {}
    evidence_identity = {
        "candidate_bundle_audit_json": str(candidate_bundle_audit_path),
        "selective_edge_summary_json": str(selective_summary_path),
        "replay_evidence_manifest_json": str(replay_dir / "REPLAY_EVIDENCE_MANIFEST.json"),
        "candidate_bundle_dir": str(candidate_bundle_audit.get("bundle_dir") or ""),
        "selective_edge_bundle_dir": str(selective_summary.get("bundle_dir") or ""),
        "replay_identity_candidate_bundle_dir": str(replay_identity.get("candidate_bundle_dir") or ""),
        "no_xgb_bundle_dir": str(selective_summary.get("no_xgb_bundle_dir") or replay_identity.get("no_xgb_bundle_dir") or ""),
        "replay_identity_ready": bool(replay_identity.get("ready")),
    }
    artifacts = {
        "candidate_readiness": str(candidate_readiness_path),
        "candidate_bundle_audit": str(candidate_bundle_audit_path),
        "selective_edge_summary": str(selective_summary_path),
        "selective_edge_metrics": str(selective_metrics_path),
        "candidate_replay_manifest": str(replay_dir / "REPLAY_EVIDENCE_MANIFEST.json"),
        "candidate_replay_metrics": str(replay_dir / "replay_policy_metrics.csv"),
        "candidate_replay_monthly": str(replay_dir / "replay_policy_monthly.csv"),
        "candidate_replay_trades": str(replay_dir / "replay_policy_trades.csv"),
    }
    artifact_fingerprints = _artifact_fingerprints(artifacts)

    gate_checks = {
        "candidate_readiness": [
            _check(
                "candidate-readiness is green",
                str(candidate_readiness.get("decision")) == "READY_FOR_CANDIDATE_TRAINING_VEDTAK",
                {"decision": candidate_readiness.get("decision"), "failures": candidate_readiness.get("failures")},
            ),
            _check(
                "candidate-readiness still blocks promotion/shadow/live",
                bool(candidate_readiness.get("promotion_shadow_live_allowed")) is False,
            ),
        ],
        "candidate_bundle_audit": _candidate_bundle_audit_checks(candidate_bundle_audit_path, candidate_bundle_audit),
        "selective_edge": _selective_edge_checks(
            selective_summary,
            selective_metrics,
            model_name=str(args.model_name),
            min_top5_mean_pnl_bps=float(args.min_top5_mean_pnl_bps),
            min_top10_mean_pnl_bps=float(args.min_top10_mean_pnl_bps),
            require_no_xgb_ablation=bool(args.require_no_xgb_ablation),
            expected_bundle_dir=expected_candidate_bundle_dir,
        ),
        "offline_replay": _replay_checks(
            replay_dir,
            replay_manifest,
            replay_metrics,
            replay_monthly,
            replay_trades,
            min_net_sum_bps=float(args.min_replay_net_sum_bps),
            min_profit_factor=float(args.min_profit_factor),
            max_drawdown_bps=float(args.max_abs_drawdown_bps),
            expected_candidate_bundle_dir=expected_candidate_bundle_dir,
        ),
        "iql_distillation_guard": [
            _check("gate never trains IQL", True),
            _check("gate never promotes", True),
            _check("gate never starts shadow/live", True),
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
        "schema_version": "entry_replay_readiness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "READY_FOR_IQL_DISTILLATION_VEDTAK" if ready else "NOT_READY_FOR_IQL_DISTILLATION",
        "iql_distillation_allowed_with_explicit_vedtak": bool(ready),
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            "IQL distillation research wrapper with explicit vedtak and post-distillation replay comparison"
            if ready
            else "run candidate train, selective-edge eval, and 2026 offline replay before IQL distillation"
        ),
        "candidate_readiness_json": str(candidate_readiness_path),
        "candidate_bundle_audit_json": str(candidate_bundle_audit_path),
        "selective_edge_summary_json": str(selective_summary_path),
        "selective_edge_metrics_csv": str(selective_metrics_path),
        "replay_dir": str(replay_dir),
        "evidence_identity": evidence_identity,
        "artifacts": artifacts,
        "artifact_fingerprints": artifact_fingerprints,
        "gates": gates,
        "failures": failures,
    }
    json_path = out_dir / f"ENTRY_REPLAY_READINESS_{timestamp}.json"
    md_path = out_dir / f"ENTRY_REPLAY_READINESS_{timestamp}.md"
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    latest_json = out_dir / "ENTRY_REPLAY_READINESS_latest.json"
    latest_md = out_dir / "ENTRY_REPLAY_READINESS_latest.md"
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
    ap.add_argument("--candidate-readiness-json", default=str(CANDIDATE_READINESS_LATEST))
    ap.add_argument("--candidate-bundle-audit-json", default=str(DEFAULT_CANDIDATE_BUNDLE_AUDIT))
    ap.add_argument("--selective-edge-summary-json", default=str(DEFAULT_SELECTIVE_EDGE_DIR / "summary.json"))
    ap.add_argument("--selective-edge-metrics-csv", default=str(DEFAULT_SELECTIVE_EDGE_DIR / "selective_edge_metrics.csv"))
    ap.add_argument("--replay-dir", default=str(DEFAULT_REPLAY_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--model-name", default="candidate")
    ap.add_argument("--min-top5-mean-pnl-bps", type=float, default=0.0)
    ap.add_argument("--min-top10-mean-pnl-bps", type=float, default=0.0)
    ap.add_argument("--min-replay-net-sum-bps", type=float, default=0.0)
    ap.add_argument("--min-profit-factor", type=float, default=1.05)
    ap.add_argument("--max-abs-drawdown-bps", type=float, default=650.0)
    ap.add_argument("--require-no-xgb-ablation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
