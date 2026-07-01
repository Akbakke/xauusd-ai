#!/usr/bin/env python3
"""Lock the required post-train audit contract for active Exit Transformer.

This gate does not inspect a trained bundle, because no Exit Transformer
trainer is approved yet. It proves that the future post-train bundle audit is
fully specified before any train-execution enablement package can be considered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


DEFAULT_TRAIN_EXECUTION_REVIEW_JSON = (
    REPORTS_ROOT
    / "entry_exit_transformer_train_execution_review_20260630_v1/ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_latest.json"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_exit_transformer_post_train_contract_20260630_v1"

READY_TRAIN_EXECUTION_REVIEW_DECISION = "ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_READY_FOR_EXPLICIT_VEDTAK_PACKAGE"
READY_TRAINING_PLAN_DECISION = "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READY_FOR_VEDTAK_REVIEW"
READY_PRETRAIN_DECISION = "ENTRY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST_READY_FOR_TRAIN_EXECUTION_REVIEW"
READY_SLICE_DECISION = "ENTRY_EXIT_MODEL_DATASET_SLICE_ROBUSTNESS_READY_WITH_WEAK_SLICE_DISCLOSURE"
READY_DECISION = "ENTRY_EXIT_TRANSFORMER_POST_TRAIN_AUDIT_CONTRACT_READY"
BLOCKED_DECISION = "BLOCKED_BY_EXIT_TRANSFORMER_POST_TRAIN_AUDIT_CONTRACT"
EXPECTED_HEADS = (
    "exit_now_logit",
    "hold_value_bps",
    "exit_now_reward_bps",
    "giveback_risk_bps",
    "mfe_capture_ratio",
)


def _check(name: str, condition: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(condition), "details": details or {}}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_if_file(path: Path) -> str:
    return _sha256_file(path) if path.is_file() else ""


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _hash_matches(report: dict[str, Any], key: str, path: Path) -> bool:
    expected = str(report.get(key) or "")
    return bool(expected and path.is_file() and _sha256_file(path) == expected)


def _path_from_report(report: dict[str, Any], key: str) -> Path:
    return Path(str(report.get(key) or "")).expanduser()


def _weak_slice_policy_ready(review: dict[str, Any]) -> bool:
    contract = review.get("review_contract") if isinstance(review.get("review_contract"), dict) else {}
    policy = contract.get("weak_slice_policy") if isinstance(contract.get("weak_slice_policy"), dict) else {}
    requirements = (
        policy.get("train_execution_requirements")
        if isinstance(policy.get("train_execution_requirements"), dict)
        else {}
    )
    required_flags = (
        "must_report_session_regime_side_metrics",
        "must_report_direction_and_tail_metrics",
        "must_not_promote_from_broad_average",
        "must_compare_weak_slices_separately",
        "must_block_shadow_live_on_unsupported_slice",
        "must_keep_weak_slice_disclosure_in_post_train_audit",
    )
    return (
        int(policy.get("unsupported_slice_count") or 0) == 0
        and "weak_slice_count" in policy
        and all(requirements.get(flag) is True for flag in required_flags)
    )


def _extract_heads(training_plan: dict[str, Any], pretrain_manifest: dict[str, Any]) -> dict[str, Any]:
    plan = training_plan.get("training_plan") if isinstance(training_plan.get("training_plan"), dict) else {}
    architecture = plan.get("architecture") if isinstance(plan.get("architecture"), dict) else {}
    preflight = (
        pretrain_manifest.get("preflight_manifest")
        if isinstance(pretrain_manifest.get("preflight_manifest"), dict)
        else {}
    )
    return {
        "training_plan_heads": list(architecture.get("output_heads") or []),
        "preflight_heads": list(preflight.get("output_heads") or []),
        "expected_heads": list(EXPECTED_HEADS),
    }


def _build_contract(train_review: dict[str, Any], training_plan: dict[str, Any], pretrain: dict[str, Any]) -> dict[str, Any]:
    plan = training_plan.get("training_plan") if isinstance(training_plan.get("training_plan"), dict) else {}
    dataset = plan.get("dataset") if isinstance(plan.get("dataset"), dict) else {}
    resources = plan.get("resource_guardrails") if isinstance(plan.get("resource_guardrails"), dict) else {}
    review_contract = (
        train_review.get("review_contract")
        if isinstance(train_review.get("review_contract"), dict)
        else {}
    )
    weak_policy = (
        review_contract.get("weak_slice_policy")
        if isinstance(review_contract.get("weak_slice_policy"), dict)
        else {}
    )
    review_json_path = Path(str(train_review.get("json_path") or ""))
    feature_schema_json = str(dataset.get("feature_schema_json") or "").strip()
    feature_schema_sha256 = str(dataset.get("feature_schema_json_sha256") or "").strip()
    return {
        "model_family": "exit_sequence_transformer_v1",
        "audit_name": "entry_exit_transformer_post_train_bundle_audit_v1",
        "exact_output_heads": list(EXPECTED_HEADS),
        "extra_output_heads_allowed": False,
        "required_bundle_identity": {
            "training_plan_json_sha256": train_review.get("training_plan_json_sha256"),
            "train_execution_review_json_sha256": _sha256_if_file(review_json_path),
            "pretrain_manifest_json_sha256": train_review.get("pretrain_manifest_json_sha256"),
            "slice_robustness_json_sha256": train_review.get("slice_robustness_json_sha256"),
            "normalization_policy": "train_split_only",
            "normalization_json": dataset.get("normalization_json"),
            "feature_schema_json": feature_schema_json,
            "feature_schema_json_sha256": feature_schema_sha256,
        },
        "required_load_checks": {
            "strict_bundle_load": True,
            "cpu_finite_forward_pass": True,
            "causal_mask_preserved": True,
            "state_dict_heads_match_exact_output_heads": True,
            "forward_output_keys_match_exact_output_heads": True,
            "metadata_train_recipe_heads_match_exact_output_heads": True,
            "train_only_normalization_hash_matches_training_plan": True,
            "dataset_shard_hashes_match_training_plan": True,
            "post_train_contract_sha_preserved_in_bundle_metadata": True,
        },
        "required_metric_slices": {
            "splits": ["train", "val", "test"],
            "required_slice_families": [
                "session",
                "regime",
                "side",
                "session_x_side",
                "volatility_regime",
                "tail_loss",
                "weak_slices",
            ],
            "weak_slice_count_to_preserve": int(weak_policy.get("weak_slice_count") or 0),
            "unsupported_slice_count_must_remain_zero": True,
            "must_not_promote_from_broad_average": True,
        },
        "required_edge_diagnostics": {
            "net_reward_proxy_bps": True,
            "drawdown_or_mae_bps": True,
            "giveback_risk_bps": True,
            "mfe_capture_ratio": True,
            "exit_now_reward_bps": True,
            "exit_now_calibration": True,
            "hold_value_regression_error": True,
            "bad_path_tail_rows": True,
            "session_regime_side_robustness": True,
        },
        "resource_guardrails": {
            "audit_num_workers": 0,
            "max_process_rss_gib": resources.get("max_process_rss_gib"),
            "abort_if_mem_available_below_gib": resources.get("abort_if_mem_available_below_gib"),
            "bounded_forward_batch_required": True,
        },
        "downstream_blocks_until_post_train_passes": {
            "replay_evidence": True,
            "exit_iql_distillation": True,
            "candidate_promotion": True,
            "shadow": True,
            "live": True,
        },
        "pretrain_manifest_decision": pretrain.get("decision"),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    contract = report["post_train_audit_contract"]
    lines = [
        "# Entry Exit Transformer Post-Train Audit Contract",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Exact output heads: `{contract['exact_output_heads']}`",
        f"- Weak slice count to preserve: `{contract['required_metric_slices']['weak_slice_count_to_preserve']}`",
        f"- Exit training allowed: `{report['exit_training_allowed']}`",
        f"- Exit IQL allowed: `{report['exit_iql_allowed']}`",
        f"- Replay allowed: `{report['replay_started']}`",
        f"- Next required gate: `{report['next_required_gate']}`",
        "",
        "## Required Diagnostics",
        "",
    ]
    for key in contract["required_edge_diagnostics"]:
        lines.append(f"- `{key}`")
    lines.extend(["", "## Failures", ""])
    if report["failures"]:
        for failure in report["failures"]:
            lines.append(f"- `{failure['check']}`")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_review_json = Path(args.train_execution_review_json).expanduser().resolve()
    train_review = _read_json_or_empty(train_review_json)
    training_plan_json = _path_from_report(train_review, "training_plan_json")
    pretrain_manifest_json = _path_from_report(train_review, "pretrain_manifest_json")
    slice_robustness_json = _path_from_report(train_review, "slice_robustness_json")
    training_plan = _read_json_or_empty(training_plan_json)
    pretrain = _read_json_or_empty(pretrain_manifest_json)
    slice_robustness = _read_json_or_empty(slice_robustness_json)
    heads = _extract_heads(training_plan, pretrain)
    contract = _build_contract(train_review, training_plan, pretrain)
    checks = [
        _check("active Exit Transformer train-execution review exists", train_review_json.exists(), {"path": str(train_review_json)}),
        _check(
            "active Exit Transformer train-execution review is ready",
            train_review.get("decision") == READY_TRAIN_EXECUTION_REVIEW_DECISION,
            {"decision": train_review.get("decision"), "required": READY_TRAIN_EXECUTION_REVIEW_DECISION},
        ),
        _check(
            "active Exit Transformer train-execution review keeps training closed",
            train_review.get("exit_training_allowed") is False
            and train_review.get("exit_training_allowed_with_explicit_vedtak") is False,
            {
                "exit_training_allowed": train_review.get("exit_training_allowed"),
                "exit_training_allowed_with_explicit_vedtak": train_review.get("exit_training_allowed_with_explicit_vedtak"),
            },
        ),
        _check("training plan hash matches train-execution review provenance", _hash_matches(train_review, "training_plan_json_sha256", training_plan_json), {"path": str(training_plan_json)}),
        _check("pretrain manifest hash matches train-execution review provenance", _hash_matches(train_review, "pretrain_manifest_json_sha256", pretrain_manifest_json), {"path": str(pretrain_manifest_json)}),
        _check("slice robustness hash matches train-execution review provenance", _hash_matches(train_review, "slice_robustness_json_sha256", slice_robustness_json), {"path": str(slice_robustness_json)}),
        _check("active Exit Transformer training plan is ready", training_plan.get("decision") == READY_TRAINING_PLAN_DECISION, {"decision": training_plan.get("decision")}),
        _check("active Exit Transformer pretrain manifest is ready", pretrain.get("decision") == READY_PRETRAIN_DECISION, {"decision": pretrain.get("decision")}),
        _check("active Exit model slice robustness is ready", slice_robustness.get("decision") == READY_SLICE_DECISION, {"decision": slice_robustness.get("decision")}),
        _check(
            "exact Exit output heads are locked across plan and preflight",
            heads["training_plan_heads"] == list(EXPECTED_HEADS)
            and heads["preflight_heads"] == list(EXPECTED_HEADS),
            heads,
        ),
        _check(
            "weak-slice policy is preserved for post-train audit",
            _weak_slice_policy_ready(train_review),
            (train_review.get("review_contract") or {}).get("weak_slice_policy") if isinstance(train_review.get("review_contract"), dict) else {},
        ),
        _check(
            "feature schema identity is pinned for post-train audit",
            bool(contract["required_bundle_identity"]["feature_schema_json"])
            and bool(contract["required_bundle_identity"]["feature_schema_json_sha256"]),
            contract["required_bundle_identity"],
        ),
        _check(
            "post-train audit contract requires edge diagnostics and blocks broad averages",
            all(contract["required_edge_diagnostics"].values())
            and contract["required_metric_slices"]["must_not_promote_from_broad_average"] is True
            and contract["downstream_blocks_until_post_train_passes"]["replay_evidence"] is True,
            contract,
        ),
        _check(
            "post-train audit contract never trains, replays, distills, promotes, shadows, or starts live",
            True,
            {
                "trainer_started": False,
                "replay_started": False,
                "iql_distillation_started": False,
                "exit_training_allowed": False,
                "exit_iql_allowed": False,
                "promotion_shadow_live_allowed": False,
            },
        ),
    ]
    failures = [{"check": check["name"], "details": check.get("details") or {}} for check in checks if not check["ok"]]
    ready = not failures
    decision = READY_DECISION if ready else BLOCKED_DECISION
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"ENTRY_EXIT_TRANSFORMER_POST_TRAIN_CONTRACT_{timestamp}.json"
    md_path = out_dir / f"ENTRY_EXIT_TRANSFORMER_POST_TRAIN_CONTRACT_{timestamp}.md"
    report = {
        "schema_version": "entry_exit_transformer_post_train_contract_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "train_execution_review_json": str(train_review_json),
        "train_execution_review_json_sha256": _sha256_if_file(train_review_json),
        "training_plan_json": str(training_plan_json),
        "training_plan_json_sha256": _sha256_if_file(training_plan_json),
        "pretrain_manifest_json": str(pretrain_manifest_json),
        "pretrain_manifest_json_sha256": _sha256_if_file(pretrain_manifest_json),
        "slice_robustness_json": str(slice_robustness_json),
        "slice_robustness_json_sha256": _sha256_if_file(slice_robustness_json),
        "head_contract": heads,
        "post_train_audit_contract": contract,
        "checks": checks,
        "failures": failures,
        "exit_training_allowed": False,
        "exit_training_allowed_with_explicit_vedtak": False,
        "exit_iql_allowed": False,
        "trainer_started": False,
        "replay_started": False,
        "iql_distillation_started": False,
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            "explicit Exit Transformer train-execution enablement vedtak package; training remains closed"
            if ready
            else "repair post-train audit contract before any Exit train enablement package"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_EXIT_TRANSFORMER_POST_TRAIN_CONTRACT_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_EXIT_TRANSFORMER_POST_TRAIN_CONTRACT_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if not args.quiet:
        print(json.dumps({"decision": decision, "failures": failures, "json_path": str(json_path)}, indent=2, sort_keys=True))
    if args.fail_on_not_ready and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-execution-review-json", default=str(DEFAULT_TRAIN_EXECUTION_REVIEW_JSON))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
