from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.analysis import shadow_meta_v1 as shadow_meta  # noqa: E402


ACTIVE_TRUTH_PIPELINE_ROOT_POINTER = (
    Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _resolve_active_truth_root() -> Path:
    if not ACTIVE_TRUTH_PIPELINE_ROOT_POINTER.exists():
        raise FileNotFoundError(
            f"[MANAGEMENT_EXIT_LOCAL_SHADOW_MANUAL_REVIEW_V1] missing active truth pointer: "
            f"{ACTIVE_TRUTH_PIPELINE_ROOT_POINTER}"
        )
    raw_value = ACTIVE_TRUTH_PIPELINE_ROOT_POINTER.read_text(encoding="utf-8").strip()
    if not raw_value:
        raise RuntimeError(
            f"[MANAGEMENT_EXIT_LOCAL_SHADOW_MANUAL_REVIEW_V1] active truth pointer is empty: "
            f"{ACTIVE_TRUTH_PIPELINE_ROOT_POINTER}"
        )
    truth_root = Path(raw_value).expanduser().resolve()
    if not truth_root.exists():
        raise FileNotFoundError(
            f"[MANAGEMENT_EXIT_LOCAL_SHADOW_MANUAL_REVIEW_V1] active truth root does not exist: {truth_root}"
        )
    return truth_root


def _resolve_runtime_ledger_dir(truth_root: Path, runtime_ledger_dir: str | None) -> Path:
    if runtime_ledger_dir:
        resolved = Path(runtime_ledger_dir).expanduser().resolve()
    else:
        rebuild_summary_path = truth_root / "truth_downstream_canonical_rebuild_v1.json"
        if not rebuild_summary_path.exists():
            raise FileNotFoundError(
                "[MANAGEMENT_EXIT_LOCAL_SHADOW_MANUAL_REVIEW_V1] missing canonical rebuild summary: "
                f"{rebuild_summary_path}"
            )
        rebuild_summary = _read_json(rebuild_summary_path)
        raw_ledger_dir = rebuild_summary.get("ledger_dir")
        if not raw_ledger_dir:
            raise RuntimeError(
                "[MANAGEMENT_EXIT_LOCAL_SHADOW_MANUAL_REVIEW_V1] canonical rebuild summary is missing ledger_dir"
            )
        resolved = Path(str(raw_ledger_dir))
        if not resolved.is_absolute():
            resolved = (truth_root / resolved).resolve()
        else:
            resolved = resolved.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"[MANAGEMENT_EXIT_LOCAL_SHADOW_MANUAL_REVIEW_V1] runtime ledger dir does not exist: {resolved}"
        )
    return resolved


def _artifact_path(runtime_ledger_dir: Path, runtime_build_summary: Dict[str, Any], key: str, filename: str) -> Path:
    artifact_paths = dict(runtime_build_summary.get("artifact_paths", {}))
    raw_path = artifact_paths.get(key)
    if raw_path and str(raw_path) != "NOT_AVAILABLE":
        path = Path(str(raw_path)).expanduser().resolve()
    else:
        path = (runtime_ledger_dir / filename).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"[MANAGEMENT_EXIT_LOCAL_SHADOW_MANUAL_REVIEW_V1] missing required runtime artifact for {key}: {path}"
        )
    return path


def _namespace_dir(truth_root: Path, timestamp_token: str, suffix: str) -> Path:
    return truth_root / f"{shadow_meta._ALL_TRADE_REVIEW_LEDGER_NAME}_{timestamp_token}_{suffix}"


def _ensure_append_only_dir(target_dir: Path) -> None:
    if target_dir.exists() and any(target_dir.iterdir()):
        raise RuntimeError(
            "[MANAGEMENT_EXIT_LOCAL_SHADOW_MANUAL_REVIEW_V1_APPEND_ONLY] target namespace already contains files: "
            f"{target_dir}"
        )
    target_dir.mkdir(parents=True, exist_ok=True)


def _build_shared_build_summary(
    *,
    truth_root: Path,
    runtime_ledger_dir: Path,
    runtime_build_summary: Dict[str, Any],
    artifact_paths: Dict[str, str],
    stage_name_v1: str,
    stage_summary_v1: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "ledger_name": runtime_build_summary.get("ledger_name", shadow_meta._ALL_TRADE_REVIEW_LEDGER_NAME),
        "ledger_mode": runtime_build_summary.get("ledger_mode", shadow_meta._ALL_TRADE_REVIEW_LEDGER_MODE),
        "review_schema_version": runtime_build_summary.get("review_schema_version"),
        "reviewer_contract_version": runtime_build_summary.get("reviewer_contract_version"),
        "control_date": runtime_build_summary.get("control_date"),
        "reports_root": str(truth_root),
        "run_universe": dict(runtime_build_summary.get("run_universe", {})),
        "closed_trade_rows": int(runtime_build_summary.get("closed_trade_rows", 0)),
        "as_of_supervision_join_coverage": dict(runtime_build_summary.get("as_of_supervision_join_coverage", {})),
        "leakage_guard": dict(runtime_build_summary.get("leakage_guard", {})),
        "source_runtime_ledger_dir_v1": str(runtime_ledger_dir),
        "build_timestamp_utc_v1": stage_summary_v1["build_timestamp_utc_v1"],
        "stage_name_v1": stage_name_v1,
        "artifact_paths": artifact_paths,
        "stage_summary_v1": stage_summary_v1,
        "source_provenance_v1": {
            "runtime_build_summary_path_v1": str(
                runtime_ledger_dir / "shadow_meta_all_trade_review_ledger_build_summary.json"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize MANAGEMENT_EXIT_LOCAL_SHADOW_AUDIT_V1 and "
            "MANAGEMENT_EXIT_LOCAL_MANUAL_REVIEW_V1 from a completed all-trade runtime ledger."
        )
    )
    parser.add_argument("--truth-root", default=None, help="Truth root. Defaults to ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
    parser.add_argument(
        "--runtime-ledger-dir",
        default=None,
        help="Runtime all-trade ledger directory. Defaults to ledger_dir from truth_downstream_canonical_rebuild_v1.json",
    )
    parser.add_argument(
        "--timestamp-token",
        default=None,
        help="Append-only namespace token. Defaults to current UTC timestamp like 20260420T123456Z",
    )
    args = parser.parse_args()

    truth_root = (
        Path(args.truth_root).expanduser().resolve() if args.truth_root else _resolve_active_truth_root()
    )
    runtime_ledger_dir = _resolve_runtime_ledger_dir(truth_root, args.runtime_ledger_dir)
    runtime_build_summary_path = runtime_ledger_dir / "shadow_meta_all_trade_review_ledger_build_summary.json"
    if not runtime_build_summary_path.exists():
        raise FileNotFoundError(
            "[MANAGEMENT_EXIT_LOCAL_SHADOW_MANUAL_REVIEW_V1] missing runtime build summary: "
            f"{runtime_build_summary_path}"
        )
    runtime_build_summary = _read_json(runtime_build_summary_path)

    timestamp_token = args.timestamp_token or pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    shadow_dir = _namespace_dir(truth_root, timestamp_token, "MANAGEMENT_EXIT_LOCAL_SHADOW_AUDIT_V1")
    manual_dir = _namespace_dir(truth_root, timestamp_token, "MANAGEMENT_EXIT_LOCAL_MANUAL_REVIEW_V1")

    management_bandit_observed_sample_view_v1_df = pd.read_parquet(
        _artifact_path(
            runtime_ledger_dir,
            runtime_build_summary,
            "management_bandit_observed_sample_view_path",
            shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_BANDIT_OBSERVED_SAMPLE_VIEW_V1_PARQUET,
        )
    )
    management_exit_local_all_eligible_scored_view_v1_df = pd.read_parquet(
        _artifact_path(
            runtime_ledger_dir,
            runtime_build_summary,
            "management_exit_local_all_eligible_scored_view_path",
            shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_ALL_ELIGIBLE_SCORED_VIEW_V1_PARQUET,
        )
    )
    management_bandit_exit_local_reward_view_v1_df = pd.read_parquet(
        _artifact_path(
            runtime_ledger_dir,
            runtime_build_summary,
            "management_bandit_exit_local_reward_view_path",
            shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_BANDIT_EXIT_LOCAL_REWARD_VIEW_V1_PARQUET,
        )
    )
    management_exit_local_benchmark_summary_v1 = _read_json(
        _artifact_path(
            runtime_ledger_dir,
            runtime_build_summary,
            "management_exit_local_benchmark_summary_path",
            shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_BENCHMARK_SUMMARY_V1,
        )
    )
    management_exit_local_status_v1 = _read_json(
        _artifact_path(
            runtime_ledger_dir,
            runtime_build_summary,
            "management_exit_local_status_path",
            shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_STATUS_V1,
        )
    )
    as_of_supervision_join_coverage_summary = dict(runtime_build_summary.get("as_of_supervision_join_coverage", {}))
    leakage_guard_summary = dict(runtime_build_summary.get("leakage_guard", {}))
    if not as_of_supervision_join_coverage_summary:
        raise RuntimeError(
            "[MANAGEMENT_EXIT_LOCAL_SHADOW_MANUAL_REVIEW_V1] runtime build summary is missing "
            "as_of_supervision_join_coverage"
        )
    if not leakage_guard_summary:
        raise RuntimeError(
            "[MANAGEMENT_EXIT_LOCAL_SHADOW_MANUAL_REVIEW_V1] runtime build summary is missing leakage_guard"
        )

    shadow_payload = shadow_meta._build_management_exit_local_shadow_audit_v1(
        management_bandit_observed_sample_view_v1_df=management_bandit_observed_sample_view_v1_df,
        management_exit_local_all_eligible_scored_view_v1_df=management_exit_local_all_eligible_scored_view_v1_df,
        management_exit_local_benchmark_summary_v1=management_exit_local_benchmark_summary_v1,
        management_exit_local_status_v1=management_exit_local_status_v1,
        as_of_supervision_join_coverage_summary=as_of_supervision_join_coverage_summary,
        leakage_guard_summary=leakage_guard_summary,
    )
    manual_payload = shadow_meta._build_management_exit_local_manual_review_v1(
        management_exit_local_shadow_candidate_review_v1_df=shadow_payload[
            "management_exit_local_shadow_candidate_review_v1_df"
        ],
        management_bandit_exit_local_reward_view_v1_df=management_bandit_exit_local_reward_view_v1_df,
        management_exit_local_shadow_status_v1=shadow_payload["management_exit_local_shadow_status_v1"],
        as_of_supervision_join_coverage_summary=as_of_supervision_join_coverage_summary,
        leakage_guard_summary=leakage_guard_summary,
    )
    if (
        shadow_payload["management_exit_local_shadow_consistency_audit_v1_summary"]["overall_status_v1"]
        != "NO_ISSUE_FOUND"
    ):
        raise RuntimeError(
            "[MANAGEMENT_EXIT_LOCAL_SHADOW_MANUAL_REVIEW_V1] shadow audit consistency is not green; refusing to write"
        )
    if (
        manual_payload["management_exit_local_manual_review_consistency_audit_v1_summary"]["overall_status_v1"]
        != "NO_ISSUE_FOUND"
    ):
        raise RuntimeError(
            "[MANAGEMENT_EXIT_LOCAL_SHADOW_MANUAL_REVIEW_V1] manual review consistency is not green; refusing to write"
        )

    _ensure_append_only_dir(shadow_dir)
    _ensure_append_only_dir(manual_dir)

    shadow_contract_path = shadow_dir / shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_SHADOW_AUDIT_CONTRACT_V1
    shadow_bucket_contract_path = shadow_dir / shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_SHADOW_BUCKET_CONTRACT_V1
    shadow_row_view_path = shadow_dir / shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_SHADOW_ROW_VIEW_V1_PARQUET
    shadow_in_domain_bucket_eval_path = (
        shadow_dir / shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_SHADOW_IN_DOMAIN_BUCKET_EVAL_V1_CSV
    )
    shadow_hold_research_view_path = (
        shadow_dir / shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_SHADOW_HOLD_RESEARCH_VIEW_V1_PARQUET
    )
    shadow_hold_research_summary_path = (
        shadow_dir / shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_SHADOW_HOLD_RESEARCH_SUMMARY_V1
    )
    shadow_support_boundary_audit_path = (
        shadow_dir / shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_SHADOW_SUPPORT_BOUNDARY_AUDIT_V1_CSV
    )
    shadow_candidate_review_path = (
        shadow_dir / shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_SHADOW_CANDIDATE_REVIEW_V1_PARQUET
    )
    shadow_consistency_audit_path = (
        shadow_dir / shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_SHADOW_CONSISTENCY_AUDIT_V1_CSV
    )
    shadow_status_path = shadow_dir / shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_SHADOW_STATUS_V1
    shadow_pack_manifest_path = (
        shadow_dir / shadow_meta._ALL_TRADE_REVIEW_DECOMPOSED_MANAGEMENT_EXIT_LOCAL_SHADOW_PACK_MANIFEST_V1
    )
    shadow_pack_index_path = (
        shadow_dir / shadow_meta._ALL_TRADE_REVIEW_DECOMPOSED_MANAGEMENT_EXIT_LOCAL_SHADOW_PACK_INDEX_V1_PARQUET
    )
    shadow_build_summary_path = shadow_dir / "shadow_meta_all_trade_review_ledger_build_summary.json"

    manual_contract_path = manual_dir / shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_MANUAL_REVIEW_CONTRACT_V1
    manual_queue_path = manual_dir / shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_MANUAL_REVIEW_QUEUE_V1_CSV
    manual_casebook_path = manual_dir / shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_MANUAL_REVIEW_CASEBOOK_V1_PARQUET
    manual_feature_position_view_path = (
        manual_dir / shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_MANUAL_REVIEW_FEATURE_POSITION_VIEW_V1_PARQUET
    )
    manual_train_exit_neighbors_path = (
        manual_dir / shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_MANUAL_REVIEW_TRAIN_EXIT_NEIGHBORS_V1_PARQUET
    )
    manual_summary_path = manual_dir / shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_MANUAL_REVIEW_SUMMARY_V1
    manual_consistency_audit_path = (
        manual_dir / shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_MANUAL_REVIEW_CONSISTENCY_AUDIT_V1_CSV
    )
    manual_status_path = manual_dir / shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_MANUAL_REVIEW_STATUS_V1
    manual_pack_manifest_path = (
        manual_dir / shadow_meta._ALL_TRADE_REVIEW_DECOMPOSED_MANAGEMENT_EXIT_LOCAL_MANUAL_REVIEW_PACK_MANIFEST_V1
    )
    manual_pack_index_path = (
        manual_dir / shadow_meta._ALL_TRADE_REVIEW_DECOMPOSED_MANAGEMENT_EXIT_LOCAL_MANUAL_REVIEW_PACK_INDEX_V1_PARQUET
    )
    manual_build_summary_path = manual_dir / "shadow_meta_all_trade_review_ledger_build_summary.json"

    _write_json(shadow_contract_path, shadow_payload["management_exit_local_shadow_audit_contract_v1"])
    _write_json(shadow_bucket_contract_path, shadow_payload["management_exit_local_shadow_bucket_contract_v1"])
    shadow_payload["management_exit_local_shadow_row_view_v1_df"].to_parquet(shadow_row_view_path, index=False)
    shadow_payload["management_exit_local_shadow_in_domain_bucket_eval_v1_df"].to_csv(
        shadow_in_domain_bucket_eval_path, index=False
    )
    shadow_payload["management_exit_local_shadow_hold_research_view_v1_df"].to_parquet(
        shadow_hold_research_view_path, index=False
    )
    _write_json(
        shadow_hold_research_summary_path,
        shadow_payload["management_exit_local_shadow_hold_research_summary_v1"],
    )
    shadow_payload["management_exit_local_shadow_support_boundary_audit_v1_df"].to_csv(
        shadow_support_boundary_audit_path, index=False
    )
    shadow_payload["management_exit_local_shadow_candidate_review_v1_df"].to_parquet(
        shadow_candidate_review_path, index=False
    )
    shadow_payload["management_exit_local_shadow_consistency_audit_v1_df"].to_csv(
        shadow_consistency_audit_path, index=False
    )
    _write_json(shadow_status_path, shadow_payload["management_exit_local_shadow_status_v1"])
    _write_json(
        shadow_pack_manifest_path,
        shadow_payload["decomposed_management_exit_local_shadow_pack_manifest_v1"],
    )
    shadow_payload["decomposed_management_exit_local_shadow_pack_index_v1_df"].to_parquet(
        shadow_pack_index_path, index=False
    )

    _write_json(manual_contract_path, manual_payload["management_exit_local_manual_review_contract_v1"])
    manual_payload["management_exit_local_manual_review_queue_v1_df"].to_csv(manual_queue_path, index=False)
    manual_payload["management_exit_local_manual_review_casebook_v1_df"].to_parquet(
        manual_casebook_path, index=False
    )
    manual_payload["management_exit_local_manual_review_feature_position_view_v1_df"].to_parquet(
        manual_feature_position_view_path, index=False
    )
    manual_payload["management_exit_local_manual_review_train_exit_neighbors_v1_df"].to_parquet(
        manual_train_exit_neighbors_path, index=False
    )
    _write_json(manual_summary_path, manual_payload["management_exit_local_manual_review_summary_v1"])
    manual_payload["management_exit_local_manual_review_consistency_audit_v1_df"].to_csv(
        manual_consistency_audit_path, index=False
    )
    _write_json(manual_status_path, manual_payload["management_exit_local_manual_review_status_v1"])
    _write_json(
        manual_pack_manifest_path,
        manual_payload["decomposed_management_exit_local_manual_review_pack_manifest_v1"],
    )
    manual_payload["decomposed_management_exit_local_manual_review_pack_index_v1_df"].to_parquet(
        manual_pack_index_path, index=False
    )

    artifact_paths = {
        "closed_trades_path": str(
            _artifact_path(runtime_ledger_dir, runtime_build_summary, "closed_trades_path", "shadow_meta_all_trade_review_ledger_closed_trades.parquet")
        ),
        "hindsight_review_export_path": str(
            _artifact_path(runtime_ledger_dir, runtime_build_summary, "hindsight_review_export_path", shadow_meta._ALL_TRADE_REVIEW_HINDSIGHT_EXPORT_PARQUET)
        ),
        "as_of_decision_moment_ledger_path": str(
            _artifact_path(runtime_ledger_dir, runtime_build_summary, "as_of_decision_moment_ledger_path", "shadow_meta_all_trade_review_as_of_decision_moment_ledger_v1.parquet")
        ),
        "policy_action_supervision_join_path": str(
            _artifact_path(runtime_ledger_dir, runtime_build_summary, "policy_action_supervision_join_path", shadow_meta._ALL_TRADE_REVIEW_POLICY_ACTION_SUPERVISION_JOIN_PARQUET)
        ),
        "management_anchor_raw_state_path": str(
            _artifact_path(runtime_ledger_dir, runtime_build_summary, "management_anchor_raw_state_path", shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_ANCHOR_RAW_STATE_V1)
        ),
        "management_bandit_observed_sample_view_path": str(
            _artifact_path(runtime_ledger_dir, runtime_build_summary, "management_bandit_observed_sample_view_path", shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_BANDIT_OBSERVED_SAMPLE_VIEW_V1_PARQUET)
        ),
        "management_bandit_direct_method_candidate_view_path": str(
            _artifact_path(runtime_ledger_dir, runtime_build_summary, "management_bandit_direct_method_candidate_view_path", shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_BANDIT_DIRECT_METHOD_CANDIDATE_VIEW_V1_PARQUET)
        ),
        "management_bandit_action_reward_contract_path": str(
            _artifact_path(runtime_ledger_dir, runtime_build_summary, "management_bandit_action_reward_contract_path", shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_BANDIT_ACTION_REWARD_CONTRACT_V1)
        ),
        "management_bandit_observed_action_contract_path": str(
            _artifact_path(runtime_ledger_dir, runtime_build_summary, "management_bandit_observed_action_contract_path", shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_BANDIT_OBSERVED_ACTION_CONTRACT_V1)
        ),
        "management_bandit_status_path": str(
            _artifact_path(runtime_ledger_dir, runtime_build_summary, "management_bandit_status_path", shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_BANDIT_STATUS_V1)
        ),
        "management_exit_local_all_eligible_scored_view_path": str(
            _artifact_path(runtime_ledger_dir, runtime_build_summary, "management_exit_local_all_eligible_scored_view_path", shadow_meta._ALL_TRADE_REVIEW_MANAGEMENT_EXIT_LOCAL_ALL_ELIGIBLE_SCORED_VIEW_V1_PARQUET)
        ),
        "management_exit_local_shadow_row_view_path": str(shadow_row_view_path),
        "management_exit_local_shadow_candidate_review_path": str(shadow_candidate_review_path),
        "management_exit_local_shadow_hold_research_view_path": str(shadow_hold_research_view_path),
        "management_exit_local_shadow_status_path": str(shadow_status_path),
        "management_exit_local_manual_review_queue_path": str(manual_queue_path),
        "management_exit_local_manual_review_casebook_path": str(manual_casebook_path),
        "management_exit_local_manual_review_feature_position_view_path": str(manual_feature_position_view_path),
        "management_exit_local_manual_review_train_exit_neighbors_path": str(manual_train_exit_neighbors_path),
        "management_exit_local_manual_review_summary_path": str(manual_summary_path),
        "management_exit_local_manual_review_status_path": str(manual_status_path),
    }

    build_timestamp_utc_v1 = pd.Timestamp.utcnow().isoformat()
    shadow_stage_summary_v1 = {
        "stage_name_v1": "MANAGEMENT_EXIT_LOCAL_SHADOW_AUDIT_V1",
        "build_timestamp_utc_v1": build_timestamp_utc_v1,
        "runtime_ledger_dir_v1": str(runtime_ledger_dir),
        "shadow_dir_v1": str(shadow_dir),
        "manual_dir_v1": str(manual_dir),
        "hold_research_row_count_v1": int(
            shadow_payload["management_exit_local_shadow_hold_research_summary_v1"]["hold_research_row_count_v1"]
        ),
        "high_score_hold_count_v1": int(
            shadow_payload["management_exit_local_shadow_hold_research_summary_v1"]["high_score_hold_count_v1"]
        ),
        "consistency_status_v1": shadow_payload["management_exit_local_shadow_consistency_audit_v1_summary"][
            "overall_status_v1"
        ],
        "consistency_failed_check_count_v1": int(
            shadow_payload["management_exit_local_shadow_consistency_audit_v1_summary"]["failed_check_count_v1"]
        ),
    }
    manual_stage_summary_v1 = {
        "stage_name_v1": "MANAGEMENT_EXIT_LOCAL_MANUAL_REVIEW_V1",
        "build_timestamp_utc_v1": build_timestamp_utc_v1,
        "runtime_ledger_dir_v1": str(runtime_ledger_dir),
        "shadow_dir_v1": str(shadow_dir),
        "manual_dir_v1": str(manual_dir),
        "total_candidates_v1": int(
            manual_payload["management_exit_local_manual_review_summary_v1"]["total_candidates_v1"]
        ),
        "consistency_status_v1": manual_payload[
            "management_exit_local_manual_review_consistency_audit_v1_summary"
        ]["overall_status_v1"],
        "consistency_failed_check_count_v1": int(
            manual_payload["management_exit_local_manual_review_consistency_audit_v1_summary"][
                "failed_check_count_v1"
            ]
        ),
    }
    _write_json(
        shadow_build_summary_path,
        _build_shared_build_summary(
            truth_root=truth_root,
            runtime_ledger_dir=runtime_ledger_dir,
            runtime_build_summary=runtime_build_summary,
            artifact_paths=artifact_paths,
            stage_name_v1="MANAGEMENT_EXIT_LOCAL_SHADOW_AUDIT_V1",
            stage_summary_v1=shadow_stage_summary_v1,
        ),
    )
    _write_json(
        manual_build_summary_path,
        _build_shared_build_summary(
            truth_root=truth_root,
            runtime_ledger_dir=runtime_ledger_dir,
            runtime_build_summary=runtime_build_summary,
            artifact_paths=artifact_paths,
            stage_name_v1="MANAGEMENT_EXIT_LOCAL_MANUAL_REVIEW_V1",
            stage_summary_v1=manual_stage_summary_v1,
        ),
    )

    result = {
        "truth_root": str(truth_root),
        "runtime_ledger_dir": str(runtime_ledger_dir),
        "shadow_dir": str(shadow_dir),
        "manual_dir": str(manual_dir),
        "shadow_consistency_status_v1": shadow_stage_summary_v1["consistency_status_v1"],
        "shadow_hold_research_row_count_v1": shadow_stage_summary_v1["hold_research_row_count_v1"],
        "shadow_high_score_hold_count_v1": shadow_stage_summary_v1["high_score_hold_count_v1"],
        "manual_consistency_status_v1": manual_stage_summary_v1["consistency_status_v1"],
        "manual_total_candidates_v1": manual_stage_summary_v1["total_candidates_v1"],
    }
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
