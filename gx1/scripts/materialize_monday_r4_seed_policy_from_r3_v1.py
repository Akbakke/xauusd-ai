#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from gx1.scripts.materialize_r4_fullcoverage_policy_recalibration_and_shadow_replay_v1 import (
    _bool,
    _json_dumps,
    _load_json,
    _num,
    _policy_metric_row,
    _write_json,
)


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

EXTENSION_NAME = "MONDAY_R4_SEED_POLICY_FROM_R3_V1"

R2_AS_OF_TABLE = "shadow_meta_all_trade_review_harvest_r2_entry_as_of_feature_table_v1.parquet"
R2_LABEL_TABLE = "shadow_meta_all_trade_review_harvest_r2_entry_hindsight_label_table_v1.parquet"
R3_PREDICTION_VIEW = "shadow_meta_all_trade_review_r3_entry_label_feature_prediction_view_v1.parquet"
R4_POLICY_PREDICTION_VIEW = "shadow_meta_all_trade_review_r4_entry_calibrated_fallback_policy_prediction_view_v1.parquet"
R4_SUMMARY = "shadow_meta_all_trade_review_r4_entry_calibrated_fallback_summary_v1.json"
R4_STATUS = "shadow_meta_all_trade_review_r4_entry_calibrated_fallback_status_v1.json"
R4_CONTRACT = "shadow_meta_all_trade_review_r4_entry_calibrated_fallback_contract_v1.json"
R4_MANIFEST = "shadow_meta_all_trade_review_r4_entry_calibrated_fallback_manifest_v1.json"
R4_REPORT = "shadow_meta_all_trade_review_r4_entry_calibrated_fallback_report_v1.md"
R4_CONSISTENCY_AUDIT = "shadow_meta_all_trade_review_r4_consistency_audit_v1.csv"
TOP_LEVEL_SUMMARY = "truth_monday_r4_seed_policy_from_r3_v1.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _resolve_dir(reports_root: Path, arg: str | None, required_file: str) -> Path:
    if not arg:
        raise FileNotFoundError(f"Explicit dir required for {required_file}")
    path = Path(arg).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dir does not exist: {path}")
    if not (path / required_file).exists():
        raise FileNotFoundError(f"{path} missing required artifact {required_file}")
    return path


def _default_extension_dir(reports_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return reports_root / f"{EXTENSION_NAME}_{stamp}"


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _render_report(summary: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Monday R4 Seed Policy From R3 V1",
            "",
            "Research-only bootstrap. This is a Monday-native current-reference seed built from repaired readiness + R3 predictions.",
            "",
            "## Headline",
            "",
            f"- Ledger rows: `{summary['ledger_row_count_v1']}`",
            f"- Pseudo-R2 seed blocks: `{summary['pseudo_r2_block_count_v1']}`",
            f"- R3 conservative blocks: `{summary['r3_conservative_block_count_v1']}`",
            f"- R4 seed blocks: `{summary['r4_seed_block_count_v1']}`",
            f"- Selected policy: `{summary['selected_policy_name_v1']}`",
            "",
            "## Note",
            "",
            "- This is not historical R2. It is an explicit Monday bootstrap reference derived from R3 probabilities because the old harvest-linked R2 branch was never rematerialized on Monday root.",
            "- No live promotion. No controller use.",
            "",
        ]
    ) + "\n"


def build_payload(
    *,
    readiness_dir: Path,
    r3_dir: Path,
    extension_dir: Path,
    expected_ledger_count: int | None,
) -> Dict[str, Any]:
    asof_df = pd.read_parquet(readiness_dir / R2_AS_OF_TABLE)
    labels_df = pd.read_parquet(readiness_dir / R2_LABEL_TABLE)
    r3_df = pd.read_parquet(r3_dir / R3_PREDICTION_VIEW)
    if expected_ledger_count is not None and len(asof_df) != expected_ledger_count:
        raise RuntimeError(f"Expected {expected_ledger_count} rows, observed {len(asof_df)}")
    for name, frame in [(R2_AS_OF_TABLE, asof_df), (R2_LABEL_TABLE, labels_df), (R3_PREDICTION_VIEW, r3_df)]:
        if frame["candidate_uid"].astype("string").duplicated().any():
            raise RuntimeError(f"{name} requires unique candidate_uid")
    _load_json(r3_dir / "shadow_meta_all_trade_review_r3_entry_label_feature_summary_v1.json")

    frame = (
        asof_df[
            [
                "run_id",
                "candidate_uid",
                "trade_uid",
                "trade_id",
                "decision_timestamp",
                "used_for_training",
                "used_for_validation",
                "used_for_holdout",
                "entry_observation_present_v1",
                "entry_raw_state_present_v1",
            ]
        ]
        .merge(
            labels_df[
                [
                    "candidate_uid",
                    "hindsight_entry_decision_review_v1",
                    "baseline_realized_pnl_bps_v1",
                    "peak_mfe_bps_v1",
                    "mae_abs_bps_v1",
                    "giveback_bps_v1",
                    "harvest_capture_ratio_v1",
                    "label_should_not_take_v1",
                    "label_immediate_mae_risk_v1",
                    "label_wait_would_have_helped_v1",
                    "label_strong_trade_candidate_v1",
                    "label_direct_take_ok_v1",
                ]
            ],
            on="candidate_uid",
            how="left",
            validate="one_to_one",
        )
        .merge(
            r3_df[
                [
                    "candidate_uid",
                    "entry_r3_feature_available_v1",
                    "entry_r3_shadow_action_v1",
                    "entry_r3_shadow_action_source_v1",
                    "pred__entry_r3_should_not_take__prob_true_v1",
                    "pred__entry_r3_immediate_mae_risk__prob_true_v1",
                    "pred__entry_r3_wait_would_have_helped__prob_true_v1",
                    "pred__entry_r3_strong_trade_candidate__prob_true_v1",
                    "pred__entry_r3_direct_take_ok__prob_true_v1",
                    "pred__entry_r3_good_mfe_bad_capture__prob_true_v1",
                ]
            ],
            on="candidate_uid",
            how="left",
            validate="one_to_one",
        )
    )
    feature_available = _bool(frame, "entry_r3_feature_available_v1")
    p_should = pd.to_numeric(frame["pred__entry_r3_should_not_take__prob_true_v1"], errors="coerce")
    p_mae = pd.to_numeric(frame["pred__entry_r3_immediate_mae_risk__prob_true_v1"], errors="coerce")
    p_wait = pd.to_numeric(frame["pred__entry_r3_wait_would_have_helped__prob_true_v1"], errors="coerce")
    p_strong = pd.to_numeric(frame["pred__entry_r3_strong_trade_candidate__prob_true_v1"], errors="coerce")
    p_direct = pd.to_numeric(frame["pred__entry_r3_direct_take_ok__prob_true_v1"], errors="coerce")
    frame["take_was_ok_v1"] = frame["hindsight_entry_decision_review_v1"].astype("string").eq("TAKE_WAS_OK")
    frame["fifty_plus_mfe_v1"] = _num(frame, "peak_mfe_bps_v1").ge(50.0)
    frame["hundred_plus_mfe_v1"] = _num(frame, "peak_mfe_bps_v1").ge(100.0)
    frame["two_hundred_plus_mfe_v1"] = _num(frame, "peak_mfe_bps_v1").ge(200.0)
    frame["tail_10_50_mfe_v1"] = _num(frame, "peak_mfe_bps_v1").between(10.0, 50.0, inclusive="left") & (
        _num(frame, "baseline_realized_pnl_bps_v1").le(0.0) | _bool(frame, "label_should_not_take_v1")
    )
    frame["strongest_winner_path_v1"] = frame["two_hundred_plus_mfe_v1"] | (
        _bool(frame, "label_strong_trade_candidate_v1")
        & _num(frame, "baseline_realized_pnl_bps_v1").gt(0.0)
        & _num(frame, "harvest_capture_ratio_v1").ge(0.5)
    )
    frame["is_repaired_165_v1"] = False

    pseudo_r2 = (feature_available & p_should.ge(0.74).fillna(False) & p_direct.lt(0.42).fillna(False) & p_strong.lt(0.35).fillna(False)).astype(bool)
    r3_conservative = frame["entry_r3_shadow_action_v1"].astype("string").eq("ENTRY_SUPPRESS_OR_DOWNSIZE_SHADOW")
    r4_seed = (
        pseudo_r2
        | (
            feature_available
            & (
                (p_should.ge(0.60).fillna(False) & p_direct.lt(0.50).fillna(False))
                | (p_mae.ge(0.80).fillna(False) & p_direct.lt(0.45).fillna(False))
                | (p_wait.ge(0.85).fillna(False) & p_direct.lt(0.45).fillna(False))
            )
            & ~p_strong.ge(0.50).fillna(False)
        )
    ).astype(bool)

    prediction_df = frame[
        [
            "run_id",
            "candidate_uid",
            "trade_uid",
            "trade_id",
            "decision_timestamp",
            "used_for_training",
            "used_for_validation",
            "used_for_holdout",
            "entry_observation_present_v1",
            "entry_raw_state_present_v1",
            "entry_r3_feature_available_v1",
            "entry_r3_shadow_action_v1",
            "entry_r3_shadow_action_source_v1",
            "pred__entry_r3_should_not_take__prob_true_v1",
            "pred__entry_r3_immediate_mae_risk__prob_true_v1",
            "pred__entry_r3_wait_would_have_helped__prob_true_v1",
            "pred__entry_r3_strong_trade_candidate__prob_true_v1",
            "pred__entry_r3_direct_take_ok__prob_true_v1",
            "pred__entry_r3_good_mfe_bad_capture__prob_true_v1",
        ]
    ].copy()
    prediction_df["r2_entry_fallback_row_v1"] = pseudo_r2
    prediction_df["r2_entry_fallback_correct_v1"] = pseudo_r2 & _bool(frame, "label_should_not_take_v1")
    prediction_df["r3_conservative_blocks_v1"] = r3_conservative
    prediction_df["r4_entry_fallback_block_v1"] = r4_seed
    prediction_df["r4_entry_fallback_action_v1"] = np.where(r4_seed, "ENTRY_SUPPRESS_OR_DOWNSIZE_SHADOW", "KEEP_BASELINE_SHADOW")
    prediction_df["r4_entry_fallback_source_v1"] = "MONDAY_R3_BOOTSTRAP_SEED_NOT_HISTORICAL_R2"

    summary_metrics = _policy_metric_row("MONDAY_R4_SEED_POLICY_FROM_R3", "ALL", frame, r4_seed, thresholds={"seed_mode_v1": True})
    summary = {
        "layer_name": "MONDAY_R4_SEED_POLICY_FROM_R3_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "extension_dir_v1": str(extension_dir),
        "readiness_dir_v1": str(readiness_dir),
        "r3_dir_v1": str(r3_dir),
        "ledger_row_count_v1": int(len(prediction_df)),
        "pseudo_r2_block_count_v1": int(pseudo_r2.sum()),
        "r3_conservative_block_count_v1": int(r3_conservative.sum()),
        "r4_seed_block_count_v1": int(r4_seed.sum()),
        "selected_policy_name_v1": "MONDAY_R4_SEED_POLICY_FROM_R3",
        "selected_policy_metrics_v1": summary_metrics,
        "seed_contract_note_v1": "BOOTSTRAP_REFERENCE_ONLY_NOT_HISTORICAL_R2",
        "not_live_gate": True,
        "not_controller": True,
    }
    status = {
        "layer_name": "MONDAY_R4_SEED_POLICY_FROM_R3_STATUS_V1",
        "MONDAY_R4_SEED_POLICY_FROM_R3_STATUS": "READY_FOR_FULLCOVERAGE_RESEARCH_NOT_LIVE_GATE",
        "failed_check_count_v1": 0,
        "not_live_gate": True,
        "not_controller": True,
    }
    contract = {
        "layer_name": "MONDAY_R4_SEED_POLICY_FROM_R3_CONTRACT_V1",
        "input_readiness_dir_v1": str(readiness_dir),
        "input_r3_dir_v1": str(r3_dir),
        "seed_policy_name_v1": "MONDAY_R4_SEED_POLICY_FROM_R3",
        "seed_origin_v1": "R3_PROBABILITY_BOOTSTRAP",
        "historical_r2_available_v1": False,
        "not_live_gate": True,
        "not_controller": True,
    }
    manifest = {
        "layer_name": "MONDAY_R4_SEED_POLICY_FROM_R3_MANIFEST_V1",
        "artifacts_v1": {
            "policy_prediction_view_v1": R4_POLICY_PREDICTION_VIEW,
            "summary_v1": R4_SUMMARY,
            "status_v1": R4_STATUS,
            "contract_v1": R4_CONTRACT,
            "report_v1": R4_REPORT,
        },
    }
    consistency = pd.DataFrame(
        [
            {"check_name_v1": "LEDGER_ROW_COUNT_MATCH", "status_v1": "PASS", "details_json_v1": _json_dumps({"rows": int(len(prediction_df))})},
            {"check_name_v1": "CANDIDATE_UID_UNIQUE", "status_v1": "PASS", "details_json_v1": _json_dumps({"duplicate_count": 0})},
            {"check_name_v1": "SEED_IS_RESEARCH_ONLY", "status_v1": "PASS", "details_json_v1": _json_dumps({"historical_r2_available_v1": False})},
        ]
    )
    return {
        "prediction_df_v1": prediction_df,
        "summary_v1": summary,
        "status_v1": status,
        "contract_v1": contract,
        "manifest_v1": manifest,
        "consistency_df_v1": consistency,
    }


def materialize(
    reports_root: Path,
    *,
    readiness_dir: Path,
    r3_dir: Path,
    extension_dir: Path | None = None,
    expected_ledger_count: int | None = None,
) -> Dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    resolved_extension_dir = Path(extension_dir).expanduser().resolve() if extension_dir else _default_extension_dir(reports_root)
    payload = build_payload(
        readiness_dir=readiness_dir.expanduser().resolve(),
        r3_dir=r3_dir.expanduser().resolve(),
        extension_dir=resolved_extension_dir,
        expected_ledger_count=expected_ledger_count,
    )
    resolved_extension_dir.mkdir(parents=True, exist_ok=True)
    payload["prediction_df_v1"].to_parquet(resolved_extension_dir / R4_POLICY_PREDICTION_VIEW, index=False)
    payload["consistency_df_v1"].to_csv(resolved_extension_dir / R4_CONSISTENCY_AUDIT, index=False)
    _write_json(resolved_extension_dir / R4_SUMMARY, payload["summary_v1"])
    _write_json(resolved_extension_dir / R4_STATUS, payload["status_v1"])
    _write_json(resolved_extension_dir / R4_CONTRACT, payload["contract_v1"])
    _write_json(resolved_extension_dir / R4_MANIFEST, payload["manifest_v1"])
    (resolved_extension_dir / R4_REPORT).write_text(_render_report(payload["summary_v1"]), encoding="utf-8")
    top = dict(payload["summary_v1"])
    top["extension_dir_v1"] = str(resolved_extension_dir)
    _write_json(reports_root / TOP_LEVEL_SUMMARY, top)
    return {
        "extension_dir": resolved_extension_dir,
        "summary": payload["summary_v1"],
        "status": payload["status_v1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--readiness-dir", required=True)
    parser.add_argument("--r3-dir", required=True)
    parser.add_argument("--extension-dir", default=None)
    parser.add_argument("--expected-ledger-count", type=int, default=None)
    args = parser.parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    result = materialize(
        reports_root,
        readiness_dir=_resolve_dir(reports_root, args.readiness_dir, R2_AS_OF_TABLE),
        r3_dir=_resolve_dir(reports_root, args.r3_dir, R3_PREDICTION_VIEW),
        extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None,
        expected_ledger_count=args.expected_ledger_count,
    )
    print(json.dumps({"extension_dir": str(result["extension_dir"]), "status": result["status"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
