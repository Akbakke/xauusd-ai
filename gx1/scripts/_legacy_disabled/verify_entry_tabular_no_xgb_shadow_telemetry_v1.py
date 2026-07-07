"""Verify journal telemetry from tabular no-XGB shadow-only observation."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


SHADOW_REVIEW_TEMPLATE = "/home/andre2/src/GX1_ENGINE/docs/ENTRY_NEXT_EDGE_SHADOW_REVIEW_TEMPLATE_20260627.md"
EXPECTED_CANDIDATE_ID = "entry_tabular_no_xgb_top5_v1_20260627"
EXPECTED_FEATURE_CONTRACT_HASH = "1d11fce818060ad5aeaabc0c00b369d22d75bc741859bbdf4b0eb03c9743c573"
EXPECTED_SCORE_THRESHOLD = 0.39048198845884335
OFFLINE_EXPECTED_WOULD_TAKE_RATE = 0.05921197446621176
CONTRACT_OVERRIDE_ACK_ENV = "GX1_ALLOW_ENTRY_SHADOW_CONTRACT_OVERRIDE"
CONTRACT_OVERRIDE_ACK_VALUE = "20260627_ALLOW_ENTRY_SHADOW_CONTRACT_OVERRIDE"
MISSING_FIELDS_OVERRIDE_ACK_ENV = "GX1_ALLOW_ENTRY_SHADOW_MISSING_FIELDS"
MISSING_FIELDS_OVERRIDE_ACK_VALUE = "20260627_ALLOW_ENTRY_SHADOW_MISSING_FIELDS"
REQUIRED_DECISION_FIELDS = [
    "shadow_no_xgb_action",
    "shadow_no_xgb_side",
    "shadow_no_xgb_score",
    "shadow_no_xgb_score_threshold",
    "shadow_no_xgb_chosen_prob",
    "shadow_no_xgb_p_long",
    "shadow_no_xgb_p_short",
    "shadow_no_xgb_p_flat",
    "shadow_no_xgb_candidate_id",
    "shadow_no_xgb_feature_contract_hash",
]
FORBIDDEN_ORDER_STATUSES = {
    "filled",
    "netted",
    "rejected",
    "api_error",
    "DRY_RUN",
    "EXIT_NOW",
    "FORCED_CLOSE_24H",
    "TRADE_OPENED_OANDA",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _contract_override_requested(args: argparse.Namespace) -> bool:
    return (
        str(args.expected_candidate_id) != EXPECTED_CANDIDATE_ID
        or str(args.expected_feature_contract_hash) != EXPECTED_FEATURE_CONTRACT_HASH
        or not math.isclose(
            float(args.expected_score_threshold),
            EXPECTED_SCORE_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(args.offline_expected_would_take_rate),
            OFFLINE_EXPECTED_WOULD_TAKE_RATE,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )


def _enforce_contract_override_ack(args: argparse.Namespace) -> bool:
    if not _contract_override_requested(args):
        return False
    observed_ack = os.environ.get(CONTRACT_OVERRIDE_ACK_ENV, "")
    if observed_ack != CONTRACT_OVERRIDE_ACK_VALUE:
        raise RuntimeError(
            "shadow contract override blocked by active Entry next-edge plan "
            f"(set {CONTRACT_OVERRIDE_ACK_ENV}={CONTRACT_OVERRIDE_ACK_VALUE} only for controlled research review)"
        )
    return True


def _enforce_missing_fields_override_ack(args: argparse.Namespace, missing_rows: list[int]) -> bool:
    if not missing_rows or bool(args.require_all_rows):
        return False
    observed_ack = os.environ.get(MISSING_FIELDS_OVERRIDE_ACK_ENV, "")
    if observed_ack != MISSING_FIELDS_OVERRIDE_ACK_VALUE:
        raise RuntimeError(
            "shadow missing-fields override blocked by active Entry next-edge plan "
            f"(set {MISSING_FIELDS_OVERRIDE_ACK_ENV}={MISSING_FIELDS_OVERRIDE_ACK_VALUE} "
            "only for controlled malformed-journal research review)"
        )
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract_override_used = _enforce_contract_override_ack(args)
    journal_path = Path(args.journal).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    rows = _read_jsonl(journal_path)
    if not rows:
        raise RuntimeError(f"journal has no rows: {journal_path}")

    observed_statuses: dict[str, int] = {}
    missing_rows: list[int] = []
    forbidden_rows: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    disabled_reasons: dict[str, int] = {}
    for idx, row in enumerate(rows):
        status = str(row.get("order_status", ""))
        observed_statuses[status] = observed_statuses.get(status, 0) + 1
        if status in FORBIDDEN_ORDER_STATUSES or "order_details" in row or "close_order_details" in row:
            forbidden_rows.append({"row": idx, "order_status": status})
        decision = row.get("v12_decision") if isinstance(row.get("v12_decision"), dict) else {}
        if "shadow_no_xgb_disabled_reason" in decision:
            reason = str(decision["shadow_no_xgb_disabled_reason"])
            disabled_reasons[reason] = disabled_reasons.get(reason, 0) + 1
        missing = [field for field in REQUIRED_DECISION_FIELDS if field not in decision]
        if missing:
            missing_rows.append(idx)
            continue
        shadow_rows.append(decision)

    if forbidden_rows:
        raise RuntimeError(f"shadow journal contains forbidden order side-effects: {forbidden_rows[:20]}")
    if len(shadow_rows) < int(args.min_shadow_rows):
        raise RuntimeError(f"not enough shadow rows: {len(shadow_rows)} < {args.min_shadow_rows}")
    if missing_rows and bool(args.require_all_rows):
        raise RuntimeError(f"rows missing required shadow fields: first={missing_rows[:20]} count={len(missing_rows)}")
    missing_fields_override_used = _enforce_missing_fields_override_ack(args, missing_rows)
    if disabled_reasons:
        raise RuntimeError(f"shadow disabled during observation: {disabled_reasons}")

    expected_candidate_id = str(args.expected_candidate_id)
    expected_feature_hash = str(args.expected_feature_contract_hash)
    expected_threshold = float(args.expected_score_threshold)
    candidate_mismatch = [
        idx for idx, row in enumerate(shadow_rows)
        if str(row["shadow_no_xgb_candidate_id"]) != expected_candidate_id
    ]
    feature_hash_mismatch = [
        idx for idx, row in enumerate(shadow_rows)
        if str(row["shadow_no_xgb_feature_contract_hash"]) != expected_feature_hash
    ]
    threshold_mismatch = [
        idx for idx, row in enumerate(shadow_rows)
        if not math.isclose(
            float(row["shadow_no_xgb_score_threshold"]),
            expected_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if candidate_mismatch:
        raise RuntimeError(
            f"shadow rows with unexpected candidate_id: first={candidate_mismatch[:20]} "
            f"count={len(candidate_mismatch)} expected={expected_candidate_id}"
        )
    if feature_hash_mismatch:
        raise RuntimeError(
            f"shadow rows with unexpected feature_contract_hash: first={feature_hash_mismatch[:20]} "
            f"count={len(feature_hash_mismatch)} expected={expected_feature_hash}"
        )
    if threshold_mismatch:
        raise RuntimeError(
            f"shadow rows with unexpected score_threshold: first={threshold_mismatch[:20]} "
            f"count={len(threshold_mismatch)} expected={expected_threshold}"
        )

    scores = np.asarray([float(row["shadow_no_xgb_score"]) for row in shadow_rows], dtype=np.float64)
    takes = np.asarray([str(row["shadow_no_xgb_action"]) != "SKIP" for row in shadow_rows], dtype=bool)
    longs = np.asarray([str(row["shadow_no_xgb_action"]) == "TAKE_LONG_NOW" for row in shadow_rows], dtype=bool)
    shorts = np.asarray([str(row["shadow_no_xgb_action"]) == "TAKE_SHORT_NOW" for row in shadow_rows], dtype=bool)
    report = {
        "schema_version": "entry_tabular_no_xgb_shadow_telemetry_verification_v1",
        "status": "PASS",
        "journal": str(journal_path),
        "rows_total": len(rows),
        "shadow_rows": len(shadow_rows),
        "missing_shadow_rows": len(missing_rows),
        "order_status_counts": observed_statuses,
        "candidate_id": expected_candidate_id,
        "feature_contract_hash": expected_feature_hash,
        "score_threshold": expected_threshold,
        "contract_override_used": contract_override_used,
        "contract_override_ack_env": CONTRACT_OVERRIDE_ACK_ENV if contract_override_used else None,
        "missing_fields_override_used": missing_fields_override_used,
        "missing_fields_override_ack_env": MISSING_FIELDS_OVERRIDE_ACK_ENV if missing_fields_override_used else None,
        "all_shadow_rows_match_expected_candidate": True,
        "all_shadow_rows_match_expected_feature_contract_hash": True,
        "all_shadow_rows_match_expected_score_threshold": True,
        "would_take_rows": int(takes.sum()),
        "would_take_rate": float(takes.mean()),
        "offline_expected_would_take_rate": float(args.offline_expected_would_take_rate),
        "would_take_rate_delta_vs_offline": float(takes.mean()) - float(args.offline_expected_would_take_rate),
        "would_take_long_rows": int(longs.sum()),
        "would_take_short_rows": int(shorts.sum()),
        "mean_score": float(scores.mean()),
        "score_p95": float(np.percentile(scores, 95)),
        "score_p99": float(np.percentile(scores, 99)),
        "decision": "NO_LIVE_PIN_REVIEW_REQUIRED",
        "next_required_gate": "manual_review_shadow_telemetry_against_offline_expectations",
        "manual_review_template": SHADOW_REVIEW_TEMPLATE,
        "allowed_manual_decisions": [
            "ACCEPT_FOR_NEXT_REVIEW_GATE",
            "HOLD_FOR_MORE_SHADOW",
            "FAIL_TO_FEATURE_LABEL_OBJECTIVE_REDESIGN",
        ],
    }
    _write_json(out_dir / "shadow_telemetry_verification.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-shadow-rows", type=int, default=1)
    ap.add_argument("--require-all-rows", dest="require_all_rows", action="store_true", default=True)
    ap.add_argument("--allow-missing-shadow-fields", dest="require_all_rows", action="store_false")
    ap.add_argument("--expected-candidate-id", default=EXPECTED_CANDIDATE_ID)
    ap.add_argument("--expected-feature-contract-hash", default=EXPECTED_FEATURE_CONTRACT_HASH)
    ap.add_argument("--expected-score-threshold", type=float, default=EXPECTED_SCORE_THRESHOLD)
    ap.add_argument("--offline-expected-would-take-rate", type=float, default=OFFLINE_EXPECTED_WOULD_TAKE_RATE)
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
