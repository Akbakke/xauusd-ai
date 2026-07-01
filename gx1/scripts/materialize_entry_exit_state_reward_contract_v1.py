#!/usr/bin/env python3
"""Materialize active Entry-bound Exit state/reward contract.

This is a report-only/data-contract gate. It reads the active per-bar
reconstruction audit and emits HOLD/EXIT_NOW state, action, reward and next-row
semantics for later Exit research. It does not train Exit, run replay, distill
IQL, or open shadow/live/promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.materialize_entry_exit_per_bar_handoff_v1 import (
    ENTRY_ALIGNMENT_CTX_CONT_FEATURES,
    ENTRY_ALIGNMENT_SNAP_FEATURES,
    ENTRY_SMART_ALIGNMENT_STATE_FEATURES,
    SPECIALIST_GATE_OUTPUT_FIELDS,
    _safe_feature_field,
)
from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


DEFAULT_RECONSTRUCTION_AUDIT_JSON = (
    REPORTS_ROOT
    / "entry_exit_per_bar_reconstruction_audit_20260630_v1/ENTRY_EXIT_PER_BAR_RECONSTRUCTION_AUDIT_latest.json"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_exit_state_reward_contract_20260630_v1"

READY_RECONSTRUCTION_DECISION = "READY_FOR_EXIT_STATE_REWARD_CONTRACT_REVIEW"
READY_DECISION = "ENTRY_EXIT_STATE_REWARD_CONTRACT_READY"
BLOCKED_DECISION = "BLOCKED_BY_EXIT_STATE_REWARD_CONTRACT"

ACTION_ID = {"HOLD": 0, "EXIT_NOW": 1}
ENTRY_ALIGNMENT_STATE_FEATURES = tuple(
    [
        *(_safe_feature_field("entry_snap", field) for field in ENTRY_ALIGNMENT_SNAP_FEATURES),
        *(_safe_feature_field("entry_ctx", field) for field in ENTRY_ALIGNMENT_CTX_CONT_FEATURES),
    ]
)
BASE_SPECIALIST_GATE_NAMES = (
    "structure_swing_encoder",
    "smc_liquidity_encoder",
    "trend_ema_encoder",
    "vol_compression_encoder",
    "momentum_flow_encoder",
    "session_regime_encoder",
)
CHALLENGER_SPECIALIST_GATE_NAMES = (
    "chart_geometry_encoder",
    "price_action_candle_encoder",
)
BASE_SPECIALIST_GATE_STATE_FEATURES = tuple(SPECIALIST_GATE_OUTPUT_FIELDS[name] for name in BASE_SPECIALIST_GATE_NAMES)
CHALLENGER_SPECIALIST_GATE_STATE_FEATURES = tuple(
    SPECIALIST_GATE_OUTPUT_FIELDS[name] for name in CHALLENGER_SPECIALIST_GATE_NAMES
)
STATE_FEATURES = (
    "running_pnl_bps",
    "running_mfe_bps",
    "running_mae_bps",
    "running_giveback_bps",
    "bars_held",
    "spread_bps",
    "atr_bps",
    "session",
    "vol_regime",
    "side",
    "entry_score",
    "entry_p_long",
    "entry_p_short",
    "entry_p_flat",
    "entry_path_quality_pred",
    "entry_bad_path_prob",
    *ENTRY_ALIGNMENT_STATE_FEATURES,
    *BASE_SPECIALIST_GATE_STATE_FEATURES,
)
NUMERIC_STATE_FEATURES = (
    "running_pnl_bps",
    "running_mfe_bps",
    "running_mae_bps",
    "running_giveback_bps",
    "bars_held",
    "spread_bps",
    "atr_bps",
    "entry_score",
    "entry_p_long",
    "entry_p_short",
    "entry_p_flat",
    "entry_path_quality_pred",
    "entry_bad_path_prob",
    *ENTRY_ALIGNMENT_STATE_FEATURES,
    *BASE_SPECIALIST_GATE_STATE_FEATURES,
)
STATE_PROVENANCE_FIELDS = (
    "entry_trade_id",
    "bar_ts",
    "bar_index",
    "bar_price_source",
    "bar_price_source_path",
    "entry_candidate_bundle_dir",
    "entry_iql_policy_id",
    "entry_replay_identity_hash",
)
FORBIDDEN_STATE_FIELDS = (
    "exit_time",
    "realized_net_pnl_bps",
    "realized_gross_pnl_bps",
    "realized_mfe_bps",
    "realized_mae_bps",
    "realized_exit_reason",
    "is_realized_exit_bar",
    "exit_now_label",
    "hold_reward_bps",
    "exit_now_reward_bps",
    "logged_reward_bps",
    "next_exit_episode_id",
    "next_exit_timestep",
    "future_",
    "hazard_",
    "oracle_",
    "positive_mfe_stopout",
)
REWARD_FIELDS = (
    "hold_reward_bps",
    "forced_terminal_hold_reward_bps",
    "exit_now_reward_bps",
    "logged_reward_bps",
    "terminal_reward_realized_net_pnl_bps",
    "exit_now_mfe_capture_ratio_reward",
    "exit_now_mae_penalty_reward_bps",
    "exit_now_giveback_penalty_reward_bps",
    "exit_now_transparent_combined_reward_bps",
    "future_max_running_pnl_bps",
    "future_min_running_pnl_bps",
    "future_best_exit_lift_bps",
    "future_adverse_excursion_bps",
    "future_giveback_from_peak_bps",
    "exit_hazard_adverse_15bps_label",
    "exit_hazard_giveback_20bps_label",
    "positive_mfe_stopout_episode_label",
    "oracle_exit_before_giveback_label",
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _check(name: str, condition: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(condition), "details": details or {}}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _path_from_report(report: dict[str, Any], key: str) -> Path:
    raw = str(report.get(key) or "").strip()
    return Path(raw).expanduser().resolve() if raw else Path("")


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _finite_review(frame: pd.DataFrame, fields: tuple[str, ...]) -> dict[str, Any]:
    field_reviews: dict[str, dict[str, Any]] = {}
    all_finite = True
    for field in fields:
        if field not in frame.columns:
            all_finite = False
            field_reviews[field] = {"present": False, "finite": False}
            continue
        values = pd.to_numeric(frame[field], errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype="float64", na_value=np.nan))
        missing_count = int(len(values) - int(finite.sum()))
        field_ok = missing_count == 0
        all_finite = all_finite and field_ok
        field_reviews[field] = {
            "present": True,
            "finite": field_ok,
            "missing_or_nonfinite_count": missing_count,
            "min": float(values.min()) if field_ok and len(values) else None,
            "max": float(values.max()) if field_ok and len(values) else None,
        }
    return {"all_finite": bool(all_finite), "fields": field_reviews}


def _active_state_features(source: pd.DataFrame) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    source_columns = set(str(column) for column in source.columns)
    present_challenger = [field for field in CHALLENGER_SPECIALIST_GATE_STATE_FEATURES if field in source_columns]
    challenger_complete = len(present_challenger) in {0, len(CHALLENGER_SPECIALIST_GATE_STATE_FEATURES)}
    present_smart_alignment = [field for field in ENTRY_SMART_ALIGNMENT_STATE_FEATURES if field in source_columns]
    active_state_features = list(STATE_FEATURES)
    if present_challenger:
        active_state_features.extend(CHALLENGER_SPECIALIST_GATE_STATE_FEATURES)
    if present_smart_alignment:
        active_state_features.extend(present_smart_alignment)
    active_numeric_features = list(NUMERIC_STATE_FEATURES)
    if present_challenger:
        active_numeric_features.extend(CHALLENGER_SPECIALIST_GATE_STATE_FEATURES)
    if present_smart_alignment:
        active_numeric_features.extend(present_smart_alignment)
    review = {
        "base_gate_fields": list(BASE_SPECIALIST_GATE_STATE_FEATURES),
        "challenger_gate_fields": list(CHALLENGER_SPECIALIST_GATE_STATE_FEATURES),
        "present_challenger_gate_fields": present_challenger,
        "challenger_gate_fields_complete": challenger_complete,
        "active_gate_mode": "challenger_or_smart_8_gate" if present_challenger else "foundation_seq146_6_gate",
        "smart_alignment_state_fields_available": len(present_smart_alignment) > 0,
        "smart_alignment_state_field_count": len(present_smart_alignment),
        "smart_alignment_state_fields": present_smart_alignment,
    }
    return tuple(active_state_features), tuple(active_numeric_features), review


def _build_state_reward_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["bar_index"] = pd.to_numeric(out["bar_index"], errors="raise").astype(int)
    out["bars_held"] = pd.to_numeric(out["bars_held"], errors="raise").astype(int)
    out["is_realized_exit_bar_bool"] = _bool_series(out["is_realized_exit_bar"])
    out = out.sort_values(["entry_trade_id", "bar_index"]).reset_index(drop=True)
    out["exit_episode_id"] = out["entry_trade_id"].astype(str)
    out["exit_timestep"] = out["bar_index"].astype(int)
    out["logged_action"] = np.where(out["is_realized_exit_bar_bool"], "EXIT_NOW", "HOLD")
    out["logged_action_id"] = out["logged_action"].map(ACTION_ID).astype(int)
    out["exit_now_label"] = out["is_realized_exit_bar_bool"].astype(bool)
    out["hold_label"] = ~out["is_realized_exit_bar_bool"].astype(bool)
    out["is_terminal_transition"] = out["is_realized_exit_bar_bool"].astype(bool)
    out["hold_reward_bps"] = 0.0
    out["forced_terminal_hold_reward_bps"] = np.where(
        out["is_terminal_transition"],
        pd.to_numeric(out["realized_net_pnl_bps"], errors="coerce"),
        0.0,
    )
    out["exit_now_reward_bps"] = pd.to_numeric(out["running_pnl_bps"], errors="coerce")
    out["terminal_reward_realized_net_pnl_bps"] = pd.to_numeric(out["realized_net_pnl_bps"], errors="coerce")
    out["logged_reward_bps"] = np.where(
        out["logged_action"].eq("EXIT_NOW"),
        out["terminal_reward_realized_net_pnl_bps"],
        0.0,
    )
    mfe = pd.to_numeric(out["running_mfe_bps"], errors="coerce")
    mae = pd.to_numeric(out["running_mae_bps"], errors="coerce")
    pnl = pd.to_numeric(out["running_pnl_bps"], errors="coerce")
    giveback = pd.to_numeric(out["running_giveback_bps"], errors="coerce")
    out["exit_now_mfe_capture_ratio_reward"] = (pnl / mfe.clip(lower=1.0)).clip(lower=-2.0, upper=2.0)
    out["exit_now_mae_penalty_reward_bps"] = pnl - 0.5 * mae.abs()
    out["exit_now_giveback_penalty_reward_bps"] = -giveback.clip(lower=0.0)
    out["exit_now_transparent_combined_reward_bps"] = pnl - 0.25 * mae.abs() - 0.25 * giveback.clip(lower=0.0)
    running_pnl = pd.to_numeric(out["running_pnl_bps"], errors="coerce")
    future_max = (
        out.groupby("exit_episode_id", sort=False)["running_pnl_bps"]
        .transform(lambda series: pd.to_numeric(series, errors="coerce").iloc[::-1].cummax().iloc[::-1])
        .astype(float)
    )
    future_min = (
        out.groupby("exit_episode_id", sort=False)["running_pnl_bps"]
        .transform(lambda series: pd.to_numeric(series, errors="coerce").iloc[::-1].cummin().iloc[::-1])
        .astype(float)
    )
    episode_max = out.groupby("exit_episode_id", sort=False)["running_mfe_bps"].transform(
        lambda series: pd.to_numeric(series, errors="coerce").max()
    )
    realized_exit_reason = out.get("realized_exit_reason", pd.Series([""] * len(out), index=out.index))
    realized_mfe = pd.to_numeric(out.get("realized_mfe_bps", pd.Series([np.nan] * len(out), index=out.index)), errors="coerce")
    stopout_episode = realized_exit_reason.astype(str).str.strip().str.upper().eq("STOP_LOSS")
    positive_mfe_stopout = stopout_episode & realized_mfe.gt(0.0)
    out["future_max_running_pnl_bps"] = future_max
    out["future_min_running_pnl_bps"] = future_min
    out["future_best_exit_lift_bps"] = (future_max - running_pnl).clip(lower=0.0)
    out["future_adverse_excursion_bps"] = (running_pnl - future_min).clip(lower=0.0)
    out["future_giveback_from_peak_bps"] = (episode_max - running_pnl).clip(lower=0.0)
    out["exit_hazard_adverse_15bps_label"] = out["future_adverse_excursion_bps"].ge(15.0).astype(int)
    out["exit_hazard_giveback_20bps_label"] = out["future_giveback_from_peak_bps"].ge(20.0).astype(int)
    out["positive_mfe_stopout_episode_label"] = positive_mfe_stopout.astype(int)
    out["oracle_exit_before_giveback_label"] = (
        out["future_best_exit_lift_bps"].le(5.0)
        & out["future_adverse_excursion_bps"].ge(15.0)
        & running_pnl.gt(0.0)
    ).astype(int)
    out["next_exit_episode_id"] = out.groupby("exit_episode_id")["exit_episode_id"].shift(-1)
    out["next_exit_timestep"] = out.groupby("exit_episode_id")["exit_timestep"].shift(-1)
    out.loc[out["is_terminal_transition"], "next_exit_episode_id"] = ""
    out.loc[out["is_terminal_transition"], "next_exit_timestep"] = np.nan
    out["next_row_available"] = (~out["is_terminal_transition"]) & out["next_exit_episode_id"].astype(str).ne("")
    return out


def _pointer_review(frame: pd.DataFrame) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for episode_id, group in frame.groupby("exit_episode_id", sort=False):
        non_terminal = group.loc[~group["is_terminal_transition"].astype(bool)]
        if not bool(non_terminal["next_row_available"].all()):
            failures.append({"exit_episode_id": str(episode_id), "reason": "nonterminal_missing_next_row"})
            continue
        expected_next = non_terminal["exit_timestep"].astype(int) + 1
        observed_next = pd.to_numeric(non_terminal["next_exit_timestep"], errors="coerce").astype(int)
        if not bool((expected_next.to_numpy() == observed_next.to_numpy()).all()):
            failures.append({"exit_episode_id": str(episode_id), "reason": "next_timestep_not_plus_one"})
        terminal = group.loc[group["is_terminal_transition"].astype(bool)]
        if len(terminal) != 1:
            failures.append({"exit_episode_id": str(episode_id), "reason": "terminal_count_not_one"})
        elif str(terminal["next_exit_episode_id"].iloc[0]).strip():
            failures.append({"exit_episode_id": str(episode_id), "reason": "terminal_has_next_episode"})
    return {"ready": not failures, "failure_count": int(len(failures)), "failures": failures[:50]}


def _hazard_label_review(frame: pd.DataFrame) -> dict[str, Any]:
    labels = (
        "exit_hazard_adverse_15bps_label",
        "exit_hazard_giveback_20bps_label",
        "positive_mfe_stopout_episode_label",
        "oracle_exit_before_giveback_label",
    )
    rows: dict[str, dict[str, Any]] = {}
    ready = True
    for label in labels:
        if label not in frame.columns:
            rows[label] = {"present": False, "finite": False, "positive_count": 0, "negative_count": 0}
            ready = False
            continue
        values = pd.to_numeric(frame[label], errors="coerce")
        finite = bool(values.notna().all())
        positive = int(values.eq(1).sum()) if finite else 0
        negative = int(values.eq(0).sum()) if finite else 0
        label_ready = finite and positive > 0 and negative > 0
        ready = ready and label_ready
        rows[label] = {
            "present": True,
            "finite": finite,
            "positive_count": positive,
            "negative_count": negative,
            "positive_rate": float(positive / len(values)) if len(values) else 0.0,
            "ready": label_ready,
        }
    return {"ready": bool(ready), "labels": rows}


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Exit State Reward Contract",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Dataset rows: `{report['dataset_rows']}`",
        f"- Episode count: `{report['episode_count']}`",
        f"- State feature count: `{len(report['state_feature_names'])}`",
        f"- Exit training allowed: `{report['exit_training_allowed']}`",
        f"- Exit IQL allowed: `{report['exit_iql_allowed']}`",
        f"- Dataset: `{report['state_reward_dataset_csv']}`",
        f"- Next required gate: `{report['next_required_gate']}`",
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
    reconstruction_json = Path(args.reconstruction_audit_json).expanduser().resolve()
    reconstruction = _read_json_or_empty(reconstruction_json)
    dataset_path = _path_from_report(reconstruction, "dataset_csv")
    dataset_exists = bool(str(reconstruction.get("dataset_csv") or "").strip()) and dataset_path.exists()
    source = pd.read_csv(dataset_path, low_memory=False) if dataset_exists else pd.DataFrame()
    state_reward = _build_state_reward_dataset(source) if not source.empty else pd.DataFrame()
    active_state_features, active_numeric_state_features, gate_mode_review = _active_state_features(source)
    missing_state_fields = [field for field in active_state_features if field not in source.columns]
    forbidden_overlap = [field for field in active_state_features if field in set(FORBIDDEN_STATE_FIELDS)]
    numeric_state_review = _finite_review(state_reward, active_numeric_state_features) if not state_reward.empty else {"all_finite": False, "fields": {}}
    reward_review = _finite_review(state_reward, REWARD_FIELDS) if not state_reward.empty else {"all_finite": False, "fields": {}}
    pointer_review = _pointer_review(state_reward) if not state_reward.empty else {"ready": False, "failure_count": 0, "failures": []}
    hazard_label_review = _hazard_label_review(state_reward) if not state_reward.empty else {"ready": False, "labels": {}}
    action_counts = (
        state_reward["logged_action"].value_counts().to_dict()
        if "logged_action" in state_reward.columns
        else {}
    )
    terminal = state_reward["is_terminal_transition"].astype(bool) if "is_terminal_transition" in state_reward.columns else pd.Series(dtype=bool)
    logged_action_ok = bool(
        not state_reward.empty
        and set(action_counts).issubset({"HOLD", "EXIT_NOW"})
        and bool(state_reward.loc[~terminal, "logged_action"].eq("HOLD").all())
        and bool(state_reward.loc[terminal, "logged_action"].eq("EXIT_NOW").all())
    )
    state_reward_dataset_csv = out_dir / "entry_exit_state_reward_dataset.csv"
    if not state_reward.empty:
        state_reward.to_csv(state_reward_dataset_csv, index=False)
    else:
        pd.DataFrame().to_csv(state_reward_dataset_csv, index=False)
    checks = [
        _check("active reconstruction audit exists", reconstruction_json.exists(), {"path": str(reconstruction_json)}),
        _check(
            "active reconstruction audit is ready for state/reward contract review",
            str(reconstruction.get("decision")) == READY_RECONSTRUCTION_DECISION,
            {"decision": reconstruction.get("decision"), "required": READY_RECONSTRUCTION_DECISION},
        ),
        _check("active reconstruction source dataset exists", dataset_exists, {"dataset_csv": str(dataset_path)}),
        _check("state/reward dataset has rows", not state_reward.empty, {"rows": int(len(state_reward))}),
        _check(
            "state/reward dataset row count matches reconstruction dataset",
            int(len(state_reward)) == int(reconstruction.get("dataset_rows") or len(source)) and not state_reward.empty,
            {"state_reward_rows": int(len(state_reward)), "reconstruction_rows": reconstruction.get("dataset_rows")},
        ),
        _check(
            "state feature contract fields are present",
            not missing_state_fields,
            {"missing_state_fields": missing_state_fields, "state_features": list(active_state_features)},
        ),
        _check(
            "specialist gate state fields match active Entry contract mode",
            bool(gate_mode_review.get("challenger_gate_fields_complete")),
            gate_mode_review,
        ),
        _check(
            "state feature contract excludes reward/outcome shortcut fields",
            not forbidden_overlap,
            {"forbidden_overlap": forbidden_overlap, "forbidden_state_fields": list(FORBIDDEN_STATE_FIELDS)},
        ),
        _check("numeric state features are finite", bool(numeric_state_review.get("all_finite")), numeric_state_review),
        _check("reward fields are finite", bool(reward_review.get("all_finite")), reward_review),
        _check("survival/hazard outcome labels are finite and live", bool(hazard_label_review.get("ready")), hazard_label_review),
        _check("logged HOLD/EXIT_NOW actions match terminal semantics", logged_action_ok, {"action_counts": action_counts}),
        _check("HOLD next-row pointers are intra-episode and terminal rows stop", bool(pointer_review.get("ready")), pointer_review),
        _check(
            "state/reward contract never trains, replays, distills, promotes, shadows, or starts live",
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
    state_feature_contract = {
        "state_timing": "AS_OF_CLOSED_M5_BAR_T_WITH_ENTRY_SNAPSHOT",
        "state_feature_names": list(active_state_features),
        "numeric_state_features": list(active_numeric_state_features),
        "categorical_state_features": ["session", "vol_regime", "side"],
        "provenance_fields_not_state": list(STATE_PROVENANCE_FIELDS),
        "forbidden_state_fields": list(FORBIDDEN_STATE_FIELDS),
        "specialist_gate_mode_review": gate_mode_review,
    }
    reward_contract = {
        "action_set": ACTION_ID,
        "logged_policy_semantics": "HOLD on non-terminal rows, EXIT_NOW on realized exit bar",
        "hold_reward_bps": "0 immediate reward on non-terminal HOLD rows",
        "forced_terminal_hold_reward_bps": "realized net PnL when historical data terminates an attempted HOLD at terminal row",
        "exit_now_reward_bps": "running close PnL at the current closed M5 bar",
        "logged_reward_bps": "0 for logged HOLD, realized net PnL for logged EXIT_NOW",
        "survival_hazard_labels": {
            "future_max_running_pnl_bps": "best future closed-bar running PnL from current row through terminal",
            "future_min_running_pnl_bps": "worst future closed-bar running PnL from current row through terminal",
            "future_best_exit_lift_bps": "remaining upside if HOLD can still improve over EXIT_NOW",
            "future_adverse_excursion_bps": "worst future loss of running PnL after current row",
            "future_giveback_from_peak_bps": "episode peak MFE still at risk of giveback from current row",
            "exit_hazard_adverse_15bps_label": "future adverse excursion reaches at least 15 bps",
            "exit_hazard_giveback_20bps_label": "future giveback from episode peak is at least 20 bps",
            "positive_mfe_stopout_episode_label": "episode eventually stopped out after having positive MFE",
            "oracle_exit_before_giveback_label": "diagnostic oracle label: little remaining upside, material adverse risk, current PnL positive",
        },
        "reward_fields": list(REWARD_FIELDS),
        "discount_default_gamma": 0.99,
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"ENTRY_EXIT_STATE_REWARD_CONTRACT_{timestamp}.json"
    md_path = out_dir / f"ENTRY_EXIT_STATE_REWARD_CONTRACT_{timestamp}.md"
    report = {
        "schema_version": "entry_exit_state_reward_contract_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "reconstruction_audit_json": str(reconstruction_json),
        "reconstruction_audit_json_sha256": _sha256_file(reconstruction_json) if reconstruction_json.exists() else "",
        "source_dataset_csv": str(dataset_path) if dataset_exists else "",
        "source_dataset_csv_sha256": _sha256_file(dataset_path) if dataset_exists else "",
        "state_reward_dataset_csv": str(state_reward_dataset_csv),
        "state_reward_dataset_csv_sha256": _sha256_file(state_reward_dataset_csv) if state_reward_dataset_csv.exists() else "",
        "dataset_rows": int(len(state_reward)),
        "episode_count": int(state_reward["exit_episode_id"].nunique()) if "exit_episode_id" in state_reward.columns else 0,
        "state_feature_contract": state_feature_contract,
        "state_feature_names": list(active_state_features),
        "specialist_gate_mode_review": gate_mode_review,
        "reward_contract": reward_contract,
        "action_counts": action_counts,
        "numeric_state_review": numeric_state_review,
        "reward_review": reward_review,
        "hazard_label_review": hazard_label_review,
        "pointer_review": pointer_review,
        "checks": checks,
        "failures": failures,
        "exit_training_allowed": False,
        "exit_iql_allowed": False,
        "trainer_started": False,
        "replay_started": False,
        "iql_distillation_started": False,
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            "audit active Exit split/leakage before any Exit Transformer/IQL training"
            if ready
            else "repair active Exit state/reward contract before split/leakage audit"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_EXIT_STATE_REWARD_CONTRACT_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_EXIT_STATE_REWARD_CONTRACT_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": decision,
                    "dataset_rows": int(len(state_reward)),
                    "episode_count": report["episode_count"],
                    "failures": failures,
                    "json_path": str(json_path),
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
    ap.add_argument("--reconstruction-audit-json", default=str(DEFAULT_RECONSTRUCTION_AUDIT_JSON))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
