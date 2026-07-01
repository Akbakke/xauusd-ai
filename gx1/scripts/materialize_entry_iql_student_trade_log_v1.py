#!/usr/bin/env python3
"""Materialize a foundation-bound Entry IQL-student replay trade log.

This is an offline research materializer for the post-replay Entry foundation
path. It consumes the ready IQL distillation contract, fits/selects a small
validation-calibrated student policy on the active seq146 candidate predictions,
and writes an explicit 2026 trade log for `iql-replay-evidence`.

It does not build adapters, promote, shadow, live, or place orders.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.apply_entry_foundation_activation_v1 import DEFAULT_SOURCE_PARQUET
from gx1.scripts.materialize_entry_candidate_replay_trade_log_v1 import (
    DEFAULT_PREDICTIONS,
    _cost_label,
    _frac_label,
    _json_default,
    _parse_float_list,
    _prepare_predictions,
    _run_fixed_horizon_policy,
)
from gx1.scripts.materialize_entry_iql_distillation_contract_v1 import (
    DEFAULT_OUT_DIR as DISTILL_CONTRACT_OUT_DIR,
    IQL_DISTILLATION_REQUIRED_ARTIFACT_KEYS,
    _sha256_file,
)
from gx1.scripts.replay_entry_tabular_no_xgb_policy_v1 import (
    EXIT_MODE_CHOICES,
    SourceTape,
    _exit_policy_config_from_args,
    _exit_policy_contract_from_args,
    _policy_hash,
    _threshold_from_scores,
    _validate_exit_policy_args,
)
from gx1.scripts.verify_entry_foundation_state_v1 import FOUNDATION_DATASET_DIR, REPORTS_ROOT


DEFAULT_DISTILL_CONTRACT_JSON = DISTILL_CONTRACT_OUT_DIR / "ENTRY_IQL_DISTILLATION_CONTRACT_latest.json"
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_iql_student_trade_log_20260628_v1"
REQUIRED_CONTRACT_DECISION = "ENTRY_IQL_DISTILLATION_CONTRACT_READY"
VEDTAK_PREFIX = "ENTRY_FOUNDATION_IQL_DISTILL_"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing JSON artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    return payload


def _contract_identity(contract: dict[str, Any]) -> dict[str, Any]:
    value = contract.get("evidence_identity")
    return value if isinstance(value, dict) else {}


def _distillation_contract_checks(contract_path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if str(contract.get("decision")) != REQUIRED_CONTRACT_DECISION:
        failures.append(f"IQL distillation contract decision is not ready: {contract.get('decision')}")
    if bool(contract.get("iql_research_distillation_allowed")) is not True:
        failures.append("IQL distillation contract does not allow research distillation")
    if bool(contract.get("promotion_shadow_live_allowed")) is not False:
        failures.append("IQL distillation contract does not block promotion/shadow/live")

    artifact_paths = contract.get("artifact_paths") if isinstance(contract.get("artifact_paths"), dict) else {}
    artifact_sha256 = contract.get("artifact_sha256") if isinstance(contract.get("artifact_sha256"), dict) else {}
    hash_checks: dict[str, dict[str, Any]] = {}
    for key in IQL_DISTILLATION_REQUIRED_ARTIFACT_KEYS:
        raw_path = str(artifact_paths.get(key) or "")
        path = Path(raw_path).expanduser().resolve() if raw_path else None
        expected = str(artifact_sha256.get(key) or "")
        exists = bool(path and path.is_file())
        observed = _sha256_file(path) if exists and path is not None else ""
        ok = bool(expected and observed and expected == observed)
        hash_checks[key] = {
            "path": str(path) if path is not None else "",
            "expected": expected,
            "observed": observed,
            "exists": exists,
            "ok": ok,
        }
        if not ok:
            failures.append(f"IQL distillation contract artifact hash is stale or missing: {key}")

    identity = _contract_identity(contract)
    candidate_bundle_dir = str(identity.get("candidate_bundle_dir") or "")
    if not candidate_bundle_dir:
        failures.append("IQL distillation contract evidence_identity has no candidate_bundle_dir")
    if candidate_bundle_dir and str(identity.get("selective_edge_bundle_dir") or "") != candidate_bundle_dir:
        failures.append("IQL distillation contract selective-edge bundle identity does not match candidate bundle")
    if candidate_bundle_dir and str(identity.get("replay_identity_candidate_bundle_dir") or "") != candidate_bundle_dir:
        failures.append("IQL distillation contract replay identity does not match candidate bundle")
    if not bool(identity.get("replay_identity_ready")):
        failures.append("IQL distillation contract replay identity is not ready")

    return {
        "ready": not failures,
        "distillation_contract_json": str(contract_path),
        "artifact_hash_checks": hash_checks,
        "evidence_identity": identity,
        "failures": failures,
    }


def _safe_profit_factor(pnl: pd.Series) -> float | None:
    values = pd.to_numeric(pnl, errors="coerce").to_numpy(np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    gains = float(values[values > 0.0].sum())
    losses = float(values[values < 0.0].sum())
    if losses == 0.0:
        return None
    return float(gains / abs(losses))


def _max_drawdown_bps(pnl: pd.Series) -> float:
    values = pd.to_numeric(pnl, errors="coerce").fillna(0.0).to_numpy(np.float64)
    if values.size == 0:
        return 0.0
    curve = np.concatenate([[0.0], np.cumsum(values)])
    drawdown = curve - np.maximum.accumulate(curve)
    return abs(float(np.min(drawdown)))


def _trade_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "n_trades": 0,
            "net_sum_bps": 0.0,
            "net_mean_bps": None,
            "profit_factor": None,
            "max_drawdown_bps": 0.0,
            "max_loss_bps": None,
            "negative_month_count": 0,
            "all_months_positive": False,
        }
    frame = pd.DataFrame(trades)
    pnl = pd.to_numeric(frame["net_pnl_bps"], errors="coerce")
    monthly = frame.groupby("entry_month")["net_pnl_bps"].sum()
    return {
        "n_trades": int(len(frame)),
        "net_sum_bps": float(pnl.sum()),
        "net_mean_bps": float(pnl.mean()),
        "profit_factor": _safe_profit_factor(pnl),
        "max_drawdown_bps": _max_drawdown_bps(pnl),
        "max_loss_bps": float(pnl.min()) if len(pnl) else None,
        "negative_month_count": int((monthly <= 0.0).sum()),
        "all_months_positive": bool(len(monthly) > 0 and (monthly > 0.0).all()),
    }


def _passes_validation_constraints(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    if int(metrics["n_trades"]) < int(args.min_validation_trades):
        return False
    pf = metrics.get("profit_factor")
    if pf is None or float(pf) < float(args.min_validation_profit_factor):
        return False
    if float(metrics["max_drawdown_bps"]) > float(args.max_validation_drawdown_bps):
        return False
    max_loss = metrics.get("max_loss_bps")
    if max_loss is None or abs(float(max_loss)) > float(args.max_abs_loss_bps):
        return False
    if bool(args.require_validation_positive_months) and not bool(metrics["all_months_positive"]):
        return False
    return True


def _selection_tuple(metrics: dict[str, Any], args: argparse.Namespace) -> tuple[float, float, float, float]:
    if not _passes_validation_constraints(metrics, args):
        return (-float("inf"), -float("inf"), -float("inf"), -float("inf"))
    mean_pnl = float(metrics.get("net_mean_bps") or 0.0)
    drawdown = float(metrics.get("max_drawdown_bps") or 0.0)
    net = float(metrics.get("net_sum_bps") or 0.0)
    pf = float(metrics.get("profit_factor") or 0.0)
    if str(getattr(args, "selection_objective", "mean")) == "net":
        return (net, -drawdown, mean_pnl, pf)
    # Path-quality first: prefer high average reward, then lower drawdown, then total reward/PF.
    return (mean_pnl, -drawdown, net, pf)


def _row_selection_tuple(row: dict[str, Any], args: argparse.Namespace) -> tuple[float, ...]:
    metrics = row.get("validation_metrics") if isinstance(row.get("validation_metrics"), dict) else {}
    base = _selection_tuple(metrics, args)
    if not np.isfinite(base[0]):
        return base
    return base + (
        float(row.get("daily_loss_limit_bps") or 0.0),
        float(row.get("validation_eligible_rows") or 0.0),
        float(metrics.get("n_trades") or 0.0),
    )


def _parse_optional_float_grid(raw: str) -> list[float | None]:
    values: list[float | None] = []
    for part in str(raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        if token.lower() in {"none", "off", "null"}:
            values.append(None)
        else:
            values.append(float(token))
    return values or [None]


def _apply_student_risk_filters(
    frame: pd.DataFrame,
    *,
    max_bad_path_prob: float | None,
    min_path_quality_pred: float | None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    if max_bad_path_prob is not None:
        if "bad_path_prob" not in frame.columns:
            return frame.iloc[0:0].copy()
        mask &= pd.to_numeric(frame["bad_path_prob"], errors="coerce").le(float(max_bad_path_prob)).fillna(False)
    if min_path_quality_pred is not None:
        if "path_quality_pred" not in frame.columns:
            return frame.iloc[0:0].copy()
        mask &= pd.to_numeric(frame["path_quality_pred"], errors="coerce").ge(float(min_path_quality_pred)).fillna(False)
    return frame.loc[mask].copy()


def _student_policy_run_args(
    args: argparse.Namespace,
    *,
    take_profit_bps: float,
    stop_loss_bps: float,
    daily_loss_limit_bps: float,
) -> argparse.Namespace:
    return argparse.Namespace(
        exit_mode=str(args.exit_mode),
        take_profit_bps=float(take_profit_bps),
        stop_loss_bps=float(stop_loss_bps),
        same_bar_policy=str(args.same_bar_policy),
        min_direction_prob=float(args.min_direction_prob),
        max_trades_per_day=int(args.max_trades_per_day),
        daily_loss_limit_bps=float(daily_loss_limit_bps),
        cooldown_bars=int(args.cooldown_bars),
        slippage_bps=float(args.slippage_bps),
        size_multiplier=float(args.size_multiplier),
        mfe_protect_activation_bps=float(getattr(args, "mfe_protect_activation_bps", 20.0)),
        mfe_protect_breakeven_offset_bps=float(getattr(args, "mfe_protect_breakeven_offset_bps", 0.0)),
        mfe_protect_trailing_capture_ratio=float(getattr(args, "mfe_protect_trailing_capture_ratio", 0.0)),
        mfe_protect_trailing_floor_bps=float(getattr(args, "mfe_protect_trailing_floor_bps", 0.0)),
    )


def _run_test_grid_diagnostics(
    *,
    validation_rows: list[dict[str, Any]],
    test: pd.DataFrame,
    tape: SourceTape,
    args: argparse.Namespace,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    candidate_rows = [row for row in validation_rows if bool(row.get("validation_constraints_pass"))]
    candidate_rows.sort(key=lambda row: _row_selection_tuple(row, args), reverse=True)
    diagnostics: list[dict[str, Any]] = []
    for row in candidate_rows[:limit]:
        filtered_test = _apply_student_risk_filters(
            test,
            max_bad_path_prob=row.get("max_bad_path_prob"),
            min_path_quality_pred=row.get("min_path_quality_pred"),
        )
        run_args = argparse.Namespace(
            exit_mode=str(row["exit_mode"]),
            take_profit_bps=float(row["take_profit_bps"]),
            stop_loss_bps=float(row["stop_loss_bps"]),
            same_bar_policy=str(row["same_bar_policy"]),
            min_direction_prob=float(row["min_direction_prob"]),
            max_trades_per_day=int(row["max_trades_per_day"]),
            daily_loss_limit_bps=float(row["daily_loss_limit_bps"]),
            cooldown_bars=int(row["cooldown_bars"]),
            slippage_bps=float(row["slippage_bps"]),
            size_multiplier=float(row["size_multiplier"]),
            mfe_protect_activation_bps=float(row["mfe_protect_activation_bps"]),
            mfe_protect_breakeven_offset_bps=float(row["mfe_protect_breakeven_offset_bps"]),
            mfe_protect_trailing_capture_ratio=float(row["mfe_protect_trailing_capture_ratio"]),
            mfe_protect_trailing_floor_bps=float(row["mfe_protect_trailing_floor_bps"]),
        )
        test_trades, test_counts = _run_fixed_horizon_policy(
            eval_df=filtered_test,
            tape=tape,
            score_threshold=float(row["score_threshold"]),
            threshold_top_frac=float(row["threshold_top_frac"]),
            cost_stress_bps=float(row["cost_stress_bps"]),
            args=run_args,
            policy_id=str(row["grid_policy_id"]),
            policy_config_hash=str(row["policy_config_hash"]),
        )
        diagnostics.append(
            {
                "diagnostic_only_not_selection_criterion": True,
                "grid_policy_id": str(row["grid_policy_id"]),
                "policy_config_hash": str(row["policy_config_hash"]),
                "threshold_top_frac": float(row["threshold_top_frac"]),
                "score_threshold": float(row["score_threshold"]),
                "exit_mode": str(row["exit_mode"]),
                "exit_policy_config_hash": str(row.get("exit_policy_config_hash", "")),
                "stop_loss_bps": float(row["stop_loss_bps"]),
                "take_profit_bps": float(row["take_profit_bps"]),
                "mfe_protect_activation_bps": float(row["mfe_protect_activation_bps"]),
                "mfe_protect_breakeven_offset_bps": float(row["mfe_protect_breakeven_offset_bps"]),
                "mfe_protect_trailing_capture_ratio": float(row["mfe_protect_trailing_capture_ratio"]),
                "mfe_protect_trailing_floor_bps": float(row["mfe_protect_trailing_floor_bps"]),
                "daily_loss_limit_bps": float(row["daily_loss_limit_bps"]),
                "cost_stress_bps": float(row["cost_stress_bps"]),
                "max_bad_path_prob": row.get("max_bad_path_prob"),
                "min_path_quality_pred": row.get("min_path_quality_pred"),
                "validation_eligible_rows": int(row.get("validation_eligible_rows") or 0),
                "validation_metrics": row.get("validation_metrics", {}),
                "test_eligible_rows": int(len(filtered_test)),
                "test_counts": test_counts,
                "test_metrics": _trade_metrics(test_trades),
            }
        )
    return diagnostics


def _annotate_student_trades(
    trades: list[dict[str, Any]],
    *,
    selected: dict[str, Any],
    args: argparse.Namespace,
    trade_log_split: str,
    diagnostic_only_not_replay_evidence: bool,
) -> pd.DataFrame:
    frame = pd.DataFrame(trades)
    if frame.empty:
        return frame
    frame["policy_id"] = str(args.policy_id)
    frame["student_policy_id"] = str(args.policy_id)
    frame["student_policy_source"] = "entry_iql_val_reward_selection_v1"
    frame["student_trade_log_split"] = str(trade_log_split)
    frame["diagnostic_only_not_replay_evidence"] = bool(diagnostic_only_not_replay_evidence)
    frame["student_selection_metric"] = (
        "validation_net_sum_bps_then_drawdown"
        if str(getattr(args, "selection_objective", "mean")) == "net"
        else "validation_net_mean_bps_then_drawdown"
    )
    frame["student_selected_top_frac"] = float(selected["threshold_top_frac"])
    frame["student_selected_exit_mode"] = str(selected["exit_mode"])
    frame["student_selected_exit_policy_config_hash"] = str(selected.get("exit_policy_config_hash", ""))
    frame["student_selected_stop_loss_bps"] = float(selected["stop_loss_bps"])
    frame["student_selected_take_profit_bps"] = float(selected["take_profit_bps"])
    frame["student_selected_mfe_protect_activation_bps"] = float(selected["mfe_protect_activation_bps"])
    frame["student_selected_mfe_protect_breakeven_offset_bps"] = float(selected["mfe_protect_breakeven_offset_bps"])
    frame["student_selected_mfe_protect_trailing_capture_ratio"] = float(selected["mfe_protect_trailing_capture_ratio"])
    frame["student_selected_mfe_protect_trailing_floor_bps"] = float(selected["mfe_protect_trailing_floor_bps"])
    frame["student_selected_daily_loss_limit_bps"] = float(selected["daily_loss_limit_bps"])
    frame["student_selected_cost_stress_bps"] = float(selected["cost_stress_bps"])
    frame["student_selected_max_bad_path_prob"] = selected.get("max_bad_path_prob")
    frame["student_selected_min_path_quality_pred"] = selected.get("min_path_quality_pred")
    frame["teacher_model"] = str(args.model_name)
    frame["teacher_score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["teacher_p_long"] = pd.to_numeric(frame["p_long"], errors="coerce")
    frame["teacher_p_short"] = pd.to_numeric(frame["p_short"], errors="coerce")
    frame["teacher_p_flat"] = pd.to_numeric(frame["p_flat"], errors="coerce")
    if "source_split" in frame.columns:
        frame["state_source_split"] = frame["source_split"].astype(str)
    if "session" in frame.columns:
        frame["state_session"] = frame["session"].astype(str)
    if "vol_regime" in frame.columns:
        frame["state_vol_regime"] = frame["vol_regime"].astype(str)
    return frame


def run(args: argparse.Namespace) -> dict[str, Any]:
    vedtak = str(args.vedtak or "").strip()
    if not vedtak.startswith(VEDTAK_PREFIX):
        raise SystemExit(f"FATAL: --vedtak must start with {VEDTAK_PREFIX}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    distillation_contract_path = Path(args.distillation_contract_json).expanduser().resolve()
    predictions_path = Path(args.selective_edge_predictions).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    source_parquet = Path(args.source_parquet).expanduser().resolve()
    _validate_exit_policy_args(_student_policy_run_args(
        args,
        take_profit_bps=float(_parse_float_list(str(args.take_profit_bps_grid))[0]),
        stop_loss_bps=float(args.stop_loss_bps),
        daily_loss_limit_bps=float(_parse_float_list(str(args.daily_loss_limit_bps_grid))[0]),
    ))

    contract = _read_json(distillation_contract_path)
    contract_check = _distillation_contract_checks(distillation_contract_path, contract)
    predictions = _prepare_predictions(predictions_path, dataset_dir, str(args.model_name))
    if "val" not in set(predictions["split"]) or "test" not in set(predictions["split"]):
        raise RuntimeError("IQL student trade log requires val and test predictions")
    val = predictions[predictions["split"].astype(str) == "val"].sort_values("time", kind="mergesort").reset_index(drop=True)
    test = predictions[predictions["split"].astype(str) == "test"].sort_values("time", kind="mergesort").reset_index(drop=True)
    if not bool((test["time"].dt.year == 2026).all()):
        raise RuntimeError("IQL student replay test split must be entirely 2026")

    tape = SourceTape.load(source_parquet)
    top_fracs = _parse_float_list(str(args.threshold_top_fracs))
    take_profit_values = _parse_float_list(str(args.take_profit_bps_grid))
    daily_loss_values = _parse_float_list(str(args.daily_loss_limit_bps_grid))
    cost_values = _parse_float_list(str(args.cost_stress_bps))
    max_bad_path_values = _parse_optional_float_grid(str(args.max_bad_path_prob_grid))
    min_path_quality_values = _parse_optional_float_grid(str(args.min_path_quality_pred_grid))
    base_val_scores = pd.to_numeric(val["edge_score"], errors="coerce").to_numpy(np.float64)

    validation_rows: list[dict[str, Any]] = []
    selected_row: dict[str, Any] | None = None
    selected_tuple: tuple[float, ...] = (-float("inf"),) * 7

    for max_bad_path_prob in max_bad_path_values:
        for min_path_quality_pred in min_path_quality_values:
            filtered_val = _apply_student_risk_filters(
                val,
                max_bad_path_prob=max_bad_path_prob,
                min_path_quality_pred=min_path_quality_pred,
            )
            for top_frac in top_fracs:
                threshold = _threshold_from_scores(base_val_scores, float(top_frac), float(args.min_score_floor))
                for take_profit_bps in take_profit_values:
                    for daily_loss_limit_bps in daily_loss_values:
                        for cost_bps in cost_values:
                            max_bad_label = "none" if max_bad_path_prob is None else str(max_bad_path_prob)
                            min_path_label = "none" if min_path_quality_pred is None else str(min_path_quality_pred)
                            run_args = _student_policy_run_args(
                                args,
                                take_profit_bps=float(take_profit_bps),
                                stop_loss_bps=float(args.stop_loss_bps),
                                daily_loss_limit_bps=float(daily_loss_limit_bps),
                            )
                            exit_policy_config = _exit_policy_config_from_args(run_args)
                            policy_config = {
                                "student_policy_source": "entry_iql_val_reward_selection_v1",
                                "model_name": str(args.model_name),
                                "threshold_top_frac": float(top_frac),
                                "score_threshold": float(threshold),
                                **exit_policy_config,
                                "cooldown_bars": int(args.cooldown_bars),
                                "max_trades_per_day": int(args.max_trades_per_day),
                                "daily_loss_limit_bps": float(daily_loss_limit_bps),
                                "min_direction_prob": float(args.min_direction_prob),
                                "cost_stress_bps": float(cost_bps),
                                "slippage_bps": float(args.slippage_bps),
                                "size_multiplier": float(args.size_multiplier),
                                "selection_objective": str(args.selection_objective),
                                "max_bad_path_prob": max_bad_path_prob,
                                "min_path_quality_pred": min_path_quality_pred,
                            }
                            config_hash = _policy_hash(policy_config)
                            exit_policy_config_hash = _policy_hash(exit_policy_config)
                            grid_policy_id = (
                                f"entry_iql_student_grid_{_frac_label(top_frac)}_sl{int(args.stop_loss_bps)}"
                                f"_tp{int(take_profit_bps)}_dl{int(daily_loss_limit_bps)}_"
                                f"{_cost_label(cost_bps)}_bad{max_bad_label}_pq{min_path_label}_{config_hash}"
                            )
                            val_trades, val_counts = _run_fixed_horizon_policy(
                                eval_df=filtered_val,
                                tape=tape,
                                score_threshold=float(threshold),
                                threshold_top_frac=float(top_frac),
                                cost_stress_bps=float(cost_bps),
                                args=run_args,
                                policy_id=grid_policy_id,
                                policy_config_hash=config_hash,
                            )
                            metrics = _trade_metrics(val_trades)
                            row = {
                                **policy_config,
                                "policy_config_hash": config_hash,
                                "exit_policy_config_hash": exit_policy_config_hash,
                                "grid_policy_id": grid_policy_id,
                                "validation_eligible_rows": int(len(filtered_val)),
                                "validation_counts": val_counts,
                                "validation_metrics": metrics,
                                "validation_constraints_pass": _passes_validation_constraints(metrics, args),
                            }
                            validation_rows.append(row)
                            rank = _row_selection_tuple(row, args)
                            if rank > selected_tuple:
                                selected_tuple = rank
                                selected_row = row

    failures: list[str] = list(contract_check["failures"])
    if selected_row is None or not bool(selected_row.get("validation_constraints_pass")):
        failures.append("no IQL student policy passed validation constraints")
        selected_row = selected_row or {}

    selected_validation_trades: list[dict[str, Any]] = []
    selected_validation_counts: dict[str, Any] = {}
    selected_trades: list[dict[str, Any]] = []
    selected_counts: dict[str, Any] = {}
    if selected_row and not failures:
        filtered_selected_val = _apply_student_risk_filters(
            val,
            max_bad_path_prob=selected_row.get("max_bad_path_prob"),
            min_path_quality_pred=selected_row.get("min_path_quality_pred"),
        )
        filtered_test = _apply_student_risk_filters(
            test,
            max_bad_path_prob=selected_row.get("max_bad_path_prob"),
            min_path_quality_pred=selected_row.get("min_path_quality_pred"),
        )
        selected_run_args = argparse.Namespace(
            exit_mode=str(selected_row["exit_mode"]),
            take_profit_bps=float(selected_row["take_profit_bps"]),
            stop_loss_bps=float(selected_row["stop_loss_bps"]),
            same_bar_policy=str(selected_row["same_bar_policy"]),
            min_direction_prob=float(selected_row["min_direction_prob"]),
            max_trades_per_day=int(selected_row["max_trades_per_day"]),
            daily_loss_limit_bps=float(selected_row["daily_loss_limit_bps"]),
            cooldown_bars=int(selected_row["cooldown_bars"]),
            slippage_bps=float(selected_row["slippage_bps"]),
            size_multiplier=float(selected_row["size_multiplier"]),
            mfe_protect_activation_bps=float(selected_row["mfe_protect_activation_bps"]),
            mfe_protect_breakeven_offset_bps=float(selected_row["mfe_protect_breakeven_offset_bps"]),
            mfe_protect_trailing_capture_ratio=float(selected_row["mfe_protect_trailing_capture_ratio"]),
            mfe_protect_trailing_floor_bps=float(selected_row["mfe_protect_trailing_floor_bps"]),
        )
        selected_validation_trades, selected_validation_counts = _run_fixed_horizon_policy(
            eval_df=filtered_selected_val,
            tape=tape,
            score_threshold=float(selected_row["score_threshold"]),
            threshold_top_frac=float(selected_row["threshold_top_frac"]),
            cost_stress_bps=float(selected_row["cost_stress_bps"]),
            args=selected_run_args,
            policy_id=str(args.policy_id),
            policy_config_hash=str(selected_row["policy_config_hash"]),
        )
        selected_trades, selected_counts = _run_fixed_horizon_policy(
            eval_df=filtered_test,
            tape=tape,
            score_threshold=float(selected_row["score_threshold"]),
            threshold_top_frac=float(selected_row["threshold_top_frac"]),
            cost_stress_bps=float(selected_row["cost_stress_bps"]),
            args=selected_run_args,
            policy_id=str(args.policy_id),
            policy_config_hash=str(selected_row["policy_config_hash"]),
        )

    validation_trades_df = (
        _annotate_student_trades(
            selected_validation_trades,
            selected=selected_row,
            args=args,
            trade_log_split="validation",
            diagnostic_only_not_replay_evidence=True,
        )
        if selected_row
        else pd.DataFrame()
    )
    trades_df = (
        _annotate_student_trades(
            selected_trades,
            selected=selected_row,
            args=args,
            trade_log_split="test",
            diagnostic_only_not_replay_evidence=False,
        )
        if selected_row
        else pd.DataFrame()
    )
    selected_validation_trade_metrics = _trade_metrics(selected_validation_trades)
    test_metrics = _trade_metrics(selected_trades)
    test_grid_diagnostics_limit = max(0, int(getattr(args, "test_grid_diagnostics_limit", 0) or 0))
    test_grid_diagnostics = (
        _run_test_grid_diagnostics(
            validation_rows=validation_rows,
            test=test,
            tape=tape,
            args=args,
            limit=test_grid_diagnostics_limit,
        )
        if not failures
        else []
    )
    test_grid_best_by_net = (
        max(
            test_grid_diagnostics,
            key=lambda row: float(
                (row.get("test_metrics") if isinstance(row.get("test_metrics"), dict) else {}).get("net_sum_bps")
                or -float("inf")
            ),
        )
        if test_grid_diagnostics
        else None
    )
    if selected_row and validation_trades_df.empty and not failures:
        failures.append("selected IQL student validation trade log produced zero trades")
    if trades_df.empty:
        failures.append("IQL student trade log produced zero trades")
    else:
        years = set(pd.to_datetime(trades_df["entry_time"], utc=True).dt.year.astype(int).unique())
        if years != {2026}:
            failures.append(f"IQL student trade log contains years outside 2026: {sorted(years)}")

    trades_path = out_dir / "entry_iql_student_trade_log.csv"
    validation_trades_path = out_dir / "entry_iql_student_validation_trade_log.csv"
    validation_grid_path = out_dir / "entry_iql_student_validation_grid.json"
    test_grid_diagnostics_path = out_dir / "entry_iql_student_test_grid_diagnostics.json"
    counts_path = out_dir / "entry_iql_student_policy_counts.json"
    validation_counts_path = out_dir / "entry_iql_student_validation_policy_counts.json"
    manifest_path = out_dir / "ENTRY_IQL_STUDENT_TRADE_LOG_MANIFEST.json"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_json = out_dir / f"ENTRY_IQL_STUDENT_TRADE_LOG_{timestamp}.json"
    report_md = out_dir / f"ENTRY_IQL_STUDENT_TRADE_LOG_{timestamp}.md"

    trades_df.to_csv(trades_path, index=False)
    validation_trades_df.to_csv(validation_trades_path, index=False)
    validation_grid_path.write_text(json.dumps(validation_rows, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    test_grid_diagnostics_path.write_text(
        json.dumps(test_grid_diagnostics, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    counts_path.write_text(json.dumps(selected_counts, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    validation_counts_path.write_text(
        json.dumps(selected_validation_counts, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "entry_iql_student_trade_log_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS" if not failures else "FAIL",
        "vedtak": vedtak,
        "distillation_contract_json": str(distillation_contract_path),
        "distillation_contract_check": contract_check,
        "selective_edge_predictions": str(predictions_path),
        "dataset_dir": str(dataset_dir),
        "source_parquet": str(source_parquet),
        "out_dir": str(out_dir),
        "trades_path": str(trades_path),
        "validation_trades_path": str(validation_trades_path),
        "validation_grid_json": str(validation_grid_path),
        "test_grid_diagnostics_json": str(test_grid_diagnostics_path),
        "counts_json": str(counts_path),
        "validation_counts_json": str(validation_counts_path),
        "policy_id": str(args.policy_id),
        "selection_method": (
            "validation_net_sum_bps_then_drawdown_v1"
            if str(args.selection_objective) == "net"
            else "validation_net_mean_bps_then_drawdown_v1"
        ),
        "selection_objective": str(args.selection_objective),
        "student_risk_filter_grid": {
            "max_bad_path_prob_grid": [value for value in max_bad_path_values],
            "min_path_quality_pred_grid": [value for value in min_path_quality_values],
        },
        "exit_policy_contract": _exit_policy_contract_from_args(
            _student_policy_run_args(
                args,
                take_profit_bps=float(selected_row.get("take_profit_bps", _parse_float_list(str(args.take_profit_bps_grid))[0])),
                stop_loss_bps=float(selected_row.get("stop_loss_bps", args.stop_loss_bps)),
                daily_loss_limit_bps=float(
                    selected_row.get("daily_loss_limit_bps", _parse_float_list(str(args.daily_loss_limit_bps_grid))[0])
                ),
            )
        ),
        "test_grid_diagnostics": {
            "enabled": bool(test_grid_diagnostics_limit > 0),
            "limit": int(test_grid_diagnostics_limit),
            "evaluated_policies": int(len(test_grid_diagnostics)),
            "diagnostic_only_not_selection_criterion": True,
            "best_by_test_net_bps": test_grid_best_by_net,
        },
        "selected_policy": selected_row,
        "selected_validation_metrics": selected_row.get("validation_metrics", {}) if selected_row else {},
        "selected_validation_trade_metrics": selected_validation_trade_metrics,
        "selected_validation_counts": selected_validation_counts,
        "selected_validation_trade_count": int(len(validation_trades_df)),
        "validation_trade_log_diagnostic_only_not_replay_evidence": True,
        "validation_trade_log_replay_evidence_allowed": False,
        "test_metrics": test_metrics,
        "n_trades": int(len(trades_df)),
        "student_policy_fit_started": True,
        "runtime_trainer_started": False,
        "adapter_built": False,
        "shadow_started": False,
        "live_started": False,
        "promotion_shadow_live_allowed": False,
        "failures": failures,
        "json_path": str(report_json),
        "md_path": str(report_md),
    }
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    report_md.write_text(
        "\n".join(
            [
                "# Entry IQL Student Trade Log",
                "",
                f"- Decision: `{report['decision']}`",
                f"- Trades: `{report['n_trades']}`",
                f"- Policy: `{report['policy_id']}`",
                f"- Promotion/shadow/live allowed: `{report['promotion_shadow_live_allowed']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_IQL_STUDENT_TRADE_LOG_latest.json").write_text(
        report_json.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_IQL_STUDENT_TRADE_LOG_latest.md").write_text(
        report_md.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    if args.fail_on_audit_fail and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vedtak", required=True)
    ap.add_argument("--distillation-contract-json", default=str(DEFAULT_DISTILL_CONTRACT_JSON))
    ap.add_argument("--selective-edge-predictions", default=str(DEFAULT_PREDICTIONS))
    ap.add_argument("--dataset-dir", default=str(FOUNDATION_DATASET_DIR))
    ap.add_argument("--source-parquet", default=str(DEFAULT_SOURCE_PARQUET))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--model-name", default="candidate")
    ap.add_argument("--policy-id", default="entry_iql_student")
    ap.add_argument("--threshold-top-fracs", default="0.05")
    ap.add_argument("--cost-stress-bps", default="0.0")
    ap.add_argument("--exit-mode", choices=EXIT_MODE_CHOICES, default="stop_tp")
    ap.add_argument("--stop-loss-bps", type=float, default=80.0)
    ap.add_argument("--take-profit-bps-grid", default="100,120,150,180")
    ap.add_argument("--daily-loss-limit-bps-grid", default="80,120,150")
    ap.add_argument("--same-bar-policy", choices=("stop_first", "target_first"), default="stop_first")
    ap.add_argument("--mfe-protect-activation-bps", type=float, default=20.0)
    ap.add_argument("--mfe-protect-breakeven-offset-bps", type=float, default=0.0)
    ap.add_argument("--mfe-protect-trailing-capture-ratio", type=float, default=0.0)
    ap.add_argument("--mfe-protect-trailing-floor-bps", type=float, default=0.0)
    ap.add_argument("--cooldown-bars", type=int, default=6)
    ap.add_argument("--max-trades-per-day", type=int, default=8)
    ap.add_argument("--min-direction-prob", type=float, default=0.0)
    ap.add_argument("--min-score-floor", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--size-multiplier", type=float, default=1.0)
    ap.add_argument("--selection-objective", choices=("mean", "net"), default="mean")
    ap.add_argument("--max-bad-path-prob-grid", default="none")
    ap.add_argument("--min-path-quality-pred-grid", default="none")
    ap.add_argument("--test-grid-diagnostics-limit", type=int, default=0)
    ap.add_argument("--min-validation-trades", type=int, default=25)
    ap.add_argument("--min-validation-profit-factor", type=float, default=1.05)
    ap.add_argument("--max-validation-drawdown-bps", type=float, default=650.0)
    ap.add_argument("--max-abs-loss-bps", type=float, default=80.0)
    ap.add_argument("--require-validation-positive-months", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
