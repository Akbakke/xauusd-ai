"""Run a manifest-resolved shadow/paper gate for the tabular no-XGB Entry candidate.

This gate does not promote anything. It loads an explicit candidate manifest,
calibrates the top-k score threshold on a calibration split, then paper-replays
an evaluation split with the packaged policy parameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from gx1.runtime.entry_tabular_no_xgb_candidate import (
    assert_no_xgb_feature_names,
    build_feature_matrix,
    feature_contract_hash,
    json_default,
    predict_proba,
    score_probabilities,
    stable_json_hash,
)
from gx1.scripts.evaluate_entry_selective_edge_v1 import SESSION_NAMES, _split_files
from gx1.scripts.replay_entry_tabular_no_xgb_policy_v1 import (
    SourceTape,
    _aggregate_outputs,
    _run_policy,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def _load_base_df(parquet_path: Path, split: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, columns=["time", "ctx_cat", "y_direction", "label_horizon_bars", "path_quality_bps"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    ctx_cat = np.vstack(df["ctx_cat"].to_numpy()).astype(np.int64)
    df["session_id"] = ctx_cat[:, 0].astype(int)
    df["session"] = df["session_id"].map(SESSION_NAMES).fillna("UNKNOWN")
    df["source_split"] = split
    return df.drop(columns=["ctx_cat"]).reset_index(drop=True)


def _threshold_from_scores(scores: np.ndarray, top_frac: float, min_score_floor: float) -> float:
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError("cannot calibrate threshold from empty/non-finite scores")
    k = max(1, int(np.ceil(finite.size * float(top_frac))))
    threshold = float(np.partition(finite, finite.size - k)[finite.size - k])
    return max(threshold, float(min_score_floor))


def _namespace_for_policy(policy: dict[str, Any], *, cost_stress_bps: float, slippage_bps: float) -> argparse.Namespace:
    del cost_stress_bps
    return argparse.Namespace(
        exit_mode=str(policy["exit_mode"]),
        take_profit_bps=float(policy.get("take_profit_bps", 60.0)),
        stop_loss_bps=float(policy.get("stop_loss_bps", 45.0)),
        same_bar_policy=str(policy.get("same_bar_policy", "stop_first")),
        cooldown_bars=int(policy["cooldown_bars"]),
        max_trades_per_day=int(policy["max_trades_per_day"]),
        daily_loss_limit_bps=float(policy["daily_loss_limit_bps"]),
        min_direction_prob=float(policy["min_direction_prob"]),
        min_score_floor=float(policy["min_score_floor"]),
        slippage_bps=float(slippage_bps),
        size_multiplier=float(policy["size_multiplier"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate_manifest_path = Path(args.candidate_manifest).expanduser().resolve()
    package_dir = candidate_manifest_path.parent
    manifest = _read_json(candidate_manifest_path)
    feature_manifest = _read_json(package_dir / "feature_manifest.json")
    policy = _read_json(package_dir / "policy_config.json")

    if manifest.get("package_status") != "NOT_PROMOTED_NOT_LIVE_READY":
        raise RuntimeError("candidate package must be NOT_PROMOTED_NOT_LIVE_READY")
    if bool(manifest.get("promotion_allowed")) or bool(manifest.get("live_ready")):
        raise RuntimeError("candidate package unexpectedly allows promotion/live")

    feature_names = [str(item["name"]) for item in feature_manifest["features"]]
    assert_no_xgb_feature_names(feature_names)
    fhash = feature_contract_hash(feature_names)
    if fhash != feature_manifest["feature_contract_hash"] or fhash != manifest["feature_contract"]["feature_contract_hash"]:
        raise RuntimeError("feature contract hash mismatch")

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    source_parquet = Path(args.source_parquet).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_files = _split_files(dataset_dir, [args.calibration_split, args.paper_split])
    model = joblib.load(Path(manifest["model"]["primary_model_path"]))

    cal_x, cal_names = build_feature_matrix(split_files[args.calibration_split], expected_feature_names=feature_names)
    paper_x, paper_names = build_feature_matrix(split_files[args.paper_split], expected_feature_names=feature_names)
    if cal_names != feature_names or paper_names != feature_names:
        raise RuntimeError("runtime feature names differ from candidate manifest")

    cal_probs = predict_proba(model, cal_x)
    _cal_side, _cal_prob, cal_score = score_probabilities(cal_probs)
    threshold_top_frac = float(args.threshold_top_frac if args.threshold_top_frac is not None else policy["threshold_top_frac"])
    threshold = _threshold_from_scores(cal_score, threshold_top_frac, float(policy["min_score_floor"]))

    paper_probs = predict_proba(model, paper_x)
    paper_df = _load_base_df(split_files[args.paper_split], args.paper_split)
    tape = SourceTape.load(source_parquet)
    source_idx = tape.indices_for_times(paper_df["time"])
    policy_config = {
        "candidate_id": manifest["candidate_id"],
        "candidate_manifest": str(candidate_manifest_path),
        "feature_contract_hash": fhash,
        "policy_config_hash": policy["policy_config_hash"],
        "calibration_split": str(args.calibration_split),
        "paper_split": str(args.paper_split),
        "threshold_top_frac": threshold_top_frac,
        "score_threshold": threshold,
        "cost_stress_bps": float(args.cost_stress_bps),
        "slippage_bps": float(args.slippage_bps),
    }
    policy_id = "shadow_paper_" + stable_json_hash(policy_config)[:16]
    trades, decisions = _run_policy(
        fold_id=str(args.paper_split),
        eval_df=paper_df,
        probs=paper_probs,
        source_idx=source_idx,
        tape=tape,
        threshold_top_frac=threshold_top_frac,
        score_threshold=threshold,
        cost_stress_bps=float(args.cost_stress_bps),
        args=_namespace_for_policy(policy, cost_stress_bps=float(args.cost_stress_bps), slippage_bps=float(args.slippage_bps)),
        policy_id=policy_id,
        policy_config_hash=stable_json_hash(policy_config),
    )
    trades_df = pd.DataFrame(trades)
    decisions_df = pd.DataFrame([decisions])
    outputs = _aggregate_outputs(trades_df, decisions_df, out_dir)
    metrics = pd.read_csv(outputs["metrics_csv"])
    all_row = metrics[metrics["scope"].astype(str) == "all"]
    if len(all_row) != 1:
        raise RuntimeError("shadow/paper metrics missing aggregate row")
    row = all_row.iloc[0]
    monthly = pd.read_csv(outputs["monthly_csv"])
    positive_months = int((monthly["net_sum_bps"] > 0).sum()) if not monthly.empty else 0
    total_months = int(len(monthly))
    pass_gate = (
        int(row["n_trades"]) > 0
        and float(row["net_sum_bps"]) > 0.0
        and float(row["net_mean_bps"]) > 0.0
        and positive_months == total_months
    )
    summary = {
        "schema_version": "entry_tabular_no_xgb_shadow_paper_gate_v1",
        "candidate_id": manifest["candidate_id"],
        "candidate_manifest": str(candidate_manifest_path),
        "package_status": manifest["package_status"],
        "promotion_allowed": False,
        "live_ready": False,
        "feature_contract_hash": fhash,
        "calibration_split": str(args.calibration_split),
        "paper_split": str(args.paper_split),
        "threshold_top_frac": threshold_top_frac,
        "score_threshold": threshold,
        "cost_stress_bps": float(args.cost_stress_bps),
        "slippage_bps": float(args.slippage_bps),
        "policy_id": policy_id,
        "outputs": outputs,
        "metrics": {
            "n_trades": int(row["n_trades"]),
            "net_sum_bps": float(row["net_sum_bps"]),
            "net_mean_bps": float(row["net_mean_bps"]),
            "win_rate": float(row["win_rate"]),
            "profit_factor": float(row["profit_factor"]),
            "max_drawdown_bps": float(row["max_drawdown_bps"]),
            "max_loss_bps": float(row["max_loss_bps"]),
            "positive_months": positive_months,
            "total_months": total_months,
        },
        "status": "PASS" if pass_gate else "FAIL",
        "decision": "NO_LIVE_PIN_REVIEW_REQUIRED",
        "next_required_gate": "manual_review_then_live_shadow_wiring_or_reject",
    }
    _write_json(out_dir / "shadow_paper_summary.json", summary)
    md = "\n".join([
        f"# Shadow/Paper Gate: {manifest['candidate_id']}",
        "",
        f"Status: {summary['status']}",
        "",
        f"- calibration split: `{args.calibration_split}`",
        f"- paper split: `{args.paper_split}`",
        f"- threshold: `{threshold:.12f}`",
        f"- trades: `{summary['metrics']['n_trades']}`",
        f"- net sum bps: `{summary['metrics']['net_sum_bps']:.2f}`",
        f"- net mean bps: `{summary['metrics']['net_mean_bps']:.2f}`",
        f"- positive months: `{positive_months}/{total_months}`",
        "- decision: NO LIVE PIN, review required",
        "",
    ])
    (out_dir / "shadow_paper_summary.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-manifest", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--source-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--calibration-split", default="val")
    ap.add_argument("--paper-split", default="test")
    ap.add_argument("--threshold-top-frac", type=float)
    ap.add_argument("--cost-stress-bps", type=float, required=True)
    ap.add_argument("--slippage-bps", type=float, required=True)
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
