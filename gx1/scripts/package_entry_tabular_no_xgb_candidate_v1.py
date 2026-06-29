"""Materialize the tabular no-XGB Entry candidate package.

This is a packaging gate, not a promotion tool. It records the model, feature
contract, replay evidence, and policy parameters for the current lead path so a
later serve/shadow gate can validate exactly what is being tested.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from gx1.runtime.entry_tabular_no_xgb_candidate import (
    EXCLUDED_XGB_FIELDS,
    assert_no_xgb_feature_names,
    feature_contract_hash,
    json_default,
    selected_feature_names,
    sha256_file,
    stable_json_hash,
)
from gx1.scripts.evaluate_entry_selective_edge_v1 import _split_files


PLAN_ACK_REQUIRED = "20260627_ENTRY_NO_XGB_TOP5_PACKAGE"
PACKAGE_SCHEMA_VERSION = "entry_tabular_no_xgb_candidate_package_v1"
PACKAGE_STATUS = "NOT_PROMOTED_NOT_LIVE_READY"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        return (out.stderr or out.stdout).strip()
    return out.stdout.rstrip("\n")


def _file_fingerprint(path: Path, *, large_meta_threshold_mb: int = 512) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing artifact: {path}")
    st = path.stat()
    if path.is_file() and st.st_size <= large_meta_threshold_mb * 1024 * 1024:
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "hash_mode": "full",
            "size_bytes": int(st.st_size),
        }
    raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime_ns)}".encode("utf-8")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "hash_mode": "meta(path,size,mtime_ns)",
        "size_bytes": int(st.st_size),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _model_n_features(model_path: Path) -> int:
    model = joblib.load(model_path)
    for attr in ("n_features_in_", "n_features_"):
        value = getattr(model, attr, None)
        if value is not None:
            return int(value)
    booster = getattr(model, "booster_", None)
    if booster is not None:
        return int(booster.num_feature())
    raise RuntimeError(f"could not determine model feature count: {model_path}")


def _select_policy_row(metrics: pd.DataFrame, *, threshold_top_frac: float, cost_bps: float, slippage_bps: float) -> pd.Series:
    rows = metrics[
        (metrics["scope"].astype(str) == "all")
        & np.isclose(metrics["threshold_top_frac"].astype(float), float(threshold_top_frac))
        & np.isclose(metrics["cost_stress_bps"].astype(float), float(cost_bps))
        & np.isclose(metrics["slippage_bps"].astype(float), float(slippage_bps))
    ]
    if len(rows) != 1:
        raise RuntimeError(
            "expected exactly one aggregate replay policy row for "
            f"top_frac={threshold_top_frac} cost={cost_bps} slippage={slippage_bps}, got {len(rows)}"
        )
    return rows.iloc[0]


def _replay_evidence(
    replay_dir: Path,
    *,
    threshold_top_frac: float,
    cost_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    metrics = pd.read_csv(replay_dir / "replay_policy_metrics.csv")
    monthly = pd.read_csv(replay_dir / "replay_policy_monthly.csv")
    row = _select_policy_row(
        metrics,
        threshold_top_frac=threshold_top_frac,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    )
    policy_id = str(row["policy_id"])
    fold_rows = metrics[(metrics["scope"].astype(str) == "fold") & (metrics["policy_id"].astype(str) == policy_id)]
    month_rows = monthly[monthly["policy_id"].astype(str) == policy_id]
    return {
        "replay_dir": str(replay_dir),
        "policy_id": policy_id,
        "threshold_top_frac": float(threshold_top_frac),
        "cost_stress_bps": float(cost_bps),
        "slippage_bps": float(slippage_bps),
        "n_trades": int(row["n_trades"]),
        "net_sum_bps": float(row["net_sum_bps"]),
        "net_mean_bps": float(row["net_mean_bps"]),
        "win_rate": float(row["win_rate"]),
        "profit_factor": float(row["profit_factor"]),
        "max_drawdown_bps": float(row["max_drawdown_bps"]),
        "max_loss_bps": float(row["max_loss_bps"]),
        "positive_folds": int((fold_rows["net_sum_bps"] > 0).sum()),
        "total_folds": int(len(fold_rows)),
        "positive_months": int((month_rows["net_sum_bps"] > 0).sum()),
        "total_months": int(len(month_rows)),
    }


def _artifact_hash_rows(paths: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path in sorted(paths.items()):
        fp = _file_fingerprint(path)
        rows.append({"artifact": name, **fp})
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.plan_ack) != PLAN_ACK_REQUIRED:
        raise SystemExit(f"--plan-ack must be {PLAN_ACK_REQUIRED}")

    repo = Path(args.repo).expanduser().resolve()
    git_status = _git(repo, "status", "--short")
    if git_status.strip() and not bool(args.allow_dirty_worktree):
        raise SystemExit("dirty git tree; pass --allow-dirty-worktree for research packaging only")

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    source_parquet = Path(args.source_parquet).expanduser().resolve()
    primary_model_path = Path(args.primary_model_path).expanduser().resolve()
    walkforward_dir = Path(args.walkforward_dir).expanduser().resolve()
    replay_dir = Path(args.replay_dir).expanduser().resolve()
    slippage_replay_dir = Path(args.slippage_replay_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    package_dir = out_root / str(args.candidate_id)
    package_dir.mkdir(parents=True, exist_ok=True)

    split_files = _split_files(dataset_dir, ["train", "val", "test"])
    feature_names = selected_feature_names(split_files["train"])
    assert_no_xgb_feature_names(feature_names)
    fhash = feature_contract_hash(feature_names)
    n_model_features = _model_n_features(primary_model_path)
    if n_model_features != len(feature_names):
        raise RuntimeError(f"PRIMARY_MODEL_FEATURE_MISMATCH: model={n_model_features} contract={len(feature_names)}")

    wf_model_paths = sorted((walkforward_dir / "models").glob(f"*__{args.walkforward_model_name}.joblib"))
    if len(wf_model_paths) != int(args.expected_walkforward_models):
        raise RuntimeError(f"expected {args.expected_walkforward_models} walk-forward models, got {len(wf_model_paths)}")
    wf_model_rows = []
    for path in wf_model_paths:
        n_features = _model_n_features(path)
        if n_features != len(feature_names):
            raise RuntimeError(f"WALKFORWARD_MODEL_FEATURE_MISMATCH: {path} model={n_features} contract={len(feature_names)}")
        wf_model_rows.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "n_features": n_features,
        })

    main_evidence = _replay_evidence(
        replay_dir,
        threshold_top_frac=float(args.threshold_top_frac),
        cost_bps=float(args.cost_stress_bps),
        slippage_bps=0.0,
    )
    slippage_evidence = _replay_evidence(
        slippage_replay_dir,
        threshold_top_frac=float(args.threshold_top_frac),
        cost_bps=float(args.cost_stress_bps),
        slippage_bps=float(args.slippage_bps),
    )
    if main_evidence["positive_folds"] != main_evidence["total_folds"]:
        raise RuntimeError("MAIN_REPLAY_FOLD_STABILITY_FAIL")
    if main_evidence["positive_months"] != main_evidence["total_months"]:
        raise RuntimeError("MAIN_REPLAY_MONTH_STABILITY_FAIL")
    if slippage_evidence["positive_folds"] != slippage_evidence["total_folds"]:
        raise RuntimeError("SLIPPAGE_REPLAY_FOLD_STABILITY_FAIL")
    if slippage_evidence["positive_months"] != slippage_evidence["total_months"]:
        raise RuntimeError("SLIPPAGE_REPLAY_MONTH_STABILITY_FAIL")

    feature_manifest = {
        "schema_version": "entry_tabular_no_xgb_feature_manifest_v1",
        "feature_contract_hash": fhash,
        "n_features": len(feature_names),
        "features": [
            {"order": i, "name": name, "dtype": "float32"}
            for i, name in enumerate(feature_names)
        ],
        "included_blocks": ["snap[7:]", "ctx_cont", "ctx_cat"],
        "excluded_xgb_fields": EXCLUDED_XGB_FIELDS,
        "no_xgb_feature_guard": "PASS",
    }
    _write_json(package_dir / "feature_manifest.json", feature_manifest)

    policy_config = {
        "schema_version": "entry_tabular_no_xgb_policy_v1",
        "lead_policy": "top5",
        "threshold_top_frac": float(args.threshold_top_frac),
        "threshold_source": "pre_fold_validation_tail",
        "cost_stress_bps_required": float(args.cost_stress_bps),
        "slippage_stress_bps_required": float(args.slippage_bps),
        "exit_mode": str(args.exit_mode),
        "cooldown_bars": int(args.cooldown_bars),
        "max_trades_per_day": int(args.max_trades_per_day),
        "daily_loss_limit_bps": float(args.daily_loss_limit_bps),
        "min_direction_prob": float(args.min_direction_prob),
        "min_score_floor": float(args.min_score_floor),
        "size_multiplier": float(args.size_multiplier),
        "one_position_at_a_time": True,
    }
    policy_config["policy_config_hash"] = stable_json_hash(policy_config)
    _write_json(package_dir / "policy_config.json", policy_config)

    artifact_paths = {
        "primary_model": primary_model_path,
        "dataset_train": split_files["train"],
        "dataset_val": split_files["val"],
        "dataset_test": split_files["test"],
        "source_parquet": source_parquet,
        "walkforward_summary": walkforward_dir / "summary.json",
        "walkforward_metrics": walkforward_dir / "walkforward_selective_edge_metrics.csv",
        "walkforward_feature_importance_mean": walkforward_dir / "walkforward_feature_importance_mean.csv",
        "replay_summary": replay_dir / "summary.json",
        "replay_metrics": replay_dir / "replay_policy_metrics.csv",
        "replay_monthly": replay_dir / "replay_policy_monthly.csv",
        "slippage_replay_summary": slippage_replay_dir / "summary.json",
        "slippage_replay_metrics": slippage_replay_dir / "replay_policy_metrics.csv",
        "slippage_replay_monthly": slippage_replay_dir / "replay_policy_monthly.csv",
    }
    artifact_rows = _artifact_hash_rows(artifact_paths)
    for row in wf_model_rows:
        artifact_rows.append({
            "artifact": f"walkforward_model::{Path(row['path']).name}",
            "path": row["path"],
            "sha256": row["sha256"],
            "hash_mode": "full",
            "size_bytes": Path(row["path"]).stat().st_size,
        })
    _write_rows(package_dir / "artifact_hashes.csv", artifact_rows)

    manifest = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "candidate_id": str(args.candidate_id),
        "created_utc": _utc_now(),
        "package_status": PACKAGE_STATUS,
        "promotion_allowed": False,
        "live_ready": False,
        "plan_ack": str(args.plan_ack),
        "git": {
            "repo": str(repo),
            "commit": _git(repo, "rev-parse", "HEAD"),
            "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git_status.strip()),
            "dirty_status": git_status.splitlines() if git_status.strip() else [],
        },
        "paths": {
            "package_dir": str(package_dir),
            "dataset_dir": str(dataset_dir),
            "source_parquet": str(source_parquet),
            "primary_model_path": str(primary_model_path),
            "walkforward_dir": str(walkforward_dir),
            "replay_dir": str(replay_dir),
            "slippage_replay_dir": str(slippage_replay_dir),
        },
        "feature_contract": {
            "feature_manifest_path": str(package_dir / "feature_manifest.json"),
            "feature_contract_hash": fhash,
            "n_features": len(feature_names),
            "included_blocks": ["snap[7:]", "ctx_cont", "ctx_cat"],
            "excluded_xgb_fields": EXCLUDED_XGB_FIELDS,
            "no_xgb_feature_guard": "PASS",
        },
        "model": {
            "primary_model_path": str(primary_model_path),
            "primary_model_sha256": sha256_file(primary_model_path),
            "n_features": n_model_features,
            "walkforward_model_name": str(args.walkforward_model_name),
            "walkforward_models": wf_model_rows,
        },
        "policy_config_path": str(package_dir / "policy_config.json"),
        "policy_config_hash": policy_config["policy_config_hash"],
        "evidence": {
            "main_replay": main_evidence,
            "slippage_replay": slippage_evidence,
        },
        "hard_guards": {
            "xgb_required_for_future_architecture": False,
            "reject_xgb_derived_features": True,
            "reject_probability_feature_names": True,
            "allow_live_pin": False,
            "next_required_gate": "serve_parity_verification_then_shadow_paper_gate",
        },
    }
    manifest["candidate_manifest_hash"] = stable_json_hash(manifest)
    _write_json(package_dir / "candidate_manifest.json", manifest)

    go_no_go = {
        "candidate_id": str(args.candidate_id),
        "package_status": PACKAGE_STATUS,
        "offline_replay_gate": "PASS",
        "serve_parity_gate": "PENDING",
        "shadow_paper_gate": "PENDING",
        "promotion_allowed": False,
        "live_ready": False,
        "decision": "NO_LIVE_PIN",
        "next_required_gate": "serve_parity_verification_then_shadow_paper_gate",
    }
    _write_json(package_dir / "candidate_go_no_go.json", go_no_go)
    readme = "\n".join([
        f"# {args.candidate_id}",
        "",
        "Status: NOT PROMOTED, NOT LIVE READY.",
        "",
        "This package records the current tabular no-XGB Entry lead policy.",
        "It is evidence for the next serve-parity/shadow gate, not a production pin.",
        "",
        f"- feature_contract_hash: `{fhash}`",
        f"- policy_config_hash: `{policy_config['policy_config_hash']}`",
        "- lead policy: top5 threshold calibrated from pre-fold validation tail",
        "- XGB-derived fields: forbidden",
        "",
    ])
    (package_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True, default=json_default))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan-ack", required=True)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--source-parquet", required=True)
    ap.add_argument("--primary-model-path", required=True)
    ap.add_argument("--walkforward-dir", required=True)
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--slippage-replay-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--threshold-top-frac", type=float, required=True)
    ap.add_argument("--cost-stress-bps", type=float, required=True)
    ap.add_argument("--slippage-bps", type=float, required=True)
    ap.add_argument("--exit-mode", choices=("horizon", "stop_tp"), required=True)
    ap.add_argument("--cooldown-bars", type=int, required=True)
    ap.add_argument("--max-trades-per-day", type=int, required=True)
    ap.add_argument("--daily-loss-limit-bps", type=float, required=True)
    ap.add_argument("--min-direction-prob", type=float, required=True)
    ap.add_argument("--min-score-floor", type=float, required=True)
    ap.add_argument("--size-multiplier", type=float, required=True)
    ap.add_argument("--walkforward-model-name", default="lightgbm_tabular_no_xgb_wf")
    ap.add_argument("--expected-walkforward-models", type=int, default=8)
    ap.add_argument("--repo", default="/home/andre2/src/GX1_ENGINE")
    ap.add_argument("--allow-dirty-worktree", action="store_true")
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
