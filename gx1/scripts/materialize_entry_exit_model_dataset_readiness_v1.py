#!/usr/bin/env python3
"""Materialize active Exit model dataset/readiness artifacts.

This gate turns the active Entry-bound Exit split/leakage dataset into
model-ready train/val/test shards plus feature schema and train-only
normalization metadata. It does not train Exit Transformer, does not distill
Exit IQL, does not replay, and does not open shadow/live/promotion.
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

from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


DEFAULT_SPLIT_LEAKAGE_JSON = (
    REPORTS_ROOT / "entry_exit_split_leakage_audit_20260630_v1/ENTRY_EXIT_SPLIT_LEAKAGE_AUDIT_latest.json"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_exit_model_dataset_readiness_20260630_v1"

READY_SPLIT_DECISION = "ENTRY_EXIT_SPLIT_LEAKAGE_AUDIT_READY"
READY_DECISION = "ENTRY_EXIT_MODEL_DATASET_READY_FOR_EXIT_TRANSFORMER_READINESS_REVIEW"
BLOCKED_DECISION = "BLOCKED_BY_EXIT_MODEL_DATASET_READINESS"
SPLITS = ("train", "val", "test")
REQUIRED_TARGET_COLUMNS = (
    "logged_action",
    "logged_action_id",
    "exit_now_label",
    "hold_label",
    "is_terminal_transition",
)
REQUIRED_REWARD_COLUMNS = (
    "hold_reward_bps",
    "forced_terminal_hold_reward_bps",
    "exit_now_reward_bps",
    "logged_reward_bps",
    "terminal_reward_realized_net_pnl_bps",
)
REQUIRED_TRANSITION_COLUMNS = (
    "next_exit_episode_id",
    "next_exit_timestep",
    "next_row_available",
)
PROVENANCE_COLUMNS = (
    "entry_trade_id",
    "bar_ts",
    "bar_index",
    "exit_episode_id",
    "exit_timestep",
    "exit_split",
    "entry_iql_policy_id",
    "entry_replay_identity_hash",
    "bar_price_source",
    "bar_price_source_path",
)
SHORTCUT_TOKENS = (
    "realized_",
    "reward",
    "exit_now_label",
    "hold_label",
    "is_terminal",
    "exit_time",
    "exit_reason",
    "next_exit_",
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


def _read_state_reward_report(split_report: dict[str, Any]) -> dict[str, Any]:
    raw = str(split_report.get("state_reward_json") or "").strip()
    if not raw:
        return {}
    return _read_json_or_empty(Path(raw).expanduser().resolve())


def _state_feature_contract(state_reward: dict[str, Any], split_report: dict[str, Any]) -> dict[str, Any]:
    contract = state_reward.get("state_feature_contract")
    if isinstance(contract, dict):
        return contract
    return {
        "state_feature_names": list(split_report.get("state_feature_names") or []),
        "numeric_state_features": [],
        "categorical_state_features": [],
        "forbidden_state_fields": [],
    }


def _finite_review(frame: pd.DataFrame, fields: list[str], *, require_liveness: bool) -> dict[str, Any]:
    all_finite = True
    reviews: dict[str, dict[str, Any]] = {}
    for field in fields:
        if field not in frame.columns:
            all_finite = False
            reviews[field] = {"present": False, "finite": False}
            continue
        values = pd.to_numeric(frame[field], errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype="float64", na_value=np.nan))
        missing = int(len(values) - int(finite.sum()))
        unique_count = int(values.nunique(dropna=True))
        ok = missing == 0 and (unique_count >= 2 if require_liveness else True)
        all_finite = all_finite and ok
        reviews[field] = {
            "present": True,
            "finite": missing == 0,
            "missing_or_nonfinite_count": missing,
            "unique_count": unique_count,
            "min": float(values.min()) if missing == 0 and len(values) else None,
            "max": float(values.max()) if missing == 0 and len(values) else None,
        }
    return {"ready": bool(all_finite), "fields": reviews}


def _categorical_review(frame: pd.DataFrame, fields: list[str]) -> dict[str, Any]:
    reviews: dict[str, dict[str, Any]] = {}
    ready = True
    train = frame.loc[frame["exit_split"].eq("train")]
    for field in fields:
        if field not in frame.columns:
            ready = False
            reviews[field] = {"present": False, "ready": False}
            continue
        train_vocab = sorted(str(value) for value in train[field].dropna().astype(str).unique() if str(value).strip())
        split_values: dict[str, list[str]] = {}
        unknown_by_split: dict[str, list[str]] = {}
        field_ready = bool(train_vocab)
        for split in SPLITS:
            values = sorted(str(value) for value in frame.loc[frame["exit_split"].eq(split), field].dropna().astype(str).unique() if str(value).strip())
            unknown = sorted(set(values).difference(set(train_vocab)))
            split_values[split] = values
            unknown_by_split[split] = unknown
            field_ready = field_ready and bool(values) and not unknown
        ready = ready and field_ready
        reviews[field] = {
            "present": True,
            "ready": bool(field_ready),
            "train_vocab": train_vocab,
            "split_values": split_values,
            "unknown_by_split": unknown_by_split,
        }
    return {"ready": bool(ready), "fields": reviews}


def _normalization_metadata(frame: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> dict[str, Any]:
    train = frame.loc[frame["exit_split"].eq("train")]
    numeric: dict[str, dict[str, float]] = {}
    for field in numeric_features:
        values = pd.to_numeric(train[field], errors="coerce")
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        numeric[field] = {"mean": mean, "std": std if std > 1e-9 else 1.0}
    categorical = {
        field: sorted(str(value) for value in train[field].dropna().astype(str).unique() if str(value).strip())
        for field in categorical_features
    }
    return {
        "normalization_policy": "fit_numeric_mean_std_and_categorical_vocab_on_train_split_only",
        "numeric": numeric,
        "categorical_vocab": categorical,
    }


def _split_review(frame: pd.DataFrame) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    ready = True
    for split in SPLITS:
        part = frame.loc[frame["exit_split"].eq(split)]
        actions = part["logged_action"].value_counts().to_dict() if "logged_action" in part else {}
        episodes = int(part["exit_episode_id"].nunique()) if "exit_episode_id" in part else 0
        split_ready = len(part) > 0 and episodes > 0 and actions.get("HOLD", 0) > 0 and actions.get("EXIT_NOW", 0) > 0
        ready = ready and split_ready
        by_split[split] = {
            "rows": int(len(part)),
            "episodes": episodes,
            "action_counts": {str(key): int(value) for key, value in actions.items()},
            "ready": bool(split_ready),
        }
    return {"ready": bool(ready), "by_split": by_split}


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Exit Model Dataset Readiness",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Dataset rows: `{report['dataset_rows']}`",
        f"- Episode count: `{report['episode_count']}`",
        f"- Feature count: `{len(report['feature_schema']['state_feature_names'])}`",
        f"- Exit training allowed: `{report['exit_training_allowed']}`",
        f"- Exit IQL allowed: `{report['exit_iql_allowed']}`",
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
    split_json = Path(args.split_leakage_json).expanduser().resolve()
    split_report = _read_json_or_empty(split_json)
    state_reward_report = _read_state_reward_report(split_report)
    split_dataset_path = _path_from_report(split_report, "split_dataset_csv")
    split_dataset_exists = bool(str(split_report.get("split_dataset_csv") or "").strip()) and split_dataset_path.exists()
    dataset = pd.read_csv(split_dataset_path, low_memory=False) if split_dataset_exists else pd.DataFrame()
    contract = _state_feature_contract(state_reward_report, split_report)
    state_features = list(contract.get("state_feature_names") or split_report.get("state_feature_names") or [])
    numeric_features = list(contract.get("numeric_state_features") or [])
    categorical_features = list(contract.get("categorical_state_features") or [])
    if not numeric_features and state_features:
        numeric_features = [
            field for field in state_features if field in dataset.columns and pd.api.types.is_numeric_dtype(dataset[field])
        ]
    if not categorical_features and state_features:
        categorical_features = [field for field in state_features if field not in numeric_features]
    required_columns = list(
        dict.fromkeys(
            [
                *PROVENANCE_COLUMNS,
                *state_features,
                *REQUIRED_TARGET_COLUMNS,
                *REQUIRED_REWARD_COLUMNS,
                *REQUIRED_TRANSITION_COLUMNS,
            ]
        )
    )
    missing_columns = [field for field in required_columns if field not in dataset.columns]
    shortcut_overlap = [
        field for field in state_features if any(token in str(field).lower() for token in SHORTCUT_TOKENS)
    ]
    numeric_review = _finite_review(dataset, numeric_features, require_liveness=True) if not dataset.empty else {"ready": False, "fields": {}}
    categorical_review = _categorical_review(dataset, categorical_features) if not dataset.empty else {"ready": False, "fields": {}}
    split_review = _split_review(dataset) if not dataset.empty else {"ready": False, "by_split": {}}
    reward_review = _finite_review(dataset, list(REQUIRED_REWARD_COLUMNS), require_liveness=False) if not dataset.empty else {"ready": False, "fields": {}}
    normalization = (
        _normalization_metadata(dataset, numeric_features, categorical_features)
        if not dataset.empty and not missing_columns
        else {"normalization_policy": "not_available_until_dataset_ready", "numeric": {}, "categorical_vocab": {}}
    )
    feature_schema = {
        "sequence_id": "exit_episode_id",
        "timestep": "exit_timestep",
        "split": "exit_split",
        "state_feature_names": state_features,
        "numeric_state_features": numeric_features,
        "categorical_state_features": categorical_features,
        "target_columns": list(REQUIRED_TARGET_COLUMNS),
        "reward_columns": list(REQUIRED_REWARD_COLUMNS),
        "transition_columns": list(REQUIRED_TRANSITION_COLUMNS),
        "provenance_columns": list(PROVENANCE_COLUMNS),
        "state_timing": contract.get("state_timing", "AS_OF_CLOSED_M5_BAR_T_WITH_ENTRY_SNAPSHOT"),
        "normalization_policy": normalization["normalization_policy"],
    }
    shard_paths: dict[str, str] = {}
    shard_hashes: dict[str, str] = {}
    model_columns = required_columns if not missing_columns else list(dataset.columns)
    for split in SPLITS:
        shard_path = out_dir / f"entry_exit_model_dataset_{split}.csv"
        shard = dataset.loc[dataset["exit_split"].eq(split), model_columns].copy() if not dataset.empty and not missing_columns else pd.DataFrame()
        shard.to_csv(shard_path, index=False)
        shard_paths[split] = str(shard_path)
        shard_hashes[split] = _sha256_file(shard_path)
    schema_path = out_dir / "entry_exit_model_dataset_feature_schema.json"
    normalization_path = out_dir / "entry_exit_model_dataset_normalization.json"
    schema_path.write_text(json.dumps(feature_schema, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    normalization_path.write_text(json.dumps(normalization, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    checks = [
        _check("active split/leakage audit exists", split_json.exists(), {"path": str(split_json)}),
        _check(
            "active split/leakage audit is ready",
            str(split_report.get("decision")) == READY_SPLIT_DECISION,
            {"decision": split_report.get("decision"), "required": READY_SPLIT_DECISION},
        ),
        _check("active split dataset exists", split_dataset_exists, {"split_dataset_csv": str(split_dataset_path)}),
        _check("model dataset has rows", not dataset.empty, {"rows": int(len(dataset))}),
        _check(
            "model dataset row count matches split/leakage audit",
            int(len(dataset)) == int(split_report.get("dataset_rows") or 0) and not dataset.empty,
            {"dataset_rows": int(len(dataset)), "split_report_rows": split_report.get("dataset_rows")},
        ),
        _check("all model dataset contract columns are present", not missing_columns, {"missing_columns": missing_columns}),
        _check("state features exclude reward/outcome/transition shortcuts", not shortcut_overlap, {"shortcut_overlap": shortcut_overlap}),
        _check("numeric state features are finite and live", bool(numeric_review.get("ready")), numeric_review),
        _check("categorical state features have train vocab coverage", bool(categorical_review.get("ready")), categorical_review),
        _check("train/val/test shards all have episodes and HOLD/EXIT_NOW labels", bool(split_review.get("ready")), split_review),
        _check("reward columns are finite and live", bool(reward_review.get("ready")), reward_review),
        _check(
            "model dataset readiness never trains, replays, distills, promotes, shadows, or starts live",
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
    json_path = out_dir / f"ENTRY_EXIT_MODEL_DATASET_READINESS_{timestamp}.json"
    md_path = out_dir / f"ENTRY_EXIT_MODEL_DATASET_READINESS_{timestamp}.md"
    report = {
        "schema_version": "entry_exit_model_dataset_readiness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "split_leakage_json": str(split_json),
        "split_leakage_json_sha256": _sha256_file(split_json) if split_json.exists() else "",
        "split_dataset_csv": str(split_dataset_path) if split_dataset_exists else "",
        "split_dataset_csv_sha256": _sha256_file(split_dataset_path) if split_dataset_exists else "",
        "dataset_rows": int(len(dataset)),
        "episode_count": int(dataset["exit_episode_id"].nunique()) if "exit_episode_id" in dataset.columns else 0,
        "feature_schema": feature_schema,
        "normalization": normalization,
        "feature_schema_json": str(schema_path),
        "feature_schema_json_sha256": _sha256_file(schema_path),
        "normalization_json": str(normalization_path),
        "normalization_json_sha256": _sha256_file(normalization_path),
        "model_dataset_shards": shard_paths,
        "model_dataset_shard_sha256": shard_hashes,
        "numeric_review": numeric_review,
        "categorical_review": categorical_review,
        "split_review": split_review,
        "reward_review": reward_review,
        "checks": checks,
        "failures": failures,
        "exit_training_allowed": False,
        "exit_iql_allowed": False,
        "trainer_started": False,
        "replay_started": False,
        "iql_distillation_started": False,
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            "audit active Exit Transformer architecture/readiness before any Exit training"
            if ready
            else "repair active Exit model dataset readiness before architecture/readiness review"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_EXIT_MODEL_DATASET_READINESS_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_EXIT_MODEL_DATASET_READINESS_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": decision,
                    "dataset_rows": int(len(dataset)),
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
    ap.add_argument("--split-leakage-json", default=str(DEFAULT_SPLIT_LEAKAGE_JSON))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
