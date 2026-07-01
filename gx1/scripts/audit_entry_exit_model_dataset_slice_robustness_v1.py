#!/usr/bin/env python3
"""Audit active Exit model dataset slice robustness.

This gate makes weak session/regime/side slices explicit before Exit train
execution review. It is report-only and never trains, replays, distills,
promotes, shadows or starts live.
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


DEFAULT_MODEL_DATASET_JSON = (
    REPORTS_ROOT
    / "entry_exit_model_dataset_readiness_20260630_v1/ENTRY_EXIT_MODEL_DATASET_READINESS_latest.json"
)
DEFAULT_PRETRAIN_MANIFEST_JSON = (
    REPORTS_ROOT
    / "entry_exit_transformer_pretrain_manifest_20260630_v1/ENTRY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST_latest.json"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_exit_model_dataset_slice_robustness_20260630_v1"

READY_MODEL_DATASET_DECISION = "ENTRY_EXIT_MODEL_DATASET_READY_FOR_EXIT_TRANSFORMER_READINESS_REVIEW"
READY_PRETRAIN_MANIFEST_DECISION = "ENTRY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST_READY_FOR_TRAIN_EXECUTION_REVIEW"
READY_DECISION = "ENTRY_EXIT_MODEL_DATASET_SLICE_ROBUSTNESS_READY_WITH_WEAK_SLICE_DISCLOSURE"
BLOCKED_DECISION = "BLOCKED_BY_EXIT_MODEL_DATASET_SLICE_ROBUSTNESS"
SLICE_COLUMNS = ("session", "vol_regime", "side")
REWARD_COLUMNS = (
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


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _load_shards(report: dict[str, Any]) -> dict[str, pd.DataFrame]:
    shards = report.get("model_dataset_shards") if isinstance(report.get("model_dataset_shards"), dict) else {}
    return {
        str(split): pd.read_csv(Path(str(path)).expanduser(), low_memory=False)
        for split, path in shards.items()
        if Path(str(path)).expanduser().exists()
    }


def _feature_liveness(
    frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    *,
    split: str,
    train_numeric_live_fields: set[str] | None = None,
) -> dict[str, Any]:
    numeric_rows: list[dict[str, Any]] = []
    categorical_rows: list[dict[str, Any]] = []
    for field in numeric:
        values = pd.to_numeric(frame[field], errors="coerce") if field in frame else pd.Series(dtype=float)
        finite = values.replace([np.inf, -np.inf], np.nan).dropna()
        finite_all = bool(field in frame and len(finite) == len(frame) and len(finite) > 0)
        std = float(finite.std(ddof=0)) if len(finite) else None
        live = bool(finite_all and std is not None and std > 1e-9)
        train_live = bool(train_numeric_live_fields is not None and field in train_numeric_live_fields)
        weak_nontrain_constant = bool(split != "train" and finite_all and not live and train_live)
        ready = bool(finite_all and (live or weak_nontrain_constant))
        numeric_rows.append(
            {
                "field": field,
                "present": field in frame,
                "finite_count": int(len(finite)),
                "finite_ratio": float(len(finite) / max(len(frame), 1)),
                "std": std,
                "finite_all": finite_all,
                "constant": bool(finite_all and not live),
                "live": live,
                "train_live": train_live if split != "train" else live,
                "weak_nontrain_constant_disclosure": weak_nontrain_constant,
                "ready": ready,
            }
        )
    for field in categorical:
        values = frame[field].astype(str) if field in frame else pd.Series(dtype=str)
        categorical_rows.append(
            {
                "field": field,
                "present": field in frame,
                "unique_count": int(values.nunique(dropna=False)) if field in frame else 0,
                "values": sorted(values.unique().tolist()) if field in frame else [],
                "live": bool(field in frame and values.nunique(dropna=False) >= 1),
            }
        )
    return {
        "numeric": numeric_rows,
        "categorical": categorical_rows,
        "all_numeric_finite_and_live": all(row["live"] for row in numeric_rows),
        "all_numeric_ready": all(row["ready"] for row in numeric_rows),
        "weak_numeric_feature_count": int(sum(1 for row in numeric_rows if row["weak_nontrain_constant_disclosure"])),
        "weak_numeric_features": [row for row in numeric_rows if row["weak_nontrain_constant_disclosure"]],
        "all_categorical_present": all(row["present"] for row in categorical_rows),
    }


def _split_review(
    split: str,
    frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    *,
    train_numeric_live_fields: set[str] | None = None,
) -> dict[str, Any]:
    exit_now = _bool_series(frame["exit_now_label"]) if "exit_now_label" in frame else pd.Series(dtype=bool)
    terminal = _bool_series(frame["is_terminal_transition"]) if "is_terminal_transition" in frame else pd.Series(dtype=bool)
    reward_finite: dict[str, bool] = {}
    for field in REWARD_COLUMNS:
        if field not in frame:
            reward_finite[field] = False
            continue
        values = pd.to_numeric(frame[field], errors="coerce").replace([np.inf, -np.inf], np.nan)
        reward_finite[field] = bool(values.notna().all())
    feature_liveness = _feature_liveness(
        frame,
        numeric,
        categorical,
        split=split,
        train_numeric_live_fields=train_numeric_live_fields,
    )
    return {
        "split": split,
        "rows": int(len(frame)),
        "episodes": int(frame["exit_episode_id"].nunique()) if "exit_episode_id" in frame else 0,
        "exit_now_positive": int(exit_now.sum()) if len(exit_now) else 0,
        "exit_now_negative": int((~exit_now).sum()) if len(exit_now) else 0,
        "terminal_rows": int(terminal.sum()) if len(terminal) else 0,
        "reward_finite": reward_finite,
        "feature_liveness": feature_liveness,
        "ready": bool(
            len(frame) > 0
            and int(frame["exit_episode_id"].nunique()) > 0
            and int(exit_now.sum()) > 0
            and int((~exit_now).sum()) > 0
            and int(terminal.sum()) == int(frame["exit_episode_id"].nunique())
            and all(reward_finite.values())
            and feature_liveness["all_numeric_ready"]
            and feature_liveness["all_categorical_present"]
        ),
    }


def _slice_rows(frame: pd.DataFrame, split: str, group_cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not all(col in frame for col in group_cols):
        return rows
    for keys, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        exit_now = _bool_series(group["exit_now_label"])
        rewards = pd.to_numeric(group["exit_now_reward_bps"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "split": split,
                "slice_type": "x".join(group_cols),
                "keys": {col: str(value) for col, value in zip(group_cols, keys)},
                "rows": int(len(group)),
                "episodes": int(group["exit_episode_id"].nunique()),
                "exit_now_positive": int(exit_now.sum()),
                "exit_now_negative": int((~exit_now).sum()),
                "exit_now_reward_mean_bps": float(rewards.mean()) if rewards.notna().any() else None,
                "exit_now_reward_p10_bps": float(rewards.quantile(0.10)) if rewards.notna().any() else None,
                "weak_slice": bool(group["exit_episode_id"].nunique() < 5 or len(group) < 25),
                "unsupported_slice": bool(group["exit_episode_id"].nunique() < 1 or int(exit_now.sum()) < 1),
            }
        )
    return rows


def _slice_review(shards: dict[str, pd.DataFrame]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for split, frame in shards.items():
        rows.extend(_slice_rows(frame, split, ["session"]))
        rows.extend(_slice_rows(frame, split, ["vol_regime"]))
        rows.extend(_slice_rows(frame, split, ["side"]))
        rows.extend(_slice_rows(frame, split, ["session", "side"]))
        rows.extend(_slice_rows(frame, split, ["vol_regime", "side"]))
    weak = [row for row in rows if row["weak_slice"]]
    unsupported = [row for row in rows if row["unsupported_slice"]]
    return {
        "rows": rows,
        "weak_slice_count": int(len(weak)),
        "unsupported_slice_count": int(len(unsupported)),
        "weak_slices": weak,
        "unsupported_slices": unsupported,
        "ready": bool(rows and not unsupported),
    }


def _feature_liveness_review(split_reviews: dict[str, dict[str, Any]]) -> dict[str, Any]:
    weak_rows: list[dict[str, Any]] = []
    blocking_rows: list[dict[str, Any]] = []
    strict_dead_rows: list[dict[str, Any]] = []
    for split, review in split_reviews.items():
        liveness = review.get("feature_liveness") if isinstance(review.get("feature_liveness"), dict) else {}
        for row in liveness.get("numeric") or []:
            if not isinstance(row, dict):
                continue
            if row.get("weak_nontrain_constant_disclosure"):
                weak_rows.append({"split": split, **row})
            if not row.get("live"):
                strict_dead_rows.append({"split": split, **row})
            if not row.get("ready"):
                blocking_rows.append({"split": split, **row})
    return {
        "all_numeric_finite_and_live_strict": bool(not strict_dead_rows),
        "all_numeric_ready": bool(not blocking_rows),
        "weak_numeric_feature_count": int(len(weak_rows)),
        "weak_numeric_features": weak_rows,
        "blocking_numeric_feature_count": int(len(blocking_rows)),
        "blocking_numeric_features": blocking_rows,
        "strict_nonlive_numeric_feature_count": int(len(strict_dead_rows)),
        "strict_nonlive_numeric_features": strict_dead_rows,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Exit Model Dataset Slice Robustness",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Weak slice count: `{report['slice_review']['weak_slice_count']}`",
        f"- Weak numeric feature disclosures: `{report['feature_liveness_review']['weak_numeric_feature_count']}`",
        f"- Unsupported slice count: `{report['slice_review']['unsupported_slice_count']}`",
        f"- Exit training allowed: `{report['exit_training_allowed']}`",
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
    model_dataset_json = Path(args.model_dataset_json).expanduser().resolve()
    pretrain_manifest_json = Path(args.pretrain_manifest_json).expanduser().resolve()
    model_dataset = _read_json_or_empty(model_dataset_json)
    pretrain_manifest = _read_json_or_empty(pretrain_manifest_json)
    schema = model_dataset.get("feature_schema") if isinstance(model_dataset.get("feature_schema"), dict) else {}
    numeric = list(schema.get("numeric_state_features") or [])
    categorical = list(schema.get("categorical_state_features") or [])
    shards = _load_shards(model_dataset)
    train_liveness = _feature_liveness(
        shards["train"],
        numeric,
        categorical,
        split="train",
    ) if "train" in shards else {"numeric": []}
    train_numeric_live_fields = {
        str(row.get("field"))
        for row in train_liveness.get("numeric", [])
        if isinstance(row, dict) and row.get("live")
    }
    split_reviews = {
        split: _split_review(
            split,
            frame,
            numeric,
            categorical,
            train_numeric_live_fields=train_numeric_live_fields,
        )
        for split, frame in shards.items()
    }
    feature_liveness_review = _feature_liveness_review(split_reviews)
    slice_review = _slice_review(shards)
    required_splits = {"train", "val", "test"}
    checks = [
        _check("active Exit model dataset readiness exists", model_dataset_json.exists(), {"path": str(model_dataset_json)}),
        _check(
            "active Exit model dataset readiness is ready",
            str(model_dataset.get("decision")) == READY_MODEL_DATASET_DECISION,
            {"decision": model_dataset.get("decision"), "required": READY_MODEL_DATASET_DECISION},
        ),
        _check("active Exit Transformer pretrain manifest exists", pretrain_manifest_json.exists(), {"path": str(pretrain_manifest_json)}),
        _check(
            "active Exit Transformer pretrain manifest is ready",
            str(pretrain_manifest.get("decision")) == READY_PRETRAIN_MANIFEST_DECISION,
            {"decision": pretrain_manifest.get("decision"), "required": READY_PRETRAIN_MANIFEST_DECISION},
        ),
        _check("train/val/test Exit model dataset shards loaded", required_splits.issubset(set(shards)), {"loaded_splits": sorted(shards)}),
        _check("split-level labels rewards and state features are live", all(row.get("ready") for row in split_reviews.values()), split_reviews),
        _check("session/regime/side slices are disclosed without unsupported slices", bool(slice_review.get("ready")), slice_review),
        _check(
            "slice robustness audit never trains, replays, distills, promotes, shadows, or starts live",
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
    json_path = out_dir / f"ENTRY_EXIT_MODEL_DATASET_SLICE_ROBUSTNESS_{timestamp}.json"
    md_path = out_dir / f"ENTRY_EXIT_MODEL_DATASET_SLICE_ROBUSTNESS_{timestamp}.md"
    report = {
        "schema_version": "entry_exit_model_dataset_slice_robustness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "model_dataset_json": str(model_dataset_json),
        "model_dataset_json_sha256": _sha256_file(model_dataset_json) if model_dataset_json.exists() else "",
        "pretrain_manifest_json": str(pretrain_manifest_json),
        "pretrain_manifest_json_sha256": _sha256_file(pretrain_manifest_json) if pretrain_manifest_json.exists() else "",
        "split_reviews": split_reviews,
        "feature_liveness_review": feature_liveness_review,
        "slice_review": slice_review,
        "checks": checks,
        "failures": failures,
        "exit_training_allowed": False,
        "exit_iql_allowed": False,
        "trainer_started": False,
        "replay_started": False,
        "iql_distillation_started": False,
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            "train-execution enablement review must account for weak session/regime/side slices before Exit training"
            if ready
            else "repair active Exit dataset slice robustness before train-execution review"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_EXIT_MODEL_DATASET_SLICE_ROBUSTNESS_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_EXIT_MODEL_DATASET_SLICE_ROBUSTNESS_latest.md").write_text(
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
    ap.add_argument("--model-dataset-json", default=str(DEFAULT_MODEL_DATASET_JSON))
    ap.add_argument("--pretrain-manifest-json", default=str(DEFAULT_PRETRAIN_MANIFEST_JSON))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
