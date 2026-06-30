#!/usr/bin/env python3
"""Audit active Entry-bound Exit split and leakage contract.

This report-only gate assigns deterministic time-ordered train/val/test splits
for the active Exit state/reward dataset and audits leakage surfaces before any
Exit Transformer/IQL training is considered.
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


DEFAULT_STATE_REWARD_JSON = (
    REPORTS_ROOT / "entry_exit_state_reward_contract_20260630_v1/ENTRY_EXIT_STATE_REWARD_CONTRACT_latest.json"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_exit_split_leakage_audit_20260630_v1"

READY_STATE_REWARD_DECISION = "ENTRY_EXIT_STATE_REWARD_CONTRACT_READY"
READY_DECISION = "ENTRY_EXIT_SPLIT_LEAKAGE_AUDIT_READY"
BLOCKED_DECISION = "BLOCKED_BY_EXIT_SPLIT_LEAKAGE_AUDIT"
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
REWARD_SHORTCUT_TOKENS = (
    "realized_",
    "reward",
    "exit_now_label",
    "hold_label",
    "is_terminal",
    "exit_time",
    "exit_reason",
    "next_exit_",
)
REQUIRED_REWARD_FIELDS = (
    "hold_reward_bps",
    "forced_terminal_hold_reward_bps",
    "exit_now_reward_bps",
    "logged_reward_bps",
    "terminal_reward_realized_net_pnl_bps",
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


def _assign_splits(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = frame.copy()
    df["bar_ts_dt"] = pd.to_datetime(df["bar_ts"], utc=True, errors="coerce")
    episode_open = (
        df.groupby("exit_episode_id", as_index=False)["bar_ts_dt"]
        .min()
        .rename(columns={"bar_ts_dt": "episode_open_ts"})
        .sort_values(["episode_open_ts", "exit_episode_id"])
        .reset_index(drop=True)
    )
    n = int(len(episode_open))
    n_train = int(np.floor(n * SPLIT_FRACTIONS["train"]))
    n_val = int(np.floor(n * SPLIT_FRACTIONS["val"]))
    if n >= 3:
        n_train = max(1, min(n - 2, n_train))
        n_val = max(1, min(n - n_train - 1, n_val))
    n_test = n - n_train - n_val
    if n_test <= 0 and n > 0:
        n_test = 1
        n_train = max(1, n - n_val - n_test)
    labels = ["train"] * n_train + ["val"] * n_val + ["test"] * max(0, n_test)
    episode_open["exit_split"] = labels[:n]
    out = df.merge(episode_open.loc[:, ["exit_episode_id", "episode_open_ts", "exit_split"]], on="exit_episode_id", how="left")
    return out, episode_open


def _split_temporal_review(episode_open: pd.DataFrame) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    ready = True
    for split in ("train", "val", "test"):
        part = episode_open.loc[episode_open["exit_split"].eq(split)]
        if part.empty:
            ready = False
            by_split[split] = {"episode_count": 0, "min_open_ts": None, "max_open_ts": None}
        else:
            by_split[split] = {
                "episode_count": int(len(part)),
                "min_open_ts": part["episode_open_ts"].min().isoformat(),
                "max_open_ts": part["episode_open_ts"].max().isoformat(),
            }
    if all(by_split[split]["episode_count"] for split in ("train", "val", "test")):
        ready = ready and (
            pd.Timestamp(by_split["train"]["max_open_ts"])
            < pd.Timestamp(by_split["val"]["min_open_ts"])
            < pd.Timestamp(by_split["test"]["min_open_ts"])
        )
    return {"ready": bool(ready), "by_split": by_split, "fractions": SPLIT_FRACTIONS}


def _intra_episode_review(frame: pd.DataFrame) -> dict[str, Any]:
    split_counts = frame.groupby("exit_episode_id")["exit_split"].nunique(dropna=False)
    bad = split_counts.loc[split_counts != 1]
    return {
        "ready": bool(bad.empty),
        "bad_episode_count": int(len(bad)),
        "bad_episode_sample": [str(idx) for idx in bad.index[:20]],
    }


def _next_pointer_split_review(frame: pd.DataFrame) -> dict[str, Any]:
    split_by_key = {
        (str(row["exit_episode_id"]), int(row["exit_timestep"])): str(row["exit_split"])
        for row in frame.loc[:, ["exit_episode_id", "exit_timestep", "exit_split"]].to_dict(orient="records")
    }
    failures: list[dict[str, Any]] = []
    nonterminal = frame.loc[~frame["is_terminal_transition"].astype(bool)]
    for row in nonterminal.to_dict(orient="records"):
        key = (str(row.get("next_exit_episode_id")), int(float(row.get("next_exit_timestep"))))
        next_split = split_by_key.get(key)
        if next_split != str(row.get("exit_split")):
            failures.append(
                {
                    "exit_episode_id": str(row.get("exit_episode_id")),
                    "exit_timestep": int(row.get("exit_timestep")),
                    "exit_split": str(row.get("exit_split")),
                    "next_key": f"{key[0]}:{key[1]}",
                    "next_split": next_split,
                }
            )
    return {"ready": not failures, "failure_count": int(len(failures)), "failures": failures[:50]}


def _state_shortcut_review(state_features: list[str]) -> dict[str, Any]:
    bad = [
        field
        for field in state_features
        if any(token in str(field).lower() for token in REWARD_SHORTCUT_TOKENS)
    ]
    return {"ready": not bad, "bad_state_features": bad, "shortcut_tokens": list(REWARD_SHORTCUT_TOKENS)}


def _reward_action_split_review(frame: pd.DataFrame) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    ready = True
    for split in ("train", "val", "test"):
        part = frame.loc[frame["exit_split"].eq(split)]
        actions = part["logged_action"].value_counts().to_dict() if "logged_action" in part.columns else {}
        reward_finite = True
        reward_missing: dict[str, int] = {}
        for field in REQUIRED_REWARD_FIELDS:
            values = pd.to_numeric(part[field], errors="coerce") if field in part.columns else pd.Series(dtype=float)
            finite = np.isfinite(values.to_numpy(dtype="float64", na_value=np.nan))
            missing = int(len(values) - int(finite.sum()))
            reward_missing[field] = missing
            reward_finite = reward_finite and len(values) > 0 and missing == 0
        split_ready = bool(len(part) and actions.get("HOLD", 0) > 0 and actions.get("EXIT_NOW", 0) > 0 and reward_finite)
        ready = ready and split_ready
        by_split[split] = {
            "rows": int(len(part)),
            "episodes": int(part["exit_episode_id"].nunique()) if "exit_episode_id" in part.columns else 0,
            "action_counts": {str(key): int(value) for key, value in actions.items()},
            "reward_missing_or_nonfinite": reward_missing,
            "ready": split_ready,
        }
    return {"ready": bool(ready), "by_split": by_split}


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Exit Split Leakage Audit",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Dataset rows: `{report['dataset_rows']}`",
        f"- Episode count: `{report['episode_count']}`",
        f"- Split dataset: `{report['split_dataset_csv']}`",
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
    state_reward_json = Path(args.state_reward_json).expanduser().resolve()
    state_reward_report = _read_json_or_empty(state_reward_json)
    dataset_path = _path_from_report(state_reward_report, "state_reward_dataset_csv")
    dataset_exists = bool(str(state_reward_report.get("state_reward_dataset_csv") or "").strip()) and dataset_path.exists()
    source = pd.read_csv(dataset_path, low_memory=False) if dataset_exists else pd.DataFrame()
    split_dataset, episode_splits = _assign_splits(source) if not source.empty else (pd.DataFrame(), pd.DataFrame())
    state_features = list(state_reward_report.get("state_feature_names") or [])
    split_dataset_csv = out_dir / "entry_exit_state_reward_dataset_with_splits.csv"
    episode_splits_csv = out_dir / "entry_exit_episode_splits.csv"
    split_dataset.to_csv(split_dataset_csv, index=False)
    episode_splits.to_csv(episode_splits_csv, index=False)
    temporal_review = _split_temporal_review(episode_splits) if not episode_splits.empty else {"ready": False, "by_split": {}}
    intra_episode_review = _intra_episode_review(split_dataset) if not split_dataset.empty else {"ready": False}
    pointer_review = _next_pointer_split_review(split_dataset) if not split_dataset.empty else {"ready": False, "failure_count": 0, "failures": []}
    shortcut_review = _state_shortcut_review(state_features)
    reward_action_review = _reward_action_split_review(split_dataset) if not split_dataset.empty else {"ready": False, "by_split": {}}
    checks = [
        _check("active state/reward contract exists", state_reward_json.exists(), {"path": str(state_reward_json)}),
        _check(
            "active state/reward contract is ready",
            str(state_reward_report.get("decision")) == READY_STATE_REWARD_DECISION,
            {"decision": state_reward_report.get("decision"), "required": READY_STATE_REWARD_DECISION},
        ),
        _check("active state/reward dataset exists", dataset_exists, {"dataset_csv": str(dataset_path)}),
        _check("split dataset has rows", not split_dataset.empty, {"rows": int(len(split_dataset))}),
        _check(
            "episode split covers all state/reward episodes",
            int(episode_splits["exit_episode_id"].nunique()) == int(state_reward_report.get("episode_count") or 0) and not episode_splits.empty,
            {
                "split_episode_count": int(episode_splits["exit_episode_id"].nunique()) if not episode_splits.empty else 0,
                "state_reward_episode_count": state_reward_report.get("episode_count"),
            },
        ),
        _check("time-ordered split has non-overlapping train/val/test windows", bool(temporal_review.get("ready")), temporal_review),
        _check("episodes are assigned to exactly one split", bool(intra_episode_review.get("ready")), intra_episode_review),
        _check("HOLD next-row pointers stay inside the same split", bool(pointer_review.get("ready")), pointer_review),
        _check("state features exclude reward/outcome shortcut fields", bool(shortcut_review.get("ready")), shortcut_review),
        _check("each split has HOLD/EXIT_NOW action and finite rewards", bool(reward_action_review.get("ready")), reward_action_review),
        _check(
            "split/leakage audit never trains, replays, distills, promotes, shadows, or starts live",
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
    json_path = out_dir / f"ENTRY_EXIT_SPLIT_LEAKAGE_AUDIT_{timestamp}.json"
    md_path = out_dir / f"ENTRY_EXIT_SPLIT_LEAKAGE_AUDIT_{timestamp}.md"
    report = {
        "schema_version": "entry_exit_split_leakage_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "state_reward_json": str(state_reward_json),
        "state_reward_json_sha256": _sha256_file(state_reward_json) if state_reward_json.exists() else "",
        "state_reward_dataset_csv": str(dataset_path) if dataset_exists else "",
        "state_reward_dataset_csv_sha256": _sha256_file(dataset_path) if dataset_exists else "",
        "split_dataset_csv": str(split_dataset_csv),
        "split_dataset_csv_sha256": _sha256_file(split_dataset_csv) if split_dataset_csv.exists() else "",
        "episode_splits_csv": str(episode_splits_csv),
        "episode_splits_csv_sha256": _sha256_file(episode_splits_csv) if episode_splits_csv.exists() else "",
        "dataset_rows": int(len(split_dataset)),
        "episode_count": int(episode_splits["exit_episode_id"].nunique()) if not episode_splits.empty else 0,
        "state_feature_names": state_features,
        "temporal_review": temporal_review,
        "intra_episode_review": intra_episode_review,
        "next_pointer_split_review": pointer_review,
        "state_shortcut_review": shortcut_review,
        "reward_action_split_review": reward_action_review,
        "checks": checks,
        "failures": failures,
        "exit_training_allowed": False,
        "exit_iql_allowed": False,
        "trainer_started": False,
        "replay_started": False,
        "iql_distillation_started": False,
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            "materialize active Exit model dataset/readiness gates before any Exit Transformer/IQL training"
            if ready
            else "repair active Exit split/leakage contract before model dataset/readiness gates"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_EXIT_SPLIT_LEAKAGE_AUDIT_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_EXIT_SPLIT_LEAKAGE_AUDIT_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": decision,
                    "dataset_rows": int(len(split_dataset)),
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
    ap.add_argument("--state-reward-json", default=str(DEFAULT_STATE_REWARD_JSON))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
