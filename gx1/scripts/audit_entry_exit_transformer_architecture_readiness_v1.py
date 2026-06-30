#!/usr/bin/env python3
"""Audit active Exit Transformer architecture/readiness contract.

This gate binds the active Exit model dataset schema to an explicit sequence
Transformer architecture contract. It is report-only: no trainer, replay, IQL,
shadow, live or promotion path is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_exit_transformer_architecture_readiness_20260630_v1"

READY_MODEL_DATASET_DECISION = "ENTRY_EXIT_MODEL_DATASET_READY_FOR_EXIT_TRANSFORMER_READINESS_REVIEW"
READY_DECISION = "ENTRY_EXIT_TRANSFORMER_ARCHITECTURE_READY_FOR_TRAINING_PLAN_REVIEW"
BLOCKED_DECISION = "BLOCKED_BY_EXIT_TRANSFORMER_ARCHITECTURE_READINESS"
REQUIRED_HEADS = (
    "exit_now_logit",
    "hold_value_bps",
    "exit_now_reward_bps",
    "giveback_risk_bps",
    "mfe_capture_ratio",
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


def _shard_paths(report: dict[str, Any]) -> dict[str, Path]:
    raw = report.get("model_dataset_shards")
    if not isinstance(raw, dict):
        return {}
    return {str(key): Path(str(value)).expanduser().resolve() for key, value in raw.items()}


def _load_shards(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for split, path in paths.items():
        out[split] = pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()
    return out


def _sequence_review(shards: dict[str, pd.DataFrame]) -> dict[str, Any]:
    ready = True
    by_split: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    max_len = 0
    for split, frame in shards.items():
        if frame.empty:
            ready = False
            by_split[split] = {"rows": 0, "episodes": 0, "max_sequence_length": 0, "ready": False}
            continue
        split_ready = True
        lengths: list[int] = []
        for episode_id, group in frame.groupby("exit_episode_id", sort=False):
            timesteps = pd.to_numeric(group["exit_timestep"], errors="coerce").astype(int).tolist()
            expected = list(range(len(group)))
            if timesteps != expected:
                split_ready = False
                failures.append({"split": split, "exit_episode_id": str(episode_id), "reason": "timesteps_not_contiguous_from_zero"})
            if int(group["is_terminal_transition"].astype(str).str.lower().isin({"true", "1"}).sum()) != 1:
                split_ready = False
                failures.append({"split": split, "exit_episode_id": str(episode_id), "reason": "terminal_count_not_one"})
            lengths.append(int(len(group)))
        max_split_len = max(lengths) if lengths else 0
        max_len = max(max_len, max_split_len)
        by_split[split] = {
            "rows": int(len(frame)),
            "episodes": int(frame["exit_episode_id"].nunique()),
            "min_sequence_length": int(min(lengths)) if lengths else 0,
            "max_sequence_length": int(max_split_len),
            "ready": bool(split_ready),
        }
        ready = ready and split_ready
    planned = int(2 ** math.ceil(math.log2(max(2, max_len))))
    planned = max(8, min(256, planned))
    return {
        "ready": bool(ready),
        "observed_max_sequence_length": int(max_len),
        "planned_max_sequence_length": int(planned),
        "by_split": by_split,
        "failure_count": int(len(failures)),
        "failures": failures[:50],
    }


def _feature_contract_review(report: dict[str, Any]) -> dict[str, Any]:
    schema = report.get("feature_schema") if isinstance(report.get("feature_schema"), dict) else {}
    normalization = report.get("normalization") if isinstance(report.get("normalization"), dict) else {}
    state_features = list(schema.get("state_feature_names") or [])
    numeric = list(schema.get("numeric_state_features") or [])
    categorical = list(schema.get("categorical_state_features") or [])
    shortcuts = [
        field for field in state_features if any(token in str(field).lower() for token in SHORTCUT_TOKENS)
    ]
    numeric_norm = normalization.get("numeric") if isinstance(normalization.get("numeric"), dict) else {}
    categorical_vocab = normalization.get("categorical_vocab") if isinstance(normalization.get("categorical_vocab"), dict) else {}
    norm_numeric_missing = sorted(set(numeric).difference(set(numeric_norm)))
    norm_cat_missing = sorted(set(categorical).difference(set(categorical_vocab)))
    return {
        "ready": bool(
            state_features
            and not shortcuts
            and set(state_features) == set(numeric).union(set(categorical))
            and not norm_numeric_missing
            and not norm_cat_missing
            and "train_split_only" in str(normalization.get("normalization_policy") or "")
        ),
        "state_feature_count": int(len(state_features)),
        "numeric_feature_count": int(len(numeric)),
        "categorical_feature_count": int(len(categorical)),
        "shortcut_overlap": shortcuts,
        "normalization_numeric_missing": norm_numeric_missing,
        "normalization_categorical_missing": norm_cat_missing,
        "normalization_policy": normalization.get("normalization_policy"),
    }


def _architecture_contract(report: dict[str, Any], sequence_review: dict[str, Any]) -> dict[str, Any]:
    schema = report.get("feature_schema") if isinstance(report.get("feature_schema"), dict) else {}
    normalization = report.get("normalization") if isinstance(report.get("normalization"), dict) else {}
    categorical_vocab = normalization.get("categorical_vocab") if isinstance(normalization.get("categorical_vocab"), dict) else {}
    categorical_embeddings = {
        field: {
            "vocab_size": int(len(values)),
            "embedding_dim": int(min(8, max(2, len(values)))),
        }
        for field, values in categorical_vocab.items()
        if isinstance(values, list)
    }
    return {
        "model_family": "exit_sequence_transformer_v1",
        "input_contract": {
            "sequence_id": "exit_episode_id",
            "timestep": "exit_timestep",
            "split": "exit_split",
            "state_feature_names": list(schema.get("state_feature_names") or []),
            "numeric_state_features": list(schema.get("numeric_state_features") or []),
            "categorical_state_features": list(schema.get("categorical_state_features") or []),
            "normalization_policy": "train_split_only",
            "categorical_embeddings": categorical_embeddings,
            "observed_max_sequence_length": sequence_review.get("observed_max_sequence_length"),
            "planned_max_sequence_length": sequence_review.get("planned_max_sequence_length"),
        },
        "sequence_encoder": {
            "architecture": "causal_masked_transformer_encoder",
            "d_model": 128,
            "n_heads": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.10,
            "position_encoding": "learned_bar_index_embedding",
            "causal_mask_required": True,
        },
        "output_heads": list(REQUIRED_HEADS),
        "training_objectives_for_later": {
            "exit_now_logit": "binary cross entropy against exit_now_label",
            "hold_value_bps": "regress future/terminal hold value from active reward contract",
            "exit_now_reward_bps": "regress current-bar exit reward",
            "giveback_risk_bps": "regress current giveback/risk proxy",
            "mfe_capture_ratio": "regress bounded MFE capture reward",
        },
        "exit_iql_bridge_for_later": {
            "state_source": "same model dataset state schema",
            "reward_source": "active state/reward contract",
            "transition_source": "next_exit_episode_id,next_exit_timestep",
            "exit_iql_allowed": False,
        },
    }


def _architecture_review(contract: dict[str, Any]) -> dict[str, Any]:
    encoder = contract.get("sequence_encoder") if isinstance(contract.get("sequence_encoder"), dict) else {}
    input_contract = contract.get("input_contract") if isinstance(contract.get("input_contract"), dict) else {}
    heads = contract.get("output_heads") if isinstance(contract.get("output_heads"), list) else []
    d_model = int(encoder.get("d_model") or 0)
    n_heads = int(encoder.get("n_heads") or 0)
    planned = int(input_contract.get("planned_max_sequence_length") or 0)
    observed = int(input_contract.get("observed_max_sequence_length") or 0)
    return {
        "ready": bool(
            contract.get("model_family") == "exit_sequence_transformer_v1"
            and d_model > 0
            and n_heads > 0
            and d_model % n_heads == 0
            and planned >= observed > 0
            and heads == list(REQUIRED_HEADS)
            and encoder.get("causal_mask_required") is True
        ),
        "d_model": d_model,
        "n_heads": n_heads,
        "planned_max_sequence_length": planned,
        "observed_max_sequence_length": observed,
        "output_heads": heads,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Exit Transformer Architecture Readiness",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Dataset rows: `{report['dataset_rows']}`",
        f"- Episode count: `{report['episode_count']}`",
        f"- Observed max sequence length: `{report['sequence_review']['observed_max_sequence_length']}`",
        f"- Planned max sequence length: `{report['sequence_review']['planned_max_sequence_length']}`",
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
    model_dataset_json = Path(args.model_dataset_json).expanduser().resolve()
    model_dataset = _read_json_or_empty(model_dataset_json)
    shard_paths = _shard_paths(model_dataset)
    shards = _load_shards(shard_paths)
    sequence_review = _sequence_review(shards) if shards else {"ready": False, "observed_max_sequence_length": 0, "planned_max_sequence_length": 0}
    feature_review = _feature_contract_review(model_dataset)
    architecture_contract = _architecture_contract(model_dataset, sequence_review)
    architecture_review = _architecture_review(architecture_contract)
    architecture_contract_path = out_dir / "entry_exit_transformer_architecture_contract.json"
    architecture_contract_path.write_text(
        json.dumps(architecture_contract, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    shard_exists = {split: path.exists() for split, path in shard_paths.items()}
    required_shards_ready = all(shard_exists.get(split) for split in ("train", "val", "test"))
    checks = [
        _check("active model dataset readiness exists", model_dataset_json.exists(), {"path": str(model_dataset_json)}),
        _check(
            "active model dataset readiness is ready",
            str(model_dataset.get("decision")) == READY_MODEL_DATASET_DECISION,
            {"decision": model_dataset.get("decision"), "required": READY_MODEL_DATASET_DECISION},
        ),
        _check("train/val/test model dataset shards exist", required_shards_ready, {"shard_exists": shard_exists}),
        _check("feature contract is exact and train-normalized", bool(feature_review.get("ready")), feature_review),
        _check("sequence episodes are contiguous and fit planned max length", bool(sequence_review.get("ready")), sequence_review),
        _check("Exit Transformer architecture contract is internally consistent", bool(architecture_review.get("ready")), architecture_review),
        _check(
            "Exit Transformer architecture readiness never trains, replays, distills, promotes, shadows, or starts live",
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
    json_path = out_dir / f"ENTRY_EXIT_TRANSFORMER_ARCHITECTURE_READINESS_{timestamp}.json"
    md_path = out_dir / f"ENTRY_EXIT_TRANSFORMER_ARCHITECTURE_READINESS_{timestamp}.md"
    report = {
        "schema_version": "entry_exit_transformer_architecture_readiness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "model_dataset_json": str(model_dataset_json),
        "model_dataset_json_sha256": _sha256_file(model_dataset_json) if model_dataset_json.exists() else "",
        "dataset_rows": int(model_dataset.get("dataset_rows") or 0),
        "episode_count": int(model_dataset.get("episode_count") or 0),
        "model_dataset_shards": {split: str(path) for split, path in shard_paths.items()},
        "model_dataset_shard_exists": shard_exists,
        "feature_contract_review": feature_review,
        "sequence_review": sequence_review,
        "architecture_contract": architecture_contract,
        "architecture_contract_json": str(architecture_contract_path),
        "architecture_contract_json_sha256": _sha256_file(architecture_contract_path),
        "architecture_review": architecture_review,
        "checks": checks,
        "failures": failures,
        "exit_training_allowed": False,
        "exit_iql_allowed": False,
        "trainer_started": False,
        "replay_started": False,
        "iql_distillation_started": False,
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            "materialize active Exit Transformer training plan/readiness manifest before any Exit training"
            if ready
            else "repair active Exit Transformer architecture readiness before training plan review"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_EXIT_TRANSFORMER_ARCHITECTURE_READINESS_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_EXIT_TRANSFORMER_ARCHITECTURE_READINESS_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": decision,
                    "dataset_rows": report["dataset_rows"],
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
    ap.add_argument("--model-dataset-json", default=str(DEFAULT_MODEL_DATASET_JSON))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
