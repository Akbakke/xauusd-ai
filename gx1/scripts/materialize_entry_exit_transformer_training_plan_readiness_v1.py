#!/usr/bin/env python3
"""Materialize active Exit Transformer training plan/readiness manifest.

This gate locks the future training plan inputs, hashes, resource guardrails and
vedtak requirements for the active Exit Transformer. It is report-only: it does
not start a trainer, replay, IQL distillation, shadow, live or promotion path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


DEFAULT_ARCHITECTURE_JSON = (
    REPORTS_ROOT
    / "entry_exit_transformer_architecture_readiness_20260630_v1/ENTRY_EXIT_TRANSFORMER_ARCHITECTURE_READINESS_latest.json"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_exit_transformer_training_plan_readiness_20260630_v1"

READY_ARCHITECTURE_DECISION = "ENTRY_EXIT_TRANSFORMER_ARCHITECTURE_READY_FOR_TRAINING_PLAN_REVIEW"
READY_MODEL_DATASET_DECISION = "ENTRY_EXIT_MODEL_DATASET_READY_FOR_EXIT_TRANSFORMER_READINESS_REVIEW"
READY_DECISION = "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READY_FOR_VEDTAK_REVIEW"
BLOCKED_DECISION = "BLOCKED_BY_EXIT_TRANSFORMER_TRAINING_PLAN_READINESS"
EXPECTED_HEADS = (
    "exit_now_logit",
    "hold_value_bps",
    "exit_now_reward_bps",
    "giveback_risk_bps",
    "mfe_capture_ratio",
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


def _path(value: Any) -> Path:
    raw = str(value or "").strip()
    return Path(raw).expanduser().resolve() if raw else Path("")


def _training_plan(architecture: dict[str, Any], model_dataset: dict[str, Any]) -> dict[str, Any]:
    contract = architecture.get("architecture_contract") if isinstance(architecture.get("architecture_contract"), dict) else {}
    encoder = contract.get("sequence_encoder") if isinstance(contract.get("sequence_encoder"), dict) else {}
    input_contract = contract.get("input_contract") if isinstance(contract.get("input_contract"), dict) else {}
    feature_schema = model_dataset.get("feature_schema") if isinstance(model_dataset.get("feature_schema"), dict) else {}
    return {
        "plan_id": "entry_exit_transformer_training_plan_20260630_v1",
        "model_family": contract.get("model_family"),
        "architecture": {
            "encoder": encoder,
            "planned_max_sequence_length": input_contract.get("planned_max_sequence_length"),
            "state_feature_count": len(input_contract.get("state_feature_names") or []),
            "numeric_state_features": input_contract.get("numeric_state_features") or [],
            "categorical_state_features": input_contract.get("categorical_state_features") or [],
            "output_heads": contract.get("output_heads") or [],
        },
        "dataset": {
            "dataset_rows": model_dataset.get("dataset_rows"),
            "episode_count": model_dataset.get("episode_count"),
            "shards": model_dataset.get("model_dataset_shards") if isinstance(model_dataset.get("model_dataset_shards"), dict) else {},
            "feature_schema": feature_schema,
            "normalization_json": model_dataset.get("normalization_json"),
        },
        "future_training_command_contract": {
            "control_command": ["scripts/entry_next_edge_control.sh", "entry-exit-transformer-train", "--vedtak", "<id>"],
            "vedtak_prefix_required": "ENTRY_EXIT_TRANSFORMER_TRAIN_",
            "requires_explicit_vedtak": True,
            "requires_clean_git": True,
            "requires_pretrain_manifest": True,
            "requires_ram_guard": True,
            "touches_shadow_or_live": False,
        },
        "resource_guardrails": {
            "max_process_rss_gib": 8,
            "num_workers": 0,
            "initial_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "max_epochs_smoke": 5,
            "max_epochs_candidate": 30,
            "early_stopping_patience": 5,
            "abort_if_mem_available_below_gib": 8,
            "write_training_heartbeat_seconds": 30,
        },
        "optimization_plan": {
            "losses": {
                "exit_now_logit": "binary_cross_entropy_with_logits",
                "hold_value_bps": "huber_loss",
                "exit_now_reward_bps": "huber_loss",
                "giveback_risk_bps": "huber_loss",
                "mfe_capture_ratio": "huber_loss",
            },
            "optimizer": "AdamW",
            "learning_rate": 0.0003,
            "weight_decay": 0.01,
            "clip_grad_norm": 1.0,
            "validation_selection_metric": "val_net_reward_proxy_then_tail_risk",
        },
        "post_train_required_evidence": [
            "strict bundle load and finite forward pass",
            "exact feature schema and train-only normalization hash match",
            "all output heads present and finite",
            "non-collapsed exit_now probabilities",
            "session/regime/side/tail slices",
            "replay evidence with explicit trades before IQL",
        ],
    }


def _hash_review(architecture: dict[str, Any], model_dataset: dict[str, Any]) -> dict[str, Any]:
    arch_contract_path = _path(architecture.get("architecture_contract_json"))
    model_dataset_json_path = _path(architecture.get("model_dataset_json"))
    shard_paths = model_dataset.get("model_dataset_shards") if isinstance(model_dataset.get("model_dataset_shards"), dict) else {}
    expected_hashes = model_dataset.get("model_dataset_shard_sha256") if isinstance(model_dataset.get("model_dataset_shard_sha256"), dict) else {}
    shard_hashes: dict[str, str] = {}
    shard_mismatches: dict[str, dict[str, str]] = {}
    for split, raw in shard_paths.items():
        path = _path(raw)
        actual = _sha256_file(path) if path.exists() else ""
        expected = str(expected_hashes.get(split) or "")
        shard_hashes[str(split)] = actual
        if expected and actual != expected:
            shard_mismatches[str(split)] = {"expected": expected, "actual": actual}
    return {
        "ready": bool(
            arch_contract_path.exists()
            and model_dataset_json_path.exists()
            and shard_paths
            and not shard_mismatches
            and all(_path(raw).exists() for raw in shard_paths.values())
        ),
        "architecture_contract_json": str(arch_contract_path) if str(arch_contract_path) != "." else "",
        "architecture_contract_sha256": _sha256_file(arch_contract_path) if arch_contract_path.exists() else "",
        "model_dataset_json": str(model_dataset_json_path) if str(model_dataset_json_path) != "." else "",
        "model_dataset_json_sha256": _sha256_file(model_dataset_json_path) if model_dataset_json_path.exists() else "",
        "model_dataset_shard_sha256": shard_hashes,
        "model_dataset_shard_mismatches": shard_mismatches,
    }


def _plan_review(plan: dict[str, Any]) -> dict[str, Any]:
    command = plan.get("future_training_command_contract") if isinstance(plan.get("future_training_command_contract"), dict) else {}
    resources = plan.get("resource_guardrails") if isinstance(plan.get("resource_guardrails"), dict) else {}
    architecture = plan.get("architecture") if isinstance(plan.get("architecture"), dict) else {}
    encoder = architecture.get("encoder") if isinstance(architecture.get("encoder"), dict) else {}
    heads = architecture.get("output_heads") if isinstance(architecture.get("output_heads"), list) else []
    num_workers = resources.get("num_workers")
    return {
        "ready": bool(
            plan.get("model_family") == "exit_sequence_transformer_v1"
            and heads == list(EXPECTED_HEADS)
            and encoder.get("causal_mask_required") is True
            and command.get("requires_explicit_vedtak") is True
            and command.get("requires_clean_git") is True
            and str(command.get("vedtak_prefix_required") or "").startswith("ENTRY_EXIT_TRANSFORMER_TRAIN_")
            and num_workers is not None
            and int(num_workers) == 0
            and float(resources.get("max_process_rss_gib") or 0.0) <= 8.0
            and float(resources.get("abort_if_mem_available_below_gib") or 0.0) >= 8.0
        ),
        "output_heads": heads,
        "vedtak_prefix_required": command.get("vedtak_prefix_required"),
        "resource_guardrails": resources,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Exit Transformer Training Plan Readiness",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Dataset rows: `{report['dataset_rows']}`",
        f"- Episode count: `{report['episode_count']}`",
        f"- Exit training allowed: `{report['exit_training_allowed']}`",
        f"- Exit training allowed with explicit vedtak: `{report['exit_training_allowed_with_explicit_vedtak']}`",
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
    architecture_json = Path(args.architecture_json).expanduser().resolve()
    architecture = _read_json_or_empty(architecture_json)
    model_dataset_json = _path(architecture.get("model_dataset_json"))
    model_dataset = _read_json_or_empty(model_dataset_json) if model_dataset_json.exists() else {}
    plan = _training_plan(architecture, model_dataset)
    hash_review = _hash_review(architecture, model_dataset)
    plan_review = _plan_review(plan)
    checks = [
        _check("active Exit Transformer architecture readiness exists", architecture_json.exists(), {"path": str(architecture_json)}),
        _check(
            "active Exit Transformer architecture readiness is ready",
            str(architecture.get("decision")) == READY_ARCHITECTURE_DECISION,
            {"decision": architecture.get("decision"), "required": READY_ARCHITECTURE_DECISION},
        ),
        _check(
            "active Exit model dataset readiness is ready",
            str(model_dataset.get("decision")) == READY_MODEL_DATASET_DECISION,
            {"decision": model_dataset.get("decision"), "required": READY_MODEL_DATASET_DECISION},
        ),
        _check("architecture, model dataset and shard hashes are pinned", bool(hash_review.get("ready")), hash_review),
        _check("future training plan requires vedtak, clean git and RAM guard", bool(plan_review.get("ready")), plan_review),
        _check(
            "training plan readiness never trains, replays, distills, promotes, shadows, or starts live",
            True,
            {
                "trainer_started": False,
                "replay_started": False,
                "iql_distillation_started": False,
                "exit_training_allowed": False,
                "exit_training_allowed_with_explicit_vedtak": False,
                "exit_iql_allowed": False,
                "promotion_shadow_live_allowed": False,
            },
        ),
    ]
    failures = [{"check": check["name"], "details": check.get("details") or {}} for check in checks if not check["ok"]]
    ready = not failures
    decision = READY_DECISION if ready else BLOCKED_DECISION
    plan_path = out_dir / "entry_exit_transformer_training_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READINESS_{timestamp}.json"
    md_path = out_dir / f"ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READINESS_{timestamp}.md"
    report = {
        "schema_version": "entry_exit_transformer_training_plan_readiness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "architecture_json": str(architecture_json),
        "architecture_json_sha256": _sha256_file(architecture_json) if architecture_json.exists() else "",
        "model_dataset_json": str(model_dataset_json) if str(model_dataset_json) != "." else "",
        "model_dataset_json_sha256": _sha256_file(model_dataset_json) if model_dataset_json.exists() else "",
        "dataset_rows": int(model_dataset.get("dataset_rows") or architecture.get("dataset_rows") or 0),
        "episode_count": int(model_dataset.get("episode_count") or architecture.get("episode_count") or 0),
        "training_plan_json": str(plan_path),
        "training_plan_json_sha256": _sha256_file(plan_path),
        "training_plan": plan,
        "hash_review": hash_review,
        "plan_review": plan_review,
        "checks": checks,
        "failures": failures,
        "exit_training_allowed": False,
        "exit_training_allowed_with_explicit_vedtak": False,
        "exit_iql_allowed": False,
        "trainer_started": False,
        "replay_started": False,
        "iql_distillation_started": False,
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            "implement/audit active Exit Transformer trainer wrapper before any vedtak-gated training"
            if ready
            else "repair active Exit Transformer training plan readiness before trainer wrapper review"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READINESS_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READINESS_latest.md").write_text(
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
    ap.add_argument("--architecture-json", default=str(DEFAULT_ARCHITECTURE_JSON))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
