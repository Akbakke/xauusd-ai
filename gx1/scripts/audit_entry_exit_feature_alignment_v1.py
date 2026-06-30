#!/usr/bin/env python3
"""Audit Entry-to-Exit feature alignment before any Exit training.

The active Exit state must speak the same multi-timeframe market-language as
Entry. This report checks whether the model-ready Exit dataset carries Entry
policy context plus the required HH/SMC/trend/volatility/momentum/session
families as model state. It never trains, replays, distills, promotes, shadows
or touches live paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


DEFAULT_MODEL_DATASET_JSON = (
    REPORTS_ROOT / "entry_exit_model_dataset_readiness_20260630_v1/ENTRY_EXIT_MODEL_DATASET_READINESS_latest.json"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_exit_feature_alignment_20260630_v1"

READY_MODEL_DATASET_DECISION = "ENTRY_EXIT_MODEL_DATASET_READY_FOR_EXIT_TRANSFORMER_READINESS_REVIEW"
READY_DECISION = "ENTRY_EXIT_FEATURE_ALIGNMENT_READY_FOR_EXIT_TRANSFORMER_TRAINING_REVIEW"
BLOCKED_DECISION = "BLOCKED_BY_ENTRY_EXIT_FEATURE_ALIGNMENT"
SPLITS = ("train", "val", "test")

BASE_REQUIRED_STATE_FIELDS = (
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
)
BASE_REQUIRED_PROVENANCE_FIELDS = (
    "entry_trade_id",
    "entry_iql_policy_id",
    "entry_replay_identity_hash",
)

ALIGNMENT_FAMILIES: dict[str, dict[str, Any]] = {
    "entry_policy_context": {
        "required_fields": (
            "entry_score",
            "entry_p_long",
            "entry_p_short",
            "entry_p_flat",
            "entry_path_quality_pred",
            "entry_bad_path_prob",
            "entry_iql_policy_id",
            "entry_replay_identity_hash",
        ),
        "scope": "state_or_provenance",
        "reason": "Exit must know which Entry/IQL policy produced the trade and its calibrated score/probability/path context.",
    },
    "exit_path_state": {
        "required_fields": (
            "running_pnl_bps",
            "running_mfe_bps",
            "running_mae_bps",
            "running_giveback_bps",
            "bars_held",
        ),
        "scope": "state",
        "reason": "Exit timing requires live PnL/MFE/MAE/giveback path state.",
    },
    "session_regime_cost": {
        "required_fields": ("session", "vol_regime", "side", "spread_bps", "atr_bps"),
        "scope": "state",
        "reason": "Exit must condition timing on session, regime, side and cost/ATR state.",
    },
    "structure_swing": {
        "tokens": ("hh", "hl", "lh", "ll", "bos", "choch", "swing", "structure", "pullback"),
        "scope": "state",
        "min_fields": 1,
        "expected_specialist": "structure_swing_encoder",
        "reason": "Entry structure state, breaks and pullbacks must remain visible to Exit.",
    },
    "smc_liquidity": {
        "tokens": ("smc", "sweep", "reclaim", "false_breakout", "liquidity", "premium", "discount", "support", "resistance", "level"),
        "scope": "state",
        "min_fields": 1,
        "expected_specialist": "smc_liquidity_encoder",
        "reason": "Exit must know whether the trade came from sweep/reclaim, false breakout or liquidity context.",
    },
    "trend_ema_mtf": {
        "tokens": ("ema", "trend", "tf_agreement", "mtf", "m15", "h1", "h4", "d1"),
        "scope": "state",
        "min_fields": 1,
        "expected_specialist": "trend_ema_encoder",
        "reason": "Exit must preserve the multi-timeframe trend/EMA agreement that Entry used.",
    },
    "vol_compression": {
        "tokens": ("compression", "squeeze", "expansion", "atr_pct", "atr_percentile", "range_compression"),
        "scope": "state",
        "min_fields": 1,
        "expected_specialist": "vol_compression_encoder",
        "reason": "ATR alone is insufficient; compression/expansion context must survive into Exit.",
    },
    "momentum_flow": {
        "tokens": ("momentum", "impulse", "return", "accel", "flow", "follow_through"),
        "scope": "state",
        "min_fields": 1,
        "expected_specialist": "momentum_flow_encoder",
        "reason": "Exit must know if Entry was based on impulse, continuation or fading momentum.",
    },
    "multi_timeframe_context": {
        "tokens": ("mtf", "tf_agreement", "m15", "h1", "h4", "d1"),
        "scope": "state",
        "min_fields": 1,
        "reason": "The Exit model must not collapse Entry's multi-timeframe picture into a single M5 path state.",
    },
    "entry_specialist_gate_outputs": {
        "tokens": ("specialist_gate", "gate_weight", "specialist_weight", "structure_swing_gate", "smc_liquidity_gate", "trend_ema_gate", "vol_compression_gate", "momentum_flow_gate", "session_regime_gate"),
        "scope": "state",
        "min_fields": 6,
        "reason": "Exit should see which Entry specialists agreed, disagreed or abstained.",
    },
}


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
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _path(raw: str) -> Path:
    return Path(str(raw or "")).expanduser().resolve()


def _field_matches(field: str, tokens: tuple[str, ...]) -> bool:
    low = field.lower()
    for token in tokens:
        pattern = r"(^|[^a-z0-9])" + re.escape(token.lower()) + r"([^a-z0-9]|$)"
        if re.search(pattern, low):
            return True
    return False


def _family_review(
    *,
    state_features: list[str],
    provenance_features: list[str],
    all_columns: list[str],
) -> dict[str, Any]:
    state_set = set(state_features)
    state_or_provenance = set(state_features).union(provenance_features)
    reviews: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for family, spec in ALIGNMENT_FAMILIES.items():
        scope = str(spec["scope"])
        required_fields = list(spec.get("required_fields") or [])
        tokens = tuple(spec.get("tokens") or ())
        universe = state_or_provenance if scope == "state_or_provenance" else state_set
        if required_fields:
            present = [field for field in required_fields if field in universe]
            missing_fields = [field for field in required_fields if field not in universe]
            ok = not missing_fields
        else:
            present = [field for field in state_features if _field_matches(field, tokens)]
            missing_fields = []
            ok = len(present) >= int(spec.get("min_fields") or 1)
        if not ok:
            missing.append(family)
        reviews[family] = {
            "ok": bool(ok),
            "scope": scope,
            "reason": spec.get("reason"),
            "expected_specialist": spec.get("expected_specialist"),
            "required_fields": required_fields,
            "tokens": list(tokens),
            "present_fields": present,
            "missing_required_fields": missing_fields,
            "present_anywhere_not_state": [
                field for field in all_columns if field not in state_set and (field in required_fields or (tokens and _field_matches(field, tokens)))
            ],
            "min_fields": spec.get("min_fields"),
        }
    return {"ready": not missing, "missing_families": missing, "families": reviews}


def _load_shards(shards: dict[str, str], columns: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    usecols = list(dict.fromkeys(columns))
    for split in SPLITS:
        path = _path(str(shards.get(split) or ""))
        if path.is_file():
            out[split] = pd.read_csv(path, usecols=lambda col: col in usecols, low_memory=False)
        else:
            out[split] = pd.DataFrame()
    return out


def _liveness_review(frames: dict[str, pd.DataFrame], fields: list[str]) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    ready = True
    for split in SPLITS:
        frame = frames.get(split, pd.DataFrame())
        split_fields: dict[str, dict[str, Any]] = {}
        split_ready = not frame.empty
        for field in fields:
            if field not in frame.columns:
                split_ready = False
                split_fields[field] = {"present": False, "ready": False}
                continue
            series = frame[field]
            if pd.api.types.is_numeric_dtype(series):
                values = pd.to_numeric(series, errors="coerce")
                finite = np.isfinite(values.to_numpy(dtype=np.float64, na_value=np.nan))
                unique_count = int(values.nunique(dropna=True))
                field_ready = bool(finite.all() and unique_count >= 1)
                split_fields[field] = {
                    "present": True,
                    "ready": field_ready,
                    "finite": bool(finite.all()),
                    "unique_count": unique_count,
                    "min": float(values.min()) if finite.all() and len(values) else None,
                    "max": float(values.max()) if finite.all() and len(values) else None,
                }
            else:
                values = series.dropna().astype(str)
                unique_count = int(values.nunique(dropna=True))
                field_ready = unique_count >= 1
                split_fields[field] = {
                    "present": True,
                    "ready": field_ready,
                    "unique_count": unique_count,
                    "examples": sorted(values.unique().tolist())[:8],
                }
            split_ready = split_ready and field_ready
        by_split[split] = {
            "rows": int(len(frame)),
            "ready": bool(split_ready),
            "fields": split_fields,
        }
        ready = ready and split_ready
    return {"ready": bool(ready), "by_split": by_split}


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Exit Feature Alignment",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Missing alignment families: `{report['family_review']['missing_families']}`",
        f"- Exit training allowed: `{report['exit_training_allowed']}`",
        f"- Exit IQL allowed: `{report['exit_iql_allowed']}`",
        f"- Next required gate: `{report['next_required_gate']}`",
        "",
        "## Family Review",
        "",
    ]
    for family, review in report["family_review"]["families"].items():
        lines.append(f"- `{family}` ready=`{review['ok']}` present=`{review['present_fields']}`")
    lines.extend(["", "## Failures", ""])
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
    model_report = _read_json_or_empty(model_dataset_json)
    feature_schema_json = _path(str(model_report.get("feature_schema_json") or ""))
    feature_schema = _read_json_or_empty(feature_schema_json)
    state_features = list(feature_schema.get("state_feature_names") or [])
    provenance_features = list(feature_schema.get("provenance_columns") or [])
    shards = model_report.get("model_dataset_shards") if isinstance(model_report.get("model_dataset_shards"), dict) else {}
    all_columns: list[str] = []
    for split in SPLITS:
        shard_path = _path(str(shards.get(split) or ""))
        if shard_path.is_file():
            all_columns = list(pd.read_csv(shard_path, nrows=0).columns)
            break
    required_base_missing = [field for field in BASE_REQUIRED_STATE_FIELDS if field not in set(state_features)]
    provenance_missing = [field for field in BASE_REQUIRED_PROVENANCE_FIELDS if field not in set(provenance_features).union(all_columns)]
    family_review = _family_review(
        state_features=state_features,
        provenance_features=provenance_features,
        all_columns=all_columns,
    )
    live_fields = list(
        dict.fromkeys(
            [
                *BASE_REQUIRED_STATE_FIELDS,
                *[
                    field
                    for review in family_review["families"].values()
                    for field in review.get("present_fields", [])
                    if field in set(state_features)
                ],
            ]
        )
    )
    frames = _load_shards(shards, live_fields)
    liveness = _liveness_review(frames, live_fields) if live_fields else {"ready": False, "by_split": {}}
    checks = [
        _check("active Exit model dataset readiness exists", model_dataset_json.is_file(), {"path": str(model_dataset_json)}),
        _check(
            "active Exit model dataset readiness is ready",
            model_report.get("decision") == READY_MODEL_DATASET_DECISION,
            {"decision": model_report.get("decision"), "required": READY_MODEL_DATASET_DECISION},
        ),
        _check("Exit model dataset feature schema exists", feature_schema_json.is_file(), {"path": str(feature_schema_json)}),
        _check("base Entry/Exit state fields are present", not required_base_missing, {"missing": required_base_missing}),
        _check("base Entry provenance fields are present", not provenance_missing, {"missing": provenance_missing}),
        _check("Entry-to-Exit market mechanism families are present as model state", bool(family_review.get("ready")), family_review),
        _check("alignment state fields are live on train/val/test", bool(liveness.get("ready")), liveness),
        _check(
            "feature alignment audit never trains, replays, distills, promotes, shadows, or starts live",
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
    json_path = out_dir / f"ENTRY_EXIT_FEATURE_ALIGNMENT_{timestamp}.json"
    md_path = out_dir / f"ENTRY_EXIT_FEATURE_ALIGNMENT_{timestamp}.md"
    report = {
        "schema_version": "entry_exit_feature_alignment_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "model_dataset_json": str(model_dataset_json),
        "model_dataset_json_sha256": _sha256_file(model_dataset_json) if model_dataset_json.is_file() else "",
        "feature_schema_json": str(feature_schema_json),
        "feature_schema_json_sha256": _sha256_file(feature_schema_json) if feature_schema_json.is_file() else "",
        "model_dataset_shards": shards,
        "state_feature_count": len(state_features),
        "state_features": state_features,
        "provenance_features": provenance_features,
        "required_alignment_families": ALIGNMENT_FAMILIES,
        "family_review": family_review,
        "liveness_review": liveness,
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
            "active Exit Transformer architecture/readiness may proceed with aligned Entry/Exit state"
            if ready
            else "extend Entry-bound Exit materializer/state-reward/model dataset with missing HH/SMC/trend/momentum/MTF/specialist-gate snapshots before Exit training"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_EXIT_FEATURE_ALIGNMENT_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_EXIT_FEATURE_ALIGNMENT_latest.md").write_text(
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
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
