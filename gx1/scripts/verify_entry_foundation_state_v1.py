"""Verify the active Entry foundation-freeze state.

This verifier replaced the stale no-XGB shadow verifier after the 2026-06-28
foundation cleanup. It is intentionally operational: it proves that old Entry
candidate artifacts are out of active discovery paths, that freeze markers are
present, and that the next allowed work is explicit vedtak-gated foundation
seq146 smoke training after the train-readiness gate.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_STRUCTURE_FEATURE_NAMES,
    FOUNDATION_STRUCTURE_SOURCE_FIELDS,
)
from gx1.features.entry_specialist_feature_groups_v1 import FOUNDATION_OBJECTIVE_SPECIALISTS
from gx1.scripts.audit_entry_foundation_features_v1 import (
    DEFAULT_OUT_DIR as FEATURE_AUDIT_DEFAULT_OUT_DIR,
    REQUIRED_FOUNDATION_OBJECTIVE_FEATURES,
)
from gx1.scripts.audit_entry_foundation_targets_v1 import DEFAULT_OUT_DIR as TARGET_AUDIT_DEFAULT_OUT_DIR


REPO = Path("/home/andre2/src/GX1_ENGINE")
DATA = Path("/home/andre2/GX1_DATA")
AUDIT_DOC = REPO / "docs/ENTRY_FOUNDATION_AUDIT_20260628.md"
BLUEPRINT_DOC = REPO / "docs/ENTRY_SEQUENTIAL_AI_SPECIALIST_BLUEPRINT_20260628.md"
LEGACY_NEXT_EDGE_PLAN = REPO / "docs/ENTRY_NEXT_EDGE_PLAN_20260627.md"
REPORTS_ROOT = DATA / "reports"
RUN_ROOT = DATA / "runs/FASE2B_REGIME_V4_20260605"
SEQ_NEUTRAL_ROOT = RUN_ROOT / "v10_6yr_rebuild_20260626_seq_structure_neutral"
SPREADFIX_ROOT = RUN_ROOT / "v10_6yr_rebuild_20260626_spreadfix"
LEGACY_REPORTS = REPORTS_ROOT / "_LEGACY_ENTRY_PRE_FOUNDATION_DO_NOT_USE_20260628"
LEGACY_SEQ_NEUTRAL = SEQ_NEUTRAL_ROOT / "_LEGACY_PRE_FOUNDATION_DO_NOT_USE_20260628"
ACTIVE_NO_XGB_PACKAGE = REPORTS_ROOT / "entry_tabular_no_xgb_candidates/entry_tabular_no_xgb_top5_v1_20260627"
LEGACY_NO_XGB_PACKAGE = (
    LEGACY_REPORTS
    / "entry_pre_foundation_reports/entry_tabular_no_xgb_candidates/entry_tabular_no_xgb_top5_v1_20260627"
)
SEQ_STRUCTURE_MANIFEST = (
    REPORTS_ROOT
    / "sequence_structure_feature_layer_20260628_v1/sequence_structure_feature_layer_manifest.json"
)
FOUNDATION_DATASET_DIR = (
    RUN_ROOT
    / "v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_foundation_seq146_neutral"
)
FOUNDATION_SANITY_BUNDLE_DIR = (
    RUN_ROOT / "v10_6yr_rebuild_20260628_foundation_seq146/sanity_entry_v10_ctx_seq146"
)
FOUNDATION_SPECIALIST_SANITY_BUNDLE_DIR = (
    RUN_ROOT / "v10_6yr_rebuild_20260628_foundation_seq146/sanity_entry_v10_ctx_seq146_specialist"
)
FOUNDATION_SMOKE_DATASET_DIR = (
    RUN_ROOT / "v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_foundation_seq146_smoke"
)
FEATURE_AUDIT_LATEST = (
    REPORTS_ROOT
    / "entry_feature_foundation_audit_20260628_v1/foundation_seq146/ENTRY_FEATURE_FOUNDATION_AUDIT_latest.json"
)
TARGET_AUDIT_LATEST = (
    REPORTS_ROOT
    / "entry_target_foundation_audit_20260628_v1/foundation_seq146/ENTRY_TARGET_FOUNDATION_AUDIT_latest.json"
)
SPECIALIST_AUDIT_LATEST = (
    REPORTS_ROOT
    / "entry_specialist_feature_group_audit_20260628_v1/ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json"
)

NEXT_REQUIRED_GATE = "explicit vedtak-gated foundation seq146 specialist-fusion smoke train; promotion, shadow, and live remain blocked"


def _require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise RuntimeError(message)
    checks.append(message)


def _read_text(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"missing file: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_path_matches(reported: Any, expected: Path) -> bool:
    reported_text = str(reported or "")
    if reported_text == str(expected):
        return True
    if not reported_text:
        return False
    try:
        return Path(reported_text).expanduser().resolve() == expected.expanduser().resolve()
    except OSError:
        return False


def _active_entry_artifact_paths() -> list[str]:
    if not REPORTS_ROOT.exists():
        return []
    allowed_names = {
        "_LEGACY_ENTRY_PRE_FOUNDATION_DO_NOT_USE_20260628",
        "sequence_feature_promotion_20260628_v1",
        "sequence_structure_feature_layer_20260628_v1",
        "sequential_feature_coverage_20260628_v1",
        "exit_feature_alignment_20260628_v1",
        "entry_feature_foundation_audit_20260628_v1",
        "entry_candidate_readiness_20260628_v1",
        "entry_foundation_adoption_candidate_20260629_v1",
        "entry_foundation_activation_plan_20260629_v1",
        "entry_foundation_activation_apply_20260629_v1",
        "entry_foundation_activation_post_apply_20260629_v1",
        "entry_foundation_smoke_bundle_audit_20260628_v1",
        "entry_foundation_guardrails_20260628_v1",
        "entry_foundation_worktree_hygiene_20260628_v1",
        "entry_foundation_smoke_train_manifests_20260628_v1",
        "entry_foundation_candidate_train_manifests_20260628_v1",
        "entry_feature_ai_inventory_20260630_v1",
        "entry_chart_geometry_challenger_audit_20260630_v1",
        "entry_candlestick_pattern_challenger_audit_20260630_v1",
        "entry_specialist_challenger_extension_manifest_20260630_v1",
        "entry_seq215_manifest_provenance_repair_20260630_v1",
        "entry_candidate_bundle_audit_20260628_v1",
        "entry_iql_distillation_contract_20260628_v1",
        "entry_iql_student_trade_log_20260628_v1",
        "entry_iql_distillation_replay_20260628_v1",
        "entry_iql_replay_comparison_20260628_v1",
        "entry_iql_replay_slice_audit_20260628_v1",
        "entry_exit_per_bar_handoff_20260630_v1",
        "entry_exit_handoff_readiness_20260630_v1",
        "entry_exit_per_bar_reconstruction_audit_20260630_v1",
        "entry_exit_state_reward_contract_20260630_v1",
        "entry_exit_split_leakage_audit_20260630_v1",
        "entry_exit_model_dataset_readiness_20260630_v1",
        "entry_exit_feature_alignment_20260630_v1",
        "entry_exit_transformer_architecture_readiness_20260630_v1",
        "entry_exit_transformer_training_plan_readiness_20260630_v1",
        "entry_exit_transformer_trainer_wrapper_readiness_20260630_v1",
        "entry_exit_transformer_pretrain_manifest_20260630_v1",
        "entry_exit_model_dataset_slice_robustness_20260630_v1",
        "entry_exit_transformer_train_execution_review_20260630_v1",
        "entry_exit_transformer_post_train_contract_20260630_v1",
        "entry_candidate_selective_edge_20260628_v1",
        "entry_candidate_replay_20260628_v1",
        "entry_candidate_replay_trade_log_20260628_v1",
        "entry_candidate_replay_trade_log_20260628_v1_stop80_tp120",
        "entry_candidate_replay_tight_probe_20260630_v1",
        "entry_candidate_replay_trade_log_tight_probe_20260630_v1",
        "entry_iql_student_trade_log_probe_sl45_20260630_v1",
        "entry_iql_student_trade_log_probe_sl60_20260630_v1",
        "entry_replay_readiness_20260628_v1",
        "entry_target_foundation_audit_20260628_v1",
        "entry_training_readiness_20260628_v1",
        "entry_specialist_feature_group_audit_20260628_v1",
        "online_replay",
        "v12_live_data",
        "xgb_fixed_label_smoke_20260626",
    }
    risky_prefixes = (
        "entry_",
        "ENTRY_",
        "transformer_seq_structure",
        "v10_dataset",
        "v10_bundle",
    )
    out: list[str] = []
    for child in sorted(REPORTS_ROOT.iterdir()):
        if child.name in allowed_names:
            continue
        if child.name.startswith(risky_prefixes):
            out.append(str(child))
    return out


def _active_seq_neutral_files() -> list[str]:
    if not SEQ_NEUTRAL_ROOT.exists():
        return []
    allowed = {
        "README_FOUNDATION_FREEZE.md",
        "MULTI_TF_V2_CACHE",
        "_LEGACY_PRE_FOUNDATION_DO_NOT_USE_20260628",
    }
    return [str(p) for p in sorted(SEQ_NEUTRAL_ROOT.iterdir()) if p.name not in allowed]


def _sequence_manifest_foundation_missing() -> list[str]:
    if not SEQ_STRUCTURE_MANIFEST.exists():
        return list(FOUNDATION_STRUCTURE_FEATURE_NAMES)
    manifest = json.loads(SEQ_STRUCTURE_MANIFEST.read_text(encoding="utf-8"))
    selected = {str(x) for x in manifest.get("selected_features", [])}
    return [name for name in FOUNDATION_STRUCTURE_FEATURE_NAMES if name not in selected]


def _split_manifest(split: str) -> Path:
    matches = sorted(FOUNDATION_DATASET_DIR.glob(f"*_{split}.manifest.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {split} split manifest under {FOUNDATION_DATASET_DIR}, got {matches}")
    return matches[0]


def _split_manifest_summary() -> dict[str, Any]:
    out: dict[str, Any] = {}
    foundation_required = set(FOUNDATION_STRUCTURE_FEATURE_NAMES)
    for split in ("train", "val", "test"):
        path = _split_manifest(split)
        manifest = _read_json(path)
        extra = manifest.get("extra") or {}
        signal_bridge = extra.get("signal_bridge") or {}
        ctx_contract = extra.get("ctx_contract") or {}
        fields = [str(x) for x in signal_bridge.get("fields", [])]
        extension = signal_bridge.get("seq_structure_extension_v1") or {}
        extension_features = {str(x) for x in extension.get("features", [])}
        out[split] = {
            "manifest_path": str(path),
            "fields": len(fields),
            "seq_input_dim": int(signal_bridge.get("seq_input_dim") or 0),
            "snap_input_dim": int(signal_bridge.get("snap_input_dim") or 0),
            "seq_structure_extension_dim": int(signal_bridge.get("seq_structure_extension_dim") or 0),
            "neutral_xgb_bridge": bool(signal_bridge.get("neutral_xgb_bridge")),
            "ctx_cont_dim": int(ctx_contract.get("ctx_cont_dim") or 0),
            "ctx_cat_dim": int(ctx_contract.get("ctx_cat_dim") or 0),
            "foundation_missing": sorted(foundation_required - extension_features),
        }
    return out


def _audit_summary(path: Path, *, expected_dataset_dir: Path) -> dict[str, Any]:
    report = _read_json(path)
    return {
        "path": str(path),
        "decision": str(report.get("decision")),
        "failures": list(report.get("failures") or []),
        "dataset_dir": str(report.get("dataset_dir") or ""),
        "created_utc": str(report.get("created_utc") or ""),
        "data_splits": [str(x) for x in report.get("data_splits", [])],
        "required_foundation_liveness_families": [
            str(x) for x in report.get("required_foundation_liveness_families", [])
        ],
        "min_required_family_active_rate": float(report.get("min_required_family_active_rate") or 0.0),
        "family_liveness": list(report.get("family_liveness") or []),
        "foundation_objective_coverage_all_present": bool(
            report.get("foundation_objective_coverage_all_present")
        ),
        "foundation_objective_coverage": list(report.get("foundation_objective_coverage") or []),
        "foundation_objective_liveness_all_live": bool(
            report.get("foundation_objective_liveness_all_live")
        ),
        "foundation_objective_liveness": list(report.get("foundation_objective_liveness") or []),
        "min_required_objective_active_rate": float(report.get("min_required_objective_active_rate") or 0.0),
        "foundation_source_field_liveness_all_live": bool(
            report.get("foundation_source_field_liveness_all_live")
        ),
        "foundation_source_field_liveness": list(report.get("foundation_source_field_liveness") or []),
        "min_required_source_active_rate": float(report.get("min_required_source_active_rate") or 0.0),
        "min_required_source_active_count": int(report.get("min_required_source_active_count") or 0),
        "emitted_contracts": report.get("emitted_contracts") if isinstance(report.get("emitted_contracts"), dict) else {},
        "target_head_contract": report.get("target_head_contract") if isinstance(report.get("target_head_contract"), dict) else {},
        "matches_expected_dataset": _dataset_path_matches(report.get("dataset_dir"), expected_dataset_dir),
    }


def _audit_decision_message(label: str, path: Path, summary: dict[str, Any]) -> str:
    failures = [str(x) for x in summary.get("failures") or []]
    sample = "; ".join(failures[:3])
    suffix = f" first_failures=[{sample}]" if sample else ""
    return (
        f"{label} requires PASS: path={path} "
        f"decision={summary.get('decision')} failures={len(failures)}{suffix}"
    )


def _specialist_audit_summary(path: Path, *, expected_dataset_dir: Path) -> dict[str, Any]:
    report = _read_json(path)
    counts = {
        str(row.get("specialist")): {
            "signal_feature_count": int(row.get("signal_feature_count") or 0),
            "selected_extension_count": int(row.get("selected_extension_count") or 0),
        }
        for row in report.get("specialist_counts", [])
    }
    return {
        "path": str(path),
        "decision": str(report.get("decision")),
        "failures": list(report.get("failures") or []),
        "dataset_dir": str(report.get("dataset_dir") or ""),
        "created_utc": str(report.get("created_utc") or ""),
        "signal_field_count": int(report.get("signal_field_count") or 0),
        "selected_feature_count": int(report.get("selected_feature_count") or 0),
        "specialist_counts": counts,
        "specialist_input_liveness_all_live": bool(report.get("specialist_input_liveness_all_live")),
        "specialist_input_liveness": list(report.get("specialist_input_liveness") or []),
        "foundation_objective_routing_all_present_and_expected": bool(
            report.get("foundation_objective_routing_all_present_and_expected")
        ),
        "foundation_objective_routing": list(report.get("foundation_objective_routing") or []),
        "matches_expected_dataset": _dataset_path_matches(report.get("dataset_dir"), expected_dataset_dir),
    }


def _sanity_bundle_summary(path: Path) -> dict[str, Any]:
    lock = _read_json(path / "MASTER_TRANSFORMER_LOCK.json")
    meta = _read_json(path / "bundle_metadata.json")
    return {
        "path": str(path),
        "seq_input_dim": int(lock.get("seq_input_dim") or 0),
        "snap_input_dim": int(lock.get("snap_input_dim") or 0),
        "seq_len": int(lock.get("seq_len") or 0),
        "ctx_cont_dim": int(lock.get("ctx_cont_dim") or 0),
        "ctx_cat_dim": int(lock.get("ctx_cat_dim") or 0),
        "num_classes": int(lock.get("num_classes") or 0),
        "sanity_bundle": bool(meta.get("sanity_bundle")),
        "specialist_fusion_enabled": bool((meta.get("specialist_fusion") or {}).get("enabled")),
        "specialist_groups": sorted(((meta.get("specialist_fusion") or {}).get("input_indices") or {}).keys()),
        "created_at_utc": str(lock.get("created_at_utc") or meta.get("created_at_utc") or ""),
    }


def _smoke_dataset_summary(path: Path) -> dict[str, Any]:
    manifest_path = path / "SMOKE_DATASET_MANIFEST.json"
    if not manifest_path.exists():
        return {"path": str(path), "materialized": False, "manifest_path": str(manifest_path)}
    manifest = _read_json(manifest_path)
    splits = manifest.get("splits") if isinstance(manifest.get("splits"), dict) else {}
    return {
        "path": str(path),
        "materialized": True,
        "manifest_path": str(manifest_path),
        "created_utc": str(manifest.get("created_utc") or ""),
        "split_rows": {str(k): int((v or {}).get("rows") or 0) for k, v in splits.items()},
        "split_label_counts": {str(k): dict((v or {}).get("label_counts") or {}) for k, v in splits.items()},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[str] = []
    audit_text = _read_text(Path(args.audit_doc))
    blueprint_text = _read_text(BLUEPRINT_DOC)
    audit_flat = " ".join(audit_text.split())
    blueprint_flat = " ".join(blueprint_text.split())

    _require("foundation pass required before any more Transformer training" in audit_text, "foundation audit declares training freeze", checks)
    _require("READY_FOR_VEDTAK_SMOKE_TRAIN" in audit_text, "foundation audit documents smoke-train readiness decision", checks)
    _require("v10_dataset_foundation_seq146_neutral" in audit_text, "foundation audit documents active seq146 dataset", checks)
    _require("smoke-train --vedtak <id> --require-edge-audit" in audit_text, "foundation audit documents next gated smoke command", checks)
    _require("foundation-guardrails" in audit_text, "foundation audit documents foundation guardrails gate", checks)
    _require("Candidate training, replay-readiness, IQL distillation, IQL replay evidence" in audit_text, "foundation audit documents downstream gates remain closed", checks)
    _require("specialist-fusion contract loader accepts the current specialist audit" in audit_flat, "foundation audit documents trainer specialist loader proof", checks)
    _require("each required specialist has mean gate weight above 1%" in audit_flat, "foundation audit documents per-required-specialist gate liveness", checks)
    _require("exact target-head contract" in audit_flat, "foundation audit documents exact target-head contract", checks)
    _require("each required specialist's mean gate weight above 1%" in blueprint_flat, "specialist blueprint documents per-required-specialist gate liveness", checks)
    _require("active and blocked head sets must be exact" in blueprint_flat, "specialist blueprint documents exact target-head sets", checks)
    _require("Feature Preflight Script" in audit_text, "foundation audit defines feature preflight phase", checks)
    _require("Target Audit" in audit_text, "foundation audit defines target audit phase", checks)

    for marker in (
        RUN_ROOT / "README_FOUNDATION_FREEZE.md",
        SEQ_NEUTRAL_ROOT / "README_FOUNDATION_FREEZE.md",
        SPREADFIX_ROOT / "README_FOUNDATION_FREEZE.md",
    ):
        text = _read_text(marker)
        _require("ENTRY_FOUNDATION_AUDIT_20260628.md" in text or "foundation" in text.lower(), f"freeze marker present: {marker}", checks)

    _require(LEGACY_REPORTS.exists(), f"legacy reports root exists: {LEGACY_REPORTS}", checks)
    _require(LEGACY_SEQ_NEUTRAL.exists(), f"legacy seq-neutral root exists: {LEGACY_SEQ_NEUTRAL}", checks)
    _require(not ACTIVE_NO_XGB_PACKAGE.exists(), "no-XGB candidate package absent from active reports path", checks)
    _require(LEGACY_NO_XGB_PACKAGE.exists(), "no-XGB candidate package is archived under legacy reports", checks)

    active_entry_artifacts = _active_entry_artifact_paths()
    active_seq_neutral_files = _active_seq_neutral_files()
    _require(not active_entry_artifacts, f"no risky Entry artifacts in active reports root: {active_entry_artifacts}", checks)
    _require(not active_seq_neutral_files, f"seq-structure-neutral root contains only cache/freeze/legacy: {active_seq_neutral_files}", checks)
    foundation_missing = _sequence_manifest_foundation_missing()
    _require(
        not foundation_missing,
        f"sequence structure manifest includes all foundation structure features: missing={foundation_missing[:30]} total={len(foundation_missing)}",
        checks,
    )
    _require(FOUNDATION_DATASET_DIR.exists(), f"foundation seq146 dataset exists: {FOUNDATION_DATASET_DIR}", checks)
    for split in ("train", "val", "test"):
        parquet_files = sorted(FOUNDATION_DATASET_DIR.glob(f"*_{split}.parquet"))
        _require(len(parquet_files) == 1, f"foundation seq146 {split} parquet exists exactly once: {parquet_files}", checks)

    split_manifest_summary = _split_manifest_summary()
    for split, row in split_manifest_summary.items():
        _require(int(row["fields"]) == 146, f"{split} emitted signal field count is 146", checks)
        _require(int(row["seq_input_dim"]) == 146, f"{split} seq_input_dim is 146", checks)
        _require(int(row["snap_input_dim"]) == 146, f"{split} snap_input_dim is 146", checks)
        _require(int(row["seq_structure_extension_dim"]) == 105, f"{split} seq_structure_extension_dim is 105", checks)
        _require(bool(row["neutral_xgb_bridge"]), f"{split} neutral_xgb_bridge is active", checks)
        _require(int(row["ctx_cont_dim"]) == 142, f"{split} ctx_cont_dim is 142", checks)
        _require(int(row["ctx_cat_dim"]) == 5, f"{split} ctx_cat_dim is 5", checks)
        _require(not row["foundation_missing"], f"{split} split manifest includes all foundation features", checks)

    feature_audit = _audit_summary(FEATURE_AUDIT_LATEST, expected_dataset_dir=FOUNDATION_DATASET_DIR)
    target_audit = _audit_summary(TARGET_AUDIT_LATEST, expected_dataset_dir=FOUNDATION_DATASET_DIR)
    specialist_audit = _specialist_audit_summary(SPECIALIST_AUDIT_LATEST, expected_dataset_dir=FOUNDATION_DATASET_DIR)
    _require(
        FEATURE_AUDIT_DEFAULT_OUT_DIR == FEATURE_AUDIT_LATEST.parent,
        "feature audit default out-dir matches active seq146 latest path",
        checks,
    )
    _require(
        TARGET_AUDIT_DEFAULT_OUT_DIR == TARGET_AUDIT_LATEST.parent,
        "target audit default out-dir matches active seq146 latest path",
        checks,
    )
    _require(
        feature_audit["decision"] == "PASS",
        _audit_decision_message("feature foundation audit", FEATURE_AUDIT_LATEST, feature_audit),
        checks,
    )
    _require(not feature_audit["failures"], "feature foundation audit has zero failures", checks)
    _require(bool(feature_audit["matches_expected_dataset"]), "feature foundation audit points at active seq146 dataset", checks)
    for split, contract in feature_audit["emitted_contracts"].items():
        if not isinstance(contract, dict):
            continue
        _require(
            int(contract.get("foundation_structure_source_field_count") or 0)
            == len(FOUNDATION_STRUCTURE_SOURCE_FIELDS),
            f"{split} foundation source-field count matches contract",
            checks,
        )
        _require(
            int(contract.get("foundation_structure_source_missing_count") or 0) == 0,
            f"{split} foundation source fields are all present",
            checks,
        )
    source_liveness = {
        (str(row.get("split")), str(row.get("source_field"))): row
        for row in feature_audit["foundation_source_field_liveness"]
        if isinstance(row, dict)
    }
    _require(
        bool(feature_audit["foundation_source_field_liveness_all_live"]),
        "feature foundation audit source-field liveness is all-live",
        checks,
    )
    for split in ("train", "val", "test"):
        for source_field in FOUNDATION_STRUCTURE_SOURCE_FIELDS:
            row = source_liveness.get((split, source_field)) or {}
            _require(bool(row), f"feature foundation audit has source-field liveness for {split} {source_field}", checks)
            _require(bool(row.get("observed")), f"{split} {source_field} source field is observed", checks)
            _require(int(row.get("nonfinite_count") or 0) == 0, f"{split} {source_field} source field has zero nonfinite values", checks)
            _require(not bool(row.get("near_constant")), f"{split} {source_field} source field is not near-constant", checks)
            _require(
                int(row.get("active_count") or 0) >= int(feature_audit["min_required_source_active_count"]),
                f"{split} {source_field} source field passes minimum active-count",
                checks,
            )
            _require(
                float(row.get("active_rate") or 0.0) >= float(feature_audit["min_required_source_active_rate"]),
                f"{split} {source_field} source field passes minimum active-rate",
                checks,
            )
    required_liveness_families = {
        "foundation_hh_hl_lh_ll",
        "foundation_bos_choch_age",
        "foundation_sweep_reclaim",
        "foundation_compression_expansion",
        "foundation_impulse_pullback",
        "foundation_session_x_structure",
    }
    _require(
        required_liveness_families.issubset(set(feature_audit["required_foundation_liveness_families"])),
        "feature foundation audit declares all required roadmap liveness families",
        checks,
    )
    objective_coverage = {
        str(row.get("objective")): row
        for row in feature_audit["foundation_objective_coverage"]
        if isinstance(row, dict)
    }
    required_objectives = set(REQUIRED_FOUNDATION_OBJECTIVE_FEATURES)
    _require(
        bool(feature_audit["foundation_objective_coverage_all_present"]),
        "feature foundation audit objective coverage is all-present",
        checks,
    )
    _require(
        required_objectives.issubset(set(objective_coverage)),
        "feature foundation audit declares every exact roadmap objective",
        checks,
    )
    for objective, required_features in REQUIRED_FOUNDATION_OBJECTIVE_FEATURES.items():
        row = objective_coverage.get(objective) or {}
        _require(
            int(row.get("required_count") or 0) == len(required_features),
            f"feature objective {objective} required-count matches contract",
            checks,
        )
        _require(
            int(row.get("present_count") or 0) == len(required_features),
            f"feature objective {objective} has all required features present",
            checks,
        )
        _require(
            int(row.get("missing_count") or 0) == 0 and not row.get("missing"),
            f"feature objective {objective} has zero missing features",
            checks,
        )
    objective_liveness = {
        (str(row.get("split")), str(row.get("objective"))): row
        for row in feature_audit["foundation_objective_liveness"]
        if isinstance(row, dict)
    }
    _require(
        bool(feature_audit["foundation_objective_liveness_all_live"]),
        "feature foundation audit objective liveness is all-live",
        checks,
    )
    for split in ("train", "val", "test"):
        for objective, required_features in REQUIRED_FOUNDATION_OBJECTIVE_FEATURES.items():
            row = objective_liveness.get((split, objective)) or {}
            _require(bool(row), f"feature foundation audit has {objective} objective liveness for {split}", checks)
            _require(
                int(row.get("required_count") or 0) == len(required_features),
                f"{split} objective {objective} liveness required-count matches contract",
                checks,
            )
            _require(
                int(row.get("observed_count") or 0) == len(required_features),
                f"{split} objective {objective} liveness observes all required features",
                checks,
            )
            _require(
                int(row.get("missing_count") or 0) == 0 and not row.get("missing"),
                f"{split} objective {objective} liveness has zero missing features",
                checks,
            )
            _require(int(row.get("nonfinite_count") or 0) == 0, f"{split} objective {objective} has zero nonfinite values", checks)
            _require(int(row.get("near_constant_count") or 0) == 0, f"{split} objective {objective} has zero near-constant features", checks)
            _require(
                float(row.get("mean_active_rate") or 0.0) >= float(feature_audit["min_required_objective_active_rate"]),
                f"{split} objective {objective} passes minimum active-rate",
                checks,
            )
    family_liveness = {
        (str(row.get("split")), str(row.get("family"))): row
        for row in feature_audit["family_liveness"]
        if isinstance(row, dict)
    }
    for split in ("train", "val", "test"):
        for family in sorted(required_liveness_families):
            row = family_liveness.get((split, family)) or {}
            _require(bool(row), f"feature foundation audit has {family} liveness for {split}", checks)
            _require(int(row.get("feature_count") or 0) > 0, f"{split} {family} has feature_count > 0", checks)
            _require(int(row.get("nonfinite_count") or 0) == 0, f"{split} {family} has zero nonfinite values", checks)
            _require(int(row.get("near_constant_count") or 0) == 0, f"{split} {family} has zero near-constant features", checks)
            _require(
                float(row.get("mean_active_rate") or 0.0) >= float(feature_audit["min_required_family_active_rate"]),
                f"{split} {family} passes minimum active-rate",
                checks,
            )
    _require(
        target_audit["decision"] == "PASS",
        _audit_decision_message("target foundation audit", TARGET_AUDIT_LATEST, target_audit),
        checks,
    )
    _require(not target_audit["failures"], "target foundation audit has zero failures", checks)
    _require(bool(target_audit["matches_expected_dataset"]), "target foundation audit points at active seq146 dataset", checks)
    target_head_contract = target_audit.get("target_head_contract") or {}
    active_training_heads = set(str(x) for x in target_head_contract.get("active_training_heads", []))
    blocked_heads = set(str(x) for x in target_head_contract.get("blocked_heads", []))
    for head in (
        "tf_agreement",
        "path_quality_log_var",
        "position_size",
        "dip",
        "forecast",
        "timing",
        "tail_risk",
        "vol_forecast",
        "mtf_direction",
    ):
        _require(head in active_training_heads, f"target head contract activates {head}", checks)
    _require("hold_horizon" in blocked_heads, "target head contract blocks hold_horizon until target is live", checks)
    _require(
        specialist_audit["decision"] == "PASS",
        _audit_decision_message("specialist feature group audit", SPECIALIST_AUDIT_LATEST, specialist_audit),
        checks,
    )
    _require(not specialist_audit["failures"], "specialist feature group audit has zero failures", checks)
    _require(bool(specialist_audit["matches_expected_dataset"]), "specialist feature group audit points at active seq146 dataset", checks)
    _require(int(specialist_audit["signal_field_count"]) == 146, "specialist audit signal field count is 146", checks)
    _require(int(specialist_audit["selected_feature_count"]) == 105, "specialist audit selected feature count is 105", checks)
    _require(
        int((specialist_audit["specialist_counts"].get("unmapped") or {}).get("signal_feature_count") or 0) == 0,
        "specialist audit has zero unmapped signal fields",
        checks,
    )
    for specialist in (
        "structure_swing_encoder",
        "smc_liquidity_encoder",
        "trend_ema_encoder",
        "vol_compression_encoder",
        "momentum_flow_encoder",
        "session_regime_encoder",
    ):
        count = int((specialist_audit["specialist_counts"].get(specialist) or {}).get("signal_feature_count") or 0)
        _require(count > 0, f"specialist audit has features for {specialist}", checks)
    specialist_liveness = {
        (str(row.get("split")), str(row.get("specialist"))): row
        for row in specialist_audit["specialist_input_liveness"]
        if isinstance(row, dict)
    }
    _require(
        bool(specialist_audit["specialist_input_liveness_all_live"]),
        "specialist audit input liveness is all-live",
        checks,
    )
    for split in ("train", "val", "test"):
        for specialist in (
            "structure_swing_encoder",
            "smc_liquidity_encoder",
            "trend_ema_encoder",
            "vol_compression_encoder",
            "momentum_flow_encoder",
            "session_regime_encoder",
        ):
            row = specialist_liveness.get((split, specialist)) or {}
            _require(bool(row), f"specialist audit has {specialist} input liveness for {split}", checks)
            _require(int(row.get("feature_count") or 0) > 0, f"{split} {specialist} input feature_count > 0", checks)
            _require(int(row.get("nonfinite_count") or 0) == 0, f"{split} {specialist} input has zero nonfinite values", checks)
            _require(
                int(row.get("live_feature_count") or 0) >= int(row.get("min_required_live_feature_count") or 1),
                f"{split} {specialist} input live feature count passes minimum",
                checks,
            )
            _require(
                float(row.get("mean_active_rate") or 0.0) > 0.0,
                f"{split} {specialist} input mean active rate is positive",
                checks,
            )
    specialist_objective_routing = {
        str(row.get("objective")): row
        for row in specialist_audit["foundation_objective_routing"]
        if isinstance(row, dict)
    }
    _require(
        bool(specialist_audit["foundation_objective_routing_all_present_and_expected"]),
        "specialist audit exact foundation objective routing is all-present",
        checks,
    )
    _require(
        set(FOUNDATION_OBJECTIVE_SPECIALISTS).issubset(set(specialist_objective_routing)),
        "specialist audit declares every exact foundation objective routing",
        checks,
    )
    for objective, expected_specialist in FOUNDATION_OBJECTIVE_SPECIALISTS.items():
        row = specialist_objective_routing.get(objective) or {}
        _require(
            str(row.get("expected_specialist") or "") == expected_specialist,
            f"specialist objective {objective} expected-specialist matches contract",
            checks,
        )
        _require(
            bool(row.get("all_present_and_routed_to_expected")),
            f"specialist objective {objective} routes all exact features to expected specialist",
            checks,
        )

    sanity_bundle = _sanity_bundle_summary(FOUNDATION_SANITY_BUNDLE_DIR)
    _require(int(sanity_bundle["seq_input_dim"]) == 146, "sanity bundle seq_input_dim is 146", checks)
    _require(int(sanity_bundle["snap_input_dim"]) == 146, "sanity bundle snap_input_dim is 146", checks)
    _require(int(sanity_bundle["seq_len"]) == 96, "sanity bundle seq_len is 96", checks)
    _require(int(sanity_bundle["ctx_cont_dim"]) == 142, "sanity bundle ctx_cont_dim is 142", checks)
    _require(int(sanity_bundle["ctx_cat_dim"]) == 5, "sanity bundle ctx_cat_dim is 5", checks)
    _require(int(sanity_bundle["num_classes"]) == 3, "sanity bundle direction head has 3 classes", checks)
    _require(bool(sanity_bundle["sanity_bundle"]), "sanity bundle strict load/forward artifact exists", checks)
    specialist_sanity_bundle = _sanity_bundle_summary(FOUNDATION_SPECIALIST_SANITY_BUNDLE_DIR)
    _require(int(specialist_sanity_bundle["seq_input_dim"]) == 146, "specialist sanity bundle seq_input_dim is 146", checks)
    _require(int(specialist_sanity_bundle["snap_input_dim"]) == 146, "specialist sanity bundle snap_input_dim is 146", checks)
    _require(int(specialist_sanity_bundle["seq_len"]) == 96, "specialist sanity bundle seq_len is 96", checks)
    _require(int(specialist_sanity_bundle["ctx_cont_dim"]) == 142, "specialist sanity bundle ctx_cont_dim is 142", checks)
    _require(int(specialist_sanity_bundle["ctx_cat_dim"]) == 5, "specialist sanity bundle ctx_cat_dim is 5", checks)
    _require(int(specialist_sanity_bundle["num_classes"]) == 3, "specialist sanity bundle direction head has 3 classes", checks)
    _require(bool(specialist_sanity_bundle["sanity_bundle"]), "specialist sanity bundle strict load/forward artifact exists", checks)
    _require(bool(specialist_sanity_bundle["specialist_fusion_enabled"]), "specialist sanity bundle enables specialist fusion", checks)
    for specialist in (
        "structure_swing_encoder",
        "smc_liquidity_encoder",
        "trend_ema_encoder",
        "vol_compression_encoder",
        "momentum_flow_encoder",
        "session_regime_encoder",
    ):
        _require(
            specialist in set(specialist_sanity_bundle["specialist_groups"]),
            f"specialist sanity bundle includes {specialist}",
            checks,
        )
    smoke_dataset = _smoke_dataset_summary(FOUNDATION_SMOKE_DATASET_DIR)

    if args.selftest:
        control = _read_text(REPO / "scripts/entry_next_edge_control.sh")
        handover = _read_text(REPO / "scripts/gx1_handover.sh")
        legacy_blocker = _read_text(REPO / "scripts/entry_next_edge_legacy_block.sh")
        live_legacy_blocker = _read_text(REPO / "scripts/entry_next_edge_live_legacy_block.sh")
        launch_live_practice = _read_text(REPO / "scripts/launch_live_practice.sh")
        legacy_live_trial = _read_text(REPO / "scripts/run_live_trial160.sh")
        legacy_nightly = _read_text(REPO / "scripts/gx1_nightly_learning.sh")
        legacy_prebuilt_daemon = _read_text(REPO / "gx1/execution/v12_prebuilt_refresh_daemon.sh")
        legacy_counterfactual_daemon = _read_text(REPO / "gx1/execution/v12_daily_counterfactual.sh")
        legacy_paper_runner = _read_text(REPO / "gx1/execution/v12_paper_runner.py")
        legacy_runtime_guard = _read_text(REPO / "gx1/runtime/entry_next_edge_legacy_guard.py")
        legacy_shadow_launcher = _read_text(REPO / "scripts/run_entry_tabular_no_xgb_shadow_only.sh")
        legacy_plan_verifier = _read_text(REPO / "gx1/scripts/verify_entry_next_edge_plan_state_v1.py")
        foundation_guardrail_verifier = _read_text(REPO / "gx1/scripts/verify_entry_foundation_guardrails_v1.py")
        legacy_guardrail_verifier = _read_text(REPO / "gx1/scripts/verify_entry_next_edge_guardrails_v1.py")
        stop_live_practice = _read_text(REPO / "scripts/stop_live_practice.sh")
        smoke_wrapper = _read_text(REPO / "scripts/run_entry_foundation_seq146_smoke_train.sh")
        stage_cleanup_wrapper = _read_text(REPO / "scripts/stage_entry_foundation_cleanup.sh")
        candidate_wrapper = _read_text(REPO / "scripts/run_entry_foundation_seq146_candidate_train.sh")
        entry_exit_transformer_train_wrapper = _read_text(REPO / "scripts/run_entry_exit_transformer_train.sh")
        iql_distill_wrapper = _read_text(REPO / "scripts/run_entry_foundation_iql_distill.sh")
        smoke_bundle_audit = _read_text(REPO / "gx1/scripts/audit_entry_foundation_smoke_bundle_v1.py")
        selective_edge = _read_text(REPO / "gx1/scripts/evaluate_entry_candidate_selective_edge_v1.py")
        replay_evidence = _read_text(REPO / "gx1/scripts/materialize_entry_candidate_replay_evidence_v1.py")
        iql_replay_evidence = _read_text(REPO / "gx1/scripts/materialize_entry_iql_replay_evidence_v1.py")
        iql_comparison = _read_text(REPO / "gx1/scripts/verify_entry_iql_replay_comparison_v1.py")
        iql_slice_audit = _read_text(REPO / "gx1/scripts/audit_entry_iql_replay_slices_v1.py")
        entry_exit_materializer = _read_text(REPO / "gx1/scripts/materialize_entry_exit_per_bar_handoff_v1.py")
        entry_exit_handoff = _read_text(REPO / "gx1/scripts/audit_entry_exit_handoff_readiness_v1.py")
        entry_exit_reconstruction = _read_text(REPO / "gx1/scripts/audit_entry_exit_per_bar_reconstruction_v1.py")
        entry_exit_state_reward = _read_text(REPO / "gx1/scripts/materialize_entry_exit_state_reward_contract_v1.py")
        entry_exit_split_leakage = _read_text(REPO / "gx1/scripts/audit_entry_exit_split_leakage_v1.py")
        entry_exit_model_dataset = _read_text(REPO / "gx1/scripts/materialize_entry_exit_model_dataset_readiness_v1.py")
        entry_exit_feature_alignment = _read_text(REPO / "gx1/scripts/audit_entry_exit_feature_alignment_v1.py")
        entry_exit_transformer_architecture = _read_text(REPO / "gx1/scripts/audit_entry_exit_transformer_architecture_readiness_v1.py")
        entry_exit_transformer_training_plan = _read_text(REPO / "gx1/scripts/materialize_entry_exit_transformer_training_plan_readiness_v1.py")
        entry_exit_transformer_trainer_wrapper = _read_text(REPO / "gx1/scripts/audit_entry_exit_transformer_trainer_wrapper_readiness_v1.py")
        entry_exit_transformer_pretrain_manifest = _read_text(REPO / "gx1/scripts/materialize_entry_exit_transformer_pretrain_manifest_v1.py")
        entry_exit_model_dataset_slice_robustness = _read_text(REPO / "gx1/scripts/audit_entry_exit_model_dataset_slice_robustness_v1.py")
        entry_exit_transformer_train_execution_review = _read_text(REPO / "gx1/scripts/audit_entry_exit_transformer_train_execution_review_v1.py")
        entry_exit_transformer_post_train_contract = _read_text(REPO / "gx1/scripts/audit_entry_exit_transformer_post_train_contract_v1.py")
        entry_exit_transformer_trainer_core = _read_text(REPO / "gx1/models/exit_sequence_transformer/train_v1.py")
        worktree_hygiene = _read_text(REPO / "gx1/scripts/audit_entry_foundation_worktree_hygiene_v1.py")
        readiness = _read_text(REPO / "gx1/scripts/verify_entry_training_readiness_v1.py")
        candidate_readiness = _read_text(REPO / "gx1/scripts/verify_entry_candidate_readiness_v1.py")
        replay_readiness = _read_text(REPO / "gx1/scripts/verify_entry_replay_readiness_v1.py")
        iql_distill_contract = _read_text(REPO / "gx1/scripts/materialize_entry_iql_distillation_contract_v1.py")
        iql_student_trade_log = _read_text(REPO / "gx1/scripts/materialize_entry_iql_student_trade_log_v1.py")
        claude_head = _read_text(REPO / "CLAUDE.md")[:1600]
        agents_head = _read_text(REPO / "AGENTS.md")[:1800]
        system_map_head = _read_text(REPO / "SYSTEM_MAP.md")[:1800]
        for name, text in (
            ("CLAUDE.md active override", claude_head),
            ("AGENTS.md active override", agents_head),
            ("SYSTEM_MAP.md active override", system_map_head),
        ):
            _require("ENTRY_FOUNDATION_AUDIT_20260628.md" in text, f"{name} points at foundation audit", checks)
            _require(
                "ENTRY_SEQUENTIAL_AI_SPECIALIST_BLUEPRINT_20260628.md" in text,
                f"{name} points at sequential specialist blueprint",
                checks,
            )
            _require("READY_FOR_VEDTAK_SMOKE_TRAIN" in text, f"{name} documents smoke readiness gate", checks)
            _require("foundation-guardrails" in text, f"{name} documents foundation guardrails gate", checks)
            _require(
                "smoke-train --vedtak <id> --require-edge-audit" in text,
                f"{name} documents gated smoke command",
                checks,
            )
            _require(
                "ENTRY_NEXT_EDGE_PLAN_20260627.md" not in text,
                f"{name} no longer points at old no-XGB plan as active",
                checks,
            )
        _require("verify_entry_foundation_state_v1" in control, "control surface calls foundation verifier", checks)
        _require("foundation-freeze" in control, "control surface labels foundation-freeze", checks)
        _require("foundation-guardrails" in control, "control surface exposes foundation guardrails", checks)
        _require(
            "verify_entry_foundation_guardrails_v1" in control,
            "control surface calls foundation guardrail verifier",
            checks,
        )
        _require("worktree-hygiene" in control, "control surface exposes worktree hygiene audit", checks)
        _require(
            "audit_entry_foundation_worktree_hygiene_v1" in control,
            "control surface calls worktree hygiene audit",
            checks,
        )
        _require(
            "GX1_READINESS_REPORT_POLICY_SNAPSHOT" in control,
            "control surface supports non-refreshing readiness policy snapshot",
            checks,
        )
        _require(
            "critical_gate_ok_count" in control and "critical gate paths ok" in control,
            "control surface reports critical gate path coverage",
            checks,
        )
        _require("stage-foundation-cleanup" in control, "control surface exposes foundation cleanup staging wrapper", checks)
        _require("stage_entry_foundation_cleanup.sh" in control, "control surface calls foundation cleanup staging wrapper", checks)
        _require("materialize-smoke" in control, "control surface exposes foundation smoke materialization", checks)
        _require("train-readiness" in control, "control surface exposes train readiness gate", checks)
        _require("candidate-readiness" in control, "control surface exposes candidate readiness gate", checks)
        _require("replay-readiness" in control, "control surface exposes replay readiness gate", checks)
        _require("candidate-train" in control, "control surface exposes gated candidate train wrapper", checks)
        _require("selective-edge" in control, "control surface exposes candidate selective-edge evaluator", checks)
        _require("replay-evidence" in control, "control surface exposes candidate replay evidence materializer", checks)
        _require("iql-distill" in control, "control surface exposes gated IQL distillation contract wrapper", checks)
        _require("iql-student-trade-log" in control, "control surface exposes IQL student trade-log materializer", checks)
        _require("iql-replay-evidence" in control, "control surface exposes IQL replay evidence materializer", checks)
        _require("iql-compare" in control, "control surface exposes IQL replay comparison gate", checks)
        _require("iql-slice-audit" in control, "control surface exposes IQL replay slice audit", checks)
        _require("audit_entry_iql_replay_slices_v1" in control, "control surface calls IQL replay slice audit", checks)
        _require("entry-exit-handoff" in control, "control surface exposes Entry-to-Exit handoff audit", checks)
        _require("audit_entry_exit_handoff_readiness_v1" in control, "control surface calls Entry-to-Exit handoff audit", checks)
        _require("entry-exit-materialize" in control, "control surface exposes Entry-bound Exit per-bar materializer", checks)
        _require("materialize_entry_exit_per_bar_handoff_v1" in control, "control surface calls Entry-bound Exit per-bar materializer", checks)
        _require("entry-exit-reconstruction-audit" in control, "control surface exposes active Exit per-bar reconstruction audit", checks)
        _require("audit_entry_exit_per_bar_reconstruction_v1" in control, "control surface calls active Exit per-bar reconstruction audit", checks)
        _require("entry-exit-state-reward-contract" in control, "control surface exposes active Exit state/reward contract", checks)
        _require("materialize_entry_exit_state_reward_contract_v1" in control, "control surface calls active Exit state/reward contract", checks)
        _require("entry-exit-split-leakage-audit" in control, "control surface exposes active Exit split/leakage audit", checks)
        _require("audit_entry_exit_split_leakage_v1" in control, "control surface calls active Exit split/leakage audit", checks)
        _require("entry-exit-model-dataset-readiness" in control, "control surface exposes active Exit model dataset readiness", checks)
        _require("materialize_entry_exit_model_dataset_readiness_v1" in control, "control surface calls active Exit model dataset readiness", checks)
        _require("entry-exit-feature-alignment" in control, "control surface exposes active Entry-to-Exit feature alignment", checks)
        _require("audit_entry_exit_feature_alignment_v1" in control, "control surface calls active Entry-to-Exit feature alignment", checks)
        _require("entry-exit-transformer-architecture-readiness" in control, "control surface exposes active Exit Transformer architecture readiness", checks)
        _require("audit_entry_exit_transformer_architecture_readiness_v1" in control, "control surface calls active Exit Transformer architecture readiness", checks)
        _require("entry-exit-transformer-training-plan-readiness" in control, "control surface exposes active Exit Transformer training plan readiness", checks)
        _require("materialize_entry_exit_transformer_training_plan_readiness_v1" in control, "control surface calls active Exit Transformer training plan readiness", checks)
        _require("entry-exit-transformer-trainer-wrapper-readiness" in control, "control surface exposes active Exit Transformer trainer wrapper readiness", checks)
        _require("audit_entry_exit_transformer_trainer_wrapper_readiness_v1" in control, "control surface calls active Exit Transformer trainer wrapper readiness", checks)
        _require("entry-exit-transformer-pretrain-manifest" in control, "control surface exposes active Exit Transformer pretrain manifest", checks)
        _require("materialize_entry_exit_transformer_pretrain_manifest_v1" in control, "control surface calls active Exit Transformer pretrain manifest", checks)
        _require("entry-exit-model-dataset-slice-robustness" in control, "control surface exposes active Exit model dataset slice robustness", checks)
        _require("audit_entry_exit_model_dataset_slice_robustness_v1" in control, "control surface calls active Exit model dataset slice robustness", checks)
        _require("entry-exit-transformer-train-execution-review" in control, "control surface exposes active Exit Transformer train execution review", checks)
        _require("audit_entry_exit_transformer_train_execution_review_v1" in control, "control surface calls active Exit Transformer train execution review", checks)
        _require("entry-exit-transformer-post-train-contract" in control, "control surface exposes active Exit Transformer post-train audit contract", checks)
        _require("audit_entry_exit_transformer_post_train_contract_v1" in control, "control surface calls active Exit Transformer post-train audit contract", checks)
        _require("entry-exit-transformer-train" in control, "control surface exposes blocked active Exit Transformer train wrapper", checks)
        _require("run_entry_exit_transformer_train.sh" in control, "control surface calls blocked active Exit Transformer train wrapper", checks)
        _require("smoke-train" in control, "control surface exposes vedtak-gated smoke train", checks)
        _require("--require-edge-audit" in control, "control surface documents edge-required smoke train", checks)
        _require("audit-smoke-bundle" in control, "control surface exposes smoke bundle audit", checks)
        _require("verify_entry_foundation_state_v1" in handover, "handover redirects to foundation verifier", checks)
        _require("foundation-guardrails --quiet" in handover, "handover runs foundation guardrails", checks)
        _require("GX1_HANDOVER_SKIP_TRAIN_READINESS" in handover, "handover exposes internal train-readiness skip for guardrail tests", checks)
        _require("critical gate paths ok" in handover, "handover reports critical gate path coverage", checks)
        for name, text in (
            ("legacy Entry blocker", legacy_blocker),
            ("legacy live blocker", live_legacy_blocker),
            ("legacy live-practice launcher", launch_live_practice),
            ("legacy trial160 live launcher", legacy_live_trial),
            ("legacy nightly loop", legacy_nightly),
            ("legacy prebuilt refresh daemon", legacy_prebuilt_daemon),
            ("legacy counterfactual daemon", legacy_counterfactual_daemon),
            ("legacy paper runner", legacy_paper_runner),
            ("legacy runtime guard", legacy_runtime_guard),
        ):
            _require("Entry foundation seq146 cleanup/audit/smoke-readiness" in text, f"{name} points at foundation seq146 path", checks)
            _require("foundation-guardrails" in text, f"{name} points at foundation guardrails", checks)
            _require("worktree-hygiene" in text, f"{name} points at worktree hygiene", checks)
            _require("stage-foundation-cleanup --dry-run" in text, f"{name} points at foundation cleanup staging dry-run", checks)
            _require("materialize-smoke" in text, f"{name} points at smoke materialization", checks)
            _require("train-readiness" in text, f"{name} points at train readiness", checks)
            _require("Current path is no-XGB tabular shadow-only observation" not in text, f"{name} no longer advertises no-XGB shadow as current path", checks)
            _require("entry_next_edge_control.sh start-shadow" not in text, f"{name} no longer points at shadow start", checks)
        _require("verify_entry_foundation_state_v1" in launch_live_practice, "legacy live-practice guard uses foundation verifier", checks)
        _require("verify_entry_next_edge_plan_state_v1" not in launch_live_practice, "legacy live-practice guard no longer depends on old shadow-plan verifier", checks)
        _require("verify_entry_next_edge_plan_state_v1" not in legacy_paper_runner, "legacy paper runner no longer depends on old shadow-plan verifier", checks)
        _require("foundation-freeze blocks direct v12_paper_runner use" in legacy_paper_runner, "legacy paper runner blocks direct use under foundation freeze", checks)
        _require("blocked by active Entry foundation-freeze" in legacy_shadow_launcher, "legacy no-XGB shadow launcher fails under foundation freeze", checks)
        _require("verify_entry_next_edge_plan_state_v1" not in legacy_shadow_launcher, "legacy no-XGB shadow launcher no longer calls old plan verifier", checks)
        _require("LEGACY_PLAN_CLOSED" in legacy_plan_verifier, "old no-XGB plan verifier is a closed-plan tombstone", checks)
        _require("verify_entry_foundation_state_v1" in legacy_plan_verifier, "old no-XGB plan verifier redirects attention to foundation verifier", checks)
        _require("entry_foundation_guardrails_v1" in foundation_guardrail_verifier, "foundation guardrail verifier writes foundation schema", checks)
        _require("GX1_HANDOVER_SKIP_TRAIN_READINESS" in foundation_guardrail_verifier, "foundation guardrail verifier prevents handover readiness recursion", checks)
        _require("GX1_READINESS_REPORT_POLICY_SNAPSHOT" in foundation_guardrail_verifier, "foundation guardrail verifier uses readiness policy snapshot", checks)
        _require("readiness_policy_checks" in foundation_guardrail_verifier, "foundation guardrail verifier reports readiness policy checks", checks)
        _require(
            "blocked_downstream" in foundation_guardrail_verifier and '"candidate_train"' in foundation_guardrail_verifier,
            "foundation guardrail verifier blocks candidate train in readiness policy",
            checks,
        )
        _require(
            "blocked_downstream" in foundation_guardrail_verifier and '"iql_distill"' in foundation_guardrail_verifier,
            "foundation guardrail verifier blocks IQL in readiness policy",
            checks,
        )
        _require(
            "blocked_downstream" in foundation_guardrail_verifier and '"live"' in foundation_guardrail_verifier,
            "foundation guardrail verifier blocks live in readiness policy",
            checks,
        )
        _require("direct_no_xgb_shadow_launcher_blocked" in foundation_guardrail_verifier, "foundation guardrail verifier checks direct no-XGB shadow launcher is blocked", checks)
        _require("control_preview_shadow_blocked" in foundation_guardrail_verifier, "foundation guardrail verifier checks preview-shadow is blocked", checks)
        _require("verify_entry_foundation_guardrails_v1" in legacy_guardrail_verifier, "old guardrail verifier delegates to foundation guardrails", checks)
        _require("STOP legacy pre-foundation no-XGB shadow runner" in stop_live_practice, "stop script reaps legacy no-XGB shadow runner under foundation freeze", checks)
        _require("SKIP active Entry next-edge no-XGB shadow runner" not in stop_live_practice, "stop script no longer preserves no-XGB shadow runner", checks)
        _require("--vedtak" in smoke_wrapper, "foundation smoke train wrapper requires vedtak", checks)
        _require("gx1_capped_run.sh" in smoke_wrapper, "foundation smoke train wrapper uses capped run", checks)
        _require("foundation-guardrails --quiet" in smoke_wrapper, "foundation smoke train wrapper enforces foundation guardrails before real training", checks)
        _require("train-readiness --quiet" in smoke_wrapper, "foundation smoke train wrapper enforces train-readiness before real training", checks)
        _require("ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST" in smoke_wrapper, "foundation smoke train wrapper writes pre-train run manifest", checks)
        _require("trainer_started_by_manifest_writer" in smoke_wrapper, "foundation smoke train manifest records writer does not start trainer", checks)
        _require("--manifest-only" in smoke_wrapper, "foundation smoke train wrapper exposes manifest-only proof mode", checks)
        _require("Manifest-only stop before training" in smoke_wrapper, "foundation smoke train manifest-only mode stops before training", checks)
        _require("require_clean_git_for_real_train" in smoke_wrapper, "foundation smoke train wrapper has explicit real-train git-clean gate", checks)
        _require(
            "require_foundation_contract_ready_for_manifest_only" in smoke_wrapper,
            "foundation smoke train wrapper lets manifest-only prove foundation contract without trainer start",
            checks,
        )
        _require("--no-fail-on-not-ready" in smoke_wrapper, "foundation smoke train wrapper permits manifest-only report mode under dirty git", checks)
        _require(
            "foundation_contract_ready_for_smoke" in smoke_wrapper,
            "foundation smoke train wrapper requires foundation contract readiness for manifest-only",
            checks,
        )
        _require(
            "foundation_cleanup_critical_gate_review" in smoke_wrapper,
            "foundation smoke train manifest records critical gate path review",
            checks,
        )
        _require("git status --short" in smoke_wrapper, "foundation smoke train wrapper inspects git status before real training", checks)
        _require(
            "real foundation smoke train requires clean git worktree" in smoke_wrapper,
            "foundation smoke train wrapper blocks real training from dirty git worktree",
            checks,
        )
        _require("--enable-specialist-fusion" in smoke_wrapper, "foundation smoke train wrapper enables specialist fusion", checks)
        for flag in (
            "--enable-tf-agreement-head",
            "--enable-path-quality-variance-head",
            "--enable-position-size-head",
            "--enable-dip-head",
            "--enable-forecast-head",
            "--enable-timing-head",
            "--enable-tail-risk-head",
            "--enable-vol-forecast-head",
            "--enable-mtf-direction-head",
        ):
            _require(flag in smoke_wrapper, f"foundation smoke train wrapper enables {flag}", checks)
            _require(flag in candidate_wrapper, f"candidate train wrapper enables {flag}", checks)
        _require("--enable-hold-horizon-head" not in smoke_wrapper, "foundation smoke train wrapper keeps hold_horizon blocked", checks)
        _require("--enable-hold-horizon-head" not in candidate_wrapper, "candidate train wrapper keeps hold_horizon blocked", checks)
        _require("ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json" in smoke_wrapper, "foundation smoke train wrapper uses specialist audit contract", checks)
        _require("audit-smoke-bundle" in smoke_wrapper, "foundation smoke train wrapper points to smoke bundle audit", checks)
        _require("AUDIT_CMD" in smoke_wrapper, "foundation smoke train wrapper builds post-train audit command", checks)
        _require("--skip-smoke-audit" in smoke_wrapper, "foundation smoke train wrapper exposes explicit audit skip", checks)
        _require("--require-edge-audit" in smoke_wrapper, "foundation smoke train wrapper exposes edge audit requirement", checks)
        _require("REQUIRE_EDGE_AUDIT=1" in smoke_wrapper, "foundation smoke train wrapper requires edge audit by default", checks)
        _require("--no-require-edge-audit" in smoke_wrapper, "foundation smoke train wrapper exposes explicit plumbing-only edge-audit opt-out", checks)
        _require("cannot be combined with --skip-smoke-audit" in smoke_wrapper, "foundation smoke train wrapper rejects skipped edge audit", checks)
        _require("--require-head-contract" in smoke_wrapper, "foundation smoke train wrapper requires post-train head contract audit", checks)
        _require("load_entry_v10_ctx_bundle" in smoke_bundle_audit, "smoke bundle audit strict-loads runtime bundle", checks)
        _require("specialist_gate" in smoke_bundle_audit, "smoke bundle audit checks specialist gate", checks)
        _require("require_head_contract" in smoke_bundle_audit, "smoke bundle audit can require active target head contract", checks)
        _require("HEAD_OUTPUT_KEYS" in smoke_bundle_audit, "smoke bundle audit maps active heads to forward outputs", checks)
        _require("HEAD_OUTPUT_TRAILING_SHAPES" in smoke_bundle_audit, "smoke bundle audit maps active heads to output shapes", checks)
        _require("unapproved heads outside target contract" in smoke_bundle_audit, "smoke bundle audit rejects unapproved extra heads", checks)
        _require(
            "required_training_specialists_for_mode" in smoke_bundle_audit,
            "smoke bundle audit requires mode-aware training specialist set",
            checks,
        )
        _require("required_training_specialists" in smoke_bundle_audit, "smoke bundle audit reports required specialist set", checks)
        _require("active_specialist_count_gt_1pct" in smoke_bundle_audit, "smoke bundle audit checks active specialist count", checks)
        _require("required specialist gate weights collapsed" in smoke_bundle_audit, "smoke bundle audit rejects collapsed required specialist gate weights", checks)
        _require(
            "worktree_critical_gate_review_ok" in smoke_bundle_audit,
            "smoke bundle audit validates pretrain critical gate path review",
            checks,
        )
        _require("min_gate_entropy" in smoke_bundle_audit, "smoke bundle audit checks specialist gate entropy", checks)
        _require("majority_baseline_accuracy" in smoke_bundle_audit, "smoke bundle audit reports majority baseline", checks)
        _require("READY_FOR_VEDTAK_SMOKE_TRAIN" in readiness, "train readiness gate emits smoke-train readiness decision", checks)
        _require("worktree_hygiene" in readiness, "train readiness gate records worktree hygiene artifact", checks)
        _require("run_worktree_hygiene" in readiness, "train readiness gate materializes worktree hygiene report", checks)
        _require("_trainer_specialist_contract_probe" in readiness, "train readiness gate probes trainer specialist contract loader", checks)
        _require("trainer specialist-fusion loader accepts current audit contract" in readiness, "train readiness gate reports trainer specialist loader proof", checks)
        _require("READY_FOR_VEDTAK_SMOKE_TRAIN_AFTER_GIT_CLEAN" in readiness, "train readiness gate distinguishes dirty-git execution blocker", checks)
        _require("foundation guardrails validate readiness command policy" in readiness, "train readiness gate requires guardrail readiness policy proof", checks)
        _require("entry_foundation_worktree_hygiene_v1" in worktree_hygiene, "worktree hygiene audit writes foundation hygiene schema", checks)
        _require("active_foundation_contract" in worktree_hygiene, "worktree hygiene audit classifies active foundation paths", checks)
        _require("legacy_tombstone_cleanup" in worktree_hygiene, "worktree hygiene audit classifies legacy tombstone cleanup paths", checks)
        _require("entry_related_review" in worktree_hygiene, "worktree hygiene audit separates broader entry-related review paths", checks)
        _require("unrelated_review" in worktree_hygiene, "worktree hygiene audit separates unrelated dirty paths", checks)
        _require("foundation_stage_paths_txt" in worktree_hygiene, "worktree hygiene audit writes foundation stage path list", checks)
        _require("review_hold_paths_txt" in worktree_hygiene, "worktree hygiene audit writes review hold path list", checks)
        _require("foundation_stage_status_tsv" in worktree_hygiene, "worktree hygiene audit writes foundation stage status table", checks)
        _require("review_hold_status_tsv" in worktree_hygiene, "worktree hygiene audit writes review hold status table", checks)
        _require("foundation_stage_summary" in worktree_hygiene, "worktree hygiene audit reports foundation stage summary", checks)
        _require("review_hold_summary" in worktree_hygiene, "worktree hygiene audit reports review hold summary", checks)
        _require("git_add_dry_run_txt" in worktree_hygiene, "worktree hygiene audit writes git add dry-run output", checks)
        _require("cached_unchanged" in worktree_hygiene, "worktree hygiene audit proves dry-run does not change index", checks)
        _require("stage_plan_safe" in worktree_hygiene, "worktree hygiene audit reports stage-plan safety", checks)
        _require("git_add_dry_run_hold_overlap_count" in worktree_hygiene, "worktree hygiene audit proves dry-run excludes hold paths", checks)
        _require("foundation_cleanup_review_decision" in worktree_hygiene, "worktree hygiene audit reports foundation cleanup review decision", checks)
        _require("FOUNDATION_CLEANUP_REQUIRED_PATHS" in worktree_hygiene, "worktree hygiene audit has required cleanup path contract", checks)
        _require("FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS" in worktree_hygiene, "worktree hygiene audit has critical gate path contract", checks)
        _require("foundation_cleanup_critical_gate_review" in worktree_hygiene, "worktree hygiene audit reports critical gate path review", checks)
        _require("foundation_cleanup_stage_ready" in worktree_hygiene, "worktree hygiene audit reports foundation cleanup stage-ready status", checks)
        _require("foundation_cleanup_stage_command" in worktree_hygiene, "worktree hygiene audit reports foundation cleanup stage command", checks)
        _require("foundation_cleanup_post_stage_verification" in worktree_hygiene, "worktree hygiene audit reports post-stage verification status", checks)
        _require("PASS_STAGED" in worktree_hygiene, "worktree hygiene audit can pass exact staged foundation cleanup set", checks)
        _require("FAIL_STAGED" in worktree_hygiene, "worktree hygiene audit can fail unsafe staged set", checks)
        _require("--apply requires --vedtak" in stage_cleanup_wrapper, "foundation cleanup staging wrapper requires vedtak before staging", checks)
        _require("foundation_cleanup_stage_ready" in stage_cleanup_wrapper, "foundation cleanup staging wrapper requires stage-ready report", checks)
        _require("PASS_STAGED" in stage_cleanup_wrapper, "foundation cleanup staging wrapper verifies exact staged set after apply", checks)
        _require("Dry-run only; no git index changes made" in stage_cleanup_wrapper, "foundation cleanup staging wrapper defaults to dry-run", checks)
        _require("--require-edge-audit" in readiness, "train readiness gate points next command at edge-required smoke audit", checks)
        _require("candidate_training_allowed" in readiness, "train readiness gate keeps candidate training explicit", checks)
        _require("READY_FOR_CANDIDATE_TRAINING_VEDTAK" in candidate_readiness, "candidate readiness gate emits candidate-train readiness decision", checks)
        _require("require_edge" in candidate_readiness, "candidate readiness gate requires smoke edge audit", checks)
        _require("require_head_contract" in candidate_readiness, "candidate readiness gate requires smoke head-contract audit", checks)
        _require(
            "worktree_critical_gate_review_ok" in candidate_readiness,
            "candidate readiness gate requires smoke worktree critical gate proof",
            checks,
        )
        _require("specialist-fusion gate contract" in candidate_readiness, "candidate readiness gate requires specialist-fusion gate contract", checks)
        _require("each required specialist has non-collapsed gate weight" in candidate_readiness, "candidate readiness requires per-required-specialist gate liveness", checks)
        _require("--vedtak" in candidate_wrapper, "candidate train wrapper requires vedtak", checks)
        _require("candidate-readiness" in candidate_wrapper, "candidate train wrapper enforces candidate readiness", checks)
        _require("--enable-specialist-fusion" in candidate_wrapper, "candidate train wrapper enables specialist fusion", checks)
        _require(
            "worktree_critical_gate_review_ok" in candidate_wrapper,
            "candidate train wrapper preserves smoke worktree critical gate proof",
            checks,
        )
        _require("AUDIT_CMD" in candidate_wrapper, "candidate train wrapper builds post-train audit command", checks)
        _require("--require-head-contract" in candidate_wrapper, "candidate train wrapper requires post-train head contract audit", checks)
        _require("entry_candidate_bundle_audit_20260628_v1" in candidate_wrapper, "candidate train wrapper writes candidate bundle audit report", checks)
        _require("--skip-candidate-audit" in candidate_wrapper, "candidate train wrapper exposes explicit audit skip", checks)
        _require("2026 offline replay" in candidate_wrapper, "candidate train wrapper leaves replay gate explicit", checks)
        _require("READY_FOR_IQL_DISTILLATION_VEDTAK" in replay_readiness, "replay readiness gate emits IQL distillation readiness decision", checks)
        _require("candidate_bundle_audit_json" in replay_readiness, "replay readiness gate accepts candidate bundle audit artifact", checks)
        _require("candidate bundle audit PASS" in replay_readiness, "replay readiness gate requires candidate bundle audit PASS", checks)
        _require("selective-edge" in replay_readiness, "replay readiness gate requires selective-edge evidence", checks)
        _require("candidate bundle required specialist gate weights are non-collapsed" in replay_readiness, "replay readiness requires per-required-specialist gate liveness", checks)
        _require("matches candidate bundle audit bundle" in replay_readiness, "replay readiness ties selective-edge evidence to candidate bundle audit", checks)
        _require("offline replay" in replay_readiness, "replay readiness gate requires offline replay evidence", checks)
        _require("selective_edge_metrics.csv" in selective_edge, "selective-edge evaluator writes replay-readiness metrics CSV", checks)
        _require("candidate_no_xgb" in selective_edge, "selective-edge evaluator supports no-XGB ablation evidence", checks)
        _require("load_entry_v10_ctx_bundle" in selective_edge, "selective-edge evaluator strict-loads runtime bundle", checks)
        _require("promotion_shadow_live_allowed" in selective_edge, "selective-edge evaluator keeps promotion/shadow/live closed", checks)
        _require("replay_policy_metrics.csv" in replay_evidence, "replay evidence materializer writes replay metrics CSV", checks)
        _require("replay_policy_monthly.csv" in replay_evidence, "replay evidence materializer writes replay monthly CSV", checks)
        _require("replay_policy_trades.csv" in replay_evidence, "replay evidence materializer writes replay trades CSV", checks)
        _require("--trades-path" in replay_evidence, "replay evidence materializer requires explicit trade log path", checks)
        _require("replay_identity_contract" in replay_evidence, "replay evidence materializer writes candidate/selective-edge identity contract", checks)
        _require("candidate_bundle_audit_json" in replay_evidence, "replay evidence materializer records candidate bundle audit artifact", checks)
        _require("selective_edge_summary_json" in replay_evidence, "replay evidence materializer records selective-edge summary artifact", checks)
        _require("IQL_TRANSITION_REQUIRED_COLUMNS" in replay_evidence, "replay evidence materializer defines IQL transition columns", checks)
        _require("require_iql_transition_fields" in replay_evidence, "replay evidence materializer requires IQL transition fields", checks)
        _require("require_year" in replay_evidence, "replay evidence materializer requires 2026 replay evidence by default", checks)
        _require("promotion_shadow_live_allowed" in replay_evidence, "replay evidence materializer keeps promotion/shadow/live closed", checks)
        _require("replay_policy_trades.csv" in replay_readiness, "replay readiness checks replay trades CSV", checks)
        _require("offline replay identity contract ready" in replay_readiness, "replay readiness requires replay identity contract", checks)
        _require("offline replay identity matches candidate bundle audit" in replay_readiness, "replay readiness ties replay evidence to candidate bundle audit", checks)
        _require(
            "smoke_edge_worktree_critical_gate_review_ok" in replay_readiness,
            "replay readiness gate requires candidate smoke worktree critical gate proof",
            checks,
        )
        _require("IQL_TRANSITION_REQUIRED_COLUMNS" in replay_readiness, "replay readiness requires IQL transition columns", checks)
        _require("--vedtak" in iql_distill_wrapper, "IQL distillation wrapper requires vedtak", checks)
        _require("entry_next_edge_control.sh replay-readiness" in iql_distill_wrapper, "IQL distillation wrapper reruns replay-readiness for latest report", checks)
        _require("READY_FOR_IQL_DISTILLATION_VEDTAK" in iql_distill_wrapper, "IQL distillation wrapper enforces replay-readiness", checks)
        _require("materialize_entry_iql_distillation_contract_v1" in iql_distill_wrapper, "IQL distillation wrapper writes foundation contract", checks)
        _require("No IQL distillation, adapter, promotion, shadow, or live path was started" in iql_distill_wrapper, "IQL distillation wrapper is fail-closed before replay evidence", checks)
        _require("ENTRY_IQL_DISTILLATION_CONTRACT_READY" in iql_distill_contract, "IQL distillation contract emits ready decision", checks)
        _require("evidence_identity" in iql_distill_contract, "IQL distillation contract preserves replay evidence identity", checks)
        _require("replay-readiness carries evidence identity" in iql_distill_contract, "IQL distillation contract requires replay evidence identity", checks)
        _require(
            "smoke_edge_worktree_critical_gate_review_ok" in iql_distill_contract,
            "IQL distillation contract preserves smoke worktree critical gate proof",
            checks,
        )
        _require("trainer_started" in iql_distill_contract, "IQL distillation contract records trainer-not-started invariant", checks)
        _require("promotion_shadow_live_allowed" in iql_distill_contract, "IQL distillation contract keeps promotion/shadow/live closed", checks)
        _require("--vedtak" in iql_student_trade_log, "IQL student trade-log materializer requires vedtak", checks)
        _require("ENTRY_IQL_STUDENT_TRADE_LOG_latest.json" in iql_student_trade_log, "IQL student trade-log materializer writes latest report", checks)
        _require("entry_iql_student_trade_log.csv" in iql_student_trade_log, "IQL student trade-log materializer writes explicit trade log", checks)
        _require("student_policy_fit_started" in iql_student_trade_log, "IQL student trade-log materializer records offline student fit", checks)
        _require("promotion_shadow_live_allowed" in iql_student_trade_log, "IQL student trade-log materializer keeps promotion/shadow/live closed", checks)
        _require("ENTRY_IQL_REPLAY_EVIDENCE_latest.json" in iql_replay_evidence, "IQL replay evidence materializer writes latest report", checks)
        _require("REPLAY_EVIDENCE_MANIFEST.json" in iql_replay_evidence, "IQL replay evidence materializer writes comparison manifest", checks)
        _require("DEFAULT_DISTILL_CONTRACT_JSON" in iql_replay_evidence, "IQL replay evidence materializer requires distillation contract", checks)
        _require("ENTRY_IQL_DISTILLATION_CONTRACT_READY" in iql_replay_evidence, "IQL replay evidence materializer requires ready distillation contract", checks)
        _require("evidence_identity" in iql_replay_evidence, "IQL replay evidence materializer preserves evidence identity", checks)
        _require("candidate replay evidence manifest is missing" in iql_replay_evidence, "IQL replay evidence materializer verifies candidate replay manifest exists", checks)
        _require("candidate replay evidence manifest decision is not PASS" in iql_replay_evidence, "IQL replay evidence materializer requires candidate replay manifest PASS", checks)
        _require("IQL replay trade log policy_id must match --policy-id exactly" in iql_replay_evidence, "IQL replay evidence materializer requires exact IQL policy id", checks)
        _require("--require-policy-id" in iql_replay_evidence, "IQL replay evidence materializer exposes policy-id strictness flag", checks)
        _require("trainer_started" in iql_replay_evidence, "IQL replay evidence materializer records trainer-not-started invariant", checks)
        _require("replay_started" in iql_replay_evidence, "IQL replay evidence materializer records replay-not-started invariant", checks)
        _require("promotion_shadow_live_allowed" in iql_replay_evidence, "IQL replay evidence materializer keeps promotion/shadow/live closed", checks)
        _require("READY_FOR_PROMOTION_REVIEW_VEDTAK" in iql_comparison, "IQL comparison gate emits promotion-review readiness decision", checks)
        _require("evidence_identity" in iql_comparison, "IQL comparison gate carries distillation evidence identity", checks)
        _require("candidate replay manifest evidence identity matches distillation contract" in iql_comparison, "IQL comparison gate ties candidate replay manifest to distillation contract", checks)
        _require("candidate replay manifest path matches distillation evidence identity" in iql_comparison, "IQL comparison gate ties candidate replay manifest path to distillation contract", checks)
        _require("IQL replay manifest evidence identity matches distillation contract" in iql_comparison, "IQL comparison gate ties IQL replay manifest to distillation contract", checks)
        _require("IQL replay manifest distillation contract matches comparison input" in iql_comparison, "IQL comparison gate ties IQL replay manifest to exact distillation contract", checks)
        _require("IQL replay manifest references candidate replay evidence from distillation contract" in iql_comparison, "IQL comparison gate ties IQL replay manifest to candidate replay evidence", checks)
        _require("IQL replay net sum beats candidate" in iql_comparison, "IQL comparison gate requires replay net lift", checks)
        _require("IQL replay drawdown does not worsen" in iql_comparison, "IQL comparison gate checks drawdown vs candidate", checks)
        _require("gate never promotes, shadows, or starts live" in iql_comparison, "IQL comparison gate keeps promotion/shadow/live closed", checks)
        _require("entry_iql_replay_slice_audit_v1" in iql_slice_audit, "IQL slice audit writes slice audit schema", checks)
        _require("session/regime/side/direction/bad-path/tail" in iql_slice_audit, "IQL slice audit covers required slice families", checks)
        _require("IQL supported edge slices keep positive net/PF/drawdown/max-loss" in iql_slice_audit, "IQL slice audit requires supported edge robustness", checks)
        _require("IQL diagnostic slices do not materially worsen tails vs candidate" in iql_slice_audit, "IQL slice audit compares diagnostic tail slices", checks)
        _require("slice audit never trains, replays, builds adapters, promotes, shadows, or starts live" in iql_slice_audit, "IQL slice audit keeps train/replay/shadow/live closed", checks)
        _require("promotion_shadow_live_allowed" in iql_slice_audit, "IQL slice audit keeps promotion/shadow/live closed", checks)
        _require("entry_exit_handoff_readiness_v1" in entry_exit_handoff, "Entry-to-Exit handoff audit writes handoff schema", checks)
        _require("BLOCKED_BY_MISSING_EXIT_PER_BAR_SUBSTRATE" in entry_exit_handoff, "Entry-to-Exit handoff audit blocks missing exit substrate", checks)
        _require("REQUIRED_EXIT_SUBSTRATE_FIELDS" in entry_exit_handoff, "Entry-to-Exit handoff audit declares required per-bar substrate fields", checks)
        _require("exit_training_allowed" in entry_exit_handoff, "Entry-to-Exit handoff audit keeps exit training closed", checks)
        _require("exit_iql_allowed" in entry_exit_handoff, "Entry-to-Exit handoff audit keeps exit IQL closed", checks)
        _require("handoff audit never trains, replays, builds adapters, promotes, shadows, or starts live" in entry_exit_handoff, "Entry-to-Exit handoff audit keeps all side-effect paths closed", checks)
        _require("entry_exit_per_bar_handoff_v1" in entry_exit_materializer, "Entry-bound Exit per-bar materializer writes substrate schema", checks)
        _require("REQUIRED_EXIT_SUBSTRATE_FIELDS" in entry_exit_materializer, "Entry-bound Exit per-bar materializer uses handoff substrate contract", checks)
        _require("atr_bps_fill_method" in entry_exit_materializer, "Entry-bound Exit per-bar materializer records deterministic ATR fill provenance", checks)
        _require("gap_exclusion_policy" in entry_exit_materializer and "never synthesize bars" in entry_exit_materializer, "Entry-bound Exit per-bar materializer documents gap exclusions without synthetic bars", checks)
        _require("materializer never trains, replays, builds adapters, promotes, shadows, or starts live" in entry_exit_materializer, "Entry-bound Exit per-bar materializer keeps all side-effect paths closed", checks)
        _require("entry_exit_per_bar_reconstruction_audit_v1" in entry_exit_reconstruction, "Entry Exit per-bar reconstruction audit writes schema", checks)
        _require("READY_FOR_EXIT_STATE_REWARD_CONTRACT_REVIEW" in entry_exit_reconstruction, "Entry Exit per-bar reconstruction audit opens only state/reward contract review", checks)
        _require("BLOCKED_BY_EXIT_RECONSTRUCTION_AUDIT" in entry_exit_reconstruction, "Entry Exit per-bar reconstruction audit blocks failed reconstruction", checks)
        _require("atr_bps is positive and live" in entry_exit_reconstruction, "Entry Exit per-bar reconstruction audit requires live ATR", checks)
        _require("per-trade timeline reconstruction is contiguous and terminal" in entry_exit_reconstruction, "Entry Exit per-bar reconstruction audit checks per-trade terminal timelines", checks)
        _require("reconstruction audit never trains, replays, distills, promotes, shadows, or starts live" in entry_exit_reconstruction, "Entry Exit per-bar reconstruction audit keeps all side-effect paths closed", checks)
        _require("entry_exit_state_reward_contract_v1" in entry_exit_state_reward, "Entry Exit state/reward contract writes schema", checks)
        _require("ENTRY_EXIT_STATE_REWARD_CONTRACT_READY" in entry_exit_state_reward, "Entry Exit state/reward contract has ready decision", checks)
        _require("FORBIDDEN_STATE_FIELDS" in entry_exit_state_reward, "Entry Exit state/reward contract blocks shortcut fields", checks)
        _require("HOLD next-row pointers are intra-episode and terminal rows stop" in entry_exit_state_reward, "Entry Exit state/reward contract checks HOLD transition pointers", checks)
        _require("state/reward contract never trains, replays, distills, promotes, shadows, or starts live" in entry_exit_state_reward, "Entry Exit state/reward contract keeps all side-effect paths closed", checks)
        _require("entry_exit_split_leakage_audit_v1" in entry_exit_split_leakage, "Entry Exit split/leakage audit writes schema", checks)
        _require("ENTRY_EXIT_SPLIT_LEAKAGE_AUDIT_READY" in entry_exit_split_leakage, "Entry Exit split/leakage audit has ready decision", checks)
        _require("HOLD next-row pointers stay inside the same split" in entry_exit_split_leakage, "Entry Exit split/leakage audit checks HOLD next-row split leakage", checks)
        _require("state features exclude reward/outcome shortcut fields" in entry_exit_split_leakage, "Entry Exit split/leakage audit blocks shortcut state features", checks)
        _require("split/leakage audit never trains, replays, distills, promotes, shadows, or starts live" in entry_exit_split_leakage, "Entry Exit split/leakage audit keeps all side-effect paths closed", checks)
        _require("entry_exit_model_dataset_readiness_v1" in entry_exit_model_dataset, "Entry Exit model dataset readiness writes schema", checks)
        _require("ENTRY_EXIT_MODEL_DATASET_READY_FOR_EXIT_TRANSFORMER_READINESS_REVIEW" in entry_exit_model_dataset, "Entry Exit model dataset readiness has ready decision", checks)
        _require("fit_numeric_mean_std_and_categorical_vocab_on_train_split_only" in entry_exit_model_dataset, "Entry Exit model dataset readiness uses train-only normalization", checks)
        _require("numeric state features are finite and live" in entry_exit_model_dataset, "Entry Exit model dataset readiness requires live numeric state", checks)
        _require("model dataset readiness never trains, replays, distills, promotes, shadows, or starts live" in entry_exit_model_dataset, "Entry Exit model dataset readiness keeps all side-effect paths closed", checks)
        _require("entry_exit_feature_alignment_v1" in entry_exit_feature_alignment, "Entry Exit feature alignment writes schema", checks)
        _require("ENTRY_EXIT_FEATURE_ALIGNMENT_READY_FOR_EXIT_TRANSFORMER_TRAINING_REVIEW" in entry_exit_feature_alignment, "Entry Exit feature alignment has ready decision", checks)
        _require("BLOCKED_BY_ENTRY_EXIT_FEATURE_ALIGNMENT" in entry_exit_feature_alignment, "Entry Exit feature alignment has blocked decision", checks)
        _require("structure_swing" in entry_exit_feature_alignment and "smc_liquidity" in entry_exit_feature_alignment and "momentum_flow" in entry_exit_feature_alignment, "Entry Exit feature alignment audits required market families", checks)
        _require("feature alignment audit never trains, replays, distills, promotes, shadows, or starts live" in entry_exit_feature_alignment, "Entry Exit feature alignment keeps all side-effect paths closed", checks)
        _require("entry_exit_transformer_architecture_readiness_v1" in entry_exit_transformer_architecture, "Entry Exit Transformer architecture readiness writes schema", checks)
        _require("ENTRY_EXIT_TRANSFORMER_ARCHITECTURE_READY_FOR_TRAINING_PLAN_REVIEW" in entry_exit_transformer_architecture, "Entry Exit Transformer architecture readiness has ready decision", checks)
        _require("exit_sequence_transformer_v1" in entry_exit_transformer_architecture, "Entry Exit Transformer architecture readiness locks model family", checks)
        _require("causal_masked_transformer_encoder" in entry_exit_transformer_architecture, "Entry Exit Transformer architecture readiness requires causal encoder", checks)
        _require("Exit Transformer architecture readiness never trains, replays, distills, promotes, shadows, or starts live" in entry_exit_transformer_architecture, "Entry Exit Transformer architecture readiness keeps all side-effect paths closed", checks)
        _require("entry_exit_transformer_training_plan_readiness_v1" in entry_exit_transformer_training_plan, "Entry Exit Transformer training plan readiness writes schema", checks)
        _require("ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READY_FOR_VEDTAK_REVIEW" in entry_exit_transformer_training_plan, "Entry Exit Transformer training plan readiness has ready decision", checks)
        _require("ENTRY_EXIT_TRANSFORMER_TRAIN_" in entry_exit_transformer_training_plan, "Entry Exit Transformer training plan readiness requires train vedtak prefix", checks)
        _require("requires_ram_guard" in entry_exit_transformer_training_plan, "Entry Exit Transformer training plan readiness requires RAM guard", checks)
        _require("training plan readiness never trains, replays, distills, promotes, shadows, or starts live" in entry_exit_transformer_training_plan, "Entry Exit Transformer training plan readiness keeps all side-effect paths closed", checks)
        _require("entry_exit_transformer_trainer_wrapper_readiness_v1" in entry_exit_transformer_trainer_wrapper, "Entry Exit Transformer trainer wrapper readiness writes schema", checks)
        _require("ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READY_FOR_IMPLEMENTATION_REVIEW" in entry_exit_transformer_trainer_wrapper, "Entry Exit Transformer trainer wrapper readiness has ready decision", checks)
        _require("active Exit Transformer train wrapper rejects missing vedtak" in entry_exit_transformer_trainer_wrapper, "Entry Exit Transformer trainer wrapper readiness exercises missing-vedtak rejection", checks)
        _require("trainer wrapper readiness never trains, replays, distills, promotes, shadows, or starts live" in entry_exit_transformer_trainer_wrapper, "Entry Exit Transformer trainer wrapper readiness keeps all side-effect paths closed", checks)
        _require("ENTRY_EXIT_TRANSFORMER_TRAIN_" in entry_exit_transformer_train_wrapper, "Entry Exit Transformer train wrapper requires train vedtak prefix", checks)
        _require("TRAINER_IMPLEMENTATION_ENABLED=0" in entry_exit_transformer_train_wrapper, "Entry Exit Transformer train wrapper keeps implementation disabled", checks)
        _require("active Exit Transformer trainer implementation is not enabled" in entry_exit_transformer_train_wrapper, "Entry Exit Transformer train wrapper fail-closed before real training", checks)
        _require("TRAIN_EXECUTION_REVIEW_JSON" in entry_exit_transformer_train_wrapper, "Entry Exit Transformer train wrapper requires train-execution review json", checks)
        _require("ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_READY_FOR_EXPLICIT_VEDTAK_PACKAGE" in entry_exit_transformer_train_wrapper, "Entry Exit Transformer train wrapper requires ready train-execution review", checks)
        _require("POST_TRAIN_AUDIT_CONTRACT_JSON" in entry_exit_transformer_train_wrapper, "Entry Exit Transformer train wrapper requires post-train audit contract json", checks)
        _require("ENTRY_EXIT_TRANSFORMER_POST_TRAIN_AUDIT_CONTRACT_READY" in entry_exit_transformer_train_wrapper, "Entry Exit Transformer train wrapper requires ready post-train audit contract", checks)
        _require("FEATURE_ALIGNMENT_JSON" in entry_exit_transformer_train_wrapper, "Entry Exit Transformer train wrapper requires Entry-to-Exit feature alignment json", checks)
        _require("ENTRY_EXIT_FEATURE_ALIGNMENT_READY_FOR_EXIT_TRANSFORMER_TRAINING_REVIEW" in entry_exit_transformer_train_wrapper, "Entry Exit Transformer train wrapper requires ready Entry-to-Exit feature alignment", checks)
        _require("scripts/gx1_capped_run.sh" in entry_exit_transformer_train_wrapper, "Entry Exit Transformer train wrapper declares capped run", checks)
        _require("--num-workers" in entry_exit_transformer_train_wrapper and "NUM_WORKERS=0" in entry_exit_transformer_train_wrapper, "Entry Exit Transformer train wrapper declares num-workers zero", checks)
        _require("ExitSequenceTransformerV1" in entry_exit_transformer_trainer_core, "Entry Exit Transformer trainer core defines active model", checks)
        _require("--preflight-only" in entry_exit_transformer_trainer_core, "Entry Exit Transformer trainer core is preflight-gated", checks)
        _require("active Exit Transformer training is not enabled" in entry_exit_transformer_trainer_core, "Entry Exit Transformer trainer core blocks non-preflight training", checks)
        _require("causal_mask" in entry_exit_transformer_trainer_core, "Entry Exit Transformer trainer core builds causal mask", checks)
        _require("optimizer_steps" in entry_exit_transformer_trainer_core and '"optimizer_steps": 0' in entry_exit_transformer_trainer_core, "Entry Exit Transformer trainer core records zero optimizer steps in preflight", checks)
        _require("entry_exit_transformer_pretrain_manifest_v1" in entry_exit_transformer_pretrain_manifest, "Entry Exit Transformer pretrain manifest writes schema", checks)
        _require("ENTRY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST_READY_FOR_TRAIN_EXECUTION_REVIEW" in entry_exit_transformer_pretrain_manifest, "Entry Exit Transformer pretrain manifest has ready decision", checks)
        _require("active Exit Transformer trainer core finite forward preflight passes" in entry_exit_transformer_pretrain_manifest, "Entry Exit Transformer pretrain manifest requires finite forward", checks)
        _require("pretrain manifest never trains, replays, distills, promotes, shadows, or starts live" in entry_exit_transformer_pretrain_manifest, "Entry Exit Transformer pretrain manifest keeps all side-effect paths closed", checks)
        _require("entry_exit_model_dataset_slice_robustness_v1" in entry_exit_model_dataset_slice_robustness, "Entry Exit model dataset slice robustness writes schema", checks)
        _require("ENTRY_EXIT_MODEL_DATASET_SLICE_ROBUSTNESS_READY_WITH_WEAK_SLICE_DISCLOSURE" in entry_exit_model_dataset_slice_robustness, "Entry Exit model dataset slice robustness has weak-slice disclosure ready decision", checks)
        _require("session/regime/side slices are disclosed without unsupported slices" in entry_exit_model_dataset_slice_robustness, "Entry Exit model dataset slice robustness audits session/regime/side slices", checks)
        _require("weak_slice_count" in entry_exit_model_dataset_slice_robustness, "Entry Exit model dataset slice robustness records weak slices", checks)
        _require("slice robustness audit never trains, replays, distills, promotes, shadows, or starts live" in entry_exit_model_dataset_slice_robustness, "Entry Exit model dataset slice robustness keeps all side-effect paths closed", checks)
        _require("entry_exit_transformer_train_execution_review_v1" in entry_exit_transformer_train_execution_review, "Entry Exit Transformer train execution review writes schema", checks)
        _require("ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_READY_FOR_EXPLICIT_VEDTAK_PACKAGE" in entry_exit_transformer_train_execution_review, "Entry Exit Transformer train execution review has ready decision", checks)
        _require("must_not_promote_from_broad_average" in entry_exit_transformer_train_execution_review, "Entry Exit Transformer train execution review accounts for weak slices", checks)
        _require("ENTRY_EXIT_TRANSFORMER_TRAIN_" in entry_exit_transformer_train_execution_review, "Entry Exit Transformer train execution review preserves train vedtak prefix", checks)
        _require("train execution review never trains, replays, distills, promotes, shadows, or starts live" in entry_exit_transformer_train_execution_review, "Entry Exit Transformer train execution review keeps all side-effect paths closed", checks)
        _require("entry_exit_transformer_post_train_contract_v1" in entry_exit_transformer_post_train_contract, "Entry Exit Transformer post-train audit contract writes schema", checks)
        _require("ENTRY_EXIT_TRANSFORMER_POST_TRAIN_AUDIT_CONTRACT_READY" in entry_exit_transformer_post_train_contract, "Entry Exit Transformer post-train audit contract has ready decision", checks)
        _require("exact_output_heads" in entry_exit_transformer_post_train_contract, "Entry Exit Transformer post-train audit contract locks exact output heads", checks)
        _require("must_not_promote_from_broad_average" in entry_exit_transformer_post_train_contract, "Entry Exit Transformer post-train audit contract blocks broad averages", checks)
        _require("post-train audit contract never trains, replays, distills, promotes, shadows, or starts live" in entry_exit_transformer_post_train_contract, "Entry Exit Transformer post-train audit contract keeps all side-effect paths closed", checks)

    report = {
        "schema_version": "entry_foundation_state_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FOUNDATION_FREEZE_ACTIVE",
        "training_allowed": False,
        "smoke_training_gate_ready": True,
        "shadow_allowed": False,
        "legacy_no_xgb_candidate_active": False,
        "next_required_gate": NEXT_REQUIRED_GATE,
        "audit_doc": str(Path(args.audit_doc)),
        "blueprint_doc": str(BLUEPRINT_DOC),
        "legacy_next_edge_plan": str(LEGACY_NEXT_EDGE_PLAN),
        "legacy_reports_root": str(LEGACY_REPORTS),
        "legacy_no_xgb_package": str(LEGACY_NO_XGB_PACKAGE),
        "active_no_xgb_package": str(ACTIVE_NO_XGB_PACKAGE),
        "sequence_structure_manifest": str(SEQ_STRUCTURE_MANIFEST),
        "foundation_dataset_dir": str(FOUNDATION_DATASET_DIR),
        "foundation_sanity_bundle": sanity_bundle,
        "foundation_specialist_sanity_bundle": specialist_sanity_bundle,
        "foundation_smoke_dataset": smoke_dataset,
        "feature_audit_latest": feature_audit,
        "target_audit_latest": target_audit,
        "specialist_audit_latest": specialist_audit,
        "split_manifest_summary": split_manifest_summary,
        "foundation_structure_feature_count": int(len(FOUNDATION_STRUCTURE_FEATURE_NAMES)),
        "checks_passed": len(checks),
        "checks": checks,
    }

    out_path: Path | None = None
    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["out"] = str(out_path)

    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-doc", default=str(AUDIT_DOC))
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
