#!/usr/bin/env python3
"""Audit git worktree hygiene for the Entry foundation smoke-training gate.

The real smoke trainer requires a clean git worktree. This report makes the
dirty state actionable by classifying changed paths instead of exposing only raw
`git status --short` output.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.scripts.verify_entry_foundation_state_v1 import REPO, REPORTS_ROOT


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_foundation_worktree_hygiene_20260628_v1"

ACTIVE_FOUNDATION_PATTERNS = (
    "AGENTS.md",
    "CLAUDE.md",
    "SYSTEM_MAP.md",
    "docs/ENTRY_FOUNDATION_AUDIT_20260628.md",
    "docs/ENTRY_SEQUENTIAL_AI_SPECIALIST_BLUEPRINT_20260628.md",
    "gx1/execution/v12_daily_counterfactual.sh",
    "gx1/execution/v12_paper_runner.py",
    "gx1/execution/v12_prebuilt_refresh_daemon.sh",
    "gx1/features/entry_foundation_structure_v1.py",
    "gx1/features/entry_specialist_feature_groups_v1.py",
    "gx1/models/entry_v10/entry_v10_bundle.py",
    "gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py",
    "gx1/models/entry_v10/entry_v10_ctx_train_v3.py",
    "gx1/runtime/entry_next_edge_legacy_guard.py",
    "gx1/scripts/audit_entry_foundation_*.py",
    "gx1/scripts/audit_entry_exit_*.py",
    "gx1/scripts/audit_entry_iql_*.py",
    "gx1/scripts/audit_entry_specialist_feature_groups_v1.py",
    "gx1/scripts/evaluate_entry_candidate_selective_edge_v1.py",
    "gx1/scripts/apply_entry_foundation_activation_v1.py",
    "gx1/scripts/materialize_entry_candidate_replay_evidence_v1.py",
    "gx1/scripts/materialize_entry_foundation_*.py",
    "gx1/scripts/materialize_entry_iql_*.py",
    "gx1/scripts/plan_entry_foundation_activation_v1.py",
    "gx1/scripts/run_entry_foundation_activation_post_apply_v1.py",
    "gx1/scripts/verify_entry_candidate_readiness_v1.py",
    "gx1/scripts/verify_entry_foundation_*.py",
    "gx1/scripts/verify_entry_iql_*.py",
    "gx1/scripts/verify_entry_replay_readiness_v1.py",
    "gx1/scripts/verify_entry_training_readiness_v1.py",
    "scripts/entry_next_edge_control.sh",
    "scripts/entry_next_edge_legacy_block.sh",
    "scripts/entry_next_edge_live_legacy_block.sh",
    "scripts/gx1_handover.sh",
    "scripts/gx1_nightly_learning.sh",
    "scripts/launch_live_practice.sh",
    "scripts/run_entry_foundation_*.sh",
    "scripts/stage_entry_foundation_cleanup.sh",
    "scripts/run_live_trial160.sh",
    "scripts/stop_live_practice.sh",
    "tests/test_entry_exit_*.py",
    "tests/test_entry_candidate_*.py",
    "tests/test_entry_foundation_*.py",
    "tests/test_entry_handover_control.py",
    "tests/test_entry_iql_*.py",
    "tests/test_entry_replay_readiness.py",
    "tests/test_entry_specialist_feature_groups.py",
    "tests/test_entry_training_readiness.py",
    "tests/test_entry_v10_specialist_fusion_model.py",
)

LEGACY_TOMBSTONE_PATTERNS = (
    "docs/ENTRY_NEXT_EDGE_PLAN_20260627.md",
    "docs/ENTRY_NEXT_EDGE_SHADOW_REVIEW_TEMPLATE_20260627.md",
    "gx1/runtime/entry_tabular_no_xgb_candidate.py",
    "gx1/scripts/*entry_tabular_no_xgb*.py",
    "gx1/scripts/verify_entry_next_edge_*.py",
    "scripts/run_entry_tabular_no_xgb_shadow_only.sh",
    "tests/test_entry_tabular_no_xgb*.py",
)

ENTRY_REVIEW_PATTERNS = (
    "*entry*",
    "*Entry*",
    "*v10*",
    "*V10*",
    "*iql*",
    "*IQL*",
)

CATEGORY_ORDER = (
    "active_foundation_contract",
    "legacy_tombstone_cleanup",
    "entry_related_review",
    "unrelated_review",
)

FOUNDATION_CLEANUP_REQUIRED_PATHS = (
    "AGENTS.md",
    "CLAUDE.md",
    "SYSTEM_MAP.md",
    "docs/ENTRY_FOUNDATION_AUDIT_20260628.md",
    "docs/ENTRY_SEQUENTIAL_AI_SPECIALIST_BLUEPRINT_20260628.md",
    "gx1/features/entry_foundation_structure_v1.py",
    "gx1/features/entry_specialist_feature_groups_v1.py",
    "gx1/models/entry_v10/entry_v10_bundle.py",
    "gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py",
    "gx1/models/entry_v10/entry_v10_ctx_train_v3.py",
    "gx1/runtime/entry_next_edge_legacy_guard.py",
    "gx1/scripts/audit_entry_foundation_features_v1.py",
    "gx1/scripts/audit_entry_foundation_smoke_bundle_v1.py",
    "gx1/scripts/audit_entry_foundation_targets_v1.py",
    "gx1/scripts/audit_entry_foundation_worktree_hygiene_v1.py",
    "gx1/scripts/audit_entry_exit_handoff_readiness_v1.py",
    "gx1/scripts/audit_entry_iql_replay_slices_v1.py",
    "gx1/scripts/audit_entry_specialist_feature_groups_v1.py",
    "gx1/scripts/apply_entry_foundation_activation_v1.py",
    "gx1/scripts/evaluate_entry_candidate_selective_edge_v1.py",
    "gx1/scripts/materialize_entry_candidate_replay_evidence_v1.py",
    "gx1/scripts/materialize_entry_foundation_smoke_dataset_v1.py",
    "gx1/scripts/materialize_entry_iql_distillation_contract_v1.py",
    "gx1/scripts/materialize_entry_iql_student_trade_log_v1.py",
    "gx1/scripts/materialize_entry_iql_replay_evidence_v1.py",
    "gx1/scripts/plan_entry_foundation_activation_v1.py",
    "gx1/scripts/run_entry_foundation_activation_post_apply_v1.py",
    "gx1/scripts/verify_entry_candidate_readiness_v1.py",
    "gx1/scripts/verify_entry_foundation_guardrails_v1.py",
    "gx1/scripts/verify_entry_foundation_state_v1.py",
    "gx1/scripts/verify_entry_iql_replay_comparison_v1.py",
    "gx1/scripts/verify_entry_replay_readiness_v1.py",
    "gx1/scripts/verify_entry_training_readiness_v1.py",
    "scripts/entry_next_edge_control.sh",
    "scripts/entry_next_edge_legacy_block.sh",
    "scripts/entry_next_edge_live_legacy_block.sh",
    "scripts/gx1_handover.sh",
    "scripts/run_entry_foundation_iql_distill.sh",
    "scripts/run_entry_foundation_seq146_candidate_train.sh",
    "scripts/run_entry_foundation_seq146_smoke_train.sh",
    "scripts/stage_entry_foundation_cleanup.sh",
    "scripts/run_entry_tabular_no_xgb_shadow_only.sh",
    "tests/test_entry_candidate_readiness.py",
    "tests/test_entry_candidate_replay_evidence.py",
    "tests/test_entry_candidate_selective_edge.py",
    "tests/test_entry_candidate_train_wrapper.py",
    "tests/test_entry_foundation_manifest_and_audit.py",
    "tests/test_entry_foundation_guardrails.py",
    "tests/test_entry_foundation_smoke_bundle_audit.py",
    "tests/test_entry_foundation_smoke_dataset.py",
    "tests/test_entry_foundation_smoke_train_wrapper.py",
    "tests/test_entry_foundation_state.py",
    "tests/test_entry_foundation_structure_features.py",
    "tests/test_entry_foundation_target_audit.py",
    "tests/test_entry_foundation_worktree_hygiene.py",
    "tests/test_entry_handover_control.py",
    "tests/test_entry_iql_distill_wrapper.py",
    "tests/test_entry_iql_distillation_contract.py",
    "tests/test_entry_iql_student_trade_log.py",
    "tests/test_entry_iql_replay_comparison.py",
    "tests/test_entry_iql_replay_evidence.py",
    "tests/test_entry_iql_replay_slice_audit.py",
    "tests/test_entry_exit_handoff_readiness.py",
    "tests/test_entry_replay_readiness.py",
    "tests/test_entry_specialist_feature_groups.py",
    "tests/test_entry_training_readiness.py",
    "tests/test_entry_v10_specialist_fusion_model.py",
)

FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS = (
    "docs/ENTRY_FOUNDATION_AUDIT_20260628.md",
    "docs/ENTRY_SEQUENTIAL_AI_SPECIALIST_BLUEPRINT_20260628.md",
    "gx1/scripts/audit_entry_foundation_smoke_bundle_v1.py",
    "gx1/scripts/audit_entry_foundation_worktree_hygiene_v1.py",
    "gx1/scripts/audit_entry_exit_handoff_readiness_v1.py",
    "gx1/scripts/audit_entry_iql_replay_slices_v1.py",
    "gx1/scripts/apply_entry_foundation_activation_v1.py",
    "gx1/scripts/materialize_entry_iql_distillation_contract_v1.py",
    "gx1/scripts/materialize_entry_iql_student_trade_log_v1.py",
    "gx1/scripts/materialize_entry_iql_replay_evidence_v1.py",
    "gx1/scripts/plan_entry_foundation_activation_v1.py",
    "gx1/scripts/run_entry_foundation_activation_post_apply_v1.py",
    "gx1/scripts/verify_entry_candidate_readiness_v1.py",
    "gx1/scripts/verify_entry_foundation_guardrails_v1.py",
    "gx1/scripts/verify_entry_iql_replay_comparison_v1.py",
    "gx1/scripts/verify_entry_replay_readiness_v1.py",
    "gx1/scripts/verify_entry_training_readiness_v1.py",
    "scripts/entry_next_edge_control.sh",
    "scripts/gx1_handover.sh",
    "scripts/stage_entry_foundation_cleanup.sh",
    "tests/test_entry_candidate_readiness.py",
    "tests/test_entry_foundation_smoke_bundle_audit.py",
    "tests/test_entry_foundation_guardrails.py",
    "tests/test_entry_foundation_worktree_hygiene.py",
    "tests/test_entry_handover_control.py",
    "tests/test_entry_iql_distillation_contract.py",
    "tests/test_entry_iql_student_trade_log.py",
    "tests/test_entry_iql_replay_comparison.py",
    "tests/test_entry_iql_replay_evidence.py",
    "tests/test_entry_iql_replay_slice_audit.py",
    "tests/test_entry_exit_handoff_readiness.py",
    "tests/test_entry_replay_readiness.py",
    "tests/test_entry_training_readiness.py",
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def classify_path(path: str) -> str:
    if _matches(path, ACTIVE_FOUNDATION_PATTERNS):
        return "active_foundation_contract"
    if _matches(path, LEGACY_TOMBSTONE_PATTERNS):
        return "legacy_tombstone_cleanup"
    if _matches(path, ENTRY_REVIEW_PATTERNS):
        return "entry_related_review"
    return "unrelated_review"


def parse_porcelain_line(line: str) -> dict[str, Any]:
    status = line[:2]
    raw_path = line[3:] if len(line) > 3 else ""
    path = raw_path.split(" -> ", 1)[-1]
    return {
        "status": status,
        "path": path,
        "raw_path": raw_path,
        "category": classify_path(path),
    }


def _git_status_entries() -> list[dict[str, Any]]:
    proc = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git status failed: {proc.stderr.strip()}")
    return [parse_porcelain_line(line) for line in proc.stdout.splitlines() if line]


def _git_cached_paths() -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--cached", "--name-only"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git diff --cached failed: {proc.stderr.strip()}")
    return [line for line in proc.stdout.splitlines() if line]


def _git_add_dry_run(pathspec_file: Path) -> dict[str, Any]:
    before = _git_cached_paths()
    proc = subprocess.run(
        ["git", "-C", str(REPO), "add", "--dry-run", f"--pathspec-from-file={pathspec_file}"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    after = _git_cached_paths()
    return {
        "cmd": ["git", "add", "--dry-run", f"--pathspec-from-file={pathspec_file}"],
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "cached_before_count": len(before),
        "cached_after_count": len(after),
        "cached_unchanged": before == after,
    }


def _parse_git_add_dry_run_paths(stdout: str) -> list[str]:
    paths: list[str] = []
    for line in stdout.splitlines():
        try:
            parts = shlex.split(line)
        except ValueError:
            continue
        if len(parts) >= 2 and parts[0] == "add":
            paths.append(parts[1])
    return paths


def _file_size_and_lines(path: str) -> tuple[int | None, int | None]:
    full = REPO / path
    if not full.exists() or not full.is_file():
        return None, None
    size = full.stat().st_size
    try:
        line_count = len(full.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        line_count = None
    return int(size), line_count


def _status_rows(entries: list[dict[str, Any]], paths: list[str]) -> list[dict[str, Any]]:
    by_path = {str(entry["path"]): entry for entry in entries}
    rows: list[dict[str, Any]] = []
    for path in paths:
        entry = by_path.get(path, {"status": "??", "category": classify_path(path), "path": path})
        size, line_count = _file_size_and_lines(path)
        rows.append(
            {
                "status": entry.get("status"),
                "category": entry.get("category"),
                "size_bytes": size,
                "line_count": line_count,
                "path": path,
            }
        )
    return rows


def _status_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_category: dict[str, int] = {}
    total_size = 0
    total_lines = 0
    for row in rows:
        status = str(row.get("status"))
        category = str(row.get("category"))
        by_status[status] = by_status.get(status, 0) + 1
        by_category[category] = by_category.get(category, 0) + 1
        total_size += int(row.get("size_bytes") or 0)
        total_lines += int(row.get("line_count") or 0)
    return {
        "count": len(rows),
        "by_status": by_status,
        "by_category": by_category,
        "total_size_bytes": total_size,
        "total_line_count": total_lines,
    }


def _write_status_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["status\tcategory\tsize_bytes\tline_count\tpath"]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row.get("status") or ""),
                    str(row.get("category") or ""),
                    "" if row.get("size_bytes") is None else str(row.get("size_bytes")),
                    "" if row.get("line_count") is None else str(row.get("line_count")),
                    str(row.get("path") or ""),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _foundation_cleanup_required_review(
    *,
    dirty_paths: set[str],
    stage_set: set[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing_from_repo: list[str] = []
    dirty_missing_from_stage: list[str] = []
    for path in FOUNDATION_CLEANUP_REQUIRED_PATHS:
        exists = (REPO / path).exists()
        dirty = path in dirty_paths
        in_stage = path in stage_set
        ok = exists and (not dirty or in_stage)
        if not exists:
            missing_from_repo.append(path)
        if dirty and not in_stage:
            dirty_missing_from_stage.append(path)
        rows.append(
            {
                "path": path,
                "exists": exists,
                "dirty": dirty,
                "in_stage": in_stage,
                "ok": ok,
            }
        )
    return {
        "required_path_count": len(FOUNDATION_CLEANUP_REQUIRED_PATHS),
        "ok_count": sum(1 for row in rows if row["ok"]),
        "missing_from_repo": missing_from_repo,
        "dirty_missing_from_stage": dirty_missing_from_stage,
        "rows": rows,
    }


def _foundation_cleanup_critical_gate_review(
    *,
    dirty_paths: set[str],
    stage_set: set[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing_from_repo: list[str] = []
    dirty_missing_from_stage: list[str] = []
    for path in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS:
        exists = (REPO / path).exists()
        dirty = path in dirty_paths
        in_stage = path in stage_set
        ok = exists and (not dirty or in_stage)
        if not exists:
            missing_from_repo.append(path)
        if dirty and not in_stage:
            dirty_missing_from_stage.append(path)
        rows.append(
            {
                "path": path,
                "exists": exists,
                "dirty": dirty,
                "in_stage": in_stage,
                "ok": ok,
            }
        )
    return {
        "critical_gate_path_count": len(FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS),
        "ok_count": sum(1 for row in rows if row["ok"]),
        "missing_from_repo": missing_from_repo,
        "dirty_missing_from_stage": dirty_missing_from_stage,
        "rows": rows,
    }


def _post_stage_verification(
    *,
    cached_paths: list[str],
    stage_set: set[str],
    hold_set: set[str],
) -> dict[str, Any]:
    cached_set = set(cached_paths)
    cached_not_in_stage = sorted(cached_set - stage_set)
    cached_hold_overlap = sorted(cached_set & hold_set)
    stage_missing_from_cached = sorted(stage_set - cached_set)
    if not cached_paths:
        decision = "NOT_STAGED"
    elif (
        not cached_not_in_stage
        and not cached_hold_overlap
        and not stage_missing_from_cached
        and cached_set == stage_set
    ):
        decision = "PASS_STAGED"
    else:
        decision = "FAIL_STAGED"
    return {
        "decision": decision,
        "cached_count": len(cached_paths),
        "expected_stage_count": len(stage_set),
        "cached_not_in_stage_count": len(cached_not_in_stage),
        "cached_not_in_stage_first_40": cached_not_in_stage[:40],
        "cached_hold_overlap_count": len(cached_hold_overlap),
        "cached_hold_overlap_first_40": cached_hold_overlap[:40],
        "stage_missing_from_cached_count": len(stage_missing_from_cached),
        "stage_missing_from_cached_first_40": stage_missing_from_cached[:40],
    }


def _clean_git_resolution(report: dict[str, Any]) -> dict[str, Any]:
    post_stage = report.get("foundation_cleanup_post_stage_verification")
    post_stage = post_stage if isinstance(post_stage, dict) else {}
    hold_summary = report.get("review_hold_summary")
    hold_summary = hold_summary if isinstance(hold_summary, dict) else {}
    hold_count = int(report.get("review_before_stage_dirty_count") or 0)
    foundation_stage_ready_for_commit = (
        post_stage.get("decision") == "PASS_STAGED"
        and int(post_stage.get("cached_count") or 0) == int(report.get("foundation_cleanup_dirty_count") or 0)
        and int(post_stage.get("cached_hold_overlap_count") or 0) == 0
    )
    real_smoke_train_allowed = bool(report.get("real_smoke_train_allowed"))
    if real_smoke_train_allowed:
        decision = "PASS_CLEAN_GIT"
    elif foundation_stage_ready_for_commit:
        decision = "READY_FOR_EXTERNAL_CLEAN_GIT_DECISION"
    elif report.get("foundation_cleanup_stage_ready"):
        decision = "STAGE_FOUNDATION_CLEANUP_FIRST"
    else:
        decision = "FIX_FOUNDATION_STAGE_PLAN_FIRST"
    return {
        "decision": decision,
        "real_smoke_train_blocked": not real_smoke_train_allowed,
        "requires_explicit_clean_git_decision": not real_smoke_train_allowed,
        "foundation_stage_ready_for_commit": bool(foundation_stage_ready_for_commit),
        "foundation_stage_paths_txt": report.get("foundation_stage_paths_txt"),
        "foundation_stage_status_tsv": report.get("foundation_stage_status_tsv"),
        "review_hold_paths_txt": report.get("review_hold_paths_txt"),
        "review_hold_status_tsv": report.get("review_hold_status_tsv"),
        "review_hold_count": hold_count,
        "review_hold_by_category": hold_summary.get("by_category") or {},
        "required_before_real_train": [
            "commit or otherwise clear the staged foundation cleanup paths",
            "commit/stash/remove review-hold paths under a separate decision",
            "rerun scripts/entry_next_edge_control.sh worktree-hygiene --no-fail-on-dirty",
            "rerun scripts/entry_next_edge_control.sh train-readiness",
        ],
        "forbidden_without_decision": [
            "do not add review-hold paths to the foundation cleanup stage plan",
            "do not start smoke-train while train-readiness is READY_FOR_VEDTAK_SMOKE_TRAIN_AFTER_GIT_CLEAN",
            "do not start candidate training, replay, IQL distillation, shadow, live or promotion",
        ],
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    clean_git_resolution = report.get("clean_git_resolution")
    clean_git_resolution = clean_git_resolution if isinstance(clean_git_resolution, dict) else {}
    lines = [
        "# Entry Foundation Worktree Hygiene",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Real smoke train allowed: `{report['real_smoke_train_allowed']}`",
        f"- Dirty count: `{report['dirty_count']}`",
        f"- Foundation change set isolated: `{report['foundation_change_set_isolated']}`",
        f"- Foundation stage paths: `{report['foundation_stage_paths_txt']}`",
        f"- Review hold paths: `{report['review_hold_paths_txt']}`",
        f"- Foundation stage status: `{report['foundation_stage_status_tsv']}`",
        f"- Review hold status: `{report['review_hold_status_tsv']}`",
        f"- Git add dry-run output: `{report['git_add_dry_run_txt']}`",
        f"- Git add dry-run rc: `{report['git_add_dry_run']['returncode']}`",
        f"- Git index unchanged by dry-run: `{report['git_add_dry_run']['cached_unchanged']}`",
        f"- Stage plan safe: `{report['stage_plan_safe']}`",
        f"- Foundation cleanup review: `{report['foundation_cleanup_review_decision']}`",
        f"- Critical gate paths ok: `{report['foundation_cleanup_critical_gate_review']['ok_count']}/{report['foundation_cleanup_critical_gate_review']['critical_gate_path_count']}`",
        f"- Foundation cleanup stage ready: `{report['foundation_cleanup_stage_ready']}`",
        f"- Foundation cleanup stage command: `{' '.join(report['foundation_cleanup_stage_command'])}`",
        f"- Post-stage verification: `{report['foundation_cleanup_post_stage_verification']['decision']}`",
        f"- Clean-git resolution: `{clean_git_resolution.get('decision')}`",
        f"- Review hold by category: `{clean_git_resolution.get('review_hold_by_category')}`",
        "",
        "## Categories",
        "",
    ]
    for category in CATEGORY_ORDER:
        rows = report["categories"][category]
        lines.append(f"- `{category}`: {rows['count']}")
    lines.extend(["", "## Next Actions", ""])
    lines.extend([f"- {item}" for item in report["recommended_next_actions"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = _git_status_entries()
    categories: dict[str, dict[str, Any]] = {
        category: {"count": 0, "paths": []}
        for category in CATEGORY_ORDER
    }
    for entry in entries:
        category = str(entry["category"])
        categories[category]["count"] += 1
        categories[category]["paths"].append(entry["path"])
    dirty_count = len(entries)
    foundation_dirty = (
        categories["active_foundation_contract"]["count"]
        + categories["legacy_tombstone_cleanup"]["count"]
    )
    review_dirty = (
        categories["entry_related_review"]["count"]
        + categories["unrelated_review"]["count"]
    )
    clean = dirty_count == 0
    foundation_change_set_isolated = dirty_count > 0 and foundation_dirty > 0 and review_dirty == 0
    foundation_stage_paths = sorted(
        categories["active_foundation_contract"]["paths"]
        + categories["legacy_tombstone_cleanup"]["paths"]
    )
    review_hold_paths = sorted(
        categories["entry_related_review"]["paths"]
        + categories["unrelated_review"]["paths"]
    )
    if clean:
        recommended_next_actions = [
            "Run scripts/entry_next_edge_control.sh train-readiness.",
            "After explicit vedtak, run smoke-train with --require-edge-audit.",
        ]
    else:
        recommended_next_actions = [
            "Review active_foundation_contract and legacy_tombstone_cleanup paths as the foundation cleanup change set.",
            "Do not stage entry_related_review or unrelated_review paths without a separate review decision.",
            "Commit/stash/remove all dirty paths, then rerun worktree-hygiene and train-readiness.",
        ]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "entry_foundation_worktree_hygiene_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS_CLEAN_GIT" if clean else "BLOCKED_BY_DIRTY_GIT",
        "real_smoke_train_allowed": bool(clean),
        "foundation_change_set_isolated": bool(foundation_change_set_isolated),
        "dirty_count": int(dirty_count),
        "foundation_cleanup_dirty_count": int(foundation_dirty),
        "review_before_stage_dirty_count": int(review_dirty),
        "categories": categories,
        "status_entries": entries,
        "recommended_next_actions": recommended_next_actions,
        "repo": str(REPO),
    }
    json_path = out_dir / f"ENTRY_FOUNDATION_WORKTREE_HYGIENE_{timestamp}.json"
    md_path = out_dir / f"ENTRY_FOUNDATION_WORKTREE_HYGIENE_{timestamp}.md"
    stage_paths_txt = out_dir / f"ENTRY_FOUNDATION_WORKTREE_STAGE_PATHS_{timestamp}.txt"
    hold_paths_txt = out_dir / f"ENTRY_FOUNDATION_WORKTREE_REVIEW_HOLD_PATHS_{timestamp}.txt"
    stage_status_tsv = out_dir / f"ENTRY_FOUNDATION_WORKTREE_STAGE_STATUS_{timestamp}.tsv"
    hold_status_tsv = out_dir / f"ENTRY_FOUNDATION_WORKTREE_REVIEW_HOLD_STATUS_{timestamp}.tsv"
    git_add_dry_run_txt = out_dir / f"ENTRY_FOUNDATION_WORKTREE_GIT_ADD_DRY_RUN_{timestamp}.txt"
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    report["foundation_stage_paths_txt"] = str(stage_paths_txt)
    report["review_hold_paths_txt"] = str(hold_paths_txt)
    report["foundation_stage_status_tsv"] = str(stage_status_tsv)
    report["review_hold_status_tsv"] = str(hold_status_tsv)
    report["git_add_dry_run_txt"] = str(git_add_dry_run_txt)
    report["foundation_stage_paths"] = foundation_stage_paths
    report["review_hold_paths"] = review_hold_paths
    stage_status_rows = _status_rows(entries, foundation_stage_paths)
    hold_status_rows = _status_rows(entries, review_hold_paths)
    report["foundation_stage_summary"] = _status_summary(stage_status_rows)
    report["review_hold_summary"] = _status_summary(hold_status_rows)
    stage_paths_txt.write_text("\n".join(foundation_stage_paths) + ("\n" if foundation_stage_paths else ""), encoding="utf-8")
    hold_paths_txt.write_text("\n".join(review_hold_paths) + ("\n" if review_hold_paths else ""), encoding="utf-8")
    _write_status_tsv(stage_status_tsv, stage_status_rows)
    _write_status_tsv(hold_status_tsv, hold_status_rows)
    if foundation_stage_paths:
        git_add_dry_run = _git_add_dry_run(stage_paths_txt)
    else:
        cached = _git_cached_paths()
        git_add_dry_run = {
            "cmd": ["git", "add", "--dry-run", f"--pathspec-from-file={stage_paths_txt}"],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "cached_before_count": len(cached),
            "cached_after_count": len(cached),
            "cached_unchanged": True,
        }
    dry_run_paths = sorted(set(_parse_git_add_dry_run_paths(str(git_add_dry_run.get("stdout") or ""))))
    dirty_paths = {str(entry["path"]) for entry in entries}
    stage_set = set(foundation_stage_paths)
    hold_set = set(review_hold_paths)
    stage_hold_overlap = sorted(stage_set & hold_set)
    stage_missing_from_dirty = sorted(stage_set - dirty_paths)
    hold_missing_from_dirty = sorted(hold_set - dirty_paths)
    dry_run_not_in_stage = sorted(set(dry_run_paths) - stage_set)
    dry_run_hold_overlap = sorted(set(dry_run_paths) & hold_set)
    stage_plan_checks = {
        "stage_hold_lists_disjoint": not stage_hold_overlap,
        "stage_paths_all_dirty": not stage_missing_from_dirty,
        "hold_paths_all_dirty": not hold_missing_from_dirty,
        "git_add_dry_run_rc_zero": int(git_add_dry_run["returncode"]) == 0,
        "git_add_dry_run_cached_unchanged": bool(git_add_dry_run["cached_unchanged"]),
        "git_add_dry_run_paths_subset_of_stage_paths": not dry_run_not_in_stage,
        "git_add_dry_run_has_no_hold_paths": not dry_run_hold_overlap,
    }
    stage_plan_safe = all(stage_plan_checks.values())
    required_review = _foundation_cleanup_required_review(
        dirty_paths=dirty_paths,
        stage_set=stage_set,
    )
    critical_gate_review = _foundation_cleanup_critical_gate_review(
        dirty_paths=dirty_paths,
        stage_set=stage_set,
    )
    foundation_cleanup_review_pass = (
        stage_plan_safe
        and not required_review["missing_from_repo"]
        and not required_review["dirty_missing_from_stage"]
        and not critical_gate_review["missing_from_repo"]
        and not critical_gate_review["dirty_missing_from_stage"]
    )
    foundation_cleanup_stage_ready = (
        bool(foundation_stage_paths)
        and foundation_cleanup_review_pass
        and int(git_add_dry_run["returncode"]) == 0
        and bool(git_add_dry_run["cached_unchanged"])
        and not dry_run_hold_overlap
        and not dry_run_not_in_stage
    )
    report["foundation_cleanup_review_decision"] = "PASS" if foundation_cleanup_review_pass else "FAIL"
    report["foundation_cleanup_required_review"] = required_review
    report["foundation_cleanup_critical_gate_review"] = critical_gate_review
    report["foundation_cleanup_stage_ready"] = bool(foundation_cleanup_stage_ready)
    report["foundation_cleanup_stage_command"] = [
        "git",
        "add",
        f"--pathspec-from-file={stage_paths_txt}",
    ]
    report["foundation_cleanup_stage_preconditions"] = {
        "stage_plan_safe": bool(stage_plan_safe),
        "foundation_cleanup_review_decision": report["foundation_cleanup_review_decision"],
        "git_add_dry_run_returncode": int(git_add_dry_run["returncode"]),
        "git_index_unchanged_by_dry_run": bool(git_add_dry_run["cached_unchanged"]),
        "git_add_dry_run_hold_overlap_count": len(dry_run_hold_overlap),
        "git_add_dry_run_not_in_stage_count": len(dry_run_not_in_stage),
    }
    report["foundation_cleanup_post_stage_verification"] = _post_stage_verification(
        cached_paths=_git_cached_paths(),
        stage_set=stage_set,
        hold_set=hold_set,
    )
    report["clean_git_resolution"] = _clean_git_resolution(report)
    report["stage_plan_safe"] = stage_plan_safe
    report["stage_plan_checks"] = stage_plan_checks
    report["stage_plan_diagnostics"] = {
        "stage_hold_overlap_count": len(stage_hold_overlap),
        "stage_hold_overlap_first_40": stage_hold_overlap[:40],
        "stage_missing_from_dirty_count": len(stage_missing_from_dirty),
        "stage_missing_from_dirty_first_40": stage_missing_from_dirty[:40],
        "hold_missing_from_dirty_count": len(hold_missing_from_dirty),
        "hold_missing_from_dirty_first_40": hold_missing_from_dirty[:40],
        "git_add_dry_run_path_count": len(dry_run_paths),
        "git_add_dry_run_not_in_stage_count": len(dry_run_not_in_stage),
        "git_add_dry_run_not_in_stage_first_40": dry_run_not_in_stage[:40],
        "git_add_dry_run_hold_overlap_count": len(dry_run_hold_overlap),
        "git_add_dry_run_hold_overlap_first_40": dry_run_hold_overlap[:40],
    }
    report["git_add_dry_run"] = git_add_dry_run
    git_add_dry_run_txt.write_text(git_add_dry_run["stdout"] + git_add_dry_run["stderr"], encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    latest_json = out_dir / "ENTRY_FOUNDATION_WORKTREE_HYGIENE_latest.json"
    latest_md = out_dir / "ENTRY_FOUNDATION_WORKTREE_HYGIENE_latest.md"
    latest_stage_paths = out_dir / "ENTRY_FOUNDATION_WORKTREE_STAGE_PATHS_latest.txt"
    latest_hold_paths = out_dir / "ENTRY_FOUNDATION_WORKTREE_REVIEW_HOLD_PATHS_latest.txt"
    latest_stage_status = out_dir / "ENTRY_FOUNDATION_WORKTREE_STAGE_STATUS_latest.tsv"
    latest_hold_status = out_dir / "ENTRY_FOUNDATION_WORKTREE_REVIEW_HOLD_STATUS_latest.tsv"
    latest_git_add_dry_run = out_dir / "ENTRY_FOUNDATION_WORKTREE_GIT_ADD_DRY_RUN_latest.txt"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_stage_paths.write_text(stage_paths_txt.read_text(encoding="utf-8"), encoding="utf-8")
    latest_hold_paths.write_text(hold_paths_txt.read_text(encoding="utf-8"), encoding="utf-8")
    latest_stage_status.write_text(stage_status_tsv.read_text(encoding="utf-8"), encoding="utf-8")
    latest_hold_status.write_text(hold_status_tsv.read_text(encoding="utf-8"), encoding="utf-8")
    latest_git_add_dry_run.write_text(git_add_dry_run_txt.read_text(encoding="utf-8"), encoding="utf-8")
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "dirty_count": report["dirty_count"],
                    "foundation_cleanup_dirty_count": report["foundation_cleanup_dirty_count"],
                    "review_before_stage_dirty_count": report["review_before_stage_dirty_count"],
                    "json_path": report["json_path"],
                    "md_path": report["md_path"],
                    "foundation_stage_paths_txt": str(latest_stage_paths),
                    "review_hold_paths_txt": str(latest_hold_paths),
                    "foundation_stage_status_tsv": str(latest_stage_status),
                    "review_hold_status_tsv": str(latest_hold_status),
                    "git_add_dry_run_txt": str(latest_git_add_dry_run),
                    "git_add_dry_run_returncode": git_add_dry_run["returncode"],
                    "git_index_unchanged_by_dry_run": git_add_dry_run["cached_unchanged"],
                    "stage_plan_safe": report["stage_plan_safe"],
                    "foundation_cleanup_review_decision": report["foundation_cleanup_review_decision"],
                    "foundation_cleanup_stage_ready": report["foundation_cleanup_stage_ready"],
                    "foundation_cleanup_stage_command": report["foundation_cleanup_stage_command"],
                    "foundation_cleanup_post_stage_verification": report["foundation_cleanup_post_stage_verification"],
                    "clean_git_resolution": report["clean_git_resolution"],
                },
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        )
    if args.fail_on_dirty and not clean:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--fail-on-dirty", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
