#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ACTION = "CLEANUP_OVERVIEW_CURRENT_BASELINES_AND_OUTDATED_RUNS_V1"
DEFAULT_ARTIFACT_BASE_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DRY_RUN_ONLY = True

CURRENT_140_ROOT_NAME = "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1_20260428T065344Z_LOCK"
REJECT_REBUILD_ROOT_NAME = "REJECT_OR_REBUILD_BEST_LANE_FROM_CAUSAL_SIGNALS_V1_20260428T063714Z_LOCK"
STUDENT_ROOT_NAME = "BUILD_MODEL_TO_LEARN_BEST_LANE_MEMBERSHIP_AS_OOF_TARGET_V1_20260427T202519Z_LOCK"
STABILITY_185_ROOT_NAME = "STABILITY_RECHECK_BEST_LANE_185_139_BEFORE_R6_V1_20260427T200530Z_LOCK"
BEST_LANE_PACKAGE_ROOT_NAME = "MATERIALIZE_BEST_LANE_CANDIDATE_PACKAGE_REQUIRES_EXPLICIT_GATE_V1_20260427T193354Z_LOCK"
LANE_PACK_ROOT_NAME = "PARALLEL_TAIL_R6_R5_2_REPAIR_LANE_PACK_V1_20260427T191454Z_LOCK"
PLUS45_SIDECAR_ROOT_NAME = "PLUS45_AS_OF_FEATURE_GAP_SHADOW_EXPLORATION_V1_20260428T074409Z_LOCK"
WEDNESDAY_SKELETON_ROOT_NAME = "FIND_BACK_TO_WEDNESDAY_R6_SKELETON_AND_REBUILD_MONDAY_FOUNDATION_V1_20260427T083808Z_LOCK"
WEDNESDAY_SNAPSHOT_ROOT_NAME = "MONDAY_WEDNESDAY_BENCHMARK_SNAPSHOT_V1_20260424T0900Z"
TAIL_REPAIRED_R5_2_PACKAGE_ROOT_NAME = (
    "BUILD_TAIL_REPAIRED_R5_2_CANDIDATE_PACKAGE_REQUIRES_EXPLICIT_GATE_V1_20260427T175754Z_LOCK"
)
TAIL_REPAIRED_R6_ROOT_NAME = "RUN_R6_RETRAIN_FROM_TAIL_REPAIRED_R5_2_PACKAGE_EXPLICIT_GATE_V1_20260427T185325Z_LOCK"
R5_2_130_86_PACKAGE_ROOT_NAME = "BUILD_R5_2_PACKAGE_FROM_CANDIDATE_REQUIRES_EXPLICIT_GATE_V1_20260427T152500Z_LOCK"
R5_2_130_86_CANDIDATE_ROOT_NAME = (
    "BUILD_R5_2_FROM_COVERAGE_AWARE_OPPORTUNITY_BASE_WITH_FIXED_CONTROLS_V1_20260427T150214Z_LOCK"
)
R5_2_COVERAGE_PROXY_ROOT_NAME = (
    "BUILD_COVERAGE_AWARE_R5_2_OPPORTUNITY_BASE_WITH_LOW_SUPPORT_POLICY_V1_20260427T142902Z_LOCK"
)

MAINLINE_FINAL_STATUS = "140_94_CAUSAL_BASELINE_NEEDS_RULE_DISTILLATION_BEFORE_ADAPTER"
MAINLINE_NEXT_ACTION = "DISTILL_140_94_CAUSAL_BASELINE_TO_RULES_AND_VETOES_V1"

CLASS_KEEP_ACTIVE = "KEEP_ACTIVE"
CLASS_KEEP_REFERENCE = "KEEP_REFERENCE"
CLASS_KEEP_DIAGNOSTIC = "KEEP_DIAGNOSTIC"
CLASS_KEEP_PLANNED_SIDECAR = "KEEP_PLANNED_SIDECAR"
CLASS_ARCHIVE_COLD = "ARCHIVE_COLD_CANDIDATE"
CLASS_DELETE_SAFE = "DELETE_SAFE_CANDIDATE"
CLASS_UNKNOWN = "UNKNOWN_DO_NOT_TOUCH"
CLASS_BLOCKED = "BLOCKED_DO_NOT_TOUCH"

CLASSIFICATIONS = {
    CLASS_KEEP_ACTIVE,
    CLASS_KEEP_REFERENCE,
    CLASS_KEEP_DIAGNOSTIC,
    CLASS_KEEP_PLANNED_SIDECAR,
    CLASS_ARCHIVE_COLD,
    CLASS_DELETE_SAFE,
    CLASS_UNKNOWN,
    CLASS_BLOCKED,
}

ALLOWED_FINAL_STATUSES = {
    "CLEANUP_OVERVIEW_READY_FOR_ARCHIVE_PLAN",
    "CLEANUP_OVERVIEW_READY_FOR_SAFE_CACHE_DELETE_PLAN",
    "CLEANUP_OVERVIEW_FOUND_REFERENCES_REQUIRING_MANUAL_REVIEW",
    "CLEANUP_OVERVIEW_BLOCKED_BY_MISSING_CURRENT_BASELINE_ARTIFACT",
    "CLEANUP_OVERVIEW_BLOCKED_BY_UNCLEAR_DEPENDENCY_GRAPH",
    "CLEANUP_OVERVIEW_BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "PLAN_ARCHIVE_OUTDATED_RUNS_WITH_MANIFEST_V1",
    "PLAN_SAFE_DELETE_CACHE_AND_TEMP_FILES_V1",
    "DEEPEN_ARTIFACT_DEPENDENCY_GRAPH_AUDIT_V1",
    "RESOLVE_MISSING_REFERENCED_ARTIFACTS_V1",
    "LOCK_CURRENT_BASELINE_REFERENCE_BUNDLE_V1",
    "CONTINUE_140_94_DISTILLATION_UNCHANGED",
}

REQUIRED_OUTPUTS = [
    "cleanup_overview_input_scope_manifest_v1.json",
    "cleanup_overview_input_scope_manifest_v1.md",
    "cleanup_overview_project_state_reference_scan_v1.json",
    "cleanup_overview_project_state_reference_scan_v1.md",
    "cleanup_overview_artifact_inventory_v1.csv",
    "cleanup_overview_artifact_inventory_v1.json",
    "cleanup_overview_artifact_inventory_v1.md",
    "cleanup_overview_repo_file_inventory_v1.csv",
    "cleanup_overview_repo_file_inventory_v1.json",
    "cleanup_overview_repo_file_inventory_v1.md",
    "cleanup_overview_dependency_graph_v1.json",
    "cleanup_overview_dependency_graph_v1.md",
    "cleanup_overview_classification_v1.csv",
    "cleanup_overview_classification_v1.json",
    "cleanup_overview_classification_v1.md",
    "cleanup_overview_current_baselines_lock_summary_v1.json",
    "cleanup_overview_current_baselines_lock_summary_v1.md",
    "cleanup_overview_future_cleanup_plan_dry_run_v1.csv",
    "cleanup_overview_future_cleanup_plan_dry_run_v1.json",
    "cleanup_overview_future_cleanup_plan_dry_run_v1.md",
    "cleanup_overview_cleanup_risk_audit_v1.json",
    "cleanup_overview_cleanup_risk_audit_v1.md",
    "cleanup_overview_recommendation_v1.json",
    "cleanup_overview_recommendation_v1.md",
    "cleanup_overview_current_baselines_and_outdated_runs_go_no_go_v1.json",
]

FORBIDDEN_CALL_NAMES = {
    "remove",
    "rmtree",
}
FORBIDDEN_ATTRIBUTE_CALLS = {
    "unlink",
    "rmdir",
    "rename",
    "move",
    "remove",
    "rmtree",
}

TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
REPO_SKIP_DIRS = {".git", ".venv", "venv", "venvs", "node_modules", ".mypy_cache", ".ruff_cache"}
REFERENCE_SCAN_MAX_BYTES = 2_000_000


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _write_report(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_text(path: Path, max_bytes: int = REFERENCE_SCAN_MAX_BYTES) -> str:
    try:
        if not path.exists() or not path.is_file() or path.stat().st_size > max_bytes:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _maybe_read_json(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _safe_stat(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"exists_v1": False}
    return {
        "exists_v1": True,
        "size_bytes_v1": int(stat.st_size),
        "modified_time_utc_v1": datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mtime_ns_v1": int(stat.st_mtime_ns),
    }


def known_roots(artifact_base_root: Path = DEFAULT_ARTIFACT_BASE_ROOT) -> dict[str, Path]:
    return {
        "current_140_94_precheck_root_v1": artifact_base_root / CURRENT_140_ROOT_NAME,
        "reject_rebuild_root_v1": artifact_base_root / REJECT_REBUILD_ROOT_NAME,
        "student_oof_root_v1": artifact_base_root / STUDENT_ROOT_NAME,
        "stability_185_139_root_v1": artifact_base_root / STABILITY_185_ROOT_NAME,
        "best_lane_package_root_v1": artifact_base_root / BEST_LANE_PACKAGE_ROOT_NAME,
        "lane_pack_root_v1": artifact_base_root / LANE_PACK_ROOT_NAME,
        "plus45_sidecar_root_v1": artifact_base_root / PLUS45_SIDECAR_ROOT_NAME,
        "wednesday_skeleton_root_v1": artifact_base_root / WEDNESDAY_SKELETON_ROOT_NAME,
        "wednesday_snapshot_root_v1": artifact_base_root / WEDNESDAY_SNAPSHOT_ROOT_NAME,
        "tail_repaired_r5_2_140_94_source_root_v1": artifact_base_root / TAIL_REPAIRED_R5_2_PACKAGE_ROOT_NAME,
        "tail_repaired_r6_140_94_reference_root_v1": artifact_base_root / TAIL_REPAIRED_R6_ROOT_NAME,
        "previous_r5_2_130_86_package_root_v1": artifact_base_root / R5_2_130_86_PACKAGE_ROOT_NAME,
        "previous_r5_2_130_86_candidate_root_v1": artifact_base_root / R5_2_130_86_CANDIDATE_ROOT_NAME,
        "coverage_proxy_188_136_root_v1": artifact_base_root / R5_2_COVERAGE_PROXY_ROOT_NAME,
    }


def validate_dry_run_only() -> bool:
    if not DRY_RUN_ONLY:
        raise RuntimeError("CLEANUP_OVERVIEW_SCRIPT_MUST_BE_DRY_RUN_ONLY")
    return True


def validate_explicit_artifact_base_root(path: Path) -> bool:
    if path != DEFAULT_ARTIFACT_BASE_ROOT:
        raise RuntimeError(f"ARTIFACT_BASE_ROOT_MUST_BE_EXPLICIT_DEFAULT_FOR_REAL_GATE: {path}")
    return True


def validate_current_roots_are_explicit(paths: Iterable[Path]) -> bool:
    failures = []
    for path in paths:
        text = str(path)
        if "*" in text or "latest" in text.lower():
            failures.append(text)
    if failures:
        raise RuntimeError(f"IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN: {failures}")
    return True


def validate_no_forbidden_actions(
    *,
    cleanup_action: bool = False,
    r6: bool = False,
    adapter: bool = False,
    package: bool = False,
    freeze: bool = False,
    promo: bool = False,
    live: bool = False,
    optuna: bool = False,
    model_training: bool = False,
    selection_materialized: bool = False,
) -> dict[str, Any]:
    failures = []
    if cleanup_action:
        failures.append("CLEANUP_ACTION_FORBIDDEN")
    if r6:
        failures.append("R6_FORBIDDEN")
    if adapter:
        failures.append("ADAPTER_BUILD_FORBIDDEN")
    if package:
        failures.append("PACKAGE_BUILD_FORBIDDEN")
    if freeze:
        failures.append("FREEZE_FORBIDDEN")
    if promo:
        failures.append("PROMO_FORBIDDEN")
    if live:
        failures.append("LIVE_FORBIDDEN")
    if optuna:
        failures.append("OPTUNA_FORBIDDEN")
    if model_training:
        failures.append("MODEL_TRAINING_FORBIDDEN")
    if selection_materialized:
        failures.append("CANDIDATE_SELECTION_MATERIALIZATION_FORBIDDEN")
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures}


def validate_script_has_no_destructive_calls(script_path: Path | None = None) -> bool:
    path = script_path or Path(__file__).resolve()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALL_NAMES:
            failures.append(func.id)
        if isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_ATTRIBUTE_CALLS:
            failures.append(func.attr)
    if failures:
        raise RuntimeError(f"DESTRUCTIVE_CALLS_FORBIDDEN_IN_DRY_RUN_SCRIPT: {sorted(set(failures))}")
    return True


def _iter_files(root: Path, *, max_depth: int | None = None) -> Iterable[Path]:
    def walk(path: Path, depth: int) -> Iterable[Path]:
        try:
            entries = sorted(path.iterdir(), key=lambda item: item.name)
        except OSError:
            return
        for entry in entries:
            if entry.is_dir():
                if max_depth is None or depth < max_depth:
                    yield from walk(entry, depth + 1)
            elif entry.is_file():
                yield entry

    if root.exists():
        yield from walk(root, 0)


def _iter_repo_paths(repo_root: Path) -> Iterable[Path]:
    def walk(path: Path) -> Iterable[Path]:
        try:
            entries = sorted(path.iterdir(), key=lambda item: item.name)
        except OSError:
            return
        for entry in entries:
            rel_parts = entry.relative_to(repo_root).parts
            if entry.is_dir():
                if entry.name in {".git", ".venv", "venv", "venvs", "node_modules", ".mypy_cache", ".ruff_cache"}:
                    continue
                if entry.name in {"__pycache__", ".pytest_cache"}:
                    yield entry
                    continue
                yield from walk(entry)
            elif entry.is_file():
                if any(part in REPO_SKIP_DIRS for part in rel_parts):
                    continue
                yield entry

    yield from walk(repo_root)


def _extract_artifact_paths(text: str) -> list[str]:
    if not text:
        return []
    pattern = re.escape(str(DEFAULT_ARTIFACT_BASE_ROOT)) + r"/[A-Za-z0-9_./=:+@-]+"
    found: list[str] = []
    for match in re.findall(pattern, text):
        clean = match.rstrip("`'\"),.]")
        if clean not in found:
            found.append(clean)
    return found


def _extract_status_tokens(text: str) -> list[str]:
    if not text:
        return []
    tokens: list[str] = []
    for token in re.findall(r"\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+){2,}\b", text):
        if token not in tokens:
            tokens.append(token)
    return tokens


def _extract_script_test_names(text: str) -> list[str]:
    if not text:
        return []
    names: list[str] = []
    for token in re.findall(r"\b(?:gx1/scripts/)?(?:materialize|run|train|audit|test)_[A-Za-z0-9_]+\.py\b", text):
        if token not in names:
            names.append(token)
    return names


def _root_gate_and_timestamp(basename: str) -> tuple[str, str | None]:
    match = re.match(r"^(?P<gate>.+?)_(?P<stamp>\d{8}T(?:\d{4}|\d{6})Z(?:_[A-Z]+)?|POST_[A-Z0-9_]+|LOCAL_SEARCH_LOCK)(?:_LOCK)?$", basename)
    if match:
        return match.group("gate"), match.group("stamp")
    match = re.match(r"^(?P<gate>.+?)_(?P<stamp>\d{8}T_LOCK)$", basename)
    if match:
        return match.group("gate"), match.group("stamp")
    return basename.removesuffix("_LOCK"), None


def _scan_status_from_files(root: Path) -> tuple[str, str, list[str]]:
    status_values: list[str] = []
    next_values: list[str] = []
    tokens: list[str] = []
    important_names = {
        "summary_v1.json",
        "status_v1.json",
        "manifest_v1.json",
        "report_v1.md",
    }
    for file_path in _iter_files(root, max_depth=1):
        name = file_path.name.lower()
        if file_path.name in important_names or "go_no_go" in name or "recommendation" in name:
            text = _read_text(file_path)
            tokens.extend(_extract_status_tokens(text))
            if file_path.suffix == ".json":
                data = _maybe_read_json(file_path)
                for key in (
                    "status_v1",
                    "final_status_v1",
                    "go_no_go_v1",
                    "decision_v1",
                    "go_no_go",
                    "current_go_no_go_v1",
                ):
                    value = data.get(key)
                    if isinstance(value, str) and value not in status_values:
                        status_values.append(value)
                for key in (
                    "next_action_v1",
                    "next_recommended_action_v1",
                    "mainline_next_action_v1",
                    "shadow_next_recommended_action_v1",
                ):
                    value = data.get(key)
                    if isinstance(value, str) and value not in next_values:
                        next_values.append(value)
    return "|".join(status_values[:5]), "|".join(next_values[:5]), sorted(set(tokens))[:40]


def classify_artifact_root(path: Path, artifact_base_root: Path = DEFAULT_ARTIFACT_BASE_ROOT) -> tuple[str, str]:
    name = path.name
    roots = known_roots(artifact_base_root)
    if path == roots["current_140_94_precheck_root_v1"]:
        return CLASS_KEEP_ACTIVE, "Current mainline 140/94 causal baseline precheck root."
    if name.startswith(ACTION):
        return CLASS_KEEP_REFERENCE, "Current cleanup overview evidence root."
    if path in {
        roots["reject_rebuild_root_v1"],
        roots["student_oof_root_v1"],
        roots["lane_pack_root_v1"],
        roots["wednesday_skeleton_root_v1"],
        roots["wednesday_snapshot_root_v1"],
        roots["tail_repaired_r5_2_140_94_source_root_v1"],
        roots["tail_repaired_r6_140_94_reference_root_v1"],
        roots["previous_r5_2_130_86_package_root_v1"],
        roots["previous_r5_2_130_86_candidate_root_v1"],
        roots["coverage_proxy_188_136_root_v1"],
    }:
        return CLASS_KEEP_REFERENCE, "Comparator, benchmark, source, or history needed to explain current state."
    if path in {roots["stability_185_139_root_v1"], roots["best_lane_package_root_v1"]}:
        return CLASS_KEEP_DIAGNOSTIC, "Diagnostic evidence for rejected/blocked 185/139 membership-bound branch."
    if path == roots["plus45_sidecar_root_v1"] or "PLUS45" in name or "PLUS_45" in name:
        return CLASS_KEEP_PLANNED_SIDECAR, "+45 diagnostic-only sidecar/shadow evidence."
    upper = name.upper()
    if "WEDNESDAY" in upper or "180_149" in upper:
        return CLASS_KEEP_REFERENCE, "Wednesday benchmark/comparator reference."
    if "185_139" in upper or "BEST_LANE" in upper or "LANE_08" in upper:
        return CLASS_KEEP_DIAGNOSTIC, "Best-lane or 185/139 diagnostic branch evidence."
    if "__PYCACHE__" in upper or ".PYTEST_CACHE" in upper or "TMP" in upper or "SCRATCH" in upper:
        return CLASS_DELETE_SAFE, "Disposable cache or scratch directory candidate, dry-run only."
    if path.is_dir():
        has_manifest = any(file_path.name.lower().startswith("manifest") for file_path in _iter_files(path, max_depth=1))
        has_summary = (path / "summary_v1.json").exists() or (path / "status_v1.json").exists()
        if has_manifest or has_summary or name.endswith("_LOCK"):
            return CLASS_ARCHIVE_COLD, "Old artifact/run with historical value; archive candidate only after manifest review."
    return CLASS_UNKNOWN, "Dependency unclear; keep untouched until manual review."


def classify_repo_file(path: Path, repo_root: Path) -> tuple[str, str, str]:
    rel = path.relative_to(repo_root)
    parts = set(rel.parts)
    name = path.name
    if "__pycache__" in parts or name.endswith(".pyc") or ".pytest_cache" in parts:
        return CLASS_DELETE_SAFE, "cache", "Disposable Python/pytest cache candidate, dry-run only."
    if name in {"PROJECT_STATE.md", "DECISION_LOG.md", "AGENTS.md"}:
        return CLASS_KEEP_ACTIVE, "state_doc", "Project state, decision log, or agent guardrail file."
    if name == "materialize_cleanup_overview_current_baselines_and_outdated_runs_v1.py":
        return CLASS_KEEP_ACTIVE, "script", "Current cleanup overview gate script."
    if name == "test_cleanup_overview_current_baselines_and_outdated_runs_v1.py":
        return CLASS_KEEP_ACTIVE, "test", "Current cleanup overview targeted test."
    if name.startswith("materialize_") and name.endswith(".py"):
        return CLASS_KEEP_REFERENCE, "script", "Historical gate script; preserve until dependency review."
    if name.startswith("test_") and name.endswith(".py"):
        return CLASS_KEEP_REFERENCE, "test", "Historical or active test; preserve until dependency review."
    if name.endswith((".log", ".tmp")):
        return CLASS_DELETE_SAFE, "temp_or_log", "Temporary or debug log candidate, dry-run only."
    if name in {"pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini"} or name.startswith("requirements"):
        return CLASS_KEEP_ACTIVE, "config", "Project config or dependency file."
    return CLASS_UNKNOWN, "repo_file", "Not enough local evidence for cleanup class; do not touch."


def _scan_project_references(repo_root: Path, artifact_base_root: Path) -> dict[str, Any]:
    files = {
        "PROJECT_STATE.md": repo_root / "PROJECT_STATE.md",
        "DECISION_LOG.md": repo_root / "DECISION_LOG.md",
        "AGENTS.md": repo_root / "AGENTS.md",
    }
    config_files = [
        path
        for path in [repo_root / "pyproject.toml", repo_root / "pytest.ini", repo_root / "setup.cfg", repo_root / "tox.ini"]
        if path.exists()
    ]
    texts = {name: _read_text(path, max_bytes=5_000_000) for name, path in files.items()}
    combined = "\n".join(texts.values())
    artifact_paths = sorted(set(_extract_artifact_paths(combined)))
    report_paths = [path for path in artifact_paths if str(artifact_base_root) in path]
    script_names = sorted(set(_extract_script_test_names(combined)))
    status_strings = sorted(set(_extract_status_tokens(combined)))
    referenced_by_file = {
        name: {
            "exists_v1": files[name].exists(),
            "artifact_paths_v1": _extract_artifact_paths(text),
            "script_or_test_names_v1": _extract_script_test_names(text),
            "status_strings_v1": _extract_status_tokens(text),
        }
        for name, text in texts.items()
    }
    keyword_refs = {
        "comparator_references_v1": [token for token in status_strings if "COMPARATOR" in token or "REFERENCE" in token],
        "r6_references_v1": sorted(set(re.findall(r"\bR6[A-Z0-9_]*\b", combined))),
        "adapter_references_v1": sorted(set(token for token in status_strings if "ADAPTER" in token)),
        "baseline_140_94_references_v1": len(re.findall(r"140[/_]94", combined)),
        "best_lane_185_139_references_v1": len(re.findall(r"185[/_]139", combined)),
        "plus45_references_v1": len(re.findall(r"\+45|PLUS45|plus45", combined)),
        "wednesday_180_149_references_v1": len(re.findall(r"180[/_]149|180 bad / 149|Wednesday 180", combined)),
    }
    missing_artifact_paths = [path for path in artifact_paths if not Path(path).exists()]
    return {
        "layer_name": "CLEANUP_OVERVIEW_PROJECT_STATE_REFERENCE_SCAN_V1",
        "repo_root_v1": str(repo_root),
        "artifact_base_root_v1": str(artifact_base_root),
        "source_files_v1": {name: str(path) for name, path in files.items()},
        "local_config_files_v1": [str(path) for path in config_files],
        "artifact_paths_v1": artifact_paths,
        "report_paths_v1": report_paths,
        "script_names_v1": script_names,
        "test_names_v1": [name for name in script_names if Path(name).name.startswith("test_")],
        "status_strings_v1": status_strings,
        "next_actions_v1": [token for token in status_strings if token.endswith("_V1") and ("ACTION" in token or "DISTILL" in token)],
        "referenced_by_file_v1": referenced_by_file,
        "keyword_references_v1": keyword_refs,
        "missing_referenced_artifact_paths_v1": missing_artifact_paths,
        "project_state_text_hash_basis_v1": {
            "PROJECT_STATE.md_chars_v1": len(texts["PROJECT_STATE.md"]),
            "DECISION_LOG.md_chars_v1": len(texts["DECISION_LOG.md"]),
            "AGENTS.md_chars_v1": len(texts["AGENTS.md"]),
        },
    }


def _repo_reference_index(repo_root: Path, artifact_basenames: Sequence[str]) -> dict[str, dict[str, Any]]:
    index = {name: {"referenced_by_scripts_tests_v1": False, "referencing_files_v1": []} for name in artifact_basenames}
    for path in _iter_repo_paths(repo_root):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel = str(path.relative_to(repo_root))
        if not (rel.startswith("gx1/scripts/") or rel.startswith("tests/") or rel in {"PROJECT_STATE.md", "DECISION_LOG.md"}):
            continue
        text = _read_text(path)
        if not text:
            continue
        for basename in artifact_basenames:
            if basename in text:
                row = index[basename]
                row["referenced_by_scripts_tests_v1"] = rel.startswith("gx1/scripts/") or rel.startswith("tests/")
                row["referencing_files_v1"].append(rel)
    return index


def _artifact_inventory(
    artifact_base_root: Path,
    repo_root: Path,
    project_state_text: str,
    decision_log_text: str,
) -> list[dict[str, Any]]:
    roots = sorted([path for path in artifact_base_root.iterdir() if path.is_dir()], key=lambda item: item.name)
    reference_index = _repo_reference_index(repo_root, [root.name for root in roots])
    rows: list[dict[str, Any]] = []
    for root in roots:
        files = list(_iter_files(root))
        suffixes = {file_path.suffix.lower() for file_path in files}
        size = 0
        for file_path in files:
            try:
                size += file_path.stat().st_size
            except OSError:
                pass
        gate_name, stamp = _root_gate_and_timestamp(root.name)
        likely_status, likely_next, status_tokens = _scan_status_from_files(root)
        classification, reason = classify_artifact_root(root, artifact_base_root)
        filenames = [file_path.name.lower() for file_path in files]
        row = {
            "path_v1": str(root),
            "basename_v1": root.name,
            "created_or_modified_time_utc_v1": _safe_stat(root).get("modified_time_utc_v1"),
            "size_estimate_bytes_v1": size,
            "file_count_v1": len(files),
            "contains_json_v1": ".json" in suffixes,
            "contains_md_v1": ".md" in suffixes,
            "contains_csv_v1": ".csv" in suffixes,
            "contains_parquet_v1": ".parquet" in suffixes,
            "contains_log_v1": ".log" in suffixes or any("log" in name for name in filenames),
            "likely_gate_name_v1": gate_name,
            "timestamp_suffix_v1": stamp,
            "lock_suffix_v1": root.name.endswith("_LOCK"),
            "referenced_in_project_state_v1": str(root) in project_state_text or root.name in project_state_text,
            "referenced_in_decision_log_v1": str(root) in decision_log_text or root.name in decision_log_text,
            "referenced_by_current_scripts_tests_v1": reference_index[root.name]["referenced_by_scripts_tests_v1"],
            "referencing_repo_files_v1": "|".join(reference_index[root.name]["referencing_files_v1"][:20]),
            "contains_go_no_go_json_v1": any("go_no_go" in name and name.endswith(".json") for name in filenames),
            "contains_recommendation_json_md_v1": any("recommendation" in name for name in filenames),
            "contains_manifest_v1": any("manifest" in name for name in filenames),
            "has_hashes_integrity_file_v1": any(
                "hash" in name or "sha256" in name or "integrity" in name for name in filenames
            ),
            "likely_status_v1": likely_status,
            "likely_next_action_v1": likely_next,
            "status_tokens_sample_v1": "|".join(status_tokens[:20]),
            "candidate_classification_preliminary_v1": classification,
            "classification_reason_v1": reason,
        }
        rows.append(row)
    return rows


def _repo_file_inventory(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _iter_repo_paths(repo_root):
        classification, file_type, reason = classify_repo_file(path, repo_root)
        stat = _safe_stat(path)
        text = _read_text(path) if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES else ""
        rel = path.relative_to(repo_root)
        row = {
            "path_v1": str(path),
            "relative_path_v1": str(rel),
            "file_type_v1": file_type,
            "is_directory_v1": path.is_dir(),
            "size_bytes_v1": stat.get("size_bytes_v1", 0),
            "modified_time_utc_v1": stat.get("modified_time_utc_v1"),
            "references_artifact_roots_v1": bool(_extract_artifact_paths(text)),
            "references_current_140_94_v1": bool(re.search(r"140[/_]94", text)),
            "references_185_139_v1": bool(re.search(r"185[/_]139", text)),
            "references_plus45_v1": bool(re.search(r"\+45|PLUS45|plus45", text)),
            "references_r6_v1": "R6" in text,
            "references_adapter_v1": "adapter" in text.lower() or "ADAPTER" in text,
            "appears_active_current_v1": classification == CLASS_KEEP_ACTIVE,
            "appears_historical_outdated_v1": classification in {CLASS_KEEP_REFERENCE, CLASS_ARCHIVE_COLD},
            "candidate_cleanup_class_v1": classification,
            "classification_reason_v1": reason,
        }
        rows.append(row)
    return rows


def _find_wednesday_reference_status(artifact_rows: list[dict[str, Any]], project_scan: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for row in artifact_rows:
        name = str(row["basename_v1"]).upper()
        status = str(row.get("likely_status_v1", ""))
        token_sample = str(row.get("status_tokens_sample_v1", ""))
        if "WEDNESDAY" in name or "180/149" in status or "180_149" in token_sample:
            candidates.append(row)
    exact_hits = []
    for row in candidates:
        root = Path(str(row["path_v1"]))
        for file_path in _iter_files(root, max_depth=2):
            text = _read_text(file_path)
            if re.search(r"180[/_]149|180 bad / 149|180.*149", text):
                exact_hits.append({"artifact_root_v1": str(root), "file_v1": str(file_path)})
                break
    if exact_hits:
        return {
            "status_v1": "WEDNESDAY_180_149_REFERENCE_FOUND_IN_SCAN",
            "artifact_roots_v1": sorted({hit["artifact_root_v1"] for hit in exact_hits}),
            "evidence_files_v1": exact_hits[:20],
        }
    if project_scan["keyword_references_v1"]["wednesday_180_149_references_v1"]:
        return {
            "status_v1": "WEDNESDAY_REFERENCE_MENTIONED_BUT_180_149_ARTIFACT_NOT_BOUND",
            "artifact_roots_v1": [str(row["path_v1"]) for row in candidates],
            "evidence_files_v1": [],
        }
    return {
        "status_v1": "WEDNESDAY_180_149_REFERENCE_NOT_FOUND_IN_SCAN",
        "artifact_roots_v1": [],
        "evidence_files_v1": [],
    }


def _dependency_graph(
    artifact_rows: list[dict[str, Any]],
    repo_rows: list[dict[str, Any]],
    project_scan: dict[str, Any],
    wednesday_status: dict[str, Any],
    artifact_base_root: Path,
) -> dict[str, Any]:
    scripts = {Path(row["relative_path_v1"]).name: row for row in repo_rows if row["file_type_v1"] == "script"}
    tests = {Path(row["relative_path_v1"]).name: row for row in repo_rows if row["file_type_v1"] == "test"}
    edges: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for row in artifact_rows:
        gate_name = str(row["likely_gate_name_v1"]).lower()
        expected_script = f"materialize_{gate_name.lower()}".replace("__", "_")
        if not expected_script.endswith(".py"):
            expected_script += ".py"
        script = scripts.get(expected_script)
        classification = row["candidate_classification_preliminary_v1"]
        nodes.append(
            {
                "id_v1": row["path_v1"],
                "type_v1": "artifact_root",
                "classification_v1": classification,
                "likely_gate_name_v1": row["likely_gate_name_v1"],
            }
        )
        if script:
            edges.append(
                {
                    "from_v1": script["relative_path_v1"],
                    "to_v1": row["path_v1"],
                    "edge_type_v1": "SCRIPT_LIKELY_CREATED_ARTIFACT_BY_GATE_NAME",
                    "confidence_v1": "MEDIUM",
                }
            )
        elif classification in {CLASS_KEEP_ACTIVE, CLASS_KEEP_REFERENCE, CLASS_KEEP_DIAGNOSTIC, CLASS_KEEP_PLANNED_SIDECAR}:
            unresolved.append(
                {
                    "artifact_root_v1": row["path_v1"],
                    "missing_inferred_script_v1": expected_script,
                    "reason_v1": "No matching materialize script inferred by basename.",
                }
            )
    for script_name in sorted(scripts):
        test_name = "test_" + script_name.removeprefix("materialize_")
        if test_name in tests:
            edges.append(
                {
                    "from_v1": tests[test_name]["relative_path_v1"],
                    "to_v1": scripts[script_name]["relative_path_v1"],
                    "edge_type_v1": "TARGETED_TEST_COVERS_SCRIPT_NAME",
                    "confidence_v1": "HIGH",
                }
            )
    for path in project_scan["referenced_by_file_v1"]["PROJECT_STATE.md"]["artifact_paths_v1"]:
        edges.append(
            {
                "from_v1": "PROJECT_STATE.md",
                "to_v1": path,
                "edge_type_v1": "PROJECT_STATE_REFERENCES_ARTIFACT",
                "confidence_v1": "HIGH",
            }
        )
    for path in project_scan["referenced_by_file_v1"]["DECISION_LOG.md"]["artifact_paths_v1"]:
        edges.append(
            {
                "from_v1": "DECISION_LOG.md",
                "to_v1": path,
                "edge_type_v1": "DECISION_LOG_REFERENCES_ARTIFACT",
                "confidence_v1": "HIGH",
            }
        )
    roots = known_roots(artifact_base_root)
    current_edges = [
        ("current_140_94_precheck_root_v1", "reject_rebuild_root_v1", "140/94 precheck was entered after reject/rebuild"),
        ("current_140_94_precheck_root_v1", "student_oof_root_v1", "140/94 decision relies on student failure to recover +45"),
        ("current_140_94_precheck_root_v1", "stability_185_139_root_v1", "185/139 comparator remained diagnostic"),
        ("current_140_94_precheck_root_v1", "best_lane_package_root_v1", "Best-lane package explains +45 branch"),
        ("current_140_94_precheck_root_v1", "lane_pack_root_v1", "Lane pack explains lane 08 and lane 10 140/94 reproduction"),
        ("plus45_sidecar_root_v1", "current_140_94_precheck_root_v1", "+45 sidecar explicitly preserved mainline"),
        ("plus45_sidecar_root_v1", "stability_185_139_root_v1", "+45 evidence depends on 185/139 stability evidence"),
    ]
    for source_key, target_key, reason in current_edges:
        edges.append(
            {
                "from_v1": str(roots[source_key]),
                "to_v1": str(roots[target_key]),
                "edge_type_v1": "CURRENT_GATE_DEPENDS_ON_EARLIER_ARTIFACT",
                "confidence_v1": "HIGH",
                "reason_v1": reason,
            }
        )
    partial = bool(unresolved)
    return {
        "layer_name": "CLEANUP_OVERVIEW_DEPENDENCY_GRAPH_V1",
        "graph_status_v1": "DEPENDENCY_GRAPH_PARTIAL_REQUIRES_MANUAL_REVIEW" if partial else "DEPENDENCY_GRAPH_INFERRED",
        "nodes_v1": nodes,
        "edges_v1": edges,
        "unresolved_dependencies_v1": unresolved,
        "needed_to_explain_140_94_v1": [
            str(roots["current_140_94_precheck_root_v1"]),
            str(roots["reject_rebuild_root_v1"]),
            str(roots["lane_pack_root_v1"]),
            str(roots["tail_repaired_r5_2_140_94_source_root_v1"]),
        ],
        "needed_to_explain_185_139_comparator_v1": [
            str(roots["stability_185_139_root_v1"]),
            str(roots["best_lane_package_root_v1"]),
            str(roots["lane_pack_root_v1"]),
        ],
        "needed_to_explain_plus45_diagnostic_v1": [
            str(roots["plus45_sidecar_root_v1"]),
            str(roots["student_oof_root_v1"]),
            str(roots["stability_185_139_root_v1"]),
            str(roots["best_lane_package_root_v1"]),
        ],
        "needed_to_explain_wednesday_180_149_v1": wednesday_status.get("artifact_roots_v1", []),
    }


def _classification_rows(artifact_rows: list[dict[str, Any]], repo_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in artifact_rows:
        rows.append(
            {
                "path_v1": row["path_v1"],
                "item_type_v1": "artifact_root",
                "classification_v1": row["candidate_classification_preliminary_v1"],
                "reason_v1": row["classification_reason_v1"],
                "referenced_in_project_state_v1": row["referenced_in_project_state_v1"],
                "referenced_in_decision_log_v1": row["referenced_in_decision_log_v1"],
                "status_v1": row.get("likely_status_v1", ""),
            }
        )
    for row in repo_rows:
        rows.append(
            {
                "path_v1": row["path_v1"],
                "item_type_v1": "repo_file_or_dir",
                "classification_v1": row["candidate_cleanup_class_v1"],
                "reason_v1": row["classification_reason_v1"],
                "referenced_in_project_state_v1": False,
                "referenced_in_decision_log_v1": False,
                "status_v1": "",
            }
        )
    present = {row["classification_v1"] for row in rows}
    synthetic_not_written = []
    for classification in sorted(CLASSIFICATIONS - present):
        synthetic_not_written.append(
            {
                "path_v1": f"CLASSIFICATION_BUCKET_EMPTY::{classification}",
                "item_type_v1": "empty_bucket_marker",
                "classification_v1": classification,
                "reason_v1": "No current scanned item landed in this class; marker is not a cleanup candidate.",
                "referenced_in_project_state_v1": False,
                "referenced_in_decision_log_v1": False,
                "status_v1": "",
            }
        )
    return rows + synthetic_not_written


def _baseline_summary(
    roots: dict[str, Path],
    artifact_rows: list[dict[str, Any]],
    wednesday_status: dict[str, Any],
) -> dict[str, Any]:
    plus45_materialized = roots["plus45_sidecar_root_v1"].exists()
    artifact_by_name = {row["basename_v1"]: row for row in artifact_rows}
    return {
        "layer_name": "CLEANUP_OVERVIEW_CURRENT_BASELINES_LOCK_SUMMARY_V1",
        "mainline_v1": {
            "current_best_v1": "140/94 current best causal baseline",
            "selected_rows_v1": 140,
            "bad_tail_v1": "140 / 94",
            "precision_v1": 1.0,
            "safety_v1": "clean",
            "not_final_live_or_canonical_v1": True,
            "needs_before_adapter_v1": "rule/veto distillation",
            "artifact_root_v1": str(roots["current_140_94_precheck_root_v1"]),
            "final_status_v1": MAINLINE_FINAL_STATUS,
            "next_action_v1": MAINLINE_NEXT_ACTION,
        },
        "reference_comparator_v1": {
            "wednesday_180_149_status_v1": wednesday_status["status_v1"],
            "wednesday_180_149_roots_v1": wednesday_status["artifact_roots_v1"],
            "best_lane_185_139_role_v1": "COMPARATOR_DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
            "best_lane_185_139_root_v1": str(roots["stability_185_139_root_v1"]),
            "coverage_proxy_188_136_root_v1": str(roots["coverage_proxy_188_136_root_v1"]),
            "previous_r5_2_130_86_roots_v1": [
                str(roots["previous_r5_2_130_86_candidate_root_v1"]),
                str(roots["previous_r5_2_130_86_package_root_v1"]),
            ],
        },
        "diagnostic_only_v1": {
            "plus45_rows_v1": "+45 diagnostic-only, not target/filter/threshold objective/adapter input",
            "student_recovered_plus45_v1": "0/45",
            "student_artifact_root_v1": str(roots["student_oof_root_v1"]),
            "membership_oracle_risk_v1": "185/139 depends on membership/coverage/tail-gap boundary risk",
            "best_lane_package_root_v1": str(roots["best_lane_package_root_v1"]),
        },
        "planned_sidecar_v1": {
            "plus45_as_of_feature_gap_shadow_exploration_v1": (
                "MATERIALIZED" if plus45_materialized else "PLANNED_NOT_MATERIALIZED"
            ),
            "artifact_root_v1": str(roots["plus45_sidecar_root_v1"]) if plus45_materialized else None,
            "status_v1": artifact_by_name.get(PLUS45_SIDECAR_ROOT_NAME, {}).get("likely_status_v1", ""),
        },
    }


def _cleanup_plan_rows(classification_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in classification_rows:
        classification = row["classification_v1"]
        path = row["path_v1"]
        if path.startswith("CLASSIFICATION_BUCKET_EMPTY::"):
            continue
        if classification in {CLASS_KEEP_ACTIVE, CLASS_KEEP_REFERENCE, CLASS_KEEP_DIAGNOSTIC, CLASS_KEEP_PLANNED_SIDECAR}:
            action = "KEEP"
            risk = "High if removed before a locked dependency bundle exists."
            preconditions = "None; keep."
            hash_snapshot = True
        elif classification == CLASS_ARCHIVE_COLD:
            action = "ARCHIVE_LATER"
            risk = "Moderate; historical comparison or reproducibility evidence may be lost if archived without manifest."
            preconditions = "Create manifest, dependency graph entry, hash snapshot, and update state/log references if needed."
            hash_snapshot = True
        elif classification == CLASS_DELETE_SAFE:
            action = "DELETE_LATER"
            risk = "Low for cache/temp only, but still requires a separate delete gate."
            preconditions = "Confirm path is cache/temp, not referenced, and tests pass before and after delete gate."
            hash_snapshot = False
        elif classification == CLASS_BLOCKED:
            action = "DO_NOT_TOUCH"
            risk = "High; blocked by known policy or dependency."
            preconditions = "Resolve blocker in a separate review gate."
            hash_snapshot = True
        else:
            action = "MANUAL_REVIEW"
            risk = "Unknown; dependency unclear."
            preconditions = "Manual dependency review and explicit reclassification."
            hash_snapshot = True
        rows.append(
            {
                "path_v1": path,
                "proposed_action_v1": action,
                "reason_v1": row["reason_v1"],
                "dependency_evidence_v1": f"PROJECT_STATE={row.get('referenced_in_project_state_v1')} DECISION_LOG={row.get('referenced_in_decision_log_v1')}",
                "risk_if_removed_v1": risk,
                "required_preconditions_before_cleanup_v1": preconditions,
                "hash_snapshot_should_be_taken_v1": hash_snapshot,
                "project_state_or_decision_log_reference_update_required_v1": bool(
                    row.get("referenced_in_project_state_v1") or row.get("referenced_in_decision_log_v1")
                ),
                "test_must_pass_before_after_cleanup_v1": action in {"ARCHIVE_LATER", "DELETE_LATER", "MANUAL_REVIEW"},
            }
        )
    return rows


def _risk_audit(
    artifact_rows: list[dict[str, Any]],
    repo_rows: list[dict[str, Any]],
    project_scan: dict[str, Any],
    dependency_graph: dict[str, Any],
    wednesday_status: dict[str, Any],
    artifact_base_root: Path,
) -> dict[str, Any]:
    roots = known_roots(artifact_base_root)
    risks: list[dict[str, Any]] = []
    for path in project_scan["missing_referenced_artifact_paths_v1"]:
        risks.append(
            {
                "risk_class_v1": "HIGH_CLEANUP_RISK_DO_NOT_TOUCH",
                "risk_v1": "artifact root referenced but missing",
                "path_v1": path,
            }
        )
    for row in artifact_rows:
        if not row["referenced_in_project_state_v1"] and not row["referenced_in_decision_log_v1"]:
            risks.append(
                {
                    "risk_class_v1": "MODERATE_CLEANUP_RISK_REQUIRES_ARCHIVE_FIRST",
                    "risk_v1": "artifact root present but unreferenced by state/log",
                    "path_v1": row["path_v1"],
                }
            )
        if row["lock_suffix_v1"] and not (row["contains_go_no_go_json_v1"] or row["likely_status_v1"]):
            risks.append(
                {
                    "risk_class_v1": "UNKNOWN_REQUIRES_MANUAL_REVIEW",
                    "risk_v1": "LOCK root missing obvious go/no-go or status",
                    "path_v1": row["path_v1"],
                }
            )
        if not row["contains_manifest_v1"]:
            risks.append(
                {
                    "risk_class_v1": "UNKNOWN_REQUIRES_MANUAL_REVIEW",
                    "risk_v1": "artifact root without manifest marker",
                    "path_v1": row["path_v1"],
                }
            )
        if not row["likely_status_v1"]:
            risks.append(
                {
                    "risk_class_v1": "UNKNOWN_REQUIRES_MANUAL_REVIEW",
                    "risk_v1": "unknown status root",
                    "path_v1": row["path_v1"],
                }
            )
    if not roots["current_140_94_precheck_root_v1"].exists():
        risks.append(
            {
                "risk_class_v1": "HIGH_CLEANUP_RISK_DO_NOT_TOUCH",
                "risk_v1": "current 140/94 artifact missing",
                "path_v1": str(roots["current_140_94_precheck_root_v1"]),
            }
        )
    if roots["plus45_sidecar_root_v1"].exists() and not roots["stability_185_139_root_v1"].exists():
        risks.append(
            {
                "risk_class_v1": "HIGH_CLEANUP_RISK_DO_NOT_TOUCH",
                "risk_v1": "+45 diagnostic evidence depends on missing 185/139 stability root",
                "path_v1": str(roots["stability_185_139_root_v1"]),
            }
        )
    if wednesday_status["status_v1"] != "WEDNESDAY_180_149_REFERENCE_FOUND_IN_SCAN":
        risks.append(
            {
                "risk_class_v1": "UNKNOWN_REQUIRES_MANUAL_REVIEW",
                "risk_v1": "Wednesday 180/149 benchmark root unknown or only partially bound",
                "path_v1": "|".join(wednesday_status.get("artifact_roots_v1", [])),
            }
        )
    implicit_latest_hits = []
    for row in repo_rows:
        if row["file_type_v1"] != "script":
            continue
        text = _read_text(Path(row["path_v1"]))
        if "latest" in text.lower() or ".glob(" in text or ".rglob(" in text:
            implicit_latest_hits.append(row["relative_path_v1"])
    for rel in implicit_latest_hits[:100]:
        risks.append(
            {
                "risk_class_v1": "UNKNOWN_REQUIRES_MANUAL_REVIEW",
                "risk_v1": "script uses latest/glob-like discovery; audit before using for decisions",
                "path_v1": rel,
            }
        )
    gate_counts: dict[str, int] = {}
    for row in artifact_rows:
        gate_counts[row["likely_gate_name_v1"]] = gate_counts.get(row["likely_gate_name_v1"], 0) + 1
    for gate, count in sorted(gate_counts.items()):
        if count > 1:
            risks.append(
                {
                    "risk_class_v1": "UNKNOWN_REQUIRES_MANUAL_REVIEW",
                    "risk_v1": "duplicate-looking roots with same gate name",
                    "path_v1": gate,
                    "count_v1": count,
                }
            )
    cache_rows = [row for row in repo_rows if row["candidate_cleanup_class_v1"] == CLASS_DELETE_SAFE]
    if cache_rows:
        risks.append(
            {
                "risk_class_v1": "LOW_CLEANUP_RISK",
                "risk_v1": "cache/temp files present in repo area",
                "path_v1": f"{len(cache_rows)} repo cache/temp entries",
            }
        )
    return {
        "layer_name": "CLEANUP_OVERVIEW_CLEANUP_RISK_AUDIT_V1",
        "risk_count_v1": len(risks),
        "risks_v1": risks,
        "dependency_graph_status_v1": dependency_graph["graph_status_v1"],
        "current_mainline_depends_on_old_artifact_v1": True,
        "plus45_diagnostic_depends_on_185_139_root_v1": roots["plus45_sidecar_root_v1"].exists(),
        "wednesday_benchmark_status_v1": wednesday_status["status_v1"],
        "cleanup_actions_performed_v1": False,
    }


def _recommendation(
    roots: dict[str, Path],
    dependency_graph: dict[str, Any],
    risk_audit: dict[str, Any],
) -> dict[str, Any]:
    if not roots["current_140_94_precheck_root_v1"].exists():
        status = "CLEANUP_OVERVIEW_BLOCKED_BY_MISSING_CURRENT_BASELINE_ARTIFACT"
        next_action = "RESOLVE_MISSING_REFERENCED_ARTIFACTS_V1"
    elif dependency_graph["graph_status_v1"] == "DEPENDENCY_GRAPH_PARTIAL_REQUIRES_MANUAL_REVIEW":
        status = "CLEANUP_OVERVIEW_FOUND_REFERENCES_REQUIRING_MANUAL_REVIEW"
        next_action = "DEEPEN_ARTIFACT_DEPENDENCY_GRAPH_AUDIT_V1"
    else:
        high_unknowns = [
            row
            for row in risk_audit["risks_v1"]
            if row["risk_class_v1"] in {"HIGH_CLEANUP_RISK_DO_NOT_TOUCH", "UNKNOWN_REQUIRES_MANUAL_REVIEW"}
        ]
        if high_unknowns:
            status = "CLEANUP_OVERVIEW_FOUND_REFERENCES_REQUIRING_MANUAL_REVIEW"
            next_action = "DEEPEN_ARTIFACT_DEPENDENCY_GRAPH_AUDIT_V1"
        else:
            status = "CLEANUP_OVERVIEW_READY_FOR_ARCHIVE_PLAN"
            next_action = "PLAN_ARCHIVE_OUTDATED_RUNS_WITH_MANIFEST_V1"
    return {
        "layer_name": "CLEANUP_OVERVIEW_RECOMMENDATION_V1",
        "status_v1": status,
        "next_recommended_action_v1": next_action,
        "mainline_remains_v1": "140/94 current best causal baseline",
        "mainline_next_action_remains_v1": MAINLINE_NEXT_ACTION,
        "no_cleanup_performed_v1": True,
        "no_archive_performed_v1": True,
        "no_delete_performed_v1": True,
        "no_r6_adapter_package_freeze_promo_live_v1": True,
    }


def _input_scope_manifest(
    repo_root: Path,
    artifact_base_root: Path,
    artifact_root: Path,
    roots: dict[str, Path],
) -> dict[str, Any]:
    return {
        "layer_name": "CLEANUP_OVERVIEW_INPUT_SCOPE_MANIFEST_V1",
        "action_v1": ACTION,
        "repo_root_v1": str(repo_root),
        "artifact_base_root_v1": str(artifact_base_root),
        "artifact_root_v1": str(artifact_root),
        "scan_timestamp_utc_v1": _utc_now(),
        "script_path_v1": str(Path(__file__).resolve()),
        "allowed_writes_v1": [
            "new overview artifact root",
            "overview outputs inside new root",
            "new gate script",
            "new targeted gate test",
            "PROJECT_STATE.md summary update",
            "DECISION_LOG.md summary update",
        ],
        "prohibited_writes_v1": [
            "delete",
            "move",
            "archive",
            "rename",
            "modify existing artifact roots",
            "R6 run",
            "adapter build",
            "package build",
            "freeze",
            "promo",
            "live",
            "Optuna",
            "model training",
            "candidate selection materialization",
        ],
        "current_known_artifact_roots_v1": {key: str(path) for key, path in roots.items()},
        "current_mainline_status_v1": {
            "current_best_v1": "140/94",
            "selected_rows_v1": 140,
            "bad_tail_v1": "140 / 94",
            "precision_v1": 1.0,
            "safety_v1": "clean",
            "final_status_v1": MAINLINE_FINAL_STATUS,
        },
        "mainline_next_action_v1": MAINLINE_NEXT_ACTION,
        "dry_run_status_v1": "DRY_RUN_ONLY",
        "cleanup_performed_v1": False,
        "archive_performed_v1": False,
        "delete_performed_v1": False,
    }


def validate_required_outputs(root: Path) -> bool:
    missing = [name for name in REQUIRED_OUTPUTS if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"CLEANUP_OVERVIEW_REQUIRED_OUTPUTS_MISSING: {missing}")
    return True


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def _snapshot_mtimes(paths: Iterable[Path]) -> dict[str, int | None]:
    snapshot: dict[str, int | None] = {}
    for path in paths:
        try:
            snapshot[str(path)] = int(path.stat().st_mtime_ns)
        except OSError:
            snapshot[str(path)] = None
    return snapshot


def _write_standard_reports(
    artifact_root: Path,
    manifest: dict[str, Any],
    project_scan: dict[str, Any],
    artifact_rows: list[dict[str, Any]],
    repo_rows: list[dict[str, Any]],
    dependency_graph: dict[str, Any],
    classification_rows: list[dict[str, Any]],
    baseline_summary: dict[str, Any],
    cleanup_plan_rows: list[dict[str, Any]],
    risk_audit: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_json(artifact_root / "cleanup_overview_input_scope_manifest_v1.json", manifest)
    _write_report(
        artifact_root / "cleanup_overview_input_scope_manifest_v1.md",
        [
            "# Cleanup Overview Input Scope Manifest V1",
            "",
            f"- Action: `{ACTION}`",
            f"- Repo root: `{manifest['repo_root_v1']}`",
            f"- Artifact base root: `{manifest['artifact_base_root_v1']}`",
            f"- Artifact root: `{manifest['artifact_root_v1']}`",
            f"- Mainline: `{manifest['current_mainline_status_v1']['current_best_v1']}`",
            f"- Mainline next action: `{MAINLINE_NEXT_ACTION}`",
            "- Dry-run only: `true`",
            "- Cleanup performed: `false`",
        ],
    )
    _write_json(artifact_root / "cleanup_overview_project_state_reference_scan_v1.json", project_scan)
    _write_report(
        artifact_root / "cleanup_overview_project_state_reference_scan_v1.md",
        [
            "# Cleanup Overview Project State Reference Scan V1",
            "",
            f"- Artifact path references: `{len(project_scan['artifact_paths_v1'])}`",
            f"- Missing referenced artifact paths: `{len(project_scan['missing_referenced_artifact_paths_v1'])}`",
            f"- 140/94 references: `{project_scan['keyword_references_v1']['baseline_140_94_references_v1']}`",
            f"- 185/139 references: `{project_scan['keyword_references_v1']['best_lane_185_139_references_v1']}`",
            f"- +45 references: `{project_scan['keyword_references_v1']['plus45_references_v1']}`",
            f"- Wednesday 180/149 references: `{project_scan['keyword_references_v1']['wednesday_180_149_references_v1']}`",
        ],
    )
    _write_rows(artifact_root / "cleanup_overview_artifact_inventory_v1.csv", artifact_rows)
    _write_json(artifact_root / "cleanup_overview_artifact_inventory_v1.json", {"rows_v1": artifact_rows})
    _write_report(
        artifact_root / "cleanup_overview_artifact_inventory_v1.md",
        [
            "# Cleanup Overview Artifact Inventory V1",
            "",
            f"- Artifact directories scanned: `{len(artifact_rows)}`",
            f"- KEEP_ACTIVE: `{sum(row['candidate_classification_preliminary_v1'] == CLASS_KEEP_ACTIVE for row in artifact_rows)}`",
            f"- KEEP_REFERENCE: `{sum(row['candidate_classification_preliminary_v1'] == CLASS_KEEP_REFERENCE for row in artifact_rows)}`",
            f"- KEEP_DIAGNOSTIC: `{sum(row['candidate_classification_preliminary_v1'] == CLASS_KEEP_DIAGNOSTIC for row in artifact_rows)}`",
            f"- ARCHIVE_COLD_CANDIDATE: `{sum(row['candidate_classification_preliminary_v1'] == CLASS_ARCHIVE_COLD for row in artifact_rows)}`",
            f"- UNKNOWN_DO_NOT_TOUCH: `{sum(row['candidate_classification_preliminary_v1'] == CLASS_UNKNOWN for row in artifact_rows)}`",
        ],
    )
    _write_rows(artifact_root / "cleanup_overview_repo_file_inventory_v1.csv", repo_rows)
    _write_json(artifact_root / "cleanup_overview_repo_file_inventory_v1.json", {"rows_v1": repo_rows})
    _write_report(
        artifact_root / "cleanup_overview_repo_file_inventory_v1.md",
        [
            "# Cleanup Overview Repo File Inventory V1",
            "",
            f"- Repo files/dirs scanned: `{len(repo_rows)}`",
            f"- Files referencing artifact roots: `{sum(bool(row['references_artifact_roots_v1']) for row in repo_rows)}`",
            f"- Cache/temp DELETE_SAFE candidates: `{sum(row['candidate_cleanup_class_v1'] == CLASS_DELETE_SAFE for row in repo_rows)}`",
            "- No repo cleanup action was performed.",
        ],
    )
    _write_json(artifact_root / "cleanup_overview_dependency_graph_v1.json", dependency_graph)
    _write_report(
        artifact_root / "cleanup_overview_dependency_graph_v1.md",
        [
            "# Cleanup Overview Dependency Graph V1",
            "",
            f"- Graph status: `{dependency_graph['graph_status_v1']}`",
            f"- Nodes: `{len(dependency_graph['nodes_v1'])}`",
            f"- Edges: `{len(dependency_graph['edges_v1'])}`",
            f"- Unresolved dependencies: `{len(dependency_graph['unresolved_dependencies_v1'])}`",
            "",
            "Required marker: `DEPENDENCY_GRAPH_PARTIAL_REQUIRES_MANUAL_REVIEW`"
            if dependency_graph["graph_status_v1"] == "DEPENDENCY_GRAPH_PARTIAL_REQUIRES_MANUAL_REVIEW"
            else "Required marker: `DEPENDENCY_GRAPH_INFERRED`",
        ],
    )
    _write_rows(artifact_root / "cleanup_overview_classification_v1.csv", classification_rows)
    _write_json(artifact_root / "cleanup_overview_classification_v1.json", {"rows_v1": classification_rows})
    counts = {
        classification: sum(row["classification_v1"] == classification for row in classification_rows)
        for classification in sorted(CLASSIFICATIONS)
    }
    _write_report(
        artifact_root / "cleanup_overview_classification_v1.md",
        ["# Cleanup Overview Classification V1", ""] + [f"- {key}: `{value}`" for key, value in counts.items()],
    )
    _write_json(artifact_root / "cleanup_overview_current_baselines_lock_summary_v1.json", baseline_summary)
    _write_report(
        artifact_root / "cleanup_overview_current_baselines_lock_summary_v1.md",
        [
            "# Cleanup Overview Current Baselines Lock Summary V1",
            "",
            "- Mainline: `140/94 current best causal baseline`, not final/live/canonical.",
            f"- Mainline next action: `{MAINLINE_NEXT_ACTION}`.",
            "- Reference/comparator: Wednesday 180/149 if found, 185/139, 188/136 coverage proxy, previous R5.2 130/86.",
            "- Diagnostic-only: `+45`, student recovered `0/45`, and 185/139 membership/oracle risk.",
            f"- Wednesday status: `{baseline_summary['reference_comparator_v1']['wednesday_180_149_status_v1']}`.",
        ],
    )
    _write_rows(artifact_root / "cleanup_overview_future_cleanup_plan_dry_run_v1.csv", cleanup_plan_rows)
    _write_json(artifact_root / "cleanup_overview_future_cleanup_plan_dry_run_v1.json", {"rows_v1": cleanup_plan_rows})
    _write_report(
        artifact_root / "cleanup_overview_future_cleanup_plan_dry_run_v1.md",
        [
            "# Cleanup Overview Future Cleanup Plan Dry Run V1",
            "",
            "- No cleanup action was performed.",
            f"- KEEP rows: `{sum(row['proposed_action_v1'] == 'KEEP' for row in cleanup_plan_rows)}`",
            f"- ARCHIVE_LATER rows: `{sum(row['proposed_action_v1'] == 'ARCHIVE_LATER' for row in cleanup_plan_rows)}`",
            f"- DELETE_LATER rows: `{sum(row['proposed_action_v1'] == 'DELETE_LATER' for row in cleanup_plan_rows)}`",
            f"- MANUAL_REVIEW rows: `{sum(row['proposed_action_v1'] == 'MANUAL_REVIEW' for row in cleanup_plan_rows)}`",
        ],
    )
    _write_json(artifact_root / "cleanup_overview_cleanup_risk_audit_v1.json", risk_audit)
    risk_counts: dict[str, int] = {}
    for row in risk_audit["risks_v1"]:
        key = row["risk_class_v1"]
        risk_counts[key] = risk_counts.get(key, 0) + 1
    _write_report(
        artifact_root / "cleanup_overview_cleanup_risk_audit_v1.md",
        ["# Cleanup Overview Cleanup Risk Audit V1", "", f"- Risk count: `{risk_audit['risk_count_v1']}`"]
        + [f"- {key}: `{value}`" for key, value in sorted(risk_counts.items())],
    )
    _write_json(artifact_root / "cleanup_overview_recommendation_v1.json", recommendation)
    _write_report(
        artifact_root / "cleanup_overview_recommendation_v1.md",
        [
            "# Cleanup Overview Recommendation V1",
            "",
            f"- Final status: `{recommendation['status_v1']}`",
            f"- Next recommended action: `{recommendation['next_recommended_action_v1']}`",
            f"- Mainline remains: `{recommendation['mainline_remains_v1']}`",
            f"- Mainline next action remains: `{MAINLINE_NEXT_ACTION}`",
            "- No deletion, archive, R6, adapter, package, freeze, promo, live, Optuna, training, or candidate materialization was performed.",
        ],
    )


def materialize(
    artifact_root: Path | None = None,
    *,
    repo_root: Path | None = None,
    artifact_base_root: Path = DEFAULT_ARTIFACT_BASE_ROOT,
    enforce_default_artifact_base_root: bool = True,
) -> dict[str, Any]:
    repo = repo_root or _repo_root_from_file()
    if enforce_default_artifact_base_root:
        validate_explicit_artifact_base_root(artifact_base_root)
    validate_dry_run_only()
    validate_script_has_no_destructive_calls()
    no_forbidden = validate_no_forbidden_actions()
    if no_forbidden["status_v1"] != "PASS":
        raise RuntimeError(f"FORBIDDEN_ACTION_FLAGS_SET: {no_forbidden}")
    roots = known_roots(artifact_base_root)
    validate_current_roots_are_explicit(roots.values())
    target = artifact_root or artifact_base_root / f"{ACTION}_{_stamp()}_LOCK"
    if target.exists():
        raise RuntimeError(f"REFUSING_TO_OVERWRITE_EXISTING_ARTIFACT_ROOT: {target}")
    watched_before = _snapshot_mtimes(path for path in roots.values() if path.exists())
    target.mkdir(parents=True, exist_ok=False)

    manifest = _input_scope_manifest(repo, artifact_base_root, target, roots)
    project_scan = _scan_project_references(repo, artifact_base_root)
    project_state_text = _read_text(repo / "PROJECT_STATE.md", max_bytes=5_000_000)
    decision_log_text = _read_text(repo / "DECISION_LOG.md", max_bytes=5_000_000)
    artifact_rows = _artifact_inventory(artifact_base_root, repo, project_state_text, decision_log_text)
    repo_rows = _repo_file_inventory(repo)
    wednesday_status = _find_wednesday_reference_status(artifact_rows, project_scan)
    dependency_graph = _dependency_graph(artifact_rows, repo_rows, project_scan, wednesday_status, artifact_base_root)
    classification_rows = _classification_rows(artifact_rows, repo_rows)
    baseline_summary = _baseline_summary(roots, artifact_rows, wednesday_status)
    cleanup_plan_rows = _cleanup_plan_rows(classification_rows)
    risk_audit = _risk_audit(artifact_rows, repo_rows, project_scan, dependency_graph, wednesday_status, artifact_base_root)
    recommendation = _recommendation(roots, dependency_graph, risk_audit)
    validate_final_status(recommendation["status_v1"], recommendation["next_recommended_action_v1"])

    _write_standard_reports(
        target,
        manifest,
        project_scan,
        artifact_rows,
        repo_rows,
        dependency_graph,
        classification_rows,
        baseline_summary,
        cleanup_plan_rows,
        risk_audit,
        recommendation,
    )
    watched_after = _snapshot_mtimes(Path(path) for path in watched_before)
    modified_existing_roots = [
        path for path, before_mtime in watched_before.items() if before_mtime is not None and watched_after.get(path) != before_mtime
    ]
    go_no_go = {
        "layer_name": "CLEANUP_OVERVIEW_CURRENT_BASELINES_AND_OUTDATED_RUNS_GO_NO_GO_V1",
        "status_v1": recommendation["status_v1"],
        "next_recommended_action_v1": recommendation["next_recommended_action_v1"],
        "artifact_root_v1": str(target),
        "dry_run_only_v1": True,
        "cleanup_actions_performed_v1": False,
        "delete_performed_v1": False,
        "archive_performed_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "package_built_v1": False,
        "freeze_performed_v1": False,
        "promo_performed_v1": False,
        "live_performed_v1": False,
        "optuna_run_v1": False,
        "model_training_run_v1": False,
        "candidate_selection_materialized_v1": False,
        "existing_artifact_roots_modified_v1": bool(modified_existing_roots),
        "modified_existing_artifact_roots_v1": modified_existing_roots,
        "current_140_94_artifact_root_v1": str(roots["current_140_94_precheck_root_v1"]),
        "mainline_next_action_v1": MAINLINE_NEXT_ACTION,
        "wednesday_180_149_reference_status_v1": wednesday_status["status_v1"],
        "dependency_graph_status_v1": dependency_graph["graph_status_v1"],
        "required_outputs_present_v1": True,
        "python_manifest_v1": {
            "python_executable_v1": sys.executable,
            "python_version_v1": sys.version,
            "platform_v1": platform.platform(),
        },
    }
    if modified_existing_roots:
        go_no_go["status_v1"] = "CLEANUP_OVERVIEW_BLOCKED_BY_UNCLEAR_DEPENDENCY_GRAPH"
        go_no_go["next_recommended_action_v1"] = "DEEPEN_ARTIFACT_DEPENDENCY_GRAPH_AUDIT_V1"
    _write_json(target / "cleanup_overview_current_baselines_and_outdated_runs_go_no_go_v1.json", go_no_go)
    validate_required_outputs(target)
    return {
        "artifact_root_v1": str(target),
        "status_v1": go_no_go["status_v1"],
        "next_recommended_action_v1": go_no_go["next_recommended_action_v1"],
        "wednesday_180_149_reference_status_v1": wednesday_status["status_v1"],
        "dependency_graph_status_v1": dependency_graph["graph_status_v1"],
        "no_cleanup_performed_v1": True,
        "existing_artifact_roots_modified_v1": bool(modified_existing_roots),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=ACTION)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--artifact-base-root", type=Path, default=DEFAULT_ARTIFACT_BASE_ROOT)
    args = parser.parse_args(argv)
    result = materialize(
        args.artifact_root,
        repo_root=args.repo_root,
        artifact_base_root=args.artifact_base_root,
        enforce_default_artifact_base_root=args.artifact_base_root == DEFAULT_ARTIFACT_BASE_ROOT,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
