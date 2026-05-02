from __future__ import annotations

import ast
import json
from pathlib import Path

from gx1.scripts import materialize_cleanup_overview_current_baselines_and_outdated_runs_v1 as gate


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_artifact_root(base: Path, name: str, status: str = "REFERENCE_STATUS") -> Path:
    root = base / name
    root.mkdir(parents=True)
    _write_json(root / "summary_v1.json", {"status_v1": status, "next_action_v1": "NEXT_ACTION_V1"})
    _write_json(root / "manifest_v1.json", {"root_v1": str(root)})
    _write_json(root / "example_go_no_go_v1.json", {"status_v1": status})
    return root


def test_script_is_dry_run_only_and_has_no_destructive_calls() -> None:
    assert gate.DRY_RUN_ONLY is True
    script_path = Path(gate.__file__).resolve()
    assert gate.validate_script_has_no_destructive_calls(script_path)
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    forbidden_imports = {"shutil"}
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not (imported & forbidden_imports)


def test_explicit_artifact_base_root_and_current_140_root_are_listed() -> None:
    assert gate.DEFAULT_ARTIFACT_BASE_ROOT == Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
    roots = gate.known_roots()
    assert roots["current_140_94_precheck_root_v1"] == gate.DEFAULT_ARTIFACT_BASE_ROOT / gate.CURRENT_140_ROOT_NAME
    assert gate.validate_current_roots_are_explicit(roots.values())


def test_core_artifact_classifications_protect_mainline_and_diagnostics() -> None:
    base = gate.DEFAULT_ARTIFACT_BASE_ROOT
    roots = gate.known_roots(base)
    assert gate.classify_artifact_root(roots["current_140_94_precheck_root_v1"], base)[0] == gate.CLASS_KEEP_ACTIVE
    assert gate.classify_artifact_root(roots["stability_185_139_root_v1"], base)[0] in {
        gate.CLASS_KEEP_REFERENCE,
        gate.CLASS_KEEP_DIAGNOSTIC,
    }
    assert gate.classify_artifact_root(roots["stability_185_139_root_v1"], base)[0] != gate.CLASS_KEEP_ACTIVE
    assert gate.classify_artifact_root(roots["plus45_sidecar_root_v1"], base)[0] in {
        gate.CLASS_KEEP_DIAGNOSTIC,
        gate.CLASS_KEEP_PLANNED_SIDECAR,
    }
    assert gate.classify_artifact_root(roots["wednesday_snapshot_root_v1"], base)[0] == gate.CLASS_KEEP_REFERENCE


def test_no_forbidden_gate_actions_are_default() -> None:
    assert gate.validate_no_forbidden_actions()["status_v1"] == "PASS"
    blocked = gate.validate_no_forbidden_actions(
        cleanup_action=True,
        r6=True,
        adapter=True,
        package=True,
        freeze=True,
        promo=True,
        live=True,
        optuna=True,
        model_training=True,
        selection_materialized=True,
    )
    assert blocked["status_v1"] == "FAIL"
    assert "CLEANUP_ACTION_FORBIDDEN" in blocked["failures_v1"]
    assert "R6_FORBIDDEN" in blocked["failures_v1"]
    assert "ADAPTER_BUILD_FORBIDDEN" in blocked["failures_v1"]
    assert "PACKAGE_BUILD_FORBIDDEN" in blocked["failures_v1"]
    assert "FREEZE_FORBIDDEN" in blocked["failures_v1"]
    assert "PROMO_FORBIDDEN" in blocked["failures_v1"]
    assert "LIVE_FORBIDDEN" in blocked["failures_v1"]
    assert "OPTUNA_FORBIDDEN" in blocked["failures_v1"]
    assert "MODEL_TRAINING_FORBIDDEN" in blocked["failures_v1"]
    assert "CANDIDATE_SELECTION_MATERIALIZATION_FORBIDDEN" in blocked["failures_v1"]


def test_materializer_writes_required_outputs_valid_go_no_go_and_no_cleanup(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    artifact_base = tmp_path / "truth_e2e_sanity"
    repo_root.mkdir()
    artifact_base.mkdir()
    (repo_root / "gx1/scripts").mkdir(parents=True)
    (repo_root / "tests").mkdir()
    _write_text(repo_root / "AGENTS.md", "Never run R6. Keep historical artifacts.\n")
    default_current = gate.DEFAULT_ARTIFACT_BASE_ROOT / gate.CURRENT_140_ROOT_NAME
    _write_text(
        repo_root / "PROJECT_STATE.md",
        "\n".join(
            [
                f"Latest 140 root `{default_current}`",
                "`140_94_CAUSAL_BASELINE_NEEDS_RULE_DISTILLATION_BEFORE_ADAPTER`",
                "`DISTILL_140_94_CAUSAL_BASELINE_TO_RULES_AND_VETOES_V1`",
                "185/139 remains comparator. +45 remains diagnostic-only.",
            ]
        ),
    )
    _write_text(
        repo_root / "DECISION_LOG.md",
        "\n".join(
            [
                "Wednesday R6 stack produced 180/149.",
                "R6, adapter build, package build, freeze, promo, and live were not run.",
            ]
        ),
    )
    _write_text(
        repo_root / "gx1/scripts/materialize_cleanup_overview_current_baselines_and_outdated_runs_v1.py",
        "ACTION = 'CLEANUP_OVERVIEW_CURRENT_BASELINES_AND_OUTDATED_RUNS_V1'\n",
    )
    _write_text(repo_root / "tests/test_cleanup_overview_current_baselines_and_outdated_runs_v1.py", "def test_x(): pass\n")
    (repo_root / "gx1/scripts/__pycache__").mkdir(parents=True)
    _write_text(repo_root / "gx1/scripts/__pycache__/x.pyc", "cache\n")

    roots = gate.known_roots(artifact_base)
    for key, root in roots.items():
        status = "REFERENCE_STATUS"
        if key == "current_140_94_precheck_root_v1":
            status = gate.MAINLINE_FINAL_STATUS
        elif key == "stability_185_139_root_v1":
            status = "BEST_LANE_SIGNAL_STRONG_BUT_MEMBERSHIP_ONLY_NOT_R6_READY"
        elif key == "plus45_sidecar_root_v1":
            status = "PLUS45_SHADOW_FOUND_ONLY_MEMBERSHIP_OR_COVERAGE_DEPENDENCY"
        _make_artifact_root(artifact_base, root.name, status=status)
    _write_text(
        roots["wednesday_skeleton_root_v1"] / "monday_vs_wednesday_skeleton_delta_matrix_v1.json",
        '{"wednesday_v1": "R5.2 plus R6 stack produced 180/149"}\n',
    )
    old_root = _make_artifact_root(artifact_base, "OLD_EXPERIMENT_V1_20260401T010101Z_LOCK", "OLD_STATUS")
    unknown_root = artifact_base / "loose_unknown_folder"
    unknown_root.mkdir()
    before = {str(path): path.stat().st_mtime_ns for path in roots.values() if path.exists()}

    artifact_root = artifact_base / "CLEANUP_OVERVIEW_CURRENT_BASELINES_AND_OUTDATED_RUNS_V1_20260428T000000Z_LOCK"
    result = gate.materialize(
        artifact_root,
        repo_root=repo_root,
        artifact_base_root=artifact_base,
        enforce_default_artifact_base_root=False,
    )

    assert result["no_cleanup_performed_v1"] is True
    assert result["existing_artifact_roots_modified_v1"] is False
    after = {str(path): path.stat().st_mtime_ns for path in roots.values() if path.exists()}
    assert after == before
    for name in gate.REQUIRED_OUTPUTS:
        assert (artifact_root / name).exists(), name

    go = json.loads((artifact_root / "cleanup_overview_current_baselines_and_outdated_runs_go_no_go_v1.json").read_text())
    assert go["status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert go["next_recommended_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert go["cleanup_actions_performed_v1"] is False
    assert go["delete_performed_v1"] is False
    assert go["archive_performed_v1"] is False
    assert go["r6_run_v1"] is False
    assert go["adapter_built_v1"] is False
    assert go["package_built_v1"] is False
    assert go["freeze_performed_v1"] is False
    assert go["promo_performed_v1"] is False
    assert go["live_performed_v1"] is False
    assert go["current_140_94_artifact_root_v1"] == str(roots["current_140_94_precheck_root_v1"])
    assert go["wednesday_180_149_reference_status_v1"] == "WEDNESDAY_180_149_REFERENCE_FOUND_IN_SCAN"

    classification = json.loads((artifact_root / "cleanup_overview_classification_v1.json").read_text())
    classes = {row["classification_v1"] for row in classification["rows_v1"]}
    for expected in {
        gate.CLASS_KEEP_ACTIVE,
        gate.CLASS_KEEP_REFERENCE,
        gate.CLASS_KEEP_DIAGNOSTIC,
        gate.CLASS_ARCHIVE_COLD,
        gate.CLASS_DELETE_SAFE,
        gate.CLASS_UNKNOWN,
    }:
        assert expected in classes
    by_path = {row["path_v1"]: row["classification_v1"] for row in classification["rows_v1"]}
    assert by_path[str(roots["current_140_94_precheck_root_v1"])] == gate.CLASS_KEEP_ACTIVE
    assert by_path[str(roots["stability_185_139_root_v1"])] != gate.CLASS_KEEP_ACTIVE
    assert by_path[str(roots["stability_185_139_root_v1"])] == gate.CLASS_KEEP_DIAGNOSTIC
    assert by_path[str(roots["plus45_sidecar_root_v1"])] == gate.CLASS_KEEP_PLANNED_SIDECAR
    assert by_path[str(old_root)] == gate.CLASS_ARCHIVE_COLD
    assert by_path[str(unknown_root)] == gate.CLASS_UNKNOWN
