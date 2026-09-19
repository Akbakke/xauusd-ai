import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


# One truth: test the checked-out tree these tests live in (worktrees
# included), never a hardcoded absolute clone path.
REPO = Path(__file__).resolve().parents[1]
HANDOVER = REPO / "HANDOVER_XAU_DIRECTION_REPAIR_20260714.md"
HANDOVER_VIEWER = REPO / "scripts/gx1_handover.sh"
CONTROL = REPO / "scripts/entry_next_edge_control.sh"
LAUNCH_STATE = REPO / "PROJECT_STATE_xau_direction_launch.json"
AUTHORITY_PATHS = (
    REPO / "AGENTS.md",
    REPO / "CLAUDE.md",
    REPO / "GX1_RULES.md",
    REPO / "README.md",
    REPO / "SYSTEM_MAP.md",
    REPO / "passord.md",
    HANDOVER,
    REPO / "docs/CURRENT_AUDIT_STATUS_20260828.md",
    REPO / "docs/CURRENT_CLOUD_TRAINING_STATUS_20260908.md",
    REPO / "docs/CURRENT_HANDOFF_20260903.md",
    REPO / "docs/REPO_CLEANUP_CANDIDATES_20260903.md",
    REPO / "docs/PREMIERE_CODE_REVIEW_20260905.md",
    REPO / "docs/PRETRAIN_READINESS_REPAIR_20260906.md",
    REPO / "docs/PRETRAIN_ARCHITECTURE_REVIEW_20260906.md",
    REPO / "docs/PRETRAIN_DATA_REVIEW_20260906.md",
    REPO / "docs/OFFLINE_CHAMPION_CHALLENGER_V1.md",
    REPO / "docs/DATA_CONTRACT.md",
    REPO / "docs/ATTENDED_STAGED_PREFLIGHT_DESIGN_20260823.md",
    REPO / "docs/CANONICAL_HOST_GPU_TELEMETRY_BRIDGE_CONTRACT.md",
    REPO / "docs/CANDIDATE_THROUGHPUT_DECISION_20260830.md",
    REPO / "docs/V8_CANDIDATE_HOST_HANG_INCIDENT_20260901.md",
    # 3c84bec9 committed this review doc without extending the authority
    # fingerprint; covered here so no tracked markdown escapes the fingerprint.
    REPO / "docs/FEATURE_VALUE_REVIEW_20260813.md",
    REPO / "docs/INDICATOR_FIDELITY_AUDIT_20260813.md",
    REPO / "docs/GIT_WORKTREE_POLICY.md",
    REPO / "docs/POST_BUILD_INTEGRITY_GATE_20260825.md",
    REPO / "docs/PREREGISTERED_DIRECTION_TEST_20260820.md",
    REPO / "docs/RECIPE_DECISION_DRAFT_20260808.md",
    REPO / "docs/V29_EVENT_SURFACE_DESIGN_20260811.md",
    REPO / "docs/TRAIN_WINDOW_WIDENING_20260819.md",
    REPO / "docs/TRAINING_EFFICIENCY_REVIEW_20260908.md",
    REPO / "PROJECT_STATE_xau_direction_launch.json",
)

RETAINED_CONTROL_ROUTES = {
    "handover",
    "model-native-state",
    "model-native-state-selftest",
    "model-native-native-m5-source",
    "model-native-native-m1-source",
    "model-native-canonical-pair",
    "model-native-fit-volatility-squeeze-artifacts",
    "model-native-m1-enriched-frame",
    "model-native-m5-enriched-frame",
    "model-native-m5-source-frame",
    "model-native-current-source-cascade-proof",
    "model-native-m1-feature-base",
    "model-native-m5-feature-base",
    "model-native-cross-surface-overlap",
    "model-native-feature-surface-liveness",
    "model-native-rebuild-preflight",
    "model-native-post-rebuild-readiness",
    "model-native-foundation-feature-audit",
    "model-native-foundation-target-audit",
    "model-native-specialist-feature-audit",
    "model-native-adoption-candidate",
    "model-native-smoke-manifest",
    "model-native-smoke-readiness",
    "model-native-trainability-readiness",
    "model-native-pretest-trainability-readiness",
    "model-native-execution-causality-audit",
    "model-native-train-recipe-audit",
    "model-native-smoke-bundle-audit",
    "model-native-feature-usefulness",
    "model-native-candidate-readiness",
    "model-native-selective-edge",
    "model-native-seed-stability",
    "model-native-smoke-train",
    "model-native-trade-path-metrics",
}


def _assert_explicit_review_hold(result: subprocess.CompletedProcess) -> bool:
    """Live integration checks must assert a recorded BLOCK, not demand GREEN.

    Keep the normal ready-state rendering checks below for a future successor;
    during a semantic rebuild hold, any successful handover is a regression.
    """
    state = json.loads(LAUNCH_STATE.read_text())
    if "pretraining_review_hold" not in state:
        return False
    hold = state["pretraining_review_hold"]
    assert hold["decision"] == "BLOCK"
    assert hold["activation_authority"] is False
    assert result.returncode == 2
    assert "decision: BLOCK" in result.stdout
    assert "pretraining_review_hold: ACTIVE" in result.stdout
    assert f"blocker: {hold['reason']}" in result.stdout
    assert "historical_recipe_and_gate: RETAINED_NOT_CURRENT_TRAINING_AUTHORITY" in result.stdout
    assert "cuda_authority: NONE" in result.stdout
    assert "test_paper_live_authority: NONE" in result.stdout
    assert "GATE_READY" not in result.stdout
    assert "FATAL: pretraining review hold blocks launch" in result.stderr
    return True


@pytest.mark.parametrize("state", [{}, {"pretraining_review_hold": None}, {"pretraining_review_hold": {}}])
def test_handover_hold_checker_distinguishes_absent_from_malformed(tmp_path, state):
    # Execute the actual small shell-embedded checker, without resolving any
    # production dataset or manufacturing a ready training evidence chain.
    source = HANDOVER_VIEWER.read_text()
    checker = source.split(
        'if ! "$PY" - "$LAUNCH_STATE" "$mode" <<\'PY\'\n', 1
    )[1].split("\nPY\nthen\n", 1)[0]
    state_path = tmp_path / "launch.json"
    state_path.write_text(json.dumps(state))
    result = subprocess.run(
        [sys.executable, "-c", checker, str(state_path), "check"],
        text=True, capture_output=True, check=False,
    )
    if "pretraining_review_hold" not in state:
        assert result.returncode == 0
    else:
        assert result.returncode != 0
        assert "hold is malformed" in result.stderr
    assert result.stdout == ""


def _source_only_repo(tmp_path: Path) -> Path:
    """Build a source-only checkout whose historical evidence cannot be read."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts/gx1_handover.sh").write_text(HANDOVER_VIEWER.read_text())
    # A tracked package file prevents git from collapsing its ignored cache
    # to `!! gx1/`; the real checkout has tracked sources in this directory.
    (repo / "gx1").mkdir()
    (repo / "gx1/__init__.py").write_text("")
    for authority in AUTHORITY_PATHS:
        path = repo / authority.relative_to(REPO)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source authority\n")
    historical = tmp_path / "UNOPENED_TEST_AND_HISTORY"
    os.mkfifo(historical)
    state = {
        "reviewed_local_runtime_exclusions": {
            "schema_version": "gx1_reviewed_local_runtime_exclusions_v1",
            "paths": [".claude/worktrees/", ".env", ".venv/"],
        },
        "pretraining_review_hold": {
            "schema_version": "gx1_pretraining_review_hold_v1",
            "decision": "BLOCK",
            "reason": "fixture requires rebuilt targets",
            "activation_authority": False,
            "report_path": "docs/PREMIERE_CODE_REVIEW_20260905.md",
        },
        "current_pair_manifest": str(historical),
        "active_candidate_training_session": {"session_dir": str(historical)},
        "current_source_technical_recipe": {"recipe_path": str(historical)},
    }
    (repo / LAUNCH_STATE.name).write_text(json.dumps(state, sort_keys=True))
    (repo / ".gitignore").write_text(
        ".venv/\n.env\n.claude/worktrees/\n*.ignored\n"
        "__pycache__/\n.pytest_cache/\n.ruff_cache/\n"
    )
    (repo / ".venv/bin").mkdir(parents=True)
    (repo / ".venv/bin/python").symlink_to(Path(sys.executable).resolve())
    (repo / ".venv/pyvenv.cfg").write_text(
        f"home = {Path(sys.executable).resolve().parent}\n"
        "include-system-site-packages = false\n"
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null",
         "-c", "user.name=GX1 fixture", "-c", "user.email=fixture@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "-qm", "source-only fixture"],
        check=True,
    )
    return repo


def _run_source_only(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(repo / "scripts/gx1_handover.sh"), "--source-only"],
        cwd=repo, text=True, capture_output=True, check=False, timeout=15,
    )


def test_source_only_handover_reuses_hygiene_without_opening_held_history(tmp_path):
    repo = _source_only_repo(tmp_path)
    result = _run_source_only(repo)
    assert result.returncode == 0, result.stderr
    assert "decision: PASS_SOURCE_HYGIENE_ONLY" in result.stdout
    assert "source_identity_gate: READY_CLEAN_WORKTREE__REVIEWED_LOCAL_EXCLUSIONS" in result.stdout
    assert "changed_path_count: 0" in result.stdout
    assert "unexpected_ignored_path_count: 0" in result.stdout
    assert "historical_artifact_access: NONE" in result.stdout
    assert "dependency_readiness: SEPARATE_REBUILD_DEPENDENCY_PREFLIGHT_REQUIRED" in result.stdout
    for authority in ("training", "cuda", "test_paper_live"):
        assert f"{authority}_authority: NONE" in result.stdout
    # A source-only PASS must not lift the existing semantic launch hold.
    blocked = subprocess.run(
        ["bash", str(repo / "scripts/gx1_handover.sh"), "--check"],
        cwd=repo, text=True, capture_output=True, check=False, timeout=15,
    )
    assert blocked.returncode == 2
    assert "pretraining_review_hold: ACTIVE" in blocked.stdout
    assert "decision: BLOCK" in blocked.stdout
    assert "pretraining review hold blocks launch" in blocked.stderr
    source = HANDOVER_VIEWER.read_text()
    assert source.count("def reviewed_ignored_path(") == 1


@pytest.mark.parametrize("pollution", ["dirty", "ignored", "unsafe-env", "unregistered-worktree", "prunable"])
def test_source_only_handover_fails_closed_on_source_hygiene(tmp_path, pollution):
    repo = _source_only_repo(tmp_path)
    if pollution == "dirty":
        (repo / "README.md").write_text("modified source\n")
    elif pollution == "ignored":
        (repo / "unreviewed.ignored").write_text("not an allowed cache\n")
    elif pollution == "unsafe-env":
        path = repo / ".env"
        path.write_text("FIXTURE_ONLY=true\n")
        path.chmod(0o644)
    elif pollution == "unregistered-worktree":
        (repo / ".claude/worktrees/unknown").mkdir(parents=True)
    else:
        worktree = tmp_path / "registered-worktree"
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "--detach", str(worktree), "HEAD"],
            check=True, capture_output=True,
        )
        worktree.rename(tmp_path / "moved-worktree")
    result = _run_source_only(repo)
    assert result.returncode != 0
    assert "PASS_SOURCE_HYGIENE_ONLY" not in result.stdout
    expected = {
        "dirty": "BLOCK_DIRTY_WORKTREE",
        "ignored": "BLOCK_UNEXPECTED_IGNORED_CONTENT",
        "unsafe-env": "local .env exclusion is unsafe",
        "unregistered-worktree": "Claude worktree exclusion is invalid",
        "prunable": "BLOCK_PRUNABLE_WORKTREE_REGISTRATION",
    }[pollution]
    assert expected in result.stdout + result.stderr


def test_source_only_handover_preserves_regenerable_cache_allowlist(tmp_path):
    repo = _source_only_repo(tmp_path)
    for directory in (".pytest_cache", ".ruff_cache", "gx1/__pycache__"):
        path = repo / directory
        path.mkdir(parents=True)
        (path / "fixture.cache").write_text("regenerable\n")
    result = _run_source_only(repo)
    assert result.returncode == 0, result.stderr
    assert "unexpected_ignored_path_count: 0" in result.stdout


@pytest.mark.parametrize(("reason", "next_action"), [
    (
        "SUCCESSOR_CPU_READY_REQUIRES_SCOPED_CUDA_AUTHORIZATION_AND_RUNTIME_EVIDENCE",
        "EXPLICIT_SCOPED_CUDA_AUTHORIZATION_AND_FRESH_SIGNED_160W_TELEMETRY_THEN_SUCCESSOR_RUNTIME_EVIDENCE",
    ),
    (
        "GUARD_EXIT_ORPHANED_CUDA_NO_RETRY",
        "CPU_GUARD_REPAIR_AND_VERIFIED_SOURCE_BOUND_CHECKPOINT_RECOVERY_NO_CUDA_RETRY",
    ),
    (
        "PRETRAIN_READINESS_REPAIR_REQUIRED__NO_TRAINING_OR_CLOUD_AUTHORITY",
        "COMPLETE_CPU_REPAIRS_AND_REVIEW_PRESERVE_CHECKPOINT_THEN_VERIFY_SOURCE_LINEAGE_NO_TRAINING_OR_PURCHASE",
    ),
    (
        "LOCAL_PRE_CLOUD_READINESS_COMPLETE__EXTERNAL_HOST_AND_FRESH_GATE_REQUIRED",
        "PRESERVE_PACKAGE_NO_TRAINING__LATER_SELECT_HOST_TRANSFER_REHASH_GUARDED_SMOKE_AND_FRESH_GATE",
    ),
])
def test_runtime_review_hold_reports_exact_recovery_and_blocks_all_execution(
    tmp_path, reason, next_action,
):
    repo = _source_only_repo(tmp_path)
    state_path = repo / "PROJECT_STATE_xau_direction_launch.json"
    state = json.loads(state_path.read_text())
    state["pretraining_review_hold"]["reason"] = reason
    state_path.write_text(json.dumps(state))
    result = subprocess.run(
        ["bash", str(repo / "scripts/gx1_handover.sh"), "--check"],
        cwd=repo, text=True, capture_output=True, check=False, timeout=15,
    )
    assert result.returncode == 2
    assert "decision: BLOCK" in result.stdout
    assert "pretraining_review_hold: ACTIVE" in result.stdout
    assert "cuda_authority: NONE" in result.stdout
    assert "test_paper_live_authority: NONE" in result.stdout
    assert f"blocker: {reason}" in result.stdout
    assert f"next_action: {next_action}" in result.stdout
    assert "next_action: CPU_SUCCESSOR" not in result.stdout


def test_handover_viewer_points_to_current_xau_direction_repair_truth() -> None:
    text = HANDOVER_VIEWER.read_text(encoding="utf-8")

    assert "rev-parse --show-toplevel" in text
    assert "REPO=/home/andre2/src/GX1_ENGINE" not in text
    assert "HANDOVER_XAU_DIRECTION_REPAIR_20260714.md" in text
    assert (
        "takeover_entrypoint: scripts/entry_next_edge_control.sh handover"
        in text
    )
    assert "handover_owner: scripts/gx1_handover.sh" in text
    assert "trading bot for gold/XAUUSD" in text
    assert "selects LONG/SHORT/FLAT direction" in text
    assert "no competing" in text
    assert "GX1_ALLOW_LEGACY_HANDOVER" not in text
    assert "SMART JOINT POLICY PROMOTED" not in text


def test_handover_authority_fingerprint_includes_current_pair_manifest() -> None:
    text = HANDOVER_VIEWER.read_text(encoding="utf-8")

    # The pair is dynamically located from launch state, then passed as an
    # additional byte-bound authority input; merely parsing it is not enough.
    assert '"${sources[@]}" "$CURRENT_PAIR_MANIFEST"' in text


def test_handover_authority_fingerprint_covers_every_markdown_file() -> None:
    markdown_paths = {
        path.resolve()
        for path in REPO.rglob("*.md")
        if not any(
            part.startswith(".")
            for part in path.relative_to(REPO).parts
        )
    }
    fingerprint_markdown = {
        path.resolve() for path in AUTHORITY_PATHS if path.suffix == ".md"
    }
    assert fingerprint_markdown == markdown_paths


def test_only_one_handover_shell_entrypoint_exists() -> None:
    handover_scripts = sorted((REPO / "scripts").rglob("*handover*.sh"))

    assert handover_scripts == [HANDOVER_VIEWER]


def test_execute_routes_reuse_handover_source_hygiene() -> None:
    for path in (
        CONTROL,
        REPO / "scripts/run_entry_model_native_seq513_train.sh",
        REPO / "scripts/run_seq513_rebuild_chain_v1.sh",
    ):
        source = path.read_text(encoding="utf-8")
        assert 'scripts/gx1_handover.sh" --check' in source
        assert "unexpected_ignored_path_count: 0" in source
        assert "prunable_worktree_count: 0" in source


def test_handover_viewer_prints_current_goal() -> None:
    result = subprocess.run(
        ["bash", str(HANDOVER_VIEWER)],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if _assert_explicit_review_hold(result):
        return
    assert result.returncode == 0
    assert "# GX1 XAU Direction Repair Takeover (compact)" in result.stdout
    assert "Build the GX1 trading bot for gold/XAUUSD" in result.stdout
    assert "selects LONG/SHORT/FLAT direction" in result.stdout
    assert (
        "takeover_entrypoint: scripts/entry_next_edge_control.sh handover"
        in result.stdout
    )
    assert "decision: BLOCK" in result.stdout
    # Read the contract mode from its owner: a restated mode literal has gone
    # stale on every surface bump (rule 13).
    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CONTRACT_MODE,
    )

    assert (
        f"required_contract_mode: {MODEL_NATIVE_CONTRACT_MODE}" in result.stdout
    )
    assert "dataset_event_id: NONE" in result.stdout
    assert "dataset_admission_stage: NO_ADMITTED_UNIFIED_DATASET" in result.stdout
    assert "accepted_bundle_dir: NONE" in result.stdout
    assert "current_audited_dataset_status: " in result.stdout
    launch_state = json.loads(LAUNCH_STATE.read_text())
    from gx1.contracts.current_audited_dataset_evidence_v1 import (
        require_blocked_launch_state_with_current_audited_dataset,
    )
    dataset_summary = require_blocked_launch_state_with_current_audited_dataset(launch_state)
    assert f"current_audited_dataset_run_id: {dataset_summary['dataset_run_id']}" in result.stdout
    assert f"current_audited_dataset_report_count: {dataset_summary['report_count']}" in result.stdout
    assert (
        "dataset_contract: "
        "HASH_BOUND_AUDITED_REPORT_ONLY_PRODUCTION_ECONOMICS_BLOCKED"
        in result.stdout
    )
    reference = launch_state["current_source_technical_recipe"]
    assert f"train_recipe: {reference['status']}" in result.stdout
    assert re.search(
        r"candidate_session: SESSION_INTACT__checkpoint=\d+__phase=(?:train|validation)__epoch=\d+__next_batch=\d+",
        result.stdout,
    )
    assert re.search(r"candidate_validation: (?:NOT_REACHED|REQUIRES_AUDIT)", result.stdout)
    assert "candidate_session_contract_sha256: " in result.stdout
    assert "candidate_session_state_sha256: " in result.stdout
    assert "candidate_recipe_sha256: " in result.stdout
    assert "candidate_source_bindings_sha256: " in result.stdout
    assert f"current_source_technical_recipe: {reference['status']}" in result.stdout
    assert "current_source_technical_recipe_closure: LIVE_SOURCE_BYTES_MATCH_RECIPE__" in result.stdout
    if reference["status"] in {
        "MATERIALIZED_CPU_LAUNCH_DRY_RUN_PENDING__CUDA_NOT_EXECUTED",
        "MATERIALIZED_CPU_LAUNCH_DRY_RUN_PASS__CUDA_NOT_EXECUTED",
    }:
        assert "current_source_technical_recipe_closure: LIVE_SOURCE_BYTES_MATCH_RECIPE__CUDA_NOT_EXECUTED" in result.stdout
        assert "CANDIDATE_GATE_READY" not in result.stdout
    assert (
        "fresh_31004_train: "
        "BLOCKED_PENDING_CLEAN_PREFLIGHT_AND_EXPLICIT_REAUTHORIZATION"
        in result.stdout
    )
    assert (
        "host_telemetry: "
        "FRESH_SIGNED_160W_RESPONSE_REQUIRED_AFTER_EACH_RESTART_OR_DRIVER_RESET"
        in result.stdout
    )
    assert (
        "ignored_content_scope: "
        "DECLARED_LOCAL_RUNTIME_EXCLUSIONS_PLUS_REGENERABLE_CACHE_ONLY"
        in result.stdout
    )
    assert "historical_pnl_winrate: UNPROVEN" in result.stdout
    launch_state = json.loads(
        (REPO / "PROJECT_STATE_xau_direction_launch.json").read_text(
            encoding="utf-8"
        )
    )
    current_pair_manifest = Path(
        str(launch_state["current_pair_manifest"])
    )
    current_pair = json.loads(
        current_pair_manifest.read_text(encoding="utf-8")
    )
    assert current_pair_manifest.is_absolute()
    assert (
        f"pair_generation_id: {current_pair['pair_generation_id']}"
        in result.stdout
    )
    # The status owner states the standing requirement and dates the last
    # verification instead of restating a count that goes stale on the
    # next added test (rule 13/25).
    assert (
        "source_regression: "
        "RELEVANT_CONTRACT_TESTS_MUST_PASS_BEFORE_EACH_SOURCE_CHANGE"
        in result.stdout
    )
    assert "source_regression_last_verified: " in result.stdout
    assert "feature_owners: SAME_8_IMPLEMENTATIONS_NATIVE_M5_AND_M1_NO_VALUE_COPY" in result.stdout
    # The dims are DERIVED from the contract owner, so this assertion reads the
    # owner rather than restating the numbers (rule 13: every restated count in
    # this repository has gone stale within days).
    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CTX_CAT_DIM,
        MODEL_NATIVE_CTX_CONT_DIM,
        MODEL_NATIVE_SEQ_LEN,
        MODEL_NATIVE_SIGNAL_DIM,
    )

    assert (
        f"entry: local=M5 sequence={MODEL_NATIVE_SEQ_LEN} "
        f"signal={MODEL_NATIVE_SIGNAL_DIM} "
        f"ctx_cont={MODEL_NATIVE_CTX_CONT_DIM} "
        f"ctx_cat={MODEL_NATIVE_CTX_CAT_DIM}" in result.stdout
    )
    assert (
        "entry_feature_surface: "
        "HASH_BOUND_NATIVE_M5_LOADED_ONCE_EXACT_ZERO_COPY_SPLIT_WINDOWS"
        in result.stdout
    )
    assert "exit: local=M1 sequence=480 mtf=M5,M15,H1,H4,D1" in result.stdout
    assert (
        "mtf_construction: "
        "CLOSED_OHLCV_BEFORE_FEATURES_NO_COMPUTED_M1_RESAMPLING"
        in result.stdout
    )
    assert "## Resume boundary" in result.stdout
    if "candidate_guard_recovery" in json.loads(LAUNCH_STATE.read_text()):
        assert "resume_stage: VERIFIED_GUARD_RECOVERY__CONTINUE_EXACT_CURRENT_SESSION__DO_NOT_RESET_TRAIN" in result.stdout
    else:
        assert (
            "resume_stage: "
            "V9_TERMINAL_TECHNICAL_RESULT_RETAINED__NO_RESUME_OR_NEW_CUDA_AUTHORITY"
            in result.stdout
        )
    assert re.search(r"source_identity_gate: [A-Z_]+", result.stdout)
    assert (
        "dataset_rebuild: "
        "NOT_REQUIRED_FOR_OFFLINE_RESEARCH; "
        "PRODUCTION_ECONOMICS_REVIEW_MAY_REQUIRE_A_SUCCESSOR"
        in result.stdout
    )
    assert (
        "production_economics_blocker: "
        "ENTRY_FITTED_Q_PRODUCTION_ECONOMICS_NOT_BOUND"
        in result.stdout
    )
    assert (
        "capacity: audits=4G training_max=20G swap=512M candidate_cpu_affinity=0-7 "
        "dataloader_workers=0 one_job_at_a_time" in result.stdout
    )
    assert "physical limit remains 160 W" in result.stdout
    assert "execute only the explicitly authorised current-source recipe" in result.stdout
    assert "production-net claims" in result.stdout
    assert "## Full Handover (--verbose)" not in result.stdout
    assert len(result.stdout.encode("utf-8")) < 10_000


def test_takeover_documents_state_the_current_shared_m5_m1_boundary() -> None:
    rules = (REPO / "GX1_RULES.md").read_text(encoding="utf-8")
    agents = (REPO / "AGENTS.md").read_text(encoding="utf-8")
    handover = HANDOVER.read_text(encoding="utf-8")

    for text in (rules, agents, handover):
        assert "offline" in text.lower()
        assert "M5" in text
        assert "M1" in text
        assert "10G" in text or "10 GiB" in text
        assert "scripts/gx1_handover.sh" in text
    assert "480-bar M1" in handover
    assert "same eight feature owners" in handover


def test_launch_authority_has_no_admitted_dataset_or_bundle() -> None:
    state = json.loads(LAUNCH_STATE.read_text(encoding="utf-8"))

    assert state["decision"] == "BLOCK"
    assert state["latest_terminal_event_id"] == "NO_CURRENT_ADMITTED_EVENT"
    assert state["latest_terminal_event_decision"] == "BLOCK"
    from gx1.models.entry_v10.direction_decision_contract import (
        UNIFIED_ENTRY_EXIT_CONTRACT_SCHEMA_VERSION,
    )

    assert (
        state["required_unified_entry_exit_contract"]
        == UNIFIED_ENTRY_EXIT_CONTRACT_SCHEMA_VERSION
    )
    assert state["required_entry_action_order"] == ["LONG", "SHORT", "FLAT"]
    assert state["required_exit_action_order"] == ["HOLD", "EXIT_NOW"]
    assert state["required_same_bundle_shared_encoder"] is True
    assert state["required_exact_closed_m1_exit_path_envelope"] is True
    assert state["external_decision_models_allowed"] is False
    assert state["reviewed_local_runtime_exclusions"] == {
        "schema_version": "gx1_reviewed_local_runtime_exclusions_v1",
        "paths": [".claude/worktrees/", ".env", ".venv/"],
    }
    assert state["dataset_event_id"] is None
    assert state["dataset_admission_stage"] == "NO_ADMITTED_UNIFIED_DATASET"
    assert state["accepted_dataset_dir"] is None
    assert state["accepted_dataset_terminal_evidence"] is None
    from gx1.contracts.current_audited_dataset_evidence_v1 import (
        CURRENT_AUDITED_DATASET_BLOCKER,
        CURRENT_AUDITED_DATASET_STATUS,
        require_blocked_launch_state_with_current_audited_dataset,
    )

    if "pretraining_review_hold" in state and "current_pretest_trainability_readiness" not in state:
        # The retained causality audit predates corrected short returns. It
        # must no longer qualify as current evidence, even though its bytes
        # and historical PASS declaration remain intact.
        with pytest.raises(RuntimeError, match="EXECUTION_CAUSALITY_EXPECTATION_INVALID"):
            require_blocked_launch_state_with_current_audited_dataset(state)
    else:
        summary = require_blocked_launch_state_with_current_audited_dataset(state)
        assert summary["status"] == CURRENT_AUDITED_DATASET_STATUS
        assert summary["blocker"] == CURRENT_AUDITED_DATASET_BLOCKER
        expected_run = (
            state["current_source_technical_recipe"]["dataset_run_id"]
            if "current_pretest_trainability_readiness" in state
            else "V46_20260825T170935Z"
        )
        assert summary["dataset_run_id"] == expected_run
    assert state["accepted_bundle_dir"] is None
    assert state["bundle_metadata_sha256"] is None
    assert state["current_smoke_launch_evidence"] is None
    candidate_session = state["active_candidate_training_session"]
    assert candidate_session["schema_version"] == (
        "gx1_active_candidate_training_session_reference_v1"
    )
    session_recipe_bytes = Path(candidate_session["recipe_audit_path"]).read_bytes()
    assert hashlib.sha256(session_recipe_bytes).hexdigest() == candidate_session["recipe_audit_sha256"]
    session_recipe = json.loads(session_recipe_bytes)
    for key in ("run_id", "dataset_run_id", "source_commit", "source_bindings_sha256"):
        assert candidate_session[key] == session_recipe[key]
    assert Path(candidate_session["session_dir"]).is_absolute()
    assert Path(candidate_session["recipe_audit_path"]).is_absolute()
    assert re.fullmatch(r"[0-9a-f]{64}", candidate_session["recipe_audit_sha256"])
    assert re.fullmatch(
        r"[0-9a-f]{64}", candidate_session["source_bindings_sha256"]
    )
    assert re.fullmatch(r"[0-9a-f]{40}", candidate_session["source_commit"])
    current_source_recipe = state["current_source_technical_recipe"]
    assert current_source_recipe["schema_version"] == (
        "gx1_current_source_technical_recipe_reference_v1"
    )
    assert current_source_recipe["dataset_run_id"] == "PRETEST_V3_20260829T173000Z"
    assert Path(current_source_recipe["recipe_path"]).is_file()
    expected_keys = {
        "schema_version", "status", "recipe_path", "recipe_sha256",
        "source_commit", "source_bindings_sha256", "run_id", "dataset_run_id",
        "out_bundle_dir",
    }
    if "candidate_guard_recovery" in state:
        from gx1.contracts.entry_pretest_candidate_launch_gate_v1 import require_pretest_candidate_launch_gate
        recovery_binding = state["candidate_guard_recovery"]
        recovery_bytes = Path(recovery_binding["path"]).read_bytes()
        assert hashlib.sha256(recovery_bytes).hexdigest() == recovery_binding["sha256"]
        recovery = json.loads(recovery_bytes)
        assert recovery["decision"] == "PASS_EXACT_STATE_TRANSFER_NOT_CUDA_AUTHORITY"
        assert recovery["successor_recipe"] == {"path": candidate_session["recipe_audit_path"], "sha256": candidate_session["recipe_audit_sha256"]}
        assert recovery["successor_session_dir"] == candidate_session["session_dir"]
        assert current_source_recipe["status"] == "FIVE_YEAR_CANDIDATE_RECIPE_GATE_READY__VERIFIED_GUARD_RECOVERY__CONTINUATION_AUTHORIZED__NO_TEST_PAPER_LIVE_AUTHORITY"
        assert current_source_recipe["recipe_path"] == candidate_session["recipe_audit_path"]
        assert current_source_recipe["recipe_sha256"] == candidate_session["recipe_audit_sha256"]
        gate = require_pretest_candidate_launch_gate(
            current_source_recipe["candidate_launch_gate_path"], current_source_recipe["candidate_launch_gate_sha256"],
            expected_recipe_path=candidate_session["recipe_audit_path"], expected_recipe_sha256=candidate_session["recipe_audit_sha256"],
        )
        assert gate["run_id"] == candidate_session["run_id"]
        for name in ("original_recipe", "original_contract", "original_pointer", "original_state"):
            binding = recovery[name]
            assert hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest() == binding["sha256"]
        expected_keys.update({
            "postrun_bundle_audit_path", "postrun_bundle_audit_sha256", "postrun_bundle_audit_decision",
            "candidate_readiness_path", "candidate_readiness_sha256", "candidate_readiness_decision",
            "candidate_launch_gate_path", "candidate_launch_gate_sha256", "candidate_launch_gate_decision",
        })
    elif "current_pretest_trainability_readiness" in state:
        from gx1.contracts.entry_model_native_pretest_technical_recipe_v1 import (
            require_pretest_technical_recipe_metadata,
        )
        raw = Path(current_source_recipe["recipe_path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == current_source_recipe["recipe_sha256"]
        recipe = require_pretest_technical_recipe_metadata(
            json.loads(raw), expected_profile="smoke",
            expected_run_id=current_source_recipe["run_id"],
            expected_out_bundle_dir=current_source_recipe["out_bundle_dir"],
        )
        assert recipe["source_commit"] == current_source_recipe["source_commit"]
        assert recipe["source_bindings_sha256"] == current_source_recipe["source_bindings_sha256"]
        executed_status = "EXECUTED_TECHNICAL_SMOKE__POSTRUN_AUDIT_PENDING__NO_CANDIDATE_AUTHORITY"
        gated_status = (
            "EXECUTED_TECHNICAL_SMOKE__POSTRUN_AUDIT_FAIL__"
            "CANDIDATE_READINESS_READY__CANDIDATE_GATE_READY__NO_PROMOTION_AUTHORITY"
        )
        assert current_source_recipe["status"] in {
            "MATERIALIZED_CPU_LAUNCH_DRY_RUN_PENDING__CUDA_NOT_EXECUTED",
            "MATERIALIZED_CPU_LAUNCH_DRY_RUN_PASS__CUDA_NOT_EXECUTED",
            executed_status,
            gated_status,
        }
        if current_source_recipe["status"] in {executed_status, gated_status}:
            from gx1.contracts.entry_model_native_bundle_commit_v1 import require_bundle_commit_manifest
            bundle = require_bundle_commit_manifest(Path(current_source_recipe["out_bundle_dir"]))
            assert bundle["commit_sha256"] == current_source_recipe["bundle_commit_sha256"]
            expected_keys.update({
                "bundle_commit_manifest_sha256", "bundle_commit_sha256", "bundle_metadata_sha256",
            })
            if current_source_recipe["status"] == gated_status:
                from gx1.contracts.entry_pretest_candidate_launch_gate_v1 import (
                    require_pretest_candidate_launch_gate,
                )
                gate_path = Path(current_source_recipe["candidate_launch_gate_path"])
                gate_json = json.loads(gate_path.read_bytes())
                gate = require_pretest_candidate_launch_gate(
                    gate_path,
                    current_source_recipe["candidate_launch_gate_sha256"],
                    expected_recipe_path=gate_json["recipe"]["path"],
                    expected_recipe_sha256=gate_json["recipe"]["sha256"],
                )
                assert gate["dataset_dir"] == recipe["dataset_dir"]
                assert gate["dataset_run_id"] == recipe["dataset_run_id"]
                assert gate["decision"] == current_source_recipe["candidate_launch_gate_decision"]
                for prefix, gate_key in (
                    ("postrun_bundle_audit", "smoke_bundle_audit"),
                    ("candidate_readiness", "candidate_readiness"),
                ):
                    assert gate[gate_key] == {
                        "path": current_source_recipe[f"{prefix}_path"],
                        "sha256": current_source_recipe[f"{prefix}_sha256"],
                    }
                    event = json.loads(Path(gate[gate_key]["path"]).read_bytes())
                    assert event["decision"] == current_source_recipe[f"{prefix}_decision"]
                expected_keys.update({
                    "postrun_bundle_audit_path", "postrun_bundle_audit_sha256", "postrun_bundle_audit_decision",
                    "candidate_readiness_path", "candidate_readiness_sha256", "candidate_readiness_decision",
                    "candidate_launch_gate_path", "candidate_launch_gate_sha256", "candidate_launch_gate_decision",
                })
        else:
            assert not Path(current_source_recipe["out_bundle_dir"]).exists()
    else:
        assert current_source_recipe["status"] == (
            "FIVE_YEAR_CANDIDATE_RECIPE_GATE_READY__CUDA_NOT_EXECUTED__EXPLICIT_CUDA_REAUTHORIZATION_REQUIRED__NO_TEST_PAPER_LIVE_AUTHORITY"
        )
        assert current_source_recipe["run_id"] == "ENTRY_V9_FIVE_YEAR_CANDIDATE_20260904T201433Z"
        assert not Path(current_source_recipe["out_bundle_dir"]).exists()
        expected_keys.update({
            "postrun_bundle_audit_path", "postrun_bundle_audit_sha256", "postrun_bundle_audit_decision",
            "candidate_readiness_path", "candidate_readiness_sha256", "candidate_readiness_decision",
            "candidate_launch_gate_path", "candidate_launch_gate_sha256", "candidate_launch_gate_decision",
        })
    assert set(current_source_recipe) == expected_keys
    for key in ("recipe_sha256", "source_bindings_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", current_source_recipe[key])
    assert re.fullmatch(r"[0-9a-f]{40}", current_source_recipe["source_commit"])
    blockers = "\n".join(state["blockers"])
    # Stage transitions change prose, never the explicit admission fields or
    # the immutable recipe/gate identities validated above.
    assert "No admitted dataset" in blockers
    assert "Untouched TEST direction edge" in blockers
    assert "remain fail-closed" in blockers
    # Keep the fail-closed authority compact enough to inspect; immutable
    # run evidence remains in its external artifact paths.
    assert len(LAUNCH_STATE.read_bytes()) < 14_000
    assert not any(
        key in state
        for key in (
            "latest_trainability_bundle",
            "latest_failed_smoke_execution",
            "latest_rejected_downstream_evidence",
            "source_repair_checkpoint",
        )
    )


def test_handover_verbose_mode_is_explicit_and_prints_exact_full_handover() -> None:
    result = subprocess.run(
        ["bash", str(HANDOVER_VIEWER), "--verbose"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if _assert_explicit_review_hold(result):
        return
    assert result.returncode == 0
    authoritative_handover = HANDOVER.read_text(encoding="utf-8")
    rendered_handover = result.stdout.split(
        "## Full Handover (--verbose)\n", maxsplit=1
    )[1]
    assert rendered_handover == authoritative_handover
    assert "## Current feature architecture" in rendered_handover
    assert "## What is implemented" in rendered_handover
    assert "## What remains empirically unproven or unadmitted" in rendered_handover
    assert "## Current shared plan — no feature expansion" in rendered_handover
    assert "## Takeover" in rendered_handover
    assert rendered_handover.splitlines()[-1] == authoritative_handover.splitlines()[-1]


def test_handover_check_mode_is_minimal_and_path_order_hash_bound() -> None:
    result = subprocess.run(
        ["bash", str(HANDOVER_VIEWER), "--check"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if _assert_explicit_review_hold(result):
        return
    assert result.returncode == 0
    # The viewer derives its root from its own path through git. Recompute
    # against that worktree root (path bytes are part of the fingerprint).
    viewer_repo = Path(
        subprocess.check_output(
            ["git", "-C", str(HANDOVER_VIEWER.parent.parent), "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
    )
    digest = hashlib.sha256()
    digest.update(b"gx1-takeover-authority-v3\0")
    launch_state = json.loads((viewer_repo / LAUNCH_STATE.name).read_text())
    current_pair = Path(str(launch_state["current_pair_manifest"]))
    authority_paths = (
        *(viewer_repo / authority_path.relative_to(REPO) for authority_path in AUTHORITY_PATHS),
        current_pair,
    )
    for index, path in enumerate(authority_paths):
        path_bytes = str(path).encode("utf-8")
        payload = path.read_bytes()
        digest.update(index.to_bytes(4, "big"))
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    assert "mode: check" in result.stdout
    assert f"authority_fingerprint: {digest.hexdigest()}" in result.stdout
    assert "decision: BLOCK" in result.stdout
    assert "head_commit:" in result.stdout
    assert "changed_path_count:" in result.stdout
    assert "ignored_path_count:" in result.stdout
    assert "reviewed_ignored_path_count:" in result.stdout
    assert "unexpected_ignored_path_count: 0" in result.stdout
    assert "prunable_worktree_count:" in result.stdout
    assert re.search(r"worktree_fingerprint: [0-9a-f]{64}", result.stdout)
    assert re.search(r"candidate_session: SESSION_INTACT__checkpoint=\d+", result.stdout)
    assert re.search(r"candidate_recipe_sha256: [0-9a-f]{64}", result.stdout)
    assert "candidate_source_closure: FROZEN_COMMIT_BYTES_MATCH_RECIPE" in result.stdout
    reference = launch_state["current_source_technical_recipe"]
    assert f"current_source_technical_recipe: {reference['status']}" in result.stdout
    assert "current_source_technical_recipe_closure: LIVE_SOURCE_BYTES_MATCH_RECIPE__" in result.stdout
    assert "## Host capacity" not in result.stdout
    assert "## Active GX1 process groups" not in result.stdout
    assert "## Full Handover (--verbose)" not in result.stdout
    assert len(result.stdout.encode("utf-8")) < 1_000


def test_control_surface_handover_alias_uses_current_handover_viewer() -> None:
    result = subprocess.run(
        ["bash", str(CONTROL), "handover"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if _assert_explicit_review_hold(result):
        return
    assert result.returncode == 0
    assert "# GX1 XAU Direction Repair Takeover (compact)" in result.stdout
    assert "decision: BLOCK" in result.stdout
    assert "## Full Handover (--verbose)" not in result.stdout
    assert "SMART JOINT POLICY PROMOTED" not in result.stdout


def test_control_surface_handover_alias_exposes_minimal_resume_check() -> None:
    result = subprocess.run(
        ["bash", str(CONTROL), "handover", "--check"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if _assert_explicit_review_hold(result):
        return
    assert result.returncode == 0
    assert "mode: check" in result.stdout
    assert "authority_fingerprint:" in result.stdout
    assert "## Resume boundary" not in result.stdout


def test_control_surface_exposes_only_exact_model_native_routes() -> None:
    result = subprocess.run(
        ["bash", str(CONTROL), "--help"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    help_routes = {
        line.strip().split(maxsplit=1)[0]
        for line in result.stdout.splitlines()
        if line.startswith("  ")
        and line.strip().split(maxsplit=1)[0] in RETAINED_CONTROL_ROUTES
    }
    assert help_routes == RETAINED_CONTROL_ROUTES

    source = CONTROL.read_text(encoding="utf-8")
    for route in RETAINED_CONTROL_ROUTES:
        assert (
            f"  {route})" in source
            or f"  {route}|" in source
            or f"|{route})" in source
        )
    for stale_route in (
        "foundation-guardrails",
        "foundation-adoption-candidate",
        "foundation-activation-plan",
        "foundation-activation-apply",
        "foundation-activation-post-apply",
        "readiness-report",
        "stage-foundation-cleanup",
        "materialize-smoke",
        "candidate-readiness-smart",
        "replay-readiness-smart",
        "feature-ai-inventory",
        "chart-geometry-audit",
        "candlestick-audit",
        "challenger-extension-manifest",
        "smart-post-rebuild-refresh",
        "smart-smoke-train",
        "candidate-train-smart",
        "entry-exit-handoff",
        "entry-exit-transformer-train",
    ):
        assert f"  {stale_route})" not in source
        assert f"  {stale_route}\n" not in result.stdout
    assert "  entry-exit-" not in source
    assert "  exit-transformer-" not in source
    assert (
        "Entry/Exit launch, promotion, shadow\n"
        "and live operation are outside this checkout. The retired fixed-proxy\n"
        "joint-Exit route is not exposed. A future production economic contract needs\n"
        "immutable broker costs, financing and shared-portfolio replay before it can\n"
        "become an authority."
    ) in result.stdout


@pytest.mark.parametrize(
    ("mode", "omitted", "expected"),
    [
        ("bootstrap", "--start-utc", "requires exactly one explicit --start-utc"),
        ("successor", "--parent-root", "requires exactly one explicit --parent-root"),
        (
            "successor",
            "--expected-parent-manifest-sha256",
            "requires exactly one explicit --expected-parent-manifest-sha256",
        ),
    ],
)
def test_native_source_route_exposes_exact_bootstrap_or_successor_contract(
    mode: str,
    omitted: str,
    expected: str,
) -> None:
    required = {
        "--publication-mode": mode,
        "--vedtak": "UNIT_NATIVE_SOURCE",
        "--start-utc": "2026-07-01T00:00:00Z",
        "--end-utc": "2026-07-02T00:00:00Z",
        "--out-root": "/tmp/native-source",
        "--parent-root": "/tmp/native-parent",
        "--expected-parent-manifest-sha256": "1" * 64,
    }
    argv = ["bash", str(CONTROL), "model-native-native-m5-source"]
    for flag, value in required.items():
        if flag != omitted:
            argv.extend((flag, value))

    result = subprocess.run(
        argv,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert expected in result.stderr


def test_launch_finalizer_route_requires_live_tail_admission() -> None:
    required = {
        "--accepted-bundle-dir": "/tmp/bundle",
        "--sizing-adoption-json": "/tmp/sizing.json",
        "--joint-exit-proof-json": "/tmp/exit.json",
        "--sizing-runtime-parity-json": "/tmp/sizing-parity.json",
        "--serve-parity-json": "/tmp/serve.json",
        "--direction-pocket-json": "/tmp/pocket.json",
        "--adaptation-lifecycle-json": "/tmp/lifecycle.json",
        "--launch-vedtak-json": "/tmp/vedtak.json",
        "--transaction-id": "UNIT_LAUNCH",
        "--max-trades": "1",
    }
    argv = ["bash", str(CONTROL), "model-native-finalize-launch"]
    for flag, value in required.items():
        argv.extend((flag, value))

    result = subprocess.run(
        argv,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "GX1_OFFLINE_SCOPE_FORBIDDEN" in result.stderr


def test_candidate_readiness_route_requires_exact_trainability_event() -> None:
    source = CONTROL.read_text(encoding="utf-8")
    route = source.split("  model-native-candidate-readiness)", 1)[1].split(
        "    ;;", 1
    )[0]

    assert "--trainability-readiness-json" in route
    assert "--upstream-readiness-json" not in route
    assert "foundation" not in route.lower()
    assert "worktree" not in route.lower()


def test_smoke_routes_use_the_legacy_wrapper_and_candidate_route_is_retired() -> None:
    source = CONTROL.read_text(encoding="utf-8")
    wrapper = "scripts/run_entry_model_native_seq513_train.sh"
    smoke_route = source.split("  model-native-smoke-train)", 1)[1].split(
        "    ;;", 1
    )[0]
    attended_smoke_route = source.split(
        "  model-native-attended-smoke-train)", 1
    )[1].split("    ;;", 1)[0]
    attended_cpu_smoke_route = source.split(
        "  model-native-attended-cpu-smoke-train)", 1
    )[1].split("    ;;", 1)[0]
    candidate_route = source.split(
        "  model-native-candidate-train)", 1
    )[1].split("    ;;", 1)[0]
    trainability_route = source.split(
        "  model-native-trainability-readiness)", 1
    )[1].split("    ;;", 1)[0]

    assert wrapper in smoke_route
    assert "--profile smoke" in smoke_route
    assert 'reject_flags "$cmd" --attended-smoke --research-smoke' in smoke_route
    assert "--train-sequence-source-audit-json" in smoke_route
    assert "--val-sequence-source-audit-json" in smoke_route
    assert wrapper in attended_smoke_route
    assert "--profile smoke" in attended_smoke_route
    assert "--attended-smoke" in attended_smoke_route
    assert "--train-sequence-source-audit-json" in attended_smoke_route
    assert "--val-sequence-source-audit-json" in attended_smoke_route
    assert wrapper in attended_cpu_smoke_route
    assert "--profile smoke" in attended_cpu_smoke_route
    assert "--attended-cpu-smoke" in attended_cpu_smoke_route
    assert "--train-sequence-source-audit-json" in attended_cpu_smoke_route
    assert "--val-sequence-source-audit-json" in attended_cpu_smoke_route
    assert wrapper not in candidate_route
    assert "model-native-candidate-train is retired" in candidate_route
    assert "immutable candidate launch gate" in candidate_route
    assert source.count(wrapper) == 3
    assert "--train-wrapper" in trainability_route


def test_recipe_and_post_smoke_audit_routes_are_explicit() -> None:
    source = CONTROL.read_text(encoding="utf-8")
    recipe = source.split("  model-native-train-recipe-audit)", 1)[1].split(
        "    ;;", 1
    )[0]
    for flag in (
        "--profile",
        "--repo",
        "--wrapper-path",
        "--run-id",
        "--dataset-dir",
        "--out-bundle-dir",
        "--m5-prebuilt-path",
        "--post-rebuild-readiness-json",
        "--prefreeze-test-seal-json",
        "--prefreeze-test-seal-sha256",
        "--full-input-liveness-audit-json",
        "--feature-audit-json",
        "--target-audit-json",
        "--specialist-audit-json",
        "--pretrain-audit-json",
        "--train-sequence-source-audit-json",
        "--val-sequence-source-audit-json",
        "--trainability-readiness-json",
        "--memory-cap",
        "--swap-cap",
        "--out-dir",
    ):
        assert flag in recipe
    assert "--test-manifest-json" not in recipe
    assert "--test-parquet" not in recipe
    assert "materialize_entry_model_native_seq513_train_recipe_audit_v1" in recipe
    assert (
        'AUDIT_CAP=("$REPO/scripts/gx1_capped_run.sh" --class audit '
        '--mem 4G --swap 512M --)'
    ) in source

    audit = source.split("  model-native-smoke-bundle-audit)", 1)[1].split(
        "    ;;", 1
    )[0]
    for flag in (
        "--bundle-dir",
        "--dataset-dir",
        "--val-manifest-json",
        "--predictions-parquet",
        "--prediction-report-json",
        "--target-audit-json",
        "--specialist-audit-json",
        "--pretrain-audit-json",
        "--out-dir",
        "--device",
    ):
        assert flag in audit
    assert "--test-manifest-json" not in audit
    assert "audit_entry_foundation_smoke_bundle_v1" in audit
    assert "CPU-only immutable proof audit" in audit
    assert 'exec "${AUDIT_CAP[@]}"' in audit
    assert "PRODUCER_CAP" not in audit

    prediction = source.split("  model-native-selective-edge)", 1)[1].split(
        "    ;;", 1
    )[0]
    for flag in (
        "--bundle-dir",
        "--dataset-dir",
        "--splits",
        "--evidence-stage",
        "--device",
        "--batch-size",
        "--stream-chunk-rows",
        "--m5-prebuilt-path",
        "--multi-tf-cache-dir",
        "--out-dir",
    ):
        assert flag in prediction
    assert 'reject_flags "$cmd" --top-fracs --model-name --selection-score-mode' in prediction
    assert "--cuda-producer" in prediction
    assert "same one-second 220 W / thermal / VRAM guard" in prediction
    assert "evaluate_entry_candidate_selective_edge_v1" in prediction


def test_report_only_admission_routes_use_the_narrow_audit_cap() -> None:
    """Evidence-only routes must not receive dataset-producer resources."""
    source = CONTROL.read_text(encoding="utf-8")
    for command, successor in (
        ("model-native-adoption-candidate", "model-native-smoke-manifest"),
        ("model-native-smoke-manifest", "model-native-smoke-readiness"),
        ("model-native-smoke-readiness", "model-native-trainability-readiness"),
        ("model-native-trainability-readiness", "model-native-train-recipe-audit"),
    ):
        route = source.split(f"  {command})", 1)[1].split(
            f"  {successor})", 1
        )[0]
        assert 'exec "${AUDIT_CAP[@]}"' in route
        assert "PRODUCER_CAP" not in route


def test_retired_separate_exit_dataset_route_is_absent() -> None:
    source = CONTROL.read_text(encoding="utf-8")
    assert "model-native-v3-exit-dataset" not in source
    assert "gx1.exits.training.thin_record_dataset" not in source


def test_pre_unified_active_registry_replay_route_is_absent() -> None:
    source = CONTROL.read_text(encoding="utf-8")
    assert "model-native-canonical-active-exit-replay" not in source
    assert "produce-canonical-joint-exit-proof" not in source


def test_rebuild_preflight_route_requires_the_exact_rebuild_wrapper_inputs() -> None:
    source = CONTROL.read_text(encoding="utf-8")
    route = source.split("  model-native-rebuild-preflight)", 1)[1].split(
        "    ;;", 1
    )[0]

    for flag in (
        "--run-id",
        "--source-parquet",
        "--canonical-v2-parquet",
        "--signal-manifest",
        "--feature-ranking-json",
        "--mtf-cache-dir",
        "--tape-root",
        "--m1-lifecycle-pair-manifest-json",
        "--m1-lifecycle-pair-generation-root",
        "--m1-feature-base-parquet",
        "--m5-feature-base-parquet",
        "--exit-lifecycle-dir",
        "--output",
        "--audit-out-dir",
        "--history-start",
        "--train-start",
        "--train-end",
        "--val-start",
        "--val-end",
        "--test-start",
        "--test-end",
        "--out-dir",
    ):
        assert flag in route
    for retired in (
        # V30 retired the whole TRAIN-rank-reference subsystem; the route may
        # not accept its artifact again.
        "--rank-reference-npz",
        "--smart-report",
        "--smart-manifest",
        "--inventory-report",
        "--gx1-data-root",
        "--source-dataset-dir",
        "--planned-dataset-dir",
        "--verify-large-input-hashes",
    ):
        assert retired not in route


def test_rebuild_preflight_help_exposes_every_required_lineage_input() -> None:
    result = subprocess.run(
        ["bash", str(CONTROL), "--help"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    usage = result.stdout.split("  model-native-rebuild-preflight \\\n", 1)[1].split(
        "  model-native-post-rebuild-readiness", 1
    )[0]
    for flag in (
        "--run-id",
        "--source-parquet",
        "--canonical-v2-parquet",
        "--signal-manifest",
        "--feature-ranking-json",
        "--mtf-cache-dir",
        "--tape-root",
        "--m1-lifecycle-pair-manifest-json",
        "--m1-lifecycle-pair-generation-root",
        "--m1-feature-base-parquet",
        "--m5-feature-base-parquet",
        "--exit-lifecycle-dir",
        "--output",
        "--audit-out-dir",
        "--history-start",
        "--train-start",
        "--train-end",
        "--val-start",
        "--val-end",
        "--test-start",
        "--test-end",
        "--out-dir",
    ):
        assert flag in usage


def test_rebuild_preflight_route_fails_before_dispatch_without_lineage_inputs() -> None:
    required = {
        "--run-id": "XAU_SEQ513_REBUILD_TEST_V1",
        "--source-parquet": "/tmp/source.parquet",
        "--canonical-v2-parquet": "/tmp/canonical.parquet",
        "--signal-manifest": "/tmp/signal.json",
        "--feature-ranking-json": "/tmp/ranking.json",
        "--mtf-cache-dir": "/tmp/mtf",
        "--tape-root": "/tmp/tape",
        "--m1-lifecycle-pair-manifest-json": "/tmp/pair/PAIR_MANIFEST.json",
        "--m1-lifecycle-pair-generation-root": "/tmp/pair-generations",
        "--m1-feature-base-parquet": "/tmp/m1-feature-base.parquet",
        "--m5-feature-base-parquet": "/tmp/m5-feature-base.parquet",
        "--exit-lifecycle-dir": "/tmp/exit-lifecycle",
        "--output": "/tmp/output__DIR_TRAIN_FIT.parquet",
        "--audit-out-dir": "/tmp/audit",
        "--history-start": "2020-01-01T00:00:00Z",
        "--train-start": "2020-01-02T00:00:00Z",
        "--train-end": "2025-01-01T00:00:00Z",
        "--val-start": "2025-01-02T00:00:00Z",
        "--val-end": "2025-06-01T00:00:00Z",
        "--test-start": "2025-06-02T00:00:00Z",
        "--test-end": "2026-01-01T00:00:00Z",
        "--out-dir": "/tmp/reports",
    }

    for missing in ("--run-id", "--feature-ranking-json", "--history-start"):
        argv = ["bash", str(CONTROL), "model-native-rebuild-preflight"]
        for flag, value in required.items():
            if flag != missing:
                argv.extend([flag, value])
        result = subprocess.run(
            argv,
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 2
        assert f"requires exactly one explicit {missing}" in result.stderr


def test_control_rejects_duplicate_required_flag_before_dispatch() -> None:
    result = subprocess.run(
        [
            "bash",
            str(CONTROL),
            "model-native-m5-feature-base",
            "--source-parquet",
            "/tmp/source.parquet",
            "--seq-structure-manifest",
            "/tmp/signal.json",
            "--output-parquet",
            "/tmp/surface-a.parquet",
            "--output-parquet",
            "/tmp/surface-b.parquet",
            "--dataset-run-id",
            "run",
            "--pair-generation-id",
            "0" * 64,
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert (
        "requires exactly one explicit --output-parquet (observed=2)"
        in result.stderr
    )


FEATURE_USEFULNESS_ARGUMENTS = {
    "--execute": None,
    "--device": "cpu",
    "--bundle-dir": "/tmp/immutable-bundle",
    "--bundle-metadata-sha256": "1" * 64,
    "--session-contract-sha256": "2" * 64,
    "--active-pointer-sha256": "3" * 64,
    "--selected-checkpoint-sha256": "4" * 64,
    "--recipe-audit-json": "/tmp/immutable-recipe.json",
    "--recipe-audit-sha256": "5" * 64,
    "--batch-size": "1",
    "--max-baseline-bytes": "1024",
    "--max-episode-bytes": "2048",
    "--max-forward-calls": "16",
    "--out-json": "/tmp/new-usefulness.json",
}


def _feature_usefulness_arguments(*, omitted=None, equals_form=False) -> list[str]:
    arguments = []
    for flag, value in FEATURE_USEFULNESS_ARGUMENTS.items():
        if flag == omitted:
            continue
        if value is None:
            arguments.append(flag)
        elif equals_form:
            arguments.append(f"{flag}={value}")
        else:
            arguments.extend((flag, value))
    return arguments


@pytest.fixture
def feature_usefulness_control(tmp_path: Path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    control = scripts / CONTROL.name
    control.write_text(CONTROL.read_text(encoding="utf-8"), encoding="utf-8")
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
    python.chmod(0o755)
    capped = scripts / "gx1_capped_run.sh"
    capped.write_text(
        "#!/usr/bin/env bash\nprintf 'CAPPED_STUB\\n'\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    capped.chmod(0o755)

    def run(
        arguments,
        *,
        handover_output=(
            "unexpected_ignored_path_count: 0\nprunable_worktree_count: 0\n"
        ),
        handover_status=0,
    ):
        handover = scripts / "gx1_handover.sh"
        handover.write_text(
            "#!/usr/bin/env bash\n"
            "printf 'HANDOVER_STUB:%s\\n' \"$*\" >&2\n"
            f"cat <<'EOF'\n{handover_output}EOF\nexit {handover_status}\n",
            encoding="utf-8",
        )
        handover.chmod(0o755)
        return subprocess.run(
            ["bash", str(control), "model-native-feature-usefulness", *arguments],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
        )

    return run


@pytest.mark.parametrize("equals_form", [False, True])
def test_feature_usefulness_dispatches_exact_cpu_audit_cap(
    tmp_path: Path, feature_usefulness_control, equals_form: bool,
) -> None:
    arguments = _feature_usefulness_arguments(equals_form=equals_form)
    result = feature_usefulness_control(arguments)

    assert result.returncode == 0, result.stderr
    assert result.stderr == "HANDOVER_STUB:--check\n"
    assert result.stdout.splitlines() == [
        "CAPPED_STUB",
        "--class", "audit", "--mem", "4G", "--swap", "512M", "--",
        str(tmp_path / ".venv/bin/python"),
        "-m", "gx1.scripts.audit_entry_exit_feature_usefulness_v1",
        *arguments,
    ]


@pytest.mark.parametrize("flag", FEATURE_USEFULNESS_ARGUMENTS)
def test_feature_usefulness_requires_each_exact_flag_before_dispatch(
    feature_usefulness_control, flag: str,
) -> None:
    result = feature_usefulness_control(_feature_usefulness_arguments(omitted=flag))

    assert result.returncode == 2
    assert f"requires exactly one explicit {flag} (observed=0)" in result.stderr
    assert "HANDOVER_STUB" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("flag", FEATURE_USEFULNESS_ARGUMENTS)
@pytest.mark.parametrize("equals_form", [False, True])
def test_feature_usefulness_rejects_duplicate_flags_before_dispatch(
    feature_usefulness_control, flag: str, equals_form: bool,
) -> None:
    arguments = _feature_usefulness_arguments()
    value = FEATURE_USEFULNESS_ARGUMENTS[flag]
    if equals_form:
        arguments.append(f"{flag}={value if value is not None else 'true'}")
    else:
        arguments.append(flag)
        if value is not None:
            arguments.append(value)
    result = feature_usefulness_control(arguments)

    assert result.returncode == 2
    assert f"requires exactly one explicit {flag} (observed=2)" in result.stderr
    assert "HANDOVER_STUB" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("flag", list(FEATURE_USEFULNESS_ARGUMENTS)[1:])
@pytest.mark.parametrize("value", [None, ""])
def test_feature_usefulness_rejects_missing_and_empty_values_before_dispatch(
    feature_usefulness_control, flag: str, value,
) -> None:
    arguments = _feature_usefulness_arguments()
    position = arguments.index(flag) + 1
    if value is None:
        del arguments[position]
    else:
        arguments[position] = value
    result = feature_usefulness_control(arguments)

    assert result.returncode == 2
    assert flag in result.stderr
    assert "requires" in result.stderr
    assert "HANDOVER_STUB" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("device", ["auto", "cuda", "cuda:0", "CPU"])
@pytest.mark.parametrize("equals_form", [False, True])
def test_feature_usefulness_rejects_non_cpu_devices_before_dispatch(
    feature_usefulness_control, device: str, equals_form: bool,
) -> None:
    arguments = _feature_usefulness_arguments(omitted="--device")
    arguments.extend([f"--device={device}"] if equals_form else ["--device", device])
    result = feature_usefulness_control(arguments)

    assert result.returncode == 2
    assert "--device must be exactly cpu" in result.stderr
    assert "HANDOVER_STUB" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "execute_arguments",
    [["--execute=true"], ["--execute=false"], ["--execute="], ["--execute", "true"]],
)
def test_feature_usefulness_requires_bare_execute(
    feature_usefulness_control, execute_arguments: list[str],
) -> None:
    result = feature_usefulness_control(
        [*execute_arguments, *_feature_usefulness_arguments(omitted="--execute")]
    )

    assert result.returncode == 2
    assert "HANDOVER_STUB" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "extra",
    [
        "--validate-json=/tmp/report.json", "--dry-run", "--allow-dirty-source",
        "--skip-source-check", "--source-only", "--repo=/tmp/other",
        "--sample-size=1", "--max-samples=1", "--max-rows=1", "--groups=trend",
        "--feature-mask-json=/tmp/mask.json", "--contract-mode=other",
        "--splits=train", "--dataset-dir=/tmp/other", "--device-auto",
        "--memory-cap=20G", "--swap-cap=1G", "--mem=20G", "--swap=1G",
        "--class=producer", "--workers=2", "--out=/tmp/abbreviated.json",
    ],
)
def test_feature_usefulness_rejects_nonproduction_flags_before_dispatch(
    feature_usefulness_control, extra: str,
) -> None:
    result = feature_usefulness_control([*_feature_usefulness_arguments(), extra])

    assert result.returncode == 2
    assert "HANDOVER_STUB" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    ("handover_output", "handover_status", "expected"),
    [
        (
            "pretraining_review_hold: ACTIVE\n"
            "unexpected_ignored_path_count: 0\nprunable_worktree_count: 0\n",
            2, "canonical handover source-identity verification failed",
        ),
        (
            "unexpected_ignored_path_count: 1\nprunable_worktree_count: 0\n",
            0, "--execute rejects unexpected ignored content",
        ),
        (
            "unexpected_ignored_path_count: 0\nprunable_worktree_count: 1\n",
            0, "--execute rejects prunable worktree registration",
        ),
        ("", 0, "--execute rejects unexpected ignored content"),
    ],
)
def test_feature_usefulness_handover_hold_and_source_hygiene_block_dispatch(
    feature_usefulness_control, handover_output: str, handover_status: int,
    expected: str,
) -> None:
    result = feature_usefulness_control(
        _feature_usefulness_arguments(), handover_output=handover_output,
        handover_status=handover_status,
    )

    assert result.returncode == 2
    assert "HANDOVER_STUB:--check" in result.stderr
    assert expected in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "argv",
    (
        ["model-native-live-tail-pair"],
        ["model-native-live-tail-admission"],
        ["model-native-finalize-launch"],
        ["model-native-adaptation-drift"],
        ["model-native-adaptation-shadow"],
        ["model-native-adaptation-lifecycle"],
    ),
)
def test_offline_scope_rejects_operational_routes(argv: list[str]) -> None:
    result = subprocess.run(
        ["bash", str(CONTROL), *argv],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "GX1_OFFLINE_SCOPE_FORBIDDEN" in result.stderr


def test_post_rebuild_route_binds_prefreeze_splits_and_exact_test_seal() -> None:
    source = CONTROL.read_text(encoding="utf-8")
    route = source.split(
        "  model-native-post-rebuild-readiness)", 1
    )[1].split("    ;;", 1)[0]

    for flag in (
        "--run-id",
        "--event-root",
        "--repo-dir",
        "--chain-terminal-json",
        "--test-seal-json",
        "--test-seal-sha256",
        "--rebuild-preflight-json",
        "--full-input-liveness-json",
        "--pretrain-audit-json",
        "--dataset-dir",
        "--smoke-dataset-dir",
        "--train-manifest-json",
        "--train-manifest-sha256",
        "--train-parquet",
        "--train-parquet-sha256",
        "--val-manifest-json",
        "--val-manifest-sha256",
        "--val-parquet",
        "--val-parquet-sha256",
        "--out-dir",
    ):
        assert flag in route
    for forbidden in (
        "--test-manifest-json",
        "--test-manifest-sha256",
        "--test-parquet",
        "--test-parquet-sha256",
    ):
        assert forbidden not in route
    assert "materialize_entry_model_native_seq513_post_rebuild_readiness_v1" in route


def test_foundation_audit_routes_bind_prefreeze_train_val_hashes_only() -> None:
    source = CONTROL.read_text(encoding="utf-8")
    routes = {
        "model-native-foundation-feature-audit": (
            "audit_entry_foundation_features_v1",
            True,
        ),
        "model-native-foundation-target-audit": (
            "audit_entry_foundation_targets_v1",
            False,
        ),
        "model-native-specialist-feature-audit": (
            "audit_entry_specialist_feature_groups_v1",
            True,
        ),
    }
    common_flags = (
        "--dataset-dir",
        "--train-manifest-json",
        "--train-manifest-sha256",
        "--train-parquet-sha256",
        "--val-manifest-json",
        "--val-manifest-sha256",
        "--val-parquet-sha256",
        "--out-dir",
    )
    for route_name, (module_name, requires_structure) in routes.items():
        route = source.split(f"  {route_name})", 1)[1].split("    ;;", 1)[0]
        assert module_name in route
        for flag in common_flags:
            assert flag in route
        assert "--test-manifest-json" not in route
        assert "--test-parquet" not in route
        assert ("--seq-structure-manifest" in route) is requires_structure


def test_obsolete_mega_guardrails_and_plan_tombstone_are_deleted() -> None:
    assert not (REPO / "gx1/scripts/verify_entry_foundation_guardrails_v1.py").exists()
    assert not (REPO / "gx1/scripts/verify_entry_next_edge_guardrails_v1.py").exists()
    assert not (REPO / "gx1/scripts/verify_entry_next_edge_plan_state_v1.py").exists()
    assert not (REPO / "gx1/scripts/verify_entry_model_native_abstention_probe_v1.py").exists()
    assert not (REPO / "gx1/contracts/entry_model_native_abstention_probe_v1.py").exists()
    assert "model-native-abstention-probe" not in CONTROL.read_text(encoding="utf-8")


def test_control_surface_selftest_is_report_only_and_launch_closed() -> None:
    result = subprocess.run(
        ["bash", str(CONTROL), "model-native-state-selftest", "--quiet"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_control_surface_rejects_mutable_latest_and_soft_pass_throughs() -> None:
    base = [
        "bash",
        str(CONTROL),
        "model-native-adoption-candidate",
        "--dataset-dir",
        "/tmp/dataset",
        "--feature-audit-json",
        "/tmp/feature.json",
        "--target-audit-json",
        "/tmp/target.json",
        "--specialist-audit-json",
        "/tmp/specialist.json",
        "--smoke-manifest-json",
        "/tmp/ENTRY_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST_latest.json",
        "--out-dir",
        "/tmp/out",
    ]
    latest = subprocess.run(
        base,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert latest.returncode == 2
    assert "mutable latest input is forbidden" in latest.stderr

    soft = subprocess.run(
        [
            "bash",
            str(CONTROL),
            "model-native-state",
            "--no-fail-on-not-ready",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert soft.returncode == 2
    assert "soft pass-through is forbidden" in soft.stderr


def test_control_surface_rejects_live_tail_inputs_on_offline_pair_route() -> None:
    result = subprocess.run(
        [
            "bash",
            str(CONTROL),
            "model-native-canonical-pair",
            "--live-tail-publication-event-root",
            "/tmp/events",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "GX1_OFFLINE_SCOPE_FORBIDDEN" in result.stderr


def test_model_native_adoption_route_requires_smoke_manifest_and_output_dir() -> None:
    without_smoke = subprocess.run(
        [
            "bash",
            str(CONTROL),
            "model-native-adoption-candidate",
            "--dataset-dir",
            "/tmp/dataset",
            "--feature-audit-json",
            "/tmp/feature.json",
            "--target-audit-json",
            "/tmp/target.json",
            "--specialist-audit-json",
            "/tmp/specialist.json",
            "--out-dir",
            "/tmp/out",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert without_smoke.returncode == 2
    assert "requires exactly one explicit --smoke-manifest-json" in without_smoke.stderr

    without_out = subprocess.run(
        [
            "bash",
            str(CONTROL),
            "model-native-adoption-candidate",
            "--dataset-dir",
            "/tmp/dataset",
            "--feature-audit-json",
            "/tmp/feature.json",
            "--target-audit-json",
            "/tmp/target.json",
            "--specialist-audit-json",
            "/tmp/specialist.json",
            "--smoke-manifest-json",
            "/tmp/ENTRY_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST_20260716T120000000000Z.json",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert without_out.returncode == 2
    assert "requires exactly one explicit --out-dir" in without_out.stderr


def test_removed_or_mutating_routes_fail_closed() -> None:
    for route in (
        "foundation-activation-apply",
        "candidate-readiness-smart",
        "entry-exit-transformer-train",
        "iql-distill",
    ):
        result = subprocess.run(
            ["bash", str(CONTROL), route],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 2
        assert f"unknown command: {route}" in result.stderr

    live = subprocess.run(
        ["bash", str(CONTROL), "live"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert live.returncode == 2
    assert "not exposed" in live.stderr


@pytest.mark.parametrize(
    ("route", "required_flag"),
    [
        ("model-native-sizing-fit-calibration", "--predictions"),
        ("model-native-sizing-bind-bundle", "--source-bundle-dir"),
        ("model-native-sizing-materialize-test-oos", "--calibration"),
        ("model-native-sizing-finalize-test-proof", "--calibration"),
        ("model-native-trade-path-metrics", "--replay-rows"),
        ("model-native-serve-parity", "--dataset-dir"),
        ("model-native-direction-pocket-audit", "--dataset-dir"),
    ],
)
def test_downstream_evidence_routes_are_exposed_but_fail_without_exact_inputs(
    route: str,
    required_flag: str,
) -> None:
    result = subprocess.run(
        ["bash", str(CONTROL), route],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 2
    assert f"requires exactly one explicit {required_flag}" in result.stderr


@pytest.mark.parametrize(
    "route",
    [
        "model-native-sizing-capture-instrument",
        "model-native-sizing-produce-unified-joint-proof",
        "model-native-sizing-adopt",
        "model-native-sizing-runtime-parity",
    ],
)
def test_retired_sizing_authority_routes_are_not_exposed(route: str) -> None:
    result = subprocess.run(
        ["bash", str(CONTROL), route],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 2
    assert f"unknown command: {route}" in result.stderr
