import hashlib
import json
import re
import subprocess
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
    HANDOVER,
    REPO / "docs/CURRENT_AUDIT_STATUS_20260828.md",
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
    "model-native-execution-causality-audit",
    "model-native-train-recipe-audit",
    "model-native-smoke-bundle-audit",
    "model-native-candidate-readiness",
    "model-native-selective-edge",
    "model-native-seed-stability",
    "model-native-smoke-train",
    "model-native-candidate-train",
    "model-native-trade-path-metrics",
}


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
    assert "current_audited_dataset_run_id: V46_20260825T170935Z" in result.stdout
    assert "current_audited_dataset_report_count: 12" in result.stdout
    assert (
        "dataset_contract: "
        "HASH_BOUND_AUDITED_REPORT_ONLY_PRODUCTION_ECONOMICS_BLOCKED"
        in result.stdout
    )
    assert (
        "train_recipe: "
            "HISTORICAL_V6_V8_BLOCKED__CURRENT_SOURCE_V9_SAFETY_PREFLIGHT_ONLY"
        in result.stdout
    )
    assert (
        "candidate_session: "
        "SESSION_INTACT__checkpoint=913__phase=train__epoch=1__next_batch=18368"
        in result.stdout
    )
    assert "candidate_validation: NOT_REACHED" in result.stdout
    assert "candidate_session_contract_sha256: " in result.stdout
    assert "candidate_session_state_sha256: " in result.stdout
    assert "candidate_recipe_sha256: " in result.stdout
    assert "candidate_source_bindings_sha256: " in result.stdout
    assert (
        "current_source_technical_recipe: "
            "MATERIALIZED_CPU_LAUNCH_DRY_RUN_PASS__CUDA_NOT_EXECUTED"
        in result.stdout
    )
    assert (
        "current_source_technical_recipe_closure: "
            "LIVE_SOURCE_BYTES_MATCH_RECIPE__CUDA_NOT_EXECUTED"
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
    assert (
        "resume_stage: "
            "RETAIN_V6_EPOCH_ONE_TECHNICAL_RESULT__V7_SMOKE_PREFLIGHT_MATERIALIZED"
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
        "capacity: audits=4G training_max=20G swap=512M cpu=0-1 "
        "dataloader_workers=0 one_job_at_a_time" in result.stdout
    )
    assert (
        "run the CPU-only V5 smoke-bundle audit"
        in result.stdout
    )
    assert "reach first complete TRAIN epoch and full VAL" in result.stdout
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

    summary = require_blocked_launch_state_with_current_audited_dataset(state)
    assert summary["status"] == CURRENT_AUDITED_DATASET_STATUS
    assert summary["blocker"] == CURRENT_AUDITED_DATASET_BLOCKER
    assert summary["dataset_run_id"] == "V46_20260825T170935Z"
    assert state["accepted_bundle_dir"] is None
    assert state["bundle_metadata_sha256"] is None
    assert state["current_smoke_launch_evidence"] is None
    candidate_session = state["active_candidate_training_session"]
    assert candidate_session["schema_version"] == (
        "gx1_active_candidate_training_session_reference_v1"
    )
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
    assert current_source_recipe["status"] == (
        "MATERIALIZED_CPU_LAUNCH_DRY_RUN_PASS__CUDA_NOT_EXECUTED"
    )
    assert current_source_recipe["run_id"].startswith(
        "V9_CURRENT_SOURCE_TECHNICAL_SMOKE_"
    )
    assert current_source_recipe["dataset_run_id"] == "PRETEST_V3_20260829T173000Z"
    assert Path(current_source_recipe["recipe_path"]).is_file()
    assert not Path(current_source_recipe["out_bundle_dir"]).exists()
    for key in ("recipe_sha256", "source_bindings_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", current_source_recipe[key])
    assert re.fullmatch(r"[0-9a-f]{40}", current_source_recipe["source_commit"])
    blockers = "\n".join(state["blockers"])
    assert "fresh immutable native M1/M5 pair" in blockers
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
    assert "candidate_session: SESSION_INTACT__checkpoint=913" in result.stdout
    assert re.search(r"candidate_recipe_sha256: [0-9a-f]{64}", result.stdout)
    assert "candidate_source_closure: FROZEN_COMMIT_BYTES_MATCH_RECIPE" in result.stdout
    assert (
        "current_source_technical_recipe: "
            "MATERIALIZED_CPU_LAUNCH_DRY_RUN_PASS__CUDA_NOT_EXECUTED"
        in result.stdout
    )
    assert (
        "current_source_technical_recipe_closure: "
            "LIVE_SOURCE_BYTES_MATCH_RECIPE__CUDA_NOT_EXECUTED"
        in result.stdout
    )
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


def test_train_routes_use_one_profile_explicit_wrapper_and_attended_route_is_isolated() -> None:
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
    assert wrapper in candidate_route
    assert "--profile candidate" in candidate_route
    assert "--train-sequence-source-audit-json" in candidate_route
    assert "--val-sequence-source-audit-json" in candidate_route
    assert source.count(wrapper) == 4
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
