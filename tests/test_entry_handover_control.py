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
    REPO / "DEVELOPMENT_NOTES.md",
    REPO / "README.md",
    REPO / "GX1_PATHS.md",
    REPO / "RISK_OF_WRONG_CODE_2026_05_24.md",
    REPO / "ROADMAP.md",
    REPO / "SYSTEM_MAP.md",
    HANDOVER,
    REPO / "PROJECT_STATE.md",
    REPO / "DECISION_LOG.md",
    REPO / "PIPELINE_AUDIT_XAU_20260723.md",
    REPO / "docs/BACKFILL_2020_2025_COMMANDS.md",
    REPO / "docs/CANONICAL_EXIT_STATUS.md",
    REPO / "docs/DATA_CONTRACT.md",
    REPO / "docs/DATA_OANDA_SCHEMA_SSOT.md",
    REPO / "docs/ENTRY_CONTEXT_FEATURES_CONTRACT.md",
    REPO / "docs/FEATURE_MANIFEST.md",
    REPO / "docs/FOUNDATION_FEATURE_ROUTING_AUDIT_20260722.md",
    REPO / "docs/GIT_WORKTREE_POLICY.md",
    REPO / "docs/SESSION_CONTEXT_OBSERVABILITY_NOTE.md",
    REPO / "docs/TRAINING_DETERMINISM_MPS.md",
    REPO / "PROJECT_STATE_artifacts.json",
    REPO / "PROJECT_STATE_entry_iql_delete_incident.json",
    REPO / "PROJECT_STATE_xau_direction_launch.json",
)

RETAINED_CONTROL_ROUTES = {
    "handover",
    "model-native-state",
    "model-native-state-selftest",
    "model-native-native-m5-source",
    "model-native-native-m1-source",
    "model-native-canonical-pair",
    "model-native-live-tail-pair",
    "model-native-live-tail-admission",
    "model-native-mtf-v4-cache",
    "model-native-rebuild-preflight",
    "model-native-post-rebuild-readiness",
    "model-native-foundation-feature-audit",
    "model-native-foundation-target-audit",
    "model-native-specialist-feature-audit",
    "model-native-adoption-candidate",
    "model-native-smoke-manifest",
    "model-native-smoke-readiness",
    "model-native-trainability-readiness",
    "model-native-train-recipe-audit",
    "model-native-smoke-bundle-audit",
    "model-native-candidate-readiness",
    "model-native-selective-edge",
    "model-native-replay-trade-log",
    "model-native-replay-evidence",
    "model-native-replay-readiness",
    "model-native-finalize-launch",
    "model-native-rebuild",
    "model-native-smoke-train",
    "model-native-candidate-train",
}


def test_handover_viewer_points_to_current_xau_direction_repair_truth() -> None:
    text = HANDOVER_VIEWER.read_text(encoding="utf-8")

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
    assert "required_contract_mode: xau_seq513_model_native_direction_v4" in result.stdout
    assert "dataset_event_id: NONE" in result.stdout
    assert "dataset_admission_stage: NO_ADMITTED_UNIFIED_DATASET" in result.stdout
    assert "dataset_terminal_evidence: NONE" in result.stdout
    assert "current_smoke_launch_evidence: NONE" in result.stdout
    assert "accepted_bundle_dir: NONE" in result.stdout
    assert (
        "v4_architecture: VERIFIED "
        "timeframes=5 families=8 fields_per_tf=111 routes=40 cells=555"
        in result.stdout
    )
    assert "v4_cache: BLOCK observed=htf_v4_disk_cache_manifest_v2" in result.stdout
    assert (
        "historical_identity="
        "ff9cac78cdf6d5d4338f4d07b77df822c95efb568ed80a1e864600580a2b361a"
        in result.stdout
    )
    assert "active_seq513_chain" in result.stdout
    assert "## Resume boundary" in result.stdout
    assert "source_publication_contract: IMPLEMENTED_NOT_EXECUTED_OR_ADMITTED" in result.stdout
    assert "resume_owner: scripts/entry_next_edge_control.sh" in result.stdout
    assert "model-native-mtf-v4-cache" in result.stdout
    assert "## Full Handover (--verbose)" not in result.stdout
    assert "## Required evidence before Entry can open" not in result.stdout
    assert len(result.stdout.encode("utf-8")) < 10_000


def test_launch_authority_has_no_admitted_dataset_or_bundle() -> None:
    state = json.loads(LAUNCH_STATE.read_text(encoding="utf-8"))

    assert state["decision"] == "BLOCK"
    assert state["latest_terminal_event_id"] == "XAU_SEQ513_REBUILD_20260725_V26"
    assert state["latest_terminal_event_decision"] == "GREEN"
    assert state["dataset_event_id"] is None
    assert state["dataset_admission_stage"] == "NO_ADMITTED_UNIFIED_DATASET"
    assert state["accepted_dataset_dir"] is None
    assert state["accepted_dataset_terminal_evidence"] is None
    assert state["current_audited_dataset_evidence"] == {}
    assert state["accepted_bundle_dir"] is None
    assert state["bundle_metadata_sha256"] is None
    retired_bundle = state["latest_trainability_bundle"]
    assert retired_bundle["artifact_present"] is False
    assert retired_bundle["bundle_dir"] is None
    assert retired_bundle["bundle_commit_path"] is None
    assert retired_bundle["bundle_commit_sha256"] is None
    assert retired_bundle["bundle_commit_identity_sha256"] is None
    assert retired_bundle["model_state_sha256"] is None
    repair = state["source_repair_checkpoint"]
    assert repair["status"] == "CODE_PROVEN_EMPIRICALLY_UNPROVEN"
    assert repair["historical_rebuild_execution_started"] is True
    assert repair["historical_training_execution_started"] is True
    assert repair["active_v4_rebuild_started"] is False
    assert repair["active_v4_training_started"] is False
    assert repair["empirical_direction_edge_proven"] is False
    assert repair["remaining_source_p0"] == [
        "publish_fresh_generation_local_native_pair_for_lifecycle_authority",
        "rebuild_htf_v4_cache_under_manifest_v3_and_bind_fresh_dataset_lineage",
        "publish_fresh_combined_entry_exit_lifecycle_dataset",
        "train_and_prove_same_bundle_entry_exit_artifact",
        "prove_exact_closed_m1_exit_train_serve_parity_on_same_candidate",
        "execute_candidate_bound_full_test_unified_entry_exit_replay_and_runtime_parity",
        "execute_two_consecutive_fresh_live_tail_successors_and_publish_admission",
    ]
    assert (
        "immutable_live_tail_successor_publication_and_two_event_admission_owner"
        in repair["completed"]
    )
    assert state["current_smoke_launch_evidence"] is None
    failed = state["latest_failed_smoke_execution"]
    assert failed["run_id"] == "XAU_SEQ513_SMOKE_20260726_V10"
    assert failed["bundle_created"] is False

    rejected = state["latest_rejected_downstream_evidence"]
    rejected_path = Path(rejected["path"])
    assert hashlib.sha256(rejected_path.read_bytes()).hexdigest() == rejected["sha256"]

    blockers = "\n".join(state["blockers"])
    assert "No accepted candidate/direction bundle" in blockers
    # The optimization-throughput hypothesis this line used to pin was withdrawn
    # on 2026-07-27 after V14 confirmed the balanced-sampler prior mismatch by
    # measurement. The blocker now records the settled cause; V8/V9/V10 remain
    # immutable failure evidence. Pin the withdrawal so it cannot silently
    # revert to a refuted claim.
    assert "The FLAT-collapse cause is settled by measurement" in blockers
    assert "V8, V9 and V10 remain immutable failure evidence" in blockers
    assert "optimization-throughput hypothesis" in blockers
    assert "source-repaired" in blockers
    assert "transactional finalizer/recovery" in blockers
    assert "unified lifecycle materializer/loader" in blockers
    assert "No fresh native-manifest-bound lifecycle dataset" in blockers
    assert "no canonical immutable" not in blockers


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
    assert "## Next implementation sequence" in rendered_handover
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
    # The viewer deliberately pins one repository root inside its own source.
    # Recompute the fingerprint against exactly that root (path bytes are part
    # of the hash), so this test stays exact when run from a worktree checkout.
    viewer_repo = Path(
        next(
            line.split("=", 1)[1].strip()
            for line in HANDOVER_VIEWER.read_text(encoding="utf-8").splitlines()
            if line.startswith("REPO=")
        )
    )
    digest = hashlib.sha256()
    digest.update(b"gx1-takeover-authority-v2\0")
    for index, authority_path in enumerate(AUTHORITY_PATHS):
        path = viewer_repo / authority_path.relative_to(REPO)
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
    assert re.search(r"worktree_fingerprint: [0-9a-f]{64}", result.stdout)
    assert "## Host capacity" not in result.stdout
    assert "## Active GX1 process groups" not in result.stdout
    assert "## Full Handover (--verbose)" not in result.stdout
    assert len(result.stdout.encode("utf-8")) < 420


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
        "Unified Exit evidence is admitted only through the\n"
        "same-candidate, full-TEST producer route above."
    ) in result.stdout


@pytest.mark.parametrize(
    ("mode", "omitted", "expected"),
    [
        ("bootstrap", "--start-utc", "requires explicit --start-utc"),
        ("successor", "--parent-root", "requires explicit --parent-root"),
        (
            "successor",
            "--expected-parent-manifest-sha256",
            "requires explicit --expected-parent-manifest-sha256",
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
    assert "requires explicit --live-tail-admission-json" in result.stderr


def test_candidate_readiness_route_requires_exact_trainability_event() -> None:
    source = CONTROL.read_text(encoding="utf-8")
    route = source.split("  model-native-candidate-readiness)", 1)[1].split(
        "    ;;", 1
    )[0]

    assert "--trainability-readiness-json" in route
    assert "--upstream-readiness-json" not in route
    assert "foundation" not in route.lower()
    assert "worktree" not in route.lower()


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
        "--full-input-liveness-audit-json",
        "--feature-audit-json",
        "--target-audit-json",
        "--specialist-audit-json",
        "--pretrain-audit-json",
        "--trainability-readiness-json",
        "--memory-cap",
        "--swap-cap",
        "--out-dir",
    ):
        assert flag in recipe
    assert "materialize_entry_model_native_seq513_train_recipe_audit_v1" in recipe

    audit = source.split("  model-native-smoke-bundle-audit)", 1)[1].split(
        "    ;;", 1
    )[0]
    for flag in (
        "--bundle-dir",
        "--dataset-dir",
        "--val-manifest-json",
        "--test-manifest-json",
        "--predictions-parquet",
        "--prediction-report-json",
        "--target-audit-json",
        "--specialist-audit-json",
        "--pretrain-audit-json",
        "--out-dir",
        "--device",
    ):
        assert flag in audit
    assert "audit_entry_foundation_smoke_bundle_v1" in audit


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
        "--rank-reference-npz",
        "--mtf-cache-dir",
        "--tape-root",
        "--m1-lifecycle-pair-manifest-json",
        "--m1-lifecycle-pair-generation-root",
        "--exit-lifecycle-dir",
        "--exit-target-lookahead-m1-steps",
        "--early-move-threshold-bps",
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
        "--rank-reference-npz",
        "--mtf-cache-dir",
        "--tape-root",
        "--m1-lifecycle-pair-manifest-json",
        "--m1-lifecycle-pair-generation-root",
        "--exit-lifecycle-dir",
        "--exit-target-lookahead-m1-steps",
        "--early-move-threshold-bps",
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
        "--rank-reference-npz": "/tmp/rank.npz",
        "--mtf-cache-dir": "/tmp/mtf",
        "--tape-root": "/tmp/tape",
        "--m1-lifecycle-pair-manifest-json": "/tmp/pair/PAIR_MANIFEST.json",
        "--m1-lifecycle-pair-generation-root": "/tmp/pair-generations",
        "--exit-lifecycle-dir": "/tmp/exit-lifecycle",
        "--exit-target-lookahead-m1-steps": "30",
        "--early-move-threshold-bps": "4.0",
        "--output": "/tmp/output__DIR_H24B.parquet",
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
        assert f"requires explicit {missing}" in result.stderr


def test_rebuild_route_requires_the_explicit_target_threshold() -> None:
    source = CONTROL.read_text(encoding="utf-8")
    route = source.split("  model-native-rebuild)", 1)[1].split(
        "    ;;", 1
    )[0]

    assert "--early-move-threshold-bps" in route
    assert 'require_flag "$cmd" "$flag" "$@"' in route
    assert "--early_move_threshold_bps \"$EARLY_MOVE_THRESHOLD_BPS\"" not in route


@pytest.mark.parametrize(
    "missing",
    (
        "--m5-prebuilt",
        "--expected-source-sha256",
        "--out-dir",
    ),
)
def test_mtf_v4_cache_route_requires_exact_source_binding(
    missing: str,
) -> None:
    required = {
        "--m5-prebuilt": "/tmp/source.parquet",
        "--expected-source-sha256": "0" * 64,
        "--out-dir": "/tmp/new-cache",
    }
    argv = ["bash", str(CONTROL), "model-native-mtf-v4-cache"]
    for flag, value in required.items():
        if flag != missing:
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
    assert f"requires explicit {missing}" in result.stderr


def test_mtf_v4_cache_route_forbids_contract_override() -> None:
    result = subprocess.run(
        [
            "bash",
            str(CONTROL),
            "model-native-mtf-v4-cache",
            "--contract",
            "v2",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "fixes --contract in the exact evidence contract" in result.stderr


@pytest.mark.parametrize(
    "missing",
    (
        "--native-m1-root",
        "--native-m5-root",
        "--vedtak",
        "--checkpoint-dir",
        "--pair-manifest",
        "--generation-root",
        "--expected-pair-generation-id",
        "--expected-manifest-sha256",
        "--live-tail-publication-event-root",
    ),
)
def test_live_tail_pair_route_requires_exact_successor_authority(
    missing: str,
) -> None:
    required = {
        "--native-m1-root": "/tmp/native-m1",
        "--native-m5-root": "/tmp/native-m5",
        "--vedtak": "XAU_LIVE_TAIL_TEST_V1",
        "--checkpoint-dir": "/tmp/checkpoint",
        "--pair-manifest": "/tmp/pair.json",
        "--generation-root": "/tmp/generations",
        "--expected-pair-generation-id": "1" * 64,
        "--expected-manifest-sha256": "2" * 64,
        "--live-tail-publication-event-root": "/tmp/events",
    }
    argv = ["bash", str(CONTROL), "model-native-live-tail-pair"]
    for flag, value in required.items():
        if flag != missing:
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
    assert f"requires explicit {missing}" in result.stderr


def test_live_tail_pair_route_fixes_successor_mode() -> None:
    result = subprocess.run(
        [
            "bash",
            str(CONTROL),
            "model-native-live-tail-pair",
            "--publication-mode",
            "bootstrap",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert (
        "fixes --publication-mode in the exact evidence contract"
        in result.stderr
    )


def test_generic_pair_route_accepts_live_tail_publication_input() -> None:
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
    assert "requires explicit --publication-mode" in result.stderr
    assert "fixes --live-tail-publication-event-root" not in result.stderr


@pytest.mark.parametrize(
    "missing",
    (
        "--pair-manifest",
        "--generation-root",
        "--live-tail-admission-event-root",
        "--parent-live-tail-publication-json",
        "--parent-live-tail-publication-sha256",
        "--child-live-tail-publication-json",
        "--child-live-tail-publication-sha256",
    ),
)
def test_live_tail_admission_route_requires_exact_authority(
    missing: str,
) -> None:
    required = {
        "--pair-manifest": "/tmp/pair.json",
        "--generation-root": "/tmp/generations",
        "--live-tail-admission-event-root": "/tmp/admissions",
        "--parent-live-tail-publication-json": "/tmp/parent.json",
        "--parent-live-tail-publication-sha256": "1" * 64,
        "--child-live-tail-publication-json": "/tmp/child.json",
        "--child-live-tail-publication-sha256": "2" * 64,
    }
    argv = ["bash", str(CONTROL), "model-native-live-tail-admission"]
    for flag, value in required.items():
        if flag != missing:
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
    assert f"requires explicit {missing}" in result.stderr


def test_live_tail_admission_route_fixes_operation_mode() -> None:
    result = subprocess.run(
        [
            "bash",
            str(CONTROL),
            "model-native-live-tail-admission",
            "--publication-mode",
            "successor",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert (
        "fixes --publication-mode in the exact evidence contract"
        in result.stderr
    )


def test_post_rebuild_route_binds_terminal_audits_and_all_split_bytes() -> None:
    source = CONTROL.read_text(encoding="utf-8")
    route = source.split(
        "  model-native-post-rebuild-readiness)", 1
    )[1].split("    ;;", 1)[0]

    for flag in (
        "--run-id",
        "--event-root",
        "--repo-dir",
        "--chain-terminal-json",
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
        "--test-manifest-json",
        "--test-manifest-sha256",
        "--test-parquet",
        "--test-parquet-sha256",
        "--out-dir",
    ):
        assert flag in route
    assert "materialize_entry_model_native_seq513_post_rebuild_readiness_v1" in route


def test_foundation_audit_routes_bind_all_canonical_split_hashes() -> None:
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
        "--test-manifest-json",
        "--test-manifest-sha256",
        "--test-parquet-sha256",
        "--out-dir",
    )
    for route_name, (module_name, requires_structure) in routes.items():
        route = source.split(f"  {route_name})", 1)[1].split("    ;;", 1)[0]
        assert module_name in route
        for flag in common_flags:
            assert flag in route
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
    assert "requires explicit --smoke-manifest-json" in without_smoke.stderr

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
    assert "requires explicit --out-dir" in without_out.stderr


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
        ("model-native-sizing-capture-instrument", "--authority-root"),
        ("model-native-sizing-fit-calibration", "--predictions"),
        ("model-native-sizing-bind-bundle", "--source-bundle-dir"),
        ("model-native-sizing-materialize-test-oos", "--calibration"),
        ("model-native-sizing-finalize-test-proof", "--calibration"),
        (
            "model-native-sizing-produce-unified-joint-proof",
            "--calibration",
        ),
        ("model-native-sizing-adopt", "--bundle-dir"),
        ("model-native-sizing-runtime-parity", "--adoption"),
        ("model-native-serve-parity", "--dataset-dir"),
        ("model-native-direction-pocket-audit", "--dataset-dir"),
        ("model-native-adaptation-drift", "--bundle-dir"),
        ("model-native-adaptation-shadow", "--incumbent-bundle-dir"),
        ("model-native-adaptation-lifecycle", "--transition"),
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
    assert f"requires explicit {required_flag}" in result.stderr
