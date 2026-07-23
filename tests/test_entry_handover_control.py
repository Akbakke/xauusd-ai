import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path("/home/andre2/src/GX1_ENGINE")
HANDOVER = REPO / "HANDOVER_XAU_DIRECTION_REPAIR_20260714.md"
HANDOVER_VIEWER = REPO / "scripts/gx1_handover.sh"
CONTROL = REPO / "scripts/entry_next_edge_control.sh"
LAUNCH_STATE = REPO / "PROJECT_STATE_xau_direction_launch.json"
AUTHORITY_PATHS = (
    REPO / "AGENTS.md",
    REPO / "ROADMAP.md",
    REPO / "SYSTEM_MAP.md",
    HANDOVER,
    REPO / "PROJECT_STATE.md",
    REPO / "DECISION_LOG.md",
    REPO / "PROJECT_STATE_artifacts.json",
    REPO / "PROJECT_STATE_entry_iql_delete_incident.json",
    REPO / "PROJECT_STATE_xau_direction_launch.json",
)

RETAINED_CONTROL_ROUTES = {
    "handover",
    "model-native-state",
    "model-native-state-selftest",
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
    "model-native-rebuild",
    "model-native-smoke-train",
    "model-native-candidate-train",
}


def test_handover_viewer_points_to_current_xau_direction_repair_truth() -> None:
    text = HANDOVER_VIEWER.read_text(encoding="utf-8")

    assert "HANDOVER_XAU_DIRECTION_REPAIR_20260714.md" in text
    assert "Use this script only: scripts/gx1_handover.sh" in text
    assert "trading bot for gold/XAUUSD" in text
    assert "selects LONG/SHORT/FLAT direction" in text
    assert "no competing" in text
    assert "GX1_ALLOW_LEGACY_HANDOVER" not in text
    assert "SMART JOINT POLICY PROMOTED" not in text


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
    assert "Use this script only: scripts/gx1_handover.sh" in result.stdout
    assert "decision: BLOCK" in result.stdout
    assert "required_contract_mode: xau_seq513_model_native_direction_v4" in result.stdout
    assert "dataset_event_id: XAU_SEQ513_REBUILD_20260722_V24" in result.stdout
    assert "dataset_admission_stage: READY_FOR_CAPPED_SMOKE_EXECUTION" in result.stdout
    assert "dataset_terminal_evidence: VERIFIED state=GREEN" in result.stdout
    assert "dataset_audit_evidence: VERIFIED count=9" in result.stdout
    assert "smoke_recipe_evidence: VERIFIED decision=PASS env_count=162" in result.stdout
    assert "source_commit=bf5c61a00500aa50890f118b6eb41ab5e91bb0c6" in result.stdout
    assert "smoke_recipe_dry_run: PASS" in result.stdout
    assert "accepted_bundle_dir: NONE" in result.stdout
    assert "active_seq513_chain" in result.stdout
    assert "## Full Handover (--verbose)" not in result.stdout
    assert "## Required evidence before Entry can open" not in result.stdout
    assert len(result.stdout.encode("utf-8")) < len(HANDOVER.read_bytes())


def test_launch_authority_binds_exact_current_v24_terminal_bytes() -> None:
    state = json.loads(LAUNCH_STATE.read_text(encoding="utf-8"))

    assert state["decision"] == "BLOCK"
    assert state["latest_terminal_event_id"] == "XAU_SEQ513_REBUILD_20260722_V24"
    assert state["latest_terminal_event_decision"] == "GREEN"
    assert state["dataset_event_id"] == "XAU_SEQ513_REBUILD_20260722_V24"
    assert state["dataset_admission_stage"] == "READY_FOR_CAPPED_SMOKE_EXECUTION"
    assert state["accepted_bundle_dir"] is None
    assert state["bundle_metadata_sha256"] is None

    terminal = state["accepted_dataset_terminal_evidence"]
    terminal_path = Path(terminal["path"])
    assert terminal_path.is_file()
    assert hashlib.sha256(terminal_path.read_bytes()).hexdigest() == terminal["sha256"]
    terminal_state = json.loads(terminal_path.read_text(encoding="utf-8"))
    assert terminal_state["entry_run_id"] == state["dataset_event_id"]
    assert terminal_state["state"] == terminal["state"] == "GREEN"

    audits = state["current_audited_dataset_evidence"]
    assert len(audits) == 9
    for binding in audits.values():
        audit_path = Path(binding["path"])
        audit_bytes = audit_path.read_bytes()
        assert hashlib.sha256(audit_bytes).hexdigest() == binding["sha256"]
        audit = json.loads(audit_bytes)
        assert audit["decision"] == binding["decision"]

    recipe_binding = state["current_smoke_launch_evidence"]["train_recipe_audit"]
    recipe_path = Path(recipe_binding["path"])
    recipe_bytes = recipe_path.read_bytes()
    assert hashlib.sha256(recipe_bytes).hexdigest() == recipe_binding["sha256"]
    recipe = json.loads(recipe_bytes)
    assert recipe["schema_version"] == "entry_model_native_seq513_train_recipe_audit_v1"
    assert recipe["decision"] == recipe_binding["decision"] == "PASS"
    assert recipe["profile"] == recipe_binding["profile"] == "smoke"
    assert recipe["run_id"] == recipe_binding["run_id"]
    assert recipe["source_commit"] == recipe_binding["source_commit"]
    assert len(recipe["trainer_env"]) == recipe_binding["trainer_env_count"] == 162
    assert (
        recipe["trainer_env_contract"]["sha256"]
        == recipe_binding["trainer_env_contract_sha256"]
    )
    assert (
        recipe["source_bindings_sha256"]
        == recipe_binding["source_bindings_sha256"]
    )
    assert recipe_binding["dry_run_decision"] == "PASS"
    assert recipe_binding["execution_started"] is False
    assert recipe_binding["out_bundle_present"] is False
    assert not Path(recipe_binding["out_bundle_dir"]).exists()
    assert subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            recipe_binding["source_commit"],
            "HEAD",
        ],
        cwd=REPO,
        check=False,
    ).returncode == 0

    rejected = state["latest_rejected_downstream_evidence"]
    rejected_path = Path(rejected["path"])
    assert hashlib.sha256(rejected_path.read_bytes()).hexdigest() == rejected["sha256"]

    blockers = "\n".join(state["blockers"])
    assert "No smoke model" in blockers
    assert "capped smoke execution" in blockers
    assert "post-smoke bundle audit have not run" in blockers
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
    assert "## Exact Entry contract" in rendered_handover
    assert "xau_seq513_model_native_direction_v4" in rendered_handover
    assert "## Required evidence before Entry can open" in rendered_handover
    assert "## Operational takeover" in rendered_handover
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
    digest = hashlib.sha256()
    digest.update(b"gx1-takeover-authority-v1\0")
    for index, path in enumerate(AUTHORITY_PATHS):
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
    assert "## Host capacity" not in result.stdout
    assert "## Active GX1 process groups" not in result.stdout
    assert "## Full Handover (--verbose)" not in result.stdout
    assert len(result.stdout.encode("utf-8")) < 320


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
        assert f"  {route})" in source
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
    assert "Exit evidence remains unavailable" in result.stdout


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
        "--output": "/tmp/output__HOLD_03B.parquet",
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
