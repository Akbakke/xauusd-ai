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
HANDOVER_VIEWER = REPO / "scripts/gx1_handover.sh"
CONTROL = REPO / "scripts/entry_next_edge_control.sh"
LAUNCH_STATE = REPO / "PROJECT_STATE_xau_direction_launch.json"

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


def test_launch_authority_has_no_admitted_dataset_or_bundle():
    state = json.loads(LAUNCH_STATE.read_text())
    assert state["decision"] == "BLOCK"
    assert state["latest_terminal_event_id"] == "NO_CURRENT_ADMITTED_EVENT"
    assert state["latest_terminal_event_decision"] == "BLOCK"
    assert state["required_entry_action_order"] == ["LONG", "SHORT", "FLAT"]
    assert state["required_exit_action_order"] == ["HOLD", "EXIT_NOW"]
    assert state["required_same_bundle_shared_encoder"] is True
    assert state["required_exact_closed_m1_exit_path_envelope"] is True
    assert state["external_decision_models_allowed"] is False
    assert state["reviewed_local_runtime_exclusions"] == {
        "schema_version": "gx1_reviewed_local_runtime_exclusions_v1",
        "paths": [".claude/worktrees/", ".env", ".venv/"],
    }
    from gx1.models.entry_v10.direction_decision_contract import UNIFIED_ENTRY_EXIT_CONTRACT_SCHEMA_VERSION
    assert state["required_unified_entry_exit_contract"] == UNIFIED_ENTRY_EXIT_CONTRACT_SCHEMA_VERSION
    for field in ("accepted_dataset_dir", "accepted_dataset_terminal_evidence",
                  "accepted_bundle_dir", "bundle_metadata_sha256",
                  "current_smoke_launch_evidence", "accepted_via_vedtak"):
        assert state[field] is None
    assert state["dataset_admission_stage"] == "NO_ADMITTED_UNIFIED_DATASET"
    assert state["pretraining_review_hold"]["decision"] == "BLOCK"
    assert state["pretraining_review_hold"]["activation_authority"] is False
    assert "active_candidate_training_session" not in state
    assert "current_source_technical_recipe" not in state


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

# Retired viewer internals are replaced by test_collect_gx1_handover_readonly.
@pytest.mark.parametrize("args", [[], ["--check"], ["--verbose"]])
def test_control_handover_alias_forwards_to_the_sole_current_viewer(tmp_path, args):
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(sys.executable)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    control = scripts / CONTROL.name
    control.write_text(CONTROL.read_text())
    viewer = scripts / "gx1_handover.sh"
    viewer.write_text('#!/bin/bash\nprintf "CURRENT_VIEWER:%s\\n" "$*"\nexit 78\n')
    viewer.chmod(0o755)
    result = subprocess.run(["bash", str(control), "handover", *args],
                            text=True, capture_output=True, check=False)
    assert result.returncode == 78
    assert result.stdout.strip() == "CURRENT_VIEWER:" + " ".join(args)
