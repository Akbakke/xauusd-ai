import json
import os
import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_seq513_rebuild_chain_v1.sh"


def test_chain_requires_explicit_fresh_immutable_inputs_without_discovery() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for required in (
        "--run-id",
        "--event-root",
        "--feature-ranking-json",
        "--preflight-out-dir",
        "--m1-lifecycle-pair-manifest-json",
        "--m1-lifecycle-pair-generation-root",
        "--registry-fit-inner-end",
        "--registry-fit-train-end",
        "--volatility-squeeze-manifest",
        "--volatility-squeeze-manifest-sha256",
        "REGISTRY_FIT_TRAIN_END=$TRAIN_END",
        'path.relative_to(event)',
        'feature ranking output must be a fresh timestamped JSON',
        'feature ranking timestamp cannot be in the future',
        'source does not cover the declared common history/test window',
        'pre_train_rows < 96',
        'declared common history does not cover the D1 receptive field',
        'pre_train_d1_bars < required_d1_warmup_bars',
        'MANIFEST="$EVENT/ENTRY_MODEL_NATIVE_SEQ513_SIGNAL_MANIFEST_${MANIFEST_STAMP}.json"',
        'fresh signal manifest allocation collided',
        'preflight output directory must be fresh',
        'fresh event outputs required',
        'preflight namespace must contain exactly one artifact',
        'preflight json_path is not an exact self-reference',
        'preflight run_id does not match chain run_id',
        'hashlib.sha256(raw).hexdigest()',
        "materialize_current_pair_source_cascade_proof_v1",
        '--source-cascade-proof "$SOURCE_CASCADE"',
        "require_source_cascade_unchanged",
        "require_pair_unchanged",
        "pair manifest must be the exact generation-local PAIR_MANIFEST.json",
        "model-native-m5-enriched-frame",
        "model-native-m1-enriched-frame",
        "model-native-m5-feature-base",
        "model-native-m1-feature-base",
        '--registry-fit-train-start "$TRAIN_START"',
        '--registry-fit-inner-end "$REGISTRY_FIT_INNER_END"',
        '--registry-fit-tape-manifest "$TAPE_MANIFEST"',
        '--expected-registry-fit-tape-manifest-sha256 "$TAPE_MANIFEST_SHA256"',
        '--registry-fit-train-end "$REGISTRY_FIT_TRAIN_END"',
        '--volatility-squeeze-manifest "$VOLATILITY_SQUEEZE_MANIFEST"',
        '--expected-volatility-squeeze-manifest-sha256 "$VOLATILITY_SQUEEZE_MANIFEST_SHA256"',
        '--v29-registry-constants-json "$MTF/manifest.json"',
        '--v29-registry-constants-json "${M1_ENRICHED}.manifest.json"',
        '--m1-feature-base-parquet "$M1_FEATURE_BASE"',
        '--m5-feature-base-parquet "$M5_FEATURE_BASE"',
        "--workers 1",
        "M5/M15/H1/H4/D1",
        '--rebuild-terminal-json "$DATASET_REBUILD_TERMINAL"',
        '--prefreeze-test-seal-json "$PREFREEZE_TEST_SEAL"',
        "dataset rebuild terminal/TEST seal binding mismatch",
    ):
        assert required in source
    assert "level-tol-quantile-q" not in source

    assert source.count('--feature-ranking-json "$RANKING"') == 3
    # One --run-id per capped producer invocation. The retired
    # train-rank-reference step took the sixth with it.
    assert source.count('--run-id "$RUN_ID"') == 5
    assert "materialize_entry_model_native_train_feature_ranker_v1" in source
    assert '--out "$RANKING"' in source
    assert "gx1_capped_run.sh --class audit --mem 4G --swap 512M" in source
    for forbidden in (
        # The fixed top-133 TRAIN rank subsystem is retired: no producer, no
        # NPZ identity, no CLI surface and no dangling variable may remain.
        "materialize_model_native_train_rank_reference",
        "model_native_train_rank_reference",
        "RANK_NPZ",
        "rank_npz",
        "--rank-reference-npz",
        "--rank-source-parquet",
        "--existing-rank-reference",
        "audit_seq513_source_cascade_v1",
        'CV2="$EVENT/canonical_features_v2.parquet"',
        'TAPE="$EVENT/m5_tape_native_v3"',
        "pgrep",
        "sleep 60",
        "ls -1",
        "sort |",
        "tail -1",
        "head -1",
        "BUILD_DONE",
        "rm -f",
        "rmdir ",
    ):
        assert forbidden not in source


def test_chain_binds_clean_source_revision_and_terminal_status() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'git -C "$ENG" rev-parse --verify HEAD' in source
    assert 'git -C "$ENG" status --porcelain --untracked-files=all' in source
    # Two of the pre-wave 17 belonged to the retired train-rank-reference step.
    assert source.count("require_source_identity") == 15
    assert 'repository HEAD changed after binding' in source
    assert 'repository worktree changed after binding' in source
    assert '"git_head": git_head or None' in source
    assert 'trap \'on_err "$?" "$LINENO"\' ERR' in source
    assert "trap 'on_signal TERM 143' TERM" in source
    assert "trap 'on_signal INT 130' INT" in source
    assert "trap 'on_signal HUP 129' HUP" in source
    assert 'terminal_status ABORTED "received $signal_name" "$exit_code"' in source
    assert 'terminal_status RED "unexpected ERR at line $line" "$exit_code"' in source
    assert source.index("trap 'on_err") < source.index("if not tape.is_dir()")
    assert source.index("trap 'on_err") < source.index(
        "not m1_lifecycle_pair_manifest.is_file()"
    )
    assert '"entry_run_id": run_id' in source
    assert '"step": step' in source
    assert '"state": state' in source
    # ONE OWNER (rule 13): the chain status schema string is imported from the
    # readiness gate, never restated. It was restated here as
    # "seq513_rebuild_chain_status_v9" while the gate required a different
    # version, which made post-rebuild readiness permanently RED with both test
    # suites green.
    from gx1.scripts.materialize_entry_model_native_seq513_post_rebuild_readiness_v1 import (
        CHAIN_SCHEMA,
    )

    assert (
        "from gx1.scripts.materialize_entry_model_native_seq513_post_rebuild_readiness_v1 import"
        in source
    )
    assert '"schema_version": CHAIN_SCHEMA' in source
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert f'"{CHAIN_SCHEMA}"' not in code
    assert '"next_boundary": (' in source
    assert '"unified_exit_lifecycle_dir": exit_lifecycle_dir' in source
    assert "stopped before post-rebuild audits/readiness" in source
    assert '"source_cascade": {' in source
    assert '"pair_authority": {' in source
    assert '"volatility_squeeze_artifact": {' in source
    assert '"dataset_rebuild_terminal": {' in source
    assert '"prefreeze_test_seal": {' in source
    assert '"m1_exit_feature_base": m1_feature_base_path' in source
    assert '"m5_entry_feature_base": m5_feature_base_path' in source
    assert '"boot_id": boot_id' in source
    assert '"chain_pid": int(chain_pid)' in source
    assert 'f"CHAIN_TERMINAL_{stamp}_{state}.json"' in source
    assert "feature-ranking-exact-checkpoint-resume" in source
    assert "dataset-rebuild-exact-checkpoint-resume" in source
    assert "DATASET_OUTPUT_STARTED" in source
    assert "dataset rebuild or post-build audit failed after immutable output materialization" in source
    assert "os.replace(temporary, path)" in source
    assert 'require_unchanged "dataset rebuild terminal"' in source
    assert 'require_unchanged "pre-freeze TEST seal"' in source


def test_chain_cli_rejects_old_positional_interface() -> None:
    help_result = subprocess.run(
        [str(SCRIPT), "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--feature-ranking-json" in help_result.stdout
    assert "--signal-manifest" not in help_result.stdout
    assert "--preflight-out-dir" in help_result.stdout
    assert "--m1-lifecycle-pair-manifest-json" in help_result.stdout
    assert "--m1-lifecycle-pair-generation-root" in help_result.stdout
    assert "--volatility-squeeze-manifest" in help_result.stdout
    assert "--volatility-squeeze-manifest-sha256" in help_result.stdout
    assert "--early-move-threshold-bps" not in help_result.stdout
    assert "--exit-target-lookahead-m1-steps" not in help_result.stdout

    old = subprocess.run(
        [str(SCRIPT), "XAU_SEQ513_REBUILD_UNIT_V1", "/tmp/unit-event"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert old.returncode == 2
    assert "unknown argument" in old.stderr


def test_chain_validation_failure_persists_red_run_lineage_and_revision(
    tmp_path: Path,
) -> None:
    event = (tmp_path / "event").resolve()
    event.mkdir()
    ranking = event / "ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING_20260719T200000000000Z.json"
    pair_generation_root = (tmp_path / "pair-generations").resolve()
    pair_generation_root.mkdir()
    pair_manifest = (tmp_path / "PAIR_MANIFEST.json").resolve()
    pair_manifest.write_bytes(b"unit fixture")
    outside = (tmp_path / "outside").resolve()
    preflight = event / "preflight-20260719T200002000000Z"
    run_id = "XAU_SEQ513_REBUILD_UNIT_V1"
    env = dict(os.environ)
    # Prevent the existing notifier from loading repository credentials during
    # this deliberate fail-closed probe. An empty chat ID makes send() a no-op.
    env["GX1_TELEGRAM_BOT_TOKEN"] = "unit-test-disabled"
    env["GX1_TELEGRAM_CHAT_ID"] = ""

    result = subprocess.run(
        [
            str(SCRIPT),
            "--run-id",
            run_id,
            "--event-root",
            str(event),
            "--feature-ranking-json",
            str(ranking),
            "--preflight-out-dir",
            str(outside / preflight.name),
            "--m1-lifecycle-pair-manifest-json",
            str(pair_manifest),
            "--m1-lifecycle-pair-generation-root",
            str(pair_generation_root),
            "--registry-fit-inner-end",
            "2024-01-01T00:00:00Z",
            "--volatility-squeeze-manifest",
            str(pair_manifest),
            "--volatility-squeeze-manifest-sha256",
            "0" * 64,
            "--history-start",
            "2021-01-05T00:00:00Z",
            "--train-start",
            "2021-03-16T00:00:00Z",
            "--train-end",
            "2026-05-31T23:59:59Z",
            "--val-start",
            "2026-06-01T00:00:00Z",
            "--val-end",
            "2026-06-30T23:59:59Z",
            "--test-start",
            "2026-07-01T00:00:00Z",
            "--test-end",
            "2026-07-21T16:25:00Z",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 2
    status = json.loads((event / "CHAIN_STATUS.json").read_text(encoding="utf-8"))
    assert status["state"] == "RED"
    # A dirty development checkout fails at the earlier source gate; a clean CI
    # checkout reaches the intentionally out-of-event preflight rejection.
    assert status["step"] in {
        "source-revision",
        "pair-authority",
        "contract-validation",
    }
    assert status["entry_run_id"] == run_id
    assert status["event_root"] == str(event)
    assert re.fullmatch(r"[0-9a-f]{40}", status["git_head"])
    assert Path(status["log_path"]).is_file()
    assert status["exit_code"] == 2
    assert status["schema_version"] == "seq513_rebuild_chain_status_v9"
    assert status["next_boundary"] == (
        "REPAIR_CURRENT_FAILED_STEP_WITHOUT_REUSING_PARTIAL_OUTPUT"
    )
    assert status["outputs"]["unified_exit_lifecycle_dir"].endswith(
        "/exit_lifecycle"
    )
    assert status["source_cascade"]["path"].endswith(".json")
    assert "rank_reference" not in status
    assert status["outputs"]["m1_exit_feature_base"].endswith(
        "/m1_feature_base.parquet"
    )
    assert status["outputs"]["m5_entry_feature_base"].endswith(
        "/m5_feature_base.parquet"
    )
    assert re.fullmatch(r"[0-9a-f-]{36}", status["boot_id"])
    assert status["chain_pid"] > 0
    terminal = Path(status["terminal_event_path"])
    assert terminal.is_file()
    assert json.loads(terminal.read_text(encoding="utf-8")) == status


def test_registry_fit_train_window_is_bound_to_the_chain_split_authority() -> None:
    """Both TRAIN-fit owners must be checked against the chain's own windows.

    The volatility-squeeze artifacts are a chain INPUT, so their declared TRAIN
    window is compared at contract-validation. The V29 registry operators are
    fitted INSIDE the chain, so their equivalent boundary is the first moment
    the frozen payload exists: right after each enriched lane publishes. Before
    2026-08-15 the registry side had no such check at all — it carried a
    split-manifest pointer instead, which could never be satisfied because the
    chain produces split manifests downstream of this fit.
    """

    source = SCRIPT.read_text(encoding="utf-8")

    for required in (
        # Operator-input bound: the registry fit may not reach past TRAIN.
        "registry fit TRAIN end must not exceed the chain's declared --train-end",
        # One owner for the published-artifact equality check, both lanes.
        "require_registry_fit_windows() {",
        'require_registry_fit_windows M5 "$MTF/manifest.json"',
        'require_registry_fit_windows M1 "${M1_ENRICHED}.manifest.json"',
        # Validated through the payload owners, not by re-reading raw keys.
        "require_v29_registry_constants",
        "require_v29_registry_m1_lane_params",
        "V29_REGISTRY_{lane}_CHAIN_TRAIN_WINDOW_MISMATCH",
        # BOTH declared bounds are compared, not only the end. The lower bound
        # was missing from the fit owners entirely until 2026-08-15.
        'declared_train_window_start',
        'pd.Timestamp(raw_train_start)',
        # The hash-bound pair pointer the fit freezes is re-checked against
        # the pair authority this chain validated at pair-authority.
        "V29_REGISTRY_{lane}_CHAIN_PAIR_BINDING_MISMATCH",
        '--expected-pair-manifest-sha256 "$PAIR_MANIFEST_SHA256"',
        # The squeeze equality check stays exactly where it was.
        "VOLATILITY_SQUEEZE_CHAIN_TRAIN_OR_PAIR_LINEAGE_MISMATCH",
    ):
        assert required in source, required

    # Both enriched lanes must supply the pair hash the producer now requires.
    assert source.count('--expected-pair-manifest-sha256 "$PAIR_MANIFEST_SHA256"') == 2

    # The M5 lane must be checked before the M5 model source is materialized,
    # and the M1 lane before the feature surfaces consume it.
    assert source.index(
        'require_registry_fit_windows M5 "$MTF/manifest.json"'
    ) < source.index("CURRENT_STEP=m5-model-source")
    assert source.index(
        'require_registry_fit_windows M1 "${M1_ENRICHED}.manifest.json"'
    ) < source.index("CURRENT_STEP=entry-exit-feature-surfaces")


def _required_flags(relative: str) -> set[str]:
    """Every ``add_argument(..., required=True)`` flag a producer declares."""

    import ast

    tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if not any(
            keyword.arg == "required"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        ):
            continue
        for argument in node.args:
            if (
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and argument.value.startswith("--")
            ):
                flags.add(argument.value)
    return flags


def test_chain_supplies_every_argument_the_enriched_producer_requires() -> None:
    """A required flag the chain never passes makes the producer unrunnable.

    That is exactly what ``--registry-fit-split-manifest`` was on
    ``prebuild_multi_tf_cache_v4.py``: declared ``required=True`` while no
    caller supplied it. Derive the required set from the producer's own parser
    (rule 13: never restate the flag list here) and prove each chain route
    passes it.
    """

    source = SCRIPT.read_text(encoding="utf-8")
    required = _required_flags(
        "gx1/scripts/build_entry_exit_m1_enriched_frame_v1.py"
    )
    assert required

    for route in ("model-native-m5-enriched-frame", "model-native-m1-enriched-frame"):
        start = source.index(route)
        end = source.index('>>"$LOG" 2>&1', start)
        invocation = source[start:end]
        missing = sorted(flag for flag in required if flag not in invocation)
        assert not missing, f"{route}: chain never passes {missing}"


def test_no_producer_route_binds_a_split_manifest_it_runs_before() -> None:
    """The retired split pointers may not re-enter any chain-facing surface.

    Derived from the owners' retired-key constants (rule 13: never restate the
    key names here).
    """

    from gx1.contracts.registry_hyperparameter_fit_v1 import RETIRED_SOURCE_KEYS
    from gx1.features.volatility_squeeze_state_v1 import (
        RETIRED_TRAIN_LINEAGE_KEYS,
    )

    retired_keys = RETIRED_SOURCE_KEYS | RETIRED_TRAIN_LINEAGE_KEYS
    assert retired_keys

    # Producers and control surfaces only. The two contract owners are excluded
    # on purpose: they are where the retired-key constants are declared.
    for relative in (
        "scripts/run_seq513_rebuild_chain_v1.sh",
        "scripts/entry_next_edge_control.sh",
        "gx1/scripts/prebuild_multi_tf_cache_v4.py",
        "gx1/scripts/build_entry_exit_m1_enriched_frame_v1.py",
        "gx1/scripts/benchmark_level_registry_v1.py",
        "gx1/features/level_registry_v1.py",
        "gx1/features/trendline_registry_v1.py",
        "gx1/features/htf_features.py",
    ):
        text = (REPO / relative).read_text(encoding="utf-8")
        for key in sorted(retired_keys):
            assert key not in text, f"{relative}: retired lineage key {key}"

    control = (REPO / "scripts/entry_next_edge_control.sh").read_text(
        encoding="utf-8"
    )
    # The control surface may name an ordinary split manifest for its
    # reconstruction audit. Only the retired *registry-fit* pointer is
    # forbidden: it would bind a producer to an artifact produced downstream.
    assert "--registry-fit-split-manifest" not in control
    # The tape and pair bindings are real source authority and must remain.
    assert "--tape-manifest" in control
    assert "--pair-manifest" in control


def test_chain_enforces_the_d1_receptive_field_warmup_from_its_owner() -> None:
    """--history-start must clear the widest per-TF receptive field, not 96 M5 rows.

    The 96-row check only covers the local M5 sequence. The dominant warmup is
    the D1 lane of PRODUCTION_MTF_PER_TF_WINDOW_BARS, and nothing checked it
    until 2026-08-19, so a --history-start that passed could still leave the
    first TRAIN rows with an incomplete daily receptive field.
    """

    from gx1.contracts.entry_exit_production_architecture_v1 import (
        PRODUCTION_MTF_PER_TF_WINDOW_BARS,
    )

    source = SCRIPT.read_text(encoding="utf-8")

    # Rule 13: the width is imported and evaluated, never restated as a literal.
    assert (
        "from gx1.contracts.entry_exit_production_architecture_v1 import" in source
    )
    assert "PRODUCTION_MTF_PER_TF_WINDOW_BARS" in source
    assert (
        'required_d1_warmup_bars = int(dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS)["D1"])'
        in source
    )
    required = int(dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS)["D1"])
    assert f"< {required}" not in source
    assert f"= {required}" not in source

    # Bars are counted on the V4 closed-D1 axis, never as calendar days: the row
    # clock skips weekends and closures, so a day rule overstates the warmup.
    assert (
        "from gx1.features.htf_features import "
        "build_multi_tf_v4_closed_timestamp_indices" in source
    )
    assert 'build_multi_tf_v4_closed_timestamp_indices(times.as_unit("ns"))["D1"]' in (
        source
    )
    assert "(d1_axis >= history_start) & (d1_axis < train_start)" in source

    # The pre-existing local M5 sequence check is kept, not replaced.
    assert "pre_train_rows < 96" in source
