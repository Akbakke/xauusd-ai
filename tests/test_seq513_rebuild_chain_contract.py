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
        "--exit-target-lookahead-m1-steps",
        "--early-move-threshold-bps",
        "--level-tol-quantile-q",
        "--registry-fit-train-end",
        "REGISTRY_FIT_TRAIN_END=$TRAIN_END",
        'path.relative_to(event)',
        'feature ranking output must be a fresh timestamped JSON',
        'feature ranking timestamp cannot be in the future',
        'source does not cover the declared common history/test window',
        'pre_train_rows < 96',
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
        '--rank-source-parquet "$CV2"',
        "model-native-m5-enriched-frame",
        "model-native-m1-enriched-frame",
        "model-native-m5-feature-base",
        "model-native-m1-feature-base",
        '--level-tol-quantile-q "$LEVEL_TOL_QUANTILE_Q"',
        '--registry-fit-train-end "$REGISTRY_FIT_TRAIN_END"',
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

    assert source.count('--feature-ranking-json "$RANKING"') == 3
    assert source.count('--run-id "$RUN_ID"') == 6
    assert "materialize_model_native_train_rank_reference_v2" in source
    assert "materialize_entry_model_native_train_feature_ranker_v1" in source
    assert '--rank-reference-npz "$RANK_NPZ"' in source
    assert '--existing-rank-reference' in source
    assert '--out "$RANKING"' in source
    assert "gx1_capped_run.sh --class audit --mem 4G --swap 512M" in source
    assert source.index("materialize_model_native_train_rank_reference_v2") < source.index(
        "materialize_entry_model_native_train_feature_ranker_v1"
    )
    for forbidden in (
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
    assert source.count("require_source_identity") == 17
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
    assert '"schema_version": "seq513_rebuild_chain_status_v7"' in source
    assert '"next_boundary": (' in source
    assert '"unified_exit_lifecycle_dir": exit_lifecycle_dir' in source
    assert "stopped before post-rebuild audits/readiness" in source
    assert '"source_cascade": {' in source
    assert '"rank_reference": {' in source
    assert '"pair_authority": {' in source
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
    assert "--early-move-threshold-bps" in help_result.stdout

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
            "--exit-target-lookahead-m1-steps",
            "30",
            "--early-move-threshold-bps",
            "4.0",
            # Explicit recipe input for the V29 registry TRAIN fit;
            # --registry-fit-train-end is deliberately omitted so the probe
            # also exercises its one-origin default (the chain's --train-end).
            "--level-tol-quantile-q",
            "0.5",
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
    assert status["schema_version"] == "seq513_rebuild_chain_status_v7"
    assert status["next_boundary"] == (
        "REPAIR_CURRENT_FAILED_STEP_WITHOUT_REUSING_PARTIAL_OUTPUT"
    )
    assert status["outputs"]["unified_exit_lifecycle_dir"].endswith(
        "/exit_lifecycle"
    )
    assert status["source_cascade"]["path"].endswith(".json")
    assert status["rank_reference"]["path"].endswith(
        "model_native_train_rank_reference_v4.npz"
    )
    assert status["rank_reference"]["sidecar_path"].endswith(
        "model_native_train_rank_reference_v4.npz.json"
    )
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
