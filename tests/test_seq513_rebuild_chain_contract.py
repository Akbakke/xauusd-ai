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
        "--vedtak",
        "--event-root",
        "--feature-ranking-json",
        "--signal-manifest",
        "--preflight-out-dir",
        'path.relative_to(event)',
        'signal manifest output must be a fresh timestamped JSON',
        'preflight output directory must be fresh',
        'fresh event outputs required',
        'preflight namespace must contain exactly one artifact',
        'preflight json_path is not an exact self-reference',
        'preflight vedtak does not match chain vedtak',
        'hashlib.sha256(raw).hexdigest()',
    ):
        assert required in source

    assert source.count('--feature-ranking-json "$RANKING"') == 3
    assert source.count('--vedtak "$VEDTAK"') == 3
    for forbidden in (
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
    assert source.count("require_source_identity") == 5
    assert 'repository HEAD changed after binding' in source
    assert 'repository worktree changed after binding' in source
    assert '"git_head": git_head or None' in source
    assert 'trap \'on_err "$?" "$LINENO"\' ERR' in source
    assert "trap 'on_signal TERM 143' TERM" in source
    assert "trap 'on_signal INT 130' INT" in source
    assert "trap 'on_signal HUP 129' HUP" in source
    assert 'terminal_status ABORTED "received $signal_name" "$exit_code"' in source
    assert 'terminal_status RED "unexpected ERR at line $line" "$exit_code"' in source
    assert '"explicit_vedtak_id": vedtak' in source
    assert '"step": step' in source
    assert '"state": state' in source
    assert "os.replace(temporary, path)" in source


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
    assert "--signal-manifest" in help_result.stdout
    assert "--preflight-out-dir" in help_result.stdout

    old = subprocess.run(
        [str(SCRIPT), "XAU_SEQ513_REBUILD_UNIT_V1", "/tmp/unit-event"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert old.returncode == 2
    assert "unknown argument" in old.stderr


def test_chain_validation_failure_persists_red_vedtak_and_revision(
    tmp_path: Path,
) -> None:
    event = (tmp_path / "event").resolve()
    event.mkdir()
    ranking = event / "ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING_20260719T200000000000Z.json"
    ranking.write_text("{}\n", encoding="utf-8")
    outside = (tmp_path / "outside").resolve()
    manifest = outside / "ENTRY_MODEL_NATIVE_SEQ513_SIGNAL_MANIFEST_20260719T200001000000Z.json"
    preflight = event / "preflight-20260719T200002000000Z"
    vedtak = "XAU_SEQ513_REBUILD_UNIT_V1"
    env = dict(os.environ)
    # Prevent the existing notifier from loading repository credentials during
    # this deliberate fail-closed probe. An empty chat ID makes send() a no-op.
    env["GX1_TELEGRAM_BOT_TOKEN"] = "unit-test-disabled"
    env["GX1_TELEGRAM_CHAT_ID"] = ""

    result = subprocess.run(
        [
            str(SCRIPT),
            "--vedtak",
            vedtak,
            "--event-root",
            str(event),
            "--feature-ranking-json",
            str(ranking),
            "--signal-manifest",
            str(manifest),
            "--preflight-out-dir",
            str(preflight),
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
    # checkout reaches the intentionally out-of-event manifest rejection.
    assert status["step"] in {"source-revision", "contract-validation"}
    assert status["explicit_vedtak_id"] == vedtak
    assert status["event_root"] == str(event)
    assert re.fullmatch(r"[0-9a-f]{40}", status["git_head"])
    assert Path(status["log_path"]).is_file()
    assert status["exit_code"] == 2
