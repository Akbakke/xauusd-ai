import subprocess
import json
from pathlib import Path


REPO = Path("/home/andre2/src/GX1_ENGINE")
WRAPPER = REPO / "scripts/run_entry_foundation_iql_distill.sh"


def _run_wrapper(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(WRAPPER), *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_iql_distill_requires_explicit_vedtak() -> None:
    result = _run_wrapper("--dry-run")

    assert result.returncode == 2
    assert "--vedtak is required" in result.stderr
    assert "IQL distillation contract command:" not in result.stdout


def test_iql_distill_blocks_when_replay_readiness_is_not_ready() -> None:
    result = _run_wrapper("--vedtak", "PYTEST_DRY_RUN", "--dry-run")

    assert result.returncode == 2
    assert "replay-readiness is NOT_READY" in result.stderr
    assert "No IQL distillation, adapter, promotion, shadow, or live path was started" in result.stderr
    assert "IQL distillation contract command:" not in result.stdout


def test_iql_distill_materialize_only_writes_not_ready_contract(tmp_path: Path) -> None:
    result = _run_wrapper(
        "--vedtak",
        "PYTEST_MATERIALIZE_ONLY",
        "--materialize-only",
        "--out-dir",
        str(tmp_path),
    )

    latest = tmp_path / "ENTRY_IQL_DISTILLATION_CONTRACT_latest.json"
    report = json.loads(latest.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert latest.exists()
    assert report["decision"] == "ENTRY_IQL_DISTILLATION_CONTRACT_NOT_READY"
    assert report["iql_research_distillation_allowed"] is False
    assert report["promotion_shadow_live_allowed"] is False
    assert "IQL distillation contract written under:" in result.stdout


def test_iql_distill_wrapper_declares_safe_no_fail_report_mode() -> None:
    text = WRAPPER.read_text(encoding="utf-8")

    assert "--no-fail-on-not-ready" in text
    assert "--materialize-only" in text
    assert "NO_FAIL_ON_NOT_READY" in text
