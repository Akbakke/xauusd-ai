from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from gx1.contracts.rebuild_dependency_readiness_v1 import (
    REBUILD_DEPENDENCY_READINESS_SCHEMA_VERSION,
    build_rebuild_dependency_readiness,
)


REPO = Path(__file__).resolve().parents[1]
CHAIN = REPO / "scripts" / "run_seq513_rebuild_chain_v1.sh"


def test_current_pinned_environment_is_importable_without_data_or_cuda() -> None:
    report = build_rebuild_dependency_readiness(repo=REPO)

    assert report["schema_version"] == REBUILD_DEPENDENCY_READINESS_SCHEMA_VERSION
    assert report["decision"] == "PASS"
    assert report["failures"] == []
    assert report["packages"]
    assert all(row["ok"] is True for row in report["packages"].values())
    assert report["packages"]["torch"]["import_module"] == "torch"


def test_version_or_import_mismatch_fails_before_a_rebuild_can_start() -> None:
    report = build_rebuild_dependency_readiness(
        repo=REPO,
        version_lookup=lambda _distribution: "0.0.0",
        importer=lambda _module: object(),
    )

    assert report["decision"] == "FAIL"
    assert "dependency_not_ready:numpy" in report["failures"]
    assert report["packages"]["numpy"]["ok"] is False


def test_chain_runs_dependency_gate_before_pair_authority() -> None:
    source = CHAIN.read_text(encoding="utf-8")

    gate = '"$PY" -m gx1.scripts.verify_rebuild_dependency_readiness_v1'
    assert gate in source
    assert 'CURRENT_STEP=dependency-readiness' in source
    assert source.index(gate) < source.index("# One pair manifest owns canonical M5")


def test_readiness_cli_succeeds_against_the_current_repository() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gx1.scripts.verify_rebuild_dependency_readiness_v1",
            "--repo",
            str(REPO),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"decision": "PASS"' in result.stdout
