from __future__ import annotations

import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_cloud_deadline_installer_is_fail_closed_and_provider_deletes() -> None:
    path = REPO / "scripts" / "install_gx1_cloud_deadline_guard.sh"
    result = subprocess.run(
        ["bash", "-n", str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    source = path.read_text(encoding="utf-8")
    for required in (
        "must run as root",
        "provider-delete-command-sha256",
        "delta > 48 * 60 * 60",
        "provider_managed_delete",
        "hard_cost_cap_nok",
        "OnCalendar=$deadline_utc",
        "Persistent=yes",
        "systemctl enable --now gx1-cloud-deadline.timer",
        "systemctl poweroff --force --no-wall",
    ):
        assert required in source


def test_host_profile_materializer_has_no_training_or_test_surface() -> None:
    path = REPO / "gx1" / "scripts" / "materialize_cloud_training_host_profile_v1.py"
    source = path.read_text(encoding="utf-8")
    assert "exactly one physical GPU" in source
    assert "verify_runtime=True" in source
    assert "PASS_HOST_QUALIFIED_NOT_TRAINING_AUTHORITY" in source
    assert "--output-json" in source
    assert "--train" not in source
    assert "test_parquet" not in source.lower()
