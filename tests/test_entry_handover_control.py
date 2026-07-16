import subprocess
from pathlib import Path


REPO = Path("/home/andre2/src/GX1_ENGINE")
HANDOVER = REPO / "HANDOVER_XAU_DIRECTION_REPAIR_20260714.md"
HANDOVER_VIEWER = REPO / "scripts/gx1_handover.sh"
CONTROL = REPO / "scripts/entry_next_edge_control.sh"


def test_handover_viewer_points_to_current_xau_direction_repair_truth() -> None:
    text = HANDOVER_VIEWER.read_text(encoding="utf-8")

    assert "HANDOVER_XAU_DIRECTION_REPAIR_20260714.md" in text
    assert "Use this script only: scripts/gx1_handover.sh" in text
    assert "trading bot for gold/XAUUSD" in text
    assert "GX1_ALLOW_LEGACY_HANDOVER" not in text
    assert "SMART JOINT POLICY PROMOTED" not in text


def test_only_one_handover_shell_entrypoint_exists() -> None:
    handover_scripts = sorted((REPO / "scripts").glob("*handover*.sh"))

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
    assert "# XAUUSD Direction Repair Handover - 2026-07-14" in result.stdout
    assert "Build the GX1 trading bot for gold/XAUUSD" in result.stdout
    assert "Use this script only: scripts/gx1_handover.sh" in result.stdout
    assert "Continue the XAUUSD-only direction repair" in result.stdout
    assert "Do not use non-XAU project artifacts" in result.stdout


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
    assert HANDOVER.read_text(encoding="utf-8").splitlines()[0] in result.stdout
    assert "SMART JOINT POLICY PROMOTED" not in result.stdout
