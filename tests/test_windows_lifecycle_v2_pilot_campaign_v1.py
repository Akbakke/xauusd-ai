from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts/windows/GX1-LifecycleV2PilotController.ps1"
TELEMETRY = ROOT / "scripts/windows/GX1-LifecycleV2PilotTelemetry.ps1"
INSTALLER = ROOT / "scripts/windows/Install-GX1LifecycleV2PilotResume.ps1"


def test_controller_is_gate_first_and_never_changes_power() -> None:
    source = CONTROLLER.read_text()
    lowered = source.lower()
    assert source.index("'inspect'") < source.index("Start-Process -FilePath 'wsl.exe'")
    assert "automatic_power_limit_change -ne $false" in source
    assert "limit -ne 160.0" in source and "draw -gt 160.0" in source
    assert "nvidia-smi -pl" not in lowered
    assert "gx1scopedkeeper" not in lowered
    assert "applygpupowerlimit" not in lowered
    assert "--test" not in lowered


def test_local_telemetry_is_one_second_and_human_status_is_15_minutes() -> None:
    source = TELEMETRY.read_text()
    assert "Start-Sleep -Seconds 1" in source
    assert "TotalSeconds -ge 900" in source
    assert "codex_polling_required = $false" in source
    assert "KILL_AND_BLOCK" in source
    assert "Stop-Process -Id $TrainerProcessId" in source
    assert "power.limit,power.draw" in source
    assert "nvidia-smi -pl" not in source.lower()


def test_task_scheduler_resumes_only_at_windows_startup() -> None:
    source = INSTALLER.read_text()
    assert "New-ScheduledTaskTrigger -AtStartup" in source
    assert "MultipleInstances IgnoreNew" in source
    assert "GX1-LifecycleV2PilotController.ps1" in source
    assert "Start-ScheduledTask" not in source
