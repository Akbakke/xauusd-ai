from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts/windows/GX1-RandomAccessCampaignV2Controller.ps1"
OBSERVER = ROOT / "scripts/windows/GX1-RandomAccessCampaignV2Progress.ps1"
INSTALLER = ROOT / "scripts/windows/Install-GX1RandomAccessCampaignV2.ps1"


def test_controller_uses_physical_reboot_and_transactional_cli() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    assert "shutdown.exe /r" in source
    assert "shutdown.exe /r /t 60" in source
    assert "'prepare-reboot'" in source
    assert "'confirm-reboot'" in source
    assert source.index("'begin'") < source.index("Start-Process -FilePath 'wsl.exe'")
    assert source.index("'record'") < source.rindex("Request-Gx1PhysicalReboot")
    assert "trainer.ExitCode" in source
    assert "observer.ExitCode" in source
    assert "$env:GX1_CAMPAIGN_PLAN_SHA256" in source
    assert "$env:GX1_CAMPAIGN_PLAN_PATH" in source
    assert "$env:GX1_CAMPAIGN_PLAN_FILE_SHA256" in source
    assert "$env:GX1_CAMPAIGN_INVOCATION_SHA256" in source
    assert "$env:GX1_CAMPAIGN_GUARD_LOG_PATH" in source
    assert "$env:WSLENV" in source
    assert "Campaign plan must be an absolute WSL path" in source
    assert "nvidia-smi -pl" not in source.lower()


def test_controller_accepts_only_plan_bound_local_windows_staging() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    assert "[Alias('WindowsSourceRepo')]" in source
    assert "WindowsControllerSourceRoot" in source
    assert "Windows controller source root must be an existing local C:" in source
    assert "[IO.FileAttributes]::ReparsePoint" in source
    assert "expectedControllerSource" in source
    assert "Running campaign controller is outside the explicit staging root" in source
    assert "controllerBinding.sha256" in source
    assert "observerBinding.sha256" in source
    assert "does not map to the exact source-bound WSL repository" not in source
    assert "Convert-Gx1WslPath -LinuxPath ([string]$invocation.progress_path)" in source


def test_boot_identity_atomically_replaces_existing_destination_with_backup() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    assert "$backup = $path + '.backup.' + [guid]::NewGuid().ToString('N')" in source
    assert "[IO.File]::Replace($temporary, $path, $backup)" in source
    assert "[IO.File]::Replace($temporary, $path, $null)" not in source
    assert source.count("Remove-Item -LiteralPath $backup -Force") == 2


def test_boot_identity_atomically_moves_new_destination_and_cleans_temp() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    existing = source.index("if (Test-Path -LiteralPath $path)")
    replace = source.index("[IO.File]::Replace($temporary, $path, $backup)")
    move = source.index("[IO.File]::Move($temporary, $path)")
    cleanup = source.index("if (Test-Path -LiteralPath $temporary)")
    assert existing < replace < move < cleanup


def test_progress_sidecar_is_token_free_and_not_a_gpu_safety_owner() -> None:
    source = OBSERVER.read_text(encoding="utf-8")
    lowered = source.lower()
    assert "Start-Sleep -Seconds 1" in source
    assert "TotalSeconds -ge 900" in source
    assert "codex_polling_required = $false" in source
    assert "gx1_guarded_trainer_exec.sh" in source
    assert "nvidia-smi" not in lowered
    assert "Stop-Process" not in source
    assert "KILL_AND_BLOCK" not in source


def test_installer_uses_explicit_wsl_owner_and_startup_only() -> None:
    source = INSTALLER.read_text(encoding="utf-8")
    assert "New-ScheduledTaskTrigger -AtStartup" in source
    assert "-UserId $WindowsTaskUser -LogonType S4U" in source
    assert "must run as the explicit Windows account" in source
    assert "MultipleInstances IgnoreNew" in source
    assert "RestartCount 3" in source
    assert "Start-ScheduledTask" not in source
    assert "first_launch_requires_next_windows_boot = $true" in source
