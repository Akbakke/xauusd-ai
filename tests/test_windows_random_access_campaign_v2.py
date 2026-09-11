from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts/windows/GX1-RandomAccessCampaignV2Controller.ps1"
OBSERVER = ROOT / "scripts/windows/GX1-RandomAccessCampaignV2Progress.ps1"
INSTALLER = ROOT / "scripts/windows/Install-GX1RandomAccessCampaignV2.ps1"
HARDENING_TEST = ROOT / "tests/windows/Test-GX1-RandomAccessCampaignV2Controller.ps1"


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


def test_progress_sidecar_atomically_handles_new_and_existing_status() -> None:
    source = OBSERVER.read_text(encoding="utf-8")
    assert "$backup = $Path + '.backup.' + [guid]::NewGuid().ToString('N')" in source
    assert "[IO.File]::Replace($temporary, $Path, $backup)" in source
    assert "[IO.File]::Move($temporary, $Path)" in source
    assert "[IO.File]::Replace($temporary, $Path, $null)" not in source
    assert source.count("Remove-Item -LiteralPath $backup -Force") == 2
    assert "if (Test-Path -LiteralPath $temporary)" in source


def test_installer_uses_explicit_wsl_owner_and_startup_only() -> None:
    source = INSTALLER.read_text(encoding="utf-8")
    assert "New-ScheduledTaskTrigger -AtStartup" in source
    assert "-UserId $WindowsTaskUser -LogonType S4U" in source
    assert "must run as the explicit Windows account" in source
    assert "MultipleInstances IgnoreNew" in source
    assert "RestartCount 3" not in source
    assert "RestartInterval" not in source
    assert "[int]$registered.Settings.RestartCount -ne 0" in source
    assert "automatic_restart_count = [int]$registered.Settings.RestartCount" in source
    assert "$trigger.Delay = 'PT60S'" in source
    assert "Disable-ScheduledTask -TaskName 'WSL SSH Bootstrap'" in source
    assert "Legacy WSL SSH Bootstrap task could not be disabled" in source
    assert "'-ExpectedControllerSha256', $controllerSha256" in source
    assert "Start-ScheduledTask -TaskName $TaskName" not in source
    assert "first_launch_requires_next_windows_boot = $true" in source


def test_boot_identity_escapes_windows_path_only_at_wslpath_boundary() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    assignment = "$escapedPathForWsl = $path.Replace('\\', '\\\\')"
    invocation = (
        "Invoke-Gx1WslBounded -Arguments @('--', 'wslpath', '-u', "
        "$escapedPathForWsl)"
    )
    assert assignment in source
    assert invocation in source
    assert source.index(assignment) < source.index(invocation)
    assert "wslpath -u $path" not in source
    assert "$linux = @(& wsl.exe" not in source


def test_controller_refreshes_exact_v4_proxy_and_signed_probe_before_active() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    gate = "Confirm-Gx1SignedHostTelemetryReady -Status $status"
    begin = "$begin = Invoke-Gx1Json -Arguments @("
    assert gate in source
    assert source.index(gate) < source.index(begin)
    assert source.index(begin) < source.index("Start-Process -FilePath 'wsl.exe'")
    assert "ACTIVE_INVOCATION.json" in source
    assert (
        'interface portproxy delete v4tov4 "listenaddress=$($Bridge.ListenAddress)" '
        '"listenport=$($Bridge.ListenPort)" protocol=tcp'
    ) in source
    assert "interface portproxy add v4tov4" in source
    assert '"connectaddress=$($Bridge.ConnectAddress)"' in source
    assert '"connectport=$($Bridge.ConnectPort)"' in source
    assert "New-NetFirewallRule" not in source
    assert "nvidia-smi" not in source.lower()


def test_controller_fails_closed_on_exact_v4_host_or_source_mismatch() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    assert "C:\\ProgramData\\GX1\\HostTelemetryBridgeV4" not in source  # composed from ProgramData
    assert "'GX1\\HostTelemetryBridgeV4'" in source
    assert "'172.30.224.1'" in source
    assert "'172.30.231.75'" in source
    assert "'http://127.0.0.1:38127/gx1/v1/telemetry/'" in source
    assert "'http://172.30.224.1:38128/gx1/v1/telemetry/'" in source
    assert "'GX1HostTelemetryBridge'" in source
    assert "Get-NetTCPConnection -State Listen -LocalPort 38127" in source
    assert "Get-NetFirewallAddressFilter" in source
    assert "Get-NetFirewallPortFilter" in source
    assert "'--', '/usr/bin/sha256sum', $bridge.QueryPath" in source
    assert "$bridge.CertificateSha256, $bridge.GpuUuid, '2'" in source
    assert "$deadlineMilliseconds = 60000" in source
    assert "$nativeCallLimitMilliseconds = 8000" in source
    assert "$process.WaitForExit($TimeoutMilliseconds)" in source
    assert "$process.StandardOutput.ReadToEndAsync()" in source
    assert "$process.StandardError.ReadToEndAsync()" in source
    assert "[Threading.Tasks.Task]::WaitAll($outputTasks, $remaining)" in source
    assert "$process.WaitForExit()" not in source
    assert "$process.Kill()" in source
    assert "Native process argument is empty or requires forbidden quoting" in source
    assert "[Diagnostics.Stopwatch]::StartNew()" in source
    assert "WSL distro or user is unsafe for direct process arguments" in source
    assert "HostTelemetryBridgeV4 boot readiness failed after bounded retry" in source
    confirm_start = source.index("function Confirm-Gx1SignedHostTelemetryReady")
    wait_call = source.index(
        "Wait-Gx1HostTelemetryBridgeV4BootReady", confirm_start
    )
    exact_assert = source.index(
        "$bridge = Assert-Gx1HostTelemetryBridgeV4 -Status $Status", confirm_start
    )
    proxy_refresh = source.index(
        "Reset-Gx1HostTelemetryPortProxy -Bridge $bridge", confirm_start
    )
    signed_retry = source.index("foreach ($attempt in 1..12)", confirm_start)
    assert wait_call < exact_assert < proxy_refresh < signed_retry
    assert "foreach ($attempt in 1..12)" in source
    assert "Start-Sleep -Seconds 2" in source
    gate_call = source.index("Confirm-Gx1SignedHostTelemetryReady -Status $status")
    begin_call = source.index("$begin = Invoke-Gx1Json -Arguments @(")
    assert gate_call < begin_call



def test_bootstrap_failure_is_recorded_once_before_active_without_masking() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    receipt_call = "Write-Gx1BootstrapErrorReceipt -Stage $bootstrapStage"
    begin = "$begin = Invoke-Gx1Json -Arguments @("
    assert "gx1_campaign_bootstrap_error_receipt_v1" in source
    assert '"boot-$bootId-$PlanFileSha256.error.json"' in source
    assert "exception_type = [string]$Exception.GetType().FullName" in source
    assert "exception_message = [string]$Exception.Message" in source
    assert "controller_sha256 = (Get-FileHash" in source
    assert "Bootstrap error receipt controller path must be an absolute existing non-reparse file" in source
    assert "-ControllerPath $PSCommandPath" in source
    assert "Bootstrap error receipt already exists for this BootId and plan" in source
    assert source.index("$bootstrapStage = 'mutex'") < source.index(receipt_call) < source.index(begin)
    assert source.index("throw $bootstrapError", source.index(receipt_call)) < source.index(begin)

def test_windows_hardening_harness_covers_runtime_failure_modes() -> None:
    source = HARDENING_TEST.read_text(encoding="utf-8")
    assert "[Console]::Out.Write(('x' * 200000 -join ''))" in source
    assert "[Console]::Error.Write(('y' * 200000 -join ''))" in source
    assert "-TimeoutMilliseconds 100" in source
    assert "Bounded process timed out" in source
    assert "one_shot_initial_state" in source
    assert "POWERSHELL_HARDENING_PASS" in source

def test_single_wsl_boot_owner_is_source_bound_before_inspect() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    early = "$controllerSourceRoot = Assert-Gx1EarlyControllerIdentity"
    initial = "$initial = Get-Gx1InitialCampaignState"
    begin = "$begin = Invoke-Gx1Json -Arguments @("
    trainer = "Start-Process -FilePath 'wsl.exe'"
    assert "[Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ExpectedControllerSha256" in source
    assert r"Global\GX1RandomAccessCampaignV2WslBootstrap" in source
    assert source.index(early) < source.index(initial) < source.index(begin) < source.index(trainer)
    assert "Running campaign controller differs from installer-bound staged identity" in source
    start = source.index("function Get-Gx1InitialCampaignState {")
    end = source.index("function Wait-Gx1HostTelemetryBridgeV4BootReady {", start)
    function = source[start:end]
    assert function.count("Write-Gx1BootIdentity") == 1
    assert function.count("'inspect'") == 1
    assert function.count("-WslTimeoutMilliseconds 30000") == 1
    assert function.count("-TimeoutMilliseconds 30000") == 1
    for forbidden in ("while (", "Start-Sleep", "continue"):
        assert forbidden not in function
    for forbidden in ("Invoke-Gx1WslBootstrapRecovery", "Invoke-Gx1WslControlBounded", "Invoke-Gx1WslRecoveryProbe", "New-Gx1WslRecoveryIntent", "--terminate", "@('--shutdown')"):
        assert forbidden not in source


def test_cold_wsl_call_fits_real_boot_identity_parameter_range() -> None:
    import re

    source = CONTROLLER.read_text(encoding="utf-8")
    boot_start = source.index("function Write-Gx1BootIdentity {")
    boot = source[boot_start:source.index("\nfunction ", boot_start + 1)]
    lower, upper = map(int, re.search(r"ValidateRange\((\d+), (\d+)\)", boot).groups())
    initial_start = source.index("function Get-Gx1InitialCampaignState {")
    initial = source[initial_start:source.index("\nfunction ", initial_start + 1)]
    requested = int(re.search(r"Write-Gx1BootIdentity -WslTimeoutMilliseconds (\d+)", initial)[1])
    assert 1 <= lower <= requested <= upper <= 30000
    assert "boot_identity_parameter_binding=PASS" in HARDENING_TEST.read_text(encoding="utf-8")
