from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
INSTALLER = REPO / "scripts" / "windows" / "Install-GX1-HostTelemetryBridge.ps1"
SENSOR_INSTALLER = REPO / "scripts" / "windows" / "Install-GX1-HostTelemetry.ps1"
GPU_IDLE_GUARD = REPO / "scripts" / "windows" / "GX1-GpuPowerAndIdleGuard.ps1"


def test_host_bridge_installer_keeps_the_signer_host_only_and_nonexportable() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    assert "Assert-Administrator" in source
    assert "Cert:\\LocalMachine\\My" in source
    assert "KeyExportPolicy = 'NonExportable'" in source
    assert "-UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest" in source
    assert "DirectorySecurity" in source
    assert "FileSystemAccessRule" in source
    assert "S-1-5-18" in source
    assert "S-1-5-32-544" in source
    assert "S-1-5-32-545" in source
    assert "WindowsIdentity]::GetCurrent().User.Value" in source
    assert "'/T', '/C'" in source
    assert "public_certificate_sha256" in source
    assert "public_certificate_wsl_path" in source
    assert "GX1-HostTelemetryBridgeRunner.ps1" in source
    assert "GX1-HostTelemetryBridgeService.log" in source
    assert "runner-start" in source
    assert "runner-fatal:" in source
    assert "BridgeDirectoryName" in source
    assert "A-Za-z0-9_-" in source
    assert "Diagnostic output:" in source
    assert "Repair-ExistingBridgeDirectoryAccess" in source
    assert "'/reset', '/T', '/C'" in source
    assert source.index("Repair-ExistingBridgeDirectoryAccess -BridgeRoot $bridgeRoot") < source.index(
        "$certificate = Get-BridgeCertificate"
    )
    assert "Stop-ExistingBridgeTask" in source
    assert "Wait-ForExclusiveFileAccess" in source
    assert source.index("Stop-ExistingBridgeTask -TaskName 'GX1HostTelemetryBridge'") < source.index(
        "New-Item -ItemType Directory -Path $bridgeRoot -Force"
    )
    assert source.index("Wait-ForExclusiveFileAccess -Path $serviceLogPath") < source.index(
        "[System.IO.File]::WriteAllText(\n    $servicePath"
    )


def test_host_bridge_wsl_transport_is_opt_in_and_single_client_restricted() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    assert "[string]$WslClientAddress = ''" in source
    assert "[int]$WslPort = 38128" in source
    assert "WslPort must differ from the loopback bridge Port" in source
    assert "Get-WslGatewayIpv4" in source
    assert "Test-Ipv4SameSubnet" in source
    assert 'vEthernet (WSL*' in source
    assert '"http://${wslListenAddress}:$WslPort/gx1/v1/telemetry/"' in source
    assert "'interface', 'portproxy', 'add', 'v4tov4'" in source
    assert '"listenaddress=$wslListenAddress"' in source
    assert '"listenport=$WslPort"' in source
    assert "'connectaddress=127.0.0.1'" in source
    assert '"connectport=$Port"' in source
    assert "New-NetFirewallRule @firewallArguments" in source
    assert "RemoteAddress = $wslClientAddressCanonical" in source
    assert "LocalAddress = $wslListenAddress" in source
    assert "EdgeTraversalPolicy = 'Block'" in source
    assert "wsl_endpoint" in source
    assert 'Prefixes.Add("http://0.0.0.0:' not in source


def test_host_bridge_service_stays_loopback_nonce_bound_and_sensor_complete() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    assert 'http://127.0.0.1:$Port/gx1/v1/telemetry/' in source
    assert "v4tov4_portproxy_to_windows_loopback" in source
    assert "request_nonce,schema_version" in source
    assert "'^[0-9a-f]{64}$'" in source
    assert "GPU Memory Junction" in source
    assert "--query-gpu=name,uuid,temperature.gpu,power.draw,power.limit,memory.used" in source
    assert "RSASignaturePadding]::Pkcs1" in source
    assert "gx1_host_gpu_telemetry_v1" in source
    assert "memory_temp_c" in source
    assert "observed_monotonic_ms" in source
    assert "untrusted" not in source.lower()


def test_host_bridge_installer_never_changes_the_gpu_power_limit() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    assert " -pl " not in source
    assert "SetPowerLimit" not in source


def test_sensor_bootstrap_uses_the_same_160_w_limit_as_canonical_cuda() -> None:
    source = SENSOR_INSTALLER.read_text(encoding="utf-8")

    assert "[ValidateRange(0, 160)]" in source
    assert "canonical_ready = ($powerLimit -le 160.0)" in source
    assert "at or below 160 W" in source
    assert "SetPowerLimitWatts 250" not in source


def test_sensor_bootstrap_registers_a_verified_persistent_160_w_startup_task() -> None:
    source = SENSOR_INSTALLER.read_text(encoding="utf-8")
    guard = GPU_IDLE_GUARD.read_text(encoding="utf-8")

    assert "function Install-PersistentPowerLimitTask" in source
    assert "$taskName = 'GX1GpuPowerLimit'" in source
    assert "New-ScheduledTaskTrigger -AtStartup" in source
    assert "-UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest" in source
    assert "expected_gpu_uuid" in source
    assert "name,uuid,pstate,temperature.gpu,power.draw,power.limit,memory.used,utilization.gpu" in guard
    assert "-pl" in source
    assert "recheck_seconds = 900" in source
    assert "-ExecutionTimeLimit (New-TimeSpan -Seconds 0)" in source
    assert "-RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)" in source
    assert "persistent_power_limit_task" in source
    assert "Install-PersistentPowerLimitTask @persistentTaskParameters" in source


def test_gpu_idle_guard_detects_only_sustained_low_memory_zero_load_high_power() -> None:
    installer = SENSOR_INSTALLER.read_text(encoding="utf-8")
    guard = GPU_IDLE_GUARD.read_text(encoding="utf-8")

    assert "gx1_gpu_power_and_idle_guard_v2" in installer
    assert "GX1-GpuPowerAndIdleGuard.ps1" in installer
    assert "Copy-Item -LiteralPath $runnerSourcePath" in installer
    assert "sample_seconds = 5" in installer
    assert "idle_power_threshold_w = 60" in installer
    assert "idle_memory_max_mib = 384" in installer
    assert "idle_utilization_max_percent = 2" in installer
    assert "idle_required_samples = 24" in installer
    assert "Test-Gx1HighIdleSample" in guard
    assert "$Sample.pstate -match '^P[0-2]$'" in guard
    assert "$Sample.power_draw_w -gt [double]$Config.idle_power_threshold_w" in guard
    assert "$Sample.memory_used_mib -le [int]$Config.idle_memory_max_mib" in guard
    assert "$Sample.utilization_percent -le [int]$Config.idle_utilization_max_percent" in guard
    assert "$highIdleSamples -ge [int]$config.idle_required_samples" in guard


def test_gpu_idle_guard_recovers_exact_device_and_fails_closed() -> None:
    installer = SENSOR_INSTALLER.read_text(encoding="utf-8")
    guard = GPU_IDLE_GUARD.read_text(encoding="utf-8")

    assert "Get-PnpDevice -PresentOnly -Class Display" in installer
    assert "gpu_pnp_instance_id" in installer
    assert "expected_gpu_uuid" in guard
    assert "pnputil.exe" in guard
    assert "'/restart-device'" in guard
    assert "Set-Gx1PowerLimit -Config $Config" in guard
    assert "Stop-Gx1TelemetryBridge" in guard
    assert "Start-Gx1TelemetryBridge" in guard
    assert "GX1-GpuIdleGuard.block.json" in guard
    assert "GPU_IDLE_RECOVERY_FAILED" in guard
    assert "GPU_IDLE_RECOVERY_RATE_LIMITED" in guard
    assert "GPU_IDLE_RECOVERY_COOLDOWN" in guard
    assert "telemetry_bridge=STOPPED" in guard
    assert "GUARD_FAIL_CLOSED_ACTIVE telemetry_bridge=STOPPED" in guard
    assert "CONFIG_FAILURE message=$configurationFailure telemetry_bridge=STOPPED" in guard
    assert "BLOCK_CLEARED_AFTER_NORMAL_IDLE" not in guard
    assert "recovery_cooldown_seconds = 1800" in installer
    assert "max_recoveries_per_window = 2" in installer


def test_gpu_idle_guard_has_offline_policy_self_test_and_one_shot_probe() -> None:
    guard = GPU_IDLE_GUARD.read_text(encoding="utf-8")

    assert "[switch]$PolicySelfTest" in guard
    assert "[switch]$Once" in guard
    assert "gx1_gpu_idle_guard_policy_self_test_v1" in guard
    assert "persistent_high_idle_detected = $true" in guard
    assert "active training was misclassified" in guard
    assert "initialized CUDA context was misclassified" in guard
    assert "small active CUDA workload was misclassified" in guard
    assert "normal sample did not reset the sustained counter" in guard
    assert "gx1_gpu_power_and_idle_guard_once_v1" in guard
    assert "exit 0" not in guard
