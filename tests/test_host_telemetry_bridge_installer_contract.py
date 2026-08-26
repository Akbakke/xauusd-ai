from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
INSTALLER = REPO / "scripts" / "windows" / "Install-GX1-HostTelemetryBridge.ps1"


def test_host_bridge_installer_keeps_the_signer_host_only_and_nonexportable() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    assert "Assert-Administrator" in source
    assert "Cert:\\LocalMachine\\My" in source
    assert "KeyExportPolicy = 'NonExportable'" in source
    assert "-UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest" in source
    assert "BUILTIN\\Users" in source
    assert "SYSTEM:(OI)(CI)F" in source
    assert "BUILTIN\\Administrators:(OI)(CI)F" in source
    assert "BUILTIN\\Users:(OI)(CI)RX" in source
    assert "'/T', '/C'" in source
    assert "public_certificate_sha256" in source
    assert "public_certificate_wsl_path" in source


def test_host_bridge_service_is_loopback_nonce_bound_and_sensor_complete() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    assert 'http://127.0.0.1:$Port/gx1/v1/telemetry/' in source
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
