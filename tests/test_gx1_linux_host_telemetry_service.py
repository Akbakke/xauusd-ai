from __future__ import annotations

import base64
import http.client
import importlib.util
import json
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest


REPO = Path(__file__).resolve().parents[1]
SERVICE_PATH = REPO / "scripts" / "gx1_linux_host_telemetry_service.py"
INSTALLER_PATH = REPO / "scripts" / "install_gx1_linux_host_telemetry.sh"
UUID = "GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29"
NONCE = "a" * 64
ERROR_BODY = b'{"error":"telemetry_unavailable"}'


def _load_service() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "gx1_linux_host_telemetry_service", SERVICE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def service() -> ModuleType:
    return _load_service()


@pytest.fixture(scope="module")
def signing_material(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    directory = tmp_path_factory.mktemp("linux-host-telemetry-signing")
    private_key = directory / "private-key.pem"
    certificate = directory / "public-cert.pem"
    result = subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(private_key),
            "-out",
            str(certificate),
            "-sha256",
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=GX1-linux-host-telemetry-test",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    private_key.chmod(0o600)
    return private_key, certificate


def _fake_nvidia_smi(
    tmp_path: Path, output: str, *, sleep_seconds: float = 0.0
) -> tuple[Path, Path]:
    executable = tmp_path / "nvidia-smi"
    argument_log = tmp_path / "nvidia-smi.args"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys, time\n"
        f"pathlib.Path({str(argument_log)!r}).write_text('\\n'.join(sys.argv[1:]))\n"
        f"time.sleep({sleep_seconds!r})\n"
        f"sys.stdout.write({output!r})\n"
    )
    executable.chmod(0o755)
    return executable, argument_log


@contextmanager
def _running_service(
    service: ModuleType,
    private_key: Path,
    nvidia_smi: Path,
    *,
    sensor_timeout_seconds: float = 1.0,
) -> Iterator[tuple[str, int]]:
    config = service.ServiceConfig(
        gpu_uuid=UUID,
        private_key_path=private_key,
        nvidia_smi_path=nvidia_smi,
        openssl_path=Path(
            subprocess.check_output(["which", "openssl"], text=True).strip()
        ).resolve(),
        sensor_timeout_seconds=sensor_timeout_seconds,
        request_timeout_seconds=1.0,
    )
    server = service.create_server(config, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _request(
    address: tuple[str, int],
    *,
    method: str = "POST",
    path: str = "/gx1/v1/telemetry/",
    body: bytes | None = None,
    content_type: str = "application/json",
) -> tuple[int, dict[str, str], bytes]:
    if body is None:
        body = json.dumps(
            {
                "schema_version": "gx1_host_gpu_telemetry_request_v1",
                "request_nonce": NONCE,
            },
            separators=(",", ":"),
        ).encode("utf-8")
    connection = http.client.HTTPConnection(address[0], address[1], timeout=3)
    connection.request(method, path, body=body, headers={"Content-Type": content_type})
    response = connection.getresponse()
    response_body = response.read()
    headers = {key: value for key, value in response.getheaders()}
    connection.close()
    return response.status, headers, response_body


def _verify_signature(
    certificate: Path, response: dict[str, object], payload: bytes, tmp_path: Path
) -> None:
    signature_path = tmp_path / "signature.bin"
    public_key_path = tmp_path / "public-key.pem"
    payload_path = tmp_path / "payload.txt"
    signature_path.write_bytes(
        base64.b64decode(str(response["signature"]), validate=True)
    )
    payload_path.write_bytes(payload)
    public_key = subprocess.run(
        ["openssl", "x509", "-in", str(certificate), "-pubkey", "-noout"],
        capture_output=True,
        check=False,
    )
    assert public_key.returncode == 0, public_key.stderr.decode()
    public_key_path.write_bytes(public_key.stdout)
    verified = subprocess.run(
        [
            "openssl",
            "dgst",
            "-sha256",
            "-verify",
            str(public_key_path),
            "-signature",
            str(signature_path),
            str(payload_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr


def test_happy_path_uses_one_exact_sensor_row_and_signs_canonical_payload(
    service: ModuleType,
    signing_material: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    private_key, certificate = signing_material
    nvidia_smi, argument_log = _fake_nvidia_smi(
        tmp_path,
        f"{UUID}, 57, 64, 120.5, 700.0, 457\n",
    )
    with _running_service(service, private_key, nvidia_smi) as address:
        status, headers, raw_response = _request(address)

    assert status == 200
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    assert headers["Content-Length"] == str(len(raw_response))
    response = json.loads(raw_response)
    assert set(response) == {
        "schema_version",
        "request_nonce",
        "gpu_uuid",
        "core_temp_c",
        "memory_temp_c",
        "power_draw_w",
        "power_limit_w",
        "memory_used_mib",
        "observed_monotonic_ms",
        "signature",
    }
    assert response["schema_version"] == "gx1_host_gpu_telemetry_v1"
    assert response["request_nonce"] == NONCE
    assert response["gpu_uuid"] == UUID
    assert response["memory_used_mib"] == 457
    assert isinstance(response["observed_monotonic_ms"], int)
    payload = (
        "gx1_host_gpu_telemetry_v1\n"
        f"{NONCE}\n"
        f"{UUID}\n"
        "57.000000\n"
        "64.000000\n"
        "120.500000\n"
        "700.000000\n"
        "457\n"
        f"{response['observed_monotonic_ms']}\n"
    ).encode("utf-8")
    _verify_signature(certificate, response, payload, tmp_path)
    assert argument_log.read_text().splitlines() == [
        "--query-gpu=uuid,temperature.gpu,temperature.memory,power.draw,power.limit,memory.used",
        "--format=csv,noheader,nounits",
        f"--id={UUID}",
    ]


@pytest.mark.parametrize(
    ("method", "path", "body", "content_type"),
    [
        ("GET", "/gx1/v1/telemetry/", b"", "application/json"),
        ("TRACE", "/gx1/v1/telemetry/", b"", "application/json"),
        ("POST", "/gx1/v1/telemetry", b"{}", "application/json"),
        ("POST", "/gx1/v1/telemetry/", b"{}", "application/json; charset=utf-8"),
        (
            "POST",
            "/gx1/v1/telemetry/",
            b'{"schema_version":"gx1_host_gpu_telemetry_request_v1","request_nonce":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","extra":1}',
            "application/json",
        ),
        (
            "POST",
            "/gx1/v1/telemetry/",
            b'{"schema_version":"gx1_host_gpu_telemetry_request_v1","request_nonce":"Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}',
            "application/json",
        ),
        (
            "POST",
            "/gx1/v1/telemetry/",
            b'{"schema_version":"gx1_host_gpu_telemetry_request_v1","request_nonce":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","request_nonce":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}',
            "application/json",
        ),
        ("POST", "/gx1/v1/telemetry/", b"{" + b" " * 512, "application/json"),
    ],
)
def test_request_envelope_rejects_fail_closed_with_exact_error(
    service: ModuleType,
    signing_material: tuple[Path, Path],
    tmp_path: Path,
    method: str,
    path: str,
    body: bytes,
    content_type: str,
) -> None:
    private_key, _certificate = signing_material
    nvidia_smi, argument_log = _fake_nvidia_smi(
        tmp_path,
        f"{UUID}, 57, 64, 120.5, 700.0, 457\n",
    )
    with _running_service(service, private_key, nvidia_smi) as address:
        status, headers, response_body = _request(
            address,
            method=method,
            path=path,
            body=body,
            content_type=content_type,
        )

    assert status == 503
    assert response_body == ERROR_BODY
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    assert headers["Content-Length"] == str(len(ERROR_BODY))
    assert not argument_log.exists()


@pytest.mark.parametrize(
    ("sensor_output", "sleep_seconds", "sensor_timeout_seconds"),
    [
        (
            "GPU-00000000-0000-0000-0000-000000000000, 57, 64, 120.5, 700.0, 457\n",
            0.0,
            1.0,
        ),
        (f"{UUID}, 57, N/A, 120.5, 700.0, 457\n", 0.0, 1.0),
        (
            f"{UUID}, 57, 64, 120.5, 700.0, 457\n{UUID}, 58, 65, 121.5, 700.0, 458\n",
            0.0,
            1.0,
        ),
        (f"{UUID}, 57, 64, 120.5, 700.0, 457\n", 0.25, 0.1),
    ],
    ids=("wrong-uuid", "missing-memory-temperature", "multiple-rows", "timeout"),
)
def test_sensor_failures_return_only_exact_unavailable_response(
    service: ModuleType,
    signing_material: tuple[Path, Path],
    tmp_path: Path,
    sensor_output: str,
    sleep_seconds: float,
    sensor_timeout_seconds: float,
) -> None:
    private_key, _certificate = signing_material
    nvidia_smi, _argument_log = _fake_nvidia_smi(
        tmp_path,
        sensor_output,
        sleep_seconds=sleep_seconds,
    )
    with _running_service(
        service,
        private_key,
        nvidia_smi,
        sensor_timeout_seconds=sensor_timeout_seconds,
    ) as address:
        status, headers, response_body = _request(address)

    assert status == 503
    assert response_body == ERROR_BODY
    assert headers["Content-Length"] == str(len(ERROR_BODY))


def test_installer_declares_root_owned_hardened_loopback_systemd_contract() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(INSTALLER_PATH)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr
    source = INSTALLER_PATH.read_text()
    for required in (
        "must run as root",
        "rsa:3072",
        "/etc/gx1-host-telemetry",
        '-m 0600 "$key_tmp" "$PRIVATE_KEY"',
        '-m 0600 "$environment_tmp" "$ENVIRONMENT_FILE"',
        "--listen-host 127.0.0.1",
        "NoNewPrivileges=yes",
        "ProtectSystem=strict",
        "IPAddressDeny=any",
        "IPAddressAllow=localhost",
        "CapabilityBoundingSet=",
        "systemctl enable --now",
        "systemctl is-active --quiet",
    ):
        assert required in source
    assert SERVICE_PATH.stat().st_mode & 0o111
