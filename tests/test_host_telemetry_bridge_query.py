from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
QUERY = REPO / "scripts" / "gx1_host_telemetry_bridge_query.sh"
UUID = "GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29"
RESPONSE_SCHEMA = "gx1_host_gpu_telemetry_v1"


@pytest.fixture(scope="module")
def signing_material(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, str]:
    directory = tmp_path_factory.mktemp("host-telemetry-bridge")
    private_key = directory / "private.pem"
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
            "/CN=GX1-host-telemetry-test",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return private_key, certificate, hashlib.sha256(certificate.read_bytes()).hexdigest()


def _sign_response(
    private_key: Path,
    *,
    nonce: str,
    gpu_uuid: str = UUID,
    core_temp_c: float = 57.0,
    memory_temp_c: float = 64.0,
    power_draw_w: float = 120.5,
    power_limit_w: float = 250.0,
    memory_used_mib: int = 457,
    observed_monotonic_ms: int = 991_234,
) -> dict[str, object]:
    payload = "\n".join(
        (
            RESPONSE_SCHEMA,
            nonce,
            gpu_uuid,
            f"{core_temp_c:.6f}",
            f"{memory_temp_c:.6f}",
            f"{power_draw_w:.6f}",
            f"{power_limit_w:.6f}",
            str(memory_used_mib),
            str(observed_monotonic_ms),
            "",
        )
    ).encode("utf-8")
    result = subprocess.run(
        ["openssl", "dgst", "-sha256", "-sign", str(private_key)],
        input=payload,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8")
    return {
        "schema_version": RESPONSE_SCHEMA,
        "request_nonce": nonce,
        "gpu_uuid": gpu_uuid,
        "core_temp_c": core_temp_c,
        "memory_temp_c": memory_temp_c,
        "power_draw_w": power_draw_w,
        "power_limit_w": power_limit_w,
        "memory_used_mib": memory_used_mib,
        "observed_monotonic_ms": observed_monotonic_ms,
        "signature": base64.b64encode(result.stdout).decode("ascii"),
    }


ResponseBuilder = Callable[[dict[str, object]], tuple[int, dict[str, object], float]]


@contextmanager
def _host_bridge(builder: ResponseBuilder) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - HTTP API name
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            status, body, delay_seconds = builder(request)
            if delay_seconds:
                time.sleep(delay_seconds)
            encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/gx1/v1/telemetry/"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _query(url: str, certificate: Path, certificate_sha256: str, *, timeout: int = 2) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(QUERY),
            url,
            str(certificate),
            certificate_sha256,
            UUID,
            str(timeout),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )


def test_host_telemetry_query_verifies_nonce_bound_signed_response(
    signing_material: tuple[Path, Path, str],
) -> None:
    private_key, certificate, certificate_sha256 = signing_material

    def builder(request: dict[str, object]) -> tuple[int, dict[str, object], float]:
        assert request == {
            "schema_version": "gx1_host_gpu_telemetry_request_v1",
            "request_nonce": request["request_nonce"],
        }
        return 200, _sign_response(private_key, nonce=str(request["request_nonce"])), 0.0

    with _host_bridge(builder) as url:
        result = _query(url, certificate, certificate_sha256)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "57,64,120.5,250,457\n"


@pytest.mark.parametrize("failure", ("wrong_nonce", "wrong_uuid", "extra_field", "bad_signature"))
def test_host_telemetry_query_rejects_tampered_or_replayed_response(
    signing_material: tuple[Path, Path, str], failure: str
) -> None:
    private_key, certificate, certificate_sha256 = signing_material

    def builder(request: dict[str, object]) -> tuple[int, dict[str, object], float]:
        response = _sign_response(private_key, nonce=str(request["request_nonce"]))
        if failure == "wrong_nonce":
            response["request_nonce"] = "0" * 64
        elif failure == "wrong_uuid":
            response["gpu_uuid"] = "GPU-00000000-0000-0000-0000-000000000000"
        elif failure == "extra_field":
            response["untrusted_extra"] = True
        elif failure == "bad_signature":
            response["signature"] = base64.b64encode(b"not a signature").decode("ascii")
        return 200, response, 0.0

    with _host_bridge(builder) as url:
        result = _query(url, certificate, certificate_sha256)

    assert result.returncode == 75
    assert "FATAL: host telemetry bridge:" in result.stderr
    assert result.stdout == ""


def test_host_telemetry_query_rejects_unbound_certificate_hash(
    signing_material: tuple[Path, Path, str],
) -> None:
    _private_key, certificate, _certificate_sha256 = signing_material
    result = _query(
        "http://127.0.0.1:38127/gx1/v1/telemetry/",
        certificate,
        "0" * 64,
    )

    assert result.returncode == 75
    assert "certificate SHA-256 does not match" in result.stderr


def test_host_telemetry_query_rejects_timeout(
    signing_material: tuple[Path, Path, str],
) -> None:
    private_key, certificate, certificate_sha256 = signing_material

    def builder(request: dict[str, object]) -> tuple[int, dict[str, object], float]:
        return 200, _sign_response(private_key, nonce=str(request["request_nonce"])), 2.0

    with _host_bridge(builder) as url:
        result = _query(url, certificate, certificate_sha256, timeout=1)

    assert result.returncode == 75
    assert "bridge request failed or timed out" in result.stderr
