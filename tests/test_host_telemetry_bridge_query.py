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
RESPONSE_SCHEMA = "gx1_host_resource_telemetry_v2"
HOST_FIELDS = (
    "cpu_max_temp_c", "cpu_package_power_w",
    "host_physical_total_bytes", "host_physical_available_bytes",
    "host_commit_used_bytes", "host_commit_limit_bytes",
    "cbs_reboot_pending", "wu_reboot_required", "maintenance_active",
)


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
    cpu_max_temp_c: float = 65.5,
    cpu_package_power_w: float = 35.25,
    host_physical_total_bytes: int = 64 * 1024**3,
    host_physical_available_bytes: int = 40 * 1024**3,
    host_commit_used_bytes: int = 20 * 1024**3,
    host_commit_limit_bytes: int = 68 * 1024**3,
    cbs_reboot_pending: bool = False,
    wu_reboot_required: bool = False,
    maintenance_active: bool = False,
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
            f"{cpu_max_temp_c:.6f}",
            f"{cpu_package_power_w:.6f}",
            str(host_physical_total_bytes),
            str(host_physical_available_bytes),
            str(host_commit_used_bytes),
            str(host_commit_limit_bytes),
            str(cbs_reboot_pending).lower(),
            str(wu_reboot_required).lower(),
            str(maintenance_active).lower(),
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
        "cpu_max_temp_c": cpu_max_temp_c,
        "cpu_package_power_w": cpu_package_power_w,
        "host_physical_total_bytes": host_physical_total_bytes,
        "host_physical_available_bytes": host_physical_available_bytes,
        "host_commit_used_bytes": host_commit_used_bytes,
        "host_commit_limit_bytes": host_commit_limit_bytes,
        "cbs_reboot_pending": cbs_reboot_pending,
        "wu_reboot_required": wu_reboot_required,
        "maintenance_active": maintenance_active,
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
            try:
                self.wfile.write(encoded)
            except BrokenPipeError:
                # The timeout test intentionally closes its curl socket before
                # this delayed fake service writes the response.
                pass

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


def _query(url: str, certificate: Path, certificate_sha256: str, *, timeout: int = 2, host_resources: bool = False, extra_args: tuple[str, ...] = ()) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(QUERY),
            url,
            str(certificate),
            certificate_sha256,
            UUID,
            str(timeout),
            *(["--host-resources"] if host_resources else []),
            *extra_args,
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


def test_extended_resource_view_preserves_all_fourteen_fields(signing_material):
    private_key, certificate, digest = signing_material
    def builder(request):
        return 200, _sign_response(private_key, nonce=request['request_nonce'],
            cbs_reboot_pending=True, wu_reboot_required=False, maintenance_active=True), 0.0
    with _host_bridge(builder) as url:
        result = _query(url, certificate, digest, host_resources=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == '57,64,120.5,250,457,65.5,35.25,68719476736,42949672960,21474836480,73014444032,true,false,true\n'
    assert len(result.stdout.strip().split(',')) == 14


def test_safe_integer_boundary_preserves_exact_decimal_bytes(signing_material):
    private_key, certificate, digest = signing_material
    exact = 2**53 - 1
    def builder(request):
        return 200, _sign_response(private_key, nonce=request['request_nonce'],
            host_physical_total_bytes=exact, host_physical_available_bytes=exact,
            host_commit_used_bytes=exact, host_commit_limit_bytes=exact), 0.0
    with _host_bridge(builder) as url:
        result = _query(url, certificate, digest, host_resources=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().split(',')[7:11] == [str(exact)] * 4


@pytest.mark.parametrize('field', HOST_FIELDS)
def test_each_host_field_is_mandatory_even_for_five_column_readiness(signing_material, field):
    private_key, certificate, digest = signing_material
    def builder(request):
        response = _sign_response(private_key, nonce=request['request_nonce'])
        del response[field]
        return 200, response, 0.0
    with _host_bridge(builder) as url:
        result = _query(url, certificate, digest)
    assert result.returncode == 75 and result.stdout == ''
    assert 'schema, nonce, UUID, or finite-value validation' in result.stderr


@pytest.mark.parametrize('field', HOST_FIELDS)
def test_each_host_field_is_covered_by_signature(signing_material, field):
    private_key, certificate, digest = signing_material
    def builder(request):
        response = _sign_response(private_key, nonce=request['request_nonce'])
        response[field] = not response[field] if isinstance(response[field], bool) else response[field] + 1
        return 200, response, 0.0
    with _host_bridge(builder) as url:
        result = _query(url, certificate, digest, host_resources=True)
    assert result.returncode == 75 and result.stdout == ''
    assert 'response signature verification failed' in result.stderr


@pytest.mark.parametrize(('field', 'bad'), [
    ('cpu_max_temp_c', 0), ('cpu_max_temp_c', -1), ('cpu_max_temp_c', None),
    ('cpu_max_temp_c', float('nan')), ('cpu_package_power_w', float('inf')),
    ('cpu_package_power_w', 0), ('cpu_package_power_w', -1), ('cpu_package_power_w', 'unknown'),
    ('host_physical_total_bytes', 0), ('host_physical_total_bytes', 2**53),
    ('host_physical_available_bytes', -1), ('host_physical_available_bytes', 1.5),
    ('host_physical_available_bytes', 65 * 1024**3),
    ('host_commit_used_bytes', -1), ('host_commit_used_bytes', True),
    ('host_commit_used_bytes', 69 * 1024**3),
    ('host_commit_limit_bytes', 0), ('host_commit_limit_bytes', 2**53),
    ('cbs_reboot_pending', 0), ('cbs_reboot_pending', 'false'),
    ('wu_reboot_required', None), ('wu_reboot_required', 1),
    ('maintenance_active', 'unknown'), ('maintenance_active', 0),
])
def test_host_resource_schema_rejects_unknown_types_bounds_and_inconsistent_memory(signing_material, field, bad):
    private_key, certificate, digest = signing_material
    def builder(request):
        response = _sign_response(private_key, nonce=request['request_nonce'])
        response[field] = bad
        return 200, response, 0.0
    with _host_bridge(builder) as url:
        result = _query(url, certificate, digest, host_resources=True)
    assert result.returncode == 75 and result.stdout == ''
    assert 'schema, nonce, UUID, or finite-value validation' in result.stderr


def test_gpu_only_v1_response_is_never_a_fallback(signing_material):
    private_key, certificate, digest = signing_material
    def builder(request):
        response = _sign_response(private_key, nonce=request['request_nonce'])
        response['schema_version'] = 'gx1_host_gpu_telemetry_v1'
        for field in HOST_FIELDS:
            del response[field]
        return 200, response, 0.0
    with _host_bridge(builder) as url:
        result = _query(url, certificate, digest)
    assert result.returncode == 75 and result.stdout == ''


@pytest.mark.parametrize('extra', [('unexpected',), ('--host-resources', 'unexpected')])
def test_only_exact_optional_host_resource_view_is_accepted(signing_material, extra):
    _, certificate, digest = signing_material
    result = _query('http://127.0.0.1:38127/gx1/v1/telemetry/', certificate, digest, extra_args=extra)
    assert result.returncode == 75 and result.stdout == ''
    assert 'optional --host-resources' in result.stderr


@pytest.mark.parametrize('host_resources', [False, True])
def test_unavailable_host_measurement_never_falls_back_to_gpu_only(signing_material, host_resources):
    _, certificate, digest = signing_material
    def builder(request):
        return 503, {'error': 'telemetry_unavailable'}, 0.0
    with _host_bridge(builder) as url:
        result = _query(url, certificate, digest, host_resources=host_resources)
    assert result.returncode == 75 and result.stdout == ''
    assert 'bridge request failed or timed out' in result.stderr
